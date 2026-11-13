//! Forward kinematics and finite-difference velocities over a motion record.
//!
//! Contact estimation and segmentation (SPECS.md Section 8.2 items 5 and 6)
//! need joint positions in the record's reference frame, which a
//! [`MotionRecord`] stores only implicitly: a root pose per frame, a rest
//! offset per channel, and one local value per channel. This module makes
//! them explicit.
//!
//! The convention is BVH's, as `motion-io` imports it: the root channel's
//! origin is the frame's root pose applied to the channel's rest offset;
//! every other channel's origin is its parent's global pose applied to its
//! own rest offset, and its orientation is the parent's orientation after
//! its own local rotation. A [`JointValue::Position`] channel (a marker) is
//! taken as already expressed in the reference frame. A
//! [`JointValue::Scalar`] channel carries no rotation and is treated as
//! rigidly attached to its parent.
//!
//! A sample marked [`Validity::Missing`] is never read: it makes its own
//! position, and every descendant's, `None` for that frame, rather than
//! silently placing it at a placeholder value.

use robot_types::frames::{Quat, Vec3};
use robot_types::motion::{JointValue, MotionKind, MotionRecord, Validity};

/// Global positions of every channel at one frame, `None` where the chain
/// from the root to that channel passes through a missing sample.
#[must_use]
pub fn frame_positions<K: MotionKind>(
    record: &MotionRecord<K>,
    frame_index: usize,
) -> Vec<Option<Vec3>> {
    frame_poses(record, frame_index)
        .into_iter()
        .map(|p| p.map(|(x, _)| x))
        .collect()
}

/// Global position and orientation of every channel at one frame, `None`
/// where the chain from the root passes through a missing sample. A marker
/// channel takes its parent's orientation.
#[must_use]
pub fn frame_poses<K: MotionKind>(
    record: &MotionRecord<K>,
    frame_index: usize,
) -> Vec<Option<(Vec3, Quat)>> {
    let frame = &record.frames[frame_index];
    let n = record.channels.len();
    let mut poses: Vec<Option<(Vec3, Quat)>> = Vec::with_capacity(n);
    for (i, channel) in record.channels.iter().enumerate() {
        if frame.validity[i] == Validity::Missing {
            poses.push(None);
            continue;
        }
        let parent = match channel.parent {
            Some(p) => match poses[p] {
                Some(pose) => pose,
                None => {
                    poses.push(None);
                    continue;
                }
            },
            None => (frame.root.translation, Quat::IDENTITY),
        };
        let pose = match &frame.values[i] {
            JointValue::Position { position } => (*position, parent.1),
            JointValue::Rotation { rotation, .. } => {
                let origin = parent.1.rotate(channel.offset) + parent.0;
                if channel.parent.is_none() {
                    // The root's local rotation *is* the frame's root
                    // rotation (motion-io keeps both); use the root pose so
                    // a caller who edits one cannot desynchronize them here.
                    (origin, frame.root.rotation)
                } else {
                    (origin, parent.1 * *rotation)
                }
            }
            JointValue::Scalar { .. } => (parent.1.rotate(channel.offset) + parent.0, parent.1),
        };
        poses.push(Some(pose));
    }
    poses
}

/// The global position of one channel across every frame.
#[must_use]
pub fn channel_trajectory<K: MotionKind>(
    record: &MotionRecord<K>,
    channel: usize,
) -> Vec<Option<Vec3>> {
    (0..record.frames.len())
        .map(|f| frame_positions(record, f)[channel])
        .collect()
}

/// Global positions of every channel across every frame, indexed
/// `[frame][channel]`.
#[must_use]
pub fn all_positions<K: MotionKind>(record: &MotionRecord<K>) -> Vec<Vec<Option<Vec3>>> {
    (0..record.frames.len())
        .map(|f| frame_positions(record, f))
        .collect()
}

/// Frame times in seconds relative to the first frame.
#[must_use]
pub fn frame_times<K: MotionKind>(record: &MotionRecord<K>) -> Vec<f64> {
    let t0 = record.frames.first().map_or(0, |f| f.timestamp.nanos());
    record
        .frames
        .iter()
        .map(|f| (f.timestamp.nanos() - t0) as f64 * 1e-9)
        .collect()
}

/// Velocity by central differences (one-sided at the ends and next to a
/// missing sample), in metres per second. `None` where no finite difference
/// can be formed from present samples at distinct times.
#[must_use]
pub fn velocities(positions: &[Option<Vec3>], times: &[f64]) -> Vec<Option<Vec3>> {
    let n = positions.len();
    (0..n)
        .map(|i| {
            positions[i]?;
            let prev = i.checked_sub(1).filter(|&j| positions[j].is_some());
            let next = Some(i + 1).filter(|&j| j < n && positions[j].is_some());
            let (a, b) = match (prev, next) {
                (Some(a), Some(b)) => (a, b),
                (Some(a), None) if positions[i].is_some() => (a, i),
                (None, Some(b)) if positions[i].is_some() => (i, b),
                _ => return None,
            };
            let dt = times[b] - times[a];
            if dt <= 0.0 {
                return None;
            }
            Some((positions[b]? - positions[a]?) * (1.0 / dt))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{chain_record, uniform_times};
    use robot_types::units::Radians;

    #[test]
    fn a_rotated_parent_carries_its_child_offset() {
        // Root at origin, child offset 1 m along x; rotating the root 90
        // degrees about z puts the child at +y.
        let rot = Quat::from_rz(Radians(core::f64::consts::FRAC_PI_2)).unwrap();
        let record = chain_record(&[(Vec3::ZERO, rot)], 100.0);
        let p = frame_positions(&record, 0);
        let child = p[1].unwrap();
        assert!(
            (child - Vec3::new(0.0, 1.0, 0.0)).norm() < 1e-12,
            "{child:?}"
        );
    }

    #[test]
    fn a_missing_parent_hides_its_descendants() {
        let mut record = chain_record(&[(Vec3::ZERO, Quat::IDENTITY)], 100.0);
        record.frames[0].validity[0] = Validity::Missing;
        let p = frame_positions(&record, 0);
        assert!(p.iter().all(Option::is_none));
    }

    #[test]
    fn velocity_of_uniform_motion_is_exact() {
        let times = uniform_times(5, 0.1);
        let positions: Vec<_> = times
            .iter()
            .map(|t| Some(Vec3::new(2.0 * t, 0.0, -t)))
            .collect();
        for v in velocities(&positions, &times) {
            let v = v.unwrap();
            assert!((v - Vec3::new(2.0, 0.0, -1.0)).norm() < 1e-12);
        }
    }

    #[test]
    fn velocity_next_to_a_gap_is_one_sided_and_isolated_samples_have_none() {
        let times = uniform_times(5, 0.1);
        let positions = vec![
            Some(Vec3::ZERO),
            Some(Vec3::new(0.1, 0.0, 0.0)),
            None,
            Some(Vec3::ZERO),
            None,
        ];
        let v = velocities(&positions, &times);
        assert!((v[1].unwrap().x - 1.0).abs() < 1e-12);
        assert!(v[2].is_none());
        assert!(v[3].is_none());
    }
}

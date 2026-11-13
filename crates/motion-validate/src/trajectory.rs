//! A robot reference as arrays, with finite-difference derivatives.
//!
//! Every check in this crate reads the same quantities from a
//! `MotionRecord<RobotReference>` (one scalar channel per actuated joint,
//! the base pose as the root), so they are extracted once here.
//!
//! Derivatives are central differences on the actual timestamps, one-sided
//! at the ends. The base's velocity is the body-frame twist `[omega_b; v_b]`
//! (angular first), which is the free joint's velocity coordinate in
//! `robot_sim::articulated`, so its finite difference is that joint's
//! acceleration coordinate and the two can be handed to inverse dynamics
//! together.

use core::fmt;
use robot_model::pose::RigidPose;
use robot_types::frames::{Quat, Vec3};
use robot_types::motion::{JointValue, MotionRecord, RobotReference, Validity};

/// Why a reference could not be read.
#[derive(Clone, PartialEq, Debug)]
pub enum TrajectoryError {
    /// Fewer than three frames: no second difference exists.
    TooShort,
    /// The channel count differed from the profile's actuated joints.
    WidthMismatch {
        /// Expected.
        expected: usize,
        /// Found.
        found: usize,
    },
    /// A channel was not a scalar, or a sample was missing.
    NotScalar {
        /// Frame.
        frame: usize,
        /// Channel.
        channel: usize,
    },
    /// Two frames shared a timestamp.
    RepeatedTimestamp {
        /// The later frame.
        frame: usize,
    },
}

impl fmt::Display for TrajectoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooShort => f.write_str("a reference needs at least three frames"),
            Self::WidthMismatch { expected, found } => {
                write!(
                    f,
                    "reference has {found} channels, profile has {expected} joints"
                )
            }
            Self::NotScalar { frame, channel } => {
                write!(f, "frame {frame} channel {channel} is not a present scalar")
            }
            Self::RepeatedTimestamp { frame } => write!(f, "frame {frame} repeats a timestamp"),
        }
    }
}

impl core::error::Error for TrajectoryError {}

/// A reference as arrays, indexed `[frame][joint]`.
#[derive(Clone, PartialEq, Debug)]
pub struct Trajectory {
    /// Seconds from the first frame.
    pub times: Vec<f64>,
    /// Joint positions.
    pub q: Vec<Vec<f64>>,
    /// Joint velocities.
    pub qd: Vec<Vec<f64>>,
    /// Joint accelerations.
    pub qdd: Vec<Vec<f64>>,
    /// Joint jerks.
    pub qddd: Vec<Vec<f64>>,
    /// Base poses (world from root link).
    pub base: Vec<RigidPose>,
    /// Base body-frame twist `[omega_b; v_b]`.
    pub base_twist: Vec<[f64; 6]>,
    /// Its time derivative.
    pub base_twist_rate: Vec<[f64; 6]>,
}

/// Central differences of a frame-major series (one-sided at the ends).
#[must_use]
pub fn derivative(series: &[Vec<f64>], times: &[f64]) -> Vec<Vec<f64>> {
    let n = series.len();
    (0..n)
        .map(|i| {
            let (a, b) = match (i.checked_sub(1), (i + 1 < n).then_some(i + 1)) {
                (Some(a), Some(b)) => (a, b),
                (None, Some(b)) => (i, b),
                (Some(a), None) => (a, i),
                (None, None) => return vec![0.0; series[i].len()],
            };
            let dt = times[b] - times[a];
            series[a]
                .iter()
                .zip(&series[b])
                .map(|(x, y)| (y - x) / dt)
                .collect()
        })
        .collect()
}

impl Trajectory {
    /// Extracts a reference with `dof` scalar channels.
    pub fn from_reference(
        reference: &MotionRecord<RobotReference>,
        dof: usize,
    ) -> Result<Self, TrajectoryError> {
        let n = reference.frames.len();
        if n < 3 {
            return Err(TrajectoryError::TooShort);
        }
        if reference.channels.len() != dof {
            return Err(TrajectoryError::WidthMismatch {
                expected: dof,
                found: reference.channels.len(),
            });
        }
        let t0 = reference.frames[0].timestamp.nanos();
        let times: Vec<f64> = reference
            .frames
            .iter()
            .map(|f| (f.timestamp.nanos() - t0) as f64 * 1e-9)
            .collect();
        if let Some(i) = times.windows(2).position(|w| w[1] <= w[0]) {
            return Err(TrajectoryError::RepeatedTimestamp { frame: i + 1 });
        }
        let mut q = Vec::with_capacity(n);
        for (fi, frame) in reference.frames.iter().enumerate() {
            let mut row = Vec::with_capacity(dof);
            for (ci, (v, valid)) in frame.values.iter().zip(&frame.validity).enumerate() {
                match (v, valid) {
                    (
                        JointValue::Scalar { value, .. },
                        Validity::Valid | Validity::Interpolated,
                    ) => row.push(*value),
                    _ => {
                        return Err(TrajectoryError::NotScalar {
                            frame: fi,
                            channel: ci,
                        })
                    }
                }
            }
            q.push(row);
        }
        let qd = derivative(&q, &times);
        let qdd = derivative(&qd, &times);
        let qddd = derivative(&qdd, &times);
        let base: Vec<RigidPose> = reference
            .frames
            .iter()
            .map(|f| RigidPose {
                translation: f.root.translation,
                rotation: f.root.rotation,
            })
            .collect();
        let base_twist: Vec<[f64; 6]> = (0..n)
            .map(|i| {
                let (a, b) = (i.saturating_sub(1), (i + 1).min(n - 1));
                let dt = times[b] - times[a];
                let r = base[i].rotation;
                let w = (base[a].rotation.conjugate() * base[b].rotation)
                    .normalized()
                    .to_rotation_vector()
                    * (1.0 / dt);
                let v = r
                    .conjugate()
                    .rotate((base[b].translation - base[a].translation) * (1.0 / dt));
                [w.x, w.y, w.z, v.x, v.y, v.z]
            })
            .collect();
        let rows: Vec<Vec<f64>> = base_twist.iter().map(|t| t.to_vec()).collect();
        let base_twist_rate = derivative(&rows, &times)
            .into_iter()
            .map(|r| [r[0], r[1], r[2], r[3], r[4], r[5]])
            .collect();
        Ok(Self {
            times,
            q,
            qd,
            qdd,
            qddd,
            base,
            base_twist,
            base_twist_rate,
        })
    }

    /// Frame count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Whether there are no frames (never true for a constructed value).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }
}

/// The `w, x, y, z` quaternion and translation of a pose as the seven
/// position coordinates of a free joint.
#[must_use]
pub fn free_joint_q(pose: &RigidPose) -> [f64; 7] {
    let (t, r): (Vec3, Quat) = (pose.translation, pose.rotation);
    [t.x, t.y, t.z, r.w, r.x, r.y, r.z]
}

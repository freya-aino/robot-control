//! Axis and unit correction, finite-value validation and gap marking, per
//! SPECS.md Section 8.2 item 3: "Correct axes/units, validate finite values
//! and timestamps, and mark gaps."
//!
//! Three separate operations, each recorded as its own provenance step:
//!
//! - [`convert_axes`]: a change of basis and a length scale applied
//!   consistently to the root pose, every rest offset, every local rotation
//!   (by conjugation, so global poses rotate rigidly) and every position or
//!   metre-valued scalar sample.
//! - [`mark_non_finite`]: a non-finite sample becomes
//!   [`Validity::Missing`] instead of rejecting the whole record; a frame
//!   whose *root* is non-finite has no channel to mark, so it is dropped and
//!   reported, leaving a timing gap that [`find_timing_gaps`] then finds.
//! - [`fill_gaps`]: bounded interpolation across short runs of missing
//!   samples, marked [`Validity::Interpolated`]; a run longer than the bound
//!   stays missing, so a long occlusion is never papered over.

use core::fmt;
use robot_types::frames::{Quat, Vec3};
use robot_types::motion::{JointValue, MotionKind, MotionRecord, ProvenanceStep, Validity};
use robot_types::time::Duration;
use robot_types::units::Radians;

use crate::REVISION;

/// A change of coordinate convention: a rotation of the basis and a length
/// scale to metres.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct AxisConversion {
    /// Rotation taking source axes to target axes.
    pub basis: Quat,
    /// Multiplier from source length units to metres.
    pub scale: f64,
    /// Stable name recorded in provenance.
    pub name: &'static str,
}

impl AxisConversion {
    /// Y-up (BVH's usual convention) to Z-up: a +90 degree rotation about
    /// x, which takes +y to +z and +z to -y. Lengths are left as they are.
    #[must_use]
    pub fn y_up_to_z_up() -> Self {
        Self {
            basis: Quat::from_rx(Radians(core::f64::consts::FRAC_PI_2))
                .expect("finite constant angle"),
            scale: 1.0,
            name: "y_up_to_z_up",
        }
    }

    /// The same basis, with lengths multiplied by `scale` (`0.01` for a
    /// centimetre source).
    #[must_use]
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }

    fn point(&self, p: Vec3) -> Vec3 {
        self.basis.rotate(p) * self.scale
    }

    fn rotation(&self, q: Quat) -> Quat {
        (self.basis * q * self.basis.conjugate()).normalized()
    }
}

/// Why conditioning failed.
#[derive(Clone, PartialEq, Debug)]
pub enum ConditionError {
    /// The length scale was zero, negative or non-finite.
    InvalidScale(f64),
    /// A bound or threshold was zero, negative or non-finite.
    InvalidParameter(&'static str),
    /// Every frame had a non-finite root and was dropped.
    NoFramesLeft,
}

impl fmt::Display for ConditionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidScale(s) => write!(f, "invalid length scale {s}"),
            Self::InvalidParameter(name) => write!(f, "invalid parameter {name}"),
            Self::NoFramesLeft => f.write_str("every frame had a non-finite root"),
        }
    }
}

impl core::error::Error for ConditionError {}

/// Applies `conversion` to every pose, offset and sample of `record`.
pub fn convert_axes<K: MotionKind>(
    record: &mut MotionRecord<K>,
    conversion: AxisConversion,
) -> Result<(), ConditionError> {
    if !conversion.scale.is_finite() || conversion.scale <= 0.0 {
        return Err(ConditionError::InvalidScale(conversion.scale));
    }
    for channel in &mut record.channels {
        channel.offset = conversion.point(channel.offset);
    }
    for frame in &mut record.frames {
        frame.root.translation = conversion.point(frame.root.translation);
        frame.root.rotation = conversion.rotation(frame.root.rotation);
        for value in &mut frame.values {
            match value {
                JointValue::Rotation { rotation, .. } => *rotation = conversion.rotation(*rotation),
                JointValue::Position { position } => *position = conversion.point(*position),
                JointValue::Scalar { value, unit } if *unit == "m" => *value *= conversion.scale,
                JointValue::Scalar { .. } => {}
            }
        }
    }
    record.record_step(
        ProvenanceStep::new("convert_axes", REVISION)
            .with_parameter("conversion", conversion.name)
            .with_parameter("scale", conversion.scale.to_string()),
    );
    Ok(())
}

/// What [`mark_non_finite`] changed.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct FiniteReport {
    /// Samples newly marked missing, as `(frame, channel)` in the output
    /// record's frame numbering.
    pub marked_missing: Vec<(usize, usize)>,
    /// Source frame indices dropped for a non-finite root pose.
    pub dropped_frames: Vec<usize>,
}

/// Marks non-finite or non-unit-norm samples [`Validity::Missing`] and drops
/// frames with a non-finite root.
///
/// A rotation whose quaternion norm is more than `1e-6` from one is treated
/// as invalid: every consumer assumes unit quaternions, and normalizing a
/// corrupt sample would invent a rotation the source never measured.
pub fn mark_non_finite<K: MotionKind>(
    record: &mut MotionRecord<K>,
) -> Result<FiniteReport, ConditionError> {
    let mut report = FiniteReport::default();
    let frames = core::mem::take(&mut record.frames);
    for (source_index, mut frame) in frames.into_iter().enumerate() {
        if !frame.root.is_finite() {
            report.dropped_frames.push(source_index);
            continue;
        }
        let out_index = record.frames.len();
        for (ci, (value, validity)) in frame.values.iter().zip(&mut frame.validity).enumerate() {
            if *validity == Validity::Missing {
                continue;
            }
            let ok = match value {
                JointValue::Rotation { rotation: q, .. } => {
                    let n2 = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
                    n2.is_finite() && (n2.sqrt() - 1.0).abs() <= 1e-6
                }
                JointValue::Scalar { value, .. } => value.is_finite(),
                JointValue::Position { position } => position.is_finite(),
            };
            if !ok {
                *validity = Validity::Missing;
                report.marked_missing.push((out_index, ci));
            }
        }
        record.frames.push(frame);
    }
    if record.frames.is_empty() {
        return Err(ConditionError::NoFramesLeft);
    }
    record.record_step(
        ProvenanceStep::new("mark_non_finite", REVISION)
            .with_parameter("marked", report.marked_missing.len().to_string())
            .with_parameter("dropped_frames", report.dropped_frames.len().to_string()),
    );
    Ok(report)
}

/// An interval between consecutive frames longer than the record's
/// nominal period allows.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct TimingGap {
    /// Index of the frame before the gap.
    pub after_frame: usize,
    /// The interval found.
    pub interval: Duration,
    /// The median interval it was compared with.
    pub nominal: Duration,
}

/// Finds intervals longer than `factor` times the median frame interval.
///
/// The median rather than the mean is the reference so that the gaps being
/// searched for do not inflate the threshold that should flag them.
pub fn find_timing_gaps<K: MotionKind>(
    record: &MotionRecord<K>,
    factor: f64,
) -> Result<Vec<TimingGap>, ConditionError> {
    if !factor.is_finite() || factor <= 1.0 {
        return Err(ConditionError::InvalidParameter("factor"));
    }
    let intervals: Vec<i64> = record
        .frames
        .windows(2)
        .map(|w| w[1].timestamp.nanos() - w[0].timestamp.nanos())
        .collect();
    if intervals.is_empty() {
        return Ok(Vec::new());
    }
    let mut sorted = intervals.clone();
    sorted.sort_unstable();
    // The lower median: with an even count, the upper one may itself be
    // the gap being looked for.
    let median = sorted[(sorted.len() - 1) / 2];
    Ok(intervals
        .iter()
        .enumerate()
        .filter(|(_, &dt)| dt as f64 > factor * median as f64)
        .map(|(i, &dt)| TimingGap {
            after_frame: i,
            interval: Duration::from_nanos(dt),
            nominal: Duration::from_nanos(median),
        })
        .collect())
}

/// What [`fill_gaps`] did.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct FillReport {
    /// Samples interpolated.
    pub filled: usize,
    /// Samples left missing because their run was too long or unbounded.
    pub left_missing: usize,
}

/// Interpolates each run of missing samples no longer than `max_gap`
/// (measured between the bracketing present samples) and bounded on both
/// sides, by slerp or lerp in time. Filled samples are marked
/// [`Validity::Interpolated`].
pub fn fill_gaps<K: MotionKind>(
    record: &mut MotionRecord<K>,
    max_gap: Duration,
) -> Result<FillReport, ConditionError> {
    if !max_gap.is_positive() {
        return Err(ConditionError::InvalidParameter("max_gap"));
    }
    let mut report = FillReport::default();
    let n_frames = record.frames.len();
    for ci in 0..record.channels.len() {
        let mut f = 0;
        while f < n_frames {
            if record.frames[f].validity[ci] != Validity::Missing {
                f += 1;
                continue;
            }
            let start = f;
            while f < n_frames && record.frames[f].validity[ci] == Validity::Missing {
                f += 1;
            }
            let end = f; // one past the run
            let run = end - start;
            if start == 0 || end == n_frames {
                report.left_missing += run;
                continue;
            }
            let (a, b) = (start - 1, end);
            let ta = record.frames[a].timestamp.nanos();
            let tb = record.frames[b].timestamp.nanos();
            if tb - ta > max_gap.nanos() || tb <= ta {
                report.left_missing += run;
                continue;
            }
            let va = record.frames[a].values[ci].clone();
            let vb = record.frames[b].values[ci].clone();
            for k in start..end {
                let s = (record.frames[k].timestamp.nanos() - ta) as f64 / (tb - ta) as f64;
                let Some(v) = interpolate(&va, &vb, s) else {
                    report.left_missing += 1;
                    continue;
                };
                record.frames[k].values[ci] = v;
                record.frames[k].validity[ci] = Validity::Interpolated;
                report.filled += 1;
            }
        }
    }
    record.record_step(
        ProvenanceStep::new("fill_gaps", REVISION)
            .with_parameter("max_gap_s", max_gap.as_secs_f64().to_string())
            .with_parameter("filled", report.filled.to_string())
            .with_parameter("left_missing", report.left_missing.to_string()),
    );
    Ok(report)
}

fn interpolate(a: &JointValue, b: &JointValue, s: f64) -> Option<JointValue> {
    Some(match (a, b) {
        (
            JointValue::Rotation {
                rotation: qa,
                source_order,
            },
            JointValue::Rotation { rotation: qb, .. },
        ) => JointValue::Rotation {
            rotation: qa.slerp(*qb, s),
            source_order: *source_order,
        },
        (JointValue::Scalar { value: x, unit }, JointValue::Scalar { value: y, .. }) => {
            JointValue::Scalar {
                value: x + (y - x) * s,
                unit,
            }
        }
        (JointValue::Position { position: p }, JointValue::Position { position: q }) => {
            JointValue::Position {
                position: p.lerp(*q, s),
            }
        }
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kinematics::frame_positions;
    use crate::test_support::{chain_record, root_path};

    #[test]
    fn y_up_to_z_up_moves_global_positions_rigidly() {
        // Root rotated 90 degrees about y (y-up yaw), child offset +x.
        let yaw = Quat::from_ry(Radians(0.7)).unwrap();
        let mut record = chain_record(&[(Vec3::new(1.0, 2.0, 3.0), yaw)], 100.0);
        let before = frame_positions(&record, 0);
        let conv = AxisConversion::y_up_to_z_up().with_scale(0.01);
        convert_axes(&mut record, conv).unwrap();
        let after = frame_positions(&record, 0);
        for (b, a) in before.iter().zip(&after) {
            let expected = conv.basis.rotate(b.unwrap()) * 0.01;
            assert!((a.unwrap() - expected).norm() < 1e-12);
        }
        // The source's +y (up) is now +z.
        assert!((conv.basis.rotate(Vec3::unit_y()) - Vec3::unit_z()).norm() < 1e-12);
        assert_eq!(record.provenance.last().unwrap().operation, "convert_axes");
    }

    #[test]
    fn a_non_positive_scale_is_rejected() {
        let mut record = root_path(2, 100.0, |_| Vec3::ZERO);
        let conv = AxisConversion::y_up_to_z_up().with_scale(0.0);
        assert_eq!(
            convert_axes(&mut record, conv),
            Err(ConditionError::InvalidScale(0.0))
        );
    }

    #[test]
    fn non_finite_samples_become_missing_and_bad_roots_drop_frames() {
        let mut record = root_path(4, 100.0, |_| Vec3::ZERO);
        record.frames[1].values[1] = JointValue::Rotation {
            rotation: Quat {
                w: f64::NAN,
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            source_order: None,
        };
        record.frames[2].root.translation.x = f64::INFINITY;
        let report = mark_non_finite(&mut record).unwrap();
        assert_eq!(report.dropped_frames, vec![2]);
        assert_eq!(report.marked_missing, vec![(1, 1)]);
        assert_eq!(record.frames.len(), 3);
        record.validate().unwrap();
        let gaps = find_timing_gaps(&record, 1.5).unwrap();
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].after_frame, 1);
    }

    #[test]
    fn short_gaps_fill_and_long_or_open_gaps_do_not() {
        let mut record = root_path(10, 100.0, |_| Vec3::ZERO);
        let q0 = Quat::IDENTITY;
        let q1 = Quat::from_rz(Radians(1.0)).unwrap();
        for (i, f) in record.frames.iter_mut().enumerate() {
            f.values[1] = JointValue::Rotation {
                rotation: if i < 3 { q0 } else { q1 },
                source_order: None,
            };
        }
        // A one-sample hole at frame 2 (bracketed 20 ms), a four-sample
        // hole at 4..8 (bracketed 50 ms) and an open hole at the end.
        record.frames[2].validity[1] = Validity::Missing;
        for f in 4..8 {
            record.frames[f].validity[1] = Validity::Missing;
        }
        record.frames[9].validity[1] = Validity::Missing;
        let report = fill_gaps(&mut record, Duration::from_millis(30)).unwrap();
        assert_eq!(report.filled, 1);
        assert_eq!(report.left_missing, 5);
        assert_eq!(record.frames[2].validity[1], Validity::Interpolated);
        let JointValue::Rotation { rotation, .. } = record.frames[2].values[1] else {
            panic!()
        };
        assert!((rotation.angle_to(q0.slerp(q1, 0.5)).value()).abs() < 1e-9);
        assert_eq!(record.frames[5].validity[1], Validity::Missing);
    }
}

//! Fixed-rate resampling of a canonical motion record.
//!
//! SPECS.md Section 8.2 item 4: "Filter and resample with documented phase
//! delay; preserve real strike-speed characteristics." This resampler is
//! linear interpolation in time (lerp for translation, slerp for rotation),
//! which has **zero phase delay** by construction — each output sample is
//! computed from the two bracketing input samples at its own exact time,
//! not from a filtered window — and that zero is the "documented" value
//! this module records, not an unstated default.
//!
//! What this module does not do: anti-aliasing. Linear interpolation
//! resampling to a *lower* rate than the source can introduce aliasing on
//! fast motion (exactly the "real strike-speed characteristics" the section
//! warns about preserving); this module does not filter before
//! downsampling, and a caller resampling boxing strikes to a much lower
//! rate should treat that as an open qualification question, not as
//! handled here.

use core::fmt;
use robot_types::frames::Transform;
use robot_types::motion::{
    JointValue, MotionError, MotionFrame, MotionKind, MotionRecord, ProvenanceStep, Validity,
};
use robot_types::time::{Duration, Timestamp};
use robot_types::units::Hertz;

/// Why resampling failed.
#[derive(Clone, PartialEq, Debug)]
pub enum ResampleError {
    /// The target rate was non-positive or non-finite.
    InvalidRate {
        /// The rate supplied.
        rate: Hertz,
    },
    /// The source record had fewer than two frames.
    InsufficientFrames {
        /// Frames found.
        found: usize,
    },
    /// The resulting record was rejected by its own validation.
    Motion(MotionError),
}

impl fmt::Display for ResampleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRate { rate } => write!(f, "invalid target rate {rate}"),
            Self::InsufficientFrames { found } => {
                write!(f, "need at least two frames to resample, found {found}")
            }
            Self::Motion(e) => write!(f, "{e}"),
        }
    }
}

impl core::error::Error for ResampleError {}

impl From<MotionError> for ResampleError {
    fn from(e: MotionError) -> Self {
        Self::Motion(e)
    }
}

/// Resamples `record` to a fixed `target_rate`, by linear interpolation
/// between bracketing source frames.
///
/// A frame whose interpolation touched any [`Validity::Missing`] source
/// sample is itself marked [`Validity::Interpolated`] for that channel —
/// never silently promoted to [`Validity::Valid`] — per SPECS.md Section
/// 8.2 item 3's requirement to mark gaps rather than paper over them.
pub fn resample<K: MotionKind>(
    record: &MotionRecord<K>,
    target_rate: Hertz,
) -> Result<MotionRecord<K>, ResampleError> {
    if !target_rate.value().is_finite() || target_rate.value() <= 0.0 {
        return Err(ResampleError::InvalidRate { rate: target_rate });
    }
    if record.frames.len() < 2 {
        return Err(ResampleError::InsufficientFrames {
            found: record.frames.len(),
        });
    }
    let period = target_rate
        .period()
        .expect("validated positive and finite above");

    let start = record.frames[0].timestamp;
    let end = record.frames[record.frames.len() - 1].timestamp;
    let (total, _) = end
        .since(start)
        .map_err(|_| ResampleError::InsufficientFrames { found: 0 })?;

    let mut new_frames = Vec::new();
    let mut t = Duration::ZERO;
    let mut search_from = 0usize;
    while t <= total {
        let target_time = start.advanced(t);
        while search_from + 2 < record.frames.len()
            && record.frames[search_from + 1].timestamp.nanos() < target_time.nanos()
        {
            search_from += 1;
        }
        let a = &record.frames[search_from];
        let b = &record.frames[(search_from + 1).min(record.frames.len() - 1)];
        new_frames.push(interpolate_frame(a, b, target_time)?);
        t += period;
    }

    let mut new_record = MotionRecord::new(
        record.kind(),
        record.source_hash.clone(),
        record.source_license.clone(),
        record.session_id.clone(),
        record.skeleton_hash.clone(),
        record.reference_frame,
        record.clock,
        record.channels.clone(),
        new_frames,
    )?;
    new_record.provenance = record.provenance.clone();
    new_record.record_step(
        ProvenanceStep::new("resample", "motion-process/0.1")
            .with_parameter("target_rate_hz", target_rate.value().to_string())
            .with_phase_delay(Duration::ZERO),
    );
    new_record.subject_id = record.subject_id.clone();
    new_record.task_labels = record.task_labels.clone();
    new_record.subject_height = record.subject_height;
    Ok(new_record)
}

fn interpolate_frame(
    a: &MotionFrame,
    b: &MotionFrame,
    at: Timestamp,
) -> Result<MotionFrame, ResampleError> {
    let (span, _) = b
        .timestamp
        .since(a.timestamp)
        .map_err(|_| ResampleError::InsufficientFrames { found: 0 })?;
    let frac = if span.nanos() == 0 {
        0.0
    } else {
        let (elapsed, _) = at
            .since(a.timestamp)
            .map_err(|_| ResampleError::InsufficientFrames { found: 0 })?;
        (elapsed.as_secs_f64() / span.as_secs_f64()).clamp(0.0, 1.0)
    };

    let root = Transform::new(
        a.root.from,
        a.root.to,
        a.root.rotation.slerp(b.root.rotation, frac),
        a.root.translation.lerp(b.root.translation, frac),
    );

    // Every output sample is synthesized from two inputs (or, in the
    // degenerate `frac == 0.0` case, numerically equal to one of them) —
    // it is never a direct carry-through of a single source sample, so it
    // is marked `Interpolated` unconditionally. This is the per-channel
    // mark SPECS.md Section 8.2 item 3 asks for when a gap is filled; a
    // resampled record is exactly that situation at every sample, not only
    // where a source gap existed.
    let mut values = Vec::with_capacity(a.values.len());
    let mut validity = Vec::with_capacity(a.validity.len());
    for i in 0..a.values.len() {
        let value = match (&a.values[i], &b.values[i]) {
            (
                JointValue::Rotation {
                    rotation: ra,
                    source_order,
                },
                JointValue::Rotation { rotation: rb, .. },
            ) => JointValue::Rotation {
                rotation: ra.slerp(*rb, frac),
                source_order: *source_order,
            },
            (JointValue::Scalar { value: sa, unit }, JointValue::Scalar { value: sb, .. }) => {
                JointValue::Scalar {
                    value: sa + (sb - sa) * frac,
                    unit,
                }
            }
            (JointValue::Position { position: pa }, JointValue::Position { position: pb }) => {
                JointValue::Position {
                    position: pa.lerp(*pb, frac),
                }
            }
            // Mismatched variants between the two bracketing frames should
            // not occur for a well-formed record; fall back to the earlier
            // sample's value rather than panic.
            (value, _) => value.clone(),
        };
        values.push(value);
        validity.push(Validity::Interpolated);
    }

    Ok(MotionFrame {
        timestamp: at,
        root,
        values,
        validity,
        contacts: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_types::frames::{FrameId, Quat};
    use robot_types::ids::Name;
    use robot_types::motion::{Channel, HumanSkeleton, SourceLicense};
    use robot_types::time::ClockId;

    fn license() -> SourceLicense {
        SourceLicense {
            identifier: "fixture".to_owned(),
            training_permitted: true,
            redistribution_permitted: false,
            commercial_permitted: false,
            retention: None,
            source_reference: "fixture".to_owned(),
        }
    }

    fn frame(nanos: i64, rot: Quat) -> MotionFrame {
        MotionFrame {
            timestamp: Timestamp::sampling(nanos, ClockId::Source(0)),
            root: Transform::identity(FrameId::WORLD),
            values: vec![JointValue::Rotation {
                rotation: rot,
                source_order: None,
            }],
            validity: vec![Validity::Valid],
            contacts: Vec::new(),
        }
    }

    fn record_at(rate_hz: f64) -> MotionRecord<HumanSkeleton> {
        let channels = vec![Channel {
            name: Name::new("joint").unwrap(),
            parent: None,
            offset: Default::default(),
        }];
        let dt = Duration::from_secs_f64(1.0 / rate_hz).nanos();
        let frames = vec![
            frame(0, Quat::IDENTITY),
            frame(dt, Quat::from_rz(robot_types::units::Radians(1.0)).unwrap()),
            frame(
                2 * dt,
                Quat::from_rz(robot_types::units::Radians(2.0)).unwrap(),
            ),
        ];
        MotionRecord::new(
            HumanSkeleton,
            "sha256:abc",
            license(),
            "session",
            "skeleton",
            FrameId::WORLD,
            ClockId::Source(0),
            channels,
            frames,
        )
        .unwrap()
    }

    #[test]
    fn upsampling_doubles_the_frame_count_and_interpolates_rotation() {
        let source = record_at(30.0); // 3 frames spanning 1/15s at 30 Hz source
        let resampled = resample(&source, Hertz(60.0)).unwrap();
        assert!(resampled.frame_count() >= 4);
        // Interpolated midpoint rotation should sit between frame 0 and 1.
        let JointValue::Rotation { rotation, .. } = &resampled.frames[1].values[0] else {
            panic!()
        };
        let angle = rotation.angle().value();
        assert!(
            angle > 0.0 && angle < 1.0,
            "expected a midpoint angle, got {angle}"
        );
    }

    #[test]
    fn resampling_records_zero_phase_delay() {
        let source = record_at(30.0);
        let resampled = resample(&source, Hertz(60.0)).unwrap();
        assert_eq!(resampled.accumulated_phase_delay(), Duration::ZERO);
    }

    #[test]
    fn an_invalid_rate_is_rejected() {
        let source = record_at(30.0);
        assert!(matches!(
            resample(&source, Hertz(0.0)),
            Err(ResampleError::InvalidRate { .. })
        ));
        assert!(matches!(
            resample(&source, Hertz(-5.0)),
            Err(ResampleError::InvalidRate { .. })
        ));
    }

    #[test]
    fn provenance_and_metadata_are_preserved() {
        let source = record_at(30.0);
        let resampled = resample(&source, Hertz(60.0)).unwrap();
        assert_eq!(resampled.source_hash, source.source_hash);
        assert_eq!(resampled.session_id, source.session_id);
        assert_eq!(resampled.provenance.len(), 1);
        assert_eq!(resampled.provenance[0].operation, "resample");
    }
}

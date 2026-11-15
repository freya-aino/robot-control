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
//! [`resample`] itself does not anti-alias: linear interpolation to a
//! *lower* rate than the source can alias fast motion (exactly the "real
//! strike-speed characteristics" the section warns about preserving).
//! [`resample_anti_aliased`] low-pass filters first when downsampling, and
//! returns the filter's measured peak-speed retention.

use core::fmt;
use robot_types::frames::Transform;
use robot_types::motion::{
    JointValue, MotionError, MotionFrame, MotionKind, MotionRecord, ProvenanceStep, Validity,
};
use robot_types::time::{Duration, Timestamp};
use robot_types::units::Hertz;

use crate::filter::{lowpass, uniform_rate, FilterError, FilterMode, FilterReport};

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
/// A sample bracketed by any [`Validity::Missing`] source sample stays
/// `Missing` with its placeholder untouched — a placeholder is not a
/// measurement, and interpolating it would fabricate data (an occluded
/// marker surfacing at the world origin). Every genuinely synthesized
/// sample is marked [`Validity::Interpolated`], never silently promoted to
/// [`Validity::Valid`], per SPECS.md Section 8.2 item 3's requirement to
/// mark gaps rather than paper over them.
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
        ProvenanceStep::new("resample", crate::REVISION)
            .with_parameter("target_rate_hz", target_rate.value().to_string())
            .with_phase_delay(Duration::ZERO),
    );
    new_record.subject_id = record.subject_id.clone();
    new_record.task_labels = record.task_labels.clone();
    new_record.subject_height = record.subject_height;
    Ok(new_record)
}

/// Resamples `record` to `target_rate`, first low-pass filtering at
/// `0.4 * target_rate` (80 % of the new Nyquist frequency) when the target is
/// lower than the source rate, so that downsampling does not alias.
///
/// The pre-filter is [`FilterMode::ZeroPhase`], so the documented phase
/// delay of the combined operation stays zero; its speed retention is
/// returned so a caller can see what the anti-aliasing cost a fast strike.
/// When the target rate is not lower than the source, no filter runs and
/// `None` is returned in place of a report.
pub fn resample_anti_aliased<K: MotionKind>(
    record: &MotionRecord<K>,
    target_rate: Hertz,
) -> Result<(MotionRecord<K>, Option<FilterReport>), AntiAliasError> {
    let source_rate = uniform_rate(record).map_err(AntiAliasError::Filter)?;
    if target_rate.value().is_finite() && target_rate.value() < source_rate.value() {
        let mut filtered = record.clone();
        let report = lowpass(
            &mut filtered,
            Hertz(0.4 * target_rate.value()),
            FilterMode::ZeroPhase,
        )
        .map_err(AntiAliasError::Filter)?;
        let out = resample(&filtered, target_rate).map_err(AntiAliasError::Resample)?;
        Ok((out, Some(report)))
    } else {
        let out = resample(record, target_rate).map_err(AntiAliasError::Resample)?;
        Ok((out, None))
    }
}

/// Why anti-aliased resampling failed.
#[derive(Clone, PartialEq, Debug)]
pub enum AntiAliasError {
    /// The pre-filter failed, typically on non-uniform input.
    Filter(FilterError),
    /// The resampler failed.
    Resample(ResampleError),
}

impl fmt::Display for AntiAliasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Filter(e) => write!(f, "anti-alias filter: {e}"),
            Self::Resample(e) => write!(f, "resample: {e}"),
        }
    }
}

impl core::error::Error for AntiAliasError {}

/// [`lowpass`] for a record that may be non-uniformly sampled, per SPECS.md
/// Section 8.2 item 4: a uniform record is filtered as it stands, and a
/// non-uniform one is first resampled to its own **median rate** through
/// [`resample`] (interpolation, zero phase delay), so the caller no longer
/// has to resample by hand before filtering. The returned record carries
/// both provenance steps when resampling ran, and one when it did not.
///
/// The median rate is the resampling target because it changes the typical
/// frame not at all and reconciles only the jittered ones; a caller wanting
/// a different rate resamples explicitly, exactly as before.
///
/// # Errors
///
/// Only what the resampler and the filter themselves refuse, by name:
/// [`ResampleError`] for a record the resampler rejects, [`FilterError`] for
/// a cutoff the (now uniform) record cannot support or a record too short to
/// filter.
pub fn lowpass_resampled<K: MotionKind>(
    record: &MotionRecord<K>,
    cutoff: Hertz,
    mode: FilterMode,
) -> Result<(MotionRecord<K>, FilterReport), AntiAliasError> {
    let mut out = match uniform_rate(record) {
        Ok(_) => record.clone(),
        Err(FilterError::NonUniformSampling { .. }) => {
            let intervals: Vec<i64> = record
                .frames
                .windows(2)
                .map(|w| w[1].timestamp.nanos() - w[0].timestamp.nanos())
                .collect();
            let mut sorted = intervals;
            sorted.sort_unstable();
            let median = sorted.get(sorted.len() / 2).copied().unwrap_or(0);
            if median <= 0 {
                // Out-of-order or coincident timestamps: no rate describes
                // this record, and the resampler's own refusal names it.
                return Err(AntiAliasError::Resample(ResampleError::InvalidRate {
                    rate: Hertz(f64::INFINITY),
                }));
            }
            resample(record, Hertz(1e9 / median as f64)).map_err(AntiAliasError::Resample)?
        }
        Err(e) => return Err(AntiAliasError::Filter(e)),
    };
    let report = lowpass(&mut out, cutoff, mode).map_err(AntiAliasError::Filter)?;
    Ok((out, report))
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
        // A `Missing` sample holds a placeholder, not a measurement; a
        // placeholder lerped into the output and relabeled `Interpolated`
        // would re-enter the pipeline as fabricated data (an occluded C3D
        // marker would surface at the world origin). The gap stays a gap.
        if a.validity.get(i) == Some(&Validity::Missing)
            || b.validity.get(i) == Some(&Validity::Missing)
        {
            values.push(a.values[i].clone());
            validity.push(Validity::Missing);
            continue;
        }
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
    fn downsampling_with_anti_aliasing_suppresses_a_component_above_the_new_nyquist() {
        // 40 Hz oscillation sampled at 200 Hz, resampled to 50 Hz: without
        // a pre-filter it aliases to 10 Hz at full amplitude.
        let rate = 200.0;
        let record = crate::test_support::root_path(400, rate, |t| {
            robot_types::frames::Vec3::new(
                (2.0 * core::f64::consts::PI * 40.0 * t + 0.3).sin(),
                0.0,
                0.0,
            )
        });
        let amplitude = |r: &MotionRecord<HumanSkeleton>| {
            r.frames[20..r.frames.len() - 20]
                .iter()
                .map(|f| f.root.translation.x.abs())
                .fold(0.0, f64::max)
        };
        let naive = resample(&record, Hertz(50.0)).unwrap();
        let (filtered, report) = resample_anti_aliased(&record, Hertz(50.0)).unwrap();
        assert!(amplitude(&naive) > 0.5);
        // Two passes of a second-order section at twice the cutoff: about
        // -21 dB, so an order of magnitude below the naive alias.
        assert!(
            amplitude(&filtered) < 0.1 * amplitude(&naive),
            "{}",
            amplitude(&filtered)
        );
        assert!(report.is_some());
        assert_eq!(filtered.accumulated_phase_delay(), Duration::ZERO);
        let ops: Vec<_> = filtered
            .provenance
            .iter()
            .map(|s| s.operation.as_str())
            .collect();
        assert_eq!(ops, ["lowpass", "resample"]);
    }

    #[test]
    fn upsampling_with_anti_aliasing_runs_no_filter() {
        let (_, report) = resample_anti_aliased(&record_at(30.0), Hertz(60.0)).unwrap();
        assert!(report.is_none());
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

    #[test]
    fn a_missing_sample_stays_missing_instead_of_becoming_data() {
        // An occluded marker's placeholder must not be lerped into the
        // output and relabeled Interpolated: the gap stays a gap.
        let mut source = record_at(30.0);
        source.frames[1].validity[0] = Validity::Missing;
        let resampled = resample(&source, Hertz(60.0)).unwrap();
        let touched: Vec<&Validity> = resampled
            .frames
            .iter()
            .map(|f| &f.validity[0])
            .filter(|v| **v == Validity::Missing)
            .collect();
        assert!(
            !touched.is_empty(),
            "every output bracketed by the missing source sample stays Missing"
        );
        assert!(resampled
            .frames
            .iter()
            .all(|f| f.validity[0] != Validity::Valid));
    }

    /// A 2 s record at a nominal 100 Hz where every seventh interval
    /// stretches to 15 ms — the dropped-frame shape real captures have.
    /// The median interval stays 10 ms, so the median rate is the right
    /// resampling target.
    fn jittered_record() -> MotionRecord<HumanSkeleton> {
        let channels = vec![Channel {
            name: Name::new("joint").unwrap(),
            parent: None,
            offset: Default::default(),
        }];
        let mut t = 0i64;
        let frames = (0..200)
            .map(|i| {
                let f = frame(t, Quat::IDENTITY);
                t += if i % 7 == 6 { 15_000_000 } else { 10_000_000 };
                f
            })
            .collect();
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
    fn lowpass_resampled_takes_the_non_uniform_record_the_filter_refuses() {
        let source = jittered_record();
        // The direct filter refuses this record by name...
        assert!(matches!(
            lowpass(&mut source.clone(), Hertz(10.0), FilterMode::ZeroPhase),
            Err(FilterError::NonUniformSampling { .. })
        ));
        // ...and the resampling entry point takes it, producing a uniform
        // record at the median rate with both provenance steps.
        let (out, report) = lowpass_resampled(&source, Hertz(10.0), FilterMode::ZeroPhase).unwrap();
        let rate = uniform_rate(&out).unwrap();
        assert!((rate.value() - 100.0).abs() < 1.0, "{rate}");
        assert_eq!(report.phase_delay, Duration::ZERO);
        let steps: Vec<&str> = out
            .provenance
            .iter()
            .map(|s| s.operation.as_str())
            .collect();
        assert!(
            steps.contains(&"resample") && steps.contains(&"lowpass"),
            "{steps:?}"
        );
    }

    #[test]
    fn lowpass_resampled_filters_a_uniform_record_without_resampling() {
        let source = record_at(100.0);
        let (out, _) = lowpass_resampled(&source, Hertz(10.0), FilterMode::ZeroPhase).unwrap();
        let steps: Vec<&str> = out
            .provenance
            .iter()
            .map(|s| s.operation.as_str())
            .collect();
        assert!(!steps.contains(&"resample"), "{steps:?}");
        assert!(steps.contains(&"lowpass"));
    }

    #[test]
    fn lowpass_resampled_passes_through_the_refusals_it_cannot_fix() {
        // Too short for the filter and the resampler alike.
        let mut short = record_at(100.0);
        short.frames.truncate(1);
        assert!(matches!(
            lowpass_resampled(&short, Hertz(10.0), FilterMode::ZeroPhase),
            Err(AntiAliasError::Filter(FilterError::TooShort))
        ));
        // A cutoff the uniform record cannot support.
        assert!(matches!(
            lowpass_resampled(&record_at(100.0), Hertz(60.0), FilterMode::ZeroPhase),
            Err(AntiAliasError::Filter(FilterError::InvalidCutoff { .. }))
        ));
    }
}

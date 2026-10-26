//! The canonical motion record of SPECS.md Section 7.2.
//!
//! Three record *kinds* are kept distinct here, because Section 7.2 says so
//! plainly: "Human skeleton motion, parametric human-body parameters, and robot
//! joint trajectories are distinct record types. BVH, C3D, and SMPL-family data
//! are not interchangeable." They are separate types rather than a flag, so a
//! retargeter cannot be handed a robot trajectory by accident.
//!
//! Every transformation applied to a record appends a [`ProvenanceStep`], which
//! is what Section 7.2's "provenance for every transformation" requires and what
//! lets a validation report say which filter introduced a phase delay.

use crate::frames::{EulerOrder, FrameId, Quat, Vec3};
use crate::ids::{Name, NameError};
use crate::schema::{SchemaError, SchemaVersion};
use crate::time::{ClockId, Duration, Timestamp};
use crate::units::Metres;
use core::fmt;

/// The schema version this crate writes for motion records.
pub const MOTION_SCHEMA_VERSION: SchemaVersion = SchemaVersion::new(1, 0);

/// What kind of motion a record holds.
///
/// Carried as a type parameter on [`MotionRecord`] so the three kinds cannot be
/// mixed in one collection without the code saying so.
pub trait MotionKind: Copy + fmt::Debug + PartialEq {
    /// The canonical identifier written into the record.
    const ID: &'static str;
}

/// Human skeleton motion: named joint rotations on a source skeleton.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct HumanSkeleton;

impl MotionKind for HumanSkeleton {
    const ID: &'static str = "human-skeleton";
}

/// Robot reference motion: joint values on a named robot profile.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RobotReference;

impl MotionKind for RobotReference {
    const ID: &'static str = "robot-reference";
}

/// Parametric human-body parameters, such as an SMPL-family fit.
///
/// Present as a distinct kind so that a body-model record cannot be read as a
/// skeleton record. SPECS.md Section 7.2 also notes that "SMPL assets and source
/// datasets may have independent license restrictions", which is why
/// [`SourceLicense`] is a required field of every record.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BodyModelParameters;

impl MotionKind for BodyModelParameters {
    const ID: &'static str = "body-model-parameters";
}

/// A dataset's licence and what it permits.
///
/// SPECS.md Section 7.5 requires tracking "license eligibility, retention
/// policy, consent where relevant, and dataset revisions". None of these fields
/// has a default: C-07 leaves corpus eligibility open, and a record that cannot
/// state its terms is not usable.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SourceLicense {
    /// Licence identifier or document reference.
    pub identifier: String,
    /// Whether the terms permit use as training data.
    pub training_permitted: bool,
    /// Whether the terms permit redistribution of derived artifacts.
    pub redistribution_permitted: bool,
    /// Whether commercial use is permitted.
    pub commercial_permitted: bool,
    /// Retention obligation in plain words, or `None` when unrestricted.
    pub retention: Option<String>,
    /// Where the terms were read from, with a date.
    pub source_reference: String,
}

impl SourceLicense {
    /// A licence whose terms have not yet been audited.
    ///
    /// Permits nothing. C-07's interim position is that "No corpus is assumed
    /// usable", so this is the correct state for a freshly imported file and
    /// [`SourceLicense::permits_training`] will refuse it.
    #[must_use]
    pub fn unaudited(identifier: impl Into<String>) -> Self {
        Self {
            identifier: identifier.into(),
            training_permitted: false,
            redistribution_permitted: false,
            commercial_permitted: false,
            retention: None,
            source_reference: "unaudited".to_owned(),
        }
    }

    /// Whether this record may enter a training set.
    #[must_use]
    pub const fn permits_training(&self) -> bool {
        self.training_permitted
    }
}

/// One transformation applied to a record.
#[derive(Clone, PartialEq, Debug)]
pub struct ProvenanceStep {
    /// What was done, as a stable operation name.
    pub operation: String,
    /// Parameters, as `key=value` pairs in a stable order.
    pub parameters: Vec<(String, String)>,
    /// Phase delay the step introduces, where it is a filter.
    ///
    /// SPECS.md Section 8.2 item 4: "Filter and resample with documented phase
    /// delay". Recorded here so a downstream delay model can account for it.
    pub phase_delay: Option<Duration>,
    /// Software revision that performed the step.
    pub revision: String,
}

impl ProvenanceStep {
    /// A step with no parameters.
    #[must_use]
    pub fn new(operation: impl Into<String>, revision: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            parameters: Vec::new(),
            phase_delay: None,
            revision: revision.into(),
        }
    }

    /// Adds a parameter.
    #[must_use]
    pub fn with_parameter(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.parameters.push((key.into(), value.into()));
        self
    }

    /// Records a phase delay.
    #[must_use]
    pub const fn with_phase_delay(mut self, delay: Duration) -> Self {
        self.phase_delay = Some(delay);
        self
    }
}

/// How a joint's value is represented in a record.
#[derive(Clone, PartialEq, Debug)]
pub enum JointValue {
    /// A rotation, with the source's Euler order recorded for provenance even
    /// though the stored value is a quaternion.
    Rotation {
        /// The rotation.
        rotation: Quat,
        /// The order the source expressed it in, if it used Euler angles.
        source_order: Option<EulerOrder>,
    },
    /// A scalar joint coordinate, in radians for a revolute joint and metres
    /// for a prismatic one. The unit is declared by the channel, not guessed.
    Scalar {
        /// The value.
        value: f64,
        /// Canonical unit symbol.
        unit: &'static str,
    },
    /// A position, for a marker-style record.
    Position {
        /// The position in metres.
        position: Vec3,
    },
}

/// A named channel of a motion record.
#[derive(Clone, PartialEq, Debug)]
pub struct Channel {
    /// Channel name, matching the source skeleton's joint name.
    pub name: Name,
    /// Index of the parent channel, or `None` for a root.
    pub parent: Option<usize>,
    /// Rest-pose offset from the parent, in metres.
    pub offset: Vec3,
}

/// Per-frame validity of a channel's sample.
///
/// SPECS.md Section 7.2 requires "validity masks"; Section 8.2 item 3 requires
/// marking gaps rather than interpolating over them silently.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Validity {
    /// Measured and usable.
    Valid,
    /// Absent in the source; the value is a placeholder and must not be used.
    Missing,
    /// Present but filled by a named interpolation step; usable with care.
    Interpolated,
}

/// A contact label with the confidence attached.
///
/// SPECS.md Section 7.2 requires "contact labels with confidence", and
/// Section 8.2 item 5 requires contact intervals "with uncertainty". A boolean
/// contact flag is exactly what this type exists to prevent.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ContactLabel {
    /// Site the label applies to.
    pub site: crate::ids::ContactSiteId,
    /// Whether contact is asserted.
    pub in_contact: bool,
    /// Confidence in `[0, 1]`.
    pub confidence: f64,
}

/// One frame of a motion record.
#[derive(Clone, PartialEq, Debug)]
pub struct MotionFrame {
    /// Sampling instant on the source clock.
    pub timestamp: Timestamp,
    /// Root pose relative to the record's reference frame.
    pub root: crate::frames::Transform,
    /// One value per channel, in channel order.
    pub values: Vec<JointValue>,
    /// One validity per channel, in channel order.
    pub validity: Vec<Validity>,
    /// Contact labels asserted at this frame.
    pub contacts: Vec<ContactLabel>,
}

/// A task label on a span of frames.
#[derive(Clone, PartialEq, Debug)]
pub struct TaskLabel {
    /// The label, such as `jab` or `guard`.
    pub label: Name,
    /// First frame of the span.
    pub start: usize,
    /// One past the last frame.
    pub end: usize,
}

/// The canonical motion record.
#[derive(Clone, PartialEq, Debug)]
pub struct MotionRecord<K: MotionKind> {
    /// Schema version.
    pub schema_version: SchemaVersion,
    /// Content hash of the immutable original, as a lowercase hex digest.
    pub source_hash: String,
    /// Licence terms of the source.
    pub source_license: SourceLicense,
    /// Subject identifier, where the source provides one.
    pub subject_id: Option<String>,
    /// Recording session identifier. Splits are taken by session before
    /// windowing (SPECS.md Section 7.5), so this field is what makes a split
    /// honest.
    pub session_id: String,
    /// Hash of the skeleton or body model the values are expressed against.
    pub skeleton_hash: String,
    /// Frame the root pose is expressed in.
    pub reference_frame: FrameId,
    /// Clock the frame timestamps are on.
    pub clock: ClockId,
    /// Channel layout.
    pub channels: Vec<Channel>,
    /// Frames, in non-decreasing timestamp order.
    pub frames: Vec<MotionFrame>,
    /// Task labels over frame spans.
    pub task_labels: Vec<TaskLabel>,
    /// Every transformation applied, oldest first.
    pub provenance: Vec<ProvenanceStep>,
    /// Approximate subject height, used to normalize endpoint error in the
    /// retargeting report (SPECS.md Section 8.3).
    pub subject_height: Option<Metres>,
    kind: K,
}

/// Why a motion record was rejected.
#[derive(Clone, PartialEq, Debug)]
pub enum MotionError {
    /// A frame's value or validity count disagreed with the channel count.
    ChannelCountMismatch {
        /// Frame index.
        frame: usize,
        /// Channels declared.
        expected: usize,
        /// Values present.
        found: usize,
    },
    /// Timestamps were not non-decreasing.
    NonMonotonicTimestamps {
        /// Index of the offending frame.
        frame: usize,
    },
    /// A frame carried a timestamp from a different clock.
    ClockMismatch {
        /// Index of the offending frame.
        frame: usize,
    },
    /// A channel's parent index did not refer to an earlier channel.
    ///
    /// Requiring the parent to precede the child makes forward kinematics a
    /// single forward pass and makes a cycle unrepresentable.
    InvalidParent {
        /// Channel index.
        channel: usize,
        /// Parent index given.
        parent: usize,
    },
    /// A contact confidence was outside `[0, 1]`.
    InvalidConfidence {
        /// Frame index.
        frame: usize,
        /// Value given.
        value: f64,
    },
    /// A task label span was empty or out of range.
    InvalidLabelSpan {
        /// The label.
        label: String,
        /// Start index.
        start: usize,
        /// End index.
        end: usize,
    },
    /// A numeric field was NaN or infinite.
    NotFinite {
        /// Frame index.
        frame: usize,
        /// Channel index, where applicable.
        channel: Option<usize>,
    },
    /// The record declared no channels or no frames.
    Empty,
    /// The licence does not permit the requested use.
    LicenseForbids {
        /// Licence identifier.
        identifier: String,
        /// The use that was refused.
        use_case: &'static str,
    },
    /// A schema-level problem.
    Schema(SchemaError),
    /// A name in the record was invalid.
    Name(NameError),
}

impl fmt::Display for MotionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChannelCountMismatch {
                frame,
                expected,
                found,
            } => write!(
                f,
                "frame {frame} has {found} values for {expected} channels"
            ),
            Self::NonMonotonicTimestamps { frame } => {
                write!(f, "frame {frame} is not after its predecessor")
            }
            Self::ClockMismatch { frame } => {
                write!(f, "frame {frame} is stamped on a different clock")
            }
            Self::InvalidParent { channel, parent } => {
                write!(
                    f,
                    "channel {channel} names parent {parent}, which is not earlier"
                )
            }
            Self::InvalidConfidence { frame, value } => {
                write!(
                    f,
                    "frame {frame} has contact confidence {value}, outside [0,1]"
                )
            }
            Self::InvalidLabelSpan { label, start, end } => {
                write!(
                    f,
                    "label {label:?} spans [{start},{end}), which is empty or out of range"
                )
            }
            Self::NotFinite { frame, channel } => match channel {
                Some(c) => write!(f, "frame {frame} channel {c} is not finite"),
                None => write!(f, "frame {frame} has a non-finite field"),
            },
            Self::Empty => write!(f, "record has no channels or no frames"),
            Self::LicenseForbids {
                identifier,
                use_case,
            } => {
                write!(f, "licence {identifier:?} does not permit {use_case}")
            }
            Self::Schema(e) => write!(f, "{e}"),
            Self::Name(e) => write!(f, "{e}"),
        }
    }
}

impl core::error::Error for MotionError {}

impl From<SchemaError> for MotionError {
    fn from(e: SchemaError) -> Self {
        Self::Schema(e)
    }
}

impl From<NameError> for MotionError {
    fn from(e: NameError) -> Self {
        Self::Name(e)
    }
}

impl<K: MotionKind> MotionRecord<K> {
    /// Builds a record and validates its internal consistency.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kind: K,
        source_hash: impl Into<String>,
        source_license: SourceLicense,
        session_id: impl Into<String>,
        skeleton_hash: impl Into<String>,
        reference_frame: FrameId,
        clock: ClockId,
        channels: Vec<Channel>,
        frames: Vec<MotionFrame>,
    ) -> Result<Self, MotionError> {
        let record = Self {
            schema_version: MOTION_SCHEMA_VERSION,
            source_hash: source_hash.into(),
            source_license,
            subject_id: None,
            session_id: session_id.into(),
            skeleton_hash: skeleton_hash.into(),
            reference_frame,
            clock,
            channels,
            frames,
            task_labels: Vec::new(),
            provenance: Vec::new(),
            subject_height: None,
            kind,
        };
        record.validate()?;
        Ok(record)
    }

    /// The record's kind.
    #[must_use]
    pub const fn kind(&self) -> K {
        self.kind
    }

    /// The canonical kind identifier.
    #[must_use]
    pub const fn kind_id(&self) -> &'static str {
        K::ID
    }

    /// Number of frames.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Duration from the first to the last frame.
    #[must_use]
    pub fn duration(&self) -> Duration {
        match (self.frames.first(), self.frames.last()) {
            (Some(a), Some(b)) => b
                .timestamp
                .since(a.timestamp)
                .map(|(d, _)| d)
                .unwrap_or(Duration::ZERO),
            _ => Duration::ZERO,
        }
    }

    /// Mean sampling rate in hertz, or `None` for fewer than two frames.
    #[must_use]
    pub fn mean_rate(&self) -> Option<crate::units::Hertz> {
        if self.frames.len() < 2 {
            return None;
        }
        let secs = self.duration().as_secs_f64();
        if secs <= 0.0 {
            return None;
        }
        Some(crate::units::Hertz((self.frames.len() - 1) as f64 / secs))
    }

    /// Index of a channel by name.
    #[must_use]
    pub fn channel_index(&self, name: &str) -> Option<usize> {
        self.channels.iter().position(|c| c.name.as_str() == name)
    }

    /// Appends a provenance step.
    pub fn record_step(&mut self, step: ProvenanceStep) {
        self.provenance.push(step);
    }

    /// Total phase delay accumulated across every filtering step.
    #[must_use]
    pub fn accumulated_phase_delay(&self) -> Duration {
        self.provenance
            .iter()
            .filter_map(|s| s.phase_delay)
            .fold(Duration::ZERO, |a, b| a + b)
    }

    /// Checks the record's internal consistency.
    pub fn validate(&self) -> Result<(), MotionError> {
        if self.channels.is_empty() || self.frames.is_empty() {
            return Err(MotionError::Empty);
        }
        for (i, c) in self.channels.iter().enumerate() {
            if let Some(p) = c.parent {
                if p >= i {
                    return Err(MotionError::InvalidParent {
                        channel: i,
                        parent: p,
                    });
                }
            }
            if !c.offset.is_finite() {
                return Err(MotionError::NotFinite {
                    frame: 0,
                    channel: Some(i),
                });
            }
        }
        let n = self.channels.len();
        let mut previous: Option<Timestamp> = None;
        for (fi, frame) in self.frames.iter().enumerate() {
            if frame.values.len() != n {
                return Err(MotionError::ChannelCountMismatch {
                    frame: fi,
                    expected: n,
                    found: frame.values.len(),
                });
            }
            if frame.validity.len() != n {
                return Err(MotionError::ChannelCountMismatch {
                    frame: fi,
                    expected: n,
                    found: frame.validity.len(),
                });
            }
            if frame.timestamp.clock() != self.clock {
                return Err(MotionError::ClockMismatch { frame: fi });
            }
            if let Some(prev) = previous {
                if frame.timestamp.nanos() < prev.nanos() {
                    return Err(MotionError::NonMonotonicTimestamps { frame: fi });
                }
            }
            previous = Some(frame.timestamp);
            if !frame.root.is_finite() {
                return Err(MotionError::NotFinite {
                    frame: fi,
                    channel: None,
                });
            }
            for (ci, (value, validity)) in frame.values.iter().zip(&frame.validity).enumerate() {
                // A missing sample's placeholder is not required to be finite,
                // because the validity mask already says not to read it.
                if *validity == Validity::Missing {
                    continue;
                }
                let finite = match value {
                    JointValue::Rotation { rotation, .. } => {
                        rotation.w.is_finite()
                            && rotation.x.is_finite()
                            && rotation.y.is_finite()
                            && rotation.z.is_finite()
                    }
                    JointValue::Scalar { value, .. } => value.is_finite(),
                    JointValue::Position { position } => position.is_finite(),
                };
                if !finite {
                    return Err(MotionError::NotFinite {
                        frame: fi,
                        channel: Some(ci),
                    });
                }
            }
            for contact in &frame.contacts {
                if !(0.0..=1.0).contains(&contact.confidence) || !contact.confidence.is_finite() {
                    return Err(MotionError::InvalidConfidence {
                        frame: fi,
                        value: contact.confidence,
                    });
                }
            }
        }
        for label in &self.task_labels {
            if label.start >= label.end || label.end > self.frames.len() {
                return Err(MotionError::InvalidLabelSpan {
                    label: label.label.to_string(),
                    start: label.start,
                    end: label.end,
                });
            }
        }
        Ok(())
    }

    /// Refuses the record if its licence does not permit training use.
    pub fn require_training_license(&self) -> Result<(), MotionError> {
        if self.source_license.permits_training() {
            Ok(())
        } else {
            Err(MotionError::LicenseForbids {
                identifier: self.source_license.identifier.clone(),
                use_case: "use as training data",
            })
        }
    }

    /// Fraction of samples marked valid, which a curation report shows next to
    /// the accept/reject decision.
    #[must_use]
    pub fn valid_fraction(&self) -> f64 {
        let total = self.frames.len() * self.channels.len();
        if total == 0 {
            return 0.0;
        }
        let valid = self
            .frames
            .iter()
            .flat_map(|f| f.validity.iter())
            .filter(|v| **v == Validity::Valid)
            .count();
        valid as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::Transform;
    use crate::ids::ContactSiteId;

    fn license() -> SourceLicense {
        SourceLicense {
            identifier: "fixture-internal".to_owned(),
            training_permitted: true,
            redistribution_permitted: false,
            commercial_permitted: false,
            retention: None,
            source_reference: "fixtures/motion/README.md 2026-10-26".to_owned(),
        }
    }

    fn channels() -> Vec<Channel> {
        vec![
            Channel {
                name: Name::new("Hips").unwrap(),
                parent: None,
                offset: Vec3::ZERO,
            },
            Channel {
                name: Name::new("Spine").unwrap(),
                parent: Some(0),
                offset: Vec3::new(0.0, 0.0, 0.2),
            },
        ]
    }

    fn frame(nanos: i64) -> MotionFrame {
        MotionFrame {
            timestamp: Timestamp::sampling(nanos, ClockId::Source(0)),
            root: Transform::identity(FrameId::WORLD),
            values: vec![
                JointValue::Rotation {
                    rotation: Quat::IDENTITY,
                    source_order: None,
                },
                JointValue::Rotation {
                    rotation: Quat::IDENTITY,
                    source_order: Some(EulerOrder::Zyx),
                },
            ],
            validity: vec![Validity::Valid, Validity::Valid],
            contacts: Vec::new(),
        }
    }

    fn record() -> MotionRecord<HumanSkeleton> {
        MotionRecord::new(
            HumanSkeleton,
            "abc",
            license(),
            "session-1",
            "skel-1",
            FrameId::WORLD,
            ClockId::Source(0),
            channels(),
            vec![frame(0), frame(10_000_000), frame(20_000_000)],
        )
        .unwrap()
    }

    #[test]
    fn a_valid_record_reports_its_rate_and_duration() {
        let r = record();
        assert_eq!(r.frame_count(), 3);
        assert_eq!(r.duration(), Duration::from_millis(20));
        assert!((r.mean_rate().unwrap().value() - 100.0).abs() < 1e-9);
        assert_eq!(r.kind_id(), "human-skeleton");
        assert_eq!(r.channel_index("Spine"), Some(1));
        assert_eq!(r.channel_index("Missing"), None);
        assert!((r.valid_fraction() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn kinds_are_distinct_identifiers() {
        assert_eq!(HumanSkeleton::ID, "human-skeleton");
        assert_eq!(RobotReference::ID, "robot-reference");
        assert_eq!(BodyModelParameters::ID, "body-model-parameters");
    }

    #[test]
    fn out_of_order_timestamps_are_rejected() {
        let built = MotionRecord::new(
            HumanSkeleton,
            "abc",
            license(),
            "s",
            "k",
            FrameId::WORLD,
            ClockId::Source(0),
            channels(),
            vec![frame(10_000_000), frame(0)],
        );
        assert!(matches!(
            built,
            Err(MotionError::NonMonotonicTimestamps { frame: 1 })
        ));
    }

    #[test]
    fn a_frame_on_the_wrong_clock_is_rejected() {
        let mut f = frame(0);
        f.timestamp = Timestamp::sampling(0, ClockId::HostMonotonic);
        let built = MotionRecord::new(
            HumanSkeleton,
            "abc",
            license(),
            "s",
            "k",
            FrameId::WORLD,
            ClockId::Source(0),
            channels(),
            vec![f],
        );
        assert!(matches!(
            built,
            Err(MotionError::ClockMismatch { frame: 0 })
        ));
    }

    #[test]
    fn a_forward_parent_reference_is_rejected() {
        let bad = vec![
            Channel {
                name: Name::new("Child").unwrap(),
                parent: Some(1),
                offset: Vec3::ZERO,
            },
            Channel {
                name: Name::new("Parent").unwrap(),
                parent: None,
                offset: Vec3::ZERO,
            },
        ];
        let built = MotionRecord::new(
            HumanSkeleton,
            "abc",
            license(),
            "s",
            "k",
            FrameId::WORLD,
            ClockId::Source(0),
            bad,
            vec![frame(0)],
        );
        assert!(matches!(
            built,
            Err(MotionError::InvalidParent {
                channel: 0,
                parent: 1
            })
        ));
    }

    #[test]
    fn channel_count_mismatch_is_rejected() {
        let mut f = frame(0);
        f.values.pop();
        let built = MotionRecord::new(
            HumanSkeleton,
            "abc",
            license(),
            "s",
            "k",
            FrameId::WORLD,
            ClockId::Source(0),
            channels(),
            vec![f],
        );
        assert!(matches!(
            built,
            Err(MotionError::ChannelCountMismatch {
                frame: 0,
                expected: 2,
                found: 1
            })
        ));
    }

    #[test]
    fn a_missing_sample_may_hold_a_placeholder_but_a_valid_one_may_not() {
        let mut f = frame(0);
        f.values[1] = JointValue::Scalar {
            value: f64::NAN,
            unit: "rad",
        };
        f.validity[1] = Validity::Missing;
        let ok = MotionRecord::new(
            HumanSkeleton,
            "abc",
            license(),
            "s",
            "k",
            FrameId::WORLD,
            ClockId::Source(0),
            channels(),
            vec![f.clone()],
        );
        assert!(
            ok.is_ok(),
            "a masked NaN is permitted: the mask says do not read it"
        );
        assert!((ok.unwrap().valid_fraction() - 0.5).abs() < 1e-12);

        f.validity[1] = Validity::Valid;
        let bad = MotionRecord::new(
            HumanSkeleton,
            "abc",
            license(),
            "s",
            "k",
            FrameId::WORLD,
            ClockId::Source(0),
            channels(),
            vec![f],
        );
        assert!(matches!(
            bad,
            Err(MotionError::NotFinite {
                frame: 0,
                channel: Some(1)
            })
        ));
    }

    #[test]
    fn contact_confidence_outside_the_unit_interval_is_rejected() {
        let mut f = frame(0);
        f.contacts.push(ContactLabel {
            site: ContactSiteId::new(0),
            in_contact: true,
            confidence: 1.5,
        });
        let built = MotionRecord::new(
            HumanSkeleton,
            "abc",
            license(),
            "s",
            "k",
            FrameId::WORLD,
            ClockId::Source(0),
            channels(),
            vec![f],
        );
        assert!(matches!(built, Err(MotionError::InvalidConfidence { .. })));
    }

    #[test]
    fn an_empty_record_is_rejected() {
        let built = MotionRecord::new(
            HumanSkeleton,
            "abc",
            license(),
            "s",
            "k",
            FrameId::WORLD,
            ClockId::Source(0),
            channels(),
            Vec::new(),
        );
        assert!(matches!(built, Err(MotionError::Empty)));
    }

    #[test]
    fn an_unaudited_license_permits_nothing() {
        let mut r = record();
        r.source_license = SourceLicense::unaudited("AMASS/unknown-component");
        assert!(matches!(
            r.require_training_license(),
            Err(MotionError::LicenseForbids { .. })
        ));
        assert!(record().require_training_license().is_ok());
    }

    #[test]
    fn phase_delay_accumulates_across_steps() {
        let mut r = record();
        r.record_step(
            ProvenanceStep::new("resample", "rev1")
                .with_parameter("rate_hz", "100")
                .with_phase_delay(Duration::from_millis(5)),
        );
        r.record_step(
            ProvenanceStep::new("lowpass", "rev1").with_phase_delay(Duration::from_millis(3)),
        );
        r.record_step(ProvenanceStep::new("segment", "rev1"));
        assert_eq!(r.accumulated_phase_delay(), Duration::from_millis(8));
        assert_eq!(r.provenance.len(), 3);
    }

    #[test]
    fn a_label_span_outside_the_frames_is_rejected() {
        let mut r = record();
        r.task_labels.push(TaskLabel {
            label: Name::new("jab").unwrap(),
            start: 1,
            end: 99,
        });
        assert!(matches!(
            r.validate(),
            Err(MotionError::InvalidLabelSpan { .. })
        ));
    }
}

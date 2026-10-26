//! The robot episode record of SPECS.md Section 7.3.
//!
//! Two requirements from that section drive the shape of this module.
//!
//! "Store both raw commands and applied/clamped commands so training cannot
//! confuse policy intent with actuator input" — so [`CommandRecord`] holds both,
//! and there is no constructor that sets only one.
//!
//! "Record validity and provenance for estimated torque; motor current is not
//! automatically calibrated joint torque" — so [`TorqueSample`] states its
//! [`TorqueProvenance`], and a consumer that needs measured torque can refuse a
//! current-derived estimate rather than train on it.

use crate::ids::{ActuatorId, ContactSiteId, JointId, SensorId};
use crate::mode::ModeContext;
use crate::rng::SeedStream;
use crate::schema::SchemaVersion;
use crate::time::{ClockId, Duration, Timestamp};
use crate::units::{Amperes, Celsius, NewtonMetres, Newtons, Radians, RadiansPerSecond, Watts};
use core::fmt;

/// The schema version this crate writes for episode records.
pub const EPISODE_SCHEMA_VERSION: SchemaVersion = SchemaVersion::new(1, 0);

/// Where an estimated torque value came from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TorqueProvenance {
    /// Measured by a joint torque sensor.
    MeasuredAtJoint,
    /// Derived from motor current through a calibrated constant, with the
    /// calibration record named.
    CurrentWithCalibration,
    /// Derived from motor current through a nominal datasheet constant.
    ///
    /// SPECS.md Section 7.3 warns that "motor current is not automatically
    /// calibrated joint torque"; this variant is the honest label for that case.
    CurrentNominal,
    /// Commanded torque echoed back by the actuator, not a measurement.
    CommandedEcho,
}

impl TorqueProvenance {
    /// Whether a value with this provenance may be used as a measurement in
    /// system identification.
    #[must_use]
    pub const fn is_measurement(self) -> bool {
        matches!(self, Self::MeasuredAtJoint | Self::CurrentWithCalibration)
    }

    /// The canonical identifier used in records.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::MeasuredAtJoint => "measured-at-joint",
            Self::CurrentWithCalibration => "current-with-calibration",
            Self::CurrentNominal => "current-nominal",
            Self::CommandedEcho => "commanded-echo",
        }
    }
}

/// A torque value with its provenance and validity.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct TorqueSample {
    /// The value.
    pub value: NewtonMetres,
    /// Where it came from.
    pub provenance: TorqueProvenance,
    /// Whether the value is usable at all.
    pub valid: bool,
}

/// A raw command and the command actually applied after clamping.
///
/// The pair exists so a learner cannot confuse intent with input; the difference
/// between them is also the record of how often the supervisor intervened.
#[derive(Clone, PartialEq, Debug)]
pub struct CommandRecord {
    /// What the controller asked for, before any clamp.
    pub raw: Vec<f64>,
    /// What the actuator interface received, after clamping and supervision.
    pub applied: Vec<f64>,
    /// Canonical unit of both vectors.
    pub unit: &'static str,
    /// When the command was emitted.
    pub emitted_at: Timestamp,
    /// Actuator acknowledgement, where the device provides one.
    pub acknowledged_at: Option<Timestamp>,
}

impl CommandRecord {
    /// Builds a pair, rejecting mismatched widths.
    pub fn new(
        raw: Vec<f64>,
        applied: Vec<f64>,
        unit: &'static str,
        emitted_at: Timestamp,
    ) -> Result<Self, EpisodeError> {
        if raw.len() != applied.len() {
            return Err(EpisodeError::WidthMismatch {
                field: "command",
                expected: raw.len(),
                found: applied.len(),
            });
        }
        Ok(Self {
            raw,
            applied,
            unit,
            emitted_at,
            acknowledged_at: None,
        })
    }

    /// True when the applied command differs from the raw command anywhere.
    #[must_use]
    pub fn was_modified(&self) -> bool {
        self.raw
            .iter()
            .zip(&self.applied)
            .any(|(r, a)| (r - a).abs() > f64::EPSILON)
    }

    /// Largest absolute difference between intent and input.
    #[must_use]
    pub fn max_modification(&self) -> f64 {
        self.raw
            .iter()
            .zip(&self.applied)
            .fold(0.0f64, |acc, (r, a)| acc.max((r - a).abs()))
    }

    /// Round-trip time to the acknowledgement, where there is one.
    #[must_use]
    pub fn acknowledgement_latency(&self) -> Option<Duration> {
        self.acknowledged_at
            .and_then(|ack| ack.since(self.emitted_at).ok())
            .map(|(d, _)| d)
    }
}

/// A per-joint sample of measured motion.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct JointSample {
    /// The joint.
    pub joint: JointId,
    /// Measured position.
    pub position: Radians,
    /// Measured velocity.
    pub velocity: RadiansPerSecond,
    /// Torque, with provenance.
    pub torque: TorqueSample,
    /// Whether the encoder reading is valid this tick.
    pub valid: bool,
}

/// A per-actuator health sample.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ActuatorHealth {
    /// The actuator.
    pub actuator: ActuatorId,
    /// Winding or case temperature.
    pub temperature: Celsius,
    /// Current draw.
    pub current: Amperes,
    /// Electrical power.
    pub power: Watts,
    /// Whether the actuator reports saturation this tick.
    pub saturated: bool,
}

/// A contact measurement at a site.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ContactSample {
    /// The site.
    pub site: ContactSiteId,
    /// Normal force.
    pub normal_force: Newtons,
    /// Whether contact is asserted.
    pub in_contact: bool,
    /// Confidence in `[0, 1]`.
    pub confidence: f64,
}

/// An uncertainty-carrying estimate produced by `robot-perception`.
#[derive(Clone, PartialEq, Debug)]
pub struct EstimateSample {
    /// Estimated values.
    pub values: Vec<f64>,
    /// Diagonal of the covariance, in the square of each value's unit.
    pub variance: Vec<f64>,
    /// Whether the estimator considers the estimate usable.
    pub valid: bool,
}

/// An intervention by the supervisor, recorded with enough context to
/// reconstruct why a command was refused (SPECS.md Section 12.1).
#[derive(Clone, PartialEq, Debug)]
pub struct SupervisorIntervention {
    /// When it happened.
    pub at: Timestamp,
    /// The check that fired, as a stable identifier.
    pub check: String,
    /// Which component of the command was affected, where applicable.
    pub component: Option<usize>,
    /// How far outside the limit the command was.
    pub excess: Option<f64>,
    /// What the supervisor did.
    pub action: String,
}

/// One tick of an episode.
#[derive(Clone, PartialEq, Debug)]
pub struct EpisodeTick {
    /// Sequence number within the episode.
    pub index: u64,
    /// Sensor sampling instant, or arrival time where the device gives none.
    pub sensor_time: Timestamp,
    /// When the host observed the sample.
    pub arrival_time: Timestamp,
    /// Raw joint measurements.
    pub joints: Vec<JointSample>,
    /// Actuator health.
    pub actuators: Vec<ActuatorHealth>,
    /// Contacts.
    pub contacts: Vec<ContactSample>,
    /// Estimated base state with uncertainty.
    pub estimate: Option<EstimateSample>,
    /// Desired joint motion from the reference or policy target.
    pub desired: Option<Vec<f64>>,
    /// The command, raw and applied.
    pub command: Option<CommandRecord>,
    /// Active policy or expert identity.
    pub active_policy: Option<String>,
    /// Transition state, where a switch is in progress.
    pub transition: Option<String>,
    /// Interventions on this tick.
    pub interventions: Vec<SupervisorIntervention>,
}

impl EpisodeTick {
    /// Sensor-to-command age, as defined in the timing table of Section 5.
    ///
    /// Measured from the sampling instant to the command's emission, and
    /// returned with the accumulated timestamp uncertainty so an age computed
    /// from arrival times does not read as exact.
    pub fn sensor_to_command_age(&self) -> Option<(Duration, Duration)> {
        let command = self.command.as_ref()?;
        command.emitted_at.since(self.sensor_time).ok()
    }

    /// Transport delay from sampling to host arrival.
    #[must_use]
    pub fn transport_delay(&self) -> Option<Duration> {
        self.arrival_time
            .since(self.sensor_time)
            .ok()
            .map(|(d, _)| d)
    }
}

/// Why an episode ended.
#[derive(Clone, PartialEq, Debug)]
pub enum CompletionReason {
    /// The task succeeded.
    TaskSucceeded,
    /// The task's termination condition fired.
    Terminated {
        /// The named condition.
        condition: String,
    },
    /// The episode hit its time limit; not a terminal failure.
    ///
    /// SPECS.md Section 9.2 requires "explicit handling of termination versus
    /// time-limit truncation", and conflating the two biases every return
    /// calculation downstream.
    Truncated,
    /// The operator stopped the run.
    OperatorStopped {
        /// Reason given.
        reason: String,
    },
    /// A fault terminated the run.
    Fault {
        /// The fault's stable identifier.
        fault: String,
    },
}

impl CompletionReason {
    /// Whether this completion is a terminal state for bootstrapping.
    ///
    /// Truncation is not: the value of the next state still counts.
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        !matches!(self, Self::Truncated)
    }

    /// The canonical identifier used in records.
    #[must_use]
    pub const fn id(&self) -> &'static str {
        match self {
            Self::TaskSucceeded => "task-succeeded",
            Self::Terminated { .. } => "terminated",
            Self::Truncated => "truncated",
            Self::OperatorStopped { .. } => "operator-stopped",
            Self::Fault { .. } => "fault",
        }
    }
}

/// A sensor's declared timing characteristics, recorded so a delay model can be
/// fitted later (SPECS.md Section 11.3, system identification).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SensorTiming {
    /// The sensor.
    pub sensor: SensorId,
    /// Nominal sampling rate.
    pub rate: crate::units::Hertz,
    /// Whether the device supplies a sampling instant at all.
    pub provides_sampling_instant: bool,
    /// Bound on timestamp uncertainty.
    pub timestamp_uncertainty: Duration,
}

/// A recorded episode on a robot or a simulator.
#[derive(Clone, PartialEq, Debug)]
pub struct EpisodeRecord {
    /// Schema version.
    pub schema_version: SchemaVersion,
    /// Stable episode identifier.
    pub episode_id: String,
    /// Robot profile identity and version this episode ran against.
    pub robot_profile: String,
    /// Operating mode and command source, recorded per SPECS.md Section 4.
    pub mode: ModeContext,
    /// Clock the tick timestamps are on.
    pub clock: ClockId,
    /// Mapping from the episode clock to wall time, where correlation with an
    /// external system is needed. Recorded separately, per Appendix A.3.
    pub wall_clock_offset: Option<Duration>,
    /// Seed stream, for reproduction.
    pub seeds: SeedStream,
    /// Configuration identity used for the run.
    pub config_hash: String,
    /// Bundle identity, where a learned policy was active.
    pub bundle_id: Option<String>,
    /// Sensor timing declarations.
    pub sensors: Vec<SensorTiming>,
    /// The ticks.
    pub ticks: Vec<EpisodeTick>,
    /// Why the episode ended.
    pub completion: CompletionReason,
}

/// Why an episode record was rejected.
#[derive(Clone, PartialEq, Debug)]
pub enum EpisodeError {
    /// Two vectors that must agree in width did not.
    WidthMismatch {
        /// Which field.
        field: &'static str,
        /// Width expected.
        expected: usize,
        /// Width found.
        found: usize,
    },
    /// Tick indices were not consecutive from zero.
    NonConsecutiveTicks {
        /// Position in the tick list.
        position: usize,
        /// Index found there.
        found: u64,
    },
    /// A tick carried a timestamp from a different clock.
    ClockMismatch {
        /// Tick index.
        tick: u64,
    },
    /// A tick's arrival time preceded its sampling instant.
    ArrivalBeforeSampling {
        /// Tick index.
        tick: u64,
    },
    /// Sensor times were not non-decreasing.
    NonMonotonicSensorTime {
        /// Tick index.
        tick: u64,
    },
    /// A command was emitted in a mode that forbids actuation.
    CommandInNonActuatingMode {
        /// The mode.
        mode: crate::mode::OperatingMode,
        /// Tick index.
        tick: u64,
    },
    /// A confidence was outside `[0, 1]`.
    InvalidConfidence {
        /// Tick index.
        tick: u64,
        /// Value found.
        value: f64,
    },
    /// The record held no ticks.
    Empty,
}

impl fmt::Display for EpisodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WidthMismatch {
                field,
                expected,
                found,
            } => {
                write!(f, "{field}: expected width {expected}, found {found}")
            }
            Self::NonConsecutiveTicks { position, found } => {
                write!(f, "tick at position {position} has index {found}")
            }
            Self::ClockMismatch { tick } => write!(f, "tick {tick} is on a different clock"),
            Self::ArrivalBeforeSampling { tick } => {
                write!(f, "tick {tick} arrived before it was sampled")
            }
            Self::NonMonotonicSensorTime { tick } => {
                write!(f, "tick {tick} has a sensor time before its predecessor")
            }
            Self::CommandInNonActuatingMode { mode, tick } => {
                write!(
                    f,
                    "tick {tick} carries a command in mode {mode}, which cannot actuate"
                )
            }
            Self::InvalidConfidence { tick, value } => {
                write!(f, "tick {tick} has confidence {value}, outside [0,1]")
            }
            Self::Empty => write!(f, "episode has no ticks"),
        }
    }
}

impl core::error::Error for EpisodeError {}

impl EpisodeRecord {
    /// Checks the record's internal consistency.
    ///
    /// A command *record* is permitted in `hw-shadow` because that mode computes
    /// commands and discards them at the adapter: the record is the evidence
    /// that it did. It is not permitted in `offline-replay`, which has no device
    /// at all.
    pub fn validate(&self) -> Result<(), EpisodeError> {
        if self.ticks.is_empty() {
            return Err(EpisodeError::Empty);
        }
        let mut previous: Option<Timestamp> = None;
        for (position, tick) in self.ticks.iter().enumerate() {
            if tick.index != position as u64 {
                return Err(EpisodeError::NonConsecutiveTicks {
                    position,
                    found: tick.index,
                });
            }
            if tick.sensor_time.clock() != self.clock || tick.arrival_time.clock() != self.clock {
                return Err(EpisodeError::ClockMismatch { tick: tick.index });
            }
            if tick.arrival_time.nanos() < tick.sensor_time.nanos() {
                return Err(EpisodeError::ArrivalBeforeSampling { tick: tick.index });
            }
            if let Some(prev) = previous {
                if tick.sensor_time.nanos() < prev.nanos() {
                    return Err(EpisodeError::NonMonotonicSensorTime { tick: tick.index });
                }
            }
            previous = Some(tick.sensor_time);
            if tick.command.is_some()
                && self.mode.mode.authority() == crate::mode::ActuationAuthority::None
            {
                return Err(EpisodeError::CommandInNonActuatingMode {
                    mode: self.mode.mode,
                    tick: tick.index,
                });
            }
            for c in &tick.contacts {
                if !(0.0..=1.0).contains(&c.confidence) || !c.confidence.is_finite() {
                    return Err(EpisodeError::InvalidConfidence {
                        tick: tick.index,
                        value: c.confidence,
                    });
                }
            }
            if let Some(est) = &tick.estimate {
                if est.values.len() != est.variance.len() {
                    return Err(EpisodeError::WidthMismatch {
                        field: "estimate",
                        expected: est.values.len(),
                        found: est.variance.len(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Number of ticks.
    #[must_use]
    pub fn tick_count(&self) -> usize {
        self.ticks.len()
    }

    /// Ticks on which the supervisor intervened.
    #[must_use]
    pub fn intervention_count(&self) -> usize {
        self.ticks
            .iter()
            .filter(|t| !t.interventions.is_empty())
            .count()
    }

    /// Ticks whose applied command differed from the raw command.
    #[must_use]
    pub fn clamped_command_count(&self) -> usize {
        self.ticks
            .iter()
            .filter(|t| t.command.as_ref().is_some_and(CommandRecord::was_modified))
            .count()
    }

    /// Whether every torque sample in the record is a genuine measurement.
    ///
    /// System identification calls this before fitting, because fitting to
    /// nominal current-derived torque produces a parameter estimate that
    /// describes the datasheet rather than the robot.
    #[must_use]
    pub fn torque_is_measured(&self) -> bool {
        self.ticks.iter().all(|t| {
            t.joints
                .iter()
                .all(|j| !j.torque.valid || j.torque.provenance.is_measurement())
        })
    }

    /// Sensor-to-command ages across the episode, for the timing report.
    #[must_use]
    pub fn sensor_to_command_ages(&self) -> Vec<Duration> {
        self.ticks
            .iter()
            .filter_map(|t| t.sensor_to_command_age().map(|(age, _)| age))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mode::{ControlSource, OperatingMode};
    use crate::units::Hertz;

    fn tick(index: u64, sensor_ns: i64, with_command: bool) -> EpisodeTick {
        let clock = ClockId::HostMonotonic;
        EpisodeTick {
            index,
            sensor_time: Timestamp::sampling(sensor_ns, clock),
            arrival_time: Timestamp::arrival(
                sensor_ns + 500_000,
                clock,
                Duration::from_micros(100),
            ),
            joints: vec![JointSample {
                joint: JointId::new(0),
                position: Radians(0.1),
                velocity: RadiansPerSecond(0.0),
                torque: TorqueSample {
                    value: NewtonMetres(1.0),
                    provenance: TorqueProvenance::MeasuredAtJoint,
                    valid: true,
                },
                valid: true,
            }],
            actuators: Vec::new(),
            contacts: Vec::new(),
            estimate: None,
            desired: None,
            command: with_command.then(|| {
                CommandRecord::new(
                    vec![1.5],
                    vec![1.0],
                    "N*m",
                    Timestamp::sampling(sensor_ns + 2_000_000, clock),
                )
                .unwrap()
            }),
            active_policy: None,
            transition: None,
            interventions: Vec::new(),
        }
    }

    fn episode(mode: OperatingMode, ticks: Vec<EpisodeTick>) -> EpisodeRecord {
        EpisodeRecord {
            schema_version: EPISODE_SCHEMA_VERSION,
            episode_id: "ep-1".to_owned(),
            robot_profile: "fixture-arm/1.0".to_owned(),
            mode: ModeContext {
                mode,
                source: ControlSource::Policy,
            },
            clock: ClockId::HostMonotonic,
            wall_clock_offset: None,
            seeds: SeedStream::new(1, 0),
            config_hash: "cfg".to_owned(),
            bundle_id: None,
            sensors: vec![SensorTiming {
                sensor: SensorId::new(0),
                rate: Hertz(1000.0),
                provides_sampling_instant: true,
                timestamp_uncertainty: Duration::ZERO,
            }],
            ticks,
            completion: CompletionReason::Truncated,
        }
    }

    #[test]
    fn raw_and_applied_commands_are_both_retained() {
        let e = episode(OperatingMode::HwSupported, vec![tick(0, 0, true)]);
        e.validate().unwrap();
        let cmd = e.ticks[0].command.as_ref().unwrap();
        assert_eq!(cmd.raw, vec![1.5]);
        assert_eq!(cmd.applied, vec![1.0]);
        assert!(cmd.was_modified());
        assert!((cmd.max_modification() - 0.5).abs() < 1e-12);
        assert_eq!(e.clamped_command_count(), 1);
    }

    #[test]
    fn command_widths_must_agree() {
        let err = CommandRecord::new(
            vec![1.0, 2.0],
            vec![1.0],
            "N*m",
            Timestamp::sampling(0, ClockId::HostMonotonic),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EpisodeError::WidthMismatch {
                field: "command",
                ..
            }
        ));
    }

    #[test]
    fn a_command_in_offline_replay_is_rejected() {
        let e = episode(OperatingMode::OfflineReplay, vec![tick(0, 0, true)]);
        assert!(matches!(
            e.validate(),
            Err(EpisodeError::CommandInNonActuatingMode { .. })
        ));
    }

    #[test]
    fn shadow_mode_may_record_a_computed_command() {
        let e = episode(OperatingMode::HwShadow, vec![tick(0, 0, true)]);
        assert!(
            e.validate().is_ok(),
            "shadow mode computes and discards; the record is the proof"
        );
    }

    #[test]
    fn arrival_before_sampling_is_rejected() {
        let mut t = tick(0, 1_000_000, false);
        t.arrival_time = Timestamp::sampling(0, ClockId::HostMonotonic);
        let e = episode(OperatingMode::SimTrain, vec![t]);
        assert!(matches!(
            e.validate(),
            Err(EpisodeError::ArrivalBeforeSampling { .. })
        ));
    }

    #[test]
    fn non_consecutive_tick_indices_are_rejected() {
        let e = episode(
            OperatingMode::SimTrain,
            vec![tick(0, 0, false), tick(2, 1_000, false)],
        );
        assert!(matches!(
            e.validate(),
            Err(EpisodeError::NonConsecutiveTicks {
                position: 1,
                found: 2
            })
        ));
    }

    #[test]
    fn backwards_sensor_time_is_rejected() {
        let e = episode(
            OperatingMode::SimTrain,
            vec![tick(0, 1_000_000, false), tick(1, 0, false)],
        );
        assert!(matches!(
            e.validate(),
            Err(EpisodeError::NonMonotonicSensorTime { .. })
        ));
    }

    #[test]
    fn nominal_current_torque_is_not_a_measurement() {
        assert!(TorqueProvenance::MeasuredAtJoint.is_measurement());
        assert!(TorqueProvenance::CurrentWithCalibration.is_measurement());
        assert!(!TorqueProvenance::CurrentNominal.is_measurement());
        assert!(!TorqueProvenance::CommandedEcho.is_measurement());

        let mut t = tick(0, 0, false);
        t.joints[0].torque.provenance = TorqueProvenance::CurrentNominal;
        let e = episode(OperatingMode::HwSupported, vec![t]);
        assert!(!e.torque_is_measured());
        assert!(episode(OperatingMode::HwSupported, vec![tick(0, 0, false)]).torque_is_measured());
    }

    #[test]
    fn an_invalid_torque_sample_does_not_fail_the_provenance_check() {
        let mut t = tick(0, 0, false);
        t.joints[0].torque.provenance = TorqueProvenance::CommandedEcho;
        t.joints[0].torque.valid = false;
        let e = episode(OperatingMode::HwSupported, vec![t]);
        assert!(
            e.torque_is_measured(),
            "an invalid sample is not used, so it is not a source"
        );
    }

    #[test]
    fn timing_quantities_are_separate() {
        let e = episode(OperatingMode::HwSupported, vec![tick(0, 1_000_000, true)]);
        let (age, uncertainty) = e.ticks[0].sensor_to_command_age().unwrap();
        assert_eq!(age, Duration::from_millis(2));
        assert_eq!(uncertainty, Duration::ZERO);
        assert_eq!(
            e.ticks[0].transport_delay(),
            Some(Duration::from_micros(500))
        );
        assert_eq!(e.sensor_to_command_ages(), vec![Duration::from_millis(2)]);
    }

    #[test]
    fn truncation_is_not_terminal() {
        assert!(!CompletionReason::Truncated.is_terminal());
        assert!(CompletionReason::TaskSucceeded.is_terminal());
        assert!(CompletionReason::Terminated {
            condition: "fell".to_owned()
        }
        .is_terminal());
        assert_eq!(CompletionReason::Truncated.id(), "truncated");
    }

    #[test]
    fn interventions_are_counted() {
        let mut t = tick(0, 0, true);
        t.interventions.push(SupervisorIntervention {
            at: Timestamp::sampling(0, ClockId::HostMonotonic),
            check: "torque-limit".to_owned(),
            component: Some(0),
            excess: Some(0.5),
            action: "clamped".to_owned(),
        });
        let e = episode(OperatingMode::HwSupported, vec![t]);
        assert_eq!(e.intervention_count(), 1);
        assert_eq!(e.tick_count(), 1);
    }

    #[test]
    fn an_estimate_with_mismatched_covariance_is_rejected() {
        let mut t = tick(0, 0, false);
        t.estimate = Some(EstimateSample {
            values: vec![1.0, 2.0],
            variance: vec![0.1],
            valid: true,
        });
        let e = episode(OperatingMode::SimTrain, vec![t]);
        assert!(matches!(
            e.validate(),
            Err(EpisodeError::WidthMismatch {
                field: "estimate",
                ..
            })
        ));
    }

    #[test]
    fn an_empty_episode_is_rejected() {
        let e = episode(OperatingMode::SimTrain, Vec::new());
        assert!(matches!(e.validate(), Err(EpisodeError::Empty)));
    }
}

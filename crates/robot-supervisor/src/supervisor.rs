//! The supervisor: the last stage before the actuator interface.
//!
//! SPECS.md Section 12.1: "The supervisor MUST be the last stage before the
//! actuator interface, MUST be reachable on every path including
//! teleoperation and scripted control, and MUST NOT import the policy,
//! retrieval or personality crates." This crate's `Cargo.toml` has no
//! dependency edge to `robot-learn`, `robot-skills` or `robot-persona` —
//! making the second half of that requirement a build-time fact, per
//! `docs/decisions/0004-nn-and-supervisor-crate-extraction.md`.
//!
//! [`Supervisor::check_tick`] performs every check in one allocation-free
//! pass: it is constructed once from a validated `robot_model::RobotProfile`
//! (which does the heap work), and every subsequent call writes into
//! caller-owned buffers and a fixed-capacity [`crate::fault::FaultLog`].

use crate::fault::{Fault, FaultLog};
use crate::policy::{FaultPolicy, Response};
use robot_model::profile::RobotProfile;
use robot_types::ids::{ActuatorId, ContactSiteId, JointId};
use robot_types::mode::{ActuationAuthority, OperatingMode};
use robot_types::time::Duration;
use robot_types::units::{Celsius, NewtonMetres, Radians, RadiansPerSecond, Range};

/// Which physical quantity the command vector represents.
///
/// Mirrors the subset of `robot_types::obs::ActionSemantics` the supervisor
/// knows how to bound; a bundle using a semantics outside this set (a latent
/// action, a position residual) is not interpretable here without the
/// reference the controller holds, and the supervisor correctly does not
/// attempt it — those semantics still pass through `NonFiniteCommand`
/// checking, just not the physical-limit checks.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CommandKind {
    /// Each component is a target joint position.
    Position,
    /// Each component is a target joint velocity.
    Velocity,
    /// Each component is a joint torque.
    Torque,
    /// A kind the supervisor does not bound physically (only finiteness is
    /// checked).
    Unbounded,
}

/// What the supervisor observed on one tick.
#[derive(Clone, Copy, Debug)]
pub struct TickInput<'a> {
    /// Current operating mode.
    pub mode: OperatingMode,
    /// Age of the observation this command was computed from.
    pub observation_age: Duration,
    /// Age of the last confirmed communications heartbeat.
    pub comms_age: Duration,
    /// Measured actuator temperatures, in profile actuator order.
    pub temperatures: &'a [Celsius],
    /// Sites currently detected in contact.
    pub active_contacts: &'a [ContactSiteId],
    /// Sites with contact permission this tick.
    pub permitted_contacts: &'a [ContactSiteId],
}

/// Why a supervisor could not be constructed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SupervisorError {
    /// The command width did not match the profile's actuated joint count.
    WidthMismatch {
        /// Expected width.
        expected: usize,
        /// Width configured.
        found: usize,
    },
}

impl core::fmt::Display for SupervisorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::WidthMismatch { expected, found } => {
                write!(
                    f,
                    "configured width {found}, profile has {expected} actuated joints"
                )
            }
        }
    }
}

impl core::error::Error for SupervisorError {}

/// Freshness limits applied to inputs.
#[derive(Clone, Copy, Debug)]
pub struct FreshnessLimits {
    /// Maximum age of the observation a command may be computed from.
    pub observation: Duration,
    /// Maximum age of the last confirmed communications heartbeat.
    pub comms: Duration,
}

/// The supervisor.
///
/// `N` bounds the fault log's capacity; choose it comfortably above
/// `3 * joint_count + actuator_count` so a worst-case tick (every joint over
/// every limit, every actuator hot) still records without overflowing.
pub struct Supervisor<const N: usize> {
    joint_ids: Vec<JointId>,
    position_limits: Vec<Option<Range<Radians>>>,
    velocity_limits: Vec<RadiansPerSecond>,
    torque_limits: Vec<NewtonMetres>,
    actuator_ids: Vec<ActuatorId>,
    thermal_limits: Vec<Celsius>,
    policy: FaultPolicy,
    freshness: FreshnessLimits,
    command_kind: CommandKind,
    last_safe_command: Vec<f64>,
    log: FaultLog<N>,
}

impl<const N: usize> Supervisor<N> {
    /// Builds a supervisor from a validated profile.
    ///
    /// Every allocation happens here, before activation, per SPECS.md
    /// Section 5: "allocate working memory before activation."
    pub fn new(
        profile: &RobotProfile,
        policy: FaultPolicy,
        freshness: FreshnessLimits,
        command_kind: CommandKind,
    ) -> Self {
        let joints: Vec<_> = profile.actuated_joints().collect();
        let joint_ids: Vec<_> = joints.iter().map(|j| j.id).collect();
        let position_limits = joints.iter().map(|j| j.limits.position).collect();
        let velocity_limits = joints.iter().map(|j| j.limits.velocity).collect();
        let torque_limits = joints.iter().map(|j| j.limits.effort).collect();
        let actuator_ids = profile.actuators.iter().map(|a| a.id).collect();
        let thermal_limits = profile.actuators.iter().map(|a| a.thermal.limit).collect();
        let width = joint_ids.len();
        Self {
            joint_ids,
            position_limits,
            velocity_limits,
            torque_limits,
            actuator_ids,
            thermal_limits,
            policy,
            freshness,
            command_kind,
            last_safe_command: vec![0.0; width],
            log: FaultLog::new(),
        }
    }

    /// Number of actuated joints this supervisor was built for.
    #[must_use]
    pub fn width(&self) -> usize {
        self.joint_ids.len()
    }

    /// The faults recorded on the most recent call to
    /// [`Supervisor::check_tick`].
    #[must_use]
    pub fn log(&self) -> &FaultLog<N> {
        &self.log
    }

    /// The command currently held as "last known safe", for inspection or
    /// for seeding a replay.
    #[must_use]
    pub fn last_safe_command(&self) -> &[f64] {
        &self.last_safe_command
    }

    /// Checks one tick's command against every rule and writes the command
    /// to actually apply into `out`.
    ///
    /// Allocates nothing. `raw` and `out` must both have length
    /// [`Supervisor::width`], and the fault log from the previous tick is
    /// cleared before this one is recorded.
    pub fn check_tick(
        &mut self,
        input: &TickInput<'_>,
        raw: &[f64],
        out: &mut [f64],
    ) -> Result<Response, SupervisorError> {
        if raw.len() != self.width() || out.len() != self.width() {
            return Err(SupervisorError::WidthMismatch {
                expected: self.width(),
                found: raw.len(),
            });
        }

        self.log.clear();

        if input.observation_age > self.freshness.observation {
            self.log.push(Fault::StaleObservation {
                age: input.observation_age,
                limit: self.freshness.observation,
            });
        }
        if input.comms_age > self.freshness.comms {
            self.log.push(Fault::CommunicationsLost {
                age: input.comms_age,
                limit: self.freshness.comms,
            });
        }
        if input.mode.authority() == ActuationAuthority::None {
            self.log
                .push(Fault::ActuationNotPermitted { mode: input.mode });
        }

        for (i, &value) in raw.iter().enumerate() {
            if !value.is_finite() {
                self.log.push(Fault::NonFiniteCommand {
                    joint: self.joint_ids[i],
                });
                continue;
            }
            match self.command_kind {
                CommandKind::Position => {
                    if let Some(limit) = self.position_limits[i] {
                        if !limit.contains(Radians(value)) {
                            self.log.push(Fault::PositionLimitExceeded {
                                joint: self.joint_ids[i],
                                commanded: Radians(value),
                                limit,
                            });
                        }
                    }
                }
                CommandKind::Velocity => {
                    let limit = self.velocity_limits[i];
                    if value.abs() > limit.value() {
                        self.log.push(Fault::VelocityLimitExceeded {
                            joint: self.joint_ids[i],
                            commanded: RadiansPerSecond(value),
                            limit,
                        });
                    }
                }
                CommandKind::Torque => {
                    let limit = self.torque_limits[i];
                    if value.abs() > limit.value() {
                        self.log.push(Fault::TorqueLimitExceeded {
                            joint: self.joint_ids[i],
                            commanded: NewtonMetres(value),
                            limit,
                        });
                    }
                }
                CommandKind::Unbounded => {}
            }
        }

        for (i, &temp) in input.temperatures.iter().enumerate() {
            if i < self.thermal_limits.len() && temp.value() >= self.thermal_limits[i].value() {
                self.log.push(Fault::OverTemperature {
                    actuator: self.actuator_ids[i],
                    measured: temp,
                    limit: self.thermal_limits[i],
                });
            }
        }

        for &site in input.active_contacts {
            if !input.permitted_contacts.contains(&site) {
                self.log.push(Fault::UnauthorizedContact { site });
            }
        }

        let response = self.most_severe_response();
        self.apply(response, raw, out);
        Ok(response)
    }

    /// The most severe declared response among this tick's recorded faults,
    /// or [`Response::ClampAndContinue`] when there are none (the vacuous
    /// case: nothing to clamp, the command passes through).
    fn most_severe_response(&self) -> Response {
        fn severity(r: Response) -> u8 {
            match r {
                Response::ClampAndContinue => 0,
                Response::HoldLastSafeCommand => 1,
                Response::RejectCommand => 2,
                Response::EmergencyStop => 3,
            }
        }
        self.log
            .entries()
            .iter()
            .flatten()
            .map(|f| self.policy.response_for(f.category()))
            .max_by_key(|r| severity(*r))
            .unwrap_or(Response::ClampAndContinue)
    }

    fn apply(&mut self, response: Response, raw: &[f64], out: &mut [f64]) {
        match response {
            Response::ClampAndContinue => {
                for (i, (&value, o)) in raw.iter().zip(out.iter_mut()).enumerate() {
                    *o = self.clamp_one(i, value);
                }
                self.last_safe_command.copy_from_slice(out);
            }
            Response::HoldLastSafeCommand | Response::RejectCommand => {
                out.copy_from_slice(&self.last_safe_command);
            }
            Response::EmergencyStop => {
                for o in out.iter_mut() {
                    *o = 0.0;
                }
                self.last_safe_command.copy_from_slice(out);
            }
        }
    }

    fn clamp_one(&self, index: usize, value: f64) -> f64 {
        if !value.is_finite() {
            return 0.0;
        }
        match self.command_kind {
            CommandKind::Position => match self.position_limits[index] {
                Some(limit) => limit.clamp(Radians(value)).value(),
                None => value,
            },
            CommandKind::Velocity => value.clamp(
                -self.velocity_limits[index].value(),
                self.velocity_limits[index].value(),
            ),
            CommandKind::Torque => value.clamp(
                -self.torque_limits[index].value(),
                self.torque_limits[index].value(),
            ),
            CommandKind::Unbounded => value,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_model::actuator::{
        ActuatorSpec, PdGains, ThermalLimits, TorqueSpeedEnvelope, Transmission,
    };
    use robot_model::joint::{JointKind, JointLimits, JointSpec};
    use robot_model::link::{LinkSpec, MassProperties};
    use robot_model::pose::RigidPose;
    use robot_model::profile::{FirmwareContract, PROFILE_SCHEMA_VERSION};
    use robot_types::frames::{FrameId, GravityConvention, Vec3};
    use robot_types::ids::{LinkId, Name};
    use robot_types::mode::OperatingMode;
    use robot_types::schema::ExtensionTable;
    use robot_types::units::{Amperes, Kilograms};

    fn profile() -> RobotProfile {
        let mut mp = MassProperties::MASSLESS;
        mp.mass = Kilograms(1.0);
        let link0 = LinkSpec {
            id: LinkId::new(0),
            name: Name::new("base").unwrap(),
            mass: mp,
            collision: Vec::new(),
        };
        let link1 = LinkSpec {
            id: LinkId::new(1),
            name: Name::new("arm").unwrap(),
            mass: mp,
            collision: Vec::new(),
        };
        let joint = JointSpec {
            id: JointId::new(0),
            name: Name::new("elbow").unwrap(),
            kind: JointKind::Revolute,
            parent: LinkId::new(0),
            child: LinkId::new(1),
            origin: RigidPose::IDENTITY,
            axis: Vec3::unit_z(),
            limits: JointLimits::bounded(
                Range::<Radians>::new(Radians(-1.0), Radians(1.0)).unwrap(),
                RadiansPerSecond(2.0),
                NewtonMetres(5.0),
            ),
            sign: 1.0,
            zero_offset: 0.0,
        };
        let actuator = ActuatorSpec {
            id: ActuatorId::new(0),
            name: Name::new("elbow-motor").unwrap(),
            joint: JointId::new(0),
            transmission: Transmission::DIRECT,
            envelope: TorqueSpeedEnvelope {
                stall_torque: NewtonMetres(5.0),
                corner_speed: RadiansPerSecond(1.0),
                no_load_speed: RadiansPerSecond(2.0),
            },
            gains: Some(PdGains { kp: 1.0, kd: 0.1 }),
            command_latency: Duration::from_millis(1),
            thermal: ThermalLimits {
                limit: Celsius(80.0),
                ambient: Celsius(25.0),
                time_constant: Duration::from_secs(10),
                current_at_limit: Amperes(5.0),
            },
            current_uncertainty: Amperes(0.01),
        };
        RobotProfile {
            schema_version: PROFILE_SCHEMA_VERSION,
            name: Name::new("fixture-arm").unwrap(),
            profile_version: "1.0".to_owned(),
            joints: vec![joint],
            links: vec![link0, link1],
            actuators: vec![actuator],
            sensors: Vec::new(),
            root_link: LinkId::new(0),
            gravity: GravityConvention::minus_z_standard(FrameId::WORLD),
            firmware: FirmwareContract {
                version: "1.0".to_owned(),
                command_expiration: Duration::from_millis(100),
                watchdog_timeout: Duration::from_millis(200),
                has_independent_estop: true,
                allowed_modes: vec![OperatingMode::HwSupported],
            },
            extensions: ExtensionTable::new(),
        }
    }

    fn nominal_input() -> TickInput<'static> {
        TickInput {
            mode: OperatingMode::HwSupported,
            observation_age: Duration::from_millis(1),
            comms_age: Duration::from_millis(1),
            temperatures: &[Celsius(40.0)],
            active_contacts: &[],
            permitted_contacts: &[],
        }
    }

    fn supervisor() -> Supervisor<16> {
        Supervisor::new(
            &profile(),
            FaultPolicy::conservative_default(),
            FreshnessLimits {
                observation: Duration::from_millis(100),
                comms: Duration::from_millis(200),
            },
            CommandKind::Position,
        )
    }

    #[test]
    fn a_nominal_command_passes_through_unclamped() {
        let mut sup = supervisor();
        let mut out = [0.0];
        let response = sup.check_tick(&nominal_input(), &[0.5], &mut out).unwrap();
        assert_eq!(response, Response::ClampAndContinue);
        assert!((out[0] - 0.5).abs() < 1e-12);
        assert!(sup.log().is_empty());
    }

    #[test]
    fn a_position_beyond_the_limit_is_clamped_and_logged() {
        let mut sup = supervisor();
        let mut out = [0.0];
        let response = sup.check_tick(&nominal_input(), &[5.0], &mut out).unwrap();
        assert_eq!(response, Response::ClampAndContinue);
        assert!(
            (out[0] - 1.0).abs() < 1e-12,
            "should clamp to the upper limit"
        );
        assert_eq!(sup.log().len(), 1);
    }

    #[test]
    fn a_stale_observation_escalates_to_emergency_stop_and_zeros_the_command() {
        let mut sup = supervisor();
        let mut input = nominal_input();
        input.observation_age = Duration::from_millis(500);
        let mut out = [0.0];
        let response = sup.check_tick(&input, &[0.3], &mut out).unwrap();
        assert_eq!(response, Response::EmergencyStop);
        assert_eq!(out[0], 0.0);
    }

    #[test]
    fn a_nonfinite_command_escalates_and_never_reaches_the_actuator() {
        let mut sup = supervisor();
        let mut out = [0.0];
        let response = sup
            .check_tick(&nominal_input(), &[f64::NAN], &mut out)
            .unwrap();
        assert_eq!(response, Response::EmergencyStop);
        assert_eq!(out[0], 0.0);
    }

    #[test]
    fn after_an_emergency_stop_the_held_safe_command_is_the_stop_itself() {
        let mut sup = supervisor();
        let mut out = [0.0];
        // Force an e-stop via stale observation.
        let mut bad = nominal_input();
        bad.observation_age = Duration::from_millis(999);
        sup.check_tick(&bad, &[0.3], &mut out).unwrap();
        assert_eq!(sup.last_safe_command(), &[0.0]);

        // A subsequent over-temperature fault (configured to Hold) must hold
        // the post-e-stop safe command, not whatever was commanded earlier.
        let mut hot = nominal_input();
        hot.temperatures = &[Celsius(90.0)];
        let response = sup.check_tick(&hot, &[0.9], &mut out).unwrap();
        assert_eq!(response, Response::HoldLastSafeCommand);
        assert_eq!(out[0], 0.0);
    }

    #[test]
    fn an_unauthorized_contact_is_detected() {
        let mut sup = supervisor();
        let mut input = nominal_input();
        let site = ContactSiteId::new(3);
        input.active_contacts = core::slice::from_ref(&site);
        let mut out = [0.0];
        let response = sup.check_tick(&input, &[0.1], &mut out).unwrap();
        assert_eq!(response, Response::HoldLastSafeCommand);
        assert_eq!(sup.log().len(), 1);
    }

    #[test]
    fn actuation_in_a_non_actuating_mode_is_flagged() {
        let mut sup = supervisor();
        let mut input = nominal_input();
        input.mode = OperatingMode::OfflineReplay;
        let mut out = [0.0];
        let response = sup.check_tick(&input, &[0.1], &mut out).unwrap();
        assert_eq!(response, Response::EmergencyStop);
    }

    #[test]
    fn width_mismatch_is_rejected_rather_than_panicking() {
        let mut sup = supervisor();
        let mut out = [0.0, 0.0];
        assert!(matches!(
            sup.check_tick(&nominal_input(), &[0.1, 0.2], &mut out),
            Err(SupervisorError::WidthMismatch { .. })
        ));
    }

    #[test]
    fn the_fault_log_overflow_counter_is_exposed_when_capacity_is_tiny() {
        let mut sup: Supervisor<1> = Supervisor::new(
            &profile(),
            FaultPolicy::conservative_default(),
            FreshnessLimits {
                observation: Duration::from_millis(1),
                comms: Duration::from_millis(1),
            },
            CommandKind::Position,
        );
        let mut bad = nominal_input();
        bad.observation_age = Duration::from_millis(999);
        bad.comms_age = Duration::from_millis(999);
        let mut out = [0.0];
        sup.check_tick(&bad, &[0.1], &mut out).unwrap();
        assert_eq!(
            sup.log().overflowed(),
            1,
            "two faults into a one-slot log must overflow once"
        );
    }
}

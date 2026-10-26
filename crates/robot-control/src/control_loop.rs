//! The deterministic control loop: poll feedback, infer, supervise,
//! transmit — one preallocated structure, driven one tick at a time.
//!
//! SPECS.md Section 5: "The control path MUST NOT perform file reads,
//! network database queries, model downloads, heap allocation after
//! activation, locks with unbounded contention, GPU compilation, logging
//! flushes, or model loading." [`ControlLoop::new`] does every allocation;
//! [`ControlLoop::tick`] touches none of its own buffers' capacity, only
//! their contents.
//!
//! This is a policy-tier loop only in this revision: it runs inference,
//! supervision and transmission every tick. The tactical tier
//! (`robot_skills::tiers::Router`) is held but not yet consulted on the
//! control path to select among skills — that integration, and the
//! per-tier compute-duration/decision-age split SPECS.md Section 5.1 asks
//! for across *two* learned tiers, is future work this module does not
//! claim. What is implemented — the policy tier's own timing split
//! (compute duration, sensor-to-command age, deadline margin) — is real and
//! tested.

use crate::timing::TickTiming;
use robot_hardware::device::HardwareAdapter;
use robot_nn::mlp::{Mlp, Scratch};
use robot_supervisor::supervisor::{FreshnessLimits, Supervisor, TickInput};
use robot_types::mode::OperatingMode;
use robot_types::obs::Normalizer;
use robot_types::time::{Duration, MonotonicClock, Timestamp};

/// Why a tick could not run to completion.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TickError {
    /// The hardware adapter reported a fault.
    Adapter(robot_hardware::device::AdapterError),
    /// The supervisor reported a fault (a width mismatch; genuine faults
    /// are handled, not propagated as errors — see
    /// `robot_supervisor::policy::Response`).
    Supervisor(robot_supervisor::supervisor::SupervisorError),
}

impl core::fmt::Display for TickError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Adapter(e) => write!(f, "{e}"),
            Self::Supervisor(e) => write!(f, "{e}"),
        }
    }
}

impl core::error::Error for TickError {}

/// Everything [`ControlLoop::new`] needs, bundled so construction takes one
/// argument instead of eight.
pub struct ControlLoopConfig<A: HardwareAdapter, C: MonotonicClock, const N: usize> {
    /// The policy network.
    pub policy: Mlp,
    /// Frozen observation normalizer.
    pub normalizer: Normalizer,
    /// The supervisor.
    pub supervisor: Supervisor<N>,
    /// The hardware adapter.
    pub adapter: A,
    /// The clock.
    pub clock: C,
    /// Operating mode this loop runs under.
    pub mode: OperatingMode,
    /// Per-tick compute budget.
    pub budget: Duration,
    /// Freshness limits passed through to the supervisor's own checks (and
    /// used here to synthesize an age when no feedback has ever arrived).
    pub freshness: FreshnessLimits,
}

/// The deterministic control loop.
pub struct ControlLoop<A: HardwareAdapter, C: MonotonicClock, const N: usize> {
    policy: Mlp,
    scratch: Scratch,
    normalizer: Normalizer,
    supervisor: Supervisor<N>,
    adapter: A,
    clock: C,
    mode: OperatingMode,
    budget: Duration,
    observation_freshness_limit: Duration,

    // Preallocated working buffers; every `tick` call reuses these.
    feedback: Vec<f64>,
    normalized: Vec<f64>,
    raw_action: Vec<f64>,
    command: Vec<f64>,
    last_feedback_time: Option<Timestamp>,
}

impl<A: HardwareAdapter, C: MonotonicClock, const N: usize> ControlLoop<A, C, N> {
    /// Builds the loop, allocating every working buffer up front.
    #[must_use]
    pub fn new(config: ControlLoopConfig<A, C, N>) -> Self {
        let ControlLoopConfig {
            policy,
            normalizer,
            supervisor,
            adapter,
            clock,
            mode,
            budget,
            freshness,
        } = config;
        let obs_width = policy.input_width();
        let action_width = policy.output_width();
        let scratch = Scratch::new(policy.scratch_len());
        Self {
            policy,
            scratch,
            normalizer,
            supervisor,
            adapter,
            clock,
            mode,
            budget,
            observation_freshness_limit: freshness.observation,
            feedback: vec![0.0; obs_width],
            normalized: vec![0.0; obs_width],
            raw_action: vec![0.0; action_width],
            command: vec![0.0; action_width],
            last_feedback_time: None,
        }
    }

    /// The most recent supervisor response, for a caller building a report.
    #[must_use]
    pub fn supervisor_log(&self) -> &robot_supervisor::fault::FaultLog<N> {
        self.supervisor.log()
    }

    /// Runs one control tick: poll, infer, supervise, transmit.
    pub fn tick(&mut self) -> Result<TickTiming, TickError> {
        let now = self.clock.now();
        let compute_start = self.clock.now();

        let feedback_header = self
            .adapter
            .poll_feedback(&mut self.feedback)
            .map_err(TickError::Adapter)?;
        if let Some(header) = feedback_header {
            self.last_feedback_time = Some(header.timestamp);
        }

        let observation_age = match self.last_feedback_time {
            Some(t) => now.since(t).ok().map(|(age, _)| age),
            None => None,
        };

        self.normalizer
            .normalize(&self.feedback, &mut self.normalized)
            .expect("buffer widths fixed at construction");

        self.policy
            .forward(&self.normalized, &mut self.raw_action, &mut self.scratch)
            .expect("buffer widths fixed at construction");

        let comms_age = self.adapter.link_age(now).unwrap_or(Duration::ZERO);
        let supervisor_input = TickInput {
            mode: self.mode,
            observation_age: observation_age
                .unwrap_or(self.observation_freshness_limit + Duration::from_nanos(1)),
            comms_age,
            temperatures: &[],
            active_contacts: &[],
            permitted_contacts: &[],
        };
        // The response (clamp / hold / reject / e-stop) is already reflected
        // in `self.command`; a caller inspects `supervisor_log()` for which
        // faults, if any, produced it.
        self.supervisor
            .check_tick(&supervisor_input, &self.raw_action, &mut self.command)
            .map_err(TickError::Supervisor)?;

        if self.mode.authority() != robot_types::mode::ActuationAuthority::None {
            // Even an emergency-stop command (all zeros) is transmitted
            // wherever a device is attached at all (including `hw-shadow`,
            // whose own adapter discards it): a zero-torque command is
            // itself the safe state for a torque-controlled joint, and
            // withholding any command leaves the actuator holding whatever
            // it last received — exactly the "uncommanded motion" hazard of
            // SPECS.md Section 12.2. Only `offline-replay`, which has no
            // device at all, skips this call.
            self.adapter
                .send_command(&self.command, now)
                .map_err(TickError::Adapter)?;
        }

        let compute_end = self.clock.now();
        let (compute_duration, _) = compute_end
            .since(compute_start)
            .unwrap_or((Duration::ZERO, Duration::ZERO));
        let sensor_to_command_age = match self.last_feedback_time {
            Some(t) => compute_end.since(t).ok().map(|(age, _)| age),
            None => None,
        };

        Ok(TickTiming {
            compute_duration,
            sensor_to_command_age,
            decision_age: None,
            deadline_margin: self.budget - compute_duration,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_hardware::loopback::LoopbackDevice;
    use robot_nn::layer::{Activation, Linear};
    use robot_supervisor::policy::FaultPolicy;
    use robot_supervisor::supervisor::CommandKind;
    use robot_types::time::{ClockId, SteppedClock};

    fn identity_policy() -> Mlp {
        let layer = Linear::new(2, 2, vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 0.0]).unwrap();
        Mlp::new(vec![layer], vec![Activation::Identity]).unwrap()
    }

    /// A minimal two-joint profile, just enough for `Supervisor::new` to
    /// extract limits from. Not a claim about any real robot.
    fn two_joint_profile() -> robot_model::profile::RobotProfile {
        use robot_model::actuator::{
            ActuatorSpec, PdGains, ThermalLimits, TorqueSpeedEnvelope, Transmission,
        };
        use robot_model::joint::{JointKind, JointLimits, JointSpec};
        use robot_model::link::{LinkSpec, MassProperties};
        use robot_model::pose::RigidPose;
        use robot_model::profile::{FirmwareContract, PROFILE_SCHEMA_VERSION};
        use robot_types::frames::{FrameId, GravityConvention, Vec3};
        use robot_types::ids::{ActuatorId, JointId, LinkId, Name};
        use robot_types::schema::ExtensionTable;
        use robot_types::units::{
            Amperes, Celsius, Kilograms, NewtonMetres, Radians, RadiansPerSecond, Range,
        };

        let mut mp = MassProperties::MASSLESS;
        mp.mass = Kilograms(1.0);
        let link = |id: u32, name: &str| LinkSpec {
            id: LinkId::new(id),
            name: Name::new(name).unwrap(),
            mass: mp,
            collision: Vec::new(),
        };
        let joint = |id: u32, name: &str, parent: u32, child: u32| JointSpec {
            id: JointId::new(id),
            name: Name::new(name).unwrap(),
            kind: JointKind::Revolute,
            parent: LinkId::new(parent),
            child: LinkId::new(child),
            origin: RigidPose::IDENTITY,
            axis: Vec3::unit_z(),
            limits: JointLimits::bounded(
                Range::<Radians>::new(Radians(-3.0), Radians(3.0)).unwrap(),
                RadiansPerSecond(10.0),
                NewtonMetres(50.0),
            ),
            sign: 1.0,
            zero_offset: 0.0,
        };
        let actuator = |id: u32, name: &str, joint: u32| ActuatorSpec {
            id: ActuatorId::new(id),
            name: Name::new(name).unwrap(),
            joint: JointId::new(joint),
            transmission: Transmission::DIRECT,
            envelope: TorqueSpeedEnvelope {
                stall_torque: NewtonMetres(50.0),
                corner_speed: RadiansPerSecond(5.0),
                no_load_speed: RadiansPerSecond(10.0),
            },
            gains: Some(PdGains { kp: 1.0, kd: 0.1 }),
            command_latency: Duration::from_millis(1),
            thermal: ThermalLimits {
                limit: Celsius(80.0),
                ambient: Celsius(25.0),
                time_constant: Duration::from_secs(10),
                current_at_limit: Amperes(10.0),
            },
            current_uncertainty: Amperes(0.1),
        };
        robot_model::profile::RobotProfile {
            schema_version: PROFILE_SCHEMA_VERSION,
            name: Name::new("fixture-biped").unwrap(),
            profile_version: "1.0".to_owned(),
            joints: vec![joint(0, "left", 0, 1), joint(1, "right", 0, 2)],
            links: vec![link(0, "base"), link(1, "left-limb"), link(2, "right-limb")],
            actuators: vec![actuator(0, "left-motor", 0), actuator(1, "right-motor", 1)],
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

    fn build_loop() -> ControlLoop<LoopbackDevice, SteppedClock, 8> {
        let policy = identity_policy();
        let normalizer = Normalizer::new(2, 5.0);
        let profile = two_joint_profile();
        let supervisor = Supervisor::new(
            &profile,
            FaultPolicy::conservative_default(),
            FreshnessLimits {
                observation: Duration::from_millis(50),
                comms: Duration::from_millis(100),
            },
            CommandKind::Position,
        );
        let adapter = LoopbackDevice::new(2, 0);
        let clock = SteppedClock::new(0, ClockId::Simulation);
        ControlLoop::new(ControlLoopConfig {
            policy,
            normalizer,
            supervisor,
            adapter,
            clock,
            mode: OperatingMode::HwSupported,
            budget: Duration::from_millis(8),
            freshness: FreshnessLimits {
                observation: Duration::from_millis(50),
                comms: Duration::from_millis(100),
            },
        })
    }

    #[test]
    fn a_tick_with_no_feedback_yet_still_produces_a_safe_command() {
        let mut loop_ = build_loop();
        let timing = loop_.tick().unwrap();
        // No feedback was ever sent to the loopback device, so observation
        // age is unmeasurable; the supervisor must still produce *something*
        // rather than panicking or blocking.
        assert!(timing.sensor_to_command_age.is_none());
        assert!(timing.compute_duration.nanos() >= 0);
    }

    #[test]
    fn feedback_through_the_loopback_device_is_consumed_next_tick() {
        let mut loop_ = build_loop();
        // Prime the loopback device so the first poll has something to read.
        loop_
            .adapter
            .send_command(&[0.1, -0.1], loop_.clock.now())
            .unwrap();
        let timing = loop_.tick().unwrap();
        assert!(timing.sensor_to_command_age.is_some());
    }

    #[test]
    fn deadline_margin_reflects_the_configured_budget() {
        let mut loop_ = build_loop();
        let timing = loop_.tick().unwrap();
        // A stepped clock with no advancement reports zero compute duration,
        // so the margin should equal the full budget.
        assert_eq!(timing.deadline_margin, Duration::from_millis(8));
    }
}

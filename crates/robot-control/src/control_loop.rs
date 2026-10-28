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

use crate::operation::{LoopOperation, NoOp, TickView, WidthMismatch};
use crate::timing::TickTiming;
use robot_hardware::device::HardwareAdapter;
use robot_nn::mlp::{Mlp, MlpError, Scratch};
use robot_nn::policy::Policy;
use robot_supervisor::policy::Response;
use robot_supervisor::supervisor::{FreshnessLimits, Supervisor, TickInput};
use robot_types::mode::OperatingMode;
use robot_types::obs::{Normalizer, VectorError};
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
    /// The policy or a hook operation failed to produce an action.
    Policy(MlpError),
    /// An observation component was NaN or infinite. The policy and the hook
    /// operation did not run; the supervisor recorded the fault, and its
    /// declared safe command was transmitted wherever a normal command would
    /// have been. The next tick proceeds normally.
    NonFiniteObservation {
        /// Index of the first non-finite component.
        index: usize,
    },
    /// The observation did not have the width the loop was built for.
    ObservationWidth {
        /// The width the loop was built for.
        width: usize,
    },
}

impl core::fmt::Display for TickError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Adapter(e) => write!(f, "{e}"),
            Self::Supervisor(e) => write!(f, "{e}"),
            Self::Policy(e) => write!(f, "policy failed: {e}"),
            Self::NonFiniteObservation { index } => {
                write!(f, "observation component {index} is not finite")
            }
            Self::ObservationWidth { width } => {
                write!(f, "observation width differs from the built width {width}")
            }
        }
    }
}

impl core::error::Error for TickError {}

/// Why a [`ControlLoop`] could not be built from its parts.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ConfigError {
    /// The normalizer's width differs from the policy's input width.
    Normalizer {
        /// The policy's input width.
        expected: usize,
        /// The normalizer's width.
        found: usize,
    },
    /// A hook operation does not fit the loop's widths.
    Operation(WidthMismatch),
    /// The supervisor was built for a different number of joints than the
    /// policy's action width.
    Supervisor {
        /// The policy's output width.
        expected: usize,
        /// The supervisor's width.
        found: usize,
    },
    /// The single-width hardware adapter differs from the selected policy
    /// input or output width.
    Adapter {
        /// Whether the adapter width must match the policy input or output.
        component: AdapterComponent,
        /// The policy width for the selected component.
        expected: usize,
        /// The adapter's width.
        found: usize,
    },
}

/// Which policy width a single-width hardware adapter must match.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AdapterComponent {
    /// Feedback width, matching the policy input.
    Input,
    /// Command width, matching the policy output.
    Output,
}

impl core::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Normalizer { expected, found } => {
                write!(f, "normalizer width is {found}, policy consumes {expected}")
            }
            Self::Operation(e) => write!(f, "{e}"),
            Self::Supervisor { expected, found } => {
                write!(f, "supervisor width is {found}, policy produces {expected}")
            }
            Self::Adapter {
                component,
                expected,
                found,
            } => {
                let role = match component {
                    AdapterComponent::Input => "consumes",
                    AdapterComponent::Output => "produces",
                };
                write!(f, "adapter width is {found}, policy {role} {expected}")
            }
        }
    }
}

impl core::error::Error for ConfigError {}

/// Everything [`ControlLoop::new`] needs, bundled so construction takes one
/// argument instead of eight.
pub struct ControlLoopConfig<A: HardwareAdapter, C: MonotonicClock, const N: usize, P: Policy = Mlp>
{
    /// The policy.
    pub policy: P,
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
///
/// `P` is the policy and `O` the hook operation. Both default to the existing
/// concrete types, so `ControlLoop<A, C, N>` names the same loop as before.
pub struct ControlLoop<
    A: HardwareAdapter,
    C: MonotonicClock,
    const N: usize,
    P: Policy = Mlp,
    O: LoopOperation<P> = NoOp,
> {
    policy: P,
    operation: O,
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

impl<A: HardwareAdapter, C: MonotonicClock, const N: usize, P: Policy>
    ControlLoop<A, C, N, P, NoOp>
{
    /// Builds the loop with no hook operation, allocating every working
    /// buffer up front.
    #[must_use]
    pub fn new(config: ControlLoopConfig<A, C, N, P>) -> Self {
        Self::with_operation(config, NoOp)
    }
}

impl<A: HardwareAdapter, C: MonotonicClock, const N: usize, P: Policy, O: LoopOperation<P>>
    ControlLoop<A, C, N, P, O>
{
    /// Builds the loop with `operation` installed, allocating every working
    /// buffer up front.
    ///
    /// # Panics
    ///
    /// Panics if the configuration does not fit together. The same mismatch is
    /// reported as an error by [`ControlLoop::try_with_operation`].
    #[must_use]
    pub fn with_operation(config: ControlLoopConfig<A, C, N, P>, operation: O) -> Self {
        Self::try_with_operation(config, operation)
            .unwrap_or_else(|error| panic!("invalid control loop configuration: {error}"))
    }

    /// Builds the loop with `operation` installed. The normalizer and the
    /// operation are checked against the policy's widths before any working
    /// buffer is allocated.
    pub fn try_with_operation(
        config: ControlLoopConfig<A, C, N, P>,
        operation: O,
    ) -> Result<Self, ConfigError> {
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
        let normalizer_width = normalizer.mean().len();
        if normalizer_width != obs_width {
            return Err(ConfigError::Normalizer {
                expected: obs_width,
                found: normalizer_width,
            });
        }
        let supervisor_width = supervisor.width();
        if supervisor_width != action_width {
            return Err(ConfigError::Supervisor {
                expected: action_width,
                found: supervisor_width,
            });
        }
        let adapter_width = adapter.width();
        if adapter_width != obs_width {
            return Err(ConfigError::Adapter {
                component: AdapterComponent::Input,
                expected: obs_width,
                found: adapter_width,
            });
        }
        if adapter_width != action_width {
            return Err(ConfigError::Adapter {
                component: AdapterComponent::Output,
                expected: action_width,
                found: adapter_width,
            });
        }
        operation
            .check_widths(obs_width, action_width)
            .map_err(ConfigError::Operation)?;
        let scratch = Scratch::new(policy.scratch_len());
        Ok(Self {
            policy,
            operation,
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
        })
    }

    /// The most recent supervisor response, for a caller building a report.
    #[must_use]
    pub fn supervisor_log(&self) -> &robot_supervisor::fault::FaultLog<N> {
        self.supervisor.log()
    }

    /// The installed hook operation, for a caller that reads or adjusts it
    /// between ticks.
    #[must_use]
    pub fn operation(&self) -> &O {
        &self.operation
    }

    /// Mutable access to the installed hook operation, for example to toggle
    /// sampling between ticks.
    pub fn operation_mut(&mut self) -> &mut O {
        &mut self.operation
    }

    /// Checks `raw_action` against the supervisor and transmits the resulting
    /// command. A transmission error takes precedence over any earlier error
    /// from the same tick; the supervisor's fault log still records what
    /// happened.
    fn supervise_and_transmit(
        &mut self,
        now: Timestamp,
        observation_age: Option<Duration>,
    ) -> Result<Response, TickError> {
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
        let response: Response = self
            .supervisor
            .check_tick(&supervisor_input, &self.raw_action, &mut self.command)
            .map_err(TickError::Supervisor)?;

        if self.mode.authority() != robot_types::mode::ActuationAuthority::None {
            // Even an emergency-stop command (all zeros) is transmitted
            // wherever a device is attached at all: withholding a command
            // leaves the actuator holding its last command, the uncommanded
            // motion hazard of SPECS.md Section 12.2. Only `offline-replay`,
            // which has no device, skips this call.
            self.adapter
                .send_command(&self.command, now)
                .map_err(TickError::Adapter)?;
        }
        Ok(response)
    }

    /// Handles a tick whose observation cannot be used: the supervisor
    /// records [`robot_supervisor::fault::Fault::NonFiniteObservation`] and
    /// produces its declared safe command, which is transmitted exactly where
    /// a normal tick's command would be. The policy and the hook operation
    /// are not run, because there is no valid observation to give them.
    fn escalate_unusable_observation(
        &mut self,
        now: Timestamp,
        observation_age: Option<Duration>,
        index: usize,
    ) -> Result<(), TickError> {
        let supervisor_input = TickInput {
            mode: self.mode,
            observation_age: observation_age
                .unwrap_or(self.observation_freshness_limit + Duration::from_nanos(1)),
            comms_age: self.adapter.link_age(now).unwrap_or(Duration::ZERO),
            temperatures: &[],
            active_contacts: &[],
            permitted_contacts: &[],
        };
        self.supervisor
            .reject_observation(&supervisor_input, index, &mut self.command)
            .map_err(TickError::Supervisor)?;
        if self.mode.authority() != robot_types::mode::ActuationAuthority::None {
            self.adapter
                .send_command(&self.command, now)
                .map_err(TickError::Adapter)?;
        }
        Ok(())
    }

    /// Runs one control tick: poll, infer, supervise, transmit.
    ///
    /// # Errors and hooks
    ///
    /// An error ends the tick at the stage that failed, and the hook
    /// operation is told only about stages that happened:
    ///
    /// - `before_tick` runs once the observation is normalized. A failure
    ///   before that (adapter poll, non-finite or mis-sized observation)
    ///   fires no hook.
    /// - `after_tick` runs only when the whole tick succeeded. If
    ///   `select_action`, the supervisor or `send_command` fails, `before_tick`
    ///   has fired and `after_tick` will not, so an operation must not assume
    ///   the two calls are paired, and a sampled action whose transmission
    ///   failed is not recorded.
    /// - A non-finite observation is not silent: the supervisor logs the
    ///   fault and its safe command is transmitted before
    ///   [`TickError::NonFiniteObservation`] is returned.
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

        if let Err(error) = self
            .normalizer
            .normalize(&self.feedback, &mut self.normalized)
        {
            return Err(match error {
                VectorError::NotFinite { index } => {
                    self.escalate_unusable_observation(now, observation_age, index)?;
                    TickError::NonFiniteObservation { index }
                }
                _ => TickError::ObservationWidth {
                    width: self.feedback.len(),
                },
            });
        }

        self.operation.before_tick(&self.normalized);
        let log_prob = match self.operation.select_action(
            &self.policy,
            &self.normalized,
            &mut self.raw_action,
            &mut self.scratch,
        ) {
            Ok(log_prob) => log_prob,
            Err(error) => {
                // A failed policy produced no usable action: supervise it as a
                // non-finite command so the declared safe response is sent.
                self.raw_action.fill(f64::NAN);
                self.supervise_and_transmit(now, observation_age)?;
                return Err(TickError::Policy(error));
            }
        };

        let response = self.supervise_and_transmit(now, observation_age)?;
        self.operation.after_tick(&TickView {
            observation: &self.normalized,
            raw_action: &self.raw_action,
            command: &self.command,
            response,
            log_prob,
        });

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
    use robot_hardware::device::AdapterError;
    use robot_hardware::loopback::LoopbackDevice;
    use robot_hardware::replay::{RecordedFrame, ReplayDevice};
    use robot_nn::layer::{Activation, Linear};
    use robot_supervisor::policy::FaultPolicy;
    use robot_supervisor::supervisor::CommandKind;
    use robot_types::time::{ClockId, SteppedClock, Timestamp};

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

    /// A policy that is not `Mlp`, to show the loop accepts any `Policy`.
    struct ConstantPolicy;

    impl Policy for ConstantPolicy {
        fn input_width(&self) -> usize {
            2
        }

        fn output_width(&self) -> usize {
            2
        }

        fn scratch_len(&self) -> usize {
            0
        }

        fn act(
            &self,
            _observation: &[f64],
            action: &mut [f64],
            _scratch: &mut Scratch,
        ) -> Result<(), robot_nn::mlp::MlpError> {
            action.fill(0.25);
            Ok(())
        }
    }

    #[derive(Default)]
    struct CountingOp {
        ticks: usize,
    }

    impl<P: Policy> LoopOperation<P> for CountingOp {
        fn after_tick(&mut self, _tick: &TickView<'_>) {
            self.ticks += 1;
        }
    }

    fn build_loop_with<P: Policy, O: LoopOperation<P>>(
        policy: P,
        operation: O,
    ) -> ControlLoop<LoopbackDevice, SteppedClock, 8, P, O> {
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
        ControlLoop::with_operation(
            ControlLoopConfig {
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
            },
            operation,
        )
    }

    #[test]
    fn external_policy_and_hook_run_without_editing_the_loop() {
        let mut loop_ = build_loop_with(ConstantPolicy, CountingOp::default());
        loop_.tick().unwrap();
        loop_.tick().unwrap();
        assert_eq!(loop_.operation().ticks, 2);
    }

    #[test]
    fn installed_operation_is_reachable_through_operation_mut() {
        let mut loop_ = build_loop_with(identity_policy(), CountingOp::default());
        loop_.operation_mut().ticks = 10;
        loop_.tick().unwrap();
        assert_eq!(loop_.operation().ticks, 11);
    }

    #[test]
    fn experience_recorder_captures_one_transition_per_tick() {
        let recorder = crate::operation::ExperienceRecorder::new(2, 2, 4);
        let mut loop_ = build_loop_with(identity_policy(), recorder);
        loop_.tick().unwrap();
        loop_.tick().unwrap();
        assert_eq!(loop_.operation().len(), 2);
        assert!(loop_.operation().last().is_some());
    }

    /// Builds the loop like `build_loop_with`, but returns the construction
    /// error instead of panicking.
    fn try_build<P: Policy, O: LoopOperation<P>>(
        policy: P,
        normalizer_width: usize,
        operation: O,
    ) -> Result<ControlLoop<LoopbackDevice, SteppedClock, 8, P, O>, ConfigError> {
        try_build_with_adapter_width(policy, normalizer_width, 2, operation)
    }

    fn try_build_with_adapter_width<P: Policy, O: LoopOperation<P>>(
        policy: P,
        normalizer_width: usize,
        adapter_width: usize,
        operation: O,
    ) -> Result<ControlLoop<LoopbackDevice, SteppedClock, 8, P, O>, ConfigError> {
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
        ControlLoop::try_with_operation(
            ControlLoopConfig {
                policy,
                normalizer: Normalizer::new(normalizer_width, 5.0),
                supervisor,
                adapter: LoopbackDevice::new(adapter_width, 0),
                clock: SteppedClock::new(0, ClockId::Simulation),
                mode: OperatingMode::HwSupported,
                budget: Duration::from_millis(8),
                freshness: FreshnessLimits {
                    observation: Duration::from_millis(50),
                    comms: Duration::from_millis(100),
                },
            },
            operation,
        )
    }

    fn build_loop_with_replay<P: Policy, O: LoopOperation<P>>(
        policy: P,
        operation: O,
        feedback: [f64; 2],
    ) -> ControlLoop<ReplayDevice, SteppedClock, 8, P, O> {
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
        let adapter = ReplayDevice::new(
            2,
            vec![RecordedFrame {
                values: feedback.to_vec(),
                timestamp: Timestamp::sampling(0, ClockId::Simulation),
                is_sampling_instant: true,
            }],
        )
        .unwrap();
        ControlLoop::with_operation(
            ControlLoopConfig {
                policy,
                normalizer: Normalizer::new(2, 5.0),
                supervisor,
                adapter,
                clock: SteppedClock::new(0, ClockId::Simulation),
                mode: OperatingMode::HwSupported,
                budget: Duration::from_millis(8),
                freshness: FreshnessLimits {
                    observation: Duration::from_millis(50),
                    comms: Duration::from_millis(100),
                },
            },
            operation,
        )
    }

    /// A policy whose every evaluation fails.
    struct FailingPolicy;

    impl Policy for FailingPolicy {
        fn input_width(&self) -> usize {
            2
        }

        fn output_width(&self) -> usize {
            2
        }

        fn scratch_len(&self) -> usize {
            0
        }

        fn act(
            &self,
            _observation: &[f64],
            _action: &mut [f64],
            _scratch: &mut Scratch,
        ) -> Result<(), robot_nn::mlp::MlpError> {
            Err(robot_nn::mlp::MlpError::Empty)
        }
    }

    /// A policy of configurable widths that always outputs zeros, for the
    /// construction checks.
    struct SizedPolicy {
        input: usize,
        output: usize,
    }

    impl Policy for SizedPolicy {
        fn input_width(&self) -> usize {
            self.input
        }

        fn output_width(&self) -> usize {
            self.output
        }

        fn scratch_len(&self) -> usize {
            0
        }

        fn act(
            &self,
            _observation: &[f64],
            action: &mut [f64],
            _scratch: &mut Scratch,
        ) -> Result<(), robot_nn::mlp::MlpError> {
            action.fill(0.0);
            Ok(())
        }
    }

    #[test]
    fn default_policy_command_is_the_policy_output_for_a_fixed_observation() {
        let mut loop_ = build_loop();
        loop_
            .adapter
            .send_command(&[0.1, -0.1], loop_.clock.now())
            .unwrap();
        loop_.tick().unwrap();
        // Fresh normalizer is the identity and the identity policy passes it
        // through, so the supervised command equals the observation.
        assert_eq!(loop_.command, vec![0.1, -0.1]);
    }

    #[test]
    fn a_nan_observation_is_a_tick_error_not_a_panic() {
        let mut loop_ = build_loop();
        loop_
            .adapter
            .send_command(&[f64::NAN, 0.0], loop_.clock.now())
            .unwrap();
        assert!(matches!(
            loop_.tick(),
            Err(TickError::NonFiniteObservation { index: 0 })
        ));
        loop_
            .adapter
            .send_command(&[0.1, -0.1], loop_.clock.now())
            .unwrap();
        assert!(loop_.tick().is_ok());
    }

    #[test]
    fn a_nan_observation_is_a_supervisor_fault_and_transmits_the_safe_command() {
        use robot_supervisor::fault::Fault;

        let mut loop_ = build_loop();
        // A good tick first, so a non-zero last command exists.
        loop_
            .adapter
            .send_command(&[0.1, -0.1], loop_.clock.now())
            .unwrap();
        loop_.tick().unwrap();
        // Drain the echo of that tick's own command.
        let mut sink = [0.0; 2];
        loop_.adapter.poll_feedback(&mut sink).unwrap();
        assert_eq!(sink, [0.1, -0.1]);

        loop_
            .adapter
            .send_command(&[0.1, f64::INFINITY], loop_.clock.now())
            .unwrap();
        assert_eq!(
            loop_.tick().err(),
            Some(TickError::NonFiniteObservation { index: 1 })
        );

        // The supervisor saw it and logged it with the index.
        assert!(loop_
            .supervisor_log()
            .entries()
            .contains(&Some(Fault::NonFiniteObservation { index: 1 })));
        // The conservative policy escalates it to an emergency stop, so the
        // command that went out is all zeros, not a value derived from the
        // bad observation and not silence.
        assert_eq!(loop_.command, vec![0.0, 0.0]);
        loop_.adapter.poll_feedback(&mut sink).unwrap();
        assert_eq!(sink, [0.0, 0.0]);
    }

    #[test]
    fn a_nan_observation_skips_the_hooks_and_the_policy() {
        let mut loop_ = build_loop_with(ConstantPolicy, CountingOp::default());
        loop_
            .adapter
            .send_command(&[f64::NAN, 0.0], loop_.clock.now())
            .unwrap();
        assert!(loop_.tick().is_err());
        assert_eq!(loop_.operation().ticks, 0);
        assert_eq!(loop_.raw_action, vec![0.0, 0.0]);
    }

    #[test]
    fn a_failed_policy_is_supervised_and_sends_the_safe_command() {
        use robot_supervisor::fault::Fault;
        use robot_types::ids::JointId;

        #[derive(Default)]
        struct PairCounter {
            before: usize,
            after: usize,
        }
        impl<P: Policy> LoopOperation<P> for PairCounter {
            fn before_tick(&mut self, _observation: &[f64]) {
                self.before += 1;
            }
            fn after_tick(&mut self, _tick: &TickView<'_>) {
                self.after += 1;
            }
        }
        let mut loop_ = build_loop_with(FailingPolicy, PairCounter::default());
        assert!(matches!(
            loop_.tick(),
            Err(TickError::Policy(robot_nn::mlp::MlpError::Empty))
        ));
        assert_eq!(loop_.operation().before, 1);
        assert_eq!(loop_.operation().after, 0);
        // The failure is a supervisor fault on every joint, not silence.
        for joint in [0, 1] {
            assert!(loop_
                .supervisor_log()
                .entries()
                .contains(&Some(Fault::NonFiniteCommand {
                    joint: JointId::new(joint),
                })));
        }
        // The conservative policy's emergency stop is transmitted: zeros.
        assert_eq!(loop_.command, vec![0.0, 0.0]);
        let mut sink = [0.0; 2];
        loop_.adapter.poll_feedback(&mut sink).unwrap();
        assert_eq!(sink, [0.0, 0.0]);
    }

    #[test]
    fn adapter_send_failure_takes_precedence_over_non_finite_observation() {
        use robot_supervisor::fault::Fault;

        let mut loop_ = build_loop_with_replay(identity_policy(), NoOp, [f64::NAN, 0.0]);
        assert!(matches!(
            loop_.tick(),
            Err(TickError::Adapter(AdapterError::TransmitDisabled))
        ));
        assert!(loop_
            .supervisor_log()
            .entries()
            .contains(&Some(Fault::NonFiniteObservation { index: 0 })));
    }

    #[test]
    fn adapter_send_failure_takes_precedence_over_policy_failure() {
        use robot_supervisor::fault::Fault;
        use robot_types::ids::JointId;

        let mut loop_ = build_loop_with_replay(FailingPolicy, NoOp, [0.0, 0.0]);
        assert!(matches!(
            loop_.tick(),
            Err(TickError::Adapter(AdapterError::TransmitDisabled))
        ));
        for joint in [0, 1] {
            assert!(loop_
                .supervisor_log()
                .entries()
                .contains(&Some(Fault::NonFiniteCommand {
                    joint: JointId::new(joint),
                })));
        }
    }

    #[test]
    fn the_tick_path_reuses_its_buffers_across_ticks() {
        // Safe in-repo regression evidence for SPECS.md Section 5 (the crate
        // forbids `unsafe`, so a counting allocator cannot live here): every
        // working buffer keeps its address and capacity, and the recorder's
        // capacity does not move, across many ticks with sampling and
        // recording installed.
        let operation = (
            crate::operation::SamplingToggle::new(robot_types::rng::SeedStream::new(7, 0), true),
            crate::operation::ExperienceRecorder::new(2, 2, 4),
        );
        let mut loop_ = build_loop_with(crate::operation::tests::identity_gaussian(), operation);
        loop_
            .adapter
            .send_command(&[0.1, -0.1], loop_.clock.now())
            .unwrap();
        loop_.tick().unwrap();
        fn addresses<P: Policy, O: LoopOperation<P>>(
            l: &ControlLoop<LoopbackDevice, SteppedClock, 8, P, O>,
        ) -> [(usize, usize); 4] {
            [
                (l.feedback.as_ptr() as usize, l.feedback.capacity()),
                (l.normalized.as_ptr() as usize, l.normalized.capacity()),
                (l.raw_action.as_ptr() as usize, l.raw_action.capacity()),
                (l.command.as_ptr() as usize, l.command.capacity()),
            ]
        }
        let before = addresses(&loop_);
        for _ in 0..50 {
            loop_.tick().unwrap();
        }
        assert_eq!(addresses(&loop_), before);
        assert_eq!(loop_.operation().1.capacity(), 4);
        assert_eq!(loop_.operation().1.len(), 4);
    }

    #[test]
    fn a_policy_error_is_a_tick_error_not_a_panic() {
        let mut loop_ = build_loop_with(FailingPolicy, NoOp);
        assert!(matches!(
            loop_.tick(),
            Err(TickError::Policy(robot_nn::mlp::MlpError::Empty))
        ));
    }

    #[test]
    fn an_operation_that_does_not_fit_the_policy_is_rejected_at_construction() {
        let recorder = crate::operation::ExperienceRecorder::new(3, 2, 4);
        assert_eq!(
            try_build(identity_policy(), 2, recorder).err(),
            Some(ConfigError::Operation(WidthMismatch {
                component: "observation",
                expected: 2,
                found: 3,
            }))
        );
    }

    #[test]
    fn a_supervisor_that_does_not_fit_the_policy_is_rejected_at_construction() {
        let wide = SizedPolicy {
            input: 2,
            output: 3,
        };
        assert_eq!(
            try_build(wide, 2, NoOp).err(),
            Some(ConfigError::Supervisor {
                expected: 3,
                found: 2,
            })
        );
    }

    #[test]
    fn an_adapter_that_does_not_fit_either_policy_width_is_rejected_at_construction() {
        assert_eq!(
            try_build_with_adapter_width(
                SizedPolicy {
                    input: 3,
                    output: 2,
                },
                3,
                3,
                NoOp,
            )
            .err(),
            Some(ConfigError::Adapter {
                component: AdapterComponent::Output,
                expected: 2,
                found: 3,
            })
        );

        let unequal_width_policy = SizedPolicy {
            input: 3,
            output: 2,
        };
        assert_eq!(
            try_build_with_adapter_width(unequal_width_policy, 3, 2, NoOp).err(),
            Some(ConfigError::Adapter {
                component: AdapterComponent::Input,
                expected: 3,
                found: 2,
            })
        );
    }

    #[test]
    fn a_normalizer_that_does_not_fit_the_policy_is_rejected_at_construction() {
        assert_eq!(
            try_build(identity_policy(), 3, NoOp).err(),
            Some(ConfigError::Normalizer {
                expected: 2,
                found: 3,
            })
        );
    }

    #[test]
    fn sampling_and_recording_run_together_through_the_loop() {
        let policy = crate::operation::tests::identity_gaussian();
        let operation = (
            crate::operation::SamplingToggle::new(robot_types::rng::SeedStream::new(5, 0), true),
            crate::operation::ExperienceRecorder::new(2, 2, 8),
        );
        let mut loop_ = build_loop_with(policy, operation);
        for _ in 0..3 {
            loop_.tick().unwrap();
        }
        let recorder = &loop_.operation().1;
        assert_eq!(recorder.len(), 3);
        let last = recorder.last().unwrap();
        assert!(last.log_prob.is_some_and(f64::is_finite));
        assert_ne!(last.raw_action, &[0.0, 0.0][..]);
    }
}

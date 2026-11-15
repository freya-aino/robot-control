//! The humanoid training environment: `robot-sim`'s `World` stepped under
//! the model's own declared drives at a control rate, observed through a
//! [`HumanoidTask`], exposed as `robot-sim`'s [`Environment`] so
//! `robot-learn`'s PPO can collect from it unchanged.
//!
//! This is the glue SPECS.md Section 9.2 implies between the three crates
//! that already exist: `robot-sim` owns physics and deliberately returns no
//! reward (`robot_sim::env`'s own documentation), `robot-tasks` owns reward
//! and termination over [`HumanoidState`], and `robot-learn` consumes any
//! `Environment` plus a task closure over flat observations. The closure
//! signature cannot see inside the environment, while a humanoid reward is
//! written against the rich [`HumanoidState`] this module extracts — so the
//! environment evaluates [`step_task`] itself and hands the outcome to the
//! closure through a shared [`OutcomeCell`], which [`outcome_task_fn`]
//! drains. The reward registry stays the single source of reward truth; the
//! closure adds nothing.
//!
//! **Action space**: one control target per declared drive, in the model's
//! actuator order, applied as the drive law `kp (ctrl - q) - kd qd` clamped
//! to the declared force limit ([`ActuationMode::Drives`]). The policy
//! commands positions and the model's own actuators track them, which is the
//! transmission a real robot offers; raw torques are deliberately not the
//! action space here.
//!
//! **Control rate**: each [`Environment::step`] advances `substeps` physics
//! steps under one held action, so the policy runs at
//! `substeps x timestep` while the solver runs at its own validated
//! timestep — the Section 5 tier separation in miniature.

use std::cell::RefCell;
use std::rc::Rc;

use robot_sim::env::{Environment, PhysicsStep};
use robot_sim::world::{ActuationMode, State, World};
use robot_types::rng::{Rng, SeedStream};
use robot_types::transition::{RewardTerms, Termination};

use crate::humanoid::{step_task, HumanoidState, HumanoidTask, StateExtractor, Status};

/// What one control step produced, for the task closure.
#[derive(Clone, Debug)]
pub struct StepOutcome {
    /// The evaluated reward terms.
    pub reward: RewardTerms,
    /// The task's episode status after the step.
    pub status: Status,
}

/// The channel between [`HumanoidEnv::step`] and [`outcome_task_fn`]: the
/// environment writes each step's outcome, the task closure takes it.
pub type OutcomeCell = Rc<RefCell<Option<StepOutcome>>>;

/// Why a [`HumanoidEnv`] could not be built.
#[derive(Clone, PartialEq, Debug)]
pub enum EnvError {
    /// The world simulates no drives, so no action could move the robot.
    NoDrives,
    /// `q_home` does not match the world's position width.
    WidthMismatch {
        /// The world's `nq`.
        expected: usize,
        /// What was supplied.
        found: usize,
    },
    /// `substeps` was zero.
    ZeroSubsteps,
    /// The initial state could not be read as a humanoid state.
    Extract(String),
}

impl core::fmt::Display for EnvError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoDrives => f.write_str("the world declares no simulated drives"),
            Self::WidthMismatch { expected, found } => {
                write!(f, "q_home has width {found}, the world expects {expected}")
            }
            Self::ZeroSubsteps => f.write_str("substeps must be at least 1"),
            Self::Extract(e) => write!(f, "initial state: {e}"),
        }
    }
}

impl core::error::Error for EnvError {}

/// A `World` driven by a [`HumanoidTask`], as an [`Environment`].
pub struct HumanoidEnv<T: HumanoidTask> {
    world: World,
    state: State,
    q_home: Vec<f64>,
    extractor: StateExtractor,
    task: T,
    substeps: usize,
    q_spread: f64,
    rng: Rng,
    time: f64,
    step_count: usize,
    prev_state: HumanoidState,
    prev_action: Vec<f64>,
    outcome: OutcomeCell,
    observation: Vec<f64>,
}

impl<T: HumanoidTask> HumanoidEnv<T> {
    /// Builds the environment around a world that already has its ground and
    /// options, a home configuration (the caller places it above the plane),
    /// and the task.
    ///
    /// `substeps` physics steps run per control step; `q_spread` is the
    /// uniform per-joint perturbation a reset draws around `q_home`.
    ///
    /// # Errors
    ///
    /// [`EnvError`] when the world has no drives, the widths disagree,
    /// `substeps` is zero, or the home state is not a humanoid state.
    pub fn new(
        world: World,
        q_home: Vec<f64>,
        extractor: StateExtractor,
        task: T,
        substeps: usize,
        q_spread: f64,
        seed: SeedStream,
    ) -> Result<Self, EnvError> {
        if world.actuator_count() == 0 {
            return Err(EnvError::NoDrives);
        }
        if q_home.len() != world.nq() {
            return Err(EnvError::WidthMismatch {
                expected: world.nq(),
                found: q_home.len(),
            });
        }
        if substeps == 0 {
            return Err(EnvError::ZeroSubsteps);
        }
        let state = State {
            qd: vec![0.0; world.nv()],
            q: q_home.clone(),
        };
        let prev_state = extractor
            .extract(&world, &state, &[], 0.0)
            .map_err(EnvError::Extract)?;
        let prev_action = vec![0.0; world.actuator_count()];
        let mut env = Self {
            world,
            state,
            q_home,
            extractor,
            task,
            substeps,
            q_spread,
            rng: Rng::new(seed),
            time: 0.0,
            step_count: 0,
            prev_state,
            prev_action,
            outcome: Rc::new(RefCell::new(None)),
            observation: Vec::new(),
        };
        env.observation = env.observe(&env.prev_state.clone());
        Ok(env)
    }

    /// The channel [`outcome_task_fn`] reads this environment's rewards and
    /// statuses from.
    #[must_use]
    pub fn outcome_cell(&self) -> OutcomeCell {
        Rc::clone(&self.outcome)
    }

    /// The task, for a caller inspecting its configuration.
    #[must_use]
    pub fn task(&self) -> &T {
        &self.task
    }

    /// The world, for a caller that needs the configuration record.
    #[must_use]
    pub fn world(&self) -> &World {
        &self.world
    }

    /// The simulated time per control step, seconds.
    #[must_use]
    pub fn control_dt(&self) -> f64 {
        self.substeps as f64 * self.world.options().timestep.as_secs_f64()
    }

    fn observe(&self, s: &HumanoidState) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.task.observation_spec().width());
        self.task.observe(s, &mut out);
        out
    }

    /// The per-drive torques of `tau` in the extractor's joint order.
    fn actuated_torques(&self, tau: &[f64]) -> Vec<f64> {
        match self.world.model().dof_to_external(tau) {
            Ok(ext) if ext.len() > 6 => ext[6..].to_vec(),
            _ => Vec::new(),
        }
    }
}

impl<T: HumanoidTask> Environment for HumanoidEnv<T> {
    fn observation_width(&self) -> usize {
        self.task.observation_spec().width()
    }

    fn action_width(&self) -> usize {
        self.world.actuator_count()
    }

    fn reset(&mut self, seed: SeedStream, observation: &mut [f64]) {
        self.rng = Rng::new(seed);
        let mut q = self.q_home.clone();
        for body in self.world.model().bodies.iter() {
            if body.dof_num == 1 && body.qpos_num == 1 {
                q[body.qpos_adr] += self.rng.uniform(-self.q_spread, self.q_spread);
            }
        }
        self.state = State {
            qd: vec![0.0; self.world.nv()],
            q,
        };
        self.time = 0.0;
        self.step_count = 0;
        self.prev_action = vec![0.0; self.world.actuator_count()];
        if let Ok(s) = self
            .extractor
            .extract(&self.world, &self.state, &[], self.time)
        {
            self.observation = self.observe(&s);
            self.prev_state = s;
        }
        let width = observation.len().min(self.observation.len());
        observation[..width].copy_from_slice(&self.observation[..width]);
        *self.outcome.borrow_mut() = None;
    }

    fn step(&mut self, action: &[f64], _dt: robot_types::time::Duration) -> PhysicsStep {
        // The physics timestep is the world's recorded knob; the control
        // step is `substeps` of it, so the trait's `dt` is ignored for the
        // same reason `WorldEnv` ignores it.
        let failure = |observation: Vec<f64>| PhysicsStep {
            observation,
            physics_failure: true,
        };
        if action.len() != self.world.actuator_count() {
            return failure(self.observation.clone());
        }
        let nv = self.world.nv();
        let mut tau = vec![0.0; nv];
        for _ in 0..self.substeps {
            tau.iter_mut().for_each(|t| *t = 0.0);
            if self
                .world
                .actuator_torques(
                    &self.state.q,
                    &self.state.qd,
                    action,
                    ActuationMode::Drives,
                    &mut tau,
                )
                .is_err()
                || self.world.step(&mut self.state, &tau).is_err()
            {
                return failure(self.observation.clone());
            }
        }
        self.time += self.control_dt();
        self.step_count += 1;
        let finite = self
            .state
            .q
            .iter()
            .chain(&self.state.qd)
            .all(|v| v.is_finite());
        let torque = self.actuated_torques(&tau);
        let next = match self
            .extractor
            .extract(&self.world, &self.state, &torque, self.time)
        {
            Ok(s) if finite => s,
            _ => return failure(self.observation.clone()),
        };
        let (reward, status) = step_task(
            &mut self.task,
            &self.prev_state,
            &self.prev_action,
            action,
            &next,
            self.step_count,
        );
        *self.outcome.borrow_mut() = Some(StepOutcome { reward, status });
        self.observation = self.observe(&next);
        self.prev_state = next;
        self.prev_action = action.to_vec();
        PhysicsStep {
            observation: self.observation.clone(),
            physics_failure: false,
        }
    }
}

/// What a task closure returns for one step: the reward terms and how the
/// episode ended, in `robot-learn`'s `TaskFn` convention.
pub type StepVerdict = (RewardTerms, Termination);

/// A task closure for `robot-learn`'s rollout collection, reading each step's
/// reward and status from `cell`.
///
/// A step whose outcome is missing, or that the physics reported as failed,
/// terminates the episode with zero reward: a simulator failure must end the
/// episode as a terminal state, never be rewarded through.
pub fn outcome_task_fn(
    cell: OutcomeCell,
) -> impl Fn(&[f64], &[f64], &[f64], u32, bool) -> StepVerdict {
    move |_obs, _action, _next, _step, physics_failure| {
        let taken = cell.borrow_mut().take();
        match taken {
            Some(outcome) if !physics_failure => {
                let termination = match outcome.status {
                    Status::Running => Termination::Continuing,
                    Status::Terminated(_) => Termination::Terminal,
                    Status::Truncated => Termination::TimeLimit,
                };
                (outcome.reward, termination)
            }
            _ => (RewardTerms::new(), Termination::Terminal),
        }
    }
}

/// The vendored G1 standing on the plane under its declared drives, with the
/// [`crate::tasks::StandTask`] — the reference humanoid training wiring.
///
/// Physics runs `implicitfast` at a 1 ms timestep, which the 2026-11-15
/// stance sweep measured as the cheapest configuration that still passes the
/// full supported-stance battery (semi-implicit Euler fails its settled-
/// energy check at this timestep and RK4 diverges outright; see
/// `crates/robot-sim/specs/09-fidelity-tiers.md`). Five substeps give a 5 ms
/// control step (200 Hz). The task's fall limits truncate at `max_steps`
/// control steps.
///
/// # Errors
///
/// [`EnvError`] as [`HumanoidEnv::new`] reports it; the fixture itself is
/// expected to load, and a failure to do so panics the way every vendored-
/// fixture consumer in this workspace does.
pub fn g1_stand(max_steps: usize, seed: SeedStream) -> Result<HumanoidEnv<StandTask>, EnvError> {
    use robot_sim::options::{Integrator, SimOptions};
    use robot_sim::world::Ground;
    use robot_types::time::Duration;

    let fixed =
        robot_model::fixtures::unitree_g1_29dof_usd().expect("the vendored G1 USD model loads");
    let floating =
        robot_model::usd::with_floating_base(fixed.clone()).expect("the floating wrap succeeds");
    let link = |n: &str| {
        fixed
            .links
            .iter()
            .position(|l| l.name.as_str() == n)
            .expect("a declared G1 link name")
    };
    let world = World::from_profile(
        &floating,
        SimOptions {
            timestep: Duration::from_millis(1),
            integrator: Integrator::ImplicitFast,
            ..SimOptions::default()
        },
    )
    .with_ground(Ground {
        height: 0.0,
        friction: 0.8,
    });
    let mut q = world.rest_state().q;
    let lowest = world
        .lowest_contact_height(&q)
        .expect("a well-formed state")
        .expect("the G1 declares contact shapes");
    let mut ext = world.model().qpos_to_external(&q).expect("widths match");
    ext[2] += 0.005 - lowest;
    q = world.model().qpos_to_internal(&ext).expect("widths match");
    // Only names the USD layer actually declares: the two ankles and the
    // pelvis. (The URDF fixture's `head_link` and `right_rubber_hand` are
    // not link names in the vendored USD.)
    let extractor = StateExtractor {
        profile: fixed.clone(),
        tracked_links: vec![link("left_ankle_roll_link"), link("right_ankle_roll_link")],
        feet: vec![0, 1],
        forbidden_contact: vec![link("pelvis")],
        contact_height: 0.08,
        ground_height: 0.0,
    };
    let task = StandTask::new(
        fixed.dof(),
        0.79,
        crate::tasks::FallLimits {
            min_height: 0.4,
            max_tilt: 0.8,
            max_steps,
        },
    );
    HumanoidEnv::new(world, q, extractor, task, 5, 0.01, seed)
}

use crate::tasks::StandTask;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::humanoid::Status;
    use crate::tasks::FallLimits;
    use robot_model::fixtures::unitree_g1_29dof_usd;
    use robot_model::usd::with_floating_base;
    use robot_sim::options::SimOptions;
    use robot_types::time::Duration;

    fn standing_env() -> HumanoidEnv<StandTask> {
        g1_stand(400, SeedStream::new(7, 0)).unwrap()
    }

    #[test]
    fn the_g1_stands_through_the_environment_under_its_declared_drives() {
        // Zero control targets command the home pose, which the declared
        // drives hold (the same stance `robot sim multibody --floating`
        // qualifies). 25 control steps are 0.125 simulated seconds.
        let mut env = standing_env();
        let cell = env.outcome_cell();
        let mut obs = vec![0.0; env.observation_width()];
        env.reset(SeedStream::new(3, 0), &mut obs);
        let action = vec![0.0; env.action_width()];
        for step in 0..25 {
            let out = env.step(&action, Duration::from_millis(5));
            assert!(!out.physics_failure, "physics failed at step {step}");
            let outcome = cell.borrow_mut().take().expect("an outcome per step");
            assert!(outcome.reward.total().is_finite());
            assert_eq!(outcome.status, Status::Running, "fell at step {step}");
        }
    }

    #[test]
    fn resets_are_seed_deterministic_and_seeds_differ() {
        let mut env = standing_env();
        let mut a = vec![0.0; env.observation_width()];
        let mut b = vec![0.0; env.observation_width()];
        env.reset(SeedStream::new(11, 0), &mut a);
        env.reset(SeedStream::new(11, 0), &mut b);
        assert_eq!(a, b, "the same seed must reproduce the same start");
        env.reset(SeedStream::new(12, 0), &mut b);
        assert_ne!(a, b, "a different seed must move the start");
    }

    #[test]
    fn a_failed_or_missing_outcome_terminates_with_zero_reward() {
        let env = standing_env();
        let f = outcome_task_fn(env.outcome_cell());
        // No outcome in the cell: terminal, zero reward.
        let (r, t) = f(&[], &[], &[], 1, false);
        assert_eq!(t, Termination::Terminal);
        assert_eq!(r.total(), 0.0);
        // An outcome, but the physics failed: still terminal.
        *env.outcome_cell().borrow_mut() = Some(StepOutcome {
            reward: RewardTerms::new(),
            status: Status::Running,
        });
        let (_, t) = f(&[], &[], &[], 1, true);
        assert_eq!(t, Termination::Terminal);
    }

    #[test]
    fn construction_refuses_the_malformed_cases_by_name() {
        let fixed = unitree_g1_29dof_usd().unwrap();
        let floating = with_floating_base(fixed.clone()).unwrap();
        let world = || World::from_profile(&floating, SimOptions::default());
        let extractor = || StateExtractor {
            profile: fixed.clone(),
            tracked_links: vec![],
            feet: vec![],
            forbidden_contact: vec![],
            contact_height: 0.08,
            ground_height: 0.0,
        };
        let task = || {
            StandTask::new(
                fixed.dof(),
                0.79,
                FallLimits {
                    min_height: 0.4,
                    max_tilt: 0.8,
                    max_steps: 10,
                },
            )
        };
        let nq = world().nq();
        assert_eq!(
            HumanoidEnv::new(
                world(),
                vec![0.0; 3],
                extractor(),
                task(),
                1,
                0.0,
                SeedStream::new(1, 0)
            )
            .err(),
            Some(EnvError::WidthMismatch {
                expected: nq,
                found: 3
            })
        );
        let q = world().rest_state().q;
        assert_eq!(
            HumanoidEnv::new(
                world(),
                q,
                extractor(),
                task(),
                0,
                0.0,
                SeedStream::new(1, 0)
            )
            .err(),
            Some(EnvError::ZeroSubsteps)
        );
    }
}

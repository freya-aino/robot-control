//! The simulator-side environment interface: reset, step, seeded
//! randomization.
//!
//! SPECS.md Section 9.1: "Implement simulator stepping, reset, batched
//! environments and independently seeded randomization." This trait is the
//! `robot-sim` half of that; reward terms and termination semantics are
//! `robot-tasks`' job (SPECS.md Section 6), so `step` here returns the raw
//! physical observation and a *physical* failure flag only (e.g. the state
//! leaving a representable region), not a reward.

use robot_types::rng::SeedStream;

/// The outcome of one simulation step, before any task-specific reward or
/// termination logic is applied.
#[derive(Clone, Debug)]
pub struct PhysicsStep {
    /// The raw observation after the step.
    pub observation: Vec<f64>,
    /// Whether the physics itself became unrepresentable (NaN, a body
    /// leaving a modeled region) — distinct from a task's own termination
    /// condition, which `robot-tasks` applies on top of this.
    pub physics_failure: bool,
}

/// A steppable, resettable simulated environment.
pub trait Environment {
    /// Width of the observation vector this environment produces.
    fn observation_width(&self) -> usize;

    /// Width of the action vector this environment accepts.
    fn action_width(&self) -> usize;

    /// Resets to a randomized initial state drawn from `seed`, writing the
    /// resulting observation into `observation`.
    fn reset(&mut self, seed: SeedStream, observation: &mut [f64]);

    /// Advances by one control step under `action`, returning the outcome.
    fn step(&mut self, action: &[f64], dt: robot_types::time::Duration) -> PhysicsStep;
}

/// A pendulum wrapped as an [`Environment`]: observation
/// `[theta, omega]`, action `[torque]`.
pub struct PendulumEnv {
    pendulum: crate::pendulum::Pendulum,
    params: crate::pendulum::PendulumParams,
    torque_limit: f64,
    theta_range: (f64, f64),
    omega_range: (f64, f64),
    rng: robot_types::rng::Rng,
}

impl PendulumEnv {
    /// Builds an environment resetting over the full angle range
    /// `(-pi, pi)` — the general swing-up task, where the policy must
    /// recover from an arbitrary starting orientation.
    #[must_use]
    pub fn new(
        params: crate::pendulum::PendulumParams,
        torque_limit: f64,
        seed: SeedStream,
    ) -> Self {
        Self::with_reset_range(
            params,
            torque_limit,
            (-core::f64::consts::PI, core::f64::consts::PI),
            (-1.0, 1.0),
            seed,
        )
    }

    /// Builds an environment resetting within an explicit `theta`/`omega`
    /// range, for a stabilization task that only needs to hold a region
    /// near one configuration rather than solve general swing-up (a
    /// substantially harder control problem a small reference task is not
    /// expected to solve).
    #[must_use]
    pub fn with_reset_range(
        params: crate::pendulum::PendulumParams,
        torque_limit: f64,
        theta_range: (f64, f64),
        omega_range: (f64, f64),
        seed: SeedStream,
    ) -> Self {
        Self {
            pendulum: crate::pendulum::Pendulum::new(
                params,
                crate::pendulum::PendulumState {
                    theta: 0.0,
                    omega: 0.0,
                },
            ),
            params,
            torque_limit,
            theta_range,
            omega_range,
            rng: robot_types::rng::Rng::new(seed),
        }
    }
}

impl Environment for PendulumEnv {
    fn observation_width(&self) -> usize {
        2
    }

    fn action_width(&self) -> usize {
        1
    }

    fn reset(&mut self, seed: SeedStream, observation: &mut [f64]) {
        self.rng = robot_types::rng::Rng::new(seed);
        let theta = self.rng.uniform(self.theta_range.0, self.theta_range.1);
        let omega = self.rng.uniform(self.omega_range.0, self.omega_range.1);
        self.pendulum = crate::pendulum::Pendulum::new(
            self.params,
            crate::pendulum::PendulumState { theta, omega },
        );
        let state = self.pendulum.state();
        observation[0] = state.theta;
        observation[1] = state.omega;
    }

    fn step(&mut self, action: &[f64], dt: robot_types::time::Duration) -> PhysicsStep {
        let torque = action[0].clamp(-self.torque_limit, self.torque_limit);
        let failed = self.pendulum.step(torque, dt).is_err();
        let state = self.pendulum.state();
        let failure = failed || !state.theta.is_finite() || !state.omega.is_finite();
        PhysicsStep {
            observation: vec![state.theta, state.omega],
            physics_failure: failure,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pendulum::PendulumParams;
    use robot_types::time::Duration;

    fn default_params() -> PendulumParams {
        PendulumParams {
            length: 1.0,
            mass: 1.0,
            gravity: 9.80665,
            damping: 0.1,
        }
    }

    #[test]
    fn reset_is_reproducible_under_the_same_seed() {
        let mut env_a = PendulumEnv::new(default_params(), 5.0, SeedStream::new(1, 0));
        let mut env_b = PendulumEnv::new(default_params(), 5.0, SeedStream::new(1, 0));
        let mut obs_a = [0.0; 2];
        let mut obs_b = [0.0; 2];
        env_a.reset(SeedStream::new(7, 3), &mut obs_a);
        env_b.reset(SeedStream::new(7, 3), &mut obs_b);
        assert_eq!(obs_a, obs_b);
    }

    #[test]
    fn different_seeds_diverge() {
        let mut env = PendulumEnv::new(default_params(), 5.0, SeedStream::new(1, 0));
        let mut obs_a = [0.0; 2];
        let mut obs_b = [0.0; 2];
        env.reset(SeedStream::new(1, 0), &mut obs_a);
        env.reset(SeedStream::new(2, 0), &mut obs_b);
        assert_ne!(obs_a, obs_b);
    }

    #[test]
    fn step_clamps_action_to_the_torque_limit_and_reports_no_failure() {
        let mut env = PendulumEnv::new(default_params(), 1.0, SeedStream::new(1, 0));
        let mut obs = [0.0; 2];
        env.reset(SeedStream::new(1, 0), &mut obs);
        let outcome = env.step(&[1000.0], Duration::from_millis(10));
        assert!(!outcome.physics_failure);
        assert_eq!(outcome.observation.len(), 2);
    }

    #[test]
    fn widths_are_fixed_and_correct() {
        let env = PendulumEnv::new(default_params(), 1.0, SeedStream::new(1, 0));
        assert_eq!(env.observation_width(), 2);
        assert_eq!(env.action_width(), 1);
    }
}

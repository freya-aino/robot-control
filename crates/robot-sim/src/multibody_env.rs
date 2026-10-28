//! An articulated body wrapped as a [`Backend`], including floating-base bodies.
//!
//! This is the multibody half of the backend seam. It steps the same
//! [`MultibodyState::step`] that `robot sim multibody` uses without
//! `--floating`, so a caller going through the trait gets exactly that
//! free-dynamics trajectory. Observation is `q` followed by `qd`; action is
//! one torque per degree of freedom.

use crate::articulated::{ArticulatedBody, MultibodyState};
use crate::backend::Backend;
use crate::env::{Environment, PhysicsStep};
use robot_types::frames::Vec3;
use robot_types::rng::{Rng, SeedStream};
use robot_types::time::Duration;

/// A configuration vector whose length does not match the body's position
/// width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WidthMismatch {
    /// The body's position width.
    pub expected: usize,
    /// The length that was supplied.
    pub found: usize,
}

/// A multibody steppable through [`Environment`] and [`Backend`].
///
/// `reset` draws each initial joint position uniformly within `q_spread` of
/// `q_home`, so a seed fixes the starting configuration.
pub struct MultibodyEnv {
    body: ArticulatedBody,
    gravity: Vec3,
    q_home: Vec<f64>,
    q_spread: f64,
    state: MultibodyState,
    rng: Rng,
}

impl MultibodyEnv {
    /// Builds the environment at rest at `q_home`. `q_home` must have one
    /// entry per position slot of `body`; any other length is rejected here
    /// rather than panicking on the first observation.
    pub fn new(
        body: ArticulatedBody,
        gravity: Vec3,
        q_home: Vec<f64>,
        q_spread: f64,
        seed: SeedStream,
    ) -> Result<Self, WidthMismatch> {
        if q_home.len() != body.position_width() {
            return Err(WidthMismatch {
                expected: body.position_width(),
                found: q_home.len(),
            });
        }
        let dof = body.dof();
        Ok(Self {
            state: MultibodyState::at_rest(q_home.clone(), dof),
            body,
            gravity,
            q_home,
            q_spread,
            rng: Rng::new(seed),
        })
    }

    /// The current joint state.
    #[must_use]
    pub fn state(&self) -> &MultibodyState {
        &self.state
    }

    fn write_observation(&self, observation: &mut [f64]) {
        let (q_out, qd_out) = observation.split_at_mut(self.state.q.len());
        q_out.copy_from_slice(&self.state.q);
        qd_out.copy_from_slice(&self.state.qd);
    }
}

impl Environment for MultibodyEnv {
    fn observation_width(&self) -> usize {
        self.body.position_width() + self.body.dof()
    }

    fn action_width(&self) -> usize {
        self.body.dof()
    }

    fn reset(&mut self, seed: SeedStream, observation: &mut [f64]) {
        self.rng = Rng::new(seed);
        let q: Vec<f64> = self
            .q_home
            .iter()
            .map(|&home| home + self.rng.uniform(-self.q_spread, self.q_spread))
            .collect();
        self.state = MultibodyState::at_rest(q, self.body.dof());
        self.write_observation(observation);
    }

    fn step(&mut self, action: &[f64], dt: Duration) -> PhysicsStep {
        let stepped = self.state.step(&self.body, action, self.gravity, dt);
        let finite = self
            .state
            .q
            .iter()
            .chain(&self.state.qd)
            .all(|v| v.is_finite());
        let mut observation = vec![0.0; self.observation_width()];
        self.write_observation(&mut observation);
        PhysicsStep {
            observation,
            physics_failure: stepped.is_err() || !finite,
        }
    }
}

impl Backend for MultibodyEnv {
    fn backend_name(&self) -> &'static str {
        "cpu-multibody"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::articulated::ArticulatedBody;
    use robot_model::profile::RobotProfile;

    const G: f64 = 9.806_65;

    fn env_for(profile: &RobotProfile, seed: SeedStream) -> MultibodyEnv {
        let body = ArticulatedBody::from_profile(profile);
        let q_home = vec![0.0; body.position_width()];
        MultibodyEnv::new(body, Vec3::new(0.0, 0.0, -G), q_home, 0.1, seed).unwrap()
    }

    /// The reference path: no `Environment`, no `Backend`, just the state
    /// stepped by hand against the same body, the way `robot sim multibody`
    /// does it.
    fn reference_trajectory(
        profile: &RobotProfile,
        q_home: Vec<f64>,
        actions: &[Vec<f64>],
        dt: Duration,
    ) -> Vec<Vec<f64>> {
        let body = ArticulatedBody::from_profile(profile);
        let mut state = MultibodyState::at_rest(q_home, body.dof());
        actions
            .iter()
            .map(|a| {
                state
                    .step(&body, a, Vec3::new(0.0, 0.0, -G), dt)
                    .expect("reference step");
                state.q.iter().chain(&state.qd).copied().collect()
            })
            .collect()
    }

    fn zero_spread_env(profile: &RobotProfile, q_home: Vec<f64>) -> MultibodyEnv {
        let body = ArticulatedBody::from_profile(profile);
        MultibodyEnv::new(
            body,
            Vec3::new(0.0, 0.0, -G),
            q_home,
            0.0,
            SeedStream::new(1, 0),
        )
        .unwrap()
    }

    #[test]
    fn stepping_through_the_trait_object_matches_a_hand_driven_state() {
        let profile = crate::articulated::tests::double_pendulum(1.0, 0.5, 1.0, 0.5);
        let q_home = vec![0.3, -0.2];
        let actions: Vec<Vec<f64>> = (0..50)
            .map(|i| vec![0.1 * f64::from(i).sin(), 0.05])
            .collect();
        let dt = Duration::from_millis(1);

        let expected = reference_trajectory(&profile, q_home.clone(), &actions, dt);

        let mut env = zero_spread_env(&profile, q_home);
        let backend: &mut dyn Backend = &mut env;
        let mut obs = vec![0.0; backend.observation_width()];
        backend.reset(SeedStream::new(11, 0), &mut obs);
        let via_trait: Vec<Vec<f64>> = actions
            .iter()
            .map(|a| {
                let outcome = backend.step(a, dt);
                assert!(!outcome.physics_failure);
                outcome.observation
            })
            .collect();

        assert_eq!(expected, via_trait);
        assert_ne!(expected[0], expected[49], "the pendulum must actually move");
    }

    #[test]
    fn reset_with_the_same_seed_reproduces_the_trajectory_and_other_seeds_differ() {
        let profile = crate::articulated::tests::double_pendulum(1.0, 0.5, 1.0, 0.5);
        let body = ArticulatedBody::from_profile(&profile);
        let mut env = MultibodyEnv::new(
            body,
            Vec3::new(0.0, 0.0, -G),
            vec![0.0; 2],
            0.1,
            SeedStream::new(1, 0),
        )
        .unwrap();
        let mut first = vec![0.0; 4];
        let mut again = vec![0.0; 4];
        let mut other = vec![0.0; 4];
        env.reset(SeedStream::new(3, 0), &mut first);
        env.step(&[0.0, 0.0], Duration::from_millis(1));
        env.reset(SeedStream::new(3, 0), &mut again);
        env.reset(SeedStream::new(4, 0), &mut other);
        assert_eq!(first, again);
        assert_ne!(first, other);
        assert_eq!(&first[2..], &[0.0, 0.0], "reset starts at rest");
    }

    #[test]
    fn a_home_posture_of_the_wrong_length_is_rejected_at_construction() {
        let profile = crate::articulated::tests::double_pendulum(1.0, 0.5, 1.0, 0.5);
        let body = ArticulatedBody::from_profile(&profile);
        let result = MultibodyEnv::new(
            body,
            Vec3::new(0.0, 0.0, -G),
            vec![0.0; 3],
            0.0,
            SeedStream::new(1, 0),
        );
        assert_eq!(
            result.err(),
            Some(WidthMismatch {
                expected: 2,
                found: 3,
            })
        );
    }

    #[test]
    fn reports_its_backend_name_and_widths() {
        let profile = crate::articulated::tests::double_pendulum(1.0, 0.5, 1.0, 0.5);
        let env = env_for(&profile, SeedStream::new(1, 0));
        assert_eq!(env.backend_name(), "cpu-multibody");
        assert_eq!(env.action_width(), 2);
        assert_eq!(env.observation_width(), 4);
    }

    #[test]
    fn a_non_finite_torque_is_reported_as_physics_failure_not_a_panic() {
        let profile = crate::articulated::tests::double_pendulum(1.0, 0.5, 1.0, 0.5);
        let mut env = env_for(&profile, SeedStream::new(1, 0));
        let mut obs = vec![0.0; env.observation_width()];
        env.reset(SeedStream::new(2, 0), &mut obs);
        let before = env.state().clone();
        let outcome = env.step(&[f64::NAN, 0.0], Duration::from_millis(1));
        assert!(outcome.physics_failure);
        assert!(outcome.observation.iter().all(|v| v.is_finite()));
        assert_eq!(
            env.state().q,
            before.q,
            "a rejected step leaves the state alone"
        );
    }

    #[test]
    fn a_wrong_width_action_is_reported_as_physics_failure_not_a_panic() {
        let profile = crate::articulated::tests::double_pendulum(1.0, 0.5, 1.0, 0.5);
        let mut env = env_for(&profile, SeedStream::new(1, 0));
        let outcome = env.step(&[0.0], Duration::from_millis(1));
        assert!(outcome.physics_failure);
    }

    /// The floating G1 without ground: the free-dynamics reference, which
    /// falls without bound. This is the no-contact path and must keep doing so.
    #[test]
    fn the_floating_g1_without_ground_falls_through_the_backend() {
        let profile = robot_model::fixtures::unitree_g1_29dof_floating_base().unwrap();
        let body = ArticulatedBody::from_profile(&profile);
        let mut q_home = vec![0.01; body.position_width()];
        q_home[3] = 1.0;
        let gravity = profile.gravity.acceleration();
        let mut env = MultibodyEnv::new(body, gravity, q_home, 0.0, SeedStream::new(1, 0)).unwrap();
        let mut obs = vec![0.0; env.observation_width()];
        env.reset(SeedStream::new(1, 0), &mut obs);
        let start_height = obs[2];
        let dof = env.action_width();
        let mut height = start_height;
        for _ in 0..200 {
            let outcome = env.step(&vec![0.0; dof], Duration::from_secs_f64(0.0001));
            assert!(!outcome.physics_failure);
            height = outcome.observation[2];
        }
        assert!(
            height < start_height,
            "free fall: {start_height} -> {height}"
        );
        assert_eq!(env.backend_name(), "cpu-multibody");
    }
}

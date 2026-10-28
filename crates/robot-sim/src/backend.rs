//! The backend seam: an [`Environment`] that also says which physics backend
//! produced its steps.
//!
//! ADR 0005 §6 requires every crate outside `robot-sim` to depend only on
//! [`Environment`], never on a backend-specific type. A future GPU-batched
//! backend must sit underneath that same interface, so [`Backend`] adds one
//! queryable identity rather than a second stepping API. The identity is
//! exposed on purpose: a backend-specific behaviour difference should be
//! visible to a caller, not hidden behind the shared trait.

use crate::env::{Environment, PendulumEnv};

/// An [`Environment`] that reports which backend implements it.
pub trait Backend: Environment {
    /// A stable identifier for the backend, for reports and comparisons.
    fn backend_name(&self) -> &'static str;
}

impl Backend for PendulumEnv {
    fn backend_name(&self) -> &'static str {
        "cpu-pendulum"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pendulum::PendulumParams;
    use robot_types::rng::SeedStream;

    fn params() -> PendulumParams {
        PendulumParams {
            mass: 1.0,
            length: 1.0,
            damping: 0.0,
            gravity: 9.81,
        }
    }

    fn dt() -> robot_types::time::Duration {
        robot_types::time::Duration::from_millis(10)
    }

    #[test]
    fn pendulum_reports_a_stable_backend_name() {
        let env = PendulumEnv::new(params(), 2.0, SeedStream::new(5, 0));
        assert_eq!(env.backend_name(), "cpu-pendulum");
    }

    #[test]
    fn stepping_through_the_trait_object_matches_stepping_the_concrete_type() {
        let actions = [0.0, 0.5, -0.5, 1.0];
        let mut direct = PendulumEnv::new(params(), 2.0, SeedStream::new(5, 0));
        let mut obs = [0.0; 2];
        direct.reset(SeedStream::new(5, 0), &mut obs);
        let direct_trace: Vec<Vec<f64>> = actions
            .iter()
            .map(|&a| Environment::step(&mut direct, &[a], dt()).observation)
            .collect();

        let mut erased = PendulumEnv::new(params(), 2.0, SeedStream::new(5, 0));
        let backend: &mut dyn Backend = &mut erased;
        let mut obs = [0.0; 2];
        backend.reset(SeedStream::new(5, 0), &mut obs);
        let backend_trace: Vec<Vec<f64>> = actions
            .iter()
            .map(|&a| backend.step(&[a], dt()).observation)
            .collect();

        assert_eq!(direct_trace, backend_trace);
    }
}

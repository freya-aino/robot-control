//! A pendulum balance task: reward and termination on top of
//! `robot-sim::env::PendulumEnv`'s raw physics.
//!
//! SPECS.md Section 9.2: "Train balance and locomotion first." This is the
//! smallest instance of that on the smallest environment this workspace
//! has — the one PLAN.md's first executable backlog item 8 names.

use crate::reward::{RewardRegistry, RewardTermSpec, TaskContext};
use robot_types::transition::{RewardTerms, Termination};

/// Configuration for the balance task's termination and reward shaping.
#[derive(Clone, Copy, Debug)]
pub struct BalanceTaskConfig {
    /// Angular distance from upright (`theta = pi`), in radians, beyond
    /// which the episode terminates as a failure.
    pub fall_angle: f64,
    /// Episode length limit, in steps, after which the episode truncates
    /// (not a failure — SPECS.md Section 9.2's termination/truncation
    /// distinction).
    pub max_steps: u32,
    /// Weight on the squared action, penalizing effort.
    pub effort_weight: f64,
}

impl Default for BalanceTaskConfig {
    fn default() -> Self {
        Self {
            fall_angle: 2.5,
            max_steps: 500,
            effort_weight: 0.01,
        }
    }
}

/// Reward terms and termination for "balance the pendulum upright."
///
/// Observation layout matches `PendulumEnv`: `[theta, omega]`, with
/// `theta = 0` upright... no — `Pendulum::angular_acceleration`'s gravity
/// term uses `theta = 0` as the *downward* vertical (see `pendulum.rs`), so
/// "upright" here is `theta = pi`. The reward term names this explicitly
/// rather than silently assuming a different convention than the physics
/// module actually uses.
pub struct BalanceTask {
    config: BalanceTaskConfig,
    registry: RewardRegistry,
}

fn upright_term(ctx: &TaskContext<'_>) -> f64 {
    // theta measured from the downward vertical; upright is theta = pi.
    let theta = ctx.next_observation[0];
    let distance_from_upright = (theta - core::f64::consts::PI).rem_euclid(core::f64::consts::TAU);
    let wrapped = if distance_from_upright > core::f64::consts::PI {
        core::f64::consts::TAU - distance_from_upright
    } else {
        distance_from_upright
    };
    -(wrapped * wrapped)
}

fn effort_term(ctx: &TaskContext<'_>) -> f64 {
    -ctx.action.iter().map(|a| a * a).sum::<f64>()
}

impl BalanceTask {
    /// Builds the task with its default two-term registry (upright, effort).
    #[must_use]
    pub fn new(config: BalanceTaskConfig) -> Self {
        let effort_weight = config.effort_weight;
        let registry = RewardRegistry::new(vec![
            RewardTermSpec {
                name: "upright-v1",
                weight: 1.0,
                compute: upright_term,
                is_style_term: false,
            },
            RewardTermSpec {
                name: "effort-v1",
                weight: effort_weight,
                compute: effort_term,
                is_style_term: false,
            },
        ])
        .expect("the two built-in term names are distinct");
        Self { config, registry }
    }

    /// The reward registry, exposed for a caller that wants to log term
    /// names before the first episode runs.
    #[must_use]
    pub fn registry(&self) -> &RewardRegistry {
        &self.registry
    }

    /// Evaluates reward and termination for one step.
    #[must_use]
    pub fn evaluate(
        &self,
        observation: &[f64],
        action: &[f64],
        next_observation: &[f64],
        step_count: u32,
        physics_failure: bool,
    ) -> (RewardTerms, Termination) {
        let ctx = TaskContext {
            observation,
            action,
            next_observation,
        };
        let terms = self.registry.evaluate(&ctx);

        let theta = next_observation[0];
        let distance_from_upright = {
            let raw = (theta - core::f64::consts::PI).rem_euclid(core::f64::consts::TAU);
            if raw > core::f64::consts::PI {
                core::f64::consts::TAU - raw
            } else {
                raw
            }
        };
        let fallen = physics_failure || distance_from_upright > self.config.fall_angle;
        let termination = if fallen {
            Termination::Terminal
        } else if step_count >= self.config.max_steps {
            Termination::TimeLimit
        } else {
            Termination::Continuing
        };
        (terms, termination)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upright_is_the_maximum_of_the_upright_term() {
        let ctx = TaskContext {
            observation: &[],
            action: &[],
            next_observation: &[core::f64::consts::PI],
        };
        assert!((upright_term(&ctx)).abs() < 1e-12);
        let ctx_down = TaskContext {
            observation: &[],
            action: &[],
            next_observation: &[0.0],
        };
        assert!(upright_term(&ctx_down) < upright_term(&ctx));
    }

    #[test]
    fn upright_term_is_periodic_and_symmetric() {
        let a = TaskContext {
            observation: &[],
            action: &[],
            next_observation: &[core::f64::consts::PI + 0.3],
        };
        let b = TaskContext {
            observation: &[],
            action: &[],
            next_observation: &[core::f64::consts::PI - 0.3],
        };
        assert!((upright_term(&a) - upright_term(&b)).abs() < 1e-9);
    }

    #[test]
    fn falling_past_the_configured_angle_terminates() {
        let task = BalanceTask::new(BalanceTaskConfig::default());
        let (_, termination) = task.evaluate(&[0.0, 0.0], &[0.0], &[0.1], 1, false);
        assert_eq!(termination, Termination::Terminal);
    }

    #[test]
    fn remaining_upright_within_the_fall_angle_continues() {
        let task = BalanceTask::new(BalanceTaskConfig::default());
        let (_, termination) = task.evaluate(
            &[core::f64::consts::PI, 0.0],
            &[0.0],
            &[core::f64::consts::PI],
            1,
            false,
        );
        assert_eq!(termination, Termination::Continuing);
    }

    #[test]
    fn reaching_the_step_limit_truncates_rather_than_terminates() {
        let config = BalanceTaskConfig {
            max_steps: 5,
            ..BalanceTaskConfig::default()
        };
        let task = BalanceTask::new(config);
        let (_, termination) = task.evaluate(
            &[core::f64::consts::PI, 0.0],
            &[0.0],
            &[core::f64::consts::PI],
            5,
            false,
        );
        assert_eq!(termination, Termination::TimeLimit);
    }

    #[test]
    fn a_physics_failure_always_terminates_even_near_upright() {
        let task = BalanceTask::new(BalanceTaskConfig::default());
        let (_, termination) = task.evaluate(
            &[core::f64::consts::PI, 0.0],
            &[0.0],
            &[core::f64::consts::PI],
            1,
            true,
        );
        assert_eq!(termination, Termination::Terminal);
    }

    #[test]
    fn effort_is_penalized_and_logged_as_its_own_term() {
        let task = BalanceTask::new(BalanceTaskConfig::default());
        let (terms, _) = task.evaluate(
            &[core::f64::consts::PI, 0.0],
            &[2.0],
            &[core::f64::consts::PI],
            1,
            false,
        );
        let effort = terms.get("effort-v1").unwrap();
        assert!((effort.value - (-4.0)).abs() < 1e-12);
    }
}

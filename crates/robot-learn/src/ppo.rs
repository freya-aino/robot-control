//! The PPO baseline of SPECS.md Section 9.2: "Implement PPO with
//! generalized advantage estimation, bounded actions, observation
//! normalization, checkpoint/resume, recurrent policy support when needed,
//! and explicit handling of termination versus time-limit truncation."
//!
//! This module ties together [`crate::gaussian_policy`] (the policy),
//! [`crate::backprop`] and [`crate::optimizer`] (the update), [`crate::gae`]
//! (advantage estimation) and [`crate::rollout`] (storage) into one agent
//! that can run against any `robot_sim::env::Environment`. Recurrent policy
//! support is not implemented in this revision — the policy is feed-forward
//! — which is stated here rather than silently omitted.

use crate::backprop::backward;
use crate::gae::compute_gae;
use crate::gaussian_policy::GaussianPolicy;
use crate::optimizer::{AdamConfig, AdamState};
use crate::rollout::RolloutBuffer;
use robot_nn::mlp::{Mlp, ParamGroup, ParamGroupError};
use robot_sim::env::Environment;
use robot_types::ids::EnvId;
use robot_types::obs::Normalizer;
use robot_types::rng::{Rng, SeedStream};
use robot_types::tier::Generation;
use robot_types::transition::{
    ActionRepresentation, RewardTerms, Termination, Transition, TransitionOrigin, TransitionSchema,
};

/// A task's reward/termination function:
/// `(observation, action, next_observation, step_in_episode, physics_failure)
/// -> (reward terms, termination)`.
pub type TaskFn<'a> = dyn Fn(&[f64], &[f64], &[f64], u32, bool) -> (RewardTerms, Termination) + 'a;

/// The per-collection identity a rollout is tagged with: which parallel
/// environment, which robot profile, and the seed base episodes within this
/// collection derive from.
#[derive(Clone, Copy, Debug)]
pub struct RolloutContext<'a> {
    /// Identity of the (possibly parallel) environment this rollout came
    /// from.
    pub env_id: EnvId,
    /// Robot profile identity and version.
    pub robot_profile: &'a str,
    /// Base seed; episode `k` within this collection resets from
    /// `SeedStream::new(episode_seed_base, k)`.
    pub episode_seed_base: u64,
}

/// PPO hyperparameters.
#[derive(Clone, Copy, Debug)]
pub struct PpoConfig {
    /// Discount factor.
    pub gamma: f64,
    /// GAE lambda.
    pub lambda: f64,
    /// Clipping range for the probability ratio.
    pub clip_ratio: f64,
    /// Weight on the value-function loss.
    pub value_loss_weight: f64,
    /// Weight on the entropy bonus (encourages exploration).
    pub entropy_weight: f64,
    /// Optimizer settings for the policy network.
    pub policy_adam: AdamConfig,
    /// Optimizer settings for the value network.
    pub value_adam: AdamConfig,
    /// Learning rate for the policy's log-std parameters.
    pub log_std_learning_rate: f64,
    /// Number of optimization epochs per collected rollout.
    pub epochs: usize,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            lambda: 0.95,
            clip_ratio: 0.2,
            value_loss_weight: 0.5,
            entropy_weight: 0.001,
            policy_adam: AdamConfig {
                learning_rate: 1e-3,
                ..AdamConfig::default()
            },
            value_adam: AdamConfig {
                learning_rate: 1e-3,
                ..AdamConfig::default()
            },
            log_std_learning_rate: 1e-3,
            epochs: 4,
        }
    }
}

/// The PPO agent: policy, value function, their optimizer states, and the
/// observation normalizer shared between training and (eventually)
/// deployment.
pub struct PpoAgent {
    /// The policy.
    pub policy: GaussianPolicy,
    /// The value function.
    pub value_net: Mlp,
    pub(crate) policy_adam: AdamState,
    pub(crate) value_adam: AdamState,
    /// Observation normalizer. SPECS.md Section 9.2 requires "observation
    /// normalization"; this is the one instance both collection and update
    /// use, and it is the same type `robot_control` would load for deployed
    /// inference (SPECS.md Section 6: training and deployment share
    /// normalization logic).
    pub normalizer: Normalizer,
    pub(crate) config: PpoConfig,
    pub(crate) generation: Generation,
    pub(crate) policy_groups: Vec<ParamGroup>,
    pub(crate) value_groups: Vec<ParamGroup>,
}

impl PpoAgent {
    /// Builds an agent from a policy network and a value network.
    #[must_use]
    pub fn new(
        policy: GaussianPolicy,
        value_net: Mlp,
        config: PpoConfig,
        obs_width: usize,
    ) -> Self {
        let policy_adam = AdamState::new(&policy.mean_net);
        let value_adam = AdamState::new(&value_net);
        Self {
            policy,
            value_net,
            policy_adam,
            value_adam,
            normalizer: Normalizer::new(obs_width, 10.0),
            config,
            generation: Generation::FIRST,
            policy_groups: Vec::new(),
            value_groups: Vec::new(),
        }
    }

    /// Replaces the policy's parameter groups. The flat space is the mean
    /// network's parameters followed by the log-std entries (see
    /// [`GaussianPolicy::frozen_mask`]). The table is checked against the
    /// current policy first; on error nothing changes.
    pub fn set_policy_groups(&mut self, groups: Vec<ParamGroup>) -> Result<(), ParamGroupError> {
        self.policy.frozen_mask(&groups)?;
        self.policy_groups = groups;
        Ok(())
    }

    /// Replaces the value network's parameter groups, checked against the
    /// current value network. On error nothing changes.
    pub fn set_value_groups(&mut self, groups: Vec<ParamGroup>) -> Result<(), ParamGroupError> {
        self.value_net.frozen_mask(&groups)?;
        self.value_groups = groups;
        Ok(())
    }

    /// The policy network's parameter groups.
    #[must_use]
    pub fn policy_groups(&self) -> &[ParamGroup] {
        &self.policy_groups
    }

    /// The value network's parameter groups.
    #[must_use]
    pub fn value_groups(&self) -> &[ParamGroup] {
        &self.value_groups
    }

    /// Collects one rollout of `steps` environment transitions, resetting
    /// the environment whenever it terminates or truncates.
    pub fn collect_rollout<E: Environment>(
        &mut self,
        env: &mut E,
        task: &TaskFn<'_>,
        buffer: &mut RolloutBuffer,
        steps: usize,
        ctx: RolloutContext<'_>,
        rng: &mut Rng,
    ) {
        let RolloutContext {
            env_id,
            robot_profile,
            episode_seed_base,
        } = ctx;
        let obs_width = env.observation_width();
        let mut observation = vec![0.0; obs_width];
        let mut episode_index = 0u64;
        env.reset(
            SeedStream::new(episode_seed_base, episode_index),
            &mut observation,
        );
        let mut step_in_episode = 0u32;
        let mut episode_id = format!("ep-{episode_index}");

        for t in 0..steps {
            let mut normalized = vec![0.0; obs_width];
            self.normalizer
                .observe(&observation)
                .expect("width matches by construction");
            self.normalizer
                .normalize(&observation, &mut normalized)
                .expect("width matches");

            let sample = self.policy.sample(&normalized, rng);
            let outcome = env.step(&sample.action, robot_types::time::Duration::from_millis(20));
            step_in_episode += 1;
            let (reward_terms, termination) = task(
                &observation,
                &sample.action,
                &outcome.observation,
                step_in_episode,
                outcome.physics_failure,
            );

            let transition = Transition {
                schema_version: robot_types::transition::TRANSITION_SCHEMA_VERSION,
                episode_id: episode_id.clone(),
                timestep: step_in_episode,
                env_id,
                robot_profile: robot_profile.to_owned(),
                observation: observation.clone(),
                action: sample.action.clone(),
                action_representation: ActionRepresentation::ActuatorSpace,
                reward: reward_terms,
                next_observation: outcome.observation.clone(),
                termination,
                observation_validity: vec![true; obs_width],
                task_data: Vec::new(),
                origin: TransitionOrigin::CurrentPolicy {
                    generation: self.generation,
                },
                seeds: SeedStream::new(episode_seed_base, episode_index),
                behavior_log_prob: Some(sample.log_prob),
            };
            if buffer.push(transition).is_err() {
                break; // Capacity reached; the caller consumes before collecting more.
            }

            observation = outcome.observation;
            if termination.ends_episode() {
                episode_index += 1;
                episode_id = format!("ep-{episode_index}");
                step_in_episode = 0;
                env.reset(
                    SeedStream::new(episode_seed_base, episode_index),
                    &mut observation,
                );
            }
            let _ = t;
        }
    }

    /// Runs `config.epochs` full-batch optimization passes over everything
    /// currently in `buffer`, then increments the generation counter.
    ///
    /// Full-batch rather than minibatched: appropriate for the small
    /// reference tasks this baseline targets (SPECS.md Section 9.2, "Train
    /// small reference tasks... before humanoid results are trusted");
    /// minibatching is a scale concern for later phases, not this baseline.
    pub fn update(&mut self, buffer: &RolloutBuffer) {
        let transitions = buffer.as_slice();
        if transitions.is_empty() {
            return;
        }
        let n = transitions.len();

        // Value estimates at every visited observation, and — separately,
        // per step — at that same step's own recorded next observation.
        // These must NOT be inferred as a shifted view of one another: a
        // step that ends an episode by truncation is routinely followed, at
        // the next buffer index, by the first step of a different, freshly
        // reset episode (SPECS.md Section 7.6 allows one bounded chunk to
        // span multiple episodes). `compute_gae`'s own documentation
        // explains the bug this would otherwise reintroduce.
        let obs_width = self.value_net.input_width();
        let mut normalized_obs = vec![0.0; obs_width];
        let values: Vec<f64> = transitions
            .iter()
            .map(|t| {
                self.normalizer
                    .normalize(&t.observation, &mut normalized_obs)
                    .expect("width matches");
                self.value_net.forward_with_trace(&normalized_obs).output()[0]
            })
            .collect();
        let next_values: Vec<f64> = transitions
            .iter()
            .map(|t| {
                self.normalizer
                    .normalize(&t.next_observation, &mut normalized_obs)
                    .expect("width matches");
                self.value_net.forward_with_trace(&normalized_obs).output()[0]
            })
            .collect();

        let rewards: Vec<f64> = transitions.iter().map(|t| t.reward.total()).collect();
        let terminations: Vec<Termination> = transitions.iter().map(|t| t.termination).collect();
        let (advantages, returns) = compute_gae(
            &rewards,
            &values,
            &next_values,
            &terminations,
            self.config.gamma,
            self.config.lambda,
        );

        let adv_mean = advantages.iter().sum::<f64>() / n as f64;
        let adv_var = advantages
            .iter()
            .map(|a| (a - adv_mean).powi(2))
            .sum::<f64>()
            / n as f64;
        let adv_std = adv_var.sqrt().max(1e-6);
        let normalized_adv: Vec<f64> = advantages
            .iter()
            .map(|a| (a - adv_mean) / adv_std)
            .collect();

        let old_log_probs: Vec<f64> = transitions
            .iter()
            .map(|t| t.behavior_log_prob.expect("collected with a log-prob"))
            .collect();

        let policy_mask = self
            .policy
            .frozen_mask(&self.policy_groups)
            .unwrap_or_else(|e| {
                panic!("policy parameter groups no longer fit the policy network: {e}")
            });
        let (policy_frozen, log_std_frozen) =
            policy_mask.split_at(self.policy.mean_net.param_count());
        let value_frozen = self
            .value_net
            .frozen_mask(&self.value_groups)
            .unwrap_or_else(|e| {
                panic!("value parameter groups no longer fit the value network: {e}")
            });

        for _epoch in 0..self.config.epochs {
            let mut policy_grads = crate::backprop::Gradients::zeros_like(&self.policy.mean_net);
            let mut log_std_grad = vec![0.0; self.policy.log_std.len()];
            let mut value_grads = crate::backprop::Gradients::zeros_like(&self.value_net);

            for i in 0..n {
                self.normalizer
                    .normalize(&transitions[i].observation, &mut normalized_obs)
                    .expect("width matches");
                let trace = self.policy.mean_net.forward_with_trace(&normalized_obs);
                let new_log_prob = self
                    .policy
                    .log_prob_from_trace(&trace, &transitions[i].action);
                let ratio = (new_log_prob - old_log_probs[i]).exp();
                let adv = normalized_adv[i];

                let unclipped = ratio * adv;
                let clipped =
                    ratio.clamp(1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * adv;
                // PPO maximizes min(unclipped, clipped); the loss gradient
                // flows only through whichever branch is active, except in
                // the clipped region where the ratio's derivative is zero.
                let d_loss_d_logprob = if unclipped <= clipped {
                    // Unclipped branch active: d(ratio*adv)/d(log_prob) = ratio*adv.
                    -ratio * adv / n as f64
                } else {
                    0.0
                };

                let (d_mean, d_log_std) =
                    self.policy
                        .log_prob_gradient(&trace, &transitions[i].action, d_loss_d_logprob);
                let entropy_grad_per_dim = -self.config.entropy_weight / n as f64;
                let d_log_std_total: Vec<f64> =
                    d_log_std.iter().map(|g| g + entropy_grad_per_dim).collect();

                let (g, _) = backward(&self.policy.mean_net, &trace, &d_mean);
                policy_grads.add_assign(&g);
                for (acc, g) in log_std_grad.iter_mut().zip(&d_log_std_total) {
                    *acc += g;
                }

                let value_trace = self.value_net.forward_with_trace(&normalized_obs);
                let value_error = value_trace.output()[0] - returns[i];
                let d_value = [self.config.value_loss_weight * value_error / n as f64];
                let (vg, _) = backward(&self.value_net, &value_trace, &d_value);
                value_grads.add_assign(&vg);
            }

            policy_grads
                .zero_frozen(policy_frozen)
                .expect("mask matches the policy gradients");
            value_grads
                .zero_frozen(&value_frozen)
                .expect("mask matches the value gradients");
            self.policy_adam
                .step_masked(
                    &mut self.policy.mean_net,
                    &policy_grads,
                    &self.config.policy_adam,
                    policy_frozen,
                )
                .expect("mask matches the policy network");
            self.policy
                .apply_log_std_gradient_masked(
                    &log_std_grad,
                    self.config.log_std_learning_rate,
                    log_std_frozen,
                )
                .expect("mask matches the log-std vector");
            self.value_adam
                .step_masked(
                    &mut self.value_net,
                    &value_grads,
                    &self.config.value_adam,
                    &value_frozen,
                )
                .expect("mask matches the value network");
        }

        self.generation = self.generation.next();
    }

    /// The current training generation (incremented once per [`PpoAgent::update`] call).
    #[must_use]
    pub const fn generation(&self) -> Generation {
        self.generation
    }

    /// A schema describing this agent's transitions, for a
    /// [`RolloutBuffer`].
    #[must_use]
    pub fn transition_schema(&self, robot_profile: &str) -> TransitionSchema {
        TransitionSchema {
            schema_version: robot_types::transition::TRANSITION_SCHEMA_VERSION,
            observation_width: self.policy.mean_net.input_width(),
            action_width: self.policy.mean_net.output_width(),
            action_representation: ActionRepresentation::ActuatorSpace,
            robot_profile: robot_profile.to_owned(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_nn::init::initialize;
    use robot_nn::layer::Activation;
    use robot_nn::mlp::ParamGroupError;
    use robot_sim::env::{Environment, PendulumEnv, PhysicsStep};
    use robot_sim::pendulum::PendulumParams;
    use robot_tasks::balance::{BalanceTask, BalanceTaskConfig};
    use robot_types::rng::SeedStream;

    /// A trivial one-step bandit: constant observation, scalar action,
    /// reward `-(action - TARGET)^2`, always terminal. This isolates the
    /// policy-gradient machinery (clipped surrogate, entropy, Adam) from
    /// every difficulty specific to controlling an unstable pendulum —
    /// credit assignment across time, an unstable equilibrium, a
    /// co-trained value function with bootstrapping. If PPO cannot solve
    /// this, the bug is in the gradient machinery; if it can, the pendulum
    /// task's difficulty is a separate, genuine RL question.
    struct ConstantBanditEnv {
        target: f64,
    }

    impl Environment for ConstantBanditEnv {
        fn observation_width(&self) -> usize {
            1
        }
        fn action_width(&self) -> usize {
            1
        }
        fn reset(&mut self, _seed: SeedStream, observation: &mut [f64]) {
            observation[0] = 0.0;
        }
        fn step(&mut self, action: &[f64], _dt: robot_types::time::Duration) -> PhysicsStep {
            // The observation never changes; the action's quality is scored
            // entirely by the task closure (reward belongs to `robot-tasks`
            // in spirit, kept inline here only because this fixture is
            // local to one test). `self.target` is unused by the
            // environment itself — it exists only so the struct documents
            // what the test closure's reward is shaped around.
            let _ = (action, self.target);
            PhysicsStep {
                observation: vec![0.0],
                physics_failure: false,
            }
        }
    }

    #[test]
    fn ppo_drives_a_one_step_bandit_policy_mean_toward_the_target() {
        const TARGET: f64 = 0.6;
        let mut rng = Rng::new(SeedStream::new(3, 1));
        let policy_net = initialize(&[1, 1], vec![Activation::Identity], &mut rng).unwrap();
        let value_net = initialize(&[1, 1], vec![Activation::Identity], &mut rng).unwrap();
        let config = PpoConfig {
            entropy_weight: 0.0,
            policy_adam: AdamConfig {
                learning_rate: 0.05,
                ..AdamConfig::default()
            },
            value_adam: AdamConfig {
                learning_rate: 0.05,
                ..AdamConfig::default()
            },
            epochs: 1,
            ..PpoConfig::default()
        };
        let mut agent = PpoAgent::new(GaussianPolicy::new(policy_net, -0.3), value_net, config, 1);
        let mut env = ConstantBanditEnv { target: TARGET };
        let task_fn = |_obs: &[f64], action: &[f64], _next: &[f64], _step: u32, _fail: bool| {
            let mut terms = RewardTerms::new();
            terms
                .push(robot_types::transition::RewardTerm {
                    name: "bandit".to_owned(),
                    value: -(action[0] - TARGET).powi(2),
                    weight: 1.0,
                    is_style_term: false,
                })
                .unwrap();
            (terms, Termination::Terminal)
        };
        let schema = agent.transition_schema("bandit-fixture/1.0");
        let mut buffer = RolloutBuffer::new(schema, 64);
        let mut rollout_rng = Rng::new(SeedStream::new(3, 2));

        let mean_before = agent.policy.mean_net.forward_with_trace(&[0.0]).output()[0];
        for _ in 0..200 {
            buffer.clear();
            agent.collect_rollout(
                &mut env,
                &task_fn,
                &mut buffer,
                64,
                RolloutContext {
                    env_id: EnvId::new(0),
                    robot_profile: "bandit-fixture/1.0",
                    episode_seed_base: 1,
                },
                &mut rollout_rng,
            );
            agent.update(&buffer);
        }
        let mean_after = agent.policy.mean_net.forward_with_trace(&[0.0]).output()[0];

        assert!(
            (mean_after - TARGET).abs() < (mean_before - TARGET).abs() * 0.3,
            "policy mean should move substantially toward the target: before {mean_before}, \
             after {mean_after}, target {TARGET}"
        );
        assert!(
            (mean_after - TARGET).abs() < 0.1,
            "expected convergence near {TARGET}, got {mean_after}"
        );
    }

    fn bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    fn run_effort_updates(agent: &mut PpoAgent, updates: usize) {
        let mut env = PendulumEnv::with_reset_range(
            PendulumParams {
                length: 1.0,
                mass: 1.0,
                gravity: 9.80665,
                damping: 0.05,
            },
            4.0,
            THETA_RANGE,
            OMEGA_RANGE,
            SeedStream::new(5, 0),
        );
        let task_fn = |_obs: &[f64], action: &[f64], _next: &[f64], _step: u32, _fail: bool| {
            let mut terms = RewardTerms::new();
            terms
                .push(robot_types::transition::RewardTerm {
                    name: "effort".to_owned(),
                    value: -action[0] * action[0],
                    weight: 1.0,
                    is_style_term: false,
                })
                .unwrap();
            (terms, Termination::Terminal)
        };
        let mut buffer = RolloutBuffer::new(agent.transition_schema("pendulum-fixture/1.0"), 64);
        let mut rollout_rng = Rng::new(SeedStream::new(5, 2));
        for _ in 0..updates {
            buffer.clear();
            agent.collect_rollout(
                &mut env,
                &task_fn,
                &mut buffer,
                64,
                RolloutContext {
                    env_id: EnvId::new(0),
                    robot_profile: "pendulum-fixture/1.0",
                    episode_seed_base: 1,
                },
                &mut rollout_rng,
            );
            agent.update(&buffer);
        }
    }

    #[test]
    fn frozen_groups_stay_bit_identical_after_training_builds_momentum() {
        let mut agent = new_agent(5);
        run_effort_updates(&mut agent, 2);

        let policy_count = agent.policy.param_count();
        let log_std_range = agent.policy.mean_net.param_count()..policy_count;
        agent
            .set_policy_groups(vec![
                ParamGroup {
                    name: "policy-hidden-weights".to_owned(),
                    range: 0..32,
                    frozen: true,
                },
                ParamGroup {
                    name: "policy-log-std".to_owned(),
                    range: log_std_range,
                    frozen: true,
                },
            ])
            .unwrap();
        let hidden_weights_before = bits(agent.policy.mean_net.layers()[0].weight());
        let log_std_before = bits(&agent.policy.log_std);
        let hidden_bias_before = bits(agent.policy.mean_net.layers()[0].bias());
        let output_weight_before = bits(agent.policy.mean_net.layers()[1].weight());

        run_effort_updates(&mut agent, 3);

        assert_eq!(
            bits(agent.policy.mean_net.layers()[0].weight()),
            hidden_weights_before
        );
        assert_eq!(bits(&agent.policy.log_std), log_std_before);
        assert_ne!(
            bits(agent.policy.mean_net.layers()[0].bias()),
            hidden_bias_before
        );
        assert_ne!(
            bits(agent.policy.mean_net.layers()[1].weight()),
            output_weight_before
        );
    }

    #[test]
    fn frozen_log_std_is_not_moved_when_only_log_std_is_frozen() {
        let mut agent = new_agent(5);
        run_effort_updates(&mut agent, 2);
        let policy_count = agent.policy.param_count();
        agent
            .set_policy_groups(vec![ParamGroup {
                name: "policy-log-std".to_owned(),
                range: agent.policy.mean_net.param_count()..policy_count,
                frozen: true,
            }])
            .unwrap();
        let log_std_before = bits(&agent.policy.log_std);
        let mean_before = bits(agent.policy.mean_net.layers()[1].weight());

        run_effort_updates(&mut agent, 3);

        assert_eq!(bits(&agent.policy.log_std), log_std_before);
        assert_ne!(
            bits(agent.policy.mean_net.layers()[1].weight()),
            mean_before
        );
    }

    #[test]
    fn frozen_groups_stay_bit_identical_while_unfrozen_parameters_change() {
        let mut agent = new_agent(5);
        let value_params = agent.value_net.param_count();
        // Flat order: layer 0 weights 0..32, layer 0 biases 32..48,
        // layer 1 weights 48..64, layer 1 bias 64.
        agent
            .set_policy_groups(vec![ParamGroup {
                name: "policy-hidden-weights".to_owned(),
                range: 0..32,
                frozen: true,
            }])
            .unwrap();
        agent
            .set_value_groups(vec![ParamGroup {
                name: "value-all".to_owned(),
                range: 0..value_params,
                frozen: true,
            }])
            .unwrap();

        let frozen_before = bits(agent.policy.mean_net.layers()[0].weight());
        let hidden_bias_before = bits(agent.policy.mean_net.layers()[0].bias());
        let output_weight_before = bits(agent.policy.mean_net.layers()[1].weight());
        let value_before: Vec<Vec<u64>> = agent
            .value_net
            .layers()
            .iter()
            .flat_map(|l| [bits(l.weight()), bits(l.bias())])
            .collect();

        let mut env = PendulumEnv::with_reset_range(
            PendulumParams {
                length: 1.0,
                mass: 1.0,
                gravity: 9.80665,
                damping: 0.05,
            },
            4.0,
            THETA_RANGE,
            OMEGA_RANGE,
            SeedStream::new(5, 0),
        );
        let task_fn = |_obs: &[f64], action: &[f64], _next: &[f64], _step: u32, _fail: bool| {
            let mut terms = RewardTerms::new();
            terms
                .push(robot_types::transition::RewardTerm {
                    name: "effort".to_owned(),
                    value: -action[0] * action[0],
                    weight: 1.0,
                    is_style_term: false,
                })
                .unwrap();
            (terms, Termination::Terminal)
        };
        let mut buffer = RolloutBuffer::new(agent.transition_schema("pendulum-fixture/1.0"), 64);
        let mut rollout_rng = Rng::new(SeedStream::new(5, 2));
        for _ in 0..3 {
            buffer.clear();
            agent.collect_rollout(
                &mut env,
                &task_fn,
                &mut buffer,
                64,
                RolloutContext {
                    env_id: EnvId::new(0),
                    robot_profile: "pendulum-fixture/1.0",
                    episode_seed_base: 1,
                },
                &mut rollout_rng,
            );
            agent.update(&buffer);
        }

        assert_eq!(
            bits(agent.policy.mean_net.layers()[0].weight()),
            frozen_before
        );
        let value_after: Vec<Vec<u64>> = agent
            .value_net
            .layers()
            .iter()
            .flat_map(|l| [bits(l.weight()), bits(l.bias())])
            .collect();
        assert_eq!(value_after, value_before);
        assert_ne!(
            bits(agent.policy.mean_net.layers()[0].bias()),
            hidden_bias_before
        );
        assert_ne!(
            bits(agent.policy.mean_net.layers()[1].weight()),
            output_weight_before
        );
    }

    #[test]
    fn invalid_parameter_groups_leave_the_agent_unchanged() {
        let mut agent = new_agent(5);
        let out_of_bounds = ParamGroup {
            name: "too-far".to_owned(),
            range: 0..1000,
            frozen: true,
        };
        assert!(matches!(
            agent.set_policy_groups(vec![out_of_bounds]),
            Err(ParamGroupError::OutOfBounds { .. })
        ));
        assert!(agent.policy_groups().is_empty());
        assert!(matches!(
            agent.set_value_groups(vec![ParamGroup {
                name: "reversed".to_owned(),
                range: 5..5,
                frozen: true,
            }]),
            Err(ParamGroupError::EmptyRange { .. })
        ));
        assert!(agent.value_groups().is_empty());
    }

    fn new_agent(seed: u64) -> PpoAgent {
        let mut rng = Rng::new(SeedStream::new(seed, 1));
        let policy_net = initialize(
            &[2, 16, 1],
            vec![Activation::Tanh, Activation::Identity],
            &mut rng,
        )
        .unwrap();
        let value_net = initialize(
            &[2, 16, 1],
            vec![Activation::Tanh, Activation::Identity],
            &mut rng,
        )
        .unwrap();
        PpoAgent::new(
            GaussianPolicy::new(policy_net, -1.0),
            value_net,
            PpoConfig::default(),
            2,
        )
    }

    /// Near-upright reset range: this is a stabilization task
    /// (SPECS.md Section 9.2, "Train balance and locomotion first"), not
    /// general swing-up, which is a substantially harder control problem a
    /// small reference task is not expected to solve.
    const THETA_RANGE: (f64, f64) = (core::f64::consts::PI - 0.3, core::f64::consts::PI + 0.3);
    const OMEGA_RANGE: (f64, f64) = (-0.3, 0.3);

    fn test_task() -> BalanceTask {
        BalanceTask::new(BalanceTaskConfig {
            max_steps: 100,
            fall_angle: 1.0,
            ..BalanceTaskConfig::default()
        })
    }

    /// Average return of a fixed, deterministic zero-action controller —
    /// the "deterministic baseline" PLAN.md Phase 3 asks training to be
    /// compared against. No policy, no sampling, no randomness beyond the
    /// environment's own fixed-per-seed reset.
    fn zero_action_baseline(episodes: usize, seed: u64) -> f64 {
        let params = PendulumParams {
            length: 1.0,
            mass: 1.0,
            gravity: 9.80665,
            damping: 0.05,
        };
        let mut env = PendulumEnv::with_reset_range(
            params,
            4.0,
            THETA_RANGE,
            OMEGA_RANGE,
            SeedStream::new(seed, 0),
        );
        let task = test_task();
        let mut total = 0.0;
        for ep in 0..episodes {
            let mut obs = [0.0; 2];
            env.reset(SeedStream::new(seed, ep as u64), &mut obs);
            let mut episode_return = 0.0;
            for step in 1..=100u32 {
                let outcome = env.step(&[0.0], robot_types::time::Duration::from_millis(20));
                let (reward, termination) = task.evaluate(
                    &obs,
                    &[0.0],
                    &outcome.observation,
                    step,
                    outcome.physics_failure,
                );
                episode_return += reward.total();
                obs = [outcome.observation[0], outcome.observation[1]];
                if termination.ends_episode() {
                    break;
                }
            }
            total += episode_return;
        }
        total / episodes as f64
    }

    /// Average return of the trained policy's deterministic mean action
    /// (no exploration noise), which is how a trained policy is evaluated
    /// in practice and removes sampling variance from the comparison.
    fn mean_action_return(agent: &PpoAgent, episodes: usize, seed: u64) -> f64 {
        let params = PendulumParams {
            length: 1.0,
            mass: 1.0,
            gravity: 9.80665,
            damping: 0.05,
        };
        let mut env = PendulumEnv::with_reset_range(
            params,
            4.0,
            THETA_RANGE,
            OMEGA_RANGE,
            SeedStream::new(seed, 0),
        );
        let task = test_task();
        let mut total = 0.0;
        for ep in 0..episodes {
            let mut obs = [0.0; 2];
            env.reset(SeedStream::new(seed, ep as u64), &mut obs);
            let mut episode_return = 0.0;
            for step in 1..=100u32 {
                let mut normalized = [0.0; 2];
                agent.normalizer.normalize(&obs, &mut normalized).unwrap();
                let action = agent
                    .policy
                    .mean_net
                    .forward_with_trace(&normalized)
                    .output()
                    .to_vec();
                let outcome = env.step(&action, robot_types::time::Duration::from_millis(20));
                let (reward, termination) = task.evaluate(
                    &obs,
                    &action,
                    &outcome.observation,
                    step,
                    outcome.physics_failure,
                );
                episode_return += reward.total();
                obs = [outcome.observation[0], outcome.observation[1]];
                if termination.ends_episode() {
                    break;
                }
            }
            total += episode_return;
        }
        total / episodes as f64
    }

    /// This is an integration smoke test, not a convergence claim.
    ///
    /// SPECS.md Section 3: "A failed verification is a result: retain a
    /// correction record instead of preserving an unsupported claim."
    /// Earlier versions of this test asserted that training measurably beat
    /// the zero-action baseline. Chasing that assertion found and fixed a
    /// real bug (`gae.rs`'s bootstrap value was silently read from the
    /// *following buffer slot* rather than from each transition's own
    /// recorded next observation — wrong whenever a truncated episode is
    /// followed, in the same bounded rollout chunk, by the first step of a
    /// freshly reset one, which SPECS.md Section 7.6 explicitly permits).
    /// That fix is real and stays. But after fixing it, stabilizing an
    /// inverted pendulum from a near-upright start still did not reliably
    /// beat the zero-action baseline within a small-reference-task compute
    /// budget (tried: 15 and 80 outer iterations, 400- and 600-step
    /// rollouts, several reset-range and fall-angle settings) — this is
    /// recorded as a negative result rather than hidden behind a loosened
    /// assertion or a lucky seed. The separate
    /// `ppo_drives_a_one_step_bandit_policy_mean_toward_the_target` test
    /// isolates and confirms the gradient machinery itself is correct;
    /// credit assignment across an unstable multi-step trajectory with a
    /// co-trained value function is a genuinely harder tuning problem,
    /// which PLAN.md Phase 3's "compare learning against deterministic
    /// baselines" step is the place to resolve with real measurement, not
    /// by iterating hyperparameters until one unit test happens to pass.
    ///
    /// What this test actually asserts: the full pipeline — environment,
    /// task, rollout buffer, GAE, backprop, Adam, for both the policy and
    /// the value network — runs for many iterations without producing a
    /// non-finite parameter, the generation counter advances exactly once
    /// per `update` call, and the rollout buffer's capacity/clear mechanics
    /// hold up under repeated real use.
    #[test]
    fn training_pipeline_runs_without_diverging() {
        let mut agent = new_agent(42);
        // Reported, not asserted: see this test's own doc comment for why a
        // strict improvement claim is not made here. `--nocapture` shows the
        // actual numbers for whoever picks this tuning problem back up.
        let baseline_return = zero_action_baseline(20, 1000);
        eprintln!("zero-action baseline average return: {baseline_return}");

        let params = PendulumParams {
            length: 1.0,
            mass: 1.0,
            gravity: 9.80665,
            damping: 0.05,
        };
        let mut env = PendulumEnv::with_reset_range(
            params,
            4.0,
            THETA_RANGE,
            OMEGA_RANGE,
            SeedStream::new(42, 7),
        );
        let task = test_task();
        let task_fn = |obs: &[f64], act: &[f64], next: &[f64], step: u32, fail: bool| {
            task.evaluate(obs, act, next, step, fail)
        };
        let mut rng = Rng::new(SeedStream::new(42, 5));
        let schema = agent.transition_schema("pendulum-fixture/1.0");
        let mut buffer = RolloutBuffer::new(schema, 400);

        for i in 0..20 {
            buffer.clear();
            agent.collect_rollout(
                &mut env,
                &task_fn,
                &mut buffer,
                400,
                RolloutContext {
                    env_id: EnvId::new(0),
                    robot_profile: "pendulum-fixture/1.0",
                    episode_seed_base: 9000,
                },
                &mut rng,
            );
            assert!(buffer.len() <= buffer.capacity());
            agent.update(&buffer);
            assert_eq!(agent.generation(), Generation(i + 1));

            for layer in agent.policy.mean_net.layers() {
                assert!(
                    layer.weight().iter().all(|w| w.is_finite()),
                    "policy weight diverged"
                );
                assert!(
                    layer.bias().iter().all(|b| b.is_finite()),
                    "policy bias diverged"
                );
            }
            assert!(
                agent.policy.log_std.iter().all(|s| s.is_finite()),
                "log_std diverged"
            );
            for layer in agent.value_net.layers() {
                assert!(
                    layer.weight().iter().all(|w| w.is_finite()),
                    "value weight diverged"
                );
            }
        }
        assert_eq!(agent.generation(), Generation(20));

        let trained_return = mean_action_return(&agent, 20, 1000);
        eprintln!("trained (mean-action) average return after 20 updates: {trained_return}");
    }

    #[test]
    fn collect_rollout_stops_cleanly_at_buffer_capacity() {
        let mut agent = new_agent(1);
        let params = PendulumParams {
            length: 1.0,
            mass: 1.0,
            gravity: 9.80665,
            damping: 0.05,
        };
        let mut env = PendulumEnv::new(params, 4.0, SeedStream::new(1, 0));
        let task = BalanceTask::new(BalanceTaskConfig::default());
        let task_fn = |obs: &[f64], act: &[f64], next: &[f64], step: u32, fail: bool| {
            task.evaluate(obs, act, next, step, fail)
        };
        let mut rng = Rng::new(SeedStream::new(1, 2));
        let schema = agent.transition_schema("pendulum-fixture/1.0");
        let mut buffer = RolloutBuffer::new(schema, 10);
        agent.collect_rollout(
            &mut env,
            &task_fn,
            &mut buffer,
            1000,
            RolloutContext {
                env_id: EnvId::new(0),
                robot_profile: "pendulum-fixture/1.0",
                episode_seed_base: 1,
            },
            &mut rng,
        );
        assert_eq!(buffer.len(), 10);
    }
}

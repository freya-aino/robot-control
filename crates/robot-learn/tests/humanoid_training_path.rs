//! The humanoid training path, end to end: the vendored G1 in `robot-sim`'s
//! world, driven by its declared drives through `robot-tasks`'
//! `HumanoidEnv`, rewarded by the stand task's registry, collected and
//! optimized by this crate's PPO.
//!
//! This is **plumbing qualification, not a training result**: a few PPO
//! iterations at smoke scale verify that every seam carries what it claims —
//! observations flow, rewards are the registry's, falls terminate episodes,
//! the normalizer and both networks update. Nothing here claims the policy
//! learns to stand; that is a compute-scale run this workspace has not
//! performed, and `robot-learn`'s README says so.

use robot_learn::ppo::{PpoAgent, PpoConfig, RolloutContext};
use robot_learn::rollout::RolloutBuffer;
use robot_nn::init::initialize;
use robot_nn::layer::Activation;
use robot_sim::env::Environment;
use robot_tasks::env::{g1_stand, outcome_task_fn};
use robot_types::ids::EnvId;
use robot_types::rng::{Rng, SeedStream};

#[test]
fn ppo_collects_and_updates_on_the_g1_stand_environment() {
    let mut env = g1_stand(1_000, SeedStream::new(21, 0)).expect("the reference wiring builds");
    let obs_width = env.observation_width();
    let act_width = env.action_width();
    assert_eq!(act_width, 29, "the G1 declares 29 drives");

    let mut rng = Rng::new(SeedStream::new(21, 1));
    let policy_net = initialize(
        &[obs_width, 32, act_width],
        vec![Activation::Tanh, Activation::Identity],
        &mut rng,
    )
    .unwrap();
    let value_net = initialize(
        &[obs_width, 32, 1],
        vec![Activation::Tanh, Activation::Identity],
        &mut rng,
    )
    .unwrap();
    let mut agent = PpoAgent::new(
        robot_learn::gaussian_policy::GaussianPolicy::new(policy_net, -1.5),
        value_net,
        PpoConfig {
            epochs: 1,
            ..PpoConfig::default()
        },
        obs_width,
    );

    let task_fn = outcome_task_fn(env.outcome_cell());
    let schema = agent.transition_schema("unitree-g1-29dof-usd/1.0");
    const STEPS: usize = 32;
    let mut buffer = RolloutBuffer::new(schema, STEPS);
    let flat_bits = |agent: &PpoAgent| -> Vec<u64> {
        agent
            .policy
            .mean_net
            .layers()
            .iter()
            .flat_map(|l| l.weight().iter().chain(l.bias()).map(|p| p.to_bits()))
            .collect()
    };
    let before = flat_bits(&agent);
    let mut rollout_rng = Rng::new(SeedStream::new(21, 2));
    for _ in 0..2 {
        buffer.clear();
        agent.collect_rollout(
            &mut env,
            &task_fn,
            &mut buffer,
            STEPS,
            RolloutContext {
                env_id: EnvId::new(0),
                robot_profile: "unitree-g1-29dof-usd/1.0",
                episode_seed_base: 5,
            },
            &mut rollout_rng,
        );
        assert_eq!(
            buffer.len(),
            STEPS,
            "every control step yields a transition"
        );
        for t in buffer.as_slice() {
            assert!(t.reward.total().is_finite(), "a non-finite reward escaped");
            assert_eq!(t.observation.len(), obs_width);
            assert_eq!(t.action.len(), act_width);
        }
        agent.update(&buffer);
    }
    let after = flat_bits(&agent);
    assert_ne!(before, after, "the update must actually move the policy");
}

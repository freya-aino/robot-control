//! Tasks, per SPECS.md Section 6's `robot-tasks` row ("Tracking, balance,
//! reach, footwork, boxing and perturbation tasks") and Section 9.2.
//!
//! | Module | Contents |
//! | --- | --- |
//! | [`reward`] | The flat-observation reward registry (Section 9.3) |
//! | [`balance`] | `BalanceTask` on the pendulum fixture |
//! | [`humanoid`] | `HumanoidState` and its extractor from `robot-sim`'s world, observation specs with the privileged-input audit, termination versus truncation, the humanoid reward registry and task trait |
//! | [`terms`] | Shared reward terms: balance, energy, smoothness, constraints, contacts |
//! | [`tasks`] | Stand, perturbation, tracking, reach, footwork and boxing-pad tasks |
//! | [`env`] | `HumanoidEnv`: the world stepped under the model's declared drives at a control rate, observed through a task, as `robot-sim`'s `Environment` for `robot-learn` |
//! | [`randomize`] | Calibrated domain randomization with held-out ranges and sensitivity ranking |

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod balance;
pub mod env;
pub mod humanoid;
pub mod randomize;
pub mod reward;
pub mod tasks;
pub mod terms;

#[cfg(test)]
mod integration {
    use crate::humanoid::StateExtractor;
    use crate::humanoid::{HumanoidTask, Status};
    use crate::tasks::{FallLimits, StandTask};
    use robot_model::fixtures::unitree_g1_29dof;
    use robot_model::usd::with_floating_base;
    use robot_sim::env::Environment;
    use robot_sim::options::SimOptions;
    use robot_sim::world::{Ground, World, WorldEnv};
    use robot_types::frames::Vec3;
    use robot_types::rng::SeedStream;

    #[test]
    fn the_g1_world_yields_a_standing_state_and_takes_a_push() {
        let fixed = unitree_g1_29dof().unwrap();
        let floating = with_floating_base(fixed.clone()).unwrap();
        let link = |n: &str| {
            fixed
                .links
                .iter()
                .position(|l| l.name.as_str() == n)
                .unwrap()
        };
        let world = World::from_profile(&floating, SimOptions::default()).with_ground(Ground {
            height: 0.0,
            friction: 0.8,
        });
        let mut q = world.rest_state().q;
        let lowest = world.lowest_contact_height(&q).unwrap().unwrap();
        let ext = world.model().qpos_to_external(&q).unwrap();
        let mut ext = ext;
        ext[2] -= lowest;
        q = world.model().qpos_to_internal(&ext).unwrap();
        let extractor = StateExtractor {
            profile: fixed.clone(),
            tracked_links: vec![
                link("left_ankle_roll_link"),
                link("right_ankle_roll_link"),
                link("right_rubber_hand"),
                link("left_rubber_hand"),
                link("head_link"),
            ],
            feet: vec![0, 1],
            forbidden_contact: vec![link("pelvis"), link("torso_link")],
            contact_height: 0.08,
            ground_height: 0.0,
        };
        let mut env = WorldEnv::new(world, q, 0.0, SeedStream::new(1, 0)).unwrap();
        let mut obs = vec![0.0; env.observation_width()];
        env.reset(SeedStream::new(1, 0), &mut obs);
        let s = extractor
            .extract(env.world(), env.state(), &[], 0.0)
            .unwrap();
        assert!(
            s.base_position.z > 0.6 && s.base_position.z < 0.9,
            "{}",
            s.base_position.z
        );
        assert_eq!(s.foot_contact, vec![true, true]);
        assert_eq!(s.undesired_contacts, 0);
        let task = StandTask::new(
            fixed.dof(),
            s.base_position.z,
            FallLimits {
                min_height: 0.4,
                max_tilt: 0.8,
                max_steps: 10,
            },
        );
        assert_eq!(task.status(&s, 0), Status::Running);

        assert!(env.push_base(Vec3::new(0.3, 0.0, 0.0)));
        let pushed = extractor
            .extract(env.world(), env.state(), &[], 0.0)
            .unwrap();
        assert!((pushed.base_linear_velocity.x - 0.3).abs() < 1e-12);
    }
}

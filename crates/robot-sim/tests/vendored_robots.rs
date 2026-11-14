//! The vendored Go2 and H1 USD profiles build worlds and step them.
//!
//! This is a smoke test of the whole import-to-simulation path for the two
//! robots vendored after the G1 (`assets/robots/unitree_go2_usd/`,
//! `assets/robots/unitree_h1_usd/`): the profile composes from its layers,
//! becomes a floating-base world, and a short zero-torque fall stays finite
//! and loses height. It asserts nothing about gait or balance quality —
//! no controller is attached.

use robot_model::fixtures::{
    unitree_go2_12dof_usd_floating_base, unitree_h1_19dof_usd_floating_base,
};
use robot_sim::options::SimOptions;
use robot_sim::world::World;

fn falls_finitely(profile: robot_model::profile::RobotProfile) {
    let mut world = World::from_profile(&profile, SimOptions::default());
    let mut state = world.rest_state();
    let tau = vec![0.0; world.nv()];
    let z0 = state.q[2];
    for _ in 0..50 {
        world.step(&mut state, &tau).expect("step");
    }
    assert!(state.q.iter().all(|v| v.is_finite()));
    assert!(state.q[2] < z0, "free fall must lose height");
}

#[test]
fn the_go2_profile_steps_in_a_floating_base_world() {
    falls_finitely(unitree_go2_12dof_usd_floating_base().unwrap());
}

#[test]
fn the_h1_profile_steps_in_a_floating_base_world() {
    falls_finitely(unitree_h1_19dof_usd_floating_base().unwrap());
}

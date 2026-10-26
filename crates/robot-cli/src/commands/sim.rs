//! `robot sim qualify` and `robot sim multibody` — PLAN.md §4's proposed
//! `robot sim qualify` command, plus a second one this revision adds for
//! the multi-body solver in `robot_sim::articulated`.
//!
//! `qualify` is SPECS.md §9.1's "analytic pendulum" energy-conservation
//! check (Appendix B.4 item 2). `multibody` runs the vendored Unitree G1
//! fixture's real 29-DOF joint tree through the Articulated-Body
//! Algorithm — bolted at the pelvis (fixed base, no ground contact; see
//! `robot_sim::articulated`'s doc comment for why), not a standing or
//! walking claim. A full qualification run (actuator step response,
//! static standing, foot slip, push recovery, controlled impact) needs
//! ground contact and a free-floating base this crate does not have yet;
//! this command does not pretend otherwise.

use super::EXIT_FAILURE;
use robot_sim::articulated::{ArticulatedBody, MultibodyState};
use robot_sim::pendulum::{Pendulum, PendulumParams, PendulumState};
use robot_types::time::Duration;

/// Relative energy drift this benchmark tolerates before reporting failure.
/// Loose enough to pass on any reasonable step count while still catching
/// a broken integrator (e.g. plain explicit Euler on this system drifts by
/// orders of magnitude more over the same run).
const MAX_RELATIVE_ENERGY_DRIFT: f64 = 0.05;

/// Runs the undamped pendulum for `steps` steps of `dt_seconds` each, with
/// zero actuator torque, and reports the worst relative energy drift from
/// the initial value. Returns the process exit code.
pub fn qualify(steps: u32, dt_seconds: f64) -> i32 {
    let params = PendulumParams {
        length: 1.0,
        mass: 1.0,
        gravity: 9.81,
        damping: 0.0,
    };
    let mut pendulum = Pendulum::new(
        params,
        PendulumState {
            theta: 1.0,
            omega: 0.0,
        },
    );
    let initial_energy = pendulum.energy();
    let dt = Duration::from_secs_f64(dt_seconds);

    let mut worst_drift = 0.0_f64;
    for step in 0..steps {
        if let Err(e) = pendulum.step(0.0, dt) {
            eprintln!("error: step {step}: {e}");
            return EXIT_FAILURE;
        }
        let drift = ((pendulum.energy() - initial_energy) / initial_energy).abs();
        worst_drift = worst_drift.max(drift);
    }

    println!("pendulum qualification: {steps} steps at {dt_seconds} s");
    println!("  initial energy:          {initial_energy:.6} J");
    println!("  final energy:            {:.6} J", pendulum.energy());
    println!("  worst relative drift:    {worst_drift:.6}");
    println!("  tolerance:               {MAX_RELATIVE_ENERGY_DRIFT:.6}");

    if worst_drift <= MAX_RELATIVE_ENERGY_DRIFT {
        println!("  result:                  PASS");
        0
    } else {
        println!("  result:                  FAIL (energy drift exceeds tolerance)");
        EXIT_FAILURE
    }
}

/// Runs the vendored Unitree G1 (29 DOF, or 35 with `floating`) through
/// `robot_sim::articulated`'s Articulated-Body Algorithm for `steps` steps
/// of `dt_seconds` each under gravity alone (zero applied torque),
/// printing a progress trace and a finiteness/energy sanity check.
/// Returns the process exit code.
///
/// `floating = false` bolts the pelvis to world (every joint swings under
/// gravity, the base cannot move) — the original demonstration. `floating
/// = true` uses `robot_model::fixtures::unitree_g1_29dof_floating_base`
/// instead: the pelvis itself is a `Free` joint and can fall, which is
/// the capability this command was extended for, and the progress trace
/// reports pelvis height instead of a generic `max|q|`.
pub fn multibody(steps: u32, dt_seconds: f64, floating: bool) -> i32 {
    let profile = if floating {
        robot_model::fixtures::unitree_g1_29dof_floating_base()
    } else {
        robot_model::fixtures::unitree_g1_29dof()
    };
    let profile = match profile {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: could not build the vendored G1 fixture: {e}");
            return EXIT_FAILURE;
        }
    };
    let body = ArticulatedBody::from_profile(&profile);
    let gravity = profile.gravity.acceleration();
    let dt = Duration::from_secs_f64(dt_seconds);

    // A small displacement from the zero posture, not a claim about any
    // real G1 standing pose: this is a demonstration of the solver, not
    // a balance controller. The floating base starts at an identity
    // quaternion (q[3] = w = 1) at the origin.
    let mut q = vec![0.01; body.position_width()];
    if floating {
        q[3] = 1.0;
    }
    let mut state = MultibodyState {
        q,
        qd: vec![0.0; body.dof()],
    };
    let tau = vec![0.0; body.dof()];

    let mode = if floating {
        "floating base, no ground contact"
    } else {
        "fixed base, no ground contact"
    };
    println!("G1 multibody simulation ({mode}): {steps} steps at {dt_seconds} s");
    println!("  degrees of freedom:      {}", body.dof());

    let report_every = (steps / 10).max(1);
    for step in 0..steps {
        if let Err(e) = state.step(&body, &tau, gravity, dt) {
            eprintln!("error: step {step}: {e}");
            return EXIT_FAILURE;
        }
        if step % report_every == 0 || step + 1 == steps {
            let energy = match body.kinetic_energy(&state.q, &state.qd) {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("error: step {step}: {e}");
                    return EXIT_FAILURE;
                }
            };
            let t = (step + 1) as f64 * dt_seconds;
            if floating {
                println!(
                    "  t={t:>7.3}s  kinetic_energy={energy:>10.4} J  pelvis_height={:.4} m",
                    state.q[2]
                );
            } else {
                let max_q = state.q.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                println!("  t={t:>7.3}s  kinetic_energy={energy:>10.4} J  max|q|={max_q:.4} rad");
            }
        }
    }

    let finite = state.q.iter().chain(&state.qd).all(|v| v.is_finite());
    if !finite {
        println!("  result:                  FAIL (non-finite state)");
        return EXIT_FAILURE;
    }
    println!("  result:                  PASS (ran to completion, finite throughout)");
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reasonable_step_count_and_timestep_passes() {
        assert_eq!(qualify(2_000, 0.001), 0);
    }

    #[test]
    fn a_wildly_large_timestep_can_fail() {
        assert_eq!(qualify(2_000, 0.5), EXIT_FAILURE);
    }

    #[test]
    fn multibody_runs_the_real_g1_without_diverging() {
        assert_eq!(multibody(200, 0.0005, false), 0);
    }

    #[test]
    fn multibody_with_a_floating_base_runs_without_diverging() {
        assert_eq!(multibody(200, 0.0001, true), 0);
    }
}

//! `robot sim qualify` — PLAN.md §4's proposed command, scoped here to the
//! one benchmark `robot-sim` actually implements: SPECS.md §9.1's
//! "analytic pendulum" energy-conservation check (Appendix B.4 item 2).
//!
//! A full qualification run (actuator step response, static standing,
//! foot slip, push recovery, controlled impact) needs the multi-body
//! simulation this crate does not have; this command does not pretend
//! otherwise.

use super::EXIT_FAILURE;
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
}

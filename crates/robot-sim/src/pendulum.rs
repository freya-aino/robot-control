//! The pendulum qualification benchmark.
//!
//! SPECS.md Section 9.1: "Qualify the Rust backend on pendulum, actuator
//! step response, static standing, foot slip, push recovery and controlled
//! impact benchmarks." Appendix B.4 item 2 names the same thing first:
//! "Test analytic pendulum and actuator fixtures." This is that first,
//! smallest benchmark: a damped, actuated pendulum integrated by
//! semi-implicit (symplectic) Euler, checked against energy conservation in
//! the undamped case and against small-angle analytic motion.
//!
//! This is deliberately a single-degree-of-freedom benchmark, not a claim of
//! multi-body articulated dynamics. See the crate-level documentation for
//! what `robot-sim` does and does not provide in this revision.

use core::fmt;
use robot_types::time::Duration;

/// Physical parameters of a simple pendulum: a point mass on a massless rod.
#[derive(Clone, Copy, Debug)]
pub struct PendulumParams {
    /// Rod length in metres.
    pub length: f64,
    /// Bob mass in kilograms (cancels out of the undamped equation of
    /// motion, but scales the torque an actuator must supply).
    pub mass: f64,
    /// Gravitational acceleration magnitude.
    pub gravity: f64,
    /// Linear damping coefficient on angular velocity.
    pub damping: f64,
}

/// State of the pendulum: angle from the downward vertical, and angular
/// velocity.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct PendulumState {
    /// Angle in radians, zero at the downward vertical.
    pub theta: f64,
    /// Angular velocity in radians per second.
    pub omega: f64,
}

/// Why a pendulum step was rejected.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct NonFiniteInput;

impl fmt::Display for NonFiniteInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "pendulum step received a non-finite torque or dt")
    }
}

impl core::error::Error for NonFiniteInput {}

/// A damped, actuated pendulum.
#[derive(Clone, Copy, Debug)]
pub struct Pendulum {
    params: PendulumParams,
    state: PendulumState,
}

impl Pendulum {
    /// Builds a pendulum at the given initial state.
    #[must_use]
    pub const fn new(params: PendulumParams, initial: PendulumState) -> Self {
        Self {
            params,
            state: initial,
        }
    }

    /// Current state.
    #[must_use]
    pub const fn state(&self) -> PendulumState {
        self.state
    }

    /// Angular acceleration at the current state under `torque` (applied
    /// about the pivot, in addition to gravity and damping).
    fn angular_acceleration(&self, torque: f64) -> f64 {
        let PendulumParams {
            length,
            mass,
            gravity,
            damping,
        } = self.params;
        let gravity_term = -(gravity / length) * self.state.theta.sin();
        let damping_term = -damping * self.state.omega / (mass * length * length);
        let actuator_term = torque / (mass * length * length);
        gravity_term + damping_term + actuator_term
    }

    /// Mechanical energy (kinetic + potential), taking the pivot as the
    /// potential-energy reference. Conserved exactly by the continuous
    /// dynamics when `damping == 0` and `torque == 0`; a numerical
    /// integrator is qualified by how well it preserves this.
    #[must_use]
    pub fn energy(&self) -> f64 {
        let PendulumParams {
            length,
            mass,
            gravity,
            ..
        } = self.params;
        let kinetic = 0.5 * mass * (length * self.state.omega).powi(2);
        let potential = -mass * gravity * length * self.state.theta.cos();
        kinetic + potential
    }

    /// Advances the state by `dt` under `torque`, using semi-implicit
    /// (symplectic) Euler: the velocity is updated first and the *new*
    /// velocity is used to update position. This is why the undamped,
    /// unforced energy drift stays bounded rather than growing without
    /// bound the way explicit Euler's does — it is the standard reason
    /// symplectic integrators are preferred for oscillatory physics.
    pub fn step(&mut self, torque: f64, dt: Duration) -> Result<(), NonFiniteInput> {
        let dt_s = dt.as_secs_f64();
        if !torque.is_finite() || !dt_s.is_finite() || dt_s <= 0.0 {
            return Err(NonFiniteInput);
        }
        let alpha = self.angular_acceleration(torque);
        self.state.omega += alpha * dt_s;
        self.state.theta += self.state.omega * dt_s;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> PendulumParams {
        PendulumParams {
            length: 1.0,
            mass: 1.0,
            gravity: 9.80665,
            damping: 0.0,
        }
    }

    #[test]
    fn small_angle_oscillation_matches_the_analytic_period() {
        // For small theta, theta(t) ≈ theta0 * cos(omega_n * t), omega_n = sqrt(g/L).
        let theta0 = 0.05; // small angle, radians
        let mut pendulum = Pendulum::new(
            params(),
            PendulumState {
                theta: theta0,
                omega: 0.0,
            },
        );
        let omega_n = (params().gravity / params().length).sqrt();
        let period = core::f64::consts::TAU / omega_n;
        let dt = Duration::from_micros(100);
        let steps = (period / dt.as_secs_f64()).round() as u64;

        for _ in 0..steps {
            pendulum.step(0.0, dt).unwrap();
        }
        // After one full period, the pendulum should return close to its
        // starting angle (small-angle approximation plus integration error).
        assert!(
            (pendulum.state().theta - theta0).abs() < 1e-3,
            "expected return to {theta0}, got {}",
            pendulum.state().theta
        );
    }

    #[test]
    fn undamped_unforced_energy_stays_bounded_over_many_periods() {
        let mut pendulum = Pendulum::new(
            params(),
            PendulumState {
                theta: 1.0,
                omega: 0.0,
            },
        );
        let e0 = pendulum.energy();
        let dt = Duration::from_micros(200);
        for _ in 0..200_000 {
            pendulum.step(0.0, dt).unwrap();
        }
        let drift = (pendulum.energy() - e0).abs() / e0.abs();
        assert!(
            drift < 0.02,
            "symplectic Euler energy drift too large: {drift}"
        );
    }

    #[test]
    fn damping_monotonically_removes_energy() {
        let damped = PendulumParams {
            damping: 0.5,
            ..params()
        };
        let mut pendulum = Pendulum::new(
            damped,
            PendulumState {
                theta: 1.0,
                omega: 0.0,
            },
        );
        let dt = Duration::from_millis(1);
        let mut previous = pendulum.energy();
        for _ in 0..5000 {
            pendulum.step(0.0, dt).unwrap();
            let e = pendulum.energy();
            assert!(
                e <= previous + 1e-9,
                "damped energy must be non-increasing: {e} vs {previous}"
            );
            previous = e;
        }
    }

    #[test]
    fn a_pd_actuator_drives_the_pendulum_toward_a_target_angle() {
        // Gravity-compensated PD: without compensating the current gravity
        // torque, a plain PD gain settles with steady-state droop (the
        // classic reason real joint controllers add a gravity feedforward
        // term), so the compensation is included here to isolate the
        // actuator's own step response from that separate, expected effect.
        let p = params();
        let mut pendulum = Pendulum::new(
            p,
            PendulumState {
                theta: 1.5,
                omega: 0.0,
            },
        );
        let target = 0.3;
        let (kp, kd) = (20.0, 4.0);
        let dt = Duration::from_millis(2);
        for _ in 0..2000 {
            let state = pendulum.state();
            let gravity_compensation = p.mass * p.gravity * p.length * state.theta.sin();
            let torque = kp * (target - state.theta) - kd * state.omega + gravity_compensation;
            pendulum.step(torque, dt).unwrap();
        }
        assert!(
            (pendulum.state().theta - target).abs() < 0.02,
            "expected convergence to {target}, got {}",
            pendulum.state().theta
        );
    }

    #[test]
    fn nonfinite_inputs_are_rejected() {
        let mut pendulum = Pendulum::new(
            params(),
            PendulumState {
                theta: 0.0,
                omega: 0.0,
            },
        );
        assert!(pendulum.step(f64::NAN, Duration::from_millis(1)).is_err());
        assert!(pendulum.step(0.0, Duration::ZERO).is_err());
    }
}

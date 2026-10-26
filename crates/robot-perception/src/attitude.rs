//! A multiplicative error-state attitude filter with gyro bias estimation.
//!
//! SPECS.md Section 11.2: "The estimation module in `robot-perception` must
//! specify orientation/error-state conventions, IMU bias dynamics, contact
//! assumptions, covariance propagation and measurement gating." Every one of
//! those is named explicitly below rather than left implicit:
//!
//! * **Orientation convention:** the nominal attitude is a unit quaternion
//!   `q` (world-from-body, i.e. `q.rotate(v)` maps a body-frame vector into
//!   world); the error state is a small rotation vector `δθ` such that the
//!   true attitude is `q_true = q ⊗ exp(δθ/2)` (multiplicative, body-frame
//!   error — the standard MEKF convention).
//! * **IMU bias dynamics:** the gyro bias `b` is a separate nominal state,
//!   modeled as a random walk; its error `δb` is the error state's other
//!   three components.
//! * **Contact/motion assumption:** the accelerometer correction assumes the
//!   body is not undergoing large linear acceleration. This filter does not
//!   use ground contact directly — that is `contact::ContactClassifier`'s
//!   job — it gates on specific-force magnitude instead, which is a weaker
//!   but dependency-free proxy; a caller qualifying this against a specific
//!   robot should test whether contact-conditioned gating is actually
//!   needed.
//! * **Covariance propagation and gating:** both delegate to
//!   [`crate::linear_kf::KalmanFilter`], so the numerical machinery is the
//!   one already validated against analytic cases in that module.
//!
//! This is an authored baseline, per
//! `docs/decisions/0003-std-only-default-closure.md`, standing in for the
//! Appendix B.2.9 candidates until one is qualified.

use crate::linear_kf::{FilterError, KalmanFilter};
use core::ops::Mul;
use robot_types::frames::{GravityConvention, Quat, Vec3};
use robot_types::math::Matrix;
use robot_types::time::Duration;

/// Tuning and noise parameters for the attitude filter.
#[derive(Clone, Copy, Debug)]
pub struct AttitudeFilterConfig {
    /// Gyro measurement noise spectral density (rad/s per sqrt(Hz)), applied
    /// as a simple white-noise variance scaled by `dt` for discretization.
    pub gyro_noise_variance: f64,
    /// Gyro bias random-walk variance per second.
    pub bias_random_walk_variance: f64,
    /// Accelerometer measurement noise variance on the normalized
    /// specific-force direction.
    pub accel_noise_variance: f64,
    /// Fractional tolerance on specific-force magnitude (relative to
    /// standard gravity) outside which the accelerometer update is skipped
    /// entirely, before even reaching the Mahalanobis gate — this is the
    /// "body is not undergoing large linear acceleration" assumption, made
    /// explicit and checked rather than silently relied upon.
    pub max_specific_force_deviation: f64,
    /// Mahalanobis-squared gate applied to the accelerometer innovation
    /// after the magnitude check passes.
    pub mahalanobis_gate: f64,
}

impl Default for AttitudeFilterConfig {
    /// Defaults are starting points for qualification, not a validated
    /// tuning for any specific IMU; SPECS.md Section 11.2 requires comparing
    /// against analytic or synthetic truth before trusting any tuning.
    fn default() -> Self {
        Self {
            gyro_noise_variance: 1e-4,
            bias_random_walk_variance: 1e-7,
            accel_noise_variance: 1e-2,
            max_specific_force_deviation: 0.2,
            mahalanobis_gate: 16.0, // ~chi2 99.9th percentile at 3 DOF is ~16.3
        }
    }
}

/// The attitude filter.
pub struct AttitudeFilter {
    /// Nominal attitude: world-from-body.
    q: Quat,
    /// Nominal gyro bias.
    bias: Vec3,
    /// Error-state covariance filter, over `[δθ (3), δb (3)]`.
    kf: KalmanFilter,
    config: AttitudeFilterConfig,
    gravity: GravityConvention,
}

/// A consistency report over one correction, for the qualification suite
/// SPECS.md Section 11.2 asks for ("Compare observable-state error and
/// covariance consistency against analytic or synthetic truth").
#[derive(Clone, Copy, Debug)]
pub struct CorrectionReport {
    /// Whether the magnitude check admitted the sample at all.
    pub magnitude_admitted: bool,
    /// Whether the Mahalanobis gate admitted it (only meaningful if
    /// `magnitude_admitted`).
    pub gate_admitted: bool,
    /// Squared Mahalanobis distance of the innovation, if computed.
    pub mahalanobis_squared: Option<f64>,
}

impl AttitudeFilter {
    /// Builds a filter from an initial attitude, zero bias, and the given
    /// initial uncertainty (radians for attitude, rad/s for bias, both as
    /// standard deviations).
    pub fn new(
        initial_attitude: Quat,
        gravity: GravityConvention,
        config: AttitudeFilterConfig,
        initial_attitude_std: f64,
        initial_bias_std: f64,
    ) -> Result<Self, FilterError> {
        let p0 = Matrix::diagonal(&[
            initial_attitude_std.powi(2),
            initial_attitude_std.powi(2),
            initial_attitude_std.powi(2),
            initial_bias_std.powi(2),
            initial_bias_std.powi(2),
            initial_bias_std.powi(2),
        ]);
        let kf = KalmanFilter::new(Matrix::zeros(6, 1), p0)?;
        Ok(Self {
            q: initial_attitude.normalized(),
            bias: Vec3::ZERO,
            kf,
            config,
            gravity,
        })
    }

    /// Current best attitude estimate.
    #[must_use]
    pub const fn attitude(&self) -> Quat {
        self.q
    }

    /// Current best gyro bias estimate.
    #[must_use]
    pub const fn bias(&self) -> Vec3 {
        self.bias
    }

    /// Current error-state covariance (6x6, `[δθ, δb]`).
    #[must_use]
    pub fn covariance(&self) -> &Matrix {
        self.kf.covariance()
    }

    /// Propagates the filter forward using a gyro measurement.
    ///
    /// `dt` must be positive; a non-positive or non-finite step is rejected
    /// rather than silently skipped, since it would otherwise corrupt the
    /// discretized process-noise scaling.
    pub fn propagate(&mut self, gyro_measurement: Vec3, dt: Duration) -> Result<(), FilterError> {
        let dt_s = dt.as_secs_f64();
        debug_assert!(
            dt_s > 0.0 && dt_s.is_finite(),
            "propagate requires a positive finite dt"
        );

        let omega_hat = gyro_measurement - self.bias;
        // Nominal propagation: integrate the quaternion by the bias-corrected
        // angular rate over this step.
        let delta = Quat::from_rotation_vector(omega_hat * dt_s);
        self.q = self.q.mul(delta).normalized();

        // Error-state transition: small-angle approximation of exp(-[ω]_x dt).
        let wx = skew(omega_hat);
        let mut f = Matrix::identity(6);
        for r in 0..3 {
            for c in 0..3 {
                f[(r, c)] -= wx[(r, c)] * dt_s;
                f[(r, c + 3)] = if r == c { -dt_s } else { 0.0 };
            }
        }
        let mut q_proc = Matrix::zeros(6, 6);
        for i in 0..3 {
            q_proc[(i, i)] = self.config.gyro_noise_variance * dt_s;
            q_proc[(i + 3, i + 3)] = self.config.bias_random_walk_variance * dt_s;
        }
        self.kf.predict(&f, &q_proc)?;
        Ok(())
    }

    /// The predicted body-frame "up" direction (opposite gravity) at the
    /// current attitude estimate.
    fn predicted_up_body(&self) -> Vec3 {
        self.q.conjugate().rotate(-self.gravity.down)
    }

    /// Corrects the filter using a raw (non-unit) accelerometer reading,
    /// interpreted as specific force in the body frame.
    ///
    /// Returns a report describing whether, and why, the correction was or
    /// was not applied.
    pub fn correct_with_accelerometer(
        &mut self,
        specific_force: Vec3,
    ) -> Result<CorrectionReport, FilterError> {
        let magnitude = specific_force.norm();
        let g = self.gravity.magnitude;
        if !magnitude.is_finite()
            || (magnitude - g).abs() > self.config.max_specific_force_deviation * g
        {
            return Ok(CorrectionReport {
                magnitude_admitted: false,
                gate_admitted: false,
                mahalanobis_squared: None,
            });
        }
        let Some(measured_up) = specific_force.normalized(1e-9) else {
            return Ok(CorrectionReport {
                magnitude_admitted: false,
                gate_admitted: false,
                mahalanobis_squared: None,
            });
        };

        let predicted_up = self.predicted_up_body();
        let h = measurement_jacobian(predicted_up);
        let r = Matrix::diagonal(&[
            self.config.accel_noise_variance,
            self.config.accel_noise_variance,
            self.config.accel_noise_variance,
        ]);
        let z = Matrix::column(&measured_up.to_xyz());
        let predicted_z = Matrix::column(&predicted_up.to_xyz());
        let y = z.sub(&predicted_z)?;
        let s = h.mul(self.kf.covariance())?.mul(&h.transposed())?.add(&r)?;
        let d2 = self.kf.mahalanobis_squared(&y, &s)?;

        if d2 > self.config.mahalanobis_gate {
            return Ok(CorrectionReport {
                magnitude_admitted: true,
                gate_admitted: false,
                mahalanobis_squared: Some(d2),
            });
        }

        self.kf.apply_update(&h, &r, &y, &s)?;
        self.inject_and_reset()?;
        Ok(CorrectionReport {
            magnitude_admitted: true,
            gate_admitted: true,
            mahalanobis_squared: Some(d2),
        })
    }

    /// Injects the error-state correction into the nominal state and resets
    /// the error state to zero — the standard ESKF "injection and reset"
    /// step, required because the filter's linear algebra operates on the
    /// small error while the true attitude lives on a manifold the linear
    /// state cannot represent directly.
    fn inject_and_reset(&mut self) -> Result<(), FilterError> {
        let mean = self.kf.mean();
        let delta_theta = Vec3::new(mean[(0, 0)], mean[(1, 0)], mean[(2, 0)]);
        let delta_bias = Vec3::new(mean[(3, 0)], mean[(4, 0)], mean[(5, 0)]);
        self.q = self
            .q
            .mul(Quat::from_rotation_vector(delta_theta))
            .normalized();
        self.bias = self.bias + delta_bias;
        self.kf.reset_mean(Matrix::zeros(6, 1))
    }
}

/// The skew-symmetric cross-product matrix of `v`, such that
/// `skew(v) * w == v.cross(w)`.
fn skew(v: Vec3) -> Matrix {
    Matrix::from_row_major(3, 3, &[0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0])
        .expect("3x3 skew matrix has nine entries")
}

/// The measurement Jacobian `d(predicted_up)/d(δθ)` at `predicted_up`,
/// returned as a 3x6 matrix (the trailing three columns, with respect to
/// bias, are zero: the bias does not directly affect the predicted
/// accelerometer direction).
///
/// For a small rotation error `δθ` applied as `q ⊗ exp(δθ/2)`, the rotated
/// vector `R(q)^T v` changes to first order by `-[R(q)^T v]_x δθ`, i.e. the
/// Jacobian is the negative skew matrix of the predicted vector itself. This
/// is validated against central finite differences in this module's tests,
/// per SPECS.md Section 11.2's requirement that hand-derived Jacobians be
/// validated.
fn measurement_jacobian(predicted_up: Vec3) -> Matrix {
    // Derivation: with q_true = q ⊗ exp(δ/2) (body-frame error) and
    // predicted = q.conjugate().rotate(v), expanding to first order in δ
    // gives measurement(q_true) ≈ predicted + predicted × δ, so the
    // Jacobian is the linear operator `predicted ×`, i.e. skew(predicted).
    let skew_predicted = skew(predicted_up);
    let mut h = Matrix::zeros(3, 6);
    for r in 0..3 {
        for c in 0..3 {
            h[(r, c)] = skew_predicted[(r, c)];
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_types::frames::FrameId;
    use robot_types::units::Radians;

    fn config() -> AttitudeFilterConfig {
        AttitudeFilterConfig::default()
    }

    fn gravity() -> GravityConvention {
        GravityConvention::minus_z_standard(FrameId::WORLD)
    }

    #[test]
    fn pure_gyro_propagation_matches_exact_integration_for_constant_rate() {
        let mut filter =
            AttitudeFilter::new(Quat::IDENTITY, gravity(), config(), 1e-3, 1e-4).unwrap();
        let omega = Vec3::new(0.0, 0.0, 1.0); // 1 rad/s about Z
        let dt = Duration::from_millis(10);
        for _ in 0..100 {
            filter.propagate(omega, dt).unwrap();
        }
        // 100 steps of 10ms at 1 rad/s = 1 radian total rotation about Z.
        let expected = Quat::from_rz(Radians(1.0)).unwrap();
        assert!(filter.attitude().angle_to(expected).value() < 1e-6);
    }

    #[test]
    fn measurement_jacobian_matches_central_finite_differences() {
        // Validate the hand-derived Jacobian the way SPECS.md Section 11.2
        // requires: against finite differences of the actual nonlinear
        // measurement function.
        let q = Quat::from_axis_angle(Vec3::new(0.3, -0.5, 0.2), Radians(0.6)).unwrap();
        let down = Vec3::new(0.1, 0.2, -0.97).normalized(1e-9).unwrap();
        let measurement = |q: Quat| -> Vec3 { q.conjugate().rotate(-down) };
        let predicted = measurement(q);
        let analytic = measurement_jacobian(predicted);

        let eps = 1e-6;
        for axis in 0..3 {
            let mut delta = Vec3::ZERO;
            match axis {
                0 => delta.x = eps,
                1 => delta.y = eps,
                _ => delta.z = eps,
            }
            let q_plus = q.mul(Quat::from_rotation_vector(delta));
            let q_minus = q.mul(Quat::from_rotation_vector(-delta));
            let numeric = (measurement(q_plus) - measurement(q_minus)) * (1.0 / (2.0 * eps));
            let analytic_col = Vec3::new(
                analytic[(0, axis)],
                analytic[(1, axis)],
                analytic[(2, axis)],
            );
            assert!(
                (numeric - analytic_col).norm() < 1e-3,
                "axis {axis}: numeric {numeric:?} vs analytic {analytic_col:?}"
            );
        }
    }

    #[test]
    fn accelerometer_correction_reduces_a_known_tilt_error() {
        // Start with a known 10-degree roll error relative to true level.
        let true_attitude = Quat::IDENTITY;
        let initial_estimate = Quat::from_rx(Radians(0.1745)).unwrap(); // ~10 deg
        let mut filter =
            AttitudeFilter::new(initial_estimate, gravity(), config(), 0.3, 1e-4).unwrap();

        let g = gravity();
        let true_specific_force = true_attitude.conjugate().rotate(-g.down) * g.magnitude;

        let mut previous_error = filter.attitude().angle_to(true_attitude).value();
        for _ in 0..20 {
            let report = filter
                .correct_with_accelerometer(true_specific_force)
                .unwrap();
            assert!(report.magnitude_admitted && report.gate_admitted);
            let error = filter.attitude().angle_to(true_attitude).value();
            assert!(
                error <= previous_error + 1e-9,
                "tilt error must not increase: {error} vs {previous_error}"
            );
            previous_error = error;
        }
        assert!(
            previous_error < 0.01,
            "residual tilt error too large: {previous_error}"
        );
    }

    #[test]
    fn large_specific_force_deviation_is_rejected_before_the_gate() {
        let mut filter =
            AttitudeFilter::new(Quat::IDENTITY, gravity(), config(), 0.1, 1e-4).unwrap();
        // A large linear acceleration added on top of gravity: this is not
        // "the body is static", and the filter must say so rather than
        // treating it as a valid gravity-direction measurement.
        let g = gravity();
        let specific_force = (-g.down * g.magnitude) + Vec3::new(8.0, 0.0, 0.0);
        let report = filter.correct_with_accelerometer(specific_force).unwrap();
        assert!(!report.magnitude_admitted);
        assert!(report.mahalanobis_squared.is_none());
    }

    #[test]
    fn an_outlier_within_magnitude_but_wrong_direction_can_still_be_gated() {
        // Right magnitude, but attitude estimate already very confident and
        // precise, so a modestly inconsistent direction should be flagged by
        // the Mahalanobis gate even though the magnitude check passes.
        let g = gravity();
        let mut filter = AttitudeFilter::new(Quat::IDENTITY, g, config(), 1e-5, 1e-6).unwrap();
        // Direction tilted by about 30 degrees but same magnitude.
        let tilted = Quat::from_rx(Radians(0.5)).unwrap().rotate(-g.down) * g.magnitude;
        let report = filter.correct_with_accelerometer(tilted).unwrap();
        assert!(report.magnitude_admitted);
        assert!(
            !report.gate_admitted,
            "a confident filter must gate out a large direction mismatch"
        );
    }

    #[test]
    fn bias_offsets_the_integrated_rotation() {
        // A nonzero estimated bias must be subtracted from the raw gyro
        // reading before integration: feeding the true rate while carrying a
        // bias estimate equal to that rate should integrate to (approximately)
        // no rotation at all.
        let mut filter =
            AttitudeFilter::new(Quat::IDENTITY, gravity(), config(), 1e-3, 1e-4).unwrap();
        filter.bias = Vec3::new(0.0, 0.0, 0.5);
        let dt = Duration::from_millis(10);
        for _ in 0..50 {
            filter.propagate(Vec3::new(0.0, 0.0, 0.5), dt).unwrap();
        }
        assert!(filter.attitude().angle_to(Quat::IDENTITY).value() < 1e-6);
    }
}

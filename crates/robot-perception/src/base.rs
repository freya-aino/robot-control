//! Floating-base position and velocity, per SPECS.md Section 11.2: "Joint
//! encoders and IMU support state estimation, contact inference".
//!
//! Conventions, stated as Section 11.2 requires:
//!
//! - **State**: base position and velocity in the world frame, `[p; v]`.
//! - **Propagation**: constant acceleration over the actual interval `dt`
//!   (timestamp jitter is handled by using the measured `dt`, never a
//!   nominal one), driven by the world-frame specific force from the IMU
//!   (`R f_body + g`, with `R` from [`crate::attitude::AttitudeFilter`]).
//!   Accelerometer noise enters as process noise `q_acc` (m/s² standard
//!   deviation) through the discrete white-noise-acceleration model.
//! - **Contact assumption**: a foot labelled in contact does not slide.
//!   Its leg kinematics then give a base-velocity measurement,
//!   [`leg_velocity`]. A sliding foot violates the assumption; its
//!   measurement disagrees with the prediction, and the Mahalanobis gate
//!   rejects it. Rejections are counted, and a run of them is the
//!   estimator's slip signal.
//! - **Dropouts**: a step with no admitted measurement only predicts; the
//!   covariance grows, and nothing is extrapolated as if measured.
//! - **Height**: on flat ground of known height, a stance foot also gives
//!   the base height ([`BaseEstimator::correct_height`]), which keeps the
//!   vertical channel from drifting.

use robot_types::frames::{Quat, Vec3};
use robot_types::math::Matrix;
use robot_types::time::Duration;

use crate::linear_kf::{FilterError, KalmanFilter};

/// The 3-degree-of-freedom chi-square value at 0.999.
pub const CHI2_3_999: f64 = 16.266;

/// Base velocity from a stance foot: if the foot is fixed in the world,
/// `0 = v + R (omega_b x r_b + r_b')`, so `v = -R (omega_b x r_b + r_b')`,
/// with `r_b` the foot position in the base frame now, `r_prev` its value
/// `dt` earlier, and `omega_b` the base angular velocity in the base frame.
#[must_use]
pub fn leg_velocity(r_b: Vec3, r_prev: Vec3, dt: f64, rotation: Quat, omega_b: Vec3) -> Vec3 {
    let r_dot = (r_b - r_prev) * (1.0 / dt);
    rotation.rotate(omega_b.cross(r_b) + r_dot) * -1.0
}

/// Counters.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct BaseStats {
    /// Leg velocity measurements admitted.
    pub admitted: u64,
    /// Leg velocity measurements rejected by the gate (slip or outlier).
    pub rejected: u64,
    /// Steps with no admitted measurement.
    pub predict_only: u64,
}

/// The base estimator.
pub struct BaseEstimator {
    kf: KalmanFilter,
    q_acc: f64,
    r_vel: f64,
    gate: f64,
    /// Counters.
    pub stats: BaseStats,
}

impl BaseEstimator {
    /// Starts at `position`/`velocity` with standard deviations `sigma_p`,
    /// `sigma_v`; `q_acc` is the accelerometer noise and `r_vel` the leg
    /// velocity noise (standard deviations).
    pub fn new(
        position: Vec3,
        velocity: Vec3,
        sigma_p: f64,
        sigma_v: f64,
        q_acc: f64,
        r_vel: f64,
    ) -> Result<Self, FilterError> {
        let x0 = Matrix::column(&[
            position.x, position.y, position.z, velocity.x, velocity.y, velocity.z,
        ]);
        let mut p0 = Matrix::zeros(6, 6);
        for i in 0..3 {
            p0[(i, i)] = sigma_p * sigma_p;
            p0[(i + 3, i + 3)] = sigma_v * sigma_v;
        }
        Ok(Self {
            kf: KalmanFilter::new(x0, p0)?,
            q_acc,
            r_vel,
            gate: CHI2_3_999,
            stats: BaseStats::default(),
        })
    }

    /// Position estimate.
    #[must_use]
    pub fn position(&self) -> Vec3 {
        let m = self.kf.mean();
        Vec3::new(m[(0, 0)], m[(1, 0)], m[(2, 0)])
    }

    /// Velocity estimate.
    #[must_use]
    pub fn velocity(&self) -> Vec3 {
        let m = self.kf.mean();
        Vec3::new(m[(3, 0)], m[(4, 0)], m[(5, 0)])
    }

    /// The 6x6 covariance.
    #[must_use]
    pub fn covariance(&self) -> &Matrix {
        self.kf.covariance()
    }

    /// Propagates by `dt` under world-frame acceleration `accel`.
    pub fn propagate(&mut self, accel: Vec3, dt: Duration) -> Result<(), FilterError> {
        let t = dt.as_secs_f64();
        let mut f = Matrix::identity(6);
        let mut q = Matrix::zeros(6, 6);
        let s2 = self.q_acc * self.q_acc;
        for i in 0..3 {
            f[(i, i + 3)] = t;
            q[(i, i)] = s2 * t.powi(4) / 4.0;
            q[(i, i + 3)] = s2 * t.powi(3) / 2.0;
            q[(i + 3, i)] = s2 * t.powi(3) / 2.0;
            q[(i + 3, i + 3)] = s2 * t * t;
        }
        self.kf.predict(&f, &q)?;
        let mut x = self.kf.mean().clone();
        let a = accel.to_xyz();
        for i in 0..3 {
            x[(i, 0)] += 0.5 * a[i] * t * t;
            x[(i + 3, 0)] += a[i] * t;
        }
        self.kf.reset_mean(x)
    }

    /// Corrects with a stance-leg velocity measurement; returns whether
    /// the gate admitted it.
    pub fn correct_velocity(&mut self, measured: Vec3) -> Result<bool, FilterError> {
        let mut h = Matrix::zeros(3, 6);
        for i in 0..3 {
            h[(i, i + 3)] = 1.0;
        }
        let r = Matrix::diagonal(&[self.r_vel.powi(2); 3]);
        let z = Matrix::column(&measured.to_xyz());
        let ok = self.kf.gated_update(&h, &r, &z, self.gate)?;
        if ok {
            self.stats.admitted += 1;
        } else {
            self.stats.rejected += 1;
        }
        Ok(ok)
    }

    /// Corrects the base height from a stance foot on flat ground.
    pub fn correct_height(&mut self, measured_z: f64, sigma: f64) -> Result<bool, FilterError> {
        let mut h = Matrix::zeros(1, 6);
        h[(0, 2)] = 1.0;
        let r = Matrix::diagonal(&[sigma * sigma]);
        self.kf
            .gated_update(&h, &r, &Matrix::column(&[measured_z]), 10.83)
    }

    /// Records a step on which nothing was admitted.
    pub fn note_predict_only(&mut self) {
        self.stats.predict_only += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consistency::Consistency;
    use robot_types::rng::{Rng, SeedStream};

    #[test]
    fn the_leg_velocity_formula_matches_a_rigidly_moving_base() {
        // Base moving at v with rotation rate w about z; a fixed world foot
        // seen from the base: r_b(t) = R(t)^T (foot - p(t)).
        let (v, w) = (Vec3::new(0.3, -0.1, 0.0), 0.5);
        let foot = Vec3::new(0.1, 0.2, -0.7);
        let r_of = |t: f64| {
            let r = Quat::from_rz(robot_types::units::Radians(w * t)).unwrap();
            r.conjugate().rotate(foot - v * t)
        };
        let dt = 1e-6;
        let rot = Quat::from_rz(robot_types::units::Radians(w * 1.0)).unwrap();
        let est = leg_velocity(r_of(1.0), r_of(1.0 - dt), dt, rot, Vec3::new(0.0, 0.0, w));
        assert!((est - v).norm() < 1e-5, "{est:?}");
    }

    /// Truth: a base swaying 0.2 m sinusoidally along x.
    fn truth(t: f64) -> (Vec3, Vec3, Vec3) {
        (
            Vec3::new(0.2 * t.sin(), 0.0, 0.0),
            Vec3::new(0.2 * t.cos(), 0.0, 0.0),
            Vec3::new(-0.2 * t.sin(), 0.0, 0.0),
        )
    }

    /// IMU and leg measurements with noise, jittered timestamps, a dropout
    /// every tenth step, and optionally 50 steps of a sliding foot.
    fn run(slip_from: Option<usize>) -> (BaseEstimator, Consistency, Vec3) {
        let mut rng = Rng::new(SeedStream::new(11, 0));
        let mut est =
            BaseEstimator::new(Vec3::ZERO, Vec3::new(0.2, 0.0, 0.0), 0.01, 0.05, 0.2, 0.03)
                .unwrap();
        let mut c = Consistency::new(6);
        let mut t = 0.0;
        let noise = |rng: &mut Rng, s: f64| {
            Vec3::new(rng.normal(0.0, s), rng.normal(0.0, s), rng.normal(0.0, s))
        };
        for k in 0..2000 {
            let dt = 0.005 + rng.uniform(-0.001, 0.001);
            t += dt;
            let (p, v, a) = truth(t);
            est.propagate(a + noise(&mut rng, 0.2), Duration::from_secs_f64(dt))
                .unwrap();
            if k % 10 == 7 {
                est.note_predict_only();
            } else {
                let mut m = v + noise(&mut rng, 0.03);
                if slip_from.is_some_and(|s| (s..s + 50).contains(&k)) {
                    m = m + Vec3::new(1.0, 0.0, 0.0);
                }
                est.correct_velocity(m).unwrap();
                est.correct_height(p.z + rng.normal(0.0, 0.005), 0.005)
                    .unwrap();
            }
            if k > 200 {
                let (ep, ev) = (est.position() - p, est.velocity() - v);
                c.add(&[ep.x, ep.y, ep.z, ev.x, ev.y, ev.z], est.covariance())
                    .unwrap();
            }
        }
        (est, c, truth(t).1)
    }

    #[test]
    fn the_estimate_is_accurate_and_consistent_under_jitter_noise_and_dropouts() {
        let (est, c, v) = run(None);
        assert!(
            (est.velocity() - v).norm() < 0.05,
            "{:?} vs {v:?}",
            est.velocity()
        );
        let s = c.summary();
        assert!(s.mean_nees > 0.5 * 6.0 && s.mean_nees < 2.0 * 6.0, "{s:?}");
        assert!(est.stats.predict_only > 0);
    }

    #[test]
    fn a_slipping_foot_is_gated_out_rather_than_believed() {
        let (est, _, v) = run(Some(1000));
        assert!(est.stats.rejected >= 40, "{:?}", est.stats);
        assert!((est.velocity() - v).norm() < 0.05);
    }
}

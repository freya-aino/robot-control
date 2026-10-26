//! A generic discrete-time linear-Gaussian Kalman filter.
//!
//! This is the authored baseline SPECS.md Appendix B.2.9 asks to compare
//! qualified candidates (`multicalc`, `numeris`, `typed_kalman`, `kfilter`)
//! against — "Audit the actual humanoid IMU/contact formulation before
//! selecting" — and, per `docs/decisions/0003-std-only-default-closure.md`,
//! the thing this project runs until one of those clears Appendix B.6.
//!
//! The filter is intentionally generic over state and measurement dimension
//! rather than hard-coded to one humanoid state vector, so [`attitude`] and
//! any later contact/velocity estimator can both build on it.

use core::fmt;
use robot_types::math::{LinalgError, Matrix};

/// Why a filter step could not be computed.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum FilterError {
    /// A matrix dimension did not match the filter's configured state or
    /// measurement size.
    DimensionMismatch {
        /// What was expected.
        expected: (usize, usize),
        /// What was found.
        found: (usize, usize),
    },
    /// The innovation covariance was singular and could not be inverted.
    SingularInnovationCovariance,
    /// An underlying linear-algebra operation failed.
    Linalg(LinalgError),
}

impl fmt::Display for FilterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, found } => write!(
                f,
                "expected a {}x{} matrix, found {}x{}",
                expected.0, expected.1, found.0, found.1
            ),
            Self::SingularInnovationCovariance => {
                write!(
                    f,
                    "innovation covariance is singular; measurement update refused"
                )
            }
            Self::Linalg(e) => write!(f, "{e}"),
        }
    }
}

impl core::error::Error for FilterError {}

impl From<LinalgError> for FilterError {
    fn from(e: LinalgError) -> Self {
        Self::Linalg(e)
    }
}

/// A linear-Gaussian Kalman filter over an `n`-dimensional state.
///
/// Holds the mean `x` (`n`x1) and covariance `p` (`n`x`n`). Process and
/// measurement models are supplied per call rather than stored, so the same
/// filter instance can be driven by a time-varying model (as an attitude
/// filter's Jacobian typically is) without rebuilding it.
#[derive(Clone, Debug)]
pub struct KalmanFilter {
    state_dim: usize,
    x: Matrix,
    p: Matrix,
}

impl KalmanFilter {
    /// Builds a filter from an initial mean and covariance.
    pub fn new(x0: Matrix, p0: Matrix) -> Result<Self, FilterError> {
        let n = x0.rows();
        if x0.cols() != 1 {
            return Err(FilterError::DimensionMismatch {
                expected: (n, 1),
                found: (x0.rows(), x0.cols()),
            });
        }
        if p0.rows() != n || p0.cols() != n {
            return Err(FilterError::DimensionMismatch {
                expected: (n, n),
                found: (p0.rows(), p0.cols()),
            });
        }
        Ok(Self {
            state_dim: n,
            x: x0,
            p: p0,
        })
    }

    /// State dimension.
    #[must_use]
    pub const fn state_dim(&self) -> usize {
        self.state_dim
    }

    /// Current mean estimate.
    #[must_use]
    pub fn mean(&self) -> &Matrix {
        &self.x
    }

    /// Current covariance estimate.
    #[must_use]
    pub fn covariance(&self) -> &Matrix {
        &self.p
    }

    /// Overwrites the mean. Used by an error-state filter after injecting a
    /// correction into a held-separately nonlinear state (e.g. a quaternion)
    /// and resetting the linear error state to zero.
    pub fn reset_mean(&mut self, x: Matrix) -> Result<(), FilterError> {
        if x.rows() != self.state_dim || x.cols() != 1 {
            return Err(FilterError::DimensionMismatch {
                expected: (self.state_dim, 1),
                found: (x.rows(), x.cols()),
            });
        }
        self.x = x;
        Ok(())
    }

    /// Predicts forward one step: `x' = F x`, `P' = F P Fᵀ + Q`.
    pub fn predict(&mut self, f: &Matrix, q: &Matrix) -> Result<(), FilterError> {
        let n = self.state_dim;
        if f.rows() != n || f.cols() != n {
            return Err(FilterError::DimensionMismatch {
                expected: (n, n),
                found: (f.rows(), f.cols()),
            });
        }
        if q.rows() != n || q.cols() != n {
            return Err(FilterError::DimensionMismatch {
                expected: (n, n),
                found: (q.rows(), q.cols()),
            });
        }
        self.x = f.mul(&self.x)?;
        let ft = f.transposed();
        self.p = f.mul(&self.p)?.mul(&ft)?.add(q)?;
        Ok(())
    }

    /// Computes the innovation `y = z - H x` and its covariance
    /// `S = H P Hᵀ + R`, without applying the update. Exposed separately so
    /// a caller can gate on `S` (and the resulting Mahalanobis distance)
    /// before committing to [`KalmanFilter::apply_update`] — the
    /// "measurement gating" SPECS.md Section 11.2 requires.
    pub fn innovation(
        &self,
        h: &Matrix,
        r: &Matrix,
        z: &Matrix,
    ) -> Result<(Matrix, Matrix), FilterError> {
        let n = self.state_dim;
        let m = h.rows();
        if h.cols() != n {
            return Err(FilterError::DimensionMismatch {
                expected: (m, n),
                found: (h.rows(), h.cols()),
            });
        }
        if r.rows() != m || r.cols() != m {
            return Err(FilterError::DimensionMismatch {
                expected: (m, m),
                found: (r.rows(), r.cols()),
            });
        }
        if z.rows() != m || z.cols() != 1 {
            return Err(FilterError::DimensionMismatch {
                expected: (m, 1),
                found: (z.rows(), z.cols()),
            });
        }
        let predicted = h.mul(&self.x)?;
        let y = z.sub(&predicted)?;
        let s = h.mul(&self.p)?.mul(&h.transposed())?.add(r)?;
        Ok((y, s))
    }

    /// The squared Mahalanobis distance of an innovation under its own
    /// covariance: `yᵀ S⁻¹ y`. This is the gating statistic — a chi-squared
    /// variable under the filter's own noise model, which is what lets a
    /// caller reject an outlier measurement with a principled threshold
    /// rather than an arbitrary one.
    pub fn mahalanobis_squared(&self, y: &Matrix, s: &Matrix) -> Result<f64, FilterError> {
        let s_inv_y = s
            .solve(y)
            .map_err(|_| FilterError::SingularInnovationCovariance)?;
        let result = y.transposed().mul(&s_inv_y)?;
        Ok(result[(0, 0)])
    }

    /// Applies a Kalman update given a precomputed innovation and its
    /// covariance: `K = P Hᵀ S⁻¹`, `x' = x + K y`, `P' = (I - K H) P`.
    ///
    /// Uses the Joseph form for the covariance update
    /// (`P' = (I-KH) P (I-KH)ᵀ + K R Kᵀ`) so that accumulated rounding or a
    /// slightly wrong gain cannot produce a non-positive-semidefinite
    /// covariance, which is the numerical-robustness property SPECS.md
    /// Section 11.2's "covariance consistency" requirement is checking for.
    pub fn apply_update(
        &mut self,
        h: &Matrix,
        r: &Matrix,
        y: &Matrix,
        s: &Matrix,
    ) -> Result<(), FilterError> {
        let n = self.state_dim;
        let k = self
            .p
            .mul(&h.transposed())?
            .mul(&s.solve(&Matrix::identity(s.rows()))?)?;
        self.x = self.x.add(&k.mul(y)?)?;
        let ikh = Matrix::identity(n).sub(&k.mul(h)?)?;
        let joseph = ikh
            .mul(&self.p)?
            .mul(&ikh.transposed())?
            .add(&k.mul(r)?.mul(&k.transposed())?)?;
        self.p = joseph;
        Ok(())
    }

    /// Computes the innovation and, if its Mahalanobis distance is within
    /// `gate`, applies the update. Returns whether the update was applied.
    ///
    /// This is the single entry point most callers should use; `innovation`
    /// and `apply_update` remain available separately for a caller that
    /// needs to log the rejected innovation's statistics too.
    pub fn gated_update(
        &mut self,
        h: &Matrix,
        r: &Matrix,
        z: &Matrix,
        gate: f64,
    ) -> Result<bool, FilterError> {
        let (y, s) = self.innovation(h, r, z)?;
        let d2 = self.mahalanobis_squared(&y, &s)?;
        if d2 > gate {
            return Ok(false);
        }
        self.apply_update(h, r, &y, &s)?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 1-D constant-velocity tracker: state [position, velocity].
    fn cv_matrices(dt: f64, process_noise: f64) -> (Matrix, Matrix) {
        let f = Matrix::from_row_major(2, 2, &[1.0, dt, 0.0, 1.0]).unwrap();
        // Standard discretized white-noise-acceleration process covariance.
        let q = Matrix::from_row_major(
            2,
            2,
            &[
                dt.powi(4) / 4.0 * process_noise,
                dt.powi(3) / 2.0 * process_noise,
                dt.powi(3) / 2.0 * process_noise,
                dt.powi(2) * process_noise,
            ],
        )
        .unwrap();
        (f, q)
    }

    #[test]
    fn predict_only_tracks_a_constant_velocity_exactly_in_mean() {
        let (f, q) = cv_matrices(0.1, 0.0);
        let x0 = Matrix::column(&[0.0, 2.0]); // position 0, velocity 2 m/s
        let p0 = Matrix::diagonal(&[0.01, 0.01]);
        let mut kf = KalmanFilter::new(x0, p0).unwrap();
        for _ in 0..10 {
            kf.predict(&f, &q).unwrap();
        }
        // After 10 steps of 0.1s at 2 m/s: position should be 2.0 m.
        assert!((kf.mean()[(0, 0)] - 2.0).abs() < 1e-9);
        assert!((kf.mean()[(1, 0)] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn a_perfect_measurement_collapses_covariance_toward_zero() {
        let (f, q) = cv_matrices(0.1, 0.001);
        let x0 = Matrix::column(&[0.0, 0.0]);
        let p0 = Matrix::diagonal(&[10.0, 10.0]);
        let mut kf = KalmanFilter::new(x0, p0).unwrap();
        let h = Matrix::from_row_major(1, 2, &[1.0, 0.0]).unwrap();
        let r = Matrix::diagonal(&[1e-8]); // near-perfect position sensor

        for _ in 0..50 {
            kf.predict(&f, &q).unwrap();
            let applied = kf
                .gated_update(&h, &r, &Matrix::column(&[1.0]), 1e9)
                .unwrap();
            assert!(applied);
        }
        assert!((kf.mean()[(0, 0)] - 1.0).abs() < 1e-3);
        assert!(
            kf.covariance()[(0, 0)] < 1e-4,
            "position variance should collapse: {}",
            kf.covariance()[(0, 0)]
        );
    }

    #[test]
    fn gating_rejects_an_outlier_measurement() {
        let (f, q) = cv_matrices(0.1, 0.001);
        let x0 = Matrix::column(&[0.0, 0.0]);
        let p0 = Matrix::diagonal(&[1.0, 1.0]);
        let mut kf = KalmanFilter::new(x0, p0).unwrap();
        let h = Matrix::from_row_major(1, 2, &[1.0, 0.0]).unwrap();
        let r = Matrix::diagonal(&[0.01]);

        kf.predict(&f, &q).unwrap();
        // A measurement 1000 units away is an obvious outlier under R=0.01.
        let applied = kf
            .gated_update(&h, &r, &Matrix::column(&[1000.0]), 9.21)
            .unwrap();
        assert!(!applied, "a wild outlier must be rejected by the gate");
        assert!(
            kf.mean()[(0, 0)].abs() < 1.0,
            "rejected update must not move the state"
        );
    }

    #[test]
    fn covariance_stays_symmetric_and_positive_semidefinite_after_many_updates() {
        let (f, q) = cv_matrices(0.05, 0.01);
        let x0 = Matrix::column(&[5.0, -1.0]);
        let p0 = Matrix::diagonal(&[2.0, 2.0]);
        let mut kf = KalmanFilter::new(x0, p0).unwrap();
        let h = Matrix::from_row_major(1, 2, &[1.0, 0.0]).unwrap();
        let r = Matrix::diagonal(&[0.5]);

        for i in 0..200 {
            kf.predict(&f, &q).unwrap();
            let z = Matrix::column(&[-(i as f64 * 0.05)]);
            kf.gated_update(&h, &r, &z, 50.0).unwrap();
        }
        let p = kf.covariance();
        // Symmetry.
        assert!(
            (p[(0, 1)] - p[(1, 0)]).abs() < 1e-9,
            "covariance must stay symmetric"
        );
        // Positive semidefiniteness via eigenvalues.
        let (eigs, _) = p.symmetric_eigen().unwrap();
        for e in eigs {
            assert!(e >= -1e-9, "covariance must stay PSD, got eigenvalue {e}");
        }
    }

    #[test]
    fn dimension_mismatches_are_rejected_rather_than_panicking() {
        let kf = KalmanFilter::new(Matrix::column(&[0.0, 0.0]), Matrix::diagonal(&[1.0, 1.0]));
        assert!(kf.is_ok());
        let bad = KalmanFilter::new(Matrix::column(&[0.0]), Matrix::diagonal(&[1.0, 1.0]));
        assert!(matches!(bad, Err(FilterError::DimensionMismatch { .. })));

        let mut kf = kf.unwrap();
        let wrong_f = Matrix::identity(3);
        let q = Matrix::diagonal(&[1.0, 1.0]);
        assert!(matches!(
            kf.predict(&wrong_f, &q),
            Err(FilterError::DimensionMismatch { .. })
        ));
    }
}

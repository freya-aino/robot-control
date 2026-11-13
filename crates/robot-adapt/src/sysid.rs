//! Linear system identification: the first row of SPECS.md Section 11.3's
//! ordered adaptation table.
//!
//! "System identification | Time-synchronized commands and measured
//! responses; update simulator parameters offline | Parameters may not be
//! identifiable from passive logs." The identifiability half of that row is
//! not a caveat this module states and ignores: [`identify`] reports the
//! regressor's singular values, and a caller is expected to treat a small
//! one as "this parameter combination cannot be trusted from this data,"
//! per SPECS.md Section 11.3's own requirement to "preserve uncertainty for
//! parameters the data cannot distinguish."
//!
//! This is ordinary linear least squares over a user-supplied feature
//! model: the caller decides what physical model (viscous friction,
//! Coulomb friction, a gear-ratio correction) the regressor columns
//! represent; this module only solves for the best-fit coefficients and
//! reports how well-determined they are.

use core::fmt;
use robot_types::math::{LinalgError, Matrix};

/// One identification sample: a feature row and the measured response it
/// should predict.
#[derive(Clone, Debug)]
pub struct Sample {
    /// Feature values (the regressor row), e.g. `[velocity, sign(velocity),
    /// 1.0]` for a viscous + Coulomb + offset friction model.
    pub features: Vec<f64>,
    /// The measured response this sample's features should predict.
    pub response: f64,
}

/// Result of a linear identification.
#[derive(Clone, PartialEq, Debug)]
pub struct IdentificationResult {
    /// Best-fit coefficients, one per feature column.
    pub coefficients: Vec<f64>,
    /// Root-mean-square residual of the fit.
    pub rms_residual: f64,
    /// Singular values of the feature matrix, descending. A value near
    /// zero (relative to the largest) means the corresponding parameter
    /// combination is not identifiable from this data — SPECS.md Section
    /// 11.3's "Parameters may not be identifiable from passive logs,"
    /// made a number a caller can threshold rather than prose to remember.
    pub singular_values: Vec<f64>,
    /// One standard error per coefficient (residual variance times the
    /// diagonal of the regularized inverse normal matrix).
    pub standard_errors: Vec<f64>,
}

impl IdentificationResult {
    /// Ratio of the smallest to the largest singular value — the standard
    /// conditioning indicator. A caller comparing this against a declared
    /// tolerance is what "preserve uncertainty for parameters the data
    /// cannot distinguish" (SPECS.md Section 11.3) means in practice; no
    /// universal tolerance is assumed here.
    #[must_use]
    pub fn condition_ratio(&self) -> f64 {
        match (self.singular_values.first(), self.singular_values.last()) {
            (Some(&max), Some(&min)) if max > 0.0 => min / max,
            _ => 0.0,
        }
    }
}

/// Why identification failed.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum IdentificationError {
    /// No samples were supplied.
    Empty,
    /// Samples had inconsistent feature widths.
    InconsistentWidth {
        /// Width of the first sample.
        expected: usize,
        /// Width found on a later sample.
        found: usize,
    },
    /// The underlying linear solve failed (e.g. a singular normal matrix
    /// with zero regularization).
    Linalg(LinalgError),
}

impl fmt::Display for IdentificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "no samples supplied"),
            Self::InconsistentWidth { expected, found } => {
                write!(f, "feature width {found}, expected {expected}")
            }
            Self::Linalg(e) => write!(f, "{e}"),
        }
    }
}

impl core::error::Error for IdentificationError {}

impl From<LinalgError> for IdentificationError {
    fn from(e: LinalgError) -> Self {
        Self::Linalg(e)
    }
}

/// Identifies the least-squares coefficients `c` minimizing
/// `sum_i (features_i . c - response_i)^2`, via Tikhonov-regularized
/// normal equations.
///
/// `ridge` is a small regularization added to the normal matrix's diagonal
/// so an under-determined or ill-conditioned fit still returns *a* result
/// (with a correspondingly reported small singular value) rather than
/// failing outright — use `0.0` for an unregularized ordinary least squares
/// fit when the regressor is known well-conditioned.
pub fn identify(
    samples: &[Sample],
    ridge: f64,
) -> Result<IdentificationResult, IdentificationError> {
    let Some(first) = samples.first() else {
        return Err(IdentificationError::Empty);
    };
    let width = first.features.len();
    for s in samples {
        if s.features.len() != width {
            return Err(IdentificationError::InconsistentWidth {
                expected: width,
                found: s.features.len(),
            });
        }
    }

    let rows: Vec<f64> = samples
        .iter()
        .flat_map(|s| s.features.iter().copied())
        .collect();
    let design = Matrix::from_row_major(samples.len(), width, &rows)
        .expect("row count matches by construction");
    let responses = Matrix::column(&samples.iter().map(|s| s.response).collect::<Vec<_>>());

    let design_t = design.transposed();
    let mut normal = design_t.mul(&design)?;
    normal.add_to_diagonal(ridge);
    let rhs = design_t.mul(&responses)?;
    let solution = normal.solve(&rhs)?;
    let coefficients: Vec<f64> = (0..width).map(|i| solution[(i, 0)]).collect();

    let predicted = design.mul(&solution)?;
    let residual = predicted.sub(&responses)?;
    let rms_residual =
        (residual.as_slice().iter().map(|r| r * r).sum::<f64>() / samples.len() as f64).sqrt();

    let singular_values = design.singular_values()?;

    // Standard errors: sqrt(diag(sigma^2 (X'X + ridge I)^-1)) with the
    // residual variance on n - p degrees of freedom. An unidentifiable
    // direction shows up here as a huge error, next to its small singular
    // value.
    let dof = samples.len().saturating_sub(width).max(1) as f64;
    let sigma2 = residual.as_slice().iter().map(|r| r * r).sum::<f64>() / dof;
    let inverse = normal.inverse()?;
    let standard_errors = (0..width)
        .map(|i| (sigma2 * inverse[(i, i)]).max(0.0).sqrt())
        .collect();

    Ok(IdentificationResult {
        coefficients,
        rms_residual,
        singular_values,
        standard_errors,
    })
}

/// One sample of a single joint's motion and the torque that drove it.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct JointSample {
    /// Velocity, rad/s.
    pub velocity: f64,
    /// Acceleration, rad/s^2.
    pub acceleration: f64,
    /// Commanded or measured torque, N m.
    pub torque: f64,
}

/// A joint's identified rigid-body and friction parameters.
#[derive(Clone, PartialEq, Debug)]
pub struct JointDynamics {
    /// Reflected inertia, kg m^2.
    pub inertia: f64,
    /// Viscous damping, N m s/rad.
    pub viscous: f64,
    /// Coulomb friction, N m.
    pub coulomb: f64,
    /// Constant torque offset (gravity at a fixed posture, sensor bias).
    pub offset: f64,
    /// The underlying fit, for its standard errors and identifiability.
    pub fit: IdentificationResult,
}

/// Fits `torque = I qdd + b qd + c sign(qd) + offset` to `samples`.
///
/// Samples within `dead_band` of zero velocity are dropped: the sign of a
/// velocity that small is noise, and Coulomb friction there is stiction,
/// which this model does not describe.
pub fn identify_joint(
    samples: &[JointSample],
    dead_band: f64,
    ridge: f64,
) -> Result<JointDynamics, IdentificationError> {
    let rows: Vec<Sample> = samples
        .iter()
        .filter(|s| s.velocity.abs() > dead_band)
        .map(|s| Sample {
            features: vec![s.acceleration, s.velocity, s.velocity.signum(), 1.0],
            response: s.torque,
        })
        .collect();
    let fit = identify(&rows, ridge)?;
    Ok(JointDynamics {
        inertia: fit.coefficients[0],
        viscous: fit.coefficients[1],
        coulomb: fit.coefficients[2],
        offset: fit.coefficients[3],
        fit,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_known_coefficients_from_noiseless_data() {
        // response = 2*x0 - 3*x1 + 1 (features: [x0, x1, 1.0])
        let samples: Vec<Sample> = (0..20)
            .map(|i| {
                let x0 = i as f64 * 0.3;
                let x1 = (i as f64 * 0.7).sin();
                Sample {
                    features: vec![x0, x1, 1.0],
                    response: 2.0 * x0 - 3.0 * x1 + 1.0,
                }
            })
            .collect();
        let result = identify(&samples, 0.0).unwrap();
        assert!((result.coefficients[0] - 2.0).abs() < 1e-8);
        assert!((result.coefficients[1] - (-3.0)).abs() < 1e-8);
        assert!((result.coefficients[2] - 1.0).abs() < 1e-8);
        assert!(result.rms_residual < 1e-8);
    }

    #[test]
    fn an_unidentifiable_parameter_combination_is_flagged_by_a_small_singular_value() {
        // x1 is always exactly 2*x0: the two cannot be told apart from this
        // data, so the regressor is rank-deficient.
        let samples: Vec<Sample> = (1..10)
            .map(|i| {
                let x0 = i as f64;
                Sample {
                    features: vec![x0, 2.0 * x0],
                    response: 3.0 * x0,
                }
            })
            .collect();
        let result = identify(&samples, 1e-9).unwrap();
        let ratio = result.condition_ratio();
        assert!(
            ratio < 1e-6,
            "expected a near-singular regressor, condition ratio {ratio}"
        );
    }

    #[test]
    fn a_joint_is_identified_with_errors_that_shrink_with_data() {
        let truth = |qd: f64, qdd: f64| 0.05 * qdd + 0.3 * qd + 0.8 * qd.signum() + 1.5;
        let mut rng = robot_types::rng::Rng::new(robot_types::rng::SeedStream::new(3, 0));
        let mut make = |n: usize| -> Vec<JointSample> {
            (0..n)
                .map(|i| {
                    let t = i as f64 * 0.01;
                    let (qd, qdd) = (2.0 * (3.0 * t).sin(), 6.0 * (3.0 * t).cos());
                    JointSample {
                        velocity: qd,
                        acceleration: qdd,
                        torque: truth(qd, qdd) + rng.normal(0.0, 0.01),
                    }
                })
                .collect()
        };
        let small = identify_joint(&make(200), 0.05, 0.0).unwrap();
        let large = identify_joint(&make(5000), 0.05, 0.0).unwrap();
        assert!((large.inertia - 0.05).abs() < 3.0 * large.fit.standard_errors[0] + 1e-4);
        assert!((large.viscous - 0.3).abs() < 1e-2);
        assert!((large.coulomb - 0.8).abs() < 1e-2);
        assert!((large.offset - 1.5).abs() < 1e-2);
        assert!(large.fit.standard_errors[2] < small.fit.standard_errors[2]);
    }

    #[test]
    fn inconsistent_feature_widths_are_rejected() {
        let samples = vec![
            Sample {
                features: vec![1.0, 2.0],
                response: 1.0,
            },
            Sample {
                features: vec![1.0],
                response: 1.0,
            },
        ];
        assert!(matches!(
            identify(&samples, 0.0),
            Err(IdentificationError::InconsistentWidth {
                expected: 2,
                found: 1
            })
        ));
    }

    #[test]
    fn an_empty_sample_set_is_rejected() {
        assert!(matches!(
            identify(&[], 0.0),
            Err(IdentificationError::Empty)
        ));
    }
}

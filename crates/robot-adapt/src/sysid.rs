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
#[derive(Clone, Debug)]
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

    Ok(IdentificationResult {
        coefficients,
        rms_residual,
        singular_values,
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

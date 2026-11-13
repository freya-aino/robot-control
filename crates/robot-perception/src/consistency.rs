//! Covariance consistency against synthetic truth, per SPECS.md Section
//! 11.2: "Compare observable-state error and covariance consistency
//! against analytic or synthetic truth; no universal percentage-divergence
//! threshold is assumed."
//!
//! The normalized estimation error squared, `NEES = e' P^-1 e`, of a
//! consistent filter with an `n`-dimensional state is chi-square with `n`
//! degrees of freedom: mean `n`. [`Consistency`] accumulates it over a run
//! and reports the mean and the fraction of samples inside the two-sided
//! 95 % chi-square interval, so a filter that is overconfident (NEES too
//! large) or underconfident (too small) shows it as a number rather than
//! as a feeling about the plots.

use robot_types::math::{LinalgError, Matrix};

/// The two-sided 95 % chi-square interval for `n` degrees of freedom,
/// by the Wilson-Hilferty approximation (within a few per cent for
/// `n >= 3`).
#[must_use]
pub fn chi2_interval_95(n: usize) -> (f64, f64) {
    let k = n as f64;
    let c = 2.0 / (9.0 * k);
    let at = |z: f64| k * (1.0 - c + z * c.sqrt()).powi(3);
    (at(-1.959_964).max(0.0), at(1.959_964))
}

/// A NEES accumulator.
#[derive(Clone, Debug)]
pub struct Consistency {
    n: usize,
    samples: Vec<f64>,
}

/// Its summary.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ConsistencySummary {
    /// State dimension.
    pub n: usize,
    /// Samples.
    pub count: usize,
    /// Mean NEES (ideal: `n`).
    pub mean_nees: f64,
    /// Fraction of samples inside the 95 % interval (ideal: 0.95).
    pub inside_95: f64,
}

impl Consistency {
    /// An accumulator for an `n`-dimensional state.
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            n,
            samples: Vec::new(),
        }
    }

    /// Adds an error vector and its covariance.
    pub fn add(&mut self, error: &[f64], covariance: &Matrix) -> Result<f64, LinalgError> {
        let e = Matrix::column(error);
        let w = covariance.solve(&e)?;
        let nees: f64 = error.iter().zip(w.as_slice()).map(|(a, b)| a * b).sum();
        self.samples.push(nees);
        Ok(nees)
    }

    /// The summary.
    #[must_use]
    pub fn summary(&self) -> ConsistencySummary {
        let (lo, hi) = chi2_interval_95(self.n);
        let count = self.samples.len();
        let c = count.max(1) as f64;
        ConsistencySummary {
            n: self.n,
            count,
            mean_nees: self.samples.iter().sum::<f64>() / c,
            inside_95: self
                .samples
                .iter()
                .filter(|v| (lo..=hi).contains(*v))
                .count() as f64
                / c,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_types::rng::{Rng, SeedStream};

    #[test]
    fn errors_drawn_from_the_stated_covariance_are_consistent() {
        let mut rng = Rng::new(SeedStream::new(2, 0));
        let p = Matrix::diagonal(&[0.04, 0.01, 0.09]);
        let mut c = Consistency::new(3);
        for _ in 0..4000 {
            let e = [
                rng.normal(0.0, 0.2),
                rng.normal(0.0, 0.1),
                rng.normal(0.0, 0.3),
            ];
            c.add(&e, &p).unwrap();
        }
        let s = c.summary();
        assert!((s.mean_nees - 3.0).abs() < 0.15, "{s:?}");
        assert!((s.inside_95 - 0.95).abs() < 0.02, "{s:?}");
        // The same errors against a covariance four times too small: NEES
        // four times too large.
        let mut over = Consistency::new(3);
        for _ in 0..1000 {
            let e = [
                rng.normal(0.0, 0.2),
                rng.normal(0.0, 0.1),
                rng.normal(0.0, 0.3),
            ];
            over.add(&e, &p.scaled(0.25)).unwrap();
        }
        assert!(over.summary().mean_nees > 10.0);
    }

    #[test]
    fn the_interval_matches_tables() {
        let (lo, hi) = chi2_interval_95(6);
        assert!(
            (lo - 1.237).abs() < 0.05 && (hi - 14.449).abs() < 0.15,
            "{lo} {hi}"
        );
    }
}

//! Statistics across seeds, for the "multiple seeds and confidence
//! intervals for stochastic outcomes" SPECS.md Section 14 requires.

use core::fmt;

/// Summary statistics over a sample of per-seed results.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SeedStatistics {
    /// Number of seeds.
    pub n: usize,
    /// Sample mean.
    pub mean: f64,
    /// Sample standard deviation (Bessel-corrected, i.e. dividing by n-1).
    pub std_dev: f64,
    /// Approximate 95% confidence half-width on the mean, via the normal
    /// approximation (`1.96 * std_dev / sqrt(n)`). This is an
    /// approximation, not a small-sample-exact interval; a caller with
    /// `n < 30` or a known-skewed metric should not treat it as more
    /// precise than that.
    pub confidence_95_halfwidth: f64,
    /// 95 % confidence half-width with Student's t quantile on `n - 1`
    /// degrees of freedom: the honest interval for a handful of seeds.
    pub t_confidence_95_halfwidth: f64,
}

/// Why statistics could not be computed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct InsufficientSamples {
    /// Samples found.
    pub found: usize,
}

impl fmt::Display for InsufficientSamples {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "need at least 2 samples for a standard deviation, found {}",
            self.found
        )
    }
}

impl core::error::Error for InsufficientSamples {}

/// Computes mean, standard deviation and an approximate 95% confidence
/// half-width over `samples`.
pub fn summarize(samples: &[f64]) -> Result<SeedStatistics, InsufficientSamples> {
    if samples.len() < 2 {
        return Err(InsufficientSamples {
            found: samples.len(),
        });
    }
    let n = samples.len();
    let mean = samples.iter().sum::<f64>() / n as f64;
    let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let std_dev = variance.sqrt();
    let confidence_95_halfwidth = 1.96 * std_dev / (n as f64).sqrt();
    let t_confidence_95_halfwidth =
        crate::distribution::t_quantile(0.975, (n - 1) as f64) * std_dev / (n as f64).sqrt();
    Ok(SeedStatistics {
        n,
        mean,
        std_dev,
        confidence_95_halfwidth,
        t_confidence_95_halfwidth,
    })
}

/// Whether two independent seed samples' 95% confidence intervals around
/// their means overlap. Non-overlapping intervals are reasonably strong
/// evidence of a real difference; overlapping intervals do **not** prove
/// equivalence (that needs a proper equivalence test this module does not
/// implement) — this function reports only what it actually checked.
#[must_use]
pub fn confidence_intervals_overlap(a: &SeedStatistics, b: &SeedStatistics) -> bool {
    let a_lo = a.mean - a.confidence_95_halfwidth;
    let a_hi = a.mean + a.confidence_95_halfwidth;
    let b_lo = b.mean - b.confidence_95_halfwidth;
    let b_hi = b.mean + b.confidence_95_halfwidth;
    a_lo <= b_hi && b_lo <= a_hi
}

/// Welch's unequal-variance t test of `a` against `b`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Welch {
    /// Mean of `b` minus mean of `a`.
    pub difference: f64,
    /// Standard error of the difference.
    pub standard_error: f64,
    /// Welch-Satterthwaite degrees of freedom.
    pub df: f64,
    /// Two-sided p-value for a nonzero difference.
    pub p_two_sided: f64,
}

/// Welch's test between two seed summaries.
#[must_use]
pub fn welch(a: &SeedStatistics, b: &SeedStatistics) -> Welch {
    let (va, vb) = (
        a.std_dev.powi(2) / a.n as f64,
        b.std_dev.powi(2) / b.n as f64,
    );
    let se = (va + vb).sqrt();
    let difference = b.mean - a.mean;
    if se == 0.0 {
        return Welch {
            difference,
            standard_error: 0.0,
            df: (a.n + b.n - 2) as f64,
            p_two_sided: if difference == 0.0 { 1.0 } else { 0.0 },
        };
    }
    let df = (va + vb).powi(2) / (va.powi(2) / (a.n - 1) as f64 + vb.powi(2) / (b.n - 1) as f64);
    let t = difference / se;
    Welch {
        difference,
        standard_error: se,
        df,
        p_two_sided: 2.0 * (1.0 - crate::distribution::t_cdf(t.abs(), df)),
    }
}

/// Two one-sided tests for equivalence within `+-margin`: the p-value for
/// rejecting "the difference is at least `margin` in either direction".
/// Below the caller's alpha, the two are equivalent within the margin;
/// this, not interval overlap, is the statistic for "the same".
#[must_use]
pub fn tost(a: &SeedStatistics, b: &SeedStatistics, margin: f64) -> f64 {
    let w = welch(a, b);
    if w.standard_error == 0.0 {
        return if w.difference.abs() < margin {
            0.0
        } else {
            1.0
        };
    }
    let lower = 1.0 - crate::distribution::t_cdf((w.difference + margin) / w.standard_error, w.df);
    let upper = crate::distribution::t_cdf((w.difference - margin) / w.standard_error, w.df);
    lower.max(upper)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_t_interval_is_wider_than_the_normal_one_for_few_seeds() {
        let s = summarize(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let ratio = s.t_confidence_95_halfwidth / s.confidence_95_halfwidth;
        assert!((ratio - 2.776 / 1.96).abs() < 1e-3, "{ratio}");
    }

    #[test]
    fn welch_separates_distinct_means_and_tost_confirms_equal_ones() {
        let a = summarize(&[1.0, 1.1, 0.9, 1.05, 0.95]).unwrap();
        let b = summarize(&[2.0, 2.1, 1.9, 2.05, 1.95]).unwrap();
        assert!(welch(&a, &b).p_two_sided < 1e-6);
        let c = summarize(&[1.01, 1.09, 0.91, 1.04, 0.96]).unwrap();
        assert!(welch(&a, &c).p_two_sided > 0.5);
        assert!(tost(&a, &c, 0.2) < 0.01);
        assert!(tost(&a, &c, 0.001) > 0.5);
    }

    #[test]
    fn matches_hand_computed_mean_and_std_dev() {
        let samples = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let stats = summarize(&samples).unwrap();
        assert!((stats.mean - 5.0).abs() < 1e-9);
        // Known sample standard deviation of this classic example is 2.138...
        assert!((stats.std_dev - 2.138_089_935).abs() < 1e-6);
        assert_eq!(stats.n, 8);
    }

    #[test]
    fn fewer_than_two_samples_is_rejected() {
        assert!(matches!(
            summarize(&[1.0]),
            Err(InsufficientSamples { found: 1 })
        ));
        assert!(matches!(
            summarize(&[]),
            Err(InsufficientSamples { found: 0 })
        ));
    }

    #[test]
    fn clearly_separated_distributions_do_not_overlap() {
        let a = summarize(&[0.0, 0.01, -0.01, 0.0, 0.01]).unwrap();
        let b = summarize(&[10.0, 10.01, 9.99, 10.0, 10.01]).unwrap();
        assert!(!confidence_intervals_overlap(&a, &b));
    }

    #[test]
    fn identical_distributions_overlap() {
        let a = summarize(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = summarize(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(confidence_intervals_overlap(&a, &b));
    }
}

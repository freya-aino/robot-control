//! Domain randomization, per SPECS.md Section 9.2: "Domain randomization
//! must be calibrated and sensitivity-driven. Include plausible
//! mass/inertia, friction, motor response, backlash where relevant, delay,
//! sensor bias and contact parameters. Reserve held-out disturbances and
//! parameter combinations for evaluation."
//!
//! - **Calibrated**: every [`Parameter`] carries the source of its range
//!   (a system-identification result, a datasheet), and an empty source is
//!   refused.
//! - **Held out**: each parameter may reserve a sub-interval for
//!   evaluation. [`Randomization::sample_training`] never draws from it
//!   (it samples the remaining length uniformly), and
//!   [`Randomization::sample_held_out`] draws only from it, so the split is
//!   by construction, not by hoping two seeds differ.
//! - **Sensitivity-driven**: [`Randomization::sensitivity`] measures how
//!   much a caller's metric moves across each parameter's range with the
//!   others at their midpoints, and returns the parameters ranked, so
//!   effort goes to the ranges that matter.

use robot_types::rng::{Rng, SeedStream};

/// One randomized parameter.
#[derive(Clone, PartialEq, Debug)]
pub struct Parameter {
    /// Name, e.g. `"foot_friction"`.
    pub name: String,
    /// Lower bound.
    pub lo: f64,
    /// Upper bound.
    pub hi: f64,
    /// Sub-interval reserved for evaluation, inside `[lo, hi]`.
    pub held_out: Option<(f64, f64)>,
    /// Where the range comes from.
    pub source: String,
}

/// Why a randomization is invalid.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum RandomizationError {
    /// A range was empty, reversed or non-finite.
    BadRange(String),
    /// The held-out interval is not inside the range or covers all of it.
    BadHeldOut(String),
    /// No calibration source.
    Uncalibrated(String),
}

impl core::fmt::Display for RandomizationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BadRange(n) => write!(f, "{n}: invalid range"),
            Self::BadHeldOut(n) => write!(
                f,
                "{n}: held-out interval must lie strictly inside the range"
            ),
            Self::Uncalibrated(n) => write!(f, "{n}: no calibration source"),
        }
    }
}

impl core::error::Error for RandomizationError {}

/// A validated set of randomized parameters.
#[derive(Clone, PartialEq, Debug)]
pub struct Randomization {
    params: Vec<Parameter>,
}

impl Randomization {
    /// Validates `params`.
    pub fn new(params: Vec<Parameter>) -> Result<Self, RandomizationError> {
        for p in &params {
            if !(p.lo.is_finite() && p.hi.is_finite() && p.lo <= p.hi) {
                return Err(RandomizationError::BadRange(p.name.clone()));
            }
            if p.source.trim().is_empty() {
                return Err(RandomizationError::Uncalibrated(p.name.clone()));
            }
            if let Some((a, b)) = p.held_out {
                let inside = p.lo <= a && a < b && b <= p.hi;
                let leaves_some = (b - a) < (p.hi - p.lo);
                if !(inside && leaves_some) {
                    return Err(RandomizationError::BadHeldOut(p.name.clone()));
                }
            }
        }
        Ok(Self { params })
    }

    /// The parameters.
    #[must_use]
    pub fn parameters(&self) -> &[Parameter] {
        &self.params
    }

    /// A training draw: uniform over each range minus its held-out part.
    #[must_use]
    pub fn sample_training(&self, seed: SeedStream) -> Vec<(String, f64)> {
        let mut rng = Rng::new(seed);
        self.params
            .iter()
            .map(|p| {
                let v = match p.held_out {
                    None => rng.uniform(p.lo, p.hi),
                    Some((a, b)) => {
                        let left = a - p.lo;
                        let u = rng.uniform(0.0, left + (p.hi - b));
                        if u < left {
                            p.lo + u
                        } else {
                            b + (u - left)
                        }
                    }
                };
                (p.name.clone(), v)
            })
            .collect()
    }

    /// An evaluation draw: uniform over each held-out interval (the full
    /// range for a parameter with none).
    #[must_use]
    pub fn sample_held_out(&self, seed: SeedStream) -> Vec<(String, f64)> {
        let mut rng = Rng::new(seed);
        self.params
            .iter()
            .map(|p| {
                let (a, b) = p.held_out.unwrap_or((p.lo, p.hi));
                (p.name.clone(), rng.uniform(a, b))
            })
            .collect()
    }

    /// Each parameter's effect on `metric`: `|metric(hi) - metric(lo)|`
    /// with every other parameter at its midpoint, ranked largest first.
    pub fn sensitivity(
        &self,
        mut metric: impl FnMut(&[(String, f64)]) -> f64,
    ) -> Vec<(String, f64)> {
        let mid: Vec<(String, f64)> = self
            .params
            .iter()
            .map(|p| (p.name.clone(), 0.5 * (p.lo + p.hi)))
            .collect();
        let mut out: Vec<(String, f64)> = self
            .params
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let mut at = mid.clone();
                at[i].1 = p.lo;
                let a = metric(&at);
                at[i].1 = p.hi;
                let b = metric(&at);
                (p.name.clone(), (b - a).abs())
            })
            .collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn friction() -> Parameter {
        Parameter {
            name: "foot_friction".into(),
            lo: 0.4,
            hi: 1.2,
            held_out: Some((0.7, 0.8)),
            source: "sysid run 2026-11-13".into(),
        }
    }

    #[test]
    fn training_never_draws_held_out_values_and_evaluation_only_does() {
        let r = Randomization::new(vec![friction()]).unwrap();
        for s in 0..2000 {
            let v = r.sample_training(SeedStream::new(s, 0))[0].1;
            assert!((0.4..=1.2).contains(&v) && !(0.7..0.8).contains(&v), "{v}");
            let h = r.sample_held_out(SeedStream::new(s, 0))[0].1;
            assert!((0.7..=0.8).contains(&h));
        }
    }

    #[test]
    fn uncalibrated_and_degenerate_ranges_are_refused() {
        let mut p = friction();
        p.source = String::new();
        assert!(matches!(
            Randomization::new(vec![p]),
            Err(RandomizationError::Uncalibrated(_))
        ));
        let mut p = friction();
        p.held_out = Some((0.4, 1.2));
        assert!(matches!(
            Randomization::new(vec![p]),
            Err(RandomizationError::BadHeldOut(_))
        ));
        let mut p = friction();
        p.hi = 0.1;
        assert!(matches!(
            Randomization::new(vec![p]),
            Err(RandomizationError::BadRange(_))
        ));
    }

    #[test]
    fn sensitivity_ranks_the_parameter_that_matters_first() {
        let r = Randomization::new(vec![
            friction(),
            Parameter {
                name: "sensor_bias".into(),
                lo: -0.01,
                hi: 0.01,
                held_out: None,
                source: "imu datasheet".into(),
            },
        ])
        .unwrap();
        let ranked = r.sensitivity(|p| 10.0 * p[0].1 + p[1].1);
        assert_eq!(ranked[0].0, "foot_friction");
        assert!((ranked[0].1 - 8.0).abs() < 1e-12);
    }
}

//! Evaluating seed results against an acceptance profile, and the
//! regression gate between two evaluations, per SPECS.md Section 14.
//!
//! A metric **passes** only when its whole 95 % t interval over seeds lies
//! on the passing side of the threshold, **fails** when the whole interval
//! lies on the failing side, and is otherwise **inconclusive**: "use
//! multiple seeds and confidence intervals for stochastic outcomes" means a
//! mean that happens to clear the line is not a pass. Seeds the profile
//! lists but the results lack are recorded, as are results for seeds the
//! profile does not list (which are excluded: the seed list is fixed before
//! evaluation). Evaluating marks the profile evaluated, so a later
//! threshold change has to go through [`AcceptanceProfile::revise`].
//!
//! The regression gate compares two evaluations of the **same** profile
//! (same content ID). Against a different profile it refuses with
//! [`GateError::ProfileMismatch`] naming both IDs: "changing a threshold
//! after observing a result produces a new profile version and invalidates
//! the comparison, which MUST be stated rather than quietly re-baselined."

use core::fmt;
use robot_artifacts::hash::ContentId;

use crate::acceptance::{AcceptanceProfile, Direction, MetricThreshold};
use crate::stats::{summarize, welch, SeedStatistics};

/// One seed's measured metrics.
#[derive(Clone, PartialEq, Debug)]
pub struct SeedResult {
    /// The seed.
    pub seed: u64,
    /// `(metric, value)` pairs.
    pub metrics: Vec<(String, f64)>,
}

/// One metric's verdict.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Outcome {
    /// The interval clears the threshold.
    Pass,
    /// The interval is on the wrong side of it.
    Fail,
    /// The interval straddles it, or fewer than two seeds reported.
    Inconclusive,
}

/// One metric's evaluation.
#[derive(Clone, PartialEq, Debug)]
pub struct MetricOutcome {
    /// The threshold it was held to.
    pub threshold: MetricThreshold,
    /// Its statistics over seeds, if at least two reported.
    pub stats: Option<SeedStatistics>,
    /// The verdict.
    pub outcome: Outcome,
    /// Listed seeds that reported no finite value for *this* metric. A
    /// non-empty list makes the verdict `Inconclusive`: a metric must not
    /// pass on a quiet subset of its declared seeds.
    pub missing_seeds: Vec<u64>,
}

/// A run's evaluation against one profile version.
#[derive(Clone, PartialEq, Debug)]
pub struct EvaluationRecord {
    /// Profile name.
    pub profile: String,
    /// Profile version.
    pub version: u32,
    /// Profile content ID.
    pub profile_id: ContentId,
    /// Per-metric outcomes, in profile order.
    pub outcomes: Vec<MetricOutcome>,
    /// Seeds listed by the profile with no result.
    pub missing_seeds: Vec<u64>,
    /// Results for seeds the profile does not list (excluded).
    pub unlisted_seeds: Vec<u64>,
    /// Seeds that reported more than once; later reports are excluded, so
    /// a repeated seed cannot shrink the confidence interval.
    pub duplicate_seeds: Vec<u64>,
}

impl EvaluationRecord {
    /// Every metric passed and every listed seed reported.
    #[must_use]
    pub fn accepted(&self) -> bool {
        self.missing_seeds.is_empty()
            && self.duplicate_seeds.is_empty()
            && self.outcomes.iter().all(|o| o.outcome == Outcome::Pass)
    }
}

/// Evaluates `results` against `profile` and marks it evaluated.
pub fn evaluate(profile: &mut AcceptanceProfile, results: &[SeedResult]) -> EvaluationRecord {
    let mut duplicate_seeds: Vec<u64> = Vec::new();
    let mut seen: Vec<u64> = Vec::new();
    let listed: Vec<&SeedResult> = results
        .iter()
        .filter(|r| profile.seed_list.contains(&r.seed))
        .filter(|r| {
            if seen.contains(&r.seed) {
                duplicate_seeds.push(r.seed);
                false
            } else {
                seen.push(r.seed);
                true
            }
        })
        .collect();
    let missing_seeds = profile
        .seed_list
        .iter()
        .filter(|s| !results.iter().any(|r| r.seed == **s))
        .copied()
        .collect();
    let unlisted_seeds = results
        .iter()
        .filter(|r| !profile.seed_list.contains(&r.seed))
        .map(|r| r.seed)
        .collect();
    let outcomes = profile
        .thresholds
        .iter()
        .map(|t| {
            let mut metric_missing: Vec<u64> = Vec::new();
            let values: Vec<f64> = listed
                .iter()
                .filter_map(|r| {
                    let v = r
                        .metrics
                        .iter()
                        .find(|(m, _)| *m == t.metric)
                        .map(|(_, v)| *v);
                    match v {
                        Some(v) if v.is_finite() => Some(v),
                        // Absent or non-finite: recorded, never silently
                        // dropped from the metric's evidence base.
                        _ => {
                            metric_missing.push(r.seed);
                            None
                        }
                    }
                })
                .collect();
            let stats = summarize(&values).ok();
            let complete = metric_missing.is_empty();
            let outcome = stats.map_or(Outcome::Inconclusive, |s| {
                let (lo, hi) = (
                    s.mean - s.t_confidence_95_halfwidth,
                    s.mean + s.t_confidence_95_halfwidth,
                );
                match t.direction {
                    Direction::AtMost if hi <= t.threshold => Outcome::Pass,
                    Direction::AtMost if lo > t.threshold => Outcome::Fail,
                    Direction::AtLeast if lo >= t.threshold => Outcome::Pass,
                    Direction::AtLeast if hi < t.threshold => Outcome::Fail,
                    _ => Outcome::Inconclusive,
                }
            });
            let outcome = if complete {
                outcome
            } else {
                Outcome::Inconclusive
            };
            MetricOutcome {
                threshold: t.clone(),
                stats,
                outcome,
                missing_seeds: metric_missing,
            }
        })
        .collect();
    profile.mark_evaluated();
    EvaluationRecord {
        profile: profile.name.clone(),
        version: profile.version,
        profile_id: profile.content_id(),
        outcomes,
        missing_seeds,
        unlisted_seeds,
        duplicate_seeds,
    }
}

/// Why two evaluations cannot be compared.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum GateError {
    /// They were evaluated against different profiles.
    ProfileMismatch {
        /// The baseline's profile ID.
        baseline: ContentId,
        /// The candidate's profile ID.
        candidate: ContentId,
    },
}

impl fmt::Display for GateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProfileMismatch { baseline, candidate } => write!(
                f,
                "comparison invalidated: baseline evaluated against {baseline}, candidate against {candidate}"
            ),
        }
    }
}

impl core::error::Error for GateError {}

/// One metric compared between baseline and candidate.
#[derive(Clone, PartialEq, Debug)]
pub struct MetricComparison {
    /// Metric name.
    pub metric: String,
    /// Welch's p-value for a difference.
    pub p: f64,
    /// Candidate mean minus baseline mean.
    pub difference: f64,
    /// Significantly worse.
    pub regressed: bool,
    /// Significantly better.
    pub improved: bool,
}

/// The gate's result.
#[derive(Clone, PartialEq, Debug)]
pub struct GateReport {
    /// Per metric.
    pub comparisons: Vec<MetricComparison>,
    /// Metrics without two seeds on both sides.
    pub not_compared: Vec<String>,
}

impl GateReport {
    /// No metric regressed at the chosen significance.
    #[must_use]
    pub fn passed(&self) -> bool {
        !self.comparisons.iter().any(|c| c.regressed)
    }
}

/// Compares `candidate` with `baseline` at significance `alpha`.
pub fn regression_gate(
    baseline: &EvaluationRecord,
    candidate: &EvaluationRecord,
    alpha: f64,
) -> Result<GateReport, GateError> {
    if baseline.profile_id != candidate.profile_id {
        return Err(GateError::ProfileMismatch {
            baseline: baseline.profile_id.clone(),
            candidate: candidate.profile_id.clone(),
        });
    }
    let mut report = GateReport {
        comparisons: Vec::new(),
        not_compared: Vec::new(),
    };
    for (b, c) in baseline.outcomes.iter().zip(&candidate.outcomes) {
        let (Some(bs), Some(cs)) = (&b.stats, &c.stats) else {
            report.not_compared.push(b.threshold.metric.clone());
            continue;
        };
        let w = welch(bs, cs);
        let worse = match b.threshold.direction {
            Direction::AtMost => w.difference > 0.0,
            Direction::AtLeast => w.difference < 0.0,
        };
        let significant = w.p_two_sided < alpha;
        report.comparisons.push(MetricComparison {
            metric: b.threshold.metric.clone(),
            p: w.p_two_sided,
            difference: w.difference,
            regressed: significant && worse,
            improved: significant && !worse,
        });
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_types::time::Duration;

    fn profile(threshold: f64) -> AcceptanceProfile {
        AcceptanceProfile::new(
            "p",
            "g1@1.0",
            "fw",
            Duration::from_millis(20),
            Duration::from_millis(10),
            "standing",
            vec![MetricThreshold {
                metric: "fall_rate".into(),
                threshold,
                direction: Direction::AtMost,
                justification: "test".into(),
            }],
            vec![1, 2, 3, 4, 5],
            1,
            Duration::from_secs(60),
            "2026-11-13",
            "test",
        )
    }

    fn results(values: &[f64]) -> Vec<SeedResult> {
        values
            .iter()
            .enumerate()
            .map(|(i, v)| SeedResult {
                seed: i as u64 + 1,
                metrics: vec![("fall_rate".into(), *v)],
            })
            .collect()
    }

    #[test]
    fn pass_fail_and_inconclusive_follow_the_interval_not_the_mean() {
        let low = [0.01, 0.02, 0.015, 0.012, 0.018];
        assert_eq!(
            evaluate(&mut profile(0.05), &results(&low)).outcomes[0].outcome,
            Outcome::Pass
        );
        let high = [0.2, 0.22, 0.19, 0.21, 0.2];
        assert_eq!(
            evaluate(&mut profile(0.05), &results(&high)).outcomes[0].outcome,
            Outcome::Fail
        );
        // Mean 0.048 under 0.05, but noisy: not a pass.
        let noisy = [0.0, 0.1, 0.02, 0.09, 0.03];
        let rec = evaluate(&mut profile(0.05), &results(&noisy));
        assert_eq!(rec.outcomes[0].outcome, Outcome::Inconclusive);
        assert!(!rec.accepted());
    }

    #[test]
    fn a_metric_cannot_pass_from_a_quiet_subset_of_its_seeds() {
        // All five seeds report, but seed 3 reports another metric only and
        // seed 5 reports NaN: the metric's evidence base is incomplete and
        // the verdict must be Inconclusive, with the gap named.
        let mut r = results(&[0.01, 0.01, 0.01, 0.01, f64::NAN]);
        r[2].metrics = vec![("other".into(), 1.0)];
        let rec = evaluate(&mut profile(0.05), &r);
        assert_eq!(rec.outcomes[0].outcome, Outcome::Inconclusive);
        assert_eq!(rec.outcomes[0].missing_seeds, vec![3, 5]);
        assert!(!rec.accepted());
    }

    #[test]
    fn a_repeated_seed_is_excluded_and_blocks_acceptance() {
        // A duplicated seed must not shrink the confidence interval.
        let mut r = results(&[0.01, 0.012, 0.011, 0.013, 0.012]);
        r.push(SeedResult {
            seed: 1,
            metrics: vec![("fall_rate".into(), 0.01)],
        });
        let rec = evaluate(&mut profile(0.05), &r);
        assert_eq!(rec.duplicate_seeds, vec![1]);
        assert_eq!(rec.outcomes[0].outcome, Outcome::Pass);
        assert!(
            !rec.accepted(),
            "a double-counted seed is not acceptance evidence"
        );
    }

    #[test]
    fn missing_and_unlisted_seeds_are_recorded_and_evaluation_marks_the_profile() {
        let mut p = profile(0.05);
        let mut r = results(&[0.01, 0.01, 0.01]);
        r.push(SeedResult {
            seed: 99,
            metrics: vec![("fall_rate".into(), 5.0)],
        });
        let rec = evaluate(&mut p, &r);
        assert_eq!(rec.missing_seeds, vec![4, 5]);
        assert_eq!(rec.unlisted_seeds, vec![99]);
        assert!(p.is_evaluated());
        assert!(!rec.accepted());
    }

    #[test]
    fn the_gate_flags_a_significant_regression_and_refuses_a_changed_profile() {
        let mut p = profile(0.5);
        let base = evaluate(&mut p, &results(&[0.01, 0.02, 0.015, 0.012, 0.018]));
        let worse = evaluate(&mut p, &results(&[0.2, 0.22, 0.19, 0.21, 0.2]));
        let same = evaluate(&mut p, &results(&[0.012, 0.019, 0.014, 0.013, 0.017]));
        assert!(!regression_gate(&base, &worse, 0.05).unwrap().passed());
        let r = regression_gate(&base, &same, 0.05).unwrap();
        assert!(r.passed());
        let mut q = p.revise(
            p.thresholds
                .clone()
                .into_iter()
                .map(|mut t| {
                    t.threshold = 0.6;
                    t
                })
                .collect(),
            "d",
            "a",
        );
        let other = evaluate(&mut q, &results(&[0.01, 0.02, 0.015, 0.012, 0.018]));
        assert!(matches!(
            regression_gate(&base, &other, 0.05),
            Err(GateError::ProfileMismatch { .. })
        ));
    }
}

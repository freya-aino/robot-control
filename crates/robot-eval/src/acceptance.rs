//! Versioned, immutable-once-evaluated acceptance profiles.
//!
//! SPECS.md Section 14: "An acceptance profile is itself an artifact and
//! MUST record: the robot profile and firmware version it applies to, the
//! control period and timing budget, the physical operating envelope,
//! per-metric thresholds with their justification, the seed list and run
//! count, the measurement duration, and the date and author... Profiles
//! are versioned and immutable once a run has been evaluated against
//! them. Changing a threshold after observing a result produces a new
//! profile version and invalidates the comparison, which MUST be stated
//! rather than quietly re-baselined."
//!
//! [`AcceptanceProfile`] carries every one of those fields, and
//! [`AcceptanceProfile::revise`] is the only way to change a threshold —
//! it always returns a new, differently-versioned profile and never
//! mutates the one a run was evaluated against.

use robot_types::time::Duration;

/// One metric's threshold, with the justification SPECS.md Section 14
/// requires rather than a bare number.
#[derive(Clone, Debug)]
pub struct MetricThreshold {
    /// Metric name.
    pub metric: String,
    /// The threshold value.
    pub threshold: f64,
    /// Why this value, in enough detail to defend it later (e.g. "95th
    /// percentile of three manual demonstrations" or "derived from
    /// actuator thermal time constant").
    pub justification: String,
}

/// A versioned, immutable acceptance profile.
#[derive(Clone, Debug)]
pub struct AcceptanceProfile {
    /// Profile identity.
    pub name: String,
    /// Monotonically increasing version; bumped on every revision.
    pub version: u32,
    /// Robot profile identity and firmware version this applies to.
    pub robot_profile: String,
    /// Control period this profile was evaluated under.
    pub control_period: Duration,
    /// Per-tick timing budget.
    pub timing_budget: Duration,
    /// Physical operating envelope, described in whatever units the task
    /// needs (free-form, since the envelope's shape is task-specific).
    pub physical_envelope: String,
    /// Per-metric thresholds.
    pub thresholds: Vec<MetricThreshold>,
    /// Seeds a run must use to be evaluated against this profile.
    pub seed_list: Vec<u64>,
    /// Required measurement duration per seed.
    pub measurement_duration: Duration,
    /// ISO-8601 date this version was authored.
    pub date: String,
    /// Author.
    pub author: String,
    /// Whether a run has already been evaluated against this exact
    /// version. Once `true`, `revise` is the only way forward.
    evaluated: bool,
}

impl AcceptanceProfile {
    /// Builds version 1 of a profile, not yet evaluated against.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        robot_profile: impl Into<String>,
        control_period: Duration,
        timing_budget: Duration,
        physical_envelope: impl Into<String>,
        thresholds: Vec<MetricThreshold>,
        seed_list: Vec<u64>,
        measurement_duration: Duration,
        date: impl Into<String>,
        author: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            version: 1,
            robot_profile: robot_profile.into(),
            control_period,
            timing_budget,
            physical_envelope: physical_envelope.into(),
            thresholds,
            seed_list,
            measurement_duration,
            date: date.into(),
            author: author.into(),
            evaluated: false,
        }
    }

    /// Marks this profile as having had a run evaluated against it. After
    /// this, the profile's own fields must not be mutated in place — only
    /// [`AcceptanceProfile::revise`] produces a profile with different
    /// thresholds, and it is a new version.
    pub fn mark_evaluated(&mut self) {
        self.evaluated = true;
    }

    /// Whether a run has been evaluated against this exact version.
    #[must_use]
    pub const fn is_evaluated(&self) -> bool {
        self.evaluated
    }

    /// Produces a new, incremented-version profile with `thresholds`
    /// replaced, explicitly stating that any comparison against the prior
    /// version is invalidated.
    ///
    /// This is the only sanctioned way to change a threshold once a run has
    /// been evaluated: it does not edit `self`, it returns a distinct
    /// profile, and the caller's own record of *why* it changed belongs in
    /// the new profile's thresholds' `justification` fields.
    #[must_use]
    pub fn revise(
        &self,
        new_thresholds: Vec<MetricThreshold>,
        date: impl Into<String>,
        author: impl Into<String>,
    ) -> Self {
        Self {
            version: self.version + 1,
            thresholds: new_thresholds,
            date: date.into(),
            author: author.into(),
            evaluated: false,
            ..self.clone()
        }
    }

    /// Looks up a declared threshold by metric name.
    #[must_use]
    pub fn threshold_for(&self, metric: &str) -> Option<&MetricThreshold> {
        self.thresholds.iter().find(|t| t.metric == metric)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> AcceptanceProfile {
        AcceptanceProfile::new(
            "fixture-arm-balance",
            "fixture-arm/1.0",
            Duration::from_millis(10),
            Duration::from_millis(8),
            "upright +/- 0.3 rad, torque <= 10 N*m",
            vec![MetricThreshold {
                metric: "fall_rate".to_owned(),
                threshold: 0.05,
                justification: "no more than 1 fall in 20 held-out episodes".to_owned(),
            }],
            vec![1, 2, 3],
            Duration::from_secs(30),
            "2026-10-26",
            "freya",
        )
    }

    #[test]
    fn revision_increments_version_and_resets_evaluated_status() {
        let mut p1 = profile();
        p1.mark_evaluated();
        assert!(p1.is_evaluated());
        assert_eq!(p1.version, 1);

        let p2 = p1.revise(
            vec![MetricThreshold {
                metric: "fall_rate".to_owned(),
                threshold: 0.1,
                justification: "relaxed after reviewing held-out episode variance".to_owned(),
            }],
            "2026-10-27",
            "freya",
        );
        assert_eq!(p2.version, 2);
        assert!(!p2.is_evaluated());
        // p1 itself is untouched.
        assert!(p1.is_evaluated());
        assert_eq!(p1.threshold_for("fall_rate").unwrap().threshold, 0.05);
        assert_eq!(p2.threshold_for("fall_rate").unwrap().threshold, 0.1);
    }

    #[test]
    fn threshold_lookup_is_by_name() {
        let p = profile();
        assert!(p.threshold_for("fall_rate").is_some());
        assert!(p.threshold_for("nonexistent").is_none());
    }
}

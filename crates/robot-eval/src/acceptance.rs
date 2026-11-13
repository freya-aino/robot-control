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

use robot_artifacts::hash::ContentId;
use robot_artifacts::toml::{self, Table, TomlError, Value};
use robot_types::time::Duration;

/// Which side of a threshold passes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Direction {
    /// The metric must not exceed the threshold (a fall rate).
    AtMost,
    /// The metric must reach at least the threshold (a success rate).
    AtLeast,
}

impl Direction {
    const fn name(self) -> &'static str {
        match self {
            Self::AtMost => "at_most",
            Self::AtLeast => "at_least",
        }
    }
}

/// One metric's threshold, with the justification SPECS.md Section 14
/// requires rather than a bare number.
#[derive(Clone, PartialEq, Debug)]
pub struct MetricThreshold {
    /// Metric name.
    pub metric: String,
    /// The threshold value.
    pub threshold: f64,
    /// Which side passes.
    pub direction: Direction,
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
    /// Firmware version the profile applies to.
    pub firmware_version: String,
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
    /// Runs per seed.
    pub run_count: u32,
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
        firmware_version: impl Into<String>,
        control_period: Duration,
        timing_budget: Duration,
        physical_envelope: impl Into<String>,
        thresholds: Vec<MetricThreshold>,
        seed_list: Vec<u64>,
        run_count: u32,
        measurement_duration: Duration,
        date: impl Into<String>,
        author: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            version: 1,
            robot_profile: robot_profile.into(),
            firmware_version: firmware_version.into(),
            control_period,
            timing_budget,
            physical_envelope: physical_envelope.into(),
            thresholds,
            seed_list,
            run_count,
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

/// Why a profile file could not be read.
#[derive(Clone, PartialEq, Debug)]
pub enum ProfileFileError {
    /// The TOML was rejected.
    Toml(TomlError),
    /// A field had the wrong type or value.
    Field(String),
}

impl core::fmt::Display for ProfileFileError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Toml(e) => write!(f, "{e}"),
            Self::Field(e) => write!(f, "acceptance profile field: {e}"),
        }
    }
}

impl core::error::Error for ProfileFileError {}

impl From<TomlError> for ProfileFileError {
    fn from(e: TomlError) -> Self {
        Self::Toml(e)
    }
}

impl AcceptanceProfile {
    /// The profile as `acceptance-profile/1` TOML. The `evaluated` flag is
    /// not part of the artifact: it is a property of a run's use of it.
    #[must_use]
    pub fn to_toml(&self) -> String {
        let mut t = Table::new();
        t.insert("schema", Value::String("acceptance-profile/1".into()));
        t.insert("name", Value::String(self.name.clone()));
        t.insert("version", Value::Integer(i64::from(self.version)));
        t.insert("robot_profile", Value::String(self.robot_profile.clone()));
        t.insert(
            "firmware_version",
            Value::String(self.firmware_version.clone()),
        );
        t.insert(
            "control_period_ns",
            Value::Integer(self.control_period.nanos()),
        );
        t.insert(
            "timing_budget_ns",
            Value::Integer(self.timing_budget.nanos()),
        );
        t.insert(
            "physical_envelope",
            Value::String(self.physical_envelope.clone()),
        );
        t.insert(
            "seed_list",
            Value::Array(
                self.seed_list
                    .iter()
                    .map(|s| Value::Integer(*s as i64))
                    .collect(),
            ),
        );
        t.insert("run_count", Value::Integer(i64::from(self.run_count)));
        t.insert(
            "measurement_duration_ns",
            Value::Integer(self.measurement_duration.nanos()),
        );
        t.insert("date", Value::String(self.date.clone()));
        t.insert("author", Value::String(self.author.clone()));
        t.insert(
            "threshold",
            Value::Array(
                self.thresholds
                    .iter()
                    .map(|m| {
                        let mut mt = Table::new();
                        mt.insert("metric", Value::String(m.metric.clone()));
                        mt.insert("threshold", Value::Float(m.threshold));
                        mt.insert("direction", Value::String(m.direction.name().into()));
                        mt.insert("justification", Value::String(m.justification.clone()));
                        Value::Table(mt)
                    })
                    .collect(),
            ),
        );
        toml::to_string(&t)
    }

    /// The content ID of [`Self::to_toml`]: the identity a run cites.
    #[must_use]
    pub fn content_id(&self) -> ContentId {
        ContentId::of(self.to_toml().as_bytes())
    }

    /// Reads an `acceptance-profile/1` file. The result is unevaluated.
    pub fn from_toml(text: &str) -> Result<Self, ProfileFileError> {
        let t = toml::parse(text)?;
        let field = |k: &str| ProfileFileError::Field(k.to_owned());
        if t.require_str("schema")? != "acceptance-profile/1" {
            return Err(field("schema"));
        }
        let int = |k: &str| t.require_integer(k).map_err(ProfileFileError::from);
        let thresholds = t
            .require_array("threshold")?
            .iter()
            .map(|v| {
                let Value::Table(m) = v else {
                    return Err(field("threshold"));
                };
                Ok(MetricThreshold {
                    metric: m.require_str("metric")?.to_owned(),
                    threshold: m.require_float("threshold")?,
                    direction: match m.require_str("direction")? {
                        "at_most" => Direction::AtMost,
                        "at_least" => Direction::AtLeast,
                        _ => return Err(field("direction")),
                    },
                    justification: m.require_str("justification")?.to_owned(),
                })
            })
            .collect::<Result<Vec<_>, ProfileFileError>>()?;
        let seed_list = t
            .require_array("seed_list")?
            .iter()
            .map(|v| match v {
                Value::Integer(i) if *i >= 0 => Ok(*i as u64),
                _ => Err(field("seed_list")),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            name: t.require_str("name")?.to_owned(),
            version: u32::try_from(int("version")?).map_err(|_| field("version"))?,
            robot_profile: t.require_str("robot_profile")?.to_owned(),
            firmware_version: t.require_str("firmware_version")?.to_owned(),
            control_period: Duration::from_nanos(int("control_period_ns")?),
            timing_budget: Duration::from_nanos(int("timing_budget_ns")?),
            physical_envelope: t.require_str("physical_envelope")?.to_owned(),
            thresholds,
            seed_list,
            run_count: u32::try_from(int("run_count")?).map_err(|_| field("run_count"))?,
            measurement_duration: Duration::from_nanos(int("measurement_duration_ns")?),
            date: t.require_str("date")?.to_owned(),
            author: t.require_str("author")?.to_owned(),
            evaluated: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> AcceptanceProfile {
        AcceptanceProfile::new(
            "fixture-arm-balance",
            "fixture-arm/1.0",
            "fw-0.0-fixture",
            Duration::from_millis(10),
            Duration::from_millis(8),
            "upright +/- 0.3 rad, torque <= 10 N*m",
            vec![MetricThreshold {
                metric: "fall_rate".to_owned(),
                threshold: 0.05,
                direction: Direction::AtMost,
                justification: "no more than 1 fall in 20 held-out episodes".to_owned(),
            }],
            vec![1, 2, 3],
            4,
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
                direction: Direction::AtMost,
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
    fn the_artifact_round_trips_and_its_id_changes_with_any_threshold() {
        let p = profile();
        let back = AcceptanceProfile::from_toml(&p.to_toml()).unwrap();
        assert_eq!(back.to_toml(), p.to_toml());
        assert_eq!(back.firmware_version, "fw-0.0-fixture");
        let q = p.revise(
            vec![MetricThreshold {
                metric: "fall_rate".to_owned(),
                threshold: 0.051,
                direction: Direction::AtMost,
                justification: "x".to_owned(),
            }],
            "2026-10-27",
            "freya",
        );
        assert_ne!(p.content_id(), q.content_id());
    }

    #[test]
    fn threshold_lookup_is_by_name() {
        let p = profile();
        assert!(p.threshold_for("fall_rate").is_some());
        assert!(p.threshold_for("nonexistent").is_none());
    }
}

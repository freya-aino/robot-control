//! The adaptation report of SPECS.md Section 11.3 obligations 5 and 6.
//!
//! Obligation 5: every method is evaluated against the frozen
//! pre-adaptation baseline on the held-out session, "reporting both
//! improvement and regression", with rollback available. Obligation 6:
//! sample-efficiency claims are "stated in the units that cost something:
//! real minutes, real episodes, physical interventions and wall-clock
//! downtime", with the simulation consumed "in the same report". An
//! [`AdaptationReport`] cannot be built without all of them, and it
//! renders to the TOML subset of `robot-artifacts` for a run directory.

use robot_artifacts::toml::{self, Table, Value};

use crate::ladder::Method;

/// What adaptation cost.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SampleCost {
    /// Real-robot minutes of data.
    pub real_minutes: f64,
    /// Real episodes.
    pub real_episodes: u64,
    /// Physical interventions (catches, resets by hand).
    pub interventions: u64,
    /// Wall-clock robot downtime, minutes.
    pub downtime_minutes: f64,
    /// Simulation steps consumed.
    pub sim_steps: u64,
}

/// One held-out metric before and after, lower is better.
#[derive(Clone, PartialEq, Debug)]
pub struct HeldOutMetric {
    /// Metric name.
    pub name: String,
    /// The frozen baseline's value.
    pub baseline: f64,
    /// The adapted value.
    pub adapted: f64,
}

impl HeldOutMetric {
    /// Relative improvement (positive) or regression (negative).
    #[must_use]
    pub fn relative_change(&self) -> f64 {
        if self.baseline == 0.0 {
            0.0
        } else {
            (self.baseline - self.adapted) / self.baseline.abs()
        }
    }
}

/// The full report.
#[derive(Clone, PartialEq, Debug)]
pub struct AdaptationReport {
    /// The method.
    pub method: Method,
    /// Robot profile.
    pub robot: String,
    /// Held-out session identifier.
    pub held_out_session: String,
    /// Content ID of the baseline artifact rollback restores.
    pub rollback_artifact: String,
    /// Metrics.
    pub metrics: Vec<HeldOutMetric>,
    /// Cost.
    pub cost: SampleCost,
}

impl AdaptationReport {
    /// Metrics that got worse.
    #[must_use]
    pub fn regressions(&self) -> Vec<&HeldOutMetric> {
        self.metrics
            .iter()
            .filter(|m| m.adapted > m.baseline)
            .collect()
    }

    /// The report as TOML.
    #[must_use]
    pub fn to_toml(&self) -> String {
        let mut t = Table::new();
        t.insert("schema", Value::String("adaptation-report/1".into()));
        t.insert("method", Value::String(self.method.name().into()));
        t.insert("robot", Value::String(self.robot.clone()));
        t.insert(
            "held_out_session",
            Value::String(self.held_out_session.clone()),
        );
        t.insert(
            "rollback_artifact",
            Value::String(self.rollback_artifact.clone()),
        );
        let mut c = Table::new();
        c.insert("real_minutes", Value::Float(self.cost.real_minutes));
        c.insert(
            "real_episodes",
            Value::Integer(self.cost.real_episodes as i64),
        );
        c.insert(
            "interventions",
            Value::Integer(self.cost.interventions as i64),
        );
        c.insert("downtime_minutes", Value::Float(self.cost.downtime_minutes));
        c.insert("sim_steps", Value::Integer(self.cost.sim_steps as i64));
        t.insert("cost", Value::Table(c));
        t.insert(
            "metric",
            Value::Array(
                self.metrics
                    .iter()
                    .map(|m| {
                        let mut mt = Table::new();
                        mt.insert("name", Value::String(m.name.clone()));
                        mt.insert("baseline", Value::Float(m.baseline));
                        mt.insert("adapted", Value::Float(m.adapted));
                        mt.insert("regressed", Value::Boolean(m.adapted > m.baseline));
                        Value::Table(mt)
                    })
                    .collect(),
            ),
        );
        toml::to_string(&t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regressions_are_reported_and_the_toml_round_trips() {
        let r = AdaptationReport {
            method: Method::SystemIdentification,
            robot: "g1@1.0".into(),
            held_out_session: "s-17".into(),
            rollback_artifact: "sha256:abc".into(),
            metrics: vec![
                HeldOutMetric {
                    name: "tracking_rms".into(),
                    baseline: 0.1,
                    adapted: 0.05,
                },
                HeldOutMetric {
                    name: "falls_per_hour".into(),
                    baseline: 1.0,
                    adapted: 2.0,
                },
            ],
            cost: SampleCost {
                real_minutes: 12.5,
                real_episodes: 30,
                interventions: 2,
                downtime_minutes: 40.0,
                sim_steps: 5_000_000,
            },
        };
        assert_eq!(r.regressions().len(), 1);
        assert!((r.metrics[0].relative_change() - 0.5).abs() < 1e-12);
        let parsed = toml::parse(&r.to_toml()).unwrap();
        assert_eq!(
            parsed.require_str("method").unwrap(),
            "system-identification"
        );
        assert_eq!(
            parsed
                .require_table("cost")
                .unwrap()
                .require_integer("interventions")
                .unwrap(),
            2
        );
    }
}

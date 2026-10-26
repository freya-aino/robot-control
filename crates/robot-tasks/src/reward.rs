//! A name-keyed, build-time-constructed reward-term registry.
//!
//! SPECS.md Section 9.3 names the reward-term registry as a "Name-keyed
//! registry built before an offline run; no control-path lookup," and
//! Section 9.2 requires: "Log reward terms independently to expose reward
//! exploitation. Persistent torque violations cannot be hidden by a good
//! aggregate return." [`RewardRegistry::evaluate`] always returns the typed
//! [`robot_types::transition::RewardTerms`], never a bare aggregate float,
//! for exactly that reason.

use robot_types::ids::Name;
use robot_types::transition::{RewardTerm, RewardTerms};

/// What a reward term is computed from.
#[derive(Clone, Copy, Debug)]
pub struct TaskContext<'a> {
    /// Observation before the action.
    pub observation: &'a [f64],
    /// The action taken.
    pub action: &'a [f64],
    /// Observation after the action.
    pub next_observation: &'a [f64],
}

/// One named, weighted term.
#[derive(Clone, Copy)]
pub struct RewardTermSpec {
    /// Stable, versioned name (e.g. `"upright-v1"`).
    pub name: &'static str,
    /// Weight applied in the aggregate.
    pub weight: f64,
    /// The unweighted value, as a function of the task context.
    pub compute: fn(&TaskContext<'_>) -> f64,
    /// Whether this is a style/motion-prior term rather than a task term.
    /// SPECS.md Section 9.4.2 item 5: a style term is logged separately and
    /// is never evidence of feasibility, stability or safety.
    pub is_style_term: bool,
}

/// A registry of reward terms, constructed once and never mutated during a
/// run — the build-time composition Section 9.3 requires, using plain
/// function pointers so no dynamic dispatch or registration machinery is
/// needed for a fixed, linked-in term set.
#[derive(Clone)]
pub struct RewardRegistry {
    terms: Vec<RewardTermSpec>,
}

/// Why a registry could not be built.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DuplicateTermName(pub String);

impl core::fmt::Display for DuplicateTermName {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "duplicate reward term name {:?}", self.0)
    }
}

impl core::error::Error for DuplicateTermName {}

impl RewardRegistry {
    /// Builds a registry, rejecting a duplicate term name.
    pub fn new(terms: Vec<RewardTermSpec>) -> Result<Self, DuplicateTermName> {
        for (i, t) in terms.iter().enumerate() {
            if terms[..i].iter().any(|o| o.name == t.name) {
                return Err(DuplicateTermName(t.name.to_owned()));
            }
        }
        Ok(Self { terms })
    }

    /// Evaluates every term against `ctx`, returning the typed collection
    /// with each term's value and weight preserved individually.
    pub fn evaluate(&self, ctx: &TaskContext<'_>) -> RewardTerms {
        let mut out = RewardTerms::new();
        for term in &self.terms {
            let value = (term.compute)(ctx);
            out.push(RewardTerm {
                name: term.name.to_owned(),
                value,
                weight: term.weight,
                is_style_term: term.is_style_term,
            })
            .expect("names were checked unique at registry construction");
        }
        out
    }

    /// Number of registered terms.
    #[must_use]
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// True when the registry has no terms.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }
}

/// Validates a term name against the project's name rules, for a registry
/// builder that wants to fail fast on a malformed identifier.
pub fn validate_term_name(name: &str) -> Result<(), robot_types::ids::NameError> {
    Name::new(name).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn term(
        name: &'static str,
        weight: f64,
        compute: fn(&TaskContext<'_>) -> f64,
    ) -> RewardTermSpec {
        RewardTermSpec {
            name,
            weight,
            compute,
            is_style_term: false,
        }
    }

    #[test]
    fn evaluate_reports_every_term_independently() {
        let registry = RewardRegistry::new(vec![
            term("alive", 1.0, |_| 1.0),
            term("effort", -0.1, |ctx| ctx.action[0].powi(2)),
        ])
        .unwrap();
        let ctx = TaskContext {
            observation: &[0.0],
            action: &[2.0],
            next_observation: &[0.0],
        };
        let terms = registry.evaluate(&ctx);
        assert_eq!(terms.terms().len(), 2);
        assert!((terms.get("alive").unwrap().value - 1.0).abs() < 1e-12);
        assert!((terms.get("effort").unwrap().value - 4.0).abs() < 1e-12);
        // total = 1*1.0 + 4*(-0.1) = 0.6
        assert!((terms.total() - 0.6).abs() < 1e-12);
    }

    #[test]
    fn duplicate_names_are_rejected_at_construction() {
        let built = RewardRegistry::new(vec![term("a", 1.0, |_| 0.0), term("a", 1.0, |_| 0.0)]);
        assert!(matches!(built, Err(DuplicateTermName(name)) if name == "a"));
    }

    #[test]
    fn style_terms_are_excluded_from_the_task_total() {
        let mut style = term("style", 10.0, |_| 1.0);
        style.is_style_term = true;
        let registry = RewardRegistry::new(vec![term("task", 1.0, |_| 1.0), style]).unwrap();
        let ctx = TaskContext {
            observation: &[],
            action: &[],
            next_observation: &[],
        };
        let terms = registry.evaluate(&ctx);
        assert!((terms.task_total() - 1.0).abs() < 1e-12);
        assert!((terms.total() - 11.0).abs() < 1e-12);
    }

    #[test]
    fn an_empty_registry_evaluates_to_zero() {
        let registry = RewardRegistry::new(vec![]).unwrap();
        assert!(registry.is_empty());
        let ctx = TaskContext {
            observation: &[],
            action: &[],
            next_observation: &[],
        };
        assert_eq!(registry.evaluate(&ctx).total(), 0.0);
    }
}

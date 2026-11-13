//! When expression may act, per SPECS.md Section 12.5: expressive actions
//! "use the same validated skills and transitions; they cannot override
//! balance recovery or supervisor decisions", and expression MUST NOT
//! "extend any dwell time against a recovery request".
//!
//! The [`Arbiter`] is consulted at skill-selection rate with the current
//! situation and a proposed expressive skill. Its answer is one of:
//!
//! - [`Decision::Yield`] whenever a recovery is requested or the supervisor
//!   holds a veto, *including* while an expressive skill is running: the
//!   expressive skill ends now, it does not finish its dwell;
//! - [`Decision::Refuse`] when the proposal is unvalidated, not part of the
//!   style, or not reachable by the current skill's transition contract
//!   (its `outgoing_transitions`): "a celebration is a skill with a
//!   transition contract, not an exemption";
//! - [`Decision::Express`] otherwise.
//!
//! The arbiter can only say no to expression or pass a validated skill
//! through; it has no path to a task skill's selection or to any command.

use robot_skills::descriptor::{SkillDescriptor, SkillId};

use crate::style::StyleConfig;

/// The situation at one decision.
#[derive(Clone, Copy, Debug)]
pub struct Situation<'a> {
    /// The skill currently executing.
    pub current: &'a SkillDescriptor,
    /// Whether the current skill is expressive (not task-driven).
    pub current_is_expressive: bool,
    /// Whether a balance recovery is requested.
    pub recovery_requested: bool,
    /// Whether the supervisor currently vetoes non-essential motion.
    pub supervisor_veto: bool,
}

/// Why a proposal was refused.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Refusal {
    /// Not in the validated skill set.
    NotValidated,
    /// Not part of this style.
    NotInStyle,
    /// The current skill's transition contract does not reach it.
    NoTransition,
}

/// The arbiter's answer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Decision {
    /// End any expressive skill now and defer to recovery or the supervisor.
    Yield,
    /// Do not express this skill.
    Refuse(Refusal),
    /// Express this skill.
    Express(SkillId),
    /// Nothing proposed and nothing to yield.
    Continue,
}

/// Decides expression for one style over one validated skill set.
#[derive(Clone, Debug)]
pub struct Arbiter<'a> {
    style: &'a StyleConfig,
    validated: &'a [SkillDescriptor],
}

impl<'a> Arbiter<'a> {
    /// An arbiter over `validated` skills for `style`.
    #[must_use]
    pub const fn new(style: &'a StyleConfig, validated: &'a [SkillDescriptor]) -> Self {
        Self { style, validated }
    }

    /// Decides on `proposal` (or on nothing) in `situation`.
    #[must_use]
    pub fn decide(&self, situation: &Situation<'_>, proposal: Option<SkillId>) -> Decision {
        if situation.recovery_requested || situation.supervisor_veto {
            return Decision::Yield;
        }
        let Some(id) = proposal else {
            return Decision::Continue;
        };
        if !self.validated.iter().any(|s| s.id == id) {
            return Decision::Refuse(Refusal::NotValidated);
        }
        if !self
            .style
            .expressive_preferences
            .iter()
            .any(|(s, _)| *s == id)
        {
            return Decision::Refuse(Refusal::NotInStyle);
        }
        if !situation.current.outgoing_transitions.contains(&id) {
            return Decision::Refuse(Refusal::NoTransition);
        }
        Decision::Express(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_skills::descriptor::EmbeddingVersion;
    use robot_types::ids::Name;
    use robot_types::time::Duration;

    fn skill(id: u32, next: &[u32]) -> SkillDescriptor {
        SkillDescriptor {
            id: SkillId(id),
            name: Name::new(&format!("s{id}")).unwrap(),
            tags: Vec::new(),
            embedding: vec![0.0],
            embedding_version: EmbeddingVersion(1),
            embodiment_compatibility: vec!["g1".into()],
            initiation_guards: Vec::new(),
            termination_condition: Name::new("done").unwrap(),
            expected_duration: Duration::from_millis(500),
            assumes_ground_contact: true,
            supported_disturbances: Vec::new(),
            inference_memory_bytes: 0,
            outgoing_transitions: next.iter().map(|&n| SkillId(n)).collect(),
        }
    }

    fn style() -> StyleConfig {
        StyleConfig {
            name: "showboat".into(),
            version: 1,
            expressive_preferences: vec![(SkillId(10), 1.0), (SkillId(11), 1.0)],
            tactical_preferences: Vec::new(),
            parameters: Vec::new(),
        }
    }

    #[test]
    fn recovery_and_veto_end_expression_even_mid_skill() {
        let s = style();
        let skills = [skill(1, &[10]), skill(10, &[1])];
        let a = Arbiter::new(&s, &skills);
        let mid_celebration = Situation {
            current: &skills[1],
            current_is_expressive: true,
            recovery_requested: true,
            supervisor_veto: false,
        };
        assert_eq!(a.decide(&mid_celebration, None), Decision::Yield);
        let veto = Situation {
            recovery_requested: false,
            supervisor_veto: true,
            ..mid_celebration
        };
        assert_eq!(a.decide(&veto, Some(SkillId(10))), Decision::Yield);
    }

    #[test]
    fn only_validated_in_style_reachable_skills_are_expressed() {
        let s = style();
        let skills = [skill(1, &[10]), skill(10, &[1]), skill(11, &[1])];
        let a = Arbiter::new(&s, &skills);
        let guard = Situation {
            current: &skills[0],
            current_is_expressive: false,
            recovery_requested: false,
            supervisor_veto: false,
        };
        assert_eq!(
            a.decide(&guard, Some(SkillId(10))),
            Decision::Express(SkillId(10))
        );
        assert_eq!(
            a.decide(&guard, Some(SkillId(11))),
            Decision::Refuse(Refusal::NoTransition)
        );
        assert_eq!(
            a.decide(&guard, Some(SkillId(99))),
            Decision::Refuse(Refusal::NotValidated)
        );
        assert_eq!(
            a.decide(&guard, Some(SkillId(1))),
            Decision::Refuse(Refusal::NotInStyle)
        );
        assert_eq!(a.decide(&guard, None), Decision::Continue);
    }
}

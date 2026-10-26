//! A skill's typed, authoritative descriptor.
//!
//! SPECS.md Section 10.2: "Each skill has semantic tags, embedding/version,
//! embodiment compatibility, initiation set, termination condition,
//! expected duration, contact assumptions, supported disturbances,
//! inference memory requirement and outgoing transition contracts." Every
//! field below is one of those. Section 10.1's Markdown prose descriptor, if
//! one exists, is a separate, non-authoritative catalog entry
//! (`markdown_descriptor` module-level note) that "MUST NOT relax any
//! feasibility check" against these typed fields.

use robot_types::ids::Name;
use robot_types::time::Duration;

/// A stable skill identifier.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct SkillId(pub u32);

/// Version of the numerical embedding used to place a skill in the
/// retrieval index.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct EmbeddingVersion(pub u32);

/// The typed, authoritative descriptor for one skill.
#[derive(Clone, Debug)]
pub struct SkillDescriptor {
    /// Stable identifier.
    pub id: SkillId,
    /// Stable name.
    pub name: Name,
    /// Semantic tags (e.g. `"jab"`, `"guard"`), for operator-facing display
    /// and as a coarse pre-filter; never a feasibility check by themselves.
    pub tags: Vec<Name>,
    /// Numerical feature embedding, in the retrieval index's feature space.
    pub embedding: Vec<f64>,
    /// Version of the embedding above.
    pub embedding_version: EmbeddingVersion,
    /// Robot profiles (by name) this skill is compatible with.
    pub embodiment_compatibility: Vec<String>,
    /// States in which this skill may legally begin — recorded as named
    /// guard predicates evaluated elsewhere; this field names them rather
    /// than encoding the predicate itself, since the predicate's inputs are
    /// specific to the deployed robot's observation layout.
    pub initiation_guards: Vec<Name>,
    /// Named termination condition.
    pub termination_condition: Name,
    /// Expected duration, for scheduling and operator display.
    pub expected_duration: Duration,
    /// Whether this skill assumes ground contact at its feet (or
    /// equivalent support), used by the feasibility filter.
    pub assumes_ground_contact: bool,
    /// Disturbance magnitudes (in the task's own units) this skill was
    /// validated to recover from; empty means untested.
    pub supported_disturbances: Vec<f64>,
    /// Resident memory this skill's inference requires, in bytes.
    pub inference_memory_bytes: u64,
    /// Skills this one may transition to, by ID. The transition *contract*
    /// itself (guard, dwell time) lives with the policy bundle
    /// (`robot_types::bundle::TransitionContract`); this is the adjacency
    /// the router consults before even considering a switch.
    pub outgoing_transitions: Vec<SkillId>,
}

impl SkillDescriptor {
    /// Whether this skill is compatible with the named robot profile.
    #[must_use]
    pub fn compatible_with(&self, robot_profile: &str) -> bool {
        self.embodiment_compatibility
            .iter()
            .any(|p| p == robot_profile)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor() -> SkillDescriptor {
        SkillDescriptor {
            id: SkillId(1),
            name: Name::new("jab").unwrap(),
            tags: vec![Name::new("strike").unwrap()],
            embedding: vec![0.1, 0.2, 0.3],
            embedding_version: EmbeddingVersion(1),
            embodiment_compatibility: vec!["fixture-humanoid/1.0".to_owned()],
            initiation_guards: vec![Name::new("double-support").unwrap()],
            termination_condition: Name::new("strike-complete").unwrap(),
            expected_duration: Duration::from_millis(400),
            assumes_ground_contact: true,
            supported_disturbances: vec![5.0, 10.0],
            inference_memory_bytes: 4096,
            outgoing_transitions: vec![SkillId(2)],
        }
    }

    #[test]
    fn compatibility_checks_the_typed_field_not_a_tag() {
        let d = descriptor();
        assert!(d.compatible_with("fixture-humanoid/1.0"));
        assert!(!d.compatible_with("other-robot/2.0"));
    }
}

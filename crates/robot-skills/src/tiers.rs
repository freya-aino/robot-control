//! The three retrieval tiers of SPECS.md Section 10.6, kept as distinct
//! types so they cannot be conflated in code the way the section warns
//! against in prose: "They MUST be distinguished in design, in
//! configuration and in every report, because the published latency
//! numbers for one say nothing about another."
//!
//! | Tier | Type here | Deadline |
//! | --- | --- | --- |
//! | R0 — corpus retrieval | not this crate's concern; SPECS.md §9.7 (`robot-learn`'s offline track) | none |
//! | R1 — between-episode working-set selection | [`WorkingSet`] | completes before mode entry |
//! | R2 — in-loop selection | [`Router`] | the only tier with a deadline |

use crate::descriptor::SkillId;
use crate::index::{ExactIndex, IndexError};
use robot_types::tier::Generation;

/// An immutable, generation-tagged working set published by R1 for R2 to
/// read. SPECS.md Section 10.3: "Switch at a defined control boundary using
/// generation-tagged immutable handles."
pub struct WorkingSet {
    generation: Generation,
    index: ExactIndex,
    resident_skills: Vec<SkillId>,
}

impl WorkingSet {
    /// Builds a working set. This is the R1 operation: "Offline-equivalent,
    /// outside the control process," per Section 10.6 — it is not called
    /// from R2's own query path.
    #[must_use]
    pub fn new(generation: Generation, index: ExactIndex, resident_skills: Vec<SkillId>) -> Self {
        Self {
            generation,
            index,
            resident_skills,
        }
    }

    /// The generation this working set was published at.
    #[must_use]
    pub const fn generation(&self) -> Generation {
        self.generation
    }

    /// The skills kept resident for this working set (safety/recovery and
    /// active experts, per Section 10.3 — "stay resident").
    #[must_use]
    pub fn resident_skills(&self) -> &[SkillId] {
        &self.resident_skills
    }
}

/// The in-loop router: R2. Holds the current working set and the currently
/// active skill, and answers queries with no file, network, embedding or
/// allocation work, per Section 10.6 item 1.
pub struct Router {
    working_set: Option<WorkingSet>,
    current_skill: Option<SkillId>,
    applied_generation: Generation,
}

impl Default for Router {
    fn default() -> Self {
        Self::new()
    }
}

impl Router {
    /// A router with no working set yet.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            working_set: None,
            current_skill: None,
            applied_generation: Generation::FIRST,
        }
    }

    /// Installs a new working set, if its generation is newer than the one
    /// currently applied (SPECS.md §5.1 item 4: stale-generation rejection).
    ///
    /// Returns whether it was installed.
    pub fn install(&mut self, working_set: WorkingSet) -> bool {
        if working_set.generation() <= self.applied_generation && self.working_set.is_some() {
            return false;
        }
        self.applied_generation = working_set.generation();
        self.working_set = Some(working_set);
        true
    }

    /// The currently active skill, if any.
    #[must_use]
    pub const fn current_skill(&self) -> Option<SkillId> {
        self.current_skill
    }

    /// Queries the resident index for the nearest skills to `features`,
    /// writing into `out` with no allocation.
    ///
    /// Section 10.6 item 6: "Every tier MUST have a defined empty-result
    /// behavior... R2 retains the current controller." Accordingly, an
    /// empty result or a missing working set here is `Ok(0)`, never an
    /// error that would stall a decision — the caller is expected to keep
    /// `current_skill` unchanged when this returns 0.
    pub fn query_into(
        &self,
        features: &[f64],
        out: &mut [(SkillId, f64)],
    ) -> Result<usize, IndexError> {
        let Some(ws) = &self.working_set else {
            return Ok(0);
        };
        if ws.index.is_empty() {
            return Ok(0);
        }
        ws.index.nearest_into(features, out)
    }

    /// Switches the active skill, if `candidate` is present in the current
    /// working set's resident skills (SPECS.md §10.3: "No expert may be
    /// activated while its input/output schema or state is incompatible" —
    /// residency is the cheap half of that check this router performs;
    /// schema compatibility is the caller's job at bundle-load time).
    pub fn switch_to(&mut self, candidate: SkillId) -> bool {
        let resident = self
            .working_set
            .as_ref()
            .is_some_and(|ws| ws.resident_skills().contains(&candidate));
        if resident {
            self.current_skill = Some(candidate);
        }
        resident
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn working_set(generation: Generation) -> WorkingSet {
        let index =
            ExactIndex::build(1, vec![(SkillId(1), vec![0.0]), (SkillId(2), vec![10.0])]).unwrap();
        WorkingSet::new(generation, index, vec![SkillId(1), SkillId(2)])
    }

    #[test]
    fn a_stale_generation_is_rejected() {
        let mut router = Router::new();
        assert!(router.install(working_set(Generation(2))));
        assert!(
            !router.install(working_set(Generation(1))),
            "older generation must be rejected"
        );
        assert!(
            !router.install(working_set(Generation(2))),
            "equal generation is not newer"
        );
        assert!(router.install(working_set(Generation(3))));
    }

    #[test]
    fn query_with_no_working_set_degrades_to_empty_rather_than_erroring() {
        let router = Router::new();
        let mut out = [(SkillId(0), 0.0); 2];
        assert_eq!(router.query_into(&[0.0], &mut out).unwrap(), 0);
    }

    #[test]
    fn switching_to_a_nonresident_skill_is_refused() {
        let mut router = Router::new();
        router.install(working_set(Generation(1)));
        assert!(router.switch_to(SkillId(1)));
        assert_eq!(router.current_skill(), Some(SkillId(1)));
        assert!(!router.switch_to(SkillId(99)));
        assert_eq!(
            router.current_skill(),
            Some(SkillId(1)),
            "a refused switch must not change state"
        );
    }
}

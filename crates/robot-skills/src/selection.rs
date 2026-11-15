//! In-loop skill selection (the R2 tier): the state-conditioned feasibility
//! filter and router of SPECS.md Sections 10.1 to 10.4, with the execution
//! state and logging of Section 10.2.
//!
//! [`Selector::decide`] runs at the tactical tier, once per decision, over
//! resident data only and without allocating (Section 10.6 item 1): its
//! candidate buffer, guard table, descriptor table and counters are all
//! sized when the selector is built at R1. Each call does, in order:
//!
//! 1. **Fallback first.** A fall or an emergency override never waits for
//!    retrieval: the configured recovery skill is chosen if it is resident
//!    and its guards pass, preempting any dwell time (Section 10.4:
//!    "Recovery can preempt dwell times"), and otherwise the
//!    [`Decision::Protective`] response is returned. "Guard posture is not
//!    a universal fallback when already falling."
//! 2. **Retrieve** the nearest resident candidates. None: keep the current
//!    controller (Section 10.6 item 6).
//! 3. **Out of distribution**: if even the nearest candidate is farther
//!    than the calibrated distance, keep the current controller.
//! 4. **Uncertainty**: if the best two candidates are closer than the
//!    margin and neither is the current skill, the choice is a coin flip;
//!    keep the current controller.
//! 5. **Feasibility, transition and dwell** per candidate: every initiation
//!    guard named by the descriptor must exist in the [`GuardTable`] and
//!    pass on the current [`FeasibilityState`] (an unknown guard fails
//!    closed); the current skill's outgoing transitions must include it; the
//!    current skill must have run its minimum dwell.
//! 6. **Hysteresis**: switch only if the candidate beats the current
//!    skill's own distance by the hysteresis margin.
//!
//! Every rejection is counted by reason and every switch by skill, in fixed
//! arrays ([`RoutingStats`]), which is Section 10.2's "Log rejection
//! reasons and routing frequency". A switch always says how recurrent
//! state is handled ([`RecurrentAction::Reset`]): Section 10.3's "Recurrent
//! state must be reset or transformed by an explicit contract".

use robot_types::ids::Name;

use crate::descriptor::{SkillDescriptor, SkillId};
use crate::tiers::Router;

/// Which feet carry the robot.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stance {
    /// Both feet.
    Double,
    /// Left foot only.
    Left,
    /// Right foot only.
    Right,
    /// Neither (a jump, or a fall).
    Flight,
}

/// The physical state feasibility is judged on (Section 10.1's "stance,
/// contact mode, joint velocity, available support, reach, actuator health
/// and opponent-relative geometry").
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct FeasibilityState {
    /// Stance.
    pub stance: Stance,
    /// Largest joint speed fraction of its limit, `[0, 1]`.
    pub joint_speed_fraction: f64,
    /// Whether external support (a wall, a rope) is available.
    pub support_available: bool,
    /// Worst actuator health, `[0, 1]` (thermal and fault margin).
    pub actuator_health: f64,
    /// Distance to the opponent, metres, if one is tracked.
    pub opponent_distance: Option<f64>,
    /// The base is falling.
    pub falling: bool,
}

/// A named initiation guard.
pub type Guard = fn(&FeasibilityState) -> bool;

/// Guard names to predicates, fixed at R1.
#[derive(Clone, Default)]
pub struct GuardTable {
    entries: Vec<(Name, Guard)>,
}

impl GuardTable {
    /// A table over `entries`.
    #[must_use]
    pub fn new(entries: Vec<(Name, Guard)>) -> Self {
        Self { entries }
    }

    /// Whether every guard in `names` exists and passes. An unknown guard
    /// fails: a descriptor naming a check nobody implemented is not
    /// feasible.
    #[must_use]
    pub fn all_pass(&self, names: &[Name], state: &FeasibilityState) -> bool {
        names.iter().all(|n| {
            self.entries
                .iter()
                .find(|(m, _)| m == n)
                .is_some_and(|(_, g)| g(state))
        })
    }
}

/// Why a candidate was not selected.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Rejection {
    /// A guard failed or is unknown.
    Infeasible = 0,
    /// The current skill's transition contract does not reach it.
    NoTransition = 1,
    /// The current skill has not run its minimum dwell.
    Dwell = 2,
    /// Not better than the current skill by the hysteresis margin.
    Hysteresis = 3,
    /// Nearest distance beyond the calibrated range.
    OutOfDistribution = 4,
    /// The best two candidates were indistinguishable.
    LowMargin = 5,
    /// The router refused (not resident).
    NotResident = 6,
    /// No descriptor for the retrieved id.
    UnknownSkill = 7,
    /// The query features contained a non-finite value, so no distance to
    /// any candidate is meaningful (and `NaN` would slip past the
    /// out-of-distribution gate, whose comparison is false for it).
    NonFiniteQuery = 8,
}

const REJECTIONS: usize = 9;

/// Routing counters, fixed-size.
#[derive(Clone, Debug)]
pub struct RoutingStats {
    /// Rejections, indexed by [`Rejection`] as `usize`.
    pub rejections: [u64; REJECTIONS],
    /// Switches per resident skill, in descriptor order.
    pub switches: Vec<u64>,
    /// Decisions that kept the current controller.
    pub kept: u64,
    /// Decisions that went to recovery or the protective response.
    pub fallbacks: u64,
}

/// What the recurrent state of the next skill starts from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RecurrentAction {
    /// Zeroed: the only contract implemented.
    Reset,
}

/// A decision.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Decision {
    /// Keep the current controller.
    Keep,
    /// Switch, at the next control boundary.
    Switch {
        /// The new skill.
        to: SkillId,
        /// Recurrent-state handling.
        recurrent: RecurrentAction,
        /// Whether a dwell was preempted by recovery.
        preempted: bool,
    },
    /// No recovery skill is available: the robot-specific protective
    /// response (controlled stop, support seeking) is the caller's to run.
    Protective,
}

/// Execution state (Section 10.2).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ExecutionState {
    /// Current skill.
    pub current: Option<SkillId>,
    /// When it started, ns.
    pub entered_ns: i64,
    /// The last accepted proposal and its expiry, ns.
    pub last_proposal: Option<(SkillId, i64)>,
    /// An emergency override is active.
    pub emergency_override: bool,
}

/// Selector settings, calibrated per working set.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SelectorConfig {
    /// Squared feature distance beyond which a query is out of
    /// distribution.
    pub ood_distance: f64,
    /// Smallest gap between the best two squared distances to decide.
    pub min_margin: f64,
    /// How much better (squared distance) a candidate must be than the
    /// current skill.
    pub hysteresis: f64,
    /// Minimum dwell, ns.
    pub min_dwell_ns: i64,
    /// How long an accepted proposal stays valid, ns.
    pub proposal_ttl_ns: i64,
    /// The recovery skill used when falling.
    pub recovery: Option<SkillId>,
}

/// The R2 selector.
pub struct Selector {
    config: SelectorConfig,
    descriptors: Vec<SkillDescriptor>,
    guards: GuardTable,
    buffer: Vec<(SkillId, f64)>,
    /// Execution state.
    pub state: ExecutionState,
    /// Counters.
    pub stats: RoutingStats,
}

impl Selector {
    /// A selector over the resident `descriptors`, considering up to
    /// `candidates` nearest skills per decision. Built at R1.
    #[must_use]
    pub fn new(
        config: SelectorConfig,
        descriptors: Vec<SkillDescriptor>,
        guards: GuardTable,
        candidates: usize,
    ) -> Self {
        let n = descriptors.len();
        Self {
            config,
            descriptors,
            guards,
            buffer: vec![(SkillId(0), 0.0); candidates.max(1)],
            state: ExecutionState {
                current: None,
                entered_ns: 0,
                last_proposal: None,
                emergency_override: false,
            },
            stats: RoutingStats {
                rejections: [0; REJECTIONS],
                switches: vec![0; n],
                kept: 0,
                fallbacks: 0,
            },
        }
    }

    fn descriptor(&self, id: SkillId) -> Option<&SkillDescriptor> {
        self.descriptors.iter().find(|d| d.id == id)
    }

    fn reject(&mut self, r: Rejection) {
        self.stats.rejections[r as usize] += 1;
    }

    fn commit(
        &mut self,
        router: &mut Router,
        to: SkillId,
        now_ns: i64,
        preempted: bool,
    ) -> Decision {
        if !router.switch_to(to) {
            self.reject(Rejection::NotResident);
            self.stats.kept += 1;
            return Decision::Keep;
        }
        if let Some(i) = self.descriptors.iter().position(|d| d.id == to) {
            self.stats.switches[i] += 1;
        }
        self.state.current = Some(to);
        self.state.entered_ns = now_ns;
        self.state.last_proposal = Some((to, now_ns + self.config.proposal_ttl_ns));
        Decision::Switch {
            to,
            recurrent: RecurrentAction::Reset,
            preempted,
        }
    }

    /// The accepted proposal still valid at `now_ns`.
    #[must_use]
    pub fn valid_proposal(&self, now_ns: i64) -> Option<SkillId> {
        self.state
            .last_proposal
            .and_then(|(id, expiry)| (now_ns < expiry).then_some(id))
    }

    /// One decision.
    pub fn decide(
        &mut self,
        router: &mut Router,
        features: &[f64],
        fs: &FeasibilityState,
        now_ns: i64,
    ) -> Decision {
        let dwelling = self.state.current.is_some()
            && now_ns - self.state.entered_ns < self.config.min_dwell_ns;
        if fs.falling || self.state.emergency_override {
            self.stats.fallbacks += 1;
            if let Some(r) = self.config.recovery {
                if self.state.current == Some(r) {
                    return Decision::Keep;
                }
                let ok = self
                    .descriptor(r)
                    .is_some_and(|d| self.guards.all_pass(&d.initiation_guards, fs));
                if ok {
                    // `commit` answers Keep when the router refuses the
                    // switch (recovery not resident). While falling, "keep
                    // the ordinary skill" is not an acceptable translation
                    // of that refusal: fall through to the protective
                    // response, as for a recovery whose guards fail.
                    let d = self.commit(router, r, now_ns, dwelling);
                    if matches!(d, Decision::Keep) {
                        return Decision::Protective;
                    }
                    return d;
                }
            }
            return Decision::Protective;
        }
        if features.iter().any(|v| !v.is_finite()) {
            self.reject(Rejection::NonFiniteQuery);
            self.stats.kept += 1;
            return Decision::Keep;
        }
        let n = router
            .query_into(features, &mut self.buffer)
            .unwrap_or_default();
        if n == 0 {
            self.stats.kept += 1;
            return Decision::Keep;
        }
        if self.buffer[0].1 > self.config.ood_distance {
            self.reject(Rejection::OutOfDistribution);
            self.stats.kept += 1;
            return Decision::Keep;
        }
        let current = self.state.current;
        if n >= 2
            && self.buffer[1].1 - self.buffer[0].1 < self.config.min_margin
            && current != Some(self.buffer[0].0)
            && current != Some(self.buffer[1].0)
        {
            self.reject(Rejection::LowMargin);
            self.stats.kept += 1;
            return Decision::Keep;
        }
        let current_distance = (0..n)
            .find(|&i| Some(self.buffer[i].0) == current)
            .map_or(f64::INFINITY, |i| self.buffer[i].1);
        for i in 0..n {
            let (id, d) = self.buffer[i];
            if Some(id) == current {
                // The current skill is still the best feasible choice.
                self.stats.kept += 1;
                return Decision::Keep;
            }
            let Some(desc) = self.descriptor(id) else {
                self.reject(Rejection::UnknownSkill);
                continue;
            };
            if !self.guards.all_pass(&desc.initiation_guards, fs) {
                self.reject(Rejection::Infeasible);
                continue;
            }
            if let Some(c) = current.and_then(|c| self.descriptor(c)) {
                if !c.outgoing_transitions.contains(&id) {
                    self.reject(Rejection::NoTransition);
                    continue;
                }
            }
            if dwelling {
                self.reject(Rejection::Dwell);
                continue;
            }
            if d + self.config.hysteresis > current_distance {
                self.reject(Rejection::Hysteresis);
                continue;
            }
            return self.commit(router, id, now_ns, false);
        }
        self.stats.kept += 1;
        Decision::Keep
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::EmbeddingVersion;
    use crate::index::ExactIndex;
    use crate::tiers::WorkingSet;
    use robot_types::tier::Generation;
    use robot_types::time::Duration;

    fn skill(id: u32, guards: &[&str], next: &[u32]) -> SkillDescriptor {
        SkillDescriptor {
            id: SkillId(id),
            name: Name::new(&format!("s{id}")).unwrap(),
            tags: Vec::new(),
            embedding: vec![0.0],
            embedding_version: EmbeddingVersion(1),
            embodiment_compatibility: vec!["g1".into()],
            initiation_guards: guards.iter().map(|g| Name::new(g).unwrap()).collect(),
            termination_condition: Name::new("done").unwrap(),
            expected_duration: Duration::from_millis(400),
            assumes_ground_contact: true,
            supported_disturbances: Vec::new(),
            inference_memory_bytes: 0,
            outgoing_transitions: next.iter().map(|&n| SkillId(n)).collect(),
        }
    }

    fn double_stance(s: &FeasibilityState) -> bool {
        s.stance == Stance::Double
    }

    fn in_range(s: &FeasibilityState) -> bool {
        s.opponent_distance.is_some_and(|d| d < 1.0)
    }

    fn standing() -> FeasibilityState {
        FeasibilityState {
            stance: Stance::Double,
            joint_speed_fraction: 0.1,
            support_available: false,
            actuator_health: 1.0,
            opponent_distance: Some(0.8),
            falling: false,
        }
    }

    /// guard (1) at feature 0, jab (2) at 1, hook (3) at 1.2, recovery (9)
    /// at 50.
    fn setup() -> (Selector, Router) {
        let skills = vec![
            skill(1, &["double_stance"], &[2, 3, 9]),
            skill(2, &["double_stance", "in_range"], &[1, 9]),
            skill(
                3,
                &["double_stance", "in_range", "unimplemented_check"],
                &[1, 9],
            ),
            skill(9, &[], &[1]),
        ];
        let index = ExactIndex::build(
            1,
            vec![
                (SkillId(1), vec![0.0]),
                (SkillId(2), vec![1.0]),
                (SkillId(3), vec![1.2]),
                (SkillId(9), vec![50.0]),
            ],
        )
        .unwrap();
        let mut router = Router::new();
        assert!(router.install(WorkingSet::new(
            Generation(2),
            index,
            vec![SkillId(1), SkillId(2), SkillId(3), SkillId(9)]
        )));
        let guards = GuardTable::new(vec![
            (Name::new("double_stance").unwrap(), double_stance as Guard),
            (Name::new("in_range").unwrap(), in_range as Guard),
        ]);
        let config = SelectorConfig {
            ood_distance: 4.0,
            min_margin: 0.01,
            hysteresis: 0.1,
            min_dwell_ns: 200_000_000,
            proposal_ttl_ns: 100_000_000,
            recovery: Some(SkillId(9)),
        };
        (Selector::new(config, skills, guards, 4), router)
    }

    #[test]
    fn a_feasible_reachable_candidate_is_switched_to_with_a_reset_contract() {
        let (mut sel, mut router) = setup();
        let d = sel.decide(&mut router, &[0.0], &standing(), 0);
        assert_eq!(
            d,
            Decision::Switch {
                to: SkillId(1),
                recurrent: RecurrentAction::Reset,
                preempted: false
            }
        );
        // Near the jab after the dwell: switch to it.
        let d = sel.decide(&mut router, &[0.9], &standing(), 300_000_000);
        assert_eq!(
            d,
            Decision::Switch {
                to: SkillId(2),
                recurrent: RecurrentAction::Reset,
                preempted: false
            }
        );
        assert_eq!(router.current_skill(), Some(SkillId(2)));
        assert_eq!(sel.valid_proposal(350_000_000), Some(SkillId(2)));
        assert_eq!(sel.valid_proposal(500_000_000), None);
    }

    #[test]
    fn dwell_out_of_distribution_and_unknown_guards_keep_the_current_skill() {
        let (mut sel, mut router) = setup();
        sel.decide(&mut router, &[0.0], &standing(), 0);
        assert_eq!(
            sel.decide(&mut router, &[0.9], &standing(), 50_000_000),
            Decision::Keep
        );
        assert!(sel.stats.rejections[Rejection::Dwell as usize] > 0);
        assert_eq!(
            sel.decide(&mut router, &[30.0], &standing(), 900_000_000),
            Decision::Keep
        );
        assert!(sel.stats.rejections[Rejection::OutOfDistribution as usize] > 0);
        // Nearest to the hook, whose guard list names an unimplemented
        // check: it is infeasible, and the jab is chosen instead.
        let d = sel.decide(&mut router, &[1.2], &standing(), 900_000_000);
        assert_eq!(
            d,
            Decision::Switch {
                to: SkillId(2),
                recurrent: RecurrentAction::Reset,
                preempted: false
            }
        );
        assert!(sel.stats.rejections[Rejection::Infeasible as usize] > 0);
    }

    #[test]
    fn falling_preempts_dwell_for_recovery_or_goes_protective() {
        let (mut sel, mut router) = setup();
        sel.decide(&mut router, &[0.0], &standing(), 0);
        let falling = FeasibilityState {
            falling: true,
            stance: Stance::Flight,
            ..standing()
        };
        let d = sel.decide(&mut router, &[0.0], &falling, 10);
        assert_eq!(
            d,
            Decision::Switch {
                to: SkillId(9),
                recurrent: RecurrentAction::Reset,
                preempted: true
            }
        );
        let (mut sel2, mut router2) = setup();
        sel2.config.recovery = None;
        assert_eq!(
            sel2.decide(&mut router2, &[0.0], &falling, 10),
            Decision::Protective
        );
        assert_eq!(sel2.stats.fallbacks, 1);
    }

    #[test]
    fn an_empty_router_keeps_the_current_controller() {
        let (mut sel, _) = setup();
        let mut empty = Router::new();
        assert_eq!(
            sel.decide(&mut empty, &[0.0], &standing(), 0),
            Decision::Keep
        );
    }

    #[test]
    fn falling_with_a_non_resident_recovery_goes_protective() {
        // The recovery skill is configured and its (empty) guard list
        // passes, but no working set is installed, so the router refuses
        // the switch. The ordinary skill must not keep running during a
        // fall: the decision falls through to Protective.
        let (mut sel, _) = setup();
        let mut empty = Router::new();
        let falling = FeasibilityState {
            falling: true,
            stance: Stance::Flight,
            ..standing()
        };
        assert_eq!(
            sel.decide(&mut empty, &[0.0], &falling, 10),
            Decision::Protective
        );
        assert_eq!(sel.stats.fallbacks, 1);
        assert!(sel.stats.rejections[Rejection::NotResident as usize] > 0);
    }

    #[test]
    fn a_non_finite_query_keeps_and_counts_instead_of_panicking() {
        // NaN distances would both panic the former partial_cmp sort and
        // slip past the out-of-distribution gate (NaN > limit is false).
        let (mut sel, mut router) = setup();
        assert_eq!(
            sel.decide(&mut router, &[f64::NAN], &standing(), 0),
            Decision::Keep
        );
        assert_eq!(sel.stats.rejections[Rejection::NonFiniteQuery as usize], 1);
    }
}

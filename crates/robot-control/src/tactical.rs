//! The tactical tier, per SPECS.md Sections 5.1 and 10: the R2 skill
//! router running above the policy tier at its own, slower rate, and a
//! skill-conditioned bank of resident experts below it.
//!
//! [`TacticalOperation`] is a [`LoopOperation`] over a [`SkillBank`]. Each
//! tick it is told the loop's time; once per tactical period it maps the
//! normalized observation to retrieval features and a feasibility state,
//! runs `robot_skills::selection::Selector::decide`, and publishes the
//! result as a [`Proposal`] stamped with its production time and expiry.
//! The policy tier then applies the newest **unexpired** proposal by
//! selecting that skill's expert, and reports its age as the tick's
//! decision age (Section 5's table; Section 5.1 item 4 attributes it to
//! the tier that produced it).
//!
//! Section 5.1's tier rules are checked when the operation is built:
//! [`TacticalConfig::validate`] refuses a tactical period shorter than the
//! policy period (rates must not increase with height) and a proposal
//! lifetime shorter than one tactical period (every proposal would expire
//! before the next replaced it). Item 3 (a tier stays functional when the
//! tier above is late, absent or expired) and item 7 (removing the highest
//! tier leaves a safe controller) are the same code path here: with no
//! valid proposal the bank runs its declared default expert.
//!
//! Everything a tick touches is sized at construction: the feature buffer,
//! the selector's candidate buffer and counters, the bank. Section 5:
//! "The control path MUST NOT perform ... heap allocation after
//! activation."

use robot_nn::mlp::{MlpError, Scratch};
use robot_nn::policy::Policy;
use robot_skills::descriptor::SkillId;
use robot_skills::selection::{Decision, FeasibilityState, Selector};
use robot_skills::tiers::Router;
use robot_types::time::{Duration, Timestamp};

use crate::operation::{LoopOperation, WidthMismatch};

/// Resident experts, one per skill, all with the same widths.
pub struct SkillBank<P: Policy> {
    experts: Vec<(SkillId, P)>,
    default: usize,
}

/// Why a bank could not be built.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BankError {
    /// No experts.
    Empty,
    /// Experts disagree on input or output width.
    WidthMismatch {
        /// The offending expert's index.
        expert: usize,
    },
    /// The default skill is not in the bank.
    NoDefault,
}

impl<P: Policy> SkillBank<P> {
    /// A bank whose default (the expert run with no valid proposal) is
    /// `default`.
    pub fn new(experts: Vec<(SkillId, P)>, default: SkillId) -> Result<Self, BankError> {
        let first = experts.first().ok_or(BankError::Empty)?;
        let (iw, ow) = (first.1.input_width(), first.1.output_width());
        if let Some(i) = experts
            .iter()
            .position(|(_, p)| p.input_width() != iw || p.output_width() != ow)
        {
            return Err(BankError::WidthMismatch { expert: i });
        }
        let default = experts
            .iter()
            .position(|(s, _)| *s == default)
            .ok_or(BankError::NoDefault)?;
        Ok(Self { experts, default })
    }

    fn index_of(&self, skill: Option<SkillId>) -> usize {
        skill
            .and_then(|s| self.experts.iter().position(|(id, _)| *id == s))
            .unwrap_or(self.default)
    }

    /// The skill whose expert runs for `skill` (the default when absent).
    #[must_use]
    pub fn resolve(&self, skill: Option<SkillId>) -> SkillId {
        self.experts[self.index_of(skill)].0
    }
}

impl<P: Policy> Policy for SkillBank<P> {
    fn input_width(&self) -> usize {
        self.experts[0].1.input_width()
    }

    fn output_width(&self) -> usize {
        self.experts[0].1.output_width()
    }

    fn scratch_len(&self) -> usize {
        self.experts
            .iter()
            .map(|(_, p)| p.scratch_len())
            .max()
            .unwrap_or(0)
    }

    /// The default expert; [`TacticalOperation`] chooses the expert itself.
    fn act(
        &self,
        observation: &[f64],
        action: &mut [f64],
        scratch: &mut Scratch,
    ) -> Result<(), MlpError> {
        self.experts[self.default]
            .1
            .act(observation, action, scratch)
    }
}

/// A tactical decision crossing into the policy tier.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Proposal {
    /// The skill.
    pub skill: SkillId,
    /// When the tactical tier produced it.
    pub produced_at: Timestamp,
    /// After this the policy tier must not apply it.
    pub expires_at: Timestamp,
}

/// Tier timing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TacticalConfig {
    /// The policy tier's period.
    pub policy_period: Duration,
    /// The tactical tier's period.
    pub tactical_period: Duration,
    /// How long a proposal stays valid.
    pub proposal_ttl: Duration,
}

/// A Section 5.1 tier-rule violation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TierError {
    /// The tactical tier would run faster than the policy tier.
    RateIncreasesWithHeight,
    /// Proposals would expire before the next one arrives.
    TtlShorterThanPeriod,
}

impl TacticalConfig {
    /// Checks the tier rules.
    pub fn validate(&self) -> Result<(), TierError> {
        if self.tactical_period.nanos() < self.policy_period.nanos() {
            return Err(TierError::RateIncreasesWithHeight);
        }
        if self.proposal_ttl.nanos() < self.tactical_period.nanos() {
            return Err(TierError::TtlShorterThanPeriod);
        }
        Ok(())
    }
}

/// Maps the normalized observation to retrieval features, in place.
pub type FeatureMap = fn(&[f64], &mut [f64]);
/// Maps the normalized observation to the feasibility state.
pub type FeasibilityMap = fn(&[f64]) -> FeasibilityState;

/// Counters of the tier boundary.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct TacticalStats {
    /// Tactical decisions run.
    pub decisions: u64,
    /// Ticks on which the newest proposal had expired.
    pub expired: u64,
    /// Ticks that ran the default expert for lack of a valid proposal.
    pub defaulted: u64,
    /// Protective responses requested.
    pub protective: u64,
}

/// The tactical tier as a loop operation.
pub struct TacticalOperation {
    config: TacticalConfig,
    selector: Selector,
    router: Router,
    features: Vec<f64>,
    feature_map: FeatureMap,
    feasibility: FeasibilityMap,
    now: Option<Timestamp>,
    next_decision: Option<Timestamp>,
    proposal: Option<Proposal>,
    applied: Option<Proposal>,
    /// The most recent protective request, for the caller to act on.
    pub protective_requested: bool,
    /// Boundary counters.
    pub stats: TacticalStats,
}

impl TacticalOperation {
    /// The tier, after checking Section 5.1's rules.
    pub fn new(
        config: TacticalConfig,
        selector: Selector,
        router: Router,
        feature_width: usize,
        feature_map: FeatureMap,
        feasibility: FeasibilityMap,
    ) -> Result<Self, TierError> {
        config.validate()?;
        Ok(Self {
            config,
            selector,
            router,
            features: vec![0.0; feature_width],
            feature_map,
            feasibility,
            now: None,
            next_decision: None,
            proposal: None,
            applied: None,
            protective_requested: false,
            stats: TacticalStats::default(),
        })
    }

    /// The selector, for its counters.
    #[must_use]
    pub fn selector(&self) -> &Selector {
        &self.selector
    }

    /// The router, for installing a new working set at R1.
    pub fn router_mut(&mut self) -> &mut Router {
        &mut self.router
    }

    /// The proposal applied on the last tick.
    #[must_use]
    pub fn applied(&self) -> Option<Proposal> {
        self.applied
    }

    fn decide(&mut self, now: Timestamp, observation: &[f64]) {
        (self.feature_map)(observation, &mut self.features);
        let fs = (self.feasibility)(observation);
        self.stats.decisions += 1;
        let skill = match self
            .selector
            .decide(&mut self.router, &self.features, &fs, now.nanos())
        {
            Decision::Switch { to, .. } => Some(to),
            Decision::Keep => self.selector.state.current,
            Decision::Protective => {
                self.stats.protective += 1;
                self.protective_requested = true;
                None
            }
        };
        if let Some(skill) = skill {
            self.protective_requested = false;
            self.proposal = Some(Proposal {
                skill,
                produced_at: now,
                expires_at: now.advanced(self.config.proposal_ttl),
            });
        }
    }
}

impl<P: Policy> LoopOperation<SkillBank<P>> for TacticalOperation {
    fn check_widths(
        &self,
        observation_width: usize,
        _action_width: usize,
    ) -> Result<(), WidthMismatch> {
        let _ = observation_width;
        Ok(())
    }

    fn observe_time(&mut self, now: Timestamp) {
        self.now = Some(now);
    }

    fn before_tick(&mut self, observation: &[f64]) {
        let Some(now) = self.now else { return };
        let due = self.next_decision.is_none_or(|t| now.nanos() >= t.nanos());
        if due {
            self.decide(now, observation);
            self.next_decision = Some(now.advanced(self.config.tactical_period));
        }
        self.applied = match self.proposal {
            Some(p) if now.nanos() < p.expires_at.nanos() => Some(p),
            Some(_) => {
                self.stats.expired += 1;
                None
            }
            None => None,
        };
        if self.applied.is_none() {
            self.stats.defaulted += 1;
        }
    }

    fn select_action(
        &mut self,
        policy: &SkillBank<P>,
        observation: &[f64],
        action: &mut [f64],
        scratch: &mut Scratch,
    ) -> Result<Option<f64>, MlpError> {
        let i = policy.index_of(self.applied.map(|p| p.skill));
        policy.experts[i].1.act(observation, action, scratch)?;
        Ok(None)
    }

    fn decision_age(&self, now: Timestamp) -> Option<Duration> {
        let p = self.applied?;
        now.since(p.produced_at).ok().map(|(age, _)| age)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_nn::layer::{Activation, Linear};
    use robot_nn::mlp::Mlp;
    use robot_skills::descriptor::{EmbeddingVersion, SkillDescriptor};
    use robot_skills::index::ExactIndex;
    use robot_skills::selection::{GuardTable, SelectorConfig, Stance};
    use robot_skills::tiers::WorkingSet;
    use robot_types::ids::Name;
    use robot_types::tier::Generation;
    use robot_types::time::ClockId;

    fn constant(value: f64) -> Mlp {
        let layer = Linear::new(1, 1, vec![0.0], vec![value]).unwrap();
        Mlp::new(vec![layer], vec![Activation::Identity]).unwrap()
    }

    fn skill(id: u32, next: &[u32]) -> SkillDescriptor {
        SkillDescriptor {
            id: SkillId(id),
            name: Name::new(&format!("s{id}")).unwrap(),
            tags: Vec::new(),
            embedding: vec![0.0],
            embedding_version: EmbeddingVersion(1),
            embodiment_compatibility: Vec::new(),
            initiation_guards: Vec::new(),
            termination_condition: Name::new("done").unwrap(),
            expected_duration: Duration::from_millis(100),
            assumes_ground_contact: true,
            supported_disturbances: Vec::new(),
            inference_memory_bytes: 0,
            outgoing_transitions: next.iter().map(|&n| SkillId(n)).collect(),
        }
    }

    fn features(obs: &[f64], out: &mut [f64]) {
        out[0] = obs[0];
    }

    fn feasibility(_obs: &[f64]) -> FeasibilityState {
        FeasibilityState {
            stance: Stance::Double,
            joint_speed_fraction: 0.0,
            support_available: false,
            actuator_health: 1.0,
            opponent_distance: None,
            falling: false,
        }
    }

    fn operation(ttl_ms: i64) -> TacticalOperation {
        let index =
            ExactIndex::build(1, vec![(SkillId(1), vec![0.0]), (SkillId(2), vec![1.0])]).unwrap();
        let mut router = Router::new();
        router.install(WorkingSet::new(
            Generation(2),
            index,
            vec![SkillId(1), SkillId(2)],
        ));
        let selector = Selector::new(
            SelectorConfig {
                ood_distance: 10.0,
                min_margin: 0.0,
                hysteresis: 0.0,
                min_dwell_ns: 0,
                proposal_ttl_ns: ttl_ms * 1_000_000,
                recovery: None,
            },
            vec![skill(1, &[2]), skill(2, &[1])],
            GuardTable::default(),
            2,
        );
        TacticalOperation::new(
            TacticalConfig {
                policy_period: Duration::from_millis(10),
                tactical_period: Duration::from_millis(40),
                proposal_ttl: Duration::from_millis(ttl_ms),
            },
            selector,
            router,
            1,
            features,
            feasibility,
        )
        .unwrap()
    }

    fn at(ms: i64) -> Timestamp {
        Timestamp::sampling(ms * 1_000_000, ClockId::HostMonotonic)
    }

    fn tick(op: &mut TacticalOperation, bank: &SkillBank<Mlp>, ms: i64, obs: f64) -> f64 {
        let mut action = [0.0];
        let mut scratch = Scratch::new(4);
        LoopOperation::<SkillBank<Mlp>>::observe_time(op, at(ms));
        LoopOperation::<SkillBank<Mlp>>::before_tick(op, &[obs]);
        LoopOperation::<SkillBank<Mlp>>::select_action(op, bank, &[obs], &mut action, &mut scratch)
            .unwrap();
        action[0]
    }

    fn bank() -> SkillBank<Mlp> {
        SkillBank::new(
            vec![(SkillId(1), constant(1.0)), (SkillId(2), constant(2.0))],
            SkillId(1),
        )
        .unwrap()
    }

    #[test]
    fn the_router_runs_at_its_own_rate_and_selects_the_expert() {
        let mut op = operation(80);
        let b = bank();
        assert_eq!(tick(&mut op, &b, 0, 0.0), 1.0);
        // Observation now near skill 2, but the next decision is at 40 ms.
        assert_eq!(tick(&mut op, &b, 10, 1.0), 1.0);
        assert_eq!(tick(&mut op, &b, 30, 1.0), 1.0);
        assert_eq!(tick(&mut op, &b, 40, 1.0), 2.0);
        assert_eq!(op.stats.decisions, 2);
        assert_eq!(
            LoopOperation::<SkillBank<Mlp>>::decision_age(&op, at(50)),
            Some(Duration::from_millis(10))
        );
    }

    #[test]
    fn an_expired_proposal_falls_back_to_the_default_expert() {
        let mut op = operation(40);
        let b = bank();
        tick(&mut op, &b, 0, 1.0);
        assert_eq!(op.applied().map(|p| p.skill), Some(SkillId(2)));
        // Freeze the router by moving its next decision past the TTL.
        op.next_decision = Some(at(1_000));
        assert_eq!(tick(&mut op, &b, 50, 1.0), 1.0);
        assert_eq!(op.stats.expired, 1);
        assert_eq!(
            LoopOperation::<SkillBank<Mlp>>::decision_age(&op, at(50)),
            None
        );
    }

    #[test]
    fn tier_rules_are_checked_at_construction() {
        let c = TacticalConfig {
            policy_period: Duration::from_millis(10),
            tactical_period: Duration::from_millis(5),
            proposal_ttl: Duration::from_millis(50),
        };
        assert_eq!(c.validate(), Err(TierError::RateIncreasesWithHeight));
        let c = TacticalConfig {
            tactical_period: Duration::from_millis(40),
            proposal_ttl: Duration::from_millis(20),
            ..c
        };
        assert_eq!(c.validate(), Err(TierError::TtlShorterThanPeriod));
    }

    #[test]
    fn a_bank_refuses_mismatched_experts_and_a_missing_default() {
        let wide = Mlp::new(
            vec![Linear::new(2, 1, vec![0.0, 0.0], vec![0.0]).unwrap()],
            vec![Activation::Identity],
        )
        .unwrap();
        assert_eq!(
            SkillBank::new(
                vec![(SkillId(1), constant(1.0)), (SkillId(2), wide)],
                SkillId(1)
            )
            .err(),
            Some(BankError::WidthMismatch { expert: 1 })
        );
        assert_eq!(
            SkillBank::new(vec![(SkillId(1), constant(1.0))], SkillId(9)).err(),
            Some(BankError::NoDefault)
        );
    }
}

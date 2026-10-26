//! The experience and replay storage contract of SPECS.md Section 7.6.
//!
//! "On-policy rollouts and off-policy replay MUST use the same versioned
//! transition and episode schema." This module defines that one schema. The
//! stores themselves live in `robot-learn`; what lives here is the record every
//! store must agree on, because item 7 requires several concurrent stores to be
//! "composable without a second transition format".
//!
//! Three details in the section are load-bearing and are encoded as types:
//!
//! * item 7 — a transition records "which policy or corpus produced them and at
//!   which generation, because a record whose origin is unknown cannot be used
//!   as a behavior target", hence [`TransitionOrigin`];
//! * item 8 — a latent-space action "MUST record the latent-space identity and
//!   version and the decoder artifact required to interpret it", and "a buffer
//!   MUST reject a mixture", hence [`ActionRepresentation`] and
//!   [`TransitionSchema::accepts`];
//! * SPECS.md Section 9.2 — termination and time-limit truncation are distinct,
//!   hence [`Termination`], whose `bootstraps` method is what a return
//!   calculation consults.

use crate::rng::SeedStream;
use crate::schema::SchemaVersion;
use crate::tier::Generation;
use core::fmt;

/// The schema version this crate writes for transitions.
pub const TRANSITION_SCHEMA_VERSION: SchemaVersion = SchemaVersion::new(1, 0);

/// How an episode step ended.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Termination {
    /// The episode continues.
    Continuing,
    /// A terminal state was reached; the value of the next state is zero.
    Terminal,
    /// A time limit ended the episode; the next state still has value.
    ///
    /// SPECS.md Section 9.2 requires this distinction explicitly, and
    /// PLAN.md Phase 3 lists "Distinguish truncation from terminal failure in
    /// return calculation" as its own work item.
    TimeLimit,
}

impl Termination {
    /// Whether the value of the next state enters the return target.
    #[must_use]
    pub const fn bootstraps(self) -> bool {
        !matches!(self, Self::Terminal)
    }

    /// Whether the episode ends here.
    #[must_use]
    pub const fn ends_episode(self) -> bool {
        !matches!(self, Self::Continuing)
    }

    /// The canonical identifier used in records.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::Continuing => "continuing",
            Self::Terminal => "terminal",
            Self::TimeLimit => "time-limit",
        }
    }
}

/// What produced a transition.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum TransitionOrigin {
    /// The policy currently being trained, at a named generation.
    CurrentPolicy {
        /// Learner update count when the transition was collected.
        generation: Generation,
    },
    /// A historical snapshot of the agent's own policy.
    HistoricalPolicy {
        /// Snapshot identity.
        snapshot_id: String,
        /// Generation the snapshot was taken at.
        generation: Generation,
    },
    /// A scripted controller.
    ScriptedPolicy {
        /// Controller identity.
        controller_id: String,
    },
    /// An external corpus, retrieved offline under SPECS.md Section 9.7.
    ///
    /// Every field here is required by Section 9.7.2 item 2: a retrieved
    /// transition is labelled with "its source corpus, source episode,
    /// embodiment and retrieval generation".
    ExternalCorpus {
        /// Corpus identity and revision.
        corpus: String,
        /// Episode within that corpus.
        source_episode: String,
        /// Embodiment the data was produced on. A corpus from another robot is
        /// a cross-embodiment claim under Section 11.1.
        embodiment: String,
        /// Which retrieval pass admitted it.
        retrieval_generation: Generation,
    },
    /// A real robot deployment episode.
    RobotDeployment {
        /// Episode identity in the Section 7.3 records.
        episode_id: String,
        /// Robot profile identity and version.
        robot_profile: String,
    },
}

impl TransitionOrigin {
    /// Whether this origin is the policy currently being optimized.
    #[must_use]
    pub const fn is_on_policy(&self) -> bool {
        matches!(self, Self::CurrentPolicy { .. })
    }

    /// Whether this origin is usable as a behaviour target for supervised
    /// averaging (SPECS.md Section 9.4.2 item 7).
    #[must_use]
    pub const fn is_behavior_target(&self) -> bool {
        matches!(
            self,
            Self::HistoricalPolicy { .. } | Self::ScriptedPolicy { .. }
        )
    }

    /// The embodiment the data came from, where it is known.
    #[must_use]
    pub fn embodiment(&self) -> Option<&str> {
        match self {
            Self::ExternalCorpus { embodiment, .. } => Some(embodiment),
            Self::RobotDeployment { robot_profile, .. } => Some(robot_profile),
            _ => None,
        }
    }

    /// The canonical kind identifier.
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::CurrentPolicy { .. } => "current-policy",
            Self::HistoricalPolicy { .. } => "historical-policy",
            Self::ScriptedPolicy { .. } => "scripted-policy",
            Self::ExternalCorpus { .. } => "external-corpus",
            Self::RobotDeployment { .. } => "robot-deployment",
        }
    }
}

/// The identity of a learned latent action space.
///
/// Inseparable from its decoder: SPECS.md Appendix A.2 defines a latent action
/// space as "identified by version and inseparable from the decoder artifact
/// that interprets it".
#[derive(Clone, PartialEq, Debug)]
pub struct LatentSpaceId {
    /// Stable name of the space.
    pub name: String,
    /// Version of the space.
    pub version: SchemaVersion,
    /// Content identity of the decoder artifact.
    pub decoder_artifact: String,
    /// Dimensionality.
    pub dimension: usize,
    /// The declared bound or manifold constraint.
    pub constraint: LatentConstraint,
}

/// The bound or manifold a latent action is constrained to.
///
/// SPECS.md Section 3 is explicit that this bounds a representation and nothing
/// physical: "A unit-norm latent vector bounds its representation, not joint
/// torques, balance, or contact forces."
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum LatentConstraint {
    /// Components bounded componentwise to `[-bound, bound]`.
    Box {
        /// The bound.
        bound: f64,
    },
    /// Constrained to the unit hypersphere.
    UnitHypersphere,
    /// Constrained to a ball of the given radius.
    Ball {
        /// The radius.
        radius: f64,
    },
}

impl LatentConstraint {
    /// Whether a vector satisfies the constraint, to the given tolerance.
    #[must_use]
    pub fn admits(&self, values: &[f64], tolerance: f64) -> bool {
        if values.iter().any(|v| !v.is_finite()) {
            return false;
        }
        match self {
            Self::Box { bound } => values.iter().all(|v| v.abs() <= bound + tolerance),
            Self::UnitHypersphere => {
                let norm = values.iter().map(|v| v * v).sum::<f64>().sqrt();
                (norm - 1.0).abs() <= tolerance
            }
            Self::Ball { radius } => {
                values.iter().map(|v| v * v).sum::<f64>().sqrt() <= radius + tolerance
            }
        }
    }

    /// The canonical identifier used in records.
    #[must_use]
    pub const fn id(&self) -> &'static str {
        match self {
            Self::Box { .. } => "box",
            Self::UnitHypersphere => "unit-hypersphere",
            Self::Ball { .. } => "ball",
        }
    }
}

/// Which space a transition's action lives in.
#[derive(Clone, PartialEq, Debug)]
pub enum ActionRepresentation {
    /// The actuator action space declared by the robot profile.
    ActuatorSpace,
    /// A learned latent space, with its identity.
    LatentSpace(LatentSpaceId),
}

impl ActionRepresentation {
    /// Whether two representations may share one buffer.
    ///
    /// SPECS.md Section 7.6 item 8: "A latent transition is not interchangeable
    /// with a joint-space transition, and a buffer MUST reject a mixture unless
    /// the schema explicitly carries both."
    #[must_use]
    pub fn interchangeable_with(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::ActuatorSpace, Self::ActuatorSpace) => true,
            (Self::LatentSpace(a), Self::LatentSpace(b)) => a == b,
            _ => false,
        }
    }
}

/// A named reward component.
///
/// SPECS.md Section 9.2: "Log reward terms independently to expose reward
/// exploitation. Persistent torque violations cannot be hidden by a good
/// aggregate return." The aggregate is therefore *derived* from the terms by
/// [`RewardTerms::total`], never stored alongside them where the two could
/// disagree.
#[derive(Clone, PartialEq, Debug)]
pub struct RewardTerm {
    /// Stable, versioned term name from the reward registry.
    pub name: String,
    /// The unweighted value.
    pub value: f64,
    /// The weight applied in the aggregate.
    pub weight: f64,
    /// Whether this term is a style or motion-prior objective.
    ///
    /// SPECS.md Section 9.4.2 item 5: a style score "MUST NOT" be presented as
    /// evidence of feasibility, stability or safety. Marking it in the record is
    /// what lets a report keep that promise mechanically.
    pub is_style_term: bool,
}

/// The reward components of one transition.
#[derive(Clone, Default, PartialEq, Debug)]
pub struct RewardTerms {
    terms: Vec<RewardTerm>,
}

impl RewardTerms {
    /// An empty set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a term, rejecting a duplicate name.
    pub fn push(&mut self, term: RewardTerm) -> Result<(), TransitionError> {
        if self.terms.iter().any(|t| t.name == term.name) {
            return Err(TransitionError::DuplicateRewardTerm { name: term.name });
        }
        self.terms.push(term);
        Ok(())
    }

    /// The terms.
    #[must_use]
    pub fn terms(&self) -> &[RewardTerm] {
        &self.terms
    }

    /// The weighted sum.
    #[must_use]
    pub fn total(&self) -> f64 {
        self.terms.iter().map(|t| t.value * t.weight).sum()
    }

    /// The weighted sum of the non-style terms only, which is the number a
    /// feasibility or stability claim may be made against.
    #[must_use]
    pub fn task_total(&self) -> f64 {
        self.terms
            .iter()
            .filter(|t| !t.is_style_term)
            .map(|t| t.value * t.weight)
            .sum()
    }

    /// A term by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&RewardTerm> {
        self.terms.iter().find(|t| t.name == name)
    }
}

/// The declared shape of transitions in a store.
///
/// A store validates every transition against this before admitting it, which is
/// what makes "the same versioned transition and episode schema" enforceable
/// rather than aspirational.
#[derive(Clone, PartialEq, Debug)]
pub struct TransitionSchema {
    /// Schema version.
    pub schema_version: SchemaVersion,
    /// Observation width.
    pub observation_width: usize,
    /// Action width.
    pub action_width: usize,
    /// Which space the actions live in.
    pub action_representation: ActionRepresentation,
    /// Robot profile identity and version these transitions apply to.
    pub robot_profile: String,
}

impl TransitionSchema {
    /// Whether a transition may be admitted to a store with this schema.
    pub fn accepts(&self, t: &Transition) -> Result<(), TransitionError> {
        if !self.schema_version.can_read(t.schema_version) {
            return Err(TransitionError::IncompatibleSchema {
                store: self.schema_version,
                transition: t.schema_version,
            });
        }
        if t.observation.len() != self.observation_width {
            return Err(TransitionError::WidthMismatch {
                field: "observation",
                expected: self.observation_width,
                found: t.observation.len(),
            });
        }
        if t.next_observation.len() != self.observation_width {
            return Err(TransitionError::WidthMismatch {
                field: "next_observation",
                expected: self.observation_width,
                found: t.next_observation.len(),
            });
        }
        if t.action.len() != self.action_width {
            return Err(TransitionError::WidthMismatch {
                field: "action",
                expected: self.action_width,
                found: t.action.len(),
            });
        }
        if !self
            .action_representation
            .interchangeable_with(&t.action_representation)
        {
            return Err(TransitionError::MixedActionSpaces {
                store: Box::new(self.action_representation.clone()),
                transition: Box::new(t.action_representation.clone()),
            });
        }
        if t.robot_profile != self.robot_profile {
            return Err(TransitionError::ProfileMismatch {
                store: self.robot_profile.clone(),
                transition: t.robot_profile.clone(),
            });
        }
        if let ActionRepresentation::LatentSpace(space) = &t.action_representation {
            if !space.constraint.admits(&t.action, 1e-6) {
                return Err(TransitionError::LatentConstraintViolated {
                    constraint: space.constraint,
                });
            }
            if space.dimension != t.action.len() {
                return Err(TransitionError::WidthMismatch {
                    field: "latent_dimension",
                    expected: space.dimension,
                    found: t.action.len(),
                });
            }
        }
        if let Some(index) = t
            .observation
            .iter()
            .chain(&t.next_observation)
            .chain(&t.action)
            .position(|v| !v.is_finite())
        {
            return Err(TransitionError::NotFinite { index });
        }
        Ok(())
    }
}

/// One stored transition.
#[derive(Clone, PartialEq, Debug)]
pub struct Transition {
    /// Schema version.
    pub schema_version: SchemaVersion,
    /// Episode this transition belongs to.
    pub episode_id: String,
    /// Timestep within the episode.
    pub timestep: u32,
    /// Environment that produced it.
    pub env_id: crate::ids::EnvId,
    /// Robot profile identity and version.
    pub robot_profile: String,
    /// Observation before the action.
    pub observation: Vec<f64>,
    /// The action.
    pub action: Vec<f64>,
    /// Which space the action lives in.
    pub action_representation: ActionRepresentation,
    /// Reward components.
    pub reward: RewardTerms,
    /// Observation after the action.
    pub next_observation: Vec<f64>,
    /// How the step ended.
    pub termination: Termination,
    /// Per-component validity of the observation.
    pub observation_validity: Vec<bool>,
    /// Task data needed to reproduce reward or goal relabeling.
    pub task_data: Vec<(String, f64)>,
    /// What produced this transition.
    pub origin: TransitionOrigin,
    /// Seed stream of the producing environment.
    pub seeds: SeedStream,
    /// Log-probability of the action under the behaviour policy, where the
    /// algorithm needs it. Absent for corpus data, which has no such policy.
    pub behavior_log_prob: Option<f64>,
}

/// Why a transition was rejected by a store.
#[derive(Clone, PartialEq, Debug)]
pub enum TransitionError {
    /// A vector width disagreed with the store's schema.
    WidthMismatch {
        /// Which field.
        field: &'static str,
        /// Width the schema declares.
        expected: usize,
        /// Width found.
        found: usize,
    },
    /// The store cannot read this transition's schema version.
    IncompatibleSchema {
        /// Store's version.
        store: SchemaVersion,
        /// Transition's version.
        transition: SchemaVersion,
    },
    /// A latent and a joint-space transition were offered to one buffer.
    ///
    /// Boxed: [`ActionRepresentation::LatentSpace`] carries a
    /// [`LatentSpaceId`] with two owned strings, which otherwise makes this
    /// the dominant variant of `TransitionError` and inflates every `Result`
    /// that returns it — including the hot-ish per-transition `accepts`
    /// check — regardless of which variant actually occurs.
    MixedActionSpaces {
        /// Store's representation.
        store: Box<ActionRepresentation>,
        /// Transition's representation.
        transition: Box<ActionRepresentation>,
    },
    /// The transition came from a different robot profile.
    ProfileMismatch {
        /// Store's profile.
        store: String,
        /// Transition's profile.
        transition: String,
    },
    /// A latent action fell outside its declared constraint.
    LatentConstraintViolated {
        /// The constraint.
        constraint: LatentConstraint,
    },
    /// A component was NaN or infinite.
    NotFinite {
        /// Index into the concatenated observation/action vectors.
        index: usize,
    },
    /// Two reward terms shared a name.
    DuplicateRewardTerm {
        /// The repeated name.
        name: String,
    },
    /// A sampled reference pointed at overwritten storage.
    ///
    /// SPECS.md Section 7.6 item 2: "stale or overwritten references MUST be
    /// rejected".
    StaleGeneration {
        /// Generation the reference expected.
        expected: Generation,
        /// Generation present in storage.
        found: Generation,
    },
    /// An on-policy store filled before the learner consumed it.
    ///
    /// SPECS.md Section 7.6 item 1: on-policy storage "MUST report capacity
    /// exhaustion rather than silently overwrite samples".
    CapacityExhausted {
        /// The store's capacity.
        capacity: usize,
    },
}

impl fmt::Display for TransitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WidthMismatch {
                field,
                expected,
                found,
            } => {
                write!(f, "{field}: expected width {expected}, found {found}")
            }
            Self::IncompatibleSchema { store, transition } => {
                write!(f, "store reads {store}, transition declares {transition}")
            }
            Self::MixedActionSpaces { store, transition } => write!(
                f,
                "store holds {store:?} actions, transition carries {transition:?}; \
                 a buffer must reject a mixture"
            ),
            Self::ProfileMismatch { store, transition } => {
                write!(
                    f,
                    "store is for {store:?}, transition came from {transition:?}"
                )
            }
            Self::LatentConstraintViolated { constraint } => {
                write!(
                    f,
                    "latent action violates its {} constraint",
                    constraint.id()
                )
            }
            Self::NotFinite { index } => write!(f, "component {index} is not finite"),
            Self::DuplicateRewardTerm { name } => write!(f, "duplicate reward term {name:?}"),
            Self::StaleGeneration { expected, found } => {
                write!(
                    f,
                    "stale reference: expected {expected}, storage holds {found}"
                )
            }
            Self::CapacityExhausted { capacity } => {
                write!(f, "on-policy storage exhausted at capacity {capacity}")
            }
        }
    }
}

impl core::error::Error for TransitionError {}

/// How a store retains what it is given.
///
/// SPECS.md Section 7.6 item 7: "each store declares its retention policy
/// (bounded ring, reservoir, append-only corpus), its admission rule, and
/// whether its contents are drawn from the current policy, a historical policy
/// or an external corpus."
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RetentionPolicy {
    /// A fixed-capacity ring that overwrites the oldest entry.
    BoundedRing {
        /// Capacity in transitions.
        capacity: usize,
    },
    /// A reservoir sample with the declared capacity.
    Reservoir {
        /// Capacity in transitions.
        capacity: usize,
    },
    /// An append-only corpus that never evicts.
    AppendOnlyCorpus,
    /// On-policy storage that reports exhaustion instead of overwriting.
    OnPolicyBatch {
        /// Capacity in transitions.
        capacity: usize,
    },
}

impl RetentionPolicy {
    /// Whether this policy may overwrite unconsumed samples.
    #[must_use]
    pub const fn may_overwrite(self) -> bool {
        matches!(self, Self::BoundedRing { .. } | Self::Reservoir { .. })
    }

    /// The declared capacity, where the policy is bounded.
    #[must_use]
    pub const fn capacity(self) -> Option<usize> {
        match self {
            Self::BoundedRing { capacity }
            | Self::Reservoir { capacity }
            | Self::OnPolicyBatch { capacity } => Some(capacity),
            Self::AppendOnlyCorpus => None,
        }
    }

    /// The canonical identifier used in records.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::BoundedRing { .. } => "bounded-ring",
            Self::Reservoir { .. } => "reservoir",
            Self::AppendOnlyCorpus => "append-only-corpus",
            Self::OnPolicyBatch { .. } => "on-policy-batch",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::EnvId;

    fn schema() -> TransitionSchema {
        TransitionSchema {
            schema_version: TRANSITION_SCHEMA_VERSION,
            observation_width: 3,
            action_width: 2,
            action_representation: ActionRepresentation::ActuatorSpace,
            robot_profile: "fixture-arm/1.0".to_owned(),
        }
    }

    fn transition() -> Transition {
        Transition {
            schema_version: TRANSITION_SCHEMA_VERSION,
            episode_id: "ep-1".to_owned(),
            timestep: 0,
            env_id: EnvId::new(0),
            robot_profile: "fixture-arm/1.0".to_owned(),
            observation: vec![0.0, 1.0, 2.0],
            action: vec![0.1, -0.1],
            action_representation: ActionRepresentation::ActuatorSpace,
            reward: RewardTerms::new(),
            next_observation: vec![0.1, 1.1, 2.1],
            termination: Termination::Continuing,
            observation_validity: vec![true; 3],
            task_data: Vec::new(),
            origin: TransitionOrigin::CurrentPolicy {
                generation: Generation(4),
            },
            seeds: SeedStream::new(1, 0),
            behavior_log_prob: Some(-1.2),
        }
    }

    fn latent_space() -> LatentSpaceId {
        LatentSpaceId {
            name: "boxing-latent".to_owned(),
            version: SchemaVersion::new(1, 0),
            decoder_artifact: "sha256:deadbeef".to_owned(),
            dimension: 2,
            constraint: LatentConstraint::UnitHypersphere,
        }
    }

    #[test]
    fn truncation_bootstraps_and_termination_does_not() {
        assert!(Termination::TimeLimit.bootstraps());
        assert!(Termination::Continuing.bootstraps());
        assert!(!Termination::Terminal.bootstraps());
        assert!(Termination::TimeLimit.ends_episode());
        assert!(!Termination::Continuing.ends_episode());
    }

    #[test]
    fn a_well_formed_transition_is_admitted() {
        assert!(schema().accepts(&transition()).is_ok());
    }

    #[test]
    fn a_latent_transition_is_rejected_by_a_joint_space_buffer() {
        let mut t = transition();
        t.action_representation = ActionRepresentation::LatentSpace(latent_space());
        t.action = vec![1.0, 0.0];
        assert!(matches!(
            schema().accepts(&t),
            Err(TransitionError::MixedActionSpaces { .. })
        ));
    }

    #[test]
    fn a_latent_buffer_enforces_the_declared_constraint() {
        let space = latent_space();
        let latent_schema = TransitionSchema {
            action_representation: ActionRepresentation::LatentSpace(space.clone()),
            ..schema()
        };
        let mut t = transition();
        t.action_representation = ActionRepresentation::LatentSpace(space);
        t.action = vec![1.0, 0.0];
        assert!(latent_schema.accepts(&t).is_ok());

        t.action = vec![0.3, 0.3];
        assert!(matches!(
            latent_schema.accepts(&t),
            Err(TransitionError::LatentConstraintViolated { .. })
        ));
    }

    #[test]
    fn two_latent_spaces_of_different_versions_do_not_mix() {
        let a = latent_space();
        let mut b = latent_space();
        b.version = SchemaVersion::new(2, 0);
        assert!(!ActionRepresentation::LatentSpace(a.clone())
            .interchangeable_with(&ActionRepresentation::LatentSpace(b)));
        assert!(ActionRepresentation::LatentSpace(a.clone())
            .interchangeable_with(&ActionRepresentation::LatentSpace(a)));
    }

    #[test]
    fn latent_constraints_check_what_they_claim() {
        assert!(LatentConstraint::UnitHypersphere.admits(&[1.0, 0.0], 1e-9));
        assert!(!LatentConstraint::UnitHypersphere.admits(&[0.5, 0.0], 1e-9));
        assert!(LatentConstraint::Box { bound: 1.0 }.admits(&[1.0, -1.0], 1e-9));
        assert!(!LatentConstraint::Box { bound: 1.0 }.admits(&[1.5, 0.0], 1e-9));
        assert!(LatentConstraint::Ball { radius: 2.0 }.admits(&[1.0, 1.0], 1e-9));
        assert!(!LatentConstraint::Ball { radius: 1.0 }.admits(&[1.0, 1.0], 1e-9));
        assert!(!LatentConstraint::Box { bound: 1.0 }.admits(&[f64::NAN], 1e-9));
    }

    #[test]
    fn a_transition_from_another_profile_is_rejected() {
        let mut t = transition();
        t.robot_profile = "other-robot/2.0".to_owned();
        assert!(matches!(
            schema().accepts(&t),
            Err(TransitionError::ProfileMismatch { .. })
        ));
    }

    #[test]
    fn nonfinite_components_are_rejected() {
        let mut t = transition();
        t.action[1] = f64::INFINITY;
        assert!(matches!(
            schema().accepts(&t),
            Err(TransitionError::NotFinite { .. })
        ));
    }

    #[test]
    fn widths_are_checked_on_every_vector() {
        let mut t = transition();
        t.next_observation.pop();
        assert!(matches!(
            schema().accepts(&t),
            Err(TransitionError::WidthMismatch {
                field: "next_observation",
                ..
            })
        ));
    }

    #[test]
    fn a_newer_transition_schema_is_refused() {
        let mut t = transition();
        t.schema_version = SchemaVersion::new(1, 7);
        assert!(matches!(
            schema().accepts(&t),
            Err(TransitionError::IncompatibleSchema { .. })
        ));
    }

    #[test]
    fn style_terms_are_excluded_from_the_task_total() {
        let mut r = RewardTerms::new();
        r.push(RewardTerm {
            name: "tracking".to_owned(),
            value: 2.0,
            weight: 1.0,
            is_style_term: false,
        })
        .unwrap();
        r.push(RewardTerm {
            name: "amp_style".to_owned(),
            value: 10.0,
            weight: 0.5,
            is_style_term: true,
        })
        .unwrap();
        assert!((r.total() - 7.0).abs() < 1e-12);
        assert!((r.task_total() - 2.0).abs() < 1e-12);
        assert!(r.get("tracking").is_some());
        assert!(r.get("absent").is_none());
    }

    #[test]
    fn duplicate_reward_terms_are_rejected() {
        let mut r = RewardTerms::new();
        let term = RewardTerm {
            name: "tracking".to_owned(),
            value: 1.0,
            weight: 1.0,
            is_style_term: false,
        };
        r.push(term.clone()).unwrap();
        assert!(matches!(
            r.push(term),
            Err(TransitionError::DuplicateRewardTerm { .. })
        ));
    }

    #[test]
    fn origins_distinguish_behavior_targets_from_on_policy_data() {
        assert!(TransitionOrigin::CurrentPolicy {
            generation: Generation(0)
        }
        .is_on_policy());
        let hist = TransitionOrigin::HistoricalPolicy {
            snapshot_id: "snap-9".to_owned(),
            generation: Generation(9),
        };
        assert!(hist.is_behavior_target());
        assert!(!hist.is_on_policy());
        assert_eq!(hist.kind(), "historical-policy");
    }

    #[test]
    fn corpus_origins_carry_their_embodiment() {
        let origin = TransitionOrigin::ExternalCorpus {
            corpus: "amass@rev7".to_owned(),
            source_episode: "cmu/02_05".to_owned(),
            embodiment: "human-skeleton".to_owned(),
            retrieval_generation: Generation(2),
        };
        assert_eq!(origin.embodiment(), Some("human-skeleton"));
        assert!(!origin.is_on_policy());
        assert!(!origin.is_behavior_target());
    }

    #[test]
    fn retention_policies_declare_their_overwrite_behavior() {
        assert!(RetentionPolicy::BoundedRing { capacity: 8 }.may_overwrite());
        assert!(!RetentionPolicy::OnPolicyBatch { capacity: 8 }.may_overwrite());
        assert!(!RetentionPolicy::AppendOnlyCorpus.may_overwrite());
        assert_eq!(
            RetentionPolicy::Reservoir { capacity: 4 }.capacity(),
            Some(4)
        );
        assert_eq!(RetentionPolicy::AppendOnlyCorpus.capacity(), None);
    }
}

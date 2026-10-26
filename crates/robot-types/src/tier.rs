//! Hierarchical tiers and the latency invariant of SPECS.md Section 5.1.
//!
//! "The governing invariant is that **the lower a component sits, the tighter
//! its latency obligation and the smaller its permitted variability**."
//!
//! Three things in Section 5.1 are mechanized here rather than reviewed:
//!
//! * item 1 — a component with no declared tier cannot be activated, so the
//!   declaration is a required field of [`TierAssignment`] and activation goes
//!   through [`TierSchedule::validate`];
//! * item 2 — tier rates are non-increasing with height, and two tiers sharing a
//!   rate is "the degenerate case and MUST be recorded as such", which
//!   [`TierSchedule::degenerate_pairs`] reports;
//! * item 4 — outputs crossing a tier boundary are bounded, validated and
//!   timestamped, and the *consumer* rejects them on its own freshness limit,
//!   which is [`TierOutput::accept`].

use crate::time::{Duration, Timestamp};
use crate::units::Hertz;
use core::fmt;

/// A level of the hierarchy, named by the loop table of SPECS.md Section 5.
///
/// Ordered by *height*: `Servo` is the lowest and tightest, `Advisory` the
/// highest and loosest. The discriminant order is the comparison order.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Tier {
    /// Embedded actuator servo; usually firmware owned.
    Servo,
    /// Host state estimation and command supervision.
    Supervision,
    /// Learned whole-body policy. The lowest *learned* tier.
    Policy,
    /// Tactical router and opponent forecast.
    Tactical,
    /// Semantic search and repertoire planning.
    Planning,
    /// Optional agent advisory, if ever enabled. Out of process, feature-gated,
    /// and never on a control path (SPECS.md Section 2.3 item 2).
    Advisory,
}

impl Tier {
    /// All tiers, lowest first.
    pub const ALL: [Self; 6] = [
        Self::Servo,
        Self::Supervision,
        Self::Policy,
        Self::Tactical,
        Self::Planning,
        Self::Advisory,
    ];

    /// The canonical identifier used in bundles and reports.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::Servo => "servo",
            Self::Supervision => "supervision",
            Self::Policy => "policy",
            Self::Tactical => "tactical",
            Self::Planning => "planning",
            Self::Advisory => "advisory",
        }
    }

    /// Parses a tier identifier.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|t| t.id() == text)
    }

    /// Height above the servo tier.
    #[must_use]
    pub const fn height(self) -> u8 {
        match self {
            Self::Servo => 0,
            Self::Supervision => 1,
            Self::Policy => 2,
            Self::Tactical => 3,
            Self::Planning => 4,
            Self::Advisory => 5,
        }
    }

    /// The initial target rate range from the table in SPECS.md Section 5.
    ///
    /// These are "initial design targets, not measured guarantees"; a robot
    /// profile establishes the actual rates and this range only bounds a
    /// plausible declaration.
    #[must_use]
    pub const fn target_rate_range(self) -> (Hertz, Hertz) {
        match self {
            Self::Servo => (Hertz(500.0), Hertz(1000.0)),
            Self::Supervision => (Hertz(500.0), Hertz(1000.0)),
            Self::Policy => (Hertz(50.0), Hertz(200.0)),
            Self::Tactical => (Hertz(10.0), Hertz(50.0)),
            Self::Planning => (Hertz(1.0), Hertz(10.0)),
            Self::Advisory => (Hertz(0.0), Hertz(1.0)),
        }
    }

    /// True when this tier executes inside the deadline-critical control
    /// process. The advisory tier never does (SPECS.md Section 5).
    #[must_use]
    pub const fn is_deadline_critical(self) -> bool {
        matches!(
            self,
            Self::Servo | Self::Supervision | Self::Policy | Self::Tactical
        )
    }
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

/// What a tier does when the tier above it is late, absent or expired.
///
/// SPECS.md Section 5.1 item 3: "the response to a missing upper-tier decision
/// is the declared hold-or-default behavior of that tier, never an extended
/// deadline below."
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MissingUpperTierBehavior {
    /// Keep applying the last accepted decision until its own expiry, then fall
    /// back to `Default`.
    HoldUntilExpiry,
    /// Immediately use the tier's declared default decision.
    UseDeclaredDefault,
    /// Invoke the robot-specific fallback of SPECS.md Section 10.4.
    InvokeFallback,
}

impl MissingUpperTierBehavior {
    /// The canonical identifier used in bundles.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::HoldUntilExpiry => "hold-until-expiry",
            Self::UseDeclaredDefault => "use-declared-default",
            Self::InvokeFallback => "invoke-fallback",
        }
    }
}

/// A learned component's declared place in the hierarchy.
///
/// Every field is required. SPECS.md Section 5.1 item 1 requires the tier
/// assignment to be recorded "alongside the component's measured compute
/// duration", so `measured_compute` is not an `Option`: a component that has not
/// been measured cannot be declared, and Section 5.1 item 6 says the budget
/// "MUST be measured, not inferred from parameter count".
#[derive(Clone, PartialEq, Debug)]
pub struct TierAssignment {
    /// Stable name of the component.
    pub component: String,
    /// The tier it runs at.
    pub tier: Tier,
    /// Configured rate.
    pub rate: Hertz,
    /// Worst-case compute budget allowed for this component.
    pub budget: Duration,
    /// Measured compute duration at the stated percentile under representative
    /// load, taken from a benchmark run rather than estimated.
    pub measured_compute: MeasuredCompute,
    /// Freshness limit this component applies to inputs from the tier above.
    pub input_freshness_limit: Duration,
    /// What it does when the tier above is late, absent or expired.
    pub on_missing_upper: MissingUpperTierBehavior,
}

/// A measured compute duration with the percentile and run it came from.
#[derive(Clone, PartialEq, Debug)]
pub struct MeasuredCompute {
    /// The percentile reported, e.g. 99.9.
    pub percentile: f64,
    /// The measured duration at that percentile.
    pub duration: Duration,
    /// Worst observed duration in the same run.
    pub max: Duration,
    /// Identifier of the benchmark run that produced these numbers.
    pub run_id: String,
}

/// Why a tier declaration or schedule was rejected.
#[derive(Clone, PartialEq, Debug)]
pub enum TierError {
    /// A component declared no tier at all.
    ///
    /// SPECS.md Section 5.1 item 1: such a component "MUST NOT be activated".
    NoDeclaredTier {
        /// The component.
        component: String,
    },
    /// Tier rates were not non-increasing with height.
    RateIncreasesWithHeight {
        /// The lower tier.
        lower: Tier,
        /// Its rate.
        lower_rate: Hertz,
        /// The higher tier.
        higher: Tier,
        /// Its rate.
        higher_rate: Hertz,
    },
    /// The lowest learned tier ran slower than the policy rate.
    LowestLearnedTierTooSlow {
        /// The tier.
        tier: Tier,
        /// Its rate.
        rate: Hertz,
        /// The required policy rate.
        policy_rate: Hertz,
    },
    /// The measured compute exceeded the declared budget.
    BudgetExceeded {
        /// The component.
        component: String,
        /// The budget.
        budget: Duration,
        /// What was measured.
        measured: Duration,
    },
    /// The declared budget exceeded the tier's own period.
    BudgetExceedsPeriod {
        /// The component.
        component: String,
        /// The budget.
        budget: Duration,
        /// The period implied by the rate.
        period: Duration,
    },
    /// The rate was not a usable positive, finite value.
    InvalidRate {
        /// The component.
        component: String,
        /// The rate supplied.
        rate: Hertz,
    },
    /// A measurement had no run identifier, so it cannot be traced.
    UnattributedMeasurement {
        /// The component.
        component: String,
    },
    /// A mechanism qualified at one tier appeared at a lower one.
    ///
    /// SPECS.md Section 5.1 item 5: "Pushing an expensive mechanism downward is
    /// a change to Section 5 and requires the change-control record."
    MechanismPushedDownward {
        /// The mechanism.
        mechanism: String,
        /// Tier it was qualified at.
        qualified_at: Tier,
        /// Tier it was found at.
        found_at: Tier,
    },
}

impl fmt::Display for TierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoDeclaredTier { component } => {
                write!(
                    f,
                    "component {component:?} declares no tier and cannot be activated"
                )
            }
            Self::RateIncreasesWithHeight {
                lower,
                lower_rate,
                higher,
                higher_rate,
            } => write!(
                f,
                "tier {higher} runs at {higher_rate} above {lower} at {lower_rate}: \
                 rates must be non-increasing with height"
            ),
            Self::LowestLearnedTierTooSlow {
                tier,
                rate,
                policy_rate,
            } => write!(
                f,
                "lowest learned tier {tier} runs at {rate}, below the policy rate {policy_rate}"
            ),
            Self::BudgetExceeded {
                component,
                budget,
                measured,
            } => write!(
                f,
                "component {component:?} measured {measured} against a budget of {budget}"
            ),
            Self::BudgetExceedsPeriod {
                component,
                budget,
                period,
            } => write!(
                f,
                "component {component:?} has budget {budget} exceeding its period {period}"
            ),
            Self::InvalidRate { component, rate } => {
                write!(
                    f,
                    "component {component:?} declares an unusable rate {rate}"
                )
            }
            Self::UnattributedMeasurement { component } => {
                write!(
                    f,
                    "component {component:?} has a measurement with no run identifier"
                )
            }
            Self::MechanismPushedDownward {
                mechanism,
                qualified_at,
                found_at,
            } => write!(
                f,
                "mechanism {mechanism:?} was qualified at {qualified_at} but appears at \
                 {found_at}: this is a Section 5 change requiring a change-control record"
            ),
        }
    }
}

impl core::error::Error for TierError {}

impl TierAssignment {
    /// Checks one declaration in isolation.
    pub fn validate(&self) -> Result<(), TierError> {
        if self.component.trim().is_empty() {
            return Err(TierError::NoDeclaredTier {
                component: self.component.clone(),
            });
        }
        if self.measured_compute.run_id.trim().is_empty() {
            return Err(TierError::UnattributedMeasurement {
                component: self.component.clone(),
            });
        }
        let period = self.rate.period().ok_or_else(|| TierError::InvalidRate {
            component: self.component.clone(),
            rate: self.rate,
        })?;
        if self.budget > period {
            return Err(TierError::BudgetExceedsPeriod {
                component: self.component.clone(),
                budget: self.budget,
                period,
            });
        }
        if self.measured_compute.duration > self.budget {
            return Err(TierError::BudgetExceeded {
                component: self.component.clone(),
                budget: self.budget,
                measured: self.measured_compute.duration,
            });
        }
        Ok(())
    }

    /// Deadline margin: budget minus the measured duration at the reported
    /// percentile. Reported separately from decision age (SPECS.md Section 5).
    #[must_use]
    pub fn deadline_margin(&self) -> Duration {
        self.budget - self.measured_compute.duration
    }
}

/// The full set of tier assignments for one deployed stack.
#[derive(Clone, Default, PartialEq, Debug)]
pub struct TierSchedule {
    assignments: Vec<TierAssignment>,
}

impl TierSchedule {
    /// An empty schedule.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a component's declaration.
    pub fn declare(&mut self, assignment: TierAssignment) {
        self.assignments.push(assignment);
    }

    /// The declarations, in insertion order.
    #[must_use]
    pub fn assignments(&self) -> &[TierAssignment] {
        &self.assignments
    }

    /// The declaration for a named component, if any.
    #[must_use]
    pub fn component(&self, name: &str) -> Option<&TierAssignment> {
        self.assignments.iter().find(|a| a.component == name)
    }

    /// Checks the whole schedule against SPECS.md Section 5.1.
    ///
    /// `policy_rate` is the rate the deployed policy tier must meet or exceed,
    /// taken from the robot's acceptance profile rather than assumed.
    pub fn validate(&self, policy_rate: Hertz) -> Result<(), TierError> {
        for a in &self.assignments {
            a.validate()?;
        }

        // Item 2: rates are non-increasing with height. Compare the slowest
        // configured rate at each tier, because a tier with several components
        // is as fast as its slowest member.
        let mut per_tier: Vec<(Tier, Hertz)> = Vec::new();
        for a in &self.assignments {
            match per_tier.iter_mut().find(|(t, _)| *t == a.tier) {
                Some((_, r)) => *r = r.min(a.rate),
                None => per_tier.push((a.tier, a.rate)),
            }
        }
        per_tier.sort_by_key(|(t, _)| t.height());
        for w in per_tier.windows(2) {
            let (lower, lower_rate) = w[0];
            let (higher, higher_rate) = w[1];
            if higher_rate.value() > lower_rate.value() {
                return Err(TierError::RateIncreasesWithHeight {
                    lower,
                    lower_rate,
                    higher,
                    higher_rate,
                });
            }
        }

        // Item 2: the lowest learned tier runs at the policy rate or faster.
        if let Some(&(tier, rate)) = per_tier
            .iter()
            .find(|(t, _)| matches!(t, Tier::Policy | Tier::Tactical | Tier::Planning))
        {
            if rate.value() < policy_rate.value() {
                return Err(TierError::LowestLearnedTierTooSlow {
                    tier,
                    rate,
                    policy_rate,
                });
            }
        }

        Ok(())
    }

    /// Pairs of tiers that share a rate.
    ///
    /// SPECS.md Section 5.1 item 2: "Two tiers MAY share a rate, which is the
    /// degenerate case and MUST be recorded as such rather than described as a
    /// hierarchy." A report calls this out; it is not an error.
    #[must_use]
    pub fn degenerate_pairs(&self) -> Vec<(Tier, Tier, Hertz)> {
        let mut out = Vec::new();
        for (i, a) in self.assignments.iter().enumerate() {
            for b in &self.assignments[i + 1..] {
                if a.tier != b.tier && (a.rate.value() - b.rate.value()).abs() < f64::EPSILON {
                    let (lo, hi) = if a.tier.height() < b.tier.height() {
                        (a.tier, b.tier)
                    } else {
                        (b.tier, a.tier)
                    };
                    if !out.iter().any(|(l, h, _)| *l == lo && *h == hi) {
                        out.push((lo, hi, a.rate));
                    }
                }
            }
        }
        out
    }

    /// True when the schedule describes a genuine hierarchy rather than the
    /// degenerate single-rate case.
    #[must_use]
    pub fn is_frequency_separated(&self) -> bool {
        self.degenerate_pairs().is_empty()
    }

    /// The highest tier present, whose removal must leave a safe controller
    /// (SPECS.md Section 5.1 item 7).
    #[must_use]
    pub fn highest_tier(&self) -> Option<Tier> {
        self.assignments.iter().map(|a| a.tier).max()
    }

    /// The schedule with every component at the highest tier removed, which is
    /// the configuration the removal test of item 7 must still find safe.
    #[must_use]
    pub fn without_highest_tier(&self) -> Self {
        match self.highest_tier() {
            None => Self::new(),
            Some(top) => Self {
                assignments: self
                    .assignments
                    .iter()
                    .filter(|a| a.tier != top)
                    .cloned()
                    .collect(),
            },
        }
    }
}

/// A monotonically increasing tag identifying one production of a tier output.
///
/// SPECS.md Section 10.3 requires switching "using generation-tagged immutable
/// handles"; the same tag is what lets a consumer tell a re-delivered stale
/// proposal from a new one.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Generation(pub u64);

impl Generation {
    /// The first generation.
    pub const FIRST: Self = Self(0);

    /// The next generation.
    #[must_use]
    pub const fn next(self) -> Self {
        Self(self.0.wrapping_add(1))
    }
}

impl fmt::Display for Generation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "gen{}", self.0)
    }
}

/// Why a consumer rejected a tier output.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CrossingRejection {
    /// The output was older than the consumer's freshness limit.
    Expired {
        /// Age at the moment of evaluation.
        age: Duration,
        /// The consumer's limit.
        limit: Duration,
    },
    /// The output's generation was not newer than the one already applied.
    StaleGeneration {
        /// Generation offered.
        offered: Generation,
        /// Generation already applied.
        applied: Generation,
    },
    /// The output and the consumer disagree about the clock.
    ClockMismatch,
    /// The producing tier was not above the consuming tier.
    NotFromAbove {
        /// Producer tier.
        producer: Tier,
        /// Consumer tier.
        consumer: Tier,
    },
}

impl fmt::Display for CrossingRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Expired { age, limit } => write!(f, "expired: age {age} exceeds limit {limit}"),
            Self::StaleGeneration { offered, applied } => {
                write!(f, "stale: offered {offered}, already applied {applied}")
            }
            Self::ClockMismatch => write!(f, "produced on a different clock"),
            Self::NotFromAbove { producer, consumer } => {
                write!(f, "{producer} is not above {consumer}")
            }
        }
    }
}

/// A bounded, validated, timestamped output crossing a tier boundary.
///
/// SPECS.md Section 5.1 item 4. The payload is owned by the caller and is
/// required to be bounded in size by its own type; this wrapper carries the
/// provenance the consumer needs in order to reject it.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct TierOutput<T> {
    /// The decision itself.
    pub payload: T,
    /// Tier that produced it. Decision age is attributed here, not to the tier
    /// that applied it (SPECS.md Section 5.1 item 4).
    pub producer: Tier,
    /// When it was produced.
    pub produced_at: Timestamp,
    /// Its generation.
    pub generation: Generation,
}

impl<T> TierOutput<T> {
    /// Builds an output.
    #[must_use]
    pub const fn new(
        payload: T,
        producer: Tier,
        produced_at: Timestamp,
        generation: Generation,
    ) -> Self {
        Self {
            payload,
            producer,
            produced_at,
            generation,
        }
    }

    /// Decision age at `now`, as defined in the timing table of Section 5.
    pub fn decision_age(&self, now: Timestamp) -> Result<Duration, crate::time::TimeError> {
        now.since(self.produced_at).map(|(age, _)| age)
    }

    /// Evaluates this output against a consumer's own freshness limit.
    ///
    /// The *consumer* decides, per Section 5.1 item 4. `applied` is the
    /// generation the consumer is currently acting on.
    pub fn accept(
        &self,
        consumer: Tier,
        now: Timestamp,
        limit: Duration,
        applied: Generation,
    ) -> Result<Duration, CrossingRejection> {
        if self.producer.height() <= consumer.height() {
            return Err(CrossingRejection::NotFromAbove {
                producer: self.producer,
                consumer,
            });
        }
        let age = self
            .decision_age(now)
            .map_err(|_| CrossingRejection::ClockMismatch)?;
        if age > limit {
            return Err(CrossingRejection::Expired { age, limit });
        }
        if self.generation <= applied {
            return Err(CrossingRejection::StaleGeneration {
                offered: self.generation,
                applied,
            });
        }
        Ok(age)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::ClockId;

    fn assignment(component: &str, tier: Tier, rate_hz: f64, measured_us: i64) -> TierAssignment {
        let period = Hertz(rate_hz).period().unwrap();
        TierAssignment {
            component: component.to_owned(),
            tier,
            rate: Hertz(rate_hz),
            budget: period,
            measured_compute: MeasuredCompute {
                percentile: 99.9,
                duration: Duration::from_micros(measured_us),
                max: Duration::from_micros(measured_us * 2),
                run_id: "bench-001".to_owned(),
            },
            input_freshness_limit: Duration::from_millis(100),
            on_missing_upper: MissingUpperTierBehavior::UseDeclaredDefault,
        }
    }

    #[test]
    fn a_component_without_a_run_id_cannot_be_declared() {
        let mut a = assignment("decoder", Tier::Policy, 100.0, 1000);
        a.measured_compute.run_id = String::new();
        assert!(matches!(
            a.validate(),
            Err(TierError::UnattributedMeasurement { .. })
        ));
    }

    #[test]
    fn measured_compute_above_budget_is_rejected() {
        let mut a = assignment("decoder", Tier::Policy, 100.0, 1000);
        a.budget = Duration::from_micros(500);
        assert!(matches!(
            a.validate(),
            Err(TierError::BudgetExceeded { .. })
        ));
    }

    #[test]
    fn budget_above_the_period_is_rejected() {
        let mut a = assignment("decoder", Tier::Policy, 100.0, 1000);
        a.budget = Duration::from_millis(20);
        assert!(matches!(
            a.validate(),
            Err(TierError::BudgetExceedsPeriod { .. })
        ));
    }

    #[test]
    fn zero_rate_is_rejected() {
        let mut a = assignment("decoder", Tier::Policy, 100.0, 1000);
        a.rate = Hertz(0.0);
        assert!(matches!(a.validate(), Err(TierError::InvalidRate { .. })));
    }

    #[test]
    fn a_faster_upper_tier_is_rejected() {
        let mut s = TierSchedule::new();
        s.declare(assignment("decoder", Tier::Policy, 100.0, 1000));
        s.declare(assignment("strategy", Tier::Tactical, 200.0, 500));
        assert!(matches!(
            s.validate(Hertz(100.0)),
            Err(TierError::RateIncreasesWithHeight { .. })
        ));
    }

    #[test]
    fn a_well_formed_two_tier_stack_validates() {
        let mut s = TierSchedule::new();
        s.declare(assignment("decoder", Tier::Policy, 100.0, 1000));
        s.declare(assignment("strategy", Tier::Tactical, 25.0, 5000));
        assert!(s.validate(Hertz(100.0)).is_ok());
        assert!(s.is_frequency_separated());
        assert_eq!(s.highest_tier(), Some(Tier::Tactical));
        assert_eq!(s.without_highest_tier().assignments().len(), 1);
        assert_eq!(s.component("decoder").map(|a| a.tier), Some(Tier::Policy));
    }

    #[test]
    fn a_single_rate_experiment_is_reported_as_degenerate() {
        // RoboStriker runs strategy and decoder both at 50 Hz; SPECS.md
        // Section 5.1 item 2 requires that to be recorded as the degenerate
        // case rather than described as a hierarchy.
        let mut s = TierSchedule::new();
        s.declare(assignment("decoder", Tier::Policy, 50.0, 1000));
        s.declare(assignment("strategy", Tier::Tactical, 50.0, 1000));
        assert!(s.validate(Hertz(50.0)).is_ok());
        assert!(!s.is_frequency_separated());
        assert_eq!(
            s.degenerate_pairs(),
            vec![(Tier::Policy, Tier::Tactical, Hertz(50.0))]
        );
    }

    #[test]
    fn a_policy_tier_slower_than_the_profile_rate_is_rejected() {
        let mut s = TierSchedule::new();
        s.declare(assignment("decoder", Tier::Policy, 40.0, 1000));
        assert!(matches!(
            s.validate(Hertz(100.0)),
            Err(TierError::LowestLearnedTierTooSlow { .. })
        ));
    }

    #[test]
    fn deadline_margin_is_budget_minus_measurement() {
        let a = assignment("decoder", Tier::Policy, 100.0, 1000);
        assert_eq!(a.deadline_margin(), Duration::from_micros(9000));
    }

    #[test]
    fn an_expired_proposal_is_dropped_by_the_consumer() {
        let clock = ClockId::HostMonotonic;
        let out = TierOutput::new(
            7u32,
            Tier::Tactical,
            Timestamp::sampling(0, clock),
            Generation(1),
        );
        let now = Timestamp::sampling(Duration::from_millis(200).nanos(), clock);
        assert!(matches!(
            out.accept(
                Tier::Policy,
                now,
                Duration::from_millis(100),
                Generation::FIRST
            ),
            Err(CrossingRejection::Expired { .. })
        ));
        let fresh_now = Timestamp::sampling(Duration::from_millis(50).nanos(), clock);
        assert_eq!(
            out.accept(
                Tier::Policy,
                fresh_now,
                Duration::from_millis(100),
                Generation::FIRST
            ),
            Ok(Duration::from_millis(50))
        );
    }

    #[test]
    fn a_replayed_generation_is_rejected() {
        let clock = ClockId::HostMonotonic;
        let out = TierOutput::new(
            7u32,
            Tier::Tactical,
            Timestamp::sampling(0, clock),
            Generation(3),
        );
        let now = Timestamp::sampling(1_000, clock);
        assert!(matches!(
            out.accept(Tier::Policy, now, Duration::from_millis(100), Generation(3)),
            Err(CrossingRejection::StaleGeneration { .. })
        ));
    }

    #[test]
    fn an_output_from_below_is_not_a_tier_crossing() {
        let clock = ClockId::HostMonotonic;
        let out = TierOutput::new(
            7u32,
            Tier::Policy,
            Timestamp::sampling(0, clock),
            Generation(1),
        );
        assert!(matches!(
            out.accept(
                Tier::Tactical,
                Timestamp::sampling(1, clock),
                Duration::from_millis(1),
                Generation::FIRST
            ),
            Err(CrossingRejection::NotFromAbove { .. })
        ));
    }

    #[test]
    fn a_mismatched_clock_is_rejected_rather_than_subtracted() {
        let out = TierOutput::new(
            7u32,
            Tier::Tactical,
            Timestamp::sampling(0, ClockId::Device(1)),
            Generation(1),
        );
        assert_eq!(
            out.accept(
                Tier::Policy,
                Timestamp::sampling(1_000, ClockId::HostMonotonic),
                Duration::from_millis(100),
                Generation::FIRST
            ),
            Err(CrossingRejection::ClockMismatch)
        );
    }

    #[test]
    fn tiers_are_ordered_by_height_and_the_advisory_tier_has_no_deadline() {
        assert!(Tier::Servo < Tier::Policy);
        assert!(Tier::Policy < Tier::Advisory);
        assert!(Tier::Policy.is_deadline_critical());
        assert!(!Tier::Advisory.is_deadline_critical());
        assert_eq!(Tier::parse("tactical"), Some(Tier::Tactical));
        assert_eq!(Tier::parse("strategy"), None);
    }
}

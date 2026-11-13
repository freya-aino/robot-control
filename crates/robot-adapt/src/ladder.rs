//! The ordering rule of SPECS.md Section 11.3, enforced rather than
//! described: "a method MUST NOT be scheduled before the methods above it
//! have been measured and found insufficient on the same robot."
//!
//! An [`AdaptationLadder`] holds measurements per robot. Scheduling a
//! method succeeds only when every method above it in [`Method::ORDER`] has
//! a recorded, evidence-backed `insufficient` measurement on that robot.
//! Two methods carry an extra gate the table itself states:
//!
//! - [`Method::SimulationGuidedFineTuning`] updates weights on hardware,
//!   which obligation 7 permits only under Section 4's mode gating *plus a
//!   separate decision*; scheduling it needs a [`HardwareUpdateDecision`].
//! - [`Method::MultiSimulatorRandomization`] needs a second qualified
//!   simulator, which Section 9.1 does not grant; it is refused outright
//!   until a decision record grants one.

use core::fmt;

/// The rows of SPECS.md Section 11.3, in their binding order.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Method {
    /// Row 1.
    SystemIdentification,
    /// Row 2.
    HistoryConditioned,
    /// Row 3.
    SimulatorCorrection,
    /// Row 4.
    BoundedResidual,
    /// Row 5.
    SmallAdapterFineTuning,
    /// Row 6.
    TwoPhaseLatent,
    /// Row 7.
    SimulationGuidedFineTuning,
    /// Row 8.
    RetrievalConditioned,
    /// Row 9.
    CrossEmbodiment,
    /// Row 10.
    MultiSimulatorRandomization,
}

impl Method {
    /// Every method, by increasing hardware exposure.
    pub const ORDER: [Self; 10] = [
        Self::SystemIdentification,
        Self::HistoryConditioned,
        Self::SimulatorCorrection,
        Self::BoundedResidual,
        Self::SmallAdapterFineTuning,
        Self::TwoPhaseLatent,
        Self::SimulationGuidedFineTuning,
        Self::RetrievalConditioned,
        Self::CrossEmbodiment,
        Self::MultiSimulatorRandomization,
    ];

    /// Position in [`Method::ORDER`].
    #[must_use]
    pub fn rank(self) -> usize {
        Self::ORDER.iter().position(|m| *m == self).expect("listed")
    }

    /// Stable name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::SystemIdentification => "system-identification",
            Self::HistoryConditioned => "history-conditioned",
            Self::SimulatorCorrection => "simulator-correction",
            Self::BoundedResidual => "bounded-residual",
            Self::SmallAdapterFineTuning => "small-adapter-fine-tuning",
            Self::TwoPhaseLatent => "two-phase-latent",
            Self::SimulationGuidedFineTuning => "simulation-guided-fine-tuning",
            Self::RetrievalConditioned => "retrieval-conditioned",
            Self::CrossEmbodiment => "cross-embodiment",
            Self::MultiSimulatorRandomization => "multi-simulator-randomization",
        }
    }
}

/// A measured outcome of one method on one robot.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Measurement {
    /// The method.
    pub method: Method,
    /// The robot (profile name and version).
    pub robot: String,
    /// Whether the method was found sufficient.
    pub sufficient: bool,
    /// Where the evidence is (a run directory, a report content ID).
    pub evidence: String,
}

/// The separate decision obligation 7 requires before on-hardware weight
/// updates.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct HardwareUpdateDecision {
    /// The decision record (an ADR path or ID).
    pub record: String,
}

/// Why a method may not be scheduled.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum LadderError {
    /// A method above has no insufficiency measurement on this robot.
    NotMeasuredInsufficient {
        /// The blocking method.
        blocking: Method,
    },
    /// A measurement had no evidence reference.
    NoEvidence,
    /// On-hardware updates need a separate decision.
    NeedsHardwareDecision,
    /// Section 9.1 grants no second simulator.
    NoSecondSimulator,
    /// A method already found sufficient makes the later one unneeded.
    AlreadySufficient {
        /// The sufficient method.
        by: Method,
    },
}

impl fmt::Display for LadderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotMeasuredInsufficient { blocking } => write!(
                f,
                "{} has not been measured and found insufficient on this robot",
                blocking.name()
            ),
            Self::NoEvidence => f.write_str("a measurement needs an evidence reference"),
            Self::NeedsHardwareDecision => f.write_str(
                "on-hardware weight updates need a separate decision record (SPECS.md 11.3 item 7)",
            ),
            Self::NoSecondSimulator => f.write_str(
                "multi-simulator randomization needs a second qualified simulator (SPECS.md 9.1)",
            ),
            Self::AlreadySufficient { by } => {
                write!(f, "{} was already found sufficient", by.name())
            }
        }
    }
}

impl core::error::Error for LadderError {}

/// Measurements, and the gate over them.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct AdaptationLadder {
    measurements: Vec<Measurement>,
}

impl AdaptationLadder {
    /// An empty ladder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a measurement. A later measurement of the same method on the
    /// same robot supersedes an earlier one.
    pub fn record(&mut self, m: Measurement) -> Result<(), LadderError> {
        if m.evidence.trim().is_empty() {
            return Err(LadderError::NoEvidence);
        }
        self.measurements
            .retain(|x| !(x.method == m.method && x.robot == m.robot));
        self.measurements.push(m);
        Ok(())
    }

    /// Whether `method` may be scheduled on `robot`.
    pub fn may_schedule(
        &self,
        method: Method,
        robot: &str,
        decision: Option<&HardwareUpdateDecision>,
    ) -> Result<(), LadderError> {
        if method == Method::MultiSimulatorRandomization {
            return Err(LadderError::NoSecondSimulator);
        }
        for above in &Method::ORDER[..method.rank()] {
            match self
                .measurements
                .iter()
                .find(|m| m.method == *above && m.robot == robot)
            {
                Some(m) if m.sufficient => {
                    return Err(LadderError::AlreadySufficient { by: *above })
                }
                Some(_) => {}
                None => return Err(LadderError::NotMeasuredInsufficient { blocking: *above }),
            }
        }
        if method == Method::SimulationGuidedFineTuning
            && decision.is_none_or(|d| d.record.trim().is_empty())
        {
            return Err(LadderError::NeedsHardwareDecision);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn insufficient(method: Method) -> Measurement {
        Measurement {
            method,
            robot: "g1@1.0".into(),
            sufficient: false,
            evidence: "runs/2026-11-13/report.toml".into(),
        }
    }

    #[test]
    fn system_identification_is_always_schedulable_and_the_rest_wait() {
        let ladder = AdaptationLadder::new();
        assert!(ladder
            .may_schedule(Method::SystemIdentification, "g1@1.0", None)
            .is_ok());
        assert_eq!(
            ladder.may_schedule(Method::HistoryConditioned, "g1@1.0", None),
            Err(LadderError::NotMeasuredInsufficient {
                blocking: Method::SystemIdentification
            })
        );
    }

    #[test]
    fn insufficiency_unlocks_the_next_row_only_on_the_same_robot() {
        let mut ladder = AdaptationLadder::new();
        ladder
            .record(insufficient(Method::SystemIdentification))
            .unwrap();
        assert!(ladder
            .may_schedule(Method::HistoryConditioned, "g1@1.0", None)
            .is_ok());
        assert!(ladder
            .may_schedule(Method::HistoryConditioned, "other@1.0", None)
            .is_err());
        assert!(ladder
            .may_schedule(Method::SimulatorCorrection, "g1@1.0", None)
            .is_err());
    }

    #[test]
    fn a_sufficient_method_closes_the_ladder_below_it() {
        let mut ladder = AdaptationLadder::new();
        ladder
            .record(Measurement {
                sufficient: true,
                ..insufficient(Method::SystemIdentification)
            })
            .unwrap();
        assert_eq!(
            ladder.may_schedule(Method::HistoryConditioned, "g1@1.0", None),
            Err(LadderError::AlreadySufficient {
                by: Method::SystemIdentification
            })
        );
    }

    #[test]
    fn hardware_updates_need_a_decision_and_a_second_simulator_is_never_granted() {
        let mut ladder = AdaptationLadder::new();
        for m in &Method::ORDER[..6] {
            ladder.record(insufficient(*m)).unwrap();
        }
        assert_eq!(
            ladder.may_schedule(Method::SimulationGuidedFineTuning, "g1@1.0", None),
            Err(LadderError::NeedsHardwareDecision)
        );
        let decision = HardwareUpdateDecision {
            record: "docs/decisions/00xx".into(),
        };
        assert!(ladder
            .may_schedule(
                Method::SimulationGuidedFineTuning,
                "g1@1.0",
                Some(&decision)
            )
            .is_ok());
        assert_eq!(
            ladder.may_schedule(
                Method::MultiSimulatorRandomization,
                "g1@1.0",
                Some(&decision)
            ),
            Err(LadderError::NoSecondSimulator)
        );
    }

    #[test]
    fn a_measurement_without_evidence_is_refused() {
        let mut ladder = AdaptationLadder::new();
        let mut m = insufficient(Method::SystemIdentification);
        m.evidence = " ".into();
        assert_eq!(ladder.record(m), Err(LadderError::NoEvidence));
    }
}

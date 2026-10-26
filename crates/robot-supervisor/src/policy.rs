//! The declared response for each fault category, and the rule that keeps
//! "clamp and continue" from being used where it is not sufficient.
//!
//! SPECS.md Section 12.1: "Clamp-and-continue is not sufficient for every
//! fault; each fault has a validated response appropriate to the robot."
//! That "validated" is load-bearing: [`FaultPolicy::new`] refuses to
//! construct a policy that assigns [`Response::ClampAndContinue`] to a fault
//! category SPECS.md's own hazard register (§12.2) names as requiring
//! escalation — a stale observation or a lost link, matching "Uncommanded
//! motion" and "Loss of transport" in that table, whose controls are a
//! watchdog and firmware-side command expiry, never a clamp.

use crate::fault::FaultCategory;
use core::fmt;

/// What the supervisor does in response to a fault category.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Response {
    /// Clamp the offending value to its limit and continue applying the
    /// (clamped) command this tick.
    ClampAndContinue,
    /// Discard this tick's command and hold the last command known to have
    /// been safe.
    HoldLastSafeCommand,
    /// Reject this tick's command outright; the actuator interface receives
    /// nothing from this cycle.
    RejectCommand,
    /// Trigger the robot's emergency stop.
    EmergencyStop,
}

impl Response {
    /// The canonical identifier used in records.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::ClampAndContinue => "clamp-and-continue",
            Self::HoldLastSafeCommand => "hold-last-safe-command",
            Self::RejectCommand => "reject-command",
            Self::EmergencyStop => "emergency-stop",
        }
    }
}

impl fmt::Display for Response {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

/// Why a fault policy was rejected.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum PolicyError {
    /// A category SPECS.md §12.2 requires escalation for was assigned
    /// `ClampAndContinue`.
    ClampInsufficientForCategory {
        /// The category.
        category: FaultCategory,
    },
    /// A category had no declared response.
    MissingCategory {
        /// The category.
        category: FaultCategory,
    },
}

impl fmt::Display for PolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ClampInsufficientForCategory { category } => write!(
                f,
                "{category:?} cannot be assigned clamp-and-continue: SPECS.md §12.2 requires \
                 escalation for this category"
            ),
            Self::MissingCategory { category } => {
                write!(f, "no response declared for {category:?}")
            }
        }
    }
}

impl core::error::Error for PolicyError {}

/// Categories for which SPECS.md §12.2's hazard register names a control
/// other than clamping: "Uncommanded motion" (stale command / watchdog) and
/// "Loss of transport" (firmware-side command expiry) are both escalation
/// hazards, and a non-finite command is a computation failure a clamp cannot
/// make sense of (there is no direction to clamp toward).
const MUST_ESCALATE: [FaultCategory; 4] = [
    FaultCategory::StaleObservation,
    FaultCategory::CommunicationsLost,
    FaultCategory::NonFiniteCommand,
    FaultCategory::ActuationNotPermitted,
];

/// All fault categories, used to check a policy is total.
const ALL_CATEGORIES: [FaultCategory; 9] = [
    FaultCategory::StaleObservation,
    FaultCategory::NonFiniteCommand,
    FaultCategory::PositionLimitExceeded,
    FaultCategory::VelocityLimitExceeded,
    FaultCategory::TorqueLimitExceeded,
    FaultCategory::OverTemperature,
    FaultCategory::CommunicationsLost,
    FaultCategory::UnauthorizedContact,
    FaultCategory::ActuationNotPermitted,
];

/// A validated, total mapping from fault category to response.
///
/// "Appropriate to the robot" (§12.1) is why this is a value the caller
/// supplies per robot profile rather than a hard-coded table: a fixture arm
/// without a fall hazard and a free-standing humanoid legitimately choose
/// different responses to the same category, but neither may choose
/// `ClampAndContinue` for a category in [`MUST_ESCALATE`].
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct FaultPolicy {
    responses: [Response; 9],
}

impl FaultPolicy {
    /// A conservative default: every category escalates to
    /// [`Response::EmergencyStop`] except the three bounded-value categories
    /// (position/velocity/torque), which clamp.
    #[must_use]
    pub fn conservative_default() -> Self {
        Self::new(|category| match category {
            FaultCategory::PositionLimitExceeded
            | FaultCategory::VelocityLimitExceeded
            | FaultCategory::TorqueLimitExceeded => Response::ClampAndContinue,
            FaultCategory::OverTemperature | FaultCategory::UnauthorizedContact => {
                Response::HoldLastSafeCommand
            }
            FaultCategory::StaleObservation
            | FaultCategory::NonFiniteCommand
            | FaultCategory::CommunicationsLost
            | FaultCategory::ActuationNotPermitted => Response::EmergencyStop,
        })
        .expect("conservative default satisfies the escalation rule by construction")
    }

    /// Builds a policy from a total function, rejecting an assignment that
    /// violates the escalation rule.
    pub fn new(mut assign: impl FnMut(FaultCategory) -> Response) -> Result<Self, PolicyError> {
        let mut responses = [Response::EmergencyStop; 9];
        for (slot, category) in responses.iter_mut().zip(ALL_CATEGORIES) {
            let response = assign(category);
            if response == Response::ClampAndContinue && MUST_ESCALATE.contains(&category) {
                return Err(PolicyError::ClampInsufficientForCategory { category });
            }
            *slot = response;
        }
        Ok(Self { responses })
    }

    /// The declared response for a category.
    #[must_use]
    pub fn response_for(&self, category: FaultCategory) -> Response {
        let index = ALL_CATEGORIES
            .iter()
            .position(|c| *c == category)
            .expect("ALL_CATEGORIES is exhaustive");
        self.responses[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_conservative_default_satisfies_the_escalation_rule() {
        let policy = FaultPolicy::conservative_default();
        assert_eq!(
            policy.response_for(FaultCategory::StaleObservation),
            Response::EmergencyStop
        );
        assert_eq!(
            policy.response_for(FaultCategory::CommunicationsLost),
            Response::EmergencyStop
        );
        assert_eq!(
            policy.response_for(FaultCategory::PositionLimitExceeded),
            Response::ClampAndContinue
        );
    }

    #[test]
    fn clamping_a_stale_observation_is_refused_at_construction() {
        let built = FaultPolicy::new(|category| match category {
            FaultCategory::StaleObservation => Response::ClampAndContinue,
            _ => Response::EmergencyStop,
        });
        assert_eq!(
            built,
            Err(PolicyError::ClampInsufficientForCategory {
                category: FaultCategory::StaleObservation
            })
        );
    }

    #[test]
    fn clamping_communications_loss_or_a_nonfinite_command_is_also_refused() {
        for bad in [
            FaultCategory::CommunicationsLost,
            FaultCategory::NonFiniteCommand,
        ] {
            let built = FaultPolicy::new(move |category| {
                if category == bad {
                    Response::ClampAndContinue
                } else {
                    Response::EmergencyStop
                }
            });
            assert!(built.is_err(), "{bad:?} must refuse clamp-and-continue");
        }
    }

    #[test]
    fn a_fully_custom_policy_may_choose_any_nonclamp_response_freely() {
        let policy = FaultPolicy::new(|category| match category {
            FaultCategory::OverTemperature => Response::RejectCommand,
            FaultCategory::UnauthorizedContact => Response::HoldLastSafeCommand,
            FaultCategory::PositionLimitExceeded
            | FaultCategory::VelocityLimitExceeded
            | FaultCategory::TorqueLimitExceeded => Response::ClampAndContinue,
            _ => Response::EmergencyStop,
        })
        .unwrap();
        assert_eq!(
            policy.response_for(FaultCategory::OverTemperature),
            Response::RejectCommand
        );
        assert_eq!(
            policy.response_for(FaultCategory::UnauthorizedContact),
            Response::HoldLastSafeCommand
        );
    }
}

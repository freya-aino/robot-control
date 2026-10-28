//! Faults the supervisor can detect, and the bounded, allocation-free log
//! that records them.
//!
//! SPECS.md Section 12.1: "its rejection decisions MUST be recorded with
//! enough context to reconstruct why a command was refused." Every
//! [`Fault`] variant below carries the measured value and the limit it
//! violated, not just a category name.
//!
//! SPECS.md Section 5: "Log overflow must never block actuation; emit
//! dropped-record counts." [`FaultLog`] is a fixed-capacity array rather than
//! a growable `Vec`, so recording a fault inside the control tick allocates
//! nothing, and an overflow increments a counter instead of blocking or
//! growing.

use robot_types::ids::{ActuatorId, ContactSiteId, JointId};
use robot_types::time::Duration;
use robot_types::units::{Celsius, NewtonMetres, Radians, RadiansPerSecond, Range};

/// A detected violation, with the measurement and limit involved.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Fault {
    /// The observation this tick is older than its freshness limit.
    StaleObservation {
        /// Measured age.
        age: Duration,
        /// The limit it exceeded.
        limit: Duration,
    },
    /// A component of the observation was NaN or infinite, so no action
    /// could be computed from it.
    NonFiniteObservation {
        /// Index of the first non-finite component.
        index: usize,
    },
    /// A commanded value was NaN or infinite.
    NonFiniteCommand {
        /// The joint.
        joint: JointId,
    },
    /// A commanded position fell outside the joint's range.
    PositionLimitExceeded {
        /// The joint.
        joint: JointId,
        /// Commanded value.
        commanded: Radians,
        /// The joint's range.
        limit: Range<Radians>,
    },
    /// A commanded velocity exceeded the joint's magnitude limit.
    VelocityLimitExceeded {
        /// The joint.
        joint: JointId,
        /// Commanded value.
        commanded: RadiansPerSecond,
        /// The magnitude limit.
        limit: RadiansPerSecond,
    },
    /// A commanded torque exceeded the joint's magnitude limit.
    TorqueLimitExceeded {
        /// The joint.
        joint: JointId,
        /// Commanded value.
        commanded: NewtonMetres,
        /// The magnitude limit.
        limit: NewtonMetres,
    },
    /// An actuator's measured temperature is at or above its limit.
    OverTemperature {
        /// The actuator.
        actuator: ActuatorId,
        /// Measured temperature.
        measured: Celsius,
        /// The limit.
        limit: Celsius,
    },
    /// The communications link is older than its health limit.
    CommunicationsLost {
        /// Measured age.
        age: Duration,
        /// The limit it exceeded.
        limit: Duration,
    },
    /// Contact was detected at a site without current permission.
    UnauthorizedContact {
        /// The site.
        site: ContactSiteId,
    },
    /// Actuation was attempted in a mode or source that does not permit it.
    ActuationNotPermitted {
        /// The mode at the time.
        mode: robot_types::mode::OperatingMode,
    },
}

/// The category of a fault, discarding its payload. Used to look up the
/// declared response in a [`crate::policy::FaultPolicy`] without requiring a
/// full match on every field.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum FaultCategory {
    /// See [`Fault::StaleObservation`].
    StaleObservation,
    /// See [`Fault::NonFiniteObservation`].
    NonFiniteObservation,
    /// See [`Fault::NonFiniteCommand`].
    NonFiniteCommand,
    /// See [`Fault::PositionLimitExceeded`].
    PositionLimitExceeded,
    /// See [`Fault::VelocityLimitExceeded`].
    VelocityLimitExceeded,
    /// See [`Fault::TorqueLimitExceeded`].
    TorqueLimitExceeded,
    /// See [`Fault::OverTemperature`].
    OverTemperature,
    /// See [`Fault::CommunicationsLost`].
    CommunicationsLost,
    /// See [`Fault::UnauthorizedContact`].
    UnauthorizedContact,
    /// See [`Fault::ActuationNotPermitted`].
    ActuationNotPermitted,
}

impl Fault {
    /// This fault's category.
    #[must_use]
    pub const fn category(&self) -> FaultCategory {
        match self {
            Self::StaleObservation { .. } => FaultCategory::StaleObservation,
            Self::NonFiniteObservation { .. } => FaultCategory::NonFiniteObservation,
            Self::NonFiniteCommand { .. } => FaultCategory::NonFiniteCommand,
            Self::PositionLimitExceeded { .. } => FaultCategory::PositionLimitExceeded,
            Self::VelocityLimitExceeded { .. } => FaultCategory::VelocityLimitExceeded,
            Self::TorqueLimitExceeded { .. } => FaultCategory::TorqueLimitExceeded,
            Self::OverTemperature { .. } => FaultCategory::OverTemperature,
            Self::CommunicationsLost { .. } => FaultCategory::CommunicationsLost,
            Self::UnauthorizedContact { .. } => FaultCategory::UnauthorizedContact,
            Self::ActuationNotPermitted { .. } => FaultCategory::ActuationNotPermitted,
        }
    }
}

/// How far outside its limit a fault's measurement was, where that is
/// meaningful. Used in reports; never used to decide the response (the
/// response is the declared [`crate::policy::FaultPolicy`] entry for the
/// fault's category, not a function of severity).
#[must_use]
pub fn excess_magnitude(fault: &Fault) -> Option<f64> {
    match fault {
        Fault::StaleObservation { age, limit } | Fault::CommunicationsLost { age, limit } => {
            Some((age.nanos() - limit.nanos()).max(0) as f64 * 1e-9)
        }
        Fault::PositionLimitExceeded {
            commanded, limit, ..
        } => Some(limit.excess(*commanded).value().abs()),
        Fault::VelocityLimitExceeded {
            commanded, limit, ..
        } => Some((commanded.value().abs() - limit.value()).max(0.0)),
        Fault::TorqueLimitExceeded {
            commanded, limit, ..
        } => Some((commanded.value().abs() - limit.value()).max(0.0)),
        Fault::OverTemperature {
            measured, limit, ..
        } => Some((measured.value() - limit.value()).max(0.0)),
        Fault::NonFiniteObservation { .. }
        | Fault::NonFiniteCommand { .. }
        | Fault::UnauthorizedContact { .. }
        | Fault::ActuationNotPermitted { .. } => None,
    }
}

/// A fixed-capacity, allocation-free record of the faults detected on one
/// tick.
///
/// `N` is chosen by the caller to exceed the number of distinct faults a
/// single tick could plausibly raise (bounded by the joint and actuator
/// counts of the robot profile); an overflow is recorded as a count, never as
/// a block or a growth.
#[derive(Clone, Copy, Debug)]
pub struct FaultLog<const N: usize> {
    entries: [Option<Fault>; N],
    len: usize,
    overflowed: u32,
}

impl<const N: usize> Default for FaultLog<N> {
    fn default() -> Self {
        Self {
            entries: [None; N],
            len: 0,
            overflowed: 0,
        }
    }
}

impl<const N: usize> FaultLog<N> {
    /// An empty log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Clears the log for reuse on the next tick, preserving its storage
    /// (no reallocation: `entries` is a fixed-size array).
    pub fn clear(&mut self) {
        self.entries = [None; N];
        self.len = 0;
        self.overflowed = 0;
    }

    /// Records a fault, or increments the overflow counter if the log is
    /// full.
    pub fn push(&mut self, fault: Fault) {
        if self.len < N {
            self.entries[self.len] = Some(fault);
            self.len += 1;
        } else {
            self.overflowed += 1;
        }
    }

    /// The recorded faults, in detection order.
    #[must_use]
    pub fn entries(&self) -> &[Option<Fault>] {
        &self.entries[..self.len]
    }

    /// Number of faults recorded (not counting overflow).
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when no fault was recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0 && self.overflowed == 0
    }

    /// Number of faults dropped because the log was already full.
    #[must_use]
    pub fn overflowed(&self) -> u32 {
        self.overflowed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn excess_magnitude_is_nonnegative_and_present_where_defined() {
        let f = Fault::VelocityLimitExceeded {
            joint: JointId::new(0),
            commanded: RadiansPerSecond(5.0),
            limit: RadiansPerSecond(3.0),
        };
        assert_eq!(excess_magnitude(&f), Some(2.0));
        let within = Fault::VelocityLimitExceeded {
            joint: JointId::new(0),
            commanded: RadiansPerSecond(1.0),
            limit: RadiansPerSecond(3.0),
        };
        assert_eq!(excess_magnitude(&within), Some(0.0));
        assert_eq!(
            excess_magnitude(&Fault::NonFiniteCommand {
                joint: JointId::new(0)
            }),
            None
        );
    }

    #[test]
    fn log_overflow_increments_a_counter_instead_of_growing() {
        let mut log: FaultLog<2> = FaultLog::new();
        log.push(Fault::NonFiniteCommand {
            joint: JointId::new(0),
        });
        log.push(Fault::NonFiniteCommand {
            joint: JointId::new(1),
        });
        log.push(Fault::NonFiniteCommand {
            joint: JointId::new(2),
        });
        assert_eq!(log.len(), 2);
        assert_eq!(log.overflowed(), 1);
        assert_eq!(log.entries().len(), 2);
    }

    #[test]
    fn clear_resets_without_reallocating() {
        let mut log: FaultLog<4> = FaultLog::new();
        log.push(Fault::CommunicationsLost {
            age: Duration::from_millis(500),
            limit: Duration::from_millis(100),
        });
        assert!(!log.is_empty());
        log.clear();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
        assert_eq!(log.overflowed(), 0);
    }

    #[test]
    fn categories_discard_payload_but_preserve_variant_identity() {
        let a = Fault::StaleObservation {
            age: Duration::from_millis(1),
            limit: Duration::from_millis(2),
        };
        let b = Fault::StaleObservation {
            age: Duration::from_millis(99),
            limit: Duration::from_millis(2),
        };
        assert_eq!(a.category(), b.category());
        assert_eq!(a.category(), FaultCategory::StaleObservation);
        let c = Fault::ActuationNotPermitted {
            mode: robot_types::mode::OperatingMode::OfflineReplay,
        };
        assert_ne!(a.category(), c.category());
    }
}

//! Motor transmission, torque-speed envelope and actuator health limits.
//!
//! SPECS.md Section 7.1: "Motor transmission, command interface, torque-speed
//! envelope, gains, saturation, latency, thermal limits and uncertainty
//! bounds." Each of those is a field below; none is inferred.

use core::fmt;
use robot_types::ids::{ActuatorId, JointId, Name};
use robot_types::time::Duration;
use robot_types::units::{Amperes, Celsius, NewtonMetres, RadiansPerSecond};

/// How an actuator's shaft motion relates to its driven joint's motion.
///
/// The ratio maps joint-side values to motor-side values:
/// `motor_value = ratio * joint_value`. A reduction gearbox has `ratio > 1`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Transmission {
    /// Motor-to-joint ratio.
    pub ratio: f64,
    /// Fraction of motor torque lost to friction and backlash, in `[0, 1)`.
    /// Applied as `joint_torque = motor_torque * (1 - loss) / ratio`.
    pub efficiency_loss: f64,
}

impl Transmission {
    /// A direct-drive transmission with no loss.
    pub const DIRECT: Self = Self {
        ratio: 1.0,
        efficiency_loss: 0.0,
    };

    /// A geared transmission with the given ratio and no modeled loss.
    #[must_use]
    pub const fn geared(ratio: f64) -> Self {
        Self {
            ratio,
            efficiency_loss: 0.0,
        }
    }

    /// Whether the parameters are usable.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.ratio.is_finite()
            && self.ratio != 0.0
            && self.efficiency_loss.is_finite()
            && (0.0..1.0).contains(&self.efficiency_loss)
    }

    /// Converts a motor-side torque to the corresponding joint-side torque.
    #[must_use]
    pub fn motor_torque_to_joint(&self, motor_torque: NewtonMetres) -> NewtonMetres {
        motor_torque * ((1.0 - self.efficiency_loss) / self.ratio)
    }

    /// Converts a joint-side velocity to the corresponding motor-side
    /// velocity.
    #[must_use]
    pub fn joint_velocity_to_motor(&self, joint_velocity: RadiansPerSecond) -> RadiansPerSecond {
        joint_velocity * self.ratio
    }
}

/// A torque-speed envelope with linear derating between a low-speed corner
/// and a no-load speed.
///
/// Below `corner_speed` the available torque is `stall_torque`; above it,
/// torque derates linearly to zero at `no_load_speed`. This is the standard
/// two-point DC-motor approximation; a robot profile that needs a richer
/// curve records one as a named extension rather than forcing it through this
/// shape.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct TorqueSpeedEnvelope {
    /// Maximum torque available at low speed.
    pub stall_torque: NewtonMetres,
    /// Speed up to which `stall_torque` is available.
    pub corner_speed: RadiansPerSecond,
    /// Speed at which available torque reaches zero.
    pub no_load_speed: RadiansPerSecond,
}

impl TorqueSpeedEnvelope {
    /// Whether the parameters describe a usable envelope.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.stall_torque.is_finite()
            && self.stall_torque.value() >= 0.0
            && self.corner_speed.is_finite()
            && self.corner_speed.value() >= 0.0
            && self.no_load_speed.is_finite()
            && self.no_load_speed.value() > self.corner_speed.value()
    }

    /// The maximum torque available at `speed` (magnitude).
    #[must_use]
    pub fn available_torque(&self, speed: RadiansPerSecond) -> NewtonMetres {
        let s = speed.value().abs();
        if s <= self.corner_speed.value() {
            self.stall_torque
        } else if s >= self.no_load_speed.value() {
            NewtonMetres::ZERO
        } else {
            let span = self.no_load_speed.value() - self.corner_speed.value();
            let frac = (self.no_load_speed.value() - s) / span;
            self.stall_torque * frac
        }
    }
}

/// Proportional/derivative gains for a position-controlled actuator.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct PdGains {
    /// Proportional gain, torque per unit position error.
    pub kp: f64,
    /// Derivative gain, torque per unit velocity error.
    pub kd: f64,
}

impl PdGains {
    /// Whether the gains are finite and non-negative.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.kp.is_finite() && self.kp >= 0.0 && self.kd.is_finite() && self.kd >= 0.0
    }
}

/// Thermal limits and the model used to project temperature from sustained
/// current draw.
///
/// The model is a single-pole thermal lag toward a current-dependent
/// steady-state temperature; it is a planning aid for the supervisor's
/// thermal derating, not a substitute for a measured thermal curve.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ThermalLimits {
    /// Temperature above which the actuator must derate or stop.
    pub limit: Celsius,
    /// Ambient temperature assumed by the steady-state model.
    pub ambient: Celsius,
    /// Time constant of the thermal lag.
    pub time_constant: Duration,
    /// Current at which the steady-state temperature reaches `limit` from
    /// `ambient`, used to scale the steady-state projection.
    pub current_at_limit: Amperes,
}

impl ThermalLimits {
    /// Whether the parameters are usable.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.limit.is_finite()
            && self.ambient.is_finite()
            && self.limit.value() > self.ambient.value()
            && self.time_constant.is_positive()
            && self.current_at_limit.is_finite()
            && self.current_at_limit.value() > 0.0
    }

    /// Steady-state temperature for sustained `current`.
    #[must_use]
    pub fn steady_state_temperature(&self, current: Amperes) -> Celsius {
        let span = self.limit.value() - self.ambient.value();
        let ratio = (current.value() / self.current_at_limit.value()).powi(2);
        Celsius(self.ambient.value() + span * ratio)
    }
}

/// One actuator driving a single joint through a transmission.
#[derive(Clone, PartialEq, Debug)]
pub struct ActuatorSpec {
    /// Stable identifier.
    pub id: ActuatorId,
    /// Stable name.
    pub name: Name,
    /// The joint this actuator drives.
    pub joint: JointId,
    /// Motor-to-joint transmission.
    pub transmission: Transmission,
    /// Joint-side torque-speed envelope (already reflected through the
    /// transmission, so callers never apply it twice).
    pub envelope: TorqueSpeedEnvelope,
    /// Default position-control gains, where the actuator supports them.
    pub gains: Option<PdGains>,
    /// End-to-end command latency from emission to physical response onset.
    pub command_latency: Duration,
    /// Thermal limits.
    pub thermal: ThermalLimits,
    /// Bound on current-measurement uncertainty, for the torque-provenance
    /// distinction of SPECS.md Section 7.3.
    pub current_uncertainty: Amperes,
}

/// Why an actuator specification was rejected.
#[derive(Clone, PartialEq, Debug)]
pub enum ActuatorError {
    /// The transmission parameters are invalid.
    InvalidTransmission {
        /// The actuator.
        actuator: Name,
    },
    /// The torque-speed envelope is invalid.
    InvalidEnvelope {
        /// The actuator.
        actuator: Name,
    },
    /// The PD gains are invalid.
    InvalidGains {
        /// The actuator.
        actuator: Name,
    },
    /// The thermal limits are invalid.
    InvalidThermal {
        /// The actuator.
        actuator: Name,
    },
    /// The command latency was negative or non-finite.
    InvalidLatency {
        /// The actuator.
        actuator: Name,
    },
    /// The current uncertainty bound was negative or non-finite.
    InvalidCurrentUncertainty {
        /// The actuator.
        actuator: Name,
    },
}

impl fmt::Display for ActuatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTransmission { actuator } => {
                write!(f, "actuator {actuator} has an invalid transmission")
            }
            Self::InvalidEnvelope { actuator } => {
                write!(
                    f,
                    "actuator {actuator} has an invalid torque-speed envelope"
                )
            }
            Self::InvalidGains { actuator } => write!(f, "actuator {actuator} has invalid gains"),
            Self::InvalidThermal { actuator } => {
                write!(f, "actuator {actuator} has invalid thermal limits")
            }
            Self::InvalidLatency { actuator } => {
                write!(f, "actuator {actuator} has an invalid command latency")
            }
            Self::InvalidCurrentUncertainty { actuator } => {
                write!(
                    f,
                    "actuator {actuator} has an invalid current uncertainty bound"
                )
            }
        }
    }
}

impl core::error::Error for ActuatorError {}

impl ActuatorSpec {
    /// Checks this actuator's internal consistency.
    pub fn validate(&self) -> Result<(), ActuatorError> {
        if !self.transmission.is_valid() {
            return Err(ActuatorError::InvalidTransmission {
                actuator: self.name.clone(),
            });
        }
        if !self.envelope.is_valid() {
            return Err(ActuatorError::InvalidEnvelope {
                actuator: self.name.clone(),
            });
        }
        if let Some(gains) = self.gains {
            if !gains.is_valid() {
                return Err(ActuatorError::InvalidGains {
                    actuator: self.name.clone(),
                });
            }
        }
        if !self.thermal.is_valid() {
            return Err(ActuatorError::InvalidThermal {
                actuator: self.name.clone(),
            });
        }
        if self.command_latency.nanos() < 0 {
            return Err(ActuatorError::InvalidLatency {
                actuator: self.name.clone(),
            });
        }
        if !self.current_uncertainty.is_finite() || self.current_uncertainty.value() < 0.0 {
            return Err(ActuatorError::InvalidCurrentUncertainty {
                actuator: self.name.clone(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn actuator() -> ActuatorSpec {
        ActuatorSpec {
            id: ActuatorId::new(0),
            name: Name::new("elbow-motor").unwrap(),
            joint: JointId::new(0),
            transmission: Transmission::geared(100.0),
            envelope: TorqueSpeedEnvelope {
                stall_torque: NewtonMetres(20.0),
                corner_speed: RadiansPerSecond(2.0),
                no_load_speed: RadiansPerSecond(10.0),
            },
            gains: Some(PdGains { kp: 50.0, kd: 2.0 }),
            command_latency: Duration::from_millis(2),
            thermal: ThermalLimits {
                limit: Celsius(80.0),
                ambient: Celsius(25.0),
                time_constant: Duration::from_secs(60),
                current_at_limit: Amperes(10.0),
            },
            current_uncertainty: Amperes(0.1),
        }
    }

    #[test]
    fn a_well_formed_actuator_validates() {
        assert!(actuator().validate().is_ok());
    }

    #[test]
    fn transmission_rejects_zero_ratio() {
        let t = Transmission {
            ratio: 0.0,
            efficiency_loss: 0.0,
        };
        assert!(!t.is_valid());
    }

    #[test]
    fn transmission_converts_torque_and_velocity() {
        let t = Transmission {
            ratio: 10.0,
            efficiency_loss: 0.1,
        };
        // motor torque 10 N*m -> joint torque = 10 * 0.9 / 10 = 0.9 N*m
        assert!((t.motor_torque_to_joint(NewtonMetres(10.0)).value() - 0.9).abs() < 1e-12);
        assert!((t.joint_velocity_to_motor(RadiansPerSecond(1.0)).value() - 10.0).abs() < 1e-12);
    }

    #[test]
    fn envelope_derates_linearly_between_corner_and_no_load() {
        let env = TorqueSpeedEnvelope {
            stall_torque: NewtonMetres(10.0),
            corner_speed: RadiansPerSecond(2.0),
            no_load_speed: RadiansPerSecond(10.0),
        };
        assert_eq!(
            env.available_torque(RadiansPerSecond(0.0)),
            NewtonMetres(10.0)
        );
        assert_eq!(
            env.available_torque(RadiansPerSecond(2.0)),
            NewtonMetres(10.0)
        );
        assert_eq!(
            env.available_torque(RadiansPerSecond(10.0)),
            NewtonMetres(0.0)
        );
        assert_eq!(
            env.available_torque(RadiansPerSecond(20.0)),
            NewtonMetres(0.0)
        );
        // Midpoint of the derating region.
        let mid = env.available_torque(RadiansPerSecond(6.0));
        assert!((mid.value() - 5.0).abs() < 1e-9);
        // Direction-agnostic: the envelope is a magnitude limit.
        assert_eq!(
            env.available_torque(RadiansPerSecond(-2.0)),
            NewtonMetres(10.0)
        );
    }

    #[test]
    fn an_envelope_with_no_load_at_or_below_corner_is_invalid() {
        let env = TorqueSpeedEnvelope {
            stall_torque: NewtonMetres(10.0),
            corner_speed: RadiansPerSecond(5.0),
            no_load_speed: RadiansPerSecond(5.0),
        };
        assert!(!env.is_valid());
    }

    #[test]
    fn thermal_steady_state_scales_with_current_squared() {
        let thermal = actuator().thermal;
        let at_limit = thermal.steady_state_temperature(Amperes(10.0));
        assert!((at_limit.value() - 80.0).abs() < 1e-9);
        let half_current = thermal.steady_state_temperature(Amperes(5.0));
        // Quarter of the span above ambient at half the rated current.
        assert!((half_current.value() - (25.0 + 55.0 * 0.25)).abs() < 1e-9);
    }

    #[test]
    fn invalid_gains_are_caught() {
        let mut a = actuator();
        a.gains = Some(PdGains { kp: -1.0, kd: 0.0 });
        assert!(matches!(
            a.validate(),
            Err(ActuatorError::InvalidGains { .. })
        ));
    }

    #[test]
    fn invalid_thermal_limits_are_caught() {
        let mut a = actuator();
        a.thermal.limit = Celsius(10.0); // below ambient
        assert!(matches!(
            a.validate(),
            Err(ActuatorError::InvalidThermal { .. })
        ));
    }

    #[test]
    fn negative_current_uncertainty_is_rejected() {
        let mut a = actuator();
        a.current_uncertainty = Amperes(-0.5);
        assert!(matches!(
            a.validate(),
            Err(ActuatorError::InvalidCurrentUncertainty { .. })
        ));
    }
}

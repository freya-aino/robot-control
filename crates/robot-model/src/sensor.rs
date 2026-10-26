//! Sensor placement, rates, covariance and missing-data behavior.
//!
//! SPECS.md Section 7.1: "Sensor frames, timestamps, rates, covariance,
//! calibration and missing-data behavior." The missing-data behavior is an
//! explicit enum rather than a convention, because `robot-perception` needs to
//! know what a gap *means* for a given sensor without guessing.

use crate::pose::RigidPose;
use core::fmt;
use robot_types::ids::{Name, SensorId};
use robot_types::units::Hertz;

/// What kind of physical quantity a sensor reports.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SensorKind {
    /// A joint encoder reporting position (and often velocity).
    JointEncoder,
    /// An inertial measurement unit.
    Imu,
    /// A contact or force/torque sensor.
    ContactForce,
    /// A current sensor on an actuator phase or bus.
    Current,
    /// A temperature sensor.
    Temperature,
    /// A sensor kind not covered above, named by its own extension.
    Generic,
}

impl SensorKind {
    /// The canonical identifier used in profiles.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::JointEncoder => "joint-encoder",
            Self::Imu => "imu",
            Self::ContactForce => "contact-force",
            Self::Current => "current",
            Self::Temperature => "temperature",
            Self::Generic => "generic",
        }
    }
}

impl fmt::Display for SensorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

/// What a consumer should do when a sample is missing.
///
/// This is a per-sensor declaration precisely because the right behavior
/// differs: holding the last IMU reading across a 2 ms gap is reasonable,
/// holding a contact-force reading across the same gap can hide a real loss of
/// contact.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MissingDataBehavior {
    /// Repeat the last valid sample, bounded by a maximum hold duration
    /// declared elsewhere (the estimator's own staleness policy).
    HoldLast,
    /// Substitute a declared default value.
    SubstituteDefault,
    /// Mark the sample invalid; the consumer must have its own policy for an
    /// invalid input rather than receiving a silently stale one.
    MarkInvalid,
}

impl MissingDataBehavior {
    /// The canonical identifier used in profiles.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::HoldLast => "hold-last",
            Self::SubstituteDefault => "substitute-default",
            Self::MarkInvalid => "mark-invalid",
        }
    }
}

/// Diagonal measurement covariance for a sensor's output vector.
///
/// A diagonal is the common case and is what most datasheets actually supply;
/// a sensor with meaningfully correlated noise terms declares a full
/// covariance as a named extension rather than forcing every consumer to
/// handle a dense matrix by default.
#[derive(Clone, PartialEq, Debug)]
pub struct DiagonalCovariance {
    /// Variance of each output component, in the square of that component's
    /// unit.
    pub variance: Vec<f64>,
}

impl DiagonalCovariance {
    /// Whether every variance is finite and non-negative.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.variance.iter().all(|v| v.is_finite() && *v >= 0.0)
    }
}

/// One sensor of a robot profile.
#[derive(Clone, PartialEq, Debug)]
pub struct SensorSpec {
    /// Stable identifier.
    pub id: SensorId,
    /// Stable name.
    pub name: Name,
    /// What kind of sensor this is.
    pub kind: SensorKind,
    /// Width of this sensor's output vector.
    pub output_width: usize,
    /// Pose of the sensor's own frame relative to the link it is mounted on,
    /// so a reading can be placed into the kinematic tree.
    pub pose: RigidPose,
    /// Nominal sampling rate.
    pub rate: Hertz,
    /// Measurement covariance.
    pub covariance: DiagonalCovariance,
    /// What to do when a sample is missing.
    pub missing_data: MissingDataBehavior,
    /// Whether the device reports a true sampling instant, as opposed to only
    /// an arrival time (SPECS.md Section 5).
    pub provides_sampling_instant: bool,
}

/// Why a sensor specification was rejected.
#[derive(Clone, PartialEq, Debug)]
pub enum SensorError {
    /// The covariance width did not match the output width.
    CovarianceWidthMismatch {
        /// The sensor.
        sensor: Name,
        /// Output width declared.
        output_width: usize,
        /// Covariance width found.
        covariance_width: usize,
    },
    /// A covariance entry was negative or non-finite.
    InvalidCovariance {
        /// The sensor.
        sensor: Name,
    },
    /// The rate was non-positive or non-finite.
    InvalidRate {
        /// The sensor.
        sensor: Name,
    },
    /// The mounting pose was not finite.
    InvalidPose {
        /// The sensor.
        sensor: Name,
    },
    /// The output width was zero.
    ZeroWidth {
        /// The sensor.
        sensor: Name,
    },
}

impl fmt::Display for SensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CovarianceWidthMismatch {
                sensor,
                output_width,
                covariance_width,
            } => write!(
                f,
                "sensor {sensor} declares output width {output_width} but covariance width \
                 {covariance_width}"
            ),
            Self::InvalidCovariance { sensor } => {
                write!(
                    f,
                    "sensor {sensor} has a negative or non-finite covariance entry"
                )
            }
            Self::InvalidRate { sensor } => write!(f, "sensor {sensor} has an invalid rate"),
            Self::InvalidPose { sensor } => {
                write!(f, "sensor {sensor} has a non-finite mounting pose")
            }
            Self::ZeroWidth { sensor } => write!(f, "sensor {sensor} declares zero output width"),
        }
    }
}

impl core::error::Error for SensorError {}

impl SensorSpec {
    /// Checks this sensor's internal consistency.
    pub fn validate(&self) -> Result<(), SensorError> {
        if self.output_width == 0 {
            return Err(SensorError::ZeroWidth {
                sensor: self.name.clone(),
            });
        }
        if self.covariance.variance.len() != self.output_width {
            return Err(SensorError::CovarianceWidthMismatch {
                sensor: self.name.clone(),
                output_width: self.output_width,
                covariance_width: self.covariance.variance.len(),
            });
        }
        if !self.covariance.is_valid() {
            return Err(SensorError::InvalidCovariance {
                sensor: self.name.clone(),
            });
        }
        if !self.rate.value().is_finite() || self.rate.value() <= 0.0 {
            return Err(SensorError::InvalidRate {
                sensor: self.name.clone(),
            });
        }
        if !self.pose.is_finite() {
            return Err(SensorError::InvalidPose {
                sensor: self.name.clone(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sensor() -> SensorSpec {
        SensorSpec {
            id: SensorId::new(0),
            name: Name::new("torso-imu").unwrap(),
            kind: SensorKind::Imu,
            output_width: 6,
            pose: RigidPose::IDENTITY,
            rate: Hertz(500.0),
            covariance: DiagonalCovariance {
                variance: vec![1e-4; 6],
            },
            missing_data: MissingDataBehavior::HoldLast,
            provides_sampling_instant: false,
        }
    }

    #[test]
    fn a_well_formed_sensor_validates() {
        assert!(sensor().validate().is_ok());
    }

    #[test]
    fn covariance_width_must_match_output_width() {
        let mut s = sensor();
        s.covariance.variance.pop();
        assert!(matches!(
            s.validate(),
            Err(SensorError::CovarianceWidthMismatch {
                output_width: 6,
                covariance_width: 5,
                ..
            })
        ));
    }

    #[test]
    fn negative_variance_is_rejected() {
        let mut s = sensor();
        s.covariance.variance[2] = -1e-4;
        assert!(matches!(
            s.validate(),
            Err(SensorError::InvalidCovariance { .. })
        ));
    }

    #[test]
    fn nonpositive_rate_is_rejected() {
        let mut s = sensor();
        s.rate = Hertz(0.0);
        assert!(matches!(s.validate(), Err(SensorError::InvalidRate { .. })));
        s.rate = Hertz(-10.0);
        assert!(matches!(s.validate(), Err(SensorError::InvalidRate { .. })));
    }

    #[test]
    fn zero_output_width_is_rejected() {
        let mut s = sensor();
        s.output_width = 0;
        s.covariance.variance.clear();
        assert!(matches!(s.validate(), Err(SensorError::ZeroWidth { .. })));
    }

    #[test]
    fn missing_data_behavior_is_a_declared_choice_not_a_default() {
        assert_eq!(MissingDataBehavior::HoldLast.id(), "hold-last");
        assert_eq!(MissingDataBehavior::MarkInvalid.id(), "mark-invalid");
        assert_eq!(
            MissingDataBehavior::SubstituteDefault.id(),
            "substitute-default"
        );
    }
}

//! Joint kinematics, limits and the parent/child link graph.
//!
//! SPECS.md Section 7.1 requires a profile to carry "Stable joint/link names and
//! explicit indices, units, axes, limits and zero offsets" and "Root convention,
//! gravity, contact surfaces". A joint here is the typed carrier of those facts:
//! its [`JointId`](robot_types::ids::JointId) never changes meaning within a
//! profile lineage, and its sign/offset are explicit fields rather than an
//! assumption baked into a importer.

use crate::pose::RigidPose;
use core::fmt;
use robot_types::frames::{Quat, Vec3};
use robot_types::ids::{JointId, LinkId, Name};
use robot_types::units::{NewtonMetres, Radians, RadiansPerSecond, Range};

/// The kind of motion a joint permits.
///
/// [`JointKind::Free`] is a simulation-time construct, never importable
/// from URDF (`robot_model::urdf` rejects a `floating` joint type by name)
/// — see `crates/robot-sim/SIMULATION_SPECS.md` Section 1.6 and Section 3.
/// It exists to let a profile's root attach to an explicit, typically
/// massless `world` link via a `Free` joint, giving the attached body six
/// unconstrained degrees of freedom (translation plus rotation) rather
/// than the single scalar every other kind here carries. A single-scalar
/// API — [`JointSpec::transform_at`], and anything built on it such as
/// `robot_model::kinematics::forward_kinematics` — cannot represent a
/// `Free` joint's position and panics loudly rather than guess; a
/// `Free`-aware caller (`robot_sim::articulated`) reads the richer
/// position representation Section 3 of that document describes instead.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum JointKind {
    /// Rotation about a single axis, with position limits.
    Revolute,
    /// Rotation about a single axis, unbounded (no position limit).
    Continuous,
    /// Translation along a single axis, with position limits.
    Prismatic,
    /// No relative motion. Still a graph edge, for a rigidly attached link.
    Fixed,
    /// Six unconstrained degrees of freedom (translation plus rotation) —
    /// a floating base. See this enum's own doc comment.
    Free,
}

impl JointKind {
    /// Whether this kind has a bounded position range.
    #[must_use]
    pub const fn is_bounded(self) -> bool {
        matches!(self, Self::Revolute | Self::Prismatic)
    }

    /// Whether this kind has any degree of freedom at all.
    #[must_use]
    pub const fn is_actuated(self) -> bool {
        !matches!(self, Self::Fixed)
    }

    /// Whether this kind rotates or translates about a single, meaningful
    /// `axis` — true for every actuated kind except [`JointKind::Free`],
    /// which has no single axis to speak of.
    #[must_use]
    pub const fn has_single_axis(self) -> bool {
        matches!(self, Self::Revolute | Self::Continuous | Self::Prismatic)
    }

    /// Degrees of freedom this kind contributes: 0 for [`JointKind::Fixed`],
    /// 1 for every single-axis kind, 6 for [`JointKind::Free`].
    #[must_use]
    pub const fn dof_count(self) -> usize {
        match self {
            Self::Fixed => 0,
            Self::Revolute | Self::Continuous | Self::Prismatic => 1,
            Self::Free => 6,
        }
    }

    /// The canonical identifier used in profiles and URDF import.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::Revolute => "revolute",
            Self::Continuous => "continuous",
            Self::Prismatic => "prismatic",
            Self::Fixed => "fixed",
            Self::Free => "free",
        }
    }
}

impl fmt::Display for JointKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

/// Position, velocity and torque limits for one joint.
///
/// `position` is `None` for a [`JointKind::Continuous`] or [`JointKind::Fixed`]
/// joint; a bounded joint without a position range is a defect the profile
/// validator catches, not a default the engine invents.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct JointLimits {
    /// Position range, present only for a bounded joint.
    pub position: Option<Range<Radians>>,
    /// Maximum magnitude of velocity.
    pub velocity: RadiansPerSecond,
    /// Maximum magnitude of torque or force the actuator may command.
    pub effort: NewtonMetres,
}

impl JointLimits {
    /// Limits for an unbounded joint (continuous or fixed).
    #[must_use]
    pub const fn unbounded(velocity: RadiansPerSecond, effort: NewtonMetres) -> Self {
        Self {
            position: None,
            velocity,
            effort,
        }
    }

    /// Limits for a bounded joint.
    #[must_use]
    pub const fn bounded(
        position: Range<Radians>,
        velocity: RadiansPerSecond,
        effort: NewtonMetres,
    ) -> Self {
        Self {
            position: Some(position),
            velocity,
            effort,
        }
    }
}

/// One joint connecting a parent link to a child link.
///
/// The joint's own frame is reached from the parent link's frame by `origin`,
/// then the joint's single degree of freedom rotates or translates about
/// `axis`. `sign` and `zero_offset` are applied between a stored canonical
/// joint value and the raw value a specific actuator or file uses: a value `v`
/// read from hardware or a reference motion maps to the canonical value as
/// `sign * v + zero_offset`. Recording this explicitly is what SPECS.md
/// Section 7.1 means by "Verify joint ordering, sign, offsets, units".
#[derive(Clone, PartialEq, Debug)]
pub struct JointSpec {
    /// Stable identifier.
    pub id: JointId,
    /// Stable name.
    pub name: Name,
    /// What kind of motion this joint permits.
    pub kind: JointKind,
    /// The link this joint is attached to.
    pub parent: LinkId,
    /// The link this joint moves.
    pub child: LinkId,
    /// Pose of this joint's frame relative to the parent link's own frame,
    /// at the joint's zero position.
    pub origin: RigidPose,
    /// Axis of rotation or translation, in the joint's own frame.
    pub axis: Vec3,
    /// Limits.
    pub limits: JointLimits,
    /// Sign applied between a raw value and the canonical joint value.
    pub sign: f64,
    /// Offset applied between a raw value and the canonical joint value, in
    /// the joint's native unit (radians for revolute/continuous, metres for
    /// prismatic).
    pub zero_offset: f64,
}

/// Why a joint specification was rejected.
#[derive(Clone, PartialEq, Debug)]
pub enum JointError {
    /// A bounded joint kind declared no position range.
    MissingPositionLimits {
        /// The joint.
        joint: Name,
    },
    /// An unbounded joint kind declared a position range anyway.
    UnexpectedPositionLimits {
        /// The joint.
        joint: Name,
    },
    /// The axis had no usable direction.
    DegenerateAxis {
        /// The joint.
        joint: Name,
    },
    /// The sign was not exactly +1 or -1.
    InvalidSign {
        /// The joint.
        joint: Name,
        /// The value found.
        sign: f64,
    },
    /// A numeric field was not finite.
    NotFinite {
        /// The joint.
        joint: Name,
        /// Which field.
        field: &'static str,
    },
    /// A limit was negative where it must be non-negative.
    NegativeLimit {
        /// The joint.
        joint: Name,
        /// Which field.
        field: &'static str,
    },
    /// The joint's own transform was not finite.
    InvalidOrigin {
        /// The joint.
        joint: Name,
    },
}

impl fmt::Display for JointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingPositionLimits { joint } => {
                write!(
                    f,
                    "joint {joint} is bounded but declares no position limits"
                )
            }
            Self::UnexpectedPositionLimits { joint } => {
                write!(f, "joint {joint} is unbounded but declares position limits")
            }
            Self::DegenerateAxis { joint } => write!(f, "joint {joint} has a zero-length axis"),
            Self::InvalidSign { joint, sign } => {
                write!(f, "joint {joint} has sign {sign}, expected +1 or -1")
            }
            Self::NotFinite { joint, field } => {
                write!(f, "joint {joint} field {field:?} is not finite")
            }
            Self::NegativeLimit { joint, field } => {
                write!(f, "joint {joint} field {field:?} must not be negative")
            }
            Self::InvalidOrigin { joint } => write!(f, "joint {joint} has a non-finite origin"),
        }
    }
}

impl core::error::Error for JointError {}

impl JointSpec {
    /// Checks this joint's internal consistency.
    pub fn validate(&self) -> Result<(), JointError> {
        if !self.origin.is_finite() {
            return Err(JointError::InvalidOrigin {
                joint: self.name.clone(),
            });
        }
        if self.axis.normalized(1e-12).is_none() && self.kind.has_single_axis() {
            return Err(JointError::DegenerateAxis {
                joint: self.name.clone(),
            });
        }
        if (self.sign.abs() - 1.0).abs() > 1e-9 {
            return Err(JointError::InvalidSign {
                joint: self.name.clone(),
                sign: self.sign,
            });
        }
        if !self.zero_offset.is_finite() {
            return Err(JointError::NotFinite {
                joint: self.name.clone(),
                field: "zero_offset",
            });
        }
        if !self.limits.velocity.is_finite() {
            return Err(JointError::NotFinite {
                joint: self.name.clone(),
                field: "velocity_limit",
            });
        }
        if self.limits.velocity.value() < 0.0 {
            return Err(JointError::NegativeLimit {
                joint: self.name.clone(),
                field: "velocity_limit",
            });
        }
        if !self.limits.effort.is_finite() {
            return Err(JointError::NotFinite {
                joint: self.name.clone(),
                field: "effort_limit",
            });
        }
        if self.limits.effort.value() < 0.0 {
            return Err(JointError::NegativeLimit {
                joint: self.name.clone(),
                field: "effort_limit",
            });
        }
        match (self.kind.is_bounded(), &self.limits.position) {
            (true, None) => {
                return Err(JointError::MissingPositionLimits {
                    joint: self.name.clone(),
                })
            }
            (false, Some(_)) => {
                return Err(JointError::UnexpectedPositionLimits {
                    joint: self.name.clone(),
                })
            }
            (true, Some(range)) => {
                if !range.lo.is_finite() || !range.hi.is_finite() {
                    return Err(JointError::NotFinite {
                        joint: self.name.clone(),
                        field: "position_limit",
                    });
                }
            }
            (false, None) => {}
        }
        Ok(())
    }

    /// Converts a raw value (as read from hardware or a file) to the
    /// canonical joint value.
    #[must_use]
    pub fn to_canonical(&self, raw: f64) -> f64 {
        self.sign * raw + self.zero_offset
    }

    /// Converts a canonical joint value back to the raw convention.
    #[must_use]
    pub fn to_raw(&self, canonical: f64) -> f64 {
        (canonical - self.zero_offset) / self.sign
    }

    /// The pose contributed by this joint at `value` (canonical units),
    /// relative to the parent link's own frame.
    ///
    /// For [`JointKind::Fixed`] the value is ignored.
    ///
    /// # Panics
    ///
    /// Panics if `self.kind` is [`JointKind::Free`]: a `Free` joint's
    /// position is seven numbers (translation plus a quaternion), which a
    /// single `f64` cannot represent. Calling this on a `Free` joint is a
    /// caller bug this project's "reject unsupported constructs explicitly"
    /// convention treats as loud failure, not a silently wrong pose — see
    /// [`JointKind`]'s own doc comment for the caller that does support it.
    #[must_use]
    pub fn transform_at(&self, value: f64) -> RigidPose {
        let motion = match self.kind {
            JointKind::Fixed => RigidPose::IDENTITY,
            JointKind::Revolute | JointKind::Continuous => RigidPose::from_rotation(
                Quat::from_axis_angle(self.axis, Radians(value)).unwrap_or(Quat::IDENTITY),
            ),
            JointKind::Prismatic => RigidPose::from_translation(
                self.axis
                    .normalized(1e-12)
                    .map_or(Vec3::ZERO, |u| u * value),
            ),
            JointKind::Free => panic!(
                "JointKind::Free has no single-scalar value representation; \
                 use robot_sim::articulated for floating-base dynamics"
            ),
        };
        self.origin.compose(&motion)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn joint(kind: JointKind, limits: JointLimits) -> JointSpec {
        JointSpec {
            id: JointId::new(0),
            name: Name::new("elbow").unwrap(),
            kind,
            parent: LinkId::new(0),
            child: LinkId::new(1),
            origin: RigidPose::IDENTITY,
            axis: Vec3::unit_z(),
            limits,
            sign: 1.0,
            zero_offset: 0.0,
        }
    }

    #[test]
    fn a_bounded_joint_needs_position_limits() {
        let j = joint(
            JointKind::Revolute,
            JointLimits::unbounded(RadiansPerSecond(1.0), NewtonMetres(1.0)),
        );
        assert!(matches!(
            j.validate(),
            Err(JointError::MissingPositionLimits { .. })
        ));
    }

    #[test]
    fn an_unbounded_joint_rejects_position_limits() {
        let j = joint(
            JointKind::Continuous,
            JointLimits::bounded(
                Range::<Radians>::new(Radians(-1.0), Radians(1.0)).unwrap(),
                RadiansPerSecond(1.0),
                NewtonMetres(1.0),
            ),
        );
        assert!(matches!(
            j.validate(),
            Err(JointError::UnexpectedPositionLimits { .. })
        ));
    }

    #[test]
    fn a_well_formed_revolute_joint_validates() {
        let j = joint(
            JointKind::Revolute,
            JointLimits::bounded(
                Range::<Radians>::new(Radians(-1.0), Radians(1.0)).unwrap(),
                RadiansPerSecond(1.0),
                NewtonMetres(1.0),
            ),
        );
        assert!(j.validate().is_ok());
    }

    #[test]
    fn a_degenerate_axis_is_rejected_for_an_actuated_joint() {
        let mut j = joint(
            JointKind::Revolute,
            JointLimits::bounded(
                Range::<Radians>::new(Radians(-1.0), Radians(1.0)).unwrap(),
                RadiansPerSecond(1.0),
                NewtonMetres(1.0),
            ),
        );
        j.axis = Vec3::ZERO;
        assert!(matches!(
            j.validate(),
            Err(JointError::DegenerateAxis { .. })
        ));

        // A fixed joint has no axis to speak of, so a zero axis is harmless.
        let mut fixed = joint(
            JointKind::Fixed,
            JointLimits::unbounded(RadiansPerSecond(0.0), NewtonMetres(0.0)),
        );
        fixed.axis = Vec3::ZERO;
        assert!(fixed.validate().is_ok());
    }

    #[test]
    fn sign_must_be_unit_magnitude() {
        let mut j = joint(
            JointKind::Continuous,
            JointLimits::unbounded(RadiansPerSecond(1.0), NewtonMetres(1.0)),
        );
        j.sign = 2.0;
        assert!(matches!(j.validate(), Err(JointError::InvalidSign { .. })));
        j.sign = -1.0;
        assert!(j.validate().is_ok());
    }

    #[test]
    fn negative_limits_are_rejected() {
        let mut j = joint(
            JointKind::Continuous,
            JointLimits::unbounded(RadiansPerSecond(-1.0), NewtonMetres(1.0)),
        );
        assert!(matches!(
            j.validate(),
            Err(JointError::NegativeLimit { .. })
        ));
        j.limits.velocity = RadiansPerSecond(1.0);
        j.limits.effort = NewtonMetres(-1.0);
        assert!(matches!(
            j.validate(),
            Err(JointError::NegativeLimit { .. })
        ));
    }

    #[test]
    fn raw_and_canonical_values_round_trip_through_sign_and_offset() {
        let mut j = joint(
            JointKind::Continuous,
            JointLimits::unbounded(RadiansPerSecond(1.0), NewtonMetres(1.0)),
        );
        j.sign = -1.0;
        j.zero_offset = 0.2;
        let canonical = j.to_canonical(1.0);
        assert!((canonical - (-0.8)).abs() < 1e-12);
        assert!((j.to_raw(canonical) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_fixed_joint_contributes_only_its_origin() {
        let j = joint(
            JointKind::Fixed,
            JointLimits::unbounded(RadiansPerSecond(0.0), NewtonMetres(0.0)),
        );
        let pose = j.transform_at(999.0);
        assert_eq!(pose.translation, Vec3::ZERO);
        assert!(pose.rotation.angle().value() < 1e-12);
    }

    #[test]
    fn a_revolute_joint_rotates_about_its_axis() {
        let j = joint(
            JointKind::Revolute,
            JointLimits::bounded(
                Range::<Radians>::new(Radians(-3.2), Radians(3.2)).unwrap(),
                RadiansPerSecond(1.0),
                NewtonMetres(1.0),
            ),
        );
        let pose = j.transform_at(core::f64::consts::FRAC_PI_2);
        let v = pose.apply_direction(Vec3::unit_x());
        assert!((v.y - 1.0).abs() < 1e-9, "{v:?}");
    }

    #[test]
    fn a_prismatic_joint_translates_along_its_axis() {
        let mut j = joint(
            JointKind::Prismatic,
            JointLimits::bounded(
                Range::<Radians>::new(Radians(-1.0), Radians(1.0)).unwrap(),
                RadiansPerSecond(1.0),
                NewtonMetres(1.0),
            ),
        );
        j.axis = Vec3::unit_x();
        let pose = j.transform_at(0.5);
        assert!((pose.translation.x - 0.5).abs() < 1e-12);
    }

    #[test]
    fn a_free_joint_has_six_degrees_of_freedom_and_is_actuated() {
        assert_eq!(JointKind::Free.dof_count(), 6);
        assert!(JointKind::Free.is_actuated());
        assert!(!JointKind::Free.is_bounded());
        assert!(!JointKind::Free.has_single_axis());
    }

    #[test]
    fn a_free_joint_with_a_zero_axis_still_validates() {
        let mut j = joint(
            JointKind::Free,
            JointLimits::unbounded(RadiansPerSecond(1.0), NewtonMetres(1.0)),
        );
        j.axis = Vec3::ZERO;
        assert!(j.validate().is_ok());
    }

    #[test]
    fn a_free_joint_rejects_position_limits() {
        let j = joint(
            JointKind::Free,
            JointLimits::bounded(
                Range::<Radians>::new(Radians(-1.0), Radians(1.0)).unwrap(),
                RadiansPerSecond(1.0),
                NewtonMetres(1.0),
            ),
        );
        assert!(matches!(
            j.validate(),
            Err(JointError::UnexpectedPositionLimits { .. })
        ));
    }

    #[test]
    #[should_panic(expected = "no single-scalar value representation")]
    fn transform_at_panics_on_a_free_joint() {
        let j = joint(
            JointKind::Free,
            JointLimits::unbounded(RadiansPerSecond(1.0), NewtonMetres(1.0)),
        );
        let _ = j.transform_at(0.0);
    }
}

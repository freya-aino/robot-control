//! A local rigid pose: translation and rotation with no named frame.
//!
//! The named-frame machinery in `robot_types::frames` exists for genuinely
//! cross-cutting frames — world, a sensor's output convention, the
//! orientation C-04 leaves open — where naming the frames a transform maps
//! between prevents silently assuming a shared convention. A joint's origin,
//! a sensor's mounting offset and a collision shape's placement are not that
//! kind of fact: each is *structurally* local to exactly one link, and
//! inventing a distinct [`robot_types::frames::FrameId`] per link would only
//! relabel "the parent link's own frame" without adding information.
//! `RigidPose` is the deliberately un-named equivalent for that local case.

use core::fmt;
use core::ops::Mul;
use robot_types::frames::{Quat, Vec3};

/// A translation and rotation, local to whatever structure holds it.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RigidPose {
    /// Translation.
    pub translation: Vec3,
    /// Rotation.
    pub rotation: Quat,
}

impl RigidPose {
    /// The identity pose.
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
    };

    /// A pose from parts.
    #[must_use]
    pub const fn new(translation: Vec3, rotation: Quat) -> Self {
        Self {
            translation,
            rotation,
        }
    }

    /// A pure translation with identity rotation.
    #[must_use]
    pub const fn from_translation(translation: Vec3) -> Self {
        Self {
            translation,
            rotation: Quat::IDENTITY,
        }
    }

    /// A pure rotation with zero translation.
    #[must_use]
    pub const fn from_rotation(rotation: Quat) -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation,
        }
    }

    /// Whether both parts are finite.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.translation.is_finite()
            && self.rotation.w.is_finite()
            && self.rotation.x.is_finite()
            && self.rotation.y.is_finite()
            && self.rotation.z.is_finite()
    }

    /// Maps a point expressed in this pose's local frame into the parent.
    #[must_use]
    pub fn apply(&self, point: Vec3) -> Vec3 {
        self.rotation.rotate(point) + self.translation
    }

    /// Maps a direction, ignoring translation.
    #[must_use]
    pub fn apply_direction(&self, direction: Vec3) -> Vec3 {
        self.rotation.rotate(direction)
    }

    /// Composes `self` (parent-from-this) with `inner` (this-from-child),
    /// producing parent-from-child.
    #[must_use]
    pub fn compose(&self, inner: &Self) -> Self {
        Self {
            translation: self.rotation.rotate(inner.translation) + self.translation,
            rotation: self.rotation.mul(inner.rotation),
        }
    }

    /// The inverse pose.
    #[must_use]
    pub fn inverse(&self) -> Self {
        let inv = self.rotation.conjugate();
        Self {
            rotation: inv,
            translation: -inv.rotate(self.translation),
        }
    }

    /// Renormalizes the rotation after accumulated rounding.
    #[must_use]
    pub fn renormalized(mut self) -> Self {
        self.rotation = self.rotation.normalized();
        self
    }
}

impl fmt::Display for RigidPose {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "t={:?} r={:?}", self.translation, self.rotation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_types::units::Radians;

    #[test]
    fn compose_then_inverse_round_trips_a_point() {
        let a = RigidPose::new(
            Vec3::new(1.0, 0.0, 0.0),
            Quat::from_rz(Radians(0.3)).unwrap(),
        );
        let b = RigidPose::new(
            Vec3::new(0.0, 2.0, 0.0),
            Quat::from_rx(Radians(0.1)).unwrap(),
        );
        let composed = a.compose(&b);
        let p = Vec3::new(0.5, -0.2, 0.1);
        let forward = composed.apply(p);
        let via_steps = a.apply(b.apply(p));
        assert!((forward - via_steps).norm() < 1e-9);
        let back = composed.inverse().apply(forward);
        assert!((back - p).norm() < 1e-9);
    }

    #[test]
    fn identity_is_a_no_op() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(RigidPose::IDENTITY.apply(p), p);
    }
}

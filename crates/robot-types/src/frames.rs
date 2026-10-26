//! Coordinate frames, named-component quaternions and rigid transforms.
//!
//! Two requirements shape this module.
//!
//! SPECS.md Appendix A.3 and [C-04] leave the canonical frame, gravity
//! direction, root convention and quaternion ordering **open**: "no code may
//! assume a project-wide convention, and every transform MUST name the frames it
//! maps between". [`Transform`] therefore carries its source and target
//! [`FrameId`] and refuses to compose with a transform that does not meet it.
//!
//! SPECS.md Section 7.2 requires "typed quaternion serialization with named
//! components over ambiguous four-element arrays". [`Quat`] has no array
//! constructor and no `as_array`; components are reached by name.
//!
//! [C-04]: ../../../SPECS.md

use crate::units::{Metres, Radians};
use core::fmt;
use core::ops::{Add, Mul, Neg, Sub};

/// A three-component vector in an unnamed frame.
///
/// The frame is named by whatever carries the vector: a [`Transform`], a
/// [`Point`] or a field whose documentation states it. A bare `Vec3` is a
/// triple of numbers, which is why the geometry below takes frames explicitly.
#[derive(Clone, Copy, PartialEq, Default)]
pub struct Vec3 {
    /// First component.
    pub x: f64,
    /// Second component.
    pub y: f64,
    /// Third component.
    pub z: f64,
}

impl Vec3 {
    /// The zero vector.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// A vector from its components.
    #[inline]
    #[must_use]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Unit vector along the first axis.
    #[must_use]
    pub const fn unit_x() -> Self {
        Self::new(1.0, 0.0, 0.0)
    }

    /// Unit vector along the second axis.
    #[must_use]
    pub const fn unit_y() -> Self {
        Self::new(0.0, 1.0, 0.0)
    }

    /// Unit vector along the third axis.
    #[must_use]
    pub const fn unit_z() -> Self {
        Self::new(0.0, 0.0, 1.0)
    }

    /// Dot product.
    #[inline]
    #[must_use]
    pub fn dot(self, rhs: Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Cross product.
    #[inline]
    #[must_use]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    /// Euclidean norm.
    #[inline]
    #[must_use]
    pub fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }

    /// Squared norm.
    #[inline]
    #[must_use]
    pub fn norm_squared(self) -> f64 {
        self.dot(self)
    }

    /// Unit vector in the same direction, or `None` when the norm is below
    /// `tolerance`. Returning `None` rather than a NaN-filled vector keeps a
    /// degenerate axis from propagating into a rotation.
    #[must_use]
    pub fn normalized(self, tolerance: f64) -> Option<Self> {
        let n = self.norm();
        if n <= tolerance || !n.is_finite() {
            None
        } else {
            Some(self * (1.0 / n))
        }
    }

    /// True when every component is finite.
    #[inline]
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// Entrywise product.
    #[inline]
    #[must_use]
    pub fn component_mul(self, rhs: Self) -> Self {
        Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }

    /// Linear interpolation, `t` unclamped.
    #[inline]
    #[must_use]
    pub fn lerp(self, rhs: Self, t: f64) -> Self {
        self + (rhs - self) * t
    }

    /// Components in a fixed order, for numerical kernels that need a slice.
    ///
    /// The order is `(x, y, z)` as declared by the field names; this is an
    /// implementation convenience and never a serialization format.
    #[inline]
    #[must_use]
    pub const fn to_xyz(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    /// Inverse of [`Vec3::to_xyz`].
    #[inline]
    #[must_use]
    pub const fn from_xyz(v: [f64; 3]) -> Self {
        Self::new(v[0], v[1], v[2])
    }
}

impl Add for Vec3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub for Vec3 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Neg for Vec3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

impl Mul<f64> for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl fmt::Debug for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(x={}, y={}, z={})", self.x, self.y, self.z)
    }
}

/// A unit quaternion with named components.
///
/// Deliberately has no four-element array constructor: SPECS.md Section 7.2
/// forbids ambiguous four-element arrays precisely because `[w,x,y,z]` and
/// `[x,y,z,w]` are indistinguishable in a file.
#[derive(Clone, Copy, PartialEq)]
pub struct Quat {
    /// Scalar component.
    pub w: f64,
    /// First vector component.
    pub x: f64,
    /// Second vector component.
    pub y: f64,
    /// Third vector component.
    pub z: f64,
}

/// Why a rotation could not be constructed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RotationError {
    /// The rotation axis had no usable direction.
    DegenerateAxis,
    /// A component was NaN or infinite.
    NotFinite,
    /// The quaternion's norm was too far from one to be normalized safely.
    NotUnit,
}

impl fmt::Display for RotationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DegenerateAxis => write!(f, "rotation axis has zero length"),
            Self::NotFinite => write!(f, "rotation component is not finite"),
            Self::NotUnit => write!(f, "quaternion norm is too far from one"),
        }
    }
}

impl core::error::Error for RotationError {}

impl Quat {
    /// The identity rotation.
    pub const IDENTITY: Self = Self {
        w: 1.0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// A quaternion from named components, normalized on construction.
    pub fn from_components(w: f64, x: f64, y: f64, z: f64) -> Result<Self, RotationError> {
        if !(w.is_finite() && x.is_finite() && y.is_finite() && z.is_finite()) {
            return Err(RotationError::NotFinite);
        }
        let n = (w * w + x * x + y * y + z * z).sqrt();
        if n < 1e-9 {
            return Err(RotationError::NotUnit);
        }
        Ok(Self {
            w: w / n,
            x: x / n,
            y: y / n,
            z: z / n,
        })
    }

    /// A rotation of `angle` about `axis`.
    pub fn from_axis_angle(axis: Vec3, angle: Radians) -> Result<Self, RotationError> {
        if !angle.is_finite() {
            return Err(RotationError::NotFinite);
        }
        let unit = axis
            .normalized(1e-12)
            .ok_or(RotationError::DegenerateAxis)?;
        let half = angle.value() * 0.5;
        let s = half.sin();
        Ok(Self {
            w: half.cos(),
            x: unit.x * s,
            y: unit.y * s,
            z: unit.z * s,
        })
    }

    /// Rotation about the first axis.
    pub fn from_rx(angle: Radians) -> Result<Self, RotationError> {
        Self::from_axis_angle(Vec3::unit_x(), angle)
    }

    /// Rotation about the second axis.
    pub fn from_ry(angle: Radians) -> Result<Self, RotationError> {
        Self::from_axis_angle(Vec3::unit_y(), angle)
    }

    /// Rotation about the third axis.
    pub fn from_rz(angle: Radians) -> Result<Self, RotationError> {
        Self::from_axis_angle(Vec3::unit_z(), angle)
    }

    /// Conjugate, which for a unit quaternion is the inverse rotation.
    #[inline]
    #[must_use]
    pub const fn conjugate(self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Applies the rotation to a vector.
    #[must_use]
    pub fn rotate(self, v: Vec3) -> Vec3 {
        let u = Vec3::new(self.x, self.y, self.z);
        let s = self.w;
        u * (2.0 * u.dot(v)) + v * (s * s - u.dot(u)) + u.cross(v) * (2.0 * s)
    }

    /// Renormalizes after accumulated rounding.
    #[must_use]
    pub fn normalized(self) -> Self {
        let n = (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if n < 1e-12 || !n.is_finite() {
            Self::IDENTITY
        } else {
            Self {
                w: self.w / n,
                x: self.x / n,
                y: self.y / n,
                z: self.z / n,
            }
        }
    }

    /// Rotation angle in `[0, pi]`.
    #[must_use]
    pub fn angle(self) -> Radians {
        let w = self.w.abs().min(1.0);
        Radians(2.0 * w.acos())
    }

    /// Geodesic distance to `other` in `[0, pi]`.
    ///
    /// Used as the orientation error term of the retargeting report; it accounts
    /// for the double cover, so `q` and `-q` are at distance zero.
    #[must_use]
    pub fn angle_to(self, other: Self) -> Radians {
        self.conjugate().mul(other).angle()
    }

    /// Spherical linear interpolation, taking the shorter arc.
    #[must_use]
    pub fn slerp(self, other: Self, t: f64) -> Self {
        let mut cos = self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z;
        let mut target = other;
        if cos < 0.0 {
            cos = -cos;
            target = Self {
                w: -other.w,
                x: -other.x,
                y: -other.y,
                z: -other.z,
            };
        }
        if cos > 1.0 - 1e-9 {
            // Nearly parallel: linear interpolation is numerically better here.
            return Self {
                w: self.w + (target.w - self.w) * t,
                x: self.x + (target.x - self.x) * t,
                y: self.y + (target.y - self.y) * t,
                z: self.z + (target.z - self.z) * t,
            }
            .normalized();
        }
        let theta = cos.clamp(-1.0, 1.0).acos();
        let sin = theta.sin();
        let a = ((1.0 - t) * theta).sin() / sin;
        let b = (t * theta).sin() / sin;
        Self {
            w: self.w * a + target.w * b,
            x: self.x * a + target.x * b,
            y: self.y * a + target.y * b,
            z: self.z * a + target.z * b,
        }
    }

    /// The rotation as a row-major 3x3 matrix.
    #[must_use]
    pub fn to_rotation_matrix(self) -> crate::math::Matrix {
        let (w, x, y, z) = (self.w, self.x, self.y, self.z);
        crate::math::Matrix::from_row_major(
            3,
            3,
            &[
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        )
        .expect("3x3 rotation matrix has nine entries")
    }

    /// The rotation vector (axis times angle), which is the minimal
    /// three-parameter error used by the IK solver and the estimator.
    #[must_use]
    pub fn to_rotation_vector(self) -> Vec3 {
        let q = if self.w < 0.0 {
            Self {
                w: -self.w,
                x: -self.x,
                y: -self.y,
                z: -self.z,
            }
        } else {
            self
        };
        let v = Vec3::new(q.x, q.y, q.z);
        let sin_half = v.norm();
        if sin_half < 1e-12 {
            // Small-angle limit: the series is better conditioned than acos.
            v * 2.0
        } else {
            let angle = 2.0 * sin_half.atan2(q.w);
            v * (angle / sin_half)
        }
    }

    /// Inverse of [`Quat::to_rotation_vector`].
    #[must_use]
    pub fn from_rotation_vector(v: Vec3) -> Self {
        let angle = v.norm();
        if angle < 1e-12 {
            Self {
                w: 1.0,
                x: v.x * 0.5,
                y: v.y * 0.5,
                z: v.z * 0.5,
            }
            .normalized()
        } else {
            let half = angle * 0.5;
            let s = half.sin() / angle;
            Self {
                w: half.cos(),
                x: v.x * s,
                y: v.y * s,
                z: v.z * s,
            }
        }
    }
}

impl fmt::Debug for Quat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(w={}, x={}, y={}, z={})",
            self.w, self.x, self.y, self.z
        )
    }
}

/// Hamilton product: the rotation `self` after `rhs`. Implemented as the
/// standard trait (rather than an inherent `fn mul`) so that `.mul(rhs)` and
/// `self * rhs` are the same, unambiguous operation.
impl Mul for Quat {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self {
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
        }
    }
}

/// The rotation order of a sequence of intrinsic Euler angles.
///
/// SPECS.md Section 7.2: "Euler-angle conventions are source-specific; never
/// assume every BVH file uses the same rotation order." A BVH importer reads the
/// order from the file's `CHANNELS` line and records it here.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EulerOrder {
    /// Intrinsic X then Y then Z.
    Xyz,
    /// Intrinsic X then Z then Y.
    Xzy,
    /// Intrinsic Y then X then Z.
    Yxz,
    /// Intrinsic Y then Z then X.
    Yzx,
    /// Intrinsic Z then X then Y.
    Zxy,
    /// Intrinsic Z then Y then X.
    Zyx,
}

impl EulerOrder {
    /// The three axes in application order.
    #[must_use]
    pub const fn axes(self) -> [Axis; 3] {
        match self {
            Self::Xyz => [Axis::X, Axis::Y, Axis::Z],
            Self::Xzy => [Axis::X, Axis::Z, Axis::Y],
            Self::Yxz => [Axis::Y, Axis::X, Axis::Z],
            Self::Yzx => [Axis::Y, Axis::Z, Axis::X],
            Self::Zxy => [Axis::Z, Axis::X, Axis::Y],
            Self::Zyx => [Axis::Z, Axis::Y, Axis::X],
        }
    }

    /// Builds an order from three distinct axes, or `None` if they repeat.
    #[must_use]
    pub fn from_axes(axes: [Axis; 3]) -> Option<Self> {
        Some(match axes {
            [Axis::X, Axis::Y, Axis::Z] => Self::Xyz,
            [Axis::X, Axis::Z, Axis::Y] => Self::Xzy,
            [Axis::Y, Axis::X, Axis::Z] => Self::Yxz,
            [Axis::Y, Axis::Z, Axis::X] => Self::Yzx,
            [Axis::Z, Axis::X, Axis::Y] => Self::Zxy,
            [Axis::Z, Axis::Y, Axis::X] => Self::Zyx,
            _ => return None,
        })
    }

    /// The canonical name, as written into a provenance record.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Xyz => "XYZ",
            Self::Xzy => "XZY",
            Self::Yxz => "YXZ",
            Self::Yzx => "YZX",
            Self::Zxy => "ZXY",
            Self::Zyx => "ZYX",
        }
    }

    /// Composes three intrinsic rotations in this order.
    pub fn to_quat(self, angles: [Radians; 3]) -> Result<Quat, RotationError> {
        let mut q = Quat::IDENTITY;
        for (axis, angle) in self.axes().into_iter().zip(angles) {
            q = q.mul(Quat::from_axis_angle(axis.unit_vector(), angle)?);
        }
        Ok(q)
    }
}

/// A principal axis.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Axis {
    /// First axis.
    X,
    /// Second axis.
    Y,
    /// Third axis.
    Z,
}

impl Axis {
    /// The corresponding unit vector.
    #[must_use]
    pub const fn unit_vector(self) -> Vec3 {
        match self {
            Self::X => Vec3::new(1.0, 0.0, 0.0),
            Self::Y => Vec3::new(0.0, 1.0, 0.0),
            Self::Z => Vec3::new(0.0, 0.0, 1.0),
        }
    }

    /// The canonical single-letter name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::X => "X",
            Self::Y => "Y",
            Self::Z => "Z",
        }
    }
}

/// The identity of a coordinate frame.
///
/// Frames are interned by the consumer: a profile assigns small indices to its
/// named frames, and every transform states the pair it maps between. Two
/// frames with the same index in different profiles are unrelated, which is why
/// a transform is validated against the profile that produced it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FrameId(pub u16);

impl FrameId {
    /// The world frame of a given profile, by convention index zero.
    ///
    /// The *orientation* of this frame is still open under C-04; only its index
    /// is fixed, so that a transform can name it.
    pub const WORLD: Self = Self(0);

    /// The index.
    #[inline]
    #[must_use]
    pub const fn index(self) -> u16 {
        self.0
    }
}

impl fmt::Display for FrameId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "frame#{}", self.0)
    }
}

/// Why two transforms could not be composed or applied.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FrameError {
    /// The target frame of one transform is not the source frame of the next.
    Mismatch {
        /// Frame the first transform maps into.
        expected: FrameId,
        /// Frame the second transform maps out of.
        found: FrameId,
    },
}

impl fmt::Display for FrameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mismatch { expected, found } => {
                write!(f, "frame mismatch: expected {expected}, found {found}")
            }
        }
    }
}

impl core::error::Error for FrameError {}

/// A rigid transform that names the frames it maps between.
///
/// `transform.apply(p)` maps a point expressed in `from` to the same point
/// expressed in `to`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Transform {
    /// Frame the transform maps out of.
    pub from: FrameId,
    /// Frame the transform maps into.
    pub to: FrameId,
    /// Rotation taking `from` axes to `to` axes.
    pub rotation: Quat,
    /// Origin of `from` expressed in `to`, in metres.
    pub translation: Vec3,
}

impl Transform {
    /// The identity transform on a frame.
    #[must_use]
    pub const fn identity(frame: FrameId) -> Self {
        Self {
            from: frame,
            to: frame,
            rotation: Quat::IDENTITY,
            translation: Vec3::ZERO,
        }
    }

    /// A transform from its parts.
    #[must_use]
    pub const fn new(from: FrameId, to: FrameId, rotation: Quat, translation: Vec3) -> Self {
        Self {
            from,
            to,
            rotation,
            translation,
        }
    }

    /// Maps a point from `self.from` into `self.to`.
    #[inline]
    #[must_use]
    pub fn apply(&self, point: Vec3) -> Vec3 {
        self.rotation.rotate(point) + self.translation
    }

    /// Maps a direction, ignoring translation.
    #[inline]
    #[must_use]
    pub fn apply_direction(&self, direction: Vec3) -> Vec3 {
        self.rotation.rotate(direction)
    }

    /// The transform in the opposite direction.
    #[must_use]
    pub fn inverse(&self) -> Self {
        let inv = self.rotation.conjugate();
        Self {
            from: self.to,
            to: self.from,
            rotation: inv,
            translation: -inv.rotate(self.translation),
        }
    }

    /// `self` after `inner`: maps `inner.from` to `self.to`.
    ///
    /// Rejects a frame mismatch rather than producing a transform between frames
    /// that were never related, which is the specific failure Appendix A.3 asks
    /// the type system to prevent.
    pub fn compose(&self, inner: &Self) -> Result<Self, FrameError> {
        if inner.to != self.from {
            return Err(FrameError::Mismatch {
                expected: self.from,
                found: inner.to,
            });
        }
        Ok(Self {
            from: inner.from,
            to: self.to,
            rotation: self.rotation.mul(inner.rotation),
            translation: self.rotation.rotate(inner.translation) + self.translation,
        })
    }

    /// Renormalizes the rotation after accumulated rounding.
    #[must_use]
    pub fn renormalized(mut self) -> Self {
        self.rotation = self.rotation.normalized();
        self
    }

    /// True when the rotation and translation are finite.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.translation.is_finite()
            && self.rotation.w.is_finite()
            && self.rotation.x.is_finite()
            && self.rotation.y.is_finite()
            && self.rotation.z.is_finite()
    }
}

/// A point with the frame it is expressed in attached.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Point {
    /// Frame of expression.
    pub frame: FrameId,
    /// Position in metres.
    pub position: Vec3,
}

impl Point {
    /// A point in a named frame.
    #[must_use]
    pub const fn new(frame: FrameId, position: Vec3) -> Self {
        Self { frame, position }
    }

    /// Re-expresses the point through a transform, rejecting a frame mismatch.
    pub fn transformed(&self, t: &Transform) -> Result<Self, FrameError> {
        if t.from != self.frame {
            return Err(FrameError::Mismatch {
                expected: t.from,
                found: self.frame,
            });
        }
        Ok(Self {
            frame: t.to,
            position: t.apply(self.position),
        })
    }

    /// Distance to another point in the same frame.
    pub fn distance_to(&self, other: &Self) -> Result<Metres, FrameError> {
        if self.frame != other.frame {
            return Err(FrameError::Mismatch {
                expected: self.frame,
                found: other.frame,
            });
        }
        Ok(Metres((other.position - self.position).norm()))
    }
}

/// The direction of gravity in a named frame.
///
/// Carried explicitly because C-04 leaves the canonical gravity direction open;
/// a simulator, an estimator and an importer each state the convention they were
/// configured with rather than assuming a shared one.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct GravityConvention {
    /// Frame the direction is expressed in.
    pub frame: FrameId,
    /// Unit vector pointing in the direction gravity accelerates a free body.
    pub down: Vec3,
    /// Magnitude in metres per second squared.
    pub magnitude: f64,
}

impl GravityConvention {
    /// The widely used `-Z` convention at standard gravity.
    ///
    /// This is a *configuration default for tests and fixtures*, not a project
    /// convention: C-04 is open, and a profile states its own.
    #[must_use]
    pub const fn minus_z_standard(frame: FrameId) -> Self {
        Self {
            frame,
            down: Vec3::new(0.0, 0.0, -1.0),
            magnitude: 9.806_65,
        }
    }

    /// The acceleration vector in `frame`.
    #[must_use]
    pub fn acceleration(&self) -> Vec3 {
        self.down * self.magnitude
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::Radians;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    #[test]
    fn quarter_turn_about_z_maps_x_to_y() {
        let q = Quat::from_rz(Radians(core::f64::consts::FRAC_PI_2)).unwrap();
        let v = q.rotate(Vec3::unit_x());
        assert!(
            close(v.x, 0.0) && close(v.y, 1.0) && close(v.z, 0.0),
            "{v:?}"
        );
    }

    #[test]
    fn degenerate_axis_is_rejected() {
        assert_eq!(
            Quat::from_axis_angle(Vec3::ZERO, Radians(1.0)),
            Err(RotationError::DegenerateAxis)
        );
        assert_eq!(
            Quat::from_axis_angle(Vec3::unit_x(), Radians(f64::NAN)),
            Err(RotationError::NotFinite)
        );
        assert_eq!(
            Quat::from_components(f64::INFINITY, 0.0, 0.0, 0.0),
            Err(RotationError::NotFinite)
        );
        assert_eq!(
            Quat::from_components(0.0, 0.0, 0.0, 0.0),
            Err(RotationError::NotUnit)
        );
    }

    #[test]
    fn euler_order_changes_the_result() {
        let angles = [Radians(0.3), Radians(0.5), Radians(0.7)];
        let xyz = EulerOrder::Xyz.to_quat(angles).unwrap();
        let zyx = EulerOrder::Zyx.to_quat(angles).unwrap();
        assert!(
            xyz.angle_to(zyx).value() > 1e-3,
            "two orders of the same angles must not coincide"
        );
        assert_eq!(
            EulerOrder::from_axes([Axis::Z, Axis::Y, Axis::X]),
            Some(EulerOrder::Zyx)
        );
        assert_eq!(EulerOrder::from_axes([Axis::X, Axis::X, Axis::Z]), None);
        assert_eq!(EulerOrder::Zyx.name(), "ZYX");
    }

    #[test]
    fn rotation_vector_round_trips() {
        for v in [
            Vec3::new(0.1, -0.2, 0.3),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1e-14, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ] {
            let q = Quat::from_rotation_vector(v);
            let back = q.to_rotation_vector();
            assert!((back - v).norm() < 1e-9, "{v:?} -> {back:?}");
        }
    }

    #[test]
    fn double_cover_is_zero_distance() {
        let q = Quat::from_rz(Radians(0.4)).unwrap();
        let neg = Quat {
            w: -q.w,
            x: -q.x,
            y: -q.y,
            z: -q.z,
        };
        assert!(q.angle_to(neg).value() < 1e-12);
    }

    #[test]
    fn slerp_endpoints_and_midpoint() {
        let a = Quat::IDENTITY;
        let b = Quat::from_rz(Radians(1.0)).unwrap();
        assert!(a.slerp(b, 0.0).angle_to(a).value() < 1e-12);
        assert!(a.slerp(b, 1.0).angle_to(b).value() < 1e-12);
        let mid = a.slerp(b, 0.5);
        assert!((mid.angle().value() - 0.5).abs() < 1e-9);
        // Nearly parallel inputs take the linear branch without producing NaN.
        let near = Quat::from_rz(Radians(1e-12)).unwrap();
        assert!(a.slerp(near, 0.5).angle().value().is_finite());
    }

    #[test]
    fn rotation_matrix_agrees_with_rotate() {
        let q = Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0), Radians(0.7)).unwrap();
        let m = q.to_rotation_matrix();
        let v = Vec3::new(0.3, -0.4, 0.5);
        let by_quat = q.rotate(v);
        let by_matrix = Vec3::new(
            m[(0, 0)] * v.x + m[(0, 1)] * v.y + m[(0, 2)] * v.z,
            m[(1, 0)] * v.x + m[(1, 1)] * v.y + m[(1, 2)] * v.z,
            m[(2, 0)] * v.x + m[(2, 1)] * v.y + m[(2, 2)] * v.z,
        );
        assert!((by_quat - by_matrix).norm() < 1e-12);
    }

    #[test]
    fn composing_mismatched_frames_is_rejected() {
        let a = Transform::new(FrameId(1), FrameId(2), Quat::IDENTITY, Vec3::ZERO);
        let b = Transform::new(FrameId(3), FrameId(4), Quat::IDENTITY, Vec3::ZERO);
        assert!(matches!(b.compose(&a), Err(FrameError::Mismatch { .. })));
        let c = Transform::new(FrameId(2), FrameId(3), Quat::IDENTITY, Vec3::ZERO);
        assert_eq!(c.compose(&a).unwrap().from, FrameId(1));
        assert_eq!(c.compose(&a).unwrap().to, FrameId(3));
    }

    #[test]
    fn transform_inverse_round_trips_a_point() {
        let t = Transform::new(
            FrameId(1),
            FrameId(2),
            Quat::from_ry(Radians(0.6)).unwrap(),
            Vec3::new(0.1, 0.2, 0.3),
        );
        let p = Point::new(FrameId(1), Vec3::new(1.0, -2.0, 0.5));
        let moved = p.transformed(&t).unwrap();
        assert_eq!(moved.frame, FrameId(2));
        let back = moved.transformed(&t.inverse()).unwrap();
        assert!((back.position - p.position).norm() < 1e-12);
    }

    #[test]
    fn point_distance_requires_matching_frames() {
        let a = Point::new(FrameId(1), Vec3::ZERO);
        let b = Point::new(FrameId(2), Vec3::unit_x());
        assert!(a.distance_to(&b).is_err());
        let c = Point::new(FrameId(1), Vec3::unit_x());
        assert!(close(a.distance_to(&c).unwrap().value(), 1.0));
    }

    #[test]
    fn gravity_convention_is_explicit() {
        let g = GravityConvention::minus_z_standard(FrameId::WORLD);
        assert!(close(g.acceleration().z, -9.806_65));
        assert_eq!(g.frame, FrameId::WORLD);
    }
}

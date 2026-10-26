//! Spatial vector algebra: the 6D motion/force representation the
//! multi-body dynamics solver in [`crate::articulated`] is built on.
//!
//! Follows the standard (Featherstone, *Rigid Body Dynamics Algorithms*)
//! convention: a 6D *motion* vector packs (angular, linear) velocity or
//! acceleration of a body-fixed frame's own origin, expressed in that
//! frame's own axes; a 6D *force* vector packs (torque, force) acting
//! about that same origin, in the same axes. They are kept as distinct
//! types — [`Twist`] and [`Wrench`] — rather than one generic six-vector,
//! so a motion value can never be substituted where a force is expected;
//! each transforms differently between frames (see
//! [`transform_motion`]/[`transform_force`] below), and conflating them
//! is a textbook source of sign errors in hand-rolled dynamics code.
//!
//! [`SpatialInertia`] is kept as a general symmetric 6x6
//! [`robot_types::math::Matrix`] rather than a (mass, COM, inertia)
//! triple: the articulated-body algorithm's backward pass produces
//! "effective" inertias that are no longer expressible in that simpler
//! form once a child's own degree of freedom has been eliminated from
//! it, so the general representation is not an added complication, it is
//! required.

use core::ops::Add;
use robot_model::link::{InertiaTensor, MassProperties};
use robot_model::pose::RigidPose;
use robot_types::frames::Vec3;
use robot_types::math::Matrix;

/// A 6D motion vector: angular and linear velocity (or acceleration) of a
/// body-fixed frame's own origin, expressed in that frame.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Twist {
    /// Angular part.
    pub angular: Vec3,
    /// Linear part (of the frame's own origin).
    pub linear: Vec3,
}

/// A 6D force vector: torque and force acting about a body-fixed frame's
/// own origin, expressed in that frame.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Wrench {
    /// Torque part.
    pub angular: Vec3,
    /// Force part.
    pub linear: Vec3,
}

impl Twist {
    /// The zero motion.
    pub const ZERO: Self = Self {
        angular: Vec3::ZERO,
        linear: Vec3::ZERO,
    };

    /// A pure rotation about `axis` (the revolute/continuous joint motion
    /// subspace), scaled by `rate`.
    #[must_use]
    pub fn angular_rate(axis: Vec3, rate: f64) -> Self {
        Self {
            angular: axis * rate,
            linear: Vec3::ZERO,
        }
    }

    /// A pure translation along `axis` (the prismatic joint motion
    /// subspace), scaled by `rate`.
    #[must_use]
    pub fn linear_rate(axis: Vec3, rate: f64) -> Self {
        Self {
            angular: Vec3::ZERO,
            linear: axis * rate,
        }
    }

    /// As a 6x1 column matrix (angular components first), for the
    /// multi-DOF articulated-body algebra in `crate::articulated`.
    #[must_use]
    pub(crate) fn to_column(self) -> Matrix {
        Matrix::column(&[
            self.angular.x,
            self.angular.y,
            self.angular.z,
            self.linear.x,
            self.linear.y,
            self.linear.z,
        ])
    }

    #[must_use]
    pub(crate) fn scaled(self, factor: f64) -> Self {
        Self {
            angular: self.angular * factor,
            linear: self.linear * factor,
        }
    }
}

impl Wrench {
    /// The zero force.
    pub const ZERO: Self = Self {
        angular: Vec3::ZERO,
        linear: Vec3::ZERO,
    };

    /// As a 6x1 column matrix (angular components first), for the
    /// multi-DOF articulated-body algebra in `crate::articulated`.
    #[must_use]
    pub(crate) fn to_column(self) -> Matrix {
        Matrix::column(&[
            self.angular.x,
            self.angular.y,
            self.angular.z,
            self.linear.x,
            self.linear.y,
            self.linear.z,
        ])
    }

    /// Inverse of [`Wrench::to_column`].
    #[must_use]
    pub(crate) fn from_column(m: &Matrix) -> Self {
        Self {
            angular: Vec3::new(m[(0, 0)], m[(1, 0)], m[(2, 0)]),
            linear: Vec3::new(m[(3, 0)], m[(4, 0)], m[(5, 0)]),
        }
    }

    /// The generalized dot product with a [`Twist`]: instantaneous power.
    #[must_use]
    pub fn dot(self, v: Twist) -> f64 {
        self.angular.dot(v.angular) + self.linear.dot(v.linear)
    }
}

impl Add for Twist {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            angular: self.angular + rhs.angular,
            linear: self.linear + rhs.linear,
        }
    }
}

impl Add for Wrench {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            angular: self.angular + rhs.angular,
            linear: self.linear + rhs.linear,
        }
    }
}

/// Transforms a motion vector expressed in a parent frame into the same
/// motion vector expressed in a child frame, given `parent_from_child`
/// (the child frame's pose expressed in the parent frame — exactly what
/// `JointSpec::transform_at` and `RigidPose::compose` produce).
///
/// Derivation: for a point rigidly fixed in the parent body at the
/// child's origin (position `p` in parent axes), the rigid-body velocity
/// field gives `v_parent_at_p = v.linear + v.angular x p`; rotating both
/// parts into the child's axes by the inverse of the child's rotation
/// gives the result.
#[must_use]
pub fn transform_motion(parent_from_child: &RigidPose, v: Twist) -> Twist {
    let r_inv = parent_from_child.rotation.conjugate();
    let p = parent_from_child.translation;
    Twist {
        angular: r_inv.rotate(v.angular),
        linear: r_inv.rotate(v.linear + v.angular.cross(p)),
    }
}

/// Transforms a force vector expressed in a child frame into the same
/// force vector expressed in the parent frame, given `parent_from_child`.
///
/// This is the dual of [`transform_motion`] in the opposite direction:
/// for any twist `v` in the parent frame and wrench `f` in the child
/// frame, `transform_motion(t, v).dot(f) == v.dot(transform_force(t, f))`
/// (power is frame-invariant). `spatial_duality_preserves_power` checks
/// this numerically.
#[must_use]
pub fn transform_force(parent_from_child: &RigidPose, f: Wrench) -> Wrench {
    let r = parent_from_child.rotation;
    let p = parent_from_child.translation;
    let force = r.rotate(f.linear);
    Wrench {
        angular: r.rotate(f.angular) + p.cross(force),
        linear: force,
    }
}

/// The spatial "motion cross product" `v x* w` of two motion vectors,
/// itself a motion vector (Featherstone's `ad` operator) — used for the
/// Coriolis/bias acceleration term contributed by a joint's own velocity.
#[must_use]
pub fn cross_motion_motion(v: Twist, w: Twist) -> Twist {
    Twist {
        angular: v.angular.cross(w.angular),
        linear: v.angular.cross(w.linear) + v.linear.cross(w.angular),
    }
}

/// The spatial "force cross product" `v x*f f`, the dual of
/// [`cross_motion_motion`]: a motion vector crossed with a force vector
/// produces a force vector — the velocity-product (Coriolis/centrifugal)
/// bias force of a rotating, translating rigid body.
#[must_use]
pub fn cross_motion_force(v: Twist, f: Wrench) -> Wrench {
    Wrench {
        angular: v.angular.cross(f.angular) + v.linear.cross(f.linear),
        linear: v.angular.cross(f.linear),
    }
}

/// A symmetric 6x6 spatial inertia, acting on a [`Twist`] to produce a
/// [`Wrench`] (momentum, or force when the twist is an acceleration).
#[derive(Clone, PartialEq, Debug)]
pub struct SpatialInertia(Matrix);

fn skew(v: Vec3) -> Matrix {
    Matrix::from_row_major(3, 3, &[0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0])
        .expect("nine entries")
}

impl SpatialInertia {
    /// The zero inertia (a massless, inertialess point).
    #[must_use]
    pub fn zero() -> Self {
        Self(Matrix::zeros(6, 6))
    }

    /// Wraps an already-assembled 6x6 matrix directly, for the
    /// multi-DOF articulated-body algebra in `crate::articulated` (the
    /// `U * D^-1 * U^T` reduction term for a joint with more than one
    /// degree of freedom, e.g. a `Free` joint, is not expressible via
    /// [`SpatialInertia::from_parts`]'s mass/COM/inertia-tensor inputs).
    #[must_use]
    pub(crate) fn from_matrix(m: Matrix) -> Self {
        Self(m)
    }

    /// Builds the spatial inertia of a rigid body about its own frame
    /// origin, from mass, centre-of-mass offset (in this frame) and
    /// rotational inertia about the centre of mass (in this frame's
    /// axes) — exactly the fields of `robot_model::link::MassProperties`.
    ///
    /// Featherstone eq. 2.63: with `c` the centre of mass relative to the
    /// reference origin, `m` the mass and `I_c` the rotational inertia
    /// about the centre of mass,
    /// `I = [[I_c + m*skew(c)*skew(c)^T, m*skew(c)], [m*skew(c)^T, m*E]]`.
    #[must_use]
    pub fn from_mass_properties(props: &MassProperties) -> Self {
        let m = props.mass.value();
        let c = props.center_of_mass;
        let ic = props.inertia.to_matrix();
        let sc = skew(c);
        // skew(c) * skew(c)^T == |c|^2 * E - outer(c, c), the parallel-axis
        // correction; computed directly via the matrix product rather than
        // that closed form, so a mistake in one doesn't mask a mistake in
        // the other when `parallel_axis_matches_the_closed_form` checks them
        // against each other.
        let sc_sc_t = sc.mul(&sc.transposed()).expect("3x3 * 3x3");
        let top_left = ic.add(&sc_sc_t.scaled(m)).expect("3x3 + 3x3");
        let top_right = sc.scaled(m);
        let bottom_left = top_right.transposed();
        let bottom_right = Matrix::identity(3).scaled(m);

        let mut out = Matrix::zeros(6, 6);
        for r in 0..3 {
            for c_ in 0..3 {
                out[(r, c_)] = top_left[(r, c_)];
                out[(r, c_ + 3)] = top_right[(r, c_)];
                out[(r + 3, c_)] = bottom_left[(r, c_)];
                out[(r + 3, c_ + 3)] = bottom_right[(r, c_)];
            }
        }
        Self(out)
    }

    /// Also usable directly from an `InertiaTensor` plus mass/COM, for
    /// tests and fixtures that don't start from a `MassProperties`.
    #[must_use]
    pub fn from_parts(mass: f64, center_of_mass: Vec3, inertia_about_com: InertiaTensor) -> Self {
        Self::from_mass_properties(&MassProperties {
            mass: robot_types::units::Kilograms(mass),
            center_of_mass,
            inertia: inertia_about_com,
        })
    }

    /// Applies the inertia to a motion vector, producing a force vector
    /// (momentum, if the input is a velocity; a force, if the input is
    /// an acceleration).
    #[must_use]
    pub fn apply(&self, v: Twist) -> Wrench {
        let out = self.0.mul(&v.to_column()).expect("6x6 * 6x1");
        Wrench::from_column(&out)
    }

    #[must_use]
    pub(crate) fn add(&self, rhs: &Self) -> Self {
        Self(self.0.add(&rhs.0).expect("6x6 + 6x6"))
    }

    #[must_use]
    pub(crate) fn sub(&self, rhs: &Self) -> Self {
        Self(self.0.sub(&rhs.0).expect("6x6 - 6x6"))
    }

    /// Transforms a spatial inertia expressed about a child frame's
    /// origin into the same inertia expressed about the parent frame's
    /// origin, given `parent_from_child`.
    ///
    /// Computed by congruence: the transformed inertia is the unique
    /// linear operator whose action on every parent-frame twist matches
    /// transforming that twist into the child frame, applying the
    /// original inertia there, and transforming the resulting wrench
    /// back — evaluated column-by-column on the six standard basis
    /// twists rather than assembled from a closed-form 6x6 formula, so
    /// this reuses [`transform_motion`]/[`transform_force`] instead of
    /// re-deriving (and risking re-breaking) the same algebra twice.
    #[must_use]
    pub(crate) fn transform_to_parent(&self, parent_from_child: &RigidPose) -> Self {
        let basis = [
            Twist {
                angular: Vec3::new(1.0, 0.0, 0.0),
                linear: Vec3::ZERO,
            },
            Twist {
                angular: Vec3::new(0.0, 1.0, 0.0),
                linear: Vec3::ZERO,
            },
            Twist {
                angular: Vec3::new(0.0, 0.0, 1.0),
                linear: Vec3::ZERO,
            },
            Twist {
                angular: Vec3::ZERO,
                linear: Vec3::new(1.0, 0.0, 0.0),
            },
            Twist {
                angular: Vec3::ZERO,
                linear: Vec3::new(0.0, 1.0, 0.0),
            },
            Twist {
                angular: Vec3::ZERO,
                linear: Vec3::new(0.0, 0.0, 1.0),
            },
        ];
        let mut out = Matrix::zeros(6, 6);
        for (col, v_parent) in basis.into_iter().enumerate() {
            let v_child = transform_motion(parent_from_child, v_parent);
            let f_child = self.apply(v_child);
            let f_parent = transform_force(parent_from_child, f_child);
            let column = f_parent.to_column();
            for row in 0..6 {
                out[(row, col)] = column[(row, 0)];
            }
        }
        Self(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_types::frames::Quat;
    use robot_types::units::Radians;

    fn pose(translation: Vec3, rotation: Quat) -> RigidPose {
        RigidPose::new(translation, rotation)
    }

    #[test]
    fn spatial_duality_preserves_power() {
        let t = pose(
            Vec3::new(0.3, -0.1, 0.5),
            Quat::from_rx(Radians(0.7)).unwrap(),
        );
        let v_parent = Twist {
            angular: Vec3::new(0.1, 0.2, -0.3),
            linear: Vec3::new(1.0, -2.0, 0.5),
        };
        let f_child = Wrench {
            angular: Vec3::new(0.4, 0.1, 0.0),
            linear: Vec3::new(-0.2, 0.3, 0.6),
        };
        let v_child = transform_motion(&t, v_parent);
        let lhs = v_child.angular.dot(f_child.angular) + v_child.linear.dot(f_child.linear);

        let f_parent = transform_force(&t, f_child);
        let rhs = v_parent.angular.dot(f_parent.angular) + v_parent.linear.dot(f_parent.linear);
        assert!((lhs - rhs).abs() < 1e-9, "lhs={lhs} rhs={rhs}");
    }

    #[test]
    fn transform_motion_round_trips_through_its_own_inverse() {
        let t = pose(
            Vec3::new(1.0, 0.0, -0.5),
            Quat::from_ry(Radians(1.1)).unwrap(),
        );
        let v = Twist {
            angular: Vec3::new(0.2, -0.4, 0.1),
            linear: Vec3::new(2.0, 1.0, -1.0),
        };
        let v_child = transform_motion(&t, v);
        // parent_from_child composed with child_from_parent (t's inverse) is identity.
        let back = transform_motion(&t.inverse(), v_child);
        assert!((back.angular - v.angular).norm() < 1e-9);
        assert!((back.linear - v.linear).norm() < 1e-9);
    }

    #[test]
    fn parallel_axis_matches_the_closed_form() {
        let m = 2.5;
        let c = Vec3::new(0.3, -0.1, 0.2);
        let ic = InertiaTensor::diagonal(0.01, 0.02, 0.03);
        let inertia = SpatialInertia::from_parts(m, c, ic);

        // Hand-computed top-left block via the closed form I_c + m(|c|^2 E - c c^T).
        let c2 = c.norm_squared();
        let expected_top_left = [
            [0.01 + m * (c2 - c.x * c.x), -m * c.x * c.y, -m * c.x * c.z],
            [-m * c.y * c.x, 0.02 + m * (c2 - c.y * c.y), -m * c.y * c.z],
            [-m * c.z * c.x, -m * c.z * c.y, 0.03 + m * (c2 - c.z * c.z)],
        ];
        for (r, row) in expected_top_left.iter().enumerate() {
            for (col, want) in row.iter().enumerate() {
                assert!(
                    (inertia.0[(r, col)] - want).abs() < 1e-9,
                    "r={r} col={col} got={} want={want}",
                    inertia.0[(r, col)],
                );
            }
        }
        // Bottom-right block is m * identity.
        for r in 0..3 {
            for col in 0..3 {
                let want = if r == col { m } else { 0.0 };
                assert!((inertia.0[(r + 3, col + 3)] - want).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn a_rigid_inertia_computes_known_kinetic_energy() {
        // Point mass at distance L from the reference origin, spinning
        // about an axis through the origin perpendicular to the offset:
        // KE = 0.5 * m * (omega * L)^2 (pure circular motion), independent
        // of this module's own formulas for I.
        let m = 1.3;
        let l = 0.7;
        let omega = 2.0;
        let c = Vec3::new(l, 0.0, 0.0);
        let inertia = SpatialInertia::from_parts(m, c, InertiaTensor::ZERO);
        let v = Twist {
            angular: Vec3::new(0.0, omega, 0.0),
            linear: Vec3::ZERO,
        };
        let momentum = inertia.apply(v);
        let ke = 0.5 * momentum.dot(v);
        let expected = 0.5 * m * (omega * l).powi(2);
        assert!((ke - expected).abs() < 1e-9, "ke={ke} expected={expected}");
    }

    #[test]
    fn transforming_an_inertia_to_its_own_frame_is_a_no_op() {
        let inertia = SpatialInertia::from_parts(
            1.0,
            Vec3::new(0.1, 0.2, -0.1),
            InertiaTensor::diagonal(0.01, 0.02, 0.03),
        );
        let identity = RigidPose {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        };
        let transformed = inertia.transform_to_parent(&identity);
        for i in 0..6 {
            for j in 0..6 {
                assert!((transformed.0[(i, j)] - inertia.0[(i, j)]).abs() < 1e-9);
            }
        }
    }
}

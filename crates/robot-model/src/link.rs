//! Link mass properties and collision geometry.
//!
//! SPECS.md Section 7.1: "Link masses, center of mass, inertia, collision
//! meshes and available articulated degrees of freedom." A mesh is referenced
//! by content identity rather than embedded, consistent with Section 7.5's
//! content-addressed blobs — the profile names the mesh it needs, the artifact
//! store holds the bytes.

use crate::eig::{eig3, Eig3};
use crate::pose::RigidPose;
use core::fmt;
use robot_types::frames::Vec3;
use robot_types::ids::{LinkId, Name};
use robot_types::units::Kilograms;

/// A symmetric 3x3 inertia tensor about the link's centre of mass, expressed
/// in the link's own frame.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct InertiaTensor {
    /// Diagonal moments.
    pub ixx: f64,
    /// Diagonal moment.
    pub iyy: f64,
    /// Diagonal moment.
    pub izz: f64,
    /// Off-diagonal product of inertia.
    pub ixy: f64,
    /// Off-diagonal product of inertia.
    pub ixz: f64,
    /// Off-diagonal product of inertia.
    pub iyz: f64,
}

impl InertiaTensor {
    /// The tensor of a point mass, which has none.
    pub const ZERO: Self = Self {
        ixx: 0.0,
        iyy: 0.0,
        izz: 0.0,
        ixy: 0.0,
        ixz: 0.0,
        iyz: 0.0,
    };

    /// A diagonal tensor with zero products of inertia.
    #[must_use]
    pub const fn diagonal(ixx: f64, iyy: f64, izz: f64) -> Self {
        Self {
            ixx,
            iyy,
            izz,
            ixy: 0.0,
            ixz: 0.0,
            iyz: 0.0,
        }
    }

    /// Whether every component is finite.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        [self.ixx, self.iyy, self.izz, self.ixy, self.ixz, self.iyz]
            .iter()
            .all(|v| v.is_finite())
    }

    /// The principal moments of inertia: the eigenvalues of the tensor, sorted
    /// decreasing, with the rotation into the principal frame.
    ///
    /// This is the example's `mjuu_fullInertia` step
    /// (`docs/example-system-specs/13_model_compiler_setconst.md` section 5,
    /// item 5), which turns a full tensor into a principal diagonal and an
    /// `iquat` before any validity test is applied.
    #[must_use]
    pub fn principal_moments(&self) -> Eig3 {
        eig3([
            self.ixx, self.ixy, self.ixz, self.ixy, self.iyy, self.iyz, self.ixz, self.iyz,
            self.izz,
        ])
    }

    /// Whether the tensor satisfies the necessary conditions for a physically
    /// realizable rigid body: its **principal moments** are non-negative and
    /// satisfy the triangle inequality `I_a + I_b >= I_c` for every
    /// permutation.
    ///
    /// The condition is applied to the eigenvalues, not to the diagonal
    /// entries, following the example's compiler: it diagonalizes a full
    /// inertia tensor and then tests `I0 + I1 < I2 || I0 + I2 < I1 ||
    /// I1 + I2 < I0` on the principal moments
    /// (`13_model_compiler_setconst.md` section 5, items 5 and 10). The two
    /// conditions coincide only for an already-diagonal tensor. Testing the
    /// diagonal entries accepts, for instance,
    /// `ixx = iyy = izz = 1` with `ixy = 2`, whose eigenvalues are `3`, `1`
    /// and `-1`: a body with negative kinetic energy about one axis. That is
    /// the measured counterexample in
    /// `crates/robot-sim/specs/appendix-d-mathematical-challenges.md` entry
    /// M-03, and `crates/robot-sim/specs/appendix-b-open-questions.md` S-08 is
    /// the decision this implements.
    ///
    /// Still necessary rather than sufficient: it does not test the joint
    /// realisability of mass, centre of mass and tensor together (the
    /// positive-semi-definiteness of the 4x4 pseudo-inertia), which S-08
    /// records as the stronger alternative and leaves for its own decision. A
    /// fixture should still be checked against a measured or CAD-derived
    /// tensor before being trusted.
    ///
    /// **One deviation from the example, deliberate.** The example applies the
    /// triangle inequality "exact, no tolerance", because its compiler tests
    /// values it has just produced in the same routine. Here the eigenvalues
    /// come from a Jacobi iteration with a relative error of order `1e-13`,
    /// and physically ordinary bodies sit exactly on the boundary — a thin rod
    /// has `I = (0, I, I)`, for which `0 + I >= I` is an equality. An exact
    /// test would reject such a body on rounding alone, so the inequality is
    /// applied with a tolerance relative to the largest principal moment.
    /// Non-negativity carries the same relative tolerance, for the same
    /// reason, and an all-zero tensor (a point mass, and
    /// [`MassProperties::MASSLESS`]) therefore stays valid.
    #[must_use]
    pub fn is_physically_plausible(&self) -> bool {
        if !self.is_finite() {
            return false;
        }
        let [a, b, c] = self.principal_moments().values;
        // Relative to the largest moment, which is `a` because the
        // eigenvalues come back sorted decreasing.
        let tol = 1e-9 * a.abs().max(1.0e-300);
        a >= -tol
            && b >= -tol
            && c >= -tol
            && a + b >= c - tol
            && b + c >= a - tol
            && c + a >= b - tol
    }

    /// As a row-major 3x3 matrix, for use in dynamics.
    #[must_use]
    pub fn to_matrix(self) -> robot_types::math::Matrix {
        robot_types::math::Matrix::from_row_major(
            3,
            3,
            &[
                self.ixx, self.ixy, self.ixz, self.ixy, self.iyy, self.iyz, self.ixz, self.iyz,
                self.izz,
            ],
        )
        .expect("3x3 inertia matrix has nine entries")
    }
}

/// Mass, centre of mass and inertia of a link.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MassProperties {
    /// Mass.
    pub mass: Kilograms,
    /// Centre of mass, in the link's own frame.
    pub center_of_mass: Vec3,
    /// Inertia tensor about the centre of mass, in the link's own frame.
    pub inertia: InertiaTensor,
}

impl MassProperties {
    /// A massless link, for a purely kinematic attachment point.
    pub const MASSLESS: Self = Self {
        mass: Kilograms::ZERO,
        center_of_mass: Vec3::ZERO,
        inertia: InertiaTensor::ZERO,
    };

    /// Whether this is physically well-formed.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.mass.is_finite()
            && self.mass.value() >= 0.0
            && self.center_of_mass.is_finite()
            && self.inertia.is_physically_plausible()
    }
}

/// A shape used for collision (and, where undistinguished, visual) geometry.
///
/// SPECS.md Section 7.1 requires storing "collision meshes"; a referenced mesh
/// is named by content identity text rather than resolved here, since mesh data
/// belongs in the content-addressed blob store of `robot-artifacts`, not in the
/// profile.
#[derive(Clone, PartialEq, Debug)]
pub enum CollisionGeometry {
    /// A sphere of the given radius, centred at the shape's origin.
    Sphere {
        /// Radius in metres.
        radius: f64,
    },
    /// An axis-aligned box with the given half-extents.
    Box {
        /// Half-extent along each local axis, in metres.
        half_extents: Vec3,
    },
    /// A cylinder along the local Z axis.
    Cylinder {
        /// Radius in metres.
        radius: f64,
        /// Length in metres.
        length: f64,
    },
    /// A reference to an externally stored mesh, by content identity.
    Mesh {
        /// Content identity text, e.g. `sha256:...`.
        content_id: String,
        /// Uniform or per-axis scale applied to the mesh.
        scale: Vec3,
    },
}

impl CollisionGeometry {
    /// Whether the shape's own parameters are finite and positive where
    /// required.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        match self {
            Self::Sphere { radius } => radius.is_finite() && *radius > 0.0,
            Self::Box { half_extents } => {
                half_extents.is_finite()
                    && half_extents.x > 0.0
                    && half_extents.y > 0.0
                    && half_extents.z > 0.0
            }
            Self::Cylinder { radius, length } => {
                radius.is_finite() && *radius > 0.0 && length.is_finite() && *length > 0.0
            }
            Self::Mesh { content_id, scale } => !content_id.trim().is_empty() && scale.is_finite(),
        }
    }

    /// The canonical shape-kind identifier.
    #[must_use]
    pub const fn kind_id(&self) -> &'static str {
        match self {
            Self::Sphere { .. } => "sphere",
            Self::Box { .. } => "box",
            Self::Cylinder { .. } => "cylinder",
            Self::Mesh { .. } => "mesh",
        }
    }
}

/// A placed collision shape: geometry plus the pose of its own frame relative
/// to the owning link's frame.
#[derive(Clone, PartialEq, Debug)]
pub struct CollisionShape {
    /// The geometry.
    pub geometry: CollisionGeometry,
    /// Pose of the shape's local frame relative to the link's own frame.
    pub pose: RigidPose,
}

/// One link of the kinematic tree.
#[derive(Clone, PartialEq, Debug)]
pub struct LinkSpec {
    /// Stable identifier.
    pub id: LinkId,
    /// Stable name.
    pub name: Name,
    /// Mass properties.
    pub mass: MassProperties,
    /// Collision shapes attached to this link.
    pub collision: Vec<CollisionShape>,
}

/// Why a link specification was rejected.
#[derive(Clone, PartialEq, Debug)]
pub enum LinkError {
    /// The mass properties are not physically valid.
    InvalidMassProperties {
        /// The link.
        link: Name,
    },
    /// A collision shape's parameters are invalid.
    InvalidCollisionShape {
        /// The link.
        link: Name,
        /// Index of the shape.
        index: usize,
    },
    /// A collision shape's pose is not finite.
    InvalidCollisionPose {
        /// The link.
        link: Name,
        /// Index of the shape.
        index: usize,
    },
}

impl fmt::Display for LinkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMassProperties { link } => {
                write!(f, "link {link} has invalid mass properties")
            }
            Self::InvalidCollisionShape { link, index } => {
                write!(
                    f,
                    "link {link} collision shape {index} has invalid parameters"
                )
            }
            Self::InvalidCollisionPose { link, index } => {
                write!(
                    f,
                    "link {link} collision shape {index} has a non-finite pose"
                )
            }
        }
    }
}

impl core::error::Error for LinkError {}

impl LinkSpec {
    /// Checks this link's internal consistency.
    pub fn validate(&self) -> Result<(), LinkError> {
        if !self.mass.is_valid() {
            return Err(LinkError::InvalidMassProperties {
                link: self.name.clone(),
            });
        }
        for (i, shape) in self.collision.iter().enumerate() {
            if !shape.pose.is_finite() {
                return Err(LinkError::InvalidCollisionPose {
                    link: self.name.clone(),
                    index: i,
                });
            }
            if !shape.geometry.is_valid() {
                return Err(LinkError::InvalidCollisionShape {
                    link: self.name.clone(),
                    index: i,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_types::frames::Quat;
    use robot_types::units::Radians;

    fn link(mass: MassProperties, collision: Vec<CollisionShape>) -> LinkSpec {
        LinkSpec {
            id: LinkId::new(0),
            name: Name::new("forearm").unwrap(),
            mass,
            collision,
        }
    }

    #[test]
    fn a_massless_link_is_valid() {
        assert!(link(MassProperties::MASSLESS, Vec::new())
            .validate()
            .is_ok());
    }

    #[test]
    fn negative_mass_is_rejected() {
        let mut m = MassProperties::MASSLESS;
        m.mass = Kilograms(-1.0);
        assert!(matches!(
            link(m, Vec::new()).validate(),
            Err(LinkError::InvalidMassProperties { .. })
        ));
    }

    #[test]
    fn an_inertia_tensor_violating_the_triangle_inequality_is_implausible() {
        // A tensor with one moment far larger than the sum of the other two
        // cannot belong to any physical rigid body.
        let bad = InertiaTensor::diagonal(100.0, 0.1, 0.1);
        assert!(!bad.is_physically_plausible());
        let good = InertiaTensor::diagonal(1.0, 1.0, 1.0);
        assert!(good.is_physically_plausible());
    }

    #[test]
    fn negative_diagonal_inertia_is_implausible() {
        assert!(!InertiaTensor::diagonal(-1.0, 1.0, 1.0).is_physically_plausible());
    }

    #[test]
    fn collision_geometry_rejects_nonpositive_dimensions() {
        assert!(!CollisionGeometry::Sphere { radius: 0.0 }.is_valid());
        assert!(!CollisionGeometry::Sphere { radius: -1.0 }.is_valid());
        assert!(CollisionGeometry::Sphere { radius: 0.05 }.is_valid());
        assert!(!CollisionGeometry::Cylinder {
            radius: 1.0,
            length: 0.0
        }
        .is_valid());
        assert!(CollisionGeometry::Box {
            half_extents: Vec3::new(0.1, 0.1, 0.1)
        }
        .is_valid());
        assert!(!CollisionGeometry::Box {
            half_extents: Vec3::new(0.0, 0.1, 0.1)
        }
        .is_valid());
    }

    #[test]
    fn a_mesh_reference_needs_a_nonempty_content_id() {
        assert!(!CollisionGeometry::Mesh {
            content_id: String::new(),
            scale: Vec3::new(1.0, 1.0, 1.0)
        }
        .is_valid());
        assert!(CollisionGeometry::Mesh {
            content_id: "sha256:abc".to_owned(),
            scale: Vec3::new(1.0, 1.0, 1.0)
        }
        .is_valid());
    }

    #[test]
    fn link_validation_checks_every_collision_shape() {
        let mut mp = MassProperties::MASSLESS;
        mp.mass = Kilograms(1.0);
        let good = CollisionShape {
            geometry: CollisionGeometry::Sphere { radius: 0.1 },
            pose: RigidPose::IDENTITY,
        };
        let bad = CollisionShape {
            geometry: CollisionGeometry::Sphere { radius: -1.0 },
            pose: RigidPose::IDENTITY,
        };
        assert!(link(mp, vec![good.clone()]).validate().is_ok());
        assert!(matches!(
            link(mp, vec![good, bad]).validate(),
            Err(LinkError::InvalidCollisionShape { index: 1, .. })
        ));
    }

    #[test]
    fn an_indefinite_tensor_is_rejected_even_though_its_diagonal_passes() {
        // Appendix D M-03's measured counterexample. The diagonal entries are
        // all 1 and satisfy every triangle inequality among themselves, but the
        // products of inertia make the tensor indefinite: its eigenvalues are
        // 3, 1 and -1, so a body carrying it has negative kinetic energy about
        // one axis.
        let bad = InertiaTensor {
            ixx: 1.0,
            iyy: 1.0,
            izz: 1.0,
            ixy: 2.0,
            ixz: 0.0,
            iyz: 0.0,
        };
        // The old diagonal-only conditions, spelled out, all hold:
        assert!(bad.ixx >= 0.0 && bad.iyy >= 0.0 && bad.izz >= 0.0);
        assert!(bad.ixx + bad.iyy >= bad.izz);
        assert!(bad.iyy + bad.izz >= bad.ixx);
        assert!(bad.izz + bad.ixx >= bad.iyy);
        // and the principal moments are what reject it.
        let m = bad.principal_moments().values;
        assert!(m[2] < -0.5, "smallest principal moment is {}", m[2]);
        assert!(!bad.is_physically_plausible());
    }

    #[test]
    fn a_valid_tensor_with_large_products_of_inertia_is_accepted() {
        // The check must not reject a legitimate tensor merely for having
        // off-diagonal terms, so build one that is valid by construction: take
        // principal moments that satisfy the triangle inequality strictly and
        // rotate them 45 degrees about z, which is a similarity transform and
        // cannot change validity.
        //
        // Worth recording why this test is built rather than hand-picked. The
        // first tensor written here by hand -- diagonal (4, 3, 2) with products
        // (1, 0.5, -0.25), which looks entirely ordinary -- turned out to have
        // principal moments (4.65, 2.69, 1.66), and 2.69 + 1.66 < 4.65. No
        // rigid body has those moments. It passes the old diagonal-entry test.
        // That is the same failure mode as Appendix D M-03, found by accident
        // while testing the fix for it.
        for (ia, ib) in [(3.0, 2.0), (5.0, 4.0), (1.0, 0.75)] {
            // diag(ia, ib, ib) rotated by 45 degrees about z.
            let mix = 0.5 * (ia + ib);
            let off = 0.5 * (ib - ia);
            let good = InertiaTensor {
                ixx: mix,
                iyy: mix,
                izz: ib,
                ixy: off,
                ixz: 0.0,
                iyz: 0.0,
            };
            assert!(
                off.abs() > 0.1 * mix,
                "the off-diagonal term should be substantial: {off} against {mix}"
            );
            let m = good.principal_moments().values;
            assert!(m[2] > 0.0, "principal moments {m:?}");
            assert!(m[1] + m[2] >= m[0], "triangle inequality on {m:?}");
            assert!(good.is_physically_plausible());
        }
    }

    #[test]
    fn a_massless_tensor_stays_valid() {
        // MassProperties::MASSLESS is used throughout the fixtures and for
        // point masses; an all-zero tensor must remain acceptable.
        assert!(InertiaTensor::ZERO.is_physically_plausible());
        assert!(MassProperties::MASSLESS.inertia.is_physically_plausible());
    }

    #[test]
    fn a_thin_rod_on_the_triangle_boundary_stays_valid() {
        // A rod about its own axis has principal moments (0, I, I), for which
        // 0 + I >= I holds with exact equality. This is the case that makes the
        // example's "exact, no tolerance" test unusable on eigenvalues computed
        // by iteration, and the reason the check carries a relative tolerance.
        for i in [1.0, 1e-7, 5.9e-2] {
            let rod = InertiaTensor {
                ixx: 0.0,
                iyy: i,
                izz: i,
                ixy: 0.0,
                ixz: 0.0,
                iyz: 0.0,
            };
            assert!(rod.is_physically_plausible(), "rod with I = {i}");
            // The same rod, rotated into a frame where it is all off-diagonal,
            // must also pass: validity cannot depend on the frame.
            let r = core::f64::consts::FRAC_1_SQRT_2;
            // Rotating diag(0, i, i) by 45 degrees about x leaves it diagonal,
            // so rotate about z instead, which mixes xx, yy and xy.
            let (c, s) = (r, r);
            let rotated = InertiaTensor {
                ixx: c * c * 0.0 + s * s * i,
                iyy: s * s * 0.0 + c * c * i,
                izz: i,
                ixy: c * s * (0.0 - i),
                ixz: 0.0,
                iyz: 0.0,
            };
            assert!(
                rotated.is_physically_plausible(),
                "rotated rod with I = {i}"
            );
        }
    }

    #[test]
    fn a_tensor_violating_the_triangle_inequality_on_its_eigenvalues_is_rejected() {
        // Diagonal and positive, but I0 + I1 < I2: no rigid body has these
        // principal moments.
        let bad = InertiaTensor {
            ixx: 1.0,
            iyy: 1.0,
            izz: 5.0,
            ixy: 0.0,
            ixz: 0.0,
            iyz: 0.0,
        };
        assert!(!bad.is_physically_plausible());
    }

    #[test]
    fn validity_does_not_depend_on_the_frame_the_tensor_is_expressed_in() {
        // Rotating a tensor is a similarity transform and leaves its
        // eigenvalues alone, so the verdict must be invariant. This is the
        // property the diagonal-entry test lacked, and it is what makes the
        // check meaningful for an imported model whose link frames are
        // arbitrary.
        let base = InertiaTensor {
            ixx: 0.05,
            iyy: 0.07,
            izz: 0.04,
            ixy: 0.0,
            ixz: 0.0,
            iyz: 0.0,
        };
        assert!(base.is_physically_plausible());
        let before = base.principal_moments().values;
        // R I R' for a rotation by 0.7 about the axis (1, 2, -1), built from
        // the quaternion so this test does not depend on any matrix helper.
        let q = Quat::from_axis_angle(Vec3::new(1.0, 2.0, -1.0), Radians(0.7)).unwrap();
        let col = |v: Vec3| {
            let r = q.rotate(v);
            [r.x, r.y, r.z]
        };
        let r = [
            col(Vec3::new(1.0, 0.0, 0.0)),
            col(Vec3::new(0.0, 1.0, 0.0)),
            col(Vec3::new(0.0, 0.0, 1.0)),
        ];
        // R has the rotated basis vectors as its columns: R[i][j] = r[j][i].
        let i_local = [
            [base.ixx, base.ixy, base.ixz],
            [base.ixy, base.iyy, base.iyz],
            [base.ixz, base.iyz, base.izz],
        ];
        let mut out = [[0.0f64; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                let mut acc = 0.0;
                for k in 0..3 {
                    for l in 0..3 {
                        acc += r[k][a] * i_local[k][l] * r[l][b];
                    }
                }
                out[a][b] = acc;
            }
        }
        let rotated = InertiaTensor {
            ixx: out[0][0],
            iyy: out[1][1],
            izz: out[2][2],
            ixy: out[0][1],
            ixz: out[0][2],
            iyz: out[1][2],
        };
        let after = rotated.principal_moments().values;
        for (x, y) in before.iter().zip(&after) {
            assert!(
                (x - y).abs() < 1e-12,
                "principal moments changed under rotation: {before:?} then {after:?}"
            );
        }
        assert!(rotated.is_physically_plausible());
    }

    #[test]
    fn every_link_of_the_vendored_g1_has_a_physically_valid_inertia() {
        // The stricter condition must not reject the real robot. It does not:
        // every one of the G1's links passes on its principal moments, which
        // is worth pinning because the check became stronger.
        for fixture in [
            crate::fixtures::unitree_g1_29dof(),
            crate::fixtures::unitree_g1_29dof_floating_base(),
        ] {
            let profile = fixture.expect("the vendored fixture loads");
            for link in &profile.links {
                assert!(
                    link.mass.inertia.is_physically_plausible(),
                    "link {:?} has principal moments {:?}",
                    link.name,
                    link.mass.inertia.principal_moments().values
                );
            }
        }
    }
}

//! Link mass properties and collision geometry.
//!
//! SPECS.md Section 7.1: "Link masses, center of mass, inertia, collision
//! meshes and available articulated degrees of freedom." A mesh is referenced
//! by content identity rather than embedded, consistent with Section 7.5's
//! content-addressed blobs — the profile names the mesh it needs, the artifact
//! store holds the bytes.

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

    /// Whether the tensor satisfies the necessary conditions for a physically
    /// realizable rigid body: non-negative diagonal entries and the triangle
    /// inequality `I_a + I_b >= I_c` for every permutation of the principal
    /// moments.
    ///
    /// This is a necessary, not sufficient, check — it does not diagonalize
    /// the tensor to confirm the *principal* moments satisfy the inequality
    /// when products of inertia are large, which is why a profile fixture
    /// should also be checked against a measured or CAD-derived tensor before
    /// being trusted.
    #[must_use]
    pub fn is_physically_plausible(&self) -> bool {
        self.is_finite()
            && self.ixx >= 0.0
            && self.iyy >= 0.0
            && self.izz >= 0.0
            && self.ixx + self.iyy >= self.izz
            && self.iyy + self.izz >= self.ixx
            && self.izz + self.ixx >= self.iyy
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
}

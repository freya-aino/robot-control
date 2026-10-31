//! Builds a [`RobotProfile`] from an OpenUSD robot description, read with
//! [`robot_usd`] (ADR 0015).
//!
//! The description is the Unitree G1 model, which Unitree publishes for Isaac
//! Sim, and it uses the USD physics schemas:
//!
//! - each link is a child of the default prim that carries `physics:mass`,
//!   `physics:centerOfMass`, `physics:diagonalInertia` and `physics:principalAxes`;
//! - each joint is a `PhysicsRevoluteJoint` under `joints/`, with
//!   `physics:body0` and `physics:body1` relationships, the joint frame in each
//!   body (`physics:localPos0/1`, `physics:localRot0/1`), `physics:axis`, and
//!   `physics:lowerLimit` and `physics:upperLimit` in degrees;
//! - the drive of each joint (`drive:angular:physics:*`) gives the effort limit,
//!   the stiffness and the damping, and `physxJoint:maxJointVelocity` gives the
//!   velocity limit in degrees per second;
//! - collision shapes are `Sphere`, `Cube`, `Cylinder` and `Mesh` prims under
//!   `<link>/collisions`, with their pose in the link frame given by `xformOp`s.
//!
//! Anything outside this subset is refused by name with
//! [`UsdImportError::Unsupported`], never dropped.
//!
//! Quaternions are read as the crate stores them: `(x, y, z, w)`.

use crate::actuator::{ActuatorSpec, PdGains};
use crate::joint::{JointKind, JointLimits, JointSpec};
use crate::link::{CollisionGeometry, CollisionShape, InertiaTensor, LinkSpec, MassProperties};
use crate::pose::RigidPose;
use crate::profile::{FirmwareContract, ProfileError, RobotProfile, PROFILE_SCHEMA_VERSION};
use crate::urdf::placeholder_actuator;
use core::fmt;
use robot_types::frames::{FrameId, GravityConvention, Quat, Vec3};
use robot_types::ids::{JointId, LinkId, Name, NameError};
use robot_types::schema::ExtensionTable;
use robot_types::time::Duration;
use robot_types::units::{Kilograms, NewtonMetres, Radians, RadiansPerSecond, Range};
use robot_usd::stage::{Prim, Stage};
use robot_usd::value::Value;
use robot_usd::UsdError;
use std::path::Path;

/// Counts of what an import read, so that nothing is dropped unseen.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ImportReport {
    /// Rigid links (prims with `physics:mass`), excluding the `world` link.
    pub links: usize,
    /// Revolute joints.
    pub joints: usize,
    /// Collision spheres.
    pub collision_spheres: usize,
    /// Collision boxes (`Cube` prims).
    pub collision_boxes: usize,
    /// Collision cylinders.
    pub collision_cylinders: usize,
    /// Collision meshes, kept as references to the USD mesh prim.
    pub collision_meshes: usize,
    /// Visual mesh prims, kept as references and not used by the physics.
    pub visual_meshes: usize,
}

/// Why a USD robot description could not be imported.
#[derive(Debug)]
pub enum UsdImportError {
    /// The layers could not be read or composed.
    Usd(UsdError),
    /// A required prim or attribute was absent.
    Missing {
        /// The prim path.
        path: String,
        /// The missing part.
        what: &'static str,
    },
    /// An attribute had a type or value this importer does not accept.
    BadValue {
        /// The prim path.
        path: String,
        /// The attribute.
        what: &'static str,
    },
    /// A construct outside the supported subset.
    Unsupported {
        /// The construct's name.
        construct: String,
        /// The prim it appeared on.
        path: String,
    },
    /// A name failed the project's name rules.
    Name(NameError),
    /// The assembled profile failed validation.
    Profile(ProfileError),
}

impl fmt::Display for UsdImportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usd(e) => write!(f, "{e}"),
            Self::Missing { path, what } => write!(f, "{path} has no {what}"),
            Self::BadValue { path, what } => write!(f, "{path} has an unusable {what}"),
            Self::Unsupported { construct, path } => {
                write!(f, "unsupported USD construct {construct} on {path}")
            }
            Self::Name(e) => write!(f, "{e}"),
            Self::Profile(e) => write!(f, "{e}"),
        }
    }
}

impl core::error::Error for UsdImportError {}

impl From<UsdError> for UsdImportError {
    fn from(e: UsdError) -> Self {
        Self::Usd(e)
    }
}
impl From<NameError> for UsdImportError {
    fn from(e: NameError) -> Self {
        Self::Name(e)
    }
}
impl From<ProfileError> for UsdImportError {
    fn from(e: ProfileError) -> Self {
        Self::Profile(e)
    }
}

/// The attribute's default value, if it has one.
fn attr<'a>(prim: &'a Prim, name: &str) -> Option<&'a Value> {
    prim.property(name).and_then(|p| p.default.as_ref())
}

fn f64_of(v: &Value) -> Option<f64> {
    match v {
        Value::Double(d) => Some(*d),
        Value::Float(f) => Some(f64::from(*f)),
        _ => None,
    }
}

fn vec3_of(v: &Value) -> Option<[f64; 3]> {
    match v {
        Value::Vec3d(a) => Some(*a),
        Value::Vec3f(a) => Some([f64::from(a[0]), f64::from(a[1]), f64::from(a[2])]),
        _ => None,
    }
}

/// A quaternion value as `[x, y, z, w]`.
fn quat_of(v: &Value) -> Option<[f64; 4]> {
    match v {
        Value::Quatd(a) => Some(*a),
        Value::Quatf(a) => Some([
            f64::from(a[0]),
            f64::from(a[1]),
            f64::from(a[2]),
            f64::from(a[3]),
        ]),
        _ => None,
    }
}

fn quat_xyzw(a: [f64; 4]) -> Quat {
    Quat {
        w: a[3],
        x: a[0],
        y: a[1],
        z: a[2],
    }
}

fn identity_xyzw() -> [f64; 4] {
    [0.0, 0.0, 0.0, 1.0]
}

fn required_f64(prim: &Prim, name: &'static str) -> Result<f64, UsdImportError> {
    let v = attr(prim, name).ok_or_else(|| UsdImportError::Missing {
        path: prim.path.clone(),
        what: name,
    })?;
    f64_of(v).ok_or(UsdImportError::BadValue {
        path: prim.path.clone(),
        what: name,
    })
}

fn optional_f64(prim: &Prim, name: &'static str) -> Result<Option<f64>, UsdImportError> {
    match attr(prim, name) {
        None => Ok(None),
        Some(v) => f64_of(v).map(Some).ok_or(UsdImportError::BadValue {
            path: prim.path.clone(),
            what: name,
        }),
    }
}

fn optional_vec3(
    prim: &Prim,
    name: &'static str,
    default: [f64; 3],
) -> Result<[f64; 3], UsdImportError> {
    match attr(prim, name) {
        None => Ok(default),
        Some(v) => vec3_of(v).ok_or(UsdImportError::BadValue {
            path: prim.path.clone(),
            what: name,
        }),
    }
}

fn optional_quat(prim: &Prim, name: &'static str) -> Result<[f64; 4], UsdImportError> {
    match attr(prim, name) {
        None => Ok(identity_xyzw()),
        Some(v) => quat_of(v).ok_or(UsdImportError::BadValue {
            path: prim.path.clone(),
            what: name,
        }),
    }
}

/// The single target of a relationship.
fn single_target(prim: &Prim, name: &'static str) -> Result<String, UsdImportError> {
    let p = prim.property(name).ok_or_else(|| UsdImportError::Missing {
        path: prim.path.clone(),
        what: name,
    })?;
    match p.targets.as_slice() {
        [one] => Ok(one.clone()),
        _ => Err(UsdImportError::BadValue {
            path: prim.path.clone(),
            what: name,
        }),
    }
}

/// `physics:axis` as a unit vector.
fn axis_of(prim: &Prim) -> Result<Vec3, UsdImportError> {
    let token = attr(prim, "physics:axis")
        .and_then(Value::as_str)
        .ok_or_else(|| UsdImportError::Missing {
            path: prim.path.clone(),
            what: "physics:axis",
        })?;
    match token {
        "X" => Ok(Vec3::new(1.0, 0.0, 0.0)),
        "Y" => Ok(Vec3::new(0.0, 1.0, 0.0)),
        "Z" => Ok(Vec3::new(0.0, 0.0, 1.0)),
        _ => Err(UsdImportError::BadValue {
            path: prim.path.clone(),
            what: "physics:axis",
        }),
    }
}

/// The rotation matrix of a unit quaternion, row-major.
fn rotation_matrix(q: Quat) -> [[f64; 3]; 3] {
    let (w, x, y, z) = (q.w, q.x, q.y, q.z);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

/// `R diag(d) Rᵀ`: the inertia tensor of principal moments `d` whose frame is
/// rotated by `q` from the body frame.
fn tensor_from_principal(q: Quat, d: [f64; 3]) -> InertiaTensor {
    let r = rotation_matrix(q);
    let m = |i: usize, j: usize| (0..3).map(|k| r[i][k] * d[k] * r[j][k]).sum::<f64>();
    InertiaTensor {
        ixx: m(0, 0),
        iyy: m(1, 1),
        izz: m(2, 2),
        ixy: m(0, 1),
        ixz: m(0, 2),
        iyz: m(1, 2),
    }
}

fn mass_of(link: &Prim) -> Result<MassProperties, UsdImportError> {
    let mass = required_f64(link, "physics:mass")?;
    let center = optional_vec3(link, "physics:centerOfMass", [0.0; 3])?;
    let diagonal = optional_vec3(link, "physics:diagonalInertia", [0.0; 3])?;
    let axes = optional_quat(link, "physics:principalAxes")?;
    Ok(MassProperties {
        mass: Kilograms(mass),
        center_of_mass: Vec3::new(center[0], center[1], center[2]),
        inertia: tensor_from_principal(quat_xyzw(axes), diagonal),
    })
}

/// A prim's local transform as `(pose, scale)`, from its `xformOpOrder`.
///
/// Only `translate`, `orient` and `scale`, in that order, are accepted. Any
/// other op is refused, so the composition below is exact.
fn local_transform(prim: &Prim) -> Result<(RigidPose, [f64; 3]), UsdImportError> {
    let order: Vec<String> = match attr(prim, "xformOpOrder") {
        None => Vec::new(),
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .ok_or(UsdImportError::BadValue {
                        path: prim.path.clone(),
                        what: "xformOpOrder",
                    })
            })
            .collect::<Result<_, _>>()?,
        Some(_) => {
            return Err(UsdImportError::BadValue {
                path: prim.path.clone(),
                what: "xformOpOrder",
            })
        }
    };
    let allowed = ["xformOp:translate", "xformOp:orient", "xformOp:scale"];
    let mut last = 0usize;
    for name in &order {
        let Some(pos) = allowed.iter().position(|a| a == name) else {
            return Err(UsdImportError::Unsupported {
                construct: format!("xform op {name}"),
                path: prim.path.clone(),
            });
        };
        if pos < last {
            return Err(UsdImportError::Unsupported {
                construct: "xform ops out of translate-orient-scale order".to_owned(),
                path: prim.path.clone(),
            });
        }
        last = pos;
    }
    let translation = optional_vec3(prim, "xformOp:translate", [0.0; 3])?;
    let rotation = optional_quat(prim, "xformOp:orient")?;
    let scale = optional_vec3(prim, "xformOp:scale", [1.0; 3])?;
    Ok((
        RigidPose::new(
            Vec3::new(translation[0], translation[1], translation[2]),
            quat_xyzw(rotation),
        ),
        scale,
    ))
}

/// Walks a link's `collisions` subtree and returns its shapes, in the link
/// frame. Dimensions include the scale of every prim on the path.
fn collect_collisions(
    prim: &Prim,
    pose: RigidPose,
    scale: [f64; 3],
    out: &mut Vec<CollisionShape>,
    report: &mut ImportReport,
) -> Result<(), UsdImportError> {
    for child in &prim.children {
        let (local, local_scale) = local_transform(child)?;
        let child_scale = [
            scale[0] * local_scale[0],
            scale[1] * local_scale[1],
            scale[2] * local_scale[2],
        ];
        let child_pose = pose.compose(&local);
        let ty = child.type_name.as_deref().unwrap_or("");
        let shape = match ty {
            "Sphere" => {
                let radius = required_f64(child, "radius")?;
                report.collision_spheres += 1;
                Some(CollisionGeometry::Sphere {
                    radius: radius
                        * child_scale[0]
                            .abs()
                            .max(child_scale[1].abs())
                            .max(child_scale[2].abs()),
                })
            }
            "Cube" => {
                let size = optional_f64(child, "size")?.unwrap_or(2.0);
                report.collision_boxes += 1;
                Some(CollisionGeometry::Box {
                    half_extents: Vec3::new(
                        size * child_scale[0].abs() / 2.0,
                        size * child_scale[1].abs() / 2.0,
                        size * child_scale[2].abs() / 2.0,
                    ),
                })
            }
            "Cylinder" => {
                let axis = attr(child, "axis").and_then(Value::as_str).unwrap_or("Z");
                if axis != "Z" {
                    return Err(UsdImportError::Unsupported {
                        construct: format!("cylinder axis {axis}"),
                        path: child.path.clone(),
                    });
                }
                let radius = required_f64(child, "radius")?;
                let height = optional_f64(child, "height")?.unwrap_or(2.0);
                report.collision_cylinders += 1;
                Some(CollisionGeometry::Cylinder {
                    radius: radius * child_scale[0].abs().max(child_scale[1].abs()),
                    length: height * child_scale[2].abs(),
                })
            }
            "Mesh" => {
                report.collision_meshes += 1;
                Some(CollisionGeometry::Mesh {
                    content_id: child.path.clone(),
                    scale: Vec3::new(child_scale[0], child_scale[1], child_scale[2]),
                })
            }
            _ => None,
        };
        match shape {
            Some(geometry) => {
                // A scaled ancestor with a child that is displaced would need the
                // scale applied to the displacement too. The G1 has none; refuse
                // it rather than place the shape wrongly.
                let displaced = local.translation != Vec3::ZERO;
                if displaced && scale != [1.0; 3] {
                    return Err(UsdImportError::Unsupported {
                        construct: "translated collision shape under a scaled ancestor".to_owned(),
                        path: child.path.clone(),
                    });
                }
                out.push(CollisionShape {
                    geometry,
                    pose: child_pose,
                });
            }
            None => collect_collisions(child, child_pose, child_scale, out, report)?,
        }
    }
    Ok(())
}

fn count_visual_meshes(prim: &Prim) -> usize {
    prim.children
        .iter()
        .map(|c| usize::from(c.type_name.as_deref() == Some("Mesh")) + count_visual_meshes(c))
        .sum()
}

/// Imports the robot described by `root_layer`, as a fixed-base profile whose
/// root is the link with no incoming joint.
///
/// # Errors
///
/// Returns the first unreadable layer, missing or ill-typed attribute, or
/// unsupported construct. The import is all or nothing.
pub fn import_fixed_base(
    root_layer: &Path,
) -> Result<(RobotProfile, ImportReport), UsdImportError> {
    let (profile, report, _warnings) = import_fixed_base_with_warnings(root_layer)?;
    Ok((profile, report))
}

/// As [`import_fixed_base`], and also returns the non-fatal findings of the
/// composed stage, such as Omniverse-only materials (see
/// `robot_usd::diagnostics`).
///
/// # Errors
///
/// As [`import_fixed_base`].
pub fn import_fixed_base_with_warnings(
    root_layer: &Path,
) -> Result<
    (
        RobotProfile,
        ImportReport,
        Vec<robot_usd::diagnostics::Warning>,
    ),
    UsdImportError,
> {
    let stage = Stage::compose(root_layer)?;
    let warnings = robot_usd::diagnostics::shader_warnings(&stage);
    let default = stage
        .metadata
        .get("defaultPrim")
        .and_then(Value::as_str)
        .ok_or(UsdImportError::Missing {
            path: "/".to_owned(),
            what: "defaultPrim",
        })?
        .to_owned();
    let robot_path = format!("/{default}");
    let robot = stage.prim(&robot_path).ok_or(UsdImportError::Missing {
        path: robot_path.clone(),
        what: "the default prim",
    })?;

    let mut report = ImportReport::default();
    let mut links: Vec<LinkSpec> = Vec::new();
    let mut link_paths: Vec<String> = Vec::new();
    for child in &robot.children {
        if child.property("physics:mass").is_none() {
            continue;
        }
        let id = LinkId::new(links.len() as u32);
        let mass = mass_of(child)?;
        let mut collision = Vec::new();
        if let Some(scope) = child.child("collisions") {
            collect_collisions(
                scope,
                RigidPose::IDENTITY,
                [1.0; 3],
                &mut collision,
                &mut report,
            )?;
        }
        if let Some(scope) = child.child("visuals") {
            report.visual_meshes += count_visual_meshes(scope);
        }
        links.push(LinkSpec {
            id,
            name: Name::new(&child.name)?,
            mass,
            collision,
        });
        link_paths.push(child.path.clone());
    }
    report.links = links.len();

    let link_index = |path: &str| link_paths.iter().position(|p| p == path);
    let mut joints: Vec<JointSpec> = Vec::new();
    let mut actuators: Vec<ActuatorSpec> = Vec::new();
    if let Some(scope) = robot.child("joints") {
        for joint in &scope.children {
            if joint.type_name.as_deref() != Some("PhysicsRevoluteJoint") {
                return Err(UsdImportError::Unsupported {
                    construct: format!(
                        "joint type {}",
                        joint.type_name.as_deref().unwrap_or("(none)")
                    ),
                    path: joint.path.clone(),
                });
            }
            let body0 = single_target(joint, "physics:body0")?;
            let body1 = single_target(joint, "physics:body1")?;
            let parent = link_index(&body0).ok_or_else(|| UsdImportError::Missing {
                path: joint.path.clone(),
                what: "a link for body0",
            })?;
            let child = link_index(&body1).ok_or_else(|| UsdImportError::Missing {
                path: joint.path.clone(),
                what: "a link for body1",
            })?;

            // The joint frame is `body0 * localPos0` and `body1 * localPos1`.
            // The child's pose relative to the parent at zero is their ratio.
            let frame0 = RigidPose::new(
                {
                    let t = optional_vec3(joint, "physics:localPos0", [0.0; 3])?;
                    Vec3::new(t[0], t[1], t[2])
                },
                quat_xyzw(optional_quat(joint, "physics:localRot0")?),
            );
            let frame1 = RigidPose::new(
                {
                    let t = optional_vec3(joint, "physics:localPos1", [0.0; 3])?;
                    Vec3::new(t[0], t[1], t[2])
                },
                quat_xyzw(optional_quat(joint, "physics:localRot1")?),
            );
            let origin = frame0.compose(&frame1.inverse());

            let lower = required_f64(joint, "physics:lowerLimit")?.to_radians();
            let upper = required_f64(joint, "physics:upperLimit")?.to_radians();
            let velocity = optional_f64(joint, "physxJoint:maxJointVelocity")?
                .map(f64::to_radians)
                .ok_or(UsdImportError::Missing {
                    path: joint.path.clone(),
                    what: "physxJoint:maxJointVelocity",
                })?;
            let effort = required_f64(joint, "drive:angular:physics:maxForce")?;
            let stiffness = required_f64(joint, "drive:angular:physics:stiffness")?;
            let damping = required_f64(joint, "drive:angular:physics:damping")?;

            let id = JointId::new(joints.len() as u32);
            let spec = JointSpec {
                id,
                name: Name::new(&joint.name)?,
                kind: JointKind::Revolute,
                parent: LinkId::new(parent as u32),
                child: LinkId::new(child as u32),
                origin,
                axis: axis_of(joint)?,
                limits: JointLimits::bounded(
                    Range {
                        lo: Radians(lower),
                        hi: Radians(upper),
                    },
                    RadiansPerSecond(velocity),
                    NewtonMetres(effort),
                ),
                sign: 1.0,
                zero_offset: 0.0,
            };
            let mut actuator = placeholder_actuator(joints.len() as u32, &spec);
            actuator.gains = Some(PdGains {
                kp: stiffness,
                kd: damping,
            });
            joints.push(spec);
            actuators.push(ActuatorSpec {
                id: actuator.id,
                name: actuator.name,
                joint: actuator.joint,
                ..actuator
            });
        }
    }
    report.joints = joints.len();

    let roots: Vec<usize> = (0..links.len())
        .filter(|i| !joints.iter().any(|j| j.child == LinkId::new(*i as u32)))
        .collect();
    let root = match roots.as_slice() {
        [one] => *one,
        _ => {
            return Err(UsdImportError::BadValue {
                path: robot_path,
                what: "single root link (no incoming joint)",
            })
        }
    };

    let profile = RobotProfile {
        schema_version: PROFILE_SCHEMA_VERSION,
        name: Name::new("unitree-g1-29dof-usd")?,
        profile_version: "unitree_model@b6a8942b0803b6c137e58cef12beb4b03e4a2fa7".to_owned(),
        joints,
        links,
        actuators,
        sensors: Vec::new(),
        root_link: LinkId::new(root as u32),
        gravity: GravityConvention::minus_z_standard(FrameId::WORLD),
        firmware: FirmwareContract {
            version: "unspecified (no cited source)".to_owned(),
            command_expiration: Duration::ZERO,
            watchdog_timeout: Duration::ZERO,
            has_independent_estop: false,
            allowed_modes: Vec::new(),
        },
        extensions: ExtensionTable::new(),
    };
    profile.validate()?;
    Ok((profile, report, warnings))
}

/// A visual mesh of a link, with its geometry, for drawing only.
#[derive(Clone, Debug, PartialEq)]
pub struct VisualMesh {
    /// The link the mesh belongs to.
    pub link: String,
    /// The mesh prim's name.
    pub name: String,
    /// Pose of the mesh in the link frame, scale applied to the points.
    pub pose: RigidPose,
    /// Vertex positions in the mesh frame, metres, scale already applied.
    pub points: Vec<[f64; 3]>,
    /// Vertices per face.
    pub face_vertex_counts: Vec<i32>,
    /// Flattened vertex indices of the faces.
    pub face_vertex_indices: Vec<i32>,
}

fn int_array(prim: &Prim, name: &'static str) -> Result<Vec<i32>, UsdImportError> {
    let value = attr(prim, name).ok_or_else(|| UsdImportError::Missing {
        path: prim.path.clone(),
        what: name,
    })?;
    let Value::Array(items) = value else {
        return Err(UsdImportError::BadValue {
            path: prim.path.clone(),
            what: name,
        });
    };
    items
        .iter()
        .map(|v| match v {
            Value::Int(i) => Ok(*i),
            _ => Err(UsdImportError::BadValue {
                path: prim.path.clone(),
                what: name,
            }),
        })
        .collect()
}

fn point_array(prim: &Prim) -> Result<Vec<[f64; 3]>, UsdImportError> {
    let value = attr(prim, "points").ok_or_else(|| UsdImportError::Missing {
        path: prim.path.clone(),
        what: "points",
    })?;
    let Value::Array(items) = value else {
        return Err(UsdImportError::BadValue {
            path: prim.path.clone(),
            what: "points",
        });
    };
    items
        .iter()
        .map(|v| {
            vec3_of(v).ok_or(UsdImportError::BadValue {
                path: prim.path.clone(),
                what: "points",
            })
        })
        .collect()
}

fn collect_visual_meshes(
    prim: &Prim,
    link: &str,
    pose: RigidPose,
    scale: [f64; 3],
    out: &mut Vec<VisualMesh>,
) -> Result<(), UsdImportError> {
    for child in &prim.children {
        let (local, local_scale) = local_transform(child)?;
        let child_scale = [
            scale[0] * local_scale[0],
            scale[1] * local_scale[1],
            scale[2] * local_scale[2],
        ];
        let child_pose = pose.compose(&local);
        if child.type_name.as_deref() == Some("Mesh") {
            let points: Vec<[f64; 3]> = point_array(child)?
                .into_iter()
                .map(|p| {
                    [
                        p[0] * child_scale[0],
                        p[1] * child_scale[1],
                        p[2] * child_scale[2],
                    ]
                })
                .collect();
            out.push(VisualMesh {
                link: link.to_owned(),
                name: child.name.clone(),
                pose: child_pose,
                points,
                face_vertex_counts: int_array(child, "faceVertexCounts")?,
                face_vertex_indices: int_array(child, "faceVertexIndices")?,
            });
        } else {
            collect_visual_meshes(child, link, child_pose, child_scale, out)?;
        }
    }
    Ok(())
}

/// The visual meshes of every link of the robot layer, with their geometry.
///
/// These are for display. Physics does not use them.
///
/// # Errors
///
/// Returns the first unreadable layer, missing attribute, or unsupported xform.
pub fn import_visual_meshes(root_layer: &Path) -> Result<Vec<VisualMesh>, UsdImportError> {
    let stage = Stage::compose(root_layer)?;
    let default = stage
        .metadata
        .get("defaultPrim")
        .and_then(Value::as_str)
        .ok_or(UsdImportError::Missing {
            path: "/".to_owned(),
            what: "defaultPrim",
        })?
        .to_owned();
    let robot_path = format!("/{default}");
    let robot = stage.prim(&robot_path).ok_or(UsdImportError::Missing {
        path: robot_path,
        what: "the default prim",
    })?;
    let mut out = Vec::new();
    for link in &robot.children {
        if link.property("physics:mass").is_none() {
            continue;
        }
        if let Some(scope) = link.child("visuals") {
            collect_visual_meshes(scope, &link.name, RigidPose::IDENTITY, [1.0; 3], &mut out)?;
        }
    }
    Ok(out)
}

/// The ground collider of a scene: an infinite plane with a Z normal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundPlane {
    /// Height of the plane above the origin, metres (its translation along Z).
    pub height: f64,
    /// Penalty normal stiffness, N per metre of penetration (`robot:contactStiffness`).
    pub stiffness: f64,
    /// Penalty normal damping, N per metre per second (`robot:contactDamping`).
    pub normal_damping: f64,
    /// Coulomb friction coefficient (`physics:dynamicFriction`).
    pub friction: f64,
}

/// The tags a scene prim can carry, in its `sim:tags` token array.
///
/// These tags are the same in every scene and are recognised by the importer
/// wherever the prim sits in the stage. A robot is recognised by its physics
/// (`physics:mass`, joints), not by a tag. A prim tagged `environment` and
/// `collider` is simulated as a static obstacle; `environment` alone is visual
/// only; any other tag is preserved by the file and ignored here.
pub const TAG_ENVIRONMENT: &str = "environment";
/// See [`TAG_ENVIRONMENT`].
pub const TAG_COLLIDER: &str = "collider";

/// The static colliders of a scene, read from its `environment` + `collider`
/// tagged prims.
#[derive(Clone, Debug, PartialEq)]
pub struct EnvironmentScene {
    /// Infinite planes with a Z normal, one per tagged `Plane`.
    pub planes: Vec<GroundPlane>,
    /// Tagged colliders of a shape this crate cannot simulate yet, by prim path.
    pub unsupported: Vec<String>,
    /// Prims tagged `environment` but not `collider`: visual only.
    pub visual_only: usize,
}

fn tags_of(prim: &Prim) -> Vec<String> {
    attr(prim, "sim:tags")
        .and_then(|v| match v {
            Value::Array(items) => Some(
                items
                    .iter()
                    .filter_map(|t| t.as_str().map(str::to_owned))
                    .collect(),
            ),
            _ => None,
        })
        .unwrap_or_default()
}

fn walk_environment(prim: &Prim, out: &mut EnvironmentScene) -> Result<(), UsdImportError> {
    let tags = tags_of(prim);
    let environment = tags.iter().any(|t| t == TAG_ENVIRONMENT);
    let collider = tags.iter().any(|t| t == TAG_COLLIDER);
    if environment && collider {
        match prim.type_name.as_deref() {
            Some("Plane") => {
                let axis = attr(prim, "axis").and_then(Value::as_str).unwrap_or("Z");
                if axis != "Z" {
                    return Err(UsdImportError::Unsupported {
                        construct: format!("plane axis {axis}"),
                        path: prim.path.clone(),
                    });
                }
                let (pose, _scale) = local_transform(prim)?;
                if pose.rotation != quat_xyzw(identity_xyzw()) {
                    return Err(UsdImportError::Unsupported {
                        construct: "rotated plane".to_owned(),
                        path: prim.path.clone(),
                    });
                }
                out.planes.push(GroundPlane {
                    height: pose.translation.z,
                    stiffness: required_f64(prim, "sim:contactStiffness")?,
                    normal_damping: required_f64(prim, "sim:contactDamping")?,
                    friction: required_f64(prim, "physics:dynamicFriction")?,
                });
            }
            _ => out.unsupported.push(prim.path.clone()),
        }
    } else if environment {
        out.visual_only += 1;
    }
    for child in &prim.children {
        walk_environment(child, out)?;
    }
    Ok(())
}

/// Reads every environment collider of a scene layer, wherever it sits in the
/// stage.
///
/// # Errors
///
/// Returns an error if the layer cannot be composed, or a tagged plane has a
/// non-Z axis, a rotation, or a missing contact parameter.
pub fn import_environment(scene_layer: &Path) -> Result<EnvironmentScene, UsdImportError> {
    let stage = Stage::compose(scene_layer)?;
    let mut out = EnvironmentScene {
        planes: Vec::new(),
        unsupported: Vec::new(),
        visual_only: 0,
    };
    for prim in &stage.prims {
        walk_environment(prim, &mut out)?;
    }
    Ok(out)
}

/// The profile with its root attached to a massless `world` link through a
/// free joint at the origin, as [`crate::fixtures::unitree_g1_29dof_floating_base`]
/// builds it. The free joint is first in the joint list.
///
/// # Errors
///
/// Returns the profile's own validation error.
pub fn with_floating_base(mut profile: RobotProfile) -> Result<RobotProfile, ProfileError> {
    let world_link = LinkId::new(profile.links.len() as u32);
    let floating_joint = JointId::new(profile.joints.len() as u32);
    let old_root = profile.root_link;
    profile.links.push(LinkSpec {
        id: world_link,
        name: Name::new("world").expect("valid literal"),
        mass: MassProperties::MASSLESS,
        collision: Vec::new(),
    });
    profile.joints.insert(
        0,
        JointSpec {
            id: floating_joint,
            name: Name::new("floating_base").expect("valid literal"),
            kind: JointKind::Free,
            parent: world_link,
            child: old_root,
            origin: RigidPose::IDENTITY,
            axis: Vec3::ZERO,
            limits: JointLimits::unbounded(RadiansPerSecond(1000.0), NewtonMetres(0.0)),
            sign: 1.0,
            zero_offset: 0.0,
        },
    );
    profile.root_link = world_link;
    profile.validate()?;
    Ok(profile)
}

/// The mass of a link as the USD states it, for callers that compare sources.
#[must_use]
pub fn link_mass(profile: &RobotProfile, name: &str) -> Option<f64> {
    profile
        .links
        .iter()
        .find(|l| l.name.as_str() == name)
        .map(|l| l.mass.mass.value())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn principal_inertia_with_identity_axes_is_the_diagonal() {
        let t = tensor_from_principal(Quat::IDENTITY, [1.0, 2.0, 3.0]);
        assert_eq!((t.ixx, t.iyy, t.izz), (1.0, 2.0, 3.0));
        assert_eq!((t.ixy, t.ixz, t.iyz), (0.0, 0.0, 0.0));
    }

    #[test]
    fn a_quarter_turn_about_z_swaps_the_principal_moments_of_x_and_y() {
        let q = Quat {
            w: core::f64::consts::FRAC_1_SQRT_2,
            x: 0.0,
            y: 0.0,
            z: core::f64::consts::FRAC_1_SQRT_2,
        };
        let t = tensor_from_principal(q, [1.0, 2.0, 3.0]);
        assert!((t.ixx - 2.0).abs() < 1e-12, "{t:?}");
        assert!((t.iyy - 1.0).abs() < 1e-12, "{t:?}");
        assert!((t.izz - 3.0).abs() < 1e-12, "{t:?}");
    }

    #[test]
    fn quaternion_values_are_read_as_x_y_z_w() {
        let q = quat_xyzw([0.0, 0.0, 0.5, 0.866]);
        assert_eq!((q.x, q.y, q.z, q.w), (0.0, 0.0, 0.5, 0.866));
    }
}

#[cfg(test)]
mod g1 {
    use super::*;
    use crate::fixtures;

    fn layer() -> &'static Path {
        Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/robots/unitree_g1_usd/g1_29dof_rev_1_0.usd"
        ))
    }

    #[test]
    fn the_official_g1_layer_imports_with_the_measured_counts() {
        // Measured on 2026-10-30 from the vendored layer (see
        // assets/robots/unitree_g1_usd/PROVENANCE.md).
        let (profile, report) = import_fixed_base(layer()).unwrap();
        assert_eq!(
            report,
            ImportReport {
                links: 30,
                joints: 29,
                collision_spheres: 10,
                collision_boxes: 2,
                collision_cylinders: 6,
                collision_meshes: 15,
                visual_meshes: 35,
            }
        );
        assert_eq!(profile.links.len(), 30);
        assert_eq!(profile.actuators.len(), 29);
        assert_eq!(profile.root_link, profile.links[0].id);
        assert_eq!(profile.links[0].name.as_str(), "pelvis");
    }

    #[test]
    fn masses_and_centres_of_mass_are_the_stated_values() {
        let (profile, _) = import_fixed_base(layer()).unwrap();
        let pelvis = profile
            .links
            .iter()
            .find(|l| l.name.as_str() == "pelvis")
            .unwrap();
        assert!(
            (pelvis.mass.mass.value() - 3.814).abs() < 1e-3,
            "{:?}",
            pelvis.mass
        );
        let hip = profile
            .links
            .iter()
            .find(|l| l.name.as_str() == "left_hip_pitch_link")
            .unwrap();
        assert!((hip.mass.mass.value() - 1.35).abs() < 1e-6);
    }

    #[test]
    fn joint_limits_and_effort_agree_with_the_urdf_of_the_same_revision() {
        let usd = fixtures::unitree_g1_29dof_usd().unwrap();
        let urdf = fixtures::unitree_g1_29dof().unwrap();
        for joint in usd.joints.iter().filter(|j| j.kind.is_actuated()) {
            let reference = urdf
                .joint_by_name(joint.name.as_str())
                .unwrap_or_else(|| panic!("URDF has no {}", joint.name));
            let (a, b) = (
                joint.limits.position.unwrap(),
                reference.limits.position.unwrap(),
            );
            assert!(
                (a.lo.value() - b.lo.value()).abs() < 1e-4,
                "{} lo",
                joint.name
            );
            assert!(
                (a.hi.value() - b.hi.value()).abs() < 1e-4,
                "{} hi",
                joint.name
            );
            assert!(
                (joint.limits.effort.value() - reference.limits.effort.value()).abs() < 1e-9,
                "{} effort",
                joint.name
            );
            assert!(
                (joint.limits.velocity.value() - reference.limits.velocity.value()).abs() < 1e-3,
                "{} velocity",
                joint.name
            );
        }
    }

    #[test]
    fn the_drive_gains_come_from_the_usd_drives() {
        let profile = fixtures::unitree_g1_29dof_usd().unwrap();
        let hip = profile
            .actuators
            .iter()
            .find(|a| a.name.as_str() == "left_hip_pitch_joint")
            .unwrap();
        let gains = hip.gains.expect("USD drives state gains");
        assert!((gains.kp - 24.844_93).abs() < 1e-4, "{gains:?}");
        assert!((gains.kd - 0.009_938).abs() < 1e-6, "{gains:?}");
    }

    #[test]
    fn the_collision_shapes_are_captured_with_their_primitive_types() {
        let profile = fixtures::unitree_g1_29dof_usd().unwrap();
        let ankle_spheres: usize = profile
            .links
            .iter()
            .filter(|l| l.name.as_str().ends_with("ankle_roll_link"))
            .map(|l| {
                l.collision
                    .iter()
                    .filter(|c| matches!(c.geometry, CollisionGeometry::Sphere { .. }))
                    .count()
            })
            .sum();
        assert_eq!(ankle_spheres, 10, "five spheres per ankle in the USD");
        let boxes = profile
            .links
            .iter()
            .flat_map(|l| l.collision.iter())
            .filter(|c| matches!(c.geometry, CollisionGeometry::Box { .. }))
            .count();
        assert_eq!(boxes, 2, "one foot box per ankle");
    }

    #[test]
    fn the_floating_variant_validates_and_still_refuses_activation() {
        let profile = fixtures::unitree_g1_29dof_usd_floating_base().unwrap();
        assert_eq!(profile.dof(), 35);
        assert_eq!(profile.joints[0].kind, JointKind::Free);
        // No extensions are declared, so the e-stop gate is the refusal.
        let err = profile.activate(&[]).unwrap_err();
        assert!(matches!(err, ProfileError::NoIndependentEstop), "{err:?}");
    }

    #[test]
    fn zero_pose_link_positions_agree_with_the_urdf_relative_to_the_pelvis() {
        let usd = fixtures::unitree_g1_29dof_usd().unwrap();
        let urdf = fixtures::unitree_g1_29dof().unwrap();
        let positions = |p: &RobotProfile| -> std::collections::BTreeMap<String, [f64; 3]> {
            let q = vec![0.0; p.actuated_joints().count()];
            let poses = crate::kinematics::forward_kinematics(p, &q).unwrap();
            let base = poses.as_slice()[p
                .links
                .iter()
                .position(|l| l.name.as_str() == "pelvis")
                .unwrap()]
            .translation;
            p.links
                .iter()
                .enumerate()
                .map(|(i, l)| {
                    let t = poses.as_slice()[i].translation;
                    (
                        l.name.to_string(),
                        [t.x - base.x, t.y - base.y, t.z - base.z],
                    )
                })
                .collect()
        };
        let (a, b) = (positions(&usd), positions(&urdf));
        let mut worst: (f64, String) = (0.0, String::new());
        for (name, expected) in &b {
            if let Some(got) = a.get(name) {
                for axis in 0..3 {
                    let d = (got[axis] - expected[axis]).abs();
                    if d > worst.0 {
                        worst = (d, name.clone());
                    }
                }
            }
        }
        println!("WORST_FK {:?}", worst);
        assert!(worst.0 < 1e-4, "largest zero-pose difference {worst:?}");
    }

    #[test]
    fn the_omniverse_only_shaders_are_warnings_and_do_not_stop_the_import() {
        let (_profile, _report, warnings) = import_fixed_base_with_warnings(layer()).unwrap();
        // Four MDL material shaders, all sourced from OmniPBR.mdl (measured
        // 2026-10-30). Each is a warning, not an error.
        assert_eq!(warnings.len(), 4, "{warnings:?}");
        assert!(warnings.iter().all(|w| w.message.contains("OmniPBR.mdl")));
        assert!(warnings.iter().all(|w| w
            .message
            .contains("Geometry, joints and physics are unaffected")));
    }

    #[test]
    fn the_scene_ground_is_read_from_its_environment_collider() {
        let scene = Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/scenes/g1_ground.usda"
        ));
        let env = import_environment(scene).unwrap();
        assert_eq!(env.planes.len(), 1);
        let ground = env.planes[0];
        assert_eq!(ground.height, 0.0);
        assert_eq!(ground.stiffness, 20_000.0);
        assert_eq!(ground.normal_damping, 200.0);
        assert!((ground.friction - 0.8).abs() < 1e-6);
        assert!(env.unsupported.is_empty());
    }

    #[test]
    fn the_robot_layer_alone_has_no_environment() {
        let env = import_environment(layer()).unwrap();
        assert!(env.planes.is_empty());
        assert!(env.unsupported.is_empty());
    }

    #[test]
    fn the_tags_select_colliders_in_any_scene_and_ignore_the_rest() {
        let scene = Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/scenes/examples/tagging_example.usda"
        ));
        let env = import_environment(scene).unwrap();
        // One tagged plane is simulated, at its own height.
        assert_eq!(env.planes.len(), 1);
        assert!((env.planes[0].height - 0.25).abs() < 1e-12);
        // A tagged box is a collider this crate cannot simulate yet: named, not dropped.
        assert_eq!(env.unsupported, vec!["/scene/tagged_box".to_owned()]);
        // A prim tagged environment only is visual.
        assert_eq!(env.visual_only, 1);
    }

    #[test]
    fn a_missing_layer_is_an_error_not_an_empty_robot() {
        let err = import_fixed_base(Path::new("/nonexistent/g1.usd")).unwrap_err();
        assert!(matches!(err, UsdImportError::Usd(_)));
    }
}

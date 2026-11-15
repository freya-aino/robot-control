//! The data half of the offline replay viewer (ADR 0019): everything the
//! renderer needs, prepared without any rendering dependency.
//!
//! This crate is the isolation boundary the Bevy decision prescribes:
//! `robot-viewer` stays the std-only inspection data layer, the Bevy
//! application lives behind this crate's off-by-default `bevy` feature
//! (`src/main.rs`), and this library — loading a recorded run, resolving
//! the robot, posing every link per frame, describing each collision shape
//! as plain data — compiles and tests in the default workspace build with
//! no foreign dependency at all. SPECS.md §2.5: a viewer is a development
//! service, never a robot-host dependency.

use robot_model::hull::hull_faces;
use robot_model::kinematics::forward_kinematics_floating;
use robot_model::link::CollisionGeometry;
use robot_model::pose::RigidPose;
use robot_model::profile::RobotProfile;
use robot_types::frames::{Quat, Vec3};
use robot_viewer::pose::PoseLog;
use std::path::Path;

/// A collision shape prepared for rendering: plain data, no toolkit types.
#[derive(Clone, Debug)]
pub enum ShapeKind {
    /// A sphere of this radius, metres.
    Sphere {
        /// Radius.
        radius: f64,
    },
    /// A box with these half extents, metres.
    Box {
        /// Half extents.
        half: [f64; 3],
    },
    /// A cylinder along local Z.
    Cylinder {
        /// Radius.
        radius: f64,
        /// Half height.
        half_height: f64,
    },
    /// A convex hull as triangles (vertex positions per corner, outward
    /// wound), ready to become a mesh.
    Hull {
        /// Triangle corners, three consecutive entries per triangle.
        triangles: Vec<[f64; 3]>,
    },
}

/// One collision shape attached to a link.
#[derive(Clone, Debug)]
pub struct ShapeInstance {
    /// Index into the profile's links.
    pub link: usize,
    /// Shape pose in the link frame.
    pub local: RigidPose,
    /// The geometry.
    pub kind: ShapeKind,
}

/// Why a replay could not be loaded.
#[derive(Debug)]
pub enum ReplayError {
    /// The robot name is not a vendored fixture this viewer knows.
    UnknownRobot(String),
    /// The fixture failed to load.
    Fixture(String),
    /// The run could not be read.
    Run(String),
    /// A frame's joints do not match the profile.
    JointMismatch(String),
}

impl core::fmt::Display for ReplayError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnknownRobot(name) => {
                write!(f, "unknown robot {name:?}; expected g1-usd or g1")
            }
            Self::Fixture(e) => write!(f, "fixture: {e}"),
            Self::Run(e) => write!(f, "run: {e}"),
            Self::JointMismatch(e) => write!(f, "joints: {e}"),
        }
    }
}

impl core::error::Error for ReplayError {}

/// A loaded replay: the fixed-base profile, the recorded frames, and the
/// mapping from the profile's actuated joints into each frame's angles.
pub struct ReplayData {
    /// The fixed-base profile the frames pose.
    pub profile: RobotProfile,
    /// The recorded run.
    pub log: PoseLog,
    /// For each actuated joint (profile order), its index into a frame's
    /// `joint_angles`, by the dump's joint names.
    joint_slots: Vec<usize>,
    /// Shapes prepared once.
    shapes: Vec<ShapeInstance>,
}

impl ReplayData {
    /// Loads a committed frame-dump directory for a named vendored robot
    /// (`g1-usd` or `g1`, matching `robot replay --robot`).
    ///
    /// # Errors
    ///
    /// [`ReplayError`] naming the failing stage.
    pub fn load_dir(dir: &Path, robot: &str) -> Result<Self, ReplayError> {
        let log = robot_viewer::pose::load_committed(dir)
            .map_err(|e| ReplayError::Run(format!("{e:?}")))?;
        Self::from_log(log, robot)
    }

    /// Loads a `viewer-frames/1` run from a blob store by its manifest id.
    ///
    /// # Errors
    ///
    /// [`ReplayError`] naming the failing stage.
    pub fn load_store(store_dir: &Path, id_hex: &str, robot: &str) -> Result<Self, ReplayError> {
        let store = robot_artifacts::store::BlobStore::open(
            store_dir,
            robot_artifacts::store::StoreQuota::default(),
        )
        .map_err(|e| ReplayError::Run(format!("{e}")))?;
        let id = robot_artifacts::ContentId::parse(id_hex)
            .map_err(|e| ReplayError::Run(format!("{e:?}")))?;
        let log = robot_viewer::artifact::load_from_store(&store, &id)
            .map_err(|e| ReplayError::Run(format!("{e}")))?;
        Self::from_log(log, robot)
    }

    /// Builds the replay from an already-loaded log.
    ///
    /// # Errors
    ///
    /// [`ReplayError`] when the robot is unknown or the dump's joint names
    /// do not cover the profile's actuated joints.
    pub fn from_log(log: PoseLog, robot: &str) -> Result<Self, ReplayError> {
        let profile = match robot {
            "g1-usd" => robot_model::fixtures::unitree_g1_29dof_usd(),
            "g1" => robot_model::fixtures::unitree_g1_29dof(),
            other => return Err(ReplayError::UnknownRobot(other.to_owned())),
        }
        .map_err(|e| ReplayError::Fixture(format!("{e:?}")))?;
        let mut joint_slots = Vec::with_capacity(profile.dof());
        for joint in profile.actuated_joints() {
            let name = joint.name.as_str();
            let slot = log.joints.iter().position(|j| j == name).ok_or_else(|| {
                ReplayError::JointMismatch(format!("the dump has no joint {name:?}"))
            })?;
            joint_slots.push(slot);
        }
        let shapes = prepared_shapes(&profile);
        Ok(Self {
            profile,
            log,
            joint_slots,
            shapes,
        })
    }

    /// Number of frames.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.log.frames.len()
    }

    /// The prepared collision shapes, one list for the whole run.
    #[must_use]
    pub fn shapes(&self) -> &[ShapeInstance] {
        &self.shapes
    }

    /// World pose of every link at a frame: the recorded pelvis pose
    /// carrying the fixed-base tree, through
    /// [`forward_kinematics_floating`].
    ///
    /// # Errors
    ///
    /// [`ReplayError`] for an out-of-range frame or a kinematics failure.
    pub fn link_poses(&self, frame: usize) -> Result<Vec<RigidPose>, ReplayError> {
        let f = self
            .log
            .frames
            .get(frame)
            .ok_or_else(|| ReplayError::Run(format!("frame {frame} out of range")))?;
        let base = RigidPose {
            translation: Vec3::new(
                f.pelvis_position[0],
                f.pelvis_position[1],
                f.pelvis_position[2],
            ),
            rotation: Quat {
                w: f.pelvis_quaternion[0],
                x: f.pelvis_quaternion[1],
                y: f.pelvis_quaternion[2],
                z: f.pelvis_quaternion[3],
            }
            .normalized(),
        };
        let q: Vec<f64> = self
            .joint_slots
            .iter()
            .map(|&slot| f.joint_angles.get(slot).copied().unwrap_or(0.0))
            .collect();
        let poses = forward_kinematics_floating(&self.profile, &base, &q)
            .map_err(|e| ReplayError::JointMismatch(format!("{e:?}")))?;
        Ok(poses.as_slice().to_vec())
    }

    /// Simulated time of a frame, seconds, or zero out of range.
    #[must_use]
    pub fn time_of(&self, frame: usize) -> f64 {
        self.log
            .frames
            .get(frame)
            .map(|f| f.timestamp.nanos() as f64 * 1e-9)
            .unwrap_or(0.0)
    }
}

/// Flattens the profile's collision geometry into render-ready shapes.
/// A raw mesh reference carries no vertices here and is skipped — the
/// simulator does not contact it either (specs/07), so the picture agrees
/// with the physics.
fn prepared_shapes(profile: &RobotProfile) -> Vec<ShapeInstance> {
    let mut out = Vec::new();
    for (index, link) in profile.links.iter().enumerate() {
        for shape in &link.collision {
            let kind = match &shape.geometry {
                CollisionGeometry::Sphere { radius } => ShapeKind::Sphere { radius: *radius },
                CollisionGeometry::Box { half_extents } => ShapeKind::Box {
                    half: [half_extents.x, half_extents.y, half_extents.z],
                },
                CollisionGeometry::Cylinder { radius, length } => ShapeKind::Cylinder {
                    radius: *radius,
                    half_height: 0.5 * length,
                },
                CollisionGeometry::ConvexHull { points, .. } => {
                    let cloud: Vec<[f64; 3]> = points.iter().map(|p| [p.x, p.y, p.z]).collect();
                    let Some(faces) = hull_faces(&cloud) else {
                        continue;
                    };
                    let mut triangles = Vec::with_capacity(faces.len() * 3);
                    for f in faces {
                        for v in f {
                            triangles.push(cloud[v]);
                        }
                    }
                    ShapeKind::Hull { triangles }
                }
                CollisionGeometry::Mesh { .. } => continue,
            };
            let local = RigidPose {
                translation: shape.pose.translation,
                rotation: shape.pose.rotation,
            };
            out.push(ShapeInstance {
                link: index,
                local,
                kind,
            });
        }
    }
    out
}

/// The bone segments of a frame: one world-space pair per joint, parent
/// link origin to child link origin, for a skeleton overlay.
#[must_use]
pub fn bones(profile: &RobotProfile, poses: &[RigidPose]) -> Vec<([f64; 3], [f64; 3])> {
    let index_of = |id| profile.links.iter().position(|l| l.id == id);
    profile
        .joints
        .iter()
        .filter_map(|j| {
            let p = poses.get(index_of(j.parent)?)?.translation;
            let c = poses.get(index_of(j.child)?)?.translation;
            Some(([p.x, p.y, p.z], [c.x, c.y, c.z]))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_types::time::{ClockId, Timestamp};
    use robot_viewer::pose::PoseFrame;

    /// A two-frame log over the USD G1's own joint names, pelvis rising.
    fn tiny_log() -> PoseLog {
        let profile = robot_model::fixtures::unitree_g1_29dof_usd().unwrap();
        let joints: Vec<String> = profile
            .actuated_joints()
            .map(|j| j.name.as_str().to_owned())
            .collect();
        let width = joints.len();
        let frame = |step: u64, z: f64| PoseFrame {
            step,
            timestamp: Timestamp::sampling(step as i64 * 1_000_000, ClockId::Simulation),
            pelvis_position: [0.0, 0.0, z],
            pelvis_quaternion: [1.0, 0.0, 0.0, 0.0],
            joint_angles: vec![0.0; width],
        };
        PoseLog {
            frames: vec![frame(1, 0.79), frame(2, 0.80)],
            joints,
        }
    }

    #[test]
    fn a_replay_poses_every_link_and_prepares_every_contactable_shape() {
        let replay = ReplayData::from_log(tiny_log(), "g1-usd").unwrap();
        assert_eq!(replay.frame_count(), 2);
        // All 33 declared colliders are renderable: 18 primitives and 15
        // hulls, matching what the simulator contacts (specs/07).
        assert_eq!(replay.shapes().len(), 33);
        assert!(replay
            .shapes()
            .iter()
            .any(|s| matches!(s.kind, ShapeKind::Hull { .. })));
        let poses = replay.link_poses(0).unwrap();
        assert_eq!(poses.len(), replay.profile.links.len());
        // The pelvis sits where the frame says.
        let pelvis = replay
            .profile
            .links
            .iter()
            .position(|l| l.name.as_str() == "pelvis")
            .unwrap();
        assert!((poses[pelvis].translation.z - 0.79).abs() < 1e-12);
        let bones = bones(&replay.profile, &poses);
        assert_eq!(bones.len(), replay.profile.joints.len());
        assert!((replay.time_of(1) - 0.002).abs() < 1e-9);
    }

    #[test]
    fn refusals_name_their_stage() {
        assert!(matches!(
            ReplayData::from_log(tiny_log(), "h1"),
            Err(ReplayError::UnknownRobot(_))
        ));
        let mut log = tiny_log();
        log.joints[0] = "not_a_joint".to_owned();
        assert!(matches!(
            ReplayData::from_log(log, "g1-usd"),
            Err(ReplayError::JointMismatch(_))
        ));
    }
}

//! Forward kinematics over the profile's joint tree.
//!
//! `robot-model` owns the kinematic tree itself (SPECS.md Section 6); the
//! constrained inverse kinematics that uses it lives in `motion-retarget`.
//! What belongs here is the one unambiguous computation: given a joint value
//! for every actuated joint, where is every link, expressed as a
//! [`RigidPose`] relative to the profile's root (which stands in for world,
//! since no project-wide world frame is fixed until C-04 resolves).

use crate::joint::JointKind;
use crate::pose::RigidPose;
use crate::profile::RobotProfile;
use core::fmt;
use robot_types::ids::{JointId, LinkId};

/// Why forward kinematics could not be evaluated.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum KinematicsError {
    /// The joint-value vector did not have one entry per actuated joint.
    WidthMismatch {
        /// Actuated joints in the profile.
        expected: usize,
        /// Values supplied.
        found: usize,
    },
    /// A supplied joint value was not finite.
    NotFinite {
        /// The joint.
        joint: JointId,
    },
}

impl fmt::Display for KinematicsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WidthMismatch { expected, found } => {
                write!(f, "expected {expected} joint values, found {found}")
            }
            Self::NotFinite { joint } => write!(f, "joint {joint} value is not finite"),
        }
    }
}

impl core::error::Error for KinematicsError {}

/// The pose of every link, computed from a set of joint values.
///
/// Poses are relative to the profile's root link, which is treated as the
/// reference origin for this computation — a choice local to this type, not
/// an assertion about a project-wide world frame.
#[derive(Clone, PartialEq, Debug)]
pub struct LinkPoses {
    /// Pose of each link, indexed the same way as `profile.links`.
    poses: Vec<RigidPose>,
}

impl LinkPoses {
    /// Builds a set of link poses directly from already-known poses (for
    /// example, ones read back from a recorded artifact), without
    /// recomputing them from a profile and joint values.
    #[must_use]
    pub const fn from_poses(poses: Vec<RigidPose>) -> Self {
        Self { poses }
    }

    /// The pose of a link, by its position in the owning profile's `links`
    /// vector (not its [`LinkId`], which may not be contiguous).
    #[must_use]
    pub fn by_index(&self, index: usize) -> Option<RigidPose> {
        self.poses.get(index).copied()
    }

    /// All poses, in profile link order.
    #[must_use]
    pub fn as_slice(&self) -> &[RigidPose] {
        &self.poses
    }
}

/// Computes the root-relative pose of every link given a value for every
/// actuated joint, supplied in [`RobotProfile::actuated_joints`] order.
///
/// A fixed joint takes no entry in `joint_values`; its contribution is its
/// own fixed pose. Assumes the profile has already passed
/// [`RobotProfile::validate`]; behavior on an invalid profile is unspecified
/// rather than checked again here, since every call site validates once at
/// load.
pub fn forward_kinematics(
    profile: &RobotProfile,
    joint_values: &[f64],
) -> Result<LinkPoses, KinematicsError> {
    let actuated: Vec<JointId> = profile.actuated_joints().map(|j| j.id).collect();
    if joint_values.len() != actuated.len() {
        return Err(KinematicsError::WidthMismatch {
            expected: actuated.len(),
            found: joint_values.len(),
        });
    }
    if let Some((joint, _)) = actuated
        .iter()
        .zip(joint_values)
        .find(|(_, v)| !v.is_finite())
    {
        return Err(KinematicsError::NotFinite { joint: *joint });
    }

    let mut poses = vec![RigidPose::IDENTITY; profile.links.len()];
    let index_of = |id: LinkId| profile.links.iter().position(|l| l.id == id);
    let root_index = index_of(profile.root_link).expect("validated profile has the root link");
    poses[root_index] = RigidPose::IDENTITY;

    let mut stack = vec![profile.root_link];
    while let Some(current) = stack.pop() {
        let current_index = index_of(current).expect("validated profile");
        let root_from_current = poses[current_index];
        for joint in profile.child_joints_of(current) {
            let value = match joint.kind {
                JointKind::Fixed => 0.0,
                _ => {
                    let pos = actuated
                        .iter()
                        .position(|id| *id == joint.id)
                        .expect("every actuated joint appears in the actuated list");
                    joint_values[pos]
                }
            };
            let current_from_child = joint.transform_at(value);
            let root_from_child = root_from_current.compose(&current_from_child);
            let child_index = index_of(joint.child).expect("validated profile");
            poses[child_index] = root_from_child;
            stack.push(joint.child);
        }
    }

    Ok(LinkPoses { poses })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actuator::{
        ActuatorSpec, PdGains, ThermalLimits, TorqueSpeedEnvelope, Transmission,
    };
    use crate::joint::{JointKind, JointLimits, JointSpec};
    use crate::link::{LinkSpec, MassProperties};
    use crate::profile::{FirmwareContract, RobotProfile, PROFILE_SCHEMA_VERSION};
    use robot_types::frames::{FrameId, GravityConvention, Vec3};
    use robot_types::ids::{ActuatorId, JointId, LinkId, Name};
    use robot_types::schema::ExtensionTable;
    use robot_types::time::Duration;
    use robot_types::units::{
        Amperes, Celsius, Kilograms, NewtonMetres, Radians, RadiansPerSecond, Range,
    };

    fn link(id: u32, name: &str) -> LinkSpec {
        let mut mp = MassProperties::MASSLESS;
        mp.mass = Kilograms(1.0);
        LinkSpec {
            id: LinkId::new(id),
            name: Name::new(name).unwrap(),
            mass: mp,
            collision: Vec::new(),
        }
    }

    fn revolute_joint(id: u32, name: &str, parent: u32, child: u32, offset_x: f64) -> JointSpec {
        JointSpec {
            id: JointId::new(id),
            name: Name::new(name).unwrap(),
            kind: JointKind::Revolute,
            parent: LinkId::new(parent),
            child: LinkId::new(child),
            origin: RigidPose::from_translation(Vec3::new(offset_x, 0.0, 0.0)),
            axis: Vec3::unit_z(),
            limits: JointLimits::bounded(
                Range::<Radians>::new(Radians(-4.0), Radians(4.0)).unwrap(),
                RadiansPerSecond(10.0),
                NewtonMetres(10.0),
            ),
            sign: 1.0,
            zero_offset: 0.0,
        }
    }

    fn two_link_arm() -> RobotProfile {
        let actuator = |id: u32, name: &str, joint: u32| ActuatorSpec {
            id: ActuatorId::new(id),
            name: Name::new(name).unwrap(),
            joint: JointId::new(joint),
            transmission: Transmission::DIRECT,
            envelope: TorqueSpeedEnvelope {
                stall_torque: NewtonMetres(1.0),
                corner_speed: RadiansPerSecond(1.0),
                no_load_speed: RadiansPerSecond(2.0),
            },
            gains: Some(PdGains { kp: 1.0, kd: 0.1 }),
            command_latency: Duration::from_millis(1),
            thermal: ThermalLimits {
                limit: Celsius(80.0),
                ambient: Celsius(25.0),
                time_constant: Duration::from_secs(10),
                current_at_limit: Amperes(5.0),
            },
            current_uncertainty: Amperes(0.01),
        };
        RobotProfile {
            schema_version: PROFILE_SCHEMA_VERSION,
            name: Name::new("two-link-arm").unwrap(),
            profile_version: "1.0".to_owned(),
            joints: vec![
                revolute_joint(0, "shoulder", 0, 1, 0.0),
                revolute_joint(1, "elbow", 1, 2, 1.0),
            ],
            links: vec![link(0, "base"), link(1, "upper"), link(2, "lower")],
            actuators: vec![
                actuator(0, "shoulder-motor", 0),
                actuator(1, "elbow-motor", 1),
            ],
            sensors: Vec::new(),
            root_link: LinkId::new(0),
            gravity: GravityConvention::minus_z_standard(FrameId::WORLD),
            firmware: FirmwareContract {
                version: "1.0".to_owned(),
                command_expiration: Duration::from_millis(100),
                watchdog_timeout: Duration::from_millis(200),
                has_independent_estop: true,
                allowed_modes: Vec::new(),
            },
            extensions: ExtensionTable::new(),
        }
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn zero_pose_places_links_at_their_chained_offsets() {
        let profile = two_link_arm();
        profile.validate().unwrap();
        let poses = forward_kinematics(&profile, &[0.0, 0.0]).unwrap();
        let base = poses.by_index(0).unwrap();
        let upper = poses.by_index(1).unwrap();
        let lower = poses.by_index(2).unwrap();
        assert!(close(base.translation.x, 0.0));
        assert!(close(upper.translation.x, 0.0));
        assert!(close(lower.translation.x, 1.0));
    }

    #[test]
    fn a_right_angle_at_the_shoulder_swings_the_elbow_offset_into_y() {
        let profile = two_link_arm();
        let poses = forward_kinematics(&profile, &[core::f64::consts::FRAC_PI_2, 0.0]).unwrap();
        let lower = poses.by_index(2).unwrap();
        // The elbow's +1 m offset along local X, after a 90 degree rotation
        // about Z at the shoulder, points along world Y.
        assert!(close(lower.translation.x, 0.0), "{:?}", lower.translation);
        assert!(close(lower.translation.y, 1.0), "{:?}", lower.translation);
    }

    #[test]
    fn elbow_rotation_composes_with_shoulder_rotation_in_orientation() {
        let profile = two_link_arm();
        // The lower link's origin sits exactly at the elbow joint, so its
        // *position* depends only on the shoulder angle (the elbow joint
        // rotates what is attached beyond it, not its own origin). Its
        // *orientation* is the composition of both joint rotations.
        let poses = forward_kinematics(
            &profile,
            &[core::f64::consts::FRAC_PI_2, core::f64::consts::FRAC_PI_2],
        )
        .unwrap();
        let lower = poses.by_index(2).unwrap();
        assert!(close(lower.translation.x, 0.0), "{:?}", lower.translation);
        assert!(close(lower.translation.y, 1.0), "{:?}", lower.translation);
        // Cumulative rotation of 180 degrees about Z maps +X to -X.
        let v = lower.apply_direction(Vec3::unit_x());
        assert!(close(v.x, -1.0) && close(v.y, 0.0), "{v:?}");
    }

    #[test]
    fn a_wrong_sized_value_vector_is_rejected() {
        let profile = two_link_arm();
        assert_eq!(
            forward_kinematics(&profile, &[0.0]),
            Err(KinematicsError::WidthMismatch {
                expected: 2,
                found: 1
            })
        );
    }

    #[test]
    fn a_nonfinite_joint_value_is_rejected() {
        let profile = two_link_arm();
        assert_eq!(
            forward_kinematics(&profile, &[f64::NAN, 0.0]),
            Err(KinematicsError::NotFinite {
                joint: JointId::new(0)
            })
        );
    }

    #[test]
    fn link_poses_as_slice_matches_profile_link_order() {
        let profile = two_link_arm();
        let poses = forward_kinematics(&profile, &[0.0, 0.0]).unwrap();
        assert_eq!(poses.as_slice().len(), profile.links.len());
        assert!(poses.by_index(99).is_none());
    }
}

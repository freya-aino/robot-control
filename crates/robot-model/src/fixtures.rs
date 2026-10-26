//! Vendored real-robot fixtures: known, cited robot descriptions usable for
//! kinematics, retargeting and inspection work without authoring a
//! `RobotProfile` by hand.
//!
//! SPECS.md Section 3's evidence discipline applies to a vendored
//! third-party description exactly as it does to anything else: a URDF
//! carries kinematics and mass properties, not the actuator, firmware or
//! safety facts a profile needs to be trusted on a control path
//! (`urdf::placeholder_actuator`'s own doc comment says this already).
//! These fixtures are for kinematics-only use — forward kinematics, IK
//! retargeting, offline inspection — and are deliberately built so
//! [`profile::RobotProfile::activate`] refuses them, rather than asserting
//! a hardware safety property this crate has no cited source for.

use crate::joint::{JointKind, JointLimits, JointSpec};
use crate::link::{LinkSpec, MassProperties};
use crate::pose::RigidPose;
use crate::profile::{FirmwareContract, ProfileError, RobotProfile, PROFILE_SCHEMA_VERSION};
use crate::urdf::{import_from_str, placeholder_actuator, UrdfError};
use core::fmt;
use robot_types::frames::{FrameId, GravityConvention, Vec3};
use robot_types::ids::{JointId, LinkId, Name};
use robot_types::schema::ExtensionTable;
use robot_types::time::Duration;
use robot_types::units::{NewtonMetres, RadiansPerSecond};

/// Unitree G1 humanoid, 29-DOF, revision 1.0, vendored from
/// `unitreerobotics/unitree_rl_gym`. See
/// `assets/robots/unitree_g1/PROVENANCE.md` for the pinned commit,
/// license and exactly what was and wasn't checked before vendoring —
/// including why this is the same variant
/// [RoboJuDo](https://github.com/HansZ8/RoboJuDo) uses as its default G1
/// asset, even though RoboJuDo itself vendors the MJCF sibling of this
/// file rather than a URDF.
const UNITREE_G1_29DOF_URDF: &str =
    include_str!("../../../assets/robots/unitree_g1/g1_29dof_rev_1_0.urdf");

/// Why a vendored fixture could not be built into a usable profile.
#[derive(Debug)]
pub enum FixtureError {
    /// The vendored URDF failed to import.
    Urdf(UrdfError),
    /// The assembled profile failed its own validation.
    Profile(ProfileError),
}

impl fmt::Display for FixtureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Urdf(e) => write!(f, "{e}"),
            Self::Profile(e) => write!(f, "{e}"),
        }
    }
}

impl core::error::Error for FixtureError {}

impl From<UrdfError> for FixtureError {
    fn from(e: UrdfError) -> Self {
        Self::Urdf(e)
    }
}
impl From<ProfileError> for FixtureError {
    fn from(e: ProfileError) -> Self {
        Self::Profile(e)
    }
}

/// Builds a [`RobotProfile`] for Unitree's G1 humanoid, 29-DOF revision
/// 1.0, from the vendored URDF, validated (not activated) before return.
///
/// Every actuator comes from [`crate::urdf::placeholder_actuator`] — the
/// joint's own declared velocity/effort limit standing in for a real
/// actuator envelope, since URDF carries no transmission ratio, gain,
/// thermal or command-latency data. No sensors are populated. The gravity
/// convention is [`GravityConvention::minus_z_standard`], documented on
/// that constructor as "a configuration default for tests and fixtures,
/// not a project convention" — C-04 remains open.
///
/// The firmware contract declares **no** independent emergency stop. This
/// is not a claim that the real G1 lacks one; it is the absence of a
/// claim this crate has no cited, verified source for. SPECS.md Section
/// 12.1 requires hardware e-stop independence before any profile may be
/// [`RobotProfile::activate`]d on a hardware-facing mode, and
/// `ProfileError::NoIndependentEstop` is exactly the gate that refuses
/// this fixture for that until a real, cited `FirmwareContract` replaces
/// this one. Use this profile for kinematics (forward kinematics, IK
/// retargeting, inspection) only.
pub fn unitree_g1_29dof() -> Result<RobotProfile, FixtureError> {
    let imported = import_from_str(UNITREE_G1_29DOF_URDF)?;

    let actuators = imported
        .joints
        .iter()
        .filter(|j| j.kind.is_actuated())
        .enumerate()
        .map(|(index, joint)| placeholder_actuator(index as u32, joint))
        .collect();

    let profile = RobotProfile {
        schema_version: PROFILE_SCHEMA_VERSION,
        name: Name::new("unitree-g1-29dof-rev-1-0").expect("valid name literal"),
        profile_version: "unitree_rl_gym@276801e46c5d433564f24658bac64f254b7d2d4b".to_owned(),
        joints: imported.joints,
        links: imported.links,
        actuators,
        sensors: Vec::new(),
        root_link: imported.root_link,
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
    Ok(profile)
}

/// The vendored G1 of [`unitree_g1_29dof`], with its former root
/// (`pelvis`) re-attached to a new, explicit, massless `world` link via
/// a [`JointKind::Free`] joint, instead of being the profile's root
/// (implicitly fixed to whatever frame holds it) directly.
///
/// `crates/robot-sim/SIMULATION_SPECS.md` Section 3 is the design record
/// for why a floating base is built this way — an explicit `world` link
/// plus a `Free` joint, assigned by the simulation layer, never by URDF
/// import — rather than as a relaxation of the URDF importer's own
/// rejection of a `floating` joint type. The new joint carries no
/// actuator (nothing drives a floating base directly; `robot_sim::
/// articulated::MultibodyState` applies whatever external wrench a
/// caller supplies through its six generalized-force slots) and, like
/// [`unitree_g1_29dof`], only `.validate()`s — the firmware contract
/// still declares no independent e-stop, so `.activate()` still
/// correctly refuses it.
///
/// The new joint is inserted *first* in `profile.joints`, ahead of the
/// G1's own 29 joints: `robot_sim::articulated::ArticulatedBody`
/// assigns every joint's `q`/`qd` slots in `RobotProfile::actuated_joints`
/// (storage) order, so this is what puts the floating joint's 7-wide
/// position and 6-wide velocity/force slots at index 0 rather than
/// scattered after the G1's own 29 — the conventional "free joint first"
/// layout, matching MuJoCo's own `qpos`/`qvel` convention.
pub fn unitree_g1_29dof_floating_base() -> Result<RobotProfile, FixtureError> {
    let mut profile = unitree_g1_29dof()?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::ProfileError;

    #[test]
    fn the_vendored_g1_profile_validates() {
        let profile = unitree_g1_29dof().expect("fixture builds and validates");
        assert_eq!(
            profile.dof(),
            29,
            "left+right legs (12) + waist (3) + left+right arms (14)"
        );
        assert_eq!(profile.links.len(), 39);
        assert_eq!(profile.joints.len(), 38);
        let pelvis = profile
            .links
            .iter()
            .find(|l| l.name.as_str() == "pelvis")
            .unwrap();
        assert_eq!(profile.root_link, pelvis.id);
    }

    #[test]
    fn it_cannot_be_activated_without_a_real_firmware_contract() {
        let profile = unitree_g1_29dof().unwrap();
        let err = profile.activate(&[]).unwrap_err();
        assert!(matches!(err, ProfileError::NoIndependentEstop));
    }

    #[test]
    fn every_expected_joint_group_is_present() {
        let profile = unitree_g1_29dof().unwrap();
        for name in [
            "left_hip_pitch_joint",
            "right_hip_pitch_joint",
            "waist_yaw_joint",
            "left_shoulder_pitch_joint",
            "right_wrist_yaw_joint",
        ] {
            assert!(
                profile.joint_by_name(name).is_some(),
                "missing joint {name:?}"
            );
        }
    }

    #[test]
    fn forward_kinematics_runs_at_the_zero_posture() {
        let profile = unitree_g1_29dof().unwrap();
        let zeros = vec![0.0; profile.dof()];
        let poses = crate::kinematics::forward_kinematics(&profile, &zeros).unwrap();
        assert_eq!(poses.as_slice().len(), profile.links.len());
    }

    #[test]
    fn the_floating_base_variant_adds_a_world_link_and_a_free_joint() {
        let fixed = unitree_g1_29dof().unwrap();
        let floating = unitree_g1_29dof_floating_base().unwrap();

        assert_eq!(floating.dof(), fixed.dof() + 6);
        assert_eq!(floating.links.len(), fixed.links.len() + 1);
        assert_eq!(floating.joints.len(), fixed.joints.len() + 1);

        let world = floating
            .links
            .iter()
            .find(|l| l.name.as_str() == "world")
            .expect("world link exists");
        assert_eq!(floating.root_link, world.id);

        let floating_joint = floating
            .joints
            .iter()
            .find(|j| j.name.as_str() == "floating_base")
            .expect("floating_base joint exists");
        assert_eq!(floating_joint.kind, JointKind::Free);
        assert_eq!(floating_joint.parent, world.id);
        assert_eq!(floating_joint.child, fixed.root_link);
    }

    #[test]
    fn the_floating_base_variant_still_refuses_activation() {
        let profile = unitree_g1_29dof_floating_base().unwrap();
        let err = profile.activate(&[]).unwrap_err();
        assert!(matches!(err, ProfileError::NoIndependentEstop));
    }
}

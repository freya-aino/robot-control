//! Robot profiles shared by this crate's unit tests.

use robot_model::actuator::{
    ActuatorSpec, PdGains, ThermalLimits, TorqueSpeedEnvelope, Transmission,
};
use robot_model::joint::{JointKind, JointLimits, JointSpec};
use robot_model::link::{LinkSpec, MassProperties};
use robot_model::pose::RigidPose;
use robot_model::profile::RobotProfile;
use robot_model::profile::{FirmwareContract, PROFILE_SCHEMA_VERSION};
use robot_types::frames::{FrameId, GravityConvention, Vec3};
use robot_types::ids::{ActuatorId, JointId, LinkId, Name};
use robot_types::schema::ExtensionTable;
use robot_types::time::Duration;
use robot_types::units::{
    Amperes, Celsius, Kilograms, NewtonMetres, Radians, RadiansPerSecond, Range,
};

pub fn link(id: u32, name: &str) -> LinkSpec {
    let mut mp = MassProperties::MASSLESS;
    mp.mass = Kilograms(1.0);
    LinkSpec {
        id: LinkId::new(id),
        name: Name::new(name).unwrap(),
        mass: mp,
        collision: Vec::new(),
    }
}

pub fn joint(id: u32, parent: u32, child: u32, offset_x: f64) -> JointSpec {
    JointSpec {
        id: JointId::new(id),
        name: Name::new(&format!("j{id}")).unwrap(),
        kind: JointKind::Revolute,
        parent: LinkId::new(parent),
        child: LinkId::new(child),
        origin: RigidPose::from_translation(Vec3::new(offset_x, 0.0, 0.0)),
        axis: Vec3::unit_z(),
        limits: JointLimits::bounded(
            Range::<Radians>::new(Radians(-3.0), Radians(3.0)).unwrap(),
            RadiansPerSecond(10.0),
            NewtonMetres(50.0),
        ),
        sign: 1.0,
        zero_offset: 0.0,
    }
}

/// A genuine 2-DOF planar arm: shoulder rotates the whole arm, elbow
/// rotates the forearm about its own pivot, and a fixed joint carries
/// the hand a further unit length beyond the elbow — without that
/// third link, the "lower" link's own position would depend on the
/// shoulder angle alone (its origin *is* the elbow pivot), which was
/// the actual cause of this test's first failure: the originally
/// chosen target was not on the elbow's reachable circle at all.
pub fn two_link_arm() -> RobotProfile {
    let actuator = |id: u32, joint_id: u32| ActuatorSpec {
        id: ActuatorId::new(id),
        name: Name::new(&format!("a{id}")).unwrap(),
        joint: JointId::new(joint_id),
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
    let hand_joint = JointSpec {
        id: JointId::new(2),
        name: Name::new("hand-fixed").unwrap(),
        kind: JointKind::Fixed,
        parent: LinkId::new(2),
        child: LinkId::new(3),
        origin: RigidPose::from_translation(Vec3::new(1.0, 0.0, 0.0)),
        axis: Vec3::unit_z(),
        limits: JointLimits::unbounded(RadiansPerSecond(0.0), NewtonMetres(0.0)),
        sign: 1.0,
        zero_offset: 0.0,
    };
    RobotProfile {
        schema_version: PROFILE_SCHEMA_VERSION,
        name: Name::new("fixture-two-link-arm").unwrap(),
        profile_version: "1.0".to_owned(),
        joints: vec![joint(0, 0, 1, 0.0), joint(1, 1, 2, 1.0), hand_joint],
        links: vec![
            link(0, "base"),
            link(1, "upper"),
            link(2, "lower"),
            link(3, "hand"),
        ],
        actuators: vec![actuator(0, 0), actuator(1, 1)],
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

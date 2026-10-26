//! The robot profile: the complete versioned description of SPECS.md
//! Section 7.1, and the validation that keeps its graph and identities honest.
//!
//! "Every profile MUST carry a `schema_version` and typed, stable IDs for
//! joints, links, actuators and sensors. IDs MUST NOT be reused for a
//! different entity within a profile lineage." This module is where that rule
//! is actually checked — [`RobotProfile::validate`] is the single place every
//! consumer (simulator, runtime, viewer) calls before trusting a profile, and
//! [`RobotProfile::activate`] additionally applies the stricter
//! [`Strictness::Activating`] extension-table check Section 7.1 requires
//! "before a profile is activated on a control path".

use crate::actuator::ActuatorSpec;
use crate::joint::JointSpec;
use crate::link::LinkSpec;
use crate::sensor::SensorSpec;
use core::fmt;
use robot_types::frames::GravityConvention;
use robot_types::ids::{JointId, LinkId, Name};
use robot_types::schema::{ExtensionTable, SchemaVersion, Strictness};
use robot_types::time::Duration;

/// The schema version this crate writes for robot profiles.
pub const PROFILE_SCHEMA_VERSION: SchemaVersion = SchemaVersion::new(1, 0);

/// The firmware, watchdog and command-expiry facts of SPECS.md Section 7.1's
/// closing bullet: "Firmware/protocol version, watchdog semantics, command
/// expiration, emergency stop interface, and allowed operating modes."
#[derive(Clone, PartialEq, Debug)]
pub struct FirmwareContract {
    /// Firmware or protocol version string.
    pub version: String,
    /// Timeout after which the firmware itself discards a stale command,
    /// independent of the host watchdog.
    pub command_expiration: Duration,
    /// Timeout after which the host's independent watchdog considers the
    /// link lost.
    pub watchdog_timeout: Duration,
    /// Whether an emergency-stop interface is present and independent of the
    /// learning stack, per SPECS.md Section 12.1.
    pub has_independent_estop: bool,
    /// Operating modes this firmware/protocol combination supports.
    pub allowed_modes: Vec<robot_types::mode::OperatingMode>,
}

/// The complete versioned description of one robot embodiment.
#[derive(Clone, PartialEq, Debug)]
pub struct RobotProfile {
    /// Schema version.
    pub schema_version: SchemaVersion,
    /// Stable profile name, e.g. `"fixture-arm"`.
    pub name: Name,
    /// Profile version, independent of `schema_version`: two profiles at the
    /// same schema version can still describe different firmware revisions.
    pub profile_version: String,
    /// Joints, in an order consistent with [`RobotProfile::validate`]'s
    /// topological requirement (a parent joint before any joint whose parent
    /// link it produces).
    pub joints: Vec<JointSpec>,
    /// Links.
    pub links: Vec<LinkSpec>,
    /// Actuators.
    pub actuators: Vec<ActuatorSpec>,
    /// Sensors.
    pub sensors: Vec<SensorSpec>,
    /// The link with no parent joint.
    pub root_link: LinkId,
    /// Gravity convention this profile was authored against.
    pub gravity: GravityConvention,
    /// Firmware contract.
    pub firmware: FirmwareContract,
    /// Namespaced extension metadata, per SPECS.md Section 7.1.
    pub extensions: ExtensionTable,
}

/// Why a robot profile was rejected.
#[derive(Clone, PartialEq, Debug)]
pub enum ProfileError {
    /// Two entities of the same category shared an ID.
    DuplicateId {
        /// The category: `"joint"`, `"link"`, `"actuator"` or `"sensor"`.
        category: &'static str,
        /// The repeated ID, formatted.
        id: String,
    },
    /// Two entities of the same category shared a name.
    DuplicateName {
        /// The category.
        category: &'static str,
        /// The repeated name.
        name: String,
    },
    /// A joint's parent or child link does not exist.
    DanglingLinkReference {
        /// The joint.
        joint: Name,
        /// Which side: `"parent"` or `"child"`.
        side: &'static str,
        /// The missing link ID.
        link: LinkId,
    },
    /// An actuator's joint does not exist.
    DanglingJointReference {
        /// The actuator.
        actuator: Name,
        /// The missing joint ID.
        joint: JointId,
    },
    /// The root link does not exist among the links.
    UnknownRootLink {
        /// The declared root.
        link: LinkId,
    },
    /// A link other than the root has no incoming joint, or has more than
    /// one, or the graph otherwise fails to be a tree.
    NotATree {
        /// The link with the problem.
        link: LinkId,
        /// How many parent joints target it.
        incoming_joint_count: usize,
    },
    /// The joint graph contains a cycle.
    Cycle {
        /// A link on the cycle.
        link: LinkId,
    },
    /// A joint failed its own validation.
    InvalidJoint(crate::joint::JointError),
    /// A link failed its own validation.
    InvalidLink(crate::link::LinkError),
    /// An actuator failed its own validation.
    InvalidActuator(crate::actuator::ActuatorError),
    /// A sensor failed its own validation.
    InvalidSensor(crate::sensor::SensorError),
    /// An unknown core or (under activation) extension field was present.
    Schema(robot_types::schema::SchemaError),
    /// The firmware contract has no independent emergency stop.
    ///
    /// SPECS.md Section 12.1: "Hardware emergency stop remains independent of
    /// the learning stack." A profile that does not declare one cannot be
    /// activated on any hardware mode.
    NoIndependentEstop,
}

impl fmt::Display for ProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateId { category, id } => write!(f, "duplicate {category} id {id}"),
            Self::DuplicateName { category, name } => {
                write!(f, "duplicate {category} name {name:?}")
            }
            Self::DanglingLinkReference { joint, side, link } => {
                write!(f, "joint {joint} names unknown {side} link {link}")
            }
            Self::DanglingJointReference { actuator, joint } => {
                write!(f, "actuator {actuator} names unknown joint {joint}")
            }
            Self::UnknownRootLink { link } => write!(f, "root link {link} is not among the links"),
            Self::NotATree {
                link,
                incoming_joint_count,
            } => write!(
                f,
                "link {link} has {incoming_joint_count} incoming joints; exactly one is required \
                 except at the root"
            ),
            Self::Cycle { link } => write!(f, "joint graph has a cycle through link {link}"),
            Self::InvalidJoint(e) => write!(f, "{e}"),
            Self::InvalidLink(e) => write!(f, "{e}"),
            Self::InvalidActuator(e) => write!(f, "{e}"),
            Self::InvalidSensor(e) => write!(f, "{e}"),
            Self::Schema(e) => write!(f, "{e}"),
            Self::NoIndependentEstop => write!(
                f,
                "firmware contract declares no independent emergency stop; \
                 cannot activate on a hardware mode"
            ),
        }
    }
}

impl core::error::Error for ProfileError {}

impl From<crate::joint::JointError> for ProfileError {
    fn from(e: crate::joint::JointError) -> Self {
        Self::InvalidJoint(e)
    }
}
impl From<crate::link::LinkError> for ProfileError {
    fn from(e: crate::link::LinkError) -> Self {
        Self::InvalidLink(e)
    }
}
impl From<crate::actuator::ActuatorError> for ProfileError {
    fn from(e: crate::actuator::ActuatorError) -> Self {
        Self::InvalidActuator(e)
    }
}
impl From<crate::sensor::SensorError> for ProfileError {
    fn from(e: crate::sensor::SensorError) -> Self {
        Self::InvalidSensor(e)
    }
}
impl From<robot_types::schema::SchemaError> for ProfileError {
    fn from(e: robot_types::schema::SchemaError) -> Self {
        Self::Schema(e)
    }
}

impl RobotProfile {
    /// Checks every entity's own validity, identity uniqueness, and that the
    /// joint graph forms a tree rooted at `root_link`.
    ///
    /// This is the check every consumer runs before trusting a profile at
    /// all; it does not enforce the stricter extension-table rule that
    /// [`RobotProfile::activate`] adds.
    pub fn validate(&self) -> Result<(), ProfileError> {
        check_unique(self.joints.iter().map(|j| (j.id, &j.name)), "joint")?;
        check_unique(self.links.iter().map(|l| (l.id, &l.name)), "link")?;
        check_unique(self.actuators.iter().map(|a| (a.id, &a.name)), "actuator")?;
        check_unique(self.sensors.iter().map(|s| (s.id, &s.name)), "sensor")?;

        for j in &self.joints {
            j.validate()?;
        }
        for l in &self.links {
            l.validate()?;
        }
        for a in &self.actuators {
            a.validate()?;
        }
        for s in &self.sensors {
            s.validate()?;
        }

        if !self.links.iter().any(|l| l.id == self.root_link) {
            return Err(ProfileError::UnknownRootLink {
                link: self.root_link,
            });
        }

        for j in &self.joints {
            if !self.links.iter().any(|l| l.id == j.parent) {
                return Err(ProfileError::DanglingLinkReference {
                    joint: j.name.clone(),
                    side: "parent",
                    link: j.parent,
                });
            }
            if !self.links.iter().any(|l| l.id == j.child) {
                return Err(ProfileError::DanglingLinkReference {
                    joint: j.name.clone(),
                    side: "child",
                    link: j.child,
                });
            }
        }
        for a in &self.actuators {
            if !self.joints.iter().any(|j| j.id == a.joint) {
                return Err(ProfileError::DanglingJointReference {
                    actuator: a.name.clone(),
                    joint: a.joint,
                });
            }
        }

        for l in &self.links {
            if l.id == self.root_link {
                continue;
            }
            let incoming = self.joints.iter().filter(|j| j.child == l.id).count();
            if incoming != 1 {
                return Err(ProfileError::NotATree {
                    link: l.id,
                    incoming_joint_count: incoming,
                });
            }
        }
        // The root must have no incoming joint: a second path to it would
        // make the graph not a tree even though the count-per-link check
        // above does not look at the root.
        let root_incoming = self
            .joints
            .iter()
            .filter(|j| j.child == self.root_link)
            .count();
        if root_incoming != 0 {
            return Err(ProfileError::NotATree {
                link: self.root_link,
                incoming_joint_count: root_incoming,
            });
        }

        self.check_acyclic()?;
        Ok(())
    }

    /// Walks from the root following child joints; any link not reached, or
    /// reached twice, indicates a cycle (a tree with the right in-degrees
    /// everywhere but the root is still cyclic if it is disconnected from the
    /// root, since a disconnected component with correct in-degrees forms its
    /// own cycle).
    fn check_acyclic(&self) -> Result<(), ProfileError> {
        let mut visited = vec![false; self.links.len()];
        let index_of = |id: LinkId| self.links.iter().position(|l| l.id == id);
        let mut stack = vec![self.root_link];
        if let Some(i) = index_of(self.root_link) {
            visited[i] = true;
        }
        while let Some(current) = stack.pop() {
            for j in self.joints.iter().filter(|j| j.parent == current) {
                let Some(i) = index_of(j.child) else { continue };
                if visited[i] {
                    return Err(ProfileError::Cycle { link: j.child });
                }
                visited[i] = true;
                stack.push(j.child);
            }
        }
        if let Some(unreached) = self
            .links
            .iter()
            .zip(&visited)
            .find(|(_, v)| !**v)
            .map(|(l, _)| l.id)
        {
            return Err(ProfileError::Cycle { link: unreached });
        }
        Ok(())
    }

    /// Validates the profile for activation on a control path.
    ///
    /// Beyond [`RobotProfile::validate`], this enforces
    /// [`Strictness::Activating`] on the profile's own extension table and
    /// the independent-emergency-stop requirement of SPECS.md Section 12.1.
    /// `understood_extensions` is the caller's declared set of recognized
    /// namespaced keys.
    pub fn activate(
        &self,
        understood_extensions: &[robot_types::ids::ExtensionKey],
    ) -> Result<(), ProfileError> {
        self.validate()?;
        self.extensions
            .validate(understood_extensions, Strictness::Activating)?;
        if !self.firmware.has_independent_estop {
            return Err(ProfileError::NoIndependentEstop);
        }
        Ok(())
    }

    /// The joint whose child is `link`, if any (i.e. its parent joint).
    #[must_use]
    pub fn parent_joint_of(&self, link: LinkId) -> Option<&JointSpec> {
        self.joints.iter().find(|j| j.child == link)
    }

    /// Every joint whose parent is `link` (its children in the tree).
    pub fn child_joints_of(&self, link: LinkId) -> impl Iterator<Item = &JointSpec> {
        self.joints.iter().filter(move |j| j.parent == link)
    }

    /// A link by ID.
    #[must_use]
    pub fn link(&self, id: LinkId) -> Option<&LinkSpec> {
        self.links.iter().find(|l| l.id == id)
    }

    /// A joint by ID.
    #[must_use]
    pub fn joint(&self, id: JointId) -> Option<&JointSpec> {
        self.joints.iter().find(|j| j.id == id)
    }

    /// A joint by name.
    #[must_use]
    pub fn joint_by_name(&self, name: &str) -> Option<&JointSpec> {
        self.joints.iter().find(|j| j.name.as_str() == name)
    }

    /// The actuated joints, in profile order — the degrees of freedom a
    /// policy's action vector addresses.
    pub fn actuated_joints(&self) -> impl Iterator<Item = &JointSpec> {
        self.joints.iter().filter(|j| j.kind.is_actuated())
    }

    /// Number of actuated degrees of freedom.
    #[must_use]
    pub fn dof(&self) -> usize {
        self.actuated_joints().map(|j| j.kind.dof_count()).sum()
    }

    /// A topological order of links, root first, such that every link
    /// appears after its parent.
    ///
    /// Assumes [`RobotProfile::validate`] has already passed; a cyclic or
    /// disconnected profile produces a shorter-than-expected order rather than
    /// panicking, which callers should treat as a bug upstream of this call.
    #[must_use]
    pub fn topological_link_order(&self) -> Vec<LinkId> {
        let mut order = Vec::with_capacity(self.links.len());
        let mut stack = vec![self.root_link];
        order.push(self.root_link);
        while let Some(current) = stack.pop() {
            for j in self.child_joints_of(current) {
                order.push(j.child);
                stack.push(j.child);
            }
        }
        order
    }
}

/// Checks that no (id, name) pair repeats within a category.
fn check_unique<'a, T, I>(items: I, category: &'static str) -> Result<(), ProfileError>
where
    T: fmt::Display,
    I: Iterator<Item = (T, &'a Name)>,
{
    let mut seen_ids: Vec<String> = Vec::new();
    let mut seen_names: Vec<String> = Vec::new();
    for (id, name) in items {
        let id_text = id.to_string();
        if seen_ids.contains(&id_text) {
            return Err(ProfileError::DuplicateId {
                category,
                id: id_text,
            });
        }
        seen_ids.push(id_text);
        let name_text = name.to_string();
        if seen_names.contains(&name_text) {
            return Err(ProfileError::DuplicateName {
                category,
                name: name_text,
            });
        }
        seen_names.push(name_text);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actuator::{PdGains, ThermalLimits, TorqueSpeedEnvelope, Transmission};
    use crate::joint::JointKind;
    use crate::link::MassProperties;
    use crate::pose::RigidPose;
    use robot_types::frames::{FrameId, Vec3};
    use robot_types::ids::ActuatorId;
    use robot_types::units::{
        Amperes, Celsius, Kilograms, NewtonMetres, Radians, RadiansPerSecond, Range,
    };

    fn make_link(id: u32, name: &str) -> LinkSpec {
        let mut mp = MassProperties::MASSLESS;
        mp.mass = Kilograms(1.0);
        LinkSpec {
            id: LinkId::new(id),
            name: Name::new(name).unwrap(),
            mass: mp,
            collision: Vec::new(),
        }
    }

    fn make_joint(id: u32, name: &str, parent: u32, child: u32) -> JointSpec {
        JointSpec {
            id: JointId::new(id),
            name: Name::new(name).unwrap(),
            kind: JointKind::Revolute,
            parent: LinkId::new(parent),
            child: LinkId::new(child),
            origin: RigidPose::IDENTITY,
            axis: Vec3::unit_z(),
            limits: crate::joint::JointLimits::bounded(
                Range::<Radians>::new(Radians(-1.0), Radians(1.0)).unwrap(),
                RadiansPerSecond(1.0),
                NewtonMetres(1.0),
            ),
            sign: 1.0,
            zero_offset: 0.0,
        }
    }

    fn make_actuator(id: u32, name: &str, joint: u32) -> ActuatorSpec {
        ActuatorSpec {
            id: ActuatorId::new(id),
            name: Name::new(name).unwrap(),
            joint: JointId::new(joint),
            transmission: Transmission::DIRECT,
            envelope: TorqueSpeedEnvelope {
                stall_torque: NewtonMetres(1.0),
                corner_speed: RadiansPerSecond(0.5),
                no_load_speed: RadiansPerSecond(1.0),
            },
            gains: Some(PdGains { kp: 1.0, kd: 0.1 }),
            command_latency: Duration::from_millis(1),
            thermal: ThermalLimits {
                limit: Celsius(80.0),
                ambient: Celsius(25.0),
                time_constant: Duration::from_secs(10),
                current_at_limit: Amperes(5.0),
            },
            current_uncertainty: Amperes(0.05),
        }
    }

    fn chain_profile() -> RobotProfile {
        RobotProfile {
            schema_version: PROFILE_SCHEMA_VERSION,
            name: Name::new("fixture-arm").unwrap(),
            profile_version: "1.0".to_owned(),
            joints: vec![
                make_joint(0, "shoulder", 0, 1),
                make_joint(1, "elbow", 1, 2),
            ],
            links: vec![
                make_link(0, "base"),
                make_link(1, "upper"),
                make_link(2, "lower"),
            ],
            actuators: vec![
                make_actuator(0, "shoulder-motor", 0),
                make_actuator(1, "elbow-motor", 1),
            ],
            sensors: Vec::new(),
            root_link: LinkId::new(0),
            gravity: GravityConvention::minus_z_standard(FrameId::WORLD),
            firmware: FirmwareContract {
                version: "1.0".to_owned(),
                command_expiration: Duration::from_millis(100),
                watchdog_timeout: Duration::from_millis(200),
                has_independent_estop: true,
                allowed_modes: vec![robot_types::mode::OperatingMode::HwShadow],
            },
            extensions: ExtensionTable::new(),
        }
    }

    #[test]
    fn a_well_formed_chain_validates() {
        let p = chain_profile();
        p.validate().unwrap();
        assert_eq!(p.dof(), 2);
        assert_eq!(
            p.topological_link_order(),
            vec![LinkId::new(0), LinkId::new(1), LinkId::new(2)]
        );
    }

    #[test]
    fn duplicate_joint_ids_are_rejected() {
        let mut p = chain_profile();
        p.joints[1].id = p.joints[0].id;
        assert!(matches!(
            p.validate(),
            Err(ProfileError::DuplicateId {
                category: "joint",
                ..
            })
        ));
    }

    #[test]
    fn duplicate_names_across_categories_are_independent() {
        // A joint and a link may share a name; categories are checked
        // separately, matching "IDs MUST NOT be reused ... within" a category.
        let mut p = chain_profile();
        p.joints[0].name = Name::new("base").unwrap();
        assert!(p.validate().is_ok());
    }

    #[test]
    fn duplicate_names_within_a_category_are_rejected() {
        let mut p = chain_profile();
        p.joints[1].name = p.joints[0].name.clone();
        assert!(matches!(
            p.validate(),
            Err(ProfileError::DuplicateName {
                category: "joint",
                ..
            })
        ));
    }

    #[test]
    fn a_dangling_link_reference_is_rejected() {
        let mut p = chain_profile();
        p.joints[1].child = LinkId::new(99);
        assert!(matches!(
            p.validate(),
            Err(ProfileError::DanglingLinkReference { side: "child", .. })
        ));
    }

    #[test]
    fn a_dangling_actuator_joint_is_rejected() {
        let mut p = chain_profile();
        p.actuators[0].joint = JointId::new(99);
        assert!(matches!(
            p.validate(),
            Err(ProfileError::DanglingJointReference { .. })
        ));
    }

    #[test]
    fn an_unknown_root_link_is_rejected() {
        let mut p = chain_profile();
        p.root_link = LinkId::new(42);
        assert!(matches!(
            p.validate(),
            Err(ProfileError::UnknownRootLink { .. })
        ));
    }

    #[test]
    fn a_link_with_two_parents_is_not_a_tree() {
        let mut p = chain_profile();
        // Add a second joint also producing link 2.
        p.joints.push(make_joint(2, "extra", 0, 2));
        assert!(
            matches!(p.validate(), Err(ProfileError::NotATree { link, .. }) if link == LinkId::new(2))
        );
    }

    #[test]
    fn a_disconnected_component_is_reported_as_a_cycle() {
        // Links 3 and 4 form a 2-cycle fully satisfying the in-degree check
        // (each has exactly one incoming joint) but never reach the root.
        let mut p = chain_profile();
        p.links.push(make_link(3, "island-a"));
        p.links.push(make_link(4, "island-b"));
        p.joints.push(make_joint(2, "island-01", 4, 3));
        p.joints.push(make_joint(3, "island-10", 3, 4));
        assert!(matches!(p.validate(), Err(ProfileError::Cycle { .. })));
    }

    #[test]
    fn root_link_may_not_have_an_incoming_joint() {
        let mut p = chain_profile();
        p.joints.push(make_joint(2, "back-to-root", 2, 0));
        assert!(
            matches!(p.validate(), Err(ProfileError::NotATree { link, .. }) if link == LinkId::new(0))
        );
    }

    #[test]
    fn activation_requires_an_independent_estop() {
        let mut p = chain_profile();
        p.firmware.has_independent_estop = false;
        assert!(matches!(
            p.activate(&[]),
            Err(ProfileError::NoIndependentEstop)
        ));
        p.firmware.has_independent_estop = true;
        assert!(p.activate(&[]).is_ok());
    }

    #[test]
    fn activation_rejects_an_unknown_extension_but_validate_does_not() {
        let mut p = chain_profile();
        let key = robot_types::ids::ExtensionKey::parse("vendor.acme:thermal_curve").unwrap();
        p.extensions.insert(
            key.clone(),
            robot_types::schema::ExtensionValue::RealList(vec![1.0, 2.0]),
        );
        assert!(
            p.validate().is_ok(),
            "validate() does not police extensions"
        );
        assert!(matches!(p.activate(&[]), Err(ProfileError::Schema(_))));
        assert!(p.activate(&[key]).is_ok());
    }

    #[test]
    fn parent_and_child_lookups_navigate_the_tree() {
        let p = chain_profile();
        assert!(p.parent_joint_of(LinkId::new(0)).is_none());
        assert_eq!(
            p.parent_joint_of(LinkId::new(1)).unwrap().name.as_str(),
            "shoulder"
        );
        let children: Vec<_> = p
            .child_joints_of(LinkId::new(0))
            .map(|j| j.name.as_str())
            .collect();
        assert_eq!(children, vec!["shoulder"]);
        assert_eq!(p.joint_by_name("elbow").unwrap().id, JointId::new(1));
        assert!(p.joint_by_name("wrist").is_none());
    }
}

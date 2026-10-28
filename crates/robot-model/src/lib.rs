//! The kinematic tree, inertias, actuator limits, sensor placement and the
//! robot profile of SPECS.md Section 7.1.
//!
//! A [`profile::RobotProfile`] is the complete versioned description of one
//! robot embodiment: its [`joint`]s and [`link`]s form the kinematic tree,
//! its [`actuator`]s and [`sensor`]s carry the hardware facts the runtime
//! needs, and [`kinematics::forward_kinematics`] is the one unambiguous
//! computation this crate performs over that tree. [`urdf`] imports the
//! kinematic/mass subset of URDF that this project supports, rejecting every
//! unsupported construct by name rather than dropping it silently.
//!
//! Local, structural poses — a joint's origin, a sensor's mounting offset, a
//! collision shape's placement — use [`pose::RigidPose`], not
//! `robot_types::frames::Transform`: that type's named-frame checking exists
//! for genuinely cross-cutting project frames, which C-04 leaves open, and a
//! per-link synthetic frame identity would not be one of those.
//!
//! Every category of entity carries a typed, stable ID from `robot-types`;
//! [`profile::RobotProfile::validate`] is the single place uniqueness, graph
//! connectivity and each entity's own validity are checked, and
//! [`profile::RobotProfile::activate`] adds the stricter extension-table rule
//! SPECS.md Section 7.1 requires before a profile reaches a control path.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod actuator;
pub mod fixtures;
pub mod importer;
pub mod joint;
pub mod kinematics;
pub mod link;
pub mod pose;
pub mod profile;
pub mod sensor;
pub mod urdf;

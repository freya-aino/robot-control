//! Kinematic and dynamic feasibility reports, per SPECS.md Section 8.3.
//!
//! "A validation report must separate kinematic error, dynamics
//! violations, contact consistency and task success. No universal fidelity
//! percentage is a release criterion." [`report::ValidationReport`] keeps
//! those as separate fields, and [`report::AcceptanceThresholds`] is a
//! caller-supplied value rather than a constant.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod contact;
pub mod dynamics;
pub mod kinematic;
pub mod package;
pub mod report;
pub mod tracking;
pub mod trajectory;
pub mod validate;

#[cfg(test)]
mod test_support;

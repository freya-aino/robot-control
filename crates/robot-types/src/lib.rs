//! Units, frames, time, typed identifiers and the record schemas that every
//! other crate in this workspace agrees on.
//!
//! This crate is the bottom of the dependency graph. It holds the contracts of
//! SPECS.md Section 7, the evidence states of Section 3, the operating modes of
//! Section 4 and the tier machinery of Section 5.1 — the things that must mean
//! the same thing in the learner, the runtime and the tooling, because a schema
//! that two crates interpret differently is not a schema.
//!
//! It has no third-party dependencies. SPECS.md Appendix B.1 starts every
//! surveyed candidate at `unverified` and Appendix B.6 admits none until its
//! gates pass, so the default build's dependency closure is empty by
//! construction rather than by preference; see
//! `docs/decisions/0003-std-only-default-closure.md`.
//!
//! # What is deliberately *not* decided here
//!
//! Appendix C leaves the canonical coordinate frame, gravity direction, root
//! convention and quaternion ordering open under C-04. This crate therefore
//! supplies [`frames::Transform`], which *names* the frames it maps between, and
//! [`frames::Quat`], whose components are reached by name — but it declares no
//! project-wide convention, because there is not yet a decision record to
//! declare.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod bundle;
pub mod episode;
pub mod evidence;
pub mod frames;
pub mod ids;
pub mod math;
pub mod mode;
pub mod motion;
pub mod obs;
pub mod rng;
pub mod schema;
pub mod tier;
pub mod time;
pub mod transition;
pub mod units;

/// The schema versions this crate currently reads and writes.
///
/// Collected in one place so a migration review can see every record type's
/// version at once, as the change-control rule of SPECS.md's document control
/// table requires for changes to Section 7.
pub mod versions {
    use crate::schema::SchemaVersion;

    /// Canonical motion record, SPECS.md Section 7.2.
    pub const MOTION: SchemaVersion = crate::motion::MOTION_SCHEMA_VERSION;
    /// Robot episode record, SPECS.md Section 7.3.
    pub const EPISODE: SchemaVersion = crate::episode::EPISODE_SCHEMA_VERSION;
    /// Policy bundle manifest, SPECS.md Section 7.4.
    pub const BUNDLE: SchemaVersion = crate::bundle::BUNDLE_SCHEMA_VERSION;
    /// Transition record, SPECS.md Section 7.6.
    pub const TRANSITION: SchemaVersion = crate::transition::TRANSITION_SCHEMA_VERSION;
}

/// Convenience re-exports for the types that appear in nearly every crate.
pub mod prelude {
    pub use crate::evidence::{Claim, EvidenceState};
    pub use crate::frames::{FrameId, Point, Quat, Transform, Vec3};
    pub use crate::ids::{ActuatorId, ContactSiteId, EnvId, JointId, LinkId, Name, SensorId};
    pub use crate::math::Matrix;
    pub use crate::mode::{ControlSource, ModeContext, OperatingMode};
    pub use crate::rng::{Rng, SeedStream};
    pub use crate::schema::{SchemaError, SchemaVersion, Strictness};
    pub use crate::tier::{Generation, Tier, TierOutput};
    pub use crate::time::{ClockId, Duration, MonotonicClock, StampKind, Timestamp};
    pub use crate::units::{
        Hertz, Kilograms, Metres, MetresPerSecond, NewtonMetres, Newtons, Radians,
        RadiansPerSecond, Range,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_versions_are_declared() {
        assert_eq!(versions::MOTION.major, 1);
        assert_eq!(versions::EPISODE.major, 1);
        assert_eq!(versions::BUNDLE.major, 1);
        assert_eq!(versions::TRANSITION.major, 1);
    }
}

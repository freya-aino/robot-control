//! The independent supervisor of SPECS.md Section 12.1: the last stage
//! before the actuator interface.
//!
//! "A supervisor independent of strategy and personality checks observation
//! freshness, finite commands, position/velocity/torque-speed limits,
//! temperatures, communications health, contact permissions and current
//! operating mode... Clamp-and-continue is not sufficient for every fault;
//! each fault has a validated response appropriate to the robot."
//!
//! This crate is not in SPECS.md §6's workspace table; its extraction from
//! `robot-control` is justified in
//! `docs/decisions/0004-nn-and-supervisor-crate-extraction.md`, and the
//! justification is this crate's own dependency list: it names only
//! `robot-types` and `robot-model`, so it cannot import the policy,
//! retrieval or personality crates — a build-time fact, not a review
//! convention.
//!
//! [`policy::FaultPolicy`] is where "clamp-and-continue is not sufficient
//! for every fault" becomes a checked invariant: a policy that assigns
//! clamping to a category SPECS.md §12.2's hazard register requires
//! escalation for (a stale observation, a lost link, a non-finite command,
//! actuation attempted where it is not permitted) is refused at
//! construction.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod fault;
pub mod policy;
pub mod supervisor;

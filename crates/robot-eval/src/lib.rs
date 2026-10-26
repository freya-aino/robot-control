//! Acceptance profiles, statistics and regression gates, per SPECS.md
//! Section 14.
//!
//! [`acceptance::AcceptanceProfile`] carries every field Section 14
//! requires and makes "immutable once a run has been evaluated against
//! them" structural: [`acceptance::AcceptanceProfile::revise`] is the only
//! way to change a threshold, and it always returns a new, higher-versioned
//! profile rather than mutating the one a run was scored against.
//! [`stats::summarize`] is the "multiple seeds and confidence intervals"
//! statistic the section asks for.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod acceptance;
pub mod stats;

//! Development-time automation, the evidence ledger and the dependency
//! knowledge base, per SPECS.md Sections 2.3.1 and 13.1.
//!
//! This crate is developer/CI-only tooling: no shipping crate in this
//! workspace depends on it. [`knowledge_base`] is the typed form of
//! SPECS.md Section 13.1's citation and staleness-on-version-change rules
//! for the dependency knowledge base. [`exclusion_gate`] holds the checks
//! behind the `exclusion-gate` binary that `devenv test` and the git hook
//! run: no excluded runtime in the lockfile, a path-only default closure,
//! and no shipping crate depending on this one (Section 2.1 item 4, row
//! A-25).

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod exclusion_gate;
pub mod knowledge_base;

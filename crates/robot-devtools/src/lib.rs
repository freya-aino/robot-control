//! Development-time automation, the evidence ledger and the dependency
//! knowledge base, per SPECS.md Sections 2.3.1 and 13.1.
//!
//! This crate is developer/CI-only tooling: no shipping crate in this
//! workspace depends on it. [`knowledge_base`] is the only module so far —
//! it is the typed form of SPECS.md Section 13.1's citation and
//! staleness-on-version-change rules for the dependency knowledge base.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod knowledge_base;

//! Skill descriptors, exact retrieval and the R0/R1/R2 tiers of SPECS.md
//! Section 10.6.
//!
//! [`descriptor::SkillDescriptor`] is the typed, authoritative record of
//! Section 10.2. [`index::ExactIndex`] is the default retrieval mechanism
//! of Section 10.1 and Appendix B.2.5, with a zero-allocation query path for
//! the R2 tier. [`tiers::WorkingSet`] and [`tiers::Router`] separate R1
//! (between-episode working-set publication) from R2 (in-loop selection)
//! as distinct types, per Section 10.6's requirement that the tiers "MUST
//! be distinguished in design, in configuration and in every report."
//!
//! R0 (offline corpus retrieval for training data, SPECS.md Section 9.7) is
//! not this crate's concern — it is `robot-learn`'s offline track, kept
//! deliberately unconflated with the runtime router here.
//!
//! Not implemented in this revision: learned embeddings (Section 10.1
//! starts with "structured skill tags and numerical features" only), the
//! Markdown skill-descriptor catalog of Section 10.1, and the
//! feasibility-filter/hysteresis machinery of Sections 10.2–10.4 beyond the
//! residency check `tiers::Router::switch_to` performs. These are named
//! here rather than silently assumed complete.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod descriptor;
pub mod index;
pub mod tiers;

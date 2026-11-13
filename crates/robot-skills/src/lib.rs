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
//! In-loop selection is [`selection::Selector`]: the state-conditioned
//! feasibility filter, out-of-distribution and margin checks, transition
//! contracts, dwell, hysteresis, recovery preemption and routing counters
//! of Sections 10.1 to 10.4. [`cache`] stages optional experts off the
//! control thread with the activation checks of Section 10.3, and
//! [`catalog`] is Section 10.1's optional lexical Markdown catalog, which
//! can only reorder numerical candidates. Learned embeddings are not
//! implemented: Section 10.1 adds them "only when they improve a measured
//! outcome", and no such measurement exists.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod cache;
pub mod catalog;
pub mod descriptor;
pub mod index;
pub mod selection;
pub mod tiers;

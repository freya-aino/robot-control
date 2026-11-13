//! Bounded expressive intent over validated skills, per SPECS.md
//! Section 12.5.
//!
//! | Module | Contents |
//! | --- | --- |
//! | [`style`] | `StyleConfig`: a sampled, versioned configuration over already-validated skills, validated before use |
//! | [`params`] | Expressive dimensions as positions inside validated ranges, so no style can widen an envelope |
//! | [`arbiter`] | When expression may act: never against recovery or the supervisor, only along a skill's transition contract |
//! | [`memory`] | Session-scoped opponent memory (C-17's interim position), degrading to opponent-agnostic behavior |
//! | [`evaluation`] | Recognizability against chance, and task and stability non-degradation against the neutral configuration |
//!
//! Nothing here calls a model or synthesizes motion: Section 12.5's "no
//! runtime model call produces posture, cadence, celebration or tactical
//! preference" holds because no type in this crate can carry one.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod arbiter;
pub mod evaluation;
pub mod memory;
pub mod params;
pub mod style;

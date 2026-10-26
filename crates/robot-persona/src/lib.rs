//! Bounded expressive intent over validated skills, per SPECS.md
//! Section 12.5.
//!
//! [`style::StyleConfig`] is "a sampled, versioned configuration over
//! already-validated skills" — not generated behavior. Enforced
//! structurally: the type has no envelope-widening field, and
//! [`style::select_expressive_skill`] can only return a skill already
//! present in both the style's own preferences and a caller-supplied
//! validated allowlist. Persistent identity and opponent memory (also
//! Section 12.5) are not implemented in this revision.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod style;

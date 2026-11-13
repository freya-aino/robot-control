//! Acceptance and evaluation, per SPECS.md Section 14.
//!
//! | Module | Contents |
//! | --- | --- |
//! | [`acceptance`] | `AcceptanceProfile`: every field Section 14 requires, immutable once evaluated, stored as `acceptance-profile/1` TOML with a content ID |
//! | [`stats`] | Seed summaries with normal and Student-t intervals, Welch's test, TOST equivalence |
//! | [`distribution`] | Student's t CDF and quantile (std-only) |
//! | [`evaluate`] | Pass/fail/inconclusive by interval against a profile; the regression gate between evaluations of one profile |
//! | [`matrix`] | The fifty acceptance rows, closed only by an artifact recorded in the evidence ledger |

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod acceptance;
pub mod distribution;
pub mod evaluate;
pub mod matrix;
pub mod stats;

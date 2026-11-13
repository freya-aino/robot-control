//! The motion processing pipeline of SPECS.md Section 8.2, items 3 to 6.
//!
//! | Item | Module |
//! | --- | --- |
//! | 3. correct axes/units, validate finite values and timestamps, mark gaps | [`condition`] |
//! | 4. filter and resample with documented phase delay | [`filter`], [`resample`] |
//! | 5. ground plane and contact intervals with uncertainty | [`contact`] |
//! | 6. stance/locomotion/strike/defense/recovery/transition windows | [`segment`] |
//!
//! Augmentation (SPECS.md Section 6) is in [`augment`].
//! [`kinematics`] supplies the global joint positions items 5 and 6 need.
//! Items 1 and 2 (inspection, immutable originals) belong to `motion-io` and
//! `robot-artifacts`; items 7 to 9 to `motion-retarget` and
//! `motion-validate`. Every operation here records a provenance step on the
//! record it changes.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod augment;
pub mod condition;
pub mod contact;
pub mod filter;
pub mod kinematics;
pub mod resample;
pub mod segment;

#[cfg(test)]
mod test_support;

/// Software revision recorded in every provenance step this crate writes.
pub const REVISION: &str = concat!("motion-process/", env!("CARGO_PKG_VERSION"));

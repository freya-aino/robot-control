//! Resampling, filtering, contact estimation and segmentation, per
//! SPECS.md Section 8.2.
//!
//! [`resample::resample`] is the only step implemented in this revision.
//! Filtering (beyond the implicit linear-interpolation smoothing resampling
//! performs), ground-plane/contact estimation and stance/strike
//! segmentation — items 3, 5 and 6 of Section 8.2's pipeline — are not
//! implemented, named here rather than silently assumed complete.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod resample;

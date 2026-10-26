//! Inspection data layer for skeletons, contacts and tracking error, per
//! SPECS.md §6 and PLAN.md's Phase 2 viewer requirement.
//!
//! This crate stops at data: [`frame::ViewerFrame`] is one inspection
//! snapshot and [`timeline::Timeline`] sequences them for synchronized
//! playback. No rendering toolkit is wired in — SPECS.md §11's viewer
//! shortlist (Rerun, egui, `urdf-viz`) has not passed the
//! dependency-qualification gate of Appendix B.6, so there is nothing
//! qualified to render with yet. See each module's doc comment and this
//! crate's README for the reasoning.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod frame;
pub mod timeline;

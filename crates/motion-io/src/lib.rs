//! Motion import and export, per SPECS.md Section 8.1.
//!
//! [`bvh::parse`] is the BVH ingestion PLAN.md's first executable backlog
//! names as item 5. C3D and other numerical human-body formats (Section
//! 8.1's "followed by C3D and supported numerical human-body formats") are
//! not implemented in this revision.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod bvh;

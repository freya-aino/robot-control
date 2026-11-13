//! Motion import and export, per SPECS.md Section 8.1.
//!
//! File formats: [`bvh`] (import) and [`bvh_export`], [`c3d`], and
//! [`amass`] over [`npz`] (with its own [`inflate`]). Live capture:
//! [`capture`]'s `robot-capture/1` protocol adapters. [`import`] detects a
//! format, stores the original in a content-addressed blob store before
//! parsing, and inspects the result (SPECS.md Section 8.2 items 1 and 2).

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod amass;
pub mod bvh;
pub mod bvh_export;
pub mod c3d;
pub mod capture;
pub mod import;
pub mod inflate;
pub mod npz;

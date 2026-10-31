//! A std-only reader for OpenUSD layers: the binary USDC "crate" container,
//! minimal layer composition, and a USDA text reader and writer.
//!
//! No third-party crate, no FFI and no `unsafe`. The byte layouts are taken
//! from OpenUSD `pxr/usd/sdf/crateFile.cpp`, `integerCoding.cpp`,
//! `pxr/base/tf/fastCompression.cpp` and `pxr/base/tf/pxrLZ4/lz4.cpp` at commit
//! `84d8ed1ae0337ce30dca8d91a3fbe7f04eebc358`, which the crate module docs cite.
//! LZ4 blocks follow the public LZ4 block format. The reader was checked
//! against every block in the four vendored files, each of which decodes to
//! the size its container states.
//!
//! Layers are read into a [`layer::Layer`] (a flat list of specs with typed
//! fields). [`stage::Stage::compose`] merges sublayers into a prim tree.
//! [`usda::write_usda`] writes a stage as text, and [`usda::parse_layer`] reads
//! back the subset it writes. Anything this crate does not decode is reported
//! by name in [`error::UsdError`].

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod compression;
pub mod crate_file;
pub mod diagnostics;
pub mod error;
pub mod layer;
pub mod replay;
pub mod stage;
pub mod usda;
pub mod value;

pub use error::{Result, UsdError};
pub use layer::{Layer, Spec, SpecKind};
pub use stage::{Prim, Property, PropertyKind, Stage};
pub use value::{Arc, ListOp, Specifier, Value, Variability};

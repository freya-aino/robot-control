//! Layers: the flat list of spec records a USD file contributes.
//!
//! A [`Layer`] is the format-independent view of one file. The binary crate
//! reader ([`crate::crate_file`]) and the text reader ([`crate::usda`]) both
//! produce one, and [`crate::stage::Stage::compose`] consumes them.

use crate::error::{Result, UsdError};
use crate::value::Value;
use std::path::{Path, PathBuf};

/// `SdfSpecType`, in OpenUSD's enum order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecKind {
    /// Unknown (never valid in a file).
    Unknown,
    /// An attribute property spec.
    Attribute,
    /// An attribute connection spec.
    Connection,
    /// An expression spec.
    Expression,
    /// A mapper spec.
    Mapper,
    /// A mapper argument spec.
    MapperArg,
    /// A prim spec.
    Prim,
    /// The pseudo-root spec (`/`).
    PseudoRoot,
    /// A relationship property spec.
    Relationship,
    /// A relationship target spec.
    RelationshipTarget,
    /// A variant spec.
    Variant,
    /// A variant set spec.
    VariantSet,
}

impl SpecKind {
    /// Maps the stored `SdfSpecType` integer. Returns `None` for values
    /// outside the enum.
    pub fn from_u32(v: u32) -> Option<SpecKind> {
        use SpecKind::*;
        Some(match v {
            0 => Unknown,
            1 => Attribute,
            2 => Connection,
            3 => Expression,
            4 => Mapper,
            5 => MapperArg,
            6 => Prim,
            7 => PseudoRoot,
            8 => Relationship,
            9 => RelationshipTarget,
            10 => Variant,
            11 => VariantSet,
            _ => return None,
        })
    }
}

/// One spec: a path, its kind and its authored fields in file order.
#[derive(Debug, Clone, PartialEq)]
pub struct Spec {
    /// The SdfPath as text, for example `/Robot/left_hip` or `/Robot/left_hip.physics:mass`.
    pub path: String,
    /// The spec kind.
    pub kind: SpecKind,
    /// Authored fields, `(name, value)`, in stored order.
    pub fields: Vec<(String, Value)>,
}

impl Spec {
    /// Returns the value of a field, if authored.
    pub fn field(&self, name: &str) -> Option<&Value> {
        self.fields.iter().find(|(n, _)| n == name).map(|(_, v)| v)
    }
}

/// A layer: its identifier (file path) and all of its specs.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    /// The file the layer was read from, or a label for text sources.
    pub identifier: String,
    /// All specs, in stored order.
    pub specs: Vec<Spec>,
}

/// The on-disk format of a layer file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerFormat {
    /// Binary crate (`PXR-USDC`).
    Crate,
    /// Text (`#usda`).
    Text,
}

/// Detects the format of a layer from its first bytes.
pub fn detect_format(bytes: &[u8]) -> Option<LayerFormat> {
    if bytes.starts_with(b"PXR-USDC") {
        Some(LayerFormat::Crate)
    } else if bytes.starts_with(b"#usda") {
        Some(LayerFormat::Text)
    } else {
        None
    }
}

/// Reads a layer file of either format.
pub fn read_layer_file(path: &Path) -> Result<Layer> {
    let bytes =
        std::fs::read(path).map_err(|e| UsdError::Io(format!("{}: {e}", path.display())))?;
    let id = path.display().to_string();
    match detect_format(&bytes) {
        Some(LayerFormat::Crate) => {
            let file = crate::crate_file::CrateFile::parse(bytes)?;
            Ok(Layer {
                identifier: id,
                specs: file.specs()?,
            })
        }
        Some(LayerFormat::Text) => {
            let text = std::str::from_utf8(&bytes)
                .map_err(|e| UsdError::Malformed(format!("{id}: not UTF-8: {e}")))?;
            crate::usda::parse_layer(text, &id)
        }
        None => Err(UsdError::Malformed(format!(
            "{id}: neither PXR-USDC nor #usda"
        ))),
    }
}

/// Resolves a sublayer reference relative to the layer that names it.
pub fn resolve_relative(base_layer: &str, reference: &str) -> PathBuf {
    let r = Path::new(reference);
    if r.is_absolute() {
        return r.to_path_buf();
    }
    Path::new(base_layer)
        .parent()
        .map(|d| d.join(r))
        .unwrap_or_else(|| r.to_path_buf())
}

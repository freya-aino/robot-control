//! Error type shared by the crate reader, the composer and the text format.

use std::fmt;

/// Every failure this crate reports. Unsupported features are named, never
/// skipped silently.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UsdError {
    /// A file could not be read.
    Io(String),
    /// The bytes do not follow the format (bad magic, bad offsets, corrupt
    /// compressed data, truncated section, ...).
    Malformed(String),
    /// A value type id in a crate file that this reader does not decode.
    UnsupportedValueType {
        /// The `TypeEnum` id as stored in the value representation.
        id: u8,
        /// The OpenUSD name of that type (from `crateDataTypes.h`).
        name: &'static str,
    },
    /// A USD feature that is valid USD but not implemented here.
    Unsupported(String),
    /// A USDA text document that does not parse.
    Parse {
        /// One-based line number of the offending token.
        line: usize,
        /// What was expected or found.
        message: String,
    },
}

impl fmt::Display for UsdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UsdError::Io(m) => write!(f, "I/O error: {m}"),
            UsdError::Malformed(m) => write!(f, "malformed USD data: {m}"),
            UsdError::UnsupportedValueType { id, name } => {
                write!(f, "unsupported crate value type id {id} ({name})")
            }
            UsdError::Unsupported(m) => write!(f, "unsupported USD feature: {m}"),
            UsdError::Parse { line, message } => {
                write!(f, "USDA parse error at line {line}: {message}")
            }
        }
    }
}

impl std::error::Error for UsdError {}

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, UsdError>;

pub(crate) fn malformed<T>(msg: impl Into<String>) -> Result<T> {
    Err(UsdError::Malformed(msg.into()))
}

//! Schema versions and the compatibility rule applied before activation.
//!
//! Every record type in SPECS.md Section 7 carries a `schema_version`, and
//! Section 7.1 requires that "Unknown core fields and unknown extensions MUST be
//! rejected before a profile is activated on a control path" while "Non-control
//! tooling MUST preserve unknown namespaced extensions without interpreting
//! them".
//!
//! Those are two different behaviours for the same input, so the decision is
//! made by an explicit [`Strictness`] rather than by which code path happens to
//! be reading.

use crate::ids::{ExtensionKey, Name};
use core::fmt;

/// A major/minor schema version.
///
/// A major change is breaking and requires a migration (SPECS.md Section 7.1,
/// "Schema changes MUST provide explicit migrations"). A minor change adds
/// optional fields only.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SchemaVersion {
    /// Breaking version.
    pub major: u16,
    /// Additive version.
    pub minor: u16,
}

impl SchemaVersion {
    /// A version from its parts.
    #[inline]
    #[must_use]
    pub const fn new(major: u16, minor: u16) -> Self {
        Self { major, minor }
    }

    /// Whether a reader supporting `self` can read data written at `data`.
    ///
    /// Readable when the major versions agree and the reader is at least as new
    /// as the data. A newer *minor* in the data is **not** readable on a control
    /// path: the reader cannot know whether the added field changed the meaning
    /// of a field it does understand. Tooling may still open it to preserve
    /// unknown content, which [`Strictness::Preserving`] expresses.
    #[must_use]
    pub const fn can_read(self, data: Self) -> bool {
        self.major == data.major && self.minor >= data.minor
    }

    /// Parses `MAJOR.MINOR`.
    pub fn parse(text: &str) -> Result<Self, SchemaError> {
        let (major, minor) = text.split_once('.').ok_or_else(|| SchemaError::Malformed {
            text: text.to_owned(),
        })?;
        let major = major.parse::<u16>().map_err(|_| SchemaError::Malformed {
            text: text.to_owned(),
        })?;
        let minor = minor.parse::<u16>().map_err(|_| SchemaError::Malformed {
            text: text.to_owned(),
        })?;
        Ok(Self { major, minor })
    }
}

impl fmt::Display for SchemaVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

impl fmt::Debug for SchemaVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{self}")
    }
}

/// How strictly a reader treats content it does not recognize.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Strictness {
    /// Reject unknown core fields and unknown extensions.
    ///
    /// Required before activation on a control path (SPECS.md Section 7.1).
    Activating,
    /// Accept and carry unknown namespaced extensions without interpreting
    /// them, while still rejecting unknown *core* fields.
    ///
    /// This is the tooling mode that makes a round trip lossless.
    Preserving,
}

/// Why a record could not be accepted.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum SchemaError {
    /// The version string was not `MAJOR.MINOR`.
    Malformed {
        /// The text that failed to parse.
        text: String,
    },
    /// The reader cannot read this major/minor combination.
    Incompatible {
        /// What the reader supports.
        reader: SchemaVersion,
        /// What the data declares.
        data: SchemaVersion,
    },
    /// A core field the reader does not know about was present.
    UnknownCoreField {
        /// The field name.
        field: String,
    },
    /// A namespaced extension was present and the reader is activating.
    UnknownExtension {
        /// The extension key.
        key: ExtensionKey,
    },
    /// A required field was absent.
    MissingField {
        /// The field name.
        field: &'static str,
    },
    /// A field held a value outside its declared domain.
    InvalidValue {
        /// The field name.
        field: &'static str,
        /// Why the value was rejected.
        reason: String,
    },
    /// Two entities claimed the same stable identifier or name.
    DuplicateIdentity {
        /// The repeated identity.
        identity: String,
    },
    /// No migration exists between two major versions.
    NoMigration {
        /// Version held.
        from: SchemaVersion,
        /// Version wanted.
        to: SchemaVersion,
    },
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Malformed { text } => write!(f, "malformed schema version {text:?}"),
            Self::Incompatible { reader, data } => {
                write!(f, "reader supports {reader} but data declares {data}")
            }
            Self::UnknownCoreField { field } => write!(f, "unknown core field {field:?}"),
            Self::UnknownExtension { key } => {
                write!(f, "unknown extension {key} present while activating")
            }
            Self::MissingField { field } => write!(f, "missing required field {field:?}"),
            Self::InvalidValue { field, reason } => {
                write!(f, "field {field:?} is invalid: {reason}")
            }
            Self::DuplicateIdentity { identity } => {
                write!(f, "duplicate identity {identity:?}")
            }
            Self::NoMigration { from, to } => {
                write!(f, "no migration from {from} to {to}")
            }
        }
    }
}

impl core::error::Error for SchemaError {}

/// A value an extension table may hold.
///
/// SPECS.md Section 7.1 requires extension tables with "declared value types".
/// The set is closed and small so that preserving an extension is a copy rather
/// than an interpretation.
#[derive(Clone, PartialEq, Debug)]
pub enum ExtensionValue {
    /// A boolean.
    Bool(bool),
    /// A signed integer.
    Integer(i64),
    /// A real number in the unit the extension's documentation declares.
    Real(f64),
    /// A text value.
    Text(String),
    /// A homogeneous list of reals.
    RealList(Vec<f64>),
    /// A homogeneous list of text values.
    TextList(Vec<String>),
}

impl ExtensionValue {
    /// The declared type name, as written in a record.
    #[must_use]
    pub const fn type_name(&self) -> &'static str {
        match self {
            Self::Bool(_) => "bool",
            Self::Integer(_) => "integer",
            Self::Real(_) => "real",
            Self::Text(_) => "text",
            Self::RealList(_) => "real-list",
            Self::TextList(_) => "text-list",
        }
    }
}

/// A set of namespaced extensions carried alongside a core record.
#[derive(Clone, Default, PartialEq, Debug)]
pub struct ExtensionTable {
    entries: Vec<(ExtensionKey, ExtensionValue)>,
}

impl ExtensionTable {
    /// An empty table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts or replaces an entry, returning the previous value.
    pub fn insert(&mut self, key: ExtensionKey, value: ExtensionValue) -> Option<ExtensionValue> {
        match self.entries.iter_mut().find(|(k, _)| *k == key) {
            Some(slot) => Some(core::mem::replace(&mut slot.1, value)),
            None => {
                self.entries.push((key, value));
                self.entries.sort_by(|a, b| a.0.cmp(&b.0));
                None
            }
        }
    }

    /// Looks up an entry.
    #[must_use]
    pub fn get(&self, key: &ExtensionKey) -> Option<&ExtensionValue> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Entries in sorted key order, which makes a re-serialized table
    /// byte-identical to the one that was read.
    pub fn iter(&self) -> impl Iterator<Item = (&ExtensionKey, &ExtensionValue)> {
        self.entries.iter().map(|(k, v)| (k, v))
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when the table holds no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Checks this table against the set of extensions a reader understands.
    ///
    /// Under [`Strictness::Activating`] an unrecognized key is an error; under
    /// [`Strictness::Preserving`] it is retained untouched.
    pub fn validate(
        &self,
        understood: &[ExtensionKey],
        strictness: Strictness,
    ) -> Result<(), SchemaError> {
        if strictness == Strictness::Preserving {
            return Ok(());
        }
        for (key, _) in &self.entries {
            if !understood.contains(key) {
                return Err(SchemaError::UnknownExtension { key: key.clone() });
            }
        }
        Ok(())
    }
}

/// A capability tag, which SPECS.md Section 7.1 defines as "a claim to validate
/// against the profile's typed fields, not a substitute for them".
///
/// [`CapabilityTag::validate_against`] takes the typed evidence as a predicate
/// result so that a tag can never be the only thing asserting a capability.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CapabilityTag {
    /// The tag's name.
    pub name: Name,
}

impl CapabilityTag {
    /// A tag from a name.
    pub fn new(name: &str) -> Result<Self, crate::ids::NameError> {
        Ok(Self {
            name: Name::new(name)?,
        })
    }

    /// Confirms the tag against typed evidence.
    ///
    /// `supported_by_typed_fields` is the result of inspecting the record's own
    /// fields. A tag that the fields do not support is a defect in the record,
    /// reported here rather than trusted.
    pub fn validate_against(&self, supported_by_typed_fields: bool) -> Result<(), SchemaError> {
        if supported_by_typed_fields {
            Ok(())
        } else {
            Err(SchemaError::InvalidValue {
                field: "capability_tags",
                reason: format!(
                    "tag {} is not supported by the record's typed fields",
                    self.name
                ),
            })
        }
    }
}

/// A migration between two major schema versions.
///
/// SPECS.md Section 7.1: "each migration MUST preserve or report loss of
/// meaning". [`MigrationOutcome`] is how that report is returned.
pub trait Migration {
    /// The record type being migrated.
    type Record;

    /// Version this migration reads.
    fn from(&self) -> SchemaVersion;

    /// Version this migration writes.
    fn to(&self) -> SchemaVersion;

    /// Performs the migration.
    fn migrate(&self, record: Self::Record) -> Result<MigrationOutcome<Self::Record>, SchemaError>;
}

/// The result of a migration, including anything it could not carry across.
#[derive(Clone, Debug)]
pub struct MigrationOutcome<T> {
    /// The migrated record.
    pub record: T,
    /// Fields whose meaning could not be preserved, named for the report.
    ///
    /// An empty vector means the migration was lossless; it is never an
    /// assumption, it is the migration's own statement.
    pub lost: Vec<String>,
}

impl<T> MigrationOutcome<T> {
    /// A lossless outcome.
    #[must_use]
    pub const fn lossless(record: T) -> Self {
        Self {
            record,
            lost: Vec::new(),
        }
    }

    /// True when nothing was dropped.
    #[must_use]
    pub fn is_lossless(&self) -> bool {
        self.lost.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compatibility_is_same_major_and_reader_not_older() {
        let reader = SchemaVersion::new(1, 3);
        assert!(reader.can_read(SchemaVersion::new(1, 0)));
        assert!(reader.can_read(SchemaVersion::new(1, 3)));
        assert!(!reader.can_read(SchemaVersion::new(1, 4)));
        assert!(!reader.can_read(SchemaVersion::new(2, 0)));
        assert!(!reader.can_read(SchemaVersion::new(0, 9)));
    }

    #[test]
    fn version_parsing_rejects_junk() {
        assert_eq!(
            SchemaVersion::parse("2.11").unwrap(),
            SchemaVersion::new(2, 11)
        );
        assert!(SchemaVersion::parse("2").is_err());
        assert!(SchemaVersion::parse("2.x").is_err());
        assert!(SchemaVersion::parse("").is_err());
    }

    #[test]
    fn activating_rejects_unknown_extensions_but_tooling_preserves_them() {
        let mut table = ExtensionTable::new();
        let key = ExtensionKey::parse("vendor.acme:thermal_curve").unwrap();
        table.insert(key.clone(), ExtensionValue::RealList(vec![1.0, 2.0]));

        assert!(matches!(
            table.validate(&[], Strictness::Activating),
            Err(SchemaError::UnknownExtension { .. })
        ));
        assert!(table.validate(&[], Strictness::Preserving).is_ok());
        assert!(table.validate(&[key], Strictness::Activating).is_ok());
    }

    #[test]
    fn extension_entries_are_sorted_for_stable_reserialization() {
        let mut table = ExtensionTable::new();
        table.insert(
            ExtensionKey::parse("z:b").unwrap(),
            ExtensionValue::Bool(true),
        );
        table.insert(
            ExtensionKey::parse("a:c").unwrap(),
            ExtensionValue::Integer(1),
        );
        table.insert(
            ExtensionKey::parse("a:a").unwrap(),
            ExtensionValue::Real(0.5),
        );
        let keys: Vec<String> = table.iter().map(|(k, _)| k.to_string()).collect();
        assert_eq!(keys, vec!["a:a", "a:c", "z:b"]);
        assert_eq!(table.len(), 3);
        assert!(!table.is_empty());
    }

    #[test]
    fn insert_replaces_and_returns_previous() {
        let mut table = ExtensionTable::new();
        let k = ExtensionKey::parse("ns:k").unwrap();
        assert!(table
            .insert(k.clone(), ExtensionValue::Integer(1))
            .is_none());
        let prev = table.insert(k.clone(), ExtensionValue::Integer(2));
        assert_eq!(prev, Some(ExtensionValue::Integer(1)));
        assert_eq!(table.get(&k), Some(&ExtensionValue::Integer(2)));
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn capability_tag_cannot_assert_itself() {
        let tag = CapabilityTag::new("torque_control").unwrap();
        assert!(tag.validate_against(true).is_ok());
        assert!(tag.validate_against(false).is_err());
    }

    #[test]
    fn migration_outcome_reports_loss() {
        let lossy: MigrationOutcome<u8> = MigrationOutcome {
            record: 1,
            lost: vec!["backlash".to_owned()],
        };
        assert!(!lossy.is_lossless());
        assert!(MigrationOutcome::lossless(1u8).is_lossless());
    }
}

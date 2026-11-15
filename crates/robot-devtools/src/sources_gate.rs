//! The data-source half of the external-source registry (ADR 0021): pure
//! text checks over `data-source/1` records, sibling to
//! [`crate::services_gate`]'s `robot-service/1`.
//!
//! A data source is a public dataset someone downloads under a license —
//! files, not a service that runs — so its record pins **content**, not an
//! image: `content_digest` is the sha256 of the fetched-artifact manifest
//! once the source is audited and fetched, empty before. The same
//! evidence discipline applies: a digest is licensed by `code_inspected`
//! or better, never asserted below it, and registration never implies
//! permission to ingest — that is the per-corpus license audit of
//! SPECS.md C-07, recorded in the evidence ledger.
//!
//! v1 admits exactly one class, `dataset`, and exactly one scope,
//! `motion-only`: the project-owner decision of ADR 0021 that the initial
//! intake carries motion and pose content, never sensor channels, is
//! enforced by the gate rather than remembered.

use core::fmt;
use robot_artifacts::toml::{self, TomlError};

/// The record format this gate validates.
pub const FORMAT: &str = "data-source/1";

/// Every key a `data-source/1` record may contain.
pub const KNOWN_KEYS: &[&str] = &[
    "format",
    "name",
    "capability",
    "class",
    "scope",
    "evidence_state",
    "access",
    "output_format",
    "ingest_command",
    "license_disposition",
    "content_digest",
];

/// The keys that must be present and nonempty in every record.
const REQUIRED_NONEMPTY: &[&str] = &[
    "capability",
    "access",
    "output_format",
    "ingest_command",
    "license_disposition",
];

/// One reason the sources gate fails.
#[derive(Clone, PartialEq, Debug)]
pub enum Violation {
    /// The record did not parse, or a required key is missing or of the
    /// wrong type.
    Record {
        /// The directory name.
        source: String,
        /// The parse or lookup error, displayed.
        error: String,
    },
    /// A key outside [`KNOWN_KEYS`].
    UnknownKey {
        /// The directory name.
        source: String,
        /// The offending key.
        key: String,
    },
    /// The record's `name` does not match its directory, or is not
    /// lowercase-kebab.
    BadName {
        /// The directory name.
        source: String,
        /// The `name` value found.
        name: String,
    },
    /// A required key is empty.
    EmptyField {
        /// The directory name.
        source: String,
        /// The empty key.
        key: String,
    },
    /// The record's `class` is not `dataset`.
    InadmissibleClass {
        /// The directory name.
        source: String,
        /// The class found.
        class: String,
    },
    /// The record's `scope` is not `motion-only` (ADR 0021's initial
    /// intake scope; widening it is a new decision, not a record edit).
    InadmissibleScope {
        /// The directory name.
        source: String,
        /// The scope found.
        scope: String,
    },
    /// The record's `evidence_state` is not one of the five states.
    UnknownEvidenceState {
        /// The directory name.
        source: String,
        /// The state found.
        state: String,
    },
    /// A nonempty `content_digest` below `code_inspected`.
    DigestWithoutEvidence {
        /// The directory name.
        source: String,
        /// The record's evidence state.
        state: String,
    },
    /// A `content_digest` that is not `sha256:` plus 64 lowercase hex.
    MalformedDigest {
        /// The directory name.
        source: String,
        /// The digest found.
        digest: String,
    },
    /// A record at `code_inspected` or better with an empty digest.
    UnpinnedQualified {
        /// The directory name.
        source: String,
        /// The record's evidence state.
        state: String,
    },
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Record { source, error } => write!(
                f,
                "services/{source}/service.toml is not a valid data-source/1 record: {error} (ADR 0021, row A-51)"
            ),
            Self::UnknownKey { source, key } => write!(
                f,
                "services/{source}/service.toml has unknown key `{key}` (data-source/1 refuses unknown keys; ADR 0021, row A-51)"
            ),
            Self::BadName { source, name } => write!(
                f,
                "services/{source}/service.toml names itself \"{name}\"; the name must equal the directory and be lowercase letters, digits and hyphens (ADR 0021, row A-51)"
            ),
            Self::EmptyField { source, key } => write!(
                f,
                "services/{source}/service.toml leaves `{key}` empty (ADR 0021, row A-51)"
            ),
            Self::InadmissibleClass { source, class } => write!(
                f,
                "services/{source}/service.toml declares class \"{class}\"; data-source/1 admits only \"dataset\" (ADR 0021)"
            ),
            Self::InadmissibleScope { source, scope } => write!(
                f,
                "services/{source}/service.toml declares scope \"{scope}\"; the initial intake scope is \"motion-only\", and widening it is a new decision (ADR 0021)"
            ),
            Self::UnknownEvidenceState { source, state } => write!(
                f,
                "services/{source}/service.toml declares evidence_state \"{state}\", not one of the five SPECS.md Section 3 states (row A-51)"
            ),
            Self::DigestWithoutEvidence { source, state } => write!(
                f,
                "services/{source}/service.toml pins a content digest at evidence_state \"{state}\"; a digest requires code_inspected or better (SPECS.md Section 3)"
            ),
            Self::MalformedDigest { source, digest } => write!(
                f,
                "services/{source}/service.toml content_digest \"{digest}\" is not \"sha256:\" followed by 64 lowercase hex digits"
            ),
            Self::UnpinnedQualified { source, state } => write!(
                f,
                "services/{source}/service.toml claims evidence_state \"{state}\" without a pinned content_digest"
            ),
        }
    }
}

fn well_formed_digest(digest: &str) -> bool {
    digest.strip_prefix("sha256:").is_some_and(|hex| {
        hex.len() == 64
            && hex
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    })
}

fn kebab(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
}

/// Whether a record's text declares the `data-source/1` format, which is
/// how the gate binary routes a registry entry between the two record
/// kinds without parsing it twice.
#[must_use]
pub fn is_data_source(text: &str) -> bool {
    toml::parse(text)
        .ok()
        .and_then(|t| t.require_str("format").ok().map(str::to_owned))
        .is_some_and(|f| f == FORMAT)
}

/// Validates one `data-source/1` record. `source` is the directory name,
/// `text` the record contents. Returns every violation found, in check
/// order.
#[must_use]
pub fn check_record(source: &str, text: &str) -> Vec<Violation> {
    let record = |error: TomlError| Violation::Record {
        source: source.to_owned(),
        error: error.to_string(),
    };
    let table = match toml::parse(text) {
        Ok(table) => table,
        Err(e) => return vec![record(e)],
    };

    let mut found = Vec::new();
    for key in table.unknown_keys(KNOWN_KEYS) {
        found.push(Violation::UnknownKey {
            source: source.to_owned(),
            key,
        });
    }

    let mut field = |key: &str| -> Option<String> {
        match table.require_str(key) {
            Ok(v) => Some(v.to_owned()),
            Err(e) => {
                found.push(record(e));
                None
            }
        }
    };
    // `format` is the caller's routing key and already checked.
    let _ = field("format");
    let name = field("name");
    let class = field("class");
    let scope = field("scope");
    let state = field("evidence_state");
    let digest = field("content_digest");
    let mut required = Vec::new();
    for key in REQUIRED_NONEMPTY {
        required.push((*key, field(key)));
    }

    if let Some(name) = name {
        if name != source || !kebab(&name) {
            found.push(Violation::BadName {
                source: source.to_owned(),
                name,
            });
        }
    }
    for (key, value) in required {
        if value.is_some_and(|v| v.trim().is_empty()) {
            found.push(Violation::EmptyField {
                source: source.to_owned(),
                key: key.to_owned(),
            });
        }
    }
    if let Some(class) = class {
        if class != "dataset" {
            found.push(Violation::InadmissibleClass {
                source: source.to_owned(),
                class,
            });
        }
    }
    if let Some(scope) = scope {
        if scope != "motion-only" {
            found.push(Violation::InadmissibleScope {
                source: source.to_owned(),
                scope,
            });
        }
    }

    let Some(state) = state else {
        return found;
    };
    if !crate::services_gate::EVIDENCE_STATES.contains(&state.as_str()) {
        found.push(Violation::UnknownEvidenceState {
            source: source.to_owned(),
            state,
        });
        return found;
    }

    let digest = digest.unwrap_or_default();
    if crate::services_gate::state_licenses_digest(&state) {
        if digest.is_empty() {
            found.push(Violation::UnpinnedQualified {
                source: source.to_owned(),
                state: state.clone(),
            });
        }
    } else if !digest.is_empty() {
        found.push(Violation::DigestWithoutEvidence {
            source: source.to_owned(),
            state: state.clone(),
        });
    }
    if !digest.is_empty() && !well_formed_digest(&digest) {
        found.push(Violation::MalformedDigest {
            source: source.to_owned(),
            digest,
        });
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(state: &str, digest: &str) -> String {
        format!(
            "format = \"data-source/1\"\nname = \"cmu-mocap\"\ncapability = \"c\"\nclass = \"dataset\"\nscope = \"motion-only\"\nevidence_state = \"{state}\"\naccess = \"a\"\noutput_format = \"o\"\ningest_command = \"i\"\nlicense_disposition = \"l\"\ncontent_digest = \"{digest}\"\n"
        )
    }

    #[test]
    fn a_well_formed_unverified_record_passes() {
        assert!(check_record("cmu-mocap", &record("unverified", "")).is_empty());
    }

    #[test]
    fn routing_recognizes_the_format_line() {
        assert!(is_data_source(&record("unverified", "")));
        assert!(!is_data_source("format = \"robot-service/1\"\n"));
        assert!(!is_data_source("not toml ["));
    }

    #[test]
    fn the_scope_and_class_admit_exactly_one_value_each() {
        let bad_scope = record("unverified", "").replace("motion-only", "sensor-data");
        assert!(matches!(
            check_record("cmu-mocap", &bad_scope)[0],
            Violation::InadmissibleScope { .. }
        ));
        let bad_class = record("unverified", "").replace("\"dataset\"", "\"development-tool\"");
        assert!(matches!(
            check_record("cmu-mocap", &bad_class)[0],
            Violation::InadmissibleClass { .. }
        ));
    }

    #[test]
    fn a_digest_needs_evidence_and_evidence_needs_a_digest() {
        let digest = format!("sha256:{}", "a".repeat(64));
        assert!(matches!(
            check_record("cmu-mocap", &record("unverified", &digest))[0],
            Violation::DigestWithoutEvidence { .. }
        ));
        assert!(matches!(
            check_record("cmu-mocap", &record("code_inspected", ""))[0],
            Violation::UnpinnedQualified { .. }
        ));
        assert!(check_record("cmu-mocap", &record("code_inspected", &digest)).is_empty());
        assert!(matches!(
            check_record("cmu-mocap", &record("code_inspected", "sha256:nope"))[0],
            Violation::MalformedDigest { .. }
        ));
    }

    #[test]
    fn unknown_keys_names_and_empty_fields_are_refused() {
        let unknown = format!("{}extra = \"x\"\n", record("unverified", ""));
        assert!(matches!(
            check_record("cmu-mocap", &unknown)[0],
            Violation::UnknownKey { .. }
        ));
        assert!(matches!(
            check_record("elsewhere", &record("unverified", ""))[0],
            Violation::BadName { .. }
        ));
        let empty = record("unverified", "").replace("access = \"a\"", "access = \"\"");
        assert!(matches!(
            check_record("cmu-mocap", &empty)[0],
            Violation::EmptyField { .. }
        ));
    }
}

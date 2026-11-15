//! The services gate of SPECS.md Section 2.4 (rows A-51 and A-52), as
//! pure text checks over `services/*/service.toml` records.
//!
//! Section 2.4 item 2: "A Rust check in `robot-devtools` validates every
//! record and runs together with the Section 2.1 exclusion gate in the
//! default environment." The checks live here; the `exclusion-gate`
//! binary gathers the directory listing and record texts and exits
//! non-zero on any violation, exactly as it does for the Section 2.1
//! checks. Like the rest of this crate, nothing here ships: the gate
//! *reads* the registry, and no shipping crate reads this module.
//!
//! What `robot-service/1` enforces by type rather than review:
//!
//! 1. The record parses in the bounded TOML subset of
//!    `robot_artifacts::toml`, with unknown keys refused ([`KNOWN_KEYS`]).
//! 2. The only admissible class is `development-tool`; a runtime class
//!    does not exist in v1 (Section 2.4 item 3).
//! 3. Pinning is evidence (Section 2.4 item 5): `image_digest` must be
//!    empty below `code_inspected`, and at `code_inspected` or better it
//!    must be a well-formed `sha256:` digest with a nonempty
//!    `image_reference` and a `Containerfile` present in the directory.

use core::fmt;
use robot_artifacts::toml::{self, TomlError};

/// The record format this gate validates.
pub const FORMAT: &str = "robot-service/1";

/// Every key a `robot-service/1` record may contain. Anything else is a
/// [`Violation::UnknownKey`].
pub const KNOWN_KEYS: &[&str] = &[
    "format",
    "name",
    "capability",
    "class",
    "evidence_state",
    "output_format",
    "ingest_command",
    "license_disposition",
    "image_reference",
    "image_digest",
];

/// The five evidence states of SPECS.md Section 3, in promotion order.
pub const EVIDENCE_STATES: &[&str] = &[
    "unverified",
    "paper_verified",
    "code_inspected",
    "reproduced_in_sim",
    "validated_on_hardware",
];

/// The keys that must be present and nonempty in every record.
const REQUIRED_NONEMPTY: &[&str] = &[
    "capability",
    "output_format",
    "ingest_command",
    "license_disposition",
];

/// One reason the services gate fails.
#[derive(Clone, PartialEq, Debug)]
pub enum Violation {
    /// A service directory has no `service.toml`.
    MissingRecord {
        /// The directory name under `services/`.
        service: String,
    },
    /// The record did not parse, or a required key is missing or of the
    /// wrong type.
    Record {
        /// The directory name.
        service: String,
        /// The parse or lookup error, displayed.
        error: String,
    },
    /// The record's `format` is not `robot-service/1`.
    WrongFormat {
        /// The directory name.
        service: String,
        /// The `format` value found.
        found: String,
    },
    /// A key outside [`KNOWN_KEYS`].
    UnknownKey {
        /// The directory name.
        service: String,
        /// The offending key.
        key: String,
    },
    /// The record's `name` does not match its directory, or is not
    /// lowercase-kebab.
    BadName {
        /// The directory name.
        service: String,
        /// The `name` value found.
        name: String,
    },
    /// A required key is empty.
    EmptyField {
        /// The directory name.
        service: String,
        /// The empty key.
        key: String,
    },
    /// The record's `class` is not `development-tool`.
    InadmissibleClass {
        /// The directory name.
        service: String,
        /// The class found.
        class: String,
    },
    /// The record's `evidence_state` is not one of the five states.
    UnknownEvidenceState {
        /// The directory name.
        service: String,
        /// The state found.
        state: String,
    },
    /// A nonempty `image_digest` below `code_inspected`.
    DigestWithoutEvidence {
        /// The directory name.
        service: String,
        /// The record's evidence state.
        state: String,
    },
    /// An `image_digest` that is not `sha256:` plus 64 lowercase hex.
    MalformedDigest {
        /// The directory name.
        service: String,
        /// The digest found.
        digest: String,
    },
    /// A record at `code_inspected` or better with an empty digest or
    /// empty `image_reference`.
    UnpinnedQualified {
        /// The directory name.
        service: String,
        /// The record's evidence state.
        state: String,
    },
    /// A record at `code_inspected` or better without a `Containerfile`.
    MissingContainerfile {
        /// The directory name.
        service: String,
    },
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingRecord { service } => write!(
                f,
                "services/{service}/ has no service.toml (SPECS.md Section 2.4 item 1, row A-51)"
            ),
            Self::Record { service, error } => write!(
                f,
                "services/{service}/service.toml is not a valid robot-service/1 record: {error} (SPECS.md Section 2.4 item 1, row A-51)"
            ),
            Self::WrongFormat { service, found } => write!(
                f,
                "services/{service}/service.toml declares format \"{found}\", not \"{FORMAT}\" (SPECS.md Section 2.4 item 1, row A-51)"
            ),
            Self::UnknownKey { service, key } => write!(
                f,
                "services/{service}/service.toml has unknown key `{key}` (robot-service/1 refuses unknown keys; SPECS.md Section 2.4 item 1, row A-51)"
            ),
            Self::BadName { service, name } => write!(
                f,
                "services/{service}/service.toml names itself \"{name}\"; the name must equal the directory and be lowercase letters, digits and hyphens (SPECS.md Section 2.4 item 1, row A-51)"
            ),
            Self::EmptyField { service, key } => write!(
                f,
                "services/{service}/service.toml leaves `{key}` empty (SPECS.md Section 2.4 item 1, row A-51)"
            ),
            Self::InadmissibleClass { service, class } => write!(
                f,
                "services/{service}/service.toml declares class \"{class}\"; robot-service/1 admits only \"development-tool\" (SPECS.md Section 2.4 item 3, row A-51)"
            ),
            Self::UnknownEvidenceState { service, state } => write!(
                f,
                "services/{service}/service.toml declares evidence_state \"{state}\", not one of the five SPECS.md Section 3 states (row A-51)"
            ),
            Self::DigestWithoutEvidence { service, state } => write!(
                f,
                "services/{service}/service.toml pins an image digest at evidence_state \"{state}\"; a digest requires code_inspected or better (SPECS.md Section 2.4 item 5, row A-51)"
            ),
            Self::MalformedDigest { service, digest } => write!(
                f,
                "services/{service}/service.toml image_digest \"{digest}\" is not \"sha256:\" followed by 64 lowercase hex digits (SPECS.md Section 2.4 item 5, row A-51)"
            ),
            Self::UnpinnedQualified { service, state } => write!(
                f,
                "services/{service}/service.toml claims evidence_state \"{state}\" without a pinned image_digest and image_reference (SPECS.md Section 2.4 item 5, row A-51)"
            ),
            Self::MissingContainerfile { service } => write!(
                f,
                "services/{service}/ claims code_inspected or better without a Containerfile (SPECS.md Section 2.4 item 5, row A-51)"
            ),
        }
    }
}

/// One registry entry as the binary gathers it: the directory name, the
/// `service.toml` text if the file exists, and whether a `Containerfile`
/// sits beside it.
pub struct ServiceEntry {
    /// The directory name under `services/`.
    pub name: String,
    /// The record text, or `None` when `service.toml` is missing.
    pub record: Option<String>,
    /// Whether `Containerfile` exists in the directory.
    pub has_containerfile: bool,
}

/// Whether an evidence state licenses a pinned digest (Section 2.4
/// item 5): `code_inspected` or better.
#[must_use]
pub fn state_licenses_digest(state: &str) -> bool {
    matches!(
        state,
        "code_inspected" | "reproduced_in_sim" | "validated_on_hardware"
    )
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

/// Validates one record. `service` is the directory name; `text` is the
/// `service.toml` contents; `has_containerfile` says whether the recipe
/// file exists beside it. Returns every violation found, in check order.
#[must_use]
pub fn check_record(service: &str, text: &str, has_containerfile: bool) -> Vec<Violation> {
    let record = |error: TomlError| Violation::Record {
        service: service.to_owned(),
        error: error.to_string(),
    };
    let table = match toml::parse(text) {
        Ok(table) => table,
        Err(e) => return vec![record(e)],
    };

    let mut found = Vec::new();
    for key in table.unknown_keys(KNOWN_KEYS) {
        found.push(Violation::UnknownKey {
            service: service.to_owned(),
            key,
        });
    }

    // Every known key is required; absence or a non-string is a record
    // violation by the key's name.
    let mut field = |key: &str| -> Option<String> {
        match table.require_str(key) {
            Ok(v) => Some(v.to_owned()),
            Err(e) => {
                found.push(record(e));
                None
            }
        }
    };
    let format = field("format");
    let name = field("name");
    let class = field("class");
    let state = field("evidence_state");
    let reference = field("image_reference");
    let digest = field("image_digest");
    let mut required = Vec::new();
    for key in REQUIRED_NONEMPTY {
        required.push((*key, field(key)));
    }

    if let Some(format) = format {
        if format != FORMAT {
            found.push(Violation::WrongFormat {
                service: service.to_owned(),
                found: format,
            });
        }
    }
    if let Some(name) = name {
        if name != service || !kebab(&name) {
            found.push(Violation::BadName {
                service: service.to_owned(),
                name,
            });
        }
    }
    for (key, value) in required {
        if value.is_some_and(|v| v.trim().is_empty()) {
            found.push(Violation::EmptyField {
                service: service.to_owned(),
                key: key.to_owned(),
            });
        }
    }
    if let Some(class) = class {
        if class != "development-tool" {
            found.push(Violation::InadmissibleClass {
                service: service.to_owned(),
                class,
            });
        }
    }

    let Some(state) = state else {
        return found;
    };
    if !EVIDENCE_STATES.contains(&state.as_str()) {
        found.push(Violation::UnknownEvidenceState {
            service: service.to_owned(),
            state,
        });
        return found;
    }

    let digest = digest.unwrap_or_default();
    let reference = reference.unwrap_or_default();
    if state_licenses_digest(&state) {
        if digest.is_empty() || reference.is_empty() {
            found.push(Violation::UnpinnedQualified {
                service: service.to_owned(),
                state: state.clone(),
            });
        }
        if !has_containerfile {
            found.push(Violation::MissingContainerfile {
                service: service.to_owned(),
            });
        }
    } else if !digest.is_empty() {
        found.push(Violation::DigestWithoutEvidence {
            service: service.to_owned(),
            state: state.clone(),
        });
    }
    if !digest.is_empty() && !well_formed_digest(&digest) {
        found.push(Violation::MalformedDigest {
            service: service.to_owned(),
            digest,
        });
    }
    found
}

/// Validates the whole registry listing. An empty listing (no
/// `services/` directory, or no service directories) passes: the class
/// is admissible, not mandatory.
#[must_use]
pub fn check_services(entries: &[ServiceEntry]) -> Vec<Violation> {
    let mut found = Vec::new();
    for entry in entries {
        match &entry.record {
            None => found.push(Violation::MissingRecord {
                service: entry.name.clone(),
            }),
            Some(text) => {
                found.extend(check_record(&entry.name, text, entry.has_containerfile));
            }
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unverified() -> String {
        "format = \"robot-service/1\"\n\
         name = \"video-to-smplx\"\n\
         capability = \"Estimate SMPL-X parameters from video\"\n\
         class = \"development-tool\"\n\
         evidence_state = \"unverified\"\n\
         output_format = \"amass-npz\"\n\
         ingest_command = \"robot data import\"\n\
         license_disposition = \"SMPL-X fetched under the user's registration\"\n\
         image_reference = \"\"\n\
         image_digest = \"\"\n"
            .to_owned()
    }

    fn with(record: &str, key: &str, value: &str) -> String {
        record
            .lines()
            .map(|line| {
                if line.starts_with(&format!("{key} = ")) {
                    format!("{key} = \"{value}\"\n")
                } else {
                    format!("{line}\n")
                }
            })
            .collect()
    }

    #[test]
    fn an_unverified_candidate_with_empty_digest_passes() {
        assert_eq!(check_record("video-to-smplx", &unverified(), false), vec![]);
    }

    #[test]
    fn a_pinned_code_inspected_record_with_containerfile_passes() {
        let record = with(
            &with(
                &with(&unverified(), "evidence_state", "code_inspected"),
                "image_digest",
                &format!("sha256:{}", "a".repeat(64)),
            ),
            "image_reference",
            "localhost/video-to-smplx:1",
        );
        assert_eq!(check_record("video-to-smplx", &record, true), vec![]);
    }

    #[test]
    fn a_digest_below_code_inspected_is_refused() {
        let record = with(
            &unverified(),
            "image_digest",
            &format!("sha256:{}", "a".repeat(64)),
        );
        let found = check_record("video-to-smplx", &record, false);
        assert_eq!(
            found,
            vec![Violation::DigestWithoutEvidence {
                service: "video-to-smplx".to_owned(),
                state: "unverified".to_owned(),
            }]
        );
    }

    #[test]
    fn code_inspected_without_pin_or_containerfile_is_refused_twice() {
        let record = with(&unverified(), "evidence_state", "code_inspected");
        let found = check_record("video-to-smplx", &record, false);
        assert_eq!(
            found,
            vec![
                Violation::UnpinnedQualified {
                    service: "video-to-smplx".to_owned(),
                    state: "code_inspected".to_owned(),
                },
                Violation::MissingContainerfile {
                    service: "video-to-smplx".to_owned(),
                },
            ]
        );
    }

    #[test]
    fn a_malformed_digest_is_refused_by_name() {
        let record = with(
            &with(
                &with(&unverified(), "evidence_state", "code_inspected"),
                "image_digest",
                "sha256:short",
            ),
            "image_reference",
            "localhost/x:1",
        );
        let found = check_record("video-to-smplx", &record, true);
        assert_eq!(
            found,
            vec![Violation::MalformedDigest {
                service: "video-to-smplx".to_owned(),
                digest: "sha256:short".to_owned(),
            }]
        );
    }

    #[test]
    fn a_runtime_class_does_not_exist_in_v1() {
        let record = with(&unverified(), "class", "runtime");
        let found = check_record("video-to-smplx", &record, false);
        assert_eq!(
            found,
            vec![Violation::InadmissibleClass {
                service: "video-to-smplx".to_owned(),
                class: "runtime".to_owned(),
            }]
        );
    }

    #[test]
    fn unknown_keys_and_wrong_format_and_bad_name_are_each_named() {
        let record = format!("{}control_path = \"none\"\n", unverified());
        let found = check_record("video-to-smplx", &record, false);
        assert_eq!(
            found,
            vec![Violation::UnknownKey {
                service: "video-to-smplx".to_owned(),
                key: "control_path".to_owned(),
            }]
        );

        let found = check_record(
            "video-to-smplx",
            &with(&unverified(), "format", "robot-service/2"),
            false,
        );
        assert_eq!(
            found,
            vec![Violation::WrongFormat {
                service: "video-to-smplx".to_owned(),
                found: "robot-service/2".to_owned(),
            }]
        );

        let found = check_record("elsewhere", &unverified(), false);
        assert_eq!(
            found,
            vec![Violation::BadName {
                service: "elsewhere".to_owned(),
                name: "video-to-smplx".to_owned(),
            }]
        );
    }

    #[test]
    fn a_missing_key_an_empty_required_field_and_a_bad_state_are_refused() {
        let record: String = unverified()
            .lines()
            .filter(|l| !l.starts_with("capability"))
            .map(|l| format!("{l}\n"))
            .collect();
        let found = check_record("video-to-smplx", &record, false);
        assert_eq!(found.len(), 1);
        assert!(
            matches!(&found[0], Violation::Record { error, .. } if error.contains("capability")),
            "{found:?}"
        );

        let found = check_record(
            "video-to-smplx",
            &with(&unverified(), "license_disposition", ""),
            false,
        );
        assert_eq!(
            found,
            vec![Violation::EmptyField {
                service: "video-to-smplx".to_owned(),
                key: "license_disposition".to_owned(),
            }]
        );

        let found = check_record(
            "video-to-smplx",
            &with(&unverified(), "evidence_state", "inspected"),
            false,
        );
        assert_eq!(
            found,
            vec![Violation::UnknownEvidenceState {
                service: "video-to-smplx".to_owned(),
                state: "inspected".to_owned(),
            }]
        );
    }

    #[test]
    fn a_directory_without_a_record_fails_and_an_empty_registry_passes() {
        assert_eq!(check_services(&[]), vec![]);
        let found = check_services(&[ServiceEntry {
            name: "mystery".to_owned(),
            record: None,
            has_containerfile: false,
        }]);
        assert_eq!(
            found,
            vec![Violation::MissingRecord {
                service: "mystery".to_owned(),
            }]
        );
    }

    #[test]
    fn violations_name_the_governing_clause() {
        let text = Violation::DigestWithoutEvidence {
            service: "x".to_owned(),
            state: "unverified".to_owned(),
        }
        .to_string();
        assert!(text.contains("Section 2.4 item 5"), "{text}");
        assert!(text.contains("A-51"), "{text}");
    }

    #[test]
    fn the_committed_registry_records_pass_this_gate() {
        // The two records committed under services/ are themselves
        // fixtures of this gate; a change to either that the gate would
        // refuse should fail here, not first in CI.
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../services");
        let mut entries = Vec::new();
        for entry in std::fs::read_dir(&root).expect("services/ exists") {
            let entry = entry.expect("readable services/ entry");
            if !entry.path().is_dir() {
                continue;
            }
            let record = std::fs::read_to_string(entry.path().join("service.toml")).ok();
            entries.push(ServiceEntry {
                name: entry.file_name().to_string_lossy().into_owned(),
                record,
                has_containerfile: entry.path().join("Containerfile").is_file(),
            });
        }
        assert!(!entries.is_empty(), "the registry has committed records");
        assert_eq!(check_services(&entries), vec![]);
    }
}

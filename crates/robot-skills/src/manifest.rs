//! The `skill-manifest/1` file format: a skill library's typed descriptors
//! as one TOML document.
//!
//! SPECS.md Section 10.2 defines the descriptor's fields;
//! [`crate::descriptor::SkillDescriptor`] is their in-memory form, and this
//! module is their on-disk form, so a skill library can be reviewed, diffed
//! and loaded by name (`robot skills`, ADR 0016). The TOML dialect is the
//! bounded subset of `robot_artifacts::toml`, with its limits applied —
//! a manifest read back from disk is untrusted input like any other.
//!
//! The format is versioned by its `format = "skill-manifest/1"` key and
//! refuses unknown top-level or per-skill keys by name, so a file written by
//! a later revision fails loudly here instead of losing fields silently.
//! Reading and writing round-trip exactly.

use crate::descriptor::{EmbeddingVersion, SkillDescriptor, SkillId};
use robot_artifacts::store::{BlobStore, StoreError};
use robot_artifacts::toml::{self, Table, TomlError, Value};
use robot_artifacts::ContentId;
use robot_types::ids::Name;
use robot_types::time::Duration;

/// The format identity this module writes and the only one it reads.
pub const FORMAT: &str = "skill-manifest/1";

/// Why a manifest could not be read.
#[derive(Clone, PartialEq, Debug)]
pub enum ManifestError {
    /// The TOML itself was rejected.
    Toml(TomlError),
    /// The `format` key is missing or names another format.
    Format(String),
    /// A key this format does not define.
    UnknownKey(String),
    /// A field failed validation.
    Invalid {
        /// The skill's name or index.
        skill: String,
        /// What was wrong.
        what: String,
    },
    /// Two skills share an id or a name.
    Duplicate(String),
}

impl core::fmt::Display for ManifestError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Toml(e) => write!(f, "{e}"),
            Self::Format(found) => {
                write!(f, "not a {FORMAT} file (format = {found:?})")
            }
            Self::UnknownKey(k) => write!(f, "unknown key {k} (this reader is {FORMAT})"),
            Self::Invalid { skill, what } => write!(f, "skill {skill}: {what}"),
            Self::Duplicate(what) => write!(f, "duplicate {what}"),
        }
    }
}

impl core::error::Error for ManifestError {}

impl From<TomlError> for ManifestError {
    fn from(e: TomlError) -> Self {
        Self::Toml(e)
    }
}

fn names(items: &[Name]) -> Value {
    Value::Array(
        items
            .iter()
            .map(|n| Value::String(n.as_str().to_owned()))
            .collect(),
    )
}

/// Writes descriptors as a `skill-manifest/1` document.
///
/// # Errors
///
/// Refuses what [`read_manifest`] would refuse to read back — duplicate
/// ids or names, and non-finite embedding values — so a written manifest
/// is a readable one.
pub fn write_manifest(skills: &[SkillDescriptor]) -> Result<String, ManifestError> {
    for (i, d) in skills.iter().enumerate() {
        if skills[..i].iter().any(|e| e.id == d.id) {
            return Err(ManifestError::Duplicate(format!("skill id {}", d.id.0)));
        }
        if skills[..i].iter().any(|e| e.name == d.name) {
            return Err(ManifestError::Duplicate(format!(
                "skill name {:?}",
                d.name.as_str()
            )));
        }
        if d.embedding.iter().any(|v| !v.is_finite()) {
            return Err(ManifestError::Invalid {
                skill: d.name.as_str().to_owned(),
                what: "embedding holds a non-finite value".to_owned(),
            });
        }
    }
    let mut root = Table::new();
    root.insert("format", Value::String(FORMAT.to_owned()));
    let mut rows = Vec::new();
    for d in skills {
        let mut t = Table::new();
        t.insert("id", Value::Integer(i64::from(d.id.0)));
        t.insert("name", Value::String(d.name.as_str().to_owned()));
        t.insert("tags", names(&d.tags));
        t.insert(
            "embedding",
            Value::Array(d.embedding.iter().map(|v| Value::Float(*v)).collect()),
        );
        t.insert(
            "embedding_version",
            Value::Integer(i64::from(d.embedding_version.0)),
        );
        t.insert(
            "embodiment_compatibility",
            Value::Array(
                d.embodiment_compatibility
                    .iter()
                    .map(|s| Value::String(s.clone()))
                    .collect(),
            ),
        );
        t.insert("initiation_guards", names(&d.initiation_guards));
        t.insert(
            "termination_condition",
            Value::String(d.termination_condition.as_str().to_owned()),
        );
        t.insert(
            "expected_duration_nanos",
            Value::Integer(d.expected_duration.nanos()),
        );
        t.insert(
            "assumes_ground_contact",
            Value::Boolean(d.assumes_ground_contact),
        );
        t.insert(
            "supported_disturbances",
            Value::Array(
                d.supported_disturbances
                    .iter()
                    .map(|v| Value::Float(*v))
                    .collect(),
            ),
        );
        t.insert(
            "inference_memory_bytes",
            Value::Integer(d.inference_memory_bytes as i64),
        );
        t.insert(
            "outgoing_transitions",
            Value::Array(
                d.outgoing_transitions
                    .iter()
                    .map(|s| Value::Integer(i64::from(s.0)))
                    .collect(),
            ),
        );
        rows.push(Value::Table(t));
    }
    root.insert("skill", Value::Array(rows));
    Ok(toml::to_string(&root))
}

const SKILL_KEYS: &[&str] = &[
    "id",
    "name",
    "tags",
    "embedding",
    "embedding_version",
    "embodiment_compatibility",
    "initiation_guards",
    "termination_condition",
    "expected_duration_nanos",
    "assumes_ground_contact",
    "supported_disturbances",
    "inference_memory_bytes",
    "outgoing_transitions",
];

fn name_of(t: &Table, key: &str, skill: &str) -> Result<Name, ManifestError> {
    Name::new(t.require_str(key)?).map_err(|e| ManifestError::Invalid {
        skill: skill.to_owned(),
        what: format!("{key}: {e:?}"),
    })
}

fn name_list(t: &Table, key: &str, skill: &str) -> Result<Vec<Name>, ManifestError> {
    t.require_array(key)?
        .iter()
        .map(|v| {
            v.as_str()
                .ok_or_else(|| ManifestError::Invalid {
                    skill: skill.to_owned(),
                    what: format!("{key} holds a non-string"),
                })
                .and_then(|s| {
                    Name::new(s).map_err(|e| ManifestError::Invalid {
                        skill: skill.to_owned(),
                        what: format!("{key}: {e:?}"),
                    })
                })
        })
        .collect()
}

fn float_list(t: &Table, key: &str, skill: &str) -> Result<Vec<f64>, ManifestError> {
    t.require_array(key)?
        .iter()
        .map(|v| {
            v.as_float().ok_or_else(|| ManifestError::Invalid {
                skill: skill.to_owned(),
                what: format!("{key} holds a non-number"),
            })
        })
        .collect()
}

fn id_of(v: i64, skill: &str, key: &str) -> Result<u32, ManifestError> {
    u32::try_from(v).map_err(|_| ManifestError::Invalid {
        skill: skill.to_owned(),
        what: format!("{key} {v} is out of range"),
    })
}

/// Reads a `skill-manifest/1` document.
pub fn read_manifest(text: &str) -> Result<Vec<SkillDescriptor>, ManifestError> {
    let root = toml::parse(text)?;
    if let Some(k) = root.unknown_keys(&["format", "skill"]).into_iter().next() {
        return Err(ManifestError::UnknownKey(k));
    }
    let format = root.require_str("format")?;
    if format != FORMAT {
        return Err(ManifestError::Format(format.to_owned()));
    }
    let rows = match root.get("skill") {
        None => &[][..],
        Some(v) => v.as_array().ok_or_else(|| ManifestError::Invalid {
            skill: "(manifest)".to_owned(),
            what: "skill is not an array of tables".to_owned(),
        })?,
    };
    let mut out = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        let Value::Table(t) = row else {
            return Err(ManifestError::Invalid {
                skill: format!("#{i}"),
                what: "not a table".to_owned(),
            });
        };
        let label = t
            .get("name")
            .and_then(Value::as_str)
            .map_or_else(|| format!("#{i}"), str::to_owned);
        if let Some(k) = t.unknown_keys(SKILL_KEYS).into_iter().next() {
            return Err(ManifestError::UnknownKey(format!("skill {label}: {k}")));
        }
        let duration = t.require_integer("expected_duration_nanos")?;
        if duration < 0 {
            return Err(ManifestError::Invalid {
                skill: label,
                what: "negative expected_duration_nanos".to_owned(),
            });
        }
        let memory = t.require_integer("inference_memory_bytes")?;
        if memory < 0 {
            return Err(ManifestError::Invalid {
                skill: label,
                what: "negative inference_memory_bytes".to_owned(),
            });
        }
        let embedding = float_list(t, "embedding", &label)?;
        if embedding.iter().any(|v| !v.is_finite()) {
            return Err(ManifestError::Invalid {
                skill: label,
                what: "non-finite embedding component".to_owned(),
            });
        }
        out.push(SkillDescriptor {
            id: SkillId(id_of(t.require_integer("id")?, &label, "id")?),
            name: name_of(t, "name", &label)?,
            tags: name_list(t, "tags", &label)?,
            embedding,
            embedding_version: EmbeddingVersion(id_of(
                t.require_integer("embedding_version")?,
                &label,
                "embedding_version",
            )?),
            embodiment_compatibility: t
                .require_array("embodiment_compatibility")?
                .iter()
                .map(|v| {
                    v.as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| ManifestError::Invalid {
                            skill: label.clone(),
                            what: "embodiment_compatibility holds a non-string".to_owned(),
                        })
                })
                .collect::<Result<_, _>>()?,
            initiation_guards: name_list(t, "initiation_guards", &label)?,
            termination_condition: name_of(t, "termination_condition", &label)?,
            expected_duration: Duration::from_nanos(duration),
            assumes_ground_contact: t.require_bool("assumes_ground_contact")?,
            supported_disturbances: float_list(t, "supported_disturbances", &label)?,
            inference_memory_bytes: memory as u64,
            outgoing_transitions: t
                .require_array("outgoing_transitions")?
                .iter()
                .map(|v| match v {
                    Value::Integer(n) => Ok(SkillId(id_of(*n, &label, "outgoing_transitions")?)),
                    _ => Err(ManifestError::Invalid {
                        skill: label.clone(),
                        what: "outgoing_transitions holds a non-integer".to_owned(),
                    }),
                })
                .collect::<Result<_, _>>()?,
        });
    }
    for (i, a) in out.iter().enumerate() {
        for b in &out[i + 1..] {
            if a.id == b.id {
                return Err(ManifestError::Duplicate(format!("skill id {}", a.id.0)));
            }
            if a.name.as_str() == b.name.as_str() {
                return Err(ManifestError::Duplicate(format!(
                    "skill name {}",
                    a.name.as_str()
                )));
            }
        }
    }
    Ok(out)
}

/// Why a manifest could not be loaded from a content-addressed blob.
///
/// Distinct from [`ManifestError`] because the failure modes differ in
/// kind: the store layer reports a missing or corrupt blob, this layer
/// reports bytes that are not text, and only then does the manifest
/// parser get a say.
#[derive(Debug)]
pub enum BlobManifestError {
    /// The store could not produce the blob: missing, unreadable, or
    /// failing its checksum (`robot_artifacts::store::BlobStore::get`
    /// verifies on read).
    Store(StoreError),
    /// The blob exists and verifies but is not UTF-8 text, so it cannot be
    /// a `skill-manifest/1` document.
    NotText,
    /// The text is not a valid manifest.
    Manifest(ManifestError),
}

impl core::fmt::Display for BlobManifestError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Store(e) => write!(f, "{e}"),
            Self::NotText => write!(f, "blob is not UTF-8 text, so it is not a {FORMAT} file"),
            Self::Manifest(e) => write!(f, "{e}"),
        }
    }
}

impl core::error::Error for BlobManifestError {}

/// Loads a `skill-manifest/1` document from a content-addressed blob.
///
/// This is the loading path the `robot-artifacts` dependency exists for
/// (ADR 0016): a skill library stored by content identity (SPECS.md
/// Section 7), fetched and checksum-verified by the store, then parsed by
/// [`read_manifest`] under the same bounded-TOML limits as a file read
/// from disk — a blob read back from the store is untrusted input like
/// any other.
///
/// # Errors
///
/// [`BlobManifestError::Store`] when the store cannot produce a verified
/// blob, [`BlobManifestError::NotText`] for non-UTF-8 bytes, and
/// [`BlobManifestError::Manifest`] for anything [`read_manifest`] refuses.
pub fn load_from_blob(
    store: &BlobStore,
    id: &ContentId,
) -> Result<Vec<SkillDescriptor>, BlobManifestError> {
    let bytes = store.get(id).map_err(BlobManifestError::Store)?;
    let text = core::str::from_utf8(&bytes).map_err(|_| BlobManifestError::NotText)?;
    read_manifest(text).map_err(BlobManifestError::Manifest)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn library() -> Vec<SkillDescriptor> {
        let skill = |id: u32, name: &str, next: &[u32]| SkillDescriptor {
            id: SkillId(id),
            name: Name::new(name).unwrap(),
            tags: vec![Name::new("strike").unwrap()],
            embedding: vec![0.25, -1.5],
            embedding_version: EmbeddingVersion(1),
            embodiment_compatibility: vec!["fixture-humanoid/1.0".to_owned()],
            initiation_guards: vec![Name::new("double-support").unwrap()],
            termination_condition: Name::new("strike-complete").unwrap(),
            expected_duration: Duration::from_millis(400),
            assumes_ground_contact: true,
            supported_disturbances: vec![5.0],
            inference_memory_bytes: 4096,
            outgoing_transitions: next.iter().map(|n| SkillId(*n)).collect(),
        };
        vec![skill(1, "jab", &[2]), skill(2, "guard", &[1])]
    }

    #[test]
    fn a_manifest_round_trips_exactly() {
        let lib = library();
        let text = write_manifest(&lib).unwrap();
        assert!(text.starts_with("format = \"skill-manifest/1\""));
        let back = read_manifest(&text).unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].id, lib[0].id);
        assert_eq!(back[0].name.as_str(), "jab");
        assert_eq!(back[0].embedding, lib[0].embedding);
        assert_eq!(back[1].outgoing_transitions, vec![SkillId(1)]);
        assert_eq!(write_manifest(&back).unwrap(), text);
    }

    #[test]
    fn wrong_format_unknown_keys_and_duplicates_are_refused() {
        assert!(matches!(
            read_manifest("format = \"other/9\"\n"),
            Err(ManifestError::Format(_))
        ));
        let mut text = write_manifest(&library()).unwrap();
        text.push_str("\nextra = 1\n");
        assert!(matches!(
            read_manifest(&text),
            Err(ManifestError::UnknownKey(_))
        ));
        let mut dup = library();
        dup[1].id = dup[0].id;
        assert!(
            matches!(write_manifest(&dup), Err(ManifestError::Duplicate(_))),
            "the writer refuses what the reader would refuse to read back"
        );
        let mut nan = library();
        nan[0].embedding[0] = f64::NAN;
        assert!(matches!(
            write_manifest(&nan),
            Err(ManifestError::Invalid { .. })
        ));
    }

    #[test]
    fn invalid_fields_are_named() {
        let text = write_manifest(&library()).unwrap().replace(
            "expected_duration_nanos = 400000000",
            "expected_duration_nanos = -1",
        );
        match read_manifest(&text) {
            Err(ManifestError::Invalid { what, .. }) => {
                assert!(what.contains("negative"), "{what}");
            }
            other => panic!("{other:?}"),
        }
    }

    /// A temporary directory that removes itself. Enough for these tests;
    /// the same pattern `robot-artifacts`' own store tests use.
    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let unique = ContentId::of(
                format!(
                    "{tag}-{:?}-{:?}",
                    std::time::SystemTime::now(),
                    std::thread::current().id()
                )
                .as_bytes(),
            )
            .short();
            let path = std::env::temp_dir().join(format!("robot-skills-{tag}-{unique}"));
            std::fs::create_dir_all(&path).expect("create temp dir");
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn a_manifest_round_trips_through_a_content_addressed_blob() {
        use robot_artifacts::store::StoreQuota;
        let dir = TempDir::new("blob-roundtrip");
        let store = BlobStore::open(&dir.0, StoreQuota::default()).unwrap();
        let lib = library();
        let text = write_manifest(&lib).unwrap();
        let id = store.put(text.as_bytes()).unwrap();
        let back = load_from_blob(&store, &id).unwrap();
        assert_eq!(back.len(), lib.len());
        assert_eq!(back[0].name.as_str(), "jab");
        assert_eq!(
            write_manifest(&back).unwrap(),
            text,
            "the blob path loads exactly what was stored"
        );
    }

    #[test]
    fn a_missing_blob_and_an_invalid_blob_are_refused_with_the_layer_that_saw_it() {
        use robot_artifacts::store::StoreQuota;
        let dir = TempDir::new("blob-refusals");
        let store = BlobStore::open(&dir.0, StoreQuota::default()).unwrap();
        // Missing: an identity nothing was stored under.
        let absent = ContentId::of(b"never stored");
        assert!(matches!(
            load_from_blob(&store, &absent),
            Err(BlobManifestError::Store(StoreError::NotFound { .. }))
        ));
        // Stored but not text.
        let binary = store.put(&[0xff, 0xfe, 0x00, 0x80]).unwrap();
        assert!(matches!(
            load_from_blob(&store, &binary),
            Err(BlobManifestError::NotText)
        ));
        // Text but not a manifest.
        let wrong = store.put(b"format = \"other/9\"\n").unwrap();
        assert!(matches!(
            load_from_blob(&store, &wrong),
            Err(BlobManifestError::Manifest(ManifestError::Format(_)))
        ));
    }
}

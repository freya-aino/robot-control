//! A committed frame dump published to, and resolved from, `robot-artifacts`
//! content-addressed storage.
//!
//! ADR 0009 fixes the on-disk dump (`frames.csv`, `frames.jsonl`, per-step
//! SVGs, the commit manifest), and [`crate::pose::load_committed`] reads it
//! back from its own directory. What was missing — and what this crate's
//! README recorded as not implemented — is the storage SPECS.md §7 actually
//! standardizes on: content-addressed artifacts, where a run is named by
//! what it contains rather than where it sits. This module closes that gap
//! the way `robot-bundle/1` and `skill-manifest/1` close theirs: a small,
//! versioned manifest format owned by the consuming crate, over
//! `robot_artifacts::store::BlobStore`.
//!
//! A published run is three blobs: the committed `frames.csv` and
//! `frames.jsonl` **byte for byte** (so republishing an identical run
//! deduplicates to the same identities), and a `viewer-frames/1` manifest
//! naming both by content ID together with the committed step count. The
//! per-step SVGs are deliberately not stored: they are derived output,
//! reproducible from the frames through [`crate::svg::render`], and §7's
//! storage holds data, not renderings.
//!
//! Resolution is as strict as the directory path: the manifest is parsed
//! with unknown keys refused by name, both frame files are re-verified
//! against each other with the same line rules and bit-for-bit frame
//! comparison the commit used, and the step count must match the manifest.
//! The store itself re-hashes every blob on `get`, so corrupt content is
//! refused below this module before a single line is parsed.

use crate::csv;
use crate::jsonl;
use crate::pose::{same_frame, FrameStream, PoseError, PoseLog, CSV_FILE, JSONL_FILE};
use robot_artifacts::hash::ContentId;
use robot_artifacts::store::{BlobStore, StoreError};
use robot_artifacts::toml::{self, Table, TomlError, Value};
use std::fmt;
use std::fs;
use std::path::Path;

/// The manifest format this module writes and accepts.
pub const FORMAT: &str = "viewer-frames/1";

/// The manifest's complete key set; anything else is refused by name.
const KEYS: &[&str] = &["format", "steps", "frames_csv", "frames_jsonl"];

/// Why a run could not be published to, or resolved from, the store.
#[derive(Debug)]
pub enum ArtifactError {
    /// The blob store refused an operation (missing blob, corrupt content,
    /// quota, I/O).
    Store(StoreError),
    /// A committed frame file could not be read from the dump directory.
    Io {
        /// The file involved.
        path: std::path::PathBuf,
        /// The underlying error.
        source: std::io::Error,
    },
    /// The manifest blob is not UTF-8 text.
    NotText,
    /// The manifest is not a well-formed bounded-TOML document.
    Toml(TomlError),
    /// The manifest carries a key this format does not define.
    UnknownKey(String),
    /// The manifest's `format` key names something else.
    Format(String),
    /// A manifest field's value is invalid (a negative step count, a content
    /// ID that does not parse).
    Invalid {
        /// The key.
        key: &'static str,
        /// What was wrong.
        what: String,
    },
    /// The frames failed the strict read-back or disagreed between formats.
    Frames(PoseError),
    /// The verified frame count is not what the manifest declares.
    StepMismatch {
        /// Steps the manifest declares.
        declared: u64,
        /// Steps the verified frames hold.
        found: u64,
    },
}

impl fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store(e) => write!(f, "{e}"),
            Self::Io { path, source } => write!(f, "{}: {source}", path.display()),
            Self::NotText => f.write_str("the manifest blob is not UTF-8 text"),
            Self::Toml(e) => write!(f, "{e}"),
            Self::UnknownKey(k) => write!(f, "unknown manifest key {k:?}"),
            Self::Format(found) => {
                write!(
                    f,
                    "manifest format is {found:?}, this reader takes {FORMAT:?}"
                )
            }
            Self::Invalid { key, what } => write!(f, "{key}: {what}"),
            Self::Frames(e) => write!(f, "{e}"),
            Self::StepMismatch { declared, found } => write!(
                f,
                "the manifest declares {declared} steps but the frames hold {found}"
            ),
        }
    }
}

impl std::error::Error for ArtifactError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Store(e) => Some(e),
            Self::Io { source, .. } => Some(source),
            Self::Toml(e) => Some(e),
            Self::Frames(e) => Some(e),
            _ => None,
        }
    }
}

impl From<StoreError> for ArtifactError {
    fn from(e: StoreError) -> Self {
        Self::Store(e)
    }
}

impl From<TomlError> for ArtifactError {
    fn from(e: TomlError) -> Self {
        Self::Toml(e)
    }
}

impl From<PoseError> for ArtifactError {
    fn from(e: PoseError) -> Self {
        Self::Frames(e)
    }
}

/// Publishes a committed dump directory into `store` and returns the
/// identity of its `viewer-frames/1` manifest.
///
/// The directory passes the full [`crate::pose::load_committed`] gate first
/// — commit manifest present and consistent, both frame files agreeing bit
/// for bit, every per-step SVG present — so nothing partial or tampered is
/// ever published. The frame files are then stored exactly as committed, so
/// publishing the same run twice yields the same three content IDs.
///
/// # Errors
///
/// [`ArtifactError`]: the directory's own strict checks, file reads, or the
/// store's refusals.
pub fn publish_committed(dir: &Path, store: &BlobStore) -> Result<ContentId, ArtifactError> {
    let log = crate::pose::load_committed(dir)?;
    let read = |name: &str| -> Result<Vec<u8>, ArtifactError> {
        let path = dir.join(name);
        fs::read(&path).map_err(|source| ArtifactError::Io { path, source })
    };
    let csv_id = store.put(&read(CSV_FILE)?)?;
    let jsonl_id = store.put(&read(JSONL_FILE)?)?;
    let mut manifest = Table::new();
    manifest.insert("format", Value::String(FORMAT.to_owned()));
    // `MAX_FRAME_STEPS` is six digits, so the count always fits an `i64`.
    manifest.insert("steps", Value::Integer(log.frames.len() as i64));
    manifest.insert("frames_csv", Value::String(csv_id.to_string()));
    manifest.insert("frames_jsonl", Value::String(jsonl_id.to_string()));
    Ok(store.put(toml::to_string(&manifest).as_bytes())?)
}

/// Resolves a published run by its manifest identity and returns the same
/// [`PoseLog`] that [`crate::pose::load_committed`] returns for the
/// directory it was published from.
///
/// # Errors
///
/// [`ArtifactError`]: a missing or corrupt blob (the store re-hashes every
/// `get`), a malformed or wrong-format manifest, frames that fail the
/// strict readers or disagree between formats, or a step count that does
/// not match the manifest.
pub fn load_from_store(store: &BlobStore, manifest: &ContentId) -> Result<PoseLog, ArtifactError> {
    let text = String::from_utf8(store.get(manifest)?).map_err(|_| ArtifactError::NotText)?;
    let root = toml::parse(&text)?;
    if let Some(key) = root.unknown_keys(KEYS).into_iter().next() {
        return Err(ArtifactError::UnknownKey(key));
    }
    let format = root.require_str("format")?;
    if format != FORMAT {
        return Err(ArtifactError::Format(format.to_owned()));
    }
    let steps = root.require_integer("steps")?;
    let declared = u64::try_from(steps).map_err(|_| ArtifactError::Invalid {
        key: "steps",
        what: format!("negative step count {steps}"),
    })?;
    let id_of = |key: &'static str| -> Result<ContentId, ArtifactError> {
        ContentId::parse(root.require_str(key)?).map_err(|e| ArtifactError::Invalid {
            key,
            what: e.to_string(),
        })
    };
    let csv_bytes = store.get(&id_of("frames_csv")?)?;
    let jsonl_bytes = store.get(&id_of("frames_jsonl")?)?;

    // The same cross-format verification the commit ran, over the retrieved
    // bytes: strict line rules, bit-for-bit frame agreement.
    let mut csv_stream = FrameStream::<_, csv::Csv>::open(csv_bytes.as_slice(), None)?;
    let mut jsonl_stream = FrameStream::<_, jsonl::Jsonl>::open(jsonl_bytes.as_slice(), None)?;
    if csv_stream.joints() != jsonl_stream.joints() {
        return Err(PoseError::Malformed {
            line: 0,
            reason: "frames.csv and frames.jsonl disagree".to_owned(),
        }
        .into());
    }
    let joints = csv_stream.joints().to_vec();
    let mut frames = Vec::new();
    loop {
        match (csv_stream.next_frame()?, jsonl_stream.next_frame()?) {
            (None, None) => break,
            (Some(a), Some(b)) => {
                if !same_frame(&a, &b) {
                    return Err(PoseError::Malformed {
                        line: 0,
                        reason: format!("frames.csv and frames.jsonl disagree at step {}", a.step),
                    }
                    .into());
                }
                frames.push(a);
            }
            _ => {
                return Err(PoseError::Malformed {
                    line: 0,
                    reason: "frames.csv and frames.jsonl hold different numbers of steps"
                        .to_owned(),
                }
                .into())
            }
        }
    }
    if frames.len() as u64 != declared {
        return Err(ArtifactError::StepMismatch {
            declared,
            found: frames.len() as u64,
        });
    }
    Ok(PoseLog { joints, frames })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pose::{FrameDump, PoseFrame};
    use robot_artifacts::store::StoreQuota;
    use robot_types::time::{ClockId, Timestamp};

    struct TempDir(std::path::PathBuf);
    impl TempDir {
        fn new(label: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "robot-viewer-artifact-{label}-{}",
                std::process::id()
            ));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn committed_run(dir: &Path, steps: u64) {
        let joints = vec!["hip".to_owned(), "knee".to_owned()];
        let mut dump = FrameDump::create(dir, &joints).unwrap();
        for step in 1..=steps {
            dump.push(&PoseFrame {
                step,
                timestamp: Timestamp::sampling(step as i64 * 1_000, ClockId::Simulation),
                pelvis_position: [0.0, 0.1 * step as f64, 0.8],
                pelvis_quaternion: [1.0, 0.0, 0.0, 0.0],
                joint_angles: vec![0.05 * step as f64, -0.02 * step as f64],
            })
            .unwrap();
        }
        assert_eq!(dump.finish().unwrap(), steps);
    }

    fn store(dir: &Path) -> BlobStore {
        BlobStore::open(dir.join("store"), StoreQuota::default()).unwrap()
    }

    #[test]
    fn a_published_run_resolves_to_the_same_frames() {
        let tmp = TempDir::new("round-trip");
        let run = tmp.0.join("run");
        committed_run(&run, 3);
        let store = store(&tmp.0);
        let id = publish_committed(&run, &store).unwrap();
        let from_store = load_from_store(&store, &id).unwrap();
        let from_dir = crate::pose::load_committed(&run).unwrap();
        assert_eq!(from_store, from_dir);
        // Republishing the identical run deduplicates to the same identity.
        assert_eq!(publish_committed(&run, &store).unwrap(), id);
    }

    #[test]
    fn a_missing_manifest_is_refused_by_name() {
        let tmp = TempDir::new("missing");
        let store = store(&tmp.0);
        let absent = ContentId::of(b"nothing was ever stored under this");
        assert!(matches!(
            load_from_store(&store, &absent),
            Err(ArtifactError::Store(StoreError::NotFound { .. }))
        ));
    }

    #[test]
    fn an_uncommitted_run_is_never_published() {
        let tmp = TempDir::new("uncommitted");
        let run = tmp.0.join("run");
        let joints = vec!["hip".to_owned()];
        let dump = FrameDump::create(&run, &joints).unwrap();
        drop(dump); // abandoned, never committed
        let store = store(&tmp.0);
        assert!(matches!(
            publish_committed(&run, &store),
            Err(ArtifactError::Frames(PoseError::NotCommitted { .. }))
        ));
    }

    #[test]
    fn tampered_frame_content_is_refused_below_the_parser() {
        let tmp = TempDir::new("tamper");
        let run = tmp.0.join("run");
        committed_run(&run, 2);
        let store = store(&tmp.0);
        let id = publish_committed(&run, &store).unwrap();
        // Corrupt the stored CSV blob in place: the store's own re-hash on
        // `get` must refuse it before a line is parsed.
        let manifest = String::from_utf8(store.get(&id).unwrap()).unwrap();
        let root = toml::parse(&manifest).unwrap();
        let csv_id = ContentId::parse(root.require_str("frames_csv").unwrap()).unwrap();
        fs::write(store.path_of(&csv_id), b"not the committed bytes").unwrap();
        assert!(matches!(
            load_from_store(&store, &id),
            Err(ArtifactError::Store(StoreError::Corrupt { .. }))
        ));
    }

    #[test]
    fn wrong_formats_and_unknown_keys_are_refused() {
        let tmp = TempDir::new("format");
        let store = store(&tmp.0);
        let wrong = store
            .put(b"format = \"robot-bundle/1\"\nsteps = 0\nframes_csv = \"x\"\nframes_jsonl = \"y\"\n")
            .unwrap();
        assert!(matches!(
            load_from_store(&store, &wrong),
            Err(ArtifactError::Format(_))
        ));
        let unknown = store
            .put(b"format = \"viewer-frames/1\"\nextra = 1\n")
            .unwrap();
        assert!(matches!(
            load_from_store(&store, &unknown),
            Err(ArtifactError::UnknownKey(_))
        ));
        let binary = store.put(&[0xFF, 0xFE, 0x00]).unwrap();
        assert!(matches!(
            load_from_store(&store, &binary),
            Err(ArtifactError::NotText)
        ));
    }

    #[test]
    fn a_step_count_that_disagrees_with_the_frames_is_refused() {
        let tmp = TempDir::new("steps");
        let run = tmp.0.join("run");
        committed_run(&run, 2);
        let store = store(&tmp.0);
        let id = publish_committed(&run, &store).unwrap();
        let manifest = String::from_utf8(store.get(&id).unwrap()).unwrap();
        let lied = manifest.replace("steps = 2", "steps = 5");
        assert_ne!(manifest, lied, "the rewrite must change the manifest");
        let lying_id = store.put(lied.as_bytes()).unwrap();
        assert!(matches!(
            load_from_store(&store, &lying_id),
            Err(ArtifactError::StepMismatch {
                declared: 5,
                found: 2
            })
        ));
    }
}

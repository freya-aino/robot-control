//! Atomic writes and the content-addressed blob store.
//!
//! SPECS.md Section 13: "Provide restartable training, atomic checkpoints,
//! bounded storage quotas, artifact retention … Production bundles are
//! immutable." PLAN.md Phase 1 asks for "content hashing, manifest validation
//! and atomic artifact writes" and "experiment configuration and run directories
//! with immutable input manifests".
//!
//! Atomicity here means: write to a temporary file in the destination's own
//! directory, flush and sync it, then rename over the target. The rename is
//! atomic within a filesystem, and the temporary file is a sibling so the rename
//! never crosses a device boundary. A reader therefore sees either the previous
//! artifact or the new one, never a half-written one — which is what makes a
//! crash during a checkpoint recoverable rather than corrupting.

use crate::hash::{ContentId, HashError};
use core::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// Why a store operation failed.
#[derive(Debug)]
pub enum StoreError {
    /// An underlying filesystem error.
    Io {
        /// What was being attempted.
        operation: &'static str,
        /// The path involved.
        path: PathBuf,
        /// The error.
        source: io::Error,
    },
    /// A stored blob did not hash to its name.
    Corrupt {
        /// The identity it was stored under.
        expected: ContentId,
        /// The hash mismatch.
        source: HashError,
    },
    /// A blob was not present.
    NotFound {
        /// The identity requested.
        id: ContentId,
    },
    /// Writing the blob would exceed the store's quota.
    QuotaExceeded {
        /// Bytes already stored.
        used: u64,
        /// Bytes this write would add.
        incoming: u64,
        /// The quota.
        quota: u64,
    },
    /// A blob exceeded the per-object size limit.
    TooLarge {
        /// Size offered.
        size: u64,
        /// The limit.
        limit: u64,
    },
    /// An immutable artifact was overwritten.
    ///
    /// SPECS.md Section 13: "Production bundles are immutable." Rewriting one
    /// with different content is refused rather than silently accepted.
    ImmutableViolation {
        /// The path.
        path: PathBuf,
    },
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io {
                operation,
                path,
                source,
            } => {
                write!(f, "{operation} failed for {}: {source}", path.display())
            }
            Self::Corrupt { expected, source } => {
                write!(f, "blob {expected} is corrupt: {source}")
            }
            Self::NotFound { id } => write!(f, "blob {id} is not in the store"),
            Self::QuotaExceeded {
                used,
                incoming,
                quota,
            } => write!(
                f,
                "storing {incoming} bytes would exceed the quota: {used} of {quota} used"
            ),
            Self::TooLarge { size, limit } => {
                write!(f, "object is {size} bytes, limit is {limit}")
            }
            Self::ImmutableViolation { path } => {
                write!(
                    f,
                    "{} already exists with different content",
                    path.display()
                )
            }
        }
    }
}

impl core::error::Error for StoreError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Corrupt { source, .. } => Some(source),
            _ => None,
        }
    }
}

fn io_err<'a>(
    operation: &'static str,
    path: &'a Path,
) -> impl FnOnce(io::Error) -> StoreError + 'a {
    move |source| StoreError::Io {
        operation,
        path: path.to_path_buf(),
        source,
    }
}

/// Writes `bytes` to `path` atomically.
///
/// The temporary file is created beside the target, written, flushed, synced and
/// renamed. The containing directory is synced afterwards where the platform
/// supports it, so the rename itself is durable and not merely queued.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), StoreError> {
    let parent = path.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(parent).map_err(io_err("create directory", parent))?;

    // The temporary name includes the target's name so a crash leaves an
    // obviously-orphaned sibling rather than an anonymous file.
    let stem = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "artifact".to_owned());
    let unique = ContentId::of(bytes).short();
    let tmp = parent.join(format!(".{stem}.{unique}.tmp"));

    {
        let mut file = fs::File::create(&tmp).map_err(io_err("create temporary file", &tmp))?;
        file.write_all(bytes)
            .map_err(io_err("write temporary file", &tmp))?;
        file.flush().map_err(io_err("flush temporary file", &tmp))?;
        file.sync_all()
            .map_err(io_err("sync temporary file", &tmp))?;
    }

    if let Err(e) = fs::rename(&tmp, path) {
        let _ = fs::remove_file(&tmp);
        return Err(StoreError::Io {
            operation: "rename into place",
            path: path.to_path_buf(),
            source: e,
        });
    }

    // Best effort: a failure to sync the directory does not invalidate the
    // rename, so it is not escalated to an error.
    if let Ok(dir) = fs::File::open(parent) {
        let _ = dir.sync_all();
    }
    Ok(())
}

/// Writes `bytes` to `path` only if absent or already identical.
///
/// This is the immutability rule of SPECS.md Section 13: rewriting an artifact
/// with the same bytes is a no-op, and rewriting it with different bytes is an
/// error rather than an update.
pub fn write_immutable(path: &Path, bytes: &[u8]) -> Result<bool, StoreError> {
    if path.exists() {
        let existing = fs::read(path).map_err(io_err("read existing artifact", path))?;
        if existing == bytes {
            return Ok(false);
        }
        return Err(StoreError::ImmutableViolation {
            path: path.to_path_buf(),
        });
    }
    write_atomic(path, bytes)?;
    Ok(true)
}

/// Limits applied by a [`BlobStore`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct StoreQuota {
    /// Maximum total bytes across all blobs.
    pub total_bytes: u64,
    /// Maximum bytes in one blob.
    pub object_bytes: u64,
}

impl Default for StoreQuota {
    fn default() -> Self {
        Self {
            total_bytes: 64 * 1024 * 1024 * 1024,
            object_bytes: 4 * 1024 * 1024 * 1024,
        }
    }
}

/// A content-addressed blob store on the filesystem.
///
/// Blobs live at `<root>/blobs/<first two hex>/<remaining hex>`. The two-level
/// fan-out keeps directory sizes manageable on filesystems that degrade with
/// very wide directories.
#[derive(Clone, Debug)]
pub struct BlobStore {
    root: PathBuf,
    quota: StoreQuota,
}

impl BlobStore {
    /// Opens or creates a store at `root`.
    pub fn open(root: impl Into<PathBuf>, quota: StoreQuota) -> Result<Self, StoreError> {
        let root = root.into();
        let blobs = root.join("blobs");
        fs::create_dir_all(&blobs).map_err(io_err("create store", &blobs))?;
        Ok(Self { root, quota })
    }

    /// The store's root directory.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The path a blob occupies.
    #[must_use]
    pub fn path_of(&self, id: &ContentId) -> PathBuf {
        let hex = id.digest().to_hex();
        self.root.join("blobs").join(&hex[..2]).join(&hex[2..])
    }

    /// True when the blob is present.
    #[must_use]
    pub fn contains(&self, id: &ContentId) -> bool {
        self.path_of(id).exists()
    }

    /// Stores bytes and returns their identity.
    ///
    /// Storing the same bytes twice is a no-op, which is what makes the store
    /// safe to write to from a retried step.
    pub fn put(&self, bytes: &[u8]) -> Result<ContentId, StoreError> {
        let size = bytes.len() as u64;
        if size > self.quota.object_bytes {
            return Err(StoreError::TooLarge {
                size,
                limit: self.quota.object_bytes,
            });
        }
        let id = ContentId::of(bytes);
        let path = self.path_of(&id);
        if path.exists() {
            return Ok(id);
        }
        let used = self.used_bytes()?;
        if used + size > self.quota.total_bytes {
            return Err(StoreError::QuotaExceeded {
                used,
                incoming: size,
                quota: self.quota.total_bytes,
            });
        }
        write_atomic(&path, bytes)?;
        Ok(id)
    }

    /// Reads a blob, verifying it against its identity.
    ///
    /// Verification on read is what turns silent bit rot into a reported fault,
    /// and SPECS.md Section 13 asks for checksum verification of stored input.
    pub fn get(&self, id: &ContentId) -> Result<Vec<u8>, StoreError> {
        let path = self.path_of(id);
        if !path.exists() {
            return Err(StoreError::NotFound { id: id.clone() });
        }
        let bytes = fs::read(&path).map_err(io_err("read blob", &path))?;
        id.verify(&bytes).map_err(|source| StoreError::Corrupt {
            expected: id.clone(),
            source,
        })?;
        Ok(bytes)
    }

    /// Total bytes currently stored.
    pub fn used_bytes(&self) -> Result<u64, StoreError> {
        let blobs = self.root.join("blobs");
        let mut total = 0u64;
        let entries = match fs::read_dir(&blobs) {
            Ok(e) => e,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(0),
            Err(e) => {
                return Err(StoreError::Io {
                    operation: "list store",
                    path: blobs,
                    source: e,
                })
            }
        };
        for prefix in entries.flatten() {
            let Ok(inner) = fs::read_dir(prefix.path()) else {
                continue;
            };
            for blob in inner.flatten() {
                if let Ok(meta) = blob.metadata() {
                    total += meta.len();
                }
            }
        }
        Ok(total)
    }

    /// Every blob identity in the store, sorted.
    pub fn list(&self) -> Result<Vec<ContentId>, StoreError> {
        let blobs = self.root.join("blobs");
        let mut out = Vec::new();
        let entries = match fs::read_dir(&blobs) {
            Ok(e) => e,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(out),
            Err(e) => {
                return Err(StoreError::Io {
                    operation: "list store",
                    path: blobs,
                    source: e,
                })
            }
        };
        for prefix_entry in entries.flatten() {
            let prefix = prefix_entry.file_name().to_string_lossy().into_owned();
            let Ok(inner) = fs::read_dir(prefix_entry.path()) else {
                continue;
            };
            for blob in inner.flatten() {
                let rest = blob.file_name().to_string_lossy().into_owned();
                if let Ok(id) = ContentId::parse(&format!("sha256:{prefix}{rest}")) {
                    out.push(id);
                }
            }
        }
        out.sort();
        Ok(out)
    }
}

/// A run directory with an immutable input manifest.
///
/// PLAN.md Phase 1: "Create experiment configuration and run directories with
/// immutable input manifests." The manifest is written once, at creation, and
/// [`RunDirectory::write_output`] refuses to touch it afterwards — so a run's
/// record of what it consumed cannot be edited to match what it produced.
#[derive(Clone, Debug)]
pub struct RunDirectory {
    path: PathBuf,
    run_id: String,
}

impl RunDirectory {
    /// The file name of the immutable input manifest.
    pub const INPUT_MANIFEST: &'static str = "inputs.toml";

    /// Creates a run directory and writes its input manifest.
    ///
    /// Fails if the directory already has an input manifest with different
    /// content, which is what stops a re-run from quietly reusing a directory.
    pub fn create(
        parent: impl AsRef<Path>,
        run_id: impl Into<String>,
        input_manifest: &str,
    ) -> Result<Self, StoreError> {
        let run_id = run_id.into();
        let path = parent.as_ref().join(&run_id);
        fs::create_dir_all(&path).map_err(io_err("create run directory", &path))?;
        write_immutable(&path.join(Self::INPUT_MANIFEST), input_manifest.as_bytes())?;
        Ok(Self { path, run_id })
    }

    /// Opens an existing run directory.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, StoreError> {
        let path = path.into();
        let manifest = path.join(Self::INPUT_MANIFEST);
        if !manifest.exists() {
            return Err(StoreError::Io {
                operation: "open run directory",
                path: manifest,
                source: io::Error::new(io::ErrorKind::NotFound, "no input manifest"),
            });
        }
        let run_id = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        Ok(Self { path, run_id })
    }

    /// The directory's path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The run identifier.
    #[must_use]
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// The recorded input manifest.
    pub fn input_manifest(&self) -> Result<String, StoreError> {
        let path = self.path.join(Self::INPUT_MANIFEST);
        let bytes = fs::read(&path).map_err(io_err("read input manifest", &path))?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Writes an output file, refusing to overwrite the input manifest.
    pub fn write_output(&self, name: &str, bytes: &[u8]) -> Result<PathBuf, StoreError> {
        if name == Self::INPUT_MANIFEST {
            return Err(StoreError::ImmutableViolation {
                path: self.path.join(name),
            });
        }
        let path = self.path.join(name);
        write_atomic(&path, bytes)?;
        Ok(path)
    }

    /// Writes a checkpoint atomically under a conventional name.
    ///
    /// Checkpoints are versioned by step so that a resume can pick the latest
    /// complete one; because each write is atomic, a crash mid-write leaves the
    /// previous checkpoint intact (SPECS.md Section 13, "atomic checkpoints").
    pub fn write_checkpoint(&self, step: u64, bytes: &[u8]) -> Result<PathBuf, StoreError> {
        self.write_output(&format!("checkpoint-{step:012}.chunk"), bytes)
    }

    /// The highest checkpoint step present.
    pub fn latest_checkpoint(&self) -> Result<Option<(u64, PathBuf)>, StoreError> {
        let entries = fs::read_dir(&self.path).map_err(io_err("list run directory", &self.path))?;
        let mut best: Option<(u64, PathBuf)> = None;
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let Some(rest) = name.strip_prefix("checkpoint-") else {
                continue;
            };
            let Some(step_text) = rest.strip_suffix(".chunk") else {
                continue;
            };
            let Ok(step) = step_text.parse::<u64>() else {
                continue;
            };
            if best.as_ref().is_none_or(|(b, _)| step > *b) {
                best = Some((step, entry.path()));
            }
        }
        Ok(best)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A temporary directory that removes itself. Enough for tests; not a
    /// general-purpose utility.
    struct TempDir(PathBuf);

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
            let path = std::env::temp_dir().join(format!("robot-artifacts-{tag}-{unique}"));
            fs::create_dir_all(&path).expect("create temp dir");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn atomic_write_leaves_no_temporary_behind() {
        let dir = TempDir::new("atomic");
        let target = dir.path().join("nested").join("artifact.bin");
        write_atomic(&target, b"hello").unwrap();
        assert_eq!(fs::read(&target).unwrap(), b"hello");

        let siblings: Vec<String> = fs::read_dir(target.parent().unwrap())
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(siblings, vec!["artifact.bin".to_owned()]);
    }

    #[test]
    fn atomic_write_replaces_previous_content_wholesale() {
        let dir = TempDir::new("replace");
        let target = dir.path().join("a.bin");
        write_atomic(&target, b"first").unwrap();
        write_atomic(&target, b"second-and-longer").unwrap();
        assert_eq!(fs::read(&target).unwrap(), b"second-and-longer");
    }

    #[test]
    fn immutable_write_rejects_a_changed_artifact() {
        let dir = TempDir::new("immutable");
        let target = dir.path().join("bundle.toml");
        assert!(write_immutable(&target, b"v1").unwrap());
        // Rewriting identical bytes is a no-op, not an error.
        assert!(!write_immutable(&target, b"v1").unwrap());
        assert!(matches!(
            write_immutable(&target, b"v2"),
            Err(StoreError::ImmutableViolation { .. })
        ));
        assert_eq!(fs::read(&target).unwrap(), b"v1");
    }

    #[test]
    fn blobs_round_trip_and_deduplicate() {
        let dir = TempDir::new("blobs");
        let store = BlobStore::open(dir.path(), StoreQuota::default()).unwrap();
        let id = store.put(b"payload").unwrap();
        let again = store.put(b"payload").unwrap();
        assert_eq!(id, again);
        assert!(store.contains(&id));
        assert_eq!(store.get(&id).unwrap(), b"payload");
        assert_eq!(store.list().unwrap(), vec![id]);
    }

    #[test]
    fn a_missing_blob_is_reported() {
        let dir = TempDir::new("missing");
        let store = BlobStore::open(dir.path(), StoreQuota::default()).unwrap();
        let id = ContentId::of(b"never stored");
        assert!(matches!(store.get(&id), Err(StoreError::NotFound { .. })));
        assert!(!store.contains(&id));
    }

    #[test]
    fn a_tampered_blob_is_reported_on_read() {
        let dir = TempDir::new("tamper");
        let store = BlobStore::open(dir.path(), StoreQuota::default()).unwrap();
        let id = store.put(b"original").unwrap();
        fs::write(store.path_of(&id), b"tampered").unwrap();
        assert!(matches!(store.get(&id), Err(StoreError::Corrupt { .. })));
    }

    #[test]
    fn quotas_are_enforced() {
        let dir = TempDir::new("quota");
        let store = BlobStore::open(
            dir.path(),
            StoreQuota {
                total_bytes: 16,
                object_bytes: 8,
            },
        )
        .unwrap();
        assert!(matches!(
            store.put(&[0u8; 9]),
            Err(StoreError::TooLarge { size: 9, limit: 8 })
        ));
        store.put(&[1u8; 8]).unwrap();
        store.put(&[2u8; 8]).unwrap();
        assert!(matches!(
            store.put(&[3u8; 8]),
            Err(StoreError::QuotaExceeded { .. })
        ));
        assert_eq!(store.used_bytes().unwrap(), 16);
    }

    #[test]
    fn a_run_directory_pins_its_inputs() {
        let dir = TempDir::new("run");
        let run = RunDirectory::create(dir.path(), "run-001", "config = \"sha256:abc\"\n").unwrap();
        assert_eq!(run.run_id(), "run-001");
        assert_eq!(run.input_manifest().unwrap(), "config = \"sha256:abc\"\n");

        // An output may be written; the input manifest may not be rewritten.
        run.write_output("report.toml", b"result = 1\n").unwrap();
        assert!(matches!(
            run.write_output(RunDirectory::INPUT_MANIFEST, b"config = \"other\"\n"),
            Err(StoreError::ImmutableViolation { .. })
        ));

        // Re-creating with different inputs is refused.
        assert!(matches!(
            RunDirectory::create(dir.path(), "run-001", "config = \"sha256:def\"\n"),
            Err(StoreError::ImmutableViolation { .. })
        ));
        // Re-creating with identical inputs is idempotent.
        assert!(RunDirectory::create(dir.path(), "run-001", "config = \"sha256:abc\"\n").is_ok());
    }

    #[test]
    fn opening_a_directory_without_a_manifest_fails() {
        let dir = TempDir::new("nomanifest");
        fs::create_dir_all(dir.path().join("bare")).unwrap();
        assert!(RunDirectory::open(dir.path().join("bare")).is_err());
    }

    #[test]
    fn checkpoints_are_ordered_and_resumable() {
        let dir = TempDir::new("ckpt");
        let run = RunDirectory::create(dir.path(), "run-002", "seed = 1\n").unwrap();
        assert!(run.latest_checkpoint().unwrap().is_none());
        run.write_checkpoint(1, b"one").unwrap();
        run.write_checkpoint(1000, b"thousand").unwrap();
        run.write_checkpoint(20, b"twenty").unwrap();
        let (step, path) = run.latest_checkpoint().unwrap().unwrap();
        assert_eq!(step, 1000);
        assert_eq!(fs::read(path).unwrap(), b"thousand");
    }
}

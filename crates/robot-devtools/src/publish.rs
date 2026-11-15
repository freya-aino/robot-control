//! Publishing a knowledge-base snapshot as a content-addressed artifact —
//! the use this crate's `robot-artifacts` dependency was reserved for.
//!
//! SPECS.md Section 13.1 keeps the dependency knowledge base as a living,
//! hand-cited store (`docs/kb/knowledge-base.toml`), and Section 3's
//! evidence discipline wants any state another artifact relies on to be
//! referenceable by content identity rather than by a path that keeps
//! changing underneath it. Publishing a snapshot writes the store's exact
//! bytes into a [`BlobStore`], so a ledger row, an automation-log entry or
//! a review can cite `sha256:…` and mean one specific knowledge-base
//! state forever. The snapshot is data about development tooling, never
//! evidence: `inventory::ledger_cites_kb` still refuses a ledger row that
//! cites the base itself, and publishing does not change that.

use std::path::Path;

use robot_artifacts::hash::ContentId;
use robot_artifacts::store::{BlobStore, StoreError, StoreQuota};

/// Why a snapshot could not be published.
#[derive(Debug)]
pub enum PublishError {
    /// The knowledge-base file could not be read.
    Read {
        /// The path that failed.
        path: String,
        /// The I/O error text.
        error: String,
    },
    /// The blob store refused the open or the write.
    Store(StoreError),
}

impl core::fmt::Display for PublishError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Read { path, error } => {
                write!(f, "cannot read knowledge base {path}: {error}")
            }
            Self::Store(e) => write!(f, "{e}"),
        }
    }
}

impl core::error::Error for PublishError {}

impl From<StoreError> for PublishError {
    fn from(e: StoreError) -> Self {
        Self::Store(e)
    }
}

/// Reads the knowledge-base file at `kb_path` and writes its exact bytes
/// into the [`BlobStore`] rooted at `store_root`, returning the content
/// identity of the snapshot.
///
/// The bytes are stored verbatim — no parsing, normalisation or reflow —
/// so the returned [`ContentId`] is reproducible from the file alone and a
/// second publish of an unchanged base deduplicates to the same identity.
///
/// # Errors
///
/// [`PublishError::Read`] names the path when the file cannot be read;
/// [`PublishError::Store`] carries the store's own refusal.
pub fn publish_knowledge_base(
    store_root: &Path,
    kb_path: &Path,
) -> Result<ContentId, PublishError> {
    let bytes = std::fs::read(kb_path).map_err(|e| PublishError::Read {
        path: kb_path.display().to_string(),
        error: e.to_string(),
    })?;
    let store = BlobStore::open(store_root, StoreQuota::default())?;
    Ok(store.put(&bytes)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A temporary directory that removes itself. Enough for tests; not a
    /// general-purpose utility (the same pattern `robot-artifacts` uses).
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
            let path = std::env::temp_dir().join(format!("robot-devtools-{tag}-{unique}"));
            std::fs::create_dir_all(&path).expect("create temp dir");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn a_snapshot_round_trips_byte_identically_under_its_own_hash() {
        let dir = TempDir::new("publish");
        let kb = dir.path().join("knowledge-base.toml");
        let content = b"format = \"kb-store/1\"\n\n# no entries\n";
        std::fs::write(&kb, content).unwrap();
        let store_root = dir.path().join("store");

        let id = publish_knowledge_base(&store_root, &kb).unwrap();
        // The identity is the file's own content hash, nothing else.
        assert_eq!(id, ContentId::of(content));
        let store = BlobStore::open(&store_root, StoreQuota::default()).unwrap();
        assert_eq!(store.get(&id).unwrap(), content, "bytes must round-trip");

        // Publishing the unchanged base deduplicates to the same identity.
        assert_eq!(publish_knowledge_base(&store_root, &kb).unwrap(), id);
    }

    #[test]
    fn a_missing_knowledge_base_is_refused_by_name() {
        let dir = TempDir::new("publish-missing");
        let missing = dir.path().join("no-such-file.toml");
        let err = publish_knowledge_base(&dir.path().join("store"), &missing).unwrap_err();
        match err {
            PublishError::Read { path, .. } => {
                assert!(path.contains("no-such-file.toml"), "{path}");
            }
            other => panic!("expected a read error, got {other}"),
        }
    }

    #[test]
    fn the_committed_knowledge_base_publishes() {
        // The real store file, published into a scratch store: the command's
        // own happy path against the repository's canonical input.
        let kb = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/kb/knowledge-base.toml");
        let dir = TempDir::new("publish-real");
        let id = publish_knowledge_base(&dir.path().join("store"), &kb).unwrap();
        assert_eq!(id, ContentId::of(&std::fs::read(&kb).unwrap()));
    }
}

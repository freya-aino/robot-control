//! Content hashing, manifests, atomic writes, lineage and promotion.
//!
//! This crate owns *how an artifact is named, written and found again*. SPECS.md
//! Section 7.5 makes that concrete: "TOML manifests refer to content-addressed
//! blobs", high-rate numerical data goes in "chunked binary arrays", and every
//! wire or archive format needs "explicit versioning, size limits, framing,
//! endianness/alignment rules where applicable and validation before access".
//!
//! The four modules map one-to-one onto those obligations:
//!
//! | Module | Obligation |
//! | --- | --- |
//! | [`hash`] | Content identity: SHA-256 and `sha256:<hex>` names |
//! | [`toml`] | The bounded TOML subset manifests are written in |
//! | [`chunk`] | The versioned, framed, digest-checked binary chunk format |
//! | [`store`] | Atomic writes, the blob store, run directories, checkpoints |
//! | [`manifest`] | Artifact manifests, lineage, and promotion with rollback |
//!
//! Every reader in this crate validates before it interprets. SPECS.md Section 13
//! requires treating downloaded datasets and model files as untrusted input, and
//! an archive read from a shared drive is the same class of input: the digest is
//! checked before a payload byte is decoded, limits are applied before an
//! allocation is sized, and an unsupported construct is named rather than
//! skipped.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod chunk;
pub mod hash;
pub mod manifest;
pub mod store;
pub mod toml;

pub use hash::{ContentId, Digest, HashError, Sha256};
pub use manifest::{ArtifactKind, ArtifactManifest, Environment, Lineage, PromotionLedger};
pub use store::{write_atomic, write_immutable, BlobStore, RunDirectory, StoreError};

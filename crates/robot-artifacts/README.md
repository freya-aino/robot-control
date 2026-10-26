# robot-artifacts

Content hashing, manifests, atomic writes, lineage and promotion — how an
artifact is named, written, found again, and rolled back from. Depends only
on `robot-types`.

## What's here

| Module | SPECS.md anchor | Contents |
| --- | --- | --- |
| `hash` | §7.5, §13 | Authored SHA-256 (FIPS 180-4 test vectors), `ContentId` (`sha256:<hex>`), `digest_parts` for order-sensitive manifest hashing |
| `toml` | §2, §7.5, §13 | A bounded, documented TOML **subset** parser/writer; unsupported constructs (dates, inline tables, literal strings, …) are named, not silently skipped |
| `chunk` | §7.5 | Versioned, framed, digest-checked binary chunks for high-rate numerical data: SoA/AoS layout, four dtypes, little-endian on the wire, digest validated before a payload byte is interpreted |
| `store` | §13 | `write_atomic`/`write_immutable` (temp file + fsync + rename), the content-addressed `BlobStore`, `RunDirectory` with an immutable input manifest and ordered checkpoints |
| `manifest` | §13, §12.2 | `ArtifactManifest` (TOML-encoded, round-trips exactly), `Lineage` with transitive held-out-exclusion testing, `PromotionLedger` with last-known-good rollback |

## Why a hand-rolled TOML parser

SPECS.md Section 7.5 makes TOML the manifest format; Section 13 requires
treating all input — including a manifest read back from disk — as untrusted:
"enforce size limits, reject malformed dimensions, fuzz parsers." Appendix B.6
admits no `toml` crate dependency until it clears qualification (and nothing
in Appendix B currently names one as a shortlist candidate). Rather than block
on that, this module implements and documents the exact subset the platform's
manifests use, with every excluded TOML construct named by
`TomlError::Unsupported` rather than mis-parsed or ignored.

## The atomicity guarantee

`write_atomic` writes to a sibling temporary file, flushes, `fsync`s it,
renames it over the target (atomic within one filesystem, since the temp file
is never on a different device), and best-effort `fsync`s the containing
directory. A reader therefore always sees either the previous artifact or the
complete new one — never a partial write — which is what makes a crash during
a checkpoint recoverable rather than corrupting (SPECS.md §13: "Provide
restartable training, atomic checkpoints").

## Promotion and rollback

`PromotionLedger` implements the hazard control SPECS.md §12.2 names for
"Wrong artifact deployed": *"Refuse to activate; last-known-good bundle
available immediately."* Promotion requires a named, fully-populated
`Environment` (§13: "A run whose environment cannot be named is not
reproducible"), compatibility validation, and — for a policy bundle — an
evaluation report. `roll_back()` is a local lookup, not a rebuild.

## Testing

`cargo test -p robot-artifacts` — 63 tests, including: FIPS SHA-256 test
vectors; a deterministic adversarial-input sweep over the TOML parser (not
coverage-guided fuzzing — Appendix B.1 correction 9 keeps `cargo-fuzz`'s
libFuzzer engine out of the default toolchain — but a seeded sweep that
asserts the parser always returns rather than panicking or hanging); bit-flip
corruption tests for the chunk format; and a transitive-lineage test proving
held-out exclusion is actually checked, not merely asserted.

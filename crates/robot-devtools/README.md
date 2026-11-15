# robot-devtools

Development-time automation, the evidence ledger and the dependency
knowledge base, per SPECS.md §§2.3.1 and 13.1. Depends on `robot-types`
and `robot-artifacts` — the latter for what it was reserved for:
publishing knowledge-base snapshots as content-addressed artifacts
(`publish`, `devtools kb-publish`).

This crate is developer/CI-only tooling: nothing under `robot-control` or
any other shipping crate depends on it.

## What's here

| Module | Contents |
| --- | --- |
| `knowledge_base` | `KnowledgeBaseEntry`, `Citation`, `require_citation`, `update_version`/`refresh` |
| `exclusion_gate` | `excluded_names_in_lock`, `foreign_packages_in_tree`, `devtools_dependents`: the text checks behind the `exclusion-gate` binary |
| `services_gate` | `check_record`, `check_services`: the §2.4 `robot-service/1` checks over `services/*/service.toml` (rows A-51 and A-52; ADR 0017), run by the same `exclusion-gate` binary |
| `inventory` | `lock_inventory` (Cargo.lock as the §13.1 primary key), the `kb-store/1` `Store` (entries with kind and dependencies, fault records), `orphans`, `sweep` (version change marks an entry and its dependents stale, transitively), local `search`, `ledger_cites_kb` |
| `ledger` | Evidence-ledger rows from every table with a `State` column, and `diff`: added, removed and changed rows, promotions classed by Section 3's order, and promotions with no admitting artifact flagged as unsupported |
| `publish` | `publish_knowledge_base`: snapshots `docs/kb/knowledge-base.toml` byte-for-byte into a `robot_artifacts::BlobStore` and returns the `ContentId`, so a ledger row or automation-log entry can cite one exact knowledge-base state by identity. The snapshot is data about tooling, never evidence; `ledger_cites_kb` still refuses a ledger row citing the base |

The `devtools` binary runs them: `kb-sweep [--write]`, `kb-search
<words>`, `kb-publish <store-dir>` and `ledger-diff <old> <new>`. It exits 1 on a finding: stale
entries left unwritten, orphans, the ledger citing the base, or an
unsupported promotion. `docs/kb/knowledge-base.toml` is the store. It has
no entries, because the default build has no third-party crate and no
hardware has been commissioned, so there is nothing cited to record.

## The exclusion gate (SPECS.md §2.1 item 4, row A-25; ADR 0013)

`cargo run -q -p robot-devtools --bin exclusion-gate` exits 1 if any of
these holds, printing each violation: an excluded runtime (Python
bindings, PyTorch, JAX, the reference simulator, CycloneDDS crate names or prefixes) is in
`Cargo.lock` under any feature; the feature-off default closure reported
by `cargo tree --workspace -e normal` contains a package that is not a
workspace path package; or a crate other than this one lists
`robot-devtools` under `[dependencies]` or `[build-dependencies]`
(`[dev-dependencies]` is allowed, because it ships in nothing). The same
binary also runs the §2.4 services gate: every directory under
`services/` must hold a `service.toml` that is a valid `robot-service/1`
record — unknown keys, a class other than `development-tool`, a pinned
image digest below `code_inspected`, or a qualified record without its
`Containerfile` are each refused by name (row A-51). An absent or empty
`services/` directory passes: the class is admissible, not mandatory.
Exit 2 means the inputs could not be gathered. `devenv test` runs it
before the test suite, and `devenv.nix` declares it as a git hook on
`Cargo.*`, `devenv.*` and `services/` changes, so §2.2 item 6's "no
branching or parsing in a Nix script" holds: the Nix side is a one-line
alias.

## Why an uncited entry can't be used

SPECS.md §13.1: "Every entry MUST be cited and attributed... An uncited
entry is a note, and a note MUST NOT be read as a specification."
`KnowledgeBaseEntry::require_citation` is the only way to read an entry as
usable, and it refuses an entry with `citation: None` — there is no second
path (e.g. trusting `claim` directly) that skips the check.

## Why staleness can't be forgotten

SPECS.md §13.1: "A firmware, driver, crate or robot-profile change marks
every dependent entry stale, exactly as it demotes a claim under Section
3. A stale entry is visibly stale, not silently wrong." `update_version` is
the only method that changes an entry's `version` field, and it
unconditionally sets `stale = true` in the same call — there is no code
path that changes the version without also flagging staleness.
`require_citation` also refuses a stale entry even if it still has a
citation from before the version change, so staleness can't be bypassed by
reusing an old citation; only `refresh` (which requires a new `Citation`)
clears it.

## What's not implemented

- **Hosted CI.** The gate and the `devtools` commands run wherever
  `devenv test` or a developer runs them. No hosted CI exists.
- **Generated ledgers.** The ledger and the automation log are still
  written by hand. `ledger-diff` checks a change to them; it does not
  author one.

## Testing

`cargo test -p robot-devtools` — 29 tests. Publishing: a snapshot
round-trips byte-identically under its own content hash and deduplicates
on republish, a missing input is refused naming its path, and the
committed base publishes. Exclusion gate: an excluded
runtime in a lockfile reported once, look-alike names pass, only workspace
path packages pass the tree check, a normal or build dependency on this
crate fails while a dev-dependency passes, and violations name their
clause. Knowledge base: an uncited entry refused, a
cited fresh entry accepted, a version change both marking an entry stale
and causing `require_citation` to refuse it even with its old citation
still attached, and `refresh` restoring usability with a new citation and
version.

The ledger and knowledge-base tooling adds tests for:
- reading the lockfile inventory with its origin;
- staleness propagating to dependents and surviving a round trip;
- orphans, search, and ledger citations of the base;
- ledger rows and an unsupported promotion;
- a demotion needing no artifact.

`ledger-diff` run on the committed ledger against the working copy reports
one renamed row and no unsupported promotion.

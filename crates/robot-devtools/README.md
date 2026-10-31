# robot-devtools

Development-time automation, the evidence ledger and the dependency
knowledge base, per SPECS.md §§2.3.1 and 13.1. Depends on `robot-types`
and `robot-artifacts` (the latter not yet used — reserved for publishing
knowledge-base snapshots as content-addressed artifacts).

This crate is developer/CI-only tooling: nothing under `robot-control` or
any other shipping crate depends on it.

## What's here

| Module | Contents |
| --- | --- |
| `knowledge_base` | `KnowledgeBaseEntry`, `Citation`, `require_citation`, `update_version`/`refresh` |
| `exclusion_gate` | `excluded_names_in_lock`, `foreign_packages_in_tree`, `devtools_dependents`: the text checks behind the `exclusion-gate` binary |

## The exclusion gate (SPECS.md §2.1 item 4, row A-25; ADR 0013)

`cargo run -q -p robot-devtools --bin exclusion-gate` exits 1 if any of
these holds, printing each violation: an excluded runtime (Python
bindings, PyTorch, JAX, the reference simulator, CycloneDDS crate names or prefixes) is in
`Cargo.lock` under any feature; the feature-off default closure reported
by `cargo tree --workspace -e normal` contains a package that is not a
workspace path package; or a crate other than this one lists
`robot-devtools` under `[dependencies]` or `[build-dependencies]`
(`[dev-dependencies]` is allowed, because it ships in nothing). Exit 2
means the inputs could not be gathered. `devenv test` runs it before the
test suite, and `devenv.nix` declares it as a git hook on `Cargo.*` and
`devenv.*` changes, so §2.2 item 6's "no branching or parsing in a Nix
script" holds: the Nix side is a one-line alias.

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

The evidence-ledger tooling named in §13.1 (ledger diffing, automated
staleness sweeps triggered by a real version bump in `Cargo.lock`) is not
built in this revision — only the data model and its two core invariants.
The "no shipping crate imports `robot-devtools`" assertion is built (the
gate above) but runs only where `devenv test` runs; no hosted CI exists.
`docs/ledger/evidence-ledger.md` and `docs/ledger/automation-log.md` are
maintained by hand for now.

## Testing

`cargo test -p robot-devtools` — 9 tests. Exclusion gate: an excluded
runtime in a lockfile reported once, look-alike names pass, only workspace
path packages pass the tree check, a normal or build dependency on this
crate fails while a dev-dependency passes, and violations name their
clause. Knowledge base: an uncited entry refused, a
cited fresh entry accepted, a version change both marking an entry stale
and causing `require_citation` to refuse it even with its old citation
still attached, and `refresh` restoring usability with a new citation and
version.

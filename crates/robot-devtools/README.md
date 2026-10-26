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

The actual evidence-ledger tooling named in §13.1 (ledger diffing, CI
assertions that no shipping crate imports `robot-devtools`, automated
staleness sweeps triggered by a real version bump in `Cargo.lock`) is not
built in this revision — only the data model and its two core invariants.
`docs/ledger/evidence-ledger.md` and `docs/ledger/automation-log.md` are
maintained by hand for now.

## Testing

`cargo test -p robot-devtools` — 4 tests: an uncited entry refused, a
cited fresh entry accepted, a version change both marking an entry stale
and causing `require_citation` to refuse it even with its old citation
still attached, and `refresh` restoring usability with a new citation and
version.

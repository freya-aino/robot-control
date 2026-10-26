# robot-skills

Skill descriptors, exact retrieval and the R0/R1/R2 tiers of SPECS.md §10.6.
Depends on `robot-types` and `robot-artifacts` (the latter not yet used —
reserved for loading a skill manifest from a content-addressed blob).

## What's here

| Module | Contents |
| --- | --- |
| `descriptor` | `SkillDescriptor`: the typed, authoritative record of §10.2 |
| `index` | `ExactIndex`: brute-force nearest-neighbor search, with a zero-allocation `nearest_into` for the R2 path |
| `tiers` | `WorkingSet` (R1, generation-tagged, immutable) and `Router` (R2, resident, stale-generation-rejecting) |

## Why exact search, and why it's zero-allocation on R2

SPECS.md §10.1: "For small libraries, benchmark exact vector search in Rust
before adopting approximate nearest-neighbor indexing." No such benchmark
has been run, so `ExactIndex` is a plain linear scan — the honest default,
not a placeholder for something better left unbuilt. §10.6 item 1 is
stricter for the R2 tier specifically: "An R2 lookup MUST... NOT read a
file, query a database, compute an embedding, allocate, or block."
`ExactIndex::nearest_into` writes into a caller-supplied fixed buffer and
is the only query path `tiers::Router` calls.

## What's explicitly not here

This is a narrow slice of §10, named as such in the crate's own module
docs: no learned embeddings (§10.1 asks for structured tags and numerical
features *first*), no Markdown skill-descriptor catalog (§10.1's
operator-facing, non-authoritative extension), and no feasibility-filter or
hysteresis machinery beyond `Router::switch_to`'s residency check. A real
feasibility filter needs the deployed robot's own state (stance, contact,
reach) as input, which belongs with whatever crate owns that state at
integration time.

## Testing

`cargo test -p robot-skills` — 8 tests: nearest-neighbor ordering and
dimension validation, an empty-index report, R1→R2 generation rejection
(including the "equal generation is not newer" edge case), the empty-result
degrade-to-current-controller path, and a refused switch to a non-resident
skill leaving state unchanged.

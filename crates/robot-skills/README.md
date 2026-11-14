# robot-skills

Skill descriptors, exact retrieval and the R0/R1/R2 tiers of SPECS.md §10.6.
Depends on `robot-types` and `robot-artifacts` (the latter not yet used —
reserved for loading a skill manifest from a content-addressed blob).

## What's here

| Module | Contents |
| --- | --- |
| `descriptor` | `SkillDescriptor`: the typed, authoritative record of §10.2 |
| `manifest` | The `skill-manifest/1` TOML format (ADR 0016): a skill library's typed descriptors on disk, round-tripping exactly; unknown keys, duplicates and invalid fields refused by name |
| `index` | `ExactIndex`: brute-force nearest-neighbor search, with a zero-allocation `nearest_into` for the R2 path |
| `tiers` | `WorkingSet` (R1, generation-tagged, immutable) and `Router` (R2, resident, stale-generation-rejecting) |
| `selection` | `Selector`, the R2 decision. Recovery or the protective response comes first when falling, preempting dwell. Then nearest resident candidates, an out-of-distribution distance, a confidence margin, and `GuardTable` feasibility over `FeasibilityState` (stance, joint speed, support, actuator health, opponent distance), where an unknown guard fails closed. Then transition contracts, minimum dwell and hysteresis. A switch carries a `RecurrentAction::Reset` contract. Accepted proposals expire. Rejections are counted by reason and switches by skill in fixed arrays |
| `cache` | `ExpertCache`: activation checks (schema, checksum, normalizer, memory budget) in §10.3's order, admission and eviction. `Stager` validates off the control thread with non-blocking `poll` |
| `catalog` | §10.1's optional Markdown catalog: TF-IDF lexical search with no model call. `compose` can only reorder numerical candidates, never add one. `disagreements` flags prose that contradicts the typed contact assumption |

## Why exact search, and why it's zero-allocation on R2

SPECS.md §10.1: "For small libraries, benchmark exact vector search in Rust
before adopting approximate nearest-neighbor indexing." No such benchmark
has been run, so `ExactIndex` is a plain linear scan — the honest default,
not a placeholder for something better left unbuilt. §10.6 item 1 is
stricter for the R2 tier specifically: "An R2 lookup MUST... NOT read a
file, query a database, compute an embedding, allocate, or block."
`ExactIndex::nearest_into` writes into a caller-supplied fixed buffer and
is the only query path `tiers::Router` calls.

## Decisions worth knowing

**R2 allocates nothing.** The selector's candidate buffer, descriptor
table, guard table and counters are sized at R1. A decision reads only
resident data (§10.6 item 1).

**Unknown guards fail closed.** A descriptor naming an initiation guard no
code implements is not feasible. Prose or a name is not a check.

**Falling is not a guard-posture problem.** §10.4: "Guard posture is not a
universal fallback when already falling." When falling, the configured
recovery skill is taken if its own guards pass, preempting dwell.
Otherwise the selector returns `Protective`, the robot-specific response
the caller runs.

**The catalog cannot widen the choice.** §10.1: the Markdown descriptor
"MUST NOT replace [numerical retrieval] or relax any feasibility check".
`compose` reorders the numerical candidates and nothing else, and
feasibility runs afterwards.

## What is not here

- **Learned embeddings.** §10.1 adds them "only when they improve a
  measured outcome", and no such measurement exists.
- **The §10.5 design comparison.** It needs trained experts at equal
  budgets. The counters it reports from (routing frequency, rejections,
  fallbacks) are here.

## Testing

`cargo test -p robot-skills`: 18 tests. 8 are the original ones: nearest-neighbor ordering and
dimension validation, an empty-index report, R1→R2 generation rejection
(including the "equal generation is not newer" edge case), the empty-result
degrade-to-current-controller path, and a refused switch to a non-resident
skill leaving state unchanged. The new ones cover:
- switching with a reset contract and proposal expiry;
- dwell, out-of-distribution and unknown-guard rejections;
- recovery preempting dwell, and the protective response;
- an empty router keeping the current skill;
- each activation check refusing in turn;
- the memory budget;
- off-thread staging;
- TF-IDF ranking;
- composition that never adds a candidate;
- prose/typed disagreement.

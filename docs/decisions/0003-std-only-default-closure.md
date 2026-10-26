# 0003 — the default dependency closure is empty (standard library only)

- Status: accepted
- Date: 2026-10-26
- Touches: SPECS.md §2, §2.1, Appendix B.6, `PLAN.md` Phase 0 item "Benchmark
  one Rust training/inference backend and one Rust physics candidate"

## Context

SPECS.md Appendix B.1 states that every Appendix B candidate — including
`nalgebra` (linear algebra), `Burn`/`CubeCL` (learning), `Rapier` (physics),
`tracing` (instrumentation) — "start[s] at `unverified` under Section 3 until
a primary-source record establishes a higher evidence state." Appendix B.6
requires, before any candidate enters the tree: a recorded canonical
repository/version/license/inspected source (gate 1), an inventory of
enabled features and native linkage (gate 2), a reproduced capability on our
own fixtures with measured resource use (gate 3), and so on through gate 8.
None of that qualification work has been performed in this revision.

## Decision

`Cargo.toml`'s `[workspace.dependencies]` table is empty, and no crate in
this workspace declares a third-party dependency, as of this revision. Every
numeric, randomness, hashing, serialization and XML-parsing need the crates
so far implemented (`robot-types`, `robot-artifacts`, `robot-model`) have
encountered is met by authored code against the standard library, each with
its own analytic, statistical or round-trip test:

| Need | Authored in | Qualification status of the Appendix B shortlist candidate |
| --- | --- | --- |
| Dense linear algebra, LU solve, symmetric eigendecomposition, damped least squares | `robot_types::math` | `nalgebra` (B.2.1): unqualified |
| Deterministic PRNG, seed streams, reservoir sampling | `robot_types::rng` | no shortlisted candidate; B.2.12 names only parallelism primitives |
| SHA-256 content hashing | `robot_artifacts::hash` | no shortlisted candidate |
| Bounded TOML subset | `robot_artifacts::toml` | no `toml`-crate candidate named in Appendix B at all |
| Versioned binary chunk framing | `robot_artifacts::chunk` | B.2.7's Arrow/Parquet/MCAP candidates are unqualified and aimed at a different use (visualization/interchange, not the transition-store framing SPECS.md §7.5/§7.6 specify) |
| Minimal XML reading for URDF | `robot_model::urdf` | `urdf-rs` (B.2.4): unqualified |

## Why this is a decision, not an oversight

SPECS.md §2 is explicit that "A Rust API alone does not establish that a
library or its enabled dependency closure satisfies this policy," and
Appendix B.6's closing line states "A failed gate leaves the candidate
reference-only... No claim in this appendix constitutes permission for
foreign-runtime exceptions." Authoring a narrow, tested slice of
functionality against `core`/`std` is compatible with every one of those
constraints by construction: there is no feature closure to inventory, no
license to review, no transitive native linkage to audit, and no foreign
runtime to exclude.

## Consequences

- This keeps the Phase 0 exit gate ("A Rust-only executable can run a small
  controlled physics task...") achievable without first completing the
  Appendix B.6 qualification backlog, at the cost of carrying more authored
  code than a qualified dependency would require.
- It is **not** a permanent architectural position. `PLAN.md`'s first
  executable backlog item 3 ("Benchmark one Rust training/inference backend
  and one Rust physics candidate") remains open work. When Burn and Rapier
  (or an alternative) clear the Appendix B.6 gates, `robot-learn` and
  `robot-sim` are the crates expected to adopt them; `robot_types::math`'s
  authored `Matrix` is not expected to be rewritten as a competitor to
  `nalgebra` — it is expected to be replaced or thinned once `nalgebra`
  itself clears qualification, since Appendix B.2.1 already shortlists it as
  the default candidate for exactly this role.
- Every module built against this decision documents the specific Appendix B
  candidate it stands in for, in its own doc comment, so a future
  qualification PR has a direct pointer to what it is expected to replace or
  retire — see the module-level docs of `robot_types::math`, `robot_types::
  rng`, `robot_artifacts::hash`, `robot_artifacts::toml`, `robot_artifacts::
  chunk` and `robot_model::urdf`.
- This record itself is the artifact that keeps "no Rust implementation was
  found" from being mistaken for "a Rust implementation is not possible":
  Section 2.1 item 1 calls the former "a search result, not a measurement,"
  and nothing above claims otherwise.

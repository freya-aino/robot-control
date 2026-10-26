# 0004 — extracting `robot-nn` and `robot-supervisor` from the §6 table

- Status: accepted
- Date: 2026-10-26
- Touches: SPECS.md §6, Appendix B.3, §12.1

## Context

SPECS.md §6's workspace table does not list `robot-nn` or `robot-supervisor`.
It places "Rollouts, autodiff, PPO baseline, losses, optimizers, checkpoints"
inside `robot-learn`, and "Deterministic runtime, inference, arbitration,
supervision" inside `robot-control`. Appendix B.3's closing paragraph is the
governing rule for any departure: *"The survey's proposed `robot-state`,
`robot-ctrl` and `robot-test` names map to the existing estimation, control
and evaluation boundaries above. Extract new crates only if ownership or
dependency isolation warrants it; do not maintain duplicate responsibilities."*

This record applies that rule to two extractions this implementation makes.

## Decision

### `robot-nn`: forward-only inference, extracted from `robot-learn`

§6 itself states the constraint this satisfies: *"Training and deployment
share observation/action schemas and tested normalization logic. Deployment
does not depend on training autograd code."* If the forward pass used by a
deployed actor lived inside `robot-learn` (which also owns autodiff,
optimizers and the PPO update), then `robot-control` would have to depend on
`robot-learn` to run inference — and would transitively carry every training
dependency into the deployed closure. Putting the shared forward-only
operator set in its own crate, with `robot-learn` depending on it (for
rollout-time inference) and `robot-control` depending on it (for deployed
inference), makes "deployment does not depend on training code" a property
Cargo's dependency graph enforces, not a convention reviewers check by hand.
This is "dependency isolation" in Appendix B.3's own words.

### `robot-supervisor`: the independent supervisor, extracted from `robot-control`

§12.1 states: *"The supervisor MUST be the last stage before the actuator
interface... and MUST NOT import the policy, retrieval or personality
crates."* As a module inside `robot-control`, that non-import rule is a
code-review convention: nothing stops a future edit to a sibling module from
adding a `use` that reaches `robot-skills` or `robot-persona` and having the
supervisor module incidentally transitively see it through shared crate-level
imports or re-exports. As its own crate with its own `Cargo.toml`, the
constraint becomes structural: `robot-supervisor` simply has no dependency
edge to `robot-skills`, `robot-persona` or `robot-learn`, and cannot acquire
one without a visible `Cargo.toml` change that any reviewer — or the
exclusion-gate tooling of §2.1 — can check mechanically. This is the same
"ownership or dependency isolation" justification Appendix B.3 names, applied
to a safety-critical boundary rather than a build-time one.

## What does not change

- No new `robot-state` or `robot-ctrl` crate is created; `robot-perception`
  and `robot-control` remain the estimation and control boundaries exactly
  as Appendix B.3 directs.
- `robot-control` still depends on `robot-supervisor` (supervision remains
  reachable from the control path, as §6 requires) — the extraction moves
  *where the code lives*, not *whether it runs*.
- `robot-learn` still owns autodiff, optimizers and the PPO update; only the
  forward-only operator set moved out.

## Consequences

- Every future crate addition should be checked against this same rule
  before being justified by convenience alone: "ownership or dependency
  isolation warrants it," not "it seemed like a clean module boundary."
- `robot-control`'s and `robot-learn`'s own `README.md` files cross-reference
  this record rather than re-deriving the justification.

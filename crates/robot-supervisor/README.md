# robot-supervisor

The independent supervisor of SPECS.md §12.1: the last stage before the
actuator interface. Depends only on `robot-types` and `robot-model`.

This crate is **not** in SPECS.md §6's workspace table. Its extraction is
justified and recorded in
[`docs/decisions/0004-nn-and-supervisor-crate-extraction.md`](../../docs/decisions/0004-nn-and-supervisor-crate-extraction.md):
§12.1 requires the supervisor to "MUST NOT import the policy, retrieval or
personality crates," and as a *crate* with no dependency edge to
`robot-learn`, `robot-skills` or `robot-persona`, that rule is something
`cargo tree` confirms rather than something a reviewer has to remember to
check.

## What's here

| Module | Contents |
| --- | --- |
| `fault` | `Fault` (ten categories, each carrying its measurement and limit), `FaultLog<N>`: a fixed-capacity, allocation-free record with an overflow counter |
| `policy` | `Response` (clamp / hold / reject / e-stop), `FaultPolicy`: a validated, total category→response mapping |
| `supervisor` | `Supervisor<N>`: construction-time setup from a `RobotProfile`, then a per-tick `check_tick` that allocates nothing |

## "Clamp-and-continue is not sufficient for every fault"

SPECS.md §12.1 states this as a requirement, and §12.2's hazard register
names which faults need more: "Uncommanded motion" (stale command / mode
confusion) is controlled by "watchdog outside the policy process," and "Loss
of transport" by "firmware-side command expiry" — neither is a clamp.
`FaultPolicy::new` enforces this as a constructor-time check: it refuses to
build a policy that assigns `Response::ClampAndContinue` to `NonFiniteObservation`,
`StaleObservation`, `CommunicationsLost`, `NonFiniteCommand` or
`ActuationNotPermitted`. A caller that tries gets a `PolicyError` back, not a
policy that silently does the wrong thing on a robot.

## Allocation discipline

SPECS.md §5 extends to the supervisor explicitly: "Its checks run within the
control tick and are subject to the same prohibitions as Section 5: no
allocation." Every `Vec` in `Supervisor` is sized once in `Supervisor::new`
(construction, "before activation," per §5) from the robot profile's joint
and actuator counts. `check_tick` writes into caller-supplied `raw`/`out`
slices and a fixed-size `FaultLog<N>` array; nothing in its call path touches
an allocator. The `N` type parameter is the caller's declared bound on
how many distinct faults one tick could raise — chosen generously (the
doc comment suggests `3 * joint_count + actuator_count` as a floor) — and an
overflow past that bound increments a counter rather than growing, matching
§5's "Log overflow must never block actuation; emit dropped-record counts."

## What it does not do

It does not decide *why* a command was wrong beyond the ten fault
categories it checks, and it does not interpret a latent or
position-residual action semantically (`CommandKind::Unbounded` skips the
physical-limit checks for those, checking only finiteness) — a controller
using such a semantics needs its own bounding logic upstream, because
bounding a residual or a latent vector requires the reference or decoder the
supervisor deliberately has no access to.

## Testing

`cargo test -p robot-supervisor` — 21 tests: the escalation rule's
construction-time refusal (including the "custom policy may still choose any
non-clamp response freely" positive case), clamping arithmetic, every
escalation path (stale observation, non-finite command, unauthorized
contact, wrong-mode actuation), the post-e-stop "the stop itself becomes the
new last-safe-command" behavior, and fault-log overflow accounting under a
deliberately undersized capacity.

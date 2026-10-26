# motion-validate

Kinematic and dynamic feasibility reports, per SPECS.md §8.3. Depends on
`robot-types` and `robot-model`.

## What's here

| Module | Contents |
| --- | --- |
| `report` | `ValidationReport` (kinematic error, dynamics violations, contact consistency, kept as separate fields), `AcceptanceThresholds` (caller-supplied, never a built-in constant), `check_position_limits` |

## Why there is no single pass/fail field

SPECS.md §8.3: "A validation report must separate kinematic error,
dynamics violations, contact consistency and task success... No universal
fidelity percentage is a release criterion." `ValidationReport` has no
`passed: bool` — `Verdict::evaluate` exists precisely because whether a
motion is accepted depends on thresholds the caller supplies per robot and
task, and a rejection names *every* violating frame, not just the first
one found, so a report can be read once rather than re-run per failure.

## What's not implemented

Root drift and planted-foot slip are fields on `KinematicError`, but
nothing in this crate yet *computes* them from a reference and a
retargeted trajectory — that requires `motion-retarget`'s output and a
contact schedule this crate does not yet consume. `check_position_limits`
is the one concrete check implemented end to end.

## Testing

`cargo test -p motion-validate` — 3 tests: acceptance within threshold,
a rejection naming every violating frame (not just the first), and
position-limit excess computed against a joint's declared range.

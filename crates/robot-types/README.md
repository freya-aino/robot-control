# robot-types

Units, frames, time, typed identifiers and the versioned record schemas every
other crate in this workspace agrees on. This is the bottom of the dependency
graph: it has no third-party dependencies (see
[`docs/decisions/0003-std-only-default-closure.md`](../../docs/decisions/0003-std-only-default-closure.md))
and everything else in the workspace depends on it, directly or transitively.

## What's here

| Module | SPECS.md anchor | Contents |
| --- | --- | --- |
| `units` | Appendix A.3 | SI newtypes (`Metres`, `Radians`, `Hertz`, …), dimensioned arithmetic, `Range<T>` |
| `time` | §5 | Monotonic `Duration`/`Timestamp`, the sampling-instant-vs-arrival-time distinction, `ClockId` |
| `frames` | §7.2, Appendix A.3 | `Vec3`, named-component `Quat` (no bare `[f64;4]`), `EulerOrder`, `Transform` that names the frames it maps between |
| `ids` | §7.1 | Typed, non-interchangeable `JointId`/`LinkId`/`ActuatorId`/`SensorId`/…, `Name`, namespaced `ExtensionKey` |
| `schema` | §7.1 | `SchemaVersion` compatibility, `Strictness` (activating vs. preserving), `ExtensionTable`, `Migration` |
| `evidence` | §3 | The five ordered evidence states, promotion/demotion, "evidence does not transfer between artifacts" |
| `mode` | §4 | `OperatingMode`, actuation authority, interlocks, entry preconditions |
| `tier` | §5.1 | `Tier`, `TierAssignment`, `TierSchedule` validation (non-increasing rates, declared-tier-or-no-activation), `TierOutput` freshness |
| `rng` | §7.6 item 4 | Authored PCG-XSH-RR, `SeedStream` derivation, `Reservoir` sampling |
| `math` | Appendix B.2.1 | Authored dense `Matrix`: LU solve, symmetric eigendecomposition, damped least squares for IK |
| `obs` | §9.2, §6 | `ObservationSpec` with the privileged-input audit, `ActionSpec`/`ActionSemantics`, the shared `Normalizer` |
| `motion` | §7.2 | The canonical motion record, kept distinct by kind (`HumanSkeleton` / `RobotReference` / `BodyModelParameters`) |
| `episode` | §7.3 | The robot episode record: raw vs. applied commands, torque provenance, supervisor interventions |
| `transition` | §7.6 | The shared on-policy/off-policy transition schema, latent-vs-joint-space rejection, `Termination` |
| `bundle` | §7.4 | The policy bundle manifest and `check_activatable`, which is the "wrong artifact deployed" hazard control of §12.2 |

## Why no dependencies

Appendix B.6 admits no third-party crate until its qualification gates pass,
and Appendix B.1 starts every surveyed candidate — including nalgebra, the
Appendix B.2.1 shortlist for linear algebra — at `unverified`. Rather than
block on that qualification, this crate authors the narrow slice of linear
algebra, hashing-adjacent randomness and dense-matrix work the platform
actually needs, each with an analytic or property test. This is a decision
record, not an oversight: see `docs/decisions/0003-std-only-default-closure.md`.

## Design rules this crate enforces

- **No project-wide frame convention.** C-04 is open. `Transform` carries its
  `from`/`to` `FrameId`s and refuses to compose across a mismatch.
- **No bare quaternion arrays.** `Quat`'s fields are named; there is no
  `[f64; 4]` constructor.
- **Sampling instant vs. arrival time is a type, not a comment.** `Timestamp`
  carries a `StampKind` and an uncertainty bound.
- **A component with no declared tier cannot be activated.** `TierAssignment`
  requires a measured compute duration as a condition of construction, and
  `TierSchedule::validate` enforces the Section 5.1 invariants mechanically.
- **Evidence does not transfer between artifacts.** `Claim::promote` takes the
  admitting artifact's own subject and rejects a mismatch.

## Testing

`cargo test -p robot-types` — unit and property-style tests per module,
no integration tests (nothing to integrate with yet). Every numeric routine
in `math` and `rng` is checked against an analytic or statistical reference
rather than against its own output.

# robot-viewer

Inspection data layer for skeletons, contacts and tracking error, per
SPECS.md §6 and PLAN.md's Phase 2. Depends on `robot-types` and
`robot-model`.

## What's here

| Module | Contents |
| --- | --- |
| `frame` | `ViewerFrame` (one inspection snapshot), `ContactSample` |
| `timeline` | `Timeline`: ordered, single-clock frame sequence with `nearest()` for synchronized playback/scrubbing |

## Why there's no rendering here

SPECS.md §11.2's viewer shortlist is "Rerun or egui; `urdf-viz` for model
inspection," but Appendix B.1 marks every survey candidate `unverified`,
and Appendix B.6's dependency-qualification gate has not been run against
any of them. This workspace also keeps `[workspace.dependencies]` empty by
policy (`docs/decisions/0003-std-only-default-closure.md`). So this crate
is the data a renderer would need — frames keyed by timestamp, source and
target skeleton poses, contact state, per-joint tracking error — and stops
there. PLAN.md's own framing supports this: §4's CLI surface and §3's
Phase 2 description describe the viewer as a `devenv processes`-hosted
development service (SPECS.md §2.5 item 5: "Nothing declared there is...
a runtime dependency of a deployed robot"), not something `robot-control`
or any shipping crate depends on.

## Why `Timeline` enforces one clock and strict ordering

SPECS.md §5 never treats readings from two different clocks as directly
comparable without a recorded mapping; a playback timeline that silently
mixed `ClockId::Simulation` and `ClockId::HostMonotonic` frames would
reintroduce exactly the confusion `Timestamp::since` exists to prevent.
`Timeline::push` rejects a frame on the wrong clock and rejects a frame
that isn't strictly later than the last one appended — a timeline is an
append-only, time-ordered log, not a general collection you insert into
out of order and sort later.

## Why `worst_tracking_error` returns `Option<f64>`

A frame with no tracking-error data at all (nothing to compare against,
e.g. no reference skeleton loaded) is a different fact than a frame whose
tracking error is genuinely zero. Collapsing both to `0.0` would make a
missing comparison look like a perfect one.

## What's not implemented

Any actual rendering, a `robot-cli` command that drives this crate, and
reading frames back out of `robot-artifacts`' episode/run storage — this
revision only has the in-memory data model and the playback-lookup logic
over it.

## Testing

`cargo test -p robot-viewer` — 8 tests: worst-tracking-error magnitude vs.
sign and the missing-data case, contact filtering, out-of-order and
wrong-clock frame rejection, nearest-frame lookup including the
tie-break-toward-earlier rule, lookup on an empty timeline returning `None`
rather than an error, and lookup with a mismatched clock returning an
error. (`robot-model` also gained one small addition in this pass:
`LinkPoses::from_poses`, letting a consumer build a `LinkPoses` from
poses it already has — e.g. read back from a recording — without needing
a `RobotProfile` and joint values to recompute them.)

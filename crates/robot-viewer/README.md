# robot-viewer

Inspection data layer for skeletons, contacts and tracking error, per
SPECS.md §6 and PLAN.md's Phase 2. Depends on `robot-types` and
`robot-model`.

## What's here

| Module | Contents |
| --- | --- |
| `frame` | `ViewerFrame` (one inspection snapshot), `ContactSample` |
| `timeline` | `Timeline`: ordered, single-clock frame sequence with `nearest()` for synchronized playback/scrubbing |
| `pose` | `PoseFrame` (pelvis pose and joint angles per step), `FrameDump` (transactional `--out` writer), `load_committed` (strict read-back) |
| `csv`, `jsonl` | Strict readers for the two frame formats written by `FrameDump` |
| `svg` | Std-only SVG renderer for one `PoseFrame`, pinned by a golden file in `tests/golden/` |

The frame dump format, bounds and commit protocol are fixed by `docs/decisions/0009-test-render-format.md`.

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

Rendering is limited to the std-only per-frame SVG in `svg`: no viewer, no skeleton (only the pelvis and joint bars are drawn), no interactive playback. Frame dumps are read back from their own directory layout (`pose::load_committed`), not from `robot-artifacts` episode/run storage. Runs are capped at 999 999 steps by the six-digit SVG names, and frame files are bounded by line length, joint count and JSON nesting depth (ADR 0009 §5).

## Testing

`cargo test -p robot-viewer` runs unit tests for the `Timeline` lookups and rejections, the CSV and JSONL strict readers (malformed, truncated and extra-record inputs), the streaming pair verifier (including `-0.0` vs `0.0`), the frame-line and nesting bounds, and the SVG renderer. Integration tests in `tests/pose_frames.rs` cover the golden SVG, the round trip and the commit protocol. Regenerate the golden with `ROBOT_VIEWER_UPDATE_GOLDEN=1 cargo test -p robot-viewer --test pose_frames`. (`robot-model` also gained one small addition in this pass: `LinkPoses::from_poses`, letting a consumer build a `LinkPoses` from poses it already has, e.g. read back from a recording, without needing a `RobotProfile` and joint values to recompute them.)

# robot-viewer-app

The offline replay viewer of ADR 0019, over `robot-viewer`'s data layer,
with Bevy 0.20.1 behind the off-by-default `bevy` feature (qualification:
ADR 0024). Depends on `robot-types`, `robot-model`, `robot-artifacts` and
`robot-viewer`; the default build compiles no foreign code at all.

## What's here

| Part | Contents |
| --- | --- |
| `lib` (std-only, always built) | `ReplayData`: loads a committed frame-dump directory (`load_dir`) or a `viewer-frames/1` run from a blob store (`load_store`), maps the dump's joint names onto the fixture's actuated joints, poses every link per frame through `robot_model::kinematics::forward_kinematics_floating`, and prepares every collision shape as plain data — spheres, boxes, cylinders and hulls triangulated via `robot_model::hull::hull_faces`. `bones` gives the skeleton segments |
| `robot-viewer-app` binary (`--features bevy`) | The Bevy window: the robot's 33 contactable collision shapes as lit meshes, a skeleton and metre-grid gizmo overlay, looped playback. `Space` play/pause, `Left`/`Right` step, `Up`/`Down` speed, `Home` restart, left-drag orbits, scroll zooms; the window title carries frame, time and speed. `--dump` prints the loaded run and exits without a window — the headless path CI exercises |

```
cargo run -p robot-viewer-app --features bevy -- <run-dir> [--robot g1-usd|g1]
cargo run -p robot-viewer-app --features bevy -- <store-dir> --id sha256:<hex>
```

The canonical frame is Z-up (ADR 0022); Bevy renders Y-up, so every
robot-frame entity hangs under one root rotated −90° about X and robot
coordinates are used unchanged below it.

## Why the picture can be trusted

The shapes rendered are exactly the collision geometry the simulator
contacts (specs/07): the 18 primitives plus the 15 import-time convex
hulls, and nothing else — a raw mesh reference that physics cannot touch
is not drawn either, so the viewer never shows surfaces the dynamics
ignores. Poses come from the same forward kinematics the rest of the
workspace uses, with the recorded pelvis pose carrying the fixed-base
tree.

## Windowing

X11 only (XWayland on a Wayland host): the stock `3d` feature's wayland
backend needs a system dev library the devenv deliberately omits, and the
deviation is recorded in ADR 0024. Rendering goes through wgpu/Vulkan; on
a host whose shell lacks the Vulkan loader, the same explicit
loader/ICD environment documented in `robot-sim-gpu`'s README applies.

## Testing

Feature off (the default gate): `cargo test -p robot-viewer-app` — the
replay loads a synthetic two-frame log over the USD G1's own joint names,
poses every link (pelvis where the frame says), prepares all 33 shapes
with hulls present, and refuses an unknown robot or a joint-name mismatch
by name. Feature on: `cargo build -p robot-viewer-app --features bevy`
plus the `--dump` round trip against a recorded rollout-tier run (ADR
0024's verification); the windowed run needs a display.

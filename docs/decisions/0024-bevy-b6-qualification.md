# 0024. Bevy 0.20.1 passes B.6 for the offline replay viewer

- Date: 2026-11-15
- Tracks: ADR 0019 (the candidate designation this qualifies)
- Touches: `crates/robot-viewer-app` (new), workspace members, `Cargo.lock`.

Status: accepted — qualified for the development-tool viewer app only.
Nothing here qualifies Bevy for any shipping, control-path or robot-host
use, and the default workspace build still compiles none of it.

## What was qualified, exactly

**`bevy = "=0.20.1"`** (released 2026-10-28, `MIT OR Apache-2.0`),
`default-features = false`, with the feature set assembled by hand as
`3d` minus `default_platform`:

```
default_app, std, multi_threaded, bevy_winit, x11,
3d_bevy_render, 3d_api, bevy_pbr, bevy_gizmos_render
```

**One deviation, and why**: the stock `3d` feature pulls
`default_platform`, whose wayland backend (`wayland-sys`) demands a
system `wayland-client` dev package at build time. The devenv
deliberately does not carry it (Section 2.2's rule against speculative
inputs), so the qualified windowing path is **X11, under XWayland on a
Wayland host**. Native wayland is one devenv input away if it is ever
wanted; that would be a devenv change reviewed on its own.

## Isolation (the ADR 0019 plan, implemented)

`crates/robot-viewer-app` (`publish = false`) follows `robot-sim-gpu`'s
cubecl pattern exactly: Bevy sits behind the off-by-default `bevy`
feature, the binary requires it, and the crate's library — run loading,
per-frame posing through `forward_kinematics_floating`, shape
preparation including hull triangulation via `hull_faces` — is std-only,
compiled and tested in the default gate. `robot-viewer` remains the data
layer and takes no new dependency.

## Measurements (this host, 2026-11-15, dev profile)

- Clean build of the enabled dependency closure: **117 s**; incremental
  app rebuild and link: **9 s**.
- Enabled external closure: **336 crates** (the lockfile grew from 473
  to 712 packages; the default-closure gate still counts only workspace
  paths, and the exclusion gate passes with no excluded runtime among
  them).
- License sweep over all 336 vendored manifests: every license is
  permissive — MIT/Apache-2.0 dual licensing dominates (223), with MIT
  (38), Unicode-3.0 (22), Apache-2.0 (7), Zlib (5), Unlicense-or-MIT
  (4), ISC, BSD-2/3-clause, MIT-0, CC0 and 0BSD making up the rest.
  **No copyleft, no unknown, no license-file-only crate.**
- Headless verification: a 400-frame rollout-tier G1 run
  (`robot sim multibody --floating --tier rollout --steps 400 --out …`)
  loads through `robot-viewer-app --dump` — 400 frames, 33 shapes (the
  same collider set the simulator contacts, hulls included), link
  positions matching the recording.
- A windowed run needs a display and a Vulkan loader; on this host the
  loader situation is the one `robot-sim-gpu`'s README records
  (`LD_LIBRARY_PATH` to the system Mesa), and the first interactive run
  is the project owner's to perform.

## Consequences

- The viewer's first deliverable (offline replay, ADR 0019) exists:
  collision geometry plus skeleton overlay, play/pause/step/speed,
  orbit camera, Z-up handled by one rotated root.
- API drift risk is real and accepted for a dev tool: 0.20 renamed
  `EventReader` to `MessageReader`, moved `AmbientLight` per-camera and
  renamed shadow flags since the APIs this project's tooling knew;
  staying pinned at `=0.20.1` makes that a choice to revisit, not a
  silent break.
- Live-simulation hooks remain future work and are not licensed by this
  qualification.

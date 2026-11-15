# robot-sim-gpu

Feature-gated CubeCL f32 backend for a subset of the `robot-sim` functions
(SPECS.md §9.1, PLAN.md Phase 3, ADR 0012). Off by default: nothing here
compiles into the default workspace build, and `cubecl` is pulled in only
with `--features cubecl`. The crate is unpublished (`publish = false`) and
keeps `#![deny(unsafe_code)]`, with the one launch block documented.

The authoritative status of what is and is not ported lives in
[`crates/robot-sim/specs/11-gpu-batched-backend.md`](../robot-sim/specs/11-gpu-batched-backend.md)
and the qualification record in
[`docs/decisions/0012-cubecl-b6-qualification-gpu-backend.md`](../../docs/decisions/0012-cubecl-b6-qualification-gpu-backend.md).
This README does not restate them.

## What's here

| Module | Contents |
| --- | --- |
| `integrators` | Semi-implicit Euler, implicit-damping Euler (`nv <= 6`), a named pendulum RK4 step |
| `constraint` | Joint-equality residual and Jacobian, `solimp` impedance with `impP`, equality-only batched solve (`nv, nrow <= 6`) |
| `contact` | Plane-sphere contact (distance, position, normal) |
| `launch` (private) | Slice-length and workgroup/buffer-limit validation before any device work |

## What's not implemented

Batched articulated (G1/ABA) dynamics, any contact, friction or limit
solver, the contact-slot budget and its exhaustion counter, the general
closure-based RK4, dense solves above six DOF, and cross-device
determinism. No throughput claim and no simulator-parity claim is made; the fidelity
battery of `robot-sim` does not run on this crate.

## Testing

Feature off: `cargo test -p robot-sim-gpu` — unit tests plus
`tests/tolerance_records.rs` and `tests/resource_accounting.rs`, which
parse `tolerances.toml` without a device. Feature on (needs a Vulkan
adapter; ADR 0012 records Intel and Mesa Lavapipe runs):
`cargo test -p robot-sim-gpu --features cubecl` runs the CPU-vs-GPU
agreement, implicit, equality and repeatability suites. The two examples
(`gpu_latency`, `gpu_resource_repeatability`) print measurements and
assert nothing.

On a host whose devenv shell carries no Vulkan loader (the current one),
the suite still runs under Mesa Lavapipe by pointing the loader and ICD at
the system's Mesa explicitly — verified green (49 tests) on 2026-11-15:

```
LD_LIBRARY_PATH=<nix-store vulkan-loader>/lib \
VK_ICD_FILENAMES=/run/opengl-driver/share/vulkan/icd.d/lvp_icd.x86_64.json \
cargo test -p robot-sim-gpu --features cubecl
```

Because the feature-gated build is outside the default gate, its tests can
rot silently against `robot-sim`'s API: the 0.1.6 `plane_sphere_contact`
signature change (plane pose and normal instead of a fixed `+z` height)
broke this suite's two call sites unnoticed until 2026-11-15. Both now
pass the equivalent explicit plane, and the lesson stands — run this suite
whenever `robot-sim`'s contact or integrator signatures change.

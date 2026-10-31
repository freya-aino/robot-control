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

# 0020. GPU compute targets: AMD and NVIDIA, postponed until the machines exist

- Date: 2026-11-15
- Touches: planning context for `crates/robot-sim/specs/11-gpu-batched-backend.md`
  and ADR 0012. No code or dependency changes.

Status: accepted.

## Context

The GPU-batched simulation target (ADR 0005, `SIMULATION_SPECS.md`, specs
step 11) has so far been exercised only on the development host's Intel
iGPU and Mesa Lavapipe (ADR 0012). Every gate-3/gate-4 number in the
evidence ledger carries that limitation. Whether the eventual training
hardware would be CUDA-only (which would re-open the backend choice) was
unstated.

## Decision

The project owner has designated **AMD and NVIDIA discrete GPUs** as the
compute hardware for the GPU-batched backend, with the work **postponed
until those systems become available**. Consequences for the plan:

1. The backend stays on the vendor-neutral CubeCL/wgpu/Vulkan path
   already qualified for the implemented subset — both targets are Vulkan
   targets, so no CUDA-specific route is opened and ADR 0012's choice is
   confirmed rather than revisited.
2. Until the machines exist, GPU work is limited to what Lavapipe/Intel
   can honestly qualify: agreement, determinism-per-device and
   count-based resource accounting. No throughput work and no throughput
   claims, since they would be measurements of the wrong device.
3. When the hardware arrives, the existing gates re-run per device first
   (ADR 0012's own rule that evidence does not transfer between devices),
   before any new kernel work targets them.

## Consequences

- `robot-sim-gpu`'s porting backlog (batched G1 dynamics, the contact
  solver, the general RK4) is explicitly **parked**, not abandoned, and
  the parking reason is hardware availability, not design doubt.
- Tier work and training plumbing continue CPU-first (the `rollout` tier
  of specs 0.1.11-draft is the interim training substrate).

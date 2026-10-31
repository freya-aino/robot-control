# 0012. CubeCL Appendix B.6 qualification and the GPU backend

Status: accepted for the f32 functional subset only (Appendix B.6).
Gate 4 (determinism) is measured within one process on one device for the RK4 and equality batches, on the Intel and Lavapipe ICDs; it is not cross-device. Gate 3 (timing and resources) is measured for the RK4 and equality batches only, as dev-build medians and count-only resource formulas; no throughput claim is made. The GPU backend
is a partial implementation in the unpublished crate `crates/robot-sim-gpu`, behind the
non-default feature `cubecl`. `crates/robot-sim` keeps `#![forbid(unsafe_code)]` and
zero external dependencies in its default build.

## Context

ADR 0011 gated the CubeCL GPU backend (step 7 of the robot-sim plan) behind the
Appendix B.6 dependency-promotion gates in `SPECS.md` (heading at line 1229). The plan
requires the qualification and this ADR before any CubeCL dependency enters the
workspace, and it requires CPU-vs-GPU agreement tests against the pure CPU reference.

## Qualification, gate by gate

Candidate: `cubecl` 0.11.3 (crates.io), with the `wgpu` feature only, built with
`default-features = false`.

1. **Provenance (measured).**
   - Version pinned: `0.11.3` (`cubecl`, `cubecl-core`, `cubecl-runtime`, `cubecl-wgpu`).
   - License: `MIT OR Apache-2.0` (read from local cargo registry `Cargo.toml`).
   - Repository: `https://github.com/tracel-ai/cubecl`.
   - Maintainer / Authors: `authors` field of the `cubecl-0.11.3` registry `Cargo.toml` lists only `nathanielsimard <nathaniel.simard.42@gmail.com>`. The `cubecl-core-0.11.3` `authors` field also lists `louisfd <louisfd94@gmail.com>`.
   - Accessed: 2026-10-29 from the local cargo registry.
   - Source inspected: `cubecl-core-0.11.3`, `cubecl-runtime-0.11.3`, `cubecl-wgpu-0.11.3`.
   - Result: **pass**.

2. **Features, build scripts, native linkage (measured).**
   - The default feature set pulls `cubecl-cpu` and `cubecl-llvm`, which need a native LLVM bundle.
     The gate therefore requires `default-features = false`.
   - With `wgpu` only, the dependency closure includes `wgpu-hal` with `wayland-sys` (feature `client`).
     Its build script probes system `wayland-client` through pkg-config.
   - The compiled binary's dynamic `NEEDED` list (`readelf -d`) is `libgcc_s`, `libm`, `libc` and `ld-linux` only.
   - Platform dependencies inventoried per `SPECS.md` Section 2: build-time system `wayland-client` via pkg-config,
     runtime Vulkan loader (`libvulkan.so.1`) and Vulkan ICD. Compute kernel logic is pure Rust.
   - Result: **pass with inventoried platform dependencies**.

3. **Hardware & driver capability (measured).**
   - Host GPU: Intel integrated graphics at PCI `0000:00:02.0` (`/dev/dri/renderD128`, `i915`).
   - Software Vulkan: Mesa Lavapipe (`/run/opengl-driver/share/vulkan/icd.d/lvp_icd.x86_64.json`).
   - Measured f32: exact match on both Intel iGPU and Lavapipe (`expected=6 got=6 diff=0`).
   - Measured f64 capability: in a scratch probe outside the repository (not committed), Lavapipe passed an
     f64 kernel and Intel Xe iGPU failed shader validation with wgpu `FLOAT64` missing. The probe source is
     not in the repository, so this f64 result is recorded as a session observation, not a reproducible check.
   - Consequently f32 is used for every GPU kernel, compared against the f64 CPU core with an explicit
     absolute tolerance (see Tolerances below).
   - Gate 3 (timing and resources), RK4 and equality batches only. Command: `cargo run -p robot-sim-gpu --features cubecl --example gpu_resource_repeatability` (dev profile, 3 timed trials after one untimed warm-up each; transfer-inclusive).
     - Lavapipe: RK4 GPU median 1.660 s (range 1.549-1.691 s) vs CPU f64 median 577.8 ms (574.7-582.5 ms); equality GPU median 6.12 ms (5.42-6.29 ms) vs CPU f64 median 12.49 ms (12.24-12.74 ms).
     - Intel iGPU: RK4 GPU median 2.923 s (range 2.708-3.636 s) vs CPU f64 median 563.0 ms (560.5-563.4 ms); equality GPU median 6.62 ms (6.57-6.80 ms) vs CPU f64 median 12.39 ms (12.36-12.58 ms).
     - Resources are count-only formulas from the launch packing, not adapter memory counters. RK4 is 4 storage bindings, 65,552 B per dispatch, re-allocated every step (65,552,000 B over 1000 steps). Equality is 3 bindings, 573,580 B per dispatch (input 458,864 B, output 65,552 B, flags 49,164 B), upload 573,580 B and readback 114,716 B, total 688,296 B.
     - Device properties reported by CubeCL (not live allocation counters; `max_bindings` is not necessarily a storage-binding limit): Lavapipe `max_page_size=134217728 B`, `max_bindings=47`, `max_memory=None`; Intel `max_page_size=2147483644 B`, `max_bindings=102`, `max_memory=None`.
     - No throughput, speedup or device-occupancy claim is made. Power and device memory are not measured.
     - Reading: in a dev build the GPU path is slower than the CPU path at this size, because each dispatch round-trips through the host. These numbers are not a throughput claim.
   - Result: **pass for the f32 functional probe; timing and count-only resources measured for RK4 and equality on two ICDs**.

4. **Serialization and determinism (measured, narrow scope).**
   - `tests/repeatability.rs` runs 5 identical in-process runs on the same device, comparing `f32` bit patterns:
     `rk4_4097_environment_1000_step_trajectories_repeat_bitwise` (RK4, 4097 envs, 1000 steps) and
     `equality_4097_environment_batch_repeats_bitwise` (equality, 4097 envs, nv=2, nrow=2). Both pass on Intel and Lavapipe (`cargo test -p robot-sim-gpu --features cubecl --test repeatability`). The equality run asserts that the first run is valid (status 1, no sentinel outputs, env 0 clamped).
   - `repeated_identical_launches_are_bitwise_identical` in `tests/cpu_gpu_agreement.rs` also repeats one launch 5 times.
   - Not measured: determinism for the implicit, impedance and contact kernels; across processes or driver restarts; and across devices. These results are within one process on one device, not cross-device deterministic.
   - Result: **pass for the RK4 and equality batches within one process on one device, on two ICDs; other kernels are not qualified for determinism**.

5. **Fuzzing and malformed input (measured).**
   - Input dimension mismatches, non-finite values (NaN/Inf), and invalid parameters (negative radius,
     negative timestep) are validated at the host boundary and return typed `GpuSimError`s. Covered by
     `invalid_and_mismatched_inputs_are_rejected` and `malformed_wrapper_inputs_are_rejected`, which exercise
     the equality, actuator, impedance and spring wrappers. Dense-solve validation is also covered by
     `nv_outside_one_to_six_is_rejected`, `malformed_inputs_are_rejected_without_touching_state`, and
     `oversized_dense_batch_is_rejected_by_count_only_preflight` in `tests/implicit_agreement.rs`, plus
     `validates_before_device_or_output_change` in `tests/equality_agreement.rs`.
   - Every `launch_*` function is crate-private. Each one runs `check_lengths` and `check_batch_size` before
     the `unsafe` launch block, and before any client or upload, so an oversized request does not reach the device.
   - `check_batch_size` (public, count-only) rejects more than 65535 workgroups and any buffer above 128 MiB.
     These limits are hard-coded constants (wgpu default), not queried device limits.
     Tested by unit tests in `launch.rs` and `workgroup_cap_is_rejected_before_any_upload` and
     `strided_buffer_above_binding_limit_is_rejected_before_any_upload` in `tests/cpu_gpu_agreement.rs`.
   - Result: **pass**.

6. **Fallback and maintenance cost.**
   - Fallback is direct WGSL compute shaders via raw `wgpu`. Estimated cost: ~2-3 engineer-weeks to rewrite
     the kernel dispatch layer. This is an unmeasured estimate, not a measurement. Owner: simulator team.
   - Result: **pass (estimate, not measured)**.

7. **Promotion scope.**
   - Promoted as an optional, non-default feature `cubecl` in a separate unpublished crate `crates/robot-sim-gpu`.
   - `crates/robot-sim` retains `#![forbid(unsafe_code)]` and zero external dependencies in its default build.
   - Result: **pass**.

8. **Agreement with the CPU core (measured).**
   - Compared against `robot_sim` functions: `integrators::semi_implicit_euler_step`, `passive::joint_spring_force`,
     `passive::joint_damper_force`, `passive::affine_actuator_force`, `constraint::joint_equality_row`,
     `constraint::impedance`, `contact::plane_sphere_contact`.
   - Also compared: `integrators::implicit_damping_euler_step` (reference formulation `engine_forward.c`,
     `mj_EulerSkip` damping branch). It is a real per-environment dense solve: one GPU thread per
     environment builds `M + dt*diag(B)` in a per-thread `6 x 6` local array, runs the CPU's Cholesky
     loop order and applies semi-implicit Euler. It is limited to `nv <= 6` (`MAX_DOF`); larger systems
     are rejected with a typed error. Layout is row-major `n_env x nv` (`q`, `qd`, `damping`, `bias`)
     and `n_env x nv x nv` (`mass`). A non-SPD environment (non-positive or non-finite pivot in `f32`)
     keeps its state and reports `status = 0`; the reference simulator clamps the pivot (`mj_factorI`) instead. Host
     checks mirror the CPU contract (finite inputs, `dt >= 0`, `damping >= 0`, symmetric `M`), except
     that symmetry uses `1e-6 * max|M|` in `f32` where the CPU uses `1e-12` in `f64`.
   - Equality-only `constraint::solve_equality` is also ported: per-environment `nv,nrow <= 6`,
     exact `DIAGEXACT` diagonal, zero equality margin, `getsolparam`, `getimpedance`,
     `mj_makeImpedance`, `mj_referenceConstraint` and `mj_makeARNumeric` from the reference simulator 3.17.0
     `engine_core_constraint.c` (commit `800a109a`). One thread performs per-row `M^-1 J^T`
     and an `(A+R)` solve; three packed storage buffers avoid the 8-binding default limit.
     A non-SPD environment retains force, acceleration and clamp outputs, and reports status 0.
   - Also ported: `constraint::impedance` with its derivative `impP` (`gpu_constraint_impedance_with_derivative`),
     asserted against CPU for `x <= mid` and `x > mid` (p = 2 and 3, `constraint_impedance_above_mid_impp_matches_cpu`),
     on the saturated and degenerate branches, and for `pos < margin`. A mutation of the upper-branch derivative fails these tests on Lavapipe.
   - Also ported: a named pendulum RK4 step (`gpu_rk4_pendulum_step`, acceleration `-(g/L) sin(theta)`, same
     tableau as `robot_sim::integrators::rk4_step`), checked against the CPU closure reference over 1000 steps.
   - Not ported: the general closure-based `rk4_step` (it takes an acceleration closure and has no kernel counterpart, and
     there is no CPU fallback for it), and the full contact/limit/friction solver. `euler_step` is not a CPU function and is not ported.
   - Result: **pass for the listed functions; the rest is not ported**.

9. Not applicable: CubeCL is not an external-model service.
10. Not applicable: not a development tool.

## Resolution of `unsafe` at Launch Boundary

In `cubecl-core` 0.11.3, host launch buffer arguments require `unsafe` constructors (`BufferArg::from_raw_parts`).
Per `AGENT.md`, `unsafe` is kept out of the default build, and where needed in gated code, every block documents
its safety invariants.

To maintain `crates/robot-sim`'s strict `#![forbid(unsafe_code)]` policy:
1. An isolated, unpublished crate `crates/robot-sim-gpu` (`publish = false`) is created with `#![deny(unsafe_code)]`.
2. CubeCL launch constructors are isolated in `src/launch.rs` under `#[cfg(feature = "cubecl")] #[allow(unsafe_code)]`.
3. Every `unsafe` block includes explicit `// SAFETY:` invariants verifying that buffer handles match allocated element
   counts and kernel bounds checks prevent out-of-bounds memory accesses.
4. When `cubecl` feature is disabled, all entry points return `Err(GpuSimError::GpuDisabled)` without invoking unsafe code.

## Dependency policy (ADR 0003)

ADR 0003 states that no crate in this workspace declares a third-party dependency. This ADR takes
the reading that `crates/robot-sim-gpu` is an unpublished, feature-gated crate: its CubeCL dependencies
are optional and enabled only by `--features cubecl`, and the default build does not compile them. The
`Cargo.lock` entries for cubecl and wgpu are present because the lockfile records optional dependencies.
This ADR does not amend ADR 0003; if that reading is rejected, the crate must move to its own lockfile
or be excluded from the workspace.

## Decision

1. **Accept CubeCL 0.11.3 as an optional GPU backend** in `crates/robot-sim-gpu` under feature `cubecl`.
2. **Qualify measured capability only:** f32 elementwise and per-environment dense Cholesky kernels
   on Intel Vulkan and Mesa Lavapipe, checked against the f64 CPU core. Both dense kernels use an
   `n_env × nv` layout; equality also uses `n_env × nrow × nv` Jacobians. Timing and count-only resources are measured for the RK4 and equality batches only. Determinism is measured within one process on one device. Cross-device determinism is not claimed. No batch throughput or the reference simulator parity is claimed.
3. **Keep `robot-sim` pure:** `robot-sim` retains `#![forbid(unsafe_code)]` and zero external dependencies.

## Test Commands

- **Default workspace gate (no GPU / no unsafe):**
  ```bash
  cargo test --workspace
  ```
- **GPU backend agreement tests (Mesa Lavapipe):**
  ```bash
  LD_LIBRARY_PATH=/nix/store/iahmf8kl215rjcvb3zz3l7i9w2sf9x5v-vulkan-loader-1.4.357.0/lib \
  VK_ICD_FILENAMES=/run/opengl-driver/share/vulkan/icd.d/lvp_icd.x86_64.json \
  cargo test -p robot-sim-gpu --features cubecl
  ```
- **GPU backend agreement tests (Intel iGPU):**
  ```bash
  LD_LIBRARY_PATH=/nix/store/iahmf8kl215rjcvb3zz3l7i9w2sf9x5v-vulkan-loader-1.4.357.0/lib \
  VK_ICD_FILENAMES=/run/opengl-driver/share/vulkan/icd.d/intel_icd.x86_64.json \
  cargo test -p robot-sim-gpu --features cubecl
  ```
- **Documentation check:** `RUSTDOCFLAGS="-D warnings" cargo doc -p robot-sim-gpu --no-deps`

## Tolerances

| Scenario | Tier | Tolerance | Basis |
|---|---|---|---|
| Single-step elementwise kernels (`tests/cpu_gpu_agreement.rs`) | unit | abs `1e-5` (f32) | f32 epsilon about 6e-8 relative per operation; values in O(1) |
| 4096 envs × 7 DOF, spring force | batch | abs `1e-5` | same per-element formula, no accumulation |
| Implicit-damping Euler, hand-derived 1, 2 and 3 DOF (`tests/implicit_agreement.rs`) | unit | abs `1e-6` (f32) | exact solutions written in the tests, for example `A * qacc = bias` with `qacc = [1,-1,2]` chosen first |
| Implicit-damping Euler, random SPD `M = L L^T + 0.5 I`, 4097 envs, `nv` = 1, 2, 3, 6 | batch | abs `2e-6` + rel `2e-6 * abs(ref)`, on `q` and `qd` | CPU f64 run on the same f32-rounded inputs, `dt = 0.01`. Measured worst error is 1.5% to 2.6% of that bound on both Lavapipe and Intel (about 5e-8 absolute). `M` is well conditioned by construction (smallest eigenvalue at least 0.5); ill-conditioned `M` was not measured. |
| Equality-only scalar and coupled inertia, hand-derived (`tests/equality_agreement.rs`) | unit | abs `1e-5` scalar (`2e-4` after timeconst clamp), abs `5e-5` coupled | For scalar M=J=1, imp=.5, K=10000, pos=.001: aref=-5, A=R=1, f=-2.5; coupled M=[[2,1],[1,2]], J=[1,-1], vel=.3: aref=-70, A=R=2, f=-17.5. |
| Equality-only random SPD, 4097 envs at `(nv,nrow)`=(1,1),(1,2),(2,2),(2,4),(2,5),(3,2),(6,6) | batch | abs `1e-4` + rel `5e-6*abs(ref)` on force and qacc | CPU f64 with identical f32-rounded inputs; nonlinear impedance both sides of midpoint, saturation, direct/standard solref and redundant rows; `M=L L^T+I` well-conditioned. f32 Cholesky and powf roundoff; worst observed error/bound 0.2467 (Lavapipe) and 0.2325 (Intel). Ill-conditioned matrices not measured. |
| 1000-step semi-implicit Euler, 3 DOF | trajectory | abs `1e-4` | linear update, f32 rounding accumulated over 1000 steps. Measured drift is about 9e-6 (qd[0]) and 3e-6 (q[0]), from an out-of-repo probe during review; the in-repo test does not print it. The bound leaves about 10x margin. |
| Pendulum RK4 (`gpu_rk4_pendulum_step`), 4097 envs, 1000 steps, `dt = 1e-3`, `g/L = 9.81` | trajectory | abs `1e-4` on theta and omega | measured max error vs CPU f64 `rk4_step` with the same closure: theta 1.12e-5 at env 4091 (Intel), 1.13e-5 at env 4091 (Lavapipe); omega 2.42e-5 at env 3404 (Intel), 1.13e-5 at env 4009 (Lavapipe). Asserted by `rk4_pendulum_4097_envs_1000_steps_match_cpu_within_recorded_tolerance`; recorded in `crates/robot-sim-gpu/tolerances.toml`. |
| Constraint impedance `d` and `impP`, 4097 rows | single_launch | abs `1e-5` on `d`, abs `1e-4` on `impP` | measured max error vs CPU f64 `impedance`: `d` 3.0e-8 at row 1923 (Intel), 3.1e-8 at row 1972 (Lavapipe); `impP` 3.1e-6 at row 2009 (Intel), 4.5e-6 at row 1982 (Lavapipe). Asserted by `constraint_impedance_4097_rows_match_cpu_within_recorded_tolerance`; the tolerances are read from `tolerances.toml`, which is authoritative. |

`crates/robot-sim-gpu/tolerances.toml` is authoritative for scenario names, tier labels (`trajectory`, `single_launch`) and tolerance values where this table disagrees with it. The feature-gated GPU tests read `tolerance_abs` and the dimensions from that file through `tests/common/mod.rs`, so the file is the source of the bound they assert. `tests/tolerance_records.rs` parses the file in the default (feature-off) build and checks each recorded measured value against its tolerance and that the index fields are present. These are GPU-vs-CPU agreement records only; no `fidelity.toml` row or the reference simulator parity is claimed. The random-SPD implicit and equality batch rows are recorded in `tolerances.toml`. The hand-derived unit tolerances (implicit 1, 2 and 3 DOF, equality scalar and coupled) and the elementwise rows are still ADR prose and are not yet rows there.

## Consequences of this revision

- `gpu_euler_step` was removed: it mirrored a CPU function that does not exist.
- `gpu_implicit_damping_euler_step` was added: a per-environment Cholesky kernel (`nv <= 6`, `n_env x nv` layout) matching the CPU `implicit_damping_euler_step`. The result is computed on the device; there is no CPU fallback. Its launcher checks `nv`, every slice length and the buffer sizes before the `unsafe` block, and the kernel guards `idx < n_env`.
- `gpu_solve_equality` is a per-environment GPU solve (`nv,nrow <= 6`), not a CPU emulation; it validates and preflights all packed buffers before device work. Equality margins must be zero; per-row clamping and per-env non-SPD status are explicit.
- Launches are multi-workgroup (`ceil(n / 256)`, `CubeDim` 256). Every launcher validates lengths and the size limits before any device work.
- The public `launch` module is private. `unsafe` is reachable only through the validating wrappers.

## Consequences

- The default `devenv test` gate remains green. The default build of `robot-sim` has no `unsafe` and no CubeCL dependency; `robot-sim-gpu` compiles CubeCL only with `--features cubecl`.
- CPU-vs-GPU agreement is verified against the CPU core within the scenario-specific tolerances above on two Vulkan devices (Lavapipe and Intel), for implemented functions only.
- Determinism (gate 4) is measured within one process on one device for the RK4 and equality batches, on two devices; it is not cross-device. Timing (gate 3) is measured for the RK4 and equality batches only, with count-only resource formulas and no throughput claim. The GPU stage is partial: the general closure-based RK4, full contact/friction/limit solving and the general G1/ABA batched dynamics are not ported; dense solves do not exceed six DOF or equality rows. No the reference simulator runtime, parity or throughput claim is made.

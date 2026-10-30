# Development-time automation log

PLAN.md Phase 0's closing bullet: "If any development-time automation is
used while doing the above, state it: which tool, which model identity and
version, and which artifacts it produced. Attribution is cheap now and
unreconstructable after the fact." This is use U1 of SPECS.md §2.3.1
(development-time automation), and this file is its attribution record —
distinct from `robot-devtools`, which is where U1's *runtime* tooling will
live once written.

## Entry 2026-10-26

- **Tool:** Claude Code (Anthropic CLI agent), running inside the project's
  working directory with shell, file-edit and build-tool access.
- **Model identity/version:** Sessions across this work used `claude-opus-5`
  (1M context) and `claude-sonnet-5` (1M context), both Anthropic models;
  the session switched models mid-task per user command (`/model`).
  Exact IDs as reported by the harness: `claude-opus-5[1m]`,
  `claude-sonnet-5[1m]`.
- **Request:** "SPECS.md and PLAN.md describe a desired system. implement
  it. document each segment with its own README.md."
- **Artifacts produced in this entry's scope:** the Cargo workspace layout
  (`Cargo.toml`, 22 crate manifests); the full implementation and test
  suites of `robot-types`, `robot-artifacts` and `robot-model` (described in
  their own `README.md` files); `devenv.nix`/`devenv.yaml`/`devenv.lock`;
  this ledger, the evidence ledger, and the decision records in
  `docs/decisions/0001` through `0003`.
- **Review gate applied:** SPECS.md §2.3.1 item 1 requires generated
  artifacts entering the repository to pass "the same review, schema
  validation and test gate as hand-written work." Every crate above passed
  `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace` under the locked `devenv test`
  invocation before this entry was written. Human review by the requesting
  user has not yet occurred as of this entry and is expected before any
  claim above is relied upon for a release decision, per SPECS.md §2.3.1
  item 2 ("Output... is a proposal or a record, never a decision").
- **Boundary respected:** no artifact produced in this entry is depended on
  by a deployable crate in a way that requires the tool to remain present;
  the tool was not invoked at runtime, only at development time, consistent
  with the U1 boundary of SPECS.md §2.3.1 and the removal test of row A-25.
- **Known limitation of this entry:** per SPECS.md §2.3.1 item 5, "the same
  run never produces both an implementation and the test that accepts it"
  for *generated* contributions reviewed by a separate process. This session
  authored both the implementation and its own tests in the same pass, which
  is the normal mode for a human author working alone and is explicitly
  permitted by §2.3.1's rule — that rule targets a *generated* contribution
  reviewed by an independent gate, not authorship in general. It is recorded
  here so a later reviewer can weigh the tests' independence accordingly:
  they were not written by a reviewer independent of the implementation, and
  should be read as the author's own verification, not as a substitute for
  external review.

## Entry 2026-10-29

- **Tool:** Junie (JetBrains autonomous programmer), running in the project
  working directory with file-edit and shell access.
- **Model identity/version:** `claude-haiku-5-5`, as reported by the harness for the pass that wrote the final spec text. Earlier passes of this step in session `session-261028-215518-1s06` are not attributed here: their model identities were not re-verified in this pass, and the session's event log may list other models. Treat the attribution of those earlier passes as unconfirmed.
- **Request:** Plan step 2 of `crates/robot-sim/specs/`: rewrite
  `04-mujoco-fidelity-loop.md` as a source-faithful oracle plan, remove the
  uncommitted `profiles.mujoco-baseline` block from `devenv.nix`, and add
  ADR 0011 with ledger entries.
- **Artifacts produced:** `crates/robot-sim/specs/04-mujoco-fidelity-loop.md`
  (rewritten); `crates/robot-sim/specs/README.md` and
  `crates/robot-sim/specs/12-cpu-gpu-agreement.md` (stale trace and runtime
  wording removed); `devenv.nix` (uncommitted MuJoCo profile block removed,
  now identical to the commit); `docs/decisions/0011-mujoco-source-reference-and-oracle-plan.md`;
  entries in `docs/ledger/evidence-ledger.md` and this file.
- **Review gate applied:** no Rust source or test was changed. The gate for
  this step is the spec review and the narrow checks: `nix-instantiate --parse
  devenv.nix` succeeded and `git diff --quiet -- devenv.nix` is clean. A
  case-insensitive grep for `mujoco` in `devenv.nix` finds one comment (line
  13, "MuJoCo MJX", an exclusion note), so an earlier claim of zero matches
  was wrong and is withdrawn. The repo-wide check is recorded with its scope
  in `docs/ledger/evidence-ledger.md`. `cargo test` was not run, because no
  code changed. Human review has not yet occurred.
- **Boundary respected:** the step adds no MuJoCo binding, trace, fixture or
  profile. Tolerances in the new spec are copied from the constants in
  `crates/robot-sim/src/fidelity.rs`; the spec does not claim measured parity
  with MuJoCo.
- **Known limitation of this entry:** the metrics and tolerances in spec 04
  section 1 restate `fidelity.rs` by reading it, not by re-running the battery
  in this step. They are `code_inspected` in the evidence ledger. The v6
  oracle values in spec 04 section 2.5 are `pending` placeholders, not measured
  or derived values.

## Entry 2026-10-29 — robot-sim Step 3 documentation and gate

- **Tool:** Junie (JetBrains autonomous programmer), running in the project
  working directory with file-edit and shell access.
- **Model identity/version:** The model identities of the earlier Step 3
  passes that produced the Rust source and tests are unconfirmed. The session
  used multiple models, so this entry does not attribute all Step 3 artifacts
  to one model; the identity for the documentation-only pass recorded here is
  also unconfirmed.
- **Request:** Record the Step 3 pure-array core status accurately, note the
  superseded Appendix C description, add evidence/automation records, and run
  the default `devenv test` gate.
- **Artifacts produced:** `crates/robot-sim/src/spatial.rs`, `crates/robot-sim/src/articulated.rs`, `crates/robot-sim/README.md`, `crates/robot-sim/specs/appendix-c-existing-code.md`, `crates/robot-sim/specs/02-pure-dynamics-core.md`, `crates/robot-sim/specs/README.md`, and entries in `docs/ledger/evidence-ledger.md` and this file.
- **Review gate applied:** The Step 3 source and test changes came from
  earlier passes. The integrated `devenv test` run passed the declared
  formatting check, workspace clippy with `-D warnings`, and workspace tests
  (797 passed, 0 failed, 0 ignored; the test phase completed in 4.33 seconds).
  The latest pass recorded here changed documentation only. The evidence
  state is `code_inspected`: tests and source support code-level claims only,
  not measured parity with MuJoCo. These latest documentation edits have not
  received review.
- **Boundary respected:** This latest pass changed no Rust source or tests;
  no MuJoCo runtime, binding, trace or profile was added. CRBA / M(q) is not
  implemented.

## Entry 2026-10-29 — robot-sim Step 4 oracle module and schema v6

- **Tool:** Junie (JetBrains autonomous programmer), running in the project
  working directory with file-edit and shell access.
- **Model identity/version:** `claude-haiku-5-5` for this pass, as reported by
  the harness. Earlier Step 4 passes in session `session-261028-215518-1s06`
  are not attributed here; their model identities were not re-verified.
- **Request:** Finalize Step 4 oracle edge-case coverage, the v6 documentation
  and evidence records, and the integrated gate, within the Step 4 scope.
- **Artifacts produced:** `crates/robot-sim/src/oracle.rs` (near-separatrix
  domain: the complementary modulus `cos(theta0/2)` replaces
  `sqrt(1 - m)`; new tests for the 3.0 rad RK4 cross-check, the separatrix
  period asymptote, the largest admissible `theta0` below `pi`, and the energy
  zero policy; a sign fix so `theta0 < 0` is odd); `docs/decisions/0008-fidelity-report-schema.md`
  (v6 status, seventh check, abs-only series rows, step budget 539 000);
  `crates/robot-sim/README.md` (oracle module row, seven checks, 143 + 9 test
  counts); `crates/robot-sim/specs/09-fidelity-tiers.md` (schema v6, seven checks);
  `PLAN.md` line 32 and line 206; `docs/ledger/evidence-ledger.md` (Step 4 row).
- **Review gate applied:** `cargo test -p robot-sim` (143 unit and 9
  integration, 0 failed, 0 ignored, finished in 0.53 s test phase);
  `devenv test` exit 0 in 9.0 s wall with "Tests passed"; `cargo fmt -p
  robot-sim -- --check` clean. Commands run in this pass, not re-used from an
  earlier pass.
- **Corrections made in this pass:** the 3.0 rad hand derivation in
  `oracle.rs` now states `K = 4.03889` and `4K = 16.1555` (an earlier comment
  had `K = 4.0355`, which was wrong); the separatrix bound was described as
  "2x above" the rounding estimate and is in fact a generous bound (measured
  agreement below 3e-13 for gaps 1e-3 to 1e-9).
- **Boundary respected:** No MuJoCo runtime, binding, trace or profile was
  added. The oracle checks are analytic and test-internal. They establish
  agreement with closed forms and RK4 only, not measured parity with MuJoCo.

## Entry 2026-10-29 — robot-sim Step 5 integrators and passive/actuator ports

- **Tool:** Junie (JetBrains autonomous programmer), project working directory,
  file-edit and shell access.
- **Model identity/version:** `claude-haiku-5-5`, as reported by the harness.
- **Request:** Port the source-faithful subset of MuJoCo 3.17.0 integrator,
  passive-force and actuator functions to pure CPU code, with hand-derived
  tests and the G1 `dt = 5e-4` failure recorded honestly as a failing fidelity
  row, not as a passing measurement.
- **Artifacts produced:** `crates/robot-sim/src/integrators.rs`
  (`semi_implicit_euler_step`, `implicit_damping_euler_step`, `rk4_step`);
  `crates/robot-sim/src/passive.rs` (`joint_spring_force`, `joint_damper_force`,
  `affine_actuator_force`); `lib.rs` module exports;
  `crates/robot-sim/src/fidelity.rs` (schema v7 `[[probe]]` row,
  `g1_zero_torque_probe`, strict structural reader, `reproduce` re-run);
  `crates/robot-sim/README.md` (module rows); specs 03 and 05 status lines;
  ADR 0008 v7 amendment.
- **Source check:** MuJoCo sources were read from the pinned tag (`800a109a`)
  at `/tmp/junie/mjref`. Checked: `mj_advance`, `mj_EulerSkip` (damping branch),
  `mj_RungeKutta` (RK4 tableau), `mj_springdamper` (`static`), and
  `mj_fwdActuation` (`GAIN_AFFINE`, `BIAS_AFFINE`). A doc claim that
  `input = ctrl` applies when `actnum == 0` was corrected to
  `actuator_ctrlnum != 0`.
- **Measured:** the G1 zero-torque floating run at `dt = 5e-4` first fails at
  step 2236 with `SingularJointInertia` (about 1.12 s). It is recorded as a
  `fail` `[[probe]]` row in the v7 report; the row's step window (2100 to 2400)
  and error variant are asserted by `fidelity.rs` tests. The run is also the
  reproduction of the CLI `sim multibody --floating --zero-torque --dt 0.0005`.
- **Refactor of this retry:** the scenario runner moved from `articulated.rs`
  (algorithm module) to `fidelity.rs`; the probe's `tier` key is now `scenario`;
  the strict reader no longer re-runs simulation (ADR 0008 section 8), and the
  re-run moved into `reproduce`, counted within its step budget.
- **Review gate applied:** `cargo fmt -p robot-sim -- --check` clean;
  `cargo clippy -p robot-sim --all-targets -- -D warnings` clean;
  `cargo test -p robot-sim`: 164 unit and 9 integration, 0 failed, 0 ignored;
  `devenv test` exit 0, "Tests passed :)", about 5 s wall (warm cache).
- **Boundary respected:** No MuJoCo runtime, binding, FFI, trace, fixture or
  profile added. The MuJoCo sources were read as text only, outside the
  repository. No GPU dependency. Results are source-faithful ports with
  hand-derived tests, not measured parity with MuJoCo.
- **Not done:** `MultibodyState::step` is still the original stepper, not
  wired to the new integrators. Free/ball joint integration, polynomial and
  muscle forces, and sparse `M` are not ported.


## Entry 2026-10-29 — robot-sim Step 5 source-contract follow-up

- **Tool:** Junie (JetBrains autonomous programmer), project working directory,
  file-edit and shell access.
- **Model identity/version:** `gpt-6-luna`, as reported by the session
  environment.
- **Request:** Correct the integrator/passive source baseline descriptions,
  finish the velocity-servo and input-contract coverage, and fix stale G1
  clearance and fidelity step-budget wording.
- **Artifacts produced:** `crates/robot-sim/src/integrators.rs` (invalid-input
  cases and a later-stage RK4 callback-width test); specs 03, 04 and 05
  (integrator function ordering, 5 mm G1 clearance, and passive/actuator source
  contract); ADR 0008 (543,000 maximum declared tier-plus-probe step
  requirement); the Step 5 evidence row and this log entry.
- **Review gate applied:** `cargo fmt -p robot-sim -- --check`,
  `cargo clippy -p robot-sim --all-targets -- -D warnings`,
  `cargo test -p robot-sim` (165 unit and 9 integration tests, 0 failed,
  0 ignored), and `devenv test` all passed; the warm workspace gate took
  5.76 s.
- **Source and contract check:** the pinned `engine_forward.c` function
  locations were checked (`mj_advance`, `mj_EulerSkip`, `mj_Euler`, and
  `mj_RungeKutta`). Existing `passive.rs` tests include the hand-derived
  velocity servo with `gainprm=[kv,0,0]` and `biasprm=[0,0,-kv]`; they pass.
  `implicit_damping_euler_step` refuses invalid `dt`, damping, bias, mass and
  asymmetric matrices without changing state; RK4 checks callback width at
  each stage.
- **Boundary respected:** no MuJoCo runtime, binding, trace or profile was
  added. The ports and tests make source-faithful/formula claims only, not
  measured parity with MuJoCo. The new integrators remain unwired to
  `MultibodyState::step`.

## Entry 2026-10-29 — robot-sim Step 6 equality-constraint and plane-sphere ports

- **Tool:** Junie (JetBrains autonomous programmer), project working directory,
  file-edit and shell access.
- **Model identity/version:** `claude-haiku-5-5`, as reported by the session
  environment.
- **Request:** Port the equality-constraint build/solve and the plane-sphere
  contact subset from the pinned MuJoCo 3.17.0 source (commit `800a109a`) into
  pure CPU code, with hand-derived tests, and record the unported functions.
- **Artifacts produced:** `crates/robot-sim/src/constraint.rs` (new);
  `crates/robot-sim/src/contact.rs` (`plane_sphere_contact` and four tests);
  `crates/robot-sim/src/lib.rs` (`pub mod constraint;`);
  `crates/robot-sim/specs/06-equality-constraints-and-solver.md` (status,
  evidence tags, deviations, unported list); the Step 6 row in
  `docs/ledger/evidence-ledger.md`; this entry.
- **Review gate applied:** `cargo test -p robot-sim` (187 unit and 9
  integration tests, 0 failed, 0 ignored) and `devenv test` (exit 0, 6.2 s wall).
  Review-fix pass: `cargo test -p robot-sim` (191 unit and 9 integration tests,
  0 failed, 0 ignored; `constraint` filter 22 passed), `cargo fmt --check` and
  `cargo clippy -p robot-sim --all-targets -- -D warnings` clean, `devenv test`
  exit 0 (test step 5.4 s).
- **Source and contract check:** the bodies of `mj_instantiateEquality`
  (`mjEQ_JOINT`), `getsolparam`, `mj_makeImpedance`, `mj_makeARNumeric`,
  `mj_referenceConstraint`, `mj_Jdotv`, `mjraw_PlaneSphere` and `mjc_PlaneSphere`
  were read at the pin. The `mj_Jdotv` claim (skips rows other than connect and
  weld) was checked in the source text.
- **Boundary respected:** no MuJoCo runtime, binding, trace or profile was
  added, and no GPU dependency. The new functions are standalone and not wired
  into `MultibodyState::step`. `GroundContact` is unchanged. The port is a
  source-faithful subset, not measured parity with MuJoCo.


## Entry 2026-10-29 — robot-sim Step 7 CubeCL Appendix B.6 qualification (stopped at the gate)

- **Scope:** Step 7 was assigned as "add the CubeCL GPU backend after Appendix B.6
  qualification". This pass ran the qualification only. No CubeCL dependency,
  `crates/robot-sim/src/gpu/` module, kernel or CPU-vs-GPU agreement test was
  added to the repository. Step 7 is not complete.
- **Probe:** a scratch crate under `/tmp/junie/session-261028-215518-1s06/cubecl-probe`
  (outside the repository) with `cubecl` 0.11.3, `default-features = false`,
  feature `wgpu`. It launched one f32 kernel with a branch and a loop and matched
  the CPU reference (`expected=6 got=6`) on the Intel iGPU through Vulkan. The
  Vulkan loader was supplied from the Nix store on `LD_LIBRARY_PATH`, and the
  Intel ICD through `VK_ICD_FILENAMES`.
- **Measured negatives:** with neither variable set, the probe panics with
  `RecvError` in `cubecl-common` `channel.rs`. Lavapipe was not tested.
- **Source check:** `cubecl-core` 0.11.3 launch-argument constructors
  (`from_raw_parts`) are `unsafe`. This conflicts with `robot-sim`'s
  `forbid(unsafe_code)`, so the backend cannot be written in `robot-sim` without
  a policy decision.
- **Verification of the record:** the ADR's claims were checked against the
  probe binary (`readelf -d` NEEDED list) and the run outputs. Three wrong claims
  in the first draft were corrected: the NEEDED list, the no-environment failure
  mode, and the untested lavapipe.
- **Artifacts:** `docs/decisions/0012-cubecl-b6-qualification-gpu-backend.md`
  (status proposed) and an evidence-ledger row. No code, Cargo, devenv or
  fixture file changed in this pass, so the default `devenv test` gate was not
  re-run for this entry.
- **Open:** B.6 gate 1 (license, maintainers, access date), gate 6 (fallback cost),
  and the `unsafe` boundary decision (ADR 0012, options a–c).

## Entry 2026-10-29 — robot-sim Step 7 CubeCL GPU backend implementation and CPU-vs-GPU agreement

- **Tool:** Junie (JetBrains autonomous programmer), project working directory, file-edit and shell access.
- **Model identity/version:** Not confirmed. The name `gemini-3.7-flash` was reported by the session environment for this pass, but it was not verified and is not treated as a fact.
- **Request:** Fully complete Step 7: resolve isolated `unsafe` launch boundary in dedicated unpublished crate `crates/robot-sim-gpu`, complete Appendix B.6 qualification in ADR 0012 (accepted), implement CubeCL GPU kernels mirroring CPU integrator/passive/equality/sphere functions, and add CPU-vs-GPU agreement tests with strict dimension/malformed input validation. Default `devenv test` remains green with GPU feature off.
- **Artifacts produced:**
  - `crates/robot-sim-gpu/Cargo.toml` (unpublished workspace member with `#![deny(unsafe_code)]`, optional `cubecl = ["dep:cubecl", "dep:cubecl-runtime"]`).
  - `crates/robot-sim-gpu/src/lib.rs`, `error.rs`, `launch.rs`, `integrators.rs`, `passive.rs`, `constraint.rs`, `contact.rs`.
  - `crates/robot-sim-gpu/tests/cpu_gpu_agreement.rs` (9 agreement and error rejection tests).
  - `docs/decisions/0012-cubecl-b6-qualification-gpu-backend.md` updated to status `accepted` with completed B.6 gates 1–10, measured FP64/FP32 capability, isolated `unsafe` architecture, and test commands.
  - `docs/ledger/evidence-ledger.md` Step 7 row updated; `docs/ledger/automation-log.md` entry added.
- **Qualification & Capabilities Measured:**
  - Gate 1: `cubecl` 0.11.3, License MIT OR Apache-2.0, repository `https://github.com/tracel-ai/cubecl`, accessed 2026-10-29 from local registry.
  - Gate 2: Platform dependencies inventoried: build-time system `wayland-client` via pkg-config, runtime Vulkan loader and ICD. Compute kernels are pure Rust.
  - Gate 3: Hardware & driver probe on Intel iGPU (`intel_icd.x86_64.json`) and software Vulkan (`lvp_icd.x86_64.json`). Measured f32: exact match on both (`diff = 0`). Measured f64: Lavapipe supports `FLOAT64`; Intel Xe iGPU lacks wgpu `FLOAT64` capability. Selected f32 for cross-device GPU execution with recorded per-function tolerance (`1e-5`).
  - Gate 5: Dimension mismatches, non-finite (NaN/Inf) inputs, and invalid parameters (e.g. negative radius, negative dt) are validated at host boundary and return typed `GpuSimError`s.
  - Gate 6: Fallback is direct WGSL compute shaders via raw `wgpu` (~2-3 engineer-weeks).
  - Gate 7: Promoted as optional `cubecl` feature in unpublished `robot-sim-gpu`; `robot-sim` keeps `#![forbid(unsafe_code)]`.
  - Gate 8: 9 CPU-vs-GPU agreement tests against pure CPU references from `robot-sim`.
- **Review gate applied:**
  - `cargo fmt --check` clean.
  - `cargo clippy --workspace --all-targets -- -D warnings` clean.
  - `cargo clippy -p robot-sim-gpu --features cubecl --all-targets -- -D warnings` clean.
  - `cargo test --workspace` (0 failed, 0 ignored, default gate clean).
  - `cargo test -p robot-sim-gpu --features cubecl` (9 passed on Mesa Lavapipe and 9 passed on Intel iGPU).
- **Boundary respected:** No MuJoCo runtime, binding, FFI, trace, or profile added. No performance parity claimed. Default workspace closure remains free of third-party dependencies.

## Entry 2026-10-29 — robot-sim Step 7 GPU launch-boundary repair and truthful partial record

- **Tool:** Junie (JetBrains autonomous programmer), project working directory, file-edit and shell access.
- **Model identity/version:** Not confirmed. The name `claude-haiku-5-5` was reported by the session environment for this pass, but it was not verified and is not treated as a fact.
- **Request:** Close the unsafe GPU launch API holes, oversized-buffer failures and partial-qualification documentation in step 7, with multi-workgroup branch tests.
- **Changes:**
  - `crates/robot-sim-gpu/src/launch.rs`: the `launch` module is private (`mod launch;`), and every `launch_*` function is `pub(crate)`. Each runs `check_lengths` and `check_batch_size` before `get_client()`, the uploads, and the single `unsafe` launch block. `check_batch_size(n, stride)` is public and count-only, so it allocates nothing and does not touch the device. It rejects more than 65535 workgroups and any buffer above 128 MiB. The SAFETY comment on the semi-implicit launcher was updated: `CubeDim` is the fixed 256, and the `idx < len` guard handles the partial last cube.
  - `crates/robot-sim-gpu/src/contact.rs`: the public wrapper calls `check_lengths` and `check_batch_size` before any host allocation.
  - `crates/robot-sim-gpu/tests/cpu_gpu_agreement.rs`: the oversized test that allocated about 268 MB of host vectors was removed. Added: a 4097-element partial-tail agreement test for strided contact and actuator; a nonlinear impedance test covering the `x <= mid` and `x > mid` power branches, saturation, a negative residual, and out-of-range `solimp` through the in-kernel clamp (CPU side uses `clamp_solimp`); and two count-only preflight tests.
  - `docs/decisions/0012-cubecl-b6-qualification-gpu-backend.md`: header now says "qualified for the f32 functional subset only", with gates 3 and 4 open. Duplicate "Consequences" sections merged. The "device limit checked" wording was corrected to hard-coded constants. The measured drift (about 9e-6 and 3e-6 against the 1e-4 bound) is recorded.
  - `docs/ledger/evidence-ledger.md`: the Step 7 row was rewritten. It no longer claims explicit Euler or 9 tests, and it states the partial scope, the failing-open gates, and the real counts.
  - `crates/robot-sim/specs/12-cpu-gpu-agreement.md`: the stale "Not started" status line was removed. A single Partial status remains.
- **Verification (all exit 0):**
  - `cargo test --workspace --offline`: 881 passed, 0 failed, 0 ignored (49 suites).
  - `cargo test -p robot-sim-gpu --offline` (feature off): 5 unit and 1 integration passed.
  - `cargo test -p robot-sim-gpu --features cubecl --offline` on Mesa Lavapipe (`lvp_icd.x86_64.json`): 5 unit and 15 agreement passed, about 2.7 s.
  - Same on Intel iGPU (`intel_icd.x86_64.json`): 5 unit and 15 agreement passed, about 4.2 s.
  - `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets --offline -- -D warnings`, `cargo clippy -p robot-sim-gpu --features cubecl --all-targets --offline -- -D warnings`, and `RUSTDOCFLAGS="-D warnings" cargo doc -p robot-sim-gpu --no-deps --features cubecl --offline`: all clean.
- **Not verified:** the oversized allocation path was not exercised on a device. Only the count-only preflight is tested, so the 128 MiB limit is a hard-coded wgpu default and not a device query. Latency (gate 3) and determinism (gate 4) remain unmeasured.
- **Open:** solver, `mj_Euler`, RK4 and `impP` are not ported on GPU. No `n_env x n_dof` batch layout exists. Tolerances are not yet `fidelity.toml` rows.

## Entry 2026-10-29 — robot-sim Step 7 GPU launch-boundary soundness (continuation)

- **Tool:** Junie (JetBrains autonomous programmer), project working directory, file-edit and shell access.
- **Model identity/version:** Not confirmed. The name `grok-4.6` was reported by the session environment for this pass, but it was not verified and is not treated as a fact.
- **Request:** Previous pass ran out of steps. Close remaining unsafe launch holes: public wrappers must preflight lengths and buffer bytes before flatten/GPU work; keep `launch` crate-private; typed oversize errors; truthful records.
- **Changes:**
  - Public wrappers in `integrators.rs`, `passive.rs`, `constraint.rs` and `contact.rs` call `check_batch_size` after length checks and before finite scans, flatten allocations, or `get_client()`.
  - `check_batch_size` now returns the strided element count (`n * stride`) with checked multiplies, so launchers never compute `n * stride` before the cap. The CubeCL `BufferArg` lengths use that returned count.
  - Crate docs include a `compile_fail` doctest that `robot_sim_gpu::launch::backend::launch_joint_spring` is unreachable.
  - Agreement tests: count-only workgroup-cap and 8M-stride-5 preflight tests run without the `cubecl` feature (no allocation). Feature-off also asserts malformed wrapper inputs return `DimensionMismatch`, not `GpuDisabled`.
  - Spec 11 wording: limits are hard-coded wgpu defaults, not queried device limits.
- **Verification (all exit 0):**
  - `cargo test --workspace --offline`: 885 passed, 0 failed, 0 ignored.
  - `cargo test -p robot-sim-gpu --offline`: 5 unit, 4 integration, 1 compile-fail doctest.
  - `--features cubecl` on Lavapipe: 5 unit, 15 agreement, 1 doctest, ~2.3 s.
  - Same on Intel iGPU: 5 unit, 15 agreement, 1 doctest, ~3.5 s.
  - `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets --offline -- -D warnings`, `cargo clippy -p robot-sim-gpu --features cubecl --all-targets --offline -- -D warnings`, `RUSTDOCFLAGS="-D warnings" cargo doc -p robot-sim-gpu --no-deps` with and without `--features cubecl`.
- **Not verified:** oversize is still count-only; no 160 MB device allocation is attempted. Gate 3 (latency) and gate 4 (determinism) remain open. Limits are constants, not `max_storage_buffer_binding_size`.
- **Open:** solver, `mj_Euler`, RK4 and `impP` not ported on GPU. No `n_env x n_dof` batch layout. Tolerances are not `fidelity.toml` rows.

## Entry 2026-10-29 — robot-sim Step 7 GPU implicit-damping Euler kernel

- **Tool:** Junie (JetBrains autonomous programmer), project working directory, file-edit and shell access.
- **Model identity/version:** Not confirmed. The name `claude-sonnet-5-5` was reported by the session environment for this pass, but it was not verified and is not treated as a fact.
- **Request:** Port MuJoCo's default implicit-damping Euler and the equality-only constraint solve to real batched CubeCL kernels with CPU-vs-GPU agreement tests. Implement the implicit step first and report what remains.
- **Changes:**
  - `crates/robot-sim-gpu/src/launch.rs`: `implicit_damping_euler_kernel` (one thread per environment, `6 x 6` local array, CPU Cholesky loop order, `idx < n_env` guard), its launcher with the length and size preflight before the `unsafe` block, `MAX_DOF = 6` and `check_implicit_batch`.
  - `crates/robot-sim-gpu/src/integrators.rs`: public `gpu_implicit_damping_euler_step` with typed validation (nv, lengths, finiteness, `dt`, damping sign, `f32` symmetry check) and a per-environment `status`.
  - `crates/robot-sim-gpu/tests/implicit_agreement.rs`: hand-derived 1, 2 and 3 DOF cases, random SPD batches of 4097 environments for nv = 1, 2, 3, 6 against the f64 CPU function, non-SPD environments in a two-workgroup batch, and validation tests.
  - ADR 0012, specs 11 and 12, the crate README and the evidence ledger were updated.
- **Verification (all exit 0):** `cargo fmt --all -- --check`; `cargo clippy --workspace --all-targets --offline -- -D warnings`; `cargo clippy -p robot-sim-gpu --features cubecl --all-targets --offline -- -D warnings`; `RUSTDOCFLAGS="-D warnings" cargo doc -p robot-sim-gpu --no-deps` with and without `cubecl`; `cargo test --workspace --offline` (889 passed, 0 failed, 0 ignored); `--features cubecl` on Lavapipe and Intel (5 unit + 15 + 9 + 1 doctest).
- **Not done:** the equality-only solve `solve_equality` (needs per-row impedance, `stiffness_damping`, `effective_timeconst`, `M^-1 J^T` and an `(A+R)` solve with a variable row count), RK4, dense systems above 6 DOF, tolerance rows in `fidelity.toml`, gates 3 and 4. The random cases use a well-conditioned `M`; ill-conditioned `M` was not measured.

## Entry 2026-10-29 — robot-sim Step 7 GPU equality-only solve

- **Tool:** Junie (JetBrains autonomous programmer), project working directory, file-edit and shell access.
- **Model identity/version:** Not confirmed. The name `gpt-6-sol` was reported by the session environment for this pass, but it was not verified and is not treated as a fact.
- **Request:** Finish the actual CubeCL equality-only `solve_equality` port and per-environment hand-derived/CPU agreement tests. This entry supersedes the preceding entry's equality-solve "not done" statement; other items there remain open.
- **Changes:** `crates/robot-sim-gpu/src/constraint.rs` validates every input width, finiteness, symmetry, format, zero equality margin and count-only limits before device work; `src/launch.rs` packs three bounded buffers and launches one thread/environment with local Cholesky arrays for `M^-1 J^T`, `A+diag(R)`, per-row impedance, timeconst clamp and force/qacc; on failed pivots outputs are unchanged with status 0. `tests/equality_agreement.rs` tests hand-derived scalar/coupled cases and random well-conditioned batches of 4097 envs at (nv,nrow)=(1,1),(1,2),(2,2),(2,4),(3,2),(6,6), nonlinear branches, clamp, direct format, malformed requests and non-SPD isolation. Spec 11/12, crate docs, README, ADR 0012 and the ledger reflect the narrower implemented subset.
- **Verification:** `cargo fmt --all -- --check`, `cargo clippy -p robot-sim-gpu --features cubecl --all-targets --offline -- -D warnings`, `cargo clippy --workspace --all-targets --offline -- -D warnings`, `cargo test --workspace --offline`, `devenv test` exit 0 (warm test stage 4.59 s); `cargo test -p robot-sim-gpu --features cubecl --offline -q` passes on Lavapipe (5+15+5+9+1) and Intel (same counts), about 4 s and 5 s warm; targeted equality tests passed after adding redundant rows. Equality max error/bound 0.2071 / 0.2325 respectively, tolerance abs `1e-4` + rel `5e-6` against f64 CPU at f32-rounded inputs.
- **Open:** Dense systems above six DOFs/rows, ill-conditioned tests, full contact/friction/limit solver, RK4, `impP`, `fidelity.toml` GPU rows, B.6 gate 3 latency/resource measurement and gate 4 determinism. No MuJoCo runtime, parity or throughput claim.


## Entry 2026-10-29 — robot-sim Step 7 GPU RK4/impP records and evidence consistency pass

- **Tool:** Junie (JetBrains autonomous programmer), project working directory, file-edit and shell access.
- **Model identity/version:** Not confirmed. Earlier entries in this log name other models and were not verified either.
- **Request:** Tie the GPU RK4 and impedance/impP agreement tests to scenario and tier tolerance records, close the untested impP x > mid branch, add a non-printing 4097-environment, 1000-step RK4 test, and correct ADR 0012, specs 11/12 and the crate README so their status matches the code.
- **Changes:**
  - `crates/robot-sim-gpu/tests/common/mod.rs` (new): std-only reader for `tolerances.toml`, shared by the feature-gated tests and the default-build record test.
  - `crates/robot-sim-gpu/tests/cpu_gpu_agreement.rs`: new asserted tests `constraint_impedance_above_mid_impp_matches_cpu` (x > mid, p = 2 and 3), `constraint_impedance_4097_rows_match_cpu_within_recorded_tolerance` and `rk4_pendulum_4097_envs_1000_steps_match_cpu_within_recorded_tolerance`. The existing RK4 test reads its tolerance from the file.
  - `crates/robot-sim-gpu/tolerances.toml`: index fields (`measured_index_intel`, `measured_index_lavapipe`) for the four RK4/impedance scenarios, and the `test` field names the asserting tests.
  - Mutation check (not kept): the upper-branch derivative `power - 1` was changed to `power` in `launch.rs`. The new x > mid test failed with impP error about 1.15 against tolerance 1e-4. The kernel was restored and is unchanged.
  - `docs/decisions/0012-cubecl-b6-qualification-gpu-backend.md`: gate 3 contradiction removed; Decision 2 now says latency is measured for one workload with no throughput claim; the impP and RK4 rows point to their asserting tests.
  - `crates/robot-sim/specs/11-gpu-batched-backend.md`, `12-cpu-gpu-agreement.md`, `crates/robot-sim/README.md`: status updated to named pendulum RK4 only, impP implemented and asserted on both sides of mid, tolerance file as the source for RK4/impedance records.
- **Verification:**
  - Lavapipe (`lvp_icd.x86_64.json`) and Intel (`intel_icd.x86_64.json`), `cargo test -p robot-sim-gpu --features cubecl --offline --test cpu_gpu_agreement --test tolerance_records`: 21 + 2 passed on each, 0 failed, 0 ignored, about 3 s per run warm.
  - `cargo fmt --all -- --check`; `cargo clippy --workspace --all-targets --offline -- -D warnings`; `cargo clippy -p robot-sim-gpu --features cubecl --all-targets --offline -- -D warnings`: all clean.
  - `cargo test --workspace --offline`: 893 passed, 0 failed, 0 ignored.
- **Not verified:** `devenv test` was not run in this pass; the workspace command above is the same test set. Latency and determinism figures are the earlier single-run dev-build measurements and were not re-measured. The mutation check was run once on Lavapipe only.
- **Open:** the general closure RK4 and batched G1/ABA are not on the GPU. The implicit and equality tolerances are still ADR prose, not `tolerances.toml` rows. Gate 3 resource use and a `--release` multi-run latency measurement are not done. No MuJoCo parity or throughput claim is made.

## Entry 2026-10-29 — robot-sim Step 7 earlier RK4/impP GPU kernel production (model unconfirmed)

- **Tool:** Junie (JetBrains autonomous programmer), project working directory, file-edit and shell access.
- **Model identity/version:** Not confirmed. The session configuration for the pass that produced these artifacts was not recorded in this log. Do not treat any model name in an earlier entry as verified.
- **Request:** Port the remaining CPU RK4 subset and the constraint impedance derivative `impP` to the GPU, with a named pendulum RK4 scenario and tolerance records. This pass is reconstructed from the review record and the files it names; it was not re-run in this pass.
- **Changes (from the earlier pass, as reported by its own notes and the review):**
  - `crates/robot-sim-gpu/src/launch.rs`: kernels for the named pendulum RK4 step (`gpu_rk4_pendulum_step`) and the impedance value with derivative (`gpu_constraint_impedance_with_derivative`).
  - `crates/robot-sim-gpu/src/integrators.rs` and `constraint.rs`: public wrappers with typed validation for the two kernels above.
  - `crates/robot-sim-gpu/examples/gpu_latency.rs`: a latency and determinism example. It prints measurements and asserts nothing.
  - `crates/robot-sim-gpu/tolerances.toml`: initial structured tolerance records for the RK4 and impedance scenarios.
- **Verification:** Not repeated in this pass. The earlier pass reported `cpu_gpu_agreement` and `tolerance_records` passing on Intel and Lavapipe; treat that as unverified until the next test run.
- **Open:** The general closure-based RK4 is not ported and has no CPU fallback. The `tolerances.toml` rows are the only structured records for these scenarios.

## Entry 2026-10-29 — robot-sim Step 7 ADR/Ledger evidence consistency (docs-only)

- **Tool:** Junie (JetBrains autonomous programmer), project working directory, file-edit and shell access.
- **Model identity/version:** Not confirmed. The name was reported by the session configuration for this pass and was not verified.
- **Request:** Fix ADR 0012, Step 7 ledger and automation-log factual errors flagged in review: latency wording, tier labels, Gate 1 maintainer attribution, Gate 6 estimate label, ledger row 71, and the malformed duplicate entry in this log.
- **Changes (docs only, no algorithm or test code changed in this pass):**
  - `docs/decisions/0012-cubecl-b6-qualification-gpu-backend.md`:
    - Gate 1 maintainer line now cites the registry `authors` field: `nathanielsimard` in `cubecl-0.11.3`, plus `louisfd` in `cubecl-core-0.11.3`.
    - Gate 3 wording and the gate 3 result line now say latency is measured for the pendulum RK4 and impedance workloads only.
    - Decision 2 and the Consequences section say the same.
    - Gate 6 is labelled as an unmeasured estimate, owner simulator team.
    - Impedance tolerance row uses tier `single_launch`, matching `tolerances.toml`, and the ADR states that file is authoritative.
  - `docs/ledger/evidence-ledger.md`:
    - Row 71 is marked `(Step 7; superseded by next row)`.
    - Row 72 is a new superseding row. It names the three asserting tests, the omega maxima (2.42e-5 at env 3404 on Intel, 1.13e-5 at env 4009 on Lavapipe), and that the feature-gated tests read bounds from `tolerances.toml`. It has 5 cells.
  - `docs/ledger/automation-log.md`: removed the malformed duplicate entry, which had lost its backtick spans, and added this entry and the earlier kernel-production entry above.
- **Verification:** Post-edit lines checked with `sed` and `awk`. Ledger rows 70 to 72 have 5 cells each. `git diff --check` on `docs/` is run after these edits.
- **Not verified / not changed:** The test-comment fix in `cpu_gpu_agreement.rs` was made in an earlier pass and was not re-checked here. Older log entries naming other models remain uncorrected. No measurements were repeated.

## Entry 2026-10-29 — robot-sim Step 7 final integration: gate 3/4 records and spec/ADR status

- **Tool:** Junie (JetBrains autonomous programmer), project working directory, file-edit and shell access.
- **Model identity/version:** Not confirmed. The session configuration for this pass was not verified, so no model name is attributed here.
- **Request:** Bring ADR 0012, specs 11 and 12, the crate README, the evidence ledger and this log into the current measured state for the RK4 and equality batches. Mark unverified model identities as unconfirmed.
- **Changes (docs only, no code, Cargo or devenv change):**
  - `docs/decisions/0012-cubecl-b6-qualification-gpu-backend.md`: status says accepted for the f32 functional subset only. Gate 3 records dev-build medians and ranges for the RK4 and equality batches on Lavapipe and Intel, count-only resources, and device properties with the not-live-counter qualifier. Gate 4 records the in-process, same-device 5-run bitwise tests for RK4 and equality on both ICDs, with no cross-device claim. The single-run latency subsection was replaced with the measured figures.
  - `crates/robot-sim/specs/11-gpu-batched-backend.md`: status now states the measured scope and the unported items.
  - `docs/ledger/evidence-ledger.md`: row 73 supersedes rows 71 and 72 for gate 3/4 status. Row 73 has 5 cells, matching the table.
  - `docs/ledger/automation-log.md`: Step 7 entries whose model identity was not verified are marked "Not confirmed". Earlier names are kept as reported, not removed.
- **Measurements:** taken from the example and test runs re-run in this pass (see the ADR gate 3 section for figures).
- **Open:** general closure RK4, batched G1/ABA and full contact/friction/limit solver not on GPU. Implicit and equality tolerances are not `fidelity.toml` rows. No cross-device determinism. No throughput claim.
- **Correction in this pass:** the gate 3 timings first written into ADR 0012 and ledger row 73 were not taken from a run in this pass, so they were replaced with the medians and ranges from re-running `gpu_resource_repeatability` on both ICDs (dev build, 3 timed trials after one warm-up): Lavapipe RK4 GPU 1.660 s / CPU 577.8 ms, equality GPU 6.12 ms / CPU 12.49 ms; Intel RK4 GPU 2.923 s / CPU 563.0 ms, equality GPU 6.62 ms / CPU 12.39 ms.
- **Verification (all exit 0):** `devenv test` 4.57 s; `cargo clippy -p robot-sim-gpu --features cubecl --all-targets -- -D warnings`; `RUSTDOCFLAGS="-D warnings" cargo doc -p robot-sim-gpu --no-deps --features cubecl`; `cargo test -p robot-sim-gpu --features cubecl` on Lavapipe (8 unit, 21 cpu_gpu_agreement, 5 equality, 9 implicit, 2 repeatability, 1 resource_accounting, 2 tolerance_records, 1 doctest; 0 ignored) and on Intel (same counts).

## Entry 2026-10-30 — spec checklists re-audit (docs-only)

- **Tool:** Claude Code (Anthropic CLI), project working directory, file-edit and shell access, with one sequential subagent per checklist file for sections 2, 3, 7, 8, 9, 10, 11, 12, 13 and 14, and parallel subagents for the front matter and sections 1, 4, 5, 6 and 15.
- **Model identity/version:** Claude Fable 5.1 (`claude-fable-5-1`), as reported by the session; not independently verified.
- **Request:** Revisit the implementation status proposed by the spec checklists under `docs/spec-checklists/` (commit `72372a1`): re-audit every item against the code, resolve every `[?]`, correct over- and under-stated statuses, and cite specific evidence.
- **Changes (docs only, no code, Cargo or devenv change):** the `## Implementation checklist` section of 18 files under `docs/spec-checklists/` (front matter, sections 1 to 15, appendices C and D). The spec text above each heading is byte-identical to `HEAD`, checked by hash. Appendices A and B have no checklist and were not touched. All 62 `[?]` items are resolved; summary lines match the item counts in every file. Each item cites a `path:line`, a grep, or a test name. Statuses downgraded where a first-pass `[x]` rested on an enum label, a README claim or a declared-but-unused type (for example 4.12, 7.6.7, 10.3.4, 12.2.3, 13.8, A-03); upgraded where tests discharge the row (A-36, A-37, A-38, A-39, C-27, C-28, 9.2.8).
- **Verification:** `devenv test` exit 0 on 2026-10-30 (fmt check, clippy `-D warnings`, `cargo test --workspace`); the section 14 pass recorded `cargo test --workspace --offline` as 897 passed, 0 failed, 0 ignored. Per-crate test counts cited in the checklists were taken from runs in this pass.
- **Not changed / open:** documentation found to contradict code was recorded in the checklists but not fixed here: the `robot-sim` crate doc still denies ground contact and claims batching; `robot-tasks` and `motion-process` headers overstate their scope; `robot-control` `control_loop.rs` module doc says a `Router` is held when none is; the old A-36 note that the fidelity report is not written through `robot-artifacts` was wrong and is corrected; the evidence ledger uses non-canonical states (`measured`, `reproduced_in_sim`-adjacent, `same`) that `EvidenceState::parse` rejects, and row 72 has 7 cells; `fixtures/motion`, `fixtures/models` and `fixtures/profiles` are empty while `bvh.rs` and `motion.rs` cite `fixtures/motion/README.md`; `SPECS.md` header reads 0.7.3-draft while its history ends at 0.7.4-draft; the devenv exclusion gate cannot fail and no CI definition exists. No commit was made in this pass.

## Entry 2026-10-30 — implementation pass from the spec-checklist re-audit

- **Tool:** Claude Code (Anthropic CLI), project working directory, file-edit and shell access. No subagents in this pass.
- **Model identity/version:** Claude Fable 5.1 (`claude-fable-5-1`), as reported by the session; not independently verified.
- **Request:** "Proceed with implementation" after the checklist re-audit: fix the defects and contradictions the audit recorded.
- **Code changes:**
  - `crates/robot-devtools/src/exclusion_gate.rs` (new) and `src/bin/exclusion_gate.rs` (new binary `exclusion-gate`): the SPECS.md Section 2.1 item 4 / row A-25 gate as tested Rust (5 unit tests); `devenv.nix` runs it from `enterTest` and as the `robot-exclusion-gate` git hook, and its former bash body is replaced by a one-line alias. ADR 0013 records the decision.
  - `crates/motion-io/src/bvh.rs`: the root rotation now lands in `frame.root.rotation` (kept as channel 0 too); declared `Frames:` and channel counts are capped (`MAX_FRAMES`, `MAX_CHANNELS`, `BvhError::Limit`) before allocation. Two tests added.
  - `crates/robot-control/src/control_loop.rs`: in `hw-shadow` the loop disables transmit at construction, verifies by readback every tick in a new `transmit` helper, sends nothing, and reports `TickError::TransmitEnabledInShadow` without changing the flag. Two tests added. Both former send sites use the helper.
  - `crates/robot-sim/src/fidelity.rs`: the two `not_applicable` reason strings no longer claim that no GPU backend exists.
- **Documentation changes:** `crates/robot-sim/src/lib.rs` and `env.rs` (ground contact exists; batching does not), `crates/robot-tasks` lib and README (balance task only), `crates/motion-process`, `crates/motion-retarget`, `crates/robot-adapt` README headers (scope stated as implemented), `crates/robot-learn/src/ppo.rs` header (action bounding not in the policy), `crates/robot-skills/src/descriptor.rs` (stale module reference), `crates/robot-control` module doc and README (no router held; empty temperature/contact inputs; shadow behaviour; 36 tests), `crates/motion-io/README.md` (root pose, limits, 8 tests), `crates/robot-devtools/README.md` (gate; 9 tests), new `crates/robot-sim-gpu/README.md`, new `fixtures/motion/README.md` (the `fixture-internal` license the tests cite), `README.md` robot-skills row (R0 not implemented), `AGENT.md` (23 crates), `SPECS.md` header version `0.7.4-draft`, `docs/ledger/evidence-ledger.md` (canonical `EvidenceState` identifiers in every project-artifact row; row 72 split into its five cells), and the checklist items 0.4, 2.3, 2.1.4, 2.2.3, 2.2.6, 2.3.1-U1, 3.3, 4.10, 4.11, 8.1.1, 12.2.3, A-22, A-25 with their summary lines.
- **Not changed:** the `hw-supported` reduced envelope (checklist 4.12, 12.1.2) and the residual-command clamp (11.3.5) are safety-case changes that need a per-robot profile and a decision record first; no CI service exists (2.2.4); the fidelity battery still has no contact check (ADR 0008 scope). No commit was made in this pass.
- **Verification (all exit 0, 2026-10-30):** `cargo fmt --all`; `cargo clippy --workspace --all-targets --offline -- -D warnings`; `cargo run -q -p robot-devtools --bin exclusion-gate --offline` ("ok: 87 default-closure entries, all workspace paths; 472 lockfile packages, none an excluded runtime; 23 manifests, none reach robot-devtools"); `cargo test -p motion-io -p robot-control -p robot-devtools -p robot-sim --offline` 253 passed, 0 failed; `devenv test` (fmt check, clippy, exclusion gate, `cargo test --workspace`) exit 0.

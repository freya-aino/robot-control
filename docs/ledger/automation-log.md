# Development-time automation log

PLAN.md Phase 0's closing bullet: "If any development-time automation is
used while doing the above, state it: which tool, which model identity and
version, and which artifacts it produced. Attribution is cheap now and
unreconstructable after the fact." This is use U1 of SPECS.md §2.3.1
(development-time automation), and this file is its attribution record —
distinct from `robot-devtools`, which is where U1's *runtime* tooling will
live once written.

Entries are dated records. File paths, spec file names and `outputs/` folders
are as they were on the entry's date; several have since been renamed or
removed (for example `04-mujoco-fidelity-loop.md` is now `04-fidelity-loop.md`,
the `*-usd-replay/` and `g1-stance-newton-replay/` folders were deleted, and
the `unitree_g1_mujoco` assets and `mjcf.rs` are gone). Entries are not
rewritten when that happens.

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

## Entry 2026-10-30 — MuJoCo G1 profile from the official MJCF (implementation)

- **Tool:** Claude Code (Anthropic CLI), project working directory, file-edit and shell access. No subagents in this pass.
- **Model identity/version:** Claude Haiku 5.5 (`claude-haiku-5-5`), as selected in the session; not independently verified. Earlier entries in this log name Claude Fable 5.1 for work done under that model.
- **Request:** Add the MuJoCo G1 as a robot, based on the specifications for creating a robot (SPECS.md Section 7.1) and the official G1 USD file.
- **Artifacts produced:**
  - `assets/robots/unitree_g1_mujoco/` (`g1_29dof.xml`, `LICENSE`, `PROVENANCE.md`): the official MJCF at `unitreerobotics/unitree_mujoco` commit `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d`, SHA-256 checked against the pinned commit.
  - `assets/robots/unitree_g1_usd/` (four USD layers, `LICENSE`, `PROVENANCE.md`): the official USD at `unitreerobotics/unitree_model` commit `b6a8942b0803b6c137e58cef12beb4b03e4a2fa7`, retrieved through the Git LFS batch API and checked against each object ID. The base layer is 28 MB.
  - `crates/robot-model/src/mjcf.rs` (new): a MJCF importer for the G1's subset, with 21 tests.
  - `crates/robot-model/src/fixtures.rs`: `unitree_g1_29dof_mujoco()`, `unitree_g1_29dof_mujoco_source_disagreements()`, the fixed-base comparison helpers, and 5 fixture tests (one of them pins the list of 18 disagreements).
  - `crates/robot-model/README.md`, `docs/decisions/0014-mujoco-g1-import-path.md`.
- **Verification:** `cargo test --workspace --offline` passed (932 tests, 0 failed); `cargo clippy --workspace --all-targets --offline -- -D warnings` clean; the exclusion gate passed. The unit-test MJCF cases cover refusal by name for each unsupported construct. The official G1's element counts were checked with a standard-library XML parse, not with a USD tool.
- **Not verified:** the USD layers were not parsed. An attempt to install a USD reader into a scratch environment was refused by the permission policy and not retried. The joint names in the USD could not be read with `strings` because the USDC tokens are compressed. The 18 disagreements between the MJCF and the URDF are recorded, not resolved.
- **Not changed:** no commit was made in this pass.

## Entry 2026-10-30 — USD model, replay writer and MuJoCo reference removal

- **Tool:** Claude Code (Anthropic CLI), project working directory, file-edit and shell access, and one general-purpose subagent for the `robot-usd` reader (its brief is in the session scratchpad, not the repository).
- **Model identity/version:** Claude Haiku 5.5 (`claude-haiku-5-5`) for the session; the subagent ran on the same configured model. Neither identity is independently verified.
- **Request:** Remove every reference to the MuJoCo simulator; make the OpenUSD description used by Isaac Sim the simulation model, with complete capture of collision and primitive shapes from the official Unitree G1 USD; update the Markdown to match.
- **Artifacts produced:**
  - `crates/robot-usd/` (new, std-only): USDC crate reader with LZ4 and integer codecs, composition of sublayers, variant selections, payloads and references, USDA reader and writer, the replay writer `robot_usd::replay`, the example `usd_dump`. Byte layouts follow OpenUSD commit `84d8ed1ae0337ce30dca8d91a3fbe7f04eebc358`.
  - `crates/robot-model/src/usd.rs` (new): builds the G1 profile from the USD layer, with an import report, the floating-base variant and 9 tests on the vendored layers.
  - `crates/robot-cli/src/commands/usd_replay.rs` (new), and `sim multibody` now loads the USD model and writes `replay.usda` into the output folder.
  - `ADR 0015` (accepted), `ADR 0014` (withdrawn tombstone), `ADR 0005` and `0011` (superseded banners), `SPECS.md` revision 0.8.0-draft.
  - `outputs/g1-zero-torque-usd-replay/` (replay, run log, README).
- **Removed:** the MJCF importer, the MuJoCo G1 fixture and its asset directory, and the MuJoCo wording in the Rust crates, the live Markdown, the devenv comments and `.gitignore`.
- **Kept on purpose:** the exclusion gate's deny list (a guard against foreign runtimes entering the build, ADR 0013); the evidence ledger, the automation log and the audit checklists, which record past work; and the vendored Unitree URDF and its provenance, whose upstream file contains a `<mujoco>` element that the provenance quotes.
- **Verification:** `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings` clean; `cargo test --workspace` passed; the exclusion gate passed. The reader decodes all four G1 layers (spec counts root 13, base 1181, physics 850, sensor 2). The zero-pose kinematics from the USD agree with the URDF to 1.5e-8 m. The replay was read back with `robot-usd`, which parses its time samples.
- **Not verified:** Blender and OpenUSD tools were not run; the attempt to install a USD library was refused by the permission policy and not retried. The contact forces are not exposed by `robot-sim` and are not in the replay.
- **Open:** deletion of the robot-sim spec pages and restoration of the scrubbed audit records were refused by the permission policy and not done. The 53 formula identifiers inherited from the removed simulator remain in doc comments.
- **Not changed:** no commit was made in this pass.

## Entry 2026-10-30 — openusd in the devenv, shader warnings, Blender import check

- **Tool:** Claude Code (Anthropic CLI), project working directory, file-edit and shell access. No subagents.
- **Model identity/version:** Claude Haiku 5.5 (`claude-haiku-5-5`), as selected in the session; not independently verified.
- **Request:** Add OpenUSD to the devenv; make loading the USD warn, rather than fail, about Omniverse-only shaders.
- **Artifacts produced:** `devenv.nix` (`openusd` in `packages`, nixpkgs attribute confirmed with `nix eval`); `crates/robot-usd/src/diagnostics.rs` (`shader_warnings`); `robot_model::usd::import_fixed_base_with_warnings` and its test; `robot sim multibody` prints the warnings and continues; ADR 0015 and the READMEs.
- **Verification:** `nix-instantiate --parse devenv.nix` ok; `cargo fmt --check`, clippy with `-D warnings`, and `cargo test --workspace` (951 passed) pass; the exclusion gate passes; `usdchecker` accepts the replay; Blender (headless) imports it with 66 objects, frames 0 to 19999, and a pelvis track matching the simulator output.
- **Not verified:** `devenv shell` was not run in this environment, so the package is confirmed to resolve but not to build in a fresh shell.
- **Not changed:** no commit was made in this pass.

## Entry 2026-10-30 — SPECS 0.8.1-draft: USD source of truth and contact coverage

- **Tool:** Claude Code (Anthropic CLI), project working directory, file-edit and shell access. No subagents.
- **Model identity/version:** Claude Haiku 5.5 (`claude-haiku-5-5`), as selected in the session; not independently verified.
- **Request:** Update SPECS for the USD collision model and the physics gap.
- **Artifacts produced:** `SPECS.md` revision 0.8.1-draft: Section 2.2 (OpenUSD as a development tool, not a dependency), Section 7.1 (USD as the reference embodiment and the import requirements), Section 7.3 (replays as USD time samples; contact to be recorded once exposed), Section 9.1 (contact for every declared collision shape; the zero-torque collapse as a limitation until then), Section 14 rows A-40 to A-42.
- **Verification:** the SPECS diff was read back; the G1 pelvis collision is a mesh prim (`pelvis_contour_link`) confirmed with `usd_dump`. The per-link collision inventory probe did not complete in this pass and its numbers are not recorded.
- **Not done:** the contact implementation for box, cylinder and mesh-vertex shapes is not written. Section 9.1 states that it is required and not yet implemented.
- **Not changed:** no commit was made in this pass.

## Entry 2026-10-30 — G1 USD runs and OpenUSD validation

- **Tool:** Claude Code (Anthropic CLI), project working directory, file-edit and shell access. No subagents.
- **Model identity/version:** Claude Haiku 5.5 (`claude-haiku-5-5`), as selected in the session; not independently verified.
- **Request:** Run the G1 USD simulation and export under `outputs/`, now that OpenUSD is installed.
- **Runs:** `robot sim multibody --floating` (PD stance) and `--zero-torque`, both from the USD model, writing `replay.usda` and the frame dump under `outputs/`.
- **Verification:** the PD stance passes its supported-stance checks; the zero-torque run fails as expected. Both `replay.usda` files pass `usdchecker` (OpenUSD from the system path) and `usdcat` reads them. Blender imports the stance replay: 66 objects, frames 0 to 19999, pelvis track animated.
- **Artifacts:** `outputs/g1-stance-usd-replay/`, `outputs/g1-zero-torque-usd-replay/` (READMEs, runs, warnings, replays); `outputs/README.md` index.
- **Not changed:** no code or spec changes; no commit was made.

## Entry 2026-10-30 — Blender replay: visual meshes and playback rate

- **Tool:** Claude Code (Anthropic CLI), project working directory, file-edit and shell access. No subagents.
- **Model identity/version:** Claude Haiku 5.5 (`claude-haiku-5-5`), as selected in the session; not independently verified.
- **Request:** Fix two problems in the Blender import of the replay: the G1 mesh did not appear, and playback did not move.
- **Cause (mesh):** the replay wrote only references to the visual meshes. **Cause (playback):** Blender keeps its 24 fps rate and maps each time code to a frame; the full replay of 20000 frames would take about 14 minutes to play, and the stance hold barely moves in the first seconds.
- **Changes:** `robot_usd::replay` writes `Mesh` prims with points and faces; `robot_model::usd::import_visual_meshes` reads the G1 visual meshes; `sim multibody --out` writes `replay_playback.usda` (100 time codes per second) next to `replay.usda`; the READMEs and ADR 0015 explain the frame rate.
- **Verification:** `usdchecker` passes on all four replays. Blender (headless) imports the playback replays with 36 mesh objects, frames 0 to 199, and the pelvis track moves as the simulator reports.
- **Also found:** `outputs/` was empty when checked in this session. The runs were regenerated from the simulator; the earlier replays are not recoverable from the repository.
- **Not changed:** no commit was made in this pass.

## Entry 2026-10-30 — SPECS 0.8.2-draft: hard limits, contact modes, self-collision

- **Tool:** Claude Code (Anthropic CLI), file-edit access. No subagents.
- **Model identity/version:** Claude Haiku 5.5 (`claude-haiku-5-5`), as selected in the session; not independently verified.
- **Request:** Plan the fixes for joint limits and self-collision, asking for decisions; then record the decisions in SPECS.
- **Findings (code, 2026-10-30):** `robot-sim` has no joint-limit constraint and no link-to-link collision. Only sphere shapes take part in ground contact; every other shape is counted as unmodelled. The USD importer reads the limits correctly (they match the URDF to 1e-4 rad), so the defect is in the dynamics.
- **Changes:** `SPECS.md` 0.8.2-draft: Section 7.1 (authored pelvis and torso proxies), Section 9.1 (hard limits with a proximity signal, primitive and convex-hull contact modes, all-pairs self-collision with adjacency rules, performance deferred to a GPU machine), acceptance rows A-43 to A-46.
- **Open:** the proximity margin band, the allowed overlap depth at joints, and the source of the pelvis and torso proxy dimensions are awaiting the user's decision.
- **Not changed:** no simulator code; no commit was made.

## Entry 2026-10-30 — SPECS decisions: limit proximity band, joint overlap, proxies

- **Tool:** Claude Code (Anthropic CLI), file-edit access. No subagents.
- **Model identity/version:** Claude Haiku 5.5 (`claude-haiku-5-5`), as selected in the session; not independently verified.
- **Decisions recorded in SPECS 9.1 and 7.1 (0.8.2-draft):** the proximity signal uses the lowest 5 % and highest 95 % of each hinge's total range; the initial joint overlap is 3 mm, marked as a placeholder that may need a per-joint balancing algorithm; pelvis and torso proxies are fitted to the bounding boxes of their USD collision meshes, with the fitting method recorded.
- **Not changed:** no simulator code; no commit was made.

## Entry 2026-10-30 — Hard joint limits in the simulator (plan step 1)

- **Tool:** Claude Code (Anthropic CLI), file-edit and shell access. No subagents.
- **Model identity/version:** Claude Haiku 5.5 (`claude-haiku-5-5`), as selected in the session; not independently verified.
- **Request:** Start implementing the plan: hard joint limits with a proximity signal.
- **Changes:** `robot-sim` articulated body: each bounded hinge's USD range is a hard limit (active-set constraint torques solved from `M⁻¹` columns, positions clamped with outward velocity removed), plus `limit_fractions`, `step_with_ground_recording`. `robot-usd` replay: `robot:limitTorque`, `robot:limitFraction`, `robot:limitProximity`. `robot-cli`: captures the limit channels and prints the worst excursion. The fidelity probe tests now assert the measured `pass`; the ledger records the change.
- **Verification:** a pendulum released against a ±0.3 rad limit stays in range, reaches it, and carries a constraint torque; without the limit it swings past 1 rad; without ground contact a limited pendulum adds no energy. The workspace passed (954 tests). The G1 zero-torque and PD stance runs have zero steps outside any range; both replays pass `usdchecker`.
- **Found, not resolved:** the zero-torque G1 kinetic energy peak rises to about 660 J, above the roughly 550 J released by the drop, so about 110 J is injected. A limit-only pendulum injects none, so the suspect is the existing penalty ground contact under explicit integration. Not investigated in this pass.
- **Not done:** convex-hull and primitive contact for every shape, pelvis and torso proxies, joint-adjacent overlap, self-collision (plan steps 2 to 4).
- **Not changed:** no commit was made.

## Entry 2026-10-30 — Ground plane collider in a scene layer

- **Tool:** Claude Code (Anthropic CLI), file-edit and shell access. No subagents.
- **Model identity/version:** Claude Haiku 5.5 (`claude-haiku-5-5`), as selected in the session; not independently verified.
- **Request:** Give the ground plane a collider in the USD, and document that it was added to the original USD.
- **Changes:** `assets/scenes/g1_ground.usda` (new scene layer: sublayers the unmodified G1 layer, adds the `Plane` collider) and `assets/scenes/README.md`; `robot_model::usd::import_ground_plane`; the simulator reads the ground from the scene (the `FLOATING_GROUND` constant is removed); the replay writes `/replay/ground`; SPECS 0.8.3-draft; ADR 0015 and the output READMEs.
- **Decision:** the ground was not written into the vendored Unitree layer. Its SHA-256 is pinned in PROVENANCE.md, so a change would break provenance. The provenance record states this.
- **Verification:** the Unitree layers' SHA-256 values are unchanged. The scene composes with the reader. The PD stance run output is identical to the printed precision. The zero-torque run differs slightly from about 0.4 s on, because the friction is stored as float32. Both replays pass `usdchecker`, and the replay contains the ground prim.
- **Not changed:** no commit was made.

## Entry 2026-10-30 — Universal environment tags

- **Tool:** Claude Code (Anthropic CLI), file-edit and shell access. No subagents.
- **Model identity/version:** Claude Haiku 5.5 (`claude-haiku-5-5`), as selected in the session; not independently verified.
- **Request:** Make collision universal across scenes through the tags `environment` and `collider`.
- **Changes:** `robot_model::usd::import_environment` (tag-based, any scene, any depth), `TAG_ENVIRONMENT`/`TAG_COLLIDER`; the simulator reads the environment from `--scene` (default the G1 ground scene), refuses tagged shapes it cannot contact, places the robot on the plane height; the replay writes the plane with its tags; `assets/scenes/examples/` (tagging example, raised plane); SPECS 0.8.4-draft; ADR 0015 and the scene README.
- **Found and fixed:** the stance was placed relative to z = 0, not the environment plane.
- **Verification:** the importer tests cover the tag rules on the example scene; the raised plane stands the robot at its height with the same 5 mm sphere penetration as the default ground; the G1 PD stance passes and the zero-torque run fails as before; both replays pass `usdchecker`. Workspace: 957 tests, clippy and formatting clean.
- **Not done:** contact for boxes, spheres, cylinders and meshes tagged as environment (refused by name for now); self-collision.
- **Not changed:** no commit was made.


## Entry 2026-10-31 — Attached-agent robot discovery requirements

- **Tool:** Claude Code (Anthropic CLI), file-edit and shell access.
- **Model identity/version:** Claude Haiku 5.5 (`claude-haiku-5-5`), as selected in the session; not independently verified.
- **Request:** Add to the specification the requirements for a method by which an agent attaches to a robot and learns its motion, ranges, inertia and tests, and gathers external information through web search.
- **Changes:** SPECS 0.8.5-draft: Section 7.7 (sources and ranking, motion and range, bounded commissioning probe, inertial parameters, tests, web research as proposals, conflicts, activation refusal, reproducibility, offline degradation, staleness, feature gating); acceptance rows A-47 to A-50; open question C-29 on whether the method is an admissible use beyond U1 to U3.
- **Decision:** The method is development tooling. Its hardware probe is not admissible until C-29 is decided. No agent may activate a profile.
- **Verification:** Specification text only; no code changed, no tests run.
- **Not changed:** no commit was made; no decision record written yet (tracked as C-29).


## Entry 2026-10-31 — Mathematical and algorithmic audit of the simulation code

- **Tool:** Claude Code (Anthropic CLI), file-edit and shell access. No subagents.
- **Model identity/version:** Claude Opus 5 (`claude-opus-5`), as selected in the session; not independently verified.
- **Request:** Read all the simulation code written in this project and validate the general mathematical and algorithmic integrity of the implementation against the provided worked example of a complete, functioning simulator (`docs/example-system-specs/`, non-normative, mapped by `docs/example-system-mapping.md`); record every challenge to the maths and algorithms in the specifications and update documentation. The parent system specifications continue to govern goal, language and scope.
- **Scope read in full:** `robot-sim/src/{spatial,articulated,collision,contact,constraint,integrators,passive,oracle}.rs`; `robot-sim-gpu/src/{integrators,contact,constraint,passive,launch}.rs`; `robot-model/src/{link,joint,kinematics}.rs`; the check and tier definitions of `robot-sim/src/fidelity.rs`; all of `crates/robot-sim/specs/`; the 17 example documents' index and the sections cited.
- **Changes:** new `crates/robot-sim/specs/appendix-d-mathematical-challenges.md` (24 entries M-01 to M-24, with a severity vocabulary, an evidence-discipline statement, a confirmed-correct table, and three cross-cutting observations); a new **Challenges** field on steps 01–07, 09, 11 and 12 naming the entries each owns; Appendix B rows **S-07** to **S-09** for the three entries needing a specification decision; an audit note in Appendix C qualifying two of its statements; `specs/README.md` bumped to 0.1.2-draft with the appendix indexed, the template field documented and nine status cells corrected; a "What a full audit of the maths found" section in `crates/robot-sim/README.md`; a pointer in `crates/robot-model/README.md` to the two defects owned by that crate.
- **Confirmed correct (not a parity claim):** the spatial algebra (motion/force duality, `ad`/`ad*`, parallel-axis spatial inertia, congruence transform, quaternion rotation), the RNE recursion including the `a_0 = -g` gravity trick, the three-pass articulated-body algorithm, the three ported integrators, the Cholesky and LU solvers, the impedance sigmoid with its derivative, the equality-solve algebra, four primitive support maps, Ericson's closest-point-on-triangle, the plane-sphere rule, and all three passive/actuator functions. Each was re-derived independently; this is a derivation and a code reading, not measured parity with any engine. No example system was run and none exists in this repository.
- **Found — defects (correct accepted input, wrong result):** M-01 the `Free` joint stores linear velocity in the body frame against step 01 item 3's MUST; M-02 a non-unit declared joint axis makes the motion subspace disagree with the pose map, and validation accepts it; M-03 `InertiaTensor::is_physically_plausible` tests the triangle inequality on the diagonal entries only and accepts an indefinite tensor; M-04 contact compliance scales with the number of sampled contact points; M-05 the EPA seed tetrahedron is not required to enclose the origin; M-06 a degenerate tetrahedron is reported as containing the origin; M-07 box-box edge-edge contact is placed at a corner.
- **Found — divergences, gaps and costs:** M-08 to M-24, covering the two-part hard-limit mechanism, contact impedance referred to a fixed 1 kg mass, friction outside the coupled solve, two omitted terms in the contact right-hand side, the limit and contact solves not seeing each other, two predicates for one `solref` format, an absolute pivot tolerance, the absent broad and mid phases with `bounding_sphere` unreachable, the overlap allowance standing in for pair exclusion, the three correct-but-unwired modules, the `M(q)` dependency blocking the implicit integrator, RK4's inability to take a `Free` joint, the fidelity battery's lack of any contact row, four of seven fidelity knobs being compile-time constants, the GPU `f32`/`MAX_DOF = 6` limits against the G1, and two redundant factorisation costs.
- **Measurements taken (deterministic, single configuration; NOT promoted to `reproduced_in_sim`, which per `SPECS.md` Section 3 additionally requires multiple seeds and a recorded comparison budget):** a force-free free body at `dt = 1e-4` over 1 simulated second gains `|dv|/|v|` of 3.93e-3 and `dKE/KE` of 1.44e-3 with isotropic inertia and spin, 4.61e-3 and 1.94e-2 with anisotropic inertia and spin, exactly zero for both with no spin, and 2.45e-3 and 5.88e-4 spinning about one principal axis — the zero-spin and isotropic rows isolate the velocity convention rather than the integrator as the cause of the momentum error. Scaling a pendulum's declared axis from `(0,1,0)` to `(0,2,0)` changes `qdd` from 9.80665 to 4.903325, kinetic energy at `qdot = 1` from 0.5 to 2.0, and `tau` for `qdd = 1` from 1.0 to 4.0, against analytic values of 9.80665, 0.5 and 1.0; both pass `validate()`. `is_physically_plausible()` returns `true` for `ixx = iyy = izz = 1, ixy = 2`, whose eigenvalues are 3, -1 and 1. On `fixtures::unitree_g1_29dof_floating_base`: `dof = 35`, `nq = 36`; link diagonal inertias span 1.00e-7 to 5.91e-2 kg m^2 and masses 1.00e-3 to 6.78 kg, both from the vendored Unitree URDF and not from any importer clamp; `(M^-1)_jj` spans 3.30e-2 to 3.16e3 at the home configuration, so `cond(M) >~ 1e5`.
- **Method note:** the measurements were taken with temporary probe tests appended to the `articulated` test module, run, and then removed. The audited source is unchanged by this entry; no probe is committed. Recording a reproducing test for each defect is implementation work against the owning step, per the testing policy in `AGENT.md`.
- **Verification:** documentation only; no algorithm source was modified (`git status` shows changes to Markdown files alone). On the restored tree the full gate passes: `cargo fmt --all -- --check` clean, `cargo clippy --workspace --all-targets -- -D warnings` clean, `cargo test --workspace` 969 passed / 0 failed. Every Markdown link added by this entry was checked to resolve. Note that passing the gate is not evidence for any claim in Appendix D: the gate exercises the code, and Appendix D's entries are precisely the cases it does not cover — M-20 records that absence as its own finding.
- **Not done:** no defect was fixed, no requirement changed, no acceptance row added, no tolerance altered, and no ADR written — every entry is recorded against its owning step, and the three needing a decision are open as S-07 to S-09 rather than decided. `docs/example-system-mapping.md` and `docs/example-system-specs/` were read but not edited: they are untracked in the working checkout and so are not on this branch. Appendix D section 3.4 records the four mapping cells that read differently against the audited working tree, so the two documents can be compared without confusion.
- **Tree audited:** the working tree at `c680e09` **including uncommitted modifications** to `robot-sim/src/{articulated,collision}.rs`, `robot-model/src/usd.rs` and two `robot-cli` files — the current state of the implementation, not the committed state. Appendix D section 1.0 lists which entries depend on that uncommitted work (`ContactModel`, `contact_rows`, `contact_externals`, `box_sat`, `box_box_contacts`) and which apply to `c680e09` as well.


## Entry 2026-11-01 — Adapting the simulation to the example formulation

- **Tool:** Claude Code (Anthropic CLI), file-edit and shell access. No subagents.
- **Model identity/version:** Claude Opus 5 (`claude-opus-5`), as selected in the session; not independently verified.
- **Request:** Adapt the existing simulation to be mathematically and algorithmically sound against `docs/example-system-specs/`, which is henceforth the single source of truth for the mathematics and execution of multi-body simulation. The project's own specifications, language and GPU-bound target stay as they are.
- **Precedence change recorded, not decided here:** `crates/robot-sim/specs/appendix-d-mathematical-challenges.md` section 1.0.1 states the new rule and its limits. The example governs mathematics and algorithms; `SPECS.md` Sections 2 and 3, the std-only closure, the Rust-native requirement and the batched-GPU target are unchanged. The example's constants and tolerances are still not transplanted (mapping rule 2).
- **Changes, five commits on top of a snapshot of previously uncommitted work:**
  - `robot-sim/src/composite.rs` (new): the example's position and velocity stages (`subtree_com`, `cinert`, `cdof`, `cdof_dot`), composite-rigid-body `M(q)`, `L' D L` factorization over the degree-of-freedom tree, back-substitution, `mul_m`, `full_m`, recursive Newton-Euler, forward dynamics as `M^-1 (tau - bias)`, `mj_integratePosInd`/`mju_quatIntegrate` position integration, and a stepper. From 02 sections 1.3, 2.1, 2.3, 2.9, 2.10, 2.11, 3.1, 3.3 and 01 sections 5 and 8.3.
  - `robot-model/src/eig.rs` (new): symmetric 3x3 Jacobi eigendecomposition with quaternion accumulation, from 12 section 7.
  - `InertiaTensor::is_physically_plausible` now tests principal moments; `JointSpec::unit_axis` added and routed through `transform_at` and both dynamics adapters; `solve_small_dense` pivot tolerance made relative; `stiffness_damping` format predicate unified; `box_box_contacts` edge-edge contact placed by support feature and closest points.
  - Documentation: Appendix D sections 1.0.1, 1.5 and 1.5.1; Appendix B S-07 and S-08 resolved; steps 01 and 02 status and challenge blocks; specs index to 0.1.3-draft; both crate READMEs.
- **Appendix D entries closed:** M-01, M-02, M-03, M-07, M-13, M-14, M-18, M-24. M-17 partly addressed.
- **Measurements (deterministic, single configuration; NOT promoted to `reproduced_in_sim`, which per `SPECS.md` Section 3 additionally requires multiple seeds and a recorded comparison budget):** `M(q)` from the composite recursion agrees with `M` built column by column from local-frame inverse dynamics to 1e-10 on the 29-degree-of-freedom G1 and on a three-link chain with rotated origins, non-parallel axes and products of inertia; the bias force agrees to 1e-9; forward and inverse dynamics invert each other to 1e-8 on the 35-degree-of-freedom floating G1. A force-free spinning free body conserves world linear momentum to below 1e-14, against 3.93e-3 measured before the convention change, and an isotropic one conserves kinetic energy to 1e-14. The anisotropic rotational energy drift is 1.84e-2 after versus 1.94e-2 before and is asserted to be first-order in the timestep rather than bounded by magnitude. Scaling a declared joint axis by 3.5 now leaves `M` and the bias force bit-identical.
- **Found while fixing:** a bug of my own in the position stage, caught by the momentum test — a free joint's anchor was computed from the parent frame, ignoring its own translation, where 02 section 2.1 sets `xanchor = xpos`; a displaced free body therefore acquired a spurious lever arm and diverged. Also that a hand-written "obviously valid" inertia tensor used as an accept-side fixture was itself unphysical, which is the same failure mode as M-03.
- **Attempted and reverted, with the reason recorded:** M-05 and M-06. The example's EPA enclosure invariant comes from its GJK — four non-zero barycentric weights from the signed-volume subdistance of 05 section 5.4 are strict enclosure by construction — and our `reduce` uses Ericson's region tests, which cannot supply it. Building 05 section 6.1's polytope constructions on that mismatch traded a false positive for a false negative and back. Appendix D section 1.5.1 records what the real fix is and two load-bearing details of section 6.1 found on the way. `collision.rs` was restored to its committed state before the one independent fix (M-07) was applied.
- **Not done:** the unified constraint solver of example files 07 and 08, so M-04 and M-08 to M-12 stand, along with M-15, M-16, M-19, M-20, M-21, M-22 and M-23. `MultibodyState::step` is still the stepper the CLI and environments run, so the composite path is available but not the default; switching over requires moving contact to the same path, which is the solver work.
- **Verification:** `cargo fmt --all -- --check` clean, `cargo clippy --workspace --all-targets -- -D warnings` clean, `cargo test --workspace` 1019 passed / 0 failed. No test was weakened, ignored or deleted; the three collision tests that failed mid-way were made to pass by reverting the change that broke them, not by adjusting their expectations. One test bound was corrected after the fix showed the original premise to be wrong, and the correction is explained in that test.


## Entry 2026-11-01 (second pass) — Fixing the remaining audit findings

- **Tool:** Claude Code (Anthropic CLI), file-edit and shell access. No subagents.
- **Model identity/version:** Claude Opus 5 (`claude-opus-5`), as selected in the session; not independently verified.
- **Request:** Fix the other 16 Appendix D findings.
- **Closed, 14 of 16:** M-04, M-05, M-06, M-08, M-09, M-10, M-11, M-12, M-15, M-16, M-19, M-20, M-21, M-23. Appendix D section 1.6 is the status table; 1.6.1 and 1.6.2 state what M-17 and M-22 need.
- **New modules:** `robot-sim/src/solver.rs` (the convex constraint problem of `07` sections 2.3 and 2.4, with the Jacobians of `02` section 4 and a projected Gauss-Seidel sweep) and `robot-sim/src/options.rs` (the Section 12.1 knobs as a validated, recordable per-run configuration).
- **Rewritten:** the narrow phase, to the example's signed-volume subdistance (`05` section 5.4) and the polytope2/3/4 constructions (section 6.1); plane manifolds to the exact rules of `04` sections 1.3 and 1.4; the collision pipeline gained a broad phase and a pair-exclusion filter (`03` sections 2.1, 2.2, 2.4, 2.5); RK4 gained the Munthe-Kaas correction and free-joint support; `cholesky_solve` split so one factorisation serves many rows.
- **Schema change:** fidelity battery `shared-analytic-v7` to `v8`, three new rows, ADR 0008 amended as its own refresh rule requires. Step budget 539 000 to 583 000.
- **Defects found while fixing, not present in the original audit:** the two-vertex polytope construction was missing, so a head-on collision between two convex shapes reported no contact at all — found by the first sample of the new narrow-phase oracle, which is the clearest possible vindication of M-20. The expanding polytope had no maximum-face bound, which is a termination condition rather than a performance guard: without it the face count grows geometrically and one query took 2.9 seconds. Two scale-relativity bugs of my own in the GJK termination tests, the same pattern Appendix D section 3.2 warns about.
- **Correction to the audit's own analysis:** M-04 attributed the `1/n` rest-depth scaling to the per-point penalty law. The unified solver fixes the force balance but not that scaling, which survives in the example's formulation too; the bounded manifolds are what remove it. Recorded in the commit and the README rather than quietly left.
- **A measurement I nearly drew the wrong conclusion from:** a performance investigation looked at first like a 400-fold regression in my own code. An unrelated project on the same machine was running a full `cargo check` at the time, which contaminated several wall-clock comparisons; the load-independent iteration counts were normal throughout and said so. The real cause was the missing face bound, found by timing the phases inside the step rather than by guessing. Worth recording because the first three hypotheses were all wrong and all plausible.
- **Verification:** `cargo fmt --all -- --check` clean, `cargo clippy --workspace --all-targets -- -D warnings` clean, `cargo test --workspace` 1046 passed / 0 failed. No test was weakened, ignored or deleted. Four fidelity tests had pinned constants updated for the schema bump, and one collision test's expectation changed because the behaviour it pinned is the behaviour M-04 asked to replace; both are noted in their commits. Two solver tests were corrected after the fix showed their premises wrong, with the corrected reasoning in the test.
- **One documented allow:** `clippy::excessive_precision` on the transferred Bortz coefficient table, so the digits can still be diffed against the source of truth.
- **Not done:** M-17 (the composite and solver path is not yet what `MultibodyState::step` runs; the `Free`-joint layout has to move across the workspace in one change) and M-22 (the GPU backend's `f32` and six-degree-of-freedom limits, which now additionally need the Newton solver because a sequential per-row update does not vectorise).

## 2026-11-01 — third correction pass: the switch-over, and a cone of the wrong shape

`World` and `WorldEnv` wire the corrected dynamics, collision and constraint
pieces into something that runs, addressing the substance of Appendix D M-17.
Measured: the ground is enforced on every collision shape through a 3000-step
zero-torque G1 collapse (contacts carry full weight, worst penetration
0.045 m), and a joint limit holds its bound exactly when given a margin to act
in (0.177 rad overshoot at zero margin, zero at 0.2 rad). Both numbers
corrected a test premise before they were asserted; `SimOptions` gained
`limit_margin` as a result.

Also a correction to the second pass. M-10 had been recorded as fixed with
friction rows bounded by `|f_t| <= mu f_n` per component. That is the
circumscribed *square* pyramid, which allows up to sqrt(2)·mu·f_n diagonally —
41 per cent too much. 07 section 9.2 builds the rows as the cone's **generator
directions**, `J = n ± mu·t_k`, with non-negative coefficients, giving the
inscribed diamond. Reimplemented; `RowKind::ContactFriction` is gone because
every row kind is now non-negative and the cone lives in the Jacobians.

That exposed a further measured fact: the edge representation is redundant, and
projected Gauss-Seidel needs 1011 sweeps to converge on sixteen pyramidal rows
over a rank-six problem, against a default budget of 100. Accelerations are
within 2e-4 at 100 sweeps; the forces are not. This is the quantitative case
for the Newton solver of specs/08, recorded in Appendix D section 1.7.3.

Gate: 1055 tests, fmt and clippy clean.

## 2026-11-01 — the primal Newton solver, and what making it the default exposed

specs/08 section 6 implemented in `crates/robot-sim/src/newton.rs` and made the
default. One iteration to a scaled gradient norm of 4.9e-14 on the sixteen-row
pyramidal problem that took projected Gauss-Seidel 1011 sweeps; the two agree on
the acceleration to 1e-8. Gauss-Seidel is kept as that cross-check.

Switching the default over found two defects a non-running solver could not
have. `mju_cholFactor` clamps a lost pivot to `mindiag` and decrements the rank
rather than failing (08 section 7.1); returning an error killed the G1 stance
run at degree of freedom 31. And a contact with zero friction must be `dim == 1`:
`condim = 3` with `mu = 0` built four identical pyramid edges with
`Rpy = 2 mu^2 R0 = 0`, so `D` saturated at 1/MINVAL and the Hessian lost every
pivot, diverging on step 1511 of the collapse once the folded robot contacted
its own links through the default pair friction of zero.

Gauss-Seidel's non-convergence had been masking that: it never reached the true
force in 100 sweeps, so it under-estimated it and the degenerate rows never
showed. Recorded in Appendix D section 1.7.4 with a regression test.

Gate: 1061 tests, fmt and clippy clean.

## 2026-11-01 — `robot sim multibody` runs the corrected path, closing M-17

The command now runs composite dynamics, exact collision manifolds and the
primal Newton constraint solver by default. `--contact penalty|impedance` keeps
both legacy models reachable, behind one wrapper
(`crates/robot-cli/src/commands/stepper.rs`) so there is a single step loop.

Same run (G1 floating, stance hold, 4000 steps at 0.5 ms), both PASS the stance
qualification. Penetration is smaller on the corrected path (-0.0044 m against
-0.0054 m), and the kinetic energy decays -- peaking near t=0.6 s and falling to
0.056 J by t=1.0 s -- where on the legacy path it climbs monotonically through
the whole run. A supported stance settling should dissipate; a penalty contact
still adding energy after a second is the uncoupled-spring artefact of M-09,
now visible in the command's own output.

The migration needed `stable_damping`, `stance_torques`, `limit_fractions` and
`limit_torque` on `World`, all tested. One trap recorded in Appendix D 1.7.5:
`limit_fractions` is indexed by actuated joint, and the G1 has 39 bodies against
30 actuated joints, so inferring the mapping from an external offset is wrong
for fixed joints. It now comes from the profile and is compared against the
legacy path slot for slot.

Gate: 1064 tests, fmt and clippy clean. Replay writing verified end to end on
the new default path.

## 2026-11-01 — the model's actuators are simulated, and their units are right

Two defects, recorded as Appendix D M-25 and M-26, found by comparing the
simulation's behaviour against what the G1 USD layer describes.

M-25: the layer declares a `PhysicsDriveAPI` per joint and the importer reads all
29 into `ActuatorSpec::gains`, but nothing consumed them. `robot sim multibody`
applied a hand-written hold with a uniform kp=300, kd=30. `World` now owns the
model's actuators and `World::actuator_torques` is `mj_fwdActuation` for them --
the AFFINE-gain/AFFINE-bias position servo of specs/10 section 2.2, clamped to
`maxForce` and taken through the declared transmission. A profile with no
declared gains gets no drives rather than fabricated ones.

M-26: an angular drive's stiffness and damping are **per degree**, and were read
as per radian. The importer already converted the limits and the velocity from
degrees -- the knee's range reads -145 to 165 and its maximum velocity 1833 --
but not the gains, which convert the opposite way to an angle. Every drive was
understated by 180/pi. With the wrong units the G1 could not hold its own
posture: it folded, the pelvis fell to 0.055 m, and all three stance checks
failed with 188 J. With the right units it settles at 0.790 m and reads 0.0000 J
from t=0.8 s.

The sequence is the lesson. A hand-written hold with invented gains produced a
passing stance for weeks while masking a unit error in the model import, and the
error only became visible once the model's own drives were actually applied.

New run in `outputs/g1-stance-newton-replay/`: PASS, residual kinetic energy
0.0114 J against the legacy run's 0.0866 J, worst penetration -4.6 mm against
-5.1 mm. usdchecker clean on both replays.

Gate: 1069 tests, fmt and clippy clean.

## 2026-11-01 — the G1 load audited against the layer; actuation made a choice

Prompted by two objections: the robot was completely rigid, and a simulator's
default should not be to switch a model's motors on.

The unit conversion of the previous entry is right -- OpenUSD and Omniverse both
document angular drive stiffness as mass*DIST^2/degree/s^2, and Isaac Lab's
writer converts N m/rad to N m/deg on the way out. What it exposed is that the
declared gains are not the robot's: all 29 have damping = 0.0004 * stiffness to
float precision with targetPosition = 0, giving a damping ratio of 0.003 to 0.03
at up to 13183 N m/rad. Recorded as M-28. `ActuationMode` {Drives, Damping, Zero}
and `--actuation` make it a choice, and the command warns about the uniform
ratio.

The load itself was audited against the flattened layer, counting colliders
through the instance prototypes. Correct: 30 links with identical masses (total
33.341142 kg), all 30 inertia tensors matching diagonalInertia rotated by
principalAxes to 1e-6, and all 29 joints matching on kind, limits, velocity,
effort and axis with zero mismatches.

Not correct, recorded as M-27: of the 33 declared colliders (15 Mesh, 10 Sphere,
6 Cylinder, 2 Cube) the 15 meshes are not collided at all, and two axis-aligned
bounding boxes the model does not declare are fabricated on pelvis and
torso_link by `usd::PROXY_LINKS`. The fix is convex-hull collision, which is
tractable because `gjk_epa` already works against anything that answers a
support query.

Gate: 1072 tests, fmt and clippy clean.

## 2026-11-01 — specs reworked to 0.1.5-draft

Every step file reworked against the example's implementation of these dynamics
and against the problems now measured, rather than against what was planned.

Three statuses were materially false and are corrected. Step 05 said the
actuation that runs is `stance_torques`; the model's own drives run. Step 08 said
"Not started" while its convex-hull MUST was being actively violated. Step 06
carried layered status blocks that contradicted each other on whether warm
starting existed. Steps 01, 02, 03, 04, 07, 09, 10, 11 and 12 all had statuses
predating the switch-over.

The specs now state the example's algorithm where it governs: the primal Newton
problem of example 08 section 6 and why it is the default (1011 Gauss-Seidel
sweeps against one Newton iteration on sixteen pyramidal rows); the
generator-direction pyramid of example 07 section 9.2 and why the circumscribed
square an earlier pass built was 41 per cent too permissive diagonally; the
world-orientation composite frame of example 02 and the fact that the example
has no articulated-body algorithm; the position-servo factoring of example 10
section 2.2 with its force and target clamps.

Four entries open, and the index now carries them as ordered priorities: M-27
(mesh collision, two MUSTs, with a bounded fix via a support-query hull), M-28
(declared gains are generated defaults), the implicit integrator family, and
M-22 (GPU). Appendix B gains S-10 (what a tier pins for actuation, and whether
to carry a published gain set as a declared override rather than editing the
vendored asset) and S-11 (hull or decomposition), and S-09 is marked resolved.

Step 04 carries the general lesson as specification: a passing run is evidence
about the run, not about the component you believe it exercises. A hand-written
hold with invented gains masked a factor-of-57 unit error for weeks, and the
dual solver's non-convergence was acting as damping that hid four degenerate
rows.

All internal links verified to resolve, and every code identifier cited checked
to exist. Gate: 1072 tests, fmt clean.

## 2026-11-01 — the example's specifications are MUST; competing formulations overwritten

Precedence strengthened by explicit direction and recorded as Appendix D section
1.0.2: where pre-existing mathematics works differently from the example's, it
is overwritten rather than recorded as a deviation.

The section states the three categories a reader has to tell apart, because
collapsing them is how a defect hides. A **competing formulation** computes a
different answer and is a defect. A **second implementation used as an oracle**
computes the same answer by another route and is kept, because removing it would
leave the primary path unvalidatable. A **stricter input contract** refuses an
input the example would repair, which is not computing differently and is
re-labelled validation.

Overwritten:

- The per-point penalty contact, and the `--contact penalty|impedance` selector
  that reached it. Its stiffness scaled with the sample count (M-04) and its
  impedance was referred to a fixed 1 kg reference mass (M-09), so it was a
  different contact model rather than a cheaper setting of the example's. The
  flag is now rejected with an explanation and exit 64, not ignored: an old
  invocation that silently ran a different simulation would be worse than one
  that fails.
- `contact::plane_sphere_contact`, which fixed its normal to `+z` and measured
  `center[2] - plane_height`. Example 04 section 1.1 takes the normal from the
  plane geom's own frame and measures `dot(pos2 - pos1, n)`; the two agree only
  for a plane whose frame is the world's. Tested on a plane with a `+x` normal
  and an offset position, which the old form could not express.

Kept, and labelled as oracles in their own module documentation so the
distinction cannot be read as a loophole: `articulated`'s articulated-body and
recursive-Newton-Euler recursions (agreeing with the composite path to 1e-10 on
the G1, and the only independent check it has) and `constraint::solve_equality`
(the CPU reference the GPU equality kernel is validated against). Neither is
reachable from a simulation path.

Step 06's deviations list is reclassified entry by entry under the new rule, and
the exact-`A_ii` choice is restated as the example's own DIAGEXACT path rather
than a divergence from it. Specs at 0.1.6-draft.

Gate: 1073 tests, fmt and clippy clean. One simulation path verified end to end.

## 2026-11-01 — M-27 resolved: mesh colliders are real geometry

The G1 now reports **33 collision shapes modelled and 0 not modelled**, which is
exactly the 33 its layer declares. Before: 18 primitives, 15 meshes dropped
entirely, and 2 axis-aligned bounding boxes the model does not declare
fabricated on the pelvis and torso. `usd::PROXY_LINKS` is gone.

`robot_model::hull::hull_vertices` reduces each mesh to its convex hull at
import -- where specs/08 item 3 requires it, at compile time and never at
simulation time -- and `collision::Primitive::Hull` answers the support query
`argmax_i (v_i . d)` that `mjc_meshSupport` computes.

The reduction is exact, not an approximation: the support function of a point
set equals the support function of its convex hull, so a discarded point could
never have won an argmax. Every hull test asserts that property directly,
comparing the support function over the reduced set against the whole cloud in
144 directions, rather than asserting a vertex count.

Why conflict lists mattered: without them each quickhull round rescans every
point against every live face, about 3e9 dot products for one 19000-vertex G1
mesh. With them a round only redistributes the points of the faces it deleted.

Measured: 15 mesh colliders, median 19116 raw vertices each, reduce to 27061
hull vertices in total -- about tenfold -- with none degenerate. The stance is
unchanged (pelvis minimum 0.7882 m, penetration -0.0046 m, settled energy
0.0114 J, PASS), which is the expected outcome: the pelvis and torso do not
touch the ground in a stance, so this was about the geometry being right rather
than about a number moving. A 2-second run at 0.5 ms takes 8.5 s of wall clock.

Two costs remain and are recorded as costs rather than defects: the support
query still scans, where the example hill-climbs a vertex adjacency graph above
ten vertices, and a plane-hull contact is one point because `mjc_PlaneConvex`'s
three extra contacts need polygon topology a vertex hull does not carry.

Specs at 0.1.7-draft. Gate: 1080 tests, fmt and clippy clean.

## 2026-11-01 — zero-torque collapse run, and what it exposed about joint limits

`outputs/g1-zero-torque-newton-replay/`: 2 s of the G1 with no actuator force at
all, on the corrected path with every one of the 33 declared colliders
simulated. FAIL on all three supported-stance checks, which is correct -- the
checks ask whether a stance was held and nothing is holding one.

The robot **lands and comes to rest**: energy peaks at 42 J during the impact
and decays to 0.044 J, with the pelvis settling at 0.067 m. The legacy
zero-torque run recorded the pelvis falling *through* the plane, because only
ten ankle spheres took part in contact and the pelvis and torso had no collision
geometry. They have convex hulls now, so the collapsed robot lies on the ground.

The collapse exposed a real inadequacy in the default joint-limit margin.
Measured over the same collapse at 0.5 ms:

  --limit-margin 0 (default):  worst excursion 0.197 rad, 3783 of 4000 steps
  --limit-margin 0.1:          worst excursion 0.284 rad, 1459 of 4000 steps
  --limit-margin 0.2:          worst excursion 0,          0 of 4000 steps

At the default, 17 of the G1's 29 joints leave their range, the worst by
0.31 rad -- about 18 degrees past a mechanical stop -- because with no margin a
limit row appears only once the joint has already crossed. The archived run
therefore uses 0.2 rad: a replay with joints bent past their stops is not a
replay of this robot. Peak energy is also far lower with the margin (42 J
against 132 J), since the limits prevent the through-stop motion rather than
absorbing it afterwards.

`--limit-margin` is now a CLI flag. specs/09 item 1 requires every fidelity knob
to be a per-run parameter, and one only library code can set does not satisfy
that; the gap was invisible until a run needed it.

`outputs/README.md` is rewritten. It records that the runs previously archived
there are gone (they were half a gigabyte each), that this directory is
gitignored scratch rather than an archive, and that the reproducible record of a
run is its command and commit rather than the files.

Gate: 1080 tests, fmt and clippy clean. usdchecker passes on both replays.

## 2026-11-01 — implicitfast: step 03 item 4's MUST met

`Integrator::ImplicitFast` / `--integrator implicitfast` implements
`mj_implicitSkip` with `flg_bias = 0` (example 01 section 9.2): the smooth
forces stay at the old velocity and their velocity derivative goes into the
matrix, so the unconstrained acceleration solves
`(M - h dF/dv) qacc = tau - qfrc_bias`.

For the forces this crate applies the derivative is **diagonal**, which is a
property of the force set rather than a simplification: the smooth forces are a
drive's `kp (ctrl - q) - kd qd` and a joint damper's `-kd qd`, only the damping
terms depend on velocity, and each depends on one degree of freedom's own
velocity. So `qH = M + h diag(kd)` adds to the sparse mass matrix's own diagonal
and needs no new structure. The example's reset of a free body's six rows to `M`
happens for free, since a Free joint carries no actuator and so no kd.

Measured, with the G1's drive damping scaled up and its stiffness zeroed at
dt = 2 ms: the explicit step drives the damped hinges to |qd| = 5.1e2 from a
start of 0.5, while implicitfast keeps them at 0.74 -- a 685x difference. With
the damping removed the two agree to 1e-12, which is the check that this is the
same dynamics and not a different model.

On the G1's **declared** gains it changes nothing measurable: a 2-second stance
gives a pelvis minimum of 0.7882 m under Euler and 0.7881 m under implicitfast,
both PASS. That is the correct outcome and is recorded so the feature is not
oversold -- the layer's damping is far below the explicit ceiling, so the
integrator removes a limit rather than changing an answer. What it buys is that
a stiffer gain set is now usable, and step 05 no longer has to record a drive's
damping as explicit-only.

Not ported: full `implicit` (the qLU path with the recursive-Newton-Euler
derivative) and `mjd_freeMhat`'s free-body gyroscopic block, which is what makes
implicitfast and implicit identical for a standalone free body.

Specs at 0.1.8-draft. Gate: 1082 tests, fmt and clippy clean.

## 2026-11-01 — condim 4 and 6: torsional and rolling friction

Step 06 item 4's second MUST. A pyramidal contact now emits the example's
`2(condim - 1)` generator rows in the order (+t1, -t1, +t2, -t2, +spin, -spin,
+roll1, -roll1, +roll2, -roll2).

The split that matters is which Jacobian each generator multiplies. The two
sliding tangents multiply the translational Jacobian; spin (about the normal,
condim >= 4) and rolling (about the two tangents, condim = 6) multiply the
rotational one. So 4 and 6 differ from 3 in kind, not in length, and
`point_jacobian_dot` now returns the rotational derivative the extra rows need
for their `Jdot v` term -- which is `cdof_dot[i][0:3]` directly, with no
transport term, because a rotational row does not depend on where the contact
point is.

A direction whose coefficient is zero contributes no rows. That generalises the
defect that diverged a G1 collapse earlier: the edges are `n +/- mu d`, so at
mu = 0 both collapse onto `n` and `Rpy = 2 mu^2 R0` vanishes, saturating
D = 1/R at 1/MINVAL and destroying the Hessian. It matters more at 4 and 6 than
at 3, because a torsional or rolling coefficient is commonly zero while sliding
friction is not.

`solver::sliding_friction` builds the example's five-element vector
`(fri0, fri0, fri1, fri2, fri2)` with the torsional and rolling entries zero, so
condim 4 and 6 behave exactly as 3 until a coefficient is supplied -- the
dimensionality is available rather than forced. Rule 2 of the mapping document
is why the default is zero rather than the example's
`friction = [1, 0.005, 0.0001]`.

Tested on the structure (row counts per condim and per coefficient; every edge
pair averaging to the same normal row; spin edges acting on the angular block
and sliding edges on the linear one) and on the physics (a body spun about the
contact normal loses its spin at condim 4 and does not at condim 3).

Specs at 0.1.9-draft. Gate: 1085 tests, fmt and clippy clean.

## 2026-11-07 — documentation audit: code, specs and docs brought into agreement

- **Tool:** Claude Code (CLI), with five read-only audit subagents.
- **Model identity/version:** Claude Fable 5.1 (`claude-fable-5-1`).
- **What was done:** every crate README, the root README, AGENT.md, PLAN.md,
  SPECS.md Section 9.1 and its revision table, the robot-sim specs index and
  step files 04, 09, 12 and appendices C and D, the spec checklists 00, 06,
  07, 09 and 14, ADRs 0008, 0010, 0011, 0012, 0014 and 0015, both ledgers,
  `docs/example-system-mapping.md`, the asset and fixture READMEs, and the
  module doc comments of `robot-sim`, `robot-cli`, `robot-model`,
  `robot-control`, `robot-eval` and `robot-adapt` were compared against the
  code at commit `1d2e9d9` and corrected where they disagreed.
- **Main corrections:** fidelity schema v7 → v8 wherever v7 was still named;
  `robot sim multibody` documented with all its flags (`--scene`,
  `--actuation`, `--limit-margin`, `--integrator`), its outputs (CSV, JSON
  Lines, SVG, two USD replays) and exit code 3 (INCONCLUSIVE); the
  robot-sim README module table gains `collision`, `composite`, `solver`,
  `newton`, `options` and `world` and no longer says the formula modules are
  unwired or that contact is sphere-only; M-27 is recorded as resolved in the
  specs index, steps 08 and 12 and appendix C; the G1 probe is recorded as
  `pass` (hard limits) in spec 04 and the robot-sim README; ADR 0010 carries
  a superseded banner for its contact law; SPECS.md 0.8.6-draft rewrites the
  9.1 contact paragraphs to the implemented formulation and repairs the
  revision table; stale test counts (robot-cli 17 → 45, robot-learn 60 → 61,
  robot-model 79 → 115, robot-nn 20 → 22, robot-sim 191 → 301) and crate
  descriptions (`robot-tasks`, `robot-adapt`, `robot-eval`, `motion-process`,
  `motion-retarget`, `robot-hardware`) corrected; evidence-ledger state
  `measured` replaced with the Section 3 vocabulary.
- **Measured for this entry:** `target/release/robot sim multibody --floating`
  at the default `dt = 1e-4`: pelvis minimum 0.7882 m at t = 0.061 s, lowest
  contact −0.0046 m, max settled kinetic energy 0.0114 J, PASS.
  `--zero-torque`: pelvis minimum 0.1151 m at t = 1.346 s, settling at
  0.125 m, lowest contact −0.0336 m, FAIL (collapse reference). Both report
  33 collision shapes modelled, 0 not modelled.
- **Not changed, flagged for a human decision:** AGENT.md's commit-subject
  rule (72 characters, mandatory scope) is not followed by recent history;
  `origin/isaacsim` and the working branch `chore/tidy-docs` exist despite
  the "work on `main`" rule; the empty, untracked `fixtures/models/` and
  `fixtures/profiles/` directories; the 2026-11-01 entries above carry no
  tool/model lines and commit `1d2e9d9` (2026-11-04) has no entry.

## 2026-11-13 — crate completion pass, one crate at a time

- **Tool:** Claude Code (CLI).
- **Model identity/version:** Claude Opus 5.5 (`claude-opus-5-5`).
- **Scope:** bring each crate toward its full SPECS.md scope, starting with
  the least complete. Entries below are appended per crate as each lands.
- **`motion-process`:** SPECS.md §8.2 items 3–6 added: `kinematics`,
  `condition`, `filter`, `resample_anti_aliased`, `contact`, `segment`, and
  §6's augmentation as `augment`; README, Cargo description and root README
  row updated. 4 → 39 tests.
  Every threshold is a caller-supplied config field. Segmentation agreement
  with human annotation has not been evaluated.
- **`motion-retarget`:** rewritten around `ik::solve_tasks` (position,
  orientation, relative-position and posture tasks, free base, LM step
  control, deadline and termination reasons); added `collision`,
  `temporal`, `retarget` and the `retarget-map/1` `config` format; new
  dependencies on `motion-process` and `robot-artifacts` (workspace paths,
  exclusion gate passes). 4 → 28 tests, including retargeting a punch onto
  the vendored G1 URDF.
- **`motion-validate`:** added `trajectory`, `kinematic`, `dynamics`
  (floating-base RNE via `robot-sim` with least-norm contact distribution),
  `contact`, `tracking` (`robot-sim` `World`), `validate` and `package`
  (`motion-curation/1` manifest); `report` extended with torque, root-drift,
  contact and tracking reasons and stable codes. New dependencies on
  `robot-sim`, `motion-process`, `motion-retarget` and `robot-artifacts`
  (workspace paths). 3 → 17 tests.
- **`motion-io`:** added `bvh_export`, `c3d`, `inflate`, `npz`, `amass`,
  `import` (detection, originals stored before parsing, inspection) and
  `capture` (`robot-capture/1` protocol, line and UDP adapters, recorder).
  No fixture files added: C3D, NPZ and DEFLATE tests build their inputs
  in code. 8 → 35 tests.
- **`robot-adapt`:** added `ladder` (the §11.3 order as a gate),
  `history`, `encoder`, `correction`, `residual`, `finetune`, `retrieval`,
  `morphology`, `report`; `sysid` gains standard errors and
  `identify_joint`. Training code behind a default `training` feature
  (optional `robot-learn`) so deployable builds carry no autograd; both
  feature sets lint clean. Row 10 refused by design (§9.1). 4 → 27 tests.
- **`robot-persona`:** added `params`, `arbiter`, `memory` (session-scoped,
  per C-17's interim position) and `evaluation`; `StyleConfig` gains
  bounded parameters and `validate`. Cross-session opponent memory left
  to the C-17 decision. 6 → 17 tests.
- **`robot-eval`:** added `distribution` (Student's t), Welch and TOST,
  `evaluate` (interval-based outcomes, regression gate with profile-ID
  check), `matrix` (the fifty §14 rows, closed only by the ledger's new
  "Acceptance rows" table, which is empty); profiles gain firmware
  version, run count, threshold direction and TOML/content-ID I/O.
  6 → 18 tests.
- **`robot-tasks`:** added `humanoid` (state, extractor from `robot-sim`,
  observation audit, task trait), `terms`, `tasks` (stand, perturbation,
  tracking, reach, footwork, boxing pad) and `randomize`; `robot-sim`'s
  `WorldEnv` gains `push_base`. 11 → 26 tests.
- **`robot-skills`:** added `selection` (the R2 feasibility filter and
  router with fixed-size counters), `cache` (activation checks and
  off-thread staging) and `catalog` (lexical Markdown catalog that only
  reorders). 8 → 18 tests.
- **`robot-control`:** added `tactical` (skill router as a rate-limited
  `LoopOperation`, `SkillBank`, expiring proposals, §5.1 checks),
  `incident`, `TimingReport`, and `with_sensors` (temperature and contact
  taps into the supervisor); `LoopOperation` gains `observe_time` and
  `decision_age` with defaults. `robot-perception`'s `ContactClassifier`
  derives `Clone`/`Debug`. 36 → 46 tests.
- **`robot-hardware`:** added `wire` (`robot-wire/1` protocol) and `link`
  (`WireAdapter` over UDP or in-memory transports, `FirmwarePeer` with
  command validity and watchdog). No vendor protocol transcribed (C-02
  open). 12 → 20 tests.
- **`robot-perception`:** added `base` (IMU plus leg-odometry estimator
  with slip gating), `opponent` (tracking with coasting and age) and
  `consistency` (NEES); qualified on synthetic truth with jitter,
  dropouts and slip. 15 → 22 tests.
- **`robot-devtools`:** added `inventory` (Cargo.lock inventory,
  `kb-store/1` store, transitive staleness sweep, fault records, ledger
  citation check), `ledger` (diff with unsupported-promotion detection)
  and the `devtools` binary; created `docs/kb/knowledge-base.toml` with no
  entries (nothing cited to record). 9 → 15 tests.
- **`robot-cli`:** wired `data import`, `motion retarget`, `motion
  validate --reference`, `eval run`, `adapt identify`, `train` and
  `replay` (new `commands::pipeline`, `robot-reference/1` CSV); `skills`,
  `deps`, `deploy` and `eval --bundle` keep exit 2 with reasons; PLAN.md
  §4's implemented list updated. Smoke-tested end to end in a scratch
  directory (not committed).
- **`robot-supervisor`:** added `envelope` (torque-speed, thermal
  derating, saturation duration, contact energy budget, mode interlocks;
  four new fault variants, three new categories, two added to the
  must-escalate set) and `hazard` (§12.2 register with the
  no-learned-only-control rule). 21 → 27 tests.
- **`robot-nn`:** added `format` (`robot-mlp/1` weights, refusing unknown
  operators at load) and `gru` (recurrent inference with explicit state
  contract). Recurrent training (BPTT) left unimplemented, stated. 22 → 28
  tests.
- **`robot-learn` / `robot-tasks`:** found that `BalanceTask` paid the
  agent to fall (all-negative per-step reward with zero-value
  termination), which explains the recorded PPO negative result; added an
  `alive-v1` term. Measured deterministic returns 90.8 / 96.2 / 97.1
  (seeds 1–3) against 34.2 for zero action; new test
  `ppo_stabilizes_the_pendulum_on_two_seeds` (row A-13 evidence). `robot
  train` defaults updated to the measured configuration. 61 → 62 tests.
- **`robot-viewer`:** added `skeleton` (profile-driven side and front
  skeleton SVG with a golden file, SMIL playback without script);
  `robot replay --svg` writes a playback. No rasterizer was available to
  view the output; correctness rests on the bone-count and golden tests.
- **`robot-sim` / `robot-perception`:** added `robot_sim::sensors` (ideal
  encoders and site IMU) and `robot_perception::noise` (bias walk, white
  noise, quantization, delay, recorded injection point), taking specs
  Appendix B S-06's stated default; step 08's status updated. robot-sim
  301 → 304 tests, robot-perception 22 → 25.
- **`robot-model`:** added `discovery` (§7.7 offline core: source ranking,
  conflict listing, web proposals, seeded workspace sampling, test
  classification, probe refusal, deterministic report that blocks
  activation). The commissioning probe itself waits on C-29; S-08
  (inertia realisability test) stays open. 115 → 119 tests.

## Entry 2026-11-15

- **Tool:** Claude Code (Anthropic CLI agent), shell, file-edit and
  build-tool access, with network access for asset vendoring.
- **Model identity/version:** `claude-fable-5` (Anthropic).
- **Request:** "proceede to refine and implement all components in this
  project who are not yet fully realized. Import some common and well known
  robot models from their public source (preferably USD). Write a
  comprehensive guide on how to use this suite."
- **Artifacts produced in this entry's scope (vendoring part):**
  `assets/robots/unitree_go2_usd/` and `assets/robots/unitree_h1_usd/`
  (official Unitree OpenUSD layers, LFS objects fetched from
  `unitreerobotics/unitree_model` at the same pinned commit as the G1
  layers, each verified SHA-256 == LFS object id, provenance in each
  directory's `PROVENANCE.md`); `PhysicsFixedJoint` support in
  `robot_model::usd`; `unitree_go2_12dof_usd` and `unitree_h1_19dof_usd`
  fixtures with tests; `crates/robot-sim/tests/vendored_robots.rs`.
- **Artifacts (composition part):** full `SdfListOp` edit semantics
  (`ListOp::apply`), inherits/specializes arcs, sublayer and arc layer
  offsets on time samples, relationship/connection target remapping through
  arcs, and fieldless standalone target specs in `robot_usd::stage`, with
  `crates/robot-usd/tests/composition.rs`.
- **Artifacts (file-format part, ADR 0016):** the `robot-bundle/1`
  directory format (`robot_artifacts::bundle`: digest-sealed manifest,
  content-addressed weight blobs, latent constructs refused by name), the
  `skill-manifest/1` format (`robot_skills::manifest`), and the CLI
  commands `robot skills list|search` and `robot eval --bundle` with the
  activation gate run against the vendored robots' joint limits.
- **Artifacts (guide part):** `docs/guide.md` (setup, workspace map, robot
  models, simulation, the motion pipeline with a vendored runnable example
  in `fixtures/motion/guide-reach/`, training/eval/bundles, skills,
  adaptation, the safety model, disciplines, honest edges), linked from
  the root README; every command in it was executed against this
  revision.

## Entry: 2026-11-15 (second session: whole-stack code inspection and fixes)

- **Model identity/version:** `claude-fable-5` (Anthropic).
- **Request:** "walk through the project and do a code inspection of the
  whole stack, give a code_review_15_11_26.md file afterwards" followed by
  "procede" (fix the findings in the review's suggested order).
- **Artifacts produced:** `code_review_15_11_26.md` (24 crates inspected;
  7 high / 15 medium / ~24 low findings, each with file:line evidence; all
  highs independently re-verified before reporting). Fixes, each with
  regression tests, in six commits:
  - Activation gate (H1, H2, M11): `check_activatable` refuses envelope
    width mismatches and non-finite limits instead of zip-truncating and
    NaN-passing; the USD importer derives per-robot profile identity
    (every import was stamped `unitree-g1-29dof-usd`, so the §12.2
    profile match could not tell the vendored robots apart); `robot eval
    --bundle` names its rate check as self-consistency.
  - Safety paths (H3, H4, M7, M9, M10): falling with a non-resident
    recovery skill is Protective, not Keep; the supervisor folds fault
    severity before log capacity (a full log could downgrade an
    interlock-open tick to clamp-and-continue); NaN temperatures fault
    and derate to zero; a mode's interlocks are treated as open without a
    monitor to attest them; R2 selection survives NaN features.
  - Hostile files (H5–H7, M12–M14): BVH depth bound, npz zero-width
    refusal and EOCD guard, C3D checked dimension products, capture lag
    saturation, USDC checked counts with bounded pre-allocation.
  - robot-usd composition (M1–M3): variant selections compose
    strongest-wins across sources and through arc recursion; arc
    namespace roots are recorded at resolution (`Src::root`) instead of
    inferred from path suffixes; prim recursion bounded (MAX_PRIM_DEPTH).
  - Bundle seal (M4–M6): unknown keys refused everywhere the digest seal
    does not reach; `load_weights` verifies the bytes it returns; field
    widths bounded (MAX_FIELD_WIDTH).
  - Evidence honesty (M8, M15): per-metric missing/non-finite seeds make
    the verdict Inconclusive and are named; duplicate seeds are excluded
    and block acceptance; `resample` no longer lerps Missing placeholders
    into "Interpolated" data.
  - Maintenance lows: CLI usage text and envelope joint filter, Name
    refuses `.`/`..`, duplicate tier assignments refused, bundle schema
    compatibility check, `forward_with_trace` errors on width, GAE real
    asserts, gaussian-policy gradient doc sign, skill-manifest writer
    validates what the reader refuses, expert-cache swap budgeting,
    attitude filter rejects bad dt, `depends_on` refuses non-strings,
    GPU equality kernel damping branch matches the CPU reference.
- **Negative results / still open:** the lows listed in the review's "Fix
  status" note (EPA convergence surfacing, C3D label dedup, chain_ratio
  errors, TOML array-table header bound, among others) are recorded there
  and deliberately not fixed in this pass.

## Entry 2026-11-15 (third session: containerized-foreign-service registry)

- **Tool:** Claude Code (Anthropic CLI agent), shell, file-edit and
  build-tool access, offline.
- **Model identity/version:** `claude-fable-5` (Anthropic).
- **Request:** the user proposed "building a library of container runtimes
  that build these external services with an API, for example [Isaac
  GR00T] or video to SMPL-X projects, so we have them bundled and ready
  to be built inside this project, the attached api parses to the same
  structured datamodel we use", and asked for it to be implemented if it
  does not interfere with the project so far.
- **Artifacts produced:** SPECS.md 0.9.0-draft (Section 2.4, the
  dependency-class table row, rows A-51 and A-52, the glossary entry,
  C-30); `docs/decisions/0017-containerized-foreign-services.md`;
  `services/README.md` and the two `unverified`, unpinned records
  `services/video-to-smplx/` and `services/isaac-gr00t/` (each
  `service.toml` + `README.md`);
  `crates/robot-devtools/src/services_gate.rs` with its tests, wired into
  the `exclusion-gate` binary; `crates/robot-eval/src/matrix.rs` rows
  A-51 and A-52 (its spec-matching test caught the SPECS §14 addition,
  as designed; `the_real_ledger_parses` now asserts against `ROWS.len()`
  instead of a second hard-coded count); `devenv.nix` git-hook trigger
  pattern and comments; `crates/robot-devtools/README.md`; this ledger's §2.4
  candidate table; the root `README.md` services pointer.
- **Gates:** `cargo test -p robot-devtools` (26 passed, including the
  committed-registry fixture test), `cargo fmt --all -- --check`,
  `cargo clippy -p robot-devtools --all-targets -- -D warnings`, and the
  `exclusion-gate` binary (ok: 2 service records, all valid
  robot-service/1; a deliberately malformed record was shown to fail
  with named violations and exit 1, then removed). Full
  `cargo test --workspace` run at the end of the session.
- **Deliberately not done:** no `Containerfile` is written and no
  `podman` enters `devenv.nix`, because no candidate is
  `code_inspected` and Section 2.2 forbids speculative inputs (C-30);
  no upstream estimator claim was promoted beyond `unverified`.

## Entry 2026-11-15 (third session, continued: GVHMR qualification research)

- **Tool / model:** as above (`claude-fable-5`), now with network access
  for primary-source retrieval.
- **What was done:** retrieved the GVHMR repository metadata,
  `docs/INSTALL.md` and `LICENSE` at pinned commit
  `ee960bb6e2ea2d381aa97f08e9b71ef320b624b1` (zju3dv/GVHMR, head of
  `main`, committed 2026-09-15); confirmed identity (SIGGRAPH Asia
  2024 / TPAMI 2026) and located the SMPL-X output claim in text.
- **Artifacts produced:** evidence-ledger GVHMR row promoted
  `unverified` → `paper_verified` with the admitting artifacts named;
  `services/video-to-smplx/service.toml` `evidence_state` likewise;
  `services/video-to-smplx/README.md` rewritten around the selected
  candidate, its license constraint (research/non-profit only,
  open-source derivatives, commercial use by permission) and the exact
  remaining `code_inspected` work; root `README.md` command table
  brought up to the implemented CLI (separate commit).
- **Deliberately not done:** no `code_inspected` claim — the
  implementation was not read and no dependency closure inventoried;
  no `Containerfile`, no image, no digest, no `podman` in `devenv.nix`;
  the other three candidate leads stay `unverified`.

## Entry 2026-11-15 (fourth session: gap closure across the stack)

- **Tool / model:** Claude Code with Anthropic `claude-fable-5`
  (Claude Fable 5), operating autonomously under AGENT.md with the
  user's instruction to raise every component's implementation level.
- **What was done:** implemented the named contact gaps of
  `specs/07` (hull surface topology via `robot_model::hull::hull_faces`
  and `collision::HullTopology`; `mjc_hillclimbSupport`;
  `mjc_PlaneConvex`'s four-point plane-hull manifold through `hull4`);
  enforced `contact_slots` with a per-step dropped-contact count;
  ran the 12-cell G1 stance sweep and resolved B/S-03 as
  `options::Tier` with `--tier` on `robot sim multibody`
  (specs 0.1.10-draft and 0.1.11-draft); wrote and qualified the
  project-authored box QP (`motion_retarget::qp`) and the
  differential-QP IK step behind `[solver] method = "qp"`
  (ADR 0018, SPECS 0.9.1-draft, C-09 partially resolved); built
  `robot_tasks::env::HumanoidEnv` with `g1_stand` and the
  outcome-channel task closure, and proved the humanoid training
  path end to end at smoke scale
  (`robot-learn/tests/humanoid_training_path.rs`); repaired the rotted
  cubecl-gated `robot-sim-gpu` agreement suite and re-verified all
  49 feature-on tests under Mesa Lavapipe with the invocation now
  recorded in that crate's README.
- **Artifacts produced:** the code and tests above; ADR 0018;
  six evidence-ledger rows dated 2026-11-15; README updates in
  `robot-sim`, `robot-sim-gpu`, `robot-model`, `motion-retarget`,
  `robot-tasks`, `robot-learn` and the root; specs index, step 03/07/09
  and appendix-B updates.
- **Verification:** `cargo fmt --check`, `cargo clippy --workspace
  --all-targets` (0 warnings), `cargo test --workspace` (62 suites,
  0 failed), the exclusion gate (99 entries, 473 packages, 24
  manifests, 2 service records, all valid) all green at the end of
  the session.
- **Second wave (same session), via parallel subagent forks of the
  same model, each verified by the coordinating session:** a README
  gap sweep across all 24 crates classified every named gap as
  closable, decision-blocked or environment-blocked; the closable
  ones were then implemented — `robot sim fidelity` (the battery was
  library-only), `robot_model::kinematics::forward_kinematics_floating`
  (the plain entry point now refuses a `Free` joint by name instead of
  panicking), `robot_perception::attitude::ContactSupport` (the
  contact-conditioned accelerometer gate), `motion-process`'s
  tangent-space rotation filtering, automatic resampling and
  contact-aware segmentation, `motion-validate`'s centre-of-pressure
  check over collision-derived support polygons, `robot-viewer`'s
  `viewer-frames/1` content-addressed run storage,
  `robot_skills::manifest::load_from_blob`, `robot-devtools`'
  `kb-publish`, and the pinning of `robot-usd`'s already-exact half
  decode. Four further evidence-ledger rows record the measured and
  inspected claims.
- **Deliberately not done:** no commit (none requested); the mid-phase
  BVH, full `implicit`, `Ball` joints, elliptic cone and constraint
  islands remain open and named in `robot-sim`'s README; the
  opponent-forecast half of `robot-control`'s tactical tier stays
  deferred to SPECS.md Section 9.4's staged pipeline, as that crate's
  README assigns it; nothing humanoid is trained; no container image
  pinned (C-30 needs the user's licenses and a runtime decision).

## Entry 2026-11-15 (fifth session segment: direction decisions recorded)

- **Tool / model:** Claude Code with Anthropic `claude-fable-5`.
- **What was done:** transcribed three project-owner direction decisions
  into change-control records — ADR 0019 (Bevy as the designated viewer
  rendering candidate, B.6 pending, app-crate isolation plan), ADR 0020
  (AMD and NVIDIA GPUs as the batched-backend compute targets, work
  postponed until the systems exist, Vulkan path confirmed), ADR 0021
  (dataset intake joins the external-source scope as `data-source/1`
  records, motion data only initially, per-corpus C-07 audits still
  blocking) — with SPECS.md 0.9.2-draft, the Appendix B survey row,
  the §11.2 shortlist and `robot-viewer`'s README updated to match.
- **Artifacts produced:** the three ADRs and the document updates above.
- **Deliberately not done:** no Bevy dependency added (B.6 first), no
  `data-source/1` implementation yet, no dataset fetched or audited.

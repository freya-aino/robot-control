# Robot Control

A proposed Rust humanoid robotics platform for motion processing, simulation,
policy training, and hardware control, including boxing research.

- [User guide](docs/guide.md): setup, the pipeline end to end, every command, and the honest edges.
- [Platform specification](SPECS.md): architecture, dependency policy, and acceptance requirements.
- [Implementation plan](PLAN.md): delivery phases and qualification gates.

The earlier Python prior art (`training/`, `unitree/`, `test.ipynb`) was removed
from the working tree on 2026-10-26 and remains available only in version
control; see [SPECS.md Section 1.1](SPECS.md#11-relationship-to-prior-repository-contents)
for its disposition.

## Implementation

This repository is a devenv-managed Rust workspace implementing SPECS.md and
PLAN.md. The default dependency policy is std-only (ADR 0003), and no reference simulator, scene description, or
other foreign simulator is in the default closure. The CubeCL GPU dependency is
optional, behind `--features cubecl`, and off by default (ADR 0012). Enter the
environment and run the full gate with:

```
devenv shell   # or: direnv allow, if configured
devenv test    # fmt-check + clippy -D warnings + exclusion gate + cargo test --workspace
```

### Running the components

The program a user runs is the `robot` command-line tool in `robot-cli`. Run it
through cargo, or build it once with `cargo build -p robot-cli --release` and
use `target/release/robot`. With no arguments it prints its usage.

| Command | What it does | Example |
| --- | --- | --- |
| `robot data inspect <path>` | Decodes a binary chunk or a TOML artifact and reports its contents | `cargo run -p robot-cli -- data inspect <file>` |
| `robot motion validate <path>` | Parses a BVH source file and checks its motion record (channels, frame rate, license, missing frames). It does not validate a retargeted robot motion | `cargo run -p robot-cli -- motion validate <file.bvh>` |
| `robot sim qualify [--steps N] [--dt S]` | Runs the analytic pendulum energy-drift check (defaults: 2000 steps, 0.001 s) | `cargo run -p robot-cli -- sim qualify` |
| `robot sim multibody [--floating] [--zero-torque] [--dt S] [--steps N] [--out DIR] [--scene FILE] [--actuation drives\|damping\|zero] [--limit-margin RAD] [--integrator euler\|implicitfast\|rk4]` | Runs the vendored Unitree G1, loaded from its USD description, through the composite-frame dynamics, exact collision manifolds and Newton constraint solver of `robot_sim::world`. Without `--floating` the pelvis is fixed. With `--floating` it is a free base on the scene's ground plane (default `assets/scenes/g1_ground.usda`), held by the model's declared USD drives, or by zero torque with `--zero-torque`. `--out DIR` writes per-step CSV, JSON Lines and SVG pose frames plus two USD replays (`replay.usda`, `replay_playback.usda`) and requires `--floating`. Defaults: `--dt 0.0001`, `--actuation drives`, `--limit-margin 0`, `--integrator euler`; a floating run covers 2 s of simulated time, a fixed-base run 2000 steps | `cargo run -p robot-cli -- sim multibody --floating --out /tmp/g1-frames` |
| `robot eval summarize <values...>` | Prints summary statistics for a list of numbers | `cargo run -p robot-cli -- eval summarize 1.0 2.0 3.0` |

Exit codes: `0` success, `1` a command ran and found a real problem (a rejected
motion, a failed qualification), `2` named but not yet implemented, `3` a
floating-base `sim multibody` run too short (under 2 s simulated) to judge,
`64` usage error. Commands named in PLAN.md but not built (`data import`, `motion retarget`,
`eval --bundle` (also spelled `eval run`), `train`, `skills`, `deps`, `adapt`, `deploy`, `replay`) exit with `2`
and a reason on stderr.

The other crates are libraries, used by the CLI and by each other, and are
exercised through their tests. Beyond the CLI, these are available:

- GPU examples (need `--features cubecl` and a working wgpu/Vulkan adapter; ADR 0012 records runs on Intel Vulkan and Mesa Lavapipe):
  - `cargo run -p robot-sim-gpu --features cubecl --example gpu_latency` prints the max error against the CPU reference, bitwise determinism over repeated launches, and wall-clock latency. It is not a throughput benchmark.
  - `cargo run -p robot-sim-gpu --features cubecl --example gpu_resource_repeatability` prints transfer-inclusive resource accounting for the pendulum RK4 and equality paths.
- Fidelity report: `robot_sim::fidelity` writes `fidelity.toml` (schema v8) through its library API. Only the tests call it, so no command currently produces the file.

Every crate below has a real, tested implementation grounded in a specific
SPECS.md section (no stub crates remain) and its own `README.md` describing
what it implements, the reasoning behind any deviation or gap-filling
decision, and what is deliberately not implemented yet. The project's
architectural decisions are recorded as ADRs in `docs/decisions/`; the
evidence and development-automation ledgers required by SPECS.md Section 3
and PLAN.md Phase 0 are in `docs/ledger/`.

### Crates

| Crate | SPECS.md section | What it is |
| --- | --- | --- |
| [`robot-types`](crates/robot-types/README.md) | §§3, 5, 6, 7, 9, 10, 12 | Shared vocabulary: units, time, math, frames, IDs, schema/evidence/mode/tier types, RNG, observation/action specs, motion and episode records, transitions, bundle manifests |
| [`robot-artifacts`](crates/robot-artifacts/README.md) | §7 | Content-addressed storage, the bounded TOML subset, the binary chunk format, run/artifact manifests, the `robot-bundle/1` policy-bundle format (ADR 0016) and promotion ledger |
| [`robot-model`](crates/robot-model/README.md) | §6, §7.1, ADR 0015 | The kinematic tree, inertias, actuators, sensors, forward kinematics, URDF and OpenUSD import (mesh colliders reduced to convex hulls), the vendored G1, Go2 and H1 fixtures |
| [`motion-io`](crates/motion-io/README.md) | §8.1 | BVH import and export, C3D, AMASS npz, live capture (`robot-capture/1`), originals kept by content ID |
| [`motion-process`](crates/motion-process/README.md) | §8.2 | Conditioning, filtering, resampling, ground-plane and contact estimation, segmentation, augmentation |
| [`motion-retarget`](crates/motion-retarget/README.md) | §8.2, §8.3 | Multi-task IK (positions, orientations, chain-relative reach, posture, planted feet, self-collision, free base), temporal refinement, `retarget-map/1` files |
| [`motion-validate`](crates/motion-validate/README.md) | §8.2, §8.3 | Kinematic error, inverse-dynamics torques with contact distribution, contact consistency, simulated tracking per skill, curation manifests |
| [`robot-nn`](crates/robot-nn/README.md) | gap-filling, §9/§10 | Allocation-free MLP and GRU inference, the `robot-mlp/1` weights format with unsupported-operator refusal, and the `Policy` seams |
| [`robot-sim`](crates/robot-sim/README.md) | §9.1 | Pure-array CPU core: pendulum benchmarks; the `world` simulation path (composite-rigid-body mass matrix, unified contact/limit/equality solver with primal Newton default, GJK/EPA collision with convex-hull meshes, semi-implicit Euler, `implicitfast` and RK4, the model's declared actuators, per-run `SimOptions`); the ABA/RNE solver kept as its oracle; an oracle module and a schema-v8 fidelity harness. The per-step spec index is [`specs/README.md`](crates/robot-sim/specs/README.md) |
| [`robot-sim-gpu`](crates/robot-sim-gpu/README.md) | §9.1, PLAN.md Phase 3 | Feature-gated CubeCL f32 backend (`--features cubecl`, off by default) for a subset of the `robot-sim` functions, with CPU-vs-GPU agreement tests and `tolerances.toml`. Not part of the default build |
| [`robot-tasks`](crates/robot-tasks/README.md) | §9 | Stand, perturbation, tracking, reach, footwork and boxing-pad tasks over a robot-agnostic humanoid state; privileged-observation audit; calibrated randomization with held-out ranges |
| [`robot-usd`](crates/robot-usd/README.md) | §7.1, ADR 0015 | Std-only OpenUSD reader (USDC and USDA) with LIVRPS composition (sublayers, inherits, variants, references, payloads, specializes, list edits, layer offsets, target remapping), and the USD replay writer |
| [`robot-learn`](crates/robot-learn/README.md) | §§7.6, 9.2, 9.5 | Backprop, Adam, GAE, a diagonal Gaussian policy, rollout buffers, PPO |
| [`robot-skills`](crates/robot-skills/README.md) | §10 | Skill descriptors, exact retrieval, R1 working sets, the R2 selector (feasibility, OOD, transitions, dwell, hysteresis, recovery preemption, counters), expert staging, the Markdown catalog, `skill-manifest/1` files |
| [`robot-adapt`](crates/robot-adapt/README.md) | §11.3 | The adaptation ladder as a gate, system identification, history encoders, simulator correction, bounded residuals, gated fine-tuning, retrieval, morphology, adaptation reports |
| [`robot-perception`](crates/robot-perception/README.md) | §11.2 | Kalman filters, error-state attitude, floating-base estimation with slip gating, contact hysteresis, opponent tracking with age, NEES consistency against synthetic truth |
| [`robot-supervisor`](crates/robot-supervisor/README.md) | §12.1, §12.2 | The last stage before the actuators: limits, freshness, comms, contacts, torque-speed and thermal envelope, saturation, contact energy budgets, mode interlocks; the per-robot hazard register |
| [`robot-hardware`](crates/robot-hardware/README.md) | §4, §6, §12 | Device traits, loopback and replay devices, the `robot-wire/1` protocol adapter (UDP, CRC, sequence accounting, clock mapping) and its reference firmware peer with watchdog |
| [`robot-control`](crates/robot-control/README.md) | §5, §5.1 | The control loop: policy and tactical tiers with decision age, sensor taps into the supervisor, four-quantity timing reports, the incident channel |
| [`robot-eval`](crates/robot-eval/README.md) | §14 | Acceptance profiles as content-addressed artifacts, t-based statistics, Welch and TOST, interval-based evaluation, regression gates, the acceptance matrix closed only by ledger artifacts |
| [`robot-persona`](crates/robot-persona/README.md) | §12.5 | Versioned styles bounded inside validated ranges, an expression arbiter that yields to recovery, session-scoped opponent memory, recognizability and non-degradation evaluation |
| [`robot-devtools`](crates/robot-devtools/README.md) | §§2.3.1, 13.1, 2.4 | The exclusion gate, the §2.4 services gate over `services/`, the dependency knowledge base (Cargo.lock inventory, staleness sweeps, fault records) and evidence-ledger diffing that refuses unsupported promotions |
| [`robot-viewer`](crates/robot-viewer/README.md) | §6, §11.2 | Inspection data layer (skeletons, contacts, tracking error, synchronized playback) — no rendering toolkit is wired in; see its README for why |
| [`robot-cli`](crates/robot-cli/README.md) | PLAN.md §4 | The `robot` binary: data import, motion retarget and validate, sim, train, eval (including `--bundle` verification and the activation gate), skills, adapt identify and replay over the library crates; `deps` and `deploy` named with their reasons |

### External services (`services/`)

Foreign stacks the project uses as development tools — a video-to-SMPL-X
estimator and NVIDIA Isaac GR00T as an offline demonstration source are
the first two registered candidates — are not dependencies: they are
**containerized foreign services** under SPECS.md Section 2.4 and
[ADR 0017](docs/decisions/0017-containerized-foreign-services.md). Each
lives in [`services/`](services/README.md) as a `robot-service/1` record
(image pinned by digest once qualified, licensed model assets fetched
under the user's own license, never vendored) and exchanges **files
only**: its output enters through `robot data import` as untrusted,
content-addressed input. Nothing in `services/` touches the Cargo
closure, a robot host, or a control path, and the `exclusion-gate`
binary validates every record (rows A-51 and A-52). Both current records
are `unverified` and unpinned; no container runtime is provisioned until
the first one qualifies (open question C-30).

### What's genuinely limited, right now

- **PPO stabilizes the pendulum reference task on three seeds** (deterministic
  returns 90.8 to 97.1 against 34.2 for zero action), after the balance task's
  reward was found to pay the agent to fall and was given an alive bonus. The
  earlier negative result and its correction are recorded in `robot-learn`'s
  README. Nothing humanoid is trained yet.
- **`robot-sim` is a partial implementation, not a high-fidelity engine.** The CPU
  path runs the example formulation end to end: composite-frame dynamics, all
  33 of the G1's declared colliders (15 meshes reduced to convex hulls at
  import), joint-limit and contact rows in one Newton-solved convex problem,
  and the model's own declared drives, with `euler`, `implicitfast` and `rk4`
  selectable per run. Not ported: full `implicit`, hill-climbing hull support
  and multi-point plane-hull manifolds, `Ball` joints, and named fidelity
  tiers. The GPU backend covers only the subset listed in
  `crates/robot-sim-gpu/src/lib.rs`; the general closure-based RK4, batched
  G1 dynamics, and the contact, limit and friction solver are not ported.
  The pelvis and torso carry no collision geometry in the vendored layer. At
  the default `dt=1e-4`, the declared-drive stance keeps the pelvis minimum at
  0.788 m over 2 s; this is not a balance result. With zero torque the robot
  folds onto its limb hulls and the pelvis settles about 0.12 m above the
  plane. The G1's declared drive gains are generated defaults, which the run
  warns about. The fidelity harness (schema v8) is an analytic battery with
  contact, friction-cone and narrow-phase rows; its per-tier G1 contact
  status is `not_measured` at the reference tier and `not_applicable` at
  coarse, so no report qualifies G1 contact. The
  GPU-batched, variable-fidelity target, built on the OpenUSD / Isaac Sim model (ADR 0015), is specified in
  `crates/robot-sim/specs/` (index at `crates/robot-sim/specs/README.md`) and informed by research against
  published GPU-batched simulators, CubeCL and `wgpu`. CubeCL passed the Appendix B.6
  qualification for the implemented subset only (ADR 0012); it is not a default
  dependency.
- **No GUI/rendering or constrained-QP-solver dependency is qualified**, so
  `robot-viewer` stops at a data layer and `motion-retarget`'s IK is a
  Levenberg-Marquardt baseline (limits by projection, self-collision as a
  penalty) rather than the QP/whole-body solve SPECS.md Section 8.3 also
  names as a later option.
- **C-04 (canonical frames/gravity/quaternion convention) and most of
  Appendix C's other open questions remain genuinely open** — not resolved
  by this implementation, by design; see the ADRs in `docs/decisions/` for
  which gaps were filled and how, versus left open.

## Vendor SDK integration recommendations

Rewriting an SDK is possible, but its value depends on the implementation,
documented interfaces, maintenance burden, and applicable rights. For a project
that permits native vendor dependencies, safe Rust bindings or a hybrid design
are usually the practical starting point.

**Project policy:** [SPECS.md Section 2](SPECS.md#2-language-and-dependency-policy)
currently excludes foreign SDKs, Python sidecars, and foreign simulation engines
from the default Rust implementation. The options below are architectural
recommendations, not approved dependencies. Using them requires the explicit
scope decision and isolated feature/profile described in Section 2.1.

### Choosing an approach

| Factor | Implication |
| --- | --- |
| Thin wrapper over a documented protocol or standard algorithms | A Rust implementation may be viable after checking compatibility and licensing. |
| Accessible hardware specifications and firmware interfaces | Make independent implementation easier; owning hardware alone does not establish protocol access or legal permission. |
| Proprietary computation, calibration, or control algorithms | Prefer retaining the vendor implementation unless there is a validated replacement. |
| Frequent vendor updates | Bindings reduce duplicated maintenance, but upgrades still need ABI and behavior checks. |
| Performance-critical native code | Measure latency and throughput before deciding to replace it. |
| License or interoperability restrictions | Review the applicable terms and jurisdiction before reverse engineering or distributing a replacement. |

### Three integration patterns

1. **Safe Rust bindings to the vendor library.** Generate low-level bindings from
   supported C headers with `bindgen` during the build or a controlled generation
   step, then expose a small, manually reviewed safe API. A C++ SDK may need a C
   shim or an appropriate bridge. Encapsulate handles, initialization, shutdown,
   ownership, lifetimes, callbacks, thread restrictions, and error conversion.
   Check target OS, architecture, ABI, and library versions. Rust safety at the
   boundary depends on valid invariants; it does not make the vendor binary safe.

2. **Hybrid Rust control with vendor hardware I/O.** Keep state machines,
   planning, RL policy execution, motion selection, opponent modeling, and
   recovery in Rust. Retain the vendor layer only for necessary driver setup,
   calibration access, low-level I/O, and firmware communication. Confirm which
   responsibilities actually reside in the SDK versus the device firmware.

3. **Full Rust implementation.** Consider this when source or sufficiently
   complete protocol documentation is available, applicable rights permit it,
   and hardware access supports validation. Budget for compatibility testing,
   calibration, timing, recovery behavior, and future firmware changes. A rewrite
   takes on maintenance responsibility and does not automatically receive vendor
   fixes.

### Candidate systems

These are investigation directions; SDK internals and demo implementations must
be verified against the exact product and version.

| System | Recommendation |
| --- | --- |
| Unitree G1 SDK | Inspect the actual transport, supported APIs, licenses, and firmware requirements. Consider bindings for essential hardware access while implementing project control logic in Rust. Do not assume the SDK is merely USB/serial wrappers or that a boxing demo establishes how its policy was trained. |
| CHINGMU motion capture | Determine whether the SDK streams already reconstructed poses or performs reconstruction itself. Investigate open motion-recovery tools such as GVHMR and retargeting tools such as GMR as alternatives, validating accuracy, latency, licensing, and hardware needs; they are not automatic drop-in replacements. |
| NVIDIA Isaac Lab / Isaac Sim | If separately authorized, use as an external simulation or validation substrate rather than attempting a rewrite. Check component-specific source availability and licenses; Isaac Lab being open source does not establish that the entire simulator is. The project's default remains the qualified Rust simulation stack. |
| Generic servo/controller SDK | A direct Rust CAN/UART/TCP implementation may be suitable when the protocol is documented. Verify device-specific sequencing, calibration, timing, and fault handling rather than assuming the DLL only wraps PID and transport. |

### Practical recommendations for the boxing stack

- Prefer a narrow vendor hardware adapter over rewriting undocumented control
  behavior when an SDK exception is justified. Keep project strategy and control
  logic in Rust.
- Evaluate open motion-capture and retargeting algorithms before committing to a
  proprietary pipeline. A Rust port needs numerical validation. A Python sidecar
  over gRPC or Zenoh is an optional exception candidate, with explicit timing and
  failure behavior, rather than part of the default architecture.
- Document every `unsafe` block's invariants. Validate command ranges, buffer
  lengths, return codes, and response contents; test error propagation, malformed
  responses, initialization failures, and disconnect/reconnect sequences. Use
  property and mutation testing; foreign fuzz engines require the development-tool
  exception described in the specification.
- Validate hardware integration early through the project's supervised,
  noncontact qualification stages. Record required initialization and shutdown
  sequences, check timing and watchdog behavior, and reproduce stateful vendor
  sequencing before optimizing.
- Pin SDK/header versions and rerun compatibility checks for vendor updates.
  Bindings preserve access to vendor fixes only when those updates are integrated
  and validated.

### Licensing and independent implementations

Review SDK licenses, redistribution terms, and applicable interoperability rules.
Reverse-engineering and anti-circumvention rules vary by jurisdiction; hardware
ownership does not settle them. Patents may apply independently of source-code
copyright, so rewriting alone does not remove that concern.

Where appropriate, a clean-room process can separate analysis from implementation
using a reviewed interface specification and documented provenance. It is not a
blanket exemption from license, patent, or anti-circumvention obligations.

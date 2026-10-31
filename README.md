# Robot Control

A proposed Rust humanoid robotics platform for motion processing, simulation,
policy training, and hardware control, including boxing research.

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
devenv test    # fmt-check + clippy -D warnings + cargo test --workspace
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
| `robot sim multibody [--floating] [--zero-torque] [--dt S] [--steps N] [--out DIR]` | Runs the vendored Unitree G1 joint tree through the articulated-body solver. Without `--floating` the pelvis is fixed. With `--floating` it is a free base on a sphere ground plane under a clamped PD stance hold, or zero torque with `--zero-torque`. `--out DIR` writes per-step SVG pose frames and requires `--floating`. Defaults: `--dt 0.0001`; a floating run covers 2 s of simulated time, a fixed-base run 2000 steps | `cargo run -p robot-cli -- sim multibody --floating --out /tmp/g1-frames` |
| `robot eval summarize <values...>` | Prints summary statistics for a list of numbers | `cargo run -p robot-cli -- eval summarize 1.0 2.0 3.0` |

Exit codes: `0` success, `1` a command ran and found a real problem (a rejected
motion, a failed qualification), `2` named but not yet implemented, `64` usage
error. Commands named in PLAN.md but not built (`data import`, `motion retarget`,
`eval run`, `train`, `skills`, `deps`, `adapt`, `deploy`, `replay`) exit with `2`
and a reason on stderr.

The other crates are libraries, used by the CLI and by each other, and are
exercised through their tests. Beyond the CLI, these are available:

- GPU examples (need `--features cubecl` and a working wgpu/Vulkan adapter; ADR 0012 records runs on Intel Vulkan and Mesa Lavapipe):
  - `cargo run -p robot-sim-gpu --features cubecl --example gpu_latency` prints the max error against the CPU reference, bitwise determinism over repeated launches, and wall-clock latency. It is not a throughput benchmark.
  - `cargo run -p robot-sim-gpu --features cubecl --example gpu_resource_repeatability` prints transfer-inclusive resource accounting for the pendulum RK4 and equality paths.
- Fidelity report: `robot_sim::fidelity` writes `fidelity.toml` (schema v7) through its library API. Only the tests call it, so no command currently produces the file.

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
| [`robot-artifacts`](crates/robot-artifacts/README.md) | §7 | Content-addressed storage, the bounded TOML subset, the binary chunk format, run/artifact manifests and promotion ledger |
| [`robot-model`](crates/robot-model/README.md) | §6, §7.1 | The kinematic tree, inertias, actuators, sensors, forward kinematics, URDF import |
| [`motion-io`](crates/motion-io/README.md) | §8.1 | BVH motion-capture import |
| [`motion-process`](crates/motion-process/README.md) | §8.2 | Motion resampling (filtering/contact/segmentation not yet implemented) |
| [`motion-retarget`](crates/motion-retarget/README.md) | §8.3 | Damped-least-squares task-space IK |
| [`motion-validate`](crates/motion-validate/README.md) | §8.3 | Retargeting validation reports and acceptance thresholds |
| [`robot-nn`](crates/robot-nn/README.md) | gap-filling, §9/§10 | A small MLP with zero-allocation inference, kept isolated from training code |
| [`robot-sim`](crates/robot-sim/README.md) | §9.1 | Pure-array CPU core: pendulum benchmarks, ABA/RNE multi-body solver with a floating base, formula modules for the integrator, passive forces, actuators, equality constraints and plane-sphere contact, an oracle module, and a schema-v7 fidelity harness. The per-step spec index is [`specs/README.md`](crates/robot-sim/specs/README.md) |
| [`robot-sim-gpu`](crates/robot-sim-gpu) | §9.1, PLAN.md Phase 3 | Feature-gated CubeCL f32 backend (`--features cubecl`, off by default) for a subset of the `robot-sim` functions, with CPU-vs-GPU agreement tests and `tolerances.toml`. Not part of the default build |
| [`robot-tasks`](crates/robot-tasks/README.md) | §9 | Reward registry and a balance/stabilization task |
| [`robot-usd`](crates/robot-usd/README.md) | §7.1, ADR 0015 | Std-only OpenUSD reader (USDC and USDA) with layer composition, and the USD replay writer |
| [`robot-learn`](crates/robot-learn/README.md) | §§7.6, 9.2, 9.5 | Backprop, Adam, GAE, a diagonal Gaussian policy, rollout buffers, PPO |
| [`robot-skills`](crates/robot-skills/README.md) | §10.6 | The R1 working set and the R2 in-loop router over an exact-search index; R0 offline corpus selection is not implemented |
| [`robot-adapt`](crates/robot-adapt/README.md) | §11.3 | System identification (first, mandatory row of the adaptation-method table) |
| [`robot-perception`](crates/robot-perception/README.md) | §11.2 | Linear/multiplicative-error-state Kalman filtering, contact classification |
| [`robot-supervisor`](crates/robot-supervisor/README.md) | §12.1, §12.2 | The last-stage fault policy and safety supervisor, structurally isolated from policy/retrieval/personality |
| [`robot-hardware`](crates/robot-hardware/README.md) | §12 | The hardware-adapter trait, a loopback device and a replay-only device |
| [`robot-control`](crates/robot-control/README.md) | §5 | The control loop tying timing, supervisor and hardware adapter together |
| [`robot-eval`](crates/robot-eval/README.md) | §14 | Acceptance profiles (immutable once evaluated) and cross-seed statistics |
| [`robot-persona`](crates/robot-persona/README.md) | §12.5 | Bounded expressive intent over already-validated skills |
| [`robot-devtools`](crates/robot-devtools/README.md) | §§2.3.1, 13.1 | The dependency knowledge base's citation/staleness data model (developer/CI-only; no shipping crate depends on it) |
| [`robot-viewer`](crates/robot-viewer/README.md) | §6, §11.2 | Inspection data layer (skeletons, contacts, tracking error, synchronized playback) — no rendering toolkit is wired in; see its README for why |
| [`robot-cli`](crates/robot-cli/README.md) | PLAN.md §4 | The `robot` binary: a real, honestly-scoped subset of the proposed CLI surface, with every unimplemented command named explicitly rather than omitted |

### What's genuinely limited, right now

- **PPO on the pendulum-stabilization task does not reliably demonstrate
  measured improvement** within the compute budget tried. The gradient
  machinery itself is validated independently (an isolated one-step-bandit
  test passes); the remaining gap is multi-step credit assignment/tuning,
  not a known implementation bug. Recorded as an honest negative result per
  `robot-learn`'s README and SPECS.md Section 3's evidence discipline,
  rather than hidden behind a passing-looking assertion.
- **`robot-sim` is a partial implementation, not a high-fidelity engine.** The CPU
  formula modules (integrators, passive forces, actuators, equality constraints,
  plane-sphere contact) are not yet wired into the multibody stepper. The GPU
  backend covers only the subset listed in `crates/robot-sim-gpu/src/lib.rs`;
  the general closure-based RK4, batched G1/ABA, and the contact, limit and
  friction solver are not ported. Contact in the multibody path is modelled for
  only the ten ankle spheres of the USD model (five per ankle); the foot boxes,
  knee and shoulder cylinders and mesh shapes are captured but not modelled, and
  modelled, and the pelvis and torso have no collision geometry. At `dt=1e-4`, a clamped-PD
  supported-stance hold keeps the vendored G1 pelvis above the plane for 2 s
  (minimum 0.787 m); this is not a balance result. With zero torque, the pelvis
  falls to −0.77 m below the plane. The fidelity harness reports an analytic
  subset: G1 contact is `not_measured` at the reference tier and
  `not_applicable` at coarse; neither status qualifies G1 contact. The
  GPU-batched, variable-fidelity target, built on the OpenUSD / Isaac Sim model (ADR 0015), is specified in
  `crates/robot-sim/SIMULATION_SPECS.md` and informed by research against
  published GPU-batched simulators, CubeCL and `wgpu`. CubeCL passed the Appendix B.6
  qualification for the implemented subset only (ADR 0012); it is not a default
  dependency.
- **No GUI/rendering or constrained-QP-solver dependency is qualified**, so
  `robot-viewer` stops at a data layer and `motion-retarget`'s IK is a
  damped-least-squares baseline rather than the QP/whole-body solve SPECS.md
  Section 8.3 also names as a later option.
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

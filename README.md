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
PLAN.md. Enter the environment and run the full gate with:

```
devenv shell   # or: direnv allow, if configured
devenv test    # fmt-check + clippy -D warnings + cargo test --workspace
```

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
| [`robot-sim`](crates/robot-sim/README.md) | §9.1 | Single-DOF pendulum/point-mass contact benchmarks plus a fixed-base multi-body (ABA/RNE) solver; [`SIMULATION_SPECS.md`](crates/robot-sim/SIMULATION_SPECS.md) specifies the GPU-batched, MuJoCo-caliber target |
| [`robot-tasks`](crates/robot-tasks/README.md) | §9 | Reward registry and a balance/stabilization task |
| [`robot-learn`](crates/robot-learn/README.md) | §§7.6, 9.2, 9.5 | Backprop, Adam, GAE, a diagonal Gaussian policy, rollout buffers, PPO |
| [`robot-skills`](crates/robot-skills/README.md) | §10.6 | The three retrieval tiers (R0 exact index, R1 working set, R2 in-loop router) |
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
- **`robot-sim`'s multi-body solver is fixed-base, CPU-only, with no
  contact or collision resolver** — a real Articulated-Body/Recursive-
  Newton-Euler dynamics engine now exists and runs the actual vendored
  G1 (see `crates/robot-sim/README.md`), but the root link is rigidly
  welded to world (no floating pelvis), there is no GPU execution path,
  and ground contact is still only the separate single-point-mass
  benchmark in `robot_sim::contact`. The target — GPU-bound, batched,
  variable-fidelity, on par with MuJoCo — is specified in
  `crates/robot-sim/SIMULATION_SPECS.md`, grounded in direct research
  against MuJoCo's own source, MJX, Brax, CubeCL and `wgpu`; none of it
  is implemented yet. No physics-engine or GPU-compute dependency has
  passed the qualification gate of Appendix B.6.
- **No GUI/rendering or constrained-QP-solver dependency is qualified**, so
  `robot-viewer` stops at a data layer and `motion-retarget`'s IK is a
  damped-least-squares baseline rather than the QP/whole-body solve SPECS.md
  Section 8.3 also names as a later option.
- **C-04 (canonical frames/gravity/quaternion convention) and most of
  Appendix C's other open questions remain genuinely open** — not resolved
  by this implementation, by design; see the ADRs in `docs/decisions/` for
  which gaps were filled and how, versus left open.
- No commit has been made to version control as part of this
  implementation pass; everything above is uncommitted in the working tree
  pending review.

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

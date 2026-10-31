# Step 00 — Target, constraints and evidence base

[Index](README.md) · [Appendix A](appendix-a-citations.md)

**Status**: Specification only. No implementation claim.

**Goal**: Fix the purpose, the "on par with the reference simulator" target, the non-goals and the evidence base that every later step is measured against. Covers old §0, §1, §2 and §15.

**Depends on**: none.

**Reference formulation**: Not applicable. This step states the target and cites the baseline (Appendix A, entries 1–17); it defines no algorithm.

**Purity contract**: Not applicable (no algorithm).

**Inputs/outputs**: Input: `SPECS.md` §2, §3 and §9.1. Output: the binding target that steps 01–12 reference.

**Done when**: The target, non-goals and evidence states are stated once here, and every later step refers back to them instead of restating them.

**Open questions**: none blocking.

---

## 0. Purpose and how this document was built

The project's target changed, by explicit direction, partway through implementing `robot-sim`'s multi-body dynamics solver: the simulator must become **GPU-bound and batched** (thousands of parallel environments, not one instance), support **variable fidelity** (a caller trades accuracy for speed, not a single fixed-quality engine), stay **as close to Rust-native as possible** (minimal FFI, no foreign runtime — continuing `SPECS.md` Section 2's standing policy, not an exception to it), and be designed so the simulation backend is **agnostic to every other component** — training, control, viewing, the CLI — which depend on an interface, never on a backend's internals.

"On par with the reference simulator" is the stated fidelity target. Sections 1-10 below are therefore grounded in direct research against the reference's own source and documentation, not its marketing description: a research agent read `google-deepmind/reference` at pinned commit `800a109a4bdd02cb60404d565db4ae16fb04a53b` (tag `3.17.0`), the `mjx` subdirectory of the same repository, `huggingface/candle`, `tracel-ai/cubecl`, `gfx-rs/wgpu`, `google/brax`, `isaac-sim/IsaacLab` and `dimforge/rapier`, each at whatever commit was current on 2026-10-26. Appendix A is the condensed citation ledger; inline citations below use the short form `[the reference simulator §N]` / `[batched backend §N]` / `[Brax §N]` / etc. pointing at Appendix A's numbered entries. Every claim about a third-party system carries the evidence-state tag that table licenses — most are `code_inspected` (source/docs read directly) or explicitly flagged `secondary`/`unverified` where the research agent could not confirm something first-hand. Claims about *our own* system start at `unverified` like everything else in this project and are promoted only by the tests Section 14 requires.

## 1. Target and explicit non-goals

### 1.1 "On par with the reference simulator" — what this binds

A `robot-sim` physics backend claiming parity with the reference simulator MUST, at minimum, address every subsystem the reference's own engine treats as load-bearing (Sections 4-10): generalized coordinates including a floating base, a mass-matrix/bias-force dynamics pass, a constraint solver that handles contact and equality constraints as one problem, a multi-stage collision pipeline, actuators with a real transmission/gain/bias model, and an integrator family that remains stable on stiff systems. "On par" does not mean bit-identical, require the reference's own exact algorithms where a documented alternative is better suited to a GPU-batched, Rust-native implementation (Section 11 makes exactly this call for the solver's batching strategy), and does not require feature parity with the reference's biology-oriented extensions (muscle activation dynamics) or its deformable-object machinery (`flex`, 1D/2D/3D soft bodies) — both are explicit non-goals under 1.6.

### 1.2 GPU-bound, batched execution

The primary execution mode MUST be thousands of environments stepped in parallel on a GPU, following the precedent of batched backend and Brax (Appendix A entries 12, 13, 17) rather than this project's existing CPU-only, single-instance `robot_sim::articulated::ArticulatedBody`. Section 11 specifies the batching architecture. A CPU reference backend remains REQUIRED regardless (Section 11.4) — not as the primary execution mode, but as the correctness oracle the GPU backend is checked against, exactly as JAX-based batched backend's own test suite is checked against CPU the reference simulator [batched backend §12].

### 1.3 Variable fidelity

The engine MUST expose fidelity as a tunable, queryable property of a simulation run, not a fixed characteristic of the backend. Section 12 enumerates the specific knobs (integrator, solver iteration budget, contact dimensionality, contact-slot budget, timestep) and proposes named fidelity tiers as this project's own addition — research found no existing named-tier concept in the reference simulator itself [the reference simulator §15], only an unnamed pile of independent knobs, so this is a deliberate design decision, not an inherited one.

### 1.4 Rust-native-first

No foreign runtime, no FFI to C/C++/CUDA-as-a-foreign-language, and no the reference simulator binding, continuing `SPECS.md` Section 2's standing exclusion rather than creating an exception to it. "As close to Rust-native as possible" is read as: author GPU kernels in Rust (via CubeCL) where feasible, fall back to hand-written WGSL compute shaders via `wgpu` (still Rust-hosted, still no foreign toolchain beyond the GPU driver) where CubeCL cannot express the needed control flow, and treat any C/CUDA-authored kernel as a foreign-dependency exception requiring the Section 2.1 procedure, not a default path. Section 11.3 details the specific candidates and their risks.

### 1.5 Backend-agnostic / plug-and-play

Every crate outside `robot-sim` that consumes a simulated environment (`robot-learn`, `robot-control`, `robot-viewer`, `robot-cli`, `robot-tasks`) MUST depend only on a stable environment interface, never on a physics-backend-specific type. Section 13 specifies the two-layer boundary this requires, modeled directly on Brax's `Env`/`PipelineEnv` split [Brax §17] — the clearest, most concrete precedent research found for exactly this separation.

### 1.6 Explicit non-goals for this revision

- Muscle/biological actuator models (the reference's `MUSCLE` gain/bias/dynamics type) — not relevant to a rigid-actuator humanoid.
- Deformable objects (`flex`: cloth, rope, soft tetrahedral bodies) and signed-distance-field collision geoms — genuinely still "under development" in the reference simulator itself as of the pinned commit [the reference simulator §11], and out of scope for a first humanoid-locomotion target.
- Differentiability (gradients through the physics step) — JAX-based batched backend has this via JAX autodiff, batched backend-Warp explicitly does not [batched backend §17]; this revision does not require it of either `robot-sim` backend, though Section 13.3 names it as a capability that MAY differ by backend later and must be declared, not assumed.
- A rendering/visualization path — `robot-viewer` remains the data-layer-only crate it already is; nothing here requires or implies a GPU rasterizer.
- Floating-base *URDF* joints. `robot_model::urdf` continues rejecting `floating`/`planar` joint types in imported URDF by name (`docs/decisions/0002`); Section 3's floating-base requirement is a `robot-model`/`robot-sim` **simulation-time** construct, added as a new `JointKind` variant, not a relaxation of what URDF import accepts. A URDF's root link remains implicitly fixed to its parent frame at import time; whether that frame itself is static or free is a property the simulation layer assigns afterward.

## 2. Evidence base at a glance

| Subsystem | the reference's approach | Evidence state (per `SPECS.md` §3) | Appendix A entry |
| --- | --- | --- | --- |
| Coordinates | Generalized (joint-space); 4 joint types incl. `FREE`/`BALL`; loops only via equality constraints | `code_inspected` (source read, pinned commit) | 1 |
| Dynamics pass | CRBA for `M`, RNE (accel=0) for bias `c`; sparse factorization | `code_inspected` | 2 |
| Constraint solver | Convex optimization (not LCP); PGS/CG/Newton; `solref`/`solimp` soft constraints | `code_inspected` | 3 |
| Collision | Broad-phase sweep + BVH mid-phase + dispatch-table narrow-phase; GJK/EPA | `code_inspected` | 4 |
| Integrators | Euler (semi-implicit + implicit joint damping by default!), RK4, implicit, implicitfast, discrete | `code_inspected` | 5 |
| Actuators/tendons/equality | Transmission × activation-dynamics × gain/bias model; 8 equality types; same solver as contact | `code_inspected` | 6 |
| Passive forces | Position/velocity-only spring-damper, dry friction, gravity comp | `code_inspected` | 7 |
| Sensors | 55 types; per-sensor noise field present, injection point unconfirmed | `code_inspected` (existence); `unverified` (noise injection) | 8 |
| Compilation | Inertia-from-geometry inference, static-body fusing, convex hull generation | `code_inspected` | 9 |
| Determinism/batching | Deterministic within one build/arch; new native CPU `mjbatch`; batched backend for GPU/TPU | `code_inspected` (mechanism); `secondary` (any throughput number) | 10 |
| batched backend architecture | Static-shape contacts via model-time enumeration + top-k selection; PGS dropped; Newton/CG kept | `code_inspected` | 12 |
| Rust GPU paths | CubeCL: real SIMT control flow, zero found physics precedent. Candle: CUDA-only custom-kernel path, narrower. wgpu: no CUDA, real non-rendering compute precedent (Bevy), all embarrassingly parallel | `code_inspected` (capabilities); `unverified` (suitability for this workload, absent precedent) | 13/16 |
| Batching strategy | Two real architectures: vectorize-per-tree-level (JAX-based batched backend) vs. one-thread-per-environment (batched backend-Warp/CUDA-style) | `code_inspected` (both exist); `secondary` (why Warp was built — reasonable inference, not a quoted team statement) | 14 |
| Variable fidelity | the reference simulator: unnamed knob pile. PhysX/IsaacLab: named TGS/PGS + multi-solver-backend family. Brax: 4 named pipelines ordered by speed/fidelity | `code_inspected` | 15 |
| Backend-agnostic boundary | batched backend leaks capability/sizing details through documented exceptions. Brax's `Env`→`PipelineEnv`→pipeline-module split is clean and narrow | `code_inspected` | 17 |

Appendix A gives full citations, pinned commits/versions, and the confidence/source-type for each entry, including every point the research agent explicitly flagged as unverified (exact version Newton became the default solver; whether `mjModel.sensor_noise` is applied automatically; any independently-reproduced throughput number; batched backend's complete feature-parity gap list).

## 15. Explicit non-goals (restated)

Identical to Section 1.6; restated here only because `SPECS.md`'s own Section 15 convention is to give non-goals their own numbered section rather than bury them in scope.

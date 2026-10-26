# 0005 — the simulation target is GPU-bound, variable-fidelity, MuJoCo-caliber, and authored

- Status: accepted
- Date: 2026-10-26
- Touches: SPECS.md §9.1, Appendix B.2.1, B.2.3, B.4, Appendix C (new row C-25); `PLAN.md` Phase 3; `crates/robot-sim/SIMULATION_SPECS.md` (new); `crates/robot-sim/README.md`, `crates/robot-sim/src/lib.rs`

## Context

This session built `robot_sim::spatial`/`robot_sim::articulated`: a fixed-base, CPU-only Articulated-Body/Recursive-Newton-Euler dynamics solver, validated against a hand-derived closed form, a round-trip cross-check between the two algorithms, and an energy-conservation check, then run against the vendored Unitree G1 fixture. That work was scoped against SPECS.md §9.1's existing direction — "qualify Rapier's existing multibody-joint and collision capabilities before proposing a replacement articulation layer" — as an authored interim, the same pattern this project has used repeatedly when no dependency has passed Appendix B.6's qualification gates (`docs/decisions/0003`).

The target then changed, by three explicit, sequential user directions in the same session:

1. "the target is a simulation on par with mujoco... look for the implementation from mujoco... create a detailed SIMULATION_SPECS.md inside the relevant component."
2. "we need variable fidelity simulation... as close to rust native as possible... keep it agnostic to all other components so the rest becomes mostly plug and play."
3. "the simulation needs to be GPU bound, take the available GPU DNN frameworks used in this project to refine the SPECS for GPU usage."

A research agent read MuJoCo's own source and documentation at a pinned commit (`google-deepmind/mujoco`, tag `3.17.0`), MJX's actual implementation in the same repository, and `huggingface/candle`, `tracel-ai/cubecl`, `gfx-rs/wgpu`, `google/brax`, `isaac-sim/IsaacLab` and `dimforge/rapier`, each read directly rather than summarized secondhand. `crates/robot-sim/SIMULATION_SPECS.md` is the resulting specification; this record is why it exists and what it changes in the documents that governed before it.

## Decision

1. **The fidelity target is MuJoCo's current engine (3.x), not the 2012 paper's scope**, covering generalized coordinates with a floating base, a convex-optimization constraint solver (not classical LCP), a multi-stage collision pipeline, a real actuator/tendon/equality-constraint model, and an integrator family beyond plain explicit/semi-implicit Euler. `SIMULATION_SPECS.md` Sections 3-10 specify each subsystem, with every MuJoCo-attributed claim tagged against SPECS.md §3's evidence-state table (most `code_inspected`, since the agent read source directly; throughput/performance numbers remain `secondary` — MuJoCo's own reported figures, never independently reproduced).
2. **Execution is GPU-bound and batched by default** — thousands of environments, not one instance — with a required CPU reference backend kept specifically as the correctness oracle the GPU backend is checked against, mirroring how MJX-JAX's own test suite is checked against CPU MuJoCo.
3. **Fidelity is a first-class, runtime-tunable interface**, not a fixed engine characteristic. Research found MuJoCo itself exposes no named fidelity-tier concept, only independent knobs; Brax's four named, speed-ordered pipelines and Isaac Lab's named multi-solver-backend family are the precedents `SIMULATION_SPECS.md` §12 follows instead, proposing named tiers as this project's own addition.
4. **The implementation stays Rust-native**: no MuJoCo binding, no foreign-runtime dependency, continuing SPECS.md §2's standing exclusion rather than creating an exception to it. CubeCL is the primary GPU-compute candidate (real SIMT control flow, native CUDA reachability, zero found precedent for non-neural-network use), with hand-written `wgpu`/WGSL as the fallback for whatever CubeCL cannot yet express reliably. This is a `reproduced_in_sim`-track decision, not a `validated_on_hardware` one — tracked open at C-25 precisely because no existing Rust-native GPU-batched rigid-body engine was found to validate the choice against.
5. **The batching strategy is one GPU thread per environment** (running the existing authored ABA/RNE recursion largely unchanged), not per-tree-level vectorization across environments (JAX/`vmap`'s approach). Rationale, in `SIMULATION_SPECS.md` §11.1: it reuses this session's validated CPU algorithm rather than requiring a rewrite into batched-tensor-operation form, and it is the better-scaling of the two per MuJoCo's own team's documented reason for building a second implementation (MJX-Warp) once the vectorized approach's branching/fixed-contact-count limits became the bottleneck.
6. **Every crate outside `robot-sim` depends only on `robot_sim::env::Environment`** (or its eventual batched generalization), never on a physics-backend-specific type. `SIMULATION_SPECS.md` §13 specifies a Brax-style two-layer boundary (`Environment` stable; a narrow internal backend-plug-point selecting CPU-reference or GPU-batched underneath) as the concrete mechanism, including Brax's own honesty pattern of exposing backend identity as a queryable property rather than pretending the abstraction never leaks — MJX's own documentation names several places where it does.

## What this does not change

- SPECS.md §2's foreign-dependency exclusion is not relaxed. MuJoCo, MJX, Brax, Isaac Lab and PhysX are design references read for their algorithms and architecture; none is a dependency, and no binding to any of them is requested or implied.
- SPECS.md §9.1's instruction to qualify Rapier before authoring a replacement articulation layer is not deleted — it is superseded for the specific subsystem this decision covers (articulation, contact, GPU batching), per the new cross-reference added to §9.1, because the project's own explicit direction is to build the authored, GPU-batched engine directly rather than wait on that qualification. Rapier qualification remains available as a comparison point if anyone later wants one; it is not the default path for this subsystem anymore.
- `robot_sim::spatial`/`articulated`'s existing fixed-base CPU solver is not retired or rewritten by this decision. It remains the CPU reference backend `SIMULATION_SPECS.md` §11.4 requires, and its RNE core was confirmed, by the same research pass, to already match MuJoCo's own RNE implementation structurally (identical gravity-as-fictitious-base-acceleration trick, same sign convention) — this decision is a continuation of that code's role, not a repudiation of it.
- No implementation work beyond what already existed before this decision was performed under it. `SIMULATION_SPECS.md` is a specification; Appendix B of that document and C-25 here track exactly what remains open, and implementation does not proceed until separately requested.

## Consequences

- `crates/robot-sim/SIMULATION_SPECS.md` is the authoritative, detailed specification for this subsystem going forward; SPECS.md §9.1 and Appendix B.2.1/B.2.3/B.4 carry pointers to it rather than duplicating its content, so the fast-moving design detail (solver algorithm, batching strategy, GPU-compute framework) can change without a `SPECS.md` revision each time.
- Future work on `robot-sim`'s physics layer is reviewed against `SIMULATION_SPECS.md`'s own Sections 1-14 and its Appendix B open questions, not solely against SPECS.md §9.1's original Rapier-qualification framing.
- `robot_model::JointKind` gaining `Free` (and, if scheduled, `Ball`) joint variants is now a named, tracked prerequisite (`SIMULATION_SPECS.md` §3, Appendix B row S-04) for a floating-base humanoid, not an unscoped future idea.
- This record, `crates/robot-sim/README.md`, `crates/robot-sim/src/lib.rs`'s doc comment, and the top-level `README.md`'s limitations section all cross-reference `SIMULATION_SPECS.md` rather than independently restating what changed, so there is one place a future reviewer checks for the current target.

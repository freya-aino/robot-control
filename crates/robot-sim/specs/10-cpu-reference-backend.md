# Step 10 — CPU reference backend and backend-agnostic architecture

[Index](README.md) · [Appendix A](appendix-a-citations.md) · [Appendix C](appendix-c-existing-code.md)

**Status**: Partially implemented. `robot_sim::articulated::ArticulatedBody` is a single-instance CPU backend, and `robot_sim::env::Environment` (`observation_width`, `action_width`, `reset`, `step`) exists. The narrow `init`/`step` backend trait does not exist yet.

**Goal**: Keep a single-instance CPU reference backend as the correctness oracle for every GPU backend, and put every caller behind one backend-agnostic `Environment` interface. Covers old §11.4 and old §13 in full.

**Depends on**: [02](02-pure-dynamics-core.md) (dynamics core), [03](03-integrators.md) (integrators).

**Reference formulation**: Not applicable to the trait design. The oracle role is modeled on JAX-based batched backend's test suite against CPU reference simulator (Appendix A, entry 12) and on Brax's `Env`/`PipelineEnv` split (Appendix A, entry 17). The CPU dynamics it calls are baselined in steps 02 and 03.

**Purity contract**: The backend's numerical code is pure math (FR-6). The `Environment` and backend trait are the boundary and may use crate types, provided no algorithm function does.

**Inputs/outputs**: Input: a profile and config (`init`), an action (`step`). Output: a backend state and observations. The `Environment` trait is the same for every backend.

**Done when**: A narrow backend trait exists with `init` and `step`. `Environment` is implemented once, generically, over it. The CPU backend implements it and is never retired. A backend identifier is queryable from any `Environment`.

**Open questions**: none blocking. The exact trait name is an implementation decision. Backend-specific sizing parameters are named in [11](11-gpu-batched-backend.md).

---

### 11.4 CPU reference backend (required, not optional)

1. A single-instance, full-control-flow CPU implementation MUST exist independent of whatever GPU-batched backend is built, and MUST remain the correctness oracle the GPU backend is checked against — mirroring exactly how JAX-based batched backend's own test suite is checked against CPU reference simulator [batched backend §12], and continuing the pattern this session already established (`robot_sim::articulated::ArticulatedBody`, validated by closed-form/round-trip/energy tests with no GPU involved at all).
2. The CPU reference backend MUST NOT be retired once a GPU backend exists. It is the fixture every fidelity-tier definition (Section 12) and every cross-backend agreement test (Section 14) is measured against.

## 13. Backend-agnostic architecture

### 13.1 The stable, caller-facing interface

`robot_sim::env::Environment` (already implemented: `observation_width`, `action_width`, `reset`, `step`) is the `robot-sim` analogue of Brax's `Env` trait (`reset`, `step`, `observation_size`, `action_size`) [Brax §17]. This interface MUST remain backend-independent: no method signature, associated type, or field may vary by whether the implementation underneath is the CPU reference backend or a GPU-batched one. `robot-learn`, `robot-control`, `robot-viewer` and `robot-cli` MUST depend only on this trait (or its eventual batched generalization, 13.2) — never on `robot_sim::articulated`, `robot_sim::spatial`, or any future GPU-backend module directly.

### 13.2 The narrow backend plug-point

Following Brax's own two-layer design exactly — `Env` (stable) wraps a `PipelineEnv` that selects one of a fixed set of backend modules by name, and from then on calls exactly two functions on it, `pipeline.init` and `pipeline.step`, with every backend-specific detail living behind those two calls and nothing else [Brax §17] — `robot-sim` MUST define an equivalently narrow internal trait (a "pipeline" or "backend" trait, exact name an implementation decision) with the minimum surface needed to implement `Environment` on top of it: something equivalent to `init(profile, config) -> BackendState` and `step(&BackendState, action) -> BackendState`. The CPU reference backend (11.4) and the GPU-batched backend (11.1-11.3) MUST both implement this same narrow trait, and `Environment`'s own implementation MUST be written once, generically, over it — not duplicated per backend.

### 13.3 What MAY legitimately differ by backend, named explicitly rather than hidden

Brax's own `Env` trait exposes `backend: str` as a stable, queryable property — an explicit, honest acknowledgment that backend identity is sometimes observable even though the call signature never changes [Brax §17]. batched backend's own documented leaks (an exception raised at load time if a model uses an batched backend-unsupported feature; `naconmax`, an batched backend-only sizing concept with no CPU-the reference simulator equivalent a caller must learn to size correctly; differentiability present in JAX-based batched backend, absent in batched backend-Warp and CPU reference simulator alike) are named directly in its own documentation, not discovered by accident [batched backend §17]. `robot-sim` MUST adopt the same honesty rather than claim a leak-free abstraction that the strongest found precedent (batched backend itself, built by the reference's own team) does not actually achieve:

1. A backend identifier MUST be queryable from any `Environment` implementation.
2. Any backend-specific sizing parameter (the GPU backend's contact-slot budget, 11.2) MUST be named and documented as such, not folded silently into a profile or a generic config struct that reads as backend-independent.
3. Differentiability, if ever added to any backend, MUST be declared via a capability query, not assumed present or absent.
4. A model feature unsupported by a given backend MUST cause a loud, named error at construction time (matching batched backend's own exception-on-load behavior), never a silent fallback to different physics.

### 13.4 Dependency direction

`robot-sim`'s GPU-batched backend module(s) MAY depend on CubeCL/`wgpu` (Section 11.3); nothing outside `robot-sim` may. This keeps the GPU-compute dependency decision fully reversible and fully contained — consistent with this project's standing pattern of isolating a not-yet-qualified dependency behind the one crate that needs it (`docs/decisions/0004`'s `robot-nn`/`robot-supervisor` extraction is the direct precedent for this kind of boundary).

# Step 02 — Pure dynamics core (mass matrix, bias forces)

[Index](README.md) · [Appendix A](appendix-a-citations.md) · [Appendix C](appendix-c-existing-code.md)

**Status**: Partially implemented. The existing spatial algebra and articulated ABA/RNE algorithm paths are now refactored to plain fixed-size `f64` arrays, with boundary adapters and the public `ArticulatedBody` API retained. CRBA / M(q) is not implemented; see Appendix C. Refactor regressions pass the closed-form, round-trip, energy, adapter and dense-solver tests, without establishing measured parity with the reference simulator.

**Goal**: Keep the mass-matrix-and-bias-force structure (CRBA for `M`, RNE with zero acceleration for the bias `c`) and state the local-frame convention. Covers old §4, items 1–2. Items 3–5 (integration) are in step 03.

**Depends on**: [01](01-state-and-coordinates.md).

**Reference formulation**: Tag `3.17.0`, commit `800a109a4bdd02cb60404d565db4ae16fb04a53b` (Appendix A, entry 2). Function definitions, located by name at the pin:
- `src/engine/engine_core_smooth.c`: `mj_rne` (recursive Newton-Euler, bias force) and `mj_crb` (composite rigid body, mass matrix).
- `src/engine/engine_forward.c`: the forward pipeline that calls them (same entry).

Evidence tag: `code_inspected` for the existence and names of `mj_rne` and `mj_crb` at the pin. The equivalence claim in old §4 item 1 (gravity-as-fictitious-base-acceleration) is `code_inspected` only as stated by the research pass (Appendix A, entry 2), not re-derived here.

**Purity contract**: The CRBA and RNE recursions use `[f64; 6]` / `[f64; 3]` arrays and explicit formulas. They MUST NOT use `robot_types::math::Matrix`, `RigidPose` or `HashMap` (FR-6). Adapters at the `robot-model` boundary convert types.

**Inputs/outputs**: Input: joint state (`q`, `qd`) and body inertias. Output: `M(q)` and the bias force `c(q, qd)`.

**Done when**: The pure recursions reproduce the existing ABA/RNE closed-form, round-trip and energy tests with unchanged expected values (Step 3 of the plan).

**Open questions**: none blocking.

---

## 4. Dynamics: mass matrix, bias forces, integration

1. The mass-matrix-and-bias-force structure MUST remain the Composite Rigid Body Algorithm (for `M`) plus Recursive Newton-Euler with zero acceleration (for the bias force `c`), the same pattern `robot_sim::articulated`'s existing `ArticulatedBody::inverse_dynamics` already implements. This is not a new requirement so much as a confirmed one: the research agent read the reference's own `mj_rne`/`mj_crb` directly and found the identical gravity-as-fictitious-base-acceleration trick (`a_0 = -g`, sign and all) our RNE already uses [the reference simulator §2] — independent, strong confirmation that this session's hand-derived implementation is structurally the same algorithm the reference's engine runs, not merely "a" Featherstone-style RNE that happens to also work.
2. One coordinate-frame divergence from the reference simulator MUST be recorded, not silently carried: the reference's own CRBA/RNE run in **global** coordinates (deliberately, to share machinery with collision detection), with local per-subtree-COM frames used only for floating-point accuracy [the reference simulator §2]. `robot_sim::spatial`/`articulated` work entirely in each body's own **local** frame via recursive pose composition on plain rotation/translation arrays — closer to the textbook Featherstone convention than to the reference's actual engine. Both are valid; this document records the choice (local frames, continuing the existing implementation) rather than reversing it to match the reference simulator exactly, since nothing in the GPU-batching or fidelity requirements depends on which frame convention is used internally.

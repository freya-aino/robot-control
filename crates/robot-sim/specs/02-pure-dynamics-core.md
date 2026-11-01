# Step 02 — Pure dynamics core (mass matrix, bias forces)

[Index](README.md) · [Appendix A](appendix-a-citations.md) · [Appendix C](appendix-c-existing-code.md)

**Status (updated 2026-11-01)**: CRBA and `M(q)` **are now implemented**, in `robot_sim::composite`: the composite-rigid-body mass matrix, its `L' D L` factorization over the degree-of-freedom tree, back-substitution, a dense view, and the bias force, all transferred from 02 §2.3, §2.9, §2.10, §2.11 and §3.3. They work in the example's world-orientation-at-tree-centre-of-mass frame rather than the body-local frame the paragraph below records, so both conventions now exist in the crate and the §4 item 2 divergence note is a statement about `articulated` only. The new path is cross-validated against the old one: `M` agrees with the mass matrix built column by column from local-frame inverse dynamics to 1e-10 on the 29-degree-of-freedom G1, and the bias force agrees to 1e-9. That is two different algorithms in two different frames, not a restatement. `InertiaTensor::is_physically_plausible` now tests the principal moments (Appendix D M-03). Original status below.

**Status**: **Implemented, in the example's frame convention, and
cross-validated.** `composite` is the example's own dynamics path: all of
`cdof`, `cinert`, `cvel` and `cacc` in **world orientation** with the reference
point at `subtree_com[body_rootid[b]]`, spatial six-vectors ordered `[omega; v]`,
per [example 02](../../../docs/example-system-specs/02_kinematics_dynamics_smooth.md).
It provides the position stage, CRBA (`mj_crb`), the `L' D L` factorization over
the degree-of-freedom tree (`mj_factorM`), back-substitution (`mj_solveLD`), the
velocity stage (`mj_comVel`), RNE bias forces, and forward dynamics as
`qacc = M^-1 (tau - qfrc_bias)`.

That last point is a design fact worth stating here rather than leaving
implicit: the example has **no articulated-body algorithm**. It forms `M`,
factorises it and solves. That execution model is what makes the correct
`Free`-joint convention expressible without a separate `S qdd` term, and it is
what supplies the dense `M` the implicit integrator of
[03](03-integrators.md) needs — Appendix D M-18 closed.

Validated by **cross-algorithm agreement** rather than assertion: `M(q)` from
CRBA against `M` built column-by-column from the independently-verified
local-frame recursion agrees to 1e-10 on the 29-degree-of-freedom G1; bias
forces to 1e-9; forward composed with inverse dynamics to 1e-8 on the 35-degree-
of-freedom floating G1; and the point Jacobian against a central finite
difference. `crate::articulated` keeps the body-local ABA/RNE path and is
retained precisely as that cross-check.

`inverse_inertia_diagonal` no longer costs `nv` forward-dynamics passes —
Appendix D M-24 closed.

**Goal**: Keep the mass-matrix-and-bias-force structure (CRBA for `M`, RNE with zero acceleration for the bias `c`) and state the local-frame convention. Covers old §4, items 1–2. Items 3–5 (integration) are in step 03.

**Depends on**: [01](01-state-and-coordinates.md).

**Reference formulation**: Tag `3.17.0`, commit `800a109a4bdd02cb60404d565db4ae16fb04a53b` (Appendix A, entry 2). Function definitions, located by name at the pin:
- `src/engine/engine_core_smooth.c`: `mj_rne` (recursive Newton-Euler, bias force) and `mj_crb` (composite rigid body, mass matrix).
- `src/engine/engine_forward.c`: the forward pipeline that calls them (same entry).

Evidence tag: `code_inspected` for the existence and names of `mj_rne` and `mj_crb` at the pin. The equivalence claim in old §4 item 1 (gravity-as-fictitious-base-acceleration) is `code_inspected` only as stated by the research pass (Appendix A, entry 2), not re-derived here.

**Purity contract**: The CRBA and RNE recursions use `[f64; 6]` / `[f64; 3]` arrays and explicit formulas. They MUST NOT use `robot_types::math::Matrix`, `RigidPose` or `HashMap` (FR-6). Adapters at the `robot-model` boundary convert types.

**Inputs/outputs**: Input: joint state (`q`, `qd`) and body inertias. Output: `M(q)` and the bias force `c(q, qd)`.

**Done when**: The pure recursions reproduce the existing ABA/RNE closed-form, round-trip and energy tests with unchanged expected values (Step 3 of the plan).

**Open questions**: none blocking for the recursions themselves. The "none blocking" above does **not** extend to step 03: [Appendix D](appendix-d-mathematical-challenges.md) M-18 records that the absent `M(q)` blocks step 03's implicit integrator, a dependency neither step stated.

**Challenges**: [Appendix D](appendix-d-mathematical-challenges.md) confirms both recursions correct term for term against Featherstone, including the `a_0 = -g` gravity trick (section 1.1). The challenges are around them: M-03 (**defect**) — `InertiaTensor::is_physically_plausible` tests the triangle inequality on the diagonal entries only, so it accepts an indefinite tensor (measured counterexample with eigenvalue `-1`), which matters because step 05 item 2 and [Appendix C](appendix-c-existing-code.md) both cite that function as the validation pattern to follow; decision routed to [Appendix B](appendix-b-open-questions.md) **S-08**. M-14 (**divergence**) — `solve_small_dense` uses an absolute pivot tolerance where `solve_dense` uses a relative one. M-18 and M-24 (**gap**, **cost**) — the missing `M(q)` blocks the implicit integrator and forces `inverse_inertia_diagonal` to spend `nv` forward-dynamics passes (35 on the G1) to recover one diagonal.

---

## 4. Dynamics: mass matrix, bias forces, integration

1. The mass-matrix-and-bias-force structure MUST remain the Composite Rigid Body Algorithm (for `M`) plus Recursive Newton-Euler with zero acceleration (for the bias force `c`), the same pattern `robot_sim::articulated`'s existing `ArticulatedBody::inverse_dynamics` already implements. This is not a new requirement so much as a confirmed one: the research agent read the reference's own `mj_rne`/`mj_crb` directly and found the identical gravity-as-fictitious-base-acceleration trick (`a_0 = -g`, sign and all) our RNE already uses [the reference simulator §2] — independent, strong confirmation that this session's hand-derived implementation is structurally the same algorithm the reference's engine runs, not merely "a" Featherstone-style RNE that happens to also work.
2. One coordinate-frame divergence from the reference simulator MUST be recorded, not silently carried: the reference's own CRBA/RNE run in **global** coordinates (deliberately, to share machinery with collision detection), with local per-subtree-COM frames used only for floating-point accuracy [the reference simulator §2]. `robot_sim::spatial`/`articulated` work entirely in each body's own **local** frame via recursive pose composition on plain rotation/translation arrays — closer to the textbook Featherstone convention than to the reference's actual engine. Both are valid; this document records the choice (local frames, continuing the existing implementation) rather than reversing it to match the reference simulator exactly, since nothing in the GPU-batching or fidelity requirements depends on which frame convention is used internally.

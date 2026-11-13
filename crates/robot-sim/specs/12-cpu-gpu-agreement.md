# Step 12 — Validation, evidence and CPU/GPU agreement

[Index](README.md) · [Appendix A](appendix-a-citations.md) · [Appendix B](appendix-b-open-questions.md)


**Goal**: Specify how every claim is validated and how the GPU backend is checked against the CPU reference. Covers old §14 in full.

**Depends on**: [10](10-cpu-reference-backend.md) (CPU reference backend), [11](11-gpu-batched-backend.md) (GPU-batched backend).

**Reference formulation**: JAX-based batched backend's correctness testing against CPU reference simulator (Appendix A, entry 12) is cited as a design precedent only, with evidence tag `code_inspected` for the pinned batched backend source and `unverified` for its results. No the reference simulator engine function is pinned here, since this step is validation and not an algorithm. The oracle plan that this step's checks follow is in [04](04-fidelity-loop.md). the reference simulator is not run, bound or sampled by this step, and no reference simulator output enters the repository. Any future binding-based cross-engine comparison remains subject to the `SPECS.md` Section 2.1 exception procedure, and is not requested here.

**Purity contract**: Validation code is test code. Algorithm modules remain pure math (FR-6). Tolerances are recorded with their scenario and tier, per FR-7.

**Status**: Partial (step 7). The CPU reference is no longer unoracled for
contact, which was this step's deepest problem: the fidelity battery is at
schema v8 with a plane force-balance, a friction-cone bound and a narrow-phase
depth row (Appendix D M-20 closed), and the two CPU solvers cross-check each
other to 1e-8 on the acceleration. A GPU backend therefore now has something to
agree *with* on contact, which it did not before.

Still open, and measured: Appendix D M-22, where the G1's `cond(M) ~ 1e5`
sets what an f32 tolerance can be. M-27 is resolved (0.1.7-draft), so the CPU
reference now has the G1's full contact geometry; a cross-backend contact
agreement test awaits a GPU contact solver.

Implemented: CPU-vs-GPU agreement in `crates/robot-sim-gpu/tests/cpu_gpu_agreement.rs` under `--features cubecl` only, on Mesa Lavapipe and Intel. It covers single-step agreement for the spring, damper, actuator, equality residual/Jacobian, `solimp` impedance with its derivative `impP` (asserted for `x <= mid` and `x > mid`, p = 2 and 3, and for `pos < margin`), plane-sphere contact (all coordinates and the normal) and semi-implicit Euler. The named pendulum RK4 kernel is asserted against the CPU `rk4_step` closure reference over 4097 environments and 1000 steps. The general closure-based RK4 is not ported and has no CPU fallback. `tests/implicit_agreement.rs` covers implicit-damping Euler (hand-derived and 4097-environment random batches, non-SPD status, validation). `tests/equality_agreement.rs` covers the equality-only dense solver: hand-derived scalar and coupled-inertia cases, both nonlinear impedance branches and direct/standard `solref` on 4097 environments at `(nv,nrow)` = (1,1), (1,2), (2,2), (2,4), (3,2), (6,6), plus the `(2,5)` edge rows; abs `1e-4` + rel `5e-6·|reference|` against the f64 CPU solver (same f32-rounded input). The measured worst error/bound is 0.2467 on Lavapipe and 0.2325 on Intel for the whole equality batch set, and 0.2262 on Intel for `(2,5)`. Malformed and oversize batches are rejected before upload, and bad-SPD environments retain their outputs. Tolerance records for the RK4 and impedance scenarios are in `crates/robot-sim-gpu/tolerances.toml`, which the feature-gated tests read and which `tests/tolerance_records.rs` checks in the default build; the random-SPD implicit and equality batch tolerances are recorded there too; the hand-derived unit tolerances remain in ADR 0012. Gate 4 is `tests/repeatability.rs`: 5 identical in-process runs of the RK4 and equality batches, bitwise, on one device, not cross-device. Gate 3 (timing and count-only resources) is `examples/gpu_resource_repeatability.rs`, RK4 and equality only, with no throughput claim. These are GPU-vs-CPU agreement records, not `fidelity.toml` rows. Not done: fidelity-tier validation, contact exhaustion, the general RK4 closure, or a full contact/friction/limit solver. The CPU closed-form, round-trip and energy checks remain CPU-only.

**Inputs/outputs**: Input: CPU reference output and GPU output on matched fixtures and configuration. Output: an agreement report with a recorded tolerance and the failing rows, if any.

**Done when**: Every fidelity tier has energy and closed-form validation. The GPU-to-CPU agreement check runs within a recorded tolerance. The contact-exhaustion counter is exercised by a deliberately over-budget test. No the reference simulator runtime, binding or trace is part of this step; any future binding-based cross-engine comparison remains subject to the `SPECS.md` Section 2.1 exception procedure. The step 2 ADR (0011) records the source-reference decision.

**Open questions**: Appendix B S-01 and S-02 are blocking for this step's agreement and tolerance items, since this step is where their results are recorded. See [Appendix B](appendix-b-open-questions.md).

**Challenges**: [Appendix D](appendix-d-mathematical-challenges.md) M-22 (**gap**) is shared with [11](11-gpu-batched-backend.md) and bears on this step's tolerances specifically. The G1's joint-space conditioning is measured at `cond(M) >~ 1e5` (via `(M^-1)_jj` spanning 3.30e-2 to 3.16e3 at the home configuration), against `f32`'s ~1.2e-7 relative precision. An agreement tolerance for a dense `f32` factorisation of that matrix has to be set against roughly two surviving significant digits — a number worth deriving here rather than discovering as a test failure. The example sets per-stage tolerances for its single-precision port separately from its double-precision reference (`15` §11), which `docs/example-system-mapping.md` §2.8 lists as candidate work; the conditioning figure above is what such a per-stage tolerance has to be justified against.

Appendix D M-20 is also relevant to this step's evidence chain: the CPU reference the GPU backend is checked against has no archival oracle coverage for contact, collision or constraint, so cross-backend agreement in those subsystems would establish that the two backends agree, not that either is right.

---

## 14. Validation and evidence requirements

1. Every claim this document or its implementation makes about `robot-sim`'s own behavior starts at `unverified` per `SPECS.md` §3, promoted only by the evidence that table requires — a reference simulator/batched backend/Brax citation never promotes a claim about our own code, no matter how closely the design follows the cited precedent.
2. The GPU-batched backend's output MUST be checked against the CPU reference backend (11.4) on shared fixtures at matched configuration, following the same spirit as JAX-based batched backend's own correctness testing against CPU reference simulator [batched backend §12] — agreement within a stated, recorded tolerance, not an assumption of agreement.
3. Every fidelity tier (12.2) MUST have its own energy-conservation and closed-form validation, following the pattern this session already established for the CPU-only solver (hand-derived pendulum closed form, RNE/ABA round-trip, multi-link free-motion energy conservation) — a cheap/fast tier that fails conservation checks a costlier tier passes is a real, reportable fidelity cost (Section 12.1's own framing), not a defect to hide.
4. The contact-slot-exhaustion counter (11.2) MUST be exercised by a test that deliberately exceeds the configured budget and confirms the counter increments rather than the simulation silently continuing as if nothing were dropped.
5. Cross-simulator or cross-engine comparison (against the reference simulator itself, via bindings) remains outside this document's scope and subject to `SPECS.md` Section 2's foreign-dependency exception procedure exactly as `SPECS.md` §9.1 already states; nothing here requests or assumes that exception.

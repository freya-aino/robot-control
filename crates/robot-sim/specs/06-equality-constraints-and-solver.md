# Step 06 — Equality constraints and the constraint solver

[Index](README.md) · [Appendix A](appendix-a-citations.md)

**Status**: Partially implemented (pure CPU subset). `robot_sim::constraint` ports the `mjEQ_JOINT` equality build (scalar joints, one or two joints with the quartic coupling), the `solref`/`solimp` impedance and stiffness/damping, the `2*timestep` time-constant clamp, and a closed-form linear solve of an equality-only problem. `robot_sim::contact::plane_sphere_contact` ports the plane-sphere geometry (`+z` plane). Both are standalone pure functions and are not wired into `MultibodyState::step`. Not ported: the contact solver with friction cones, the iterative `mj_solNewton`/`mj_solCG`/`mj_solPGS`, `mjEQ_CONNECT`/`mjEQ_WELD`/`mjEQ_TENDON`, limit and friction rows, warm-starting. The single-point `GroundContact` penalty model is unchanged and remains the baseline (Appendix C).

**Goal**: Solve contact and equality constraints as one convex problem, with soft-constraint impedance (`solref`/`solimp` equivalent), a time-constant safety check, at least two solver algorithms, and warm-starting. Covers old §5 in full and §7 items 2–3.

**Depends on**: [03](03-integrators.md) (time-constant check relies on the integrator timestep), [05](05-passive-forces-and-actuators.md) (passive-force and actuator terms enter the same problem).

**Reference formulation**: Tag `3.17.0`, commit `800a109a4bdd02cb60404d565db4ae16fb04a53b` (Appendix A, entry 3). Function definitions, located by name at the pin:
- `src/engine/engine_solver.c`: `mj_solNewton`, `mj_solCG`, `mj_solPGS` (the three solver algorithms). `mj_dualFinish` is a public function in the same file. `mj_solPrimal` is `static`.
- `src/engine/engine_forward.c`: `mj_fwdConstraint` (constraint stage of the forward pipeline).
- `doc/computation.rst` (same entry) for the formulation and `solref`/`solimp`.

Evidence tags: `code_inspected` for the bodies of the ported functions (read at the pin): `mj_instantiateEquality` `mjEQ_JOINT` case (`engine_core_constraint.c`, lines 628-705), `getsolparam` (lines 1853-1927), `mj_makeImpedance` `R` formula (lines 2036-2057), `mj_makeARNumeric` (line 3063), `mj_referenceConstraint` (line 3299), `mjraw_PlaneSphere` and `mjc_PlaneSphere` (`engine_collision_primitive.c`, lines 28-62). `unverified` for every other formula-level claim in this step, including `mj_solNewton`, `mj_solCG`, `mj_solPGS`, and the default-solver claim (Appendix A, entry 3). The batched backend solver counterpart is `mjx/.../solver.py` (Appendix A, entry 12).

**Purity contract**: Solver math uses `std` and `[f64]` arrays with explicit formulas. No `Matrix`, `RigidPose` or `HashMap` (FR-6). Pure factorizations may be shared between algorithm modules: `constraint.rs` reuses `integrators::cholesky_solve`, a plain `[f64]` routine, not a project model helper. Model types and project utilities remain excluded.

**Inputs/outputs**: Input: constraint Jacobians, residuals, `solref`/`solimp` parameters, timestep and the previous step's forces (for warm-start). Output: constraint forces and the resulting accelerations.

**Done when**: Contact and equality constraints share one solver. The time-constant check rejects or reports a constraint whose time-constant is below about twice `dt`. Newton-style and CG-style solvers both exist, and warm-start is measured against the cold start.

**Step 6 status against Done when**: the time-constant check is implemented and reports clamped rows (`effective_timeconst`, `EqualitySolution::clamped_rows`). The equality-only linear solve exists. The shared contact-plus-equality solver, Newton, CG and warm-start are not implemented.

**Deviations from the pinned source (recorded per FR-4)**:
- `R` uses the exact diagonal `A_ii` of `J M^-1 J^T`. This matches the `mjENBL_DIAGEXACT` path (`mj_makeYNumeric`, `mj_projectConstraint`), not the default `mj_diagApprox` (`engine_core_constraint.c` line 1605). For `mjEQ_JOINT` rows `mj_diagApprox` sums `dof_invweight0`, the inverse inertia at `qpos0` with no `poly'(d)^2` factor. So `R` differs from the default path whenever `|poly'(d)| != 1` or `M(q) != M(qpos0)`, even for diagonal `M`.
- A mixed-format `solref` is refused with `MixedSolrefFormat`. the reference simulator warns and replaces it with the default.
- The `2*timestep` clamp is unconditional. the reference simulator skips it under `mjINT_DISCRETE` (`mj_isMetric`), which this port has no path for. the reference simulator also skips it when `mjDSBL_REFSAFE` is set (`getsolparam`, line 1906); this port has no disable switch.
- A mass matrix that is not symmetric is refused with `NotSymmetric`. `cholesky_solve` reads only the lower triangle, so an asymmetric input would otherwise be misread silently.
- `plane_sphere_contact` fixes the normal to `+z` and omits `con->tangent`. The source takes the normal from `mat1[2,5,8]`.

**Not yet ported from `engine_core_constraint.c`**: `mjEQ_CONNECT`, `mjEQ_WELD`, `mjEQ_TENDON` build rows (need body or tendon kinematics), `mj_Jdotv` (needed only for the connect and weld Jdot term), friction, limit and contact rows, `mj_diagApprox`, `mj_makeARSymbolic` sparse pattern, and the iterative solvers in `engine_solver.c`. `mjc_PlaneCapsule`, `mjc_PlaneCylinder`, `mjc_PlaneBox`, `mjc_SphereSphere` and the other collision pairs in `engine_collision_primitive.c` are not ported.

**Open questions**: none blocking for the CPU solver. Appendix B S-02 (CubeCL control flow for an iterative solver) concerns the GPU backend and is owned by [11](11-gpu-batched-backend.md).

---

## 5. Contact and constraint solver

1. Contact and all equality constraints (Section 7) MUST be solved as one problem, following the reference's own reformulation: a convex optimization (QP for pyramidal friction cones, a conic program for elliptic cones) over constraint forces, not a classical LCP [the reference simulator §3]. the reference's own stated rationale — dropping strict complementarity lets soft/penetrating contact be modeled as physically correct rather than a numerical defect, and avoids LCP-with-friction's worst-case NP-hardness — is adopted as this project's rationale too, not merely reported as the reference's opinion.
2. A soft-constraint mechanism equivalent to `solref`/`solimp` MUST exist: a per-constraint impedance value controlling how strongly a violation can generate force, and a reference-acceleration spring-damper on the constraint residual itself (not on the bodies), parameterized by a time-constant/damping-ratio pair in the common case and raw stiffness/damping in the direct case (needed for e.g. perfectly elastic restitution) [the reference simulator §3].
3. `robot-sim` MUST implement the internal safety check the reference simulator itself relies on: a constraint's effective time-constant must exceed roughly twice the simulation timestep, or the constraint becomes too stiff for the integrator to resolve stably — the reference's own documentation names this directly as the mechanism that prevents exactly the kind of timestep/stiffness divergence this session hit empirically [the reference simulator §3]. `robot-sim` currently has no analogous check anywhere.
4. Both friction-cone models (pyramidal and elliptic) SHOULD be supported, with the explicit, counter-intuitive fact from Section 3's research recorded in the implementation's own documentation: a pyramidal cone with more edges does not converge to the elliptic-cone solution for soft contacts — they define genuinely different soft-contact dynamics, not points on one approximation-quality spectrum [the reference simulator §3]. Contact dimensionality (`condim` 1/3/4/6) MUST be a per-contact, queryable fidelity parameter (Section 12).
5. At least two solver algorithms MUST be available: a Newton-style method (exact second derivatives, few iterations needed, the reference's own documentation for batched backend specifically endorses "often just one" iteration as sufficient for RL-training fidelity [batched backend §15]) and a Gauss-Seidel/CG-style iterative method retained for comparison and for any execution context where Newton's per-iteration cost is unaffordable. Classical PGS MAY be retained for the CPU reference backend; it MUST NOT be required of the GPU-batched backend, since JAX-based batched backend itself drops PGS specifically because its strictly-sequential per-constraint update does not vectorize the way CG/Newton's linear algebra does [batched backend §12] — direct precedent for the general batching rule Section 11.1 states.
6. Solver iteration count and convergence tolerance MUST be per-run, queryable fidelity knobs (Section 12).
7. Warm-starting (initializing each step's solve from the previous step's constraint forces, falling back to the unconstrained solution if it scores worse) SHOULD be implemented; the reference's own documentation credits it with eliminating "nearly the entire cost of the constraint solver" in a quiescent scene [the reference simulator §3].

## 7. Actuators, tendons, equality constraints

2. Equality constraints (connect/point-to-point, weld/full rigid lock, a polynomial joint-coupling equivalent to and strictly more general than URDF's rejected `mimic`, and a tendon-length equivalent) MUST be solved through the *same* constraint solver as contact (Section 5), not a separate mechanism — this is a the reference simulator design choice (one unified Jacobian/optimization problem) [the reference simulator §6], and splitting it into two solvers would be a step backward from what research found, not a simplification.
3. A weld constraint's orientation residual MUST use an exponential-map-style encoding (`sin(θ/2)·axis`, matching the reference's own choice) rather than a naive Euler-angle or raw-quaternion-difference residual, specifically to avoid wrap-around error at large angles [the reference simulator §6].

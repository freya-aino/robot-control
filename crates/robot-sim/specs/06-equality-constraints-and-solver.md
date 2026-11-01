# Step 06 — Equality constraints and the constraint solver

[Index](README.md) · [Appendix A](appendix-a-citations.md)

**Status**: **Implemented, both algorithms, and running.** `robot_sim::solver`
and `robot_sim::newton` solve contact, joint-limit and equality rows as one
convex problem, and `World` runs it as the stepper behind `robot sim multibody`.
Every item of old Section 5 is met except the elliptic cone (item 4's second
half) and `condim` 4 and 6.

The two algorithms item 5 requires both exist, and they are each other's
cross-check — they solve the same problem by different routes and agree on the
acceleration to 1e-8:

- **Dual, projected Gauss-Seidel** (`solver::solve`), over constraint forces:
  `min_{f in Omega} 0.5 f' (A + R) f + f' (a_u - a_ref)` per
  [example 07](../../../docs/example-system-specs/07_constraints.md) section 2.3,
  with `A = J M^-1 J'` from the composite factorization,
  `R_ii = (1-d)/d A_ii` on the exact diagonal, `a_ref = -B (Jv) - K d (pos -
  margin)`, `a_u = J qacc_smooth + Jdot v`, and warm starting.
- **Primal Newton** (`newton::solve_newton`), over the acceleration, per
  [example 08](../../../docs/example-system-specs/08_solvers_islands.md) section
  6: `min_a 0.5 (a - a0)' M (a - a0) + s(J a - a_ref)` with the
  piecewise-quadratic constraint cost of section 2, gradient
  `M a - qfrc_smooth - J' f`, Hessian `M + J' diag(D_q) J` (section 6.10), and
  an exact line search. **This is the default**, for a measured reason below.

Both carry the same per-row impedance, built once by `solver::row_impedance`:
the dual needs all of `A`, the primal only its diagonal, because that is the
only place `A` enters `R`.

**Why Newton is the default, measured.** The pyramidal representation makes the
dual's rows redundant by construction — four non-negative coefficients encode
three force components and every edge of a contact contains the normal — so four
contacts on a six-degree-of-freedom body give sixteen rows for a rank-six
problem. Gauss-Seidel needs **1011 sweeps** there to bring the per-sweep force
change below 1e-10, against a default budget of 100 at which the residual is
still 1.1e-2. Newton reaches the same answer in **one** iteration at a scaled
gradient norm of 4.9e-14, because its variable is the acceleration, one per
degree of freedom however many contacts there are, and its Hessian is positive
definite because `M` is. At 100 sweeps the dual's *acceleration* is already
within 2e-4 relative, so the budget was never wrong for stepping — it is the
forces that were unsettled. Appendix D section 1.7.3 records the measurement.

This is a stronger statement than item 5's, which cites the batched backend's
"often just one iteration" as a fidelity claim about RL training. Here it is a
convergence requirement on the CPU reference as well.

**The friction cone is the example's pyramid, not a box.** An earlier pass
emitted one normal row plus two tangential rows and clamped each tangential
force to `mu f_n` inside the solver. That is a cone, and it couples friction to
the normal solve, but it is the pyramid whose cross-section is a **square** of
half-width `mu f_n`, which circumscribes the true circular cone and admits up to
`sqrt(2) mu f_n` diagonally — about 41 per cent too much friction, in the
direction a foot slides. Example 07 section 9.2 builds the rows as the cone's
**generator directions** instead: for `k = 1..dim-1`,
`row+- = jac[0] +- friction[k-1] jac[k]`, so `J = n +- mu t_k`, giving
`2(condim-1)` rows that each carry a non-negative coefficient and all carry
`pos = dist`. With `lambda >= 0` the realised force satisfies
`|f_t1| + |f_t2| <= mu f_n` — the inscribed diamond, conservative rather than
permissive. `solver::decode_pyramid` inverts it.

A consequence worth stating as specification rather than implementation detail:
**every row kind is now either unbounded or non-negative**, because the cone
lives entirely in the row Jacobians. Neither solver contains a cone projection.

Not implemented: the elliptic cone and its `H_c` Hessian block, `condim` 4 and
6, conjugate gradient (item 5 is satisfied by Newton plus Gauss-Seidel, so CG is
optional rather than owed), islands, adhesion, friction-loss rows, and the
sparse Hessian — `newton` forms a dense `nv x nv` Hessian and refactorises it
each iteration rather than updating it rank-1 as `HessianIncremental` does, a
cost and not a difference of result.

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

**Status against Done when**: all four clauses are met. Contact and equality
constraints share one solver; the time-constant check reports clamped rows
(`effective_timeconst`, `Solution::clamped_rows`); a Newton-style and a
Gauss-Seidel-style solver both exist and agree to 1e-8; and warm start is
measured against the cold start. CG is not implemented and is no longer owed.

**Deviations from the example, re-read under the MUST precedence of
[Appendix D](appendix-d-mathematical-challenges.md) section 1.0.2.** Each entry
below is now classified, because a "deviation" is no longer something a reader
may take as settled: a competing formulation is a defect and is overwritten,
while a stricter input contract is validation and is kept.

*Formulation, kept because it is the example's own better path:*

- `R` uses the exact diagonal `A_ii` of `J M^-1 J'`. This is the example's
  `mjENBL_DIAGEXACT` path (`mj_makeYNumeric`, `mj_projectConstraint`), not its
  default `mj_diagApprox`, which for `mjEQ_JOINT` rows sums `dof_invweight0` —
  the inverse inertia at `qpos0` with no `poly'(d)^2` factor. The two differ
  whenever `|poly'(d)| != 1` or `M(q) != M(qpos0)`. Choosing the exact diagonal
  is choosing the more accurate of two paths the example provides, not diverging
  from it, and it is what let M-09 close: the G1's effective masses span about
  five orders of magnitude, so no single approximation stands in across them.

*Validation, kept and re-labelled — the same mathematics, refusing an input
rather than repairing it:*

- A mixed-format `solref` is refused with `MixedSolrefFormat`. The example warns
  and substitutes its default. Refusing to compute is not computing
  differently, and a silently substituted default is exactly the class of thing
  this project's evidence discipline exists to prevent.
- An asymmetric mass matrix is refused with `NotSymmetric`. The example's
  factorisation reads only the lower triangle, so an asymmetric input would be
  half-read without complaint.
- The `2*timestep` time-constant clamp is unconditional. The example skips it
  under `mjINT_DISCRETE` (`mj_isMetric`) and under `mjDSBL_REFSAFE`
  (`getsolparam`), and this port has neither an implicit-discrete integrator nor
  a disable switch — so there is no configuration in which the two differ. If
  either is added, the skip comes with it rather than being decided again.

*Overwritten, and no longer a deviation:*

- `contact::plane_sphere_contact` fixed its normal to `+z` and measured the
  sphere centre's height as `center[2] - plane_height`. Example 04 section 1.1
  takes the normal from the plane geom's own frame and measures
  `cdist = dot(pos2 - pos1, n)`; the two agree only for a plane whose frame is
  the world's. The general form replaced it.
- The per-point penalty contact law, and the `--contact penalty|impedance`
  selector that reached it, are gone. M-04 and M-09 record why it was a
  competing formulation rather than a cheaper setting.

*A cost gap, not a formulation difference:* `newton` forms a dense `nv x nv`
Hessian and refactorises it each iteration, where `HessianIncremental` updates
it rank-1 and `FactorizeHessian` keeps a sparse pattern. Same answer, more work;
it is the first thing to change if the solver becomes a bottleneck, and it is
also what [11](11-gpu-batched-backend.md) needs.

*One implementation choice stated rather than hidden:* section 6.8's line search
is iterative — a Newton march to bracket the minimum, then bisection against a
tolerance. The cost along a search direction is a convex piecewise quadratic in
`alpha` whose breakpoints are where each inequality row's `jar` crosses zero, so
`newton` sorts the breakpoints and minimises the quadratic on each segment,
finding the minimiser **exactly**. It computes what section 6.8 approaches; the
mathematics is the example's and the iteration sequence is not.

**Not yet ported from `engine_core_constraint.c`**: `mjEQ_CONNECT`, `mjEQ_WELD`, `mjEQ_TENDON` build rows (need body or tendon kinematics), `mj_Jdotv` (needed only for the connect and weld Jdot term), friction, limit and contact rows, `mj_diagApprox`, `mj_makeARSymbolic` sparse pattern, and the iterative solvers in `engine_solver.c`. `mjc_PlaneCapsule`, `mjc_PlaneCylinder`, `mjc_PlaneBox`, `mjc_SphereSphere` and the other collision pairs in `engine_collision_primitive.c` are not ported.

**Open questions**: none blocking for the CPU solver. Appendix B S-02 (CubeCL control flow for an iterative solver) concerns the GPU backend and is owned by [11](11-gpu-batched-backend.md).

**Challenges**: this step owned more of
[Appendix D](appendix-d-mathematical-challenges.md) than any other, and the
ported equality algebra was confirmed correct from the start — the impedance
sigmoid and its derivative, the stiffness/damping map, the `2*timestep` clamp,
and the substitution giving `(A + diag(R)) f = aref - J qacc_smooth`
(Appendix D section 1.1).

**All of M-08 to M-13 and M-23 are closed**, each by the unified problem above
rather than by a local patch:

- M-09, contact impedance referred to a constant 1 kg instead of the row's own
  `A_ii`: the exact diagonal is used for every row kind.
- M-10, friction applied outside the coupled solve with no cone and no
  `condim`: friction rows are in the same problem, as pyramid edges, with the
  contact frame from `mju_makeFrame` rather than a hard-coded `z`.
- M-11, the unconstrained acceleration omitting `Jdot qd`: `a_u` carries it.
- M-12, the limit solve and the contact solve not seeing each other: one
  problem, one solve.
- M-08, the hard-limit position clamp discarding kinetic energy: limits are rows
  with the same impedance as any other, and `SimOptions::limit_margin` gives
  them room to act. Measured: a G1 joint driven at 50 N m into its bound
  overshoots by 0.177 rad with no margin and by **exactly zero** with a 0.2 rad
  margin, because with no margin a row appears only once `dist < 0` and the
  constraint brakes the joint after it has crossed.
- M-13, two different `solref`-format predicates: one predicate.
- M-23, refactorising `M` once per row: factor once, substitute per row.

Two findings came out of making the solver actually run, and they are the live
content of this step rather than history:

- **`mju_cholFactor` clamps, it does not fail** (example 08 section 7.1): a
  pivot below `mindiag` is set to `mindiag`, the rank is decremented and that
  column below the diagonal is zeroed. `D = 1/R` reaches 1e15 when a row's
  regularisation sits on its floor, so `M + J' D_q J` can lose a pivot to
  cancellation even though it is positive definite exactly.
- **A contact with zero friction must be `dim == 1`.** `condim = 3` with
  `mu = 0` builds four *identical* pyramid edges, since `n +- mu t_k` all
  collapse onto `n`, while `Rpy = 2 mu^2 R0` is zero so `D` saturates at
  `1/MINVAL`. Example 07 section 9.2 already says what to do: a frictionless
  contact gets one row along the normal.

And one methodological point, recorded because it shaped the debugging: **the
dual solver's non-convergence had been acting as damping.** It never reached the
true force in 100 sweeps, so it under-estimated it, and that under-estimate
masked the degenerate rows above. A solver that converges is what surfaced them.
A simulation that does not blow up is weak evidence about its own solver.

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

2. Equality constraints (connect/point-to-point, weld/full rigid lock, a polynomial joint-coupling equivalent to and strictly more general than URDF's rejected `mimic`, and a tendon-length equivalent) MUST be solved through the *same* constraint solver as contact (Section 5), not a separate mechanism — this is a reference simulator design choice (one unified Jacobian/optimization problem) [the reference simulator §6], and splitting it into two solvers would be a step backward from what research found, not a simplification.
3. A weld constraint's orientation residual MUST use an exponential-map-style encoding (`sin(θ/2)·axis`, matching the reference's own choice) rather than a naive Euler-angle or raw-quaternion-difference residual, specifically to avoid wrap-around error at large angles [the reference simulator §6].

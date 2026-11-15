# 0018. A project-authored box-constrained QP qualifies the differential-QP IK step

- Date: 2026-11-15
- Touches: `crates/motion-retarget/src/qp.rs` (new),
  `crates/motion-retarget/src/ik.rs` (`StepSolver`, `IkConfig.step_solver`,
  `IkError::Qp`), `crates/motion-retarget/src/config.rs` (`[solver] method`),
  `SPECS.md` (C-09 status).

Status: accepted.

## Context

SPECS.md Section 8.3: "Qualify a Rust-native constrained solver before
selecting differential-QP IK or whole-body optimization", with measured
setup, solve time, residuals and failure detection on representative
full-body constraints, bounded iterations/deadlines, and a validated
response to infeasibility or timeout. Appendix C's C-09 tracked the open
question; Appendix B.2.8's candidates (Clarabel first among them) are all
`unverified`, outside the empty default closure (ADR 0003), and each would
need the full Appendix B.6 dependency gate.

Meanwhile the IK in `motion-retarget` enforced joint limits by projecting
each damped-least-squares iterate back into the box. Projection holds the
limits exactly, but it distorts the step direction at an active limit: the
unconstrained step is computed as if the limited joint could help, then
clamped, so the remaining joints never see the work they must take over
within the same iteration.

## Decision

Write the solver this problem actually needs, in this repository, rather
than qualifying a general-purpose dependency for it.

One differential-IK step with hard limits is exactly a **dense, strictly
convex, box-constrained QP**: `min ||J s + r||^2 + lambda ||s||^2` subject
to per-variable bounds from the joint limits and the trust region. That
problem class needs no sparsity, no general linear rows, no conic cones —
so the heavy candidates of B.2.8 are not required by anything Section 8.3
asks of IK. `motion_retarget::qp::solve_box_qp` implements the textbook
primal active-set method for bounds (Nocedal & Wright 16.5, specialized),
`std`-only, about 250 lines, with:

- **bounded iterations**: a caller-supplied cap, `IterationCap` reported
  rather than looped on; the iterate is feasible in every outcome;
- **validated infeasibility**: `lo > hi` (and NaN bounds) refused as
  `InfeasibleBounds` before any work — and the IK's own subproblem is
  constructed from a feasible iterate, so its box always contains zero;
- **KKT-checked optimality**: the test suite checks the Karush-Kuhn-Tucker
  conditions (which for a strictly convex QP are global optimality) on 200
  randomized problems with mixed finite and infinite bounds, plus the
  blocking, releasing and cap edge cases.

`ik::StepSolver` selects the step: `DampedLeastSquares` (the retained C-09
baseline, and still the default) or `BoxQp`, reachable from a
`retarget-map/1` file as `[solver] method = "qp"`. The QP step needs no
post-hoc projection; the limits are constraints of the solve itself.

## Measurement (Section 8.3's qualification)

Representative full-body constraints: the 29-DOF vendored G1, 6-frame punch
retarget with a relative-reach task, two planted feet at weight 100 and a
free base —
`cargo test -p motion-retarget --release the_two_step_solvers -- --nocapture`
(Linux 6.18.56, the devenv.lock toolchain, 2026-11-15):

| Step solver | 6-frame retarget | Worst position error |
| --- | --- | --- |
| `DampedLeastSquares` | 16.0 ms | 0.004139 m |
| `BoxQp` | 11.4 ms | 0.004139 m |

Identical acceptance (reach preserved within 2 cm, limits held in every
frame, planted-foot slip under 2 cm — the pipeline test asserts all three
for both solvers), and the QP step is measurably cheaper here: solving the
limits inside the step saves outer iterations, which dominate through the
finite-difference Jacobian. Allocations are per-iteration `Vec`/`Matrix`
buffers in both paths; neither allocates per joint, and no allocation-free
claim is made for either.

The solve remains bounded by `IkConfig.max_iterations` and the wall-clock
`deadline`, and `Termination` reports which bound fired, unchanged.

## Consequences

- C-09's answer is no longer "no solver": a qualified Rust-native
  constrained solver exists for the IK problem class, project-authored, and
  the non-QP baseline is retained as the default until the QP step has
  history on real clips. Clarabel and the other B.2.8 candidates stay
  `unverified` and would still need the B.6 gate — this ADR qualifies
  nothing beyond the box-constrained class `qp.rs` implements.
- Whole-body optimization with general linear constraints (self-collision
  planes, torque rows) stays open; the active-set structure extends to
  general inequalities, but that extension is new work and a new
  qualification, not implied by this one.
- The `retarget-map/1` `[solver]` table accepts a `method` key; existing
  maps are unchanged (the key is optional and defaults to the baseline).

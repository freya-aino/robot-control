# Appendix D — Mathematical and algorithmic challenges to the current implementation

[Index](README.md) · [Appendix A](appendix-a-citations.md) · [Appendix B](appendix-b-open-questions.md) · [Appendix C](appendix-c-existing-code.md)

Status: **audit record**. Every entry below is a challenge to what the code
does today. No entry changes a requirement in steps 00–12, adds an acceptance
row, or amends `SPECS.md`. Where an entry says the implementation contradicts a
requirement, the requirement governs and the code is the thing out of
agreement; where an entry says a requirement is silent, the gap is recorded and
routed to [Appendix B](appendix-b-open-questions.md), not resolved here.

## 1. What this appendix is and how it was produced

The simulation code of `robot-sim`, `robot-sim-gpu` and the mass-property and
joint layers of `robot-model` it depends on were read in full against two
references at once:

1. The step files 00–12 of this directory, which state what the subsystem is
   required to do.
2. `docs/example-system-specs/` (17 documents; see
   `docs/example-system-mapping.md` for how that folder came into use). The
   example describes a complete, working rigid-body and contact simulator in
   equations and algorithm steps.

### 1.0.1 Precedence, changed 2026-11-01

When this appendix was first written the example folder was non-normative, and
rule 3 of the mapping document governed: where the example and a parent
requirement disagreed, the parent won. **That is no longer the case.** By
explicit direction, `docs/example-system-specs/` is now the **single source of
truth for the mathematics and the execution model of multi-body simulation**.

What that changes, and what it does not:

- **Mathematics and algorithms**: the example governs. A divergence from its
  formulation is a defect to fix, not a question to ask, and the deviations
  recorded under step 06 and step 02 are re-read in that light.
- **Everything else is unchanged**: the project's own goals, scope, normative
  language, evidence discipline (`SPECS.md` Section 3), the std-only dependency
  closure (`SPECS.md` Section 2), the Rust-native requirement, and the standing
  requirement that the backend end up GPU-bound and batched
  (`00-target-and-constraints.md` Section 1.2). The example is read for how to
  compute, not for what to build.
- Rule 2 of the mapping document still holds: the example's numerical
  constants, defaults and tolerances belong to the engine it describes and are
  not transplanted. Adopting one needs its own measurement.

This resolved one open question outright. Appendix B **S-07** asked whether to
change the `Free` joint's velocity convention or to amend step 01 item 3 to
match the code. Under the new precedence the question does not arise: the
example's convention is the mathematics, so the code changed.

Files read in full: `robot-sim/src/{spatial,articulated,collision,contact,constraint,integrators,passive,oracle}.rs`,
`robot-sim-gpu/src/{integrators,contact,constraint,passive,launch}.rs`,
`robot-model/src/{link,joint,kinematics}.rs`, and the check and tier definitions
of `robot-sim/src/fidelity.rs`.

### 1.0 Which tree was audited

The audit read the **working tree** at commit `c680e09`
(`feat(robot-cli): simulate the G1 on the USD model and write USD replays`)
**including uncommitted modifications** to `robot-sim/src/articulated.rs`,
`robot-sim/src/collision.rs`, `robot-model/src/usd.rs` and two `robot-cli`
files. That is deliberate: it is the current state of the implementation, which
is what the audit was asked about. It is not the committed state, and the
difference is load-bearing for several entries.

Specifically, the following are present only in that uncommitted work and so are
not in `c680e09` itself: `ContactModel` and its `Impedance` path,
`ArticulatedBody::{contact_rows, contact_externals, add_ground_friction,
add_row_wrench, row_normal_velocity}`, and `collision::{box_sat,
box_box_contacts}`. Entries [M-04](#m-04), [M-07](#m-07), [M-09](#m-09),
[M-10](#m-10), [M-11](#m-11), [M-15](#m-15), [M-16](#m-16) and [M-21](#m-21)
refer to code in that set. A reader on a tree without those changes will not
find the named functions; the entries are not stale, they are ahead of the
commit. Entries [M-01](#m-01) to [M-03](#m-03), [M-05](#m-05),
[M-06](#m-06), [M-08](#m-08), [M-12](#m-12) to [M-14](#m-14),
[M-17](#m-17) to [M-20](#m-20) and [M-22](#m-22) to [M-24](#m-24) apply to the
committed state as well.

The measurements in this appendix were taken on the audited working tree.
Where a measurement used a fixture or helper that exists only in the
uncommitted work it is noted in the entry; none of the four measurement sets
reported here did.

### 1.1 What the audit confirmed as correct

Recording this matters as much as the challenge list, because several entries
below are about things built *on top of* a core that is sound. The following
were derived independently and agree with the implementation:

| Subsystem | Checked against | Result |
| --- | --- | --- |
| `spatial::transform_motion` / `transform_force` | Featherstone motion/force duality; power invariance | Agree. The force transform is the transpose dual of the motion transform |
| `spatial::cross_motion_motion` / `cross_motion_force` | Featherstone's `ad` and `ad*` operators | Agree, including the operator's block structure and signs |
| `spatial::inertia_from_mass` | Featherstone eq. 2.63 and the parallel-axis closed form | Agree. The `skew(c) skew(c)^T` product equals `\|c\|^2 E - c c^T` |
| `spatial::inertia_transform_to_parent` | Congruence `X^T I X` | Agree. Evaluating column-by-column on the six basis twists is the congruence |
| `spatial::rotation_from_quat` | `(w^2 - \|u\|^2) E + 2 u u^T + 2 w [u]x` | Agree |
| `articulated::inverse_dynamics` | Featherstone RNE, with `a_0 = -g` | Agree term for term: `a_i = X a_lambda + S qdd + v_i x S qd`, `f_i = I a_i + v_i x* I v_i`, `tau = S^T f`, `f_lambda += X^T f_i`. This is the same recursion and the same gravity sign the example records for its own bias-force pass (`02` §3.3) |
| `articulated::aba_with_externals` | Featherstone ABA, Table 7.1 | Agree in all three passes, including `p^a = p^A + I^a c + U D^-1 u` using the *reduced* `I^a`, and the `k = 0` degenerate branch |
| `articulated::integrate_free_position` orientation term | `qdot = 0.5 q (x) omega_local` | Agree. The Hamilton product is expanded correctly |
| `articulated::solve_dense` | LU with partial pivoting, relative pivot tolerance | Agree |
| `integrators::rk4_step` | Classical RK4 on the second-order system | Agree, including that the position update uses the `B`-weighted mean velocity rather than the end-of-step velocity |
| `integrators::cholesky_solve` | Cholesky factorisation, forward and back substitution | Agree. Only the lower triangle is read, which the callers' symmetry refusal covers |
| `constraint::impedance` | The example's impedance sigmoid and its derivative (`07` §12.3) | Agree, including the sign carried onto `impP` and the `/width` factor |
| `constraint::solve_equality` algebra | `(A + diag(R)) f = aref - J qacc_smooth` from `J qacc - aref = -R f` | Agree. The substitution and the final `qacc = qacc_smooth + M^-1 J^T f` are consistent |
| `collision::closest_point_triangle_origin` | Ericson §5.1.5 | Agree, including the `va`/`vb`/`vc` edge-region tests |
| `collision` support maps (sphere, capsule, box, cylinder) | Support function of each primitive | Agree |
| `contact::plane_sphere_contact` | The example's plane-sphere rule | Agree: `dist = cdist - radius`, `pos = centre - normal (dist/2 + radius)` |
| `passive::{joint_spring_force, joint_damper_force, affine_actuator_force}` | The example's linear spring-damper and affine gain/bias | Agree |

So the challenge list below is not a verdict on the dynamics core. The ABA/RNE
pair and the spatial algebra under it are correct. The challenges concentrate in
four places: the floating-base velocity convention, the joint-parameter
contract, everything downstream of a contact normal, and the gap between the
pure ported functions and the stepper that is actually run.

### 1.2 Severity vocabulary

| Severity | Meaning |
| --- | --- |
| **defect** | A correct, accepted input produces a result that contradicts the implementation's own stated model, a conservation law, or an identity the code relies on elsewhere. Reachable without misuse |
| **divergence** | The implementation is self-consistent but departs from a requirement in steps 00–12, or from the example formulation the step cites. The departure is not recorded as a deviation where the step requires deviations to be recorded |
| **gap** | A requirement has no implementation, and the absence changes a numerical result rather than merely deferring a feature |
| **cost** | The result is correct but the algorithm is asymptotically or by a large constant factor worse than the formulation already available in the codebase |

### 1.3 Evidence discipline

Entries marked **measured** were obtained by running the implementation during
this audit on the fixture named in the entry. They are deterministic,
single-configuration measurements of our own code. Per `SPECS.md` Section 3 they
are **not** promoted to `reproduced_in_sim`, which additionally requires
multiple seeds and a recorded comparison budget. Entries marked
**code_inspected** rest on reading our own source; per the same table that
licenses statements about what the code does, not about its correctness, so each
such entry states the mathematics it is being compared against. Entries marked
**derived** are hand derivations written out in the entry so they can be checked.

No entry rests on running the example system. None was run; none exists in this
repository.

### 1.4 Register

| ID | Challenge | Severity | Owning step |
| --- | --- | --- | --- |
| [M-01](#m-01) | `Free` joint stores linear velocity in the body frame, not the world frame | defect | [01](01-state-and-coordinates.md) |
| [M-02](#m-02) | A joint axis that is not unit length makes the velocity map disagree with the position map | defect | [01](01-state-and-coordinates.md) |
| [M-03](#m-03) | `InertiaTensor::is_physically_plausible` accepts an indefinite tensor | defect | [02](02-pure-dynamics-core.md) |
| [M-04](#m-04) | Contact compliance scales with the number of sampled contact points | defect | [07](07-collision-pipeline.md) |
| [M-05](#m-05) | The EPA seed tetrahedron is not required to enclose the origin | defect | [07](07-collision-pipeline.md) |
| [M-06](#m-06) | A degenerate tetrahedron is reported as containing the origin | defect | [07](07-collision-pipeline.md) |
| [M-07](#m-07) | Box-box edge-edge contact is placed at a corner, not on the edges | defect | [07](07-collision-pipeline.md) |
| [M-08](#m-08) | The hard-limit position clamp discards kinetic energy with no accounting | divergence | [06](06-equality-constraints-and-solver.md) |
| [M-09](#m-09) | Contact impedance is referred to a fixed 1 kg mass, not the row's own effective mass | divergence | [06](06-equality-constraints-and-solver.md) |
| [M-10](#m-10) | Friction is applied outside the coupled contact solve, and pair contacts have none | divergence | [06](06-equality-constraints-and-solver.md) |
| [M-11](#m-11) | The contact solve's unconstrained acceleration omits two terms, not one | divergence | [06](06-equality-constraints-and-solver.md) |
| [M-12](#m-12) | The hard-limit solve and the contact solve do not see each other | divergence | [06](06-equality-constraints-and-solver.md) |
| [M-13](#m-13) | `stiffness_damping` selects the `solref` format with two different predicates | divergence | [06](06-equality-constraints-and-solver.md) |
| [M-14](#m-14) | `solve_small_dense` uses an absolute pivot tolerance | divergence | [02](02-pure-dynamics-core.md) |
| [M-15](#m-15) | No broad phase and no mid phase; `bounding_sphere` is unreachable | gap | [07](07-collision-pipeline.md) |
| [M-16](#m-16) | Body-pair exclusion is a fixed overlap allowance, not an exclusion mechanism | gap | [07](07-collision-pipeline.md) |
| [M-17](#m-17) | The ported integrators, passive forces and solver are not wired into the stepper | gap | [03](03-integrators.md), [05](05-passive-forces-and-actuators.md) |
| [M-18](#m-18) | The implicit integrator cannot be reached without `M(q)`, which CRBA would supply | gap | [02](02-pure-dynamics-core.md), [03](03-integrators.md) |
| [M-19](#m-19) | `rk4_step` cannot integrate a `Free` joint, the case that motivates it | gap | [03](03-integrators.md) |
| [M-20](#m-20) | The fidelity battery has no contact, collision or constraint row | gap | [04](04-fidelity-loop.md) |
| [M-21](#m-21) | Fidelity knobs are compile-time constants | gap | [09](09-fidelity-tiers.md) |
| [M-22](#m-22) | The GPU dense path is `f32` and capped at 6 degrees of freedom; the G1 needs 35 at `cond(M) ~ 1e5` | gap | [11](11-gpu-batched-backend.md), [12](12-cpu-gpu-agreement.md) |
| [M-23](#m-23) | `solve_equality` refactorises `M` once per constraint row | cost | [06](06-equality-constraints-and-solver.md) |
| [M-24](#m-24) | `inverse_inertia_diagonal` costs `nv` forward-dynamics passes and runs inside the stance loop | cost | [02](02-pure-dynamics-core.md) |

### 1.5 Status after the 2026-11-01 corrections

The register below was written as an audit. This section records what has since
been done against it, so that the entries are read as a work list with known
state rather than as a flat list of open defects. Each **fixed** row names the
test that would fail if the fix regressed.

| ID | State | Where |
| --- | --- | --- |
| [M-01](#m-01) | **fixed** | `composite`: world-frame linear velocity, translation-first, per 02 §2.3; exact exponential-map quaternion update per 01 §8.3. World linear momentum now conserved to below 1e-14 against a measured 3.93e-3 drift. `a_force_free_body_conserves_world_linear_momentum_exactly` |
| [M-02](#m-02) | **fixed** | `JointSpec::unit_axis` is the single accessor, used by `transform_at` and both dynamics adapters. `a_non_unit_joint_axis_gives_the_same_dynamics_as_a_unit_one` |
| [M-03](#m-03) | **fixed** | The condition moved to the principal moments, via the new `robot_model::eig` (12 §7). `an_indefinite_tensor_is_rejected_even_though_its_diagonal_passes` |
| [M-04](#m-04) | open | Folded into the constraint-solver work; see S-09 and the note below |
| [M-05](#m-05) | **attempted, reverted** | Needs the signed-volume subdistance; see 1.5.1 |
| [M-06](#m-06) | **attempted, reverted** | Same; see 1.5.1 |
| [M-07](#m-07) | **fixed** | Support-feature edge pair plus closest-points, replacing the extreme-corner fallback. `an_edge_edge_box_contact_is_placed_on_the_edges_not_at_a_corner` |
| [M-08](#m-08) | open | Needs limit rows in a shared solver |
| [M-09](#m-09) to [M-12](#m-12) | open | The constraint solver, per 07 and 08 |
| [M-13](#m-13) | **fixed** | One predicate for one `solref` format choice |
| [M-14](#m-14) | **fixed** | Pivot tolerance relative to the matrix, matching `solve_dense` |
| [M-15](#m-15), [M-16](#m-16) | open | Broad and mid phase; pair exclusion |
| [M-17](#m-17) | **partly addressed** | The composite module is now a complete forward-dynamics path (`CompositeState::step`) rather than another unreachable function, but `MultibodyState::step` is still the stepper the CLI and the environments run, and the three ported modules are still not on it. Switching over waits on the solver, because contact has to move at the same time |
| [M-18](#m-18) | **fixed** | `crb`, `factor_m`, `solve_ld`, `full_m` supply `M(q)` and its factorization. Cross-validated against the local-frame recursion on the 29-degree-of-freedom G1 to 1e-10 |
| [M-19](#m-19) | open | RK4 still cannot take a `Free` joint. The drift it is wanted for is now asserted to be first-order in the timestep, which is the measurement that would show an implicit or RK4 step working |
| [M-20](#m-20) | open | The battery still has no contact, collision or constraint row. The new cross-algorithm checks live in the module's own tests, not in the archival battery |
| [M-21](#m-21) | open | Still compile-time constants |
| [M-22](#m-22) | open, better understood | The composite module's degree-of-freedom-tree factorization is the structure a batched backend parallelises over, so the sparse route named in M-22 now exists on the CPU. The `f32` and `MAX_DOF = 6` limits in `robot-sim-gpu` are untouched |
| [M-23](#m-23) | open | `solve_equality` still refactorises per row. The composite path does not have the defect, so this is confined to the standalone equality solver |
| [M-24](#m-24) | **fixed** | `composite::inverse_inertia_diagonal` is one factorization plus `nv` back-substitutions instead of `nv` articulated-body passes. `the_inverse_inertia_diagonal_matches_the_articulated_construction` |

#### 1.5.1 What M-05 and M-06 actually need

This is the one finding that changed shape on contact with the code, so it is
recorded rather than left as "open".

The enclosure invariant EPA depends on does not come from the example's EPA. It
comes from its **GJK**: penetration is signalled only when the signed-volume
subdistance of 05 §5.4 returns four non-zero barycentric weights, and four
non-zero weights *are* strict enclosure by construction. The polytope
constructions of 05 §6.1 then rely on that invariant rather than establishing
it. Our `reduce` uses Ericson's region tests, which give a correct closest
feature but no such guarantee.

Building §6.1's constructions on top of Ericson's `reduce` was tried and
reverted. That included the five-vertex hexahedron §6.1 uses for a triangle
simplex, which is itself the explanation for the original defect: **no**
tetrahedron built on three vertices whose plane passes through the origin can
strictly contain the origin without being degenerate, so the previous
"grow to four affinely independent points" step could not have established the
invariant whatever its tolerances. Two further details of §6.1 turned out to be
load-bearing and are worth carrying into the real fix: the `P3_MISSING_ORIGIN`
test must be skipped when the GJK distance is essentially zero (the example
guards it with `dist[0] > 10*mjMINVAL`), because the origin is then on the
shared face of both half-tetrahedra and neither can pass; and an empty EPA
horizon means the support point adds nothing, which is *convergence* for a
polyhedron, not the `nedges < 3` failure.

Each attempt traded one failure for another. Requiring strict interiority in
`reduce` turned M-06's false positive into a false negative for shallow
penetrations; relaxing it reintroduced false overlaps for separated pairs. That
oscillation is the signature of the structural mismatch, not of a missing
tolerance.

So the work is: replace `reduce` with the signed-volume subdistance (05 §5.4),
then build §6.1's polytope2/3/4 on the invariant it provides. Bounded, fully
specified, and a different piece of work from the one attempted. Until then the
entries stand as written, and the `depth.max(0.0)` that hides the failure is
still in place.

### 1.6 Status after the 2026-11-01 corrections, second pass

The table in 1.5 is superseded by this one. Fourteen of the twenty-four entries
are now closed, each naming the test that would fail if it regressed. The two
that remain are named with what they need.

| ID | State | Where |
| --- | --- | --- |
| [M-01](#m-01) | **fixed** | `composite`: world-frame linear velocity, exact quaternion update. Momentum conserved to below 1e-14 against a measured 3.93e-3 |
| [M-02](#m-02) | **fixed** | `JointSpec::unit_axis` is the single accessor |
| [M-03](#m-03) | **fixed** | Principal moments via `robot_model::eig` |
| [M-04](#m-04) | **fixed** | Both halves: the solver couples the rows so the total force balances the weight however many divide it, and the exact plane manifolds of `04` sections 1.3 and 1.4 bound the point count at four as a property of the geometry. The commit records what the solver alone does *not* fix, since the rest depth still scales as `1/n` in the example's formulation too |
| [M-05](#m-05) | **fixed** | Signed-volume subdistance (05 §5.4) plus the polytope2, polytope3 and polytope4 constructions of §6.1, which is where the enclosure invariant actually comes from |
| [M-06](#m-06) | **fixed** | Same. Four non-zero barycentric weights *are* strict enclosure, so there is no containment test left to get wrong |
| [M-07](#m-07) | **fixed** | Support-feature edge pair plus closest points |
| [M-08](#m-08) | **fixed** | Limits are rows with the same `solref`/`solimp` impedance as any other |
| [M-09](#m-09) | **fixed** | `R_ii = (1 - d)/d * A_ii` with the exact diagonal |
| [M-10](#m-10) | **fixed** | Friction in the same problem as the normal force, as the example's **pyramid edges**: `2(condim-1)` rows with Jacobians `n ± μ·t_k` and non-negative coefficients (07 §9.2). Contact frame from `mju_makeFrame`, so no hard-coded tangent plane. See §1.7 — the first fix used the wrong cone |
| [M-11](#m-11) | **fixed** | `a_u = J qacc_smooth + Jdot v`. The point Jacobian is checked against a central finite difference |
| [M-12](#m-12) | **fixed** | Limit and contact rows share one `A` |
| [M-13](#m-13) | **fixed** | One predicate for one `solref` format |
| [M-14](#m-14) | **fixed** | Relative pivot tolerance |
| [M-15](#m-15) | **fixed** | Bounding-sphere and world-AABB filters before the narrow phase, with `world_aabb` checked against the support function |
| [M-16](#m-16) | **fixed** | `pair_can_collide`: bitmask, weld body, immovable pair and parent-child, replacing the global 3 mm allowance |
| [M-17](#m-17) | **open** | See 1.6.1 |
| [M-18](#m-18) | **fixed** | `crb`, `factor_m`, `solve_ld` |
| [M-19](#m-19) | **fixed** | `CompositeState::rk4_step` with the Runge-Kutta-Munthe-Kaas correction, asserted higher order rather than merely better |
| [M-20](#m-20) | **fixed** | Battery schema v8 adds a plane force-balance, a friction-cone and a narrow-phase row (ADR 0008 amendment) |
| [M-21](#m-21) | **fixed** | `options::SimOptions` holds all seven knobs, rejects invalid values rather than clamping, and records every knob for a run record. `CcdConfig` is threaded through the narrow phase |
| [M-22](#m-22) | **open** | See 1.6.2 |
| [M-23](#m-23) | **fixed** | `cholesky_factor` / `cholesky_substitute`, one factorisation per solve |
| [M-24](#m-24) | **fixed** | One factorisation plus `nv` back-substitutions |

#### 1.6.1 What M-17 needs

The pieces now exist: `composite` is a complete forward-dynamics path with the
correct `Free`-joint convention, `solver` is the unified constraint problem, and
`options` is the per-run configuration. What is missing is the switch-over.
`MultibodyState::step` in `articulated` is still what the CLI, the environments
and most tests run, and it still uses its inline semi-implicit Euler, its
clamped-PD actuation, its penalty contact and its two-part joint limit.

The switch is not a small edit, and the reason is the `Free`-joint layout.
`composite` orders a free joint's degrees of freedom translation-first in the
world frame; `articulated` orders them angular-first in the body frame. Every
`q` and `qd` vector that crosses that boundary has to move at once, which means
`MultibodyState`, `MultibodyEnv`, the `robot sim multibody` command, the USD
replay writer, the G1 probe rows in `fidelity.rs`, and the free-body tests in
`articulated.rs`. Doing it piecemeal would leave two conventions live in one
crate, which is worse than either.

It is also the one remaining change that alters measured behaviour on the G1,
so it wants its own before-and-after on the zero-torque probe and the stance
hold rather than being folded into another commit.

#### 1.6.2 What M-22 needs

Unchanged from the entry, with one addition: the sparse route it names now
exists on the CPU. `composite`'s `factor_m` is the `L' D L` factorization over
the degree-of-freedom tree, which is the structure a batched backend
parallelises over, so the GPU work is a port rather than a design. The `f32`
and `MAX_DOF = 6` limits in `robot-sim-gpu` are untouched, and the measured
`cond(M) ~ 1e5` on the G1 still says a dense `f32` factorisation of that matrix
is unsuitable whatever the budget.

The solver adds a second requirement the entry did not name. Projected
Gauss-Seidel is what `solver` implements, and a strictly sequential per-row
update is exactly what the example's own GPU backend drops it for. A batched
backend therefore needs the Newton path of `08` as well as the kernels, which
makes step 06 item 5's "at least two solver algorithms" a prerequisite for
M-22 rather than an independent nicety.

---

### 1.7 Third pass, 2026-11-01: the switch-over, and a cone that was the wrong shape

Two things happened in this pass. The corrected pieces were wired into
something that runs, and one of the fixes claimed in §1.6 turned out to be
wrong in a way that only showed up once I checked it against the example
instead of against my own reasoning.

#### 1.7.1 `World` and `WorldEnv` — what M-17 asked for

[M-17](#m-17) recorded that `composite`, `solver`, `collision` and `options`
were each validated and that nothing ran them. `crates/robot-sim/src/world.rs`
is the path that does: it owns the composite model, the collision shapes behind
the pair filter and broad phase, the ground plane and the per-run
configuration, and steps them as one — position and velocity stages, contact
detection, contact rows at the configured `condim`, joint-limit rows per 07 §8,
one solve, then integration by the configured integrator. Under Runge-Kutta the
constraint problem is solved at **every stage**, as `mj_forwardSkip` is per
stage in 01 §9.4; solving once and reusing the force would quietly reduce the
method to first order. `WorldEnv` puts it behind the `Environment` and
`Backend` interface that `specs/10` requires external crates to use, reporting
`cpu-world`.

Two measurements, both of which corrected a test premise of mine first:

- **The ground is enforced.** Over 3000 steps of the zero-torque G1 the
  contacts carry up to the robot's full weight and no collision shape goes
  more than 0.045 m below the plane. The pelvis *does* pass through, and that is
  not a contact failure — the vendored G1's pelvis and torso carry no collision
  geometry at all. A test asserting "the robot stands" was also wrong on
  physics: a zero-torque humanoid cannot stand, because nothing holds its
  joints. The assertion is now on the shapes that exist.
- **Limits hold exactly when given room to act.** A G1 joint driven at 50 N·m
  into its bound overshoots by 0.177 rad with no margin and by **exactly zero**
  with a 0.2 rad margin. With no margin a row appears only once `dist < 0`, so
  the constraint brakes the joint after it has already crossed. Raising the
  impedance barely moves that transient while it does pull the steady state
  onto the bound, which is what pointed at the margin. `SimOptions` gained
  `limit_margin`, separate from the collision `margin` as the example keeps
  `jnt_margin` separate from `geom_margin`.

What remains of M-17: `robot sim multibody` and the USD replay writer still run
`MultibodyState`. Moving them needs the replay writer's `Free`-joint layout and
the stance controller moved at the same time, and it changes the G1 numbers the
command prints, so it wants its own change with a before-and-after.

#### 1.7.2 The friction cone was the wrong shape — a correction to §1.6

§1.6 recorded [M-10](#m-10) as fixed by "friction rows bounded by the pyramidal
cone against the normal row's current force". That is a cone, and it couples
friction to the normal solve, but it is **not the example's cone**.

I had implemented one normal row plus two tangential rows, bounding each
tangential force by `μ·f_n` inside the solver. Bounding each component
independently gives the pyramid whose cross-section is a **square** of
half-width `μ·f_n`. That square *circumscribes* the true circular cone, so it
allows up to `√2·μ·f_n` of friction along a diagonal — about 41 % too much.

07 §9.2 is explicit about what the example does, and it is structurally
different: the rows **are the generator directions**. For `k = 1..dim-1`,
`row± = jac[0] ± friction[k-1]·jac[k]`, i.e. `J = n ± μ·t_k`, giving
`2(condim-1)` rows each with a non-negative coefficient and all carrying
`pos = dist`. With `λ ≥ 0` the realised force is `Σ λ_k (n ± μ t_k)`, whose
tangential part satisfies `|f_t1| + |f_t2| ≤ μ f_n` — the **inscribed diamond**.
Conservative rather than permissive, which is the right side to err on, and it
is what `mju_decodePyramid` inverts.

`solver::contact_rows` now emits the edges, `solver::decode_pyramid` reads the
contact-frame force back out, and `RowKind::ContactFriction` is gone: every row
kind is now unbounded or non-negative, because the cone lives entirely in the
row Jacobians. The solver needs no cone projection at all. This is the one place
in the audit where consulting material outside the example changed what I
looked at — a search on MuJoCo's pyramidal representation said the cone is
parameterised by non-negative coefficients over generator directions rather
than by a box, which sent me to 07 §9.2 to read the construction properly.

#### 1.7.3 What the pyramid then exposed: projected Gauss-Seidel is too slow

The edge representation is redundant: four non-negative coefficients encode
three force components, and every edge contains the normal. Four pyramidal
contacts on a six-degree-of-freedom body give **sixteen rows for a rank-six
problem**, and projected Gauss-Seidel needs **1011 sweeps** to bring the
per-sweep force change below 1e-10 there — against the default budget of 100,
at which the residual is still 1.1e-2. The example's `Rpy = 2μ²R₀` shared
across a contact's edges (07 §2.4) is now implemented and does not close that
gap on its own.

The default budget is not wrong for *stepping*: at 100 sweeps the acceleration
is already within 2e-4 relative of the converged value, and acceleration is what
the integrator consumes. It is the forces that are far from settled. A warm-start
test had been comparing two unconverged iterates and reading the difference as
disagreement; it now sets its own budget and tests what it claims.

This is the quantitative case for the Newton solver of
[specs/08](../../../docs/example-system-specs/08_solvers.md), which
[M-22](#m-22) already records as outstanding for the GPU backend on the separate
ground that Gauss-Seidel does not vectorise. It is now also outstanding on
accuracy grounds on the CPU, which is a stronger reason and a measured one.

#### 1.7.4 The Newton solver, and what making it the default exposed

[specs/08](../../../docs/example-system-specs/08_solvers_islands.md) section 6
is now implemented in `crates/robot-sim/src/newton.rs` and is the default
([`SolverAlgorithm::Newton`]). Measured on the same sixteen-row problem that
took projected Gauss-Seidel 1011 sweeps: **one** Newton iteration, final scaled
gradient norm 4.9e-14. The two agree on the acceleration to 1e-8, which is the
cross-algorithm check that makes the new path trustworthy — the same discipline
the composite mass matrix was validated under against the articulated-body
inverse dynamics. Gauss-Seidel is kept for exactly that purpose.

The primal form also removes the projection entirely. A row's force is
`-D jar`, and an inequality row contributes to the cost only while `jar < 0`,
so non-negativity is a property of the hinge rather than of a projection step.
Nothing in the solver knows what a friction cone is; the cone is in the row
Jacobians.

Two corrections came out of switching the default over, and both are the kind
that only a running system produces.

**`mju_cholFactor` clamps, it does not fail.** 08 section 7.1 is explicit: a
pivot below `mindiag` is set to `mindiag`, the rank is decremented and that
column below the diagonal is zeroed. The first implementation returned an error
instead, and the G1 stance run died at degree of freedom 31. `D = 1/R` reaches
1e15 when a row's regularisation sits on its floor, so `M + J' D_q J` can lose
a pivot to cancellation even though it is positive definite exactly.

**A contact with zero friction must be `dim == 1`.** This is the substantive
one. `condim = 3` with `mu = 0` was building four pyramid edges, and at `mu = 0`
the directions `n ± mu t_k` all collapse onto `n` — four *identical* rows — while
`Rpy = 2 mu^2 R0` is zero, so `D` saturated at `1/MINVAL` and the Hessian lost
every pivot. The Newton solver diverged on step 1511 of the G1 collapse, once
the folded robot began contacting its own links through `World`'s default pair
friction of zero. 07 section 9.2 already says what to do: a frictionless contact
gets one row along the normal. With that fix the full 3000-step collapse runs
with Newton converging in two to four iterations at a residual around 1e-16.

Worth recording separately: **Gauss-Seidel's non-convergence had been acting as
damping.** It never reached the true force in 100 sweeps, so it systematically
under-estimated it, and that under-estimate masked the degenerate rows. A solver
that actually converges is what surfaced them. This is a general hazard in
reading a simulation as validation of its own solver, and it is why the
zero-friction case now has a regression test of its own rather than only a
passing stance run.

## 2. Entries

### M-01

**`Free` joint stores linear velocity in the body frame, not the world frame.**
Severity: **defect**. Owning step: [01](01-state-and-coordinates.md).

Requirement. Step 01 item 3: "A `Free` joint's velocity representation MUST
follow the reference's own documented asymmetry — global-frame linear velocity,
body-local-frame angular velocity — rather than inventing a different
convention, since that asymmetry is described in the reference's own
documentation as 'correct use of the topology of quaternions,' not an arbitrary
choice." The example states the same asymmetry (`01` data model, `02` §1.2).

What the code does. `articulated::motion_subspace_basis` gives a `Free` joint the
six-column identity basis **in the joint's own local frame**, so both halves of
`qd[dof_offset .. dof_offset + 6]` are body-local: angular first, then linear.
`integrate_free_position` is consistent with that and says so in its own doc
comment — it rotates the stored linear velocity into the world before adding it
to the translation (`translation += dt * R * linear_local`). The convention is
therefore fully local, not the required mixed convention. Evidence:
**code_inspected**.

Why this is a defect and not a recorded deviation. Step 01's requirement is a
MUST and the step's "Done when" names it ("The `Free` velocity convention
matches the reference's documented asymmetry"). The step's Status field reports
`Free` as "implemented ... and validated" without recording a convention
divergence, and no step file lists this under a deviations heading. Steps 02 and
06 both demonstrate the project's practice of recording such a divergence
explicitly; this one is not recorded anywhere.

The numerical consequence, derived. For a free body with its centre of mass at
the frame origin, the body-frame equation of motion is
`I a + v x* (I v) = 0` with no external wrench. Writing `v = (omega, u)`, the
velocity-product term is `(omega x I omega, m omega x u)`, so the body-frame
linear acceleration is exactly `-omega x u`. That is not physics acting on the
body; it is the bookkeeping that cancels the rotation of the frame the velocity
is expressed in, so that the *world* velocity stays constant. An explicit step
performs that cancellation only to first order in `dt`, so the world linear
velocity of a force-free spinning body drifts at `O(dt)` per step. Under the
required world-frame convention the same quantity is a constant of the motion
and the integrator has nothing to cancel: the drift is identically zero,
whatever the spin. Evidence: **derived**.

The numerical consequence, measured. Force-free, torque-free free body
(`free_body` fixture, mass 2 kg), initial world velocity `(1, 0, 0)` m/s,
`dt = 1e-4` s, 10 000 steps (1 simulated second). Linear momentum and kinetic
energy must both be exactly conserved.

| Inertia | Initial `omega` (local) | `\|dv\|/\|v\|` after 1 s | `dKE/KE` after 1 s |
| --- | --- | --- | --- |
| isotropic `(0.1, 0.1, 0.1)` | `(3, 5, 7)` | 3.93e-3 | 1.44e-3 |
| anisotropic `(0.02, 0.10, 0.30)` | `(3, 5, 7)` | 4.61e-3 | 1.94e-2 |
| anisotropic `(0.02, 0.10, 0.30)` | `(0, 0, 0)` | 0 | 0 |
| anisotropic `(0.02, 0.10, 0.30)` | `(0, 0, 7)` (principal axis) | 2.45e-3 | 5.88e-4 |

The three controls isolate the mechanism rather than merely showing drift. With
an isotropic inertia, Euler's equations give `omega_dot = 0` exactly, so the
rotational integration is exact and the entire 0.39 % linear-momentum error is
attributable to the frame convention. Spinning about a single principal axis
gives `omega_dot = 0` for the same reason and the same conclusion. With
`omega = 0` the error is identically zero, which is exactly what the `-omega x u`
mechanism predicts and what rules out an unrelated cause. The anisotropic
spinning row adds the explicit-Euler error on Euler's equations on top, which is
why its energy drift is an order of magnitude larger than its momentum drift.
Evidence: **measured** (this audit; deterministic).

Scope note. This is a separate matter from the integrator order. A
higher-order or exponential-map orientation update (already named as an open
option in `integrate_free_position`'s doc comment) would reduce the drift but
would not remove it, because the `-omega x u` cancellation is still being
integrated numerically. Only the convention change makes it exact.

Decision needed. Either change the representation to the required mixed
convention, or amend step 01 item 3 and record the local convention as a
deviation with this measurement attached. Both are real options and the choice
is not this appendix's to make; routed to Appendix B as **S-07**.

### M-02

**A joint axis that is not unit length makes the velocity map disagree with the
position map.** Severity: **defect**. Owning step:
[01](01-state-and-coordinates.md).

The contradiction. For any joint, the motion subspace `S` must be the derivative
of the joint's pose map with respect to its coordinate, so that `v = S qdot`
holds. Two functions disagree about that map:

- `articulated::joint_transform` **normalises** the declared axis. For a
  revolute joint it calls `axis_angle_rotation`, which divides by the axis norm,
  so the pose is a rotation of `q` radians about the *unit* axis. The prismatic
  branch normalises explicitly. `robot_model::joint::JointSpec::transform_at`
  does the same at the model layer.
- `articulated::motion_subspace_basis` uses the **raw declared axis**:
  `join6(axis, [0; 3])` for revolute, `join6([0; 3], axis)` for prismatic.

So the position map says the coordinate is an angle in radians about the unit
axis, while the velocity map says the body's angular velocity is
`\|axis\| * qdot`. Unless `\|axis\| = 1` the two contradict each other, and every
quantity built on `S` — the velocity recursion, `D = S^T I^A S`, the bias force,
the generalised force `tau = S^T f` — is scaled while the kinematics are not.
Evidence: **code_inspected**.

Reachability. Nothing normalises or rejects a non-unit axis.
`JointSpec::validate` requires only that the axis have a usable direction
(`axis.normalized(1e-12).is_none()` is the rejection test), and
`robot_model::urdf` parses `<axis xyz="...">` verbatim. A non-unit axis is
common in hand-written and exported URDF; the URDF specification states the axis
*should* be normalised without requiring it, and the example system's own model
compiler normalises joint axes at compile time (`13`, compile-order and frame
conventions). This repository has no such step.

Measured. `point_mass_pendulum(1.0, 1.0)` — a 1 kg point mass 1 m from the
pivot — with the declared axis scaled while its direction is unchanged, so the
modelled machine is identical in both rows:

| Declared axis | `validate()` | `qdd` at `q = 0`, `tau = 0`, `g = 9.80665` | Kinetic energy at `qdot = 1` | `tau` for `qdd = 1` |
| --- | --- | --- | --- | --- |
| `(0, 1, 0)` | passes | 9.80665 | 0.5 | 1.0 |
| `(0, 2, 0)` | passes | 4.903325 | 2.0 | 4.0 |

The analytic answers for this pendulum are `qdd = g/L = 9.80665`,
`KE = 0.5 m L^2 qdot^2 = 0.5`, and `tau = m L^2 qdd = 1.0`. The second row is
wrong by exactly the powers of `\|axis\|` the derivation predicts: the mass
matrix picks up `\|axis\|^2`, the gravity torque `\|axis\|`, so `qdd` picks up
`1/\|axis\|`. Both rows pass profile validation and neither reports an error.
Evidence: **measured** (this audit; deterministic).

Why the error does not cancel. If the whole system consistently treated the
coordinate as `theta/\|axis\|`, the simulation would merely be in rescaled units
and the world motion would be identical. It does not: the position update in
`MultibodyState::step_inner` adds `qdot * dt` to `q`, and `joint_transform`
reads `q` as radians about the unit axis. So the simulated *angle* advances at
`qdot` while the body's angular velocity is `\|axis\| qdot`. The trajectory is
wrong, not rescaled. Joint limits, which are declared in the joint's own unit,
are clamped against the same mismatched coordinate.

Candidate fix. Normalise the axis once, at the boundary — either in
`JointSpec::validate` (reject a non-unit axis, matching the importer's
"reject unsupported constructs explicitly" ethos of `SPECS.md` §7.1) or in
`ArticulatedBody::from_profile` and `JointSpec::transform_at` together (accept
and normalise, matching the example's compiler). A regression test that scales a
declared axis and asserts the dynamics are unchanged is the reproducing test the
testing policy in `AGENT.md` asks for.

### M-03

**`InertiaTensor::is_physically_plausible` accepts an indefinite tensor.**
Severity: **defect**. Owning step: [02](02-pure-dynamics-core.md).

What the check tests. `robot_model::link::InertiaTensor::is_physically_plausible`
tests finiteness, `ixx, iyy, izz >= 0`, and the three triangle inequalities on
the **diagonal entries** (`ixx + iyy >= izz` and its permutations). The
off-diagonal products of inertia `ixy`, `ixz`, `iyz` are not used.

What the mathematics requires. A rotational inertia tensor is physically
realisable only if it is symmetric positive semi-definite and its **principal
moments** — its eigenvalues — satisfy the triangle inequality. The
diagonal-entry test is equivalent to that condition only when the tensor is
already diagonal. With non-zero products of inertia a tensor can pass every
diagonal test and still have a negative eigenvalue.

Measured counterexample. `ixx = iyy = izz = 1`, `ixy = 2`, `ixz = iyz = 0`. The
matrix is `[[1, 2, 0], [2, 1, 0], [0, 0, 1]]`, whose eigenvalues are `3`, `-1`
and `1`. `is_physically_plausible()` returns **true**. Evidence: **measured**
(this audit; deterministic).

Consequence. A body carrying that tensor has negative kinetic energy for
rotation about the eigenvector of `-1`, so `ArticulatedBody::kinetic_energy` can
return a negative total, and the energy-conservation oracles of
[04](04-fidelity-loop.md) lose their meaning for that model. Worse, the joint
inertia `D = S^T I^A S` can become non-positive, which is a precondition the ABA
relies on: `aba_with_externals` solves with `solve_dense` and
`implicit_damping_euler_step` and `solve_equality` use `cholesky_solve`, which
requires positive definiteness and refuses otherwise. An accepted-but-indefinite
inertia therefore surfaces either as a `SingularJointInertia` far from its
cause, or as a silently non-physical trajectory.

Relevance to the specs. Step 05 item 2 names this exact function as the pattern
a future spring-damper sign-preservation check should follow, and Appendix C
cites it as the existing validation idiom. The pattern is weaker than those
references assume.

Candidate fix. Test the eigenvalues. A symmetric 3x3 eigenvalue solve is a
closed-form or a few-iteration Jacobi sweep in plain `f64`, inside the purity
contract. The stronger and equally cheap condition is the physical-realisability
test on the 4x4 pseudo-inertia built from mass, centre of mass and the tensor
together, which also rejects a mass/COM/inertia triple that is individually
plausible but jointly impossible; that is a larger change and is noted here
rather than recommended. Record which condition is adopted, since the two reject
different inputs.

### M-04

**Contact compliance scales with the number of sampled contact points.**
Severity: **defect**. Owning step: [07](07-collision-pipeline.md).

Mechanism. `collision::plane_contact_points` turns a primitive into a set of
`(centre, radius)` samples: one for a sphere, two for a capsule, **all eight
corners** for a box, and `2 * rim_samples` for a cylinder, with `rim_samples`
fixed at `RIM_SAMPLES = 8` by `articulated.rs`. `ArticulatedBody::contact_rows`
emits one row per penetrating sample and, under the default
`ContactModel::Penalty`, `contact_externals` applies the full spring-damper law
`max(0, k * depth + c * closing)` to **each row independently**. The rows are
not normalised by their count or by the area they represent.

Consequence, derived. A body resting flat on `n` co-planar sample points at
equal depth carries `n` independent springs in parallel, so its equilibrium
penetration is `m g / (n k)` rather than `m g / k`. The stated stiffness `k` is
therefore not the contact stiffness of the body; the effective stiffness is
`n k`, and `n` is a property of the shape's discretisation, not of the contact.
A box resting on a face settles four times shallower than a sphere of the same
mass against the same ground, and changing `RIM_SAMPLES` from 8 to 16 doubles a
cylinder's contact stiffness and halves its rest penetration, with no parameter
change. The same `n` multiplies the damping, so the contact's damping ratio
`zeta = c / (2 sqrt(k m))` is preserved while its natural frequency rises as
`sqrt(n)` — which also tightens the explicit stability limit `omega dt < 1` by
`sqrt(n)` for no modelled reason. Evidence: **derived**, and consistent with the
repository's own `a_flat_box_settles_on_its_four_bottom_corners_at_the_analytic_height`
test, which encodes the four-corner rest height as the expected value.

Not fixed by the coupled model. Under `ContactModel::Impedance` the rows are
solved together, but the regularisation `R = 1 / CONTACT_REFERENCE_MASS` is
per-row, so the `n`-dependence survives: with `A_ij ~ 1/m` for co-planar
translational rows, equilibrium gives `f_i = k d - m_ref g` and
`sum_i f_i = m g`, hence `d = (m g / n + m_ref g) / k`. The leading term still
carries `1/n`.

What the example does instead. The example assigns each contact a reference
acceleration and an impedance and solves the set as one convex problem (`07`
§2.1–§2.4), so adding points redistributes one total force rather than adding
force; and its primitive routines emit a *bounded, geometrically chosen*
manifold — up to four points for plane-box using the lower corners only (`04`
§1.4), up to four for plane-cylinder by an exact rim rule (`04` §1.3) — instead
of sampling. `docs/example-system-mapping.md` §2.6 already lists both exact
rules as candidate work; this entry supplies the reason it is a correctness
matter and not only an accuracy improvement.

### M-05

**The EPA seed tetrahedron is not required to enclose the origin.** Severity:
**defect**. Owning step: [07](07-collision-pipeline.md).

EPA's invariant. Expanding-polytope depth is only defined when the origin lies
inside the polytope being expanded: each face's signed distance
`d = dot(n, v_i)` is then non-negative, and the closest face is a lower bound on
the penetration depth that the expansion tightens.

What the code does. `collision::gjk_epa` enters EPA whenever GJK set
`overlap`, which includes the branch `norm2(closest) < 1e-24` — reached with a
simplex of one, two or three vertices. `complete_simplex` then grows the simplex
by probing the six coordinate axes and some cross products, accepting a new
support point when it is *affinely independent* of the ones already held
(`norm2(cross(...)) > 1e-18`, or `orient(...).abs() > 1e-18`). Independence is
not enclosure: nothing in `complete_simplex` tests whether the resulting
tetrahedron contains the origin. `epa` orients its faces against the seed
tetrahedron's centroid, which is correct and remains valid as the hull grows,
but if the origin is outside that tetrahedron the face distances `d` can be
negative, the `min_by` over `d` then selects the most negative face, and
`proj = scale(n, dist)` lies on the wrong side of the origin. Evidence:
**code_inspected**.

Consequence. The returned `depth` is clamped by `result_depth.max(0.0)`, so the
failure does not show up as a negative depth; it shows up as a zero depth and an
arbitrary normal for a pair that is genuinely touching or shallowly
interpenetrating. A contact row built from that normal pushes in an unrelated
direction. Because the clamp hides it, there is no diagnostic.

Second, smaller issue in the same function. `epa` runs a fixed 128 iterations
and assigns `result_normal`/`result_depth` from the current closest face on
every pass, so exhausting the budget returns an unconverged lower bound with no
flag. The example stops on the gap between the upper and lower bound of the
penetration and reports a measured accuracy (`05` §6.3), which
`docs/example-system-mapping.md` §2.7 already lists as candidate work. The
missing convergence flag is the part worth adding first, because it converts a
silent inaccuracy into a reportable one.

Third. The independence thresholds `1e-18` are absolute, compared against a
squared cross-product or a signed volume, so they scale as the fourth and sixth
power of the model's length unit. At metre scale a tetrahedron with volume
`1e-18 / 6` is numerically flat yet passes.

### M-06

**A degenerate tetrahedron is reported as containing the origin.** Severity:
**defect**. Owning step: [07](07-collision-pipeline.md).

`collision::reduce`, four-vertex branch, decides containment by testing, for
each face, whether the origin and the opposite vertex fall on the same side:
`if s_origin * s_opp < 0.0 { inside = false; ... }`. The comparison is strictly
less than zero, so a zero product leaves `inside` at its initial `true`.

Two inputs produce a zero product. The origin lying exactly on a face plane
(`s_origin == 0`), and a flat tetrahedron, where `s_opp == 0` for every face. In
the second case *all four* products are zero, `inside` stays `true` for a
simplex with no interior at all, and `reduce` returns "the origin is inside"
together with a closest point of `[0, 0, 0]` that it never computed. `gjk_epa`
reads that as overlap and proceeds to EPA. Evidence: **code_inspected**.

Consequence. A false overlap report for a separated pair, when the simplex
collapses — which is exactly what happens for the symmetric, axis-aligned,
face-parallel configurations a humanoid standing on a plane produces constantly.
The downstream effect is a spurious contact row, with the zero depth and
arbitrary normal of [M-05](#m-05).

Relation to M-05. These are the two halves of the same weakness: `reduce`
over-reports containment, and `epa` does not re-check it. Either fix alone
narrows the failure; the robust form is to test containment explicitly, with a
scale-relative tolerance, at the point where EPA is entered.

The example's own answer. `05` §5.4 uses the signed-volume subdistance
algorithm of Montanari and others (2017) precisely for its exact sign tests on
degenerate simplices. `docs/example-system-mapping.md` §2.7 records it as
"replace only if a degenerate-simplex failure is measured". This entry argues
the failure is reachable by construction rather than by chance, which changes
what "measured" would have to look for: a symmetric flat-on-flat configuration,
not a random one.

### M-07

**Box-box edge-edge contact is placed at a corner, not on the edges.** Severity:
**defect**. Owning step: [07](07-collision-pipeline.md).

`collision::box_box_contacts` builds its manifold from corners of either box
that lie inside the other. When no corner qualifies — the edge-edge case, where
two boxes meet along crossing edges — it falls back to a single contact whose
`point_a` is the most negative corner of `a` along the normal and whose
`point_b` is the most positive corner of `b`. Those corners are not the contact
points: in an edge-edge configuration the touching points lie in the interiors
of the two edges. Evidence: **code_inspected**.

Consequence. The depth from `box_sat` is right — the separating-axis minimum is
the correct penetration along the correct axis — but the force is applied at the
wrong place, so the moment arm about each body's origin is wrong, and
`wrench_at` produces a torque that does not correspond to the contact. A box
resting on an edge therefore receives a correct push and an incorrect spin.

Also in `box_box_contacts` and `box_sat`. The lateral-duplicate rejection
(`< 1e-7`), the `inside` test slack (`1e-12`) and the near-parallel edge-axis
cutoff (`norm2(cross) > 1e-18`) are all absolute lengths or squared lengths,
so they carry an implicit assumption that the model is at metre scale. The
near-parallel cutoff is the one that matters numerically: an axis just above the
threshold is a nearly-zero cross product normalised to unit length, whose
overlap value is dominated by rounding, and the separating-axis minimum is then
taken over a noisy candidate. This is the classic near-parallel degeneracy of
box SAT, and the mitigation in the example is face clipping on the incident face
rather than reliance on the minimising axis alone (`04` §3.4).

### M-08

**The hard-limit position clamp discards kinetic energy with no accounting.**
Severity: **divergence**. Owning step:
[06](06-equality-constraints-and-solver.md).

Hard joint limits are enforced in two unrelated places:

1. `ArticulatedBody::apply_hard_limits` runs an acceleration-level active set:
   a bounded joint at a limit and accelerating outward is given a constraint
   torque that zeroes its acceleration, with all simultaneously active joints
   solved as one small linear system and wrong-signed multipliers released. This
   part is sound, and it is a recognisable active-set method.
2. `MultibodyState::step_inner`, after the position update, clamps a violating
   coordinate back to the bound and sets its outward velocity to zero.

Step 2 is a projection, not a constraint force. It removes whatever kinetic
energy the outward velocity carried, at no modelled cost and with no record. The
joint is therefore perfectly inelastic at its stops, with no restitution
parameter and no way to express one, and the removed energy does not appear in
the constraint torque that `step_with_ground_recording` reports — so the
reported constraint torque understates what the limit actually did. Evidence:
**code_inspected**.

Why it is a divergence and not a defect. The behaviour is self-consistent, only
ever dissipative, and the repository's own
`a_hard_limit_does_not_inject_energy_into_a_pendulum` test pins the dissipative
direction. The divergence is from step 06 item 2, which requires a soft-constraint
impedance (`solref`/`solimp` equivalent) parameterised by a time constant and
damping ratio, with the raw stiffness/damping case explicitly motivated by
"perfectly elastic restitution". A clamp admits neither. The example treats a
limit as one solver row per side, activated below a margin, with the same
impedance machinery as every other row (`07` §8).

Note. `constraint.rs` already implements the impedance, the stiffness/damping
map and the `2 * timestep` time-constant clamp that a limit row would need. The
missing piece is a limit row in a shared solver, not the soft-constraint
mathematics. See [M-17](#m-17).

### M-09

**Contact impedance is referred to a fixed 1 kg mass, not the row's own
effective mass.** Severity: **divergence**. Owning step:
[06](06-equality-constraints-and-solver.md).

Under `ContactModel::Impedance`, `contact_externals` builds its target
acceleration and its regularisation from the constant
`CONTACT_REFERENCE_MASS = 1.0`:

```
omega2           = ground.stiffness      / CONTACT_REFERENCE_MASS
two_zeta_omega   = ground.normal_damping / CONTACT_REFERENCE_MASS
regularization R = 1.0                   / CONTACT_REFERENCE_MASS
```

The code documents the intent: a row whose effective mass happens to be `m_ref`
reproduces the penalty law exactly. The divergence is what happens for every
other row. The example's regularisation is
`R = max(MINVAL, (1 - imp) * A_ii / imp)` — proportional to the row's own
`A_ii`, the diagonal of `J M^-1 J^T`, which *is* the row's inverse effective
mass. `constraint::solve_equality` in this same repository implements that form
correctly for equality rows. The contact path uses a constant instead, so a
contact row's impedance, its natural frequency and its damping ratio all drift
away from their stated values in proportion to how far the row's effective mass
is from 1 kg. Evidence: **code_inspected**.

Scale of the discrepancy on the real fixture. The G1's per-DOF inverse inertia
spans `(M^-1)_jj` from 3.30e-2 to 3.16e3 at the home configuration (measured;
see [M-22](#m-22)), so the effective masses the contact rows of a standing G1
present span roughly five orders of magnitude. A single constant cannot stand in
for `A_ii` across that range: the heavy rows are over-regularised into mush and
the light rows are under-regularised into stiffness the timestep cannot carry.

Cheapness of the fix. `contact_externals` already assembles the full matrix `a`,
whose diagonal `a[i][i]` is exactly the `A_ii` the example's formula wants.
Replacing the constant with `(1 - imp) * a[i][i] / imp` is a one-line change at
the point where `regularization` is used, and it brings the contact path into
agreement with the equality path already in the crate.

### M-10

**Friction is applied outside the coupled contact solve, and pair contacts have
none.** Severity: **divergence**. Owning step:
[06](06-equality-constraints-and-solver.md).

Three separate points, all in the friction path:

1. **Friction is not in the solve.** Both contact models compute the normal
   forces first and then call `add_ground_friction` with the solved normal
   magnitude. Friction therefore never influences the normal solution, and the
   tangential force is not constrained jointly with it. Step 06 items 1 and 4
   require contact to be one convex problem over constraint forces with a
   friction cone — pyramidal or elliptic — and require `condim` to be a
   queryable per-contact parameter. There is no cone of either kind and no
   `condim`; the model is a normal solve followed by a decoupled tangential
   force.
2. **Pair contacts are frictionless.** `add_ground_friction` returns
   immediately when `row.b.is_some()`, so link-against-link contact carries a
   normal force only. `docs/example-system-mapping.md` §2.5 records this and
   notes that the parameter-mixing rule for a pair would have to be defined
   first (`03` §6.6–§6.7); the entry is repeated here because it is a
   *dynamics* consequence, not only a missing feature: a humanoid's limbs slide
   across each other without resistance.
3. **The tangent plane is hard-coded.** The tangential velocity is taken as
   `[vt[0], vt[1], 0.0]`, which assumes the contact normal is `+z`. That holds
   for the current ground plane and for nothing else, including the pair rows
   whose normals come from `box_sat`, `capsule_pair` or EPA — for which the
   early return in point 2 is, accidentally, the only thing preventing a
   wrong-plane friction force.

Evidence: **code_inspected**.

Also recorded, as a divergence between two friction laws in one crate.
`GroundContact::force` regularises Coulomb friction by dividing by
`max(\|v_t\|, eps)`, which caps the force at `mu N` and becomes a viscous stick
of slope `mu N / eps` below `eps`. `ContactBody::step` instead caps the impulse
at `\|v_t\| m / dt`, a timestep-dependent cap. Both are documented in place and
both are defensible; they are not the same law, and the single-point-mass
benchmark that step 07 item 1 keeps as a separate reference therefore does not
exercise the law the articulated path uses.

### M-11

**The contact solve's unconstrained acceleration omits two terms, not one.**
Severity: **divergence**. Owning step:
[06](06-equality-constraints-and-solver.md).

`contact_externals` obtains each row's unconstrained normal acceleration `a_u`
by calling `self.kinematics(q, &unconstrained)` — that is, by passing the
acceleration vector `qdd` into the function that propagates *velocities* — and
then reading `row_normal_velocity` off the result. The doc comment records one
approximation: "`a_u` uses `J qdd` without the `Jdot qd` term, so the
centripetal part of the contact-point acceleration is neglected."

A second term is also missing and is not recorded. `point_velocity` computes
`linear + omega x p_body`. Reinterpreted as an acceleration, that yields
`a_lin + alpha x p`. The classical acceleration of a body-fixed point is
`a_lin + alpha x p + omega x (omega x p)`, and in addition the linear part of a
*spatial* acceleration differs from the classical acceleration of the frame
origin by `omega x v_O`. So the quantity used as `a_u` differs from the true
normal acceleration by both the centripetal term and the spatial-to-classical
conversion. Both vanish at `qd = 0` and grow with velocity. Evidence:
**derived** and **code_inspected**.

Consequence and bound on it. The matrix `a` is unaffected: it is assembled with
`qd = zeros`, so `omega = 0` in the response twists and `a` is exactly
`J M^-1 J^T`, symmetric and positive semi-definite as it should be. The error is
confined to the right-hand side, so the contact forces are the exact solution of
a problem whose target acceleration is biased at `O(\|qd\|^2)`. For a standing
robot that is small; for a foot moving at walking speed against a stiff ground
it is not, and it is the regime the contact model is for.

Minimal correction. Either add both terms where `a_u` is formed, or state both
in the doc comment and in step 06's deviations list. The present comment names
one and so understates the approximation.

### M-12

**The hard-limit solve and the contact solve do not see each other.** Severity:
**divergence**. Owning step: [06](06-equality-constraints-and-solver.md).

`apply_hard_limits` builds the columns of its small linear system by calling
`self.articulated_accelerations(q, &zeros, &e, Vec3::new(0.0, 0.0, 0.0), None)`
— with the ground argument `None`. Its system is therefore `M^-1` of the
*contact-free* body. `step_inner` applies the limit correction to a `qdd` that
already contains contact accelerations. When a joint limit and a contact are
active in the same step — a knee at its stop while the foot carries load, the
ordinary case for a standing humanoid — the limit torque is computed from a
response matrix that does not include the contact's constraint on the same
degrees of freedom, so neither force accounts for the other. Evidence:
**code_inspected**.

This is the concrete cost of the architecture step 06 item 1 rules out: contact
and all other constraints solved as one problem. Here they are two solves,
sequentially composed, each ignoring the other's Jacobian. Step 07 item 2 of
`07` §8 in the example puts limit rows in the same matrix as contact rows for
exactly this reason.

### M-13

**`stiffness_damping` selects the `solref` format with two different
predicates.** Severity: **divergence**. Owning step:
[06](06-equality-constraints-and-solver.md).

`constraint::stiffness_damping` chooses the standard
(time-constant/damping-ratio) branch for the stiffness on `solref[0] > 0.0`,
but chooses it for the damping on `solref[1] > 0.0`. The example's formulation
selects both from one condition on the first element. Evidence:
**code_inspected**.

Why it is latent rather than live. `effective_timeconst` refuses a mixed-format
`solref` with `MixedSolrefFormat`, and `solve_equality` calls it before
`stiffness_damping`, so on the only path through the crate the two predicates
always agree. But `stiffness_damping` is `pub`, performs no validation of its
own, and its doc comment says only "Takes the clamped `solref` and `solimp`" —
so a direct caller with `solref = [0.02, -1.0]` gets a standard-formulation
stiffness and a direct-formulation damping, silently. Either tie both branches
to one predicate, or make the precondition a checked one.

### M-14

**`solve_small_dense` uses an absolute pivot tolerance.** Severity:
**divergence**. Owning step: [02](02-pure-dynamics-core.md).

Two dense solvers sit in `articulated.rs` with different singularity rules.
`solve_dense` scales its tolerance to the matrix: `tol = max\|A\| * 1e-14`, with
`1e-300` for an all-zero matrix, and its doc comment states the rule.
`solve_small_dense` — the one `apply_hard_limits` uses — rejects a pivot on the
absolute test `a[pivot][col].abs() < 1e-14`. Evidence: **code_inspected**.

Consequence, both directions. The limit system's entries are elements of
`M^-1`, whose scale varies by five orders of magnitude on the G1 alone
([M-22](#m-22)). A well-conditioned system whose entries are legitimately small
is rejected as singular, so the limit silently fails to act for that step
(`solve_small_dense` returning `None` makes `apply_hard_limits` `break` with no
constraint applied and no error). A badly conditioned system whose entries are
large passes, and its solution is returned with no indication of how little of
it is significant. Using the same relative rule as `solve_dense` removes both.

### M-15

**No broad phase and no mid phase; `bounding_sphere` is unreachable.**
Severity: **gap**. Owning step: [07](07-collision-pipeline.md).

Step 07 item 1 is a MUST: broad-phase candidate-pair generation, a mid-phase
per-pair bounding-volume test, and a narrow-phase dispatch keyed by the
geometry-type pair, as three separate stages. `ArticulatedBody::contact_rows`
has one stage. It loops over every unordered pair of contact shapes and calls
`collision::proximity` — which for any non-capsule pair is full GJK, and EPA on
overlap — with no bounding test first. `Placed::bounding_sphere` exists,
computes the exact bounding radius for all four primitives, and is called from
nowhere in the workspace. Evidence: **code_inspected**.

Cost, on the fixture. The G1 profile is 35 degrees of freedom over 36 position
coordinates. With `S` contact shapes the loop is `S (S - 1) / 2` narrow-phase
calls per step, every step, at `dt = 1e-4` — and the shapes of a humanoid are
overwhelmingly far apart. A sphere-overlap rejection using the already-written
`bounding_sphere` would discard most pairs before any support-mapping call, and
is strictly cheaper than the cheapest GJK iteration.

What the example does. A hierarchy of cheap filters — AABB, bounding sphere and
body-pair tests — before any narrow-phase call (`03` §2), then sweep-and-prune
along the axis of largest spread (`03` §3), then a bounding-volume tree for
meshes (`03` §5). `docs/example-system-mapping.md` §2.5 lists the AABB filter as
candidate work "if the shape count grows". This entry adds that the mid-phase
test the step requires is already written and merely unwired, which makes the
first stage of the requirement a smaller change than the mapping suggests.

### M-16

**Body-pair exclusion is a fixed overlap allowance, not an exclusion
mechanism.** Severity: **gap**. Owning step: [07](07-collision-pipeline.md).

Step 07 item 3 requires a body-pair exclusion mechanism equivalent to
`contype`/`conaffinity` bitmasks plus parent-child and same-body exclusion, so
that "adjacent-link self-collision ... does not require per-model manual
exclusion lists". What exists: `contact_rows` skips pairs on the same body, and
for a parent-child pair subtracts the constant
`ADJACENT_OVERLAP_ALLOWANCE = 0.003` m from the computed depth. Evidence:
**code_inspected**.

Why the allowance is not the mechanism. It is a single global length applied to
every adjacent pair regardless of the geometry's actual designed overlap, so it
is simultaneously too small for joints whose shapes interpenetrate by more than
3 mm — which generate a permanent spurious contact force at the joint, pushing
the links apart for the whole run — and too large for shapes that should
genuinely collide at 2 mm. It also covers only the immediate parent-child
relation: grandparent-grandchild and sibling pairs across a shared joint get no
allowance at all, and those are the pairs a humanoid's shoulder and hip produce.
And there is no per-pair or per-shape mask of any kind, so a model that needs one
exception needs a code change.

### M-17

**The ported integrators, passive forces and solver are not wired into the
stepper.** Severity: **gap**. Owning steps: [03](03-integrators.md),
[05](05-passive-forces-and-actuators.md),
[06](06-equality-constraints-and-solver.md).

The three pure modules `integrators`, `passive` and `constraint` are
source-faithful, unit-tested, and — as section 1.1 records — correct. None of
them is reachable from `MultibodyState::step`. The step files state this
individually ("Not yet wired into `MultibodyState::step`"). Recorded here as one
entry because the *combined* effect is larger than the three separate notes
suggest: the simulator that actually runs shares no code with the ported
formulation, so the ports are not an implementation of the specified behaviour
but a second, parallel one.

The duplication is concrete:

| Concern | What `MultibodyState::step` uses | What the port provides |
| --- | --- | --- |
| Integration | An inline semi-implicit Euler written into `step_inner` | `integrators::semi_implicit_euler_step`, `implicit_damping_euler_step`, `rk4_step` |
| Actuation | `stance_torques`: one global `kp`, a per-DOF `kd`, clamped to the joint's effort limit | `passive::affine_actuator_force`, the gain/bias factoring step 05 item 1 requires |
| Joint springs and dampers | nothing | `passive::joint_spring_force`, `joint_damper_force` |
| Soft constraints | nothing; penalty contact and the limit clamp | `constraint::{impedance, stiffness_damping, effective_timeconst, solve_equality}` |

Two consequences worth naming. First, `stance_torques` does not consult
`robot_model`'s `ActuatorSpec` or `PdGains` at all, so the actuator factoring of
step 05 item 1 is neither used nor tested against the model layer it is supposed
to re-examine. Second, the `2 * timestep` time-constant check of step 06 item 3
— the one the step calls "the internal safety check the reference simulator
itself relies on", and which exists and is tested in `constraint.rs` — never
runs against the contact stiffness the simulator actually uses, which is the one
place a too-stiff constraint has been measured to diverge.

### M-18

**The implicit integrator cannot be reached without `M(q)`, which CRBA would
supply.** Severity: **gap**. Owning steps: [02](02-pure-dynamics-core.md),
[03](03-integrators.md).

`integrators::implicit_damping_euler_step` takes `mass` as a dense row-major
`nv * nv` matrix. Nothing in the workspace produces one: step 02 records that
CRBA and `M(q)` are not implemented, and a grep for a mass-matrix producer finds
only the spec files. The articulated-body algorithm deliberately never forms
`M`. Evidence: **code_inspected**.

Therefore step 03's "Done when" — "Each integrator option is selectable per
run" — and its item 4 MUST ("an integrator equivalent to the reference's
`implicitfast` ... before any claim of 'stable at a usable timestep on a real
humanoid' is made") are blocked on step 02's unimplemented CRBA, and this
dependency is not stated in either step. Step 03's "Depends on" names step 02,
but step 02's own open-questions field says "none blocking", so the chain reads
as unblocked when it is not.

The dependency is on `M(q)` specifically, not on CRBA as such. Three routes
exist and they are not equivalent:

1. Implement CRBA, as step 02 and `02` §2.9 of the example describe. `O(nv^2)`
   to form `M`, then `O(nv^3)` dense or `O(nv)` sparse factorisation over the
   tree (`02` §2.10–§2.11).
2. Form `M` column by column from `nv` inverse-dynamics passes. Correct, needs
   no new algorithm, and costs `nv` RNE passes per step — 35 for the G1.
3. Keep ABA and make the damping implicit inside it. This is what the example's
   own `implicitfast` effectively does for the diagonal case, and `11` §3.4
   (`mjd_freeMhat`, the effective mass `M + hD`) is the formulation
   `docs/example-system-mapping.md` §2.1 already points at for a free base.

Route 2 is the cheapest way to *unblock* the step and measure whether the
implicit step fixes the `dt = 5e-4` divergence recorded in `fidelity.rs`; route
1 is the one step 02 requires. Recording the distinction matters because the
measurement that would justify the work does not need the work.

### M-19

**`rk4_step` cannot integrate a `Free` joint, the case that motivates it.**
Severity: **gap**. Owning step: [03](03-integrators.md).

`integrators::rk4_step` asserts `q.len() == qd.len()` and advances stage
positions by `q + h * v`. Both are valid only for scalar coordinates. A `Free`
joint has seven position values against six velocities, so the assertion fails
outright; and even with the widths reconciled, `q + h * v` is not a valid
quaternion update — it leaves the manifold, which is why the orientation path
needs the integration the module's own doc comment defers ("quaternion
`Free`/`Ball` joints need the reference's `integratePos` and are not ported
here"). Evidence: **code_inspected**.

The gap is pointed. Step 03 item 4's stated reason for wanting RK4 is
"long-horizon energy-conservation-sensitive use (free-floating tumbling bodies
specifically; the reference's own docs note `implicitfast`'s dropped gyroscopic
term 'mildly damps' a standalone spinning free body, which is now directly
relevant once `Free` joints exist)". The one configuration RK4 is specified for
is the one configuration `rk4_step` rejects. The measurement in
[M-01](#m-01) — 1.9 % energy gain on an anisotropic spinning free body in one
simulated second — is the quantity RK4 was wanted for, and it cannot currently
be taken with RK4.

`docs/example-system-mapping.md` §2.2 already names the check to run
(`11` §1.7 `mjd_quatIntegrate` against `integrate_free_position` on a spinning
free body). That check and this gap are the same work.

### M-20

**The fidelity battery has no contact, collision or constraint row.** Severity:
**gap**. Owning step: [04](04-fidelity-loop.md).

The archival battery in `fidelity.rs` — the thing that produces the schema-v7
report and the `all_pass` gate — declares seven checks:
`aba_rne_closed_form`, `articulated_energy_drift`, `energy_drift`, `free_fall`,
`pd_equilibrium`, `pendulum_period`, `pendulum_trajectory`, plus the measured G1
zero-torque probe rows. Every one exercises a one-degree-of-freedom pendulum, a
single free body, or a PD equilibrium. Evidence: **code_inspected**.

Nothing in the battery exercises a contact force, a collision routine, the
joint-limit solve, the equality solver, or a multi-degree-of-freedom tree with
coupled inertia. Those subsystems have unit tests — `collision.rs` checks sphere
overlap and capsule distance against analytic values, `articulated.rs` checks
rest heights and a double pendulum round trip — but unit tests are not the
battery, do not carry tier-indexed tolerances, and do not appear in the
fidelity report that step 04 defines as the record.

Why this is the entry that matters most for this appendix. Of the challenges
above, M-04 through M-12 and M-16 all live in the contact and collision path.
That is precisely the region the oracle plan does not cover, and the two facts
are related: an uncovered subsystem is where divergence accumulates without
being noticed. Step 04's own framing supports the extension — its oracles are
closed forms, conservation laws, and hand-derived reference values, and all
three are available here. Analytic rest penetration under a known penalty
stiffness is a closed form (and would have caught [M-04](#m-04) immediately);
frictionless sliding at equilibrium is a conservation law; an analytic
sphere-sphere or plane-box penetration depth is a hand-derived value for the
narrow phase.

### M-21

**Fidelity knobs are compile-time constants.** Severity: **gap**. Owning step:
[09](09-fidelity-tiers.md).

Step 09 section 12.1 states that each of seven knobs "MUST each be an explicit,
per-run, queryable parameter of a simulation configuration — never a
compile-time constant and never silently defaulted without the default being
documented". Current state of each:

| Knob (step 09 §12.1) | Where it lives now | Per-run? | Queryable? |
| --- | --- | --- | --- |
| Timestep | argument to `step` and `Tier::timestep` | yes | yes |
| Integrator | `Tier::integrator`, a `String` validated to accept only `INTEGRATOR_SEMI_IMPLICIT_EULER` | declared only | yes, but single-valued |
| Solver algorithm | `Tier::solver`, validated to accept only `SOLVER_EXPLICIT_ABA` | declared only | yes, but single-valued |
| Solver iteration / tolerance budget | `CONTACT_PGS_SWEEPS = 500`, `CONTACT_PGS_TOLERANCE = 1e-10`, private consts in `articulated.rs` | no | no |
| Contact dimensionality (`condim`) | does not exist | no | no |
| Contact-slot budget | does not exist on the CPU path | no | no |
| Collision margin / gap | does not exist; `plane_sphere_contact` takes a `margin` argument but no caller in the contact path passes one | no | no |

Three further constants belong on that list because they change the physics, not
only the cost: `RIM_SAMPLES = 8` (see [M-04](#m-04) — it scales contact
stiffness), `ADJACENT_OVERLAP_ALLOWANCE = 0.003` (see [M-16](#m-16)), and
`CONTACT_REFERENCE_MASS = 1.0` (see [M-09](#m-09)). None is reachable or
recordable per run.

`Tier` therefore distinguishes runs by timestep alone, and step 09's closing
requirement — "every tier MUST be expressible purely as a named bundle of the
Section 12.1 knobs" — has four of seven knobs with nothing to bundle. This does
not block step 09's design work, but it does mean a tier cannot currently be
*recorded* faithfully with a run, which is what makes a fidelity claim auditable.

### M-22

**The GPU dense path is `f32` and capped at 6 degrees of freedom; the G1 needs
35 at `cond(M) ~ 1e5`.** Severity: **gap**. Owning steps:
[11](11-gpu-batched-backend.md), [12](12-cpu-gpu-agreement.md).

Step 00 section 1.2 makes batched GPU execution the primary execution mode. What
`robot-sim-gpu` implements is a set of elementwise and small-dense `f32`
kernels: semi-implicit Euler, implicit-damping Euler, an equality-only dense
solve, spring/damper and affine actuator forces, a pendulum-specific RK4, and
plane-sphere contact. No ABA, no RNE, no collision pipeline, no contact solver.
The dense kernels are capped at `MAX_DOF = 6` because each thread holds its
Cholesky workspace in a per-thread `Array::<f32>::new(36)` local array.
Evidence: **code_inspected**.

Two quantified obstacles to scaling that design to the real fixture, measured on
the vendored G1:

**Size.** `ArticulatedBody::from_profile` on
`fixtures::unitree_g1_29dof_floating_base` gives `dof = 35`, `nq = 36`
(measured). One thread per environment with a dense `35 x 35 f32` Cholesky
workspace needs 1225 floats — 4.9 KiB of per-thread local storage, against
`MAX_DOF`'s present 144 bytes. That is well beyond a per-thread register or
local-memory budget on the hardware this backend targets, so raising `MAX_DOF`
is not the change: either the factorisation becomes sparse over the kinematic
tree, as the example's `qLD` does (`02` §2.10–§2.11), or the work assignment
moves off one-thread-per-environment. This is the substance of Appendix B
**S-01** (does one-thread-per-environment actually win?) and the first concrete
number against it.

**Conditioning.** Measured on the same fixture: link diagonal inertias span
1.00e-7 to 5.91e-2 kg m^2, a ratio of 5.9e5; link masses span 1.00e-3 to 6.78 kg,
a ratio of 6.8e3. Both extremes come from the vendored Unitree URDF itself, not
from any clamp in the importer — the smallest `ixx` in
`assets/robots/unitree_g1/g1_29dof_rev_1_0.urdf` is literally `1e-7` and the
smallest mass `0.001`. The joint-space consequence, measured through
`inverse_inertia_diagonal` at the home configuration: `(M^-1)_jj` spans 3.30e-2
to 3.16e3, a ratio of 9.58e4, so `cond(M) >~ 1e5` there.

`f32` carries about 1.2e-7 relative precision. A Cholesky solve loses roughly
`cond(M)` in relative accuracy, so `1e5` leaves on the order of `1e-2`
relative — two significant digits — before any accumulation over a rollout. That
is a quantitative argument that a dense `f32` factorisation of this robot's mass
matrix is not merely less accurate than the `f64` CPU path but structurally
unsuitable, and it belongs in step 12's agreement-tolerance reasoning rather
than being discovered as an agreement-test failure. The example's own GPU
sections set per-stage tolerances for a single-precision port separately from
the double-precision reference (`15` §11), which
`docs/example-system-mapping.md` §2.8 records as candidate work; the
conditioning number above is what such a tolerance has to be set against.

Evidence for the measurements in this entry: **measured** (this audit;
deterministic).

### M-23

**`solve_equality` refactorises `M` once per constraint row.** Severity:
**cost**. Owning step: [06](06-equality-constraints-and-solver.md).

```rust
for row in rows {
    let mut factor = mass.to_vec();
    let mut x = row.jacobian.clone();
    if !cholesky_solve(&mut factor, &mut x, nv) { ... }
    minv_jt.push(x);
}
```

`cholesky_solve` factorises in place and destroys its input, so the loop copies
and refactorises the same `M` for every row: `O(m nv^3)` where
`O(nv^3 + m nv^2)` suffices. Factorising once and reusing the triangle for `m`
forward/back substitutions is the standard form and the one the example uses —
it factorises `M` once per step and back-substitutes per row (`02`
§2.10–§2.11). Evidence: **code_inspected**.

Not urgent at today's sizes, and recorded because the fix also removes a
correctness hazard: with one factorisation there is one place where
non-positive-definiteness is detected, rather than `m` places that can disagree
after `m` independent copies.

### M-24

**`inverse_inertia_diagonal` costs `nv` forward-dynamics passes and runs inside
the stance loop.** Severity: **cost**. Owning step:
[02](02-pure-dynamics-core.md).

`ArticulatedBody::inverse_inertia_diagonal` extracts the diagonal of `M^-1` by
calling `forward_dynamics` once per degree of freedom with a unit torque — 35
full ABA passes for the G1 — and keeps only `column[j]` from each, discarding the
other `nv - 1` entries it computed. `stable_damping` calls it, and
`stable_damping` is what the stance-hold qualification run uses to bound its
explicit damping gains. Evidence: **code_inspected**.

This is the same `M(q)` dependency as [M-18](#m-18) seen from the other side:
the quantity wanted is a property of the mass matrix, obtained by inverting the
dynamics 35 times because no mass matrix exists. A CRBA pass would give the
whole of `M` for less than the cost of this diagonal, which strengthens the case
for step 02's CRBA on grounds independent of the implicit integrator. It is also
worth noting that the diagonal of `M^-1` is not the reciprocal of the diagonal
of `M`, so the quantity cannot be had more cheaply from `M` without a
factorisation — the comparison is CRBA plus one factorisation against 35 ABA
passes, not CRBA alone.

---

## 3. Cross-cutting observations

Three patterns run through the entries above and are worth stating once rather
than per entry.

**3.1 The divergence is concentrated after the contact normal.** The dynamics
core is correct (section 1.1) and the ported formulation functions are correct.
Of 24 entries, eleven (M-04 to M-12, M-16, M-20) are in contact, collision or
the constraint path, and one more (M-15) is the pipeline that feeds it. This
matches what `docs/example-system-mapping.md` §2.4 says of itself — "the area
the example treats most fully, and the one where this repository is weakest" —
and it is also the region with no archival oracle coverage ([M-20](#m-20)). The
ordering suggested by that coincidence is to extend the battery before, or
alongside, fixing the entries, so that each fix is measured rather than asserted.

**3.2 Absolute tolerances on scale-dependent quantities.** `1e-18` on a squared
cross product and on a signed volume (`complete_simplex`, `box_sat`), `1e-12`
and `1e-7` on lengths (`box_box_contacts`), `1e-14` absolute on a pivot
(`solve_small_dense`), `1e-12` on a quaternion norm and `1e-9` on another
(`normalize_quaternion` versus `axis_angle_rotation`). Each embeds an assumption
that the model is at metre and kilogram scale. `solve_dense` shows the
alternative the crate already knows how to write — tolerance relative to the
largest entry, with the rule stated in the doc comment. The G1's five-orders-of-
magnitude spread in `(M^-1)_jj` ([M-22](#m-22)) is the concrete reason this is
not a theoretical tidiness point.

**3.3 Two implementations of the same concern.** Integration, actuation and
soft-constraint handling each exist twice: once as a source-faithful pure
function that is correct and unreachable, once as an inline path in the stepper
that runs ([M-17](#m-17)). Friction exists twice with two different
regularisations ([M-10](#m-10)). The hard limit exists twice, as an active set
and as a clamp ([M-08](#m-08)). In each case the ported or principled version is
the one the step files describe and the ad-hoc version is the one that executes,
so the Status fields of steps 03, 05 and 06 describe code that does not
participate in a simulation run. That is the single largest reason the step files
read as further along than the running system is.

**3.4 Notes on the non-normative mapping document.** `docs/example-system-mapping.md` was read as the guide to how the example folder is used, and its rules were followed. Four of its "Current state" cells read differently against the audited working tree than against the commit, which is expected given section 1.0 — they are recorded so a reader comparing the two documents is not misled, not offered as errors. The file is untracked in the working checkout and so is not on this branch, which is the second reason they are recorded here rather than edited there.

| Mapping cell | Says | Against the audited working tree |
| --- | --- | --- |
| §2.5, the AABB filter row | "All pairs of contact shapes are tested (`contact_wrenches`)" | Still all pairs; the function is now `ArticulatedBody::contact_rows`. The cell's conclusion holds, and [M-15](#m-15) adds that `Placed::bounding_sphere` already exists unused, so the first stage is a smaller change than "add an AABB filter" suggests |
| §2.6, the box-box row | "GJK and EPA give one contact point (`gjk_epa`)" | `collision::box_box_contacts` now implements the 15-axis separating-axis test and a corner manifold of up to eight points, so the SAT half of that candidate work is done. The remaining gap is face clipping, and the corner manifold has its own defect in the edge-edge case ([M-07](#m-07)) |
| §2.4, the limit row | "Hard limits with a proximity band of 5 and 95 percent of range" | The proximity band is a sensor signal (`limit_fractions`); enforcement is `apply_hard_limits` plus a position clamp in the stepper. Keeping the two definitions aligned, which the cell asks for, has to account for the clamp being a third mechanism ([M-08](#m-08)) |
| §2.1, the RK4 row | "Not implemented" | `integrators::rk4_step` exists and is correct for scalar joints; the gap is that it cannot take a `Free` joint ([M-19](#m-19)), which is narrower and more pointed than "not implemented" |

Two further cells are accurate but understate a consequence this appendix supplies: §2.6's cylinder and box point rules are a correctness matter, not only an accuracy one ([M-04](#m-04)), and §2.7's "replace only if a degenerate-simplex failure is measured" asks for a measurement whose target is a symmetric flat-on-flat configuration rather than a random one ([M-06](#m-06)).

## 4. Routing

Entries needing a decision rather than an implementation are routed to
[Appendix B](appendix-b-open-questions.md) as new rows S-07 to S-09. Everything
else is an implementation task against its owning step, and nothing here is
scheduled by this appendix.

| Entry | Routed to |
| --- | --- |
| [M-01](#m-01) | Appendix B **S-07** — change the representation, or amend step 01 item 3 and record the deviation |
| [M-03](#m-03) | Appendix B **S-08** — eigenvalue triangle inequality, or the full pseudo-inertia realisability condition |
| [M-09](#m-09), [M-04](#m-04) | Appendix B **S-09** — whether contact adopts the example's `A_ii`-proportional impedance, which subsumes both |
| all others | implementation task against the owning step in section 1.4 |

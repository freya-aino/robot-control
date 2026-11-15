# robot-sim

Simulation interface, authored physics, and the `Environment` trait, per
SPECS.md §9.1. Depends on `robot-types` and `robot-model`.

**The target changed mid-implementation.** `SIMULATION_SPECS.md` in this
directory is now the authoritative specification for where this crate is
going: GPU-bound, batched execution across thousands of environments,
variable fidelity as a first-class interface, a Rust-native GPU-compute
path, and a backend-agnostic boundary so every other crate depends on
`Environment` and never on a physics-backend-specific type. It is grounded
in direct research against published GPU-batched simulators, CubeCL,
`wgpu` and Candle — read `SIMULATION_SPECS.md` before extending anything
below; this README describes what exists today, that document describes
what it's becoming.

## What's here

| Module | Contents |
| --- | --- |
| `pendulum` | `Pendulum`: a damped, actuated pendulum integrated by semi-implicit Euler — the first qualification benchmark Appendix B.4 names |
| `contact` | `ContactBody`: a point mass with penalty-spring/Coulomb-friction ground contact — the second benchmark Appendix B.4 names. `GroundContact`: the per-sphere penalty law (regularised Coulomb friction, separation clamp) used by articulated bodies; see ADR 0010. `plane_sphere_contact`: pure plane-sphere geometry from the standard formulation against a plane given in its own frame (signed distance, normal, contact point, margin). `GroundContact` is the legacy penalty law; no command reaches it since the 0.1.6 switch-over |
| `env` | `Environment` trait (reset/step/seeded randomization) and `PendulumEnv`, a trainable RL environment wrapping `pendulum` |
| `backend` | `Backend: Environment` adds `backend_name()`; implemented for `PendulumEnv` (`cpu-pendulum`) |
| `multibody_env` | `MultibodyEnv`: `Environment` and `Backend` over `ArticulatedBody`. Without `with_ground` it steps `MultibodyState::step` (`cpu-multibody`, a floating base falls). With a `GroundContact` attached it steps `step_with_ground` (`cpu-multibody-ground`). Rejects a wrong-length `q_home` with `WidthMismatch` |
| `spatial` | Spatial-vector algebra on plain arrays (`[f64; 3]` vectors, row-major `[f64; 9]` rotations, `[f64; 6]` motion/force vectors, `[f64; 36]` spatial inertia) — the 6D motion/force representation the multi-body solver is built on. Uses `std` only; no `robot_types` or `robot_model` types |
| `fidelity` | Fidelity harness (ADR 0008, schema v8): `FidelityCheck` plug-in trait, `run_checks`, the archival ten-check analytic battery (v8 added `plane_normal_force_balance`, `friction_cone_bound` and `narrowphase_depth`; (the v6 seventh check `pendulum_trajectory` compares the 1 rad angle series with `oracle::pendulum_closed_form` through `oracle::compare_series`, abs-only, with a per-check `sample` index) at the `reference` (100 µs) and `coarse` (1 ms) tiers, per-check `system` attribution, `[[contact]]` G1 applicability per tier, GPU checks recorded as not applicable, the measured G1 zero-torque `[[probe]]` row (v7, scenario runner `g1_zero_torque_probe`), and a strict TOML report reader/writer through `robot-artifacts`. Scope is the analytic subset plus one measured probe; it does not qualify G1 ground contact |
| `oracle` | Pure `std` oracle module, no `robot_types`, `robot-model` or `HashMap` types: `pendulum_closed_form` (exact pendulum angle for any release angle, Landen/AGM `sn`, domain `\|theta0\| < pi`, complementary modulus from `cos`), `pendulum_period`, `energy_drift` (zero and non-finite `E(0)` guarded only), `compare_series` (explicit `LengthMismatch`, `EmptySeries`, `InvalidTolerance` errors; pass per sample `\|r - c\| <= abs + rel*\|r\|`) returning `OracleReport` |
| `integrators` | Pure `std` integrators implemented from the standard formulation: `semi_implicit_euler_step` (`mj_advance` order), `implicit_damping_euler_step` (`mj_EulerSkip` damping branch, Cholesky solve, `None` and unchanged state on a non-SPD system or non-finite inputs), `rk4_step` (`mj_RungeKutta`, N=4). Scalar joints only; `World::step` selects semi-implicit Euler, `implicitfast` or RK4 per run through `SimOptions::integrator`, while `MultibodyState` remains the legacy semi-implicit stepper |
| `passive` | Pure `std` passive and actuator forces from the standard formulation: `joint_spring_force` and `joint_damper_force` (`mj_springdamper`, linear case), `affine_actuator_force` (`mj_fwdActuation`, `GAIN_AFFINE`/`BIAS_AFFINE`). Polynomial, muscle, tendon, flex and gravity-compensation terms are not ported |
| `constraint` | Pure `std` equality-constraint subset implemented from the standard formulation: `joint_equality_row` (`mjEQ_JOINT`, one or two scalar joints with the quartic coupling), `impedance`, `stiffness_damping`, `effective_timeconst` (`2*dt` clamp, mixed-format refusal), and `solve_equality`, a closed-form solve of an equality-only problem returning forces, `qacc` and clamped rows. Kept as the CPU oracle for the GPU equality kernel; simulation uses `solver`/`newton`, which carry contact, friction, limit and equality rows together. Not ported: connect, weld and tendon rows |
| `articulated` | `ArticulatedBody`: multi-body forward (Articulated-Body Algorithm) and inverse (Recursive Newton-Euler) dynamics over any `robot_model::RobotProfile`, including a floating base. Sphere ground contact (`forward_dynamics_with_ground`, `step_with_ground`, `lowest_contact_height`), a clamped PD `stance_torques` hold, plus `MultibodyState`, a semi-implicit-Euler stepper. Legacy path: its ABA/RNE recursions are the oracle the `composite` path is validated against, and its sphere-penalty contact is retired and reached by no command |
| `composite` | Example `02`'s dynamics on plain arrays: `crb` (composite-rigid-body mass matrix), `factor_m`/`solve_ld` (`L' D L`), `rne` bias forces, `forward_dynamics`, `integrate_pos` with the `Free`-joint quaternion path, and `rk_correct_free_velocity` (Munthe-Kaas correction for RK4) |
| `collision` | Broad-phase rejection (`spheres_apart`, `aabbs_apart`) and pair filters, signed-volume GJK with EPA (`CcdConfig` budget), exact plane manifolds for sphere, box, cylinder and convex hull, and box-box SAT with face clipping. `HullTopology` (adjacency graph, coplanar-merged face polygons, 27-direction seed grid, built once per hull) gives a hull the hill-climbing support walk of `mjc_hillclimbSupport` at ten vertices and above and the up-to-four-point plane manifold of `mjc_PlaneConvex` through `hull4` |
| `solver` | The unified constraint problem of examples `07` and `08`: contact rows as the generator-direction pyramid (`contact_rows`, all four `condim` values, `sliding_friction`, `decode_pyramid`), limit and equality rows, `solref`/`solimp` impedance (`row_impedance`) and the projected Gauss-Seidel `solve` used as the cross-check |
| `newton` | `solve_newton`: the primal Newton solver of example `08` section 6, the default algorithm (Gauss-Seidel needs about a thousand sweeps where Newton needs one on sixteen pyramidal rows) |
| `options` | `SimOptions` (every Section 12.1 knob plus `limit_margin`, `warm_start`, `solref`, `solimp` and `CcdConfig`), `Integrator` (`Euler`, `ImplicitFast`, `Rk4`), `SolverAlgorithm`, and `Tier` — the named fidelity bundles `rollout`/`reference`/`acceptance` of S-03, grounded in the measured stance sweep; validated rather than clamped, and `record()` emits them so a run carries its configuration |
| `sensors` | `SensorSuite`: noise-free joint encoders and an IMU at a link site (specific force `R'(a - g)` and angular velocity from finite differences of the site pose); noise is `robot-perception`'s, per specs Appendix B S-06 |
| `world` | `World`: the default simulation path (`robot sim multibody`) — composite dynamics, exact collision manifolds, the unified solver, the model's declared actuators through `actuator_torques` (`ActuationMode`), `Ground` from the scene layer; `WorldEnv` is its `Environment`/`Backend` (`cpu-world`) |

## What this crate has and does not yet have

The `world` path (specs 0.1.5-draft onward, see `specs/README.md`) runs
composite dynamics, exact collision manifolds, the unified contact, limit
and equality solver with primal Newton as default, the model's declared
actuators, and semi-implicit Euler, `implicitfast` or RK4 per run. Every
one of the G1's 33 declared colliders is contacted (10 spheres, 2 boxes, 6
cylinders and 15 meshes reduced to convex hulls at import). Not yet
implemented: `Ball` joints; full `implicit` (the qLU path and
`mjd_freeMhat`'s free-body gyroscopic block); polynomial, muscle, DC-motor,
PID, tendon, flex, gravity-compensation and fluid passive terms; connect,
weld and tendon equality rows; the elliptic friction cone and constraint
islands; and a mid-phase BVH. Closed in
0.1.10-draft: hull support hill-climbs the vertex adjacency graph of
`collision::HullTopology` (built once per hull at `World` construction from
`robot_model::hull::hull_faces`), a hull resting on the plane gets the
up-to-four-point incident-polygon manifold of `mjc_PlaneConvex` instead of
one point, and the `contact_slots` budget is enforced with the dropped
count reported per step. Closed in 0.1.11-draft: the named fidelity tiers
(S-03), `options::Tier` — `rollout` (`implicitfast` at 1 ms, the only
integrator that passes the stance battery there), `reference` (identical to
`SimOptions::default()`) and `acceptance` (RK4 at 100 µs) — grounded in the
measured stance sweep recorded in `specs/09-fidelity-tiers.md`, and
reachable per run as `robot sim multibody --tier`. The paragraphs that
follow describe the legacy `articulated` path, which is kept as the oracle.

`spatial`/`articulated` are a real, validated multi-body dynamics solver
(see their own doc comments and `SIMULATION_SPECS.md` §4 item 1: their
RNE core is a research-pass implementation, informed by direct comparison
against a textbook derivation, not a parity proof), and it now handles
a floating base:
`robot_model::joint::JointKind::Free` gives a profile's root six
unconstrained degrees of freedom against an explicit `world` link, and
`robot_model::fixtures::unitree_g1_29dof_floating_base` is a real G1
whose pelvis is a `Free` joint. Without contact the pelvis falls under
gravity, which the `the_real_g1_with_a_floating_base_falls_under_gravity`
test checks on the no-contact `step` path. With sphere ground contact
(`step_with_ground`) the same robot is held above the plane by a clamped PD
stance (`robot sim multibody --floating`, ADR 0010), which is a supported-stance
qualification and not a balance result. `SIMULATION_SPECS.md` §3 item 1 named
the floating-base gap; it is closed for the solver.

What remains missing, named precisely: there is no balance controller, so
the robot cannot stand, balance or walk on its own; the declared drive
gains are generated defaults (Appendix D M-28); the pelvis and torso carry
no collision geometry in the vendored layer, so with zero torque the robot
folds onto its limb hulls (the pelvis settles about 0.12 m above the plane
at `dt = 1e-4`); and there is
**no GPU batched execution path in this crate**. The separate unpublished crate `crates/robot-sim-gpu` (feature `cubecl`, ADR 0012) has CubeCL kernels for semi-implicit Euler, implicit-damping Euler and equality-only `solve_equality` (per-environment dense Cholesky, `nv,nrow ≤ 6`), spring/damper/actuator forces, equality residual, `solimp` impedance with its derivative `impP` (asserted for both sides of the midpoint), and plane-sphere contact. It also has a named pendulum RK4 kernel (`gpu_rk4_pendulum_step`, asserted over 4097 environments and 1000 steps). The general closure-based `rk4_step` is not ported and has no CPU fallback on the GPU, and there is no contact/friction/limit solver or batched G1/ABA on the GPU (`SIMULATION_SPECS.md` §11). Its gate 3 timing (dev build, RK4 and equality only) and gate 4 in-process repeatability are on one device per run, with no throughput claim. Tolerance records are in `crates/robot-sim-gpu/tolerances.toml`. Run its tests with `cargo test -p robot-sim-gpu --features cubecl` under a Vulkan ICD (see ADR 0012). The
robot cannot stand, balance or walk on its own, and it cannot run as a
GPU-batched training environment yet.

Everything here remains qualified the way SPECS.md §9.1 and Appendix B.4
ask: against an analytic reference (the pendulum's small-angle period,
the PD actuator's compensated equilibrium, a hand-derived point-mass-
pendulum closed form for the ABA/RNE solver) or an energy-conservation/
dissipation check (symplectic-integrator energy drift bounded over
200,000 pendulum steps and over 50,000 multi-link-chain steps; a lossy
contact model strictly losing energy on impact; friction that damps
without ever reversing a slide).

## Second correction pass (2026-11-01)

Fourteen of the twenty-four audit entries are now closed.
`specs/appendix-d-mathematical-challenges.md` section 1.6 is the current status
table, and 1.6.1 and 1.6.2 say what the two remaining entries need.

**New module `solver`.** Contact, joint-limit and equality rows solved as one
convex problem over constraint forces, which is what `specs/06` item 1 asks for
and what five audit entries folded into: `A = J M^-1 J^T` from the composite
factorization, `R_ii = (1-d)/d A_ii` with the exact diagonal instead of a fixed
1 kg reference mass, `a_u = J qacc_smooth + Jdot v` with both of the terms that
were missing, friction as rows bounded by the pyramidal cone against the normal
row's current force, and warm starting. The point Jacobian is checked against a
central finite difference of the kinematics.

**New module `options`.** The seven fidelity knobs of `specs/09` section 12.1 as
a per-run configuration that validates rather than clamps and that records every
knob for a run record. The narrow-phase budget is threaded through collision,
which is where the need for it was actually discovered.

**Collision.** The narrow phase is now the example's signed-volume GJK with the
polytope2, polytope3 and polytope4 constructions, which is where the
expanding-polytope algorithm's enclosure invariant comes from -- four non-zero
barycentric weights *are* strict enclosure, so there is no containment test left
to get wrong. Plane contact uses the exact bounded manifolds instead of sampling
eight corners or sixteen rim points, so the point count is a property of the
contact geometry. A broad phase rejects pairs before any support-mapping call,
and a real pair-exclusion filter replaces the global 3 mm overlap allowance.

**RK4 takes a free joint**, with the Runge-Kutta-Munthe-Kaas correction, and is
asserted higher order rather than merely better: more than a hundred times below
the semi-implicit step's energy drift on a tumbling body, and shrinking much
faster than first order when the timestep halves.

**The battery covers contact now** (schema v8), and writing those rows paid for
itself immediately: the first sample of the collision row exposed a head-on
collision between two convex shapes reporting no contact at all, because the
two-vertex polytope construction was missing. Random fuzzing had never produced
that configuration because random orientations are never collinear. That is the
argument for `specs/04`'s coverage gap in one example.

**One correction to the audit's own analysis.** Appendix D M-04 claimed the
per-point penalty law was the reason rest penetration scaled with the sampled
point count. The solver fixes the force balance -- the total settles at the
weight however many rows divide it -- but the `1/n` depth scaling survives,
because with `A_ii ~ 1/m` and `R_i = (1-d)/d A_ii` the per-row force is `m g / n`
either way. The example's formulation has the same property. What removes it is
the bounded manifolds, which cap the count geometrically. Two of that commit's
tests were first written asserting the opposite, from penalty-spring intuition.

**What was not done at this point** (historical: the 0.1.5-draft switch-over
made `world::World` the default path and closed M-17). `MultibodyState::step`
was still the stepper the CLI and the
environments ran, so `composite`, `solver` and `options` were available but not
the default path (M-17); section 1.6.1 explains why that switch wants to be one
change rather than several, since the `Free`-joint layout has to move everywhere
at once. And the GPU backend's `f32` and six-degree-of-freedom limits are
untouched (M-22), now with the added requirement that a batched solver needs the
Newton path, because a sequential per-row update is exactly what the example's
own GPU backend drops projected Gauss-Seidel for.

## What the corrections did (2026-11-01)

`docs/example-system-specs/` is now the single source of truth for the
mathematics and execution model of multi-body simulation. The project's goals,
scope, language, dependency policy, evidence discipline and GPU-bound target are
unchanged; the example is read for how to compute, not for what to build.
`specs/appendix-d-mathematical-challenges.md` section 1.0.1 records the change
and 1.5 records which of its 24 entries are now closed.

**New module `composite`.** The example's own dynamics path, which is not the
articulated-body algorithm: it forms the mass matrix and factorises it. Position
and velocity stages (`subtree_com`, `cinert`, `cdof`, `cdof_dot`), the
composite-rigid-body `M(q)`, its `L' D L` factorization over the
degree-of-freedom tree, back-substitution, recursive Newton-Euler, forward
dynamics as `qacc = M^-1 (tau - bias)`, position integration, and a stepper.
Everything is in world orientation with the reference point at the owning tree's
centre of mass, which is the example's frame.

It is validated by cross-algorithm agreement, not by assertion: `M` from the
composite recursion matches `M` built column by column from `articulated`'s
local-frame inverse dynamics to 1e-10 on the full 29-degree-of-freedom G1, and
the bias force to 1e-9. Two different algorithms in two different coordinate
frames. The degree-of-freedom-tree factorization is also the structure a batched
GPU backend parallelises over, so this is the shape the eventual GPU path wants.

**The `Free` joint convention is fixed.** World-frame linear velocity in the
first three degrees of freedom, body-frame angular in the last three,
translation first, with an exact exponential-map quaternion update. A force-free
spinning body conserved world linear momentum to only 3.93e-3 per simulated
second before; it is now exact to below 1e-14, because under this convention the
translational rows of `M` are a constant `m*I` with zero bias and there is no
`-omega x u` cancellation left for an explicit step to approximate. The
anisotropic rotational energy drift is unchanged at about 1.8e-2 and is asserted
to be first-order in the timestep, because it never was the convention's fault:
it is explicit integration of Euler's equations, which is what an implicit or
RK4 step would address.

**Also fixed**: inertia validity now tests the principal moments rather than the
diagonal entries, via the new `robot_model::eig` (a tensor with eigenvalues
3, 1, -1 used to pass); joint axes are normalised at a single accessor, so
scaling a declared axis no longer changes the dynamics; the small dense solver's
pivot tolerance is relative to its matrix; `stiffness_damping` selects the
`solref` format with one predicate; and an edge-edge box contact is placed on
the edges instead of at a corner about 0.7 m away.

**What was not done at this point, and the honest reason** (historical: the
unified solver, the Newton default and the switch-over all landed in
0.1.4-draft and 0.1.5-draft; see the specs index). The unified constraint solver of
example files 07 and 08 was not yet built, so Appendix D M-04 and M-08 to M-12 stood:
contact is still the penalty and coupled-impedance pair, friction is still
outside the solve, and limits are still a separate active set plus a position
clamp. `MultibodyState::step` is still the stepper the CLI and the environments
run, so the composite path is available but not yet the default — switching over
means moving contact to the same path, which is the same piece of work.

M-05 and M-06, the GJK and EPA degeneracies, were attempted and reverted.
Appendix D section 1.5.1 records what they turned out to need: the example's
enclosure invariant comes from its GJK, not its EPA — four non-zero barycentric
weights from the signed-volume subdistance of 05 section 5.4 *are* strict
enclosure — and our `reduce` uses Ericson's region tests, which cannot provide
it. Building the polytope constructions of 05 section 6.1 on top of that
mismatch traded a false positive for a false negative and back. The fix is to
replace `reduce` with the subdistance first.

## What a full audit of the maths found (2026-10-31)

Every algorithm module in this crate, in `robot-sim-gpu`, and the mass-property
and joint layers of `robot-model` they depend on were read against two
references at once: the step files in `specs/`, which say what the subsystem must
do, and `docs/example-system-specs/` — a non-normative, 17-document description
of a complete working rigid-body and contact simulator, read as the mathematics
of a system known to work. The result is `specs/appendix-d-mathematical-challenges.md`,
24 numbered entries (M-01 to M-24). Read it before changing anything in
`collision`, `contact`, `constraint` or the contact path of `articulated`.

**What the audit confirmed.** The dynamics core is sound. `spatial`'s
motion/force duality, its `ad`/`ad*` operators and its parallel-axis spatial
inertia; `articulated`'s RNE (including the `a_0 = -g` gravity trick) and its
three-pass ABA; `integrators`' three steppers and its Cholesky; `constraint`'s
impedance sigmoid and its equality algebra; `collision`'s four support maps and
Ericson's closest-point-on-triangle; `contact`'s plane-sphere rule; and all of
`passive` were independently re-derived and agree term for term. Appendix D
section 1.1 lists each with what it was checked against. That is a derivation and
a code reading, not measured parity with any other engine.

**Two defects reachable through the public API, both measured.** A `Free` joint
stores its linear velocity in the **body** frame, where `specs/01` item 3 requires
the world frame; a force-free spinning free body therefore gains 0.39 % world
linear velocity and 0.14 % kinetic energy per simulated second at `dt = 1e-4`,
and controls at zero spin and at isotropic inertia isolate the convention as the
cause rather than the integrator (M-01). And a declared joint `axis` that is not
unit length makes the velocity map disagree with the pose map, because
`joint_transform` normalises and `motion_subspace_basis` does not: scaling a
pendulum's declared axis from `(0,1,0)` to `(0,2,0)` — the same physical axis —
halves its acceleration and quadruples its kinetic energy, and profile
validation accepts both (M-02). Separately, `robot_model`'s
`InertiaTensor::is_physically_plausible` tests the triangle inequality on the
diagonal entries only, so it accepts an indefinite tensor (M-03).

**Where the divergence concentrates.** Eleven of the 24 entries are in the
contact and collision path, plus one more for the pipeline feeding it. Contact
compliance scales with the number of sampled contact points, so a box resting on
a face settles four times shallower than a sphere of the same mass and changing
`RIM_SAMPLES` changes the physics (M-04); the EPA seed tetrahedron is not
required to enclose the origin and a flat tetrahedron is reported as containing
it (M-05, M-06); box-box edge-edge contact is placed at a corner, giving the
right depth with the wrong moment arm (M-07); contact impedance is referred to a
fixed 1 kg mass rather than the row's own effective mass, across a robot whose
effective masses span five orders of magnitude (M-09); friction sits outside the
solve, and pair contacts have none (M-10). That same region has no row in the
archival fidelity battery (M-20), which is not a coincidence worth leaving
unstated: it is where divergence accumulates without being noticed.

**The three "implemented but unreachable" modules.** `integrators`, `passive`
and `constraint` are correct and are not called from `MultibodyState::step`,
which runs its own inline Euler, its own clamped-PD actuation and its own
penalty contact. So integration, actuation and soft-constraint handling each
exist twice in this crate, and the version the step files describe is not the
version that executes (M-17). One consequence worth knowing: the `2*timestep`
time-constant check — the safety check `specs/06` item 3 calls the mechanism that
prevents exactly the timestep divergence measured on the G1 — exists, is tested,
and never runs against the contact stiffness the simulator actually uses.

Three entries need a specification decision rather than a fix and are open as
`specs/appendix-b-open-questions.md` **S-07** (the `Free` velocity convention),
**S-08** (which inertia-validity condition to adopt) and **S-09** (whether
contact takes the `A_ii`-proportional impedance, which subsumes M-04 and M-09).
The rest are implementation tasks against their owning step.

## A test that needed a second look

The first version of `a_pd_actuator_drives_the_pendulum_toward_a_target_angle`
used a plain PD law and failed: the pendulum settled near 0.20 rad instead
of the 0.3 rad target. That is not a bug in the integrator — it is the
textbook steady-state droop of an uncompensated PD controller fighting
gravity, and the fix was to add a gravity-feedforward term (compensating the
*current* gravity torque, not the target's), which is what a real joint
controller does for exactly this reason. The test and its comment now say
so, rather than quietly loosening the tolerance until the original bug
would have passed unnoticed.

## A real limit found by actually running it, not by theory

This probe still runs the legacy `MultibodyState::step_with_ground`
sphere-penalty path, not `world`. Since hard joint limits landed (2026-10-30)
the probe completes its 4000 steps and its row is `pass`; the failure
described below is the earlier, superseded measurement. On the `world` path
the zero-torque G1 folds onto its hull geometry instead.

Running the vendored G1 fixture through the legacy stepper (then the path
behind `robot sim multibody`) surfaced a genuine, now-documented instability. With zero
applied torque on the floating G1 and sphere ground contact, semi-implicit Euler
at `dt = 5e-4` s hits a singular joint inertia matrix at step 2236 (about 1.12
simulated seconds), recorded as a measured `[[probe]]` row (`status = "fail"`, `error = "SingularJointInertia"`, owned by `fidelity.rs`); the row does not change `all_pass`. The failure is also a model-coverage limit: the pelvis and torso carry no collision geometry, so the trunk reaches the plane through unmodelled shapes once the legs fold, and the row does not isolate the integrator's share. The clamped PD stance hold stays finite over 2 s at the same `dt`. `SIMULATION_SPECS.md` §4 item 3 ties this directly to a
textbook cause, confirmed by the integrator's own
documentation: even the simple default integrator already folds
joint-damping derivatives into an implicit solve, specifically because
plain explicit/semi-implicit integration isn't stable at a usable
timestep on a system with small distal masses and large effective
torque ratios — exactly what a wrist is. §4 item 4 requires an
implicit-in-velocity integrator before any further stability claim.

## The `Free` joint: one general implementation, not two

`ArticulatedBody`'s ABA/RNE are written once, generically over a joint's
degree-of-freedom count `k` (`[f64; 6]` motion-subspace bases, `k x k`
matrix solves by an explicit partial-pivot LU on plain arrays), rather than a scalar
fast path plus a separately-maintained multi-DOF path. Every existing
single-DOF test (the point-mass-pendulum closed form, the double-
pendulum round-trip, the real fixed-base G1) exercises that same general
code at `k = 1`, which is why all of them kept passing, unchanged,
through the `Free`-joint (`k = 6`) generalization — a strong regression
signal that the generalization didn't silently change the `k = 1`
behavior, not just a convenient implementation choice.

The `Free` joint itself is validated the same layered way everything
else in this crate is, starting from the smallest possible fixture (a
single free-floating body, no children) before the real G1 is ever
given a floating base: a body at rest with no forces stays at rest; a
body with no forces translates at exactly constant velocity; a body
falling under gravity alone matches the *exact discrete* projectile
formula semi-implicit Euler implies (not the continuous one — see the
test's own comment for why those differ and which one is the correct
check); a spinning body with isotropic inertia conserves its angular
velocity exactly, independent of this crate's own formulas for inertia;
and RNE/ABA round-trip through each other on a free body at a genuinely
rotated, non-identity orientation. Every one of these passed on the
first implementation attempt — a different experience from this
crate's attitude-filter and IK-solver history elsewhere in this project,
recorded here because a suspiciously clean first pass is itself a fact
worth stating, not quietly assumed to mean the tests were too weak
(they aren't: each is an independent closed form or conservation law,
not a tautology).

## Testing

`cargo test -p robot-sim` — 310 unit tests and 9 integration tests (`tests/fidelity_extension.rs`, the external `FidelityCheck` API, including the v6 series-tolerance rejections for `rel > 0` and `abs = NaN`). The `oracle` tests cover the separatrix period asymptote, a 3.0 rad RK4 cross-check, the largest admissible `theta0` below `pi`, and the energy zero policy. The articulated-body and spatial-algebra modules have dedicated regression coverage for the dense solver and array adapters. Fidelity tests (`cargo test -p robot-sim fidelity`) cover the pass rule, every declared tier, the report round trip, strict reader rejection (tampered pass/all_pass, relabelled system, wrong scope, contact rows that disagree with the timestep rule), duplicate names, the step budget, and failure injection at a 200 ms tier. Earlier coverage: the pendulum (analytic period,
energy bound, monotonic dissipation under damping, gravity-compensated PD
convergence), the contact body (steady-state penetration depth, energy loss
on impact, non-reversing friction, push-and-recover), the environment
wrapper (seeded reproducibility, divergence under different seeds, action
clamping), the spatial-algebra module (power-invariant duality, a
transform round-tripping through its own inverse, a hand-derived
kinetic-energy closed form), and the articulated-dynamics module: ABA
and RNE each matching a hand-derived point-mass-pendulum closed form,
ABA/RNE round-tripping through each other on a toy double pendulum, a
free-floating body and the real G1 (both fixed- and floating-base), a
two-link chain conserving energy over 50,000 free-motion steps, the
`Free`-joint closed-form/conservation checks described above, the real
G1 running 2,000 steps bolted at the pelvis without diverging, and the
real G1 with a floating base falling without ground (no-contact `step`), the
sphere-contact model (rest height, no-energy-injection, friction), the G1
stance hold (`stance_torques`, `stable_damping`), and `MultibodyEnv`'s backend
trajectory, seeded reset, invalid-input and physics-failure behavior. The
zero-torque G1 collapse is a known-limitation characterization test, not a
physical claim (ADR 0010).

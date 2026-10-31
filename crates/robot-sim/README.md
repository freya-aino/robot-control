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
| `contact` | `ContactBody`: a point mass with penalty-spring/Coulomb-friction ground contact — the second benchmark Appendix B.4 names. `GroundContact`: the per-sphere penalty law (regularised Coulomb friction, separation clamp) used by articulated bodies; see ADR 0010. `plane_sphere_contact`: pure `+z` plane-sphere geometry from the standard formulation (signed distance, normal, contact point, margin); the penalty force stays in `GroundContact` |
| `env` | `Environment` trait (reset/step/seeded randomization) and `PendulumEnv`, a trainable RL environment wrapping `pendulum` |
| `backend` | `Backend: Environment` adds `backend_name()`; implemented for `PendulumEnv` (`cpu-pendulum`) |
| `multibody_env` | `MultibodyEnv`: `Environment` and `Backend` over `ArticulatedBody`. Without `with_ground` it steps `MultibodyState::step` (`cpu-multibody`, a floating base falls). With a `GroundContact` attached it steps `step_with_ground` (`cpu-multibody-ground`). Rejects a wrong-length `q_home` with `WidthMismatch` |
| `spatial` | Spatial-vector algebra on plain arrays (`[f64; 3]` vectors, row-major `[f64; 9]` rotations, `[f64; 6]` motion/force vectors, `[f64; 36]` spatial inertia) — the 6D motion/force representation the multi-body solver is built on. Uses `std` only; no `robot_types` or `robot_model` types |
| `fidelity` | Fidelity harness (ADR 0008, schema v7): `FidelityCheck` plug-in trait, `run_checks`, the archival seven-check analytic battery (the v6 seventh check `pendulum_trajectory` compares the 1 rad angle series with `oracle::pendulum_closed_form` through `oracle::compare_series`, abs-only, with a per-check `sample` index) at the `reference` (100 µs) and `coarse` (1 ms) tiers, per-check `system` attribution, `[[contact]]` G1 applicability per tier, GPU checks recorded as not applicable, the measured G1 zero-torque `[[probe]]` row (v7, scenario runner `g1_zero_torque_probe`), and a strict TOML report reader/writer through `robot-artifacts`. Scope is the analytic subset plus one measured probe; it does not qualify G1 ground contact |
| `oracle` | Pure `std` oracle module, no `robot_types`, `robot-model` or `HashMap` types: `pendulum_closed_form` (exact pendulum angle for any release angle, Landen/AGM `sn`, domain `\|theta0\| < pi`, complementary modulus from `cos`), `pendulum_period`, `energy_drift` (zero and non-finite `E(0)` guarded only), `compare_series` (explicit `LengthMismatch`, `EmptySeries`, `InvalidTolerance` errors; pass per sample `\|r - c\| <= abs + rel*\|r\|`) returning `OracleReport` |
| `integrators` | Pure `std` integrators implemented from the standard formulation: `semi_implicit_euler_step` (`mj_advance` order), `implicit_damping_euler_step` (`mj_EulerSkip` damping branch, Cholesky solve, `None` and unchanged state on a non-SPD system or non-finite inputs), `rk4_step` (`mj_RungeKutta`, N=4). Scalar joints only; not wired into `MultibodyState::step` |
| `passive` | Pure `std` passive and actuator forces from the standard formulation: `joint_spring_force` and `joint_damper_force` (`mj_springdamper`, linear case), `affine_actuator_force` (`mj_fwdActuation`, `GAIN_AFFINE`/`BIAS_AFFINE`). Polynomial, muscle, tendon, flex and gravity-compensation terms are not ported |
| `constraint` | Pure `std` equality-constraint subset implemented from the standard formulation: `joint_equality_row` (`mjEQ_JOINT`, one or two scalar joints with the quartic coupling), `impedance`, `stiffness_damping`, `effective_timeconst` (`2*dt` clamp, mixed-format refusal), and `solve_equality`, a closed-form solve of an equality-only problem returning forces, `qacc` and clamped rows. Not wired into `MultibodyState::step`. Not ported: connect, weld and tendon rows, friction, limit and contact rows, the iterative solvers |
| `articulated` | `ArticulatedBody`: multi-body forward (Articulated-Body Algorithm) and inverse (Recursive Newton-Euler) dynamics over any `robot_model::RobotProfile`, including a floating base. Sphere ground contact (`forward_dynamics_with_ground`, `step_with_ground`, `lowest_contact_height`), a clamped PD `stance_torques` hold, plus `MultibodyState`, a semi-implicit-Euler stepper. Box, cylinder and mesh shapes are counted, not modelled (ADR 0010) |

## What this crate has and does not yet have

Steps 5 and 6 (standard-formulation ports, pure `std`: `integrators`,
`passive`, `constraint`, and the plane-sphere geometry in `contact`) are
implemented as standalone functions with hand-derived tests. Not yet
implemented: wiring them into `MultibodyState::step`; free and ball
joint integration (`mj_integratePos` quaternion path); sparse-`M`
`implicit`/`implicitfast`; polynomial, muscle, DC-motor, PID, tendon, flex,
gravity-compensation, fluid and contact passive terms; connect, weld and
tendon equality rows; friction, limit and contact rows, the shared
contact-plus-equality solver and its iterative solvers; and the non-plane
collision pairs.

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

What remains missing, named precisely: **contact is sphere-only**. Box,
cylinder and mesh collision shapes are counted but not modelled, so the
G1's pelvis and torso have no collision geometry and zero torque collapses
through the plane. There is no joint-limit or actuator model, no balance
controller, **no constraint solver wired into stepping; equality-only solver available as standalone** (penalty sphere contact forces used otherwise), and
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

Running the vendored G1 fixture (`robot sim multibody` in `robot-cli`)
through the stepper surfaced a genuine, now-documented instability. With zero
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

`cargo test -p robot-sim` — 191 unit tests and 9 integration tests (`tests/fidelity_extension.rs`, the external `FidelityCheck` API, including the v6 series-tolerance rejections for `rel > 0` and `abs = NaN`). The `oracle` tests cover the separatrix period asymptote, a 3.0 rad RK4 cross-check, the largest admissible `theta0` below `pi`, and the energy zero policy. The articulated-body and spatial-algebra modules have dedicated regression coverage for the dense solver and array adapters. Fidelity tests (`cargo test -p robot-sim fidelity`) cover the pass rule, every declared tier, the report round trip, strict reader rejection (tampered pass/all_pass, relabelled system, wrong scope, contact rows that disagree with the timestep rule), duplicate names, the step budget, and failure injection at a 200 ms tier. Earlier coverage: the pendulum (analytic period,
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

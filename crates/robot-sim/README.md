# robot-sim

Simulation interface, authored physics, and the `Environment` trait, per
SPECS.md §9.1. Depends on `robot-types` and `robot-model`.

**The target changed mid-implementation.** `SIMULATION_SPECS.md` in this
directory is now the authoritative specification for where this crate is
going: GPU-bound, batched execution across thousands of environments,
variable fidelity as a first-class interface, a Rust-native GPU-compute
path, and a backend-agnostic boundary so every other crate depends on
`Environment` and never on a physics-backend-specific type. It is grounded
in direct research against MuJoCo's own source/docs, MJX, Brax, CubeCL,
`wgpu` and Candle — read `SIMULATION_SPECS.md` before extending anything
below; this README describes what exists today, that document describes
what it's becoming.

## What's here

| Module | Contents |
| --- | --- |
| `pendulum` | `Pendulum`: a damped, actuated pendulum integrated by semi-implicit Euler — the first qualification benchmark Appendix B.4 names |
| `contact` | `ContactBody`: a point mass with penalty-spring/Coulomb-friction ground contact — the second benchmark Appendix B.4 names. `GroundContact`: the per-sphere penalty law (regularised Coulomb friction, separation clamp) used by articulated bodies; see ADR 0010 |
| `env` | `Environment` trait (reset/step/seeded randomization) and `PendulumEnv`, a trainable RL environment wrapping `pendulum` |
| `backend` | `Backend: Environment` adds `backend_name()`; implemented for `PendulumEnv` (`cpu-pendulum`) |
| `multibody_env` | `MultibodyEnv`: `Environment` and `Backend` over `ArticulatedBody`. Without `with_ground` it steps `MultibodyState::step` (`cpu-multibody`, a floating base falls). With a `GroundContact` attached it steps `step_with_ground` (`cpu-multibody-ground`). Rejects a wrong-length `q_home` with `WidthMismatch` |
| `spatial` | `Twist`/`Wrench` spatial-vector algebra and `SpatialInertia` — the 6D motion/force representation the multi-body solver is built on |
| `fidelity` | Fidelity harness (ADR 0008, schema v5): `FidelityCheck` plug-in trait, `run_checks`, the archival six-check analytic battery at the `reference` (100 µs) and `coarse` (1 ms) tiers, per-check `system` attribution, `[[contact]]` G1 applicability per tier, GPU checks recorded as not applicable, and a strict TOML report reader/writer through `robot-artifacts`. Scope is the analytic subset only; it does not qualify G1 ground contact |
| `articulated` | `ArticulatedBody`: multi-body forward (Articulated-Body Algorithm) and inverse (Recursive Newton-Euler) dynamics over any `robot_model::RobotProfile`, including a floating base. Sphere ground contact (`forward_dynamics_with_ground`, `step_with_ground`, `lowest_contact_height`), a clamped PD `stance_torques` hold, plus `MultibodyState`, a semi-implicit-Euler stepper. Box, cylinder and mesh shapes are counted, not modelled (ADR 0010) |

## What this crate has and does not yet have

`spatial`/`articulated` are a real, validated multi-body dynamics solver
(see their own doc comments and `SIMULATION_SPECS.md` §4 item 1: their
RNE core was confirmed, by direct comparison against MuJoCo's own source,
to be the same algorithm MuJoCo's engine runs, gravity-as-fictitious-
base-acceleration trick and all), and it now handles a floating base:
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
controller, and **no GPU execution path** (`SIMULATION_SPECS.md` §11). The
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
through `MultibodyState::step` surfaced a genuine, now-documented
instability: the semi-implicit Euler integrator diverges to NaN within
about 1.3 simulated seconds at `dt = 5e-4`, but runs stably at
`dt = 1e-4`. `SIMULATION_SPECS.md` §4 item 3 ties this directly to a
textbook cause confirmed by reading MuJoCo's own integrator
documentation: even MuJoCo's "simple" default integrator already folds
joint-damping derivatives into an implicit solve, specifically because
plain explicit/semi-implicit integration isn't stable at a usable
timestep on a system with small distal masses and large effective
torque ratios — exactly what a wrist is. §4 item 4 requires an
implicit-in-velocity integrator before any further stability claim.

## The `Free` joint: one general implementation, not two

`ArticulatedBody`'s ABA/RNE are written once, generically over a joint's
degree-of-freedom count `k` (`Vec<Twist>` motion-subspace bases, `k x k`
matrix solves via `robot_types::math::Matrix`), rather than a scalar
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

`cargo test -p robot-sim` — 107 unit tests and 6 integration tests (`tests/fidelity_extension.rs`, the external `FidelityCheck` API). Fidelity tests (`cargo test -p robot-sim fidelity`) cover the pass rule, every declared tier, the report round trip, strict reader rejection (tampered pass/all_pass, relabelled system, wrong scope, contact rows that disagree with the timestep rule), duplicate names, the step budget, and failure injection at a 200 ms tier. Earlier coverage: the pendulum (analytic period,
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

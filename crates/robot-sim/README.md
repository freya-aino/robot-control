# robot-sim

Simulation interface, authored single-body physics, and the `Environment`
trait, per SPECS.md §9.1. Depends on `robot-types` (and declares `robot-model`
for the articulated-dynamics work described below, not yet used).

## What's here

| Module | Contents |
| --- | --- |
| `pendulum` | `Pendulum`: a damped, actuated pendulum integrated by semi-implicit Euler — the first qualification benchmark Appendix B.4 names |
| `contact` | `ContactBody`: a point mass with penalty-spring/Coulomb-friction ground contact — the second benchmark Appendix B.4 names |
| `env` | `Environment` trait (reset/step/seeded randomization) and `PendulumEnv`, a trainable RL environment wrapping `pendulum` |

## What this crate deliberately does not claim

There is no multi-body articulated-dynamics engine here — no coupled
multi-joint humanoid physics. SPECS.md §9.1 requires qualifying Rapier's
existing multibody/collision path (or a verified alternative) *before*
authoring one, and that qualification has not happened in this revision
(see `docs/decisions/0003-std-only-default-closure.md`). Saying so here,
in the crate's own top-level doc comment, is what PLAN.md's Phase 0 exit
gate means by "Simulator throughput and fidelity limitations are
documented" — the gap is named, not hidden behind a `pendulum.rs` that
quietly stands in for something larger.

What *is* here is qualified the way SPECS.md §9.1 and Appendix B.4 ask:
against an analytic reference (the pendulum's small-angle period, the PD
actuator's compensated equilibrium) or an energy-conservation/dissipation
check (symplectic-integrator energy drift bounded over 200,000 steps; a
lossy contact model strictly losing energy on impact; friction that damps
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

## Testing

`cargo test -p robot-sim` — 14 tests across the pendulum (analytic period,
energy bound, monotonic dissipation under damping, gravity-compensated PD
convergence), the contact body (steady-state penetration depth, energy loss
on impact, non-reversing friction, push-and-recover), and the environment
wrapper (seeded reproducibility, divergence under different seeds, action
clamping).

# Step 05 — Passive forces and actuators

[Index](README.md) · [Appendix A](appendix-a-citations.md)

**Status**: **Actuators implemented and running.** `World::actuator_torques` is
`mj_fwdActuation` for a model's declared drives, as the AFFINE-gain/AFFINE-bias
position servo of
[example 10](../../../docs/example-system-specs/10_actuation_sensors_plugins.md)
section 2.2: `gainprm = [kp, 0, 0]`, `biasprm = [0, -kp, -kd]`, giving
`force = kp (ctrl - length) - kd velocity`, with the section 2.5 force clamp to
the drive's `maxForce`, the section 2.8 `inheritrange` clamp of the target into
the joint's own range, and the declared transmission applied as
`joint = motor (1 - loss) / ratio`. `robot sim multibody` runs it: the G1's 29
`PhysicsDriveAPI` drives hold the robot's posture. Appendix D M-25 and M-26 are
closed.

`robot_sim::passive` still holds the scalar-joint subset it always did —
`joint_spring_force` and `joint_damper_force` (`mj_springdamper`, linear
hinge/slide spring and dof damper) — and `affine_actuator_force`, which is now
the formula `World::actuator_torques` calls rather than an unreachable function.

**The declared gains are not the robot's**, which is Appendix D M-28 and a live
divergence rather than a closed one. All 29 of the G1's drives carry
`damping = 0.0004 x stiffness` to float precision and `targetPosition = 0`. A
tuned set cannot look like that: critical damping is `kd = 2 sqrt(kp I)`, so
`kd/kp = 2 sqrt(I/kp)` varies with each joint's inertia. Converted to
per-radian the gains span 132 to 13183 N m/rad at a damping ratio of 0.003 to
0.03. `ActuationMode` therefore makes the choice the caller's — `Drives` (the
declared servo), `Damping` (the declared damping with the stiffness dropped) or
`Zero` (no actuator force, which is what loading a model into the example gives
before anything is commanded) — and `World::drive_gains_look_generated` reports
the uniform ratio so a reader does not attribute the rigidity to the robot's own
tuning.

**Now covered** that the first pass listed as missing: `ctrlrange` clamping
(section 2.8, as the joint range), `forcerange` clamping (section 2.5, as
`maxForce`), and the direct-joint transmission with its ratio and efficiency
loss.

**Not ported** (each one is a gap in the port, not a claim of absence in the
reference):
- Activation dynamics: `mjDYN_INTEGRATOR` and `mjDYN_FILTER` `act_dot`,
  `actearly`, `mj_nextActivation` (section 2.3). Every drive here is stateless.
- `mj_transmission` beyond the direct joint case: tendon, site and slider-crank
  transmissions, so `moment' * force` is the identity on one degree of freedom.
- The actuator-delay / `mj_readCtrl` history buffer and the bad-control zeroing.
- `mj_actuatorDamping` as a separately-named force. A drive's damping **is**
  now folded into the implicit solve: [03](03-integrators.md)'s
  `Integrator::ImplicitFast` builds `qH = M + h diag(kd)` from the declared
  drive damping, so a stiffer gain set than the layer's is usable. What is not
  ported is the example's separate accounting of actuator damping as its own
  passive term.
- `mj_springdamper`: polynomial stiffness and damping (only the linear
  coefficient is used); free and ball joint springs.
- `mj_passive`: gravity compensation, fluid, contact-passive and adhesion.
- Muscle, DC-motor, PID, SO3 and wrapped-setpoint gains; tendon and flex forces.

**Goal**: Represent actuators as transmission × activation dynamics × gain/bias, and passive spring-damper, dry-friction and gravity-compensation forces. Tendons may be deferred. Covers old §7 items 1 and 4, and §8. Equality constraints (§7 items 2–3) are in step 06.

**Depends on**: [02](02-pure-dynamics-core.md), [03](03-integrators.md) (passive-force derivatives feed the implicit integrator).

**Reference formulation**: the governing algorithm is
[example 10](../../../docs/example-system-specs/10_actuation_sensors_plugins.md)
sections 1, 2.2, 2.5, 2.7 and 2.8 (`mj_transmission`, the gain/bias force loop,
the force clamp, `qfrc_actuator = moment' force`, and the control-input
semantics per gain type), per the precedence of
[Appendix D](appendix-d-mathematical-challenges.md) section 1.0.1. The pinned
C source below remains the secondary reference. Tag `3.17.0`, commit `800a109a4bdd02cb60404d565db4ae16fb04a53b` (Appendix A, entries 6 and 7). Functions transferred in Step 5, with evidence tag `code_inspected` (bodies read as text at the pin):
- `src/engine/engine_passive.c`: `mj_springdamper` (static; linear hinge/slide spring and dof damper, ported as `joint_spring_force` and `joint_damper_force`; `mju_polyForce`/`mjd_xPolyForce` in `src/engine/engine_util_misc.c` L2326/L2341 return the linear coefficient for a linear curve).
- `src/engine/engine_forward.c`: `mj_fwdActuation` (L364), `GAIN_AFFINE` and `BIAS_AFFINE` branches, ported as `affine_actuator_force`.

Located at the pin, not yet ported (the "Not ported" list above gives the missing terms of the two ported functions):
- `src/engine/engine_passive.c`: `mj_passive` (the non-spring-damper terms: gravity compensation, fluid, contact-passive and adhesion).
- `src/engine/engine_core_constraint.c`: `mj_instantiateFriction` (static; instantiates dry-friction constraints). the reference simulator treats friction loss as a constraint, not a passive force; it is handled by the solver in [step 06](06-equality-constraints-and-solver.md).
- `src/engine/engine_core_smooth.c`: `mj_transmission` (transmission to joint space).

Formula-level claims for the ported functions are `code_inspected`. The unported functions above have no formula claim in this step. `doc/computation.rst` §Actuation and §Passive forces and `mjtype.h` (entry 6) are the documentation sources.

**Purity contract**: The gain/bias and passive-force formulas use `std` and `f64` scalars and plain arrays only. No `Matrix`, `RigidPose` or `HashMap` in algorithm functions (FR-6). Profile validation may use the existing `InertiaTensor::is_physically_plausible` pattern at the crate edge.

**Inputs/outputs**: Input: actuator command, joint state, `ActuatorSpec` and `PdGains`, passive parameters. Output: joint-space generalized force and its velocity derivative (for the implicit integrator).

**Done when**: The affine gain/bias formula expresses both a position servo and a velocity servo as special cases, each with a hand-derived test (position servo: `gainprm = [kp,0,0]`, `biasprm = [0,-kp,-kv]`; velocity servo: `gainprm = [kv,0,0]`, `biasprm = [0,0,-kv]`, which gives `force = kv*(setpoint - velocity)`). The spring-damper sign-preservation check runs at profile validation. Passive-force velocity derivatives are available to step 03. **The model's own declared actuators drive a simulation run**, at the parameters the model declares and in the model's own units.

**Status against Done when**: the servo tests pass; the model's declared drives
run a simulation and are tested against the layer's own attributes. Still open:
the spring-damper sign-preservation check, and the passive-force velocity
derivatives [03](03-integrators.md) needs for an implicit step.

**Open questions**: none blocking. Tendon deferral is a scope choice recorded in old §7 item 4 below, not an open question. Per-tier actuator settings belong to [09](09-fidelity-tiers.md).

**Challenges**: [Appendix D](appendix-d-mathematical-challenges.md) confirms
`joint_spring_force`, `joint_damper_force` and `affine_actuator_force` correct as
written (section 1.1).

Closed since: **M-17** (the three formulas were unreachable; `World` now runs
them), **M-25** (the model's declared actuators were not simulated at all — the
stepper applied a hand-written hold with a uniform `kp = 300, kd = 30`, so a run
was a simulation of a different robot with the same geometry), and **M-26** (an
angular drive's stiffness and damping are per degree and were read as per
radian, understating every drive by 180/pi; with the wrong units the G1 could not
hold its own posture).

Open: **M-28** (**divergence**) — the declared gains are generated defaults, so
applying them is faithful to the asset but says nothing about the robot. Item 1's
re-examination of `ActuatorSpec`/`PdGains` against the example's factoring is now
*exercised* rather than merely implemented, which is what M-17 blocked; what it
exercises is a default gain set.

**M-03** (**defect**, owned by [02](02-pure-dynamics-core.md)) bore on item 2 and
is closed: `InertiaTensor::is_physically_plausible` now tests the triangle
inequality on principal moments, so the precedent item 2 names for the
spring-damper sign-preservation check is a sound one.

The sequence of M-25, M-26 and M-28 is the part worth carrying forward, and
Appendix D section 1.7.6 records it: a hand-written hold with invented gains
produced a *passing* stance while masking a unit error in the model import, and
the error only became visible once the model's own drives were applied. A
simulation that looks plausible under a controller of our own choosing is weak
evidence that the model is being read correctly.

---

## 7. Actuators, tendons, equality constraints

1. An actuator MUST be describable as the composition of three independent, orthogonal choices, matching the reference's own model exactly because it is the cleanest known factoring that still expresses a torque motor, a position servo and a velocity servo as the *same* underlying mechanism (an affine gain/bias function) rather than three special cases [the reference simulator §6]:
   - **Transmission**: how the actuator's scalar output maps into joint-space force (direct joint, tendon, site, slider-crank).
   - **Activation dynamics**: whether the actuator carries internal state between its command and its output (none, for a direct torque motor; a first-order filter, for a more realistic motor response).
   - **Gain/bias**: the function from (activation, length, velocity) to scalar force. An affine gain/bias (`force = gain·input + bias`, `bias = const + kp·length_error + kv·velocity`) already expresses both a position servo and a velocity servo as special cases of one formula — `robot-model`'s existing `ActuatorSpec`/`PdGains` should be re-examined against this factoring rather than extended ad hoc.
4. Tendons (spatial, with pulleys and geometric wrapping around obstacles; fixed, a linear combination of joint positions) MAY be deferred past the first GPU-batched milestone — no part of a humanoid locomotion/balance task strictly requires them — but the actuator factoring of item 1 MUST NOT preclude adding them later as another transmission type.

## 8. Passive forces

1. Joint/tendon spring-dampers (position- and velocity-dependent only — never dependent on control or acceleration, which is exactly why they are identical inputs to both forward and inverse dynamics [the reference simulator §7]), dry friction loss, and a per-body gravity-compensation term MUST all be representable.
2. A spring-damper's sign-preservation condition (for a polynomial stiffness/damping curve: non-negative leading coefficients, discriminant bound) SHOULD be checked at profile-validation time rather than left as an unenforced convention a bad parameter set can silently violate — the reference's own documentation flags this exact failure mode (silent energy injection) without preventing it [the reference simulator §7]; `RobotProfile::validate` already has a pattern (`InertiaTensor::is_physically_plausible`) this can follow.
3. Passive-force derivatives (with respect to velocity, specifically) MUST be available to whichever implicit integrator Section 4 requires — this is literally why the reference's own documentation frames implicit integration's primary motivation as "systems with substantial damping in tendons and actuators" [the reference simulator §5, §7], not contact stiffness.

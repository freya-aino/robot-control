# Step 05 — Passive forces and actuators

[Index](README.md) · [Appendix A](appendix-a-citations.md)

**Status**: Partially implemented (Step 5). `robot_sim::passive` ports the linear scalar-joint subset: `joint_spring_force` (`mj_springdamper`, hinge/slide, linear stiffness), `joint_damper_force` (`mj_springdamper`, dof damper, linear) and `affine_actuator_force` (`mj_fwdActuation`, `GAIN_AFFINE`/`BIAS_AFFINE`, `input = ctrl` only with a control). `robot-model` `ActuatorSpec` and `PdGains` are still not re-examined against this factoring (old §7 item 1).

**Not ported** (each one is a gap in the port, not a claim of absence in the reference simulator):
- `mj_fwdActuation`: `ctrlrange` clamping (`clampVec`) and `forcerange` clamping; the actuator_delay / `mj_readCtrl` history-buffer override; the bad-ctrl zeroing; activation dynamics (`mjDYN_INTEGRATOR`/`mjDYN_FILTER` `act_dot`, `actearly`); the transmission step `mj_transmission` (moment to `qfrc_actuator`), so only direct joint actuators are covered; actuator damping `mj_actuatorDamping`, which feeds both the damper and the `qH` term of `mj_EulerSkip`.
- `mj_springdamper`: polynomial stiffness and damping (`mju_polyForce`, only the linear coefficient is used); free and ball joint springs.
- `mj_passive`: gravity compensation, fluid, contact-passive and adhesion terms.
- Muscle, DC-motor, PID and wrapped-setpoint gains; tendon and flex forces.

**Goal**: Represent actuators as transmission × activation dynamics × gain/bias, and passive spring-damper, dry-friction and gravity-compensation forces. Tendons may be deferred. Covers old §7 items 1 and 4, and §8. Equality constraints (§7 items 2–3) are in step 06.

**Depends on**: [02](02-pure-dynamics-core.md), [03](03-integrators.md) (passive-force derivatives feed the implicit integrator).

**Reference formulation**: Tag `3.17.0`, commit `800a109a4bdd02cb60404d565db4ae16fb04a53b` (Appendix A, entries 6 and 7). Functions transferred in Step 5, with evidence tag `code_inspected` (bodies read as text at the pin):
- `src/engine/engine_passive.c`: `mj_springdamper` (static; linear hinge/slide spring and dof damper, ported as `joint_spring_force` and `joint_damper_force`; `mju_polyForce`/`mjd_xPolyForce` in `src/engine/engine_util_misc.c` L2326/L2341 return the linear coefficient for a linear curve).
- `src/engine/engine_forward.c`: `mj_fwdActuation` (L364), `GAIN_AFFINE` and `BIAS_AFFINE` branches, ported as `affine_actuator_force`.

Located at the pin, not yet ported (the "Not ported" list above gives the missing terms of the two ported functions):
- `src/engine/engine_passive.c`: `mj_passive` (the non-spring-damper terms: gravity compensation, fluid, contact-passive and adhesion).
- `src/engine/engine_core_constraint.c`: `mj_instantiateFriction` (static; instantiates dry-friction constraints). the reference simulator treats friction loss as a constraint, not a passive force; it is handled by the solver in [step 06](06-equality-constraints-and-solver.md).
- `src/engine/engine_core_smooth.c`: `mj_transmission` (transmission to joint space).

Formula-level claims for the ported functions are `code_inspected`. The unported functions above have no formula claim in this step. `doc/computation.rst` §Actuation and §Passive forces and `mjtype.h` (entry 6) are the documentation sources.

**Purity contract**: The gain/bias and passive-force formulas use `std` and `f64` scalars and plain arrays only. No `Matrix`, `RigidPose` or `HashMap` in algorithm functions (FR-6). Profile validation may use the existing `InertiaTensor::is_physically_plausible` pattern at the crate edge.

**Inputs/outputs**: Input: actuator command, joint state, `ActuatorSpec` and `PdGains`, passive parameters. Output: joint-space generalized force and its velocity derivative (for the implicit integrator).

**Done when**: The affine gain/bias formula expresses both a position servo and a velocity servo as special cases, each with a hand-derived test (position servo: `gainprm = [kp,0,0]`, `biasprm = [0,-kp,-kv]`; velocity servo: `gainprm = [kv,0,0]`, `biasprm = [0,0,-kv]`, which gives `force = kv*(setpoint - velocity)`). The spring-damper sign-preservation check runs at profile validation. Passive-force velocity derivatives are available to step 03.

**Step 5 status**: The position-servo and velocity-servo tests in `passive.rs` pass. The sign-preservation check and the passive-force velocity derivatives are not yet implemented.

**Open questions**: none blocking. Tendon deferral is a scope choice recorded in old §7 item 4 below, not an open question. Per-tier actuator settings belong to [09](09-fidelity-tiers.md).

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

# Step 01 — State representation and coordinates

[Index](README.md) · [Appendix A](appendix-a-citations.md) · [Appendix B](appendix-b-open-questions.md)

**Status**: **`Free` implemented in the example's convention; `Ball` deferred.**
`composite` stores a `Free` joint's velocity as **world-frame linear in the
first three degrees of freedom and body-frame angular in the last three,
translation first**, which is
[example 02](../../../docs/example-system-specs/02_kinematics_dynamics_smooth.md)
section 2.1's layout (`cdof[da+3] = cdof[da+10] = cdof[da+17] = 1`), with
`xanchor = xpos` for the free joint itself. World linear momentum is conserved
to below 1e-14 on a force-free body, against 3.93e-3 before — Appendix D M-01
closed, Appendix B S-07 resolved.

`crate::articulated` keeps its own layout, angular first and both halves
body-local, and the two are **not interchangeable**: a `q` or `qd` vector cannot
be passed between them without conversion, which `World::from_articulated_velocity`
exists to do deliberately. This is a specification-level fact about the crate,
not an implementation detail, because it determines what an observation vector
means — `WorldEnv` reports `cpu-world` and `MultibodyEnv` reports
`cpu-multibody` for exactly this reason, and a policy trained against one has
not been trained against the other.

A joint axis is normalised at one accessor (`unit_axis()`), so the velocity map
and the position map cannot disagree — Appendix D M-02 closed.

`Ball` is not implemented and is deferred (see Appendix B, S-04).

**Goal**: Track generalized coordinates with distinct position (`nq`) and velocity (`nv`) widths, including a floating base (`Free`) and, later, spherical joints (`Ball`). Covers old §3.

**Depends on**: [00](00-target-and-constraints.md).

**Reference formulation**: Tag `3.17.0`, commit `800a109a4bdd02cb60404d565db4ae16fb04a53b` (Appendix A, entry 1). Sources: `include/reference/mjtype.h` (joint-type enum) and `doc/overview.rst` (free-joint velocity convention). This step is a data-model step with no algorithm function, so no `src/engine/` function is pinned. Evidence tag: `code_inspected` for the joint-type enum and the velocity convention as cited in entry 1.

**Purity contract**: Applies to any new coordinate or joint algorithm code. Plain arrays and explicit formulas only (FR-6). The `robot_model` `Quat` type is kept at the crate boundary.

**Inputs/outputs**: Input: `robot_model` joint and profile types. Output: `nq`/`nv`-tracked state for the dynamics core in step 02.

**Done when**: `Free` joints carry 7 position values and 6 velocity values. The `Free` velocity convention matches the reference's documented asymmetry. Kinematic loops are rejected in the joint tree. If `Ball` is implemented, it carries 4 position and 3 velocity values; otherwise it stays deferred per Appendix B S-04.

**Open questions**: none blocking. `Ball` is optional under S-04 and does not gate this step. C-04 (canonical frame and quaternion ordering, `docs/decisions/0002`) is an existing project-level question and is not owned here.

**Challenges**: both resolved on 2026-11-01; see [Appendix D](appendix-d-mathematical-challenges.md) section 1.5. M-01 is fixed in `robot_sim::composite`, which carries the example's convention — world-frame linear velocity in the first three degrees of freedom, body-frame angular in the last three, translation first (02 §2.3) — with the exact exponential-map quaternion update of 01 §8.3. A force-free spinning body now conserves world linear momentum to below 1e-14, against the measured 3.93e-3 drift that opened the entry, and an isotropic one conserves kinetic energy to the same bound. `robot_sim::articulated` keeps its body-local convention internally and is still the stepper the CLI runs, so the fix is not yet the project-wide default; Appendix D M-17 tracks the switch-over, which waits on the constraint solver. M-02 is fixed by `JointSpec::unit_axis`, now the single accessor every consumer uses. The original entries, with their measurements, are below.

**Original challenges (for the record)**: [Appendix D](appendix-d-mathematical-challenges.md) M-01 and M-02, both **defect**. M-01: the implemented `Free` joint stores its linear velocity in the **body** frame, contradicting item 3's MUST and this step's own "Done when"; a force-free spinning free body consequently gains 0.39 % world linear velocity and 0.14 % kinetic energy per simulated second at `dt = 1e-4` (measured, with controls isolating the convention as the cause). The decision — change the representation or amend item 3 and record the deviation — is [Appendix B](appendix-b-open-questions.md) **S-07**. M-02: a declared joint axis that is not unit length makes the motion subspace disagree with the pose map, because `joint_transform` normalises and `motion_subspace_basis` does not; profile validation accepts it and the resulting trajectory is wrong rather than rescaled (measured on the point-mass pendulum).

---

## 3. State representation and coordinates

**Implementation status (2026-10-26, see Appendix B row S-04): item 1's `Free` joint is implemented** (`robot_model::joint::JointKind::Free`), `ArticulatedBody`'s ABA/RNE are generalized to it and validated (closed-form, conservation and RNE/ABA round-trip checks on an isolated free body, then the real G1 via `robot_model::fixtures::unitree_g1_29dof_floating_base`, which genuinely falls under gravity when nothing holds its joints — note that
`robot sim multibody --floating` now applies the model's own drives and so holds
a stance instead, and `--actuation zero` is the command that reproduces the
fall). `Ball` remains unimplemented and deferred. This status note does not relax the requirement text below, which continues to state the target this document specifies, not merely what exists.

1. `robot-model`'s `JointKind` enum (currently `Revolute | Continuous | Prismatic | Fixed`, each ≤1 DOF) MUST gain `Free` (6 DOF: 3 translation + 3 rotation, quaternion-valued orientation) and SHOULD gain `Ball` (3 DOF, quaternion-valued, no translation) to represent a floating base and spherical joints respectively. This is the single largest structural gap identified against the reference simulator [the reference simulator §1]: without it, `robot_sim::articulated::ArticulatedBody` cannot represent anything but a base rigidly welded to world, which is why the existing G1 simulation runs "bolted at the pelvis."
2. A profile's position-coordinate count (`nq`-equivalent) and velocity/DOF count (`nv`-equivalent) MUST be tracked as distinct quantities once `Free`/`Ball` joints exist, mirroring the reference's own `qpos`/`qvel` width mismatch for exactly the same reason (a unit quaternion needs 4 numbers to represent 3 rotational DOF) [the reference simulator §1]. `robot_types::frames::Quat` already exists and already forbids the bare four-element array the reference's own convention would otherwise invite (`SPECS.md` §7.2); the new joint types MUST use it, not a raw `[f64; 4]`.
3. A `Free` joint's velocity representation MUST follow the reference's own documented asymmetry — global-frame linear velocity, body-local-frame angular velocity — rather than inventing a different convention, since that asymmetry is described in the reference's own documentation as "correct use of the topology of quaternions," not an arbitrary choice [the reference simulator §1].
4. Kinematic loops remain disallowed in the joint tree; a closed loop MUST be modeled as an equality constraint (Section 7), never as an extra tree edge — this matches both the reference's own design [the reference simulator §1] and this project's existing tree-only `RobotProfile::validate`.
5. C-04 (canonical frames/gravity/quaternion ordering, `docs/decisions/0002`) remains open. Nothing in this section resolves it; `Free`/`Ball` joints carry their own local quaternion exactly as `Quat` already requires, independent of any project-wide frame convention.

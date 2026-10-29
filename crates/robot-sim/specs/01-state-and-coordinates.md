# Step 01 — State representation and coordinates

[Index](README.md) · [Appendix A](appendix-a-citations.md) · [Appendix B](appendix-b-open-questions.md)

**Status**: Partially implemented. The `Free` joint is implemented (`robot_model::joint::JointKind::Free`) and validated. `Ball` is not implemented and is deferred (see Appendix B, S-04).

**Goal**: Track generalized coordinates with distinct position (`nq`) and velocity (`nv`) widths, including a floating base (`Free`) and, later, spherical joints (`Ball`). Covers old §3.

**Depends on**: [00](00-target-and-constraints.md).

**MuJoCo baseline**: Tag `3.17.0`, commit `800a109a4bdd02cb60404d565db4ae16fb04a53b` (Appendix A, entry 1). Sources: `include/mujoco/mjtype.h` (joint-type enum) and `doc/overview.rst` (free-joint velocity convention). This step is a data-model step with no algorithm function, so no `src/engine/` function is pinned. Evidence tag: `code_inspected` for the joint-type enum and the velocity convention as cited in entry 1.

**Purity contract**: Applies to any new coordinate or joint algorithm code. Plain arrays and explicit formulas only (FR-6). The `robot_model` `Quat` type is kept at the crate boundary.

**Inputs/outputs**: Input: `robot_model` joint and profile types. Output: `nq`/`nv`-tracked state for the dynamics core in step 02.

**Done when**: `Free` joints carry 7 position values and 6 velocity values. The `Free` velocity convention matches MuJoCo's documented asymmetry. Kinematic loops are rejected in the joint tree. If `Ball` is implemented, it carries 4 position and 3 velocity values; otherwise it stays deferred per Appendix B S-04.

**Open questions**: none blocking. `Ball` is optional under S-04 and does not gate this step. C-04 (canonical frame and quaternion ordering, `docs/decisions/0002`) is an existing project-level question and is not owned here.

---

## 3. State representation and coordinates

**Implementation status (2026-10-26, see Appendix B row S-04): item 1's `Free` joint is implemented** (`robot_model::joint::JointKind::Free`), `ArticulatedBody`'s ABA/RNE are generalized to it and validated (closed-form, conservation and RNE/ABA round-trip checks on an isolated free body, then the real G1 via `robot_model::fixtures::unitree_g1_29dof_floating_base`, which genuinely falls under gravity through `robot sim multibody --floating`). `Ball` remains unimplemented and deferred. This status note does not relax the requirement text below, which continues to state the target this document specifies, not merely what exists.

1. `robot-model`'s `JointKind` enum (currently `Revolute | Continuous | Prismatic | Fixed`, each ≤1 DOF) MUST gain `Free` (6 DOF: 3 translation + 3 rotation, quaternion-valued orientation) and SHOULD gain `Ball` (3 DOF, quaternion-valued, no translation) to represent a floating base and spherical joints respectively. This is the single largest structural gap identified against MuJoCo [MuJoCo §1]: without it, `robot_sim::articulated::ArticulatedBody` cannot represent anything but a base rigidly welded to world, which is why the existing G1 simulation runs "bolted at the pelvis."
2. A profile's position-coordinate count (`nq`-equivalent) and velocity/DOF count (`nv`-equivalent) MUST be tracked as distinct quantities once `Free`/`Ball` joints exist, mirroring MuJoCo's own `qpos`/`qvel` width mismatch for exactly the same reason (a unit quaternion needs 4 numbers to represent 3 rotational DOF) [MuJoCo §1]. `robot_types::frames::Quat` already exists and already forbids the bare four-element array MuJoCo's own convention would otherwise invite (`SPECS.md` §7.2); the new joint types MUST use it, not a raw `[f64; 4]`.
3. A `Free` joint's velocity representation MUST follow MuJoCo's own documented asymmetry — global-frame linear velocity, body-local-frame angular velocity — rather than inventing a different convention, since that asymmetry is described in MuJoCo's own documentation as "correct use of the topology of quaternions," not an arbitrary choice [MuJoCo §1].
4. Kinematic loops remain disallowed in the joint tree; a closed loop MUST be modeled as an equality constraint (Section 7), never as an extra tree edge — this matches both MuJoCo's own design [MuJoCo §1] and this project's existing tree-only `RobotProfile::validate`.
5. C-04 (canonical frames/gravity/quaternion ordering, `docs/decisions/0002`) remains open. Nothing in this section resolves it; `Free`/`Ball` joints carry their own local quaternion exactly as `Quat` already requires, independent of any project-wide frame convention.

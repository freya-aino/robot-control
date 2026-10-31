# Step 07 — Collision detection pipeline

[Index](README.md) · [Appendix A](appendix-a-citations.md)

**Status**: Partially implemented (primitive subset only). `robot_sim::contact::plane_sphere_contact` handles plane-sphere geometry with a fixed `+z` plane. The broad/mid/narrow collision pipeline is not implemented; the single-point ground-plane model in `robot_sim::contact::ContactBody` remains a separate benchmark.

**Goal**: A multi-stage pipeline (broad phase, mid phase, narrow phase dispatch), exact primitive routines, a convex fallback via GJK/EPA, body-pair exclusion, and per-geometry-pair margin and gap. Covers old §6 in full.

**Depends on**: [01](01-state-and-coordinates.md) (body and joint topology for pair exclusion).

**Reference formulation**: Tag `3.17.0`, commit `800a109a4bdd02cb60404d565db4ae16fb04a53b` (Appendix A, entry 4). Function definitions, located by name at the pin:
- `src/engine/engine_collision_driver.c`: `mj_collision` (pipeline entry point). `mj_narrowphase`, `mj_setContact` and `mj_collideOBB` are static or helper functions in the same file.

Evidence tag: `unverified` for every stage-level and dispatch-level claim in this step. The function definition is located at the pin, but its body was not read in Step 1. Convex hulls are built at compile time (entry 9, see [08](08-model-compilation-and-sensors.md)).

**Purity contract**: Primitive and GJK/EPA routines use `std` and plain `[f64; 3]` arrays with explicit formulas. No `Matrix`, `RigidPose` or `HashMap` in algorithm functions (FR-6).

**Inputs/outputs**: Input: geometry set, body poses, margin and gap per pair. Output: a contact list with depth, normal and position.

**Done when**: Broad, mid and narrow phases exist as separate stages. Each primitive shape has an exact narrow-phase routine. The convex fallback uses GJK/EPA. Adjacent-link self-collision needs no manual exclusion list.

**Open questions**: none blocking. SDF collision is a non-goal ([00](00-target-and-constraints.md), Section 1.6).

---

## 6. Collision detection pipeline

1. A multi-stage pipeline MUST exist: broad-phase candidate-pair generation, a mid-phase per-pair bounding-volume test, and a narrow-phase dispatch keyed by geometry-type pair — mirroring the reference's own sweep-and-prune-then-BVH-then-dispatch-table structure [the reference simulator §4]. `robot_sim::contact::ContactBody`'s existing single hard-coded point-vs-ground-plane model satisfies none of this and remains a separate, smaller benchmark (Section 1.6 does not retire it).
2. Primitive collision shapes (sphere, box, capsule, cylinder, plane at minimum) MUST each have an exact narrow-phase routine; general convex shapes (meshes, replaced by their convex hull for collision purposes, matching the reference's own `qhull`-based approach [the reference simulator §4] and §9 below) MUST fall back to a generic convex-convex algorithm — GJK for distance/separation, EPA for penetration depth, matching the reference's own native pipeline [the reference simulator §4] — rather than a bespoke per-pair routine for every possible shape combination.
3. A body-pair exclusion mechanism equivalent to `contype`/`conaffinity` bitmasks and parent-child/same-body exclusion MUST exist, so adjacent-link self-collision (e.g. a URDF-imported humanoid's own forearm and upper arm) does not require per-model manual exclusion lists.
4. Collision margin (a surface-inflation distance below which contacts are detected but only generate force below a tighter threshold) and gap (detection-only, for adhesion-style effects) SHOULD both be supported as per-geometry-pair parameters, matching the reference's own [the reference simulator §4].
5. Signed-distance-field collision (non-convex shapes via SDF gradient descent, the reference simulator 3.x) is an explicit non-goal (Section 1.6).

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

**Challenges**: [Appendix D](appendix-d-mathematical-challenges.md) confirms the support maps of all four primitives, Ericson's closest-point-on-triangle, and the plane-sphere rule correct as written (section 1.1). Six challenges sit above them:

- M-04 (**defect**): `plane_contact_points` samples a shape into points (eight box corners, `2 * RIM_SAMPLES` cylinder rim points) and the penalty law is applied to each independently, so a body resting on `n` co-planar points carries `n` springs in parallel. Its rest penetration is `m g / (n k)`, its contact stiffness is `n k`, and changing `RIM_SAMPLES` changes the physics with no parameter change. The coupled model does not remove the `1/n`. The exact bounded manifolds of the example (`04` §1.3, §1.4), already listed as candidate work in `docs/example-system-mapping.md` §2.6, are the fix; this entry supplies the reason it is a correctness matter rather than an accuracy one.
- M-05 (**defect**): `complete_simplex` grows the EPA seed simplex to affine independence but never to origin **enclosure**, so face distances can be negative, the closest-face selection picks the wrong face, and `depth.max(0.0)` turns the failure into a zero depth with an arbitrary normal. `epa` also exhausts its 128-iteration budget without a convergence flag.
- M-06 (**defect**): `reduce`'s containment test is `s_origin * s_opp < 0.0`, so a zero product leaves `inside` at `true`. A flat tetrahedron zeroes all four products and is reported as containing the origin — a false overlap, reachable by construction in the symmetric face-parallel configurations a humanoid on a plane produces constantly. M-05 and M-06 are two halves of one weakness.
- M-07 (**defect**): `box_box_contacts`' edge-edge fallback places the contact at the deepest **corner** of each box rather than on the crossing edges, so the depth is right and the moment arm is wrong.
- M-15 (**gap**): item 1 is a MUST for three separate stages; `contact_rows` has one. Every shape pair goes straight to `collision::proximity` (full GJK, EPA on overlap) with no bounding test. `Placed::bounding_sphere` is written, exact for all four primitives, and called from nowhere in the workspace — so the mid-phase test this item requires already exists and is merely unwired.
- M-16 (**gap**): item 3's exclusion mechanism is stood in for by `ADJACENT_OVERLAP_ALLOWANCE = 0.003` m subtracted from parent-child depths. One global length, no per-pair or per-shape mask, and no allowance at all for the sibling and grandparent pairs a shoulder or hip produces.

Appendix D section 3.2 also records that `complete_simplex`, `box_sat` and `box_box_contacts` set their degeneracy thresholds as absolute lengths, squared lengths and volumes, which embeds a metre-scale assumption.

---

## 6. Collision detection pipeline

1. A multi-stage pipeline MUST exist: broad-phase candidate-pair generation, a mid-phase per-pair bounding-volume test, and a narrow-phase dispatch keyed by geometry-type pair — mirroring the reference's own sweep-and-prune-then-BVH-then-dispatch-table structure [the reference simulator §4]. `robot_sim::contact::ContactBody`'s existing single hard-coded point-vs-ground-plane model satisfies none of this and remains a separate, smaller benchmark (Section 1.6 does not retire it).
2. Primitive collision shapes (sphere, box, capsule, cylinder, plane at minimum) MUST each have an exact narrow-phase routine; general convex shapes (meshes, replaced by their convex hull for collision purposes, matching the reference's own `qhull`-based approach [the reference simulator §4] and §9 below) MUST fall back to a generic convex-convex algorithm — GJK for distance/separation, EPA for penetration depth, matching the reference's own native pipeline [the reference simulator §4] — rather than a bespoke per-pair routine for every possible shape combination.
3. A body-pair exclusion mechanism equivalent to `contype`/`conaffinity` bitmasks and parent-child/same-body exclusion MUST exist, so adjacent-link self-collision (e.g. a URDF-imported humanoid's own forearm and upper arm) does not require per-model manual exclusion lists.
4. Collision margin (a surface-inflation distance below which contacts are detected but only generate force below a tighter threshold) and gap (detection-only, for adhesion-style effects) SHOULD both be supported as per-geometry-pair parameters, matching the reference's own [the reference simulator §4].
5. Signed-distance-field collision (non-convex shapes via SDF gradient descent, the reference simulator 3.x) is an explicit non-goal (Section 1.6).

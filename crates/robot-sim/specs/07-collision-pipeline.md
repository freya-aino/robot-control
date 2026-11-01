# Step 07 — Collision detection pipeline

[Index](README.md) · [Appendix A](appendix-a-citations.md)

**Status**: **Implemented, including mesh collision.** Items 1 to 4 are met
except the mid-phase bounding-volume tree. The G1 reports 33 collision shapes
modelled and 0 not modelled, which is exactly what its layer declares; Appendix
D M-27 is resolved.

What runs:

- **Broad phase**: bounding-sphere then world-AABB rejection
  ([example 03](../../../docs/example-system-specs/03_collision_driver_broadphase.md)
  sections 2.2 and 2.4).
- **Pair exclusion**: the example's own filters rather than a length allowance —
  `contype`/`conaffinity` bitmask, weld body, immovable pair and parent-child
  (example 03 sections 2.1 and 2.5).
- **Narrow phase**: the signed-volume GJK of
  [example 05](../../../docs/example-system-specs/05_collision_convex_gjk_epa.md)
  with the S1D/S2D/S3D subdistance of Montanari et al., and EPA with the
  polytope2, polytope3 and polytope4 constructions of section 6.1. Four non-zero
  barycentric weights is strict origin enclosure, which is EPA's precondition —
  the enclosure invariant comes from GJK's subdistance, not from EPA.
- **Plane contact**: the exact bounded manifolds of
  [example 04](../../../docs/example-system-specs/04_collision_primitives.md)
  sections 1.3 and 1.4 — a box's four lowest corners, a cylinder's four-point
  rule — instead of sampling a shape into points.
- **Margin and gap** per run (item 4), through `options::SimOptions`.

- **Mesh colliders**: reduced to their convex hull at import by
  `robot_model::hull`, and contacted through the same support query every other
  primitive answers — `argmax_i (v_i . d)`, which is `mjc_meshSupport`
  (example 04 section 1.5). `collision::Primitive::Hull` borrows the vertices
  rather than owning them, so a hull costs no allocation per query.

Still absent, and each a cost rather than a wrong answer:

- The **mid-phase bounding-volume tree** (item 1's third stage).
- **Hill-climbing support** (`mjc_hillclimbSupport`). The example scans
  exhaustively below ten vertices and walks a vertex adjacency graph above it;
  this implementation scans at every size. The pelvis hull is 5583 vertices, so
  the scan is the dominant narrow-phase cost on a mesh pair. A 2-second G1
  stance at 0.5 ms runs in 8.5 s of wall clock today, so it is affordable and
  it is the first thing that will stop being so.
- **The three extra plane-hull contacts** of `mjc_PlaneConvex`, which need the
  mesh's polygon topology. A hull reduced to vertices gets the one contact the
  example's own no-polygon branch returns, so a hull lying flat on the plane
  cannot resist tipping about it.

**Retired, not kept as an option.** Item 1 said the single hard-coded
point-vs-plane penalty model "remains a separate, smaller benchmark". Under the
MUST precedence of
[Appendix D](appendix-d-mathematical-challenges.md) section 1.0.2 it does not:
its stiffness scaled with the number of sampled points (M-04) and its impedance
was referred to a fixed 1 kg reference mass (M-09), so it is a competing
formulation rather than a cheaper setting of the example's contact model. No
command reaches it. `contact::plane_sphere_contact` was also overwritten — it
fixed the normal to `+z`, where example 04 section 1.1 takes it from the plane
geom's own frame — and now takes the plane pose and normal.

**Goal**: A multi-stage pipeline (broad phase, mid phase, narrow phase dispatch), exact primitive routines, a convex fallback via GJK/EPA, body-pair exclusion, and per-geometry-pair margin and gap. Covers old §6 in full.

**Depends on**: [01](01-state-and-coordinates.md) (body and joint topology for pair exclusion).

**Reference formulation**: Tag `3.17.0`, commit `800a109a4bdd02cb60404d565db4ae16fb04a53b` (Appendix A, entry 4). Function definitions, located by name at the pin:
- `src/engine/engine_collision_driver.c`: `mj_collision` (pipeline entry point). `mj_narrowphase`, `mj_setContact` and `mj_collideOBB` are static or helper functions in the same file.

Evidence tag: `unverified` for every stage-level and dispatch-level claim in this step. The function definition is located at the pin, but its body was not read in Step 1. Convex hulls are built at compile time (entry 9, see [08](08-model-compilation-and-sensors.md)).

**Purity contract**: Primitive and GJK/EPA routines use `std` and plain `[f64; 3]` arrays with explicit formulas. No `Matrix`, `RigidPose` or `HashMap` in algorithm functions (FR-6).

**Inputs/outputs**: Input: geometry set, body poses, margin and gap per pair. Output: a contact list with depth, normal and position.

**Done when**: Broad, mid and narrow phases exist as separate stages. Each primitive shape has an exact narrow-phase routine. The convex fallback uses GJK/EPA. Adjacent-link self-collision needs no manual exclusion list.

**Open questions**: none blocking. SDF collision is a non-goal ([00](00-target-and-constraints.md), Section 1.6).

**Challenges**: [Appendix D](appendix-d-mathematical-challenges.md) confirms the
support maps of all four primitives, Ericson's closest-point-on-triangle and the
plane-sphere rule correct as written (section 1.1).

**M-04 to M-07, M-15 and M-16 are closed**: the sampled penalty manifold whose
stiffness scaled with the sample count, the EPA seed simplex that was not
required to enclose the origin, the degenerate tetrahedron reported as
containing it, the box-box edge-edge contact placed at a corner, the absent
broad phase, and the fixed `ADJACENT_OVERLAP_ALLOWANCE` standing in for an
exclusion mechanism.

**M-27 is resolved** (details in Appendix D). It was found by
flattening the G1 layer and counting colliders *through the instance
prototypes*, since the model instances its collision geometry and a naive count
reads prototypes rather than instances. Every prototype is instanced exactly
once, so the scene has **33 colliders: 15 Mesh, 10 Sphere, 6 Cylinder, 2 Cube**,
and the two cubes are the feet.

The import produces 35 shapes: the 18 primitives, all correct; the 15 meshes,
kept as references and **never collided**, so the arms, head, thighs and shins
have no contact geometry at all; and **2 axis-aligned bounding boxes the model
does not declare**, fitted to the pelvis and torso collision meshes by
`usd::PROXY_LINKS`.

The proxies are deliberate and documented — this step has no mesh support, so a
link described only by mesh would otherwise have none — but a bounding box is
not the shape. It strictly *encloses* the mesh, where every other simulator
reading this layer collides its convex hull. A torso modelled as its own AABB is
a different robot from the waist up, and it is why the pelvis has any contact
geometry in these runs.

It was closed by reducing each mesh to its hull at import and answering the
support query over the hull's vertices. The reduction is exact, not an
approximation, because the support function of a point set equals the support
function of its convex hull.

Appendix D section 3.2 also records that several degeneracy thresholds were set
as absolute lengths, squared lengths and volumes, embedding a metre-scale
assumption; the ones on the GJK/EPA path are now relative.

---

## 6. Collision detection pipeline

1. A multi-stage pipeline MUST exist: broad-phase candidate-pair generation, a mid-phase per-pair bounding-volume test, and a narrow-phase dispatch keyed by geometry-type pair — mirroring the reference's own sweep-and-prune-then-BVH-then-dispatch-table structure [the reference simulator §4]. `robot_sim::contact::ContactBody`'s existing single hard-coded point-vs-ground-plane model satisfies none of this and remains a separate, smaller benchmark (Section 1.6 does not retire it).
2. Primitive collision shapes (sphere, box, capsule, cylinder, plane at minimum) MUST each have an exact narrow-phase routine; general convex shapes (meshes, replaced by their convex hull for collision purposes, matching the reference's own `qhull`-based approach [the reference simulator §4] and §9 below) MUST fall back to a generic convex-convex algorithm — GJK for distance/separation, EPA for penetration depth, matching the reference's own native pipeline [the reference simulator §4] — rather than a bespoke per-pair routine for every possible shape combination.
3. A body-pair exclusion mechanism equivalent to `contype`/`conaffinity` bitmasks and parent-child/same-body exclusion MUST exist, so adjacent-link self-collision (e.g. a URDF-imported humanoid's own forearm and upper arm) does not require per-model manual exclusion lists.
4. Collision margin (a surface-inflation distance below which contacts are detected but only generate force below a tighter threshold) and gap (detection-only, for adhesion-style effects) SHOULD both be supported as per-geometry-pair parameters, matching the reference's own [the reference simulator §4].
5. Signed-distance-field collision (non-convex shapes via SDF gradient descent, the reference simulator 3.x) is an explicit non-goal (Section 1.6).

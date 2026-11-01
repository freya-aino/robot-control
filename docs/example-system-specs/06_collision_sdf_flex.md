# 06 — SDF and Deformable (Flex) Collision

**Scope:** signed-distance-field (SDF) narrowphase (geom–SDF, SDF–SDF, mesh–SDF, flex–SDF), octree-interpolated SDFs, analytic SDF plugins, and flex collision primitives (plane–flex, geom–element, element–element, self-collision filters for interpolated flexes). The driver logic that *dispatches* flex pairs (midphase modes, filter groups, FPS reduction) is documented here as far as it is specific to flex/SDF; generic broadphase/midphase/contact-parameter mixing is in the collision-driver spec.

**Source files covered**

| Layer | File | Notes |
|---|---|---|
| C engine | `src/engine/engine_collision_sdf.c` (1399 lines) | octree SDF, primitive SDFs, gradient descent, Frank–Wolfe, mesh/flex BVH traversal, `mjc_SDF`, `mjc_MeshSDF`, `mjc_FlexSDF`, `mjc_HFieldSDF` |
| C engine | `src/engine/engine_collision_flex.c` (510 lines) | `mjc_PlaneFlex`, `mjc_GeomElem`, `mjc_ElemElem`, `sphereTriangle`, `boxTriangle`, `capsuleTriangle`, `makeCapsule`, interpolation-grid exclusion (`mjc_vertCellRange`, `mjc_cellRangeShared`, `mjc_ElemSharedNode`) |
| C engine (context) | `engine_collision_driver.c` (`mj_collision`, `mj_collideTree`, `collisionTask`, `pairMaxContact`, `filterPreContacts`, `mj_collideFlexSAPPairs`, `mj_collideFlexCellPairs`, `addPairContacts`), `engine_collision_convex.c` (`mjc_ConvexElem`, `mjc_flexSupport`, `mjc_fixNormal`, `mjc_HFieldElem`), `engine_collision_primitive.c` (`mjc_triWeights`), `engine_core_smooth.c` (flex element AABBs, dynamic BVH), `engine_util_misc.c` (`mju_Halton`) | only parts relevant here |
| Compiler (context) | `src/user/user_objects.cc` (`mjCOctree::*`), `src/user/user_mesh.cc` (plugin SDF → marching cubes mesh, octree build) | octree construction |
| Plugins | `plugin/sdf/{sdf,torus,bowl,gear,nut,bolt,register}.cc/.h` | analytic SDFs |
| MuJoCo Warp | `mujoco_warp/_src/collision_sdf.py` (1121 lines), `collision_flex.py` (3626 lines), plus `collision_driver.py`, `io.py`, `collision_primitive_core.py` (triangle primitives), `collision_gjk.py` (support for TRIANGLE/FLEX), `test_data/collision_sdf/*.py` (Warp ports of plugins) | |
| MJX | `mjx/_src/collision_sdf.py` (275 lines) | SDF-style optimization used for ellipsoid/cylinder primitive pairs only |

**Role in the pipeline.** Called from `mj_collision` (position stage, after `mj_kinematics`/`mj_comPos`/`mj_flex` have produced `geom_xpos/xmat`, `flexvert_xpos`, `flexelem_aabb`, `bvh_aabb_dyn`). The collision driver forms candidate pairs (`mjcPair` of type GEOM_GEOM / GEOM_FLEX / GEOM_ELEM / ELEM_ELEM); for each pair the functions in this spec fill an `mjPreContact` buffer (`dist`, `pos`, `normal`, `tangent`), optionally with per-contact flex element/vertex ids. The driver then (optionally) reduces contacts per filter group with farthest-point sampling and converts them to `mjContact` (`addPairContacts`/`mj_setContact`). Contact normal convention everywhere: **from object 1 (geom[0]/flex[0]) to object 2**, `dist < 0` = penetration, `pos` = midpoint between surfaces.

---

## 0. Constants, options and data fields

| Name | Value | Where |
|---|---|---|
| `mjMINVAL` | `1e-15` (double), `1e-15f` (single) | mjtype.h |
| `mjMAXVAL` | `1e10` | mjtype.h |
| `mjMAXCONPAIR` | `50` | mjmodel.h (max contacts per geom pair) |
| `m->opt.sdf_initpoints` | default `40` | engine_init.c (`<option sdf_initpoints>`) |
| `m->opt.sdf_iterations` | default `10` | engine_init.c (`<option sdf_iterations>`) |
| `mesh.octree_maxdepth` | default `6` | user_init.c / `mjCOctree::max_depth_` |
| octree Laplacian smoothing | `smoothing_iterations_ = 0` (disabled), `alpha = 0.2` | user_objects.h/.cc |
| `stepGradient` line search | `c = 0.1`, `rho = 0.5`, `amin = 1e-4`, initial `alpha = 2` | engine_collision_sdf.c |
| `findOct` | `eps = 1e-8` (double) / `1e-5` (`mjUSESINGLE`), `niter = 100` | |
| `boxProjection` | `eps = 1e-6` | |
| `oct_gradient` exterior FD step | `eps = 1e-8` (forward difference) | |
| `triangleIntersect` | `kDistanceScl = 10`, `kMinHeight = 0.1` | |
| `traverseBVH` stack | fixed `int stack[64]` | |
| plugin FD gradient step (bolt/bowl/gear/nut) | `eps = 1e-8` forward difference | plugin/sdf |
| `mjc_vertCellRange` eps | `1e-6 * cellnum[i]` (in cell-index units) | |
| SAP pair hard limit (flex self) | `min(nactive*(nactive-1)/2, 1000000)` | driver |

Relevant enums:

```
mjtSDFType: mjSDFTYPE_SINGLE=0  (A)
            mjSDFTYPE_INTERSECTION (max(A,B))
            mjSDFTYPE_MIDSURFACE   (A-B)
            mjSDFTYPE_COLLISION    (A+B+|max(A,B)|)
mjtFlexSelf: NONE=0, NARROW, BVH, SAP, AUTO
```

`mjSDF` (mjplugin.h): `{const mjpPlugin** plugin; int* id; mjtSDFType type; mjtNum* relpos; mjtNum* relmat; mjtGeom* geomtype;}`. Index 0 is always "the SDF being queried at x"; index 1 is "the other object queried at y = relmat·x + relpos".

Plugin SDF interface (`mjpPlugin`, capability `mjPLUGIN_SDF`):
- `mjtNum sdf_distance(const mjtNum p[3], const mjData* d, int instance)` — runtime distance in the geom-local frame.
- `void sdf_gradient(mjtNum g[3], const mjtNum p[3], const mjData* d, int instance)`.
- `mjtNum sdf_staticdistance(const mjtNum p[3], const mjtNum* attributes)` — compile time (marching cubes).
- `void sdf_attribute(mjtNum attr[], const char* name[], const char* value[])` — parse/default attributes.
- `void sdf_aabb(mjtNum aabb[6], const mjtNum* attributes)` — `(center[3], half-size[3])`.
- `reset(...)` and `compute(m,d,instance,mjPLUGIN_SDF)` are called by `mjc_SDF` purely for the iteration visualizer.

Model/Data fields read:
- Geoms: `geom_type, geom_size, geom_aabb (6/geom: local center, half size), geom_dataid, geom_plugin, geom_bodyid, geom_rbound, geom_margin/gap`; `d->geom_xpos, d->geom_xmat`.
- Octree: `mesh_octadr[nmesh]` (−1 = none), `mesh_octnum`, `oct_child[8*noct]` (child index relative to the mesh's first node, −1 = none), `oct_aabb[6*noct]` (center, half size), `oct_coeff[8*noct]` (SDF values at the 8 node corners), `oct_depth`.
- Mesh: `mesh_vert` (**float32** in mjModel), `mesh_face`, `mesh_vertadr`, `mesh_faceadr`, `mesh_bvhadr`, `bvh_aabb` (static BVH, mesh frame), `bvh_child`, `bvh_nodeid`; `d->bvh_active` (visualization).
- Flex: `flex_dim, flex_radius, flex_vertadr, flex_vertnum, flex_elemadr, flex_elemnum, flex_elemdataadr, flex_elem, flex_vertbodyid, flex_bvhadr, flex_bvhnum, flex_interp, flex_rigid, flex_selfcollide, flex_cellnum[3], flex_vertcell, flex_vert0 (normalized [0,1]^3 reference coords), flex_elemlayer, flex_activelayers, flex_margin, flex_gap, flex_contype/conaffinity`; `d->flexvert_xpos, d->flexelem_aabb (6/elem: center, half size incl. radius), d->bvh_aabb_dyn` (indexed `bvhadr - m->nbvhstatic`).

---

## 1. Octree (interpolated) SDF — `engine_collision_sdf.c`

### 1.1 Compile-time construction (user_objects.cc / user_mesh.cc)

Built when `mesh.needsdf` (or internal `needsdf_`, e.g. tactile sensors, SDF geoms) is set (`mjCMesh::Process`, "make octree"):

1. `octree_.SetFace(dvert, face_)`; `SetMaxDepth(spec.octree_maxdepth)`; `CreateOctree(aamm_)`.
2. `CreateOctree(aamm)`: `aabb = (center, half)` of mesh AAMM; root box = `center ± 1.1*half` (10% enlargement). Triangles are transformed into the octree frame `R(iquat)^T (v - ipos)` (body-inertial/geom frame, i.e. the frame used at runtime in `geom_xpos/xmat`).
3. `MakeOctree`: BFS queue. For each node keep triangles intersecting its box (`boxTriangle`: Schwarz–Seidel triangle/box overlap test: per-axis AABB rejection, plane–box test, 9 projected edge-normal tests). Skip subdivision if no triangle or `lev >= max_depth`. `Subdivide` creates 8 children; child slot bit 0→x, bit 1→y, bit 2→z (`slot & 1` ⇒ upper half in x, etc.). Corner vertices are deduplicated in a hash map `vert_map` (exact coordinate match). Corner index `j` of a node: bit0 → x max, bit1 → y max, bit2 → z max.
4. `BalanceOctree` (2:1 balancing via neighbor search) and `MarkHangingNodes` (vertices lying on a coarser neighbour's face/edge get a dependency list `Hang(v)`).
5. `ComputeSdfCoeffs` (only if the mesh is not a plugin mesh): BFS over nodes; for each unprocessed corner vertex: if not hanging, `coeff = tree.QuerySignedDistance(p_mesh, …, MeshPseudonormals)` with `p_mesh = R(iquat)·v + ipos` (signed distance to the triangle mesh via its BVH, sign from angle-weighted pseudonormals); if hanging, `coeff = mean(coeff[dependencies])` (parents always processed first due to BFS). Optional Laplacian smoothing `c_i ← (1-0.2)c_i + 0.2·mean(neighbors)` over the 12 cube edges per node, `smoothing_iterations_` times (default 0). Finally node `i` stores `oct_coeff[8i+j] = coeff[VertId(i,j)]`.
6. Copied into `m->oct_aabb`, `oct_child`, `oct_depth`, `oct_coeff`; `mesh_octadr[i] = oct_adr` or −1.

Plugin SDF meshes (`<mesh><plugin/>`): `sdf_attribute` fills attributes (NaN ⇒ compile error), `sdf_aabb` gives the box, a grid with `nx = floor(300/total*aabb[3])+1` etc. (`total = aabb[3]+aabb[4]+aabb[5]`) is sampled with `sdf_staticdistance` at `aabb[c]-aabb[3+c] + 2*aabb[3+c]*i/(n-1)` and triangulated with marching cubes (visual/inertia/convex hull mesh). The octree is still built but coefficients are not computed (plugin evaluated at runtime).

### 1.2 `boxProjection(point[3], box[6]) -> mjtNum` (modifies `point`)

```
r = point - box.center ; q = |r| - box.half   (componentwise)
if q0<=0 && q1<=0 && q2<=0: return max(q0,q1,q2)      # inside, point unchanged
dist_sqr = 0
for k in 0..2: if q_k >= 0:
    dist_sqr += q_k^2
    point_k -= (r_k > 0 ? +1 : -1) * (q_k + 1e-6)       # project slightly inside
return sqrt(dist_sqr)
```
Note: projection is per axis with `q_k >= 0` (a coordinate exactly on the face is pushed 1e-6 inside).

### 1.3 `findOct(w[8], dw[8][3], oct_aabb, oct_child, p) -> node` (static)

Descends from node 0 (pointers already offset by `octadr`), at most `niter = 100` steps:
```
node = stack
vmin = c - h ; vmax = c + h
if p outside [vmin-eps, vmax+eps] on any axis: continue   # NOTE: does not change `stack` -> loops until niter exhausted, then mju_error("Node not found")
coord = (p - vmin) / (vmax - vmin)                          # componentwise
if all 8 children == -1:                                    # leaf
    inv_cell = 1/(vmax - vmin)
    for j in 0..7:
       sx = (j&1 ? coord0 : 1-coord0); sy = (j&2 ? coord1 : 1-coord1); sz = (j&4 ? coord2 : 1-coord2)
       w[j] = sx*sy*sz                                      (if w)
       dw[j] = ( (j&1?1:-1)*sy*sz*inv_cell0,
                 sx*(j&2?1:-1)*sz*inv_cell1,
                 sx*sy*(j&4?1:-1)*inv_cell2 )                (if dw)
    return node
x = coord0 < .5 ? 0 : 1 (same y,z) ; stack = oct_child[8*node + 4z + 2y + x]
```
`eps = 1e-8` (double) / `1e-5` (single). `node == -1` → `mju_error("Invalid node number")`. Trilinear weights; `coord` is *not* clamped (points within eps outside a leaf extrapolate slightly).

### 1.4 `oct_distance(m, p, meshid)`

```
point = p ; boxDist = boxProjection(point, oct_aabb[root])   # root aabb of this mesh
node = findOct(w, NULL, ..., point)
sdf = Σ_{i=0..7} w[i] * oct_coeff[8*node+i]                  # summed in order i=0..7
return boxDist > 0 ? sdf + boxDist : sdf
```
Outside the root box: value at the projected point plus the Euclidean distance to the box (a lower-quality but continuous extrapolation). `octadr == -1` → `mjERROR`.

### 1.5 `oct_gradient(m, grad, point, meshid)`

- If `boxProjection(p_copy, root) <= 0` (inside root box): analytic trilinear gradient `grad = Σ dw[i] * coeff[i]` (component sums in order i=0..7). Discontinuous across cell faces.
- Else: forward finite differences with `eps = 1e-8`: `grad_k = (oct_distance(point + eps e_k) - oct_distance(point)) / eps`. (Effectively the box-distance gradient plus the tangential variation of the projected value.)

---

## 2. Primitive SDFs and gradients (geom-local frame)

`geomDistance(m, d, p, i, x, type)` / `geomGradient(...)`. `size = m->geom_size + 3*i` (only meaningful when `i` is a geom id). References: Inigo Quilez distance functions.

| type | distance `f(x)` | gradient |
|---|---|---|
| PLANE | `x2` | `(0,0,1)` |
| SPHERE | `|x| - r` | `x/|x|` (no zero guard: NaN at origin; computed as `x * (1/c)`) |
| BOX | `a = |x| - s`. If any `a_k >= 0`: `|max(a,0)| + min(max(a0,a1,a2),0)`. Else (strictly inside): `b = radialField3d(a,x,s)`, `t_k = -a_k/|b_k|`, `f = -min(t) * |b|` (|b| = 1) | inside (`max a < 0`, max index chosen by `k = a0>a1?0:1; l = a2>a[k]?2:k`): `radialField3d`. Outside: `g_k = a_k>0 ? b_k/|b| * x_k/|x_k| : 0` with `b = max(a,0)` |
| CAPSULE | `|(x0, x1, x2 - clip(x2,-h,h))| - r` | that vector / its norm (NaN on the segment) |
| ELLIPSOID | `a = x/s`, `b = a/s`, `k0=|a|`, `k1=|b|`, `f = k0(k0-1)/k1` | `gk0 = b/k0`, `gk1 = b/(k1 s²)`, `df/dk0 = (2k0-1)/k1`, `df/dk1 = k0(k0-1)/k1²`, `g = gk0·df/dk0 - gk1·df/dk1`, then `mju_normalize3(g)` |
| CYLINDER | `a0 = sqrt(x0²+x1²) - r`, `a1 = |x2| - h`; `f = min(max(a0,a1),0) + |max(a,0)|_2` | `c=sqrt(x0²+x1²)`, `e=|x2|`, `grada = (x0/max(c,1/mjMAXVAL), x1/max(c,1/mjMAXVAL), x2/max(e,1/mjMAXVAL))`; `j = a0>a1 ? 0 : 1`; if `a[j] < 0`: `g = j==0 ? (grada0, grada1, 0) : (0,0,grada2)`; else `b = max(a,0)`, `bn = max(|b|, 1/mjMAXVAL)`, `g = (grada0 b0/bn, grada1 b0/bn, grada2 b1/bn)` |
| SDF | plugin present: `p->sdf_distance(x, d, i)` (i = plugin instance); else `oct_distance(m, x, i)` (i = mesh id) | `p->sdf_gradient` or `oct_gradient` |
| MESH | requires `mesh_octadr[i] != -1` else `mjERROR("sdf queries require needsdf=\"true\"...")`; `oct_distance` | `oct_gradient` |
| other (HFIELD…) | `mjERROR("sdf collisions not available ...")`, returns 0 | |

`radialField3d(field, a, x, size)`: `field = normalize(-size/a)` (componentwise division by negative `a`, so all positive), then `field_k = -field_k` if `x_k < 0`. This is a unit vector that rotates linearly from radial (center) toward the face normal near the surface; the interior box distance is the ray–surface intersection along it.

### 2.1 Composite SDFs — `mjc_distance(m,d,s,x)` / `mjc_gradient(m,d,s,g,x)`

Let `y = relmat·x + relpos` (only for types other than SINGLE), `A = f0(x)` (object `id[0]`, type `geomtype[0]`), `B = f1(y)`.

| type | distance | gradient (in frame of x) |
|---|---|---|
| SINGLE | `A` | `∇f0(x)` |
| INTERSECTION | `max(A,B)` | `i = (A > B) ? 0 : 1` (tie → 1); `g = ∇f_i`; if `i==1`: `g = relmatᵀ ∇f1(y)` (note: A and B are re-evaluated inside the gradient call) |
| MIDSURFACE | `A - B` | `g1 = normalize(∇f0(x))`, `g2 = normalize(relmatᵀ ∇f1(y))`, `g = normalize(g1 - g2)` (all via `mju_normalize3`, which returns (1,0,0) for norm < mjMINVAL) |
| COLLISION | `A + B + |max(A,B)|` | `g = ∇f0 + relmatᵀ∇f1 + sgn·(A > B ? ∇f0 : relmatᵀ∇f1)` with `sgn = max(A,B) > 0 ? +1 : -1` (zero → −1) |

COLLISION objective: if `max(A,B) < 0` (point inside both) it equals `A + B − max(A,B) = min(A,B)`; if `max(A,B) >= 0` it equals `A + B + max(A,B)`. Minimizing it drives the point into the intersection and toward its deepest region while staying well defined (and penalized) outside.

`mjc_getSDF(m, id)`: `instance = geom_plugin[id]`, `slot = m->plugin[instance]`, returns `mjp_getPluginAtSlotUnsafe(slot, nslot)`; errors if NULL or lacks `mjPLUGIN_SDF`.

`mapPose(xpos1, xquat1, xpos2, xquat2, pos12, mat12)`: pose of frame 1 expressed in frame 2: `(negpos,negquat) = inverse(xpos2,xquat2)`; `(pos12, quat12) = (negpos,negquat) ∘ (xpos1,xquat1)` i.e. `pos12 = R2ᵀ(xpos1 - xpos2)`, `quat12 = q2⁻¹ q1`; `mat12 = quat2Mat(quat12)`. **All callers first convert `geom_xmat` to quaternions with `mju_mat2Quat`** — the rotation actually used is the mat→quat→mat round trip (floating-point relevant).

---

## 3. Optimizers

### 3.1 `stepGradient(x, m, s, d, niter) -> dist` (backtracking gradient descent, Armijo)

```
dist = mjMAXVAL
for step in 0..niter-1:
    alpha = 2.0
    grad = mjc_gradient(s, x)
    if any grad_k is NaN or |grad_k| > mjMAXVAL: return mjMAXVAL         # x left as is
    x0 = x ; dist0 = mjc_distance(s, x0)
    wolfe = -0.1 * alpha * dot(grad,grad)
    do:
        alpha *= 0.5 ; wolfe *= 0.5
        x = x0 - alpha*grad
        dist = mjc_distance(s, x)
    while alpha > 1e-4 && dist - dist0 > wolfe
    if dist0 < dist: x = x0; return dist0                                  # no improvement -> early stop
return dist
```
First trial step is `alpha = 1` (after the first halving). Minimum trial `alpha` ≈ `6.1e-5` (halvings 1, .5, …, until ≤1e-4: 2⁻¹⁴ = 6.1035e-5 is the last evaluated). The Armijo threshold is `-0.1*alpha*|g|²` for the *current* alpha. `dist` returned is at the final x; if `niter == 0` returns `mjMAXVAL`.

### 3.2 `stepFrankWolfe(x, corners, ncorners, m, sdf, d) -> dist`

Minimizes the SDF over the convex hull of `corners` (triangle): for `step in 0..sdf_iterations-1`: `grad = mjc_gradient(sdf, x)`; `s = argmin_i dot(corner_i, grad)` (strict `<`, first wins on ties; initial best `mjMAXVAL`); `x += (2/(step+2)) * (s - x)`. No line search, no improvement test. Returns `mjc_distance(sdf, x)`. (Step 0 jumps fully to the best vertex: `2/2 = 1`.)

### 3.3 `mju_Halton(index, base)` (engine_util_misc.c)

```
n0=index; f=1/base; hn=0
while n0>0: n1=n0/base (int); r=n0-n1*base; hn += f*r; f /= base; n0=n1
return hn            # Halton(0,·) = 0
```

---

## 4. Narrowphase entry points (engine_collision_sdf.c)

Dispatch table (`mjCOLLISIONFUNC`, last column = SDF): PLANE–SDF → `mjc_PlaneConvex` (SDF treated as its mesh convex hull, not covered here); HFIELD–SDF → `mjc_HFieldSDF`; SPHERE/CAPSULE/ELLIPSOID/CYLINDER/BOX–SDF → `mjc_SDF`; MESH–SDF → `mjc_MeshSDF`; SDF–SDF → `mjc_SDF`. Geom pairs are ordered so `type(g1) <= type(g2)`, hence the SDF is always `g2`. `mj_maxContact` returns `opt.sdf_initpoints` for any pair involving an SDF geom (this sizes the per-pair buffer slice; see pitfalls for MeshSDF).

**`margin` is ignored by all SDF functions**: contacts are generated only for `dist <= 0` (see `addPreContact`).

### 4.1 `isknown(points, x, cnt)` and `addPreContact(points, con, x, pos2, quat2, dist, cnt, m, s, d, flipNormal) -> cnt'`

```
if dist > 0 or any(|x - points_i| < mjMINVAL for i<cnt): return cnt
points[cnt] = x                                    # recorded even if normal is degenerate below
norm = mjc_gradient(s, x); len = mju_normalize3(norm)
if len < mjMINVAL: return cnt                      # skip degenerate normal
if !flipNormal: norm = -norm                       # 0: into SDF (SDF is object 2); 1: out of SDF (SDF is object 1)
con->dist   = dist
con->normal = rotVecQuat(norm, quat2)
con->pos    = rotVecQuat(x, quat2) + pos2 + normal * (flipNormal ? -0.5*dist : +0.5*dist)
con->tangent = 0
return cnt+1
```
`x` is in the SDF (object `id[0]`) local frame; `quat2`/`pos2` its world pose. Note: the point is stored in `points` before the normal test, so a later identical point is still rejected.

### 4.2 `mjc_HFieldSDF` — not supported

Emits `mju_warning("HField vs SDF collision not yet supported!")`, returns 0.

### 4.3 `mjc_SDF(m, d, con, g1, g2, margin)` — primitive/SDF vs SDF

Inputs: `geom_xpos/xmat[g1,g2]`, `geom_aabb[g1,g2]` (`size1/size2` = 6-vectors center/half), plugin/mesh ids. Requires `geom_type[g2] == mjGEOM_SDF` (else `mjERROR`).

1. `quat1 = mat2Quat(mat1)`, `quat2 = mat2Quat(mat2)`; `(offset2, rotation2) = mapPose(pos1,quat1,pos1,quat1)` (≈ identity, kept for exactness); `(offset21, rotation21) = mapPose(pos2,quat2,pos1,quat1)` (g2-local → g1-local); `(offset12, rotation12) = mapPose(pos1,quat1,pos2,quat2)` (g1-local → g2-local).
2. AABB intersection in **g1 frame**: for `i in 0..7`, corner of g1 box `vec1_k = c1_k ± h1_k` (bit k of i), corner of g2 box mapped `vec2 = rotation21·(c2 ± h2) + offset21`; accumulate min/max (init ±mjMAXVAL). `aabb = [max(min1,min2), min(max1,max2)]`. If `max < min` on any axis → return 0.
3. Build `mjSDF sdf`: `id = {inst2, inst1}`, `geomtype = {type[g2], type[g1]}`, `plugin = {sdf2, sdf1}`, `relpos = offset21`, `relmat = rotation21` (so `x` lives in the **g2 frame**, `y` in the g1 frame). `inst2 = geom_plugin[g2]` or, if −1, `geom_dataid[g2]` (octree mesh). For g1: if SDF, same rule; else `inst1 = g1` (geom id, used for `geom_size`), plugin NULL. If `sdf_ptr[0]` exists, `reset(...)` is called (visualizer).
4. Sampling loop (`i` counts started points, `j` the Halton index; currently identical, both start at 0):
```
while i < sdf_initpoints:
    x = aabb.min + (aabb.max-aabb.min) ∘ (Halton(j,2), Halton(j,3), Halton(j,5))   # g1 frame; j=0 -> min corner
    y = rotation2·x + offset2 ; x = rotation12·y + offset12                          # to g2 frame
    j++ ; i++
    if sdf_ptr[0]: sdf_ptr[0]->compute(m, d, instance[0], mjPLUGIN_SDF)              # visualizer bookkeeping
    sdf.type = COLLISION;    dist = stepGradient(x, m, &sdf, d, sdf_iterations)
    sdf.type = INTERSECTION; dist = stepGradient(x, m, &sdf, d, 1)                   # projection to filter spurious minima
    sdf.type = MIDSURFACE;   cnt  = addPreContact(contacts, con+cnt, x, pos2, quat2, dist, cnt, m, &sdf, d, 0)
    if cnt >= mjMAXCONPAIR: break
```
Result: contact distance = value of `max(f_g2, f_g1)` after the single intersection step; normal = `−normalize(n̂_g2 − n̂_g1)` (from g1 into g2); position = point + `0.5*dist*normal`. Each start is independent (no warm start). Duplicates (|Δx| < 1e-15) removed. Order of contacts = Halton order.

### 4.4 Triangle culling helpers

`triangleIntersect(triangle[9], m, sdf, d) -> bool` (triangle already in SDF frame):
```
e0 = t1 - t0 ; e1 = t2 - t0 ; n = normalize(e0 × e1)
p  = t0/3 + t1/3 + t2/3                       # computed as scl then two addToScl (order matters)
dc = mjc_distance(sdf, p) ; h = -dc/10
if |h| < 0.1:                                 # degenerate tetrahedron fallback
    a=|t0-t1|, b=|t1-t2|, c=|t2-t0|, s=(a+b+c)/2, area=sqrt(s(s-a)(s-b)(s-c))
    R = a*b*c / (4*max(area, mjMINVAL)) ; return dc < R
p4 = p - h*n                                   # mju_addToScl3(p, normal, -h)
v_i = t_i - p4 ; cross0 = v1×v2, cross1 = v2×v0, cross2 = v0×v1
center = |v0|² cross0 + |v1|² cross1 + |v2|² cross2
denom = 2*dot(v0, cross0)
if |denom| < mjMINVAL: return dc < |e0|
center /= denom ; r = |center| ; center += p4
return mjc_distance(sdf, center) < r           # circumsphere of tetra (t0,t1,t2,p4) touches SDF
```
`boxIntersect(bvh[6], offset, rotation, m, s, d)`: `r = |half|`, `c = rotation·center + offset`; `return mjc_distance(s, c) < r` (circumsphere of the BV).

### 4.5 `processSdfCorners(corners[9], m, d, sdf, nstartpts, candidate, dist, ncandidate)`

```
if !triangleIntersect(corners): return
for sp in 0..nstartpts-1:
    if *ncandidate >= mjMAXCONPAIR: break
    u = Halton(sp+1, 2); v = Halton(sp+1, 3)          # index starts at 1
    if u+v > 1: u = 1-u; v = 1-v
    x = (1-u-v)*c0 + u*c1 + v*c2                       # per component: b0*c0 + b1*c1 + b2*c2
    depth = stepFrankWolfe(x, corners, 3, m, sdf, d)
    if depth < 0: candidate[n] = x; dist[n] = depth; n++
```

### 4.6 `traverseBVH(bvh, nodeid, child, bvh_active, offset, rotation, m, d, sdf, callback, ctx)`

Iterative DFS with `int stack[64]`, starting at node 0. Pop node; **leaf** (`nodeid[node] != -1`): if `boxIntersect` → `active = callback(nodeid[node], node, ctx)`, mark `bvh_active` if active. **Internal**: if `!boxIntersect` skip; mark active; push `child[2n+0]` then `child[2n+1]` (if != −1; `nstack >= 64` → `mjERROR("BVH stack depth exceeded.")`). Because of LIFO, **child 1 is processed before child 0** — leaf visitation order determines which candidates survive the 50-candidate cap.

### 4.7 `mjc_MeshSDF(m, d, con, g1, g2, margin)` — mesh (g1) vs SDF (g2)

1. SDF: `instance = geom_plugin[g2]`; plugin pointer via `mjc_getSDF` or NULL (then instance = `geom_dataid[g2]`, octree); `sdf.type = SINGLE`, `geomtype = mjGEOM_SDF`.
2. `quat1 = mat2Quat(mat1)`, `sdf_quat = mat2Quat(mat2)`; `(offset, rotation) = mapPose(pos1, quat1, pos2, sdf_quat)` (mesh-local → SDF-local).
3. `nstartpts = max(1, sdf_initpoints)`. If the mesh has a BVH (`mesh_bvhadr >= 0`): `traverseBVH` over `m->bvh_aabb` (mesh frame) with `meshFaceCallback` → `processOneFace(face)`: transform the 3 vertices (`mesh_vert` floats → mjtNum) `corner = rotation·v + offset`, call `processSdfCorners`; returns 1 if new candidates appeared. If no BVH: **no contacts** (nothing traversed).
4. If `ncandidate <= mjMAXCONPAIR` (always true, since `processSdfCorners` stops at 50): for each candidate in order, `addPreContact(..., pos2, sdf_quat, dist, ..., flipNormal=0)`. (The FPS branch `selectFPS` is only reached if `ncandidate > 50`, which cannot happen; documented for completeness below.)
5. Contact: `x` (the Frank–Wolfe point on the mesh triangle, SDF frame), normal = −∇SDF (into the SDF, from mesh to SDF), `pos = R2 x + pos2 + 0.5*dist*normal`, `dist = SDF(x) < 0`.

`selectFPS(candidate, dist, n, sel, max)`: seed = argmax(−dist) (first wins ties); loop while `nsel < max && nsel < 50 && best >= 0`: select best, update `min_dist2[i] = min(min_dist2[i], |c_i − c_best|²)` for unselected, next best = argmax `min_dist2` (strict `>`, initial −1, so zero-distance duplicates can still be picked).

### 4.8 `mjc_FlexSDF(m, d, con, elem, g, f, margin)` — SDF geom (g, object 1) vs flex (f, object 2)

Only `flex_dim[f] == 2` (triangles); otherwise returns 0. The driver only creates this pair for dim-2 flexes.

1. SDF setup as in MeshSDF. `sdf_quat = mat2Quat(geom_xmat[g])`; `(offset, rotation) = mapPose(0, (1,0,0,0), sdf_pos, sdf_quat)` → world → SDF-local (`rotation = R_sdfᵀ` via quat, `offset = −R_sdfᵀ p_sdf`).
2. `vertxpos = flexvert_xpos + 3*flex_vertadr[f]`, `nstartpts = max(1, sdf_initpoints)`.
3. **No flex BVH** (`flex_bvhadr[f] < 0`): for each element `e` in order: corners = `rotation·x_v + offset`; local candidates via `processSdfCorners` (own 50-cap per element); then for each element candidate (while global `cnt < mjMAXCONPAIR`) `addPreContact(elem_points, ..., flipNormal=1)` with **per-element** duplicate list; on success `elem[cnt] = e; cnt++`. Returns as soon as all elements are done (contacts in element order, up to 50).
4. **With BVH**: `bvh = d->bvh_aabb_dyn + 6*(bvhadr − nbvhstatic)` (world frame, refreshed in `mj_flex`), `traverseBVH(..., flexElemCallback)`. `flexElemCallback(e)`: corners as above; `processSdfCorners` into a local buffer; keep only the **deepest** candidate of this element (first wins ties, strict `<`); append to global list if `nc < 50` (records `elem_id`), return 1. Then (always ≤ 50) `addPreContact(points, ..., flipNormal=1)` for each candidate in order; `elem[old_cnt] = elem_id[i]` on success.
5. Contact: normal = +∇SDF rotated to world (out of SDF toward flex), `pos = R x + p − 0.5*dist*normal`. Driver writes `geom = {g,−1}`, `flex = {−1,f}`, `elem = {−1, elem[i]}`.

---

## 5. Analytic SDF plugins (`plugin/sdf/`)

Registered by `mjPLUGIN_LIB_INIT(sdf)`: Bolt, Bowl, Gear, Nut, Torus (names `mujoco.sdf.bolt|bowl|gear|nut|torus`). Common helpers (sdf.h): `Union(a,b)=min`, `Intersection(a,b)=max`, `Subtraction(a,b)=max(a,−b)`, `Fract(x)=x−floor(x)`. Attribute parsing: `SdfDefault<T>::GetDefault` — empty/whitespace → default, otherwise `strtod` with full-string check (NaN if invalid); `CheckAttr` validates at `init`. Instance object stored in `d->plugin_data[instance]`. `nstate = 0`.

Gradient: **Torus is analytic; Bolt, Bowl, Gear, Nut use forward differences with `eps = 1e-8`**: `g_k = (f(p+eps e_k) − f(p))/eps`. Bolt/Bowl/Gear/Nut `sdf_gradient` also calls `visualizer_.AddPoint(p)` (side effect only; buffer `10*(50+1)^2*3` doubles); their `compute` appends a new trajectory (`Next()`), `reset` clears it; `visualize` draws iterations when `mjVIS_SDFITER`.

| Plugin | Attributes (default) | Distance `f(p)` | `sdf_aabb` (center, half) |
|---|---|---|---|
| **torus** | `radius1` R (0.35), `radius2` r (0.15) | `q = sqrt(px²+py²) − R`; `f = sqrt(q² + pz²) − r` | `(0,0,0)`, `(R+r, R+r, r)` |
| **bowl** | `height` h (0.4), `radius` R (1.0), `thickness` t (0.02) | `w = sqrt(R²−h²)`; `q = (sqrt(px²+py²), pz)`; `f = (h·q0 < w·q1 ? |q − (w,h)| : | |q| − R |) − t` (cut hollow sphere) | `(0,0,0)`, `(R+t)` all axes |
| **bolt** | `radius` ρ (0.26) | see below | `(0,0,0)`, `(0.6, 0.6, 1)` |
| **nut** | `radius` ρ (0.26) | see below | `(0,0,0)`, `(0.6, 0.6, 1)` |
| **gear** | `alpha` (0), `diameter` D (2.8), `teeth` N (25), `thickness` (0.2), `innerdiameter` (−1 ⇒ auto) | 2D involute gear extruded | `(0,0,0)`, `(1.25 D/2, 1.25 D/2, 1.1·thickness/2)` |

Torus gradient (analytic): `lxy = sqrt(px²+py²)`, `q = lxy − R`, `lqz = sqrt(q²+pz²)`; `g = (q px/lxy, q py/lxy, pz) / max(lqz, mjMINVAL)` (no guard on `lxy`).

Bolt (`screw = 12`, `s2 = sqrt(2)/2`):
```
rad = sqrt(px²+py²) − ρ ; az = atan2(py,px)
tri = |Fract(pz*12 − az/(2π)) − 0.5|
thread = (rad − tri/12) * s2
bolt = Subtraction(thread, 0.5 − |pz + 0.5|)          # clip top/bottom
cone = (pz − rad) * s2 ; bolt = Subtraction(bolt, cone + 1*s2)
k = 6/(2π) ; ang = −floor(atan2(py,px)*k + 0.5)/k     # hexagon sector
(s0,s1) = (sin ang, sin(ang + π/2)) ; [x';y'] = [[s1,−s0],[s0,s1]]·[px;py]
head = x' − 0.5
head = Intersection(head, |pz + 0.25| − 0.25)
head = Intersection(head, (pz + rad − 0.22) * s2)
f = Union(bolt, head)
```
Nut: same thread with `rad2 = sqrt(px²+py²) − ρ`, `thread2`, `cone2 = (pz − rad2)*s2`; `hole = Subtraction(thread2, cone2 + 0.5 s2)`; `hole = Union(hole, −cone2 − 0.05 s2)`; head as in bolt (using `rad2`); `f = Subtraction(head, hole)`.

Gear (`distance = extrusion(p, distance2D(p), thickness/2)`, `extrusion: w = (sdf2d, |pz| − h); min(max(w0,w1),0) + |max(w,0)|`):
```
psi = 3.096e-5 N² − 6.557e-3 N + 0.551      # pressure angle
R = D/2 ; rho = sqrt(px²+py²) ; Pd = N/D ; P = π/Pd ; a = 1/Pd
Ro = (D + 2a)/2 ; h = 2.2/Pd
innerR = (innerdiameter >= 0) ? innerdiameter/2 : Ro − h − 0.14 D
if innerR − rho > 0: return innerR − rho          # early exit (2D)
if Ro − rho < −0.2: return rho − Ro               # early exit (2D)
Rb = D cos(psi)/2 ; fi = atan2(py,px) + alpha ; stride = P/R
invAlpha = acos(Rb/R) ; invPhi = tan(invAlpha) − invAlpha
shift = stride/2 − 2 invPhi
fia = mod(fi + shift/2, stride) − shift/2 ; fib = mod(−fi − shift + shift/2, stride) − shift/2   # mod(x,y)=x−y floor(x/y)
dista = distb = −1e6
if Rb < rho: t = sqrt(rho²−Rb²); ac = acos(Rb/rho); dista = t − Rb(fia+ac); distb = t − Rb(fib+ac)
cogs = max(max(fia − (stride−shift), fib − (stride−shift)), max(dista, distb))
cogs = smoothIntersection(rho − Ro, cogs, 0.0035 D)
cogs = smoothUnion(rho − (Ro − h), cogs, Rb − Ro + h)
cogs = Subtraction(cogs, rho − innerR)
smoothUnion(a,b,k): hh = clip(0.5 + 0.5(b−a)/k, 0, 1); b(1−hh) + a hh − k hh(1−hh)
smoothIntersection(a,b,k) = Subtraction(Intersection(a,b), smoothUnion(Subtraction(a,b), Subtraction(b,a), k))
```

---

## 6. Flex collision primitives — `engine_collision_flex.c`

All positions are world coordinates (`flexvert_xpos`). `rt = flex_radius[f]` (flex elements are swept by a sphere of this radius). `margin` passed in = `margin + gap` from the driver.

### 6.1 `mjc_triWeights(w, p, a, b, c)` (engine_collision_primitive.c)
Closest point on triangle (Ericson 5.1.5): `ab, ac, ap, bp, cp`; `d1=ab·ap, d2=ac·ap, d3=ab·bp, d4=ac·bp, d5=ab·cp, d6=ac·cp`.
Regions in order: A if `d1<=0 && d2<=0` → (1,0,0); B if `d3>=0 && d4<=d3` → (0,1,0); C if `d6>=0 && d5<=d6` → (0,0,1); `va=d3d6−d5d4, vb=d5d2−d1d6, vc=d1d4−d3d2`; AB if `vc<=0 && d1>=0 && d3<=0` → t=d1/(d1−d3), (1−t,t,0); AC if `vb<=0 && d2>=0 && d6<=0` → t=d2/(d2−d6), (1−t,0,t); BC if `va<=0 && d4−d3>=0 && d5−d6>=0` → t=(d4−d3)/((d4−d3)+(d5−d6)), (0,1−t,t); else face: `dd=1/(va+vb+vc)`, `t=vb dd, u=vc dd`, (1−t−u, t, u).

### 6.2 `sphereTriangle(con, margin, s, rs, t1, t2, t3, rt)` → 0/1
```
w = triWeights(s, t1,t2,t3); X = w0 t1 + w1 t2 + w2 t3     # per component k
nrm = X − s ; dst = mju_normalize3(nrm)                    # (1,0,0) if dst < mjMINVAL
if dst > margin + rs + rt: return 0
dist = dst − rs − rt
pos = s + nrm * (rs + dist/2) ; normal = nrm (sphere → triangle) ; tangent = 0
```

### 6.3 `boxTriangle(con, margin, pos, mat, size, t1,t2,t3, rt)` → up to 11
1. For each triangle vertex `v_i`: `local = matᵀ(v_i − pos)`; `maxaxis = argmax_j (|local_j| − size_j)` (strict `>`, j=0 first); `maxval`. Skip if `maxval − rt > margin`. Skip if any `|local_j| > size_j + margin + rt`. Else `n_local = ±e_maxaxis` (sign of `local[maxaxis]`, `>0` → +), `normal = mat·n_local` (box → triangle), `dist = maxval − rt`, `pos = v_i − normal*(rt + dist/2)`.
2. For box corners `i=0..7` (`±size`, bit k of i → +), `corner = mat·vec + pos`; `sphereTriangle(con+cnt, margin, corner, 0, t1,t2,t3, rt)` (normal corner → triangle). Stops at `mjMAXCONPAIR`.
Note: interior vertex contacts use the face of max |local|−size, valid also for vertices outside.

### 6.4 `capsuleTriangle(con, margin, pos, mat, size, t1,t2,t3, rt)` → up to 5
`r = size0`, `len = size1`, `axis = mat column 2 = (mat2, mat5, mat8)`, `p1 = pos − len·axis`, `p2 = pos + len·axis`.
1. `sphereTriangle(p1, r)`, then `sphereTriangle(p2, r)`.
2. Each vertex `v`: `t = (v−p1)·(p2−p1) / (4 len²)` (no zero guard); only if `mjMINVAL < t < 1−mjMINVAL`: `closest = p1 + t(p2−p1)`, `vec = v − closest`, `dist = normalize(vec)`; skip if `dist > r + rt + margin`; `con.dist = dist − r − rt`, `normal = vec` (capsule → vertex), `pos = 0.5*(closest + v + vec*(r − rt))`.

### 6.5 `mjc_PlaneFlex(m, d, con, vert, g, f, margin)` → up to `flex_vertnum[f]`
`normal = plane z axis = (mat2, mat5, mat8)`. For every vertex `i` (local index) in order: `dist = (v − pos)·normal`; skip if `dist > margin + rt`; `con.dist = dist − rt`, `pos = v − normal*(con.dist/2 + rt)`, `normal` (plane → vertex), `vert[ncon] = i`. Driver writes `con.vert = {−1, i}`.

### 6.6 Filters and `makeCapsule`
- `filterSphereBox(s, bound, aabb)`: reject if `s_k + bound < c_k − h_k` or `s_k − bound > c_k + h_k` for any k.
- `filterBox(aabb1, aabb2, margin)`: reject if `c1_k + h1_k + margin < c2_k − h2_k` or symmetric, any k.
- `makeCapsule(m, d, f, vid[2], pos, mat, size)`: `dif = v1 − v2`; `size0 = flex_radius[f]`; `size1 = 0.5*normalize(dif)` (half length); `pos = 0.5(v1 + v2)` (as `(v1+v2)*0.5`); `quat = quatZ2Vec(dif)`; `mat = quat2Mat(quat)` (capsule axis = +z along v1 − v2).

### 6.7 `mjc_GeomElem(m, d, con, g, f, e, margin)`
1. Only if midphase is **disabled**: `filterSphereBox(geom_xpos[g], geom_rbound[g] + margin, flexelem_aabb[elemadr+e])` → reject.
2. Same-body skip: if `b = geom_bodyid[g] >= 0` equals `flex_vertbodyid[vertadr + edata[i]]` for any element vertex → 0.
3. Dispatch:
   - `dim == 1` and type ∈ {SPHERE, CAPSULE, BOX}: capsule from the 2 vertices (`makeCapsule`) and `mjraw_SphereCapsule(geom, cap)`, `mjraw_CapsuleCapsule(geom, cap)`, or `mjraw_CapsuleBox(cap, geom)` with **all normals negated** (so normal is geom → element).
   - type HFIELD: `mjc_HFieldElem` (in convex spec; per-prism CCD over the hfield sub-grid around the element's hfield-frame AABB inflated by `radius + margin`).
   - `dim == 2`: SPHERE → `sphereTriangle(geom_xpos, geom_size0, t1,t2,t3, rt)`; BOX → `boxTriangle`; CAPSULE → `capsuleTriangle`.
   - otherwise (ellipsoid, cylinder, mesh, any geom vs dim 3, dim 1 vs cylinder/ellipsoid/mesh): `mjc_ConvexElem(m, d, con, g, −1, −1, −1, f, e, margin)`.
4. `pairMaxContact` (driver) for GEOM_ELEM: HFIELD or (dim 2 and BOX/CAPSULE) → 50; dim 1 and CAPSULE/BOX → 2; else 1.

`mjc_ConvexElem(g1, f1, e1, v1, f2, e2, margin)` (convex spec): CCD objects; flex support (`mjc_flexSupport`) = element vertex with largest `dot(v, dir)` (first wins ties), then `+ dir*(flex_radius + 0.5*margin)`; vertex objects: vertex + same inflation. `mjc_penetration(..., maxcon=1, margin)`. If `ncon` and native CCD enabled and either side is a 2D flex: `mjc_fixNormal(con, g1, −1)` (replaces the normal by the analytic surface normal of sphere/capsule/ellipsoid/cylinder geoms; no-op for others).

### 6.8 Interpolation-grid exclusion (interpolated flexes, `flex_interp != 0`)

`mjc_vertCellRange(m, f, v, lo[3], hi[3])`: `coord = flex_vert0[vertadr+v]` (∈[0,1]³), `cellnum = flex_cellnum[3f..]`, `cell = flex_vertcell[gv]`.
- If `cell >= 0`: compute the nominal cell `ci = clamp(floor(coord0·cx), 0, cx−1)` (same cj, ck), linear `ci·cy·cz + cj·cz + ck`. If it differs from `cell` (vertex was assigned to an active cell that extrapolates it): `lo = hi = (cell/(cy cz), (cell/cz) % cy, cell % cz)`; return.
- Else per axis: `u = coord_i · cellnum_i`, `eps = 1e-6·cellnum_i`, `lo_i = clamp(floor(u − eps), 0, n−1)`, `hi_i = clamp(floor(u + eps), 0, n−1)` (vertex on an internal plane belongs to both cells).

`mjc_cellRangeShared(lo1,hi1,lo2,hi2)` = for all axes `lo2 − hi1 <= 1 && lo1 − hi2 <= 1`.

`mjc_ElemSharedNode(m, f, e1, e2)`: 0 if not interpolated; else 1 if any vertex pair (i1, i2) of the two elements has shared/neighbouring cell ranges. Used by `pushSelfElemElem` to skip self-collision of elements driven by a common grid node.

### 6.9 `mjc_ElemElem(m, d, con, f1, e1, f2, e2, margin)`
1. `if f1 == f2: margin = 0` (self-collision ignores margin and gap).
2. `filterBox(flexelem_aabb[e1], flexelem_aabb[e2], margin)` → reject.
3. Shared-body skip: any vertex pair with `b1 >= 0 && b1 == b2`.
4. `dim1 == 1 && dim2 == 1`: two capsules via `makeCapsule`, `mjraw_CapsuleCapsule` (≤ 2 contacts). Else `mjc_ConvexElem(m, d, con, −1, f1, e1, −1, f2, e2, margin)` (1 contact).

---

## 7. Driver logic specific to flex/SDF (engine_collision_driver.c)

### 7.1 Pair generation
- `mj_isElemActive(m,f,e)`: `dim < 3` → 1; else `flex_elemlayer[elemadr+e] < flex_activelayers[f]`.
- `flexReducible(f) = flex_rigid[f] || flex_interp[f]` — only such flexes get a filter `group` (FPS reduction to 50 per group).
- **Body : flex, all-to-all path** (midphase disabled or no BVH): for each geom g of the body: bitmask filter geom vs flex (`filterBitmask(contype/conaffinity)`); skip if `mjc_ipcOwnsFlexGeom` (IPC flag, dim 2, static geom of supported type); `group = reducible ? ngroup++ : −1`; PLANE → `pushGeomFlex`; SDF → `pushGeomFlex` only if `flex_dim == 2` (else nothing); other types → `pushGeomElem` for every active element.
- **Body : flex, midphase (`mj_collideTree`)**: before traversal, for dof-less bodies (`body_dofnum[body_weldid[bf1]] == 0`) each PLANE geom → `pushGeomFlex`; for any body each SDF geom → `pushGeomFlex` if dim 2. During traversal, leaf–leaf (geom, element): margin `mj_assignMargin(geom_margin + flex_margin)`, gap sum; bitmask filter; `filterSphereBox(geom pos, rbound + margin + gap, element BV)`; `mj_collideOBB(geom_aabb (rotated), element box (world-aligned), margin+gap)`; push GEOM_ELEM unless geom is PLANE or SDF (handled above) or IPC-owned. Group for body:flex in midphase: `group + g − body_geomadr` (one group per geom).
- **Flex : flex** (different flexes): all-to-all `pushElemElem` over active elements (group if both reducible), or BVH midphase: leaf–leaf `filterBox(flexelem_aabb, flexelem_aabb, margin)` with `margin = mj_assignMargin(flex_margin1+2) + gap1 + gap2`; internal nodes `filterBox(bvh1, bvh2, mj_assignMargin(margin sum))`. Traversal descends into the node with larger surface `xy+yz+zx` computed from `bvh[3]-bvh[0]` etc. (note: uses center/half fields as if min/max — heuristic only). Self (bf1 == bf2): skip `node1 > node2`; push via `pushSelfElemElem` only if `nodeid1 != nodeid2`. Pairs generated by one tree are sorted (`pairSort`: by group, then type, then object ids).
- **Flex self-collision** (after all bodyflex pairs): requires `!flex_rigid && (contype & conaffinity)`, awake dofs, not IPC-owned; skip if `flex_selfcollide == NONE` or *all_excluded* (`interp && cellnum <= 2` on every axis). `group = reducible ? ngroup++ : −1`. Mode:
  - midphase enabled, mode != NARROW, and (`flex_bvhadr >= 0` or AUTO+interp):
    - AUTO + interp → `mj_collideFlexCellPairs`;
    - BVH, or AUTO + dim 3 → `mj_collideTree(f,f)`;
    - else (SAP, or AUTO with dim < 3) → `mj_collideFlexSAPPairs`.
  - otherwise direct: all `e1 < e2` active pairs via `pushSelfElemElem`.
- `mj_collideFlexSAPPairs`: active elements; AAMM from `flexelem_aabb` (center ± half, radius included, **no margin**); axis = largest half-size of the flex BVH root dynamic box (`bvh[3] > bvh[4] && bvh[3] > bvh[5] ? 0 : (bvh[4] > bvh[5] ? 1 : 2)`); `mj_SAP` (generic, see driver spec) with max `min(n(n−1)/2, 1e6)` pairs; each → `pushSelfElemElem`.
- `mj_collideFlexCellPairs` (interpolated flexes): vertex cell ranges; each active element assigned to the `lo` cell of its **first vertex** and given the union range `[elo, ehi]` over its vertices; CSR bucket of non-empty cells (cell id `(x·ny + y)·nz + z`); per cell union range and AABB of member element AABBs. For cell pairs `s1 < s2`: skip if first-vertex cells are neighbours (`cellRangeShared(coord1,coord1,coord2,coord2)`) — all pairs excluded; skip if cell AABBs don't overlap (`filterBox(…, 0)`); `disjoint = !cellRangeShared(lo1,hi1,lo2,hi2)`; for member pairs passing `filterBox(elem aabbs, 0)`: if not disjoint, exclude if any vertex pair shares cell ranges; push `(min(e1,e2), max(e1,e2))`. Elements in the same cell are never paired. Pairs sorted at the end.

### 7.2 Narrowphase dispatch (`collisionTask`) and contact assembly
- GEOM_FLEX: `margin = mj_assignMargin(geom_margin + flex_margin)`, `gap` sum; PLANE → `mjc_PlaneFlex(..., flexbuffer)`; SDF → `mjc_FlexSDF(..., flexbuffer)`. `pairMaxContact`: PLANE → `flex_vertnum[f]`, else 50.
- GEOM_ELEM: `mjc_GeomElem(..., margin + gap)`.
- ELEM_ELEM: `margin = mj_assignMargin(sum)`, gap sum; if `f1 == f2` margin = 0 (gap kept but `mjc_ElemElem` zeroes the total). `pairMaxContact`: 2 if both dim 1 else 1.
- `addPairContacts`: GEOM_FLEX/ELEM: `geom = {g, −1}`, `flex = {−1, f}`, `elem = {−1, e or elem[i]}`, `vert = {−1, vert[i]}` (plane); `includemargin = mj_assignMargin(geom_margin + flex_margin)` (gap excluded → in-gap contacts are `exclude`d). ELEM_ELEM: `flex = {f1, f2}`, `elem = {e1, e2}`, `includemargin = (f1 == f2) ? 0 : mj_assignMargin(sum)`. Parameters via `mj_contactParam(m, ..., g1, g2, f1, f2)`.
- `filterPreContacts(group)` (only for reducible-flex groups, after all tasks): concatenate contacts of the contiguous group pairs in pair order; if count > 50: seed = deepest (largest −dist, strict `>`, first wins); greedy FPS with **squared** distances `min_dist[idx] = min(min_dist, d²)`, next = argmax (strict `>`, initial −1) until 50 selected or none left; compact in place preserving original order (and `flexbuffer` ids).

---

## 8. Programmatic optimizations (C)

- SDF: AABB-intersection pre-cull (`mjc_SDF`); circumsphere culls of BVH nodes (`boxIntersect`) and triangles (`triangleIntersect`) using a single SDF evaluation each; inline BVH traversal with callbacks (no intermediate face lists); early break at `mjMAXCONPAIR` candidates/contacts; Frank–Wolfe restricted to 3 vertices (no projection needed).
- Octree: hierarchical lookup O(depth) ≤ 100 steps; trilinear weights computed only for the needed variant (`w` or `dw`); analytic gradient inside, FD only outside the box.
- Flex: midphase BVH over elements with dynamic AABBs (`mj_updateDynamicBVH`: leaves copied from `flexelem_aabb`, internal nodes recomputed backward only if a child changed); SAP for self-collision; cell bucketing for interpolated flexes (O(#cells² + pairs)); `pairMaxContact` pre-sizes per-pair slices so threads write without locks; narrowphase chunked (`chunksize = npair/(5·nthread)`) via `mju_dispatch`, per-thread CCD/EPA buffer; arena/stack allocation (`mj_markStack`/`mjSTACKALLOC`) for all temporaries; pairs sorted deterministically, independent of thread count.

---

## 9. GPU (MuJoCo Warp) implementation

### 9.1 SDF (`collision_sdf.py`)

**Entry**: `collision_driver._narrowphase` calls `sdf_narrowphase(m, d, ctx)` if `m.has_sdf_geom` (any `mjGEOM_SDF` geom), after convex and primitive narrowphase. Broadphase pairs are stored with `type1 <= type2` so the SDF is `geoms[1]`.

**Kernel `_sdf_narrowphase`**: launched with `dim = (opt.sdf_initpoints, naconmax)`; thread `(i, contact_tid)` = start point `i` of broadphase pair `contact_tid`. Early-outs: `i >= sdf_initpoints`, `contact_tid >= ncollision[0]`, `geom_type[g2] != SDF`. **Every pair with an SDF g2 goes through this kernel regardless of g1 type** (including MESH; PLANE pairs, if produced by the broadphase, would use `sdf = p_z`; HFIELD hits the "SDF type not implemented" path returning 0). There is no `mjc_MeshSDF` / Frank–Wolfe / triangle path and no flex–SDF.

Per thread:
1. `contact_params(...)` → margin, gap, condim, friction, solref, solreffriction, solimp, adhesion (standard geom/pair mixing).
2. `geom_collision_pair` → `geom1/geom2` (pos, rot, size …).
3. AABB intersection in g1 frame: `transform_aabb(aabb_pos, aabb_size, pos, ori)` = min/max over 8 corners of `ori·(±size + aabb_pos) + pos` (init ±1e9). `aabb1` with identity, `aabb2` with `g1_to_g2_rot = R1ᵀR2`, `g1_to_g2_pos = R1ᵀ(p2 − p1)`. Empty intersection (`min > max`, strict) → return. (Uses `geom_xmat` directly — no quaternion round trip.)
4. `get_sdf_params` per geom: `attributes[0..2] = geom_size` by default; SDF with plugin → `plugin_attr[plugin_id]` (vec of 128 floats) and `plugin_index = m.plugin[plugin_id]` (registered plugin slot, used as `sdf_type`); SDF without plugin, or MESH with `mesh_octadr != -1` → `VolumeData` (octree root `octadr`, center/half from `oct_aabb[octadr]`).
5. `MeshData` for both geoms is filled with `valid = True` (see mesh caveat below).
6. Start point: `x_g1 = aabb.min + (max−min) ∘ (halton(i,2), halton(i,3), halton(i,5))` (i from 0, identical to C's `j`), `x = R1 x_g1 + p1`, `x0 = R2ᵀ(x − p2)`.
7. `gradient_descent(...)`: `rel_mat = R1ᵀR2`, `rel_pos = R1ᵀ(p2 − p1)` (g2-local → g1-local). Phase 0: `gradient_step(niter = sdf_iterations, intersection = False)` (COLLISION objective); phase 1: `niter = 1`, `intersection = True`. Then `grad1 = normalize(rel_matᵀ ∇f_g1(x1))`, `grad2 = normalize(∇f_g2(x))`, `n = normalize(grad1 − grad2)`, `pos = R2 x + p2`, `n = R2 n`, **`pos3 = pos − n·dist/2`**, returns `(dist, pos3, n)`.
8. `write_contact(naconmax, 0, dist, pos3, make_frame(n), margin, gap, …)`: written if `dist < margin + gap` (or collision sensor); `atomic_add(nacon)`; overflow beyond `naconmax` dropped.

`gradient_step` (float32):
```
amin=1e-4; rho=0.5; c=0.1; dist=1e10
for i in range(niter):
   alpha = alpha0 (=2.0, passed at launch to prevent NVRTC unrolling)
   x2 = x ; x1 = rel_mat x2 + rel_pos
   grad, dist0 = compute_grad(...)            # dist0 from the same evaluation
   if dot(grad,grad) < 1e-12: return dist0, x # (C: no such test; C has NaN/>mjMAXVAL test instead)
   wolfe = -c*alpha*|grad|²
   loop: alpha*=rho; wolfe*=rho; x = x2 - alpha grad; dist = clearance(...)
         break if alpha <= amin or dist - dist0 <= wolfe
   if dist > dist0: return dist0, x2
return dist, x
```
`compute_grad`: `A = f_g1(x1)`, `B = f_g2(x2)`; `grad1_t = rel_matᵀ∇f_g1`; intersection: `dist = max(A,B)`, `grad = A > B ? grad1_t : grad2` (tie → g2; **C tie → g1**); collision: `dist = A + B + |max|`, `grad = grad2 + grad1_t + sign(max)·(A > B ? grad1_t : grad2)` (Warp `sign(0)` = +1 per `wp.sign`; **C uses −1 for max == 0**, and C's tie picks g1 while Warp's picks g2 — note C's A/B are swapped relative to Warp: C `A = f_g2`).

Primitive SDFs (Warp vs C):
- sphere, capsule, cylinder, box distances: same formulas. Ellipsoid: `k1 == 0 → denom 1e-12`.
- `grad_sphere`: `p/|p|` if `|p| > 1e-9` else 0 (C: NaN). `grad_capsule`: 0 if `c <= MJ_MINVAL`. `grad_box`: same branches (`max(a) < 0` → radial). `grad_ellipsoid`: same math, normalized by `/length` (no zero guard). `grad_cylinder`: rewritten: `u = (px,py,0)/rxy` (0 if `rxy <= MINVAL`), `w = (0,0,sign(pz))`; `dx>0 && dy>0` → `u dx/|v| + w dy/|v|`; `dx>0` → u; `dy>0` → w; inside → `dx > dy ? u : w`. Equal to C except at `dx == 0` or `dy == 0` exactly, and `pz == 0` (`sign(0)=1` vs C `0/1e-10 = 0`).
- `radial_field` uses `wp.sign(x)` (x == 0 → +1, same as C which only flips for `x < 0`).
- Octree (`find_oct`, `box_project`, `sample_volume_sdf/grad`): `find_oct` eps **1e-6** (C 1e-8); child index `oct_child[node][k] + root` (relative storage); `box_project` eps **1e-4** (C 1e-6) and returns 0 inside (equivalent for distance); exterior gradient FD step **h = 1e-4** (C 1e-8); interior trilinear gradient identical. `oct_aabb` stored as `array2d[vec3]` `[node, 0=center/1=half]`, `oct_child` as `vec8i`, `oct_coeff` as `vec8`; interpolation via `wp.dot(weights, coeff)` (summation order may differ).
- **Mesh caveat**: `sdf()`/`sdf_grad()` test `type == MESH and mesh_data.valid` *before* the octree branch and the kernel always sets `mesh_data.valid = True`, so MESH geoms use a **ray-cast pseudo-distance**: cast from p toward `−normalize(p)` with `ray_mesh` (brute force over all faces of the mesh, plus box test); if hit distance `> |p|` cast the opposite way and return `−dist`. The "gradient" is `(1,1,1)` if `dist > |p|` else `(−1,−1,−1)`. `ray_mesh` receives the geom's world `pos/mat` while `p` is already in local coordinates. Mesh–SDF results therefore differ completely from C's `mjc_MeshSDF`.
- **Plugin SDFs**: `user_sdf(p, attr, sdf_type)` / `user_sdf_grad` are placeholders that `printf` an error and return 0; users must monkey-patch `mujoco_warp._src.collision_sdf.user_sdf/user_sdf_grad` with `@wp.func`s dispatching on `sdf_type` (the plugin slot). `test_data/collision_sdf/{nut,bolt,torus,gear}.py` provide float32 ports (FD gradients with **eps = 1e-6**; torus analytic with `max(len_xy,1e-8)` and denominator `max(len_qz, max(r, 1e-8))`); **bowl has no port**. `io.py` builds `plugin_attr` by parsing `mjm.plugin_attr` (null-separated ASCII number strings) in stored order, padded to `_NPLUGINATTR = 128`; only geom plugins are kept (`geom_plugin_index`). Defaults are not applied by Warp.

**Differences summary (SDF)**
| Aspect | C | Warp |
|---|---|---|
| Mesh–SDF | triangle BVH + circumsphere culls + Halton barycentric starts + Frank–Wolfe on faces, ≤50 candidates | generic two-SDF gradient descent with ray-cast mesh "SDF" |
| Flex–SDF | `mjc_FlexSDF` | `NotImplementedError` at `put_model` if any flex and any SDF geom |
| HField–SDF | warning, 0 contacts | not handled (error print, dist 0) |
| contact acceptance | `dist <= 0`, margin ignored | `dist < margin + gap` (margin honoured; positive-distance contacts possible) |
| duplicates | `isknown` (< 1e-15) | none |
| max contacts / pair | `min(sdf_initpoints, 50)` | `sdf_initpoints` (no 50 cap) |
| degenerate normal | contact skipped | written with `normalize(0)` frame |
| contact position | `x + 0.5·dist·n` | `x − 0.5·dist·n` (opposite offset) |
| NaN guard | grad NaN/huge → `mjMAXVAL` | `|grad|² < 1e-12` early return |
| rotation | mat→quat→mat | xmat directly |
| precision | float64 | float32 |
| order | Halton order per pair | atomic order (nondeterministic) |
| default buffers | — | `_default_nconmax ≥ 64` and `njmax ≥ 256` if any SDF geom |

### 9.2 Flex (`collision_flex.py`)

**Entry**: `collision_driver.collision` calls `flex_collision(m, d, ctx)` when `m.nflex > 0` and not in the incremental (sleep) pass — flex collision is **not sleeping-aware**. Unsupported (raised in `io.put_model` when `nflex > 0`): any SDF geom (flex–SDF), any HFIELD geom (flex–hfield), `flex_internal != 0`, `flex_rigid != 0`, self-collision of interpolated flexes (`flex_interp && selfcollide && contype&conaffinity`), passive contact of interpolated dim≥2 flexes. `MJ_FLEX_COLLISION_TABLE`: PLANE, SPHERE, CAPSULE, BOX, CYLINDER → primitive; MESH, ELLIPSOID → convex.

**Pipeline** (`flex_collision`):
1. `flex_broadphase_aabb` → kernel `_flex_broadphase_bounds`, dim `(nworld, nflex)`: whole-flex AABB of vertices inflated by `flex_radius + flex_margin + flex_gap` → `d.flex_aabb_min/max [nworld, nflex]`.
2. `_allocate_flex_workspace` (allocated **every call**): candidate SoA buffers of size `naconmax` (`dist, pos, nrm, geom(vec2i), flex(vec2i), elem(vec2i), vert(vec2i), worldid, ncand`), sort keys `2·naconmax` int64, EPA buffers `capacity = naccdmax` (if any mesh/ellipsoid/selfcollide/multi-flex/1D/3D flex) with `10 + 2·ccd_iterations` vertices and `6 + MJ_MAX_EPAFACES·ccd_iterations` faces; FPS buffers sized `nworld·(ngeom·nflex + nflex²)` groups × `_FPS_BLOCK_SIZE = 64` if self-collide or `nflex > 1`.
3. SAP sort once if self-collide or `nflex > 1` (`_run_flex_sap_sort`).
4. `_flex_geom_collision` → plane candidates, element–geom candidates, `_filter_and_write_contacts(enable_fps=False)`.
5. `_flex_sap_collision(is_self=True)` then `(is_self=False)`, each: SAP sweep → `_flex_narrowphase` → `_filter_and_write_contacts(enable_fps=True)`.

**Plane–flex** (`_flex_plane_narrowphase`, dim `(nworld, nflexvert)`): per vertex, loop over all geoms of type PLANE; bitmask `(g_contype & f_conaffinity) || (f_contype & g_conaffinity)`; `margin = geom_margin + flex_margin` (**no gap, no `mj_assignMargin`**); coarse whole-flex-AABB vs plane test (`dist_center − proj_half > margin` → skip); `dist = (v − p)·n − r`; accept if **`dist < margin`** (C: `dist <= margin`, with margin incl. gap); `pos = v − n(dist/2 + r)`. No check whether the plane's body has dofs (C pushes plane–flex in midphase only for dof-less bodies; in all-to-all for any).

**Element–geom** (`_flex_narrowphase_elem_detect`, dim `(nworld, nflexelem)`): skip inactive dim-3 elements. Element geometry: dim 1 → capsule (`centroid`, axis `(t1 − t2)/|·|` (or +z if `< MINVAL`), half length, radius `r`; frame from `orthogonals`); dim 2 → `GeomType.TRIANGLE` (vertices packed in `rot` rows), `r_elem` = max vertex distance to centroid; dim 3 → `GeomType.FLEX` (4 vertices in `polyvert`, `margin = 2r`). Element AABB inflated by r. Loop over **all geoms** of type SPHERE/CAPSULE/BOX/CYLINDER/ELLIPSOID/MESH: bitmask; same-body skip (any element vertex body == geom body, body ≥ 0); `margin = geom_margin + flex_margin`; stage 1: geom world AABB (from `geom_aabb` rotated with |R|, inflated by margin) vs whole-flex AABB; stage 2: vs element AABB. Then:
- MESH: `_collide_mesh_convex` (CCD slot via `atomic_add(nccd)`; bounding-sphere gate `|c2 − c_mesh| <= |size| + rbound + margin + r + 0.04`; `ccd(tolerance, margin + r, ccd_iterations, ...)`; accept `dist < margin + r`; normal from witness points (`w1 − w2` if penetrating else `w2 − w1`); **polygon-normal snapping heuristic**: find the mesh polygon adjacent to the support vertex with minimal plane distance; if `≤ 0.005` and the witness point is inside its first triangle with tolerance `0.05` (barycentric), use the polygon normal (sign aligned with GJK normal); candidate dist `= dist − r`, `pos = 0.5(w1 + w2)`). Not in C.
- dim 1 & SPHERE/CAPSULE/BOX: `collision_primitive_core.sphere_capsule / capsule_capsule / capsule_box(−normals)`, accept `dist < margin` (strict).
- dim 2 & SPHERE/CAPSULE/BOX/CYLINDER: `sphere_triangle`, `capsule_triangle` (5 slots), `box_triangle` (11 slots), `cylinder_triangle` (≤2). Ports of the C functions, except: `sphere_triangle` finds the closest point by plane projection + 2D sign test + nearest of three edge projections (strict `<` comparisons), with an early-out `|plane distance| > margin + rs + rt` — equal to `mjc_triWeights` except at ties/degenerate triangles (`normalize(0)` normal); capsule guard `1/max(MINVAL, 4h²)`; **cylinder–triangle is a Warp-only analytic routine** (C uses CCD via `mjc_ConvexElem`): vertex-vs-axis/cap/rim tests then edge–axis segment tests, max 2 contacts.
- Otherwise (ELLIPSOID; CYLINDER vs dim 1; any geom vs dim 3): CCD (`atomic_add(nccd)`, overflow → `OverflowType.CCD`), gate `|centroid − p| <= |size| + r_elem + margin + r + 0.04`, cutoff `margin + r` (dim 2) or `margin` (dims 1, 3 — radius carried in capsule size / FLEX margin `2r`), `cand_dist = dist − r` (dim 2) else `dist`, accept `< margin`, `pos = 0.5(w1 + w2)` minus `0.5 r n` for dim 2. **No `mjc_fixNormal`**.

**Self / flex–flex broadphase (SAP)**: `_flex_sap_project` dim `(nworld, nflexelem)`: element AABB from vertices inflated by `r + margin + gap` (C: radius only); projection onto fixed direction `normalize(0.5935, 0.7790, 0.1235)` (C: largest axis of flex root box); inactive elements get `±MJ_MAXVAL`. `segmented_sort_pairs` per world, `sap_range`, `array_scan`. `_flex_sap_sweep(is_self, …)`: grid-stride over work packages (`nsweep = 5·nworld·nflexelem` threads); self: require same flex, `flex_selfcollide != 0` (**mode ignored**), contype&conaffinity of that flex; flex–flex: different flexes, whole-flex AABB overlap, cross bitmask. Element AABB overlap. Self: `_exclude_self_collision` excludes pairs **sharing a vertex index** or a vertex body (C: only shared body ≥ 0); flex–flex: shared-body test. Optional SAT prefilter (`ENABLE_SAT_PREFILTER = True`) for triangle–triangle: `_triangle_sat_separated` over 44 axes (2 face normals, 9 edge×edge, 6 in-plane edge normals, 9 vertex–vertex, 18 vertex–edge), separated if `gap² > cutoff²·|axis|²` with `|axis|² > 1e-12`; `cutoff = 2 r1` (self) or `r1 + r2 + margins + gaps`. Pairs `(min elem, max elem)` appended with `atomic_add(ncollision)` into `ctx.collision_pair` (capacity `naconmax`, overflow → `OverflowType.BROADPHASE`).

**Element–element narrowphase** (`_flex_narrowphase`, dim `naconmax`): skip inactive; `mix_margin = 0` (self) or `margins + gaps`; both dim 1 → `capsule_capsule`, accept `d <= mix_margin`; else (requires `pairid < naccdmax`, else CCD overflow) vertices copied to `workspace_verts[8·pairid ...]`, both as MESH-type `Geom` with `margin = 2 r_i + mix_margin` (support inflated by `r_i + mix_margin/2`), `ccd(tol, cutoff = r1 + r2 + mix_margin, ...)`; accept `dist < 0` (inflated); `phys_dist = dist + mix_margin`, `pos = 0.5(w1 + w2)`, `normal = normalize(w1 − w2)`.

**Filtering and writing** (`_filter_and_write_contacts`):
1. `_compute_filter_key`: group = `geom·nflex + flex` (geom–flex) or `ngeom·nflex + f1·nflex + f2` (flex–flex, `f1 <= f2`); `group_id = worldid·(ngeom·nflex + nflex²) + group`; key = `((group_id << 1) | is_self) << 32 | clamp(int64(1e6·(x+y+z)/√3) + 1e8, 0, 2³¹−1)`; unused slots get `INT64_MAX`. `radix_sort_pairs`.
2. `_filter_flex_candidates_sorted`: **proximity deduplication** with `_FLEX_CONTACT_DEDUP_TOLERANCE = 1e-3` m: a candidate is dropped if a same-group neighbour (scanned while spatial key within `ceil(1e-3·1e6) + 1`) lies within 1 mm and is deeper (or equal depth and lower candidate index). Applied to **all** flex contacts (including geom–flex). C has no such dedup.
3. FPS (`enable_fps`, self/flex–flex only, and only if global `ncand > 50` and the group's active count `> 50`): group starts via scan; seed = minimum dist with tie-break `_tie_break_fps` (smaller elem[0], then elem[1], then candidate index); then up to `MJ_MAXCONPAIR − 1` iterations (CUDA-graph `capture_while` if `opt.graph_conditional`, else unrolled 49 launches) selecting the max of **Euclidean** min-distance (C: squared — same ordering), stopping a group when max distance `<= 0`. Block-parallel reductions with 64 threads per group. Applied regardless of `flexReducible` (C reduces only rigid/interpolated flexes, which Warp does not support → **C would keep all self/flex–flex contacts**). Geom–flex groups: no FPS (C: FPS only for reducible flexes, i.e. none in Warp-supported models).
4. `_write_filtered_contacts`: params via `_mix_flex_contact_params` (priority → higher wins; equal → condim max, solmix mix `a/(a+b)` with MJ_MINVAL rules, solref mixed if both positive else elementwise min, solimp mixed, friction max; friction unpacked to 5D with `MJ_MINMU` floor); margin = `geom_margin + flex_margin` or `0` (self) or `flex_margin1 + flex_margin2`; element–vertex contacts in a dim-3 flex get condim 1; reject `cand_dist >= margin`; `includemargin = margin − gap` for plane contacts else `margin`; `solreffriction = solref`; contact type `PASSIVE` if a participating flex has `flex_passive && !interp && dim >= 2`, integrator is DISCRETE, and every non-flex side is a static body; else CONSTRAINT; `atomic_add(nacon)`, overflow → NARROWPHASE flag.

Precision float32; all buffers `[naconmax]` global across worlds; candidate/contact order depends on atomics (nondeterministic). No `<exclude>`/`<pair>` handling for flex in these kernels; `mj_assignMargin` (`o_margin` override) not applied.

### 9.3 MJX (JAX)

`mjx/_src/collision_sdf.py` does **not** implement SDF geoms; it reuses the SDF optimization idea for primitive pairs without analytic colliders: sphere–ellipsoid, sphere–cylinder, capsule–ellipsoid, capsule–cylinder (2 starts: `mid ± axis·half_len`), ellipsoid–ellipsoid, ellipsoid–cylinder, cylinder–cylinder (4 starts `mid ± r·basis1/2`, `r = max radius`). Start `x0 = midpoint of centers` (in g2 frame). Objective = COLLISION clearance `d1 + d2 + |max(d1,d2)|`, 10 iterations of `_gradient_step`: gradient by `jax.grad` (cylinder has a `custom_jvp` with the C-like singularity-free gradient), line search = evaluate 10 step sizes `geomspace(1e-4, 2)` and take the argmin (not backtracking). No intersection refinement. Output: `dist = d1(x) + d2(x)` (not `max`), normal `= ∇d1 − ∇d2` (unnormalized, then `make_frame`), position = optimized point (no half-distance offset). Capsule SDF via clamped segment projection with `1e-12` guard; ellipsoid `k1 + (k1==0)·1e-12`. `make_condim` budgets `sdf_initpoints` contacts for SDF-typed pairs but there is no collision function for them. MJX has no flex collision.

---

## 10. Reproduction pitfalls / parity notes

1. **Quaternion round trip**: all C SDF functions build relative transforms via `mju_mat2Quat` → `mju_mulPose` → `mju_quat2Mat`; using `xmat` directly changes low bits and, through iterative descent, can change contacts.
2. **Halton indexing**: `mjc_SDF` uses indices `0..sdf_initpoints−1` (first sample = AABB min corner, bases 2,3,5); `processSdfCorners` uses `1..nstartpts` (bases 2,3) with the `u+v>1` reflection.
3. **Line search details**: first trial step is `alpha=1` (initial 2 halved before use); Armijo threshold halves with alpha; termination `alpha <= 1e-4`; early stop restores `x0` when the best trial is worse than `dist0` (strict `<` in C: `dist0 < dist`). Frank–Wolfe has no line search and step `2/(k+2)`.
4. **Gradients of composite SDFs re-evaluate distances**; tie rules: INTERSECTION/COLLISION choose `id[0]` (the SDF g2 in `mjc_SDF`) only if `A > B` strictly; COLLISION sign is −1 when `max(A,B) == 0`.
5. **Normal/position convention**: `flipNormal=0` (mesh–SDF, SDF–SDF): `n = −∇`, `pos = x + 0.5·dist·n`; `flipNormal=1` (flex–SDF): `n = +∇`, `pos = x − 0.5·dist·n`. Both place `pos` half a penetration from the sampled point toward the other body's surface.
6. **SDF margin**: C SDF colliders ignore margin entirely (contacts iff `dist <= 0`); duplicates within `1e-15` removed; degenerate gradients (norm `< 1e-15`) skip the contact but still register the point.
7. **50-candidate cap is order dependent**: `processSdfCorners` stops at 50 total candidates and `traverseBVH` visits child 1 before child 0, so which faces contribute depends on BVH layout; the FPS branch in `mjc_MeshSDF`/`mjc_FlexSDF` (BVH path) is unreachable. Flex–SDF without BVH appends contacts in element order until 50.
8. **Buffer sizing hazard**: `mj_maxContact` returns `sdf_initpoints` for *any* SDF pair, while `mjc_MeshSDF` can emit up to 50 contacts; with `sdf_initpoints < 50` the narrowphase raises "collision function returned %d contacts" (or worse) if more are produced. Reproduce the check semantics, not just the happy path.
9. **mesh_vert is float32** in mjModel; MeshSDF converts each vertex to double before transforming.
10. **Octree evaluation**: exact trilinear corner ordering (bit0 x, bit1 y, bit2 z), `coord < .5` child selection, eps 1e-8 containment, projection eps 1e-6 inside the box with exterior distance added only if `> 0`, FD step 1e-8 outside. Hanging-node coefficients are parent averages; coefficients are signed distances to the original mesh via pseudonormals; root box is 1.1× mesh AABB; depth default 6.
11. **Plugin gradients** (bolt/bowl/gear/nut) are forward differences with step 1e-8 in double; a float32 port (Warp uses 1e-6) will differ substantially near the thread/teeth discontinuities.
12. **Flex primitives**: thresholds are inclusive in C (`dst > rbound` rejects; equality accepted), Warp uses strict `<` in several places; C box–triangle tests vertices against the max-penetration face even when outside; capsule–triangle only uses vertices strictly interior in axis parameter (`mjMINVAL < t < 1 − mjMINVAL`); capsule–box dim-1 normals are negated.
13. **Flex margins**: self-collision forces margin=0 and gap=0 (both in pair filtering and in `mjc_ElemElem`); geom/elem detection uses `margin + gap`, `includemargin` excludes gap; `mj_assignMargin` applies `o_margin` when override is enabled. Warp omits gap from detection and does not apply overrides.
14. **Contact reduction**: C applies FPS (squared distances, deepest seed, first-wins ties) only to reducible (rigid/interpolated) flex groups; normal deformable flexes keep every contact. Warp additionally deduplicates within 1 mm and applies FPS to every self/flex–flex group with > 50 candidates → contact sets differ for dense cloth/soft bodies.
15. **Interpolated self-collision exclusion** depends on `flex_vert0` rounding: use the `1e-6·cellnum` tolerance and the "extrapolating active cell" override exactly; elements whose first vertices fall in neighbouring cells are never paired in `mj_collideFlexCellPairs`, and when all `cellnum <= 2` self-collision is skipped entirely.
16. **Pair ordering**: C contacts are deterministic (pair sort within trees and cell pairs, sequential appends); GPU ordering is atomic-based — compare contact sets, not sequences.
17. **Self-collision mode selection** (NARROW/BVH/SAP/AUTO) changes the candidate pair set only via filtering (all modes are conservative with the same leaf `filterBox`), but the SAP path uses AAMMs without margin while BVH leaves use `filterBox(..., margin=0)` for self pairs — equivalent; flex–flex pairs use margins+gaps.
18. **mjc_fixNormal** is applied to CCD contacts involving 2D flexes and sphere/capsule/ellipsoid/cylinder geoms in C (native CCD only); Warp does not, and Warp uses an extra mesh polygon-normal heuristic.

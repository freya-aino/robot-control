# 14 — Ray casting, Sleeping, Support utilities, Rendering & bindings

**Source files covered**

| Area | C engine (commit 7ba52b5) | MuJoCo Warp (commit d41a47c) | MJX |
|---|---|---|---|
| Ray casting | `src/engine/engine_ray.c` (+ callers in `engine_sensor.c`, `engine_vis_interact.c`, `engine_collision_continuous.c`; `mju_camPixelRay`, `mju_insidePrimitive` in `engine_util_misc.c`) | `_src/ray.py`, `_src/sensor.py` (rangefinder) | `mjx/_src/ray.py` |
| Sleeping | `src/engine/engine_sleep.c` (+ sleep hooks in `engine_forward.c`, `engine_core_smooth.c`, `engine_collision_driver.c`, `engine_core_constraint.c`, `engine_passive.c`, `engine_sensor.c`, `engine_island.c`, `engine_setconst.c`, `engine_io.c`) | `_src/sleep.py`, `_src/island.py` (sleep relation, DOF compaction), `_src/forward.py`, `_src/collision_driver.py`, `_src/io.py` | not supported |
| Support | `src/engine/engine_support.c` (+ `mj_contactForce`, `mj_objectVelocity/Acceleration`, `mj_jac*`, `mj_bodyChain` in `engine_core_util.c`; `mj_mulJacVec/TVec` in `engine_core_constraint.c`) | `_src/support.py`, parts of `_src/io.py` | `mjx/_src/support.py` |
| Visualization | `engine_vis_init.c`, `engine_vis_interact.c`, `engine_vis_visualize.c` (brief) | — | — |
| Rendering | `src/render/{classic,filament,noop}` (brief), `src/experimental/{filament,usd}` (brief) | `_src/render.py`, `_src/render_util.py`, `_src/bvh.py` | `mjx/_src/render.py`, `render_util.py` (thin wrappers over Warp) |
| Batch / bindings | `src/batch/batch.cc`, `include/mujoco/mjbatch.h`, `python/mujoco/{rollout.py,rollout.cc,batch.py}`, `wasm/`, `unity/` (brief) | — | — |

**Role in the pipeline.** Ray casting is not part of the dynamics proper but feeds the
`rangefinder` sensor (stage POS), continuous-collision tunnelling checks, the contact-zone
`mjSENS_TOUCH` ray test and interactive selection; any reimplementation that reports
`sensordata` must reproduce it bit-for-bit including its culling heuristics. Sleeping
(`mjENBL_SLEEP`) is a *physics-affecting* optimisation: trees whose velocities stay below
`opt.sleep_tolerance` for `mjMINAWAKE` steps are put to sleep (their `qvel`, `qacc` are
zeroed and they are frozen), and almost every pipeline stage then iterates only over awake
bodies/DOFs; the wake rules decide when frozen trees rejoin. Support functions
(`mj_getState/setState`, `mj_applyFT`, `mj_integratePos`, `mj_differentiatePos`,
`mj_contactForce`, `mj_geomDistance`, `mj_nextActivation`, history buffers, …) are used inside
the pipeline (integrators, sensors, actuators, perturbations) and by every rollout/binding
layer. Visualization and rendering are output-only and never feed back into physics (except
`mjv_applyPerturbForce/Pose`, which write `xfrc_applied`/`qpos`/`mocap_*`).

---

## Table of contents

1. Constants used in this chapter
2. Ray casting (C) — utilities, per-geom intersection, mesh BVH, hfield, SDF, flex, skin, `mj_ray`, `mj_multiRay`, callers
3. Sleeping (C) — data model, policies, update, wake rules, sleep rule, per-stage effects, init/reset
4. Support functions (C) — state, inertia helpers, perturbations, geom distance, integration helpers, activation, misc
5. Related core utilities (C) — `mj_contactForce`, `mj_objectVelocity/Acceleration`, Jacobians, `mj_mulJacVec`
6. Visualization (C, brief)
7. Rendering, experimental, batch, bindings (brief)
8. GPU (MuJoCo Warp) implementation — ray, sleep/island, support, io, renderer (BVH + shading)
9. MJX notes
10. Reproduction pitfalls / parity notes

---

## 1. Constants

| Name | Value | Where |
|---|---|---|
| `mjMINVAL` | `1e-15` (double), `1e-15f` single | `mjtype.h` |
| `mjMAXVAL` | `1e10` | `mjtype.h` |
| `mjPI` | `3.14159265358979323846` | `mjtype.h` |
| `mjNGROUP` | `6` | `mjvisualize.h` |
| `mjMAXTREEDEPTH` | `50` (BVH traversal stack size) | `mjmodel.h` |
| `mjMINAWAKE` | `10` (min. steps under tolerance before sleeping) | `mjmodel.h` |
| `mjMAXCONPAIR` | `50` | `mjmodel.h` |
| `opt.sleep_tolerance` default | `1e-3` | `engine_init.c` |
| `kAwake` (tree_asleep of fully awake tree) | `-(1+mjMINAWAKE) = -11` | `engine_sleep.c`, `engine_io.c` |
| SDF ray-march: `kMinDist` | `1e-7`, max iterations `40`, escape distance `1e6` | `mj_raySdf` |
| Warp `RAY_TOL_REL`, `RAY_TOL_ABS` | `1e-6`, `1e-9` | `ray.py` |
| Warp splat constants | `_MAX_SPLAT_HITS=32`, `_MIN_SPLAT_TRANSMITTANCE=0.005`, `_MIN_SPLAT_ALPHA=1/255`, `SPLAT_MIN_RESPONSE=0.01` | `render.py`, `bvh.py` |
| Warp shading constants | `MAX_SHININESS=128`, `DEFAULT_MAT_SHININESS_EXPONENT=64`, `DEFAULT_MAT_SPECULAR=0.5`, `DEFAULT_MAT_EMISSION=0`, `NO_LIGHT_AMBIENT_FALLBACK=0.3` | `render.py` |
| `mjVERSION` / string | `3017001` / `"3.17.1"` | `engine_support.c` |

`mjtSleepState`: `mjS_STATIC=-1`, `mjS_ASLEEP=0`, `mjS_AWAKE=1`.
`mjtSleepPolicy`: `mjSLEEP_AUTO=0`, `AUTO_NEVER=1`, `AUTO_ALLOWED=2`, `NEVER=3`, `ALLOWED=4`, `INIT=5`.
`mjtState` bits (order matters for state vectors): `TIME=1<<0, QPOS=1<<1, QVEL=1<<2, ACT=1<<3, HISTORY=1<<4, WARMSTART=1<<5, CTRL=1<<6, QFRC_APPLIED=1<<7, XFRC_APPLIED=1<<8, EQ_ACTIVE=1<<9, MOCAP_POS=1<<10, MOCAP_QUAT=1<<11, USERDATA=1<<12, PLUGIN=1<<13, PLASTIC=1<<14`, `mjNSTATE=15`.
Composites: `PHYSICS = QPOS|QVEL|ACT|HISTORY|PLASTIC`; `FULLPHYSICS = TIME|PHYSICS|PLUGIN`;
`USER = CTRL|QFRC_APPLIED|XFRC_APPLIED|EQ_ACTIVE|MOCAP_POS|MOCAP_QUAT|USERDATA`;
`INTEGRATION = FULLPHYSICS|USER|WARMSTART`.

---

## 2. Ray casting (C, `engine_ray.c`)

### 2.1 Conventions

* A ray is `p(x) = pnt + x·vec`, `x ≥ 0`. **`vec` is NOT normalised** by `mj_ray`; the
  returned distance `x` is in units of `|vec|` (i.e. it is the ray parameter). The only
  exception is `mj_raySdf`, which normalises the local direction and returns Euclidean
  distance (see 2.6). Callers (rangefinder, `mjv_select`) pass unit vectors, so this only
  matters for direct API users.
* "No intersection" is encoded as `-1`. Normals (when requested) are unit, world-frame;
  zero vector on miss.
* All functions take `normal` nullable; when NULL, no normal work is done.

### 2.2 Utilities

**`ray_map(pos, mat, pnt, vec) → (lpnt, lvec)`** — express ray in a local frame:
`lpnt = matᵀ(pnt − pos)`, `lvec = matᵀ vec` (row-major `mat`, explicit expansion
`lpnt[0] = mat[0]*dif[0] + mat[3]*dif[1] + mat[6]*dif[2]` etc.).

**`longitude(v) = atan2(v[1], v[0])`**, **`latitude(v) = atan2(sqrt(v0²+v1²), v[2])`**
(polar angle from +z, in `[0, π]`).

**`ray_eliminate(m, d, g, geomgroup, flg_static, bodyexclude)`** returns 1 (skip) if, in order:
1. `geom_bodyid[g] == bodyexclude`;
2. `geom_matid[g] < 0 && geom_rgba[4g+3] == 0` (invisible geom, exact float compare);
3. `geom_matid[g] ≥ 0 && mat_rgba[4*matid+3] == 0` (invisible material; note the geom's own rgba alpha is then ignored);
4. `!flg_static && body_weldid[geom_bodyid[g]] == 0` (static geom excluded);
5. if `geomgroup == NULL` → keep (return 0);
6. `groupid = clamp(geom_group[g], 0, mjNGROUP−1)`; skip iff `geomgroup[groupid] == 0`.

**`ray_quad(a, b, c, x[2])`** — solves `a x² + 2 b x + c = 0`:
```
det = b*b - a*c
if (det < 0 || a < mjMINVAL) { x = {-1,-1}; return -1; }
det = sqrt(det); x0 = (-b-det)/a; x1 = (-b+det)/a     // x0 <= x1
return x0 >= 0 ? x0 : (x1 >= 0 ? x1 : -1)
```
Both roots are always written (the capsule caps use both).

**`mju_rayTriangle(v[3][3], lpnt, lvec, b0, b1, normal)`** (public). `b0, b1` are an
orthonormal basis of the plane ⟂ `lvec` (computed by caller, see below).
1. `dif_i = v_i − lpnt`; `planar_i = (b0·dif_i, b1·dif_i)`.
2. Quick reject if all three `planar_i[k] > 0` or all `< 0` for k = 0 or k = 1 (strict).
3. Solve `A t = b` with `A = [[p0x−p2x, p1x−p2x],[p0y−p2y, p1y−p2y]]`, `b = −p2`;
   `det = A00 A11 − A01 A10`; reject if `|det| < mjMINVAL`;
   `t0 = (A11 b0 − A01 b1)/det`, `t1 = (−A10 b0 + A00 b1)/det`; reject if
   `t0 < 0 || t1 < 0 || t0+t1 > 1`.
4. `nrm = (v0−v2) × (v1−v2)`; `denom = lvec·nrm`; reject if `|denom| < mjMINVAL`.
5. `x = −((lpnt − v2)·nrm)/denom` — **can be negative**; callers test `sol ≥ 0`.
6. `normal = normalize(nrm)` (mju_normalize3: if norm < mjMINVAL → (1,0,0)); the normal is
   **not** flipped toward the ray; its sign follows triangle winding.

**Normal-plane basis (used by hfield, mesh tree, flex, skin; identical code each time):**
```
b0 = (1,1,1)
if |v0| >= |v1| && |v0| >= |v2|:  b0[0] = 0
elif |v1| >= |v2|:                b0[1] = 0
else:                             b0[2] = 0
b1 = b0 - (v·b0)/(v·v) * v ; normalize(b1)
b0 = b1 × v ; normalize(b0)
```
(`v` = local ray direction, or world `vec` for flex/skin.)

**`mju_raySlab(aabb[6], xpos, xmat, pnt, vec)`** — slab test vs an oriented box given as
`aabb = (center[3], halfsize[3])` in the frame `(xpos, xmat)`:
```
tmin = 0, tmax = +INF
min = c - h, max = c + h; (src, dir) = ray_map(xpos, xmat, pnt, vec)
invdir = 1/dir  (IEEE: ±inf for zero components)
for d in 0..2: t1=(min-src)*invdir; t2=(max-src)*invdir;
               tmin = max(tmin, min(t1,t2)); tmax = min(tmax, max(t1,t2))
return tmin <= tmax
```
Comparisons are written as ternaries (`tmin > minval ? tmin : minval`) so a NaN
(0·inf when `src` lies exactly on a slab plane with `dir=0`) propagates as "keep current
value" — replicate exactly.

**`point_in_box(aabb, xpos, xmat, pnt)`** — `p = xmatᵀ(pnt−xpos) − c`; inside iff
`|p_j| ≤ h_j` for all j.

### 2.3 Primitive intersections (`mju_rayGeom`)

`mju_rayGeom(pos, mat, size, pnt, vec, type, normal)` dispatches:
PLANE→`ray_plane`, SPHERE→`ray_sphere(size0²)`, CAPSULE, ELLIPSOID, CYLINDER, BOX; any
other type → `mjERROR`. Mesh/hfield/SDF are handled by `mj_ray` directly.

**Plane** (`ray_plane`): local frame; reject if `lvec_z > −mjMINVAL` (back-face / parallel:
rays from below never hit); `x = −lpnt_z / lvec_z`; reject `x < 0`;
`p = lpnt_xy + x lvec_xy`; accept iff `(size0 ≤ 0 || |p0| ≤ size0) && (size1 ≤ 0 || |p1| ≤ size1)`
(non-positive size = infinite in that axis). Normal = third column of `mat`
(`mat[2], mat[5], mat[8]`).

**Sphere** (`ray_sphere(pos, mat, r², …)`, `mat` unused): `dif = pnt−pos`,
`a = vec·vec`, `b = vec·dif`, `c = dif·dif − r²`, `x = ray_quad`. Origin inside → returns
the far root (x0<0≤x1). Normal = `normalize(pnt + x vec − pos)`.

**Capsule** (`size = (r, halfheight)`):
1. Bounding sphere radius `r + h` (`ssz = (r+h)²`) via `ray_sphere`; miss → −1.
2. Local frame. Cylinder side: `a = lvx²+lvy²`, `b = lvx lpx + lvy lpy`,
   `c = lpx²+lpy² − r²`; `sol = ray_quad(...)`; accept if `sol ≥ 0 && |lpz + sol lvz| ≤ h`
   (`type=0`).
3. Top cap sphere centred at `(0,0,+h)`: `a = |lvec|²` (reused for bottom), `ldif = lpnt − (0,0,h)`;
   `ray_quad`; for **both** roots `xx[i] ≥ 0` with `lpz + xx[i] lvz ≥ h`, keep if smaller (`type=1`).
4. Bottom cap at `(0,0,−h)`: `ldif.z = lpz + h`; same with `≤ −h` (`type=−1`).
5. Normal: `n = (lpx + x lvx, lpy + x lvy, type==0 ? 0 : lpz + x lvz − h·type)`, normalise, rotate by `mat`.

**Ellipsoid**: `s_i = 1/size_i²`; `a = Σ s_i lv_i²`, `b = Σ s_i lv_i lp_i`,
`c = Σ s_i lp_i² − 1`; `x = ray_quad`; normal = `normalize(s ⊙ (lpnt + x lvec))` rotated.

**Cylinder** (`size = (r, h)`):
1. Bounding sphere `ssz = r² + h²`.
2. Flat caps, only if `|lvz| > mjMINVAL`: for `side = −1, +1` (in that order):
   `sol = (side·h − lpz)/lvz`; if `sol ≥ 0` and `p0²+p1² ≤ r²` → keep if `x<0 || sol<x` (`type=side`).
3. Round side as capsule step 2; accepted if strictly smaller (`sol < x`) → `type=0`.
4. Normal: side → normalised `(px, py, 0)`; caps → `(0,0,type)`; rotate.

**Box** (`ray_box(pos, mat, size, pnt, vec, all[6], normal)`):
1. `all[0..5] = −1` (if given); bounding sphere `ssz = |size|²`.
2. Local frame; for axis `i = 0,1,2` with `|lvec_i| > mjMINVAL`, for `side = −1, +1`:
   `sol = (side·size_i − lp_i)/lv_i`; if `sol ≥ 0`: with face axes `iface = {{1,2},{0,2},{0,1}}`,
   accept iff `|p_a| ≤ size_a && |p_b| ≤ size_b`; update min (`face_axis=i`, `face_side=side`)
   on strict `<`; **always** write `all[2i + (side+1)/2] = sol` for accepted faces.
3. Normal = `mat · (e_{face_axis} · face_side)`.
Order of tests matters for ties: on exact ties the first (lowest axis, `side=−1` first) wins.

### 2.4 Heightfield (`mj_rayHfield`, public)

Inputs: `hid = geom_dataid[g]`, `nrow, ncol`, `size = hfield_size[4 hid]` = (sx, sy, sz_top, sz_base),
`data = hfield_data + hfield_adr[hid]` (float, row-major, values in [0,1]).
Geom frame `(xpos, xmat) = (geom_xpos, geom_xmat)`; `zaxis = (xmat[2], xmat[5], xmat[8])`.

1. **Base box**: `base_size = (sx, sy, 0.5·sz_base)`, `base_pos = xpos − zaxis·0.5·sz_base`.
   `x = ray_box(base…)` (with normal into `normal_base`).
2. **Top box**: `top_size = (sx, sy, 0.5·sz_top)`, `top_pos = xpos + zaxis·0.5·sz_top`;
   `top_intersect = ray_box(top…, all, NULL)`. If `< 0` → return base result.
3. Local ray (w.r.t. `xpos, xmat`), normal-plane basis `b0, b1`.
4. Segment inside top box: `seg = (0, top_intersect)`; for i in 0..5, if `all[i] > seg[1]`
   then `seg = (top_intersect, all[i])` (i.e. entry→exit if origin is outside; 0→exit if
   inside; later larger `all[i]` overwrite).
5. Grid spacing `dx = 2 sx/(ncol−1)`, `dy = 2 sy/(nrow−1)`;
   `SX[i] = (lpx + seg[i] lvx + sx)/dx`, `SY[i] = (lpy + seg[i] lvy + sy)/dy`;
   `cmin = max(0, floor(min SX)−1)`, `cmax = min(ncol−1, ceil(max SX)+1)`, same for rows
   (casts `(int)mju_floor(...)`).
6. `normal_local` initialised to base-box normal in local frame (if base hit).
7. For `r ∈ [rmin, rmax)`, `c ∈ [cmin, cmax)`: two triangles (vertex `(c,r)` →
   `(dx·c − sx, dy·r − sy, data[r·ncol+c]·sz_top)`):
   * T1 = `{(c,r), (c+1,r), (c+1,r+1)}`
   * T2 = `{(c,r), (c+1,r+1), (c,r+1)}`
   each via `mju_rayTriangle`; keep strictly smaller `sol ≥ 0`.
8. **Top-box side walls** `i = 0..3` (x−, x+, y−, y+): if `all[i] ≥ 0 && (all[i] < x || x < 0)`:
   `z = (lpz + all[i] lvz)/sz_top`;
   * `i < 2`: `y = (lpy + all[i] lvy + sy)/dy`; `y0 = clamp(floor(y), 0, nrow−2)`;
     column `= (i==1 ? ncol−1 : 0)`; `z0 = data[round(y0)·ncol + col]`, `z1 = data[round(y0+1)·ncol + col]`.
   * `i ≥ 2`: `y = (lpx + all[i] lvx + sx)/dx`; `y0 = clamp(floor(y), 0, ncol−2)`;
     row offset `(i==3 ? (nrow−1)·ncol : 0)`; `z0 = data[round(y0) + off]`, `z1 = data[round(y0+1)+off]`.
   * Hit if `z < z0 (y0+1−y) + z1 (y−y0)` → `x = all[i]`, `normal_local = ∓e_x / ∓e_y`
     (`i=0: −x, 1: +x, 2: −y, 3: +y`).
   Note the side-wall test is evaluated after triangles and can override a triangle hit
   (it only requires `all[i] < x`).
9. Rotate `normal_local` by `xmat`.

### 2.5 Mesh (`mj_rayMesh`, `mju_rayTree`)

`mj_rayMesh`: error unless `geom_type == mjGEOM_MESH`; **bounding-box prefilter**
`ray_box(geom_xpos, geom_xmat, geom_size, …) < 0 → −1` (for meshes `geom_size` holds the
half-extents of the mesh AABB in the geom frame); then `mju_rayTree`.

`mju_rayTree(m, d, g, pnt, vec, normal)` — iterative DFS over the mesh BVH:
* `meshid = geom_dataid[g]`, `bvhadr = mesh_bvhadr[meshid]`; arrays `bvh_nodeid`,
  `bvh_aabb` (6 per node, center+half, mesh/geom frame), `bvh_child` (2 per node, −1 = none).
* Stack `int stack[mjMAXTREEDEPTH]`, start with node 0; error "BVH stack depth exceeded"
  if a push would exceed 50.
* Local ray + basis `b0,b1` (geom frame).
* Pop node; `mju_raySlab(bvh_aabb+6node, geom_xpos, geom_xmat, pnt, vec)` (slab test in
  world→local each time); skip on miss.
* Leaf (`nodeid[node] ≠ −1`): `face = nodeid + mesh_faceadr[meshid]`; vertices
  `mesh_vert[3(mesh_face[3face+k] + mesh_vertadr)]` converted float→mjtNum;
  `mju_rayTriangle`; keep strictly smaller `sol ≥ 0`; if `m->vis.global.bvactive`, mark
  `d->bvh_active[node+bvhadr] = 1` on improvement.
* Internal: mark active (visualisation), push `child[2node+0]` then `child[2node+1]`
  (so child 1 is popped first). No front-to-back ordering, no early termination by
  current best `x` — every intersected leaf is tested; result = global min, ties go to the
  first-tested triangle (strict `<`).
* Normal rotated by `geom_xmat`. **No back-face culling** in C (`mj_ray` returns
  exit hits from inside a mesh). Warp's BVH path culls back-faces (see §8).

### 2.6 SDF (`mj_raySdf`, static)

1. Prefilter `ray_box(geom_xpos, geom_xmat, geom_size)` (box of SDF bounds).
2. `instance = geom_plugin[g]`; if −1 the SDF is a mesh-SDF (`mjc_getSDF` NULL, instance =
   `geom_dataid`). Build `mjSDF{id=&instance, type=mjSDFTYPE_SINGLE, plugin, geomtype=mjGEOM_SDF}`.
   Plugin `reset` called before marching (and after a failed march).
3. Local ray, **`lvec` normalised**.
4. Sphere tracing: `t = 0`; for `i < 40`: `p = lpnt + t lvec`; `dist = |mjc_distance(p)|`;
   `t += dist`; if `dist < 1e-7` → hit: normal = `mjc_gradient` at `lpnt + t lvec`,
   normalised, rotated; return `t`. If `dist > 1e6` → break. Return −1 after loop.
   Because `|·|` is taken, marching from inside an SDF moves forward by |d| and may "hit"
   the inner surface.

### 2.7 Flex (`mj_rayFlex`) and skin (`mju_raySkin`) — selection only

`mj_rayFlex(m, d, flex_layer, flg_vert, flg_edge, flg_face, flg_skin, f, pnt, vec, vertid, normal)`:
1. World AABB of `flexvert_xpos` of flex f (first vertex initialises), inflated by
   `flex_radius[f]`; prefilter with `ray_box` (identity mat).
2. Basis from **world** `vec`.
3. If `flg_edge || (dim>1 && flg_skin)`: each edge e as capsule `(r=flex_radius,
   h=0.5|v2−v1|)`, centre mid-point, `mat = quat2Mat(quatZ2Vec(v2−v1))`; on improvement
   `vertid` = nearer endpoint (`dist(v1) < dist(v2)` → v1).
   Else if `flg_vert` (and not `dim>1 && skin`): each vertex as sphere radius `flex_radius`.
4. If `dim>1 && (flg_face || flg_skin)`: for each element, skip 3-D elements with
   `(flg_skin && elayer>0) || (!flg_skin && elayer != flex_layer)`; triangles: dim 2 → `{0,1,2}`,
   dim 3 → `{012, 013, 023, 123}`; `mju_rayTriangle` with world `pnt, vec`; `vertid` =
   nearest of 3 vertices (`≤` tie rules: v0 if `d0≤d1 && d0≤d2`, else v1 if `d1≤d2`, else v2).
Not used by physics (only `mjv_select`).

`mju_raySkin(nface, nvert, face, vert(float), pnt, vec, vertid)`: AABB prefilter of the skin
vertices, then brute force over all faces with `mju_rayTriangle` (no normal), `vertid` =
nearest vertex with strict `<` starting from vertex 0.

### 2.8 `mj_ray` (main entry, public)

```
mjtNum mj_ray(m, d, pnt, vec, geomgroup, flg_static, bodyexclude, geomid*, normal*)
  if |vec| < mjMINVAL: mjERROR("vector length is too small")
  dist = -1; *geomid = -1; normal = 0
  for g in 0..ngeom-1:
    if ray_eliminate(g): continue
    type MESH  -> mj_rayMesh;  HFIELD -> mj_rayHfield;  SDF -> mj_raySdf;
    else       -> mju_rayGeom(geom_xpos, geom_xmat, geom_size, type)
    if newdist >= 0 && (newdist < dist || dist < 0): update dist, geomid, normal
  return dist
```
Brute force, O(ngeom); ties → lowest geom id. Uses `geom_xpos/xmat` as stored in `mjData`
(including stale poses of sleeping bodies — which are exact because sleeping bodies don't move).

### 2.9 `mj_multiRay` (public) and helpers

Used by the camera rangefinder (perspective). All rays share origin `pnt`.

`mju_multiRayPrepare(m, d, pnt, ray_xmat(must be NULL), geomgroup, flg_static, bodyexclude, cutoff, geom_ba[4ngeom], geom_eliminate[ngeom])`:
1. `geom_eliminate[g] = ray_eliminate(g)` for all geoms.
2. For each body with `body_bvhadr[b] ≠ −1` (bodies with ≥1 collision geom; see pitfalls),
   for each geom g of the body not eliminated:
   * if `dist(geom_xpos, pnt) > cutoff + geom_rbound[g]` → `geom_eliminate[g]=1`;
   * if `point_in_box(geom_aabb[6g], geom_xpos, geom_xmat, pnt)` → `geom_ba = (−π, 0, π, π)` (no culling);
   * else: for the 8 corners of the geom-frame AABB, rotate+translate to world, take
     `azimuth = longitude(corner−pnt)`, `elevation = latitude(corner−pnt)`; accumulate
     `AABB = (min az, min el, max az, max el)` starting from `(mjMAXVAL, mjMAXVAL, −mjMAXVAL, −mjMAXVAL)`;
     margin `= atan2(max_half, dist(pnt, geom_xpos))` with `max_half = max(aabb[3..5])`,
     applied (subtract from mins, add to maxes) only if `dist > mjMINVAL`;
     if `az_max − az_min > π` **or** `el_max − el_min > π` → `(−π, 0, π, π)`.
   Layout `geom_ba[4g] = {az_min, el_min, az_max, el_max}`.

`mju_singleRay` (static), per ray:
1. `azimuth/elevation` of `vec`.
2. For each body b: if `body_bvhadr[b] ≠ −1`, body sphere test:
   `center = bvh_aabb[6·body_bvhadr] (first 3) + xipos[b]` — **the root-node centre (stored in
   the body inertial frame) is added to `xipos` without rotation** — radius² =
   `|halfsize|²`; `ray_sphere` miss → skip body.
3. For each geom of the body: skip if `geom_eliminate`; if body has a BVH, angular test:
   reject if `el < el_min || el > el_max`; azimuth: `c = (az_min+az_max)/2`,
   `hw = (az_max−az_min)/2`, `diff = az − c` wrapped once into (−π, π] by ±2π; reject if
   `|diff| > hw`.
4. Dispatch as in `mj_ray`; keep strictly smaller.

`mj_multiRay(m, d, pnt, vec[3nray], geomgroup, flg_static, bodyexclude, geomid, dist, normal, nray, cutoff)`:
stack-allocates `geom_ba`, `geom_eliminate`, prepares, then for each ray:
`if vec·vec < mjMINVAL → dist = −1` (note: geomid/normal are NOT written in this case) else
`mju_singleRay`.

### 2.10 Callers of ray functions that affect simulation outputs

* **Rangefinder sensor** (`engine_sensor.c`, stage POS):
  * Site-attached: origin `site_xpos`, direction = site z-axis (`site_xmat[2,5,8]`),
    `mj_ray(m, d, origin, rvec, geomgroup=NULL, flg_static=1, bodyexclude=site_bodyid, …)`;
    normal requested only if dataspec has `mjRAYDATA_NORMAL`.
  * Camera-attached: `mju_camIntrinsics` → `(fx, fy, cx, cy, extent)`; perspective: compute
    all pixel directions with `mju_camPixelRay` and call **`mj_multiRay`** with
    `bodyexclude = cam_bodyid`, `cutoff = mjMAXVAL`, normals if requested; orthographic:
    per-pixel `mj_ray` with origin offsets (normals always requested).
  * `fill_raydata`: fields in order DIST(1), DIR(3), ORIGIN(3), POINT(3), NORMAL(3),
    DEPTH(1) (`mjRAYDATA_SIZE`), each present iff its bit is in dataspec. DIR, POINT,
    NORMAL, DEPTH are zero on miss; DIST = −1 on miss; ORIGIN always written; POINT =
    `origin + dist·dir`; DEPTH = `dist` for site sensors, `−(point − cam_xpos)·cam_z` for
    cameras (`cam_z = (xmat[2], xmat[5], xmat[8])`).
* **`mju_camIntrinsics`**: perspective with `sensorsize ≠ 0`:
  `fx = intrinsic0/sensor0·W`, `fy = intrinsic1/sensor1·H`, `cx = intrinsic2/sensor0·W`,
  `cy = intrinsic3/sensor1·H`; fovy-based: `fx = fy = 0.5/tan(fovy·π/360)·H`, `cx = W/2`,
  `cy = H/2`; orthographic: `fx = cx = W/2`, `fy = cy = H/2`; `extent = cam_fovy`.
* **`mju_camPixelRay`**: `px = col + 0.5 − cx`, `py = row + 0.5 − cy`; perspective:
  origin = cam pos, `dir = normalize(cam_xmat · (px/fx, −py/fy, −1))`; orthographic:
  `dir = −z_cam`, origin = `cam_xpos + cam_xmat·(px/fx·E/2, −py/fy·E/2, 0)`.
* **Touch sensor** zone test: `mju_rayGeom(site pose, site_size, con->pos, conray, site_type)`
  (conray = normalised contact normal × normal force, flipped if the sensor body is body 2).
* **Continuous collision** (`engine_collision_continuous.c`): `mju_rayGeom`/`mj_rayMesh`
  on a segment `pnt + x·vec`, hit iff `0 ≤ x ≤ 1`.
* `mju_insidePrimitive` (used by `mj_insideSite`): sphere `|v|² < r²`; capsule
  `x²+y²+(z−clip(z,−h,h))² < r²`; ellipsoid `Σ p²/s² < 1`; cylinder `|z|<h && x²+y²<r²`;
  box `|p_i| < s_i` all; plane `z < 0`; others 0. Strict inequalities.


---

## 3. Sleeping (C, `engine_sleep.c` + hooks)

### 3.1 Data model

The unit of sleep is the **kinematic tree** (`ntree` trees; `tree_bodyadr/bodynum`,
`tree_dofadr/dofnum` contiguous ranges; static/world-welded bodies have `body_treeid = −1`).
The source of truth is `d->tree_asleep[ntree]` (int):

* `tree_asleep[t] < 0` → **awake**. The value is a countdown: `−(1+mjMINAWAKE) = −11` is
  "fully awake"; each step under tolerance increments it by 1 until `−1` = **ready to sleep**.
* `tree_asleep[t] ≥ 0` → **asleep**; the value is the *index of the next tree* in a cyclic
  linked list ("sleep cycle") linking all trees of the island that fell asleep together. A
  singleton sleeping tree points to itself.

Derived arrays (recomputed by `mj_updateSleep`):

| Field | Size | Meaning |
|---|---|---|
| `tree_awake` | ntree | 1 if `tree_asleep<0`, else 0. **Also used as a "mismatch" flag**: `mj_kinematics1` sets `tree_awake[t]=1` for a sleeping tree whose recomputed pose differs; `mj_wake` then wakes it. |
| `body_awake` | nbody | `mjtSleepState` per body |
| `body_awake_ind`, `nbody_awake` | nbody | indices of bodies with state ≠ ASLEEP (awake **and static**), ascending |
| `parent_awake_ind`, `nparent_awake` | nbody | indices `i ≥ 1` whose parent's state ≠ ASLEEP, ascending |
| `dof_awake_ind`, `nv_awake` | nv | DOFs whose body is dynamic and AWAKE |
| `ntree_awake` | 1 | count of awake trees |

Model fields: `tree_sleep_policy[ntree]` (mjtSleepPolicy), `opt.sleep_tolerance`,
`dof_length[nv]` (velocity weight), `tendon_treenum/treeid[2]`.

`dof_length` (compile-time, `engine_setconst.c:setDofLength`): 1 for translational DOFs
(slide, free-translation); for hinge, ball and free-rotation DOFs = `bodySize(body)`:
max over (joint-anchor−body-COM distances for joints of the body and its children
joints counted for the parent), (`geom_rbound + |geom_xpos − xipos|` for geoms with
rbound>0), (flex edge rest lengths / node distances), clamped below by a minimum (see
setconst chapter). So the sleep criterion compares an approximate linear speed.

### 3.2 Sleep policies (compile time, `mj_setConst` "apply compiler AUTO tree sleep policy")

1. Reset any `AUTO_NEVER`/`AUTO_ALLOWED` back to `AUTO`.
2. **Actuators**: any tree containing the body of an actuator's transmission target
   (joint/jointinparent: `jnt_bodyid`; SO3: site bodies or joint body; site/slidercrank:
   site body and refsite body; body: the body; tendon: every tree touched by a wrap object
   of the tendon) → `AUTO` becomes `AUTO_NEVER`.
3. **Tendons** spanning ≥2 trees with stiffness or damping (`mj_tendonHasStiffness ||
   mj_tendonHasDamping`; damping includes an attached actuator) or spanning >2 trees: all
   their trees `AUTO → AUTO_NEVER`; if any such tree has user policy `ALLOWED` or `INIT` →
   compile error.
4. **Flexes** without edge equality and not `mj_effFlexPossible` (no metric stiffness/passive
   contact): trees of all their vertex/node bodies `AUTO → AUTO_NEVER`.
5. Remaining `AUTO → AUTO_ALLOWED`.

At run time `treeCanSleep` refuses `NEVER` and `AUTO_NEVER`.

### 3.3 `mj_updateSleepInit(m, d, flg_staticawake)` / `mj_updateSleep = …(…, 0)`

```
ntree_awake = Σ (tree_awake[t] = tree_asleep[t] < 0)
for i in 0..nbody-1:
  if body_treeid[i] < 0:                       // static (world-welded or mocap subtree)
     body_awake[i] = body_mocapid[body_rootid[i]] >= 0 ? AWAKE
                   : (flg_staticawake ? AWAKE : STATIC)
  else body_awake[i] = tree_awake[treeid] ? AWAKE : ASLEEP
  if body_awake[i] != ASLEEP: body_awake_ind[nbody_awake++] = i
  if i>0 && body_awake[parent(i)] != ASLEEP: parent_awake_ind[nparent_awake++] = i
for dof i: if body_treeid[dof_bodyid]>=0 && body_awake[dof_bodyid]==AWAKE: dof_awake_ind[nv_awake++]=i
```
Note the world body (i=0) is STATIC (or AWAKE with flg_staticawake) and is always in
`body_awake_ind[0]`; mocap bodies are always AWAKE.

### 3.4 Utilities

* `isSmaller(vec, weight, n, tol)`: running `max = max(max, weight_i·|vec_i|)`, return 0 as
  soon as `max ≥ tol`; i.e. criterion `max_i dof_length_i·|qvel_i| < tol` (strict).
* `treeCanSleep(m, d, t, tol)`: 0 if policy ∈ {NEVER, AUTO_NEVER}; 0 if any byte of
  `xfrc_applied[6·tree_bodyadr … 6·(adr+num)]` is non-zero (`mju_isZeroByte` — note: `-0.0`
  is non-zero bytewise!); 0 if any byte of `qfrc_applied` over the tree DOFs is non-zero;
  then if `tol ≠ 0`: `isSmaller(qvel, dof_length, …, tol)`; if `tol == 0`: `qvel` bytes all zero.
* `mj_sleepCycle(tree_asleep, ntree, i)`: walks the cycle from i, returns the smallest
  tree index in it, −1 on invalid index/over-long walk (> ntree steps).
* `mj_wakeIsland(tree_asleep, ntree, i, wakeval, reason, time)`:
  * if tree i awake: `tree_asleep[i] = min(wakeval, tree_asleep[i])` (lower = "more awake",
    resets the countdown toward −11), returns 0;
  * else walks the cycle starting at i, setting each tree to `wakeval`, returns number woken.
    Errors if the walk does not return to i within ntree steps. Logging (topic
    `mjTOPIC_SLEEP`) only.
* `mj_flexBody(m, con, side)`: representative body of a flex contact side: vertex contact on
  non-interpolated flex → `flex_vertbodyid`; element contact → first element vertex body
  with a tree (else first vertex body); interpolated → first node body with a tree (else
  node 0's body).

### 3.5 Wake rules

All return the number of woken trees; callers re-run `mj_updateSleep` if nonzero.

**`mj_wake(m, d)`** — "user/perturbation" wake, called inside `mj_kinematics` after
`mj_kinematics1`:
* If sleep disabled: if `ntree_awake < ntree` fill `tree_asleep = kAwake` and return the count.
* If some trees asleep: `wakeNonMetricFlex(m, d, flg_wake=1)` — only when
  `!mj_isMetric(m)` (integrator ≠ DISCRETE): for each flex with no edge equality and not
  `mj_effFlexPossible`, for each vertex/node body tree: wake sleeping trees with
  `kAwake`, and reset awake ones to `kAwake`.
* For each sleeping tree: wake (`kAwake`) if `tree_awake[t]` (pose mismatch flagged in
  `mj_kinematics1`) or `!treeCanSleep(m, d, t, tol=0)` (any nonzero applied force or any
  nonzero qvel bit).

**Pose-mismatch detection** (`mj_kinematics1`, sleep enabled): for bodies with
`body_awake == ASLEEP` and `jntnum>0`, the new `xpos, xquat` are compared to stored values
with **exact `==`**; on match the body is skipped (stored `xpos/xquat/xmat` retained),
on mismatch `tree_awake[treeid] = 1` and the new pose is written. STATIC bodies are skipped
entirely (their poses are never recomputed when sleep is enabled — they were computed once
at `mj_resetData` with `flg_staticawake=1`).

**`mj_wakeTendon`** (after `mj_tendon` in `mj_fwdKinematics`): for tendons with
`tendon_treenum ≥ 2`: `is_limit = treenum==2 && tendonLimit(ten_length) > 0` (limited and
`side·(range−length) < margin` on either side); `is_metric = mj_isMetric(m) &&
mj_effTendonPossible(m, i)`; skip if neither. Two trees: if exactly one awake, wake the
other with `wakeval = tree_asleep[awake tree]` (inherits its countdown). >2 trees (metric
only): if any tree in `ten_J` sparsity is awake, `wakeval = min(0, tree_asleep of awake
trees)`, wake every sleeping tree in the sparsity.

**`mj_wakeCollision`** (after `mj_collision` in `mj_fwdPosition`): for each contact: bodies
from geoms or `mj_flexBody`; if one side has no tree: wake the other side's tree iff it is
asleep and the dof-less body is AWAKE (mocap) with `kAwake`; if both trees awake skip; both
asleep → `mjERROR` (cannot happen because broadphase filters such pairs); else wake the
sleeping tree with `wakeval = tree_asleep[awake tree]`. If any tree woke, `mj_updateSleep`
and **`mj_collision` is re-run from scratch** (a full second collision pass).

**`mj_wakeEquality`** (after the collision wake): for active equalities:
* CONNECT/WELD: bodies (or site bodies) → trees; JOINT: joint bodies (id2 may be −1);
  TENDON → `mjERROR("tendon equality does not yet support sleeping")`; FLEX/FLEXSTRAIN: find
  first awake vertex/node tree, then wake the first sleeping tree found (its whole cycle)
  with that tree's `tree_asleep` as wakeval.
* States: tree≥0 → `tree_awake` (0/1 cast to ASLEEP/AWAKE); tree<0 → `body_awake[body]`
  (mocap AWAKE, otherwise STATIC), or STATIC for missing object.
* Neither asleep → skip; any STATIC → skip; same tree → skip; both asleep and in different
  cycles (`mj_sleepCycle` mismatch) → wake both with `kAwake`; one asleep → wake it with `kAwake`.

### 3.6 Sleep rule — `mj_sleep(m, d)` (end of `advanceStart`, every integrator)

```
if !SLEEP: return 0
if nefc && !nisland: return 0                       // constraints but islands disabled
for each awake tree t:
   if treeCanSleep(t, opt.sleep_tolerance): tree_asleep[t] += (tree_asleep[t] < -1)
   else tree_asleep[t] = -(1+mjMINAWAKE)
wakeNonMetricFlex(m, d, flg_wake=0)                 // non-metric constraint-free flexes reset to kAwake
for island i in 0..nisland-1 (island order):
   trees = map_itree2tree[island_itreeadr[i] .. +island_ntree[i]]
   if all tree_asleep == -1: mj_sleepTrees(trees)    // any < -1 blocks; any >= 0 -> mjERROR
for j from (end of last island) to ntree-1:          // unconstrained trees
   t = nisland ? map_itree2tree[j] : j
   if tree_asleep[t] == -1: mj_sleepTrees({t})
```
`mj_sleepTrees(tree[n])`: for k: `tree_asleep[tree[k]] = tree[(k+1) % n]` (cycle in
`map_itree2tree` order, which is ascending tree id within the island) — must currently be
`−1`, else error; then `qvel[tree dofs] = 0`, `qacc[tree dofs] = 0`.

Key properties:
* A tree must be under tolerance for **10 consecutive `mj_sleep` calls** (−11 → −1), and
  the whole island must reach −1 simultaneously.
* The islands used are those computed in the *current* step's `mj_island` (constraint
  graph incl. contacts, equality, limits, friction, plus metric couplings — see island chapter).
* When anything sleeps (`nslept>0`) inside `advanceStart`: **`mj_forwardSkip(m, d,
  mjSTAGE_POS, 0)`** is re-run (velocity + acceleration stages and sensors with the zeroed
  velocities, at the *old* qpos because `advanceStart` runs before positions are
  integrated), then `mj_updateSleep`. Note `advanceStart` is called before the qvel/qpos
  update in `mj_advance`, so the just-slept trees are excluded (via `dof_awake_ind`,
  `body_awake_ind`) from the integration that follows in the same step.

### 3.7 Order inside `mj_step` with sleep enabled

```
mj_forward:
  mj_fwdPosition:
    mj_kinematics1 (skip STATIC; detect sleeping-pose mismatch)
    mj_wake -> mj_updateSleep (if any)
    mj_kinematics2, mj_comPos, mj_camlight, mj_flex, mj_tendon
    mj_wakeTendon -> mj_updateSleep
    mj_makeM, mj_factorM
    mj_collision; mj_wakeCollision -> (mj_updateSleep; mj_collision again)
    mj_wakeEquality -> mj_updateSleep
    mj_makeConstraint; mj_island; …
  sensors, velocity, actuation, acceleration, constraint solve (awake-only)
integrator (Euler/implicit/discrete/RK4):
  mj_advance -> advanceStart (history, act) -> mj_sleep -> [mj_forwardSkip(POS); mj_updateSleep]
             -> qvel += h·qacc (awake dofs) ; integratePos (awake bodies) ; advanceFinish
```

### 3.8 How sleeping affects each pipeline stage (C)

Generic pattern: `sleep_filter = mjENABLED(SLEEP) && n?_awake < n?`; when false the full
loop runs (so with all trees awake the code path is identical to sleep-disabled, except
for the STATIC-body skips in kinematics which use only `mjENABLED(SLEEP)`).

| Stage / function | Effect |
|---|---|
| `mj_kinematics1` | STATIC bodies skipped (pose from reset); sleeping bodies recomputed and compared (exact) |
| `mj_kinematics2` | inertial frames & geoms: only `body_awake_ind` bodies (STATIC skipped); sites: only bodies with state AWAKE |
| `mj_comPos` | subtree_com over awake+static bodies; backward accumulation over `parent_awake_ind`; for a **sleeping child of an awake/static parent** the child's stored (normalised) `subtree_com` is re-multiplied by `body_subtreemass` before adding; cinert and cdof only for `body_awake_ind` |
| `mj_camlight` | camera skipped if its body and its target body (if any) are not AWAKE; lights analogous |
| `mj_tendon`, `mj_tendonArmature`, `mj_tendonBias`, tendon passive | tendons with `mj_tendonSleepState == ASLEEP` skipped (values frozen) |
| `mj_transmission`, actuation (`mj_fwdActuation` loops), actuator derivatives | actuators whose sleep state is ASLEEP skipped |
| `mj_crb` / `mj_factorM` | composite inertias over `parent_awake_ind`; M rows zeroed/filled and factorised only for `dof_awake_ind` (`mj_factorI` with index list) |
| collision broadphase | pair filtered if both bodies asleep, or one asleep and the other world-welded (`weldbody==0`; mocap has weldbody≠0 so mocap–sleeping pairs survive and trigger a wake); flex sides require awake dofs on at least one side (`hasAwakeDofs`) |
| `filterCollisionPair` for explicit `<pair>` | skipped if neither body AWAKE |
| constraints: equality, friction (DOF), limits (joint/tendon) | rows not instantiated for ASLEEP objects; contacts with sleeping geoms → `mjERROR` (cannot happen) |
| `mj_island` | only awake trees enter metric-coupling unions |
| passive (`mj_springdamper`, gravcomp, fluid, flex) | per-body loops over `body_awake_ind`, per-dof loops over `dof_awake_ind`, outputs zeroed only at awake indices (sleeping entries keep stale values but are never used) |
| `mj_comVel`, `mj_subtreeVel`, `mj_rne`, `mj_rnePostConstraint` | awake/static bodies only |
| sensors (`mj_sensorPos/Vel/Acc`) | sensor skipped if `mj_sensorSleepState == ASLEEP` → its `sensordata` retains the last value |
| `mj_fwdAcceleration` | `qfrc_smooth` and `qacc_smooth` only at awake dofs; effective-metric path solves with zeroed asleep entries |
| integrators | Euler damping/factor over awake dofs; `mj_advance`: `qvel += h·qacc` only at `dof_awake_ind`; `mj_integratePosInd` over `body_awake_ind` (static bodies have no joints) |
| plastic return mapping | joints/tendons/flexes not AWAKE skipped |
| `mj_checkVel/Acc` | only awake dofs checked |
| finite-difference derivatives (`mjd_transitionFD`, `mjd_inverseFD`) | `mjERROR("sleeping is not supported")` |
| IPC flag | incompatible: `mjERROR("flag ipc does not support flag sleep")` |
| discrete integrator | requires islands (`mjERROR` if `mjDSBL_ISLAND`) |

Sleep state of composite objects (`mj_sleepState`):
* body/xbody/joint/site/dof/geom/camera/light → `body_awake` of owning body;
* tendon: 0 trees → STATIC; 1 tree → that tree; 2 trees → AWAKE if either awake; >2 → AWAKE;
* actuator: by transmission (joint→joint; slidercrank/SO3-with-refsite: AWAKE if either site
  AWAKE else ASLEEP; tendon; site; body; undefined → AWAKE);
* equality: ASLEEP iff neither object is AWAKE (both ASLEEP/STATIC);
* sensor: USER, PLUGIN, RANGEFINDER, and CONTACT with site obj/ref → AWAKE; both UNKNOWN →
  AWAKE; one UNKNOWN → other's state (STATIC→AWAKE); otherwise AWAKE if either AWAKE else
  ASLEEP. (Hence a sensor between a sleeping body and the world is ASLEEP.)
* flex: state of the first vertex/node body with a tree, STATIC if none.

### 3.9 Initialisation / reset (`engine_io.c`)

* `mj_makeData`: `tree_asleep = kAwake`; `mj_updateSleepInit(flg_staticawake=1)` (so plugin
  init can run kinematics on everything).
* `mj_resetData`: `tree_asleep = kAwake`. If SLEEP enabled:
  * count trees with policy `INIT`; `mj_updateSleepInit(…, 1)`;
  * if none: partial position pass `mj_kinematics; mj_comPos; mj_camlight; mj_tendon` — this
    computes poses of STATIC bodies (which later steps skip);
  * else: full `mj_forward`; set INIT trees to `−1`, others `kAwake`; `mj_sleep`; if fewer
    slept than requested, wake leftovers (`kAwake`) and warn; zero `qacc_smooth`,
    `qfrc_smooth`; `mj_clearEfc`.
* Finally (always) `mj_updateSleep(m, d)` — with `flg_staticawake=0`, so world-welded bodies
  become STATIC.

Consequence: a model with SLEEP enabled must be reset with `mj_resetData` (or equivalent)
for static geoms/sites to have valid poses; editing `body_pos` of a static body at run time
has no effect while SLEEP is on.


---

## 4. Support functions (C, `engine_support.c`)

### 4.1 Name tables / sizes

`mjDISABLESTRING` (20): Constraint, Equality, Frictionloss, Limit, Contact, Spring, Damper,
Gravity, Clampctrl, Warmstart, Filterparent, Actuation, Refsafe, Sensor, Midphase,
Eulerdamp, AutoReset, NativeCCD, Island, MultiCCD.
`mjENABLESTRING` (8): Override, Energy, Fwdinv, InvDiscrete, Sleep, DiagExact, IPC, Speculative.
`mjTIMERSTRING`: step, forward, inverse, position, velocity, actuation, constraint, advance,
pos_kinematics, pos_inertia, pos_collision, pos_make, pos_project, col_broadphase, col_narrowphase.
`mjCONDATA_SIZE` = {FOUND 1, FORCE 3, TORQUE 3, DIST 1, POS 3, NORMAL 3, TANGENT 3, LINVEL 3, ANGVEL 3};
`mju_condataSize(spec)` = Σ sizes of set bits. `mjRAYDATA_SIZE` = {DIST 1, DIR 3, ORIGIN 3,
POINT 3, NORMAL 3, DEPTH 1}; `mju_raydataSize` likewise.

### 4.2 State get/set (`mjtState`)

Element sizes (`mj_stateElemSize`): TIME 1, QPOS nq, QVEL nv, ACT na, HISTORY nhistory,
WARMSTART nv, CTRL nu, QFRC_APPLIED nv, XFRC_APPLIED 6·nbody, EQ_ACTIVE neq, MOCAP_POS
3·nmocap, MOCAP_QUAT 4·nmocap, USERDATA nuserdata, PLUGIN npluginstate, PLASTIC nplastic.
Pointers: `time, qpos, qvel, act, history, qacc_warmstart, ctrl, qfrc_applied,
xfrc_applied, mocap_pos, mocap_quat, userdata, plugin_state, plastic`.

* `mj_stateSize(m, sig)`: error if `sig < 0` or `sig ≥ 2^15`; sum of element sizes for set bits.
* `mj_getState(m, d, state, sig)`: iterate bits **in increasing bit order** i = 0..14;
  concatenate. EQ_ACTIVE: `state[adr++] = (mjtNum) eq_active[j]` (mjtByte → 0.0/1.0).
* `mj_setState(m, d, state, sig)`: inverse; EQ_ACTIVE: `eq_active[j] = state[adr++]`
  (C conversion double→unsigned char: truncation; non-0/1 values are implementation-defined
  if out of range). After PLASTIC: if `d->flex_hessian_valid`, set
  `flex_hessian_valid[f]=0` for flexes with `flex_plasticadr[f] ≥ 0`.
* `mj_copyState(m, src, dst, sig)`: element-wise copy with the same plastic invalidation.
* `mj_extractState(m, src, srcsig, dst, dstsig)`: requires `dstsig ⊆ srcsig`; walks
  `srcsig` bits in order, copying the element when also in `dstsig`.
* `mj_setKeyframe(m, d, k)`: copies time, qpos, qvel, act, mocap pos/quat, ctrl, plastic into
  keyframe k (not history, not eq_active, not userdata).

### 4.3 Inertia helpers

* `mj_fullM(m, d, dst)`: `mju_sym2dense(dst, M, nv, M_rownnz, M_rowadr, M_colind)` (sparse
  lower-triangle CSR incl. diagonal → dense symmetric).
* `mj_mulM(m, d, res, vec)`: `mju_mulSymVecSparse`.
* `mj_mulM2(m, d, res, vec)`: `res = sqrt(D) · Lᵀ?` — precisely: `res_i = vec_i +
  Σ_{j in row i off-diag} qLD[adr+j]·vec[colind]` (skipped for simple DOFs), then
  `res_i *= sqrt(qLD[diag_i])`, where `qLD` stores the unit-lower L (off-diagonals) and D
  on the diagonal (`M = Lᵀ D L` convention of `mj_factorM`).
* `mj_addM(m, d, dst, rownnz, rowadr, colind)`: sparse → `mju_addToMatSparse`; dense
  (NULL index arrays) → `mju_addToSymSparse(dst, M, nv, …, flg_upper=0)`.

### 4.4 Perturbations

**`mj_applyFT(m, d, force, torque, point, body, qfrc_target)`** — accumulates
`qfrc_target += J_pᵀ force + J_rᵀ torque` (point Jacobian of `point` attached to `body`):
* error if body out of range;
* sparse (`mj_isSparse`: `opt.jacobian == SPARSE` or `AUTO && nv ≥ 60`): `NV = mj_bodyChain`,
  `mj_jacSparse(…, flg_skipcommon=0)`, `qforce = jacpᵀ force` (3×NV) then scattered with
  `+=` into `qfrc_target[chain[i]]`, then torque likewise (two separate accumulations);
* dense: `mj_jac`, `qforce = jacpᵀ force` (3×nv), `mju_addTo`, then torque.
`force`/`torque` may be NULL.

**`mj_xfrcAccumulate(m, d, qfrc)`**: quick return if bytes of `xfrc_applied[6..6nbody]` all
zero; else for `i = 1..nbody−1` with any nonzero (value compare `mju_isZero`) component:
`mj_applyFT(xfrc[6i..6i+2], xfrc[6i+3..], xipos[i], i, qfrc)`. Applied at the body **COM**.

### 4.5 `mj_geomDistance(m, d, g1, g2, distmax, fromto)`

1. `dist = distmax`; zero `fromto`.
2. Order so `type(g1) ≤ type(g2)` (`flip` if swapped); `func = mjCOLLISIONFUNC[t1][t2]`;
   none → return `distmax`.
3. If native CCD enabled (`!mjDSBL_NATIVECCD`) and `func ∈ {mjc_Convex, mjc_BoxBox}`:
   `mj_geomDistanceCCD` with the **original (unflipped)** g1,g2: config
   `max_iterations = opt.ccd_iterations`, `tolerance = opt.ccd_tolerance`,
   `npolygonmax = nmeshdegmax = 0`, `max_contacts = 1`, `dist_cutoff = distmax`;
   `mjc_initCCDObj(obj, g, margin=0)`; `dist = mjc_ccd(...)`; witness points `x1, x2` copied
   to fromto if `status.nx > 0`; return `min(dist, distmax)` (mjc_ccd returns DBL_MAX beyond cutoff).
4. Else call `func(m, d, con, g1, g2, margin=distmax)`, take the smallest `con[i].dist <
   dist` (strict); if found and `fromto`: `sign = flip ? −1 : 1`;
   `from = pos − 0.5·sign·dist·normal`, `to = pos + 0.5·sign·dist·normal`.
   Returns `distmax` if no contact closer than distmax.
Used by `mjSENS_GEOMDIST/NORMAL/FROMTO` sensors (sensor chapter).

### 4.6 `mj_insideSite(m, d, s, point)`
Mesh sites: local point `p = R_siteᵀ(point − site_xpos)`, then canonical mesh frame
`p' = R(mesh_quat)ᵀ(p − mesh_pos)`; AABB check against `site_size`; then all convex hull
polygon planes: `n·p' − n·v0 > 0` → outside (`mesh_polynormal`, first polygon vertex).
Returns 0 if no mesh/polygons. Other types → `mju_insidePrimitive` (§2.10).

### 4.7 Integration helpers

**`mj_differentiatePos(m, qvel, dt, qpos1, qpos2)`** per joint:
* FREE: `qvel[0..2] = (qpos2 − qpos1)/dt`; then rotation as BALL on the next 4/3 entries.
* BALL: `qvel = mju_subQuat(qpos2, qpos1) / dt` where `subQuat(qa, qb)`: `qdif = conj(qb)·qa`
  (`mji_negQuat` = conjugate), then `mju_quat2Vel(qdif, 1)`:
  `axis = qdif[1..3]`, `s = normalize(axis)` (norm<mjMINVAL → (1,0,0)),
  `speed = 2·atan2(s, qdif[0])`, if `speed > π` then `speed −= 2π`; `res = axis·speed`.
  Then scaled by `1/dt` (multiply by reciprocal: `mju_scl3(…, 1/dt)`).
* HINGE/SLIDE: `(qpos2 − qpos1)/dt`.
No quaternion normalisation of inputs.

**`mj_integratePosInd(m, qpos, qvel, dt, index, nbody)`** — loops `b = 1..nbody−1`,
`k = index ? index[b] : b` (so with `body_awake_ind`, entry 0 — the world — is skipped),
for each joint of body k:
* FREE: `qpos[0..2] += dt·qvel[0..2]`; then rotation.
* BALL/FREE-rot: `mju_quatIntegrate(quat, vel, dt)`:
  `tmp = vel; angle = dt·normalize(tmp)` (zero vel → axis (1,0,0), angle 0);
  `qrot = axisAngle2Quat(tmp, angle)` (angle exactly 0 → identity);
  `normalize4(quat)` **before** multiplying; `quat = quat ⊗ qrot` (body-frame angular
  velocity, right multiplication). **No normalisation after** the multiply.
* HINGE/SLIDE: `qpos += dt·qvel`.
`mj_integratePos(m, qpos, qvel, dt) = mj_integratePosInd(…, NULL, nbody)`.

**`mj_normalizeQuat(m, qpos)`**: `mju_normalize4` of ball quats and free quats (offset 3).
`mju_normalize4`: if norm < mjMINVAL → (1,0,0,0); else divide.

### 4.8 Actuator helpers

* `mj_actuatorDisabled(m, i)`: group outside [0,30] → 0; else bit `group` of
  `opt.disableactuator`.
* **`mj_nextActivation(m, d, a, act_adr, act_dot)`** (called per activation slot from
  `advanceStart`, with act_dot forced to 0 for disabled actuators):
  * FILTEREXACT: `tau = max(mjMINVAL, dynprm[0])`; `act += act_dot·tau·(1 − exp(−h/tau))`.
  * DCMOTOR: slots from `mj_dcmotorSlots(dynprm, gainprm)` assigned in order slew
    (`dynprm[7]>0`), integral (`gainprm[5]>0`), temperature (`dynprm[2]>0`), bristle
    (`dynprm[5]>0`), current (`dynprm[0]>0`); `offset = act_adr − actuator_actadr`:
    * current: `te = max(mjMINVAL, dynprm[0])`, exact filter as above;
    * bristle (LuGre): `F_C, F_S, v_S = biasprm[3..5]`, `σ0 = dynprm[5]`,
      `v = actuator_velocity[actuator_outadr[a]]`, `g = F_C + (F_S−F_C)·exp(−(v/max(mjMINVAL,v_S))²)`,
      `α = −σ0|v|/max(mjMINVAL,g)`, `e = exp(α h)`, `I = |α|>mjMINVAL ? (e−1)/α : h`,
      `act = e·act + I·v`;
    * integral: Euler then clip to `±dynprm[8]` if `dynprm[8] > 0`;
    * temperature, slew: Euler.
  * otherwise Euler `act += act_dot·h`.
  * Clamp to `actuator_actrange` if `actlimited` and dyntype ≠ DCMOTOR.

### 4.9 Mass, version, history, camera

* `mj_getTotalmass(m) = Σ_{i≥1} body_mass[i]` (sequential sum).
* `mj_setTotalmass(m, M)`: `scale = max(mjMINVAL, M / max(mjMINVAL, total))`; scales
  `body_mass` and `body_inertia` of bodies ≥1 (user must call `mj_set0`).
* `mju_camIntrinsics` — see §2.10.
* `mj_readCtrl(m, d, id, time, result, interp)`: if `actuator_history[2id] == 0` return
  `ctrl + ctrladr`; `interp < 0` → model's `actuator_history[2id+1]`; read
  `history + actuator_historyadr[id]` at `time − actuator_delay[id]` via `mju_historyRead`.
* `mj_readSensor` analogous with `sensor_history`, `sensor_delay`, `sensor_dim`, falling back
  to `sensordata + sensor_adr`.
* `mj_initCtrlHistory / mj_initSensorHistory(…, times, values[, phase])`: error if no
  buffer; `times == NULL` → reuse current buffer times; `mju_historyInit(buf, n, dim,
  times, values, user)` with `user = buf[0]` preserved for ctrl, `= phase` for sensor.

History buffer layout (`mju_history*`, `engine_util_misc.c`): `[user, cursor, times[n],
values[n·dim]]`; `cursor` = physical index of the newest sample; logical index `l` ↔
physical `(cursor+1+l) % n`. Init requires strictly increasing times (`Δ < mjMINVAL` → error),
sets `cursor = n−1`. `historyFindIndex`: binary search for logical `i` with
`times[i−1] < t ≤ times[i]` (0 if `t ≤ oldest`, n if `t > newest`). Insert: exact match
(`|t−times| < mjMINVAL`) → overwrite slot; `i==0` → replace oldest; `i==n` → advance cursor,
overwrite the oldest slot; else shift logical `[1, i−1]` left and write at `i−1`.
Read: `t ≤ oldest+mjMINVAL` → oldest; `t ≥ newest−mjMINVAL` → newest; exact match → slot;
`interp 0` → lower sample (ZOH); `1` → linear `lo + α(hi−lo)`, `α = (t−t_lo)/(t_hi−t_lo)`;
`2` → cubic Hermite with Catmull-Rom slopes `m_lo = (v_hi − v_{lo−1})/(t_hi − t_{lo−1})`
(0 if `i ≤ 1`), `m_hi = (v_{hi+1} − v_lo)/(t_{hi+1} − t_lo)` (0 if `i ≥ n−1`),
`res = h00 v_lo + h10 dt m_lo + h01 v_hi + h11 dt m_hi` with standard `h00 = 2α³−3α²+1`,
`h10 = α³−2α²+α`, `h01 = −2α³+3α²`, `h11 = α³−α²`. Read returns a pointer into the buffer
for non-interpolated cases (result untouched) and NULL when it wrote `res`.

---

## 5. Related core utilities (C, `engine_core_util.c`, `engine_core_constraint.c`)

**`mj_contactForce(m, d, id, result[6])`**: zero result; if `0 ≤ id < ncon` and
`contact[id].efc_address ≥ 0`: pyramidal cone → `mju_decodePyramid(result,
efc_force+adr, friction, dim)`; elliptic → copy `dim` values. Then **`result[0] −=
con->adhesion`** (net interface force). Contact frame (normal, tangents); torques are
torsional/rolling.
`mju_decodePyramid`: `dim==1` → `f0 = p0`; else `f0 = Σ_{i<2(dim−1)} p_i` (sequential),
`f_{i+1} = (p_{2i} − p_{2i+1})·μ_i`.

**`mj_objectVelocity(m, d, objtype, id, res[6], flg_local)`**: objtype ∈ {BODY (xipos/ximat),
XBODY (xpos/xmat), GEOM, SITE, CAMERA}; if `body_dofnum[body_weldid[body]] == 0` → zeros.
Else `mju_transformSpatial(res, cvel[body], flg_force=0, newpos=pos,
oldpos=subtree_com[rootid], rot = flg_local ? obj xmat : NULL)`:
`tran = vec; tran[3:6] = vec[3:6] − (newpos − oldpos) × vec[0:3]`; with rot:
`res = (Rᵀ tran[0:3], Rᵀ tran[3:6])`. Output layout `[ω; v]`.

**`mj_objectAcceleration`**: same transform applied to `cacc` and to `cvel`; then
`res[3:6] += vel[0:3] × vel[3:6]` (Coriolis correction; classical linear acceleration of
the point). Requires `cacc` (from `mj_rnePostConstraint`).

**`mj_geomSurfaceVelocity`**: `lin = R_g sv[0:3] + (R_g sv[3:6]) × (point − geom_xpos)`, `ang = R_g sv[3:6]`.

**`mj_local2Global`**: by `mjtSameFrame`: NONE/BODYROT/INERTIAROT → `xpos = xmat_b·pos +
xpos_b`; BODY → `xpos_b`; INERTIA → `xipos_b`. Orientation NONE → `quat2Mat(xquat_b ⊗ quat)`;
BODY/BODYROT → `xmat_b`; INERTIA/INERTIAROT → `ximat_b`.

**`mj_bodyChain(m, body, chain)`**: simple body (`body_simple`) → its own DOFs; else from
`weldid`'s last DOF walk `dof_parentid` to −1, then reverse to ascending order.

**`mj_jac(m, d, jacp, jacr, point, body)`**: zero (3×nv, row-major); `offset = point −
subtree_com[rootid(body)]` (uses the *original* body's root); `body = weldid`; for DOFs from
last DOF of the weld body following `dof_parentid`: `jacr[:,i] = cdof[0:3]`,
`jacp[:,i] = cdof[3:6] + cdof[0:3] × offset`. `mj_jacSparse` does the same into the
compressed chain columns (`ci` decreasing search), erroring if a DOF is missing from the
chain unless `flg_skipcommon`.

**`mj_mulJacVec / mj_mulJacTVec`**: no-op if `nefc == 0`; sparse uses
`mju_mulMatVecSparse(efc_J, …, efc_J_rowsuper)` / `mju_mulMatTVecSparse`; dense
`mju_mulMatVec / mju_mulMatTVec` (nefc × nv).

---

## 6. Visualization (C, brief — not physics)

* `engine_vis_init.c`: defaults (`mjv_defaultOption`: groups 0–2 visible, flags from
  `mjVISSTRING` default column, `bvh_depth=1`, `flex_layer=0`; `mjv_defaultCamera`:
  free, distance 2, azimuth 90, elevation −45; `mjv_defaultFreeCamera` from `m->stat` and
  `m->vis.global`), `mjv_makeScene/freeScene`, `mjv_rbound` (transparent sorting).
* `engine_vis_visualize.c`: `mjv_updateScene` = plugin `visualize` hooks + `mjv_addGeoms`
  (abstract geoms per category) + `mjv_makeLights` + `mjv_updateCamera` + skins/flex
  (`mjv_updateActiveFlex/Skin`). Includes tendon catenary/geodesic drawing
  (`mjv_isCatenary`, `mjv_catenary`, `mjv_geodesic`) and depth-image visualisation using
  `mju_camPixelRay`. Reads `d->bvh_active` (set by `mju_rayTree`/collision midphase when
  `vis.global.bvactive`). Output only.
* `engine_vis_interact.c`: camera math (`mjv_room2model`, `mjv_cameraInModel`,
  `mjv_moveCamera`, …), `mjv_select` (casts one ray via `mj_ray` with the viewer's
  geomgroup and `mjVIS_STATIC`, plus `mj_rayFlex` per flex and `mju_raySkin` per skin;
  closest wins), and the only functions that write physics inputs:
  * `mjv_applyPerturbPose`: mocap body → `mocap_pos/quat = ref pose` (converted from
    inertial frame `refpos/refquat · inv(body_ipos, body_iquat)`); paused + free root →
    writes `qpos` of the free joint (directly or by `ref·inv(child)·root`).
  * `mjv_applyPerturbForce`: writes `xfrc_applied[sel]`: translation:
    `F = −k·m_loc·(selpos − refselpos) − sqrt(k)·m_loc·v_sel` (`k = vis.map.stiffness`,
    `m_loc = pert->localmass`), `τ = (selpos − xipos) × F − sqrt(k_rot)·I·(d̂·ω) d̂`;
    rotation: `τ = quat2Vel(refquat·conj(xquat·iquat), 1/(k_rot·I)) − sqrt(k_rot)·I·ω`, with
    `I = 1/max(mjMINVAL, body_invweight0[2sel+1])` (or 1 if invweight is 0). Since
    `xfrc_applied ≠ 0` prevents sleep, perturbing wakes trees.

---

## 7. Rendering, experimental, batch, bindings (brief)

* **`src/render/classic`** (OpenGL 2/3 fixed-function + shaders: `render_context.c`
  uploads meshes/hfields/textures — `mjr_uploadMesh` computes smooth vertex normals, a
  vertex normal deviating more than ~37° (dot < 0.8) from its face normal is replaced by
  the face normal; `render_gl3.c` draws `mjvScene`, shadows, reflections, skybox, segmentation).
  **`src/render/filament`** (`mjrf_*` API, PBR via Google Filament) and **`src/render/noop`**
  (stubs). None influence physics.
* **`src/experimental/filament/mjr_compat.cc`**: implements the classic `mjr_*` API on top of
  the Filament renderer (UI text functions error). **`src/experimental/usd`**: OpenUSD
  writer/layer sink and the `mjcPhysics` USD schema (codegen'd APIs for joints,
  actuators, equality, collision…) for importing/exporting MuJoCo models to USD.
* **`src/batch/batch.cc` + `include/mujoco/mjbatch.h`** (new C batch API): `mjBatch` = nsim
  simulations of one model on a thread pool (`mjb_makeBatch(m, nsim, config)`; config
  `nthread` (≤0 = one per logical CPU), `persistent` (one `mjData` per sim vs per thread)).
  State rows are `mj_stateSize(m, mjSTATE_INTEGRATION)` long in `mj_getState` order.
  `mjb_step(ids, nstep, stage)` runs `nstep × mj_step` then computes stages up to `stage`;
  `mjb_forward`, `mjb_reset(key)`, `mjb_rollout(nstep, control, control_spec, records)`
  (per substep `mj_setState(control, control_spec)` + `mj_step`, recording mjData fields or
  states), `mjb_apply(func)`, per-simulation model field expansion (`mjb_expand`),
  `mjb_setConst`, `mjb_setModels`. Non-persistent batches reload the state row with
  `mj_setState(…, mjSTATE_INTEGRATION)` before each call and save it after; a persistent
  batch whose mjData still matches its row may continue with `mj_step2` (not for RK4 or when
  callbacks are set). **Sleep and IPC require `persistent`** (their bookkeeping —
  `tree_asleep` etc. — is not part of the state vector). Errors are trapped per simulation
  with setjmp/longjmp (`mjb_status`, `mjb_error`).
* **Python** (`python/mujoco`): pybind11 bindings of the full C API (`functions.cc`,
  `structs.cc`, …); `rollout.py`/`rollout.cc` — `mujoco.rollout.rollout(model(s), data(s),
  initial_state, control, control_spec=mjSTATE_CTRL, nstep, initial_warmstart, …)` runs
  open-loop rollouts on a thread pool: for each rollout it zeroes ctrl/qfrc_applied/xfrc_applied
  not in `control_spec`, resets mocap to `body_pos/quat` and `eq_active` to `eq_active0` when not
  in the spec, `mj_setState(state0, mjSTATE_FULLPHYSICS)`, sets `qacc_warmstart` (given or 0),
  clears warning counters; per step: if any warning occurred, fill remaining outputs with the
  current state and stop; else `mj_setState(control_t, control_spec)`, `mj_step`, record
  `mj_getState(FULLPHYSICS)` and `sensordata`. `batch.py` wraps `mjBatch`. Also viewer,
  renderer (EGL/OSMesa/GLFW), `minimize.py` (least squares), `sysid`, `usd` exporters.
* **WASM** (`wasm/`): Emscripten-compiled engine with auto-generated JS/TS bindings
  (`@mujoco/mujoco` npm). **Unity** (`unity/`): C# plug-in importing MJCF and stepping via
  the C API. Both are thin interfaces; physics identical to the C library.


---

## 8. GPU (MuJoCo Warp) implementation

General: all arrays are `[nworld, …]` (Model arrays may be batched with `worldid %
shape[0]`), precision **float32** throughout (C is float64). Kernels are launched with
`dim=(nworld, n…)`; one thread per (world, item) unless noted.

### 8.1 Ray casting (`ray.py`)

**Primitive functions** (`@wp.func`, return `(dist, normal)`; boxes also `all: vec6`):
`ray_plane, ray_sphere(pos, r², pnt, vec), ray_capsule, ray_ellipsoid, ray_cylinder,
ray_box, ray_hfield, ray_mesh, ray_geom(…, geomtype)` mirror the C code line by line
(same bounding-sphere prefilters, same side/face loop order, same normals) with these
differences:

| Item | C | Warp |
|---|---|---|
| `_ray_quad` reject | `det < 0 || a < mjMINVAL` | `det < MJ_MINVAL` (tangent/near-tangent rays rejected); division via `safe_div(1, a)` (a==0 → divide by 1e-15) and multiplication by the reciprocal `(−b∓√det)·(1/a)` |
| Ellipsoid `1/size²` | plain division | `safe_div` |
| Triangle basis | ad-hoc `b0=(1,1,1)` Gram-Schmidt (§2.2) | **Duff et al. 2017 branchless ONB**: `s = sign(v_z)` (≥0 → +1), `a = −1/(s+v_z)`, `b = v_x v_y a`, `b0 = (1 + s v_x² a, s b, −s v_x)`, `b1 = (b, s + v_y² a, −v_y)` — assumes **unit** `vec` (not normalised by the caller!) |
| `_ray_triangle` | returns possibly negative x | returns −1 when `dist < 0` |
| Mesh (`ray_mesh`, non-BVH path) | BVH tree traversal | brute-force over all faces `mesh_faceadr[id] .. mesh_faceadr[id+1]` (or `nmeshface`); bounding-box prefilter with `geom_size` |
| Hfield (`ray_hfield`) | as §2.4 | same algorithm (base box, top box, triangles, side walls); dx/dy and divisions via `safe_div` |
| SDF, flex, skin | supported | `ray_geom` returns −1 for SDF; no flex/skin ray in `rays()` |
| Normal on miss for mesh | zero | `mat @ 0 = 0` |

**`_ray_eliminate`**: same five tests; "no geomgroup" is signalled by all six entries of
`geomgroup: vec6` equal to −1; `groupid = clamp(geom_group, 0, 5)`.

**Kernel `_ray`** (non-BVH): `wp.launch_tiled(dim=(nworld, nray), block_dim=m.block_dim.ray=64)`;
each block handles one (world, ray); lanes stride over geoms
(`geomid = tid, tid+64, …`); per chunk a tile of 64 distances (miss → `MJ_MAXVAL=1e10`),
`wp.tile_argmin` (first minimal lane) and a strict `<` running minimum across chunks ⇒
ties resolve to the lowest geom id, like C. Output `dist = −1` if the min is `≥ MJ_MAXVAL`.
Origins/directions: `pnt, vec: [1 or nworld, nray]`; `bodyexclude: [nray]` per ray.

**Kernel `_ray_bvh`** (when a `RenderContext` is passed): one thread per (world, ray);
traverses the scene BVH (`wp.bvh_query_ray` on the world's group root,
`wp.bvh_query_next(query, idx, min_dist)` which prunes by current best); geom id from
`enabled_geom_ids` (only geoms in the render context's enabled groups!). Mesh and hfield
use `ray_mesh_with_bvh` (Warp `wp.Mesh` BVH, `wp.mesh_query_ray`) with
**back-face culling on** (hits with `dot(lvec, n) ≥ 0` skipped, marching up to 8 times
past culled hits with offset `t(1+1e-6)+1e-9`) — differs from C, which reports exit hits.
Hfield BVH meshes are built from a greedily merged triangulation (§8.5), without the base box
and side walls. Strict `<` ⇒ ties go to the first geom visited in BVH order (not lowest id).

**Public API**: `ray(m, d, pnt, vec, geomgroup=None, flg_static=True, bodyexclude=-1, rc=None)`
(single ray per world) and `rays(m, d, pnt, vec, geomgroup, flg_static, bodyexclude[nray],
dist, geomid, normal, rc=None)`.

**Rangefinder** (`sensor.py`): `_sensor_rangefinder_init` (one thread per (world,
rangefinder ray); camera rangefinders expand to `width×height` rays, row-major) computes
origin/direction exactly like `mju_camIntrinsics` + `mju_camPixelRay` (fovy path:
`fx = (0.5H)/tan(fovy·π/360)`), then calls `rays(…, geomgroup=vec6(MJ_MAXVAL…) [all groups
enabled], flg_static=True, bodyexclude=m.sensor_rangefinder_bodyid)` — the **brute-force**
path for every pixel (no `mj_multiRay` angular culling; see pitfalls), then
`_sensor_rangefinder` writes the dataspec fields with the same layout as `fill_raydata`.

### 8.2 Sleeping (`sleep.py`, hooks in `forward.py`, `collision_driver.py`, `solver.py`, `island.py`, `io.py`)

Enabled only if `SLEEP` and not `DSBL_ISLAND` (`sleep_enabled` in `forward.py`).
`put_model` restrictions: sleep requires **solver = Newton**; tree policies must be
AUTO/AUTO_NEVER/AUTO_ALLOWED (user `NEVER`, `ALLOWED`, **`INIT` unsupported**); flex
equality + sleep rejected; discrete integrator + sleep rejects flex metric features.

Data fields: `tree_asleep[nworld,ntree]`, `tree_awake[nworld,ntree]`,
`body_awake[nworld,nbody]`, `body_awake_ind`, `dof_awake_ind`, counters `ntree_awake,
nbody_awake, nv_awake [nworld]` (no `parent_awake_ind`). `K_AWAKE_VAL = −(1+MJ_MINAWAKE)`.

**`update_sleep(m, d, flg_staticawake=0)`**: `_zero_sleep_counters`;
`_update_sleep_trees` (per (world,tree): `tree_awake = tree_asleep < 0`, atomic count);
`_update_sleep_bodies` (per body, same classification as C, index lists appended with
`wp.atomic_add` → **list order is non-deterministic** unless `opt.deterministic & ATOMICS`
selects `wp.DeterministicMode.RUN_TO_RUN`); `_update_sleep_dofs` likewise.
`update_sleep_trees` only refreshes `tree_awake` (used after the tendon wake).

**Wake kernels** (all write `tree_asleep` in place; per-world races are possible when two
threads wake overlapping cycles — writes store identical values in practice):
* `_wake_kernel` (per (world,tree)) = `mj_wake` without the flex rule and **without
  pose-mismatch detection** (Warp kinematics never compares/skips sleeping bodies, so a
  user edit of a sleeping tree's `qpos` does not wake it; only nonzero `qvel`,
  `qfrc_applied`, `xfrc_applied` do). `_tree_can_sleep` compares by value (`!= 0.0`), so
  `−0.0` counts as zero (C compares bytes). Policy check only for `AUTO_NEVER`.
* `_wake_collision_kernel` (per contact, `dim=naconmax`, guard `conid < nacon`): only
  geom–geom contacts (flex contacts skipped); returns if either tree < 0 (**no mocap-contact
  wake**) or both asleep; else wakes the sleeping tree with the awake tree's `tree_asleep`.
* `_wake_tendon_kernel` (per (world,tendon)): trees from wrap objects
  (JOINT → `jnt_bodyid`, SITE → `site_bodyid`, SPHERE/CYLINDER → `geom_bodyid`; pulleys
  ignored); if any is awake and the tendon limit is active (`length−r0 < margin` or
  `r1−length < margin`, `tendon_limited` required), wake all sleeping trees with
  `wakeval = min(K_AWAKE_VAL, awake tree_asleep values)` — always ≤ −11 (C passes the awake
  tree's countdown). No `treenum ≥ 2` gate and **no metric-tendon rule**.
* `_wake_equality_kernel` (per (world,eq)): CONNECT/WELD/JOINT as C but static sides are
  always STATIC (mocap not treated as awake); TENDON equalities **are** handled (wake all
  sleeping trees of both tendons if either has an awake tree, `wakeval = min(K_AWAKE, w1,
  w2)`) — C errors on tendon equality with sleep; FLEX: not implemented.

**Sleep** (`sleep(m, d)`): three launches:
1. `_sweep_awake_trees` (per (world,tree)): as C's first loop (increment toward −1 or reset
   to −11); tolerance `opt.sleep_tolerance[worldid % …]` (batchable).
2. `_check_island_can_sleep` (per (world,tree)): `island_can_sleep[w, island] = 1`
   initially (`wp.ones((nworld, ntree))`), `atomic_min` to 0 if any tree of the island has
   `tree_asleep < −1`.
3. `_build_cycles` (one thread per world, serial): for each island that can sleep, link its
   trees in ascending tree id into a cycle (same as C's `map_itree2tree` order) and zero
   their `qvel`, `qacc`; then for trees with no island (`tree_island < 0`), `−1 → self-cycle`;
   also **re-zeroes qvel/qacc of every sleeping unconstrained tree** each call.
No `wakeNonMetricFlex` equivalent (flexes cannot sleep under the restrictions anyway).

**Pipeline placement (differs from C):**
```
forward():  wake; update_sleep; fwd_position; sensors…; fwd_velocity; …; fwd_acceleration; solve
fwd_kinematics(): kinematics, com_pos, camlight, flex, tendon; [wake_tendon; update_sleep_trees]
fwd_position(): fwd_kinematics; crb; tendon_armature; factor_m;
                collision (pass 1); wake_collision; awake_prev = clone(body_awake); update_sleep;
                collision(awake_prev) (pass 2, incremental: only pairs pass 1 skipped
                  because of sleep and that involve a newly-awakened body, appended to the
                  pass-1 contact buffer; broadphase skipped if no body changed state; flex
                  collision not re-run since it is not sleep-aware);
                make_constraint; [wake_equality]; update_sleep; island(); transmission; …
_advance(): next_activation; qvel += h·qacc (ALL dofs); qpos integrate (ALL joints);
            plastic (awake only); history insert; time += h; qacc_warmstart = qacc;
            sleep(); fwd_velocity(); update_sleep()   # unconditionally when sleep enabled
```
Thus in Warp the sleep test uses the **post-integration** `qvel`, happens **after** the
position update, and only velocity-stage quantities are refreshed afterwards (C: test on
pre-integration qvel inside `advanceStart`, sleeping trees excluded from the same step's
integration, full `mj_forwardSkip(POS)` incl. acceleration and sensors). Expect
trajectories to differ by one step around sleep events.

**Effect on the rest of the Warp pipeline** (no per-stage `body_awake_ind` loops):
* kinematics, com_pos, crb, factor_m, passive, sensors, constraint instantiation run for all
  bodies/DOFs; sleeping trees produce constraint rows (limits, equality, friction) — they
  are neutralised in the solve by DOF compaction.
* Collision broadphase (`nxn`/`sap` pair kernels): skip pairs with both bodies ASLEEP or
  one ASLEEP and the other STATIC (world-welded, non-mocap) — equals C's filter except C
  uses `weldbody == 0` (same set) — mocap bodies are AWAKE.
* `fwd_acceleration`: `_qfrc_smooth` sets `qfrc_smooth = 0` for DOFs of sleeping trees;
  then (non-discrete) `island.update_active_dofs` + `solver.smooth_solve_compact`.
* **DOF compaction** (`island.update_active_dofs`): per world (serial over trees), awake
  trees' DOFs are packed into `[0, ncdof)` (`dof_cdof[nv]` → compact index or −1,
  `cdof_dof[nvmax_pad]` → DOF or −1); overflow if active DOFs > `nvmax`
  (`overflow |= NVMAX`, ncdof clamped; behaviour undefined). `nvmax` (make_data argument,
  default nv) and `nvmax_pad = ((max(nvmax,1)+T)//T)·T` with `T = TILE_SIZE_JTDAJ_DENSE`
  (reserves one augmented column). `smooth_solve_compact`: gather sparse M into dense
  `cM [nworld, nvmax_pad, nvmax_pad]` (padding diagonal = 1 for `i ≥ ncdof`), gather RHS,
  blocked tiled Cholesky factor+solve, scatter (`qacc_smooth = 0` for inactive DOFs).
  The Newton solver then runs in compact space (`dof_cdof` remaps `efc_J` column indices;
  columns with −1 are dropped); `_scatter_dof_vecs` writes `qacc = 0` on inactive DOFs.
* Discrete integrator + sleep: `_zero_sleeping_dofs` applied to `qfrc_constraint`,
  preconditioned gradient and search direction; `_add_qfrc_smooth_eff` zeroes sleeping DOFs.
* `implicit_free_body` (free-body implicit damping) skips sleeping trees.
* Plastic updates skip joints/tendons/flexes whose bodies are not AWAKE.
* Since `_next_position` integrates every joint, sleeping free/ball quaternions are
  renormalised every step (`quat_integrate(q, 0, h) = normalize(normalize(q)⊗(1,0,0,0))`),
  whereas C leaves them untouched — tiny drift in `qpos` only if the stored quaternion was
  not exactly unit.

**Island relation**: `island.island()` (EFC-parallel DSU with `atomic_min` root hooks:
`_reset_dsu`, `_island_dsu` (tiled over (world, efc chunk)), `_compress_roots`,
`_label_roots` (serial per world, roots numbered in ascending tree order = MuJoCo island
order), `_propagate_labels`) produces `tree_island[nworld, ntree]` (−1 if unconstrained)
used by the sleep kernels. It is only called when sleep is enabled (or by island-based
solvers). Contacts with flex sides union the trees of all weighted bodies.

**Reset/init** (`io.py`): `make_data`: `tree_asleep = −11`, `tree_awake = 1`, `body_awake`
from `_initial_body_awake` (mocap test uses `body_mocapid[body_rootid[b]]`); `put_data`
copies `mjd.tree_asleep`/`body_awake` captured **before** running `mj_kinematics`;
`reset_data` kernel `reset_sleep` sets `tree_asleep = −11`, `tree_awake = 1`,
`body_awake` (mocap test with `body_mocapid[b]` — not the root's, a subtle difference for
children of mocap bodies, corrected by the subsequent `update_sleep`), identity index lists,
then `update_sleep` if sleep is enabled. Static geom poses are computed once in
`make_data/put_data` (smooth.py never updates world-welded, non-mocap geoms), independent of
sleep — unlike C, where this skip happens only under sleep.

### 8.3 Support (`support.py`)

* `mul_m(m, d, res, vec, skip=None, M=None)`: sparse gather kernel, one thread per
  (world, dof) over a precomputed row list (`M_mulm_rowadr/col/madr`, both triangles), no
  atomics; dense variant for compact `[nworld, nv, nv]` blocks.
* `apply_ft` / `xfrc_accumulate`: kernel per (world, dof): for every body `b` in the
  subtree of `dof_bodyid` (contiguous range `[b0, b0+body_subtreenum)`) with nonzero `ft`:
  `acc += cdof_lin·f + cdof_ang·τ + (cdof_ang × (xipos_b − subtree_com[root]))·f`.
  Mathematically equal to C's `Σ_b J_bᵀ ft_b`, but summation order differs (per-dof over
  bodies vs per body over dofs), float32.
* `contact_force(m, d, contact_ids, to_world_frame, force)` /
  `contact_force_fn`: pyramidal decode (guards `adr < njmax`) or elliptic copy via
  `contact.efc_address[c, i]` per component; `force[0] −= contact.adhesion`; optional world
  frame: `f_world = f_contactᵀ · frame` for both 3-vectors.
* `jac(m, d, jacp, jacr, point[nworld], body[nworld])`: per (world, dof); DOF participates
  iff `body_isdofancestor[body, dof]` (precomputed table); same formula as C. `jac_dot_dof`
  for `mj_jacDot` (ball / free-rotational DOFs recompute `cdof_dot = cvel × cdof`).
* `get_state / set_state(m, d, state[nworld, nstate], sig, active=None)`: one thread per
  world, identical layout to C (bit order); `PLUGIN` unsupported; `eq_active` as
  `float(bool)` / `bool(x)`; PLASTIC invalidates `flex_hessian_valid`.
* `next_act`: FILTEREXACT exact; **`DynType.USER` returns `act` unchanged** (C integrates
  Euler); else Euler; optional clamp. DC motor slots handled in `forward._next_activation`
  with the same formulas as §4.8.
* `quat_integrate(q, v, h)` (math.py): `angle = h|v|`, `axis = normalize(v)` (Warp normalize
  of zero vector → zero, then `axis_angle_to_quat(0, 0) = (1,0,0,0)`),
  `q = normalize(normalize(q) ⊗ qrot)` — **renormalises after** the product (C does not).
* No `mj_geomDistance` API function (the distance sensors are implemented in
  `collision_driver`/`sensor` via the collision kernels).

### 8.4 `io.py` support functions (summary)

`put_model(mjm, batch_sizes)`: converts MjModel → device Model, rejecting unsupported
features (plugins, PGS/noslip combos, unsupported enum values, sleep w/o Newton, flex eq
with sleep, quadratic flex interpolation, dense Jacobian with nv>60, …); precomputes
tables (`body_isdofancestor`, `M_mulm_*`, block layouts, rangefinder ray maps, etc.).
`make_data(mjm, nworld, nconmax, nccdmax, njmax, njmax_nnz, naconmax, naccdmax, nvmax)`:
fixed-size buffers; defaults `_default_nconmax = round-up-to {16,24,32,48,…,8192}` of
`max(nv·0.35·(nhfield>0)·10 + 45, 256·has_flex, 64·has_sdf)`; `_default_njmax` likewise from
`max(nv·2.26·(nhfield>0)·18 + 53, 512·has_flex, 256·has_sdf)`; `naconmax = nworld·nconmax`
unless given; padded sizes (`njmax` rounded to tile size, `nv` to tile or multiple of 4).
`put_data(mjm, mjd, …)` copies a host MjData (running `mj_kinematics`/`mj_comPos` to
populate static poses); `get_data_into(result, mjm, d, world_id)` copies back;
`reset_data(m, d, reset_mask)` / `reset_data_keyframe(m, d, key)`; `override_model`
(string overrides such as `opt.iterations=1`). Overflows of `naconmax`, `njmax`, `nvmax`
are flagged in `d.overflow` (bitmask `OverflowType`) and optionally printed
(`opt.warn_overflow`), never raised.

### 8.5 GPU batch renderer (`render.py`, `render_util.py`, `bvh.py`)

A ray-traced replacement for OpenGL, rendering RGB (packed uint32), planar depth and
segmentation for many cameras × worlds. Not physics; parity target is *visual similarity*
with MuJoCo's OpenGL renderer, not bitwise identity.

**Render context** (`create_render_context(mjm, nworld, cam_res, render_rgb/depth/seg,
use_textures=True, use_fast_math=True, use_shadows=False, use_ambient_lighting=True,
enabled_geom_groups=[0,1,2], cam_active, background_color, flex_render_smooth,
use_precomputed_rays=True, render_skybox=False, enable_backface_culling=True,
shadow_light_fraction=0.3, samples_per_pixel=1, enable_vertex_normals, enable_specular,
enable_emission, enable_per_light_ambient, splat_*, deterministic)`):
* `znear = vis.map.znear·stat.extent`, `zfar = vis.map.zfar·stat.extent`.
* Per active camera, rays precomputed (`_build_rays`, one thread per pixel, sub-pixel
  samples `n×n` by rendering at `n·W × n·H` and picking sample `(sx, sy)`) via
  `compute_ray(projection, fovy, sensorsize, intrinsic, W, H, px, py, znear)`:
  * perspective, intrinsics (`sensorsize[1] ≠ 0`): crop sensor to image aspect
    (`aspect > sensor_aspect → sensor_h = sensor_w/aspect`, `< → sensor_w = sensor_h·aspect`);
    `left = −znear/fx·(sw/2 − cx)`, `right = znear/fx·(sw/2 + cx)`,
    `top = znear/fy·(sh/2 − cy)`, `bottom = −znear/fy·(sh/2 + cy)`;
  * fovy: `half_h = znear·tan(fovy/2)`, `half_w = half_h·aspect`;
  * `u = (px+0.5)/W`, `v = (py+0.5)/H`, `x = left + (right−left)u`, `y = top + (bottom−top)v`;
    `dir = normalize(x, y, −znear)` in camera frame;
  * orthographic: `dir = (0,0,−1)`, origin offset `(x, y, 0)` with sensor height = fovy (extent).
  World ray: `origin = cam_xpos (+ cam_xmat·offset)`, `dir = cam_xmat·dir_local`.
  `use_precomputed_rays=False` computes rays in-kernel with per-world camera parameters.
* Geometry acceleration structures:
  * **Scene BVH** (`build_scene_bvh` / `refit_scene_bvh`): one `wp.Bvh` (SAH constructor)
    over `nworld × (bvh_ngeom + bvh_nflexgeom)` primitives, grouped by world (`group =
    worldid`, `group_root[world]` from `wp.bvh_get_group_root`). Per-primitive world AABBs
    (`_compute_bvh_bounds`): sphere `pos ± r`; capsule segment ends ± r; plane: 4 corners of
    a square of half-size `2·max(size0,size1)` (or 1000 if infinite) ± 0.01; ellipsoid:
    half-extent row norms of `R·diag(size)`; cylinder: `r·sqrt(Rx_i²+Ry_i²) + h|Rz_i|`;
    box/mesh/SDF: 8 rotated corners (mesh half = `max(|vmin|,|vmax|)` per axis of the mesh
    vertices); hfield: box centred `pos + z·half_z` from the hfield mesh extent. Flex
    primitives: dim-1 flex → one capsule AABB per edge (inflated by radius); dim≥2 → one
    AABB over all vertices. **`refit_bvh(m, d, rc)` must be called after each physics step
    before `render`** (and before BVH-accelerated `rays`).
  * **Mesh BVHs**: one `wp.Mesh` per mesh asset (`build_mesh_bvh`, SAH, leaf size 2), local
    vertices.
  * **Hfield BVHs**: `_optimize_hfield_mesh` greedy meshing: cells planar iff
    `|(z00+z11) − (z01+z10)| < 1e-5`; planar regions grown in +c then +r while coplanar
    (slopes within 1e-5) → large quads; triangles `(TL, TR, BR)`, `(TL, BR, BL)`; vertex
    `x = sx(c/((nc−1)/2) − 1)`, `y = sy(r/((nr−1)/2) − 1)`, `z = data·sz` (top only, no base).
  * **Flex BVHs** (dim≥2): `wp.Mesh` per flex with per-world groups, refit with updated
    vertex positions each frame (`refit_flex_bvh`), optional smooth vertex normals
    (`accumulate_flex_vertex_normals`, atomics; deterministic variant available).
  * **Splat BVH** (optional Gaussian splats, grouped per world).
* **Megakernel** (`_build_megakernel`, cached on `rc`): specialised at build time with
  `wp.static` to the geom types present (`geom_ray_types`) and to static flags; one thread
  per (world, global ray id) (`dim=(nworld, total_rays)`, `block_dim=m.block_dim.render=64`),
  global ray id mapped to (camera, pixel) by cumulative resolutions. For each pixel:
  1. `cast_ray` (closest hit): BVH traversal with `max_t = dist(1+1e-6)+1e-9`; per
     primitive: skip geoms with effective alpha 0 (`_geom_rgba`: if material and geom rgba
     is the default `(0.5,0.5,0.5,1)` use material rgba); primitive intersections from
     `ray.py`; mesh/SDF (SDF rendered as its mesh) via `ray_mesh_with_bvh`; flex edges as
     capsules; flex surfaces via flex BVH. Back-face culling (default on): drop hits with
     `dot(dir, n) > 0`. Tie rule: `closer = d < dist − tol || (|d − dist| ≤ tol && hit_id >
     geom_id)`, `tol = 1e-6·max(dist,1)` → ties go to the **higher** geom id (mimics GL depth test).
     Flex hits report `geom_id = −2` and flex id in `mesh_id`.
  2. zfar clip: miss if `dist·(−dir_local_z) > zfar`.
  3. Optional smooth normals for meshes: `n = normalize(R·Σ bary_k · vn_k)` with
     `vertex_normal(vn, face) = (vn·face < 0.8) ? face : vn` (matches `mjr_uploadMesh`).
  4. If culling disabled: flip normals facing away (two-sided shading).
  5. Splats composited front-to-back (≤32 hits per pass, alpha `min(exp(−½|δ|²)·opacity, 1)`,
     culled below `max(0.01, 1/255)`, stop when transmittance < 0.005).
  6. Segmentation `(geom_id, mjOBJ_GEOM)` or `(flex_id, mjOBJ_FLEX)`, background `(−1,−1)`;
     depth = planar `dist·(−dir_local_z)` (0 on miss); both only on sample 0.
  7. Miss: skybox (cube map sampled with MuJoCo's x/z/−y remap and face order
     +X,−X,+Y,−Y,+Z,−Z stacked vertically; `t` clamped half a texel from face edges) or
     background colour.
  8. Shading: `base = rgba.rgb` (× texture if material has texture role 1: plane/box/sphere/
     cylinder/capsule/hfield/mesh-specific UV generation reproducing `render_gl3.c` texgen,
     mesh UVs from barycentrics `uv0·u + uv1·v + uv2·(1−u−v)`);
     `result = emission·base` + ambient (`headlight.ambient·base` if headlight active, else
     `0.3·base` if the model has no lights; plus Σ active `light_ambient·base`) +
     Σ_lights `base ⊙ diffuse + specular` + headlight (directional along camera −z, no
     shadow); clamp to [0,1].
     Per light (`compute_lighting`): directional `L = −dir`; point/spot `L = normalize(pos −
     hit)`, attenuation `1/(a0 + a1 d + a2 d²)` unless default `(1,0,0)`; spot: zero if
     `cos θ < cos(cutoff)` else `× cos θ^exponent`; `ndotl = max(0, n·L)`; shadows (if
     enabled and light casts): any-hit ray from `hit + 1e-4 n` along L, `max_t = d − 1e-3`
     (1e8 directional) → `visible = shadow_light_fraction`; diffuse `= light_diffuse·ndotl·w`;
     specular (Blinn) `= light_specular·mat_specular·max(0, n·H)^(mat_shininess·128)·w`
     (defaults specular 0.5, exponent 64).
  9. Pixel packing `uint32 = (a<<24)|(r<<16)|(g<<8)|b` with `int(c·255)` truncation;
     supersampling accumulates `255·color` in float and resolves by averaging (`_aa_resolve`,
     clamp [0,255]).
* Readback helpers: `get_rgb` (unpack to float [0,1]), `get_depth(depth_scale)` (clamped
  `depth/scale` to [0,1]), `get_segmentation`.
* `fast_math` is enabled by default for the megakernel.

---

## 9. MJX (JAX) notes

* **`mjx/_src/ray.py`**: `ray(m, d, pnt, vec, geomgroup=(), flg_static=True,
  bodyexclude=-1 | sequence)` returns `(dist, geom_id)` — **no normals**. Supported types:
  PLANE, SPHERE, CAPSULE, ELLIPSOID, BOX, MESH (brute force, all triangles vmapped);
  **no CYLINDER, HFIELD, SDF**. All geoms are evaluated (no bounding-sphere prefilter), misses are
  `+inf`, final `argmin` (first minimal index — per-type groups are concatenated in
  `_RAY_FUNC` order PLANE, SPHERE, CAPSULE, ELLIPSOID, BOX, MESH, so ties resolve by type
  order then id, unlike C). `_ray_quad`: `det < mjMINVAL` rejects and `safe_div`.
  `_ray_box` uses `safe_div(size − pnt, vec)` for all 6 faces (division by `mjMINVAL`
  instead of skipping parallel axes). Triangle basis from `math.orthogonals(normalize(vec))`.
  Geom group mask indexes `geomgroup[clip(geom_group, 0, mjNGROUP)]`. Static filtering and
  visibility (rgba/material alpha) done with masks.
* **`mjx/_src/support.py`**: `jac`, `jac_dot`, `apply_ft = jacpᵀf + jacrᵀτ`,
  `xfrc_accumulate` (vmapped over bodies then summed), `mul_m`, `full_m`, `contact_force`
  (pyramidal/elliptic; **no adhesion subtraction**; masked by `efc_address ≥ 0`),
  `contact_force_dim`, name/id binding helpers, tendon wrapping and muscle helpers.
* **Rendering**: `mjx.render(m, d, ctx)` / `render_with_segmentation` only for
  `impl=WARP`, delegating to the Warp renderer through an FFI call with a JAX token.
* **Sleeping**: `EnableBit.SLEEP` is listed but MJX-JAX has no sleep logic; with
  `impl=WARP` the Warp implementation above applies.

---

## 10. Reproduction pitfalls / parity notes

**Ray casting**
1. `mj_ray` distances are in units of `|vec|` (not normalised), except SDF which returns
   Euclidean distance (direction normalised internally) — mixing these silently breaks
   unnormalised-direction callers.
2. Ties: C `mj_ray` → lowest geom id (strict `<`); `mj_multiRay` → first in body order
   then geom order (same as id order since geoms are body-sorted); Warp `_ray` → lowest id;
   Warp `_ray_bvh` → BVH order; Warp renderer → highest id; MJX → type order.
3. C mesh rays have **no back-face culling**; Warp BVH rays (renderer and `rays(rc=…)`)
   cull back-faces; Warp non-BVH `ray_mesh` does not cull (matches C up to float32).
4. `ray_quad` tangent handling differs (`det<0` vs `det<1e-15`); rays grazing spheres,
   capsules, cylinders, ellipsoids may hit in C and miss in Warp/MJX.
5. The triangle test depends on the projection basis only through rounding; C and Warp
   use different bases (C: `(1,1,1)`-Gram-Schmidt; Warp: Duff ONB which also assumes a
   unit direction). Hits exactly on triangle edges may flip.
6. `mj_multiRay` (camera rangefinder, perspective) uses aggressive culling: (a) only bodies
   with a BVH (≥1 collision-enabled geom: contype|conaffinity ≠ 0) get body-sphere and
   angular culling — bodies with purely visual geoms are not culled; (b) the body bounding
   sphere centre is the root BVH node centre (in the body **inertial** frame) added to
   `xipos` **without rotation**, radius = |half-extent|; this can produce false
   negatives when the root AABB is not centred on the COM; (c) angular bounds widen by
   `atan2(max_half, dist)` and fall back to "no culling" when spans exceed π. A faithful
   reimplementation must replicate these exactly; Warp does no culling, so camera
   rangefinders can differ from C where (b) misses.
7. `mj_multiRay` leaves `geomid`/`normal` unwritten for zero-length rays (dist = −1).
8. Rangefinder ignores geom groups (`geomgroup=NULL`), includes static geoms
   (`flg_static=1`) and excludes the sensor's own body; invisible geoms (alpha 0 or
   material alpha 0) are always skipped — changing rgba alpha changes sensor output.
9. Hfield side-wall logic uses `mju_round(y0)` on an already floored value and can override
   a closer triangle hit only if `all[i] < x`; replicate order (triangles first, then walls).
10. `mju_raySlab` relies on IEEE `±inf` from `1/0` and NaN-propagation via ternaries.

**Sleeping**
11. Countdown semantics: `−11 … −1` awake; the increment only happens while `< −1`;
    `mj_wakeIsland` on an awake tree applies `min(wakeval, current)`; contact/tendon wakes
    propagate the *awake neighbour's* countdown (C) vs fully-awake (Warp tendon wake).
12. `treeCanSleep` in `mj_wake` uses tolerance 0 → **any** nonzero qvel bit (incl. `−0.0`)
    wakes a sleeping tree; in `mj_sleep` uses `max_i dof_length_i |qvel_i| < tol` (strict)
    with early exit; `xfrc_applied`/`qfrc_applied` compared bytewise (C) / by value (Warp).
13. Sleeping happens inside `advanceStart` **before** velocity/position integration and
    triggers `mj_forwardSkip(POS)` (recomputing velocity/acceleration-stage results and
    sensors with zeroed qvel at the pre-step qpos). Warp sleeps at the end of `_advance`
    using post-integration qvel and refreshes only `fwd_velocity`. Trajectories differ in
    the step where sleeping occurs.
14. C detects external edits to sleeping trees by exact comparison of recomputed
    `xpos/xquat`; Warp does not (only velocity/applied forces wake). Float32 vs float64
    also makes "exact" comparisons meaningless across backends.
15. With SLEEP on, C never recomputes poses of STATIC bodies (world-welded, non-mocap)
    after reset — `mj_resetData` (or `mj_makeData` + reset) is mandatory; Warp never
    recomputes static geom poses regardless of SLEEP.
16. Sensors of sleeping objects keep their last value in C (stage skipped); in Warp they
    are recomputed every step (same values for position sensors; velocity sensors read
    exactly zero velocities, matching C's post-sleep recomputation).
17. `body_awake_ind`, `dof_awake_ind` are ascending in C; Warp builds them with atomics
    (arbitrary order unless deterministic mode). Any reimplementation iterating these lists
    must keep ascending order to reproduce C summation orders (e.g. `mj_comPos`).
18. `mj_comPos` with a sleeping child of an awake parent re-multiplies the child's stored
    normalised `subtree_com` by `body_subtreemass` — do not recompute the sleeping subtree.
19. Collision with sleep: both-asleep and asleep–world pairs are pruned in broadphase; a
    wake from contacts triggers a **second full `mj_collision`** in C (Warp: incremental
    pass appending only newly enabled pairs — contact ordering in the buffer therefore
    differs from C).
20. Unsupported in Warp: user sleep policies (NEVER/ALLOWED/INIT), flex equalities with
    sleep, non-Newton solvers with sleep, mocap-contact wake, metric-tendon wake,
    non-metric flex wake rule. Unsupported in C: tendon equality with sleep (error), FD
    derivatives with sleep, IPC with sleep.
21. Non-persistent `mjBatch` cannot run sleep (the sleep bookkeeping is not part of
    `mjSTATE_INTEGRATION`); similarly `mj_getState/mj_setState` do not capture
    `tree_asleep` — restoring a state does not restore sleep status.

**Support**
22. State vectors are concatenated in **bit order**, not the order of the flags in the
    user's expression; `eq_active` is stored as 0.0/1.0.
23. `mj_integratePos` normalises the quaternion **before** the rotation and not after
    (Warp normalises both before and after); `mj_differentiatePos` uses
    `2·atan2(|v|, w)` with wrap `> π → −2π` (no normalisation of inputs).
24. `mj_applyFT` (sparse vs dense) gives identical math but different summation order;
    `mj_xfrcAccumulate` applies forces at `xipos` (COM), skipping bodies whose 6 values are
    all exactly zero.
25. `mj_contactForce` subtracts `con->adhesion` from the normal component; MJX does not.
26. `mj_geomDistance` uses native CCD (unflipped geom order) for convex/box-box pairs when
    native CCD is enabled, otherwise the collision function with `margin = distmax`; the
    `fromto` sign uses `flip`. Result is clamped to `distmax`.
27. `mj_nextActivation`: FILTEREXACT and DC-motor current use the exact exponential filter
    with `tau ≥ mjMINVAL`; DC motor activations are never clamped to `actrange`; Warp returns
    USER-dyn activations unchanged.
28. History buffers: exact-time matches use `|Δt| < mjMINVAL`; read extrapolation clamps
    to oldest/newest with `±mjMINVAL` slack; cubic slopes are zero at the ends.

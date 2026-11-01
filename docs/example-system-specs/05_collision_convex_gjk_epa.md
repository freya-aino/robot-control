# 05 — General convex narrowphase: support functions, GJK, EPA, MultiCCD, MPR, height fields

**Sources covered**

| Backend | File | Scope |
|---|---|---|
| C engine | `src/engine/engine_collision_convex.c/.h` | `mjCCDObj`, all support functions (incl. mesh hill-climbing), libccd (MPR) wrapper vs native CCD selection, `mjc_Convex` (perturbation multi-contact), `mjc_PlaneConvex`, `mjc_ConvexHField` (prism decomposition), `mjc_fixNormal`, flex elem convex/hfield, `mjc_hausdorff` |
| C engine | `src/engine/engine_collision_gjk.c/.h` | native GJK (signed-volume subdistance), `gjkIntersect`, EPA (polytope2/3/4, horizon, witness), MultiCCD (face/edge feature recovery, polygon clipping), `mjc_ccd`, `mjc_gjk`, `mjc_ccdSize` |
| C (compile-time inputs) | `src/user/user_mesh.cc` (`MakeGraph`, `MakePolygons`, `MeshPolygonKey`), `src/user/user_model.cc` (mesh_extrema), `src/engine/engine_setconst.c` (npolygonmax / nmeshdegmax) | precomputed graph / polygon data consumed here |
| External | libccd `ccdMPRPenetration` (vendored dependency, not in repo) | used only when `mjDSBL_NATIVECCD` is set |
| MJWarp | `mujoco_warp/_src/collision_gjk.py` | `support`, `gjk`, `_epa`, polytope builders, `multicontact`, `gjk_phase`, `epa_phase`, `ccd` |
| MJWarp | `mujoco_warp/_src/collision_convex.py` | `convex_narrowphase`, `ccd_kernel_builder`, `ccd_hfield_kernel_builder`, `_hfield_filter`, buffer sizing |
| MJWarp (related) | `collision_driver.py` (`MJ_COLLISION_TABLE`), `collision_core.py` (`Geom`, `write_contact`), `collision_primitive.py` (`plane_convex`, `_hull4`) | dispatch and plane–mesh |
| MJX | `mjx/_src/collision_convex.py` | SAT-based convex colliders (not GJK/EPA) |

**Role in the pipeline.** After the broadphase produces geom pairs, `mj_narrowphase` (engine_collision_driver.c) calls `mjCOLLISIONFUNC[type1][type2](m, d, con, g1, g2, margin+gap)` per pair. Every pair that has no analytic primitive collider ends up in this subsystem: `mjc_Convex` (sphere–ellipsoid/mesh, capsule–ellipsoid/cylinder/mesh, ellipsoid–*, cylinder–cylinder/box/mesh, box–mesh, mesh–mesh), `mjc_PlaneConvex` (plane–ellipsoid/mesh/SDF), `mjc_ConvexHField` (hfield–everything except plane/hfield/SDF). These fill an `mjPreContact` array (`dist`, `pos`, `normal`, `tangent`) that the driver turns into `mjContact` (frame completed by `mju_makeFrame`). The same machinery is reused by flex collisions (`mjc_ConvexElem`, `mjc_HFieldElem`), `mj_geomDistance` (distance queries with `dist_cutoff>0`), continuous/speculative collision (`mjc_gjk`), and the `insidesite` sensor (`mjc_hausdorff`).

Key constants used throughout (double build):

| Symbol | Value | Where |
|---|---|---|
| `mjMINVAL` | 1e-15 | mjtype.h |
| `mjMAXVAL` | 1e10 | mjtype.h |
| `mjMINVAL2` | 1e-30 | both .c files |
| `mjMAXVAL2` | 1e20 | gjk.c |
| `mjMAX_LIMIT` | `DBL_MAX` (float: `FLT_MAX`) | gjk.h |
| `mjFACE_TOL` | 0.996 (≈ cos 5.1°) | gjk.h |
| `mjEDGE_TOL` | 0.0888 (≈ sin 5.1°) | gjk.h |
| `mjMINDIST2/3/4` | `mjMINVAL2`=1e-30 (float: 1e-10, 1e-10, 1e-17) | gjk.c |
| `mjMINEPATOL` | `mjMINVAL`=1e-15 (float: 1e-7) | gjk.c |
| `mjMAXCONPAIR` | 50 | mjmodel.h |
| `mjMESH_HILLCLIMB_MIN` | 10 | convex.h |
| `opt.ccd_iterations` default | 35 | engine_init.c |
| `opt.ccd_tolerance` default | 1e-6 | engine_init.c |
| `mjDSBL_NATIVECCD` | 1<<17 | mjtype.h |
| `mjDSBL_MULTICCD` | 1<<19 | mjtype.h |

---

## 1. Data structures

### 1.1 `mjCCDObj` (engine_collision_convex.h)

```
int geom;            // geom id; -1 for flex objects or site objects
int geom_type;       // mjtGeom (mjGEOM_FLEX for flex)
mjtNum size[4];      // geom_size (3) ; for hfield: hfield_size (4)
mjtNum pos[3];       // world position (or hfield-frame position for hfield partner)
mjtNum mat[9];       // world rotation, row-major (mat[3*i+j])
int vertindex;       // last support vertex (box corner bitmask, cylinder cap 0/1, mesh global vert id); -1 init
int meshindex;       // last hill-climb vertex (graph-local id); -1 init
int flex, elem, vert;
mjtNum margin;       // margin assigned to this object (full margin, not half)
mjtNum rotate[4];    // init to identity quat (1,0,0,0); unused by algorithms here
mjtNum suppvert[12]; // up to 4 support vertices on this object (simplex/EPA face) — output of GJK/EPA
int suppind[4];      // corresponding vertex indices (vertindex at time of support call)
int nsupp;           // number of valid suppvert entries
union data { mesh{nvert, mesh_polynum, vert*, mpolymapadr*, mpolymapnum*, polymap*, polyvertadr*,
                  polyvertnum*, polyvert*, polynormal*, graph*, extrema*},
             hfield{prism[6][3], hfield_data*, hfield_nrow, hfield_ncol},
             flex{elem*, dim*, aabb*, elemadr*, elemdataadr*, vert_xpos*, vertadr*, xradius*} };
center(res, obj);    // function pointer
support(res, obj, dir);  // function pointer; dir assumed unit length
```

Notes:
- Mesh vertex data `mesh_vert` is **float32** in mjModel; all dot products cast to `mjtNum` per component (`dot3f`).
- `polynormal`, `polyvertadr`, `polyvertnum`, `mpolymapadr`, `mpolymapnum` pointers are pre-offset by the mesh's `polyadr`/`vertadr`; `polymap` and `polyvert` are **not** offset (they contain mesh-local indices: polymap entries are mesh-local polygon ids, polyvert entries mesh-local vertex ids).

### 1.2 Mesh precomputation consumed here

**Convex hull graph** `mesh_graph + mesh_graphadr[mesh]` (built by `mjCMesh::MakeGraph` via qhull `"qhull Qt"`):
```
graph[0] = numvert, graph[1] = numface
vert_edgeadr [numvert]          at graph+2
vert_globalid[numvert]          at graph+2+numvert      (hull-local id -> mesh vertex id)
edge_localid [numvert+3*numface] at graph+2+2*numvert   (per hull vertex: neighbor hull-local ids, terminated by -1)
face_globalid[3*numface]        at graph+2+3*numvert+3*numface
```
`mesh_graphadr = -1` if no graph.

**Extrema** `mesh_extrema[27*mesh + k]` (user_model.cc): for k enumerating `cx,cy,cz ∈ {-1,0,1}` in nested order (cx outer, cz inner, `k = (cx+1)*9 + (cy+1)*3 + (cz+1)`), the hull-local id maximizing `x*cx + y*cy + z*cz` in **float**, strict `>` (first max wins; init `-FLT_MAX`-like `lowest()`). All zero if no graph.

**Polygons** (`MakePolygons`): coplanar hull triangles are merged into polygons keyed by rounded spherical angles of the face normal (`kAngleTol = 0.01` rad; `MeshPolygonKey`: `rtheta = round(atan2(ny,nx)/0.01)`, `rphi = round(acos(nz)/0.01)`; if `|nz| > 1-1e-7` → `(0, nz<0 ? round(pi/0.01) : 0)`; triangle skipped if `|cross| < mjMINVAL`). Each connected boundary path with ≥3 vertices becomes a polygon; `mesh_polynormal` = the representative normal from the key; `mesh_polyvert` = vertex loop; `mesh_polymap[mpolymapadr[v] .. +mpolymapnum[v]]` = polygons incident to vertex v.

**Global sizes** (engine_setconst.c, `mjSCGROUP_FIXED`): `npolygonmax = max_i mesh_polyvertnum[i]`, `nmeshdegmax = max_v mesh_polymapnum[v]` (max number of polygons incident on a vertex).

### 1.3 `mjCCDConfig` / `mjCCDStatus` (gjk.h)

```
mjCCDConfig { int max_iterations; mjtNum tolerance; int max_contacts; mjtNum dist_cutoff;
              void* buffer; int npolygonmax; int nmeshdegmax; }
mjCCDStatus { int separated; mjtNum dist[50]; mjtNum x1[150], x2[150]; int nx;
              int max_iterations; mjtNum tolerance; int max_contacts; mjtNum dist_cutoff;
              int gjk_iterations, epa_iterations; mjEPAStatus epa_status; }
mjEPAStatus: NOCONTACT=-1, SUCCESS=0, P2_INVALID_FACES=1, P2_NONCONVEX=2, P2_ORIGIN_ON_FACE=3,
             P3_BAD_NORMAL=4, P3_INVALID_V4=5, P3_INVALID_V5=6, P3_MISSING_ORIGIN=7,
             P3_ORIGIN_ON_FACE=8, P4_MISSING_ORIGIN=9
```
`dist_cutoff == 0` means "contacts only, no distances needed" (enables early-outs). `dist_cutoff > 0` means "recover distance if ≤ cutoff". `max_contacts == 0` means "distance only, no penetration recovery".

### 1.4 EPA internal types (gjk.c)

```
Vertex { mjtNum vert[3] /* v1-v2 */, vert1[3], vert2[3]; int index1, index2; }
Face   { int verts /* v1 + (v2<<10) + (v3<<20) */; int adj[3] /* faces across edges [v1v2],[v2v3],[v3v1] */;
         mjtNum v[3] /* projection of origin on face plane (outward) */; mjtNum dist2 /* |v|^2 */;
         int index /* position in map; -1 not in map; -2 deleted */; }
Polytope { Vertex* verts; int nverts; Face* faces; int nfaces, maxfaces; mjtNum center[3];
           Face** map; int nmap; Horizon { int* indices; int* edges; int nedges, maxedges; const mjtNum* w; } }
```
Vertex ids packed into 10 bits each → max 1024 vertices → EPA iterations capped at 1000.

### 1.5 Buffer sizing `mjc_ccdSize(npolygonmax, nmeshdegmax, iterations)`

```
maxedges   = 24 + max(0, iterations - 35)
epa_size   = A(sizeof(Vertex)*(5+N)) + A(sizeof(Face)*6N) + A(sizeof(Face*)*6N)
           + A(sizeof(int)*maxedges) * 2                                   // N = iterations, A = align8
npolygonmax = max(npolygonmax, 16); nmeshdegmax = max(nmeshdegmax, 3)
multiccd   = A(8*3*2*P)*2 + A(8*3*P) + A(8*P) + A(4*D)*2 + A(8*3*D)*3 + A(8*3*P)*2   // P=npolygonmax, D=nmeshdegmax
return max(epa_size, multiccd)
```
EPA and MultiCCD share the same buffer (MultiCCD runs after EPA finishes, reading only `obj->suppvert/suppind` and `status->x1/x2`).

Allocation: in `mj_narrowphase`, `ccd_size = mjc_ccdSize(MULTICCD disabled ? 0 : m->npolygonmax, ... nmeshdegmax, m->opt.ccd_iterations)`; a buffer of `nthread * ccd_size` bytes is arena/stack-allocated; each task calls `mjc_setCCDBuffer(epabuffer + thread_id*ccd_size)` (thread-local static `ccd_buffer`), reset to NULL at the end. If `ccd_buffer == NULL` (e.g., calls outside the driver), `mjc_penetration` allocates a fresh stack buffer per call (`mj_markStack/mj_freeStack`).

---

## 2. Object initialization

### 2.1 `mjc_initCCDObj(obj, m, d, g, margin)`
- `geom=g, margin=margin, center=mjc_center, vertindex=-1, meshindex=-1, nsupp=0, flex=elem=vert=-1, rotate=(1,0,0,0)`.
- If `g>=0`: `size[0..2]=geom_size[g]`, `pos=geom_xpos[g]`, `mat=geom_xmat[g]` (if d), `geom_type=geom_type[g]`, then `mjc_initCCDObjSupport(obj, m, geom_dataid[g])`.
- Else (flex): `geom_type=mjGEOM_FLEX`, `support=mjc_flexSupport`, flex pointers set to whole-model arrays (`flexelem_aabb`, `flexvert_xpos`, `flex_radius`, ...), later offset by `mjc_setCCDObjFlex`.

### 2.2 `mjc_initCCDObjSupport(obj, m, dataid)` — support function selection (native path)

| type | support | extra |
|---|---|---|
| SPHERE | `mjc_sphereSupport` | |
| CAPSULE | `mjc_capsuleSupport` | |
| ELLIPSOID | `mjc_ellipsoidSupport` | |
| CYLINDER | `mjc_cylinderSupport` | |
| BOX | `mjc_boxSupport` | |
| MESH, SDF | if `dataid<0`: NULL. if `mesh_graphadr[dataid] < 0 || mesh_vertnum[dataid] < 10`: `mjc_meshSupport` (exhaustive), `graph=extrema=NULL`; else `mjc_hillclimbSupport`, `graph=mesh_graph+graphadr`, `extrema=mesh_extrema+27*dataid` | sets all mesh/poly pointers, `mesh_polynum` |
| HFIELD | `mjc_prism_support`, `center=mjc_center` | if dataid≥0: `nrow,ncol`, `size[0..3]=hfield_size[4*dataid..]`, `hfield_data` |
| other | NULL | |

### 2.3 `mjc_initCCDObjSite` (used by insidesite sensor)
As above with `geom=-1`, sizes from `site_size`, `size[3]=0`, pose from `site_xpos/xmat`, `geom_type=site_type`. For mesh sites with `dataid>=0`: `mat = site_xmat * quat2Mat(mesh_quat)`, `pos = site_xmat*mesh_pos + site_xpos` (mesh frame offset applied).

### 2.4 `mjc_setCCDObjFlex(obj, flex, elem, vert)`
Offsets `vert_xpos += 3*flex_vertadr[flex]`; for elem≥0 `aabb += 6*(flex_elemadr[flex]+elem)`, `elem += flex_elemdataadr[flex] + elem*(dim+1)`.

### 2.5 `mjc_center(res, obj)`
- HFIELD: mean of the 6 prism vertices (`sum/6` computed as sum then `scl 1.0/6.0`).
- geom≥0: `obj->pos`.
- flex elem: `aabb[0..2]` (element AABB center).
- flex vertex: `vert_xpos + 3*vert`.

---

## 3. Support functions (native path, engine_collision_convex.c)

All take a **unit** direction `dir` in world frame, return world point `res`. **Margin is not added** by these (except flex); the GJK/EPA `support()` wrapper adds `0.5*margin*dir` (§5.2). Transform helpers:
- `mulMatTVec3(local, mat, dir)`: `local = matᵀ dir` (explicit expansion `mat[0]*d0+mat[3]*d1+mat[6]*d2`, ...).
- `localToGlobal(res, mat, l, pos)`: `res = mat l` then `+= pos` (adds after full product).

| Function | Math | Side effect |
|---|---|---|
| `mjc_pointSupport` | `res = pos` | — |
| `mjc_lineSupport` | `dot = mat[2]d0+mat[5]d1+mat[8]d2`; `s = dot>=0 ? L : -L` (L=size[1]); `res = (mat[2],mat[5],mat[8])·s + pos` | — |
| `mjc_sphereSupport` | `res = r·dir + pos` (no rotation) | — |
| `mjc_capsuleSupport` | `l = matᵀdir`; `s = r·l`; `s_z += (l_z>=0 ? L : -L)`; `res = mat s + pos` | — |
| `mjc_ellipsoidSupport` | `l = matᵀdir`; `t = l ∘ size`; `n2=|t|²`; if `n2 < 1e-30`: `res = (mat[0],mat[3],mat[6])·size[0] + pos`; else `t *= (1/√n2) ∘ size` (i.e. `t_i = l_i size_i · inv · size_i`); `res = mat t + pos` | — |
| `mjc_cylinderSupport` | `l = matᵀdir`; `n2 = l_x²+l_y²`; `scl = n2 >= 1e-30 ? size[0]/√n2 : 0`; `s=(scl l_x, scl l_y, l_z>=0 ? size[1] : -size[1])`; `res = mat s + pos` | `vertindex = l_z>=0 ? 0 : 1` (cap id) |
| `mjc_boxSupport` | `s_i = l_i>=0 ? size_i : -size_i`; `res = mat s + pos` | `vertindex = (s0>0)·1 | (s1>0)·2 | (s2>0)·4` (corner bitmask) |
| `mjc_meshSupport` (exhaustive) | `l = matᵀdir`; `max=-FLT_MAX, imax=0`; if `vertindex>=0`: `imax=vertindex, max=dot3f(l, v[imax])`; loop i=0..nvert-1: `if dot > max` (strict) update; `res = mat v[imax] + pos` | `vertindex = imax` (warm start = previous vertex; ties keep the cached vertex) |
| `mjc_hillclimbSupport` | see §3.1 | `meshindex` (graph-local), `vertindex` (global) |
| `mjc_prism_support` (hfield) | `istart = dir_z<0 ? 0 : 3`; best of prism[istart..istart+2] by `dot(prism[i],dir)` strict `>` (first wins); `res = prism[ibest]` (hfield-local frame) | none (vertindex stays -1) |
| `mjc_flexSupport` | elem: vertex of element (0..dim) with max dot (strict >, first wins), `res += dir·(xradius[f] + 0.5·margin)`; vertex: `res = vert + dir·(xradius + 0.5·margin)` | — |

### 3.1 Mesh hill-climbing `mjc_hillclimbSupport`
```
numvert = graph[0]; vert_edgeadr = graph+2; vert_globalid = graph+2+numvert; edge_localid = graph+2+2*numvert
l = matᵀ dir
cx = (l0 > 0.4) - (l0 < -0.4) + 1   // likewise cy, cz  (0,1,2)
grid_idx = extrema[cx*9 + cy*3 + cz]
if meshindex >= 0:
    cached = dot3f(l, v[globalid[meshindex]]); seed = dot3f(l, v[globalid[grid_idx]])
    imax = seed > cached ? grid_idx : meshindex          // ties -> cached
else imax = grid_idx
max = dot3f(l, v[globalid[imax]]); prev = -1
while imax != prev:
    prev = imax
    for i = vert_edgeadr[imax]; (sub = edge_localid[i]) >= 0; i++:
        d = dot3f(l, v[globalid[sub]]); if d > max: max = d; imax = sub     // updates imax mid-scan, scan continues over the ORIGINAL prev's neighbor list
meshindex = imax; vertindex = globalid[imax]; res = mat v[vertindex] + pos
```
Steepest-ascent within a neighbor list: the scan of `prev`'s neighbors is completed even after imax changes; then the next pass restarts from the new imax. Termination when a full neighbor pass yields no improvement.

### 3.2 Support selection inside libccd path: `mjccd_support(obj, dir, vec)`
Single monolithic function used only when `mjDSBL_NATIVECCD` is set (libccd callbacks):
- flex (geom<0): identical to `mjc_flexSupport`.
- `local_dir = mju_mulMatTVec3(mat, dir)`; per type (local frame result `res`):
  - SPHERE: `res = size0·l`
  - CAPSULE: `res = size0·l; res_z += mju_sign(l_z)·size1` (**mju_sign(0)=0** → at l_z=0 the point is on the cylinder mid-plane — differs from native)
  - ELLIPSOID: `res_i = l_i size_i`, `mju_normalize3(res)` (fallback (1,0,0) if norm<mjMINVAL), `res_i *= size_i`
  - CYLINDER: `t = √(l_x²+l_y²)`; if `t > mjMINVAL`: `res_xy = l_xy/t·size0` else 0; `res_z = mju_sign(l_z)·size1`
  - BOX: `res_i = mju_sign(l_i)·size_i` (0 at l_i=0)
  - MESH/SDF: `tmp=-1e10, ibest=-1`; if `graph==NULL`: exhaustive strict `>` over all verts, `meshindex=ibest` (global id); else hill-climb starting at `meshindex<0 ? 0 : meshindex` (**no extrema seed**) with `change` flag loop (same neighbor-scan semantics as 3.1), `meshindex=ibest` (local), `ibest=globalid[ibest]`. If `ibest<0` warning + zero.
  - HFIELD: `mjc_prism_support(res, obj, dir)` then `res += 0.5·margin·dir`, **return** (prism is already in the hfield frame; no rotation).
- then `res += l·margin/2`, `res = mat·res` (`mju_mulMatVec3`), `res += pos`.

Note libccd path adds margin in the support itself, whereas native path adds margin in the GJK wrapper.

---

## 4. Narrowphase entry points (engine_collision_convex.c)

### 4.1 `mjc_penetration(m, d, obj1, obj2, con, nconmax, margin)` — path selection
```
if mjDISABLED(mjDSBL_NATIVECCD): return _libccd_wrapper(m, obj1, obj2, con, margin)
config = {max_iterations = opt.ccd_iterations, tolerance = opt.ccd_tolerance, max_contacts = nconmax,
          dist_cutoff = 0, npolygonmax = m->npolygonmax, nmeshdegmax = m->nmeshdegmax,
          buffer = ccd_buffer or stack(mjc_ccdSize(MULTICCD?0:npolygonmax, MULTICCD?0:nmeshdegmax, iters))}
if (dist = mjc_ccd(&config, &status, obj1, obj2)) < 0:
    for i < status.nx:
        con[i].dist   = margin + status.dist[i]
        con[i].pos    = 0.5*(x1_i + x2_i)          // add then scale 0.5
        con[i].normal = normalize(x1_i - x2_i)     // mju_normalize3: (1,0,0) if norm < 1e-15
        con[i].tangent = 0
    ncon = status.nx
```
`margin` here is the value passed to the collision function, i.e. **`margin + gap`** from the driver. The objects were initialized with that same full margin, so each shape is inflated by `margin/2` and `+margin` converts back to the true surface distance. A contact is produced iff the inflated shapes penetrate (dist<0), i.e. true dist < margin+gap.

### 4.2 `_libccd_wrapper` (MPR path)
```
CCD_INIT(&ccd); ccd.mpr_tolerance = ccd.epa_tolerance = opt.ccd_tolerance; ccd.max_iterations = opt.ccd_iterations
ccd.support1 = ccd.support2 = mjccd_support; ccd.center1 = ccd.center2 = mjccd_center
ccd.first_dir = (either geom is HFIELD) ? prism_firstdir /* (0,0,1) */ : ccdFirstDirDefault
ret = ccdMPRPenetration(obj1, obj2, &ccd, &depth, &dir, &pos)
if ret == 0:
    if dir == (0,0,0) exactly: return 0          // touching contact discarded
    con[0] = {dist = margin - depth, normal = dir, pos = pos, tangent = 0}; return 1
return 0
```
(`first_dir` is only used by libccd's GJK, not MPR; it has no effect here.)

**libccd MPR (upstream algorithm, external; summarized for reproduction)** — all in double (`CCD_EPS = DBL_EPSILON`, `ccdIsZero(x) = |x| < EPS`, `ccdEq(a,b)`: `|a-b|<EPS` or relative `|a-b| < EPS·max(|a|,|b|)`). Minkowski points are `support1(d) - support2(-d)`; portal `v0..v4`.
1. *discoverPortal*: `v0 = center1 - center2`; if `v0 == 0` set `v0 = (10·EPS, 0, 0)`. `d = normalize(-v0)`; `v1 = S(d)`; if `dot(v1,d) <= 0` → no contact. `d = v0 × v1`; if `d==0`: return 1 if `v1==0` (touch) else 2 (origin on segment). `d=normalize(d)`, `v2=S(d)`, if `dot(v2,d)<=0` → none. `d = normalize((v1-v0)×(v2-v0))`; if `dot(d,v0)>0` swap v1,v2 and negate d. Loop: `v3=S(d)`; if `dot(v3,d)<=0` → none; if `dot(v1×v3, v0) < 0`: `v2=v3`, recompute `d`, continue; else if `dot(v3×v2, v0) < 0`: `v1=v3`, recompute, continue; else break.
2. res 1 → touch: depth 0, dir 0 (MuJoCo discards). res 2 → `dir = v1`, `depth=|v1|`, normalize, `pos = 0.5(v1.supp1+v1.supp2)`.
3. *refinePortal* (res 0): loop: `d = normalize((v2-v1)×(v3-v1))`; if `dot(d,v1) >= 0` (or ≈0) → origin enclosed, go to 4; `v4=S(d)`; if `dot(v4,d)` < 0 (not ≈0) or *reachTolerance* → no contact; *expandPortal*.
   - reachTolerance: `min(dot(v4-v3,d), dot(v4-v1,d), dot(v4-v2,d))` ≤ mpr_tolerance (or ccdEq).
   - expandPortal: `t = v4 × v0`; if `dot(v1,t)>0`: (if `dot(v2,t)>0` replace v1 else replace v3) else (if `dot(v3,t)>0` replace v2 else replace v1) with v4.
4. *findPenetr*: `it=0`; loop: `d = portal normal`; `v4=S(d)`; if reachTolerance or `it > max_iterations`: `depth² = pointTriDist2(origin, v1,v2,v3, &dir)`; `depth=√`; `dir = 0` if depth≈0 else normalize(dir); `pos = findPos`; return. else expandPortal, `it++`.
   - findPos: barycentric weights `b0=dot(v1×v2,v3)`, `b1=dot(v3×v2,v0)`, `b2=dot(v0×v1,v3)`, `b3=dot(v2×v1,v0)`; if sum ≤ 0 (≈0): `b0=0, b1=dot(v2×v3,d), b2=dot(v3×v1,d), b3=dot(v1×v2,d)`; `p1=Σ b_i v_i.supp1 / Σb`, `p2` likewise; `pos = 0.5(p1+p2)`.

The MPR normal points along `A−B` from origin to the portal, i.e. from geom1 toward geom2 penetration direction; afterward `mjc_fixNormal` may replace it for smooth shapes (§4.7).

### 4.3 `maxContacts(m, obj1, obj2)` — single-pass contact budget (native only)
```
if NATIVECCD disabled: return 1
if both BOX: return 8                          // independent of MULTICCD
if MULTICCD disabled: return 1
hasface = BOX|MESH|CYLINDER ; hasedge = CAPSULE
face-face → 4 ; edge-face / face-edge / edge-edge → 2 ; else 1
```

### 4.4 `mjc_Convex(m, d, con, g1, g2, margin)`
```
init obj1, obj2 with margin (full margin on both)
max_contacts = maxContacts(...)
isellipsoid_i = type_i ∈ {ELLIPSOID, SPHERE}
ncon = mjc_penetration(m, d, &obj1, &obj2, con, max_contacts, margin)
if NATIVECCD disabled && ncon && g1>=0 && g2>=0: mjc_fixNormal(m, d, con, g1, g2)
if max_contacts > 1 || isellipsoid1 || isellipsoid2: return ncon
// perturbation multi-contact (only reached when max_contacts == 1:
//   libccd path, or native with MULTICCD disabled ... but then next condition fails,
//   or native+MULTICCD for pairs without faces/edges e.g. capsule-capsule via mjc_Convex)
if ncon == 1 && !MULTICCD disabled:
    relative_tolerance = 1e-3 ; perturbation_angle = 1e-3 (rad)
    frame = [con[0].normal, 0, 0]; mju_makeFrame(frame)        // axes frame+3 (tangent1), frame+6 (tangent2)
    tol = 1e-3 * min(geom_rbound[g1], geom_rbound[g2])
    for axis in (frame+3, frame+6):            // outer loop over axes
      for angle in (-1e-3, +1e-3):             // inner loop over angles
        quat = axisAngle2Quat(axis, angle); rot = quat2Mat(quat)
        mju_rotateFrame(con[0].pos, rot,  obj1.mat, obj1.pos)    // rotate g1 about contact point
        mju_rotateFrame(con[0].pos, rotᵀ, obj2.mat, obj2.pos)    // inverse-rotate g2
        n = mjc_penetration(..., con + ncon, 1, margin)
        if NATIVECCD disabled && n && g1,g2>=0: mjc_fixNormal(con+ncon)
        if n && mjc_isDistinctContact(con, ncon+1, tol):
            con[ncon].dist = con[0].dist ; ncon++              // depth copied from initial contact
        reset obj1/obj2 pos & mat from d->geom_xpos/xmat
return ncon   // ≤ 5
```
- `mju_rotateFrame(origin, rot, xmat, xpos)`: `xmat = rot·xmat`; `rel = origin - xpos`; `vec = rot·rel - rel`; `xpos -= vec`.
- `mjc_isDistinctContact(con, n, tol)`: new contact `con[n-1]` is distinct iff `dist3(con[i].pos, new.pos) > tol` for all i < n-1.
- Note: obj warm-start indices (`vertindex/meshindex`) persist across the 5 calls.
- Driver capacity for this path: `mj_maxContact` returns 5 when NATIVECCD is disabled and MULTICCD enabled (for non-sphere/ellipsoid, non-primitive pairs).

### 4.5 `mjc_PlaneConvex(m, d, con, g1=plane, g2, margin)` (plane–ellipsoid, plane–mesh, plane–SDF)
```
normal = plane z-axis (mat1[2], mat1[5], mat1[8])
init obj (g2) with margin 0
v = obj.support(-normal)                  // native support fn regardless of NATIVECCD
dist = dot(normal, v - pos1); if dist > margin: return 0
con[0] = {dist, pos = v - 0.5*dist*normal, normal, tangent=0}
if geom_dataid[g2] == -1 || mesh_polynum == 0 || obj.vertindex < 0: return 1
polymap of support vertex: adr = mesh_polymapadr[vertadr+vertindex], num = mesh_polymapnum[...]
local_normal = mat2ᵀ normal
best_poly = argmin over incident polygons of dot(polynormal, local_normal), strict <, init best_dot = 1
if best_poly < 0: return 1
face = polygon best_poly (vertadr/vertnum from mesh_polyvertadr/num[polyadr+best_poly])
a = index in face of obj.vertindex (0 if absent)
n = hull4f(indices, vert, face_polyvert, face_vertnum, a)       // ≤4, anchor first
for i = 1..n-1:
    pnt = mat2·v_local + pos2 ; vdist = dot(normal, pnt - pos1)
    if vdist > margin || dot(normal, pnt - pos2) > 0: skip      // above margin or above mesh center
    con[ncon++] = {vdist, pnt - 0.5*vdist*normal, normal, 0}
return ncon   // ≤ 4
```
`hull4f(res, hull(float), idx, nhull, a)`: anchored maximum-area quadrilateral (rotating-caliper-like). `b=a+1, c=a+2, d=a+3 (mod nhull)`; if `nhull<=4` return nhull (indices a..a+nhull-1). `m = area4f(a,b,c,d)` where `area4f = 0.5·|(v_a−v_c)×(v_b−v_d)|` (float→double subtraction per component). Loop: try `d+1`; if `area ≤ m` break; else accept, then greedily advance `c` while strictly increasing, then advance `b` while strictly increasing; repeat. Returns 4.

### 4.6 `mjc_ConvexHField(m, d, con, g1=hfield, g2, margin)`
```
hid = geom_dataid[g1]; nrow, ncol; size0..3 = hfield_size[4*hid ..]
local_pos = mat1ᵀ (pos2 - pos1)
radius = geom_rbound[g2] + margin
// box-sphere early-out
if size0 < lp0-radius || -size0 > lp0+radius || size1 < lp1-radius || -size1 > lp1+radius
   || size2 < lp2-radius || -size3 > lp2+radius: return 0
init obj1 (hfield, margin), obj2 (g2, margin 0)
obj2.mat = mat1ᵀ mat2 ; obj2.pos = local_pos            // geom2 expressed in hfield frame
xmax = obj2.support(+x)[0], xmin = support(-x)[0], ymax, ymin, zmax, zmin likewise (unit axes)
xmin -= margin; xmax += margin; ... (all six)
if xmin > size0 || xmax < -size0 || ymin > size1 || ymax < -size1 || zmin > size2 || zmax < -size3: return 0
cmin = floor((xmin+size0)/(2*size0)*(ncol-1)); cmax = ceil((xmax+size0)/(2*size0)*(ncol-1))
rmin = floor((ymin+size1)/(2*size1)*(nrow-1)); rmax = ceil((ymax+size1)/(2*size1)*(nrow-1))
clamp cmin>=0, cmax<=ncol-1, rmin>=0, rmax<=nrow-1
obj2.margin = margin
dx = 2*size0/(ncol-1); dy = 2*size1/(nrow-1)
prism[0..2].z = -size3                                   // bottom (never changed again)
ncon = 0
for r in [rmin, rmax):
    addPrismVert(r, cmin, i=0); addPrismVert(r, cmin, i=1)
    for c in [cmin+1, cmax]:
        for i in {0,1}:
            addPrismVert(r, c, i)
            if prism[3].z < zmin && prism[4].z < zmin && prism[5].z < zmin: continue
            if mjc_penetration(m, d, &obj1, &obj2, con+ncon, 1, margin):
                con[ncon].normal = mat1 · normal_local ; con[ncon].pos = mat1 · pos_local + pos1
                if ++ncon >= mjMAXCONPAIR: break out of all loops
if NATIVECCD disabled: mjc_fixNormal on every contact
return ncon
```
`addPrismVert(obj, r, c, i, dx, dy)`: shift `prism[0]←prism[1], prism[1]←prism[2], prism[3]←prism[4], prism[4]←prism[5]` (full xyz copies; bottom z remains -size3); with `dr = 1-i`: `prism[2].xy = prism[5].xy = (dx·c − size0, dy·(r+dr) − size1)`, `prism[5].z = hfield_data[(r+dr)·ncol + c]·size2`. So for i=0 the new vertex is at row r+1, for i=1 at row r. After the two init calls slots 1,2 hold (cmin,r+1),(cmin,r). Each cell (r, c-1..c) yields triangles {(c-1,r+1),(c-1,r),(c,r+1)} then {(c-1,r),(c,r+1),(c,r)}.

The prism is a triangular prism with top vertices `prism[3..5]` (terrain heights) and bottom vertices `prism[0..2]` at `z=-size3` (so prisms extend down to the hfield base). One contact per prism max; up to 50 total. Note obj2 warm-start indices persist across prisms. Hfield prism support never sets `vertindex` (stays -1), which matters for EPA repeated-vertex detection (§6.6) and MultiCCD (not used: max_contacts=1).

### 4.7 `mjc_fixNormal(m, d, con, g1, g2)` — libccd normal correction
Only for types SPHERE, CAPSULE, ELLIPSOID, CYLINDER (others → NONE). If neither processable return. `normal[0]=con.normal`, `normal[1]=-con.normal`. For each processable geom i: `pos1 = matᵀ(con.pos − xpos)`, `nrm = matᵀ normal[i]`:
- SPHERE: `nrm = pos1`.
- CAPSULE: `nrm_z = pos1_z + size1` if `pos1_z < -size1`; `pos1_z − size1` if `> size1`; else 0. `nrm_xy = pos1_xy`.
- ELLIPSOID: skip if any `size < mjMINVAL`. `dst = Σ pos1_i²/size_i²`; if `dst <= 1` → `mjc_ellipsoidInside` else `mjc_ellipsoidOutside`.
- CYLINDER: skip if `|pos1_z| > 0.95·size1`; `dst1 = |size1 − |pos1_z||`, `dst2 = |size0 − ‖pos1_xy‖|`; skip if `dst1 < 0.25·dst2`; `nrm = (pos1_x, pos1_y, 0)`.
- if processed: normalize, `normal[i] = mat·nrm`.
Combine: both → `con.normal = normalize(normal[0] − normal[1])`; only 0 → `normal[0]`; only 1 → `−normal[1]`.

`mjc_ellipsoidInside(nrm, pos, size)` (point inside or on): `S2inv_i = 1/size_i²`, `C = Σ pos_i² S2inv_i − 1`; if `C>0` return 0; normalize nrm; up to 30 iters: `A = Σ nrm_i² S2inv_i`, `B = Σ pos_i nrm_i S2inv_i`, `det = B²−AC`; if `det<mjMINVAL || A<mjMINVAL` return iter>0; `x = (−B+√det)/A`; if `x<0` return iter>0; `pnt = pos + x nrm`; `newnrm = normalize(pnt ∘ S2inv)`; `change = ‖nrm − newnrm‖`; `nrm=newnrm`; break if `change < 1e-6`. return 1.

`mjc_ellipsoidOutside(nrm, pos, size)`: `S2 = size²`, `PS2_i = pos_i² S2_i`, `la=0`; up to 30 iters: `R_i = 1/(S2_i+la)`; `val = Σ PS2_i R_i² − 1`; break if `val < 1e-6`; `deriv = −2 Σ PS2_i R_i³`; break if `deriv > −mjMINVAL`; `delta = −val/deriv`; break if `delta < 1e-6`; `la += delta`. `nrm = normalize(pos_i/(S2_i+la))`. return 1. (Comments in source say "inside/outside" swapped; code is as described.)

### 4.8 Flex: `mjc_ConvexElem`, `mjc_HFieldElem`
- `mjc_ConvexElem(g1,f1,e1,v1,f2,e2,margin)`: obj1 from geom g1 (or flex if g1<0) + `setCCDObjFlex(f1,e1,v1)`; obj2 flex `(f2,e2,-1)`; `ncon = mjc_penetration(..., 1, margin)`. If native and ncon and either flex has dim 2: `mjc_fixNormal(con+i, g1, -1)` (only acts if g1 is a smooth geom).
- `mjc_HFieldElem(g,f,e,margin)`: element vertices `evert[i] = hmatᵀ(v − hpos)` and center from AABB transformed likewise; obj2 is a flex object pointed at these local arrays (`local_edata={0,1,2,3}`). AABB of evert expanded by `flex_radius[f]+margin`; same box test and subgrid formulas as 4.6; prism construction via `addVert(x,y,z)` with `dr={1,0}` and a per-row `nvert` counter (penetration only once `++nvert > 2`); same height test; contacts rotated/translated back to world; stop at 50.

### 4.9 `mjc_hausdorff(obj1, obj2, nitermax, stepsize, tol)` (insidesite sensor; stepsize 0.5, nitermax = ccd_iterations, tol = ccd_tolerance)
Lower bound of the directed Hausdorff distance `max_{|x|=1} h_A(x) − h_B(x)` by projected subgradient ascent on S² from 6 seeds (±columns of obj2.mat, seed s: axis s/2, sign + for even s). Errors if types not in [SPHERE, MESH]. Per active seed each iteration: `vert = S1(x) − S2(x)`, `val = dot(vert,x)`, `grad = vert − val·x`.
- if `val > best_val`: store best_x; deactivate if `‖grad‖ ≤ mjMINVAL || ‖grad‖ ≤ tol·|val|`; else `best_grad = grad/‖grad‖`.
- else: `step *= 0.5`; deactivate if `step < tol`; if `‖grad‖>mjMINVAL` `best_grad += grad/‖grad‖`; project out `best_x` component; deactivate if norm ≤ tol; normalize.
- `x = normalize(best_x + step·best_grad)`.
Return `max_s best_val[s]` (init `-mjMAXVAL`). Uses native supports without margin.

---

## 5. Native GJK (engine_collision_gjk.c)

### 5.1 `discreteGeom(obj)`
Returns 1 iff `obj->margin == 0` and (`support == mjc_pointSupport || support == mjc_lineSupport` or type ∈ {MESH, BOX, HFIELD}). A pair is *discrete* iff both are; discrete → finite convergence → zero tolerances.

### 5.2 Minkowski support wrapper `support(v, obj1, obj2, dir, dir_neg)`
```
v.vert1 = obj1.support(dir);     if obj1.margin > 0 && obj1.geom >= 0: v.vert1 += 0.5*margin1*dir
v.vert2 = obj2.support(dir_neg); if obj2.margin > 0 && obj2.geom >= 0: v.vert2 += 0.5*margin2*dir_neg
v.vert = v.vert1 - v.vert2 ; v.index1 = obj1.vertindex ; v.index2 = obj2.vertindex
```
- `gjkSupport(v, x_k, x_norm)`: `dir_neg = x_k·(1/x_norm)` (multiply by reciprocal), `dir = −dir_neg`.
- `epaSupport(pt, d, dnorm)`: `dir = d/dnorm` (per-component divide) if `dnorm > mjMINVAL`, else `dir=(1,0,0)`; `dir_neg = −dir`; appends vertex at `pt->nverts++`.
- `gjkIntersectSupport(v, dir)`: no normalization, `dir_neg = −dir`.

### 5.3 `gjk(status, obj1, obj2)`
```
get_dist   = status.dist_cutoff > 0
backup_gjk = !get_dist
simplex[12] local; simplex1/2 = obj1/2->suppvert; index1/2 = obj->suppind
n = 0; k = 0; kmax = status.max_iterations
x1_k, x2_k = status.x1, status.x2 (initial guesses = centers)
tol2 = tolerance²; separated = 0; obj1.nsupp = obj2.nsupp = 0
discrete = discreteGeom(obj1) && discreteGeom(obj2)
epsilon  = discrete ? 0 : 0.5*tol2
min_norm = (discrete || get_dist) ? mjMINVAL : tolerance
x_k = x1_k − x2_k ; x_norm = |x_k| ; x_norm_prev = 0
for (; k < kmax; k++):
    if x_norm < min_norm || |x_norm_prev − x_norm| < mjMINVAL: break
    v = gjkSupport(x_k, x_norm); s_k = simplex[n] = v.vert; simplex1/2[n] = v.vert1/2; index1/2[n] = v.index1/2
    if dot(x_k, x_k − s_k) < epsilon: break                     // Frank–Wolfe duality gap
    lower = dot(x_k, s_k)
    if !get_dist:
        if lower > 0: separated=1; gjk_iterations=k; nx=0; dist[0]=MAX_LIMIT; return
    elif dist_cutoff < MAX_LIMIT:
        if lower > 0 && lower >= dist_cutoff * x_norm: (same separated return)
    if n == 3 && backup_gjk:                                     // 4 vertices now in simplex
        gjk_iterations = k; ret = gjkIntersect(status, obj1, obj2)
        if ret != −1: nx=0; separated = (ret==0); dist[0] = ret>0 ? 0 : MAX_LIMIT; return
        k = gjk_iterations; backup_gjk = 0
    lambda = subdistance(n+1, simplex)
    compact: keep entries with lambda[i] != 0 (exact compare), preserving order; n = count
    if n < 1: (SHOULD NOT OCCUR) separated return with dist MAX_LIMIT
    x_k = lincomb(lambda, n, simplex); x_norm_prev = x_norm; x_norm = |x_k|
    if n == 4: break
if n > 0: x1_k = lincomb(lambda, n, simplex1); x2_k = lincomb(lambda, n, simplex2)
tmp = gjkSupport(x_k, x_norm); if dot(x_k, tmp.vert) > 0: separated = 1      // final separation check
nx = 1; gjk_iterations = k; obj1.nsupp = obj2.nsupp = n
dist[0] = (n == 4 && !separated) ? 0 : x_norm
```
Note: the final check divides by `x_norm` (may be 0 → inf/NaN direction when converged exactly to the origin; the dot with x_k=0 is then 0 or NaN, not >0, so not separated).

`lincomb(res, coef, n, v1..v4)`: explicit sums in order `c0 v1 + c1 v2 + c2 v3 + c3 v4` (left to right), case n=0 → 0.

### 5.4 Signed-volume subdistance (Montanari et al. 2017)

`subdistance(lambda, n, simplex)`: zero lambda; n=4 → S3D, 3 → S2D, 2 → S1D, else `lambda[0]=1`.

`sameSign2(a,b)`: 1 if both >0, −1 if both <0, else 0 (zero counts as "not same").

**S3D(s1..s4)**:
```
C41 = −det3(s2,s3,s4); C42 = det3(s1,s3,s4); C43 = −det3(s1,s2,s4); C44 = det3(s1,s2,s3)
det3(a,b,c) = a0(b1c2 − b2c1) + a1(b2c0 − b0c2) + a2(b0c1 − b1c0)
m_det = C41 + C42 + C43 + C44
comp_i = sameSign2(m_det, C4i)
if all comp: lambda = C4i / m_det; return
dmin = MAX_LIMIT
if !comp1: l = S2D(s2,s3,s4); x = lincomb; d=|x|²; lambda=(0,l0,l1,l2); dmin=d     // unconditional
if !comp2: l = S2D(s1,s3,s4); if d < dmin: lambda=(l0,0,l1,l2)
if !comp3: l = S2D(s1,s2,s4); if d < dmin: lambda=(l0,l1,0,l2)
if !comp4: l = S2D(s1,s2,s3); if d < dmin: lambda=(l0,l1,l2,0)
```
**S2D(s1,s2,s3)**:
```
if projectOriginPlane(p_o, s1,s2,s3) fails: lambda = (S1D(s1,s2), 0); return
M_14 = s2y s3z − s2z s3y − s1y s3z + s1z s3y + s1y s2z − s1z s2y        // exact term order
M_24 = s2x s3z − s2z s3x − s1x s3z + s1z s3x + s1x s2z − s1z s2x
M_34 = s2x s3y − s2y s3x − s1x s3y + s1y s3x + s1x s2y − s1y s2x
choose projection dropping axis with largest |M|: if |M14|≥|M24| && |M14|≥|M34| → (y,z), M_max=M14;
  elif |M24|≥|M34| → (x,z), M_max=M24; else (x,y), M_max=M34
C31 = p0 b1 + p1 c0 + b0 c1 − p0 c1 − p1 b0 − c0 b1     (p=p_o2D, b=s2_2D, c=s3_2D)
C32 = p0 c1 + p1 a0 + c0 a1 − p0 a1 − p1 c0 − a0 c1     (a=s1_2D)
C33 = p0 a1 + p1 b0 + a0 b1 − p0 b1 − p1 a0 − b0 a1
comp_i = sameSign2(M_max, C3i); all → lambda = C3i/M_max
else edge candidates (S1D) in order (s2,s3) [unconditional first], (s1,s3), (s1,s2), strict d < dmin
```
**S1D(s1,s2)**:
```
p_o = projectOriginLine(s1,s2):  diff = s2 − s1; scl = −(dot(s2,diff)/dot(diff,diff)); p_o = s2 + scl·diff
mu_max = s1x − s2x, index 0; if |s1y−s2y| ≥ |mu_max|: index 1; if |s1z−s2z| ≥ |mu_max|: index 2
C1 = p_o[idx] − s2[idx]; C2 = s1[idx] − p_o[idx]
same = sameSign2(mu_max,C1) && sameSign2(mu_max,C2)
lambda = same ? (C1/mu_max, C2/mu_max) : (0, 1)
```
(When outside, the vertex s2 is always chosen — the newest vertex in GJK ordering.)

**projectOriginPlane(res, v1,v2,v3)** — three attempts for robustness:
```
d21=v2−v1, d31=v3−v1, d32=v3−v2
n = d32 × d21; nv = dot(n,v2); nn = dot(n,n); if nn == 0: return 1
  if nv != 0 && nn > mjMINVAL: res = n·(nv/nn); return 0
n = d21 × d31; nv = dot(n,v1); nn; if nn == 0 return 1; if nv != 0 && nn > mjMINVAL: res=..., return 0
n = d31 × d32; nv = dot(n,v3); nn; res = n·(nv/nn); return 0       // no checks
```

### 5.5 `gjkIntersect(status, obj1, obj2)` — boolean intersection fallback (contacts-only mode)
Copies the 4 simplex vertices from `obj->suppvert/suppind`. `s = {0,1,2,3}`. For `k = gjk_iterations .. max_iterations−1`:
```
dist[0] = signedDistance(n0, s2, s1, s3); dist[1] = sD(n1, s0, s2, s3)
dist[2] = sD(n2, s1, s0, s3);             dist[3] = sD(n3, s0, s1, s2)
  signedDistance(n, v1,v2,v3): n = (v3−v1) × (v2−v1); n2=|n|²;
      if 1e-30 < n2 < 1e20: n /= √n2; return dot(n, v1)   else return MAX_LIMIT
if any dist == 0 exactly: gjk_iterations = k; return −1
i = dist0 < dist1 ? 0 : 1; j = dist2 < dist3 ? 2 : 3; index = dist_i < dist_j ? i : j
if dist[index] > 0: copy simplex (in s order) back to obj suppvert/suppind, nsupp=4; return 1
replace simplex[s[index]] = gjkIntersectSupport(normals[index])      // unnormalized? (n is unit here)
if dot(normals[index], new.vert) < 0: nsupp=0; return 0              // separating plane found
swap s[(index+1)&3] and s[(index+2)&3]                                 // keep orientation
```
After `kmax`: return −1 (inconclusive; GJK then continues in distance mode with its own copy of the 4-simplex).

### 5.6 `mjc_gjk(obj1, obj2, x1, x2, dist_cutoff, tolerance, max_iterations)` (continuous collision)
Zero status, copy x1,x2 as warm start, run `gjk`, copy back witness points, return `dist[0]` (0 if penetrating within GJK).

---

## 6. EPA (engine_collision_gjk.c)

### 6.1 Polytope construction from GJK simplex (`nsupp` = 2, 3, 4)

`insertVertex(pt, obj1, obj2, i)` copies `suppvert[i]`/`suppind[i]` into `verts[nverts++]`, `vert = vert1 − vert2`.

`attachFace(pt, v1,v2,v3, adj1,adj2,adj3)` → returns `dist2`:
```
face = faces[nfaces++]; face.verts = v1 + (v2<<10) + (v3<<20); face.adj = (adj1,adj2,adj3)
if projectOriginPlane(face.v, V[v3], V[v2], V[v1]) fails: return 0        // NOTE argument order v3,v2,v1
if dot(face.v, V[v1] − center) < 0: face.v = −face.v                         // point outward
face.dist2 = |face.v|²; face.index = −1; return dist2
```

**polytope2 (segment)**:
```
v1i, v2i = insert 0,1; center = 0.5(v1+v2); diff = v2 − v1
index = argmin_i |diff_i| (strict <, init MAX_LIMIT); e = unit axis index
d1 = e × diff; R = rotmat(diff) (120° about diff); d2 = R d1; d3 = R d2      (mju_mulMatVec3)
v3i = epaSupport(d1, |d1|); v4i = epaSupport(d2,|d2|); v5i = epaSupport(d3,|d3|)
faces (hexahedron), each: if attachFace(...) < mjMINDIST2 → replaceSimplex3(that triangle) → return polytope3
  F0 (v1,v3,v4 | 1,3,2), F1 (v1,v5,v3 | 2,4,0), F2 (v1,v4,v5 | 0,5,1),
  F3 (v2,v4,v3 | 5,0,4), F4 (v2,v3,v5 | 3,1,5), F5 (v2,v5,v4 | 4,2,3)
if !rayTriangle(v1, v2, v3, v4, v5): return mjEPA_P2_NONCONVEX
map[i] = faces+i, faces[i].index = i (i<6); nmap = 6; return 0
```
`rotmat(axis)`: Rodrigues with `sin=0.86602540378`, `cos=−0.5`, `u = axis/|axis|`:
`R = [[c+u1²(1−c), u1u2(1−c)−u3 s, u1u3(1−c)+u2 s], [u2u1(1−c)+u3 s, c+u2²(1−c), u2u3(1−c)−u1 s], [u1u3(1−c)−u2 s, u2u3(1−c)+u1 s, c+u3²(1−c)]]`.
`rayTriangle(v1,v2,v3,v4,v5)`: `d12=v2−v1, d1k=vk−v1`; `vol1=det3(d13,d14,d12)`, `vol2=det3(d14,d15,d12)`, `vol3=det3(d15,d13,d12)`; all ≥0 → 1, all ≤0 → −1, else 0.
`replaceSimplex3(pt, obj1,obj2, a,b,c)`: writes the 3 vertices into `obj->suppvert/suppind`, `nsupp=3`, resets `nfaces=nverts=nmap=0`.

**polytope3 (triangle)**:
```
insert 0,1,2; center = (v1+v2+v3)·(1/3) (add, add, scale)
n = (v2−v1) × (v3−v1); n_norm = |n|; if n_norm < mjMINVAL: return P3_BAD_NORMAL
v5i = epaSupport(−n, n_norm)      // inserted FIRST (index 3)
v4i = epaSupport(+n, n_norm)      // index 4
if triPointIntersect(v1,v2,v3,v4): return P3_INVALID_V4
if triPointIntersect(v1,v2,v3,v5): return P3_INVALID_V5
if dot(n, v4−v1) < mjMINEPATOL·n_norm || dot(−n, v5−v1) < mjMINEPATOL·n_norm: return P3_ORIGIN_ON_FACE
if status.dist[0] > 10·mjMINVAL && !testTetra(v1,v2,v3,v4) && !testTetra(v1,v2,v3,v5): return P3_MISSING_ORIGIN
faces (each < mjMINDIST3 → P3_ORIGIN_ON_FACE):
  F0 (v4,v1,v2 | 1,3,2), F1 (v4,v3,v1 | 2,4,0), F2 (v4,v2,v3 | 0,5,1),
  F3 (v5,v2,v1 | 5,0,4), F4 (v5,v1,v3 | 3,1,5), F5 (v5,v3,v2 | 4,2,3)
map all 6; return 0
```
`triAffineCoord(λ, v1,v2,v3, p)`: same minors/axis selection as S2D but cofactors computed with `p` directly (not projected): `C31 = p_x v2_y + p_y v3_x + v2_x v3_y − p_x v3_y − p_y v2_x − v3_x v2_y` (with the 2 selected coordinates), etc.; `λ_i = C3i/M_max` (no sign test).
`triPointIntersect`: λ from triAffineCoord; any λ<0 → 0; else `‖Σλ_i v_i − p‖ < mjMINVAL`.
`sameSide(p0,p1,p2,p3)`: `n=(p1−p0)×(p2−p0)`; `dot1 = n·(p3−p0)`, `dot2 = n·(−p0)`; same strict sign → 1.
`testTetra(p0..p3) = sameSide(p0,p1,p2,p3) && sameSide(p1,p2,p3,p0) && sameSide(p2,p3,p0,p1) && sameSide(p3,p0,p1,p2)`.

**polytope4 (tetrahedron)**:
```
insert 0..3; center = 0.25·(v1+v2+v3+v4)
F0 (v1,v2,v3 | 1,3,2), F1 (v1,v4,v2 | 2,3,0), F2 (v1,v3,v4 | 0,3,1), F3 (v4,v3,v2 | 2,0,1)
  each: if attachFace < mjMINDIST4 → replaceSimplex3(that face) → polytope3
if !testTetra(v1..v4): return P4_MISSING_ORIGIN
map 4 faces; return 0
```

### 6.2 Horizon construction
- `deleteFace(pt, face)`: if `face.index >= 0`: swap-remove from map (`map[idx] = map[--nmap]; map[idx]->index = idx`); `face.index = −2`.
- `getEdge(face, vertex)`: 0 if verts[0]==vertex, 1 if verts[1]==vertex, else 2.
- `addEdge(pt, faceIndex, edge)`: always `i = nedges++`; if `i < maxedges` store and return 1, else return 0.
- `horizonRec(pt, face, e)`: visible if `dot(face.v, w) − face.dist2 > mjMINVAL`. If visible: delete; for k in {1,2}: `i=(e+k)%3`; adj = faces[face.adj[i]]; if `adj.index > −2`: `adjEdge = getEdge(adj, verts[(i+1)%3])`; `r = horizonRec(adj, adjEdge)`; if `r<0` or (`r==0` and addEdge fails) → ret=−1. Return ret (1) or 0 if not visible.
- `horizon(pt, face)`: delete face; for i in 0..2 same procedure over all three edges. Returns 0 or −1 (overflow).
The DFS order yields the horizon edges as an ordered closed loop.

### 6.3 `epa(status, pt, obj1, obj2)` main loop
```
upper = upper2 = MAX_LIMIT; face = pface = NULL
discrete = discreteGeom(obj1) && discreteGeom(obj2)
tolerance = discrete ? mjMINEPATOL : status.tolerance
kmax = min(max_iterations, 1000)
for k in 0..kmax−1:
    pface = face
    lower2 = MAX_LIMIT; scan map[0..nmap): if map[i].dist2 < lower2 (strict): face = map[i]
    if lower2 > upper2 || !face: face = pface; break
    if lower2 <= 0: warning "EPA: origin lies on affine hull of face"; break
    lower = √lower2
    wi = epaSupport(face.v, lower); w = verts[wi]
    upper_k = dot(face.v, w.vert) / lower
    if upper_k < upper: upper = upper_k; upper2 = upper²
    if upper − lower < tolerance:
        if k == 0 && upper < lower − 1e-10: face = NULL; epa_status = NOCONTACT
        break
    if discrete:   // repeated support vertex => converged
        if ∃ i < nverts−1: w.index1 == verts[i].index1 && w.index2 == verts[i].index2: break
    horizon.w = w.vert
    if horizon(face) < 0: warning "...set ccd_iterations to at least %d" (nedges+11); face = NULL; break
    if horizon.nedges < 3: face = NULL; break
    nfaces0 = nfaces; nedges = horizon.nedges
    if nedges > maxfaces − nfaces: warning out of memory for faces; break     // keeps face
    for i in 0..nedges−1:
        hznFace = faces[horizon.indices[i]]; e = horizon.edges[i]
        v1 = hznFace.verts[e]; v2 = hznFace.verts[(e+1)%3]
        hznFace.adj[e] = nfaces0 + i
        prev = (i==0) ? nfaces0+nedges−1 : nfaces0+i−1 ; next = nfaces0 + (i+1)%nedges
        dist2 = attachFace(wi, v2, v1, prev, horizon.indices[i], next)
        if dist2 == 0: face = NULL; break
        if lower2 <= dist2 <= upper2: append to map
    horizon.nedges = 0
    if !nmap || !face: break
epa_iterations = k
if face: dist[0] = epaWitness(face, x1, x2); nx=1; obj1/2.suppvert/suppind = face's 3 vertices, nsupp=3
else: nx = 0; dist[0] = 0
return face
```
**Only faces whose distance lies in `[lower2, upper2]` enter the candidate map**; others remain in the face array (for adjacency) but are never selected.

`epaWitness(pt, face, x1, x2)`: `λ = triAffineCoord(V1,V2,V3, face.v)`; `x1 = lincomb(λ, vert1's)`, `x2 = lincomb(λ, vert2's)`; return `−√dist2`.

### 6.4 Memory limits
`maxfaces = 6N`, verts capacity `5+N` (N = ccd_iterations), `maxedges = 24 + max(0, N−35)`. Default N=35 → 210 faces, 40 verts, 24 horizon edges.

---

## 7. Top-level `mjc_ccd(config, status, obj1, obj2)`

```
status.x1 = obj1.center(); status.x2 = obj2.center()
gjk_iterations = epa_iterations = 0; epa_status = NOCONTACT
tolerance, max_iterations copied; max_contacts = min(config.max_contacts, 50); dist_cutoff copied

// --- phase A: shrink to core shape when sphere/capsule involved or any geom margin>0
if type1∈{SPHERE,CAPSULE} || type2∈{SPHERE,CAPSULE} || (geom1>=0 && geom2>=0 && (margin1>0 || margin2>0)):
    save supports, margins
    full_margin_i = 0
    if obj_i.geom >= 0: full_margin_i = 0.5·margin_i; obj_i.margin = 0
    if SPHERE:  full_margin_i += size0; support = mjc_pointSupport
    if CAPSULE: full_margin_i += size0; support = mjc_lineSupport
    status.dist_cutoff += full_margin1 + full_margin2
    gjk(status, obj1, obj2)
    status.dist_cutoff = config.dist_cutoff
    restore margins and supports
    if status.dist[0] > status.tolerance:                     // shallow: answer from core distance
        inflate(nx, x1, x2, dist, full_margin1, full_margin2, normal = NULL)
        if dist[0] > dist_cutoff: dist[0] = MAX_LIMIT
        elif config.max_contacts > 1 && dist[0] < 0 && config.buffer && obj1.nsupp > 0:
            nx = mjc_multicontact(obj1, obj2, x1, x2, dist, max_contacts, nmeshdegmax, npolygonmax, buffer)
            return min_i dist[i]
        return dist[0]
    if !config.max_contacts: nx = 0; dist[0] = 0; return 0
    gjk_iterations = 0; status.x1/x2 = centers again          // deep: restart with full shapes
// --- phase B: full (margin-inflated) shapes
gjk(status, obj1, obj2)
if !config.max_contacts: return dist[0]
if dist[0] <= config.tolerance && obj1.nsupp > 1 && config.buffer && !separated:
    gjk_dist = dist[0]; dist[0] = 0
    carve Polytope from buffer (verts, faces, map, horizon.indices, horizon.edges; align8 each)
    ret = nsupp==2 ? polytope2 : nsupp==3 ? polytope3 : polytope4
    epa_status = ret
    if ret == 0:
        face = epa(...)
        if config.max_contacts > 1 && face: nx = mjc_multicontact(...)
        if !face && config.dist_cutoff > 0 && epa_status == NOCONTACT: nx = 1; dist[0] = gjk_dist
return min_{i<nx} dist[i]    (dist[0] if nx ≤ 1)
```
Important consequences:
- In the shrunk GJK, `get_dist` inside `gjk()` uses the *incremented* cutoff: with `config.dist_cutoff=0` but margins/radii present, cutoff = full margins > 0, so GJK runs in distance mode (no `gjkIntersect` backup, `min_norm = mjMINVAL`) and early-outs only when `lower ≥ cutoff·x_norm`.
- `inflate(nx, x1, x2, dist, m1, m2, normal)`: no-op if both margins zero; `n = normal ? normal : normalize(x2 − x1)` (computed once from the first pair); for each i: `x1_i += m1·n` (if m1≠0), `x2_i −= m2·n` (if m2≠0), `dist_i −= (m1+m2)`.
- If a polytope builder fails (ret≠0) with `dist[0]` already set to 0 and `nx` from GJK = 1: the reported distance is 0 → `mjc_penetration` sees `0 < 0` false → **no contact** (origin "on boundary" treated as touching).
- EPA failure (face NULL) → `nx=0, dist=0` → no contact.
- Sphere/capsule pairs that are deeply penetrating (core shapes intersect) fall into phase B with **full** sphere/capsule supports (not point/line).

---

## 8. MultiCCD: `mjc_multicontact(obj1, obj2, x1, x2, dist, nconmax, nmeshdegmax, npolygonmax, buffer)`

Called after shallow GJK (phase A, with core-shape simplex in suppvert) or after EPA (with the closest face). Recovers the contact features (vertex/edge/face) on each geom from the support vertex indices, finds an aligned face pair or edge–face pair, then clips polygons.

### 8.1 Preconditions and capacity adjustments
```
if (type1 != PLANE && nsupp1 < 1) || (type2 != PLANE && nsupp2 < 1): return 1
if MESH && mesh_polynum == 0 (either): return 1
PLANE present: nmeshdegmax ≥ 1, npolygonmax ≥ 1 ; CYLINDER: ≥2, ≥16 ; BOX: ≥3, ≥4 ; CAPSULE: ≥2, ≥2
```
(PLANE objects reach here only from callers outside `mjc_Convex`, e.g. continuous collision; `mjc_PlaneConvex` does not use it.)

### 8.2 Feature extraction
For each non-plane obj: `triface_i = suppind[0], suppind[nsupp>1?1:0], suppind[nsupp>2?2:0]` with matching `suppvert`; `nface = simplexDim(triface_i, triface)`:
```
simplexDim(vi, v): if vi0 != vi1: return (vi2 == vi0 || vi2 == vi1) ? 2 : 3
                   if vi0 != vi2: vi1 = vi2; v[1] = v[2]; return 2
                   return 1
```
(dimension = number of distinct vertex indices: 1 vertex, 2 edge, 3 face). For non-discrete supports (ellipsoid, sphere) the index is −1 throughout → dim 1, but they never get normals so multicontact returns 1. Plane objects: `nface = 3` placeholder.

Buffer layout (align8 each): `idx1[D], idx2[D] (int), n1[3D], n2[3D], endverts[3D], face1[3P], face2[3P], polygon (rest: polygon[6P], clipped[6P], pn[3P], pd[P])`.

Direction: plane1 → `dir_neg = plane1 z`, `dir = −dir_neg`; plane2 → `dir = plane2 z`, `dir_neg = −dir`; else `dir = x2 − x1`, `dir_neg = x1 − x2` (unnormalized).

### 8.3 Face normal candidates
- geom1: BOX `boxNormals(n1, idx1, nface1, obj1, triface1i, dir_neg)`, MESH `meshNormals`, CYLINDER `cylinderNormals`, PLANE `planeNormals`. geom2 same with `dir`.
- `boxNormals(res, resind, dim, obj, vi, dir)` (corner bitmask vi):
  - dim 3: `x = [all three have bit0] − [none has bit0]` (∈{−1,0,1}), same for y (bit1), z (bit2); `res = mat·(x,y,z)`; `resind` lists 0/2/4 for nonzero axes, `+1` on resind[0] if `x+y+z == −1`; if exactly one nonzero (c==1) return 1, else `boxNormals2(res, resind, mat, dir)`.
  - dim 2: per axis with both bits set/unset: append `mat·axis·sign`, resind 0/1, 2/3, 4/5 (note z is written at `res+3`, which is the correct slot whenever two axes are set); return 2 if c==2 else `boxNormals2`.
  - dim 1: three faces at the corner: `mat·(±1,0,0)` etc. with indices; return 3.
- `boxNormals2(res, resind, mat, n)`: `local_n = normalize(matᵀ n)` (via `1/√dot`); first face of list `+x,−x,+y,−y,+z,−z` with `dot > mjFACE_TOL` → 1 normal; else 0.
- `meshNormals(res, resind, dim, obj, vi)`:
  - dim 3: `edgeset = intersect(polymap[v1], polymap[v2])` (≤2), then `faceset = intersect(edgeset, polymap[v3])`; any empty → 0; result 1 normal `mat·polynormal[faceset[0]]`.
  - dim 2: up to 2 polygons common to v1, v2.
  - dim 1: all `mpolymapnum[v1]` incident polygons.
  - `intersect(res, a, b, n, m)`: nested loops (i over a, j over b), append matches, stop at 2.
- `cylinderNormals(dim==1 only)`: `res = mat·(0,0, vi0 ? −1 : 1)`, `resind = vi0` (cap id); else 0. (A cylinder support's `vertindex` is the cap id, so dim 1 means both GJK/EPA supports on the same cap.)
- `planeNormals`: plane z-axis, index 0.

### 8.4 Face alignment / edge fallback
```
face_tol = (any PLANE) ? 0 : mjFACE_TOL
alignedFaces(res, n1, nn1, n2, nn2, dir, face_tol):
    best = −face_tol; dir_tol = mjEDGE_TOL·|dir|
    for i: if dot(n1_i, dir) >= −dir_tol: continue          // geom1 normals must face along −dir
      for j: if dot(n2_j, dir) <= dir_tol: continue          // geom2 normals must face along +dir
        t = dot(n1_i, n2_j); if t < best: best = t; res = (i,j); found = 1
if !found:
    if nface1 < 3 && nn2 && (nface1 <= nface2 || !nn1):      // edge(geom1)–face(geom2)
        nn1 = edge normals of geom1 (box/mesh/cylinder/capsule)
        if !alignedFaceEdge(res, n1, nn1, n2, nn2, dir): return 1 ; edgecon1 = 1
    elif nface2 < 3:                                           // face(geom1)–edge(geom2)
        nn2 = edge normals of geom2
        if !alignedFaceEdge(res, n2, nn2, n1, nn1, dir_neg): return 1 ; edgecon2 = 1
    else return 1
alignedFaceEdge(res, edge, ne, face, nf, dir):
    for i < nf: if dot(face_i, dir) <= mjMINVAL: continue
      for j < ne: if |dot(edge_j, face_i)| < mjEDGE_TOL: res = (j, i); return 1
```
Edge-direction generators (write unit edge directions into `res`, far endpoints into `endverts`, and overwrite the start vertex `v` = triface[0]):
- `boxEdgeNormals`: `v1` = corner (±size by bits) in world; dim 2: single edge to corner v2i; dim 1: three edges to corners with x, y, z negated.
- `meshEdgeNormals`: dim 2: single edge v1→v2; dim 1: for each incident polygon, the edge from v1 to the **previous** vertex in that polygon's loop (`k = j==0 ? nvert−1 : j−1`), normalized.
- `cylinderEdgeNormals(dim 1 or 2)`: `sgn = vi ? 1 : −1`; edge dir = `sgn·axis_z`; if `margin>0 && geom>=0`: project the (margin-inflated) vertex radially onto the rim: local xy of `v−pos`, `scl = n2 ≥ 1e-30 ? size0/√n2 : 0`, `v = mat·(scl·lx, scl·ly, −sgn·size1) + pos`; `endvert = v + dir·2·size1`.
- `capsuleEdgeNormals(res, end, obj, v, w=x_i)`: `res = axis_z`; `pr = dot(w − pos, axis)`; `v = w + axis·(−pr − L)`; `end = w + axis·(−pr + L)` — a segment parallel to the capsule axis through the (already radius/margin-inflated) witness point.

### 8.5 Feature polygon recovery
- geom1: if edgecon1: `face1 = [triface1[0], endverts[i]]`, `nface1=2`; else face index `ind = edgecon2 ? idx1[j] : idx1[i]` → `boxFace`, `meshFace`, `cylinderFace`, or `planeFace` (point = plane pos, nface=0).
- geom2: if edgecon2: `face2 = [triface2[0], endverts[i]]`; else `idx2[j]`.
- `boxFace(idx)` vertex loops (local, then `mat·v+pos`):
  - 0 (+x): (s0,s1,s2),(s0,s1,−s2),(s0,−s1,−s2),(s0,−s1,s2)
  - 1 (−x): (−s0,s1,−s2),(−s0,s1,s2),(−s0,−s1,s2),(−s0,−s1,−s2)
  - 2 (+y): (−s0,s1,−s2),(s0,s1,−s2),(s0,s1,s2),(−s0,s1,s2)
  - 3 (−y): (−s0,−s1,s2),(s0,−s1,s2),(s0,−s1,−s2),(−s0,−s1,−s2)
  - 4 (+z): (−s0,s1,s2),(s0,s1,s2),(s0,−s1,s2),(−s0,−s1,s2)
  - 5 (−z): (s0,s1,−s2),(−s0,s1,−s2),(−s0,−s1,−s2),(s0,−s1,−s2)
  (all clockwise when viewed from outside, i.e. about the outward normal).
- `meshFace(idx)`: polygon vertices in **reverse** order of `mesh_polyvert` (i from nvert−1 down to 0), transformed.
- `cylinderFace(idx)`: 16-gon, `sgn = idx ? −1 : 1`, vertex i: local `(cos16[i]·r, −sin16[i]·r·sgn, sgn·size1)`, with tables `cos16/sin16` of `i·π/8` given to 15 decimals in source (e.g. 0.923879532511287, 0.707106781186548/0.707106781186547 alternating as written, 0.382683432365090).

### 8.6 Clipping dispatch and margins
```
margin_i = (geom_i >= 0 && type_i != CAPSULE) ? 0.5·obj_i.margin : 0      // capsule already inflated via x
margin = margin1 + margin2
if edgecon1 || type2 == PLANE:      // clip geom1's edge/face against geom2's face
    wit_dir = −n2[j]; normal = wit_dir
    nx = polygonClip(x2, x1, dist, nconmax, face2, nface2, face1, nface1, n2[j], wit_dir, buf, P, margin)
elif edgecon2 || type1 == PLANE:
    ind = edgecon2 ? j : i; wit_dir = −n1[ind]; normal = n1[ind]
    nx = polygonClip(x1, x2, ..., face1, nface1, face2, nface2, n1[ind], wit_dir, ...)
else:                                // face–face
    wit_dir = n2[j]; normal = −n2[j]
    nx = polygonClip(x1, x2, ..., face1, nface1, face2, nface2, n1[i], wit_dir, ...)
if nx > 0: inflate(nx, x1, x2, dist, margin1, margin2, normal); return nx
return 1                              // keep original single contact
```

### 8.7 `polygonClip(x1, x2, dist, nconmax, face1, nface1, face2, nface2, n, dir, buffer, P, margin)`
Sutherland–Hodgman clip of `face2` (subject) by the side planes of `face1` (clipper, lying in plane with normal `n`).
```
if nface1 != 0 && nface1 < 3: return 0
polygon = buffer; clipped = polygon + 6P; pn = clipped + 6P; pd = pn + 3P
for each clipper edge e=(face1[k], face1[k+1 mod]):  pd[k] = planeNormal(pn[k], v1, v2, n)
    planeNormal: v3 = v1 + n; res = (v2−v1) × (v3−v1); mju_normalize3(res); return dot(res, v1)
polygon = face2 copy; npolygon = nface2
for e in clipper edges (in order):
    for i < npolygon: P = polygon[i]; Q = polygon[i+1] (wrap to 0)
        in1 = halfspace(face1[e], pn[e], P); in2 = halfspace(face1[e], pn[e], Q)
            halfspace(a, n, p) = dot(p − a, n) > −mjMINVAL
        if !in1 && !in2: continue
        if in1 && in2: append Q; continue
        tmp = dot(pn[e], Q−P); if tmp != 0: t = (pd[e] − dot(pn[e], P))/tmp; if 0 ≤ t ≤ 1: append P + t(Q−P)
        if in2: append Q
    swap(polygon, clipped); npolygon = nclipped; nclipped = 0
prune: keep vertices with dot(v − face1[0], n) <= margin   (in order, compacting)
if npolygon < 1: return 0
if nconmax < 5 && npolygon > 4:
    hull4(idx, polygon, npolygon); for i<4: dist_i = witnessOnFace(x1_i, x2_i, polygon[idx_i], face1, n, dir); return 4
if nface2 == 2 && npolygon > 2:     // edge subject: keep the two farthest points
    (best1,best2) = argmax_{i<j} |p_j − p_i|² (strict >, init (0,1), d=0); write 2; return 2
npolygon = min(npolygon, nconmax); write each in order; return npolygon
```
For a plane clipper (`nface1 == 0`) no clipping occurs; only the margin prune applies. With nface1 vertices the "inside" half-space is the one where `pn` points (the polygons are CW about `n`, making `pn` point inward).

`witnessOnFace(w1, w2, v, p, n, dir)`: `w1 = v − dot(v−p, n)·n` (projection of subject vertex onto clipper plane); `dist = dot(w1 − v, dir)`; `w2 = w1 − dist·dir`; return dist.

`hull4(res, hull, nhull)` (multi-anchor max-area quad): `a,b,c,d = 0,1,2,3`, `m = area4(0,1,2,3)`; for `a` in 0..nhull−1: repeat {try `d+1` (strictly greater area accepted, else break), then advance `c` greedily, then `b` greedily} ; after the inner loop, if `b == a`: `b++` (and cascade to c, d if they collide). `area4 = 0.5·|(h_a − h_c)×(h_b − h_d)|`. Result is the last recorded improving (a,b,c,d).

---

## 9. Algorithm-selection summary (C)

| Condition | Path |
|---|---|
| `mjDSBL_NATIVECCD` set | libccd MPR (1 contact) + `mjc_fixNormal` + (if MULTICCD enabled) 4 perturbation re-runs → ≤5 contacts |
| native, MULTICCD disabled, not box–box | GJK/EPA, 1 contact |
| native, box–box through `mjc_Convex` (not normally: table uses `mjc_BoxBox`) | GJK/EPA + MultiCCD up to 8 |
| native, face–face (box/mesh/cylinder pairs) | GJK/EPA + MultiCCD ≤4 |
| native, capsule with face/edge geom | ≤2 |
| native, sphere/ellipsoid involved | 1 |
| hfield–X | per-prism `mjc_penetration(nconmax=1)`, ≤50 |
| plane–mesh/ellipsoid/SDF | support point + anchored face quad ≤4 (independent of CCD flags) |

`ccd_tolerance` is used as: GJK duality-gap ε = ½tol² (non-discrete), GJK `min_norm` (contact-only, non-discrete), EPA termination `upper − lower < tol` (non-discrete), shallow/deep split `dist > tol`, EPA trigger `dist ≤ tol`, MPR tolerance (libccd path), Hausdorff. `ccd_iterations` caps GJK iterations (shared counter with gjkIntersect), EPA iterations (min with 1000), MPR findPenetr iterations, buffer sizes, Hausdorff iterations.

---

## 10. GPU (MuJoCo Warp) implementation

### 10.1 Dispatch (collision_driver.py, collision_convex.py)
`MJ_COLLISION_TABLE` marks as CONVEX (GJK/EPA): HFIELD–{SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX, MESH}, SPHERE–{ELLIPSOID, MESH}, CAPSULE–{ELLIPSOID, CYLINDER, MESH}, ELLIPSOID–{ELLIPSOID, CYLINDER, BOX, MESH}, CYLINDER–{CYLINDER, BOX, MESH}, **BOX–BOX** (switched to PRIMITIVE only if `NATIVECCD` disabled), BOX–MESH, MESH–MESH. PLANE–MESH/ELLIPSOID are PRIMITIVE (`collision_primitive.plane_convex`). There is **no libccd/MPR path** in Warp: `DisableBit.NATIVECCD` only changes box–box routing. Flex–mesh/ellipsoid also use `ccd` (collision_flex.py).

`convex_narrowphase(m, d, ctx, convex_pairs)`:
- Pair counts from `m.geom_pair_type_count[upper_trid_index(len(GeomType), t1, t2)]`; returns early if no convex pairs.
- `epa_iterations = 16` if **all** convex collisions are box–box, else `opt.ccd_iterations`; `gjk_iterations = opt.ccd_iterations` (compile-time constants baked into kernels).
- `use_multiccd = !(disableflags & MULTICCD)`. Box–box multicontact is always on.
- Buffer sizing: `npolygonmax, nmeshdegmax = (4,3)` if box–box present else `(0,0)`; if multiccd and any of cyl–cyl/cyl–box/cyl–mesh/cap–cyl pairs: `(max(m.npolygonmax,16), max(m.nmeshdegmax,3))`; elif multiccd and any mesh–mesh/box–mesh/cap–mesh: `(max(m.npolygonmax, 4 if boxmesh else current), max(m.nmeshdegmax,3))`.
- Scratch arrays (allocated each call with `wp.empty`, shape `[naccdmax, ...]`): `epa_vert/epa_vert_index [10 + 2·E]` (vertex pairs packed: index `2i` = geom1 point, `2i+1` = geom2 point), `epa_face/epa_pr/epa_norm2 [6 + MJ_MAX_EPAFACES·E]` with `MJ_MAX_EPAFACES = 5`, `epa_horizon [MJ_MAX_EPAHORIZON = 24]`; multiccd: `polygon, clipped [2P]`, `pnormal, pdist [P]`, `idx1, idx2, n1, n2, endvert [D]`, `face1, face2 [P]`.
- `nccd`: int counter per pair type (`[ntype·(ntype+1)/2]`); each CCD that needs scratch memory does `ccdid = atomic_add(nccd[geomgeomid], 1)`; if `ccdid >= naccdmax` → overflow flag `OverflowType.CCD` (optional printf) and the pair produces **no contact**. Pair types run as sequential launches so slots are reused across types.
- One kernel specialization per (type1, type2) via `@cache_kernel` builders; hfield kernels launched first (`dim = naconmax`, one thread per candidate), then non-hfield kernels with grid-stride loop: `block_dim = m.block_dim.convex_ccd` (default 64), `launch_bounds=(block_dim, 8)`, grid = `min(naconmax, 2·block_size·min_grid_size)` from `wp.get_suggested_block_size` (CPU: naconmax). Each thread iterates `collisionid in range(tid, min(ncollision, naconmax), grid)`, skipping candidates whose geom types don't match the specialization.
- Precision: float32 (Warp `float`); constants imported from MuJoCo's single-precision branch (§10.3). Per-world parameters indexed `worldid % array.shape[0]` (batched model fields).
- Contacts written by `write_contact` with an atomic `nacon` counter → contact order across pairs/worlds is nondeterministic (no sorting implemented: `_sort_contacts` is a stub). `write_contact` keeps a contact iff `dist < margin + gap` (detected); `active = dist < margin || (adhesion && dist < margin+gap)`.

### 10.2 Non-hfield kernel `eval_ccd_write_contact`
```
geom1.margin = geom2.margin = margin           // NOT margin+gap (contrast C)
tolerance = opt_ccd_tolerance[world]
is_collision_sensor = pairid[1] >= 0 ; cutoff = is_sensor ? 1e32 : gap
origin = geom1.pos; geom1.pos = 0; geom2.pos −= origin          // translation-invariance shift (C has none)
needs_epa, dist, ncollision, w1, w2, gjk_result, geom1, geom2 = gjk_phase(tol, cutoff, gjk_iters, ..., geom1.pos, geom2.pos)
if needs_epa: ccdid = atomic slot (overflow → return); dist, ncollision, w1, w2, multiccd_idx = epa_phase(...)
if dist >= gap && !is_sensor: return
dist += margin                                   // back to true surface distance
if (box–box) or (use_multiccd and face/edge pair types):
    if !needs_epa && dist < margin && gjk_result.dim > 0:          // shallow penetration
        ccdid = atomic slot; multiccd_idx = _gjk_simplex_to_epa(gjk_result, epa arrays)   // writes 3 vertex pairs + face 0|1<<10|2<<20
    if mesh geom has mesh_polyadr < 0: multiccd_idx = −1
    if multiccd_idx > −1: ncollision, witness1, witness2, dists = multicontact(...)
        if ncollision < 1: ncollision = 1; witness = (w1, w2)
frame = make_frame(dist <= margin ? witness1[0] − witness2[0] : witness2[0] − witness1[0])   // ONE frame for all contacts
for i < ncollision: write_contact(dist_i = (ncollision > 1 ? dists[i] + margin : dist),
                                  pos = 0.5(witness1[i] + witness2[i]) + origin, frame, ...)
```
The initial GJK guesses are the geom positions after the shift (x1 = 0, x2 = geom2.pos − origin), which matches C's `mjc_center` for geoms. For hfield prisms both backends start from the prism centroid.

### 10.3 Warp constants (collision_gjk.py)
`FLOAT_MIN = −1e30`, `FLOAT_MAX = 1e30` (stand-ins for ∓MAX_LIMIT), `MINVAL = 1e-15`, `MINVAL2 = 1e-30`, `MAXVAL = 1e15`, `MAXVAL2 = 1e30` (C: mjMAXVAL2 = 1e20), `MIN_DIST2 = MIN_DIST3 = 1e-10`, `MIN_DIST4 = 1e-17`, `MIN_EPATOL = 1e-7`, `FACE_TOL = cos(0.0889) ≈ 0.996051`, `EDGE_TOL = sin(0.0889) ≈ 0.088783`, `INTERSECT_TOL = 3e-7`. Face status bits: deleted `0x80000000`, invalid `0x40000000`.

### 10.4 `support(geom, geomtype, dir)` → `SupportPoint{point, cached_index, vertex_index}`
- SPHERE: `pos + (size0 + 0.5·margin)·dir` (no rotation), indices −1.
- BOX: `tmp = wp.sign(l)` (sign(0) = +1), `point = rot·(tmp∘size) + pos`, `vertex_index = (tmp0>0) + 2(tmp1>0) + 4(tmp2>0)`.
- CAPSULE: `r·l`, `z += sign(l_z)·L` (sign(0)=+1), rotate.
- ELLIPSOID: `normalize(l∘size)∘size` (wp.normalize of zero → zero, C falls back to `x̂·size0`).
- CYLINDER: as C with `d > MINVAL` (C: `n2 >= MINVAL²`), `vertex_index = l_z ≥ 0 ? 0 : 1`.
- MESH: if `graphadr == −1 || vertnum < 10`: exhaustive; warm start from `geom.index` (absolute index `vertadr+i`) if > −1; strict `>`; `cached_index` absolute, `vertex_index = cached − vertadr`. Else hill-climb starting at `geom.index` (graph-local) or **0** — **no extrema 3×3×3 seeding** (C uses `mesh_extrema`); neighbor scan uses `wp.where` updates (same steepest-in-list semantics). `cached_index = imax` (local), `vertex_index = globalid[imax]`.
- HFIELD: point = argmax over **all 6** prism vertices (`geom.polyvert`, hfield frame, strict `>`; C restricts to bottom/top triangle by `dir_z` sign → ties at `dir_z=0` differ: C picks top, Warp bottom); `vertex_index = dir_z < 0 ? −2 : −3`.
- TRIANGLE / FLEX: vertex argmax (flex uses 4 vertices).
- Finally `if margin > 0: point += 0.5·margin·dir` (applies to all types including hfield/flex; C applies only for `geom>=0` objects, and flex adds its own radius+margin).
In GJK each iteration writes `geom.index = sp.cached_index` (warm start), so non-mesh types reset it to −1.

### 10.5 `gjk(tolerance, gjk_iterations, geom1, geom2, x1_0, x2_0, type1, type2, cutoff, is_discrete)`
Same structure as C §5.3 with these differences:
- `is_discrete` is passed in: `_discrete_geoms(type1,type2)` (both ∈ {MESH, BOX, HFIELD}) **and** both margins 0. Shrunk spheres/capsules (point/line) are **not** discrete in Warp (C treats `mjc_pointSupport/mjc_lineSupport` with zero margin as discrete → C uses ε=0, Warp ε=½tol²).
- `min_norm = is_discrete ? MINVAL : tolerance` (no `get_dist` term).
- Distance mode keyed on `cutoff == 0.0` (contacts only, early-out when `lower > 0`) vs `cutoff < FLOAT_MAX` (early-out when `lower > 0 && lower ≥ cutoff·xnorm`).
- **No `gjkIntersect` fallback**.
- Direction tuning `_gjk_support` when `is_discrete && x_norm < 1e-4`: n==2 → remove the edge component from `dir_neg` (`proj = dot(dir,edge)/|edge|²`, renormalize if > MINVAL); n==3 → `dir_neg = sign(dot(dir, normal))·normal/|normal|`. The tests then use `x_dir = x_norm·dir_neg` instead of `x_k` (duality gap `dot(x_dir, x_dir − s)` and `lower = dot(x_dir, s)`).
- If `n<1` it just breaks (C returns separated).
- Witness: `n==0` → initial points.
- Final separation check only `if xnorm > 0`: `dot(x_k, sp1 − sp2) > 0`.
- Returns `GJKResult{separated, dist, x1, x2, dim=n, simplex/simplex1/simplex2 (mat43), simplex_index1/2, index1/2}`.

### 10.6 `gjk_phase` (Warp equivalent of `mjc_ccd` phase A)
```
has_non_hfield_margin = _supports_multiccd(types) && (margin1 > 0 || margin2 > 0)
   (_supports_multiccd: face–face, capsule–face, face–capsule; NOT capsule–capsule)
SPHERE/CAPSULE i: full_margin_i = size0 + 0.5·margin_i; margin_i = 0; size_i[0] = 0     // radius zeroed (point/line via size)
elif has_non_hfield_margin: full_margin_i = 0.5·margin_i; margin_i = 0
if full_margin1 + full_margin2 > 0:
    is_discrete_shrunk = _discrete_geoms(types) && both margins 0
    cutoff += full margins ; result = gjk(...)
    if result.dist > tolerance:
        if result.dist == FLOAT_MAX: restore; return (False, FLOAT_MAX, 1, ...)
        dist, x1, x2 = _inflate(result, ..., full_margin1, full_margin2); restore; return (False, dist, 1, x1, x2, result)
    restore; cutoff −= full margins
is_discrete = _discrete_geoms(types) && both margins 0
result = gjk(..., cutoff, is_discrete)
if result.dist > tolerance || result.dim < 2 || result.separated: return (False, result.dist, 1, x1, x2, empty)
return (True, result.dist, 1, x1, x2, result)
```
Differences from C: C shrinks *all* margins whenever any margin > 0 for any geom pair (e.g. ellipsoid–box with margin); Warp shrinks the margin only for face/edge pair types (ellipsoid keeps its inflated support). Also the shrunk-GJK "distance > cutoff → MAX" test is replaced by the caller's `dist >= gap` check.

`_inflate`: if geom1 is HFIELD and the simplex vertex indices differ (`is_side`: simplex touches both top (−3) and bottom (−2) / multiple prism features): `x2 = support(geom2, x2 as direction) − margin2·ẑ`; project onto the prism top triangle `(polyvert[3], [4], [5])` via `_tri_affine_coord`; if all coordinates > 0 `x1 = Σλ·top`, else `x1 = x2 − dot(x2 − p, ẑ)ẑ` with `p` chosen as `a` if λ0>0, else `b` if λ1>0, else `c`; `dist = −|x1 − x2|`. Otherwise standard inflate along `normalize(x2 − x1)` (`dist −= m1+m2` always; points moved only if margin > 0).

### 10.7 EPA (`epa_phase`, `_polytope2/3/4`, `_epa`)
Polytope storage: vertices as pairs (`vert[2i]` geom1, `vert[2i+1]` geom2), Minkowski point recomputed as difference on demand; faces as packed ints with deleted/invalid bits; `face_pr` (projection), `face_norm2`; no adjacency, no map.
- `_attach_face(pt, idx, v1,v2,v3)`: returns 0 if `nface == capacity` (out of memory → treated like degenerate) or projection fails; same `projectOriginPlane(p3,p2,p1)` + outward flip vs `center`.
- `_polytope2`: same construction (vertices stored at indices 0..4: two simplex points then three supports with `d/|d|` normalized directly). On face dist² < `MIN_DIST2=1e-10` (C: 1e-30), falls back to polytope3 on that triangle (`status=−1`, `dim=3`). Non-convex → status 1 (failure).
- `_polytope3`: `n` normalized first; supports `−n` (index 3) then `+n` (index 4) — same ordering as C. Checks: `norm < MINVAL` → 2; triPointIntersect v4/v5 → 3/4; **missing the C `P3_ORIGIN_ON_FACE` coplanarity test (`mjMINEPATOL·n_norm`)**; origin-containment test only if `dist > 1e-5` (C: `> 1e-14`); faces with dist² < `MIN_DIST3 = 1e-10` → failure codes 6..11.
- `_polytope4`: faces F0 (0,1,2), F1 (0,3,1), F2 (0,2,3), F3 (3,2,1); dist² < `MIN_DIST4 = 1e-17` → fallback to polytope3; if `!testTetra`: if the min face dist² > MINVAL → failure (12), **else fall back to polytope3 on the closest face** (C fails with P4_MISSING_ORIGIN).
- Any failure status → `epa_phase` returns `(result.dist, 1, x1, x2, −1)` i.e. the GJK distance (≈ ≤ tol) — **one contact with the GJK witness points** (C: no contact).
- `_epa` loop (`epa_iterations = min(E, 1000)`):
  - select face: linear scan of **all** faces in creation order, skipping deleted/invalid, strict `<` (C scans the candidate map whose order changes with swap-removals → tie-breaking differs).
  - same `lower2 > upper2 || idx<0` → previous face; `lower2 <= 0` → break (no warning).
  - support along `face_pr/lower`; `upper_k = dot(face_pr, w)/lower`; break if `upper − lower < ε` (`ε = is_discrete ? 1e-7 : tol`). **No first-iteration NOCONTACT test** (`upper < lower − 1e-10` at k=0).
  - discrete repeated-vertex test on `vert_index` pairs (Warp's indices are vertex indices of the support; for mesh `vertex_index` is the global mesh id).
  - horizon: delete selected face, add its 3 edges; then **scan all non-deleted faces** and delete every one with `dot(face_pr, w) − norm2 > 1e-10` (C: recursive flood from the selected face with threshold `mjMINVAL`; Warp may also delete visible faces not connected to the selected one). Edges collected with `_add_edge(e1,e2)`: undirected key `(min<<10)|max`; if already present it is removed (swap with last) — so shared edges cancel and only boundary edges remain; overflow at 24 edges → `OverflowType.EPA_HORIZON`, EPA aborts (no contact).
  - new faces `(wi, edge[0], edge[1])` for each horizon edge, in horizon array order; orientation fixed by the outward test against `center` (C relies on adjacency winding + the same outward test); faces with dist² outside `[lower2, upper2]` are marked invalid; `nvalid` tracks candidates; dist² == 0 → abort (no contact).
  - stop when `nvalid == 0`.
- `_epa_witness`: barycentric (`_tri_affine_coord`) on face → `x2` first; hfield special case: if geom1 is HFIELD and the face's geom1 vertex indices differ, recompute `x2` from a support of geom2 (sphere/capsule: support of the zero-radius core along `x2`, then `x2 −= (0.5·margin + r)·ẑ`; others: support along `normalize(x2)`) and project onto the prism top triangle as in `_inflate`; `dist = −|x1 − x2|`. Otherwise `dist = −√norm2`.
- `epa_phase` result: EPA failure (`idx == −1`) → `(FLOAT_MAX, 0 contacts)`; `multiccd idx` is the face index if `_supports_multiccd` else −1.

### 10.8 `multicontact` (Warp)
Same feature logic as C §8 with these differences:
- No PLANE handling (Warp plane–convex is a separate primitive). Mesh arrays taken from geom1 if mesh else geom2 (shared model arrays).
- Features from the EPA face's packed vertex pairs (`_feature_dim`, identical to `simplexDim`).
- `dir = x2 − x1` always.
- `_box_normals` dim 2: returns directly if `c == 1 or c == 2` (C: only `c == 2`; for c==1 — the edge is a face diagonal — C calls `boxNormals2`).
- `_cylinder_normals` same; `_cylinder_edge_normals` projects onto the rim if `margin > 0` (C additionally requires `geom >= 0`; uses `safe_div`).
- `_polygon_clip`: requires `nface1 ≥ 3`; plane normals **not normalized** (`_plane_normal` returns `(dot(res,v1), res)` with `res = (v2−v1)×n`); `_halfspace` threshold `> −1e-10` (scale-dependent since pn unnormalized; C `−1e-15` on unit normals); intersection `t` via `_plane_intersect` (returns FLOAT_MAX if `|dot(pn, Q−P)| < 1e-10`), accepted if `−3e-7 < t < 1 + 3e-7` then **clamped** to [0,1] (C: exact `0 ≤ t ≤ 1`, `tmp != 0`). Clip buffers sized `2P`.
- Output capped at 4 (`mat43`): order of reductions is (1) edge subject (`nface2 == 2 && npolygon > 2`) → 2 farthest points; (2) `npolygon > 4` → `_polygon_quad` (same algorithm as C `hull4`); (3) all (≤4). C applies the 4-point hull first only when `nconmax < 5`, and otherwise truncates to `nconmax` (box–box via CCD in C could return up to 8).
- Margins: `margin_i = type_i != CAPSULE ? 0.5·margin_i : 0`; after clipping, witnesses moved by `±margin_i·normal` and `d −= margin`.
- Returns `(n, witness1[4], witness2[4], dists[4])`; `n=1` returns the EPA witness unchanged.

### 10.9 Height-field kernel `ccd_hfield_kernel`
- `_hfield_filter`: `margin = geom_margin[g1] + geom_margin[g2]` (**ignores pair margin/gap and `mj_assignMargin` override**; C uses the driver's `margin+gap`); box–sphere test per axis with `r2 + margin` (z: `size2 < pz − r2 − margin` up, `−size3 > pz + r2 + margin` down); geom2 in hfield frame (`pos = mat1ᵀ(pos2−pos1)`, `rot = mat1ᵀ mat2`, margin 0) and six axis supports; box–box test with margin applied inline; returns **un-expanded** `xmin..zmax` (C expands the bounds by margin before computing the subgrid and the prism height test).
- After filter: atomic CCD slot always (overflow → no contacts).
- Subgrid: `x_scale = 0.5·(ncol−1)/size0`; `cmin = max(0, floor((xmin+size0)·x_scale))`, `cmax = min(ncol−1, ceil(...))`, same for rows (different FP ordering than C's `/(2 size0)·(ncol−1)`).
- geom1 (hfield) identity pose, margin 0; **geom2.margin = margin**; prism top z = `data·size2 + margin` (**top raised by margin**); bottom `−size3`. Same prism vertex walk (`dr=(1,0)`, two init vertices then 2 triangles per cell), same height test vs `zmin`.
- Per prism: `ccd(tol, cutoff = 0.0, ...)` with `x1 = prism center (mean of 6)`, `x2 = geom2.pos`. Since hfield pairs aren't `_supports_multiccd`, only spheres/capsules are shrunk; box/mesh/ellipsoid/cylinder keep `0.5·margin` inflation. Contact kept if `ncontact != 0 && dist < margin + gap`; **no `+margin` correction** is applied to `dist` here (unlike the non-hfield kernel), so with nonzero margin the reported distance is relative to the inflated shapes (prism +margin on top, geom2 +½margin) — a reimplementation targeting Warp parity must replicate this; C reports true distance.
- Normal: `make_frame(w1 − w2)` row 0 in hfield frame → world; pos = midpoint → world.
- Up to `MJ_MAXCONPAIR` (50) prism contacts cached in registers (`vec_maxconpair`, `mat_maxconpair`); overflow at 50 → `OverflowType.HFIELD` (C silently stops at 50).
- **Contact reduction to ≤4** (C emits all ≤50): contact 0 = min-dist; contact 1 = farthest from contact 0; contact 2 = max `|dot(p − p0, n0 × (p0 − p1))|`; contact 3 = max `|dot(p − p0, n0 × (p0 − p2))| + |dot(p1 − p, n0 × (p1 − p2))|`; each stage stops if no candidate or `0 < score < 1e-3` (`MIN_DIST_TO_NEXT_CONTACT`). Each contact uses its own prism normal.
- Per-prism `ccd` gets geoms by value: no warm start of `geom2.index` across prisms (C keeps obj2 indices across prisms).

### 10.10 Warp plane–convex (collision_primitive.plane_convex) — counterpart of `mjc_PlaneConvex`
Support vertex by exhaustive search (`graphadr == −1 || vertnum < 10`) or hill-climb from local vertex 0 (no extrema seed, no warm start); first contact `dist0 = dot(n, v0 − plane_pos)`, pos `v0 − 0.5·dist0·n`; skip face if `mesh_polynum <= 0 || dist0 > margin`; incident polygon by min `dot(polynormal, local_normal)` (init 1.0); anchor search; `_hull4` = C `hull4f`; extra vertices filtered only by the mesh-center test (`dot(n, p − pos) > 0`), margin filtering deferred to `write_contact` (`margin+gap`). Plane–ellipsoid in Warp is a separate primitive (not support-based).

### 10.11 Matching Warp from a C-like reimplementation (checklist)
1. float32 arithmetic everywhere, constants of §10.3.
2. Box–box through GJK/EPA + MultiCCD (unless NATIVECCD disabled), EPA iterations 16 when the model's only convex pairs are box–box.
3. Contact distance: non-hfield `dist_ccd + margin` with geoms inflated by `margin/2` each and cutoff `gap`; hfield as §10.9.
4. One frame per multi-contact pair from the first witness pair; per-contact `dists[i] + margin`.
5. GJK origin shifted to geom1 position.
6. Polytope failure → 1 contact with GJK witnesses; EPA failure → none.
7. No contact sorting; contact order is scheduling-dependent.

### 10.12 MJX (JAX) — different approach
MJX (`mjx/_src/collision_convex.py`) does **not** use GJK/EPA. Colliders are fixed-shape, vmapped SAT/clipping routines on precomputed convex data (`m._impl.mesh_convex`: vertices, faces padded to equal size — at most `_MAX_HULL_FACE_VERTICES = 20` vertices per face, subsampled beyond — face normals, unique edges with their two adjacent face normals; boxes converted to such meshes via `mesh.box`):
- `plane_convex`: all vertices' support depth; choose ≤4 points by `_manifold_points` (deepest-in-skin `a`, farthest `b`, farthest from line ab `c`, farthest from triangle edges `d`) among vertices within a 1e-3 skin of the deepest; duplicates get dist=1 (inactive).
- `sphere_convex`: face SAT on sphere center, closest point on best face polygon (with edge projection), 1 contact.
- `capsule_convex`: face and edge axes, ≤2 contacts.
- `box_box`, `convex_convex` (box–mesh, mesh–mesh): SAT over all face normals of both hulls plus Gauss-map-pruned edge–edge axes (`_arcs_intersect`, Gregorius), reference/incident face clipping (`_clip`, `_create_contact_manifold`) producing 4 contacts; edge contact uses closest segment points; fewer-vertex hull treated first.
- Height fields: fixed static `subgrid_size` around the object's rbound footprint; each grid cell yields 2 prisms (`mesh.hfield_prism`) collided with the object via the above routines; results reduced to 4 by `_manifold_points` within a 1e-3 skin; normals not pointing along the hfield up-axis (`dot < 1e-6`) replaced by the prism top-face normal.
- Ellipsoid/cylinder pairs use SDF-based colliders (`collision_sdf`), not convex.
- No margin-inflated supports, no ccd_tolerance/ccd_iterations use; results differ from C beyond tolerance in general.

---

## 11. Reproduction pitfalls / parity notes

1. **`margin` meaning at the call site.** C collision functions receive `margin + gap` (driver), objects are initialized with it, each shape inflated by half, and `dist += margin+gap` at output. Warp's non-hfield kernel inflates by `margin` and uses `cutoff=gap`; the hfield kernel does neither correction.
2. **Shrink-then-inflate (phase A).** Any sphere/capsule, or any geom margin > 0 (C), triggers a first GJK on core shapes (point/segment, zero margin) with cutoff increased by the radii/half-margins. Only if that core distance ≤ tolerance is GJK rerun on full shapes followed by EPA. Getting this wrong changes witness points (contact position = midpoint) and normals.
3. **Normal = `normalize(x1 − x2)`** of the EPA/GJK witnesses, not the EPA face normal. With `mju_normalize3` fallback `(1,0,0)` for degenerate witnesses. Contact pos is the midpoint of the witnesses.
4. **Per-contact normals in C MultiCCD** are recomputed from each witness pair (`normalize(x1_i − x2_i)` in `mjc_penetration`). `witnessOnFace` gives `x1_i − x2_i = dist_i·wit_dir`, which `inflate` then shifts along `normal`. If one clipped point has a positive inflated distance (it lies in the margin band) while the pair's minimum distance is negative, that point's normal can come out **flipped** relative to the penetrating ones. Warp uses a single frame, taken from the first witness pair, for every contact of the pair.
5. **Exact-zero tests**: GJK compaction uses `lambda != 0`; `sameSign2` treats 0 as "different sign"; `gjkIntersect` returns inconclusive on `dist == 0` exactly; `projectOriginPlane` tests `nn == 0` and `nv != 0`. Reproduce bit-exact operation order (cofactor term ordering given above) to match tie behavior.
6. **Tie-breaking** everywhere is "first strict improvement": support searches (mesh exhaustive warm-starts from cached vertex: ties keep the cached one; hill-climb seed vs cached prefers cached on ties), EPA face selection order (map order after swap-removals in C), `alignedFaces` (strictly less), `alignedFaceEdge` (first match), `hull4`/`hull4f` (strict `>` area).
7. **Warm-start state** (`vertindex`, `meshindex`) persists across: GJK iterations, the second GJK of phase B, EPA, multicontact feature lookup, mjc_Convex perturbations, and hfield prisms (obj2). Discrete repeated-vertex EPA termination and MultiCCD feature recovery rely on `vertindex` reflecting the vertex returned by the **last** support call of that object.
8. **Hfield prisms have no vertex indices in C** (`vertindex = −1`), so EPA's discrete repeated-support check compares only geom2 indices; Warp tags −2/−3 (bottom/top).
9. **Hill-climb seeding** (C native): 3×3×3 extrema table with 0.4 threshold on the *local* direction; libccd path and Warp start from vertex 0 / cached vertex. Results agree on the max vertex unless ties (coplanar faces aligned with dir), but `vertindex` differences change MultiCCD features.
10. **libccd vs native capsule/box/cylinder support at zero direction components**: `mju_sign(0)=0` in `mjccd_support` vs `>=0 → +` in native supports.
11. **Polytope failure semantics**: C treats failure of polytope2/3/4 (origin on boundary) as **no contact** (dist forced to 0, and `mjc_penetration` requires dist < 0). Warp instead returns one contact carrying the GJK distance (0 ≤ dist ≤ tolerance) and the GJK witnesses. The kernel then keeps it only if `dist < gap`, and reports `dist + margin`. So with `gap > tolerance`, Warp emits a touching contact where C emits nothing.
12. **EPA first-iteration no-contact** (`upper < lower − 1e-10` at k==0) exists only in C.
13. **Horizon**: C floods from the removed face over adjacency with `> 1e-15`; Warp scans all faces with `> 1e-10` and builds the horizon by edge cancellation. With Warp's 24-edge horizon limit overflow aborts the contact; in C the horizon limit grows with `ccd_iterations` (`24 + max(0,N−35)`) and overflow also returns no contact (with a warning suggesting `nedges+11` iterations).
14. **EPA face capacity**: C 6N faces / Warp 6+5E; C "out of memory for faces" returns the current face (contact kept), Warp `_attach_face` returning 0 aborts (no contact).
15. **MultiCCD margin prune** uses `margin = ½margin1 + ½margin2` excluding capsules (capsule edges are already inflated through x1/x2); `dot(v − face1[0], n) <= margin`.
16. **polygonClip inside test** uses unit plane normals and `> −1e-15` (C) vs unnormalized normals and `> −1e-10` plus t-clamping (Warp) — boundary vertices of exactly aligned faces (very common for stacked boxes) can be included/excluded differently, changing contact counts.
17. **Box–box**: C never uses CCD for box–box in the normal pipeline (`mjc_BoxBox` primitive, ≤8 contacts); Warp uses CCD+MultiCCD (≤4) unless NATIVECCD is disabled.
18. **Hfield contact count**: C up to 50 contacts per pair (subject to driver `filterPreContacts` across groups); Warp ≤4 selected via the min-dist/farthest heuristic.
19. **Hfield bounds**: C expands the support AABB by margin before computing the subgrid and the prism height test; Warp does not expand but raises the prism tops by margin.
20. **`mjc_PlaneConvex` uses the native support function even with NATIVECCD disabled**, and reports only ≤4 contacts from the most anti-aligned polygon incident to the support vertex, with a 4-vertex anchored hull; vertices above the mesh center along the plane normal are rejected.
21. **perturbation multi-contact** (libccd path) copies the initial contact's depth to every additional contact and uses tolerance `1e-3·min(rbound1, rbound2)` for distinctness; perturbation axes come from `mju_makeFrame` of the initial normal (y-axis seed (0,1,0) unless |n_y| ≥ 0.5, then (0,0,1)).
22. **Precision of mesh data**: vertices float32 in both; C promotes to double per component before dot products; extrema precomputed in float.
23. **Iteration counters**: `gjkIntersect` continues the GJK iteration counter `k`; after an inconclusive `gjkIntersect`, GJK resumes at the advanced `k` with fewer remaining iterations.
24. **Center/initial guess**: C GJK starts from `x1 = center1, x2 = center2` (hfield: mean of 6 prism vertices; flex elem: AABB center). Changing the initial guess changes the GJK path and thus EPA's starting simplex and final witnesses within tolerance.

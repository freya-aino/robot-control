# 03 — Collision driver, broadphase, midphase, contact assembly, CCD-bounded continuous collision, IPC

**Sources covered**

C engine (commit 7ba52b5):
- `src/engine/engine_collision_driver.c` / `.h` (full): `mjCOLLISIONFUNC`, `mj_maxContact`, `mj_collision`, `mj_broadphase`, `mj_SAP`, `makeAAMM`, `add_pair`, `mj_collideTree`, `mj_collideOBB`, `mj_collideFlexSAPPairs`, `mj_collideFlexCellPairs`, `mj_narrowphase`, `collisionTask`, `filterPreContacts`, `addPairContacts`, `mj_setContact`, all filters.
- `src/engine/engine_sort.h` (`mjSORT` timsort, `mjPARTIAL_SORT`).
- `src/engine/engine_core_constraint.c`: `mj_contactParam`, `mj_assignRef/Imp/Friction/Margin`, `mj_addContact` (contact-parameter mixing used by the driver).
- `src/engine/engine_util_spatial.c:mju_makeFrame`, `engine_core_util.c:mj_geomSurfaceVelocity`, `mj_isPyramidal`.
- `src/engine/engine_collision_continuous.c` / `.h` (full): speculative rigid CCD (`mjc_geomStepMotion`, `mjc_advanceGeomPair`) and deformable continuous-collision geometry (`mjc_pairGap`, `mjc_candidates`, `mjc_advance`, ...).
- `src/engine/engine_ipc.c` / `.h` (full): `mj_ipc`, the IPC (augmented-Lagrangian, CCD-bounded) contact step for dim-2 flexes.
- Compile-time inputs: `user_objects.cc` (`mjCPair`/`mjCBodyPair` signatures, `mjCBoundingVolumeHierarchy::MakeBVH`), `user_model.cc` (pair/exclude sorting), `engine_setconst.c` (geom_aabb/rbound, body_contype/conaffinity/margin, body BVH refit).

GPU / JAX:
- MuJoCo Warp `mujoco_warp/_src/collision_driver.py` (full), `collision_core.py` (full), `bvh.py` (full; note: rendering-only), relevant parts of `io.py` (pair precomputation, `naconmax`, broadphase auto-selection), `collision_primitive.py` / `collision_convex.py` (dispatch only), `collision_flex.py` (orchestration only), `forward.py` (overflow check, sleep two-pass).
- MJX `mjx/_src/collision_driver.py` (brief).

**Role in the pipeline.** `mj_collision` runs in the position stage (`mj_fwdPosition`, after `mj_fwdKinematics`/`mj_makeM`/`mj_factorM` and the flex/BVH updates of `mj_flex`/`mj_kinematics`), and again once if `mj_wakeCollision` wakes a sleeping tree. It resets `d->ncon`, finds candidate object pairs (broadphase over bodies+flexes → midphase over BVHs → explicit/excluded/filtered pairs), calls the narrowphase collider for each candidate pair (in parallel), optionally reduces the contacts of "groups" to `mjMAXCONPAIR`, and writes `mjContact` records (geometry from the collider, parameters from `mj_contactParam` or `<pair>`) into the start of the `mjData` arena. `mj_makeConstraint` then turns non-excluded contacts into efc rows. The continuous-collision module supplies (a) speculative CCD contacts inside `mj_collision` when `mjENBL_SPECULATIVE` is set and (b) all geometry for the IPC integrator `mj_ipc`, which replaces `mj_discrete` when `integrator=discrete` and `mjENBL_IPC` is set, and which owns dim-2-flex contact (those pairs are then skipped by `mj_collision`).

---

## 0. Constants, flags, data fields used

| Symbol | Value | Where |
|---|---|---|
| `mjMINVAL` | `1e-15` (double build) | denominators, sweep guards |
| `mjMAXVAL` | `1e10` | infinite plane AABB, sentinels |
| `mjMINMU` | `1e-5` | friction floor (`mj_assignFriction`) |
| `mjMAXCONPAIR` | `50` | max contacts for hfield/SDF/flex groups |
| `mjNREF`, `mjNIMP` | 2, 5 | solref / solimp lengths |
| `mjMINIMP`, `mjMAXIMP` | 0.0001, 0.9999 | (continuous `contactDist`) |
| `_mjRUNSIZE` | 32 | timsort run length |
| SAP max pairs for flex self | `min(n(n-1)/2, 1000000)` | `mj_collideFlexSAPPairs` |
| `eigTOL`, `eigEPS` (double) | `4e-15`, `1e-12` | `mju_eig3Tol` floor |
| compiler `mjEPS` | `1e-14` | BVH flat-box inflation, axis choice |

Disable bits (`opt.disableflags`): `CONSTRAINT=1<<0`, `CONTACT=1<<4`, `FILTERPARENT=1<<10`, `MIDPHASE=1<<14`, `NATIVECCD=1<<17`, `MULTICCD=1<<19`.
Enable bits (`opt.enableflags`): `OVERRIDE=1<<0`, `SLEEP=1<<4`, `IPC=1<<6`, `SPECULATIVE=1<<7`.
`mjtFlexSelf`: `NONE=0, NARROW=1, BVH=2, SAP=3, AUTO=4`.

Model fields read: `geom_type, geom_contype, geom_conaffinity, geom_condim, geom_priority, geom_solmix, geom_solref, geom_solimp, geom_friction, geom_adhesion, geom_margin, geom_gap, geom_rbound, geom_aabb (center[3],half[3] in geom frame), geom_bodyid, geom_surfacevel, body_geomadr/num, body_weldid, body_parentid, body_dofnum, body_treeid, body_contype, body_conaffinity, body_margin, body_bvhadr/num, flex_* (contype, conaffinity, condim, priority, solmix, solref, solimp, friction, margin, gap, radius, dim, rigid, interp, selfcollide, bvhadr/num, elemnum/adr, elemlayer, activelayers, cellnum, vertadr/num, vertbodyid), bvh_child, bvh_nodeid, bvh_aabb (static), nbvhstatic, pair_* (geom1, geom2, signature, dim, margin, gap, solref, solreffriction, solimp, friction, adhesion), exclude_signature, nexclude, npair, opt.{timestep, o_margin, o_solref, o_solimp, o_friction, ccd_iterations, ccd_tolerance, sdf_initpoints}, npolygonmax, nmeshdegmax, flg_surfacevel, vis.global.bvactive`.

Data fields: in — `geom_xpos, geom_xmat, xipos, ximat, flexvert_xpos, flexelem_aabb, bvh_aabb_dyn, body_awake, nbody_awake, qvel, cdof, subtree_com`; out — `ncon, contact[] (arena), parena, bvh_active`.

### Compile-time preprocessing that the driver depends on
- **geom_aabb/rbound** (`engine_setconst.c:geomBound`): plane → aamm `(-MAXVAL,-MAXVAL,-MAXVAL, MAXVAL,MAXVAL,0)`, rbound 0 (so plane `geom_aabb` half-size = `(MAXVAL, MAXVAL, MAXVAL/2)`, center `(0,0,-MAXVAL/2)`); hfield: `[-sx,-sy,-sz_base] .. [sx,sy,sz_top]`, rbound `sqrt(sx²+sy²+max(sz²,base²))`; sphere half = r; capsule half `(r, r, r+h)`, rbound r+h; mesh/SDF computed by the mesh compiler.
- **body_contype / body_conaffinity**: bitwise OR over the body's geoms. **body_margin**: `max_g(geom_margin+geom_gap)`.
- **pair_signature** = `(body(geom1) << 16) + body(geom2)` after swapping geoms so `body1 <= body2`. **exclude_signature** = `(body1<<16)+body2` with `body1<=body2`. Both arrays are ordered by signature (`std::stable_sort` on signature → stable among equal signatures, original declaration order kept). The driver requires sorted order (merge-join). Body ids must be < 65536 (16-bit packing; SAP returns -1 when `n >= 0x10000`).
- **Body BVH** (`mjCBoundingVolumeHierarchy::MakeBVH`): built in the body inertial frame (`body_ipos`, `body_iquat`) over geoms with `contype||conaffinity` (visual-only geoms dropped). Each node box = AAMM of the 8 rotated corners of each element's AABB; flat boxes inflated by ±mjEPS. Split: longest axis (axis 1 only if `edge1 >= edge0 + mjEPS`; axis 2 if `edge2 >= edge[axis]+mjEPS`), median by element center `lpos[axis]` via `std::nth_element` with ties broken by element pointer (declaration order), `m = n/2` elements left. Node index assigned in preorder (`nbvh_++`). Leaves have `child=(-1,-1)` and `nodeid = geom id`; internal nodes `nodeid=-1`. `mj_setConst` refits `bvh_aabb` in (center, half) format from children (children-before-parents loop `j=num-1..0`). Flex BVHs live in the dynamic block (`bvh_aabb_dyn`, index `bvhadr - nbvhstatic`), refit every step from `flexelem_aabb` (`engine_core_smooth.c`, leaves copy element AABB, `mj_updateDynamicBVH` merges children).

---

## 1. Collision function table `mjCOLLISIONFUNC[9][9]`

Indexed `[type1][type2]` with `type1 <= type2` (upper triangle; lower triangle is 0). Geom type order: `PLANE=0, HFIELD=1, SPHERE=2, CAPSULE=3, ELLIPSOID=4, CYLINDER=5, BOX=6, MESH=7, SDF=8`.

| t1 \ t2 | SPHERE | CAPSULE | ELLIPSOID | CYLINDER | BOX | MESH | SDF |
|---|---|---|---|---|---|---|---|
| PLANE | PlaneSphere | PlaneCapsule | PlaneConvex | PlaneCylinder | PlaneBox | PlaneConvex | PlaneConvex |
| HFIELD | ConvexHField | ConvexHField | ConvexHField | ConvexHField | ConvexHField | ConvexHField | HFieldSDF |
| SPHERE | SphereSphere | SphereCapsule | Convex | SphereCylinder | SphereBox | Convex | SDF |
| CAPSULE | – | CapsuleCapsule | Convex | Convex | CapsuleBox | Convex | SDF |
| ELLIPSOID | – | – | Convex | Convex | Convex | Convex | SDF |
| CYLINDER | – | – | – | Convex | Convex | Convex | SDF |
| BOX | – | – | – | – | **BoxBox** | Convex | SDF |
| MESH | – | – | – | – | – | Convex | MeshSDF |
| SDF | – | – | – | – | – | – | SDF |

PLANE–PLANE, PLANE–HFIELD, HFIELD–HFIELD are NULL (never collide). `mjc_BoxBox` is a dedicated primitive (SAT + clipping, up to 8 contacts) in this commit regardless of NATIVECCD/MULTICCD. Every collider has signature `int f(const mjModel*, mjData*, mjPreContact* con, int g1, int g2, mjtNum margin)` and writes `mjPreContact{dist, pos[3], normal[3], tangent[3]}`; the normal points from g1 to g2. The `margin` argument passed is **margin+gap** (detection distance).

### `mj_maxContact(m, g1, g2, has_margin)` — upper bound used to size the per-pair precontact buffer (`has_margin` ignored)
Evaluated in this order:
1. either SDF → `opt.sdf_initpoints`
2. either HFIELD → other type not PLANE/HFIELD ? `mjMAXCONPAIR (50)` : 0
3. either SPHERE or ELLIPSOID → 1
4. BOX–BOX → 8
5. CAPSULE–CAPSULE → 2
6. CAPSULE–BOX → 4
7. `hasface = type∈{BOX, MESH, CYLINDER}`, `hasedge = type==CAPSULE`. If either PLANE: any hasface → 4; any hasedge → 2; else 0.
8. MULTICCD disabled → 1
9. NATIVECCD disabled → 5
10. hasface1 && hasface2 → 4; (edge,face)/(face,edge)/(edge,edge) → 2; else 1.

A collider returning more than this is a fatal error (`mjERROR "collision function returned %d contacts, expected at most %d"`).

---

## 2. Filters (all `static` in the driver)

### 2.1 `filterBitmask(ct1, ca1, ct2, ca2)` → 1 = discard
$$\text{discard} = \neg(ct_1 \,\&\, ca_2) \wedge \neg(ct_2 \,\&\, ca_1)$$
`canCollide(bf)` = body: `body_contype||body_conaffinity`; flex: `flex_contype||flex_conaffinity`. `canCollide2(bf1,bf2)` = `!filterBitmask` using body-level (OR-aggregated) or flex masks.

### 2.2 `filterBox(aabb1, aabb2, margin)` (world AABBs in center/half form) → 1 = separated
Separated on any axis k if `c1k + h1k + margin < c2k - h2k` or `c2k + h2k + margin < c1k - h1k`. Strict `<` (touching = overlap).

### 2.3 `filterSphereBox(s, bound, aabb)` → 1 if `s_k + bound < c_k - h_k` or `s_k - bound > c_k + h_k` for any k.

### 2.4 `mj_filterSphere(m, d, g1, g2, margin)` → 1 = discard (margin here = margin+gap)
- If speculative enabled (`ENBL_SPECULATIVE && !DSBL_NATIVECCD`): `min_size = pairMinSize(g1,g2)` (min AABB half-thickness of the non-plane geom(s): `min(geom_aabb[3..5])`), and `s_i = geomStepSweep(g_i, min_size, dx_i)`. Otherwise `s1=s2=0`.
- Both `rbound > 0`: `bound = rb1 + rb2 + margin`, `diff = xpos2 - xpos1`.
  - `s1==s2==0`: discard iff `|diff|² > bound²`.
  - else: `v = dx2 - dx1`, `τ = clip(-diff·v / v·v, 0, 1)` if `v·v > mjMINVAL` else 0; `closest = diff + τ v`; discard iff `|closest|² > bound²`.
- g1 plane, rb2>0: `dist = (xpos2 - xpos1)·n1` (n1 = plane z-axis = column 2 of xmat: `mat[2], mat[5], mat[8]`); if `s2>0`: `dist = min(dist, dist + n1·dx2)`; discard iff `dist > margin + rb2`. Symmetric test if g2 is a plane (both tests run; either can discard).
- Otherwise (e.g., both zero rbound) keep.

### 2.5 `filterBodyPair(weld1, wparent1, asleep1, dof1, weld2, wparent2, asleep2, dof2, dsbl_filterparent)` → 1 = discard
In order:
1. `weld1 == weld2` (same welded rigid body)
2. `dof1 == 0 && dof2 == 0` (`body_dofnum` of the weld bodies; no forces can act)
3. `asleep1 && asleep2`
4. `(asleep1 && weld2 == 0) || (asleep2 && weld1 == 0)` — asleep vs world-static (mocap bodies have weld≠0 and so still trigger wake)
5. `!dsbl_filterparent && weld1 != 0 && weld2 != 0 && (weld1 == wparent2 || weld2 == wparent1)` where `wparent = body_weldid[body_parentid[weld]]`.

`asleep = sleep_filter ? body_awake[b]==mjS_ASLEEP : 0`, with `sleep_filter = ENBL_SLEEP && nbody_awake < nbody`.

### 2.6 `hasAwakeDofs(bf, sleep_filter)`
Body: `body_treeid[bf] >= 0 && (!sleep_filter || body_awake[bf]==mjS_AWAKE)`. Flex: `state = mj_sleepState(mjOBJ_FLEX, f)`; `state != mjS_STATIC && (!sleep_filter || state == mjS_AWAKE)`.

### 2.7 `filterCollisionPair(m, d, g1, g2, ipair, merged, startadr, pairadr)` → 0 = discard, 1 = keep
1. If `merged`: linear scan `k ∈ [startadr, pairadr)` of explicit pairs; if any has `{pair_geom1,pair_geom2} == {g1,g2}` (either order) → discard (the explicit pair supersedes the dynamic one).
2. If `ipair >= 0` and `ENBL_SLEEP`: discard if both geoms' bodies are not `mjS_AWAKE`.
3. If `ipair < 0`: if `mjcb_contactfilter` is set, discard iff callback returns nonzero (callback REPLACES the bitmask test); else discard iff `filterBitmask(geom masks)`. Explicit pairs skip bitmask filtering entirely.
4. `margin = getMargin`, `gap = getGap`; discard iff `mj_filterSphere(g1, g2, margin+gap)`.
5. Keep iff `mjCOLLISIONFUNC[min type][max type] != NULL`.

`getMargin(g1,g2,ipair)` = `mj_assignMargin(ipair>=0 ? pair_margin : geom_margin[g1]+geom_margin[g2])`; `mj_assignMargin` returns `opt.o_margin` under `ENBL_OVERRIDE`. `getGap` = `pair_gap` or `geom_gap[g1]+geom_gap[g2]` (**gap is never overridden**).

### 2.8 `mj_collideOBB(aabb1, aabb2, xpos1, xmat1, xpos2, xmat2, margin, product, offset, initialize)` → 1 = overlap
Separating-axis test on the 6 face normals only (no 9 edge cross-products; conservative).
- `inf_i[k] = aabb_i[3+k] >= mjMAXVAL`. If all three dims infinite for either box → return 1.
- `infinite[i] = any(inf_i)`.
- Centers (only when `product==NULL`): `xcenter_i = xmat_i·aabb_i[0:3] + xpos_i` (missing xmat ⇒ identity; missing xpos ⇒ 0).
- Normals: `normal[i][j][k] = xmat_i[3k+j]` (column j of xmat = body axis j in world), or identity.
- If `product && offset && *initialize`: precompute `product[18i+9j+3k+l] = normal[i][l]·normal[j][k]`, `offset[6i+3j+k] = xpos_i·normal[j][k]` (0 if no xpos); set `*initialize=0`. This cache is valid for all node pairs of one body-pair traversal (poses constant).
- For `j∈{0,1}` (skip if `infinite[1-j]`), `k∈{0,1,2}`: for each box i
  - no cache: `proj_i = xcenter_i·n_jk`, `radius_i = Σ_l |aabb_i[3+l] (n_il·n_jk)|`
  - cache: `proj_i = Σ_l aabb_i[l]·product[adr+l] + offset[6i+3j+k]`, `radius_i = Σ_l |aabb_i[3+l]·product[adr+l]|`, `adr = 18i+9j+3k`.
  - separated (return 0) if `radius_0 + radius_1 + margin < |proj_1 - proj_0|`.
- Return 1.

---

## 3. Broadphase (`mj_broadphase`) — bodyflex pairs

Objects are "bodyflexes": ids `0..nbody-1` are bodies, `nbody..nbody+nflex-1` are flexes. Output: array of `mjPacked32{uint16 hi, lo}` with `hi < lo`, sorted lexicographically, possibly with duplicates. Allocated on the mjData stack with capacity `nbodyflex(nbodyflex-1)/2`; overflow → `mjERROR("broadphase buffer full")`.

### 3.1 Always-colliding pre-pass (planes and world)
For each body `b1` with `canCollide(b1)` that is either (`b1==0 && body_geomnum[0]>0`) or (dof-less weld `body_dofnum[body_weldid[b1]]==0` and `hasPlane(b1)`):
- for each `b2 ∈ [0,nbody)` with `canCollide(b2)`: `filterBodyPair(weld1, pw1, asleep1=0, dof1, weld2, pw2, asleep2, dof2)`; if kept, `add_pair(b1, b2)`.
- for each flex `f` with `hasAwakeDofs(nbody+f)`: `add_pair(b1, nbody+f)`.

Rationale: planes have rbound 0 and a degenerate SAP interval (see 3.3), and the world body is excluded from SAP.

### 3.2 Principal-axis frame
- `cen` = mean of `geom_xpos[i]` over geoms with `geom_bodyid[i] != 0`, plus `flexvert_xpos[i]` over flex vertices with `flex_vertbodyid[i] != 0`. If count 0 → return the pre-pass pairs without the final sort (reachable only when every geom is on the world body, so all pairs are `(0, b2)` with increasing b2 and already sorted; any dof-less non-world plane body would make the count nonzero).
- `cov = (1/cnt) Σ (p - cen)(p - cen)^T` (accumulated with `updateCov`, symmetric fill).
- `mju_eig3Tol(eigval, eigvec, quat, cov, 1e-3)`: cyclic Jacobi with quaternion accumulation, up to 500 iterations, terminates when max |off-diag| ≤ `scale·max(1e-3, eigTOL)` where `scale = max|cov_ij|`; eigenvalues bubble-sorted descending (swap only if differ by > `scale·eigEPS`). `frame = eigvec^T` (rows = eigenvectors). Axis 0 of the frame = largest variance direction. The low tolerance 1e-3 is intentional: "SAP is exact in any frame, a rough one prunes as well". The frame affects only pruning, not results (but does affect pair ORDER before the final sort — irrelevant since pairs are sorted).

### 3.3 AAMM per collidable bodyflex (`makeAAMM`)
Collidable list `bfid` = all `i ∈ [1, nbodyflex)` with `canCollide(i)` (world body 0 never in SAP). If `ncollide > 1`, build AAMMs stored **column-major** (`aamm[axis*ncollide + i]`, axes `xmin,ymin,zmin,xmax,ymax,zmax`) in the principal frame.

Body: for each geom (all geoms, including non-colliding ones):
- `margin = ENBL_OVERRIDE ? 0.5*o_margin : geom_margin + geom_gap`.
- `pos = xmat·aabb_center + xpos`; `axis = xmat^T` (rows = geom axes); `r_half = geom_rbound`.
- speculative: `s = geomStepSweep(g, 0, dx, dw)` (any motion counts), `s_rot = s>0 ? |dw|·r_half : 0`.
- for each frame row `f_j`:
  - `aabb_cen = pos·f_j`, `aabb_half = Σ_l |size_l (axis_l·f_j)| + s_rot`
  - `r_cen = xpos·f_j`
  - `lo = max(r_cen - r_half, aabb_cen - aabb_half)`, `hi = min(r_cen + r_half, aabb_cen + aabb_half)` (intersection of bounding-sphere and OBB projections)
  - if `s>0`: `d = dx·f_j`, `lo += min(0,d)`, `hi += max(0,d)`
  - `lo -= margin`, `hi += margin`
- union over geoms (first geom initializes).

Note: for a plane geom, `r_half = 0` ⇒ interval collapses to `r_cen ± margin` (hence the pre-pass).

Flex: AAMM of all `flexvert_xpos` of the flex projected (`mju_mulMatVec(v, frame, x, 3, 3)`), then expanded by `bound = flex_radius + (OVERRIDE ? 0.5*o_margin : flex_margin + flex_gap)`.

### 3.4 Sweep-and-prune `mj_SAP(d, aamm, n, axis_x, pair, maxpair)`
- Reject (`return -1`) if `n >= 0x10000` or bad axis or `maxpair < 1`.
- `sortbuf[2i] = {value=(float)xmin[i], id_ismax=i}`, `sortbuf[2i+1] = {(float)xmax[i], i + 0x10000}`. **Primary-axis values are cast to single-precision float.**
- Sort with `SAPsort` (stable timsort, §8) using `SAPcmp`: by value; on ties `(a.id_ismax & 0x10000) - (b.id_ismax & 0x10000)` ⇒ min-events before max-events (touching intervals overlap).
- Secondary axes: axis_x=0 → (1,2); 1 → (0,2); 2 → (0,1).
- Sweep `i = 0..2n-1`:
  - min-event: for each active entry `j` in insertion order (`id1 = activebuf[j].id`, `id2 = sortbuf[i].id`): skip if `ymin[id1] > ymax[id2] || ymin[id2] > ymax[id1] || zmin[id1] > zmax[id2] || zmin[id2] > zmax[id1]` (mjtNum compares, strict); else `pair[npair] = {hi=id1, lo=id2}`, `npair++`, and if `npair >= maxpair` **return maxpair (silent truncation)**. Then append to active list.
  - max-event: remove its id from the active list by linear search + `memmove` (preserves order).
- Return npair. Broadphase calls with `axis_x = 0`, `maxpair = ncollide(ncollide-1)/2` (cannot truncate).

### 3.5 Post-SAP filtering and conversion
For each SAP pair `(bf1, bf2) = (bfid[hi], bfid[lo])`:
- both bodies: `filterBodyPair(...)` with real `asleep1/2`; discard on 1.
- otherwise (any flex): discard iff `!hasAwakeDofs(min) && !hasAwakeDofs(max)`.
- `add_pair(m, bf1, bf2)`: recompute contype/conaffinity (body: OR over geoms; flex: flex masks) and discard on bitmask failure; store `hi=min, lo=max`.

Finally sort all pairs with `bfsort` (stable timsort, lexicographic `(hi, lo)`) if `npair > 1`. Duplicates remain (pre-pass ∩ SAP) and are removed by the caller.

---

## 4. `mj_collision` — orchestration

```
mj_collision(m, d):
  d->ncon = 0; resetArena(d)  (parena = ncon*sizeof(mjContact) = 0); mj_clearEfc(d)
  if DSBL_CONSTRAINT || DSBL_CONTACT || nbody+nflex < 2: return
  if vis.global.bvactive: memset(bvh_active, 0, nbvh)
  markStack
  bfpair = stack[nbf*(nbf-1)/2]; nbfpair = mj_broadphase(...)
  parena = alignArena(d, alignof(mjcPair))     # candidate pairs pushed on the ARENA
  pairadr = 0; ncandidate = 0; ngroup = 0; last_signature = (unsigned)-1
  for each (bf1,bf2) in bfpair (sorted):
     signature = (bf1<<16) + bf2
     if signature == last_signature: continue       # dedupe
     last_signature = signature
     # merge-join explicit <pair>s
     merged = 0; startadr = pairadr
     while pairadr < npair and pair_signature[pairadr] <= signature:
        merged = (pair_signature[pairadr] == signature)
        if filterCollisionPair(pair_geom1, pair_geom2, ipair=pairadr, merged=0): push GEOM_GEOM(ipair)
        pairadr++
     if !canCollide2(bf1,bf2): continue
     if both flexes and mjc_ipcOwnsFlexFlex(f1,f2): continue
     # exclude (merge-scan, exadr reset to 0 for every bodyflex pair)
     exadr = 0; while exadr<nexclude and exclude_signature[exadr] < signature: exadr++
     if exadr<nexclude and exclude_signature[exadr]==signature: continue
     if both bodies with exactly 1 geom each:
        if filterCollisionPair(g1,g2,-1,merged,startadr,pairadr): push GEOM_GEOM(-1)
     elif !DSBL_MIDPHASE and bvh1>=0 and bvh2>=0 and
          !(speculative && both bodies && bodyPairHasFastGeom(bf1,bf2)):
        group = -1
        if body:flex and flexReducible(f2): group = ngroup; ngroup += body_geomnum[bf1]
        elif flex:flex and both reducible: group = ngroup++
        ncandidate = mj_collideTree(bf1,bf2,merged,startadr,pairadr,parena,ncandidate,group)
     else:  # all-to-all
        body:body : for g1 in geoms(bf1): for g2 in geoms(bf2): filterCollisionPair → push GEOM_GEOM(-1)
        body:flex : for g in geoms(bf1):
                      skip if filterBitmask(geom g, flex f) or mjc_ipcOwnsFlexGeom(f,g)
                      group = flexReducible(f) ? ngroup++ : -1
                      PLANE: push GEOM_FLEX(g,f,group); continue
                      SDF:   if flex_dim[f]==2: push GEOM_FLEX(g,f,group); continue
                      for e in 0..elemnum-1: if mj_isElemActive(f,e): push GEOM_ELEM(g,f,e,group)
        flex:flex : group = both reducible ? ngroup++ : -1
                    for e1 active in f1: for e2 active in f2: push ELEM_ELEM(f1,e1,f2,e2,group)
  freeStack
  # remaining explicit pairs (signature > every broadphase signature)
  for pairadr..npair-1: if filterCollisionPair(…, pairadr, merged=0): push GEOM_GEOM(pairadr)
  # flex self-collisions
  for f in flexes:
     if flex_rigid[f] or !(flex_contype & flex_conaffinity): continue      # self-mask test uses AND of own masks
     if !hasAwakeDofs(nbody+f): continue
     if mjc_ipcOwnsFlexFlex(f,f): continue
     all_excluded = interp && cellnum[0..2] <= 2
     if selfcollide != NONE and !all_excluded:
        group = flexReducible ? ngroup++ : -1
        if !DSBL_MIDPHASE and selfcollide != NARROW and (flex_bvhadr>=0 or (AUTO and interp)):
           if AUTO and interp:           mj_collideFlexCellPairs
           elif BVH or (AUTO and dim==3): mj_collideTree(nbody+f, nbody+f, …)
           else:                          mj_collideFlexSAPPairs
        else:  for e1<e2 both active: pushSelfElemElem
  if ncandidate > 0: mj_narrowphase(m, d, ncandidate, parena)
```

Key semantics:
- **Explicit pairs bypass broadphase/bitmask/exclude/parent filters**, are only subject to the sleep test, the bounding-sphere test (with pair margin+gap) and the NULL-function test. An explicit pair suppresses the dynamically generated pair for the same geoms in that bodyflex pair (only via `merged` — the scan range `[startadr, pairadr)` covers explicit pairs with signature ≤ current; only those with equal signature can match).
- Explicit pairs are emitted interleaved in signature order with the broadphase pairs; those with a signature that never appears in the broadphase list are still emitted (either while advancing past them, or in the final loop).
- Excluded bodyflex pairs skip all dynamic geom pairs, but not explicit pairs (they were already pushed before the exclude test).
- `pushGeomGeom` swaps so `geom_type[g1] <= geom_type[g2]` (ties keep the given order: for dynamic pairs g1 ∈ lower body id; for explicit pairs `pair_geom1` first, whose body ≤ geom2's body).
- **Filter groups.** `flexReducible(f) = flex_rigid[f] || flex_interp[f]`. A group id is assigned per (body geom, reducible flex) or per (reducible flex, reducible flex) or per reducible self-colliding flex. All candidate pairs in a group are later reduced jointly to ≤ `mjMAXCONPAIR` contacts (§6.4). In `mj_collideTree`, body:flex pairs of geom g get group `group + g - body_geomadr[bf1]` (`geomGroup`).
- `mj_isElemActive(f,e)` = `flex_dim<3 ? 1 : flex_elemlayer[elemadr+e] < flex_activelayers[f]`.
- `pushSelfElemElem` drops pairs with `mjc_ElemSharedNode(f,e1,e2)` (interpolated flexes only: elements whose vertices' interpolation-grid cell ranges are within distance 1 on every axis).

### 4.1 Candidate pair record `mjcPair` (arena)
```c
typedef struct {
  enum {GEOM_GEOM=0, GEOM_FLEX, GEOM_ELEM, ELEM_ELEM} type;
  int conpos;   // set in narrowphase: offset into precontact buffer
  int group;    // -1: none
  union { {g1,g2,ipair,spec}, {g,f}, {g,f,e}, {f1,e1,f2,e2} };
} mjcPair;
```
Pushed one at a time with `mj_arenaAllocByte(sizeof(mjcPair), alignof(mjcPair))`; arena exhaustion → `mjERROR("arena too small to allocate geom pair")` (fatal, unlike contacts). `defaultPair` sets group=-1 and clears the union to -1.

`pairCompare(p1,p2)` (used for sorting tree/cell outputs): if both groups ≥0 and differ → by group; else by type; else by `pairObjects` = (g1,g2) / (g,f) / (g,e) / (e1,e2) lexicographic. Note: GEOM_ELEM compares `(g, e)` — flex id not compared.

---

## 5. Midphase

### 5.1 `mj_collideTree(bf1, bf2, merged, startadr, pairadr, parena, npair, group)`
Simultaneous descent of two BVHs (or one BVH with itself for self-collision).
- Body BVH boxes are static (`m->bvh_aabb`, inertial frame, posed by `xipos/ximat`); flex BVH boxes are dynamic (`d->bvh_aabb_dyn`, world frame, no pose).
- Early return if `!canCollide2`.
- Stack (mjData stack) capacity `max_stack = bvhnum1 + bvhnum2` (`mjERROR "BVH stack depth exceeded"` should not occur). Start with `(0,0)` (roots).
- body:flex pre-pushes (before traversal):
  - if the body's weld has 0 dofs: each PLANE geom of the body (not IPC-owned) → `pushGeomFlex(g, f2, geomGroup(...))`.
  - each SDF geom of the body, if `flex_dim[f2]==2` → `pushGeomFlex`.
- Loop: pop `(node1,node2)`. `isleaf = child[2n] < 0 && child[2n+1] < 0`; `nodeid = bvh_nodeid[adr+node]` (leaf with nodeid<0 → fatal). Self-collision: skip if `node1 > node2`.
  - **body:body**
    - both leaves (geoms g1=nodeid1, g2=nodeid2): `margin = assignMargin(geom_margin sum)`, `gap = geom_gap sum`; if `!mj_filterSphere(margin+gap)` and `mj_collideOBB(geom_aabb[g1], geom_aabb[g2], geom_xpos/xmat each, margin+gap, NULL,…)` then `filterCollisionPair(g1,g2,-1,merged,…)` → push GEOM_GEOM; mark `bvh_active` on overlap (independent of filterCollisionPair). `continue`.
    - else `margin = assignMargin(body_margin[bf1]+body_margin[bf2])`; if `!mj_collideOBB(node boxes, xipos/ximat of each body, margin, product, offset, &initialize)` → continue.
  - **body:flex**
    - both leaves (geom g=nodeid1, element e=nodeid2): `margin = assignMargin(geom_margin+flex_margin)`, `gap = geom_gap+flex_gap`; require `!filterBitmask(geom,flex)`, `!filterSphereBox(geom_xpos, rbound+margin+gap, flex node box)`, and `mj_collideOBB(geom_aabb, node box, geom pose, NULL pose, margin+gap)`. Then if geom type is not PLANE/SDF and not IPC-owned → `pushGeomElem(g, f2, e, group)`. Mark active. continue.
    - else `margin = assignMargin(body_margin+flex_margin)`; OBB test body node (posed) vs flex node (identity) with cache.
  - flex:body → fatal (`bf1 < bf2` always, bodies first).
  - **flex:flex**
    - both leaves: `margin = 0` if self else `assignMargin(flex_margin1+flex_margin2) + flex_gap1 + flex_gap2`; if `!filterBox(flexelem_aabb[e1], flexelem_aabb[e2], margin)`: cross → `pushElemElem`; self and `e1 != e2` → `pushSelfElemElem`. Mark active only if a pair was pushed. continue.
    - else `margin = assignMargin(flex_margin1+flex_margin2)` (no gap at internal nodes) and `filterBox(node boxes, margin)`.
  - mark both nodes active (internal-node overlap).
  - Descend: if only node1 is internal, push its existing children (child index ≠ -1) paired with node2 (child 0 pushed first, so child 1 popped first); symmetric if only node2 internal; if both internal, compare "surface" `x·y + y·z + z·x` using `aabb[3..5] - aabb[0..2]` (**note: this mixes half-size minus center, not true extents — a heuristic quirk to reproduce exactly for identical pair order**) and descend into the larger (`surface1 > surface2` ⇒ node1; ties → node2).
- After traversal: sort the pairs pushed by this call (`pairs + start_pair .. npair`) with `pairSort` (stable), so output order is independent of traversal order.

### 5.2 `mj_collideFlexSAPPairs(f, group, npair)`
- Active element ids `elid`. If `<2`, return.
- AAMM from `flexelem_aabb` (center ± half), column-major.
- Axis = largest half-size of the flex BVH root box (`bvh_aabb_dyn[flex_bvhadr - nbvhstatic]`): `axis = (h0>h1 && h0>h2) ? 0 : (h1>h2 ? 1 : 2)`.
- `maxsappair = min(nactive(nactive-1)/2, 1e6)`; `mj_SAP` (may truncate silently at 1e6).
- push `pushSelfElemElem(f, elid[hi], elid[lo])` in SAP output order (NOT sorted afterwards; SAP order is deterministic).

### 5.3 `mj_collideFlexCellPairs(f, group, parena, npair)` — interpolated flex (AUTO)
- `cellnum (nx,ny,nz)`, return if `nelem<2 || ncell<1`.
- Per vertex: cell range `[lo,hi]` via `mjc_vertCellRange` (vertex reference coordinate `flex_vert0 ∈[0,1]³` times cellnum, floor of `u ± 1e-6·cellnum` clamped; or the vertex's assigned `flex_vertcell` if it disagrees).
- Each active element assigned to cell `c = (lo0·ny + lo1)·nz + lo2` of its **first** vertex; element range = union of its vertices' ranges.
- CSR of non-empty cells (in increasing c), per-cell union range and AABB (min/max of element AABBs, stored center/half).
- For `s1 < s2` over non-empty cells: skip if first-vertex cells are neighbours (`mjc_cellRangeShared(coord1,coord1,coord2,coord2)`: all `|Δ| ≤ 1`); skip if `filterBox(cell1, cell2, 0)`; `disjoint = !cellRangeShared(cell ranges)`; for each element pair with `!filterBox(elem aabbs, 0)`: if not disjoint, skip if any vertex pair shares a cell node (range test); push `pushElemElem(f, min(e1,e2), f, max(e1,e2), group)`.
- Elements within the same cell are never paired (they share a node via their first vertex).
- Sort the generated pairs with `pairSort`.

---

## 6. Narrowphase (`mj_narrowphase`)

### 6.1 Buffers
```
nthread = mju_numThread(d)
npolygonmax = DSBL_MULTICCD ? 0 : m->npolygonmax;  nmeshdegmax likewise
ccd_size = mjc_ccdSize(npolygonmax, nmeshdegmax, opt.ccd_iterations)   # per-thread EPA/multiccd scratch
chunksize = nthread==1 ? npair : max(1, npair/(5*nthread));  nchunk = ceil(npair/chunksize)
markStack
pairbuffer = stack copy of arena pairs; pairbuffer[i].conpos = prefix sum of pairMaxContact
d->parena = parena      # arena region of candidate pairs is reclaimed; contacts overwrite it
nconbuffer[npair] (int), conbuffer[maxcon] (mjPreContact), flexbuffer[maxcon] (int, only if any GEOM_FLEX)
epabuffer = stack[ccd_size * (nchunk<2 ? 1 : nthread)]
mju_dispatch(collisionTask, nchunk)
```
`pairMaxContact`: GEOM_GEOM → `mj_maxContact`; GEOM_FLEX → plane ? `flex_vertnum[f]` : `mjMAXCONPAIR`; GEOM_ELEM → hfield, or dim 2 with box/capsule → `mjMAXCONPAIR`; dim 1 with capsule/box → 2; else 1; ELEM_ELEM → both dim 1 ? 2 : 1.

### 6.2 `collisionTask(thread_id, idx)` (per chunk)
`mjc_setCCDBuffer(epabuffer + thread_id*ccd_size)` (thread-local), then for each pair in the chunk:
- GEOM_GEOM: `f = mjCOLLISIONFUNC[type(g1)][type(g2)]`, `ncon = f(m, d, conbuffer+conpos, g1, g2, margin+gap)`. If speculative enabled: `active` = any returned contact with `dist < margin` (in-gap contacts don't count); if `!active` and either geom sweeps more than `pairMinSize` (`geomStepSweep > 0`): `nccd = mjc_advanceGeomPair(g1, g2, margin, conbuffer+conpos, thread epabuffer)`; if `nccd > 0`: `ncon = nccd`, `pair.spec = 1` (CCD contacts replace the discrete ones).
- GEOM_FLEX: `margin = assignMargin(geom_margin+flex_margin)`, `gap = geom_gap+flex_gap`; PLANE → `mjc_PlaneFlex(…, flexbuffer+conpos, g, f, margin+gap)` (writes vertex ids); SDF → `mjc_FlexSDF` (writes element ids); else 0.
- GEOM_ELEM: same margin/gap; `mjc_GeomElem(g, f, e, margin+gap)`.
- ELEM_ELEM: `margin = assignMargin(flex_margin1+flex_margin2)`, `gap = flex_gap1+flex_gap2`; **self (f1==f2): margin = 0 but gap is kept**; `mjc_ElemElem(f1,e1,f2,e2,margin+gap)`.
- Check `ncon <= next.conpos - conpos` (fatal otherwise).

Threading: chunks are disjoint and each pair writes only its own precontact slice, so results are deterministic regardless of thread count (no atomics); only `ccd` scratch is per thread.

### 6.3 Contact emission (serial, in candidate order)
```
lastgroup = -1
for i in 0..npair-1:
   if pair[i].group >= 0:
      if group <= lastgroup: mjERROR("filter group %d is not contiguous")
      lastgroup = group; filterPreContacts(d, &arg, i, group)     # reduces the whole group, resets their group to -1
   ncon = nconbuffer[i]; if 0: continue
   vert = (GEOM_FLEX && plane) ? flexbuffer+conpos : NULL
   elem = (GEOM_FLEX && SDF)   ? flexbuffer+conpos : NULL
   addPairContacts(precon, ncon, pair, elem, vert)
freeStack
```
Groups must be contiguous and increasing in candidate order — guaranteed because groups are assigned in push order and tree/cell outputs are sorted by group first.

### 6.4 `filterPreContacts(d, arg, start, group)` — greedy farthest-point reduction to `mjMAXCONPAIR`
- Group extent: consecutive pairs from `start` with the same group (their group reset to -1); `count = Σ ncon`. If `count <= 50` return.
- `selected[count]=0`, `min_dist[count]=mjMAXVAL`.
- Seed = deepest precontact: maximize `-dist` with strict `>` (first wins ties) scanning pairs then contacts in order.
- Repeat while `nselected < 50 && best >= 0`: select best; for every unselected precontact compute squared distance to the just-selected position, `min_dist = min(min_dist, d²)`; next best = argmax `min_dist` (strict `>`, initial threshold -1, first wins ties).
- Compact selected precontacts in place within each pair's slice (preserving order) and also the `flexbuffer` entries; update `nconbuffer`.

### 6.5 `addPairContacts(m, d, precon, ncon, pair, elem, vert)`
- Allocate `ncon` contiguous `mjContact` on the arena; if allocation fails → `mj_warning(mjWARN_CONTACTFULL, ncon)` and **drop all contacts of this pair** and return (the arena is linear; a later pair with fewer contacts may still fit). Otherwise `d->ncon += ncon`.
- Parameters per pair type:
  - GEOM_GEOM: `margin = getMargin`; explicit (`ipair>=0`): `condim = pair_dim`, `solref = pair_solref`, `solimp = pair_solimp`, `friction[5] = pair_friction` (already 5-D), `adhesion = pair_adhesion`, `solreffriction = pair_solreffriction` iff either entry nonzero else `{0,0}`; dynamic: `mj_contactParam(g1,g2,-1,-1)`.
  - GEOM_FLEX / GEOM_ELEM: `geom[0]=g`, `flex[1]=f`, `margin = assignMargin(geom_margin+flex_margin)`, `mj_contactParam(g, -1, -1, f)`.
  - ELEM_ELEM: `flex = (f1,f2)`, `elem = (e1,e2)`, `margin = self ? 0 : assignMargin(flex_margin sum)`, `mj_contactParam(-1,-1,f1,f2)`.
- For each precontact: `dist, pos`, `frame[0:3] = normal`, `frame[3:6] = tangent`; `geom = (g1,g2)`, `flex = (f1,f2)`, `elem = (e1, elem ? elem[i] : e2)`, `vert = (v1=-1, vert ? vert[i] : -1)`; then `mj_setContact(...)` with `includemargin = margin` (**margin, not margin+gap**) and `speculative = pair.spec`.

### 6.6 `mj_setContact(con, condim, includemargin, solref, solreffriction, solimp, friction, adhesion, speculative)`
1. `con->dim = condim; includemargin; adhesion`; `mj_assignRef(solref)`, `mj_assignRef(solreffriction)`, `mj_assignImp(solimp)` (copy, or `opt.o_solref/o_solimp` under OVERRIDE — note solreffriction is also replaced by `o_solref` under OVERRIDE); `mj_assignFriction`: `friction[i] = max(mjMINMU, src[i])` or `max(mjMINMU, o_friction[i])` under OVERRIDE.
2. `exclude = (dist >= includemargin) && !adhesion && !speculative` (1 = "in gap": detected within gap but no force).
3. If `adhesion && !speculative && dist >= includemargin`: `dim = 1` (in-gap adhesive contacts are frictionless).
4. `mju_makeFrame(frame)`:
   - normalize x-axis (normal); if its norm < 0.5 → `mjERROR("xaxis of contact frame undefined")`.
   - if `|y|² < 0.25` (tangent not supplied): `y = (0,1,0)` if `-0.5 < x[1] < 0.5` else `(0,0,1)`.
   - `y -= (x·y) x`; normalize y; `z = x × y`.
5. Surface velocity (`m->flg_surfacevel`): for each side with `geom[side] >= 0` and nonzero `geom_surfacevel[6]`: `(vw, ww) = mj_geomSurfaceVelocity(g, con->pos)` (`vw = R·sv[0:3] + (R·sv[3:6]) × (pos - geom_xpos)`, `ww = R·sv[3:6]`), accumulate with sign `-1` for side 0 and `+1` for side 1. If any active: `surfacevel = (frame_y·svel, frame_z·svel, frame_x·sang)`; else zeros. Flex sides ignored.
6. `efc_address = -1; mu = 0; H = 0`; deprecated `geom1 = geom[0], geom2 = geom[1]`.

### 6.7 `mj_contactParam(condim, solref, solimp, friction, adhesion, g1, g2, f1, f2)`
Each side reads geom or flex arrays (`f<0` ⇒ geom). Flex adhesion is 0.
- `adhesion = a1 + a2` (default).
- `priority1 > priority2`: copy side 1's condim, solref, solimp, friction(3), **adhesion = a1**. `<`: side 2.
- Equal priority:
  - `condim = max(condim1, condim2)`
  - `mix = s1/(s1+s2)` if both `solmix >= mjMINVAL`; 0.5 if both `< mjMINVAL`; 0 if only s1 small; 1 if only s2 small.
  - solref: if `solref1[0] > 0 && solref2[0] > 0` ("standard" timeconst/dampratio): `mix·solref1 + (1-mix)·solref2` elementwise; else ("direct" negative stiffness/damping): elementwise `min`.
  - `solimp = mix·solimp1 + (1-mix)·solimp2` (all 5 entries, including width/midpoint/power).
  - `fri = max(friction1, friction2)` elementwise.
  - adhesion stays `a1+a2`.
- 5-D friction: `(fri0, fri0, fri1, fri2, fri2)` = (tangent1, tangent2, torsional, rolling1, rolling2).
- `condim ∉ [1,6]` → fatal.

### 6.8 Contact dimension and cone
`condim ∈ {1,3,4,6}` (validated at compile time). `mj_makeConstraint` (other spec file) adds: dim 1 → 1 frictionless row; pyramidal (`mj_isPyramidal`: `opt.cone == mjCONE_PYRAMIDAL`) → `2(dim-1)` rows (pairs of opposing edges `J_n ± μ_k J_k`); elliptic → `dim` rows. There is no function named `mj_isElliptic` in this commit; ellipticity is tested as `m->opt.cone == mjCONE_ELLIPTIC` (e.g. solver `ctx->is_elliptic`). Contacts with `exclude != 0` produce no rows; `mj_instantiateContact` additionally sets `exclude=3` (no dofs) and `exclude=4` (passive: flex contact handled by the effective metric when `mj_effFlexContactPossible` and the pair is flex-flex or flex vs world-welded geom).

### 6.9 Arena layout and "nconmax"
The contact array is the bottom of the mjData arena: `d->contact = arena`, `parena = ncon·sizeof(mjContact)`. Candidate pairs are pushed above it, copied to the stack, and the region is reclaimed before contacts are written. There is **no fixed nconmax** in the C engine (`m->nconmax` is legacy); the limit is `narena` (model `memory` attribute). Overflow of contacts is a warning (`mjWARN_CONTACTFULL`), overflow of candidate pairs is a fatal error. Under ASAN the free arena region is poisoned. `mj_addContact` (user API) appends one contact after `mj_clearEfc`.

---

## 7. Speculative rigid CCD (`engine_collision_continuous.c`, rigid part)

Enabled iff `ENBL_SPECULATIVE && !DSBL_NATIVECCD` (`speculativeEnabled`).

### 7.1 `mjc_geomStepMotion(m, d, g, t, dx, dw)` → surface sweep bound
- 0 if `g<0` or geom on a world-welded body.
- `cvel = Σ_{b from weld(body) up to root, b>0} Σ_{dofs j of b} cdof_j · qvel_j` (skip `qv==0`). Spatial velocity (ang, lin) at `subtree_com[root]`.
- `gvel = mju_transformSpatial(cvel, flg_force=0, newpos=geom_xpos, oldpos=subtree_com[rootid], rot=NULL)` — moves the reference point to the geom center.
- `dw = t·gvel[0:3]` (zero for spheres — rotation doesn't move a sphere's surface); `dx = t·gvel[3:6]`.
- return `|dx| + |dw|·geom_rbound`.

`geomStepSweep(g, min_size, dx, dw)` (driver): 0 for `g<0`, PLANE, or unsupported types; returns `s` only if `s >= min_size`, else 0 (and dx, dw zeroed).
`bodyPairHasFastGeom(b1,b2)`: `min_size` = min AABB half-thickness over non-plane supported geoms of both bodies (none → 0); true if any geom of either body has `geomStepSweep > 0`. Used to route such body pairs to all-to-all instead of the (non-swept) BVH midphase.

### 7.2 `mjc_advanceGeomPair(g1, g2, margin, con, buffer)` → up to 4 speculative precontacts
Supported types (`mjc_GeomSupported`): PLANE, SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX, MESH; plane–plane rejected.
1. `s_i = geomStepMotion(g_i, timestep, dx_i, dw_i)`; `sweep = s1+s2`; return 0 if `< mjMINVAL`. `dx = dx1 - dx2`; `rot_sweep = (s1-|dx1|) + (s2-|dx2|)`.
2. CCD objects at current poses (`mjc_initCCDObj(…, margin 0)`); spheres use point support, capsules line support (radius handled analytically).
3. `distmax = sweep + margin`. `dist0 = evalGeomPairDist(...)`:
   - plane involved: support point of the other geom along `-n`, shifted by `-r n` (sphere/capsule radius); `dist = n·x - n·plane_pos`; plane witness = `x - dist·n`; normal = `±n` (sign makes it g1→g2). The other object's `nsupp=1`, `suppind[0]=vertindex`, `suppvert = s` (needed for multicontact).
   - else GJK (`mjc_gjk(obj1, obj2, x1, x2, cutoff = distmax + r1 + r2, ccd_tolerance, ccd_iterations)`); if `dist <= cutoff`: normal = normalize(x2-x1) (default (0,0,1)), `x1 += r1 n`, `x2 -= r2 n`. Returns `dist - (r1+r2)`.
4. Return 0 if `dist0 <= margin` (already in contact) or `dist0 - margin > sweep` (cannot reach).
5. `target = margin + 0.02 (dist0 - margin)`; `rel_tol = max(ccd_tolerance, 0.5(target - margin))`; `t = 0`.
6. Directed conservative advancement, ≤16 iterations: `clearance = dist - target`; break if `≤ rel_tol`; `dir_sweep = normal·dx + rot_sweep`; return 0 if `≤ mjMINVAL` (separating); `step = clearance/dir_sweep`; `t += step`; return 0 if `t >= 1`; advance poses: `pos += step·dx_i`, `mat = R(axis=dw/|dw|, angle=|dw|·step)·mat` (skip rotation if `|dw|·step < mjMINVAL`); re-evaluate dist.
7. Witness full-step displacements `dp_i = dx_i + dw_i × (x_i - pos_i)` (at advanced poses); `vn = normal·(dp1 - dp2)`; `closing = max(vn, (dist0 - dist)/max(t, mjMINVAL))`.
8. If the loop hit the cap without converging (`dist - target > rel_tol`): return 0 if `vn ≤ mjMINVAL` or `dist - margin > (1-t) vn`; else `t += (dist - target)/vn` (linear extrapolation). Return 0 if `closing ≤ 0`.
9. With `con`: initial witness set {x1,x2,dist}; if `buffer && (plane involved || MULTICCD enabled)`: `pad = 2·max(0,dist) + (1-t)·closing + ccd_tolerance`, set both obj margins = pad, shift witnesses by `±0.5 pad n`, `nx = mjc_multicontact(obj1, obj2, x1_all, x2_all, dist_all, 4, nmeshdegmax, npolygonmax, buffer)`, unshift.
10. `cdist = contactDist(g1, g2, margin, t, closing)`:
    - solref/solimp from the explicit pair (linear scan of `pair_geom1/2`) or `mj_contactParam`; `mj_assignRef/Imp`; if `(solref0>0) XOR (solref1>0)` → default (0.02, 1).
    - `imp = clip(solimp[1], mjMINIMP, mjMAXIMP)`, `delta = max(1e-4, solimp[2])`.
    - standard: `τ = (!DSBL_REFSAFE && !metric) ? max(2dt, solref0) : max(mjMINVAL, solref0)`, `ζ = max(mjMINVAL, solref1)`, `k0 = dt²/max(mjMINVAL, τ²ζ²)`, `b0 = 2dt/τ`. direct: `k0 = -solref0·dt²`, `b0 = -solref1·dt`.
    - metric (`integrator == DISCRETE`): if `!REFSAFE-disabled && solref0>0 && k0 > imp`: `excess = k0/imp`, `fmax = mjMAXIMP(1-imp)/max(mjMINVAL, imp(1-mjMAXIMP))`, `k0 = imp`, `b0 = min(b0/√excess, imp·max(0, fmax-2))`; if `k0 > mjMINVAL`: `delta = ((1 - t - t(b0+k0))/k0)·closing`.
    - non-metric: if `k0 > mjMINVAL`: `delta = ((1 - t - b0)/k0)·closing`.
    - return `margin - delta`.
11. Weights `w_i = s_i/sweep`; `g_fast` = larger sweep. For each witness pair: pull back to t=0 poses (`x_0 = R_0 R_t^T (x_t - pos_t) + pos_0`); discard if any of three segments is occluded (`isRayOccluded(g1,g2,x1_0,x1_t)`, `(g2,g1,x2_0,x2_t)`, `(g_fast,g_slow,x1_t,x2_t)`); else `pos = w1 x1_0 + w2 x2_0`, `dist = cdist`, `normal`, `tangent = 0`.
12. Without `con`: returns 1 iff the single witness survives occlusion.

`isRayOccluded(g1, g2, a, b)`: degenerate if `|ab|² < mjMINVAL²`. Segment shrunk by `eps = 1e-5` at both ends. For each other supported geom g3 (≠g1,g2): bounding-sphere prefilter against the segment's sphere; skip if same weld as g1, or same moving weld as g2; unless an explicit pair (g1,g3) exists, skip if bitmask fails, body pair excluded, or parent-filtered. Hit tests: plane — sign change of signed distances `da·db ≤ 0` with one `|·| > eps`; mesh — `mj_rayMesh ∈ [0,1]`; others — `mju_rayGeom ∈ [0,1]`. On hit, occluded only if g1 and g3 are NOT within margin at t=0 (`isGeomPairWithinMargin` with `tol = margin3 + ccd_tolerance`).

Speculative contacts keep `exclude = 0` and full condim even though `dist >= includemargin` may hold; `dist = margin - delta` is chosen so the solver's reference acceleration arrests `(1-t)` of the closing velocity this step.

---

## 8. Sorting utility (`engine_sort.h`)

`mjSORT(name, type, cmp)` defines `name(arr, buf, n, ctx)`: **stable** bottom-up merge sort ("tiled timsort"):
1. insertion sort each run of 32 (`_mjRUNSIZE`), shifting while `cmp(arr[k], tmp) > 0` (stable);
2. merge passes with `len = 32, 64, …`: `_mjMERGE` takes from the left run when `cmp(left, right) <= 0` (stable); odd tail copied; ping-pong between `arr` and `buf`; final copy back if needed.

`mjPARTIAL_SORT(name, type, cmp)`: bottom-k selection via a max-heap of size k (`_mjSIFT_DOWN`), replacing root when `cmp(x, root) < 0`, then insertion-sorting the k results (not stable w.r.t. input). Not used by the driver directly.

Uses in the driver: `SAPsort` (mjtSAP, SAPcmp), `bfsort` (mjPacked32, lexicographic), `pairSort` (mjcPair, pairCompare).

---

## 9. Deformable continuous collision geometry (`engine_collision_continuous.c`, flex part)

Consumers: `mj_ipc` (only consumer in this commit). Gaps are measured at the **flex midsurface** (flex radii NOT subtracted); for geom pairs, flex midsurface to geom **surface**. The broadphase adds radii back into its reach (`bandOffset`).

### 9.1 Pair types (`mjcFlexPair{type, idx[4], g}`)
| type | meaning | idx | gradient vertices |
|---|---|---|---|
| 0 `FLEX_VERT_TRI` | flex vertex v vs flex triangle (A,B,C) | v,A,B,C | 4 |
| 1 `FLEX_EDGE_EDGE` | flex edge (a1,b1) vs flex edge (a2,b2) | a1,b1,a2,b2 | 4 |
| 2 `FLEX_VERT_GEOM` | flex vertex vs static geom surface | v | 1 |
| 3 `GEOM_CORNER_TRI` | geom corner feature gv[c] vs flex triangle | c,A,B,C | 3 (A,B,C) |
| 4 `GEOM_EDGE_EDGE` | geom edge ge[c] vs flex edge (a,b) | c,a,b | 2 |
Indices are into the concatenated "free point" array (dim-2 flex vertices concatenated in flex order, offset `fxadr[k]`). `mjc_pairVerts` returns the flex-side indices (types 3/4 skip idx[0]).

### 9.2 Distance kernels
- `mjc_PtTri(p,a,b,c)`: barycentric weights via `mjc_triWeights` (Ericson 5.1.5 Voronoi regions: vertex A if `d1≤0 && d2≤0`; B if `d3≥0 && d4≤d3`; C if `d6≥0 && d5≤d6`; edge AB if `vc≤0 && d1≥0 && d3≤0` with `t=d1/(d1-d3)`; AC if `vb≤0 && d2≥0 && d6≤0`, `t=d2/(d2-d6)`; BC if `va≤0 && (d4-d3)≥0 && (d5-d6)≥0`, `t=(d4-d3)/((d4-d3)+(d5-d6))`; else face `t=vb/(va+vb+vc)`, `u=vc/(…)`), `cp = Σ w_i v_i`, return `|p - cp|`.
- `mjc_SegSeg(p1,p2,q1,q2)`: weights from `mjc_segWeights` (degenerate segments when `|ab|² ≤ mjMINVAL`; parallel cutoff `|ab×cd|² ≤ 1e-8·|ab|²|cd|²`; non-parallel `s = clip((ab×cd)·(cd×ca)/|ab×cd|², 0, 1)`, then `t=(dot5·s+dot3)/dot2` clamped with `s` recomputed as the projection of the clamped endpoint; parallel case uses the first of two candidate pairs), closest points, return distance.
- `mjc_GeomDist(gi, gpos, gmat, x, n, distmax)` signed distance to static geom surface, outward normal n (world):
  - plane: `n = z-axis`, `d = n·(x - pos)`.
  - sphere: `L = |x-pos|`; `L < 1e-12` → `-r`; else `n = dx/L`, `L - r`.
  - capsule (local `p`): `zc = clip(p_z, ±h)`, `q = p - (0,0,zc)`, `L=|q|`, `n = q/L` (or (1,0,0) if `L<1e-12`), `d = L - r`.
  - box: clamp to box; outside → `d = |q|`, `n = q/|q|`; inside → least-penetration face `pen_k = size_k - |p_k|`, `d = -min pen`, `n = ±e_ax`.
  - mesh/cylinder/ellipsoid: mesh with `polynum ≤ 0` → `1e30`. GJK point-vs-geom with cutoff `distmax`; `> distmax` → `1e30`; `> 0` → `n = normalize(x - x2)`, `d = gdist`. Inside: mesh → max over polygons of `n_p·p - n_p·v0` (best face normal); cylinder → `pr = r - ρ`, `pz = h - |p_z|`, radial if `pr < pz` else cap; ellipsoid → `d = (|p/size| - 1)·min(size)`, `n = normalize(p/size²)` (or (0,0,1)).
  - other types → `1e30`.
  - local normal rotated to world.
- `featRadius(g)`: sphere/capsule → `geom_size[0]` (their "corner" features are center/axis endpoints, which stand inside the surface); else 0.
- `mjc_GeomVerts`: sphere center (1); capsule endpoints `pos ∓ R·(0,0,h)` (2); box 8 corners in loop order `sx,sy,sz ∈ {-1,+1}` nested; mesh all vertices; else 0.
- `mjc_GeomEdges`: capsule axis (1); box 12 edges (axis-major, then s1, s2); mesh polygon edges with `a < b` only (dedup); else 0.

### 9.3 `mjc_pairGap(pair, m, d, x, gv, ge, radii, n, idv, cw, nidx, cutoff)`
Returns gap g and gradient `∂g/∂x_{idv[p]} = cw[p]·n`:
- VT: `dd = PtTri(v; A,B,C)`, `n = (x_v - cp)/dd` (`gapNormal`: zero normal if `dd < mjMINVAL`), `cw = (1, -w0, -w1, -w2)`, return `dd`.
- EE: `n = (cp1-cp2)/dd`, `cw = (1-s, s, -(1-t), -t)`, return `dd`.
- VG: `dd = GeomDist(g; x_v, distmax = cutoff + radii[v])`, `cw = 1`, return `dd`.
- CT: `n = (corner - cp)/dd`, `cw = (-w0,-w1,-w2)`, return `dd - featRadius(g)`.
- GE: `n = (cp1-cp2)/dd`, `cw = (-(1-t), -t)`, return `dd - featRadius(g)`.

`mjc_standoff(band, cap) = min(band, cap)`. `mjc_pairBand(pair, radii, band) = min(band, min over flex-side vertices radii[v])`.

### 9.4 Candidate generation `mjc_candidates(...)` (swept broadphase)
Inputs: configuration `x` (query), sweep `dfrom → dto`, base reaches `thresh` (flex-flex) and `threshGeom` (feature-flex), band `ghat`, geom features `gv/ge` with owners `gvgeom/gegeom`, flex lists. Returns 0 under DSBL_CONSTRAINT/CONTACT.
1. Per point travel `dl[v] = |dto - dfrom|`; per flex `fsweep[k] = max dl`, `fstale[k] = max |x - flexvert_xpos|` (distance from the configuration the BVH was refit at).
2. Bounding box `[pglo, pghi]` of points that can meet static geoms (free `fidx>=0` or riding a moving body `pjnv>0`), `pgmarg = max(thresh + dl + radii)`, padded `+= 1e-9|pgmarg| + 1e-9`.
3. Work units in serial order: (a) `candGEOMVERT` over all geoms; then per dim-2 flex k with a BVH and elements: `candCORNER` (all features), `candGEDGE`, `candVERT` (all points), and `candEDGE` for each `kj ≤ k` with `fok[k,kj]`.
4. `fok[k,kj] = !maskFiltered(flex masks)`, and for `kj==k` also `selfcollide != NONE`. `ptbody[v] = flex_vertbodyid` (−1 for interpolated).
5. Unit bodies (`candRun`):
   - GEOMVERT: static (world-welded), supported, colliding geoms only; non-plane geoms culled by world AABB (rotated `geom_aabb`: `wh_k = Σ_l |R_kl| h_l`) vs the padded point box; per eligible point: geom/flex mask, `reach = thresh + dl[v]`, AABB cull with `reach + radii[v]`; `addCand(VERT_GEOM)`.
   - CORNER: `reachG = threshGeom + fsweep_k`, BVH box half `= reachG + r_k + fstale_k + featRadius(g)`; `bvhBox` query of flex k's element BVH; `addCand(CORNER_TRI, reach=reachG)`.
   - GEDGE: query box = edge AABB inflated as above; for each returned element's 3 edges (dedup per query via `stampG[e]==qid`) `addCand(GEOM_EDGE_EDGE)`.
   - VERT: for point v of flex kv with `fok[k,kv]`: `thv = 3·min(ghat, radii[v], r_k) + dl[v] + fsweep_k`, box half `thv + radii[v] + fstale_k`; skip self-adjacent triangles and triangles with a vertex on the same body as v; `addCand(VERT_TRI, thv)`.
   - EDGE: canonical (self: `e2 > e1`), dedup per query, skip shared vertices and same-body endpoints; `the = 3·min(ghat, radii[a1], r_k) + max(dl[a1], dl[b1]) + fsweep_k`; box = edge AABB + `the + radii[a1] + fstale_k`.
   - `bvhBox`: stack DFS of the flex BVH (`bvh_aabb_dyn`, center/half), prune if `|c_node - c_q| > h_node + h_q` on any axis; children pushed c0 then c1 (c1 popped first); leaves output element id (capped at `maxout`).
6. `addCand(con, …, thresh, dfrom, dto, ghat, sink)`:
   - flex-flex (types 0/1) AABB pre-test: `lim = bandOffset + thresh`, `lim += 1e-9|lim| + 1e-12`; per axis gap between the two features' coordinate ranges; reject if `lim > 0 && Σ gap² > lim²`.
   - `g = mjc_pairGap(…, cutoff = thresh)`; `bo = bandOffset` (VT: `r_v + r_A`; EE: `r_a1 + r_a2`; VG: `r_v`; CT: `r_A`; GE: `r_a`); reject if `g >= bo + thresh`.
   - reject flex-flex with `g <= 0` (unsigned distance, coincident features); geom pairs with `g < 0` are KEPT (penetration must be pushed out).
   - closing bound: `bound = max over (i in side 1, j in side 2) |Δd_i - Δd_j|` where `Δd = dto - dfrom` per vertex (a static feature is a zero row); push iff `g < bo + pairBand(ghat) + bound`.
7. Sinks: serial → each candidate appended to the **arena** (`mj_arenaAllocByte`), forming a contiguous array (`*cand` = first). Threaded (`nthread ≥ 2`): units chunked (`CAND_CHUNK(n) = n/nsplit > minchunk ? ceil(n/nsplit) : minchunk`, `nsplit = 4·nthread`, `minchunk = 16`; geom units chunk `ceil(ngeom/(4·nthread))` when > 1), dispatched; each unit appends into a private linked list of 64-pair blocks drawn from a shared stack pool via `mj_atomic_add_size_t` (pool sized to half the free stack); after dispatch, blocks are copied to the arena **in serial unit order**, so the candidate list is identical to the serial run. Overflow (arena or pool) drops the remainder with `mju_warning("IPC: arena full after %d candidate pairs…")`.

### 9.5 `mjc_advance(…, cand, ncand, cgap, pt2flex, approut, toiout)` — conservative advancement over `x → x + α·dxw`
- `α = 1`; `approut[c] = 0`, `toiout[c] = 1`.
- For each candidate: `self` = type ≤ 1 and both sides in the same flex → subtract the mean displacement of its 4 vertices (coherent motion doesn't close a self gap).
- Closing-rate bound `l`: VT `|dp_v| + max_{A,B,C}|dp|`; EE `max(|dp_a1|,|dp_b1|) + max(|dp_a2|,|dp_b2|)`; others `max |dp|`.
- skip if `l < 1e-12`; `g0 = cgap[c]`; skip if `g0 <= 0` (solver owns it); skip if `l <= 0.8 g0` (cannot lose 80% this step).
- `approut[c] = 1`; per-pair TOI by iteration (≤32): `g = conGapAdv(t)` (gap with vertices at `x + t·dxw`; the multiply-add is guarded `dk != 0 ? t*dk : 0` to prevent FMA contraction), `room = g - 0.2 g0`; break if `room ≤ 1e-9 g0`; `t += room/l`; cap at 1.
- `toiout[c] = t`; `α = min(α, t)`. Each pair's TOI is computed independently (not cut at the running minimum).

---

## 10. IPC contact step (`engine_ipc.c:mj_ipc`)

Barrier-free augmented-Lagrangian contact for dim-2 flexes (Li et al., arXiv:2512.12151): per-pair multiplier λ and persistent active set replace the log barrier; a CCD-bounded committed position `xfree` keeps the configuration intersection-free. Called from `mj_step`/`mj_step2` instead of `mj_discrete` when `integrator == mjINT_DISCRETE && ENBL_IPC`. Elasticity is not solved here (elastic2d is in the effective metric; edge equality in native efc rows). Rigid contacts stay native (with friction). Ownership (`mj_collision` skips these): `mjc_ipcOwnsFlexFlex(f1,f2) = IPC && dim(f1)==2 && dim(f2)==2` (includes self); `mjc_ipcOwnsFlexGeom(f,g) = IPC && dim(f)==2 && body_weldid[geom_bodyid[g]]==0 && mjc_GeomSupported(type)`.

### 10.1 Constants
| name | value | meaning |
|---|---|---|
| `IPC_NPT` | 8 | max vertex slots per pair row |
| `IPC_ASET_AGE` | 25 | drop never-engaged pair after this many quiet updates |
| `IPC_EVICT` | 44 | drop engaged pair after this many quiet updates |
| `IPC_DECAY` | 0.9 | stiffness decay per quiet update |
| `IPC_ALPHA_LB` | 1e-6 | advance `xfree` only if CCD fraction exceeds |
| `IPC_STALL_MAX` | 64 | outer iterations without feasible advance → stop |
| `IPC_WS_TOL` | 1e-8 | working-set residual guard (m) |
| `IPC_DELTACAP` | 0.001 | standoff cap (m) |
| `IPC_GHAT` | 0.003 | detection band (m) |
| `IPC_VEL_TOL` | 0.05 | convergence: `|Δw|_∞ ≤ 0.05` (m/s) |
| `mjFLEXCONTACT_OMEGA2` | 5e7 | stiffness = OMEGA2 × min nonzero vertex mass |
| outer cap | 1024 | |

### 10.2 Per-pair quantities
- `μ_pair = 5e7 · min_{v, mass>0} mass[v]` (all-zero → mass 1e-9).
- standoff `δ = min(pairBand(ghat), 0.001)`.
- age exponent `c(cnt) = cnt ≥ 0 ? cnt : max(-cnt-6, 0)`; effective stiffness `D = μ·0.9^c`.
- linearization at `xfree`: `ld0 = pairGap(xfree)`, `ln`, `liv`, `lcw`, `lniv`; linearized raw constraint `c_raw(x) = ld0 - δ + Σ_p lcw_p ln·(x_{liv_p} - xfree_{liv_p})`.
- residual `r(x) = c_raw(x) - λ/μ` (`mjd_flexContactResidual(k, gap, s, lam) = gap - s - (lam ? lam/k : 0)`).
- contact cost `E_c = Σ_{r<0} 0.5·D·r²`.

### 10.3 Setup
- `flist` = dim-2 flex ids; points = their vertices concatenated (`fxadr`, `pt2vg`, `pt2flex`).
- A point is **free** iff its body has exactly 3 slide joints, carries exactly one flex vertex, and the vertex sits at the body origin: `dofadr`, `fidx`, `mass = mjd_flexVertMass` (= `d->M[M_rowadr[da] + M_rownnz[da] - 1]`, the diagonal entry of the first dof row `da = body_dofadr[weld]` of the vertex's weld body, if that body has exactly 3 dofs; else 0), `rad = flex_radius`.
- Articulated trees = trees with dofs that are not flex-vertex bodies; `own[dof]` = free-vertex slide dofs ∪ articulated tree dofs; `isflexdof`.
- Free vertex under a jointed tree → `mjERROR`. Pinned points on articulated bodies must have slide-only chains (else `mjERROR`); their mass = `body_subtreemass / (#pinned points on that body)`; point Jacobian `pjac` (`mj_jacSparse` at `flexvert_xpos`), velocity `pvel`.
- Static geom features `gv, ge` (static colliding geoms only).
- Tangent state: `w` (end-of-step velocity on owned dofs; displacement = `h·w`). `wtil = qvel + h·qacc_smooth` (free-flight predictor), `wfree = w = 0`. Points from tangent (`ipc_points`): free `x = xold + h·R_body·w[dofadr]`; pinned on articulated bodies by FK at `q_n ⊕ h·w` (`ipc_treePose` → `mj_integratePos`, `mj_kinematics`); static pinned unchanged.
- Initial candidates `mjc_candidates(x=xold, thresh=3·ghat, threshGeom=ghat, sweep xold→xtil)`. If none: flex dofs start at `wtil`.
- Active set seeded from all candidates with `λ = max over vertices flexvert_lambda`, `cnt = min over vertices flexvert_conage` (0 if none). `xfree = xold`.

### 10.4 Outer loop (≤ 1024 iterations, while `nown > 0`)
1. Linearize all active pairs at `xfree` (`ipc_gapLoop`: chunks of ≥256 pairs, `8·nthread` tasks, disjoint → deterministic).
2. **Working-set inner solve** (one inner iteration):
   - `inws[c] = r(xtil) < 0 || r(x) < 0`.
   - Loop: build one frictionless efc row per working pair: Jacobian over vertex slide dofs `lcw_p·(R^T ln)·h²` (body frame), plus body-chain columns for pinned points `lcw_p·(ln·pjac_j)·h²`; `D_row = D/h²`, `R = 1/D_row`; reference `aref = -(ld0 - δ - λ/μ) - Σ_p lcw_p ln·(xold + h·v - xfree)` (for free vertices `v = R·qvel`, pinned `pvel`). Rows appended after the native rows (`ipc_efcPublish`: sparse rows with columns insertion-sorted and duplicates merged; dense rows zero-filled), `efc_type = CONTACT_FRICTIONLESS`, `efc_id = -1`, initial state SATISFIED.
   - `mj_fwdConstraintCG` (nonlinear CG irrespective of `opt.solver`, iteration cap `opt.iterations`), collect rows' generalized force `qfrc_rows`, restore efc pointers. `solver_niter` accumulated.
   - Proposal `wprop = qvel + h·qacc` on owned dofs; `xprop`. Add every omitted pair with `r(xprop) < -1e-8`; if any added, warm-start from this `qacc` and re-solve; else stop. Restore `qacc_warmstart` afterwards.
   - Direction `ddw = wprop - w` (owned dofs).
3. Convergence flags: `flex_conv = max |ddw| over flex dofs ≤ 0.05`, `artic_conv` likewise over articulated dofs.
4. Line search on merit `E(w) = ½h²(a-a*)ᵀ M̂ (a-a*) + h²·cost_efc(a) + E_c(x)` with `a = (w - qvel)/h` on owned dofs (`qacc_smooth` elsewhere), `M̂` = M + effective metric (`mj_effMulAdd(flg_contact=1)`), `cost_efc` via `mj_constraintUpdate` over the native rows `nefc_qp`: `α = 1`, up to 8 halvings, accept when `E(w+α ddw) ≤ E0 + 1e-12` or Newton-converged; **always accept the last trial**. `w ← w + α·ddw`, `x ← xn`.
5. Multiplier update `ipc_flexLamUpdate` at x (with λ before update): inactive if `c_raw - λ/μ > 0` → `λ = 0`, `cnt += (cnt ≥ 0 ? 1 : -1)`; else `λ ← λ - c_raw·μ` (no clamp), `cnt = (cnt == 0 || cnt > 5) ? 0 : -1`. Per-vertex store `flexvert_lambda[vg] = max λ` over loaded pairs (zeroed first).
6. `ipc_ageStep(flexvert_conage)`: vertices touched by a pair with `λ>0` are loaded: `age = (age==0 || age>5) ? 0 : -1`; else `age += (age ≥ 0 ? 1 : -1)`.
7. Sweep `dxp = x - xfree` (zero for static pinned points); re-query `mjc_candidates(x=xfree, thresh=threshGeom=3·ghat, sweep xfree→x)`; gaps `cgap` at xfree; `ac = mjc_advance(xfree, dxp, …, appr, toi)`.
8. Admission: `actc = appr || cgap ≤ 0`; per vertex earliest TOI among approaching pairs `vmin`; an approaching pair with `cgap > 0` is kept only if `toi ≤ vmin[v] + 1e-12` for at least one of its vertices (first-layer filter).
9. `ipc_mergeActiveSet`: keep old pairs unless (`λ ≤ 0 && cnt < -44`) or `cnt > 25`; dedupe with open-addressing hash (`ipc_pairHash`: type·1000003 + (g+1), then for sorted idx `hh = hh·1000003 + (id+1)`, 64-bit; table size = next power of 2 ≥ `2·bound+16`, key 0 = empty); add admitted new candidates warm-started from per-vertex stores. The merged set is memmoved back to its fixed arena offset and `parena` reset (arena holds one set + one candidate list).
10. Commit: on owned dofs, articulated dofs not carrying pinned points take `wfree = w`; flex dofs and pinned-carrying trees advance `wfree += ac·(w - wfree)` if `ac > 1e-6`. `xfree` recomputed. `β ← β + (1-β)·ac`.
11. Terminate if `β ≥ 1 - 1e-6` and (`last α ≥ 1 - 1e-9` or both blocks converged). Stall counter resets when `ac > 1e-6`; 64 consecutive stalls → stop (`stalled`).

### 10.5 Finish
- Restore `qpos = q_n`, `mj_kinematics`, `mj_comPos` if articulated.
- `vend = wfree` on owned dofs (else `qvel`), `aeff = (wfree - qvel)/h` (else `qacc_smooth`), `qend = q_n ⊕ h·wfree` (non-owned dofs get zero displacement here).
- `qfrc_constraint` = native rows evaluated at `aeff` (`mj_constraintUpdate`) + `qfrc_rows` of the last solve.
- If any dim-2 flex: `qacc = aeff`, `flg_rnepost = 0`, `mj_sensorAcc` (acc-stage sensors recomputed).
- `mj_commit(m, d, act_dot, qend, vend, aeff)`.
- If `β < 1 - 1e-6`: warning ("CCD stalled…" or "outer iteration limit reached…").

---

## 11. Programmatic optimizations (C)

- Two-level pruning: body-level (bodyflex) SAP, then per-geom filters; precomputed OR-aggregated body masks; single-geom body fast path skipping the BVH.
- SAP in a covariance-aligned frame (cheap, inexact Jacobi tol 1e-3), float32 primary keys (halves sort bandwidth), column-major AAMM (cache locality), active list kept compact.
- OBB SAT with per-body-pair cached dot products (`product[36]`, `offset[12]`) for internal BVH nodes.
- BVH traversal descends into the larger "surface" node; explicit stack on mjData stack.
- Candidate pairs streamed onto the arena, then copied to stack so the arena region can be reused for contacts (no double allocation).
- Precontact buffer sized by `mj_maxContact` prefix sums → lock-free parallel narrowphase with `mju_dispatch`, 5 chunks/thread load balancing, thread-local CCD/EPA buffers via `mjc_setCCDBuffer`.
- Merge-join of explicit pairs and excludes against the sorted broadphase output.
- Contact-count reduction (greedy FPS) only for reducible flex groups.
- Continuous module: AABB pre-tests with ULP-scale padding, closing-bound pruning (relative displacement), serial-order-preserving threaded candidate generation, FMA-guarded advanced positions.
- IPC: working-set rows (only pairs that push), loose CG subsolve, memmove-compaction of the active set on the arena.

---

## 12. GPU (MuJoCo Warp) implementation

### 12.1 Data layout and precision
- All arrays batched over worlds: `geom_xpos[nworld, ngeom]`, `geom_xmat[nworld, ngeom]`; model arrays that can be randomized per world are 2-D `[nbatch, n]` indexed `worldid % shape[0]` (e.g. `geom_margin`, `geom_gap`, `geom_rbound`, `geom_aabb[nbatch, ngeom, 2]` (center, half), `pair_margin`, ...).
- **float32** throughout (Warp `float`), vs float64 in C.
- One global contact buffer for all worlds of size **`naconmax`** (default `nconmax·nworld`; `nconmax` default from `_default_nconmax`: smallest of {16,24,32,48,…,8192} ≥ `max(nv·0.35·10·(nhfield>0) + 45, 256·has_flex, 64·has_sdf, mjd.ncon)`); each contact carries `contact.worldid`. `d.nacon[0]` is a global atomic counter. Broadphase pair buffer (`CollisionContext.collision_pair/pairid/worldid`) also has `naconmax` slots, counter `d.ncollision[0]`. CCD scratch sized `naccdmax` (≤ naconmax).
- Unsupported (raises `NotImplementedError` at `put_model`): `ENBL_OVERRIDE`, `ENBL_IPC`, `ENBL_SPECULATIVE` (EnableBit only has ENERGY, INVDISCRETE, SLEEP), `DSBL_MIDPHASE`, `DSBL_AUTORESET`; `NATIVECCD` only changes BOX–BOX routing. `mjcb_contactfilter` replaced by `m.callback.contactfilter(m, d)` run after all contacts are written (it edits `d.contact`).

### 12.2 Compile-time pair precomputation (`io.py`)
- All geom pairs `g1 < g2` (`np.triu_indices(ngeom, k=1)`), linear index `upper_tri_index(n,i,j) = i(2n-i-3)/2 + j - 1`.
- `nxn_pairid[:,0]` (contact): `-1` = dynamic candidate; `-2` = filtered out by any of: bitmask (`(ct1&ca2)|(ct2&ca1) == 0`), same weld, both welds dof-less, parent filter (if enabled; weld≠0 both and parent-weld relation), exclude (`(body1<<16)+body2 ∈ exclude_signature`); `≥0` = explicit pair id (overrides, even if filtered). Geom-level filtering replaces C's body-level canCollide2 + per-geom bitmask (equivalent result).
- `nxn_pairid[:,1]` (collision sensors GEOMDIST/GEOMNORMAL/GEOMFROMTO): id into sensor collision list or -1. Sensor pairs are always emitted (no bounding-volume filter) and produce contacts of type `SENSOR` (even beyond margin) for sensors.
- `nxn_include = pairid0 > -2 || pairid1 >= 0`; `nxn_geom_pair_filtered`, `nxn_pairid_filtered`.
- `geom_pair_type_count[upper_trid_index(ntypes, t1, t2)]` counts included pairs per type pair; used to skip narrowphase kernels with zero pairs.
- **Broadphase auto-selection** (overrides any user choice at put_model): `nxn_include.sum() < 250000` → `NXN`; else `ngeom < 1000` → `SAP_TILE`; else `SAP_SEGMENTED`.
- Default `opt.broadphase_filter = PLANE | SPHERE | OBB` (AABB off).
- Sleeping: no `no_dofs`-type awake filter at compile time; runtime filter below.

### 12.3 Broadphase filter (`_broadphase_filter`, per candidate pair)
Effective margins: explicit pair → `m1 = pair_margin + pair_gap`, `m2 = 0`; else `m_i = geom_margin_i + geom_gap_i`.
- If `rbound1 == 0 || rbound2 == 0` (plane, or zero-size): only `PLANE` filter: if g1 plane, `dist = (xpos2-xpos1)·z1` keep iff `dist ≤ rbound2 + m1 + m2` (symmetric for g2; if both 0 → keep). Note: an `rbound==0` non-plane would be treated as a plane.
- else: `SPHERE`: keep iff `|xpos2-xpos1|² ≤ (rb1+rb2+m1+m2)²`; `AABB` (if enabled): world AABBs from rotated 8 corners of `geom_aabb` with sum margin, `<` separation tests; `OBB`: same 6-axis SAT as `mj_collideOBB` (no cache, no infinite handling — planes never reach it).
- Pair is emitted if filter passes **or** `pairid[1] >= 0` (sensor).

Compared with C: C applies OBB only to geom leaves of multi-geom bodies in the midphase; Warp applies it to every non-plane pair (by default). Since the OBB contains the geometry, this only removes pairs that cannot produce contacts (`dist < margin+gap`), so the contact SET matches in exact arithmetic.

### 12.4 `_add_geom_pair` (atomic emission)
`cid = atomic_add(ncollision, 0, 1)`; if `cid >= naconmax` → dropped (counter still incremented, flagged later as BROADPHASE overflow); store `pair = (type1 > type2) ? (g2, g1) : (g1, g2)`, `pairid` (vec2i), `worldid`.

### 12.5 `nxn_broadphase` — kernel `_nxn_broadphase`
- Launch `dim = (nworld, nnxn_geom_pair_filtered)`, grid-stride, `block_dim = m.block_dim.nxn_broadphase`, `max_blocks = 8·block_size·min_grid_size / block_dim`. One thread per (world, precomputed pair).
- Per-world mesh absence: `geom_dataid[world, g] < 0` for a MESH → skip (batched mesh assets).
- Sleep (if `ENBL_SLEEP`): skip if both bodies `ASLEEP`, or one `ASLEEP` and the other `STATIC`. Incremental pass (`awake_prev` given): only re-emit pairs that pass 1 skipped; wrapped in `wp.capture_if` on "any body awake state changed" when `graph_conditional`.
- Filter, then `_add_geom_pair`.

### 12.6 `sap_broadphase` (SAP_TILE / SAP_SEGMENTED)
1. `_sap_pair_margin`: `geom_pair_margin[batch, g] = max over explicit pairs containing g of (pair_margin + pair_gap)` (atomic_max), batched only if pair margins/gaps are.
2. `_sap_project` (`dim = (nworld, ngeom)`): fixed direction `normalize(0.5935, 0.7790, 0.1235)` (no PCA). `radius = rbound + max(geom_margin+geom_gap, geom_pair_margin)`, `rbound==0` (plane) → `MJ_MAXVAL`; `center = dir·xpos`; `lower = center - radius`, `upper = center + radius`; NaN center → `lower = upper = MJ_MAXVAL`; missing per-world mesh → inverted interval `(MAXVAL, -MAXVAL)`. `sort_index = geomid`; SAP_SEGMENTED also writes segment offsets `world·ngeom`.
3. Sort lower bounds with values=geom ids per world: SAP_TILE → `wp.tile_sort` in shared memory (`launch_tiled`, tile = ngeom, `block_dim.segmented_sort`); SAP_SEGMENTED → `wp.utils.segmented_sort_pairs` (radix). Tie order differs between the two (bitonic/tile vs radix).
4. `sap_range` (`dim=(nworld, ngeom)`): for sorted position i, `limit = first index in [i+1, n) with lower > upper[sort_index[i]]` (binary search `sap_binary_search`: first element `> value`), `limit = min(n-1, limit)`, `range[i] = limit - i`. Note: when a non-overlapping element exists, the candidates are sorted positions `i+1 … limit` **inclusive**, so one non-overlapping neighbour is included (harmless: filtered by the bounding-volume tests, but consumes a buffer slot if filters are disabled).
5. Inclusive `array_scan` of ranges (flattened over worlds) → work packages.
6. `_sap_broadphase`: `nsweep = 5·nworld·ngeom` threads, each strides `worldgeomid += nsweep` while `< total`; binary search maps package → (flattened sorted index i, offset) → `j = i + off + 1`; `worldid = i // ngeom`; geoms `sort_index[world, i%ngeom]`, `sort_index[world, j%ngeom]`; look up `nxn_pairid[upper_tri_index(min,max)]` (full, unfiltered table); skip if `pairid0 < -1 && pairid1 < 0`; mesh/sleep/incremental tests as NXN; filter; `_add_geom_pair`.
SAP cannot use the CUDA graph conditional (sort/scan allocate scratch).

### 12.7 Collision orchestration (`collision`)
```
if naconmax == 0 or DSBL_CONSTRAINT|CONTACT: nacon = 0; return
ctx = create_collision_context(naconmax)       # fresh arrays each call
ncollision = 0; if not incremental: nacon = 0
NXN ? nxn_broadphase : sap_broadphase
_narrowphase: table = MJ_COLLISION_TABLE (BOX-BOX → PRIMITIVE if DSBL_NATIVECCD)
              convex_narrowphase(CONVEX pairs); primitive_narrowphase(PRIMITIVE pairs); sdf_narrowphase if has_sdf_geom
flex_collision (only full pass; not sleeping-aware)
_contact_surfacevel if flg_surfacevel
m.callback.contactfilter(m, d) if set
_sort_contacts if opt.deterministic & CONTACTS   # TODO: currently a no-op
```
Sleeping: `forward.py` runs pass 1, `sleep.wake_collision`, snapshot `awake_prev`, `update_sleep`, then pass 2 (incremental, appends). C instead reruns the full `mj_collision` (replacing contacts) when anything woke.

### 12.8 Collision type routing (`MJ_COLLISION_TABLE`, `MJ_FLEX_COLLISION_TABLE`)
| pair | C function | Warp |
|---|---|---|
| PLANE–SPHERE/CAPSULE/CYLINDER/BOX | primitives | PRIMITIVE |
| PLANE–ELLIPSOID | `mjc_PlaneConvex` | PRIMITIVE |
| PLANE–MESH | `mjc_PlaneConvex` | PRIMITIVE (plane_convex) |
| PLANE–SDF | PlaneConvex | SDF kernel |
| HFIELD–{SPH,CAP,ELL,CYL,BOX,MESH} | `mjc_ConvexHField` | CONVEX (hfield CCD kernel) |
| SPHERE–SPHERE/CAPSULE/CYLINDER/BOX | primitives | PRIMITIVE |
| SPHERE–ELLIPSOID/MESH | `mjc_Convex` | CONVEX |
| CAPSULE–CAPSULE/BOX | primitives | PRIMITIVE |
| CAPSULE–ELL/CYL/MESH | Convex | CONVEX |
| ELL–*, CYL–CYL/BOX/MESH, BOX–MESH, MESH–MESH | Convex | CONVEX |
| **BOX–BOX** | **`mjc_BoxBox` primitive** | **CONVEX (GJK/EPA + multiccd box path)**, PRIMITIVE only if `DSBL_NATIVECCD` |
| *–SDF, HFIELD–SDF, MESH–SDF | SDF colliders | SDF kernel |
Flex table: PLANE/SPHERE/CAPSULE/BOX/CYLINDER–FLEX primitive, MESH/ELLIPSOID–FLEX convex; HFIELD–FLEX and SDF–FLEX not implemented.

Narrowphase kernels: one kernel per present type pair (specialized via `wp.static`); primitive kernel `dim = naconmax`, early-out `tid >= ncollision[0]`, branches over the present type pairs; convex kernels are grid-stride over `min(ncollision, naconmax)` with launch width `min(naconmax, waves·block·mingrid)` (`_ccd_grid_size`), each thread first checks the type pair, then claims a CCD scratch slot `atomic_add(nccd[geomgeomid])`, overflow (`>= naccdmax`) → `OverflowType.CCD` flag and drop. `epa_iterations = 16` if all convex pairs are box–box, else `opt.ccd_iterations`; `npolygonmax/nmeshdegmax` derived from present pair types (≥4/3 for box, ≥16 for cylinders under multiccd). Hfield pre-filter `_hfield_filter` uses `geom_margin` sum only (no gap; TODO in code).

### 12.9 Contact parameters (`collision_core.py`)
- `contact_margin_gap`: explicit → `(pair_margin, pair_gap)`; else sums. No override.
- `contact_material_params`: explicit → `pair_dim`, `pair_friction` (5-D), `pair_solref`, `pair_solreffriction` (copied as-is; C copies only if nonzero — same result when zero), `pair_solimp`, `pair_adhesion`. Dynamic: priority branch copies condim/friction/adhesion of the higher-priority geom and sets `mix = 1.0` or `0.0` so that solref and solimp are taken from that geom **through the mix formula** (`solref = mix·s1 + (1-mix)·s2` if both standard, else elementwise min!). Equal priority: `mix = safe_div(s1, s1+s2)` with the same three MJ_MINVAL special cases, `condim = max`, `friction = max`, `adhesion = a1 + a2`.
  - **Divergence:** with unequal priorities and at least one non-positive `solref[0]` (direct mode), C copies the higher-priority geom's solref, Warp takes the elementwise min of both. With equal priorities identical.
  - `solreffriction = (0,0)` for dynamic pairs.
  - Friction floor `max(MJ_MINMU, ·)` applied to all 5 entries (both explicit and dynamic), like `mj_assignFriction`.
- `write_contact(...)` (called by every collider for each contact):
  - `active = dist < margin || (adhesion != 0 && dist < margin + gap)`; `detected = dist < margin + gap`; `condim = (adhesion != 0 && dist >= margin) ? 1 : condim`.
  - skip if `(pairid0 == -2 || !detected) && pairid1 == -1`.
  - `contact.type = CONSTRAINT` if `pairid0 >= -1 && detected`; `|= SENSOR` if `pairid1 >= 0`.
  - `cid = atomic_add(nacon, 0, 1)`; write only if `cid < naconmax`: dist, pos, frame (supplied by collider), geom, worldid, `includemargin = margin`, dim, friction, solref, solreffriction, solimp, adhesion, type, `geomcollisionid` (index within pair, for sensors), `efc_address[:] = -1`.
  - In-gap contacts are written (C: written with `exclude=1`); Warp has no `exclude` field — the constraint builder recomputes `pos = dist - includemargin < 0 || adhesion != 0` and skips non-CONSTRAINT types.
- Contact frame: Warp colliders build frames with `math.make_frame(normal)`: `b = (0,1,0)` if `-0.5 < a_y < 0.5` else `(0,0,1)`, Gram–Schmidt, `c = a × b` — identical to `mju_makeFrame` when the collider supplies no tangent. C colliders that supply a tangent (|t|² ≥ 0.25: e.g. box-box, capsule-box, convex multicontact) get that tangent; Warp colliders may choose a different tangent → identical normal-direction physics but different friction-axis orientation (affects pyramidal cones and solver iterates).
- `_contact_surfacevel`: same formula as C (`frame[1]·svel, frame[2]·svel, frame[0]·sang`), with flex sides (`geom < 0`) contributing zero.

### 12.10 Flex collision in Warp (orchestration only)
`flex_collision`: `flex_broadphase_aabb` (per-flex AABB of vertices + radius + margin + gap), workspace, one SAP sort of all flex elements over all worlds (fixed direction as above, `segmented_sort_pairs`, `sap_range`, scan) shared by self and flex-flex; then (1) flex–geom candidates (plane–vertex, element–geom) → filter/write, (2) self-collision SAP sweep + narrowphase, (3) flex–flex. Candidates are written with atomics to a candidate buffer, grouped by a 64-bit key `(worldid, group(geom,flex)/(flex1,flex2), is_self) << 32 | spatial key` (projection on `(1,1,1)/√3` quantized at 1 µm, offset 1e8, clamped to int32), sorted, deduplicated by proximity (≈1 mm) and, for self/flex-flex, reduced with a parallel FPS to `MJ_MAXCONPAIR` (tie-break by element ids then candidate index). **Divergence:** C reduces only "reducible" (rigid or interpolated) flex groups, by greedy FPS seeded at the deepest contact, with no proximity dedup; Warp dedups all flex–geom contacts (no FPS) and dedups + FPS for flex self/flex–flex. Flex collision is not sleeping-aware. `mjc_ElemSharedNode`/cell midphase, IPC ownership, and `flex_selfcollide` mode selection (BVH/SAP/cells) are not replicated (Warp always uses SAP).

### 12.11 `bvh.py`
Not part of collision. It builds Warp `wp.Bvh`/`wp.Mesh` structures for **rendering and ray casting** (`render.py`, `ray.py`): scene BVH over enabled geoms and flex pieces (per-geom world AABB kernels: box/sphere/capsule/plane (finite plane extent `2·max(size)` or 1000 if non-positive, ±0.01 pad)/ellipsoid (row norms of `R·diag(size)`)/cylinder/mesh (box of mesh half-extent)/hfield), groups = world id, SAH constructor, `refit()` each frame; mesh and hfield triangle BVHs (leaf size 2); flex 2-D/3-D render meshes with vertex normals accumulated by atomics (`DeterministicMode.RUN_TO_RUN` when `opt.deterministic & ATOMICS`); splat BVHs. MJWarp has **no collision midphase**: there are no per-body BVHs; multi-geom bodies are handled pairwise by NXN/SAP over geoms. `DSBL_MIDPHASE` is therefore unsupported (meaningless).

### 12.12 Ordering, determinism, overflow
- Contact order in `d.contact` is the order of atomic increments across all worlds — **non-deterministic run to run** and different from C (C: broadphase-signature order, explicit pairs merged, nested geom loops, tree output sorted by (type,g1,g2), contacts within a pair in collider order). `DeterminismType.CONTACTS` (`_sort_contacts`) and `CONSTRAINT` (`_sort_constraints`) are declared but are no-ops at commit d41a47c. Efc rows are built per contact with atomics, so constraint order (and float summation order in the solver) varies; for PGS-like or order-dependent solvers and float32 reductions results can differ bitwise between runs. Only per-element atomics in some kernels honor `DeterminismType.ATOMICS`.
- Overflow is silent in-kernel: counters keep incrementing past capacity; `forward.py` step kernel sets `OverflowType.BROADPHASE` if `ncollision > naconmax` and `NARROWPHASE` if `nacon > naconmax` (prints on world 0 when `warn_overflow`), per-world `d.overflow` bitmask. Which pairs/contacts survive an overflow is scheduling-dependent. Downstream kernels clamp to `min(nacon, naconmax)`.
- Because the broadphase buffer shares the size `naconmax` with contacts, a model whose candidate pair count exceeds `naconmax` drops pairs even if the true contact count is small.

### 12.13 How to match C with MJWarp
- Use NXN (or accept SAP: same pair set modulo ties and the extra-neighbour slot) and generous `nconmax`/`naccdmax`.
- Expect contact-set equality (up to float32) for primitive/convex pairs except: BOX–BOX (CCD vs `mjc_BoxBox`; set `DSBL_NATIVECCD` in Warp to use its primitive box-box, but then C's convex pairs would also switch to libccd — so exact parity is not attainable for mixed models), PLANE–ELLIPSOID/PLANE–MESH (analytic vs support-point colliders, generally identical point sets), unequal-priority direct-solref mixing, tangent orientation, flex contacts (dedup/FPS rules).
- Sort contacts on the host by (worldid, geom1, geom2, geomcollisionid / position) to compare with C.

### 12.14 MJX (JAX) differences (brief)
- No runtime broadphase: `geom_pairs` enumerates candidate pairs statically (explicit pairs first, then body pairs `b1 ≤ b2` with bitmask/exclude/weld/parent filters at geom level; skips PLANE–PLANE and PLANE–HFIELD; does **not** skip "both dof-less" pairs or apply sleep). Pairs grouped by `FunctionKey(types, mesh data ids, condim[, hfield grid bounds])` so shapes are static.
- Every pair always emits exactly `func.ncon` contacts (non-colliding ones have large `dist`); contacts concatenated grouped by condim ascending. Optional `numeric` `max_geom_pairs` (top-k by bounding-sphere distance per group, excluding plane/hfield groups) and `max_contact_points` (top-k by dist per condim).
- Parameters: `includemargin = margin` with **no gap** and no adhesion; geom friction NOT floored at mjMINMU (pair friction clipped at `mjMINVAL`); priority copies friction/solref/solimp of the priority geom (correct C semantics, unlike Warp); condim from priority/max.
- Collider coverage differs: sphere/capsule/ellipsoid–cylinder and ellipsoid pairs use MJX's own SDF-style colliders (`collision_sdf.py`); BOX–BOX uses `box_box`; there is no flex collision and no `<pair>`-level gap; SDF geoms are sized by `sdf_initpoints` in `make_condim` but have no table entry.
- `collision()` returns immediately if the static contact count is 0; there is no overflow (sizes are static).

---

## 13. Reproduction pitfalls / parity notes

1. **Margin vs margin+gap.** Colliders and all filters receive `margin+gap`; `includemargin = margin` only. `exclude=1` iff `dist >= margin` (and not adhesive/speculative). Gap is never overridden; margin is overridden by `o_margin` under OVERRIDE, and SAP AAMMs use `0.5·o_margin` per object.
2. **Self flex margins.** ELEM_ELEM self pairs use margin 0 but **keep gap** in the narrowphase, while the BVH leaf filter uses margin 0 and no gap; `includemargin = 0`.
3. **Contact ordering** determines efc row order and therefore solver results bit-for-bit. Reproduce exactly: sorted unique bodyflex pairs; explicit pairs interleaved by signature (`<=` loop before the pair, emitted even when the bodyflex pair is later skipped by bitmask/exclude); nested loops `g1` over bf1's geoms outer, `g2` inner; midphase/cell outputs sorted by `(group, type, obj1, obj2)` with a stable sort; flex SAP outputs in sweep order; flex self-collision after all body pairs and leftover explicit pairs; contacts per pair in collider order; group reduction compacts preserving order.
4. **Type swap**: `pushGeomGeom` swaps only on strictly greater type; equal types keep (lower body first / pair_geom1 first). The normal sign follows g1→g2.
5. **SAP float cast**: primary-axis keys are `float`; only affects which pairs are found if intervals nearly touch (then later filters decide), but can change pair discovery when margins are tiny. Touching intervals count as overlapping.
6. `mj_SAP` silently truncates at `maxpair`; flex self SAP capped at 1e6 pairs.
7. **Plane handling**: planes are found only via the pre-pass (world body with geoms, or dof-less body with a plane). A plane on a body with dofs has a degenerate SAP interval (rbound 0) and can miss collisions.
8. **Sleep**: pre-pass passes `asleep1 = 0` for b1; explicit pairs are dropped if both bodies are not AWAKE; flex pairs need at least one side with awake dofs. Warp's "STATIC" sleep state includes mocap bodies, so asleep–mocap pairs are filtered in Warp but not in C (C: `weld==0` test, mocap weld ≠ 0).
9. **contactfilter callback** replaces (not augments) the bitmask test for dynamic pairs; explicit pairs never see it.
10. **mj_contactParam priority** copies adhesion of the winner only; equal priority sums adhesion; solref mixes only if both `solref[0] > 0`, else elementwise min (including mixing a standard with a direct one!). Friction is floored at `1e-5` after mixing. `pair_solreffriction` is used only if nonzero.
11. **Adhesive in-gap contacts** become `dim=1` (frictionless) but stay included; speculative contacts keep dim and are included even when `dist >= includemargin`.
12. **mju_makeFrame** fatal if normal norm < 0.5 (colliders must return unit normals); tangent used only if `|t|² ≥ 0.25`, then orthogonalized.
13. **BVH traversal "surface" heuristic** uses `aabb[3..5]-aabb[0..2]` (half minus center) — reproduce literally if you want the same pre-sort traversal (only matters for `bvh_active` visualization and stack depth; output is sorted anyway).
14. `filterPreContacts` tie-breaking: strict `>` → first in (pair, contact) order wins; initial next-best threshold −1.
15. Contact arena overflow drops whole pairs with a warning (subsequent smaller pairs can still be added). Candidate-pair arena overflow is fatal. Warp: overflow keeps counting, drops are scheduling-dependent.
16. **Speculative CCD** depends on `qvel`, `cdof`, `subtree_com` at collision time and on solref/solimp (via `contactDist`), integrator (metric vs not), and REFSAFE; occlusion tests depend on every other geom. Not available in Warp/MJX.
17. **IPC** reproducibility depends on: exact candidate order (serial order preserved under threads), the FMA guard in advanced positions, CG solver iterations (`opt.iterations`), cross-step per-vertex `flexvert_lambda`/`flexvert_conage`, hash-based dedup (64-bit wraparound arithmetic), absolute-length constants tuned for metre scale. Not available in Warp/MJX.
18. **Warp vs C priority mixing** with direct solref (see §12.9) and Warp's BOX–BOX CCD routing are the main deterministic (not just rounding) divergences for rigid contacts; MJX additionally ignores gap/adhesion and the friction floor.
19. **float32** in Warp: bounding-sphere/plane filters use `<=` (keep) mirroring C's `>` (discard) — equal at boundaries, but float32 rounding can flip borderline pairs.
20. Warp SAP direction is fixed and does not adapt to the scene; C's PCA frame changes per step — neither changes the final pair set (modulo float SAP keys), only efficiency.

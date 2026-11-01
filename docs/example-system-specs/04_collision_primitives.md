# 04 — Analytic narrowphase collision primitives

**Sources covered**

| Backend | File | Functions |
|---|---|---|
| C | `src/engine/engine_collision_primitive.c` | `mjraw_PlaneSphere`, `mjc_PlaneSphere`, `mjc_PlaneCapsule`, `mjc_PlaneCylinder`, `mjc_PlaneBox`, `mjraw_SphereSphere`, `mjc_SphereSphere`, `mjraw_SphereCapsule`, `mjc_SphereCapsule`, `mjc_SphereCylinder`, `mjraw_CapsuleCapsule`, `mjc_CapsuleCapsule`, `mjc_triWeights`, `clampSeg`, `mjc_segWeights` |
| C | `src/engine/engine_collision_box.c` | `mju_clampVec`, `mjraw_SphereBox`, `mjc_SphereBox`, `mjraw_CapsuleBox`, `mjc_CapsuleBox`, `clipHalfPlane`, `mjc_BoxBox` |
| C (related, plane:convex) | `src/engine/engine_collision_convex.c` | `mjc_PlaneConvex`, `hull4f`, `area4f`, `mjc_ellipsoidSupport`, `mjc_meshSupport`, `mjc_hillclimbSupport` (only as used by plane:convex) |
| C (dispatch context) | `src/engine/engine_collision_driver.c` | `mjCOLLISIONFUNC`, `mj_maxContact`, `collisionTask`, `getMargin`, `getGap`, `mj_setContact`; `engine_util_spatial.c:mju_makeFrame` |
| Warp | `mujoco_warp/_src/collision_primitive_core.py` | `plane_sphere`, `sphere_sphere`, `sphere_capsule`, `_clamp_segment_weights`, `_segment_weights`, `capsule_capsule`, `plane_capsule`, `plane_ellipsoid`, `plane_box`, `sphere_cylinder`, `plane_cylinder`, `_box_box_clip`, `box_box`, `sphere_box`, `capsule_box`, `sphere_triangle`, `box_triangle`, `tri_weights`, `capsule_triangle`, `cylinder_triangle` |
| Warp | `mujoco_warp/_src/collision_primitive.py` | `_area4`, `_hull4`, `plane_convex`, all `*_wrapper` funcs, `_PRIMITIVE_COLLISIONS`, `_primitive_narrowphase` kernel factory, `primitive_narrowphase` |
| Warp (context) | `collision_core.py` (`Geom`, `write_contact`, `contact_margin_gap`), `math.py` (`make_frame`, `orthogonals`, `safe_div`, `normalize_with_norm`), `collision_driver.py` (`MJ_COLLISION_TABLE`, `_narrowphase`) |
| MJX | `mjx/_src/collision_primitive.py` | `plane_sphere`, `plane_capsule`, `plane_ellipsoid`, `plane_cylinder`, `sphere_sphere`, `sphere_capsule`, `capsule_capsule` |

**Role in the pipeline.** After the broadphase produces candidate geom pairs (ordered so that `geom_type[g1] <= geom_type[g2]`), the narrowphase dispatches each pair to a collision function selected by the type pair. For the analytic pairs (plane with sphere/capsule/cylinder/box, sphere with sphere/capsule/cylinder/box, capsule with capsule/box, box with box, and plane with ellipsoid/mesh via a support-point routine) these closed-form routines compute up to N "pre-contacts" each consisting of signed distance `dist` (negative = penetration), world position `pos` (midway between the two surfaces), contact normal (unit, pointing **from geom1 to geom2**), and an optional first tangent direction. The driver then completes the contact frame, attaches solver parameters (margin, gap, friction, solref, solimp, condim) and writes into `d->contact`. All other pairs go to the general convex (GJK/EPA/MPR) collider, height-field or SDF routines (other spec files).

---

## 0. Common conventions (C)

### 0.1 Function signature and outputs

```c
typedef int (*mjfCollision)(const mjModel* m, mjData* d, mjPreContact* con,
                            int g1, int g2, mjtNum margin);
typedef struct mjPreContact_ { mjtNum dist; mjtNum pos[3]; mjtNum normal[3]; mjtNum tangent[3]; } mjPreContact;
```

- Inputs per geom: `pos = d->geom_xpos+3*g`, `mat = d->geom_xmat+9*g` (row-major 3x3, **column k = local axis k in world**, so local z-axis = `{mat[2], mat[5], mat[8]}`), `size = m->geom_size+3*g`.
- Size semantics: sphere `size[0]=r`; capsule `size[0]=r, size[1]=half-length of the segment` (cylindrical part); cylinder `size[0]=r, size[1]=half-height`; box `size[0..2]=half-extents`; ellipsoid `size[0..2]=semi-axes`; plane: **size is ignored by all collision routines — planes are infinite half-spaces** with outward normal = local z-axis, surface passing through `geom_xpos`.
- Return value = number of contacts written into `con[0..n-1]`.
- `margin` argument is **margin + gap** (detection band). The driver (`collisionTask`) computes `margin = getMargin(m,g1,g2,ipair)` (pair margin if explicit pair, else `geom_margin[g1]+geom_margin[g2]`, replaced by `opt.o_margin` when `mjENBL_OVERRIDE` is on) and `gap = getGap` (pair gap, else `geom_gap[g1]+geom_gap[g2]`), then calls `collision_func(m, d, conbuffer+conpos, g1, g2, margin + gap)`.
- **Acceptance test inside primitives is "dist <= margin(+gap)"** (rejection tests are written as `x > margin`, so equality is accepted). The driver then sets `con->includemargin = margin` (without gap) and `con->exclude = (dist >= includemargin) && !adhesion && !speculative`.
- `normal` points from geom1 to geom2. Because the driver swaps so that `type1 <= type2` (enum order `PLANE=0, HFIELD=1, SPHERE=2, CAPSULE=3, ELLIPSOID=4, CYLINDER=5, BOX=6, MESH=7, SDF=8`), geom1 is always the lower-type geom (e.g. the plane in plane:X, the sphere in sphere:box).
- `tangent`: zero in every primitive except plane:capsule (set to the capsule axis). The driver copies `normal` → `frame[0..2]`, `tangent` → `frame[3..5]`, then `mju_makeFrame(frame)`.

### 0.2 `mju_makeFrame` (engine_util_spatial.c) — completes the contact frame

```
normalize frame[0:3]   (error "xaxis of contact frame undefined" if norm < 0.5)
if dot(frame[3:6], frame[3:6]) < 0.25:          # tangent undefined
    frame[3:6] = (0,1,0) if -0.5 < frame[1] < 0.5 else (0,0,1)
frame[3:6] -= frame[0:3] * dot(frame[0:3], frame[3:6])
mju_normalize3(frame[3:6])                      # (1,0,0) if norm < mjMINVAL
frame[6:9] = cross(frame[0:3], frame[3:6])
```

### 0.3 Helpers and constants

- `mjMINVAL = 1e-15` (double; `1e-15f` single), `mjMAXVAL = 1e10`, `mjMAXCONPAIR = 50`.
- `mju_normalize3(v)`: `norm = sqrt(v·v)`; if `norm < mjMINVAL` sets `v=(1,0,0)` else scales by `1/norm` (multiplies by reciprocal, not divides); returns `norm` (pre-normalization).
- `mju_clip(x,lo,hi)`: `x<lo?lo:(x>hi?hi:x)`.
- `mji_*` are inline versions of `mju_*` (identical arithmetic).

### 0.4 Maximum contacts per pair (`mj_maxContact`, driver) — buffer sizing

| Pair | max | function actually produces |
|---|---|---|
| any with SPHERE or ELLIPSOID | 1 | 1 |
| BOX:BOX | 8 | ≤8 (clip buffer sized 12; exceeding 8 would trigger `mjERROR("collision function returned %d contacts, expected at most %d ...")`) |
| CAPSULE:CAPSULE | 2 | ≤2 |
| CAPSULE:BOX | 4 | ≤2 |
| PLANE with BOX/MESH/CYLINDER | 4 | ≤4 |
| PLANE:CAPSULE | 2 | 2 |
| others (convex) | depends on multiccd flags | — |

### 0.5 Dispatch table (subset relevant here, `mjCOLLISIONFUNC[type1][type2]`)

```
PLANE  : SPHERE→mjc_PlaneSphere  CAPSULE→mjc_PlaneCapsule  ELLIPSOID→mjc_PlaneConvex
         CYLINDER→mjc_PlaneCylinder  BOX→mjc_PlaneBox  MESH→mjc_PlaneConvex  SDF→mjc_PlaneConvex
SPHERE : SPHERE→mjc_SphereSphere  CAPSULE→mjc_SphereCapsule  CYLINDER→mjc_SphereCylinder  BOX→mjc_SphereBox
         (ELLIPSOID, MESH → mjc_Convex)
CAPSULE: CAPSULE→mjc_CapsuleCapsule  BOX→mjc_CapsuleBox   (ELLIPSOID, CYLINDER, MESH → mjc_Convex)
BOX    : BOX→mjc_BoxBox (always, independent of NATIVECCD flags)   MESH→mjc_Convex
PLANE:PLANE, PLANE:HFIELD, HFIELD:HFIELD → NULL (never collide)
```

---

## 1. Plane collisions (engine_collision_primitive.c)

Notation: `n = (mat1[2], mat1[5], mat1[8])` plane normal; `p1 = pos1` plane point.

### 1.1 `mjraw_PlaneSphere(con, margin, pos1, mat1, size1, pos2, mat2, size2)` (static)

Purpose: sphere (pos2, r=size2[0]) against half-space.

```
con[0].normal = n
tmp   = pos2 - pos1
cdist = dot(tmp, n)                    # center height above plane
if cdist > margin + r: return 0
con[0].dist = cdist - r
tmp = n * (-con[0].dist/2 - r)         # scale computed as one scalar: -dist/2 - r
con[0].pos = pos2 + tmp                # = sphere center - n*(r + dist/2)
con[0].tangent = 0
return 1
```

Note rejection is on `cdist > margin + r` (so `dist <= margin` accepted). The position lies halfway between the plane surface and the deepest sphere point along `-n`.

`mjc_PlaneSphere` just unpacks geom data and calls the raw function. Max 1 contact.

### 1.2 `mjc_PlaneCapsule`

```
axis    = (mat2[2], mat2[5], mat2[8])
segment = size2[1] * axis
endpoint = pos2 + segment
n1 = mjraw_PlaneSphere(con,      margin, pos1, mat1, size1, endpoint, mat2, size2)   # radius size2[0]
endpoint = pos2 - segment
n2 = mjraw_PlaneSphere(con + n1, margin, pos1, mat1, size1, endpoint, mat2, size2)
if n1: con[0].tangent  = axis
if n2: con[n1].tangent = axis
return n1 + n2
```

- Contact order: +axis endpoint first, then −axis endpoint; if the first is rejected, the second becomes `con[0]`.
- Tangent = capsule axis → `mju_makeFrame` orthogonalizes it against `n`, so the first friction direction is the projection of the capsule axis on the plane. When the axis is (numerically) parallel to `n`, the orthogonalized vector may be tiny; if its norm is `< mjMINVAL` it becomes `(1,0,0)` (frame then not orthonormal in general); otherwise it is the normalized noise direction.
- Max 2 contacts.

### 1.3 `mjc_PlaneCylinder` — up to 4 contacts

Cylinder: center `pos2`, axis `a = mat2 col 2`, radius `R = size2[0]`, half-height `H = size2[1]`.

Algorithm (order matters for parity):

```
normal = n ; axis = a
prjaxis = dot(normal, axis)
if prjaxis > 0: axis = -axis; prjaxis = -prjaxis        # axis now points toward the plane (prjaxis <= 0)
vec   = pos2 - pos1
dist0 = dot(vec, normal)                                # center height
vec   = axis*prjaxis ; vec -= normal                    # vec = axis*(a·n) - n  (component of -n orthogonal to axis)
len_sqr = dot(vec, vec)
if len_sqr >= mjMINVAL*mjMINVAL:                        # i.e. >= 1e-30
    scl = R / sqrt(len_sqr); vec *= scl                 # radial vector to the rim point nearest the plane
else:                                                   # disk parallel to plane
    vec = R * (mat2[0], mat2[3], mat2[6])               # cylinder local x-axis
prjvec = dot(vec, normal)                               # <= 0
axis   *= H ; prjaxis *= H                              # half-axis toward plane and its projection

# point 1: rim point on the near cap
d1 = dist0 + prjaxis + prjvec
if d1 <= margin:
    con[0] = {dist=d1, pos = pos2 + vec + axis - normal*(d1*0.5), normal, tangent=0}; cnt=1
else:
    return 0                                            # nearest point beyond margin → nothing
# point 2: rim point on the far cap (same radial direction)
d2 = dist0 - prjaxis + prjvec
if d2 <= margin:
    con[cnt++] = {dist=d2, pos = pos2 + vec - axis - normal*(d2*0.5), normal}
# points 3,4: two more rim points on the near cap, 120° from point 1
prjvec1 = -prjvec*0.5
d3 = dist0 + prjaxis + prjvec1
if d3 <= margin:
    vec1 = cross(vec, axis)            # axis is the H-scaled half-axis here
    mju_normalize3(vec1)               # (1,0,0) if degenerate
    vec1 *= R*sqrt(3)/2
    con[cnt++] = {dist=d3, pos = pos2 + vec1 + axis - 0.5*vec - normal*(d3*0.5), normal}   # A
    con[cnt++] = {dist=d3, pos = pos2 - vec1 + axis - 0.5*vec - normal*(d3*0.5), normal}   # B
return cnt
```

Geometry: points 1, A, B are the vertices of an equilateral triangle inscribed in the near cap rim (`vec`, `-vec/2 ± vec1`, with `|vec1| = R√3/2`). Their depth uses the fact that `dot(±vec1, normal) = 0` (vec1 ⟂ vec and ⟂ axis, and normal lies in span(vec, axis)), hence the shared `d3`. Since `prjvec <= 0`, `d1 <= d3` and `d1 <= d2`, so the early return is exact. Output order: [near rim, far rim (if within margin), A, B (both or neither)]. Possible counts: 0,1,2,3,4. Exact positions written with `mji_add3/addTo3/addToScl3` in the listed order (pos2 + vec (+/-) axis, then the `-0.5*vec`, then `-normal*dist*0.5`).

### 1.4 `mjc_PlaneBox` — up to 4 contacts

```
norm = n ; dif = pos2 - pos1 ; dist = dot(dif, norm)
cnt = 0
for i in 0..7:
    vec = ( i&1 ? +s0 : -s0, i&2 ? +s1 : -s1, i&4 ? +s2 : -s2 )        # local corner
    corner = mat2 * vec                                                 # world offset from box center
    ldist  = dot(norm, corner)
    if dist + ldist > margin or ldist > 0: continue                     # too far, or upper-half corner
    con[cnt] = {dist = dist+ldist, normal = norm,
                pos  = (corner + pos2) + norm * (-(dist+ldist)/2), tangent = 0}
    if ++cnt >= 4: return 4
return cnt
```

- Only corners on the plane-facing half of the box (`ldist <= 0`) are considered; corners are visited in binary index order (bit0 = x, bit1 = y, bit2 = z; set bit = +half-size) and the first 4 accepted are returned (no depth sorting).
- In generic orientation exactly 4 corners have `ldist < 0`; if some `ldist == 0` exactly (edge/face perpendicular to plane), more than 4 qualify and index order decides.

### 1.5 `mjc_PlaneConvex` (engine_collision_convex.c) — plane:ellipsoid, plane:mesh, plane:sdf

```
normal = n ; obj = CCD object for g2 with margin 0 ; vertindex = -1
v = support(obj, dir = -normal)          # world point
dist = dot(normal, v - pos1)
if dist > margin: return 0
con[0] = {dist, pos = v - 0.5*dist*normal, normal, tangent 0}
if geom_dataid[g2] == -1 or mesh_polynum == 0 or obj.vertindex < 0: return 1   # ellipsoid etc.
# mesh: add up to 3 more vertices of the incident polygon
local_normal = mat2ᵀ normal
best_poly = argmin over polygons adjacent to support vertex (mesh_polymap) of dot(polynormal, local_normal)
            (initial best_dot = 1, strict <, first wins)
if none: return 1
a = position of vertindex in that polygon's vertex list (0 if not found)
n = hull4f(indices, verts, polyvert, nvert, a)       # ≤4 indices, indices[0]=a
for i in 1..n-1:
    pnt = mat2 * vert[indices[i]] + pos2
    vdist = dot(normal, pnt - pos1)
    if vdist > margin or dot(normal, pnt - pos2) > 0: continue   # beyond band, or above mesh center
    con[ncon++] = {vdist, pnt - 0.5*vdist*normal, normal, 0}
return ncon
```

Support functions used:
- **Ellipsoid** (`mjc_ellipsoidSupport`): `ld = matᵀ dir`; `s = ld ⊙ size`; if `|s|² < mjMINVAL²` → returns `pos + size[0]*matcol0`; else `local = (s/|s|) ⊙ size`; `res = mat*local + pos` (via `localToGlobal`: three row dot products then add pos).
- **Mesh, vertnum < 10 or no graph** (`mjc_meshSupport`): exhaustive over vertices (float → mjtNum), `max=-FLT_MAX`, strict `>` (first index wins ties); records `vertindex`.
- **Mesh, vertnum ≥ 10 (mjMESH_HILLCLIMB_MIN) with graph** (`mjc_hillclimbSupport`): seed = `mesh_extrema[27*dataid + cx*9+cy*3+cz]` where `c = (ld>0.4) - (ld<-0.4) + 1` per component (fresh object → no warm start); hill-climb over vertex adjacency in the mesh graph (`graph[0]=numvert`, `vert_edgeadr = graph+2`, `vert_globalid = graph+2+numvert`, `edge_localid = graph+2+2*numvert`, neighbor lists terminated by -1), strict `>`; `vertindex = vert_globalid[imax]`.

`hull4f(res, hull, idx, nhull, a)`: if `nhull <= 4` returns `{a, a+1, a+2, a+3} mod nhull`, count nhull. Otherwise greedy rotating-caliper-like maximization of the quadrilateral area `area4f = 0.5*|(v_a - v_c) × (v_b - v_d)|` (float vertex differences done in mjtNum): advance d while area strictly increases; after each d advance, advance c while increasing, then b while increasing; stop when advancing d doesn't increase. Returns 4.

---

## 2. Sphere and capsule collisions (engine_collision_primitive.c)

### 2.1 `mjraw_SphereSphere` (static) — core used by many routines

```
dif = pos1 - pos2
cdist_sqr = dot(dif, dif)
min_dist  = margin + size1[0] + size2[0]
if cdist_sqr > min_dist*min_dist: return 0
con[0].dist   = sqrt(cdist_sqr) - size1[0] - size2[0]
con[0].normal = pos2 - pos1 ; len = mju_normalize3(normal)
if len < mjMINVAL:                                    # coincident centers
    normal = cross(mat1 col2, mat2 col2) ; mju_normalize3(normal)   # (1,0,0) if z-axes parallel
con[0].pos = pos1 + normal*(size1[0] + dist/2)
con[0].tangent = 0
return 1
```

Notes: squared-distance early-out (fails to reject if `min_dist < 0`; not an issue for nonnegative margins). `mat1/mat2` are only used in the degenerate branch; callers pass the original geoms' matrices even when `pos1/pos2` are synthesized points.

### 2.2 `mjraw_SphereCapsule` / `mjc_SphereCapsule`

```
axis = mat2 col 2 ; len = size2[1]
x = clip(dot(axis, pos1 - pos2), -len, len)
pt = pos2 + axis*x
return mjraw_SphereSphere(con, margin, pos1, mat1, size1, pt, mat2, size2)   # r2 = capsule radius
```
Max 1. Also used by flex (`mjc_GeomElem`, sphere vs 1-D flex element capsule).

### 2.3 `mjc_SphereCylinder` — 1 contact, three regimes

```
R = size2[0]; H = size2[1]; axis = mat2 col 2
vec = pos1 - pos2 ; x = dot(axis, vec)
a_proj = axis*x ; p_proj = vec - a_proj ; p_proj_sqr = dot(p_proj, p_proj)
collide_side = |x| < H
collide_cap  = p_proj_sqr < R*R
if collide_side and collide_cap:                    # sphere center inside cylinder
    dist_cap    = H - |x|
    dist_radius = R - sqrt(p_proj_sqr)
    if dist_cap < dist_radius: collide_side = 0     # strict <: tie → cap disabled, side kept
    else:                      collide_cap  = 0
if collide_side:                                    # side: sphere vs axis point with radius R
    return mjraw_SphereSphere(con, margin, pos1, mat1, size1, pos2 + a_proj, mat2, size2)
if collide_cap:                                     # cap: plane(cap) vs sphere, then flip normal
    flipmat = mat2 with columns 0 and 2 negated     # (-c0, c1, -c2): still a rotation
    if x > 0: pos_cap = pos2 + H*axis ; mat_cap = mat2
    else:     pos_cap = pos2 - H*axis ; mat_cap = flipmat
    ncon = mjraw_PlaneSphere(con, margin, pos_cap, mat_cap, size2, pos1, mat1, size1)
    if ncon: con[0].normal *= -1                    # plane normal points away from cylinder; flip to sphere→cyl
    return ncon
# corner (rim) region: point "sphere" of radius 0 on the rim
p_proj *= R / sqrt(p_proj_sqr)                       # nonzero since !collide_cap ⇒ p_proj_sqr >= R²
vec = axis*(x > 0 ? H : -H) + p_proj + pos2
return mjraw_SphereSphere(con, margin, pos1, mat1, size1, vec, mat2, {0})
```

Cap-case position is computed with the un-flipped plane normal: `pos = sphere_center - n_cap*(r + dist/2)` where `n_cap = ±axis`; `dist = dot(pos1 - pos_cap, n_cap) - r`.

### 2.4 `mjraw_CapsuleCapsule` / `mjc_CapsuleCapsule` — up to 2 contacts

```
axis1 = size1[1]*mat1col2 ; axis2 = size2[1]*mat2col2
p1 = pos1 - axis1 ; p2 = pos1 + axis1 ; q1 = pos2 - axis2 ; q2 = pos2 + axis2
n = mjc_segWeights(w, p1, p2, q1, q2)                 # 1, or 2 when parallel
ncon = 0
for i < n:
    vec1[k] = p1[k] + w[2i]   * (p2[k] - p1[k])
    vec2[k] = q1[k] + w[2i+1] * (q2[k] - q1[k])
    ncon += mjraw_SphereSphere(con+ncon, margin, vec1, mat1, size1, vec2, mat2, size2)
return ncon
```
Degenerate normal (coincident closest points) → `cross(axis1, axis2)` normalized, `(1,0,0)` if parallel. Also used for flex capsule elements.

### 2.5 `mjc_segWeights(w[4], a, b, c, d)` — closest points between segments AB, CD

Returns weights `(s,t)` in [0,1] of `A + s(B−A)`, `C + t(D−C)`; returns 1 or 2 pairs.

```
sin2 = 1e-8                                   # sin²(≈0.0057°) parallel cutoff
ab = b-a ; cd = d-c ; ca = a-c
dot1 = ab·ab ; dot2 = cd·cd ; dot3 = cd·ca ; dot4 = ab·ca ; dot5 = ab·cd
if dot1 <= mjMINVAL and dot2 <= mjMINVAL: w = (0,0); return 1
if dot1 <= mjMINVAL: w = (0, clip(dot3/dot2, 0, 1)); return 1
if dot2 <= mjMINVAL: w = (clip(-dot4/dot1, 0, 1), 0); return 1
s0 = -dot4/dot1                               # projection of C onto AB
s1 = (dot5 - dot4)/dot1                       # projection of D onto AB
cross = ab × cd ; det = cross·cross           # Lagrange identity: = dot1*dot2 - dot5²
if det > sin2*dot1*dot2:                      # non-parallel
    s = clip( (cross · (cd × ca)) / det, 0, 1 )   # Binet–Cauchy numerator
    clampSeg(w, s, (dot5*s + dot3)/dot2, s0, s1); return 1
clampSeg(w,   0, dot3/dot2,          s0, s1)  # parallel: pair at s=0 ...
clampSeg(w+2, 1, (dot5+dot3)/dot2,   s0, s1)  # ... and at s=1
return 2

clampSeg(w, s, t, s0, s1):
    if t < 0: w = (clip(s0,0,1), 0)
    elif t > 1: w = (clip(s1,0,1), 1)
    else: w = (s, t)
```
In the parallel case the two pairs bound the overlap interval; if the segments do not overlap both pairs may collapse to the same endpoint pair (both then produce identical contacts if within margin).

### 2.6 `mjc_triWeights(w[3], p, a, b, c)` — closest point on triangle (Ericson 5.1.5)

Used by flex (`sphereTriangle`), continuous collision, and core utilities; listed here because it lives in this file.

```
ab=b-a ac=c-a ap=p-a bp=p-b cp=p-c
d1=ab·ap d2=ac·ap d3=ab·bp d4=ac·bp d5=ab·cp d6=ac·cp
if d1<=0 and d2<=0: w=(1,0,0)
elif d3>=0 and d4<=d3: w=(0,1,0)
elif d6>=0 and d5<=d6: w=(0,0,1)
else:
  va=d3*d6-d5*d4 ; vb=d5*d2-d1*d6 ; vc=d1*d4-d3*d2
  if vc<=0 and d1>=0 and d3<=0: t=d1/(d1-d3); w=(1-t,t,0)
  elif vb<=0 and d2>=0 and d6<=0: t=d2/(d2-d6); w=(1-t,0,t)
  elif va<=0 and (d4-d3)>=0 and (d5-d6)>=0: t=(d4-d3)/((d4-d3)+(d5-d6)); w=(0,1-t,t)
  else: dd=1/(va+vb+vc); t=vb*dd; u=vc*dd; w=(1-t-u,t,u)
```

---

## 3. Box collisions (engine_collision_box.c)

### 3.1 `mju_clampVec(vec, limit, n)` (static)

`for i: if limit[i] > 0: vec[i] = clip(vec[i], -limit[i], limit[i])` (zero/negative limits leave the component unclamped).

### 3.2 `mjraw_SphereBox` / `mjc_SphereBox` — 1 contact

Sphere: `pos1`, `r = size1[0]`. Box: `pos2, mat2, size2`.

```
center  = mat2ᵀ (pos1 - pos2)                 # sphere center in box frame
clamped = clampVec(center, size2)             # closest box point
deepest = center
tmp = clamped - center ; dist = mju_normalize3(tmp)   # tmp: unit dir sphere→box (or (1,0,0))
if dist - r > margin: return 0
if dist <= mjMINVAL:                          # center inside (or on surface of) box
    closest = 2*(s0+s1+s2)
    for i in 0..5:                            # faces in order -x,+x,-y,+y,-z,+z
        f = | (i%2 ? +1 : -1)*size2[i/2] - center[i/2] |
        if closest > f: closest = f; k = i    # strict: first minimal face wins
    nearest = 0 ; nearest[k/2] = (k%2 ? -1 : +1)      # inward-pointing face normal (from sphere toward box interior → push direction)
    pos = center + nearest*((r - closest)/2)
    normal = mat2 * nearest
    dist = -closest
else:
    deepest = center + r*tmp                  # deepest sphere point toward box
    pos = 0.5*clamped + 0.5*deepest           # computed as zero, +=0.5*clamped, +=0.5*deepest
    normal = mat2 * tmp
con[0].pos = mat2*pos + pos2
con[0].dist = dist - r
con[0].tangent = 0
return 1
```
Normal direction: from sphere to box (geom1→geom2). In the inside case, for the −x face (k=0) the normal is `+x_box`, i.e. pointing from the sphere into the box (the box is pushed in +x relative to the sphere, the sphere out through −x). `con.dist = -closest - r`.

### 3.3 `mjraw_CapsuleBox` / `mjc_CapsuleBox` — up to 2 contacts

Strategy: find the point on the capsule segment closest to the box (testing segment endpoints vs faces and segment vs all 12 box edges), optionally a second segment point chosen heuristically when the segment lies at a low angle to a face/edge, then call `mjraw_SphereBox` for each (sphere radius = capsule radius). Everything is done in the box frame.

**Setup**
```
halflength = size1[1] ; secondpos = -4                # -4 = "no second contact"
pos      = mat2ᵀ (pos1 - pos2)                        # capsule center in box frame
axis     = mat2ᵀ (mat1 col 2)                         # unit capsule axis in box frame
halfaxis = halflength*axis
axisdir  = (halfaxis[0]>0) + 2*(halfaxis[1]>0) + 4*(halfaxis[2]>0)   # octant code of +axis
bestdistmax = (margin + 2*(size1[0] + halflength + s0 + s1 + s2))²    # compared vs squared dists
bestdist = bestdistmax ; bestsegmentpos = 0 ; cltype = -4 ; clcorner = 0
```

**Phase A: endpoint vs face.** For `i ∈ {-1, +1}`:
```
tip = pos + i*halfaxis ; tmp1 = tip clamped per-axis to [-s_j, s_j]
c1 = number of clamped axes ; c2 = last clamped axis (or -1)
if c1 > 1: continue                                   # endpoint in edge/corner Voronoi region
dist = |tmp1 - tip|²
if dist < bestdist: bestdist = dist; bestsegmentpos = i; cltype = -2 + i (→ -3 or -1); clface = c2
```
(`clface = -1` means the endpoint is inside the box.)

**Phase B: segment vs each box edge.** Loop `j = 0..2` (edge axis), `i = 0..7` with bit j of i clear (12 edges):
```
tmp3 = corner(i) with component j set to 0            # edge midpoint; edge = tmp3 + x1*s_j*e_j, x1∈[-1,1]
dif  = tmp3 - pos                                      # capsule point = pos + x2*halfaxis, x2∈[-1,1]
ma = s_j² ; mb = -s_j*halfaxis[j] ; mc = size1[1]²     # (mc uses halflength², = |halfaxis|²)
u  = -s_j*dif[j] ; v = halfaxis·dif
det = ma*mc - mb*mb ; if |det| < mjMINVAL: continue    # edge parallel to segment
idet = 1/det
x1 = (mc*u - mb*v)*idet ; x2 = (ma*v - mb*u)*idet ; s1 = s2 = 1
if   x1 > 1:  x1 = 1;  s1 = 2; x2 = (v - mb)*(1/mc)
elif x1 < -1: x1 = -1; s1 = 0; x2 = (v + mb)*(1/mc)
if   x2 > 1:  x2 = 1;  s2 = 2; x1 = (u - mb)*(1/ma); clamp x1 to [-1,1] updating s1 (2 or 0)
elif x2 < -1: x2 = -1; s2 = 0; x1 = (u + mb)*(1/ma); clamp x1 likewise
dif = tmp3 - pos - x2*halfaxis ; dif[j] += s_j*x1      # box point − capsule point
d2 = dif·dif ; c1 = 3*s1 + s2
if d2 < bestdist - mjMINVAL:                           # -mjMINVAL: prefer earlier candidate on near-ties
    bestdist = d2 ; bestsegmentpos = x2 ; bestboxpos = x1
    clcorner = i + (1<<j)*(c1/6)                       # c1/6 == 1 iff s1==2 (upper end of edge)
    cledge = j ; cltype = c1
```
The normal equations: minimize `|dif + x1 s_j e_j − x2 h|²` ⇒ `[ma mb; mb mc][x1; x2] = [u; v]`.

**Dead code.** A loop `for j<3 { if j==2 {...} }` computes 2-D quantities (`uu, vv, w, ee1, best, c1, l`), resets `bestdist = bestdistmax`, and divides by `dd.y` (may be inf/NaN); none of its results are used afterward (`c1` is overwritten before use; `bestdist` is no longer read). A reimplementation can omit it.

**cltype encoding.** `-3`/`-1`: face closest to the −/+ endpoint; `0..8`: edge, `cltype/3` = box edge position (0 lower corner, 1 middle, 2 upper corner), `cltype%3` = capsule position (0 lower end, 1 middle, 2 upper end). If `cltype == -4` → `return 0`.

**Phase C: second point.**

Case 1 — closest feature is a box corner (`cltype >= 0 && cltype/3 != 1`):
```
c1 = axisdir ^ clcorner
if c1 == 0 or c1 == 7: goto skip                       # axis points at/away from the corner
if c1 ∈ {1,2,4}: mul = +1; de = 1 - bestsegmentpos; dp = 1 + bestsegmentpos
if c1 ∈ {3,5,6}: mul = -1; c1 = 7 - c1; dp = 1 - bestsegmentpos; de = 1 + bestsegmentpos
(ax, ax1, ax2) = c1==1 ? (0,1,2) : c1==2 ? (1,2,0) : (2,0,1)
if axis[ax]² > 0.5:                                    # along a box edge (angle < 45°)
    secondpos = min(de, 2*s_ax/|halfaxis[ax]|) ; secondpos *= mul
else:                                                  # along a face
    secondpos = min(dp, 2*s_ax1/|halfaxis[ax1]|, 2*s_ax2/|halfaxis[ax2]|) ; secondpos *= -mul
```
(the `min` is done by sequential `if (e1 < secondpos) secondpos = e1`; division by zero gives inf which never wins.)

Case 2 — closest feature is the middle of a box edge (`cltype/3 == 1`):
```
c1 = (axisdir ^ clcorner) & (7 - (1<<cledge))
if c1 not in {1,2,4}: goto skip                        # T configuration
(ax1, ax2) = cledge==0 ? (1,2) : cledge==1 ? (2,0) : (0,1) ; ax = cledge
if |axis[ax1]| > |axis[ax2]|: ax1 = ax2
ax2 = 3 - ax - ax1
if c1 & (1<<ax2): mul = +1; secondpos = 1 - bestsegmentpos
else:             mul = -1; secondpos = 1 + bestsegmentpos
secondpos = min(secondpos, 2*s_ax2/|halfaxis[ax2]|)
e2 = ( ((axisdir & (1<<ax)) != 0) == ((c1 & (1<<ax2)) != 0) ) ? 1 - bestboxpos : 1 + bestboxpos
secondpos = min(secondpos, s_ax*e2/|halfaxis[ax]|)
secondpos *= mul
```

Case 3 — face closest to an endpoint (`cltype < 0`):
```
if clface == -1: goto skip                             # endpoint inside the box
mul = (cltype == -3) ? +1 : -1
secondpos = 2
tmp1 = pos - mul*halfaxis                              # the closest endpoint
for i in 0..2, i != clface:
    e1 = (s_i - tmp1[i]) / halfaxis[i] * mul ; if e1 > 0 and e1 < secondpos: secondpos = e1
    e1 = (-s_i - tmp1[i]) / halfaxis[i] * mul ; if e1 > 0 and e1 < secondpos: secondpos = e1
secondpos *= mul
```
(division then multiply by `mul`, in that order.)

**skip / output**
```
tmp1 = pos + bestsegmentpos*halfaxis ; tmp2 = mat2*tmp1 + pos2
n = mjraw_SphereBox(con, margin, tmp2, mat1, size1, pos2, mat2, size2)
if secondpos > -3:
    tmp1 = pos + (secondpos + bestsegmentpos)*halfaxis ; tmp2 = mat2*tmp1 + pos2
    n += mjraw_SphereBox(con + n, margin, tmp2, mat1, size1, pos2, mat2, size2)
return n
```
`secondpos` is an offset in segment-parameter units relative to the first point. Each sphere test applies its own margin early-out, so 0, 1 or 2 contacts. Normals from capsule to box. Also called from flex (box vs 1-D element, there with arguments swapped and normals negated afterwards).

### 3.4 `mjc_BoxBox` — SAT + face clipping, up to 8 contacts

Constants (double build; `mjUSESINGLE` values in parentheses):

| name | double | single | meaning |
|---|---|---|---|
| `mjBOXBOX_SEPEPS` | 1e-13 | 1e-6f | separation slack × Σ half-sizes |
| `mjBOXBOX_PAREPS` | 1e-16 | 1e-7f | sin² below which an edge-cross axis is skipped |
| `mjBOXBOX_SGNEPS` | 1e-9 | 1e-5f | axis component below which a support corner is ambiguous |
| `mjBOXBOX_DUPEPS` | 1e-14 | 1e-10f | squared relative radius for clip-vertex dedup |
| `mjBOXBOX_EDGEBIAS` | 1e-6 | 1e-6 | relative penalty on edge axes |
| `mjBOXBOX_MAXVERT` | 12 | 12 | clip buffer capacity |

**Setup**
```
pos21 = mat1ᵀ (pos2 - pos1)        # box2 center in box1 frame
pos12 = mat2ᵀ (pos1 - pos2)        # box1 center in box2 frame
rot   = mat1ᵀ mat2                 # rot[3i+j] = e1_i · e2_j ; column j = box2 axis j in box1 frame
rotabs = |rot| elementwise
septol = margin + SEPEPS*(Σsize1 + Σsize2)
```

**Stage 1: separating-axis test (15 axes)**
```
sep_best = -mjMAXVAL ; code = -1
for i in 0..2:   # box1 faces
    r2 = rotabs[3i+0]*s2_0 + rotabs[3i+1]*s2_1 + rotabs[3i+2]*s2_2
    sep = |pos21[i]| - s1_i - r2
    if sep > septol: return 0
    if sep > sep_best: sep_best = sep; code = i
for j in 0..2:   # box2 faces
    r1 = rotabs[0+j]*s1_0 + rotabs[3+j]*s1_1 + rotabs[6+j]*s1_2
    sep = |pos12[j]| - s2_j - r1
    if sep > septol: return 0
    if sep > sep_best: sep_best = sep; code = 3 + j
sep_face = sep_best ; code_face = code
for i in 0..2, j in 0..2:   # edge pairs, axis = e_i × (box2 axis j), in box1 frame
    i1=(i+1)%3 ; i2=(i+2)%3
    ax1 = -rot[3*i2+j] ; ax2 = rot[3*i1+j]           # components i1, i2 (component i is 0)
    norm2 = ax1² + ax2² ; if norm2 < PAREPS: continue
    inv = 1/sqrt(norm2) ; ax1 *= inv ; ax2 *= inv
    r1 = s1_i1*|ax1| + s1_i2*|ax2|
    j1=(j+1)%3 ; j2=(j+2)%3
    a2_1 = ax1*rot[3*i1+j1] + ax2*rot[3*i2+j1]
    a2_2 = ax1*rot[3*i1+j2] + ax2*rot[3*i2+j2]
    r2 = s2_j1*|a2_1| + s2_j2*|a2_2|
    sep = |ax1*pos21[i1] + ax2*pos21[i2]| - r1 - r2
    if sep > septol: return 0
    if sep - EDGEBIAS*|sep| > sep_best and sep > sep_face: sep_best = sep; code = 6 + 3i + j
if code < 0: return 0       # unreachable
```
Tie-breaking: strict `>` everywhere → lowest code wins among faces; box1 faces before box2 faces; an edge must exceed the current best by the relative bias and exceed the best face.

**Edge→face substitution.** If `code >= 6`: rebuild unit `axis` (box1 frame: `axis[i]=0, axis[i1]=-rot[3*i2+j], axis[i2]=rot[3*i1+j]`, `mju_normalize3`), `face_dot = |axis[code_face]|` if `code_face<3`, else `|axis · rot col (code_face-3)|`. If `face_dot > 0.99 && sep_best < sep_face + 0.05*|sep_face| + mjMINVAL` → `code = code_face; sep_best = sep_face`.

**Stage 2a: edge–edge contact (code ≥ 6)** — exactly 1 contact
```
(i, j) = ((code-6)/3, (code-6)%3) ; i1,i2,j1,j2 as above
axis = normalized as above ; if axis·pos21 < 0: axis = -axis      # from box1 toward box2
a2[k] = axis · rot col k                                          # axis in box2 frame
amb1 = i1 if |axis[i1]| < SGNEPS else (i2 if |axis[i2]| < SGNEPS else -1)
amb2 = j1 if |a2[j1]| < SGNEPS else (j2 if |a2[j2]| < SGNEPS else -1)
d2 = rot col j ; b = d2[i] ; denom = 1 - b²
best_d2 = mjMAXVAL
for v1 in 0..(amb1>=0 ? 1 : 0), v2 in 0..(amb2>=0 ? 1 : 0):
    c1 = 0 ; c1[i1] = axis[i1]>=0 ? s1_i1 : -s1_i1 ; c1[i2] = axis[i2]>=0 ? s1_i2 : -s1_i2
    if amb1>=0 and v1: c1[amb1] = -c1[amb1]
    cc = 0 ; cc[j1] = a2[j1]>=0 ? -s2_j1 : s2_j1 ; cc[j2] = a2[j2]>=0 ? -s2_j2 : s2_j2
    if amb2>=0 and v2: cc[amb2] = -cc[amb2]
    c2 = rot*cc + pos21                                           # box2 edge midpoint in box1 frame
    e = c2 - c1 ; d1e = e[i] ; d2e = d2·e
    s = denom < mjMINVAL ? 0 : (d1e - b*d2e)/denom
    s = clip(s, -s1_i, s1_i)
    t = clip(b*s - d2e, -s2_j, s2_j)
    s = clip(d1e + b*t, -s1_i, s1_i)
    p1 = c1 ; p1[i] += s ; p2 = c2 + t*d2
    gap2 = |p2 - p1|² ; if gap2 < best_d2: best_d2 = gap2; w1 = p1; w2 = p2
dist = (w2 - w1)·axis
if dist > septol: return 0            # NOTE: uses septol (with slack), not margin
mid = 0.5*(w1 + w2)
con[0] = {dist, pos = mat1*mid + pos1, normal = mat1*axis, tangent 0}
return 1
```

**Stage 2b: face contact (code < 6)**
```
ref1 = code < 3 ; a = ref1 ? code : code-3
sizeref/sizeinc/posref/matref = box (1 or 2) accordingly ; posoi = ref1 ? pos21 : pos12
rinc = ref1 ? rot : rotᵀ           # rinc(r,c) = component r of incident axis c in reference frame
sgn = posoi[a] >= 0 ? +1 : -1
binc = argmax_k |rinc(a,k)|        # strict >, starting from k=0
tinc = (sgn*rinc(a,binc) > 0) ? -1 : +1      # incident face normal opposes the reference normal
ax=(a+1)%3 ; ay=(a+2)%3 ; bu=(binc+1)%3 ; bv=(binc+2)%3
for r in 0..2 (c = [ax, ay, a][r]):
    cx[r] = posoi[c] + tinc*sizeinc[binc]*rinc(c,binc)
    du[r] = sizeinc[bu]*rinc(c,bu)
    dv[r] = sizeinc[bv]*rinc(c,bv)
cx[2] = sgn*cx[2] - sizeref[a] ; du[2] *= sgn ; dv[2] *= sgn     # z = height above reference face
poly[k] = cx + su*du + sv*dv for (su,sv) in [(1,1),(-1,1),(-1,-1),(1,-1)]
nvert = 4
nvert = clipHalfPlane(nvert, x <= +sizeref[ax])     # coord 0, sign +1
nvert = clipHalfPlane(nvert, -x <= sizeref[ax])     # coord 0, sign -1
nvert = clipHalfPlane(nvert, y <= +sizeref[ay])     # coord 1, sign +1
nvert = clipHalfPlane(nvert, -y <= sizeref[ay])     # coord 1, sign -1
dupe2 = DUPEPS*(sizeref[ax]² + sizeref[ay]²)
accepted = []
for k < nvert:
    if poly[k].z > margin: continue                 # margin here = margin+gap, no slack
    if any(dx²+dy² < dupe2 for accepted q): continue
    accepted.append(poly[k])
if none: return 0
normal = (ref1 ? sgn : -sgn) * matref col a          # world, box1 → box2
for each accepted v (in order):
    posc[ax] = v.x ; posc[ay] = v.y ; posc[a] = sgn*(sizeref[a] + 0.5*v.z)
    con[k] = {dist = v.z, pos = matref*posc + posref, normal, tangent 0}
return naccept
```

**`clipHalfPlane(nin, cur, spare, coord, sign, limit)`** (Sutherland–Hodgman, one plane):
```
d[k] = sign*in[k][coord] - limit ; if all d[k] <= 0: return nin (no copy, buffers not swapped)
nout = 0
for k: k1 = (k+1)%nin ; dp = d[k] ; dq = d[k1]
    if dp <= 0 and nout < MAXVERT: out[nout++] = in[k]
    if ((dp<0 and dq>0) or (dp>0 and dq<0)) and nout < MAXVERT:
        t = dp/(dp - dq) ; out[nout++] = in[k] + t*(in[k1] - in[k])     # x, y, z all interpolated
swap(cur, spare) ; return nout
```
A vertex exactly on the plane (`d=0`) is emitted once (as an inside vertex) and creates no intersection. The z coordinate is interpolated linearly — exact since the incident face is planar.

Properties: every surviving clipped vertex within the band becomes a contact (no reduction to 4); all contacts share one normal; depth measured along the reference normal; position halfway between reference face plane and incident surface.

---

## 4. GPU (MuJoCo Warp) implementation

### 4.1 Batching and kernel structure (`collision_primitive.py`)

- `collision_driver._narrowphase`: builds `collision_table` from `MJ_COLLISION_TABLE`. Primitive pairs: PLANE×{SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX, MESH}, SPHERE×{SPHERE, CAPSULE, CYLINDER, BOX}, CAPSULE×{CAPSULE, BOX}. **BOX:BOX is CONVEX (GJK/EPA multi-contact) by default; it is switched to the primitive `box_box` only when `opt.disableflags & DisableBit.NATIVECCD`.** (C always uses `mjc_BoxBox`.) Convex narrowphase runs first, then `primitive_narrowphase`.
- `primitive_narrowphase(m, d, ctx, collision_table)`: selects the subset of `_PRIMITIVE_COLLISIONS` present in the table **and** with `m.geom_pair_type_count[upper_trid_index(len(GeomType), t1, t2)] > 0` (count of filtered geom pairs of that type, computed at `io.put_model`). A kernel is generated per unique list of pair types (`@cache_kernel`, `module="unique"`) with a `wp.static` unrolled if-chain over the types.
- Launch: `dim = d.naconmax`; each thread `tid` handles candidate pair `ctx.collision_pair[tid]` (vec2i, already type-ordered: broadphase writes `(geom2, geom1)` when `type1 > type2`), `worldid = ctx.collision_worldid[tid]`, and exits if `tid >= d.ncollision[0]`. Pairs whose types match none of the compiled branches (e.g. convex pairs) do nothing. Parallelization axis: one thread per (world, geom pair).
- Per thread: `contact_params(...)` → margin, gap, condim, friction, solref, solreffriction, solimp, adhesion (pair overrides if `pairid[0] >= 0`; per-world parameter batching via `worldid % array.shape[0]`). `geom_collision_pair(...)` builds two `Geom` structs: `pos = geom_xpos[worldid,g]`, `rot = geom_xmat[worldid,g]`, `size = geom_size[worldid % nsize, g]`, `normal = rot column 2`, mesh topology pointers.
- Each wrapper calls a core function returning fixed-size arrays of candidate contacts and then calls `write_contact` once per slot.

### 4.2 `write_contact` (collision_core.py) — filtering and output

```
active   = dist < margin or (adhesion != 0 and dist < margin + gap)
detected = dist < margin + gap                       # STRICT <  (C primitives accept <=)
condim   = 1 if (adhesion != 0 and dist >= margin) else condim
if (pairid[0] == -2 or not detected) and pairid[1] == -1: return 0
type = CONSTRAINT if (pairid[0] >= -1 and detected) ; |= SENSOR if pairid[1] >= 0
cid = atomic_add(nacon, 0, 1)
if cid < naconmax: write dist, pos, frame, geom, worldid, includemargin = margin, dim, friction,
                   solref, solreffriction, solimp, adhesion, type, geomcollisionid = slot id, efc_address = -1
```
Consequences:
- Core functions mostly do **not** early-out on margin; they return all candidate slots (unused = `MJ_MAXVAL` or `inf`) and filtering happens here. Rejected contacts consume no slot.
- Contact order across pairs is **nondeterministic** (atomics); within a pair, slots are written in order but other threads interleave. `geomcollisionid` = the core slot index (e.g. corner index 0..7 for plane:box), not a compacted index.
- Overflow beyond `naconmax` silently drops contacts (counter still increments).
- Frames: `make_frame(normal)` = `normalize(a)`, `b = (0,1,0) if -0.5<a.y<0.5 else (0,0,1)`, `b -= a(a·b)`, `normalize(b)` (zero if `|a|==0`), `c = a×b`; rows `[a; b; c]`. Same rule as `mju_makeFrame` with zero tangent. `wp.normalize` of a zero vector returns zero (C returns (1,0,0)).

### 4.3 Precision

All arrays and math are float32 (`float` in Warp kernels is float32; `wp.vec3`, `wp.mat33` float32). `MJ_MINVAL = mujoco.mjMINVAL = 1e-15` and `MJ_MAXVAL = 1e10` are used as float32 literals. `safe_div(x, y) = x / (y if y != 0 else MJ_MINVAL)`.

### 4.4 Per-pair kernels and differences vs C

| Pair | Warp core (slots) | Algorithm parity | Differences |
|---|---|---|---|
| plane:sphere | `plane_sphere` (1) | same formula: `dist = (p2-p1)·n - r`, `pos = p2 - n(r + dist/2)` | no early-out; strict `<` acceptance; `pos` formula `r + 0.5*dist` (C uses `-dist/2 - r`, same value, rounding can differ in last bit) |
| sphere:sphere | `sphere_sphere` (1) | `dir = p2-p1`, `dist=|dir|`, `n = dir/dist` (division, C multiplies by reciprocal), `pos = p1 + n(r1 + 0.5 dist)` | coincident centers: test `dist == 0.0` (C: `< mjMINVAL`) and normal `(1,0,0)` (C: normalized `z1 × z2`, `(1,0,0)` only if parallel) |
| sphere:capsule | `sphere_capsule` (1) | identical: clamp projection to `[-h, h]`, then sphere_sphere | inherits sphere_sphere degenerate-normal difference |
| capsule:capsule | `capsule_capsule` (2) + `_segment_weights` (port of `mjc_segWeights`, same constants `1e-8`, `MJ_MINVAL`) | identical structure, 2 pairs if parallel | slot 1 dist = `inf` when not parallel; degenerate normal (1,0,0) |
| plane:capsule | `plane_capsule` (2) | endpoints +axis then −axis via plane_sphere | **frame**: `b = normalize(axis - n(n·axis))`; if `b_norm < 0.5` (axis within 30° of the plane normal) `b = (0,1,0)` if `-0.5<n.y<0.5` else `(0,0,1)` — **not orthogonalized against n**; `c = n×b`. C uses the normalized projected axis for all `|proj| >= mjMINVAL`. Frame tangents differ for near-upright capsules (affects pyramidal cones/anisotropic friction). |
| plane:ellipsoid | `plane_ellipsoid` (1) | `s = -normalize((Rᵀn) ⊙ size)`, `pos = c + R(s ⊙ size)`, `dist = n·(pos - p_plane)`, `pos -= 0.5 dist n` — same as C ellipsoid support | no degenerate fallback (`wp.normalize(0)=0`); C goes through generic `mjc_PlaneConvex` |
| plane:cylinder | `plane_cylinder` (4) | same 4 points and formulas | no early-out (all 4 always computed, filtered individually — equivalent given `d1 <= d2, d3`); **degenerate threshold `len_sqr >= 1e-12`** (C `1e-30`) and **fallback vec = world x-axis `(R,0,0)`** (C: cylinder local x-axis `R*mat2[:,0]`). If the plane is tilted, world x is not ⟂ n, so `prjvec ≠ 0` and depths are wrong in the degenerate (cylinder flat on plane) case; positions differ from C even on a horizontal plane. `vec1` uses `wp.normalize` (zero if degenerate; C (1,0,0)). |
| plane:box | `plane_box` (8) | corner enumeration identical | **no `ldist > 0` filter and no cap at 4**: all 8 corners are candidates, each accepted if `dist < margin+gap`. With large margin or penetration deeper than the box extent along n, Warp emits up to 8 contacts (C ≤ 4, lower half only). Slot index = corner index. |
| plane:mesh | `plane_convex` (4) + `_hull4` | same structure as `mjc_PlaneConvex` (support vertex, most anti-aligned adjacent polygon, `hull4` with anchor, skip vertices with `n·(p - mesh_center) > 0`) | support search: exhaustive if `graphadr == -1 or vertnum < 10` (same threshold); hill-climb **cold-starts at graph local vertex 0** (C seeds from 27-entry `mesh_extrema` grid). Ties (face parallel to plane) can choose a different support vertex → different anchor/polygon/contacts. Initial max `-MJ_MAXVAL` (C `-FLT_MAX`). Incident face skipped if `dist0 > margin+gap`. Area/cross products in float32. |
| sphere:cylinder | `sphere_cylinder` (1) | identical three regimes, same tie rule (`dist_cap < dist_radius` → side) | corner uses `wp.sign(x)*H` (sign(0)=+1; C uses `x > 0 ? H : -H`, i.e. −H at 0 — only reachable when H=0); `safe_div` for `1/sqrt(p_proj_sqr)` |
| sphere:box | `sphere_box` (1) | identical (inside test `dist <= MJ_MINVAL`, same face order and strict tie rule, `dist = -closest - r`) | clamps unconditionally (`max(-s, min(s, c))`; C skips non-positive limits); no early-out |
| capsule:box | `capsule_box` (2) | faithful port of `mjraw_CapsuleBox` phases A/B/C | `bestdist` initialized to `1e32` (C: `bestdistmax²`), `bestsegmentpos` init −12 (C 0), sentinels −123; divisions via `safe_div` (and `(s - t)*(mul/h)` instead of `(s - t)/h*mul` in case 3; `safe_div(v-mb, mc)` instead of `(v-mb)*(1/mc)`), so last-bit rounding differs; the dead 2-D block is replaced by different (also dead) code; second slot `MJ_MAXVAL` when no second point; per-contact frames via `make_frame` |
| box:box | `box_box` (≤8) | line-by-line port of `mjc_BoxBox` | uses the **single-precision constants** (`SEPEPS 1e-6, PAREPS 1e-7, SGNEPS 1e-5, DUPEPS 1e-10`, `EDGEBIAS 1e-6`); clip capacity `_BOXBOX_MAXVERT = 8` (C 12); called with `margin + gap`; returns `ncon`, wrapper writes only `ncon` slots, then `write_contact` re-filters with strict `dist < margin+gap` — **edge contacts in C's slack band `(margin+gap, septol]` are dropped in Warp** (C keeps them as excluded contacts). Only active when NATIVECCD is disabled. |

Additional core functions in `collision_primitive_core.py` used by flex collisions (`collision_flex.py`), not by `primitive_narrowphase`: `sphere_triangle` (2-D projected closest-point method with area signs — **not** the `mjc_triWeights`-based C `sphereTriangle`; returns `MJ_MAXVAL` beyond margin), `box_triangle` (11 slots: 3 triangle vertices vs box faces, 8 box corners vs triangle; mirrors C `boxTriangle` but slots are not compacted and no `mjMAXCONPAIR` cap), `capsule_triangle` (5 slots, mirrors C `capsuleTriangle`), `cylinder_triangle` (≤2, Warp-only heuristic), `tri_weights` (exact port of `mjc_triWeights`, used by `support.py`).

### 4.5 Box-box face path in Warp (implementation notes)

- `_box_box_clip(nin, poly: mat83f, coord, sign, limit) -> (nout, mat83f)`: identical rules to `clipHalfPlane` (returns the input matrix unchanged when all inside).
- Reference/incident selection uses `wp.where(ref1, ...)` on vectors/matrices; `rinc = where(ref1, rot, transpose(rot))`; corner signs generated as `su = +1 if k in {0,3} else -1`, `sv = +1 if k < 2 else -1` (same cyclic order as C).
- Accepted vertices are staged in `contact_pos` rows before being transformed in place.

### 4.6 How to match Warp exactly

Use float32 throughout, the Warp constants above, the strict acceptance `dist < margin + gap` applied per slot after the core function, the fixed slot layouts (plane:box 8 corners in index order, plane:cylinder 4, plane:capsule 2, capsule:capsule 2, capsule:box 2, plane:mesh 4), the plane:capsule frame rule, the plane:cylinder world-x fallback, and the BOX:BOX routing (convex unless NATIVECCD disabled). Contact ordering in `d.contact` cannot be reproduced bit-for-bit across runs (atomics); compare as sets keyed by `(worldid, geom, geomcollisionid)`.

### 4.7 MJX (JAX) — brief

`mjx/_src/collision_primitive.py` functions are vmapped over all pairs of a type group; they return fixed counts and **never filter by margin** (later stages use `dist`):
- `plane_sphere` (1), `sphere_sphere` (1; coincident → (1,0,0)), `plane_ellipsoid` (1; same formula as Warp).
- `plane_capsule` (2) with the same non-orthogonalized fallback frame as Warp (`b_norm < 0.5`).
- `plane_cylinder` (**3** contacts: near rim point + two triangle points; when `|prjaxis| < 1e-3` (cylinder lying on the plane) contact 1 is replaced by the far-cap rim point `d3`). Axis flip uses `-sign(prjaxis)` with `sign(0)=+1` (flips at 0). Degenerate threshold `len < 1e-12` with cylinder local x-axis fallback.
- `sphere_capsule` (1) via `closest_segment_point` with `t = (pt-a)·ab/(ab·ab + 1e-6)`.
- `capsule_capsule` (**1** contact only, no parallel two-point case) via `closest_segment_to_segment_points` (denominator regularized `+1e-6`).
- Other "primitive" pairs in MJX are not analytic: plane:box and plane:mesh → `plane_convex` (vertices within a 1 mm skin of the deepest, up to 4 manifold points), sphere:box → `sphere_convex`, capsule:box → `capsule_convex`, box:box → SAT `_box_box` in `collision_convex.py` (4 contacts), sphere:cylinder → SDF optimization (`collision_sdf.sphere_cylinder`).

---

## 5. Reproduction pitfalls / parity notes

1. **Margin semantics.** The `margin` argument of every C primitive is `margin + gap`; acceptance is `<=`. Warp/`write_contact` uses strict `<`. Contacts exactly on the boundary differ.
2. **Normal direction.** Always geom1→geom2 after the type-order swap. Sphere:cylinder cap case negates the plane normal after computing the position; sphere:box inside case uses `nearest[k/2] = (k%2 ? -1 : +1)` (sign opposite to the face's outward direction).
3. **Plane:box** C returns ≤4 contacts from the lower half (`ldist <= 0`) in corner-index order, stopping at the 4th; Warp returns up to 8 (no half filter). Do not depth-sort.
4. **Plane:cylinder degenerate branch** threshold and fallback axis differ (C: `len_sqr < 1e-30`, cylinder x-axis; Warp: `< 1e-12`, world x-axis; MJX: `len < 1e-12`, cylinder x-axis, only 3 contacts). For an upright cylinder in double precision, C usually takes the *general* branch with a noise direction (len_sqr ~1e-32..1e-20) — the rim contact locations then depend on rounding noise of `geom_xmat`; depths are unaffected.
5. **Plane:capsule tangent.** C stores the capsule axis as tangent and lets `mju_makeFrame` orthogonalize; Warp/MJX substitute a fixed axis when the projected axis norm < 0.5. Matters for pyramidal friction and torsional/rolling directions.
6. **Sphere-sphere degenerate normal** (coincident centers / coincident closest points): C uses normalized `z1×z2` (requires the original geoms' `xmat`), Warp/MJX `(1,0,0)`.
7. **Capsule:capsule parallel case** produces two contacts in C and Warp (cutoff `det <= 1e-8*|ab|²|cd|²`), one in MJX. `det` must be computed as `|ab×cd|²` (not `dot1*dot2 - dot5²`) and `s` via `(ab×cd)·(cd×ca)/det` to match rounding.
8. **Capsule:box** relies on exact comparisons (`< bestdist - mjMINVAL`, `axis[ax]² > 0.5`, octant XOR codes, `secondpos > -3`). Rounding of `halfaxis` components near 0 flips `axisdir` bits and can change whether a second contact exists; reproduce the exact operation order (`pos`, `axis` via `mat2ᵀ`, `halfaxis = axis*halflength`). The C edge solve uses `mc = size1[1]²`, not `|halfaxis|²`.
9. **Box:box constants** are precision-dependent: use the double set for C double builds, the float set for Warp/mjUSESINGLE. The edge-contact acceptance uses `septol` (with slack) while face contacts use the raw margin. Edge→face substitution thresholds `0.99` and `5%` are part of the algorithm; the face/edge tie bias `1e-6` is relative.
10. **Box:box contact count** is the number of deduplicated clipped vertices within the band (up to 8), not 4. Dedup is against previously *accepted* vertices only, in polygon order after clipping order (+x, −x, +y, −y).
11. **Order of contacts within a pair** matters for the C solver (warmstart mapping, constraint row order): reproduce the exact emission order of each routine (e.g. plane:capsule +axis end first; plane:cylinder [near, far, A, B]; box:box polygon order starting from incident corner (su,sv)=(1,1) after clipping).
12. **Planes are infinite**: no routine uses the plane's size.
13. **Plane:mesh support ties**: the chosen support vertex (and therefore the anchor and polygon) depends on the search method (exhaustive first-index vs. hill-climb seeded from `mesh_extrema`) — Warp seeds hill-climbing differently.
14. **BOX:BOX in Warp** goes through the convex (GJK/EPA, multi-contact) pipeline unless `NATIVECCD` is disabled; to compare with C `mjc_BoxBox`, disable NATIVECCD in Warp.
15. **Speculative contacts (C)**: after the primitive returns, if `mjENBL_SPECULATIVE` is on (and NATIVECCD not disabled) and no returned contact has `dist < margin` (true margin), the driver may replace the result by CCD (`mjc_advanceGeomPair`) — outside this file but changes the final contact set.
16. **Frame completion** in C errors if the normal norm < 0.5; all primitives here produce unit normals (fallbacks to (1,0,0) guarantee this). Warp silently normalizes; a zero normal yields a zero frame.

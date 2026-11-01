# 13 — Model Compiler and mj_setConst: everything computed at compile time

**Sources covered**

C/C++ (MuJoCo commit 7ba52b5):
- `src/user/user_model.cc` — `mjCModel::Compile`, `CompileTree`, `TryCompile`, `SetSizes`, `ComputeSparseSizes`, `CopyTree`, `CopyObjects`, `FinalizeSimple`, `AutoSpringDamper`, `LengthRange`, `FuseStatic`, `DiscardVisual`, `MakeInstances`, `CheckBodyMassInertia`, arena sizing, sparse structure construction
- `src/user/user_objects.cc` — `ResolveOrientation`, `mjCBoundingVolumeHierarchy` (BVH build), `mjCOctree` (octree + SDF coefficients), `mjCGauge` (frame scaling), `mjCBody::{Compile, InertiaFromGeom, AccumulateInertia, MergeInertial, ComputeBVH}`, `mjCFrame::Compile`, `mjCJoint::Compile`, `mjCGeom::{Compile, GetVolume, SetInertia, ComputeAABB, GetRBound}`, `mjCSite::Compile`, `mjCCamera::Compile`, `mjCHField::Compile`, `mjCPair::Compile`, `mjCEquality::Compile`, `mjCTendon::Compile`, `mjCWrap::ResolveReferences`, `mjCActuator::{Compile, ApplyGauge}`
- `src/user/user_mesh.cc` — `mjCMesh::{ProcessVertices, TryCompile, Process, ComputeVolume, ComputeSurfaceArea, ComputeInertia, ComputeFaceCentroid, ApplyTransformations, Rotate, MakeGraph (qhull), CopyGraph, MakeNormal, MakeCenter, MakePolygons, MakePolygonNormals, FitGeom, SetBoundingVolume}`, `triangle()`
- `src/user/user_util.cc` — `mjuu_*` math: `normvec`, `quat2mat`, `mulquat`, `z2quat`, `frame2quat`, `frameaccum*`, `globalinertia`, `offcenter`, `rotVecQuat`, `eig3`, `fullInertia`, `updateFrame`
- `src/user/user_init.c` — `mjs_default*` (spec defaults)
- `src/user/user_api.cc` — actuator shortcut setters (`mjs_setToPosition` etc.)
- `src/user/user_flexcomp.cc`, `user_composite.cc` — flexcomp body/mass generation, `mjCFlex::Compile` (structural parts), cable composite
- `src/engine/engine_setconst.c` (in full) — `mj_setConst`, `mj_setConstGroups`, `mj_setCompiledConst`, `mj_setLengthRange` and all static helpers
- `src/engine/engine_io.c` — `mj_makeDofDofSparse`, `mj_makeBSparse`, `mj_makeDofDofMaps`
- `src/engine/engine_init.c` — `mj_defaultOption`, `mj_defaultSolRefImp`, `mj_defaultLROpt`, `mj_defaultStatistic`
- `src/xml/generated/mjcf_read_table.inc`, `xml_native_reader.cc` — compiler/lengthrange attributes (only as they affect values)

GPU / JAX:
- MuJoCo Warp (commit d41a47c): `mujoco_warp/_src/set_const.py` (in full), `io.py::put_model` precomputations, `smooth.py::crb`
- MJX: `mjx/_src/io.py` (`_put_model_jax`, `_put_model_warp`)

**Role in the pipeline.** Everything in mjModel that is not literally copied from the spec is produced here, before the first `mj_step`. The compiler (C++, double precision) resolves defaults, orientations and units, computes body inertias from geoms and meshes, processes meshes (dedup, convex hull, inertia, principal-axes re-centering, BVH, octree/SDF), orders bodies/joints/dofs, builds the static sparse structures (M, D, B, tendon Jacobian, flex edge Jacobian), and then calls the engine routine `mj_setCompiledConst` (= `mj_setConst` + compile-only extras) which runs kinematics in `qpos0` and fills `dof_M0`, `*_invweight0`, `actuator_acc0`, `*_length0`, bounding volumes, `geom_rbound/aabb`, fluid added mass, tree/sleep metadata, statistics, damping-ratio conversion and spring lengths. Then `AutoSpringDamper` and `mj_setLengthRange` (simulation-based) run. MuJoCo Warp does **not** compile models: it ingests a compiled `mjModel` via `put_model` (copying all compile-time values, casting to float32) and offers a partial GPU `set_const` for runtime model edits.

---

## 0. Constants used in this file

| Name | Value | Where |
|---|---|---|
| `mjMINVAL` | `1e-15` (double build) / `1e-15f` | `mjtype.h` |
| `mjMAXVAL` | `1e10` | `mjtype.h` |
| `mjPI` | `3.14159265358979323846` | `mjtype.h` |
| `mjEPS` (compiler) | `1e-14` | `user_util.h` |
| `kFrameEps` (compiler) / `kSameFrameEps` (setconst) | `1e-6` | `user_model.cc`, `engine_setconst.c` (must match) |
| `kBoundEps` (setconst) | `1e-14` (must equal compiler mjEPS) | `engine_setconst.c` |
| `kEigTOL` | `4e-15` (Jacobi off-diagonal tol, relative to max abs element) | `user_util.cc` |
| `kEigEPS` | `1e-12` (eigenvalue swap threshold, relative) | `user_util.cc` |
| mesh `eig3` reltol | `1e-7` | `mjCMesh::Process` |
| mesh triangle inequality tolerances | `atol=1e-9`, `rtol=1e-6` | `mjCMesh::Process` |
| `kAlignTol` (gauge alignment) | `0.0349` (sine of ~2°) | `user_objects.cc` |
| `mjMAXTREEDEPTH` | `50` (warning only) | `mjmodel.h` |
| `mjMINAWAKE` | `10` | `mjmodel.h` |
| mesh polygon `kAngleTol` | `0.01` rad | `MakePolygons` |
| `mjuu_z2quat` degeneracy | `1e-10` | `user_util.cc` |
| body size floor (`bodySize`) | `1e-5` | `engine_setconst.c` |
| stat extent floor | `1e-5` | `setStat` |
| plane rbound for stat (infinite) | `0.01` | `setStat` |
| LR velocity-decay timeconst floor | `0.01` | `evalAct` |
| octree box inflation | `1.1 ×` half-size | `CreateOctree` |
| octree hanging-node coincidence | `1e-9` | `MarkHangingNodes` |
| octree Laplacian smoothing alpha | `0.2` | `ComputeSdfCoeffs` |
| `MakeNormal` large-angle removal threshold | dot `< 0.8` | `MakeNormal` |
| ellipsoid shell inertia eps | `1e-6` | `SetInertia` |
| Thomsen exponent (ellipsoid surface area) | `p = 1.6075` | `GetVolume` |
| mesh halfedge min area | `sqrt(mjMINVAL)` ≈ `3.16e-8` | `Process` |
| polygon-normal key clamp | `|n_z| > 1 - 1e-7` | `MeshPolygonKey` |
| body+flex count limit | `< 65534` | `CompileTree` |
| actuator history nsample limit | `≤ 2^24 = 16777216` | `mjCActuator::Compile` |

Spec defaults (from `mjs_default*`, `mj_defaultOption`, `mj_defaultSolRefImp`, `mj_defaultLROpt`, `mj_defaultStatistic`):

- `solref = (0.02, 1)`; `solimp = (0.9, 0.95, 0.001, 0.5, 2)` — for joint limit & friction, geom, flex, pair, equality, tendon limit & friction, `opt.o_solref/o_solimp`.
- compiler: `autolimits=1`, `degree=1` (MJCF angles in **degrees** by default; URDF parser sets `degree=false`), `eulerseq="xyz"`, `usethread=1`, `inertiafromgeom=AUTO`, `inertiagrouprange=[0, mjNGROUP-1=5]`, `boundmass=0`, `boundinertia=0`, `balanceinertia=0`, `settotalmass=-1`, `fitaabb=0`, `alignfree=0`, `fusestatic=0`, `discardvisual=0`.
- body: `quat=(1,0,0,0)`, `ipos[0]=NaN` (undefined), `iquat=(1,0,0,0)`, `fullinertia[0]=NaN`, `simple=1`, `fuse=1`.
- joint: `type=hinge`, `axis=(0,0,1)`, `limited=AUTO`, `actfrclimited=AUTO`, `align=AUTO`.
- geom: `type=sphere`, `quat=(1,0,0,0)`, `fromto[0]=NaN`, `contype=conaffinity=1`, `condim=3`, `friction=(1, 0.005, 0.0001)`, `solmix=1`, `mass=NaN`, `density=1000`, `typeinertia=VOLUME`, `fluid_coefs=(0.5, 0.25, 1.5, 1.0, 1.0)`, `fitscale=1`, `priority=0`, `margin=gap=0`.
- site: `size=(0.005,0.005,0.005)`.
- camera: `fovy=45` (degrees, stored as degrees), `ipd=0.068`, `resolution=(1,1)`.
- mesh: `scale=(1,1,1)`, `refquat=(1,0,0,0)`, `maxhullvert=-1`, `inertia=LEGACY`, `octree_maxdepth=6`.
- pair: `condim=3`, `friction=(1,1,0.005,0.0001,0.0001)` (but undefined-ness is tracked; see §8.2).
- equality: `type=connect`, `active=1`, `data[1]=1`, `data[10]=1` (torque:force ratio).
- tendon: `springlength=(-1,-1)` (meaning "compute at qpos_spring"), `width=0.003`, `limited=actfrclimited=AUTO`.
- actuator: `gaintype=FIXED`, `gainprm[0]=1`, `biastype=NONE`, `dyntype=NONE`, `dynprm[0]=1`, `actdim=-1`, `gear=(1,0,0,0,0,0)`, `trntype=UNDEFINED`, `ctrllimited=forcelimited=actlimited=AUTO`.
- flex: `dim=2`, `radius=0.005`, `cellcount=(1,1,1)`, `condim=3`, friction as geom, `thickness=-1`, `activelayers=1`, `selfcollide=AUTO`.
- option: `timestep=0.002`, `impratio=1`, `tolerance=1e-8`, `ls_tolerance=0.01`, `noslip_tolerance=1e-6`, `ccd_tolerance=1e-6`, `sleep_tolerance=1e-3`, `gravity=(0,0,-9.81)`, `magnetic=(0,-0.5,0)`, `density=viscosity=0`, `o_friction=(1,1,0.005,0.0001,0.0001)`, `integrator=Euler`, `cone=pyramidal`, `jacobian=auto`, `solver=Newton`, `iterations=100`, `ls_iterations=50`.
- LRopt: `mode=MUSCLE`, `useexisting=1`, `uselimit=0`, `accel=20`, `maxforce=0`, `timeconst=1`, `timestep=0.01`, `inttotal=10`, `interval=2`, `tolrange=0.05`.
- statistic default (`mj_defaultStatistic`, overwritten at compile): `center=0`, `extent=2`, `meaninertia=1`, `meanmass=1`, `meansize=0.2`. Spec stat fields default `NaN` (= auto).
- `memory=-1`, `njmax=-1`, `nconmax=-1`, `nstack=-1`, `nuser_*=-1` (auto).

`mjuu_defined(x)` ⇔ `!isnan(x)`; NaN is the universal "undefined" sentinel for spec doubles.

---

## 1. Compilation driver and order of operations

### 1.1 `mjCModel::Compile(vfs, m)` (outer)

1. If `compiler.discardvisual`: `DiscardVisual(vfs)` (§11.2) — mutates the spec.
2. If `compiler.fusestatic`: `FuseStatic(vfs)` (§11.1) — mutates the spec.
   Both are "operations on the spec" applied *before* compilation; failure aborts. Assets are compiled once and reused (`reuse_assets_`).
3. `Compile(vfs, m, treeonly=false, textures=true)` → `CopyFromSpec()`, install compiler log handler, `setjmp` for engine `mju_error` → converted to `mjCError`, then `TryCompile`.

### 1.2 `CompileTree(vfs, textures, keyframes)`

1. NaN self-test (`mjuu_defined(mjNAN)` must be false).
2. Error if the world body has joints; error if `nbody + nflex ≥ 65534`.
3. Add missing keyframes; zero `subtreedofs`.
4. `CopyFromSpec` for meshes/textures (when compiling assets) and skins; `CheckEmptyNames`.
5. `ProcessLists()` — assign ids, check name repeats (ids = list order; see §2).
6. `IndexAssets()`, `MakeInstances(reuse)` — each geom/site that uses a mesh/hfield gets the *instance* of the asset at the scale map of its use (`ShapeMap()`, the gauge rotated into the geom frame). A non-identity map spawns a separate instance named `"<name>@k"` with `instancemap_`; ids of instances follow the assets. SDF geoms mark their instance `needsdf`.
7. `pair->Compile()` for all pairs (to find geoms in pairs).
8. Mark meshes needing convex hull: any mesh/SDF geom with `contype||conaffinity` or in an explicit pair; any mesh with `inertia == CONVEX`; any mesh site.
9. `SetNuser()`: `nuser_X = max(userdata size)` over elements when `-1`.
10. Compile meshes and textures (multi-threaded when `usethread`; §6). Cached between operations.
11. Mark frames uncompiled; for every body in list order: `bodies_[i]->Compile()` (also compiles its frames, joints, geoms, sites, cameras, lights). Because the list is DFS preorder, parents compile before children (needed for `xpos0`, weldid).

### 1.3 `TryCompile(m, d, vfs)` — exact order

```
CompileTree(vfs, textures=true, keyframes=true)
flex->Compile(vfs)        for all flexes          (§10)
skin->Compile(vfs)        for all skins
hfield->Compile(vfs)      for compiled hfields    (§6.12)
material, pair, exclude, tendon, equality, actuator, sensor, numeric, text, tuple, plugin ->Compile()
def->Compile(this)        (userdata lengths of defaults)
sortid(pairs_, comparePair)       // ids by increasing signature, stable
sortid(excludes_, compareBodyPair)
IndexAssets(); MakeInstances(reuse=true)
SetSizes()                         (§2.3)
SaveDofOffsets(computesize=false)  // jnt->qposadr_, dofadr_
ComputeSparseSizes()               (§3.1)
assign nmocap / body.mocapid in body order
CheckBodyMassInertia for each body i>=1 with joints        (§4.6)
mj_makeModel(...)                   // allocate
m->opt = option; m->vis = visual
CopyNames; CopyPaths; CopyTree (§2.4); CopyPlugins
ResolveKeyframes; keys->Compile(m); ScaleKeyframe
CopyObjects (§2.5)
FinalizeSimple (§3.3)
m->nJmom = CountNJmom(m)
if compiler.settotalmass > 0: mj_setTotalmass(m, settotalmass)   (deprecated)
narena computation (§3.6)
sparsity structures: D, B, M, maps (§3.4)
history requires timestep>0
make raw mjData with opt.disableflags |= mjDSBL_CONTACT and enableflags &= ~mjENBL_SLEEP
mj_resetData
normalize keyframe quaternions (mj_normalizeQuat on each key_qpos)
mj_setCompiledConst(m, d)          (§12)
mj_resetPlastic; fill default key_plastic
AutoSpringDamper(m)                (§13.1)
LengthRange(m, d)                  (§14)
save *_auto statistics; override stat.{extent,meaninertia,meanmass,meansize,center} if defined in spec
mj_validateReferences
delete d; d = mj_makeData(m) (restoring flags first if any tree is mjSLEEP_INIT and sleep enabled)
mj_step(m, d) once (validation, unless asleep_init)
delete d; restore opt.disableflags/enableflags
BackValues; m->signature = Signature()
```

Note the temporary `disableflags |= mjDSBL_CONTACT` and sleep disable are in effect during `mj_setCompiledConst`, `AutoSpringDamper` and (partially overwritten) `LengthRange`; they are restored at the end.

`settotalmass` (deprecated) is applied **before** `mj_setConst`, so all derived quantities see the scaled masses:
```
scale = max(mjMINVAL, newmass / max(mjMINVAL, mj_getTotalmass(m)))
for i in 1..nbody-1: body_mass[i] *= scale; body_inertia[3i..3i+2] *= scale
```

---

## 2. Element ordering, ids, addresses (spec → model)

### 2.1 Tree lists (`MakeTreeLists`)

Recursive DFS **preorder** from world:
```
MakeTreeLists(body):
  if body != world: bodies.push(body)
  geoms  += body->geoms   (in body's list order)
  joints += body->joints
  sites  += body->sites
  cameras+= body->cameras
  lights += body->lights
  frames += body->frames
  for child in body->bodies: MakeTreeLists(child)
```
Consequences (must be replicated exactly):
- `body_id` = DFS preorder index, world = 0. Children always have larger ids than parents. A subtree is a contiguous id range `[b, b + subtreenum)` (Warp relies on this: `body_subtreenum`).
- Geoms (joints, sites, …) of one body are contiguous; `body_geomadr[i] = id of first geom` or −1; `body_jntadr` similarly.
- Joint order within a body = authoring order; dofs/qpos addresses are assigned sequentially over the global joint list: `qposadr += nq(type)`, `dofadr += nv(type)` with nq/nv = free 7/6, ball 4/3, slide/hinge 1/1.
- Frames do not create ids; elements inside frames belong to the frame's body in their insertion order in the body's lists (frames are flattened at spec-attach time; `MapFrame` appends child lists to the parent list).

Non-tree lists (flexes, meshes, skins, hfields, textures, materials, equalities, tendons, actuators, sensors, numerics, texts, tuples, keys, plugins) use authoring order. Exceptions:
- **pairs** and **excludes**: model ids are assigned by `std::stable_sort` on `signature = (body1 << 16) + body2` (with body1 ≤ body2 after swapping, see §8.2) — `pair_*` arrays are written at `pair->id`, so the model arrays are **sorted by signature**, ties in authoring order.
- mesh/hfield **instances** come after all primary assets.

### 2.2 `SaveDofOffsets`

Joints: `qposadr_`, `dofadr_`, `plasticadr_` (= running plastic counter if `springyield>0`, nq slots). Tendons: plastic 2 slots each if `springyield>0` (after all joints). Flexes: `PlasticNum()` slots (after tendons). Actuators: `actdim_ = actdim>0 ? actdim : spec.actdim>0 ? spec.actdim : (dyntype != NONE)`, `actadr_`, `ctrladr_` (−1 if `ctrlnum_==0`), `outadr_`. Bodies: `mocapid` sequential over mocap bodies in body order.

### 2.3 `SetSizes` (sizes that affect simulation)

- `ntree`: count joints that are the **first joint of their body** and whose body has **no ancestor with joints** (static ancestry).
- `nu = Σ ctrlnum_`, `nout = Σ outnum_`, `na = Σ actdim`.
- `nbvhstatic = Σ_body tree.Nbvh() + Σ_mesh tree.Nbvh()`, `noct = Σ_mesh octree nodes`, `nbvhdynamic = Σ_flex tree.Nbvh()`, `nbvh = static + dynamic`. Layout in `bvh_*` arrays: all bodies (in body order), then all meshes, then all flexes.
- flex counts: `nflexelemdata += nelem*(dim+1)`, `nflexelemedge += nelem*kNumEdges[dim-1]` (kNumEdges = {1,3,6}), `nJfe` = for each edge of non-rigid non-interpolated flex, Σ dofnum over the union of ancestor chains of the two vertex bodies (including world which has 0).
- `nefm0dof`, `nefm0L`: constant implicit bending-metric factor size (see §12.11); computed symbolically with `mju_cholFactorSymbolic` on the 3×3-block adjacency of flap stencils over "qualifying" dim-2 simple flexes with bending; vertices are bodies whose **weld body has exactly 3 dofs**. Must match the engine's recomputation.
- `nefmCvert = min(Σ nvert over non-rigid non-interpolated flexes that are (dim==1 with bending) or dim==2, nv/3)`.
- `nhistory`: per actuator with `nsample>0`: `2 + nsample + nsample*ctrlnum_`; per sensor `2 + nsample + nsample*dim`.
- `nplastic`: joints `nq` each (springyield>0), tendons 2, flexes `PlasticNum()`.
- `nemax`: connect 3, weld 7, else 1 per equality.
- `nwrap = Σ path sizes`, `nsensordata = Σ dim`.

### 2.4 `CopyTree` — body/joint/geom/site/cam/light fields

Per body i (in order): `body_parentid = parent ? parent.id : 0`, `body_weldid`, `body_mocapid`, `jntnum/jntadr`, `dofnum/dofadr` (−1 if none), `geomnum/geomadr`, `pos, quat, ipos, iquat, mass, inertia, gravcomp`, BVH (`body_bvhadr` running address, `bvh_child`, `bvh_depth`, `bvh_nodeid` = geom id of leaf or −1).

Checks: at most one free joint and only alone in its body; free joint only in children of world (`par->name == "world"`).

`body_rootid[i] = i` if i==0 or parent is world; else `rootid[parent]`.

**`body_simple`** (initial):
```
simple = IsNullPose(ipos, iquat)  &&
         (rootid[i] == i  ||  (parentid[parentid[i]] == 0 && body_dofnum[parentid[i]] == 0))
if (!spec.simple) simple = 0
if (parentid[i] > 0) body_simple[parentid[i]] = 0      // a parent is never simple (unless world)
for each joint j of body (in order):
   axis_aligned = (#components with |axis|>mjEPS) == 1
   if rotfound || !IsNullPose(jnt_pos) || ((hinge||slide) && !axis_aligned): simple = 0
   if ball||hinge: rotfound = 1
if simple && dofnum: simple = 2 if all joints are slides else 1
```
`IsNullPose(pos, quat)`: all `|pos_k| < 1e-6`, and quat within `1e-6` (componentwise) of `±(1,0,0,0)`. Note the "a parent is never simple" assignment happens while iterating bodies in order, so a parent processed earlier is overwritten when its child is processed. Final demotion by tendon armature in `FinalizeSimple` (§3.3).

Joint fields: `jnt_type, group, plasticadr, limited (is_limited), actfrclimited, actgravcomp, qposadr, dofadr, bodyid, pos, axis, stiffness (=stiffness[0]), stiffnesspoly (stiffness[1..mjNPOLY]), springyield, range, actfrcrange, solref(=solref_limit), solimp(=solimp_limit), margin`.

**qpos0 / qpos_spring**:
- free: `qpos0 = (body.pos (compiled), body.quat (compiled))`, `qpos_spring = qpos0`.
- ball: `qpos0 = (1,0,0,0)`, `qpos_spring = qpos0`.
- slide/hinge: `qpos0 = ref`, `qpos_spring = springref` (radians for hinge after degree conversion).

dof fields per joint dof: `dof_bodyid, dof_jntid, dof_solref(=solref_friction), dof_solimp(=solimp_friction), frictionloss, armature, damping(=damping[0]), dampingpoly(damping[1..])`, `dof_parentid = body.lastdof` (initialized from parent's `lastdof`, i.e. the last dof in the ancestor chain), then `lastdof = this dof`. So within one body, dof k's parent is dof k−1; the first dof's parent is the last dof of the nearest ancestor body with dofs, or −1.

Geom fields: `type, contype, conaffinity, condim, bodyid, dataid (mesh id for mesh/SDF type; hfield id; else −1), matid, group, priority, castshadow, size, aabb, pos, quat, friction, solmix, solref, solimp, margin, gap, surfacevel, adhesion, fluid, user, rgba`, `geom_rbound = GetRBound()` (non-zero only for mesh/SDF, §7.3).

Site/camera/light: straightforward copies (`cam_fovy` in degrees; `cam_intrinsic` computed in `mjCCamera::Compile`).

After the loop: `dof_treeid[i] = (#dofs j ≤ i with dof_parentid[j] == -1) - 1` (tree ids increase with dof index). `body_treeid[i] = body_dofnum[weldid] ? dof_treeid[body_dofadr[weldid]] : -1`. `tree_sleep_policy` initialized AUTO, overridden by body `sleep` attribute (only legal on the first body of a tree). Recompute `dof_Madr[i]` = Σ_{k<i} (depth(k)+1), validated against `nM`.

### 2.5 `CopyObjects` (non-tree)

- meshes: addresses accumulate; `mesh_scale, mesh_pos (=pos_, CoM offset), mesh_quat (=quat_, principal-axes rotation)`; `mesh_vert` (float32, re-centered/rotated), normals, faces, facenormal, texcoords, `mesh_graph`, `mesh_extrema[27]` (for each of the 27 cube features `(cx,cy,cz) ∈ {-1,0,1}^3` in loop order cx outer, cy, cz inner, the **local graph vertex id** maximizing `x*cx + y*cy + z*cz` in float, strict `>` so the first max wins; initialized `float lowest`; zero if no graph), polygons, polygon map, polygon normals (as mjtNum), mesh BVH (`bvh_aabb` copied — mesh BVH aabbs are fixed and **not** refit by setConst), octree (`oct_aabb` center/half-size, `oct_child[8]`, `oct_depth`, `oct_coeff[8]`).
- flexes: `flex_*` copies; `flex_edgeequality` = 1 if an `mjEQ_FLEX` equality names it, 3 if `mjEQ_FLEXSTRAIN`, else 0 (first match in equality order); `flex_vert` zeroed if `centered && !interpolated`; `flex_vertbodyid` all = `vertbodyid[0]` if rigid; `flex_interp = elastic2d ? -order : order`; `flex_edge`, `flex_edgeflap` (vertices 2,3 of the flap if dim==2 and elastic2d ∈ {1,3}, else −1); `flexedge_rigid = rigid ? 1 : (!interpolated ? weldid(b1)==weldid(b2) : 0)`.
- hfields: `hfield_size[4]`, `nrow, ncol`, data (normalized to [0,1], §6.12).
- pairs at `pair->id` (sorted order): `pair_dim, geom1, geom2, signature, solref, solreffriction, solimp, margin, gap, adhesion, friction[5]`.
- excludes: `exclude_signature[id]`.
- equalities: `eq_type, obj1id, obj2id, objtype, active0, solref, solimp, data[mjNEQDATA]`.
- tendons: fields; `tendon_lengthspring = springlength` (may be (−1,−1)); wraps: `wrap_prm = prm * prmscale` except sphere/cylinder where `wrap_prm = sideid` (side site id or −1).
- actuators: `actuator_trntype = so3_ ? mjTRN_SO3 : trntype`, gain/bias/dyn prm, `actnum=actdim`, `actadr`, ctrl/out blocks, `actuator_gear` and `actuator_lengthrange` replicated per output row, `actuator_ctrlrange/ctrllimited` per input.
- keyframes: copied then quaternions normalized (`mjuu_normvec`, which skips if `|norm-1| ≤ mjEPS` and returns 0/no-op if `norm² < mjEPS`).

---

## 3. Sparse structure precomputation

### 3.1 `ComputeSparseSizes` (pre-model)

With `dof_parentid_pre` built exactly as in §2.4:
```
nM = Σ_i (1 + #ancestors of dof i)            // legacy M (dof_Madr layout)
nD = 2*nM - nv                                // full symmetric D
nB = Σ_bodies (subtreedofs(b) + Σ_{ancestors a != world} dofnum(a))
nC = nv + Σ_{i : dof_simplenum[i]==0} #ancestors(i)     // reduced CSR M
nJten = Σ_fixed tendons |path|  +  Σ_spatial tendons |{dofs in ancestor chains of wrap bodies}|
```
`body_simple_pre` mirrors §2.4 but additionally demotes bodies touched by tendons with `armature > 0`. It is re-validated after `CopyTree` + `FinalizeSimple` (`nC_post == nC` else error).

### 3.2 `dof_simplenum`
```
count = 0
for i = nv-1 .. 0:
   count = body_simple[dof_bodyid[i]] ? count+1 : 0
   dof_simplenum[i] = count
```
(Run-length of consecutive simple dofs starting at i, going forward.)

### 3.3 `FinalizeSimple`
For each tendon with `tendon_armature != 0`: for each wrap of type SITE → `body_simple[site_bodyid] = 0`; CYLINDER/SPHERE → `body_simple[geom_bodyid] = 0`. Then recompute `dof_simplenum` and validate `nC`.

### 3.4 Structures built in `TryCompile` (engine `engine_io.c`)

**`mj_makeDofDofSparse(nv, nC, nD, nM, dof_parentid, dof_simplenum, rownnz, rowadr, diag, colind, reduced, upper, scratch)`**
```
rownnz = 0
for i = nv-1..0:
   rownnz[i]++                        // diagonal
   if !(reduced && dof_simplenum[i]):
      for j in ancestors(i): rownnz[i]++; if upper: rownnz[j]++
rowadr = exclusive prefix sum
remaining = rownnz
for i = nv-1..0:
   colind[rowadr[i] + --remaining[i]] = i
   if !(reduced && simplenum[i]):
      for j in ancestors(i) (nearest first):
          colind[rowadr[i] + --remaining[i]] = j
          if upper: colind[rowadr[j] + --remaining[j]] = i
```
Rows are filled from the end, so **colind is ascending within each row** (ancestors have smaller indices; in the upper part, descendants are inserted from high i to low i at decreasing positions, ending ascending). The diagonal is the **last** entry of each lower (M) row: `M[rowadr[i] + rownnz[i] - 1]` is `M(i,i)` — used by `setStat` and everywhere else.
- **D** (`reduced=0, upper=1`): full symmetric pattern, nnz = nD, `D_diag[i]` = position of i in row i.
- **M** (`reduced=1, upper=0`): lower-triangular CSR including diagonal; **simple dofs have diagonal-only rows**; nnz = nC.

**`mj_makeBSparse`**: body×dof pattern: row b = dofs of b's subtree ∪ dofs of b's ancestors (excluding world), sorted ascending (insertion sort). nnz = nB. (D row j equals B row `dof_bodyid[j]`; checked by `checkDBSparse` elsewhere.)

**`mj_makeDofDofMaps`**: `mapM2D` (lower → symmetric, via `mju_lower2SymMap`), `mapD2M` (`mju_sparseMap`), `mapM2M` (legacy `dof_Madr` layout → reduced CSR; −1 never allowed).

### 3.5 Tendon and flex-edge Jacobian sparsity (in `mj_setConst`, group FIXED)

`makeTendonSparse` (fixed rows `ten_J_rownnz/rowadr/colind`):
- fixed tendon: `colind = jnt_dofadr[wrap_objid[...]]` for each wrap in path order, `rownnz = num` (duplicates possible if a joint is repeated — not deduplicated; then sorted).
- spatial: for each wrap of type SITE (body = site body) or SPHERE/CYLINDER (geom body), walk `bid` up to (excluding) world, append each dof of each body if not already present (linear search). Pulleys contribute nothing.
- Then sort colind of the row by **O(n²) exchange sort** (stable result is just ascending order).
- `rowadr[i] = rowadr[i-1] + rownnz[i-1]`.

`makeFlexSparse` (`flexedge_J_*`): for each non-rigid, non-interpolated flex whose edges need a Jacobian (`flex_edgeequality || edgedamping || edgestiffness || flex_damping`), for each edge: `rowadr = prev rowadr + prev rownnz` (only for `ebase+e > 0`), `NV = mj_jacDifPair(..., b1, b2, ..., issparse=1, skipcommon=0)` gives the dof chain union (sorted); `rownnz = NV`. Edges of skipped flexes keep `rowadr=rownnz=0` (all zeroed first, including colind). **Pitfall:** a skipped flex's edges have `rowadr=0`, and the next active flex's first edge computes `rowadr` from the previous edge entry (which is 0+0) — so addresses restart at 0 after skipped flexes only if all previous were skipped; replicate literally.

### 3.6 Arena size (`m->narena`)

If `memory != -1`: `narena = memory` bytes. Else:
```
nconmax' = nconmax==-1 ? 100 : nconmax;  njmax' = njmax==-1 ? 500 : njmax
if nstack == -1:
    if mj_isSparse(m): EstimateFlexArena(...)  -> nflexdof, nflexdof_collide, nH_flex, nL_flex
    ndense  = njmax' + neq + nv - nflexdof
    nlinear = nq+nv+nu+na+nbody+njnt+ngeom+nsite+neq+ntendon+nwrap
    narena  = sizeof(mjtNum) * max(1000, 5*ndense^2 + 20*nlinear)
else: narena = sizeof(mjtNum) * nstack
narena += nconmax'*sizeof(mjContact) + njmax'*(8*sizeof(int)+14*sizeof(mjtNum)) + nv*3*sizeof(int)
        + njmax'*nv*(2*sizeof(int)+2*sizeof(mjtNum)) + njmax'^2*(sizeof(int)+sizeof(mjtNum))
if nflexdof: flex_solver_bytes = nH_flex*(6 int + 5 mjtNum) + nL_flex*(3 int + nL_num mjtNum),
             nL_num = elliptic ? 2 : 1;  njmax_flex_bytes = njmax'*nflexdof*(2 int + 2 mjtNum)
             narena += max(0, flex_solver_bytes - njmax_flex_bytes) + flex_solver_bytes/2
             if nflexdof_collide: narena += min(njmax'*nflexdof_collide*(2 int+2 mjtNum), 64 MiB)
round narena up to a multiple of 1 MiB
```
`mj_isSparse`: `jacobian==SPARSE || (jacobian==AUTO && nv >= 60)`. (Warp's `is_sparse`: AUTO → `nv > 32`. **Different threshold.**)

---

## 4. Frames, orientations, units, frame scaling

### 4.1 Compiler math primitives (`user_util.cc`, all double)

- `mjuu_normvec(v, n)`: `nrm2 = Σ v²`; if `nrm2 < mjEPS (1e-14)` return 0 **without modifying v**; `nrm = sqrt(nrm2)`; if `|nrm - 1| > mjEPS` divide; return nrm. (Note: the threshold is on the *squared* norm.)
- `mjuu_quat2mat(R, q)`: exact identity if q == (1,0,0,0) bitwise; else standard
  `R00=q0²+q1²−q2²−q3²`, `R11=q0²−q1²+q2²−q3²`, `R22=q0²−q1²−q2²+q3²`, `R01=2(q1q2−q0q3)`, `R02=2(q1q3+q0q2)`, `R10=2(q1q2+q0q3)`, `R12=2(q2q3−q0q1)`, `R20=2(q1q3−q0q2)`, `R21=2(q2q3+q0q1)` (row-major). Not normalized internally.
- `mjuu_mulquat(res, a, b)`: Hamilton product then **`mjuu_normvec(res,4)`**.
- `mjuu_rotVecQuat(res, v, q)`: zero v → zero; q == identity bitwise → copy; else `t = q0 v + q⃗ × v`, `res = v + 2 q⃗ × t`.
- `mjuu_frameaccum(pos, quat, cpos, cquat)`: `pos += R(quat) cpos`, `quat = normalize(quat * cquat)`.
- `mjuu_frameaccumChild(pos, quat, cpos, cquat)`: same but writes the result into `cpos, cquat` (child expressed in parent frame).
- `mjuu_frameinvert`: `newpos = -R(q)^T pos`, `newquat = conj(q)`.
- `mjuu_frameaccuminv(pos, quat, cpos, cquat)`: `quat = normalize(quat * conj(cquat))`, then `pos -= R(quat) cpos`.
- `mjuu_z2quat(q, v)`: `q⃗ = (0,0,1) × v`, `s = normvec(q⃗)`; if `s < 1e-10`: `q⃗ = (1,0,0)`; `ang = atan2(s, v_z)`; `q = (cos(ang/2), q⃗ sin(ang/2))`. (v need not be unit; for antiparallel v this yields rotation by π about x.)
- `mjuu_frame2quat(q, x, y, z)` (axes as matrix columns `mat[c][r]`): Shepperd's method:
  - if trace > 0: `q0 = ½√(1+tr)`, `q1 = ¼(m[1][2]−m[2][1])/q0`, `q2 = ¼(m[2][0]−m[0][2])/q0`, `q3 = ¼(m[0][1]−m[1][0])/q0`
  - elif `x0 > y1 && x0 > z2`: `q1 = ½√(1+x0−y1−z2)`, `q0 = ¼(m[1][2]−m[2][1])/q1`, `q2 = ¼(m[1][0]+m[0][1])/q1`, `q3 = ¼(m[2][0]+m[0][2])/q1`
  - elif `y1 > z2`: `q2 = ½√(1−x0+y1−z2)`, `q0 = ¼(m[2][0]−m[0][2])/q2`, `q1 = ¼(m[1][0]+m[0][1])/q2`, `q3 = ¼(m[2][1]+m[1][2])/q2`
  - else `q3 = ½√(1−x0−y1+z2)`, `q0 = ¼(m[0][1]−m[1][0])/q3`, `q1 = ¼(m[2][0]+m[0][2])/q3`, `q2 = ¼(m[2][1]+m[1][2])/q3`
  - then `normvec(q,4)`.
  (Here `m[c][r]` = component r of axis c.)
- `mjuu_globalinertia(G, I, q)`: `G = R diag(I) R^T` in (xx, yy, zz, xy, xz, yz) order, computed as `tmp[3i+k] = R[k*3+i]*I[i]` then sums (see source order; replicate for bitwise parity).
- `mjuu_offcenter(res, m, d)`: parallel axis `m*(dy²+dz², dx²+dz², dx²+dy², −dx dy, −dx dz, −dy dz)`.
- `mjuu_mulRMRT(res, R, M)`: `res = R M R^T` (tmp = R*M first, then tmp*R^T).

### 4.2 `ResolveOrientation(quat, degree, eulerseq, orient)`

Called for bodies (`alt`), inertial (`ialt`), geoms/sites (unless fromto), cameras, frames. If `orient.type == QUAT` the input quat is left as-is (normalization happens elsewhere). Otherwise:

- **axisangle** `(ax, ay, az, θ)`: if `degree`: `θ = θ/180*π`; `normvec(axis)` must be ≥ mjEPS (else error "axisangle too small"); `q = (cos(θ/2), sin(θ/2)·axis)`.
- **xyaxes** `(x, y)`: `normvec(x)`; `y -= x (x·y)`; `normvec(y)`; `z = x × y`, `normvec(z)`; each must be ≥ mjEPS; `q = frame2quat(x, y, z)`.
- **zaxis**: `normvec(z)`; `q = z2quat(z)`.
- **euler** `(e0,e1,e2)` with `eulerseq` 3 chars: if degree convert each `e/180*π`; `q=(1,0,0,0)`; for i=0..2: `qrot = (cos(e_i/2), sin(e_i/2) on axis of seq[i])`; lowercase (x,y,z) = intrinsic/moving axes → `q = q * qrot`; uppercase (X,Y,Z) = extrinsic/fixed → `q = qrot * q`; (each `mulquat` normalizes). Finally `normvec(q)`. Invalid char → error.

Note conversion uses `θ/180.0*mjPI` (division first) for orientation, while joint ranges use `range *= mjPI/180.0` (multiplication by precomputed constant). These differ in the last ulp; replicate.

### 4.3 Angle unit conversion (compiler.degree == true)

| Quantity | Conversion | Where |
|---|---|---|
| axisangle angle, euler angles | `x/180.0*mjPI` | `ResolveOrientation` |
| hinge & ball `range` (only if `is_limited()`, and only nonzero entries) | `range[k] *= mjPI/180.0` | `mjCJoint::Compile` |
| hinge `ref`, `springref` | `*= mjPI/180.0` | `mjCJoint::Compile` |
| hinge & ball `springyield` | `*= mjPI/180.0` | `mjCJoint::Compile` |
| keyframe qpos | **not converted** (keyframe values are in model units, radians); `ScaleKeyframe` only applies frame-scaling gauges | — |
| camera `fovy`, light `cutoff` | stored in degrees (no conversion) | — |
| actuator ctrlrange of a position actuator on a hinge | **not converted** (given in radians even with degree=true); `inheritrange` copies the *compiled* joint range (radians), since joints compile in `CompileTree` before actuators (§8.6) | — |
| `mesh` / flexcomp orientation | via `ResolveOrientation` | — |

Slide-joint quantities are never converted. Free/ball `ref` is ignored (qpos0 from body pose / identity).

### 4.4 `fromto` (geoms and sites)

Only for capsule, cylinder, ellipsoid, box; error if `pos ≠ 0` also given.
```
v = from - to                      // note: from minus to
size[1] = normvec(v) / 2           // must be ≥ mjEPS
if ellipsoid or box: size[2] = size[1]; size[1] = size[0]
pos = (from + to)/2
quat = z2quat(v)                   // z-axis points from 'to' toward 'from'
```
(With `v = from − to`, the geom's +z points to `from`. For capsule/cylinder the sign is irrelevant physically, but quat values differ from using `to − from`; replicate.)

### 4.5 Frames (`mjCFrame::Compile`)

`pos,quat` from spec; `ResolveOrientation`; validate scale (finite, positive; anisotropic requires `scaling == GEOMETRY`; massscale ≥ 0); apply enclosing gauge to `pos`; if nested in a parent frame: compile parent and `frameaccumChild(parent.pos, parent.quat, pos, quat)`; normalize quat. Elements in a frame then do `frameaccumChild(frame.pos, frame.quat, elem.pos, elem.quat)` at the end of their own compile (geoms, sites, cameras: after fromto/orientation resolution; joints: rotate axis by frame quat and translate pos with identity quat; bodies: right after inertia processing; inertial element in its own frame `iframe`: `frameaccumChild(iframe.pos, iframe.quat, ipos, iquat)`).

### 4.6 Frame scaling — `mjCGauge`

A frame `scale[3]`, `scaling` (∈ {default/units, GEOMETRY}) and `massscale` define a gauge:
```
uniform = s0==s1==s2
shape   = uniform ? s0 : cbrt(s0 s1 s2)
inertia = massscale > 0 ? massscale : s0 s1 s2
length = shape, mass = inertia   if scaling != GEOMETRY  (else length=mass=1: forces unchanged)
map = diag(s)
Shape(a,b) = shape^a * inertia^b ;  Force(a,b) = length^a * mass^b
```
Gauges compose along the frame/body ancestry (`operator*`: products; maps multiplied); an element's gauge is its innermost frame's `ContentGauge` (or parent body's). `Rotated(q)` conjugates the map: `R^T M R`. Applied in each element's `ApplyGauge` (before anything else in `Compile`):
- body: `pos` scaled by `Point` (map), inertial by the inertial frame's gauge: `ipos` by Point; mass `*= Shape(0,1)`; inertia: uniform → `*= Shape(2,1)`; non-uniform → second moments `Σ = tr(I)/2·1 − I` mapped `A Σ A^T` (A = map in iquat frame), `I' = (tr(Σ')·1 − Σ')·Shape(0,1)`, result written as `fullinertia` (symmetrized), diagonal zeroed.
- geom: `pos`, `surfacevel` (rotated map), `size` via `Size()` (type-specific, errors if shape can't represent the scaling; tolerance `kAlignTol=0.0349`; sphere scaled by `cbrt(f0 f1 f2)`, capsule/cylinder radius by `sqrt(f0 f1)`, half-length by `f2`; plane: `f0, f1`, size[2] by shape), `fromto` points, `friction[1], friction[2], margin, gap, solimp[2]` `*= Shape(1)`, `mass *= Shape(0,1)`, `density *= Shape(-3,1)` (shell: `Shape(-2,1)`), `adhesion *= Force(1,1)`.
- joint (not free): pos; axis must be Aligned; slide coordinate factor `fq = Stretch(axis) = |map·axis|/|axis|`; `range, ref, springref, springyield, margin, solimp_limit[2], solimp_friction[2] *= fq`; `stiffness[n], damping[n] *= Force(a−(n+1)q, 1)` with `q = slide?1:0`, `a = 2−q`; `armature *= Force(a−q,1)`; `frictionloss, actfrcrange *= Force(a,1)`.
- actuator, pair, equality, tendon, flex: analogous dimensional rules (see `mjCActuator::ApplyGauge` — e.g. fixed gain with pos/vel input: `gainprm[0] *= Force(a−q,1)`, `ctrlrange *= fq`; positive `biasprm[2]` (damping ratio) is dimensionless and left untouched; negative scaled by `Force(a−q,1)`).
With the default (no scale) all gauges are identity and every `ApplyGauge` returns immediately.

---

## 5. Body inertia

### 5.1 `mjCBody::Compile` (exact order)

1. `CopyFromSpec`; `ApplyGauge`; compile own frames; resize userdata.
2. `normvec(quat)`, `normvec(iquat)`.
3. Children weldid: `child.weldid = (child has joints || child.mocap) ? child.id : this.weldid` (world weldid = 0).
4. `ResolveOrientation(quat, alt)` if alt given.
5. Inertia orientation: errors if `fullinertia` given together with `ialt` or with nonzero diagonal `inertia`.
   If `fullinertia` defined: rotate the tensor from the (given `iquat`) frame into body frame: `F_body = R(iquat) F R(iquat)^T`, then `mjuu_fullInertia(iquat, inertia, F_body)` (eig3, §5.4) → new principal `iquat` and diagonal `inertia`. Error if smallest eigenvalue `< mjEPS`.
   If `ialt` given: `ResolveOrientation(iquat, ialt)`.
6. If inertial lives in a frame: `frameaccumChild(iframe.pos, iframe.quat, ipos, iquat)`.
7. Compile geoms, each with `inferinertia = id>0 && (!explicitinertial || inertiafromgeom==TRUE) && group ∈ inertiagrouprange`.
8. `inertia_inferred = id>0 && (inertiafromgeom == TRUE || (!defined(ipos[0]) && inertiafromgeom == AUTO))`; if so `InertiaFromGeom()` (§5.2).
9. If `ipos` still undefined (no inertial, no inference): `ipos = pos`, `iquat = quat` (**sic**: the body's own pose, not identity — then sameframe tests compare against zero; for a body at the origin with identity quat this is the null pose. A body with non-zero `pos` and no inertial and no inferable geoms gets ipos = pos, which is nonsensical but only reachable for massless bodies, which `CheckBodyMassInertia` rejects if they have joints).
10. Mass/inertia correction (only id>0):
    ```
    mass       = max(mass, boundmass)
    inertia[k] = max(inertia[k], boundinertia)
    error if any negative
    if inertia violates triangle inequality (I0+I1 < I2 or I0+I2 < I1 or I1+I2 < I0) — exact, no tolerance:
        if balanceinertia: I0=I1=I2=(I0+I1+I2)/3   else error
    ```
11. Save `ipos_compiled_, iquat_compiled_, mass_compiled_, inertia_compiled_`.
12. Apply body frame: `frameaccumChild(frame.pos, frame.quat, pos, quat)`.
13. **Free-joint alignment** if: exactly one joint, it is free, body has no child bodies, and (`joint.align == TRUE` or (`AUTO` and `compiler.alignfree`)):
    ```
    frameaccum(pos, quat, ipos, iquat)           // body frame := inertial frame
    (ipos_inv, iquat_inv) = frameinvert(ipos, iquat)
    ipos = 0; iquat = identity
    for geoms: frameaccumChild(ipos_inv, iquat_inv, geom.pos, geom.quat)
    ```
    (sites, cameras, lights, frames re-expressed after they compile; lights: pos with identity quat, `dir = rot(iquat_inv) dir`).
14. `ComputeBVH()` (§7.4).
15. Compile joints (dofnum = Σ); error if `dofnum > 6`; error if ball followed by hinge/ball; mocap body must be a fixed child of world.
16. `xpos0 = rot(parent.xquat0) pos + parent.xpos0`, `xquat0 = parent.xquat0 * quat` (compile-time global pose in qpos0, used by flex/composite).
17. Compile sites, cameras, lights; plugin checks; alignment phase 2.

### 5.2 `InertiaFromGeom`

```
sel = [g in geoms if inertiagrouprange[0] <= g.group <= inertiagrouprange[1] and g.mass_ > mjEPS]
if |sel| == 1: ipos = g.pos; iquat = g.quat; mass = g.mass_; inertia = g.inertia     (copy, no eig)
elif |sel| > 1:
   mass = Σ m_g ; com = Σ m_g pos_g   (accumulated in geom order)
   error if mass < mjEPS
   ipos = com / mass
   toti = Σ_g [ globalinertia(I_g, quat_g) + offcenter(m_g, pos_g - ipos) ]   (6-vector, geom order, (inert0+inert1) added as toti[j] + inert0[j] + inert1[j])
   fullinertia = toti
   (iquat, inertia) = mjuu_fullInertia(toti)       // eig3 with default reltol 4e-15
else: nothing (mass/inertia stay as spec: 0 unless given)
```
`g.pos, g.quat` here are the compiled geom frame (after fromto, mesh frame accumulation and the geom's own frame), in body coordinates, **before** free-joint alignment.

### 5.3 Geom mass/inertia (`mjCGeom::Compile`, `GetVolume`, `SetInertia`)

Only when `inferinertia`:
```
if defined(mass):
    if mass == 0: mass_ = 0; density = 0
    elif GetVolume() > mjEPS: mass_ = mass; density = mass / GetVolume(); SetInertia()
    (else: mass_ stays 0, inertia stays 0)
else:
    if density == 0: mass_ = 0
    else: mass_ = density * GetVolume(); SetInertia()
error if mass_ < 0 or any inertia < 0 or density < 0
```
`GetVolume` (V = volume, or surface area for `typeinertia == SHELL`); r = size[0], h = 2·size[1]:

| type | VOLUME | SHELL |
|---|---|---|
| sphere | `4π r³/3` | `4π r²` |
| capsule | `π(r² h + 4r³/3)` | `4π r² + 2π r h` |
| cylinder | `π r² h` | `2π r² + 2π r h` |
| ellipsoid | `4π a b c / 3` | Thomsen: `4π ((( (ab)^p + (bc)^p + (ca)^p ) / 3)^(1/p))`, p=1.6075 |
| box, hfield | `8 a b c` | `8(ab + bc + ca)` |
| mesh/SDF | `mesh->GetVolumeRef()` (= `volume_` or `surface_` per mesh inertia type) | idem |
| plane, other | 0 | 0 |

`SetInertia` (diagonal, geom frame; m = mass_):
- mesh/SDF: inertia box `b = mesh->boxsz_`: `I = m·(b1²+b2², b0²+b2², b0²+b1²)/3`.
- sphere: VOLUME `2 m r²/5`; SHELL `2 m r²/3`.
- capsule VOLUME: `ms = m·4r/(4r+3h)`, `mc = m − ms`; `Ixy = mc(3r²+h²)/12`, `Iz = mc r²/2`; `Is = 2 ms r²/5`; `Ixy += Is + ms·h(3r+2h)/8`; `Iz += Is`.
- capsule SHELL: `As = 4πr²`, `Ac = 2πrh`; `ms = m·As/(As+Ac)`, `mc = m−ms`; `Ixy = mc(6r²+h²)/12`, `Iz = mc r²`; `Is = 2ms r²/3`; `hs_com = r/2`, `hs_pos = size[1] + r/2`; `Ixy += Is + ms(hs_pos² − hs_com²)`; `Iz += Is`.
- cylinder VOLUME: `Ixy = m(3r²+h²)/12`, `Iz = m r²/2`.
- cylinder SHELL: `Ad = πr²`, `Ac = 2πrh`, `md = m·Ad/(2Ad+Ac)`, `mc = m − 2md`; `Ixy = mc(6r²+h²)/12`, `Iz = mc r²`; `Idx = md r²/4 + md·size[1]²`, `Idz = md r²/2`; `Ixy += 2Idx`, `Iz += 2Idz`.
- ellipsoid VOLUME: `I = m(b²+c², a²+c², a²+b²)/5`.
- ellipsoid SHELL: eps = 1e-6; `Va = 4πabc/3`, `Vb = 4π(a+ε)(b+ε)(c+ε)/3`, `ρ = m/(Vb−Va)`; `I = ρVb·((b+ε)²+(c+ε)², …)/5 − ρVa·(b²+c², …)/5`.
- box/hfield VOLUME: `I = m(b²+c², a²+c², a²+b²)/3` (a,b,c half-sizes).
- box SHELL: `l = 2·size`; `A0 = lx ly`, `A1 = ly lz`, `A2 = lz lx`, `At = 2(A0+A1+A2)`; `mk = m Ak/At`;
  `Ix0 = m0 ly²/12, Iy0 = m0 lx²/12, Iz0 = m0(lx²+ly²)/12`; `Ix1 = m1(ly²+lz²)/12, Iy1 = m1 lz²/12, Iz1 = m1 ly²/12`; `Ix2 = m2 lz²/12, Iy2 = m2(lx²+lz²)/12, Iz2 = m2 lx²/12`;
  `Ix = 2(m0 s2² + m2 s1² + Ix0+Ix1+Ix2)`, `Iy = 2(m0 s2² + m1 s0² + Iy0+Iy1+Iy2)`, `Iz = 2(m1 s0² + m2 s1² + Iz0+Iz1+Iz2)` with `sk = size[k]`.
- other: 0.
- Hfield geoms use box formulas with the geom `size` *after* the hfield override in §7.1 (`size = (hx, hy, 0.25·elev + 0.5·base)`) — note `SetInertia` runs after the size override, so hfield volume = `8·hx·hy·(0.25 h_elev + 0.5 h_base)`. (Hfields are in static bodies in practice.)
- A mesh geom with `typeinertia == SHELL` is an error ("inertia should be specified in the mesh asset").

### 5.4 `mjuu_eig3` — Jacobi with quaternion accumulation

```
scale = max_ij |mat_ij|;   tol = scale * max(reltol, 4e-15)
quat = (1,0,0,0)
for iter in 0..499:
    E = quat2mat(quat);  D = E^T * mat * E   (tmp2 = transpose(E); tmp = tmp2*mat; D = tmp*E)
    eigval = diag(D)
    pick largest off-diagonal (ties: strict > comparisons):
        if |D01| > |D02| and |D01| > |D12|: (r,c,rot) = (0,1,2)
        elif |D02| > |D12|:                  (0,2,1)
        else:                                (1,2,0)
    if |D[r][c]| <= tol: break
    tau = (D[c][c] - D[r][r]) / (2 D[r][c])
    t = tau >= 0 ? 1/(tau + sqrt(1+tau²)) : -1/(-tau + sqrt(1+tau²))
    h = t / (1 + sqrt(1+t²))                 // tan(angle/2)
    dq = (1/sqrt(1+h²), 0,0,0);  dq[rot+1] = (rot==1 ? h : -h) * dq[0]
    quat = normalize(quat * dq)              // mulquat normalizes, then normvec again
sort eigenvalues decreasing (bubble passes j = 0,1,2 with lead index j1 = j%2 → 0,1,0):
    eps = scale * max(reltol, 1e-12)
    if eigval[j1] + eps < eigval[j1+1]:
        swap; quat = normalize(quat * (cos45, unit axis (j1+2)%3 * sin45))   with cos45 = 0.707106781186548
eigvec = quat2mat(quat)
```
Output: `eigval` (descending, I0 ≥ I1 ≥ I2 up to eps), `quat` whose rotation matrix columns are the principal axes. `mjuu_fullInertia(quat, inertia, full6)` builds the symmetric 3×3 (`[xx, xy, xz; xy, yy, yz; xz, yz, zz]`), calls eig3 (reltol 4e-15), errors if `eigval[2] < mjEPS`.

**The body inertia is therefore sorted descending** whenever it came through eig3 (multiple geoms, fullinertia, fused bodies), with the corresponding `iquat`. A single geom copies the geom's diagonal unsorted.

### 5.5 `AccumulateInertia` (fusestatic)

Combine `this` (mass, ipos, iquat, inertia) with another body expressed in this body's frame: `other_ipos/iquat = frameaccum(other.pos, other.quat, other.ipos, other.iquat)`.
```
M = m0 + m1;  com = (m0 ipos0 + m1 ipos1)   [accumulated]
if M < mjMINVAL: mass=0, inertia=0, ipos=0, iquat=identity
else: ipos = com/M;
      toti = Σ_j globalinertia(I_j, q_j) + offcenter(m_j, ipos_j − ipos)
      (iquat, inertia) = fullInertia(toti)     (error message "error in fusing static body inertias")
```

### 5.6 `CheckBodyMassInertia` (moving bodies only)

A body with joints must have `mass ≥ mjMINVAL` and all three `inertia ≥ mjMINVAL`, **or** have (recursively) a joint-less child body satisfying this check. Otherwise compile error "mass and inertia of moving bodies must be larger than mjMINVAL".

### 5.7 sameframe flags

Recomputed in `mj_setConst` (FIXED group, `setSameframe`, tolerance `1e-6`, quaternion double cover handled):
- `body_sameframe[i]` (i ≥ 1; world set to BODY in CopyTree): `BODY` if ipos≈0 and iquat≈±identity; `BODYROT` if only iquat≈±identity; else `NONE`.
- `geom_sameframe` / `site_sameframe`: `BODY` if pos≈0 & quat≈±id; `BODYROT` if quat≈±id; `INERTIA` if pos≈ipos & quat≈±iquat; `INERTIAROT` if quat≈±iquat; else `NONE`. Comparison is componentwise `|a−b| < 1e-6`.
- Enum values: `NONE=0, BODY=1, INERTIA=2, BODYROT=3, INERTIAROT=4`.
- Error if a body with `body_simple > 0` no longer has `sameframe == BODY`.

---

## 6. Mesh processing (`user_mesh.cc`)

All mesh processing is in double (`dvert`), final vertices narrowed to **float32** (`mesh_vert`). The mesh asset is per *instance* (scale map); identical meshes with different frame-scales are distinct model meshes.

### 6.1 Loading and vertex dedup

- `CopyFromSpec`: if `InstanceDiagonal()` (instance map is diagonal) the instance scale is multiplied into `scale[k] *= instancemap[4k]`; otherwise the full map is applied later in `ApplyTransformations`. Resets `aamm_ = (+1e10 ×3, −1e10 ×3)`, clears graph/BVH/octree/centers/halfedges. `ProcessVertices(spec_vert_, remove_repeated=false)`.
- `ProcessVertices(vert, remove_repeated)`: checks multiple-of-3 and finiteness. If `!remove_repeated` (the default for user vertices and decoder output in `TryCompile`): float → double copy, **no dedup**. If `remove_repeated` (only for flexcomp's local mesh loading): hash map keyed on exact float triples; if no repeats, plain copy; else faces remapped to first-seen index (index assignment order = first appearance); the vertex array is filled by iterating the unordered map (order of writes irrelevant since each index is written once).
- The STL decoder plugin itself deduplicates (`std::map<Vec3Key,int>`, ids in first-appearance order, ≤200000 faces). OBJ/MSH decoders provide vertices as in file.
- `CheckInitialMesh`: ≥ 4 vertices (`vert_.size() >= 12`), sizes multiples of 3/2, face indices in range.
- Plugin SDF meshes (`LoadSDF`): marching cubes from the plugin's SDF; sets `needreorient_ = false` (no re-centering/rotation; `pos_ = 0`, `quat_ = identity`), cannot be scaled.

### 6.2 `mjCMesh::Process()` (order matters)

```
dvert = double(vert_)
1. halfedges (if empty): for each face with triangle area > sqrt(mjMINVAL): push (v0,v1),(v1,v2),(v2,v0)
   stable_sort; adjacent duplicate => inconsistent orientation: error only if inertia == EXACT
2. if needhull_ or no faces: MakeGraph(dvert)    (qhull convex hull, §6.5)
   if no faces: CopyGraph()  (faces := hull faces)
3. if no normals: MakeNormal(dvert)              (§6.7)
   facenormal size check; facetexcoord := face if missing and texcoords exist
   facenormal (if empty) := normal_per_vertex * face index
4. if graph: MakePolygons(dvert) else polygon_map_.resize(nvert)
5. if scale[0]*scale[1]*scale[2] <= 0 (left-handed): swap face[i+1],face[i+2] (and facenormal, facetexcoord,
   hull faces, reverse polygons except first vertex)
6. ApplyTransformations(dvert)                   (§6.3)
7. facecen = ComputeFaceCentroid(dvert); error if total area < mjMINVAL
8. CoM & volume:
     SHELL:  surface_ = ComputeSurfaceArea(CoM, facecen);  error if < mjMINVAL
     else:   volume_  = ComputeVolume(CoM, facecen);  error if < mjMINVAL ("negative (misoriented)" if <0)
9. total = ComputeInertia(inert, CoM, dvert);  SHELL ? surface_ = total : volume_ = total
10. full = [[i0,i3,i4],[i3,i1,i5],[i4,i5,i2]];  eig3(eigval, eigvec, quattmp, full, reltol=1e-7)
    error if eigval[2] <= 0
    error if eigval[a]+eigval[b] < eigval[c]*(1-1e-6) - 1e-9 for any permutation
11. inertia box (V = GetVolumeRef()):
      boxsz[0] = 0.5*sqrt(6*(e1+e2-e0)/V); boxsz[1] = 0.5*sqrt(6*(e0+e2-e1)/V); boxsz[2] = 0.5*sqrt(6*(e0+e1-e2)/V)
12. if !needreorient_: CoM = 0, quattmp = identity
13. dvert -= CoM;  Rotate(quattmp, dvert)   -> dvert = R(quattmp)^T dvert, accumulates aamm_, rotates normals (float)
14. pos_ = CoM; quat_ = quattmp               (mesh_pos, mesh_quat)
15. TransformNodes(CoM, quattmp)              (volumetric nodes follow the same transform)
16. if no centers: MakeCenter(dvert)          (face circumcenters, used as BVH element positions)
17. MakePolygonNormals(dvert)
18. BVH over faces (§6.9)
19. if needsdf: octree + SDF coefficients (§6.10)
20. vert_ = float(dvert)
```
The inertia tensor computed is "unitless" (per unit density): mass enters via geom density, `SetInertia` uses the equivalent box `boxsz` (so geom inertia = `m (b1²+b2²)/3`… = exactly `ρ·eigval` up to roundoff: `m/V · e`).

### 6.3 `ApplyTransformations`

1. `refpos`: `v -= refpos` if non-zero.
2. `refquat` (if not identity): `q = normalize(refquat)`, `v = R(q)^T v` (mulvecmatT), normals likewise (float stored).
3. `scale` (if not all 1): `v *= scale` componentwise; normals `/= scale` (covectors).
4. Non-diagonal instance map M: `v = M v`; normals by cofactor matrix (inverse transpose up to det).
5. Normalize normals in float: `len = n·n` (float); if `len > mjMINVAL`: `n *= 1/sqrtf(len)` else `(0,0,1)`.

### 6.4 Volume, CoM, inertia — signed tetrahedra about the face centroid

`triangle(normal, center, v1, v2, v3)`: `center = (v1+v2+v3)/3`; `n = (v2−v1)×(v3−v1)`; `len = |n|`; if `len < mjMINVAL` return 0 (normal not normalized in that case); else `normal = n/len`; return `area = len/2`.

`ComputeFaceCentroid`: `facecen = Σ area_f·center_f / Σ area_f` (only normalized if total ≥ mjMINVAL). Faces used: all `face_`.

`ComputeVolume(CoM, facecen)` — faces: hull faces (`graph`) if `inertia == CONVEX`, else `face_`:
```
for each face: area, normal, center = triangle(...)
   vol = dot(center - facecen, normal) * area / 3            // signed pyramid with apex at facecen
   if LEGACY: vol = |vol|
   total += vol;  CoM += vol * (0.75*center + 0.25*facecen)
if total >= mjMINVAL: CoM /= total
```
`ComputeSurfaceArea` (SHELL): `total += area`, `CoM += area*(0.75 center + 0.25 facecen)` (note: same 3/4–1/4 weighting even for shell — replicate).

`ComputeInertia(inert, CoM)`: translate vertices by −CoM (into `vert_centered`), faces as above:
```
for each face (D,E,F centered):
   area, normal, center = triangle(D,E,F)
   vol = SHELL ? area : dot(center, normal)*area/3      // apex at the CoM (origin)
   if LEGACY: vol = |vol|
   total += vol
   C = SHELL ? 12 : 20
   for j, (a,b) in [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]:
      P[j] += vol/C * ( 2(Da Db + Ea Eb + Fa Fb) + Da Eb + Db Ea + Da Fb + Db Fa + Ea Fb + Eb Fa )
inert = (P1+P2, P0+P2, P0+P1, -P3, -P4, -P5)        // (xx, yy, zz, xy, xz, yz)
return total
```
This is the standard second-moment integral over a tetrahedron (origin, D, E, F): `∫ x_a x_b dV = V/20 (Σ_i 2 x_ia x_ib + Σ_{i≠j} x_ia x_jb)` (vertex at origin contributes zero); for shells `A/12` for triangle second moments. Mesh `inertia` types:
- **CONVEX** (0): hull faces for volume & inertia (hull needed regardless of collisions).
- **EXACT** (1): mesh faces, signed volumes; requires consistent orientation (error otherwise).
- **LEGACY** (2, the default): mesh faces, **absolute** pyramid volumes (wrong for non-convex meshes, kept for backward compatibility).
- **SHELL** (3): surface area weighting (thin shell, density per area).
(The old `exactmeshinertia` compiler flag is replaced by mesh `inertia="exact"`.)

The volume used for `GetVolumeRef` is the one from `ComputeInertia` (about CoM), not the facecen one — they agree analytically for closed meshes; for open meshes/legacy they can differ.

### 6.5 Convex hull (`MakeGraph`, qhull)

Skipped if `nvert < 4`. Degeneracy pre-checks (errors): all vertices colocated (dist ≤ mjMINVAL from v0); collinear (no `|e1 × e2| > mjMINVAL·|e1||e2|`); coplanar (all `|n·v − d| ≤ mjMINVAL·|e1|`).
qhull options: `"qhull Qt"`; if `maxhullvert > -1`: append `" Q9 TA<maxhullvert−4>"`. Calls `qh_qhull`, `qh_triangulate`, `qh_vertexneighbors`.

Graph layout (`mesh_graph`, int array, size `szgraph = 2 + 3·numvert + 6·numface`):
```
[0] numvert, [1] numface,
vert_edgeadr[numvert], vert_globalid[numvert],
edge_localid[numvert + 3*numface]   // per hull vertex: list of neighbor *local* ids, terminated by -1
face_globalid[3*numface]            // hull triangles in global vertex ids
```
Order: qhull's `FORALLvertices` order (implementation-defined, deterministic for a given qhull version and input). Neighbor list per vertex: iterate neighbor facets, append each other facet vertex if not already in the list. Face orientation: if `facet->toporient`, swap first two vertex slots (ind = {1,0,2}). Neighbor global ids are then mapped to local hull indices. The graph is used by GJK/EPA hill-climbing support search; `mesh_extrema` seeds it (§2.5).
**Reproducing hull vertex/face order requires qhull with identical options**; the set of hull faces is geometric but their order and the triangulation of coplanar facets (`Qt`) is qhull-specific. Collision results using hill-climbing should be order-independent up to ties.

`CopyGraph` (no user faces): faces = hull triangles.

### 6.6 Polygons (`MakePolygons`) — used by box/mesh-face collision & normals

Over hull faces only. Each face → key `(round(atan2(ny,nx)/0.01), round(acos(nz)/0.01))` of its unit normal (normal components `+0.0` to kill −0); if `|nz| > 1−1e-7`: key `(0, nz<0 ? round(π/0.01) : 0)`. Faces with identical keys are merged into a `MeshPolygon` (edge-cancellation by shared opposite half-edges, islands merged). Each connected island with ≥ 3 vertices becomes a polygon (vertex loop by following edges). Polygon normal initially from the *quantized* angles, then **recomputed** in `MakePolygonNormals` by Newell's method on the final (transformed) vertices: `n += ((y0−y1)(z0+z1), (z0−z1)(x0+x1), (x0−x1)(y0+y1))`, normalized; `(1,0,0)` if norm < mjEPS. Iteration order of the `unordered_map` determines polygon order (implementation-defined hash order!). `polygon_map[v]` = list of polygons containing v. `npolygonmax`, `nmeshdegmax` computed in setConst FIXED.

### 6.7 Normals (`MakeNormal`, if none given)

```
normal = 0 (per vertex); facenormal := vertex ids
for each face: nrm = (v1-v0)×(v2-v0); area = normvec(nrm)  (|cross|, i.e. 2×triangle area)
               normal[v] += nrm * area  for its 3 vertices
if !smoothnormal:
   for each face, each vertex: if dot(nrm, normalize(normal[v])) < 0.8: nremove[v] += nrm*area
   normal -= nremove
normalize in float: len = sqrtf(...); if len > mjMINVAL: /= len else (0,0,1)
```
Normals are visual/ray-only except where collision uses `mesh_normal` (not in core collisions).

### 6.8 Face circumcenters (`MakeCenter`)

`a = v0−v2`, `b = v1−v2`, `n = a×b`; `center = ((|a|² b − |b|² a) × n)/(2|n|²) + v2`. Degenerate faces produce inf/NaN (not guarded). Used as BVH element positions (median split).

### 6.9 BVH construction (`mjCBoundingVolumeHierarchy`)

Used for: mesh faces (frame: mesh frame, no rotation: `Quat()` null for mesh face elements so their AABBs are already in mesh frame), body geoms (frame: body inertial frame `ipos, iquat`), flex elements.

Elements with `contype == 0 && conaffinity == 0` are **dropped** (mesh faces and flex elements are added with (1,1)). For each kept element: `lpos = R(iquat)^T (pos − ipos)` (its center in the hierarchy frame).

`MakeBVH(elements, level)` recursive:
```
AAMM = (+DBL_MAX ×3, −DBL_MAX ×3)
for each element: its aabb (center c, half h) -> 8 corners (v&1 ? max : min on x, v&2 on y, v&4 on z)
    if element has a quat: corner = R(qinv)( R(elem.quat) corner + elem.pos − ipos )
    AAMM = min/max with corner
inflate: for k, if |AAMM_lo − AAMM_hi| < mjEPS(1e-14): lo -= mjEPS, hi += mjEPS
index = nbvh++ ; push child (-1,-1), nodeid -1, level
bvh_aabb[index] = ((hi+lo)/2, (hi−lo)/2)
if n == 1: leaf, nodeid = element id; return
axis = 0; edges = hi−lo;  if edges[1] >= edges[0] + mjEPS: axis=1;  if edges[2] >= edges[axis] + mjEPS: axis=2
compare(e1,e2) = |lpos1[axis]−lpos2[axis]| > mjEPS ? lpos1[axis] < lpos2[axis] : (pointer e1 < e2)
m = n/2;  std::nth_element(begin, begin+m, end, compare)
child0 = MakeBVH(begin, begin+m, level+1)   (if m > 0)
child1 = MakeBVH(begin+m, end, level+1)
warn if level > mjMAXTREEDEPTH (50)
```
Node numbering is DFS preorder (children after parents), so `bvh_child[node] > node`. **`std::nth_element` is unspecified beyond the partition property**: the exact split of elements into halves is deterministic for a given libstdc++/libc++ but the *membership* of the two halves is fully determined (m smallest by `compare` go left), so tree topology is reproducible if you implement "select the m smallest by compare (ties broken by element index)" — element pointer order = insertion order. Within a half, the order passed down affects only further nth_element calls, which again depend only on membership. Therefore any correct selection algorithm reproduces the same tree.

For body BVHs the compile-time AABBs are later **refit** by `mj_setConst::setBound` (§12.2) using `geom_aabb` from setConst; topology (child, nodeid, depth) is kept. Mesh BVH AABBs are not refit. Flex BVH aabbs are computed at runtime (`flex_bvh` in mjData).

### 6.10 Octree and SDF coefficients (mesh SDF geoms)

Built only if the mesh instance `needsdf` (used by an SDF-type geom), in the mesh frame (`ipos_=0, iquat_=identity` effectively; `Make` rotates by iquat inverse).
- `CreateOctree(aamm_)`: root box = mesh AABB inflated ×1.1 about its center. BFS subdivision (`MakeOctree`): a node is subdivided into 8 children if any triangle intersects its box (`boxTriangle`, Schwarz–Seidel test: bbox reject, plane–box overlap, 3×3 projected edge tests) and `level < max_depth` (`octree_maxdepth`, default 6). Child slot bit layout: bit0 = x upper half, bit1 = y, bit2 = z. Node vertex ids deduplicated by exact coordinate hash (`vert_map`).
- `BalanceOctree`: repeatedly subdivide leaves whose face neighbor (found via `FindNeighbor` = coarse neighbor + descent toward the node center) differs by > 1 level, until no change (2:1 balance).
- `MarkHangingNodes`: for leaves adjacent to a coarser neighbor, vertices on the shared face that are not vertices of the neighbor become hanging: edge-hanging (on 2 boundary planes, |Δ| < 1e-9) → 2 parents; face-hanging (1 plane) → 4 parents.
- `ComputeSdfCoeffs`: BFS over nodes; for each non-processed node vertex: if non-hanging, `coeff = signed distance` from the vertex (mapped back to mesh frame) to the mesh via the face BVH: closest triangle by recursive BVH search with box-distance pruning (`pointBoxDistSq ≥ best` prune; leaf → `pointTriDistSqWithUV` Voronoi-region closest point), sign from the **angle-weighted pseudonormal** of the closest feature (vertex normal weighted by corner angle `atan2(|e1×e2|, a·b)`, edge normal = sum of unit face normals, face normal for interior): `sign = dot(p − closest, normal) ≥ 0 ? +1 : −1`; hanging vertices: mean of parent coefficients (parents processed before children in BFS). Optional Laplacian smoothing (`smoothing_iterations_`, α = 0.2). `oct_coeff[8*node + j]` = coefficient at node vertex j (trilinear corner values).
- `oct_aabb` stored as center/half-size.

### 6.11 Mesh-fitted primitive geoms (`FitGeom`, geom with mesh but type ≠ mesh/SDF)

- default (`fitaabb=false`, inertia box `b = boxsz`): sphere `r = (b0+b1+b2)/3`; capsule `r = (b0+b1)/2`, `half = max(0, b2 − r/2)`; cylinder `r = (b0+b1)/2`, `half = b2`; ellipsoid/box `size = b`. `center = 0`.
- `fitaabb=true`: `center = aabb center`, `half = aabb half-size`: sphere `r = max(half)`; capsule/cylinder `r = max(hx,hy)`, `half = hz` (capsule: `half −= r`); ellipsoid/box `size = half`.
- then `size *= fitscale`.
- Geom pose: `meshpos = R(mesh_quat)·center + mesh_pos`; `frameaccum(geom.pos, geom.quat, meshpos, mesh_quat)` — i.e. geom frame absorbs the mesh's CoM/principal frame. The mesh reference is then dropped (`dataid = −1`).
- For true mesh geoms the same `frameaccum` with `center = 0` is applied, and `surfacevel` is re-expressed (linear part += ω × meshpos, then both rotated by `conj(mesh_quat)`).
- Mesh/SDF geom `size` := `max(|aamm_lo|, |aamm_hi|)` per axis (in the re-centered mesh frame).

### 6.12 Height fields (`mjCHField::Compile`)

- `size[4]` = (radius_x, radius_y, elevation_z, base_z), all must be > 0.
- Data from user (`nrow*ncol` floats), custom binary file (`int nrow, int ncol, float data[nrow*ncol]`, size must be exactly `8 + 4·nrow·ncol`), or PNG (grey, rows reversed: `data[r*ncol + c] = image[c + (nrow−1−r)*ncol]`).
- Normalization (float32): `emin, emax` over data (init 1e10 / −1e10); `data -= emin`; if `emax − emin > mjEPS`: `data /= (emax − emin)`. Final data ∈ [0,1].
- Geom: `geom_size = (size[0], size[1], 0.25*size[2] + 0.5*size[3])` (also re-imposed in setConst).

---

## 7. Geoms, sites, cameras — compile-time geometry

### 7.1 `mjCGeom::Compile` order

1. `CopyFromSpec`, `ApplyGauge`, userdata size.
2. Checks: type valid; `condim ∈ {1,3,4,6}`; mesh/SDF needs mesh; hfield type ⇔ hfield ref; plane only in static bodies (weld root world, or mocap weld root).
3. `normvec(quat)`.
4. `fromto` (§4.4) or `ResolveOrientation(quat, alt)`.
5. Mesh handling (§6.11).
6. `checksize`: plane requires `size[2] > 0`; others require the first `mjGEOMINFO[type]` sizes > 0.
7. Size overrides: hfield `size = (hx, hy, 0.25·elev + 0.5·base)`; mesh/SDF `size = max(|aamm lo|, |aamm hi|)` per axis. NaN size → error.
8. `ComputeAABB` (§7.2).
9. Mass/inertia if `inferinertia` (§5.3).
10. Fluid: `fluid[0..11] = 0`; if `fluid_ellipsoid > 0`: `fluid[0] = fluid_ellipsoid`, `fluid[1..5] = fluid_coefs` (blunt drag, slender drag, angular drag, Kutta lift, Magnus lift). `fluid[6..11]` (virtual mass/inertia) computed in setConst (§12.3).
11. Plugin checks (SDF capability).
12. Geom frame: `frameaccumChild(frame.pos, frame.quat, pos, quat)`.

### 7.2 `geom_aabb` (compile) — center/half-size in geom frame

| type | aamm (min, max) |
|---|---|
| hfield | `(−hx, −hy, −base, hx, hy, elev)` |
| sphere | `±(r, r, r)` |
| capsule | `±(r, r, r + half)` |
| cylinder | `±(r, r, half)` |
| mesh/SDF | mesh `aamm_` (CoM/principal frame) |
| plane | `(−MAXVAL, −MAXVAL, −MAXVAL, MAXVAL, MAXVAL, 0)` |
| default (box, ellipsoid) | `±size` |

`aabb = ((max+min)/2, (max−min)/2)`. **Recomputed** for non-mesh/SDF geoms in setConst (§12.2); mesh/SDF retain the compiler's.

### 7.3 `geom_rbound`

- Compiler (`GetRBound`): only mesh/SDF: `|(max(|lo_x|,|hi_x|), max(|lo_y|,|hi_y|), max(|lo_z|,|hi_z|))|` (norm of the corner of the symmetric box; ≥ true radius). Others 0.
- setConst (`geomBound`) for others: sphere `r`; capsule `r + half`; cylinder `sqrt(r² + half²)`; ellipsoid `max(size)`; box `|size|`; hfield `sqrt(hx² + hy² + max(elev², base²))`; plane `0`; default 0.

### 7.4 Body BVH (`mjCBody::ComputeBVH`)

Hierarchy frame = body inertial frame (`ipos, iquat` at this point: after alignment, i.e. identity for aligned free bodies). One leaf per geom with `contype || conaffinity` (non-colliding geoms excluded), element pos/quat = geom pose in body frame, aabb = `geom_aabb`. Built per §6.9. Bodies without colliding geoms have `body_bvhnum = 0`, `bvhadr = −1`. Runtime broadphase/midphase uses `bvh_aabb` (refit by setConst) with `body_margin`.

### 7.5 Sites

Like geoms: types except hfield/plane; mesh sites only with type mesh; fromto; `ResolveOrientation`; mesh site size from mesh aabb; frame accumulation; **normalize quat after** frame accumulation; `checksize`.

### 7.6 Cameras

`ResolveOrientation`; frame accumulation; normalize; resolve target body; `fovy < 180` (degrees). If `sensorsize > 0` (both): `pixel_density = resolution/sensorsize` (float), `intrinsic = (fx, fy, cx, cy)` with pixel values overriding length values (`focal_pixel/density`), and **`fovy` recomputed** `= atan2(sensorsize_y/2, intrinsic[1]) · 360/π`. Else `intrinsic[0] = intrinsic[1] = visual.map.znear`. Camera values matter only for rendering/rangefinder sensors.

---

## 8. Joints, tendons, actuators, pairs, equalities — compile-time values

### 8.1 Joints (`mjCJoint::Compile`)

- `springdamper` both > 0 if either nonzero; `springyield ≥ 0`.
- Free: `limited = FALSE`; else if `limited == AUTO`: `checklimited` — with `autolimits == false` and a nonzero range, error; with autolimits, `is_limited() = (limited == TRUE) || (limited == AUTO && range[0] < range[1])`.
- If limited: `range[0] < range[1]` unless ball; ball requires `range[0] == 0`. Degree conversion of hinge/ball ranges (nonzero entries only).
- Free/ball: `actfrclimited = FALSE`; else AUTO resolution like limited; actfrcrange must satisfy `lo < hi` if limited.
- Axis: free/ball forced `(0,0,1)`; else rotated by frame quat; `normvec(axis)` must be ≥ mjEPS (on squared norm) — **axis is normalized**.
- Free: `pos = 0`; else translated by frame.
- Degree conversion of hinge `ref`, `springref`; hinge/ball `springyield`.

### 8.2 Contact pairs (`mjCPair::Compile`) — explicit pair parameter resolution

Geoms resolved; **swap so that `geom1->body->id ≤ geom2->body->id`** (strict `>` triggers swap; equal bodies keep authoring order). `signature = (body1 << 16) + body2`.

Undefined pair attributes (NaN) are filled from the geoms:
- `margin = max(g1.margin, g2.margin)`; `gap = max(g1.gap, g2.gap)` (if undefined).
- If priorities differ, take from the higher-priority geom `pgh`: `condim` (if < 0 i.e. unset), friction `(f0, f0, f1, f2, f2)`, solref, solimp.
- If equal priority:
  - `condim = max`; friction: `(max f0, max f0, max f1, max f2, max f2)`.
  - mix: `s1, s2 = solmix`; if both ≥ mjEPS: `mix = s1/(s1+s2)`; both < mjEPS: 0.5; only s1 < mjEPS: 0; else 1.
  - solref: if (the pair's own, undefined → NaN, so this checks the default?) — code: `if (!defined(solref[0])) { if (solref[0] > 0) mix-average else min }`. Since `solref[0]` is NaN inside this branch, `NaN > 0` is false → **always elementwise `min(g1.solref, g2.solref)`** for undefined pair solref with equal priority. (Matches runtime contact mixing only for "direct" solref; replicate the literal behavior.)
  - solimp: `mix·g1 + (1−mix)·g2`.
The geom-geom (non-pair) contact parameter mixing happens at runtime in collision (`mj_contactParam`), not here.

### 8.3 Excludes

Bodies resolved, swapped so body1 ≤ body2, `signature = (body1<<16) + body2`; model ids sorted by signature.

### 8.4 Equalities (`mjCEquality::Compile`)

- Resolve object types (weld/connect: site or body; joint; tendon; flex/flexstrain); missing body2 → world (0); objects must differ; joint equalities require hinge/slide.
- Connect between sites: `data` zeroed entirely.
- Weld: `data[11] = (objtype == BODY) && data[6..9] all zero` (flag: compute relative pose in setConst); if body weld and not flagged: normalize `data[6..9]` (if norm² < mjEPS → identity).
- Connect between bodies: `data[0..2]` = anchor in body1 frame (spec); `data[3..5]` computed in setConst.
- Weld between bodies: `data[0..2]` anchor in body2 frame; `data[3..5]` (anchor in body1 frame) and `data[6..9]` (relative quat) are computed in setConst **only if flagged** (`data[11] != 0`, i.e. no relpose given); otherwise the spec's `relpose` (`data[3..9]`) is kept; `data[10]` torquescale.
- Flex equalities: flex must be non-rigid; no `young` stretch elasticity (dim>1 and elastic2d≠1) and no edgestiffness; flexstrain requires interpolated flex, index ranges checked.

### 8.5 Tendons and wraps

- Wraps: sphere/cylinder type determined from the geom's spec type; side site id stored (`wrap_prm = sideid`); pulley `prm` (divisor) ≥ 0; joint wrap `prm` = coefficient.
- Path rules: fixed tendons only joints; spatial: ≥2 objects, positive width, armature ≥ 0, no consecutive pulleys, path cannot end in pulley, site needs non-pulley neighbor, no repeated consecutive site, geoms bracketed by sites, geom wrapping incompatible with armature > 0.
- `limited`/`actfrclimited` AUTO resolution (`checklimited`), `range[0] < range[1]` if limited; actfrcrange must contain 0 (`lo ≤ 0 ≤ hi`).
- `springlength[0] ≤ springlength[1]`; `(-1,-1)` means "set to tendon length at `qpos_spring`" (setConst compile-only, §12.12). Note a single −1 is not special.

### 8.6 Actuators (`mjCActuator::Compile`) and shortcut semantics

Shortcuts (XML elements) set gain/bias/dyn parameters through `mjs_setTo*` (values in model units; damping-ratio is resolved in setConst):

| shortcut | gaintype/prm | biastype/prm | dyntype/prm | ctrlspec |
|---|---|---|---|---|
| motor | FIXED, gain0=1 | NONE | NONE | – |
| position(kp, kv \| dampratio, timeconst, inheritrange) | FIXED, gain0=kp | AFFINE, bias1=−kp, bias2 = −kv (kv given) or +dampratio (dampratio given) | timeconst>0: FILTEREXACT, dyn0=timeconst; else NONE | POS |
| intvelocity | as position | as position | INTEGRATOR | VEL |
| velocity(kv) | FIXED, gain0=kv | AFFINE, bias2=−kv | NONE | VEL |
| damper(kv) | AFFINE, gain0=0, gain2=−kv; ctrllimited=TRUE | – | – | – |
| cylinder(timeconst, bias, area, diameter) | FIXED, gain0 = area or π/4·d² (if diameter ≥ 0) | AFFINE, bias0..2=bias | FILTER, dyn0 = timeconst | PRESSURE |
| muscle | MUSCLE, gain0..8 = (range0=0.75, range1=1.05, force=−1, scale=200, lmin=0.5, lmax=1.6, vmax=1.5, fpmax=1.3, fvmax=1.2) overridden if given ≥ 0; biasprm = gainprm | MUSCLE | MUSCLE, dyn0=0.01, dyn1=0.04, dyn2=tausmooth | – |
| adhesion(gain) | FIXED, gain0=gain; ctrllimited=TRUE | – | – | – |
| pid(kp, kv\|dampratio, ki, imax, slewmax) | PID, gain0=ki | AFFINE, bias1=−kp, bias2 like position | PID if ki or slewmax nonzero (dyn0 = imax if ki, dyn1 = slewmax) | subset of pos/vel/ff (default pos\|vel) |
| orientation | SO3, gain0=kp | SO3, bias1=−kp, bias2 like position | NONE (or INTEGRATOR) | expmap (3) / quat (4) |

Compile resolution:
- `ResolveReferences`: `trnid[0]` target id; slider-crank `trnid[1]` = slider site, `cranklength > 0`; site transmission `trnid[1]` = refsite or −1.
- SO3: validation; `actdim=3` for integrator; `ctrlnum = 4` (quat) or 3; `outnum = 3`; `so3_ = true` (model trntype `mjTRN_SO3`).
- PID: `actdim = (dynprm[1] > 0) + (gainprm[0] > 0)` when dyntype PID; `ctrlnum` = popcount of pos/vel/ff.
- DC motor: input validation; `ctrlnum` = popcount over pos/vel/ff/voltage.
- `ApplyGauge` (frame scaling; identity by default).
- damping/armature only for joint/tendon transmissions.
- **inheritrange** (position-like `gaintype==FIXED && gain0 == −bias1` or PID, with AFFINE bias, `inheritrange > 0`): target range (joint hinge/slide, *compiled*, i.e. radians, or tendon) must be nonzero; `mean = (lo+hi)/2`, `radius = (hi−lo)/2 · inheritrange`; written into `ctrlrange` (dyntype NONE/FILTEREXACT/PID) or `actrange` (INTEGRATOR).
- `forcelimited/ctrllimited/actlimited` AUTO: `checklimited`; validation `lo < hi` if limited; actlimited requires dyntype ≠ NONE.
- `actdim`: errors for invalid combos; if `< 0` → `(dyntype != NONE && dyntype != DCMOTOR)`.
- DC motor requires `actearly`.
- Muscle checks: `range0 < range1`, `lmin < 1 < lmax`, positive scale/vmax/fpmax/fvmax.
- delay > 0 requires `nsample > 0`; `nsample ≤ 2^24`.
- Per-input ctrl ranges: broadcast `ctrlrange/ctrllimited` to all inputs; PID overrides the vel input range with `velrange` (limited iff `lo < hi`) and ff with `ffrange`.
- Model copy: `actuator_gear` and `actuator_lengthrange` replicated to each output row.

---

## 9. Defaults and class resolution

- Each `mjCDef` holds one prototype per element kind (`joint_, geom_, site_, camera_, light_, flex_, mesh_, texture_, material_, pair_, equality_, tendon_, actuator_`), each initialized by `mjs_default*`. A child class is a full copy of its parent (`CopyWithoutChildren` + `+=` recursion); attributes set in the child class overwrite the copy.
- Element creation with a class: `*this = def->Joint()` (full struct copy of the class prototype) before the element's own attributes are applied; the default class is "main". Values are thus resolved at **spec construction** (parse) time, not at compile time; compiling uses the element's own spec values. `childclass` on bodies selects the default for descendants (parser-level).
- Actuator shortcuts in XML start from the nearest class written with the same shortcut or `general`, else documented shortcut defaults (`mjXShortcutSource` / `mjXShortcutDefaults`), then apply attributes via `mjs_setTo*` — order matters for `kv` vs `dampratio` (both given → error).
- `mjCDef::Compile` only resizes default userdata arrays.
- Spec-level "undefined" values (NaN) persist into compile where they trigger inference: body `ipos` (inertia inference), geom `mass` (density used), geom `fromto`, pair attributes, stat values.

---

## 10. Flex / flexcomp / composite generation (values that affect simulation)

### 10.1 flexcomp → bodies (`mjCFlexcomp::Make`)

- Grid/box/mesh/gmsh point generation; mesh files loaded with **vertex dedup** (`remove_repeated=true`).
- Requires `mass ≥ mjMINVAL` and `inertiabox ≥ mjMINVAL`.
- Pinning: `pin` ids / ranges / grid coords / gridrange; radial dof pins point 0.
- Per-vertex body (dof ∈ {full, radial, 2d}) for unpinned used points:
  `bodymass = mass / npnt` (**npnt = total points, including pinned and unused**), `bodyinertia = bodymass · 2·inertiabox² / 3` (all three axes), `ipos = 0`, explicit inertial. Joints: full → 3 slides along x,y,z (pos 0); radial → 1 slide along `normalize(pos)`; 2d → slides x, y. Body named `"<name>_<i>"`. Vertex coordinates cleared (body-centered) unless... (`point = 0` if `!centered`).
- Pinned points and trilinear/quadratic flexes: vertices attached to the parent body.
- **Trilinear/quadratic (interpolated)** flexes: nodes on a `(cx·order+1)×(cy·order+1)×(cz·order+1)` grid spanning the point AABB `minmax`; pinned nodes attached to the parent body; each free node gets a body with 3 slides and mass: volume mode `NodeMassWeight(gi,gj,gk)`; shell trilinear 1.0; shell quadratic Simpson weights `(1/6, 2/3, 1/6)` per local index times cell multiplicities; then all node masses normalized so `Σ = mass` (`scale = mass/total`, applied to mass and inertia); `inertia = mass_node · 2·inertiabox²/3` (before scaling). Shell mode pins all interior nodes. Empty cells (volume mode) pin nodes lying only in empty cells.
- Equality creation: `equality=1` → one `mjEQ_FLEX`; `3` → one `mjEQ_FLEXSTRAIN` per non-empty cell (volume) or per boundary face element (shell).

### 10.2 `mjCFlex::Compile` (structural / constant parts)

- Validation: dim 1..3; elem nonempty and size multiple of dim+1; edgestiffness only dim 1; elastic2d requires thickness > 0 and Poisson ∈ [0, 0.5); SNH only non-interpolated dim-3 with discrete integrator; yield checks; node count matches cellcount for interpolated.
- `rigid` if all vertex bodies equal (or vertbody size 1); `centered` if all vertex (or node) offsets are zero.
- `vertxpos = body.xpos0 + R(body.xquat0)·vert` (compile-time world positions in qpos0).
- dim 3: reorient tetrahedra so `((v1−v0)×(v2−v0))·(v3−v0) ≤ 0` (swap elem[1],elem[2] if > 0).
- Edges: per element local edges (`eledge` tables), keyed by sorted vertex pair, global edge index by first appearance (element order, local edge order).
- dim 2: flap stencil per edge (`CreateFlapStencil`).
- Elasticity precomputation if `young > 0`: per-element stiffness (`ComputeStiffness<Stencil2D/3D>`, 21 or 24 coefficients per element; SNH), rod bending (`I = π r⁴/4` circle or `min(w h³, h w³)/12` box with `w,h = 2·sectionsize`), 2D plate bending (rigidity `E t³/(12(1−ν²))`, 17 values per edge flap). Details belong to the flex spec file.
- BVH over elements (`CreateBVH`): per element AABB of its vertices in world (qpos0) coordinates inflated by `radius`; dim 3 elements with `elemlayer ≥ activelayers` skipped. (Flex BVH aabbs are dynamic at runtime; only topology is stored.)
- Shells (`CreateShell`), cell emptiness and FCM quadrature for interpolated volume flexes.

### 10.3 Cable composite (`MakeCable`, values)

Vertices from curve shapes (`line: ix·size0/(count−1)`, `cos/sin: size1·cos/sin(π·ix·size2/(count−1))`), rotated by `quat`. Bodies chained; frame via `mjuu_updateFrame` (parallel transport: first frame from `binormal = normalize(t × t_next)`, `normal = normalize(b × t)`; later frames rotate the normal by the Darboux rotation `axis = tprev × t`, `angle = atan2(|tprev×t|, tprev·t)`); `body.quat = conj(prev_quat) * this_quat`, `body.pos = (length_prev, 0, 0)`. Geoms: capsule/cylinder `fromto = (0,0,0, length,0,0)`; box `pos = (length/2,0,0)`, `size[0] = length/2`. Joints: ball (first may be free if `initial="free"`, then damping/armature/frictionloss zeroed). Excludes between consecutive bodies.

---

## 11. Spec-restructuring operations

### 11.1 `FuseStatic` (`compiler.fusestatic`)

Candidate body: no joints, not mocap, `spec.fuse` true, no plugin, `sleep == AUTO`, and **not referenced** by name (tested by temporarily erasing its name and re-resolving references of cameras, lights, skins, flexes, pairs, excludes, equalities, tendons, actuators, tuples, sensors; force/torque sensors on its sites also prevent fusing). Candidates collected in tree order before any fusing; then for each:
- `fusemass = parent is not world && body.mass ≥ mjMINVAL`.
- skip if `fusemass && gravcomp differs`.
- skip in fluid (`density ≠ 0 || viscosity ≠ 0`) if parent not world and (fusemass or any body geom has `fluid_ellipsoid > 0`).
- `reinfers` = both parent and body infer inertia from geoms, neither was adjusted by bound/balance, same `inertiagrouprange`; `keepinertia` = !fusemass && !reinfers && parent not world && parent infers && body has geoms; skip if `(fusemass || keepinertia) && !reinfers && parent.inertiafromgeom == TRUE`.
- `fusemass`: `parent.AccumulateInertia(body)` (§5.5, using the bodies' compiled values), `inertia_adjusted |= body's`; if `!reinfers` the parent adopts the result as explicit inertial (written back to spec in spec units).
- Children of the body: `changeframe(child.pos, child.quat, body.pos, body.quat)` = `frameaccum(body.pos, body.quat, child.pos, child.quat)` into child.
- `body->ToFrame()`: the body becomes a frame in its parent (geoms, sites, etc. move to the parent in place of the body; children take its position among the parent's children via `std::rotate`). Hence **element order changes**: after fusing, the parent's geom list contains its own geoms followed by the fused body's geoms (frame contents appended), and body ids shift.
- Then the model is recompiled from the modified spec (so geom inertias are re-inferred where `reinfers`).

### 11.2 `DiscardVisual` (`compiler.discardvisual`)

Keeps geoms with `contype || conaffinity || fluid_ellipsoid > 0`, or referenced by pairs, tendon wraps, sensors, tuples; meshes used by kept geoms or any site or referenced; materials/textures only if referenced by sensors/tuples (then they remain as elements; all material references are cleared). Bodies inferring inertia from geoms that lose a counted geom with `mass_ > mjEPS` adopt their compiled inertial as explicit (error if `inertiafromgeom == TRUE`). The kept geom order is preserved; ids are recompacted.

---

## 12. `engine_setconst.c` — `mj_setConst`, `mj_setCompiledConst`

### 12.1 Entry points and groups

```c
#define mjSCGROUP_FIXED  1   // independent of configuration
#define mjSCGROUP_KIN0   2   // kinematics in qpos0
#define mjSCGROUP_INERT0 4   // inertia matrix in qpos0
#define mjSCGROUP_SPRING 8   // kinematics in qpos_spring
```
```
setConst(m, d, compiled):
   mj_setConstGroups(m, d, FIXED | KIN0 | INERT0)
   if compiled: setDampRatio(m, d); setStat(m, d)          // still at qpos0 state (kinematics0, M factored)
   mj_setConstGroups(m, d, SPRING)
   if compiled: setSpringLength(m, d)                       // at qpos_spring state
mj_setConst(m,d)         = setConst(m, d, 0)    // public; for runtime model edits
mj_setCompiledConst(m,d) = setConst(m, d, 1)    // compiler only
```
`mj_setConstGroups(m, d, groups)`:
```
if FIXED:
   npolygonmax = max mesh_polyvertnum;  nmeshdegmax = max mesh_polymapnum
   setSameframe(m)                  (§5.7); error if a simple body lost sameframe BODY
   setBound(m, d)                   (§12.2)
   setFluid(m)                      (§12.3)
   setFixed(m, d)                   (§12.4)
   makeTendonSparse(m); makeFlexSparse(m, d)     (§3.5)
save enableflags; enableflags &= ~mjENBL_SLEEP           // all bodies processed (not restored on error)
if KIN0|INERT0|SPRING: d->ncon = 0; mj_clearEfc(d)      // adhesion transmission averages over contacts
if nmocap and KIN0|INERT0|SPRING:
   save mocap_pos/quat; for each mocap body: mocap_pos = body_pos, mocap_quat = body_quat
if KIN0|INERT0:
   kinematics0(m, d)                (§12.5)
   if KIN0:   setKinematics0(m, d) (§12.6); setDofLength(m, d) (§12.7)
   if INERT0: setInertia0(m, d)    (§12.8)
if SPRING: setSpring(m, d)          (qpos := qpos_spring; kinematics, comPos, tendon, transmission)
restore mocap
if SPRING: setEfm0Factor(m, d) (§12.11); mj_effCholSetConst(m, d)
restore enableflags
if KIN0|INERT0|SPRING: tree_asleep[:] = -(1+mjMINAWAKE) = -11; mj_updateSleep(m, d)    // wake all
```
Note `d->qpos` is left at `qpos_spring` (or qpos0) after the call; the caller is responsible for the state of d. The compiler throws d away.

### 12.2 `setBound` — geom bounds, body masks/margins, BVH refit

1. For each geom: hfield geoms: `geom_size = (hx, hy, 0.25·elev + 0.5·base)`. If `geomBound` applies (not mesh/SDF): `geom_aabb = ((hi+lo)/2, (hi−lo)/2)`, `geom_rbound` per §7.3. (Primitive aabbs are symmetric so center = 0, except hfield z and plane.)
2. Per body: `body_contype = OR geom_contype`, `body_conaffinity = OR geom_conaffinity`, `body_margin = max(0, max_g (geom_margin + geom_gap))` over the body's geoms.
3. BVH refit for each body with `bvhadr ≥ 0`: nodes processed **children before parents** (j from num−1 down to 0, valid because children have larger indices). Leaf (nodeid = geom g ≥ 0): 8 corners of `geom_aabb` → `corner' = R(conj(iquat)) · (R(geom_quat)·corner + geom_pos − ipos)`, min/max accumulated (init ±mjMAXVAL). Internal node: union of the two children. Store `bvh_aabb = ((hi+lo)/2, (hi−lo)/2)` after inflating flat axes: if `|lo − hi| < 1e-14`: `lo −= 1e-14, hi += 1e-14`.
   Rotations here use `mju_rotVecQuat` (engine version, mjtNum).
4. Error if a geom with `contype || conaffinity` belongs to a body that has a BVH but is not a leaf of it ("recompile the model").

Differences from compile-time BVH AABB: compile time used `mjuu_*` (compiler) math and the compiler's geom aabb (identical for primitives); setConst uses mjMAXVAL init and engine rotations. The runtime uses the setConst values.

### 12.3 `setFluid` — ellipsoid model added mass

For geoms with `fluid[0] > 0`:
```
(dx, dy, dz) = mju_geomSemiAxes(size, type):
     sphere (r,r,r); capsule (r, r, half + r); cylinder (r, r, half); else (s0, s1, s2)
V = 4/3 π dx dy dz
kx = κ(dx, dy, dz); ky = κ(dy, dz, dx); kz = κ(dz, dx, dy)
yz = dy² − dz²;  zx = dz² − dx²;  xy = dx² − dy²
Ixfac = yz² |kz − ky| / max(1e-14, |2 yz + (dy²+dz²)(ky − kz)|)
Iyfac = zx² |kx − kz| / max(1e-14, |2 zx + (dz²+dx²)(kz − kx)|)
Izfac = xy² |ky − kx| / max(1e-14, |2 xy + (dx²+dy²)(kx − ky)|)
fluid[6]  = V kx / max(1e-14, 2 − kx)       // virtual mass x
fluid[7]  = V ky / max(1e-14, 2 − ky)
fluid[8]  = V kz / max(1e-14, 2 − kz)
fluid[9]  = V Ixfac / 5                     // virtual inertia
fluid[10] = V Iyfac / 5
fluid[11] = V Izfac / 5
```
`κ(dx,dy,dz) = dx dy dz ∫₀^∞ dλ / sqrt((dx²+λ)³ (dy²+λ)(dz²+λ))` evaluated with a fixed 15-point Gauss–Kronrod rule after `λ = x³/(1−x)²`, nondimensionalized by `scale = (dx³ dy dz)^0.4`:
```
kappa = Σ_i scale * d_i / [(1 + λ_i/dx²) * sqrt((1+λ_i/dx²)(1+λ_i/dy²)(1+λ_i/dz²))] * w_i ,  λ_i = scale * l_i
return kappa / dx²
```
Tables (exact literals):
```
w = {0.01146766, 0.03154605, 0.05239501, 0.07032663, 0.08450236, 0.09517529, 0.10221647, 0.10474107,
     0.10221647, 0.09517529, 0.08450236, 0.07032663, 0.05239501, 0.03154605, 0.01146766}
l = {7.865151709349917e-08, 1.7347976913907274e-05, 0.0003548008144506193, 0.002846636252924549,
     0.014094260903596077, 0.053063261727396636, 0.17041978741317773, 0.5, 1.4036301548686991,
     3.9353484827022642, 11.644841677041734, 39.53187807410903, 177.5711362220801,
     1429.4772912937397, 54087.416549217705}
d = {5.538677720489877e-05, 0.002080868285293228, 0.016514126520723166, 0.07261900344370877,
     0.23985243401862602, 0.6868318249020725, 1.8551129519182894, 5.0, 14.060031152313941,
     43.28941239611009, 156.58546376397112, 747.9826085305024, 5827.4042950027115,
     116754.0197944512, 25482945.327264845}
```
(computed in `double` regardless of mjtNum; `pow(...,0.4)`.) Mesh geoms use their `geom_size` (= aabb half extents).

### 12.4 `setFixed` — configuration-independent quantities

```
body_subtreemass[i] = body_mass[i];  for i = nbody-1..1: subtreemass[parent[i]] += subtreemass[i]
ngravcomp = #(body_gravcomp > 0);  flg_gravcomp = ngravcomp > 0
flg_surfacevel = any geom_surfacevel nonzero
flg_adhesion   = any geom_adhesion or pair_adhesion nonzero
jnt_actuatorid[:] = tendon_actuatorid[:] = -1
for actuator i with damping != 0 or dampingpoly nonzero or armature != 0:
   joint/jointinparent: jnt_actuatorid[trnid] = (== -1) ? i : -2      // -2: multiple actuators
   tendon:              tendon_actuatorid[trnid] = (== -1) ? i : -2
body_treeid[i] = body_dofnum[weldid] ? dof_treeid[body_dofadr[weldid]] : -1
tree_bodyadr/num: bodies 1.. with treeid != -1; new tree when treeid > current (trees appear in increasing order)
tree_dofadr/num: dofs in order, new tree when dof_treeid > current
tendon_treeid[2], tendon_treenum: distinct trees of wrap bodies in path order (first two stored);
   wrap body: joint→jnt_bodyid, site→site_bodyid, sphere/cylinder→geom_bodyid, pulley→none
AUTO sleep policy resolution:
   reset AUTO_NEVER/AUTO_ALLOWED -> AUTO
   actuated trees -> AUTO_NEVER (joint body; SO3: site or joint body + second site; site/slidercrank: site bodies; body; tendon: all wrap trees)
   tendons spanning >2 trees, or 2 trees with stiffness/damping (mj_tendonHasStiffness/Damping, damping includes attached actuator) -> AUTO_NEVER (error if user policy ALLOWED/INIT)
   flexes without edge equality and not "metric-coupled" (mj_effFlexPossible) -> all vertex/node trees AUTO_NEVER
   remaining AUTO -> AUTO_ALLOWED
```
`mj_actuatorArmature(m, JOINT|TENDON, id)`: −1 → 0; `id ≥ 0` → `actuator_armature[a] · gear[6·outadr]²`; −2 → sum over all joint(-in-parent)/tendon actuators targeting id of `armature · gear0²`. **Because jnt_actuatorid is set here, actuator armature contributes to M (and dof_M0) only after setFixed has run**; it is part of the inertia in all INERT0 quantities.

### 12.5 `kinematics0`

```
qpos = qpos0;  mj_kinematics; mj_comPos; mj_tendon; mj_flexSkip(m, d, 0); mj_transmission
```
(with mocap at body_pos/quat, no contacts, sleep disabled).

### 12.6 `setKinematics0`

- `flexedge_length0 = d->flexedge_length` (zeroed for interpolated flexes).
- `tendon_length0 = ten_length`; `actuator_length0[0..nout) = actuator_length`.
- Body equalities:
  - CONNECT: `pos = xpos[b1] + xmat[b1]·data[0..2]` (`mj_local2Global`); `data[3..5] = xmat[b2]^T (pos − xpos[b2])`.
  - WELD (only if `data[11] != 0`): `pos = xpos[b2] + xmat[b2]·data[0..2]`; `data[3..5] = xmat[b1]^T (pos − xpos[b1])`; `data[6..9] = conj(xquat[b1]) * xquat[b2]` (`mju_mulQuat`, not normalized).
  Unflagged welds keep their spec `data[3..9]` (relpose given explicitly); the whole weld branch is gated on `data[11]`. `data[11]` stays 1 in the model, so a runtime `mj_setConst` after changing `qpos0` recomputes the relative pose.
- Cameras: `(pos, mat) = local2Global(cam_pos, cam_quat, body)`; `cam_pos0 = pos − xpos[body]`; `cam_poscom0 = pos − subtree_com[target ≥ 0 ? target : body]`; `cam_mat0 = mat`.
- Lights: `pos = local2Global(light_pos)`; `dir = normalize(rot(xquat[body]) light_dir)`; `light_pos0`, `light_poscom0` as cameras; `light_dir0 = dir`.

### 12.7 `bodySize` and `setDofLength`

```
bodySize(body[]):
  body = 0
  for each joint i: b = jnt_bodyid
      body[b]        = max(body[b],        |xipos[b] - xanchor[i]|)
      body[parent b] = max(body[parent b], |xipos[parent b] - xanchor[i]|)
  body[0] = 0
  for bodies i>=1, geoms g of i with geom_rbound > 0:
      body[i] = max(body[i], rbound_g + |xipos[i] - geom_xpos[g]|)
  for flexes: interpolated: for all node pairs (v1, v2): body[b(v1)] = max(., |xpos[b(v1)] - xpos[b(v2)]|)
              else for edges: body[b1] = body[b2] = max(., flexedge_length0[e])
  body[i] = max(body[i], 1e-5) for i>=1
dof_length[i] = 1 for slide and free-translation dofs; = body[dof_bodyid] for hinge, ball, free-rotation (offset >= 3)
```
`dof_length` is used by sleep detection (`qvel < tol·dof_length`) and as a velocity scale.

### 12.8 `setInertia0` — dof_M0, invweights, acc0

**`mj_setM0`** (dof_M0 = diagonal of M at qpos0 using CRB):
```
crb = cinert (10 per body);  for i = nbody-1..1: if parent[i] > 0: crb[parent] += crb[i]
for dof i: buf = crb[dof_bodyid[i]] * cdof[i]  (mju_mulInertVec)
           dof_M0[i] = dof_armature[i] + mj_actuatorArmature(JOINT, dof_jntid[i]) + dot(cdof[i], buf)
```
`mju_mulInertVec(res, I10, v)` with `I10 = (Ixx, Iyy, Izz, Ixy, Ixz, Iyz, m·c_x, m·c_y, m·c_z, m)`:
```
res0 = i0 v0 + i3 v1 + i4 v2 − i8 v4 + i7 v5
res1 = i3 v0 + i1 v1 + i5 v2 + i8 v3 − i6 v5
res2 = i4 v0 + i5 v1 + i2 v2 − i7 v3 + i6 v4
res3 = i8 v1 − i7 v2 + i9 v3
res4 = i6 v2 − i8 v0 + i9 v4
res5 = i7 v0 − i6 v1 + i9 v5
```
`dof_M0` is used by `mj_crb` for **simple dofs** (diagonal-only rows: `M[adr] = dof_M0[i]` without recomputation) — so simple dofs' inertia is frozen at qpos0 values (valid because simple bodies' diagonal inertia is configuration-independent). Note dof_M0 excludes tendon armature (added later by `mj_tendonArmature`, which cannot touch simple bodies because tendon-armature bodies are demoted).

Then `mj_makeM` (`mj_crb` + `mj_tendonArmature`) and `mj_factorM` (sparse LDLᵀ, `qLD`, `qLDiagInv`).

**body_invweight0** (`[2·nbody]`, translational, rotational):
```
invweight0[0] = invweight0[1] = 0     (world)
for body i >= 1:
   if body_dofnum[weldid[i]] == 0:  (0, 0)                         // static / mocap / welded to world
   elif body_simple[i] == 2:        (1/max(mjMINVAL, mass), 0)       // error if mass == 0
   else:
      jac (6×nv) = [jacp; jacr] = mj_jacBodyCom(i)  (point xipos[i])
      tmp = M⁻¹ jac^T (mj_solveM, 6 rhs);  A = jac · tmp  (6×6)
      invweight0 = ((A00+A11+A22)/3, (A33+A44+A55)/3)
```
(If `nv == 0` A stays 0.)

**dof_invweight0** (per joint):
```
if body_simple[jnt_bodyid] == 2: dof_invweight0[dofadr] = 1/max(mjMINVAL, body_mass)     (slide-only simple body; one entry per joint)
else: dnum = 6 (free) | 3 (ball) | 1
      J = rows of identity at dofs [adr, adr+dnum);  A = J M⁻¹ J^T  (i.e. the dnum×dnum block of M⁻¹)
      free: dofs 0-2 = mean(A00,A11,A22), dofs 3-5 = mean(A33,A44,A55)   (indices into the dnum×dnum A stored row-major: A[0],A[7],A[14] / A[21],A[28],A[35])
      ball: all 3 = (A[0]+A[4]+A[8])/3
      else: A[0]
```

**flexedge_invweight0** (`nv > 0`; zeroed first; interpolated flexes skipped):
```
rigid edge: 0
both vertex bodies simple==2: (1/mass(b1) + 1/mass(b2)) / 2     (no MINVAL guard)
else: dense J_e from flexedge_J (d->flexedge_J values from mj_flexSkip/kinematics0) ; w = J_e M⁻¹ J_eᵀ
```
**tendon_invweight0**: `J_t` dense from `ten_J` (sparse), `w = J_t M⁻¹ J_tᵀ`.
**actuator_acc0** (per output row j < nout): `acc0 = ‖M⁻¹ moment_j‖₂` (Euclidean norm of the acceleration from unit actuator force). If `nv == 0`: tendon_invweight0 = actuator_acc0 = 0.

These invweights set the constraint regularization scale `R = (1−d)/d · invweight` at runtime (efc_diagApprox), and `actuator_acc0` scales muscle/derivative quantities; reproduction must match to the precision of the solve.

### 12.9 `setStat` (compile only, after INERT0 at qpos0)

```
xmin = (1e10,…), xmax = (−1e10,…)
for bodies i>=1: box with xpos[i] (r=0) and xipos[i] (r=0)
for joints: xanchor;  for sites: site_xpos
for geoms: r = geom_rbound if > 0
           elif plane: (size0 || size1) ? 0.1*max(size0,size1) : 0.01
           elif hfield: 0.1 * max(hsize0..3)
           else 0
           box with geom_xpos ± r
center = (xmin + xmax)/2
if xmax[0] > xmin[0]: extent = max(1e-5, max_k(xmax_k − xmin_k))      (else extent keeps its previous value: default 2)
if nbody > 1: meansize = Σ_{i>=1} body[i]/(nbody−1)   (bodySize, accumulated as body[i]/(nbody-1) per term)
extent = max(extent, 2*meansize)
if nbody > 1: meanmass = Σ_{i>=1} mass / (nbody−1)
if nv: meaninertia = Σ_i M[M_rowadr[i] + M_rownnz[i] − 1] / nv      (mean diagonal of M incl. armature & tendon armature)
```
Spec overrides (`<statistic>`) replace these after `LengthRange`. In the current engine `stat.meaninertia` is **not used by the solver** (deprecated in spec; affects only visualization `meanmass/meansize/extent`). It used to scale solver tolerance in older versions — check your target version.

### 12.10 `setDampRatio` (compile only)

For each actuator that is position-like (`gaintype == PID` or `gainprm[0] == −biasprm[1]`) with `biasprm[2] > 0` (a damping ratio):
```
w = 0
for each output row j of the actuator:  m_j = dense moment;  w += m_j · M⁻¹ m_j
w /= outnum
mass = (nv && w > mjMINVAL) ? 1/w : 0
kp = PID ? −biasprm[1] : gainprm[0]
biasprm[2] = −(dampratio · 2 · sqrt(kp · mass))
```
I.e. critical damping w.r.t. the reflected inertia at qpos0. **This is a one-time compile conversion**; `mj_setConst` at runtime does not redo it (biasprm[2] is now negative). Warp/MJX receive the converted value.

### 12.11 `setSpring` and `setEfm0Factor`

`setSpring`: `qpos = qpos_spring`; kinematics, comPos, tendon, transmission (frames used by flex bending metric).

`setEfm0Factor` (only if `nefm0dof > 0`): constant part of the implicit flex-bending effective metric: for dim-2, non-rigid, non-interpolated, `mj_flexSimple` flexes with bending, the vertex weld bodies with exactly 3 dofs get compact slots (body order); `K = (h²·[spring enabled] + h·[damper enabled]) · K_bend` assembled by `mjd_flexStiff_assemble(bend only)`; `H = M0 (dof_M0 diagonal) + K` restricted to slots, keeping off-diagonal 3×3 blocks between different bodies, dropping same-body off-diagonal coordinates; sparse reverse Cholesky (`mju_cholFactorSymbolic` / `mju_cholFactorNumeric`, pivot floor mjMINVAL) into `efm0_L*`. Sizes must equal the compiler's `nefm0dof/nefm0L`. Rank deficiency is an error unless per-step stiffness assembly replaces the factor. `mj_effCholSetConst` computes a minimum-degree vertex ordering for the 1D/2D metric factor (`efmC_perm`). (Details in the flex/implicit spec file.)

### 12.12 `setSpringLength` (compile only, at qpos_spring)

For tendons with `tendon_lengthspring == (−1, −1)` exactly: both entries := `ten_length` at `qpos_spring`.

---

## 13. Post-setConst compile steps

### 13.1 `AutoSpringDamper` — joint `springdamper=(timeconst, dampratio)`

For joints with both > 0:
```
ndim = nv(type);  inertia = ndim / max(mjMINVAL, Σ_{k<ndim} dof_invweight0[adr+k])
stiffness = inertia / max(mjMINVAL, timeconst² · dampratio²)
damping   = 2 · inertia / max(mjMINVAL, timeconst)
jnt_stiffness = stiffness;  dof_damping[adr..adr+ndim) = damping
```
(Note: stiffness = inertia/(tc·dr)², damping = 2·inertia/tc — the "same as solref" convention, where `dampratio` scales stiffness, not damping.) Uses invweights from qpos0; does not re-run setConst.

### 13.2 Statistic overrides and validation

`extent_auto` etc. saved; spec values override. `mj_validateReferences`; a full `mj_makeData` and one `mj_step` validate the model.

---

## 14. Actuator length range — `mj_setLengthRange` (simulation-based)

### 14.1 Compiler wrapper `mjCModel::LengthRange`

```
saveopt = m->opt
m->opt.disableflags = FRICTIONLOSS | CONTACT | SPRING | DAMPER | GRAVITY | ACTUATION   (assignment, not OR: all other disable bits cleared; e.g. constraints/limits/equality remain ENABLED)
if LRopt.timestep > 0: m->opt.timestep = LRopt.timestep      (default 0.01)
cnt = #actuators passing mode/useexisting filters
nthread = min(hardware_concurrency/2, cnt), at least 1
if !usethread || cnt < 2 || nthread < 2: for i in 0..nactuator-1: mj_setLengthRange(m, d, i, &LRopt) (error aborts compile)
else: split actuators into nthread contiguous chunks of size ceil(nactuator/nthread), one mjData each
      (thread 0 reuses d), run in a pool; report first error by thread index
m->opt = saveopt
```
(The enable flags keep whatever they were, with sleep disabled from TryCompile.) Each actuator's computation is independent (fresh `mj_resetData` per side) so threading does not change results, except that actuators write `actuator_lengthrange` of the shared model — and **other actuators' lengthranges are not used by the simulation of actuator i** (actuation disabled), so results are thread-order independent.

### 14.2 `mj_setLengthRange(m, d, index, opt, error, error_sz)`

```
out = actuator_outadr[index]
ismuscle = gaintype==MUSCLE || biastype==MUSCLE;  isuser = gaintype==USER || biastype==USER
skip (return 1) if mode==NONE, or (mode==MUSCLE && !ismuscle), or (mode==MUSCLEUSER && !ismuscle && !isuser)
                   (mode ALL: compute for every actuator)
if useexisting && lengthrange[2out] < lengthrange[2out+1]: return 1
if uselimit:
   joint/jointinparent and jnt_limited[trnid]:  lengthrange[out] = jnt_range[trnid]; return 1
   tendon and tendon_limited[trnid]:            lengthrange[out] = tendon_range[trnid]; return 1
   (no gear scaling!)
for side in {0, 1}:
   mj_resetData(m, d)                      // qpos = qpos0, qvel = 0, time = 0
   updated = 0
   while d->time < opt->inttotal:          // default 10 s
       len = evalAct(m, d, index, side, opt)
       if d->time == 0: error "Unstable lengthrange simulation" (mj_step2 reset due to instability)
       if d->time > inttotal - interval:   // last 2 s
           if len < lmin[side] || !updated: lmin[side] = len
           if len > lmax[side] || !updated: lmax[side] = len
           updated = 1
   lengthrange[2*index + side] = side == 0 ? lmin[0] : lmax[1]
dif = lengthrange[2out+1] - lengthrange[2out];  error if dif <= 0
error if lmax[0]-lmin[0] > tolrange*dif  or  lmax[1]-lmin[1] > tolrange*dif    ("did not converge")
```
**Pitfall:** the assignment uses `2*index + side`, while all checks use `2*out`. For single-output actuators `out == index` only if every earlier actuator has `outnum == 1`; with multi-output (SO3) actuators earlier in the list, this writes the wrong slot. Replicate literally if bit-parity is required.

`evalAct(m, d, index, side, opt)`:
```
qvel *= exp(-timestep / max(0.01, opt->timeconst))          // velocity damping, per step
mj_step1(m, d)                                              // position/velocity stages: M, moments, constraints
moment = dense(actuator_moment row out)
qfrc_applied = M⁻¹ moment  (mj_solveM);  nrm = ‖qfrc_applied‖
qfrc_applied = moment * (2*side − 1) * accel / max(mjMINVAL, nrm)   // force along moment, magnitude so that ‖M⁻¹ f‖ = accel
nrm = ‖qfrc_applied‖;  if maxforce > 0 && nrm > maxforce: qfrc_applied *= maxforce / max(mjMINVAL, nrm)
mj_step2(m, d)                                              // integrate (with joint limits etc. active)
return actuator_length[out]                                 // computed in step1, i.e. before this step's integration
```
So the actuator is pushed with constant acceleration `accel = 20` (default) toward min (side 0) / max (side 1) length against joint limits, tendon limits and equality constraints (contacts, gravity, springs, dampers, frictionloss disabled); the extreme length over the last `interval` seconds is recorded. Uses the model's integrator and solver.

---

## 15. GPU (MuJoCo Warp) implementation

MJWarp has **no compiler**. `put_model(mjm)` copies every compile-time quantity from a C-compiled `mujoco.MjModel` (computed by the C/C++ code above in float64) into Warp arrays (`wp.float32` by default). Therefore, to match MJWarp results one must reproduce the C compiler in double precision and then **round each model field to float32** (MJX `_put_model_warp` also explicitly casts all float64 leaves via `_tree_to_float32`). Integer structures (sparsity, ids, BVH topology, mesh graph) are copied verbatim.

### 15.1 `put_model` precomputations relevant to compile-time quantities (`io.py`)

- Feature checks (NotImplementedError for unsupported enums, plugins on bodies/actuators/sensors, quadratic flex, rigid flex, flex-SDF/hfield collisions, etc.); `opt.tolerance = max(opt.tolerance, 1e-6)` (**changes the solver tolerance relative to C**); `impratio_invsqrt = 1/sqrt(max(impratio, mjMINVAL))`.
- `nv_pad`, `is_sparse(mjm)`: `jacobian == AUTO → nv > 32` (C uses `nv >= 60`!). So for 33 ≤ nv < 60 Warp uses sparse while C uses dense; algorithmically equivalent, numerically different summation order.
- `body_tree` (bodies grouped by depth, for level-parallel kernels), `body_subtreenum` (contiguous subtree sizes, relies on DFS preorder ids), `body_branches` (root-to-leaf chains), `body_isdofancestor` (nbody × nv_pad), `mocap_bodyid`, `body_fluid_ellipsoid` (mass ≥ MJ_MINVAL), `body_is_free`, `jnt_limited_*_adr`, `dof_tri_row/col`.
- Collision pair precomputation `nxn_geom_pair_filtered`: all geom pairs `i<j` filtered by contype/conaffinity, same weld body, both welds dofless (`body_dofnum[weldid]==0`), parent-child filter (unless disabled), excludes (`(b1<<16)+b2 ∈ exclude_signature`); explicit pairs mapped via upper-triangular index. Broadphase choice from the count (< 250000 → NXN; ngeom < 1000 → SAP_TILE; else SAP_SEGMENTED).
- Equality/tendon/sensor address lists, `wrap_pulley_scale = 1/divisor` for wraps after a pulley.
- M layout: `m_block_layout` (per kinematic tree: scalar Cholesky ≤ `M_BLOCK_SCALAR_MAX` (6) dofs, tile Cholesky ≤ `M_BLOCK_DENSE_MAX`, sparse LDL beyond), `qLD_updates` grouped by dof depth, `M_fullm_i/j`, `M_elemid`, `M_mulm_*` gather lists, `qD_fullm_*`. These depend only on C's `dof_parentid`, `M_rownnz/rowadr/colind`.
- Flex maps, strain Jacobian sparsity (`flexstrain_J_*` from `body_isdofancestor`), efm (effective metric) data obtained by calling C (`_get_efm_data`).
- Batched fields: arrays declared with a leading `*` dimension may be given per-world batch sizes (`batch_sizes`) → shape `[nbatch, ...]`, indexed `worldid % nbatch` (domain randomization of mass, geom_size, etc.).

### 15.2 `set_const(m, d, groups=ALL, restore=True)` (`set_const.py`)

`SetConstGroup`: `FIXED`, `KIN0`, `INERT0` (no `SPRING`). Parallelization axis: per world (`d.nworld`) × per element. All computations float32. Model outputs are written with `worldid % out.shape[0]` (if a field is not batched, all worlds write the same slot → **race; last writer wins**, values differ only if inputs are batched — users must batch outputs whose inputs are batched).

FIXED:
- `_compute_geom_bound` (dim = (nworld_geom, ngeom)): same formulas as `geomBound`; hfield sets `geom_size`; plane `hi = (MAXVAL, MAXVAL, 0)`, `lo = −MAXVAL`, rbound 0; writes `geom_aabb[world, g, 0] = center`, `[1] = half`. Mesh/SDF untouched.
- `_subtreemass` (dim = (nworld, nbody)): `Σ_{j ∈ [b, b+subtreenum)} body_mass[j]` — direct summation over the contiguous subtree (C accumulates child-to-parent bottom-up; **different summation order**).
- **Not computed in Warp FIXED**: `body_margin/contype/conaffinity` (from put_model), BVH refit (`bvh_aabb`), sameframe, fluid added mass (`geom_fluid[6..11]`), `jnt_actuatorid/tendon_actuatorid`, tree/sleep metadata, ngravcomp, tendon/flex Jacobian sparsity. Changing geom size at runtime therefore does not refit BVHs or fluid added mass.

KIN0|INERT0 (shared prefix): save `qpos`, `mocap_pos/quat`, and (if body-transmission actuators exist) `nacon` which is zeroed; `_kinematics0`: `qpos = qpos0` (per world, from possibly batched qpos0), mocap = body pose, `smooth.kinematics`, `com_pos`, `tendon`.
KIN0 (`_set_kinematics0`):
- `tendon_length0 = ten_length` (per world).
- `_compute_eq_data0`: connect anchors and flagged welds — same math as C (`data[6..9] = mul_quat(quat_inv(xquat1), xquat2)`).
- `_compute_cam_pos0` / `_compute_light_pos0`: `rel_pos = rot(xquat, cam_pos)`; `cam_pos0 = rel_pos`, `cam_poscom0 = xpos + rel_pos − subtree_com[target or body]`, `cam_mat0 = quat_to_mat(xquat * cam_quat)`; light dir normalized.
- **Not computed**: `actuator_length0`, `dof_length`, `flexedge_length0`.
INERT0 (`_set_inertia0`): `smooth.crb` (Warp's `_M` kernel computes **all** dofs including simple ones from crb — it does not use `dof_M0`), `tendon_armature`, `factor_m`, `transmission`. Then:
- `dof_invweight0`: for each dof (host loop over nv, kernel per world): unit vector → `solve_m` → extract diagonal `(M⁻¹)_ii`; finalize: free = mean of 3 trans / 3 rot diagonals, ball = mean of 3, else diagonal. **Difference:** C uses `1/mass` for joints of `body_simple == 2` bodies; Warp always uses the M⁻¹ diagonal (equal analytically for slide-only simple bodies since M is diagonal with entry = mass + armature… **not equal if armature ≠ 0**: C ignores armature in the simple shortcut, Warp includes it).
- `body_invweight0`: host loop over bodies × 6 rows: `_compute_body_jac_row` builds one row of `[jacp; jacr]` at `xipos` (walking up to the first body with dofs, then the dof ancestor chain; `jacp = cdof_lin + cdof_ang × (xipos − subtree_com[rootid])`, `jacr = cdof_ang`), `solve_m`, dot → `A_rr`. Finalize:
  - zero if `bodyid == 0 or body_weldid[bodyid] == 0` (C: if `body_dofnum[weldid] == 0` — a mocap weld root has `weldid = self ≠ 0` but no dofs: C gives 0, **Warp computes a (zero) jacobian → 0 anyway** since the jac row walk finds no dofs).
  - **Extra fallback not in C**: if `inv_trans < MINVAL and inv_rot > MINVAL`: `inv_trans = inv_rot`; elif `inv_rot < MINVAL and inv_trans > MINVAL`: `inv_rot = inv_trans`. C sets rot = 0 for `body_simple == 2` bodies and leaves zeros; Warp replaces with the other component. Also C's simple==2 shortcut `1/mass` (ignores armature) vs Warp's exact `J M⁻¹ Jᵀ`.
- `tendon_invweight0`: dense J from `ten_J`, `solve_m`, sparse dot.
- `actuator_acc0`: loop over **`nactuator`** (not nout; assumes one output row per actuator, i.e. `moment_rownnz[world, actid]`), `‖M⁻¹ moment‖`.
- Not computed: `dof_M0`, `flexedge_invweight0`, stat, damping ratio, spring lengths.
Restore: `qpos`, mocap, `nacon` copied back; if `restore`: re-run kinematics/com_pos/tendon (and crb/tendon_armature/factor_m/transmission if INERT0) at the saved qpos.

Performance notes: `dof_invweight0` costs nv triangular solves, `body_invweight0` 6·nbody solves, each a separate kernel launch from Python (O(nv + nbody) launches) — acceptable as an offline/occasional operation; not graph-capturable as written.

### 15.3 `set_length_range(m, d, index=-1)` (Warp)

**Not simulation-based.** Kernel `_set_length_range` (dim = (nworld, nu)): for JOINT/JOINTINPARENT with `jnt_limited`: `lr = gear0 > 0 ? (lo·gear0, hi·gear0) : (hi·gear0, lo·gear0)`; TENDON with `tendon_limited`: same with tendon range; all others `(0, 0)`. Differences vs C: (1) includes gear scaling (C's `uselimit` copies the raw range without gear), (2) ignores mode/useexisting (overwrites every actuator, including ones with existing ranges, with (0,0) if unlimited!), (3) no simulation for unlimited/site/body transmissions. Loops over `nu` (number of controls) indexing actuators — mismatch for multi-input actuators. Normally one should keep the compiler's `actuator_lengthrange` from the C model.

### 15.4 MJX (JAX)

- `_put_model_jax` copies all compiled fields from `MjModel` (float64 or float32 per JAX config), adds `dof_hasfrictionloss`, `tendon_hasfrictionloss`, `geom_rbound_hfield = geom_rbound`, `is_wrap_inside` (side site inside wrap geom: `|side − geom_pos| < radius && side ≥ 0`), wrap-inside solver constants (`maxiter=5`, `tolerance=1e-4`, `z_init=1−1e-5`), and precomputed convex mesh data (`mesh.convex`) for meshes in collision pairs. No flex, no plasticity.
- MJX has its own `support.py`/`smooth` but no full `set_const`; constraint regularization uses the compiled `*_invweight0`. Runtime edits of mass etc. in MJX require recompiling in C (or user-side recomputation).
- `_put_model_warp` (MJX-on-Warp): uses MJWarp's arrays for shapes, then casts every float64 to float32.

---

## 16. Reproduction pitfalls / parity notes

1. **Precision pipeline**: the compiler is double throughout, then narrows `mesh_vert`, `mesh_normal`, `hfield_data`, textures, `flex_texcoord`, skin data to float32; everything else is mjtNum (double). MJWarp/MJX-Warp then cast *all* float fields to float32 — so GPU parity requires computing in double and rounding at the end, not computing in float32.
2. **Mesh re-centering**: mesh vertices in the model are relative to the mesh CoM and rotated into principal axes (`mesh_pos`, `mesh_quat` record the transform), and the geom frame absorbs it. A reimplementation that keeps original vertices must still use the transformed geom pose; collision results depend on the vertex float32 rounding *after* transformation.
3. **eig3 ordering and sign**: principal axes from Jacobi + quaternion accumulation with sorting by bubble passes (0,1,0) and the specific swap quaternion. Different eigen solvers give axis permutations/sign flips → different `body_iquat`/`mesh_quat`, which changes `ximat`, cinert rounding, geom poses (for meshes) and `body_sameframe`. Use the exact algorithm, including `mulquat`'s renormalization and the `reltol` (4e-15 default, 1e-7 for meshes).
4. **Inertia sorted descending** for multi-geom bodies, fullinertia, fused bodies; single-geom bodies copy geom inertia unsorted.
5. **`mjuu_normvec` thresholds on the squared norm** (`< 1e-14`) and skips normalization if `|‖v‖−1| ≤ 1e-14`; quaternions very close to unit are left untouched (bitwise). `mjuu_quat2mat` short-circuits exactly-identity quaternions.
6. **Degree conversion arithmetic**: `x/180*π` for orientations vs `x*(π/180)` for ranges/refs — 1-ulp differences propagate into `qpos0` and limits.
7. **`fromto` direction**: z axis = normalize(from − to).
8. **LEGACY mesh inertia** (the default) uses absolute pyramid volumes about the face-area centroid, so non-convex meshes get inflated volume/inertia; EXACT uses signed volumes; CONVEX uses the qhull hull. Volume used for density→mass is the one recomputed about the CoM.
9. **qhull dependence**: hull vertex/face order and `Qt` triangulation of coplanar facets; `mesh_graph`, polygons (unordered_map iteration order!), and `mesh_extrema` depend on it. Polygon order is hash-order dependent (libstdc++ `std::hash<double>` on the pair) — replicate with the same STL or treat order as irrelevant to physics.
10. **BVH split**: median by `nth_element` with `|Δ| > 1e-14` tie threshold falling back to insertion order; the axis choice needs a margin of 1e-14 to prefer a later axis. Flat boxes inflated by 1e-14. The body BVH is refit by setConst; mesh BVHs are not.
11. **`dof_M0` and simple dofs**: C's `mj_crb` uses `dof_M0` for simple dofs (diagonal M rows), frozen at qpos0. Warp recomputes those entries from `crb` every step. Same in exact arithmetic for truly simple bodies; float summation order differs.
12. **Actuator armature** enters M only after `setFixed` sets `jnt_actuatorid`; `jnt_actuatorid == −2` triggers a scan over all actuators. Warp copies `jnt_actuatorid` from put_model and does not update it in set_const.
13. **Invweights shortcuts**: C uses `1/max(MINVAL, mass)` for `body_simple == 2` bodies/joints (ignores armature) and `(1/m1 + 1/m2)/2` for flex edges between simple bodies; Warp uses exact solves and an extra trans↔rot fallback for body invweights.
14. **Damping ratio conversion** (`biasprm[2] > 0` → negative damping) happens once at compile time using `M⁻¹` at qpos0 averaged over output rows; runtime set_const does not redo it.
15. **`tendon_lengthspring = (−1,−1)`** is resolved at `qpos_spring` (compile only); `qpos_spring` for free/ball joints equals qpos0 (springref ignored).
16. **Pair defaults**: undefined pair solref with equal priorities → elementwise **min** of the two geom solrefs (due to the NaN comparison), not the mix.
17. **Signatures and order**: `pair_*` and `exclude_signature` arrays are sorted by `(body1<<16)+body2`; `pair_geom1` is always the geom on the lower-id body.
18. **Element ordering after fusestatic/discardvisual** changes body/geom ids, and fused geoms are appended after the parent's own geoms.
19. **Hfield geom size** is overridden to `(hx, hy, 0.25·elev + 0.5·base)` (a "mean height" used for inertia and `geom_size`); collisions use `hfield_size` directly.
20. **Stat**: `extent` only updated if the bbox is nondegenerate in x; `meansize` sums `body[i]/(nbody−1)` term by term; spec overrides applied last. `stat.meaninertia` is unused by the current solver.
21. **Arena size** depends on `mj_isSparse` (nv ≥ 60) — affects only memory, not results, unless the arena overflows.
22. **mj_setLengthRange index bug**: stores into `2*index+side` but checks `2*out`; only matters with multi-output actuators preceding the actuator.
23. **Lengthrange disableflags** are *assigned* (not OR'ed), so e.g. `mjDSBL_LIMIT` or `mjDSBL_EQUALITY` set by the user are cleared during the lengthrange simulation; limits/equalities are active. Sleep is disabled. Timestep = 0.01 by default; integrator/solver from the model.
24. **Warp `is_sparse` threshold** (nv > 32) differs from C (nv ≥ 60), and Warp raises `opt.tolerance` to ≥ 1e-6.
25. **Warp set_const output batching**: outputs indexed `worldid % shape[0]`; unbatched outputs with batched inputs race.
26. **Warp set_length_range** applies gear and zeros unlimited actuators; do not use it to reproduce C's `actuator_lengthrange`.
27. **mjuu_z2quat** for antiparallel vectors uses axis (1,0,0) and angle π (from `atan2(0⁺, negative)`); for tiny cross products (< 1e-10) the axis is replaced by x even when not antiparallel (angle ≈ 0 then).
28. **Free joint alignment** (`alignfree`) moves the body frame to the inertial frame: `qpos0` (free joint) = aligned body pose, geoms re-expressed; `body_ipos = 0`, `body_iquat = identity` → `body_sameframe = BODY` and potentially `body_simple`.
29. **Frame scaling** (gauges): identity unless frames carry `scale`/`massscale`; anisotropic scales change primitive sizes by axis factors with alignment tolerance 0.0349 and produce full inertias.
30. **Geom mass with `mass` given but zero volume** (`GetVolume ≤ 1e-14`): `mass_` stays 0 (silently), unlike density path.

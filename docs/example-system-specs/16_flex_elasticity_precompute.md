# 16 — Flex elasticity precomputation (compiler) and the runtime formulas it feeds

**Source files covered**

| Layer | Files / functions |
|---|---|
| Compiler, flexcomp generation | `src/user/user_flexcomp.cc`: `mjCFlexcomp::Make`, `MarkEmptyCells`, `MakeGrid`, `MakeSquare`, `MakeBox`, `MakeMesh`, `MakeGMSH`, `GridID`, `AppendMeshSpec`, `ToAssetFrame`; `src/user/user_mesh.cc`: `GridData`, `BoxData`, `BoxProject`, `BoxID`, `mjCMesh::MakeGrid/MakeCircle/MakeDisc/MakeBox/MakeCylinder/MakeEllipsoid`; `src/user/user_flexcomp.h` (enums `mjtFcompType`, `mjtDof`, `TriangleWinding`, `FastWindingBVH`) |
| Compiler, nonlinear elasticity | `user_flexcomp.cc`: `Stencil2D/3D`, `ComputeVolume<T>`, `MetricTensor<T>`, `ComputeBasis<T>`, `ComputeStiffness<T>`, `ComputeSNH`, `CreateFlapStencil`, `cot`, `ComputeBending`, `RodCurvatureBinormal`, `RodEdgeFrame`, `ComputeRodBending` |
| Compiler, linear (interpolated) elasticity | `quadratureGaussLegendre`, `phi`, `dphi`, `ComputeLinearStiffness` (full cell + FCM), `ComputeLinearStiffness2D`, `ComputeWarpMode`, `ComputeWarpStiffness`, `EigendecomposeStiffness` (+ `mjuu_eigendecompose` in `user_util.cc`), `ComputeInterpBending`, `ComputeUnrotatedNodePositions` |
| Compiler, FCM geometry | `ComputeCellEmpty`, `TriangleIntersectsBox`, `SegmentIntersectsBox`, `PointInTet`, `TetIntersectsBox`, `EvalShiftedLegendre1D`, `BuildLegendreMomentProjector`, `ClipPolygonAxis`, `AccumulateClippedTetMoments`, `IsClosedSurfaceMesh`, `NodeInEmptyCells`, `MassBasis1D`, `NodeMassWeight`, stiffness/FCM caches |
| Compiler, other | `mjCFlex::Compile`, `PlasticNum`, `ApplyGauge`, `IsSimple`, `CreateBVH`, `CreateShell`; `user_model.cc` (flex → mjModel copy, lines ~3600–3830); `user_objects.cc` `mjCEquality::Compile` (flex equality checks) |
| Runtime (C) | `engine_core_util.c/.h`: `mj_stretch*` helpers, `mj_snhCubic`, `mj_snhVolume`, `snhSVD`, `snhEigen3`, `snhPositive3`, `snhCubicMetric`, `snhProject`, `snhProjectedBlock`, `mj_flexHessian`, `mj_flexHessianMul`, `mj_flexHasStiffness`, `mju_flexGatherState`; `engine_passive.c`: `mj_flexPassiveStretch`, `mj_flexPassiveInterp` (plastic part), `mj_flexPassiveBend` (plastic part); `engine_forward.c`: plastic return mapping inside `advanceFinish`, `gramDet3D`; `engine_util_misc.c/.h`: `mju_flexPhi/Dphi`, `mju_defGradient`, `mju_flexPolarRotation`, `flexInterpRotation`, `mju_flexGatherCellState`, `mju_flexGatherFaceState`, `mju_flexInterpRotation2D`, `mju_flexFaceNormal2D`, `mju_cellLookup`; `engine_util_spatial.c`: `mju_mat2Rot`; `engine_io.c` plastic initialisation/validation |
| MuJoCo Warp | `mujoco_warp/_src/io.py` (`_flex_has_stiffness`, `_get_nflexintcell`, `_get_nflexface`, `_flex_body_simple`, `_flex_simple`, `_eff_flex_stiff_possible`, `_get_efm_data`, maps), `types.py` (`FLEX_STIFFNESS_3D`, `ROD_HINGE_SIZE`, `FLEX_INTERP_MAX_NODES`), `passive.py` (`_snh_*`, `snh_project`, `snh_projected_block`, `stretch_*`, `_flex_elasticity`, `_flex_passive_interp`, `flex_hessian`, `flex_stretch_mul`), `forward.py` (`_next_plastic_flex_stretch/_bend/_interp`, `_gram_det_3d`, `_face_trpk`), `support.py` (`mat33_to_quat_polar`, `compute_interp_cell_quat`, `compute_interp_face_quat`), `smooth.py` (`_flex_face_kinematics`) |
| MJX (JAX) | no flex elasticity / plasticity support (see §12) |

**Role in the pipeline.** Everything a flex needs for elastic passive forces is precomputed once by the compiler in double precision from the reference (qpos0, `xpos0/xquat0`) configuration: per-element stretch metrics (`flex_stiffness`), per-edge bending operators or rod-hinge records (`flex_bending`), dense corotational stiffness blocks or constraint eigenmodes for interpolated (trilinear/quadratic) flexes, FCM (finite-cell-method) cut-cell quadrature for interpolated volumes (which only influences the stiffness blocks and node masses; nothing FCM-specific is stored in mjModel), node/vertex reference data (`flex_node0`, `flex_vert0`, `flex_vertcell`), flap topology (`flex_edgeflap`), shells/layers, and plastic-state sizes (`flex_plasticadr`). At runtime `mj_passive` (spec 09) consumes these arrays; the plastic return mapping in `advanceFinish` (after the state update of every integrator) updates `d->plastic`. This file documents the precomputation formula by formula and fills in the runtime math that spec 09 only summarised (SNH energy/gradient/Hessian and its PSD projection, plasticity return mapping, polar rotation, deformation gradient). Spec 09 remains the reference for the force assembly/scatter loops and for 1D rod runtime bending; spec 13 for the general compiler sequence.

---

## 0. Notation and constants

- `E` = `young`, `ν` = `poisson`, `t` = `thickness`. Lamé: `μ = E/(2(1+ν))`; 3D `λ = Eν/((1+ν)(1−2ν))`; plane stress `λ* = Eν/(1−ν²)`. Plate rigidity `D = E t³/(12(1−ν²))`.
- `s_e = L_e² − L0_e² (− plastic_e)` = squared-length elongation of element-local edge `e`.
- Local edge orderings (`eledge` / `mj_stretchEdges`):
  - dim 1: `{0,1}`
  - dim 2 (triangle): `{1,2}, {2,0}, {0,1}` (edge e is opposite vertex e)
  - dim 3 (tet): `{0,1}, {1,2}, {2,0}, {2,3}, {0,3}, {1,3}`
- `kNumEdges[dim-1] = {1, 3, 6}`.
- Constants: `mjMINVAL = 1e-15`, `mjMAXVAL = 1e10`, `mjEPS = 1e-14`, `mjPI`.
  FCM: `kFcmSubdiv = 2`, `kFcmAlphaMin = 1e-6`, `kFcmMassMin = 2e-2`, `kSliverMargin = 0.02` (= `kMaxExtrap` in `mju_cellLookup`). Rod: `kRodStraightTol = 1e-4`. Interp bending record size `BEND_EDGE_SIZE = 10`; rod record `mjROD_HINGE_SIZE = 10`.
- Enums (`user_flexcomp.h`): `mjtFcompType = {GRID=0, BOX, CYLINDER, ELLIPSOID, SQUARE, DISC, CIRCLE, MESH, GMSH, DIRECT}`; `mjtDof = {FULL=0, RADIAL, TRILINEAR, QUADRATIC, 2D}`. `elastic2d ∈ {0 none, 1 bend, 2 stretch, 3 both}`; `elastic3d ∈ {0 StVK, 1 SNH}`; `section ∈ {mjFLEXSECTION_CIRCLE=0, mjFLEXSECTION_BOX}`.

---

## 1. mjModel arrays produced (exact layout)

Per flex `f` (copied in `user_model.cc`, flex loop):

| field | value |
|---|---|
| `flex_stiffnessadr[f]` | `-1` if `pfl->stiffness` empty, else running offset into `flex_stiffness`; `nflexstiffness = Σ stiffness.size()` |
| `flex_bendingadr[f]` | `-1` if `bending` empty, else running offset into `flex_bending`; `nflexbending = Σ bending.size()` |
| `flex_damping[f]` | `damping` (spec) |
| `flex_yield[f]` | `stressyield` |
| `flex_plasticadr[f]` | `PlasticNum() > 0 ? running plastic offset : -1` (flex slots after all joint slots (`nq` per joint with `springyield>0`) and tendon slots (2 each)) |
| `flex_edgeequality[f]` | first equality (in equality order) whose `name1 == flex name`: `mjEQ_FLEX → 1`, `mjEQ_FLEXSTRAIN → 3`; else 0 |
| `flex_interp[f]` | `elastic2d ? -order : order` (`order` = `spec.order`, 0 for non-interpolated) |
| `flex_cellnum[3f..]` | `spec.cellcount` (copied even for non-interpolated flexes; default (1,1,1)) |
| `flex_node0[3·(nodeadr+i)]` | `node0_` = unrotated reference node positions (§7.2) |
| `flex_vert0[3·(vertadr+j)]` | normalised vertex coordinates (§7.10) |
| `flex_vertcell[vertadr+j]` | containing (or nearest active) cell for interpolated; `-1` otherwise |
| `flex_edge[2e]`, `flex_edgeflap[2e]` | edge `(min,max)` vertex pair; flap vertices `(v2,v3)` iff `dim==2 && elastic2d∈{1,3}` (even if `young==0`), else `(-1,-1)` |
| `flex_elemedge` | element-local → flex edge index (`edgeidx_`), `kNumEdges[dim-1]` per element |
| `flex_elemlayer`, `flex_shell`, `flex_shellnum = shell.size()/dim`, `flex_shelldataadr` | from `CreateShell` (§8) |
| `flex_vert` | zero if `centered && !interpolated`, else `vert_` |
| `flex_node` | zero if `centered && interpolated`; else `node_` for interpolated |
| `flex_vertbodyid` | all `vertbodyid[0]` if rigid; else per vertex. For **interpolated** flexes the compiler overwrote every `vertbodyid[i] = -1` during `Compile` (vertices do not have bodies; their positions are interpolated from nodes) |
| `flex_nodebodyid` | all `nodebodyid[0]` if rigid; else per node |
| `flexedge_rigid` | rigid flex → 1; non-interpolated → `weldid(b1)==weldid(b2)`; interpolated → 0 |
| `flex_size` | interpolated: half extents of the unrotated node AABB; else root BVH half-size − radius (§7.10) |

A warning "flex ... is not rigid and has no equality constraints or passive forces" is issued if `!rigid && edgeequality==0 && !edgestiffness && !edgedamping && !damping && bending.empty()`.

### 1.1 `flex_stiffness` record per flex

| flex kind | condition | layout |
|---|---|---|
| standard 2D (`!interp`, dim 2) | `young>0` | `21·nelem` doubles, all zeros, then per element `t`: entries `[21t + 0..5]` = packed upper-triangular 3×3 metric `K` (if `elastic2d≥2 && thickness>0`), `[21t+6..20]` = 0. **Special**: if `stressyield>0` the global slot `stiffness[6]` (element 0, entry 6) is overwritten with `thickness` (only for `elastic2d∈{1,3}` path, see §4.4) |
| standard 3D (`!interp`, dim 3) | `young>0` | `24·nelem`: `[24t+0..20]` packed upper-triangular 6×6 metric; `[21],[22],[23]` = 0 (StVK) or `γ, β, 1/det(Dm)` (SNH) |
| 1D | — | none (`stiffness` stays empty; 1D uses `flex_bending`) |
| interpolated volume (`interp>0`) | `young>0` or strain equality | `nelem_fe · (3npe)²` with `npe=(order+1)³`, `nelem_fe = cx·cy·cz`; dense row-major blocks, **negative** sign (`K_stored = −K_phys`); empty cells keep an all-zero block. Strain-equality flexes store eigenmodes instead (§6.6) |
| interpolated shell (`interp<0`) | `young>0 && elastic2d≥2`, or strain equality | `nelem_fe·(3npe)²` with `npe=(order+1)²`, `nelem_fe = 2(cy cz + cx cz + cx cy)` |

Packing order of a symmetric `n×n` metric (`MetricTensor`): `for a in 0..n-1: for b in a..n-1: packed[id++] = K[a][b]` (3 edges → 6 numbers, 6 edges → 21 numbers). Unpacked at runtime by `mj_stretchMetric`.

### 1.2 `flex_bending` record per flex

| flex kind | layout |
|---|---|
| standard 2D, `elastic2d∈{1,3}`, `young>0` | `17·nedge`: per flex edge `e`, `b[17e+4i+j]` (4×4 over stencil `{v0,v1,v2,v3}` = `{edge[0], edge[1], flap[0], flap[1]}`), `b[17e+16]` = curved-reference coefficient. Boundary edges (no second flap) remain all zero |
| 1D (not interp, not rigid), `young>0` | `[nrec, rec_0(10), rec_1(10), ...]`; cleared (bendingadr −1) if `nrec==0` (§4.5) |
| interpolated shell, `elastic2d∈{1,3}`, `thickness>0`, `young>0` | `[nedge, edge_0(10), ...]` with `[fe_A, fe_B, lA0, lA1, lB0, lB1, stiffness, dn0x, dn0y, dn0z]` (§6.8) |

### 1.3 Plastic state size (`mjCFlex::PlasticNum`, mirrored by `engine_io.c` validation and `mj_resetData` zeroing)

Compiled flex:
- **Interpolated**: 0 if rigid, `stressyield ≤ 0`, `young ≤ 0`, has a strain equality (or any `mjEQ_FLEXSTRAIN` names it), or any cellcount ≤ 0. Else volume (`elastic2d==0`): `6·cx·cy·cz`; shell with `(elastic2d & 2) && thickness>0`: `3·2(cy cz + cx cz + cx cy)`; otherwise 0.
- **Standard**: if `stressyield>0 && !rigid && dim≥2`: `+ (dim==2?3:6)·nelem` if `stiffness` non-empty and `stiffness[0] != 0`; `+ nedge` if `dim==2 && bending non-empty && thickness>0`.

Layout inside the flex's plastic block: stretch first (`nedge_local` numbers per element: 3 or 6, element-major), then bending (one per flex edge). Interpolated: 6 per cell (`ε_xx, ε_yy, ε_zz, γ_xy, γ_xz, γ_yz` engineering shear) or 3 per face element (`ε_00, ε_11, γ_01`). `mj_resetData` zeros the block and sets `flex_hessian_valid[f]=0`; keyframes may carry `key_plastic`.

Note the runtime test in `engine_io.c`/`mj_flexPassiveBend` uses `flex_stiffness[adr] != 0` to decide the stretch part count — consistent with the compiler only because a 2D bend-only flex has an all-zero stretch record (§4.4).

---

## 2. flexcomp generation (`mjCFlexcomp::Make`)

### 2.1 Defaults and checks
Defaults: `type=GRID`, `count=(10,10,10)`, `cellcount=(-1,-1,-1)`, `spacing=(0.02,0.02,0.02)`, `scale=(1,1,1)`, `origin=0`, `mass=1`, `inertiabox=0.005`, `equality=0`, `pos=0`, `quat=(1,0,0,0)`, `rigid=false`, `centered=false`, `doftype=FULL`.

Checks (in order): mesh asset only with MESH/GMSH; parent body must be named; `1≤dim≤3`; each `count[i] ≥ 1` (and ≥2 for RADIAL when dim 3); for non-direct types `spacing[i] ≥ 2·radius + margin`; `scale[i] ≥ mjMINVAL`; `mass, inertiabox ≥ mjMINVAL`; `equality ∈ {0,1,3}`; orientation resolved (`mjs_resolveOrientation`). "Direct" = `{DIRECT, MESH, GMSH}`.

### 2.2 Point/element generators (all in the flexcomp frame, before pose)

**GRID** (`GridData`, `disc=false`), positions centered: `p = spacing ∘ (i − 0.5(count−1))`.
- dim 1: `count[0]` points on x; elements `(i, i+1)` for `i < n−1` (`AppendMeshSpec`, open).
- dim 2: points row-major `ix` outer, `iy` inner, `z=0`; per quad `(ix,iy)` with `ix<count0−1, iy<count1−1`: `vert = {c2c1·ix + c2·iy, c2c1(ix+1)+c2·iy, c2c1(ix+1)+c2(iy+1), c2c1·ix + c2(iy+1)}` (note `count[2]` multiplies! with default `count[2]=10` → **indices are only correct if `count[2]==1`**; flexcomp grid dim 2 is normally used with `count[2]=1`), triangles `quad2tri = {0,1,2},{0,2,3}`. Texcoords (if material given and none supplied): `(ix/max(c0−1,1), iy/max(c1−1,1))`.
- dim 3: points `ix,iy,iz` nested; per cube 8 corners `vert[0..7]` (bit order x then y then z as listed in source: `000,100,110,010,001,101,111,011`), 6 tets `cube2tets = {0,3,1,7},{0,1,4,7},{1,3,2,7},{1,2,6,7},{1,5,4,7},{1,6,5,7}`.

**CIRCLE** (dim 1): requires `count[0] ≥ 4`; `n = count0−1` points, `θ = 2π/n`, `r = spacing0/(2 sin(θ/2))`, `p_i = r(cos θi, sin θi, 0)`; elements closed loop `(i, (i+1)%n)`.

**SQUARE** forces dim 2 = GRID dim 2. **DISC**: grid dim 2 with `disc=true` (quads in quadrants with `(x<−ε, y>−ε)` or `(x>−ε, y<−ε)`, `ε=mjEPS`, use flipped split `{0,1,3},{1,2,3}`), then radial projection per point: `L0 = max(|x|,|y|)`, normalise `(x,y)`, multiply by `size·L0` with `size = 0.5·spacing·(count−1)`.

**BOX / CYLINDER / ELLIPSOID** (`BoxData`, requires all counts ≥ 2; `open` defaults to `true` in `mjCFlexcomp::MakeBox`, which only matters for dim 2): surface lattice points visited face group by face group (iz∈{0,max}: all ix,iy; iy∈{0,max}: interior iz; ix∈{0,max}: interior iy,iz), mapped by `BoxProject`: `p = 2i/(count−1) − 1`, `size = 0.5·spacing·(count−1)`; box: `p ∘ size`; cylinder: `L0=max(|px|,|py|)`, normalise `(px,py)`, `(px,py)·size·L0`, `pz·size_z`; ellipsoid: normalise 3-vector, `∘ size`.
- dim 3: point 0 = center `(0,0,0)`; surface points use index `BoxID(ix,iy,iz)` (closed-form numbering, +1 for the center); each surface quad gives two tets `(0, q0, q1, q2)`, `(0, q0, q3, q2)`.
- dim 2: triangles per surface quad with winding chosen per face so normals point outward (step1/step2, ix0/dx, iy0/dy formulas in `BoxData`); with `open && dim==2`, the iz=max cap is omitted and the side rows include iz=max.

**MESH / GMSH** (`MakeMesh`): from a mesh asset (compiled; if the asset has a non-identity `instancemap_`, a copy with identity map is compiled) or a file (`meshdir` + file; GMSH files set content type `model/vnd.gmsh`; legacy `.msh` rejected).
- If the mesh has tetrahedra: dim forced 3 (error if user dim≠3), `point = Node()`, `element = Tet()`; asset meshes are mapped back to asset frame by `ToAssetFrame` (undo inertial re-centering: `p ← R(quat⁻¹)ᵀ…`, i.e. `p ← mat·p + pos` with `mat = quat2mat(conj(q))` applied transposed); per-node texcoords kept only if `2·npoint` long.
- GMSH without tets → dim 2. dim 2: `element = Face()`. dim 1: edges `(f0,f1)` from each face triple (degenerate triangles `(i1,i2,i2)`). dim 3 from a surface mesh: `origin` required; `origin` inserted as point 0; for each face `(a,b,c)` (indices +1) a tet `(0,a,b,c)` is added only if `((a−o)×(b−o))·(c−o) ≥ mjMINVAL`.
- Points are the (possibly deduplicated) mesh *vertices*, not nodes.

**DIRECT**: user-provided `point`/`element`.

### 2.3 Common post-processing
1. Default `flatskin = true` for BOX, CYLINDER and GRID dim 3 unless set.
2. Pin array size checks: `pinrange` multiple of 2, `pingrid` multiple of dim, `pingridrange` multiple of 2·dim; pingrid(range) only for GRID or interpolated, not dim 1.
3. Require non-empty point/element; element ids in `[0, npnt)`.
4. Direct types: `point *= scale` (componentwise) if scale ≠ 1.
5. All points: `p ← pos + R(quat)·p` (`mjuu_trnVecPose`).
6. `minmax` = AABB of transformed points.
7. **Pinning** (`pinned` sized `max(npnt, nnode)`, init = `rigid`). If `!rigid`:
   - `pinid` (range-checked `< npnt`), `pinrange` inclusive.
   - `pingrid`: range check against `count` — except for non-grid interpolated types where the check bound is 2 (trilinear) or 3 (quadratic); index by `GridID(ix,iy) = ix·count1 + iy` or `GridID(ix,iy,iz) = ix·count1·count2 + iy·count2 + iz` using the flexcomp `count` (pitfall: for non-grid interpolated flexes this formula uses the flexcomp counts, not node-grid dims).
   - `pingridrange`: bounds `count[k % dim]`, inclusive boxes.
   - RADIAL: `pinned[0] = true`.
   - `allpin` (over the first `npnt`) → `rigid = true`; `nopin` → `centered = true`.
   - For interpolated types the same `pinned` array is later indexed by **node** index (§2.5); pins given by point id therefore pin the node with that index.
8. Direct types: remove unreferenced points (reindex elements, compact point/texcoord/pinned; `pinned.resize(max(new_npnt,nnode))`).
9. Create the mjsFlex from the flexcomp default class (`def.Flex()`), set name, `elem`, `texcoord`, `elemtexcoord`; `vert = point` unless `centered`.
10. `rigid`: `vertbody = [parent]` and return.

### 2.4 Vertex bodies (non-rigid)
`bodymass = mass/npnt` (npnt counts pinned points too, so pinned points' share is lost), `bodyinertia = bodymass·2·inertiabox²/3` (diagonal, `explicitinertial=true`, `ipos=0`).
For each used point: if pinned or interpolated doftype → vertbody = parent (and plugin attached to parent). Else child body at `pos = point`, named `"<name>_<i>"`, with joints:
- FULL: 3 slide joints, axes x, y, z (order matters for `mj_flexBodySimple`).
- RADIAL: one slide along normalised `pos`.
- 2D: slides x, y.
Then `point[i] = 0` (vertex at body origin) unless centered.

### 2.5 Node grid (TRILINEAR order 1 / QUADRATIC order 2)
- `spec.order = 1|2`; `cellcount` copied if user set (≥0), else default (1,1,1).
- `nx = cx·order+1` etc.; `nnode = nx·ny·nz`; node linear index `idx = gi·ny·nz + gj·nz + gk`.
- Volume mode (`elastic2d==0`): `MarkEmptyCells` → `flex->ComputeCellEmpty(point, element, nelem, dim, minmax)` (§6.4) then pin every node for which `NodeInEmptyCells` is true.
- Shell mode (`elastic2d≠0`): pin every non-boundary node (`0<gi<nx−1 && 0<gj<ny−1 && 0<gk<nz−1`).
- If any node pinned, `centered=false`.
- Node position: `s = gi/(cx·order)`, `p = minmax_lo + s·(minmax_hi − minmax_lo)` per axis.
- Pinned node: `node[idx] = p` (in parent frame), nodebody = parent.
- Free node: body at `p`, 3 slides x,y,z, named `"<name>_<gi>_<gj>_<gk>"`, mass:
  - volume: `NodeMassWeight(gi,gj,gk)` (§6.5);
  - shell trilinear: 1.0;
  - shell quadratic: `li = gi % order` etc., `w = massP2[li]` with `massP2 = {1/6, 2/3, 1/6}` (index 0 for li=0, 1 for li=1), multiplicity `ncells_i = 2` if `0<gi<nx−1 && li==0` else 1; `mass = wi wj wk · ncells_i ncells_j ncells_k`.
  - `inertia = mass·2·inertiabox²/3`.
- Normalise: `scale = mass_total_spec / Σ node masses` applied to mass and inertia of all created node bodies.
- `node` array stored unless centered; `vert = point` always stored for interpolated.

### 2.6 Equality setup
- `equality==1`: one `mjEQ_FLEX` (active, `name1 = flex`).
- `equality==3`: shell (interpolated && elastic2d): one `mjEQ_FLEXSTRAIN` per face element `fe ∈ [0, nelem_fe)`, `data = (fe, −1, −1)`; volume: one per non-empty cell, `data = (ci, cj, ck)`.
- Compile-time checks (`mjCEquality::Compile`): flex non-rigid; error if `dim>1 && elastic2d≠1 && young>0` ("flex constraints and elasticity cannot both be present"; bend-only and 1D rod bending are compatible); error if `edgestiffness>0`; FLEXSTRAIN requires interpolated and data matching shell/volume mode and in range. Gauge: `solimp[2] *= BodyGauge().Shape(1)`.

---

## 3. `mjCFlex::Compile` sequence (elasticity-relevant steps)

1. `CopyFromSpec` (clears `edge`, `shell`, `flaps`, `stiffness`, `bending`, `cell_empty/full/vol/alphas`, `has_strain_eq`, `rigid`, `centered`). Consequence: **`cell_empty` computed by flexcomp is discarded and always recomputed in step 13** (normally a cache hit, §6.4.6).
2. Checks: `1≤dim≤3`; elem non-empty, multiple of dim+1; vertbody or nodebody; `edgestiffness>0 && dim>1` → error; elastic2d requires `thickness>0`, `0≤ν<0.5`, and dim 2 unless interpolated; `elastic3d∈{0,1}`, SNH requires dim 3, non-interpolated, and `integrator == mjINT_DISCRETE`; `stressyield ≥ 0`; plasticity requires dim ≥2 unless interpolated.
3. `nvert`, `rigid` (vertbody size 1), `centered` (vert empty); node count must equal `(cx·order+1)(cy·order+1)(cz·order+1)`; order required if nodes present; cellcount ≠ 0.
4. Texcoords, materials, `ResolveReferences`, `ApplyGauge` (positions by body content gauges; `radius, thickness, margin, gap, solimp[2], friction[1..2]` × length scale; `young × Force(−1,1)`, `edgestiffness, edgedamping × Force(0,1)`).
5. Repeated-vertex check; `rigid` if all vertbodies equal (non-interp); `centered` if all vert (or node) coordinates are exactly 0.
6. Joint damping on free 3-slide vertex/node bodies is an error.
7. `vertxpos[i] = body.xpos0 + R(body.xquat0)·vert[i]` (offset only if `!centered || interpolated`); interpolated: `vertbodyid[i] = -1` afterwards.
8. `nodexpos[i] = body.xpos0 + R(xquat0)·node[i]` (offset only if `!centered`).
9. `nodexpos_local = ComputeUnrotatedNodePositions(nodexpos, R0)` (§7.2; `R0` = identity if not interpolated).
10. **Tet orientation fix** (dim 3): if `((v1−v0)×(v2−v0))·(v3−v0) > 0`, swap element entries 1 and 2. Stored tets thus have **non-positive** signed volume `V = ((v1−v0)×(v2−v0))·(v3−v0)/6 ≤ 0`.
11. Edges: iterate elements in order, local edges in `eledge[dim−1]` order, key `(min,max)`; first occurrence appends to `edge` and assigns the next index; `edgeidx_[f·ne + e]`. dim 2: `CreateFlapStencil` (§4.3) (assert indices match).
12. If `young > 0`: Poisson check; mocap ancestors forbidden for `!rigid && !interpolated && (dim>1 || damping>0)`; stretch (§4.1/4.2), rod (§4.5), bending (§4.4).
13. `CreateShell` (§8). Interpolated volume with empty `cell_empty`: compute `vertxpos_local = R0·vertxpos` (`mjuu_mulvecmat`: rows of R0 dotted with x, i.e. `R0ᵀ`-as-rotation applied), `minmax` = AABB of `nodexpos_local`, `ComputeCellEmpty(vertxpos_local, elem, nelem, dim, minmax)`, then recompute `nodexpos_local` (and R0) with the first non-empty cell.
14. Last flex of the model only: free node bodies in shell interiors / empty-cell-only nodes → error (bodies shared by several flexes or appearing more than once are "supported").
15. Joints on the common parent/ancestors of a fully unpinned 3-slider flex → error.
16. `has_strain_eq` if any `mjEQ_FLEXSTRAIN` names the flex.
17. Interpolated stiffness (cached) (§6.6); interpolated shell bending (§6.8).
18. `CreateBVH`; `vert0_`, `vertcell_`, `size` (§7.10); `node0_ = nodexpos_local`.

---

## 4. Non-interpolated elasticity precomputation

### 4.1 `ComputeStiffness<Stencil2D|Stencil3D>` (StVK, "metric tensor of edge lengths")

Purpose: per element, a symmetric matrix `K` (3×3 or 6×6) such that the elastic energy is

$$ U = \tfrac14\, s^\top K\, s, \qquad s_e = L_e^2 - L0_e^2 $$

equal to St. Venant–Kirchhoff `U = measure·(μ tr(ε²) + (λ/2) tr(ε)²)` with Green strain ε. Edge tension used by the runtime: `T = K s` (= `2 ∂U/∂s`), force on edge endpoint `i0`: `−T_e (x_{i0} − x_{i1})`, on `i1`: `+T_e(...)`.

Steps:
1. `volume = ComputeVolume<T>(x, v)`:
   - 2D: `A = |(x1−x0)×(x2−x0)|/2` (always ≥0).
   - 3D: `V = ((x1−x0)×(x2−x0))·(x3−x0)/6` (**signed**; ≤0 after §3 step 10).
2. `measure = |volume| · (2D ? thickness : 1)`.
3. `μ = E/(2(1+ν))`; `λ = 2D ? Eν/(1−ν²) : Eν/((1+ν)(1−2ν))`.
4. `mu_k = μ·measure`, `la_k = (λ/2)·measure` (the ½ because `Σ_e s_e B_e = 2ε`).
5. For each local edge `e`: `basis[e] = ComputeBasis<T>(x, v, faceL = face[edge2face[e][0]], faceR = face[edge2face[e][1]], volume)`.
6. `MetricTensor<T>(stiffness, t, mu_k, la_k, basis, stride = 3D ? 24 : 21)`.

**`ComputeBasis<Stencil2D>`** (Weischedel, "discrete geometric view on shear-deformable shell models", §4.1 remark). Tables: `face = edge2face = {{1,2},{2,0},{0,1}}`. For edge `e` (opposite vertex `i=e`, `j=(i+1)%3`, `k=(i+2)%3`) this resolves to:

```
edgesL = x_k − x_i ;  edgesR = x_j − x_i
n      = normalize(edgesR × edgesL)
bL     = n × edgesL ;  bR = edgesR × n
B_e[r][c] = (bL[r] bR[c] + bR[r] bL[c]) / (8 A²)
```
(`edgesL = xL0 − xL1` with `faceL = face[edge2face[e][0]]`, `edgesR = xR1 − xR0` with `faceR = face[edge2face[e][1]]`; e.g. e=0: faceL={2,0}, faceR={0,1}.)

**`ComputeBasis<Stencil3D>`**: tables
```
face      = {{2,1,0},{0,1,3},{1,2,3},{2,0,3}}
edge2face = {{2,3},{1,3},{2,1},{1,0},{0,2},{0,3}}
```
For edge `e`: `L = face[edge2face[e][0]]`, `R = face[edge2face[e][1]]` (the two faces not adjacent to the edge),
```
nL = (x_{L1}−x_{L0}) × (x_{L2}−x_{L0}) ;  nR = (x_{R1}−x_{R0}) × (x_{R2}−x_{R0})   (unnormalised)
B_e[r][c] = (nL[r] nR[c] + nR[r] nL[c]) / (72 V²)
```

**`MetricTensor<T>`** (`n = kNumEdges`):
```
trE[e]     = B_e[0][0] + B_e[1][1] + B_e[2][2]
trEE[a][b] = Σ_{i,j} B_a[i][j] · B_b[j][i]          (= tr(B_a B_b))
K[a][b]    = mu · trEE[a][b] + la · trE[a] · trE[b]
packed upper triangle into metric[stride·t + id], id = 0..n(n+1)/2−1
```
(`mju_error` if the count ≠ `n(n+1)/2`.) Accumulation order: `trE` loops `i` 0..2; `trEE` loops `i` then `j`.

Only computed for `dim==2 && elastic2d≥2 && thickness>0` or `dim==3`; otherwise the zero-initialised record (`21·nelem` / `24·nelem`) remains.

### 4.2 `ComputeSNH` (stable Neo-Hookean tet record, `elastic3d=1`)

```
V   = ComputeVolume<Stencil3D>  (signed),  V0 = |V|
μ   = E/(2(1+ν)) ;  λ = Eν/((1+ν)(1−2ν))
B_e = ComputeBasis<Stencil3D>(…, V)           (same as StVK)
K   = MetricTensor(mu = μV0, la = −μV0)       → K_ab = μV0 (tr(B_aB_b) − tr(B_a)tr(B_b))
k[21] = γ = −μ / (72 V0)
k[22] = β = V0 (λ + 2μ) / 2
k[23] = 1 / (6V)        (= 1/det(Dm), signed, negative for stored orientation)
```
Energy (header comment, `engine_core_util.h`):

$$ U = \tfrac14 s^\top K s + \gamma\,P(s) + \beta (J-1)^2 $$

where `P(s) = det D(s)` with the Gram-difference matrix at vertex 0 (edges to 1,2,3)

$$ D(s)=\begin{pmatrix} a & d & e\\ d & b & f\\ e & f & c\end{pmatrix},\ a=s_0,\ b=s_2,\ c=s_4,\ d=\tfrac12(s_0+s_2-s_1),\ e=\tfrac12(s_0+s_4-s_5),\ f=\tfrac12(s_2+s_4-s_3) $$

and `J = det[x1−x0, x2−x0, x3−x0] · k[23]`. The header states this equals `V0[μ/2(tr FᵀF − 3) − μ(J−1) + (λ+μ)/2 (J−1)²]` exactly, with no division by the current volume. The runtime detects SNH by `dim==3 && k[21] != 0`.

### 4.3 `CreateFlapStencil` (dim 2 edge → 4-vertex flap)
Iterate triangles `t` in order, local edges `e` in `{1,2},{2,0},{0,1}` order. Key `(min,max)`:
- new edge: `flap = {min, max, v[(edge[e][1]+1)%3] (= opposite vertex), −1}`, index `ne++`;
- existing edge: `flaps[idx].vertices[3] = opposite vertex` (for non-manifold edges the **last** triangle wins).
The vertex order `(min, max)` must match `flex_edge` because the bending coefficients are not symmetric under swapping `v0,v1`.

### 4.4 `ComputeBending` (2D discrete shell bending: cotangent operator + cubic shells)

Only for `dim==2 && elastic2d∈{1,3} && !interpolated`, `young>0`: `bending.assign(17·nedge, 0)`, `D = E t³/(12(1−ν²))` (with `t` = thickness after gauge; `thickness ≤ 0` would already have errored), for each edge `ComputeBending(b + 17e, vertxpos, flaps[e].vertices, D)`. Then if `stressyield>0 && stiffness non-empty`: `stiffness[6] = thickness` (stiffness is non-empty here because `young>0 && dim>1` always allocates the zero record).

Per edge with `v = {v0, v1, v2, v3}` (skip if `v3 == −1`: the 17 values stay 0):

```
cot(x, a, b, c) = ((x_b−x_a)·(x_c−x_a)) / |(x_b−x_a)×(x_c−x_a)|     (cotangent of angle at a)
a01 = cot(v0; v1, v2)     a02 = cot(v0; v3, v1)
a03 = cot(v1; v2, v0)     a04 = cot(v1; v0, v3)
c   = { a03 + a04,  a01 + a02,  −(a01 + a03),  −(a02 + a04) }        (Wardetzky et al. 2007)
A   = area(v0,v1,v2) + area(v1,v0,v3)        (areas = |cross|/2 via sqrt(dot))
k   = D / A
e0 = x1−x0, e1 = x2−x0, e2 = x3−x0, e3 = x2−x1, e4 = x3−x1
t0 = −(a03 e1 + a01 e3) ;  t1 = −(a04 e2 + a02 e4)
cosθ = −(t0·t1) / |e0|²
b[4i+j] += c_i c_j cosθ k            (i,j = 0..3)
b[16]    = ((e0×e1)·e2) (a01−a03)(a04−a02) k / |e0|³                 (Garg et al., "Cubic Shells")
```
Runtime (spec 09 §7.2): `spring_i = Σ_j b_ij x_j + (b16 + plastic_e)·∂T/∂x_i` with `T = (e0×e1)·e2` (triple product of current flap edges; `∂T/∂x1 = e1×e2`, `∂T/∂x2 = e2×e0`, `∂T/∂x3 = e0×e1`, `∂T/∂x0 = −Σ`), scattered with factor −1, i.e. energy `½ Σ b_ij x_i·x_j + b16 T`. `b16` vanishes for flat rest flaps (`e2` in the plane of `e0,e1`). Normalisation: bending a flat sheet into a cylinder of curvature κ has energy density `Dκ²/2`.

Edge plasticity uses `b23 = b[17e+11]` (the (2,3) entry, = `c2 c3 cosθ k`) as the "dihedral stiffness" (§9.3).

### 4.5 `ComputeRodBending` (1D, Bergou et al. "Discrete Elastic Rods", no twist)

Precondition: `dim==1 && !interpolated && !rigid && young>0`. Second moment of area:
- circle: `radius>0` required, `I = π r⁴ / 4`;
- box: `sectionsize>0` required, `w = 2·sectionsize[0]`, `h = 2·sectionsize[1]`, `I = min(w h³, h w³)/12` (weak axis, isotropic model).
`EI = E·I`.

Algorithm:
1. Vertex adjacency from `edge` (in edge order; each vertex lists neighbours in insertion order).
2. For each vertex `v` with exactly 2 neighbours `(a = nb[0], b = nb[1])`: `RodCurvatureBinormal(x_a, x_v, x_b)`:
   ```
   e0 = x_v − x_a ;  e1 = x_b − x_v ;  l0 = |e0| ;  l1 = |e1| ;  den = l0 l1 + e0·e1
   fail if l0 < mjMINVAL || l1 < mjMINVAL || den < mjMINVAL   → compile error "degenerate 1D flex hinge at vertex v"
   kb = 2 (e0 × e1) / den        (|kb| = 2 tan(φ/2))
   ```
   `stiffness[v] = EI/(l0 + l1)`; `curvature[v] = |kb|` if `|kb| > kRodStraightTol (1e-4)` else 0.
3. `npair[v]` = number of edges whose both endpoints are curved.
4. Output `bending = [0]`, then for each edge `(v1, v2)` with both curved: `v0` = the neighbour of v1 other than v2, `v3` = neighbour of v2 other than v1; `kb1 = kb(v0,v1,v2)`, `kb2 = kb(v1,v2,v3)`; frame `RodEdgeFrame(x1, x2)`:
   ```
   t = normalize(x2 − x1); axis = (−0.5 < t_y < 0.5) ? y : z; a = e_axis
   u = normalize(a − (t·a) t) ;  v = t × u
   ```
   record `[v0, v1, v2, v3, k(v1)/npair(v1), k(v2)/npair(v2), kb1·u, kb1·v, kb2·u, kb2·v]`, `bending[0]++`.
5. Then each 2-neighbour vertex with `npair==0`: `[nb0, v, nb1, −1, k(v), 0, curvature(v), 0, 0, 0]`, `bending[0]++`.
6. If `bending[0]==0` the vector is cleared (no bending address).

Records are stored as doubles (vertex ids as exact doubles). Runtime energies (spec 09 §7.3): single `k(|kb| − kb0)²`; pair `min_α Σ_i w_i |ω_i − R(α) ω0_i|²`.

### 4.6 Other `young>0`-gated allocations
`if (!interpolated && dim > 1) stiffness.assign((dim==3 ? 24 : 21)·nelem, 0)` — so every non-interpolated 2D/3D flex with `young>0` has a stiffness record even when it only bends (`elastic2d==1`) or has neither (`elastic2d==0` dim 2: all zeros). Runtime gates on `k[0] != 0`.

---

## 5. Shape functions and quadrature (shared by compiler and runtime)

### 5.1 1D Lagrange bases on `[0,1]` (`phi`/`dphi` compiler, `mju_flexPhi/Dphi` runtime)

| order | φ0 | φ1 | φ2 | φ0' | φ1' | φ2' |
|---|---|---|---|---|---|---|
| 1 | `1−s` | `s` | — | `−1` | `1` | — |
| 2 | `2s²−3s+1` | `4(s−s²)` | `2s²−s` | `4s−3` | `4(1−2s)` | `4s−1` |

(Compiler errors for other orders; runtime returns 0 for out-of-range i with order 2, and treats any order ≠1 as order 2.) Tensor product node numbering inside a cell: `idx = li·(o+1)² + lj·(o+1) + lk` (x slowest, z fastest); face elements: `idx = l0·(o+1) + l1` (slow in-plane axis `na0 = (n+1)%3`, fast `na1 = (n+2)%3`).

At `s = 0.5`: order 1: `φ = (0.5, 0.5)`, `φ' = (−1, 1)`; order 2: `φ = (0, 1, 0)`, `φ' = (−1, 0, 1)` (Warp uses these closed forms).

### 5.2 `quadratureGaussLegendre(points, weights, n, a, b)`
`p0 = (a+b)/2`, `h = (b−a)/2`; points `p0 + h x_i`, weights `h w_i` with:
- n=1: `x={0}`, `w={2}`
- n=2: `x=∓1/√3`, `w={1,1}` (order: −, +)
- n=3: `x={0, −√(3/5), +√(3/5)}`, `w={8/9, 5/9, 5/9}` (**center first**)
- n=4: `x1 = √((3−2√(6/5))/7)`, `x2 = √((3+2√(6/5))/7)`; order `{−x2, −x1, x1, x2}`, `w = {(18−√30)/36, (18+√30)/36, (18+√30)/36, (18−√30)/36}`
- n=5: `x1 = √(5−2√(10/7))/3`, `x2 = √(5+2√(10/7))/3`; order `{0, −x1, x1, −x2, x2}`, `w = {128/225, (322+13√70)/900, same, (322−13√70)/900, same}`
- n=8: `kX = {0.183434642495649805, 0.525532409916328986, 0.796666477413626740, 0.960289856497536232}`, `kW = {0.362683783378361983, 0.313706645877887287, 0.222381034453374471, 0.101228536290376259}`; order `{−x0, +x0, −x1, +x1, …}`
- else `mju_error`.
The point order matters for floating-point accumulation order of K.

### 5.3 `mju_defGradient(res, p, dof, order)` (runtime)
```
res = 0 (3x3 row-major)
idx = 0
for i, j, k in 0..order (nested, k fastest):
   g = ( φ'_i(p0) φ_j(p1) φ_k(p2),  φ_i(p0) φ'_j(p1) φ_k(p2),  φ_i(p0) φ_j(p1) φ'_k(p2) )
   res[r][c] += dof[3·idx + r] · g[c]
   idx++
```
i.e. `F = Σ_n x_n ⊗ ∇_ξ N_n` in parametric (unit-cell) coordinates (not divided by cell size).

### 5.4 `mju_cellLookup(coord, cellnum, order, cell, local, nodeindices)`
Natural cell `ci = clamp(floor(coord0·cx), 0, cx−1)` etc. If `cell ≥ 0` and differs from the natural cell: use `cell` (`ck = cell%cz`, `cj = (cell/cz)%cy`, `ci = cell/(cy cz)`) and `local = clip(coord·c − ci, −0.02, 1.02)` (extrapolation, `kMaxExtrap = 0.02`); else `local = clip(coord·c − ci, 0, 1)`. Node indices in cell numbering order; returns `(order+1)³`. Used with `flex_vert0` + `flex_vertcell` to interpolate vertex positions from nodes.

---

## 6. Interpolated flex precomputation (`dof="trilinear" | "quadratic"`)

### 6.1 Element sets
- Volume (`elastic2d==0`, `flex_interp>0`): cells `fe = ci·cy·cz + cj·cz + ck`, nodes per element `npe=(o+1)³`.
- Shell (`elastic2d≠0`, `flex_interp<0`): 6 box faces in order `x=0, x=max, y=0, y=max, z=0, z=max`; sizes `{cy cz, cy cz, cx cz, cx cz, cx cy, cx cy}`; `normal = {0,0,1,1,2,2}`; `count1 = {cz,cz,cx,cx,cy,cy}` (fast in-plane), `count0 = {cy,cy,cz,cz,cx,cx}` (slow); fixed grid index along normal `{0, cx·o, 0, cy·o, 0, cz·o}`. Face element `fe` → face id by cumulative sizes, `within = fe − cumul`, `q0 = within / count1`, `q1 = within % count1`; node `g[n]=fixed, g[na0]=q0·o+l0, g[na1]=q1·o+l1`, global `g0·ny·nz + g1·nz + g2`.

### 6.2 `ComputeLinearStiffness` (3D hex cell, full cell path)
Inputs: `pos` = cell node positions in unrotated frame (`nodexpos_local`), assumed axis-aligned box. `nbasis = o+1`, `n = nbasis³`, `ndof = 3n`.
```
dx,dy,dz = pos[n−1] − pos[0]   (corner-to-corner),  detJ = dx dy dz,  invJ = (1/dx, 1/dy, 1/dz)
la = Eν/(1+ν)/(1−2ν) ;  mu = E/(2(1+ν))         (full 3D λ — not halved)
GL points/weights: n_q = nbasis on [0,1]
for ps, pt, pu (nested):
   dvol = w_s w_t w_u detJ
   F[dof] = (φ'_bx(s) φ_by(t) φ_bz(u), φ_bx φ'_by φ_bz, φ_bx φ_by φ'_bz)   (bx,by,bz nested)
   for i, j in 0..n−1, k, l in 0..2:
      du = e_k ⊗ (invJ ∘ F[i]),  dv = e_l ⊗ (invJ ∘ F[j])
      K[ndof(3i+k) + 3j+l] −= la tr(du) tr(dv) dvol
      K[ndof(3i+k) + 3j+l] −= 2 mu tr(sym(du) sym(dv)) dvol
```
With `a = ∇N_i`, `b = ∇N_j` (physical): `tr(du)tr(dv) = a_k b_l`, `2 tr(sym du sym dv) = δ_kl (a·b) + a_l b_k`. So per quadrature point
$$ K^{\text{stored}}_{(ik),(jl)} \mathrel{-}= \big[\lambda a_k b_l + \mu(\delta_{kl}\, a\cdot b + a_l b_k)\big]\, dV $$
(two separate subtractions, in that order). Sign: stored = −K_phys so `K_stored · u` is the restoring force.

### 6.3 FCM path (`alphas` non-empty, cut cells)
`nq_axis = kFcmSubdiv·nbasis = 2(o+1)` composite points: GL with `nbasis` points on `[0,½]` then on `[½,1]` (sub-interval major). Tables `phi_tab[b·nq+p]`, `dphi_tab`. Per quadrature point `q_idx` (ps, pt, pu nested): `dvol = w_s w_t w_u detJ α_q`;
```
F[dof] = ( invJx φ'_bx(s) φ_by(t) φ_bz(u),  φ_bx(s) invJy φ'_by(t) φ_bz(u),  φ_bx φ_by invJz φ'_bz(u) )
          (products formed as st_x = dph_s·ph_t, st_y = ph_s·dph_t, st_z = ph_s·ph_t·invJz, then ×ph_u / ×dph_u)
G[i·n+j][a][b] += (dvol F[i][a]) F[j][b]     for j ≥ i only
```
Then `G[j·n+i] = G[i·n+j]ᵀ`, and
```
dot_g = mu (G00 + G11 + G22)
K[ndof(3i+k) + 3j+l] −= la G[k][l] + mu G[l][k] + (k==l ? dot_g : 0)
```
(Same bilinear form, different accumulation order from §6.2 → not bit-identical to the full-cell path even for α≡1.)

### 6.4 `ComputeCellEmpty` — cell emptiness and FCM quadrature weights

Inputs: element vertices `vpos` (grid-aligned frame), `elems`, `ne`, `fdim`, `minmax` (node AABB). Outputs per cell: `cell_empty`, `cell_full`, `cell_vol`, `cell_alphas` (empty vector = full cell; else `nq_cell = nq_axis³` weights).

Initialisation: `empty=false, full=true, vol=1, alphas={}`. Return early if `ne ≤ 0`. Cache lookup by key (§6.4.6).

Helpers: `cellIdxLo(c) = clamp(floor((c − 1e-9·d − lo)/d · nc), 0, nc−1)`, `cellIdxHi` with `+1e-9·d` (return 0 if `d ≤ 0`).

**Degenerate box** (`dx|dy|dz ≤ mjMINVAL`): AABB-overlap marking of `has_element`; if `ncells>1`: `empty = !has`, `full = has`, `vol = has ? 1 : 0`. Return.

Otherwise `h = d/c` per axis; core margin `m = (c>1) ? 0.02 h : 0`. `order = max(1, spec.order)`, `nmom = 2·order+1`, composite points `qpts/qwts` as §6.3, `H = BuildLegendreMomentProjector`, Duffy rules: `n_duffy = order==1 ? 5 : 8` GL points on [0,1], cap rule `n_duffy_cap = 2·order+1` points.

**`TriangleIntersectsBox(v0,v1,v2,bmin,bmax)`** — SAT test with box half-extents inflated by `eps = 1e-9·max(e0,e1,e2,1e-6)`: (1) 3 box axes, (2) 9 axes `e_i × f_j` (`f = {v1−v0, v2−v1, v0−v2}`, radius `e_{i1}|a_{i1}| + e_{i2}|a_{i2}|`), (3) triangle normal `f0×f1` (`|n·u0| > r` → separate). `SegmentIntersectsBox(v0,v1) = TriangleIntersectsBox(v0,v1,v1)`.

**`PointInTet`**: inside AABB `[lo,hi]` (inflated by `bbox_eps = 1e-7·max(h)`), barycentrics `l1..l3 = invM·(x−v0)`, `l0 = 1−Σ`; inside iff all `≥ −1e-7`. `TetIntersectsBox` = any of 4 faces intersects box, or (valid tet) box center in tet.

**`moments_to_cell_quad(M, c)`** (M = exact shifted-Legendre moments `M[a][b][c] = ∫_{cell∩solid} P_a(2s−1)P_b(2t−1)P_c(2u−1)` in unit-cell coordinates):
```
vol = M[0][0][0]; empty=false
if vol ≥ 1 − 1e-7:   full, vol=1, alphas={}
elif vol ≤ 1e-12:    full=false, vol=1e-6, alphas = 1e-6 everywhere
else:
   α_q = 1e-6 + (1 − 1e-6) · Σ_{a,b,c} H[ps][a] H[pt][b] H[pu][c] M[a][b][c]     (separable contraction: u, then t, then s)
   full=false, vol = 1e-6 + (1−1e-6)·min(1, vol)
```
**`BuildLegendreMomentProjector`**: `V[r][k] = P_r(2q_k − 1)` (r < nmom, k < nq_axis); Gram `G = V W Vᵀ` (W = composite weights); invert by Gauss–Jordan with partial pivoting on `[G | I]`; `H[k][r] = Σ_c V[c][k] G⁻¹[c][r]` (weighted minimum-norm moment fitting: `Σ_k w_k α_k P_r(q_k) = μ_r`).

**`EvalShiftedLegendre1D(s)`**: `x = 2s−1`, `P0=1, P1=x, P2=(3x²−1)/2, P3=x(5x²−3)/2, P4=(35x⁴−30x²+3)/8, P5=x(63x⁴−70x²+15)/8`; antiderivatives `∫_0^s P_r = {s, (P2−P0)/6, (P3−P1)/10, (P4−P2)/14, (P5−P3)/18}`.

**`AccumulateClippedTetMoments(v0..v3 (unit-cell coords, positively oriented), boundary_mask, sign_weight, …, M)`** — divergence theorem with the field `(∫_0^s P_a, 0, 0)·P_b(t)P_c(u)`, so only the x-component of face normals matters:
- early out if the tet AABB misses `[0,1]³` by `kClipEps = 1e-12`.
- outward faces `(v0,v2,v1), (v0,v1,v3), (v0,v3,v2), (v1,v2,v3)`; face `f` skipped if not in `boundary_mask` and not lying on `s≥1−ε`; skipped if out of range; `n_x = (b−a)_y(c−a)_z − (b−a)_z(c−a)_y`, skip if `|n_x| ≤ 1e-16`. Faces with all `s ∈ [1−ε, 1+ε]` are "on the s=1 cap" (no s clipping). Otherwise clip by Sutherland–Hodgman (`ClipPolygonAxis`, sign convention `sign·(p_axis − bound) ≤ 0` kept, intersection snapped to the bound) against `s≥0, s≤1`, then `t∈[0,1]`, `u∈[0,1]` (buffers of 16 points).
- integrate polygon as fan `(p0, p_m, p_{m+1})` with Duffy map: for GL α (weights wα), β: `c0 = 1−α, c1 = α(1−β), c2 = αβ`, weight `w = sign_weight · n_x · α · wα · wβ` (n_x of the fan triangle); point `(s,t,u) = c0 p0 + c1 p1 + c2 p2`; accumulate `M[a][b][c] += w ∫_0^s P_a · P_b(t) P_c(u)`; on the s=1 cap only `a=0` (∫_0^1 P_0 = 1, others 0) with the cap rule.
- If the tet strictly straddles `s=1` (`min_s < 1−ε`, `max_s > 1+ε`): build cap polygon from edge intersections with `s=1` (1-vs-3: triangle; 2-vs-2: quad in cyclic order `(le0,gt0),(le0,gt1),(le1,gt1),(le1,gt0)`), orient to positive area in (t,u) (reverse if negative), clip to t,u ∈ [0,1], integrate as cap (closes the volume cut by the s ≤ 1 clip).

**Volume elements (fdim 3)** — `integrate_tet_collection`:
1. Each tet: AABB; `det = d1·(d2×d3)`; valid if `|det| > 1e-14·hx hy hz`: store with vertices reordered for positive orientation (swap v1,v2 if `det<0`), `invM` rows `c1/det, c2/det, c3/det` (`c1=d2×d3, c2=d3×d1, c3=d1×d2`), inflated bbox.
2. For candidate cells (index range from AABB): if `TetIntersectsBox(cell)`: `has_element`; if it intersects the shrunk core box (`±m`): `has_core`; record tet in `cell_tets[c]` if valid.
3. `active = any_core ? has_core : has_element` (sliver pruning).
4. No valid tets: `empty = (ncells>1) && !active`, `full = !empty`, `vol = empty?0:1`; return.
5. Shared interior faces cancellation: weld vertices by quantising `(p − lo)·1e9/max(dx,dy,dz,1e-6)` (`llround`); face key = sorted welded ids with permutation sign; faces seen exactly twice with opposite signs are removed from both tets' `boundary_mask` (bit f for face list `{0,2,1},{0,1,3},{0,3,2},{1,2,3}`).
6. Per cell: inactive → `ncells>1`: empty (vol 0); single cell: not empty, vol `1e-6`, alphas all `1e-6`. Active but no boundary face of its tets intersects the cell → full. Else accumulate moments of all its tets (in local `[0,1]³` coordinates `(v − bmin)/h`) with their boundary masks, `moments_to_cell_quad`.

**Surface elements (fdim 2)**:
1. `IsClosedSurfaceMesh`: ≥4 triangles; weld vertices (quantisation as above with `minmax`); skip degenerate (repeated welded ids); ≥4 non-degenerate; every undirected edge must have even count ≥2 and zero signed orientation sum.
2. If closed: cone determinants about the box center `o`: `det_e = (a−o)·((b−o)×(c−o))`, `total_det = Σ`; `has_pos/has_neg` with threshold `±1e-14·hx hy hz`; if `|total_det| ≤ 1e-12·hx hy hz` → treat as not closed.
3. Closed and star-shaped (no mixed signs): `integrate_tet_collection` with tets `(o, a, b, c)`; cache; return.
4. Else (open, or closed non-star-shaped): per triangle candidate cells by SAT → `has_element`, `has_core`, `cell_tris`. Build `FastWindingBVH` (§6.4.5). Cell loop (ci, cj, ck order):
   - If `!has_element || (!has_core && ncells>1)`: decide `inside`: propagate from the previous cell along z (then y, then x) if that neighbour also had no element (`inside = cell_full[neighbour]`), else winding number at the cell center `|w| ≥ 0.5 − 1e-5`. No element: inside → full (vol 1, `any_inside`), outside → empty if `ncells>1` (vol 0) else vol `1e-6` + alphas `1e-6`. Sliver (`has_element && !has_core`) outside → pruned (empty).
   - Closed: mark cut cell, continue (moments later).
   - Open fallback (point sampling): for 2×2×2 sub-boxes test triangle intersection; uncut sub-box: winding at its center; per composite Gauss point: inside = sub-box result or (cut) winding at the point; `α = 1` if inside else `1e-6`; `vol = Σ w α`; all inside → full, else store alphas/vol.
   - Closed non-star: per triangle tet `(o, a, b, c)` (or `(o, b, a, c)` if `det<0`, with `sign_w = positive ? orient : −orient`, `orient = sign(total_det)`, near-zero det treated positive) accumulated into each cut cell's moments with `boundary_mask = 0x8` (only face (v1,v2,v3) = the surface triangle); then `moments_to_cell_quad`.
   - If no cell was found inside (`!any_inside`): fall back to `active = any_core ? has_core : has_element`, full occupancy, alphas cleared.
5. **Line elements (fdim 1)**: segment SAT, `active` rule, full occupancy.
6. Cache insert.

**6.4.5 `FastWindingBVH`** (Barill et al. 2018): per triangle area-weighted normal `A_t = ½(b−a)×(c−a)`, centroid; node: area-weighted centroid `p` (or plain mean if total area ≤1e-30), `r_max` = max vertex distance from p, dipole `A = ΣA_t`, quadrupole `Q[j][k] = Σ (c_t − p)_j A_t,k`; split if `count > 16` on the largest centroid-extent axis (ties: y if `ext1 > ext0 && ext1 ≥ ext2`, z if `ext2 > ext0 && ext2 > ext1`), median by `nth_element`, only if extent > 1e-20. Evaluate: if `r² > 4 r_max²` and `r² > 1e-30`: `(d·A/r³ + (trQ − 3 dᵀQd/r²)/r³)/(4π)` with `d = p − x`; leaf: Σ `TriangleWinding` (solid angle/4π: `2 atan2(ra·(rb×rc), |ra||rb||rc| + (ra·rb)|rc| + (rb·rc)|ra| + (rc·ra)|rb|)/(4π)`; returns 1 ("on surface") if `labc < 1e-30` or the triple product `≤ 1e-12 labc` with denominator `≤ 1e-12 labc`; returns 0 if coplanar otherwise); an on-surface hit short-circuits the whole evaluation to 1.

**6.4.6 Caches** (global `mj_getCache`): FCM key = hash (boost-style `h ^= v + 0x9e3779b9 + (h<<6) + (h>>2)`) of `fdim, order, cellcount, ne`, and per element its (tet: sorted) vertex ids and coordinates normalised to the box and quantised `llround(u·1e9)`. Stiffness key: `young, poisson, dim, order, cellcount, elastic2d≠0, thickness, has_strain_eq`, vertex AABB extents, `nodexpos_local`, `vert_`, `elem_`, `cell_empty`, `cell_vol`, `cell_alphas`, `shell`. Cached values are byte-identical to recomputation, so caching does not affect results (except cross-model hash collisions).

### 6.5 Node masses — `NodeMassWeight(gi, gj, gk)` (volume mode)
```
MassBasis1D(s, l, o): o=1: (1−s, s);  o=2: l=0: ½(1−s)²,  l=1: ½ + s(1−s),  l=2: ½ s²     (positive, exact Simpson/trilinear integrals)
m1d[l][p] = qw_p · MassBasis1D(q_p, l, o)    (composite rule of §6.3)
full_w = o==1 ? {½, ½} : {1/6, 2/3, 1/6}
for each non-empty cell c adjacent to the node (ci ∈ [max(0, gi==0 ? 0 : (gi−1)/o), min(cx−1, gi/o)], same for j,k):
   li = gi − ci·o, …;  w_full = full_w[li] full_w[lj] full_w[lk]
   full cell (no alphas): weight += w_full
   cut cell: c_w = Σ_q m1d[li][ps] m1d[lj][pt] m1d[lk][pu] α_q;  weight += max(c_w, 0.02·w_full)
```
`NodeInEmptyCells` uses the same adjacent-cell range and returns true iff all adjacent cells are empty (false if `cell_empty` empty or order ≤ 0).

### 6.6 Interpolated stiffness assembly (`Compile`, step 17)
`need = interpolated && (has_strain_eq || (young>0 && (elastic2d==0 || elastic2d≥2)))`. If not cached:
- `K_young = has_strain_eq ? 10.0 : young`, `K_poisson = has_strain_eq ? 0.3 : poisson` (eigenvectors are geometry-only; the strain constraint uses `sqrt(λ)` scaled modes — so the absolute scale 10 matters for constraint residual units, see constraint spec).
- `stiffness.assign(nelem_fe · ndof²)` zeros.
- Shell: gather face nodes (`nodexpos_local`) → `ComputeLinearStiffness2D`.
- Volume: skip empty cells (block stays 0). Cell with alphas → `ComputeLinearStiffness(…, alphas)`. Full cell → `K_full_cell` computed **once** from the first full cell's positions and copied to every full cell (assumes a uniform grid).
- Strain equality: block zeroed, then
  - shell: `neig = EigendecomposeStiffness(K, out, ndof)`; `ws = ComputeWarpStiffness`; if `ws > 0`: warp mode written at `out + 1 + neig·ndof`, scaled by `√ws`, `out[0] = neig+1`;
  - full volume cell: eigendecomposition done once and copied; cut cell: per cell.
- Else raw `K_elem` copied.

### 6.7 `ComputeLinearStiffness2D` (membrane face element)
```
axis0 = (n+1)%3, axis1 = (n+2)%3;  d0 = pos[npe−1][axis0] − pos[0][axis0], d1 likewise (error if 0)
detJ = d0 d1;  λ* = Eν/(1−ν²);  μ = E/(2(1+ν))
GL nbasis points on [0,1]; for ps, pt: dvol = w_s w_t detJ t
   F[dof] = (φ'_b0(s) φ_b1(t), φ_b0(s) φ'_b1(t))
   for i, j; k ∈ {axis0, axis1}, l ∈ {axis0, axis1}:
      du = e_k ⊗ (F[i][0]/d0 e_axis0 + F[i][1]/d1 e_axis1), dv likewise
      K[ndof(3i+k)+3j+l] −= λ* tr(du) tr(dv) dvol ;  −= 2μ tr(sym du sym dv) dvol
```
Only in-plane displacement components (no transverse shear); the normal-direction rows/columns are zero — hence for x-normal faces (faces 0, 1) `K[0] = 0` (the runtime therefore must not use `k[0]==0` to skip shell elements).

**`ComputeWarpStiffness`**: `d0 = |Δpos[axis0]|`, `d1 = |Δpos[axis1]|`; 0 if either `< 1e-30`; `D = E t³/(12(1−ν²))`; returns `D (1−ν)·4/(d0 d1)`.

**`ComputeWarpMode`**: `w[3·node + n] = (1−2s)(1−2t)` with `s = b0/(nbasis−1)`, `t = b1/(nbasis−1)`; orthogonalise against 6 rigid modes (3 translations; rotations about the node centroid: `(0,−rz,ry), (rz,0,−rx), (−ry,rx,0)`) orthonormalised by modified Gram–Schmidt (normalise if `‖·‖² > 1e-20`); project out; normalise (if `>1e-20`).

**`EigendecomposeStiffness(K, out, n)`** (`mjuu_eigendecompose`: cyclic Jacobi, ≤200 sweeps, stop when `Σ_{i<j} K_ij² < 1e-24` (tol 1e-12 squared), skip pairs with `|a_pq| < 1e-15`; rotation `τ = (a_qq−a_pp)/(2a_pq)`, `t = sign(τ)/(|τ|+√(1+τ²))`): `max|eig|`, threshold `1e-8·max`; for each `i` (ascending index, not sorted) with `λ_phys = −eig_i > threshold`: `out[1 + neig·n + j] = √λ_phys · V[j][i]`; `out[0] = neig`.

### 6.8 `ComputeInterpBending` (interpolated shell bending)
Enabled if `interpolated && elastic2d∈{1,3} && thickness>0 && young>0`. `D = E t³/(12(1−ν²))`.

Edge enumeration: for faces f=0..5, `q0 < count0[f]`, `q1 < count1[f]`, `within_A = q0·count1 + q1`, `fe_A = cumul[f] + within_A`; for `dir ∈ {0,1}`, `side ∈ {−1,+1}`: `get_neighbor`:
```
nax = fid/2, sign_f = fid%2, a0 = (nax+1)%3, a1 = (nax+2)%3
local_A = (dir==0 ? (side>0 ? 1 : 0) : 0.5,  dir==1 ? (side>0 ? 1 : 0) : 0.5)
q_nb = (dir==0 ? q0 : q1) + side
internal (0 ≤ q_nb < (dir==0 ? count0 : count1)): same face, local_B = mirrored (0↔1, 0.5 stays)
boundary: ax = (dir==0 ? a0 : a1); fid_B = 2·ax + (side>0);
   q_run = (dir==0 ? q1 : q0);  q_boundary = sign_f ? cells[nax]−1 : 0
   dir==0: (q0_B, q1_B) = (q_run, q_boundary), local_B = (0.5, sign_f ? 1 : 0)
   dir==1: (q0_B, q1_B) = (q_boundary, q_run), local_B = (sign_f ? 1 : 0, 0.5)
   within_B = q0_B · count1[fid_B] + q1_B
```
Keep the edge if `fe_A < fe_B`. Record count `bending[0] = nedges`, size `1 + 10·nedges`.

Per edge: gather both faces' nodes; `t1 = Σ x dφ⊗φ`, `t2 = Σ x φ⊗dφ` at `local`, `n = t1×t2` normalised (`len < 1e-12` → **skip, leaving an all-zero record that is still counted**); `dn0 = n_A − n_B`; along/across lengths: if `local[0] == 0.5` the edge runs along ξ: `l = |t1|`, `h = |t2|`; else `h = |t1|`, `l = |t2|`; `stiffness = D·(l_A+l_B)/2 / max((h_A+h_B)/2, 1e-12)`. Record `[fe_A, fe_B, lA0, lA1, lB0, lB1, stiffness, dn0]`. Runtime energy (spec 09 §7.5): `E = (k/2)|Δn − Δn0|²` with `Δn` evaluated in corotated face frames.

---

## 7. Reference geometry for interpolated flexes

### 7.1 Why
Runtime corotational forces use `u = R⁻¹ x − x0` with `x0 = flex_node0`; at rest `R` must equal the grid rotation `R0`, and `ComputeLinearStiffness` assumes axis-aligned cells (diagonal Jacobian). Hence stiffness and `node0` are computed in the grid frame.

### 7.2 `ComputeUnrotatedNodePositions(nodexpos, R0_out)`
- First non-empty cell `(ref_ci, ref_cj, ref_ck)` (lexicographic; all cells if `cell_empty` empty).
- Corner nodes `g000, g100 (+order in i), g010, g001`.
- `R0` rows: `normalize(x[g100]−x[g000])`, `normalize(x[g010]−x[g000])`, `normalize(x[g001]−x[g000])`.
- Orthonormality check: `|R0_a·R0_b − δ_ab| ≤ 1e-8` else error "flex grid rotation R0 is not orthonormal".
- `nodexpos_local[i] = R0 · nodexpos[i]` (`mjuu_mulvecmat`: each component is a row of R0 dotted with x).
- Non-interpolated: identity copy.

Note: absolute translation is kept (node0 is in a rotated *world* frame, not cell-local).

### 7.3 Runtime gather (`mju_flexGatherState`)
Node position `x_i = xpos[b]` if `flex_centered` or `flex_node[i]==0`, else `xpos[b] + xmat[b]·node[i]`; velocity `v = v_com(b) + ω(b) × (x_i − xipos[b])` via `mj_objectVelocity(..., flg_local=0)`. Shell mode: interior node positions/velocities reconstructed by transfinite interpolation (`mju_shellTrackInterior`, needs ≥3 nodes per axis).

### 7.4 Corotational rotation, volume cell (`mju_flexGatherCellState` + `flexInterpRotation`)
```
dref = x0[g((ci+1)o, (cj+1)o, (ck+1)o)] − x0[g(ci·o, cj·o, ck·o)]     (rest cell diagonal; dref = (1,1,1) if no x0)
F = mju_defGradient(p = (½,½,½), cell nodes x_c, order)
for col: if dref[col] > mjMINVAL: F[:,col] /= dref[col]
quat = mju_flexPolarRotation(F) ;  quat = conj(quat)        (quat represents R⁻¹)
```
### 7.5 Corotational rotation, shell face (`mju_flexGatherFaceState`)
With rest data: `d0 = x0[last][na0] − x0[first][na0]`, `d1` likewise (first/last = local node 0 and npe−1); `t1 = Σ x φ'(½)φ(½)`, `t2 = Σ x φ(½)φ'(½)`; `t1 /= d0` if `d0 > mjMINVAL`, `t2 /= d1` likewise; `n = t1 × t2`; matrix columns `col[na0] = t1, col[na1] = t2, col[normal] = n`; `quat = conj(polar(mat))`. Without rest data: `mju_flexInterpRotation2D` (same without division).

### 7.6 `mju_flexPolarRotation(quat, mat)` (warm-started Müller rotation extraction)
```
quat = (1,0,0,0)
c0 = column 0, c1 = column 1
if tr(mat) ≤ 0 and normalize3(c0) > mjMINVAL:
    c1 −= (c0·c1) c0
    if normalize3(c1) > mjMINVAL:
        c2 = c0 × c1;  quat = mju_mat2Quat([c0 c1 c2])
mju_mat2Rot(quat, mat)
```
(`mju_normalize3` returns the pre-normalisation norm.) The warm start avoids the identity saddle (e.g. 180° rotations).

**`mju_mat2Rot(quat, mat)`** (Müller, Bender, Chentanez, Macklin 2016):
```
for iter in 0..499:
    R = quat2Mat(quat)
    ω = (R_c0 × M_c0 + R_c1 × M_c1 + R_c2 × M_c2) / (|R_c0·M_c0 + R_c1·M_c1 + R_c2·M_c2| + mjMINVAL)
    w = normalize3(ω)            (ω becomes unit axis)
    if w < rotEPS: break          rotEPS = 1e-9 (double), 1e-6 (mjUSESINGLE)
    quat = axisAngle2Quat(ω, w) * quat ;  normalize4(quat)
return iter
```
Summation order inside ω: `(v1 + v2) + v3`.

### 7.7 Runtime use of `K` (summary; full loop in spec 09 §7.4)
Per element: rotate `x_e, v_e` by `quat` (R⁻¹); `u = x_e^rot − x0_e`; subtract plastic strain (§9.4); `f = K u`, `d = K v`; rotate back by `conj(quat)` (R); scatter; damper force scaled by `flex_damping` (not ÷timestep).

### 7.8 Plastic displacement correction (volume)
`dx0_n = x0_n − x0_{node 0 of element}`; `u_n −= S dx0_n` with `S = [[sp0, sp3/2, sp4/2],[sp3/2, sp1, sp5/2],[sp4/2, sp5/2, sp2]]`. Shell: in-plane only: `u_n[a0] −= sp0 Δ0 + ½sp2 Δ1`, `u_n[a1] −= ½sp2 Δ0 + sp1 Δ1`.

### 7.9 Interpolated vertex positions
Vertices are evaluated from nodes with `mju_cellLookup(flex_vert0, flex_cellnum, order, flex_vertcell)` + `mju_interpolate3D` (basis order `i&1 → z`, `i&2 → y`, `i&4 → x` for order 1 in `mju_evalBasis`; `mju_evalBasisArray` uses x-outer/z-inner product, matching node numbering) — see kinematics spec.

### 7.10 `vert0_`, `vertcell_`, `size` (Compile, step 18)
- Interpolated: `vertxpos_local = R0·vertxpos`; `lo/hi` = AABB of `nodexpos_local` (init ±1e30); `size = (hi−lo)/2`; `vert0 = (x_local − lo)/(hi−lo)` per axis if extent > mjMINVAL else 0.5. `vertcell`: natural cell `clamp(floor(coord·c), 0, c−1)`; if that cell is empty, nearest non-empty cell by squared distance from `u = coord·c` to the unit cell box `[n, n+1]` (strict `<`, so the lowest index wins ties).
- Non-interpolated: `bvh` = root node AABB `(center[0..2], half[3..5])` of `CreateBVH` (element AABBs inflated by `radius`; dim 3 elements with `elemlayer ≥ activelayers` excluded); `size = half − radius`; `vert0 = (x − center)/(2 size) + 0.5` if `size > mjMINVAL` else 0.5. `vertcell = −1`.

SNH relies on this: `snhProject` reconstructs reference edges as `2·size·(vert0_j − vert0_0)`; this equals the true rest edge up to rounding only when `size_k > mjMINVAL` for all axes.

---

## 8. `CreateShell` and `elemlayer`
Fragments per element (vertex lists in original element order):
- dim 1: `[v0]`, `[v1]` (slots n, n+1);
- dim 2: slot n `[v0,v1]`, slot n+2 `[v1,v2]`, slot n+1 `[v2,v0]`;
- dim 3: slot n `[v0,v1,v2]`, n+2 `[v0,v2,v3]`, n+1 `[v0,v3,v1]`, n+3 `[v1,v3,v2]`.
Each fragspec = `[sorted vertices (dim of them), elem id, original-order vertices]` (dim 1 not sorted). `std::sort` the whole list lexicographically. Sequential equal sorted keys → `connectspec (e1, e2, verts)`; runs of length 1 are border fragments → `border[elem] = true`. `shell` = concatenation of border fragments' original-order vertices in sorted fragspec order. `elemlayer`: dim<3 all 0; dim 3: border elements 0, others relaxed `layer[e1] = min(layer[e1], layer[e2]+1)` over `connectspec` until no change (init `nelem+1`).

---

## 9. Plasticity return mapping (`advanceFinish`, engine_forward.c)

Runs at the end of every integrator step (Euler/implicit via `mj_advance`, RK4 final `mj_advance` after restoring `flexvert_xpos`, `cdof`, `subtree_com` saved at the RK4 start), after `time += h` and plugin `advance`, if `m->nplastic && !DISABLED(SPRING)`. Order: joints, tendons, flexes (independent). Sleep: if `ENBL_SLEEP && ntree_awake < ntree`, skip flexes whose `mj_sleepState(mjOBJ_FLEX) != AWAKE`. `qvel` = the velocity used for position integration (argument `v`). All flex mappings use the **predicted** end-of-step positions `x⁺ = x + h·v` computed from the *pre-update* kinematics (`flexvert_xpos`, `cdof`, `subtree_com` are from the start of the step, not recomputed).

`yield = max(0, flex_yield[f])`; `plastic = d->plastic + flex_plasticadr[f]`.

### 9.1 Interpolated flexes
Skip if `edgeequality==3 || !mj_flexHasStiffness`. Node predicted positions: `x_i = xpos-based node position + h·v_i`, with `v_i = Σ_j J_j qvel_j` (body_simple==2: `Σ cdof[j][3:6] qvel_j` over the body's dofs; else walk `dof_parentid` from the body's last dof: column `cdof[j][0:3] × (x_i − subtree_com[rootid]) + cdof[j][3:6]`; bodies with `body_dofnum[weldid]==0` get no velocity; `bid<0` → 0). Per element (volume skips `k[0]==0` blocks): gather + quat as in §7.4/7.5 (from predicted positions), `displ_n = R⁻¹x_n − x0_n`.

**Shell** (`normal_axis` from fe ranges `< 2cy cz → 0`, `< 2(cy cz + cx cz) → 1`, else 2; skip if `d0 ≤ mjMINVAL || d1 ≤ mjMINVAL`):
```
t1 = Σ displ φ'(½)φ(½),  t2 = Σ displ φ(½)φ'(½)
e0  = t1[a0]/d0 − sp0 ;  e1 = t2[a1]/d1 − sp1 ;  g01 = (t2[a0]/d1 + t1[a1]/d0) − sp2
α = (e0+e1)/2 ;  α+ = max(0, α) ;  dev0 = (e0−e1)/2, dev1 = −dev0
‖dev‖² = 2 dev0² + ½ g01² ;  ε_eq = sqrt(2‖dev‖²/3 + α+²)
if ε_eq > yield: s = 1 − yield/ε_eq
    sp0 += s (dev0 + α+) ;  sp1 += s (dev1 + α+) ;  sp2 += s g01
    if sp0² + sp1² + ½ sp2² > 0.1225: (sp0,sp1,sp2) *= 0.35 / sqrt(that)
```
**Volume** (skip if any rest extent `≤ mjMINVAL`): `G = mju_defGradient((½,½,½), displ, order)`;
```
e = ( G00/dx − sp0, G11/dy − sp1, G22/dz − sp2,
      (G01/dy + G10/dx) − sp3, (G02/dz + G20/dx) − sp4, (G12/dz + G21/dy) − sp5 )
mean = (e0+e1+e2)/3 ; dev = (e0−mean, e1−mean, e2−mean, e3, e4, e5)
‖dev‖² = dev0²+dev1²+dev2² + ½(dev3²+dev4²+dev5²) ;  ε_eq = sqrt(2‖dev‖²/3)
if ε_eq > yield: sp += (1 − yield/ε_eq)·dev ;  clamp ‖sp‖ (same norm) to 0.35 (threshold 0.1225)
```
(Volume plasticity is purely deviatoric; shell adds positive mean (tensile) strain.)

### 9.2 Standard 2D/3D stretch (von Mises in edge space)
Skip if `dim < 2`. `x⁺ = flexvert_xpos + h·J qvel` (`mj_flexGather`). If `stiffnessadr ≥ 0 && k[0] != 0 && nelem > 0`: `stretch_pnum = nedge·nelem`, deformed squared lengths `q_e = |x⁺_{e1} − x⁺_{e0}|²` per flex edge. Per element `t`:
```
g_e   = L0_e²                         (flexedge_length0)
s_e   = q_e − g_e − sp_e              (elastic elongation)
K     = unpacked metric (first 21/6 numbers; SNH extra coefficients ignored)
f = K s ;  h = K g ;  g_h = g·h ;  skip if |g_h| ≤ mjMINVAL
α     = (g·f) / g_h                   (volumetric part: s_vol = α g)
s_dev = s − α g ;  f_dev = f − α h
trPK  = Σ_a g_a² K_aa + cross terms:
   2D: d01 = g0+g1−g2, d02 = g0+g2−g1, d12 = g1+g2−g0;  trPK += ½(d01² K01 + d02² K02 + d12² K12)
   3D: faces {0,1,2},{0,4,5},{1,3,5},{2,3,4}: for (a,b,c): d_ab = g_a+g_b−g_c, …; trPK += ½(d_ab² K_ab + d_ac² K_ac + d_bc² K_bc)
       opposite pairs: S0 = g0+g3, S1 = g1+g4, S2 = g2+g5; d03 = S1−S2, d14 = S2−S0, d25 = S0−S1;
       trPK += ½(d03² K03 + d14² K14 + d25² K25)
μ_t   = (trPK − g_h/dim)/(nedge − 1) ;  skip if μ_t ≤ mjMINVAL
‖dev‖² = max(0, (s_dev·f_dev)/(4 μ_t))
α+    = dim==2 ? max(0, α) : 0
ε_eq  = sqrt(2‖dev‖²/3 + α+²)
if ε_eq > yield: s = 1 − yield/ε_eq
   2D: sp_e = max(sp_e + s(s_dev_e + α+ g_e), −0.99 g_e)
   3D: p = sp + s·s_dev ;  β = (h·p)/g_h ;  p −= β g
       volume-preserving correction: find u with det G(u g + p) = det G(g):
         c3 = gramDet3D(g) ;  c0, ∇ = gramDet3D(p) ;  c1 = ∇·g ;  u = 1
         if c3 > mjMINVAL: m2 = max(0, −c1/(3c3));  if m2 > mjMINVAL:
             m = sqrt(m2) ;  R = (c3 − c0)/(2 c3 m m2)
             R ≥ 1: w = (R + sqrt(max(0, R²−1)))^(1/3) ;  u = m (w + 1/w)
             else:  u = 2 m cos(acos(max(−1, R))/3)
       sp_e = max(p_e + (u−1) g_e, −0.99 g_e)
   flex_hessian_valid[f] = 0
```
`gramDet3D(q)` (edge order of `mj_stretchEdges[1]`): `a=q0, b=q2, c=q4, d=½(q0+q2−q1), e=½(q0+q4−q5), f=½(q2+q4−q3)`; cofactors `c0=bc−f², c1=ac−e², c2=ab−d², c3=ef−cd, c4=df−be, c5=de−af`; `det = a c0 + d c3 + e c4`; gradient `(c0+c3+c4, −c3, c1+c3+c5, −c5, c2+c4+c5, −c4)`.

### 9.3 Standard 2D bending (dihedral-angle space)
If `dim==2 && bendingadr ≥ 0 && stiffnessadr ≥ 0` and `t = flex_stiffness[stiffnessadr+6] > mjMINVAL` (thickness slot, §4.4): `bend_plastic = plastic + stretch_pnum`, `κ_y = 2·yield/t`. Per flex edge with `v3 ≠ −1` and `b23 = b[17e+11] > mjMINVAL`:
```
ed0 = x⁺1−x⁺0, ed1 = x⁺2−x⁺0, ed2 = x⁺3−x⁺0 ;  n2 = ed2 × ed0 ;  n3 = ed0 × ed1
2A1 = |n3|, 2A2 = |n2| ;  l0 = flexedge_length0[e] ;  skip if 2A1·2A2 ≤ mjMINVAL or l0 ≤ mjMINVAL
ω = ed2·n3 ;  θ = atan2(ω l0, n2·n3)
θ_p = atan2(−(b16 + bp_e) l0, b23)
θ_y = ½(2A1 + 2A2)/l0 · κ_y
if |θ − θ_p| > θ_y:  θ_p = θ ∓ θ_y (toward θ);  θ_p = clip(θ_p, −1.3, 1.3);  bp_e = −(b23/l0) tan(θ_p) − b16
```
(No Hessian invalidation for bending.)

### 9.4 Consumption
Stretch plastic: `s_e = L² − L0² − sp_e` in passive forces, Hessian and solver metrics. Bend plastic: `b16 + bp_e`. Interpolated: §7.8.

---

## 10. Stable Neo-Hookean runtime (complete)

### 10.1 Force (`mj_flexPassiveStretch`, SNH branch)
Per element (record `k`, `edgevec[e] = x_{e0} − x_{e1}`):
```
s = elongation (with plastic)                       (mj_stretchElongation)
T = K s                                             (mj_stretchElasticity, metric unpacked from k[0..20])
T += 2γ ∇P(s)                                       (mj_snhCubic: ∇P = (c0+c3+c4, −c3, c1+c3+c5, −c5, c2+c4+c5, −c4) with cofactors of D(s), see §9.2 gramDet3D)
a = −edgevec0 = x1−x0, b = edgevec2 = x2−x0, c = −edgevec4 = x3−x0
∇J_1 = k23 (b×c), ∇J_2 = k23 (c×a), ∇J_3 = k23 (a×b), ∇J_0 = −(∇J_1+∇J_2+∇J_3)   (accumulated as grad0 −= grad_v, v=1..3)
J = k23 · a·(b×c)
p = 2β (J − 1)
frc[v_{e0}] −= T_e edgevec_e ;  frc[v_{e1}] += T_e edgevec_e       (mj_stretchForce, e = 0..5, endpoint 0 then 1)
frc[v] −= p ∇J_v   (v = 0..3)                       (only if spring enabled)
```
SNH elements skip the edge-velocity Rayleigh damper; instead, after the element loop, if `kD = flex_damping/h > 0` (damper enabled, h>0): `worldvel = flexGather(qvel)`, `mj_flexHessian(f)`, `dmp += −flex_damping · H · worldvel` (note: `flex_damping`, not `kD`).

### 10.2 Cached Hessian (`mj_flexHessian`), SNH element path
For each element: `edges = mj_stretchEdgeVectors`, `s = mj_stretchElongation(…, plastic)`, `snhProject(&P, k, edges, s, flex_vert0+3·vertadr, vert, flex_size+3f)`; for each local edge `e = (i,j)` oriented so that `flex_edge[id][0] == vert[i]`: `block = snhProjectedBlock(P, i, j)`, `offdiag[id] += block`. Diagonal blocks `H_ii = −Σ_{j≠i} H_ij` (translation invariance) packed `(00,01,02,11,12,22)`: for each flex edge `(v0,v1)` and `r ≤ c`: `diag[v0][id] −= block[r][c]`, `diag[v1][id] −= block[c][r]`. Lazy: `flex_hessian_valid[f]` (cleared by plasticity yield, reset, and per-step invalidation elsewhere).

### 10.3 `snhProject` (Smith et al., "Stable Neo-Hookean Flesh Simulation", §4: project ∂²E/∂F², not the vertex Hessian)
```
rest[v] = 2·size ∘ (vert0[vert[v+1]] − vert0[vert[0]])     v = 0..2
gradient[0] = rest1 × rest2, gradient[1] = rest2 × rest0, gradient[2] = rest0 × rest1;  all ×= k23   (= rows of Dm⁻¹ᵀ… : ∇N_1..3)
F[x][y] = −edgevec0[x] gradient0[y] + edgevec2[x] gradient1[y] − edgevec4[x] gradient2[y]      (= Σ_v (x_v−x_0) ⊗ ∇N_v)
(U, σ, V) = snhSVD(F)                    (signed SVD)
hg[v] = Vᵀ gradient[v−1] (v=1..3);  hg[0] = −Σ hg[v]
metric, tension = mj_stretchElasticity(k, s)      (K, K s)
tension += 2γ∇P(s) ;  metric += snhCubicMetric(s, γ)     (2γ ∇²P)
vertex[0] = 0, vertex[v] = Vᵀ rest[v−1];  reference[e] = vertex[e0] − vertex[e1]    (rest edges in principal frame)
square[i][e] = reference[e][i]²
product[i] = metric · square[i] ;  geo[i] = tension · square[i]
J = σ0 σ1 σ2 ;  vs = 2 k22 ;  pressure = vs (J−1) ;  cof = (σ1σ2, σ0σ2, σ0σ1)
A_ij (i ≤ j) = 2 σi σj (square[i]·product[j]) + vs cof_i cof_j + (i==j ? geo[i] : pressure·σ_{3−i−j})
stretch = snhPositive3(A) ? A : Σ_e max(0, λ_e) q_e q_eᵀ   (snhEigen3)
for (i,j) ∈ {(0,1),(0,2),(1,2)} (mode 0,1,2):
   dir_e = reference[e][i] reference[e][j] ;  material = dirᵀ metric dir ;  geometric = ½(geo_i + geo_j)
   cross = pressure·σ_{3−i−j}
   symmetric[mode] = max(0, (σi+σj)² material + geometric − cross)
   skew[mode]      = max(0, (σi−σj)² material + geometric + cross)
store: rotation = U, gradient = hg, stretch, symmetric, skew
```
Rationale (from source comments): a variation `δ` of F in the principal frame changes `s_e` by `2 (diag(σ) r_e)ᵀ δ r_e` with second variation `2|δ r_e|²`.

### 10.4 `snhProjectedBlock(P, i, j)`
```
a = hg[i], b = hg[j]
local[r][c] = stretch[r][c] a_r b_c
for (r,c) ∈ {(0,1),(0,2),(1,2)} (mode order):
   S = ½(sym+skew), Dm = ½(sym−skew)
   local[r][r] += S a_c b_c ;  local[c][c] += S a_r b_r
   local[r][c] += Dm a_c b_r ;  local[c][r] += Dm a_r b_c
block = U · local · Uᵀ          (mji_mulMatMat3 then mju_mulMatMatT3)
```

### 10.5 `snhCubicMetric(metric, s, γ)` (adds `2γ ∇²P`)
`basis` columns (map `s → (a,b,c,d,e,f)` derivatives) for j=0..5: `{1,0,0,.5,.5,0}, {0,0,0,−.5,0,0}, {0,1,0,.5,0,.5}, {0,0,0,0,0,−.5}, {0,0,1,0,.5,.5}, {0,0,0,0,−.5,0}`. For each j with `h = basis[j]`:
```
dc = ( h1 c + b h2 − 2 f h5,  h0 c + a h2 − 2 e h4,  h0 b + a h1 − 2 d h3,
       h4 f + e h5 − h2 d − c h3,  h3 f + d h5 − h1 e − b h4,  h3 e + d h4 − h0 f − a h5 )
dg = (dc0+dc3+dc4, −dc3, dc1+dc3+dc5, −dc5, dc2+dc4+dc5, −dc4)
for i ≤ j: metric[i][j] += 2γ dg_i ; if i≠j: metric[j][i] += 2γ dg_i
```

### 10.6 `snhSVD(F)` — one-sided Jacobi, signed
```
scale = max|F_ij| ;  A = scale ? F/scale : 0 ;  V = I
reltol = 4e-15 (double) | 2e-6 (single)
for sweep in 0..23:
   converged = 1
   for (p,q) in (0,1),(0,2),(1,2):
      pp = |A_:p|², qq = |A_:q|², pq = A_:p·A_:q
      if |pq| ≤ reltol sqrt(pp) sqrt(qq): continue
      converged = 0 ;  δ = ½(qq−pp) ;  t = sign(δ)·pq/(|δ| + sqrt(δ²+pq²))  (sign: δ≥0 → +pq)
      c = 1/sqrt(1+t²), s = t c
      columns p,q of A and V: x' = c x − s y ;  y' = s x + c y
   if converged: break
σ_i = |A_:i|² (squared, for sorting); sort descending by pairs (0,1),(0,2),(1,2) swapping columns of A and V
if (V_:0 × V_:1)·V_:2 < 0: negate column 2 of V and A           (proper V)
u0 = A_:0 ; σ0 = |u0| ; if σ0 == 0: U = I, σ = 0, return
u0 /= σ0 ; u1 = A_:1 − (u0·A_:1) u0
if |u1| > tol σ0 (tol 1e-12 double | 1e-6 single): u1 /= |u1|
else: axis = argmin |u0_x| (first minimum); u1 = e_axis − u0[axis] u0; normalise
u2 = u0 × u1 ;  σ1 = u1·A_:1 ;  σ2 = u2·A_:2  (signed: σ2 < 0 for inverted elements) ;  σ *= scale
U = [u0 u1 u2]
```
### 10.7 `snhPositive3(A)` (conservative Cholesky test)
`tol = 1e-12·max|A_ij|` (single: 1e-5); require `A00 > tol`; `p = A11 − A01²/A00 > tol` (computed `A01·(A01/A00)`); `cross = A12 − A01 (A02/A00)`; `A22 − A02(A02/A00) − cross(cross/p) > tol`.

### 10.8 `snhEigen3(A)` (Eberly §5, non-iterative)
```
scale = max|A_ij| ; Q = I ; if scale == 0: values 0, return
D = A/scale ; if D01 == D02 == D12 == 0: values = diag(A), return
mean = tr(D)/3 ; B = D − mean I ; bscale = max|B_ij| ; B /= bscale
p = sqrt((B00²+B11²+B22² + 2(B01²+B02²+B12²))/6) ; B /= p
r = ½ det(B) (cofactor expansion along row 0 with symmetric entries)
root = 2 cos(acos(min(1, |r|))/3) ; if r < 0: root = −root          (isolated extreme eigenvalue of normalised B)
B −= root I
cross candidates: B_0×B_1, B_0×B_2, B_1×B_2 (rows); pick largest squared norm (first max wins, strict >)
q = cross/|cross|
u = |q0| > |q1| ? (−q2, 0, q0)/sqrt(q0²+q2²) : (0, q2, −q1)/sqrt(q1²+q2²) ;  v = q × u
a = uᵀDu, b = uᵀDv, c = vᵀDv ;  cos=1, sin=0, t=0
if b ≠ 0: δ = ½(c−a); w = max(|δ|,|b|); x = δ/w, y = b/w; t = sign(x)·y/(|x| + sqrt(x²+y²)) (x≥0 → +y); cos = 1/sqrt(1+t²); sin = t cos
values = (qᵀDq·scale, (a − t b)·scale, (c + t b)·scale)
Q columns = (q, cos u − sin v, sin u + cos v)
```
Note eigenvalues are not sorted; the projection clamps each with `max(0, ·)`.

### 10.9 StVK Hessian path (for contrast)
`mj_stretchStiffness`: tension = `max(K s, 0)` per edge (compressed edges' negative geometric term dropped; passive forces keep full tension); block `(i,j) = Σ_{a,b} 2 K_ab σ_a^i σ_b^j e_a e_bᵀ + (Σ_a T_a σ_a^i σ_a^j) I`, with `σ_a^i = +1` if i is edge a's first endpoint, −1 if second, 0 otherwise.

---

## 11. GPU (MuJoCo Warp) implementation

### 11.1 What is precomputed in `io.py` (host, numpy, from the compiled mjModel)
Warp **does not recompute** any elastic precomputation; `flex_stiffness`, `flex_bending`, `flex_node0`, `flex_vert0`, `flex_vertcell`, `flex_edgeflap` are copied from mjModel and cast to float32. Extra host-side tables:

| field | definition |
|---|---|
| `flex_elemflexid[nflexelem]`, `flex_edgeflexid[nflexedge]`, `flex_vertflexid`, `flex_shellflexid`, `flex_shelladr` | owning flex per element/edge/vertex/shell fragment |
| `nflexintcell`, `flex_cell_map[n,4]` | for each interpolated flex with `edgeequality≠3` and `_flex_has_stiffness`: shell: `(f, fe, 0, 0)` for all faces; volume: `(f, ci, cj, ck)` for all cells (empty cells included; skipped in kernel by `k[0]==0`) |
| `nflexface`, `flex_face_map[n,2]`, `flex_faceadr[nflex]`, `flex_face[nflexface, 9]` | all shell faces of `interp<0` flexes; `flex_face` = global node indices (`nstart + gidx`, `−1` padding beyond `(o+1)²`) |
| `flex_bend_interp_map[n,2]`, `nflexbend_interp` | `(f, e)` for interpolated shell bending records |
| `flex_rodhinge_map[n,2]`, `nflexrodhinge` | `(f, bendingadr + 1 + ROD_HINGE_SIZE·h)` for 1D non-interp non-rigid flexes |
| `flex_simple[nflex]` | `_flex_simple`: every vertex's weld body has 0 dofs or is `_flex_body_simple` (3 dofs, 3 slide joints with axes x,y,z within `MJ_MINVAL`, parent weld body without dofs) |
| `has_flex_snh` | any non-interp dim-3 flex with `stiffness[adr+21] != 0` (requires `FLEX_STIFFNESS_3D == 24`); error unless integrator DISCRETE |
| `has_non_simple_flex`, `has_flex_passive`, `has_flex_edge_stiff_damp`, `efm0_active`, `flex_interp_assemblable`, `has_unsupported_flex_interp` | dispatch flags |
| efm data | obtained by running C `mj_fwdPosition` on a copy with DISCRETE/NEWTON, vertbodies replaced by weld ids, non-simple flexes marked rigid (`_get_efm_data`) |
| `flexstrain_J_*` | strain-equality Jacobian sparsity |

`types.FLEX_STIFFNESS_3D = 24` if `mujoco >= 3.14.1.dev989511280` else 21 (older models without SNH record slots); `ROD_HINGE_SIZE` probed (4 or 10) by compiling a 3-vertex rod; `FLEX_INTERP_MAX_NODES = 27`.

Unsupported (NotImplementedError in `put_model`): quadratic interpolation (`|flex_interp| == 2`), rigid flexes, self-collision or passive contact of interpolated flexes; interpolated shell bending damping only warns ("Bending damping is not yet supported").

### 11.2 Kernels and parallelisation (float32, `[nworld, …]`)
- `_flex_elasticity`: `dim=(nworld, nflexelem)`; StVK/SNH force per element (§10.1 math, identical formulas incl. `_snh_cubic`, `_snh_volume`), forces converted to body spatial forces `(f, −(xipos − x)×f)` and `atomic_add` into `flex_spring_body_force[nworld, nbody]`, later mapped by `support.apply_ft`. Mathematically equal to C's Jacobian scatter; summation order differs (atomics).
- SNH damping: `flex_hessian` (`_flex_hessian_clear`, `_flex_hessian_elem` per element using `snh_project`/`snh_projected_block`, `_flex_hessian_diag`) then `flex_stretch_mul(qfrc_damper, qvel, s1=0, s2=−1, use_timestep=False, snh_only=True)` → `−flex_damping·Jᵀ H J qvel`. `snh_project` additionally returns an `active` flag (any positive diagonal/mode) used to skip zero blocks.
- `_flex_passive_interp`: `dim=(nworld, nflexintcell)`; one thread per cell/face, scratch `[nworld, nflexintcell, 27]` vec3 for displacements and corotated velocities; dense `K` row loop reading `flex_stiffness` directly; forces rotated back, damper ×`flex_damping`, atomics into body spatial forces. Node velocity from `cvel` and `subtree_com[rootid]` (`v = v_com + ω×(x − com)`), equal to C's `mj_objectVelocity` form.
- Face rotations: `smooth._flex_face_kinematics` per face → `d.face_quat`, `d.face_xpos`.
- Plasticity (`forward.py`, after the step): `_next_plastic_flex_stretch` `(nworld, nflexelem)`, `_next_plastic_flex_bend` `(nworld, nflexedge)`, `_next_plastic_flex_interp` `(nworld, nflexintcell)`; read `plastic_in`, write `plastic_out` (the same array `d.plastic`, disjoint slots per thread). Stretch kernel also clears `flex_hessian_valid`. Sleep filter: element/edge/cell processed if any of its vertex/node bodies is awake (C filters by whole-flex sleep state).

### 11.3 Algorithmic differences vs C (parity notes)
1. **Precision**: all float32. SNH constants are the C *single-precision* set: Jacobi reltol `2e-6`, rank tol `1e-6`, `_snh_positive3` tol `1e-5·scale` (C double: `4e-15`, `1e-12`, `1e-12·scale`).
2. **Polar rotation** (`support.mat33_to_quat_polar`): same warm start, but **≤50 iterations** (C 500), convergence `w < 1e-6` (C double `1e-9`), denominator epsilon `1e-10` (C `mjMINVAL = 1e-15`), quaternion layout `(x,y,z,w)`; axis-angle built directly from `sin/cos(w/2)`.
3. **Interpolated passive rotation**: `compute_interp_cell_quat` builds `F = Σ x ⊗ ∇N(½)` **without dividing columns by the rest cell extents** `dref`, and the face kinematics likewise omit the `1/d0, 1/d1` scaling of `t1, t2`. For cubic cells (equal extents) the polar rotation is identical; for non-cubic cells `polar(F) ≠ polar(F·diag(1/d))` under shear, so corotational forces differ from C. The plasticity kernel (`_next_plastic_flex_interp`) *does* divide by `dx,dy,dz` / `d0,d1` (matches C), then computes the strain directly as `(R⁻¹ f_k)[k]/d_k − 1` (equivalent to C's `defGradient(displacement)` form for axis-aligned rest cells).
4. Closed-form shape functions at ½ (order 1: φ=½, φ'=∓1; order 2: φ=(0,1,0), φ'=(−1,0,1)) instead of evaluating polynomials — exact in both cases.
5. `_flex_passive_interp` computes `K u` and `K v` together in one loop; no `k_elem` symmetric shortcuts.
6. Shell-mode interior TFI reconstruction is done in Warp's flex kinematics (not re-derived here); faces with x-normal have `k[0]==0`, and the kernel only applies the `k[0]==0` skip in volume mode (same as C).
7. Deterministic mode: kernels compiled with `wp.DeterministicMode.RUN_TO_RUN` (`deterministic_max_records = 27` for interpolated) when `opt.deterministic & ATOMICS`.
8. Compile-time data are double in mjModel and rounded once to float32 — reproduce by computing in double and casting.

### 11.4 MJX (JAX)
`_put_model_jax` copies no flex elasticity; MJX has no flex passive forces, no SNH, no plasticity (see spec 13).

---

## 12. Reproduction pitfalls / parity notes

1. **Tet orientation**: stored tets are flipped to non-positive signed volume; `k[23] = 1/(6V)` is therefore negative and `J = k23·det[...]` is +1 at rest. Do not "fix" signs.
2. `ComputeBasis` uses `V²` (sign-free) but `ComputeStiffness` uses `|V|` for the measure; StVK uses `la = λ/2·measure`; SNH uses `(μV0, −μV0)`; interpolated 3D uses full `λ` and 2D shells plane-stress `λ*`.
3. `MetricTensor` packing order and the 21/24 stride; 2D records have stride 21 even though only 6 entries are used; slot `stiffness[6]` (element 0 only) stores thickness when `stressyield > 0` for bend-capable 2D flexes.
4. `young > 0` always allocates a zero stiffness record for non-interpolated 2D/3D flexes; runtime gates use `k[0] != 0`. For interpolated volume flexes `mj_flexHasStiffness` scans all cells (`k[e·9npe²]`); shells return true whenever a record exists.
5. Flap vertex order `(min, max, opp_first, opp_second)` must match `flex_edge`; non-manifold edges keep the last opposite vertex.
6. `ComputeBending` boundary edges: 17 zeros (no stiffness). The `b16` formula uses `|e0|³` computed as `sqr·sqrt(sqr)`.
7. Rod bending: hinge pairs are emitted in **edge order**, then single hinges in vertex order; vertex ids stored as doubles; `kb` uses the denominator `l0 l1 + e0·e1` (`2 tan(φ/2)`), not `|e0||e1|` alone.
8. Gauss–Legendre point orders (center-first for n=3 and n=5) affect K bit patterns; the full-cell and FCM integrators accumulate in different orders.
9. `K_full_cell` is computed once from the first full cell; if your grid were non-uniform (not possible via flexcomp, possible via direct XML nodes) the C result would still copy it.
10. Strain-equality flexes use `E = 10, ν = 0.3` for the eigenmodes regardless of user values; modes are stored unsorted in Jacobi index order with threshold `1e-8·max|λ|`; shells append one warp mode scaled by `√(D(1−ν)4/(d0d1))` computed with `E = 10`.
11. `cell_empty` from flexcomp is discarded by `CopyFromSpec` and recomputed in `Compile` (vertices in the R0 frame, box = node AABB). The two calls see different coordinates (parent-body frame vs rotated world frame); the result is identical only up to the quantised cache key; when the cache is disabled the computation reruns in the new frame (rounding can differ in borderline cut tests).
12. `ComputeUnrotatedNodePositions` is called before and after `ComputeCellEmpty`; R0 comes from the **first non-empty** cell's corner edges and must be orthonormal to 1e-8 (rotated grids with non-uniform scaling error out).
13. `ComputeInterpBending` counts edges whose rest normal is degenerate (`< 1e-12`) but leaves their records zero (fe_A = fe_B = 0, stiffness 0) — the runtime still iterates them.
14. Node masses: volume mode normalises FCM weights with floor `0.02·w_full` for cut cells; empty cells contribute nothing; shell quadratic uses Simpson weights with boundary multiplicities; all normalised to the flexcomp `mass`; inertia scales with the same factor.
15. Pins for interpolated flexcomps index the node array; `pingrid` uses `count`-based `GridID` (not the node grid).
16. `vert0` for non-interpolated flexes is normalised by the *BVH root box minus radius*; SNH's rest-shape reconstruction depends on it (`2·size·Δvert0`). Excluded inner layers (`activelayers`) shrink the box but keep linearity.
17. Plasticity uses predicted positions `x + h·v` with *start-of-step* Jacobian data (and RK4 restores those arrays first); the 3D stretch return mapping solves a depressed cubic for volume preservation; all stretch plastic values are floored at `−0.99·L0²`; interpolated plastic strain norms are capped at 0.35; bend plastic angles clipped to ±1.3 rad.
18. `flex_hessian_valid[f]` is cleared on stretch yield (C and Warp); any reimplementation that caches the SNH/StVK Hessian across calls must honour it.
19. SNH damping uses `−flex_damping · H_PSD · v` (not divided by timestep), whereas StVK edge damping uses `kD = flex_damping/h` times the factored elongation rate `dL(2L − dL)`.
20. Warp parity: float32, SNH single-precision tolerances, polar rotation capped at 50 iterations with tol 1e-6, missing `1/dref` column scaling in interpolated passive rotations, per-element sleep filtering in plasticity, atomic (non-deterministic) accumulation unless deterministic mode.

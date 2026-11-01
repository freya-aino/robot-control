# 15 — GPU Architecture of MuJoCo Warp (MJWarp) and MJX

**Scope:** the cross-cutting design of the GPU simulator: data model, step graph, kernel launch strategy,
precision, fixed-capacity buffers and overflow, determinism and atomics, CUDA graph capture and
conditional loops, batching and domain randomization, JAX interop (MJX `impl="warp"`), performance
techniques, and how parity with C MuJoCo is checked. The subsystem specs (smooth, collision, constraint,
solver, sensors, and so on) give the math. This file gives the *execution model* that a reimplementation
must reproduce to match MJWarp's behaviour and performance.

**Sources covered**

| Area | Files |
|---|---|
| MJWarp public API | `mujoco_warp/__init__.py` |
| Types / layout | `mujoco_warp/_src/types.py` (Model, Data, Option, Contact, Constraint, BlockDim, enums, SolverContext) |
| Infrastructure | `mujoco_warp/_src/warp_util.py` (event_scope, EventTracer, cache_kernel, mark_batched, check_toolkit_driver) |
| Step graph | `mujoco_warp/_src/forward.py` |
| Device I/O, feature gate | `mujoco_warp/_src/io.py` (put_model, make_data, put_data, get_data_into, reset_data, override_model) |
| Solver loop / graph conditionals | `mujoco_warp/_src/solver.py`, `block_cholesky.py` |
| Others referenced | `smooth.py`, `collision_driver.py`, `collision_core.py`, `collision_convex.py`, `collision_primitive.py`, `constraint.py`, `derivative.py`, `island.py`, `sleep.py`, `sensor.py`, `support.py`, `cli.py`, `testspeed.py` |
| Tests | `determinism_test.py`, `forward_test.py`, `solver_test.py`, `constraint_test.py`, `collision_driver_test.py`, `io_test.py`, `io_jax_test.py`, `jax_test.py`, `unroll_test.py`, `test_data/__init__.py` |
| Repo meta | `README.md`, `AGENTS.md`, `CONTRIBUTING.md`, `benchmarks/*`, `contrib/*`, `notebooks/tutorial.ipynb` |
| Docs | `mujoco/doc/mjwarp/index.rst`, `mujoco/doc/mjx.rst` |
| MJX bridge | `mujoco/mjx/mujoco/mjx/warp/{__init__,ffi,forward,smooth,collision_driver,render,bvh,io,types}.py`, `mujoco/mjx/mujoco/mjx/_src/{io,forward,smooth,types}.py` |
| C reference points | `engine_core_util.c:mj_isSparse`, `engine_forward.c:warmstart`, `engine_solver.c:inertiaScale`, `mjtype.h`, `mjmodel.h` |

Versions: MJWarp commit d41a47c (2026-10-31), MuJoCo commit 7ba52b5. MJWarp needs `warp-lang>=1.15` and
`mujoco>=3.13.0`, and Python 3.11 or later.

**Role in the pipeline.** MJWarp runs the same pipeline as `mj_step`
(kinematics → collision → constraints → velocity terms → actuation → smooth acceleration → constraint
solve → integration). The difference is that every stage is a set of NVIDIA Warp kernels. Each kernel
runs over a leading `nworld` axis, so thousands of independent simulations (worlds) step at once.
Host-side `mujoco.MjModel` preprocessing (`put_model`) turns the model into a device `Model` with many
extra precomputed index arrays. `make_data`/`put_data` allocate **fixed-capacity** device buffers. The
whole `step` is meant to be recorded once into a CUDA graph and replayed, with constraint-solver
iterations running inside a CUDA *conditional while* node, so converged batches stop early. Everything
runs in float32.

---

## 1. Architecture overview

### 1.1 Software stack

```
user (numpy / torch / JAX)
  └─ mujoco_warp (Python): put_model / make_data / step ...    (host orchestration, kernel launches)
       └─ NVIDIA Warp: @wp.kernel / @wp.func -> CUDA C++ (JIT, cached in ~/.cache/warp)
            └─ CUDA driver: kernel launches, CUDA graphs, conditional graph nodes (CUDA >= 12.4)
MJX (JAX) impl="warp": mjx.put_model/make_data -> jax arrays; mjx.step -> wp.jax_callable FFI -> mujoco_warp.step
```

* All simulation code is Python. Functions such as `forward`, `step` and `fwd_position` only *launch*
  kernels. They hold no per-world logic, and in normal operation they never read device memory back to
  the host. CONTRIBUTING.md rule: no `.numpy()`, `.item()` or D2H copies during simulation steps
  (CUDA graph compatibility).
* Every module sets `wp.set_module_options({"enable_backward": False, ...})`, so no adjoint kernels are
  generated. MJWarp is **not differentiable** (issue #500). Most modules also set
  `"default_grid_stride": False`.
* The CPU device works (`wp.set_device("cpu")`) for development. On CPU `wp.block_dim()` is 1, tile
  kernels run one lane, and several code paths special-case `device.is_cpu`.

### 1.2 Public API (`mujoco_warp/__init__.py`)

| Group | Functions / types |
|---|---|
| Top-level dynamics | `step`, `step1`, `step2`, `forward`, `inverse`, `euler`, `rungekutta4`, `implicit`, `discrete` |
| Forward stages | `fwd_kinematics`, `fwd_position`, `fwd_velocity`, `fwd_actuation`, `fwd_acceleration` |
| Smooth | `kinematics`, `com_pos`, `camlight`, `flex`, `tendon`, `crb`, `factor_m`, `solve_m`, `rne`, `rne_postconstraint`, `com_vel`, `subtree_vel`, `transmission` |
| Collision | `collision`, `nxn_broadphase`, `sap_broadphase`, `primitive_narrowphase`, `sdf_narrowphase` |
| Constraint/solver | `make_constraint`, `solve` |
| Passive/derivative | `passive`, `deriv_smooth_vel` |
| Sensors | `sensor_pos`, `sensor_vel`, `sensor_acc`, `energy_pos`, `energy_vel` |
| Support | `jac`, `mul_m`, `xfrc_accumulate`, `contact_force`, `get_state`, `set_state` |
| I/O | `put_model`, `make_data`, `put_data`, `get_data_into`, `reset_data`, `reset_data_keyframe` |
| Constants | `set_const`, `set_const_0`, `set_const_fixed` (deprecated), `set_length_range` |
| History (delays) | `init_ctrl_history`, `init_sensor_history`, `read_ctrl`, `read_sensor`, `reset_history` |
| Islands | `island` |
| Ray / render | `ray`, `rays`, `render`, `refit_bvh`, `refit_splat_bvh`, `create_render_context`, `get_rgb`, `get_depth`, `get_segmentation` |
| Types/enums | `Model`, `Data`, `Option`, `Contact`, `Constraint`, `State`, `Callback`, `RenderContext`, `BiasType`, `BroadphaseFilter`, `BroadphaseType`, `ConDataField`, `ConeType`, `CtrlChart`, `CtrlInput`, `DeterminismType`, `DisableBit`, `DynType`, `EnableBit`, `GainType`, `GeomType`, `IntegratorType`, `JointType`, `ObjType`, `OverflowType`, `RayDataField`, `SetConstGroup`, `SolverType`, `TrnType` |

Names follow PEP 8 (`mj_fwdPosition` becomes `fwd_position`). Every simulation function takes
`(m: Model, d: Data)` and works in place on `d`.

CLI entry points: `mjwarp-testspeed` (`testspeed.py`), `mjwarp-viewer`, `mjwarp-record`.

### 1.3 Data model and memory layout

**Array spec convention** (`types.array(*dims, dtype)`): each dataclass field is annotated with a
symbolic shape. A dimension can be an int, the name of a Model/Data size (`"nq"`, `"nworld"`,
`"naconmax"`, `"njmax_pad"`, ...), or `"*"`. A leading `"*"` marks a **batchable model field** (size 1 by
default; see §8). `io._create_array` resolves the symbols from a `sizes` dict and broadcasts the data.

Three layout classes:

1. **Model** arrays: `(nX, ...)` or `("*", nX, ...)`. Kernels index batchable fields as
   `field[worldid % field.shape[0], i]`. That pattern appears in every kernel that reads such a field.
2. **Data** arrays: `(nworld, nX, ...)`, world-major. For example `qpos (nworld,nq)`,
   `xpos (nworld,nbody) vec3`, `cdof (nworld,nv) spatial_vector`, `cinert (nworld,nbody) vec10`,
   `M (nworld,nC)` (CSR values in MuJoCo's `M_rowadr/M_colind` layout), and
   `qLD (nworld, qLD_block_total + nC_if_sparse)`.
3. **Contact** arrays: **flat across all worlds**, `(naconmax, ...)`, each with a `contact.worldid` entry.
   `d.nacon` and `d.ncollision` are shape `(1,)` global counters. There is no per-world `ncon`.
   C MuJoCo's `d->ncon` corresponds to counting contacts with `worldid == w` and type
   `CONSTRAINT|PASSIVE` (that is what `get_data_into` does).

**Constraint (efc) arrays**: `(nworld, njmax)` per world, with these exceptions:

| Field | Dense Jacobian | Sparse Jacobian |
|---|---|---|
| `efc.J` | `(nworld, njmax_pad, nv_pad)` | `(nworld, 1, njmax_nnz)` (values) |
| `efc.J_colind` | `(nworld, 0, 0)` | `(nworld, 1, njmax_nnz)` |
| `efc.J_rownnz`, `J_rowadr` | `(nworld, 0)` | `(nworld, njmax)` |
| `efc.D`, `efc.state` | `(nworld, njmax_pad)` | `(nworld, njmax_pad)` |
| `efc.jtdaj_adr/nrow` | `(nworld, 0)` | `(nworld, njmax)` only if sparse **and** Newton |

Warp-only efc fields: `efc.Ma (nworld,nv)` (= M·qacc) and `efc.Jqvel (nworld,njmax)`.

**Warp dtypes** (`types.py`). `float` means float32 and `int` means int32. Custom vector types:
`vec5, vec6, vec8, vec10, vec11, vec12, vec16` (float), `vec6i, vec8i, vec16i`,
`vec_pluginattr = vector(128)`, and matrix types `mat23, mat43, mat53, mat63, mat55, mat66, mat113`.
Builtins used: `wp.vec2/3/4`, `wp.vec2i/3i/4i`, `wp.quat` (**stored (w,x,y,z), MuJoCo order**, not
Warp's xyzw; quaternion math lives in `math.py`), `wp.mat33`, `wp.spatial_vector` (rot:lin, as MuJoCo).
`solref → wp.vec2`, `solimp → vec5`, `actuator_gainprm/biasprm/dynprm → vec10`, `eq_data → vec12`,
`pair_friction/contact.friction → vec5`, `geom_aabb → (*, ngeom, 2) vec3` (center, half-size),
`cam_mat0 → (*, ncam) mat33`, `actuator_gear → spatial_vector`, `geom_fluid → (ngeom, 12)`.

**Padded sizes** (`io._get_padded_sizes`, `_nvmax_pad`):

```
round_up(x, k) = ((x + k - 1) // k) * k
tile_size      = TILE_SIZE_JTDAJ_SPARSE (16) if sparse else TILE_SIZE_JTDAJ_DENSE (16)
augment        = (solver == NEWTON and nv > 32)            # extra column for augmented blocked Cholesky
njmax_pad      = round_up(njmax, tile_size)
nv_pad         = round_up(nv + augment, tile_size) if (sparse or nv > 32) else round_up(nv, 4)
nvmax_pad      = ((max(nvmax, 1) + 16) // 16) * 16         # always reserves >= 1 augmented column
nmaxcondim     = max(0, geom_condim, pair_dim, flex_condim)
nmaxpyramid    = max(1, 2 * (nmaxcondim - 1))              # width of contact.efc_address
qld_total      = qLD_block_total + (nC if any sparse M block else 0)
```

**Sparse vs dense constraint Jacobian** (`io.is_sparse`). This **differs from C**:

```
MJWarp: jacobian == AUTO -> sparse iff nv > 32 ; else mj_isSparse(m)
C:      jacobian == AUTO -> sparse iff nv >= 60   (engine_core_util.c:mj_isSparse)
```

`put_model` raises `ValueError` for explicit `jacobian=DENSE` when `nv > 60`. `Model.is_sparse` controls
only the efc Jacobian/Hessian layout. The factorization of M is chosen separately for each block (§10.4).

**Model "warp-only" fields** (all precomputed on the host in `put_model`). These are lookup tables that
replace loops with data-dependent control flow:

* Tree traversal: `body_tree` (tuple of per-depth body id arrays), `body_subtreenum` (bodies are
  depth-first ordered, so a subtree is the contiguous range `[b, b+body_subtreenum[b])`),
  `body_branches` + `body_branch_start` (every leaf-to-root path, `nbranch` = number of leaves),
  `body_isdofancestor (nbody, nv_pad)`, `dof_tri_row/col` (`np.triu_indices(nv)`).
* Type-partitioned address lists (each kernel launches over exactly the relevant objects):
  `eq_connect_adr, eq_wld_adr, eq_jnt_adr, eq_ten_adr, eq_flex_adr, eq_flexstrain_adr`,
  `jnt_limited_slide_hinge_adr, jnt_limited_ball_adr, tendon_limited_adr, tendon_jnt_adr,
  tendon_site_pair_adr, tendon_geom_adr, wrap_*_adr`, `sensor_{pos,vel,acc,limitpos,limitvel,
  limitfrc,touch,tendonactfrc,contact_*,stage_*}_adr`, `actuator_trntype_body_adr`, `body_freeadr`,
  `body_fluid_ellipsoid_adr`, `body_fluid_box_adr`, `mocap_bodyid`.
* Collision pair tables: `nxn_geom_pair` (all `ngeom(ngeom-1)/2` pairs), `nxn_geom_pair_filtered`
  (statically viable pairs), `nxn_pairid[_filtered] (vec2i)` = (contact pair id: `-1` default,
  `>=0` explicit `<pair>`, `-2` skipped; collision-sensor id or `-1`), `geom_pair_type_count` (number of
  viable pairs for each unordered geom-type pair; drives which narrowphase kernels get compiled and
  launched).
* Mass-matrix tables: `M_tiles` (tuple of `TileSet(adr, size, elemid)`), `qLD_block_adr`,
  `qLD_block_total`, `qLD_updates` (sparse LDL updates `(i,k,Madr_ki)` grouped by tree depth),
  `qLD_all_updates` + `qLD_level_offsets` (concatenated for the fused kernel), `M_fullm_i/j`,
  `M_elemid (nv,nv)`, `M_hinit_i`, `M_mulm_rowadr/col/madr` (gather-based symmetric `M·v`, one thread per
  row, no atomics), `qD_fullm_i/j`.
* Feature flags that turn on static kernel specialization: `is_sparse, has_fluid, flg_adhesion,
  flg_surfacevel, has_sdf_geom, has_ellipsoid_geom, has_plane_geom, has_flex_*, has_tendon_stiffness,
  has_tendon_damping, has_efm_actuator, efm0_active, flex_interp_assemblable, max_flex_dim,
  sensor_e_potential, sensor_e_kinetic, sensor_subtree_vel, sensor_rne_postconstraint`, `nmaxcondim`,
  `nmaxpyramid`, `max_ten_J_rownnz`.
* `block_dim: BlockDim` (§4.1), `callback: Callback` (Python callables).

**Data "warp-only" fields:** `nworld, naconmax, naccdmax, njmax, nvmax, nvmax_pad, njmax_pad,
njmax_nnz` (Python ints), `nacon (1,)`, `ncollision (1,)`, `overflow (nworld,) int` (OverflowType
bitmask), `flex_aabb_min/max`, `flexnode_xpos`, `face_xpos`, `face_quat`. Island (`tree_island,
dof_island, island_*, map_*`) and compact-solver (`ncdof, dof_cdof, cdof_dof, cM, cqLD, cJ, cq*`)
workspaces are allocated by `_allocate_island_arrays` / `_allocate_compact_arrays`.

**C fields with no MJWarp counterpart** (comparison of the struct definitions). In Data: `efc_R, efc_b,
efc_AR*, efc_diagA, efc_KBIP, efc_Y*, efc_J_rowsuper, qDeriv` (MJWarp has `qLU/qH` instead), `iefc_*`,
`iacc*`, `ifrc_*`, `solver, solver_fwdinv, solver_nnz, warning, timer, maxuse_*`, `ncon, nJ, nA`,
`plugin_state/plugin_data`, `bvh_aabb_dyn, bvh_active`, `flexelem_aabb`, `tendon_efcadr`, arena/stack
pointers. In Model: BVH tables, `*_user`, names, numeric/text/tuple, skins, textures (the renderer keeps
these in `RenderContext`), `sensor_noise`, `sensor_needstage` (folded into the stage address lists),
`dof_M0`, `dof_simplenum`, `B_*`, `actuator_length0`, `tendon_treeid`. Removed Option attributes:
`ls_parallel`, `ls_parallel_min_step` (raise `AttributeError` since 3.9.1).

### 1.4 Option (`types.Option`)

| Field | Type / shape | Notes |
|---|---|---|
| `timestep, tolerance, ls_tolerance, noslip_tolerance, ccd_tolerance, sleep_tolerance, density, viscosity, impratio_invsqrt` | `array("*", float)` | batchable per world |
| `gravity, wind, magnetic` | `array("*", vec3)` | batchable |
| `integrator, cone, solver, iterations, ls_iterations, noslip_iterations, ccd_iterations, disableflags, enableflags, sdf_initpoints, sdf_iterations` | Python `int` | **compile-time / host-time**; baked into kernels with `wp.static` or used for host branching |
| `broadphase` | `BroadphaseType` | NXN=0, SAP_TILE=1, SAP_SEGMENTED=2 |
| `broadphase_filter` | `BroadphaseFilter` bitmask | PLANE=1, SPHERE=2, AABB=4, OBB=8 |
| `graph_conditional` | bool | use `wp.capture_while` for solver loops |
| `run_collision_detection` | bool | False: skip collision so user-filled contacts are used |
| `run_rne_postconstraint` | bool | force `rne_postconstraint` |
| `run_qacc_smooth` | bool | False (experimental): Newton starts from `qacc_warmstart`, skips qacc_smooth and factor_m |
| `contact_sensor_maxmatch` | int | default 64, or custom numeric `contact_sensor_maxmatch` |
| `warn_overflow` | int (OverflowType) | `True`→ALL, `False`→0 |
| `deterministic` | int (DeterminismType) | bool/None → ALL/NONE; negative complements masked with ALL; rejects other bits |

`put_model` defaults: `tolerance = max(mjm.opt.tolerance, 1e-6)`;
`impratio_invsqrt = 1/sqrt(max(impratio, mjMINVAL))`; `broadphase = NXN` and then auto-selected (§10.2);
`broadphase_filter = PLANE|SPHERE|OBB`; `graph_conditional=True`; `run_collision_detection=True`;
`warn_overflow=ALL`; `deterministic=NONE`; `run_rne_postconstraint=False`; `run_qacc_smooth=True`.
Contact override parameters (`o_margin, o_solref, o_solimp, o_friction`) are absent.

**Option values are baked into the captured graph and the compiled kernels.** After changing any
`Option` field (or any Python-int Model field) you must re-capture the graph (doc: "A new graph capture
may be necessary after modifying an mjw.Option field"). If `density`, `viscosity` or `wind` changes from
zero to nonzero, `Model.has_fluid` must be updated by hand.

### 1.5 The step graph (`forward.py`)

```
step(m,d):
  forward(m,d)
  integrator dispatch: EULER→euler | RK4→rungekutta4 | IMPLICIT/IMPLICITFAST→implicit | DISCRETE→discrete
                       else NotImplementedError

forward(m,d):
  check_discrete(m)                                   # host-side validation, raises
  if sleep: sleep.wake ; sleep.update_sleep
  fwd_position(m,d, factorize=False)
  d.sensordata.zero_() ; sensor_pos ; _energy_pos     # energy_pos only if ENERGY and no E_POTENTIAL sensor
  fwd_velocity ; sensor_vel ; _energy_vel
  if not ACTUATION disabled and m.callback.control: callback.control(m,d)
  fwd_actuation
  fwd_acceleration(factorize=True)                    # factor_m happens HERE (fused factor+solve)
  solver.solve
  if DISCRETE: _launch_discrete_free_gyro(qacc)
  if run_rne_postconstraint or (sensors enabled and m.sensor_rne_postconstraint): rne_postconstraint
  sensor_acc(skip_rne_postconstraint=True)

fwd_position(m,d,factorize):
  fwd_kinematics = kinematics ; com_pos ; camlight ; flex ; tendon ; [sleep: wake_tendon, update_sleep_trees]
  crb ; tendon_armature ; [factor_m if factorize]
  if run_collision_detection:
     sleep ? collision(pass1) ; wake_collision ; awake_prev=clone(body_awake) ; update_sleep ;
             collision(awake_prev)   # incremental pass 2: appends only pairs touching newly-awake bodies
           : collision
  make_constraint
  [sleep: wake_equality ; update_sleep ; island.island]
  transmission
  if DISCRETE: derivative.eff_build

fwd_velocity: _actuator_velocity ; _tendon_velocity ; com_vel ; passive ; rne ; tendon_bias ; [DISCRETE: eff_shift]
fwd_actuation: (ctrl history read) ; _actuator_force ; callbacks act_dyn/act_gain/act_bias ;
               tendon force clamp ; _qfrc_actuator (atomics) ; gravcomp & actfrc limits ;
               [DISCRETE: eff_actuation ; regularize_constraint]
fwd_acceleration: _qfrc_smooth (zero for sleeping trees) ; xfrc_accumulate ;
               skip_qacc_smooth ? (update_active_dofs) :
               DISCRETE ? eff_solve (PCG) : sleep ? update_active_dofs + smooth_solve_compact :
               factorize ? factor_solve_i (fused) : solve_m

_advance(m,d,qacc,qvel=None):            # shared tail of every integrator
  _next_activation (nworld,nactuator) ; _next_velocity (nworld,nv) ; _next_position (nworld,njnt)
  plastic updates (joint/tendon/flex) if nplastic>0 and SPRING enabled
  history.insert_ctrl_history
  _next_time (nworld): time += timestep ; sets NEFC/BROADPHASE/NARROWPHASE overflow bits
  wp.copy(d.qacc_warmstart, d.qacc)
  if sleep: sleep.sleep ; fwd_velocity ; update_sleep
```

**Ordering differences from C that change numbers:**

* In `forward`, the inertia factorization is postponed into `fwd_acceleration(factorize=True)`. That
  call launches the fused *factor+solve* tile kernel (`factor_solve_i`), so `qLD` is written there and
  not in `fwd_position`. `step1` calls `fwd_position(factorize=not skip_qacc_smooth)`, so `step1/step2`
  factor in `fwd_position` as C does.
* `step2` integrates with Euler when the integrator is RK4. C does the same.
* `euler` (when EULERDAMP and DAMPER are both enabled): clone `M`, add `h·damping_deriv` to the diagonal
  (the damping derivative includes `actuator_damping`/`dampingpoly` for joint actuators), factor-solve
  `qacc = (M + hD)^{-1}·efc.Ma`. That scratch copy of `M` is per call. Otherwise `_advance(qacc)`.
* `implicit`: IMPLICIT builds `qLU = M − h·qDeriv` in D-structure (`deriv_smooth_vel`, `_map_m2d`,
  `deriv_rne_vel`) and solves with a fused sparse LU (`_factor_solve_lu_sparse_fused`, one launch per
  world). IMPLICITFAST builds `qDeriv` in M-structure, resets free-body blocks to M
  (`_implicit_free_body_reset_m`), and factor-solves. Both then call `implicit_free_body` (gyroscopic
  6×6 unsymmetric solve per standalone free body, `math.lu_factor_6x6`). This matches C's replacement of
  midpoint integration by gyroscopic derivatives (changelog commit f0fa3d82). The README's "IMPLICITFAST
  midpoint integrator feature is not supported" refers to the removed C feature.
* `rungekutta4`: tableau `A=[0.5,0.5,1.0]`, `B=[1/6,1/3,1/3,1/6]`, with an RKMK correction
  (`_rk_correct_velocity`) for ball and free joints. It clones and restores `qpos, qvel, act`, and for
  plasticity also `ten_length, ten_J, flexvert_xpos, cdof, subtree_com`. It allocates scratch every call.

### 1.6 Per-world masking and early termination

There is no per-world control flow on the host. Each kernel launches over the full `nworld × capacity`
grid and threads decide whether to work:

| Mechanism | Where | Rule |
|---|---|---|
| Capacity guards | almost every efc/contact kernel | `if r >= nefc_in[worldid]: return`; `if conid >= nacon_in[0]: return`; `if collisionid >= ncollision_in[0]`; loops run to `min(njmax, nefc)` / `min(ncollision, naconmax)` |
| Contact type filter | constraint/sensor kernels | `if not (contact.type & ContactType.CONSTRAINT): return` (CONSTRAINT=1, SENSOR=2, PASSIVE=4) |
| Solver convergence | every solver-iteration kernel | `if ctx_done_in[worldid]: return`; `nsolving` counts unconverged worlds; the loop exits when it reaches 0 |
| Newton stable fast path | `_update_constraint*`, `_update_gradient_incremental`, Cholesky | skip worlds with `state_changed_count == 0`; reuse `hfactor` |
| Sleep | `_qfrc_smooth`, broadphase, `_zero_sleeping_dofs` | `tree_awake[worldid,tree]==0` → zero force / skip pair (both ASLEEP, or ASLEEP vs STATIC) |
| Static geoms | `_geom_local_to_global` | if `body_weldid[body]==0` and the root is not mocap, the thread returns. `geom_xpos/xmat` is computed **once** in `make_data`/`put_data` via host `mj_kinematics` |
| Incremental broadphase | `nxn_broadphase` pass 2 | whole launch inside `wp.capture_if(cond)`, where `cond` = any body changed awake state |
| Mesh dataid per world | broadphase | `_mesh_missing(geom_dataid[worldid % n])` → pair skipped (per-world asset slots, §8) |

Kernels never branch on `nworld` on the host. CONTRIBUTING.md forbids guards like `if na > 0:
wp.launch(...)` when the size is already a launch dimension, because a dim-0 launch is a no-op.

### 1.7 Kernel construction and caching (`warp_util.py`)

* **Static specialization by factory.** Many kernels are built by a Python factory decorated with
  `@cache_kernel` that returns `@wp.kernel(module="unique", ...)`. Model properties (cone type,
  `is_sparse`, `ls_iterations`, `nv`, tile sizes, `warn_overflow`, `deterministic`, the list of geom pair
  types present, ...) are closed over and used through `wp.static(...)`, so dead branches are removed at
  codegen time. `module="unique"` gives each specialization its own CUDA module. There are 177
  `module="unique"` kernels in `_src`.
* **`cache_kernel`** memoizes factory results in a global `_KERNEL_CACHE`:

  ```python
  key = tuple(a.size if hasattr(a, "size") else hash(tuple(a)) if isinstance(a, list) else hash(a)
              for a in args) + (func.__module__, func.__name__)
  ```

  Arrays (and `TileSet`, which has a `size` attribute) are keyed **by size only**. The comment says the
  cache exists "to avoid crashes in graph_conditional code": Warp must not build new modules while a
  conditional graph body is being captured, so the same kernel object has to be returned. A
  reimplementation must keep any argument that changes kernel code out of `size`-only keys. The
  determinism test (`_assert_default_path_keys`) asserts that `deterministic_max_records`-only
  arguments are 0 on the non-deterministic path, so that path does not recompile.
* **`event_scope`** wraps public stage functions. When an `EventTracer` is active it records a
  `wp.Event(enable_timing=True)` pair around the call and builds a nested `{name: (events, substack)}`
  tree; `trace()` turns it into elapsed times. It is a **no-op inside `capture_while`/`capture_if`**
  (detected by walking `inspect.stack()`), because events cannot be recorded inside conditional bodies.
  Only one tracer can be active at a time. `testspeed --event_trace` prints the tree
  (`humanoid.step.forward.fwd_position.kinematics ...`).
* **`mark_batched`** sets `arr._is_batched = True` on every array whose spec begins with `"*"` or
  `"nworld"`.
* **`check_toolkit_driver`** runs `wp.init()`. If the device is CUDA and
  `wp.is_conditional_graph_supported()` is False, it warns: "CUDA version < 12.4 … graph capture may be
  unreliable for < 12.3, conditional graph nodes are not available for < 12.4, set
  `Model.opt.graph_conditional = False`".
* **Kernel parameter conventions** (enforced by `contrib/kernel_analyzer` in pre-commit and CI):
  parameters are grouped `# Model:`, `# Data in:`, `# In:`, `# Data out:`, `# Out:`, in Model/Data field
  order, with `_in`/`_out` suffixes. Atomics must be annotated `# kernel_analyzer: ignore[atomic]`.

### 1.8 Callbacks (`types.Callback`)

These are Python callables on `m.callback`, not global C function pointers. Each is invoked as
`cb(m, d)` and is expected to launch kernels: `control` (writes `d.ctrl`, called after `sensor_vel`),
`passive` (`qfrc_passive`), `act_dyn`, `act_gain`, `act_bias` (after `_actuator_force`), `sensor`
(writes `sensordata`, gets an extra `stage` argument), `contactfilter` (after narrowphase, before the
determinism sort). Model types `DynType.USER`, `GainType.USER`, `BiasType.USER`, `SensorType.USER` route
to these callbacks.

---

## 2. Feature support matrix: C MuJoCo vs MJWarp vs MJX

Legend: ✓ supported, ✗ raises at `put_model` (or not available), ~ partial or with caveats. "MJX-Warp"
is MJWarp behind the JAX FFI, so it inherits the MJWarp column except where noted.

### 2.1 How MJWarp gates features (`io.put_model`)

`put_model` raises `NotImplementedError` if any array value is not a member of MJWarp's enum:
`actuator_trntype ∉ TrnType`, `actuator_dyntype ∉ DynType`, `actuator_gaintype ∉ GainType`,
`actuator_biastype ∉ BiasType`, `eq_type ∉ EqType`, `geom_type ∉ GeomType`, `sensor_type ∉ SensorType`,
`wrap_type ∉ WrapType`, `tree_sleep_policy ∉ SleepPolicy`. It also raises for `opt.integrator`,
`opt.cone`, `opt.solver` outside their enums, and for any `disableflags`/`enableflags` bit outside
`DisableBit`/`EnableBit`. Further explicit checks:

| Condition | Error |
|---|---|
| `SLEEP` and any `eq_type == FLEX` | NotImplementedError "Flex equality constraints are not supported with sleeping enabled." |
| `noslip_iterations > 0` and `integrator == DISCRETE` | NotImplementedError |
| any `body_plugin / actuator_plugin / sensor_plugin != -1` | NotImplementedError (geom SDF plugins are fine) |
| contact sensor requests LINVEL/ANGVEL data and `nflex > 0` | NotImplementedError |
| `abs(flex_interp) == 2` (quadratic) | NotImplementedError |
| interpolated shell flex with bending damping | warning only |
| `mjNPOLY != 2` | warning |
| `nv > 60` and `jacobian == DENSE` | ValueError |
| `SLEEP` and `solver != NEWTON` | ValueError "sleeping requires the Newton solver" |
| flex present **and** any SDF geom / HFIELD geom / `flex_internal` / `flex_rigid` | NotImplementedError (Flex-SDF, Flex-HField, internal collisions, rigid flexes) |
| interpolated flex with selfcollide (and contype&conaffinity) | NotImplementedError |
| interpolated flex dim≥2 with passive contact | NotImplementedError |
| 3D SNH elasticity (`flex_stiffness[adr+21] != 0`) and integrator ≠ DISCRETE | ValueError |
| DISCRETE + NEWTON + non-simple flex | ValueError ("requires solver='CG'") |
| friction component < `mjMINMU` (1e-5) for a used condim | warning "may cause NaN" |
| plugin with > 128 attributes | ValueError |

At step time `forward.check_discrete` adds: SNH or passive flex contact needs DISCRETE; IMPLICIT or
IMPLICITFAST with flex elasticity (`nefmK>0` or non-simple flex) → ValueError ("use discrete, or Euler
with small timestep"); DISCRETE with flex edge stiffness/damping → ValueError; DISCRETE with
quadratic/unsupported interp → NotImplementedError; DISCRETE+NEWTON with flex interp not assemblable →
ValueError; DISCRETE + SLEEP + `DSBL_ISLAND` → ValueError; DISCRETE + SLEEP + flex → ValueError.

### 2.2 Matrix

| Category | C MuJoCo | MJWarp / MJX-Warp | MJX-JAX |
|---|---|---|---|
| Precision | float64 (`mjtNum`; `mjUSESINGLE` build gives float) | float32 only | float32 (float64 with `jax_enable_x64`) |
| Forward / inverse dynamics | ✓ / ✓ | ✓ / ✓ (`inverse`, incl. INVDISCRETE and DISCRETE) | ✓ / ✓ |
| Differentiability | finite-difference APIs (`mjd_*`) | ✗ (`enable_backward=False`) | ✓ autodiff |
| Integrators | EULER, RK4, IMPLICIT, IMPLICITFAST, DISCRETE | all 5 | EULER, RK4, IMPLICITFAST (DISCRETE accepted by put_model but `step` raises; IMPLICITFAST+fluid raises) |
| Solvers | PGS, CG, NEWTON | CG, NEWTON (PGS ✗) | CG, NEWTON |
| noslip | ✓ | ✓ except with DISCRETE | ✗ |
| Cone | pyramidal, elliptic | both | both (condim 1 not with elliptic) |
| Jacobian | dense / sparse / auto (sparse at nv≥60) | dense / sparse / auto (**sparse at nv>32**); dense ✗ for nv>60 | DENSE only in practice (MJX doc); auto=sparse if nv≥60 or TPU |
| Mass matrix factor | sparse LDLᵀ | per-tree: scalar Cholesky (≤6), tile Cholesky UᵀU (≤64), sparse LDLᵀ (>64) | dense/sparse |
| Joints | free, ball, slide, hinge | all | all |
| Transmission | JOINT, JOINTINPARENT, SLIDERCRANK, TENDON, SITE, BODY, SO3 | all **except SO3** | JOINT, JOINTINPARENT, SITE, TENDON |
| Actuator dynamics | NONE, INTEGRATOR, FILTER, FILTEREXACT, MUSCLE, DCMOTOR, PID, USER | all **except PID** (USER via callback) | NONE, INTEGRATOR, FILTER, FILTEREXACT, MUSCLE |
| Actuator gain | FIXED, AFFINE, MUSCLE, DCMOTOR, SO3, PID, USER | FIXED, AFFINE, MUSCLE, DCMOTOR, SO3, USER (**PID ✗**) | FIXED, AFFINE, MUSCLE |
| Actuator bias | NONE, AFFINE, MUSCLE, DCMOTOR, SO3, USER | NONE, AFFINE, MUSCLE, DCMOTOR, USER (**SO3 ✗**) | NONE, AFFINE, MUSCLE |
| Actuator delay/history, ctrl interval | ✓ | ✓ (`history.py`) | ✗ |
| Plasticity (joint/tendon/flex yield) | ✓ | ✓ | ✗ ("Plasticity not implemented") |
| Geoms | PLANE, HFIELD, SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX, MESH, SDF, FLEX | all (plus internal TRIANGLE=999) | all but SDF; ELLIPSOID/CYLINDER only vs primitives; BOX treated as a mesh |
| Narrowphase | analytic + MPR/GJK-EPA (nativeccd), multiccd | analytic primitives + GJK/EPA (with multiccd polygon clipping) + SDF gradient descent + HField prisms; see the collider table §2.3 | analytic + SAT for meshes |
| Midphase (BVH) | ✓ (`DSBL_MIDPHASE`) | ✗ flag (broadphase NXN/SAP only; BVH used for ray/render) | ✗ |
| Equality | CONNECT, WELD, JOINT, TENDON, FLEX, FLEXSTRAIN (FLEXVERT, DISTANCE unsupported in C too) | CONNECT, WELD, JOINT, TENDON, FLEX, FLEXSTRAIN (FLEX ✗ with sleep) | CONNECT, WELD, JOINT, TENDON |
| Constraint types | all | all | EQUALITY, LIMIT_JOINT, CONTACT_*, FRICTION_DOF, FRICTION_TENDON |
| Tendons / wrapping | fixed, spatial; JOINT, PULLEY, SITE, SPHERE, CYLINDER | all | all |
| Fluid | inertia + ellipsoid | all | inertia model only |
| Sensors | all incl. PLUGIN, USER | all **except PLUGIN** (USER via callback; DataType AXIS/QUATERNION unsupported) | subset (doc footnote [2]); CONTACT limited |
| Collision sensors (GEOMDIST/NORMAL/FROMTO) | separate collision calls | computed in the main collision pass; each sensor geom pair always emits 1 `ContactType.SENSOR` contact, even if not touching | ✓ |
| Flex | full | most: elasticity (incl. SNH under DISCRETE), bending, edge eq, interp (trilinear only), selfcollide (non-interp), passive contact (DISCRETE); ✗: quadratic interp, flex–SDF, flex–HField, internal, rigid flex | ✗ |
| Plugins | body, actuator, sensor, geom(SDF) | geom SDF only (override `collision_sdf.user_sdf/user_sdf_grad`) | ✗ |
| Sleep / islands | ✓ (policies AUTO*, NEVER, ALLOWED, INIT) | ✓ Newton only; policies AUTO, AUTO_NEVER, AUTO_ALLOWED (NEVER/ALLOWED/INIT ✗); compact solver with `nvmax` | ✗ |
| Disable flags | all 20 | all except MIDPHASE, AUTORESET; NATIVECCD means "use primitive box-box" | CONSTRAINT…FILTERPARENT subset |
| Enable flags | OVERRIDE, ENERGY, FWDINV, INVDISCRETE, SLEEP, DIAGEXACT, IPC, SPECULATIVE | ENERGY, INVDISCRETE, SLEEP | INVDISCRETE, SLEEP, IPC (accepted) |
| Contact override (`o_*`) | ✓ | ✗ | ✗ |
| Warmstart policy | min-cost of (qacc_warmstart, qacc_smooth) | always qacc_warmstart when nefc>0 | — |
| Ray casting | ✓ | ✓ (`ray`, `rays`, rangefinder) with BVH for mesh/hfield/flex | slow for mesh, no hfield/flex |
| Rendering | OpenGL | batch ray-tracer (`render`), 3DGS splats, shadows, textures | via MJX-Warp |
| `set_const` | `mj_setConst` | `set_const` (groups FIXED, KIN0, INERT0; no SPRING) | — |
| Named access / bind | ✓ | ✗ (issue #884) | ✓ |
| `mj_forwardSkip`, `mj_fwdPosition` skip stages | ✓ | ✗ (no skip-stage API) | — |
| Callbacks | global function pointers | per-model `m.callback` | — |
| Multi-GPU | n/a | one Model/Data/graph per device (`wp.ScopedDevice`) | `pmap`/sharding |

### 2.3 Pair-wise colliders (MJWarp, `collision_driver.MJ_COLLISION_TABLE`)

PRIMITIVE = analytic kernel; CONVEX = GJK/EPA CCD kernel (one specialized kernel for each geom pair type
present). Maximum contacts per pair: default flags (nativeccd on, multiccd on) / multiccd off.

| | SPHERE | CAPSULE | ELLIPSOID | CYLINDER | BOX | MESH | SDF |
|---|---|---|---|---|---|---|---|
| PLANE | prim 1 | prim 2 | prim 1 | prim 4 | prim 4 | prim **4** (C: 3) | prim 1 |
| HFIELD | CCD 4 | CCD 4 | CCD 4 | CCD 4 | CCD 4 | CCD 4 (C: mjMAXCONPAIR) | HFieldSDF (sdf_initpoints) |
| SPHERE | prim 1 | prim 1 | CCD 1 | prim 1 | prim 1 | CCD 1 | SDF |
| CAPSULE | | prim 2 | CCD 1 | CCD 2/1 | prim 2 | CCD 2/1 | SDF |
| ELLIPSOID | | | CCD 1 | CCD 1 | CCD 1 | CCD 1 | SDF |
| CYLINDER | | | | CCD 4/1 | CCD 4/1 | CCD 4/1 | SDF |
| BOX | | | | | CCD 4/1 (**primitive 8** with `DSBL_NATIVECCD`) | CCD 4/1 | SDF |
| MESH | | | | | | CCD 4/1 | MeshSDF |
| SDF | | | | | | | SDF (sdf_initpoints) |

Flex collisions (`MJ_FLEX_COLLISION_TABLE`): PLANE, SPHERE, CAPSULE, BOX, CYLINDER vs FLEX are
primitive; MESH and ELLIPSOID vs FLEX are convex.

---

## 3. Precision policy

* **Everything on the device is float32.** Warp's `float` annotation is `wp.float32`, and every
  `types.py` field and every kernel signature uses `float`, `wp.vec3` (f32), `wp.mat33` (f32), and so on.
  There is no float64 option and no mixed-precision accumulator anywhere in `_src`. The only float64 is
  host-side numpy scratch in `io.py`.
* MJX-Warp also casts every float64 leaf (from `MjModel`, or created under `jax_enable_x64`) to float32
  in `_tree_to_float32` before `device_put`, because "the MuJoCo Warp FFI rejects" float64.
* Constants come straight from C: `MJ_MINVAL = mujoco.mjMINVAL = 1e-15`, `MJ_MAXVAL = 1e10`,
  `MJ_MINIMP = 1e-4`, `MJ_MAXIMP = 0.9999`, `MJ_MAXCONPAIR = 50`, `MJ_MINMU = 1e-5`, `MJ_MINAWAKE = 10`.
  These are float64-era values. 1e-15 is representable in f32 (min normal ≈ 1.18e-38), but differences
  of O(1) quantities never get that small in f32 (eps ≈ 1.19e-7). Guards like `x < MJ_MINVAL` therefore
  trigger in fewer cases than in C, and only for genuinely tiny values. `math.safe_div(x,y) = x /
  (y != 0 ? y : MJ_MINVAL)`.
* GJK/EPA module constants (`collision_gjk.py`): `FLOAT_MIN=-1e30`, `FLOAT_MAX=1e30`, `MINVAL=1e-15`,
  `MAXVAL=1e15`, `MIN_EPATOL=1e-7`, `FACE_TOL=cos(0.0889)`, `EDGE_TOL=sin(0.0889)`,
  `INTERSECT_TOL=3e-7`. Box-box primitive (`collision_primitive_core.py`): `_BOXBOX_SEPEPS=1e-6`,
  `_PAREPS=1e-7`, `_SGNEPS=1e-5`, `_DUPEPS=1e-10`, `_EDGEBIAS=1e-6`, `_MAXVERT=8`. Ray: `RAY_TOL_REL=1e-6`,
  `RAY_TOL_ABS=1e-9`. Rod bending `_ROD_SMOOTH=1e-4`.
* **Tolerance adjustments for f32:**
  1. `opt.tolerance = max(opt.tolerance, 1e-6)` in `put_model`. Comment: "C MuJoCo tolerance was chosen
     for float64 architecture, … adjust the tolerance for lower precision, to avoid the solver spending
     iterations needlessly bouncing around the optimal solution." The C default `1e-8` becomes `1e-6`.
  2. Linesearch gradient tolerance gets a floor:
     `gtol = max(tolerance · ls_tolerance · ‖search‖ / inertia_scale, 1e-6)`.
     C: `gtol = tolerance·ls_tolerance·snorm/scale` with no floor.
  3. `_ALPHA_NOISE_EPS = 8 · 1.1920929e-7` (8 f32 ulps): linesearch acceptance tolerance is
     `gtol_accept = max(gtol, _ALPHA_NOISE_EPS·|q1|)`, and steps below a noise floor count as exhausted.
     This forces a gradient rebuild in the incremental Newton path.
* **fast_math**: the linesearch kernel is compiled with `module_options={"fast_math": True}` (comment:
  "fast math only approximates div/sqrt here; keep transcendentals out of this kernel"). The render
  megakernel uses it when `rc.use_fast_math` (default True). Approximate div/sqrt means linesearch
  results are not IEEE-exact.
* Tile Cholesky (`wp.tile_cholesky_*`) and `tile_matmul` follow Warp's own summation order. For blocked
  Cholesky kernels mathdx GEMM is disabled (`enable_mathdx_gemm: False`).
* Practical consequence: parity with C is ~1e-4 relative for single-stage quantities and degrades over
  steps. Tests use `rtol=atol=5e-4` for most fields and much looser bounds for solver outputs (§11). The
  MJWarp FAQ says: "Solver settings, including iterations, collision detection, and small friction values
  may be sensitive to differences in floating point representation."

---

## 4. Parallelization strategy

### 4.1 Launch primitives and BlockDim

* `wp.launch(kernel, dim=(d.nworld, N[, K]))`: **one thread per (world, element)**. This is the default
  pattern for almost every kernel.
* `wp.launch_tiled(kernel, dim=(d.nworld[, nblock]), block_dim=B)`: one **thread block** per world (or
  per world×block), with `B` lanes cooperating through `wp.tile_*` ops (reduce, load, matmul, cholesky,
  sort). Inside, `tid` is the lane and loops stride by `wp.block_dim()`.
* Grid-stride launches (`grid_stride=True` / `max_blocks`) are used for capacity-sized work
  (contacts/pairs). Their width comes from occupancy: `wp.get_suggested_block_size(kernel)` returns
  `(block_size, min_grid_size)`, and the grid is a few "waves" of resident threads instead of
  `naconmax` threads.

`types.BlockDim` defaults (marked experimental):

| Field | Default | Used by |
|---|---|---|
| `segmented_sort` | 128 | SAP_TILE `_segmented_sort` |
| `nxn_broadphase` | 128 | NXN broadphase |
| `convex_ccd` | 64 | GJK/EPA kernels |
| `island_dsu` | 32 | `_island_dsu` |
| `ray` | 64 | `rays` tiled kernel |
| `contact_sort` | 64 | contact-sensor match sort |
| `energy_vel_kinetic` | 32 | kinetic energy |
| `cholesky_factorize` / `cholesky_factorize_solve` | 32 / 32 | M dense-block tile Cholesky |
| `cholesky_solve` | 64 | M dense-block solve |
| `small_cholesky` | 64 | scalar small-block Cholesky |
| `solve_LD_sparse_fused` | 128 | fused sparse LDL backsubstitution |
| `update_gradient_cholesky` | 64 | Newton H Cholesky (nv ≤ 32) |
| `update_gradient_cholesky_blocked` | 32 | Newton blocked Cholesky (nv > 32) |
| `update_gradient_JTDAJ_sparse` / `_dense` | 128 / 128 | Hessian assembly |
| `linesearch_iterative` | 32 (**256 if nv > 500**) | linesearch |
| `update_gradient_grad`, `solve_beta_accumulate`, `solve_search_update_cg`, `solve_init_search_cg` | 256 → **set to `clamp(round_up(nv,32), 32, 256)`** in put_model | CG tile kernels |
| `contact_jac_tiled` | 32 | dense contact Jacobian (also the tile size) |
| `eff_pcg` | 128 | DISCRETE PCG |
| `render` | 64 | render megakernel |

Tile-size constants: `TILE_SIZE_JTDAJ_SPARSE = TILE_SIZE_JTDAJ_DENSE = 16`, `M_BLOCK_DENSE_MAX = 64`,
`M_BLOCK_SCALAR_MAX = 6`, `solver._BLOCK_CHOLESKY_DIM = 32`, `_INIT_H_LANES = 8`,
`_JTDAJ_THREADS_PER_GROUP = 32`, `_JTDAJ_OVERSUBSCRIBE_WAVES = 6`, `constraint._CONTACT_JAC_LANES = 8`,
`collision_convex._CCD_OVERSUBSCRIBE_WAVES = 2`, `_CCD_MIN_BLOCKS = 8`,
`island._DSU_MIN_CHUNK = 256`, `_DSU_TARGET_BLOCKS = 2048`.

### 4.2 Stage-by-stage parallelization table

`W = d.nworld`. "Atomic" means the kernel uses float atomics (non-deterministic order unless
`DeterminismType.ATOMICS`; §6). Counts: there are ~426 `wp.launch[_tiled]` call sites outside tests.

**Kinematics / smooth (`smooth.py`)**

| Function | Kernel | Launch dim | Notes |
|---|---|---|---|
| kinematics | `_kinematics_branch` | `(W, nbranch)` | one thread walks a whole **leaf-to-root branch** (`body_branches[start:end]`). Ancestors shared by several branches are recomputed redundantly (and written identically), so there are no inter-thread dependencies and no level synchronization. Free joint: `xquat = normalize(qpos quat)` |
| | `_compute_body_matrices` | `(W, nbody)` | `xmat = quat_to_mat(xquat)` |
| | `_compute_body_inertial_frames` | `(W, nbody)` | `xipos, ximat` |
| | `_geom_local_to_global` | `(W, ngeom)` | skips static geoms (§1.6) |
| | `_site_local_to_global` | `(W, nsite)` | |
| flex | `_flex_nodes`, `_flex_vertices`, `_flex_edges`, `_flex_face_kinematics` | `(W, nflexnode/nflexvert/nflexedge/nflexface)` | |
| com_pos | `_subtree_com` | `(W, nbody)` | each body sums its **contiguous** subtree range `[b, b+subtreenum)` itself: O(depth) redundant work, no atomics; body 0 handled by `_subtree_com_world` `(W)` |
| | `_cinert` / `_cdof` | `(W, nbody)` / `(W, njnt)` | |
| camlight | `_cam_local_to_global` / `_light_local_to_global` | `(W, ncam)` / `(W, nlight)` | |
| tendon | `_joint_tendon` (atomic) | `(W, nwrap_jnt)` | fixed tendons |
| | spatial site-pair / geom kernels | `(W, nwrap_site_pair)`, `(W, nwrap_geom)` | |
| | `_spatial_tendon_wrap` | `(W)` | one thread per world builds wrap paths serially |
| crb | `_crb_subtree` | `(W, nbody)` | contiguous subtree sum (no atomics; body 0 excluded) |
| | `_M` | `(W, nv)` | per-dof row of CSR M: diagonal `armature + cdofᵀ·crb·cdof`, then walks `dof_parentid` within the row |
| tendon_armature | — | `(W, ntendon, max_ten_J_rownnz)` | |
| factor_m | `_small_cholesky_factorize_block(size)` | `(W, nblocks_of_size)`, block_dim 64 | scalar in-register Cholesky for blocks ≤ 6 dofs (or reciprocal diagonal for "compact" diagonal blocks) |
| | `_tile_cholesky_factorize_block(tile)` | tiled `(W, nblocks)`, block_dim 32 | `tile_load_indexed` gathers the dense block from CSR M by a precomputed per-slot index (absent → index nC → reads 0), then `wp.tile_cholesky_inplace(fill_mode="upper")` |
| | `_qLD_acc` (+`_qLD_scale` when deterministic) | `(W, |qLD_updates[level]|)` per tree level, deepest first | sparse LDLᵀ for blocks > 64 dofs (atomic_sub) |
| | `_qLDiag_div` | `(W, nv)` | `D = 1/L_ii` |
| factor_solve_i | `_*_factorize_solve_block` | same as factor | fused factor + back-substitution (forward path) |
| solve_m | `_small_cholesky_solve_block` / `_tile_cholesky_solve_block` | `(W, nblocks)` | |
| | `_solve_LD_sparse_fused(nv, nlevels)` | `(W, 128)` single block per world | UP sweep over levels (deepest first), diag, DOWN sweep; `__syncthreads` (`WP_TILE_SYNC()` native snippet) between levels; atomics within a level |
| | `_solve_LD_sparse_level` | `(W, |level|)` per level | deterministic alternative: one launch per level with a copied input |
| com_vel | `_comvel_root` `(W,6)`, `_comvel_branch` `(W, nbranch)` | | branch walk again |
| rne | `_cacc_world` `(W)`, `_cacc_branch` `(W, nbranch)`, `_cfrc` `(W, nbody)`, `_cfrc_subtree` `(W, nbody)`, `_cfrc_world` `(W)`, `_qfrc_bias` `(W, nv)` | | forward pass by branches; backward pass by contiguous subtree sums |
| tendon_bias | `_tendon_dot` `(W, ntendon)`, `_tendon_bias_coef`/`_tendon_bias_qfrc` (atomic) `(W, ntendon, max_ten_J_rownnz)` | | |
| transmission | `_transmission` (atomic) `(W, nactuator)`, `_transmission_body_moment_scale` `(W, nacttrnbody, nv)` | | |
| rne_postconstraint | `_cfrc_ext` `(W,nbody)`, `_cfrc_ext_equality` `(W,neq)`, `_cfrc_ext_contact` `(naconmax)`, `_cfrc_ext_tendon_*` (all atomic) | | |
| subtree_vel | `_subtree_vel_forward`, `_subtree_momentum` `(W, nbody)`, `_subtree_momentum_world` `(W)` | | |

**Collision (`collision_driver.py`, `collision_*.py`)**

| Function | Kernel | Launch | Notes |
|---|---|---|---|
| nxn_broadphase | `_nxn_broadphase(filter, batch sizes, sleep, incremental)` | `(W, nnxn_geom_pair_filtered)`, block_dim 128, grid-stride `max_blocks = 8·block_size·min_grid_size/128` | filter functions `_plane_filter`, `_sphere_filter`, `_aabb_filter`, `_obb_filter` by bitmask; survivors claim a slot with `cid = atomic_add(ncollision, 0, 1)` and write `collision_pair/pairid/worldid[cid]` if `cid < naconmax`. Pair is ordered so `type1 <= type2` |
| sap_broadphase | `_sap_pair_margin` `(npairmargin_batch, npair)`; `_sap_project` `(W, ngeom)`; sort: `_segmented_sort(ngeom)` tiled `(W)` block 128 using `wp.tile_sort` (SAP_TILE) **or** `wp.utils.segmented_sort_pairs` (SAP_SEGMENTED); `sap_range` `(W, ngeom)`; `wp.utils.array_scan` for load balancing; sweep kernel | | projection direction is fixed `normalize(0.5935, 0.7790, 0.1235)`. SAP cannot go inside a conditional graph (its sort/scan allocate scratch) |
| primitive_narrowphase | `_primitive_narrowphase(types, funcs)` | `(naconmax)` | one specialized kernel containing only the pair types present (`geom_pair_type_count`); thread `tid < ncollision` |
| convex_narrowphase | `ccd_kernel_builder(g1, g2, ccd_iterations, epa_iterations, multiccd, geomgeomid, 64, warn)` | grid `min(naconmax, 2·block·min_grid)` (CPU: naconmax), block 64 | **one kernel per convex geom-type pair present**; each grid-strides over all collision slots and processes only its pair type. HField pairs get separate kernels. Per-CCD scratch is sized by `naccdmax` (§5) |
| sdf_narrowphase | `_sdf_narrowphase` | `(sdf_initpoints, naconmax)` | gradient descent from several start points |
| flex | `flex_collision` (broadphase AABB, SAP sort, FPS contact reduction inside `capture_while`) | various | not sleep-aware |
| surface vel | `_contact_surfacevel` | `(naconmax)` | if `flg_surfacevel` |

Contact write (`collision_core.write_contact`): `active = dist < margin || (adhesion≠0 && dist <
margin+gap)`; `detected = dist < margin+gap`; condim becomes 1 for adhesive in-gap contacts; skip if
(`pairid[0]==-2` or not detected) and no sensor; `type |= CONSTRAINT` if `pairid[0] >= -1 && detected`;
`type |= SENSOR` if `pairid[1] >= 0`; `cid = atomic_add(nacon,0,1)`; write only if `cid < naconmax`.

**Constraints (`constraint.py: make_constraint`)**

| Kernel | Launch |
|---|---|
| `_zero_constraint_counts` | `(W)` |
| `_equality_connect/weld/joint/tendon(sparse,newton,discrete)` | `(W, neq_type)` |
| `_equality_flex` | `(W, neq_flex, nflexedge)` |
| `_equality_flexstrain` | `(W, neq_flexstrain)` |
| `_friction_dof` / `_friction_tendon` | `(W, nv)` / `(W, ntendon)` |
| `_limit_ball` / `_limit_slide_hinge` / `_limit_tendon` | `(W, n_limited_of_type)` |
| `_efc_contact_init(cone, sparse, newton, adhesion)` (grid-stride) | `(naconmax)`: allocates rows `base = atomic_add(nefc[w], ndim)` with `ndim = condim` (elliptic) or `2(condim−1)` (pyramidal; 1 if condim 1) |
| `_efc_contact_jac_sparse(cone, nmaxdim, max_records, det)` | `ncontact_stride·8` lanes; `ncontact_per_pass = min(naconmax, 1024·sm_count)` |
| `_efc_contact_jac_dense(tile=32, cone, ...)` | tiled `(W, ceil(nv_pad/32))`, block 32; `tile_atomic_add` into `efc.Jqvel` |
| `_efc_contact_update(cone, adhesion, discrete)` (grid-stride) | `(naconmax, nmaxdim)` |
| `_add_surface_vel` | `(naconmax)` |
| `_efc_nnz_overflow` | `(W)` |
| `_regularize_constraint` (DISCRETE) | `(W, njmax_pad)` |

Every row-emitting kernel uses the same allocation protocol: `wp.atomic_add(ne/nf/nl_out, w, nrow)`;
`efcid = wp.atomic_add(nefc_out, w, nrow)`; `if efcid >= njmax: return`; `nrow = min(nrow, njmax −
efcid)`. In sparse mode, storage is reserved **before** the row count (`_reserve_efc_nnz`): `rowadr =
atomic_add(efc_nnz[w], nrow·rownnz)`; if `rowadr + nnz > njmax_nnz` the rows are dropped (and for
contacts `efc_address = -1`). For sparse Newton each row block also claims a JTDAJ group slot
`jgid = atomic_add(jtdaj_nblock[w], 1)`. **Row order is therefore whatever order the atomics happen to
resolve in.**

**Solver (`solver.py`)**

| Kernel | Launch | Notes |
|---|---|---|
| `_solve_init_dof(warmstart, sparse, run_qacc_smooth)` | `(W, nv)` | qacc ← warmstart if nefc>0 else qacc_smooth |
| `_inertia_scale` | `(W)` | `1/max(MJ_MINVAL, Σ M_ii)` (serial over nv) |
| `_solve_init_efc` | `(W)` grid-stride | niter=0, done=False |
| `_solve_init_jaref_kernel(sparse, nv, dofs_per_thread, ...)` | `(W, njmax, threads_per_efc)` | sparse: 1 thread per row; dense: `dofs_per_thread = 20 if nv>50 else 50`, atomics across threads when >1 |
| `mul_m_kernel` | `(W, nv)` | gather via `M_mulm_*`, no atomics |
| `_update_constraint_efc(track_changes, det)` | `(W, njmax)` | force/state per row; Newton non-elliptic tracks state flips (atomic counters) |
| `_update_constraint_init_qfrc_constraint_sparse/dense` | `(W, nv)` | qfrc = Jᵀf |
| `_update_gradient_grad_tiled` | tiled `(W)`, block `clamp(round32(nv),32,256)` | grad = Ma − qfrc_smooth − qfrc_constraint, `grad_dot` via `tile_reduce` |
| `_update_gradient_init_h_sparse` | `(W, nv_pad, 8)` | H ← M (or qH) densified |
| `_JTDACJ_sparse(sc, cone, maxcondim, block_dim)` | `(W, groups_per_world, 32)` | `groups_per_world = max(1, min(njmax, 6·device_warps / W))`; atomic_add into dense `h` |
| `_update_gradient_JTDAJ_dense_tiled(nv_pad, 16, njmax, nC)` | tiled `(W)`, block 128 | scatter CSR M into a shared `(nv_pad²)` tile (`tile_scatter_masked`), then `Σ_k tile_matmul((J_kᵀ·D_k), J_k)` over 16-row chunks up to nefc |
| `_update_gradient_cholesky(nv)` | tiled `(W)`, block 64 | nv ≤ 32: `tile_cholesky_inplace(upper)` + `tile_cholesky_solve`; search = −H⁻¹g; also search·search and grad·solution |
| `_update_gradient_cholesky_blocked[_skip_unchanged](16, nv_pad, nv)` | tiled `(W)`, block 32 | nv > 32: blocked right-looking UᵀU with 16×16 tiles (`block_cholesky.py`), augmented column carries the RHS; the skip variant reuses `hfactor` when no constraint changed |
| `_update_gradient_incremental` | `(W, slots, 32)` + `_update_gradient_h_incremental` `(W, nv(nv+1)/2)` | Newton pyramidal: rank updates only for rows whose state changed |
| `_linesearch_iterative_kernel(ls_iter, cone, fuse_jv, sparse, incremental, warn)` | tiled `(W)`, block 32 (256 if nv>500), fast_math, grid-stride | **whole linesearch for a world in one block**: optional fused `jv = J·search` (if `nv ≤ 50 && dense`), quad prep (elliptic needs a block barrier), 3-alpha bracketing with a mat33 `tile_reduce`, final qacc/Ma/Jaref update |
| `_linesearch_jv_fused_kernel` | `(W, njmax, threads_per_efc)` | when jv is not fused |
| `_solve_beta_finalize_tiled` | tiled `(W)` | CG Polak-Ribière `beta = max(0, gᵀ(Mg−Mg_prev)/max(MJ_MINVAL, g_prevᵀMg_prev))`, convergence, `atomic_add(nsolving,0,−1)` |
| `_solve_search_update_cg_tiled` | tiled `(W)` | search = −Mgrad + beta·search |
| `_solve_done` | `(W)` | Newton convergence + decrement of nsolving |
| `_qfrc_constraint_from_grad` | `(W, nv)` | Newton incremental: rebuild qfrc_constraint from grad at the end |
| noslip: `_noslip_factor_m` `(W)`, `_noslip_delassus` `(W, njmax)`, `_noslip_qacc_init` `(W,nv)`, `_noslip_solve`, `_noslip_state` `(W, njmax)` | | |

**Passive, actuation, sensors, sleep, islands**

| Function | Kernel(s) | Launch |
|---|---|---|
| passive | `_spring_damper_dof_passive` `(W, njnt)`; `_spring_damper_tendon_passive` (atomic) `(W, ntendon, max_ten_J_rownnz)`; `_gravity_force` (atomic) `(W, nbody−1, nv)`; `_fluid_force` `(W, nbody)`; flex `_flex_elasticity` `(W, nflexelem)`, `_flex_bending` `(W, nflexedge)`, `_flex_passive_interp` `(W, nflexintcell)`; `_qfrc_adhesion` | |
| actuation | `_actuator_force` `(W, nactuator)`; `_tendon_actuator_force` (atomic) `(W, nactuator)`; `_tendon_actuator_force_clamp`; `_qfrc_actuator` (atomic, scatter over moment row) `(W, nactuator)`; `_qfrc_actuator_gravcomp_limits` `(W, nv)` | |
| velocity | `_actuator_velocity` `(W, nactuator)`, `_tendon_velocity` `(W, ntendon)` | gather, no atomics |
| sensors | `_sensor_pos` `(W, nsensor_pos)`, `_sensor_vel`, `_sensor_acc` similar; `_limit_pos/vel/frc` `(W, njmax, nsensor_limit*)`; `_sensor_rangefinder` `(W, nrangefinder)`; `_sensor_collision` `(naconmax)`; `_sensor_touch` (atomic) `(naconmax, ntouch)`; `_contact_sort(maxmatch)` tiled `(W, ncontactsensor)` block 64 using `wp.tile_sort`; `_sensor_tactile` `(W, nsensortaxel)`; `_energy_vel_kinetic(nv)` tiled `(W)` block 32 | |
| sleep | `_update_sleep_trees` `(W, ntree)`, `_update_sleep_bodies` `(W, nbody)`, `_update_sleep_dofs` `(W, nv)`, `_wake_*`, `_build_cycles` `(W)` | |
| island | `direct_dsu`: `_reset_dsu` `(W, ntree)`, `_island_dsu` tiled `(W, nchunk)` block 32 with atomic-min root hooks over efc rows (`nchunk = min(max(1, ceil(2048/W)), ceil(njmax/256))`), `_compress_roots`, `_label_roots` `(W)`, `_propagate_labels` | |
| compact solve | `_compact_dofs_builder` `(W)` serial over trees; gather/scatter `(W, nv)`; Cholesky blocked tiled `(W)` at size `nvmax_pad` | |
| integration | `_next_activation` `(W, nactuator)`, `_next_velocity` `(W, nv)`, `_next_position` `(W, njnt)`, `_next_time` `(W)` | |

---

## 5. Fixed-capacity buffers and overflow behaviour

### 5.1 Capacities

| Capacity | Scope | Set by | Default | Buffers sized by it |
|---|---|---|---|---|
| `naconmax` | **all worlds combined** | `naconmax`, or `nconmax · nworld` | `nconmax = _default_nconmax` (below) | all `Contact` fields, `collision_pair/pairid/worldid` (broadphase output), contact-indexed launches |
| `nconmax` | per world (soft) | `make_data/put_data` | heuristic | only through `naconmax = nconmax·nworld`. One world may use more than nconmax as long as the total stays ≤ naconmax |
| `naccdmax` / `nccdmax` | all worlds / per world | `naccdmax`, or `nccdmax·nworld`; default = naconmax | naconmax | GJK/EPA/multiccd scratch: `epa_vert (naccdmax, 10+2·epa_iter) vec3`, `epa_vert_index`, `epa_face/epa_pr/epa_norm2 (naccdmax, 6+5·epa_iter)`, `epa_horizon (naccdmax, 24)`, `multiccd_* (naccdmax, 2·npolygonmax or nmeshdegmax)`. Must satisfy `naccdmax ≤ naconmax`, `nccdmax ≤ nconmax` |
| `njmax` | **per world, hard** | `make_data/put_data` | `_default_njmax` | all efc arrays `(nworld, njmax[_pad])`, solver context (`Jaref, jv, quad (W, njmax)`), island maps |
| `njmax_nnz` | per world | `njmax_nnz` | sparse: `_default_njmax_nnz` (heuristic below); dense: `njmax·nv` | `efc.J`, `efc.J_colind` (sparse) |
| `nvmax` | per world | `nvmax` | `nv` | compact-solver arrays `(W, nvmax_pad[, nvmax_pad])`; must be in `[0, nv]` |
| `contact_sensor_maxmatch` | per contact sensor | `m.opt.contact_sensor_maxmatch` or custom numeric | 64 | `sensor_contact_matchid/criteria (W, ncontactsensor, maxmatch)` |
| `MJ_MAXCONPAIR = 50` | per hfield pair | constant | 50 | hfield prism candidates |
| `MJ_MAX_EPAHORIZON = 24` | per EPA | constant | 24 | horizon edges |
| `MJ_MAX_EPAFACES = 5` | per EPA iteration | constant | 5 | faces added per iteration |
| `epa_iterations` | | `16 if all CCD pairs are box-box else opt.ccd_iterations` | | EPA buffers |
| `npolygonmax`, `nmeshdegmax` | | 4/3 if any box-box; with multiccd and cylinders: `max(m.npolygonmax,16)`, `max(m.nmeshdegmax,3)`; mesh/box-mesh/capsule-mesh: `max(m.npolygonmax, 4 or prev)`, `max(nmeshdegmax,3)` | | multiccd scratch |

Heuristic defaults (`io.py`):

```
valid_sizes = (2 + (arange(19) % 2)) * 2**(arange(19)//2 + 3)      # 16,24,32,48,64,96,128,...,6144,8192
nconmax = max(nv*0.35*(nhfield>0)*10 + 45, 256*has_flex, 64*has_sdf, mjd.ncon if mjd else 0)
njmax   = max(nv*2.26*(nhfield>0)*18 + 53, 512*has_flex, 256*has_sdf, mjd.nefc if mjd else 0)
round each up to the next value in valid_sizes (left as is if > 8192)
```

(Without hfields both defaults are constant: nconmax → 48 (45 rounded up), njmax → 64 (53 rounded up).)

`_default_njmax_nnz(mjm, nconmax, njmax)`, sparse only: sum over all equalities assumed active (CONNECT:
`3·pair_nnz`, WELD: `6·pair_nnz`, JOINT: 1 or 2, TENDON: union of the two tendons' `ten_J` columns,
FLEX: Σ non-rigid edge `flexedge_J_rownnz`, FLEXSTRAIN: `neig·nv`), plus frictional dofs and tendons,
plus limits (ball 3, slide/hinge 1, tendon `ten_J_rownnz`), plus `njmax · max_contact_nnz` (maximum over
enabled collision pairs of the dof count of the union of both bodies' kinematic chains). Result:
`min(max(total,1), njmax·nv)`.

`put_data` additionally raises `ValueError` if `nconmax < mjd.ncon`, `naconmax < mjd.ncon·nworld`, or
`njmax < mjd.nefc`. The MJCF `size/nconmax` and `size/njmax` are **not** read.

### 5.2 Overflow semantics: drop and flag, never resize

Capacity is never grown at runtime: that would require a host synchronization and would break the
graph. Counters keep incrementing past capacity, writes past capacity are discarded, and a bit is set in
`d.overflow[world]`:

| OverflowType bit | Value | Raised where | Behaviour on overflow |
|---|---|---|---|
| `NEFC` | 1<<0 | `_next_time` (end of step) if `nefc[w] > njmax` | rows with `efcid >= njmax` are dropped (constraint silently missing; contact `efc_address = -1`) |
| `NJMAX_NNZ` | 1<<1 | `_efc_nnz_overflow` if `efc_nnz[w] > njmax_nnz` | the whole row block that did not fit is dropped |
| `BROADPHASE` | 1<<2 | `_next_time` if `ncollision > naconmax`; flex broadphase | pairs with `cid >= naconmax` dropped (all worlds flagged) |
| `NARROWPHASE` | 1<<3 | `_next_time` if `nacon > naconmax`; flex narrowphase | contacts with `cid >= naconmax` dropped (all worlds flagged) |
| `CCD` | 1<<4 | CCD kernels if `atomic_add(nccd[pairtype],1) >= naccdmax` | that CCD pair skipped |
| `HFIELD` | 1<<5 | hfield collider when candidate prisms exceed `MJ_MAXCONPAIR` | extra prisms ignored (one warning per pair) |
| `CONTACT_MATCH` | 1<<6 | contact sensor matching | matches beyond `maxmatch` ignored |
| `NVMAX` | 1<<7 | `_compact_dofs` when awake dofs > nvmax | `ncdof = nvmax`, extra dofs not mapped: **"behavior undefined"** |
| `EPA_HORIZON` | 1<<8 | EPA when the horizon exceeds 24 edges | EPA aborts for that pair |
| `ITERATIONS` | 1<<9 | `_solve_done` / `_solve_beta_finalize_tiled` when `niter == iterations` without convergence | world marked done |
| `LS_ITERATIONS` | 1<<10 | linesearch when not converged within ls_iterations | best alpha so far used |
| `TACTILE` | 1<<11 | tactile pair collection | extra pairs ignored |
| `ALL` | OR of all | | |

* Bits are OR-ed with plain `|=` for per-world-owned writes, or `wp.atomic_or` for contact-parallel
  writers. `d.overflow` starts at zero, is cleared by `reset_data`, and is **sticky** across steps
  otherwise.
* When the matching bit in `opt.warn_overflow` is set (it is baked with `wp.static`, so changing it
  needs a recompile/re-capture), the kernel also calls `wp.printf` with a message like "nefc overflow -
  please increase njmax beyond %u … To disable the print warning: m.opt.warn_overflow &=
  ~mjw.OverflowType.NEFC". Global-counter warnings print only from `worldid == 0`. The docs warn that
  device printf serializes and syncs, and can severely degrade throughput when many worlds overflow.
* Benchmark example: `unitree_g1_*` sets `opt.warn_overflow=~ITERATIONS|~LS_ITERATIONS`.
  `testspeed --overflow_behavior=error` (default) aborts on any overflow. Benchmarks must run with zero
  overflow and `converged_worlds == nworld`.
* `get_data_into` clamps: `nacon = min(nacon, naconmax)`, `nefc = min(nefc[w], njmax)`.
* Reproduction rule: with no overflow the dropped-row logic never runs. With overflow, which rows
  survive depends on atomic arrival order and is **not reproducible**.

### 5.3 Memory notes (doc §Memory)

CCD colliders need more memory than primitives; multiccd needs more than CCD; CCD memory grows linearly
with `ccd_iterations`; with meshes and multiccd it grows with the maximum vertices per face and the
maximum edges per vertex. Sparse efc example: Aloha clutter (nv=136, 2048 worlds, njmax=384): sparse
`efc.J`+`J_colind` ≈ 84 MB versus ≈ 408 MB dense. `testspeed --memory` reports per-field memory and
"Other memory" (inline allocations).

---

## 6. Determinism and atomics

### 6.1 Sources of non-determinism

1. **Float atomics** (`wp.atomic_add/sub` on float). Used for scatter reductions in passive (gravity,
   tendon springs, flex), actuation (`_qfrc_actuator`, tendon actuator force), transmission, tendon
   Jacobian/bias, rne_postconstraint (`cfrc_ext_*`), sparse LDL factor and solve, Jaref/jv when several
   threads share a row, dense contact Jacobian (`tile_atomic_add` into Jqvel), Newton Hessian assembly
   (`_JTDACJ_sparse`, incremental H), derivatives (`qDeriv`, RNE derivative), the effective metric
   (DISCRETE), sensors (touch, energy), BVH normals. Float addition is not associative, so results can
   differ in the last bits from run to run.
2. **Integer slot allocation** with `atomic_add` on `nacon`, `ncollision`, `nefc[w]`, `ne/nf/nl`,
   `efc_nnz`, `jtdaj_nblock`, `nccd`. The values are deterministic but the *order* is not, so contact
   order and constraint row order differ between runs. Row order then changes solver summation order,
   and that changes results.
3. Integer atomics for counters (`state_changed_count`, island DSU atomic-min). Results are deterministic
   except where order matters downstream.

### 6.2 `Option.deterministic` (DeterminismType bitmask)

| Bit | Value | Status at d41a47c |
|---|---|---|
| `CONTACTS` | 1 | "stable contact ordering via geometric radix sort". **Not implemented**: `collision_driver._sort_contacts` is `pass` (TODO) |
| `CONSTRAINT` | 2 | "canonical constraint row ordering". **Not implemented**: `constraint._sort_constraints` is `pass` (TODO) |
| `ATOMICS` | 4 | implemented: every float-atomic kernel factory takes `deterministic: bool`; when set it compiles with `module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN` and `module_options["deterministic_max_records"] = <max atomics per thread>` (e.g. `nv`, `njmax`, `max_ten_J_rownnz`, `3·nv`, `EFM_CON_MAX_DOF=24`, `9·27²`) |
| `ALL` | 7 | |

What ATOMICS changes:

* Kernels that used atomics are recompiled under Warp's RUN_TO_RUN mode. Warp records atomic
  contributions (up to `deterministic_max_records` per thread) and reduces them in a fixed order.
* Algorithm switches: sparse LDL factorization runs level by level with a copied input (`L_in`) plus a
  separate `_qLD_scale` kernel. Sparse back-substitution uses `_solve_LD_sparse_level` per level instead
  of the fused single-block kernel (the test checks the fused kernel is **not** launched). CG allocates
  a `solve_ld_scratch`.
* Non-atomic paths are not touched. The `_RECORDS_ONLY_ARGS` check makes sure the default path uses
  `max_records = 0` so no extra variants compile.

`determinism_test.py` verifies, for tree / flex / flex-shell / flex-contact / flex-passive / contact /
sleep scenes across integrators (euler, implicit, implicitfast, rk4, discrete, discrete+CG), jacobians
(sparse, dense, forced sparse M blocks), and cones:

1. With `NONE`, no kernel is launched in RUN_TO_RUN mode.
2. With `ATOMICS`, a launched kernel is RUN_TO_RUN **iff** its factory has a `deterministic` parameter,
   and every factory listed in `covers` actually runs with a nonzero dim.
3. NONE and ATOMICS results differ only by summation order: `assert_allclose(rtol=atol=1e-3)` (the
   per-scene `tol` parameter) on `qpos qvel act qacc qacc_smooth qfrc_* actuator_* ten_* subtree_*
   cinert crb cacc cfrc_int cfrc_ext energy flexvert_xpos` and sensordata (excluding CONTACT sensors).
4. Worlds 0 and 1 (perturbed qvel and ctrl) give different qacc.
5. Two more ATOMICS runs of 3 steps are **bitwise identical** to the first ATOMICS run, over every array
   in m, d and the render context.
6. Because CONTACTS/CONSTRAINT are not implemented, the test uses a `_Replay` harness. It records the
   first run's `collision` and `make_constraint` outputs. Later runs check the inputs are bitwise equal
   and the float outputs are equal as multisets (sorted bit patterns), then **restore the first run's
   outputs**. So today full bitwise determinism with contacts holds only modulo contact/row order.

The FAQ answer "Is MJWarp on GPU deterministic? No …" (issue #562) is the user-facing statement. The CPU
device is deterministic.

### 6.3 Matching MJWarp's reductions

To get bit-identical results with MJWarp you would need Warp's RUN_TO_RUN reduction order and the same
tile reduction trees (`wp.tile_reduce` over `block_dim` lanes, then a strided per-lane partial sum
`for i in range(tid, n, block_dim)`). In practice, aim for agreement within the tolerances in §11.

---

## 7. Graph capture, CUDA graphs and conditional loops

### 7.1 Whole-step capture

```python
with wp.ScopedCapture() as capture:
  mjw.step(m, d)                 # records all kernel launches + memsets + copies into a CUDA graph
for i in range(nstep):
  wp.capture_launch(capture.graph)
```

* Everything inside `step` must be capturable. That means no host reads of device data and no host
  branching on device values. Branching on **Model/Option Python values** is fine, because it is frozen
  at capture time.
* Temporary arrays created during capture (`wp.empty/zeros/clone` in integrators, solver contexts,
  collision contexts, CCD scratch) come from Warp's graph-capture allocator. Their addresses are fixed
  into the graph, so they are reused on every replay.
* `cli.unroll`/`testspeed`: capture once ("jit_duration" includes compile and capture), then
  `capture_launch` + `wp.synchronize()` each step. `unroll_test` replays a captured `step` for a
  trajectory and checks the final state roughly.
* Multi-GPU: one `put_model/make_data/ScopedCapture` per device inside `wp.ScopedDevice(device)`; launch
  each graph separately.

### 7.2 Conditional nodes (CUDA ≥ 12.4, `opt.graph_conditional=True`)

* `wp.capture_while(cond_array, while_body=fn, **kwargs)` adds a conditional while node. The body graph
  is captured once and the GPU re-runs it while `cond_array[0] != 0`. No host round-trip.
* **Constraint solver loop** (`solver._solve`):

  ```
  nsolving = wp.full((1,), nworld, int)
  [CG: _solve_init_search_cg_tiled may already mark worlds done (gap test) and decrement nsolving]
  if iterations != 0 and graph_conditional:
      wp.capture_while(nsolving, while_body=_solver_iteration, m=m, d=d, ctx=ctx, nsolving=nsolving, compact=compact)
  else:
      for _ in range(iterations): _solver_iteration(...)      # fixed count; done worlds no-op
  ```

  Each iteration runs linesearch → `_update_constraint` → gradient/Hessian → convergence kernel. The
  convergence kernel sets `done[w]`, increments `solver_niter[w]` and does
  `wp.atomic_add(nsolving, 0, −1)` once per world as it finishes (converged **or** `niter ==
  iterations`, which also sets the ITERATIONS bit). This bounds the loop at `iterations` and stops it as
  soon as every world in the batch has converged. The batch's cost is set by the slowest world. Worlds
  that are done keep paying kernel-launch overhead but return immediately.
* Convergence tests (per world, `scale = inertia_scale`, `tolerance` = opt.tolerance after the 1e-6
  clamp):

  ```
  Newton: done = alpha == 0 || (0 < scale·improvement < tol) || scale·sqrt(grad·grad) < tol
                 || scale·0.5·newton_decrement < tol
  CG:     done = alpha == 0 || (0 < scale·improvement < tol) || scale·sqrt(grad·grad) < tol
  CG init (before loop): done if max(0, 0.5·scale·grad·Mgrad) < tol
  ```

  (newton_decrement = gradᵀH⁻¹grad, computed with the Cholesky solve.)
* **Other conditional loops**: DISCRETE PCG `eff_solve` uses `wp.capture_while(efm_nsolving, ...)` with
  at most `opt.iterations` iterations. Flex contact FPS reduction uses `wp.capture_while(fps_condition,
  ...)`, or `range(1, MJ_MAXCONPAIR)` iterations without conditionals. The incremental sleep broadphase
  uses `wp.capture_if(cond, on_true=_launch)` (NXN only).
* Rules for conditional bodies: **no memory allocation** inside the body. Scratch such as `efm_con`,
  `flex_scratch` and `solve_ld_scratch` is allocated before the loop and kept in `SolverContext`. Kernel
  objects must already exist (hence `cache_kernel`). `event_scope` timing is disabled inside.
* With `graph_conditional=False` (CUDA < 12.4, JAX FFI, or CPU via `solve_compact`, which forces it
  False off CUDA), the solver always runs exactly `opt.iterations` iteration launches. That is slower,
  and worlds that are already done do nothing. Results are the same, because done worlds are frozen.
  This is why MJX docs recommend lowering iterations for MJX-JAX, while MJWarp "can early exit".

### 7.3 MJX graph modes (`mjx.warp.GraphMode = wp.JaxCallableGraphMode`)

| Mode | Behaviour |
|---|---|
| `JAX` | XLA captures the graph. **Does not work** with MJWarp (child graph nodes from conditionals cannot be rolled into the XLA graph). Default when the JAX device is CPU |
| `WARP` (default on GPU) | Warp captures and caches the CUDA graph keyed on XLA buffer pointers. It re-captures whenever pointers change. Forced re-capture every step: humanoid 2.96M → 0.80M SPS |
| `WARP_STAGED` | stable staging buffers; XLA buffers are copied in and out, capture happens once (humanoid 2.67M SPS, aloha pot 1.96M) |
| `WARP_STAGED_EX` | like STAGED but the copies are outside the captured graph |

Reference: pure Warp (no FFI) humanoid 3.35M SPS, aloha pot 2.45M SPS.

---

## 8. Batching and domain randomization

### 8.1 Batched Model fields

Every Model field whose spec starts with `"*"` has shape `(B, ...)`, and kernels read
`field[worldid % B, ...]`. `put_model(mjm, batch_sizes={"field": B})` allocates `B` copies (the MjModel
value broadcast); the user then assigns per-world values. A field can also be replaced wholesale with any
`wp.array` of leading dimension `B` (tutorial:
`model.opt.gravity = wp.array([g, -g], dtype=wp.vec3f)` → world `w` uses `g` if `w` is even and `-g` if
odd). **The modulo means B need not equal nworld.** Worlds cycle through the B variants.

Batchable fields (complete list from `types.py`):

* Option: `timestep, tolerance, ls_tolerance, noslip_tolerance, ccd_tolerance, sleep_tolerance, gravity,
  wind, magnetic, density, viscosity, impratio_invsqrt`.
* Model: `qpos0, qpos_spring`; `body_pos, body_quat, body_ipos, body_iquat, body_mass, body_subtreemass,
  body_inertia, body_invweight0, body_gravcomp`; `jnt_solref, jnt_solimp, jnt_pos, jnt_axis,
  jnt_stiffness, jnt_stiffnesspoly, jnt_springyield, jnt_range, jnt_actfrcrange, jnt_margin`;
  `dof_solref, dof_solimp, dof_frictionloss, dof_armature, dof_damping, dof_dampingpoly,
  dof_invweight0`; `geom_dataid, geom_matid, geom_solmix, geom_solref, geom_solimp, geom_size, geom_aabb,
  geom_rbound, geom_pos, geom_quat, geom_friction, geom_margin, geom_gap, geom_surfacevel,
  geom_adhesion, geom_rgba`; `site_pos, site_quat`; `cam_pos, cam_quat, cam_poscom0, cam_pos0, cam_mat0,
  cam_fovy, cam_intrinsic`; `light_type, light_castshadow, light_active, light_pos, light_dir,
  light_poscom0, light_pos0, light_dir0, light_attenuation, light_cutoff, light_exponent, light_ambient,
  light_diffuse, light_specular`; `mat_texid, mat_texuniform, mat_texrepeat, mat_emission, mat_specular,
  mat_shininess, mat_rgba`; `pair_solref, pair_solreffriction, pair_solimp, pair_margin, pair_gap,
  pair_adhesion, pair_friction`; `eq_solref, eq_solimp, eq_data`; `tendon_solref_lim, tendon_solimp_lim,
  tendon_solref_fri, tendon_solimp_fri, tendon_range, tendon_actfrcrange, tendon_margin,
  tendon_stiffness, tendon_stiffnesspoly, tendon_damping, tendon_dampingpoly, tendon_armature,
  tendon_frictionloss, tendon_lengthspring, tendon_springyield, tendon_length0, tendon_invweight0`;
  `actuator_cranklength, actuator_dynprm, actuator_gainprm, actuator_biasprm, actuator_actrange,
  actuator_history, actuator_historyadr, actuator_delay, actuator_damping, actuator_dampingpoly,
  actuator_armature, actuator_forcerange, actuator_ctrlrange, actuator_gear, actuator_acc0,
  actuator_lengthrange`; `sensor_history, sensor_historyadr, sensor_delay, sensor_interval`.

Not batchable: topology and all int index fields, `geom_type`, `geom_condim`, `geom_contype/conaffinity`,
`geom_priority`, flex parameters, mesh/hfield data, `body_contype`, `actuator_*type`, limits flags, and
`opt` Python ints (integrator, solver, iterations, flags).

### 8.2 Consistency obligations

Many fields are *derived* (`body_subtreemass`, `body_invweight0`, `dof_invweight0`, `geom_aabb`,
`geom_rbound`, `actuator_acc0`, `tendon_length0`, `cam_pos0`, `eq_data` anchors, ...). If you randomize
a primary field (mass, inertia, size, gear, armature, qpos0, eq_data, cam/light pose), recompute the
dependents with **`set_const(m, d, groups)`**:

* `SetConstGroup.FIXED` (1): independent of configuration (subtreemass, geom bounds).
* `KIN0` (2): kinematics at qpos0 (tendon_length0, eq_data0, cam/light pos0).
* `INERT0` (4): inertia at qpos0 (dof/body/tendon invweight0, actuator_acc0).
* `ALL` = 7. SPRING is unsupported. `set_const` launches over `field.shape[0]` batches, so batched
  fields stay batched.

Doc caveats: `body_pos/quat` of static bodies is unsafe (static geoms are computed once in make_data and
the BVH would be stale). Actuator armature changing from or to zero is unsafe (`jnt_actuatorid`).
Tendon stiffness/damping changing from or to zero is unsafe (`has_tendon_*` flags fixed at put_model).
Changing fluid parameters from zero requires updating `has_fluid`.

### 8.3 Static-specialization coupling

Several kernels compile against the **batch size** of a field (for example `_nxn_broadphase(…,
geom_aabb.shape[0], geom_rbound.shape[0], geom_margin.shape[0], …, geom_dataid.shape[0])` and
`_sap_project(..., ngeom_dataid, ngeom_pair_margin)`). Changing a field's batch size after capture
therefore requires a new capture.

### 8.4 Per-world assets

Recipe from the docs: make one spec containing **all** assets, compile each variant, put the base model,
then override `geom_dataid` (per world, shape `(nworld, ngeom)`) together with the dependent
`geom_size, geom_aabb, geom_rbound, geom_pos, body_mass, body_subtreemass, body_inertia, body_invweight0,
body_ipos, body_iquat`. For body-level variants the base model must hold the **maximum number of geom
slots**, with unused slots disabled (`contype=conaffinity=0`, `dataid=-1`). Broadphase skips mesh geoms
whose per-world `dataid == -1` (`_mesh_missing`). MJX-Warp does not support batched `geom_dataid` (it
takes `field[0]`).

### 8.5 Data batching and resets

* All Data is per world. `put_data(nworld=N)` tiles one MjData across worlds. Contacts: the first `ncon`
  contacts of mjd are replicated `nworld` times into the flat buffer (`worldid` set, padded to naconmax).
  The efc arrays are tiled per world.
* `reset_data(m, d, reset=None|wp.array(bool/int, (nworld,)))` resets selected worlds to the model
  defaults on device (no host transfer) and clears overflow. `reset_data_keyframe(m, d, key: int |
  wp.array(int, (nworld,)))` gives a per-world keyframe index.
* `get_state/set_state` (`State` bitmask: TIME, QPOS, QVEL, ACT, HISTORY, PLASTIC, WARMSTART, CTRL,
  QFRC_APPLIED, XFRC_APPLIED, EQ_ACTIVE, MOCAP_POS, MOCAP_QUAT, USERDATA; PLUGIN unsupported) work per
  world, `(nworld,)`.
* Domain randomization of `qpos/qvel/ctrl` is just writing to the `(nworld, ·)` arrays.

---

## 9. JAX interop (MJX `impl="warp"`)

### 9.1 Direct Warp ↔ JAX

`wp.jax_callable(fn, num_outputs, output_dims, graph_mode)` wraps a Python function that launches Warp
work as an XLA custom call (FFI). `jax_test.py` and `contrib/jax_unroll.py` show the minimal pattern:
`warp_step(qpos_in, qvel_in, qpos_out, qvel_out)` copies inputs into a persistent `d`, calls `mjw.step`,
and copies out. The result is jitted inside `jax.lax.scan`, with
`XLA_FLAGS=--xla_gpu_graph_min_graph_size=1` and `graph_mode=WARP`. Both `graph_conditional=True` and
`False` are tested.

### 9.2 MJX-Warp architecture

* `mjx.put_model(mjm, impl='warp', graph_mode=None, batch_sizes=None)`: runs `mujoco_warp.put_model` (a
  vendored copy in `mujoco.mjx.third_party.mujoco_warp`) **on the Warp CPU device**, then converts every
  array to numpy, then to JAX (`_wp_to_np_type`), and casts float64 to float32. Unless a field is in
  `batch_sizes`, batchable fields have their leading size-1 dimension **squeezed**. Public `mjx.Model`
  fields come from the MJWarp Model so shapes and dtypes match. Private MJWarp fields go to
  `Model._impl: ModelWarp`. `Option._impl: OptionWarp` carries `broadphase, broadphase_filter,
  ccd_iterations, ccd_tolerance, contact_sensor_maxmatch, deterministic, graph_conditional, graph_mode,
  impratio_invsqrt, noslip_*, run_collision_detection, run_rne_postconstraint, sdf_*, sleep_tolerance,
  warn_overflow`. Default `graph_mode`: `JAX` if the device is CPU, else `WARP`. Static fields
  (`TileSet`, `BlockDim`, `Callback`, tuples) become static pytree metadata.
* `mjx.make_data(m, impl='warp', naconmax=None, naccdmax=None, njmax=None, nvmax=None)`: **no
  `nworld`**. The batch comes from `jax.vmap`, so contact capacity is given as `naconmax` (all worlds)
  and `naccdmax`. Users must scale `naconmax` by the eventual vmap size. Contacts live in the private
  `Data._impl.contact__*`, not `Data.contact`. The docs advise reading contacts through contact sensors.
* Dispatch: `mjx.forward`, `mjx.step`, `mjx.kinematics`, `mjx.com_pos`, `mjx.tendon`, `mjx.collision`,
  `mjx.render`, `mjx.refit_bvh` check `m.impl == d.impl == Impl.WARP` and call `mujoco.mjx.warp.<module>`.
  Those files are **auto-generated** ("DO NOT EDIT"): each function has a `_<fn>_shim(nworld, <every
  Model field>, <every Data field>, …)` that rebuilds an MJWarp `Model/Data` from flat arrays and calls
  `mjwarp.<fn>(_m, _d)`.
* FFI wrapper (`ffi.jax_callable_variadic_tuple`): flattens pytrees, dataclasses and variadic tuples into
  a synthetic signature, and caches the `wp.jax_callable` per structural key (func, signature, treedef,
  num_outputs, graph_mode, vmap_method, output_dims, in_out/stage argnames, side effect) under a lock.
  Call parameters: `num_outputs=164` (forward) / `171` (step); `in_out_argnames` = every Data field
  written; `stage_in_argnames` = fields to stage (for WARP_STAGED); `output_dims` from current shapes;
  `vmap_method=None`; `graph_mode=m.opt._impl.graph_mode`; `nworld = d.qpos.shape[0]`.
* vmap: `@jax.custom_batching.custom_vmap` + `marshal_custom_vmap`. Data fields that are not batched
  yet are broadcast to `axis_size` (`_BATCH_DIM` map). Leading dims are checked (Data batched fields ==
  batch, `contact__*` == naconmax, `efc__*` == njmax; otherwise an error with "use Data.where instead of
  jax.tree.map"). Nested vmaps are flattened into one batch axis (`_flatten_batch_dim`), and outputs are
  marked batched. Outside vmap, `marshal_jax_warp_callable` adds a leading world dim of 1 (`_NDIM` map)
  and squeezes it afterwards.
* `Data.where(done, other)`: per-leaf `jnp.where` only for batched fields. Unbatched fields (contacts,
  global counters) are taken from `self`. A plain `jax.tree.map(jnp.where, …)` does not work for Warp.
* Not supported in MJX-Warp: autodiff, `GraphMode.JAX`, batched `geom_dataid`.
* `mjx.render` / `create_render_context(mjm, nworld, devices=...)`: one Warp render context per device
  stored in a global registry keyed by an integer. A `_jax_token` in `Data._impl` orders render calls.

### 9.3 Default implementation choice

`mjx._resolve_impl(device)`: on a CUDA GPU it picks WARP only if the env `MJX_GPU_DEFAULT_WARP=true` and
warp is installed. Otherwise JAX on GPU/TPU/CPU. `impl='warp'` without a GPU picks the Warp CPU device.

---

## 10. Performance techniques

### 10.1 General

1. **World-parallel launch + CUDA graph replay + conditional loops** (§7). Launch overhead of hundreds of
   tiny kernels is amortized by the graph. Doc: graph replay is "typically significantly faster".
2. **Static specialization** (`wp.static`, `cache_kernel`, `module="unique"`): only the code for the
   joint, geom-pair, cone and sensor types present gets compiled. Compile time grows with the number of
   distinct convex pair types (FAQ: limit unique CCD colliders to reduce compile time).
3. **Host precomputation of index lists** so that kernels launch over exactly the relevant subset, and
   **branch-based traversal** (`body_branches`) that removes level-by-level tree passes. Contiguous
   subtree ranges allow atomic-free reductions.
4. **Gather instead of scatter** where possible: `mul_m` via `M_mulm_*` (no atomics); actuator and tendon
   velocities are gathers.
5. **Tiles / shared memory** for per-world dense linear algebra: Newton Hessian `JᵀDJ` with
   `tile_matmul` over 16-row chunks; `tile_cholesky` for nv ≤ 32; blocked 16×16 Cholesky with an
   augmented RHS column for nv > 32 (`nv_pad` gets +1 column). M dense blocks use `tile_load_indexed`
   gathers. `tile_sort` is used for SAP and contact-sensor matching. Reductions use `wp.tile(x,
   preserve_type=True)` + `tile_reduce(wp.add)`, and the linesearch packs 3 vec3 reductions into one
   mat33 reduction.
6. **Kernel fusion** in the linesearch: jv (small nv), quad preparation, Gauss term, 3-alpha bracketing,
   and the final qacc/Ma/Jaref update all run in one kernel. Also fused factor+solve for M, the fused
   sparse LDL solve, and fused LU for IMPLICIT. The design notes list fusion attempts that did not help
   (caching efc data in shared tiles).
7. **Occupancy-sized grids** for capacity-bound work: CCD `min(naconmax, 2·block·min_grid)`, NXN
   `max_blocks = 8·block·min_grid/128`, sparse contact Jacobian `1024·sm_count` contacts per pass with 8
   lanes, JTDAJ `6·device_warps/W` groups per world, DSU `ceil(2048/W)` chunks.
8. **Newton incremental Hessian** (pyramidal cone only, `_use_incremental`): count constraint state flips
   in each iteration. If a world has no flips, skip qfrc/grad/Cholesky (`grad_scale` tracks the scalar
   change along the same ray) and reuse `hfactor`. Otherwise apply rank updates only for the changed
   rows. `search_unchanged` lets the next linesearch reuse `mv`/`jv`.
9. **Early exit** of converged worlds (`ctx.done`) in every solver kernel, and of the whole loop
   (`nsolving`).
10. **Sleeping + compact solver** (§10.5).
11. **Static geoms** are computed once. Contact capacity is shared across worlds (naconmax) so
    imbalanced worlds share one pool.
12. **Low-precision friendly defaults**: tolerance clamp 1e-6, fast_math in the linesearch.
13. **Global memory discipline** (CONTRIBUTING): hoist `m.*`/`d.*` reads into registers before loops,
    precompute reciprocals, keep global writes to a minimum, defer loading efc_D/frictionloss into the
    branches that use them.
14. **Sparse efc Jacobian** when nv > 32 (memory ≈ 5× lower in the Aloha example). Constraint row
    blocks with identical support are grouped into JTDAJ blocks (`jtdaj_adr/nrow`) and assembled by
    warp-sized thread groups.

### 10.2 Broadphase selection (`put_model`)

```
n_include = number of statically viable pairs (contact or collision-sensor)
broadphase = NXN if n_include < 250_000 else (SAP_TILE if ngeom < 1000 else SAP_SEGMENTED)
```

Static pair filtering (in `put_model`, reproducing C's `filterBodyPair`): excluded if
`(contype1&conaffinity2 | contype2&conaffinity1) == 0`, same weld body, both weld bodies dof-less,
parent-child welds (when FILTERPARENT is enabled and neither weld is world), or listed in `<exclude>`
(`signature = body1<<16 + body2`). Explicit `<pair>` entries override with their pair id. Collision
sensors add pairs even when contact is excluded.

### 10.3 Solver dispatch thresholds

| Condition | Choice |
|---|---|
| `nv ≤ 32` | Newton H Cholesky via one 32-wide `tile_cholesky` per world |
| `nv > 32` | blocked Cholesky, 16×16 tiles, `nv_pad = round_up(nv+1, 16)` |
| `nv ≤ 50` and dense J | jv fused into the linesearch kernel |
| `nv > 50` dense | jv/Jaref with 20 dofs per thread, atomics; else 50 |
| `nv > 500` | linesearch block_dim 256 |
| Newton & cone ≠ ELLIPTIC | incremental Hessian + stable fast path |
| `nv > 60` & jacobian DENSE | rejected |

### 10.4 Mass-matrix block layout (`io.m_block_layout`)

The blocks are the kinematic trees (`tree_dofadr/tree_dofnum`). For each block with `size` dofs and `nnz`
CSR entries:

```
compact    = nnz == size                    # diagonal only (e.g. decoupled slider dofs)
triangular = nnz == size*(size+1)/2         # fully coupled
if size <= 6 and (compact or triangular): scalar tile: compact → Q_LD_BLOCK_COMPACT (-2), reciprocal diag;
                                          else dense factor of size² at packed offset
elif size <= 64: dense tile Cholesky (UᵀU, upper) at packed offset, size² floats
else: sparse LDLᵀ (Q_LD_BLOCK_SPARSE = -1), stored in qLD[:, qLD_block_total:] in M's CSR layout
```

`qLDiagInv` holds reciprocal diagonals for compact and sparse blocks. The doc calls this "per-tree
factorization": Warp `U'U` tile Cholesky, MuJoCo's sparse reverse-mode `L'DL`, or a diagonal inverse.
Numerically, UᵀU Cholesky gives slightly different rounding from C's LᵀDL. The comment notes that dense
tiles win up to ~64 dofs and degrade past ~80.

### 10.5 Sleeping and compact solver

With `SLEEP` (Newton only, islands enabled): `update_active_dofs` packs awake dofs into `[0, ncdof)`
(`dof_cdof`, `cdof_dof`). Smooth acceleration and the constrained solve run densely at size `nvmax_pad`
(`smooth_solve_compact`, `solve_compact` gather M/J/vectors, run the stock dense Newton at `nvmax_pad`,
then scatter back, freezing inactive dofs to 0). Collision runs twice (full pass, then an incremental
pass for newly woken bodies). Recommendations: choose a tight `nvmax`, initialize trees asleep (`--init_asleep`,
`d.tree_asleep = arange(ntree)`), and consider raising `sleep_tolerance` to 0.01.

### 10.6 Benchmarks (`benchmarks/*/__init__.py`, nightly)

| Name | nworld | nconmax | njmax | nccdmax | other |
|---|---|---|---|---|---|
| humanoid | 8192 | 24 | 64 | | |
| three_humanoids | 8192 | 100 | 192 | | |
| franka_emika_panda | 32768 | 1 | 5 | | |
| franka_threading | 8192 | 4096 | 4096 | 512 | replay, noise_rate 1e-6 |
| unitree_g1_flat / _hfield | 8192 | 48 | 192 | | replay, warn_overflow=~ITERATIONS\|~LS_ITERATIONS |
| myoarm | 8192 | 16 | 48 | | |
| aloha_pot | 8192 | 24 | 128 | 1 | replay |
| aloha_sdf | 8192 | 32 | 226 | | |
| aloha_cloth | 256 | 3600 | 6400 | 3000 | |
| aloha_clutter | 2048 | 256 | 384 | 16 | nvmax 56, SLEEP, init_asleep |
| cloth | 256 | 3600 | 3600 | 3600 | |
| render: primitives, *_render, aloha_clutter_vision | | | | | function=render |

Example output: humanoid ≈ 2.73M steps/s at 8192 worlds (README). The ctrl noise used by testspeed is
an Ornstein–Uhlenbeck process with Halton-sequence noise (`cli._ctrl_noise`: `rate = exp(−dt/noise_rate)`,
`scale = std·sqrt(1−rate²)`), so benchmark runs are reproducible.

Throughput-versus-latency position (doc): MJWarp maximizes steps per second over many worlds. A single
step is slower than C MuJoCo, and it scales worse than C for single large trees (beyond ~60 dofs),
unless sleeping can divide the scene into islands.

---

## 11. Validation methodology (parity with C MuJoCo)

### 11.1 Test harness (`mujoco_warp/test_data/__init__.py: fixture`)

```
np.random.seed(42)
mjm = MjModel.from_xml_{path,string}; override_model(mjm, overrides)
mjd = MjData; mj_resetData or mj_resetDataKeyframe
optional uniform noise on qpos / qvel / ctrl / qfrc_applied / xfrc_applied / mocap
if any noise: mj_step(mjm, mjd, 10)        # move away from identity rotations / zero velocities
mj_forward(mjm, mjd); mjd.qacc_warmstart = mjd.qacc
m = put_model(mjm, batch_sizes); override_model(m, overrides)   # warp-only overrides (opt.deterministic, ...)
d = put_data(mjm, mjd, nworld, nconmax, njmax, nvmax, njmax_nnz)
```

Standard pattern (CONTRIBUTING.md): fill the outputs under test with `wp.inf` (floats) or `-1` (ints),
call the single MJWarp stage (`mjw.fwd_velocity`, `mjw.collision`, `mjw.solve`, ...), call the matching C
function (`mj_forward`, `mj_sensorPos`, `mj_step`, `mj_collision`, `mj_RungeKutta`, ...), and compare
world 0 with `np.testing.assert_allclose(atol=tol, rtol=tol)`. For batching changes, parameterize
`nworld ∈ {1, 2}`, perturb world 1, compute per-world C references, and assert the two worlds differ.
Scenes must use nonzero velocities and rotations.

### 11.2 Tolerances used (what parity is achievable)

| Area | Tolerance | Notes |
|---|---|---|
| CONTRIBUTING default | `rtol=atol=1e-5` | aspirational for simple fields |
| `forward_test` | `_TOLERANCE=5e-5`, compared at `×10 = 5e-4` | fwd_velocity, actuation, act, qfrc_bias, step1/step2 fields |
| step2 with IMPLICIT | `5e-3` | |
| RK4 qpos | `atol=2e-5` | |
| implicit qDeriv / qvel | `1e-3` | |
| `smooth_test`, `sensor_test`, `passive_test`, `ray_test`, `support_test`, `derivative_test`, `collision_driver_test` | `_TOLERANCE=5e-5` (×10 in `_assert_eq`) | |
| `constraint_test` | `5e-4` | efc rows compared **after lexsort** on (type, id, D, aref, vel, pos) on both sides, because row order differs |
| `flex_test` | `5e-4` | |
| `solver_test` | `_TOLERANCE=5e-3`, compared at `×20 = 0.1` (!) | Newton `qacc`, `qfrc_constraint`, `efc_force`; CG compared only through cost |
| solver cost criterion | `cost_mjwarp ≤ 1.025 · cost_C` | cost evaluated with C's `mj_constraintUpdate` at each qacc; the solution may be a different (equally good) point of a flat optimum |
| noslip | `1e-4` | elliptic cone-boundary states excluded (rounding-dependent) |
| collision (GJK/EPA) | `rtol=5e-2, atol=1e-2` on dist, pos, frame; **each C contact must match some MJWarp contact** (order-free), and `nacon == ncon` | |
| collision primitive core | `atol=1e-6` | unit-level |
| `history_test` | `1e-8`, `1e-6` | integer/time bookkeeping |
| `determinism_test` | ATOMICS vs NONE `1e-3`; ATOMICS vs ATOMICS bitwise | |
| MJX-Warp `forward_test` | `qacc rtol=1e-5, atol=1.0`; other fields via `test_util.assert_eq` (`_TOLERANCE·10`) | |
| `unroll_test` | coarse task outcome (Aloha lifts pot: `pot z > 0.069`, `lid z > 0.16`) for both cones | catches long-horizon instability |

What these numbers mean for a reimplementation:

* Single-stage, non-iterative quantities (kinematics, com, crb/M, rne, passive, actuation, sensors)
  should agree with float64 C to about **1e-5 to 5e-4** relative.
* Collision geometry agrees to about **1e-2** absolute for GJK/EPA (different algorithm details and f32).
  Analytic primitives agree to about 1e-6.
* Constraint rows agree to about 5e-4 **as a multiset**. Row and contact order is not defined.
* Solver outputs agree to about **1e-2 to 1e-1**, plus cost within 2.5%. The tolerance clamp (1e-6 vs
  1e-8), the always-warmstart policy, and f32 linesearch tolerances all move the converged point.
* Multi-step trajectories diverge chaotically. Parity is validated per stage and with qualitative
  long-horizon tests, not by comparing trajectories.

### 11.3 Other checks

* `io_test` covers every `NotImplementedError`/`ValueError` gate, batch_sizes validation, reset semantics
  and override parsing. `io_jax_test` ("do not edit without consulting the MJX team") checks that
  `put_data`/`make_data` produce identical field shapes for several models (pendula, tactile, floppy flex,
  tendon force limit, hfield): the leading-dimension contract that MJX relies on.
* `types_test` checks dataclass/annotation consistency. `kernel_analyzer` enforces signature conventions.
* `testspeed` provides `--measure_alloc` (contacts/constraints per step), `--measure_solver` (iterations),
  `--memory`, `--event_trace`, `--overflow_behavior`, `--format`, `--function`.
* Benchmarks require zero overflow and `converged_worlds == nworld`.

---

## 12. GPU (MuJoCo Warp) implementation summary

This whole file describes the GPU implementation. The checklist below collects what a GPU
reimplementation must reproduce:

1. **Layout**: world-major `(nworld, n)` Data; flat global contact pool `(naconmax)` with `worldid`;
   per-world efc `(nworld, njmax)` with padded `njmax_pad` / `nv_pad`; CSR M in MuJoCo's layout; batched
   Model fields read with `worldid % B`.
2. **Parallel axes**: per (world, element) threads; per-world thread blocks for reductions and dense
   linear algebra (Newton H, Cholesky, linesearch, CG updates, PCG); global grid-stride over contacts and
   collision pairs; per-branch tree walks; contiguous-subtree sums.
3. **Precision**: float32 throughout; tolerance ≥ 1e-6; linesearch `gtol ≥ 1e-6`; fast_math in the
   linesearch.
4. **Algorithmic differences from C**: always warmstart from `qacc_warmstart` when nefc>0 (C picks the
   lower-cost of warmstart and qacc_smooth); sparse efc at nv>32 (C: ≥60); per-tree M factor (UᵀU tiles
   ≤64 / scalar ≤6 / sparse LDLᵀ) with the factorization done in `fwd_acceleration` in `forward`; Newton
   incremental Hessian; CG/Newton convergence tests as in §7.2 (C additionally requires the gradient
   criterion for Newton's zero-iteration exit); different collider set and contact counts (plane–mesh 4,
   hfield 4, box-box CCD 4 unless NATIVECCD is disabled); collision sensors share the collision pass;
   no midphase; static geoms computed once; no PGS; no plugins except SDF.
5. **Fixed buffers** with drop-and-flag overflow (§5).
6. **Non-determinism** from atomics and slot allocation; optional RUN_TO_RUN reductions (§6).
7. **CUDA graph** capture with a conditional while loop for the solver (§7).

MJX-JAX (pure JAX) is a separate re-implementation with its own limits (§2.2): dense-only Jacobian in
practice, fixed iteration counts, SAT for meshes, no flex/sleep/plasticity. MJX-Warp is MJWarp through
the FFI.

---

## 13. Reproduction pitfalls / parity notes

1. **Default solver tolerance is not the MJCF value.** `put_model` raises it to 1e-6 when smaller
   (MuJoCo default 1e-8). Solver convergence, iteration counts and final forces all depend on this.
2. **Warmstart policy differs**: with warmstart enabled and `nefc > 0`, MJWarp always starts from
   `qacc_warmstart`. C's `warmstart()` compares the cost at `qacc_warmstart` (plus the Gauss term) against
   `qacc_smooth` and picks the smaller. Without constraints MJWarp uses `qacc_smooth`. With
   `run_qacc_smooth=False` it always uses `qacc_warmstart`.
3. **Sparse/dense switch at nv > 32**, not nv ≥ 60. This changes summation order in Jaref/jv and the
   Hessian.
4. **M factorization is UᵀU Cholesky for trees of 7–64 dofs** (tile), scalar Cholesky for ≤ 6, and C's
   LᵀDL only beyond 64. `qLD` contents are **not** comparable to C's `qLD`. Only solves (`qacc_smooth`)
   are comparable.
5. **Where factorization happens**: in `forward()`, `factor_m` is skipped in `fwd_position` and done
   fused with the solve in `fwd_acceleration`. A pipeline that reads `qLD` between those stages sees
   stale data.
6. **Contact order and constraint row order are arbitrary** (atomics). Any comparison or hash must sort
   first (`constraint_test` lexsorts by type, id, D, aref, vel, pos). `DeterminismType.CONTACTS` and
   `CONSTRAINT` exist but are no-ops at this commit.
7. **Float atomic summation order** changes low bits between runs unless `deterministic |= ATOMICS`.
   Even then results match only up to ~1e-3 between NONE and ATOMICS.
8. **Contacts are a global pool**: `nconmax` is not a per-world limit, but `njmax` is a hard per-world
   limit. Overflowed contacts/rows are silently dropped (bit in `d.overflow`). Survivors depend on
   atomic order.
9. **Collision-sensor geom pairs always produce one contact** (type SENSOR) even when far apart. Do not
   count them as physical contacts. Filter on `type & (CONSTRAINT|PASSIVE)` as `get_data_into` does.
10. **Different max contact counts**: plane–mesh up to 4 (C 3), hfield CCD up to 4 (C up to 50),
    box–box via CCD (≤4) by default; `DSBL_NATIVECCD` switches to the primitive box–box collider (≤8).
    The meaning of NATIVECCD is inverted relative to C's MPR/GJK choice.
11. **Static geoms are never recomputed.** After changing `geom_pos/quat` or `body_pos/quat` of a body
    welded to the world, `geom_xpos/xmat` stay stale unless Data is recreated.
12. **Option values are compile-time constants**: integrator, solver, cone, iterations, ls_iterations,
    flags, warn_overflow, deterministic, broadphase, and field *batch sizes* are baked into kernels and
    graphs. Re-capture after changing them. `has_fluid` and other `has_*` flags do not update themselves.
13. **Solver loop length**: with conditional graphs the loop ends when every world has converged or
    reached `iterations`. Without them it always runs `iterations` launches. Per-world results are the
    same; throughput is not.
14. **Convergence counting**: `solver_niter` is incremented inside the convergence kernel, after
    linesearch and gradient update. CG and Newton termination formulas are in §7.2; `improvement > 0`
    must hold for the improvement criterion; `alpha == 0` means done.
15. **Linesearch tolerances**: `gtol = max(tol·ls_tol·‖s‖/scale, 1e-6)`, noise floor
    `8·eps_f32·|q1|`, fast_math. These change the chosen step sizes compared with C.
16. **Quaternion normalization in kinematics** works on local copies: free and ball joint quaternions
    taken from qpos are `wp.normalize`d (qpos is not written back), and every body's accumulated `xquat`
    is normalized again before it is stored (`_kinematics_branch`). Integration uses
    `math.quat_integrate`.
17. **impratio** is stored as `impratio_invsqrt = 1/sqrt(max(impratio, 1e-15))`. Randomize this field,
    not `impratio`.
18. **RK4 under sleep/plasticity** saves and restores extra Data fields. Its scratch is allocated per call
    (fine under graph capture: fixed addresses).
19. **Euler implicit damping** uses a scratch copy of M and a separate factorization. `d.qLD` is not
    overwritten by Euler.
20. **set_const** must be called after randomizing primary inertial/geometric fields. Otherwise derived
    fields (`body_invweight0`, `dof_invweight0`, `actuator_acc0`, `geom_aabb`, ...) are stale and the
    constraint regularization (`efc_D`, `efc_R`) uses wrong weights.
21. **Quaternion convention** is MuJoCo (w,x,y,z) even though the storage type is `wp.quat`. Never use
    Warp's builtin quaternion ops on these values.
22. **MJX-Warp**: `make_data` takes `naconmax` (all worlds), not `nconmax`; batch dims are added and
    squeezed automatically; batched Model fields keep their batch dim only if listed in `batch_sizes`;
    use `Data.where` for resets.
23. **CUDA < 12.4**: conditional graphs are unavailable; set `m.opt.graph_conditional = False`, or the
    capture will fail or be unreliable (< 12.3).
24. **Upgrades**: a stale Warp kernel cache can cause wrong physics. Clear it (`wp.clear_kernel_cache()`
    or delete `~/.cache/warp`).

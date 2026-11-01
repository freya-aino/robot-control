# MuJoCo Reimplementation Dictionary — Index

**Purpose.** This folder is an exhaustive, source-derived specification of MuJoCo's mathematics, algorithms and programmatic optimizations. It was written so that an agent that reads only these files can rebuild the simulator and reproduce its results, including the GPU backend MuJoCo Warp (MJWarp).

**Snapshot.**

| Repo | Commit | Date |
|---|---|---|
| google-deepmind/mujoco (C engine, compiler, MJX) | `7ba52b53300d84d22f7b0bfa1fbbe7557058bc46` | 2026-10-30 |
| google-deepmind/mujoco_warp (GPU) | `d41a47c0837fc71abb44d7d4799bf85ca19ff7ab` | 2026-10-31 |

Everything here is tied to these commits. MuJoCo moves fast (new integrators, IPC, sleeping and history buffers all landed recently), so re-check the changelog (`doc/changelog.rst`) before trusting a detail against a newer checkout.

---

## 1. What "identical" can mean — read first

There are three different reproduction targets, and they are not equally achievable:

| Target | Achievable? | What it takes |
|---|---|---|
| **Bit-identical to C MuJoCo (CPU, float64)** | Yes, with discipline | Same operation order and summation order, the same AVX-vs-scalar choice, and **no FMA and no fast-math**. ARM/Clang contracts multiply-adds into FMA by default; disable it with `-ffp-contract=off`. Same element ordering from the compiler. See 12 §13 and the pitfall sections of every file. |
| **Bit-identical to MuJoCo Warp (GPU, float32)** | Only within one GPU, driver and Warp version, with `DeterminismType.ATOMICS` | MJWarp is float32-only. Contact and constraint row order comes from atomics, and the canonical-sort modes (`CONTACTS`, `CONSTRAINT`) are empty stubs at this commit. See 15 §6. |
| **Behaviourally equivalent to MJWarp** | Yes | Implement the MJWarp algorithm variants listed in §4 below. MJWarp's own tests accept about 5e-4 per stage, about 1e-2 for convex collision geometry, and 10% relative (cost within 2.5%) for solver outputs against C (15 §11). |

Important: **MJWarp is not a port of C MuJoCo at float32.** In many places it runs a different algorithm (§4). To match it on GPU, follow the GPU section of each file, not the C section.

---

## 2. File dictionary

| # | File | Covers | Main sources |
|---|---|---|---|
| 00 | `00_INDEX.md` | This index, source map, parity targets, cross-cutting divergences | — |
| 01 | `01_pipeline_integrators_datamodel.md` | `mj_step`/`mj_forward` stage order, every `mjOption` default and flag, warmstart, integrators (Euler, implicit, implicitfast, discrete, RK4 tableau), activation and history integration, `mjModel`/`mjData` layout, arena/stack allocator, callbacks, plugins, thread pool | `engine_forward.c`, `engine_io.c`, `engine_memory.c`, `engine_init.c`, `engine_callback.c`, `engine_plugin.cc`, `engine_thread.cc`, `engine_name.c`, `engine_print.c`; Warp `forward.py`, `types.py`, `io.py`, `history.py` |
| 02 | `02_kinematics_dynamics_smooth.md` | Forward kinematics, quaternion and spatial algebra, CoM frames, composite rigid body, sparse mass matrix and LDLᵀ factorization, RNE, tendons (wrapping, pulleys), transmissions, Jacobians, `mj_rnePostConstraint` | `engine_core_smooth.c`, `engine_core_util.c`, `engine_util_spatial.c`; Warp `smooth.py`, `math.py`, `util_misc.py` |
| 03 | `03_collision_driver_broadphase.md` | Collision function table, filters, PCA-axis sweep-and-prune, midphase BVH, contact parameter mixing, contact frames, speculative CCD, continuous collision for deformables, IPC step for 2D flexes | `engine_collision_driver.c`, `engine_sort.h`, `engine_collision_continuous.c`, `engine_ipc.c`; Warp `collision_driver.py`, `collision_core.py`, `bvh.py` |
| 04 | `04_collision_primitives.md` | Every analytic collider (plane/sphere/capsule/cylinder/box), box–box SAT with clipping, capsule–box heuristic | `engine_collision_primitive.c`, `engine_collision_box.c`; Warp `collision_primitive.py`, `collision_primitive_core.py` |
| 05 | `05_collision_convex_gjk_epa.md` | Support functions (mesh hill-climbing), native GJK, EPA, MPR (libccd) path, MultiCCD, height-field prisms, mesh–mesh | `engine_collision_convex.c`, `engine_collision_gjk.c`; Warp `collision_gjk.py`, `collision_convex.py` |
| 06 | `06_collision_sdf_flex.md` | Octree SDF, SDF contact optimizers (gradient descent, Frank–Wolfe, Halton), analytic SDF plugins, flex collision primitives and self-collision | `engine_collision_sdf.c`, `engine_collision_flex.c`, `plugin/sdf/*`; Warp `collision_sdf.py`, `collision_flex.py` |
| 07 | `07_constraints.md` | Soft-constraint formulation (Gauss principle, primal/dual), every constraint type's Jacobian, solref/solimp → impedance, R, aref, `mj_constraintUpdate` and the cone Hessian | `engine_core_constraint.c`; Warp `constraint.py` |
| 08 | `08_solvers_islands.md` | PGS, CG (Hager–Zhang / Polak–Ribière), Newton with rank-1 Cholesky updates, exact line search, NoSlip, islands, warmstart choice | `engine_solver.c`, `engine_island.c`, `engine_util_solve.c` (solver parts); Warp `solver.py`, `block_cholesky.py`, `island.py` |
| 09 | `09_passive_fluid_flex_elasticity.md` | Springs and dampers, gravity compensation, inertia-box and ellipsoid fluid models, passive contact and adhesion, flex elasticity (StVK, stable Neo-Hookean, bending), cable plugin | `engine_passive.c`, `plugin/elasticity/*`; Warp `passive.py` |
| 10 | `10_actuation_sensors_plugins.md` | Actuator dynamics, gain and bias types (muscle, PID, DC motor with LuGre/cogging), force clamps, history and delay buffers, every sensor type, sensor plugins | `engine_forward.c` (actuation), `engine_util_misc.c` (muscle), `engine_sensor.c`, `plugin/actuator/*`, `plugin/sensor/*`; Warp `sensor.py`, `forward.py`, `history.py` |
| 11 | `11_derivatives_inverse_dynamics.md` | Analytic velocity derivatives (used by implicit), finite-difference transition and inverse Jacobians, inverse dynamics, the implicit effective metric M + hD + h²K behind `integrator="discrete"` (`engine_metric.c`), flex stiffness operators and the passive flex-contact law. MJWarp has no FD derivatives; MJX uses JAX autodiff | `engine_derivative.c`, `engine_derivative_fd.c`, `engine_inverse.c`, `engine_metric.c`; Warp `derivative.py`, `inverse.py` |
| 12 | `12_linear_algebra_sparse_numerics.md` | BLAS and AVX paths with exact summation order, CSR and supernodes, dense/sparse/band Cholesky, LU, eig3, QCQP, boxQP, sort macros, misc numerics, build flags for bitwise parity | `engine_util_blas.c`, `*_avx.h`, `engine_util_sparse.c`, `engine_util_solve.c`, `engine_util_misc.c`, `engine_util_errmem.c`; Warp `math.py`, `block_cholesky.py`, `warp_util.py` |
| 13 | `13_model_compiler_setconst.md` | Compile order and element numbering, sparse structure precompute, frames and units, inertia from geoms and meshes, qhull and mesh graph, BVH and octree, defaults, flexcomp, fusestatic, `mj_setConst`, `mj_setLengthRange` | `src/user/*.cc`, `src/xml/*`, `engine_setconst.c`; Warp `set_const.py`, `io.py` |
| 14 | `14_ray_sleep_support_render.md` | Ray casting for every geom type, sleeping and waking, state API, support functions, perturbations, renderers, batch API, bindings | `engine_ray.c`, `engine_sleep.c`, `engine_support.c`, `engine_vis_*.c`, `src/render`, `src/experimental`, `src/batch`; Warp `ray.py`, `sleep.py`, `support.py`, `render.py`, `render_util.py` |
| 15 | `15_gpu_architecture_mjwarp_mjx.md` | MJWarp architecture, C vs MJWarp vs MJX feature matrix, float32 policy, per-stage kernel and launch tables, fixed buffers and overflow, determinism, CUDA graph capture and conditional loops, batching and domain randomization, JAX interop, validation tolerances | Warp `types.py`, `warp_util.py`, `forward.py`, `io.py`, tests, benchmarks; `doc/mjwarp`, `doc/mjx.rst` |
| 16 | `16_flex_elasticity_precompute.md` | Compile-time flex stiffness and bending coefficients, interpolated (trilinear/quadratic) flex shape functions, FCM quadrature, Neo-Hookean Hessian projection, plasticity, polar rotation and deformation gradient | `user_flexcomp.cc`, `user_objects.cc` (flex), `engine_passive.c`, `engine_util_misc.c`; Warp `passive.py`, `io.py` |

`_AGENT_BRIEF.md` holds the writing contract each file followed (structure, required sections). It is kept for provenance.

---

## 3. Recommended build order for a reimplementation

1. **12** numerics primitives → **02** kinematics, CRB, `M`, RNE (validate `qM`, `qfrc_bias` against C on a few models).
2. **13** compiler outputs. Either load real `mjModel`s from MuJoCo directly (recommended) or reimplement the compiler. Element ordering and inertia frames must match exactly.
3. **01** pipeline skeleton plus the Euler integrator with no contacts → **09** passive forces → **10** actuation and sensors.
4. **03** → **04** → **05** → **06** collisions (validate contact count, `dist`, `pos`, `frame` per pair type).
5. **07** constraints → **08** solvers (validate `efc_*`, `qacc`, `solver_niter`).
6. **11** derivatives → implicit/implicitfast integrators; RK4; discrete/IPC; **16** flex.
7. **14** ray, sleep and support; **15** GPU port: batching, fixed buffers, kernels, graph capture.

Validate at every stage against C MuJoCo (`mj_forward` field by field). The tolerance tables in 15 §11 give the realistic bar for a float32 port.

---

## 4. Cross-cutting C ↔ MuJoCo Warp divergences (consolidated)

Each row is detailed in the cited file. To reproduce **GPU** behaviour, implement the MJWarp column.

| Area | C MuJoCo | MuJoCo Warp | File |
|---|---|---|---|
| Precision | float64 (`mjtNum`) | float32 only, no float64 option | 15 §3 |
| Solver tolerance | default 1e-8 | raised to at least 1e-6 in `put_model`; line-search gradient tolerance at least 1e-6 | 01, 08, 15 |
| Sparse Jacobian (AUTO) | `nv >= 60` | `nv > 32` | 01, 07, 15 |
| Warmstart | picks the cheaper of `qacc_warmstart` and `qacc_smooth` | always `qacc_warmstart` when constraints exist | 01, 08 |
| Solvers | PGS, CG, Newton; per-island solving | CG and Newton only; no per-island solve; parallel line search | 08 |
| CG direction update | Hager–Zhang (default) | Polak–Ribière-plus | 08 |
| `solver_niter` | a zero-step iteration is not counted | it is counted | 08 |
| Integrator RHS | `qfrc_smooth + qfrc_constraint` | `M·qacc` | 01 |
| RK4 | time advanced at stages | time not advanced at intermediate stages; sensors with history recomputed per stage | 01, 10 |
| Box–box | analytic `mjc_BoxBox`, ≤ 8 contacts | GJK/EPA + MultiCCD (≤ 4) unless NATIVECCD is disabled; EPA capped at 16 iterations when box–box is the only convex type | 04, 05 |
| Plane–box | ≤ 4 contacts (lower-half corners) | ≤ 8 contacts, no lower-half filter | 04 |
| Height field | ≤ 50 contacts per pair, true surface distance | ≤ 4 contacts, distance not corrected for margin | 05 |
| Convex margins | inflate each shape by (margin+gap)/2 | inflate by margin, gap used as cutoff | 05 |
| Mesh–SDF | triangle / Frank–Wolfe method | ray-cast pseudo-distance branch | 06 |
| Flex contact reduction | only rigid/interpolated flexes reduced | 1 mm dedup + farthest-point sampling for all self/flex–flex groups | 06 |
| Contact params, direct solref, unequal priority | copy higher-priority geom's solref | elementwise minimum | 03 |
| Contact / constraint row order | deterministic | atomics order; sort modes are stubs | 03, 07, 15 |
| Midphase | BVH per mesh/flex | none for collisions (BVH only for ray/render) | 03 |
| Joint/tendon limits | rows for both sides possible | only the nearer side | 07 |
| Impedance | not clamped to [dmin, dmax] | clamped | 07 |
| J̇v in connect/weld | current `cvel` | `cvel` from the previous step (constraints built before `com_vel`) | 07 |
| Overflow | allocation error | silently drop rows/contacts/pairs beyond `naconmax`/`njmax`/`njmax_nnz`, set overflow flag; pair buffer = contact buffer size | 03, 07, 15 |
| NaN checks / autoreset | yes (`mj_checkPos/Vel/Acc`) | none | 01 |
| History time match | 1e-15 | 1e-6 | 01, 10 |
| Actuator clamps | C order | different order; only first ctrl clamped | 10 |
| Touch / limit sensors | adhesion subtracted, cutoff applied; first matching row | neither; last matching row | 10 |
| Interpolated flex polar decomposition | 500 iterations, 1e-9 | 50 iterations, 1e-6 | 09 |
| Sleep timing | before integration; recompute vel+acc | after integration; recompute vel only; no wake on `qpos` edit | 14 |
| Mesh rays | back-face hits reported | back faces skipped | 14 |
| `set_const` | `1/mass` shortcut for simple bodies; BVH refit, fluid added mass | no shortcut, no refit, different subtree-mass summation order, extra invweight fallback | 13 |
| Analytic `qDeriv` (implicit) | all actuators incl. multi-output/SO3; full asymmetric term under `implicit` | single-output actuators only, no SO3; asymmetric part dropped; extra free-body correction; no concave-stencil filter for discrete bending | 11 |
| Finite-difference derivatives | `mjd_transitionFD`, `mjd_inverseFD` | not available | 11 |
| Discrete-integrator smooth solve | C tolerance | relative tolerance floored at 1e-5 | 11 |
| MJX passive flags | — | (MJX, not Warp) zero passive force if either spring or damper flag is disabled | 09 |

---

## 5. Notable C-side issues found while writing

These are behaviours a faithful bit-exact reimplementation must **copy**, not fix:

- `mj_setLengthRange` writes its result to slot `2*index+side` but checks `2*out`. These differ when a multi-output actuator comes earlier (13 §14).
- The length-range simulation **assigns** disable flags instead of OR-ing them, so user limit and equality disables are dropped during it (13 §14).
- SDF pair buffers are sized by `sdf_initpoints`, but `mjc_MeshSDF` can return up to 50 contacts, so the buffer can overflow if `sdf_initpoints < 50` (06).
- Unreachable branches: farthest-point sampling in `mjc_MeshSDF` and the BVH path of `mjc_FlexSDF` (06).
- `mju_dotSparseX3` is the one reduction whose summation order differs between AVX and non-AVX builds (12).
- The compiler's `mjuu_eig3` and the engine's `mju_eig3` sort eigenvalues with different thresholds, which affects inertia principal frames (12, 13).
- `mj_multiRay` culls with an unrotated bounding-sphere centre and can drop real hits (14).
- `mjd_transitionFD` with centered differences: `clampedDiff` calls `diff(dx, x_plus, x_minus, 2h)`, and `diff` computes `(x2-x1)/h`. The sensor-vs-control block `D` therefore comes out as `(s(u-ε) - s(u+ε))/2ε`, a sign flip. Only `clampedDiff` has this; the state blocks use `clampedStateDiff`. Verified in source (11 §10).
- The finite-difference buffers are sized without room for the plastic state, so models with `nplastic > 0` are effectively unsupported in `transitionFD` (11).
- The Warp `quat_z2vec` returns different quaternions from C in degenerate (anti-)parallel cases (12).

---

## 6. Source → spec map

Every engine, compiler and MJWarp source file, mapped to where it is specified. Headers (`*.h`) are covered through their `.c` counterparts.

**C engine `src/engine/`**

| Source | Spec |
|---|---|
| engine_forward.c | 01 (pipeline, integrators), 10 (actuation) |
| engine_io.c, engine_memory.c, engine_init.c, engine_macro.h, engine_global_table.h | 01 |
| engine_callback.c, engine_plugin.cc, engine_thread.cc, engine_name.c, engine_print.c | 01 |
| engine_core_smooth.c, engine_core_util.c, engine_util_spatial.c | 02 |
| engine_collision_driver.c, engine_sort.h, engine_collision_continuous.c, engine_ipc.c | 03 |
| engine_collision_primitive.c, engine_collision_box.c | 04 |
| engine_collision_convex.c, engine_collision_gjk.c | 05 |
| engine_collision_sdf.c, engine_collision_flex.c | 06 |
| engine_core_constraint.c | 07 |
| engine_solver.c, engine_island.c | 08 |
| engine_passive.c | 09, 16 |
| engine_sensor.c | 10 |
| engine_derivative.c, engine_derivative_fd.c, engine_inverse.c, engine_metric.c | 11 |
| engine_util_blas.c, engine_util_blas_avx.h, engine_util_sparse.c, engine_util_sparse_avx.h, engine_util_solve.c, engine_util_errmem.c, engine_array_safety.h, engine_crossplatform.* | 12 |
| engine_util_misc.c | 12 (general), 10 (muscle), 16 (flex helpers), 01 (history) |
| engine_setconst.c | 13 |
| engine_ray.c, engine_sleep.c, engine_support.c, engine_vis_init.c, engine_vis_interact.c, engine_vis_visualize.c | 14 |
| engine_inline.h | 12 (inlined helpers) |

**Compiler `src/user/`, `src/xml/`**

| Source | Spec |
|---|---|
| user_model.cc, user_objects.cc, user_mesh.cc, user_composite.cc, user_util.cc, user_api.cc, user_init.c, `src/xml/*` | 13 |
| user_flexcomp.cc (+ flex parts of user_objects.cc) | 13 (generation), 16 (elasticity precompute) |
| user_cache.cc, user_resolver.cc, user_resource.cc, user_vfs.cc, user_threadpool.cc | Not specified: asset caching, file resolution, virtual filesystem and compile thread pool. They do not affect numerical results. |

**Plugins `plugin/`**

| Source | Spec |
|---|---|
| plugin/sdf | 06 |
| plugin/elasticity (cable) | 09 |
| plugin/actuator (PID), plugin/sensor (touch_grid) | 10 |
| plugin/*_decoder (obj, stl, gmsh, usd) | Asset loaders only, no physics. |

**MuJoCo Warp `mujoco_warp/_src/`**

| Source | Spec |
|---|---|
| forward.py | 01, 10, 15 |
| types.py, io.py, warp_util.py, history.py, util_pkg.py, cli.py | 01, 15 |
| smooth.py, math.py, util_misc.py | 02, 12 |
| collision_driver.py, collision_core.py, bvh.py | 03 |
| collision_primitive.py, collision_primitive_core.py | 04 |
| collision_gjk.py, collision_convex.py | 05 |
| collision_sdf.py, collision_flex.py | 06 |
| constraint.py | 07 |
| solver.py, block_cholesky.py, island.py | 08 |
| passive.py | 09, 16 |
| sensor.py | 10 |
| derivative.py, inverse.py | 11 |
| set_const.py | 13 |
| ray.py, sleep.py, support.py, render.py, render_util.py | 14 |
| `*_test.py`, determinism_test.py | 15 §11 (validation and tolerances) |

**MJX `mjx/mujoco/mjx/_src/`**: covered briefly in the "MJX" subsection of each file, and as a backend in 15.

# 01 — Simulation Pipeline, Integrators, Options and Data Model

**Sources covered**

- C engine (commit 7ba52b5): `src/engine/engine_forward.c` (all), `engine_io.c`, `engine_io.h`, `engine_memory.c`, `engine_memory.h`, `engine_init.c`, `engine_macro.h`, `engine_callback.c`, `engine_plugin.cc` (pipeline hook points), `engine_thread.cc`, `engine_global_table.h`, `engine_name.c` (brief), `engine_print.c` (brief). Helpers that the pipeline calls and whose exact behavior matters for parity are reproduced: `engine_support.c` (`mj_integratePosInd`, `mj_nextActivation`, `mj_actuatorDisabled`, `mj_readCtrl`), `engine_core_util.c` (`mj_isSparse`, `mj_isMetric`, `mj_warning`), `engine_util_misc.c` (`mju_isBad`, history ring buffer, muscle and DC-motor helpers, `mjd_xPolyForce`), `engine_util_spatial.c` (`mju_quatIntegrate`, `mju_derivExpmap`), `engine_core_smooth.c` (`mj_factorI`, `mj_solveLD`), `engine_derivative.c` (`mj_isFreeBody`, `mjd_freeMhat`, `mjd_freeGyroPossible`), and the arena sizing in `src/user/user_model.cc`.
- Public headers: `include/mujoco/mjmodel.h`, `mjdata.h`, `mjtype.h`, `mjmacro.h`, `mjxmacro.h`, `mjplugin.h`.
- MuJoCo Warp (commit d41a47c): `mujoco_warp/_src/forward.py`, `types.py`, `io.py`, `warp_util.py`, `history.py`, `util_pkg.py`, `cli.py` (brief); touched helpers `support.py:next_act`, `solver.py:solve/_solve_init_dof/_add_qfrc_smooth_eff`, `smooth.py:skip_qacc_smooth/factor_solve_i`, `math.py:quat_integrate/deriv_expmap`.
- MJX (JAX): `mjx/_src/forward.py`, `types.py`, `io.py` (differences only).

**Role in the pipeline.** This file describes the top-level driver of MuJoCo: how `mj_step` sequences the position, velocity, actuation, acceleration and constraint stages (each delegated to subsystems specified in other files), how the five integrators (Euler, RK4, implicit, implicitfast, discrete) turn `qacc` into the next state, how actuator activations are integrated, how the state and scratch memory (`mjModel`, `mjData`, arena/stack) are laid out and reset, which `mjOption` fields and flags gate which computations, and where callbacks and plugins enter. The GPU section maps all of this onto MuJoCo Warp's batched `Model`/`Data` (`[nworld, ...]` arrays, fixed-capacity contact and constraint buffers) and lists every algorithmic divergence that changes numerical results.

---

## Table of contents

1. Global constants and type definitions
2. `mjOption`: full field list, defaults, and enums
3. Disable and enable flags: exact effect of each bit
4. Pipeline stage order and the `mjData` field each stage writes
5. Top-level API (`mj_step`, `mj_step1/2`, `mj_forward`, `mj_forwardSkip`)
6. Stage functions (`mj_fwdPosition`, `mj_fwdVelocity`, `mj_fwdActuation`, `mj_fwdAcceleration`, `mj_fwdConstraint`)
7. Warmstart
8. State advancement (`mj_advance`, `advanceStart`, `advanceFinish`, `mj_commit`)
9. Integrators (Euler, implicit/implicitfast, discrete, RK4)
10. Validity checks, warnings, timers
11. Data model: `mjModel` and `mjData` layout, arena/stack allocator, reset
12. Callbacks, plugins, thread pool, global tables, names, printing
13. GPU (MuJoCo Warp) implementation
14. MJX (JAX) differences
15. Reproduction pitfalls and parity notes

---

## 1. Global constants and type definitions

| Symbol | Value | Source / meaning |
|---|---|---|
| `mjtNum` | `double` (or `float` when built with `mjUSESINGLE`) | `mjtype.h` |
| `mjMINVAL` | `1E-15` | minimum value in any denominator |
| `mjMAXVAL` | `1E+10` | values with `|x| > mjMAXVAL` count as bad in qpos, qvel, qacc, ctrl |
| `mjPI` | `3.14159265358979323846` | |
| `mjMINMU` | `1E-5` | minimum friction coefficient (clamp in `mj_assignFriction`) |
| `mjMINIMP`, `mjMAXIMP` | `0.0001`, `0.9999` | impedance clamp |
| `mjMAXCONPAIR` | `50` | max contacts per geom pair |
| `mjMAXTREEDEPTH` | `50` | BVH depth |
| `mjMAXFLEXNODES` | `27` | |
| `mjMINAWAKE` | `10` | minimum number of timesteps before sleeping |
| `mjNEQDATA` / `mjNDYN` / `mjNGAIN` / `mjNBIAS` | 12 / 10 / 10 / 10 | parameter vector lengths |
| `mjNFLUID` / `mjNREF` / `mjNIMP` / `mjNPOLY` / `mjNSENS` | 12 / 2 / 5 / 2 / 3 | |
| `mjNSOLVER` / `mjNISLAND` | 200 / 20 | solver stat slots per island, number of islands with stats |
| `BUFFER_ALIGN` (engine_io.c) | 64 | byte alignment of every array in the `mjModel`/`mjData` buffers |
| `mjLOAD_MULTIPLE` | 2 | name hash table load factor (slots = 2 × objects) |
| `kAwake` (engine_io.c) | `-(1+mjMINAWAKE) = -11` | `tree_asleep` value for a fully awake tree |
| `mjREDZONE` | 32 under ASAN, else 0 | stack red zone |
| `nPOS[4]`, `nVEL[4]` | `{7,4,1,1}`, `{6,3,1,1}` | qpos/qvel widths for FREE, BALL, SLIDE, HINGE |

`mju_isBad(x)` returns `x != x || x > mjMAXVAL || x < -mjMAXVAL` (NaN, or magnitude above 1e10).

`mjDISABLED(x) = (m->opt.disableflags & (x))`, `mjENABLED(x) = (m->opt.enableflags & (x))` (`mjmacro.h`). Both macros require a local variable named `m`.

---

## 2. `mjOption`: full field list and defaults

The defaults are set by `mj_defaultOption` (`engine_init.c`), which first `memset`s the struct to 0 so padding is zeroed. `mj_makeModel` calls it, and the XML/spec compiler then overrides it.

| Field | Type | Default | Meaning / consumers |
|---|---|---|---|
| `timestep` | mjtNum | `0.002` | h; integrators, refsafe clamp, history buffers, slew limits |
| `impratio` | mjtNum | `1` | ratio of friction to normal contact impedance (constraint file) |
| `tolerance` | mjtNum | `1e-8` | main solver tolerance |
| `ls_tolerance` | mjtNum | `0.01` | CG/Newton linesearch tolerance |
| `noslip_tolerance` | mjtNum | `1e-6` | noslip tolerance |
| `ccd_tolerance` | mjtNum | `1e-6` | convex collision tolerance |
| `sleep_tolerance` | mjtNum | `1e-3` | sleep velocity tolerance |
| `gravity[3]` | mjtNum | `(0, 0, -9.81)` | |
| `wind[3]` | mjtNum | `(0, 0, 0)` | fluid models |
| `magnetic[3]` | mjtNum | `(0, -0.5, 0)` | magnetometer |
| `density` | mjtNum | `0` | fluid |
| `viscosity` | mjtNum | `0` | fluid |
| `o_margin` | mjtNum | `0` | contact override (with `mjENBL_OVERRIDE`) |
| `o_solref[2]` | mjtNum | `(0.02, 1)` | from `mj_defaultSolRefImp` |
| `o_solimp[5]` | mjtNum | `(0.9, 0.95, 0.001, 0.5, 2)` | from `mj_defaultSolRefImp` |
| `o_friction[5]` | mjtNum | `(1, 1, 0.005, 0.0001, 0.0001)` | |
| `integrator` | int | `mjINT_EULER` (0) | `mjtIntegrator` |
| `cone` | int | `mjCONE_PYRAMIDAL` (0) | `mjtCone` |
| `jacobian` | int | `mjJAC_AUTO` (2) | `mjtJacobian` |
| `solver` | int | `mjSOL_NEWTON` (2) | `mjtSolver` |
| `iterations` | int | `100` | |
| `ls_iterations` | int | `50` | |
| `noslip_iterations` | int | `0` | >0 enables the noslip post-pass |
| `ccd_iterations` | int | `35` | |
| `disableflags` | int | `0` | `mjtDisableBit` |
| `enableflags` | int | `0` | `mjtEnableBit` |
| `disableactuator` | int | `0` | bit g disables actuators of group g (only groups 0..30) |
| `sdf_initpoints` | int | `40` | |
| `sdf_iterations` | int | `10` | |

Defaults of the other structs set at the same time: `mj_defaultSolRefImp` gives solref `(0.02, 1)` and solimp `(0.9, 0.95, 0.001, 0.5, 2)`. `mj_defaultStatistic` gives center 0, extent 2, meaninertia 1, meanmass 1, meansize 0.2. `mj_defaultLROpt` gives mode `mjLRMODE_MUSCLE`, useexisting 1, uselimit 0, accel 20, maxforce 0, timeconst 1, timestep 0.01, inttotal 10, interval 2, tolrange 0.05. `mj_defaultVisual` sets the visual defaults (not physics-relevant).

### Enums

```
mjtIntegrator: mjINT_EULER=0, mjINT_RK4=1, mjINT_IMPLICIT=2, mjINT_IMPLICITFAST=3, mjINT_DISCRETE=4
mjtCone:       mjCONE_PYRAMIDAL=0, mjCONE_ELLIPTIC=1
mjtJacobian:   mjJAC_DENSE=0, mjJAC_SPARSE=1, mjJAC_AUTO=2   (AUTO: dense if nv<60, sparse otherwise)
mjtSolver:     mjSOL_PGS=0 (dual), mjSOL_CG=1 (primal), mjSOL_NEWTON=2 (primal)
mjtStage:      mjSTAGE_NONE=0, mjSTAGE_POS=1, mjSTAGE_VEL=2, mjSTAGE_ACC=3
mjtWarning:    mjWARN_INERTIA=0, CONTACTFULL, CNSTRFULL, BADQPOS, BADQVEL, BADQACC, BADCTRL, mjNWARNING=7
mjtTimer:      STEP=0, FORWARD, INVERSE, POSITION, VELOCITY, ACTUATION, CONSTRAINT, ADVANCE,
               POS_KINEMATICS, POS_INERTIA, POS_COLLISION, POS_MAKE, POS_PROJECT, COL_BROAD, COL_NARROW, mjNTIMER=15
mjtState bits: TIME=1<<0, QPOS, QVEL, ACT, HISTORY, WARMSTART, CTRL, QFRC_APPLIED, XFRC_APPLIED,
               EQ_ACTIVE, MOCAP_POS, MOCAP_QUAT, USERDATA, PLUGIN, PLASTIC=1<<14 (mjNSTATE=15)
  PHYSICS = QPOS|QVEL|ACT|HISTORY|PLASTIC; FULLPHYSICS = TIME|PHYSICS|PLUGIN
  USER = CTRL|QFRC_APPLIED|XFRC_APPLIED|EQ_ACTIVE|MOCAP_POS|MOCAP_QUAT|USERDATA
  INTEGRATION = FULLPHYSICS|USER|WARMSTART
mjtSleepState: mjS_STATIC=-1, mjS_ASLEEP=0, mjS_AWAKE=1
mjtDyn:   NONE=0, INTEGRATOR, FILTER, FILTEREXACT, MUSCLE, DCMOTOR, PID, USER
mjtGain:  FIXED=0, AFFINE, MUSCLE, DCMOTOR, SO3, PID, USER
mjtBias:  NONE=0, AFFINE, MUSCLE, DCMOTOR, SO3, USER
mjtTrn:   JOINT=0, JOINTINPARENT, SLIDERCRANK, TENDON, SITE, BODY, SO3, UNDEFINED=1000
mjtCtrlInput bits (actuator_ctrlspec): POS=1, VEL=2, FF=4, VOLTAGE=8, NONE=16
mjtCtrlChart (SO3 ctrlspec): EXPMAP=1 (3 controls), QUAT=2 (4 controls)
```

Predicates (`engine_core_util.c`):

- `mj_isSparse(m) = jacobian==SPARSE || (jacobian==AUTO && nv >= 60)`.
- `mj_isPyramidal(m) = cone==PYRAMIDAL`.
- `mj_isDual(m) = solver==PGS || noslip_iterations > 0`.
- `mj_isMetric(m) = integrator==DISCRETE`. This is the option-level gate. `d->efm_active` reports whether the per-step metric build actually ran.
- `mj_effCouplings(m) = mj_isMetric(m) && solver != PGS`. Under PGS, tendon and actuator metric terms are excluded and integrated explicitly.

---

## 3. Disable and enable flags: what each bit changes

### `mjtDisableBit` (`mjNDISABLE = 20`)

| Bit | Name | Exact effect (file:function) |
|---|---|---|
| 1<<0 | `CONSTRAINT` | `mj_makeConstraint` returns with ne=nf=nl=nefc=nJ=nA=nY=0. `mj_collision` returns immediately after `mj_clearEfc` (no contacts at all). |
| 1<<1 | `EQUALITY` | equality rows (`mj_instantiateEquality`, counting) skipped |
| 1<<2 | `FRICTIONLOSS` | dof/tendon frictionloss rows skipped |
| 1<<3 | `LIMIT` | joint/tendon limit rows skipped |
| 1<<4 | `CONTACT` | `mj_collision` returns early (ncon=0), contact rows skipped, `mj_contactPassive` (discrete passive contact) returns 0 |
| 1<<5 | `SPRING` | passive springs off (`mj_springdamper`), spring energy off, plastic return mapping in `advanceFinish` skipped, metric stiffness terms off (`engine_metric.c`) |
| 1<<6 | `DAMPER` | passive damping off, implicit-damping derivative off. In Euler, `dof_damping` is not detected, so the explicit path is taken. |
| 1<<7 | `GRAVITY` | RNE world acceleration not set to −g (`mj_rne`, `mj_rnePostConstraint`), gravcomp off (passive and actuator-level), gravity potential energy off |
| 1<<8 | `CLAMPCTRL` | the ctrl copy is not clamped to `ctrlrange` in `mj_fwdActuation` (also in derivatives) |
| 1<<9 | `WARMSTART` | `warmstart()` cold-starts: qacc = qacc_smooth, efc_force = 0 |
| 1<<10 | `FILTERPARENT` | broadphase does not filter parent-child body pairs |
| 1<<11 | `ACTUATION` | `mj_fwdActuation` zeroes qfrc_actuator and returns. `actuator_velocity` is zeroed in `mj_fwdVelocity`. Activations are not advanced. `mjcb_control` is not called inside `mj_forwardSkip` (it is still called in `mj_step1`). |
| 1<<12 | `REFSAFE` | Disables `solref[0] = max(solref[0], 2*timestep)` for non-discrete integrators. Under discrete, disables replacing over-stiff rows by the resolved row. |
| 1<<13 | `SENSOR` | `mj_sensorPos/Vel/Acc` return immediately |
| 1<<14 | `MIDPHASE` | BVH midphase off (collision driver, flex BVH AABB update in `mj_flex`) |
| 1<<15 | `EULERDAMP` | Euler integrator skips implicit dof damping |
| 1<<16 | `AUTORESET` | `mj_checkPos/Vel/Acc` warn but do not call `mj_resetData` |
| 1<<17 | `NATIVECCD` | uses libccd instead of native CCD. Also disables speculative contacts. |
| 1<<18 | `ISLAND` | `mj_island` returns with nisland=0, so the constraint solve is monolithic. With sleep plus discrete this is an error. |
| 1<<19 | `MULTICCD` | single CCD contact per convex pair |

### `mjtEnableBit` (`mjNENABLE = 8`)

| Bit | Name | Exact effect |
|---|---|---|
| 1<<0 | `OVERRIDE` | contacts use `o_margin`, `o_solref`, `o_solimp`, and `max(mjMINMU, o_friction[i])`. Broadphase AABB inflated by `0.5*o_margin`. |
| 1<<1 | `ENERGY` | `mj_energyPos/Vel` computed. Otherwise `energy[0]=energy[1]=0` (only energy[0..1] zeroed in the position stage). |
| 1<<2 | `FWDINV` | after forward, `mj_compareFwdInv` runs inverse dynamics and writes `solver_fwdinv[0]=‖qfrc_constraint_fwd − qfrc_constraint_inv‖`, `solver_fwdinv[1]=‖qfrc_applied+J'xfrc+qfrc_actuator − qfrc_inverse‖` and restores qfrc_constraint and efc_force |
| 1<<3 | `INVDISCRETE` | discrete-time inverse dynamics (inverse file) |
| 1<<4 | `SLEEP` | sleeping/island-based skipping throughout (sleep file) |
| 1<<5 | `DIAGEXACT` | exact `efc_diagA` via whitened Jacobian Y instead of the approximation |
| 1<<6 | `IPC` | IPC flex-contact mode of the discrete integrator. Requires integrator=DISCRETE, solver=CG, not FWDINV, not SLEEP (`mj_checkDiscrete`). |
| 1<<7 | `SPECULATIVE` | speculative contacts (only if NATIVECCD is not disabled) |

`disableactuator`: `mj_actuatorDisabled(m,i)` returns 1 iff `0 <= group <= 30` and `disableactuator & (1<<group)`. A disabled actuator contributes no force, and its activation is integrated with `act_dot = 0`, though clamping still applies.

---

## 4. Pipeline stage order: which function computes which `mjData` field

Order of `mj_step` for non-RK4 integrators. The ACC stage always runs. POS and VEL can be skipped by `mj_forwardSkip(skipstage)`.

| # | Function (file) | Writes |
|---|---|---|
| 0 | `mj_checkPos`, `mj_checkVel` | `warning[]`, possible full reset |
| 0' | `mj_checkDiscrete` (inside `mj_forwardSkip`) | errors only |
| **POS** | `mj_fwdPosition` | clears `flg_energypos` |
| 1 | `mj_kinematics` | xpos, xquat, xmat, xipos, ximat, xanchor, xaxis, geom_xpos/xmat, site_xpos/xmat |
| 2 | `mj_comPos` | subtree_com, cinert, cdof |
| 3 | `mj_camlight` | cam_xpos/xmat, light_xpos/xdir |
| 4 | `mj_flex` | flexvert_xpos, flexedge_J, flexedge_length, flexelem_aabb, bvh_aabb_dyn (flex) |
| 5 | `mj_tendon` | ten_wrapadr, ten_wrapnum, ten_J, ten_length, wrap_obj, wrap_xpos |
| 5' | `mj_wakeTendon`→`mj_updateSleep` | sleep arrays (only if a tendon woke a tree) |
| 6 | `mj_makeM` = `mj_crb` + `mj_tendonArmature` | crb, M (CSR, nC) |
| 7 | `mj_factorM` | qLD, qLDiagInv (`mjWARN_INERTIA` on clamped pivot) |
| 8 | `mj_collision` (re-run once if `mj_wakeCollision`) | contact[], ncon, bvh_active; `mj_clearEfc` first |
| 8' | `mj_wakeEquality`→`mj_updateSleep` | |
| 9 | `mj_makeConstraint` | efc_type, efc_id, efc_J*, efc_pos, efc_margin, efc_frictionloss, efc_diagA, efc_KBIP, efc_D, efc_R, tendon_efcadr, ne, nf, nl, nefc, nJ |
| 10 | `mj_island` | tree_island, island_*, map_*, dof_island, efc_island, iefc_* (pos-dependent copies) |
| 11 | `mj_projectConstraint` | efc_Y*, efc_AR* (dual solver or DIAGEXACT) |
| 12 | `mj_transmission` | actuator_length, actuator_moment, moment_rownnz/rowadr/colind |
| 13 | `mj_effBuild(m,d,mj_isMetric(m),1)` | efm_* arrays, efm_active, efm sizes (discrete only) |
| — | `mj_sensorPos` (unless skipsensor) | sensordata (pos sensors) |
| — | `mj_energyPos` if ENERGY and !flg_energypos, else energy[0..1]=0 | energy[0] |
| **VEL** | `mj_fwdVelocity` | clears flg_subtreevel, flg_energyvel |
| 14 | flexedge velocity (sparse, skips rigid/interp flexes) | flexedge_velocity |
| 15 | `ten_velocity = ten_J·qvel` (always sparse) | ten_velocity |
| 16 | `actuator_velocity = moment·qvel` (zero if ACTUATION disabled) | actuator_velocity (nout) |
| 17 | `mj_comVel` | cvel, cdof_dot |
| 18 | `mj_passive` (incl. `mjcb_passive`, passive plugins) | qfrc_spring, qfrc_damper, qfrc_gravcomp, qfrc_fluid, qfrc_adhesion, qfrc_passive |
| 19 | `mj_referenceConstraint` (non-metric) / `mj_velocityConstraint` (metric) | efc_vel, efc_aref (metric: only efc_vel here) |
| 20 | `mj_rne(m,d,0,qfrc_bias)` + `mj_tendonBias` | qfrc_bias |
| 21 | `mj_effShift` | efm velocity-stage values (efm_c, …) |
| — | `mj_sensorVel`, `mj_energyVel` | sensordata, energy[1] |
| **ACC** | `mjcb_control` (if set and ACTUATION not disabled) | user writes ctrl etc. |
| 22 | `mj_fwdActuation` | act_dot, actuator_force, qfrc_actuator |
| 23 | `mj_effActuation(m,d,1)` | efm_ca, actuator metric terms |
| 24 | metric only: `mj_regularizeConstraint(m,d,1)`, `mj_referenceConstraint` | efc_R/D/aref (and AR) |
| 25 | `mj_fwdAcceleration` | qfrc_smooth, qacc_smooth |
| 26 | `fwdConstraintStage` → `mj_fwdConstraint` | efc_b, efc_force, efc_state, qfrc_constraint, qacc, solver_niter, solver stats; iacc/ifrc/iefc_* |
| — | `flg_rnepost=0`; `mj_sensorAcc` | sensordata; lazily cacc, cfrc_int, cfrc_ext |
| 27 | `mj_checkAcc` | |
| 28 | `mj_compareFwdInv` (if FWDINV) | solver_fwdinv |
| **INT** | integrator | qH/qHDiagInv (Euler with damping, implicitfast), qDeriv + qLU (implicit), qpos, qvel, act, history, plastic, time, qacc_warmstart, plugin_state |

Lazily computed (flags reset where indicated): `mj_subtreeVel` → subtree_linvel, subtree_angmom (`flg_subtreevel`). `mj_rnePostConstraint` → cacc, cfrc_int, cfrc_ext (`flg_rnepost`). `mj_energyPos/Vel` (`flg_energypos/vel`).

---

## 5. Top-level API

### 5.1 `mj_step(m,d)` (engine_forward.c)

```
TM_START
mj_checkPos(m,d); mj_checkVel(m,d)
mj_forward(m,d)                 // = mj_forwardSkip(m,d,mjSTAGE_NONE,0)
mj_checkAcc(m,d)
if ENABLED(FWDINV): mj_compareFwdInv(m,d)
switch integrator:
  EULER:               mj_Euler
  RK4:                 mj_RungeKutta(m,d,4)
  IMPLICIT, IMPLICITFAST: mj_implicit      // = mj_implicitSkip(m,d,0)
  DISCRETE:            ENABLED(IPC) ? mj_ipc : mj_discrete
  default:             mjERROR("invalid integrator")
TM_END(mjTIMER_STEP)
```

### 5.2 `mj_forwardSkip(m,d,skipstage,skipsensor)`

```
mj_checkDiscrete(m)                        // option validation (§9.3)
if skipstage < POS:
  mj_fwdPosition; if !skipsensor: mj_sensorPos
  if !flg_energypos: ENERGY ? mj_energyPos : energy[0]=energy[1]=0
if skipstage < VEL:
  mj_fwdVelocity; if !skipsensor: mj_sensorVel
  if ENERGY && !flg_energyvel: mj_energyVel
if mjcb_control && !DISABLED(ACTUATION): mjcb_control(m,d)
mj_fwdActuation
mj_effActuation(m,d,1)
if mj_isMetric(m): mj_regularizeConstraint(m,d,1); mj_referenceConstraint(m,d)
mj_fwdAcceleration
fwdConstraintStage:  if ipcSkipConstraint(m) { qacc = qacc_smooth; solver_niter[0..19]=0 }
                     else mj_fwdConstraint(m,d)
if !skipsensor: flg_rnepost=0; mj_sensorAcc
TM_END(mjTIMER_FORWARD)
```

`ipcSkipConstraint(m)` is true iff IPC is enabled and some flex has `flex_dim==2`. In that case `mj_ipc` performs the constraint solves itself.

Energy is recomputed even when `skipsensor=1`, because `flg_energypos` is cleared at the start of `mj_fwdPosition`. The comment in RK4 ("do not recompute sensors and energy") applies only to sensors.

### 5.3 `mj_step1` / `mj_step2`

`mj_step1`: `mj_checkDiscrete; mj_checkPos; mj_checkVel; mj_fwdPosition; mj_sensorPos; energyPos-or-zero; mj_fwdVelocity; mj_sensorVel; energyVel; if mjcb_control: mjcb_control`. The control callback here is called even when ACTUATION is disabled. `mj_step1` times itself under `mjTIMER_STEP`.

Between step1 and step2 the user may modify ctrl, qfrc_applied and xfrc_applied.

`mj_step2`: `mj_fwdActuation; mj_effActuation; [metric: regularize + reference]; mj_fwdAcceleration; fwdConstraintStage; flg_rnepost=0; mj_sensorAcc; mj_checkAcc; [FWDINV compare]`. It then integrates: DISCRETE → `mj_ipc`/`mj_discrete`; IMPLICIT/IMPLICITFAST → `mj_implicit`; anything else (including RK4) → `mj_Euler`. Finally `timer[STEP].number--` so that one step1+step2 pair counts as one step.

---

## 6. Stage functions

### 6.1 `mj_fwdKinematics` and `mj_fwdPosition`

`mj_fwdKinematics = mj_kinematics; mj_comPos; mj_camlight; mj_flex; mj_tendon; if mj_wakeTendon: mj_updateSleep`.

`mj_fwdPosition` (timers in brackets):

```
TM_START1; flg_energypos = 0
TM_START; mj_fwdKinematics; TM_END(POS_KINEMATICS)
mj_makeM; mj_factorM                          [POS_INERTIA internal]
mj_collision                                  [POS_COLLISION internal]
if mj_wakeCollision: mj_updateSleep; mj_collision      // second full pass
if mj_wakeEquality:  mj_updateSleep
TM_RESTART; mj_makeConstraint; mj_island; TM_END(POS_MAKE)
TM_RESTART; mj_projectConstraint; TM_END(POS_PROJECT)
TM_RESTART; mj_transmission; TM_ADD(POS_KINEMATICS)    // adds duration, no count++
mj_effBuild(m,d, mj_isMetric(m), 1)
TM_END1(POSITION)
```

### 6.2 `mj_fwdVelocity`

```
flg_subtreevel = flg_energyvel = 0
flexedge_velocity = 0 (nflexedge)
for f: if flex_rigid[f] || flex_interp[f]: continue
  flexedge_velocity[adr:adr+num] = flexedge_J[rows] · qvel     (mju_mulMatVecSparse, CSR rows of edge f)
ten_velocity = ten_J · qvel                      (CSR: ten_J_rownnz/rowadr/colind)
actuator_velocity = DISABLED(ACTUATION) ? 0 : actuator_moment · qvel   (CSR: moment_rownnz/rowadr/colind, nout rows)
mj_comVel; mj_passive
mj_isMetric ? mj_velocityConstraint : mj_referenceConstraint
mj_rne(m, d, 0, qfrc_bias)        // flg_acc = 0: bias = C(q,v) incl. gravity
mj_tendonBias(m, d, qfrc_bias)    // tendon armature bias
mj_effShift
```

### 6.3 `mj_fwdActuation`: (qpos, qvel, ctrl, act) → (act_dot, actuator_force, qfrc_actuator)

Output space: `nout` force outputs (one per scalar transmission; 3 for SO3). `actuator_outadr[i]` and `actuator_outnum[i]` locate the outputs of actuator i. `actuator_ctrladr[i]` and `actuator_ctrlnum[i]` locate its inputs in `ctrl` (nu). `actuator_actadr[i]` and `actuator_actnum[i]` locate its states in `act` (na; `actadr=-1` means stateless).

**Step 0. Early out.** `actuator_force[0:nout] = 0`. If `nactuator==0 || DISABLED(ACTUATION)`, set `qfrc_actuator = 0` and return. `act_dot` is **not** touched.

**Step 1. Local ctrl copy** (stack, length nu).
1. `ctrl = d->ctrl`.
2. For each actuator with `actuator_delay[i] != 0`: `ptr = mj_readCtrl(m,d,i,d->time,ctrl+adr,-1)`. If it returns a pointer (exact match, ZOH, or extrapolation), copy `ctrlnum` values from it. If it returns NULL, the interpolated values were already written into `ctrl+adr`.
3. If CLAMPCTRL is not disabled: `ctrl[j] = clip(ctrl[j], ctrlrange[2j], ctrlrange[2j+1])` for `ctrllimited[j]` (indexed per control j in [0,nu)).
4. If any `mju_isBad(ctrl[j])`: `mj_warning(BADCTRL, j)`, zero **all** of ctrl, and stop checking.

`mj_readCtrl(m,d,id,time,result,interp)`: if `actuator_history[2id]==0`, it returns `d->ctrl+ctrladr`. Otherwise `interp<0` selects `actuator_history[2id+1]`, and it returns `mju_historyRead(buf, nsample, ctrlnum, result, time - actuator_delay[id], interp)` with `buf = d->history + actuator_historyadr[id]`.

**Step 2. act_dot** for each actuator, skipping actuators that are asleep (sleep enabled) or have `actadr<0`. If `actnum>0`, zero `act_dot[actadr:actadr+actnum]`. Let `act_last = actadr+actnum-1`, `uadr = ctrladr`, `oadr = outadr`, `dynprm = actuator_dynprm+10i`.

- `INTEGRATOR`: for j in [0,ctrlnum): `act_dot[act_last-ctrlnum+1+j] = ctrl[uadr+j]`.
- `FILTER`, `FILTEREXACT`: `tau = max(mjMINVAL, dynprm[0])`; `act_dot[act_last] = (ctrl[uadr] - act[act_last]) / tau`.
- `MUSCLE`: `act_dot[act_last] = mju_muscleDynamics(ctrl[uadr], act[act_last], dynprm)`:
  ```
  c = clip(ctrl,0,1); a = clip(act,0,1)
  tau_act   = prm[0]*(0.5+1.5a);  tau_deact = prm[1]/(0.5+1.5a)
  dctrl = c - act        // note: unclamped act
  tau = (prm[2] < mjMINVAL) ? (dctrl>0 ? tau_act : tau_deact)
                            : tau_deact + (tau_act-tau_deact)*mju_sigmoid(dctrl/prm[2] + 0.5)
  return dctrl / max(mjMINVAL, tau)
  ```
- `PID`: state slots in order [slew (if `dynprm[1]>0`), integral (if `gainprm[0]>0`)]. `period = wrapPeriod(m,i)`.
  - Slew: `ctrl[uadr] = slewLimit(ctrl[uadr], act[adr], dynprm[1], h, period, &act_dot[adr]); adr++`. `slewLimit` first wraps u to the representative nearest `u_prev` if period>0, then computes `slew = slew_s*h`, `u_eff = clip(u, u_prev-slew, u_prev+slew)`, `act_dot = (u_eff-u_prev)/h`, and returns `u_eff`. The local ctrl copy is overwritten.
  - Integral: `err = ctrl[uadr] - actuator_length[oadr]`. If period>0, `err -= period*round(err/period)`. Anti-windup with `imax = dynprm[0] > 0`: if `act[adr] >= imax` then `err = min(err,0)`; else if `act[adr] <= -imax` then `err = max(err,0)`. Set `act_dot[adr] = err`.
- `DCMOTOR`: `slots = mj_dcmotorSlots(dynprm, gainprm)`. Slot order is slew (`dynprm[7]>0`), integral (`gainprm[5]>0`), temperature (`dynprm[2]>0`), bristle (`dynprm[5]>0`), current (`dynprm[0]>0`). It is an error if `num_slots != actnum`. With `velocity = actuator_velocity[oadr]`, `R = gainprm[0]`, `K = gainprm[1]`, `ki = gainprm[5]`, `te = dynprm[0]`:
  1. Slew (`dynprm[7]>0`): `ctrl[uadr] = slewLimit(ctrl[uadr], act[adr], dynprm[7], h, 0, &act_dot[adr]); adr++`.
  2. Integral (`ki>0`): `x_I = act[adr]`, `ad = ctrl[uadr] - actuator_length[oadr]`, anti-windup with `Imax = dynprm[8]` as in PID, `act_dot[adr] = ad`, `adr++`.
  3. `V = dcmotorVoltage(ctrl+uadr, ctrlspec, length, velocity, x_I, gainprm)`:
     ```
     unpack u4 = [pos, vel, ff, voltage] in canonical order from present inputs (absent = 0)
     voltage = 0
     if spec & (POS|VEL|FF):
       torque = kp*(u4[0]-length) + kd*(u4[1]-velocity) + ki*x_I + u4[2]   // kp=g[4], ki=g[5], kd=g[6]
       voltage = R/K*torque + K*velocity
       if Vmax=g[7] > 0: voltage = clip(voltage, -Vmax, Vmax)
     return voltage + u4[3]      // raw voltage is added after the clamp, unclamped
     ```
  4. Temperature (`RT=dynprm[2]>0`): `C=dynprm[3]`, `Ta=dynprm[4]`, `alpha=gainprm[2]`, `T0=gainprm[3]`, `T=act[adr]`. Then `R *= 1+alpha*(T+Ta-T0)`, `current = te>0 ? act[act_last] : (V-K*velocity)/R`, `act_dot[adr] = (R*current² - T/RT)/C`, `adr++`.
  5. Bristle (`sigma0=dynprm[5]>0`): `g = F_C + (F_S-F_C)*exp(-(v/max(mjMINVAL,v_S))²)` with `F_C,F_S,v_S = biasprm[3..5]`. Then `a = -sigma0*|v|/max(mjMINVAL,g)`, `act_dot[adr] = a*z + v`, `adr++`.
  6. Current (`te>0`): `i_dot = (V/R - K/R*velocity - act[act_last])/te`, clipped to `±dimax` if `dimax=dynprm[1]>0`. Set `act_dot[act_last] = i_dot`. R here is the temperature-adjusted value if the thermal model is on.
- default (USER): if `mjcb_act_dyn`: when `actnum==1`, `act_dot[act_last] = mjcb_act_dyn(m,d,i)`; otherwise call it and it writes act_dot directly.

Then for each plugin instance with the `mjPLUGIN_ACTUATOR` capability: `plugin->actuator_act_dot(m,d,i)` if non-null.

**Step 3. force = gain·input + bias** for each actuator. Skip it if asleep, if `mj_actuatorDisabled`, or if `actuator_plugin[i] >= 0`.

- **SO3 gain** (3 outputs). The target quaternion is computed first. For `ctrlspec==QUAT`: `q_tgt = normalize4(ctrl[uadr:uadr+4])`, where a zero quaternion becomes the identity. Otherwise (expmap) `u` comes from ctrl (dyntype NONE), from `mj_nextActivation` of each of the 3 act slots (actearly), or from act. Then `q_tgt = expmap2Quat(u)`: identity if `|u|<mjMINVAL`, else axis-angle. `q_cur = expmap2Quat(actuator_length[oadr:+3])`, `e = mji_subQuat(q_tgt, q_cur)` (log of `q_cur⁻¹ q_tgt`, local frame). `force[oadr+k] = kp*e[k] + biasprm[0] + biasprm[2]*actuator_velocity[oadr+k]` with `kp=gainprm[0]`. Then `continue` (no SISO path).
- If ntendon>0, `tendon_frclimited` is not yet set, and trntype==TENDON: `tendon_frclimited = tendon_actfrclimited[trnid]`. This means the clamp pass runs if the first tendon actuator encountered targets a limited tendon. See the pitfall in §15.
- Gain: FIXED `g0`. PID `0`. AFFINE `g0 + g1*len + g2*vel`. MUSCLE `mju_muscleGain(len, vel, lengthrange, acc0, gainprm)`. DCMOTOR: `R = mj_dcmotorResistance` (temperature-adjusted, `max(mjMINVAL,R)`), `gain = dynprm[0]>0 ? K : K/R`, and if `ctrlspec != VOLTAGE && ctrlnum>0` the local `ctrl[uadr]` is overwritten by `dcmotorVoltage(...)` using `x_I = act[actadr+slots.integral]` (or 0). USER: `mjcb_act_gain` or 1.
- Input:
  - PID: `u4 = unpack(ctrl)`, `qref=u4[0]`, `vref=u4[1]`, `ff=u4[2]`. If period>0, `qref = wrapSetpoint(qref, len, period)`. Then `force = -biasprm[1]*qref - biasprm[2]*vref + ff`. If `actnum && gainprm[0]>0`: `z` = act of the last slot (or `mj_nextActivation` if actearly), and `force += gainprm[0]*z`. Here kp=−biasprm[1] and kv=−biasprm[2] are single-sourced in the affine bias.
  - Stateless or dcmotor-without-current: `input = ctrlnum ? ctrl[uadr] : 0`. Wrap to the length if period>0. `force = gain*input`.
  - Otherwise: `act = actearly ? mj_nextActivation(m,d,i,act_last,act_dot[act_last]) : act[act_last]`. Wrap if period>0. `force = gain*act`.
- Bias: NONE 0. AFFINE `b0 + b1*len + b2*vel`. MUSCLE `mju_muscleBias(len, lengthrange, acc0, biasprm)`. DCMOTOR: if `te<=0` then `bias = -gain*K*vel` (back-EMF), else 0. USER: `mjcb_act_bias` or 0. Then `force[oadr] += bias`.

`wrapPeriod(m,i)`: it is a servo if gaintype FIXED, biastype AFFINE, `gainprm[0] == -biasprm[1]` (exact equality), and dyntype ∈ {NONE, INTEGRATOR}; it is PID if gaintype PID. If neither, return 0. A SITE transmission with refsite (`trnid[1]>=0`) and zero linear gear returns `2π·‖gear[3:6]‖`. A JOINT/JOINTINPARENT transmission on a BALL joint returns `2π·‖gear[0:3]‖`. Anything else returns 0. `wrapSetpoint(u,len,P) = u - P*round((u-len)/P)`.

**Step 4. Actuator plugins.** For each plugin with ACTUATOR capability: `plugin->compute(m,d,i,mjPLUGIN_ACTUATOR)`. It is an error if compute is NULL.

**Step 5. Tendon total-force limit** (only if `tendon_frclimited`):
```
total[t] = Σ_{i: trntype==TENDON, t=trnid[2i], actfrclimited[t]} force[outadr[i]]
for each tendon actuator i (t=trnid): if actfrclimited[t] && total[t] != 0:
   if total < range[0]: force *= range[0]/total ; elif total > range[1]: force *= range[1]/total
```

**Step 6. Per-actuator force clamp** (`actuator_forcelimited`). SO3: if `‖f‖ > range[1]`, scale f by `range[1]/‖f‖` (direction preserved). Otherwise clip each output to `[range0, range1]`.

**Step 7. DC-motor mechanical forces** (not subject to the current/force limits). For biastype DCMOTOR actuators that are not asleep, not disabled, and not plugins: cogging `A=b[0]≠0` adds `A*sin(b[1]*len + b[2])`. LuGre with `sigma0>0` subtracts `sigma0*z + sigma1*z_dot`, where `z=act[actadr+slots.bristle]`, `z_dot=act_dot[same]`, `sigma1=dynprm[6]`.

**Step 8.** `qfrc_actuator = momentᵀ · force` (`mju_mulMatTVecSparse`, nout rows). Gravity compensation: if `flg_gravcomp && !DISABLED(GRAVITY) && ‖gravity‖≠0`, then for every joint with `jnt_actgravcomp`, add `qfrc_gravcomp[dofadr : dofadr+{6,3,1,1}[type]]`. Finally clamp `qfrc_actuator[jnt_dofadr[j]]` to `jnt_actfrcrange` for `jnt_actfrclimited[j]`. Only the **first dof** of each joint is clamped, because `clampVec` indexes `index[i]=jnt_dofadr[i]`.

Muscle gain and bias (`engine_util_misc.c`), `prm = (range0, range1, force, scale, lmin, lmax, vmax, fpmax, fvmax)`:
```
force<0 ⇒ force = scale/max(mjMINVAL, acc0)
L0 = (lr1-lr0)/max(mjMINVAL, range1-range0);  L = range0 + (len-lr0)/max(mjMINVAL,L0);  V = vel/max(mjMINVAL, L0*vmax)
FL(L): 0 outside [lmin,lmax]; a=0.5(lmin+1), b=0.5(1+lmax);
   L<=a: 0.5x², x=(L-lmin)/max(ε,a-lmin);  L<=1: 1-0.5x², x=(1-L)/max(ε,1-a)
   L<=b: 1-0.5x², x=(L-1)/max(ε,b-1);      else: 0.5x², x=(lmax-L)/max(ε,lmax-b)
FV: V<=-1:0; V<=0:(V+1)²; V<=y:fvmax-(y-V)²/max(ε,y); else fvmax  (y=fvmax-1)
gain = -force*FL*FV
bias: b=0.5(1+lmax); L<=1:0; L<=b: -force*fpmax*0.5*((L-1)/max(ε,b-1))²; else -force*fpmax*(0.5+(L-b)/max(ε,b-1))
```

### 6.4 `mj_fwdAcceleration`

Let `sleep_filter = SLEEP && nv_awake < nv`. Compute `qfrc_smooth = qfrc_passive - qfrc_bias + qfrc_applied + qfrc_actuator` in exactly this order: `sub(passive,bias)`, then `+=applied`, then `+=actuator`. With sleep_filter, only awake dofs (`dof_awake_ind`) are written and asleep entries keep stale values. Then `mj_xfrcAccumulate(m,d,qfrc_smooth)`, which returns early if `xfrc_applied[6:]` is all bytes zero; otherwise, for each body i≥1 with a nonzero wrench, `mj_applyFT(force=xfrc[6i:6i+3], torque=xfrc[6i+3:6i+6], point=xipos[i], body i)`.

If `d->efm_active` (discrete): `qfrc_eff = qfrc_smooth + efm_c (+ efm_ca)` and `qacc_smooth = mj_effSolve(qfrc_eff)`, i.e. `(M+K_eff) qacc_smooth = qfrc_smooth + c`. Under sleep, the asleep entries of the rhs are zero and only awake results are scattered. Return.

Otherwise `qacc_smooth = qfrc_smooth`, followed by in-place `mj_solveLD(qacc_smooth, qLD, qLDiagInv, nv, 1, M_rownnz, M_rowadr, M_colind, index)`.

`mj_solveLD` (in-place, sparse L'DL with the convention that row k stores L(k,ancestors) followed by the diagonal):
```
for k = nv-1 .. 0 (i=index? index[k] : k): if rownnz[i]==1: continue
     if x_i = x[i] != 0: for adr in [rowadr[i], rowadr[i]+rownnz[i]-1): x[colind[adr]] -= qLD[adr]*x_i
for k = 0..nv-1: x[i] *= qLDiagInv[i]
for k = 0..nv-1 (x <- L^-1 x): if rownnz[i]==1: continue
     x[i] -= mju_dotSparse(qLD+rowadr[i], x, rownnz[i]-1, colind+rowadr[i])
```
The early skip of `x_i==0` matters for NaN propagation (a 0 never multiplies an Inf factor). `mju_dotSparse` may use an AVX path (see the BLAS file), which affects bitwise summation order.

`mj_factorI(mat, diaginv, nv, rownnz, rowadr, colind, index)`: backward over rows k. Pivot `mat[diag]`: if `< mjMINVAL`, set it to `mjMINVAL` and record the first clamped k (returned; the caller issues `mjWARN_INERTIA`). Then `invD = 1/pivot`, `diaginv[k]=invD`. For each off-diagonal adr (descending), with `i=colind[adr]`: `row_i[0:rownnz[i]] -= mat[adr]*invD * row_k[0:rownnz[i]]`. Finally scale row k's off-diagonals by invD.

### 6.5 `mj_fwdConstraint` / `fwdConstraint(m,d,solver,flg_island)`

`mj_fwdConstraint` errors on an unknown solver (only when `nefc>0`), then calls `fwdConstraint(m,d,opt.solver, !DISABLED(ISLAND))`. `mj_fwdConstraintCG` pins CG monolithic (used by IPC).

```
qfrc_constraint = 0
if nefc == 0: qacc = qacc_smooth; solver_niter[0..19] = 0; mj_discreteGyro; return
efc_b = J·qacc_smooth − efc_aref              (mj_mulJacVec)
warmstart(m,d)
solver_niter[0..19] = 0
islands_supported = flg_island && nisland>0 && !IPC && !mj_effFoldMonolithic(m,d,solver)
if islands_supported:
   PGS: mju_dispatch(solveIslandTask, nisland)
   CG/NEWTON: gather ifrc_smooth, ifrc_constraint, iacc_smooth, iacc (map_idof2dof, nidof) and
              iefc_force, iefc_aref (map_iefc2efc, nefc); dispatch; scatter qacc, qfrc_constraint
              (map_idof2dof); efc_force = gather(iefc_force, map_efc2iefc)
   if noslip_iterations>0: for each island: mj_solNoSlip_island
else: PGS → mj_solPGS, CG → mj_solCG, NEWTON → mj_solNewton (all with opt.iterations)
      if noslip_iterations>0: mj_solNoSlip
if solver==PGS || noslip_iterations>0: mj_dualFinish      // efc_force → qfrc_constraint, qacc (monolithic)
mj_discreteGyro
```

`solveIslandTask` calls `mj_sol{Newton,CG,PGS}_island(m,d,island,iterations)`. Per-island tasks write disjoint memory, so results do not depend on thread count.

`mj_discreteGyro` (discrete with `efm_active` only): for each joint where `mjd_freeGyroPossible` holds (free joint, standalone free body, no efm_K row, no flex coupling, no constraint Jacobian support on its 6 dofs) and `mjd_freeMhat(...,flg_discrete=1)` succeeds: `rhs = qfrc_smooth + efm_c (+efm_ca)` on the 6 dofs. If `mju_factorLU6(A)` succeeds, overwrite `qacc[adr:adr+6] = A⁻¹ rhs`.

---

## 7. Warmstart (`engine_forward.c:warmstart`)

```
if !DISABLED(WARMSTART):
  qacc = qacc_warmstart
  jar  = J·qacc_warmstart − efc_aref
  mj_constraintUpdate(m,d,jar,&cost_ws,0)          // sets efc_force/state, qfrc_constraint for qacc_warmstart
  if solver == PGS:
     PGS_ws = efc_force·efc_b + 0.5·efc_force·(AR·efc_force)    (AR sparse if mj_isSparse else dense nefc×nefc)
     if PGS_ws > 0: efc_force = 0; qfrc_constraint = 0           // zero force is better
  else:
     da = qacc_warmstart − qacc_smooth; Mda = M·da (+ effMulAdd(da, contact=1) if efm_active)
     cost_ws += 0.5·da·Mda
     mj_constraintUpdate(m,d,efc_b,&cost_smooth,0)   // efc_b = jar at qacc_smooth
     if cost_ws > cost_smooth: qacc = qacc_smooth     // strict >: ties keep the warmstart
  if nisland > 0: for i in [nidof, nv): qacc[map_idof2dof[i]] = qacc_smooth[...]   // unconstrained dofs
else:
  qacc = qacc_smooth; efc_force = 0
```

Notes. In the non-PGS branch, efc_force and state end up evaluated at `qacc_smooth` (the last `mj_constraintUpdate` call), even when `qacc` is reset to the warmstart. The primal solvers recompute them at the start of their iterations. `qacc_warmstart` is written in `mj_advance` as `qacc_warmstart = qacc` after the step, and is zeroed by reset.

---

## 8. State advancement

### 8.1 `advanceStart(m,d,act_dot)`: work at the pre-step state

1. **History buffers** (if `nhistory>0`):
   - ctrl: for each actuator with `nsample=actuator_history[2i] > 0`: `slot = mju_historyInsert(buf, nsample, ctrlnum, d->time)`; copy `d->ctrl[ctrladr : +ctrlnum]` (the raw, unclamped, undelayed ctrl).
   - sensors, with `delay=sensor_delay[i]`, `interval=sensor_interval[2i]`:
     - interval>0: if `buf[0] + interval <= time`, do `buf[0] += interval` (exact continuous time), then insert, filling the slot either with `mj_computeSensor` (if delay>0) or by copying sensordata.
     - delay>0 only: insert and `mj_computeSensor`.
     - otherwise (history only): insert and copy sensordata.
2. **Activations** (if `na>0` and ACTUATION is not disabled): for every actuator and every slot j: `act[j] = mj_nextActivation(m,d,i,j, disabled? 0 : act_dot[j])`. Then the re-anchoring pass over actuators with dyntype INTEGRATOR:
   - `period = wrapPeriod > 0`: `act[last] = wrapSetpoint(act[last], actuator_length[outadr], period)`.
   - gaintype SO3: `angle=‖act[adr:adr+3]‖`. If `angle > π`, scale the vector by `(angle − 2π·round(angle/2π))/angle`.
3. **Sleep**: `if mj_sleep(m,d) { mj_forwardSkip(m,d,mjSTAGE_POS,0); mj_updateSleep(m,d); }`. Sleeping zeroes qvel of the slept trees, and all velocity-dependent quantities are then recomputed, including the solve.

`mj_nextActivation(m,d,i,j,act_dot)` (`engine_support.c`), with `act = d->act[j]`:
- FILTEREXACT: `tau = max(mjMINVAL, dynprm[0])`, `act += act_dot*tau*(1-exp(-h/tau))`.
- DCMOTOR with `offset = j - actadr`:
  - current slot: `te=max(mjMINVAL,dynprm[0])`, `act += act_dot*te*(1-exp(-h/te))`.
  - bristle: `a=-sigma0|v|/max(mjMINVAL,g(v))`, `act = exp(a h)*act + int_h*v` with `int_h = |a|>mjMINVAL ? (exp(ah)-1)/a : h` (exact zero-order hold).
  - integral: `act += act_dot*h`, then clip to `±Imax` if `Imax=dynprm[8]>0`.
  - temperature/slew: `act += act_dot*h`.
- otherwise: `act += act_dot*h` (Euler).
- Clamp to `actrange` if `actlimited` and dyntype ≠ DCMOTOR.

History ring buffer layout (`buf`): `[user_slot, cursor, times[n], values[n*dim]]`. Logical index L (0=oldest, n−1=newest) maps to physical index `(cursor+1+L) % n`. `historyFindIndex`: return 0 if `t <= t_oldest`, n if `t > t_newest`, otherwise binary search (`mid=(lo+hi)/2`) for the smallest logical i with `times[i] >= t`. `mju_historyInsert(buf,n,dim,t)`:
- exact match (`|t-times[i]| < mjMINVAL`): return the existing slot.
- `i==0`: overwrite the oldest slot's time.
- `i==n`: `cursor=(cursor+1)%n`, write at the new cursor.
- otherwise shift logical [1,i−1] left (dropping the oldest) and insert at i−1.

`mju_historyRead(buf,n,dim,res,t,interp)`:
- `t <= t_oldest+mjMINVAL`: return a pointer to the oldest value.
- `t >= t_newest−mjMINVAL`: return a pointer to the newest value.
- exact match: return a pointer.
- `interp==0` (ZOH): return a pointer to lo.
- `interp==1`: linear, written to res, return NULL.
- `interp==2`: cubic Hermite with Catmull-Rom slopes `m_lo=(v_hi−v_{i−2})/(t_hi−t_{i−2})` (0 if i≤1) and `m_hi=(v_{i+1}−v_lo)/(t_{i+1}−t_lo)` (0 if i≥n−1), basis `h00=2α³−3α²+1`, `h10=α³−2α²+α`, `h01=−2α³+3α²`, `h11=α³−α²`, `res = h00 v_lo + h10 dt m_lo + h01 v_hi + h11 dt m_hi`.

### 8.2 `advanceFinish(m,d,qvel)`: after the new state is written

1. `time += timestep`.
2. Plugin `advance(m,d,i)` for each plugin instance that has it. It is an error if a plugin slot is invalid.
3. **Plastic return mapping** (if `nplastic>0` and SPRING is not disabled). Elements that are not awake are skipped under sleep. `yield = max(0, springyield)`.
   - SLIDE/HINGE joints: `x = qpos − qspring`. If `x > yield`, `qspring = qpos − yield`. If `x < −yield`, `qspring = qpos + yield`.
   - BALL: `quat = normalize4(qpos)`, `dif = subQuat(quat, qspring)`, `r=‖dif‖`. If `r>yield`: `quatIntegrate(qspring, dif, (r−yield)/r)`, then normalize4.
   - Tendons: `L_next` is `Σ wrap_prm·qpos[jnt]` for fixed (first wrap is JOINT), else `ten_length + h·(ten_J·qvel)`, using the **qvel argument**. `x` is the distance outside `[lower, upper] = plastic[0:2]`. If `x>yield`: `delta=x−yield`; `lower += delta` only if `lower≠0 || upper<=0`; always `upper += delta`. If `x<−yield`: `delta=x+yield`, shift both.
   - Flexes: interpolated (solid/shell cell corotational strain return mapping, with a Frobenius cap `‖sp‖² > 0.1225 → scale 0.35/‖sp‖`), dim≥2 element stretch J2 return mapping with the 2D clamp `splastic ≥ −0.99 g`, a 3D volume-preserving depressed-cubic solve, and 2D shell bending in dihedral angle space with `theta_p` clipped to `±1.3`. They set `flex_hessian_valid[f]=0` when plasticity changes. Predicted positions are `xpos + h·(J·qvel)`. The exact formulas are in `engine_forward.c:advanceFinish` lines 1546–1942. The flex elasticity file is responsible for the underlying metrics (`mj_stretchElasticity`, `mj_stretchTension`). `gramDet3D(grad,q)` is reproduced in §15 notes for reference.

### 8.3 `mj_advance(m,d,act_dot,qacc,qvel)`

```
advanceStart(m,d,act_dot)
sleep_filter = SLEEP && ntree_awake < ntree
qvel_d += h·qacc            (awake dofs only if sleep_filter; mju_addToScl ordering: qvel[i] += h*qacc[i])
v = qvel ? qvel : d->qvel   (semi-implicit: the NEW velocity)
mj_integratePosInd(m, qpos, v, h, sleep_filter ? body_awake_ind : NULL, sleep_filter ? nbody_awake : nbody)
advanceFinish(m,d,v)
qacc_warmstart = qacc_d     (d->qacc, not the qacc argument)
```

`mj_integratePosInd`: for b = 1..nbody−1 (body k = index[b] or b), for each of its joints:
- FREE: `qpos[0:3] += h·v[0:3]`, then the quaternion update on `qpos[3:7]` with `v[3:6]`.
- BALL: `mju_quatIntegrate(qpos+padr, v+vadr, h)`.
- SLIDE/HINGE: `qpos += h·v`.

`mju_quatIntegrate(quat, vel, scale)`:
```
tmp = vel; angle = scale * mju_normalize3(tmp)   // normalize3: if norm<mjMINVAL → tmp=(1,0,0), returns norm
qrot = axisAngle2Quat(tmp, angle)               // angle==0 → identity; else (cos(a/2), axis*sin(a/2))
mju_normalize4(quat)                            // norm<mjMINVAL → (1,0,0,0); renormalize only if |norm−1| > mjMINVAL
quat = quat ⊗ qrot                              // body-frame (right) multiplication; NOT renormalized afterwards
```

### 8.4 `mj_commit(m,d,act_dot,qpos,qvel,qacc)` (position-level integrators, IPC)

`qacc_d = qacc; advanceStart; qpos_d = qpos; qvel_d = qvel; advanceFinish(m,d,qvel_d)`. The warmstart is left to the caller.

---

## 9. Integrators

### 9.1 Euler: `mj_EulerSkip(m,d,skipfactor)`

```
dof_damping = 0
if !DISABLED(EULERDAMP) && !DISABLED(DAMPER):
  for awake dofs i: if dof_damping[i]>0 || dampingpoly[i] nonzero || jnt_actuatorid[dof_jntid[i]] != -1: dof_damping=1
if !dof_damping: qacc_loc = qacc                       // plain semi-implicit
else:
  if !skipfactor:
     qH = M (awake rows only under sleep)
     for awake i: poly = dof_dampingpoly[i]; damping = dof_damping[i] + mj_actuatorDamping(JOINT, dof_jntid[i], poly)
                  qH[diag_i] += h · mjd_xPolyForce(damping, poly, qvel[i], 2, flg_odd=1)
     clamped = mj_factorI(qH, qHDiagInv, ...); if clamped>=0: mj_warning(INERTIA, clamped)
  qacc_loc = qfrc_smooth + qfrc_constraint; mj_solveLD(qacc_loc, qH, qHDiagInv, ...)
mj_advance(m,d, act_dot, qacc_loc, NULL)
```

- `mjd_xPolyForce(lin, poly, x, n, odd) = lin + Σ_{i<n} (i+2)·poly[i]·|x|^{i+1}` (x replaced by |x| when odd).
- `mj_actuatorDamping(type,id,poly)`: `aid = jnt_actuatorid[id]`. If −1, return 0. If ≥0, `gear2 = gear[6·outadr]²`, `damping = actuator_damping·gear2`, and `poly += actuator_dampingpoly·gear2`. If < −1, scan all actuators targeting id (see engine_core_util.c).
- The Euler damping update is `(M + h·B(v)) a = qfrc_smooth + qfrc_constraint`, where B is diagonal.

`mj_Euler = mj_EulerSkip(m,d,0)`. Under sleep, M is copied with `mju_copySparse` for awake rows only.

### 9.2 Implicit and implicitfast: `mj_implicitSkip(m,d,skipfactor)`

```
qfrc = qfrc_smooth + qfrc_constraint   (awake dofs)
if !skipfactor:
  IMPLICIT:      mjd_smooth_vel(m,d, flg_bias=1)        // qDeriv (D-structure, nD) incl. RNE derivative
                 qLU = gatherMasked(M, mapM2D, nD)      // symmetric fill from lower-triangle CSR
                 qLU += −h·qDeriv
                 clamped = mju_factorLUSparse(qLU, nv, scratch, D_rownnz, D_rowadr, D_colind, index)
  IMPLICITFAST:  mjd_smooth_vel(m,d, flg_bias=0)        // no RNE derivative; symmetrized (see derivative file)
                 qH = gather(qDeriv, mapD2M, nC)        // full → lower
                 qH = M − h·qH
                 for each FREE joint whose body is a standalone free body (mj_isFreeBody):
                     reset its 6 rows of qH to M (keeps L'DL well defined)
                 clamped = mj_factorI(qH, qHDiagInv, ...)
  if clamped>=0: mj_warning(INERTIA, clamped)
IMPLICIT:     qacc_loc = mju_solveLUSparse(qLU, qfrc, D_rownnz, D_rowadr, D_diag, D_colind, index)
IMPLICITFAST: qacc_loc = qfrc; mj_solveLD(qacc_loc, qH, qHDiagInv, ...)
              for each joint j with mjd_freeMhat(m,d,j,h,A,flg_discrete=0):
                  if mju_factorLU6(A,pivot): qacc_loc[adr:adr+6] = solveLU6(A, qfrc[adr:adr+6])
mj_advance(m,d, act_dot, qacc_loc, NULL)
```

`mj_isFreeBody(body)`: `body_jntnum==1`, the joint is FREE, and `tree_dofnum[dof_treeid[adr]]==6` (fixed descendants only).

`mjd_freeMhat(m,d,jnt,h,A,flg_discrete)` returns 0 unless the joint's body is a free body whose tree is awake (`tree_awake`). Steps:
1. A = 6×6 block of M, gathered from lower CSR and mirrored.
2. If discrete: add `efm_diag` to the diagonal and `efm_fluid` (mirrored). Otherwise `A −= h·qDeriv` block (D-structure rows, `D_colind − adr`).
3. `A += h·B`, where B = `mjd_freeBias_vel` (−d(qfrc_bias)/dqvel; gyroscopic block).

The gyroscopic and fluid derivative formulas belong to the derivative file. Since qDeriv for implicitfast excludes the bias derivative, the free-body block reinstates it exactly, which makes implicitfast and implicit identical for standalone free bodies.

`mj_implicit = mj_implicitSkip(m,d,0)`. Any other integrator value raises "integrator must be implicit or implicitfast".

### 9.3 Discrete: `mj_discrete`, `mj_checkDiscrete`

`mj_discrete(m,d) = mj_advance(m,d, act_dot, d->qacc, NULL)`, timed as ADVANCE. The solver's qacc already is the step map `(v⁺−v)/h` in the metric `M̂ = M + hD + h²K`, solved jointly with the constraints (the effective-metric file specifies `mj_effBuild/Shift/Actuation/Solve`).

`mj_checkDiscrete(m)` (runs at every forward):
- IPC: requires DISCRETE ("flag ipc requires integrator='discrete'"), requires CG, and is incompatible with FWDINV and with SLEEP.
- Non-metric integrators error on: SNH elasticity (3D flex, non-interp, `flex_stiffness[adr+21] != 0`); passive flex contact (`mj_effFlexContactPossible`); and, for IMPLICIT/IMPLICITFAST, any flex with `mj_effFlexStiffPossible` ("flex elasticity is no longer integrated implicitly…").
- DISCRETE errors on:
  - non-rigid flex with `flex_edgestiffness>0 || flex_edgedamping>0`;
  - NEWTON with a non-interp stiff flex that is not `mj_flexSimple` ("general attachments requires solver='CG'");
  - NEWTON with `!mjd_flexInterpAssemblable`;
  - (PGS or noslip) with any `mj_effFlexAny`;
  - SLEEP with ISLAND disabled.

### 9.4 Runge-Kutta 4: `mj_RungeKutta(m,d,N)` (only N=4)

Butcher tableau (globals `RK4_A`, `RK4_B`, laid out as 3×3 row-major for stages 2..4):

```
A = [[0.5, 0,   0  ],     C = row sums = (0.5, 0.5, 1)
     [0,   0.5, 0  ],     B = (1/6, 1/3, 1/3, 1/6)
     [0,   0,   1  ]]     T[i-1] = time + C[i-1]·h
```

`mj_forward` has already been called by `mj_step`. State vectors: `X[i] = (qpos nq, qvel nv, act na)`, `F[i] = (qacc nv, act_dot na)`, `dX` of length `2nv+na`, all on the stack.

```
X[0] = (qpos, qvel, act); F[0] = (qacc, act_dot)        // rotation vector 0: no RKMK correction
save ten_length, ten_J if nplastic && ntendon;  flexvert_xpos, cdof, subtree_com if nplastic && nflexvert
for i = 1..3:
   dX = 0
   for j < i:  dX[0:nv]     += A[i-1][j] · X[j].qvel        // X[j].qvel is RKMK-corrected for j≥1
               dX[nv:2nv+na] += A[i-1][j] · F[j]
   X[i] = X[0];  mj_integratePos(m, X[i].qpos, dX[0:nv], h);  X[i].(qvel,act) += h·dX[nv:]
   d->qpos, d->qvel, d->act = X[i];  d->time = T[i-1]
   rkCorrectVel(m, X[i].qvel, dX[0:nv], h)    // in place; d->qvel keeps the uncorrected stage velocity
   mj_forwardSkip(m,d,mjSTAGE_NONE,1)         // skip sensors
   F[i] = (d->qacc, d->act_dot)
dX = Σ_{j=0..3} B[j]·(X[j].qvel, F[j])
restore time, qpos, qvel, act (and the saved plastic-related arrays)
mj_advance(m,d, act_dot = dX[2nv:], qacc = dX[nv:2nv], qvel = dX[0:nv])
```

**RKMK correction** `rkCorrectVel(m,vel,dqvel,h)`: for each BALL joint, and for the rotational dofs `vadr+3..5` of each FREE joint: `rot = h·dqvel[vadr:vadr+3]`, `vel[vadr:+3] = mju_derivExpmap(rot, vel[vadr:+3])`.

`mju_derivExpmap(r, v)` computes `dexp⁻¹_{−r}(v) = v + ½ r×v + c(|r|²) r×(r×v)`, where r is first reduced to angle (−π,π]:
```
xx = r·r; if xx > π²: x=√xx; r *= (x − 2π·floor(x/(2π)+0.5))/x; xx = r·r
c = bortzCoef(xx):
  if xx < 1.75² (=3.0625): Horner minimax polynomial in xx with coefficients (double build)
     k = [0.083333333333333384, 0.0013888888888859755, 3.3068783094433201e-05, 8.2671949071520734e-07,
          2.0876900642039142e-08, 5.2828494303844588e-10, 1.3455485951057888e-11,
          3.1615515500778219e-13, 1.230313397310169e-14]
     res = k[8]; for i=7..0: res = k[i] + xx*res
     (single build: [0.0833333358, 0.00138884981, 3.32480922e-05, 4.57548566e-07, 4.23758962e-07,
                     -2.48791736e-07, 8.78585737e-08, -1.64186869e-08, 1.26339661e-09])
  else: x=√xx; c = (1 − 0.5x/tan(0.5x))/xx
```

Notes:
- The intermediate activation is a **plain linear update** `act0 + h·Σ A F_act`, with no clamp, no exact filter and no DC-motor exact integration. Only the final `mj_advance` uses `mj_nextActivation`.
- The intermediate qvel is linear in qacc. Intermediate positions use `mj_integratePos` from `X[0].qpos` with the weighted corrected velocities.
- `qacc_warmstart` after RK4 is the **last stage's** `d->qacc` (`mj_advance` copies `d->qacc`, which holds stage 4's solution), not the B-averaged acceleration.
- `mj_step2` never runs RK4; it falls back to Euler.

---

## 10. Validity checks, warnings, timers

- `mj_checkPos`: for i<nq, if `mju_isBad(qpos[i])`: `mj_warning(BADQPOS,i)`. If AUTORESET is not disabled: `mj_resetData`, then `warning[BADQPOS].number++`, `lastinfo=i` (restoring the count that the reset cleared). Return at the first bad entry.
- `mj_checkVel`: same for qvel over awake dofs only (when `SLEEP && nv_awake<nv`, using `dof_awake_ind`).
- `mj_checkAcc`: same for qacc over awake dofs. After a reset it also calls **`mj_forward`**, so the outputs correspond to the reset state.
- `mj_warning(d,w,info)`: errors on an invalid type. It sets `lastinfo=info`, prints `"<text> Time = %.4f."` via `mju_warning` **only when `number==0`**, then `number++`. The texts are in `mju_warningText` (e.g. BADQPOS: "Nan, Inf or huge value in QPOS at DOF %zu. The simulation is unstable.").
- `mjWARN_INERTIA` is raised by `mj_factorM`, Euler and implicit when a pivot is clamped to mjMINVAL. `CONTACTFULL` and `CNSTRFULL` come from collision/constraint arena overflow.
- Timers (`engine_macro.h`):
  - `TM_START` does `_tm = mjcb_time ? mjcb_time() : 0`.
  - `TM_END(i)` adds `now − _tm` to `timer[i].duration` and does `number++`.
  - `TM_ADD(i)` adds duration without counting.
  - `TM_START1/TM_END1` use a second variable `_tm1`.
  - Without `mjcb_time`, all durations are 0 and the counts still increment.
- `mj_resetData` calls `mj_logTimingDiagnostics` before clearing, which logs the per-step average breakdown (POSITION..ADVANCE with POS_* and COL_* sub-rows) at `mjLOG_INFO`, topic `mjTOPIC_TIME_STP`, when `timer[STEP].number>0` and the average >0.

---

## 11. Data model

### 11.1 `mjModel` layout (engine_io.c)

- **Sizes.** The `mjtSize` (int64) fields listed in `MJMODEL_SIZES` are the 84 construction sizes passed to `mj_makeModel`, in this order: `nq nv nu nactuator nout na nbody nbvh nbvhstatic nbvhdynamic noct njnt ntree nM nB nC nD ngeom nsite ncam nlight nflex nflexnode nflexvert nflexedge nflexelem nflexelemdata nflexstiffness nflexbending nefm0dof nefm0L nefmCvert nflexelemedge nflexshelldata nflextexcoord nJfe nmesh nmeshvert nmeshnormal nmeshtexcoord nmeshface nmeshgraph nmeshpoly nmeshpolyvert nmeshpolymap nskin nskinvert nskintexvert nskinface nskinbone nskinbonevert nhfield nhfielddata ntex ntexdata nmat npair nexclude neq ntendon nJten nwrap nsensor nnumeric nnumericdata ntext ntextdata ntuple ntupledata nkey nmocap nplugin npluginattr nuser_body nuser_jnt nuser_geom nuser_site nuser_cam nuser_tendon nuser_actuator nuser_sensor nnames npaths nplastic`. They are followed by sizes set after construction: `nnames_map nJmom ngravcomp nemax njmax nconmax npolygonmax nmeshdegmax nuserdata nsensordata npluginstate nhistory narena nbuffer`.
- **Validation in `mj_makeModel`.** Every size must be ≥0 and < INT_MAX (except ntexdata and ntextdata), and nbody must be > 0. `nnames_map = 2·(nbody+njnt+ngeom+nsite+ncam+nlight+nflex+nmesh+nskin+nhfield+ntex+nmat+npair+nexclude+neq+ntendon+nactuator+nsensor+nnumeric+ntext+ntuple+nkey+nplugin)` must be < INT_MAX/2.
- **Single buffer.** `nbuffer = 63 + Σ (SKIP(offset) + sizeof(type)·nr·nc)` over `MJMODEL_POINTERS`, in X-macro order. `SKIP(o) = (64 − o%64)%64`. The sum uses overflow-checked arithmetic (`safeAddToBufferSize`). The buffer is `mju_malloc`ed and zeroed. `mj_setPtrModel` assigns each pointer at the next 64-byte-aligned address of the actual buffer and checks `ptr − buffer − SKIP(buffer) + 63 == nbuffer`. `mj_makeModel` then sets default option, visual and statistic.
- `mj_copyModel`: struct copy, restore the buffer pointer, re-run setPtr, memcpy each array. `mjv_copyModel` skips arrays marked `XNV`.
- **Binary MJB** (`mj_saveModel`/`mj_loadModelBuffer`): header `int[5] = {54321, sizeof(mjtNum), getnsize(), mj_version(), getnptr()}`, then all `mjtSize` sizes, `mjOption`, `mjVisual`, `mjStatistic`, `flg_gravcomp`, `flg_surfacevel`, `flg_adhesion` (mjtBool each), then the arrays in X-macro order with no padding. Loading checks every header field (precision mismatch is rejected), `nbuffer == sizes[nsize−1]`, that all bytes are consumed exactly, and `mj_validateReferences` (each id/adr field is within range of its target count; adr+num ≤ ntarget; sensor obj/ref types; exclude signatures; tuples).
- **Sparsity structures** built at compile time:
  - `mj_makeDofDofSparse` (reduced → C: simple dofs keep only the diagonal; upper → D: full symmetric; plain → M). Row nonzeros are the dof itself plus its ancestor chain via `dof_parentid`. Columns are sorted ascending within each row and the diagonal is last for the lower forms. `D_diag` gives the diagonal position in D rows.
  - `mj_makeBSparse`: body→dof rows = subtree dofs ∪ ancestor dofs, insertion-sorted.
  - `mj_makeDofDofMaps`: `mapM2D` (lower→symmetric, via `mju_lower2SymMap`), `mapD2M`, `mapM2M` (legacy → CSR).
  - `checkDBSparse` (called from `_resetData`) asserts that D row j equals B row `dof_bodyid[j]`.
- **Arena size `m->narena`** (`user_model.cc`):
  - If `memory` is given, use it verbatim (bytes).
  - Otherwise: `nconmax = (m->nconmax==−1 ? 100 : …)`, `njmax = (m->njmax==−1 ? 500 : …)`.
    - If `nstack==−1`: `narena = 8·max(1000, 5·ndense² + 20·nlinear)` with `ndense = njmax+neq+nv−nflexdof` and `nlinear = nq+nv+nu+na+nbody+njnt+ngeom+nsite+neq+ntendon+nwrap`. Here `nflexdof` comes from `EstimateFlexArena`, sparse only.
    - Else `narena = 8·nstack`.
    - Add `nconmax·sizeof(mjContact) + njmax·(8·4+14·8) + nv·12 + njmax·nv·(2·4+2·8) + njmax²·(4+8)`, plus flex solver allowances: the excess of `nH·(6·4+5·8) + nL·(3·4+nL_num·8)` over the linear term, plus half of it, plus `min(njmax·nflexdof_collide·24, 64 MB)`.
    - Round up to a whole MiB.

### 11.2 `mjData` layout

Header fields (scalars, not in the buffer): `narena, nbuffer, nplugin, pstack, pbase, parena, threadpool, threadlock, maxuse_stack, maxuse_arena, maxuse_con, maxuse_efc, solver[mjNISLAND·mjNSOLVER] (mjSolverStat: improvement, gradient, lineslope, nactive, nchange, neval, nupdate), solver_niter[20], solver_nnz[20], solver_fwdinv[2], warning[7] (lastinfo, number), timer[15] (duration, number), ncon, ne, nf, nl, nefc, nJ, efm_active, nefmK, nefmcon, nefmT, nefmA, nefmdof, nefmL, nY, nA, nisland, nidof, ntree_awake, nbody_awake, nparent_awake, nv_awake, flg_energypos, flg_energyvel, flg_subtreevel, flg_rnepost, time, energy[2]`, then `buffer`, `arena`, and finally `signature` (uint64).

**Main buffer** (`MJDATA_POINTERS`, fixed size, model-dependent; 64-byte aligned exactly like mjModel):

- State: `qpos(nq) qvel(nv) act(na) history(nhistory) plastic(nplastic) qacc_warmstart(nv) plugin_state(npluginstate)`
- Control: `ctrl(nu) qfrc_applied(nv) xfrc_applied(nbody×6) eq_active(neq, mjtBool)`
- Mocap: `mocap_pos(nmocap×3) mocap_quat(nmocap×4)`
- Dynamics: `qacc(nv) act_dot(na)`
- Other inputs and outputs: `userdata(nuserdata) sensordata(nsensordata) tree_asleep(ntree,int) plugin(nplugin,int) plugin_data(nplugin,uintptr)`
- POSITION fields as in §4, plus `crb(nbody×10) M(nC) qLD(nC) qLDiagInv(nv) bvh_active(nbvh) tree_awake body_awake body_awake_ind parent_awake_ind dof_awake_ind`, flex fields `flexvert_xpos flexelem_aabb flexelem_krot flex_hessian_valid flexvert_hessian(×6) flexedge_hessian(×9) flexedge_J flexedge_length bvh_aabb_dyn flexvert_lambda flexvert_conage`, and transmission fields `actuator_length(nout) moment_rownnz(nout) moment_rowadr(nout) moment_colind(nJmom) actuator_moment(nJmom)`
- VEL: `flexedge_velocity ten_velocity actuator_velocity(nout) cvel(nbody×6) cdof_dot(nv×6) qfrc_bias qfrc_spring qfrc_damper qfrc_gravcomp qfrc_fluid qfrc_adhesion qfrc_passive subtree_linvel subtree_angmom qH(nC) qHDiagInv(nv) qDeriv(nD) qLU(nD)`
- ACC: `actuator_force(nout) qfrc_actuator qfrc_smooth qacc_smooth qfrc_constraint qfrc_inverse cacc cfrc_int cfrc_ext`

**Arena** (`narena` bytes, separate malloc). The bottom part is the "arena", growing upward from offset 0 by `parena`, and holds variable-size step data (`MJDATA_ARENA_POINTERS`):
- contact (ncon);
- solver arrays: efc_type, efc_id, efc_J_rownnz/rowadr/rowsuper, efc_J_colind(nJ), efc_J(nJ), efc_pos, efc_margin, efc_frictionloss, efc_diagA, efc_KBIP(×4), efc_D, efc_R, tendon_efcadr(ntendon), efc_vel, efc_aref, efc_b, efc_state, efc_force;
- dual arrays: efc_Y* (nY), efc_AR* (nA);
- island arrays: tree_island … ifrc_constraint (sizes nisland, nidof, nefc, ntree, nv);
- efm_* arrays (sizes nv, nC, ntendon, nactuator, nefmK, nefmdof, nefmcon, nefmL).

`d->contact` always starts at `arena` offset 0. The top part is the **stack**, growing downward from `arena+narena` by `pstack`.

`mj_clearEfc(d)` (called at the start of `mj_collision`) NULLs all arena pointers. It sets ne=nf=nl=nefc=nisland=nidof=nJ=nY=nA=0, deactivates efm (efm_active=0, all nefm*=0), sets `contact=arena`, sets `efc_address=-1` for existing contacts, and `parena = ncon·sizeof(mjContact)`.

### 11.3 Arena and stack allocator (engine_memory.c)

- `mj_arenaAllocByte(d,bytes,align)`: `padding = (align − parena%align)%align` (fast mask path when align is a power of two). It fails and returns NULL if `parena+padding+bytes > narena − pstack`. Otherwise it returns `arena+parena+padding`, does `parena += padding+bytes`, and updates `maxuse_arena = max(., pstack+parena)`.
- **Stack.** With `bottom = arena+narena`, `top = bottom − pstack`, `limit = arena+parena`:
  - `stackalloc(size,align)`: `start = top − (size+redzone)`, `start −= start % align`, `new_top = start − redzone`. If `top − new_top > top − limit`, call `mju_error("mj_stackAlloc: out of memory, stack overflow…")`, a fatal error that includes nefc and ncon. On success `pstack = bottom − new_top`, and `maxuse_stack` and `maxuse_arena` are updated. A size of 0 returns NULL.
  - `mj_markStack`: pushes an `mjStackFrame{pbase, pstack, pc}` onto the stack itself and sets `pbase` to the frame address.
  - `mj_freeStack`: restores `pstack` and `pbase` from the frame. Under ASAN it checks that mark and free happen in the same function.
  - `mjSTACKALLOC(d,n,type)` = `mj_stackAllocInfo(d, n·sizeof(type), _Alignof(type), __func__, __LINE__)`. `mj_stackAllocNum` and `mj_stackAllocInt` guard against size overflow.
  - **Threaded mode** (`d->threadlock`): mark and free are no-ops, and allocations atomically reserve `mj_stackBytes(size,align) = size + align − 1 + 2·redzone` via `mj_atomic_add_size_t(&pstack, …)`. `mju_dispatch` marks before and frees after the parallel region.
- `mj_stackBytesAvailable = narena − pstack − parena`.

### 11.4 Construction and reset

`mj_makeRawData(&d,m)` does the following:
1. Compute nbuffer (same scheme as the model), copy `narena = m->narena`, and malloc buffer and arena.
2. `mj_setPtrData`: assign pointers with 64-byte alignment, NULL the arena pointers, `contact = arena`.
3. Set `threadpool=0`, `threadlock=0`, `nplugin=0`.
4. Fill `tree_asleep = kAwake`.
5. `mj_updateSleepInit(m,d,1)`.

`mj_makeData` = makeRaw + `mj_initPlugin` (calls `plugin->init(m,d,i)`; on failure, destroys the earlier instances and errors) + `mj_resetData`.

`_resetData(m,d,debug_value)` (`mj_resetData` uses 0; `mj_resetDataDebug` uses a byte pattern):
1. Warn if `nhistory>0 && timestep<=0`. Save plugin_state and plugin_data.
2. `pstack=0` (unless a threadpool exists), `pbase=0`, `parena=0`, arena pointers NULL, `contact=arena`. Zero `maxuse_*`, warnings, timers, solver stats, `solver_niter`, `solver_nnz`, `solver_fwdinv`. Zero all variable sizes (ncon … nefmL), `time=0`, `energy=0`, and the lazy flags.
3. `memset(buffer, debug_value, nbuffer)`. Then explicitly zero qpos, qvel, act, plastic. `mj_resetCtrl` sets ctrl=0 except `ctrl[ctrladr]=1` for SO3 actuators with QUAT ctrlspec. Set `eq_active=eq_active0`, and zero qfrc_applied, xfrc_applied, qacc, qacc_warmstart, act_dot, userdata, mocap_pos, mocap_quat, flexvert_lambda, flexvert_conage, flex_hessian_valid.
4. ctrl history: `buf[0]=0`, `buf[1]=n−1`, `times[j] = −(n−j)·dt`, values 0 (quaternion identity for SO3 QUAT).
5. Sensor history: `buf[0] = period>0 ? (phase==0 ? −period : phase) : −dt`, `buf[1]=n−1`. With period>0, `times[j] = ceil((t0 − (n−1−j)·period)/dt)·dt` with `t0 = phase==0 ? −period : phase` (unrounded if dt<=0). Otherwise `−(n−j)·dt`. Values 0.
6. `qpos = qpos0`. `mj_resetPlastic`: joints get `plastic[adr:adr+nPOS] = qpos_spring`; tendons get `tendon_lengthspring[2]`; flexes are zeroed (counts per interp mode) and `flex_hessian_valid=0`.
7. Restore plugin state and data, and call `plugin->reset(m, plugin_state+stateadr, plugin_data, i)`.
8. `tree_asleep = kAwake`. If SLEEP: `mj_updateSleepInit(m,d,1)`. If no tree has `mjSLEEP_INIT`, run `mj_kinematics; mj_comPos; mj_camlight; mj_tendon` to fill static values. Otherwise run `mj_forward`, set the init trees' `tree_asleep=−1`, `mj_sleep`, warn if fewer slept than requested (waking those that did not), zero qacc_smooth and qfrc_smooth, and `mj_clearEfc`.
9. `mj_updateSleep`. Mocap: `mocap_pos/quat = body_pos/quat` of the mocap bodies (or identity quaternions if `body_mocapid` is NULL). Run `checkDBSparse` and `signature = m->signature`.

`mj_resetDataKeyframe(m,d,key)`: logs timing, `_resetData`, then for a valid key copies time, qpos, qvel, act, mocap_pos, mocap_quat, ctrl, plastic from the `key_*` arrays.

`mj_copyData` (flg_all=1) and `mjv_copyData` (flg_all=0, skips XNV) error if buffer sizes differ, arena sizes differ, or `src->pstack != 0`. They struct-copy, then memcpy the buffer arrays, then copy arena arrays at the same offsets relative to `arena`. plugin_data is preserved and `plugin->copy` is called. `threadpool` is not copied.

`mj_deleteData`: `mju_threadpool(d,0)`, plugin `destroy`, free buffer and arena.

---

## 12. Callbacks, plugins, thread pool, global tables, names, printing

**Callbacks** (`engine_callback.c`, global, default NULL; `mj_resetCallbacks` NULLs all):

| Callback | Called from | Semantics |
|---|---|---|
| `mjcb_control(m,d)` | `mj_forwardSkip` before `mj_fwdActuation` (skipped if ACTUATION disabled); end of `mj_step1` (unconditional) | user writes ctrl/applied forces |
| `mjcb_passive(m,d)` | end of `mj_passive`, before passive plugins | adds to qfrc_passive |
| `mjcb_contactfilter(m,d,g1,g2)` | collision driver | return 1 to discard the pair |
| `mjcb_sensor(m,d,stage)` | sensor stages, for user sensors | |
| `mjcb_time()` | timer macros | seconds-like counter |
| `mjcb_act_dyn/gain/bias(m,d,i)` | `mj_fwdActuation` for USER types | see §6.3 |

**Plugin hook points** (`mjpPlugin`, `mjplugin.h`). Capabilities: `mjPLUGIN_ACTUATOR=1`, `SENSOR=2`, `PASSIVE=4`, `SDF=8`.
- `init` is called by `mj_makeData`, `mj_copyData` (when it allocates) and `mj_initPlugin`.
- `reset(m, state, data, i)` is called by `_resetData`.
- `copy` by `mj_copyData`; `destroy` by `mj_deleteData` and `mj_makeRawData` (re-make).
- `actuator_act_dot` runs after the built-in act_dot. ACTUATOR `compute` runs after the built-in forces, before the tendon/force clamps.
- PASSIVE `compute` runs after `mjcb_passive`.
- SENSOR `compute` runs in the stage that matches `needstage` (POS also runs `needstage==NONE`). It is preceded by `mj_subtreeVel` (VEL) or `mj_rnePostConstraint` (ACC) if not already done, and followed by cutoff application.
- `advance` runs in `advanceFinish` after `time += h`. SDF hooks are used by collision. `visualize` is used by the scene.
- `nstate` sizes `plugin_state`.
- Plugins are looked up per call with `mjp_pluginCount()` plus `mjp_getPluginAtSlotUnsafe(slot,nslot)`. An invalid slot is an error.
- `mjp_registerPlugin` validates name (non-null, non-empty) and `nattribute` (0..255) and appends to the global table if unique (thread-safe). `mj_getPluginConfig` returns the attribute string of an instance.

**Global table** (`engine_global_table.h`): a linked list of `TableBlock<T>` with `kBlockSize=15` objects per block, aligned to `kCacheLineBytes=256`. Pointers are never invalidated, so reads are lock-free. Writes go through a reentrant mutex (`ReentrantWriteLock` with a thread-local depth counter). The count is an atomic with acquire/release ordering. Keys are compared case-insensitively. The same mechanism holds plugins, resource providers, archive providers, decoders and encoders.

**Thread pool** (`engine_thread.cc`). `mju_threadpool(d,n)` creates `ThreadPoolContext` with n workers, replacing any existing pool of a different size (n=0 deletes it). `mju_dispatch(m,d,func,arg,ntask)`:
- With no pool or `ntask<2`, it runs sequentially on the calling thread with `thread_id=0`.
- Otherwise it calls `mj_markStack`, sets `threadlock=1`, and `Dispatch` flips the sign of the `signal_` atomic to wake the workers (`atomic::wait/notify_all`). Tasks are pulled with `next_.fetch_add(1, relaxed)` by the main thread (`thread_id 0`) and the workers (`1..n`). The main thread spins with `_mm_pause`/`yield` until `ndone_ == nthread`. It then updates `maxuse_*`, clears threadlock, and calls `mj_freeStack`.
- The atomics are aligned to 64 bytes. `mju_numThread` returns workers+1.
- Task-to-thread assignment is dynamic, so a task must not depend on thread_id for results. The island solvers satisfy this.

**Names** (`engine_name.c`). `mj_hashString(s,n)`: `h=5381; h = ((h<<5)+h) ^ c` (djb2-xor) `% n`. `names_map` holds per-type sub-tables of size `2·count`. The type order from the end of the map follows the `_getnumadr` fallthrough chain (body, joint, geom, site, camera, light, …). `mj_name2id` uses linear probing from the hash slot and stops at a −1 slot. `mj_id2name` returns NULL for the empty name. `mj_actuatorInputName` names actuator inputs.

**Printing** (`engine_print.c`). `mj_printModel` and `mj_printFormattedModel` (float format default `"% -9.2g"`, validated) dump all sizes, options and arrays via the X-macros. `mj_printData` dumps mjData. There are no numerical side effects.

---

## 13. GPU (MuJoCo Warp) implementation

### 13.1 Batching model, precision, layout

- **Precision.** Warp's `float` is `wp.float32`, and every `Model`/`Data` array of `float` is float32. Constants come from the Python `mujoco` module (`MJ_MINVAL = mujoco.mjMINVAL = 1e-15`, `MJ_MAXVAL = 1e10`, `MJ_MINIMP`, `MJ_MAXIMP`, `MJ_MAXCONPAIR`, `MJ_MINMU`, `MJ_MINAWAKE`) and are used unchanged in float32 kernels. Host-side precomputation in `io.py` uses numpy float64 and converts on upload.
- **Data.** Every per-step field is `[nworld, ...]`. Scalars per world are `[nworld]` (`time`, `nefc`, `ne`, `nf`, `nl`, `solver_niter`, `nisland`, `nidof`, `ntree_awake`, `nbody_awake`, `nv_awake`, `overflow`, `energy (vec2)`). Vector/matrix types replace MuJoCo's flat strides: `xpos: vec3`, `xquat: quat`, `xmat: mat33`, `cdof/cvel/xfrc_applied: spatial_vector` (rotation first), `cinert/crb: vec10`, `mocap_quat: quat`.
- **Model.** Fields whose spec starts with `"*"` are batched with a leading dimension of size `batch_sizes.get(name, 1)` (`put_model(mjm, batch_sizes)`). Every kernel reads `arr[worldid % arr.shape[0], ...]`, so a length-1 leading dimension is shared by all worlds and length nworld gives per-world randomization (e.g. `opt.timestep`, `gravity`, `body_mass`, `actuator_gainprm`, `geom_size`). Non-`*` fields are unbatched. `warp_util.mark_batched` sets `_is_batched=True` on such arrays.
- **Option** (Warp `types.Option`). Array-valued per world: `timestep, tolerance, ls_tolerance, noslip_tolerance, ccd_tolerance, sleep_tolerance, gravity, wind, magnetic, density, viscosity, impratio_invsqrt`. Python ints: `integrator, cone, solver, iterations, ls_iterations, noslip_iterations, ccd_iterations, disableflags, enableflags, sdf_initpoints, sdf_iterations`. Warp-only fields:

  | Field | Default (put_model) | Meaning |
  |---|---|---|
  | `impratio_invsqrt` | `1/sqrt(max(impratio, 1e-15))` | replaces impratio |
  | `broadphase` | NXN if #included geom pairs < 250000; else SAP_TILE if ngeom<1000; else SAP_SEGMENTED | |
  | `broadphase_filter` | `PLANE|SPHERE|OBB` | |
  | `graph_conditional` | True | CUDA-graph conditional while-loop for solver iterations |
  | `run_collision_detection` | True | False keeps user-populated contacts |
  | `run_rne_postconstraint` | False | force rne_postconstraint even without sensors |
  | `run_qacc_smooth` | True | False: Newton starts from qacc_warmstart and skips qacc_smooth/factorization |
  | `contact_sensor_maxmatch` | 64 (or numeric field `contact_sensor_maxmatch`) | |
  | `warn_overflow` | `OverflowType.ALL` | bool True/False maps to ALL/0 |
  | `deterministic` | `DeterminismType.NONE` | bits CONTACTS=1, CONSTRAINT=2, ATOMICS=4 |

  `opt.tolerance` is raised to `max(tolerance, 1e-6)` at `put_model`, because float32 cannot reach 1e-8. **`jacobian`, `o_margin`, `o_solref`, `o_solimp`, `o_friction`, `disableactuator`, `impratio` (raw) do not exist in Warp.**
- **Unsupported flags** (`put_model` raises `NotImplementedError` for any bit outside the Warp enum). DisableBit: MIDPHASE, AUTORESET (NATIVECCD is accepted but ignored). EnableBit: only ENERGY, INVDISCRETE and SLEEP exist; OVERRIDE, FWDINV, DIAGEXACT, IPC and SPECULATIVE are rejected. Solver: PGS is rejected. Integrator: all five are accepted. Enum checks in `put_model`: any `actuator_trntype/dyntype/gaintype/biastype`, `eq_type`, `geom_type`, `sensor_type`, `wrap_type` or `tree_sleep_policy` value not in Warp's enums raises. Warp's enums omit **TrnType SO3, DynType PID, GainType PID** (GainType.SO3 is declared but BiasType.SO3 is missing, so SO3 actuators fail) and SleepPolicy NEVER/ALLOWED/INIT. Also rejected: body, actuator and sensor plugins (only geom SDF plugins are kept, with up to 128 float attributes); noslip with the discrete integrator; flex equality under SLEEP; quadratic flex interpolation; contact sensor linvel/angvel with flexes; flex with SDF or hfield geoms; flex internal collisions; rigid flexes; self-collision or passive contact of interpolated flexes; dense Jacobian with nv>60; SLEEP with a solver other than Newton; SNH without discrete; discrete+Newton with non-simple flex.
- **`is_sparse` differs from C.** For `jacobian==AUTO`, Warp is sparse iff `nv > 32`, while C is sparse iff `nv >= 60`. Explicit DENSE/SPARSE follow `mj_isSparse`. This changes the constraint Jacobian layout (`efc.J` is `[nworld, njmax_pad, nv_pad]` dense or `[nworld, 1, njmax_nnz]` sparse CSR).
- **Fixed-capacity buffers** (chosen at `make_data`/`put_data`):
  - `naconmax`: total contacts across **all worlds** (one flat contact array of length naconmax; `contact.worldid` tags each). Default `nconmax·nworld`.
  - `naccdmax` (CCD contacts, default naconmax, must be ≤ naconmax).
  - `njmax`: constraints **per world**, so `efc.*` is `[nworld, njmax]`.
  - `njmax_nnz`: sparse J nonzeros per world.
  - `nvmax`: compacted active-dof capacity under sleep (default nv).
  - Padded sizes: `njmax_pad = roundup(njmax, 16)`. `nv_pad = roundup(nv + augment, 16)` if sparse or nv>32 (augment=1 for Newton with nv>32), else `roundup(nv,4)`. `nvmax_pad = ((max(nvmax,1)+16)//16)·16`.
  - Default sizes: `valid_sizes = (2+(i%2))·2^(i//2+3)` for i=0..18 gives {16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048,3072,4096,6144,8192}. `nconmax = max(nv·0.35·[nhfield>0]·10 + 45, 256·has_flex, 64·has_sdf, mjd.ncon)` rounded up to the next valid size (48 for a plain model). `njmax = max(nv·2.26·[nhfield>0]·18 + 53, 512·has_flex, 256·has_sdf, mjd.nefc)` rounded up the same way (64 for a plain model). Values above 8192 are used verbatim.
  - Default `njmax_nnz` (sparse): `min(max(total,1), njmax·nv)`. `total` assumes all equalities active (connect 3×, weld 6× the union-of-trees dof count; joint 1 or 2; tendon union nnz; flex edges; flexstrain neig·nv), all frictionloss rows, all limit rows, and `njmax × max_contact_nnz`, where max_contact_nnz is the maximum tree-union dof count over contact pairs, filtered geom pairs and flex combinations. Dense: `njmax·nv`.
  - **Overflow.** C never silently drops rows (its arena is sized and it warns with CONTACTFULL/CNSTRFULL). Warp truncates and records `d.overflow[world] |= OverflowType` bits: NEFC (nefc>njmax), BROADPHASE (ncollision>naconmax), NARROWPHASE (nacon>naconmax), and others set by other subsystems (NJMAX_NNZ, CCD, HFIELD, CONTACT_MATCH, NVMAX, EPA_HORIZON, ITERATIONS, LS_ITERATIONS, TACTILE). These are set in the `_next_time` kernel at the end of each step, which prints once per kernel if the `warn_overflow` bit is set.
- **M factor layout** (`m_block_layout`). The diagonal blocks of M are the kinematic trees. A block with ≤6 dofs that is compact (nnz==size) or fully triangular uses the scalar path: compact uses reciprocal diagonals with no factor storage (`Q_LD_BLOCK_COMPACT=-2`), triangular uses a packed dense Cholesky. A block with ≤64 dofs (`M_BLOCK_DENSE_MAX`) uses tile Cholesky. Larger blocks use sparse LDL (`Q_LD_BLOCK_SPARSE=-1`). `qLD` is `[nworld, qLD_block_total + (nC if any sparse block)]`. The dense blocks store **upper Cholesky factors** (U with UᵀU=M), not MuJoCo's L'DL, so `d.qLD` is not bit-compatible with C `qLD`. `put_data` computes `np.linalg.cholesky(blk).T` for dense blocks and copies `mjd.qLD` into the LDL region. The sparse LDL updates are grouped by tree depth (`qLD_updates`, `qLD_all_updates`, `qLD_level_offsets`) for level-parallel factorization.

### 13.2 Pipeline (`forward.py`)

`forward(m,d)`:
```
check_discrete(m)                                     // host-side Python validation (subset of mj_checkDiscrete)
sleep_enabled = SLEEP && !ISLAND-disabled
if sleep_enabled: sleep.wake; sleep.update_sleep
fwd_position(m,d, factorize=False)
d.sensordata.zero_(); sensor_pos; _energy_pos
fwd_velocity; sensor_vel; _energy_vel
if !ACTUATION-disabled and m.callback.control: callback.control(m,d)
fwd_actuation; fwd_acceleration(m,d, factorize=True)
solver.solve(m,d)
if DISCRETE: _launch_discrete_free_gyro(m,d,d.qacc,d.qacc,False)
if opt.run_rne_postconstraint or (!SENSOR-disabled and m.sensor_rne_postconstraint): smooth.rne_postconstraint
sensor_acc(m,d, skip_rne_postconstraint=True)
```

`step(m,d)`: `forward`, then by integrator EULER→`euler`, RK4→`rungekutta4`, IMPLICIT/IMPLICITFAST→`implicit`, DISCRETE→`discrete` (`_advance(m,d,d.qacc)`). **No `mj_checkPos/Vel/Acc`, no autoreset, no FWDINV comparison, no BADCTRL check.** NaNs propagate.

`step1`: `check_discrete; fwd_position(factorize = not skip_qacc_smooth); sensordata.zero_(); sensor_pos; energy_pos; fwd_velocity; sensor_vel; energy_vel; control callback (only if ACTUATION not disabled)`. `step2`: `fwd_actuation; fwd_acceleration (factorize=False, uses the factor from step1); solver.solve; discrete gyro; rne_postconstraint; sensor_acc`, then IMPLICIT*/DISCRETE/else Euler (RK4 falls back to Euler).

`fwd_kinematics`: `smooth.kinematics, com_pos, camlight, flex, tendon`. With sleep and ntendon>0: `sleep.wake_tendon; sleep.update_sleep_trees`. **`smooth.flex` also computes `flexedge_J`, `flexedge_length` and `flexedge_velocity` in one kernel (`_flex_edges`, per edge)**. It uses the qvel present at kinematics time and zeroes velocity for edges with an unattached vertex or no Jacobian row.

`fwd_position(factorize)`:
```
fwd_kinematics; smooth.crb; smooth.tendon_armature; if factorize: smooth.factor_m
if opt.run_collision_detection:
   sleep: collision; wake_collision; awake_prev = clone(body_awake); update_sleep;
          collision(awake_prev=awake_prev)      // incremental 2nd pass: only pairs with newly woken bodies, appended
   else:  collision
constraint.make_constraint          // also computes efc vel/aref (no separate reference stage)
if sleep: (neq>0: wake_equality); update_sleep; island.island     // islands only built when sleeping
smooth.transmission
if DISCRETE: derivative.eff_build
```
There is no `mj_projectConstraint` (no PGS, no DIAGEXACT). Islands are not used to split the solve unless sleeping. The C pipeline always builds islands, which only changes the parallel structure; mathematically the solve is the same.

`fwd_velocity`: kernel `_actuator_velocity` (nworld, nactuator) gives `vel = Σ moment·qvel` per actuator row; `_tendon_velocity` (nworld, ntendon) sums `J·qvel`, skipping `J==0.0` entries. Then `smooth.com_vel; passive.passive; smooth.rne; smooth.tendon_bias(qfrc_bias); if DISCRETE: derivative.eff_shift`. **actuator_velocity is computed even with ACTUATION disabled** (C zeroes it).

`fwd_actuation`:
```
if nactuator==0 or ACTUATION-disabled: act_dot=0, qfrc_actuator=0, actuator_force=0
else:
  ctrl = nhistory>0 ? read_ctrl_delayed(...) : d.ctrl
  _actuator_force            (nworld, nactuator)  → act_dot, actuator_force
  callbacks act_dyn, act_gain, act_bias (if set; they write arrays directly)
  if ntendon: ten_actfrc = 0; _tendon_actuator_force (atomic_add per tendon); _tendon_actuator_force_clamp
  qfrc_actuator = 0; _qfrc_actuator (nworld, nactuator; atomic_add moment·force into dof)
  _qfrc_actuator_gravcomp_limits (nworld, nv): += qfrc_gravcomp if gravity enabled && jnt_actgravcomp; clamp by jnt_actfrcrange
if DISCRETE: derivative.eff_actuation; constraint.regularize_constraint
```

`_actuator_force` per (world, actuator), with `ctrl` = first input `ctrl[uadr]`, clamped with `actuator_ctrllimited[uadr]`/`ctrlrange[uadr]` unless CLAMPCTRL is disabled:
- act_dot: INTEGRATOR `ctrl` (**scalar only, the last slot**); FILTER/FILTEREXACT `(ctrl−act)/max(dynprm0, MINVAL)`; MUSCLE `muscle_dynamics`; DCMOTOR (slew, integral, voltage via `dcmotor_voltage(ctrl_in,...)`, temperature, bristle, current; same formulas as C, written per slot); USER `0` (the callback writes it); NONE `0`. Finally `act_dot_out[act_last] = act_dot`.
- actearly: INTEGRATOR/NONE/DCMOTOR read `act_in[act_last]`; DCMOTOR uses per-slot exact or Euler updates and clamps to actrange if actlimited; everything else uses `next_act(...)`. Without actearly, `ctrl_act = act[act_last]`. Stateless actuators use the clamped ctrl.
- gain FIXED/AFFINE/MUSCLE/DCMOTOR (`dcmotor_resistance` temperature-adjusted; for te<=0 the input is 0 with no ctrl, `dcmotor_voltage` if `ctrlspec & 7`, else the slew-limited `u_first`). USER: gain 0.
- bias AFFINE/MUSCLE/DCMOTOR back-EMF.
- `_wrap_period` (servo shape only; no PID) applies `ctrl_act −= P·round((ctrl_act−len)/P)`.
- `force = gain·ctrl_act + bias`, **clamped by forcerange here**, then the DC-motor cogging and LuGre terms are added. LuGre `z_dot` is read back from `act_dot_out`, written earlier in the same thread.

`fwd_acceleration(factorize)`:
```
_qfrc_smooth(enable_sleep) (nworld, nv): passive − bias + actuator + applied   // asleep tree dofs set to 0
xfrc_accumulate (support.apply_ft over all bodies)
if smooth.skip_qacc_smooth(m,d):  (sleep && !DISCRETE: island.update_active_dofs)       // qacc_smooth left stale
elif DISCRETE: derivative.eff_solve(m,d,d.qacc_smooth)
elif sleep:    island.update_active_dofs; solver.smooth_solve_compact      // dense compacted Cholesky in nvmax_pad
elif factorize: smooth.factor_solve_i(m,d,d.M,d.qLD,d.qLDiagInv,d.qacc_smooth,d.qfrc_smooth)   // fused factor+solve
else: smooth.solve_m(m,d,d.qacc_smooth,d.qfrc_smooth)
```
`skip_qacc_smooth` is true iff `!run_qacc_smooth && solver==NEWTON && !WARMSTART-disabled && noslip_iterations==0 && iterations>0 && njmax>0`. The operation order in `_qfrc_smooth` is `passive − bias + actuator + applied`; C does `(passive−bias)+applied+actuator`. See §15.

`solver.solve` (detail in the solver file):
- With sleep: `update_active_dofs`. If not DISCRETE: `solve_compact` (+noslip) and island mapping, then return.
- If `njmax==0 or nv==0`: `qacc = qacc_smooth`, `solver_niter=0`.
- Otherwise `_solve` (+noslip).
- In `_solve`: DISCRETE replaces qfrc_smooth by `qfrc_smooth + efm_c + efm_ca` (zero for asleep trees). The warmstart kernel `_solve_init_dof` per (world, dof) does:
  ```
  if !run_qacc_smooth:  qacc = qacc_warmstart                     // always
  elif warmstart (WARMSTART not disabled): qacc = nefc>0 ? qacc_warmstart : qacc_smooth
  else: qacc = qacc_smooth
  sparse && nefc==0: qfrc_constraint = 0
  ```
  **Warp does not compare warmstart cost against qacc_smooth cost** (C picks the cheaper of the two). With constraints present, Warp always starts from `qacc_warmstart`. It also does not reset unconstrained-island dofs to qacc_smooth. Starting points differ, so iterates and (with finite iterations or tolerance) results differ.
- `efc.Ma = M·qacc` is maintained by the solver and used as the integrators' right-hand side.

### 13.3 Integration kernels

`_advance(m,d,qacc,qvel=None)`, in launch order:
1. `_next_activation` (nworld, nactuator), `act_dot_scale=1`, `limit=True`. DCMOTOR is per slot (current exact `te=max(MINVAL,dynprm0)`, bristle exact ZOH, integral Euler + Imax clip, other slots Euler; **no actrange clamp for DCMOTOR**, as in C). Everything else uses `next_act`: FILTEREXACT exact; **USER returns act unchanged**; others Euler; clamp if actlimited. INTEGRATOR servo-shaped actuators with period>0 are wrapped to `act − P·round((act−len)/P)`. There is no SO3 re-anchor, no `disableactuator` and no ACTUATION check (with actuation disabled, act_dot was zeroed so act only gets clamped and wrapped).
2. `_next_velocity` (nworld, nv): `qvel += qacc_scale·qacc·h`, with the factors evaluated as `qacc_scale * qacc * timestep`.
3. `_next_position` (nworld, njnt), using `qvel_in = qvel or d.qvel`: FREE `pos += h·v·scale`, `quat = math.quat_integrate(quat, ω·scale, h)`; BALL the same; SLIDE/HINGE `q += h·v·scale`. `math.quat_integrate(q,v,dt)`: `n=|v|; v=normalize(v); angle=dt·n; q_res=axis_angle(v,angle); q=normalize(q); return normalize(q ⊗ q_res)`. **It renormalizes the result and always renormalizes the input**, whereas C renormalizes only if `|‖q‖−1| > 1e-15` and does not renormalize the result.
4. Plastic, if `nplastic>0 && !SPRING-disabled`: `_next_plastic_joint` (nworld, njnt), `_next_plastic_tendon` (nworld, ntendon; uses qvel_in), `_next_plastic_flex_stretch` (nworld, nflexelem), `_next_plastic_flex_bend` (nworld, nflexedge), `_next_plastic_flex_interp` (nworld, nflexintcell). These are per-element parallel ports of `advanceFinish`. They run **after** qpos is updated, so they see the new qpos, as C does. Element-parallel stretch writes `flex_hessian_valid`.
5. `history.insert_ctrl_history` (nworld, nactuator): inserts the current `d.ctrl` at `d.time` (pre-advance time). Sensor history is inserted inside the sensor stages (`apply_sensor_delay`: save fresh values, overwrite with the delayed read at `t − delay`, insert the fresh values at `t`; for interval-only sensors, read from the buffer when `time_prev + period > t`). C inserts sensor history in `advanceStart`.
6. `_next_time_builder(warn_overflow)` (nworld): `time += timestep` plus the overflow checks.
7. `wp.copy(d.qacc_warmstart, d.qacc)`.
8. If sleep is enabled: `sleep.sleep; fwd_velocity; sleep.update_sleep`, **after** the state update. In C, sleeping happens in `advanceStart`, before integration, followed by a full `mj_forwardSkip(POS)`.

There is no plugin `advance`.

`euler(m,d)`: if neither EULERDAMP nor DAMPER is disabled, **always** takes the implicit-damping path, even when no dof has damping:
1. `_compute_damping_deriv` (nworld, nv): `util_misc.dof_damping_total` (dof damping + actuator damping × gear², including polys), then `_poly_force_deriv(damping, poly, v, 1)`.
2. `M_int = clone(d.M)`; `_euler_damp_qfrc` adds `h·deriv` to the diagonal `M_rowadr+M_rownnz−1`.
3. `smooth.factor_solve_i(m,d,M_int,qLD_tmp,qLDiagInv_tmp,qacc,d.efc.Ma)`, i.e. the rhs is **`efc.Ma = M·qacc`, not `qfrc_smooth+qfrc_constraint`**.
4. `_advance(qacc)`.

Otherwise `_advance(d.qacc)`.

`implicit(m,d)`:
- IMPLICIT: `implicit_qlu`: `deriv_smooth_vel` gives M−h·qDeriv in M-structure; `_map_m2d` (nworld, nD) scatters via `mapM2D`; `deriv_rne_vel` adds `+h·d(qfrc_bias)/dv`. Then `factor_solve_lu(qLU, qacc, efc.Ma)`, `implicit_free_body(solve=True)`, `_advance`.
- IMPLICITFAST, only if not all of {ACTUATION, SPRING, DAMPER} are disabled: `qDeriv_M = deriv_smooth_vel` (= M − h·qDeriv in M-structure); `_implicit_free_body_reset_m` (nworld, nbodyfree) resets the free-body rows to M; `factor_solve_i(qDeriv_M, …, qacc, efc.Ma)`; `implicit_free_body(efc.Ma → qacc, solve=True)`; `_advance`.
- Otherwise `_advance(d.qacc)`.

`_implicit_free_body` (nworld, nbodyfree; `body_freeadr` = bodies that are the root of a 6-dof tree with a single FREE joint):
1. Skip if the tree is asleep.
2. A = M block. If DAMPER is not disabled, add `h·poly_deriv(damping)` to the diagonal.
3. Add the gyroscopic `free_bias_vel_blocks`: `A[0:3,3:6] += −h·mass·lin` and `A[3:6,3:6] += h·rot`.
4. Fluid derivatives, if density>0 or viscosity>0: ellipsoid-model geoms via `_geom_ellipsoid_fluid_B` projected by `_project_spatial_B`, or the inertia-box model `_deriv_box_fluid`, each subtracting `h·JᵀBJ`.
5. LU-factor the 6×6 (`math.lu_factor_6x6`) and solve, overwriting the 6 rows of the output.

This rebuilds the block from scratch rather than using qDeriv. Under IMPLICIT, C's `mj_implicitSkip` does not do the free-body local solve at all, because the full LU already includes the RNE derivative. Warp applies `implicit_free_body` under IMPLICIT as well.

`rungekutta4(m,d)`: `A=[0.5,0.5,1]` (diagonal only), `B=[1/6,1/3,1/3,1/6]`. It clones qpos, qvel, act (and, if plastic, ten_length, ten_J, flexvert_xpos, cdof, subtree_com), with `qvel_stage = clone(qvel)`, and accumulates `qvel_rk += B0·qvel_stage`, `qacc_rk += B0·qacc`, `act_dot_rk += B0·act_dot`. For i in 0..2 with `a=A[i]`, `b=B[i+1]`:
1. `_rk_perturb_state`: `qpos = qpos_t0 ⊕ h·a·qvel_stage`; `qvel = qvel_t0 + a·qacc·h`; act through `_next_activation(act_t0, scale=a, limit=False)`, so FILTEREXACT uses the exact formula, USER is unchanged and DCMOTOR is per slot. C uses plain linear `act0 + h·a·act_dot` at stages.
2. **`forward(m,d)`**, the full forward with sensors.
3. `_rk_correct_velocity`: `qvel_stage = deriv_expmap(h·a·qvel_stage_prev, d.qvel)` on rotational dofs; other dofs are copied from d.qvel.
4. Accumulate with b.

Afterwards it restores the state, sets `d.act_dot = act_dot_rk`, and calls `_advance(qacc_rk, qvel_rk)`. **`d.time` is not advanced at intermediate stages** (C sets `time = t + C_i·h`), which matters for delayed ctrl/sensor reads and time-dependent callbacks. `math.deriv_expmap` uses the **single-precision** Bortz coefficients `[0.0833333358, 0.00138884981, 3.32480922e-05, 4.57548566e-07, 4.23758962e-07, -2.48791736e-07, 8.78585737e-08, -1.64186869e-08, 1.26339661e-09]` with threshold `xx < 3.0625`, otherwise the closed form, with the same (−π,π] reduction.

### 13.4 Determinism and atomics

`_tendon_actuator_force` and `_qfrc_actuator` use `wp.atomic_add`. The float summation order is nondeterministic unless `opt.deterministic & ATOMICS`, which compiles with `wp.DeterministicMode.RUN_TO_RUN` (`_qfrc_actuator` with `deterministic_max_records = nv`). C sums `momentᵀ·force` in row order. Contact ordering (CONTACTS) and constraint row ordering (CONSTRAINT) are covered in the collision and constraint files. Kernels are cached by argument size/hash (`warp_util.cache_kernel`) so that conditional CUDA graphs can capture them.

### 13.5 `io.py`: model and data transfer

`put_model(mjm, batch_sizes)`:
- Checks the CUDA toolkit/driver (warns if conditional graphs are unsupported, i.e. CUDA < 12.4) and runs the feature checks (§13.1). Warns about friction < MJ_MINMU for condim≥3/4/6.
- Builds `Option` (tolerance clamp, warp-only defaults) and copies all same-named `MjModel` fields. Compatibility shims for older mujoco versions (eq_data padding, plastic fields, tendon_type).
- Precomputed fields:
  - `nv_pad`; `nacttrnbody`; `nsensortaxel`; `nrangefinder`; `nmaxcondim = max(geom_condim, pair_dim, flex_condim, 0)`; `nmaxpyramid = max(1, 2(nmaxcondim−1))`.
  - Geometry and flex flags: `has_sdf_geom/ellipsoid/plane`, `has_flex_selfcollide/passive/snh`, `flex_simple`, `has_non_simple_flex`, `has_flex_edge_stiff_damp`, `has_tendon_stiffness/damping`, `has_efm_actuator`, flex dims.
  - Block dims: CG block = `clamp(roundup32(nv), 32, 256)`; `linesearch_iterative=256` for nv>500.
  - `is_sparse`; `has_fluid`; `nflexintcell`.
  - Effective-metric sparsity from a temporary `MjData` under the discrete integrator (`_get_efm_data`: nefmK, nefmdof, nefmL=9·nefmdof, efm_K CSR, efm_dofid, efm_dofblk, efm0_active).
  - `flex_interp_assemblable`; `flex_cell_map`; `max_ten_J_rownnz`.
  - `body_tree` (bodies grouped by depth, for level-parallel kinematics); `body_subtreenum`; branch lists `body_branches`/`body_branch_start` (leaf-to-root paths); `mocap_bodyid`.
  - `body_fluid_ellipsoid`, `body_is_free`/`body_freeadr`, `body_fluid_box_adr`, limited-joint address lists, `dof_tri_row/col` (triu of nv), `body_isdofancestor [nbody, nv_pad]`.
  - NXN geom pair tables `nxn_geom_pair(_filtered)` and `nxn_pairid` (contact pair id, −1 default, −2 filtered; filtering by contype/conaffinity, same weld body, both weld bodies dof-less, parent filter, exclude signature; predefined pairs override), collision-sensor pairs, `geom_pair_type_count`.
  - Plugin attribute vectors; equality-type address lists; flexstrain Jacobian sparsity.
  - Tendon helper lists (fixed/site/geom, pulley scale `1/divisor` after each pulley); sensor address lists per stage (excluding limit/contact/touch etc.); `sensor_rne_postconstraint`.
  - Tactile maps; `m_block_layout`/`M_tiles`/`qLD_updates`; `M_fullm_i/j` (dof–ancestor pairs), `M_elemid`, `M_hinit_i`, dense-block gather indices, `qD_fullm_i/j`, the gather-based `M_mulm_*` (each row lists all (col, madr) including the mirror); flex lookup maps (elem/edge/shell/vert → flex), face maps, rod hinge maps.
- Finally every array spec is turned into a `wp.array` with batch sizes, and `mark_batched` is applied.

`make_data(mjm, nworld, nconmax, nccdmax, njmax, njmax_nnz, naconmax, naccdmax, nvmax)`:
- Validates sizes (all ≥0, `naccdmax ≤ naconmax`, `nccdmax ≤ nconmax`, `0 ≤ nvmax ≤ nv`, `nworld ≥ 1`). `naconmax = naconmax ?? nconmax·nworld`.
- Allocates contacts (`efc_address [naconmax, nmaxpyramid]` filled with −1; flex/elem/vert vec2i(−1,−1) when nflex>0) and constraints (sparse or dense J).
- Runs `mujoco.mj_kinematics` on a fresh `MjData` to seed `xquat`, `xmat`, `ximat`, `geom_xpos`, `geom_xmat`. **Static geom poses are never recomputed on device.**
- `qpos = qpos0`; mocap from `body_pos/quat`; `eq_active0`; history and plastic from the fresh `MjData` (which `mj_resetData` initialized); `tree_asleep = −(1+MJ_MINAWAKE)`, `tree_awake=1`; body_awake STATIC for static bodies (AWAKE if the root is mocap).
- Island arrays (sized by ntree/nv/njmax) and compact arrays (allocated only if sleep or nvmax was given).

`put_data(mjm, mjd, ...)`: same size logic. It checks `nconmax ≥ mjd.ncon`, `naconmax ≥ ncon·nworld`, `njmax ≥ nefc`. It runs `mj_kinematics`, and if M is all zero also `mj_comPos, camlight, tendon, crb, factorM`. It tiles the contacts nworld times (contact i of world j at index `j·ncon+i`), expands `efc_address` to all pyramid/elliptic rows, sets `worldid` and `type` (PASSIVE for `exclude==4`), and leaves `geomcollisionid` uninitialized. Constraint rows are tiled. For the discrete integrator, `efc_D`/`efc_aref` come from a temporary `mj_forwardSkip` with springs, dampers, actuation and fluid disabled, if nefc matches. The JTDAJ block list is rebuilt for sparse Newton. J is converted dense↔sparse as needed. The qLD block re-factor is described in §13.1. `efm_ts`/`efm_as` are scattered by tid/aid; `dof_frictionadr` is rebuilt.

`reset_data(m,d,reset=None)` (per-world mask):
- Zeros xfrc_applied and M.
- Mocap from body_pos/quat.
- Clears contacts for the masked worlds (`nacon` is zeroed by world 0's thread).
- Sleep reset: tree_asleep=−11, tree_awake=1, body_awake STATIC/AWAKE, identity awake index lists.
- `reset_nworld`: solver_niter, ne, nf, nl, nefc = 0; time=0; energy=0; n*_awake = full; qpos=qpos0; qvel, qacc_warmstart, qfrc_applied, qacc, ctrl, act, act_dot, sensordata, userdata = 0; eq_active=eq_active0; flex_hessian_valid=False; overflow=0.
- `history.reset_history` (same initial timestamps as C).
- `reset_plastic`: zero all, then copy qpos_spring for joints with 4 values if BALL else 1, and tendon lengthspring.
- `sleep.update_sleep` if SLEEP.

Differences from C reset: no kinematics, `mjSLEEP_INIT` is unsupported, SO3 quaternion ctrl is not set to identity, and qacc_smooth/qfrc_smooth etc. are left stale. `reset_data_keyframe` copies `key_*` per world.

`get_data_into`: copies one world back into an `MjData`, limited to `min(nacon, naconmax)` contacts and `min(nefc, njmax)` constraints.

`override_model`: CLI-style `field=value` overrides including enum names. `find_keys`, `make_trajectory`, `load_trajectory`: keyframe/ctrl replay helpers.

### 13.6 Other modules

- `warp_util.py`: `is_array_spec`, `mark_batched`, `EventTracer` + `event_scope` (CUDA-event timing tree; disabled inside `capture_while/if`), `cache_kernel` (keyed by argument `.size` or hash, plus function name), `check_toolkit_driver`.
- `history.py`: Warp port of the C ring buffer. `_history_find_index`, `_history_read_scalar/vector` and `_history_insert_scalar/vector` use **`1e-6` as the time-match and extrapolation epsilon** (C uses `mjMINVAL=1e-15`). `read_ctrl_delayed` does a direct copy when `nsample==0 or delay==0`. `read_ctrl`/`read_sensor` are public query kernels. `init_ctrl_history`/`init_sensor_history` set user buffers.
- `util_pkg.py`: `check_version("pkg>=x.y.z")`, a version comparator that strips `-local` suffixes; dev/pre-release components sort below the release.
- `cli.py`: absl flags (`nworld` default 8192, `nstep` 1000, `nconmax`, `njmax`, `njmax_nnz`, `nccdmax`, `override`, `keyframe` 0, `noise_std` 0.01, `noise_rate` 0.1, `nvmax`, render flags), `load_model` (MJB or XML with SDF plugin registration), the `_ctrl_noise` kernel (Halton-sequence ctrl noise with rate `exp(−h/noise_rate)`), `init_structs`, and `unroll` (benchmark loop).

---

## 14. MJX (JAX) differences

- `mjx.forward/step` dispatch to MuJoCo Warp through `mujoco.mjx.warp` when `impl==WARP`, or to the C++ FFI for `impl==CPP`. For the JAX implementation, the following hold.
- Supported integrators: EULER, RK4, IMPLICITFAST (DISCRETE is listed in the enum). IMPLICIT is unsupported. `implicitfast` with fluid parameters raises. Supported solvers: CG and Newton (no PGS). Unsupported enable bits: OVERRIDE, ENERGY, FWDINV, ISLAND, DIAGEXACT, SPECULATIVE (only INVDISCRETE, SLEEP, IPC are enumerated). Disable bits: all except MIDPHASE. No flex, no plasticity. Restricted contact sensors. No PID/DCMOTOR/SO3 (dyntypes limited to NONE, INTEGRATOR, FILTER, FILTEREXACT, MUSCLE).
- `fwd_position`: kinematics, com_pos, camlight, tendon, crb, tendon_armature, factor_m, collision, make_constraint, transmission. `fwd_velocity`: `actuator_velocity = moment @ qvel`, `ten_velocity = ten_J @ qvel`, com_vel, passive, rne, tendon_bias. `fwd_acceleration`: `qfrc_smooth = passive − bias + actuator + (applied + xfrc)`, then `solve_m`. `forward` skips the solver if `efc_J.size == 0`.
- Warmstart: picks `qacc_warmstart` if `cost(warm) < cost(smooth)` (strict <), as in C's tie handling.
- `_advance`: activation (FILTEREXACT exact, else Euler, clip to actrange or ±inf, INTEGRATOR wrap), `qvel += qacc·h`, `qpos = integrate(qvel)`, `time += h`, `qacc_warmstart = qacc`. No history, no plastic, no sleep.
- `euler`: if EULERDAMP is not disabled, `M + h·diag(dof_damping)` (**linear dof damping only**, no polynomial, no actuator damping, and no check whether damping exists), factor and solve `qfrc_smooth + qfrc_constraint`.
- `implicit` (implicitfast): `qm = M − h·qderiv` (dense), Cholesky via `jax.scipy.linalg.cho_factor/cho_solve`. **No free-body gyroscopic local solve.**
- `rungekutta4`: `jax.lax.scan` over `(diag(A), B[1:], T)` with `unroll=3`. Stage `kact = act0 + a·act_dot·h` (linear, as in C), stage time `t` set (as in C), **full `forward` with sensors**, and the RKMK correction only if any FREE/BALL joint exists.

---

## 15. Reproduction pitfalls and parity notes

1. **Floating-point accumulation order** matters for bitwise parity:
   - `qfrc_smooth` in C is `((passive − bias) + applied) + actuator`, then `+ xfrc` via `mj_applyFT` per body in body order. Warp computes `passive − bias + actuator + applied` in one expression, then atomically adds the xfrc contributions.
   - `qfrc_actuator = momentᵀ·force` in C walks rows in actuator order. Warp uses atomics (nondeterministic unless `deterministic & ATOMICS`).
   - `mj_advance` uses `qvel[i] += h*qacc[i]` (product h·qacc first). Warp computes `qvel + scale*qacc*timestep`, which is `(1.0*qacc)*h` and gives the same value since scale=1, but RK stages use `a*qacc*h` vs C's `h*(a*F)`.
2. **Warmstart selection.** C evaluates the Gauss + constraint cost at both `qacc_warmstart` and `qacc_smooth` and picks the warmstart unless it is strictly worse; unconstrained island dofs are then reset to `qacc_smooth`. Warp always uses the warmstart when `nefc>0`. With finite iterations or tolerance this changes results.
3. **Integrator right-hand side.** C Euler/implicit solve `M̂ a = qfrc_smooth + qfrc_constraint`. Warp solves `M̂ a = efc.Ma (= M·qacc)`. They are identical only if the solver residual `M qacc − qfrc_smooth − qfrc_constraint` is exactly zero.
4. **Euler without damping.** C skips the extra factorization when no dof has damping (no `dof_damping`, no dampingpoly, no `jnt_actuatorid`). Warp always factorizes and solves unless EULERDAMP or DAMPER is disabled. Mathematically this is a no-op, but rounding differs.
5. **Quaternion integration.** C: `normalize4(q)` only if `|‖q‖−1| > 1e-15`, `q ← q ⊗ exp(ω h)`, no final normalization; axis `(1,0,0)` if `|ω| < 1e-15`; exact identity if `angle==0`. Warp: always normalizes the input and the output, and uses `wp.normalize` (zero vector for zero input).
6. **Jacobian layout threshold.** For `jacobian=AUTO`, C is sparse for nv≥60 and Warp for nv>32. Sparse and dense paths differ in summation order.
7. **RK4 stage differences.** C stage activations are linear (no clamp, no exact filter) and the stage time is advanced. Warp stage activations go through `next_act` (exact filter for FILTEREXACT, DC-motor per slot) and **the stage time is not advanced**. Warp and MJX run sensors at stages; C skips them (energy is still recomputed in C). `qacc_warmstart` is stage 4's qacc in all three.
8. **Order of operations in `advance`.** C: history insert, activation update, sleep (followed by a full `mj_forwardSkip(POS)` if anything slept), then qvel, qpos, time, plugin advance, plastic, warmstart. Warp: activation, qvel, qpos, plastic, ctrl history, time, warmstart, then sleep and `fwd_velocity`. Sensor history insertion happens inside the sensor stage in Warp and in `advanceStart` in C.
9. **Actuator force pipeline order.** C: gain·input + bias, then actuator plugins, then the tendon total-force scaling (on pre-clamp forces, DC-motor mechanical terms excluded), then the per-actuator forcerange clamp, then DC-motor cogging/LuGre added. Warp: forcerange clamp and DC-motor terms inside `_actuator_force`, then the tendon total scaling (whose sums include clamped forces and DC-motor terms). In C, the flag `tendon_frclimited` is set by `if (ntendon && !tendon_frclimited && trntype==TENDON) tendon_frclimited = tendon_actfrclimited[trnid]`. Because the loop keeps reassigning while the flag is 0, the flag becomes 1 as soon as any tendon actuator targets a limited tendon. Implement it as "any tendon actuator whose tendon is limited", with one caveat: sleeping, disabled, plugin and SO3 actuators reach `continue` before this check, so they do not contribute to the flag. They do still contribute to `tendon_total_force`, which loops over all actuators.
10. **INTEGRATOR with multiple controls.** C writes `act_dot` for all ctrlnum slots. Warp writes only the last slot.
11. **Unsupported in Warp:** PID gain/dyn, SO3 transmission/gain/bias, actuator/sensor/body plugins, `disableactuator`, OVERRIDE, FWDINV, DIAGEXACT, PGS, IPC, SPECULATIVE, MIDPHASE disable, AUTORESET (no NaN checks at all), BADCTRL zeroing, plugin `advance`, `mjSLEEP_INIT`.
12. **Joint actuator-force limit.** In C, `clampVec(qfrc_actuator, jnt_actfrcrange, jnt_actfrclimited, njnt, jnt_dofadr)` clamps only the first dof of each joint (relevant for BALL/FREE joints with actfrclimited). Warp's `_qfrc_actuator_gravcomp_limits` clamps **every dof** of a limited joint.
13. **Actuation disabled.** C leaves `act_dot` untouched (stale), zeroes `actuator_velocity`, and does not advance `act`. Warp zeroes `act_dot` and `actuator_force`, computes `actuator_velocity`, and still runs `_next_activation` with act_dot=0, which clamps to actrange and wraps.
14. **Time accumulates in float32 in Warp.** After many steps, `time` drifts (float32 ulp at t=1000 s is about 6e-5 s). Delayed ctrl/sensor lookups also use a 1e-6 epsilon instead of 1e-15, so exact-match and ZOH choices can differ from C near sample times.
15. **Solver tolerance.** Warp clamps `tolerance` to ≥1e-6 at `put_model`, so solvers may stop earlier than C (default 1e-8).
16. **Free-body local solve.** C applies the 6×6 LU free-body correction only for IMPLICITFAST (after the global LDL), using qDeriv rows + h·B. Warp applies it for both IMPLICIT and IMPLICITFAST and rebuilds A from M, damping, the gyroscopic term and fluid terms directly. Under IMPLICIT, Warp therefore overwrites the global LU solution for free bodies. For a standalone free body the two agree mathematically, but not bitwise. Under discrete, C's `mj_discreteGyro` requires no constraint Jacobian support on the body (checked per step); Warp's `_discrete_free_gyro_kernel` checks the analogous conditions in-kernel (nefc rows, efm_K rows, tendon/actuator metric terms, passive flex contact).
17. **Static geoms in Warp** are positioned once at `make_data`/`put_data` time from `mj_kinematics` at qpos0 (or the given mjd). Model edits to static geom pose after `make_data` are not reflected.
18. **Buffer overflow.** When Warp's `nefc > njmax` or `nacon > naconmax`, constraints and contacts are silently dropped (only flagged in `d.overflow`). C grows within its arena or warns. Choose njmax and naconmax large enough for parity.
19. **Checks and resets.** C `mj_step` can reset the entire state mid-step when NaN/Inf or |x|>1e10 appears (qpos, qvel, qacc) unless AUTORESET is disabled. `mj_checkAcc` additionally reruns `mj_forward`. A parity harness must disable AUTORESET or avoid bad states.
20. **`mj_step2` with RK4** uses Euler in all implementations.
21. **Warp's `forward` zeros `sensordata`** before the position sensors. In C, sensordata of skipped sensors retains old values. Compare only computed sensors.
22. **`d.qLD` content differs:** Warp stores upper Cholesky for dense blocks vs C L'DL. Do not compare factor arrays directly; compare solves.
23. **Islands.** C always solves per island when islands exist (not disabled, not IPC, not monolithic fold), possibly in parallel. The result equals the monolithic solve only up to per-island convergence (each island stops independently at tolerance). Warp solves each world monolithically unless sleeping. Per-island convergence vs monolithic convergence gives different iterates when iterations are capped.
24. **`gramDet3D(grad,q)`** (used by 3D flex plasticity), with `q` = 6 squared edge lengths in `mj_stretchEdges` order: `a=q0, b=q2, c=q4, d=½(q0+q2−q1), e=½(q0+q4−q5), f=½(q2+q4−q3)`; `c0=bc−f²`, `c1=ac−e²`, `c2=ab−d²`, `c3=ef−cd`, `c4=df−be`, `c5=de−af`; `det = a c0 + d c3 + e c4`; `grad = (c0+c3+c4, −c3, c1+c3+c5, −c5, c2+c4+c5, −c4)`. The depressed cubic `c3 u³ + c1 u + c0 − c3 = 0` is solved with Cardano when `R=(c3−c0)/(2 c3 m m²) ≥ 1` (`u = m(w+1/w)`, `w = cbrt(R+√(R²−1))`), otherwise trigonometrically (`u = 2m cos(acos(max(−1,R))/3)`), where `m=√max(0,−c1/(3c3))` and u stays 1 if `c3 ≤ 1e-15` or `m² ≤ 1e-15`.
25. **`mj_warning` prints only the first occurrence** per warning type until reset. Its counters are part of `mjData` and are not part of the physics state.

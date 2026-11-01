# 09 — Passive forces: springs, dampers, gravity compensation, fluid, flex elasticity

**Sources covered**

| Layer | Files |
|---|---|
| C engine (core) | `src/engine/engine_passive.c` (all of it), `src/engine/engine_passive.h` |
| C engine (helpers used here) | `engine_core_util.c/.h` (`mj_tendonSpringDamper`, `mj_actuatorDamping`, `mj_flexGather/Scatter`, `flexMap`, `mju_flexGatherState`, `mj_flexHasStiffness`, `mj_effFlexContactPossible`, `mj_flexHessian`, `mj_flexHessianMul`, rod-bending `mj_rodHinge/mj_rodPair/mj_rodHingeWeights/mj_rodPairWeights`, stretch helpers `mj_stretch*`, SNH `mj_snhCubic/mj_snhVolume/snhProject/snhSVD/snhEigen3/snhPositive3/snhCubicMetric`), `engine_util_misc.c/.h` (`mju_polyForce`, `mju_geomSemiAxes`, `mju_flexPhi/Dphi`, `mju_flexGatherCellState`, `mju_flexGatherFaceState`, `mju_flexFaceNormal2D`, `mju_flexPolarRotation`), `engine_util_spatial.c` (`mju_mat2Rot`, `mju_transformSpatial`), `engine_inline.h` (`mji_subQuat`, `mji_quat2Vel`), `engine_setconst.c` (`setFluid`, `addedMassKappa`, `jnt/tendon_actuatorid`, `flg_gravcomp`), `engine_forward.c` (call site; actuator-side gravcomp) |
| Compiler (what the runtime reads) | `src/user/user_flexcomp.cc` (`ComputeStiffness`, `MetricTensor`, `ComputeBasis`, `ComputeSNH`, `ComputeBending`, `ComputeRodBending`, `ComputeLinearStiffness2D/3D`, `ComputeInterpBending`), `user_objects.cc` (geom `fluid[]` packing) |
| Docs | `doc/computation/fluid.rst` |
| Plugins | `plugin/elasticity/{cable.cc,cable.h,elasticity.cc,elasticity.h,register.cc}` — only the **cable** plugin remains; the former shell/membrane/solid plugins have been folded into the engine's flex elasticity (`elastic2d`, `elastic3d`, interpolated flexes) and are specified here as engine code. |
| MuJoCo Warp | `mujoco_warp/_src/passive.py` (all kernels), plus `support.py` (`apply_ft`, `jac_dof`, `mat33_to_quat_polar`, `compute_interp_cell_quat`, `compute_interp_face_quat`, `flex_phi/dphi/dphi2D`), `util_misc.py` (`_poly_force`, `actuator_target_damping`, `tendon_damping_total`), `math.py` (`quat_sub`, `quat_to_vel`, `orthogonals`), `io.py` (precomputed maps/flags) |
| MJX | `mjx/mujoco/mjx/_src/passive.py` |

**Role in the pipeline.** `mj_passive` is called from `mj_fwdVelocity` (engine_forward.c) after `flexedge_velocity`, `ten_velocity`, `actuator_velocity` and `mj_comVel` (so `cvel`, `cdof_dot` are fresh) and before constraint references and `mj_rne`. It produces the generalized force components `qfrc_spring`, `qfrc_damper`, `qfrc_gravcomp`, `qfrc_fluid`, `qfrc_adhesion` and their sum `qfrc_passive` (gravcomp only for joints without `actgravcomp`). Passive forces are explicit forces in the smooth dynamics (`qfrc_smooth = qfrc_passive - qfrc_bias + qfrc_applied + qfrc_actuator + ...`); their velocity derivatives (`mjd_passive_vel`, engine_derivative.c) are consumed by the implicit/implicitfast integrators, and the stiffness/damping operators are re-used by the discrete ("effective metric") integrator. Inputs: kinematics (`xpos`, `xmat`, `xipos`, `ximat`, `geom_xpos/xmat`, `subtree_com`, `cdof`), `cvel`, `qpos`, `qvel`, `ten_length/velocity/J`, `flexedge_length/velocity/J`, `flexvert_xpos`, node positions for interpolated flexes, `d->plastic` (plasticity state), contacts (adhesion, passive flex contact).

---

## 1. Constants, layouts and conventions

| Symbol | Value | Where |
|---|---|---|
| `mjMINVAL` | `1e-15` (double build) | mjtnum |
| `mjPI` | π | |
| `mjNPOLY` | 2 (high-order polynomial coefficients for stiffness/damping) | mjmodel.h |
| `mjNFLUID` | 12 (per-geom fluid params) | mjmodel.h |
| `mjROD_HINGE_SIZE` | 10 (per 1D-rod bending record) | engine_core_util.h |
| `mjROD_SMOOTH` | `1e-4` (relative smoothing of curved rod references) | engine_core_util.c |
| `BEND_EDGE_SIZE` | 10 (per interpolated-shell bending edge) | engine_passive.c |
| `kBoundEps` | `1e-14` (setFluid clamping) | engine_setconst.c |
| `rotEPS` (mju_mat2Rot) | `1e-9` double / `1e-6` single; max 500 iterations | engine_util_spatial.c |
| SNH SVD Jacobi reltol | `4e-15` double / `2e-6` single; max 24 sweeps | snhSVD |
| SNH SVD rank tol | `1e-12` double / `1e-6` single | snhSVD |
| SNH Cholesky test tol | `1e-12*scale` double / `1e-5*scale` single | snhPositive3 |

Disable/enable flags used: `mjDSBL_SPRING`, `mjDSBL_DAMPER`, `mjDSBL_GRAVITY`, `mjDSBL_CONTACT`, `mjENBL_SLEEP`, `mjENBL_IPC` (only to gate passive flex contact).

`geom_fluid` layout (12 per geom): `[0]` interaction coef (fluidshape: 0 = none, >0 = ellipsoid; used as a multiplicative scale, compiled as 0/1), `[1]` blunt drag C_D,blunt (default 0.5), `[2]` slender drag C_D,slender (0.25), `[3]` angular drag C_D,ang (1.5), `[4]` Kutta lift C_K (1.0), `[5]` Magnus lift C_M (1.0), `[6..8]` virtual mass (per unit density), `[9..11]` virtual inertia (per unit density). `[1..5]` come from MJCF `fluidcoef`; `[6..11]` are computed by `setFluid` (see §6.3). `user_objects.cc`: `fluid[0]=fluid_ellipsoid; fluid[i+1]=fluid_coefs[i]` (i<5), others 0 until setConst.

Polynomial force convention (`mju_polyForce`, engine_util_misc.c):

```
mju_polyForce(linear, poly, x, n, flg_odd):
  x = flg_odd ? |x| : x
  res = linear; xpow = 1
  for i in 0..n-1: xpow *= x; res += poly[i]*xpow
  return res
force = -x * mju_polyForce(...)
```
So stiffness (flg_odd=0): `f = -x*(k + p0 x + p1 x²) = -k x - p0 x² - p1 x³` — the p0 term is even in x (same sign for ±x). Damping (flg_odd=1): `f = -v (b + p0 |v| + p1 v²)`, odd in v.

---

## 2. `mj_passive` — top level (engine_passive.c)

```
mj_passive(m, d):
  sleep_filter = ENABLED(SLEEP) && d->nv_awake < m->nv
  nv  = sleep_filter ? d->nv_awake : m->nv
  zero qfrc_spring, qfrc_damper, qfrc_gravcomp, qfrc_fluid, qfrc_adhesion, qfrc_passive
      (indexed by dof_awake_ind if sleep_filter, else all nv)
  if DISABLED(SPRING) && DISABLED(DAMPER): return          # nothing else, not even gravcomp/fluid/adhesion
  mj_springdamper(m,d)
  has_gravcomp = mj_gravcomp(m,d)
  has_fluid    = mj_fluid(m,d)
  mj_contactPassive(m,d)       # adds into qfrc_spring (effective-metric passive flex contact)
  has_adhesion = mj_adhesion(m,d)
  qfrc_passive = qfrc_spring + qfrc_damper                (awake dofs / all)
  if has_fluid:    qfrc_passive += qfrc_fluid
  if has_adhesion: qfrc_passive += qfrc_adhesion
  if has_gravcomp: for each (awake) dof: if !jnt_actgravcomp[dof_jntid[dof]]: qfrc_passive[dof] += qfrc_gravcomp[dof]
  if mjcb_passive: mjcb_passive(m,d)
  for each plugin instance i with capabilityflags & mjPLUGIN_PASSIVE: plugin->compute(m,d,i,mjPLUGIN_PASSIVE)
```

Notes:
- **Summation order** (affects FP): `(spring + damper) + fluid + adhesion + gravcomp`. Note `mj_contactPassive` adds into `qfrc_spring` *before* the sum, so it is part of `spring`.
- Both disable flags set → *everything* passive (incl. gravcomp, fluid, adhesion) is zero. Hence actuator-side gravcomp (`mj_fwdActuation`, adds `qfrc_gravcomp` for `jnt_actgravcomp` joints) also sees zeros.
- The sleep-filtered branch uses `mju_zeroInd/mju_addInd/mju_addToInd` over `dof_awake_ind`; sleeping dofs keep their previous values (not zeroed).
- `mjcb_passive` and passive plugins write directly into `qfrc_passive` (e.g. the cable plugin uses `mj_applyFT(..., d->qfrc_passive)`), *after* composition. They are not reflected in `qfrc_spring`/`qfrc_damper`.
- Plugin slot validity: `mjp_getPluginAtSlotUnsafe`; null plugin → `mjERROR("invalid plugin slot")`, null `compute` → `mjERROR`.

---

## 3. `mj_springdamper` — joints, dofs, flexes, flex edges, tendons

```
enbl_spring = !DISABLED(SPRING); enbl_damper = !DISABLED(DAMPER)
sleep_filter = ENABLED(SLEEP) && d->ntree_awake < m->ntree
nbody = sleep_filter ? d->nbody_awake : m->nbody
```

### 3.1 Joint springs (body-major loop)

For each (awake) body `i` (via `body_awake_ind` if filtered), each joint `j` in `[body_jntadr[i], +body_jntnum[i])`:
- `stiffness = jnt_stiffness[j]`, `spoly = jnt_stiffnesspoly + 2j`. Skip if `stiffness == 0 && spoly == 0` (`mju_isZero`).
- `qspring = jnt_plasticadr[j] >= 0 ? d->plastic + jnt_plasticadr[j] : m->qpos_spring + jnt_qposadr[j]` (plastic state replaces the reference; for free joints the plastic block is 7 numbers like qpos).

Per type (`padr = jnt_qposadr[j]`, `dadr = jnt_dofadr[j]`):

**FREE** (then falls through to BALL with `padr+=3, dadr+=3, qspring+=3`):
$$ \Delta = q_{0:3} - q^s_{0:3},\quad r = \|\Delta\|,\quad k = \text{polyForce}(stiffness, spoly, r, 2, 0),\quad f_{0:3} \mathrel{+}= -k\,\Delta $$
(note `+=`, `mji_addToScl3`; the polynomial is evaluated on the *distance* r, making the spring isotropic.)

**BALL** (and rotational part of FREE):
```
quat = qpos[padr:padr+4]; mju_normalize4(quat)      # normalize only if |norm-1| > mjMINVAL; norm<mjMINVAL -> (1,0,0,0)
dif  = mji_subQuat(quat, qspring)                   # 3-vector: qspring * quat(dif) = quat
r = |dif|;  k = polyForce(stiffness, spoly, r, 2, 0)
qfrc_spring[dadr:dadr+3] += -k * dif
```
`mji_subQuat(qa,qb)`: `qdif = conj(qb) * qa`; `mji_quat2Vel(res, qdif, dt=1)`:
```
axis = qdif[1:4]; s = |axis| (mji__normalize3: if s<mjMINVAL axis=(1,0,0) else axis/=s)
angle = 2*atan2(s, qdif[0]); if angle > π: angle -= 2π
res = axis * angle
```
The reference `qspring` (from `qpos_spring` or plastic) is **not** normalized here.

**SLIDE / HINGE** (assignment, not accumulation):
$$ x = q - q^s,\qquad f = -x\cdot\text{polyForce}(stiffness, spoly, x, 2, 0) $$

### 3.2 DOF dampers (dof-major loop)

For each (awake) dof `i` (`dof_awake_ind` if filtered; loop count `d->nv_awake`):
```
poly = copy(dof_dampingpoly + 2i)
damping = dof_damping[i] + mj_actuatorDamping(m, mjOBJ_JOINT, dof_jntid[i], poly)   # adds actuator poly into poly
if damping != 0 || poly != 0:
    v = qvel[i];  qfrc_damper[i] = -v * polyForce(damping, poly, v, 2, flg_odd=1)
```
`mj_actuatorDamping(type, id, poly)`:
- `actuatorid = jnt_actuatorid[id]` (or `tendon_actuatorid`). Precomputed in setConst: −1 none; ≥0 single actuator; −2 multiple actuators. An actuator is registered only if it has nonzero `actuator_damping`, `actuator_dampingpoly` or `actuator_armature`, and trntype is JOINT/JOINTINPARENT (joints) or TENDON (tendons).
- Single: `g2 = actuator_gear[6*actuator_outadr[a]]^2`; `damping = actuator_damping[a]*g2`; `poly[k] += actuator_dampingpoly[2a+k]*g2`.
- Multiple (`< -1`): scan all actuators `k` with `actuator_trnid[2k]==id` and matching trntype, accumulate the same.
- Note: for multi-dof joints (ball/free) *every* dof of the joint receives the same actuator damping.

### 3.3 Flex elasticity dispatch

```
for f in 0..nflex-1:
  if flex_rigid[f]: continue
  if flex_dim[f] == 1: mj_flexPassiveBend1D(f); continue
  if flex_interp[f] != 0:
      mj_flexPassiveInterp(f)          # corotational linear FE stretch (trilinear/quadratic)
      mj_flexPassiveBendInterp(f)      # shell bending (only interp<0)
  else:
      mj_flexPassiveBend(f)            # 2D cotangent / cubic-shell bending
      mj_flexPassiveStretch(f)         # StVK (2D/3D) or Stable Neo-Hookean (3D)
```
Note: flex elasticity is *not* sleep-filtered and not gated by `enbl_spring`/`enbl_damper` at the dispatch level (each function checks). See §7 for each routine.

### 3.4 Flex-edge spring-dampers

```
for f: stiffness = enbl_spring ? flex_edgestiffness[f] : 0
       damping   = enbl_damper ? flex_edgedamping[f]   : 0
       if flex_rigid[f] || (stiffness==0 && damping==0): continue
       for e in [flex_edgeadr[f], +flex_edgenum[f]):
         if flexedge_rigid[e]: continue
         fs = stiffness * (flexedge_length0[e] - flexedge_length[e])
         fd = -damping * flexedge_velocity[e]
         for j in sparse row e of flexedge_J (rowadr/rownnz/colind):
             qfrc_spring[col] += J*fs; qfrc_damper[col] += J*fd
```
`flexedge_velocity` is zero for interpolated and rigid flexes (computed in `mj_fwdVelocity` only for non-interp non-rigid). Always sparse.

### 3.5 Tendon spring-dampers

For each tendon `i` (skipped if `sleep_filter && mj_sleepState(TENDON,i) != mjS_AWAKE`):

`mj_tendonSpringDamper` (engine_core_util.c):
```
fs = fd = 0
if !DISABLED(SPRING) and (tendon_stiffness[i] || tendon_stiffnesspoly[i] != 0):
    L = ten_length[i]
    [lower, upper] = tendon_plasticadr[i] >= 0 ? plastic[adr:adr+2] : tendon_lengthspring[2i:2i+2]
    x = L > upper ? L-upper : (L < lower ? L-lower : 0)          # dead-band
    fs = -x * polyForce(stiffness, spoly, x, 2, 0)
if !DISABLED(DAMPER):
    dpoly = copy(tendon_dampingpoly[i]); damping = tendon_damping[i] + mj_actuatorDamping(TENDON, i, dpoly)
    if damping || dpoly != 0: v = ten_velocity[i]; fd = -v * polyForce(damping, dpoly, v, 2, 1)
```
Then if `fs || fd`: for each sparse entry of `ten_J`: `qfrc_spring[k] += J*fs; qfrc_damper[k] += J*fd`. `tendon_lengthspring` = `[lower, upper]` (from `springlength`, one or two values; `-1` = use length at `qpos_spring`, resolved at compile time by `setSpringLength`).

---

## 4. Gravity compensation — `mj_gravcomp`

```
if !flg_gravcomp || DISABLED(GRAVITY) || |opt.gravity| == 0: return 0
for b in 1..nbody-1 (awake set if sleep_filter = ENABLED(SLEEP) && nbody_awake < nbody):
   if body_gravcomp[i] != 0:
      force = -gravity * (body_mass[i] * body_gravcomp[i]);  torque = 0
      mj_applyFT(force, torque, xipos[i], body i, qfrc_gravcomp)
      has = 1
return has
```
`flg_gravcomp = (number of bodies with body_gravcomp > 0) > 0` (setConst). `mj_applyFT` uses `mj_jacSparse` over the body chain (sparse) or dense `mj_jac`; with a non-NULL zero torque the rotational Jacobian product still runs (adds zeros).

Composition: dofs of joints with `jnt_actgravcomp=1` do **not** receive gravcomp in `qfrc_passive`; instead `mj_fwdActuation` adds `qfrc_gravcomp[jnt_dofadr : +dofnum]` into `qfrc_actuator` (dofnum by type {free:6, ball:3, slide:1, hinge:1}) **before** clamping by `jnt_actfrcrange` — i.e. actuator gravcomp is subject to actuator force limits. That branch re-checks `flg_gravcomp && !DISABLED(GRAVITY) && |gravity|`.

---

## 5. Fluid forces — `mj_fluid`

```
if opt.viscosity == 0 && opt.density == 0: return 0    # wind alone does nothing
for b in 0..nbody-1 (awake set if filtered):
   if body_mass[i] < mjMINVAL: continue
   use_ellipsoid = any geom j of body i with geom_fluid[12*j] > 0
   use_ellipsoid ? mj_ellipsoidFluidModel(i) : mj_inertiaBoxFluidModel(i)
return 1
```
If any child geom uses the ellipsoid model, the inertia-box model is disabled for that body and *only* geoms with `fluid[0] != 0` contribute.

### 5.1 Inertia-box model — `mj_inertiaBoxFluidModel(m,d,i)`

Equivalent box **full** dimensions (from principal inertia `I = body_inertia[3i]`, mass `M`):
```
box[0] = sqrt( max(mjMINVAL, I1 + I2 - I0) / M * 6.0 )
box[1] = sqrt( max(mjMINVAL, I0 + I2 - I1) / M * 6.0 )
box[2] = sqrt( max(mjMINVAL, I0 + I1 - I2) / M * 6.0 )
```
(`box = 2 r` with r the half-sizes in fluid.rst.)

Local velocity at CoM in inertial frame: `mj_objectVelocity(mjOBJ_BODY, i, lvel, flg_local=1)`:
`lvel = [R^T ω ; R^T (v_root − (xipos − subtree_com[root]) × ω)]` where `cvel=[ω; v_root]`, `R = ximat`. (`mju_transformSpatial(res, cvel, 0, xipos, subtree_com[rootid], ximat)` — motion transform: `lin -= (newpos-oldpos) × ang`, then rotate by Rᵀ.) Dof-less (static/mocap-welded) body → zero velocity.

Wind: `wind6 = [0,0,0, opt.wind]`, `lwind = transformSpatial(wind6, flg_force=0, xipos, subtree_com[root], ximat)` → `lwind_lin = R^T wind` (angular part 0 so no translation term). `lvel_lin -= lwind_lin`.

Forces in local frame (`lfrc=[torque; force]`, initialized 0), ρ = density, β = viscosity:
```
if β > 0:
   diam = (box0+box1+box2)/3
   lfrc[0:3]  = -π diam^3 β ω            # computed as lvel * (-mjPI*diam*diam*diam*β)
   lfrc[3:6]  = -3π diam β v
if ρ > 0:
   lfrc[3] -= 0.5 ρ box1 box2 |v0| v0
   lfrc[4] -= 0.5 ρ box0 box2 |v1| v1
   lfrc[5] -= 0.5 ρ box0 box1 |v2| v2
   lfrc[0] -= ρ box0 (box1^4 + box2^4) |ω0| ω0 / 64
   lfrc[1] -= ρ box1 (box0^4 + box2^4) |ω1| ω1 / 64
   lfrc[2] -= ρ box2 (box0^4 + box1^4) |ω2| ω2 / 64
```
(`box^4` is written as `b*b*b*b`.) In half-size notation this is f_D,i = −2ρ r_j r_k |v_i| v_i, g_D,i = −½ρ r_i (r_j⁴+r_k⁴)|ω_i|ω_i, f_V = −6πβ r_eq v, g_V = −8πβ r_eq³ ω, r_eq = mean(r).

Rotate: `bfrc_torque = R lfrc[0:3]`, `bfrc_force = R lfrc[3:6]`; `mj_applyFT(force, torque, xipos[i], i, qfrc_fluid)`.

### 5.2 Ellipsoid model — `mj_ellipsoidFluidModel(m,d,bodyid)`

For each geom `g` of the body:
1. `semiaxes = mju_geomSemiAxes(geom_size, type)`: sphere `(r,r,r)`; capsule `(r,r,h+r)`; cylinder `(r,r,h)`; otherwise `size[0..2]` (ellipsoid, box, mesh, sdf…).
2. Unpack `geom_fluid` (`readFluidGeomInteraction`; `mjERROR` if count ≠ 12). If interaction coef `== 0.0` skip.
3. `lvel = mj_objectVelocity(mjOBJ_GEOM, g, flg_local=1)`: velocity of the geom center in the **geom frame** (`geom_xmat`), linear part `R_gᵀ(v_root − (geom_xpos − subtree_com[root]) × ω)`.
4. `lwind = R_gᵀ wind` (transformSpatial with geom_xpos as origin); `v -= lwind`.
5. `lfrc = 0`; `mj_addedMassForces(lvel, NULL, ρ, virtual_mass, virtual_inertia, lfrc)`; `mj_viscousForces(lvel, ρ, β, semiaxes, C_M, C_K, C_blunt, C_slender, C_ang, lfrc)`.
6. `lfrc *= coef` (all 6, `mju_scl`).
7. Rotate with `geom_xmat` and `mj_applyFT(force, torque, geom_xpos[g], bodyid, qfrc_fluid)` — applied at the **geom center**.

#### `mj_addedMassForces(v6, accel=NULL, ρ, m_A, I_A, f6)`
```
v = v6[3:6], ω = v6[0:3]
p_lin = ρ m_A ∘ v,  p_ang = ρ I_A ∘ ω
(if accel: f[0:3] -= ρ I_A∘α, f[3:6] -= ρ m_A∘a   -- never used in the forward pass)
f[0:3] += p_lin × v
f[0:3] += p_ang × ω
f[3:6] += p_lin × ω
```
(Order: torque1 = p_lin×v added first, then torque2 = p_ang×ω.) The acceleration (true added-mass inertia) term is disabled because it would depend on qacc.

#### `mj_viscousForces`
Let `size = semiaxes`.
```
volume = 4/3 π s0 s1 s2
d_max = max(s), d_min = min(s), d_mid = s0+s1+s2-d_max-d_min
A_max = π d_max d_mid
magnus = (ω × v) * (C_M ρ volume)
speed = |v|;  inv_speed = speed > mjMINVAL ? 1/speed : 0;  vhat = v*inv_speed
inv_dmax = d_max > mjMINVAL ? 1/d_max : 0;  sn = size*inv_dmax          # normalized semi-axes
a_raw = (sn1 sn2)^2, b_raw = (sn2 sn0)^2, c_raw = (sn0 sn1)^2
proj_num = a_raw vhat0^2 + b_raw vhat1^2 + c_raw vhat2^2
inv_proj_num = proj_num > mjMINVAL ? 1/proj_num : 0
a = a_raw*inv_proj_num, b = b_raw*inv_proj_num, c = c_raw*inv_proj_num
proj_denom = a^2 vhat0^2 + b^2 vhat1^2 + c^2 vhat2^2
A_proj = π d_max^2 sqrt(proj_num * proj_denom)
norm = (a vhat0, b vhat1, c vhat2)                                      # non-unit normal
cos_alpha = proj_denom > mjMINVAL ? 1/proj_denom : 0
kutta_circ = (norm × v) * (C_K ρ cos_alpha A_proj)
kutta = kutta_circ × v
eq_D = 2/3 (s0+s1+s2)
lin_visc_force_coef = 3π eq_D;  lin_visc_torq_coef = π eq_D^3
I_max = 8/15 π d_mid d_max^4
II[i] = 8/15 π s_i max(s_{i+1}, s_{i+2})^4                              # mji_ellipsoid_max_moment
mom_visc[i] = ω_i (C_ang II[i] + C_slender (I_max - II[i]))
drag_lin = β lin_visc_force_coef + ρ speed (A_proj C_blunt + C_slender (A_max - A_proj))
drag_ang = β lin_visc_torq_coef + ρ |mom_visc|
f[0:3] -= drag_ang * ω
f[3:6] += magnus + kutta - drag_lin * v
```
`A_proj` equals the lemma `π sqrt((r_y⁴r_z⁴u_x² + …)/(r_y²r_z²u_x² + …))` (fluid.rst). The order `magnus + kutta - drag_lin*v` per component is the exact FP expression.

Mathematical summary (fluid.rst): f = f_A + f_D + f_M + f_K + f_V, g = g_A + g_D + g_V with
f_M = C_M ρ V ω×v; f_K = C_K ρ A_proj (v̂·n̂)(n̂×v)×v; f_D = −ρ[C_blunt A_proj + C_slender(A_max−A_proj)]|v|v; g_D = −ρ ω |[C_ang I_D + C_slender(I_max − I_D)]∘ω|; f_V = −6πβ r_D v, g_V = −8πβ r_D³ ω.

### 5.3 Compile-time virtual mass/inertia — `setFluid` (engine_setconst.c)

For each geom with `fluid[0] > 0`, semi-axes (dx,dy,dz) via `mju_geomSemiAxes`, `V = 4/3 π dx dy dz`:

`addedMassKappa(dx,dy,dz)` = κ_x = ∫₀^∞ dx dy dz / sqrt((dx²+λ)³(dy²+λ)(dz²+λ)) dλ evaluated by a fixed 15-point Gauss–Kronrod rule after the substitution λ = x³/(1−x)², with λ non-dimensionalized by `scale = (dx³ dy dz)^0.4`:
```
kappa = Σ_{i<15} scale * kd[i] / ((1+λ_i/dx²) sqrt((1+λ_i/dx²)(1+λ_i/dy²)(1+λ_i/dz²))) * w[i],  λ_i = scale*kl[i]
return kappa / dx²
```
Tables (exact):
```
w  = {0.01146766,0.03154605,0.05239501,0.07032663,0.08450236,0.09517529,0.10221647,0.10474107,
      0.10221647,0.09517529,0.08450236,0.07032663,0.05239501,0.03154605,0.01146766}
kl = {7.865151709349917e-08,1.7347976913907274e-05,0.0003548008144506193,0.002846636252924549,
      0.014094260903596077,0.053063261727396636,0.17041978741317773,0.5,1.4036301548686991,
      3.9353484827022642,11.644841677041734,39.53187807410903,177.5711362220801,
      1429.4772912937397,54087.416549217705}
kd = {5.538677720489877e-05,0.002080868285293228,0.016514126520723166,0.07261900344370877,
      0.23985243401862602,0.6868318249020725,1.8551129519182894,5.0,14.060031152313941,
      43.28941239611009,156.58546376397112,747.9826085305024,5827.4042950027115,
      116754.0197944512,25482945.327264845}
```
Computed in `double`. `kx = κ(dx,dy,dz)`, `ky = κ(dy,dz,dx)`, `kz = κ(dz,dx,dy)`.
```
yz = dy²-dz², zx = dz²-dx², xy = dx²-dy²
Ixfac = yz² |kz-ky| / max(1e-14, |2 yz + (dy²+dz²)(ky-kz)|)
Iyfac = zx² |kx-kz| / max(1e-14, |2 zx + (dz²+dx²)(kz-kx)|)
Izfac = xy² |ky-kx| / max(1e-14, |2 xy + (dx²+dy²)(kx-ky)|)
fluid[6] = V kx / max(1e-14, 2-kx);  fluid[7] = V ky/max(..,2-ky);  fluid[8] = V kz/max(..,2-kz)
fluid[9] = V Ixfac/5;  fluid[10] = V Iyfac/5;  fluid[11] = V Izfac/5
```
(Stored per unit density; density multiplies at runtime.) A sphere gives κ=2/3, m_A = V/2, I_A = 0.

---

## 6. Passive contact and adhesion (engine_passive.c)

### 6.1 `mj_contactPassive`
```
if DISABLED(CONTACT) || ncon == 0 || nv == 0: return 0
if !d->efm_active:
    if any flex f with mj_effFlexContactPossible(f): mjERROR("passive flex contact needs the effective metric ...")
    return 0
if !d->nefmcon: return 0
mj_effContactForce(d, d->qfrc_spring)     # engine_metric.c; adds contact repulsion rows built at the position stage
```
`mj_effFlexContactPossible(f) = flex_passive[f] && !flex_rigid[f] && flex_dim[f] >= 2 && !ENABLED(IPC)`. The force law itself is in the effective-metric spec (engine_metric.c); here it only rides into `qfrc_spring`.

### 6.2 `mj_adhesion`
```
if !flg_adhesion || DISABLED(CONTACT) || ncon == 0 || nv == 0: return 0
if no contact with (adhesion != 0 && exclude <= 1): return 0
for each such contact con:
    NV = mj_contactJacobian(m,d,con, dim=1, jacdif, ...)          # J2 - J1 translational, sparse chain or dense
    jac = con->frame[0:3] · jacdif                                 # 1 x NV normal Jacobian
    qfrc_adhesion += jac^T * (-con->adhesion)                      # dense: mju_addToScl; sparse: scatter by chain
```
Adhesion is a constant attraction along the contact normal (frame row 0 points from geom1 to geom2).

---

## 7. Flex elasticity

All flex vertex/node forces are computed in world Cartesian coordinates and mapped to generalized forces with point Jacobians.

### 7.0 Gather / scatter (`flexMap`, engine_core_util.c)

`mj_flexGather(f, res, vec)` computes `res[3v] = J_v vec` (world velocity of vertex v for vec=qvel); `mj_flexScatter(f, res, vec, scale)` computes `res += scale J_vᵀ vec_v`. For vertex v: `body = body_weldid[flex_vertbodyid[vadr+v]]`; skip if `body_dofnum[body]==0`.
- If `body_simple[body] == 2` (three independent ordered world-aligned slide joints, see `mj_flexBodySimple`): columns are `cdof[da+j][3:6]` directly.
- Otherwise walk dof ancestors `i = da+dofnum-1; i >= 0; i = dof_parentid[i]`: column = `cdof[i][0:3] × (flexvert_xpos[v] − subtree_com[rootid]) + cdof[i][3:6]`.
Gather overwrites; scatter accumulates.

### 7.1 `mj_flexPassiveStretch` — StVK / SNH for standard 2D triangles and 3D tetrahedra

Gating: `stiffnessadr = flex_stiffnessadr[f]`; return if `< 0` or `k[0]==0`. `kD = (enbl_damper && timestep > 0) ? flex_damping[f]/timestep : 0`. Return if `!enbl_spring && !kD`.

Data: `dim`, `nedge = dim==2 ? 3 : 6`, `snh = dim==3 && k[21] != 0`, record stride 21 (2D) or 24 (3D). Local edge orderings (`mj_stretchEdges`): 2D `{1,2},{2,0},{0,1}`; 3D `{0,1},{1,2},{2,0},{2,3},{0,3},{1,3}`. `flex_elemedge` maps element-local edges to flex edges; `deformed = flexedge_length`, `reference = flexedge_length0`, `vel = flexedge_velocity`, `plastic = flex_plasticadr>=0 ? d->plastic + adr : NULL` (`nedge` numbers per element first).

Per element t (vertices `vert = flex_elem + elemdataadr + (dim+1)t`):
```
edgevec[e] = x[vert[edge[e][0]]] - x[vert[edge[e][1]]]              # flexvert_xpos
s[e] = L_e^2 - L0_e^2 - (plastic ? plastic[nedge*t+e] : 0)          # "elongation"
metric = unpack upper-tri packed K (21 numbers for 6 edges, 6 for 3 edges) from k + stride*t
tension = metric * s                                                # = 2 dE/ds for E = s'Ks/4
if snh:
    mj_snhCubic(tension, s, k[21])                                  # tension += 2γ ∂P/∂s
    J = mj_snhVolume(grad, edgevec, packed); pressure = 2*packed[22]*(J-1)
if enbl_spring:
    for e, endpoints (i0,i1): f[i0] -= tension[e]*edgevec[e]; f[i1] += tension[e]*edgevec[e]
    if snh: f[vert[v]] -= pressure * grad[v]   (v=0..3)
if snh: continue                                                     # no per-element damping for SNH
if kD:                                                               # generalized Rayleigh (Kharevych et al.)
    for e: dL = vel[e]*timestep;  s_d[e] = dL*(2*L_e - dL) * kD       # = (L^2 - Lprev^2) kD, no cancellation
    tension_d = metric * s_d;  dmp -= tension_d ⊗ edgevec (same stencil)
```
After the loop:
- If `snh && kD`: `worldvel = flexGather(qvel)`; `mj_flexHessian(f)` (cached, PSD-projected); `dmp += -flex_damping[f] * H * worldvel` (`mj_flexHessianMul`). Note: SNH damping uses `flex_damping` directly (not ÷ timestep), times the PSD tangent → non-positive power.
- `if enbl_spring: flexScatter(qfrc_spring, f, 1)`; `if kD: flexScatter(qfrc_damper, dmp, 1)`.

Why: E = ¼ sᵀKs with s_e = L_e² − L0_e²; ∂s_e/∂x_{i0} = 2 edgevec_e, so F_{i0} = −Σ_e (Ks)_e edgevec_e.

**Compile-time K (StVK)** — `ComputeStiffness<T>` / `MetricTensor` / `ComputeBasis` (user_flexcomp.cc):
- `volume` = signed triangle area (2D) or tet volume (3D); `measure = |volume| * (2D ? thickness : 1)`.
- `mu = E/(2(1+ν))`; `lambda = 2D ? Eν/(1−ν²)` (plane stress) `: Eν/((1+ν)(1−2ν))`.
- `mu_k = mu*measure`, `la_k = lambda/2*measure` (factor ½ because Σ_e s_e B_e = 2·GreenStrain).
- Edge basis B_e (3×3): 2D: with the two faces (edges) opposite… `normal = normalize(edgesR × edgesL)`, `basisL = normal × edgesL`, `basisR = edgesR × normal`, `B_e = (basisL basisRᵀ + basisR basisLᵀ)/(8 volume²)`. 3D: area normals `nL = (xL1−xL0)×(xL2−xL0)`, `nR` likewise for the two faces adjacent to the edge's complement, `B_e = (nL nRᵀ + nR nLᵀ)/(72 volume²)`. Face tables: `Stencil3D::face = {{2,1,0},{0,1,3},{1,2,3},{2,0,3}}`, `edge2face = {{2,3},{1,3},{2,1},{1,0},{0,2},{0,3}}`; `Stencil2D::face = edge2face = {{1,2},{2,0},{0,1}}`.
- `K_ef = mu_k tr(B_e B_f) + la_k tr(B_e) tr(B_f)`, stored upper-triangular row-major into stride 21 (2D, only first 6 used) or 24 (3D, `[21..23]=0`).
- `elastic2d`: 0 none, 1 bending only, 2 stretch only, 3 both (stretch requires `thickness > 0`). If `stressyield > 0`, 2D stores `stiffness[6] = thickness` (slot 6 of element 0's unused space).

**Stable Neo-Hookean (`elastic3d=1`)** — `ComputeSNH`: `K` via `MetricTensor(mu V0, -mu V0)` (i.e. `K_ef = μV0 [tr(B_eB_f) − tr(B_e)tr(B_f)]`), `k[21] = γ = −μ/(72 V0)`, `k[22] = β = V0(λ+2μ)/2`, `k[23] = 1/(6·volume_signed) = 1/det(Dm)`. Energy: `sᵀKs/4 + γ P(s) + β (J−1)²`, equal to `V0[μ/2(tr FᵀF − 3) − μ(J−1) + (λ+μ)/2 (J−1)²]`. The nonzero `k[21]` is the SNH marker.

`mj_snhCubic(tension, s, γ)`: Gram-difference at vertex 0: `a=s0, b=s2, c=s4, d=½(s0+s2−s1), e=½(s0+s4−s5), f=½(s2+s4−s3)`; `P = abc + 2def − af² − be² − cd²`. Cofactors `cof = {bc−f², ac−e², ab−d², ef−cd, df−be, de−af}`; `g = {cof0+cof3+cof4, −cof3, cof1+cof3+cof5, −cof5, cof2+cof4+cof5, −cof4}`; `tension += 2γ g`.

`mj_snhVolume(grad, edgevec, k)`: `a = −edgevec[0] = x1−x0`, `b = edgevec[2] = x2−x0`, `c = −edgevec[4] = x3−x0`; `grad1 = k23 (b×c)`, `grad2 = k23 (c×a)`, `grad3 = k23 (a×b)`, `grad0 = −(grad1+grad2+grad3)`; returns `J = k23 · a·(b×c)` (computed before scaling grads, using unscaled `b×c`).

**Cached stretch Hessian** — `mj_flexHessian(f)` (lazy, `d->flex_hessian_valid[f]`): per element, StVK: `mj_stretchStiffness` (tension clamped `max(tension,0)` — compressed edges' negative geometric term dropped) then per edge (i,j) block `mj_stretchStiffnessBlock`: `Σ_{a,b} 2 K_ab s_a s_b edgevec_a edgevec_bᵀ + (Σ_a tension_a s_a^i s_a^j) I`, with `s_a^i = +1/−1/0` if i is first/second/not endpoint of edge a. SNH: `snhProject` (signed SVD of F, stretch block A in principal frame with volume penalty, PSD projection by `snhPositive3` Cholesky test else `snhEigen3` + clamp, off-diagonal symmetric/skew modes clamped ≥0, Smith et al. "Stable Neo-Hookean", Sec.4) and `snhProjectedBlock`. Off-diagonal blocks stored per flex edge (`flexedge_hessian`, 9, oriented first→second endpoint); diagonal blocks per vertex (`flexvert_hessian`, 6 packed `00,01,02,11,12,22`) by translation invariance `H_ii = −Σ_{j≠i} H_ij`. `mj_flexHessianMul(res += scale H vec)`. The full SNH projection math (SVD/eigen routines) is shared with the solver/metric spec; tolerances in §1.

### 7.2 `mj_flexPassiveBend` — 2D discrete shell bending (non-interpolated)

Gating: `flex_dim==2`, `flex_bendingadr >= 0`; `damp = enbl_damper ? flex_damping : 0`; `enbl_damper = damp != 0`; return if neither.
Bending record per flex edge e: 17 numbers `b[17e + 4i + j]` (4×4 matrix over stencil vertices `v = {edge0, edge1, flap0, flap1}`) and `b[17e+16]` (curved-reference coefficient). Plastic: `bend_plastic = d->plastic + plasticadr + stretch_pnum`, with `stretch_pnum = 3*elemnum` if stretch stiffness present (`flex_stiffness[adr] != 0`) else 0.
```
if enbl_damper: vvel = flexGather(qvel)
for each edge e with flap[1] != -1:                      # boundary edges skipped
   ed0 = x[v1]-x[v0]; ed1 = x[v2]-x[v0]; ed2 = x[v3]-x[v0]
   frc1 = ed1×ed2; frc2 = ed2×ed0; frc3 = ed0×ed1; frc0 = -(frc1+frc2+frc3)
   b16 = b[17e+16] + (bend_plastic ? bend_plastic[e] : 0)
   for i<4, x<3:
      for j<4: spring[v_i,x] += b[17e+4i+j]*xpos[v_j,x];  damper[v_i,x] += b[17e+4i+j]*vvel[v_j,x]
      spring[v_i,x] += b16 * frc_i[x]
flexScatter(qfrc_spring, spring, -1);  flexScatter(qfrc_damper, damper, -damp)
```
Compile time (`ComputeBending`, Wardetzky et al. cotangent operator + Garg et al. cubic shells), rigidity `D = E t³/(12(1−ν²))`:
```
a01 = cot(v0,v1,v2), a02 = cot(v0,v3,v1), a03 = cot(v1,v2,v0), a04 = cot(v1,v0,v3)
c = {a03+a04, a01+a02, -(a01+a03), -(a02+a04)}
area = A(v0,v1,v2) + A(v1,v0,v3);  stiffness = D/area
t0 = -(a03 e1 + a01 e3), t1 = -(a04 e2 + a02 e4)    (e0=v1-v0,e1=v2-v0,e2=v3-v0,e3=v2-v1,e4=v3-v1)
cos_theta = -(t0·t1)/|e0|^2
B[4i+j] += c_i c_j cos_theta stiffness
B[16] = ((e0×e1)·e2) (a01-a03)(a04-a02) stiffness / |e0|^3
```
Energy is quadratic in positions (`½ xᵀ B x` per edge) plus the curved-reference term (`b16 · det(ed0,ed1,ed2)` whose gradient is `frc`); forces are `−Bx − b16 frc`.

### 7.3 `mj_flexPassiveBend1D` — discrete elastic rod bending (no twist)

Gating: `bendingadr >= 0`; `damp = enbl_damper ? flex_damping : 0`; `enbl_damper = damp != 0`; return if neither. `b[0] = nhinge`; record h at `b + 1 + 10h`. `vvel = flexGather(qvel)` if damper.

Record layouts (`ComputeRodBending`): single hinge `[v_prev, v, v_next, −1, k, 0, kb0, 0,0,0]`, `kb0 = |kb|` of reference if bent by more than `kRodStraightTol = 1e-4` else 0; hinge pair (both hinges curved, across shared edge (v1,v2)) `[v0,v1,v2,v3, k1/n1, k2/n2, om0[4]]` with `n` = number of pairs per hinge and `om0` the reference binormals projected on `RodEdgeFrame(v1,v2)` axes. `k = EI/(l0+l1)`; `I = πr⁴/4` (circle) or `min(w h³, h w³)/12` with `w,h = 2·sectionsize` (box); vertices with valence ≠ 2 carry no hinge.

`mj_rodHinge(xa, xv, xb)`: `e0 = xv−xa, e1 = xb−xv, l0,l1` norms, `denom = l0 l1 + e0·e1`; fail if `l0<mjMINVAL || l1<mjMINVAL || denom<mjMINVAL`. `kb = 2 (e0×e1)/denom`. Jacobian: `g0 = (l1/l0)e0 + e1`, `g1 = (l0/l1)e1 + e0`, `A0 = (−2[e1]× − kb g0ᵀ)/denom`, `A1 = (2[e0]× − kb g1ᵀ)/denom` (the code writes `[e1]×`-like matrices then `(2A − kb gᵀ)/denom`), `J = [−A0, A0−A1, A1]` (3×9).

`mj_rodHingeWeights(g, Wk, Wd, kb, k, kb0)`, `r = |kb|`, `n = kb/r` if `r ≥ mjMINVAL` else 0:
- straight (`kb0==0`): `g = 2k kb`; `Wk = 2k I`; `Wd = r<mjMINVAL ? 2k I : 2k n nᵀ`.
- curved: `e² = 1e-8`, `γ = sqrt(1+e²)`, `c = r/kb0`, `ρ² = c²+e²`, `ρ = sqrt(ρ²)`, `s = 1 − γ/ρ`; `g = 2k s kb`; `tang = max(0,s)`, `rad = max(0, 1 − γe²/(ρρ²))`; `Wk = r<mjMINVAL ? 0 : 2k(tang(I−nnᵀ) + rad nnᵀ)`; `Wd = r<mjMINVAL ? 2kI : 2k nnᵀ`. Energy `k (ρ−γ)² kb0²`.

`mj_rodPair(x0..x3)`: two hinges `(x0,x1,x2)`, `(x1,x2,x3)` and `rodEdgeFrame(x1,x2)` (`frame[0:3] = normalize(x2−x1)` fail if `< mjMINVAL`, then `mju_makeFrame` → u = frame[3:6], v = frame[6:9]); `om = (kba·u, kba·v, kbb·u, kbb·v)`; J (4×12) rows `uᵀJa, vᵀJa` on columns x0..x2 and `uᵀJb, vᵀJb` on x1..x3 (frame held fixed).

`mj_rodPairWeights(g, Wk, Wd, om, w0, w1, om0)`: `w = (w0,w0,w1,w1)`, `wmax = max(w0,w1)`; if `wmax ≤ 0` all zero. `u0 = w0/wmax`, `u1 = w1/wmax`.
- Damping: `t = (−om1, om0, −om3, om2)`, `ut = (u0t0,u0t1,u1t2,u1t3)`, `tnorm = sqrt(t·ut)`; `ut /= tnorm` if `tnorm ≥ mjMINVAL`; `scl = tnorm<mjMINVAL ? 0 : 2 wmax`; `Wd = 2diag(w) − scl ut utᵀ`.
- `r = u0(om0_0²+om0_1²) + u1(om0_2²+om0_3²)`; if `r ≤ 0` (straight): `g = 2 w∘om`, `Wk = 2diag(w)`.
- Else `c = (u0 om0_0, u0 om0_1, u1 om0_2, u1 om0_3)`, `s = (−u0 om0_1, u0 om0_0, −u1 om0_3, u1 om0_2)`, `cs = c·om/r`, `sn = s·om/r`, `ρ² = cs²+sn²+e²`, `g = 2w∘om − (2γ wmax/ρ)(cs c + sn s)`, `q = (cs s − sn c)/ρ`, `Wk = 2diag(w) − (2 wmax min(1,γ/ρ)/r)(q qᵀ + (e²/ρ²)(c cᵀ + s sᵀ))`.

Forces per hinge record (nvrt = 3 or 4, skip if hinge/pair degenerate):
```
w = g (spring),  wd = Wd · (J · vvel_stencil)  (damper)
frc[v_i] -= J[:, i]^T w;  dmp[v_i] -= J[:, i]^T wd
flexScatter(qfrc_spring, frc, 1);  flexScatter(qfrc_damper, dmp, damp)
```

### 7.4 `mj_flexPassiveInterp` — corotational linear FE for interpolated flexes

Gating: return if `flex_edgeequality[f]==3` (strain equality reuses the stiffness block for eigenmodes) or `!mj_flexHasStiffness(f)`. `order = |flex_interp|`, shell mode if `flex_interp < 0`. `cx,cy,cz = flex_cellnum`. Shell: `npe = (order+1)²`, `nelem = 2(cy cz + cx cz + cx cy)` (6 box faces); volume: `npe = (order+1)³`, `nelem = cx cy cz`. Stiffness block per element: dense `3npe × 3npe` row-major at `flex_stiffness + stiffnessadr + fe·(3npe)²`, stored with **negative sign** (`K_stored = −K_phys`), so `K_stored·u` is directly the restoring force.

```
xpos_g, vel_g = mju_flexGatherState(f)            # node world pos (xpos[bid] if centered or flex_node==0, else xpos+xmat*node)
                                                  # vel = v_com(world, mj_objectVelocity flg_local=0) + ω × (x − xipos)
                                                  # shell mode: interior nodes rebuilt by transfinite interpolation (mju_shellTrackInterior) for x and v
for fe in 0..nelem-1:
   K = block(fe); if !shell && K[0]==0: continue  # empty cell
   gather element nodes (Cell: lexicographic li,lj,lk over (order+1)^3; Face: see below) and quat = R^{-1}
   x_e, v_e rotated by quat (to corotational frame)
   u_e = x_e − x0_e   (x0 = flex_node0, rest node positions)
   if plastic (flex_plasticadr>=0):
      shell: normal_axis = fe<2cycz ? 0 : (fe<2(cycz+cxcz) ? 1 : 2); a0=(n+1)%3, a1=(n+2)%3; sp = plastic + 3fe
             u[a0] -= sp0 dx0[a0] + ½sp2 dx0[a1];  u[a1] -= ½sp2 dx0[a0] + sp1 dx0[a1]       (dx0 = x0_n − x0_0)
      volume: sp = plastic + 6fe (Voigt xx,yy,zz,xy,xz,yz with half off-diagonals): u -= Sp · dx0
   f_e = K u_e (if spring);  d_e = K v_e (if damper)       # mju_mulMatVec
   rotate both back by conj(quat) (= R) and accumulate per global node
for each node i: dmp_g[i] *= flex_damping[f]               # damping: K scaled by flex_damping, NOT /timestep
   bid = flex_nodebodyid[i]
   if body_dofnum[bid] > 0 && (flex_centered || flex_node[i]==0):
        qfrc_spring[body_dofadr[bid]:+3] += xmat[bid]^T f      # fast path: body has 3 slide dofs at node
        qfrc_damper[...]                 += xmat[bid]^T d
   else: mj_applyFT(f, 0, xpos_g[i], bid, qfrc_spring/damper)
```
Note the fast path assumes the node body's first 3 dofs are local-frame translations (flexcomp interpolated nodes are bodies with 3 slide joints).

Corotational rotation: `mju_flexGatherCellState` (volume): `quat = flexInterpRotation(order, x_c, dref, p=(.5,.5,.5))` where `dref = x0[gidx1] − x0[gidx0]` (cell diagonal extents in rest; `dref=1` fallback), F = deformation gradient at cell center (`mju_defGradient`), columns divided by `dref[col]` when `> mjMINVAL`; `mju_flexPolarRotation(quat, F)` then `quat = conj(quat)`.

`mju_flexPolarRotation(quat, mat)`: `quat=(1,0,0,0)`; if `tr(mat) ≤ 0` and `|c0| > mjMINVAL` (normalize c0) and Gram–Schmidt `c1 ⊥ c0` with `|c1| > mjMINVAL`: warm-start `quat = mju_mat2Quat([c0, c1, c0×c1])`. Then `mju_mat2Rot(quat, mat)` (Müller et al. 2016): iterate ≤500: `R = quat2Mat(quat)`, `ω = Σ_k R_k × M_k / (|Σ_k R_k·M_k| + mjMINVAL)`, `w = |ω|` (normalize), stop if `w < 1e-9`, `quat = axisAngle(ω̂, w) * quat`, `normalize4`.

Face gather (`mju_flexGatherFaceState`): faces ordered x=0, x=max, y=0, y=max, z=0, z=max with sizes `{cycz, cycz, cxcz, cxcz, cxcy, cxcy}`, `normal = {0,0,1,1,2,2}`, `count1 = {cz,cz,cx,cx,cy,cy}`, fixed index `{0, cx·order, 0, cy·order, 0, cz·order}`; in-plane `na0=(n+1)%3` (slow), `na1=(n+2)%3` (fast); `q0 = within/count1`, `q1 = within%count1`; nodes `g[n]=fixed, g[na0]=q0·order+l0, g[na1]=q1·order+l1`, global index `g0·ny·nz + g1·nz + g2` (`ny = cy·order+1`, `nz = cz·order+1`). Rotation: with rest data, `t1 = Σ x dphi(.5)phi(.5)`, `t2 = Σ x phi(.5)dphi(.5)`, divided by rest extents `d0,d1` (if `> mjMINVAL`), `normal = t1×t2`, matrix columns `[na0]=t1, [na1]=t2, [n]=normal`, polar rotation, conjugate.

Shape functions: order 1: `φ0 = 1−s, φ1 = s`, `φ' = ∓1`; order 2: `φ0 = 2s²−3s+1, φ1 = 4(s−s²), φ2 = 2s²−s`, `φ0' = 4s−3, φ1' = 4(1−2s), φ2' = 4s−1`.

Compile-time K (user_flexcomp.cc): volume cells: Gauss–Legendre `order+1` points per axis, `K -= [λ tr(∇u)tr(∇v) + 2μ sym(∇u):sym(∇v)] dV` with the 3D Lamé λ (`ComputeLinearStiffness`; elastic2d face variant `ComputeLinearStiffness2D` uses plane-stress `λ* = Eν/(1−ν²)`, `dvol = w_s w_t d0 d1 thickness`, and only in-plane displacement components to avoid spurious transverse shear). Shell faces may additionally get a warp (twist) mode with stiffness `D(1−ν)·4/(d0 d1)` (`ComputeWarpStiffness`, `D = Et³/(12(1−ν²))`). Strain-equality flexes store eigenmodes (`EigendecomposeStiffness`) instead — hence the `edgeequality==3` skip.

### 7.5 `mj_flexPassiveBendInterp` — interpolated shell bending

Only `flex_interp < 0` with `flex_bendingadr >= 0`. `bdata[0] = nedge`; edge record (10): `[fe_A, fe_B, local_A(2), local_B(2), stiffness, dn0(3)]`. Compile time (`ComputeInterpBending`): edges between adjacent face elements (intra-surface and box-corner edges); `local_*` = parametric midpoint of the shared edge on each face; `dn0 = n̂_A − n̂_B` at rest; `stiffness = D l_avg / max(h_avg, 1e-12)` with `D = Et³/(12(1−ν²))` (discrete Crouzeix–Raviart energy `E = D/(2h)|Δn − Δn0|² l`).

Runtime:
```
cache per face: x_face, gidx, quat (=R^{-1}, from mju_flexGatherFaceState with rest data)
mjERROR if any cached node index out of [0, nodenum)
if enbl_damper && flex_damping/timestep > 0: mju_warning("Bending damping is not yet supported ...")   # every call!
for each edge e (skip stiffness == 0):
   (n_A, t1_A, t2_A) = mju_flexFaceNormal2D(order, x_A, local_A)   # t1=Σx dφ⊗φ, t2=Σx φ⊗dφ, n=t1×t2 (unnormalized)
   same for B; len_A=|n_A|, len_B=|n_B|; skip if either < mjMINVAL; normalize
   if quat_A·quat_B < 0: quat_B = -quat_B
   quat_avg = normalize(quat_A + quat_B); quat_avg = conj(quat_avg)          # R_avg
   r = (n̂_A − n̂_B) − R_avg dn0
   if spring:
      w_A = (r − n̂_A (n̂_A·r))/len_A ;  w_B = (r − n̂_B(n̂_B·r))/len_B
      for node k of A (l0,l1): g0 = dφ(lA0)φ(lA1), g1 = φ(lA0)dφ(lA1):  f[k] += stiffness (g0 (w_A×t2_A) − g1 (w_A×t1_A))
      for node k of B: f[k] -= stiffness (g0 (w_B×t2_B) − g1 (w_B×t1_B))
   (no damping; see TODO in source)
apply to bodies exactly as in 7.4 (fast path or mj_applyFT), dmp_g stays zero
```

---

## 8. Cable plugin (`plugin/elasticity/cable.cc`, `mujoco.elasticity.cable`)

Capability `mjPLUGIN_PASSIVE`; attributes `twist` (G), `bend` (E), `flat`, `vmax`; `nstate = 0`. Applied to a **contiguous** set of bodies (`body_plugin[i]==instance`), else init fails (`-1`, warning). `twist` and `bend` must parse as numbers (`CheckAttr`).

Init (runs `mj_kinematics` at `qpos0` with zeroed mocap quats):
- `prev[b] = (b==0) ? 0 : −1`, `next[b] = (b==n−1) ? 0 : +1`.
- `omega0[b] = (prev[b] && flat != "true") ? mju_subQuat(body_quat[i], qpos[qadr:qadr+4]) : 0`, with `qadr = jnt_qposadr[body_jntadr[i]] + body_dofnum[i] − 3` (the quaternion of the body's ball joint, or of a free joint's rotation).
- Section from the body's first geom: cylinder/capsule radius r: `J = πr⁴/2`, `Iy = Iz = πr⁴/4`; box (h = size[1], w = size[2], a=max, b=min): `J = a b³ (16/3 − 3.36 (b/a)(1 − b⁴/(12a⁴)))`, `Iy = (2w)³(2h)/12`, `Iz = (2h)³(2w)/12`; other types → 0.
- `stiffness[4b] = {J G, Iy E, Iz E, L}` with `L = |xpos[i] − xpos[i−1]|` if `prev[b]` else 0.

Compute (each body b, skipped if first three stiffnesses are zero; `mju_error` if body not associated):
```
lfrc = 0
if prev[b]:
   quat = body_quat[i] * qpos_joint[i]                 # QuatDiff, relative orientation of i in parent frame
   ω = mju_quat2Vel(quat, 1)
   tmp = -(k_x(ω_x-ω0_x), k_y(ω_y-ω0_y), k_z(ω_z-ω0_z)) / L_b
   stress[b] = rotate(tmp, conj(quat))                 # pulled back into body i's frame
   lfrc += stress[b]
if next[b]:
   quat = body_quat[i+1] * qpos_joint[i+1]
   stress[b+1] = -(k(ω-ω0))/L_{b+1}   (no pullback; parameters of body b+1)
   lfrc -= stress[b+1]
xfrc = rotate(lfrc, xquat[i]);  mj_applyFT(force=NULL, torque=xfrc, xpos[i], i, d->qfrc_passive)
```
Note: torque-only; written straight into `qfrc_passive` after composition (so not in `qfrc_spring`); the division by `L` uses `stiffness[3]` of the hinge's child body; for b=0 `L=0` but `prev[0]=0` so it's never used as divisor except via `next` of b−1 (fine). Visualization colors geoms by mean stress norm (jet colormap, `vmax`).

---

## 9. GPU (MuJoCo Warp) implementation — `passive.py`

Precision: Warp `float` is **float32** by default; all arrays `[nworld, ...]`. Model arrays that may be batched per world use `arr[worldid % arr.shape[0], ...]` (e.g. `jnt_stiffness`, `dof_damping`, `body_mass`, `geom_size`, `opt.density`, `opt.gravity`, `opt.timestep`). Atomics (`wp.atomic_add`) are used for nearly all accumulations → **non-deterministic summation order** unless `m.opt.deterministic & DeterminismType.ATOMICS`, in which case kernels are compiled with `wp.DeterministicMode.RUN_TO_RUN` (`deterministic_max_records` = nv, 2nv, 4nv, `FLEX_INTERP_MAX_NODES=27`, `EFM_CON_MAX_DOF=24` depending on kernel).

### 9.1 `passive(m, d)` launch sequence

```
if dsbl_spring and dsbl_damper: zero qfrc_spring, qfrc_damper, qfrc_gravcomp, qfrc_fluid, qfrc_passive; return
                                                       # (qfrc_adhesion NOT zeroed here)
1. _spring_damper_dof_passive            dim (nworld, njnt)        writes (not adds) spring & damper for all dofs of each joint
2. _spring_damper_tendon_passive         dim (nworld, ntendon, max_ten_J_rownnz)   atomic adds
3. _spring_damper_flexedge_passive       dim (nworld, nflexedge)   atomic adds
4. _flex_rod_bending                     dim (nworld, nflexrodhinge)  atomic adds directly into qfrc_spring/damper
5. if nflex: flex_spring_body_force, flex_damper_body_force = zeros (nworld, nbody) spatial_vector
6. _flex_elasticity                      dim (nworld, nflexelem)   -> body spatial forces (atomic)
7. _flex_bending                         dim (nworld, nflexedge)   -> body spatial forces
8. _flex_passive_bend_interp             dim (nworld, nflexbend_interp) -> spring body forces
9. qfrc_gravcomp.zero_(); if gravity enabled: _gravity_force dim (nworld, nbody-1, nv) atomic
10. if nflexintcell: _flex_passive_interp dim (nworld, nflexintcell) -> body forces (+ scratch [nworld,ncell,27] vec3)
11. if nflex: apply_ft(flex_spring_body_force -> qfrc_spring, add) if !dsbl_spring; same for damper
    if has_flex_snh and !dsbl_damper: flex_hessian(m,d); flex_stretch_mul(qfrc_damper, qvel, s1=0, s2=-1, use_timestep=False, snh_only=True)
    if integrator == DISCRETE and has_flex_passive: build_efm_contact(rebuild=True); _eff_contact_force dim naconmax -> qfrc_spring
12. if has_fluid: _fluid (kernel _fluid_force dim (nworld, nbody) -> body wrench; apply_ft(... qfrc_fluid, flg_add=False))
13. qfrc_adhesion.zero_(); if flg_adhesion and contacts enabled and nv>0: _qfrc_adhesion dim naconmax
14. _qfrc_passive_kernel dim (nworld, nv): qfrc_passive = spring + damper (+ gravcomp if gravity enabled & !actgravcomp) (+ fluid if has_fluid) (+ adhesion if flg_adhesion)
15. if m.callback.passive: callback(m, d)
```

### 9.2 Kernel details and differences from C

**`_spring_damper_dof_passive`** (per world × joint): computes damping for every dof of the joint first (actuator damping via `util_misc.actuator_target_damping` using `jnt_actuatorid`, gear `actuator_gear[...][0]`, same −1/≥0/<−1 semantics; damper set to 0 when disabled or zero), then spring. Differences:
- `_poly_force` hard-codes `mjNPOLY=2`: `linear + p0·x + p1·x·x` (C: `xpow*=x` cumulative — same values, op order `p1*x*x` vs `p1*(x*x)`).
- FREE translational and rotational spring use `qpos_spring` only — **plastic reference is ignored for free joints** (ball/slide/hinge honor `jnt_plasticadr`).
- `wp.normalize(quat)` always divides (C `mju_normalize4` skips if `|norm−1| ≤ 1e-15` and maps tiny norms to identity).
- `math.quat_to_vel`: returns 0 if `|axis| == 0` exactly, else `axis·angle/|axis|`; C normalizes axis with mjMINVAL guard. Tiny-angle FP differs slightly.
- No sleep filtering (all joints/dofs computed).
- Writes spring for all joint dofs (zero if no stiffness) — this is what zeroes the arrays before tendon/flex atomics.

**`_spring_damper_tendon_passive`** (world × tendon × sparse entry): recomputes the tendon spring/damper per J entry, atomically adds `J*f`. Dead-band and plastic `[lower,upper]` honored. No sleep filtering.

**`_spring_damper_flexedge_passive`**: same law as C; skips rigid edges and `rownnz==0`; returns early if both forces exactly 0.

**`_gravity_force`** (world × body(≥1) × dof): `force = −gravity·mass·gravcomp`; `jac_dof(...)` translational Jacobian column at `xipos` (`body_isdofancestor` table), `atomic_add(qfrc_gravcomp[dof], jac·force)`. Doesn't check `flg_gravcomp` or `|gravity|==0` (results are zero anyway). Composition adds gravcomp only when gravity is enabled.

**`_fluid_force`** (world × body): body 0 and `mass < MJ_MINVAL` → zero wrench. Uses `body_fluid_ellipsoid[b]` precomputed in `io.py` (true if any geom of the body has `geom_fluid[0] > 0` and the body's mass ≥ MJ_MINVAL). Velocity from `cvel`: `lin_com = v − (xipos − subtree_com[root]) × ω`.
- Ellipsoid path: per geom with `coef > 0` (C: `!= 0`): `lin_point = lin_com + ω × (geom_xpos − xipos)`, local `l_ang = R_gᵀω`, `l_lin = R_gᵀ lin_point`, wind subtracted only if nonzero. Added mass only `if density > 0` (C always; identical result since ρ multiplies). **Kutta force only if `density > 0 and kutta_coef != 0 and lin_speed > MJ_MINVAL`** (C computes it unconditionally; for `speed ≤ 1e-15` C gives a ~0 but nonzero value). Formulae otherwise identical, including `cos_alpha`, `A_proj`, `II`, `I_max`. Geom wrench accumulated into a body wrench at `xipos`: `torque += R_g lfrc_t + (geom_xpos − xipos) × (R_g lfrc_f)` (C applies at `geom_xpos` via `mj_applyFT` — same mapping, different FP path).
- Inertia-box path: `scl = 6/mass; box_i = sqrt(max(MJ_MINVAL, …)·scl)` (C: `max(...)/mass*6`); viscous torque uses `wp.pow(diam, 3.0)`, quadratic torque uses `wp.pow(box, 4.0)` and `scl = density/64` multiplied last (C: products and `/64.0`). These reorderings plus float32 cause small differences.
- `m.has_fluid = wind.any() or density>0 or viscosity>0` (C: only density/viscosity — but with wind only all terms vanish). Wrench → `support.apply_ft(..., qfrc_fluid, flg_add=False)` (overwrites).
- `apply_ft` kernel (per world × dof): loops the contiguous subtree `[dofbody, dofbody+body_subtreenum)`; `qfrc[dof] = Σ_b cdof_ang·τ_b + cdof_lin·f_b + (cdof_ang × (xipos_b − subtree_com))·f_b`, skipping zero wrenches. No atomics (deterministic).

**`_qfrc_adhesion`** (per contact): requires geom contacts (`g ≥ 0`; flex-involved contacts are ignored since body = −1), walks body chain from `body_weldid[geom_bodyid]`, `jac_dof` at `contact.pos`, `+adhesion·n` on body1 and `−adhesion·n` on body2. Skips contacts with `adhesion == 0`; **does not check `exclude`** (C: `exclude <= 1`). Atomics.

**`_flex_elasticity`** (world × flex element): mirrors `mj_flexPassiveStretch` per element: gates on `stiffnessadr ≥ 0 && flex_stiffness[adr] != 0`, `kD = flex_damping/timestep` if `timestep > 0 && !dsbl_damper`; SNH detection `FLEX_STIFFNESS_3D == 24 && dim == 3 && k[adr+21] != 0` (`FLEX_STIFFNESS_3D = 24` for mujoco ≥ 3.14.1.dev…, else 21 → no SNH); if SNH and spring disabled: return. Returns early if vertex 0 has `bodyid < 0`. Forces converted to **body spatial wrenches at `xipos`** (`torque = (x_v − xipos) × f`, i.e. `-cross(xipos - x_v, f)`) and atomically accumulated per body, later mapped by `apply_ft` (C maps vertex points through `flexMap` — same Jᵀf for a point on that body). Note: no `flex_interp`/`flex_rigid` check in this kernel (rigid flexes are rejected by io; interpolated flexes do not carry 21/24-stride element records).
- SNH damping is done afterwards via the cached Hessian (`flex_hessian` + `flex_stretch_mul(..., snh_only=True)`), identical in form to C (`−flex_damping·Jᵀ H J qvel`). `has_flex_snh` requires `integrator == DISCRETE` (io raises `ValueError` otherwise).
- SNH Hessian routines (`_snh_svd`, `_snh_eigen3`, `_snh_positive3`, `_snh_jacobi_rot`) use the **single-precision tolerances** of C's `mjUSESINGLE` branch: Jacobi `2e-6`, rank `1e-6`, Cholesky `1e-5·scale`. A float64 C reference uses `4e-15`, `1e-12`, `1e-12·scale`. Sorting is by three explicit swaps (0,1),(0,2),(1,2) — same as C's selection loop; determinant sign check via `wp.determinant(V)` vs C's `(a×b)·c`.

**`_flex_rod_bending`** (world × rod hinge record, map `flex_rodhinge_map[(f, adr)]` built in io for non-rigid non-interp 1D flexes): `rod_stencil` (same math as §7.3; `math.orthogonals(edge/|edge|)` replaces `mju_makeFrame` for the pair frame — energy is frame-independent so forces agree up to FP), vertex velocities via `flex_gather_vert` (qvel), forces scattered with `flex_scatter_vert` (point Jacobian, `body_simple==2` fast path, atomics). `ROD_HINGE_SIZE==4` legacy layout (straight hinges `[v0,v1,v2,k]`) supported for older mujoco.

**`_flex_bending`** (world × flex edge): `dim==2`, `interp==0`, `bendingadr ≥ 0`, non-boundary; damping active only if `damping > 0.0` (C: `!= 0`). Velocities from `cvel`: `v = v_com + ω × (x − subtree_com[root])` for vertices whose weld body has dofs (C uses `flexGather(qvel)`, same quantity). Curved term only computed if `c16 != 0`. Forces as body wrenches (`frc_s = −(Bx + c16 frc_i)`, `frc_d = −Bv·damping`), skipped for vertices without dofs.

**`_flex_passive_interp`** (world × interpolated element, `flex_cell_map[(f, ci, cj, ck)]`; shell elements enumerate faces in `ci`): `nflexintcell` excludes flexes with `edgeequality==3` or no stiffness. Differences:
- **Quadratic interpolation is unsupported** (`io.py` raises `NotImplementedError` for `|flex_interp| == 2`); `support.flex_phi/flex_dphi` are linear-only.
- Rotation: volume cells via `support.compute_interp_cell_quat` (deformation gradient at center from current node positions, **without dividing columns by the rest extents `dref`**); shells use `d.face_quat` from `smooth._flex_face_kinematics` → `compute_interp_face_quat` (also **without the `d0,d1` rest-extent normalization** C applies). For non-cubic cells/faces (unequal rest extents) the polar rotation differs from C.
- `support.mat33_to_quat_polar`: same warm start (`trace ≤ 0` Gram–Schmidt) but ≤**50** iterations, stop at `w < 1e-6`, denominator eps `1e-10` (C: 500, `1e-9`, `mjMINVAL`), and `wp.quat_from_matrix` + normalize for the warm start.
- Warp quaternions are `(x,y,z,w)`; face/cell quats store R (C caches R⁻¹), inverses taken as `(−x,−y,−z,w)`.
- Displacements/velocities cached in scratch arrays `[nworld, nflexintcell, 27]`; `K·u` computed per output row by reading `flex_stiffness` directly (no matvec routine).
- Plastic correction honored only when spring enabled; damping computed whenever damper enabled (no `flex_damping == 0` shortcut — result zero).
- Node velocities from `cvel` (`v_com + ω × (x − subtree_com[root])`); C uses `mj_objectVelocity` at CoM + `ω × (x − xipos)` (same rigid velocity).
- Shell interior nodes: Warp uses `d.flexnode_xpos` (computed in smooth) for all nodes; C reconstructs interior nodes by TFI inside `mju_flexGatherState` — shells only use boundary face nodes so equivalent.
- Forces as body wrenches at `xipos` (C fast path writes `xmatᵀ f` into the node body's 3 dofs; equivalent for slide-joint node bodies).

**`_flex_passive_bend_interp`** (world × bending edge via `flex_bend_interp_map`): skips `stiffness ≤ 0` (C: `== 0`); uses cached `d.face_xpos`, `d.face_quat`; averaging `quat_avg = normalize(qA + qB)` after hemisphere flip, rotation of `dn0` by `quat_avg` (already R); force formula identical; no damping (C only warns). No node-index bounds check.

**`_eff_contact_force`**: passive flex contact under the discrete integrator, `atomic_add(qfrc_spring[dof], f·val)` from `build_efm_contact` rows (see effective-metric spec). Interpolated flexes with passive contact raise `NotImplementedError` in io.

### 9.3 Unsupported in Warp (passive scope)
- Body/actuator/sensor plugins → `NotImplementedError` (so the **cable plugin is unsupported**); no generic passive-plugin hook. `m.callback.passive` is a Python-level callback.
- Rigid flexes; quadratic interpolated flexes; passive contact of interpolated flexes; SNH without `integrator=discrete`.
- Sleep filtering inside `passive` (forces computed for all bodies/dofs; C skips sleeping ones and leaves their entries untouched).
- Plastic reference for free joints.

### 9.4 Matching Warp
To reproduce Warp bit-for-bit is impractical (float32, atomics); to match within tolerance: use float32 semantics, replicate the composition order `spring + damper + gravcomp + fluid + adhesion`, the Kutta/density gating, the inertia-box `pow` forms, the un-normalized interp rotations and 50-iteration polar decomposition, and single-precision SNH tolerances.

### 9.5 MJX (JAX) `passive.py` — brief
- `passive`: **if either** SPRING **or** DAMPER is disabled (`disableflags & (SPRING|DAMPER)`), returns `qfrc_passive = 0`, `qfrc_gravcomp = 0` (C requires *both*). Does not populate `qfrc_spring/qfrc_damper/qfrc_fluid` separately.
- `_spring_damper`: linear stiffness only (no `stiffnesspoly`/`dampingpoly`, no actuator damping, no plasticity); free joint `−k(q−qs)` per component (equal to C for linear k since `k(r)=k`), ball via `math.quat_sub` (no normalization of qpos quaternion); dof damping `−b v`; tendon dead-band `below = lower − L > 0 → k·below`, `above = upper − L < 0 → k·above`; tendon damping `−b v`; `qfrc += ten_Jᵀ(fs+fd)` dense.
- `_gravcomp`: vmapped `support.jac` over all bodies (incl. world, mass 0).
- `_fluid`: **inertia-box model only for every body** (ellipsoid model / `geom_fluid` ignored); box clip `1e-12` and `sqrt(box/max(mass,1e-12))·(mass>0)` (C: `mjMINVAL` clip and `mass < mjMINVAL` skip); local velocity by `math.transform_motion(cvel, xipos − root_com, ximat)`. Enabled when `density>0 or viscosity>0 or wind.any()`; `implicitfast` with fluid → `NotImplementedError`.
- No flex (io raises for `nflex`), no plasticity (`nplastic` raises), no adhesion, no callbacks/plugins.

---

## 10. Reproduction pitfalls / parity notes

1. **Both-disabled early return** zeroes gravcomp/fluid/adhesion too; MJX's "either disabled" is a divergence.
2. **Composition order** in C: `(spring+damper)+fluid+adhesion+gravcomp`; Warp: `spring+damper+gravcomp+fluid+adhesion`.
3. `qfrc_spring` includes passive flex contact (`mj_contactPassive`) — added before the sum.
4. **Polynomial forces**: the stiffness polynomial is evaluated on signed x, so its p0 term `−p0·x²` is even (not odd) in x; the damping polynomial uses `|v|` and is odd. Free-joint polynomial is evaluated at the distance/angle magnitude, then multiplied by the vector.
5. Ball/free spring: normalize `qpos` quaternion with `mju_normalize4` semantics; `subQuat` wraps angles > π.
6. Joint damping includes `actuator_damping·gear[0]²` from `jnt_actuatorid` (−2 → scan). The gear used is `actuator_gear[6*actuator_outadr[a]]` (first component).
7. Tendon spring uses the dead-band `[lower, upper]`; plastic state overrides it.
8. Flex StVK damping uses `kD = flex_damping/timestep` and the factorization `dL(2L − dL)`, `dL = v·dt` (no subtraction of squared lengths). 2D/1D bending and interpolated damping use `flex_damping` without `/timestep`. SNH damping uses `flex_damping` × PSD Hessian, no `/timestep`.
9. `flexedge_velocity` is zero for interp/rigid flexes; `mj_flexPassiveStretch` needs it for non-SNH damping.
10. Interpolated-flex K blocks are stored negated; forces are `+K u` in the corotational frame, rotated back.
11. The plastic correction for interpolated elements is relative to node 0 of the element (`dx0 = x0_n − x0_0`) with half off-diagonal Voigt strains.
12. Fluid: the ellipsoid model is selected per *body* (any geom with `fluid[0] > 0`); then non-ellipsoid geoms of that body contribute nothing. Forces are applied at `geom_xpos` (ellipsoid) or `xipos` (box), both in the respective local frames. Wind enters only the linear velocity.
13. Fluid `setFluid` uses a fixed 15-point quadrature in **double** with the exact tables above; using an adaptive integrator will change `geom_fluid[6..11]` in the last digits.
14. Inertia box uses **full** box sizes (`sqrt(6(...)/M)`), not half sizes; `mjMINVAL` clamp inside the sqrt.
15. `mj_gravcomp` skips the world body and requires `flg_gravcomp && !DISABLED(GRAVITY) && |g|>0`; actuator gravcomp is added in `mj_fwdActuation` before `actfrcrange` clamping.
16. Sleep: C skips sleeping bodies/dofs/tendons and leaves their passive entries untouched (not zeroed); flex elasticity and flex-edge forces are not sleep-filtered.
17. `mjcb_passive` and passive plugins (cable) run after `qfrc_passive` is composed and write into `qfrc_passive` only.
18. `mj_flexPassiveBendInterp` emits `mju_warning` on every call when bending damping would be active; it never applies damping.
19. Warp: float32, atomics, 50-iteration polar decomposition, missing rest-extent normalization in interpolated rotations, Kutta gating, adhesion ignores `exclude`, plastic free joints ignored, no sleep filtering.
20. Rod bending curved references are smoothed with `e = 1e-4` (`gam = sqrt(1+1e-8)`); Hessians are PSD-clamped but forces use the exact smoothed gradient.
21. 2D bending: only interior edges (flap[1] ≠ −1); `b16` gets the per-edge plastic offset located after the stretch plastic block (`3·elemnum` numbers, only if stretch stiffness is nonzero).

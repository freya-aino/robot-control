# 10 — Actuation, Sensors, History Buffers, Actuator/Sensor Plugins

**Sources covered**

C engine (commit 7ba52b5):
- `src/engine/engine_forward.c`: `mj_fwdActuation`, helpers `unpackServoInputs`, `dcmotorVoltage`, `clampVec`, `expmap2Quat`, `wrapPeriod`, `wrapSetpoint`, `slewLimit`; `advanceStart` (history + activation integration); `mj_fwdVelocity` (actuator_velocity); stage ordering in `mj_forwardSkip`, `mj_step1/2`.
- `src/engine/engine_util_misc.c`: `mju_muscleGainLength`, `mju_muscleGain`, `mju_muscleBias`, `mju_muscleDynamicsTimescale`, `mju_muscleDynamics`, `mju_sigmoid`, `mj_lugreStribeck`, `mj_dcmotorSlots`, history ring buffer (`historyPhysicalIndex`, `historyFindIndex`, `mju_historyInit`, `mju_historyInsert`, `mju_historyRead`), `mju_camPixelRay`, `mju_insidePrimitive`, `mju_polyPotential`, `mju_round`, `mju_isBad`.
- `src/engine/engine_support.c`: `mj_actuatorDisabled`, `mj_nextActivation`, `mj_readCtrl`, `mj_readSensor`, `mj_initCtrlHistory`, `mj_initSensorHistory`, `mju_condataSize`, `mju_raydataSize`, `mju_camIntrinsics`, `mj_geomDistance`, `mj_insideSite`, `mjCONDATA_SIZE`, `mjRAYDATA_SIZE`.
- `src/engine/engine_core_util.c`: `mj_dcmotorResistance`, `mj_objectVelocity`, `mj_objectAcceleration`, `mj_contactForce`, `mj_effActuatorPossible`.
- `src/engine/engine_core_smooth.c`: `mj_transmission` (actuator_length / actuator_moment), `mj_subtreeVel`, `mj_rnePostConstraint` (as consumed by sensors).
- `src/engine/engine_sensor.c` (entire file): all sensor types, cutoff, contact sensor matching/reduction, tactile, rangefinder, history/interval handling, user/plugin sensors, `mj_sensorPos/Vel/Acc`, `mj_energyPos/Vel`.
- `src/engine/engine_setconst.c`: `actuator_acc0`, `setDampRatio`, `evalAct`, `mj_setLengthRange`; `engine_init.c: mj_defaultLROpt`; `engine_io.c: _resetData` history init.
- Compiler: `src/user/user_objects.cc` (`mjCActuator::Compile`, `mjCSensor::Compile`, `sensorDatatype`, `sensorNeedstage`, `ApplyGauge`), `src/user/user_api.cc` (`mjs_setTo*` shortcuts incl. DC motor, `mjs_sensorDim`), `src/user/user_model.cc: LengthRange`.
- Plugins: `plugin/actuator/pid.cc` (`mujoco.pid`), `plugin/sensor/touch_grid.cc` (`mujoco.sensor.touch_grid`).
- Docs: `doc/modeling.rst` (CSensor, CDelay), `doc/XMLreference.rst` (collision sensors), `doc/dcmotor/dcmotor.tex`.

GPU / JAX:
- MuJoCo Warp (commit d41a47c): `_src/forward.py` (`_actuator_force`, `_tendon_actuator_force`, `_tendon_actuator_force_clamp`, `_qfrc_actuator`, `_qfrc_actuator_gravcomp_limits`, `fwd_actuation`, `_actuator_velocity`, `_next_activation`, `_wrap_period`, `_advance`, `forward`, `step1/2`, `rungekutta4`), `_src/sensor.py` (all), `_src/history.py` (all), `_src/util_misc.py` (muscle/DC motor helpers, `inside_geom`), `_src/support.py: next_act`, `_src/io.py` (feature checks and sensor address precomputation), `_src/collision_core.py` (collision-sensor contact tagging).
- MJX: `mjx/_src/forward.py: fwd_actuation`, `mjx/_src/sensor.py`, `mjx/_src/io.py` checks.

**Role in the pipeline.** Actuation converts user controls `ctrl` (optionally delayed through a history buffer) and actuator internal state `act` into scalar actuator forces `actuator_force` (one per *force output*, `nout`), activation derivatives `act_dot`, and generalized forces `qfrc_actuator = actuator_momentᵀ · actuator_force` (plus actuator-level gravity compensation, clamped by joint actuator-force limits). Transmission (`mj_transmission`, position stage) provides `actuator_length` and the sparse `actuator_moment`; `actuator_velocity = moment·qvel` is computed in the velocity stage. Sensors read simulation quantities at three stages (`mj_sensorPos` after `mj_fwdPosition`, `mj_sensorVel` after `mj_fwdVelocity`, `mj_sensorAcc` after the constraint solver) into `sensordata`; they never affect dynamics, except that some lazily trigger `mj_subtreeVel`, `mj_rnePostConstraint`, `mj_energyPos/Vel` which fill mjData fields. History buffers (`d->history`) are part of the physics state and are advanced at the start of every integrator step (`advanceStart`).

---

## 0. Data layout and constants

### 0.1 Constants

| constant | value | where |
|---|---|---|
| `mjMINVAL` | `1e-15` (double build; `1e-15f` single) | mjtype.h |
| `mjMAXVAL` | `1e10` | mjtype.h — `mju_isBad(x)` ⇔ `x!=x || x>1e10 || x<-1e10` |
| `mjPI` | π | |
| `mjNDYN`, `mjNGAIN`, `mjNBIAS` | 10 each | mjmodel.h |
| `mjNSENS` | 3 (sensor_intprm per sensor) | mjmodel.h |
| `mjMAXCONPAIR` | 50 | mjmodel.h (Warp tactile geom-list capacity) |
| `mjNCONDATA` | 9 | mjtype.h |
| `mjNRAYDATA` | 6 | mjtype.h |
| history `nsample` max | `16777216 = 2^24` (cursor stored as mjtNum; exact in float32) | compiler |
| tactile parallel threshold | `kTactileParallelThreshold = 1000` taxels | engine_sensor.c |
| LR default opts | mode=MUSCLE, useexisting=1, uselimit=0, accel=20, maxforce=0, timeconst=1, timestep=0.01, inttotal=10, interval=2, tolrange=0.05 | `mj_defaultLROpt` |
| Warp `contact_sensor_maxmatch` | 64 default; overridable via `<numeric name="contact_sensor_maxmatch">` | warp io.py |
| Warp history time tolerance | `1e-6` (vs C `mjMINVAL=1e-15`) | history.py |

`mjCONDATA_SIZE = {FOUND:1, FORCE:3, TORQUE:3, DIST:1, POS:3, NORMAL:3, TANGENT:3, LINVEL:3, ANGVEL:3}`.
`mjRAYDATA_SIZE = {DIST:1, DIR:3, ORIGIN:3, POINT:3, NORMAL:3, DEPTH:1}`.
`mju_condataSize(spec) = Σ_{bit i set} mjCONDATA_SIZE[i]`; `mju_raydataSize` analogous.

### 0.2 Enums (values matter: switch order and enum-comparisons are used)

```
mjtTrn : JOINT=0, JOINTINPARENT, SLIDERCRANK, TENDON, SITE, BODY, SO3, UNDEFINED=1000
mjtDyn : NONE=0, INTEGRATOR, FILTER, FILTEREXACT, MUSCLE, DCMOTOR, PID, USER
mjtGain: FIXED=0, AFFINE, MUSCLE, DCMOTOR, SO3, PID, USER
mjtBias: NONE=0, AFFINE, MUSCLE, DCMOTOR, SO3, USER
mjtCtrlChart (ctrlspec for SO3): EXPMAP=1, QUAT=2
mjtCtrlInput (ctrlspec bitmask): POS=1, VEL=2, FF=4, VOLTAGE=8, NONE=16, PRESSURE=32
mjtStage: NONE=0, POS, VEL, ACC
mjtDataType: REAL=0, POSITIVE, AXIS, QUATERNION
mjtSensor: TOUCH=0, ACCELEROMETER, VELOCIMETER, GYRO, FORCE, TORQUE, MAGNETOMETER, RANGEFINDER,
  CAMPROJECTION, JOINTPOS, JOINTVEL, TENDONPOS, TENDONVEL, ACTUATORPOS, ACTUATORVEL, ACTUATORFRC,
  JOINTACTFRC, TENDONACTFRC, BALLQUAT, BALLANGVEL, JOINTLIMITPOS, JOINTLIMITVEL, JOINTLIMITFRC,
  TENDONLIMITPOS, TENDONLIMITVEL, TENDONLIMITFRC, FRAMEPOS, FRAMEQUAT, FRAMEXAXIS, FRAMEYAXIS,
  FRAMEZAXIS, FRAMELINVEL, FRAMEANGVEL, FRAMELINACC, FRAMEANGACC, SUBTREECOM, SUBTREELINVEL,
  SUBTREEANGMOM, INSIDESITE, GEOMDIST, GEOMNORMAL, GEOMFROMTO, CONTACT, E_POTENTIAL, E_KINETIC,
  CLOCK, TACTILE, PLUGIN, USER
mjtConDataField: FOUND=0, FORCE, TORQUE, DIST, POS, NORMAL, TANGENT, LINVEL, ANGVEL
mjtRayDataField: DIST=0, DIR, ORIGIN, POINT, NORMAL, DEPTH
mjtLRMode: NONE=0, MUSCLE, MUSCLEUSER, ALL
```
`FRAMEXAXIS/Y/Z` are consecutive: `offset = type - mjSENS_FRAMEXAXIS ∈ {0,1,2}` is used directly as a column index.

### 0.3 mjModel actuator fields

| field | shape | meaning |
|---|---|---|
| `actuator_trntype/dyntype/gaintype/biastype` | nu | enums |
| `actuator_ctrladr`, `actuator_ctrlnum` | nu | input block in `ctrl` (width 0–4; 1 for standard actuators) |
| `actuator_ctrlspec` | nu | input signature (meaning scoped by gaintype) |
| `actuator_outadr`, `actuator_outnum` | nu | output block (1, or 3 for SO3) in `actuator_force/length/velocity/gear/acc0/lengthrange` |
| `actuator_actadr` (−1 stateless), `actuator_actnum` | nu | activation block in `act` |
| `actuator_trnid` | nu×2 | target id (+ refsite / slider site / −1) |
| `actuator_cranklength` | nu (indexed by out in C) | slider-crank rod |
| `actuator_dynprm/gainprm/biasprm` | nu×10 | parameters |
| `actuator_actlimited/actrange` | nu, nu×2 | activation clamp |
| `actuator_actearly` | nu | use next-step activation in force |
| `actuator_history` | nu×2 | `[nsample, interp]` |
| `actuator_historyadr` | nu | offset in `d->history`, −1 none |
| `actuator_delay` | nu | seconds |
| `actuator_group` | nu | for `opt.disableactuator` |
| `actuator_plugin` | nu | instance id or −1 |
| `actuator_forcelimited/forcerange` | nu, nu×2 | output clamp |
| `actuator_ctrllimited/ctrlrange` | **nu_ctrl** (= `m->nu` total controls) ×(1/2) | per-input clamp |
| `actuator_gear` | nout×6 | |
| `actuator_acc0` | nout | ‖M⁻¹·moment‖ at qpos0 |
| `actuator_length0`, `actuator_lengthrange` | nout, nout×2 | |
| `actuator_damping/dampingpoly/armature` | | passive (other spec file) |

Note: in the C source, `nu` loops over `m->nu` controls (`clampVec(ctrl, actuator_ctrlrange, actuator_ctrllimited, nu, NULL)`), and `nactuator` loops over actuators. For standard models `nu == nactuator == nout`.

mjData: `ctrl`(nu), `act`(na), `act_dot`(na), `actuator_length/velocity/force`(nout), `moment_rownnz/rowadr`(nout), `moment_colind/actuator_moment`(nJmom), `qfrc_actuator`(nv), `history`(nhistory), `sensordata`(nsensordata), `energy[2]`, flags `flg_energypos, flg_energyvel, flg_subtreevel, flg_rnepost`.

### 0.4 History buffer memory layout

Per actuator/sensor with `nsample = n > 0`, size `2 + n + n·dim`:
```
buf[0]          user slot: actuators: 0 (preserved); sensors: time of last interval tick ("time_prev")
buf[1]          cursor (stored as mjtNum): PHYSICAL index of newest sample
buf[2 .. 2+n)   times[n]   (physical ring)
buf[2+n ..)     values[n*dim]
```
`dim = actuator_ctrlnum` (actuators) or `sensor_dim` (sensors). `nhistory = Σ (2 + n + n·dim)` over actuators then sensors (compiler order: actuators first).

---

## 1. Transmission summary (`mj_transmission`, engine_core_smooth.c)

Computed in `mj_fwdPosition` after `mj_projectConstraint` (so BODY transmissions can use `efc_J`). Outputs per output row `out`: `actuator_length[out]`, sparse row `(rownnz, rowadr, colind, moment)`. Row addresses are packed: `rowadr[out] = out==0 ? 0 : rowadr[out-1] + rownnz[out-1]`. Sleeping actuators: all rows of their output block get `rownnz=0` (and `rowadr` chained).

- **JOINT / JOINTINPARENT**
  - slide/hinge: `rownnz=1, colind=dofadr, length = qpos·gear[0], moment = gear[0]`.
  - ball: `quat = normalize(qpos[adr:adr+4])`, `axis = quat2Vel(quat, dt=1)` (expmap: `axis=(q1,q2,q3)/‖·‖`, `angle=2·atan2(‖(q1,q2,q3)‖, q0)`, minus 2π if > π). `gearAxis = gear[0:3]` (JOINT) or `rot(conj(quat))·gear[0:3]` (JOINTINPARENT). `length = axis·gearAxis`, 3 nnz at dofadr..+2 with `moment = gearAxis`.
  - free: `length = 0`; 6 nnz; `moment[0:3] = gear[0:3]`, `moment[3:6] = gear[3:6]` (JOINT) or `rot(conj(normalize(qpos quat)))·gear[3:6]` (JOINTINPARENT).
- **SLIDERCRANK**: `axis` = slider site z-axis (column 2 of `site_xmat[idslider]`), `vec = site_xpos[id] - site_xpos[idslider]`, `av = vec·axis`, `det = av² + rod² − vec·vec`. If `det <= 0`: `ok=0, sdet=0, length=av`, `dlda=vec, dldv=axis`; else `sdet=√det`, `length=av−sdet`, `dldv = axis·(1−av/sdet) + vec/sdet`, `dlda = vec·(1−av/sdet)`. Jacobians: `mj_jacPointAxis` at slider site for axis (jacA) and point (jacS), `jac = jacSite(id) − jacS`; `moment_row[j] = Σ_k dlda[k]·jacA[k,j] + dldv[k]·jac[k,j]`; `length *= gear[0]`; compress nonzeros with `moment = row·gear[0]`.
- **TENDON**: `length = ten_length·gear[0]`; row = tendon's sparse `ten_J` scaled by `gear[0]` (same colind).
- **SITE**: `length = 0` initially.
  - no refsite: `wrench_t = site_xmat·gear[0:3]`, `wrench_r = site_xmat·gear[3:6]`, `moment_row = jacpᵀ·wrench_t + jacrᵀ·wrench_r` (length stays 0).
  - refsite: find common ancestral dof by walking `dof_parentid` from the last dof of each weld body; translational part (if gear[0:3]≠0): `vec = ref_xmatᵀ(site_xpos − ref_xpos)`, `length += vec·gear[0:3]`; `jac = jacp(site) − jacp(ref)` with columns of the common ancestral dof chain zeroed; `moment_row = jacᵀ·(ref_xmat·gear[0:3])`. Rotational part (if gear[3:6]≠0): `quat = xquat[site body]*site_quat`, `refquat` likewise, `vec = subQuat(quat, refquat)` (= quat2Vel(conj(refquat)*quat)), `length += vec·gear[3:6]`; `jacS = jacr(site) − jacr(ref)` with common chain zeroed; `moment_row += jacSᵀ·(ref_xmat·gear[3:6])`.
  - compress nonzeros (exact `!= 0` test).
- **SO3** (3 outputs): ball joint (trnid[1]==−1): `length[k] = expmap(normalize(q))[k]`, row k = unit at dof k. Site+refsite: `length[k] = subQuat(quat, refquat)[k]`; relative rotational Jacobian (common chain zeroed); row k = `jacSᵀ·(site_xmat column k)`.
- **BODY** (adhesion): `length = 0`. For each contact with both geoms ≥0 involving body `id`: if `!con->exclude` and `efc_address>=0`: `counter++`; mark `efc_force` (temp) with 1 at the normal row (condim 1 or elliptic) or `0.5/npyramid` on each of the `2·(dim−1)` pyramid rows; if `exclude==1` (in gap): `counter++`, `moment_exclude += frameᵀ[normal]·(jac_b2 − jac_b1)` at con->pos. If `counter`: `moment_row = J_efcᵀ·efc_force_tmp + moment_exclude`, then `moment_row *= −1/counter`. Compress.

`actuator_velocity` (fwdVelocity): `mju_mulMatVecSparse(actuator_velocity, actuator_moment, qvel, nout, ...)` if actuation enabled, else zero.

---

## 2. `mj_fwdActuation` (engine_forward.c) — exact algorithm

Inputs: `ctrl, act, actuator_length, actuator_velocity, actuator_moment, history, qfrc_gravcomp`. Outputs: `act_dot, actuator_force, qfrc_actuator`.

```
1  force := actuator_force; zero(force, nout)
2  sleep_filter := ENABLED(SLEEP)        (note: here not conditioned on nbody_awake)
3  if nactuator==0 || DISABLED(ACTUATION): zero(qfrc_actuator); return   (act_dot untouched!)
4  ctrl_local := copy(d->ctrl, nu)
5  for each actuator i with actuator_delay[i] != 0:
       ptr = mj_readCtrl(m,d,i,d->time, ctrl_local+ctrladr, interp=-1)
       if ptr: copy(ctrl_local+ctrladr, ptr, ctrlnum)      (NULL => interpolated into ctrl_local already)
6  if !DISABLED(CLAMPCTRL): for j<nu: if ctrllimited[j]: ctrl[j] = clip(ctrl[j], range[2j], range[2j+1])
7  for j<nu: if mju_isBad(ctrl[j]) { mj_warning(BADCTRL, j); zero(ctrl, nu); break; }
8  act_dot loop (section 2.1)
9  actuator plugins: plugin->actuator_act_dot(m,d,instance) for every plugin with mjPLUGIN_ACTUATOR
10 force loop (section 2.2)
11 actuator plugins: plugin->compute(m,d,instance,mjPLUGIN_ACTUATOR)
12 tendon actuator-force limits (section 2.4)
13 actuator force clamp (section 2.5)
14 DC motor mechanical forces (section 2.6)
15 qfrc_actuator = momentᵀ · force   (mju_mulMatTVecSparse; zero-initialised inside)
16 actuator-level gravcomp (section 2.7)
17 clamp qfrc_actuator per joint actfrcrange (section 2.7)
```
Ordering facts that affect results: delay read → ctrl clamp → NaN check; `act_dot` computed for ALL stateful actuators including disabled-group ones (disabled ones get 0 force; their act_dot is zeroed only at integration, see §4); plugins' act_dot after builtin; tendon force scaling BEFORE per-actuator forcerange clamp; DC motor cogging/LuGre AFTER both clamps.

### 2.1 Activation derivative loop

For each actuator i (skip if sleeping; skip if `actadr < 0`):
- `uadr = ctrladr[i]`, `oadr = outadr[i]`, `actnum = actuator_actnum[i]`; `if actnum: zero(act_dot[actadr .. +actnum])`.
- `act_last = actadr + actnum − 1` (actuator plugins may prepend their own state; builtin dynamics use the LAST slot).

| dyntype | act_dot |
|---|---|
| INTEGRATOR | for `j<ctrlnum`: `act_dot[act_last−ctrlnum+1+j] = ctrl[uadr+j]` (SO3 integrator: 3 slots) |
| FILTER, FILTEREXACT | `tau = max(mjMINVAL, dynprm[0])`; `act_dot[last] = (ctrl[uadr] − act[last]) / tau` |
| MUSCLE | `act_dot[last] = mju_muscleDynamics(ctrl[uadr], act[last], dynprm)` (§3.3) |
| PID | §2.1.1 |
| DCMOTOR | §2.1.2 |
| default (USER) | if `mjcb_act_dyn`: `actnum==1` ⇒ `act_dot[last] = mjcb_act_dyn(m,d,i)`; else callback writes act_dot itself. No callback ⇒ act_dot stays 0 |
| NONE with actadr≥0 | only possible for plugin actuators; act_dot zeroed |

#### 2.1.1 PID dynamics (slots in order: [slew], [integral])
```
adr = actadr; period = wrapPeriod(m,i)
slew_s = dynprm[1]
if slew_s > 0:
    ctrl[uadr] = slewLimit(ctrl[uadr], act[adr], slew_s, timestep, period, &act_dot[adr]); adr++
if gainprm[0] (=ki) > 0:
    err = ctrl[uadr] − actuator_length[oadr]
    if period>0: err -= period*round(err/period)
    imax = dynprm[0]
    if imax>0: z=act[adr]; if z>=imax: err=min(err,0) elif z<=-imax: err=max(err,0)
    act_dot[adr] = err
```
`slewLimit(u, u_prev, slew_s, dt, period, *act_dot)`: if period>0, `u = wrapSetpoint(u, u_prev, period)`; `slew = slew_s·dt`; `u_eff = clip(u, u_prev−slew, u_prev+slew)`; `*act_dot = (u_eff − u_prev)/dt`; returns `u_eff`. Note: the slew-limited setpoint **overwrites the local ctrl** and is used by the force loop.

`wrapSetpoint(u, length, period) = u − period·round((u−length)/period)` with `round` = C `round()` (half away from zero), via `mju_round` returning int (saturates to INT_MAX/MIN).

`wrapPeriod(m,i)`: servo shape = `gaintype==FIXED && biastype==AFFINE && gainprm[0]==−biasprm[1] && dyntype∈{NONE,INTEGRATOR}`; or PID gaintype. If neither: 0. Then: SITE transmission with refsite (`trnid[1]>=0`) and `gear[0..2]` all exactly zero ⇒ `2π·‖gear[3:6]‖`; JOINT/JOINTINPARENT on a BALL joint ⇒ `2π·‖gear[0:3]‖`; else 0. (`gear` = `actuator_gear + 6·outadr`.)

#### 2.1.2 DC motor dynamics

Slot assignment `mj_dcmotorSlots(dynprm, gainprm)` (contiguous, in this fixed order; −1 if absent):
```
slew        if dynprm[7] > 0
integral    if gainprm[5] > 0   (ki)
temperature if dynprm[2] > 0    (RT)
bristle     if dynprm[5] > 0    (sigma0)
current     if dynprm[0] > 0    (te)
num_slots = count   (must equal actuator_actnum, else mjERROR)
```
Parameter map (from `mjs_setToDCMotor`):
```
gainprm: [0]=R  [1]=K  [2]=alpha(temp coeff)  [3]=T0(ref temp)  [4]=kp [5]=ki [6]=kd [7]=Vmax(driver)  [8]=0
dynprm : [0]=te(electrical time const; = L/R if inductance given) [1]=(di/dt)max [2]=RT [3]=C [4]=Ta
         [5]=sigma0 [6]=sigma1 [7]=slewmax [8]=Imax(integral clamp)
biasprm: [0]=cogging amplitude A [1]=Np [2]=phase phi [3]=F_C(Coulomb) [4]=F_S(static) [5]=v_S(Stribeck)
forcerange = ±tau_max (from saturation; tau_max = K·i_max if only i_max given), forcelimited=TRUE
actlimited forced FALSE; actearly forced 1 (compiler error if false)
K = sqrt(Kt·Ke) if both>0 else Kt if Kt>0 else Ke ; R from nominal: R = K·vn/tau0 if R==0
```
Algorithm (`velocity = actuator_velocity[oadr]`, `R=gainprm[0]`, `K=gainprm[1]`, `ki=gainprm[5]`, `te=dynprm[0]`):
```
adr = actadr
if dynprm[7] > 0:  ctrl[uadr] = slewLimit(ctrl[uadr], act[adr], dynprm[7], dt, 0, &act_dot[adr]); adr++
x_I = 0
if ki > 0:
   x_I = act[adr]; Imax = dynprm[8]; ad = ctrl[uadr] − actuator_length[oadr]
   if Imax>0: if x_I>=Imax: ad=min(ad,0) elif x_I<=-Imax: ad=max(ad,0)
   act_dot[adr] = ad; adr++
V = dcmotorVoltage(ctrl+uadr, ctrlspec, actuator_length[oadr], velocity, x_I, gainprm)
if RT=dynprm[2] > 0:
   C=dynprm[3]; Ta=dynprm[4]; alpha=gainprm[2]; T0=gainprm[3]; T=act[adr]
   R *= 1 + alpha*(T + Ta − T0)                     (R now temperature-corrected; NOT clamped here)
   current = (te>0) ? act[act_last] : (V − K*velocity)/R
   act_dot[adr] = (R*current² − T/RT)/C ; adr++
if sigma0=dynprm[5] > 0:
   g = mj_lugreStribeck(velocity, biasprm[3], biasprm[4], biasprm[5])
   a = −sigma0*|velocity| / max(mjMINVAL, g)
   act_dot[adr] = a*act[adr] + velocity ; adr++
if te > 0:
   i_dot = (V/R − K/R*velocity − act[act_last]) / te
   if dynprm[1] > 0: i_dot = clip(i_dot, −dimax, dimax)
   act_dot[act_last] = i_dot
```
`mj_lugreStribeck(v, F_C, F_S, v_S) = F_C + (F_S − F_C)·exp(−(v/max(mjMINVAL,v_S))²)`.

`unpackServoInputs(u, spec, out[4])`: packed order `[pos, vel, ff, voltage]`; `adr=0`; `out[0] = spec&POS ? u[adr++] : 0`; `out[1] = spec&VEL ? u[adr++] : 0`; `out[2] = spec&FF ? u[adr++] : 0`; `out[3] = spec&VOLTAGE ? u[adr] : 0`.

`dcmotorVoltage(u, spec, length, velocity, x_I, gainprm)`:
```
u4 = unpackServoInputs(u, spec)
V = 0
if spec & (POS|VEL|FF):
   torque = kp*(u4[0]−length) + kd*(u4[1]−velocity) + ki*x_I + u4[2]
   V = R/K*torque + K*velocity                       (R = gainprm[0], NOT temperature-corrected)
   if Vmax=gainprm[7] > 0: V = clip(V, −Vmax, Vmax)
return V + u4[3]                                     (raw voltage: downstream, unclamped)
```
With `spec == mjINPUT_NONE (16)`: no bits in POS|VEL|FF|VOLTAGE, so V = 0 (ctrlnum = 0).

### 2.2 Force loop (gain · input + bias)

For each actuator i: skip if sleeping; skip if `mj_actuatorDisabled(m,i)` (group g ∈ [0,30] and bit g of `opt.disableactuator` set); skip if `actuator_plugin[i] >= 0`. (Skipped actuators keep `force = 0`.)

**SO3 gain (early-continue path)** — 3 outputs:
```
if ctrlspec == QUAT: q_tgt = normalize4(ctrl[uadr:uadr+4])   (mju_normalize4: zero → identity)
else:
   u = (dyntype==NONE) ? ctrl[uadr:uadr+3]
       : actearly ? [mj_nextActivation(m,d,i,actadr+k, act_dot[actadr+k]) for k<3] : act[actadr:actadr+3]
   q_tgt = expmap2Quat(u)          (angle=‖u‖; angle<mjMINVAL → identity; else axisAngle2Quat(u/angle, angle))
q_cur = expmap2Quat(actuator_length[oadr:oadr+3])
e = subQuat(q_tgt, q_cur)          (= quat2Vel(conj(q_cur)*q_tgt))
force[oadr+k] = kp*e[k] + biasprm[0] + biasprm[2]*actuator_velocity[oadr+k],   kp = gainprm[0]
```
**Tendon limit detection**: if `ntendon && !tendon_frclimited && trntype==TENDON`: `tendon_frclimited = tendon_actfrclimited[trnid[0]]` (sets the flag to run §2.4).

**Gain** (`len = actuator_length[oadr]`, `vel = actuator_velocity[oadr]`):

| gaintype | gain |
|---|---|
| FIXED | `gainprm[0]` |
| PID | `0` (input side computed separately below) |
| AFFINE | `gainprm[0] + gainprm[1]·len + gainprm[2]·vel` |
| MUSCLE | `mju_muscleGain(len, vel, lengthrange+2·oadr, acc0[oadr], gainprm)` |
| DCMOTOR | `R = mj_dcmotorResistance(m,d,i)` (temperature-corrected, `max(mjMINVAL,R)`); `gain = (dynprm[0] > 0) ? K : K/R`; then if `ctrlspec != VOLTAGE && ctrlnum > 0`: `x_I = act[actadr+slots.integral]` (or 0), `ctrl[uadr] = dcmotorVoltage(ctrl+uadr, ...)` (overwrites with the controller voltage; uses the possibly slew-limited ctrl[uadr]) |
| USER (default) | `mjcb_act_gain ? mjcb_act_gain(m,d,i) : 1` |

`mj_dcmotorResistance`: `R = gainprm[0]`; if temperature slot exists: `T = act[actadr+slots.temperature]`, `R *= 1 + alpha·(T + Ta − T0)`; return `max(mjMINVAL, R)`.

**Input selection** (`dcmotor_no_current = gaintype==DCMOTOR && dynprm[0] <= 0`):
- **PID**: `u4 = unpackServoInputs(ctrl+uadr, ctrlspec)`; `qref,vref,ff = u4[0..2]`; `period = wrapPeriod`; if >0: `qref = wrapSetpoint(qref, len, period)`; `force[oadr] = −biasprm[1]·qref − biasprm[2]·vref + ff`; if `actnum && gainprm[0] > 0`: `z = actearly ? mj_nextActivation(last) : act[last]`; `force += gainprm[0]·z`. (The bias is added below with biastype AFFINE: `biasprm[0] + biasprm[1]·len + biasprm[2]·vel`, giving `kp(qref−len) + kv(vref−vel)` since biasprm[1]=−kp, biasprm[2]=−kv.)
- **actnum == 0 or dcmotor_no_current**: `input = ctrlnum ? ctrl[uadr] : 0`; if `period=wrapPeriod>0`: `input = wrapSetpoint(input, len, period)`; `force[oadr] = gain·input`.
- **else (stateful)**: `act_adr = actadr + actnum − 1`; `a = actearly ? mj_nextActivation(m,d,i,act_adr,act_dot[act_adr]) : act[act_adr]`; if `period>0`: `a = wrapSetpoint(a, len, period)`; `force[oadr] = gain·a`.

**Bias**:

| biastype | bias |
|---|---|
| NONE | 0 |
| AFFINE | `biasprm[0] + biasprm[1]·len + biasprm[2]·vel` |
| MUSCLE | `mju_muscleBias(len, lengthrange+2·oadr, acc0[oadr], biasprm)` |
| DCMOTOR | `0`; if `dynprm[0] <= 0` (stateless current): `bias −= gain·K·vel` (back-EMF; with gain=K/R ⇒ −K²/R·vel) |
| USER (default) | `mjcb_act_bias ? mjcb_act_bias(m,d,i) : 0` |

`force[oadr] += bias`.

Resulting DC motor torque: stateful current `τ = K·i_next` (actearly always) ; stateless `τ = (K/R)·V − (K²/R)·ω`.

### 2.3 `mj_nextActivation(m, d, i, act_adr, act_dot)` (engine_support.c)
```
act = d->act[act_adr]
if dyntype == FILTEREXACT:
    tau = max(mjMINVAL, dynprm[0]);  act += act_dot*tau*(1 − exp(−timestep/tau))
elif dyntype == DCMOTOR:
    offset = act_adr − actadr; slots = mj_dcmotorSlots
    if offset == slots.current:  te=max(mjMINVAL,dynprm[0]); act += act_dot*te*(1−exp(−h/te))
    elif offset == slots.bristle:
        g = lugreStribeck(actuator_velocity[outadr], F_C,F_S,v_S); a = −sigma0*|v|/max(mjMINVAL,g)
        exp_ah = exp(a*h); int_h = (|a| > mjMINVAL) ? (exp_ah−1)/a : h
        act = exp_ah*act + int_h*v                     (exact ZOH; act_dot argument ignored)
    elif offset == slots.integral: act += act_dot*h; if Imax>0: act = clip(act, −Imax, Imax)
    else (temperature, slew): act += act_dot*h
else: act += act_dot*h                                (INTEGRATOR, FILTER, MUSCLE, USER, PID)
if dyntype != DCMOTOR && actlimited: act = clip(act, actrange[0], actrange[1])
return act
```
Used (a) in the force loop when `actearly`, (b) in `advanceStart` to integrate every activation slot.

### 2.4 Tendon actuator-force limits
Only if some TENDON-transmission actuator's tendon has `tendon_actfrclimited`:
```
total[t] = Σ_{i: trntype==TENDON, trnid[0]==t, tendon_actfrclimited[t]} force[outadr[i]]
for each TENDON actuator i, t=trnid[0]:
   F = total[t]
   if actfrclimited[t] && F != 0:
       if F < range[0]: force[outadr] *= range[0]/F
       elif F > range[1]: force[outadr] *= range[1]/F
```
Note: summed over ALL actuators including disabled / sleeping (their force is 0). Computed after actuator plugins.

### 2.5 Actuator force clamp
For each i with `forcelimited`: SO3 gain ⇒ clamp the norm: `n = ‖f[0:3]‖; if n > range[1]: f *= range[1]/n` (compiler requires range[0]==0); otherwise for `j<outnum`: `f[j] = clip(f[j], range[0], range[1])`.

### 2.6 DC motor mechanical forces (not subject to current limits)
For each i with `biastype==DCMOTOR`, not sleeping, not disabled, not plugin:
```
if A=biasprm[0] != 0:  force[oadr] += A*sin(Np*actuator_length[oadr] + phi)      (cogging)
if sigma0=dynprm[5] > 0:
   z = act[actadr+slots.bristle]; z_dot = act_dot[actadr+slots.bristle]
   force[oadr] −= sigma0*z + sigma1*z_dot                                           (LuGre)
```
Uses current `act` (not next-step) and the just-computed `act_dot`.

### 2.7 qfrc_actuator, gravcomp, joint limits
- `qfrc_actuator = Σ_out moment_rowᵀ·force[out]` (sparse transpose multiply, ordered by output row then nnz).
- If `m->flg_gravcomp && !DISABLED(GRAVITY) && ‖gravity‖ != 0`: for each joint with `jnt_actgravcomp[j]`: `qfrc_actuator[dofadr..+dofnum] += qfrc_gravcomp[...]` (dofnum {FREE:6,BALL:3,SLIDE:1,HINGE:1}). `qfrc_gravcomp` was computed in `mj_passive`.
- `clampVec(qfrc_actuator, jnt_actfrcrange, jnt_actfrclimited, njnt, jnt_dofadr)`: per joint j with actfrclimited: clamp ONLY the first dof `qfrc_actuator[jnt_dofadr[j]]` (compiler forces actfrclimited=FALSE for free/ball joints).

### 2.8 Control input semantics per gaintype (compiler, `mjCActuator::Compile`)
- FIXED/AFFINE: optional single declared input POS, VEL or PRESSURE (ctrlnum=1).
- MUSCLE, USER: no ctrlspec allowed.
- PID: biastype must be AFFINE; dyntype NONE or PID; ki (gainprm[0]) requires PID dyntype; no BODY transmission; ctrlspec ⊆ {POS,VEL,FF}, default POS|VEL; PID dynamics require POS; ctrlnum = popcount. Per-input ranges: pos uses ctrlrange; vel uses `velrange` (limited iff velrange[0]<velrange[1]); ff uses `ffrange`. actdim = (dynprm[1]>0)+(gainprm[0]>0).
- DCMOTOR: ctrlspec default VOLTAGE; NONE or subset of {POS,VEL,FF,VOLTAGE}; controller inputs require K>0; gains kp/ki/kd require a controller input; ki requires POS; slew requires controller input.
- SO3: gain and bias both SO3; dyntype NONE or INTEGRATOR (actdim 3); `gainprm[0] == −biasprm[1]`; SITE (with refsite) or ball JOINT transmission (compiled as `mjTRN_SO3`); ctrlspec EXPMAP (3 ctrl) or QUAT (4 ctrl, dyntype NONE only); outnum=3.
- `actdim` default: 1 for stateful standard dyntypes, 0 for NONE/DCMOTOR; actdim>1 only for USER/DCMOTOR/PID/SO3.
- `actrange` with dyntype NONE is an error; `inheritrange` for position-like actuators sets `ctrlrange` (NONE/FILTEREXACT/PID) or `actrange` (INTEGRATOR) to `mean ± 0.5·(hi−lo)·inheritrange` of the hinge/slide joint or tendon range.
- muscle parameter checks: `range[0]<range[1]`, `lmin<1<lmax`, `scale,vmax,fpmax,fvmax > 0`.
- delay>0 requires nsample>0; history requires ctrlnum>0.

Shortcut presets (`mjs_setTo*`; `SetControl` zeros prm arrays then sets `gainprm[0]=1, dynprm[0]=1`, FIXED/NONE/NONE, actearly=0):
- motor: defaults.
- position: `gainprm[0]=kp, biasprm[1]=−kp, biasprm[2] = −kv` or `+dampratio` (positive ⇒ converted at compile by `setDampRatio`), timeconst>0 ⇒ FILTEREXACT with dynprm[0]=timeconst; ctrlspec=POS; AFFINE bias.
- intvelocity: position + INTEGRATOR, ctrlspec=VEL, inheritrange applies to actrange.
- velocity: `gainprm[0]=kv, biasprm[2]=−kv`, ctrlspec VEL.
- damper: `gainprm = [0,0,−kv]`, AFFINE gain, ctrllimited TRUE, ctrlrange must be ≥0.
- cylinder: FILTER, dynprm[0]=timeconst, gainprm[0]=area (or π/4·d²), biasprm[0]=bias, ctrlspec PRESSURE, AFFINE bias.
- muscle: dynprm=[0.01, 0.04, tausmooth]; gainprm=biasprm=[0.75, 1.05, −1, 200, 0.5, 1.6, 1.5, 1.3, 1.2] (range0, range1, force, scale, lmin, lmax, vmax, fpmax, fvmax) overridden by any non-negative argument; MUSCLE dyn/gain/bias.
- adhesion: gainprm[0]=gain, ctrllimited TRUE, ctrlrange ≥ 0 (trntype BODY).
- orientation: SO3 gain/bias, kp, biasprm[1]=−kp, biasprm[2]=−kv or dampratio.
- pid: biasprm[1]=−kp, biasprm[2]=−kv|dampratio, gainprm[0]=ki, dynprm[1]=slewmax, dynprm[0]=imax (only if ki), dyntype PID iff ki or slewmax.

`setDampRatio` (compile time, `mj_setCompiledConst`): for actuators with `gainprm[0] == −biasprm[1]` (or PID) and `biasprm[2] > 0`: `w = mean_k (moment_kᵀ M⁻¹ moment_k)` over outputs at qpos0; `mass = (w > mjMINVAL) ? 1/w : 0`; `kp = PID ? −biasprm[1] : gainprm[0]`; `biasprm[2] = −dampratio·2·sqrt(kp·mass)`.

`actuator_acc0[out] = ‖M⁻¹·moment_row(out)‖₂` at qpos0 (`mj_solveM`), 0 if nv==0.

---

## 3. Muscle model (engine_util_misc.c) — exact formulas

Parameter vector `prm = gainprm` (or `biasprm`): `[range0, range1, force, scale, lmin, lmax, vmax, fpmax, fvmax]`.

### 3.1 `mju_muscleGainLength(L, lmin, lmax)` (active FL curve)
```
if !(lmin <= L <= lmax): return 0
a = 0.5*(lmin+1);  b = 0.5*(1+lmax)
L <= a : x = (L−lmin)/max(mjMINVAL, a−lmin);  return 0.5x²
L <= 1 : x = (1−L)/max(mjMINVAL, 1−a);        return 1 − 0.5x²
L <= b : x = (L−1)/max(mjMINVAL, b−1);        return 1 − 0.5x²
else   : x = (lmax−L)/max(mjMINVAL, lmax−b);  return 0.5x²
```

### 3.2 `mju_muscleGain(len, vel, lengthrange, acc0, prm)` and `mju_muscleBias`
```
F0 = (prm[2] < 0) ? prm[3]/max(mjMINVAL, acc0) : prm[2]           (force; negative ⇒ auto-scale)
L0 = (LR[1]−LR[0]) / max(mjMINVAL, range1−range0)                    (optimal length)
L  = range0 + (len − LR[0]) / max(mjMINVAL, L0)                      (normalized length)
V  = vel / max(mjMINVAL, L0*vmax)                                    (normalized velocity)
FL = muscleGainLength(L, lmin, lmax)
y = fvmax − 1
FV = V <= −1 : 0
     V <= 0  : (V+1)²
     V <= y  : fvmax − (y−V)²/max(mjMINVAL, y)
     else    : fvmax
gain = −F0 · FL · FV
```
Bias (passive, uses only `range, force, scale, lmax, fpmax`):
```
b = 0.5*(1+lmax)
L <= 1 : 0
L <= b : x = (L−1)/max(mjMINVAL, b−1);  bias = −F0·fpmax·0.5x²
else   : x = (L−b)/max(mjMINVAL, b−1);  bias = −F0·fpmax·(0.5 + x)
```
Muscle force: `force = gain·act + bias` (act = activation, from `act` or next activation if actearly). Force is negative (pulling) for positive activation; gear/moment maps it.

### 3.3 Activation dynamics `mju_muscleDynamics(ctrl, act, prm=[tau_act, tau_deact, smoothing_width])`
```
u  = clip(ctrl, 0, 1)
ac = clip(act, 0, 1)
tau_act   = prm[0]·(0.5 + 1.5·ac)
tau_deact = prm[1] / (0.5 + 1.5·ac)
dctrl = u − act                 (NOTE: uses the UNclamped act)
tau = muscleDynamicsTimescale(dctrl, tau_act, tau_deact, prm[2])
return dctrl / max(mjMINVAL, tau)
```
`mju_muscleDynamicsTimescale(dctrl, ta, td, w)`: if `w < mjMINVAL`: `tau = dctrl > 0 ? ta : td` (note `dctrl == 0` ⇒ td); else `tau = td + (ta−td)·sigmoid(dctrl/w + 0.5)`.
`mju_sigmoid(x)`: `x<=0 → 0`, `x>=1 → 1`, else `x³·(3x·(2x−5) + 10)` (= 6x⁵−15x⁴+10x³; this exact Horner order).

Muscle activation is then Euler-integrated (`act += act_dot·h`), and clamped to actrange only if `actuator_actlimited` (the `mjs_setToMuscle` preset does NOT set an actrange; `actlimited` stays AUTO ⇒ limited only if the user gives `actrange`). Without a clamp, `act` can leave [0,1]; the dynamics then still use `clip(act,0,1)` for timescales but the unclamped `act` in `dctrl` and in the force `gain·act`.

### 3.4 Length range: `mj_setLengthRange(m, d, index, opt, error, error_sz)` (engine_setconst.c)
Called at compile time by `mjCModel::LengthRange` for every actuator after setting `m->opt.disableflags = FRICTIONLOSS|CONTACT|SPRING|DAMPER|GRAVITY|ACTUATION` and `m->opt.timestep = LRopt.timestep` if > 0 (restored afterwards). May be multithreaded (one mjData per thread; actuators split in contiguous chunks of `ceil(nactuator/nthread)`); results are independent per actuator.
```
ismuscle = gaintype==MUSCLE || biastype==MUSCLE ; isuser = gaintype==USER || biastype==USER
skip (return 1) if mode==NONE, or mode==MUSCLE && !ismuscle, or mode==MUSCLEUSER && !ismuscle && !isuser
if useexisting && LR[2out] < LR[2out+1]: return 1
if uselimit:
   JOINT/JOINTINPARENT with jnt_limited: LR = jnt_range; return 1
   TENDON with tendon_limited: LR = tendon_range; return 1
for side in {0,1}:
   mj_resetData(m,d); updated=0; lmin[side]=lmax[side]=0
   while d->time < inttotal:
       len = evalAct(m,d,index,side,opt)
       if d->time == 0: error "Unstable lengthrange simulation"; return 0   (step reset by instability)
       if d->time > inttotal − interval:
           if len < lmin[side] || !updated: lmin[side] = len
           if len > lmax[side] || !updated: lmax[side] = len
           updated = 1
   LR[2*index + side] = side==0 ? lmin[0] : lmax[1]     (NOTE: indexed by `index`, not `out`)
dif = LR[1]−LR[0] (out-indexed); dif <= 0 ⇒ error
if lmax[0]−lmin[0] > tolrange*dif or lmax[1]−lmin[1] > tolrange*dif ⇒ "did not converge" error
```
`evalAct(m,d,index,side,opt)`:
```
qvel *= exp(−timestep / max(0.01, opt.timeconst))
mj_step1(m,d)
moment = dense row(out)
qfrc_applied = M⁻¹·moment (solveM);  nrm = ‖qfrc_applied‖
qfrc_applied = moment · (2side−1)·accel / max(mjMINVAL, nrm)
nrm = ‖qfrc_applied‖; if maxforce>0 && nrm>maxforce: qfrc_applied *= maxforce/max(mjMINVAL,nrm)
mj_step2(m,d)
return actuator_length[out]
```
(Force direction `moment` scaled so that the resulting unconstrained acceleration magnitude would be `accel`.)

---

## 4. Activation integration and history advancement: `advanceStart` (engine_forward.c)

Called by every integrator (Euler, RK4 via `mj_advance`, implicit, discrete) at the PRE-step state, with the integrator's `act_dot` (RK4: weighted RK combination):
1. **History** (if `nhistory > 0`):
   - For each actuator with `nsample>0`: `slot = mju_historyInsert(buf, n, ctrlnum, d->time)`; copy **raw `d->ctrl`** (unclamped, undelayed) into slot.
   - For each sensor with `nsample>0`, `delay = sensor_delay`, `interval = sensor_interval[2i]`:
     - interval > 0: `time_prev = buf[0]`; if `time_prev + interval <= d->time`: `buf[0] += interval` (exact continuous ticks); `slot = insert(buf, n, dim, d->time)`; if delay>0: `mj_computeSensor(m,d,i,slot)` (fresh compute, cutoff applied) else copy current `sensordata` segment.
     - else if delay > 0: `slot = insert(...)`; `mj_computeSensor(m,d,i,slot)`.
     - else (history-only): `slot = insert(...)`; copy `sensordata`.
   (User and plugin sensors with history: `mj_computeSensor` errors for USER/PLUGIN stage types — `mj_computeSensor` dispatches on needstage to the builtin switch, which raises "invalid sensor type" for USER/PLUGIN; the compiler permits nsample on them, so avoid delay>0 there.)
2. **Activations** (if `na && !DISABLED(ACTUATION)`): for every actuator, every slot j in its block: `act[j] = mj_nextActivation(m, d, i, j, mj_actuatorDisabled(m,i) ? 0 : act_dot[j])` (slots processed in increasing j; each uses `d->act[j]` before overwrite — slots are independent).
3. **Rotational setpoint re-anchoring** for INTEGRATOR actuators: if `period = wrapPeriod>0`: `adr = actadr+actnum−1`, `act[adr] = wrapSetpoint(act[adr], actuator_length[outadr], period)`; else if SO3 gain: `angle = ‖act[adr:adr+3]‖`; if `angle > π`: `scale = (angle − 2π·round(angle/2π))/angle`, `act *= scale`.
4. Sleep (`mj_sleep`, may call `mj_forwardSkip(POS, 0)` and `mj_updateSleep`).

Then the integrator writes qvel/qpos, and `advanceFinish` does `time += timestep`, plugin advance, plasticity.

### 4.1 History ring-buffer functions (exact)
```
physical(cursor, n, logical) = (cursor + 1 + logical) % n          (logical 0 = oldest, n−1 = newest)

findIndex(times, n, cursor, t):                     smallest logical i with times[i] >= t
   t_old = times[phys(0)], t_new = times[phys(n−1)]
   if t <= t_old: return 0
   if t > t_new: return n
   lo=0, hi=n−1; while hi−lo > 1: mid=(lo+hi)/2; if times[phys(mid)] < t: lo=mid else hi=mid
   return hi

historyInsert(buf, n, dim, t) -> pointer to value slot:
   i = findIndex(...)
   if i < n and |t − times[phys(i)]| < mjMINVAL: return values[phys(i)]       (overwrite existing)
   if i == 0:  times[phys(0)] = t; return values[phys(0)]                     (older than oldest: replace oldest)
   if i == n:  cursor = (cursor+1)%n; buf[1]=cursor; times[cursor]=t; return values[cursor]
   else (out of order): for j in 0..i−2: copy logical j+1 → j (times and values);
               times[phys(i−1)] = t; return values[phys(i−1)]

historyRead(buf, n, dim, res, t, interp) -> pointer or NULL:
   if t <= t_old + mjMINVAL: return values[phys(0)]
   if t >= t_new − mjMINVAL: return values[phys(n−1)]
   i = findIndex; if |t − times[phys(i)]| < mjMINVAL: return values[phys(i)]
   lo = phys(i−1), hi = phys(i)
   interp 0 (ZOH): return values[lo]
   dt = times[hi]−times[lo]; α = (t − times[lo])/dt
   interp 1: res = v_lo + α(v_hi − v_lo); return NULL
   interp 2 (cubic Hermite / Catmull–Rom):
       h00=2α³−3α²+1, h10=α³−2α²+α, h01=−2α³+3α², h11=α³−α²
       m_lo = (i > 1)   ? (v_hi − v[phys(i−2)]) / (times[hi] − times[phys(i−2)]) : 0
       m_hi = (i < n−1) ? (v[phys(i+1)] − v_lo) / (times[phys(i+1)] − times[lo]) : 0
       res = h00 v_lo + h10 dt m_lo + h01 v_hi + h11 dt m_hi; return NULL
```
`mj_readCtrl(m,d,id,time,result,interp)`: nsample==0 ⇒ returns `d->ctrl+ctrladr`; interp<0 ⇒ model interp; reads at `time − actuator_delay`. `mj_readSensor` analogous with `sensor_delay`, returns `d->sensordata+adr` if no history.

`mju_historyInit(buf, n, dim, times, values, user)`: requires strictly increasing times (diff ≥ mjMINVAL, else mjERROR); `buf[0]=user; buf[1]=n−1`; times copied to physical 0..n−1; values copied if given. `mj_initCtrlHistory` preserves buf[0]; `mj_initSensorHistory` sets buf[0]=phase argument.

### 4.2 Reset initialization (`_resetData`, engine_io.c)
- Warn if `nhistory && timestep <= 0`.
- Actuator buffers: `buf[0]=0`, `buf[1]=n−1`, `times[j] = −(n−j)·dt` (j=0..n−1 ⇒ −n·dt … −dt), values 0; SO3 QUAT actuators: `values[4j] = 1` (identity quaternions).
- Sensor buffers: `period = interval[0]`, `phase = interval[1]`; `buf[0] = period>0 ? (phase==0 ? −period : phase) : −dt`; `buf[1] = n−1`; times: if period>0: `t0 = phase==0 ? −period : phase`, `times[j] = ceil((t0 − (n−1−j)·period)/dt)·dt` (dt>0), else `−(n−j)·dt`; values 0.

Effect: with `delay = n·dt` and interp ZOH, the delayed read at time t returns exactly the ctrl written n steps earlier (reads happen before writes, so the minimum effective delay is one step).

---

## 5. Sensors

### 5.1 Compile-time properties (`mjCSensor::Compile`)

**datatype** (`sensorDatatype`): POSITIVE: TOUCH, INSIDESITE (non-enclosed). AXIS: FRAMEXAXIS/Y/Z, GEOMNORMAL. QUATERNION: BALLQUAT, FRAMEQUAT. REAL: everything else (enclosed INSIDESITE forced REAL). USER keeps user datatype.

**needstage** (`sensorNeedstage`):
- ACC: TOUCH, ACCELEROMETER, FORCE, TORQUE, ACTUATORFRC, JOINTACTFRC, TENDONACTFRC, JOINTLIMITFRC, TENDONLIMITFRC, FRAMELINACC, FRAMEANGACC, (CONTACT default; overridden).
- VEL: VELOCIMETER, GYRO, JOINTVEL, TENDONVEL, ACTUATORVEL, BALLANGVEL, JOINTLIMITVEL, TENDONLIMITVEL, FRAMELINVEL, FRAMEANGVEL, SUBTREELINVEL, SUBTREEANGMOM, TACTILE.
- POS: MAGNETOMETER, RANGEFINDER, CAMPROJECTION, JOINTPOS, TENDONPOS, ACTUATORPOS, BALLQUAT, JOINTLIMITPOS, TENDONLIMITPOS, FRAMEPOS, FRAMEQUAT, FRAMEX/Y/ZAXIS, SUBTREECOM, INSIDESITE, GEOMDIST, GEOMNORMAL, GEOMFROMTO, E_POTENTIAL, E_KINETIC, CLOCK.
- CONTACT: `need_acc = FORCE|TORQUE bits || reduce ∈ {2,3}`; `need_vel = LINVEL|ANGVEL bits`; stage = ACC if need_acc else VEL if need_vel else POS; `reduce==3 && need_vel` is an error.
- USER: user-specified; PLUGIN: `plugin->needstage`.

**dim** (`mjs_sensorDim`): 1: TOUCH, JOINTPOS/VEL, TENDONPOS/VEL, ACTUATORPOS/VEL/FRC (then overridden to `outnum`), JOINTACTFRC, TENDONACTFRC, all LIMIT*, GEOMDIST, INSIDESITE, E_POTENTIAL, E_KINETIC, CLOCK. 2: CAMPROJECTION. 3: ACCELEROMETER, VELOCIMETER, GYRO, FORCE, TORQUE, MAGNETOMETER, BALLANGVEL, FRAMEPOS, FRAMEX/Y/ZAXIS, FRAMELINVEL/ANGVEL/LINACC/ANGACC, SUBTREECOM/LINVEL/ANGMOM, GEOMNORMAL. 4: BALLQUAT, FRAMEQUAT. 6: GEOMFROMTO. CONTACT: `intprm[2]·condataSize(intprm[0])`. TACTILE: `3·mesh_nvert`. RANGEFINDER: `raydataSize(intprm[0]) · (site ? 1 : cam_res[0]·cam_res[1])`. USER: user dim. PLUGIN: `plugin->nsensordata`.

**intprm**: RANGEFINDER `[dataspec]` (default `1<<DIST`; XML `data` keys must be in enum order). CONTACT `[dataspec (default 1<<FOUND), reduce (0 none,1 mindist,2 maxforce,3 netforce), num (≥1, default 1)]`. INSIDESITE `[enclosed flag]`.

**Validation**: noise ≥0, cutoff ≥0, interval[0] ≥0, phase ≤0 and > −period; delay>0 needs nsample>0; cutoff>0 forbidden for QUATERNION or AXIS datatypes except GEOMNORMAL. Object-type constraints: TOUCH/ACC/VEL/GYRO/FORCE/TORQUE/MAGNETOMETER/CAMPROJECTION → site (camprojection needs ref camera with nonzero resolution); RANGEFINDER → site or camera; JOINTPOS/VEL/JOINTACTFRC → hinge/slide joint; BALL* → ball joint; LIMIT* → limited joint/tendon; FRAME* → (x)body/geom/site/camera; SUBTREE* → body; GEOM* → body/geom (distinct, no hfield); CONTACT obj → site/body/xbody/geom/unknown, ref → body/xbody/geom/unknown; TACTILE obj mesh, ref geom; reference frame objects must be (x)body/geom/site/camera.

**Gauge scaling** (`ApplyGauge`, compile-time): `cutoff` and `noise` are multiplied by a dimension factor of the object's gauge (Force/Shape factors). Only affects stored `sensor_cutoff/noise` values; read them from mjModel.

**noise**: stored in `sensor_noise`, **never applied** by the engine (documentation: "does not affect the simulation").

### 5.2 `apply_cutoff(m, i, data)`
```
cutoff = sensor_cutoff[i]; if cutoff <= 0: return
if type ∈ {CONTACT, GEOMFROMTO}: return
for j < dim: REAL: data[j] = clip(data[j], −cutoff, cutoff); POSITIVE: data[j] = min(cutoff, data[j])
(AXIS/QUATERNION: untouched — GEOMNORMAL is AXIS, so its cutoff only acts as the distance bound)
```
Applied after every builtin computation (`mj_computeSensor`), after user callback (per stage, for USER sensors of that stage), after plugin compute (for that plugin's sensors of that stage).

### 5.3 Stage drivers `mj_sensorPos / mj_sensorVel / mj_sensorAcc`
```
if DISABLED(SENSOR): return                        (sensordata keeps stale values!)
sleep_filter = ENABLED(SLEEP) && nbody_awake < nbody
for i in 0..nsensor−1:
   if sleep_filter && mj_sleepState(SENSOR,i)==ASLEEP: continue      (stale value kept)
   if type == PLUGIN: continue
   if needstage[i] == STAGE:
      if type == USER:
          [VEL: if !flg_subtreevel: mj_subtreeVel] [ACC: if !flg_rnepost: mj_rnePostConstraint]
          zero(sensordata+adr, dim); nuser++
      else compute_or_read_sensor(m,d,i, sensordata+adr)
if nuser: mjcb_sensor(m,d,STAGE) (if set); apply cutoff to USER sensors of this stage
compute_plugin_sensors(m,d,STAGE)
```
(Pos-stage loop checks PLUGIN before sleep; vel/acc check sleep first — no behavioral difference.)

`compute_or_read_sensor`:
```
n = sensor_history[2i]; if n <= 0: mj_computeSensor; return
if delay > 0: ptr = mj_readSensor(m,d,i,d->time, sensordata, interp); if ptr: copy; return
if interval > 0:
   if buf[0] + interval <= d->time: mj_computeSensor
   else: ptr = mj_readSensor(... d->time ...) (delay==0 ⇒ reads at d->time: newest sample ≤ t, ZOH/interp)
   return
mj_computeSensor          (history-only)
```
`mj_computeSensor(m,d,i,out)`: dispatch on `sensor_needstage[i]` to `mj_computeSensorPos/Vel/Acc`, then `apply_cutoff`.

`compute_plugin_sensors(stage)`: if `nplugin==0` return. For each plugin instance whose plugin has `mjPLUGIN_SENSOR` and (`plugin->needstage == stage` or (`stage==POS` and needstage==NONE)): if stage==VEL && !flg_subtreevel → `mj_subtreeVel`; if stage==ACC && !flg_rnepost → `mj_rnePostConstraint` (called regardless of need); `plugin->compute(m,d,instance,mjPLUGIN_SENSOR)`; cutoff on its sensors with matching needstage.

Lazy flags: `flg_energypos` cleared at start of `mj_fwdPosition`; `flg_subtreevel, flg_energyvel` cleared at start of `mj_fwdVelocity`; `flg_rnepost` cleared right before `mj_sensorAcc` in `mj_forwardSkip`/`mj_step2`. VEL-stage builtin: `mj_subtreeVel` called if needed for SUBTREELINVEL/SUBTREEANGMOM; ACC-stage builtin: `mj_rnePostConstraint` if needed for ACCELEROMETER, FORCE, TORQUE, FRAMELINACC, FRAMEANGACC.

### 5.4 Helper functions

- `get_xpos_xmat(type,id)`: XBODY→(`xpos`,`xmat`); BODY→(`xipos`,`ximat`); GEOM→`geom_xpos/xmat`; SITE→`site_xpos/xmat`; CAMERA→`cam_xpos/xmat`.
- `get_xquat(type,id)`: XBODY→`xquat`; BODY→`xquat*body_iquat`; GEOM→`xquat[geom_body]*geom_quat`; SITE→`xquat[site_body]*site_quat`; CAMERA→`xquat[cam_body]*cam_quat` (quaternion products, not mat2Quat).
- `mj_objectVelocity(type,id,res,flg_local)`: (bodyid,pos,rot) per type as above (rot only if local); if `body_dofnum[body_weldid[bodyid]] == 0` → zero; else `res = transformSpatial(cvel[bodyid], flg_force=0, newpos=pos, oldpos=subtree_com[body_rootid[bodyid]], rot)`, i.e. `ang = ω`, `lin = v − (pos−com)×ω`, then rotated by rotᵀ if local. Output `[ang(3), lin(3)]`.
- `mj_objectAcceleration(type,id,res,flg_local)`: same zero rule; `res = transformSpatial(cacc[bodyid], 0, pos, com, rot)`; `vel = transformSpatial(cvel[bodyid], 0, pos, com, rot)`; `res[3:6] += vel[0:3] × vel[3:6]` (Coriolis/centripetal correction; computed in the output frame).
- `mju_transformSpatial(res, vec, flg_force, newpos, oldpos, rot)`: `dif = newpos − oldpos`; motion: `tran = [vec[0:3], vec[3:6] − dif×vec[0:3]]`; force: `tran = [vec[0:3] − dif×vec[3:6], vec[3:6]]`; if rot: `res = [rotᵀ tran[0:3], rotᵀ tran[3:6]]`.
- `mj_contactForce(m,d,id,result[6])`: zero; if valid id and `efc_address>=0`: pyramidal ⇒ `mju_decodePyramid(result, efc_force+adr, friction, dim)` (normal = Σ all 2(dim−1) pyramid forces; dim==1 ⇒ efc_force[adr]; tangent_i = (p0_i − p1_i)·mu_i); elliptic ⇒ copy `dim` entries; then `result[0] −= con->adhesion` (net interface force). Contact-frame coordinates `[normal, t1, t2, torsion, roll1, roll2]`.

### 5.5 Position-stage sensors (`mj_computeSensorPos`)

| type | output |
|---|---|
| MAGNETOMETER | `site_xmatᵀ · opt.magnetic` |
| CAMPROJECTION | `cam_project(site_xpos[obj], cam ref)` — below |
| RANGEFINDER | below |
| JOINTPOS | `qpos[jnt_qposadr]` |
| TENDONPOS | `ten_length` |
| ACTUATORPOS | `actuator_length[outadr .. +dim]` |
| BALLQUAT | `normalize4(qpos[adr:adr+4])` (mju_normalize4: if norm < mjMINVAL → (1,0,0,0)) |
| JOINTLIMITPOS / TENDONLIMITPOS | 0; scan efc rows `j ∈ [ne+nf, nefc)` for first row with `efc_type == LIMIT_JOINT/LIMIT_TENDON` and `efc_id == obj`: `efc_pos[j] − efc_margin[j]` (break at first) |
| FRAMEPOS | refid==−1: `xpos`; else `xmat_refᵀ(xpos − xpos_ref)` |
| FRAMEX/Y/ZAXIS | column `offset` of xmat (`xmat[offset], xmat[offset+3], xmat[offset+6]`); ref: `xmat_refᵀ·axis` |
| FRAMEQUAT | `q_obj`; ref: `conj(q_ref) * q_obj` (mju_negQuat then mulQuat; no normalization) |
| SUBTREECOM | `subtree_com[obj]` |
| INSIDESITE | below |
| GEOMDIST / GEOMNORMAL / GEOMFROMTO | below |
| E_POTENTIAL | `if !flg_energypos: mj_energyPos`; `energy[0]` |
| E_KINETIC | `if !flg_energyvel: mj_energyVel`; `energy[1]` (pos-stage, but velocity stage hasn't run yet in this step's `mj_fwdVelocity` → computes with current qvel and the newly computed M; note `flg_energyvel` will be cleared by the subsequent `mj_fwdVelocity`) |
| CLOCK | `d->time` |
| CONTACT (stage POS) | `compute_contact_sensor` |

**CAMPROJECTION** (`cam_project`): `fx,fy = (sensorsize[0]&&sensorsize[1]) ? (intrinsic[0]/sensorsize[0]·res[0], intrinsic[1]/sensorsize[1]·res[1]) : 0.5/tan(fovy·π/360)·res[1]` (both); `p = cam_xmatᵀ(target − cam_xpos)`; `denom = p[2]`; if `|denom| < mjMINVAL`: `denom = denom<0 ? min(denom, −mjMINVAL) : max(denom, mjMINVAL)`; `out[0] = −fx·p[0]/denom + 0.5·res[0]`, `out[1] = fy·p[1]/denom + 0.5·res[1]`. (Principal point = image center; `intrinsic[2:4]` NOT used here.)

**RANGEFINDER**:
- Site: direction `rvec` = site z-axis (column 2 of site_xmat), origin `site_xpos`; `dist = mj_ray(m,d,origin,rvec, geomgroup=NULL, flg_static=1, bodyexclude=site_bodyid, &geomid, normal?)`; normal requested only if NORMAL bit; `fill_raydata(..., cam_z=NULL)` (depth = dist).
- Camera: `width,height = cam_resolution`; `bodyexclude = cam_bodyid`; `cam_z = column 2 of cam_xmat`; `mju_camIntrinsics`: PERSPECTIVE: sensorsize nonzero ⇒ `fx=intr[0]/ss[0]·W, fy=intr[1]/ss[1]·H, cx=intr[2]/ss[0]·W, cy=intr[3]/ss[1]·H`; else `fx=fy=0.5/tan(fovy·π/360)·H, cx=W/2, cy=H/2`. ORTHOGRAPHIC: `fx=cx=W/2, fy=cy=H/2`; `extent = cam_fovy` (ortho full height).
  `mju_camPixelRay(col,row)`: `px = col+0.5−cx`, `py = row+0.5−cy`; perspective: origin=cam_xpos, `dir = normalize(cam_xmat·(px/fx, −py/fy, −1))`; ortho: `dir = −z-axis`, `origin = cam_xpos + cam_xmat·(px/fx·extent/2, −py/fy·extent/2, 0)`.
  Perspective: all rays via `mj_multiRay(m,d,cam_xpos, vec, NULL, 1, bodyexclude, geomid, dist, normals, npixel, cutoff=mjMAXVAL)`; ortho: per-pixel `mj_ray`. Pixel order row-major (row 0 = top), outputs concatenated per pixel.
- `fill_raydata(ptr, spec, dist, origin, dir, normal, cam_xpos, cam_z)`: `hit = dist >= 0` (miss ⇒ −1). Fields written in enum order: DIST: `dist`; DIR: hit ? dir : 0; ORIGIN: origin (always); POINT: hit ? origin + dist·dir : 0; NORMAL: hit ? normal : 0; DEPTH: !hit ⇒ −1; camera ⇒ `−(point − cam_xpos)·cam_z`; site ⇒ `dist`.
- Cutoff (REAL) then clamps every component (including −1 miss values, unless cutoff < 1).

**INSIDESITE**:
- enclosed (`intprm[0]`): Hausdorff-distance mode using CCD: `obj2 = site(refid)`; `niter = opt.ccd_iterations, tol = opt.ccd_tolerance`; GEOM/SITE: `mjc_hausdorff(obj1, obj2, niter, 0.5, tol)`; BODY: max over its geoms; XBODY: max over all geoms in subtree (body walk `while b > objid: b = parent`). Initial `−mjMAXVAL`.
- otherwise: point = `get_xpos` (BODY: `xipos`; but if `objid>0 && body_mass < mjMINVAL && body_subtreemass >= mjMINVAL` ⇒ use `subtree_com`); `mj_insideSite(refid, point)` ∈ {0,1}.
- `mj_insideSite`: mesh site: rotate into site frame, then into canonical mesh frame (`mesh_pos/quat`), bounding box `|p_k| > size_k` ⇒ 0, then all convex-hull polygon planes: `n·p − n·v0 > 0` ⇒ 0; else 1 (no mesh/polys ⇒ 0). Primitive: `mju_insidePrimitive`: sphere `|v|² < r²`; capsule: `x²+y²+(z−clip(z,−h,h))² < r²`; ellipsoid `Σ p_k²/s_k² < 1`; cylinder `|z|<h && x²+y²<r²`; box all `|p_k| < s_k`; plane `z < 0`; strict inequalities.

**GEOMDIST / GEOMNORMAL / GEOMFROMTO**:
```
cutoff = sensor_cutoff[i]; dist = cutoff; fromto = 0
geoms1 = BODY ? [body_geomadr .. +body_geomnum) : [objid]; geoms2 likewise for ref
for g1 in geoms1: for g2 in geoms2:            (row-major order; strict "<" keeps the first minimum)
    fromto_new = 0; d_new = mj_geomDistance(m,d,g1,g2, cutoff, fromto_new)
    if d_new < dist: dist = d_new; fromto = fromto_new
GEOMDIST:   dist
GEOMNORMAL: n = fromto[3:6]−fromto[0:3]; if any nonzero: normalize3(n); output n
GEOMFROMTO: fromto (no cutoff clamp)
```
`mj_geomDistance(m,d,g1,g2,distmax,fromto)`: `flip = geom_type[g1] > geom_type[g2]` (swap so lower type first); `func = mjCOLLISIONFUNC[t1][t2]`; none ⇒ return distmax; if `!DISABLED(NATIVECCD)` and func ∈ {mjc_Convex, mjc_BoxBox} ⇒ `mj_geomDistanceCCD(m,d,geom1,geom2,distmax,fromto)` (unswapped ids); else `num = func(m,d,con,g1,g2, margin=distmax)` (up to mjMAXCONPAIR=50 contacts), take smallest `dist_i < dist` (strict, first wins); if found and fromto: `sign = flip ? −1 : 1`; `fromto[0:3] = pos − 0.5·sign·dist·normal`, `fromto[3:6] = pos + 0.5·sign·dist·normal`. Hence with dist ≥ cutoff nothing is found: GEOMDIST reports `cutoff` (0 by default, so only penetrations are seen), normal/fromto zero.

**`mj_energyPos`** (sets `flg_energypos=1`):
```
energy[0] = 0
if !DISABLED(GRAVITY): for b=1..nbody−1: energy[0] −= body_mass[b] · (gravity · xipos[b])
sleep_filter = ENABLED(SLEEP) && nbody_awake < nbody
if !DISABLED(SPRING):
  joints (by body, skipping bodies with body_awake != AWAKE under sleep filter):
    stiffness = jnt_stiffness; poly = jnt_stiffnesspoly (mjNPOLY); skip if both zero
    qspring = jnt_plasticadr>=0 ? d->plastic+plasticadr : qpos_spring+qposadr
    FREE:  x = ‖qpos[0:3] − qspring[0:3]‖; E += polyPotential(k, poly, x, n, 0); then fall through with adr+=3
    BALL:  dif = subQuat(qpos[adr:adr+4] (UNNORMALIZED raw qpos — the normalized copy is unused), qspring)
           x = ‖dif‖; E += polyPotential(k,poly,x,n,0)
    SLIDE/HINGE: x = qpos − qspring; E += polyPotential(k,poly,x,n,0)
  tendons (skip if sleep-filtered and not AWAKE):
    ls = tendon_plasticadr>=0 ? plastic : tendon_lengthspring[2i..]; x = L>hi ? L−hi : L<lo ? L−lo : 0
    E += polyPotential(tendon_stiffness, tendon_stiffnesspoly, x, n, 0)
  flex (dim<=1, not rigid, edgestiffness != 0): for non-rigid edges: E += 0.5·k·(length0 − length)²
```
`mju_polyPotential(linear, poly, x, n, flg_odd)`: `x = flg_odd ? |x| : x`; `res = 0.5·linear·x²`; `xpow = x`; for i<n: `xpow *= x; res += poly[i]/(i+3)·(xpow·x)` (i.e., poly[i]·x^{i+3}/(i+3)).

**`mj_energyVel`**: `vec = M·qvel` (`mj_mulM`); `energy[1] = 0.5·dot(vec, qvel)`; `flg_energyvel=1`.

Energy outside sensors: in `mj_forwardSkip` after `mj_sensorPos`: if `!flg_energypos`: `ENABLED(ENERGY) ? mj_energyPos : energy[0]=energy[1]=0`; after `mj_sensorVel`: `if ENABLED(ENERGY) && !flg_energyvel: mj_energyVel`.

### 5.6 Velocity-stage sensors (`mj_computeSensorVel`)

| type | output |
|---|---|
| VELOCIMETER | `objectVelocity(SITE, local=1)[3:6]` |
| GYRO | `objectVelocity(SITE, local=1)[0:3]` |
| JOINTVEL | `qvel[jnt_dofadr]` |
| TENDONVEL | `ten_velocity` |
| ACTUATORVEL | `actuator_velocity[outadr .. +dim]` |
| BALLANGVEL | `qvel[dofadr .. +3]` |
| JOINTLIMITVEL / TENDONLIMITVEL | first matching limit row: `efc_vel[j]`, else 0 |
| FRAMELINVEL / FRAMEANGVEL | below |
| SUBTREELINVEL | `subtree_linvel[obj]` (after `mj_subtreeVel`) |
| SUBTREEANGMOM | `subtree_angmom[obj]` |
| CONTACT (stage VEL) | `compute_contact_sensor` |
| TACTILE | below |

**FRAMELINVEL/ANGVEL**: `xvel = objectVelocity(obj, local=0)`. If ref: `xvel_ref = objectVelocity(ref, 0)`; `rel = xvel − xvel_ref` (6-vector); `rel[3:6] += (xpos − xpos_ref) × xvel_ref[0:3]`; `xvel = [xmat_refᵀ rel[0:3], xmat_refᵀ rel[3:6]]`. Output lin `xvel[3:6]` or ang `xvel[0:3]`. Positions from `get_xpos_xmat` (BODY uses xipos).

**`mj_subtreeVel`** (sleep-aware over awake bodies; sets flg_subtreevel):
```
for b: body_vel[b] = objectVelocity(BODY, b, local=0)   ([ω, v_com])
       subtree_linvel[b] = mass_b · v_com
       subtree_angmom[b] = ximat · diag(inertia) · ximatᵀ · ω
for b = nbody−1..0: if b: subtree_linvel[parent] += subtree_linvel[b]
                    subtree_linvel[b] /= max(mjMINVAL, subtreemass_b)     (momentum→velocity, after children added)
for b = nbody−1..1: dx = xipos_b − subtree_com_b; dv = v_com,b − subtree_linvel_b
                    angmom_b += dx × (mass_b·dv); angmom_parent += angmom_b
                    dx = com_b − com_parent; dv = (linvel_b − linvel_parent)·subtreemass_b
                    angmom_parent += dx × dv
```

**TACTILE** (obj = mesh id, ref = geom id; dim = nchannel·nvert with nchannel = dim / mesh_vertnum = 3):
```
parent_weld = body_weldid[geom_bodyid[geom_id]]
zero(sensordata, dim)
contact geoms: for each contact k: body_s = body_weldid of side s body (flex: mj_flexBody)
   if body1 == parent_weld && 0<=g2<ngeom && g2 != geom_id && !seen[g2]: add g2   (dedup, contact order)
   if body2 == parent_weld && 0<=g1<ngeom && g1 != geom_id && !seen[g1]: add g1
if none: done
has_frame = mesh_normalnum == 3·mesh_vertnum ; normal_stride = has_frame ? 9 : 3
for taxel j (mesh vertex):
   xpos = geom_xmat·vert_j + geom_xpos
   for each contact geom g (≠ geom_id):
      sdf object: SDF-type geom → plugin sdf; MESH → geom_dataid (skip if mesh_octadr == −1); else geom primitive
      lpos = geom_xmat[g]ᵀ (xpos − geom_xpos[g]); if SDF plugin: lpos = R(mesh_quat)·lpos + mesh_pos
      depth = min(mjc_distance(sdf, lpos), 0); if depth == 0: continue
      vel_sensor = transformSpatial(cvel[parent_weld], 0, xpos, subtree_com[root(parent_weld)])
      vel_other  = transformSpatial(cvel[body_g],      0, xpos, subtree_com[root(body_g)])
      vel_rel = vel_sensor[3:6] − vel_other[3:6]
      out[0·nvert + j] = max(out, −depth)                     (max penetration)
      if has_frame: v = geom_xmatᵀ vel_rel; t1 = normal[9j+3..5], t2 = normal[9j+6..8]
                    out[1·nvert+j] += |v·t1| ; out[2·nvert+j] += |v·t2|   (sum over geoms)
```
Threading: if `mju_numThread(d) > 0 && nvert >= 1000`: `batch = ceil(nvert/nthread)`, `ntask = ceil(nvert/batch)`, `mju_dispatch` over taxel ranges (results independent per taxel → deterministic).

### 5.7 Acceleration-stage sensors (`mj_computeSensorAcc`)

| type | output |
|---|---|
| TOUCH | below |
| CONTACT (stage ACC) | `compute_contact_sensor` |
| ACCELEROMETER | `objectAcceleration(SITE, local=1)[3:6]` (cacc includes −gravity at world body ⇒ measures proper acceleration) |
| FORCE | `tmp = transformSpatial(cfrc_int[body], flg_force=1, site_xpos, subtree_com[root], site_xmat)`; `tmp[3:6]` |
| TORQUE | same; `tmp[0:3]` (torque about site position, in site frame) |
| ACTUATORFRC | `actuator_force[outadr .. +dim]` |
| JOINTACTFRC | `qfrc_actuator[jnt_dofadr]` |
| TENDONACTFRC | `Σ_{j: trntype==TENDON, trnid[0]==obj} actuator_force[outadr[j]]` (no gear factor) |
| JOINTLIMITFRC / TENDONLIMITFRC | first matching limit row: `efc_force[j]`, else 0 |
| FRAMELINACC | `objectAcceleration(obj, local=0)[3:6]` |
| FRAMEANGACC | `objectAcceleration(obj, local=0)[0:3]` |

**TOUCH**:
```
bodyid = site_bodyid[obj]; out = 0
for each contact j: conbody[k] = geom≥0 ? geom_bodyid : mj_flexBody(con,k)
   if efc_address>=0 && bodyid ∈ conbody:
      f = mj_contactForce(j)            (adhesion subtracted)
      if f[0] <= 0: continue
      ray = normalize(frame[0:3]·f[0]); if bodyid == conbody[1]: ray = −ray
      if mju_rayGeom(site_xpos, site_xmat, site_size, con->pos, ray, site_type, NULL) >= 0: out += f[0]
```
(Ray from contact point along normal toward the sensor body tests intersection with the site volume; always true if contact point inside.)

`mj_rnePostConstraint` (acc-sensor prerequisite, summarized): `cacc[0] = [0, −gravity]` (if gravity enabled); `cfrc_ext` from `xfrc_applied` (at xipos), contacts (non-flex, efc_address≥0: world force `frameᵀ·f` and torque, applied at contact pos, −1 on body1, +1 on body2), connect/weld equality forces, tendon forces (spatial tendons: limit/friction/equality/actuator(force·gear)/spring/damper/armature reactions applied along path); forward pass `cacc_j = cacc_parent + cdof_dot·qvel + cdof·qacc`, `cfrc_body = cinert·cacc + cvel ×* (cinert·cvel)`, `cfrc_int = cfrc_body − cfrc_ext`; backward accumulation into parents. Sets `flg_rnepost=1`.

### 5.8 Contact sensor (`compute_contact_sensor`)

Inputs: `objtype/objid` (match 1: SITE/BODY/XBODY(subtree)/GEOM/UNKNOWN), `reftype/refid` (match 2: BODY/XBODY/GEOM/UNKNOWN), `dataspec = intprm[0]`, `reduce = intprm[1]`, `num = dim/size`, `size = condataSize(dataspec)`.
```
zero(sensordata, dim); data[k] pointers into slot 0 in enum order for each set bit
nmatch = 0
for j in 0..ncon−1:                                  (ALL contacts in d->contact, incl. excluded/gap ones)
   r = matchContact(m,d,j, objtype,objid, reftype,refid); if !r: continue
   match[nmatch] = {id=j, flip = r<0, criterion}
   criterion: MINDIST → contact.dist ; MAXFORCE → −‖f[0:3]‖² (mj_contactForce, adhesion subtracted)
   nmatch++
nslot = min(num, nmatch)
MINDIST/MAXFORCE: ContactSelect(match, heap, nmatch, nslot)   (k smallest by (criterion, id), sorted ascending)
NETFORCE: (below; return)
for j < nslot: copySensorData(m,d,data, match[j].id, match[j].flip, nmatch); advance pointers by size
```
Unused slots remain zero. Without reduction (`reduce==0`), slots are the first `nslot` matches in contact-array order.

`ContactSelect` = `mjPARTIAL_SORT`: if `k<=0 || n<k` return (n==k still runs); heap of first k (max-heap by comparator), scan rest replacing root when `cmp(arr[i], root) < 0`; copy back and insertion-sort ascending. Comparator: criterion then contact id (deterministic ties).

`matchContact(conid, type1,id1, type2,id2)`:
```
if type1==UNKNOWN && type2==UNKNOWN: return 1
if type1==SITE && !mj_insideSite(id1, con.pos): return 0
geom_s = con.geom[s]; body_s = geom_s>=0 ? geom_bodyid : mj_flexBody(con, s)
checkMatch(body, geom, type, id): UNKNOWN→1; SITE→1; GEOM→id==geom; BODY→id==body;
                                  XBODY→ walk body=parent while body>id; return body==id; else 0
m11=check(b1,g1,t1,i1) m12=check(b2,g2,t1,i1) m21=check(b1,g1,t2,i2) m22=check(b2,g2,t2,i2)
if !m11 && !m12: 0 ; if !m21 && !m22: 0
if both specified: reg = m11&&m22; rev = m12&&m21; reg&&!rev → 1; rev&&!reg → −1; both → 1; neither → 0
elif type1 specified: m11 ? 1 : −1
elif type2 specified: m22 ? 1 : −1
```
`copySensorData(data, id, flip, nfound)`:
- FOUND: `nfound` (total matches, not slots).
- FORCE/TORQUE: `mj_contactForce`; copy `[0:3]` / `[3:6]`; if flip: negate z (third) component only.
- DIST: `con.dist`. POS: `con.pos`.
- NORMAL: `frame[0:3]`, if flip: negated (whole vector). TANGENT: `frame[3:6]`, negated if flip. (Flip = 180° rotation about frame[2] (2nd tangent): normal and t1 negate, t2 unchanged; so force/torque components in new frame: x→−(−x)… implemented as only z sign change per the source comment.)
- LINVEL/ANGVEL: if both sides are rigid (geom ≥0, or flex vertex with `flex_interp==0`): `vel_s = transformSpatial(cvel[bid_s], 0, con.pos, subtree_com[root(bid_s)])` with `bid` = geom body or `flex_vertbodyid[flex_vertadr+vert]`; `ang = vel1[0:3]−vel0[0:3]`, `lin = vel1[3:6]−vel0[3:6]`. Else (interpolated flex) via `mj_contactJacobian` difference times qvel (sparse chain or dense). Then `out = frame·v` (frame rows; i.e. `mju_mulMatVec3(out, frame, v)`), and if flip: negate z component.

NETFORCE (reduce 3):
```
if nmatch == 0: return (all zeros, FOUND=0)
for each match j: wrench_j = mj_contactForce(id_j) (6); if flip: wrench_j *= −1 (all 6); pos_j, frame_j
w_j = ‖wrench_j[0:3]‖ ; point = Σ w_j pos_j / max(Σ w_j, mjMINVAL)
total_wrench about point (global): for j: f_j = frame_jᵀ wrench_j[0:3]; t_j = frame_jᵀ wrench_j[3:6]
    force += f_j; torque += t_j + (pos_j − point) × f_j          (accumulated in match order)
slot 0: FOUND=nmatch, FORCE=force, TORQUE=torque, DIST=0, POS=point, NORMAL=(1,0,0), TANGENT=(0,1,0)
```
(LINVEL/ANGVEL incompatible with netforce at compile time.)

### 5.9 Plugin sensors / `mujoco.sensor.touch_grid`

Attributes: `nchannel` (1–6, 0→1), `size = [nx, ny]` (>0), `fov = [fovx ∈ (0,180], fovy ∈ (0,90]]` degrees, `gamma ∈ [0,1]`. `nsensordata = nchannel·nx·ny`; `needstage = ACC`; stateless; must be attached to a site.

Compute:
```
id = first sensor with type PLUGIN and sensor_plugin == instance
zero(sensordata, dim); zero(distance_, nx*ny)
parent_weld = body_weldid[body_weldid[site_bodyid[site]]]
count contacts with weld(geom_bodyid[geom1]) or weld(geom2) == parent_weld  (assumes geoms ≥ 0)
for each such contact (contact order):
   f = mj_contactForce (6, contact frame) ; f_site = site_xmatᵀ · frameᵀ · f[0:3]; t_site likewise
   if parent_body < max(body1, body2): f,t *= −1          (parent_body = weld of site body; body ids are geom bodies)
   permute (x,y,z) → (z,x,y) for both
   p = site_xmatᵀ (con.pos − site_xpos); spherical: a = atan2(x, −z), e = atan2(y, sqrt(x²+z²)), r = |p|
bin edges: linspace(−1,1, n+1) (incremental: lower += (u−l)/(n−1)), foveate x' = g·x⁵ + (1−g)·x (if gamma≠0, g=clip(gamma,0,1)),
           scale by fov·π/180
for channel c < nchannel: if channel forces not all zero: Histogram2D(a, e, weights=f_perm[c] (c<3: force, 3..5: torque),
           x_edges, y_edges) into sensordata + c·nx·ny
Histogram2D: x_idx = LowerBound(edges, x) (first edge >= x); skip if x_idx==0 or ==n_edges; same for y;
             index = (y_idx−1)·nx + (x_idx−1); hist[index] += w
distance_ (visualization only): mean r per bin
```
Cutoff then applied (REAL clamp).

---

## 6. Actuator plugin `mujoco.pid` (plugin/actuator/pid.cc)

Attributes `kp, ki, kd, imax, slewmax` (empty ⇒ unset). `i_max = imax/ki` if imax set and ki≠0 (clamp on integral of error). State in `act`: `[integral] (if ki≠0), [previous ctrl] (if slewmax set)`; plus one builtin slot appended if dyntype ∈ {FILTER, FILTEREXACT, INTEGRATOR} (actdim must equal this). `nstate = 0`. Applies to actuators with `actuator_plugin == instance` (searches over `m->nu`; uses actuator index as ctrl/length/force index — assumes standard 1-in/1-out).

```
GetCtrl(i, state, actearly):
  if dyntype==NONE: ctrl = d->ctrl[i], clipped to ctrlrange if ctrllimited   (raw d->ctrl; no delay)
  else: adr = actadr+actnum−1; ctrl = actearly ? NextActivation(adr, act_dot[adr]) : act[adr]
        (plugin-local NextActivation: FILTEREXACT exact, else Euler; actrange clamp)
  if slewmax set && state.previous_ctrl_exists (d->time > 0):
        ctrl = clip(ctrl, prev − slewmax·dt, prev + slewmax·dt)
ActDot (called in act_dot phase, after builtin act_dot):
  ctrl = GetCtrl(actearly=false); err = ctrl − actuator_length[i]
  if ki: integ = clip_opt(state.integral + err·dt, ±i_max); act_dot[s] = (integ − act[s])/dt; s++
  if slewmax: act_dot[s] = (ctrl − act[s])/dt
Compute (force phase):
  ctrl = GetCtrl(actearly = actuator_actearly[i]); err = ctrl − length
  ctrl_dot = dyntype==NONE ? 0 : act_dot[last]; err_dot = ctrl_dot − actuator_velocity[i]
  integral = ki ? clip_opt(state.integral + err·dt, ±i_max) : 0
  actuator_force[i] = kp·err + kd·err_dot + ki·integral
```
Euler integration of these act_dots reproduces `integral_{t+1} = clip(integral_t + err·dt)` and `prev_ctrl = ctrl` exactly (up to rounding of `(x−a)/dt·dt`). Builtin forcerange clamp, tendon clamp and qfrc mapping then apply normally (plugin forces are written before §2.4).

---

## 7. Stage ordering in the top-level pipeline (C)

`mj_forwardSkip(skipstage, skipsensor)`:
```
POS : mj_fwdPosition (…, mj_transmission) ; mj_sensorPos ; energyPos fallback
VEL : mj_fwdVelocity (actuator_velocity, comVel, passive, …) ; mj_sensorVel ; energyVel if enabled
      mjcb_control (if !DISABLED(ACTUATION))
ACC : mj_fwdActuation ; mj_effActuation ; [metric: regularize, reference] ; mj_fwdAcceleration ;
      constraint solve ; flg_rnepost = 0 ; mj_sensorAcc
```
`mj_step`: `mj_forward` then integrator (each calls `advanceStart` → history insert + activation integration). RK4 internal stages call `mj_forwardSkip(..., skipsensor=1)` — no sensor/history work at intermediate stages; `d->time` set to stage times during RK.
`mj_step1`: position, sensorPos, energy, velocity, sensorVel, energy, `mjcb_control` (no actuation-disable check here). `mj_step2`: actuation → … → sensorAcc → integrate (RK4 falls back to Euler).

---

## 8. GPU (MuJoCo Warp) implementation

General: all arrays are `[nworld, ...]`; Model fields with leading `*` dim are batched and indexed `worldid % shape[0]` (e.g. `actuator_gainprm[worldid % n, uid]`). Default precision float32 (`wp.float32`); C is float64. No sleeping/threading branch in these kernels; no `mj_actuatorDisabled` group filtering.

### 8.1 Feature support (io.py `put_model` checks)
- Enums accepted (others ⇒ `NotImplementedError`): `TrnType {JOINT, JOINTINPARENT, SLIDERCRANK, TENDON, BODY, SITE}` (no SO3), `DynType {NONE, INTEGRATOR, FILTER, FILTEREXACT, MUSCLE, USER, DCMOTOR}` (no PID), `GainType {FIXED, AFFINE, MUSCLE, USER, DCMOTOR, SO3}`, `BiasType {NONE, AFFINE, MUSCLE, USER, DCMOTOR}` (no SO3 ⇒ SO3 actuators rejected; no PID gain ⇒ PID rejected).
- Sensor types: all except PLUGIN (USER via `m.callback.sensor`). `DataType` only REAL/POSITIVE are named but AXIS/QUATERNION values pass through (cutoff only acts on REAL/POSITIVE).
- `actuator_plugin`, `sensor_plugin`, `body_plugin` any ≠ −1 ⇒ NotImplementedError (so `mujoco.pid` and `touch_grid` unsupported).
- Contact sensor LINVEL/ANGVEL with any flex ⇒ NotImplementedError.
- Unsupported silently/structurally: enclosed INSIDESITE (Hausdorff mode; `intprm[0]` ignored ⇒ always point test), mesh sites in INSIDESITE/contact-site filter (`inside_geom` returns False for MESH), flex contacts in TOUCH/tactile/contact sensors (geom ids assumed ≥0; tactile explicitly skips flex), sleeping filters in sensors, `opt.disableactuator` groups, `mju_isBad` control zeroing (no BADCTRL handling), callbacks: `m.callback.act_dyn/act_gain/act_bias/control/sensor` are host-side Python hooks invoked after the kernel.
- `opt.contact_sensor_maxmatch` (default 64; from numeric `contact_sensor_maxmatch`).

Precomputed in io.py: `sensor_pos_adr` (POS-stage sensors excluding limit and contact), `sensor_limitpos_adr`, `sensor_vel_adr`, `sensor_limitvel_adr`, `sensor_acc_adr` (ACC excluding TOUCH, CONTACT, LIMITFRC, TENDONACTFRC), `sensor_stage_{pos,vel,acc}_adr` (all sensors per stage, for history), `sensor_touch_adr`, `sensor_limitfrc_adr`, `sensor_tendonactfrc_adr`, `sensor_contact_{pos,vel,acc}_adr`, flags `sensor_e_potential/e_kinetic/subtree_vel/rne_postconstraint`, rangefinder ray tables (`nrangefinder` = 1 per site sensor + W·H per camera sensor; `sensor_rangefinder_adr[ray] = sensorid`, `rangefinder_sensor_adr[sensor] = first ray`, `sensor_rangefinder_bodyid` exclude body), tactile tables (`nsensortaxel`, `taxel_vertadr`, `taxel_sensorid`, `ntactileweld`, `weld_tactile_id`), collision-sensor pair tables (`nxn_pairid[:,1]`, `nsensorcollision`).

### 8.2 Actuation kernels (`fwd_actuation`)
```
if nactuator==0 or DISABLED(ACTUATION): act_dot=0, qfrc_actuator=0, actuator_force=0   (C leaves act_dot)
else:
  ctrl = history.read_ctrl_delayed(...) if nhistory>0 else d.ctrl     (kernel dim (nworld, nactuator))
  _actuator_force               dim (nworld, nactuator)  → act_dot, actuator_force
  callbacks act_dyn / act_gain / act_bias (host)
  if ntendon: ten_actfrc = zeros; _tendon_actuator_force (atomic_add per tendon) ; _tendon_actuator_force_clamp
  qfrc_actuator.zero_(); _qfrc_actuator dim (nworld, nactuator): atomic_add moment·force into dof
  _qfrc_actuator_gravcomp_limits dim (nworld, nv)
if integrator == DISCRETE: derivative.eff_actuation; constraint.regularize_constraint
```
`_actuator_force` per (world, actuator) — differences from C:
1. **Ctrl clamp**: only the FIRST input `ctrl_in[uadr]` is clamped (`actuator_ctrllimited[uadr]`, `ctrlrange[uadr]`); additional inputs (DC motor multi-input) are read raw by `unpack_servo_inputs`. C clamps all nu controls.
2. **No NaN/bad ctrl zeroing.**
3. act_dot formulas identical to C for INTEGRATOR (single slot — `act_dot_out[act_last]=ctrl`), FILTER/FILTEREXACT, MUSCLE, DCMOTOR (slot logic replicated; `u_first` carries the slew-limited first input); USER ⇒ 0 (callback fills); NONE ⇒ 0.
4. actearly: computes next activation of the last slot inline (DC motor per-slot rules replicated; for DC motor it additionally clamps to actrange if actlimited — compiler forces actlimited false so harmless); non-DC uses `support.next_act` (FILTEREXACT exact, USER returns `act_in` unchanged, else Euler; clamp if actlimited).
5. Gain: USER gain = **0** before callback (C default `1` when no `mjcb_act_gain`). DCMOTOR stateless with controller bits `(ctrlspec & 7) != 0` recomputes voltage from `u_first`; `ctrlspec == VOLTAGE` uses `u_first`; `ctrlnum == 0` ⇒ 0.
6. **Order of clamps differs**: Warp computes `force = gain·ctrl_act + bias`, clamps to forcerange immediately, then adds DC motor cogging/LuGre, writes; tendon actfrc scaling happens AFTER (in a later kernel) and its total includes DC mechanical terms. C: tendon scaling → forcerange clamp → DC mechanical. Results differ whenever a tendon-limited actuator also has forcerange, or a DC motor drives a force-limited tendon.
7. `wrap_period` only handles the servo shape (no PID).
8. `_qfrc_actuator` uses `wp.atomic_add` (non-deterministic summation order unless `opt.deterministic & ATOMICS`, which builds the kernel with `wp.DeterministicMode.RUN_TO_RUN`, `deterministic_max_records = nv`).
9. `_qfrc_actuator_gravcomp_limits`: per dof: `qfrc += qfrc_gravcomp` if gravity enabled and `jnt_actgravcomp[dof_jntid]` (C additionally requires `m->flg_gravcomp` and `‖gravity‖≠0` — equivalent result since qfrc_gravcomp would be 0); then clamps **every dof** of a joint with actfrclimited (C clamps only `jnt_dofadr`; same since only hinge/slide can be limited).
10. Transmission (`smooth._transmission`): one thread per actuator, row addresses allocated by `wp.atomic_add(moment_nnz, worldid, nnz)` ⇒ `moment_rowadr` order non-deterministic (values per row identical; `qfrc_actuator` sum order unaffected per row). Slider-crank/site moments enumerate the dof ancestor chains directly (`support.jac_dof`) rather than dense Jacobian + nonzero compression — rows may contain explicit zeros and different column ordering (sorted descending by traversal, written `ptr = ndof−1 … 0`, so ascending). BODY transmission row has all nv columns (`rownnz = nv`), filled by `_transmission_body_moment` kernels with atomics.
11. `actuator_velocity`: `_actuator_velocity` per (world, actuator) loops its sparse row (C sparse matvec; same order within row).

`_next_activation` (in `_advance`, dim (nworld, nactuator)): DC motor per-slot rules identical to C `mj_nextActivation` (no actrange clamp); others via `next_act` with clamp `limit and actlimited`, then integrator rotational wrap (`err = act − length; act −= period·round(err/period)`; `wp.round` rounds half away from zero like C `round`). Differences: USER dyntype is **not integrated** (C Euler-integrates act_dot from callback); no `mj_actuatorDisabled` zeroing; no SO3 re-anchor. RK4 passes `act_dot_scale` and stage combination (Warp `_rk_accumulate`).

`_advance` ordering: activations → qvel → qpos → plasticity → `history.insert_ctrl_history` (inserts raw `d.ctrl` at `d.time`) → time advance → `qacc_warmstart` copy → sleep. C inserts history before integrating activations; equivalent because ctrl/time unchanged in between. Sensor history is NOT advanced in `_advance` (see 8.4).

### 8.3 Muscle / DC motor helpers (util_misc.py)
`muscle_gain_length`, `muscle_gain`, `muscle_bias`, `_sigmoid`, `muscle_dynamics_timescale`, `muscle_dynamics`, `dcmotor_slots` (returns `vec6i(slew, integral, temperature, bristle, current, num)`), `dcmotor_resistance`, `lugre_stribeck`, `unpack_servo_inputs`, `dcmotor_voltage` — formula-identical to C (same `max(MJ_MINVAL, ·)` guards, same Horner sigmoid). Note `dcmotor_voltage` returns `u_first` (not `0 + u4[3]`) when no controller bits are set; with `spec == VOLTAGE` this equals C. Precision: float32 ⇒ `MJ_MINVAL = 1e-15` guards are effectively "avoid exact zero" only.

### 8.4 History (history.py)
Buffer layout and algorithms identical to C (`_history_physical_index`, circular binary search `_history_find_index`, insert with exact-match overwrite / replace-oldest / advance-cursor / out-of-order shift, read with ZOH / linear / Catmull-Rom Hermite) **except the time tolerance is `1e-6`** (absolute, for "exact match" and endpoint extrapolation) instead of `mjMINVAL = 1e-15`. History is per world: `d.history[nworld, nhistory]`; `actuator_history`, `actuator_historyadr`, `actuator_delay`, `sensor_*` are batched (`[*, n]`).

Kernels: `_read_ctrl_delayed_kernel` (per world×actuator; direct copy if `nsample==0 or delay==0`), `_insert_ctrl_history_kernel`, `_insert_sensor_history_stage`, `_apply_sensor_delay_kernel`, `_reset_actuator_history_kernel` / `_reset_sensor_history_kernel` (identical init incl. SO3-quat identity and `ceil(t/dt)·dt` interval timestamps), `read_ctrl`, `read_sensor`, `init_ctrl_history`, `init_sensor_history` (host-side monotonicity check `< MJ_MINVAL`).

**Sensor delay architecture difference**: Warp always *computes* every sensor at its stage, then `apply_sensor_delay(m, d, sensor_stage_X_adr)` (called at the end of `sensor_pos/vel/acc`): (1) copy fresh `sensordata`; (2) overwrite delayed/interval-held sensors by reading the buffer (delay>0: at `time − delay`; interval-only and `time_prev + period > t`: at `t`); (3) immediately insert the saved FRESH values into the buffer at `d.time` (interval mode: only if `time_prev + period <= t`, then `time_prev += period`). In C, insertion happens later in `advanceStart`, and for delay>0 sensors the value inserted is recomputed there (same state ⇒ same value). Consequences:
- Calling `forward()` repeatedly without stepping inserts again at the same timestamp (exact-match overwrite ⇒ harmless) but **advances `time_prev` for interval sensors in forward, not in step**.
- Warp `rungekutta4` calls the full `forward()` at the 3 intermediate stages (with `d.time` unchanged) ⇒ sensors are recomputed and RK-stage values overwrite the history slot at time t, and interval `time_prev` may advance at an intermediate stage. C RK4 uses `skipsensor=1`. For RK4 + sensor history, final sensordata/history diverge from C.
- `forward()` zeroes `d.sensordata` before `sensor_pos` (C never zeroes; with `DISABLED(SENSOR)` C keeps stale values while Warp shows zeros).

### 8.5 Sensor kernels (sensor.py)

Cutoff: `_write_scalar/_write_vector` apply cutoff for REAL/POSITIVE except GEOMFROMTO (contact sensors are written by a separate kernel without cutoff ⇒ equivalent to C).

`sensor_pos(m, d)`:
1. Rangefinder (if `nrangefinder`): `_sensor_rangefinder_init` (dim (nworld, nrangefinder)) computes per-ray origin/direction (camera intrinsics replicated; `math.safe_div`), `ray.rays(...)` batched ray casting (flg_static=True, per-ray body exclusion, cutoff `MJ_MAXVAL`), `_sensor_rangefinder` writes fields with `_clamp_real(cutoff)` per component. DEPTH for camera: `−dist·(dir · cam_z)` (algebraically equals C's `−(point − cam_xpos)·cam_z` for perspective; for ORTHOGRAPHIC cameras C uses `point − cam_xpos` with shifted origins — differs: C depth includes the in-plane offset projection (which is 0 since offset ⟂ z) ⇒ equal). Normal/dir zeroed on miss.
2. Energy (`energy_pos`, `energy_vel`) if E_POTENTIAL / E_KINETIC sensors exist (always recomputed, no lazy flag).
3. Collision sensors: `sensor_collision = full((nworld, nsensorcollision, 8, 7), 1e32)`; `_sensor_collision` (dim naconmax) scatters each contact tagged `ContactType.SENSOR` into `[world, collisionid, geomcollisionid] = (dist, pos[3], frame row 0 = normal[3])`.
4. `_sensor_pos` (dim (nworld, n_pos_sensors)): MAGNETOMETER, CAMPROJECTION (adds `MJ_MINVAL` to sensorsize in the division — negligible), JOINTPOS, TENDONPOS, ACTUATORPOS (scalar only; outnum=1 since SO3 unsupported), BALLQUAT (`wp.normalize`), FRAMEPOS, FRAMEAXIS, FRAMEQUAT (`quat_inv(ref)*q`), SUBTREECOM, GEOMDIST/NORMAL/FROMTO, INSIDESITE (point test only), E_POTENTIAL/E_KINETIC (read `d.energy`), CLOCK.
5. `_limit_pos` (dim (nworld, njmax, nlimitpos)): every efc row in the limit block matching type and id writes `efc_pos − efc_margin` (last writer wins if several rows match; C takes the first). Rows of the *other* limit type with same id are also accepted (kernel checks `efc_type ∈ {LIMIT_JOINT, LIMIT_TENDON}` and `efc_id == objid` without checking that the type matches the sensor kind ⇒ a joint-limit sensor could pick up a tendon-limit row whose tendon id equals the joint id. Pitfall/bug).
6. `_eval_sensor_contact(sensor_contact_pos_adr)`.
7. `history.apply_sensor_delay(stage_pos)`; `m.callback.sensor(m, d, POS)` (cutoff not reapplied for USER sensors — C applies cutoff after the callback).

GEOM distance sensors in Warp: collision pairs required by sensors are always added in the broadphase (`pairid[1] >= 0`) and pass through the normal narrowphase; contacts are tagged SENSOR and written even when not "detected" for constraint purposes. Convex (GJK/EPA) narrowphase uses `cutoff = 1e32` for sensor pairs (no distance limit) and returns `dist` corrected by `+margin`; primitive colliders run with the pair's contact `margin + gap` (not `sensor_cutoff`) — whether they emit contacts beyond that margin depends on each collider (many early-out at `dist > margin`). The sensor kernel then picks, over up to 8 stored contacts per pair (`geomcollisionid` slot), the minimum `dist < cutoff` (init `dist = cutoff`), with normal from contact frame, and:
```
flip = geom_type[g1] > geom_type[g2] || (equal types && g1 > g2)
segment = (dist >= cutoff || dist == 0) ? 0 : (flip != (dist < 0) ? −normal : normal)
GEOMNORMAL = segment ; GEOMFROMTO = dist<cutoff ? [pos − 0.5|dist|·segment, pos + 0.5|dist|·segment] : 0
```
vs C: normal = normalized(fromto[3:6] − fromto[0:3]) with fromto from `pos ± 0.5·sign·dist·normal` (sign = type flip only). For equal geom types Warp uses an id-order flip (Warp's collision pairs store geoms sorted by id); C never flips equal types (calls func(g1, g2) with sensor order). Distances beyond the contact margin may be missing for primitive pairs ⇒ GEOMDIST returns `cutoff` where C returns the true distance. Requires `DISABLED(NATIVECCD)`-independent behavior: Warp always uses its own GJK/EPA (≈ nativeccd).

`sensor_vel(m, d)`: `smooth.subtree_vel` if needed (always, not lazy); `_sensor_vel` (VELOCIMETER, GYRO, JOINTVEL, TENDONVEL, ACTUATORVEL, BALLANGVEL, FRAMELINVEL, FRAMEANGVEL, SUBTREELINVEL, SUBTREEANGMOM) — formulas expanded inline (`lin − (pos − com)×ω`, rotate by `xmatᵀ`); **no dof-less body zeroing** (C returns 0 for static/mocap welds; Warp uses `cvel`, which is 0 for world-welded static bodies; for mocap bodies cvel is also 0 ⇒ same in practice). FRAMELINVEL with ref: `R_refᵀ(v_obj − v_ref + (x_obj − x_ref) × ω_ref)` = C. `_limit_vel` (same multi-row caveat). Tactile: `_mark_tactile_contacts` (dim naconmax; per contact set bit `geom` in `weld_geom_seen[world, tactile_weld, word]` via `atomic_or`, both directions; skips flex), `_collect_tactile_contacts` (per world×tactile weld: list geoms in ascending id order up to `MJ_MAXCONPAIR = 50`, overflow flag `OverflowType.TACTILE`), `_sensor_tactile` (dim (nworld, nsensortaxel)): per taxel SDF depth via `collision_sdf.sdf` / `get_sdf_params` (octree for meshes), max penetration, tangential speeds `|v·n[offset+1]|`, `|v·n[offset+2]|` (mesh_normal as vec3 array: 3 vectors per vertex). `has_frame` additionally requires `nchannel >= 3`. Geom iteration order = ascending geom id (C: contact order) — affects only floating sum order of the tangential channels. Cutoff applied as REAL clamp. Then contact sensors (vel stage), `apply_sensor_delay(stage_vel)`, callback.

`sensor_acc(m, d, skip_rne_postconstraint)`: `forward()`/`step2()` call `smooth.rne_postconstraint` beforehand if `opt.run_rne_postconstraint` or (sensors enabled and `sensor_rne_postconstraint`), then `sensor_acc(skip=True)`.
1. `_sensor_touch` (dim (naconmax, ntouch)): normal force = `efc_force[adr0]` (+ rows 1..2(dim−1)−1 for pyramidal) — **adhesion is NOT subtracted** (C `mj_contactForce` subtracts `con->adhesion`); `wp.atomic_add` into sensordata (order-dependent rounding; deterministic mode available); no flex. Touch cutoff: **not applied** in Warp (written by atomic add without `_write_scalar`) — C applies POSITIVE cutoff `min(cutoff, value)`.
2. Contact sensors (acc stage).
3. `_sensor_acc`: ACCELEROMETER (`R(acc_lin − dif×acc_ang) + (Rω)×(R(v−dif×ω))` — identical math, different operation order), FORCE (`site_xmatᵀ · cfrc_int[3:6]` — translation does not affect force part), TORQUE (`site_xmatᵀ(cfrc_int[0:3] − dif × cfrc_int[3:6])`), ACTUATORFRC, JOINTACTFRC, FRAMELINACC, FRAMEANGACC (no dof-less zeroing).
4. `_tendon_actuator_force` (dim (nworld, ntenactfrc, nactuator), atomic add) + `_tendon_actuator_force_cutoff`.
5. `_limit_frc` (same multi-row caveat).
6. `apply_sensor_delay(stage_acc)`; callback.

Contact sensor pipeline `_eval_sensor_contact(adr)`:
- `_contact_match` (dim (ncontactsensors, naconmax)): considers only contacts with `ContactType.CONSTRAINT` (C: every contact in `d->contact`, which in C contains only contacts that passed margin/gap filtering, including excluded-in-gap ones). Site filter via `inside_geom` (no mesh sites). Same match/direction logic except: when both objects specified and neither order matches, return (no match) — same as C. Matches appended via `wp.atomic_add(nmatch)` ⇒ **match order is non-deterministic** (unless deterministic atomics), stored as `contactid` or `~contactid` (direction encoded by bitwise complement). Overflow beyond `contact_sensor_maxmatch` sets `OverflowType.CONTACT_MATCH` and drops matches. Criteria: mindist `dist`, maxforce `−‖f[0:3]‖²` via `support.contact_force_fn` (adhesion subtracted).
- `_contact_sort` (tiled, `wp.tile_sort` on `maxmatch` padded entries, padding criterion 1e32): ascending criterion; ties broken by sort implementation (not by contact id as in C).
- `_sensor_contact` (dim (nworld, ncontactsensors)): FOUND = `nmatch_raw` (may exceed maxmatch); `reduce==0` slots take the first `min(nmatch, num)` entries in **atomic arrival order** (C: contact array order) ⇒ non-deterministic slot assignment for unsorted contact sensors with num < nmatch or with multiple slots. Netforce: point and torque accumulated about the origin then shifted (`torque −= point × force`) — algebraically equal to C's about-point accumulation, rounding differs; `weight = ‖f[0:3]‖` before flip (same). Per-slot fields: force/torque z-component times dir; normal and tangent multiplied by dir; LINVEL/ANGVEL from cvel of geom bodies (`frame @ rel`, z times dir). Remaining slots zeroed.

`energy_pos`: atomic accumulations per body / joint / tendon / flex edge (non-deterministic order); BALL/FREE rotational spring uses the **normalized** quaternion in `quat_sub` (C uses raw unnormalized qpos — differs when |q|≠1); no sleep filter. `energy_vel`: `support.mul_m` then tiled reduction `0.5·Σ qvel·(M qvel)`. `forward` calls `_energy_pos/_energy_vel` only if ENERGY enabled and no energy sensor computed it; if ENERGY disabled `d.energy.zero_()` (both components).

### 8.6 How to match MJWarp
- Use float32 everywhere with the same expression trees (muscle and DC motor helpers are line-for-line ports).
- Reproduce clamp ordering: forcerange clamp inside the per-actuator force kernel, DC mechanical terms after it, tendon scaling after that; only clamp the first ctrl input.
- USER gain default 0; USER dyntype activation not integrated by `_next_activation`.
- History time tolerance 1e-6; sensor history inserted at sensor-evaluation time inside forward.
- Contact sensors: order-dependent outputs require `opt.deterministic` (ATOMICS) to be reproducible run-to-run, and even then order follows contact generation order inside the GPU collision pipeline, not C's.
- Touch: no adhesion subtraction, no cutoff.
- Limit sensors: last-matching-row semantics.

### 8.7 MJX (JAX) notes
- `fwd_actuation`: dyntypes NONE/INTEGRATOR/FILTER/FILTEREXACT/MUSCLE only; gain FIXED/AFFINE/MUSCLE; bias NONE/AFFINE/MUSCLE; transmissions JOINT/JOINTINPARENT/SITE/TENDON. Clamps ctrl (all inputs), computes act_dot, `ctrl_act = act_last` for stateful actuators (**actearly ignored**), applies `_wrap_setpoint` (round half away from zero via `sign·floor(|x|+0.5)`), tendon actfrc scaling then forcerange clamp (C order), `qfrc_actuator = momentᵀ force` (dense), gravcomp, joint actfrc clamp. No delays/history usage (history array exists in Data but is not read), no DC motor, no PID/SO3, no plugins, no NaN check.
- Activation integration: FILTEREXACT exact, else Euler; clip to actrange (jnp.inf when unlimited); integrator wrap.
- Sensors: supports MAGNETOMETER, CAMPROJECTION, RANGEFINDER, JOINTPOS, TENDONPOS, ACTUATORPOS, BALLQUAT, FRAMEPOS/QUAT/AXES, SUBTREECOM, CLOCK, VELOCIMETER, GYRO, JOINTVEL, TENDONVEL, ACTUATORVEL, BALLANGVEL, FRAMELINVEL/ANGVEL, SUBTREELINVEL/ANGMOM, TOUCH, CONTACT, ACCELEROMETER, FORCE, TORQUE, ACTUATORFRC, JOINTACTFRC, TENDONACTFRC, FRAMELINACC/ANGACC. Not supported: limit sensors, geom distance sensors, INSIDESITE, energy sensors, tactile, user/plugin. Contact sensor: no SITE, BODY or subtree matching, no netforce. Cutoff via `_apply_cutoff` (REAL/POSITIVE).

---

## 9. Reproduction pitfalls / parity notes

1. **Clamp order (C)**: delay read → ctrl clamp → bad-ctrl zeroing → act_dot → force → tendon scaling (ratio applied to each actuator's force) → forcerange clamp → DC cogging/LuGre → qfrc → gravcomp → joint clamp. Any reordering changes results.
2. **Slew limiting mutates local ctrl** (PID and DC motor): the force uses the slew-limited setpoint; `act_dot = (u_eff − u_prev)/dt` so after Euler `act = u_eff` (up to rounding).
3. **PID force** is assembled from `−biasprm[1]·qref − biasprm[2]·vref + ff + ki·z` plus AFFINE bias; kp and kv live in biasprm (negated); ki in gainprm[0]; `imax` in dynprm[0] (direct integral clamp via anti-windup on act_dot, not on act; act is not clamped by imax for builtin PID, only act_dot sign-limited).
4. **DC motor**: the controller voltage uses nominal `R = gainprm[0]` (not temperature-corrected), while current dynamics and gain use temperature-corrected R; stateless gain uses `mj_dcmotorResistance` (with `max(mjMINVAL,·)`), the temperature ODE uses the uncapped corrected R. LuGre bristle uses an exact ZOH update independent of act_dot; force uses current `z` and `z_dot`. DC motor always actearly; actlimited ignored.
5. **Muscle dynamics** uses `dctrl = clip(ctrl,0,1) − act` with UNclamped act, but timescales from clamped act. `smoothing_width < mjMINVAL` ⇒ hard switch with `dctrl > 0` strict.
6. **Muscle auto force** `F0 = scale/acc0` depends on `actuator_acc0` computed at qpos0 via `M⁻¹`; any difference in M (armature, inertia) changes muscle strength. `lengthrange` from `mj_setLengthRange` is the result of a 10 s simulated optimization at `LRopt.timestep=0.01` with specific disable flags — reproducing it requires the full simulator; prefer reading `actuator_lengthrange` from the compiled model. The assignment `LR[2*index+side]` uses the actuator index (equals `out` for standard models).
7. **actearly** changes the force to use `mj_nextActivation` (includes actrange clamp and FILTEREXACT exact update) — the same value the integrator will produce (with Euler/implicit; RK4 combines stages).
8. **Disabled actuator groups** (`opt.disableactuator`): force 0, act_dot still computed but integrated with 0. Sleeping actuators: no act_dot, no force; their act is still integrated with whatever act_dot holds (zeroed only when not sleeping…: `act_dot` slots are zeroed only inside the loop for non-sleeping actuators — sleeping ones keep the previous act_dot value; under sleep qvel=0 anyway).
9. **History tolerance** `mjMINVAL=1e-15` for timestamp equality in C; time accumulates as `time += dt` (float64) so timestamps computed as `−(n−j)·dt` vs accumulated times may differ by rounding; reads use `t ≥ t_newest − mjMINVAL` / `t ≤ t_oldest + mjMINVAL` endpoints and exact-match tests — reproduce bit-exactly. Warp uses 1e-6.
10. **Sensor interval**: `buf[0] += interval` (continuous ticks), test `buf[0] + interval <= time` — floating accumulation order matters. Interval ticks in C happen only in `advanceStart`.
11. **mj_computeSensor in advanceStart** (delay>0 sensors) recomputes using current mjData (post-forward) — for ACC sensors `flg_rnepost` is still set from forward, so it reuses `cfrc_int`/`cacc`.
12. **Cutoff** is skipped for CONTACT and GEOMFROMTO; GEOMNORMAL (AXIS) is never clipped; for collision sensors the cutoff is the `distmax` (default 0 ⇒ only penetration reported; GEOMDIST outputs `cutoff` when nothing found).
13. **mj_geomDistance** flips geom order by type and uses CCD (`mj_geomDistanceCCD`) for convex/box-box pairs unless `NATIVECCD` disabled; other pairs use the primitive collider with margin = cutoff and the *first* minimal contact.
14. **Limit sensors** pick the FIRST efc row of the right type and id in `[ne+nf, nefc)` — order of limit rows (lower before upper for tendons) matters when both sides are active.
15. **Contact sensor**: slot order is contact-array order for `reduce=0`; FOUND reports total matches; flip negates only z of force/torque/linvel/angvel but all of normal/tangent; netforce flips the whole 6-vector before transforming. Partial sort tie-break by contact id. Contacts with `efc_address<0` (excluded / not in constraint) still match; their force is zero.
16. **TOUCH** uses adhesion-subtracted normal force and requires `f[0] > 0`; ray test with direction flipped when sensor body is body 2.
17. **FORCE/TORQUE** sensors measure `cfrc_int` of the site's body (force between body and parent, including all children), expressed at site position in site frame; depend on `mj_rnePostConstraint` which includes `xfrc_applied`, contacts, equality and spatial-tendon forces and gravity via `cacc[0] = −g`.
18. **ACCELEROMETER** includes gravity (−g at world), Coriolis correction `ω × v` computed in the site frame.
19. **mj_objectVelocity/Acceleration** return exactly zero for objects whose weld body has no dofs (static/mocap).
20. **Energy**: ball/free rotational spring energy uses raw (unnormalized) qpos in `mju_subQuat` in C; E_KINETIC as a POS-stage sensor is computed with `M` from the current position stage and current qvel; energies are lazily cached by flags.
21. **Tactile** sums tangential slip over contacting geoms in contact order; penetration uses SDF distance (`mjc_distance`) — mesh geoms require an octree; dedup is by geom id.
22. **Sensor sleeping**: sleeping sensors keep stale values (not zeroed). `DISABLED(SENSOR)` keeps stale values in C.
23. **Plugin sensors** with needstage NONE are computed in POS stage; plugin compute triggers `mj_subtreeVel`/`mj_rnePostConstraint` unconditionally at VEL/ACC.
24. **USER sensors** are zeroed before `mjcb_sensor`, and cutoff applied after; for VEL/ACC user sensors `mj_subtreeVel`/`mj_rnePostConstraint` are run first.
25. **Warp-specific**: float32; atomic summation (qfrc_actuator, touch, tendon actfrc, energy, contact matching) non-deterministic unless `DeterminismType.ATOMICS`; clamp-order differences (item 1 vs §8.2.6); first-input-only ctrl clamp; touch without adhesion subtraction and without cutoff; limit sensors last-writer; contact sensor order; RK4 + sensor history; `forward` zeroes sensordata; GEOMNORMAL/FROMTO equal-type flip and margin-limited primitive distances; no enclosed INSIDESITE / mesh sites; USER gain 0 and USER act not integrated; history tolerance 1e-6.

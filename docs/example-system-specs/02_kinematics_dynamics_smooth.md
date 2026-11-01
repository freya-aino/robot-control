# 02 — Smooth (unconstrained) kinematics and dynamics

**Sources covered**

| Backend | Files |
|---|---|
| C engine (commit 7ba52b5) | `src/engine/engine_core_smooth.c` (all), `src/engine/engine_core_util.c` (chains, Jacobians, jacDot, object velocity/acceleration, local2Global, contactForce, tendon spring/damper, actuator armature/damping), `src/engine/engine_util_spatial.c` (all), plus the helpers they call: `engine_inline.h` (`mji_*`), `engine_util_blas.c` (`mju_normalize3/4`, `mju_dot`), `engine_util_sparse.c/.h` (`mju_combineSparseInc`, `mju_addToSclSparseInc`, `mju_mulSymVecSparse`, `mju_dotSparse`), `engine_util_misc.c` (tendon wrapping `mju_wrap`, `wrap_circle`, `wrap_inside`, `is_intersect`, `length_circle`), `engine_support.c` (`mj_mulM`, `mj_mulM2`, `mj_fullM`), `engine_io.c` (`mj_makeDofDofSparse`, `mj_makeBSparse`, `mj_makeDofDofMaps`), `engine_setconst.c` (`mj_setM0`, `makeTendonSparse`), `user/user_model.cc` (`body_simple`, `dof_simplenum`) |
| MuJoCo Warp (commit d41a47c) | `mujoco_warp/_src/smooth.py` (all), `math.py` (all), `util_misc.py` (wrapping + armature/damping helpers), relevant parts of `io.py` (precomputed layouts), `support.py` (`jac_dof`, `transform_force`, `mul_m`) |
| MJX (JAX) | `mjx/_src/smooth.py`, `mjx/_src/math.py` (differences only) |

**Role in the pipeline.** These routines turn the generalized state (`qpos`, `qvel`, `mocap_*`) into Cartesian body/geom/site/camera/light frames, the CoM-frame spatial quantities used everywhere else (`subtree_com`, `cinert`, `cdof`, `cvel`, `cdof_dot`), tendon lengths and Jacobians, actuator transmission lengths and moment arms, the joint-space inertia matrix `M` (CRB + armature + tendon armature), its sparse `L'DL` factorization `qLD/qLDiagInv` and back-substitution, and the bias force `qfrc_bias = C(q,v) + g(q)` (RNE without acceleration, plus tendon-armature bias). After the constraint solve, `mj_rnePostConstraint` recomputes `cacc`, `cfrc_ext`, `cfrc_int`. Order inside `mj_forward`:

```
mj_fwdPosition:
  mj_fwdKinematics: mj_kinematics; mj_comPos; mj_camlight; mj_flex; mj_tendon; [mj_wakeTendon -> mj_updateSleep]
  mj_makeM (= mj_crb; mj_tendonArmature); mj_factorM
  mj_collision ... mj_makeConstraint; mj_island; mj_projectConstraint
  mj_transmission                      # AFTER collision/constraints (mjTRN_BODY needs contacts + efc_J)
  mj_effBuild
mj_fwdVelocity:
  flexedge_velocity = flexedge_J*qvel; ten_velocity = ten_J*qvel; actuator_velocity = moment*qvel
  mj_comVel; mj_passive; mj_velocityConstraint/mj_referenceConstraint
  mj_rne(flg_acc=0) -> qfrc_bias; mj_tendonBias(qfrc_bias)
... later: qfrc_smooth = qfrc_passive - qfrc_bias + qfrc_applied + qfrc_actuator;
           qacc_smooth = mj_solveM(qfrc_smooth)  (uses qLD)
mj_rnePostConstraint is lazy (sensors needing cacc/cfrc_*; flg_rnepost)
mj_subtreeVel is lazy (flg_subtreevel)
```

Warp `forward.py` uses the same order: `fwd_kinematics` = `kinematics, com_pos, camlight, flex, tendon`; `fwd_position` = `crb, tendon_armature, factor_m(if factorize), collision, ...`; `transmission` after constraints; `fwd_velocity` = velocities, `com_vel`, `passive`, `rne`, `tendon_bias`.

---

## 0. Conventions and constants

* `mjtNum` = `double` (C default; `mjUSESINGLE` makes it `float`). Warp: `float` = `wp.float32` everywhere in these kernels (single precision on GPU). MJX: JAX default dtype (float32 unless `jax_enable_x64`).
* `mjMINVAL = 1e-15` (1e-15f single). `mjMAXVAL = 1e10`. `mjPI = 3.14159265358979323846`. `mjNPOLY = 2`. Warp `MJ_MINVAL = mujoco.mjMINVAL` (1e-15, cast to float32 ≈ 1.0000000036e-15), `MJ_MAXVAL = 1e10`.
* Quaternions: `[w, x, y, z]`, Hamilton product, active rotation `v' = q v q*`. Rotation matrices are row-major 3×3 (`mat[3*r + c]`); `xmat` maps body-local to world (`x_world = xmat * x_local + xpos`). Columns of `xmat` are the body axes in world.
* Spatial (6D) vectors: **rotation first, then translation**: motion `[ω; v]`, force `[τ; f]`. All "c"-quantities (`cdof`, `cvel`, `cacc`, `cinert`, `cfrc_*`) are expressed in the **world orientation** with reference point at **`subtree_com[body_rootid[b]]`** (the CoM of the kinematic tree the body belongs to; one reference point per tree, not per body). Note: `xfrc_applied` is stored `[force; torque]` (opposite order).
* `body_rootid[b]`: top-most ancestor below world. `body_weldid[b]`: top dof-less ancestor chain collapsed ("the body that owns the dofs that move b"; mocap: own root). `body_dofadr[b] = -1` when `body_dofnum[b] == 0`.
* `dof_parentid[i]`: previous dof in the tree (the last dof of the closest ancestor body that has dofs, or the previous dof of the same joint/body), −1 at a tree root. Dofs are numbered depth-first so `dof_parentid[i] < i`. This is the elimination tree for the inertia factorization.

### Normalization helpers (exact semantics — they matter for parity)

```
mju_normalize3(v):  n = sqrt(v0²+v1²+v2²)
                    if n < mjMINVAL: v = (1,0,0)  else v *= 1/n      (multiply by reciprocal)
                    return n
mju_normalize4(q):  n = sqrt(Σ q²)
                    if n < mjMINVAL: q = (1,0,0,0)
                    elif |n-1| > mjMINVAL: q *= 1/n     (skip rescale if already unit to 1e-15)
                    return n
mju_normalize(v,n): like normalize3 for length n: if norm<mjMINVAL -> (1,0,...,0)
```

`mju_dot(a,b,n)` (and AVX version) uses 4 interleaved accumulators: `res = (r0+r2)+(r1+r3)` then the 1–3 remaining elements added as a single expression. `mji_dot6(a,b) = ((a0b0 + a2b2) + (a1b1 + a3b3)) + (a4b4 + a5b5)` (matches `mju_dot` order for n=6). `mju_dotSparse` uses the same 4-accumulator pattern over the nnz list. `mju_mulMatTVec(res, mat, vec, nr, nc)`: `res = 0; for r: if vec[r] != 0: res += mat[r,:]*vec[r]` (rows with zero coefficient are skipped).

`mju_mulDofVec(res, dof, vec, n)`: n==1 → `res = dof*vec[0]` (6-vector scale); n<=0 → zero; else `mju_mulMatTVec(res, dof, vec, n, 6)`.

---

## 1. Spatial / quaternion utilities (`engine_util_spatial.c`, `engine_inline.h`)

Inline `mji_*` variants have identical arithmetic to `mju_*` but use `restrict` (no aliasing); a few differ in early-outs (noted).

### 1.1 `mju_rotVecQuat(res, vec, quat)`
* If `vec == 0` exactly → `res = 0` (only in `mju_` version; `mji_rotVecQuat` lacks this branch but the formula yields 0 anyway).
* If `quat == (1,0,0,0)` exactly → `res = vec`.
* Else:
```
t = q_w*v + q_xyz × v:
  t0 = q0 v0 + q2 v2 - q3 v1
  t1 = q0 v1 + q3 v0 - q1 v2
  t2 = q0 v2 + q1 v1 - q2 v0
res = v + 2*(q_xyz × t):
  r0 = v0 + 2*(q2 t2 - q3 t1)
  r1 = v1 + 2*(q3 t0 - q1 t2)
  r2 = v2 + 2*(q1 t1 - q2 t0)
```
Assumes unit quaternion (not renormalized).

### 1.2 Quaternion algebra
```
negQuat(q) = (q0, -q1, -q2, -q3)                     (conjugate)
mulQuat(a,b) = ( a0b0 - a1b1 - a2b2 - a3b3,
                 a0b1 + a1b0 + a2b3 - a3b2,
                 a0b2 - a1b3 + a2b0 + a3b1,
                 a0b3 + a1b2 - a2b1 + a3b0 )         (computed into tmp; aliasing safe)
mulQuatAxis(q, ax) = q * (0, ax):
   ( -q1a0 - q2a1 - q3a2,  q0a0 + q2a2 - q3a1,  q0a1 + q3a0 - q1a2,  q0a2 + q1a1 - q2a0 )
axisAngle2Quat(axis, θ): θ == 0 → (1,0,0,0); else s=sin(θ/2): (cos(θ/2), axis*s)   [axis assumed unit]
quat2Vel(q, dt): axis = q_xyz; s = normalize3(axis); speed = 2*atan2(s, q0);
                 if speed > π: speed -= 2π;  speed /= dt;  res = axis*speed
subQuat(qa, qb) = quat2Vel(neg(qb)*qa, 1)    → rotation vector r with qb*exp(r) = qa (local frame of qb)
quat2Mat(q): identity if q == (1,0,0,0) exactly; else with qij = qi*qj:
   R00 = q00+q11-q22-q33   R01 = 2(q12-q03)   R02 = 2(q13+q02)
   R10 = 2(q12+q03)        R11 = q00-q11+q22-q33  R12 = 2(q23-q01)
   R20 = 2(q13-q02)        R21 = 2(q23+q01)   R22 = q00-q11-q22+q33
```
(`quat2Mat` does not assume unit norm in the diagonal formula: `q00+q11-q22-q33`, i.e. non-unit quats produce non-orthogonal matrices.)

`mju_mat2Quat(mat)`: Shepperd branch selection: if trace > 0: `q0 = 0.5 sqrt(1+tr)`, `q1 = 0.25(m7-m5)/q0`, `q2 = 0.25(m2-m6)/q0`, `q3 = 0.25(m3-m1)/q0`; elif m0>m4 && m0>m8: `q1 = 0.5 sqrt(1+m0-m4-m8)`, `q0=0.25(m7-m5)/q1`, `q2=0.25(m1+m3)/q1`, `q3=0.25(m2+m6)/q1`; elif m4>m8: `q2 = 0.5 sqrt(1-m0+m4-m8)`, `q0=0.25(m2-m6)/q2`, `q1=0.25(m1+m3)/q2`, `q3=0.25(m5+m7)/q2`; else `q3=0.5 sqrt(1-m0-m4+m8)`, `q0=0.25(m3-m1)/q3`, `q1=0.25(m2+m6)/q3`, `q2=0.25(m5+m7)/q3`; then `mju_normalize4`.

`mju_derivQuat(q, ω)` = ½ q ⊗ (0, ω):
```
r0 = 0.5(-ω0q1 - ω1q2 - ω2q3);  r1 = 0.5( ω0q0 + ω1q3 - ω2q2)
r2 = 0.5(-ω0q3 + ω1q0 + ω2q1);  r3 = 0.5( ω0q2 - ω1q1 + ω2q0)
```

`mju_quatIntegrate(quat, vel, scale)` (used by `mj_integratePos` for ball/free joints with `scale = dt`):
```
tmp = vel; angle = scale * normalize3(tmp)        # tmp → unit axis ((1,0,0) if |vel|<1e-15)
qrot = axisAngle2Quat(tmp, angle)                 # exact identity if angle == 0
normalize4(quat)                                  # normalize BEFORE multiply
quat = quat * qrot                                # right-multiply: rotation in LOCAL frame
```
No normalization after the product (the next `mj_kinematics` normalizes copies, not `qpos`).

`mju_quatZ2Vec(quat, vec)`: default identity; `vn = vec`, if `normalize3(vn) < mjMINVAL` return; `axis = z × vn`, `a = normalize3(axis)`; if `|a| < mjMINVAL`: if `vn·z < 0` → `(0,1,0,0)`; return. Else `angle = atan2(a, vn·z)`; `axisAngle2Quat(axis, angle)`.

`mju_derivExpmap(res, r, vel)`: if `|r|² > π²`, wrap angle: `x=|r|`, `r *= (x - 2π floor(x/2π + 0.5))/x`. `c = bortzCoef(|r|²)`; `res = vel + 0.5 r×vel + c r×(r×vel)`. `bortzCoef(xx)`: for `xx < 1.75² = 3.0625` Horner polynomial with 9 coefficients (double: `0.083333333333333384, 0.0013888888888859755, 3.3068783094433201e-05, 8.2671949071520734e-07, 2.0876900642039142e-08, 5.2828494303844588e-10, 1.3455485951057888e-11, 3.1615515500778219e-13, 1.230313397310169e-14`, lowest order first; single: `0.0833333358, 0.00138884981, 3.32480922e-05, 4.57548566e-07, 4.23758962e-07, -2.48791736e-07, 8.78585737e-08, -1.64186869e-08, 1.26339661e-09`); else `(1 - 0.5x/tan(0.5x))/xx`.

`mju_mat2Rot` (Müller et al. 2016 polar rotation extraction): up to 500 iterations, `ω = Σ_k col_k(R) × col_k(A) / (|Σ col_k(R)·col_k(A)| + mjMINVAL)`, stop when `|ω| < rotEPS` (1e-9 double, 1e-6 single), `q = axisAngle(ω/|ω|, |ω|) * q`, normalize. Returns iteration count.

Pose ops: `mulPose`: `quat = q1*q2` (then normalize4), `pos = rot(pos2, q1) + pos1`. `negPose`: `q = neg(q)`, `pos = -rot(pos, neg(q))`. `trnVecPose`: `rot(vec,quat)+pos`. `euler2Quat(seq)`: lowercase = intrinsic (post-multiply), uppercase = extrinsic (pre-multiply), each `rot = (cos(e/2), sin(e/2) on axis)`.

### 1.3 Spatial algebra
```
cross(a,b) = (a1b2-a2b1, a2b0-a0b2, a0b1-a1b0)

crossMotion(vel=[w;v], m=[w';v']) = [ w×w' ; w×v' + v×w' ]
   res0 = -w2 m1 + w1 m2;  res1 = w2 m0 - w0 m2;  res2 = -w1 m0 + w0 m1
   res3 = -w2 m4 + w1 m5;  res4 = w2 m3 - w0 m5;  res5 = -w1 m3 + w0 m4
   res3 += -v2 m1 + v1 m2; res4 += v2 m0 - v0 m2; res5 += -v1 m0 + v0 m1      (exact add order)

crossForce(vel=[w;v], f=[τ;F]) = [ w×τ + v×F ; w×F ]
   res0..2 = w×τ (same pattern as above), res3..5 = w×F,
   then res0 += -v2 F1 + v1 F2; res1 += v2 F0 - v0 F2; res2 += -v1 F0 + v0 F1
```

**`mju_inertCom(res[10], inert[3], mat[9], dif[3], mass)`** — 10-vector spatial inertia about a point (the tree CoM) with world orientation:
```
tmp = diag(inert) * mat'       (tmp[3r+c] = mat[3c+r]*inert[r])
res0 = m0 tmp0 + m1 tmp3 + m2 tmp6      (I_xx)   res1 = m3 tmp1 + m4 tmp4 + m5 tmp7  (I_yy)
res2 = m6 tmp2 + m7 tmp5 + m8 tmp8      (I_zz)   res3 = m0 tmp1 + m1 tmp4 + m2 tmp7  (I_xy)
res4 = m0 tmp2 + m1 tmp5 + m2 tmp8      (I_xz)   res5 = m3 tmp2 + m4 tmp5 + m5 tmp8  (I_yz)
res0 += mass(d1²+d2²); res1 += mass(d0²+d2²); res2 += mass(d0²+d1²)
res3 -= mass d0 d1;    res4 -= mass d0 d2;    res5 -= mass d1 d2
res6..8 = mass*dif ;   res9 = mass
```
Layout of `cinert`/`crb` 10-vectors: `[Ixx, Iyy, Izz, Ixy, Ixz, Iyz, h_x, h_y, h_z, m]` with `h = m (x_com_body − x_ref)`.

**`mju_mulInertVec(res, i, v)`** (6×6 inertia times motion `[ω; v]`):
```
res0 = i0 v0 + i3 v1 + i4 v2 - i8 v4 + i7 v5
res1 = i3 v0 + i1 v1 + i5 v2 + i8 v3 - i6 v5
res2 = i4 v0 + i5 v1 + i2 v2 - i7 v3 + i6 v4
res3 = i8 v1 - i7 v2 + i9 v3
res4 = i6 v2 - i8 v0 + i9 v4
res5 = i7 v0 - i6 v1 + i9 v5
```
i.e. `f = [Iω + h×v ; m v − h×ω]`.

**`mju_dofCom(res, axis, offset)`**: if `offset != NULL` (rotational dof): `res = [axis ; axis × offset]`; else (slide): `res = [0 ; axis]`. With `offset = subtree_com − xanchor`, `axis × offset` is the linear velocity of the reference point for unit joint rate.

**`mju_transformSpatial(res, vec, flg_force, newpos, oldpos, rotnew2old)`**:
```
tran = vec; dif = newpos - oldpos
force:  tran[0:3] = vec[0:3] - dif × vec[3:6]          (torque about new point)
motion: tran[3:6] = vec[3:6] - dif × vec[0:3]          (linear vel at new point)
if rotnew2old: res[0:3] = R' tran[0:3]; res[3:6] = R' tran[3:6]  else res = tran
```

`mju_makeFrame(frame)` (contact frame from normal): normalize x (error if |x| < 0.5); if `|y|² < 0.25` set y = (0,1,0) if `-0.5 < x1 < 0.5` else (0,0,1); Gram-Schmidt `y -= x(x·y)`, normalize; `z = x × y`.

---

## 2. Position stage

### 2.1 `mj_kinematics1` — body poses
Inputs: `qpos, qpos0, mocap_pos/quat, body_{parentid,jntadr,jntnum,mocapid,pos,quat}, jnt_{type,qposadr,pos,axis}`. Outputs: `xpos, xquat, xmat, xanchor, xaxis` (and `xipos, ximat` of world).

World: `xpos[0]=0, xquat[0]=(1,0,0,0), xipos[0]=0, xmat[0]=ximat[0]=I`.

For `i = 1..nbody-1` (in body order; parents precede children). With sleep enabled, skip `body_awake[i] == mjS_STATIC` (−1).

**Free joint body** (`jntnum == 1 && type == FREE`):
```
xpos = qpos[qadr:qadr+3];  xquat = normalize4(copy(qpos[qadr+3:qadr+7]))
xanchor[j] = xpos;  xaxis[j] = jnt_axis[j]   (no rotation applied)
```
(the free body's `body_pos/quat` are ignored at runtime; they seed `qpos0`).

**Otherwise**:
```
if body_mocapid[i] >= 0:  bodypos = mocap_pos[id];  bodyquat = normalize4(copy(mocap_quat[id]))
else:                     bodypos = body_pos[i];     bodyquat = body_quat[i]
if pid != 0:  xpos = xmat[pid] * bodypos + xpos[pid]     # NOTE: rotation via MATRIX, not quaternion
              xquat = xquat[pid] * bodyquat
else:         xpos = bodypos;  xquat = bodyquat
for each joint j of the body (in order):
   xaxis   = rotVecQuat(jnt_axis[j], xquat)        # current (accumulated) frame
   xanchor = rotVecQuat(jnt_pos[j], xquat) + xpos
   SLIDE:  xpos += xaxis * (qpos[qadr] - qpos0[qadr])
   BALL:   qloc = normalize4(copy(qpos[qadr:qadr+4]))
   HINGE:  qloc = axisAngle2Quat(jnt_axis[j], qpos[qadr] - qpos0[qadr])    # LOCAL axis
   BALL/HINGE: xquat = xquat * qloc                 # mju_mulQuat (aliasing-safe)
               xpos  = xanchor - rotVecQuat(jnt_pos[j], xquat)   # rotation about the anchor
   store xanchor[j], xaxis[j]
```
(Ball `jnt_axis` is set to (0,0,1) by the compiler; slide/hinge offsets are relative to `qpos0`, i.e. `ref`. Ball/free quaternions are absolute.)

Then `normalize4(xquat)`. Sleep check: if sleep enabled, `jntnum > 0` and `body_awake[i] == mjS_ASLEEP` (0): compare new `xpos/xquat` bitwise with stored values; equal → skip storing (continue); different → `tree_awake[body_treeid[i]] = 1` (woken in `mj_wake`). Store `xquat`, `xpos`, `xmat = quat2Mat(xquat)`.

`mj_kinematics = kinematics1; if mj_wake(): mj_updateSleep(); kinematics2`.

### 2.2 `mj_kinematics2` — inertial frames, geoms, sites
Uses `mj_local2Global(d, xpos_out, xmat_out, pos, quat, body, sameframe)`:
```
position (if xpos and pos):
  NONE / BODYROT / INERTIAROT:  xpos = xmat[body]*pos + xpos[body]
  BODY:                         xpos = xpos[body]
  INERTIA:                      xpos = xipos[body]
orientation (if xmat and quat):
  NONE:               xmat = quat2Mat(xquat[body] * quat)
  BODY / BODYROT:     xmat = xmat[body]
  INERTIA / INERTIAROT: xmat = ximat[body]
```
`mjtSameFrame`: NONE=0, BODY=1, INERTIA=2, BODYROT=3, INERTIAROT=4 (compile-time detection of identical frames — an optimization that also changes rounding: copies instead of recomputation).

* Bodies `b ≥ 1` (awake/non-static subset under sleep): `xipos, ximat` from `body_ipos, body_iquat, body_sameframe`.
* Geoms of all bodies including world (skip static under sleep): `geom_xpos, geom_xmat` from `geom_pos, geom_quat, geom_sameframe`.
* Sites: `site_xpos, site_xmat`; skipped if body not `mjS_AWAKE` under sleep.

### 2.3 `mj_comPos` — subtree CoM, `cinert`, `cdof`
```
subtree_com[i] = body_mass[i] * xipos[i]                 for all bodies
for i = nbody-1 .. 1:  subtree_com[parent] += subtree_com[i]          (backward accumulation)
   (sleep: if body asleep, the stored subtree_com is a position; add subtree_com[i]*body_subtreemass[i])
for all i: if body_subtreemass[i] < mjMINVAL: subtree_com[i] = xipos[i]
           else subtree_com[i] *= 1/body_subtreemass[i]   (mju_scl3 by reciprocal)
cinert[0] = 0
for i >= 1: offset = xipos[i] - subtree_com[body_rootid[i]]
            cinert[i] = inertCom(body_inertia[i], ximat[i], offset, body_mass[i])
for i >= 1 with joints, for j in body joints: da = 6*jnt_dofadr[j]
   offset = subtree_com[rootid[i]] - xanchor[j]
   FREE:  cdof[da..da+18) = 0; cdof[da+3]=cdof[da+10]=cdof[da+17]=1    # unit world translations
          then 3 rotational dofs as BALL below at da+18
   BALL:  for k=0..2: axis = column k of xmat[i] (= k-th child-frame axis in world)
          cdof = dofCom(axis, offset)
   SLIDE: cdof = dofCom(xaxis[j], NULL) = [0; xaxis]
   HINGE: cdof = dofCom(xaxis[j], offset) = [xaxis; xaxis × offset]
```
Important: ball/free rotational dofs use the **body** frame axes (`xmat[i]` after all joints of that body), "assuming no subsequent rotations" — consistent with the compiler restriction that a ball joint is the last rotational joint in its body. `body_subtreemass` is a model constant (compile time).

### 2.4 `mj_camlight`
Cameras (skip if sleep and body not awake and (no target or target not awake)):
```
cam_xpos, cam_xmat = local2Global(cam_pos, cam_quat, cam_bodyid, sameframe=0)
FIXED:          as above
TRACK:          cam_xmat = cam_mat0;  cam_xpos = xpos[body] + cam_pos0
TRACKCOM:       cam_xmat = cam_mat0;  cam_xpos = subtree_com[body] + cam_poscom0
TARGETBODY(COM) (only if targetbodyid >= 0):
   pos = xpos[target] (or subtree_com[target])
   z = normalize3(cam_xpos - pos); x = normalize3((0,0,1) × z); y = normalize3(z × x)
   cam_xmat = transpose([x; y; z])          (columns x,y,z)
```
Lights: `light_xpos = local2Global(light_pos)`, `light_xdir = rotVecQuat(light_dir, xquat[body])`; TRACK/TRACKCOM: `light_xdir = light_dir0`, pos = `xpos[body] + light_pos0` / `subtree_com[body] + light_poscom0`; TARGETBODY(COM) with target ≥ 0: `light_xdir = lookat − light_xpos`. Finally always `normalize3(light_xdir)`.

### 2.5 `mj_flex` / `mj_flexSkip(m, d, skiprigid=1)` (kinematic part only)
(Flex physics is covered elsewhere; this is what the smooth stage computes.)
* Invalidates `flex_hessian_valid`.
* Vertex positions: `flex_interp[f] == 0`: `flexvert_xpos = xpos[vertbody]` if `flex_centered[f]` or `flex_vert == 0` exactly, else `xmat[vertbody]*flex_vert + xpos`. Interpolated (`interp != 0`, order `|interp|`): node positions (same rule with `flex_node`), shell mode (`interp < 0`) fills interior nodes via `mju_shellTrackInterior` (transfinite interpolation); grid `nx = cx*order+1` etc. (error if node count mismatches); each vertex: `mju_cellLookup(flex_vert0, cellnum, order, flex_vertcell)` + `mju_interpolate3D`.
* Element AABBs `flexelem_aabb` (center, half-size + `flex_radius`), dynamic BVH update (`mj_updateDynamicBVH`: leaves marked, internal nodes recomputed back-to-front from children if any child modified) unless `mjDSBL_MIDPHASE`.
* Edge lengths and edge Jacobians (`flexedge_J`, sparse rows `flexedge_J_rowadr/rownnz`, zeroed first over `nJfe`): skipped for rigid (if `skiprigid`) or interpolated flexes; Jacobian skipped unless `flex_edgeequality==1 || edgedamping || edgestiffness || flex_damping`. `vec = normalize3(pos2-pos1)`, `flexedge_length = norm`; `NV = mj_jacDifPair(sparse, skipcommon=0)`; `J = jacdif' * vec` (3×NV transposed product). Error if `NV > flexedge_J_rownnz`.

### 2.6 `mj_tendon` — lengths, Jacobians, wrap paths
Outputs: `ten_length` (L), `ten_J` (sparse CSR with **model-fixed** pattern `ten_J_rownnz/rowadr/colind`), `ten_wrapadr, ten_wrapnum, wrap_obj, wrap_xpos`. Work arrays: `jac1, jac2, jacdif (3nv), tmp (nv), chain (nv)`.

`ten_J` sparsity (`engine_setconst.c:makeTendonSparse`): fixed tendon: `colind = jnt_dofadr[wrap_objid]` for each joint, `rownnz = num` (duplicates possible if a joint repeats); spatial tendon: union of all dofs of every body on the path from each site/geom body up to the world (walking `body_parentid`, adding `body_dofadr..+dofnum`), sorted ascending (bubble sort).

```
L = 0 (all tendons); J = 0 (all nJten)
wrapcount = 0
for each tendon i (skip asleep under sleep filter → its L stays 0 and path is not rebuilt):
  ten_wrapadr[i] = wrapcount; ten_wrapnum[i] = 0
  FIXED: for each joint k with coef = wrap_prm:
           L += coef * qpos[jnt_qposadr[k]]          # first qpos of the joint (meant for slide/hinge)
           J[row i] += coef at colind == jnt_dofadr[k]   (mju_combineSparseInc, see note)
         continue
  SPATIAL: divisor = 1; j = 0
  while j < num-1:
    type0,type1,id0,id1 = wrap objects j, j+1
    if type0 == PULLEY or type1 == PULLEY:
        if type0 == PULLEY: divisor = wrap_prm[j]; wrap_xpos[wc]=0; wrap_obj[wc]=-2; wrapnum++, wc++
        j++; continue
    wpnt[0:3] = site_xpos[id0]; wbody[0] = site_bodyid[id0];  wlen = -1
    if type1 in {SPHERE, CYLINDER}:
        wraptype = type1; wrapid = id1; (type1,id1) = object j+2 (must be a site)
        sideid = round(wrap_prm[j+1]);  (error if < -1 or >= nsite)
        wlen = mju_wrap(wpnt+3, site_xpos[id0], site_xpos[id1], geom_xpos[wrapid], geom_xmat[wrapid],
                        geom_size[3*wrapid+0] (radius), wraptype, sideid>=0 ? site_xpos[sideid] : NULL)
    else wraptype = NONE
    if wlen < 0:  wpnt[3:6] = site_xpos[id1]; wbody[1] = site_bodyid[id1]
                  L += dist(wpnt0, wpnt1) / divisor
    else:         wpnt[9:12] = site_xpos[id1]; wbody[1]=wbody[2]=geom_bodyid[wrapid]; wbody[3]=site_bodyid[id1]
                  L += (dist(p0,p1) + wlen + dist(p2,p3)) / divisor
    for k in 0 .. (wlen<0 ? 0 : 2):          # straight segments only; the arc carries no Jacobian
        if wbody[k] != wbody[k+1]:
           dif = normalize3(wpnt[k+1] - wpnt[k])
           NV = mj_jacDifPair(chain, wbody[k], wbody[k+1], wpnt[k], wpnt[k+1], ..., issparse=1, skipcommon=0)
           if NV: tmp = jacdif' * dif  (NV)
                  J[row i] += (1/divisor) * tmp at common indices (mju_combineSparseInc, src_ind = chain)
    wrap_xpos[wc..] = wpnt[0] (or wpnt[0..2] if wrapped: site0, tangent0, tangent1)
    wrap_obj[wc] = -1; if wrapped: wrap_obj[wc+1] = wrap_obj[wc+2] = wrapid
    wrapnum, wc += (wrapped ? 3 : 1)
    j += (wraptype != NONE ? 2 : 1)          # a geom that failed to wrap is still skipped
    if j == num-1 or wrap_type[j+1] == PULLEY:  # append the last site of the branch
        wrap_xpos[wc] = site_xpos[id1]; wrap_obj[wc] = -1; wrapnum++, wc++
```
Notes:
* The Jacobian at a segment is `d|p_b − p_a|/dq = û' (J_b − J_a)`. Because the arc length depends on the tangent points whose motion is orthogonal-ish, MuJoCo uses only the straight segments (exact for smooth wraps since the tangent-point derivative terms cancel).
* `mju_combineSparseInc(dst, src, n, a, b, dst_nnz, src_nnz, dst_ind, src_ind)`: if patterns identical (`nnz` equal and `mju_compare` of indices) → `dst = a*dst + b*src` elementwise (`mju_addToSclScl`); else scale `dst *= a` if `a != 1`, then merge-walk adding `b*src` only at **common** indices (src indices not in dst are dropped). With duplicate indices in `dst_ind`, only the first match receives the contribution.
* `mj_jacDifPair` with `issparse=1` returns Jacobians on the merged chain `chain[0..NV)` (sorted ascending), see §4.

**`mju_wrap(wpnt[6], x0, x1, xpos, xmat, radius, type, side)`** (`engine_util_misc.c`):
```
p0 = xmat' (x0 - xpos);  p1 = xmat' (x1 - xpos)            # local frame of the geom
if |p0| < mjMINVAL or |p1| < mjMINVAL: return -1
SPHERE:  axis0 = normalize3(p0)
         normal = p0 × p1; nrm = normalize3(normal)
         if nrm < mjMINVAL:     # p0 ∥ p1
            i = argmax|axis0| (ties → lowest index by strict '>' tests: i=1 if |a1|>|a0|&&|a1|>|a2|; i=2 if |a2|>|a0|&&|a2|>|a1|)
            axis1 = (1,1,1) with axis1[i] = 0;  normal = normalize3(axis0 × axis1)
         axis1 = normalize3(normal × axis0)
CYLINDER: axis0 = (1,0,0), axis1 = (0,1,0)     # cylinder axis = local z
d = (p0·axis0, p0·axis1, p1·axis0, p1·axis1)
if side: s = xmat'(side - xpos); sd = (s·axis0, s·axis1); mju_normalize(sd,2) ((1,0) if tiny); sd *= radius
if side and |s| < radius:  wlen = wrap_inside(pnt, d, radius)
else:                      wlen = wrap_circle(pnt, d, side ? sd : NULL, radius)
if wlen < 0: return -1
res_k = axis0*pnt[2k] + axis1*pnt[2k+1]   (k=0,1)
CYLINDER: L0 = |p0.xy - res0.xy|, L1 = |p1.xy - res1.xy|
          res0.z = p0.z + (p1.z-p0.z) * L0/(L0+wlen+L1)
          res1.z = p0.z + (p1.z-p0.z) * (L0+wlen)/(L0+wlen+L1)
          wlen = sqrt(wlen² + (res1.z-res0.z)²)          # helix length
wpnt = xmat*res + xpos ;  return wlen
```

**`wrap_circle(pnt[4], end[4], side, radius)`** (2D, circle centered at origin):
```
sqlen0 = |end0|², sqlen1 = |end1|², sqrad = r²
if sqlen0 < sqrad or sqlen1 < sqrad or r < mjMINVAL: return -1
dif = end1 - end0; dd = |dif|²; if dd < mjMINVAL: return -1
a = clamp(-(dif·end0)/dd, 0, 1); clear = |a*dif + end0|² > sqrad
if side: wside = wind(end0, side) + wind(side, end1)          # wind(p,q) = atan2(p×q, p·q)
         if clear and |wside - wind(end0,end1)| < π: return -1   # straight segment homotopic to side path
         ccw = (wside > 0)
elif clear: return -1
sqrt0 = sqrt(sqlen0 - sqrad), sqrt1 = sqrt(sqlen1 - sqrad)
for i in {0,1}, sgn = (i==0 ? +1 : -1):
   sol[i][0] = ( (e0x sqrad + sgn r e0y sqrt0)/sqlen0, (e0y sqrad - sgn r e0x sqrt0)/sqlen0 )
   sol[i][1] = ( (e1x sqrad - sgn r e1y sqrt1)/sqlen1, (e1y sqrad + sgn r e1x sqrt1)/sqlen1 )
   good[i] = -|sol[i][0] - sol[i][1]|²;  if is_intersect(end0, sol[i][0], end1, sol[i][1]): good[i] = -10000
i = side ? ccw : (good[0] > good[1] ? 0 : 1)
pnt = sol[i]; if is_intersect(end0, pnt0, end1, pnt1): return -1
if side: return r * |wside - wind(end0, pnt0) - wind(pnt1, end1)|
return length_circle(pnt0, pnt1, i, r)
```
`length_circle(p0,p1,ind,r)`: `cross = p0y p1x − p0x p1y`; `angle = atan2(|cross|, p0·p1)`; if `angle > 1e-6 && ((cross>0 && ind) || (cross<0 && !ind))`: `angle = 2π − angle`; return `r*angle`.
`is_intersect(p1,p2,p3,p4)`: `u=p2−p1, v=p4−p3, det = v1 u0 − v0 u1`; `|det| < mjMINVAL` → false; `s = sign(det)`; `w=p1−p3, d=p2−p4`; `a0=(v0w1−v1w0)s, a1=(v0d1−v1d0)s, b0=(u0w1−u1w0)s, b1=(u0d1−u1d0)s`; return `a0≥0 && a1<0 && b0≥0 && b1<0`.

**`wrap_inside(pnt, end, radius)`** (sidesite inside the circle: tendon passes "through" the geom touching at one point): maxiter 20, tolerance 1e-6.
```
len0=|end0|, len1=|end1|, dif=end1-end0, dd=|dif|²
if len0 <= r or len1 <= r or r < mjMINVAL or len0 < mjMINVAL or len1 < mjMINVAL: return -1
if dd > mjMINVAL: a = -(dif·end0)/dd; if 0 < a < 1 and |end0 + a dif| <= r: return -1
pnt0 = pnt1 = r * normalize(0.5(end0+end1))           # fallback
A = r/len0, B = r/len1, cosG = (len0²+len1²-dd)/(2 len0 len1)
if cosG < -1+mjMINVAL: return -1;  if cosG > 1-mjMINVAL: return 0
G = acos(cosG); phi = π/2; z = sin(phi); f = asin(Az)+asin(Bz)-2phi+G
if f > 0: return 0
Newton (iter < 20 and |f| > 1e-6):
   df = cos(phi)(A/max(mjMINVAL, sqrt(1-z²A²)) + B/max(mjMINVAL, sqrt(1-z²B²))) - 2
   if df > -mjMINVAL: return 0;  phi1 = phi - f/df;  if phi1 > phi: return 0
   phi = phi1; z = sin(phi); f = ...;  if f > 1e-6: return 0
if iter >= 20: return 0
if end0×end1 > 0: vec = end0, ang = phi - asin(A z) else vec = end1, ang = phi - asin(B z)
vec = normalize(vec); pnt0 = pnt1 = r * Rot(ang) vec;  return 0        # zero arc length
```

### 2.7 `mj_tendonDot(m, d, id, vec)` — `(dJ_ten/dt)·vec`
Returns 0 for invalid id or fixed tendons. Only site-only spatial tendons supported (geom wrap → `mjERROR("geom wrapping not supported")`; the compiler forbids geom wrapping with tendon armature). For each straight site–site segment (pulleys set divisor, as in `mj_tendon`):
```
wpnt0, wpnt1 = site positions; wvel_k = linear part of mj_objectVelocity(SITE, idk, flg_local=0)
if bodies differ:
   dpnt = normalize3(wpnt1 - wpnt0) (norm kept)
   dvel = (wvel1 - wvel0);  dvel -= dpnt (dpnt·dvel);  dvel *= (norm > mjMINVAL ? 1/norm : 0)   # d/dt of unit vector
   term1 = (Jdot1 - Jdot0)' dpnt ;  term2 = (J1 - J0)' dvel
   res += Σ_k (term_k / divisor) * vec[k]     (sparse: over merged chain; dense: mju_dot/divisor)
```
Sparse vs dense chosen by `mj_isSparse(m)`; sparse uses `mj_mergeChain(skipcommon=0)` + `mj_jacDotSparse`/`mj_jacSparse`. Summation order: sparse adds `(tmp[k]/divisor)*vec[chain[k]]` per element; dense adds `mju_dot(tmp,vec,nv)/divisor`.

### 2.8 `mj_transmission` — actuator lengths and moments
Outputs: `actuator_length[nout]`, sparse `actuator_moment` with **data-dependent** pattern `moment_rownnz/rowadr/colind` (in `mjData`). Each actuator `i` owns output rows `actuator_outadr[i] .. +actuator_outnum[i]` (1 row for all types except `mjTRN_SO3` = 3 rows). `rowadr[out] = (out==0 ? 0 : rowadr[out-1] + rownnz[out-1])` — contiguous, in output order. Sleeping actuators (sleep filter, `nv_awake < nv`): all their rows get `rownnz = 0`. `gear = actuator_gear + 6*out`.

* **JOINT / JOINTINPARENT**, `id = trnid[0]`:
  * slide/hinge: `rownnz=1`, `colind = jnt_dofadr`, `length = qpos[qposadr]*gear0`, `moment = gear0`.
  * ball: `q = normalize4(qpos)`, `axis = quat2Vel(q, 1)` (expmap); `gearAxis = gear[0:3]` (JOINT) or `rotVecQuat(gear[0:3], neg(q))` (JOINTINPARENT); `length = axis·gearAxis`; `rownnz=3`, `colind = dofadr+0..2`, `moment = gearAxis`.
  * free: `length = 0`; `gearAxis = gear[3:6]` (JOINT) or `rotVecQuat(gear[3:6], neg(normalize4(qpos[3:7])))`; `rownnz = 6`, `moment = [gear[0:3], gearAxis]`.
* **SLIDERCRANK** (`id` = crank site, `idslider = trnid[1]`, `rod = actuator_cranklength`):
```
axis = z-column of site_xmat[idslider]; vec = site_xpos[id] - site_xpos[idslider]
av = vec·axis; det = av² + rod² - vec·vec
if det <= 0: ok=0; sdet=0; length = av      else sdet = sqrt(det); length = av - sdet
ok:  dldv = axis*(1 - av/sdet) + vec/sdet ;  dlda = vec*(1 - av/sdet)
!ok: dlda = vec ; dldv = axis
mj_jacPointAxis(jacS, jacA, site_xpos[idslider], axis, site_bodyid[idslider])   # jacA = jacr × axis (columnwise)
mj_jacSite(jac, 0, id);  jac -= jacS
moment_row[j] = Σ_k dlda_k jacA[k,j] + dldv_k jac[k,j]        (dense, nv)
length *= gear0;  compress: keep j with moment_row[j] != 0 exactly, moment = moment_row*gear0
```
* **TENDON**: `length = ten_length*gear0`; pattern copied from `ten_J` (`rownnz = ten_J_rownnz`, colind copy), `moment = ten_J*gear0` (no compression; explicit zeros kept).
* **SO3** (3 output rows, no gear): 
  * `trnid[1] == -1` (ball joint): `q = normalize4(qpos)`, `axis = quat2Vel(q,1)`; row k: `length = axis[k]`, `rownnz=1`, `colind = dofadr+k`, `moment = 1`.
  * site+refsite: `quat = xquat[body(id)]*site_quat[id]`, `refquat` likewise; `vec = subQuat(quat, refquat)`; `jacS = jacr(site) − jacr(refsite)`; clear columns along the dof chain of the common ancestral dof (algorithm below); row k: `length = vec[k]`, wrench = k-th column of `site_xmat[id]`, `moment_row = jacS' wrench`, compress exact zeros.
* **SITE** (`mj_jacSite(jac, jacS, id)`; `length = 0`):
  * no refsite: `wrench_t = site_xmat[id]*gear[0:3]`, `wrench_r = site_xmat[id]*gear[3:6]`; `moment_row = jac' wrench_t + jacS' wrench_r` (computed as `mulMatTVec` then `mulMatTVec` into `jac` then `addTo`). length stays 0.
  * refsite `refid`: common ancestral dof:
```
b0 = weldid[site_body[id]], b1 = weldid[site_body[refid]]
d0 = dofadr[b0]+dofnum[b0]-1, d1 = likewise        (=-1/-2 if no dofs)
common = -1; if d0>=0 and d1>=0: while d0 != d1: (larger one ← dof_parentid); break if either -1
         if d0 == d1: common = d0
moment_row = 0
if gear[0:3] != 0:  vec = site_xmat[refid]' (site_xpos[id]-site_xpos[refid]); length += vec·gear[0:3]
                    jac -= jacp(refid); zero columns da = common, parent(common), ... (all 3 rows)
                    moment_row = jac' (site_xmat[refid]*gear[0:3])
if gear[3:6] != 0:  quat/refquat from xquat*site_quat (avoids mat2Quat)
                    vec = subQuat(quat, refquat); length += vec·gear[3:6]
                    jacS -= jacr(refid); zero common chain; moment_row += jacS' (site_xmat[refid]*gear[3:6])
```
  then compress exact zeros. (Wrench expressed in the **reference** frame when refsite exists.)
* **BODY** (adhesion; `id` = body): `length = 0`. On first BODY actuator allocate `efc_force` marker (nefc), `moment_exclude` (nv), jac buffers. For each contact `j` with both geoms ≥ 0 (non-flex) and `geom_bodyid` of either equal to `id`:
  * `!con->exclude`: skip if `efc_address < 0`; `counter++`; if `dim == 1` or elliptic cone: `efc_force[adr] = 1`; else pyramidal: `efc_force[adr+k] = 0.5/(dim-1)` for `k < 2(dim-1)`.
  * `con->exclude == 1` (in gap): `counter++`; `NV = mj_jacDifPair(b1,b2,con->pos,con->pos, issparse)`; `jac = frame[0:3]' jacdif` (normal row, 1×NV); accumulate into `moment_exclude` (scatter by chain in sparse case).
  * other exclude values (2 fused, 3 no dofs, 4 passive) ignored.
  * if `counter`: `moment_row = J_efc' efc_force` (`mj_mulJacTVec`) `+ moment_exclude`, then `*= -1/counter`. Compress exact zeros.

### 2.9 Composite rigid body inertia: `mj_crb`, `mj_tendonArmature`, `mj_makeM`

**Sparsity of M (model constants).** `M` is stored lower-triangular CSR, "reduced" by simple dofs:
* `body_simple[b]` (compiler, `user_model.cc`): 1 if body inertial frame = body frame (`IsNullPose(ipos, iquat)`) and (body is a tree root, or its parent is a fixed child of world), the user did not disable `simple`, the body has no children (a parent body is never simple), no joint has non-zero `pos`, hinge/slide axes are axis-aligned (exactly one component with |·| > mjEPS), and no rotational joint follows another rotation. Promoted to 2 if all its joints are slides. Demoted to 0 for bodies touched by a tendon with non-zero `tendon_armature` (`FinalizeSimple`).
* `dof_simplenum[i]` = number of consecutive simple dofs starting at i (run-length computed from the end; 0 if dof's body not simple).
* `M_rownnz[i] = 1` if `dof_simplenum[i]` (diagonal only), else `1 + (#ancestors of i in dof_parentid chain)`. `M_colind` for row i: ancestors in **ascending** order followed by i (diagonal is the **last** element: `M_rowadr[i] + M_rownnz[i] − 1`). `nC = nv + Σ_{non-simple} #ancestors`.
* `D_*`: full symmetric (non-reduced, upper+lower) pattern, `D_diag[i]` = position of diagonal in row i; `nD`. Identical rows to `B` (body-dof) pattern for `dof_bodyid`.
* `nM`: legacy "qM" layout size (non-reduced lower): row i at `dof_Madr[i]`, stored diagonal first then ancestors **descending**.
* `mapM2D[nD]`: for each D entry the index in M (or −1 if absent, e.g. off-diagonals of simple dofs which are identically 0) — used with `mju_gatherMasked`. `mapD2M[nC]`: for each M entry its index in D. `mapM2M[nC]`: for each M (CSR) entry its index in legacy qM.
* `B_*`: body-dof incidence (subtree dofs ∪ ancestor dofs, sorted).

*Worked example.* World → body A (hinge, dof 0) → body B (ball, dofs 1,2,3); separate tree: body C (free joint, simple, dofs 4..9; assume `body_simple[C]=1`). `dof_parentid = [-1, 0, 1, 2, -1, 4, 5, 6, 7, 8]`. `dof_simplenum = [0,0,0,0,6,5,4,3,2,1]`.
```
row 0: colind [0]            rownnz 1  rowadr 0
row 1: colind [0,1]          rownnz 2  rowadr 1
row 2: colind [0,1,2]        rownnz 3  rowadr 3
row 3: colind [0,1,2,3]      rownnz 4  rowadr 6
rows 4..9: colind [i]        rownnz 1  rowadr 10..15        → nC = 16
```
(Without the simple flag, row 9 would be `[4,5,6,7,8,9]`; the free joint's 6×6 block would be dense-lower.) The D pattern stores both triangles: row 0 = `[0,1,2,3]`, row 1 = `[0,1,2,3]`, …

**`mj_crb`**:
```
crb = cinert (awake bodies)
for i = nbody-1 .. 0 (parents order): if body_parentid[i] > 0: crb[parent] += crb[i]      # world excluded
M = 0 over nC (or awake rows)
for each dof i (ascending):
  adr = M_rowadr[i]
  if dof_simplenum[i]: M[adr] = dof_M0[i]; continue          # constant diagonal precomputed at qpos0
  Madr = adr + M_rownnz[i] - 1                                # diagonal
  M[Madr] = dof_armature[i] + mj_actuatorArmature(JOINT, dof_jntid[i])
  buf = mulInertVec(crb[dof_bodyid[i]], cdof[i])
  for j = i, dof_parentid[j], ... while j >= 0:  M[Madr--] += dot6(cdof[j], buf)
```
i.e. `M_ij = cdof_j' · (I^c_{body(i)} cdof_i)` for j ancestor-or-self of i, `I^c` = composite inertia of the subtree of body(i).

`dof_M0` (`engine_setconst.c:mj_setM0`, at qpos0 during `mj_setConst`): same formula for all dofs `M0_i = armature_i + actuatorArmature + cdof_i·(crb cdof_i)` (with `mju_dot`, not dot6).

`mj_actuatorArmature(m, type, id)`: `actuatorid = jnt_actuatorid[id]` or `tendon_actuatorid[id]`; −1 → 0; ≥0 → `actuator_armature[a] * gear0²` (`gear0 = actuator_gear[6*actuator_outadr[a]]`); < −1 → scan all actuators with `trnid[0]==id` and matching type (JOINT/JOINTINPARENT for joints, TENDON for tendons), sum `armature*gear0²`. `mj_actuatorDamping` analogous (also adds `actuator_dampingpoly*gear0²` into `poly`).

**`mj_tendonArmature`**: for each awake tendon with `a = tendon_armature + actuatorArmature(TENDON) != 0`:
```
for each nonzero J entry (col i, value J_i) in the tendon row:
   M[row i, :] += a * J_i * J[:]     at common indices of M row i and the tendon row (mju_addToSclSparseInc)
```
→ `M += a J'J` restricted to the M pattern (row i holds ancestors-or-self of i, so only entries `j ≤ i` that are dof-ancestors of i receive contributions; couplings between non-ancestor dofs are **dropped**). No entry for simple dofs is touched since such bodies were demoted.

`mj_makeM = mj_crb + mj_tendonArmature` (timer `mjTIMER_POS_INERTIA`).

### 2.10 Factorization `mj_factorM` / `mj_factorI`
`qLD = copy(M)` (nC; awake rows only under sleep), then in-place `L'DL`:
```
clamped = -1
for j = nv-1 .. 0:            k = index ? index[j] : j
   start = rowadr[k]; diag = rownnz[k]-1; end = start+diag
   if mat[end] < mjMINVAL: mat[end] = mjMINVAL; clamped = (clamped<0 ? k : clamped)
   invD = 1/mat[end];  diaginv[k] = invD
   for adr = end-1 .. start:  i = colind[adr]
       mat[rowadr[i] : rowadr[i]+rownnz[i]] -= mat[start : start+rownnz[i]] * (mat[adr]*invD)
   mat[start : end] *= invD               # row k off-diagonals become L(k,·); diagonal stays D_k
return clamped
```
Correctness of the prefix update: row i's columns are exactly the first `rownnz[i]` columns of row k when i is an ancestor of k (both lists are ascending ancestor chains). If `clamped >= 0`: `mj_warning(d, mjWARN_INERTIA, clamped)` (count + print once). Result: `M = L' D L`, L unit lower-triangular in the M pattern; `qLD` diagonal holds D, off-diagonals hold L; `qLDiagInv = 1/D`. Simple rows (rownnz = 1) just get `invD`. Note the multiplication by `invD` rather than division.

### 2.11 Back-substitution
**`mj_solveLD(x, qLD, qLDiagInv, nv, n, rownnz, rowadr, colind, index)`** — in place `x ← M⁻¹ x` for n right-hand sides stored consecutively (stride nv):
```
# x <- L^-T x
for k = nv-1 .. 0: i = index?index[k]:k; if rownnz[i]==1: continue
   for each rhs: xi = x[i]; if xi != 0: for adr in row i (excluding diag): x[colind[adr]] -= qLD[adr]*xi
# x <- D^-1 x
for k: x[i] *= qLDiagInv[i]
# x <- L^-1 x
for k = 0 .. nv-1: if rownnz[i]==1: continue
   x[i] -= dotSparse(qLD[row i], x, rownnz[i]-1, colind)      (4-accumulator dot)
```
`mj_solveM(m,d,x,y,n)`: copy `y→x` if different, then `mj_solveLD(..., index=NULL)`.

**`mj_solveM2` / `mj_solveM2_impl(x, y, sqrtInvD, factor, ...)`**: `x = sqrt(D⁻¹) L⁻ᵀ y` (so that `x'x = y'M⁻¹y`): copy; `for i = nv-1 .. 1`: skip if `diagnum[i]` (`dof_simplenum`), scatter as above; then `x[i] *= sqrtInvD[i]`.

**`mj_mulM`** = `mju_mulSymVecSparse`: `res=0; for i: res[i] = row[diag]*vec[i]; for k = diag-1..0: j=colind; res[i] += val*vec[j]; res[j] += val*vec[i]`.
**`mj_mulM2`**: `res = sqrt(D) L vec` with `res[i] = vec[i] + (non-simple ? dotSparse(qLD row i offdiag, vec) : 0)`, then `res[i] *= sqrt(qLD[diag_i])` (so `res'res = vec'Mvec`).
**`mj_fullM`**: `mju_sym2dense`.

---

## 3. Velocity stage

### 3.1 `mj_comVel` — `cvel`, `cdof_dot`
```
cvel[0] = 0
for i = 1..nbody-1 (awake): cvel = cvel[parent]; bda = body_dofadr[i]
  for j = 0 .. dofnum-1 (by joint type of dof bda+j):
    FREE:  cdofdot[0..3) = 0
           cvel += mulDofVec(cdof[bda..], qvel[bda..], 3)        # translation first
           j += 3;  fall through to BALL
    BALL:  cdofdot[j+k] = crossMotion(cvel, cdof[bda+j+k]) k=0..2   # uses velocity BEFORE the ball's own rotation
           cvel += mulDofVec(cdof[bda+j..], qvel[bda+j..], 3);  j += 2
    default (slide/hinge): cdofdot[j] = crossMotion(cvel, cdof[bda+j]) (pre-update velocity; cdof×cdof = 0)
           cvel += cdof[bda+j]*qvel[bda+j]
  cvel[i] = cvel; cdof_dot[bda..] = cdofdot
```
`mulDofVec(...,3)` computes `tmp = Σ_r cdof_r qvel_r` (skipping zero qvel) before adding to `cvel`.

### 3.2 `mj_objectVelocity(type, id, res, flg_local)` / `mj_objectAcceleration`
Object frame point `pos` and rotation (BODY → `xipos/ximat`; XBODY → `xpos/xmat`; GEOM, SITE, CAMERA). If `body_dofnum[body_weldid[body]] == 0` (static/mocap) → 0. Else `res = transformSpatial(cvel[body], motion, pos, subtree_com[rootid], flg_local ? rot : NULL)`. Acceleration: same with `cacc`, plus Coriolis `res[3:6] += vel_rot × vel_lin` where `vel = transformSpatial(cvel, ...)` in the same frame.

`mj_geomSurfaceVelocity`: `linear = geom_xmat*sv[0:3] + (geom_xmat*sv[3:6]) × (point − geom_xpos)`, `angular = geom_xmat*sv[3:6]`.

### 3.3 `mj_rne(m, d, flg_acc, result)` — inverse dynamics without external/contact forces
```
cacc_loc[0] = [0; -gravity]  (0 if mjDSBL_GRAVITY)          # gravity as fictitious base acceleration
for i = 1.. (awake, body order):
   tmp = mulDofVec(cdof_dot[bda..], qvel[bda..], dofnum)
   cacc_loc[i] = cacc_loc[parent] + tmp                       (mju_add)
   if flg_acc: cacc_loc[i] += mulDofVec(cdof[bda..], qacc[bda..], dofnum)
   f[i] = mulInertVec(cinert[i], cacc_loc[i])
   f[i] += crossForce(cvel[i], mulInertVec(cinert[i], cvel[i]))
f[0] = 0
for i = nbody-1 .. 1: if parent != 0: f[parent] += f[i]
result[i] = dot6(cdof[i], f[dof_bodyid[i]])        for each (awake) dof
```
`qfrc_bias = mj_rne(flg_acc=0)` = Coriolis/centrifugal + gravity. Local stack copies; `d->cacc` is **not** written here.

### 3.4 `mj_tendonBias(m, d, qfrc)`
For each awake tendon with armature `a = tendon_armature + actuatorArmature(TENDON) != 0`: `coef = a * mj_tendonDot(tendon, qvel)`; if nonzero, `qfrc[colind] += coef * ten_J` over the sparse row. (Bias from `d/dt(a J'J q̇) = a J'J q̈ + a J' (J̇ q̇)`.)

### 3.5 `mj_subtreeVel` (lazy; sets `flg_subtreevel = 1`)
```
for each body: v_i = mj_objectVelocity(BODY i, flg_local=0)   ([ω; v_com_i], world orientation)
   subtree_linvel[i] = mass_i * v_i.lin
   subtree_angmom[i] = ximat_i diag(inertia_i) ximat_i' ω_i      (computed as R(I∘(R'ω)))
for i = nbody-1 .. 0: if i: linvel[parent] += linvel[i];  linvel[i] /= max(mjMINVAL, subtreemass[i])
for i = nbody-1 .. 1:
   dx = xipos_i - subtree_com_i; dv = v_i.lin - subtree_linvel_i; angmom_i += dx × (mass_i dv)
   angmom[parent] += angmom_i
   dx = subtree_com_i - subtree_com_parent; dv = (linvel_i - linvel_parent)*subtreemass_i
   angmom[parent] += dx × dv
```
(Note: the linvel backward pass divides each body when it's visited, which happens after all its children are added because children have larger indices.)

### 3.6 Gravity compensation interplay (from `engine_passive.c`, `engine_forward.c`)
`mj_gravcomp` (only if `m->flg_gravcomp`, gravity not disabled, `|g| != 0`): for each body with `body_gravcomp[i] != 0`: `force = −g * mass_i * gravcomp_i` applied at `xipos_i` via `mj_applyFT` into `qfrc_gravcomp` (`J_p' f`, sparse or dense by `mj_isSparse`). In `mj_passive`, `qfrc_passive[dof] += qfrc_gravcomp[dof]` unless `jnt_actgravcomp[dof_jntid]`; for such joints `mj_fwdActuation` adds `qfrc_gravcomp` into `qfrc_actuator` **before** the `jnt_actfrcrange` clamp. Passive stage is skipped entirely (including gravcomp) when both spring and damper are disabled. The gravity itself enters only through `qfrc_bias` via `cacc[0] = −g`; gravcomp cancels it in `qfrc_smooth = qfrc_passive − qfrc_bias + ...`.

---

## 4. Jacobians and dof chains (`engine_core_util.c`)

`mj_isSparse(m)`: `opt.jacobian == SPARSE || (AUTO && nv >= 60)`. Affects `efc_J` format and dense vs sparse paths in many helpers; `M` is always CSR in C.

**Chains** (lists of dof indices, ascending):
* `mj_mergeChain(chain, b1, b2, skipcommon)`: map to weld bodies; if neither has dofs return 0; `da1/da2 = last dof`; loop while either ≥ 0: `da = max(da1, da2)`; if `skipcommon` and both equal `da` → stop; append `da`; advance whichever equals `da` to `dof_parentid`. Reverse to ascending. Returns NV.
* `mj_mergeChainSimple(b1,b2)` (two different simple bodies): swap so `b1 < b2`; concatenate `b1` dofs then `b2` dofs.
* `mj_bodyChain(body)`: simple body → its own dofs; else weld body's ancestor chain ascending.

**`mj_jac(jacp, jacr, point, body)`** (dense 3×nv each, row-major `jac[r*nv + i]`): zero; `offset = point − subtree_com[rootid[body]]` (computed from the ORIGINAL body's root); `body = weldid[body]`; if no dofs return; walk `i = last dof; i = dof_parentid[i]`:
```
jacr[:, i] = cdof[i][0:3]
jacp[:, i] = cdof[i][3:6] + cdof[i][0:3] × offset
```
Wrappers: `mj_jacBody` (point `xpos`), `mj_jacBodyCom` (`xipos`), `mj_jacGeom` (`geom_xpos`), `mj_jacSite` (`site_xpos`). `mj_jacSubtreeCom(body)`: `Σ_{b in subtree} mass_b * jacp(xipos_b, b) / subtreemass[body]` (subtree = contiguous bodies after `body` until `parentid < body`). `mj_jacPointAxis`: `jacAxis[:,i] = jacr[:,i] × axis`.

**`mj_jacSparse(jacp, jacr, point, body, NV, chain, skipcommon)`**: same values placed at the chain positions: walk ancestors `da`, moving `ci` down while `chain[ci] > da`; if not found: skip when `skipcommon` else `mjERROR`. Layout `jac[r*NV + ci]`.

**`mj_jacSparseSimple(jacdifp, jacdifr, point, body, flg_second, NV, start)`**: writes `±` the body's own dofs consecutively starting at column `start` (sign + if `flg_second`).

**`mj_jacDifPair(chain, b1, b2, pos1, pos2, jac1p, jac2p, jacdifp, jac1r, jac2r, jacdifr, issparse, skipcommon)`** → `jacdif = J(b2,pos2) − J(b1,pos1)`:
* `issimple = b1 != b2 && body_simple[b1] && body_simple[b2]`.
* `nv == 0` → 0. Sparse: build chain (simple → `mergeChainSimple`; else `mergeChain`). `NV == 0` → 0. Count-only mode if all outputs NULL.
* Sparse simple: `jacSparseSimple(b1, −, start = b1<b2 ? 0 : dofnum[b2])`, `jacSparseSimple(b2, +, start = b2<b1 ? 0 : dofnum[b1])` (written directly into jacdif; jac1/jac2 untouched).
* Sparse general: `jacSparse` both, then `mju_sub`.
* Dense: `mj_jac` both, `mju_sub` over `3*nv`.

**`mj_jacSum`**: weighted sum of several body Jacobians at the same point; sparse path packs `[jacp; jacr]` and merges via `mju_addToSparseMat`.

**`mj_jacDot(jacp, jacr, point, body)`** (time derivative of the point Jacobian):
```
offset = point - com_root;  pvel = transformSpatial(cvel[body], motion, point, com_root)   # point velocity
for dof i in ancestor chain of weldid[body]:
   cdd = cdof_dot[i]
   if jnt_type == BALL or (FREE and i >= jnt_dofadr+3):  cdd = crossMotion(cvel[dof_bodyid[i]], cdof[i])
        # quaternion dofs: cdof (body-frame axes) rotates with the body's OWN velocity
   jacr[:, i] += cdd[0:3]
   jacp[:, i] += cdd[3:6] + cdd[0:3] × offset + cdof[i][0:3] × pvel[3:6]
```
`mj_jacDotSparse` identical on a chain.

**`mj_angmomMat(mat, body)`**: `Σ_{b in subtree} [ximat diag(I) ximat' jacr_b + mass_b skew(xipos_b − subtree_com[body]) jacp_b]` with `jacp/jacr = mj_jacBodyCom(b)`.

**`mj_contactForce(id, res[6])`**: zero; if valid and `efc_address >= 0`: pyramidal → `mju_decodePyramid(efc_force+adr, friction, dim)`; else copy `dim` values; then `res[0] -= con->adhesion` (net interface force). Contact-frame `[normal, t1, t2, torsion, roll1, roll2]`.

**`mj_tendonSpringDamper(i)`**: spring (unless `mjDSBL_SPRING`; if `stiffness || poly≠0`): `[lower, upper] = plastic state (if tendon_plasticadr ≥ 0) else tendon_lengthspring`; `x = L>upper ? L−upper : L<lower ? L−lower : 0`; `frc_spring = −x * polyForce(stiffness, poly, x, 2, odd=0)`. Damper (unless `mjDSBL_DAMPER`): `damping = tendon_damping + actuatorDamping(TENDON)` (poly also augmented); `frc_damper = −v * polyForce(damping, dpoly, v, 2, odd=1)` with `v = ten_velocity`. `mju_polyForce(lin, poly, x, n, odd) = lin + Σ_k poly[k] * xpow_k`, `x ← |x|` if odd.

`tendonLimit(i)`: count of sides with `side*(range[(side+1)/2] − L) < margin`, side ∈ {−1, +1}.

---

## 5. `mj_rnePostConstraint` — `cacc`, `cfrc_ext`, `cfrc_int`
Called lazily (after the solver, by sensors). Not sleep-filtered (full nbody).
```
cacc[0] = [0; -g] (0 if gravity disabled)
ten_frc[ntendon] = 0; ten_nlimit = 0
cfrc_ext = 0
# perturbations
for i >= 1 with xfrc_applied[i] != 0: cfrc = [xfrc[3:6]; xfrc[0:3]]  (torque:force); addExternal(i, xipos_i, cfrc, +1)
# contacts (efc_address >= 0, non-flex)
for each contact: lfrc = mj_contactForce (contact frame, force:torque)
   cfrc[0:3] = frame' lfrc[3:6] (torque), cfrc[3:6] = frame' lfrc[0:3] (force)
   addExternal(body(geom0), con->pos, cfrc, -1); addExternal(body(geom1), con->pos, cfrc, +1)
# equality rows i in [0, ne), walking by type (efc rows ordered by eq id as instantiated):
   CONNECT/WELD: obj1, obj2 (bodies or sites per eq_objtype); cfrc[3:6] = efc_force[i:i+3]
      CONNECT: cfrc[0:3] = 0
      WELD: q0 = xquat[obj1]*eq_data[6:10] (body) | xquat[sitebody1]*site_quat1 ;  q1 = xquat[obj2] | site quat
            quat = (q1 ⊗ (0, efc_force[i+3:i+6])) ⊗ neg(q0);  cfrc[0:3] = 0.5*eq_data[10]*quat[1:4]   # torquescale
      body1 = obj1 (or site body) if != 0: point = local2Global(offset1) where
            offset1 = body ? eq_data + 3*(type==WELD) : site_pos[obj1];  addExternal(body1, point, cfrc, +1)
      body2 analog with offset2 = body ? eq_data + 3*(type==CONNECT) : site_pos[obj2]; sign -1
      i += (WELD ? 6 : 3)
   JOINT: i += 1
   TENDON: ten_frc[obj1] += f; if obj2 >= 0: dif = L[obj2] - tendon_length0[obj2]
           deriv = d1 + 2 d2 dif + 3 d3 dif² + 4 d4 dif³ (eq_data[1..4]); ten_frc[obj2] -= deriv*f;  i += 1
   FLEX: i += number of non-rigid edges;  FLEXSTRAIN: i += ncells*(order==1 ? 2+3*ngauss : 6*ngauss), ngauss=(order+1)³
   other: mjERROR (and non-equality rows in [0,ne) → mjERROR)
if ntendon:
   for rows r in [ne, ne+nf+nl): FRICTION_TENDON: ten_frc[t] += f
       LIMIT_TENDON: lower = (ten_nlimit[t]++ == 0) && -(range_lo - L) < margin ; ten_frc[t] += lower ? f : -f
   for TENDON actuators: ten_frc[trnid] += actuator_force[out] * gear0
   for each SPATIAL, awake tendon:
       ten_frc += frc_spring + frc_damper            (mj_tendonSpringDamper)
       if armature a: acc = mj_tendonDot(t, qvel) + Σ ten_J*qacc ; ten_frc -= a*acc
       if ten_frc != 0: addTendon(t, ten_frc)
# forward pass
cfrc_int[0] = 0
for j = 1..nbody-1:
   cacc[j] = cacc[parent] + mulDofVec(cdof_dot, qvel) ;  cacc[j] += mulDofVec(cdof, qacc)
   cfrc_body = mulInertVec(cinert, cacc[j]) + crossForce(cvel, mulInertVec(cinert, cvel))
   cfrc_int[j] = cfrc_body - cfrc_ext[j]
for j = nbody-1 .. 1: cfrc_int[parent] += cfrc_int[j]        # includes parent 0 → cfrc_int[0] = Σ roots
flg_rnepost = 1
```
`addExternal(body, point, cfrc, scale)`: no-op for world; `cfrc_com = transformSpatial(cfrc, force, subtree_com[rootid[body]], point)`; `cfrc_ext[body] += scale*cfrc_com`.

`addTendon(t, frc)`: walk wrap objects in lockstep with the stored path `wrap_xpos[ten_wrapadr .. +ten_wrapnum]`: PULLEY → `divisor = wrap_prm`, reset `prevbody=-1`, `p++`. SITE → 1 point (site body). Geom → 2 points if `p < pend && wrap_obj[p] == objid` (tendon wrapped) else 0 points; body = geom body. For each point: if `prevbody >= 0 && body != prevbody`: `u = pnt − prevpnt`; if `normalize3(u) >= mjMINVAL`: `f = u*frc/divisor`, `addExternal(body, pnt, [0; f], +1)`, `addExternal(prevbody, prevpnt, [0; f], −1)`. Then `prevbody = body, prevpnt = pnt, p++`.

Note: fixed tendons contribute nothing to `cfrc_ext` (they act through joints and appear in `cfrc_int`).

---

## 6. Programmatic optimizations (C)

* **Stack arena**: all temporaries via `mj_markStack / mjSTACKALLOC / mj_freeStack` on `d->arena`; lazily allocated in `mj_transmission` (BODY/site buffers only when first needed).
* **Tree-sparse inertia** (CSR in elimination-tree order, `dof_parentid`): CRB fill is O(Σ depth), LDL has **no fill-in** (row k's ancestors ⊆ pattern), solve O(nC).
* **Simple dofs**: `dof_simplenum` collapses rows to diagonal and uses precomputed `dof_M0`; `mj_jacSparseSimple` / `mj_mergeChainSimple` fast paths; `body_simple == 2` (slider-only) also used by flex mapping.
* **sameframe** flags avoid recomputation in `mj_local2Global`.
* **Early outs**: `rotVecQuat` identity/zero, `quat2Mat` identity, `axisAngle2Quat` zero angle, `normalize4` skip if unit, `mulMatTVec` skipping zero coefficients, `solveLD` skipping zero `x_i` in the transposed pass and diagonal-only rows.
* **AVX** (`mjUSEAVX`): `mju_dot`, `mju_dotSparse`, `scl/add/sub/addTo/addToScl` vectorized; scalar code mimics the AVX accumulation order exactly (bit-identical).
* **Sleep filtering** (`mjENBL_SLEEP`): `body_awake_ind`, `parent_awake_ind`, `dof_awake_ind`, `nbody_awake`, `nparent_awake`, `nv_awake` index lists restrict loops; static bodies skipped; asleep bodies' frames compared bitwise to detect motion.
* **Lazy evaluation flags**: `flg_subtreevel`, `flg_rnepost`, `flg_energypos/vel`.
* **Precompute at compile time**: `body_subtreemass`, `dof_M0`, `ten_J` pattern, M/D/B patterns and maps, `body_simple`, `dof_simplenum`, `qpos0`, `cam_mat0/pos0/poscom0`, `light_dir0/pos0/poscom0`, `tendon_length0`.

---

## 7. GPU (MuJoCo Warp) implementation

General: every Data array has a leading `nworld` axis (`[nworld, ...]`); Model arrays that may be batched per-world are 2D with leading size 1 or nworld and indexed `arr[worldid % arr.shape[0], ...]`. Precision `float32`. Spatial vectors are `wp.spatial_vector` (`spatial_top` = rotation part [0:3], `spatial_bottom` = translation [3:6]); `cinert/crb` are `vec10`; matrices `wp.mat33`; quaternions `wp.quat` with MuJoCo's `[w,x,y,z]` order stored in the 4 components (not Warp's native xyzw semantics; all quaternion math is done by `math.py` helpers). No sleep filtering inside these kernels (Warp has separate `sleep.py` kernels for waking, but `smooth.py` kernels compute every body/dof).

### 7.1 Precomputed traversal structures (`io.py`)
* `body_branches`, `body_branch_start`, `nbranch`: one branch per **leaf body**: the full ancestor chain `[root, ..., leaf]` (world excluded). Kernels parallelized over `(nworld, nbranch)` walk a branch sequentially from its root; shared ancestors are **recomputed redundantly** by every branch through them and written with identical values (benign races).
* `body_subtreenum[b]`: subtree size including b; subtree = contiguous body range `[b, b + subtreenum)`. Used for "gather" reductions instead of backward accumulation.
* `body_tree`: bodies grouped by depth (used elsewhere).
* `body_isdofancestor[nbody, nv_pad]` (int): 1 if dof affects body (dof on weld-body's ancestor chain). Used by `support.jac_dof` (O(1) test per (body,dof)).
* Tendon: `tendon_jnt_adr/wrap_jnt_adr` (one element per fixed-tendon joint), `wrap_site_pair_adr` (index of first site of each consecutive site–site pair, excluding pairs straddling tendons), `tendon_site_pair_adr`, `wrap_geom_adr` (wrap index of each sphere/cylinder), `tendon_geom_adr`, `wrap_pulley_scale[nwrap]` (= `1/wrap_prm[pulley]` for all wrap entries after a pulley until the tendon end, else 1), `max_ten_J_rownnz`.
* `actuator_trntype_body_adr`, `nacttrnbody`.
* `M` layout: identical CSR (`M_rownnz/rowadr/colind`, nC) to C. `M_mulm_rowadr/col/madr`: gather-CSR for `mul_m` (row i gathers lower entries of row i and the transposed entries where i is the column) — one thread per dof, no atomics.
* **Block layout for the factor** (`m_block_layout`): M's diagonal blocks are the kinematic trees (`tree_dofadr/tree_dofnum`). For each tree of `size` dofs with nnz = sum of its rows:
  * `size <= M_BLOCK_SCALAR_MAX (6)` and (`compact`: nnz == size, i.e. diagonal-only, or `triangular`: nnz == size(size+1)/2, i.e. dense lower triangle): **scalar path**. Compact → `qLD_block_adr = Q_LD_BLOCK_COMPACT (−2)` (no factor storage, only `qLDiagInv`); triangular → dense `size×size` factor slot in the packed region.
  * else `size <= M_BLOCK_DENSE_MAX (64)`: **tile Cholesky** ("gather tiles"), dense slot `size²`.
  * else: **sparse LDL**, `qLD_block_adr = Q_LD_BLOCK_SPARSE (−1)`.
  * `qLD` (Data) = `[packed dense block factors (qLD_block_total) | nC LDL region (only if some block is sparse)]`. Dense block factors store the **upper Cholesky factor U** (`M_blk = Uᵀ U`), row-major `U[i*size + j]`. `qLDiagInv[nv]` is only meaningful for compact and sparse dofs.
  * Tiled blocks have `tile.elemid` = per-slot CSR gather index (`M[max(i,j), min(i,j)]` address, or `nC` for structurally absent → reads 0) for `wp.tile_load_indexed`.
  * `qLD_updates`: tuple (one array per elimination level) of `(i, k, Madr_ki)` triples for **sparse-block** dofs only: for each non-diagonal row k, for each ancestor i (walking `dof_parentid`) with `Madr_ki` = address of M(k,i); grouped by `dof_depth[i]` (depth of the ancestor). `qLD_all_updates`/`qLD_level_offsets`: flattened for the fused solve kernel.

### 7.2 Kinematics (`smooth.kinematics`)
Kernels and launch dims:
1. `_kinematics_branch` — `(nworld, nbranch)`: per branch, sequential over bodies; writes `xpos, xquat, xanchor, xaxis`.
2. `_compute_body_matrices` — `(nworld, nbody)`: `xmat = quat_to_mat(xquat)`.
3. `_compute_body_inertial_frames` — `(nworld, nbody)`: `xipos = xpos + rot_vec_quat(body_ipos, xquat)`, `ximat = quat_to_mat(mul_quat(xquat, body_iquat))`.
4. `_geom_local_to_global` — `(nworld, ngeom)`: **skips** geoms with `body_weldid[body]==0 && body_mocapid[body_rootid[body]] == -1` (static: computed once in `make_data`/`put_data`); else `xpos + rot_vec_quat(geom_pos, xquat)`, `quat_to_mat(mul_quat(xquat, geom_quat))`.
5. `_site_local_to_global` — `(nworld, nsite)`: same for sites (all sites).

Differences vs C:
* Positions use `rot_vec_quat(pos, xquat_parent)` instead of `xmat_parent * pos` (C uses matrix for body positions, geoms, sites, xipos). No `sameframe` shortcuts (geom/site/inertial frames always recomputed via quaternion product + `quat_to_mat`).
* `math.rot_vec_quat(v, q) = 2(u·v)u + (s² − u·u)v + 2s(u×v)` (u = q_xyz, s = q_w) — algebraically equal to C's formula for unit q, different rounding; no early-outs.
* Mocap quaternion is **not normalized** (C normalizes a copy). Parent transform is always applied (`pid >= 0` always true; world has identity so it is exact).
* Free joint: `xquat = wp.normalize(q)`; `wp.normalize` returns 0 for zero-length input (C returns (1,0,0,0)) and always rescales (C skips when |n−1| ≤ 1e-15).
* Hinge `axis_angle_to_quat` has no θ==0 branch (gives exact identity anyway).
* Kinematics never wakes sleeping trees inside this kernel; no bitwise-equality sleep check.

### 7.3 `com_pos`
* `_subtree_com` — `(nworld, nbody)`, skips body 0: `com = Σ_{j∈[b, b+subtreenum)} xipos_j*mass_j`; `mass = body_subtreemass[b]` (model constant); `< MJ_MINVAL` → `xipos_b`; else `com /= mass` (true division; C multiplies by reciprocal).
* `_subtree_com_world` — `(nworld)`: world = `(xipos_0*mass_0 + Σ_roots subtree_com_r*subtreemass_r)/subtreemass_0` iterating roots via `rootid += subtreenum[rootid]`.
* `_cinert` — `(nworld, nbody)`: same formula as `mju_inertCom` (rows `mat[r]∘inert` dotted with `mat[c]`, same products as C).
* `_cdof` — `(nworld, njnt)`: per joint, writes 1/3/6 dofs; ball/free rotational dofs use rows of `transpose(xmat)` (= columns of xmat); identical math to C (`cross(axis, offset)`).
Summation order of subtree sums differs from C (C accumulates children into parents bottom-up).

### 7.4 `camlight`
`_cam_local_to_global` / `_light_local_to_global` — `(nworld, ncam)` / `(nworld, nlight)`. Target modes with invalid target fall back to fixed computation. Uses `wp.normalize` (returns 0 on zero vector vs C (1,0,0)). Same formulas otherwise.

### 7.5 `flex` (kinematic part)
`_flex_nodes` `(nworld, nflexnode)`, `_flex_vertices` `(nworld, nflexvert)` (each thread searches its flex by linear scan over `nflex`; interpolated vertices use **trilinear only** `eval_basis_trilinear` over the 8 cell nodes with local coords clamped to `[0,1]` (or `[−0.02, 1.02]` when the precomputed cell differs from the natural cell), non-interpolated: centered → `xpos`, else `xmat*vert + xpos` (no exact-zero shortcut)), `_flex_edges` `(nworld, nflexedge)`: length + Jacobian on the model's `flexedge_J` pattern using `body_isdofancestor` and **also `flexedge_velocity`** (C computes velocity in `mj_fwdVelocity`); edges with `rownnz == 0` get zero velocity. `_flex_face_kinematics` for interpolated faces (`face_xpos`, `face_quat` via polar decomposition). No AABB/BVH update in this function (collision side). No rigid/edge-property skipping beyond `rownnz`.

### 7.6 `crb`, `tendon_armature`
* `_crb_subtree` `(nworld, nbody)`: `crb_b = cinert_b + Σ_{j∈(b, b+subtreenum)} cinert_j`; world: `crb_0 = cinert_0` (zero). Gather sum order differs from C.
* `d.M.zero_()`, `_M` `(nworld, nv)`: per dof i: `buf = inert_vec(crb[body(i)], cdof_i)`; diagonal `= dof_armature_total + dot(cdof_i, buf)`; then walk ancestors writing `M[madr--] = dot(cdof_j, buf)` while `madr >= rowadr` (rows of simple dofs stop after the diagonal). **Simple dofs are recomputed from the current state instead of using `dof_M0`** (equal in exact arithmetic for simple bodies; differs in rounding). `wp.dot` of spatial vectors is a plain sequential sum (C uses `mji_dot6` pairing). Armature: `util_misc.dof_armature_total` = `dof_armature + actuator_target_armature(JOINT)` (`gear0²` scaling, single or scan mode like C).
* `_tendon_armature` `(nworld, ntendon, max_ten_J_rownnz)`: thread per (tendon, J entry k): dof i = colind[k], `J_i`; walks i's ancestor chain (including i), finds `J_j` by scanning the tendon row backward (0 if absent), `atomic_add(M[madr], a*J_j*J_i)`. Same restriction to the M pattern as C. Atomics → nondeterministic order unless `opt.deterministic & ATOMICS` (Warp `DeterministicMode.RUN_TO_RUN`).

### 7.7 `factor_m`, `solve_m`, `factor_solve_i`
`factor_m`: `_factor_blocks` for every `TileSet` in `m.M_tiles`, then `_factor_i_sparse` on `qLD[:, qLD_block_total:]` if present.
* **Scalar blocks** `_small_cholesky_factorize_block(size)` — `(nworld, nblocks)`, one thread per block: compact → `qLDiagInv = 1/M_ii`; triangular → in-register **Cholesky** `U`: `U_ii = sqrt(M_ii − Σ_{k<i} U_ki²)`, `U_ij = (M_ji − Σ_k U_ki U_kj)/U_ii` (multiply by `1/U_ii`). M read directly from CSR assuming the block is a dense lower triangle: `M[matrix_adr + j(j+1)/2 + i]`.
* **Tile blocks** `_tile_cholesky_factorize_block(tile)` — `wp.launch_tiled`, `(nworld, nblocks)`, `block_dim = m.block_dim.cholesky_factorize`: gather the dense symmetric block via `tile_load_indexed`, `wp.tile_cholesky_inplace(L, fill_mode="upper")`, store.
* **Sparse LDL** `_factor_i_sparse`: `L = copy(M)`; for each level in **reversed** order (deepest ancestor depth first), `_qLD_acc` `(nworld, len(level))`: for update `(i, k, Madr_ki)`: `tmp = L[Madr_ki] / L[diag_k]` (division, not reciprocal multiply); `for j < rownnz[i]: atomic_sub(L[rowadr_i + j], L[rowadr_k + j]*tmp)` (prefix property as in C); then `L[Madr_ki] = tmp`. Deterministic mode: reads from a snapshot `L_in` and writes `L[Madr_ki]` in a separate `_qLD_scale` kernel. Finally `_qLDiag_div` `(nworld, nv)`: `qLDiagInv = 1/L[diag]` (all dofs; overwritten values for block dofs are irrelevant).
* **No pivot clamping** (C clamps pivots < 1e-15 and warns): non-PD blocks produce NaN (`sqrt` of negative) or inf.

`solve_m(x, y)` → `solve_LD`: `_solve_blocks` then `_solve_LD_sparse` on the LDL region.
* Scalar blocks `_small_cholesky_solve_block`: compact `x = qLDiagInv*y`; dense: forward `Uᵀ z = y` then backward `U x = z` with divisions by `U_ii`.
* Tile blocks `_tile_cholesky_solve_block`: `wp.tile_cholesky_solve(U, rhs, fill_mode="upper")`; `block_dim = m.block_dim.cholesky_solve` if size ≤ 40 else 32.
* Sparse: non-deterministic default `_solve_LD_sparse_fused(nv, nlevels)` — `(nworld, block_dim)` launched with one CUDA block per world (`block_dim = m.block_dim.solve_LD_sparse_fused`; CPU fallback 1 thread): copy y→x for sparse dofs; forward levels from deepest: `atomic_sub(x[i], L[Madr_ki]*x[k])`; `x *= qLDiagInv` (sparse dofs); backward levels shallow→deep: `atomic_sub(x[k], L[Madr_ki]*x[i])`; `WP_TILE_SYNC()` between levels. Deterministic mode: one kernel launch per level with input snapshot `x_in`.
  The level grouping means the order of subtractions into a given `x[i]` differs from C (C processes rows in index order, scalar sequential).
* `factor_solve_i(M, L, D, x, y)`: fused block factor+solve (`_small_cholesky_factorize_solve_block`, `_tile_cholesky_factorize_solve_block`) + sparse factor + sparse solve (used by implicit integrators / effective metric).
* `factor_solve_lu` (`_factor_solve_lu_sparse_fused`): single-thread-per-world sparse LU on the D pattern for non-symmetric implicit systems (in-place, uses `qacc` as scratch counters).
* `skip_qacc_smooth`: when `!opt.run_qacc_smooth && solver == NEWTON && warmstart enabled && noslip_iterations == 0 && iterations > 0 && njmax > 0`, Warp skips computing `qacc_smooth` (and its factorization path) — a GPU-only option.
* Dense vs sparse choice: **M is always CSR in Warp** (no dense M); `m.is_sparse` controls `efc_J`/constraint layouts: `AUTO → nv > 32` (C: `nv >= 60`); explicit DENSE/SPARSE follow `mj_isSparse`. `mul_m` uses the gather-CSR (`M_mulm_*`) or a dense `(nworld, nv, nv)` variant for the compact solver.

### 7.8 `rne`
* `_rne_cacc_world` (`_cacc_world` kernel `(nworld)`: `cacc[0] = [0; −gravity]`; or `cacc.zero_()` if gravity disabled). **Writes `d.cacc`** (C uses local scratch in `mj_rne`; `d.cacc` is only defined after `rnePostConstraint`).
* `_cacc_branch` `(nworld, nbranch)`: `local = cacc[parent(first body)]`; for each body along the branch and each of its dofs j **in sequence**: `local += cdof_dot_j*qvel_j; if flg_acc: local += cdof_j*qacc_j` (interleaved per dof; C sums `Σ cdof_dot qvel` into a temp, adds to parent, then adds `Σ cdof qacc`).
* `_cfrc` `(nworld, nbody)`: `frc = inert_vec(cinert, cacc) + motion_cross_force(cvel, inert_vec(cinert, cvel))` (minus `cfrc_ext` when `flg_cfrc_ext`); body 0 → 0. `math.motion_cross_force(v, f) = [v_ang×f_ang + v_lin×f_lin ; v_ang×f_lin]` (same as `mju_crossForce`, built from `wp.cross`).
* `_cfrc_subtree` `(nworld, nbody)`: `cfrc_int_b = Σ_{subtree} cfrc_body` (gather); `_cfrc_world`: `cfrc_int_0 = Σ_roots cfrc_int_r`. **Writes `d.cfrc_int`** as scratch during `rne` (so after `fwd_velocity`, `d.cfrc_int` holds the bias-only internal forces until `rne_postconstraint` overwrites it).
* `_qfrc_bias` `(nworld, nv)`: `qfrc_bias = dot(cdof_i, cfrc_int[body(i)])`.

### 7.9 `tendon_bias`
* `_tendon_dot` `(nworld, ntendon)`: skip if armature 0 or fixed; site-only segments; geom wrap → **return** (abandons the rest of the tendon; C errors). Computes per-entry `ten_Jdot[nJten]` via `_accumulate_jac_dot_chain` walking `body_parentid` from each segment body to world, dofs descending, matching against `ten_J_colind` with a backward pointer; per dof: quaternion override (ball/free-rotational → `motion_cross(cvel[body], cdof)`), `jacp_dot = cdd_lin + cdd_ang×offset + cdof_ang×pvel_lin`, `jacp = cdof_lin + cdof_ang×offset`, `Jdot += (jacp_dot·dpnt + jacp·dvel) * (±1/divisor)` (`inv_divisor = safe_div(1, divisor)`). `pvel_lin = cvel_lin − offset×cvel_ang`. `dvel /= norm` if `norm > MJ_MINVAL` else 0. Non-atomic `+=` (each tendon owned by one thread).
* `_tendon_bias_coef` `(nworld, ntendon, max_ten_J_rownnz)`: `coef[t] += Jdot*qvel` (atomic).
* `_tendon_bias_qfrc` same dims: `qfrc[col] += J * a * coef` (atomic).
Note C computes `coef = a * (Jdot·qvel)` then `qfrc += coef*J`; Warp computes `J * a * coef` — different association.

### 7.10 `com_vel`
`_comvel_root` zeroes `cvel[0]`; `_comvel_branch` `(nworld, nbranch)`: as C, but `cvel += cdof_j*qvel_j` one dof at a time (C groups the 3 translational / 3 rotational dofs through `mulMatTVec` into a temp first, which also skips zero `qvel`). Joints iterated by joint (free: 3 zero `cdof_dot`, then rotational `cdof_dot` with post-translation `cvel`; ball; hinge/slide). Bodies without joints copy parent velocity.

### 7.11 `subtree_vel`
Different algorithm (same math): `_subtree_vel_forward` `(nworld, nbody)`: per body momentum `[ximat (I∘(ximatᵀ ω)) ; mass*(v_lin − (xipos − com_root)×ω)]`; `_subtree_momentum` `(nworld, nbody)`, body ≥ 1: `linmom = Σ_subtree p_j`, `angmom = Σ_subtree [L_j + (xipos_j − subtree_com_b)×p_j]` (the CoM-velocity term vanishes since `Σ m_j (x_j − c) = 0`), `linvel = linmom / max(MJ_MINVAL, subtreemass)`; `_subtree_momentum_world`: world from roots (undoing the root division, then shifting angular momentum: `+ angmom_r + (com_r − com_0)×((linvel_r − linvel_0)·subtreemass_r)`). Not lazy (no `flg_subtreevel`). Differences: O(nbody·depth) gathers; rounding differs from the C recursion, and differs in exact arithmetic only by the vanishing term.

### 7.12 `transmission`
`_transmission` `(nworld, nactuator)`: **rowadr allocated with `wp.atomic_add(moment_nnz[world], nnz)`** — row storage order in `actuator_moment` is thread-schedule dependent (not actuator order) unless deterministic atomics; consumers use `rowadr` so results are equal, but raw arrays differ from C.
* JOINT/JOINTINPARENT: as C (`quat_to_vel` for ball; `quat_inv` = conjugate).
* SLIDERCRANK: dof set = merged chain of `weldid(site body)` and `weldid(slider body)` (including common dofs), entries written for every chain dof **without compressing exact zeros** (C compresses), `safe_div(av, sdet)` and `safe_div(vec, sdet)`; Jacobian columns from `support.jac_dof` (O(1) per dof via `body_isdofancestor`); `jacA = jacr × axis`.
* TENDON: copy of `ten_J` row × gear0.
* SITE no refsite: entries for every ancestor dof of the weld body, no zero compression, `moment = jacp·wrench_t + jacr·wrench_r`.
* SITE refsite: merged chain **excluding common ancestors** (`break` when `da1 == da2`), `length` = trans + rot parts as C; moment = `(jacp−jacpref)·wrench_t + (jacr−jacrref)·wrench_r` (only parts with nonzero gear). Matches C except zero compression.
* BODY: row of full length `nv` (colind 0..nv−1), zeros; then `_transmission_body_moment` `(nacttrnbody, naconmax, nv)` — per (body actuator, contact, dof): `contact_exclude = dist >= includemargin` (C uses `con->exclude == 1`, which is also false for adhesion/speculative contacts — potential mismatch); counts contacts with `atomic_add` (counts every relevant contact, including excluded-in-gap, but there is no `efc_address < 0` skip); included contacts: atomic add of `efc_J` rows (normal row if `dim == 1` or elliptic, else `0.5/(dim−1)` × each of the `2(dim−1)` pyramid rows), sparse or dense `efc_J`; excluded: normal·(jacp2 − jacp1) at the contact point. `_transmission_body_moment_scale`: `moment /= −ncon`.
* **SO3 transmission not implemented** (`TrnType` lacks SO3; prints "unhandled transmission type").
* No sleep filtering.

### 7.13 `tendon`
`d.ten_length.zero_(); d.ten_J.zero_()`.
* `_joint_tendon` `(nworld, nfixedwrap)`: `atomic_add(L[t], prm*qpos[qposadr])`; `ten_J[first k with colind==dofadr] = prm` (**assignment**: repeated joints overwrite instead of accumulate).
* `_spatial_site_tendon` `(nworld, n_site_pairs)`: per consecutive site pair: `vec, length = normalize_with_norm(p1−p0)`; `atomic_add(L, length*pulley_scale)`; if `length < MJ_MINVAL`: `vec = (1,0,0)`; if bodies differ: `_accumulate_jac_chain(body0, −scale)`, `(body1, +scale)`: walks body chain to world, per dof `J += (cdof_lin + cdof_ang×offset)·vec * scale` via `atomic_add` (skip if exactly 0).
* `_spatial_geom_tendon` `(nworld, n_geom_wraps)`: site–geom–site: `util_misc.wrap(...)` (side sentinel `vec3(MJ_MAXVAL)` = "no sidesite"); stores tangent points in scratch `wrap_geom_xpos` (sentinel `MJ_MAXVAL` when no wrap); wrapped: length = `|g0−s0| + wlen + |s1−g1|` (added only if nonzero), Jacobians on the two straight segments; not wrapped: straight site–site segment.
* `_spatial_tendon_wrap` `(nworld)`: single thread per world rebuilds `ten_wrapadr/ten_wrapnum/wrap_obj/wrap_xpos` sequentially over all tendons (same logic as C). **Layout**: `wrap_xpos` is `[nworld, nwrap]` of `spatial_vector` holding **two points per entry** (point p at row `p//2`, top/bottom = `p%2`); `wrap_obj` is `[nworld, nwrap]` of `vec2i` (same pairing). Values −1 (site), geom id (tangent points), −2 (pulley). Wrap failure detected by `|tangent0| < MJ_MAXVAL`.
* Pulley division replaced by multiplication with precomputed `1/divisor` (`wrap_pulley_scale`).
* No tendon sleep handling (separate `sleep.wake_tendon` in `fwd_kinematics`).

`util_misc.wrap/wrap_circle/wrap_inside/is_intersect/length_circle`: line-by-line ports of the C functions with these differences: sentinel vectors (`MJ_MAXVAL`) instead of NULL pointers; `math.safe_div` (`x / (y != 0 ? y : MJ_MINVAL)`) in tangent formulas, `A=r/len0`, `B`, `cosG`, Newton step, cylinder height ratios; `normalize_with_norm` returns the input unchanged when norm == 0 (C returns (1,0,0)) — explicitly patched for the sidesite projection (`sidenrm < MJ_MINVAL → (1,0)`); `wrap_inside` initializes `z = 1.0` and `asin(A), asin(B)` directly (C computes `sin(π/2)`, which is exactly 1 in double); geom type check returns `MJ_MAXVAL` (C errors). `wrap_inside` on `cosG < −1+MJ_MINVAL` returns `(−1, pnt, pnt)`.

### 7.14 `rne_postconstraint`
1. `_cfrc_ext` `(nworld, nbody)`: `cfrc_ext = transform_force(xfrc_applied, subtree_com_root − xipos)` (for all bodies, also zero ones; world 0).
2. If `neq > 0`: `_count_equality_constraints` `(nworld, njmax)` counts connect rows and weld rows (atomics); `_cfrc_ext_equality` `(nworld, neq)`: assumes the equality efc rows are ordered **all connect rows first (3 each), then all weld rows (6 each)** — this is Warp's constraint ordering (`make_constraint` launches connect, weld, joint, tendon kernels in that order), which differs from C's eq-id ordering. Weld torque `0.5*torquescale*vec(mul_quat(quat_mul_axis(q1, f_rot), quat_inv(q0)))`. Points via `xmat @ offset + xpos`. `atomic_add` body1, `atomic_sub` body2.
3. `_cfrc_ext_contact` `(naconmax)`: skip if both bodies are world; `support.contact_force_fn(to_world_frame=True)` (includes `−adhesion`); `atomic_sub` on body1, `atomic_add` on body2 (no flex-contact check: assumes geom ids ≥ 0).
4. If `ntendon > 0`: `ten_frc` from `_cfrc_ext_tendon_constraint` `(nworld, njmax)` — equality tendon rows (same polynomial derivative), friction rows, limit rows with sign `+1 if (L − lo) < (hi − L) else −1` (**differs from C** when both limits are active: C assigns the first row to the lower side, the second to the upper); `_cfrc_ext_tendon_actuator` `(nworld, nu)`; `_tendon_dot` + `_tendon_bias_coef` for armature acceleration; `_cfrc_ext_spatial_tendon` `(nworld, ntendon)`: spring/damper (`_poly_force`), `− a*(Jdot·qvel + J·qacc)`, path walk like `addTendon` but skipping segments with `norm > 0.0` false (C: `< mjMINVAL`), torques built directly (`−(com − pnt)×f`), `atomic_add/sub`. Spatial tendon sleep not considered.
5. `_rne_cacc_world`, `_rne_cacc_forward(flg_acc=True)`, `_rne_cfrc(flg_cfrc_ext=True)` → `cacc`, `cfrc_int` (gather sums; `cfrc_int[0] = Σ roots`, matches C).

### 7.15 `math.py` (Warp) formulas vs C
| Warp | C equivalent | Notes |
|---|---|---|
| `mul_quat` | `mju_mulQuat` | identical expression |
| `quat_mul_axis` | `mju_mulQuatAxis` | identical |
| `rot_vec_quat` | `mju_rotVecQuat` | different (but equivalent for unit q) formula; no shortcuts |
| `axis_angle_to_quat` | `mju_axisAngle2Quat` | no θ==0 branch |
| `quat_to_mat` | `mju_quat2Mat` | identical formula, no identity shortcut |
| `quat_inv` | `mju_negQuat` | conjugate |
| `inert_vec` | `mju_mulInertVec` | identical |
| `motion_cross`, `motion_cross_force` | `mju_crossMotion/Force` | built from `wp.cross`; different add grouping |
| `quat_to_vel` | `mju_quat2Vel(dt=1)` | returns 0 if `|q_xyz| == 0`; result `axis*speed/sin_a_2` (divides instead of normalize-then-scale) |
| `quat_sub` | `mju_subQuat` | |
| `quat_integrate(q, v, dt)` | `mju_quatIntegrate` | `wp.normalize(v)` (0 for zero v), normalizes q, multiplies, **normalizes result** (C does not normalize after product) |
| `deriv_expmap` | `mju_derivExpmap` | uses the **single-precision** Bortz coefficients |
| `quat_z2vec` | `mju_quatZ2Vec` | axis `(−v1, v0, 0)` = z×v, `angle = atan2(|axis|, v2)`. Literal fallbacks differ from C: tiny-vector return is `wp.quat(0,0,0,1)` (index 0 = w = 0, i.e. 180° about z in MuJoCo order; C returns identity) and the anti-parallel case returns `wp.quat(1,0,0,0)` (identity; C returns (0,1,0,0)). Not used by the smooth pipeline; reproduce literally if parity on these edge cases matters |
| `make_frame`, `orthogonals` | `mju_makeFrame` | y-axis choice `(0,1,0)` if `−0.5 < a1 < 0.5` else `(0,0,1)` |
| `normalize_with_norm` | `mju_normalize*` | zero-norm → returns input unchanged, norm 0; threshold `== 0` not `< mjMINVAL` |
| `safe_div(x,y)` | — | `x / where(y != 0, y, MJ_MINVAL)` |
| `upper_tri_index`, `upper_trid_index` | — | packed upper-triangle index helpers |
| `free_bias_vel_blocks`, `lu_factor_6x6`, `lu_solve_6x6` | derivative helpers | out of scope (derivatives) |

### 7.16 Determinism and fixed buffers
* Atomics in: `tendon_armature`, `factor_i_sparse`, sparse solve (fused kernel), `tendon` (length and J), `tendon_bias`, `transmission` (rowadr allocation, BODY moments), `rne_postconstraint` (cfrc_ext). With `opt.deterministic & DeterminismType.ATOMICS`, kernels are compiled with `wp.DeterministicMode.RUN_TO_RUN` (`deterministic_max_records` sized per kernel) and the sparse LDL/solve switch to snapshot-per-level launches.
* Fixed sizes: `naconmax` (contacts), `njmax` (efc rows), `nv_pad`, `max_ten_J_rownnz`, `nmaxpyramid`. `moment_colind/actuator_moment` capacity must accommodate dense BODY rows (nv each) and non-compressed site/slidercrank rows.

### 7.17 MJX (JAX) differences (brief)
* `kinematics`: `scan.body_tree`; **writes the normalized ball/free quaternions back into `qpos`**; free joint `xaxis = (0,0,1)`; mocap bodies overwrite `xpos/xquat/xmat` after the scan (mocap quat normalized). `math.normalize` uses `x / (n + 1e-6*(n==0))`, `math.norm` is gradient-safe (zero → 0). Positions via `math.rotate` (same formula as Warp's `rot_vec_quat`). No sameframe shortcuts, no static-geom caching.
* `com_pos`: subtree sums via reverse body scan; `subtree_com = pos / max(mass, mjMINVAL)` then `where(mass < mjMINVAL, xipos, ...)`; `cinert` via `ximat*inert @ ximat.T + h h.T * mass` with `h = cross(off, −I)`.
* `crb`: reverse scan; `crb[0] = 0`; `support.make_m(m, crb·cdof, cdof, dof_armature)` builds dense `nv×nv` (if not sparse) or the legacy sparse layout. `is_sparse`: `AUTO → nv >= 60 or TPU backend`.
* `factor_m`: dense → `jax.scipy.linalg.cho_factor` (Cholesky, not LDL; `qLD` holds the Cholesky factor); sparse → level-batched LDL emulating C (updates grouped by depth, `qld[out] += −(qld[madr]/qld[pivot])*qld[rows]`, then divide off-diagonals by their row pivot; diagonal kept; `qLDiagInv = 1/diag`). No pivot clamping. `solve_m`: `cho_solve` or level-batched sparse back-substitution.
* `transmission`: JOINT, JOINTINPARENT, SITE (with refsite), TENDON only (no SLIDERCRANK, BODY, SO3); moments dense `nu × nv`. Ball length via `quat_to_axis_angle`.
* `tendon`: supports fixed, site, sphere/cylinder (with static precomputed `is_wrap_inside` mask per wrap), pulleys. Flex and plasticity not implemented in the JAX backend.
* `rne`, `rne_postconstraint`, `com_vel`, `subtree_vel`, `tendon_armature`, `tendon_dot`, `tendon_bias` present. When `impl == WARP`, MJX forwards these calls to MuJoCo Warp kernels.

---

## 8. Reproduction pitfalls / parity notes

1. **Matrix vs quaternion rotation of offsets.** C rotates body positions (in `kinematics1`), xipos, geom and site positions with the parent/body **matrix** (`xmat*pos`), and joint anchors/axes with the **quaternion** (`rotVecQuat`). Warp/MJX use quaternion rotation everywhere. Bit parity with C requires following C exactly.
2. **Normalization semantics.** `mju_normalize4` leaves already-unit quaternions untouched (|n−1| ≤ 1e-15) and returns (1,0,0,0) for tiny input; `normalize3` returns (1,0,0); both multiply by `1/n`. Warp `wp.normalize`/`normalize_with_norm` return 0 / input for zero vectors and always rescale; MJX adds `1e-6` to zero norms. Mocap quaternions are normalized in C (not Warp). `qpos` quaternions are never normalized in place by C kinematics (MJX writes them back).
3. **Slide/hinge use `qpos − qpos0`** (joint `ref`), but fixed tendons and joint transmissions use raw `qpos` (no `qpos0` subtraction): `ten_length = Σ coef*qpos`, `actuator_length = qpos*gear0`.
4. **Reference point of c-quantities is per tree** (`subtree_com[body_rootid]`), not per body; Jacobian offsets use the root of the original (non-weld) body.
5. **Ball/free rotational `cdof` uses the body frame after all its joints** and `cdof_dot` for these dofs uses the *parent-side* velocity (before the joint's own contribution) in `mj_comVel`, but `mj_jacDot` overrides it with `cvel[dof_bodyid] × cdof`. Keep both conventions.
6. **Summation order** (affects last bits): `mulDofVec` (temp sum with zero-skipping, then add) vs per-dof accumulation (Warp); `mji_dot6` pairing; `mju_dot` 4-way accumulation; subtree reductions bottom-up (C) vs gather (Warp); tendon Jacobian `û'(J2−J1)` (C) vs `û'J2 − û'J1` via atomics (Warp); `1/divisor` division (C) vs multiplication by precomputed reciprocal (Warp).
7. **Simple dofs** take their diagonal inertia from `dof_M0` (computed once at qpos0 with `mju_dot`), not recomputed. Warp recomputes. `M` never contains off-diagonals for simple dofs.
8. **Tendon armature** only enters M at the existing sparsity pattern (ancestor pairs); cross-branch couplings of `a J'J` are silently dropped in both C and Warp. `mj_tendonBias` adds `a J'(J̇q̇)`; `rnePostConstraint` subtracts `a(J̇q̇ + Jq̈)` from the tendon force.
9. **Factorization**: C clamps pivots below `mjMINVAL` (and warns `mjWARN_INERTIA` with the first clamped dof); Warp and MJX do not. C multiplies by `invD`; Warp divides. Warp uses Cholesky (`UᵀU`) for trees ≤ 64 dofs and LDL only for larger trees; the stored `qLD` contents are therefore not comparable to C's except in the pure-sparse case. MJX dense uses Cholesky as well.
10. **Sparse LDL prefix trick**: the update of row i by row k uses the first `rownnz[i]` entries of row k — requires the ancestor-ordered CSR layout exactly as built by `mj_makeDofDofSparse` (ascending ancestors, diagonal last).
11. **`mj_isSparse` threshold**: C `nv >= 60`, Warp `nv > 32`, MJX `nv >= 60` (or TPU). Affects which code path (and hence rounding) is used for `efc_J`, `mj_tendonDot`, `mj_applyFT`, transmission BODY.
12. **Transmission sparsity**: C drops exact-zero moment entries for SLIDERCRANK, SITE, SO3 (site variant) and BODY; Warp keeps structural zeros and allocates BODY rows densely; Warp allocates `rowadr` with atomics (non-contiguous order). Compare by densifying.
13. **BODY transmission exclusion test**: C uses `con->exclude == 1` (in gap; excludes adhesion/speculative from that class) and skips non-excluded contacts with `efc_address < 0` without counting; Warp uses `dist >= includemargin` and counts every relevant contact.
14. **Tendon limit force sign in `cfrc_ext`**: C distinguishes lower/upper rows by order (lower first if active); Warp by proximity.
15. **Equality rows order in `rnePostConstraint`**: C walks rows by eq type in efc order; Warp assumes connect-then-weld blocks (its own constraint ordering). Equality row ordering of the two engines differs.
16. **`addTendon` threshold**: C skips segments with length `< mjMINVAL`; Warp skips only exact zero.
17. **Tendon wrap failure**: when a geom does not wrap, C still advances by 2 (skips the geom) and records only the first site; same in Warp. A wrap path with sidesite inside the geom (`wrap_inside`) returns zero arc length and two identical tangent points.
18. **`mj_tendon` zeroes all lengths first**: sleeping tendons (C, sleep enabled) end with `ten_length = 0` and empty Jacobians; Warp computes all tendons.
19. **Fixed tendons with repeated joints**: C accumulates into the first matching column (`combineSparseInc`); Warp overwrites with the last coefficient.
20. **`d.cacc` / `d.cfrc_int` after the velocity stage**: C leaves them untouched by `mj_rne` (stack locals); Warp's `rne` writes `cacc` (with `flg_acc=False`) and `cfrc_int` (bias-only) into Data.
21. **`mj_subtreeVel`** is lazy in C (`flg_subtreevel`), computed on demand by sensors/energy; Warp's `subtree_vel` is called explicitly and uses a mathematically equivalent but differently-associated formula.
22. **Gravity**: `cacc[0] = −opt.gravity` (both C and Warp, `mjDSBL_GRAVITY` → 0). Warp supports per-world gravity (`gravity[worldid % n]`). Gravcomp forces live in `qfrc_passive` or `qfrc_actuator` (actgravcomp), never in `qfrc_bias`.
23. **Free joint `xaxis`** = `jnt_axis` (C, Warp) vs (0,0,1) (MJX). Free joint `xanchor = xpos`.
24. **Ball joint transmission length** uses the expmap `quat2Vel(normalize4(q), 1)` which wraps angles > π to the opposite direction (`speed −= 2π`).
25. **`wrap_circle` with sidesite** uses winding angles (`atan2(p×q, p·q)`) to choose the orientation and compute the arc; without sidesite, uses the shorter-path heuristic and `length_circle` with the `1e-6` angle threshold. The intersection penalty is exactly `−10000`.
26. **Float32 GPU**: all of Warp's kinematics/dynamics run in float32; tolerances such as `MJ_MINVAL = 1e-15` become effectively "exactly zero" tests for most quantities; `wrap_inside` tolerance `1e-6` and the `length_circle` 1e-6 threshold are close to float32 resolution of angles.

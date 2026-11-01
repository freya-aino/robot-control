# 11 — Derivatives, Inverse Dynamics, and the Implicit Effective Metric

**Sources covered**

C engine (commit 7ba52b5):
- `src/engine/engine_derivative.c` / `.h` — analytic derivatives: spatial-algebra helpers, `mjd_subQuat`, `mjd_quatIntegrate`, RNE velocity derivatives (`mjd_rne_vel`, dense test variant), free-body gyroscopic block (`mjd_freeBias_vel`, `mjd_freeMhat`, `mjd_freeGyroPossible`), actuator derivatives (`mjd_actuator_vel`, `mjd_actuatorVelDeriv`, `mjd_actuatorLenDeriv`, muscle curves), passive derivatives (`mjd_passive_vel`: DOF/tendon damping, ellipsoid and inertia-box fluid), entry point `mjd_smooth_vel`, flex implicit stiffness operators and CSR assembly (`mjd_flexInterp_*`, `mjd_flexBend_*`, `mjd_flexStretch_*`, rod hinges, `mjd_flexStiff_assemble`), and the flex-contact law (`mjd_flexContactSlack/Residual/Stiffness`).
- `src/engine/engine_derivative_fd.c` / `.h` — `mj_stepSkip`, `mjd_stepFD`, `mjd_transitionFD`, `mjd_inverseFD`, `mjd_smooth_velFD`, `mjd_passive_velFD`.
- `src/engine/engine_inverse.c` / `.h` — `mj_invPosition`, `mj_invVelocity`, `mj_discreteAcc` (invdiscrete), `mj_invConstraint`, `mj_inverseSkip`, `mj_inverse`, `mj_compareFwdInv`.
- `src/engine/engine_metric.c` / `.h` — the **implicit effective metric** $\widehat M = M + hD + h^2K$ used by `integrator="discrete"` (the efm_* machinery: build, shift, actuation refresh, multiply, PCG solve, preconditioners incl. 3x3 blocks, sparse blocked Cholesky, dense component blocks, island variants, flex passive-contact rank-1 rows).
- Helpers read for exactness: `engine_core_util.c` (`mj_isMetric`, `mj_effCouplings`, `mj_eff*Possible`, `mj_flexSimple`, `mj_flexBodySimple`, `mj_actuatorDamping`, `mj_dcmotorResistance`), `engine_util_misc.c` (`mjd_xPolyForce`), `engine_util_solve.c` (`mju_cholFactor`, `mju_factorLU6`), `engine_core_constraint.c` (`mj_constraintUpdate_impl`, `mj_regularizeConstraint`), `engine_forward.c` (callers: `mj_implicitSkip`, `mj_discreteGyro`, `mj_fwdAcceleration`, `mj_forwardSkip`).

GPU / JAX:
- MuJoCo Warp `mujoco_warp/_src/derivative.py` (`deriv_smooth_vel`, `deriv_rne_vel`, fluid kernels, all `eff_*` metric functions and PCG), `inverse.py`, plus the derivative-related parts of `forward.py` (`implicit`, `implicit_qlu`, `_implicit_free_body`, `_discrete_free_gyro_kernel`, `check_discrete`) and `io.py` (precomputed index maps).
- MJX `mjx/_src/derivative.py`, `mjx/_src/inverse.py`.

**Role in the pipeline.** Analytic velocity derivatives `qDeriv = ∂(qfrc_actuator + qfrc_passive − qfrc_bias)/∂qvel` feed the `implicit` (full, with RNE term, non-symmetric LU) and `implicitfast` (no RNE term, symmetrized, LDLᵀ) integrators, and the `invdiscrete` inverse. The effective metric (engine_metric.c) is the per-step object that makes `integrator="discrete"` linearly implicit in **both** velocity and position: forward dynamics solves $(M + hD + h^2K)\,\dot v = f + c$ with the smooth shift $c=-hKv$; the constraint solver, warmstart, and inverse dynamics all consume the same metric. Finite-difference routines (`mjd_transitionFD`, `mjd_inverseFD`) produce linearizations of `mj_step` / `mj_inverse` by re-running the pipeline with stage skipping. Inverse dynamics computes `qfrc_inverse` analytically from `qacc`, using the convex-dual relation (forces from `jar = J qacc − aref` via `mj_constraintUpdate`, no optimization).

---

## 0. Notation, constants, conventions

| Symbol | Meaning |
|---|---|
| `h` | `m->opt.timestep` |
| `mjMINVAL` | `1e-15` (double), `1e-15f` (float) |
| `mjNPOLY` | 2 (polynomial stiffness/damping coefficients) |
| `mjNGAIN = mjNBIAS = mjNDYN` | 10 |
| `mjNFLUID` | 12 |
| `mjFLEXCONTACT_OMEGA2` | `5e7` |
| `EFF_CHOL_MAXVERT` | 8192 |
| `EFF_CHOL_PIVOT` | `1e-12` |
| `EFF_DENSE_MAXNU` | 600 |
| `EFF_HDR` | 1; `EFF_CHOL_PENDING` = 2; `EFF_ISLAND_TAG` = −1 |
| `mjROD_HINGE_SIZE` | 10 |

**Sparse layouts (model-level, precomputed at compile):**
- **D** (`D_rownnz/D_rowadr/D_colind/D_diag`, nnz `nD`): full (both triangles) dof-dof pattern; row `j` equals the row of `dof_bodyid[j]` in **B** (asserted by `checkDBSparse`). Columns sorted ascending. `qDeriv` and `qLU` live here.
- **B** (`B_rownnz/B_rowadr/B_colind`, nnz `nB`): body × dof; row `i` = dofs of all ancestors of body `i` plus all dofs of its subtree, sorted.
- **M** (`M_rownnz/M_rowadr/M_colind`, nnz `nC`): *reduced* lower triangle including the diagonal (last entry of each row is the diagonal). For "simple" dofs (`dof_simplenum`) the within-body off-diagonal entries are omitted. `qH`, `M`, `efm_fluid` live here.
- `mapM2D` (nD): for each D entry, the M index of the symmetric lower element (or −1 if not in M). `mapD2M` (nC): for each M entry, its D index.

**qDeriv sign convention.** `qDeriv` is the derivative of the *force* side. Implicit integration uses `M − h·qDeriv`. `mjd_rne_vel` *subtracts* `∂qfrc_bias/∂qvel`.

**Polynomial derivative** (`mjd_xPolyForce(linear, poly, x, n, flg_odd)`): derivative of $x\cdot p(x)$ where $p(x)=\text{linear}+\sum_i \text{poly}_i x^{i+1}$:
```
x = flg_odd ? |x| : x
res = linear; xpow = 1
for i in 0..n-1: xpow *= x; res += (i+2)*poly[i]*xpow
```
With n = 2: `linear + 2*poly[0]*x + 3*poly[1]*x^2`. Damping uses `flg_odd=1`, stiffness `flg_odd=0`.

**Actuator-inherited damping** (`mj_actuatorDamping(m, type, id, poly)`): `actuatorid = jnt_actuatorid[id]` or `tendon_actuatorid[id]`; −1: 0. If ≥ 0: `gear2 = actuator_gear[6*actuator_outadr[a]]^2`, returns `actuator_damping[a]*gear2` and adds `actuator_dampingpoly[a]*gear2` into `poly`. If < −1 (several actuators): scan all actuators with `actuator_trnid[2k]==id` and matching transmission type (JOINT/JOINTINPARENT for joints, TENDON for tendons), summing contributions in actuator order. The total damping used everywhere is `dof_damping[i] + mj_actuatorDamping(...)` with `poly = dof_dampingpoly[i] + actuator polys`.

---

## 1. Spatial-algebra derivative helpers (`engine_derivative.c`)

All 6-vectors are MuJoCo spatial motion/force vectors `[ang(3); lin(3)]`. Matrices are row-major 6×6, `D[6*r + c] = ∂res[r]/∂x[c]`.

### 1.1 `mjd_cross(a, b, Da, Db)` (static)
Derivatives of `c = a × b` (3×3, row-major), either output optional (NULL skip):
```
Da = -[b]x = [[0, b2, -b1], [-b2, 0, b0], [b1, -b0, 0]]
Db =  [a]x = [[0,-a2, a1], [a2, 0,-a0], [-a1, a0, 0]]
```
(exactly: `Da[1]=b2, Da[2]=-b1, Da[3]=-b2, Da[5]=b0, Da[6]=b1, Da[7]=-b0`; `Db[1]=-a2, Db[2]=a1, Db[3]=a2, Db[5]=-a0, Db[6]=-a1, Db[7]=a0`.)

### 1.2 `mjd_crossMotion_vel(D, v)` — ∂(vel ×ₘ v)/∂vel
`mju_crossMotion(res, vel, v)`: `res_ang = vel_ang × v_ang`, `res_lin = vel_ang × v_lin + vel_lin × v_ang`. Nonzeros:
```
row0: D[2]=-v1, D[1]=v2
row1: D[8]=v0,  D[6]=-v2
row2: D[13]=-v0, D[12]=v1
row3: D[20]=-v4, D[19]=v5, D[23]=-v1, D[22]=v2
row4: D[26]=v3,  D[24]=-v5, D[29]=v0, D[27]=-v2
row5: D[31]=-v3, D[30]=v4, D[34]=-v0, D[33]=v1
```

### 1.3 `mjd_crossForce_vel(D, f)` — ∂(vel ×* f)/∂vel
`res_ang = vel_ang × f_ang + vel_lin × f_lin`, `res_lin = vel_ang × f_lin`:
```
row0: [2]=-f1,[1]=f2,[5]=-f4,[4]=f5
row1: [8]=f0,[6]=-f2,[11]=f3,[9]=-f5
row2: [13]=-f0,[12]=f1,[16]=-f3,[15]=f4
row3: [20]=-f4,[19]=f5
row4: [26]=f3,[24]=-f5
row5: [31]=-f3,[30]=f4
```

### 1.4 `mjd_crossForce_frc(D, vel)` — ∂(vel ×* f)/∂f
```
row0: [1]=-v2,[2]=v1,[4]=-v5,[5]=v4
row1: [6]=v2,[8]=-v0,[9]=v5,[11]=-v3
row2: [12]=-v1,[13]=v0,[15]=-v4,[16]=v3
row3: [22]=-v2,[23]=v1
row4: [27]=v2,[29]=-v0
row5: [33]=-v1,[34]=v0
```

### 1.5 `mjd_mulInertVec_vel(D, i)` — ∂(cinert · v)/∂v
`i[10] = [I00,I11,I22,I01,I02,I12, m·c0, m·c1, m·c2, m]` (com-based inertia). Nonzeros:
```
row0: [0]=i0,[1]=i3,[2]=i4,[4]=-i8,[5]=i7
row1: [6]=i3,[7]=i1,[8]=i5,[9]=i8,[11]=-i6
row2: [12]=i4,[13]=i5,[14]=i2,[15]=-i7,[16]=i6
row3: [19]=i8,[20]=-i7,[21]=i9
row4: [26]=i6,[24]=-i8,[28]=i9
row5: [30]=i7,[31]=-i6,[35]=i9
```

### 1.6 `mjd_subQuat(qa, qb, Da, Db)` (public)
Derivatives of `mju_subQuat(res, qa, qb)` = `quat2Vel(qb⁻¹·qa, 1)` (3-vector axis·angle).
1. If both outputs NULL return.
2. `axis = mju_subQuat(qa, qb)`; `half_angle = 0.5*mju_normalize3(axis)` (normalize3: if norm < mjMINVAL sets axis = (1,0,0) and returns the norm).
3. `K = [axis]x` = `[[0,-a2,a1],[a2,0,-a0],[-a1,a0,0]]`.
4. `Da = I + half_angle*K + coef*(K·K)` with
   `coef = 1 − (half_angle < 6e-8 ? 1 : half_angle / tan(half_angle))`.
5. `Db = −Daᵀ`.

### 1.7 `mjd_quatIntegrate(vel, scale, Dquat, Dvel, Dscale)` (public)
Reference arXiv:1711.02508 Eq. 183. `s = scale*vel`, `xx = s·s`, `x = sqrt(xx)`.
Basis: `eye`, `cross = [[0,s2,-s1],[-s2,0,s0],[s1,-s0,0]]` (note: this is **−[s]x**), `outer = s sᵀ`.
Coefficients: `a = cos x`;
- if `|x| > 1/32`: `b = sin x / x`, `c = (1−a)/xx`, `d = (1−b)/xx`;
- else (6th-order Horner Taylor):
  `b = 1 + xx/6*(xx/20*(1 − xx/42) − 1)`
  `c = (1 + xx/12*(xx/30*(1 − xx/56) − 1))/2`
  `d = (1 + xx/20*(xx/42*(1 − xx/72) − 1))/6`.

Outputs (each optional): `Dquat = a·eye + b·cross + c·outer`; `Dvel = b·eye + c·cross + d·outer` (derivative w.r.t. the *scaled* velocity, as documented by the test which compares it to FD on scaled velocity); `Dscale = Dvel · vel` (3-vector, `mji_mulMatVec3`).

---

## 2. RNE velocity derivative: `mjd_rne_vel` (static) and dense test variant

Goal: subtract `∂qfrc_bias/∂qvel` (centripetal/Coriolis + gyroscopic; gravity independent of qvel) from `qDeriv`. qfrc_bias is computed by RNE with `qacc = 0`:
```
cvel_b   = cvel_parent + Σ_j cdof_j qvel_j
cdof_dot_j = cvel_b(before adding j's own contribution, see joint rules) ×ₘ cdof_j
cacc_b   = cacc_parent + Σ_j cdof_dot_j qvel_j          (gravity term constant)
cfrc_b   = cinert_b·cacc_b + cvel_b ×* (cinert_b·cvel_b)
cfrc_parent += cfrc_child  (backward)
qfrc_bias_j = cdof_jᵀ cfrc_{body(j)}
```

### 2.1 Internal sparse format
"dense body/dof × sparse dof × 6 (inner size 6)": `Dcvel`, `Dcacc`, `Dcfrcbody` have `6*nB` entries; row of body `i` starts at `6*B_rowadr[i]`, one 6-vector per column in `B_colind`. `Dcdofdot` has `6*nD`, row `j` at `6*D_rowadr[j]` (same sparsity as the B-row of its body).

`Jadr` for dof `j` = number of dof ancestors of dof j = `(j < nv−1 ? dof_Madr[j+1] : nM) − (dof_Madr[j] + 1)`. Because B-rows are sorted and ancestor dofs have lower indices, the ancestors occupy the first `Jadr` slots of the row and dof `j` itself is at slot `Jadr`.

`copyFromParent(m,d,mat,n)`: returns if `n==0` or `body_weldid[body_parentid[n]]==0`. Otherwise counts ancestor dofs by walking `np = body_weldid[parent]` chains (`ndof += body_dofnum[np]; np = body_weldid[body_parentid[np]]`) and copies `6*ndof` numbers from the parent's row start to the child's row start (ancestor dofs are the first entries of both rows).

`addToParent(m,d,mat,n)`: same early-out; merges child row into parent row by two-pointer walk on `B_colind`; equal columns → add 6-vector; child column > parent → advance parent; child column < parent → `mjERROR("child nonzeroes must be subset of parent")`.

### 2.2 `mjd_comVel_vel(m, d, Dcvel, Dcdofdot)`
Sleep filter: `sleep_filter = ENBL_SLEEP && nbody_awake < nbody`; iterate `b = 1..nbody'` with `i = body_awake_ind[b]` if filtered.
For each body `i`: `copyFromParent(Dcvel, i)`; then for each dof `j` of body `i`:
- **FREE**: `Dcdofdot` rows of the three translational dofs stay zero (cleared beforehand). `Dcvel[slot Jadr+0..2] += cdof[j+0..2]`. Then `j += 3; Jadr += 3;` fall through to BALL for the rotational dofs.
- **BALL** (and free rotational part): for `dj=0..2`: `mat = mjd_crossMotion_vel(cdof[j+dj])`, `matT = matᵀ`, `Dcdofdot[row j+dj] (first Jadr+dj columns) = Dcvel[row i][first Jadr+dj cols] · matT` (`mju_mulMatMat(dst, Dcvel+6*Badr[i], matT, Jadr+dj, 6, 6)` — each 6-vector column times matT, i.e. D·x for each column). Then `Dcvel[slot Jadr+0..2] += cdof[j+0..2]`; `j += 2`.
  Important ordering: the three rotational cdof_dot derivatives for dof dj see Dcvel *before* adding any of the ball's own dofs, with column count `Jadr+dj` (columns of earlier dofs of the same ball are included in the multiplication count but are zero at that moment).
- **HINGE/SLIDE**: `Dcdofdot[row j][Jadr cols] = Dcvel[row i]·(crossMotion_vel(cdof_j))ᵀ`; then `Dcvel[slot Jadr] += cdof_j`.
- default: `mjERROR("unknown joint type")`.

### 2.3 `mjd_rne_vel(m, d)`
1. Stack allocate `Dcdofdot (6nD)`, `Dcvel, Dcacc, Dcfrcbody (6nB each)`, `row (nv)`. Clear all (or, under sleep filter, only awake dof rows / awake body rows).
2. `mjd_comVel_vel`.
3. Forward over bodies: `copyFromParent(Dcacc, i)`; for each dof `j` of body i: `Dcacc[slot Jadr] += cdof_dot[j]`; `Dcacc[row i] += qvel[j]·Dcdofdot[row j]` (over `6*B_rownnz[i]` numbers; identical sparsity).
   Then:
   - `dmul = mjd_mulInertVec_vel(cinert_i)`; `Dcfrcbody[row i] = Dcacc[row i]·dmulᵀ` (per column 6-vector: `dmul·x`).
   - `tmp = cinert_i·cvel_i`; `mat = crossForce_vel(tmp) + crossForce_frc(cvel_i)·dmul`.
   - `Dcfrcbody[row 0 as scratch] = Dcvel[row i]·matᵀ`; `Dcfrcbody[row i] += scratch` (`6*B_rownnz[i]`). The world row is used as temp storage.
4. Clear world row (`6*B_rownnz[0]`).
5. Backward over `b = nparent−1..1` (`parent_awake_ind` under sleep): `addToParent(Dcfrcbody, i)`.
6. For each (awake) dof `j`: `i = dof_bodyid[j]`; `row = Dcfrcbody[row i]·cdof_j` (`mju_mulMatVec(row, ., cdof_j, Bnnz[i], 6)` → one scalar per column); `qDeriv[D_rowadr[j] .. +Bnnz[i]] −= row`.

### 2.4 Dense variants (tests only)
`mjd_comVel_vel_dense` / `mjd_rne_vel_dense` (public) implement the same recursion with dense `nbody×6×nv` arrays and a final dense→sparse scatter: `qDeriv[adr] −= row[D_colind[adr]]` where `row = cdof[i*6+k]·Dcfrcbody[(body*6+k) row]` accumulated over k. Mathematically identical; different rounding order.

---

## 3. Free rigid subtree gyroscopic block

### 3.1 `mj_isFreeBody(m, body)`
True iff `body_jntnum[body]==1`, that joint is FREE, and `tree_dofnum[dof_treeid[jnt_dofadr]] == 6` (descendants add no dofs).

### 3.2 `freeBias_vel_blocks(mass, R, Iw, s, qvel_rot, lin, rot)` (static)
Derivation in source: `w = R·qvel_rot` (world angular velocity), `s = subtree_com − xpos`, `Iw` composite world inertia about the subtree CoM. With qacc = 0: `f = mass·w×(w×s)`, `τ = w×(Iw w)`; bias = `[f; Rᵀ(s×f + τ)]`.
```
ws = w × s;  Iww = Iw·w;  w_dot_s = w·s
K = s wᵀ − (w·s) I + [ws]x        (explicit entries:
  K0=s0w0−wds, K1=s0w1−ws2, K2=s0w2+ws1,
  K3=s1w0+ws2, K4=s1w1−wds, K5=s1w2−ws0,
  K6=s2w0−ws1, K7=s2w1+ws0, K8=s2w2−wds)
lin = K·R
C(:,c) = −mass·(s × K(:,c)) + (w × Iw(:,c)) + (−[Iww]x)(:,c)
rot = Rᵀ·C·R
```
The `−[Iww]x` column terms are added via ternaries: column 0 gets `(0, −Iww2, +Iww1)`, column 1 `(+Iww2, 0, −Iww0)`, column 2 `(−Iww1, +Iww0, 0)`.

### 3.3 `mjd_freeBias_vel(m, d, jnt, B[36])` (public)
Requires `d->crb` (from `mj_crb`). `crb = d->crb + 10*body`, `mass = crb[9]`, `Iw = [[c0,c3,c4],[c3,c1,c5],[c4,c5,c2]]`, `s = subtree_com[body] − xpos[body]`. `B` zeroed; rotational columns only:
`B[6*(r)+3+c] = −mass·lin[3r+c]`, `B[6*(r+3)+3+c] = rot[3r+c]` for r,c ∈ 0..2. Linear columns are zero.

### 3.4 `mjd_freeMhat(m, d, jnt, h, A[36], flg_discrete)` (public)
Returns 0 unless `mj_isFreeBody(body)` and `tree_awake[dof_treeid[adr]]`.
1. `A` = 6×6 M block gathered symmetric from the sparse lower triangle rows `adr..adr+5` (`A[6r+c]=A[6c+r]=M[...]`).
2. If `flg_discrete`: add `efm_diag[adr+r]` on the diagonal (if allocated); add `efm_fluid` in M's pattern, mirrored for off-diagonals (if allocated). (qDeriv is not computed under discrete.)
   Else: `A[6r+c] −= h·qDeriv[D entries of rows adr..adr+5]` (column `D_colind − adr`; requires qDeriv computed with `flg_bias = 0`).
3. `A += h·mjd_freeBias_vel(...)` (since qfrc_smooth contains −qfrc_bias, subtracting h·∂/∂v of it adds +h·B).
Returns 1.

### 3.5 `mjd_freeGyroPossible(m, d, jnt)` (public)
Under discrete, may the free body take the local unsymmetric solve? Requires (all):
- jnt is FREE and `mj_isFreeBody`;
- no `efm_K_rownnz[adr+r]` nonzero (when `nefmK`);
- no flex with `mj_effFlexPossible(f)` has any vertex (interp: node) body in the same tree (`body_treeid`);
- no constraint Jacobian support on dofs `adr..adr+5`: sparse — scan `efc_J_colind` per row (break at `c ≥ adr+6`, structural presence counts even if value 0); dense — any nonzero value in those 6 columns;
- no tendon in the metric list with `efm_ts[t] ≠ 0` whose `ten_J_colind` hits the 6 dofs;
- no metric actuator (`efm_aid[0..nefmA)`) whose moment rows hit the 6 dofs.

### 3.6 Consumers
- `mj_implicitSkip` (implicitfast): the qH rows of free rigid subtrees are reset to M before `mj_factorI`, then after the global solve each such joint's 6 rows of qacc are overwritten by `A x = qfrc[adr..]` solved with `mju_factorLU6` / `mju_solveLU6` (partial pivoting; returns 0 if a pivot `|.| < mjMINVAL` → rows left from global solve).
- `mj_discreteAcc` implicitfast: `qfrc[adr..] = A·qacc[adr..]` overwrite.
- `mj_discreteGyro` (forward, discrete): if `mjd_freeGyroPossible && mjd_freeMhat(…, flg_discrete=1)`: rhs = `qfrc_smooth + efm_c (+ efm_ca)`, LU6 solve, overwrite qacc rows.
- `mj_inverseSkip` (discrete, metric active): overwrite `Ma` rows by `A·qacc`.

---

## 4. Jacobian utilities for `res += Jᵀ B J`

### 4.1 `addJTBJ(m,d,res,J,B,n)` (dense J, n×nv)
For each `(i,j)` with `B[i*n+j] ≠ 0`, for each `k` with `J[i,k] ≠ 0`: `row = J(j,:)·(J(i,k)·B(i,j))` (dense nv), then for each D-entry of row k: `res[adr] += row[D_colind[adr]]`. Loop order i, j, k.

### 4.2 `addJTBJSparse(m,d,res,J,B,n,offset,J_rownnz,J_rowadr,J_colind)`
Rows of J are `offset..offset+n−1` in a CSR. For each `(i,j)` with `B≠0`, for each nonzero `k` of row i (column `colik`): `mju_addToSclSparseInc(res + D_rowadr[colik], J + adr_j, D_rownnz[colik], D_colind+D_rowadr[colik], nnz_j, J_colind+adr_j, J[ik]*B[i*n+j])` — a two-pointer merge adding `scl*src` where column indices match (src entries not present in the D row are silently dropped).

### 4.3 `addJTBJ_mulSparse(...)` (matrix-free)
`Jv = J·vec` (CSR), `BJv = B·Jv` (dense `mju_mulMatVec`), `res[J_colind] += J·BJv` (Jᵀ).

---

## 5. Actuator derivatives

### 5.1 Muscle curve derivatives
`mjd_muscleGainLength(L, lmin, lmax)` (derivative of the piecewise-quadratic FL curve): returns 0 outside `[lmin, lmax]`; with `a = 0.5(lmin+1)`, `b = 0.5(1+lmax)`:
- `L ≤ a`: `x = (L−lmin)/max(MINVAL, a−lmin)`, return `x/max(MINVAL, a−lmin)`
- `L ≤ 1`: `x = (1−L)/max(MINVAL, 1−a)`, return `x/max(MINVAL, 1−a)`
- `L ≤ b`: `x = (L−1)/max(MINVAL, b−1)`, return `−x/max(MINVAL, b−1)`
- else: `x = (lmax−L)/max(MINVAL, lmax−b)`, return `−x/max(MINVAL, lmax−b)`

Common parameter unpack (`prm = gainprm` or `biasprm`): `range = prm[0..1]`, `force = prm[2]`, `scale = prm[3]`, `lmin = prm[4]`, `lmax = prm[5]`, `vmax = prm[6]`, `fpmax = prm[7]`, `fvmax = prm[8]`. If `force < 0`: `force = scale / max(MINVAL, acc0)`.
`L0 = (lengthrange1 − lengthrange0)/max(MINVAL, range1 − range0)`; `L = range0 + (len − lengthrange0)/max(MINVAL, L0)`; `V = vel/max(MINVAL, L0·vmax)`; `y = fvmax − 1`.

FV curve: `V ≤ −1: 0`; `V ≤ 0: (V+1)²`; `V ≤ y: fvmax − (y−V)²/max(MINVAL, y)`; else `fvmax`.
dFV: `V ≤ −1: 0`; `V ≤ 0: 2V+2`; `V ≤ y: (−2V + 2y)/max(MINVAL, y)`; else 0.

- `mjd_muscleGain_len = −force·dFL(L)·FV(V)/max(MINVAL, L0)`
- `mjd_muscleGain_vel = −force·FL(L)·dFV(V)/max(MINVAL, L0·vmax)` (FL = `mju_muscleGainLength`)
- `mjd_muscleBias_len`: `b = 0.5(1+lmax)`; `L ≤ 1: dFP = 0`; `L ≤ b: x=(L−1)/max(MINVAL,b−1), dFP = x/max(MINVAL,b−1)`; else `dFP = 1/max(MINVAL, b−1)`; return `−force·fpmax·dFP/max(MINVAL, L0)`.

(The negative sign: muscle forces are negative/pulling.)

### 5.2 `mjd_actuatorDerivSkip(m, d, i, sleep_filter)` → 1 to skip
- `mj_actuatorDisabled(m,i)` (actuator group disabled) → skip.
- sleep filter and `mj_sleepState(ACTUATOR,i) == mjS_ASLEEP` → skip.
- `actuator_forcelimited[i]`: with `range = actuator_forcerange[2i..]`, `oadr = actuator_outadr[i]`:
  - gaintype SO3: skip if `‖actuator_force[oadr..oadr+2]‖ ≥ range[1]`;
  - else: skip if `force ≤ range[0] || force ≥ range[1]` (`force = actuator_force[oadr]`; note: inclusive at the bounds).

### 5.3 `actuatorInput(m, d, i)` (static)
- dyntype NONE: `adr = actuator_ctrladr[i]`; if `actuator_delay[i]`: `mj_readCtrl(m,d,i,d->time,ctrl_buf,−1)` (first element); else `ctrl[adr]`. If `!DISABLED(CLAMPCTRL) && actuator_ctrllimited[adr]`: clip to `actuator_ctrlrange[2adr..]`. (ctrllimited/ctrlrange are indexed by ctrl address.)
- else: `act_adr = actadr + actnum − 1` (last activation); `act = act[act_adr]`; if `actuator_actearly[i]`: `act = mj_nextActivation(m,d,i,act_adr,act_dot[act_adr])`.

### 5.4 `mjd_actuatorVelDeriv(m, d, i)` — ∂force/∂(actuator velocity)
```
bias_vel = 0, gain_vel = 0
biastype AFFINE:  bias_vel = biasprm[2]
biastype SO3:     bias_vel = biasprm[2]
biastype DCMOTOR: if dynprm[0] <= 0: bias_vel -= K²/R   (R = mj_dcmotorResistance, K = gainprm[1])
gaintype AFFINE:  gain_vel = gainprm[2]
gaintype MUSCLE:  gain_vel = mjd_muscleGain_vel(len, vel, lengthrange, acc0, gainprm)
gaintype DCMOTOR:
   te = dynprm[0]
   dVdw = 0
   if ctrlspec & (POS|VEL|FF)  (= 1|2|4):  R0 = max(MINVAL, gainprm[0]); dVdw = −gainprm[6]*R0/K + K
   if te > 0:        R = dcmotorResistance; s = 1 − exp(−h/te); bias_vel += K*(dVdw − K)*s/R
   else if dVdw != 0: R = dcmotorResistance; bias_vel += K*dVdw/R
   sigma1 = dynprm[6]; if sigma1 > 0: bias_vel −= sigma1      (LuGre micro-damping)
if gain_vel != 0: bias_vel += gain_vel * actuatorInput(i)
return bias_vel
```
`mj_dcmotorResistance`: `R = gainprm[0]`; if thermal slot (`mj_dcmotorSlots(...).temperature ≥ 0`): `T = act[actadr + slot]`, `R *= 1 + gainprm[2]*(T + dynprm[4] − gainprm[3])`; return `max(MINVAL, R)`.

### 5.5 `mjd_actuatorLenDeriv(m, d, i)` — ∂force/∂length (used only by the metric)
```
bias AFFINE or SO3:  bias_len = biasprm[1]
bias MUSCLE:         bias_len = mjd_muscleBias_len(...)
gain AFFINE:         gain_len = gainprm[1]
gain MUSCLE:         gain_len = mjd_muscleGain_len(...)
gain DCMOTOR:
   te = dynprm[0]; dVdl = 0
   if ctrlspec & 7: R = max(MINVAL, gainprm[0]); K = gainprm[1]; dVdl = −gainprm[4]*R/K
   if te > 0:  R = max(MINVAL,gainprm[0]); s = 1 − exp(−h/te); bias_len += K*dVdl*s/R
   elif dVdl != 0: bias_len += K*dVdl/R     (R = max(MINVAL,gainprm[0]); equals −kp)
if gain_len != 0: bias_len += gain_len * actuatorInput(i)
```

### 5.6 `mjd_actuator_vel(m, d)` (public)
Returns if `DISABLED(ACTUATION)`. `sleep_filter = ENBL_SLEEP && ntree_awake < ntree`. For each actuator not skipped: `bv = mjd_actuatorVelDeriv`; if `bv ≠ 0`, for each output row `k = 0..actuator_outnum[i]−1`: `addJTBJSparse(qDeriv, actuator_moment, &bv, 1, outadr+k, moment_rownnz/rowadr/colind)` i.e. `qDeriv += bv · m_kᵀ m_k`. (The **same** scalar is used for every output row of a multi-output actuator.)

---

## 6. Fluid-force derivatives

### 6.1 Ellipsoid model components
Local velocity `lvel = [ang(3); lin(3)]` in the geom frame (with wind subtracted from lin). `B` is 6×6 with **columns** indexing the output and rows the input? Careful: `addToQuadrant(B, D, col_quad, row_quad)` writes `B[6*(c+k) + r+l] += D[3k+l]` with `r = 3*row_quad`, `c = 3*col_quad`. Hence D's row index k maps to B row `c+k` (B rows given by `col_quad`), D's column index l maps to B column `r+l`. So with `B[row,col] = ∂force_row/∂vel_col`, a call `addToQuadrant(B, D, q_out, q_in)` adds `D` to the block (output quadrant q_out, input quadrant q_in). Quadrant 0 = angular, 1 = linear.

**`mjd_addedMassForces(B, lvel, ρ, vm, vi)`**: `vlm = ρ·vm∘lin`, `vam = ρ·vi∘ang`.
- torque += vam × ang: `mjd_cross(vam, ang, Da, Db)`; add `Db` to (0,0); scale `Da[i] *= ρ·vi[i%3]` (column scaling = chain rule through vam) and add to (0,0).
- torque += vlm × lin: `mjd_cross(vlm, lin)`; `Db`→(0,1); `Da*=ρ·vm[col]`→(0,1).
- force += vlm × ang: `mjd_cross(vlm, ang)`; `Db`→(1,0); `Da*=ρ·vm[col]`→(1,1).

**`mjd_viscous_torque(D, lvel, ρ, μ, size, slender, angdrag)`** (3×3):
```
d_max, d_min, d_mid = max, min, sum−max−min of size
eqD = 2/3 (s0+s1+s2);  lin_visc_torq = π eqD³
I_max = 8/15 π d_mid d_max⁴
II[k] = ellipsoid_max_moment(size,k) = 8/15 π size[k] · (max(size[k+1], size[k+2]))⁴
x,y,z = ang
mom_coef[k] = angdrag·II[k] + slender·(I_max − II[k])
mom_visc = (x,y,z)∘mom_coef
density = ρ / max(MINVAL, ‖mom_visc‖)
mom_sq[k] = −density·ang[k]·mom_coef[k]²
lin_coef = μ·lin_visc_torq
D = diag(x·mom_sq0 + y·mom_sq1 + z·mom_sq2 − lin_coef) + [ (row r) mom_sq·ang[r] ]   (D[3r+c] += mom_sq[c]*ang[r])
```

**`mjd_viscous_drag(D, lvel, ρ, μ, size, blunt, slender)`**:
```
A_max = π d_max d_mid
inv_dmax = d_max > MINVAL ? 1/d_max : 0;  s_k = size_k·inv_dmax
norm = ‖lin‖; inv_norm = norm > MINVAL ? 1/norm : 0; (x,y,z) = lin·inv_norm
a_raw=(s1 s2)², b_raw=(s2 s0)², c_raw=(s0 s1)²
proj_num = a_raw xx + b_raw yy + c_raw zz; inv = proj_num > MINVAL ? 1/proj_num : 0
a,b,c = raw·inv
proj_denom = a²xx + b²yy + c²zz
area_scale = d_max² sqrt(proj_num)
dA_coef = proj_denom > MINVAL ? π area_scale / sqrt(proj_denom) : 0
A_proj = π area_scale sqrt(proj_denom)
lin_coef = μ·3π·eqD
quad_coef = ρ(A_proj·blunt + slender·(A_max − A_proj))
Aproj_coef = ρ·norm·(blunt − slender)
dAproj_dv = Aproj_coef·dA_coef·( a x (b yy (a−b) + c zz (a−c)),
                                 b y (a xx (b−a) + c zz (b−c)),
                                 c z (a xx (c−a) + b yy (c−b)) )
D = −quad_coef·norm·(u uᵀ + (u·u) I)      (u = (x,y,z))
D(r,:) += −u[r]·dAproj_dv
diag(D) −= lin_coef
```

**`mjd_kutta_lift(D, lvel, ρ, size, kutta)`**: same `s_k, u, a,b,c, proj_num, proj_denom, area_scale`;
```
df_denom = proj_denom > MINVAL ? π·kutta·ρ·area_scale·norm/sqrt(proj_denom) : 0
dfx = yy(a−b) + zz(a−c); dfy = xx(b−a) + zz(b−c); dfz = xx(c−a) + yy(c−b)
proj_term = proj_denom > MINVAL ? 1/proj_denom : 0
D = 2·[[0, b−a, c−a],[a−b, 0, c−b],[a−c, b−c, 0]]     (D[3r+c'] = 2(coef_c' − coef_r))
inner = (a² proj_term − a + 1, b² proj_term − b + 1, c² proj_term − c + 1)
D(row0) += dfx·inner; D(row1) += dfy·inner; D(row2) += dfz·inner
D[r][c'] *= u_r u_c'
diag(D) −= (dfx, dfy, dfz)
D *= df_denom
```

**`mjd_magnus_force(B, lvel, ρ, size, magnus)`**: `vol = 4/3 π s0 s1 s2`, `coef = magnus·ρ·vol`; `lin' = coef·lin`, `ang' = coef·ang`; `mjd_cross(ang', lin', D_ang, D_lin)`; add `D_ang` to (1,0), `D_lin` to (1,1). (Note the double application of `coef`: once in pre-scaling, the product of the two scaled vectors gives coef², but the derivative w.r.t. the *unscaled* velocity contains coef from the other factor only… the code differentiates w.r.t. the pre-scaled vectors and uses those matrices directly, so effectively `D_ang = −[coef·lin]x` and `D_lin = [coef·ang]x` — reproduce literally.)

### 6.2 `mjd_ellipsoidFluid(m, d, res, bodyid, flg_dragonly)`
1. Stack: `J (6nv)`, `tmp (3nv)`, `colind (6nv)`, `colind_compressed (6nv)`. If sparse: `nnz = mj_bodyChain(bodyid, colind)`; 6 rows each with `rownnz = nnz`, `rowadr = i*nnz`, replicated colind. Else `nnz = nv`.
2. For each geom `g` of the body: `semiaxes = mju_geomSemiAxes(geom_size, type)`; `readFluidGeomInteraction(geom_fluid + 12g → coef, blunt, slender, angdrag, kutta, magnus, vm[3], vi[3])`. Skip if `coef == 0.0`.
3. `lvel = mj_objectVelocity(GEOM, g, flg_local=1)`; wind: `wind6 = [0,0,0, opt.wind]`, `lwind = mju_transformSpatial(wind6, 0, geom_xpos, subtree_com[body_rootid], geom_xmat)`; `lvel.lin −= lwind.lin`.
4. Geom Jacobian (rotation first then translation in J: `J[0..3nnz)` = jacr, `J[3nnz..6nnz)` = jacp): sparse `mj_jacSparse(m,d,J+3nnz, J, geom_xpos, geom_bodyid, nnz, colind, 0)`, dense `mj_jacGeom(J+3nv, J, g)`. Rotate both 3-row blocks to geom frame: `J_block = geom_xmatᵀ·J_block`.
5. `B = 0`. If `!flg_dragonly`: `mjd_magnus_force`; `mjd_kutta_lift` → (1,1). Always: `mjd_viscous_drag` → (1,1); `mjd_viscous_torque` → (0,0). If `!flg_dragonly`: `mjd_addedMassForces`.
6. **Symmetrize** (`mju_symmetrize`: `0.5(B+Bᵀ)`) iff `flg_dragonly || (integrator == IMPLICITFAST && !mj_isFreeBody(body_rootid[bodyid]))`.
7. `res += Jᵀ B J` (`addJTBJSparse` with offset 0 or `addJTBJ`).

Note B here is applied as `Jᵀ B J` where B rows/cols are `[ang; lin]` in geom frame; force is in geom frame, consistent with rotated J.

### 6.3 `mjd_inertiaBoxFluid(m, d, res, i)`
```
box_k = sqrt( max(MINVAL, I_{k+1}+I_{k+2}−I_k) / body_mass · 6 )     (inertia = body_inertia[3i..])
lvel = mj_objectVelocity(BODY, i, 1); lwind via transformSpatial at xipos with ximat; lvel.lin −= lwind.lin
J: sparse: nnz=mj_bodyChain(i); mj_jacSparse at xipos (rot rows J[0..3nnz), trans J[3nnz..)); dense mj_jacBodyCom(J+3nv, J, i)
rotate both blocks by ximatᵀ
if viscosity > 0:
   diam = (box0+box1+box2)/3
   B = −π diam³ μ             → rows 0,1,2 (each addJTBJ with 1×1 B on a single J row)
   B = −3π diam μ             → rows 3,4,5
if density > 0:
   row0: B = −ρ box0 (box1⁴+box2⁴)·2|lvel0|/64
   row1: B = −ρ box1 (box0⁴+box2⁴)·2|lvel1|/64
   row2: B = −ρ box2 (box0⁴+box1⁴)·2|lvel2|/64
   row3: B = −0.5ρ box1 box2 ·2|lvel3|
   row4: B = −0.5ρ box0 box2 ·2|lvel4|
   row5: B = −0.5ρ box0 box1 ·2|lvel5|
```
Each term is a separate rank-1 `res += B·J_rᵀ J_r` call (order: viscous rot rows, viscous lin rows, density rows 0..5). Note the sparse branch replicates `colind` in place (`colind[j*nnz+k] = colind[k]`).

---

## 7. Passive derivatives: `mjd_passive_vel(m, d)` (public)

1. Return if `DISABLED(SPRING) && DISABLED(DAMPER)`.
2. `sleep_filter = ENBL_SLEEP && ntree_awake < ntree`; iterate bodies `b < nbody'` (all bodies, including world, when unfiltered).
3. Fluid if `opt.viscosity > 0 || opt.density > 0`: for each body with `body_mass ≥ mjMINVAL`: `use_ellipsoid` = any child geom with `geom_fluid[12g] > 0`; then `mjd_ellipsoidFluid(qDeriv, i, 0)` or `mjd_inertiaBoxFluid(qDeriv, i)`.
4. Return if `DISABLED(DAMPER)`.
5. DOF damping: for each (awake) dof `i`: `poly = dof_dampingpoly[i]`, `damping = dof_damping[i] + mj_actuatorDamping(JOINT, dof_jntid[i], poly)`; `qDeriv[D_rowadr[i] + D_diag[i]] −= mjd_xPolyForce(damping, poly, qvel[i], 2, 1)`.
6. Tendon damping: for each tendon (skip if 1 tree asleep, or 2 trees both asleep under filter): `damping = tendon_damping + mj_actuatorDamping(TENDON, i, poly)`, `B = −mjd_xPolyForce(damping, poly, ten_velocity[i], 2, 1)`; skip if 0; `addJTBJSparse(qDeriv, ten_J, &B, 1, i, ten_J_rownnz/rowadr/colind)`.

Spring stiffness does not enter qDeriv (position derivative).

---

## 8. Entry point `mjd_smooth_vel(m, d, flg_bias)` (public)

```
sleep_filter = ENBL_SLEEP && nv_awake < nv
clear qDeriv: all nD (or mju_zeroSparse over awake dof rows)
mjd_actuator_vel(m, d)
mjd_passive_vel(m, d)
if flg_bias: mjd_rne_vel(m, d)
```
Order of accumulation into qDeriv (affects rounding): actuators (actuator order, output rows), fluid (body order, geom order), DOF damping, tendon damping, RNE.

Callers:
- `mj_implicitSkip`, IMPLICIT: `mjd_smooth_vel(flg_bias=1)`; `qLU = gatherMasked(M, mapM2D)` (full symmetric M into D); `qLU += −h·qDeriv`; `mju_factorLUSparse(qLU, nv, scratch, D_*, dof_awake_ind)` (returns index of clamped pivot or −1 → `mjWARN_INERTIA`); solve `mju_solveLUSparse`.
- IMPLICITFAST: `mjd_smooth_vel(flg_bias=0)`; `qH = gather(qDeriv, mapD2M)` (lower part only, in M pattern; entries of D not in M are dropped); `qH = M − h·qH` (`mju_addScl(qH, M, qH, −h, nC)`); free rigid subtree rows reset to M; `mj_factorI` (LDLᵀ); `mj_solveLD`; free-body LU6 overwrite (§3.6). Note IMPLICITFAST takes only the **lower triangle** of qDeriv; asymmetric parts in the upper triangle are lost (this is why fluid B is symmetrized under implicitfast).

---

## 9. Flex implicit-stiffness operators (consumed by the metric)

These implement `res += scale · K_flex · vec` matrix-free or assemble it into dof-level CSR. `scale = s1 + s2·flex_damping[f]`; the metric uses `s1 = h²` (0 if SPRING disabled), `s2 = h` (0 if DAMPER disabled), giving `(h² + h·d)K`. The shift uses `s1 = −h, s2 = 0` (bend/stretch) or `+h` for interp (opposite sign convention, see 9.1).

### 9.1 Interpolated flexes: `mjd_flexInterp_kernel` (+ `_mul`, `_mulRange`, `_cacheKrot`)
- `flexInterpBounds`: over non-rigid interp flexes: `order = |flex_interp|`, shell mode if `flex_interp < 0`; `npe = (order+1)²` (shell) or `(order+1)³` (volume); max nodes / max npe. If `max_npe == 0` → nothing.
- `mjd_flexInterp_processed(f)`: interp, `flex_edgeequality[f] != 3` (strain constraints handled by solver), not rigid, `mj_flexHasStiffness`.
- Per processed flex: `K = flex_stiffness + flex_stiffnessadr[f]`; `scale = s1 + s2·damping`; skip if 0. Element count: shell `2(cy·cz + cx·cz + cx·cy)` faces, volume `cx·cy·cz` cells; `dim_e = 3·npe`; element stiffness at `K + fe·dim_e²`.
- `xpos` = raw node positions (`mju_flexGatherState`). Fast path if `flex_centered[f]` and every node body has `body_simple == 2` (diag M, sliders only).
- Per element: skip volume elements with `k_elem[0] == 0` (empty cells). If `K_rot_cache`: `K_rot_cell = scale·cache[krot_adr ..]`, recompute node indices with the index-only gather. Else gather element state (`mju_flexGatherFaceState` / `mju_flexGatherCellState`, cell index decomposition `ci = fe/(cy·cz)`, `cj = (fe/cz)%cy`, `ck = fe%cz`) giving `quat`; `R = quat2Mat(quat)` (global→local), `RT = Rᵀ`; for each node block pair `(a,b)`: `K_rot(a,b) = scale·RT·K(a,b)·R`; optionally store unscaled `K_rot_out = K_rot_cell/scale`.
- If `res == NULL` (cache-only) continue.
- Fast path: `fast_tmp[n] = xmat[body_n]·vec[dofadr..+3]`; `row_a = Σ_b K_rot(a,b)·fast_tmp[b]`; `res[dof_a..+3] += xmat[body_a]ᵀ·row_a`.
- General path: per node `mj_bodyChain` + `mj_jacSparse` (point Jacobian at node xpos, 3×chain), build a 3npe-row CSR, `addJTBJ_mulSparse`.
- **Sign convention:** the interp stiffness blocks are stored with the opposite sign to bend/stretch; consumers call `mjd_flexInterp_mul` with `(−s1, −s2)` for the metric, `(+h, 0)` for the shift, and the CSR assembler folds `iscale = −(s1 + s2·damping)`.
- `mjd_flexInterpAssemblable(m)`: every node of every processed interp flex either has 0 dofs or `body_simple == 2 && body_dofnum == 3` (all-or-nothing).
- `mjd_flexInterpDynNodes(f)`: `flex_centered` → all nodes; else count nodes with `b ≥ 0 && body_dofnum[body_weldid[b]] == 3`.

### 9.2 2D bending stencils: `mjd_flexBend_mulRange`
For flexes with `!interp && !rigid && dim == 2 && bendingadr ≥ 0`, `scale ≠ 0`: gather vertex vectors (`mj_flexGather`), for each edge with second flap (`flap[1] != −1`) and **convex** stencil (`flexBendConvex`: `b[0]+b[5]+b[10]+b[15] ≥ 0`, trace of the 4×4 part of the 17-number record `b + 17e`): `wres[3v_i+x] += b[17e + 4i + j]·wvec[3v_j+x]` with `v = (edge0, edge1, flap0, flap1)`; scatter `mj_flexScatter(res, wres, scale)`. Concave stencils (rest creases sharper than 90°) are dropped to keep SPD.

### 9.3 Stretch and 1D rods: `mjd_flexStretch_mulRange`
- 1D rod (`flexRod_active`: `dim==1 && !rigid && bendingadr ≥ 0`): `flexRod_mul` — per hinge record `flex_bending + bendingadr + 1 + 10h` (count in `flex_bending[bendingadr]`): `rodHingeWeight` computes `J` (n×3n, n = 3 single hinge [kb space] or 4 hinge pair [om space]) and `W = s1·Wk + s2·damping·Wd` from `mj_rodHinge/mj_rodPair` + `mj_rodHingeWeights/mj_rodPairWeights` (degenerate → skip); `u = J vec`, `w = W u`, `wres += Jᵀ w`; scatter with scale 1.
- 2D/3D (`!interp && !rigid && dim ≥ 2`, `flex_stiffness[stiffnessadr] ≠ 0`, scale ≠ 0): `mj_flexHessian(m,d,f)` then `mj_flexHessianMul(..., scale)` (SNH: PSD-projected material Hessian; StVK: material + tensile geometric term), scatter.

### 9.4 Predicates
- `mjd_flexStiff_active(f, flg_bend, flg_stretch)`: rod → `flg_stretch && mj_flexSimple(f)`; else requires `!interp && !rigid && dim ≥ 2 && mj_flexSimple`, and `(flg_bend && bendingadr ≥ 0) || (flg_stretch && stiffnessadr ≥ 0 && stiffness[adr] ≠ 0)`.
- `mjd_flexStiff_any(flg_interp)`: any processed interp (if flg_interp), any simple stretch flex, or any simple rod.
- `mjd_flexPassiveContact_any`: any `mj_effFlexContactPossible(f)` = `flex_passive[f] && !rigid && dim ≥ 2 && !ENBL_IPC`.
- `mj_flexSimple(f)`: not interp and every vertex body (welded) with dofs satisfies `mj_flexBodySimple`: exactly 3 dofs, 3 joints, all SLIDE with axes = identity (tolerance `mjMINVAL` per component), parent weld has 0 dofs.

### 9.5 CSR assembly: `mjd_flexStiff_assemble(m, d, rownnz, rowadr, colind, val, s1, s2, flg_bend, flg_stretch, Krot)`
Two-phase. Returns nnz.
1. **Slots:** `bodyslot[b] = 1` for welded vertex bodies with 3 dofs of active flexes; also interp node bodies with 3 dofs if `Krot`. Renumber slots in body order → `nvert`. `vslot[gv]`, `nslot[node]` map to slots (−1 if none). If `nvert == 0`: zero rownnz (and rowadr), return 0. `vdof[slot] = body_dofadr`.
2. **Candidate cliques** (with duplicates): rod hinges (3 or 4 vertices, each slot gets nr candidates), bending flaps (4×4, only edges with `flap[1] != −1`, concavity *not* checked here — "the pattern keeps the stencil"), stretch elements ((dim+1)-cliques of `flex_elem`), interp elements (npe-cliques via `FLEXINTERP_WALK`, skipping empty volume cells).
3. **Neighbor lists:** per slot insertion sort by `vdof`, dedupe.
4. **Structure:** `rownnz[vdof[s]+k] = 3·nneigh` for k = 0..2; `nnz += 9·nneigh`; `rowadr` prefix sum over all nv rows. Phase 1 (`colind == NULL`) returns here.
5. `colind`: per vertex row, for each neighbor in sorted order, its 3 dofs. If `val == NULL` return.
6. **Values** (zeroed first); position of neighbor block found by binary search on `vdof` (`FLEXSTIFF_BLOCK`). Per active flex with `scale ≠ 0`:
   - rods: `rodHingeStiffness` (`K = Jᵀ(W J)`, 3n×3n); block `(i,j)` rotated `R_biᵀ·block·R_bj` (`xmat` of welded bodies), added.
   - bending (convex stencils only): `q = scale·b[17e+4i+j]`, skip if 0; if same body or bit-identical `xquat`: add `q` on the 3 diagonal entries of the block; else add `q·R_biᵀ R_bj`.
   - stretch: `mj_flexHessian(f)`; iterate `count = vertnum + 2·edgenum` blocks: first `vertnum` diagonal blocks from `flexvert_hessian` (6 numbers, symmetric `[h0 h1 h2; h1 h3 h4; h2 h4 h5]`), then for each edge two blocks (`transpose = id%2`, `vi = edge[transpose]`, `vj = edge[1−transpose]`, block = `flexedge_hessian` or its transpose). Skip if either slot < 0 or `rownnz[vdof[si]] == 0`. Add `scale·R_biᵀ·block·R_bj`.
   - interp (if Krot): `iscale = −(s1 + s2·damping)`, per element and node pair: `R_biᵀ·(iscale·Krot_block)·R_bj` (node bodies are not weld-mapped here).

### 9.6 Flex contact law (shared by passive penalty and IPC)
Per pair with normal row `row`, stiffness basis `k`, multiplier `lam`:
`E = 0.5·scale·d²`, `d = gap − s − lam/k`, `s = max(0, gap − lam/k)`. Force `−scale·d` along the row; curvature `scale·rowᵀrow`; convention: `scale·rowᵀrow = h²·k·JᵀJ`.
- `mjd_flexContactSlack(k, gap, lam) = max(0, gap − (lam ? lam/k : 0))`
- `mjd_flexContactResidual(k, gap, s, lam) = gap − s − (lam ? lam/k : 0)`
- With lam = 0: residual = `min(gap, 0)`.
- `flexBodyMass(body)`: `b = body_weldid[body]`; 0 unless `body_dofnum[b] == 3`; returns the diagonal of M at `body_dofadr[b]` (`M[M_rowadr+M_rownnz−1]`).
- `mjd_flexContactStiffness(m, d, con, flex_ndyn)`: `mmin = 0`; for each side with `con->flex[side] ≥ 0`:
  - interp: `mj_contactBodyWeight` (≤ 27 bodies, weights); `inv_m = Σ w²/m_b` over massive bodies; if > 0: `meff = (ndyn/flex_vertnum)/inv_m` (scale 1 if vertnum 0); `mmin = min` (first nonzero).
  - else: vertices = `con->vert[side]` or the `dim+1` element vertices (≤ 8); `mmin` = min positive vertex mass.
  Return `5e7·mmin` (0 → no stiffness).

---

## 10. Finite differences (`engine_derivative_fd.c`)

### 10.1 Utilities
- `getState(m,d,state,sensordata)`: `mj_getState(mjSTATE_PHYSICS)` = qpos, qvel, act, history, plastic (in that bit order); copies sensordata if given. **Note:** `state` buffers are sized `nq+nv+na` but PHYSICS also contains `nhistory` and `nplastic`; history is rejected (`m->nhistory` → error), but `nplastic > 0` would overflow these buffers — treat as "only nplastic = 0 supported".
- `diff(dx, x1, x2, h, n)`: `dx = (1/h)·(x2 − x1)` (multiplication by precomputed `inv_h`).
- `stateDiff(ds, s1, s2, h)`: if `nq == nv`: `diff` over `nq+nv+na`; else `mj_differentiatePos(m, ds, h, s1, s2)` (quaternion-aware, nv outputs) then `diff` on the `nv+na` tail.
- `clampedDiff(dx, x, x_plus, x_minus, h, n)`: forward `diff(x, x_plus, h)`; backward `diff(x_minus, x, h)`; centered **`diff(dx, x_plus, x_minus, 2h)` = (x_minus − x_plus)/2h** — sign-inverted relative to the state version; this is the actual behavior for `DsDu`/`D` with centered differencing when both nudges are in range (reproduce as-is for parity); neither → zeros.
- `clampedStateDiff(...)`: forward `stateDiff(s, s_plus, h)`; backward `stateDiff(s_minus, s, h)`; centered `stateDiff(s_minus, s_plus, 2h)` (correct sign); neither → zeros (`2nv+na`).
- `inRange(x1, x2, range)`: both within `[range0, range1]` inclusive.

### 10.2 `mj_stepSkip(m, d, skipstage, skipsensor)` (public)
```
mj_checkPos; mj_checkVel
mj_forwardSkip(m, d, skipstage, skipsensor)
mj_checkAcc
if ENBL_FWDINV: mj_compareFwdInv
EULER:            mj_EulerSkip(m, d, skipfactor = skipstage >= mjSTAGE_POS)
RK4:              mj_RungeKutta(m, d, 4)   (skipstage ignored)
IMPLICIT/FAST:    mj_implicitSkip(m, d, skipfactor = skipstage >= mjSTAGE_VEL)
DISCRETE:         mj_discrete(m, d) = mj_advance(act_dot, qacc)  (solve already did the velocity update)
```

### 10.3 `mjd_stepFD(m, d, eps, flg_centered, DyDq, DyDv, DyDa, DyDu, DsDq, DsDv, DsDa, DsDu)`
Outputs are **transposed** w.r.t. control convention: each input index i writes a row of length `ndx = 2nv+na` (Dy) or `ns` (Ds). Error if `m->nhistory`.
1. `restore_spec = mjSTATE_FULLPHYSICS | mjSTATE_CTRL` (+ `mjSTATE_WARMSTART` unless warmstart disabled). FULLPHYSICS = TIME | PHYSICS | PLUGIN.
2. Save `fullstate`, `ctrl` copy, `state`. `skipsensor = all Ds outputs NULL`.
3. Nominal: `mj_stepSkip(mjSTAGE_NONE)`; `next, sensor = getState`; `mj_setState(fullstate)`.
   After this, position/velocity-stage quantities in `d` correspond to the *initial* state (mj_step computes forward before integrating), which the skips rely on.
4. **Controls** (if DyDu or DsDu), `i = 0..nu−1`, skip = `mjSTAGE_VEL`:
   - `limited = actuator_ctrllimited[i]`; `nudge_fwd = !limited || inRange(ctrl, ctrl+eps)`; if so: `ctrl[i] += eps`, step, record `next_plus/sensor_plus`, restore.
   - `nudge_back = (flg_centered || !nudge_fwd) && (!limited || inRange(ctrl−eps, ctrl))`; if so: `ctrl[i] −= eps`, step, record minus, restore.
   - `DyDu row i = clampedStateDiff(next, plus?, minus?, eps)`; `DsDu row i = clampedDiff(sensor, plus?, minus?, eps)`.
   (Note: uses `actuator_ctrllimited[i]` / `ctrlrange[2i]` indexed by actuator, assuming ctrl index = actuator index.)
5. **Activations** (`i < na`), skip `mjSTAGE_VEL`: `act[i] += eps` step/record/restore; if centered `act[i] −= eps` step/record/restore. Forward: `stateDiff(next, next_plus, eps)` / `diff(sensor, sensor_plus, eps)`; centered: `stateDiff(next_minus, next_plus, 2eps)` / `diff(sensor_minus, sensor_plus, 2eps)`. No actrange clamping.
6. **Velocities** (`i < nv`), skip `mjSTAGE_POS`: `qvel[i] ± eps`, same differencing.
7. **Positions** (`i < nv`), skip `mjSTAGE_NONE`: `dpos = e_i`, `mj_integratePos(m, qpos, dpos, ±eps)` (tangent-space perturbation, quaternions renormalized by integratePos), same differencing.
Order of loops: u, a, v, q. Every evaluation restores `fullstate` (time, qpos, qvel, act, plugin, ctrl, warmstart) — warmstart is identical for all perturbations.

### 10.4 `mjd_transitionFD(m, d, eps, flg_centered, A, B, C, D)` (public)
Errors: RK4 ("RK4 integrator is not supported"), `nhistory` ("delays are not supported"), ENBL_SLEEP ("sleeping is not supported"). Allocates transposed buffers `AT (ndx×ndx)`, `BT (nu×ndx)`, `CT (ndx×ns)`, `DT (nu×ns)` as requested; offsets `DyDq = AT`, `DyDv = AT + ndx·nv`, `DyDa = AT + 2ndx·nv`; `DsDq = CT`, `DsDv = CT + ns·nv`, `DsDa = CT + 2ns·nv`; calls `mjd_stepFD(…, DyDq, DyDv, DyDa, BT, DsDq, DsDv, DsDa, DT)`; transposes into `A (ndx×ndx)`, `B (ndx×nu)`, `C (ns×ndx)`, `D (ns×nu)`. State vector `x = [dq (nv tangent); qvel; act]`.

### 10.5 `mjd_inverseFD(m, d, eps, flg_actuation, DfDq, DfDv, DfDa, DsDq, DsDv, DsDa, DmDq)` (public)
Errors: RK4, `noslip_iterations`, ENBL_SLEEP. **Forward differences only.** Output rows (transposed): `DfD* (nv×nv)`, `DsD* (nv×ns)`, `DmDq (nv×nC)` (derivative of the sparse M vector).
- `inverseSkip(stage, …, flg_actuation, force)`: `mj_inverseSkip(stage, skipsensor)`; `force = qfrc_inverse`; if `flg_actuation`: `mj_fwdActuation(m,d)`; `force −= qfrc_actuator`.
- Center: `inverseSkip(mjSTAGE_NONE)`; save sensor, `M` (nC).
- Accelerations (skip `mjSTAGE_VEL`): `qacc[i] += eps`, inverse, restore `qacc[i]`; `DfDa row i = diff(force, force_plus, eps)`; `DsDa row = diff(sensor, d->sensordata, eps)`.
- Velocities (skip `mjSTAGE_POS`): `qvel[i] += eps` likewise.
- Positions (skip NONE): `mj_integratePos(qpos, e_i, eps)`, inverse, `qpos = saved copy`; rows of DfDq, DsDq, `DmDq = diff(mass, d->M, eps, nC)`.
- `d` is **not** re-evaluated at the nominal point at the end: qfrc_inverse, sensordata, M, etc. hold the last perturbed evaluation (only qpos/qvel/qacc are restored).

### 10.6 `mjd_smooth_velFD(m, d, eps)` / `mjd_passive_velFD(m, d, eps)` (public, testing aids)
- smooth: centered; for each dof i: `qvel[i] = saved ± eps`; `mj_fwdVelocity; mj_fwdActuation`; `f± = qfrc_actuator + qfrc_passive − qfrc_bias`; `fd = (f+ − f−)·(0.5/eps)`; scattered into qDeriv by walking the D pattern column by column using per-row counters (`cnt[j]` advances when `D_colind[D_rowadr[j]+cnt[j]] == i`); error if any `cnt != rownnz`; final `mj_fwdVelocity; mj_fwdActuation` restore.
- passive: forward, uses `d->qfrc_passive` as base (assumes `mj_fwdVelocity` already called), `fd = (qfrc_passive(qvel+eps) − base)·(1/eps)` (computed as sub then `mju_scl(1/eps)`); writes into qDeriv entries (overwrites); restores with `mj_fwdVelocity`.

---

## 11. Inverse dynamics (`engine_inverse.c`)

### 11.1 `mj_invPosition(m, d)` (public)
```
flg_energypos = 0
mj_kinematics; mj_comPos; mj_camlight; mj_flex; mj_tendon
mj_makeM; mj_factorM
mj_collision
mj_makeConstraint
if ENBL_DIAGEXACT: mj_projectConstraint
mj_transmission
mj_effBuild(m, d, active = mj_isMetric(m), flg_factor = 0)   // multiply-only metric
```
Differences from `mj_fwdPosition`: no sleep wake (`mj_wakeTendon/Collision/Equality`), no `mj_island`, `mj_projectConstraint` only with DIAGEXACT, metric built without factorization (no `effBlocks`, so no 3x3 blocks/sparse factor).

### 11.2 `mj_invVelocity` = `mj_fwdVelocity` (includes `mj_effShift`).

### 11.3 `mj_discreteAcc(m, d)` (static; used when `ENBL_INVDISCRETE`)
Converts the discrete-time `qacc` (velocity difference / h) to continuous-time: solves `M qacc_c = Ĥ qacc_d` where `Ĥ` is the integrator's modified mass matrix.
- **DISCRETE**: error if `ENBL_IPC` ("discrete inverse dynamics is not supported with flag ipc"); else return (metric handled natively in `mj_inverseSkip`; `mj_effShift` is **not** called on this path).
- **RK4**: `mjERROR("discrete inverse dynamics is not supported by RK4 integrator")`.
- **EULER**: if `!DISABLED(EULERDAMP)`, check any dof with `dof_damping>0`, nonzero `dof_dampingpoly`, or `jnt_actuatorid != −1`. If none (or EULERDAMP disabled): return with qacc unchanged. (Unlike `mj_EulerSkip`, this does **not** check `DISABLED(DAMPER)`.) Else `qfrc = M·qacc`; `qfrc[i] += h·mjd_xPolyForce(damping_i, poly_i, qvel_i, 2, 1)·qacc[i]`.
- **IMPLICIT**: `mjd_smooth_vel(1)`; `qLU = gatherMasked(M, mapM2D)`; `qLU += −h·qDeriv`; `qfrc = qLU·qacc` (`mju_mulMatVecSparse` with D pattern). No free-body special case.
- **IMPLICITFAST**: `mjd_smooth_vel(0)`; save M copy (unused afterwards); `qH = gather(qDeriv, mapD2M)`; `qH = M − h·qH`; `qfrc = sym(qH)·qacc` (`mju_mulSymVecSparse`); free rigid subtrees: `qfrc[adr..] = mjd_freeMhat(…,0)·qacc[adr..]` overwrite.
- Then `mj_solveM(m, d, qacc, qfrc, 1)` (qLD solve), free stack, `mj_effShift(m, d)` (no-op unless metric active).

### 11.4 `mj_invConstraint(m, d)` (public)
`nefc == 0` → `qfrc_constraint = 0`. Else `jar = J·qacc − efc_aref` (`mj_mulJacVec`), then `mj_constraintUpdate(m, d, jar, cost=NULL, flg_coneHessian=0)`:
```
force_i = −D_i·jar_i                          (all rows, initial)
equality (i < ne): QUADRATIC
friction (i < ne+nf): jar ≤ −R·floss → force = +floss, LINEARNEG
                      jar ≥  R·floss → force = −floss, LINEARPOS
                      else QUADRATIC
contact pyramidal / limits: jar ≥ 0 → force 0, SATISFIED; else QUADRATIC
elliptic contact (dim rows, mu = con->mu, friction[]):
   U0 = jar_i·mu, Uj = jar_{i+j}·friction_{j−1}; N = U0; T = ‖U_{1:}‖
   top:    N ≥ mu·T || (T ≤ 0 && N ≥ 0)  → force 0, SATISFIED
   bottom: mu·N + T ≤ 0 || (T ≤ 0 && N < 0) → QUADRATIC (force −D·jar)
   middle: Dm = D_i/(mu²(1+mu²)); NmT = N − mu·T
           force_i = −Dm·NmT·mu; force_{i+j} = −force_i/T·U_j·friction_{j−1}; CONE
qfrc_constraint = Jᵀ·efc_force   (mj_mulJacTVec)
```
This is the analytic solution of the convex inverse problem (A cancels, only R weights remain; see docs). Requires efc_D, efc_R, efc_aref, efc_frictionloss to be consistent; under discrete these are recomputed by `mj_regularizeConstraint(flg_AR=0)` + `mj_referenceConstraint` (11.5).

### 11.5 `mj_inverseSkip(m, d, skipstage, skipsensor)` (public)
```
mj_checkDiscrete(m)
if skipstage < POS: mj_invPosition; if !skipsensor mj_sensorPos; if ENBL_ENERGY && !flg_energypos: mj_energyPos
if skipstage < VEL: mj_invVelocity; if !skipsensor mj_sensorVel; if ENBL_ENERGY && !flg_energyvel: mj_energyVel
mj_effActuation(m, d, flg_factor = ENBL_DIAGEXACT)            // no-op unless metric active
if mj_isMetric(m): mj_regularizeConstraint(m, d, flg_AR=0); mj_referenceConstraint(m, d)
if ENBL_INVDISCRETE: save qacc; mj_discreteAcc (modifies qacc in place)
mj_invConstraint
qfrc_inverse = mj_rne(flg_acc=0)   (bias: centripetal+Coriolis+gravity)
mj_tendonBias(qfrc_inverse)        (tendon armature bias)
if !skipsensor: flg_rnepost = 0; mj_sensorAcc
Ma = M·qacc                         (mj_mulM)
if efm_active:
    mj_effMulAdd(Ma, qacc, flg_contact=1)            // Ma += (hD + h²K)·qacc incl. passive-contact rows
    for each joint j with mjd_freeGyroPossible && mjd_freeMhat(…, flg_discrete=1):
        Ma[adr..adr+5] = A·qacc[adr..]               // overwrite rows
    Ma −= efm_c; if efm_ca: Ma −= efm_ca
qfrc_inverse[i] += Ma[i] − qfrc_passive[i] − qfrc_constraint[i]
if ENBL_INVDISCRETE: restore qacc
```
Note: `mj_sensorAcc` runs *before* `qfrc_inverse` is complete but after the constraint forces; the acceleration sensors see the (possibly continuous-converted) qacc. Note also that `mj_effActuation` reads `ctrl/act` from `mjData` without running `mj_fwdActuation`, so `actuator_force` used by the forcerange skip in `mjd_actuatorDerivSkip` is whatever is in `d`.

Final formula (discrete, metric active): $\tau_{inv} = c_{bias} + (M + hD + h^2K)\,\dot v - c - c_a - f_{passive} - J^T f$, mirroring forward $(M+hD+h^2K)\dot v = \tau + f_{passive} - c_{bias} + c + c_a + J^T f$.

### 11.6 `mj_inverse` = `mj_inverseSkip(NONE, 0)`.

### 11.7 `mj_compareFwdInv(m, d)` (public)
```
solver_fwdinv[0] = solver_fwdinv[1] = 0; return if nefc == 0
qforce = qfrc_applied + qfrc_actuator; mj_xfrcAccumulate(qforce)
save qfrc_constraint, efc_force
mj_inverseSkip(m, d, mjSTAGE_VEL, skipsensor=1)
solver_fwdinv[0] = ‖qfrc_constraint_saved − qfrc_constraint_inv‖₂
solver_fwdinv[1] = ‖qforce − qfrc_inverse‖₂
restore qfrc_constraint, efc_force
```
Side effects persisting: `qfrc_inverse`, `efc_state`, and anything `mj_inverseSkip` touched at actuation level (efm actuation terms recomputed, efc_D/aref recomputed under discrete). Called from `mj_step` and `mj_stepSkip` after forward. `ENBL_IPC && ENBL_FWDINV` is an error (`mj_checkDiscrete`). Not normalized despite the XML doc's "relative norm" wording.

---

## 12. The implicit effective metric (`engine_metric.c`) — what it is

**Purpose.** Under `integrator="discrete"` (`mj_isMetric(m)` ⇔ `opt.integrator == mjINT_DISCRETE`), forward dynamics computes the discrete step map. Linearizing smooth forces about the current state over one step:
$$ M\frac{v^+ - v}{h} = f(q + h v^+, v^+) \approx f - (hD + h^2K)\frac{v^+-v}{h} - hKv $$
gives $\widehat M\,\dot v = f + c$, $\widehat M = M + hD + h^2K$, $c = -hKv$ (K: stiffness, D: damping, both clamped PSD). `qacc` therefore holds $(v^+ - v)/h$.

**Classes of terms** (each clamped PSD, "stabilizing sign"):
1. **Per-dof diagonal** `efm_diag[i] = h·max(0, dfd_i) + h·efm_ck[i]`, `efm_ck[i] = h·max(0, k_i)` (joint damping incl. actuator-inherited + poly; joint stiffness via spring derivative).
2. **Fluid drag blocks** `efm_fluid` (M pattern, symmetric): `−h·gather(drag-only fluid derivative)`.
3. **Rank-1 tendon terms**: `efm_ts[t] = h²k + h·b`, row `ten_J`.
4. **Rank-1 actuator terms**: `efm_as[t] = h²gp + h·gv`, rows = moment rows (all outputs).
5. **Rank-1 passive flex contact**: packed rows with `scale = h²k`.
6. **Flex stiffness** `(h² + h·d)·(K_bend + K_stretch [+ K_interp])`: CSR `efm_K_*` when assemblable; matrix-free operators otherwise.

Coriolis/centripetal derivatives are excluded (not PSD); free bodies get the local gyroscopic correction (§3).

**Backbone** `qH = M + diag(all classes' diagonal projections) + efm_fluid` (inside M's tree sparsity, no fill-in): exact metric when no couplings, preconditioner otherwise, and the only metric seen by the dual (PGS) solver.

**mjData fields** (per-step arena allocations, reset by `mj_effBuild`): `efm_active, nefmK, nefmcon, nefmT, nefmA, nefmdof, nefmL, efm_c, efm_diag, efm_ck, efm_sdiag, efm_fluid, efm_tid, efm_ts, efm_tk, efm_aid, efm_as, efm_ak, efm_ca, efm_K_rownnz/rowadr/colind/val, efm_dofid, efm_con_ind, efm_con_val, efm_L, flexelem_krot` (latter not arena).
Model fields from `mj_setConst`: `nefm0dof, nefm0L, efm0_dofid, efm0_L_* , efm0_L` (constant factor of `M + (h²+h·d)·K_bend` for bending-only models), `nefmCvert, efmC_perm` (minimum-degree vertex ordering for the sparse factor).

### 12.1 Gating predicates (`engine_core_util.c`)
- `mj_effCouplings(m) = mj_isMetric && solver != PGS` — tendon/actuator classes only if not PGS.
- `mj_effTendonPossible(i) = (!DSBL_SPRING && (stiffness≠0 || stiffnesspoly≠0)) || (!DSBL_DAMPER && (damping≠0 || dampingpoly≠0 || tendon_actuatorid≠−1))`.
- `mj_effActuatorPossible(i)`: `!DSBL_ACTUATION` and biastype ∈ {AFFINE, SO3, DCMOTOR, MUSCLE} or gaintype ∈ {AFFINE, SO3, MUSCLE, DCMOTOR}.
- `mj_effFlexStiffPossible(f)`: not rigid; dim 1 → `bendingadr ≥ 0`; stretch (edgeequality≠3 and `mj_flexHasStiffness`) or nonzero bending block (scans `flex_bending[badr .. next flex's badr or nflexbending)`).
- `mj_effFlexPossible = Stiff || ContactPossible`; `mj_effFlexAny` over flexes.

### 12.2 `mj_effBuild(m, d, active, flg_factor)` — position stage
Called by `mj_fwdPosition` (flg_factor=1) and `mj_invPosition` (0), after `mj_transmission`.
1. Reset all counters and pointers to 0/NULL; return if `!active`.
2. `flexelem_krot` zeroed; `mjd_flexInterp_cacheKrot` (unscaled corotated element stiffness).
3. `efm_c = alloc(nv)`; `efm_ck = alloc(nv)`; `any_diag = effDiagStiff(m,d)` (fills `efm_ck`); also true if any `dofDampPossible(i)` (`!DSBL_DAMPER && (dof_damping>0 || dampingpoly≠0 || jnt_actuatorid≠−1)`).
4. Tendons (if `ntendon && mj_effCouplings`): allocate `efm_tid/ts/tk (ntendon)`; list tendons with `mj_effTendonPossible` and (under sleep filter) `mj_sleepState(TENDON) == mjS_AWAKE`; `efm_ts = 0` initially.
5. Actuators: `any_act` if `mj_effCouplings` and any `mj_effActuatorPossible`; allocate `efm_aid/as/ak (nactuator)`, `efm_ca (nv)` zeroed.
6. `passive = !(DSBL_SPRING && DSBL_DAMPER)`; `any_fluid = passive && (viscosity>0 || density>0)`. `efm_diag` allocated iff `any_diag || nefmT || any_act || any_fluid`; `efm_sdiag` iff efm_diag; `efm_fluid (nC)` iff any_fluid.
7. Flex CSR: `s1 = DSBL_SPRING ? 0 : h²`, `s2 = DSBL_DAMPER ? 0 : h`; `krot = mjd_flexInterpAssemblable ? flexelem_krot : NULL`; allocate `efm_K_rownnz/rowadr (nv)`. `assemble_any = mjd_flexStiff_any(krot≠NULL) || mjd_flexPassiveContact_any`; under NEWTON also any `mjd_flexStiff_active(f,1,1)` (forces CSR for bending-only models). If so phase-1 `mjd_flexStiff_assemble(..., NULL, NULL, s1, s2, 1, 1, krot)` → `nefmK`.
8. If `passive`: `effContactBuild(m, d, h²)` (§12.9).
9. If `nefmK`: allocate colind/val; phase-2 assemble; if `flg_factor`: `effBlocks(m,d)` (§12.6). Else zero rownnz/rowadr.
10. If `nisland > 0 && nefc > 0 && nflex > 0 && !ENBL_IPC && !effCholReserved(d)`: `effIslandBuild` (§12.12).
11. `efm_active = 1`.

**`effDiagStiff`** (returns any-nonzero; zeroes `efm_ck`; returns 0 if SPRING disabled). Per joint with nonzero stiffness or stiffnesspoly; `qspring = jnt_plasticadr ≥ 0 ? plastic + adr : qpos_spring + qposadr`:
- FREE: `dif = qpos[0..3] − qspring[0..3]`; `k = max(0, mjd_xPolyForce(stiffness, spoly, ‖dif‖, 2, 0))`; `efm_ck[dadr..+3] = h·k`; then fall through to rotation with `dadr+=3, padr+=3, qspring+=3`.
- BALL: `quat = normalize4(qpos[padr..])`; `dif = mju_subQuat(quat, qspring)`; `k = max(0, xPolyForce(…, ‖dif‖, 0))`; 3 dofs `= h·k`.
- SLIDE/HINGE: `x = qpos − qspring`; `k = max(0, xPolyForce(…, x, 0))`; `efm_ck = h·k`.

### 12.3 `mj_effShift(m, d)` — velocity stage (end of `mj_fwdVelocity`, and `mj_discreteAcc`)
No-op if `!efm_active`. Order:
1. `efm_c = 0`.
2. If `!DSBL_SPRING`: `mjd_flexInterp_mul(efm_c, qvel, h, 0, flexelem_krot)`; `mjd_flexBend_mul(efm_c, qvel, −h, 0)`; `mjd_flexStretch_mul(efm_c, qvel, −h, 0)`. (Note the bend/stretch shift uses *all* stencils incl. non-simple flexes; bending shift over convex stencils only.)
3. Contact rows: for each packed row: `dot = (row·qvel)·(−scale/h)`; `efm_c[colind] += dot·row` (scale = h²k ⇒ −h·k·rowᵀrow·qvel).
4. If `efm_diag`: `effDiagDamp` → `efm_diag[i] = h·max(0, xPolyForce(damping_i, poly_i, qvel_i, 2, 1)) + h·efm_ck[i]` (damping part only if `!DSBL_DAMPER`); then `efm_c[i] −= efm_ck[i]·qvel[i]`.
5. Tendons in the list: `k = max(0, xPolyForce(stiffness, spoly, x, 2, 0))` where `x` is deadband displacement (`length > upper ? length−upper : length < lower ? length−lower : 0`, `lengthspring` from plastic if `tendon_plasticadr ≥ 0`; k = 0 if x == 0 or SPRING disabled); `b = max(0, xPolyForce(tendon_damping + act damping, dpoly, ten_velocity, 2, 1))` (0 if DAMPER disabled). `efm_ts = h²k + h·b`, `efm_tk = h·k`; if `k`: `efm_c[colind] −= (h·k·ten_velocity)·ten_J`.
6. Fluid (if `efm_fluid`): stack `scratch (nD)` zeroed; per (awake) body with mass ≥ MINVAL: ellipsoid (drag only, `flg_dragonly=1` ⇒ symmetrized, no lift/added mass) or inertia box (full — box has no asymmetric terms); `efm_fluid = gather(scratch, mapD2M)`; `efm_fluid *= −h`.

### 12.4 `mj_effActuation(m, d, flg_factor)` — actuation stage
Called after `mj_fwdActuation` in `mj_forwardSkip` (flg_factor=1) and in `mj_inverseSkip` (flg_factor = DIAGEXACT). No-op if inactive. Idempotent.
1. If `efm_aid`: `efm_ca = 0`, `nefmA = 0`; per actuator with `mj_effActuatorPossible` and not `mjd_actuatorDerivSkip`: `gv = max(0, −mjd_actuatorVelDeriv)`, `gp = max(0, −mjd_actuatorLenDeriv)`; skip if both 0; append `efm_aid=i, efm_as = h²gp + h·gv, efm_ak = h·gp`; if `gp`: for each output row r: `efm_ca[colind] −= (h·gp·actuator_velocity[r])·moment_row`.
2. If `efm_diag` (backbone): `qH = M`; `qH[diag_i] += efm_diag[i]`; for listed tendons with `efm_ts ≠ 0`: `qH[diag_c] += ts·J_c²`; for listed actuators (all output rows): `qH[diag_c] += as·moment_c²`; `qH += efm_fluid` (full M pattern). `efm_sdiag[i] = qH[diag] − M[diag]`. If `flg_factor`: `mj_factorI(qH, qHDiagInv, nv' , M pattern, dof_awake_ind)`.

### 12.5 `mj_effMulAdd(m, d, res, vec, flg_contact)` — `res += (Mtilde − M)·vec`
Order (affects rounding):
1. diag: `res[i] += efm_diag[i]·vec[i]`.
2. fluid (lower-triangle symmetric product): per row i: `acc = F_ii·vec_i + Σ_{a<diag} F_a·vec[c_a]`; `res[c_a] += F_a·vec_i` (inside loop); `res[i] += acc`.
3. rank-1 tendon then actuator rows (iterator `mj_effRank1Next` with flg_contact=0): `dot = scale·(row·vec)`; `res[colind] += dot·row`.
4. if `flg_contact`: packed contact rows, same formula inline.
5. CSR: if `nefmK && nefmdof`: vertex-triple kernel — for each covered block k (`i = efm_dofid[k]`), rows i, i+1, i+2 share columns, `nn = rownnz[i]/3` neighbor blocks; `r_a = Σ_j v_a[3j..3j+2]·vec[c..c+2]` (c = `colind[rowadr[i]+3j]`), accumulated per neighbor in order, then `res[i+a] += r_a`. Else if `nefmK`: per row `mju_dotSparse`.
6. Matrix-free fallback (`s1 = h²` or 0, `s2 = h` or 0): for each non-interp, non-rigid flex (excluding dim-1 without bending) with `!nefmK || !mj_flexSimple(f)`: `mjd_flexBend_mulRange(f)`, `mjd_flexStretch_mulRange(f)`. If `!nefmK || !mjd_flexInterpAssemblable`: `mjd_flexInterp_mul(res, vec, −s1, −s2, flexelem_krot)`.

`mj_effRank1Next` iterator: class 0 tendons (skip `ts==0` or `nnz==0`), class 1 actuator output rows (skip `as==0` or nnz 0; walks `k < actuator_outnum`), class 2 contact rows if `flg_contact` (packed `[nnz, conid, colind…]/[scale, force, val…]`), class 3 = done.

### 12.6 Preconditioner blocks: `effBlocks` and the sparse factor
`effBlocks(m, d)` (only with `nefmK` and `flg_factor`):
- `nb = effCoveredBlocks(nv, efm_K_rownnz)`: walk rows; each nonzero row starts a triple (i += 3), else i++ (covered dofs need not be 3-aligned).
- `efm_dofid` = triple starts (arena).
- `B = effCholReserve(...)` (sparse factor, see below); if NULL: island tail size `effIslandTail` if `nisland>0 && nefc>0 && nflex>0 && !IPC`; allocate `9nb + ntail` (tag `−1` at `B[9nb]` if tail).
- For each block: `effBlockRaw` = unfactored 3×3 diagonal block of `M + K` (M entries with column in [i, i+3) of rows i..i+2, plus CSR entries in that range), then `mju_cholFactor(Bk, 3, mjMINVAL)` (in-place lower Cholesky; deficient pivot `< mindiag` → set to `mindiag`, zero the column below, rank−−).
- `efm_L = B`, `nefmdof = nb`, `nefmL = 9nb + ntail`.

`chol3Solve(x, L, b)` (b may alias x):
```
r0 = b0/L0; r1 = (b1 − L3 r0)/L4; r2 = (b2 − (L6 r0 + L7 r1))/L8
r2 /= L8; r1 = (r1 − L7 r2)/L4; r0 −= L3 r1; r0 −= L6 r2; r0 /= L0
```

**Sparse blocked Cholesky** (only when every covered flex is a 1D rod or, if allowed, a 2D flex — `effChol12D(allow2d)`; `allow2d = ENBL_IPC || effFoldPossible(solver)`, `effFoldPossible = solver == CG && cone != ELLIPTIC && !ENBL_SLEEP`):
- `effCholApplies`: `nb ≥ 1`, `nb ≤ 8192`, `nefmCvert > 0`, `efmC_perm[0] ≥ 0`, `effChol12D`, no processed interp flex when interp assemblable, last triple within nv.
- `mj_effCholSetConst(m, d)` (from `mj_setConst`): `efmC_perm[*] = −1`; if `effChol12D(1)`: phase-1 CSR with s1=s2=0, covered blocks, `effCholBegin(order=1)` → minimum-degree order; `efmC_perm[perm[k]] = dofid[k]`. Stack shortage → `mjWARN_CNSTRFULL`.
- `effCholBegin`: builds the K vertex graph (neighbors from CSR rows: column `col` whose block start equals `col`), the **allowed edge CSR** (flex edges between covered 3-dof vertices; for rods also hinge spans (0,2) or (0,2),(1,3),(0,3) for pairs; plus extra pairs `xpair` not already flex pairs). Graph neighbors are restricted to allowed pairs (if any allowed pairs exist) plus all allowed pairs. Minimum-degree (`effCholMinDegree`: greedy, dense bitset adjacency, picks smallest degree with lowest index on ties, first eliminated gets perm `nb−1` — "reverse-order Cholesky"; elimination merges neighbor adjacency) when `order==1` or `order==2 && nnew`. Then symbolic `mju_cholFactorSymbolicBlocked(..., nb, block=3, …)` → `nL`.
- `effCholReserve`: perm from model ordering (`efmC_perm` restricted to the step's covered blocks; must cover all). If CG-fold or IPC: `effCholConPairs` collects the vertex pairs coupled by the rank-1 classes and efc rows (count pass then collect pass). `effCholBegin(order=2)`; require `pass.nfe > 0`. Allocate `[9nb blocks][status=PENDING][analysis][nL values]` in `efm_L`. Stack shortfalls warn `mjWARN_CNSTRFULL` and fall back to blocks.
- Lazy numeric factorization `effCholEnsure` → `effCholFill(prefilled=0)`: assemble covered `M + K` into the factor pattern: K block exact where the factor has a block (an allowed pair, or any fill without IPC); otherwise the block is dropped and its **absolute row sums** added to the diagonal of k and **absolute column sums** to that of q (keeps the matrix ≥ M+K, SPD). Each pair taken from the row of the vertex eliminated first (larger perm). M entries with `col ≥ i0` of the triple added. Then `effCholNumericBlocked(mindiag = max(MINVAL, 1e-12·maxdiag))`: right-looking by 3×3 vertex blocks from `p = nb−1` down; deficient pivots set to `mindiag` with the row zeroed; status = 1 only if full rank.
- `effCholSolveBlocked`: backward then forward blocked triangular sweeps (`H = LᵀL` form); `effCholApply` permutes rhs into factor order and back.
- `mj_effCholFoldFactor` (CG fold, IPC): builds the solver's copy `fold->F` = same analysis with efc rows (`D>0`) and rank-1 classes folded in (`effConTerm`: per term, per-vertex 3-vectors `v_k`; diagonal blocks `+= s·v_k v_kᵀ`; off-diagonal pairs `+= s·v_k v_qᵀ` if in the pattern, else complemented on both diagonals by `s·v_k v_kᵀ` and `s·v_q v_qᵀ`); if nothing folded `F[0] = PENDING` (reuse metric factor); else `effCholFill(prefilled=1)`.

### 12.7 `mj_effPrec(m, d, x, b)` and `effBlockApply`
- Inactive metric: `x = M⁻¹b` via `mj_solveLD(qLD)` (exact).
- Active: `effBlockApply(x, b, efm_L, fold=NULL)`:
  1. If fold with dense blocks covering all uncovered dofs: covered solve + dense apply, no backbone.
  2. If every dof is covered (`3·nefmdof == nv`) and not bending-only: covered solve only.
  3. Else: `rhs = copy(b)`; `x = rhs` with covered triples zeroed (and `efm0_dofid` dofs zeroed if bending-only); `mj_solveLD(x, efm_diag ? qH : qLD, …)` (backbone solve over the tree factor; the zeroed entries keep covered and uncovered parts decoupled ⇒ symmetric preconditioner); then covered solve (`effCovSolve`: sparse factor if valid else `chol3Solve` per block) on `rhs`; then fold dense components overwrite; then bending-only: `x[efm0_dofid] = cholSolveSparse(efm0_L, rhs[efm0_dofid])`.
`flg_bend = nefm0dof && !nefmdof`.

### 12.8 `mj_effSolve(m, d, x, b)` — PCG for qacc_smooth
- Inactive → `mj_effPrec` (exact M solve).
- Backbone-only (`!nefmT && !nefmA && !mj_effFlexAny`): `effBlockApply` once (exact: qH solve).
- Else PCG, `x0 = 0`:
```
r = b; x = 0; bn = r·r
if bn > MINVAL:
   tol_rel = opt.tolerance  (single precision: max(opt.tolerance, 1e-5))
   tol = tol_rel²·bn
   z = Prec(r); p = z; rz = r·z
   for it < opt.iterations:
      Ap = M·p (mju_mulSymVecSparse) ; Ap += mj_effMulAdd(p, flg_contact=1)
      pAp = p·Ap; if pAp <= 0: break (no warning)
      alpha = rz/pAp; x += alpha p; r −= alpha Ap
      if r·r < tol: break
      z = Prec(r); rznew = r·z; p = z + (rznew/rz)·p; rz = rznew
   if capped and r·r ≥ tol: worst = argmax|r_i| (first max);
       first time: mju_warning("Flex stiffness is too ill-conditioned…", sqrt(r·r/bn)); mj_warning(mjWARN_INERTIA, worst)
```
Forward use (`mj_fwdAcceleration`): `qfrc_eff = qfrc_smooth + efm_c (+ efm_ca)`; `mj_effSolve(qacc_smooth, qfrc_eff)` (sleep: awake dofs only, others zero, scatter back awake). Constraint solver (`engine_solver.c`) also uses `mj_effSolve(qacc, qfrc_constraint)` and the preconditioners.

### 12.9 `effContactBuild(m, d, scale = h²)` and `mj_effContactForce`
`nefmcon = 0`; return if `!ncon || !mjd_flexPassiveContact_any`. `flex_ndyn[f]` precomputed for interp flexes.
- Pass 1: for each contact with `con->exclude == 4` (passive-contact marker): `kcon = mjd_flexContactStiffness`; if > 0, `NV = mj_contactJacobian(m,d,con,con->dim, jacdif, …, chain)`; `nefmcon += 2 + NV` if NV else mark `exclude = 3` (affects no dofs).
- Allocate packed arrays.
- Pass 2: `jacn = con->frame(rows: 1 if dim==1 else 3) · jacdif` — only the first row (normal) is used; `s = slack(k, dist, 0)`; entries: sparse — all chain dofs (including zeros); dense — compact nonzeros (`jacn[a] != 0`). Header `ind = [nnz, contact id]`, `val = [scale·k, −k·residual(k, dist, s, 0)]` = `[h²k, −k·min(dist,0)]`. `nefmcon = adr` (trimmed).
- `mj_effContactForce(d, res)`: `res[colind] += force·row` for each row — called by `mj_passive` into `qfrc_spring` (the passive penalty force is taken from the published rows).

### 12.10 `mj_effPrecFold` (CG constraint solve)
Copies `effBlockRaw` blocks into `fold->L`, adds per-vertex 3×3 diagonal projections of: rank-1 classes incl. contact (`scale·v_a v_b` for a, b in the same block), **all** efc rows with `D ≠ 0` (active or not — comment: folding only rows active at the start gave 212 vs 59 mean CG iterations on a test), sparse or dense J; `mju_cholFactor(3, MINVAL)` each. If `fold->nu` and sparse: recompute components (`effComponents`: union-find over trees, joined by rank-1 terms and efc rows with `D > 0`, touching uncovered dofs; components ≤ 600 dofs kept, numbered by first dof; `partial` if some dropped) and if sizes match the sizing pass, lay out `U/Uadr/Sadr` and `effDenseBlock`: `S = M_uu + efm_fluid + efm_diag + Σ rank-1 + Σ_{D>0} D J_uᵀJ_u` (dense, row-major), `mju_cholFactor(n, MINVAL)` each; `S_valid` only if all full rank.
`mj_effFoldDenseSize`: 0 unless `nefmdof`, sparse J, and not all dofs covered. `mj_effPrecBlocks(fold)` = `effBlockApply(fold->L, fold)` if active.

### 12.11 Scratch-size helpers
`mj_effFoldScratch(m,d,nu)`, `mj_effMulAddScratch(m,d)`, `mjd_flexInterp_mulBytes(m)` — upper bounds on stack bytes (frame sizes via `mj_stackFrameBytes`, aligned `mj_stackBytes`). Only matter for avoiding stack overflow; not numerics.

### 12.12 Island variants
- `effIslandBuild`: tail `[EFF_ISLAND_TAG][nisland, nb, nbd, nflex, cov_adr(nisland+1), cov_ind(nb), bend_adr, bend_ind(nbd), flex_adr, flex_ind]` — per-island lists of covered blocks (by `dof_island`), bending-only dofs (only when `nb == 0`), and matrix-free flexes (`flexMatrixFree`: interp → `!interp_asm && processed`; else `mjd_flexStiff_active(1,1) && (!nefmK || !simple)`), island of a flex = island of its first body in an island.
- `mj_effMulAddIsland(island, …)`: island-local version of 12.5 (vectors in island dof coordinates via `map_dof2idof`/`map_idof2dof`; rank-1 term membership decided by the first column's tree island; CSR restricted to the island's covered triples, `mjERROR` if a column falls outside; matrix-free fallback by scatter/gather to full nv).
- `mj_effPrecIsland(...)`: backbone `solveLD` with the island's gathered factor on uncovered dofs, then sparse factor (`effCholSolveIsland`, solving the full factor with rhs zeroed outside the island) or `chol3Solve` per covered block; bending-only constant factor restricted to the island.
- `mj_effFoldMonolithic(m, d, solver)` = `efm_active && effFoldPossible(solver) && effCholReserved(d)`.

---

## 13. Programmatic optimizations (C)
- **Stack/arena:** all derivative scratch is `mj_markStack`/`mjSTACKALLOC`; efm objects are arena allocations (`mj_arenaAllocByte`, hard `mjERROR("arena overflow in implicit effective metric")`). The sparse-factor analysis is packed into `efm_L` with no pointers so `mj_copyData` carries it.
- **B/D sparsity** for RNE derivatives: O(nB) instead of O(nbody·nv); `copyFromParent` copies only the ancestor prefix.
- **Sleep filtering** everywhere via `*_awake_ind` lists.
- **Matrix-free vs CSR:** flex stiffness assembled once per step into CSR (vertex-triple kernel sharing loads across 3 rows); non-simple attachments fall back to matrix-free operators per flex.
- **K_rot cache** (`flexelem_krot`) avoids recomputing element rotations in every PCG iteration.
- **Lazy numeric factorization** (`EFF_CHOL_PENDING`) of the sparse factor; fold reuses the analysis.
- **Inline contact class** in `mj_effMulAdd` (iterator call measurable at thousands of pairs).
- **Fast path** for centered interp flexes on slider bodies (`J = R_body`).
- FD: stage skipping (`mj_forwardSkip`), factor reuse (`skipfactor`), warmstart restoration so perturbations share the solver initial guess.

---

## 14. GPU (MuJoCo Warp) implementation

All arrays are `[nworld, …]`; `float` = float32 by default; model parameters with leading batch dimension are indexed `worldid % shape[0]` (batched parameters). Warp kernels are compiled with `enable_backward=False`: **no autodiff**. There is **no** `transitionFD`/`inverseFD`/`stepFD`/`compareFwdInv` in Warp. `opt.tolerance` is clamped to `≥ 1e-6` at model load (`io.py`).

### 14.1 Precomputed index maps (`io.py`)
- `M_fullm_i/j`: for each dof i, the chain `j = i, dof_parentid[i], …` (chain-aware; may contain pairs absent from the reduced M).
- `M_elemid[nv, nv]`: (row, col) → M CSR address, −1 if not in reduced M (off-pattern pairs are **skipped**).
- `qD_fullm_i/j`: all D entries (row, col).
- `body_isdofancestor[nbody, nv_pad]`: dofs on the chain of the body (starting from its first ancestor with dofs).
- `body_fluid_ellipsoid_adr`: bodies with any geom `geom_fluid[:,0] > 0` and mass ≥ MINVAL; `body_fluid_box_adr`: other bodies b ≥ 1 with mass ≥ MINVAL.
- `body_freeadr`: bodies with exactly one FREE joint and `tree_dofnum == 6`.
- efm static pattern: `_get_efm_data` runs **C** `mj_fwdPosition` once on a temp copy with integrator=DISCRETE, solver=NEWTON, vertex bodies weld-mapped and non-simple flexes set rigid, and copies `nefmK, nefmdof, efm_K_rownnz/rowadr/colind, efm_dofid` into the Model (pattern is static on GPU; `nefmL = 9·nefmdof`). `efm_dofblk[dof]` = block index (or, when `efm0_active`, 0 on the constant-factor dofs). `efm0_active = nefm0dof > 0 && no stretch/interp stiffness && !has_flex_passive && !has_non_simple_flex`.

### 14.2 `deriv_smooth_vel(m, d, out)` — **outputs `M − h·qDeriv` in M structure (nC)**, not qDeriv
1. If not (ACTUATION and DAMPER both... precisely `not (disableflags & (ACTUATION|DAMPER))` — i.e. neither disabled): `out = 0`;
   - if `nactuator > 0` and actuation enabled: `_qderiv_actuator_passive_vel` [nworld, nactuator] computes a per-actuator scalar `vel` (see differences), then `_qderiv_actuator_passive_actuation_sparse` [nworld, nactuator]: for moment entries (i ≥ j within the actuator's single moment row), `out[M_elemid[dofi, dofj]] += m_i m_j vel` (atomics; skipped if not in M pattern).
   - `_qderiv_actuator_passive` [nworld, |M_fullm|]: for pairs in M: `q = out[madr]`; diagonal: `q −= dpoly'(damping_total, qvel)`; `out[madr] = M[madr] − h·q`.
   Else (either disabled): `out = M` copy (actuator and dof damping derivative both skipped, even if only one is disabled).
2. If DAMPER not disabled: `_qderiv_tendon_damping` [nworld, |M_fullm|]: per pair, loops over **all** tendons, finds `Ji, Jj` by scanning the tendon row, `out[madr] −= h·(−Σ_t Ji Jj dpoly'(…))` (i.e. adds h·…). O(ntendon·nnz) per pair.
3. If `m.has_fluid` (wind≠0 or density>0 or viscosity>0; not gated by SPRING/DAMPER disable flags): `_qderiv_ellipsoid_fluid` [nworld, nfluidbody, |M_fullm|] and `_qderiv_box_fluid` [same]: each thread computes the full 6×6 B (or diagonal box B) for its body and one dof pair, contracts `J_iᵀ B J_j`, `atomic_add(out[madr], −h·contrib)`. B is symmetrized for IMPLICITFAST **and DISCRETE** (all bodies; free bodies are fixed separately); drag-only (`flg_dragonly`) under DISCRETE. Box B symmetrized likewise (no-op, diagonal).

### 14.3 `deriv_rne_vel(m, d, out)` (D structure)
Dense temporaries `Dcvel, Dcacc, Dcfrcbody [nworld, nbody, nv]` and `Dcdof_dot [nworld, nv, nv]` of spatial vectors, allocated per call (O(nbody·nv) memory). Kernels launched per tree level (`m.body_tree` levels):
- `deriv_rne_cvel_cdof_dot` [nworld, level_size, nv] (thread = body × column dof k): same joint-type rules as C (the free-joint comments label dofs 0..2 as "rotation" but the code adds dofs 0..2 first and computes `Dcdof_dot` for dofs 3..5, i.e. identical to C's ordering).
- `deriv_rne_cacc_cfrcbody_forward`: `Dcacc = parent + cdof_dot[k]·[j==k] + Σ_j Dcdof_dot[j,k]·qvel[j]`; `Dcfrc = I·Dcacc + Dcvel ×* (I·cvel) + cvel ×* (I·Dcvel)` (algebraically equal to C's product of matrices; different rounding).
- `deriv_rne_cfrcbody_backward` (reverse levels; atomics into parent; deterministic mode copies the array first).
- `deriv_rne_body2jnt_sparse` [nworld, nD]: `out[elem] += h·(cdof_i · Dcfrcbody[body_i, j])` (atomic). Used only by `implicit_qlu`: `qLU = map_m2d(deriv_smooth_vel output) + h·∂bias/∂v`.

### 14.4 Implicit integration (`forward.implicit`)
- IMPLICIT: `implicit_qlu`: M-structure `M − h·qDeriv_sym` scattered into D via `mapM2D` (**both triangles get the lower-triangle value**: the asymmetric fluid part of C's implicit is lost, and D entries not in reduced M — within-body pairs of simple dofs — get 0 from this term), plus RNE term; `smooth.factor_solve_lu`; then `implicit_free_body(solve=True)` overwrites free-body rows (C does **not** do this under IMPLICIT).
- IMPLICITFAST (unless ACTUATION, SPRING, DAMPER all disabled — then plain `_advance(qacc)`): `deriv_smooth_vel`, free-body rows reset to M (`_implicit_free_body_reset_m`), `factor_solve_i` (LDLᵀ), `implicit_free_body(solve=True)`.
- `_implicit_free_body` [nworld, nfree]: builds A = M block + h·dof damping derivative (actuator-target damping included) + gyroscopic `free_bias_vel_blocks` (same formulas as C) + **recomputed** fluid derivative (unsymmetrized ellipsoid with `flg_dragonly=False`, box) projected via `_project_spatial_B`; `lu_factor_6x6` (pivot threshold `MJ_MINVAL`) / solve; multiply variant for inverse. Note: C's `mjd_freeMhat` uses qDeriv rows (actuator + passive incl. tendon damping + fluid from qDeriv); Warp ignores actuator and tendon derivative terms on free-body rows (they are recomputed from damping and fluid only).

### 14.5 Effective metric on GPU (integrator DISCRETE)
- `eff_build` (end of `fwd_position`): zero `efm_c, efm_ca, efm_diag, efm_ts, efm_as`; `passive.flex_hessian`; if `nefmK > 0`: zero `efm_K_val, efm_L`; CSR values by atomics: `_eff_flex_stretch_stiff_vert` [nworld, nflexvert], `_eff_flex_stretch_stiff_edge` [nworld, nflexedge, 2], `_eff_flex_bend_stiff` [nworld, nflexedge] (note: **no convexity filter** — C drops concave stencils `flexBendConvex`; verify `flex_bending` content/compile-time), `_eff_flex_rod_stiff` [nworld, nflexrodhinge], `_eff_flex_interp_stiff` [nworld, nflexintcell] if interp assemblable; column found by binary search `_find_csr_col`. Then `_eff_factor_blocks` [nworld, nefmdof]: 3×3 blocks via `_write_cholesky33` (each pivot `sqrt(max(MINVAL, ·))`, no zeroing of sub-column on deficiency — differs from `mju_cholFactor`). No sparse blocked Cholesky, no island tail, no dense component blocks.
- `eff_shift` (velocity stage): `efm_diag = 0`; `_eff_diag_stiff` [nworld, njnt] writes `h·k` into `efm_diag` (as ck, position-dependent part computed here rather than in build); `_eff_diag_damp_shift` [nworld, nv]: `efm_diag = h·max(0,dpoly') + h·ck`, `efm_c = −ck·qvel` (assignment); `_eff_tendon_shift` [nworld, ntendon] (atomics into efm_c; computes ts for **all** tendons, no possible/sleep filtering; no `efm_tk` array); if spring enabled: flex hessian, `flex_bend_mul / flex_rod_mul / flex_stretch_mul` with `s1=−1` scaled by timestep, `_eff_flex_interp_mul(is_shift)` with `scale = +h`; passive contact: `build_efm_contact` + `_eff_contact_shift` [naconmax] (`−scale/h`); fluid: reuse `_qderiv_*_fluid` kernels writing `−h·contrib` atomically into `efm_fluid` (zeroed first).
- `eff_actuation`: ctrl (delayed via `history.read_ctrl_delayed` if `nhistory`); `_eff_actuator_actuation` [nworld, nactuator] → `efm_as`, atomics into `efm_ca`; `qH = M`; `_eff_init_sdiag` adds `efm_diag` on diagonals; tendon/actuator `ts·J²`, `as·m²` atomics; `efm_fluid` added; factor qH (`_factor_blocks` dense tiles + `_factor_i_sparse`) unless `skip_qacc_smooth` (Newton with warmstart and `run_qacc_smooth == False`, etc.).
- `eff_mul_m(res, vec, skip)`: `res = M·vec` then `+= diag·vec + CSR·vec` (one row per thread, row-by-row `mju_dotSparse`-like, **not** the vertex-triple kernel), tendon rank-1 (atomics), actuator rank-1 (atomics), contact rank-1 (atomics, [naconmax]), fluid (gather-based row product via `M_mulm_*`), matrix-free flex if `nefmK == 0 || has_non_simple_flex` (non-simple only when nefmK>0), interp operator if not assemblable, zero sleeping dofs. Order differs from C (C: diag, fluid, rank-1, contact, CSR, flex).
- `eff_prec`: `efm0_active` → backbone `solve_LD(qHLD)` on non-constant dofs + single-thread-per-world sparse triangular solve of `efm0_L`; if `3·nefmdof == nv` → block solve only; else backbone solve on rhs with covered dofs zeroed, then `_eff_block_solve` per block. Same algorithm as `effBlockApply` without fold/dense/sparse-factor options.
- `eff_solve` (qacc_smooth): rhs = `qfrc_smooth + efm_c + efm_ca` (sleeping dofs zeroed). Shortcut to a single `solve_LD(qHLD)` if no flex terms (`nefmK>0 || non-simple || efm0_active || flex_passive || !interp_assemblable` with springs/dampers enabled), no tendon stiffness/damping, no metric actuators (model-level). Otherwise PCG with tiled reductions (`block_dim.eff_pcg`), per-world `done` flags, a global `nsolving` counter, `wp.capture_while` if `graph_conditional` on CUDA (else fixed `opt.iterations` loop). `_pcg_init_tiled` additionally short-circuits per world if no flex and all `efm_ts`/`efm_as` zero (`qacc = z0`). Convergence: `pAp ≤ 0` → done; `r·r < max(tol, 1e-5)²·bn` → done (the 1e-5 floor applies always on GPU, C only in single precision); `beta = safe_div(rznew, rz)`. Global iteration count `efm_iter` stops all worlds at `opt.iterations`. No `mjWARN_INERTIA` warning. Note the PCG updates `p` even when the iteration hits the max (harmless).
- `eff_prec_fold` (CG only, called in `solver.init_context`, **writes into `d.efm_L`**): raw blocks (+ dense-J efc fold in-kernel), tendon, actuator, sparse efc (all rows with `D ≠ 0`, `r < nefc`), contact rank-1; `_write_cholesky33`.
- Passive contact rows (`passive.build_efm_contact`, `_eff_contact_build`): fixed `EFM_CON_MAX_DOF = 24` dofs per contact, only flex-vertex/element sides on 3-dof welded bodies (normal rotated into each body frame, element barycentric weights `flex_elem_weights`), entries with `|val| ≤ 1e-12` compacted out, `force = −k·min(gap,0)`, `scale = h²k`. Rebuilt (allocating) on each call unless prebuilt (`eff_mul_m_scratch` for graph capture).
- Discrete free-body gyro (`_discrete_free_gyro_kernel`, launched only if `flex_interp_assemblable && !has_non_simple_flex`): decoupling guard like `mjd_freeGyroPossible` but sparse efc check tests `efc_J` value ≠ 0 (C: structural presence), no flex-tree check (relies on the launch gate), contact rows via `efm_con_*`; A = M + efm_diag + efm_fluid + h·bias blocks; LU6 solve (`rhs = qfrc_smooth + efm_c + efm_ca`) or multiply (inverse).
- `check_discrete`: same set of rejections as C `mj_checkDiscrete` for flex/solver/sleep combos, plus "quadratic interpolated flexes (|flex_interp| ≥ 2) not supported" and "sleep with flex not yet supported".

### 14.6 Warp inverse (`inverse.py`)
```
check_discrete; fwd_position; sensor_pos; fwd_velocity; sensor_vel
is_discrete = integrator == DISCRETE; invdiscrete = ENBL_INVDISCRETE || is_discrete
if invdiscrete: save qacc; discrete_acc(m, d, d.qacc)
if is_discrete: eff_actuation; constraint.regularize_constraint (recomputes efc D and aref)
inv_constraint (solver.init_context(grad=False): jaref + constraint update → efc_force, qfrc_constraint; zero qfrc_constraint where nefc==0 for sparse)
smooth.rne → qfrc_bias; tendon_bias(qfrc_bias); sensor_acc
discrete: qfrc_inverse = eff_mul_m(qacc) (incl. M); free-gyro multiply overwrite;
          qfrc_inverse = ((((bias + Ma) − efm_c) − efm_ca) − passive) − constraint
else:     qfrc_inverse = M·qacc; qfrc_inverse = bias + Ma − passive − constraint
restore qacc if invdiscrete
```
`discrete_acc`: RK4 raises; EULER: if EULERDAMP disabled → `qacc` unchanged; else **always** `M·qacc + h·dpoly'·qacc` then `solve_m` (no "any damping?" early-out → round-trip rounding vs C's exact pass-through); IMPLICIT: `implicit_qlu` (with the 14.4 differences), `qLU·qacc`, **free-body multiply overwrite** (C has none for implicit); IMPLICITFAST: `mul_m(M = deriv_smooth_vel output)` + free-body overwrite; DISCRETE: copy. Unlike C, `mj_effShift` is not re-run after `discrete_acc`, and `qfrc_bias` is used directly (C writes RNE into `qfrc_inverse`; same values). Warp's position stage is the full forward `fwd_position` (with sleep/island and with factorization of the metric blocks), not C's leaner `mj_invPosition`.

### 14.7 Warp actuator-derivative differences (`_qderiv_actuator_passive_vel`, `_eff_actuator_actuation`)
- One moment row per actuator (`actid` indexes moment rows): multi-output (`actuator_outnum > 1`, SO3) actuators are not handled; **no SO3 gain/bias case** (C: SO3 bias `biasprm[1]`/`[2]`, norm-based forcerange skip).
- No `mj_actuatorDisabled` (group) or sleep skip.
- `_qderiv_actuator_passive_vel`: for dyntype NONE uses `ctrl[worldid, actid]` **unclamped and undelayed** (C clamps with ctrlrange and reads delayed ctrl); `_eff_actuator_actuation` uses `ctrladr` with clamp.
- Forcerange check is applied only if the computed derivative is nonzero (same effect).
- DCMOTOR: `safe_div` (`x / (y if y≠0 else MINVAL)`) instead of `max(MINVAL, ·)`; bias-block resistance from `gainprm[0]` with thermal correction but not clamped in `_eff_actuator_actuation`.

### 14.8 Determinism and precision
- Most accumulations use `wp.atomic_add`; with `opt.deterministic & ATOMICS` kernels compile with `wp.DeterministicMode.RUN_TO_RUN` (`deterministic_max_records` set per kernel). Otherwise results are run-to-run nondeterministic at the ulp level.
- float32 throughout: analytic derivatives are fine, but the PCG relative tolerance cannot go below 1e-5 (hard floor) and 1e-6 global tolerance clamp.
- Fixed-size buffers: `naconmax` contacts, `njmax` efc rows, `EFM_CON_MAX_DOF=24`, `FLEX_INTERP_MAX_NODES=27`.

### 14.9 MJX (JAX)
- `derivative.deriv_smooth_vel` (dense nv×nv): actuators — only AFFINE gain/bias velocity terms (`biasprm[:,2]`, `gainprm[:,2]·ctrl`, with `ctrl.at[dyntype != NONE].set(act)` — assumes one activation per stateful actuator in order); `qderiv = momentᵀ·(moment·vel)`; DOF damping `−diag(dof_damping)` (no polynomial, no actuator damping); tendon `−ten_Jᵀ diag(tendon_damping) ten_J`; fluid raises `NotImplementedError`; no RNE derivative; no muscle/DC motor; no forcerange skip.
- `forward.implicit` (IMPLICITFAST only; IMPLICIT/DISCRETE not supported by `step`): `qm = full M − h·qderiv`, `jax.scipy.linalg.cho_factor` / `cho_solve` (assumes SPD; no free-body special case).
- `inverse`: `discrete_acc` only EULER (`dof_damping` only, early-out if all zero or EULERDAMP disabled) and IMPLICITFAST (dense); others raise. `inv_constraint` via `solver.Context.create(grad=False)`. `qfrc_inverse = bias + M·qacc − passive − constraint`.
- MJX is differentiable by JAX autodiff (`jax.grad`/`jax.jacfwd` through `step`), giving exact derivatives of the implemented (smoothed-where-needed) program; this replaces `mjd_transitionFD` on the JAX side. MuJoCo Warp offers neither autodiff nor FD utilities.

---

## 15. Reproduction pitfalls / parity notes

1. **qDeriv vs `M − h·qDeriv`**: C stores qDeriv (D layout, both triangles); Warp `deriv_smooth_vel` returns `M − h·qDeriv` in reduced-M layout. Don't mix them.
2. **Implicitfast keeps only the lower triangle** of qDeriv (`mapD2M` gather). Under implicitfast, ellipsoid fluid B is symmetrized except for free rigid subtrees (which are solved with the unsymmetric 6×6 LU including the gyroscopic block).
3. **Reduced M pattern drops within-body couplings of simple dofs.** C implicit keeps them (D layout); C implicitfast and Warp (both modes) drop them.
4. **Bias derivative ordering** in `mjd_comVel_vel` for ball/free joints: cdof_dot derivatives of the three rotational dofs are computed before any of them is added to Dcvel.
5. **Multi-output actuators**: C applies the same scalar `bias_vel` to every output row; Warp handles single-row actuators only.
6. **ctrl used in derivatives**: C clamps (unless CLAMPCTRL disabled) and reads delayed ctrl; Warp's qDeriv path does neither.
7. **Forcerange skip is inclusive** (`force ≤ lo || force ≥ hi`), using `actuator_force` already in `d`.
8. **Magnus derivative**: reproduce the literal pre-scaled-vector construction (§6.1).
9. **`mjd_subQuat`** small-angle switch at `half_angle < 6e-8`; **`mjd_quatIntegrate`** Taylor switch at `|x| ≤ 1/32` with the exact Horner forms.
10. **FD sign bug**: centered `DsDu` (`D` in transitionFD) is `(s(u−ε) − s(u+ε))/2ε` because `clampedDiff` passes arguments in the wrong order; forward/backward and all other centered blocks are correct.
11. **FD stage reuse**: Euler reuses its factorization (`skipfactor`) for qvel perturbations even though polynomial damping derivatives depend on qvel; implicit reuses factorizations only for ctrl/act perturbations; discrete never skips the integration. Warmstart is restored before every perturbed step.
12. **FD state buffers** sized `nq+nv+na` while `getState` writes `mjSTATE_PHYSICS` (also plastic); models with `nplastic > 0` are effectively unsupported; `nhistory` is rejected.
13. **`mjd_inverseFD`** is forward-difference only and leaves `d` at the last perturbed evaluation.
14. **invdiscrete Euler**: C passes qacc through untouched when no damping exists or EULERDAMP is disabled (exact); it does *not* consult `DISABLED(DAMPER)`, unlike `mj_EulerSkip`. Warp always round-trips through `M·qacc` → `solve_m` (EULERDAMP check only).
15. **Discrete inverse**: requires the same efm state as forward. `mj_invPosition` builds the metric without factorization; `mj_effActuation` refactors qH only with DIAGEXACT; efc D/aref are recomputed by `mj_regularizeConstraint(flg_AR=0)` + `mj_referenceConstraint`. The final `Ma` includes passive-contact rows (`flg_contact=1`) and the free-gyro overwrite, then subtracts `efm_c` and `efm_ca`. Summation order: C computes `bias + ((Ma−c−ca) − passive − constraint)` per component.
16. **Metric PSD clamping**: every scalar (dof damping/stiffness derivative, tendon k and b, actuator gp/gv) is clamped `max(0, ·)` with the sign convention `gv = max(0, −dF/dv)`, `gp = max(0, −dF/dl)`.
17. **Free/ball joint stiffness in the metric** uses the radial derivative `xPolyForce(stiffness, spoly, ‖dif‖, flg_odd=0)` isotropically on all 3 dofs (small-angle treatment for rotations).
18. **Bending stencils**: only convex stencils (trace ≥ 0) contribute values in C (pattern keeps them). Check the Warp CSR kernel for parity (it has no convexity filter).
19. **3×3 Cholesky** degeneracy handling differs (C zeros sub-column; Warp clamps pivots only). PCG convergence floors differ (Warp: `max(tol, 1e-5)`, tol ≥ 1e-6).
20. **PCG iteration count** is a global counter in Warp (shared across worlds) vs per-solve in C; both cap at `opt.iterations`; C warns `mjWARN_INERTIA` when capped and unconverged.
21. **Order of accumulation** in `efm_c` (C: flex interp, bend, stretch, contact, diag, tendon) and in `mj_effMulAdd` vs Warp's kernels and atomics — expect ulp-level differences; use deterministic mode on GPU for reproducibility.
22. **effContactBuild** marks contacts with zero-dof Jacobians as `exclude = 3` (mutates `d->contact`); Warp leaves contacts untouched and compacts entries `|v| ≤ 1e-12` (C keeps exact sparse chain entries, compacts only exact zeros in dense mode).
23. **Sparse factor preconditioner** (1D rods / 2D flexes with CG fold or IPC), dense component blocks (≤ 600 dofs), island-local metric products: C only; Warp uses 3×3 blocks + backbone. These change CG iteration counts and hence results at solver tolerance, not the converged solution.
24. **`mj_compareFwdInv`** norms are absolute L2 norms, `[0]` constraint-force discrepancy, `[1]` applied-force discrepancy; it mutates `qfrc_inverse` and efc state but restores `qfrc_constraint`/`efc_force`.

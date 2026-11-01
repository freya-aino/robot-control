# 08 — Constraint Solvers (PGS, CG, Newton, NoSlip), Line Search, Islands, Dense/Sparse Cholesky

**Sources covered**

C engine (commit 7ba52b5):
- `src/engine/engine_solver.c` / `.h` (in full): `mj_solPGS[_island]`, `mj_solNoSlip[_island]`, `mj_solCG[_island]`, `mj_solNewton[_island]`, `mj_dualFinish`, and every static helper (`saveStats`, `dualFinish`, `ARdiaginv`, `extractBlock`, `residual`, `costChange`, `pcg32_next`, `shuffle_int`, `dualState`, `dualStateChange`, `projectEllipsoid`, `solveQCQP`, `projectCone`, `inertiaScale`, `solPGS`, `solNoSlip`, `PrimalPointers`, `PrimalAllocate`, `PrimalUpdateConstraint`, `PrimalUpdateGrad`, `PrimalUpdateMgrad`, `PrimalPrepare`, `frictionCost`, `frictionCostDif`, `ellipticCostDif`, `PrimalEval`, `updateBracket`, `PrimalSearch`, `MakeMetricLower`, `MakeHessian`, `FactorizeHessian`, `HessianConeFolded`, `HessianConeUpdate`, `HessianConeFoldFaster`, `HessianCone`, `HessianIncremental`, `mj_solPrimal`).
- `src/engine/engine_island.c` (in full): `mj_dsuRoot`, `mj_dsuMerge`, `mj_dsuAssign`, `mj_floodFill`, `treeNext`, `treeIterInit`, `unionContactTrees`, `unionTendonTrees`, `unionConstraintTrees`, `mj_island`.
- `src/engine/engine_util_solve.c` (relevant parts): dense `mju_cholFactor/Solve/Update`, sparse reverse Cholesky (`mju_cholFactorSparse`, `mju_cholFactorSymbolic`, `mju_cholFactorSymbolicBlocked`, `mju_cholFactorNumeric` + supernode kernels, `mju_cholSolveSparse`, `mju_cholUpdateSparse`), band-dense Cholesky, `mju_QCQP2/3/QCQP`, `mju_boxQP` (not used by the constraint solver; summarized).
- Callers / context: `engine_forward.c:warmstart`, `fwdConstraint`, `solveIslandTask`, `mj_fwdConstraint`; `engine_core_constraint.c:mj_constraintUpdate_impl` (primal constraint state/force/cost/cone Hessian, reproduced here because the solver depends on it bit-for-bit).

GPU / JAX:
- MuJoCo Warp `mujoco_warp/_src/solver.py` (commit d41a47c), `block_cholesky.py`, `island.py`; option defaults in `io.py`.
- MJX `mjx/_src/solver.py` (brief).

**Role in the pipeline.** After `mj_fwdAcceleration` produced `qfrc_smooth` and `qacc_smooth = M^{-1} qfrc_smooth` (or the effective-metric solve), `mj_fwdConstraint` computes `efc_b = J qacc_smooth − aref`, warm-starts `qacc`/`efc_force`, then solves the convex constraint problem (Todorov's "Gauss principle with soft constraints") for `qacc`, `efc_force`, `efc_state`, `qfrc_constraint`. Dual solvers (PGS, NoSlip) work on forces using `efc_AR = J M^{-1} J' + diag(R)`; primal solvers (CG, Newton) minimize over `qacc` the cost `½(a−a₀)'M(a−a₀) + s(Ja − aref)`. Islands (computed in `mj_fwdPosition` right after `mj_makeConstraint`) partition dofs/constraints into independent blocks solved separately (optionally in parallel). The resulting `qacc` is then saved into `qacc_warmstart` at the end of the step (`mj_forward`/`mj_step` tail, `engine_forward.c` line ~1972).

Constants used throughout: `mjMINVAL = 1e-15` (double build; `1e-15f` float build), `mjMAXVAL = 1e10`, `mjNSOLVER = 200`, `mjNISLAND = 20`. Option defaults (`engine_init.c`): `tolerance = 1e-8`, `ls_tolerance = 0.01`, `noslip_tolerance = 1e-6`, `iterations = 100`, `ls_iterations = 50`, `noslip_iterations = 0`, `impratio = 1`. `mj_isSparse(m)`: `jacobian==mjJAC_SPARSE || (jacobian==mjJAC_AUTO && nv >= 60)`.

Constraint state enum (`mjtConstraintState`): `QUADRATIC=0, LINEARNEG=1, LINEARPOS=2, SATISFIED=3, CONE=4` (check `mjmodel.h`; Warp `types.ConstraintState` mirrors it). Constraint ordering in `efc_*` arrays: `[0,ne)` equality, `[ne, ne+nf)` friction (dof/tendon frictionloss), `[ne+nf, nefc)` limits then contacts. Elliptic contacts occupy `dim` consecutive rows (normal first), pyramidal contacts `2(dim−1)` rows (pairs of opposing edges), frictionless contacts 1 row.

---

## 1. Top-level control flow (`engine_forward.c`)

### 1.1 `fwdConstraint(m, d, solver, flg_island)`

```
mju_zero(qfrc_constraint, nv)
if nefc == 0:
    qacc = qacc_smooth; solver_niter[0..mjNISLAND) = 0; mj_discreteGyro; return
efc_b = J*qacc_smooth − efc_aref                       (mj_mulJacVec; sparse or dense)
warmstart(m, d)
solver_niter[0..mjNISLAND) = 0
islands_supported = flg_island && nisland>0 && !mjENABLED(mjENBL_IPC) && !mj_effFoldMonolithic(m,d,solver)
if islands_supported:
    PGS:        mju_dispatch(solveIslandTask, nisland)          (works directly on efc_* via efclist)
    CG/Newton:  gather island inputs:
                  ifrc_smooth[k]     = qfrc_smooth[map_idof2dof[k]]     k<nidof
                  ifrc_constraint[k] = qfrc_constraint[map_idof2dof[k]] (zero at this point)
                  iacc_smooth[k]     = qacc_smooth[...]
                  iacc[k]            = qacc[...]                         (warm-started)
                  iefc_force[c]      = efc_force[map_iefc2efc[c]]  c<nefc
                  iefc_aref[c]       = efc_aref[map_iefc2efc[c]]
                mju_dispatch(solveIslandTask, nisland)
                scatter: qacc[map_idof2dof[k]] = iacc[k]; qfrc_constraint[...] = ifrc_constraint[k]
                         efc_force[c] = iefc_force[map_efc2iefc[c]]
    if noslip_iterations>0: for island in 0..nisland-1 (serial): mj_solNoSlip_island
else:
    PGS: mj_solPGS; CG: mj_solCG; Newton: mj_solNewton   (maxiter = opt.iterations)
    if noslip_iterations>0: mj_solNoSlip(noslip_iterations)
if solver==PGS || noslip_iterations>0: mj_dualFinish     (always monolithic)
mj_discreteGyro
```

Notes:
- `iefc_type, iefc_id, iefc_frictionloss, iefc_D, iefc_R` are filled by `mj_island` (and D/R re-gathered by `mj_makeImpedance` refreshes, `engine_core_constraint.c` ~2210, ~3131). Unconstrained dofs (`idof >= nidof`) keep `qacc = qacc_smooth` (set in `warmstart`) and `qfrc_constraint = 0`.
- `solveIslandTask` dispatches on `m->opt.solver` (not the `solver` argument): Newton→`mj_solNewton_island`, CG→`mj_solCG_island`, else PGS.
- `mju_dispatch`: if no thread pool or `ntask<2` runs tasks serially in order `0..ntask-1` on the main thread; else uses the pool (results are independent per island, so order does not matter numerically, except that `saveStats` per island writes disjoint memory).
- `mj_fwdConstraint` errors (`mjERROR "unknown solver type"`) if `nefc>0` and solver not in {PGS,CG,Newton}. It calls `fwdConstraint(m,d,opt.solver, !mjDISABLED(mjDSBL_ISLAND))`. `mj_fwdConstraintCG` (IPC inner solve) forces `solver=CG, flg_island=0`.

### 1.2 `warmstart(m, d)`

```
if !mjDISABLED(mjDSBL_WARMSTART):
    qacc = qacc_warmstart
    jar  = J*qacc_warmstart − aref
    mj_constraintUpdate(m,d,jar,&cost_warmstart, flg_coneHessian=0)   // sets efc_force, efc_state, qfrc_constraint
    if solver == PGS:
        PGS_warmstart = efc_force·efc_b + ½ efc_force·(AR efc_force)     // dual cost of primal-consistent forces
        if PGS_warmstart > 0: efc_force = 0; qfrc_constraint = 0
    else:
        da = qacc_warmstart − qacc_smooth; Mda = M*da (+ mj_effMulAdd(..,flg_contact=1) if efm_active)
        cost_warmstart += ½ da·Mda
        mj_constraintUpdate(m,d,efc_b,&cost_smooth,0)                     // NOTE: overwrites efc_force/state with qacc_smooth's
        if cost_warmstart > cost_smooth: qacc = qacc_smooth               // tie keeps warmstart
    if nisland > 0: for i in [nidof, nv): qacc[map_idof2dof[i]] = qacc_smooth[...]
else:
    qacc = qacc_smooth; efc_force = 0
```

Important subtleties:
- For non-PGS solvers the **final `efc_force/efc_state` after warmstart correspond to `qacc_smooth`** (second `mj_constraintUpdate`), regardless of which `qacc` was selected. Primal solvers recompute them from `qacc` anyway; only `iefc_force` gathering uses them (and is overwritten).
- For PGS, `efc_force` is the primal-consistent force `−D·jar` with state projection evaluated at `qacc_warmstart` (or zero). PGS then iterates from these forces; `qacc` set here is irrelevant to PGS (overwritten by `dualFinish`).
- `qacc_warmstart` is written after the step from final `qacc`.

### 1.3 `mj_dualFinish` / `dualFinish`

```
qfrc_constraint = J' * efc_force               (mj_mulJacTVec)
qacc = mj_effSolve(qfrc_constraint)            (plain M^{-1} via qLD when metric inactive)
qacc += qacc_smooth
```

---

## 2. Primal constraint update — `mj_constraintUpdate_impl` (engine_core_constraint.c)

Used by warmstart and by the primal solvers (through `PrimalUpdateConstraint`). Inputs: `ne, nf, nefc, D, R, floss, jar (= J a − aref), type, id, contact, flg_coneHessian`. Outputs: `force`, `state`, optional `cost` (`s(jar)`), cone Hessians in `contact[id].H` (dim×dim row-major).

```
if nefc==0: cost=0; return
force[i] = −D[i]*jar[i]  for all i
s = 0
for i in 0..nefc-1:
  equality (i<ne):      s += ½ D jar²; state=QUADRATIC
  friction (i<ne+nf):
     if jar <= −R*floss:  s += −½ R floss² − floss*jar; force = +floss; state=LINEARNEG
     elif jar >= R*floss: s += −½ R floss² + floss*jar; force = −floss; state=LINEARPOS
     else:                s += ½ D jar²;                 state=QUADRATIC
  type != CONTACT_ELLIPTIC (limits, frictionless, pyramidal):
     if jar >= 0: force=0; state=SATISFIED
     else: s += ½ D jar²; state=QUADRATIC
  elliptic (dim rows starting at i):
     mu = con->mu (regularized: friction[0]*sqrt(R[i+1]/R[i]), set in mj_makeImpedance)
     U[0] = jar[i]*mu;  U[j] = jar[i+j]*friction[j-1]  (j=1..dim-1)
     N = U[0]; T = ||U[1..dim-1]||  (mju_norm)
     top    if (N >= mu*T || (T<=0 && N>=0)):  force[i..i+dim)=0; state=SATISFIED
     bottom elif (mu*N+T <= 0 || (T<=0 && N<0)): s += Σ_j ½ D[i+j] jar[i+j]²; state=QUADRATIC
     middle else:
        Dm = D[i]/(mu²(1+mu²));  NmT = N − mu*T
        s += ½ Dm NmT²
        force[i]   = −Dm*NmT*mu
        force[i+j] = −force[i]/T * U[j]*friction[j-1]
        state = CONE
        if flg_coneHessian: H (dim×dim) =
            H[0]=1; H[j] = (−mu/T) U[j]                     (first row, j≥1)
            H[k*dim+j] = (mu N / T³) U[j] U[k]   (1≤k≤j)
            H[j*(dim+1)] += mu² − mu N/T           (j≥1)
            H[k*dim+j] *= Dm * s_k * s_j  for j≥k, where s_0=mu, s_j=friction[j-1]
            copy upper → lower
     state[i+1..i+dim) = state[i];  i += dim−1
cost = s
```

`mj_constraintUpdate` additionally does `qfrc_constraint = J' force`.

The elliptic primal cost per contact (middle zone): $\tfrac12 D_m (N-\mu T)^2$ with $N=\mu\,jar_0$, $T=\|(\,jar_j \mu_j)\|$. Bottom zone equals the full quadratic $\sum_j \tfrac12 D_j jar_j^2$ (consistent because $R_j\mu_j^2=R_1\mu_1^2$, see impedance spec).

---

## 3. Utilities in `engine_solver.c`

### 3.1 `saveStats(m,d,island,iter, improvement, gradient, lineslope, nactive, nchange, neval, nupdate)`
- return if `island >= mjNISLAND`; `island = max(0, island)`; return if `iter >= mjNSOLVER`.
- writes `d->solver[island*mjNSOLVER + iter] = {improvement, gradient, lineslope, nactive, nchange, neval, nupdate}`.

### 3.2 `inertiaScale(m, d, island)` → `1 / max(mjMINVAL, trace)`
`trace = Σ_dof M[M_rowadr[dof] + M_rownnz[dof] − 1]` (diagonal of CSR-lower `d->M`), over all `m->nv` dofs (island<0) or the island's dofs `map_idof2dof[island_idofadr[island] + k]`, k<`island_nv`. This replaced `stat.meaninertia` (changelog c8362756d). Summation order: ascending dof index (monolithic) / ascending idof (island).

### 3.3 `ARdiaginv(m,d,res,nefc,efclist,flg_subR)`
`res[c] = 1/AR_ii` (i = efclist? efclist[c] : c), or with `flg_subR`: `1/max(mjMINVAL, AR_ii − R_i)`. Sparse: linear search in row i of `efc_AR` (CSR `efc_AR_rowadr/rownnz/colind`) for `colind==i`. Dense: `AR[i*(nefc+1)]` with global `d->nefc`.

### 3.4 `extractBlock(m,d,Ac,start,n,flg_subR)`
Copies the n×n diagonal block `AR[start..start+n, start..start+n]` row-major into `Ac`. Sparse: assumes the block is fully dense and the same column offset `k` (position of column `start` in row `start`) applies to all n rows; `mjERROR("internal error")` if not found. With `flg_subR`: `Ac_jj = max(1e-10, Ac_jj − R[start+j])` (note **1e-10**, not mjMINVAL).

### 3.5 `residual(m,d,res,i,dim,flg_subR)`
`res[j] = efc_b[i+j] + AR[i+j,:]·efc_force` (sparse: `mju_dotSparse` over the CSR row; dense: `mju_dot` over nefc); with `flg_subR`: `res[j] −= R[i+j]*force[i+j]` (i.e. residual of unregularized A).

### 3.6 `costChange(A, force, oldforce, res, dim)`
```
delta = force − oldforce
change = (dim==1) ? ½ δ² A[0] + δ res[0]
                  : ½ δ'Aδ (mju_mulVecMatVec) + δ·res
if change > 1e-10: force = oldforce; change = 0          // reject cost increase
return change
```

### 3.7 PCG32 RNG and shuffle
```
pcg32_next(rng): old = state; state = old*6364136223846793005ULL + (inc|1)
                 xorshifted = (uint32)(((old >> 18) ^ old) >> 27); rot = (uint32)(old >> 59)
                 return (xorshifted >> rot) | (xorshifted << ((-rot) & 31))
shuffle_int(a, n): for i=n-1 down to 1: j = pcg32_next() % (i+1); swap(a[i], a[j])   (Fisher-Yates)
```
Seed: `state=0, inc=1`, then one `pcg32_next` call discarded. Fixed seed (changelog 7bfdbad8); each solPGS call (each island) reseeds identically.

### 3.8 `dualState(d, state, ne, nf, nefc, efclist)` → nactive
Dual-space classification (opposite conventions to primal for friction):
- equality: `QUADRATIC`; nactive starts at `ne+nf`.
- friction: `force <= −floss → LINEARPOS`, `force >= floss → LINEARNEG`, else `QUADRATIC`.
- non-elliptic limit/contact: `force <= 0 → SATISFIED`, else `QUADRATIC` (nactive++).
- elliptic: `f0 = force[i]/mu`, `fj = force[i+j]/friction[j-1]`; `N=f0`, `T=||f1..||`; `mu*N >= T → SATISFIED`; `N + mu*T <= 0 → QUADRATIC` (nactive += dim); else `CONE` (nactive += dim); replicate state over dim rows; advance `c += dim−1`.

`dualStateChange`: saves `oldstate[c] = state[i]`, recomputes, counts `nchange = #(oldstate != state)`.

### 3.9 Cone projections
- `projectEllipsoid(friction, normal, mu, dim, feasible)`: `s = Σ f_j²/mu_j²`; if `!feasible || s > normal²`: `f_j *= sqrt(normal²/max(mjMINVAL,s))`.
- `projectCone(force, mu, dim, type)`: elliptic: `force[0]<0 → zero all dim`, else `projectEllipsoid(force+1, force[0], mu, dim, feasible=1)`; others: `force[0] = max(force[0],0)`. (Note `mu` here is `contact.friction`, the unregularized friction coefficients.)
- `solveQCQP(force, i, dim, Ac, bc, mu)`: dim 3→`mju_QCQP2`, 4→`mju_QCQP3`, else (5, i.e. condim 6 has dim−1=5) `mju_QCQP(n=dim−1)`, with radius `r = force[i]` (normal force) and scaling `d = mu = contact.friction`. If constrained (`flg_active`), `projectEllipsoid(v, force[i], mu, dim, feasible=0)` puts it exactly on the boundary. `force[i+1..] = v`.

---

## 4. PGS — `solPGS(m, d, island, ne, nf, nefc, efclist, maxiter)`

Problem: $\min_f \tfrac12 f'ARf + f'b$ s.t. friction box $|f_i|\le floss_i$, $f_i\ge0$ for limits/frictionless/pyramidal, elliptic cone $\sum_j f_j^2/\mu_j^2 \le f_0^2$, $f_0\ge 0$. `efclist = map_iefc2efc + island_iefcadr[island]` for islands, NULL monolithic. Works in place on `d->efc_force`, `d->efc_state` (global indexing).

Allocations (stack): `ARinv[nefc]`, `oldstate[nefc]`, `blockstart[nefc]`, and if Nesterov: `force_prev[nefc]`, `force_momentum[nefc]` (list-indexed `c`).

```
nesterov = (mj_nesterov_momentum != 0)        // thread-local global, default 1
if nesterov: force_prev[c] = force[efclist[c]]
scale = inertiaScale(island); ARdiaginv(ARinv, flg_subR=0)
dualState(efc_state)
blocks: c=0; while c<nefc: blockstart[nb++]=c; c += (type==ELLIPTIC ? contact.dim : 1)
rng seeded (§3.7)
iter = 0; nesterov_k = 0
while iter < maxiter:
  if nesterov:
     beta = (iter>0) ? (nesterov_k − 1)/(nesterov_k + 2) : 0      // doubles
     if beta > 0:
        for c: f_save = force[i]; force[i] += beta*(force[i] − force_prev[c]); force_prev[c] = f_save
        for c in [ne, ne+nf): force[i] = clip(force[i], −floss, floss)
        for c in [ne+nf, nefc) by block: projectCone(force+i, contact.friction (elliptic) or NULL, dim, type)
     else: force_prev[c] = force[i]
     force_momentum[c] = force[i]
  improvement = 0
  shuffle_int(blockstart, nb)            // in place, cumulative across iterations
  for bi in 0..nb-1:
     c = blockstart[bi]; i = efclist[c]; dim = elliptic ? contact.dim : 1
     residual(res, i, dim, flg_subR=0); oldforce = force[i..i+dim)
     if not elliptic:
        force[i] −= res[0]*ARinv[c]
        if ne<=c<ne+nf: clamp to [−floss, floss]     (if/else-if form)
        elif c>=ne+nf:  force[i] = max(force[i], 0)
     else (elliptic):
        mu = contact.friction
        Athis = extractBlock(i, dim, 0)
        // normal / ray update
        if force[i] < mjMINVAL:
            force[i] −= res[0]*ARinv[c]; force[i]=max(force[i],0); force[i+1..]=0
        else:
            v = force[i..i+dim); denom = v'·Athis·v
            if denom >= mjMINVAL:
               x = −(v·res)/denom
               if force[i] + x*v[0] < 0: x = −v[0]/force[i]          // = −1: zero the whole block
               force[i+j] += x*v[j]
        // friction update with normal fixed
        for j<dim−1: Ac[j,:] = Athis[j+1, 1..dim)
                     bc[j] = res[j+1] − Ac[j,:]·oldforce[1..dim) + Athis[(j+1)*dim]*(force[i]−oldforce[0])
        if force[i] < mjMINVAL: force[i+1..] = 0
        else: solveQCQP(force, i, dim, Ac, bc, mu)
     if dim==1: Athis[0] = 1/ARinv[c]
     improvement −= costChange(Athis, force+i, oldforce, res, dim)
  nactive = dualStateChange(efc_state, oldstate, ..., &nchange)
  improvement *= scale
  saveStats(island_stat=max(0,island), iter, improvement, 0, 0, nactive, nchange, 0, 0)
  if nesterov:
     restart = false
     if iter>0: dot = Σ_c (force[i] − force_momentum[c])*(force_momentum[c] − force_prev[c]); restart = dot<0
     nesterov_k = restart ? 0 : nesterov_k+1
  iter++
  if improvement < opt.tolerance: break
finalize: if island_stat < mjNISLAND:
    solver_niter[island_stat] += iter
    solver_nnz[island_stat] = sparse ? Σ_c AR_rownnz[i] : nefc*nefc
```

Notes:
- `bc` derivation: the residual already contains `Ac·oldforce_t + A_{t,0}·oldforce_0`; the friction subproblem uses the *new* normal, hence the correction terms.
- The elliptic `costChange` uses the full dim×dim `Athis` and full `res` (with `oldforce` of all dim components).
- Termination has no `improvement>0` guard; tiny negative improvements (rejected steps contribute 0, accepted ones ≤ 1e-10 positive change) end iteration.
- Nesterov extrapolated forces are evaluated by the sweep; `costChange` restores to `oldforce` = the extrapolated value, not the pre-extrapolation value.
- `mj_solPGS` / `mj_solPGS_island` do **not** call `dualFinish`; caller does (`fwdConstraint`).

---

## 5. NoSlip — `solNoSlip(m, d, island, ne, nf, nefc, efclist, maxiter)`

Post-processing PGS over **friction-type rows only** (dof/tendon friction loss and contact friction components) using the *unregularized* operator `A = AR − diag(R)`, keeping equality/limit/normal forces fixed.

```
ARinv = ARdiaginv(flg_subR=1)   // 1/max(mjMINVAL, AR_ii − R_i)
dualState(efc_state); scale = inertiaScale(island)
while iter < maxiter:
  improvement = 0
  if iter == 0: improvement += Σ_c ½ force[i]² R[i]       // removes regularizer cost from baseline
  // dry friction rows
  for c in [ne, ne+nf):
     residual(res, i, 1, flg_subR=1); old = force[i]
     force[i] −= res*ARinv[c]; clamp to [−floss, floss]
     δ = force−old; improvement −= ½ δ²/ARinv[c] + δ*res
  // contact friction
  for c in [ne+nf, nefc):
     pyramidal (type==CONTACT_PYRAMIDAL): dim = contact.dim
        for j = i, i+2, ..., i+2(dim−1)−2:                  // pairs of opposing edges (j, j+1)
           residual(res, j, 2, 1); old = force[j..j+2)
           Ac = extractBlock(j, 2, flg_subR=1)              // diag clamped ≥1e-10
           bc[k] = res[k] − Ac[k,:]·old
           mid = ½(f_j + f_{j+1}); y = ½(f_j − f_{j+1})
           K1 = Ac00 + Ac11 − Ac01 − Ac10;  K0 = mid*(Ac00 − Ac11) + bc0 − bc1
           if K1 < mjMINVAL: f_j = f_{j+1} = mid
           else: y = −K0/K1
                 y < −mid: (0, 2mid);  y > mid: (2mid, 0);  else (mid+y, mid−y)
           improvement −= costChange(Ac, force+j, old, res, 2)
        c += 2(dim−1) − 1
     elliptic: dim = contact.dim
        residual(res, i+1, dim−1, 1); old = force[i+1..i+dim)
        Ac = extractBlock(i+1, dim−1, 1); bc = res − Ac·old
        if force[i] < mjMINVAL: zero friction  else solveQCQP(force, i, dim, Ac, bc, mu=contact.friction)
        improvement −= costChange(Ac, force+i+1, old, res, dim−1)
        c += dim−1
     (limits / frictionless: skipped)
  nactive = dualStateChange(...)
  improvement *= scale
  if island_stat < mjNISLAND: saveStats(island_stat, iter + solver_niter[island_stat], improvement, 0,0,nactive,nchange,0,0)
  iter++
  if improvement < opt.noslip_tolerance: break
if island_stat < mjNISLAND: solver_niter[island_stat] += iter
```
Pyramidal pair math: with `f_j = mid+y`, `f_{j+1} = mid−y` the pair's normal-force contribution (sum) is preserved; minimize the 1-D quadratic in y on `[−mid, mid]`. NoSlip stats are appended after the main solver's iterations in the same `d->solver` row.

The pyramidal loop uses monolithic indices `j` directly (rows of a contact are contiguous also in island order because maps preserve order within category).

---

## 6. Primal solvers (CG & Newton) — `mj_solPrimal(m, d, island, maxiter, flg_Newton)`

### 6.1 Problem
$$ \min_a\; f(a) = \underbrace{\tfrac12 (a-a_0)' \tilde M (a-a_0)}_{\text{Gauss}} + s(Ja - a_{ref}) $$
with $a_0$ = `qacc_smooth`, $\tilde M = M$ (or the effective metric $M + hD + h^2K$ when `efm_active`), $s$ the piecewise-quadratic constraint cost of §2. Gradient $g = \tilde M a - q_{smooth} - J'f$ (with $f=-\partial s/\partial(Ja)$). Implementation uses `Ma = M a`, so Gauss cost $=\tfrac12 (Ma - q_{smooth})'(a - a_0)$ (valid because $Ma_0 = q_{smooth}$).

### 6.2 Context (`mjPrimalContext`) and `PrimalPointers`
Monolithic: sizes `nv, ne=d->ne, nf=d->nf, nefc, nJ=d->nJ`; pointers to `qfrc_smooth, qfrc_constraint, qacc_smooth, qacc, M (CSR) M_rownnz/rowadr/colind, qLD, qLDiagInv, efc_D, efc_R, efc_frictionloss, efc_aref, efc_id, efc_type, efc_force, efc_state, efc_J` (+ sparse J structure incl. `efc_J_rowsuper`).
Island: sizes `island_nv/ne/nf/nefc`; dof arrays `ifrc_smooth, ifrc_constraint, iacc_smooth, iacc` (+idofadr); efc arrays `iefc_*` (+iefcadr). M/qLD/J island blocks are carved and populated in `PrimalAllocate`. `cone_fold = -1`. `is_elliptic = (opt.cone == mjCONE_ELLIPTIC)`.

### 6.3 `PrimalAllocate(m, d, ctx, flg_Newton)` (stack, two-pass "carve")
Pass 0 totals sizes, pass 1 hands out pointers in the same order (so arrays are contiguous in a `mjtNum` block and an `int` block).
- Island matrices: `M_rownnz[nv], M_rowadr[nv], M_colind[nC], M[nC], qLD[nC], qLDiagInv[nv]` (nC = Σ M_rownnz over island dofs); `J[nJ]` (+`J_rownnz/rowadr/rowsuper[nefc], J_colind[nJ]` sparse; dense `nJ = nefc*nv`).
- Common: `Jaref[nefc], Jv[nefc], Ma[nv], Mv[nv], grad[nv], Mgrad[nv], search[nv], quad[3*nefc]`, sparse `JT[nJ]`.
- Newton: `D[nefc], cholupd[nv]`, elliptic `LTJ[6*nv]`; sparse `cholscratch[nv]`; dense `L[nv*nv]` and elliptic `Lcone[nv*nv]` (`nL = nv*nv`).
- CG: `gradold, Mgradold, graddif, Mgraddif [nv]`.
- Metric-fold blocks (`effFoldWanted`: `efm_active && island<0 && !elliptic && (IPC || (!Newton && mj_effFoldMonolithic(m,d,mjSOL_CG)))`) — `ep->L[9*nefmdof], S, U, Uadr, Sadr, F`; may be dropped with `mjWARN_CNSTRFULL` if stack is short.
- ints: `oldstate[nefc]`, sparse `JT_rownnz/rowadr/rowsuper[nv], JT_colind[nJ]`; Newton sparse `H_rowadr, H_rownnz, HT_rownnz, HT_rowadr, L_rownnz, L_rowadr, LT_rownnz, LT_rowadr [nv]`; metric Newton `S_rownnz, S_rowadr[nv], S_colind[nScap], S_val[nScap]`; metric `qfrc_eff[nv]`.

Island population: `mju_blockSparse` gathers the island block of M (values) and of `qLD` (or `qH` if `efm_active && efm_diag`), `qLDiagInv` (or `qHDiagInv`) gathered by idof; J block via `mju_block` (dense, rows `map_iefc2efc`, cols `map_idof2dof`) or `mju_blockSparse` + `mju_superSparse` (recomputes `J_rowsuper`). Since M and its LD factor are block-diagonal by tree, the island block of the global factor is the factor of the island block.

Sparse: `JT = transpose(J)` via `mju_transposeSparse` (offset by `J_rowadr[0]`); `nefc==0` → zero JT structure.

Metric (`efm_active`): `flg_metric=1`, `qfrc_smooth ← qfrc_eff = qfrc_smooth + efm_c (+ efm_ca)` (island: gathered per idof).

### 6.4 `PrimalUpdateConstraint(ctx, flg_HessianCone)`
`mj_constraintUpdate_impl(ne,nf,nefc,efc_D,efc_R,floss,Jaref,...,&cost, flg_HessianCone)`; `qfrc_constraint = J' force` (dense `mju_mulMatTVec`, sparse via JT `mju_mulMatVecSparse`); `nactive = #(state != SATISFIED)`, `ncone = #(state == CONE)` (counted per **row**, so a cone contact counts dim); `Gauss = Σ_i ½ (Ma_i − qfrc_smooth_i)(qacc_i − qacc_smooth_i)`; `quadGauss[0] = Gauss`; `cost += Gauss`.

`flg_HessianCone = flg_Newton & (opt.cone == mjCONE_ELLIPTIC)`.

### 6.5 `PrimalUpdateGrad`, `PrimalUpdateMgrad`
- `grad = Ma − qfrc_smooth − qfrc_constraint`.
- Mgrad:
  - Newton: `cholSolveSparse(L or Lcone if ncone)` / dense `mju_cholSolve(L or Lcone)`.
  - CG monolithic + metric: `PrimalMetricPrecond` (`mj_effPrecBlocks` with fold, else `mj_effPrec`).
  - CG island + metric: `mj_effPrecIsland`.
  - CG plain: `Mgrad = grad; mj_solveLD(Mgrad, qLD, qLDiagInv, nv, 1, M_rownnz, M_rowadr, M_colind, NULL)` — i.e. `M^{-1} grad`.

### 6.6 Line-search preparation — `PrimalPrepare(ctx)`
Given `search = v`, `Jv = J v`, `Mv = M̃ v`:
```
quadGauss[1] = v·Ma − qfrc_smooth·v        (two separate dots, subtracted)
quadGauss[2] = ½ v·Mv
for each row i (elliptic contacts as a block):
   DJ0 = D_i Jaref_i
   quad[3i+0] = Jaref_i*DJ0;  quad[3i+1] = Jv_i*DJ0;  quad[3i+2] = Jv_i*D_i*Jv_i
   elliptic (dim rows): accumulate rows j=1..dim−1 into the same quad (vector quadratic, bottom zone);
       UU = Σ_j (Jaref_{i+j} friction[j−1])²
       quad[3i+3] = Jaref_i*mu        (U0)     — stored into the *next rows'* slots (dim≥3 so 7 slots fit)
       quad[3i+4] = Jv_i*mu           (V0)
       quad[3i+5] = sqrt(UU)          (T0)
       quad[3i+6] = D_i/(mu²(1+mu²))  (Dm)
       i += dim−1
   quad[3i+0] *= ½; quad[3i+2] *= ½
```
So each row's polynomial is `cost(α) = quad0 + α quad1 + α² quad2` with quad0 = ½ D Jaref², quad1 = D Jaref Jv, quad2 = ½ D Jv².

### 6.7 Line-search evaluation — `PrimalEval(ctx, p)`: shifted cost `f(α) − f(0)` and derivatives

```
cost = 0; deriv = (0,0); quadTotal = (0, quadGauss[1], quadGauss[2])
for i:
  equality: quadTotal[1] += quad1; quadTotal[2] += quad2           (quad0 dropped: shift)
  friction:  start = Jaref, dir = Jv, x = start + α dir, f = floss, D, Rf = R*f
             cost += frictionCostDif(start, x, f, Rf, D)
             if −Rf < x < Rf: d0 += D x dir; d1 += D dir²
             elif x <= −Rf:  d0 += −f dir
             else:           d0 += f dir
  elliptic block:
     U0, V0, Dm from quad[3i+3..6]; N = U0 + α V0
     for j<dim−1: Uj = Jaref_{i+1+j} fr_j; Vj = Jv_{i+1+j} fr_j; Wj = Uj + α Vj
                  Tsqr += Wj²; WV += Wj Vj; Tsqr_delta += α Vj (Uj + Wj)
     T = sqrt(Tsqr)
     cost += ellipticCostDif(quad+3i, α, mu, N, T, Tsqr_delta)
     if Tsqr <= 0: if N<0: d0 += 2α quad2 + quad1; d1 += 2 quad2      (bottom)
     else if N >= mu T: (top) nothing
     else if mu N + T <= 0: bottom quad derivatives as above
     else (middle):
        det = Σ_{j<k} (Wj Vk − Wk Vj)²        // ||W∧V||², Lagrange identity
        N1 = V0; T1 = WV/T; T2 = det/(T*Tsqr)
        d0 += Dm (N − mu T)(N1 − mu T1)
        d1 += Dm [ (N1 − mu T1)² + (N − mu T)(−mu T2) ]
     i += dim−1
  other inequality (limit/frictionless/pyramidal row):
     start = Jaref; x = start + α Jv; cost0 = (start<0) ? quad0 : 0
     if x < 0: quadTotal += (quad0 − cost0, quad1, quad2)
     else: cost −= cost0
cost += α² qT2 + α qT1 + qT0;  d0 += 2α qT2 + qT1;  d1 += 2 qT2
if d1 <= 0: mju_warning("Linesearch objective is not convex"); d1 = mjMINVAL
p.cost = cost; p.deriv = (d0, d1); ctx->LSiter++
```
`frictionCost(x,f,Rf,D)`: `−Rf<x<Rf → ½Dx²`; `x<=−Rf → f(−½Rf − x)`; else `f(−½Rf + x)`.
`frictionCostDif(start,x,...)`: zone of each point (0 quad, −1 neg, +1 pos); same zone: quad `½D(x−start)(x+start)`, neg `f(start−x)`, pos `f(x−start)`; else absolute difference.

`ellipticCostDif(quad, α, mu, N, T, Tsqr_delta)` with `U0,V0,T0,Dm = quad[3..6]`:
- zone0 at α=0: `T0<=0 → (U0<0 ? 2 : 1)`; `U0 >= mu T0 → 1`; `mu U0 + T0 <= 0 → 2`; else 3. zone at α analogously with (N, T).
- 1→1: 0. 2→2: `α² q2 + α q1`. 3→3 (rationalized): `T_delta = Tsqr_delta/(T+T0)`, `r_delta = α V0 − mu T_delta`, `r0 = U0 − mu T0`, return `½ Dm r_delta (2 r0 + r_delta)`.
- 3→2: `α(α q2 + q1) + ½ Dm (mu U0 + T0)²`. 2→3: `α(α q2 + q1) − ½ Dm (mu N + T)²`. 1→2: `α² q2 + α q1 + q0`. 1→3: `½ Dm (N − mu T)²`. 3→1: `−½ Dm (U0 − mu T0)²`. 2→1: `−q0`. (The 2↔3 formulas use the identity quadratic − cone = ½Dm(muN+T)² at the same point.)

### 6.8 Exact line search — `PrimalSearch(ctx, tolerance, ls_iterations, &improvement)`
Called with `tolerance = opt.tolerance * opt.ls_tolerance`, `ls_iterations = opt.ls_iterations` (passed as mjtNum). `LSiter` counts *every* `PrimalEval` (including p0 and the first Newton point).

```
LSiter=0; LSresult=0; LSslope=1; improvement=0
snorm = ||search||; if snorm < mjMINVAL: LSresult=1; return 0
gtol = tolerance*snorm/scale            (scale = 1/trace(M)) ; slopescl = scale/snorm
Mv = M*search (mju_mulSymVecSparse) (+ mj_effMulAdd(...flg_contact=1) / Island variant if metric)
Jv = J*search
PrimalPrepare
p0.alpha=0; Eval(p0)
p1.alpha = −p0.d0/p0.d1; Eval(p1)                                   // always one Newton step
if |p1.d0| < gtol && (p1.alpha == 0 || p1.cost < 0):
    LSresult = (p1.alpha==0) ? 2 : 0; LSslope = |p1.d0|*slopescl; improvement = −p1.cost; return p1.alpha
dir = (p1.d0 < 0) ? +1 : −1
p2 = p0; p2update = 1
while p1.d0*dir <= −gtol && LSiter < ls_iterations:                  // one-sided Newton march
    p2 = p1; p1.alpha −= p1.d0/p1.d1; Eval(p1)
    if |p1.d0| < gtol && p1.cost < 0: slope/improvement from p1; return p1.alpha
if LSiter >= ls_iterations: LSresult=3; slope, improvement=−p1.cost; return p1.alpha     // could not bracket
(p2update always 1; result 6 unreachable)
p2next = p1; p1next.alpha = p1.alpha − p1.d0/p1.d1; Eval(p1next)
while LSiter < ls_iterations:                                        // bracketed phase
    pmid.alpha = ½(p1.alpha + p2.alpha); Eval(pmid)
    candidates = {p1next, p2next, pmid}     (copies)
    best = argmin cost over candidates with |d0| < gtol (first wins ties; strict <)
    if found: LSslope, improvement = −best.cost; return best.alpha
    b1 = updateBracket(&p1, candidates, &p1next); b2 = updateBracket(&p2, candidates, &p2next)
    if !b1 && !b2: LSresult = (pmid.cost<0) ? 0 : 7; slope from pmid; improvement = −pmid.cost; return pmid.alpha
if p1.cost <= p2.cost && p1.cost < 0: LSresult=4; return p1.alpha (improvement = −p1.cost)
elif p2.cost <= p1.cost && p2.cost < 0: LSresult=4; return p2.alpha
else: LSresult=5; return 0
```
`updateBracket(p, cand[3], pnext)`: for i=0..2 sequentially (p may change mid-loop): if `p.d0<0 && cand.d0<0 && p.d0 < cand.d0` → `p=cand`, flag=1; elif `p.d0>0 && cand.d0>0 && p.d0 > cand.d0` → `p=cand`, flag=2. If flag: `pnext.alpha = p.alpha − p.d0/p.d1`; Eval(pnext). Note: no sign-change bracket logic beyond "derivative closer to zero on the same side". Note no check of `LSiter` inside updateBracket, so LSiter may exceed `ls_iterations` by up to 2.

Note the returned `alpha` may be negative in principle (no constraint α ≥ 0).

### 6.9 Main driver
```
mark stack; PrimalPointers
if island>=0 && nefc==0: qacc = qacc_smooth; qfrc_constraint = 0; solver_niter[island] = 0; return
     (writes solver_niter[island] without the mjNISLAND bound check)
PrimalAllocate
Ma = M*qacc (+ metric MulAdd); Jaref = J*qacc − aref
PrimalUpdateConstraint(flg_Newton & elliptic); PrimalUpdateGrad
scale = inertiaScale(island)
IPC fold: if ctx.ep and !mj_effPrecFold(...): ep=NULL
PrimalUpdateMgrad(flg_Newton=0)                     // M^{-1} g for both solvers (certificate)
flg_gap      = max(0, ½ scale g·Mgrad) < tol
flg_gradient = scale*||g|| < tol
flg_certificate = flg_gap && (!Newton || flg_gradient);  flg_done = flg_certificate
if !done && Newton:
   if MakeHessian(d, ctx):
       FactorizeHessian(recompute=0); PrimalUpdateMgrad(Newton=1)
       flg_done = flg_gradient && max(0, ½ scale g·H^{-1}g) < tol
   else: (arena exhausted, warning already issued) Newton := CG
       flg_done = flg_gap; reuse cholupd / cholscratch|L / LTJ|fresh 2nv for gradold, Mgradold, graddif, Mgraddif
elif !done && ep && ep->F: mj_effCholFoldFactor(...); PrimalUpdateMgrad(0)
if !done: search = −Mgrad
iter = 0
while !done && iter < maxiter:
   alpha = PrimalSearch(tol*ls_tol, ls_iterations, &ls_improvement)
   if alpha == 0: break                                    // iter NOT incremented, no stats
   qacc += α search; Ma += α Mv; Jaref += α Jv            // incremental, never recomputed from scratch
   if CG: gradold = grad; Mgradold = Mgrad
   oldstate = efc_state
   PrimalUpdateConstraint(Newton & elliptic)
   if Newton: HessianIncremental(oldstate)
   PrimalUpdateGradient(Newton)                           // grad, then Mgrad (H\g or M\g)
   nchange = #(state != oldstate)
   improvement = scale*ls_improvement; gradient = scale*||grad||
   saveStats(island, iter, improvement, gradient, LSslope, nactive, nchange, LSiter, nupdate)
   decrement = Newton ? max(0, ½ scale grad·Mgrad) : 0
   iter++
   if (improvement>0 && improvement<tol) || gradient<tol || (Newton && decrement<tol): break
   direction update:
     Newton: search = −Mgrad
     CG (default, Hager-Zhang; #ifndef mjCG_PRP):
        y = grad − gradold; My = Mgrad − Mgradold
        dy = search·y
        if dy < mjMINVAL: beta = 0
        else:
          yMy = y·My; yMg = y·Mgrad; dg = search·grad
          beta_hz = (yMg − 2 (yMy/dy) dg)/dy
          eta_k = −1/max(mjMINVAL, ||search|| * min(0.01, ||grad||))
          beta = max(eta_k, beta_hz)
        search = −Mgrad + beta*search
     CG (#ifdef mjCG_PRP): beta = max(0, grad·(Mgrad−Mgradold)/max(mjMINVAL, gradold·Mgradold))
finalize (island < mjNISLAND; island_stat = max(0,island)):
   solver_niter[island_stat] += iter
   solver_nnz = (Newton && !certificate) ? (sparse ? (1 + (Lcone!=NULL))*nL + nH : nv*nv) : nJ
```
"Newton decrement" uses the H-preconditioned `Mgrad` from the *current* (post-update) factor.

### 6.10 Newton Hessian

`H = M̃ + J' diag(D_q) J + Σ_cone J_c' H_c J_c`, `D_q[i] = efc_D[i]` if state==QUADRATIC else 0. Cone contributions are added to a separate factor `Lcone` (elliptic only).

**`MakeHessian(d, ctx)`** (returns 0 on arena shortage with `mjWARN_CNSTRFULL`):
- `D[i] = (state==QUADRATIC) ? efc_D : 0`.
- metric: `MakeMetricLower` builds a lower-triangle CSR view `S` of `diag(h D + h²K)` (`efm_diag`) + `efm_K` CSR + rank-1 tendon/actuator/contact terms (`mj_effRank1Next`, each `scale*val val'`) + fluid M-pattern `efm_fluid`, merged by three-way sorted merge, duplicates summed, diagonal folded into one trailing entry (only if nonzero). Island: columns remapped via `map_dof2idof − idofadr`; a rank-1 term spanning islands → `mjERROR`.
- **Sparse**: `nH = mju_sqrMatTDSparseSymbolic(...)` (count) + `nnz(M)` + `nS` (over-allocation); allocate `H, H_colind, HT_colind`; shift `H_rowadr[r+1] += Σ_{r'≤r}(M_rownnz[r'] + S_rownnz[r'])`; symbolic again (fills colind); numeric `H = J' D J` (`mju_sqrMatTDSparseNumeric`, uses `JT_rowsuper` supernodes); `mju_addToMatSparse(H, M)`; `+S`; `HT` structure via `mju_transposeSparse` (upper triangle pattern needed); `nL = mju_cholFactorSymbolic(count)`; allocate `L_colind, LT_colind, LT_map` (+`pathcost[nv]` elliptic), `L` (+`Lcone`); symbolic fill. Elliptic: `pathcost[r] = L_rownnz[r] + pathcost[parent(r)]`, `parent(r) = L_colind[L_rowadr[r] + nnz − 2]` (largest off-diagonal column) if nnz≥2; `Lflops = Σ nnz²`.
- **Dense**: `L = J' D J` lower triangle (`mju_sqrMatTD_impl(..., flg_upper=0)`), `+= M` (`mju_addToSymSparse` lower), `+= S`.

**`FactorizeHessian(d, ctx, flg_recompute)`**: optionally recompute D and H numerically (same ops, sparsity fixed); sparse `mju_cholFactorNumeric(L, nv, mjMINVAL, ...)`; dense `mju_cholFactor(L, nv, mjMINVAL)`; if `ncone`: `cone_fold = −1; HessianCone`. `nupdate = nefc`.

**`HessianIncremental(d, ctx, oldstate)`**: `nupdate = 0`; for each row i in order: any change → `cone_fold=−1`; if became QUADRATIC → rank-1 update (+), if left QUADRATIC → downdate (−) with `cholupd = sqrt(efc_D[i]) * J[i,:]` (sparse `mju_cholUpdateSparse` on row pattern; dense `mju_cholUpdate`); `nupdate++`; if returned rank < nv → `FactorizeHessian(recompute=1)` and **return** (cones re-added inside). After loop: if `ncone` → `HessianCone`.

**`HessianCone`**: sparse: decide `cone_fold` (cached; `HessianConeFoldFaster`) → `HessianConeFolded` else `HessianConeUpdate`.
- `HessianConeUpdate`: `Lcone = L`; for each CONE contact (dim rows at i): `local = chol(contact.H)` (dense dim×dim, `mju_cholFactor(local, dim, mjMINVAL)`, lower L_c with H_c = L_c L_c'); `LTJ[c,:] = Σ_{r≥c} local[r*dim+c] * J[i+r,:]` (i.e. rows of `L_c' J_c`); for r<dim: rank-1 update `Lcone += LTJ[r] LTJ[r]'` (sparse: uses row-i's column pattern, assumed identical for all dim rows of the contact); `nupdate += dim`.
- `HessianConeFoldFaster`: `U = Σ_cone dim * pathcost[lastcol]`, `S = Σ_quad nnz² + Σ_cone dim*nnz²`; fold iff `10U > 3(Lflops + 2S)`.
- `HessianConeFolded` (sparse): allocate once per solve `Jmod, JTmod, Dmod, Hcone, JTmod_*` (fallback to update path on shortage); `Jmod = J`, `Dmod = D_q`; cone rows replaced by `L_c' J_c` and `Dmod=1`; `Hcone = Jmod' Dmod Jmod + M + S`; `Lcone = cholNumeric(Hcone)` with the same pattern as L.

Gauss/constraint gradient is solved against `Lcone` when `ncone>0`, else `L`.

---

## 7. Dense and sparse Cholesky (`engine_util_solve.c`)

### 7.1 Dense (row-major, lower triangle, `A = L L'`)
`mju_cholFactor(mat, n, mindiag)`: for j: `tmp = A_jj − dot(L[j,0:j], L[j,0:j])`; if `tmp < mindiag`: `tmp=mindiag`, rank−−, and column j below diagonal set to 0; `L_jj = sqrt(tmp)`; else `L_ij = (A_ij − dot(L[i,0:j], L[j,0:j])) / L_jj` (multiplication by `1/L_jj`). Returns rank.
`mju_cholSolve`: forward `res_i = (res_i − dot(L[i,0:i],res))/L_ii`; backward `res_i /= L_ii; res[0:i] −= res_i L[i,0:i]` (skip if res_i==0).
`mju_cholUpdate(L, x, n, plus)` (x overwritten): for k with `x_k≠0`: `tmp = L_kk² ± x_k²`, clamp `<mjMINVAL → mjMINVAL`, rank−−; `r=sqrt(tmp)`, `c=r/L_kk`, `cinv=1/c`, `s=x_k/L_kk`; `L_kk=r`; for i>k: `m=(L_ik ± s x_i)*cinv; L_ik=m; x_i = c x_i − s m`.

### 7.2 Sparse reverse Cholesky (`A = L' L`, L lower CSR, eliminating from the last row up)
Chosen because for tree-structured M (CSR with ancestor columns) the reverse order produces no fill.
- `mju_cholFactorSparse` (legacy in-place with fill via `mju_combineSparse`): for r = n−1..0: diag clamp to mindiag (rank−−), `L_rr = sqrt`, scale off-diagonals by `1/L_rr`, then for each off-diag column c of row r: `row_c −= L_rc * row_r[0..c]`.
- `mju_cholFactorSymbolic` (ldl_symbolic-based, reads upper triangle via `HT`): rows r = n−1..0; for each upper column i>r of row r, walk elimination tree from i upward until a node flagged for r; each visited node gets one nonzero (count phase: `L_rownnz[i]++`, `LT_rownnz[r]++`; fill phase writes L columns right-to-left with per-row cursors and LT entries with `LT_map` = index into L). Diagonal stored last in each L row, first in each LT row. `parent[i]` set to r on first contact.
- `mju_cholFactorNumeric(L, n, mindiag, ..., H, ..., scratch)`: right-looking with a dense accumulator. For r from n−1 downward with step ns:
  - leaf fast path (`LT_rownnz[r]==1` and H row has no fill): `L_rr = sqrt(max-clamped H_rr)`, `L_r,: = H_r,:/L_rr` (or zero if deficient).
  - supernode size ns∈{1,2,3}: rows r, r−1, r−2 sharing the same off-diagonal columns and descendant set (`L_colind[L_nnz−1−ns]==r−ns && L_rownnz[r−ns]==L_nnz−ns && LT_rownnz[r−ns]==LT_nnz+ns`).
  - `cholFactorSupernode`: initialize rows from H (copy or scatter via dense); for each descendant row c>r in LT[r] (in LT order), batching up to 4 (3 when ns==3) descendants with identical patterns: suffix fast path `row[off+i] −= Σ_b s[t][b]*p_b[i]` (b ascending, `s[t][b] = L[idx_b − t]`), else scatter into dense; flush dense; then factor rows s=0..ns−1: diag clamp (rank_dec), `L_rr=sqrt`, scale with `1/L_rr`, eliminate inside the supernode.
  - Bit-exact reproduction requires the same batching/ordering; mathematically it is the standard reverse Cholesky.
- `mju_cholSolveSparse`: first back pass `res ← L^{-T} res` (i = n−1..0; fused 2-/3-row supernodes; `res_i /= L_ii; res[cols] −= L_ij res_i`), then forward `res ← L^{-1} res` (`res_i = (res_i − Σ L_ij res_j)/L_ii`, fused supernodes). Skips rows with `res_i==0` in the back pass.
- `mju_cholUpdateSparse(L, x, n, plus, ..., x_nnz, x_ind, scratch)`: x scattered to dense; rows from `x_ind[last]` down to `min_row` (extended as rows reach earlier columns): skip `x_row==0`; `tmp = diag² ± x_row²` (clamp mjMINVAL, rank−−); `r=sqrt(tmp)`; `c = diag/r`, `s = −x_row/r`, `s_signed = plus ? −s : s`; for off-diagonals j: `L_rj ← c L_rj + s_signed dense_j`, `dense_j ← s L_rj(old) + c dense_j`. Pattern of L must already contain x's pattern (true for J rows of a quadratic constraint since H's pattern includes all J rows).
- `mju_cholFactorSymbolicBlocked`: same as symbolic, for matrices of dense bs×bs blocks (used by flex metric code), produces identical L arrays.

### 7.3 Band-dense Cholesky (`mju_cholFactorBand`, `mju_cholSolveBand`)
Layout: first `nsparse=ntotal−ndense` rows store `nband` entries left of and including the diagonal; last ndense rows stored as full rows of length ntotal. Factor adds `diagadd + diagmul*A_jj` to the diagonal; returns min pre-sqrt diagonal, or 0 if any `< mjMINVAL` (stops early). Not used by the constraint solver (used by derivative/implicit code); `mju_bandDiag(i)` = `i*nband + nband − 1` (sparse part) else `nsparse*nband + (i−nsparse)*ntotal + i`.

### 7.4 QCQP (`mju_QCQP2/3/QCQP`)
Solve $\min \tfrac12 x'Ax + x'b$ s.t. $\sum (x_i/d_i)^2 \le r^2$. Scale `b_i *= d_i`, `A_ij *= d_i d_j` so the constraint is `x'x ≤ r²`. Newton on λ (≤20 iterations, λ₀=0):
- 2D: `det = (A11+λ)(A22+λ) − A12²`; `det < 1e-10 → res=0, return 0`; `P = inv`; `v = −P b`; `val = v'v − r²`; `val < 1e-10 → break`; `deriv = −2 v'Pv`; `δ = −val/deriv`; `δ < 1e-10 → break`; `λ += δ`.
- 3D: adjugate formulas (unscaled P entries, `det = (A11+λ)P11 + A12P12 + A13P13`), same thresholds.
- nD (n≤5): `Ala = A + λI`, `mju_cholFactor(Ala, n, 1e-10) < n → res=0, return 0`; `res = −Ala^{-1} b`; `val = res'res − r²`; `deriv = −2 res'·(Ala^{-1} res)`.
- undo scaling `res_i *= d_i`; return `λ != 0`.
Note: A uses only `Ain[0],Ain[1],Ain[3]` (2D) / upper entries (3D); asymmetric input would be read from the upper triangle.

### 7.5 `mju_boxQP` (not part of the constraint solver)
Projected-Newton box QP with Armijo backtracking, defaults `maxiter=100, mingrad=1e-16, backtrack=0.5, minstep=1e-22, armijo=0.1`, incremental Cholesky insert/delete (`cholInsert/cholDelete`) when `nclamp < oldnfree && nclamp + 2 nunclamp <= max(2, nfree/8)`. Used by actuator/MPC utilities; listed for completeness.

---

## 8. Islands — `engine_island.c`

### 8.1 Disjoint-set primitives
- `mj_dsuRoot(parent, tree)`: follow parents to root, then path compression (every node on the path set to root).
- `mj_dsuMerge(parent, t1, t2)`: −1 denotes a static endpoint (`both −1 → mjERROR`); a −1 is replaced by the other; activate (`parent[t]==−1 → parent[t]=t`); if `parent[t1]==parent[t2]` return; roots r1,r2; **link larger root to smaller root**. Hence each component's root is its minimum tree index.
- `mj_dsuAssign(island, parent, tree_dofnum, ntree, &nidof)`: trees ascending; inactive (`parent==−1`) → `island=−1`; root → new id `nisland++`; non-root: `parent[t] = parent[parent[t]]`, `island[t] = island[parent[t]]` (works because roots are smaller and already labeled; after compression the grandparent is the root). `nidof += tree_dofnum` for active trees. Island ids are thus ordered by ascending minimum tree index.
- `mj_floodFill` (legacy, adjacency-matrix DFS): kept as utility; not used by `mj_island` any more.

### 8.2 Tree incidence of each constraint — `unionConstraintTrees(m, d, parent, efc_tree, &err)`
For each row i: if same `(efc_type, efc_id)` as previous row and not a flex equality (`mjEQ_FLEX`/`mjEQ_FLEXSTRAIN`), copy `efc_tree[i] = efc_tree[i−1]` and skip.
- Contacts (frictionless/pyramidal/elliptic): `unionContactTrees`: geom-geom → trees of both geom bodies (static −1 handled; if first is static swap); flex → bodies of both sides via `mj_contactBodyWeight` (up to 729 bodies), first dynamic tree activated, others unioned to it; `efc_tree = first tree`; if none: error "contact %d is between two static bodies".
- Others via `treeIterInit/treeNext`:
  - `FRICTION_DOF`: `dof_treeid[id]`; `LIMIT_JOINT`: `dof_treeid[jnt_dofadr[id]]`; equality CONNECT/WELD: trees of bodies (site → `site_bodyid`), error if both static.
  - Generic scan (other equality, tendon friction/limit, etc.): sparse → iterate `efc_J_colind` row, return each new `dof_treeid` different from the previous one; dense → scan nonzero `J[j]`, after a hit skip to the end of that tree's dof block.
  - `efc_tree[i] = tree1>=0 ? tree1 : tree2`; error "constraint %d is between two static bodies" / "no tree found". Single tree → `mj_dsuMerge(t1, −1)` (activate); multiple → chain-merge consecutive trees.
- Additional metric couplings (no efc rows):
  - every flex with `mj_effFlexPossible`: union awake trees of all vertices (nodes if interpolated), star around the first.
  - if `mj_isMetric(m)`: passive flex contacts (`exclude == 4`): union awake trees of both sides' bodies.
  - if `mj_effCouplings(m)`: tendons with `tendon_treenum ≥ 2 && mj_effTendonPossible` → `unionTendonTrees` (awake trees in `ten_J_colind`); actuators with `mj_effActuatorPossible`: tendon transmission → tendon union; SITE/SLIDERCRANK → trees of `site_bodyid[trnid[0]]` and `trnid[1]` (if ≥0); SO3 with `trnid[1]≥0`; union if both awake and distinct.

### 8.3 `mj_island(m, d)`
```
if mjDISABLED(mjDSBL_ISLAND) || (nefc==0 && !(SLEEP enabled && mj_isMetric(m) && mj_effFlexAny(m))):
    nisland = nidof = 0; return
parent[ntree] = −1; unionConstraintTrees (errors → mjERROR)
nisland = mj_dsuAssign(tree_island, parent, ...)
if nisland == 0: nidof=0; return
allocate MJDATA_ARENA_POINTERS_ISLAND on arena (failure: mjWARN_CNSTRFULL, mj_clearEfc, return)
tree_island (copy); island_ntree (counts); island_itreeadr (prefix sums)
map_itree2tree: islands' trees in ascending tree order, followed by unconstrained trees
dof_island[dof] = tree_island[dof_treeid[dof]]; island_nv counts; island_idofadr prefix sums
map_dof2idof / map_idof2dof: dofs ascending; constrained dof → island_idofadr[isl] + running count;
                             unconstrained → nidof + running count  (so idof ≥ nidof ⇒ unconstrained)
island_dofadr[i] = map_idof2dof[island_idofadr[i]]
efc_island[i] = tree_island[efc_tree[i]]; island_nefc/ne/nf counts (ne: EQUALITY; nf: FRICTION_DOF/TENDON)
island_iefcadr prefix sums
map_efc2iefc / map_iefc2efc: efc ascending within each island; copy iefc_type, iefc_id,
     iefc_frictionloss, iefc_D, iefc_R
```
Within an island the iefc order is the global efc order restricted to that island; because global efc is sorted by category (eq, friction, then rest), the island's rows also are `[ne | nf | rest]`, which the island solvers rely on (`island_ne`, `island_nf`).

Consistency checks (`mjERROR` "miscount") on counts.

### 8.4 Per-island solving
- Each island solves an independent problem with its own `scale = 1/trace(M_island)`, own iteration count (`solver_niter[island]`, stats in `d->solver[island*mjNSOLVER...]` for island < 20; islands ≥ 20 are solved but no stats).
- Because termination, `scale`, the PGS shuffle, and line search are per island, **island and monolithic solves give different results** (not just reordered).
- Islands are disabled (monolithic path) for IPC and when CG folds into a monolithic flex factor (`mj_effFoldMonolithic`).

---

## 9. Programmatic optimizations (C)

- All solver scratch lives on the mjData stack (`mj_markStack/freeStack`); `PrimalAllocate` uses a two-pass carve so one `mjtNum` and one `int` block are allocated. Newton checks available stack bytes before each allocation stage and falls back to CG (with `mjWARN_CNSTRFULL`) instead of failing.
- Primal solvers never recompute `Ma`, `Jaref` from scratch after initialization: updated with `α Mv`, `α Jv` (round-off accumulates; reproduce exactly).
- `Mv`, `Jv` computed once per line search; every `PrimalEval` is O(nefc) scalar work.
- Newton: rank-1 Cholesky updates/downdates for state changes (fallback to full refactor on rank loss); cone contributions via rank-dim updates or one folded refactorization chosen by a flop model cached until state change.
- Sparse matrices: CSR with `rowsuper` supernode hints for J; H pattern computed symbolically once per solve; reverse Cholesky symbolic once per solve, numeric per refactor; supernodal numeric kernels (ns ≤ 3, batch ≤ 4, AVX-friendly).
- PGS: per-block residual via sparse row dot of `efc_AR`; precomputed `1/AR_ii`; Nesterov extrapolation with O'Donoghue–Candès gradient restart; randomized block order.
- Islands: threaded via `mju_dispatch` (thread pool), only for the main solver; noslip per island serial.
- Certificate early-out: with good warmstart, no Hessian is built and zero iterations run.

---

## 10. GPU (MuJoCo Warp) implementation

### 10.1 Scope and supported features
- `types.SolverType` = {CG, NEWTON} only. **PGS is not supported** (`put_model` raises `NotImplementedError("mjSOL_PGS is unsupported")`). NoSlip **is** implemented (`solver.noslip`), but not with the discrete integrator.
- No per-island solving. Islands (`island.py`) are computed only when sleeping is enabled (`forward.py`: `if sleep_enabled: island.island(m,d)`) and mapping arrays (`compute_island_mapping`) after the solve, for sleep bookkeeping/compatibility. Sleep requires the Newton solver.
- Precision: Warp `float` = float32 by default. `put_model` sets `opt.tolerance = max(opt.tolerance, 1e-6)`. Many options are per-world arrays indexed `worldid % shape[0]` (`opt.tolerance`, `ls_tolerance`, `impratio_invsqrt`, `noslip_tolerance`).
- `is_sparse` in Warp: `jacobian==AUTO → nv > 32` (C uses `nv >= 60`); explicit settings follow `mj_isSparse`.
- Data layout: per-world 2-D/3-D arrays `[nworld, ...]`; constraint rows padded to `njmax` (capacity), `nefc` clamped `min(njmax, nefc)`; sparse J stored as `efc.J[worldid, 0, sparseid]` with `J_rownnz/rowadr [nworld, njmax]`, `J_colind[worldid, 0, sparseid]`; dense J `efc.J[worldid, efcid, dof]` padded to `nv_pad`. Contacts are global across worlds (`contact.worldid`, `nacon`, `naconmax`); `contact.efc_address[conid, j]` gives row of component j (−1 if dropped past njmax). `impratio_invsqrt = 1/sqrt(max(impratio, mjMINVAL))`; Warp computes `mu = friction[0]*impratio_invsqrt` on the fly (C uses `con->mu = friction[0]*sqrt(R1/R0)` — same up to rounding).
- `nv_pad`: `round_up(nv + augment, 16)` if `is_sparse or nv > 32` (augment = 1 iff Newton and nv>32), else `round_up(nv, 4)`. `TILE_SIZE_JTDAJ_DENSE = TILE_SIZE_JTDAJ_SPARSE = 16`, `_BLOCK_CHOLESKY_DIM = 32`.

### 10.2 `solve(m, d)` / `_solve(m, d, ctx)`
```
if SLEEP: update_active_dofs; if integrator != DISCRETE: solve_compact (dense Newton in compacted dof space,
          size nvmax_pad); noslip; compute_island_mapping; return
if njmax==0 or nv==0: qacc = qacc_smooth; solver_niter = 0
else: ctx = _create_solver_context; _solve; if noslip_iterations>0: noslip
```
`_solve`:
1. Discrete integrator: `qfrc_smooth ← qfrc_smooth + efm_c + efm_ca` (zero for sleeping dofs).
2. `_solve_init_dof(warmstart, sparse, run_qacc_smooth)` per (world, dof): if `skip_qacc_smooth` (opt.run_qacc_smooth false, Newton, warmstart on, no noslip, iterations>0, njmax>0) → `qacc = qacc_warmstart`; elif warmstart → `qacc = (nefc>0) ? qacc_warmstart : qacc_smooth`; else `qacc_smooth`. Sparse: `qfrc_constraint = 0` if nefc==0.
   **Difference:** no comparison of warmstart cost vs `qacc_smooth` cost (C and MJX do compare).
3. `init_context(grad=False)`: inertia scale `1/max(MJ_MINVAL, Σ M_diag)` (full model even in compact solve); `solver_niter=0, done=False, search_dot=0`; `Jaref = J qacc − aref` (sparse: one thread per row; dense: `dofs_per_thread = 20 if nv>50 else 50`, atomics when >1 thread per row, so FP order is non-deterministic unless `deterministic` mode); `efc.Ma = M qacc` (`support.mul_m`, or `derivative.eff_mul_m` for discrete); `_update_constraint`.
4. `_update_gradient` (grad, Mgrad/H factor, search for Newton).
5. CG: `_solve_init_search_cg_tiled`: `search = −Mgrad`, `search_dot`, `prev_grad = grad`, `prev_Mgrad = Mgrad`; if `max(0, ½ scale grad·Mgrad) < tol` → `done=True`, `nsolving −= 1` (CG certificate; Newton has **no** zero-iteration certificate).
6. Iterate: with `opt.graph_conditional` (default True) `wp.capture_while(nsolving, _solver_iteration)` (CUDA graph conditional loop runs until all worlds done); else a fixed Python loop of `opt.iterations` launches (done worlds early-return inside kernels).
7. Pyramidal Newton (incremental mode): recover `qfrc_constraint = Ma − qfrc_smooth − grad_scale*grad`.

`_solver_iteration`: `_linesearch` → (incremental: zero change counters) → `_update_constraint(track_changes, stable_fast)` → `_update_gradient_incremental` (Newton & pyramidal) or `_update_gradient` → CG: `_solve_beta_finalize_tiled` + `_solve_search_update_cg_tiled`; Newton: `_solve_done`.

### 10.3 Constraint update (`_update_constraint_efc`, per (world, efc row))
Same zones/forces as `mj_constraintUpdate_impl`, evaluated independently per row (elliptic rows each recompute N, T from all rows of the contact):
- friction: `rf = safe_div(frictionloss, D)` (C: `R*floss`); `jaref <= −rf → force=+f, LINEARNEG`; `jaref >= rf → −f, LINEARPOS`; else `−D jaref`, QUADRATIC.
- elliptic: `N = jaref0*mu`, `T = sqrt(TT)` (`TT = Σ (jaref_j fr_j)²`, 0 if ≤0); top/bottom/middle as C; middle: `dm = safe_div(D0, mu²(1+mu²))`, normal force `−dm (N−mu T) mu`, tangent `−safe_div(f_n, T) * jaref_j fr_j²`. Contacts with `conid >= nacon` or any `efc_address < 0` → row untouched.
- No cone Hessian stored (Hessian terms recomputed in the Hessian kernels from Jaref).
- `qfrc_constraint = J' force`: dense per (world, dof) loop over rows (deterministic order); sparse per row with `atomic_add` (non-deterministic order).
- Tracking (pyramidal Newton): `quad_changed_ids/count` (QUADRATIC flag flips, atomically appended — order nondeterministic), `state_changed_count` (any state change, + 1 if the previous linesearch flagged `ls_exhausted`).
- Cost is never computed (Warp only needs shifted line-search costs).

### 10.4 Gradient / Hessian (`_update_gradient`)
- `grad = Ma − qfrc_smooth − qfrc_constraint`; `grad_dot = Σ grad²` (CG tiled reduction; Newton atomic adds).
- CG: `Mgrad = M^{-1} grad` (`smooth.solve_m`; discrete: `eff_prec` or `solve_LD(qHLD)`).
- Newton: **dense** H always (`ctx.h [nworld, nv_pad, nv_pad]`, upper triangle used):
  - Dense J: `_update_gradient_JTDAJ_dense_tiled`: M CSR scattered into upper triangle of a shared tile (`(col, i)`), then `H += (J_k' diag(D_k·[state==QUAD]·[row<nefc])) J_k` tile-matmul over 16-row tiles of constraints. Padded dofs get identity (`_padding_h`, nv>32).
  - Sparse J: `_update_gradient_init_h_sparse` (M entries + DOF-friction diagonal D when quadratic; padded rows identity) then `_JTDACJ_sparse`: groups of consecutive rows with identical sparse support (`efc.jtdaj_adr/nrow/nblock`, built in `constraint.py`), 32 lanes per group, decode upper-triangle entry index with `sqrt(8e+1)`; quadratic rows add `Σ_member J_r D J_c`; elliptic CONE groups add the cone curvature (below); `atomic_add` into `h[min(r,c), max(r,c)]` (nondeterministic summation order).
  - Elliptic cone curvature (dense path `_update_gradient_JTCJ_dense` per (contact, upper-triangle dof pair); sparse path inside `_JTDACJ_sparse`), written in contracted form without forming the dim×dim H_c:
    ```
    dm = D0/(mu²(1+mu²)); n = Jaref0*mu; u_j = Jaref_j fr_j; t = max(sqrt(Σu_j²), MJ_MINVAL)
    z0a = mu J0[a]; z_j[a] = fr_j J_j[a]; p[a] = Σ_j (u_j/t) z_j[a]; tdot = Σ_j z_j[a] z_j[b]
    h_ab = dm [ z0a z0b − mu (z0a p_b + z0b p_a) + (mu n/t) p_a p_b + (mu² − mu n/t) tdot ]
    ```
    which equals `(J' H_c J)_ab` for C's cone Hessian. Dense path skips contacts with `dist − includemargin ≥ 0`, condim 1, non-CONE state, `dm==0`.
  - Discrete integrator: add tendon (`efm_ts`), actuator (`efm_as`), `efm_K`, flex-contact metric outer products (atomics, upper triangle).
  - Factor+solve (`_cholesky_factorize_solve`): nv ≤ 32 → `_update_gradient_cholesky`: one tile `(nv, nv)`, `wp.tile_cholesky_inplace(fill_mode="upper")` and `tile_cholesky_solve`; nv > 32 → blocked Cholesky (block 16) in `block_cholesky.py`. Outputs `search = −H^{-1}grad`, `search_dot = ||H^{-1}g||²`, `newton_decrement = g·H^{-1}g`. **No rank-1 updates, no rank-deficiency clamping** (tile Cholesky has no mindiag); a full refactorization every iteration (except the incremental skip paths).
- Augmented blocked Cholesky (`_create_blocked_cholesky_augmented_factorize_solve_func`): the RHS `g` is placed in the last column of the final block (`border_size = 15`), with `A[15,15] = 1e30` in the last diagonal block, so the factorization's forward substitution yields `U^{-T} g` in that column; then block back-substitution. Requires the extra padded dof (`augment_cholesky`), thus nv_pad ≥ nv+1.
- Incremental pyramidal Newton (`_use_incremental`: Newton && cone != ELLIPTIC):
  - `_update_gradient_h_incremental[_sparse]`: `H_ij += ±D J_i J_j` for rows in `quad_changed_ids` (+ when now QUADRATIC).
  - `_update_gradient_cholesky_blocked_skip_unchanged`: if `quad_changed_count==0` reuse `hfactor` and only solve; else refactor.
  - **Stable fast path**: worlds with `state_changed_count==0` skip `qfrc_constraint`, grad, and solve: search kept (line search is scale invariant), gradient tracked as `grad_true = grad_scale * grad_stale`, updated `grad_scale ← grad_scale − α`; `grad_dot`, `newton_decrement` scaled by `(new/old)²`; `search_unchanged=True` lets the next line search reuse `mv`, `jv`. This is exact only when no constraint changes state (cost exactly quadratic along the ray). A world whose accepted step is below the noise floor (`ls_exhausted`) is forced to rebuild.

### 10.5 Line search (`_linesearch_iterative_kernel`, one tile/block per world, threads stride over rows)
Not C's exact search; a parallel 3-point bracketing Newton (same as MJX):
1. `mv = M search` (skipped on unchanged search), `jv = J search` (fused into kernel when dense & nv ≤ 50; otherwise separate kernel, possibly with atomics).
2. Elliptic quad prep (per primary row, barrier-synchronized): `quad = Σ_rows (½ D Jaref², D Jaref jv, ½ D jv²)` stored at row 0; `quad1 = (u0=Jaref0 mu, v0=jv0 mu, u_perp_sq)` at row 1; `quad2 = (u_par, v_norm, Dm)` at row 2, where the tangential `u` (Jaref_j fr_j) is decomposed relative to `v` (jv_j fr_j): `v_norm = ||v||`, `u_par = u·v/||v||`, `u_perp_sq = ||u∧v||²/||v||²` (`=||u||²` if v=0), so `T(α)² = (u_par + α v_norm)² + u_perp_sq`. Non-elliptic rows use direct evaluation from (Jaref, jv, D) without quad.
3. `gtol = max(tol*ls_tol*sqrt(search_dot)/inertia_scale, 1e-6)` (C has no 1e-6 floor).
4. Gauss: `quad_gauss = (0, Σ search(Ma − qfrc_smooth) + Σ_eq D jv Jaref, Σ ½ search mv + Σ_eq ½ D jv²)` — equality rows folded into the Gauss quadratic (they are always quadratic). Note C computes `v·Ma − qfrc_smooth·v` as two dots; Warp sums `search*(Ma − qfrc_smooth)` per element.
5. `p0 = (cost=0, d0, d1)` by tile reduction over rows `[ne, nefc)` + Gauss (`d1 = 2 quad_gauss[2] + Σ`).
6. Noise floor: `q1_abs = sqrt(2 max(rows.cost,0) max(rows.hess,0)) + |gauss.d0|`; `gtol_accept = max(gtol, 8·FLT_EPSILON·q1_abs)` (`_ALPHA_NOISE_EPS = 8*1.1920929e-7`); if `|p0.d0| < 8ε q1_abs` → converged with α=0. Incremental: `noise_floor = 8ε max(1, q1_abs/p0.d1)`, `ls_exhausted = |α| < noise_floor`.
7. `lo_in` at `α = −p0.d0/p0.d1` (`math.safe_div`); accept if `|d0| < gtol_accept && cost < 0`.
8. Bracket init: `lo_less = lo_in.d0 < p0.d0`; `lo = lo_less ? lo_in : p0`, `hi = lo_less ? p0 : lo_in`.
9. Loop `ls_iterations` times (compile-time unrolled constant): evaluate 3 points in one reduction (packed `mat33`): `lo_next = lo.α − lo.d0/lo.d1`, `hi_next = hi.α − hi.d0/hi.d1`, `mid = ½(lo.α + hi.α)`. If any candidate has `|d0| < gtol_accept && cost < 0`: take the lowest cost (order lo_next, hi_next, mid, strict `<`) as both lo and hi. Else swaps with `_in_bracket(x,y) = (x.d0 < y.d0 < 0) || (x.d0 > y.d0 > 0)`: lo ← lo_next, then mid, then hi_next (each tested against the updated lo); hi ← hi_next (also if `hi.d0<0 && hi_next.d0>0`), then mid, then lo_next. Done if converged, no swap, `lo.cost<0 && −gtol < lo.d0 < 0`, or `hi.cost<0 && 0 < hi.d0 < gtol`. After each iteration α ← argmin(lo.cost, hi.cost) if either < 0 (tie → hi since `lo_better = lo.cost < hi.cost`). Exhausting iterations sets `OverflowType.LS_ITERATIONS` (and prints a warning).
10. Apply: `qacc += α search`, `Ma += α mv`, `Jaref += α jv` (incremental, like C), write `alpha`, `improvement = −best.cost`.
- Kernel compiled with `fast_math=True` (approximate div/sqrt).
- Elliptic shifted cost (`_eval_elliptic_shifted`) mirrors C's zone-transition table, but SATISFIED→QUADRATIC uses `½ Dm (1+mu²)(N² + Tsqr)` (algebraically equal to C's `q0 + α q1 + α² q2` only through the `R_j mu_j² = const` identity) and middle-zone second derivative uses `T2 = (u_perp_sq/Tsqr)(v_norm²/T)` (equal to C's `det/(T Tsqr)`).
- `ls_iterations` semantics differ: Warp/MJX count bracket iterations (3 evaluations each, plus p0 and lo_in outside), C counts total evaluations.

### 10.6 Termination
- Newton `_solve_done`: `solver_niter += 1`; `improvement = scale*improvement`, `gradient = scale*sqrt(grad_dot)`, `model_improvement = scale*½*newton_decrement`; done if `α == 0` **or** `(0<improvement<tol)` or `gradient<tol` or `model_improvement<tol`; or `niter == iterations` (sets `OverflowType.ITERATIONS`). **Difference:** an `α==0` iteration is counted in `solver_niter` (C breaks without incrementing). Newton decrement in C is gradient-gated only for the zero-iteration exit; in-loop both use plain decrement.
- CG `_solve_beta_finalize_tiled`: **Polak–Ribière-plus** `beta = max(0, g·(Mg − Mg_prev)/max(MJ_MINVAL, g_prev·Mg_prev))` — C defaults to **Hager–Zhang** (PRP only with `-DmjCG_PRP`). Done criteria as Newton without the decrement term. Then `search = −Mgrad + beta*search`.

### 10.7 NoSlip in Warp (`noslip`)
Per world, serial PGS (one thread per world) over friction rows, mathematically equal to C `solNoSlip` but organized differently:
- `_noslip_factor_m`: dense lower Cholesky of M (tree-bounded, `diag = sqrt(max(diag, MJ_MINVAL))`; sleeping dofs identity).
- `_noslip_minv_jt`: `Y[efc] = M^{-1} J[efc]'` (dense nv per row).
- `_noslip_delassus`: needed blocks of `A = J M^{-1} J'` (friction diag; pyramidal pair `A_ii`, `A_i,i+1` for even offset rows; elliptic upper rows of the (dim−1) block) — **unregularized**, no R.
- `_noslip_qacc_init`: `qacc = qacc_smooth + Σ Y f`; residuals are computed as `J qacc − aref` (= `b + A f` without R), and `qacc` is updated incrementally `qacc += δ Y`.
- Sweep identical in structure: iter 0 adds `½ f²/D` for `D>0`; dry friction uses `Adiag = max(MJ_MINVAL, A_ii)`; pyramidal pairs with diag clamp `1e-10`, `K1 ≥ MJ_MINVAL` test; elliptic QCQP with Warp ports `_qcqp2/_qcqp3/_qcqp` (same thresholds 1e-10, 20 iterations); costChange rejection `> 1e-10`. Monolithic over all rows (no islands). Stops if a contact's rows exceed nefc (truncation).
- `_noslip_qacc_qfrc` recomputes `qacc`, `qfrc_constraint`; `_noslip_state` dual states; `efc.Ma = M qacc`.
- Uses the main `solver_niter += niter`; no per-iteration stats.

### 10.8 Islands in Warp (`island.py`)
- `direct_dsu`: `_reset_dsu` (parent=identity, `tree_island=−1`), `_island_dsu` (EFC-parallel, chunks of `ceil(njmax/nchunk)` rows, `nchunk = min(max(1, ceil(2048/nworld)), ceil(njmax/256))`, 32-lane blocks) performing lock-free unions: `_dsu_find` with atomic_min path shortening, `_dsu_union` hooks higher root to lower root via `atomic_cas` retry; activation marks `tree_island=0` via atomic_max. Same incidence rules as C (contact geoms/flex bodies ≤4 per side, connect/weld bodies, friction dof, joint limit, generic J scan); repeated `(type,id)` rows skipped for fixed-support types. **Not included:** C's flex/tendon/actuator metric couplings and passive flex-contact unions.
- `_compress_roots`, `_label_roots` (serial per world: islands numbered by ascending root tree — identical ids to C), `_propagate_labels`.
- `compute_island_mapping`: `efc_tree` (first nonnegative tree), dof/efc counts via atomics, scans, `_island_map_dofs` (closed-form rank: ascending dof order within island, deterministic), `_island_map_constraints` (one thread per island scanning rows ascending; category order eq/friction/other) → same maps as C.
- `update_active_dofs`: compaction maps `dof_cdof/cdof_dof` from `tree_awake` (overflow past `nvmax` flagged `OverflowType.NVMAX`).

### 10.9 Determinism / atomics
Atomic accumulations with nondeterministic order: dense `Jaref`/`jv` when >1 thread per row, sparse `qfrc_constraint`, `grad_dot` (Newton), sparse H assembly, cone curvature, incremental H, metric terms, `quad_changed_ids` order. `opt.deterministic & DeterminismType.ATOMICS` switches these kernels to Warp `DeterministicMode.RUN_TO_RUN` (with `deterministic_max_records`). Tile reductions (`wp.tile_reduce`) are deterministic for a fixed block_dim but their order differs from C's sequential sums.

### 10.10 How to match Warp
- Implement CG with PRP+, Newton with dense H rebuilt (or incremental) and full Cholesky each iteration, the 3-point parallel line search with gtol floor 1e-6 and float32 noise floors, warmstart without cost comparison, iteration counting including α=0 iterations, `tolerance ≥ 1e-6`, AUTO sparsity at nv>32, mu via `impratio_invsqrt`.
- Float32 arithmetic everywhere (`fast_math` in the line-search kernel), with the specific reduction trees of Warp tiles — bitwise parity with Warp on another platform is generally not achievable; aim for tolerance-level agreement.

### 10.11 MJX (JAX) differences (brief)
- CG/Newton only; dense J; `jax.lax.while_loop` (or a single `body` when `iterations == 1`).
- Warmstart: compares full costs of `Context(qacc_warmstart)` vs `Context(qacc_smooth)`, picks warmstart only if strictly smaller (C: ties keep warmstart).
- Termination: `improvement = (prev_cost − cost)/trace(M)` (no `>0` guard), `gradient = ||grad||/trace(M)`; no Newton decrement, no certificate; CG uses PRP+.
- Newton Hessian: dense `full_m + J'(D·active)J` (+ cone blocks `ctx.h` placed in a dense nefc×nefc matrix), symmetrized `(h+h')/2`, `jax.scipy.linalg.cho_factor/cho_solve`.
- Line search: same 3-point scheme as Warp (scan-based, `ls_iterations` scan length), `gtol = tol*ls_tol*||search||*trace(M)` (no floor), `gtol_accept = max(gtol, 8 eps dmag)`; step applied only if improved (`cost < p0.cost`).
- Frictionloss handled by masking (`r = 1/(D + (D==0)mjMINVAL)`).
- `deriv_1 += (quad_total[2]==0)*mjMINVAL` instead of C's convexity clamp.

---

## 11. Reproduction pitfalls / parity notes

1. **Warmstart side effects (C):** for CG/Newton, after `warmstart` `efc_force/efc_state` correspond to `qacc_smooth`; for PGS they are the primal forces at `qacc_warmstart` or zero (dual-cost test `f·b + ½ f'ARf > 0`).
2. **Unconstrained dofs with islands:** `qacc = qacc_smooth` exactly for `idof ≥ nidof`, even with warmstart.
3. **Scale:** termination uses `1/trace(M)` of the island (or full model), not `meaninertia`. Gtol in the line search is `tol*ls_tol*||search||*trace`.
4. **Iteration counting (C):** α==0 → break before `iter++`; `solver_niter` accumulates (`+=`) and NoSlip adds its iterations on top; island stats only for islands < 20; `saveStats` ignores iter ≥ 200.
5. **Line search ls_iterations (C)** counts *all* evaluations (p0, first Newton point, march, midpoints, next-points); bracket updates can exceed the limit by ≤ 2.
6. **Shifted costs:** C line search uses `cost(α) − cost(0)` with careful zone-transition formulas (friction `frictionCostDif`, elliptic rationalized middle-zone difference). Using absolute costs changes acceptance near convergence.
7. **Incremental state:** `qacc, Ma, Jaref` are updated incrementally with `α·search, α·Mv, α·Jv` (never recomputed); `Mv` is recomputed each line search from `search`.
8. **Gauss linear term** is `v·Ma − qfrc_smooth·v` (two separate dot products).
9. **Newton factor:** dense `L L'` lower with `mindiag = mjMINVAL` clamping (rank-deficient columns zeroed); sparse reverse `L'L`; rank-1 updates in constraint order with immediate full refactor on rank loss; cone contributions applied to a copy (`Lcone`) after all quadratic updates; cone fold decision cached until any state change.
10. **Certificate:** CG exits with 0 iterations if `½·scale·g'M^{-1}g < tol`; Newton additionally requires `scale‖g‖ < tol`, then after building H, may exit on `½·scale·g'H^{-1}g < tol` (also gradient-gated). `solver_nnz = nJ` when the certificate fired.
11. **Hager–Zhang** CG direction (η = 0.01, restart when `d'y < mjMINVAL`) is the default in C; PRP+ in Warp/MJX.
12. **PGS:** fixed-seed PCG32 shuffle applied cumulatively to the block list each iteration; Nesterov momentum on by default (`mj_nesterov_momentum` thread-local); extrapolated forces are projected (equality: none; friction: box; limits, frictionless and pyramidal rows: clamp ≥ 0 via `projectCone`; elliptic: normal ≥ 0 plus ellipsoid scaling); restart test uses list order sums. `costChange` rejection threshold 1e-10 (absolute). Elliptic ray update degenerate case zeroes the block.
13. **NoSlip** uses `A = AR − R` with diag clamps `mjMINVAL` (scalar) and `1e-10` (blocks), baseline correction `Σ ½ f² R` at iter 0, friction rows first then contacts in efc order, monolithic or per island (serial).
14. **Dual state labels** for friction are reversed relative to primal (force ≤ −floss → LINEARPOS).
15. **Cone mu:** solvers use `con->mu` (regularized, = friction[0]/sqrt(impratio) via R ratio) for zones and `con->friction` for scaling; PGS/NoSlip QCQP use `con->friction` and the normal force as radius.
16. **Sparse threshold:** C `nv ≥ 60` vs Warp `nv > 32` for `jacobian=auto`; affects Warp kernels choice (and in C changes summation orders of products).
17. **Islands change numerics:** per-island scale, termination and PGS RNG. Warp never solves per island; for parity with Warp, compare against C with islands disabled (`mjDSBL_ISLAND`) — and also note C island discovery includes metric couplings Warp omits.
18. **Warp tolerance clamp** (`tolerance ≥ 1e-6`), `gtol ≥ 1e-6`, float32 noise floors (`8·FLT_EPSILON`), fast-math, no Cholesky regularization, stable fast path for pyramidal Newton, no warmstart-vs-smooth comparison, α==0 iterations counted.
19. **Elliptic cone Hessian guard:** Warp uses `t = max(T, MJ_MINVAL)`; C divides by T (CONE state guarantees T > 0).
20. **Friction bound:** C uses `R·floss`, Warp uses `floss/D`. Equal in exact arithmetic (`efc_D = 1/R`, engine_core_constraint.c:2140), but rounding can flip the zone exactly at the boundary.
21. **Metric (discrete integrator / IPC):** `qfrc_smooth` is shifted by `efm_c (+efm_ca)`, `Ma/Mv` include `mj_effMulAdd`, CG preconditions with the metric preconditioner, Newton merges the lower-triangle metric view S into H. Islands are disabled when the metric requires monolithic folding.

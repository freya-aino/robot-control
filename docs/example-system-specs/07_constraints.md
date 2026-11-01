# 07 — Constraint Model ("soft constraints") and Constraint Construction

**Sources covered**

| Backend | Files |
|---|---|
| C engine | `src/engine/engine_core_constraint.c` (all of it, 3475 lines), `engine_core_constraint.h`; helpers referenced from `engine_core_util.c` (`mj_isSparse`, `mj_isPyramidal`, `mj_isDual`, `mj_isMetric`, `mj_mergeChain`, `mj_mergeChainSimple`, `mj_jacDifPair`, `mj_jacSum`, `mj_contactBodyWeight`), `engine_init.c:mj_defaultSolRefImp`, `engine_util_sparse.{c,h}` (`mju_superSparse`, `mju_combineSparse`), `engine_forward.c` (call sites, `efc_b`) |
| Public headers | `include/mujoco/mjmodel.h` (mjMINMU, mjMINIMP, mjMAXIMP, mjNREF, mjNIMP, mjNEQDATA), `mjtype.h` (mjtConstraint, mjtConstraintState, mjtEq, flags), `mjdata.h` (efc_* fields, mjContact), `mjxmacro.h` (`MJDATA_ARENA_POINTERS_SOLVER/DUAL`) |
| Docs | `doc/computation/index.rst` §Constraint model, §Constraint solver (Primal, Reduced primal, Dual, Parameters, Friction cones); `doc/modeling.rst` §Solver parameters (solref/solimp) |
| MuJoCo Warp | `mujoco_warp/_src/constraint.py` (5868 lines), relevant parts of `solver.py` (`_eval_constraint`, `_eval_elliptic_middle`, `_update_constraint_efc`), `forward.py` (call order), `io.py` (`is_sparse`, `njmax`, `impratio_invsqrt`), `types.py` (Constraint struct) |
| MJX (JAX) | `mjx/_src/constraint.py` |

**Role in the pipeline.** After collision detection has filled `d->contact`, the position stage calls `mj_makeConstraint` (`engine_forward.c:mj_fwdPosition`, line ~161), which (1) counts and arena-allocates all constraint ("efc") arrays, (2) instantiates every active scalar constraint row — equality, friction loss, limit, contact, in that fixed order — writing the constraint Jacobian `efc_J` (dense or CSR-sparse), `efc_pos`, `efc_margin`, `efc_frictionloss`, `efc_type`, `efc_id`, and (3) unless the `discrete` integrator is active, computes the regularization: `efc_diagA` (diagonal approximation of `A = J M⁻¹ Jᵀ`), `efc_R`, `efc_D = 1/R`, and `efc_KBIP` (stiffness, damping, impedance, impedance derivative). `mj_island` then runs, then `mj_projectConstraint` builds the whitened Jacobian `Y = J M^{-1/2}` and `efc_AR = J M⁻¹ Jᵀ + diag(R)` when a dual solver (PGS / noslip) or `diagexact` needs them. The velocity stage calls `mj_referenceConstraint` (`efc_vel = J qvel`, `efc_aref = -B·vel - K·I·(pos-margin)` minus `J̇v` for connect/weld). The constraint stage computes `efc_b = J·qacc_smooth − aref` and the solver (doc 08) repeatedly calls `mj_constraintUpdate`, which evaluates the piecewise-quadratic soft-constraint cost `s(jar)`, the constraint states and forces `efc_force = −∇s`, and `qfrc_constraint = Jᵀ efc_force`.

---

## 1. Data layout, enums and constants

### 1.1 Constants (exact values)

| Name | Value | Where |
|---|---|---|
| `mjMINVAL` | `1E-15` (double), `1E-15f` (single build) | `mjtype.h` |
| `mjMINIMP` | `0.0001` | `mjmodel.h` |
| `mjMAXIMP` | `0.9999` | `mjmodel.h` |
| `mjMINMU` | `1E-5` (min friction coefficient) | `mjmodel.h` |
| `mjNREF` | 2 (solref size) | |
| `mjNIMP` | 5 (solimp size) | |
| `mjNEQDATA` | 12 (eq_data row size) | |
| default solref | `(0.02, 1)` = (timeconst, dampratio) | `engine_init.c:mj_defaultSolRefImp` |
| default solimp | `(0.9, 0.95, 0.001, 0.5, 2)` = (dmin, dmax, width, midpoint, power) | same |
| sparse threshold | `mjJAC_AUTO` → sparse iff `nv >= 60` | `engine_core_util.c:mj_isSparse` |

### 1.2 Enums

```
mjtConstraint:      mjCNSTR_EQUALITY=0, FRICTION_DOF=1, FRICTION_TENDON=2, LIMIT_JOINT=3,
                    LIMIT_TENDON=4, CONTACT_FRICTIONLESS=5, CONTACT_PYRAMIDAL=6, CONTACT_ELLIPTIC=7
mjtConstraintState: SATISFIED=0, QUADRATIC=1, LINEARNEG=2, LINEARPOS=3, CONE=4
mjtEq:              CONNECT=0, WELD=1, JOINT=2, TENDON=3, FLEX=4, FLEXVERT=5 (unsupported),
                    FLEXSTRAIN=6, DISTANCE=7 (removed in 2.2.2; error if used)
mjtCone:            PYRAMIDAL=0, ELLIPTIC=1
mjtJacobian:        DENSE=0, SPARSE=1, AUTO=2
```

Predicates (`engine_core_util.c`):
- `mj_isPyramidal(m)` = `m->opt.cone == mjCONE_PYRAMIDAL`.
- `mj_isSparse(m)` = `jacobian == SPARSE || (jacobian == AUTO && nv >= 60)`.
- `mj_isDual(m)` = `solver == mjSOL_PGS || noslip_iterations > 0`.
- `mj_isMetric(m)` = `integrator == mjINT_DISCRETE`.

Relevant flags: `mjDSBL_CONSTRAINT`, `mjDSBL_EQUALITY`, `mjDSBL_FRICTIONLOSS`, `mjDSBL_LIMIT`, `mjDSBL_CONTACT`, `mjDSBL_REFSAFE` (bit 12), `mjENBL_OVERRIDE` (bit 0), `mjENBL_DIAGEXACT` (bit 5), `mjENBL_SLEEP`.

### 1.3 mjData sizes and efc arrays

Sizes: `ne` (equality rows), `nf` (friction-loss rows), `nl` (limit rows), `nefc` (all rows), `nJ` (non-zeros of J; `nefc*nv` if dense), `nY` (non-zeros of Y), `nA` (non-zeros of AR).

Arena block `MJDATA_ARENA_POINTERS_SOLVER` (allocated in this order by `arenaAllocEfc`, immediately after the contact array):

| Field | Size | Meaning |
|---|---|---|
| `efc_type` | nefc | mjtConstraint |
| `efc_id` | nefc | id of eq / dof / tendon / joint / contact |
| `efc_J_rownnz, efc_J_rowadr, efc_J_rowsuper` | nefc (sparse only) | CSR row descriptors; rowsuper = #subsequent rows with identical pattern |
| `efc_J_colind` | nJ (sparse only) | column indices |
| `efc_J` | nJ | values (dense: row-major nefc×nv) |
| `efc_pos` | nefc | constraint position/residual (distance for limits/contacts) |
| `efc_margin` | nefc | inclusion margin |
| `efc_frictionloss` | nefc | friction-loss bound η (0 for non-friction rows) |
| `efc_diagA` | nefc | approx (or exact) `A_ii` |
| `efc_KBIP` | 4·nefc | (K, B, I=imp, P=imp') per row |
| `efc_D` | nefc | 1/R |
| `efc_R` | nefc | regularizer |
| `tendon_efcadr` | ntendon | first efc row of tendon (see §7.4 quirk) |
| `efc_vel` | nefc | J·qvel (+ surface velocity) |
| `efc_aref` | nefc | reference acceleration |
| `efc_b` | nefc | J·qacc_smooth − aref |
| `efc_state` | nefc | mjtConstraintState |
| `efc_force` | nefc | constraint force |

Dual block (`MJDATA_ARENA_POINTERS_DUAL`, sparse descriptors only in sparse mode): `efc_Y_rownnz/rowadr/colind`, `efc_Y` (nY), `efc_AR_rownnz/rowadr/colind`, `efc_AR` (nA).

`mjContact` fields used: `dist, pos[3], frame[9]` (rows: normal, tangent1, tangent2), `includemargin, friction[5]` (t1, t2, spin, roll1, roll2), `solref[2], solreffriction[2], solimp[5], adhesion, surfacevel[3]`, `mu` (set here), `H[36]` (cone Hessian, set by constraintUpdate), `dim`, `geom[2], flex[2], elem[2], vert[2]`, `exclude` (0 include, 1 in gap, 2 fused, 3 no dofs, 4 passive), `efc_address`.

---

## 2. The convex optimization problem (math)

Reference: `doc/computation/index.rst` §Constraint solver. Notation: `M` inertia (nv×nv), `τ` applied force, `c` bias (Coriolis+gravity), `qacc_smooth = M⁻¹(τ − c)` ("unconstrained acceleration", `qfrc_smooth = τ − c`), `J` the efc Jacobian (nefc×nv), `R` diagonal positive regularizer, `a_ref` the reference acceleration (`efc_aref`), `K` product of friction cones, `η` friction-loss bounds.

### 2.1 Gauss principle and its generalization

Gauss's principle of least constraint for `M v̇ = τ` with hard constraint `J v̇ = a_ref`:

$$ \dot v = \arg\min_x \|x - M^{-1}\tau\|^2_M \quad \text{s.t. } Jx = a_{ref},\qquad \|x\|_M^2 = x^T M x .$$

MuJoCo augments the system with deformation coordinates `z` (velocity ω) having inertia `R⁻¹` and applied force `R⁻¹ a_ref`:

$$ \tilde M = \begin{bmatrix} M & 0\\ 0 & R^{-1}\end{bmatrix},\ \tilde J = [\,J\ \ -I\,],\ \tilde\tau = \begin{bmatrix}\tau\\ R^{-1}a_{ref}\end{bmatrix}. $$

Applying Gauss to the augmented system (and replacing the friction-loss quadratic by a Huber penalty with threshold η) yields the **primal problem**

$$ (\dot v,\dot\omega) = \arg\min_{x,y} \|x - M^{-1}(\tau-c)\|^2_M + \|y - a_{ref}\|^{\mathrm{Huber}(\eta)}_{R^{-1}} $$
$$ \text{s.t. } J_E x - y_E = 0,\ J_F x - y_F = 0,\ J_C x - y_C \in K^* , $$

with unpacked dynamics `M v̇ + c = τ + Jᵀf`, `ω̇ = a_ref − R f`. `K*` is the dual cone (the pyramid is self-dual; the elliptic cone is not).

### 2.2 Reduced primal (what Newton/CG minimize)

Minimizing over `y` in closed form (projection onto plane/cone):

$$ \dot v = \arg\min_x \ \tfrac12 (x - q_{smooth})^T M (x - q_{smooth}) + s(Jx - a_{ref}) $$

(the factor ½ is the implementation's convention: the solver's cost is `0.5·(Ma − qfrc_smooth)ᵀ(a − qacc_smooth) + s`). `s(·)` is convex, C¹, a quadratic spline for pyramidal cones, and is exactly the function evaluated by `mj_constraintUpdate` (§14) with `jar = J·qacc − aref`. Stationarity gives the analytical inverse dynamics `M v̇ + c = τ − Jᵀ∇s(Jv̇ − a_ref)`, i.e. `efc_force = −∇s`, and Hessian `M + Jᵀ H[s] J`.

### 2.3 Dual (what PGS minimizes)

$$ f = \arg\min_{\lambda\in\Omega} \tfrac12 \lambda^T (A+R)\lambda + \lambda^T (a_u - a_{ref}),\quad A = J M^{-1}J^T,\quad a_u = J M^{-1}(\tau-c) + \dot J v . $$

Ω: equality rows free; friction rows `|λ_F| ≤ η`; contacts `λ_C ∈ K`. In code `a_u − a_ref` for the unconstrained part is `efc_b = J·qacc_smooth − aref` (with the `J̇v` term folded into `aref` with a minus sign for connect/weld only, §13.3). `efc_AR = A + R` (§12).

### 2.4 Impedance, R and a_ref

Unconstrained dual minimizer `f⁺ = (A+R)⁻¹(a_ref − a_u)` gives `a_c = A(A+R)⁻¹ a_ref + R(A+R)⁻¹ a_u`. With impedance `d ∈ (0,1)`:

$$ R_{ii} = \frac{1-d_i}{d_i}\,\hat A_{ii} $$

where `Â_ii = efc_diagA` (approximation from compile-time `*_invweight0`, §10, or exact `‖Y_i‖²` under diagexact). If A were diagonal and exact, `a_c,i = d_i a_ref,i + (1−d_i) a_u,i`.

Reference acceleration (spring-damper): `a_ref,i = −b_i (Jv)_i − k_i r_i`, implemented as `aref = −B·vel − K·I·(pos − margin)`, i.e. `b = B`, `k = K·I` (one factor of `d(r)` absorbed into stiffness). Friction loss and elliptic friction rows have `r ≡ 0, K = 0`.

---

## 3. Contact-parameter helpers (top of engine_core_constraint.c)

### 3.1 `mj_contactParam(m, &condim, solref, solimp, friction[5], &adhesion, g1, g2, f1, f2)`

Combines geom (if `f<0`) or flex parameters of the two sides (called from collision, doc on collision; listed here because it lives in this file):

1. Per side `k`: `priority_k, condim_k, solmix_k, solref_k, solimp_k, friction_k[3]`, `adhesion_k` (= `geom_adhesion[g]`, 0 for flex).
2. `adhesion = adhesion1 + adhesion2`.
3. If `priority1 > priority2`: copy everything (condim, solref, solimp, friction, **adhesion = adhesion1**) from side 1; `<`: from side 2.
4. Equal priority:
   - `condim = max(condim1, condim2)`.
   - mix: if both `solmix ≥ mjMINVAL`: `mix = solmix1/(solmix1+solmix2)`; both `< mjMINVAL`: 0.5; only side 1 small: 0.0; only side 2 small: 1.0.
   - solref: if `solref1[0] > 0 && solref2[0] > 0`: `solref = mix·solref1 + (1−mix)·solref2` (both components); else (direct format on either side) element-wise `min(solref1[i], solref2[i])`.
   - `solimp = mix·solimp1 + (1−mix)·solimp2` (all 5).
   - `fri[i] = max(friction1[i], friction2[i])`, i=0..2.
5. Unpack to 5D: `friction = (fri0, fri0, fri1, fri2, fri2)`.
6. `condim ∉ [1,6]` → `mjERROR`.

### 3.2 Override helpers
- `mj_assignFriction(m, target, source)`: `target[i] = max(mjMINMU, OVERRIDE ? opt.o_friction[i] : source[i])`, i=0..4.
- `mj_assignRef / mj_assignImp`: copy `opt.o_solref / o_solimp` if `mjENBL_OVERRIDE`, else source.
- `mj_assignMargin`: `opt.o_margin` if override, else source.

### 3.3 `mj_addContact(m, d, con)`
Calls `mj_clearEfc(d)` (moves arena pointer back to end of contacts; invalidates efc arrays), arena-allocates one `mjContact`, copies; on failure `mj_warning(mjWARN_CONTACTFULL, ncon)` and returns 1; else `ncon++`, returns 0.

---

## 4. `mj_makeConstraint` — top-level driver

```
d->ne = d->nf = d->nl = d->nefc = d->nJ = d->nA = d->nY = 0
if DISABLED(CONSTRAINT): return
nnz = isSparse ? &d->nJ : NULL
ne_alloc = mj_ne(m,d,nnz)                       // §11
nf_alloc = mj_instantiateFriction(m,d,count_only=1,nnz)
nl_alloc = mj_instantiateLimit(m,d,count_only=1,nnz)
nc_alloc = mj_nc(m,d,nnz)                        // also marks passive flex contacts exclude=4
nefc_alloc = sum
if dense: d->nJ = nefc_alloc*nv
d->nefc = nefc_alloc
if !arenaAllocEfc(m,d): return                   // warning mjWARN_CNSTRFULL, mj_clearEfc
fill tendon_efcadr with -1
d->nefc = 0
mj_instantiateEquality; mj_instantiateFriction(0); mj_instantiateLimit(0); mj_instantiateContact
sparse: mjERROR unless ne,nf,nl,nefc == allocated and nJ == rownnz[last]+rowadr[last]
dense : mjERROR if nefc > nefc_alloc  (dense counts are upper bounds)
maxuse_con = max(maxuse_con, ncon); maxuse_efc = max(maxuse_efc, nefc)
if nefc == 0: return
sparse: mju_superSparse(nefc, efc_J_rowsuper, rownnz, rowadr, colind)
if !isMetric: mj_diagApprox; mj_makeImpedance      // discrete: deferred to mj_regularizeConstraint
```

`arenaAllocEfc`: sets `d->parena = ncon*sizeof(mjContact)` (efc arrays start right after contacts; ASAN poisons the rest), allocates each X-macro field with `mj_arenaAllocByte(d, sizeof(type)*nr*nc, alignof(type))`. Note `nefc` used for sizing is the *allocated* count.

`mju_superSparse`: `rowsuper[r] = (rownnz[r]==rownnz[r+1] && colind equal) ? 1 : 0`, last = 0, then reverse accumulate `rowsuper[r] += rowsuper[r+1]` if nonzero. Used by sparse matvec and `mju_sqrMatTDSparse`.

**Row order** (fixed, matters for solvers that iterate in order and for `ne/nf` index ranges):
1. Equality rows, by eq id `i = 0..neq-1` (all types interleaved by id).
2. Friction: dofs `0..nv-1`, then tendons `0..ntendon-1`.
3. Limits: joints `0..njnt-1` (for slide/hinge: lower side then upper side), then tendons (lower then upper).
4. Contacts by contact index, each contributing 1 (frictionless), `2(dim−1)` (pyramidal) or `dim` (elliptic) contiguous rows.

Solvers rely on: rows `[0, ne)` equality, `[ne, ne+nf)` friction, rest limit+contact.

---

## 5. `mj_addConstraint` (static) — row insertion

Signature: `(m, d, jac, pos, margin, frictionloss, size, type, id, NV, chain)`.

1. `empty = 0` for contact types; `empty = 1` otherwise (non-contacts may be dropped if their Jacobian is identically zero).
2. **Dense**: if `empty`, scan `size*nv` entries; any non-zero → `empty = 0`. If not empty, copy `size*nv` into `J + nefc*nv`.
3. **Sparse**: `NV = max(0, NV)`. If `NV>0`: `empty = 0`; else if `empty`: return (no rows). `NV && !chain` → `mjERROR`. For each of the `size` rows: `rowadr[nefc+i] = (nefc+i) ? rowadr[prev]+rownnz[prev] : 0`, `rownnz = NV`, copy `chain` into colind and `jac + i*NV` into values (jac is row-major `size×NV`).
4. If `empty`: return (in sparse mode with NV=0 the row descriptors were already written for contacts; for non-contacts it returned in step 3).
5. For each row: `efc_pos = pos ? pos[i] : 0`, `efc_margin = margin ? margin[i] : 0`, `efc_frictionloss = frictionloss`, `efc_type = type`, `efc_id = id`.
6. `nefc += size`; type EQUALITY → `ne += size`; FRICTION_* → `nf`; LIMIT_* → `nl`.

Note: contacts with `NV==0` never reach here (excluded earlier with `exclude=3`).

`mj_mulJacVec(m,d,res,vec)` / `mj_mulJacTVec`: no-op when `nefc==0`; sparse uses `mju_mulMatVecSparse(..., efc_J_rowsuper)` / `mju_mulMatTVecSparse`; dense uses `mju_mulMatVec / mju_mulMatTVec`.

---

## 6. Equality constraints — `mj_instantiateEquality`

Returns immediately if `DISABLED(EQUALITY)` or `m->nemax == 0`. Sleep filter: `sleep_filter = ENABLED(SLEEP) && ntree_awake < ntree`; skip `!eq_active[i]` and (if filtering) `mj_sleepState(EQUALITY,i) == ASLEEP`. Stack: `jac[0], jac[1], jacdif` each `6·nv`; sparse: `chain, chain2` of size nv.

### 6.1 Anchors — `mj_equalityAnchors(m,d,eq,pos1,pos2,&b1,&b2)`

`data = eq_data + 12·eq`.
- Body semantics (`eq_objtype == mjOBJ_BODY`):
  - CONNECT: `pos1 = xmat[obj1]·data[0:3] + xpos[obj1]`, `pos2 = xmat[obj2]·data[3:6] + xpos[obj2]`.
  - WELD (note swapped anchors): `pos1 = xmat[obj1]·data[3:6] + xpos[obj1]`, `pos2 = xmat[obj2]·data[0:3] + xpos[obj2]`.
  - `b1 = obj1, b2 = obj2`.
- Site semantics: `pos1 = site_xpos[obj1]`, `pos2 = site_xpos[obj2]`, `b_k = site_bodyid[obj_k]`.

### 6.2 CONNECT (3 rows)

```
cpos[0:3] = pos1 - pos2
NV = mj_jacDifPair(m,d,chain, b2, b1, pos2, pos1, jac[1], jac[0], jacdif, NULL..., issparse, skipcommon=0)
   // jacdif = J(b1,pos1) - J(b2,pos2)   ("opposite of contact": 0 − 1)
jac[0][0:3*NV] = jacdif
size = 3
```
`mj_jacDifPair(b1', b2', p1', p2')` returns `jac(b2',p2') − jac(b1',p1')`; called with `(b2, b1)` so the result is `J_1 − J_2`. Sparse chain = `mj_mergeChain(b2,b1,0)` (union of dof ancestors of both weld-roots, increasing order) or, if both bodies are "simple" and distinct, `mj_mergeChainSimple` (concatenation of their dofs, lower body id first).

Point Jacobian (doc 03/04): for each ancestor dof `j`, `jacp_j = cdof_lin_j + cdof_ang_j × (p − subtree_com[rootid[b]])`, `jacr_j = cdof_ang_j`.

### 6.3 WELD (6 rows), with torquescale

```
cpos[0:3] = pos1 - pos2
torquescale = data[10]
NV = mj_jacDifPair(b2,b1,pos2,pos1, jac1p=jac[1], jac2p=jac[0], jacdifp=jacdif,
                   jac1r=jac[1]+3nv, jac2r=jac[0]+3nv, jacdifr=jacdif+3nv, ...)
jac[0][0:3NV]   = jacdif[0:3NV]              // translational
jac[0][3NV:6NV] = jacdif[3nv:3nv+3NV]        // rotational (compressed to NV stride)
// orientation
body:  quat = xquat[obj1] * relpose (data[6:10]);  quat1 = neg(xquat[obj2])
site:  quat = xquat[b1]*site_quat[obj1]; quat1 = neg(xquat[b2]*site_quat[obj2])
quat2 = quat1 * quat
cpos[3:6] = torquescale * quat2[1:4]          // sin(θ/2)·axis, scaled
// rotational Jacobian correction, per column j < NV:
axis = (jac0[3NV+j], jac0[4NV+j], jac0[5NV+j])     // ω-Jacobian difference column
quat2 = mulQuatAxis(quat1, axis)                    // neg(q1)·(0,axis)
quat3 = quat2 * quat                                 // ·q0·relpose
jac0[(3..5)NV+j] = 0.5 * quat3[1..3]
then scale rotational block by torquescale
size = 6
```
`mju_mulQuatAxis(res, q, a)` = `q ⊗ (0, a)`:
```
res = (-q1a0 - q2a1 - q3a2,  q0a0 + q2a2 - q3a1,  q0a1 + q3a0 - q1a2,  q0a2 + q1a1 - q2a0)
```
`neg(q) = (q0, -q1, -q2, -q3)`. Note: the residual is the vector part of `q1⁻¹ q0 relpose`, which equals `sin(θ/2)·axis` (doc), weak near 180°.

### 6.4 JOINT / TENDON polynomial (1 row)

For `j = 0` and, if `id[1] >= 0`, `j = 1`:
- JOINT: `pos_j = qpos[jnt_qposadr[id_j]]`, `ref_j = qpos0[jnt_qposadr[id_j]]`; Jacobian = unit vector at `jnt_dofadr[id_j]` (sparse: `NV=1, chain=[dofadr], jac=[1]`; second joint: `NV2, chain2`).
- TENDON: `pos_j = ten_length[id_j]`, `ref_j = tendon_length0[id_j]`; set `tendon_efcadr[id_j] = i` **if still −1** (stores the *equality id* `i`, not an efc address — a quirk preserved here); Jacobian = tendon row of `ten_J` (sparse copy of `ten_J_colind/ten_J` segment; dense `mju_sparse2dense`).

If both objects:
```
dif  = pos[1] - ref[1]
cpos = pos[0] - ref[0] - data[0] - (data[1]*dif + data[2]*dif*dif + data[3]*dif*dif*dif + data[4]*dif*dif*dif*dif)
deriv = data[1] + 2*data[2]*dif + 3*data[3]*dif*dif + 4*data[4]*dif*dif*dif
sparse: NV = mju_combineSparse(jac[0], jac[1], 1, -deriv, NV, NV2, chain, chain2)
dense : jac[0] += -deriv * jac[1]
```
(Exact association: `((pos0 − ref0) − data0) − (Σ)`; Σ evaluated left to right with explicit products.)
`mju_combineSparse(dst, src, a, b, ...)`: if patterns identical → `dst = a·dst + b·src`; else backward merge of sorted index sets producing the union (entries present only in one operand are scaled `a·dst` or `b·src`; zeros retained).

Single object: `cpos = pos0 − ref0 − data[0]`. `size = 1`.

### 6.5 FLEX (edge length; 1 row per non-rigid edge)

For `e` in `[flex_edgeadr[f], +flex_edgenum[f])`, skip `flexedge_rigid[e]`:
`cpos = flexedge_length[e] − flexedge_length0[e]`; sparse: `mj_addConstraint(flexedge_J + rowadr, ..., NV=flexedge_J_rownnz[e], chain=flexedge_J_colind+rowadr)`; dense: scatter into `jac[0]` (zeroed). Each edge is added inside the case (size stays 0 for the final add). Rows with all-zero Jacobian are dropped in dense mode (see pitfall on diagApprox alignment).

### 6.6 FLEXSTRAIN (eigenmode strain constraints, per cell or shell face)

`f = obj1`, `interp = flex_interp[f]`, `order = |interp|`, `shell = interp < 0`. Skip if `order == 0 || flex_nodenum[f] == 0`. `order > 2` → error.
- `npe = (order+1)^2` (shell, `elem_idx = (int)data[0]`) or `(order+1)^3` (`ci,cj,ck = data[0..2]`, `elem_idx = ci·cy·cz + cj·cz + ck`), `(cx,cy,cz) = flex_cellnum[3f..]`.
- Gather node indices `gindices` (≤125) with `mju_flexGatherFaceState / mju_flexGatherCellState`.
- Node world positions: if `flex_centered[f]` or local node offset is zero → `xpos[body]`; else `xmat[body]·flex_node + xpos[body]`. Reference positions `flex_node0`.
- Corotational quaternion:
  - Solid: `mat = mju_defGradient(center=(.5,.5,.5), xpos_e, order)`; each column `c` divided by `dc = refpos[last][c] − refpos[0][c]` if `dc > mjMINVAL`; `elem_quat = neg(mju_flexPolarRotation(mat))`.
  - Shell: normal axis from cumulative face blocks `face_sizes = {cy·cz, cy·cz, cx·cz, cx·cz, cx·cy, cx·cy}`, normals `{0,0,1,1,2,2}`; tangents `t1 = Σ x_n·φ'(.5)φ(.5)`, `t2 = Σ x_n·φ(.5)φ'(.5)` (scaled by 1/d0, 1/d1 if > mjMINVAL), `normal = t1 × t2`; columns placed at axes `(na0, na1, normal_axis)`; polar rotation, negated.
- `cell_pos_and_jac`: per-cell chain = union of `mj_bodyChain` of all node bodies (sorted by scanning `dof_used[0..nv)`), per-node 3×cell_nnz Jacobians via `mj_jacSparse` remapped by linear scan.
- Eigen data: `k_elem = flex_stiffness + flex_stiffnessadr[f] + elem_idx·(3npe)²`; `neig = (int)k_elem[0]`; eigenvector `eig` at `k_elem + 1 + eig·3npe`.
- `displ_e[n] = rotVecQuat(xpos_e[n], elem_quat) − refpos_e[n]`.
- For each eigenmode: `cpos = Σ_j eigvec[j]·displ_e[j]`; `dSdx_local[n] = rotVecQuat(eigvec[n], neg(elem_quat))`; `strain_jac[k] = Σ_{n,c} dSdx[3n+c]·node_jac[3n+c][k]` (skipping zero weights); add 1 row (sparse with chain; dense via a scattered/unscattered nv buffer).

### 6.7 Final add
`if (size) mj_addConstraint(jac[0], cpos, NULL, 0, size, EQUALITY, i, sparse?NV:0, sparse?chain:NULL)`.

---

## 7. Friction loss — `mj_instantiateFriction(m,d,count_only,nnz)`

Return 0 if `DISABLED(FRICTIONLOSS)`.
- DOFs `i = 0..nv-1`: skip if `dof_frictionloss[i] == 0` (test is `!value`), skip sleeping dof. Count: `nf += mj_addConstraintCount(m,1,1)`, `nnz += 1`. Instantiate: sparse `jac=[1]`, chain `&i`; dense unit vector `e_i`; `mj_addConstraint(jac, pos=NULL, margin=NULL, frictionloss=dof_frictionloss[i], 1, FRICTION_DOF, id=i)`.
- Tendons `i`: if `tendon_frictionloss[i] > 0` (no sleep check): count `mj_addConstraintCount(m,1,ten_J_rownnz[i])`; instantiate with `ten_J` row (sparse direct pointers; dense via `mju_sparse2dense`), type FRICTION_TENDON, id i; `tendon_efcadr[i] = efcadr` (row address before insertion) if −1.

`mj_addConstraintCount(m,size,NV)` = dense: `nv ? size : 0`; sparse: `max(0,NV) ? size : 0`.

`efc_pos = 0`, `efc_margin = 0`.

---

## 8. Limits — `mj_instantiateLimit(m,d,count_only,nnz)`

Return 0 if `DISABLED(LIMIT)`.

**Joints** `i` with `jnt_limited[i]` (skip sleeping): `margin = jnt_margin[i]`.
- SLIDE/HINGE: `value = qpos[jnt_qposadr[i]]`; for `side = −1` then `+1`:
  `dist = side·(jnt_range[2i + (side+1)/2] − value)` (lower: `value − range0`, upper: `range1 − value`); **if `dist < margin`** add 1 row with `J = −side` at `jnt_dofadr[i]`, `pos = dist`, `margin = margin`, type LIMIT_JOINT, id i. Both sides can be active simultaneously (two rows).
- BALL: `quat = normalize4(qpos[adr..adr+3])`; `angleAxis = mju_quat2Vel(quat, 1)`; `value = mju_normalize3(angleAxis)` (returns the norm; if `< mjMINVAL` axis set to (1,0,0)); `dist = max(range0, range1) − value`; if `dist < margin`: 1 row, Jacobian `−axis` on dofs `dofadr..dofadr+2`.
  `mju_quat2Vel(q, dt)`: `axis = q[1:4]`, `s = normalize3(axis)`, `speed = 2·atan2(s, q0)`, if `speed > π` → `speed −= 2π`; result `axis·speed/dt`.
- FREE joints never limited.

**Tendons** `i` with `tendon_limited[i]` (no sleep check): `value = ten_length[i]`, `margin = tendon_margin[i]`; for `side = −1, +1`: `dist = side·(tendon_range[2i+(side+1)/2] − value)`; if `dist < margin`: Jacobian `−side·ten_J_row` (sparse: `mju_scl` into stack then chain = ten colind; dense: densify then scale), type LIMIT_TENDON, `tendon_efcadr` set if −1.

Activation criterion is strictly `dist < margin` (row present with `pos − margin < 0`).

---

## 9. Contacts

### 9.1 `mj_contactJacobian(m,d,con,dim, jacdifp, jacdifr, jac1p, jac2p, jac1r, jac2r, chain)`

- **Special case** (each side is a geom or a vertex of a non-interpolated flex): `bid[s] = geom_bodyid[geom[s]]` or `flex_vertbodyid[flex_vertadr[flex[s]] + vert[s]]`. If `noDofs(bid,2)` → return 0. Else
  `NV = mj_jacDifPair(m,d,chain, bid[0], bid[1], con->pos, con->pos, jac1p, jac2p, jacdifp, dim>3 ? (jac1r,jac2r,jacdifr) : NULL, isSparse, flg_skipcommon=1)` → `jacdif = J(body2) − J(body1)` at the contact point; **common ancestor dofs are skipped** in sparse mode (their contribution cancels exactly).
- **General case** (flex elements / interpolated flex): `nb = mj_contactBodyWeight(side 0)` with weights negated, then append side 1; `noDofs` check; `NV = mj_jacSum(m,d,chain,nb,bid,bweight,con->pos, jacdifp, jacdifr, dim>3)` (weighted sum of point Jacobians; sparse merges chains with `mju_addToSparseMat`; bid buffer size 729 = 27·27).

`noDofs(body, n)`: `wid(b) = body_treeid[b] >= 0 ? body_weldid[b] : 0`; returns 1 iff all `wid` equal (all welded together or all static).

### 9.2 `mj_instantiateContact`

Return if `DISABLED(CONTACT) || ncon == 0 || nv == 0`. Stack: `jac` 6nv, `jacdifp, jacdifr, jac1p, jac2p, jac1r, jac2r` 3nv each, sparse `chain`.

For `i = 0..ncon-1`, skip `contact[i].exclude != 0`:
```
con->efc_address = d->nefc
NV = mj_contactJacobian(...)
if NV == 0: con->efc_address = -1; con->exclude = 3; continue
// rotate to contact frame: jac (rows×NV) = frame(rows×3) · jacdif(3×NV)
jac[0 : (dim>1?3:1)·NV] = frame[0:(dim>1?3:1)] · jacdifp
if dim > 3: jac[3NV : dim·NV] = frame[0 : dim-3] · jacdifr
```
So rows are: `0`=normal·Δv_lin, `1,2`=tangents·Δv_lin, `3`=normal·Δω (spin), `4,5`=tangents·Δω (rolling).

- **dim == 1** (frictionless): 1 row, `pos = &con->dist`, `margin = &con->includemargin`, type CONTACT_FRICTIONLESS.
- **Pyramidal** (dim>1): `cpos[0]=cpos[1]=dist`, `cmargin[0]=cmargin[1]=includemargin`; for `k = 1..dim-1`:
  ```
  row+ = jac[0] + friction[k-1]·jac[k]      (mju_addScl into jacdifp)
  row- = jac[0] − friction[k-1]·jac[k]      (into jacdifp+NV)
  mj_addConstraint(jacdifp, cpos, cmargin, 0, size=2, CONTACT_PYRAMIDAL, id=i, NV, chain)
  ```
  Row order: `(+t1, −t1, +t2, −t2, +spin, −spin, +roll1, −roll1, +roll2, −roll2)`. All rows carry `pos = dist`. Basis matrix E column for edge `(k,±)` = `e_0 ± μ_k e_k`.
- **Elliptic**: `cpos = (dist, 0, …)`, `cmargin = (includemargin, 0, …)`, one call with `size = dim`, type CONTACT_ELLIPTIC (E = I).

The contact force acts from geom[0] to geom[1] (normal points from geom[0] to geom[1]); J = Eᵀ·frame·(J₂ − J₁).

Inactive contacts (`dist ≥ includemargin` without adhesion/speculative) arrive with `exclude = 1` (set in collision: `exclude = (dist >= includemargin) && !adhesion && !speculative`), so they generate no rows. Adhesive contacts are kept even when separated (within gap).

---

## 10. Counting — `mj_ne`, `mj_nc`, `mj_jacSumCount`

These mirror instantiation exactly in sparse mode (verified by the post-checks in §4).

`mj_ne(m,d,nnz)` (`issparse = nnz != NULL`): same filters (disabled, `nemax==0`, inactive, sleeping). Per type:
- CONNECT size 3 / WELD size 6: if counting nnz, map sites to bodies, `NV = mj_jacDifPair(m, NULL, chain, id1, id0, NULL…, issparse, 0)` (count-only mode returns chain length).
- JOINT/TENDON size 1: chains as in instantiation; if two objects `NV = mju_combineSparseCount(...)` (union size).
- FLEX: `size = #non-rigid edges`; `NV += mj_jacDifPair(edge vertex bodies)` per edge (accumulated, already total nnz).
- FLEXSTRAIN: `size = neig` (0 if `flex_stiffnessadr < 0`); `NV = size · mj_jacSumCount(node bodies)`.
- default → `mjERROR` ("now-removed distance").
- `ne += mj_addConstraintCount(size, NV)`; `nnze += (FLEX||FLEXSTRAIN) ? NV : size·NV`.

`mj_nc(m,d,nnz)`: return 0 if contact disabled or `ncon==0`. For each contact:
1. **Passive flex marking**: `wants = (flex[0]>-1 && mj_effFlexContactPossible(m,flex[0])) || (same for flex[1])`; `ok = flex[0]>-1 && flex[1]>-1` or, for a geom side, `body_weldid[geom_bodyid[g]] == 0` (world-welded). If `wants && ok`: `efc_address=-1, exclude=4`.
2. Skip excluded.
3. Sleep consistency check (`mjERROR` if a contact touches a sleeping body — should not occur).
4. If counting nnz: NV as in `mj_contactJacobian` (skipcommon=1 for special case; `mj_jacSumCount` otherwise); `NV==0` → skip (not counted).
5. `dim==1`: `nc += 1, nnz += NV`; pyramidal: `nc += 2(dim−1)`, `nnz += 2(dim−1)·NV`; elliptic: `nc += dim`, `nnz += dim·NV`.

In dense mode NV is not computed, so contacts with no dofs are over-counted (allowed; dense only checks `nefc ≤ allocated`).

`mj_jacSumCount(m,d,chain,n,body)`: `NV = mj_bodyChain(body[0])`, then union with each further body chain via `mju_addChains`.

---

## 11. `mj_diagApprox` — approximate `A_ii`

Uses compile-time inverse weights (`body_invweight0[2b+0]` translational, `[2b+1]` rotational; `dof_invweight0`, `tendon_invweight0`, `flexedge_invweight0`), computed at `qpos0` (doc on compiler).

Loop `i` over rows (may advance `i` by more than one):
- EQUALITY:
  - CONNECT: `b1,b2` (site → site_bodyid): `dA = iw0[2b1] + iw0[2b2]`.
  - WELD: `dA = iw0[2b1 + (weldcnt>2)] + iw0[2b2 + (weldcnt>2)]`, `weldcnt = (weldcnt+1) % 6` (rows 0–2 translational, 3–5 rotational; weldcnt is a single running counter across all welds).
  - JOINT/TENDON: `dof_invweight0[jnt_dofadr[obj1]]` or `tendon_invweight0[obj1]`, plus obj2 contribution if `obj2 >= 0`.
  - FLEX: for every non-rigid edge of the flex, `dA[i++] = flexedge_invweight0[e]`; then `i--`.
  - FLEXSTRAIN: `avg = (1/npe) Σ_nodes body_invweight0[2·nodebody]`; `dA[i..i+neig) = avg`.
- FRICTION_DOF: `dof_invweight0[id]`. LIMIT_JOINT: `dof_invweight0[jnt_dofadr[id]]`. FRICTION_TENDON / LIMIT_TENDON: `tendon_invweight0[id]`.
- Contacts: for both sides, `(bid, bweight) = mj_contactBodyWeight(side)` (geom: single body weight 1); `tran = Σ iw0[2b]·w`, `rot = Σ iw0[2b+1]·w` (weights positive here).
  - FRICTIONLESS: `dA = tran`.
  - ELLIPTIC: `dA[i+j] = (j < 3 ? tran : rot)`, `i += dim−1`.
  - PYRAMIDAL: for `j = 0..dim−2`: `fri = friction[j]`, `dA[i+2j] = dA[i+2j+1] = tran + fri²·(j < 2 ? tran : rot)`; `i += 2dim−3`.

(Pyramidal edge `n ± μ t` has `A = A_nn + μ² A_tt` if cross terms vanish.)

---

## 12. Solver parameters and impedance — `getsolparam`, `getposdim`, `getimpedance`, `mj_makeImpedance`

### 12.1 `getsolparam(m,d,i, solref, solreffriction, solimp)`

`solreffriction = (0,0)` by default. Source by type: EQUALITY → `eq_solref/eq_solimp[id]`; LIMIT_JOINT → `jnt_solref/solimp`; FRICTION_DOF → `dof_solref/solimp`; LIMIT_TENDON → `tendon_solref_lim/solimp_lim`; FRICTION_TENDON → `tendon_solref_fri/solimp_fri`; contacts → `contact[id].solref, .solreffriction, .solimp`.

Then, in order:
1. Mixed sign: `(solref[0]>0) XOR (solref[1]>0)` → `mju_warning("mixed solref format…")`, replace solref with default `(0.02, 1)`.
2. refsafe: if `!DISABLED(REFSAFE) && solref[0] > 0 && !isMetric`: `solref[0] = max(solref[0], 2·timestep)`.
3. Same mixed-sign check for solreffriction → replaced by `(0,0)` (with warning).
4. refsafe for `solreffriction[0] > 0` likewise.
5. Clamp solimp:
   ```
   solimp[0] = min(mjMAXIMP, max(mjMINIMP, solimp[0]))   // dmin
   solimp[1] = min(mjMAXIMP, max(mjMINIMP, solimp[1]))   // dmax
   solimp[2] = max(0, solimp[2])                          // width
   solimp[3] = min(mjMAXIMP, max(mjMINIMP, solimp[3]))   // midpoint
   solimp[4] = max(1, solimp[4])                          // power
   ```

### 12.2 `getposdim(m,d,i,&pos,&dim)`

Default `dim = 1, pos = efc_pos[i]`. CONTACT_ELLIPTIC: `dim = contact.dim`. CONTACT_PYRAMIDAL: `dim = 2(contact.dim−1)`. EQUALITY WELD: `dim = 6, pos = ‖efc_pos[i..i+6)‖₂`; CONNECT: `dim = 3, pos = ‖efc_pos[i..i+3)‖₂`. All rows of such a group share one impedance.

### 12.3 `getimpedance(solimp, pos, margin, &imp, &impP)` — the sigmoid

```
if solimp[0] == solimp[1] || solimp[2] <= mjMINVAL:  imp = 0.5*(dmin+dmax); impP = 0; return
x = (pos - margin)/width;  sgn = 1; if x < 0: x = -x, sgn = -1
if x >= 1 || x <= 0:  imp = (x >= 1 ? dmax : dmin); impP = 0; return
if power == 1:        y = x;  yP = 1
elif x <= mid:        a = 1/power(mid, p-1);    y = a*x^p;         yP = p*a*x^(p-1)
else:                 b = 1/power(1-mid, p-1);  y = 1 - b*(1-x)^p; yP = p*b*(1-x)^(p-1)
imp  = dmin + y*(dmax - dmin)
impP = yP * sgn * (dmax - dmin) / width
```
`power(a,b)`: returns `a` if `b==1`, `a*a` if `b==2`, else `mju_pow(a,b)` (exact fast paths; default power 2 hits `a*a`). No clamp of `imp` to `[dmin,dmax]` (dmin > dmax is allowed and gives a decreasing curve). For friction rows and elliptic friction rows `pos = 0, margin = 0` ⇒ `imp = dmin` (unless flat).

The impedance argument is `r = pos − margin` (contacts: `dist − includemargin`; limits: `dist − margin`; equality: residual / residual norm; margin 0).

### 12.4 `mj_makeImpedance(m,d)` — R, KBIP, D, diagA

`h = opt.timestep`, `metric = isMetric(m)`.

**Pass 1** (loop `i`, step = dim):
```
getsolparam; getposdim; getimpedance(solimp, pos, efc_margin[i], &imp, &impP)
for j in 0..dim-1:
  R[i+j] = max(mjMINVAL, (1-imp)*diagA[i+j]/imp)
  tp = efc_type[i+j]
  elliptic_friction = (tp == ELLIPTIC && j > 0)
  ref = (elliptic_friction && (solreffriction[0] || solreffriction[1])) ? solreffriction : solref
  // K
  if tp in {FRICTION_DOF, FRICTION_TENDON} or elliptic_friction:  K = 0
  elif ref[0] > 0:  K = 1 / max(mjMINVAL, dmax² · ref[0]² · ref[1]²)       // standard
  else:             K = -ref[0] / max(mjMINVAL, dmax²)                    // direct
  // B
  if ref[1] > 0:    B = 2 / max(mjMINVAL, dmax · ref[0])                   // standard
  else:             B = -ref[1] / max(mjMINVAL, dmax)                     // direct
  KBIP[4(i+j)..] = (K, B, imp, impP)
  if metric:   (discrete integrator)
    if !DISABLED(REFSAFE) && ref[0] > 0 && K > 0 && tp ∈ {LIMIT_JOINT, LIMIT_TENDON, CONTACT_*}:
      excess = h²·K·I
      if excess > 1:
        fmax = mjMAXIMP·(1-I) / max(mjMINVAL, I·(1-mjMAXIMP))
        K = 1/(h²·I)
        B = min(B/sqrt(excess), max(0, fmax-2)/h)
    f = 1 + h·B + h²·K·I                       // implicitFactor
    Rmin = max(mjMINVAL, (1-mjMAXIMP)·diagA[i+j]/mjMAXIMP)
    R[i+j] = max(R[i+j]/f, Rmin)
```
Here `dmax = solimp[1]` ("d_width"), standard `ref = (timeconst, dampratio)`. Note in the standard B formula the `ref[1] > 0` test chooses the B format independently of K's `ref[0] > 0` test (mixed formats were already sanitized). For standard format with dampratio, effective stiffness `K·I = d(r)/(dmax² tc² dr²)` and `B = 2/(dmax·tc)`.

**Pass 2** (frictional contacts, `i` from `ne+nf`):
```
if type is PYRAMIDAL or ELLIPTIC (first row of a contact):
  R[i+1] = R[i] / max(mjMINVAL, impratio)
  contact.mu = friction[0] · sqrt(R[i+1]/R[i])           // = friction[0]/sqrt(impratio)
  ELLIPTIC: for j = 1..dim-2: R[i+j+1] = R[i+1]·friction[0]²/friction[j]²;  i += dim-1
  PYRAMIDAL: Rpy = 2·mu²·R[i];  R[i..i+2(dim-1)) = Rpy;   i += 2(dim-1)-1
```
So for elliptic cones the friction rows' R (and D) are *derived from the normal row* (its diagA, imp, and under discrete its implicit factor), satisfying `R_j μ_j² = R_1 μ_1²` (required for the analytical elliptic inverse). For pyramidal, all edges share `Rpy = 2μ²R₀` where `R₀` is computed from the first edge's `diagA = tran + fri0²·tran`.

**Pass 3**: `efc_D[i] = 1/R[i]`.
**Pass 4**: `efc_diagA[i] = R[i]·I/(1−I)` (so `R = (1−imp)/imp·diagA` holds exactly after the adjustments).

---

## 13. Velocity and reference — `mj_velocityConstraint`, `mj_referenceConstraint`

### 13.1 `mj_velocityConstraint`: `efc_vel = J·qvel` (`mj_mulJacVec`), then `mj_addSurfaceVel`.

`mj_addSurfaceVel` (only if `m->flg_surfacevel`): for each contact with `efc_address >= 0 && dim > 1` and nonzero `surfacevel` (sv = tangent1, tangent2, spin; geom[1] − geom[0]); `n = min(dim, 4)`:
- elliptic: `efc_vel[adr+k] += sv[k−1]`, k=1..n−1;
- pyramidal: `mu = friction[k−1]`; `efc_vel[adr+2(k−1)] += mu·sv[k−1]`, `efc_vel[adr+2(k−1)+1] −= mu·sv[k−1]`.

### 13.2 `mj_referenceConstraint`

```
mj_velocityConstraint
shift = metric ? h : 0
aref[i] = -B·vel[i] - K·I·(pos[i] - margin[i] + shift·vel[i])
if ne > 0: mj_Jdotv(m, d, aref)               // aref -= J̇·qvel for connect/weld rows
if metric: aref[i] /= (1 + h·B + h²·K·I)        // same factor f that divided R
mj_adhesionRef                                   // after division
```
Discrete variant = backward-Euler on each row's spring–damper (position transported by `h·vel`).

### 13.3 `mj_Jdotv(m,d,result)` — Coriolis term of equality constraints

Iterates rows `row < ne` grouped by `efc_id`. For CONNECT/WELD:
- anchors via `mj_equalityAnchors`; sparse: `NV = mj_mergeChain(b1,b2,0)`, `mj_jacDotSparse(…pos1,b1…)`, `mj_jacDotSparse(…pos2,b2…)`, `jdv_k = J̇_k·qvel` (`mju_dotSparseX3`); dense: `mj_jacDot` + `mju_mulMatVec(3×nv)`. Welds also get rotational `jrdv_k`.
- `result[row+0..2] -= jdv1 − jdv2`; `row += 3`.
- WELD rotational rows: `q0r = xquat[b1]·relpose` (site: `xquat[b1]·site_quat[obj1]`), `negq1 = neg(xquat[b2])` (site: `neg(xquat[b2]·site_quat[obj2])`); `ω_k = cvel[6b_k + 0..2]`, `dω = ω1 − ω2`; `qdot0 = mju_derivQuat(q_full0, ω1)` where `derivQuat(q,ω) = 0.5·(0,ω)⊗q` (spatial/world-frame angular velocity; verify: vector part `q0ω + ω×q_v`):
  ```
  res = 0.5*(-ω0q1-ω1q2-ω2q3,  ω0q0+ω1q3-ω2q2,  -ω0q3+ω1q0+ω2q1,  ω0q2-ω1q1+ω2q0)
  ```
  `qdot0r = qdot0·relpose` (body) or `qdot0` (site); `negqdot1 = neg(derivQuat(q_full1, ω2))`; `djrdv = jrdv1 − jrdv2`;
  ```
  t1 = (negqdot1 ⊗ (0,dω)) ⊗ q0r
  t2 = (negq1    ⊗ (0,djrdv)) ⊗ q0r
  t3 = (negq1    ⊗ (0,dω)) ⊗ qdot0r
  result[row+k] -= 0.5·(t1+t2+t3)[k+1]·torquescale, k = 0..2;  row += 3
  ```
- Other types: skip all rows with the same id.
The J̇v term is not computed for contacts, limits, joint/tendon/flex equalities.

### 13.4 `mj_adhesionRef` (only if `m->flg_adhesion`)

For contacts with `adhesion != 0 && efc_address >= 0`: if `dim == 1 || !pyramidal`: `aref[adr] += R[adr]·adhesion` (normal row only); else `edge = adhesion/(2(dim−1))`, `aref[adr+j] += R[adr+j]·edge` for all `2(dim−1)` edges. Combined with a passive attractive force `δ` (`qfrc_adhesion`, doc on passive), this translates the cone by δ along the normal.

---

## 14. Regularization in the solve metric — `mj_regularizeConstraint(m,d,flg_AR)`

Called at the actuation stage under the discrete integrator (`engine_forward.c` ~2551, `flg_AR=1`) and from inverse dynamics (`flg_AR=0`).
```
if nefc == 0: return
isDual = mj_isDual(m)
if DIAGEXACT || (isDual && flg_AR): mj_makeYNumeric(m, d, DIAGEXACT)
if !DIAGEXACT:
  mj_diagApprox
  if isMetric && d->efm_sdiag:
    for each row r:  num = Σ_a J²/max(mjMINVAL, Mdiag_c + efm_sdiag[c]);  den = Σ_a J²/max(mjMINVAL, Mdiag_c)
                     (skip J == 0; Mdiag_c = M[M_rowadr[c]+M_rownnz[c]-1])
                     if den > mjMINVAL: diagA[r] *= num/den
mj_makeImpedance
if isDual && flg_AR: mj_makeARNumeric
if nisland: gather iefc_D, iefc_R via map_iefc2efc
```

---

## 15. Projection — `mj_projectConstraint`, Y and AR

Purpose: `Y = J·M^{-1/2}` with `M = Lᵀ D L` (MuJoCo's `qLD` factor, L unit lower-triangular following the dof tree), so `Y Yᵀ = J M⁻¹ Jᵀ = A`; `efc_AR = A + diag(R)`. Only computed when needed:

```
if nefc == 0: return
isDual, diagexact
if isMetric:                                   // discrete: symbolic only, numeric deferred
  if isDual || diagexact: mj_makeYSymbolic; if isDual && nefc: mj_makeARSymbolic
  return
if isDual || diagexact: mj_makeYSymbolic; if nefc: mj_makeYNumeric(diagexact)
if diagexact && nefc: mj_makeImpedance; regather island D/R
if isDual && nefc:   mj_makeARSymbolic; if nefc: mj_makeARNumeric
```
(Newton/CG without noslip and without diagexact never build Y or AR.)

**`mj_makeYSymbolic`**: sparse: arena-allocate `efc_Y_rownnz/rowadr`; `nY = computeY_precount(...)`; allocate `efc_Y`, `efc_Y_colind`; under discrete also fill the pattern now (`computeY_fill(NULL, …)`). Dense: `nY = nefc·nv`, allocate `efc_Y`. Any allocation failure → `mjWARN_CNSTRFULL`, `mj_clearEfc`.

**`computeY_precount`**: Y has the pattern of `J·L⁻ᵀ`-closure = union over each J row nonzero `j` of row `j` of M's pattern (all ancestors of `j` incl. `j`). Uses a marker array (`marker[c] = r`) traversing J's row in reverse and skipping already-marked dofs.

**`computeY_fill`**: per row, merge backwards J's column list with the "parent chain" (parent of last added dof `j` = `M_colind[M_rowadr[j] + M_rownnz[j] − 2]` if `M_rownnz[j] > 1`); choose src (J column) when `prev_src ≥ prev_dst` (copy value), else dst (value 0). Verifies count equals precount (`mjERROR` otherwise).

**`computeY_backsub`** (in place): for each row, for `i` from last to first nonzero with `val = Y[i] ≠ 0`, `j = colind[i]`: `Y[ancestors of j] −= L[j, ancestors]·val` (`mju_addToSclSparseInc` with `qLD + M_rowadr[j]`, `M_rownnz[j]−1` off-diagonal entries) — this computes `Y ← Y L⁻¹`; then `Y[i] *= sqrtInvD[colind[i]]`.

**`mj_makeYNumeric(flg_diagexact)`**: returns if `efc_Y == NULL`. `factor = qLD`, or `qH` if `isMetric && efm_diag` (effective-metric backbone). `sqrtInvD[i] = 1/sqrt(factor[M_rowadr[i]+M_rownnz[i]−1])`. Sparse: fill + backsub; dense: `mj_solveM2_impl(efc_Y, efc_J, sqrtInvD, factor, nv, nefc, M_rownnz, M_rowadr, M_colind, dof_simplenum)`. If diagexact: `diagA[i] = ‖Y_i‖²` (`mju_dot`).

**`mj_makeARSymbolic`**: sparse: transpose Y pattern (`mju_transposeSparse` on stack); allocate `efc_AR_rownnz/rowadr`; `nA = mju_sqrMatTDSparseSymbolic(...)` (counting pass, using `Y_rowsuper = efc_J_rowsuper` — Y and J have identical supernodes), allocate `efc_AR`, `efc_AR_colind`, second symbolic pass fills colind. Dense: `nA = nefc²`.

**`mj_makeARNumeric`**: sparse: numeric transpose `YT`, recover diagonal positions from sorted rows, `mju_sqrMatTDSparseNumeric` (`AR = Y Yᵀ`), `AR[diag_i] += R[i]`. Dense: `YT = transpose(Y)`, `mju_sqrMatTD(AR, YT, NULL, nv, nefc)` (= `YTᵀ YT`), `AR[r(nefc+1)] += R[r]`.

---

## 16. Constraint update — `mj_constraintUpdate_impl` / `mj_constraintUpdate`

Inputs: `ne, nf, nefc, D, R, floss = efc_frictionloss, jar` (`= J·qacc − aref`), `type, id, contact`. Outputs: `state, force, cost` (optional), cone Hessians if `flg_coneHessian`.

```
if nefc == 0: cost = 0; return
force[i] = -D[i]·jar[i]  for all i
s = 0
for i:
  if i < ne:                                     // equality
     s += 0.5·D·jar²;  state = QUADRATIC
  elif i < ne+nf:                                // friction loss (Huber)
     if   jar <= -R·floss: s += -0.5·R·floss² - floss·jar; force =  floss; state = LINEARNEG
     elif jar >=  R·floss: s += -0.5·R·floss² + floss·jar; force = -floss; state = LINEARPOS
     else:                 s += 0.5·D·jar²;                               state = QUADRATIC
  elif type != ELLIPTIC:                         // limits, frictionless, pyramidal edges
     if jar >= 0: force = 0; state = SATISFIED
     else:        s += 0.5·D·jar²; state = QUADRATIC
  else:                                          // elliptic cone, rows i..i+dim-1
     mu = con.mu; fr = con.friction; dim = con.dim
     U[0] = jar[i]·mu;  U[j] = jar[i+j]·fr[j-1]   (j=1..dim-1)
     N = U[0];  T = ‖U[1:dim]‖₂  (mju_norm)
     top:    N >= mu·T || (T <= 0 && N >= 0):   force[i..] = 0; state = SATISFIED
     bottom: mu·N + T <= 0 || (T <= 0 && N < 0): s += Σ_j 0.5·D[i+j]·jar[i+j]²; state = QUADRATIC (forces = -D·jar)
     middle: Dm = D[i]/(mu²(1+mu²)); NmT = N - mu·T
             s += 0.5·Dm·NmT²
             force[i]   = -Dm·NmT·mu
             force[i+j] = -force[i]/T · U[j]·fr[j-1]
             state = CONE; (cone Hessian below)
     state[i+j] = state[i]; i += dim-1
cost = s
```
Then `mj_constraintUpdate` computes `qfrc_constraint = Jᵀ·efc_force` (`mj_mulJacTVec`).

Derivation of the middle zone: `s = ½ Dm (N − μT)²` with `U = diag(μ, fr)·jar`; `f = −∂s/∂jar = −diag(μ,fr)·∂s/∂U`, `∂s/∂U₀ = Dm(N−μT)`, `∂s/∂U_j = −Dm(N−μT) μ U_j/T`. Zones: top = `U` inside the (dual) cone → zero cost; bottom = inside the polar cone → full quadratic `½ Σ D jar²`; middle = squared distance to the cone boundary.

**Cone Hessian** (stored in `contact[id].H`, dim×dim, row-major), only in the middle zone:
```
H[0][0] = 1;  H[0][j] = -mu/T · U[j]                                   (j ≥ 1)
H[k][j] = mu·N/T³ · U[j]·U[k]        for 1 ≤ k ≤ j
H[j][j] += mu² - mu·N/T              for j ≥ 1
H[k][j] *= Dm · s_k · s_j  (k ≤ j),  s_0 = mu, s_j = fr[j-1]
symmetrize lower from upper
```
i.e. `H = Dm·diag(s)·[ [1, −μUᵀ/T]; [·, μN/T³ U Uᵀ + (μ² − μN/T) I] ]·diag(s)` = Hessian of `½Dm(N−μT)²` w.r.t. jar.

---

## 17. Programmatic optimizations (C)

- **Two-phase count/instantiate** with exact sparse non-zero pre-count (`mj_ne`, count-only `mj_instantiateFriction/Limit`, `mj_nc`) → single arena allocation, no reallocation; mis-allocation is a hard error in sparse mode.
- **Arena** placement: efc arrays directly after `d->contact` in `d->arena`; `mj_addContact` and `mj_clearEfc` move `parena` back, invalidating efc; stack (`mj_markStack/mjSTACKALLOC/mj_freeStack`) for per-call scratch.
- **Sparse chains**: rows store only the union of dof-ancestor chains (`mj_mergeChain`, increasing order); contacts skip common ancestors (`flg_skipcommon=1`); "simple body" fast path (`mj_mergeChainSimple`, `mj_jacSparseSimple`); all rows of one constraint share the same chain (enables supernodes).
- **Supernodes** (`efc_J_rowsuper`) for faster sparse J·x and AR assembly.
- **Empty-row elimination** for non-contact constraints (all-zero J in dense, NV=0 in sparse).
- **`power()` fast paths** for exponents 1 and 2.
- **Deferred numerics under discrete**: symbolic Y/AR at position stage (all allocation position-stage, so velocity-stage reevaluation is allocation-free), numerics at actuation stage.
- **Islands**: R/D copies gathered into island order (`iefc_*`).
- AVX macro `mjUSEAVX` is defined (`mjUSEPLATFORMSIMD && __AVX__ && !mjUSESINGLE`) but no AVX code path is used inside this file (SIMD lives in the BLAS/sparse helpers called).

---

## 18. GPU (MuJoCo Warp) implementation — `mujoco_warp/_src/constraint.py`

### 18.1 Entry points and call order
- `make_constraint(m, d)` (called from `forward.fwd_position` right after collision, before island/transmission). It both instantiates the rows **and computes D and aref immediately** (`_efc_row`), i.e. Warp fuses C's `mj_makeConstraint` + `mj_diagApprox` + `mj_makeImpedance` + `mj_referenceConstraint` into the instantiation kernels. There is no separate velocity-stage reference computation.
- `regularize_constraint(m, d)` (called from `fwd_actuation` only for the discrete integrator) — per-row metric-ratio correction of D (and adhesion aref) (§18.7).
- Constraint update / cost lives in `solver.py` (`_eval_constraint`, `_update_constraint_efc`).
- No `mj_projectConstraint` equivalent: **PGS is unsupported** (`SolverType` = CG, NEWTON only), `DIAGEXACT` and `OVERRIDE` unsupported (`types.py` comments), so no Y / AR (noslip, if used, is implemented separately in solver.py).

### 18.2 Data layout (`types.Constraint`, per world)
| Field | Shape |
|---|---|
| `type, id, pos, margin, vel, aref, frictionloss, force, island` | `(nworld, njmax)` |
| `D, state` | `(nworld, njmax_pad)` |
| dense `J` | `(nworld, njmax_pad, nv_pad)`; `J_rownnz/J_rowadr` `(nworld, 0)`, `J_colind` `(nworld,0,0)` |
| sparse `J, J_colind` | `(nworld, 1, njmax_nnz)`; `J_rownnz, J_rowadr` `(nworld, njmax)` |
| `jtdaj_adr, jtdaj_nrow` `(nworld, njmax)`, `jtdaj_nblock` `(nworld,)` | row blocks sharing columns (sparse Newton) |
| `Jqvel` `(nworld, njmax)` | warp-only scratch: J·qvel for contacts |
| counters `ne, nf, nl, nefc` | `(nworld,)` |
No `efc_R`, `efc_KBIP`, `efc_diagA`, `efc_b` storage (R = 1/D implicitly), no `contact.mu` (recomputed as `friction[0]·impratio_invsqrt` wherever needed), no `tendon_efcadr`. Warp adds `dof_frictionadr` (efc row of each dof friction or −1).

Precision: Warp `float` = **float32** throughout these kernels (C uses float64). Batched model parameters are indexed `field[worldid % field.shape[0]]`.

Fixed-size buffers: `njmax` rows per world (default heuristic `io._default_njmax`), `njmax_nnz` sparse nonzeros per world (default `_default_njmax_nnz`, or `njmax·nv`), `naconmax` contacts total. `contact.efc_address` is `(naconmax, nmaxpyramid)` (`nmaxcondim` = max condim in model, `nmaxpyramid = max(1, 2(nmaxcondim−1))`) — one address **per row** rather than per contact.

Sparse decision differs: `io.is_sparse`: AUTO → sparse iff `nv > 32` (C: `nv ≥ 60`); explicit DENSE/SPARSE follow `mj_isSparse`.

### 18.3 Row allocation with atomics
Each kernel thread that produces rows does:
1. (sparse) `_reserve_efc_nnz`: `nrow = min(nrow, njmax − nefc[w])`; `rowadr = atomic_add(efc_nnz[w], nrow·rownnz)`; if `rowadr + nnz > njmax_nnz` → return −1 → rows dropped (later `_efc_nnz_overflow` sets `OverflowType.NJMAX_NNZ` in `d.overflow` and optionally prints).
2. `atomic_add(ne|nf|nl[w], size)` (counted even if later dropped), `efcid = atomic_add(nefc[w], size)`; if `efcid >= njmax` → return; rows past njmax are truncated (`nrow = min(size, njmax − efcid)`).
3. (sparse Newton) append a JTDAJ block via `atomic_add(jtdaj_nblock)`.
Consequences: `nefc` may exceed `njmax` (consumers clamp with `min(nefc, njmax)`); row order **within each kernel launch is nondeterministic** (atomic order). Kernel launch order (sequential) is: connect, weld, joint-eq, tendon-eq, flex-eq (if nflex), flexstrain; friction dof, friction tendon; limit ball, limit slide/hinge, limit tendon; contacts. So the efc layout differs from C: **equality rows grouped by type rather than by eq id; ball-joint limits before slide/hinge limits; contacts in atomic order**. `DeterminismType.CONSTRAINT` calls `_sort_constraints`, which is currently a TODO no-op. `ne`/`nf` block structure (equality first, then friction, then limits/contacts) is preserved, which is all the solver needs.

### 18.4 Shared row finalizer `_efc_row` and `_contact_kbimp`
`_contact_kbimp(disableflags, h, solref, solimp, pos_imp, is_discrete)` → `(k, b, imp)`:
- refsafe: `timeconst = max(solref[0], 2h)` if `!REFSAFE disabled && !is_discrete` (applied to timeconst regardless of sign checks; only used in the standard branch).
- clamps: `dmin, dmax, mid ∈ [MINIMP, MAXIMP]`, `width = max(MINVAL, width)`, `power = max(1, power)`.
- `k = solref[0] <= 0 ? −solref[0]/max(MINVAL,dmax²) : 1/max(MINVAL, dmax²·tc²·dr²)`; `b = solref[1] <= 0 ? −solref[1]/max(MINVAL,dmax) : 2/max(MINVAL, dmax·tc)` (note: `<= 0` tests vs C's `> 0` tests; a zero component picks the direct branch in both).
- impedance: `x = |pos_imp|/width`; flat if `dmin == dmax || width_raw <= MINVAL`; `x<=0 → dmin`; `x>=1 → dmax`; `power==1` linear; poly branches identical to C but **result clamped** `wp.clamp(imp, dmin, dmax)`. No mixed-sign solref sanitization (C replaces with defaults + warning). No `impP` computed.

`_efc_row(..., pos_aref, pos_imp, invweight, solref, solimp, margin, vel, frictionloss, type, id, is_discrete, aref_shift)`:
```
k,b,imp = _contact_kbimp(...);  k = 0 if type is FRICTION_DOF/FRICTION_TENDON
continuous: D = 1/max(invweight·(1-imp)/imp, MINVAL);  aref = -k·imp·pos_aref - b·vel - aref_shift
discrete:   refsafe on LIMIT_*/CONTACT_* with solref[0]>0 and k>0 (same formula as C)
            f = 1 + h·b + h²·k·imp
            R = max(max(MINVAL,(1-imp)·invweight/imp)/f, max(MINVAL,(1-MAXIMP)·invweight/MAXIMP))
            D = 1/R;  aref = (-k·imp·pos_aref - (b + h·k·imp)·vel - aref_shift)/f
vel_out = vel;  pos_out = pos_aref + margin;  margin_out = margin;  frictionloss_out; type; id
```
`pos_aref` is the already margin-subtracted position (`dist − margin`), `aref_shift = J̇v` for connect/weld (Warp subtracts it inside the same expression, before division by f — equivalent to C).

### 18.5 Per-type kernels (launch dims)
- `_equality_connect` `(nworld, n_connect)`: anchors as C; sparse: maps `body_k = body_weldid[body_k]` **before** computing invweight (C uses the original body / site_bodyid → `body_invweight0` of a welded child may differ); walks the merged chain from the highest dof down (`da = max(da1,da2)`; no common-dof skip) → **colind stored in decreasing dof order** (C: increasing). Computes `jac_dof` (`jacp = cdof_lin + cdof_ang × (p − subtree_com[root])`) and `jac_dot_dof` per dof, accumulating `Jqvel` and `Jdotv` in registers. Dense: loop over all `nv` dofs. `pos_imp = |pos|`.
- `_equality_weld`: as C (body semantic `pos1 = xmat1·data[3:6]+xpos1`, `pos2 = xmat2·data[0:3]+xpos2`); rotational Jacobian computed as `0.5·vec((quat1 ⊗ (0, torquescale·(jr1−jr2))) ⊗ quat)` (torquescale applied before the quaternion map — mathematically equal to C's post-scaling); `pos_imp = sqrt(|cpos|² + |crot|²)`; translational rows use `invweight_t = iw0[b1].x + iw0[b2].x`, rotational `iw0[b1].y + iw0[b2].y`; J̇v rotational terms t1, t2, t3 exactly as C. Uses `math.mul_quat(omega_q, q)·0.5` for `qdot` = `0.5·(0,ω)⊗q`, identical to C's `mju_derivQuat`. Site semantics: `qdot0r = 0.5·(0,ω1)⊗(xquat1·site_quat1)` (same as C).
- `_equality_joint`: polynomial via **Horner**: `rhs = d0 + dif·(d1 + dif·(d2 + dif·(d3 + dif·d4)))`, `deriv = d1 + dif·(2d2 + dif·(3d3 + dif·4d4))`; `pos = qpos1 − qpos0_1 − rhs`. Sparse colind `[dofadr1, dofadr2]` unsorted, rownnz 2 even if `deriv == 0` or both joints share a dof.
- `_equality_tendon`: explicit powers like C but `pos = pos1 − (d0 + d1·dif + …)` (different association); J₂ merged only if `deriv != 0`; sparse path writes only nonzero `J` (exact zeros dropped, rownnz may be smaller than reserved).
- `_equality_flex` `(nworld, n_eqflex, nflexedge)`: skip if `flex_interp != 0`, edge outside flex range, or rigid; copies `flexedge_J`; invweight `flexedge_invweight0[e]`.
- `_equality_flexstrain`: **order-1 only** (`npc = 4` shell / `8` solid; C supports order 2); uses precomputed `flexstrain_J_*` patterns, `support.compute_interp_cell_quat` or `face_quat`, Warp-native xyzw quaternion ops (`wp.quat_rotate`); avg invweight as C.
- `_friction_dof` `(nworld, nv)`: skip `frictionloss <= 0`; no JTDAJ block (sparse Newton adds friction-dof rows to the Hessian diagonal directly); writes `dof_frictionadr`.
- `_friction_tendon` `(nworld, ntendon)`.
- `_limit_slide_hinge` `(nworld, n_limited_slide_hinge)`: `pos = min(qpos − r0, r1 − qpos) − margin`; active iff `pos < 0`; `J = +1 if dist_min < dist_max else −1`. **Only one row per joint (the nearer side)** — C emits two rows when both sides are within margin.
- `_limit_ball`: `quat_to_vel(normalize(q))`, `normalize_with_norm` (zero vector → axis 0, C → (1,0,0)); `pos = max(r0,r1) − angle − margin`.
- `_limit_tendon`: one row (nearer side), like slide/hinge.
- Contacts (four kernels):
  - `_efc_contact_init` `(naconmax)`: skip `conid >= nacon` or not `ContactType.CONSTRAINT`; active iff `dist − includemargin < 0` (or `adhesion != 0` when `flg_adhesion`); `ndim = condim` (elliptic) / `1` or `2(condim−1)` (pyramidal); sparse rownnz = merged chain length **with common-dof skip**; contiguous block of `ndim` rows via one `atomic_add`; writes `contact.efc_address[conid, dim]` (−1 for dropped rows), `efc_id`, zeroes `Jqvel`. Contacts whose bodies share all dofs are *not* excluded (C sets `exclude=3`); in sparse mode they get rownnz 0.
  - `_efc_contact_jac_sparse`: 8 lanes per contact (`_CONTACT_JAC_LANES = 8`); lane `k` computes nonzeros `k, k+8, …` for **all rows** of the contact (coalesced stores); grid-strided over contacts (`ncontact_per_pass = min(naconmax, 1024·SM count)`); per-lane partial `Jqvel` then one `atomic_add` per row (nondeterministic sum order unless `DeterminismType.ATOMICS`, which enables Warp's run-to-run deterministic atomics with `max_records`).
    Row values: elliptic `J = frame[dimid]·Δjacp` (dimid<3) or `frame[dimid−3]·Δjacr`; pyramidal `J = frame0·Δjacp ± fri[dimid2−1]·(frame[dimid2]·Δjacp or frame[dimid2−3]·Δjacr)`, `dimid2 = dimid/2 + 1`, `+` for even dimid. `Δjac = jac(body2) − jac(body1)` (same sign as C).
  - `_efc_contact_jac_dense` (`launch_tiled`, dims `(nworld, ceil(nv_pad/TILE))`, block_dim = `m.block_dim.contact_jac_tiled`): per world, per dof tile, iterates rows `[ne+nf+nl, min(nefc,njmax))`, recomputing per-contact Jacobian tiles only when `efc_id` changes; `tile_atomic_add` of tile-reduced `J·qvel` into `Jqvel`.
  - flex variants (`_efc_contact_init_flex`, `_efc_contact_jac_sparse_flex`, `_efc_contact_jac_dense_flex`, `_efc_contact_update_flex`) handle flex vertex/element/interpolated contacts with up to 16 de-duplicated weighted bodies (`_add_weight`, weights `< 1e-10` dropped; trilinear basis weights `≤ 1e-5` dropped) — C allows up to 729 bodies.
  - `_add_surface_vel` (if `flg_surfacevel`) adds surface velocity to `Jqvel` (same formulas as C) before the update kernel, so `efc_vel` and aref include it.
  - `_efc_contact_update` `(naconmax, nmaxdim)`: one thread per row. `pos = dist − includemargin`; `invweight = iw0[geom body1].x + iw0[geom body2].x` (translational only; geom body, not weld root — same as C).
    - pyramidal (condim>1): `invweight ← (invweight + fri0²·invweight)·2·fri0²·impratio_invsqrt²` — this is exactly C's `Rpy = 2μ²R₀` with `R₀ ∝ tran(1+fri0²)` folded into the invweight (C applies the `max(MINVAL,·)` clamp before the `2μ²` scaling; Warp after).
    - elliptic row 0: normal `_efc_row`. Rows `dimid>0`: `pos_aref = 0`, `_efc_row` with the *normal* solref (so D includes the normal row's discrete factor f, as in C), then `D /= invweight_scale` with `invweight_scale = impratio_invsqrt²·(dimid>1 ? fri0²/fri_{dimid−1}² : 1)`; aref overwritten: `aref = −b_fri·Jqvel / f_fri`, `b_fri` from `solreffriction` if nonzero else solref, `f_fri = 1 + h·b_fri` (discrete) else 1.
    - Elliptic friction rows store `efc_pos = includemargin`, `efc_margin = includemargin` (C stores 0, 0) — cosmetic, K·pos term is zero.
    - adhesion (`flg_adhesion`): for the normal row (elliptic/frictionless) or each pyramidal edge (`adhesion/(2(condim−1))`): `aref += (1/D)·adhesion` (= R·δ, after the discrete division, as C).
- No sleep filtering in any constraint kernel (C skips sleeping equality/dof/joint rows); no `exclude=4` passive-flex marking in this file.

### 18.6 Jacobian-derivative term timing (parity concern)
Warp computes `J̇v` inside `make_constraint` (position stage) from `d.cvel` and `d.cdof_dot`, but `smooth.com_vel` (which fills those) runs later in `fwd_velocity`. Therefore, in `forward`, the J̇v correction for connect/weld uses **cvel/cdof_dot left over from the previous call** (paired with the current `qvel`), whereas C evaluates it in the velocity stage with fresh values. Matching C exactly requires recomputing com_vel before make_constraint (or moving the J̇v term to the velocity stage).

### 18.7 `regularize_constraint` (discrete only)
Kernel `_regularize_constraint` `(nworld, njmax_pad)`: for row `r < nefc` (for contact rows, uses the first row `efc_address[conid,0]`'s Jacobian for every row of the contact — matching C, where all friction rows derive from the first row): `num = Σ J²/qH_cc`, `den = Σ J²/M_cc` (`math.safe_div`: divisor 0 → MINVAL; C uses `max(MINVAL, ·)`), skip `J == 0`; if `den > MINVAL`: `D ← min(D/(num/den), 1/MINVAL)`; adhesion rows: `aref += (1/D_new − 1/D_old)·adhesion` (per-edge share for pyramidal). Runs whenever the integrator is discrete (C only when `efm_sdiag` exists). This is valid because both `R0/f` and `Rmin` scale linearly with diagA.

### 18.8 Constraint update (solver.py) differences
`_eval_constraint` mirrors §16 with: friction thresholds `rf = frictionloss/D` (vs C `R·floss`), costs `−floss·(0.5·rf ± jaref)`; elliptic `mu = friction[0]·impratio_invsqrt`, `T = sqrt(TT)` from a manual sum of squares (C `mju_norm`), middle zone `Dm = safe_div(D0, mu²(1+mu²))`, tangent force `−safe_div(f_n, T)·U_j·fr_j`, cost attributed to the normal row only; one thread per row (each tangent row recomputes `TT` for its contact). Optional change tracking (`quad_changed_ids`, `state_changed_count`) for incremental Hessian updates.

### 18.9 How to match Warp
Use float32, Warp's row order (type-grouped kernels, atomic order within), single-sided limits, Horner for joint equality, `nv > 32` sparsity rule, njmax/njmax_nnz truncation, the stale-cvel J̇v, and the per-row D formulas above. Exact bitwise match of row order requires replaying atomics in the same order (not guaranteed); compare results after permuting rows by (type, id).

---

## 19. MJX (JAX) — `mjx/_src/constraint.py`

- **Static shapes**: `make_efc_type` allocates rows for every *potential* constraint: all equalities of type CONNECT(3)/WELD(6)/JOINT(1)/TENDON(1) (flex not supported), all frictional dofs/tendons, **one row per limited joint/tendon** (`jnt_limited.sum()`), and all contacts from `collision_driver.make_condim` grouped by condim (1, 3, 4, 6). Inactive rows are kept with `J = 0, pos = 0` (`x * active`) so they produce zero force (jar = 0 → satisfied / zero).
- Row order: connect, weld, joint, tendon (type-grouped), friction (dof then tendon), ball limits, slide/hinge limits, tendon limits, frictionless contacts, then condim 3, 4, 6 contacts. `make_efc_address` computes contact row addresses from this layout.
- Dense J only (`efc_J` (nefc, nv)); per-type `jax.vmap` row builders returning `_Efc(J, pos_aref, pos_imp, invweight, solref, solimp, margin, frictionloss)`, concatenated, then one vmapped `fn`: `k,b,imp = _kbi(...)`, `r = max(invweight(1−imp)/imp, MINVAL)`, `aref = −b·(J·qvel) − k·imp·pos_aref`, `D = 1/r`.
- `_kbi`: refsafe always applied (no discrete integrator support); impedance via `jp.where` of both polynomial branches (`imp_x < mid` strict), clipped to `[dmin, dmax]`, `imp_x > 1 → dmax`; no flat-function branch (equivalent except `width ≈ 0`); K not zeroed for friction rows (harmless since `pos_aref = 0`).
- Missing vs C: J̇v correction for connect/weld, adhesion, surface velocity, discrete integrator, flex equalities/contacts, sleep, override, mixed-solref sanitization; pyramidal invweight uses `/m.opt.impratio` (no MINVAL clamp); elliptic friction rows use `invweight/impratio·fri0²/fri_j²` and `solreffriction` (or solref) — friction-row D computed from the same imp as the normal row (pos_imp = dist − margin for all rows).
- Precision: float32 unless JAX x64 is enabled.

---

## 20. Reproduction pitfalls / parity notes

1. **Row ordering** differs among C (by eq id; joint limits interleaved by joint id; contacts by index), Warp (type-grouped kernels, atomic order) and MJX (static type-grouped). Iterative solvers (PGS sweeps, CG/Newton reductions) are order-sensitive at the rounding level; compare per (type,id).
2. **Two-sided limits**: C can emit 2 rows for one joint/tendon (both `dist < margin`); Warp and MJX emit only the nearer side.
3. **Impedance clamping**: C does not clamp `imp` to `[dmin,dmax]` (dmin > dmax allowed: decreasing curve). Warp/MJX `clamp(imp, dmin, dmax)` — with `dmin > dmax`, `wp.clamp`/`jp.clip` return `dmax`-dominated values, changing behaviour.
4. **R clamp placement**: C clamps `R ≥ mjMINVAL` per row *before* the pyramidal `2μ²` / elliptic `1/impratio·μ₀²/μ_j²` scaling; Warp/MJX clamp after folding the factors into invweight.
5. **diagA is overwritten** at the end of `mj_makeImpedance` (`diagA = R·I/(1−I)`), so after the call it reflects the friction-adjusted R; code reading `efc_diagA` must not assume it is the raw approximation.
6. **refsafe**: applied only to `solref[0] > 0` (standard format) and not under the discrete integrator; under discrete a separate refsafe modifies K and B of contact/limit rows when `h²KI > 1`. The impedance derivative `impP` (KBIP[3]) is only stored in C.
7. **Mixed-sign solref** (`(a>0) XOR (b>0)`) is replaced by `(0.02, 1)` in C (solreffriction by `(0,0)`); Warp/MJX do not sanitize.
8. **Elliptic friction rows**: K = 0, impedance = the normal row's (shared pos), B from solreffriction if either component nonzero, R derived from normal R (`/impratio`, `·μ₀²/μ_j²`). `contact.mu = friction[0]/sqrt(max(impratio, MINVAL))` (computed as `friction[0]·sqrt(R1/R0)` — rounding of the sqrt of a ratio vs. Warp's precomputed `1/sqrt(impratio)`).
9. **Pyramidal**: all edges share pos = dist, one impedance, one R = `2μ²R₀`, `R₀` from the first edge's `diagA = tran(1 + μ₀²)` (uses `friction[0]` only; dA for other edges is computed but overwritten).
10. **Weld anchors are swapped** relative to connect (`pos1` uses `data[3:6]`).
11. **Weld J̇v** uses `mju_derivQuat(q, ω) = 0.5·(0,ω)⊗q` with ω taken from the `cvel` angular part; reproduce the exact three-term product rule and torquescale placement (C applies torquescale after the 0.5·vec(...) sum; Warp scales the Jacobian difference before the quaternion map — equal in exact arithmetic).
12. **J̇v only for connect/weld**; never for contacts, limits, joint/tendon polynomials, flex.
13. **Joint-polynomial arithmetic**: C uses explicit products `data[1]*dif + data[2]*dif*dif + …` and `cpos = ((pos−ref)−data0) − Σ`; Warp uses Horner (joint) or a different association (tendon) → last-bit differences.
14. **Sparse colind order** is increasing in C (merge chain reversed); Warp connect/weld/contact rows are stored in decreasing order and joint-eq rows unsorted. Any code consuming CSR must not assume sortedness when matching Warp; C's sparse helpers *do* assume sorted indices.
15. **Contacts with no relative dofs**: C excludes them (`exclude=3`, `efc_address=-1`, no rows); Warp keeps them with zero (dense) or empty (sparse) Jacobian rows.
16. **Dense empty-row elimination**: a non-contact constraint whose Jacobian is entirely zero is dropped in dense mode (and in sparse mode iff NV=0). Because `mj_diagApprox` for FLEX equality writes one entry per non-rigid edge assuming contiguous rows, dropping an edge row in dense mode would misalign diagA — avoid such models when matching.
17. **Sleep**: C skips sleeping equality/dof-friction/joint-limit rows, but *not* tendon friction / tendon limits. Warp does no sleep filtering in constraint construction.
18. **tendon_efcadr** for tendon equalities stores the equality id `i`, not an efc row (quirk).
19. **Discrete integrator**: R divided by `f = 1 + hB + h²KI` and floored at `Rmin = (1−MAXIMP)/MAXIMP·diagA`; aref divided by the same f *after* subtracting J̇v and *before* adding the adhesion bias. D/R/AR numerics are computed at the actuation stage after the metric correction `diagA *= num/den`.
20. **Sparsity threshold**: C AUTO → sparse at `nv ≥ 60`; Warp at `nv > 32`. Dense/sparse produce mathematically identical J but different summation orders.
21. **Friction-loss thresholds**: C compares `jar` with `±R·floss` using stored R; Warp with `floss/D`; MJX similar to Warp. Values at the switching boundary may classify differently.
22. **Warp truncation**: rows beyond `njmax` or sparse storage beyond `njmax_nnz` are silently dropped (overflow flag); `ne/nf/nl` still count dropped rows; C instead fails the whole allocation (`mjWARN_CNSTRFULL`, efc cleared, no constraints).
23. **Float precision**: Warp float32 (and MJX default float32) vs C float64; impedance sigmoid near `x ≈ 0 or 1`, `mjMINVAL` clamps (1e-15 is below float32 resolution for typical magnitudes) and `D = 1/R` with tiny R behave differently.
24. **Warp J̇v staleness** (§18.6): uses previous-step `cvel`/`cdof_dot`.
25. **Active-contact criterion**: rows exist iff `dist < includemargin` (strict) or adhesion ≠ 0 (or speculative contacts in C); the impedance is evaluated at `r = dist − includemargin ≤ 0` with `|r|` (both signs symmetric in `getimpedance`).

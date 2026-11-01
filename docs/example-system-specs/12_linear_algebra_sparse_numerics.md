# 12 — Linear Algebra, Sparse Formats and Numerical Foundations

**Scope.** Dense BLAS-like vector/matrix routines (and their AVX paths), the sparse CSR toolkit
(supernodes, transpose, merge/combine, `M'·D·M` squaring), dense / banded / sparse Cholesky,
rank-1 updates, LU (dense, 6×6, sparse tree), 3×3 solves and Jacobi eigen-decomposition,
QCQP, box-QP, sort macros, miscellaneous scalar utilities, allocation alignment, compile flags
and floating-point determinism; and the GPU counterparts in MuJoCo Warp (MJWarp) and briefly MJX.

**Source files covered**

| Engine | Files |
|---|---|
| C (commit 7ba52b5) | `src/engine/engine_util_blas.c/.h`, `engine_util_blas_avx.h`, `engine_util_sparse.c/.h`, `engine_util_sparse_avx.h`, `engine_util_solve.c/.h`, `engine_sort.h`, `engine_util_misc.c/.h` (non-muscle, non-flex parts), `engine_util_errmem.c/.h`, `engine_array_safety.h`, `engine_crossplatform.h/.cc`, `engine_memory.h` (alloc macros only), `include/mujoco/mjtype.h`, `mjmodel.h` (constants), `CMakeLists.txt`, `cmake/MujocoOptions.cmake`, `cmake/CheckAvxSupport.cmake`, `cmake/MujocoHarden.cmake`, `BUILD.bazel`; reference: `src/user/user_util.cc:mjuu_eig3` |
| MJWarp (commit d41a47c) | `mujoco_warp/_src/math.py`, `block_cholesky.py`, `warp_util.py`, `determinism_test.py`, plus the numerics-relevant parts of `smooth.py` (`factor_m`, `solve_LD`), `solver.py` (Cholesky launches, fast-math), `io.py` (`_get_padded_sizes`, `m_block_layout`, qLD update levels), `types.py` (constants, `DeterminismType`), `util_misc.py` (halton/sigmoid/wrap ports) |
| MJX | `mjx/mujoco/mjx/_src/math.py`, `smooth.py:factor_m/solve_m` |

**Role in the pipeline.** Every stage of `mj_step` calls these routines: kinematics (3×3/quaternion
helpers, `mju_normalize4`), CRB/mass matrix factorization (`mj_factorI` uses `mju_addToScl`/`mju_scl`,
`mj_solveLD` uses `mju_dotSparse`), tendon/actuator velocities (`mju_mulMatVecSparse`), constraint
assembly (`mju_transposeSparse`, `mju_superSparse`, `mju_sqrMatTDSparse*`, `mju_addChains`,
`mju_combineSparseInc`), the primal solvers (Newton: `mju_sqrMatTD_impl`/`mju_sqrMatTDSparseNumeric`
+ `mju_cholFactor`/`mju_cholFactorSymbolic`/`mju_cholFactorNumeric`/`mju_cholUpdate(Sparse)`/
`mju_cholSolve(Sparse)`; PGS: `mju_QCQP2/3/n`), implicit integrators (`mju_factorLU6`, `mju_factorLUSparse`),
broadphase (`mju_eig3Tol`, `mjSORT`), tendon wrapping (`mju_wrap`), and SDF collisions (`mju_Halton`).
Bitwise reproduction of MuJoCo therefore requires reproducing the **exact operation order** of these
routines, which this document specifies.

---

## 0. Global constants and types

From `include/mujoco/mjtype.h`:

```c
#ifndef mjUSESINGLE
  typedef double mjtNum;
  #define mjMINVAL 1E-15     // minimum value in any denominator
  #define mjMAXVAL 1E+10     // maximum value in qpos, qvel, qacc
#else
  typedef float mjtNum;
  #define mjMINVAL 1E-15f
  #define mjMAXVAL 1E+10f
#endif
#define mjPI 3.14159265358979323846
typedef unsigned char mjtByte;  typedef _Bool mjtBool;  typedef int64_t mjtSize;
```

From `include/mujoco/mjmodel.h`:

| Constant | Value | Meaning |
|---|---|---|
| `mjMINMU` | `1E-5` | minimum friction coefficient |
| `mjMINIMP` | `0.0001` | minimum constraint impedance |
| `mjMAXIMP` | `0.9999` | maximum constraint impedance |
| `mjMAXCONPAIR` | `50` | max contacts per geom pair |
| `mjMAXTREEDEPTH` | `50` | max BVH depth |
| `mjMAXFLEXNODES` | `27` | max flex nodes |
| `mjMINAWAKE` | `10` | min timesteps before sleeping |
| `mjNEQDATA` 12, `mjNDYN` 10, `mjNGAIN` 10, `mjNBIAS` 10, `mjNFLUID` 12, `mjNREF` 2, `mjNIMP` 5, `mjNPOLY` 2, `mjNSENS` 3, `mjNSOLVER` 200, `mjNISLAND` 20 | | array sizes |

`mjmacro.h`: `mjMAX(a,b) (((a) > (b)) ? (a) : (b))`, `mjMIN(a,b) (((a) < (b)) ? (a) : (b))`
(double evaluation!). Function versions (`engine_util_misc.c`):

```c
mju_min(a,b)  = a <= b ? a : b;      // ties -> a
mju_max(a,b)  = a >= b ? a : b;      // ties -> a ; any NaN operand makes the test false -> returns b
mju_clip(x,lo,hi) = x < lo ? lo : (x > hi ? hi : x);   // NaN passes through
mju_sign(x)   = (x > 0.0) - (x < 0.0);                 // returns 0 for 0 and NaN
mju_isBad(x)  = (x != x || x > mjMAXVAL || x < -mjMAXVAL);
```

Note on NaN semantics: `mju_max(NaN, b)` → `NaN >= b` is false → returns `b`; `mju_max(a, NaN)` →
`a >= NaN` false → returns `NaN`. Reimplementations using `fmax` (NaN-suppressing in both
arguments) or Warp `wp.max` (CUDA `fmaxf`) differ for NaN inputs.

Standard math macros (`engine_util_blas.h`) map to libm: double build `mju_sqrt=sqrt, mju_exp=exp,
mju_sin, mju_cos, mju_tan, mju_asin, mju_acos, mju_atan2, mju_tanh, mju_pow, mju_abs=fabs, mju_log,
mju_log10, mju_floor, mju_ceil`; single build uses the `*f` variants. **Transcendentals are therefore
platform-libm-dependent** (glibc vs. Apple libm vs. MSVC CRT give different last bits for
`exp/sin/cos/atan2/pow`), while `sqrt`, `fabs`, `floor`, `ceil` are correctly rounded everywhere
(IEEE-754 requires `sqrt` correct rounding).

Precision-dependent tolerances elsewhere in the engine (for reference; used under `#ifdef mjUSESINGLE`):
`eigTOL 4e-15 / 2e-6f`, `eigEPS 1e-12 / 1e-5f` (solve.c), `rotEPS 1e-9 / 1e-6f` (`mju_mat2Rot`),
`bortzCoef` polynomial coefficient sets (spatial.c), metric CG tolerance floor `max(tol,1e-5)` in single,
SVD/Jacobi `reltol 4e-15 / 2e-6f`, box-box `mjBOXBOX_SEPEPS 1e-13 / 1e-6f`, `PAREPS 1e-16/1e-7f`,
`SGNEPS 1e-9/1e-5f`, GJK `mjMAX_LIMIT DBL_MAX/FLT_MAX`.

---

## 1. Dense BLAS (`engine_util_blas.c`, `engine_util_blas_avx.h`)

### 1.1 AVX activation

`engine_util_blas_avx.h` and `engine_util_sparse_avx.h`:

```c
#ifdef mjUSEPLATFORMSIMD
#if defined(__AVX__) && !defined(mjUSESINGLE)
#define mjUSEAVX
#include <immintrin.h>
...
```

- `mjUSEPLATFORMSIMD` is defined PUBLIC on target `mujoco` when CMake option
  `MUJOCO_ENABLE_AVX_INTRINSICS=ON` (default ON).
- `__AVX__` is defined by the compiler when `-mavx` (GCC/Clang) or `/arch:AVX` (MSVC) is passed;
  CMake option `MUJOCO_ENABLE_AVX=ON` (default) adds those flags if a test program with `__m256d`
  compiles (`cmake/CheckAvxSupport.cmake`; on Windows non-MSVC also `-mpclmul`).
- **Only AVX (not AVX2, not FMA) is enabled.** No `-mfma`/`-march` is passed, so on x86-64 the compiler
  cannot emit FMA instructions; all intrinsics are separate `_mm256_mul_pd` + `_mm256_add_pd`.
- AVX paths exist only for double precision. `engine_support.c` and `engine_core_constraint.c` define
  `mjUSEAVX` the same way for their own local intrinsics.
- `engine_crossplatform.cc`: on macOS x86_64 AVX builds a constructor `CheckRosetta()` aborts
  (`ud2`/SIGILL) if `sysctl.proc_translated` is set ("cannot be run under Rosetta 2").

Every AVX routine processes blocks of 4 doubles with unaligned loads/stores (`loadu/storeu`), then a
scalar tail of `n mod 4` elements written out explicitly for tail 3, 2, 1. Element-wise routines are
bit-identical to the scalar loops (same per-element operations, no reassociation). Only the
reductions (`dot`, `dotSparse`, `dotSparseX3`) change summation order; MuJoCo makes the scalar `mju_dot`
and `mju_dotSparse` follow the **same** lane order as AVX (see 1.4), except `mju_dotSparseX3` (see 2.4).

### 1.2 3D / 4D fixed-size routines

All are straightforward; the exact expression order (left-to-right evaluation, C semantics,
no contraction on x86 builds) is:

| Function | Formula (exact order) |
|---|---|
| `mju_zero3(res)` | `res[i]=0` |
| `mju_equal3(a,b)` | `|a0-b0|<mjMINVAL && |a1-b1|<mjMINVAL && |a2-b2|<mjMINVAL` (absolute tol) |
| `mju_copy3`, `mju_copy9` | element copy |
| `mju_scl3(res,v,s)` | `res[i]=v[i]*s` |
| `mju_add3`, `mju_sub3`, `mju_addTo3`, `mju_subFrom3` | elementwise |
| `mju_addToScl3(res,v,s)` | `res[i] += v[i]*s` |
| `mju_addScl3(res,a,b,s)` | `res[i] = a[i] + s*b[i]` |
| `mju_norm3(v)` | `sqrt(v0*v0 + v1*v1 + v2*v2)` |
| `mju_dot3(a,b)` | `a0*b0 + a1*b1 + a2*b2` = `((a0b0 + a1b1) + a2b2)` |
| `mju_dist3(p,q)` | `d=p-q; sqrt(d0*d0+d1*d1+d2*d2)` |
| `mju_normalize3(v)` | `norm=sqrt(v·v)`; if `norm < mjMINVAL`: `v=(1,0,0)`; else `inv=1/norm; v[i]*=inv` (multiply by reciprocal, **not** divide). Returns pre-norm |
| `mju_mulMatVec3(res,M,v)` | row-major: `res_r = M[3r]*v0 + M[3r+1]*v1 + M[3r+2]*v2`, via tmp (alias-safe) |
| `mju_mulMatTVec3` | `res_c = M[c]*v0 + M[3+c]*v1 + M[6+c]*v2`, via tmp |
| `mju_mulMatMat3(res,A,B)` | `res[3i+j] = A[3i]*B[j] + A[3i+1]*B[3+j] + A[3i+2]*B[6+j]` (**not alias-safe**) |
| `mju_mulMatTMat3` | `res[3i+j] = A[i]*B[j] + A[3+i]*B[3+j] + A[6+i]*B[6+j]` |
| `mju_mulMatMatT3` | `res[3i+j] = A[3i]*B[3j] + A[3i+1]*B[3j+1] + A[3i+2]*B[3j+2]` |
| `mju_zero4`, `mju_unit4` (`1,0,0,0`), `mju_copy4` | trivial |
| `mju_normalize4(q)` | `norm=sqrt(q0²+q1²+q2²+q3²)` (left-to-right sum); if `norm<mjMINVAL`: `q=(1,0,0,0)`; **else if `|norm-1| > mjMINVAL`**: `inv=1/norm; q[i]*=inv`; (if already unit within 1e-15, untouched). Returns norm |

`mji_mulMatMat3` / `mji_mulMatTMat3` (`engine_inline.h`) are inlined copies with identical arithmetic
(assert no aliasing).

### 1.3 General vector routines

| Function | Semantics / order |
|---|---|
| `mju_zero(res,n)` | `memset` (produces +0.0) |
| `mju_zeroInd(res,n,ind)` | `res[ind[i]]=0` |
| `mju_fill(res,val,n)` | loop |
| `mju_copy(res,vec,n)` | `memcpy` (must not overlap) |
| `mju_copyInd(res,vec,ind,n)` | `res[ind[i]]=vec[ind[i]]` |
| `mju_sum(vec,n)` | sequential `res += vec[i]` from 0 (**not** 4-lane) |
| `mju_L1(vec,n)` | sequential `res += |vec[i]|` |
| `mju_scl(res,vec,s,n)` | `res[i]=vec[i]*s` (AVX same) |
| `mju_add`, `mju_sub`, `mju_addTo`, `mju_subFrom` | elementwise (AVX same) |
| `mju_addInd`, `mju_subInd`, `mju_addToInd` | at indices `j=ind[i]` |
| `mju_addToScl(res,vec,s,n)` | `res[i] += vec[i]*s` (AVX: `res + (vec*s)`) |
| `mju_addToSclInd(res,vec,ind,s,n)` | `res[k] += vec[k]*s`, `k=ind[i]` |
| `mju_addScl(res,a,b,s,n)` | `res[i] = a[i] + b[i]*s` |
| `mju_normalize(res,n)` | `norm=sqrt(mju_dot(res,res,n))`; `<mjMINVAL` → `res=(1,0,…,0)`; else `inv=1/norm; res[i]*=inv` |
| `mju_norm(res,n)` | `sqrt(mju_dot(res,res,n))` |
| `mju_dotInd(a,b,ind,n)` | sequential single accumulator over `k=ind[i]` |

### 1.4 `mju_dot` — the canonical reduction order (critical)

Scalar build (and identical AVX build):

```
r0=r1=r2=r3=0; i=0
for (; i <= n-4; i += 4):  r0 += a[i]b[i]; r1 += a[i+1]b[i+1]; r2 += a[i+2]b[i+2]; r3 += a[i+3]b[i+3]
res = (r0 + r2) + (r1 + r3)
tail t = n - i:
  t==3: res += a[i]b[i] + a[i+1]b[i+1] + a[i+2]b[i+2]     # tail summed left-to-right FIRST, then added
  t==2: res += a[i]b[i] + a[i+1]b[i+1]
  t==1: res += a[i]b[i]
```

AVX version: `sum = mul(a[0:4], b[0:4])` (first block initialises — identical to `0 + x` except the sign
of a zero), loop `sum = add(sum, mul(...))`, horizontal reduction `vlow=(s0,s1)`, `vhigh=(s2,s3)`,
`vlow+vhigh = (s0+s2, s1+s3)`, then `(s0+s2) + (s1+s3)`; the same tail. For `n<4`: `res = 0 + tail`.

Consequences:
- `mju_dot(a,b,3)` = `0 + (a0b0 + a1b1 + a2b2)` which equals `mju_dot3` bitwise (adding +0 is exact,
  except `-0` becomes `+0`).
- For `n ≥ 4` the result differs from naive sequential summation; any reimplementation must use this
  4-lane / pairwise-final / grouped-tail order everywhere `mju_dot` is used (dense Cholesky, cholSolve,
  `mju_mulMatVec`, `mju_mulMatMatT`, `mju_mulVecMatVec`, band Cholesky, boxQP, `mju_normalize`).

### 1.5 Matrix-vector and matrix routines (row-major)

| Function | Algorithm |
|---|---|
| `mju_mulMatVec(res,M,v,nr,nc)` | `res[r] = mju_dot(M + r*nc, v, nc)` |
| `mju_mulMatTVec(res,M,v,nr,nc)` | `res=0`; for `r=0..nr-1`: if `v[r]!=0`: `mju_addToScl(res, M+r*nc, v[r], nc)` (skips exact zeros; accumulation order = row order) |
| `mju_mulVecMatVec(a,M,b,n)` | `res=0; for i: res += a[i]*mju_dot(M+i*n, b, n)` |
| `mju_transpose(res,M,nr,nc)` | `res[j*nr+i] = M[i*nc+j]` (not in-place) |
| `mju_symmetrize(res,M,n)` | diag copied; `res[i,j]=res[j,i]=0.5*(M[i,j]+M[j,i])` for `j<i` (note `0.5*(…)`, not `/2`) |
| `mju_eye(M,n)` | zero then diag 1 |
| `mju_copyRows(res,M,ind,n,nc)` | copy listed rows |
| `mju_mulMatMat(res,A,B,r1,c1,c2)` | `res=0`; for `i`, for `k=0..c1-1`: if `A[i,k]!=0`: `mju_addToScl(res+i*c2, B+k*c2, A[i,k], c2)` |
| `mju_mulMatMatT(res,A,B,r1,c1,r2)` | `res[i,j] = mju_dot(A+i*c1, B+j*c1, c1)` |
| `mju_mulMatTMat(res,A,B,r1,c1,c2)` | `res=0`; for `i<r1`, `j<c1`: if `A[i,j]!=0`: `mju_addToScl(res+j*c2, B+i*c2, A[i,j], c2)` |

#### `mju_sqrMatTD_impl(res, mat, diag, nr, nc, flg_upper)` — dense `M'·diag·M`

```
res = 0  (nc×nc)
for j in 0..nr-1:                        # rows of mat
  dj = diag ? diag[j] : 1
  if dj == 0: continue
  mat_j = mat + j*nc
  for i in 0..nc-1:
    tmp = mat_j[i]
    if tmp != 0:
      mju_addToScl(res + i*nc, mat_j, tmp*dj, i+1)   # res[i, 0..i] += mat_j[0..i] * (mat_j[i]*dj)
if flg_upper: res[i,j] = res[j,i] for j>i
```

Scale is `(tmp*dj)` computed first, then multiplied by `mat_j[k]`. Only the lower triangle (incl.
diagonal) is accumulated. `mju_sqrMatTD(res,mat,diag,nr,nc)` = `_impl(..., flg_upper=1)`.
Used by the dense Newton Hessian (`flg_upper=0`, `diag = D` with zeros for non-quadratic rows) and by
`efc_AR` dense (`mju_sqrMatTD(d->efc_AR, YT, NULL, nv, nefc)`).

---

## 2. Sparse toolkit (`engine_util_sparse.c/.h`, `engine_util_sparse_avx.h`)

### 2.1 CSR conventions

A sparse matrix with `nr` rows is `(mat, rownnz[nr], rowadr[nr], colind[nnz])`:

- Row `r` occupies `mat[rowadr[r] .. rowadr[r]+rownnz[r]-1]`, with column indices `colind[...]`
  **sorted strictly ascending** within each row (all merge routines rely on this).
- Rows need not be contiguous; `rowadr` may leave gaps (e.g. the Newton `H` reserves room for `M` and
  the metric: `H_rowadr` shifted by cumulative `M_rownnz[r] + S_rownnz[r]`), and `mju_compressSparse`
  closes them. `mju_sqrMatTDUncompressedInit` sets `rowadr[r] = r*nc` (dense-stride storage).
- Lower-triangular symmetric matrices (e.g. `M` in `M_rownnz/M_rowadr/M_colind`, `qLD`, `H`) store the
  diagonal as the **last** element of each row: `diag(r) = rowadr[r] + rownnz[r] - 1`.
- **Supernodes** `rowsuper[r]`: number of *subsequent* rows with identical sparsity pattern to row `r`.
  `rowsuper[r] = k` means rows `r..r+k` share `colind`; `rowsuper[last]=0`. Built by
  `mju_superSparse` or (column version) `mju_transposeSparse`.
- "Reverse-order" Cholesky factors are `M = L'·L` (or `L'·D·L`) with `L` lower triangular, factored
  from the last row upward, so a tree-structured `M` (rows = dof, columns = ancestor dofs) has no fill-in.
- A transposed CSR (`rownnzT/rowadrT/colindT`) is reinterpreted as CSC of the original (and vice versa);
  `mju_sqrMatTDSparse*` uses this explicitly (`colnnz = rownnzT`, etc.).

### 2.2 Inline helpers (`engine_util_sparse.h`)

**`mju_dotSparse(vec1, vec2, nnz1, ind1)`** — `Σ vec1[i]*vec2[ind1[i]]`:

```
non-AVX: 4 accumulators over i%4 while i <= nnz1-4; res = (r0+r2)+(r1+r3);
         then SEQUENTIAL tail: for (; i<nnz1; i++) res += vec1[i]*vec2[ind1[i]]
AVX:     sum = mul(vec1[0:4], gather(vec2, ind1[0:4])) ; loop add/mul; reduce (s0+s2)+(s1+s3);
         same sequential scalar tail. For nnz1<4: res = 0 + sequential.
```

Note the tail differs from `mju_dot` (sequential `res += p_i` per element vs. grouped
`res += (p0+p1+p2)`). Gather uses `_mm256_set_pd(vec2[ind[i+3]],…,vec2[ind[i]])`.

**`mju_compare(a,b,n)`** — `!memcmp` (AVX: 4 ints per `_mm_cmpeq_epi32`, early out, memcmp tail).

**`mj_mergeSorted(merge, c1, n1, c2, n2)`** — union of two sorted unique int lists; special cases:
both empty → 0; one empty → copy other; identical → copy. Returns count.

**`mju_addToSclScl(res, vec, s1, s2, n)`** — `res[i] = res[i]*s1 + vec[i]*s2` (AVX: `mul`,`mul`,`add`).

**`mju_combineSparse(dst, src, a, b, dst_nnz, src_nnz, dst_ind, src_ind)`** — `dst = a*dst + b*src`,
returns new nnz; `dst` must have room for the union:

```
if dst_nnz==src_nnz && compare(dst_ind,src_ind): addToSclScl(dst,src,a,b,nnz); return nnz
nnz = combineSparseCount(...)                      # |union|
bi=dst_nnz-1; si=src_nnz-1; w=nnz-1               # merge BACKWARDS in place
while bi>=0 && si>=0:
  if dst_ind[bi]==src_ind[si]: dst[w]=a*dst[bi]+b*src[si]; ind=...; bi--, si--
  elif dst_ind[bi] > src_ind[si]: dst[w]=a*dst[bi]; bi--
  else: dst[w]=b*src[si]; si--
  w--
while si>=0: dst[w]=b*src[si]; ...
if a != 1: for remaining bi>=0: dst[bi] *= a     # already in place
```

Element formula for common entries: `a*dst + b*src` (two products then add). With `a=1`: `1*dst`
is exact, so `dst + b*src`.

### 2.3 Sparse routines (`engine_util_sparse.c`)

| Function | Exact behavior |
|---|---|
| `mju_dotSparse2(v1,i1,n1,v2,i2,n2)` | Two-pointer merge on sorted indices, sequential accumulation `res += v1*v2` on matches; 0 if either empty |
| `mju_dense2sparse(res,mat,nr,nc,rownnz,rowadr,colind,nnz)` | row-major scan; nonzero test `mat[r*nc+c] != 0`; returns 1 if `nnz<=0` or capacity exceeded, else 0 |
| `mju_sparse2dense` | zero then scatter |
| `mju_sym2dense(res,mat,n,…)` | zero; for entries with `col<=i` write both `(i,col)` and `(col,i)` |
| `mju_copySparse(res,mat,rownnz,rowadr,row,nrow)` | copy selected rows (same layout) |
| `mju_zeroSparse(...)` | zero selected rows |
| `mju_mulMatVecSparse(res,mat,vec,nr,rownnz,rowadr,colind,rowsuper)` | non-AVX: `res[r]=mju_dotSparse(row r)` (rowsuper ignored). AVX: if `rowsuper==NULL` same; else for supernode `rs=rowsuper[r]+1` rows process blocks of 3 with `mju_dotSparseX3_avx`, remainder with `mju_dotSparse_avx`. Since X3_avx uses the identical lane/tail order per row, results equal per-row `dotSparse_avx` |
| `mju_mulMatTVecSparse(res,mat,vec,nr,nc,…)` | `res=0`; for row `i`: skip if `vec[i]==0`; `res[colind[j]] += row[j]*vec[i]` sequential in row order |
| `mju_addToMatSparse(dst,rownnz,rowadr,colind,nr,M,…)` | per row: `rownnz[i] = mju_combineSparse(dst row, M row, 1, 1, …)` — row capacity must be pre-reserved |
| `mju_addToSymSparse(res,mat,n,…,flg_upper)` | dense `res[i*n+j] += val`; if `flg_upper && j<i` also `res[j*n+i] += val` |
| `mju_mulSymVecSparse(res,mat,vec,n,…)` | lower-stored symmetric: `res=0`; for `i`: `res[i] = row[diag]*vec[i]` (**assignment**; safe, see note below) then for `k = diag-1 down to 0`: `j=ind[k]`; `res[i] += val*vec[j]`; `res[j] += val*vec[i]` |
| `mju_combineSparseCount(a_nnz,b_nnz,a_ind,b_ind)` | `a_nnz + b_nnz − |intersection|` |
| `mju_combineSparseInc(dst,src,n,a,b,dst_nnz,src_nnz,dst_ind,src_ind)` | "incomplete": pattern of dst unchanged. Identical pattern → `addToSclScl`. Else `if a!=1: mju_scl(dst,dst,a)`; then `dst[di] += b*src[si]` on common indices only |
| `mju_addToSclSparseInc(dst,src,nnzdst,inddst,nnzsrc,indsrc,scl)` | `dst[d] += scl*src[s]` at common indices only |
| `mju_addToSparseMat(dst,src,n,nrow,scl,dst_nnz,src_nnz,dst_ind,src_ind,buf,buf_ind)` | Matrix with `nrow` rows **sharing one column pattern** (stored row-major, stride = nnz). Identical pattern: `mju_addToScl(dst,src,scl,nrow*nnz)`. Else merge into `buf` stored transposed (`buf[nrow*nnz+k]`): common → `dst + scl*src`; dst-only → `dst`; src-only → `scl*src`; then `mju_transpose(dst, buf, nnz, nrow)` and copy indices. Used for Jacobian chain accumulation (`engine_core_util.c`) |
| `mju_addChains(res,n,NV1,NV2,c1,c2)` | sorted union of two dof chains (sentinel `n+1`); identical shortcut |
| `mju_compressSparse(mat,nr,nc,rownnz,rowadr,colind,minval)` | Shift rows to be contiguous. If `minval >= 0`, drop entries with `|v| <= minval` and update `rownnz`; if `minval < 0`, only compact. Returns `rowadr[nr-1]+rownnz[nr-1]` |
| `mju_transposeSparse(res,mat,nr,nc,res_rownnz,res_rowadr,res_colind,res_rowsuper,rownnz,rowadr,colind)` | see 2.5 |
| `mju_superSparse(nr,rowsuper,rownnz,rowadr,colind)` | `rowsuper[r] = (rownnz[r]==rownnz[r+1]) && compare(colind rows)`; `rowsuper[nr-1]=0`; accumulate in reverse: `if rowsuper[r]: rowsuper[r] += rowsuper[r+1]` |
| `mju_block`, `mju_blockDiag` | dense block extraction with row/col reverse permutations, via `mju_gather` |
| `mju_blockSparse`, `mju_blockDiagSparse` | sparse block extraction: rows `perm_r`, columns mapped `perm_c[col] - col_offset`, contiguous rowadr starting at `res_offset`; optional second value array `mat2→res2` sharing structure (island block-diagonalisation) |

Note on `mju_mulSymVecSparse`: row `i`'s diagonal **assigns** `res[i]`. Contributions to `res[i]` from
other rows come from rows `i' > i` (their strict-lower column `j=i`), which are processed **after**
row `i`; contributions from rows `i' < i` go only to `res[j]` with `j < i'`. Hence the assignment never
overwrites an earlier contribution. Accumulation order for `res[i]`: `diag*vec[i]`, then own
off-diagonals in **descending** column order, then later rows' upper mirror terms in ascending row order.

### 2.4 `mju_dotSparseX3` — **AVX/non-AVX divergence**

`mju_dotSparseX3(res0,res1,res2, vec10,vec11,vec12, vec2, nnz1, ind1)`: three sparse rows sharing one
pattern dotted with dense `vec2`.

- non-AVX: **single sequential accumulator** per row (`RES0 += vec10[i]*v2` for all i).
- AVX: 4-lane accumulation + `(s0+s2)+(s1+s3)` + sequential tail (same as `dotSparse_avx`).

So the non-AVX and AVX builds produce different bits for `mju_dotSparseX3` (called directly from
`engine_core_constraint.c` for 3-row contact/connect Jacobian-dot-velocity terms), unlike `mju_dot`
/`mju_dotSparse` which are made order-identical. A bitwise reimplementation must pick the build
variant it is matching.

### 2.5 `mju_transposeSparse` with row supernodes

```
if !nr || !nc: return
res_rownnz = 0
row_offset = rowadr[0]       # allows input whose first row starts at nonzero address (mat pointer pre-offset)
count: for each entry: res_rownnz[colind[j]]++   (j indexed relative: rowadr[r]-row_offset)
if res_rowsuper: res_rowsuper[i] = (res_rownnz[i] == res_rownnz[i+1]) for i<nc-1 ; [nc-1]=0
res_rowadr = exclusive prefix sum of res_rownnz
for r in 0..nr-1:
  c_prev = -1
  for entries (c = colind) in row r ascending:
    adr = res_rowadr[c]++ ; res_colind[adr] = r ; if res: res[adr] = mat[i]
    if res_rowsuper and c>0 and c != c_prev+1 and res_rowsuper[c-1]: res_rowsuper[c-1] = 0
    c_prev = c
shift res_rowadr back by one (res_rowadr[i]=res_rowadr[i-1], [0]=0)
accumulate supernodes in reverse (as in superSparse)
```

Supernode criterion here: columns `c-1` and `c` of the input have equal counts, and every input row
that contains `c` also contains `c-1` immediately before it (contiguous). This is weaker-looking than
an explicit compare but equivalent for equal counts. Output rows (transposed) have sorted `colind`
because input rows are scanned in order. `res` may be NULL (structure only).

### 2.6 Sparse `M'·diag·M` ("sqrMatTD") family

Inputs: `mat` (`nr×nc` CSR), its transpose `matT` (`nc×nr` CSR, with `rowsuperT`), optional `diag[nr]`.
Output: `res` (`nc×nc`), lower triangle + diagonal in each row (diagonal last), optionally the
upper triangle appended after the diagonal with `diagind[r]` = address of the diagonal.

`mjMAXSUPER = 8` (max supernode width handled in one batch).

#### 2.6.1 `mju_sqrMatTDSparseSymbolic` (current, column-based)

Count mode (`res_colind==NULL`) / fill mode. For result column batch `c .. c+ns-1`:

```
ns = 1 + min(colsuper[c], mjMAXSUPER-1)   (colsuper = rowsuperT)
if colnnz[c]==0: continue    # (after computing ns: loop steps by ns)
for each r in column c of M^T (i.e. rows k of M touching c):
   rowstart[r] += ns                      # columns c..c+ns-1 sit at rowstart..rowstart+ns-1 in row r of M
   for k from rowstart[r] to end of row r of M: j = colind; if !marker[j]: mark; buffer_idx += j
fill: ns==1: row c gets c (diag); each marked j gets c.
      ns>1:  rows c+s get c..c+s (supernode lower block incl diag); marked j get c..c+ns-1
count: res_rownnz[j] += ns for marked; diag block: += s+1 (or += ns + buffer_nnz if diagind requested → full symmetric count)
count mode end: res_rowadr = prefix sum
fill mode with res_diagind: diagind[r] = rowadr[r] + lower_nnz[r] - 1; append upper entries: for (r, c<r) write r into row c
```

The pattern equals `{(j,c) : j ≥ c, ∃k: M[k,j]≠0 ∧ M[k,c]≠0}` with sorted column indices (columns
are visited in increasing `c`). This relies on the fact that within supernode columns, all rows
of M containing `c` also contain `c..c+ns-1` consecutively (guaranteed by the transpose supernode
definition).

#### 2.6.2 `mju_sqrMatTDSparseNumeric` (current, row-batch accumulation)

Given the pattern, computes values. Uses a dense buffer `buffer[nc*mjMAXSUPER]` (lazily zeroed up to
`buf_zeroed`) and processes **result rows** `r` in batches of `ns = min(rowsuperT[r]+1, 8)`.

Single row (`ns==1`):

```
diag_r = 0
for i over entries of M^T row r (k = colindT, valT = matT):
   scale = diag ? diag[k]*valT : valT
   if scale == 0: continue
   diag_r += scale*valT                         # = diag[k]*valT*valT, as (diag*valT)*valT
   if off_nnz>0:
      for j over row k of M with c = colind < r:     # column r is in row k: acts as sentinel
          buffer[c] += scale*val_k[j]
res row r: off-diagonals = buffer[c] in pattern order (then buffer[c]=0); last = diag_r
```

Supernode (`ns>1`): per k, `scale[s] = (diag ? diag[k]*matT_s[s][i] : matT_s[s][i])` (skip whole k if
`diag[k]==0`), block `block[t][s] += scale[s]*matT_s[t][i]` (note: **`scale[s]` times
`matT_s[t][i]`**, i.e. `(d_k·M[k,r+s])·M[k,r+t]`), off-diagonal `buffer[c*8+s] += scale[s]*val_k[j]`
for `c<r`. SIMD width specialisation (`width=4` when `ns<=4`, else 8) computes extra lanes whose
`scale` is 0 — harmless. Output row `r+s` = off-diagonals from buffer then `block[t][s]`, `t=0..s`.

Empty `M^T` row (`rownnzT[r]==0`) → zero result rows. If `res_diagind`, mirror lower into upper at
`upper_pos[c] = diagind[c]+1, +1, …` in row-scan order.

Summation order for entry `(r,c)`: over `k` in ascending order of `colindT` (i.e. ascending constraint
row index), each term `(diag[k]*M[k,r]) * M[k,c]`. **Diagonal term**: `(diag[k]*M[k,r])*M[k,r]`
(single-row path) — the legacy column-based path uses `(M*M)*diag` (see below), so the two paths differ.

Callers: Newton sparse Hessian (`engine_solver.c`: `H = J'·D·J` with `D=0` for non-quadratic rows,
then `mju_addToMatSparse(H, M)` and metric `S`), cone Hessian, and sparse `efc_AR` (`engine_core_constraint.c`).

#### 2.6.3 `mju_sqrMatTDSparse` (legacy column API, still exported)

Column-oriented with dense `buffer[nc*8]` column-major: for column `c` (batch `ns`), diagonal
`diag_c[s] = Σ_k (val[s][k]*val[s][k])*diag[ind[k]]` (sequential) or `mju_dot(val,val,nnz)` without diag
(4-lane order!); strict-lower entries `buffer[s*nc+j] (=|+=) valT[k]*v[s]` with `v[s]=diag[r]*val[s][i]`.
Writes `nm = min(ns, j-c)` entries per row j (truncating at the diagonal). Optional upper fill via
`diagind`. Requires precomputed `res_rowadr` (e.g. from `mju_sqrMatTDSparseCount` or
`mju_sqrMatTDUncompressedInit`).

#### 2.6.4 `mju_sqrMatTDSparseCount` (legacy)

Chain-merge counting of the lower pattern per row using stack scratch `2*nr` ints; supernode rows copy
the previous count (`+1` for diagonal if `rownnzT[r]`); `flg_upper` adds mirrored counts. Returns total.

#### 2.6.5 `mju_sqrMatTDSparse_row` (legacy reference)

Row-based: per result row `i`, accumulate `buffer[cc] += matT[r]*diag[t]*mat[c]` (product order
`(matT*diag)*mat`) for `cc ≤ i`, insertion of new columns by binary search; supernode rows reuse the
previous pattern.

---

## 3. Dense Cholesky family (`engine_util_solve.c`)

All dense matrices are row-major `n×n`, factor stored in the **lower** triangle (`L`, `mat = L·L'`).
Upper triangle is never read (except where noted) nor written.

### 3.1 `mju_cholFactor(mat, n, mindiag)` → rank

```
rank = n
for j in 0..n-1:
  tmp = mat[j,j] - (j ? mju_dot(L[j,0:j], L[j,0:j], j) : 0)     # mju_dot 4-lane order
  deficient = tmp < mindiag
  if deficient: tmp = mindiag; rank--
  L[j,j] = sqrt(tmp)
  if deficient: L[i,j] = 0 for i>j
  else:
    inv = 1/L[j,j]
    for i in j+1..n-1: L[i,j] = (mat[i,j] - mju_dot(L[i,0:j], L[j,0:j], j)) * inv
```

Left-looking (row-dot) Cholesky. Off-diagonal uses **multiplication by reciprocal**. Note `j==0`: no
dot is subtracted (`tmp = mat[0,0]`; off-diagonals `(mat[i,0] - mju_dot(...,0)) * inv` where
`mju_dot(·,·,0) = 0`, so `mat[i,0]-0`). Callers: dense Newton (`mindiag = mjMINVAL`), boxQP
(`mjMINVAL`), QCQP (`1e-10`), metric 3×3 blocks.

### 3.2 `mju_cholSolve(res, mat, vec, n)`

```
if res != vec: copy
forward: for i in 0..n-1:  if i: res[i] -= mju_dot(L[i,0:i], res, i);  res[i] /= L[i,i]     # division
backward: for i = n-1..0:  res[i] /= L[i,i];  if i && res[i] != 0: mju_addToScl(res, L[i,0:i], -res[i], i)
```

Forward is row-dot (4-lane), backward is column-axpy (right-looking), skipping zero `res[i]`.

### 3.3 `mju_cholUpdate(mat, x, n, flg_plus)` — rank-1 update/downdate of `L·L' ± x·x'`

```
rank = n
for k in 0..n-1:
  if x[k] != 0:
    Lkk = L[k,k]
    tmp = Lkk*Lkk + (flg_plus ? x[k]*x[k] : -x[k]*x[k])
    if tmp < mjMINVAL: tmp = mjMINVAL; rank--
    r = sqrt(tmp); c = r/Lkk; cinv = 1/c; s = x[k]/Lkk
    L[k,k] = r
    s_signed = flg_plus ? s : -s
    for i in k+1..n-1:
       m = (L[i,k] + s_signed*x[i]) * cinv
       L[i,k] = m
       x[i] = c*x[i] - s*m
```

`x` is overwritten. Used in dense Newton incremental factor update (`engine_solver.c`) and boxQP
`cholDelete`/`cholInsert`.

---

## 4. Sparse Cholesky family

All are **reverse-order**: `H = L'·L`, rows processed from `n-1` down to `0`, `L` lower-triangular CSR
with diagonal last in each row.

### 4.1 `mju_cholFactorSparse(mat, n, mindiag, rownnz, rowadr, colind, d)` (legacy in-place)

`mat` holds the lower triangle with pre-reserved room for fill-in in each row.

```
for r = n-1 .. 0:
  nnz = rownnz[r]; adr = rowadr[r]
  tmp = mat[adr+nnz-1]; if tmp < mindiag: tmp = mindiag; rank--
  mat[diag] = sqrt(tmp); inv = 1/mat[diag]
  mat[adr+i] *= inv for i < nnz-1
  for i < nnz-1:  c = colind[adr+i]
     rownnz[c] = mju_combineSparse(row c, row r prefix (first i+1 entries), 1, -mat[adr+i], rownnz[c], i+1, ...)
```

(Rank-deficient rows are not zeroed here, unlike dense.) Row `c` is updated with the prefix of row `r`
up to and including column `c` (which is the diagonal-position entry of row c), giving
`L[c,0:c] -= L[r,c]·L[r,0:c]`.

### 4.2 `mju_cholFactorSymbolic(L_colind, L_rownnz, L_rowadr, LT_colind, LT_rownnz, LT_rowadr, LT_map, rownnz, rowadr, colind, n, d)`

Elimination-tree symbolic factorization (after `ldl_symbolic`, "Algorithm 8xx", but reversed).
Input is the **upper** triangle pattern (`HT` = transpose of lower `H`); for row `r` the loop visits
columns `> r` starting at `upperStart(colind, start, end, r)` (binary search for first column `> r`;
fast path when first column `== r`).

Counting phase (`L_colind == NULL`):

```
for r = n-1 .. 0:
  parent[r] = -1; flag[r] = r; L_rownnz[r] = 1; lt = 1
  for each upper column i of row r:
    while flag[i] != r:                # walk etree from i to root
      L_rownnz[i]++ ; lt++ ; flag[i] = r
      if parent[i] == -1: parent[i] = r; break
      i = parent[i]
  LT_rownnz[r] = lt
L_rowadr, LT_rowadr = exclusive prefix sums; return total nnz(L)
```

Filling phase: rows of `L` are filled **right-to-left** via `cursor[i]` starting just before the diagonal;
`L_colind[diag]=r`; for each visited `i`: `L_colind[cursor[i]--] = r`, `LT_colind[lt_write] = i`,
`LT_map[lt_write] = index in L of (i, r)`. `LT[r]` lists the diagonal first, then the descendant rows
`i > r` in **etree walk order** (not sorted). Since `r` decreases, each row of `L` gets its columns
in ascending order. Scratch from mjData stack, or heap (`mju_malloc`) when `d == NULL` (used by the
compiler, `user_model.cc`).

### 4.3 `mju_cholFactorSymbolicBlocked(...)`

Same `L` arrays as `mju_cholFactorSymbolic` on the dof-expanded pattern, computed from a pattern of
`nb` dense `bs×bs` blocks (diagonal blocks implicitly dense). Block etree walk; `L_rownnz[bs*p+a] =
bs*count[p] + a + 1`; block transpose `BT_adr[nb+1]`, `BT_row`, `BT_off` (column offset `bs*ip` in
the reached block row). Scratch `3*nb` ints from caller. Used by the effective-metric (`engine_metric.c`).

### 4.4 `mju_cholFactorNumeric(L, n, mindiag, L_*, LT_*, LT_map, H, H_*, scratch)` → rank

Left-looking (row-wise in reverse) numeric factorization with fixed pattern and supernode fusion.
`scratch` = dense accumulator of size `n` (zeroed once; each row cleans what it used).

```
for r = n-1 .. 0 step ns:
  H_nofill = (H_rownnz[r] >= L_nnz && H_colind[H_adr + L_nnz - 1] == r)
  # leaf fast path: no descendants and no fill on row r
  if LT_rownnz[r] == 1 and H_nofill:
     ns = 1; diag = H[r, L_nnz-1]; if diag < mindiag: diag = mindiag; rank--
     L_rr = sqrt(diag); off-diags: deficient ? 0 : H[i]*(1.0/L_rr); L[diag] = L_rr; continue
  # supernode detection ns ∈ {1,2,3}:
  ns = 1
  while ns<3 && r-ns>=0 && L_nnz>ns && L_colptr[L_nnz-1-ns] == r-ns
        && L_rownnz[r-ns] == L_nnz-ns && LT_rownnz[r-ns] == LT_nnz+ns: ns++
  rank -= cholFactorSupernode(r, ns, ...)
```

`cholFactorSupernode(r, ns)` (rows `r, r-1, …, r-ns+1`; row `r-s` has `L_nnz-s` entries, a prefix of
row `r`'s columns):

1. **Initialise** each row from `H`: if no fill (`h_nnz ≥ nnz_s` and `H_colind[h_adr+nnz_s-1]==row`)
   `mju_copy` the first `nnz_s` values (H row may contain upper entries after the diagonal); otherwise
   scatter H's lower part into `dense` and gather into `L`'s pattern (ns==1: zero L row and leave the
   values in `dense` to be flushed later).
2. **Descendant updates** (`c` in `LT[r]` excluding the diagonal entry k=0). For descendant row `c`
   with `idx0 = LT_map` = position of `L[c,r]`, `num_cols = idx0 - L_rowadr[c] + 1` = entries of
   `L[c, 0..r]`. *Suffix test*: `is_suffix = (L_colptr[L_nnz-num_cols] == colind of first entry of row c)`
   — by elimination, cols(L[c,0:r]) ⊆ cols(L[r,0:r]); it is the exact suffix iff first columns match.
   Up to `max_batch` (=4, or 3 when ns==3) consecutive descendants sharing the same `num_cols`
   (and suffix start, or — non-suffix, ns==1 only — nested patterns) are fused.
   - Suffix path `cholSuffixUpdate`: with `s[t][b] = L[idx[b] - t]` (= `L[c_b, r-t]`), for the
     `nlead = num_cols-(ns-1)` leading shared columns: `dptr_t[i] -= s[t][b]*p_b[i]` looping
     `b = 0..batch-1` sequentially for t = 0, then t=1, then t=2; the triangular tail columns
     `nlead+u` update row 0 always and row 1 only when `u==0 && ns>=3`.
   - Non-suffix path `cholScatterUpdate`: `acc = dense[col] - s0*p0[i]; acc -= s1*p1[i]; … ; dense[col]=acc`
     (updates for row r only; rows r-1, r-2 revisited in the flush).
3. **Flush** if dense was used: for `s>0`, recompute non-suffix updates for row `r-s` into `dense`
   (`dense[colptr[i]] -= L[L_cr0_idx - s]*p0[i]` for `i < num_cols-s`); then `row_s[i] += dense[c]`,
   `dense[c] = 0`.
4. **Factor** rows `s = 0..ns-1`: `diag = row_s[last]`; deficient if `< mindiag` (→ `mindiag`,
   zero off-diagonals). Else `inv = 1.0/L_rr`; scale intra-supernode entries
   `s_st[t] = (row_s[L_nnz-1-t] *= inv)` for `t>s`; for shared off-supernode columns `i < n_off =
   L_nnz-ns`: `l = (row_s[i] *= inv)`, `row_{s+1}[i] -= s_st[s+1]*l`, `row_{s+2}[i] -= s_st[s+2]*l`;
   intra-supernode: `L_t[L_nnz-1-u] -= s_st[t]*s_st[u]` for `s<t≤u<ns`. Finally `row_s[last] = L_rr`.

Mathematically: `L[r,r] = sqrt(H[r,r] − Σ_{c>r} L[c,r]²)`, `L[r,j] = (H[r,j] − Σ_{c>r} L[c,r]L[c,j]) / L[r,r]`
for `j<r`, where the sums run over descendants in `LT[r]`. Floating-point order of the sums is the
batch/LT order above — **reproducing bits requires reproducing the batching** (batch accumulates
`b` innermost per element, in LT order).

Caller: sparse Newton (`FactorizeHessian`, `mindiag = mjMINVAL`), cone Hessian, `engine_setconst.c`
(efm0).

### 4.5 `mju_cholSolveSparse(res, mat, vec, n, rownnz, rowadr, colind)`

Solves `(L'L) res = vec`: first `res ← L^{-T} res` (backward over rows), then `res ← L^{-1} res`.
Fuses up to 3 adjacent rows that form a supernode:

Backward (`i = n-1` down):
- 3-row fusion if `rownnz[i-1]==nnz-1 && colind[adr+nnz-2]==i-1` and `rownnz[i-2]==nnz-2 &&
  colind[adr+nnz-3]==i-2`: with `nob = nnz-3`:
  `x0 = res[i] /= row0[nob+2]`; `x1 = res[i-1] = (res[i-1] - row0[nob+1]*x0)/row1[nob+1]`;
  `x2 = res[i-2] = (res[i-2] - row0[nob]*x0 - row1[nob]*x1)/row2[nob]`;
  if any nonzero: `res[c] = res[c] - row0[j]*x0 - row1[j]*x1 - row2[j]*x2` for shared cols.
- 2-row fusion analogous.
- single: if `res[i] != 0`: `tmp = res[i] /= diag`; `res[ci[j]] -= row[j]*tmp`.

Forward (`i = 0` up), fusion if row `i+1` has `nnz+1` entries with `colind[rowadr[i+1]+nnz-1]==i`:
- 3-row: `s0,s1,s2 = Σ_j row_t[j]*res[ci[j]]` sequential; `res[i] = (res[i]-s0)/row0[nob]`;
  `res[i+1] = (res[i+1] - s1 - row1[nob]*res[i])/row1[nob+1]`; `res[i+2] = (res[i+2] - s2 -
  row2[nob]*res[i] - row2[nob+1]*res[i+1])/row2[nob+2]`.
- single: `res[i] -= mju_dotSparse(row, res, nnz-1, ci)` (4-lane), then `res[i] /= diag`.

Note fused paths use **sequential** sums while the single path uses 4-lane `mju_dotSparse`, so the
result depends on the supernode structure.

### 4.6 `mju_cholUpdateSparse(mat, x, n, flg_plus, rownnz, rowadr, colind, x_nnz, x_ind, scratch)`

Rank-1 update of reverse factor (`L'L ± xx'`) with Givens rotations, pattern fixed:

```
if x_nnz == 0: return n
start = x_ind[last]; min_row = x_ind[0]
dense[min_row..start] = 0; scatter x into dense
for row = start down to min_row:
  x_row = dense[row]; if x_row == 0: continue
  nnz = rownnz[row]-1; diag = mat[adr+nnz]
  tmp = diag*diag + (flg_plus ? x_row² : -x_row²); if tmp < mjMINVAL: tmp = mjMINVAL; rank--
  r = sqrt(tmp); mat[adr+nnz] = r
  if nnz == 0: continue
  c = diag/r; s = -x_row/r; s_signed = flg_plus ? -s : s
  extend zeroed range of dense if colind[0] < min_row
  for each off-diag i (col j):  m = mat[i]; dj = dense[j]
       mat[i]   = c*m + s_signed*dj
       dense[j] = s*m + c*dj
```

(Contiguous-column fast path identical arithmetic.) Used by sparse Newton incremental updates and
elliptic cone folding.

---

## 5. Band-dense Cholesky (API only; no internal engine caller in this commit)

Storage for `ntotal` rows, `ndense` trailing dense rows, `nsparse = ntotal-ndense`, band width
`nband` (incl. diagonal): first `nsparse*nband` entries store each sparse row's band left of and including
the diagonal, right-aligned (`diag(i) = i*nband + nband-1`); then `ndense` full rows of length
`ntotal` (`diag(i) = nsparse*nband + (i-nsparse)*ntotal + i`). `mju_bandDiag(i,…)` returns these.

`mju_cholFactorBand(mat, ntotal, nband, ndense, diagadd, diagmul)` → min diagonal before sqrt, or 0
if rank-deficient:

```
mindiag = -1
sparse rows j < nsparse:
  width_jj = min(j, nband-1); height = min(nsparse-j-1, nband-1); adr_jj = (j+1)*nband-1
  left = width>1 ? mju_dot(...) : (width==1 ? m[adr-1]*m[adr-1] : 0)
  Ljj = diagadd + diagmul*mat[jj] + mat[jj] - left          # exactly this order
  if Ljj < mindiag or mindiag < 0: mindiag = Ljj
  if Ljj < mjMINVAL: return 0
  Ljj = sqrt(Ljj); scale = 1/Ljj
  band below: width_ij = min(j, nband-1-i+j); L(i,j) = scale*(mat - left_ij)
  dense rows i ≥ nsparse: L(i,j) = scale*(mat[adr_ij] - left with width_jj)
  mat[jj] = Ljj
dense rows j ≥ nsparse: Ljj = diagadd + diagmul*m + m - mju_dot(row j, row j, j); ... full rows
return mindiag
```

`mju_cholSolveBand` (forward band rows with `mju_dot` when width>1, explicit product when width==1;
dense rows `mju_dot`; backward dense with `mju_addToScl`, then band with `mju_addToScl` or explicit
for width 1; divisions by diagonal). `mju_band2Dense(res,mat,…,flg_sym)`, `mju_dense2Band`,
`mju_bandMulMatVec(res,mat,vec,ntotal,nband,ndense,nvec,flg_sym)` (lower via `mju_dot`, strict upper via
`mju_addToScl(res+offset, row, vec[i], width-1)`).

---

## 6. LU factorizations

### 6.1 Dense `mju_factorLU(A, n, pivot)` / `mju_solveLU`

Partial pivoting, in place, row-major. Returns 0 if `max |A[i,k]| < mjMINVAL` at some step.

```
for k: pivot[k]=k; maxrow = argmax_{i≥k} |A[i,k]| (strict >, first max wins)
       if maxval < mjMINVAL: return 0
       if maxrow != k: pivot[k] = maxrow; swap rows k, maxrow (full rows)
       diaginv = 1.0/A[k,k]
       for i>k: A[i,k] *= diaginv; for j>k: A[i,j] -= A[i,k]*A[k,j]
```

Solve: `x=b`; for `i`: swap `x[i] ↔ x[pivot[i]]` if needed, `x[i] -= Σ_{j<i} LU[i,j]x[j]` (sequential,
j ascending); back: `x[i] -= Σ_{j>i} LU[i,j]x[j]` (ascending), `x[i] /= LU[i,i]`.
`mju_factorLU6`/`mju_solveLU6`: identical algorithm with `n=6` constant (bitwise identical results).
Callers: implicit/implicitfast integrator 6×6 free-body blocks (`engine_forward.c`).

### 6.2 Sparse tree LU `mju_factorLUSparse(LU, n, scratch, rownnz, rowadr, colind, index)` → first clamped dof or −1

Reverse-order LU without fill-in (`original = (U+I)·L`), for `qLU` in implicit integration
(non-symmetric `M − dt·∂f/∂v`). Optional `index` (awake dofs).

```
remaining[r] = rownnz[index? index[r] : r]
for r = n-1..0: i = idx(r)
   ii = rowadr[i] + remaining[r] - 1; remaining[r]--
   if colind[ii] != i: mjERROR("missing diagonal element")
   if |LU[ii]| < mjMINVAL: LU[ii] = LU[ii] < 0 ? -mjMINVAL : mjMINVAL; record first clamped
   for c = r-1..0: j = idx(c); ji = rowadr[j] + remaining[c] - 1
      if colind[ji] == i:
         remaining[c]--; LU[ji] = LU[ji] / LU[ii]           # division (not reciprocal)
         merge row i (first remaining[r] entries) into row j (first remaining[c]):
            equal col: LU[j,k] -= LU[i,k]*LU[j,i]; i-only col: mjERROR("requires fill-in")
         if not both exhausted: mjERROR("row processing incomplete")
final check: colind[rowadr[i]+remaining[r]] == i
```

`mju_solveLUSparse(res, LU, vec, n, rownnz, rowadr, diag, colind, index)`:
`(U+I)` back-solve over k descending: `res[i] = vec[i] - mju_dotSparse(U part after diag)`; then
`L` forward: `res[i] -= mju_dotSparse(L part before diag, d entries)`; `res[i] /= LU[adr+d]`.

### 6.3 `mju_solve3(x, A, b)`

Gauss–Jordan on augmented 3×4 **without pivoting**: for `i`: `pivot = M[i][i]`, `M[i][j] /= pivot`
for `j≥i`; eliminate column `i` in other rows `M[k][j] -= M[k][i]*M[i][j]` (factor read before loop).

---

## 7. Symmetric 3×3 eigendecomposition (Jacobi with quaternion accumulation)

`mju_eig3(eigval, eigvec, quat, mat)` = `mju_eig3Tol(..., eigTOL)`; constants:

| | double | single |
|---|---|---|
| `eigTOL` | `4e-15` | `2e-6f` |
| `eigEPS` | `1e-12` | `1e-5f` |

`mju_eig3Tol(eigval, eigvec, quat, mat, reltol)` → iterations:

```
scale = max_i |mat[i]| over all 9 entries
tol = scale * max(reltol, eigTOL)
quat = (1,0,0,0)
for iter in 0..499:
  eigvec = quat2Mat(quat)                     # mju_quat2Mat (identity shortcut for exact unit quat)
  tmp = eigvec' * mat ; D = tmp * eigvec      # mji_mulMatTMat3 then mji_mulMatMat3
  eigval = (D[0], D[4], D[8])
  # pick max off-diagonal with strict comparisons:
  if |D1| > |D2| and |D1| > |D5|:  (rk,ck,rotk) = (0,1,2)
  elif |D2| > |D5|:                (0,2,1)
  else:                            (1,2,0)
  if |D[3rk+ck]| <= tol: break
  tau = (D[4ck] - D[4rk]) / (2*D[3rk+ck])
  t = tau >= 0 ? 1.0/(tau + sqrt(1+tau²)) : -1.0/(-tau + sqrt(1+tau²))
  h = t / (1 + sqrt(1 + t²))                  # tan(angle/2)
  tmp = (1/sqrt(1+h²), 0, 0, 0); tmp[rotk+1] = (rotk==1 ? h : -h) * tmp[0]
  quat = quat * tmp (mju_mulQuat); mju_normalize4(quat)
# sort decreasing with 3-pass bubble (j1 = 0,1,0):
eps = scale * eigEPS
for j in 0..2: j1 = j%2
  if eigval[j1] + eps < eigval[j1+1]:
     swap; tmp=(0.707106781186548, 0,0,0); tmp[(j1+2)%3+1] = 0.707106781186548
     quat = quat*tmp; normalize4
eigvec = quat2Mat(quat)       # eigenvectors are COLUMNS of eigvec
return iter
```

Engine caller: broadphase covariance frame `mju_eig3Tol(eigval, eigvec, quat, cov, 1e-3)`
(`engine_collision_driver.c`; "SAP is exact in any frame"). The compiler uses its own double copy
`mjuu_eig3` (`user_util.cc`, default `reltol=4e-15`, sort `eps = scale*max(reltol, 1e-12)` — **note:
the compiler's sort threshold includes `reltol`, the engine's does not**; normalisation via
`mjuu_normvec` which divides by `nrm` and skips if `|nrm-1| ≤ mjEPS`). Mesh inertia calls it with
`reltol = 1e-7` (float32 vertices); body full-inertia with the default. Principal inertia ordering
(decreasing) and `body_iquat` come from this — required for parity of compiled models.

---

## 8. QCQP solvers (PGS elliptic friction)

Problem: `min ½xᵀAx + xᵀb  s.t.  Σ (x_i/d_i)² ≤ r²`. Scale `b̃_i = b_i d_i`, `Ã_ij = A_ij d_i d_j` so the
constraint becomes `‖x̃‖ ≤ r`; Newton on the Lagrange multiplier `λ` of the secular equation
`‖(Ã+λI)^{-1} b̃‖² = r²`.

`mju_QCQP2(res, Ain, bin, d, r)` (A is 2×2 row-major: uses `Ain[0], Ain[1], Ain[3]`):

```
la = 0
for iter < 20:
  det = (A11+la)*(A22+la) - A12*A12
  if det < 1e-10: res = 0; return 0
  detinv = 1/det; P11=(A22+la)*detinv; P22=(A11+la)*detinv; P12=-A12*detinv
  v1 = -P11*b1 - P12*b2 ; v2 = -P12*b1 - P22*b2
  val = v1*v1 + v2*v2 - r*r
  if val < 1e-10: break
  deriv = -2.0*(P11*v1*v1 + 2.0*P12*v1*v2 + P22*v2*v2)
  delta = -val/deriv ; if delta < 1e-10: break
  la += delta
res = (v1*d0, v2*d1); return la != 0
```

`mju_QCQP3`: same with adjugate `P` (unscaled cofactors `P11=(A22+la)(A33+la)-A23², P12=A13A23-A12(A33+la)`,
etc.), `det = (A11+la)P11 + A12P12 + A13P13`, then scale by `detinv`;
`deriv = -2(P11v1²+P22v2²+P33v3²) - 4(P12v1v2+P13v1v3+P23v2v3)`.

`mju_QCQP(res, Ain, bin, d, r, n)` n≤5 (else `mjERROR`): per iteration `Ala = Ã + la·I`;
`mju_cholFactor(Ala, n, 1e-10) < n` → zero, return 0; `res = -cholSolve(Ala, b̃)` (`mju_scl(-1)`);
`val = mju_dot(res,res) - r*r`; `deriv = -2*mju_dot(res, cholSolve(Ala,res))`; same thresholds `1e-10`.
All thresholds are absolute `1e-10` (for `det`, `val`, `delta`). Callers: `engine_solver.c` PGS
elliptic cones (dim-1 = 2 → QCQP2, 3 → QCQP3, else QCQP).

---

## 9. Box-constrained QP (`mju_boxQP`, projected Newton; API only)

`mju_boxQP(res, R, index, H, g, n, lower, upper)` → `nfree` or −1. Defaults passed to
`mju_boxQPoption`:

| Option | Value |
|---|---|
| `maxiter` | 100 |
| `mingrad` | `1E-16` (min **squared** norm of free gradient) |
| `backtrack` | 0.5 |
| `minstep` | `1E-22` |
| `armijo` | 0.1 |
| `log`, `logsz` | NULL, 0 |

Memory: `R` must have `n*(n+7)` mjtNums; scratch carved from `R + n*n`: `grad`, `search`, `candidate`,
`temp` (n each), `clamped` and `oldclamped` (int, each in an n-mjtNum slot), `index` fallback at
`scratch + 6n`. `mju_boxQPmalloc` allocates these with `mju_malloc`. Only lower triangles of H and R
are read/written.

Status enum: `NOT_SPD=-1, NO_DESCENT=0, MAX_ITER=1, MAX_LS_ITER=2, TOL_GRAD=3, UNBOUNDED=4,
ALL_CLAMPED=5`. Return `-1` if status is `NO_DESCENT` or `NOT_SPD`, else `nfree`.

Algorithm (`mju_boxQPoption`):

```
checks: n>0 ; if both bounds: lower[i] < upper[i] strictly (else mjERROR)
if !lower && !upper:              # unconstrained: Newton point
   R = H; rank = cholFactor(R, n, mjMINVAL)
   if rank==n: res = -cholSolve(R,g); value = 0.5*dot(res,g); status=UNBOUNDED  else NOT_SPD
   index = 0..n-1
else: res = clamp(res, lower, upper)   (max with lower, then min with upper)
for iter < maxiter:
  if status != NO_DESCENT: break
  grad = mulSymVec(H, res)          # lower-tri symmetric: res[i] = H[i,i]v[i] + dot(H[i,0:i], v); then axpy mirror
  if iter == 0: value = 0.5*dot(res,grad) + dot(res,g)
  grad += g
  oldvalue = value
  clamped[i] = (lower && res[i]==lower[i] && grad[i] > 0) || (upper && res[i]==upper[i] && grad[i] < 0)
  nfree = #!clamped ; if nfree==0: status = ALL_CLAMPED; break
  factorize = (iter==0)
  if iter>0 and clamped set changed (nclamp newly clamped, nunclamp newly free):
     if nclamp < oldnfree and nclamp + 2*nunclamp <= max(2, nfree/8):   # rank-1 path
        delete newly-clamped positions p = m-1..0 via cholDelete (rank must stay m-1)
        insert newly-unclamped vars in index order via cholInsert (rank must be m+1)
        any failure → factorize = 1
     else factorize = 1
  if factorize: index = free dims ascending; R[i,k] = H[index[i], index[k]] (k≤i); rank = cholFactor(R, nfree, mjMINVAL); nfactor++
  oldclamped = clamped
  if rank != nfree: status = NOT_SPD; break
  temp = grad[index]; norm2 = Σ temp² (sequential)
  if norm2 < mingrad: status = (nfree==n ? UNBOUNDED : TOL_GRAD); break
  temp = cholSolve(R, temp); search = -temp; sdotg = -Σ temp[i]*grad[index[i]] (sequential, starts 0)
  if sdotg >= 0: break                       # NO_DESCENT
  step = 1
  do:  candidate_i = res[idx] + step*search[i], clipped to bounds (lower checked first)
       reduction = !clipped ? -step*(1 - 0.5*step)*sdotg
                            : -ddotg - 0.5*dot(temp,temp) with temp = R'·delta (built by addToScl of rows R[i,0:i+1]*di), ddotg = Σ di*grad
       value = oldvalue - reduction
       nstep++; step *= backtrack; if step < minstep: status = MAX_LS_ITER; break
       improvement = -reduction / (step*sdotg)          # NOTE: uses the already-halved step
  while improvement < armijo
  res[index] = candidate
  if !clipped and value >= oldvalue: status = (nfree==n ? UNBOUNDED : TOL_GRAD); break
if iter == maxiter: status = MAX_ITER
```

Quirk to replicate: `improvement` is computed with the *post-backtrack* `step` (i.e. half the step
that produced `reduction`), so the Armijo test effectively compares against twice the linear
prediction ratio.

`cholDelete(R, temp, m, p)`: repacks rows from stride `m` to `m-1` removing row/col `p`, extracting old
column `p` (rows > p) into `temp[p..m-2]` (leading zeros), then `mju_cholUpdate(R, temp, m-1, +1)`.
`cholInsert(R, temp, rowp, H, n, index, m, p, idx)`: solves new row `p` (`rowp[j] = (H[idx,index[j]] -
dot(R[j], rowp, j))/R[j,j]`), `diag = H[idx,idx] - dot(rowp,rowp,p)` (`< mjMINVAL` → −1),
new column below `temp[i+1] = (H[index[i],idx] - dot(R[i],rowp,p)) * (1/l22)` for `i ≥ p`, expands R
in place to stride `m+1`, then **downdate** `mju_cholUpdate(R, temp, m+1, 0)`.

---

## 10. Sort macros (`engine_sort.h`)

`mjSORT(name, type, cmp)` defines `static inline void name(type* arr, type* buf, int n, void* context)`:
a **stable** bottom-up tiled merge sort ("timsort-like"):

1. Insertion sort each run of `_mjRUNSIZE = 32` elements (`cmp(arr+k, &tmp, ctx) > 0` shifts → stable).
2. Merge passes with `len = 32, 64, …`: `_mjMERGE` takes from left when `cmp(left,right) <= 0` (stable),
   ping-ponging between `arr` and `buf`; odd tail chunks copied; final `memcpy` back if needed.

`cmp(a, b, context)` returns `<0, 0, >0`. Users: `pairSort` (collision pairs), `SAPsort`, `bfsort`
(broadphase, `engine_collision_driver.c`).

`mjPARTIAL_SORT(name, type, cmp)` defines `name(arr, buf, n, k, ctx)`: selects the `k` smallest via a
max-heap in `buf` (`_mjSIFT_DOWN`; replace root when `cmp(arr+i, buf) < 0`), copies back to `arr[0..k)`
and insertion-sorts them. Not stable w.r.t. original order among equals (heap). Returns immediately if
`k<=0 || n<k`. User: `ContactSelect` in `engine_sensor.c` (contact sensors).

Scalar helpers: `mju_insertionSort(list,n)` (mjtNum, ascending, `list[j] > x` shift — stable),
`mju_insertionSortInt`.

Parity: contact ordering and broadphase pair ordering depend on these stable sorts and comparator tie
rules; GPU implementations use different sorts (radix/segmented), see §14.

---

## 11. Miscellaneous scalar utilities (`engine_util_misc.c`, in-scope subset)

### 11.1 `mju_springDamper(pos0, vel0, k, b, t)` — analytic solution of `x'' + b x' + k x = 0`

```
det = b*b - 4*k
det_tol = mjMINVAL * max(b*b, 4*|k|)          # scale-invariant regime selection
if det > det_tol:                             # overdamped
   w = sqrt(det)/2
   if b >= 0: r2 = -b/2 - w; r1 = k/r2        # larger-magnitude root first, then Vieta r1*r2 = k
   else:      r1 = -b/2 + w; r2 = k/r1
   c1 = (pos0*r2 - vel0)/(r2 - r1); c2 = (pos0*r1 - vel0)/(r1 - r2)
   return c1*exp(r1*t) + c2*exp(r2*t)
elif -det_tol <= det <= det_tol:              # critical
   c1 = pos0; c2 = vel0 + b*c1/2
   return exp(-b*t/2) * (c1 + c2*t)
else:                                         # underdamped
   w = sqrt(|det|)/2; c1 = pos0; c2 = (vel0 + b*c1/2)/w
   return exp(-b*t/2) * (c1*cos(w*t) + c2*sin(w*t))
```

(API function; no internal engine caller in this commit.)

### 11.2 `mju_sigmoid(x)`

`x <= 0 → 0`; `x >= 1 → 1`; else `x*x*x * (3*x*(2*x - 5) + 10)` (= 6x⁵−15x⁴+10x³; C^2 smooth step).
Evaluation order: `((x*x)*x) * ((3*x)*(2*x-5) + 10)`. Used by `mju_muscleDynamicsTimescale` as
`sigmoid(dctrl/smoothing_width + 0.5)`. MJWarp `util_misc._sigmoid` is the same expression.

### 11.3 `mju_Halton(index, base)`

```
n0 = index; b = (mjtNum)base; f = 1/b; hn = 0
while n0 > 0: n1 = n0/base; r = n0 - n1*base; hn += f*r; f /= b; n0 = n1
```

Note `f /= b` (repeated division, not `pow`). Users: SDF collision sample points
(`mju_Halton(sp+1, 2/3)`, AABB sampling bases 2,3,5), visualisation colors. MJWarp `util_misc.halton`
is identical (float32).

### 11.4 `mju_round(x)`

`d=(double)x; d > INT_MAX → INT_MAX; d < INT_MIN → INT_MIN; (int)round(d)` — `round` = half away from
zero. (NaN → `(int)round(NaN)` is UB/implementation-defined.)

### 11.5 `mju_wrap(wpnt, x0, x1, xpos, xmat, radius, type, side)` — tendon wrapping (summary)

(Full tendon semantics are in the tendon spec; numerics here.) Steps: map endpoints to the geom frame
(`p = xmatᵀ(x−xpos)`); return −1 if either `‖p‖ < mjMINVAL`. Sphere: plane frame `axis0 = p0/‖p0‖`,
`normal = p0×p1` normalised; if `nrm < mjMINVAL` choose `axis1 = ones with 0 at argmax|axis0|`
(strict comparisons) and recompute `normal`; `axis1 = normal×axis0` normalised. Cylinder: axes x,y.
Project to 2D `d`. Side site: `s` projected, `sd` normalised (`mju_normalize` → `(1,0)` if tiny) and
scaled by radius. If `side && ‖s‖ < radius` → `wrap_inside`, else `wrap_circle`.

`wrap_circle(pnt, end, side, radius)`: no wrap if `sqlen0<r²`, `sqlen1<r²`, `r<mjMINVAL`, or
`dd<mjMINVAL`; closest point param `a = clamp(-(dif·end0)/dd, 0, 1)`; `clear = |a·dif+end0|² > r²`;
with side: `wside = wind(end0,side) + wind(side,end1)` (`wind(p,q) = atan2(p×q, p·q)`); if `clear &&
|wside − wind(end0,end1)| < π` no wrap; `ccw = wside > 0`; without side: `clear` → no wrap. Tangent
points for `sgn=±1`: `sol[i][0] = ((e0x r² + sgn r e0y √(l0−r²))/l0, (e0y r² − sgn r e0x √…)/l0)`,
`sol[i][1] = ((e1x r² − sgn r e1y √(l1−r²))/l1, (e1y r² + sgn r e1x √…)/l1)`; goodness
`−|sol0−sol1|²`, `−10000` if segments `end0→sol0` and `end1→sol1` intersect (`is_intersect`, det
threshold `mjMINVAL`, conditions `a0>=0 && a1<0 && b0>=0 && b1<0`). Choose `i = side ? ccw :
(good0 > good1 ? 0 : 1)`; reject if chosen intersects; length: side → `r·|wside − wind(end0,pnt0) −
wind(pnt1,end1)|`; else `length_circle` (angle `atan2(|cross|, dot)`, flip to `2π−angle` if `angle >
1e-6` and orientation mismatches `ind`).

`wrap_inside`: Newton on `asin(A z) + asin(B z) − 2φ + G = 0`, `z = sin φ`, `A=r/l0`, `B=r/l1`,
`cosG = (l0²+l1²−dd)/(2 l0 l1)` (`< −1+mjMINVAL` → −1; `> 1−mjMINVAL` → 0 with default point);
init `φ = π/2`; `maxiter = 20`, `tolerance = 1e-6`; derivative
`df = cos φ (A/max(mjMINVAL, √(1−z²A²)) + B/max(mjMINVAL, √(1−z²B²))) − 2`; several "SHOULD NOT OCCUR"
exits returning 0 with the default point (average direction scaled to radius). Final point: rotate
`vec` (end0 if `end0×end1 > 0` else end1, normalised) by `ang = φ − asin(A z)` (resp. B).
Cylinder: heights `res_z = p0z + (p1z−p0z)·L0/(L0+wlen+L1)` and `…(L0+wlen)/(…)`; `wlen =
√(wlen² + height²)`. Map back `wpnt = xmat·res + xpos`.

MJWarp `util_misc.wrap/wrap_circle/wrap_inside` port this with differences: sentinel `vec(MJ_MAXVAL)`
for "no side"; `safe_div` in tangent-point and height divisions; side point with `‖sd‖<MJ_MINVAL`
replaced by `(1,0)` explicitly; Newton initial `f` uses `asin(A) + asin(B) − π + G` (identical value
since `z=1`) and caches `asin(A z)`, `asin(B z)` for the final angle (C recomputes `asin(A*z)` —
same value); invalid wrap type returns `MJ_MAXVAL` instead of erroring.

### 11.6 Other small utilities

| Function | Behavior |
|---|---|
| `mju_encodePyramid(pyr, force, mu, dim)` | `a = f0/(dim-1)`; `b = min(a, f_{i+1}/mu_i)`; `pyr[2i] = 0.5(a+b)`, `pyr[2i+1] = 0.5(a−b)` |
| `mju_decodePyramid(force, pyr, mu, dim)` | `dim==1`: `f0 = pyr0`; else `f0 = Σ pyr` (sequential over `2(dim-1)`), `f_{i+1} = (pyr[2i] − pyr[2i+1])·mu_i` |
| `mju_outsideBox(point,pos,mat,size,inflate)` | `inflate<1` → error; 1 if outside inflated box, −1 if inside deflated (`size/inflate`) or `inflate==1` and inside, 0 between |
| `mju_isZero(vec,n)`, `mju_isZeroByte(vec,n)` | all `== 0` (byte version: first byte 0 and `memcmp(vec, vec+1, n-1)==0`) |
| `mju_zeroInt`, `mju_copyInt`, `mju_fillInt` | memset/memcpy/loop |
| `mju_standardNormal(num2)` | Marsaglia polar method with C `rand()`: `x = 2/RAND_MAX·rand() − 1`, reject `w ≥ 1 || w == 0`, `w = sqrt(-2 log w / w)`. **Platform-dependent** (`rand`, `RAND_MAX`) |
| `mju_f2n`, `mju_n2f`, `mju_d2n`, `mju_n2d` | casts |
| `mju_gather(res,vec,ind,n)` / `mju_scatter` | `ind==NULL` → copy |
| `mju_gatherMasked` | `ind<0` → 0 |
| `mju_gatherInt`, `mju_scatterInt` | int versions |
| `mju_sparseMap(map, nr, res_*, src_*)` | for pattern(res) ⊆ pattern(src): `map[res_adr] = src_adr` |
| `mju_lower2SymMap(map, nr, res_*, src_*, cursor)` | map from full symmetric `res` to lower `src` (both triangles), −1 where absent (builds `mapM2D`/`mapD2M`, `engine_io.c`) |
| `mju_polyForce(linear, poly, x, n, flg_odd)` | `x = flg_odd ? |x| : x`; `res = linear`; `xpow=1`; loop `xpow *= x; res += poly[i]*xpow` |
| `mjd_xPolyForce` | same with `(i+2)*poly[i]*xpow` |
| `mju_polyPotential` | `0.5*linear*(x*x) + Σ poly[i]/(i+3) * (x^{i+2} * x)` (`xpow` starts at `x`) |
| `mj_lugreStribeck(v, F_C, F_S, v_S)` | `ratio = v/max(mjMINVAL, v_S)`; `F_C + (F_S−F_C)·exp(−ratio²)` |
| `mj_dcmotorSlots(dynprm, gainprm)` | slot order: slew (`dynprm[7]>0`), integral (`gainprm[5]>0`), temperature (`dynprm[2]>0`), bristle (`dynprm[5]>0`), current (`dynprm[0]>0`); each `-1` if absent |
| `mju_strncpy(dst, src, n)` | `strncpy` then `dst[n-1]=0` |

(Muscle functions, flex/shell interpolation, history buffers, Base64 and string tables in
`engine_util_misc.c` are specified in their subsystem documents.)

---

## 12. Memory, allocation, error infrastructure

### 12.1 Heap: `mju_malloc` / `mju_free` (`engine_util_errmem.c`)

```
mju_malloc(size):
  if mju_user_malloc: ptr = mju_user_malloc(size)
  else: if size>0 && size%64: size += 64 - size%64          # pad to multiple of 64
        if size>0: ptr = aligned_alloc(64, size)            # _aligned_malloc on Windows
  if !ptr && size>0: mju_error("Could not allocate memory")
```

64-byte (cache-line) alignment; `mju_free` → `_aligned_free`/`free` or user handler.
mjModel/mjData buffers are allocated through this; individual arrays inside are laid out by
`engine_io.c` (8-byte aligned at least; see the memory spec).

### 12.2 Stack/arena (`engine_memory.h`)

`mjSTACKALLOC(d, num, type)` → `mj_stackAllocInfo(d, num*sizeof(type), _Alignof(type), __func__,
__LINE__)`; alignment = natural alignment of the type (8 for double, 4 for int). Stack frames are
created with `mj_markStack`/`mj_freeStack`; `mj_stackBytes(bytes, align) = bytes + align − 1 + 2·mjREDZONE`
(upper bound used for capacity checks, e.g. in the sparse Newton setup which emits
`mjWARN_CNSTRFULL` if the stack is insufficient). Alignment does not affect numerical results (all SIMD
loads are unaligned).

### 12.3 Logging and errors

`mju_error` builds an `mjLogMessage` (level ERROR) → active handler (thread-local override > global);
default handler writes to `MUJOCO_LOG.TXT` (append, timestamp) and stderr, then `exit(EXIT_FAILURE)`
unless legacy `mju_user_error` is set. `mju_warning` (stderr), `mju_info(topic, …)` filtered by topic
bitmask (env `MUJOCO_LOG_TOPICS="time_stp,time_cmp,sleep"`), recursion guard via thread-local
`in_log`. `mjERROR(...)` macro (in headers) prefixes the function name.

### 12.4 `engine_array_safety.h`, `engine_crossplatform.h`

- `mjSIZEOFARRAY(arr)`: `_Generic(&(arr), char(*)[sizeof(arr)]: sizeof(arr))` — compile error for non-arrays;
  `mjSNPRINTF`, `mjSTRNCAT`, `mjSTRNCPY` bounded helpers. C only.
- `strcasecmp`→`_stricmp` on Windows; `setenv`/`unsetenv` shims; `mjFALLTHROUGH`; `mjtMaxAlign`;
  `mjLIKELY/mjUNLIKELY` (`__builtin_expect`); `mjALWAYS_INLINE` (`__forceinline` / `inline
  __attribute__((always_inline))`); `mjTHREADLOCAL` (`_Thread_local`/`__declspec(thread)`); ASAN detection
  (`mjUSEASAN`); relaxed atomic `mj_atomic_add_size_t`; acquire/release `mj_atomic_load/store/exchange_bool`.

---

## 13. Floating-point determinism of the C engine

### 13.1 Build configuration (as shipped)

| Setting | Value / effect |
|---|---|
| Language | C11 (`CMAKE_C_STANDARD 11`, `CMAKE_C_EXTENSIONS OFF` → `-std=c11`; Emscripten: extensions ON), C++20 |
| Build type default | `Release` (CMake default flags `-O3 -DNDEBUG` for GCC/Clang) |
| LTO | `MUJOCO_ENABLE_LTO=ON` → IPO for non-Debug builds |
| AVX | `MUJOCO_ENABLE_AVX=ON` → `-mavx` (`/arch:AVX`); `MUJOCO_ENABLE_AVX_INTRINSICS=ON` → `-DmjUSEPLATFORMSIMD` |
| FMA | **not enabled** (no `-mfma`, no `-march=native`) on x86 |
| fast-math | **not used**; no `-ffast-math`, `-Ofast`, `-ffp-contract=…`, `/fp:fast` anywhere in the build files |
| Other flags | `-fdata-sections -ffunction-sections`, warnings `-Werror -Wall -Wpedantic …`; optional hardening (`-D_FORTIFY_SOURCE=2 -fstack-protector`, Clang only) |
| Single precision | `-DmjUSESINGLE` (CMake: add to `CMAKE_C_FLAGS`/`CMAKE_CXX_FLAGS`; Bazel: `--//:single_precision` → `defines = ["mjUSESINGLE"]`); disables AVX paths; libccd also built single when `CMAKE_C_FLAGS` matches `mjUSESINGLE` |

Implications for FP contraction (fusing `a*b+c` into FMA):
- x86-64 with only `-mavx`: FMA instructions are unavailable to the compiler, so **no contraction**
  can happen regardless of `-ffp-contract` defaults; results are plain IEEE double with round-to-nearest.
  GCC in ISO mode (`-std=c11`) additionally defaults to `-ffp-contract=off`.
- AArch64 (Apple Silicon, Linux ARM): FMA is baseline. Clang's default is `-ffp-contract=on`
  (contract within a single expression), so expressions like `res[i] += vec[i]*scl` or
  `a0*b0 + a1*b1 + a2*b2` may compile to `fmadd`; GCC in `-std=c11` mode keeps contraction off for C.
  **Bits on ARM Clang builds can therefore differ from x86 builds.** To get portable bits compile with
  `-ffp-contract=off` (Clang/GCC) or `/fp:precise` without `/fp:contract` (MSVC).
- x87 is not used on x86-64 (SSE2 scalar double), so no extended-precision intermediate issues.

### 13.2 What determines bitwise results

1. Operation order of every reduction (documented per function above: 4-lane `mju_dot`/`mju_dotSparse`
   vs sequential `mju_sum`, `mju_L1`, `mju_dotInd`, `mju_dotSparse2`, `mju_mulMatTVec*`, fused cholSolveSparse
   paths, supernode batching in `mju_cholFactorNumeric`/`mju_sqrMatTDSparseNumeric`).
2. Build variant: AVX vs non-AVX differ only in `mju_dotSparseX3` (and possibly the sign of zero in
   `mju_dot`/`mju_dotSparse` when the first block's products are `−0`).
3. Zero-skipping branches (`if (tmp)`, `if (x[k])`, `if (vec[r])`, `if (res[i])`) change which
   operations occur; skipping a `+ 0*y` term is not a no-op when `y` is Inf/NaN, and can flip `−0`.
4. Reciprocal vs division choices (e.g. `mju_normalize3` multiplies by `1/norm`; `mju_cholSolve`
   divides; `mju_cholFactor` multiplies by `1/L_jj`; `mju_factorLUSparse` divides).
5. libm transcendentals (`exp`, `sin`, `cos`, `atan2`, `asin`, `acos`, `pow`, `tanh`, `log`) — platform
   dependent last-bit; `sqrt` is exact.
6. Multi-threading: `mju_*` routines are single-threaded; engine threading (thread pools, island
   parallelism) is documented elsewhere — but these kernels are deterministic for a given thread
   assignment because each island's work uses these sequential routines.
7. Compiler auto-vectorisation at `-O3` does not reassociate FP reductions without `-ffast-math`
   (GCC/Clang only vectorise FP reductions with `-fassociative-math`), so plain loops keep source order.

### 13.3 Recipe for bitwise reproduction of MuJoCo C (double)

- Use IEEE-754 binary64, round-to-nearest-even, no FMA contraction, no flush-to-zero/denormals-are-zero.
- Implement `mju_dot` and `mju_dotSparse` exactly with the 4-lane / `(r0+r2)+(r1+r3)` / tail rules.
- Reproduce every branch on exact zero, every `<` vs `<=`, every threshold constant listed.
- Match libm (`glibc` for Linux x86-64 reference builds) or avoid transcendental-dependent comparisons.
- Choose AVX (default x86 build) semantics for `mju_dotSparseX3`.
- Respect the supernode-detection logic in sparse Cholesky and `mju_cholSolveSparse` (results depend
  on structure, not just values).

---

## 14. GPU (MuJoCo Warp) implementation

### 14.1 Precision and global configuration

- Warp kernels use the Python annotation `float`, which is **`wp.float32`**; MJWarp has no float64 build
  switch in this commit — all `Model`/`Data` arrays are float32 (`wp.vec3` = `vec3f`, `wp.mat33` =
  `mat33f`, `wp.spatial_vector` = float32 6-vector). Integer types are int32.
- Constants mirror C (`types.py`): `MJ_MINVAL = mujoco.mjMINVAL` (1e-15, representable in float32),
  `MJ_MAXVAL = 1e10`, `MJ_MINIMP = 1e-4`, `MJ_MAXIMP = 0.9999`, `MJ_MAXCONPAIR = 50`, `MJ_MINMU = 1e-5`,
  `MJ_MINAWAKE = 10`. Float32-specific: `MJ_MAX_EPAHORIZON = 24`, `MJ_MAX_EPAFACES = 5`,
  `TILE_SIZE_JTDAJ_SPARSE = TILE_SIZE_JTDAJ_DENSE = 16`, `M_BLOCK_DENSE_MAX = 64`, `M_BLOCK_SCALAR_MAX = 6`,
  `Q_LD_BLOCK_COMPACT = -2`, `Q_LD_BLOCK_SPARSE = -1`.
- Where C has `mjUSESINGLE` tolerances, MJWarp generally uses the single-precision values (e.g.
  `math.deriv_expmap` uses the float32 Bortz coefficients `0.0833333358, 0.00138884981, 3.32480922e-05,
  4.57548566e-07, 4.23758962e-07, -2.48791736e-07, 8.78585737e-08, -1.64186869e-08, 1.26339661e-09`
  in Horner form for `xx < 3.0625` (= 1.75²), closed form `(1 - 0.5x/tan(0.5x))/xx` otherwise; collision
  primitives use "rounding scales in units of float32 epsilon (MuJoCo's mjUSESINGLE values)").
- `io.put_model`: `opt.tolerance = max(opt.tolerance, 1e-6)` ("C MuJoCo tolerance was chosen for float64").
- Module options: most modules set `wp.set_module_options({"enable_backward": False,
  "default_grid_stride": False})`. **fast-math** is enabled only in (a) the iterative linesearch kernel
  (`solver.py`: `module_options={"fast_math": True}`, comment: "fast math only approximates div/sqrt here;
  keep transcendentals out of this kernel") and (b) the render kernel (`use_fast_math=True` default). CUDA
  compiles with FMA contraction enabled by default (`--fmad=true` semantics), so `a*b+c` patterns in Warp
  kernels are typically fused — another reason MJWarp results cannot match C bits.
- Tile GEMMs in the Cholesky kernels pass `module_options={"enable_mathdx_gemm": False}`, i.e. Warp's
  built-in FP32 tile matmul (no cuBLASDx/MathDx path, no TF32 tensor cores). No TF32 is used anywhere.

### 14.2 `math.py` (`@wp.func` helpers)

| Function | Notes vs C |
|---|---|
| `mul_quat(u,v)` | same component formulas as `mju_mulQuat` |
| `quat_mul_axis(q, axis)` | `mju_mulQuatAxis` |
| `rot_vec_quat(vec, quat)` | `r = 2(u·v)u + (s² − u·u)v + 2s(u×v)` — **different formula** from C `mju_rotVecQuat` (which goes via matrix/cross products) |
| `axis_angle_to_quat(axis, angle)` | `(cos(a/2), axis·sin(a/2))` (no normalisation of axis) |
| `quat_to_mat(q)` | standard formula with `q00…q33` products (no identity shortcut as in `mju_quat2Mat`) |
| `quat_z2vec(vec)` | Default `wp.quat(0.0, 0.0, 0.0, 1.0)`, returned when `‖vec‖ < MJ_MINVAL` or when `a = ‖(−v1, v0, 0)‖ < MJ_MINVAL` and `v2 ≥ 0`; `wp.quat(1,0,0,0)` when `a < MJ_MINVAL` and `v2 < 0`; otherwise `axis_angle_to_quat(axis/a, atan2(a, v2))`. Elsewhere MJWarp indexes quaternions in MuJoCo order (`q[0]` = w). Read that way, the two degenerate returns are `(0,0,0,1)` and `(1,0,0,0)`. C `mju_quatZ2Vec` returns `(1,0,0,0)` and `(0,1,0,0)` (180° about x) in those cases. The degenerate cases do not match C, so check them before porting |
| `quat_inv(q)` | conjugate |
| `inert_vec(i, v)` | `mju_mulInertVec` on 10-vector inertia `(I00,I11,I22,I01,I02,I12, c0,c1,c2, m)` |
| `motion_cross`, `motion_cross_force` | spatial cross products (rotation first) |
| `quat_to_vel(q)` | `axis=(q1,q2,q3)`, `s = ‖axis‖`; `s==0` → 0; `speed = 2 atan2(s, q0)`, `>π` → `−2π`; `axis*speed/s` |
| `quat_sub(qa, qb)` | `quat_to_vel(conj(qb)*qa)` |
| `quat_integrate(q, v, dt)` | `angle = dt‖v‖`, `q_res = axis_angle(normalize(v), angle)`; `q = normalize(q)`; `normalize(q*q_res)` |
| `deriv_expmap(expmap, vel)` | port of `mju_derivExpmap` (angle reduced to (−π, π] if `xx > π²`), float32 coefficients |
| `orthogonals(a)` | `b = (−0.5<a1<0.5) ? y : z`; Gram–Schmidt; zero if `‖a‖==0`; `c = a×b` |
| `orthonormal(normal)`, `orthonormal_to_z` | pick smallest component; `gjk_normalize` |
| `gjk_normalize(a)` | normalise only if `1e-8 < ‖a‖ < 1e12`, returns flag |
| `make_frame(a)` | rows `(a, b, c)` |
| `normalize_with_norm(x)` | `‖x‖==0` → `(x, 0)` else `(x/‖x‖, ‖x‖)` (division, vs C's reciprocal multiply; and threshold exactly 0 vs C's `mjMINVAL` with fallback `(1,0,0)`) |
| `closest_segment_point(_and_dist)`, `closest_segment_to_segment_points` | collision helpers; latter uses `denom + 1e-6` |
| `safe_div(x, y)` | `x / (y != 0 ? y : MJ_MINVAL)` |
| `upper_tri_index(n,i,j)` | `(i(2n−i−3))//2 + j − 1` (strict upper) |
| `upper_trid_index(n,i,j)` | swaps to `i≤j`, `(i(2n−i−1))//2 + j` (incl. diagonal) |
| `free_bias_vel_blocks(mass, R, Iw, s, qvel_rot)` | 3×3 blocks of ∂qfrc_bias/∂qvel for a free subtree (derivatives) |
| `lu_factor_6x6(A)` / `lu_solve_6x6(A, piv, b)` | port of `mju_factorLU6`/`mju_solveLU6`; differences: `pivot[k] = maxrow` always written; returns `(A, pivot, ok)`; failure returns partially-factored A |

Warp `wp.normalize`, `wp.length`, `wp.dot`, `wp.cross` are builtins (dot of vec3 computes
`x*x+y*y+z*z` sequentially, possibly FMA-contracted).

### 14.3 Mass-matrix factorization on GPU (`smooth.factor_m`, `solve_LD`)

C: `mj_factorI` (sparse reverse `L'DL`, clamps pivots `< mjMINVAL` up to `mjMINVAL`, uses
`mju_addToScl`/`mju_scl`), `mj_solveLD` (backward column axpy, `D^{-1}`, forward `mju_dotSparse`).

MJWarp (`io.m_block_layout`): `M`'s diagonal blocks are the kinematic trees (`tree_dofadr`,
`tree_dofnum`). Per tree block of `size` dofs with `nnz` stored entries:

| Condition | Path | Factor storage |
|---|---|---|
| `size ≤ M_BLOCK_SCALAR_MAX (6)` and compact (`nnz == size`, diagonal-only) | scalar; `D = 1/M_ii` | none (`qLD_block_adr = Q_LD_BLOCK_COMPACT`) |
| `size ≤ 6` and triangular (`nnz == size(size+1)/2`) | `_small_cholesky_factorize_block`: per-thread scalar **Cholesky** (`L Lᵀ`, not `L'DL`): `L_ii = sqrt(M_ii − Σ_k L_ki²)`, `L_ij = (M_ji − Σ L_ki L_kj)·(1/L_ii)` stored row `i` col `j` in a dense `size×size` slab (upper fill) | `qLD[qLD_block_adr + i*size + j]` |
| `size ≤ M_BLOCK_DENSE_MAX (64)` | `_tile_cholesky_factorize_block`: `tile_load_indexed` gathers dense block from CSR (absent pairs read 0), `wp.tile_cholesky_inplace(L, fill_mode="upper")` | dense slab |
| larger | sparse L'DL by tree depth levels (below) | region after `qLD_block_total` |

Sparse L'DL (`_factor_i_sparse`): updates `(i, k, Madr_ki)` grouped by **depth of i** (deepest first);
each level launched as `(nworld, nupdates)`:
`tmp = L[Madr_ki]/L[diag_k]` (division, not reciprocal-multiply), then
`wp.atomic_sub(L[Madr_i + j], L[M_rowadr[k]+j]*tmp)` for `j < M_rownnz[i]`, and `L[Madr_ki] = tmp`.
Finally `D = 1/L[diag]` (`_qLDiag_div`). **No pivot clamping** (C clamps to `mjMINVAL` and warns).
Atomics make the summation order of concurrent updates to the same row nondeterministic unless
`DeterminismType.ATOMICS` (then: copy `L_in`, launch with `wp.DeterministicMode.RUN_TO_RUN`,
`deterministic_max_records = nv`, and apply `_qLD_scale` in a separate launch).

`solve_LD`: blocks → `_small_cholesky_solve_block` (compact: `x = D*y`; else forward/back substitution
with divisions by `L_ii`) or `_tile_cholesky_solve_block` (`wp.tile_cholesky_solve(..., "upper")`,
block_dim `cholesky_solve` if `size ≤ 40` else 32). Sparse region: fused kernel
`_solve_LD_sparse_fused(nv, nlevels)` (one CTA per world, `__syncthreads` between levels; atomic
updates per level; UP over levels deepest→root, diag multiply by `D`, DOWN root→deepest), or with
`ATOMICS`: per-level launches with `RUN_TO_RUN` determinism and a scratch copy of `x`.

Consequence: for trees with ≤64 dofs MJWarp uses dense **Cholesky** `LLᵀ` in float32 on the
tree block, a different factorization (and different rounding) than C's sparse `L'DL`. Results agree
only to float32 conditioning (`κ(M)·ε32`).

### 14.4 Newton Hessian factorization (`solver.py`, `block_cholesky.py`)

- Hessian `h` is dense `[nworld, nv_pad, nv_pad]`, built by tiled `JᵀDJ + M` kernels
  (`_update_gradient_JTDAJ_dense_tiled*`, tile `TILE_SIZE_JTDAJ_DENSE=16`); padded diagonal entries set
  to 1.0 (`_padding_h`); padded gradient entries 0.
- Padding (`io._get_padded_sizes`): `nv_pad = round_up(nv + augment, 16)` if sparse or `nv > 32`, else
  `round_up(nv, 4)`; `augment = 1` when solver is Newton and `nv > 32`.
- `nv ≤ _BLOCK_CHOLESKY_DIM (32)`: `_update_gradient_cholesky(nv)` — single tile `(nv, nv)`:
  `wp.tile_cholesky_inplace(fill_mode="upper")`, `wp.tile_cholesky_solve`, then
  `sums = Σ solve_search_sums(grad, x) = (Σ x², Σ grad·x)` → `search_dot`, `newton_decrement`;
  `search = −x`. One CTA per world (`block_dim.update_gradient_cholesky`).
- `nv > 32`: blocked right-looking Cholesky with 16×16 tiles in shared memory
  (`create_blocked_cholesky_augmented_factorize_solve_newton_func`):
  - Upper factor `U` (`A = UᵀU`), stored to `ctx.hfactor`.
  - For diagonal block `k`: `A_kk −= Σ_{j<k} U_jkᵀ U_jk` (`tile_matmul(..., alpha=-1)`),
    `tile_cholesky_inplace(upper)`; panels `A_ki −= Σ_j U_jkᵀ U_ji`, then `tile_lower_solve_inplace(U_kkᵀ, A_ki)`.
  - **Augmented RHS trick**: the last padded column (index `nv_pad−1`, guaranteed free by `augment=1`)
    is overwritten with the gradient `b` (rows of the last block row via `input_diagonal_rhs`, other
    block rows via `input_rhs` into column `border_size = 15` of panels with `i+16 == matrix_size`),
    and `A[last,last] = 1.0e30`. Cholesky of `[[H, b],[bᵀ, 1e30]]` yields `U_{:,last} = U_H^{-T} b`
    (the forward solve) for free; these border values are gathered into `rhs_tile`. Comment: "Keep the
    augmented rank-one correction below float32 precision" (`U_last,last ≈ 1e15`).
  - Backward solve block-wise: `tmp −= U_ij x_j` (`tile_matmul`), `tile_upper_solve_inplace(U_ii, tmp)`.
  - Newton decrement: `(Σ x², Σ grad·x)` via `tile_reduce`; `search = −x`.
  - `_update_gradient_cholesky_blocked_skip_unchanged`: if no constraint changed quadratic state
    (`quad_changed_count == 0`), reuse `hfactor` with `create_blocked_cholesky_solve_newton_func`
    (forward+backward only); with `skip_noflip`, worlds with `state_changed_count==0` keep the previous
    search direction ("linesearch is invariant to direction scale").
  - Non-augmented `create_blocked_cholesky_factorize_solve_func` (forward solve interleaved with
    factorization) is used by `_cholesky_factorize_solve_blocked` (other callers, e.g. derivatives).
- Differences vs C: C Newton uses sparse reverse Cholesky (`mju_cholFactorNumeric`) with
  `mindiag = mjMINVAL` clamping and rank-1 incremental updates (`mju_cholUpdate(Sparse)`), and C
  elliptic cones get exact cone Hessian folding; MJWarp refactorizes the dense Hessian (or reuses) with
  no rank-deficiency clamping (Warp `tile_cholesky` on a non-PD tile yields NaN), float32 arithmetic,
  tile-based summation order.
- Warp function factories are memoised with `functools.lru_cache` (block_cholesky) and kernels with
  `warp_util.cache_kernel` (key = hashed args + module + function name; arrays hash by `.size`):
  "caching kernels to avoid crashes in graph_conditional code" — kernels must not be created inside
  `wp.capture_while`.

### 14.5 Graph capture and tooling (`warp_util.py`)

- `check_toolkit_driver()`: warns if `wp.is_conditional_graph_supported()` is false (CUDA < 12.4:
  graph capture unreliable < 12.3, no conditional nodes < 12.4 → set `Model.opt.graph_conditional=False`).
  With `graph_conditional`, solver iterations run in `wp.capture_while(nsolving, ...)`, so iteration
  counts are data-dependent per launch but all worlds iterate until all are done (`ctx.done` per world
  masks finished worlds).
- `EventTracer` / `event_scope`: CUDA-event timing tree; disabled inside `capture_while/capture_if`.
- `mark_batched(obj)`: flags arrays whose spec's leading dim is `"*"` or `"nworld"` (`_is_batched`) —
  Model fields with leading `*` may be batched per world (`batch_sizes` in `put_model`), indexed with
  `worldid % shape[0]`.
- `is_array_spec`.

### 14.6 Determinism on GPU (`types.DeterminismType`, `determinism_test.py`)

`Option.deterministic` bitmask (`DeterminismType`, IntFlag): `NONE=0`, `CONTACTS=1` (stable contact
ordering via geometric radix sort), `CONSTRAINT=2` (canonical constraint row ordering), `ATOMICS=4`
(invariant floating-point atomic reductions), `ALL=7`. Accepts bools (`True→ALL`), ints, names
(`"contacts | atomics"`), negation (`~CONTACTS`). Default `NONE`.

- Default (NONE): floating-point `atomic_add/sub` in many kernels (qLD updates, sparse solve, JᵀDJ,
  passive forces, contact Jacobians, flex normals, sleep, etc.) → summation order depends on thread
  scheduling → **run-to-run nondeterminism** in the last bits; contact and `efc` row order depend on
  atomic slot allocation.
- ATOMICS: each factory taking a `deterministic` argument compiles a variant with
  `module_options["deterministic"] = wp.DeterministicMode.RUN_TO_RUN` and
  `deterministic_max_records` (number of contributions per output to record, e.g. `nv`, `njmax`,
  `nv(nv−1)/2`, `EFM_CON_MAX_DOF(EFM_CON_MAX_DOF+1)/2`) so Warp records atomic contributions and
  reduces them in a fixed order. Some algorithms switch structure (e.g. sparse solve uses per-level
  launches instead of the fused kernel; qLD writes go through a copy + separate scale kernel).
- Test contract (`_check_scene`): (1) NONE launches no RUN_TO_RUN kernels; (2) ATOMICS launches
  RUN_TO_RUN exactly for factories with a `deterministic` parameter; (3) NONE vs ATOMICS physics fields
  agree within `rtol=atol=1e-3` ("only differ in floating-point summation order"); (4) repeated ATOMICS
  runs (`_NREPEAT=2` × `_NSTEP=3` steps) are **bitwise identical** across all Model/Data arrays (with
  contacts/rows replayed when CONTACTS/CONSTRAINT are not set); (5) "records-only" factory arguments
  are 0 on the default path so the default kernels are not recompiled.
- Determinism here means **run-to-run on the same GPU/driver/Warp version**, not cross-device or
  cross-backend equality, and never equality with C MuJoCo.

### 14.7 Other ported utilities

`util_misc.halton` (identical algorithm, float32), `_sigmoid` (identical expression), `wrap*`
(see §11.5), `lugre_stribeck`, `dcmotor_slots` (returns `vec6i`), muscle functions (other spec).

### 14.8 MJX (JAX) notes

- Precision: float32 unless `jax_enable_x64` is set (then float64 fields mostly). XLA may fuse/reorder.
- `math.safe_div(num, den) = num / (den + mjMINVAL*(den==0))`; `norm(x)` is gradient-safe (`allclose(x,0)`
  → swap with ones, return 0); `normalize_with_norm`: `x/(n + 1e-6*(n==0))`; `sign(x) ∈ {−1, 1}`
  (`x<0 ? −1 : 1`, differs from `mju_sign` at 0); `matmul_unroll` explicit triple loop for small
  matrices; `deriv_expmap` picks F64 or F32 Bortz coefficients by dtype, polynomial threshold
  `xx < 1.75²`; `quat_integrate` (no pre-normalisation of q, unlike Warp).
- Mass matrix: dense models → `jax.scipy.linalg.cho_factor/cho_solve` (LAPACK/cuSolver Cholesky); sparse
  models → depth-levelled L'DL with `qld.at[out].add(-(qld[madr]/qld[pivot])*qld[rows])` scatter-adds
  (then rows divided by diagonals), `qLDiagInv = 1/diag`; no clamping.
- Newton solver: `cho_factor` of the symmetrised Hessian; collision dot products use
  `precision=jax.lax.Precision.HIGH` (on TPU this avoids bf16 passes; on GPU with TF32-enabled matmuls,
  default precision may use TF32 — MJX explicitly raises it where it matters).

---

## 15. Reproduction pitfalls / parity notes

1. **`mju_dot` order**: 4 interleaved accumulators, combined `(r0+r2)+(r1+r3)`, then grouped tail
   `res += (p0+p1+p2)`. `mju_dotSparse` uses the same lanes but a *sequential* tail. `mju_sum`, `mju_L1`,
   `mju_dotInd`, `mju_dotSparse2` are fully sequential. Mixing these up changes bits everywhere.
2. **`mju_dotSparseX3`** non-AVX is sequential; AVX is 4-lane — the default x86 build (AVX) differs from
   portable builds (`MUJOCO_ENABLE_AVX_INTRINSICS=OFF`, ARM, single precision).
3. **Zero-skipping** in `mju_mulMatTVec`, `mju_mulMatMat`, `mju_mulMatTMat`, `mju_sqrMatTD_impl`
   (`dj==0`, `tmp==0`), `mju_cholSolve` (backward `res[i]==0`), `mju_cholUpdate` (`x[k]==0`),
   `mju_mulMatTVecSparse` (`vec[i]==0`), `mju_sqrMatTDSparseNumeric` (`scale==0`, `diag[k]==0`),
   `mju_cholSolveSparse` single-row backward (`res[i]==0`) — must be replicated (matters for `−0`,
   Inf/NaN propagation and for which ops occur).
4. **Reciprocal multiply vs divide**: `mju_normalize*` (reciprocal), `mju_cholFactor` off-diagonals
   (reciprocal), `mju_cholFactorNumeric`/`Sparse` (reciprocal `1.0/L_rr`), `mju_cholSolve*` (divide),
   `mju_factorLU` (reciprocal), `mju_factorLUSparse` and `mj_factorI` diag (`invD = 1/x` then multiply),
   `mju_Halton` (`f /= b`). MJWarp `normalize_with_norm` divides.
5. **`mju_normalize4`** leaves a quaternion untouched if `|‖q‖−1| ≤ 1e-15`; normalising unconditionally
   changes bits.
6. **Thresholds**: `mju_cholUpdate(Sparse)` clamps `tmp<mjMINVAL`; dense/sparse factor clamp `<mindiag`
   and dense zeros the column (sparse numeric zeros the row); band Cholesky **returns 0** on `<mjMINVAL`;
   LU sparse clamps magnitude `<mjMINVAL` preserving sign; QCQP uses absolute `1e-10`; boxQP `mingrad
   1e-16` is squared; eig3 `tol = scale*max(reltol, 4e-15)`, sort `eps = scale*1e-12`.
7. **Supernode-dependent arithmetic**: `mju_cholFactorNumeric` (ns≤3, batch≤4/3), `mju_cholSolveSparse`
   (2/3-row fusion uses sequential sums, single rows use 4-lane dotSparse), `mju_sqrMatTDSparseNumeric`
   (ns≤8 with lanes), `mju_mulMatVecSparse` AVX (X3 blocks). The structure (rowsuper/etree) must be
   computed exactly as specified to get the same arithmetic.
8. **`mju_sqrMatTDSparseNumeric` diagonal**: `(diag[k]*M[k,r])*M[k,r]` per term, accumulated over k in
   `colindT` order; the legacy `mju_sqrMatTDSparse` uses `(M*M)*diag` or 4-lane `mju_dot`. Don't mix.
9. **Combine ordering**: `mju_combineSparse` merges *backwards*; common entries `a*dst + b*src`; with
   `a != 1` dst-only entries are scaled. `mju_addToMatSparse` passes `a=b=1` (adds exactly).
10. **eig3 compile vs runtime**: compiler `mjuu_eig3` uses `eps = scale*max(reltol, 1e-12)` for sorting
    and a normalisation that divides; engine `mju_eig3Tol` uses `scale*eigEPS` and reciprocal-multiply
    normalisation — principal inertia frames come from the compiler version.
11. **Strict comparisons** in argmax selections (eig3 off-diagonal choice, LU pivot `val > maxval`,
    wrap `axis1` component choice) resolve ties toward the first candidate.
12. **`mju_max/mju_min` NaN behavior** differs from `fmax/fmin`/Warp `wp.max`.
13. **libm**: transcendental results differ by platform; this changes e.g. springDamper, wrapping,
    eig3 is sqrt-only (portable). QCQP2/3 use only `+ − × ÷`, so they reproduce exactly.
14. **ARM / Clang FMA contraction** (default `-ffp-contract=on`) changes results versus x86 builds;
    reproduce x86 bits with `-ffp-contract=off`.
15. **Single precision C build** (`mjUSESINGLE`) disables AVX and switches many tolerances; use it as the
    closest CPU reference for MJWarp/MJX (float32) comparisons (verify `mujoco.MJTNUM_DTYPE == float32`),
    though bitwise equality with GPU is still not attainable (different algorithms, FMA, atomics).
16. **MJWarp mass matrix** uses dense `LLᵀ` (scalar ≤6 dofs triangular, tile ≤64) or level-parallel
    `L'DL` with atomics; no `mjMINVAL` pivot clamp (C clamps and raises `mjWARN_INERTIA`).
17. **MJWarp Newton** uses dense float32 Hessian with blocked tile Cholesky (16×16), augmented-column
    forward solve with `1e30` border, optional factor reuse when no constraint changed; C uses sparse
    reverse Cholesky with incremental rank-1 updates. Solver `tolerance` floored to `1e-6`.
18. **MJWarp fast-math** in the linesearch kernel approximates `div`/`sqrt` (`--use_fast_math`), and CUDA
    FMA contraction is on by default — expect ulp-level differences from IEEE-exact formulas.
19. **Atomics nondeterminism**: unless `opt.deterministic` includes `ATOMICS`, MJWarp is not run-to-run
    bitwise reproducible; contact/efc ordering additionally needs `CONTACTS`/`CONSTRAINT`.
20. **Padding**: MJWarp pads `nv` to `nv_pad` (multiple of 4 or 16, +1 augmented column for Newton when
    `nv>32`) and `njmax` to a multiple of 16; padded Hessian diagonal = 1, padded gradient = 0. Any
    reimplementation of the GPU path must keep the padding to reproduce tile shapes and reduction order.
21. **Sort stability**: C `mjSORT` is stable (insertion runs of 32 + stable merges, ties keep input order);
    `mjPARTIAL_SORT` is heap-based (ties unstable). GPU radix/segmented sorts give different tie orders
    unless `DeterminismType.CONTACTS`.
22. **`mju_standardNormal`** depends on C `rand()`; never use it in a reproducible pipeline.
23. **Allocation alignment** (64-byte heap, natural-alignment stack) has no numerical effect since SIMD
    uses unaligned loads; but stack capacity limits (`mjWARN_CNSTRFULL`) can change control flow (the
    sparse Newton setup bails out when `mj_stackBytesAvailable` is insufficient).

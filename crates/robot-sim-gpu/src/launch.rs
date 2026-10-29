//! GPU launch layer with CubeCL compute kernels.
//!
//! The `launch_*` functions are crate-private. Each one validates every buffer
//! length against its element count, and checks the device buffer size, before
//! the single `unsafe` launch block. The public wrappers in `integrators`,
//! `passive`, `constraint` and `contact` are the only callers.

use crate::error::GpuSimError;

/// Largest storage buffer, in bytes, that the launch layer will request. It is
/// wgpu's default `max_storage_buffer_binding_size` (128 MiB). This is a
/// conservative constant, not a queried device limit.
pub const MAX_BUFFER_BYTES: usize = 128 * 1024 * 1024;

/// Largest workgroup count one dispatch may request along x. The wgpu default,
/// not a queried device limit.
pub const MAX_CUBES_PER_DISPATCH: usize = 65_535;

/// Threads per workgroup. Every kernel is one thread per element.
pub const BLOCK_SIZE: usize = 256;

/// Checks that a batch of `n` elements, whose widest buffer holds
/// `n * stride` `f32` values, fits the single-dispatch limits. The check
/// uses counts only, so it allocates nothing and never touches the device.
///
/// On success, returns the strided element count `n * stride`.
///
/// # Errors
///
/// Returns [`GpuSimError::InvalidParameter`] if the workgroup count exceeds
/// 65535, the widest buffer exceeds 128 MiB, or `n * stride` overflows.
pub fn check_batch_size(n: usize, stride: usize) -> Result<usize, GpuSimError> {
    if n.div_ceil(BLOCK_SIZE) > MAX_CUBES_PER_DISPATCH {
        return Err(GpuSimError::InvalidParameter {
            what:
                "batch length exceeds the single-dispatch limit of 65535 workgroups x 256 threads",
        });
    }
    let elements = n.checked_mul(stride).ok_or(GpuSimError::InvalidParameter {
        what: "strided element count overflows usize",
    })?;
    let bytes = elements
        .checked_mul(4)
        .ok_or(GpuSimError::InvalidParameter {
            what: "buffer byte size overflows usize",
        })?;
    if bytes > MAX_BUFFER_BYTES {
        return Err(GpuSimError::InvalidParameter {
            what: "buffer exceeds the 128 MiB storage binding limit",
        });
    }
    Ok(elements)
}

/// Largest degrees-of-freedom count per environment that the per-env dense
/// kernels support. The kernel keeps a `6 x 6` matrix and a 6-vector in
/// per-thread local arrays, so a larger system would write out of bounds.
pub const MAX_DOF: usize = 6;

/// Preflight for the per-env dense solve with `nv` degrees of freedom and `n`
/// environments. Checks `1 <= nv <= MAX_DOF` and the batch limits for the
/// widest (`nv * nv` per env) buffer, using counts only.
///
/// On success, returns `(n * nv, n * nv * nv)`: the vector and matrix element
/// counts.
///
/// # Errors
///
/// Returns [`GpuSimError::InvalidParameter`] if `nv` is 0 or above
/// [`MAX_DOF`], or if [`check_batch_size`] rejects the batch.
pub fn check_implicit_batch(nv: usize, n: usize) -> Result<(usize, usize), GpuSimError> {
    if nv == 0 || nv > MAX_DOF {
        return Err(GpuSimError::InvalidParameter {
            what: "nv must be between 1 and 6 for the per-env dense solve",
        });
    }
    let vec_len = check_batch_size(n, nv)?;
    let mat_len = check_batch_size(n, nv * nv)?;
    Ok((vec_len, mat_len))
}

/// Count-only preflight for equality-only systems. Returns `(n*nv, n*nv²,
/// n*nrow, n*nrow*nv, n*nrow²)`; all device buffers are bounded independently.
pub fn check_equality_batch(
    nv: usize,
    nrow: usize,
    n: usize,
) -> Result<(usize, usize, usize, usize, usize), GpuSimError> {
    if nv == 0 || nv > MAX_DOF || nrow == 0 || nrow > MAX_DOF {
        return Err(GpuSimError::InvalidParameter {
            what: "nv and nrow must be between 1 and 6",
        });
    }
    let vec_len = check_batch_size(n, nv)?;
    let mass_len = check_batch_size(n, nv * nv)?;
    let row_len = check_batch_size(n, nrow)?;
    let jac_len = check_batch_size(n, nrow * nv)?;
    let ar_len = check_batch_size(n, nrow * nrow)?;
    check_batch_size(n, nrow * 5)?;
    let strides = equality_strides(nv, nrow);
    check_batch_size(n, strides.input)?;
    check_batch_size(n, strides.output)?;
    check_batch_size(n, strides.flags)?;
    Ok((vec_len, mass_len, row_len, jac_len, ar_len))
}

/// Bytes per element. Every device buffer in this crate holds `f32` or `u32`.
const ELEMENT_BYTES: usize = 4;

/// Storage bindings in one RK4 pendulum launch: theta, omega and their outputs.
pub const RK4_BINDINGS: usize = 4;

/// Storage bindings in one equality launch: packed input, output and flags.
pub const EQUALITY_BINDINGS: usize = 3;

/// Per-environment element strides of the packed equality batch. The launch
/// packing and [`equality_resources`] both read these, so each formula is
/// written once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EqualityStrides {
    /// `mass` (`nv*nv`), `qvel`, `smooth` (`nv` each), `jac` (`nrow*nv`),
    /// `pos` (`nrow`), `solref` (`2*nrow`), `solimp` (`5*nrow`).
    pub input: usize,
    /// `force` (`nrow`) then `qacc` (`nv`).
    pub output: usize,
    /// `clamped` (`nrow`) then `status` (1).
    pub flags: usize,
}

pub(crate) fn equality_strides(nv: usize, nrow: usize) -> EqualityStrides {
    EqualityStrides {
        input: nv * nv + 2 * nv + nrow * nv + 8 * nrow,
        output: nv + nrow,
        flags: nrow + 1,
    }
}

/// Device bytes of one RK4 pendulum call, by buffer. Count-only: nothing is
/// allocated and the device is not touched. The two `theta_in`/`omega_in`
/// buffers are created from host slices; the two outputs are created empty
/// and read back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rk4Resources {
    /// Input `theta`, `n` `f32`.
    pub theta_in_bytes: usize,
    /// Input `omega`, `n` `f32`.
    pub omega_in_bytes: usize,
    /// Output `theta`, `n` `f32`.
    pub theta_out_bytes: usize,
    /// Output `omega`, `n` `f32`.
    pub omega_out_bytes: usize,
    /// Storage bindings per call.
    pub bindings: usize,
}

impl Rk4Resources {
    /// Bytes allocated on the device per call.
    pub fn allocated_bytes(&self) -> usize {
        self.theta_in_bytes + self.omega_in_bytes + self.theta_out_bytes + self.omega_out_bytes
    }

    /// Host-to-device bytes per call.
    pub fn upload_bytes(&self) -> usize {
        self.theta_in_bytes + self.omega_in_bytes
    }

    /// Device-to-host bytes per call.
    pub fn readback_bytes(&self) -> usize {
        self.theta_out_bytes + self.omega_out_bytes
    }
}

/// Count-only RK4 accounting for `n` environments, from the same batch check
/// the launch runs.
///
/// # Errors
///
/// Returns [`GpuSimError::InvalidParameter`] if [`check_batch_size`] rejects `n`.
pub fn rk4_resources(n: usize) -> Result<Rk4Resources, GpuSimError> {
    let bytes = check_batch_size(n, 1)? * ELEMENT_BYTES;
    Ok(Rk4Resources {
        theta_in_bytes: bytes,
        omega_in_bytes: bytes,
        theta_out_bytes: bytes,
        omega_out_bytes: bytes,
        bindings: RK4_BINDINGS,
    })
}

/// Device bytes of one equality batch call, by buffer. Count-only: nothing is
/// allocated. All three buffers are created from host data, so the upload is
/// the full allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EqualityResources {
    /// Packed input, `n * (nv*nv + 2*nv + nrow*nv + 8*nrow)` `f32`.
    pub input_bytes: usize,
    /// Output (force then qacc), `n * (nv + nrow)` `f32`.
    pub output_bytes: usize,
    /// Flags (clamped then status), `n * (nrow + 1)` `u32`.
    pub flags_bytes: usize,
    /// Storage bindings per call.
    pub bindings: usize,
}

impl EqualityResources {
    /// Bytes allocated on the device per call.
    pub fn allocated_bytes(&self) -> usize {
        self.input_bytes + self.output_bytes + self.flags_bytes
    }

    /// Host-to-device bytes per call.
    pub fn upload_bytes(&self) -> usize {
        self.allocated_bytes()
    }

    /// Device-to-host bytes per call: output and flags.
    pub fn readback_bytes(&self) -> usize {
        self.output_bytes + self.flags_bytes
    }
}

/// Count-only equality accounting for `n` environments, from the same checks
/// and strides the launch packs with.
///
/// # Errors
///
/// Returns [`GpuSimError::InvalidParameter`] if the dimensions or the batch
/// are rejected by [`check_equality_batch`].
pub fn equality_resources(
    nv: usize,
    nrow: usize,
    n: usize,
) -> Result<EqualityResources, GpuSimError> {
    check_equality_batch(nv, nrow, n)?;
    let strides = equality_strides(nv, nrow);
    Ok(EqualityResources {
        input_bytes: check_batch_size(n, strides.input)? * ELEMENT_BYTES,
        output_bytes: check_batch_size(n, strides.output)? * ELEMENT_BYTES,
        flags_bytes: check_batch_size(n, strides.flags)? * ELEMENT_BYTES,
        bindings: EQUALITY_BINDINGS,
    })
}

/// Checks that each named slice has the expected element count, returning the
/// first mismatch as a typed error.
pub(crate) fn check_lengths(
    expected: usize,
    lengths: &[(&'static str, usize)],
) -> Result<(), GpuSimError> {
    for &(what, found) in lengths {
        if found != expected {
            return Err(GpuSimError::DimensionMismatch {
                what,
                expected,
                found,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Count-only checks: nothing is allocated and the device is not touched.

    #[test]
    fn length_mismatch_names_the_slice() {
        assert_eq!(
            check_lengths(4, &[("out", 3)]),
            Err(GpuSimError::DimensionMismatch {
                what: "out",
                expected: 4,
                found: 3,
            })
        );
    }

    #[test]
    fn rejects_more_than_the_workgroup_cap() {
        let n = MAX_CUBES_PER_DISPATCH * BLOCK_SIZE + 1;
        assert!(matches!(
            check_batch_size(n, 1),
            Err(GpuSimError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn accepts_the_last_full_workgroup_count() {
        let n = MAX_CUBES_PER_DISPATCH * BLOCK_SIZE;
        assert_eq!(check_batch_size(n, 1).unwrap(), n);
        assert_eq!(check_batch_size(4, 5).unwrap(), 20);
    }

    #[test]
    fn strided_buffer_above_binding_limit_is_rejected() {
        // 8M rows pass the workgroup cap (31250 cubes), but a 5-wide buffer
        // is 5 * 8M * 4 bytes = 160 MB, above the 128 MiB limit.
        let n = 8_000_000usize;
        assert!(check_batch_size(n, 1).is_ok());
        assert!(matches!(
            check_batch_size(n, 5),
            Err(GpuSimError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn stride_overflow_is_a_typed_error() {
        assert!(matches!(
            check_batch_size(usize::MAX / 2, 5),
            Err(GpuSimError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn rk4_accounting_for_4097_envs_is_per_buffer() {
        let r = rk4_resources(4097).unwrap();
        assert_eq!(r.theta_in_bytes, 16388);
        assert_eq!(r.omega_in_bytes, 16388);
        assert_eq!(r.theta_out_bytes, 16388);
        assert_eq!(r.omega_out_bytes, 16388);
        assert_eq!(r.allocated_bytes(), 65552);
        assert_eq!(r.upload_bytes(), 32776);
        assert_eq!(r.readback_bytes(), 32776);
        assert_eq!(r.bindings, 4);
    }

    #[test]
    fn equality_accounting_for_4097_envs_is_per_buffer() {
        // nv = nrow = 2: input stride 4 + 4 + 4 + 16 = 28, output stride 4,
        // flags stride 3.
        let r = equality_resources(2, 2, 4097).unwrap();
        assert_eq!(r.input_bytes, 458_864);
        assert_eq!(r.output_bytes, 65_552);
        assert_eq!(r.flags_bytes, 49_164);
        assert_eq!(r.allocated_bytes(), 573_580);
        assert_eq!(r.upload_bytes(), 573_580);
        assert_eq!(r.readback_bytes(), 114_716);
        assert_eq!(r.bindings, 3);
    }

    #[test]
    fn equality_accounting_rejects_out_of_range_dimensions() {
        assert!(equality_resources(0, 2, 4).is_err());
        assert!(equality_resources(2, 7, 4).is_err());
    }
}

#[cfg(feature = "cubecl")]
#[allow(unsafe_code)]
pub(crate) mod backend {
    use super::*;
    use cubecl::prelude::*;
    use cubecl::wgpu::{AutoCompiler, WgpuRuntime};
    use cubecl_runtime::runtime::Runtime as _;

    // --- KERNELS ---

    #[cube(launch)]
    pub fn semi_implicit_euler_kernel(q: &mut [f32], qd: &mut [f32], qacc: &[f32], dt: f32) {
        let idx = ABSOLUTE_POS;
        if idx < q.len() {
            qd[idx] += dt * qacc[idx];
            q[idx] += dt * qd[idx];
        }
    }

    /// One thread per environment. Mirrors `robot_sim::integrators::
    /// implicit_damping_euler_step`: builds `M + dt*diag(B)` in a local array,
    /// factors it by Cholesky in the CPU loop order (lower triangle only),
    /// solves for `qacc`, then applies semi-implicit Euler. An environment whose
    /// pivot is not positive and finite keeps its state and gets `status = 0`.
    #[cube(launch)]
    pub fn implicit_damping_euler_kernel(
        q: &mut [f32],
        qd: &mut [f32],
        mass: &[f32],
        damping: &[f32],
        bias: &[f32],
        status: &mut [u32],
        nv: u32,
        dt: f32,
    ) {
        let idx = ABSOLUTE_POS;
        if idx < status.len() {
            let n = nv as usize;
            let vbase = idx * n;
            let mbase = idx * n * n;
            let mut a = Array::<f32>::new(36usize);
            let mut b = Array::<f32>::new(6usize);
            for i in 0..n {
                for j in 0..n {
                    a[i * n + j] = mass[mbase + i * n + j];
                }
                a[i * n + i] += dt * damping[vbase + i];
                b[i] = bias[vbase + i];
            }

            let mut ok = 1u32;
            for j in 0..n {
                let mut d = a[j * n + j];
                for k in 0..j {
                    d -= a[j * n + k] * a[j * n + k];
                }
                if d > 0.0f32 && d < 3.0e38f32 {
                    let l = f32::sqrt(d);
                    a[j * n + j] = l;
                    for i in (j + 1)..n {
                        let mut s = a[i * n + j];
                        for k in 0..j {
                            s -= a[i * n + k] * a[j * n + k];
                        }
                        a[i * n + j] = s / l;
                    }
                } else {
                    ok = 0u32;
                }
            }

            if ok == 1u32 {
                for i in 0..n {
                    let mut s = b[i];
                    for k in 0..i {
                        s -= a[i * n + k] * b[k];
                    }
                    b[i] = s / a[i * n + i];
                }
                for r in 0..n {
                    let i = n - 1 - r;
                    let mut s = b[i];
                    for k in (i + 1)..n {
                        s -= a[k * n + i] * b[k];
                    }
                    b[i] = s / a[i * n + i];
                }
                for i in 0..n {
                    qd[vbase + i] += dt * b[i];
                    q[vbase + i] += dt * qd[vbase + i];
                }
                status[idx] = 1u32;
            } else {
                status[idx] = 0u32;
            }
        }
    }

    /// Equality-only `solve_equality`, MuJoCo 3.17.0 `engine_core_constraint.c`
    /// `getsolparam`, `getimpedance`, `mj_makeImpedance`,
    /// `mj_referenceConstraint`, `mj_makeARNumeric` (DIAGEXACT diagonal).
    /// Mirrors the CPU row order and two Cholesky solve stages. Each thread
    /// owns one environment; a failed pivot leaves its outputs unchanged.
    #[cube(launch)]
    #[allow(clippy::too_many_arguments)]
    pub fn equality_solve_kernel(
        input: &[f32],
        output: &mut [f32],
        flags: &mut [u32],
        nv: u32,
        nrow: u32,
        dt: f32,
    ) {
        let idx = ABSOLUTE_POS;
        let n = nv as usize;
        let m = nrow as usize;
        let input_stride = n * n + 2 * n + m * n + 8 * m;
        let out_stride = n + m;
        if idx < flags.len() / (m + 1) {
            let base = idx * input_stride;
            let mb = base;
            let vb = base + n * n;
            let smooth_b = vb + n;
            let jb = smooth_b + n;
            let rb = jb + m * n;
            let solref_b = rb + m;
            let solimp_b = solref_b + 2 * m;
            let ob = idx * out_stride;
            let fb = idx * (m + 1);
            let mut minv_jt = Array::<f32>::new(36usize);
            let mut aref = Array::<f32>::new(6usize);
            let mut imp = Array::<f32>::new(6usize);
            let mut row_clamped = Array::<u32>::new(6usize);
            let mut a = Array::<f32>::new(36usize);
            let mut b = Array::<f32>::new(6usize);
            let mut ok = 1u32;

            for r in 0..m {
                let tc = input[solref_b + r * 2];
                let dr = input[solref_b + r * 2 + 1];
                let floor = 2.0f32 * dt;
                let mut timeconst = tc;
                row_clamped[r] = 0u32;
                if tc > 0.0f32 && tc < floor {
                    timeconst = floor;
                    row_clamped[r] = 1u32;
                }
                let sb = solimp_b + r * 5;
                let d0 = f32::clamp(input[sb], 0.0001f32, 0.9999f32);
                let d1 = f32::clamp(input[sb + 1], 0.0001f32, 0.9999f32);
                let width = f32::max(input[sb + 2], 0.0f32);
                let mid = f32::clamp(input[sb + 3], 0.0001f32, 0.9999f32);
                let p = f32::max(input[sb + 4], 1.0f32);
                let mut impedance = 0.5f32 * (d0 + d1);
                if d0 != d1 && width > 1.0e-15f32 {
                    let x = f32::abs(input[rb + r] / width);
                    if x >= 1.0f32 {
                        impedance = d1;
                    } else if x > 0.0f32 {
                        let mut y = x;
                        if p != 1.0f32 {
                            if x <= mid {
                                y = (1.0f32 / f32::powf(mid, p - 1.0f32)) * f32::powf(x, p);
                            } else {
                                y = 1.0f32
                                    - (1.0f32 / f32::powf(1.0f32 - mid, p - 1.0f32))
                                        * f32::powf(1.0f32 - x, p);
                            }
                        }
                        impedance = d0 + y * (d1 - d0);
                    } else {
                        impedance = d0;
                    }
                }
                imp[r] = impedance;
                let k = if timeconst > 0.0f32 {
                    1.0f32 / f32::max(d1 * d1 * timeconst * timeconst * dr * dr, 1.0e-15f32)
                } else {
                    -timeconst / f32::max(d1 * d1, 1.0e-15f32)
                };
                let damp = if dr > 0.0f32 {
                    2.0f32 / f32::max(d1 * timeconst, 1.0e-15f32)
                } else {
                    -dr / f32::max(d1, 1.0e-15f32)
                };
                let mut vel = 0.0f32;
                for i in 0..n {
                    vel += input[jb + r * n + i] * input[vb + i];
                }
                aref[r] = -damp * vel - k * impedance * input[rb + r];

                // Factor M anew for each row, as in the CPU port.
                for i in 0..n {
                    for j in 0..n {
                        a[i * n + j] = input[mb + i * n + j];
                    }
                    b[i] = input[jb + r * n + i];
                }
                for j in 0..n {
                    let mut d = a[j * n + j];
                    for h in 0..j {
                        d -= a[j * n + h] * a[j * n + h];
                    }
                    if d > 0.0f32 && d < 3.0e38f32 {
                        let l = f32::sqrt(d);
                        a[j * n + j] = l;
                        for i in (j + 1)..n {
                            let mut s = a[i * n + j];
                            for h in 0..j {
                                s -= a[i * n + h] * a[j * n + h];
                            }
                            a[i * n + j] = s / l;
                        }
                    } else {
                        ok = 0u32;
                    }
                }
                if ok == 1u32 {
                    for i in 0..n {
                        let mut s = b[i];
                        for h in 0..i {
                            s -= a[i * n + h] * b[h];
                        }
                        b[i] = s / a[i * n + i];
                    }
                    for rev in 0..n {
                        let i = n - 1 - rev;
                        let mut s = b[i];
                        for h in (i + 1)..n {
                            s -= a[h * n + i] * b[h];
                        }
                        b[i] = s / a[i * n + i];
                    }
                    for i in 0..n {
                        minv_jt[r * n + i] = b[i];
                    }
                }
            }

            if ok == 1u32 {
                for i in 0..m {
                    for j in 0..m {
                        let mut dot = 0.0f32;
                        for h in 0..n {
                            dot += input[jb + i * n + h] * minv_jt[j * n + h];
                        }
                        a[i * m + j] = dot;
                    }
                    let diag = a[i * m + i];
                    a[i * m + i] += f32::max((1.0f32 - imp[i]) * diag / imp[i], 1.0e-15f32);
                    let mut dot = 0.0f32;
                    for h in 0..n {
                        dot += input[jb + i * n + h] * input[smooth_b + h];
                    }
                    b[i] = aref[i] - dot;
                }
                for j in 0..m {
                    let mut d = a[j * m + j];
                    for h in 0..j {
                        d -= a[j * m + h] * a[j * m + h];
                    }
                    if d > 0.0f32 && d < 3.0e38f32 {
                        let l = f32::sqrt(d);
                        a[j * m + j] = l;
                        for i in (j + 1)..m {
                            let mut s = a[i * m + j];
                            for h in 0..j {
                                s -= a[i * m + h] * a[j * m + h];
                            }
                            a[i * m + j] = s / l;
                        }
                    } else {
                        ok = 0u32;
                    }
                }
            }
            if ok == 1u32 {
                for i in 0..m {
                    let mut s = b[i];
                    for h in 0..i {
                        s -= a[i * m + h] * b[h];
                    }
                    b[i] = s / a[i * m + i];
                }
                for rev in 0..m {
                    let i = m - 1 - rev;
                    let mut s = b[i];
                    for h in (i + 1)..m {
                        s -= a[h * m + i] * b[h];
                    }
                    b[i] = s / a[i * m + i];
                }
                for r in 0..m {
                    output[ob + r] = b[r];
                    flags[fb + r] = row_clamped[r];
                }
                for i in 0..n {
                    let mut acc = input[smooth_b + i];
                    for r in 0..m {
                        acc += b[r] * minv_jt[r * n + i];
                    }
                    output[ob + m + i] = acc;
                }
            }
            flags[fb + m] = ok;
        }
    }

    #[cube(launch)]
    pub fn joint_spring_kernel(q: &[f32], q_ref: &[f32], stiffness: &[f32], out: &mut [f32]) {
        let idx = ABSOLUTE_POS;
        if idx < q.len() {
            out[idx] = -(q[idx] - q_ref[idx]) * stiffness[idx];
        }
    }

    #[cube(launch)]
    pub fn joint_damper_kernel(qd: &[f32], damping: &[f32], out: &mut [f32]) {
        let idx = ABSOLUTE_POS;
        if idx < qd.len() {
            out[idx] = -qd[idx] * damping[idx];
        }
    }

    #[cube(launch)]
    pub fn affine_actuator_kernel(
        gain_prm: &[f32],
        bias_prm: &[f32],
        ctrl: &[f32],
        length: &[f32],
        velocity: &[f32],
        out: &mut [f32],
    ) {
        let idx = ABSOLUTE_POS;
        if idx < ctrl.len() {
            let g0 = gain_prm[idx * 3];
            let g1 = gain_prm[idx * 3 + 1];
            let g2 = gain_prm[idx * 3 + 2];
            let gain = g0 + g1 * length[idx] + g2 * velocity[idx];

            let b0 = bias_prm[idx * 3];
            let b1 = bias_prm[idx * 3 + 1];
            let b2 = bias_prm[idx * 3 + 2];
            let bias = b0 + b1 * length[idx] + b2 * velocity[idx];

            out[idx] = gain * ctrl[idx] + bias;
        }
    }

    #[cube(launch)]
    pub fn joint_equality_eval_kernel(
        q1: &[f32],
        q2: &[f32],
        ref1: &[f32],
        ref2: &[f32],
        coef: &[f32],
        pos_out: &mut [f32],
        jac2_out: &mut [f32],
    ) {
        let idx = ABSOLUTE_POS;
        if idx < q1.len() {
            let d = q2[idx] - ref2[idx];
            let c0 = coef[idx * 5];
            let c1 = coef[idx * 5 + 1];
            let c2 = coef[idx * 5 + 2];
            let c3 = coef[idx * 5 + 3];
            let c4 = coef[idx * 5 + 4];

            let d2 = d * d;
            let d3 = d2 * d;
            let d4 = d3 * d;

            let poly = c0 + c1 * d + c2 * d2 + c3 * d3 + c4 * d4;
            let poly_prime = c1 + 2.0f32 * c2 * d + 3.0f32 * c3 * d2 + 4.0f32 * c4 * d3;

            pos_out[idx] = q1[idx] - ref1[idx] - poly;
            jac2_out[idx] = -poly_prime;
        }
    }

    #[cube(launch)]
    pub fn constraint_impedance_kernel(
        solimp: &[f32],
        pos: &[f32],
        margin: &[f32],
        out_imp: &mut [f32],
        out_imp_prime: &mut [f32],
    ) {
        let idx = ABSOLUTE_POS;
        if idx < pos.len() {
            let dmin = f32::clamp(solimp[idx * 5], 0.0001f32, 0.9999f32);
            let dmax = f32::clamp(solimp[idx * 5 + 1], 0.0001f32, 0.9999f32);
            let width = f32::max(solimp[idx * 5 + 2], 0.0f32);
            let mid = f32::clamp(solimp[idx * 5 + 3], 0.0001f32, 0.9999f32);
            let power = f32::max(solimp[idx * 5 + 4], 1.0f32);

            let minval = 1e-15f32;
            if dmin == dmax || width <= minval {
                out_imp[idx] = 0.5f32 * (dmin + dmax);
                out_imp_prime[idx] = 0.0f32;
            } else {
                let raw = (pos[idx] - margin[idx]) / width;
                let mut x = raw;
                let mut sign = 1.0f32;
                if x < 0.0f32 {
                    x = -x;
                    sign = -1.0f32;
                }
                if x >= 1.0f32 {
                    out_imp[idx] = dmax;
                    out_imp_prime[idx] = 0.0f32;
                } else if x <= 0.0f32 {
                    out_imp[idx] = dmin;
                    out_imp_prime[idx] = 0.0f32;
                } else {
                    let a = 1.0f32 / f32::powf(mid, power - 1.0f32);
                    let b = 1.0f32 / f32::powf(1.0f32 - mid, power - 1.0f32);
                    let y = if power == 1.0f32 {
                        x
                    } else if x <= mid {
                        a * f32::powf(x, power)
                    } else {
                        1.0f32 - b * f32::powf(1.0f32 - x, power)
                    };
                    let y_prime = if power == 1.0f32 {
                        1.0f32
                    } else if x <= mid {
                        power * a * f32::powf(x, power - 1.0f32)
                    } else {
                        power * b * f32::powf(1.0f32 - x, power - 1.0f32)
                    };
                    out_imp[idx] = dmin + y * (dmax - dmin);
                    out_imp_prime[idx] = y_prime * sign * (dmax - dmin) / width;
                }
            }
        }
    }

    #[cube(launch)]
    pub fn rk4_pendulum_kernel(
        theta: &[f32],
        omega: &[f32],
        g_over_l: f32,
        dt: f32,
        out_theta: &mut [f32],
        out_omega: &mut [f32],
    ) {
        let idx = ABSOLUTE_POS;
        if idx < theta.len() {
            let th = theta[idx];
            let w = omega[idx];
            let half = 0.5f32 * dt;

            let a1 = -g_over_l * f32::sin(th);
            let q2 = th + half * w;
            let v2 = w + half * a1;
            let a2 = -g_over_l * f32::sin(q2);
            let q3 = th + half * v2;
            let v3 = w + half * a2;
            let a3 = -g_over_l * f32::sin(q3);
            let q4 = th + dt * v3;
            let v4 = w + dt * a3;
            let a4 = -g_over_l * f32::sin(q4);

            out_theta[idx] = th + dt / 6.0f32 * (w + 2.0f32 * v2 + 2.0f32 * v3 + v4);
            out_omega[idx] = w + dt / 6.0f32 * (a1 + 2.0f32 * a2 + 2.0f32 * a3 + a4);
        }
    }

    #[cube(launch)]
    pub fn plane_sphere_contact_kernel(
        centers: &[f32],
        radii: &[f32],
        plane_height: f32,
        margin: f32,
        valid: &mut [u32],
        dists: &mut [f32],
        pos: &mut [f32],
    ) {
        let idx = ABSOLUTE_POS;
        if idx < radii.len() {
            let cz = centers[idx * 3 + 2];
            let r = radii[idx];
            let cdist = cz - plane_height;
            if cdist <= margin + r {
                let d = cdist - r;
                let offset = -d / 2.0f32 - r;
                valid[idx] = 1u32;
                dists[idx] = d;
                pos[idx * 3] = centers[idx * 3];
                pos[idx * 3 + 1] = centers[idx * 3 + 1];
                pos[idx * 3 + 2] = cz + offset;
            } else {
                valid[idx] = 0u32;
                dists[idx] = 0.0f32;
                pos[idx * 3] = 0.0f32;
                pos[idx * 3 + 1] = 0.0f32;
                pos[idx * 3 + 2] = 0.0f32;
            }
        }
    }

    // --- LAUNCH HELPERS ---

    /// Cube count and cube dimension for `n` elements. Callers have already
    /// run [`check_batch_size`], so the workgroup count fits the cap.
    fn dispatch(n: usize) -> Result<(CubeCount, CubeDim), GpuSimError> {
        let cubes =
            u32::try_from(n.div_ceil(BLOCK_SIZE)).map_err(|_| GpuSimError::InvalidParameter {
                what: "workgroup count exceeds u32 range",
            })?;
        Ok((
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(BLOCK_SIZE as u32),
        ))
    }

    fn get_client() -> Result<cubecl_runtime::client::Client, GpuSimError> {
        let device = Default::default();
        std::panic::catch_unwind(|| WgpuRuntime::<AutoCompiler>::client(&device)).map_err(|e| {
            GpuSimError::LaunchError(format!("Failed to acquire WGPU client device: {e:?}"))
        })
    }

    // --- LAUNCHERS ---
    //
    // Each launcher runs, in order: length checks, the buffer byte-size
    // preflight, `dispatch(n)`, and only then `get_client()` and the uploads.
    // A rejected request never reaches the device.

    pub(crate) fn launch_semi_implicit_euler(
        q: &mut [f32],
        qd: &mut [f32],
        qacc: &[f32],
        dt: f32,
    ) -> Result<(), GpuSimError> {
        let n = q.len();
        check_lengths(n, &[("qd", qd.len()), ("qacc", qacc.len())])?;
        check_batch_size(n, 1)?;
        if n == 0 {
            return Ok(());
        }
        let (cube_count, cube_dim) = dispatch(n)?;

        let client = get_client()?;
        let q_handle = client.create_from_slice(f32::as_bytes(q));
        let qd_handle = client.create_from_slice(f32::as_bytes(qd));
        let qacc_handle = client.create_from_slice(f32::as_bytes(qacc));

        // SAFETY:
        // 1. The three handles are created from slices of exactly `n` f32 elements, checked above.
        // 2. Each BufferArg length is `n`, which equals the element count of its allocation.
        // 3. `CubeDim` is the fixed `BLOCK_SIZE` (256), not `n`. The grid is `ceil(n / 256)` cubes, so
        //    the last cube can have threads with `ABSOLUTE_POS >= n`. Each thread checks `idx < q.len()`
        //    (where `q.len() == n`) before any access, so those threads do nothing.
        unsafe {
            semi_implicit_euler_kernel::launch(
                &client,
                cube_count,
                cube_dim,
                BufferArg::from_raw_parts(q_handle.clone(), n),
                BufferArg::from_raw_parts(qd_handle.clone(), n),
                BufferArg::from_raw_parts(qacc_handle, n),
                dt,
            );
        }

        let q_bytes = client.read_one_unchecked(q_handle);
        let qd_bytes = client.read_one_unchecked(qd_handle);
        q.copy_from_slice(f32::from_bytes(&q_bytes));
        qd.copy_from_slice(f32::from_bytes(&qd_bytes));
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_implicit_damping_euler(
        nv: usize,
        q: &mut [f32],
        qd: &mut [f32],
        mass: &[f32],
        damping: &[f32],
        bias: &[f32],
        dt: f32,
        status: &mut [u32],
    ) -> Result<(), GpuSimError> {
        let n = status.len();
        let (vec_len, mat_len) = check_implicit_batch(nv, n)?;
        check_lengths(
            vec_len,
            &[
                ("q", q.len()),
                ("qd", qd.len()),
                ("damping", damping.len()),
                ("bias", bias.len()),
            ],
        )?;
        check_lengths(mat_len, &[("mass", mass.len())])?;
        if n == 0 {
            return Ok(());
        }
        let (cube_count, cube_dim) = dispatch(n)?;
        let nv_u32 = u32::try_from(nv).map_err(|_| GpuSimError::InvalidParameter {
            what: "nv exceeds u32 range",
        })?;

        let client = get_client()?;
        let q_handle = client.create_from_slice(f32::as_bytes(q));
        let qd_handle = client.create_from_slice(f32::as_bytes(qd));
        let mass_handle = client.create_from_slice(f32::as_bytes(mass));
        let damping_handle = client.create_from_slice(f32::as_bytes(damping));
        let bias_handle = client.create_from_slice(f32::as_bytes(bias));
        let status_handle = client.empty(core::mem::size_of_val(status));

        // SAFETY:
        // 1. `check_implicit_batch` guarantees `1 <= nv <= 6`, so the kernel's 36-element matrix and
        //    6-element vector local arrays hold every `i * nv + j` and `i` it indexes.
        // 2. `q`, `qd`, `damping`, `bias` hold `n * nv` elements and `mass` holds `n * nv * nv`, all
        //    checked above; `status` holds `n`. BufferArg lengths use those checked counts.
        // 3. The kernel body runs only inside `if idx < status.len()` (`== n`), so it reads
        //    `idx * nv + i` (< `n * nv`) and `idx * nv * nv + i * nv + j` (< `n * nv * nv`) and writes
        //    `q`, `qd` at `idx * nv + i` and `status[idx]`; all are in range. `CubeDim` is the fixed 256.
        unsafe {
            implicit_damping_euler_kernel::launch(
                &client,
                cube_count,
                cube_dim,
                BufferArg::from_raw_parts(q_handle.clone(), vec_len),
                BufferArg::from_raw_parts(qd_handle.clone(), vec_len),
                BufferArg::from_raw_parts(mass_handle, mat_len),
                BufferArg::from_raw_parts(damping_handle, vec_len),
                BufferArg::from_raw_parts(bias_handle, vec_len),
                BufferArg::from_raw_parts(status_handle.clone(), n),
                nv_u32,
                dt,
            );
        }

        let q_bytes = client.read_one_unchecked(q_handle);
        let qd_bytes = client.read_one_unchecked(qd_handle);
        let status_bytes = client.read_one_unchecked(status_handle);
        q.copy_from_slice(f32::from_bytes(&q_bytes));
        qd.copy_from_slice(f32::from_bytes(&qd_bytes));
        status.copy_from_slice(u32::from_bytes(&status_bytes));
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_equality_solve(
        nv: usize,
        nrow: usize,
        mass: &[f32],
        qvel: &[f32],
        smooth: &[f32],
        jac: &[f32],
        pos: &[f32],
        solref: &[f32],
        solimp: &[f32],
        dt: f32,
        force: &mut [f32],
        qacc: &mut [f32],
        clamped: &mut [u32],
        status: &mut [u32],
    ) -> Result<(), GpuSimError> {
        let n = status.len();
        let (vec_len, mass_len, row_len, jac_len, _) = check_equality_batch(nv, nrow, n)?;
        check_lengths(
            vec_len,
            &[
                ("qvel", qvel.len()),
                ("smooth", smooth.len()),
                ("qacc", qacc.len()),
            ],
        )?;
        check_lengths(mass_len, &[("mass", mass.len())])?;
        check_lengths(jac_len, &[("jac", jac.len())])?;
        check_lengths(
            row_len,
            &[
                ("pos", pos.len()),
                ("force", force.len()),
                ("clamped", clamped.len()),
            ],
        )?;
        check_lengths(row_len * 2, &[("solref", solref.len())])?;
        check_lengths(row_len * 5, &[("solimp", solimp.len())])?;
        if n == 0 {
            return Ok(());
        }
        let (cube_count, cube_dim) = dispatch(n)?;
        let strides = equality_strides(nv, nrow);
        let out_stride = strides.output;
        let flag_stride = strides.flags;
        let mut input = Vec::with_capacity(n * strides.input);
        let mut output = Vec::with_capacity(n * out_stride);
        let mut flags = Vec::with_capacity(n * flag_stride);
        for e in 0..n {
            input.extend_from_slice(&mass[e * nv * nv..(e + 1) * nv * nv]);
            input.extend_from_slice(&qvel[e * nv..(e + 1) * nv]);
            input.extend_from_slice(&smooth[e * nv..(e + 1) * nv]);
            input.extend_from_slice(&jac[e * nrow * nv..(e + 1) * nrow * nv]);
            input.extend_from_slice(&pos[e * nrow..(e + 1) * nrow]);
            input.extend_from_slice(&solref[e * nrow * 2..(e + 1) * nrow * 2]);
            input.extend_from_slice(&solimp[e * nrow * 5..(e + 1) * nrow * 5]);
            output.extend_from_slice(&force[e * nrow..(e + 1) * nrow]);
            output.extend_from_slice(&qacc[e * nv..(e + 1) * nv]);
            flags.extend_from_slice(&clamped[e * nrow..(e + 1) * nrow]);
            flags.push(status[e]);
        }
        let client = get_client()?;
        let input_h = client.create_from_slice(f32::as_bytes(&input));
        let output_h = client.create_from_slice(f32::as_bytes(&output));
        let flags_h = client.create_from_slice(u32::as_bytes(&flags));

        // SAFETY: preflight before packing limits all 3 buffers to 128 MiB and
        // the dispatch to 65535 workgroups. 1<=nv,nrow<=6 bounds local arrays.
        // idx < flags.len()/(nrow+1) guards each per-env offset; packed
        // strides equal the checked lengths, including both row parameter arrays.
        unsafe {
            equality_solve_kernel::launch(
                &client,
                cube_count,
                cube_dim,
                BufferArg::from_raw_parts(input_h, input.len()),
                BufferArg::from_raw_parts(output_h.clone(), output.len()),
                BufferArg::from_raw_parts(flags_h.clone(), flags.len()),
                nv as u32,
                nrow as u32,
                dt,
            );
        }
        let result = client.read_one_unchecked(output_h);
        let result = f32::from_bytes(&result);
        let bits = client.read_one_unchecked(flags_h);
        let bits = u32::from_bytes(&bits);
        for e in 0..n {
            force[e * nrow..(e + 1) * nrow]
                .copy_from_slice(&result[e * out_stride..e * out_stride + nrow]);
            qacc[e * nv..(e + 1) * nv]
                .copy_from_slice(&result[e * out_stride + nrow..(e + 1) * out_stride]);
            clamped[e * nrow..(e + 1) * nrow]
                .copy_from_slice(&bits[e * flag_stride..e * flag_stride + nrow]);
            status[e] = bits[e * flag_stride + nrow];
        }
        Ok(())
    }

    pub(crate) fn launch_joint_spring(
        q: &[f32],
        q_ref: &[f32],
        stiffness: &[f32],
        out: &mut [f32],
    ) -> Result<(), GpuSimError> {
        let n = q.len();
        check_lengths(
            n,
            &[
                ("q_ref", q_ref.len()),
                ("stiffness", stiffness.len()),
                ("out", out.len()),
            ],
        )?;
        check_batch_size(n, 1)?;
        if n == 0 {
            return Ok(());
        }
        let (cube_count, cube_dim) = dispatch(n)?;

        let client = get_client()?;
        let q_handle = client.create_from_slice(f32::as_bytes(q));
        let q_ref_handle = client.create_from_slice(f32::as_bytes(q_ref));
        let stiffness_handle = client.create_from_slice(f32::as_bytes(stiffness));
        let out_handle = client.empty(core::mem::size_of_val(out));

        // SAFETY:
        // 1. All four buffers hold exactly `n` f32 elements, checked above.
        // 2. Each BufferArg length is `n`, matching its allocation.
        // 3. The kernel reads and writes only inside `if idx < q.len()`, where `q.len() == n`.
        unsafe {
            joint_spring_kernel::launch(
                &client,
                cube_count,
                cube_dim,
                BufferArg::from_raw_parts(q_handle, n),
                BufferArg::from_raw_parts(q_ref_handle, n),
                BufferArg::from_raw_parts(stiffness_handle, n),
                BufferArg::from_raw_parts(out_handle.clone(), n),
            );
        }

        let out_bytes = client.read_one_unchecked(out_handle);
        out.copy_from_slice(f32::from_bytes(&out_bytes));
        Ok(())
    }

    pub(crate) fn launch_joint_damper(
        qd: &[f32],
        damping: &[f32],
        out: &mut [f32],
    ) -> Result<(), GpuSimError> {
        let n = qd.len();
        check_lengths(n, &[("damping", damping.len()), ("out", out.len())])?;
        check_batch_size(n, 1)?;
        if n == 0 {
            return Ok(());
        }
        let (cube_count, cube_dim) = dispatch(n)?;

        let client = get_client()?;
        let qd_handle = client.create_from_slice(f32::as_bytes(qd));
        let damping_handle = client.create_from_slice(f32::as_bytes(damping));
        let out_handle = client.empty(core::mem::size_of_val(out));

        // SAFETY:
        // 1. All three buffers hold exactly `n` f32 elements, checked above.
        // 2. Each BufferArg length is `n`, matching its allocation.
        // 3. The kernel reads and writes only inside `if idx < qd.len()`, where `qd.len() == n`.
        unsafe {
            joint_damper_kernel::launch(
                &client,
                cube_count,
                cube_dim,
                BufferArg::from_raw_parts(qd_handle, n),
                BufferArg::from_raw_parts(damping_handle, n),
                BufferArg::from_raw_parts(out_handle.clone(), n),
            );
        }

        let out_bytes = client.read_one_unchecked(out_handle);
        out.copy_from_slice(f32::from_bytes(&out_bytes));
        Ok(())
    }

    pub(crate) fn launch_affine_actuator(
        gain_prm_flat: &[f32],
        bias_prm_flat: &[f32],
        ctrl: &[f32],
        length: &[f32],
        velocity: &[f32],
        out: &mut [f32],
    ) -> Result<(), GpuSimError> {
        let n = ctrl.len();
        check_lengths(
            n,
            &[
                ("length", length.len()),
                ("velocity", velocity.len()),
                ("out", out.len()),
            ],
        )?;
        let wide = check_batch_size(n, 3)?;
        check_lengths(
            wide,
            &[
                ("gain_prm_flat", gain_prm_flat.len()),
                ("bias_prm_flat", bias_prm_flat.len()),
            ],
        )?;
        if n == 0 {
            return Ok(());
        }
        let (cube_count, cube_dim) = dispatch(n)?;

        let client = get_client()?;
        let gain_handle = client.create_from_slice(f32::as_bytes(gain_prm_flat));
        let bias_handle = client.create_from_slice(f32::as_bytes(bias_prm_flat));
        let ctrl_handle = client.create_from_slice(f32::as_bytes(ctrl));
        let length_handle = client.create_from_slice(f32::as_bytes(length));
        let vel_handle = client.create_from_slice(f32::as_bytes(velocity));
        let out_handle = client.empty(core::mem::size_of_val(out));

        // SAFETY:
        // 1. `gain_prm_flat` and `bias_prm_flat` hold `3 * n` elements, and the other buffers hold `n`, all checked above.
        // 2. BufferArg lengths match those allocation sizes.
        // 3. The kernel reads `idx * 3 + {0,1,2}` and writes `out[idx]` only inside `if idx < ctrl.len()`, where
        //    `ctrl.len() == n`, so every access is in range.
        unsafe {
            affine_actuator_kernel::launch(
                &client,
                cube_count,
                cube_dim,
                BufferArg::from_raw_parts(gain_handle, wide),
                BufferArg::from_raw_parts(bias_handle, wide),
                BufferArg::from_raw_parts(ctrl_handle, n),
                BufferArg::from_raw_parts(length_handle, n),
                BufferArg::from_raw_parts(vel_handle, n),
                BufferArg::from_raw_parts(out_handle.clone(), n),
            );
        }

        let out_bytes = client.read_one_unchecked(out_handle);
        out.copy_from_slice(f32::from_bytes(&out_bytes));
        Ok(())
    }

    pub(crate) fn launch_joint_equality_eval(
        q1: &[f32],
        q2: &[f32],
        ref1: &[f32],
        ref2: &[f32],
        coef_flat: &[f32],
        pos_out: &mut [f32],
        jac2_out: &mut [f32],
    ) -> Result<(), GpuSimError> {
        let n = q1.len();
        check_lengths(
            n,
            &[
                ("q2", q2.len()),
                ("ref1", ref1.len()),
                ("ref2", ref2.len()),
                ("pos_out", pos_out.len()),
                ("jac2_out", jac2_out.len()),
            ],
        )?;
        let wide = check_batch_size(n, 5)?;
        check_lengths(wide, &[("coef_flat", coef_flat.len())])?;
        if n == 0 {
            return Ok(());
        }
        let (cube_count, cube_dim) = dispatch(n)?;

        let client = get_client()?;
        let q1_handle = client.create_from_slice(f32::as_bytes(q1));
        let q2_handle = client.create_from_slice(f32::as_bytes(q2));
        let ref1_handle = client.create_from_slice(f32::as_bytes(ref1));
        let ref2_handle = client.create_from_slice(f32::as_bytes(ref2));
        let coef_handle = client.create_from_slice(f32::as_bytes(coef_flat));
        let pos_handle = client.empty(core::mem::size_of_val(pos_out));
        let jac2_handle = client.empty(core::mem::size_of_val(jac2_out));

        // SAFETY:
        // 1. `coef_flat` holds `5 * n` elements and the other buffers hold `n`, all checked above.
        // 2. BufferArg lengths match those allocation sizes.
        // 3. The kernel reads `idx * 5 + {0..4}` and writes `pos_out[idx]`, `jac2_out[idx]` only inside
        //    `if idx < q1.len()`, where `q1.len() == n`, so every access is in range.
        unsafe {
            joint_equality_eval_kernel::launch(
                &client,
                cube_count,
                cube_dim,
                BufferArg::from_raw_parts(q1_handle, n),
                BufferArg::from_raw_parts(q2_handle, n),
                BufferArg::from_raw_parts(ref1_handle, n),
                BufferArg::from_raw_parts(ref2_handle, n),
                BufferArg::from_raw_parts(coef_handle, wide),
                BufferArg::from_raw_parts(pos_handle.clone(), n),
                BufferArg::from_raw_parts(jac2_handle.clone(), n),
            );
        }

        let pos_bytes = client.read_one_unchecked(pos_handle);
        let jac2_bytes = client.read_one_unchecked(jac2_handle);
        pos_out.copy_from_slice(f32::from_bytes(&pos_bytes));
        jac2_out.copy_from_slice(f32::from_bytes(&jac2_bytes));
        Ok(())
    }

    pub(crate) fn launch_constraint_impedance(
        solimp_flat: &[f32],
        pos: &[f32],
        margin: &[f32],
        out_imp: &mut [f32],
        out_imp_prime: &mut [f32],
    ) -> Result<(), GpuSimError> {
        let n = pos.len();
        check_lengths(
            n,
            &[
                ("margin", margin.len()),
                ("out_imp", out_imp.len()),
                ("out_imp_prime", out_imp_prime.len()),
            ],
        )?;
        let wide = check_batch_size(n, 5)?;
        check_lengths(wide, &[("solimp_flat", solimp_flat.len())])?;
        if n == 0 {
            return Ok(());
        }
        let (cube_count, cube_dim) = dispatch(n)?;

        let client = get_client()?;
        let solimp_handle = client.create_from_slice(f32::as_bytes(solimp_flat));
        let pos_handle = client.create_from_slice(f32::as_bytes(pos));
        let margin_handle = client.create_from_slice(f32::as_bytes(margin));
        let out_handle = client.empty(core::mem::size_of_val(out_imp));
        let prime_handle = client.empty(core::mem::size_of_val(out_imp_prime));

        // SAFETY:
        // 1. `solimp_flat` holds `5 * n` elements and the other buffers hold `n`, all checked above.
        // 2. BufferArg lengths match those allocation sizes.
        // 3. The kernel reads `idx * 5 + {0..4}` and writes `out_imp[idx]` and `out_imp_prime[idx]` only inside
        //    `if idx < pos.len()`, where `pos.len() == n`, so every access is in range.
        unsafe {
            constraint_impedance_kernel::launch(
                &client,
                cube_count,
                cube_dim,
                BufferArg::from_raw_parts(solimp_handle, wide),
                BufferArg::from_raw_parts(pos_handle, n),
                BufferArg::from_raw_parts(margin_handle, n),
                BufferArg::from_raw_parts(out_handle.clone(), n),
                BufferArg::from_raw_parts(prime_handle.clone(), n),
            );
        }

        let out_bytes = client.read_one_unchecked(out_handle);
        let prime_bytes = client.read_one_unchecked(prime_handle);
        out_imp.copy_from_slice(f32::from_bytes(&out_bytes));
        out_imp_prime.copy_from_slice(f32::from_bytes(&prime_bytes));
        Ok(())
    }

    pub(crate) fn launch_rk4_pendulum(
        theta: &mut [f32],
        omega: &mut [f32],
        g_over_l: f32,
        dt: f32,
    ) -> Result<(), GpuSimError> {
        let n = theta.len();
        check_lengths(n, &[("omega", omega.len())])?;
        check_batch_size(n, 1)?;
        if n == 0 {
            return Ok(());
        }
        let (cube_count, cube_dim) = dispatch(n)?;

        let client = get_client()?;
        let theta_handle = client.create_from_slice(f32::as_bytes(theta));
        let omega_handle = client.create_from_slice(f32::as_bytes(omega));
        let out_theta_handle = client.empty(core::mem::size_of_val(theta));
        let out_omega_handle = client.empty(core::mem::size_of_val(omega));

        // SAFETY:
        // 1. `theta`, `omega` and both outputs hold `n` elements, checked above.
        // 2. BufferArg lengths match those allocation sizes.
        // 3. The kernel reads and writes index `idx` only inside `if idx < theta.len()`, so every access is in range.
        unsafe {
            rk4_pendulum_kernel::launch(
                &client,
                cube_count,
                cube_dim,
                BufferArg::from_raw_parts(theta_handle, n),
                BufferArg::from_raw_parts(omega_handle, n),
                g_over_l,
                dt,
                BufferArg::from_raw_parts(out_theta_handle.clone(), n),
                BufferArg::from_raw_parts(out_omega_handle.clone(), n),
            );
        }

        let theta_bytes = client.read_one_unchecked(out_theta_handle);
        let omega_bytes = client.read_one_unchecked(out_omega_handle);
        theta.copy_from_slice(f32::from_bytes(&theta_bytes));
        omega.copy_from_slice(f32::from_bytes(&omega_bytes));
        Ok(())
    }

    pub(crate) fn launch_plane_sphere_contact(
        centers_flat: &[f32],
        radii: &[f32],
        plane_height: f32,
        margin: f32,
        valid: &mut [u32],
        dists: &mut [f32],
        pos_flat: &mut [f32],
    ) -> Result<(), GpuSimError> {
        let n = radii.len();
        check_lengths(n, &[("valid", valid.len()), ("dists", dists.len())])?;
        let wide = check_batch_size(n, 3)?;
        check_lengths(
            wide,
            &[
                ("centers_flat", centers_flat.len()),
                ("pos_flat", pos_flat.len()),
            ],
        )?;
        if n == 0 {
            return Ok(());
        }
        let (cube_count, cube_dim) = dispatch(n)?;

        let client = get_client()?;
        let centers_handle = client.create_from_slice(f32::as_bytes(centers_flat));
        let radii_handle = client.create_from_slice(f32::as_bytes(radii));
        let valid_handle = client.empty(core::mem::size_of_val(valid));
        let dists_handle = client.empty(core::mem::size_of_val(dists));
        let pos_handle = client.empty(core::mem::size_of_val(pos_flat));

        // SAFETY:
        // 1. `centers_flat` and `pos_flat` hold `3 * n` elements, and `valid`, `dists`, `radii` hold `n`, all checked above.
        // 2. BufferArg lengths match those allocation sizes.
        // 3. The kernel reads `idx * 3 + {0,1,2}` and writes `valid[idx]`, `dists[idx]`, `pos[idx * 3 + {0,1,2}]` only
        //    inside `if idx < radii.len()`, where `radii.len() == n`, so every access is in range.
        unsafe {
            plane_sphere_contact_kernel::launch(
                &client,
                cube_count,
                cube_dim,
                BufferArg::from_raw_parts(centers_handle, wide),
                BufferArg::from_raw_parts(radii_handle, n),
                plane_height,
                margin,
                BufferArg::from_raw_parts(valid_handle.clone(), n),
                BufferArg::from_raw_parts(dists_handle.clone(), n),
                BufferArg::from_raw_parts(pos_handle.clone(), wide),
            );
        }

        let valid_bytes = client.read_one_unchecked(valid_handle);
        let dists_bytes = client.read_one_unchecked(dists_handle);
        let pos_bytes = client.read_one_unchecked(pos_handle);

        valid.copy_from_slice(u32::from_bytes(&valid_bytes));
        dists.copy_from_slice(f32::from_bytes(&dists_bytes));
        pos_flat.copy_from_slice(f32::from_bytes(&pos_bytes));
        Ok(())
    }
}

#[cfg(not(feature = "cubecl"))]
pub(crate) mod backend {
    use super::*;

    pub(crate) fn launch_semi_implicit_euler(
        _q: &mut [f32],
        _qd: &mut [f32],
        _qacc: &[f32],
        _dt: f32,
    ) -> Result<(), GpuSimError> {
        Err(GpuSimError::GpuDisabled)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_implicit_damping_euler(
        _nv: usize,
        _q: &mut [f32],
        _qd: &mut [f32],
        _mass: &[f32],
        _damping: &[f32],
        _bias: &[f32],
        _dt: f32,
        _status: &mut [u32],
    ) -> Result<(), GpuSimError> {
        Err(GpuSimError::GpuDisabled)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_equality_solve(
        _nv: usize,
        _nrow: usize,
        _mass: &[f32],
        _qvel: &[f32],
        _smooth: &[f32],
        _jac: &[f32],
        _pos: &[f32],
        _solref: &[f32],
        _solimp: &[f32],
        _dt: f32,
        _force: &mut [f32],
        _qacc: &mut [f32],
        _clamped: &mut [u32],
        _status: &mut [u32],
    ) -> Result<(), GpuSimError> {
        Err(GpuSimError::GpuDisabled)
    }

    pub(crate) fn launch_joint_spring(
        _q: &[f32],
        _q_ref: &[f32],
        _stiffness: &[f32],
        _out: &mut [f32],
    ) -> Result<(), GpuSimError> {
        Err(GpuSimError::GpuDisabled)
    }

    pub(crate) fn launch_joint_damper(
        _qd: &[f32],
        _damping: &[f32],
        _out: &mut [f32],
    ) -> Result<(), GpuSimError> {
        Err(GpuSimError::GpuDisabled)
    }

    pub(crate) fn launch_affine_actuator(
        _gain_prm_flat: &[f32],
        _bias_prm_flat: &[f32],
        _ctrl: &[f32],
        _length: &[f32],
        _velocity: &[f32],
        _out: &mut [f32],
    ) -> Result<(), GpuSimError> {
        Err(GpuSimError::GpuDisabled)
    }

    pub(crate) fn launch_joint_equality_eval(
        _q1: &[f32],
        _q2: &[f32],
        _ref1: &[f32],
        _ref2: &[f32],
        _coef_flat: &[f32],
        _pos_out: &mut [f32],
        _jac2_out: &mut [f32],
    ) -> Result<(), GpuSimError> {
        Err(GpuSimError::GpuDisabled)
    }

    pub(crate) fn launch_constraint_impedance(
        _solimp_flat: &[f32],
        _pos: &[f32],
        _margin: &[f32],
        _out_imp: &mut [f32],
        _out_imp_prime: &mut [f32],
    ) -> Result<(), GpuSimError> {
        Err(GpuSimError::GpuDisabled)
    }

    pub(crate) fn launch_rk4_pendulum(
        _theta: &mut [f32],
        _omega: &mut [f32],
        _g_over_l: f32,
        _dt: f32,
    ) -> Result<(), GpuSimError> {
        Err(GpuSimError::GpuDisabled)
    }

    pub(crate) fn launch_plane_sphere_contact(
        _centers_flat: &[f32],
        _radii: &[f32],
        _plane_height: f32,
        _margin: f32,
        _valid: &mut [u32],
        _dists: &mut [f32],
        _pos_flat: &mut [f32],
    ) -> Result<(), GpuSimError> {
        Err(GpuSimError::GpuDisabled)
    }
}

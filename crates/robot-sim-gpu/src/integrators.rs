//! GPU time integrators mirroring `robot_sim::integrators`.
//!
//! Ported: `semi_implicit_euler_step` (elementwise) and
//! `implicit_damping_euler_step` (one thread per environment, dense Cholesky
//! for at most [`crate::MAX_DOF`] degrees of freedom), and `rk4_pendulum_step`, a
//! named RK4 step for the pendulum acceleration `-(g/L) sin(theta)`. The general
//! `rk4_step` is not ported: it takes an acceleration closure, which has no kernel
//! counterpart, and there is no CPU fallback for it.

use crate::error::GpuSimError;
use crate::launch::{backend, check_batch_size, check_implicit_batch};

/// GPU semi-implicit (symplectic) Euler step for scalar joints.
///
/// Mirrors [`robot_sim::integrators::semi_implicit_euler_step`].
///
/// # Errors
///
/// Returns [`GpuSimError::DimensionMismatch`] if slice lengths differ,
/// [`GpuSimError::NonFinite`] if any float is NaN/Inf, or
/// [`GpuSimError::InvalidParameter`] if `dt < 0.0` or non-finite.
pub fn gpu_semi_implicit_euler_step(
    q: &mut [f32],
    qd: &mut [f32],
    qacc: &[f32],
    dt: f32,
) -> Result<(), GpuSimError> {
    let n = q.len();
    if qd.len() != n {
        return Err(GpuSimError::DimensionMismatch {
            what: "qd",
            expected: n,
            found: qd.len(),
        });
    }
    if qacc.len() != n {
        return Err(GpuSimError::DimensionMismatch {
            what: "qacc",
            expected: n,
            found: qacc.len(),
        });
    }
    check_batch_size(n, 1)?;
    if !dt.is_finite() || dt < 0.0 {
        return Err(GpuSimError::InvalidParameter {
            what: "dt must be finite and non-negative",
        });
    }
    if q.iter().any(|x| !x.is_finite()) {
        return Err(GpuSimError::NonFinite { what: "q" });
    }
    if qd.iter().any(|x| !x.is_finite()) {
        return Err(GpuSimError::NonFinite { what: "qd" });
    }
    if qacc.iter().any(|x| !x.is_finite()) {
        return Err(GpuSimError::NonFinite { what: "qacc" });
    }

    backend::launch_semi_implicit_euler(q, qd, qacc, dt)
}

/// GPU classical fourth-order Runge-Kutta step for the pendulum `theta'' = -(g/L) sin(theta)`.
///
/// Mirrors [`robot_sim::integrators::rk4_step`] with the acceleration closure
/// fixed to `-(g_over_l) * sin(theta)`, using the same tableau (`0.5, 0.5, 1` and
/// `1/6, 1/3, 1/3, 1/6`). One environment per element; `theta` and `omega` are
/// updated in place.
///
/// # Errors
///
/// Returns [`GpuSimError::DimensionMismatch`] if the slice lengths differ,
/// [`GpuSimError::NonFinite`] for a NaN or infinite input, or
/// [`GpuSimError::InvalidParameter`] if `dt` is negative or non-finite or
/// `g_over_l` is non-finite.
pub fn gpu_rk4_pendulum_step(
    theta: &mut [f32],
    omega: &mut [f32],
    g_over_l: f32,
    dt: f32,
) -> Result<(), GpuSimError> {
    let n = theta.len();
    if omega.len() != n {
        return Err(GpuSimError::DimensionMismatch {
            what: "omega",
            expected: n,
            found: omega.len(),
        });
    }
    check_batch_size(n, 1)?;
    if !dt.is_finite() || dt < 0.0 {
        return Err(GpuSimError::InvalidParameter {
            what: "dt must be finite and non-negative",
        });
    }
    if !g_over_l.is_finite() {
        return Err(GpuSimError::InvalidParameter {
            what: "g_over_l must be finite",
        });
    }
    if theta.iter().any(|x| !x.is_finite()) {
        return Err(GpuSimError::NonFinite { what: "theta" });
    }
    if omega.iter().any(|x| !x.is_finite()) {
        return Err(GpuSimError::NonFinite { what: "omega" });
    }

    backend::launch_rk4_pendulum(theta, omega, g_over_l, dt)
}

/// Relative symmetry tolerance for `f32` mass matrices. The CPU port uses
/// `1e-12` in `f64`; `1e-6` is about eight `f32` epsilons.
const SYMMETRY_TOLERANCE_F32: f32 = 1e-6;

/// GPU implicit-in-velocity Euler step with diagonal damping, for a batch of
/// `n_env = status.len()` independent systems of `nv` degrees of freedom each.
///
/// Mirrors [`robot_sim::integrators::implicit_damping_euler_step`], whose
/// source is MuJoCo 3.17.0 (commit `800a109a`) `engine_forward.c`,
/// `mj_EulerSkip` damping branch: solve `(M + dt*diag(B)) qacc = bias`, then
/// `qd += dt*qacc`, `q += dt*qd`. One GPU thread handles one environment and
/// runs the CPU Cholesky loop order on a per-thread `6 x 6` local array, so
/// `nv` is limited to [`crate::MAX_DOF`]; larger systems are rejected.
///
/// Layout, row-major per environment: `q`, `qd`, `damping`, `bias` hold
/// `n_env * nv` values (`env * nv + i`); `mass` holds `n_env * nv * nv`
/// (`env * nv * nv + i * nv + j`). `bias` is `qfrc_smooth + qfrc_constraint`
/// and already contains the explicit damper force, as on the CPU.
///
/// `status[env]` is `1` if the environment was stepped, `0` if its
/// `M + dt*diag(B)` was not symmetric positive definite in `f32`; that
/// environment's `q` and `qd` are then left unchanged. This is the per-env
/// counterpart of the CPU `None`. MuJoCo clamps the pivot instead (`mj_factorI`).
///
/// The CPU function checks symmetry at `1e-12` in `f64`; here the batch is
/// rejected if any `mass` is asymmetric by more than `1e-6 * max|M|` (that
/// environment's `f32` values).
///
/// # Errors
///
/// Returns [`GpuSimError::InvalidParameter`] if `nv` is 0 or above 6, `dt` is
/// negative or non-finite, a `damping` entry is negative, a `mass` is not
/// symmetric, or the batch exceeds the launch limits;
/// [`GpuSimError::DimensionMismatch`] if a slice length does not match;
/// [`GpuSimError::NonFinite`] for a NaN or infinite input. On any error no
/// output is modified.
#[allow(clippy::too_many_arguments)]
pub fn gpu_implicit_damping_euler_step(
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
    for (what, found, expected) in [
        ("q", q.len(), vec_len),
        ("qd", qd.len(), vec_len),
        ("damping", damping.len(), vec_len),
        ("bias", bias.len(), vec_len),
        ("mass", mass.len(), mat_len),
    ] {
        if found != expected {
            return Err(GpuSimError::DimensionMismatch {
                what,
                expected,
                found,
            });
        }
    }
    if !dt.is_finite() || dt < 0.0 {
        return Err(GpuSimError::InvalidParameter {
            what: "dt must be finite and non-negative",
        });
    }
    for (what, values) in [
        ("q", &*q),
        ("qd", &*qd),
        ("mass", mass),
        ("damping", damping),
        ("bias", bias),
    ] {
        if values.iter().any(|x| !x.is_finite()) {
            return Err(GpuSimError::NonFinite { what });
        }
    }
    if damping.iter().any(|b| *b < 0.0) {
        return Err(GpuSimError::InvalidParameter {
            what: "damping entries must be non-negative",
        });
    }
    for m in mass.chunks_exact(nv * nv) {
        let scale = m.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        for i in 0..nv {
            for j in (i + 1)..nv {
                if (m[i * nv + j] - m[j * nv + i]).abs() > SYMMETRY_TOLERANCE_F32 * scale {
                    return Err(GpuSimError::InvalidParameter {
                        what: "mass must be symmetric",
                    });
                }
            }
        }
    }
    if n == 0 {
        return Ok(());
    }

    backend::launch_implicit_damping_euler(nv, q, qd, mass, damping, bias, dt, status)
}

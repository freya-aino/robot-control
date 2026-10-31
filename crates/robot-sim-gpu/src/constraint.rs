//! GPU equality constraints and soft constraint impedance mirroring the reference simulator 3.17.0.
//!
//! Mirrors [`robot_sim::constraint`].

use crate::error::GpuSimError;
use crate::launch::{backend, check_batch_size, check_equality_batch, check_lengths};

/// Batched, equality-only dense solve mirroring
/// [`robot_sim::constraint::solve_equality`] (the reference formulation,
/// `engine_core_constraint.c`: `getsolparam`, `getimpedance`,
/// `makeImpedance`, `referenceConstraint`, `makeARNumeric`).
/// The `DIAGEXACT` diagonal, not the reference's default approximate diagonal, is
/// used; there is no contact/limit/friction solve or CPU fallback.
///
/// `n_env = status.len()`, `1 <= nv,nrow <= 6`. Per environment, `mass` is
/// row-major `nv*nv`, `qvel`/`smooth`/`qacc` are `nv`, `jac` is row-major
/// `nrow*nv`, and `pos`/`margin`/`force`/`clamped` are `nrow`; `solref` and
/// `solimp` are flattened `nrow*2` and `nrow*5`. Equality margins must be
/// zero, as in the CPU port. `status[e] = 0` on a non-SPD mass or regularized
/// constraint matrix, leaving that environment's outputs unchanged. On a
/// successful environment, `clamped` marks rows whose time constant was
/// raised to `2*dt`. Host validation errors leave every output untouched.
///
/// # Errors
///
/// Returns typed [`GpuSimError`] for invalid dimensions, inputs or device
/// limits. A mixed direct/standard `solref` is rejected as in the CPU port.
#[allow(clippy::too_many_arguments)]
pub fn gpu_solve_equality(
    nv: usize,
    nrow: usize,
    mass: &[f32],
    qvel: &[f32],
    smooth: &[f32],
    jac: &[f32],
    pos: &[f32],
    margin: &[f32],
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
            ("margin", margin.len()),
            ("force", force.len()),
            ("clamped", clamped.len()),
        ],
    )?;
    check_lengths(row_len * 2, &[("solref", solref.len())])?;
    check_lengths(row_len * 5, &[("solimp", solimp.len())])?;
    if !dt.is_finite() || dt <= 0.0 {
        return Err(GpuSimError::InvalidParameter {
            what: "dt must be finite and positive",
        });
    }
    for (what, values) in [
        ("mass", mass),
        ("qvel", qvel),
        ("smooth", smooth),
        ("jac", jac),
        ("pos", pos),
        ("margin", margin),
        ("solref", solref),
        ("solimp", solimp),
    ] {
        if values.iter().any(|x| !x.is_finite()) {
            return Err(GpuSimError::NonFinite { what });
        }
    }
    if margin.iter().any(|x| *x != 0.0) {
        return Err(GpuSimError::InvalidParameter {
            what: "equality margin must be zero",
        });
    }
    if solref.chunks_exact(2).any(|r| (r[0] > 0.0) != (r[1] > 0.0)) {
        return Err(GpuSimError::InvalidParameter {
            what: "mixed solref format",
        });
    }
    for m in mass.chunks_exact(nv * nv) {
        let scale = m.iter().fold(1.0f32, |acc, v| acc.max(v.abs()));
        for i in 0..nv {
            for j in (i + 1)..nv {
                if (m[i * nv + j] - m[j * nv + i]).abs() > 1e-6 * scale {
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
    backend::launch_equality_solve(
        nv, nrow, mass, qvel, smooth, jac, pos, solref, solimp, dt, force, qacc, clamped, status,
    )
}

/// Evaluates joint equality residuals and Jacobians for coupled scalar joints on GPU.
///
/// Mirrors [`robot_sim::constraint::joint_equality_row`].
pub fn gpu_joint_equality_eval(
    q1: &[f32],
    q2: &[f32],
    ref1: &[f32],
    ref2: &[f32],
    coef: &[[f32; 5]],
    pos_out: &mut [f32],
    jac2_out: &mut [f32],
) -> Result<(), GpuSimError> {
    let n = q1.len();
    if q2.len() != n {
        return Err(GpuSimError::DimensionMismatch {
            what: "q2",
            expected: n,
            found: q2.len(),
        });
    }
    if ref1.len() != n {
        return Err(GpuSimError::DimensionMismatch {
            what: "ref1",
            expected: n,
            found: ref1.len(),
        });
    }
    if ref2.len() != n {
        return Err(GpuSimError::DimensionMismatch {
            what: "ref2",
            expected: n,
            found: ref2.len(),
        });
    }
    if coef.len() != n {
        return Err(GpuSimError::DimensionMismatch {
            what: "coef",
            expected: n,
            found: coef.len(),
        });
    }
    if pos_out.len() != n {
        return Err(GpuSimError::DimensionMismatch {
            what: "pos_out",
            expected: n,
            found: pos_out.len(),
        });
    }
    if jac2_out.len() != n {
        return Err(GpuSimError::DimensionMismatch {
            what: "jac2_out",
            expected: n,
            found: jac2_out.len(),
        });
    }
    check_batch_size(n, 5)?;

    if q1.iter().any(|x| !x.is_finite()) {
        return Err(GpuSimError::NonFinite { what: "q1" });
    }
    if q2.iter().any(|x| !x.is_finite()) {
        return Err(GpuSimError::NonFinite { what: "q2" });
    }
    if ref1.iter().any(|x| !x.is_finite()) {
        return Err(GpuSimError::NonFinite { what: "ref1" });
    }
    if ref2.iter().any(|x| !x.is_finite()) {
        return Err(GpuSimError::NonFinite { what: "ref2" });
    }
    for c in coef {
        if c.iter().any(|x| !x.is_finite()) {
            return Err(GpuSimError::NonFinite { what: "coef" });
        }
    }

    let mut coef_flat = Vec::with_capacity(n * 5);
    for c in coef {
        coef_flat.extend_from_slice(c);
    }

    backend::launch_joint_equality_eval(q1, q2, ref1, ref2, &coef_flat, pos_out, jac2_out)
}

/// Evaluates constraint impedance `d` for soft constraint rows on GPU.
///
/// Mirrors the `d` output of [`robot_sim::constraint::impedance`]. Use
/// [`gpu_constraint_impedance_with_derivative`] to also obtain `impP`.
pub fn gpu_constraint_impedance(
    solimp: &[[f32; 5]],
    pos: &[f32],
    margin: &[f32],
    out_imp: &mut [f32],
) -> Result<(), GpuSimError> {
    let mut out_imp_prime = vec![0.0f32; pos.len()];
    gpu_constraint_impedance_with_derivative(solimp, pos, margin, out_imp, &mut out_imp_prime)
}

/// Evaluates constraint impedance `d` and its position derivative `impP` on GPU.
///
/// Mirrors [`robot_sim::constraint::impedance`] including the sign of `impP`
/// for `pos < margin`; both outputs are zero on the saturated and degenerate
/// branches, as on the CPU.
pub fn gpu_constraint_impedance_with_derivative(
    solimp: &[[f32; 5]],
    pos: &[f32],
    margin: &[f32],
    out_imp: &mut [f32],
    out_imp_prime: &mut [f32],
) -> Result<(), GpuSimError> {
    let n = pos.len();
    if solimp.len() != n {
        return Err(GpuSimError::DimensionMismatch {
            what: "solimp",
            expected: n,
            found: solimp.len(),
        });
    }
    if margin.len() != n {
        return Err(GpuSimError::DimensionMismatch {
            what: "margin",
            expected: n,
            found: margin.len(),
        });
    }
    if out_imp.len() != n {
        return Err(GpuSimError::DimensionMismatch {
            what: "out_imp",
            expected: n,
            found: out_imp.len(),
        });
    }
    if out_imp_prime.len() != n {
        return Err(GpuSimError::DimensionMismatch {
            what: "out_imp_prime",
            expected: n,
            found: out_imp_prime.len(),
        });
    }
    check_batch_size(n, 5)?;

    if pos.iter().any(|x| !x.is_finite()) {
        return Err(GpuSimError::NonFinite { what: "pos" });
    }
    if margin.iter().any(|x| !x.is_finite()) {
        return Err(GpuSimError::NonFinite { what: "margin" });
    }
    for s in solimp {
        if s.iter().any(|x| !x.is_finite()) {
            return Err(GpuSimError::NonFinite { what: "solimp" });
        }
    }

    let mut solimp_flat = Vec::with_capacity(n * 5);
    for s in solimp {
        solimp_flat.extend_from_slice(s);
    }

    backend::launch_constraint_impedance(&solimp_flat, pos, margin, out_imp, out_imp_prime)
}

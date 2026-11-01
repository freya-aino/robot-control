//! Symmetric 3x3 eigendecomposition by Jacobi rotations with quaternion
//! accumulation.
//!
//! Transferred from `docs/example-system-specs/12_linear_algebra_sparse_numerics.md`
//! section 7 (`mju_eig3Tol`), which the project treats as the governing
//! reference for simulation mathematics.
//!
//! It exists here, in `robot-model`, because the first thing that needs it is
//! [`crate::link::InertiaTensor::is_physically_plausible`]: a rotational inertia
//! is physically realisable only if its **principal moments** — its eigenvalues
//! — are non-negative and satisfy the triangle inequality. Testing the diagonal
//! entries instead is equivalent only for an already-diagonal tensor, and
//! `crates/robot-sim/specs/appendix-d-mathematical-challenges.md` entry M-03
//! records a measured counterexample that the diagonal test accepts.
//!
//! The same routine is what a principal-inertia frame needs, so it returns the
//! eigenvectors and the rotation quaternion as well, matching the example's own
//! signature. The example's compiler uses exactly this to turn a full inertia
//! tensor into a principal diagonal plus an `iquat`
//! (`13_model_compiler_setconst.md` section 5).
//!
//! Plain arrays and explicit formulas only: row-major `[f64; 9]` matrices,
//! `[f64; 3]` eigenvalues, `[f64; 4]` quaternions as `(w, x, y, z)`. Nothing
//! here uses `robot_types::math::Matrix`.

/// `eigTOL` of section 7, double precision.
pub const EIG_TOL: f64 = 4e-15;
/// `eigEPS` of section 7, double precision: the sort threshold.
pub const EIG_EPS: f64 = 1e-12;
/// Iteration cap of section 7.
const MAX_ITER: usize = 500;

/// A symmetric 3x3 eigendecomposition.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Eig3 {
    /// Eigenvalues, sorted decreasing.
    pub values: [f64; 3],
    /// Eigenvectors as the **columns** of a row-major 3x3 matrix, matching the
    /// example's `eigvec`.
    pub vectors: [f64; 9],
    /// The rotation taking the original frame to the eigenframe, as a unit
    /// quaternion `(w, x, y, z)`.
    pub quat: [f64; 4],
    /// Jacobi iterations performed. The example returns this too; it is
    /// reported rather than discarded so a caller can see a decomposition that
    /// hit the cap.
    pub iterations: usize,
}

/// Row-major 3x3 matrix of a unit quaternion (`mju_quat2Mat`).
fn quat2mat(q: [f64; 4]) -> [f64; 9] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    let d = w * w - (x * x + y * y + z * z);
    [
        d + 2.0 * x * x,
        2.0 * (x * y - w * z),
        2.0 * (x * z + w * y),
        2.0 * (x * y + w * z),
        d + 2.0 * y * y,
        2.0 * (y * z - w * x),
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        d + 2.0 * z * z,
    ]
}

/// Hamilton product `a (x) b` (`mju_mulQuat`).
fn mul_quat(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] + a[2] * b[0] + a[3] * b[1] - a[1] * b[3],
        a[0] * b[3] + a[3] * b[0] + a[1] * b[2] - a[2] * b[1],
    ]
}

/// `mju_normalize4`: fall back to the identity below `MINVAL`, and skip the
/// rescale when already unit.
fn normalize4(q: [f64; 4]) -> [f64; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if !n.is_finite() || n < 1e-15 {
        [1.0, 0.0, 0.0, 0.0]
    } else if (n - 1.0).abs() > 1e-15 {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    } else {
        q
    }
}

/// `res = a' b` for row-major 3x3.
fn mul_mat_t_mat(a: [f64; 9], b: [f64; 9]) -> [f64; 9] {
    let mut out = [0.0; 9];
    for r in 0..3 {
        for c in 0..3 {
            out[r * 3 + c] = (0..3).map(|k| a[k * 3 + r] * b[k * 3 + c]).sum();
        }
    }
    out
}

/// `res = a b` for row-major 3x3.
fn mul_mat_mat(a: [f64; 9], b: [f64; 9]) -> [f64; 9] {
    let mut out = [0.0; 9];
    for r in 0..3 {
        for c in 0..3 {
            out[r * 3 + c] = (0..3).map(|k| a[r * 3 + k] * b[k * 3 + c]).sum();
        }
    }
    out
}

/// Eigendecomposition of the symmetric row-major 3x3 `mat`, with the relative
/// tolerance `reltol` (`mju_eig3Tol`).
///
/// Only the symmetric part is meaningful; a caller passing an asymmetric matrix
/// gets an unspecified result, exactly as in the example.
#[must_use]
pub fn eig3_tol(mat: [f64; 9], reltol: f64) -> Eig3 {
    let scale = mat.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
    let tol = scale * reltol.max(EIG_TOL);
    let mut quat = [1.0, 0.0, 0.0, 0.0];
    let mut values = [0.0; 3];
    let mut iterations = 0;

    for iter in 0..MAX_ITER {
        iterations = iter;
        let eigvec = quat2mat(quat);
        let tmp = mul_mat_t_mat(eigvec, mat);
        let d = mul_mat_mat(tmp, eigvec);
        values = [d[0], d[4], d[8]];

        // Largest off-diagonal, with the example's strict comparisons so the
        // same element wins on a tie.
        let (rk, ck, rotk) = if d[1].abs() > d[2].abs() && d[1].abs() > d[5].abs() {
            (0usize, 1usize, 2usize)
        } else if d[2].abs() > d[5].abs() {
            (0, 2, 1)
        } else {
            (1, 2, 0)
        };
        if d[3 * rk + ck].abs() <= tol {
            break;
        }
        let tau = (d[4 * ck] - d[4 * rk]) / (2.0 * d[3 * rk + ck]);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let h = t / (1.0 + (1.0 + t * t).sqrt());
        let mut rot = [1.0 / (1.0 + h * h).sqrt(), 0.0, 0.0, 0.0];
        rot[rotk + 1] = if rotk == 1 { h } else { -h } * rot[0];
        quat = normalize4(mul_quat(quat, rot));
    }

    // Sort decreasing with the example's three-pass bubble, rotating the
    // quaternion by a half turn about the remaining axis on each swap.
    let eps = scale * EIG_EPS;
    // The example writes this as the literal 0.707106781186548, which differs
    // from the exact constant by about one unit in the last place. The exact
    // value is used here: this project does not chase bit-parity with the
    // example (00_INDEX.md section 1 lists that as a separate, much stricter
    // target), and the quaternion is renormalised immediately afterwards.
    const R: f64 = core::f64::consts::FRAC_1_SQRT_2;
    for j in 0..3 {
        let j1 = j % 2;
        if values[j1] + eps < values[j1 + 1] {
            values.swap(j1, j1 + 1);
            let mut rot = [R, 0.0, 0.0, 0.0];
            rot[(j1 + 2) % 3 + 1] = R;
            quat = normalize4(mul_quat(quat, rot));
        }
    }

    Eig3 {
        values,
        vectors: quat2mat(quat),
        quat,
        iterations,
    }
}

/// [`eig3_tol`] with the example's default tolerance.
#[must_use]
pub fn eig3(mat: [f64; 9]) -> Eig3 {
    eig3_tol(mat, EIG_TOL)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: &[f64], b: &[f64], tol: f64, what: &str) {
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x - y).abs() <= tol * 1.0_f64.max(y.abs()),
                "{what}: entry {i} is {x}, expected {y}"
            );
        }
    }

    #[test]
    fn a_diagonal_matrix_is_returned_sorted_with_its_own_axes() {
        let d = eig3([2.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 1.0]);
        assert_close(&d.values, &[5.0, 2.0, 1.0], 1e-14, "sorted decreasing");
    }

    #[test]
    fn the_indefinite_counterexample_has_a_negative_eigenvalue() {
        // Appendix D M-03's measured counterexample: the diagonal-entry test
        // accepts this, and its eigenvalues are 3, 1 and -1.
        let d = eig3([1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert_close(&d.values, &[3.0, 1.0, -1.0], 1e-13, "eigenvalues");
    }

    #[test]
    fn eigenvalues_reproduce_the_trace_and_determinant() {
        // Independent of this module's own algorithm: for any symmetric
        // matrix, the eigenvalues must sum to the trace and multiply to the
        // determinant.
        let cases = [
            [4.0, 1.0, 0.5, 1.0, 3.0, -0.25, 0.5, -0.25, 2.0],
            [0.05, 0.01, -0.02, 0.01, 0.07, 0.015, -0.02, 0.015, 0.04],
            [1.0, 0.9, 0.9, 0.9, 1.0, 0.9, 0.9, 0.9, 1.0],
            [1e-7, 0.0, 0.0, 0.0, 1e-7, 0.0, 0.0, 0.0, 5.9e-2],
        ];
        for m in cases {
            let d = eig3(m);
            let trace = m[0] + m[4] + m[8];
            let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
                + m[2] * (m[3] * m[7] - m[4] * m[6]);
            let sum: f64 = d.values.iter().sum();
            let prod: f64 = d.values.iter().product();
            assert!(
                (sum - trace).abs() <= 1e-12 * 1.0_f64.max(trace.abs()),
                "trace: {sum} vs {trace} for {m:?}"
            );
            assert!(
                (prod - det).abs() <= 1e-10 * 1.0_f64.max(det.abs()),
                "determinant: {prod} vs {det} for {m:?}"
            );
        }
    }

    #[test]
    fn the_eigenvectors_diagonalise_the_matrix() {
        let m = [4.0, 1.0, 0.5, 1.0, 3.0, -0.25, 0.5, -0.25, 2.0];
        let d = eig3(m);
        // V' M V must be diag(values), with V's columns the eigenvectors.
        let vt_m = mul_mat_t_mat(d.vectors, m);
        let diag = mul_mat_mat(vt_m, d.vectors);
        assert_close(
            &[diag[0], diag[4], diag[8]],
            &d.values,
            1e-12,
            "diagonalised",
        );
        for (i, off) in [diag[1], diag[2], diag[3], diag[5], diag[6], diag[7]]
            .iter()
            .enumerate()
        {
            assert!(off.abs() < 1e-12, "off-diagonal {i} is {off}");
        }
        // and V is a rotation.
        let vtv = mul_mat_t_mat(d.vectors, d.vectors);
        assert_close(
            &vtv,
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            1e-12,
            "orthonormal",
        );
    }

    #[test]
    fn a_zero_matrix_decomposes_without_dividing_by_zero() {
        let d = eig3([0.0; 9]);
        assert_eq!(d.values, [0.0, 0.0, 0.0]);
        assert_eq!(d.iterations, 0);
    }

    #[test]
    fn a_repeated_eigenvalue_is_handled() {
        // Two equal eigenvalues leave the rotation in that plane arbitrary; the
        // values must still be right.
        let d = eig3([2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 7.0]);
        assert_close(&d.values, &[7.0, 2.0, 2.0], 1e-14, "repeated");
    }
}

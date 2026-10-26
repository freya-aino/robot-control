//! Small dense linear algebra, authored.
//!
//! SPECS.md Appendix B.2.1 lists nalgebra as an *established numerical
//! foundation* and a candidate, and Appendix B.6 admits no candidate until its
//! gates pass. Until that qualification exists this crate carries the handful of
//! operations the platform actually needs: a dynamically sized dense matrix, an
//! LU solve with partial pivoting, a symmetric eigenvalue/singular-value route
//! via Jacobi rotations, and damped least squares for the inverse kinematics of
//! SPECS.md Section 8.3.
//!
//! The scope is deliberately small. This is not a linear algebra library; it is
//! the subset the retargeter, the estimator, the dynamics and the identification
//! baseline require, each with an analytic test.

use core::fmt;
use core::ops::{Index, IndexMut};

/// A dense row-major matrix of `f64`.
#[derive(Clone, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

/// Why a linear algebra operation could not produce an answer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LinalgError {
    /// Operand shapes do not admit the operation.
    ShapeMismatch {
        /// Rows of the left operand.
        lhs_rows: usize,
        /// Columns of the left operand.
        lhs_cols: usize,
        /// Rows of the right operand.
        rhs_rows: usize,
        /// Columns of the right operand.
        rhs_cols: usize,
    },
    /// The matrix is not square where the operation requires it.
    NotSquare {
        /// Row count.
        rows: usize,
        /// Column count.
        cols: usize,
    },
    /// A pivot fell below the tolerance; the system is singular to working
    /// precision. Reported rather than silently regularized.
    Singular {
        /// Index of the column whose pivot vanished.
        column: usize,
    },
    /// An entry was NaN or infinite. Propagating these produces a plausible
    /// looking result from corrupt input, so they are rejected at the boundary.
    NotFinite,
    /// An iterative routine did not converge within its iteration budget.
    NotConverged {
        /// Iterations performed.
        iterations: usize,
    },
}

impl fmt::Display for LinalgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch {
                lhs_rows,
                lhs_cols,
                rhs_rows,
                rhs_cols,
            } => write!(
                f,
                "shape mismatch: {lhs_rows}x{lhs_cols} against {rhs_rows}x{rhs_cols}"
            ),
            Self::NotSquare { rows, cols } => write!(f, "matrix is {rows}x{cols}, not square"),
            Self::Singular { column } => write!(f, "singular at column {column}"),
            Self::NotFinite => write!(f, "matrix contains a non-finite entry"),
            Self::NotConverged { iterations } => {
                write!(f, "did not converge in {iterations} iterations")
            }
        }
    }
}

impl core::error::Error for LinalgError {}

impl Matrix {
    /// A `rows` by `cols` matrix of zeros.
    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// The `n` by `n` identity.
    #[must_use]
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m[(i, i)] = 1.0;
        }
        m
    }

    /// Builds from a row-major slice.
    ///
    /// Returns `None` when the slice length is not `rows * cols`.
    #[must_use]
    pub fn from_row_major(rows: usize, cols: usize, data: &[f64]) -> Option<Self> {
        if data.len() != rows * cols {
            return None;
        }
        Some(Self {
            rows,
            cols,
            data: data.to_vec(),
        })
    }

    /// Builds a column vector from a slice.
    #[must_use]
    pub fn column(values: &[f64]) -> Self {
        Self {
            rows: values.len(),
            cols: 1,
            data: values.to_vec(),
        }
    }

    /// Builds a diagonal matrix from a slice.
    #[must_use]
    pub fn diagonal(values: &[f64]) -> Self {
        let mut m = Self::zeros(values.len(), values.len());
        for (i, &v) in values.iter().enumerate() {
            m[(i, i)] = v;
        }
        m
    }

    /// Row count.
    #[inline]
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Column count.
    #[inline]
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Row-major entries.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Mutable row-major entries.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// True when every entry is finite.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.data.iter().all(|v| v.is_finite())
    }

    /// Transpose.
    #[must_use]
    pub fn transposed(&self) -> Self {
        let mut t = Self::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                t[(c, r)] = self[(r, c)];
            }
        }
        t
    }

    /// Matrix product.
    pub fn mul(&self, rhs: &Self) -> Result<Self, LinalgError> {
        if self.cols != rhs.rows {
            return Err(LinalgError::ShapeMismatch {
                lhs_rows: self.rows,
                lhs_cols: self.cols,
                rhs_rows: rhs.rows,
                rhs_cols: rhs.cols,
            });
        }
        let mut out = Self::zeros(self.rows, rhs.cols);
        for r in 0..self.rows {
            for k in 0..self.cols {
                let a = self[(r, k)];
                if a == 0.0 {
                    continue;
                }
                for c in 0..rhs.cols {
                    out[(r, c)] += a * rhs[(k, c)];
                }
            }
        }
        Ok(out)
    }

    /// Entrywise sum.
    pub fn add(&self, rhs: &Self) -> Result<Self, LinalgError> {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            return Err(LinalgError::ShapeMismatch {
                lhs_rows: self.rows,
                lhs_cols: self.cols,
                rhs_rows: rhs.rows,
                rhs_cols: rhs.cols,
            });
        }
        let mut out = self.clone();
        for (o, r) in out.data.iter_mut().zip(&rhs.data) {
            *o += *r;
        }
        Ok(out)
    }

    /// Entrywise difference.
    pub fn sub(&self, rhs: &Self) -> Result<Self, LinalgError> {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            return Err(LinalgError::ShapeMismatch {
                lhs_rows: self.rows,
                lhs_cols: self.cols,
                rhs_rows: rhs.rows,
                rhs_cols: rhs.cols,
            });
        }
        let mut out = self.clone();
        for (o, r) in out.data.iter_mut().zip(&rhs.data) {
            *o -= *r;
        }
        Ok(out)
    }

    /// Scalar multiple.
    #[must_use]
    pub fn scaled(&self, factor: f64) -> Self {
        let mut out = self.clone();
        for v in &mut out.data {
            *v *= factor;
        }
        out
    }

    /// Adds `value` to every diagonal entry. Used for Levenberg-style damping.
    pub fn add_to_diagonal(&mut self, value: f64) {
        let n = self.rows.min(self.cols);
        for i in 0..n {
            self[(i, i)] += value;
        }
    }

    /// Frobenius norm.
    #[must_use]
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// Largest absolute entry.
    #[must_use]
    pub fn max_abs(&self) -> f64 {
        self.data.iter().fold(0.0f64, |acc, v| acc.max(v.abs()))
    }

    /// Solves `self * x = rhs` by LU decomposition with partial pivoting.
    ///
    /// `rhs` may have several columns. The pivot tolerance is relative to the
    /// largest entry of the matrix, so the singularity report does not depend on
    /// the problem's units.
    pub fn solve(&self, rhs: &Self) -> Result<Self, LinalgError> {
        if self.rows != self.cols {
            return Err(LinalgError::NotSquare {
                rows: self.rows,
                cols: self.cols,
            });
        }
        if self.rows != rhs.rows {
            return Err(LinalgError::ShapeMismatch {
                lhs_rows: self.rows,
                lhs_cols: self.cols,
                rhs_rows: rhs.rows,
                rhs_cols: rhs.cols,
            });
        }
        if !self.is_finite() || !rhs.is_finite() {
            return Err(LinalgError::NotFinite);
        }

        let n = self.rows;
        let mut a = self.clone();
        let mut b = rhs.clone();
        let scale = a.max_abs();
        let tol = if scale > 0.0 { scale * 1e-14 } else { 1e-300 };

        for col in 0..n {
            let mut pivot_row = col;
            let mut pivot_mag = a[(col, col)].abs();
            for r in (col + 1)..n {
                let m = a[(r, col)].abs();
                if m > pivot_mag {
                    pivot_mag = m;
                    pivot_row = r;
                }
            }
            if pivot_mag <= tol {
                return Err(LinalgError::Singular { column: col });
            }
            if pivot_row != col {
                a.swap_rows(col, pivot_row);
                b.swap_rows(col, pivot_row);
            }
            let pivot = a[(col, col)];
            for r in (col + 1)..n {
                let factor = a[(r, col)] / pivot;
                if factor == 0.0 {
                    continue;
                }
                for c in col..n {
                    let v = a[(col, c)];
                    a[(r, c)] -= factor * v;
                }
                for c in 0..b.cols {
                    let v = b[(col, c)];
                    b[(r, c)] -= factor * v;
                }
            }
        }

        let mut x = Self::zeros(n, b.cols);
        for c in 0..b.cols {
            for r in (0..n).rev() {
                let mut sum = b[(r, c)];
                for k in (r + 1)..n {
                    sum -= a[(r, k)] * x[(k, c)];
                }
                x[(r, c)] = sum / a[(r, r)];
            }
        }
        Ok(x)
    }

    /// Inverse by solving against the identity. Present for tests and reports;
    /// prefer [`Matrix::solve`] in numerical code.
    pub fn inverse(&self) -> Result<Self, LinalgError> {
        self.solve(&Self::identity(self.rows))
    }

    /// Swaps two rows in place.
    pub fn swap_rows(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        for c in 0..self.cols {
            self.data.swap(a * self.cols + c, b * self.cols + c);
        }
    }

    /// Eigenvalues and eigenvectors of a symmetric matrix by the cyclic Jacobi
    /// method.
    ///
    /// Returns `(values, vectors)` with eigenvalues in descending order and the
    /// corresponding eigenvectors as columns of `vectors`. Symmetry is checked
    /// rather than assumed, because a covariance matrix that has lost symmetry
    /// to accumulated rounding is a defect the estimator wants reported
    /// (SPECS.md Section 11.2 covariance consistency).
    pub fn symmetric_eigen(&self) -> Result<(Vec<f64>, Self), LinalgError> {
        if self.rows != self.cols {
            return Err(LinalgError::NotSquare {
                rows: self.rows,
                cols: self.cols,
            });
        }
        if !self.is_finite() {
            return Err(LinalgError::NotFinite);
        }
        let n = self.rows;
        let scale = self.max_abs().max(1e-300);
        for r in 0..n {
            for c in (r + 1)..n {
                if (self[(r, c)] - self[(c, r)]).abs() > 1e-9 * scale {
                    return Err(LinalgError::NotConverged { iterations: 0 });
                }
            }
        }

        let mut a = self.clone();
        let mut v = Self::identity(n);
        let max_sweeps = 100;
        for sweep in 0..max_sweeps {
            let mut off = 0.0;
            for r in 0..n {
                for c in (r + 1)..n {
                    off += a[(r, c)] * a[(r, c)];
                }
            }
            if off.sqrt() <= 1e-14 * scale {
                let mut idx: Vec<usize> = (0..n).collect();
                let diag: Vec<f64> = (0..n).map(|i| a[(i, i)]).collect();
                idx.sort_by(|&x, &y| {
                    diag[y]
                        .partial_cmp(&diag[x])
                        .unwrap_or(core::cmp::Ordering::Equal)
                });
                let values = idx.iter().map(|&i| diag[i]).collect();
                let mut vectors = Self::zeros(n, n);
                for (new_c, &old_c) in idx.iter().enumerate() {
                    for r in 0..n {
                        vectors[(r, new_c)] = v[(r, old_c)];
                    }
                }
                return Ok((values, vectors));
            }
            let _ = sweep;
            for p in 0..n {
                for q in (p + 1)..n {
                    let apq = a[(p, q)];
                    if apq.abs() <= 1e-300 {
                        continue;
                    }
                    let theta = (a[(q, q)] - a[(p, p)]) / (2.0 * apq);
                    let t = if theta >= 0.0 {
                        1.0 / (theta + (1.0 + theta * theta).sqrt())
                    } else {
                        -1.0 / (-theta + (1.0 + theta * theta).sqrt())
                    };
                    let c = 1.0 / (1.0 + t * t).sqrt();
                    let s = t * c;
                    for k in 0..n {
                        let akp = a[(k, p)];
                        let akq = a[(k, q)];
                        a[(k, p)] = c * akp - s * akq;
                        a[(k, q)] = s * akp + c * akq;
                    }
                    for k in 0..n {
                        let apk = a[(p, k)];
                        let aqk = a[(q, k)];
                        a[(p, k)] = c * apk - s * aqk;
                        a[(q, k)] = s * apk + c * aqk;
                    }
                    for k in 0..n {
                        let vkp = v[(k, p)];
                        let vkq = v[(k, q)];
                        v[(k, p)] = c * vkp - s * vkq;
                        v[(k, q)] = s * vkp + c * vkq;
                    }
                }
            }
        }
        Err(LinalgError::NotConverged {
            iterations: max_sweeps,
        })
    }

    /// Singular values in descending order, via the eigenvalues of `AᵀA`.
    ///
    /// Used by the identifiability report of SPECS.md Section 11.3, where the
    /// question is whether a parameter is distinguishable from the data at all.
    pub fn singular_values(&self) -> Result<Vec<f64>, LinalgError> {
        let ata = self.transposed().mul(self)?;
        let (values, _) = ata.symmetric_eigen()?;
        Ok(values.into_iter().map(|v| v.max(0.0).sqrt()).collect())
    }

    /// Solves the damped least-squares problem
    /// `min ||J dx - e||² + lambda² ||dx||²`.
    ///
    /// This is the step of the inverse kinematics in SPECS.md Section 8.3. The
    /// damping is what keeps a step bounded near a kinematic singularity; the
    /// undamped pseudo-inverse produces an arbitrarily large joint velocity
    /// there, which a robot cannot execute.
    ///
    /// `weights` are per-row task weights (SPECS.md Section 8.3 "weighted
    /// task-space IK"), applied as `sqrt(w)` on both sides so the normal
    /// equations stay symmetric.
    pub fn damped_least_squares(
        jacobian: &Self,
        error: &Self,
        lambda: f64,
        weights: Option<&[f64]>,
    ) -> Result<Self, LinalgError> {
        if error.cols != 1 || jacobian.rows != error.rows {
            return Err(LinalgError::ShapeMismatch {
                lhs_rows: jacobian.rows,
                lhs_cols: jacobian.cols,
                rhs_rows: error.rows,
                rhs_cols: error.cols,
            });
        }
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(LinalgError::NotFinite);
        }

        let mut j = jacobian.clone();
        let mut e = error.clone();
        if let Some(w) = weights {
            if w.len() != jacobian.rows {
                return Err(LinalgError::ShapeMismatch {
                    lhs_rows: jacobian.rows,
                    lhs_cols: jacobian.cols,
                    rhs_rows: w.len(),
                    rhs_cols: 1,
                });
            }
            for (r, &wr) in w.iter().enumerate() {
                if !wr.is_finite() || wr < 0.0 {
                    return Err(LinalgError::NotFinite);
                }
                let s = wr.sqrt();
                for c in 0..j.cols {
                    j[(r, c)] *= s;
                }
                e[(r, 0)] *= s;
            }
        }

        let jt = j.transposed();
        let mut normal = jt.mul(&j)?;
        normal.add_to_diagonal(lambda * lambda);
        let rhs = jt.mul(&e)?;
        normal.solve(&rhs)
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f64;
    #[inline]
    fn index(&self, (r, c): (usize, usize)) -> &f64 {
        debug_assert!(r < self.rows && c < self.cols, "matrix index out of bounds");
        &self.data[r * self.cols + c]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    #[inline]
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut f64 {
        debug_assert!(r < self.rows && c < self.cols, "matrix index out of bounds");
        &mut self.data[r * self.cols + c]
    }
}

impl fmt::Debug for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Matrix {}x{} [", self.rows, self.cols)?;
        for r in 0..self.rows {
            write!(f, "  ")?;
            for c in 0..self.cols {
                write!(f, "{:>12.6} ", self[(r, c)])?;
            }
            writeln!(f)?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_a_known_system() {
        // 2x + y = 5 ; x - 3y = -1  =>  x = 2, y = 1
        let a = Matrix::from_row_major(2, 2, &[2.0, 1.0, 1.0, -3.0]).unwrap();
        let b = Matrix::column(&[5.0, -1.0]);
        let x = a.solve(&b).unwrap();
        assert!((x[(0, 0)] - 2.0).abs() < 1e-12);
        assert!((x[(1, 0)] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn solve_needs_pivoting() {
        // A zero leading pivot is fine with partial pivoting and fatal without.
        let a = Matrix::from_row_major(2, 2, &[0.0, 1.0, 1.0, 0.0]).unwrap();
        let b = Matrix::column(&[3.0, 4.0]);
        let x = a.solve(&b).unwrap();
        assert!((x[(0, 0)] - 4.0).abs() < 1e-12);
        assert!((x[(1, 0)] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn singular_matrix_is_reported_not_regularized() {
        let a = Matrix::from_row_major(2, 2, &[1.0, 2.0, 2.0, 4.0]).unwrap();
        let b = Matrix::column(&[1.0, 2.0]);
        assert!(matches!(a.solve(&b), Err(LinalgError::Singular { .. })));
    }

    #[test]
    fn nonfinite_input_is_rejected() {
        let a = Matrix::from_row_major(1, 1, &[f64::NAN]).unwrap();
        let b = Matrix::column(&[1.0]);
        assert_eq!(a.solve(&b), Err(LinalgError::NotFinite));
    }

    #[test]
    fn inverse_round_trips() {
        let a =
            Matrix::from_row_major(3, 3, &[4.0, 7.0, 2.0, 3.0, 6.0, 1.0, 2.0, 5.0, 3.0]).unwrap();
        let inv = a.inverse().unwrap();
        let prod = a.mul(&inv).unwrap();
        for r in 0..3 {
            for c in 0..3 {
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!((prod[(r, c)] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn symmetric_eigen_matches_analytic_values() {
        // [[2,1],[1,2]] has eigenvalues 3 and 1.
        let a = Matrix::from_row_major(2, 2, &[2.0, 1.0, 1.0, 2.0]).unwrap();
        let (values, vectors) = a.symmetric_eigen().unwrap();
        assert!((values[0] - 3.0).abs() < 1e-10);
        assert!((values[1] - 1.0).abs() < 1e-10);
        // Eigenvectors are orthonormal.
        let vtv = vectors.transposed().mul(&vectors).unwrap();
        for r in 0..2 {
            for c in 0..2 {
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!((vtv[(r, c)] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn asymmetric_input_to_symmetric_eigen_is_rejected() {
        let a = Matrix::from_row_major(2, 2, &[1.0, 2.0, 0.0, 1.0]).unwrap();
        assert!(a.symmetric_eigen().is_err());
    }

    #[test]
    fn singular_values_expose_a_rank_deficient_regressor() {
        // Second column is twice the first: one parameter is unidentifiable.
        let a = Matrix::from_row_major(3, 2, &[1.0, 2.0, 2.0, 4.0, 3.0, 6.0]).unwrap();
        let sv = a.singular_values().unwrap();
        assert!(sv[0] > 1.0);
        assert!(
            sv[1] < 1e-7,
            "expected a vanishing singular value, got {sv:?}"
        );
    }

    #[test]
    fn damping_bounds_the_step_at_a_singularity() {
        // A rank-deficient Jacobian: undamped the step is unbounded.
        let j = Matrix::from_row_major(2, 2, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        let e = Matrix::column(&[0.1, 0.1]);
        let step = Matrix::damped_least_squares(&j, &e, 1e-2, None).unwrap();
        assert!(
            step.norm() < 1.0,
            "damped step should stay bounded: {}",
            step.norm()
        );
        assert!(Matrix::damped_least_squares(&j, &e, 0.0, None).is_err());
    }

    #[test]
    fn task_weights_bias_the_solution() {
        // Two conflicting one-dimensional tasks on one joint; the heavier wins.
        let j = Matrix::from_row_major(2, 1, &[1.0, 1.0]).unwrap();
        let e = Matrix::column(&[1.0, -1.0]);
        let balanced = Matrix::damped_least_squares(&j, &e, 1e-6, None).unwrap();
        assert!(balanced[(0, 0)].abs() < 1e-6);
        let biased = Matrix::damped_least_squares(&j, &e, 1e-6, Some(&[10.0, 1.0])).unwrap();
        assert!(biased[(0, 0)] > 0.5);
    }

    #[test]
    fn weight_length_mismatch_is_rejected() {
        let j = Matrix::from_row_major(2, 1, &[1.0, 1.0]).unwrap();
        let e = Matrix::column(&[1.0, -1.0]);
        assert!(Matrix::damped_least_squares(&j, &e, 1e-3, Some(&[1.0])).is_err());
        assert!(Matrix::damped_least_squares(&j, &e, 1e-3, Some(&[1.0, -1.0])).is_err());
    }
}

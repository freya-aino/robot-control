//! A dense, strictly convex, box-constrained quadratic program, solved by a
//! primal active-set method — the Rust-native constrained solver SPECS.md
//! Section 8.3 requires qualifying "before selecting differential-QP IK".
//!
//! The problem is `min 0.5 x' H x + g' x` subject to `lo <= x <= hi`, with
//! `H` symmetric positive definite. This is exactly the shape of one
//! differential-IK step with hard joint limits inside the solve
//! (`H = J' J + lambda I`, `g = J' r`, bounds from the joint limits and the
//! trust region), and it is deliberately nothing more: no general linear
//! constraints, no conic rows, no sparsity. Appendix B.2.8's candidates
//! (Clarabel and the others) remain `unverified` and outside the closure
//! (ADR 0003); this module is project-authored, `std`-only, and qualified by
//! its own tests the way Section 8.3 asks — bounded iterations, explicit
//! termination, KKT-checked optimality (`docs/decisions/0018`).
//!
//! The method is the textbook primal active set for box constraints
//! (Nocedal & Wright, *Numerical Optimization*, section 16.5, specialized to
//! bounds): hold an active set of variables fixed at a bound, minimize over
//! the free variables, take the longest feasible step toward that minimizer
//! (activating the first blocking bound), and when the subproblem is already
//! minimal, release the active variable whose multiplier says releasing it
//! lowers the objective — lowest index first, which is the Bland-style
//! tie-break that avoids cycling. Each iteration either moves strictly
//! downhill or changes the active set, and the iteration cap bounds the solve
//! regardless (Section 8.3: "bounded iterations/deadlines").

use core::fmt;
use robot_types::math::{LinalgError, Matrix};

/// Which bound, if any, a variable currently sits on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BoundState {
    Free,
    AtLower,
    AtUpper,
}

/// Why the solver stopped.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QpTermination {
    /// The Karush-Kuhn-Tucker conditions hold to the solver's tolerance,
    /// which for a strictly convex problem is global optimality.
    Converged,
    /// The iteration cap was reached first. The iterate is feasible and no
    /// worse than the start, and the caller decides whether it is usable.
    IterationCap,
}

/// A solved (or capped) box QP.
#[derive(Clone, Debug)]
pub struct QpSolution {
    /// The final iterate, always within the bounds.
    pub x: Vec<f64>,
    /// Active-set iterations performed.
    pub iterations: usize,
    /// Why the solve stopped.
    pub termination: QpTermination,
}

/// Why a solve could not run.
#[derive(Clone, PartialEq, Debug)]
pub enum QpError {
    /// `H`, `g`, `lo` and `hi` disagree on the problem's width.
    WidthMismatch,
    /// A bound pair has `lo > hi`, or a bound is NaN: there is no feasible
    /// point, and Section 8.3 wants infeasibility answered, not looped on.
    InfeasibleBounds {
        /// The offending variable.
        variable: usize,
    },
    /// `g` or `H` carries a non-finite entry.
    NotFinite,
    /// The reduced system could not be solved, which for this problem class
    /// means `H` is not positive definite on the free variables.
    Linalg(LinalgError),
}

impl fmt::Display for QpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WidthMismatch => f.write_str("H, g and the bounds disagree on width"),
            Self::InfeasibleBounds { variable } => {
                write!(f, "bounds of variable {variable} admit no feasible point")
            }
            Self::NotFinite => f.write_str("H or g carries a non-finite entry"),
            Self::Linalg(e) => write!(f, "{e}"),
        }
    }
}

impl core::error::Error for QpError {}

impl From<LinalgError> for QpError {
    fn from(e: LinalgError) -> Self {
        Self::Linalg(e)
    }
}

/// Feasibility tolerance on bound activation and multiplier signs, relative
/// to the problem's own scale.
const KKT_TOLERANCE: f64 = 1e-10;

/// Solves `min 0.5 x' H x + g' x` subject to `lo <= x <= hi`.
///
/// `hi[i] = f64::INFINITY` (or `lo[i] = NEG_INFINITY`) leaves that side
/// unbounded. The start is the projection of zero into the box. The solve is
/// bounded by `max_iterations` active-set iterations and reports which
/// termination it reached; the returned iterate is feasible in every case.
///
/// # Errors
///
/// [`QpError::WidthMismatch`] and [`QpError::InfeasibleBounds`] for a
/// malformed problem, [`QpError::NotFinite`] for non-finite data, and
/// [`QpError::Linalg`] when `H` is not positive definite on the free
/// variables.
pub fn solve_box_qp(
    h: &Matrix,
    g: &[f64],
    lo: &[f64],
    hi: &[f64],
    max_iterations: usize,
) -> Result<QpSolution, QpError> {
    let n = g.len();
    if h.rows() != n || h.cols() != n || lo.len() != n || hi.len() != n {
        return Err(QpError::WidthMismatch);
    }
    if !h.is_finite() || g.iter().any(|v| !v.is_finite()) {
        return Err(QpError::NotFinite);
    }
    for i in 0..n {
        // A NaN bound is incomparable and must be refused with the rest.
        if lo[i].partial_cmp(&hi[i]) != Some(core::cmp::Ordering::Less)
            && lo[i].partial_cmp(&hi[i]) != Some(core::cmp::Ordering::Equal)
        {
            return Err(QpError::InfeasibleBounds { variable: i });
        }
    }

    // Start at the projection of zero, with the matching active set.
    let mut x = vec![0.0; n];
    let mut state = vec![BoundState::Free; n];
    for i in 0..n {
        if lo[i] > 0.0 {
            x[i] = lo[i];
            state[i] = BoundState::AtLower;
        } else if hi[i] < 0.0 {
            x[i] = hi[i];
            state[i] = BoundState::AtUpper;
        }
    }
    if n == 0 {
        return Ok(QpSolution {
            x,
            iterations: 0,
            termination: QpTermination::Converged,
        });
    }

    // The scale the tolerance is relative to.
    let scale = g
        .iter()
        .map(|v| v.abs())
        .fold(h.max_abs(), f64::max)
        .max(1.0);
    let tol = KKT_TOLERANCE * scale;

    let mut iterations = 0;
    while iterations < max_iterations {
        iterations += 1;

        // Minimize over the free variables with the active ones fixed:
        // H_FF d_F = -(g + H x)_F  gives the step to the subproblem minimum.
        let free: Vec<usize> = (0..n).filter(|i| state[*i] == BoundState::Free).collect();
        let gradient = full_gradient(h, g, &x);
        let mut step = vec![0.0; n];
        if !free.is_empty() {
            let m = free.len();
            let mut hff = Matrix::zeros(m, m);
            let mut rhs = Matrix::zeros(m, 1);
            for (a, &i) in free.iter().enumerate() {
                rhs[(a, 0)] = -gradient[i];
                for (b, &j) in free.iter().enumerate() {
                    hff[(a, b)] = h[(i, j)];
                }
            }
            let d = hff.solve(&rhs)?;
            for (a, &i) in free.iter().enumerate() {
                step[i] = d[(a, 0)];
            }
        }

        let step_norm = step.iter().map(|v| v.abs()).fold(0.0, f64::max);
        if step_norm <= tol {
            // At the subproblem minimum. Optimal unless an active variable's
            // multiplier says its bound is holding the objective up:
            // at a lower bound optimality needs `grad >= 0`, at an upper
            // bound `grad <= 0`. Release the lowest-index violator.
            let release = (0..n).find(|&i| match state[i] {
                BoundState::AtLower => gradient[i] < -tol,
                BoundState::AtUpper => gradient[i] > tol,
                BoundState::Free => false,
            });
            match release {
                Some(i) => {
                    state[i] = BoundState::Free;
                    continue;
                }
                None => {
                    return Ok(QpSolution {
                        x,
                        iterations,
                        termination: QpTermination::Converged,
                    });
                }
            }
        }

        // Longest feasible step toward the subproblem minimum; the first
        // blocking bound joins the active set.
        let mut alpha = 1.0f64;
        let mut blocking: Option<(usize, BoundState)> = None;
        for &i in &free {
            if step[i] > tol && hi[i].is_finite() {
                let room = (hi[i] - x[i]) / step[i];
                if room < alpha {
                    alpha = room;
                    blocking = Some((i, BoundState::AtUpper));
                }
            } else if step[i] < -tol && lo[i].is_finite() {
                let room = (lo[i] - x[i]) / step[i];
                if room < alpha {
                    alpha = room;
                    blocking = Some((i, BoundState::AtLower));
                }
            }
        }
        let alpha = alpha.max(0.0);
        for &i in &free {
            x[i] += alpha * step[i];
        }
        if let Some((i, side)) = blocking {
            // Land exactly on the bound rather than within float noise of it.
            x[i] = match side {
                BoundState::AtLower => lo[i],
                BoundState::AtUpper => hi[i],
                BoundState::Free => unreachable!("a blocking bound is a bound"),
            };
            state[i] = side;
        }
    }

    Ok(QpSolution {
        x,
        iterations,
        termination: QpTermination::IterationCap,
    })
}

/// `H x + g`.
fn full_gradient(h: &Matrix, g: &[f64], x: &[f64]) -> Vec<f64> {
    let n = g.len();
    let mut out = g.to_vec();
    for i in 0..n {
        for j in 0..n {
            out[i] += h[(i, j)] * x[j];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spd(n: usize, seed: u64) -> Matrix {
        // A' A + I is symmetric positive definite for any A.
        let mut s = seed;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        };
        let mut a = Matrix::zeros(n, n);
        for r in 0..n {
            for c in 0..n {
                a[(r, c)] = next();
            }
        }
        let mut h = a.transposed().mul(&a).unwrap();
        h.add_to_diagonal(1.0);
        h
    }

    /// The KKT conditions, which for a strictly convex problem are global
    /// optimality: free variables at zero gradient, a variable at its lower
    /// bound with non-negative gradient, at its upper with non-positive.
    fn assert_kkt(h: &Matrix, g: &[f64], lo: &[f64], hi: &[f64], x: &[f64], tol: f64) {
        let grad = full_gradient(h, g, x);
        for i in 0..g.len() {
            assert!(
                x[i] >= lo[i] - tol && x[i] <= hi[i] + tol,
                "variable {i} infeasible: {} outside [{}, {}]",
                x[i],
                lo[i],
                hi[i]
            );
            let at_lo = (x[i] - lo[i]).abs() <= tol;
            let at_hi = (hi[i] - x[i]).abs() <= tol;
            if at_lo {
                assert!(
                    grad[i] >= -tol,
                    "variable {i} at lower, gradient {}",
                    grad[i]
                );
            } else if at_hi {
                assert!(
                    grad[i] <= tol,
                    "variable {i} at upper, gradient {}",
                    grad[i]
                );
            } else {
                assert!(
                    grad[i].abs() <= tol,
                    "free variable {i} has gradient {}",
                    grad[i]
                );
            }
        }
    }

    #[test]
    fn an_unconstrained_problem_matches_the_linear_solve() {
        let h = spd(6, 7);
        let g: Vec<f64> = (0..6).map(|i| (i as f64) - 2.5).collect();
        let lo = vec![f64::NEG_INFINITY; 6];
        let hi = vec![f64::INFINITY; 6];
        let s = solve_box_qp(&h, &g, &lo, &hi, 100).unwrap();
        assert_eq!(s.termination, QpTermination::Converged);
        let reference = h
            .solve(&Matrix::column(&g.iter().map(|v| -v).collect::<Vec<_>>()))
            .unwrap();
        for i in 0..6 {
            assert!((s.x[i] - reference[(i, 0)]).abs() < 1e-9);
        }
    }

    #[test]
    fn a_blocking_bound_is_hit_exactly() {
        // One variable, minimum at x = 2, upper bound at 1.
        let h = Matrix::identity(1);
        let s = solve_box_qp(&h, &[-2.0], &[f64::NEG_INFINITY], &[1.0], 100).unwrap();
        assert_eq!(s.termination, QpTermination::Converged);
        assert_eq!(s.x[0], 1.0);
    }

    #[test]
    fn a_wrongly_active_start_is_released() {
        // The projection of zero starts x at its lower bound 1, but the
        // minimum is at 3: the solver must release the bound and move.
        let h = Matrix::identity(1);
        let s = solve_box_qp(&h, &[-3.0], &[1.0], &[10.0], 100).unwrap();
        assert_eq!(s.termination, QpTermination::Converged);
        assert!((s.x[0] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn randomized_problems_satisfy_the_kkt_conditions() {
        for case in 0..200u64 {
            let n = 1 + (case % 12) as usize;
            let h = spd(n, case * 31 + 5);
            let mut s = case * 77 + 13;
            let mut next = || {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 11) as f64) / ((1u64 << 53) as f64) * 4.0 - 2.0
            };
            let g: Vec<f64> = (0..n).map(|_| next()).collect();
            let (mut lo, mut hi) = (Vec::new(), Vec::new());
            for _ in 0..n {
                let a = next();
                let b = next();
                lo.push(a.min(b));
                hi.push(a.max(b));
            }
            // Mix in unbounded sides.
            if n > 2 {
                lo[0] = f64::NEG_INFINITY;
                hi[1] = f64::INFINITY;
            }
            let sol = solve_box_qp(&h, &g, &lo, &hi, 200).unwrap();
            assert_eq!(
                sol.termination,
                QpTermination::Converged,
                "case {case} hit the cap"
            );
            assert_kkt(&h, &g, &lo, &hi, &sol.x, 1e-7);
        }
    }

    #[test]
    fn malformed_problems_are_refused_with_the_reason() {
        let h = Matrix::identity(2);
        assert_eq!(
            solve_box_qp(&h, &[0.0], &[0.0], &[1.0], 10).unwrap_err(),
            QpError::WidthMismatch
        );
        assert_eq!(
            solve_box_qp(&h, &[0.0, 0.0], &[0.0, 2.0], &[1.0, 1.0], 10).unwrap_err(),
            QpError::InfeasibleBounds { variable: 1 }
        );
        assert_eq!(
            solve_box_qp(&h, &[f64::NAN, 0.0], &[0.0, 0.0], &[1.0, 1.0], 10).unwrap_err(),
            QpError::NotFinite
        );
        assert_eq!(
            solve_box_qp(&h, &[0.0, 0.0], &[f64::NAN, 0.0], &[1.0, 1.0], 10).unwrap_err(),
            QpError::InfeasibleBounds { variable: 0 }
        );
    }

    #[test]
    fn the_iteration_cap_is_reported_and_the_iterate_stays_feasible() {
        let h = spd(8, 3);
        let g = vec![-1.0; 8];
        let lo = vec![-0.1; 8];
        let hi = vec![0.1; 8];
        let s = solve_box_qp(&h, &g, &lo, &hi, 1).unwrap();
        assert_eq!(s.termination, QpTermination::IterationCap);
        for (i, v) in s.x.iter().enumerate() {
            assert!(*v >= lo[i] - 1e-12 && *v <= hi[i] + 1e-12);
        }
    }
}

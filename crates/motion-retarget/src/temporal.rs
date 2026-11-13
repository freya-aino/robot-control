//! Temporal refinement, per SPECS.md Section 8.3: "weighted task-space IK
//! followed by temporal refinement."
//!
//! Per-frame IK solutions are independent and jitter where tasks conflict
//! or a frame is noisy. [`smooth`] replaces each coordinate's trajectory
//! `y` with the minimizer of
//!
//! ```text
//! sum_t (x_t - y_t)^2 + lambda * sum_t (x_{t-1} - 2 x_t + x_{t+1})^2
//! ```
//!
//! a discrete smoothing spline. Its normal equations `(I + lambda D'D) x = y`
//! are symmetric positive definite and pentadiagonal, solved here by a
//! banded Cholesky factorization in `O(n)`. The second difference of a
//! straight line is zero, so constant-velocity motion passes through
//! unchanged and a strike's velocity profile is changed only where it
//! bends; `lambda` sets how much bending is penalized, measured in frames
//! (it is not rescaled by the frame interval).
//!
//! The smoother does not know about joint limits or contacts. The
//! retargeting pipeline therefore re-projects every smoothed frame onto the
//! IK tasks (see [`crate::retarget`]), which restores limits by projection
//! and pulls planted feet back.

use robot_types::frames::Quat;

/// Why smoothing failed.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum SmoothError {
    /// `lambda` was negative or non-finite.
    InvalidLambda,
    /// Rows had differing widths.
    Ragged,
}

impl core::fmt::Display for SmoothError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidLambda => f.write_str("lambda must be finite and non-negative"),
            Self::Ragged => f.write_str("every frame must have the same width"),
        }
    }
}

impl core::error::Error for SmoothError {}

/// Smooths a single scalar series.
pub fn smooth_series(y: &[f64], lambda: f64) -> Result<Vec<f64>, SmoothError> {
    if !lambda.is_finite() || lambda < 0.0 {
        return Err(SmoothError::InvalidLambda);
    }
    let n = y.len();
    if n < 3 || lambda == 0.0 {
        return Ok(y.to_vec());
    }
    // Bands of A = I + lambda D'D: a[i][k] is A(i, i + k), k = 0..=2.
    let mut a = vec![[0.0f64; 3]; n];
    for row in a.iter_mut() {
        row[0] = 1.0;
    }
    // Each second-difference row r touches i = r, r+1, r+2 with
    // coefficients (1, -2, 1); add lambda * c c' into the bands.
    let c = [1.0, -2.0, 1.0];
    for r in 0..n - 2 {
        for p in 0..3 {
            for q in p..3 {
                a[r + p][q - p] += lambda * c[p] * c[q];
            }
        }
    }
    // Banded Cholesky A = L L', L lower with bandwidth 2, stored as
    // l[i][k] = L(i, i - k).
    let mut l = vec![[0.0f64; 3]; n];
    for i in 0..n {
        for k in (0..=2usize.min(i)).rev() {
            let j = i - k;
            let mut sum = a[j][k];
            for m in 1..=2usize {
                if k + m <= 2 && m <= j {
                    sum -= l[i][k + m] * l[j][m];
                }
            }
            if k == 0 {
                l[i][0] = sum.sqrt();
            } else {
                l[i][k] = sum / l[j][0];
            }
        }
    }
    let mut z = vec![0.0; n];
    for i in 0..n {
        let mut s = y[i];
        for k in 1..=2usize.min(i) {
            s -= l[i][k] * z[i - k];
        }
        z[i] = s / l[i][0];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = z[i];
        for k in 1..=2usize {
            if i + k < n {
                s -= l[i + k][k] * x[i + k];
            }
        }
        x[i] = s / l[i][0];
    }
    Ok(x)
}

/// Smooths every column of a frame-major trajectory.
pub fn smooth(trajectory: &[Vec<f64>], lambda: f64) -> Result<Vec<Vec<f64>>, SmoothError> {
    let width = trajectory.first().map_or(0, Vec::len);
    if trajectory.iter().any(|r| r.len() != width) {
        return Err(SmoothError::Ragged);
    }
    let mut out = trajectory.to_vec();
    for c in 0..width {
        let col: Vec<f64> = trajectory.iter().map(|r| r[c]).collect();
        for (row, v) in out.iter_mut().zip(smooth_series(&col, lambda)?) {
            row[c] = v;
        }
    }
    Ok(out)
}

/// Smooths a rotation trajectory by its hemisphere-aligned quaternion
/// components, renormalized; accurate while successive rotations are close,
/// which is the case for a per-frame trajectory.
pub fn smooth_rotations(rotations: &[Quat], lambda: f64) -> Result<Vec<Quat>, SmoothError> {
    let mut aligned = rotations.to_vec();
    for i in 1..aligned.len() {
        let (p, q) = (aligned[i - 1], aligned[i]);
        if p.w * q.w + p.x * q.x + p.y * q.y + p.z * q.z < 0.0 {
            aligned[i] = Quat {
                w: -q.w,
                x: -q.x,
                y: -q.y,
                z: -q.z,
            };
        }
    }
    let rows: Vec<Vec<f64>> = aligned.iter().map(|q| vec![q.w, q.x, q.y, q.z]).collect();
    Ok(smooth(&rows, lambda)?
        .into_iter()
        .map(|r| {
            Quat {
                w: r[0],
                x: r[1],
                y: r[2],
                z: r[3],
            }
            .normalized()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_straight_line_passes_through_unchanged() {
        let y: Vec<f64> = (0..50).map(|i| 0.3 * f64::from(i) - 2.0).collect();
        let x = smooth_series(&y, 100.0).unwrap();
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn the_solution_satisfies_the_normal_equations() {
        let y: Vec<f64> = (0..20)
            .map(|i| (f64::from(i) * 0.7).sin() + if i % 3 == 0 { 0.5 } else { 0.0 })
            .collect();
        let lambda = 3.0;
        let x = smooth_series(&y, lambda).unwrap();
        // Residual of (I + lambda D'D) x - y, computed directly.
        let n = y.len();
        let mut ax = x.clone();
        for r in 0..n - 2 {
            let d = x[r] - 2.0 * x[r + 1] + x[r + 2];
            ax[r] += lambda * d;
            ax[r + 1] -= 2.0 * lambda * d;
            ax[r + 2] += lambda * d;
        }
        for (a, b) in ax.iter().zip(&y) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn alternating_jitter_is_attenuated() {
        let y: Vec<f64> = (0..100)
            .map(|i| if i % 2 == 0 { 0.1 } else { -0.1 })
            .collect();
        let x = smooth_series(&y, 10.0).unwrap();
        let peak = x[10..90].iter().fold(0.0f64, |m, v| m.max(v.abs()));
        // The alternating mode's interior gain is 1 / (1 + 16 lambda) =
        // 1/161; the ends perturb it slowly, so the bound allows a margin.
        assert!(peak < 0.1 / 100.0, "{peak}");
    }

    #[test]
    fn rotations_stay_unit_and_hemisphere_flips_are_harmless() {
        let q = Quat::from_rz(robot_types::units::Radians(0.4)).unwrap();
        let flipped = Quat {
            w: -q.w,
            x: -q.x,
            y: -q.y,
            z: -q.z,
        };
        let s = smooth_rotations(&[q, flipped, q, flipped, q], 5.0).unwrap();
        for r in s {
            assert!(r.angle_to(q).value() < 1e-6);
        }
    }

    #[test]
    fn invalid_input_is_rejected() {
        assert_eq!(
            smooth_series(&[1.0, 2.0, 3.0], -1.0),
            Err(SmoothError::InvalidLambda)
        );
        assert_eq!(smooth(&[vec![1.0], vec![]], 1.0), Err(SmoothError::Ragged));
    }
}

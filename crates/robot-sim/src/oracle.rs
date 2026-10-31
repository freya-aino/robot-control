//! Pure analytic oracles and pointwise tolerance comparison for fidelity
//! checks.
//!
//! Everything here is `std` only and free of side effects. Reference values
//! come from closed forms or conservation laws, never from a reference simulator output,
//! so a pass shows agreement with the stated analytic reference only. It is
//! not measured parity with any external simulator.

use core::f64::consts::{FRAC_PI_2, PI};

/// Absolute and relative tolerance for a pointwise series comparison. Sample
/// `i` passes when `|reference_i - candidate_i| <= abs + rel * |reference_i|`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tolerance {
    /// Absolute tolerance. Must be finite and non-negative.
    pub abs: f64,
    /// Relative tolerance. Must be finite and non-negative.
    pub rel: f64,
}

/// Outcome of comparing a candidate series with its reference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OracleReport {
    /// Largest absolute pointwise error. `f64::INFINITY` when some sample's
    /// error is not finite.
    pub max_error: f64,
    /// Index of the first sample attaining `max_error`.
    pub at_sample: usize,
    /// True when every sample meets its tolerance and no error is non-finite.
    pub pass: bool,
}

/// Why a comparison or tolerance was rejected.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum OracleError {
    /// The reference and candidate series have different lengths. Series are
    /// never truncated to a common length.
    LengthMismatch {
        /// Length of the reference series.
        reference: usize,
        /// Length of the candidate series.
        candidate: usize,
    },
    /// Both series are empty, so there is no sample to compare.
    EmptySeries,
    /// A tolerance component is negative or not finite.
    InvalidTolerance,
}

impl core::fmt::Display for OracleError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LengthMismatch {
                reference,
                candidate,
            } => write!(
                f,
                "series length mismatch: reference has {reference} samples, candidate has {candidate}"
            ),
            Self::EmptySeries => write!(f, "cannot compare empty series"),
            Self::InvalidTolerance => {
                write!(f, "tolerance components must be finite and non-negative")
            }
        }
    }
}

impl core::error::Error for OracleError {}

/// Exact angle of an undamped pendulum `theta'' = -sin(theta)` released from
/// rest at `theta0`, evaluated at `t`.
///
/// Time is scaled by `sqrt(g / L)`: a physical pendulum of length `L` under
/// gravity `g` has `theta(t_phys) = pendulum_closed_form(theta0, t_phys * sqrt(g / L))`.
///
/// Domain: `|theta0| < pi` (the oscillatory swing) and finite `t`. Any other
/// input returns `f64::NAN`, which [`compare_series`] reports as a failure.
/// Inside the domain, no input is rejected for its distance from `pi`: the
/// complementary modulus is computed directly and stays positive (see
/// [`pendulum_period`] for the precision analysis).
///
/// Solution: with `k = sin(theta0 / 2)`, `k' = cos(theta0 / 2)`, and `K = K(k)`
/// the complete elliptic integral of the first kind, `theta(t) = 2 asin(k sn(K - t | k))`.
/// At `t = 0`, `sn(K) = 1` gives `theta0`. The quarter period is `K`, so the
/// full period is `4 K`. `sn` is evaluated by the descending Landen (AGM)
/// recursion of Abramowitz and Stegun, section 16.4.
///
/// Precision near the separatrix (`|theta0| -> pi`): with gap `g = pi - |theta0|`,
/// `k' = cos(theta0 / 2) ~ g / 2`, and `K ~ ln(4 / k')`. Forming
/// `k' = sqrt(1 - k^2)` loses every digit once `g` is near `2e-8`, because
/// `1 - k^2 ~ g^2 / 4` falls below the rounding of `k^2`, so `k'` comes from
/// `cos`. The rounding of `theta0 / 2` then gives a relative error of about
/// `eps / g` in `k'`, which is an absolute error of about `eps / g` in `K`. The
/// closed form is therefore accurate to roughly `eps / g` near the separatrix,
/// and is only a statement about the rounded input when `g` is near `4.4e-16`,
/// the spacing of `f64` values just below `pi`.
#[must_use]
pub fn pendulum_closed_form(theta0: f64, t: f64) -> f64 {
    let Some((k, kc)) = pendulum_moduli(theta0) else {
        return f64::NAN;
    };
    if !t.is_finite() {
        return f64::NAN;
    }
    let s = jacobi_sn(complete_k(kc) - t, k, kc);
    2.0 * (k * s).clamp(-1.0, 1.0).asin()
}

/// Period of the undamped pendulum released from rest at `theta0`, in units of
/// `sqrt(L / g)`: `4 K(k)` with `k = sin(theta0 / 2)`. The physical period is
/// `pendulum_period(theta0) * sqrt(L / g)`. Returns `f64::NAN` outside the
/// domain of [`pendulum_closed_form`].
///
/// Near `|theta0| = pi` the period is `4 K ~ 4 ln(8 / g)` with `g = pi - |theta0|`,
/// so it grows only logarithmically. For `g = 1e-6` it is about 63.6. The
/// asymptote is checked in `pendulum_period_near_the_separatrix_matches_its_logarithmic_asymptote`.
#[must_use]
pub fn pendulum_period(theta0: f64) -> f64 {
    match pendulum_moduli(theta0) {
        Some((_, kc)) => 4.0 * complete_k(kc),
        None => f64::NAN,
    }
}

/// Modulus `k = sin(theta0 / 2)` and complementary modulus `k' = cos(theta0 / 2)`,
/// or `None` outside the domain `|theta0| < pi`. `k'` is the only input the
/// near-separatrix limit needs, so it is taken from `cos` and never as
/// `sqrt(1 - k^2)`. A non-finite or non-positive `k'` is rejected: it would make
/// `K` infinite, which is the case where `m = k^2` rounds to 1.
fn pendulum_moduli(theta0: f64) -> Option<(f64, f64)> {
    if !theta0.is_finite() || theta0.abs() >= PI {
        return None;
    }
    let half = theta0 / 2.0;
    let kc = half.cos();
    if kc.is_finite() && kc > 0.0 {
        Some((half.sin(), kc))
    } else {
        None
    }
}

/// Complete elliptic integral of the first kind `K` at complementary modulus
/// `kc = k'`, via the arithmetic-geometric mean: `K = pi / (2 AGM(1, k'))`.
fn complete_k(kc: f64) -> f64 {
    FRAC_PI_2 / agm(1.0, kc)
}

/// Arithmetic-geometric mean of two positive numbers.
fn agm(mut a: f64, mut b: f64) -> f64 {
    for _ in 0..64 {
        let next_a = 0.5 * (a + b);
        b = (a * b).sqrt();
        a = next_a;
    }
    a
}

/// Jacobi elliptic function `sn(u | k)` for modulus `k` in `[0, 1)`, given its
/// complementary modulus `kc = k'`, by the descending Landen recursion
/// (Abramowitz and Stegun 16.4).
///
/// `sn` depends on `k` only through `k^2`, so `|k|` is used for `c_0`. The
/// outer `2 asin(k sn)` keeps the sign of `k`, which makes the solution odd in
/// `theta0`.
///
/// Ascending AGM levels start from `a_0 = 1`, `b_0 = k'`, `c_0 = |k|`, and give
/// `c_n / a_n` shrinking to zero. Starting from `phi_N = 2^N a_N u`, the
/// recursion `phi_{n-1} = (phi_n + asin((c_n / a_n) sin phi_n)) / 2` runs down
/// to `phi_0`, and `sn(u | k) = sin(phi_0)`.
fn jacobi_sn(u: f64, k: f64, kc: f64) -> f64 {
    const MAX_LEVELS: usize = 64;
    let mut ratio = [0.0_f64; MAX_LEVELS + 1];
    let (mut a, mut b, mut c) = (1.0_f64, kc, k.abs());
    let mut levels = 0_usize;
    while c / a > f64::EPSILON && levels < MAX_LEVELS {
        let next_a = 0.5 * (a + b);
        let next_b = (a * b).sqrt();
        let next_c = 0.5 * (a - b);
        a = next_a;
        b = next_b;
        c = next_c;
        levels += 1;
        ratio[levels] = c / a;
    }
    let mut phi = 2.0_f64.powi(levels as i32) * a * u;
    for n in (1..=levels).rev() {
        phi = 0.5 * (phi + (ratio[n] * phi.sin()).clamp(-1.0, 1.0).asin());
    }
    phi.sin()
}

/// Relative mechanical-energy drift `max_i |E_i - E_0| / |E_0|` over a series
/// of energies. A conserved system has drift zero.
///
/// Returns `f64::NAN` when the series is empty, when `E_0` is zero, or when any
/// energy is not finite. [`compare_series`] treats NaN as a failure.
#[must_use]
pub fn energy_drift(energies: &[f64]) -> f64 {
    let Some((&e0, rest)) = energies.split_first() else {
        return f64::NAN;
    };
    if e0 == 0.0 || !e0.is_finite() {
        return f64::NAN;
    }
    let mut worst = 0.0_f64;
    for &e in rest {
        if !e.is_finite() {
            return f64::NAN;
        }
        worst = worst.max((e - e0).abs() / e0.abs());
    }
    worst
}

/// Compares `candidate` with `reference` sample by sample.
///
/// Sample `i` passes when its absolute error is finite and at most
/// `tol.abs + tol.rel * |reference_i|`. The report carries the largest
/// absolute error and its first index. The pass flag depends on every sample,
/// not only on the largest error.
///
/// # Errors
///
/// [`OracleError::LengthMismatch`] when the lengths differ (never truncated),
/// [`OracleError::EmptySeries`] when both are empty, and
/// [`OracleError::InvalidTolerance`] when a tolerance component is negative or
/// not finite.
pub fn compare_series(
    reference: &[f64],
    candidate: &[f64],
    tol: &Tolerance,
) -> Result<OracleReport, OracleError> {
    if reference.len() != candidate.len() {
        return Err(OracleError::LengthMismatch {
            reference: reference.len(),
            candidate: candidate.len(),
        });
    }
    if reference.is_empty() {
        return Err(OracleError::EmptySeries);
    }
    if !(tol.abs.is_finite() && tol.abs >= 0.0 && tol.rel.is_finite() && tol.rel >= 0.0) {
        return Err(OracleError::InvalidTolerance);
    }

    let mut max_error = 0.0_f64;
    let mut at_sample = 0_usize;
    let mut pass = true;
    for (i, (&r, &c)) in reference.iter().zip(candidate).enumerate() {
        let error = (r - c).abs();
        if !error.is_finite() {
            return Ok(OracleReport {
                max_error: f64::INFINITY,
                at_sample: i,
                pass: false,
            });
        }
        if error > max_error {
            max_error = error;
            at_sample = i;
        }
        if error > tol.abs + tol.rel * r.abs() {
            pass = false;
        }
    }
    Ok(OracleReport {
        max_error,
        at_sample,
        pass,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent reference: fixed-step RK4 of `theta'' = -sin(theta)`.
    /// No closed form is used, so agreement with `pendulum_closed_form` is a
    /// cross-check of the Landen evaluation.
    fn rk4_pendulum(theta0: f64, t_end: f64, steps: usize) -> Vec<f64> {
        let dt = t_end / steps as f64;
        let accel = |theta: f64| -theta.sin();
        let (mut theta, mut omega) = (theta0, 0.0_f64);
        let mut out = Vec::with_capacity(steps + 1);
        out.push(theta);
        for _ in 0..steps {
            let k1t = omega;
            let k1w = accel(theta);
            let k2t = omega + 0.5 * dt * k1w;
            let k2w = accel(theta + 0.5 * dt * k1t);
            let k3t = omega + 0.5 * dt * k2w;
            let k3w = accel(theta + 0.5 * dt * k2t);
            let k4t = omega + dt * k3w;
            let k4w = accel(theta + dt * k3t);
            theta += dt / 6.0 * (k1t + 2.0 * k2t + 2.0 * k3t + k4t);
            omega += dt / 6.0 * (k1w + 2.0 * k2w + 2.0 * k3w + k4w);
            out.push(theta);
        }
        out
    }

    /// Hand derivation of K(m) by its power series in m:
    /// K(m) = (pi/2) * sum_n [ (2n)! / (4^n (n!)^2) ]^2 m^n. The coefficient
    /// ratio is c_n = c_{n-1} * ((2n-1)/(2n))^2, starting from c_0 = 1.
    fn complete_k_series(m: f64) -> f64 {
        let mut coefficient = 1.0_f64;
        let mut power = 1.0_f64;
        let mut sum = 0.0_f64;
        for n in 0..60 {
            if n > 0 {
                let ratio = (2 * n - 1) as f64 / (2 * n) as f64;
                coefficient *= ratio * ratio;
                power *= m;
            }
            sum += coefficient * power;
        }
        FRAC_PI_2 * sum
    }

    #[test]
    fn pendulum_closed_form_starts_at_the_release_angle() {
        for theta0 in [0.2_f64, 1.0, -1.0, 2.5] {
            let got = pendulum_closed_form(theta0, 0.0);
            assert!((got - theta0).abs() < 1e-12, "theta0={theta0} got={got}");
        }
    }

    #[test]
    fn pendulum_closed_form_is_zero_at_the_quarter_period() {
        // sn(0 | k) = 0, so theta vanishes at t = K(k), the quarter period.
        for theta0 in [0.2_f64, 1.0, -1.0, 2.5] {
            let quarter = complete_k((theta0 / 2.0).cos());
            let got = pendulum_closed_form(theta0, quarter);
            assert!(got.abs() < 1e-12, "theta0={theta0} got={got}");
        }
    }

    #[test]
    fn pendulum_closed_form_reaches_the_opposite_turning_point_at_half_period() {
        // sn(-K) = -1, so theta(2K) = -theta0 and theta(4K) = theta0.
        let theta0 = 1.0_f64;
        let quarter = complete_k((theta0 / 2.0).cos());
        let half = pendulum_closed_form(theta0, 2.0 * quarter);
        let full = pendulum_closed_form(theta0, 4.0 * quarter);
        assert!((half + theta0).abs() < 1e-12, "half={half}");
        assert!((full - theta0).abs() < 1e-12, "full={full}");
    }

    #[test]
    fn pendulum_closed_form_is_odd_about_the_quarter_period() {
        // theta(K + s) = -theta(K - s), since theta(K) = 0 and the solution is odd about it.
        let theta0 = 1.0_f64;
        let quarter = complete_k((theta0 / 2.0).cos());
        for s in [0.1_f64, 0.37, 1.2] {
            let before = pendulum_closed_form(theta0, quarter - s);
            let after = pendulum_closed_form(theta0, quarter + s);
            assert!((before + after).abs() < 1e-12, "s={s}");
        }
    }

    #[test]
    fn pendulum_closed_form_is_identically_zero_for_zero_amplitude() {
        for t in [0.0_f64, 1.0, 17.5] {
            assert_eq!(pendulum_closed_form(0.0, t), 0.0);
        }
    }

    #[test]
    fn complete_elliptic_integral_agrees_with_its_hand_derived_series() {
        // m = k^2 = sin^2(0.5) and k' = cos(0.5) for a 1 rad amplitude. Both
        // derivations must agree to machine precision.
        let m = 0.5_f64.sin().powi(2);
        let agm_value = complete_k(0.5_f64.cos());
        let series_value = complete_k_series(m);
        assert!(
            (agm_value - series_value).abs() < 1e-14,
            "agm={agm_value} series={series_value}"
        );
        // Known ratio T / T0 = 4K(m) / (2 pi) = 1.0663 for a 1 rad amplitude.
        let period_ratio = 4.0 * agm_value / (2.0 * PI);
        assert!((period_ratio - 1.0663).abs() < 1e-4, "ratio={period_ratio}");
    }

    #[test]
    fn pendulum_closed_form_matches_an_independent_rk4_integration() {
        // Integrate one full period with 4000 RK4 steps (about 1.9e-3 each) and
        // compare the trajectory to the closed form at the same instants.
        let theta0 = 1.0_f64;
        let period = pendulum_period(theta0);
        let steps = 4000;
        let numeric = rk4_pendulum(theta0, period, steps);
        let times: Vec<f64> = (0..=steps)
            .map(|i| i as f64 * period / steps as f64)
            .collect();
        let closed: Vec<f64> = times
            .iter()
            .map(|&t| pendulum_closed_form(theta0, t))
            .collect();
        let tol = Tolerance {
            abs: 1e-8,
            rel: 0.0,
        };
        let report = compare_series(&numeric, &closed, &tol).expect("equal lengths");
        assert!(
            report.pass,
            "max_error={} at {}",
            report.max_error, report.at_sample
        );
    }

    #[test]
    fn pendulum_closed_form_rejects_inputs_outside_its_domain() {
        assert!(pendulum_closed_form(PI, 0.0).is_nan());
        assert!(pendulum_closed_form(-4.0, 0.0).is_nan());
        assert!(pendulum_closed_form(f64::NAN, 0.0).is_nan());
        assert!(pendulum_closed_form(1.0, f64::INFINITY).is_nan());
        assert!(pendulum_period(PI).is_nan());
        assert!(pendulum_period(-PI).is_nan());
    }

    #[test]
    fn pendulum_closed_form_matches_an_independent_rk4_integration_at_three_radians() {
        // Hand derivation: theta0 = 3.0 gives k' = cos(1.5) = 0.07074, so the
        // gap to pi is 0.1416 and L = ln(4 / k') = 4.03508. The first asymptotic
        // term is 4L = 16.1403. The next term, k'^2 (L - 1), adds 4 * 0.00380 =
        // 0.0152, giving the period 4K = 16.1555 (AGM value 16.15554). The
        // bound below is 0.05, so the check does not depend on the O(k'^4)
        // remainder.
        let theta0 = 3.0_f64;
        let kc = (theta0 / 2.0).cos();
        let period = pendulum_period(theta0);
        assert!(
            (period - 4.0 * (4.0 / kc).ln()).abs() < 0.05,
            "period={period}"
        );
        // RK4 with 4000 steps over the period (dt = 4e-3). Its global error is
        // of order dt^4 times the solution's fourth derivative, about 1e-9 here,
        // so the 1e-6 bound leaves about three orders of margin.
        let steps = 4000;
        let numeric = rk4_pendulum(theta0, period, steps);
        let closed: Vec<f64> = (0..=steps)
            .map(|i| pendulum_closed_form(theta0, i as f64 * period / steps as f64))
            .collect();
        let tol = Tolerance {
            abs: 1e-6,
            rel: 0.0,
        };
        let report = compare_series(&numeric, &closed, &tol).expect("equal lengths");
        assert!(
            report.pass,
            "max_error={} at {}",
            report.max_error, report.at_sample
        );
    }

    #[test]
    fn pendulum_period_near_the_separatrix_matches_its_logarithmic_asymptote() {
        // Exact expansion of K in the complementary modulus:
        // K = L + (k'^2 / 4)(L - 1) + O(k'^4 L), with L = ln(4 / k') and k' = cos(theta0/2).
        // For gap = pi - |theta0| the period is 4K. The rounding of theta0 moves
        // k' by a relative eps / gap, which moves 4K by about 4 eps / gap. The
        // bound 16 eps / gap is generous over that, plus 1e-9 for the omitted
        // O(k'^4 L) terms. Measured agreement is below 3e-13 for gap in {1e-3,
        // 1e-6, 1e-9}, so the bound is not tight.
        for gap in [1e-3_f64, 1e-6, 1e-9] {
            let theta0 = PI - gap;
            let kc = (theta0 / 2.0).cos();
            let l = (4.0 / kc).ln();
            let expected = 4.0 * (l + 0.25 * kc * kc * (l - 1.0));
            let got = pendulum_period(theta0);
            let tol = 1e-9 + 16.0 * f64::EPSILON / gap;
            assert!(
                (got - expected).abs() < tol,
                "gap={gap} got={got} expected={expected} tol={tol}"
            );
        }
    }

    #[test]
    fn pendulum_closed_form_is_finite_at_the_largest_admissible_angle() {
        // The largest f64 below pi is admissible. Its complementary modulus is
        // positive (about 2.8e-16), so the Landen recursion stays finite and the
        // release angle is recovered at t = 0.
        let theta0 = f64::from_bits(PI.to_bits() - 1);
        assert!(theta0 < PI);
        let period = pendulum_period(theta0);
        assert!(
            period.is_finite() && period > 100.0 && period < 160.0,
            "period={period}"
        );
        let got = pendulum_closed_form(theta0, 0.0);
        assert!((got - theta0).abs() < 1e-9, "got={got}");
    }

    #[test]
    fn energy_drift_guards_only_exact_zero_and_non_finite_energy() {
        // Spec section 2.2: no floor constant. Only exact zero and non-finite
        // E(0) are guarded, so a near-zero E(0) gives its true relative drift.
        assert!((energy_drift(&[1e-300, 2e-300]) - 1.0).abs() < 1e-12);
        assert_eq!(energy_drift(&[5e-324, 5e-324]), 0.0);
        // The denominator is |E(0)|, so a negative start gives a positive drift.
        assert!((energy_drift(&[-2.0, -2.2]) - 0.1).abs() < 1e-12);
        assert!(energy_drift(&[f64::INFINITY, 1.0]).is_nan());
    }

    #[test]
    fn energy_drift_is_zero_for_a_conserved_series() {
        assert_eq!(energy_drift(&[2.0, 2.0, 2.0]), 0.0);
    }

    #[test]
    fn energy_drift_reports_the_largest_relative_excursion() {
        // Relative to E_0 = 4: deviations 0.04 and 0.02 at samples 1 and 2.
        let drift = energy_drift(&[4.0, 4.16, 3.92]);
        assert!((drift - 0.04).abs() < 1e-12, "drift={drift}");
    }

    #[test]
    fn energy_drift_is_nan_when_undefined() {
        assert!(energy_drift(&[]).is_nan());
        assert!(energy_drift(&[0.0, 1.0]).is_nan());
        assert!(energy_drift(&[1.0, f64::NAN]).is_nan());
    }

    #[test]
    fn compare_series_passes_identical_series_with_zero_error() {
        let tol = Tolerance { abs: 0.0, rel: 0.0 };
        let report = compare_series(&[1.0, 2.0], &[1.0, 2.0], &tol).expect("equal lengths");
        assert_eq!(report.max_error, 0.0);
        assert_eq!(report.at_sample, 0);
        assert!(report.pass);
    }

    #[test]
    fn compare_series_rejects_a_length_mismatch_instead_of_truncating() {
        let tol = Tolerance { abs: 1.0, rel: 0.0 };
        let err = compare_series(&[1.0, 2.0, 3.0], &[1.0, 2.0], &tol).unwrap_err();
        assert_eq!(
            err,
            OracleError::LengthMismatch {
                reference: 3,
                candidate: 2
            }
        );
        let err = compare_series(&[1.0], &[1.0, 2.0], &tol).unwrap_err();
        assert!(matches!(err, OracleError::LengthMismatch { .. }));
    }

    #[test]
    fn compare_series_rejects_empty_and_invalid_tolerance() {
        let tol = Tolerance { abs: 1.0, rel: 0.0 };
        assert_eq!(
            compare_series(&[], &[], &tol),
            Err(OracleError::EmptySeries)
        );
        let negative = Tolerance {
            abs: -1.0,
            rel: 0.0,
        };
        assert_eq!(
            compare_series(&[1.0], &[1.0], &negative),
            Err(OracleError::InvalidTolerance)
        );
        let nan = Tolerance {
            abs: f64::NAN,
            rel: 0.0,
        };
        assert_eq!(
            compare_series(&[1.0], &[1.0], &nan),
            Err(OracleError::InvalidTolerance)
        );
    }

    #[test]
    fn compare_series_reports_the_first_sample_over_tolerance() {
        let tol = Tolerance {
            abs: 0.05,
            rel: 0.0,
        };
        let reference = [0.0, 0.0, 0.0, 0.0];
        let candidate = [0.01, 0.02, 0.2, 0.3];
        let report = compare_series(&reference, &candidate, &tol).expect("equal lengths");
        assert!(!report.pass);
        assert!((report.max_error - 0.3).abs() < 1e-12);
        assert_eq!(report.at_sample, 3);
    }

    #[test]
    fn compare_series_applies_the_relative_part_of_the_tolerance() {
        // Sample 1: error 0.4 against |ref| = 10 is within rel 0.05 (allowed 0.5).
        let tol = Tolerance {
            abs: 0.0,
            rel: 0.05,
        };
        let passing = compare_series(&[1.0, 10.0], &[1.0, 9.6], &tol).expect("equal lengths");
        assert!(passing.pass);
        assert!((passing.max_error - 0.4).abs() < 1e-12);
        assert_eq!(passing.at_sample, 1);
        let failing = compare_series(&[1.0, 10.0], &[1.0, 9.4], &tol).expect("equal lengths");
        assert!(!failing.pass);
    }

    #[test]
    fn compare_series_fails_a_non_finite_candidate() {
        let tol = Tolerance { abs: 1.0, rel: 0.0 };
        let report = compare_series(&[0.0, 0.0], &[0.0, f64::NAN], &tol).expect("equal lengths");
        assert!(!report.pass);
        assert_eq!(report.max_error, f64::INFINITY);
        assert_eq!(report.at_sample, 1);
    }
}

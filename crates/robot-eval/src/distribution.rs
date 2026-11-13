//! Student's t distribution, for the small-seed confidence intervals and
//! tests SPECS.md Section 14 asks for ("multiple seeds and confidence
//! intervals for stochastic outcomes").
//!
//! With five or ten seeds the normal approximation's 1.96 understates a 95 %
//! interval by up to 40 %; the t quantile does not. Authored against `std`
//! (`docs/decisions/0003-std-only-default-closure.md`): the log-gamma
//! function by Lanczos' approximation (g = 7, nine coefficients, relative
//! error near 1e-15), the regularized incomplete beta function by its
//! continued fraction (modified Lentz), and the quantile by bisection on
//! the CDF.

/// `ln Gamma(x)` for `x > 0`.
#[must_use]
pub fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection.
        let pi = core::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + G + 0.5;
    for (i, c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * core::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// The regularized incomplete beta function `I_x(a, b)`.
#[must_use]
pub fn incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let front =
        (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_fraction(x, a, b) / a
    } else {
        1.0 - front * beta_fraction(1.0 - x, b, a) / b
    }
}

fn beta_fraction(x: f64, a: f64, b: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=300 {
        let m = f64::from(m);
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        d = if d.abs() < TINY { TINY } else { d };
        c = 1.0 + aa / c;
        c = if c.abs() < TINY { TINY } else { c };
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        d = if d.abs() < TINY { TINY } else { d };
        c = 1.0 + aa / c;
        c = if c.abs() < TINY { TINY } else { c };
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 1e-15 {
            break;
        }
    }
    h
}

/// The CDF of Student's t with `df` degrees of freedom.
#[must_use]
pub fn t_cdf(t: f64, df: f64) -> f64 {
    if !t.is_finite() {
        return if t > 0.0 { 1.0 } else { 0.0 };
    }
    let x = df / (df + t * t);
    let tail = 0.5 * incomplete_beta(x, 0.5 * df, 0.5);
    if t >= 0.0 {
        1.0 - tail
    } else {
        tail
    }
}

/// The `p` quantile of Student's t with `df` degrees of freedom.
#[must_use]
pub fn t_quantile(p: f64, df: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let (mut lo, mut hi) = (-1e3, 1e3);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if t_cdf(mid, df) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ln_gamma_matches_factorials() {
        for (n, fact) in [(1.0, 1.0f64), (5.0, 24.0), (11.0, 3_628_800.0)] {
            assert!((ln_gamma(n) - fact.ln()).abs() < 1e-12, "{n}");
        }
        assert!((ln_gamma(0.5) - core::f64::consts::PI.sqrt().ln()).abs() < 1e-12);
    }

    #[test]
    fn t_quantiles_match_published_tables() {
        // Two-sided 95 % critical values.
        for (df, t) in [(1.0, 12.706), (4.0, 2.776), (9.0, 2.262), (30.0, 2.042)] {
            assert!((t_quantile(0.975, df) - t).abs() < 1e-3, "df {df}");
        }
        assert!((t_cdf(0.0, 7.0) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn many_degrees_of_freedom_approach_the_normal() {
        assert!((t_quantile(0.975, 1e6) - 1.959_964).abs() < 1e-4);
    }
}

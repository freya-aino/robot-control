//! Time integrators ported from MuJoCo 3.17.0 (commit `800a109a`).
//!
//! Pure algorithm module: `std` and plain `f64` slices only, no `robot_types`,
//! `robot_model` or `HashMap` types (FR-6). Each function names the pinned
//! MuJoCo source function it transfers. The scope is scalar (hinge/slide)
//! joints: quaternion `Free`/`Ball` joints need MuJoCo's `mj_integratePos`
//! and are not ported here. Deviations are listed in
//! `crates/robot-sim/specs/03-integrators.md`.

/// Semi-implicit (symplectic) Euler step for scalar joints, in place.
///
/// Source: MuJoCo 3.17.0 `src/engine/engine_forward.c`, `mj_advance`
/// (`qvel += h*qacc` first, then the position is integrated with the updated
/// `qvel`). For scalar joints `mj_integratePos` reduces to `qpos += h*qvel`.
///
/// # Panics
///
/// Panics if the three slices do not have the same length.
pub fn semi_implicit_euler_step(q: &mut [f64], qd: &mut [f64], qacc: &[f64], dt: f64) {
    assert_eq!(q.len(), qd.len(), "q and qd widths differ");
    assert_eq!(qd.len(), qacc.len(), "qd and qacc widths differ");
    for (v, a) in qd.iter_mut().zip(qacc) {
        *v += dt * a;
    }
    for (x, v) in q.iter_mut().zip(qd.iter()) {
        *x += dt * v;
    }
}

/// Implicit-in-velocity Euler step with diagonal joint damping, in place.
///
/// Source: MuJoCo 3.17.0 `src/engine/engine_forward.c`, `mj_EulerSkip`, damping
/// branch: `qH = M`, `qH[i,i] += h * B_i` (the damping derivative, see
/// `mjd_xPolyForce` in `src/engine/engine_util_misc.c`; for a linear damper it is
/// the damping coefficient), the solve `(M + h*B) * qacc = qfrc_smooth +
/// qfrc_constraint`, then `mj_advance`.
///
/// `mass` is the dense row-major `n x n` inertia `M`, `damping` the diagonal
/// `B`, and `bias` is `qfrc_smooth + qfrc_constraint`. As in MuJoCo, `bias`
/// already contains the explicit damper force `-B * qd`, since `qfrc_smooth`
/// includes `qfrc_passive`.
///
/// Input contract, checked before any state is touched:
/// - `dt` must be finite and non-negative;
/// - every `damping` entry must be finite and non-negative;
/// - every `bias` and `mass` entry must be finite;
/// - `mass` must be symmetric to a relative tolerance of `1e-12 * max|M|`. Only
///   the lower triangle is read, so an asymmetric `M` would be silently
///   misread; it is refused instead.
///
/// Returns `None` when any check fails or when `M + h*diag(B)` is not
/// symmetric positive definite; the state is then left unchanged. MuJoCo
/// instead clamps the pivot and emits a warning (`mj_factorI`); this port
/// reports it instead.
///
/// # Panics
///
/// Panics if the slice widths do not match `qd.len()`.
pub fn implicit_damping_euler_step(
    q: &mut [f64],
    qd: &mut [f64],
    mass: &[f64],
    damping: &[f64],
    bias: &[f64],
    dt: f64,
) -> Option<()> {
    let n = qd.len();
    assert_eq!(q.len(), n, "q and qd widths differ");
    assert_eq!(mass.len(), n * n, "mass must be n x n");
    assert_eq!(damping.len(), n, "damping width differs");
    assert_eq!(bias.len(), n, "bias width differs");
    if !dt.is_finite() || dt < 0.0 {
        return None;
    }
    if damping.iter().any(|b| !b.is_finite() || *b < 0.0) {
        return None;
    }
    if bias.iter().any(|b| !b.is_finite()) || mass.iter().any(|m| !m.is_finite()) {
        return None;
    }
    let scale = mass.iter().fold(0.0f64, |acc, m| acc.max(m.abs()));
    for i in 0..n {
        for j in (i + 1)..n {
            if (mass[i * n + j] - mass[j * n + i]).abs() > 1e-12 * scale {
                return None;
            }
        }
    }
    let mut system = mass.to_vec();
    for (i, b) in damping.iter().enumerate() {
        system[i * n + i] += dt * b;
    }
    let mut qacc = bias.to_vec();
    if !cholesky_solve(&mut system, &mut qacc, n) {
        return None;
    }
    semi_implicit_euler_step(q, qd, &qacc, dt);
    Some(())
}

/// Classical fourth-order Runge-Kutta step for scalar joints, in place.
///
/// Source: MuJoCo 3.17.0 `src/engine/engine_forward.c`, `mj_RungeKutta` with
/// `N = 4`, tableau `RK4_A` (`0.5, 0.5, 1`) and `RK4_B` (`1/6, 1/3, 1/3, 1/6`).
/// Stage positions advance by `h` times the stage velocities, and stage
/// velocities by `h` times the stage accelerations `F`, both weighted by `A`
/// (`mj_integratePos` and `mju_addToScl` in MuJoCo). The final update uses `B`.
/// `accel(q, qd)` returns the accelerations at an arbitrary state, as
/// `mj_forwardSkip` does per stage in MuJoCo. Stage times are not used,
/// because no time-dependent input is modelled here.
///
/// # Panics
///
/// Panics if `q` and `qd` differ in length, or if `accel` returns a different
/// length from `qd`.
pub fn rk4_step<F>(q: &mut [f64], qd: &mut [f64], dt: f64, mut accel: F)
where
    F: FnMut(&[f64], &[f64]) -> Vec<f64>,
{
    assert_eq!(q.len(), qd.len(), "q and qd widths differ");
    let half = 0.5 * dt;

    let a1 = evaluate_accel(&mut accel, q, qd);
    let q2 = offset(q, qd, half);
    let v2 = offset(qd, &a1, half);
    let a2 = evaluate_accel(&mut accel, &q2, &v2);
    let q3 = offset(q, &v2, half);
    let v3 = offset(qd, &a2, half);
    let a3 = evaluate_accel(&mut accel, &q3, &v3);
    let q4 = offset(q, &v3, dt);
    let v4 = offset(qd, &a3, dt);
    let a4 = evaluate_accel(&mut accel, &q4, &v4);

    for (i, x) in q.iter_mut().enumerate() {
        *x += dt / 6.0 * (qd[i] + 2.0 * v2[i] + 2.0 * v3[i] + v4[i]);
    }
    for (i, v) in qd.iter_mut().enumerate() {
        *v += dt / 6.0 * (a1[i] + 2.0 * a2[i] + 2.0 * a3[i] + a4[i]);
    }
}

/// Calls `accel` and checks that it returns one acceleration per velocity.
fn evaluate_accel<F>(accel: &mut F, q: &[f64], qd: &[f64]) -> Vec<f64>
where
    F: FnMut(&[f64], &[f64]) -> Vec<f64>,
{
    let a = accel(q, qd);
    assert_eq!(
        a.len(),
        qd.len(),
        "accel must return one acceleration per velocity"
    );
    a
}

/// `base + scale * dir`, element-wise.
fn offset(base: &[f64], dir: &[f64], scale: f64) -> Vec<f64> {
    base.iter().zip(dir).map(|(b, d)| b + scale * d).collect()
}

/// Solves `a * x = b` for a symmetric `n x n` matrix `a` (row-major) by
/// Cholesky factorisation, writing `x` into `b`. Returns `false` when a pivot
/// is not finite and positive; `b` is then unchanged, but `a` is partially
/// overwritten and must be rebuilt by the caller.
pub(crate) fn cholesky_solve(a: &mut [f64], b: &mut [f64], n: usize) -> bool {
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        if !d.is_finite() || d <= 0.0 {
            return false;
        }
        let l_jj = d.sqrt();
        a[j * n + j] = l_jj;
        for i in (j + 1)..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s / l_jj;
        }
    }
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i * n + k] * b[k];
        }
        b[i] = s / a[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in (i + 1)..n {
            s -= a[k * n + i] * b[k];
        }
        b[i] = s / a[i * n + i];
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semi_implicit_euler_uses_the_updated_velocity_for_position() {
        // Hand derivation: qd1 = qd0 + h*qacc = 1 + 0.1*2 = 1.2;
        // q1 = q0 + h*qd1 = 0 + 0.1*1.2 = 0.12. Explicit Euler with the old
        // velocity would give q1 = 0.1, so 0.12 identifies mj_advance's ordering.
        let mut q = [0.0];
        let mut qd = [1.0];
        semi_implicit_euler_step(&mut q, &mut qd, &[2.0], 0.1);
        assert!((qd[0] - 1.2).abs() < 1e-15);
        assert!((q[0] - 0.12).abs() < 1e-15);
    }

    #[test]
    fn implicit_damping_matches_the_discrete_identity_for_one_dof() {
        // Hand derivation: M = 2, B = 1, qd0 = 1, no applied force. The explicit
        // damper term -B*qd0 = -1 is part of the bias. Then (M + h*B) * qacc = bias
        // gives qacc = -1 / (2 + 0.1) = -1/2.1, so qd1 = 1 - 0.1/2.1 = 2/2.1,
        // which is the implicit-Euler identity M*qd0/(M + h*B). The explicit
        // answer qd1 = 1 + 0.1*(-1/2) = 0.95 must not come out.
        let mut q = [0.0];
        let mut qd = [1.0];
        let r = implicit_damping_euler_step(&mut q, &mut qd, &[2.0], &[1.0], &[-1.0], 0.1);
        assert_eq!(r, Some(()));
        assert!((qd[0] - 2.0 / 2.1).abs() < 1e-14, "qd1 = {}", qd[0]);
        assert!((q[0] - 0.1 * 2.0 / 2.1).abs() < 1e-14, "q1 = {}", q[0]);
        assert!((qd[0] - 0.95).abs() > 1e-3);
    }

    #[test]
    fn implicit_solve_handles_off_diagonal_inertia() {
        // Hand derivation: M = [[2,1],[1,2]], B = 0, so M*qacc = bias. With
        // bias = [3,3]: 2a + b = 3 and a + 2b = 3 give a = b = 1. With qd0 = 0 and
        // h = 0.5: qd1 = 0.5*[1,1] = [0.5,0.5] and q1 = 0.5*qd1 = [0.25,0.25].
        let mut q = [0.0, 0.0];
        let mut qd = [0.0, 0.0];
        let r = implicit_damping_euler_step(
            &mut q,
            &mut qd,
            &[2.0, 1.0, 1.0, 2.0],
            &[0.0, 0.0],
            &[3.0, 3.0],
            0.5,
        );
        assert_eq!(r, Some(()));
        assert!((qd[0] - 0.5).abs() < 1e-14 && (qd[1] - 0.5).abs() < 1e-14);
        assert!((q[0] - 0.25).abs() < 1e-14 && (q[1] - 0.25).abs() < 1e-14);
    }

    #[test]
    fn implicit_step_rejects_a_singular_system_without_changing_state() {
        // Hand derivation: M + h*B = 0 + 0.1*0 = 0, so the Cholesky pivot is 0
        // and the step is refused. The state must be unchanged.
        let mut q = [0.3];
        let mut qd = [0.7];
        let r = implicit_damping_euler_step(&mut q, &mut qd, &[0.0], &[0.0], &[1.0], 0.1);
        assert_eq!(r, None);
        assert_eq!(q, [0.3]);
        assert_eq!(qd, [0.7]);
    }

    #[test]
    fn rk4_reproduces_the_taylor_polynomial_of_the_harmonic_oscillator() {
        // Hand derivation: (q, v)' = (v, -q) is linear with A = [[0,1],[-1,0]].
        // The RK4 stage combination is exactly the degree-4 Taylor polynomial of
        // exp(hA). With A^2 = -I, A^3 = -A and A^4 = I, starting from (q0, v0) = (1, 0):
        //   q1 = 1 - h^2/2 + h^4/24,  v1 = -h + h^3/6.
        // With h = 0.1: q1 = 1 - 0.005 + 0.0000041666... = 0.9950041666...
        //               v1 = -0.1 + 0.0001666...       = -0.0998333333...
        let mut q = [1.0];
        let mut qd = [0.0];
        rk4_step(&mut q, &mut qd, 0.1, |q, _| vec![-q[0]]);
        assert!((q[0] - 0.995_004_166_666_7).abs() < 1e-12, "q1 = {}", q[0]);
        assert!(
            (qd[0] + 0.099_833_333_333_3).abs() < 1e-12,
            "v1 = {}",
            qd[0]
        );
    }

    #[test]
    fn stiff_damping_stays_bounded_where_explicit_euler_diverges() {
        // Hand derivation: M = 1, B = 1000, qd0 = 1, h = 0.1, bias = -B*qd0 = -1000.
        // System: (M + h*B) * qacc = bias, so qacc = -1000/101 and
        // qd1 = 1 + 0.1*(-1000/101) = 1/101 = 0.009900990...
        // The closed form is qd1 = M*qd0/(M + h*B), which stays in (0, 1).
        // The explicit step with the same damper would give qd1 = 1 - h*B = -99,
        // which grows in magnitude every step; the implicit step must not.
        let mut q = [0.0];
        let mut qd = [1.0];
        let r = implicit_damping_euler_step(&mut q, &mut qd, &[1.0], &[1000.0], &[-1000.0], 0.1);
        assert_eq!(r, Some(()));
        assert!((qd[0] - 1.0 / 101.0).abs() < 1e-14, "qd1 = {}", qd[0]);
        assert!((q[0] - 0.1 / 101.0).abs() < 1e-14, "q1 = {}", q[0]);
        assert!(qd[0].abs() < 1.0);
    }

    #[test]
    fn implicit_step_refuses_non_finite_or_negative_inputs_without_changing_state() {
        // Each case is refused before the solve, and the state must be unchanged.
        type Case<'a> = (&'a [f64], &'a [f64], &'a [f64], f64);
        let cases: [Case; 8] = [
            (&[2.0], &[1.0], &[-1.0], f64::NAN),
            (&[2.0], &[1.0], &[-1.0], f64::INFINITY),
            (&[2.0], &[1.0], &[-1.0], -0.1),
            (&[2.0], &[f64::INFINITY], &[-1.0], 0.1),
            (&[2.0], &[1.0], &[f64::NAN], 0.1),
            (&[f64::NAN], &[1.0], &[-1.0], 0.1),
            (&[2.0], &[-1.0], &[-1.0], 0.1),
            (&[2.0], &[f64::NAN], &[-1.0], 0.1),
        ];
        for (mass, damping, bias, dt) in cases {
            let mut q = [0.3];
            let mut qd = [0.7];
            let r = implicit_damping_euler_step(&mut q, &mut qd, mass, damping, bias, dt);
            assert_eq!(
                r, None,
                "mass={mass:?} damping={damping:?} bias={bias:?} dt={dt}"
            );
            assert_eq!(q, [0.3]);
            assert_eq!(qd, [0.7]);
        }
    }

    #[test]
    fn implicit_step_refuses_an_asymmetric_mass_matrix() {
        // Only the lower triangle is read by the Cholesky factor, so M = [[2,1],[0,2]]
        // would be silently treated as [[2,0],[0,2]]. It must be refused instead.
        let mut q = [0.0, 0.0];
        let mut qd = [0.0, 0.0];
        let r = implicit_damping_euler_step(
            &mut q,
            &mut qd,
            &[2.0, 1.0, 0.0, 2.0],
            &[0.0, 0.0],
            &[3.0, 3.0],
            0.5,
        );
        assert_eq!(r, None);
        assert_eq!(q, [0.0, 0.0]);
        assert_eq!(qd, [0.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "accel must return one acceleration per velocity")]
    fn rk4_panics_when_accel_returns_a_longer_vector() {
        let mut q = [1.0];
        let mut qd = [0.0];
        rk4_step(&mut q, &mut qd, 0.1, |q, _| vec![-q[0], 0.0]);
    }

    #[test]
    #[should_panic(expected = "accel must return one acceleration per velocity")]
    fn rk4_panics_when_accel_returns_a_shorter_vector() {
        let mut q = [1.0, 0.0];
        let mut qd = [0.0, 0.0];
        rk4_step(&mut q, &mut qd, 0.1, |_, _| vec![-1.0]);
    }

    #[test]
    #[should_panic(expected = "accel must return one acceleration per velocity")]
    fn rk4_checks_accel_width_on_later_stages_too() {
        let mut calls = 0;
        let mut q = [1.0];
        let mut qd = [0.0];
        rk4_step(&mut q, &mut qd, 0.1, |q, _| {
            calls += 1;
            if calls == 3 {
                vec![-q[0], 0.0]
            } else {
                vec![-q[0]]
            }
        });
    }
}

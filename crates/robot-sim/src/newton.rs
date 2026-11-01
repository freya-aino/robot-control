//! The primal constraint solver of `specs/08-solvers-islands.md` section 6:
//! Newton's method on the acceleration, with an exact line search.
//!
//! # Why a second solver
//!
//! [`crate::solver::solve`] is projected Gauss-Seidel on the **dual** problem,
//! whose variable is the constraint force. That is one variable per constraint
//! row, and the pyramidal contact representation of 07 section 9.2 makes the
//! rows redundant on purpose: four non-negative coefficients encode three force
//! components, and every edge of a contact contains the normal direction. Four
//! contacts on a six-degree-of-freedom body give sixteen rows for a rank-six
//! problem, and Gauss-Seidel needs **1011 sweeps** there to bring the per-sweep
//! force change below 1e-10, against a default budget of 100 at which the
//! residual is still 1.1e-2. `specs/appendix-d-mathematical-challenges.md`
//! section 1.7.3 records the measurement.
//!
//! The primal problem has no such redundancy. Its variable is the acceleration,
//! one per degree of freedom however many contacts there are, and its Hessian
//! `M + J' D_q J` is positive definite because `M` is. Sixteen rows over six
//! degrees of freedom is a six-variable problem, not a sixteen-variable one.
//!
//! # The problem (section 6.1)
//!
//! `min_a  0.5 (a - a0)' M (a - a0) + s(J a - a_ref)`
//!
//! with `a0 = qacc_smooth` and `s` the piecewise-quadratic constraint cost of
//! section 2. For the row kinds this crate builds, `s` is a hinge: an equality
//! row always contributes `0.5 D jar^2`, and a limit, frictionless-contact or
//! pyramid-edge row contributes `0.5 D jar^2` only while `jar < 0`, with
//! `force = -D jar` and zero otherwise.
//!
//! That hinge is where the inequality goes. A row's force comes out
//! non-negative because `jar < 0` is exactly the condition under which the row
//! contributes at all -- there is no projection step, and nothing to get wrong
//! about a cone. The friction cone is already in the row Jacobians.
//!
//! # Divergence from the example, stated
//!
//! Section 6.8's line search is an iterative one: a Newton march along the
//! direction to bracket the minimum, then bisection against a tolerance. The
//! cost along a search direction is a convex piecewise quadratic in `alpha`
//! whose breakpoints are the `alpha` at which each inequality row's `jar`
//! crosses zero, so this implementation finds the minimiser **exactly**, by
//! sorting the breakpoints and minimising the quadratic on each segment. It
//! computes what section 6.8 approaches, so the mathematics is the example's
//! and the iteration sequence is not. The cost is a sort per line search, which
//! is nothing beside the Hessian factorisation.
//!
//! Not implemented: elliptic cones (this crate builds pyramidal rows only),
//! friction-loss rows, islands, and the sparse Hessian. The Hessian here is
//! dense `nv * nv` and refactorised each iteration rather than updated rank-1
//! as `HessianIncremental` does; that is a cost, not a difference of result,
//! and it is the thing to change first if this becomes a bottleneck.

use crate::composite::{crb, factor_m, full_m, solve_ld, CompositeError, CompositeModel, Position};
use crate::solver::{row_impedance, Row, RowKind, Solution, SolverConfig};

/// Smallest value the example admits before treating a quantity as zero.
const MINVAL: f64 = 1e-15;

/// Per-row state of the piecewise-quadratic cost, section 2.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RowState {
    /// The row contributes its quadratic.
    Quadratic,
    /// The row is inactive and contributes nothing.
    Satisfied,
}

/// Whether a row is an equality, which contributes its quadratic always.
fn is_equality(kind: RowKind) -> bool {
    matches!(kind, RowKind::Equality)
}

/// `mj_constraintUpdate_impl` of section 2, restricted to the row kinds this
/// crate builds: equality rows, and inequality rows with a hinge at zero.
///
/// Returns the constraint cost `s`, and fills `force` and `state`.
fn constraint_update(
    rows: &[Row],
    d: &[f64],
    jar: &[f64],
    force: &mut [f64],
    state: &mut [RowState],
) -> f64 {
    let mut cost = 0.0;
    for i in 0..rows.len() {
        if is_equality(rows[i].kind) {
            force[i] = -d[i] * jar[i];
            cost += 0.5 * d[i] * jar[i] * jar[i];
            state[i] = RowState::Quadratic;
        } else if jar[i] >= 0.0 {
            // "if jar >= 0: force=0; state=SATISFIED".
            force[i] = 0.0;
            state[i] = RowState::Satisfied;
        } else {
            force[i] = -d[i] * jar[i];
            cost += 0.5 * d[i] * jar[i] * jar[i];
            state[i] = RowState::Quadratic;
        }
    }
    cost
}

/// `mju_cholFactor` of 08 section 7.1: dense lower-triangular Cholesky in
/// place, `A = L L'`, reading and writing the lower triangle only.
///
/// Returns the rank. A pivot below `mindiag` is **clamped to `mindiag`** and
/// that column below the diagonal is zeroed, exactly as the example does,
/// rather than reported as a failure. That matters here and is not defensive
/// coding: `D = 1/R` reaches 1e15 when a row's regularisation is at its floor,
/// so `M + J' D_q J` can lose a pivot to cancellation on a 35-degree-of-freedom
/// G1 even though it is positive definite in exact arithmetic. An earlier
/// version of this function returned an error there and the G1 stance run died
/// at degree of freedom 31. Clamping factorises a nearby positive-definite
/// matrix instead, which still gives a descent direction; if the direction is
/// useless the line search returns a zero step and the iteration stops.
fn chol_factor(a: &mut [f64], n: usize, mindiag: f64) -> usize {
    let mut rank = n;
    for j in 0..n {
        let mut tmp = a[j * n + j];
        for k in 0..j {
            tmp -= a[j * n + k] * a[j * n + k];
        }
        if tmp < mindiag {
            rank -= 1;
            a[j * n + j] = mindiag.sqrt();
            for i in (j + 1)..n {
                a[i * n + j] = 0.0;
            }
        } else {
            a[j * n + j] = tmp.sqrt();
            let inv = 1.0 / a[j * n + j];
            for i in (j + 1)..n {
                let mut sum = a[i * n + j];
                for k in 0..j {
                    sum -= a[i * n + k] * a[j * n + k];
                }
                a[i * n + j] = sum * inv;
            }
        }
    }
    rank
}

/// Solves `L L' x = b` in place for a factor from [`chol_factor`].
fn chol_solve(l: &[f64], n: usize, x: &mut [f64]) {
    for i in 0..n {
        let mut sum = x[i];
        for k in 0..i {
            sum -= l[i * n + k] * x[k];
        }
        x[i] = sum / l[i * n + i];
    }
    for i in (0..n).rev() {
        let mut sum = x[i];
        for k in (i + 1)..n {
            sum -= l[k * n + i] * x[k];
        }
        x[i] = sum / l[i * n + i];
    }
}

/// The exact minimiser of the cost along `search`, and the improvement it buys.
///
/// The cost is
/// `f(alpha) = qg0 + alpha qg1 + alpha^2 qg2 + sum_i c_i(alpha)`
/// where the Gauss part is a plain quadratic and each row contributes
/// `0.5 D (jar_i + alpha jv_i)^2` while that is negative (or always, for an
/// equality). Each inequality row therefore switches on or off at one
/// breakpoint `alpha_i = -jar_i / jv_i`. Between consecutive breakpoints the
/// active set is fixed and the cost is one quadratic, so the exact minimiser is
/// the best of the per-segment minimisers.
fn exact_line_search(
    rows: &[Row],
    d: &[f64],
    jar: &[f64],
    jv: &[f64],
    quad_gauss: [f64; 3],
) -> (f64, f64) {
    let m = rows.len();
    // Breakpoints, each with the sign of jv so we know which way the row
    // switches as alpha grows.
    let mut breaks: Vec<f64> = Vec::with_capacity(m);
    for i in 0..m {
        if is_equality(rows[i].kind) {
            continue;
        }
        if jv[i].abs() > MINVAL {
            let alpha = -jar[i] / jv[i];
            if alpha.is_finite() {
                breaks.push(alpha);
            }
        }
    }
    breaks.push(0.0);
    breaks.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    breaks.dedup_by(|a, b| (*a - *b).abs() <= MINVAL);

    // The quadratic coefficients of the active set on the segment containing
    // `probe`, plus the cost at alpha = 0 of the rows that are active there
    // (needed because the cost is continuous: each row's quadratic is its own
    // and we sum whichever are on).
    let coeffs_at = |probe: f64| -> [f64; 3] {
        let mut c = quad_gauss;
        for i in 0..m {
            let active = is_equality(rows[i].kind) || (jar[i] + probe * jv[i]) < 0.0;
            if !active {
                continue;
            }
            // 0.5 D (jar + a jv)^2 = 0.5 D jar^2 + a D jar jv + a^2 0.5 D jv^2
            c[0] += 0.5 * d[i] * jar[i] * jar[i];
            c[1] += d[i] * jar[i] * jv[i];
            c[2] += 0.5 * d[i] * jv[i] * jv[i];
        }
        c
    };

    let eval = |alpha: f64| -> f64 {
        let c = coeffs_at(alpha);
        c[0] + alpha * c[1] + alpha * alpha * c[2]
    };

    let base = eval(0.0);
    let mut best_alpha = 0.0;
    let mut best_cost = base;

    // Each segment is [breaks[k], breaks[k+1]], plus the two open ends. On a
    // segment the active set is constant, so take the midpoint as the probe.
    let n = breaks.len();
    let span = breaks
        .last()
        .copied()
        .unwrap_or(0.0)
        .abs()
        .max(breaks.first().copied().unwrap_or(0.0).abs())
        .max(1.0);
    for k in 0..=n {
        let (lo, hi) = match (k, n) {
            (0, _) => (f64::NEG_INFINITY, breaks[0]),
            (k, n) if k == n => (breaks[n - 1], f64::INFINITY),
            (k, _) => (breaks[k - 1], breaks[k]),
        };
        let probe = match (lo.is_finite(), hi.is_finite()) {
            (true, true) => 0.5 * (lo + hi),
            (false, true) => hi - span,
            (true, false) => lo + span,
            (false, false) => 0.0,
        };
        let c = coeffs_at(probe);
        if c[2] <= MINVAL {
            continue;
        }
        // The unconstrained minimiser of this segment's quadratic, clamped into
        // the segment.
        let mut alpha = -0.5 * c[1] / c[2];
        if alpha < lo {
            alpha = lo;
        }
        if alpha > hi {
            alpha = hi;
        }
        if !alpha.is_finite() {
            continue;
        }
        let cost = eval(alpha);
        if cost < best_cost {
            best_cost = cost;
            best_alpha = alpha;
        }
    }
    (best_alpha, base - best_cost)
}

/// Solves the constraint problem of 07 section 2.3 by the primal Newton method
/// of 08 section 6.
///
/// Arguments and return match [`crate::solver::solve`], so the two are
/// interchangeable at a call site. `warm` is a previous step's **force** vector,
/// used only to decide a starting acceleration: the primal variable is the
/// acceleration, so a force warm start enters as `qacc_smooth + M^-1 J' f`.
///
/// # Errors
///
/// [`CompositeError::WidthMismatch`] or [`CompositeError::NotFinite`] on a bad
/// input, [`CompositeError::SingularInertia`] if the mass matrix or the Hessian
/// cannot be factorised.
#[allow(clippy::too_many_arguments)]
pub fn solve_newton(
    model: &CompositeModel,
    pos: &Position,
    qd: &[f64],
    qacc_smooth: &[f64],
    rows: &[Row],
    timestep: f64,
    cfg: SolverConfig,
    warm: Option<&[f64]>,
) -> Result<Solution, CompositeError> {
    let nv = model.tree.nv;
    for (what, v) in [("qd", qd), ("qacc_smooth", qacc_smooth)] {
        if v.len() != nv {
            return Err(CompositeError::WidthMismatch {
                what,
                expected: nv,
                found: v.len(),
            });
        }
        if v.iter().any(|x| !x.is_finite()) {
            return Err(CompositeError::NotFinite { what });
        }
    }
    if !timestep.is_finite() || timestep <= 0.0 {
        return Err(CompositeError::NotFinite { what: "timestep" });
    }
    for row in rows {
        if row.jacobian.len() != nv {
            return Err(CompositeError::WidthMismatch {
                what: "row jacobian",
                expected: nv,
                found: row.jacobian.len(),
            });
        }
        if !row.pos.is_finite()
            || !row.jdot_v.is_finite()
            || row.jacobian.iter().any(|x| !x.is_finite())
        {
            return Err(CompositeError::NotFinite { what: "row" });
        }
    }

    let m = rows.len();
    let mut solution = Solution {
        force: vec![0.0; m],
        qacc: qacc_smooth.to_vec(),
        iterations: 0,
        residual: 0.0,
        clamped_rows: Vec::new(),
    };
    if m == 0 {
        return Ok(solution);
    }

    let mass = crb(model, pos);
    let (ld, diag_inv) = factor_m(&model.tree, &mass)?;
    let dense_m = full_m(&model.tree, &mass);

    // Only the diagonal of A = J M^-1 J' is needed, for R.
    let mut minv_jt: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut diag_a = vec![0.0; m];
    for (i, row) in rows.iter().enumerate() {
        let mut x = row.jacobian.clone();
        solve_ld(&model.tree, &ld, &diag_inv, &mut x);
        diag_a[i] = row.jacobian.iter().zip(&x).map(|(a, b)| a * b).sum::<f64>();
        minv_jt.push(x);
    }
    let imp = row_impedance(rows, qd, timestep, &diag_a);
    solution.clamped_rows = imp.clamped;
    let d = imp.d;

    // a_ref is a reference *acceleration*, and `jar = J a - a_ref` must measure
    // the same thing the dual's `a_u - a_ref` does, so the Jdot v term that
    // enters a_u belongs in the residual here too.
    let aref: Vec<f64> = (0..m).map(|i| imp.aref[i] - rows[i].jdot_v).collect();

    // `scale = 1 / trace(M)` (section 3.2), the gradient's units.
    let trace: f64 = (0..nv).map(|i| dense_m[i * nv + i]).sum();
    let scale = 1.0 / trace.max(MINVAL);

    // Starting acceleration. A force warm start enters through M^-1 J' f.
    let mut a = qacc_smooth.to_vec();
    if cfg.warm_start {
        if let Some(f0) = warm {
            if f0.len() == m {
                for (i, fi) in f0.iter().enumerate() {
                    if *fi != 0.0 {
                        for k in 0..nv {
                            a[k] += fi * minv_jt[i][k];
                        }
                    }
                }
            }
        }
    }

    let jrow = |i: usize| &rows[i].jacobian;
    let mul_j = |x: &[f64]| -> Vec<f64> {
        (0..m)
            .map(|i| jrow(i).iter().zip(x).map(|(p, q)| p * q).sum::<f64>())
            .collect()
    };
    let mul_m_dense = |x: &[f64]| -> Vec<f64> {
        (0..nv)
            .map(|i| (0..nv).map(|k| dense_m[i * nv + k] * x[k]).sum::<f64>())
            .collect()
    };

    let qfrc_smooth = mul_m_dense(qacc_smooth);
    let mut force = vec![0.0; m];
    let mut state = vec![RowState::Satisfied; m];
    let mut ma = mul_m_dense(&a);
    let mut jar: Vec<f64> = mul_j(&a).iter().zip(&aref).map(|(x, r)| x - r).collect();
    constraint_update(rows, &d, &jar, &mut force, &mut state);

    let mut hessian = vec![0.0; nv * nv];
    let mut factor = vec![0.0; nv * nv];
    let mut iterations = 0usize;
    let mut gradient_norm = 0.0;

    for _ in 0..cfg.iterations {
        // grad = M a - qfrc_smooth - J' force.
        let mut grad: Vec<f64> = (0..nv).map(|k| ma[k] - qfrc_smooth[k]).collect();
        for (i, fi) in force.iter().enumerate() {
            if *fi != 0.0 {
                for (g, j) in grad.iter_mut().zip(jrow(i)) {
                    *g -= j * fi;
                }
            }
        }
        gradient_norm = scale * grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if gradient_norm < cfg.tolerance {
            break;
        }

        // H = M + J' diag(D_q) J, D_q zero on satisfied rows (section 6.10).
        hessian.copy_from_slice(&dense_m);
        for i in 0..m {
            if state[i] != RowState::Quadratic {
                continue;
            }
            let ji = jrow(i);
            let di = d[i];
            for r in 0..nv {
                if ji[r] == 0.0 {
                    continue;
                }
                let w = di * ji[r];
                for c in 0..=r {
                    hessian[r * nv + c] += w * ji[c];
                }
            }
        }
        factor.copy_from_slice(&hessian);
        // The factorisation reads and writes the lower triangle only, and
        // clamps a lost pivot rather than failing; see `chol_factor`.
        chol_factor(&mut factor, nv, MINVAL);
        // search = -H^-1 grad.
        let mut search: Vec<f64> = grad.iter().map(|g| -g).collect();
        chol_solve(&factor, nv, &mut search);

        // The Newton decrement, which is the primal certificate of section 6.9.
        let decrement = 0.5 * scale * grad.iter().zip(&search).map(|(g, s)| -g * s).sum::<f64>();
        if decrement.max(0.0) < cfg.tolerance {
            break;
        }

        let mv = mul_m_dense(&search);
        let jv = mul_j(&search);
        let quad_gauss = [
            0.0,
            search.iter().zip(&ma).map(|(v, x)| v * x).sum::<f64>()
                - search
                    .iter()
                    .zip(&qfrc_smooth)
                    .map(|(v, x)| v * x)
                    .sum::<f64>(),
            0.5 * search.iter().zip(&mv).map(|(v, x)| v * x).sum::<f64>(),
        ];
        let (alpha, improvement) = exact_line_search(rows, &d, &jar, &jv, quad_gauss);
        if alpha == 0.0 {
            break;
        }

        // Incremental, never recomputed from scratch (section 6.9).
        for k in 0..nv {
            a[k] += alpha * search[k];
            ma[k] += alpha * mv[k];
        }
        for i in 0..m {
            jar[i] += alpha * jv[i];
        }
        constraint_update(rows, &d, &jar, &mut force, &mut state);
        iterations += 1;
        if improvement > 0.0 && scale * improvement < cfg.tolerance {
            break;
        }
    }

    solution.qacc = a;
    solution.force = force;
    solution.iterations = iterations;
    solution.residual = gradient_norm;
    Ok(solution)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composite::{com_vel, forward_dynamics, position, CompositeState};
    use crate::solver::{contact_rows, default_solimp, default_solref, solve};
    use robot_model::joint::{JointKind, JointLimits, JointSpec};
    use robot_model::link::{InertiaTensor, LinkSpec, MassProperties};
    use robot_model::pose::RigidPose;
    use robot_model::profile::{FirmwareContract, RobotProfile, PROFILE_SCHEMA_VERSION};
    use robot_types::frames::{GravityConvention, Vec3};
    use robot_types::ids::{JointId, LinkId, Name};
    use robot_types::schema::ExtensionTable;
    use robot_types::time::Duration;
    use robot_types::units::{Kilograms, NewtonMetres, RadiansPerSecond};

    /// A single free body of `mass`, the smallest model that can carry a
    /// contact. Same fixture the Gauss-Seidel tests use, so the two solvers
    /// are compared on identical input.
    fn free_body(mass: f64) -> RobotProfile {
        RobotProfile {
            schema_version: PROFILE_SCHEMA_VERSION,
            name: Name::new("free").unwrap(),
            profile_version: "test".to_owned(),
            joints: vec![JointSpec {
                id: JointId::new(0),
                name: Name::new("floating").unwrap(),
                kind: JointKind::Free,
                parent: LinkId::new(0),
                child: LinkId::new(1),
                origin: RigidPose::IDENTITY,
                axis: Vec3::unit_z(),
                limits: JointLimits::unbounded(RadiansPerSecond(1e6), NewtonMetres(1e6)),
                sign: 1.0,
                zero_offset: 0.0,
            }],
            links: vec![
                LinkSpec {
                    id: LinkId::new(0),
                    name: Name::new("world").unwrap(),
                    mass: MassProperties::MASSLESS,
                    collision: Vec::new(),
                },
                LinkSpec {
                    id: LinkId::new(1),
                    name: Name::new("body").unwrap(),
                    mass: MassProperties {
                        mass: Kilograms(mass),
                        center_of_mass: Vec3::ZERO,
                        inertia: InertiaTensor {
                            ixx: 0.02,
                            iyy: 0.02,
                            izz: 0.02,
                            ixy: 0.0,
                            ixz: 0.0,
                            iyz: 0.0,
                        },
                    },
                    collision: Vec::new(),
                },
            ],
            actuators: Vec::new(),
            sensors: Vec::new(),
            root_link: LinkId::new(0),
            gravity: GravityConvention::minus_z_standard(robot_types::frames::FrameId::WORLD),
            firmware: FirmwareContract {
                version: "test".to_owned(),
                command_expiration: Duration::ZERO,
                watchdog_timeout: Duration::ZERO,
                has_independent_estop: false,
                allowed_modes: Vec::new(),
            },
            extensions: ExtensionTable::new(),
        }
    }

    /// The four-contact, sixteen-row pyramidal problem of
    /// `solver::tests::warm_starting_reaches_the_same_answer_in_fewer_sweeps`.
    fn four_contact_problem() -> (CompositeModel, CompositeState, Vec<Row>, Vec<f64>, Position) {
        let profile = free_body(2.0);
        let model = CompositeModel::from_profile(&profile);
        let mut state = CompositeState::identity(&model);
        state.q[2] = -0.002;
        let pos = position(&model, &state.q).unwrap();
        let (cvel, cdof_dot) = com_vel(&model, &pos, &state.qd).unwrap();
        let tau = vec![0.0; model.nv()];
        let qacc_smooth = forward_dynamics(&model, &pos, &state.qd, &tau).unwrap();
        let mut rows = Vec::new();
        for k in 0..4 {
            let angle = 2.0 * core::f64::consts::PI * (k as f64) / 4.0;
            contact_rows(
                &model,
                &pos,
                &cvel,
                &cdof_dot,
                &state.qd,
                [0.05 * angle.cos(), 0.05 * angle.sin(), -0.002],
                [0.0, 0.0, 1.0],
                0.002,
                0,
                None,
                3,
                0.6,
                default_solref(),
                default_solimp(),
                0.0,
                &mut rows,
            )
            .unwrap();
        }
        (model, state, rows, qacc_smooth, pos)
    }

    #[test]
    fn newton_agrees_with_converged_gauss_seidel_on_the_same_problem() {
        // The two solve the same convex problem by different routes -- one in
        // force space, one in acceleration space -- so they must land on the
        // same acceleration. This is the cross-algorithm check that makes the
        // new solver trustworthy, in the same spirit as validating the
        // composite mass matrix against the articulated-body inverse dynamics.
        let (model, state, rows, qacc_smooth, pos) = four_contact_problem();
        let dt = 1e-3;
        let pgs = solve(
            &model,
            &pos,
            &state.qd,
            &qacc_smooth,
            &rows,
            dt,
            SolverConfig {
                iterations: 20_000,
                tolerance: 1e-12,
                warm_start: true,
            },
            None,
        )
        .unwrap();
        let newton = solve_newton(
            &model,
            &pos,
            &state.qd,
            &qacc_smooth,
            &rows,
            dt,
            SolverConfig::default(),
            None,
        )
        .unwrap();
        for (i, (p, n)) in pgs.qacc.iter().zip(&newton.qacc).enumerate() {
            let scale = p.abs().max(n.abs()).max(1.0);
            assert!(
                (p - n).abs() / scale < 1e-8,
                "degree of freedom {i}: Gauss-Seidel {p} against Newton {n}"
            );
        }
    }

    #[test]
    fn newton_needs_a_handful_of_iterations_where_gauss_seidel_needs_a_thousand() {
        // The measurement that motivated this solver, asserted rather than
        // recorded in prose. Projected Gauss-Seidel takes 1011 sweeps on this
        // problem to reach 1e-10; Newton reaches the same answer in single
        // digits, because its variable is the six-dimensional acceleration
        // rather than sixteen redundant edge coefficients.
        let (model, state, rows, qacc_smooth, pos) = four_contact_problem();
        assert_eq!(rows.len(), 16, "four pyramidal contacts are sixteen rows");
        let newton = solve_newton(
            &model,
            &pos,
            &state.qd,
            &qacc_smooth,
            &rows,
            1e-3,
            SolverConfig::default(),
            None,
        )
        .unwrap();
        // Measured: **one** iteration, with a final scaled gradient norm of
        // 4.9e-14. Once the active set is settled the problem is a pure
        // quadratic, the Hessian is exact, and the exact line search lands on
        // the minimiser in a single step. The bound is loose so that a change
        // in the active-set path shows up as a real failure rather than an
        // off-by-one.
        assert!(
            newton.iterations <= 5,
            "Newton took {} iterations",
            newton.iterations
        );
        assert!(
            newton.residual < 1e-10,
            "Newton stopped at a gradient norm of {}",
            newton.residual
        );
    }

    #[test]
    fn every_inequality_force_is_non_negative_without_a_projection() {
        // The primal solver never projects: a row's force is `-D jar` and the
        // row only contributes while `jar < 0`, so non-negativity is a property
        // of the hinge. Worth asserting, because it is the structural claim the
        // module's documentation makes.
        let (model, state, rows, qacc_smooth, pos) = four_contact_problem();
        let newton = solve_newton(
            &model,
            &pos,
            &state.qd,
            &qacc_smooth,
            &rows,
            1e-3,
            SolverConfig::default(),
            None,
        )
        .unwrap();
        for (i, f) in newton.force.iter().enumerate() {
            assert!(*f >= 0.0, "pyramid edge {i} pulled: {f}");
        }
        // And the decoded force is inside the pyramid.
        let [normal, t1, t2] = crate::solver::decode_pyramid(&newton.force[0..4], 0.6);
        assert!(normal >= 0.0);
        assert!(t1.abs() + t2.abs() <= 0.6 * normal + 1e-9);
    }

    #[test]
    fn an_empty_problem_returns_the_unconstrained_acceleration() {
        let (model, state, _, qacc_smooth, pos) = four_contact_problem();
        let out = solve_newton(
            &model,
            &pos,
            &state.qd,
            &qacc_smooth,
            &[],
            1e-3,
            SolverConfig::default(),
            None,
        )
        .unwrap();
        assert_eq!(out.qacc, qacc_smooth);
        assert!(out.force.is_empty());
    }

    #[test]
    fn bad_input_is_reported_rather_than_panicking() {
        let (model, state, rows, qacc_smooth, pos) = four_contact_problem();
        assert!(solve_newton(
            &model,
            &pos,
            &[0.0; 2],
            &qacc_smooth,
            &rows,
            1e-3,
            SolverConfig::default(),
            None,
        )
        .is_err());
        assert!(solve_newton(
            &model,
            &pos,
            &state.qd,
            &qacc_smooth,
            &rows,
            0.0,
            SolverConfig::default(),
            None,
        )
        .is_err());
    }
}

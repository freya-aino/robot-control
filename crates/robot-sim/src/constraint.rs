//! Equality-constraint build and solve ported from MuJoCo 3.17.0 (commit
//! `800a109a`, `src/engine/engine_core_constraint.c`).
//!
//! Pure algorithm module: `std` and plain `f64` slices and vectors only (FR-6).
//! Each function names the pinned MuJoCo function it transfers. The scope is
//! the scalar `mjEQ_JOINT` coupling, the soft-constraint impedance and
//! reference (`solref`/`solimp`), and the closed linear solve of an
//! equality-only problem. Not ported: `mjEQ_CONNECT`, `mjEQ_WELD` and
//! `mjEQ_TENDON` (they need body or tendon kinematics), `mj_Jdotv`,
//! friction, limit and contact rows, the iterative solvers in
//! `engine_solver.c`, the discrete-integrator metric branch, and the default
//! `mj_diagApprox` path (the port uses the exact diagonal, as the
//! `mjENBL_DIAGEXACT` path does). Deviations are listed in
//! `crates/robot-sim/specs/06-equality-constraints-and-solver.md`.

use crate::integrators::cholesky_solve;

/// `mjMINVAL`, minimum value in any denominator (`mjtype.h`).
pub const MINVAL: f64 = 1e-15;
/// `mjMINIMP`, minimum constraint impedance (`mujoco/mjmodel.h`).
pub const MINIMP: f64 = 0.0001;
/// `mjMAXIMP`, maximum constraint impedance (`mujoco/mjmodel.h`).
pub const MAXIMP: f64 = 0.9999;
/// Default `solref` from `mj_defaultSolRefImp` (`engine_init.c`).
pub const DEFAULT_SOLREF: [f64; 2] = [0.02, 1.0];
/// Default `solimp` from `mj_defaultSolRefImp` (`engine_init.c`).
pub const DEFAULT_SOLIMP: [f64; 5] = [0.9, 0.95, 0.001, 0.5, 2.0];

/// Why a constraint build or solve was rejected. Rejected inputs never touch
/// the caller's state, because every function here takes shared references.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ConstraintError {
    /// A value that must be finite is NaN or infinite.
    NonFinite,
    /// The timestep is zero or negative.
    NonPositiveTimestep,
    /// `solref` mixes the standard (`timeconst > 0`) and direct formats. MuJoCo
    /// replaces it with the default and warns; this port refuses it instead.
    MixedSolrefFormat,
    /// A slice width does not match the number of degrees of freedom.
    WidthMismatch,
    /// The mass matrix or the regularized constraint matrix is not symmetric
    /// positive definite.
    NotPositiveDefinite,
    /// The mass matrix is not symmetric. `cholesky_solve` reads only the lower
    /// triangle, so an asymmetric input would be misread silently; it is refused.
    NotSymmetric,
}

impl core::fmt::Display for ConstraintError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonFinite => write!(f, "constraint input is non-finite"),
            Self::NonPositiveTimestep => write!(f, "constraint timestep must be positive"),
            Self::MixedSolrefFormat => {
                write!(f, "solref mixes the standard and direct formats")
            }
            Self::WidthMismatch => write!(f, "constraint slice width does not match nv"),
            Self::NotPositiveDefinite => {
                write!(f, "constraint system is not symmetric positive definite")
            }
            Self::NotSymmetric => write!(f, "mass matrix is not symmetric"),
        }
    }
}

impl core::error::Error for ConstraintError {}

/// The `solimp` clamp of `getsolparam` in `engine_core_constraint.c`:
/// both impedance bounds into `[MINIMP, MAXIMP]`, width `>= 0`, midpoint into
/// `[MINIMP, MAXIMP]`, power `>= 1`. Call only after finiteness is checked.
#[must_use]
pub fn clamp_solimp(solimp: [f64; 5]) -> [f64; 5] {
    [
        solimp[0].clamp(MINIMP, MAXIMP),
        solimp[1].clamp(MINIMP, MAXIMP),
        solimp[2].max(0.0),
        solimp[3].clamp(MINIMP, MAXIMP),
        solimp[4].max(1.0),
    ]
}

/// The integrator safety clamp of `getsolparam` (`engine_core_constraint.c`,
/// line 1906): for the standard format (`solref[0] > 0`) the time constant is
/// raised to `2 * timestep`, and the returned flag records that it was raised.
/// The source applies this clamp only when `mj_isMetric` is false, that is,
/// for every integrator except `mjINT_DISCRETE`; this port has no discrete
/// path, so the clamp is unconditional.
///
/// Deviation: the source warns and replaces a mixed-format `solref` with
/// `mj_defaultSolRefImp`. This port returns `MixedSolrefFormat` instead, so
/// the caller decides. Deviation: the source skips this clamp when
/// `mjDSBL_REFSAFE` is set (`getsolparam`, `engine_core_constraint.c` line
/// 1906); this port has no disable switch, so the clamp always applies. The
/// direct format (`solref[0] <= 0` and `solref[1] <= 0`) is returned unchanged.
pub fn effective_timeconst(
    solref: [f64; 2],
    timestep: f64,
) -> Result<(f64, bool), ConstraintError> {
    if !solref.iter().all(|x| x.is_finite()) {
        return Err(ConstraintError::NonFinite);
    }
    if !timestep.is_finite() {
        return Err(ConstraintError::NonFinite);
    }
    if timestep <= 0.0 {
        return Err(ConstraintError::NonPositiveTimestep);
    }
    if (solref[0] > 0.0) != (solref[1] > 0.0) {
        return Err(ConstraintError::MixedSolrefFormat);
    }
    if solref[0] > 0.0 {
        let floor = 2.0 * timestep;
        Ok((solref[0].max(floor), solref[0] < floor))
    } else {
        Ok((solref[0], false))
    }
}

/// `getimpedance` in `engine_core_constraint.c`: the impedance `imp(|pos -
/// margin| / width)` and its derivative `impP`, for an already clamped
/// `solimp`. The impedance depends on the residual only, never on the force,
/// so for a fixed state it is a constant of the solve.
#[must_use]
pub fn impedance(solimp: [f64; 5], pos: f64, margin: f64) -> (f64, f64) {
    if solimp[0] == solimp[1] || solimp[2] <= MINVAL {
        return (0.5 * (solimp[0] + solimp[1]), 0.0);
    }
    let mut x = (pos - margin) / solimp[2];
    let mut sign = 1.0;
    if x < 0.0 {
        x = -x;
        sign = -1.0;
    }
    if x >= 1.0 {
        return (solimp[1], 0.0);
    }
    if x <= 0.0 {
        return (solimp[0], 0.0);
    }
    let p = solimp[4];
    let (y, y_prime) = if p == 1.0 {
        (x, 1.0)
    } else if x <= solimp[3] {
        let a = 1.0 / solimp[3].powf(p - 1.0);
        (a * x.powf(p), p * a * x.powf(p - 1.0))
    } else {
        let b = 1.0 / (1.0 - solimp[3]).powf(p - 1.0);
        (1.0 - b * (1.0 - x).powf(p), p * b * (1.0 - x).powf(p - 1.0))
    };
    (
        solimp[0] + y * (solimp[1] - solimp[0]),
        y_prime * sign * (solimp[1] - solimp[0]) / solimp[2],
    )
}

/// The stiffness `K` and damping `B` of a constraint row, from
/// `mj_makeImpedance` in `engine_core_constraint.c`:
/// standard `K = 1 / max(MINVAL, d_width^2 * timeconst^2 * dampratio^2)`,
/// `B = 2 / max(MINVAL, d_width * timeconst)`; direct `K = -solref[0] /
/// max(MINVAL, d_width^2)`, `B = -solref[1] / max(MINVAL, d_width)`. Here
/// `d_width = solimp[1]`. Takes the clamped `solref` and `solimp`.
#[must_use]
pub fn stiffness_damping(solref: [f64; 2], solimp: [f64; 5]) -> (f64, f64) {
    let d_width = solimp[1];
    let k = if solref[0] > 0.0 {
        1.0 / (d_width * d_width * solref[0] * solref[0] * solref[1] * solref[1]).max(MINVAL)
    } else {
        -solref[0] / (d_width * d_width).max(MINVAL)
    };
    let b = if solref[1] > 0.0 {
        2.0 / (d_width * solref[0]).max(MINVAL)
    } else {
        -solref[1] / d_width.max(MINVAL)
    };
    (k, b)
}

/// One equality constraint row: a Jacobian row over the `nv` degrees of
/// freedom, the residual `pos`, and its soft-constraint parameters. Equality
/// rows have zero margin (`mj_addConstraint(..., 0, 0, ...)` in
/// `mj_instantiateEquality`).
#[derive(Clone, Debug, PartialEq)]
pub struct EqualityRow {
    /// Jacobian row, length `nv`.
    pub jacobian: Vec<f64>,
    /// Constraint residual `efc_pos`.
    pub pos: f64,
    /// `solref` as given by the model, before the timestep clamp.
    pub solref: [f64; 2],
    /// `solimp` as given by the model, before the clamp.
    pub solimp: [f64; 5],
}

/// A scalar `mjEQ_JOINT` equality: one joint, or two joints coupled by the
/// polynomial `q1 - q1_ref - c0 - (c1*d + c2*d^2 + c3*d^3 + c4*d^4)` with
/// `d = q2 - q2_ref`. Scalar (hinge/slide) joints only, so the qpos and dof
/// index of a joint coincide.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointEquality {
    /// First joint index.
    pub joint1: usize,
    /// Optional second joint index.
    pub joint2: Option<usize>,
    /// Reference (`qpos0`) of the first joint.
    pub reference1: f64,
    /// Reference (`qpos0`) of the second joint.
    pub reference2: f64,
    /// Polynomial coefficients `c0..c4`.
    pub coef: [f64; 5],
    /// Model `solref`.
    pub solref: [f64; 2],
    /// Model `solimp`.
    pub solimp: [f64; 5],
}

/// Builds the residual and Jacobian row of one `mjEQ_JOINT` equality, from
/// `mj_instantiateEquality` in `engine_core_constraint.c` (the `mjEQ_JOINT`
/// case). Two joints: `cpos = q1 - ref1 - c0 - poly(d)` and the Jacobian
/// `e1 - poly'(d) * e2`, with `poly'(d) = c1 + 2*c2*d + 3*c3*d^2 + 4*c4*d^3`.
/// One joint: `cpos = q1 - ref1 - c0` and Jacobian `e1`.
pub fn joint_equality_row(
    eq: &JointEquality,
    qpos: &[f64],
) -> Result<EqualityRow, ConstraintError> {
    let nv = qpos.len();
    if eq.joint1 >= nv || eq.joint2.is_some_and(|j| j >= nv) {
        return Err(ConstraintError::WidthMismatch);
    }
    if !qpos.iter().all(|x| x.is_finite())
        || !eq.reference1.is_finite()
        || !eq.reference2.is_finite()
        || !eq.coef.iter().all(|x| x.is_finite())
    {
        return Err(ConstraintError::NonFinite);
    }
    let mut jacobian = vec![0.0; nv];
    jacobian[eq.joint1] = 1.0;
    let pos = match eq.joint2 {
        Some(j2) => {
            let [c0, c1, c2, c3, c4] = eq.coef;
            let d = qpos[j2] - eq.reference2;
            let poly = c1 * d + c2 * d * d + c3 * d * d * d + c4 * d * d * d * d;
            let deriv = c1 + 2.0 * c2 * d + 3.0 * c3 * d * d + 4.0 * c4 * d * d * d;
            jacobian[j2] -= deriv;
            qpos[eq.joint1] - eq.reference1 - c0 - poly
        }
        None => qpos[eq.joint1] - eq.reference1 - eq.coef[0],
    };
    Ok(EqualityRow {
        jacobian,
        pos,
        solref: eq.solref,
        solimp: eq.solimp,
    })
}

/// The result of an equality solve.
#[derive(Clone, Debug, PartialEq)]
pub struct EqualitySolution {
    /// Constraint force per row, `f`.
    pub force: Vec<f64>,
    /// Acceleration `qacc = qacc_smooth + M^-1 J^T f`.
    pub qacc: Vec<f64>,
    /// Indices of rows whose standard time constant was raised to `2*dt`.
    /// Reported, not hidden: MuJoCo applies the same clamp silently.
    pub clamped_rows: Vec<usize>,
}

/// Solves an equality-only constraint problem in closed form.
///
/// Source: MuJoCo 3.17.0 `engine_core_constraint.c`. With `A = J M^-1 J^T`,
/// `R = max(MINVAL, (1 - imp) * A_ii / imp)` (`mj_makeImpedance`), and the
/// reference `aref = -B * (J qvel) - K * imp * pos` (`mj_referenceConstraint`,
/// line 3299; the `shift * vel` term is zero because `mj_isMetric` is false
/// for every integrator except `mjINT_DISCRETE`, see `engine_core_util.c`
/// line 1615; `mj_Jdotv` is applied only to connect and weld rows, so it is
/// absent here), the soft row acceleration must satisfy `J qacc - aref = -R f`.
/// Substituting `qacc = qacc_smooth + M^-1 J^T f` gives
/// `(A + diag(R)) f = aref - J qacc_smooth`. The matrix `A + diag(R)` is
/// assembled as `AR` (`mj_makeARNumeric`, line 3063, `AR[diag] += R`).
/// Impedance depends only on `pos`, so this is one linear solve, not an
/// iteration.
///
/// Deviation: `A_ii` is the exact diagonal of `J M^-1 J^T`. This matches the
/// `mjENBL_DIAGEXACT` path, which overwrites `efc_diagA` with the same
/// quantity (`mj_makeYNumeric`, `mj_projectConstraint`). It does not match
/// the default `mj_diagApprox` (`engine_core_constraint.c` line 1605). For
/// `mjEQ_JOINT` rows that path sums `dof_invweight0` (the inverse inertia at
/// `qpos0`) and has no `poly'(d)^2` coupling factor. So `R` differs whenever
/// `|poly'(d)| != 1` or `M(q) != M(qpos0)`, even for a diagonal `M`.
pub fn solve_equality(
    mass: &[f64],
    qvel: &[f64],
    qacc_smooth: &[f64],
    rows: &[EqualityRow],
    timestep: f64,
) -> Result<EqualitySolution, ConstraintError> {
    let nv = qvel.len();
    if !timestep.is_finite() {
        return Err(ConstraintError::NonFinite);
    }
    if timestep <= 0.0 {
        return Err(ConstraintError::NonPositiveTimestep);
    }
    if mass.len() != nv * nv || qacc_smooth.len() != nv {
        return Err(ConstraintError::WidthMismatch);
    }
    if rows.iter().any(|row| row.jacobian.len() != nv) {
        return Err(ConstraintError::WidthMismatch);
    }
    if !mass.iter().all(|x| x.is_finite())
        || !qvel.iter().all(|x| x.is_finite())
        || !qacc_smooth.iter().all(|x| x.is_finite())
    {
        return Err(ConstraintError::NonFinite);
    }
    let scale = mass.iter().fold(1.0f64, |acc, v| acc.max(v.abs()));
    for i in 0..nv {
        for j in (i + 1)..nv {
            if (mass[i * nv + j] - mass[j * nv + i]).abs() > 1e-12 * scale {
                return Err(ConstraintError::NotSymmetric);
            }
        }
    }

    let m = rows.len();
    let mut imp = Vec::with_capacity(m);
    let mut aref = Vec::with_capacity(m);
    let mut clamped_rows = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        if !row.pos.is_finite()
            || !row.jacobian.iter().all(|x| x.is_finite())
            || !row.solref.iter().all(|x| x.is_finite())
            || !row.solimp.iter().all(|x| x.is_finite())
        {
            return Err(ConstraintError::NonFinite);
        }
        let (timeconst, clamped) = effective_timeconst(row.solref, timestep)?;
        if clamped {
            clamped_rows.push(i);
        }
        let solref = [timeconst, row.solref[1]];
        let solimp = clamp_solimp(row.solimp);
        let (imp_i, _) = impedance(solimp, row.pos, 0.0);
        let (stiffness, damping) = stiffness_damping(solref, solimp);
        let vel = dot(&row.jacobian, qvel);
        aref.push(-damping * vel - stiffness * imp_i * row.pos);
        imp.push(imp_i);
    }

    let mut minv_jt = Vec::with_capacity(m);
    for row in rows {
        let mut factor = mass.to_vec();
        let mut x = row.jacobian.clone();
        if !cholesky_solve(&mut factor, &mut x, nv) {
            return Err(ConstraintError::NotPositiveDefinite);
        }
        minv_jt.push(x);
    }

    let mut ar = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..m {
            ar[i * m + j] = dot(&rows[i].jacobian, &minv_jt[j]);
        }
    }
    for i in 0..m {
        let diag = ar[i * m + i];
        ar[i * m + i] += ((1.0 - imp[i]) * diag / imp[i]).max(MINVAL);
    }

    let mut force: Vec<f64> = (0..m)
        .map(|i| aref[i] - dot(&rows[i].jacobian, qacc_smooth))
        .collect();
    if !cholesky_solve(&mut ar, &mut force, m) {
        return Err(ConstraintError::NotPositiveDefinite);
    }

    let mut qacc = qacc_smooth.to_vec();
    for (f, x) in force.iter().zip(&minv_jt) {
        for (a, xk) in qacc.iter_mut().zip(x) {
            *a += f * xk;
        }
    }

    Ok(EqualitySolution {
        force,
        qacc,
        clamped_rows,
    })
}

/// Plain dot product of two equal-length slices.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    const FLAT_HALF: [f64; 5] = [0.5, 0.5, 0.001, 0.5, 2.0];

    #[test]
    fn impedance_is_saturated_at_both_ends_of_the_width() {
        // Hand derivation: solimp = [0.9, 0.95, 0.001, 0.5, 2], margin 0.
        // pos = 0 gives x = 0, so imp = dmin = 0.9. pos = 0.001 gives x = 1, so
        // imp = dmax = 0.95. pos = -0.002 gives |x| = 2 >= 1, so imp = dmax again:
        // the impedance depends on |pos|, not on the sign of the violation.
        let solimp = [0.9, 0.95, 0.001, 0.5, 2.0];
        assert_eq!(impedance(solimp, 0.0, 0.0), (0.9, 0.0));
        assert_eq!(impedance(solimp, 0.001, 0.0), (0.95, 0.0));
        assert_eq!(impedance(solimp, -0.002, 0.0), (0.95, 0.0));
    }

    #[test]
    fn impedance_quadratic_branch_at_the_midpoint() {
        // Hand derivation: pos = 0.0005 gives x = 0.5, which is the midpoint, so
        // the x <= midpoint branch applies. a = 1 / 0.5^(p-1) = 1 / 0.5 = 2,
        // y = a*x^2 = 2*0.25 = 0.5. imp = 0.9 + 0.5*(0.95 - 0.9) = 0.925.
        // yP = p*a*x^(p-1) = 2*2*0.5 = 2, and impP = yP*(0.05)/0.001 = 100.
        let (imp, imp_prime) = impedance([0.9, 0.95, 0.001, 0.5, 2.0], 0.0005, 0.0);
        assert!((imp - 0.925).abs() < 1e-15, "imp = {imp}");
        assert!((imp_prime - 100.0).abs() < 1e-12, "impP = {imp_prime}");
    }

    #[test]
    fn impedance_linear_branch_when_power_is_one() {
        // Hand derivation: p = 1 means y(x) = x. pos = 0.00025 gives x = 0.25,
        // so imp = 0.9 + 0.25*0.05 = 0.9125 and impP = 1*0.05/0.001 = 50.
        let (imp, imp_prime) = impedance([0.9, 0.95, 0.001, 0.5, 1.0], 0.00025, 0.0);
        assert!((imp - 0.9125).abs() < 1e-15, "imp = {imp}");
        assert!((imp_prime - 50.0).abs() < 1e-12, "impP = {imp_prime}");
    }

    #[test]
    fn impedance_is_flat_when_the_bounds_are_equal() {
        // Hand derivation: dmin = dmax = 0.7 is the flat branch, so imp = 0.7
        // for any residual, with zero derivative.
        assert_eq!(impedance([0.7, 0.7, 0.001, 0.5, 2.0], 0.5, 0.0), (0.7, 0.0));
    }

    #[test]
    fn clamp_solimp_bounds_each_entry() {
        // Hand derivation: dmin 0 -> 1e-4, dmax 2 -> 0.9999, width -1 -> 0,
        // midpoint 0 -> 1e-4, power 0.5 -> 1.
        let clamped = clamp_solimp([0.0, 2.0, -1.0, 0.0, 0.5]);
        assert_eq!(clamped, [MINIMP, MAXIMP, 0.0, MINIMP, 1.0]);
    }

    #[test]
    fn standard_stiffness_and_damping_match_the_closed_forms() {
        // Hand derivation: solref = [0.02, 1], d_width = 0.95.
        // K = 1 / (0.95^2 * 0.02^2 * 1^2) = 1 / 0.000361.
        // B = 2 / (0.95 * 0.02) = 2 / 0.019.
        let (k, b) = stiffness_damping([0.02, 1.0], [0.9, 0.95, 0.001, 0.5, 2.0]);
        assert!((k - 1.0 / 0.000361).abs() < 1e-9 * k, "K = {k}");
        assert!((b - 2.0 / 0.019).abs() < 1e-12, "B = {b}");
    }

    #[test]
    fn direct_stiffness_and_damping_use_the_negated_references() {
        // Hand derivation: solref = [-100, -10], d_width = 0.95.
        // K = -(-100) / 0.95^2 = 100 / 0.9025 and B = -(-10) / 0.95 = 10 / 0.95.
        let (k, b) = stiffness_damping([-100.0, -10.0], [0.9, 0.95, 0.001, 0.5, 2.0]);
        assert!((k - 100.0 / 0.9025).abs() < 1e-9, "K = {k}");
        assert!((b - 10.0 / 0.95).abs() < 1e-12, "B = {b}");
    }

    #[test]
    fn short_time_constant_is_raised_to_twice_the_timestep() {
        // Hand derivation: timeconst 0.0005 with dt = 0.001 is below 2*dt = 0.002,
        // so it becomes 0.002 and the clamp is reported. timeconst 0.02 is kept.
        assert_eq!(effective_timeconst([0.0005, 1.0], 0.001), Ok((0.002, true)));
        assert_eq!(effective_timeconst([0.02, 1.0], 0.001), Ok((0.02, false)));
        assert_eq!(
            effective_timeconst([-100.0, -10.0], 0.001),
            Ok((-100.0, false))
        );
    }

    #[test]
    fn mixed_solref_format_is_rejected() {
        // Hand derivation: solref[0] = 0.02 > 0 but solref[1] = -1 <= 0, so the
        // two formats are mixed and the row is refused.
        assert_eq!(
            effective_timeconst([0.02, -1.0], 0.001),
            Err(ConstraintError::MixedSolrefFormat)
        );
    }

    #[test]
    fn single_row_solve_matches_the_hand_derived_force() {
        // Hand derivation: M = [1], J = [1], qvel = 0, qacc_smooth = 0, dt = 0.001,
        // flat imp = 0.5, pos = 0.001, default solref [0.02, 1].
        // K = 1/(0.5^2 * 0.02^2) = 10000, B = 2/(0.5*0.02) = 200.
        // aref = -K*imp*pos = -10000*0.5*0.001 = -5.
        // A = 1, R = (1-0.5)*1/0.5 = 1, so AR = 2 and f = (aref - 0)/2 = -2.5.
        // qacc = f * M^-1 J^T = -2.5. Check: J qacc - aref + R f = -2.5 + 5 - 2.5 = 0.
        let row = EqualityRow {
            jacobian: vec![1.0],
            pos: 0.001,
            solref: DEFAULT_SOLREF,
            solimp: FLAT_HALF,
        };
        let sol = solve_equality(&[1.0], &[0.0], &[0.0], &[row], 0.001).unwrap();
        assert!((sol.force[0] + 2.5).abs() < 1e-12, "f = {:?}", sol.force);
        assert!((sol.qacc[0] + 2.5).abs() < 1e-12, "qacc = {:?}", sol.qacc);
        assert!(sol.clamped_rows.is_empty());
    }

    #[test]
    fn duplicate_rows_share_the_load_without_singularity() {
        // Hand derivation: two identical rows J = [1], pos = 0.001 give A = [[1,1],[1,1]],
        // R = 1 per row, so AR = [[2,1],[1,2]]. rhs = aref = [-5, -5] (as in the
        // single-row case). Solving 2a + b = -5 and a + 2b = -5 gives a = b = -5/3.
        // qacc = (a + b) = -10/3. Check: J qacc - aref + R f = -10/3 + 5 - 5/3 = 0.
        let row = EqualityRow {
            jacobian: vec![1.0],
            pos: 0.001,
            solref: DEFAULT_SOLREF,
            solimp: FLAT_HALF,
        };
        let sol = solve_equality(&[1.0], &[0.0], &[0.0], &[row.clone(), row], 0.001).unwrap();
        assert!(
            (sol.force[0] + 5.0 / 3.0).abs() < 1e-12,
            "f = {:?}",
            sol.force
        );
        assert!(
            (sol.force[1] + 5.0 / 3.0).abs() < 1e-12,
            "f = {:?}",
            sol.force
        );
        assert!(
            (sol.qacc[0] + 10.0 / 3.0).abs() < 1e-12,
            "qacc = {:?}",
            sol.qacc
        );
    }

    #[test]
    fn joint_coupling_solve_matches_the_hand_derived_forces() {
        // Hand derivation: two hinge joints, M = I_2, q = (0.2, 0), qvel = 0,
        // qacc_smooth = 0, coupling q1 = q2 (c0 = 0, c1 = 1, c2..c4 = 0), so
        // pos = 0.2 and J = [1, -1]. flat imp = 0.5, K = 10000, B = 200.
        // aref = -K*imp*pos = -1000. A = J J^T = 2, R = (1-0.5)*2/0.5 = 2,
        // AR = 4, f = -1000/4 = -250. qacc = f * J^T = [-250, 250].
        // Check: J qacc - aref + R f = -500 + 1000 - 500 = 0.
        let eq = JointEquality {
            joint1: 0,
            joint2: Some(1),
            reference1: 0.0,
            reference2: 0.0,
            coef: [0.0, 1.0, 0.0, 0.0, 0.0],
            solref: DEFAULT_SOLREF,
            solimp: FLAT_HALF,
        };
        let row = joint_equality_row(&eq, &[0.2, 0.0]).unwrap();
        assert_eq!(row.jacobian, vec![1.0, -1.0]);
        assert!((row.pos - 0.2).abs() < 1e-15);
        let mass = [1.0, 0.0, 0.0, 1.0];
        let sol = solve_equality(&mass, &[0.0, 0.0], &[0.0, 0.0], &[row], 0.001).unwrap();
        assert!((sol.force[0] + 250.0).abs() < 1e-9, "f = {:?}", sol.force);
        assert!((sol.qacc[0] + 250.0).abs() < 1e-9, "qacc = {:?}", sol.qacc);
        assert!((sol.qacc[1] - 250.0).abs() < 1e-9, "qacc = {:?}", sol.qacc);
        let residual = (sol.qacc[0] - sol.qacc[1]) + 1000.0 + 2.0 * sol.force[0];
        assert!(residual.abs() < 1e-9, "residual = {residual}");
    }

    #[test]
    fn scalar_solve_with_velocity_and_smooth_acceleration_is_hand_derived() {
        // Hand derivation: M = [1], J = [1], dt = 0.001, flat imp = 0.5, K = 10000,
        // B = 200, R = 1, AR = 2 (as in the single-row case). pos = 0.001,
        // qvel = 0.5, qacc_smooth = 1. J qvel = 0.5, so aref = -200*0.5 - 5 = -105.
        // rhs = aref - J qacc_smooth = -106, so f = -53 and qacc = 1 - 53 = -52.
        // Residual J qacc - aref + R f = -52 + 105 - 53 = 0.
        let row = EqualityRow {
            jacobian: vec![1.0],
            pos: 0.001,
            solref: DEFAULT_SOLREF,
            solimp: FLAT_HALF,
        };
        let sol = solve_equality(&[1.0], &[0.5], &[1.0], &[row], 0.001).unwrap();
        assert!((sol.force[0] + 53.0).abs() < 1e-9, "f = {:?}", sol.force);
        assert!((sol.qacc[0] + 52.0).abs() < 1e-9, "qacc = {:?}", sol.qacc);
        let residual = sol.qacc[0] + 105.0 + sol.force[0];
        assert!(residual.abs() < 1e-9, "residual = {residual}");
        let second = 1.0 + sol.force[0];
        assert!((sol.qacc[0] - second).abs() < 1e-9);
    }

    #[test]
    fn coupled_inertia_solve_matches_the_hand_derived_forces_and_residual() {
        // Hand derivation: M = [[2,1],[1,2]], det = 3, M^-1 = (1/3)[[2,-1],[-1,2]].
        // J = [1,-1], so M^-1 J^T = (1/3)(3,-3) = (1,-1) and A = J M^-1 J^T = 2.
        // dt = 0.001, flat imp = 0.5, K = 10000, B = 200, R = (1-0.5)*2/0.5 = 2, AR = 4.
        // pos = 0.001, qvel = (0.5, 0), qacc_smooth = (0.6, 0.2).
        // J qvel = 0.5, so aref = -200*0.5 - 5 = -105.
        // J qacc_smooth = 0.4, so rhs = -105.4 and f = -105.4/4 = -26.35.
        // qacc = (0.6, 0.2) + (-26.35)*(1,-1) = (-25.75, 26.55).
        // Residual J qacc - aref + R f = -52.3 + 105 - 52.7 = 0.
        let row = EqualityRow {
            jacobian: vec![1.0, -1.0],
            pos: 0.001,
            solref: DEFAULT_SOLREF,
            solimp: FLAT_HALF,
        };
        let mass = [2.0, 1.0, 1.0, 2.0];
        let sol = solve_equality(&mass, &[0.5, 0.0], &[0.6, 0.2], &[row], 0.001).unwrap();
        assert!((sol.force[0] + 26.35).abs() < 1e-9, "f = {:?}", sol.force);
        assert!((sol.qacc[0] + 25.75).abs() < 1e-9, "qacc = {:?}", sol.qacc);
        assert!((sol.qacc[1] - 26.55).abs() < 1e-9, "qacc = {:?}", sol.qacc);
        let residual = (sol.qacc[0] - sol.qacc[1]) + 105.0 + 2.0 * sol.force[0];
        assert!(residual.abs() < 1e-9, "residual = {residual}");
        assert!((sol.qacc[0] - (0.6 + sol.force[0])).abs() < 1e-9);
        assert!((sol.qacc[1] - (0.2 - sol.force[0])).abs() < 1e-9);
    }

    #[test]
    fn asymmetric_mass_is_refused_rather_than_read_from_one_triangle() {
        // Hand derivation: M = [[2,1],[0,2]] has M12 = 1 but M21 = 0, so it is not
        // symmetric. cholesky_solve reads only the lower triangle and would solve for
        // a different matrix, so the solve is refused before any row is processed.
        let row = EqualityRow {
            jacobian: vec![1.0, -1.0],
            pos: 0.001,
            solref: DEFAULT_SOLREF,
            solimp: FLAT_HALF,
        };
        assert_eq!(
            solve_equality(
                &[2.0, 1.0, 0.0, 2.0],
                &[0.0, 0.0],
                &[0.0, 0.0],
                &[row],
                0.001
            ),
            Err(ConstraintError::NotSymmetric)
        );
    }

    #[test]
    fn polynomial_coupling_uses_the_cubic_and_quartic_terms() {
        // Hand derivation: coef = [0.1, 1, 2, 3, 4], reference2 = 0.2, q = (0.3, 0.5),
        // reference1 = 0, so d = 0.5 - 0.2 = 0.3.
        // poly = d + 2d^2 + 3d^3 + 4d^4 = 0.3 + 0.18 + 0.081 + 0.0324 = 0.5934.
        // deriv = 1 + 4d + 9d^2 + 16d^3 = 1 + 1.2 + 0.81 + 0.432 = 3.442.
        // pos = 0.3 - 0 - 0.1 - 0.5934 = -0.3934 and J = [1, -3.442].
        let eq = JointEquality {
            joint1: 0,
            joint2: Some(1),
            reference1: 0.0,
            reference2: 0.2,
            coef: [0.1, 1.0, 2.0, 3.0, 4.0],
            solref: DEFAULT_SOLREF,
            solimp: FLAT_HALF,
        };
        let row = joint_equality_row(&eq, &[0.3, 0.5]).unwrap();
        assert!((row.pos + 0.3934).abs() < 1e-12, "pos = {}", row.pos);
        assert!(
            (row.jacobian[1] + 3.442).abs() < 1e-12,
            "J = {:?}",
            row.jacobian
        );
    }

    #[test]
    fn polynomial_coupling_uses_the_derivative_of_the_quadratic() {
        // Hand derivation: coef = [0, 0, 1, 0, 0] (quadratic c2 = 1), q = (0, 0.4),
        // refs 0. d = 0.4. pos = q1 - c2*d^2 = 0 - 0.16 = -0.16.
        // deriv = 2*c2*d = 0.8, so J = [1, -0.8].
        let eq = JointEquality {
            joint1: 0,
            joint2: Some(1),
            reference1: 0.0,
            reference2: 0.0,
            coef: [0.0, 0.0, 1.0, 0.0, 0.0],
            solref: DEFAULT_SOLREF,
            solimp: FLAT_HALF,
        };
        let row = joint_equality_row(&eq, &[0.0, 0.4]).unwrap();
        assert!((row.pos + 0.16).abs() < 1e-15, "pos = {}", row.pos);
        assert!(
            (row.jacobian[1] + 0.8).abs() < 1e-15,
            "J = {:?}",
            row.jacobian
        );
    }

    #[test]
    fn single_joint_equality_is_a_lock_to_the_offset() {
        // Hand derivation: one joint, q1 = 0.5, reference 0.1, c0 = 0.3, so
        // pos = 0.5 - 0.1 - 0.3 = 0.1 and J = [1, 0].
        let eq = JointEquality {
            joint1: 0,
            joint2: None,
            reference1: 0.1,
            reference2: 0.0,
            coef: [0.3, 0.0, 0.0, 0.0, 0.0],
            solref: DEFAULT_SOLREF,
            solimp: FLAT_HALF,
        };
        let row = joint_equality_row(&eq, &[0.5, 0.0]).unwrap();
        assert!((row.pos - 0.1).abs() < 1e-15);
        assert_eq!(row.jacobian, vec![1.0, 0.0]);
    }

    #[test]
    fn no_rows_returns_the_smooth_acceleration() {
        let sol = solve_equality(&[2.0], &[0.0], &[3.0], &[], 0.001).unwrap();
        assert!(sol.force.is_empty());
        assert_eq!(sol.qacc, vec![3.0]);
    }

    #[test]
    fn short_time_constant_is_reported_by_the_solve() {
        // Hand derivation: solref [0.0005, 1] with dt = 0.001 clamps to 0.002, and
        // the solve reports row 0 as clamped.
        let row = EqualityRow {
            jacobian: vec![1.0],
            pos: 0.0,
            solref: [0.0005, 1.0],
            solimp: FLAT_HALF,
        };
        let sol = solve_equality(&[1.0], &[0.0], &[0.0], &[row], 0.001).unwrap();
        assert_eq!(sol.clamped_rows, vec![0]);
    }

    #[test]
    fn invalid_inputs_are_refused_before_any_work() {
        let good = EqualityRow {
            jacobian: vec![1.0],
            pos: 0.001,
            solref: DEFAULT_SOLREF,
            solimp: FLAT_HALF,
        };
        let nan_pos = EqualityRow {
            pos: f64::NAN,
            ..good.clone()
        };
        assert_eq!(
            solve_equality(&[1.0], &[0.0], &[0.0], &[nan_pos], 0.001),
            Err(ConstraintError::NonFinite)
        );
        assert_eq!(
            solve_equality(&[0.0], &[0.0], &[0.0], std::slice::from_ref(&good), 0.001),
            Err(ConstraintError::NotPositiveDefinite)
        );
        assert_eq!(
            solve_equality(
                &[1.0, 0.0, 0.0],
                &[0.0],
                &[0.0],
                std::slice::from_ref(&good),
                0.001
            ),
            Err(ConstraintError::WidthMismatch)
        );
        assert_eq!(
            solve_equality(&[1.0], &[0.0], &[0.0], std::slice::from_ref(&good), 0.0),
            Err(ConstraintError::NonPositiveTimestep)
        );
        let mixed = EqualityRow {
            solref: [0.02, -1.0],
            ..good
        };
        assert_eq!(
            solve_equality(&[1.0], &[0.0], &[0.0], &[mixed], 0.001),
            Err(ConstraintError::MixedSolrefFormat)
        );
    }

    #[test]
    fn joint_index_outside_the_state_is_refused() {
        let eq = JointEquality {
            joint1: 0,
            joint2: Some(2),
            reference1: 0.0,
            reference2: 0.0,
            coef: [0.0; 5],
            solref: DEFAULT_SOLREF,
            solimp: FLAT_HALF,
        };
        assert_eq!(
            joint_equality_row(&eq, &[0.0, 0.0]),
            Err(ConstraintError::WidthMismatch)
        );
    }
}

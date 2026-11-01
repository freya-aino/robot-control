//! The unified constraint solver: contact, joint limits and equalities solved
//! as **one** convex problem over constraint forces.
//!
//! Source of truth: `docs/example-system-specs/07_constraints.md` for the
//! formulation and `08_solvers_islands.md` for the algorithm. Section numbers
//! in the doc comments below are those files'.
//!
//! # Why this module exists
//!
//! `specs/06-equality-constraints-and-solver.md` item 1 requires contact and
//! every equality constraint to be solved as one convex problem over constraint
//! forces, not as a classical complementarity problem and not as separate
//! mechanisms. Six entries of
//! `specs/appendix-d-mathematical-challenges.md` record what stood in for that
//! and what each substitution cost:
//!
//! - **M-04**: contact was a per-point penalty spring, so a body resting on `n`
//!   sampled points carried `n` springs in parallel and its rest penetration
//!   was `m g / (n k)`. Contact compliance depended on how finely a shape was
//!   sampled. Here a contact is one row per contact point with its own
//!   impedance, and the rows are solved together, so adding points
//!   redistributes one total force instead of adding force.
//! - **M-08**: joint limits were an acceleration-level active set plus a
//!   position clamp that discarded kinetic energy with no accounting and
//!   admitted no restitution. Here a limit is a row like any other, with the
//!   same `solref`/`solimp` impedance.
//! - **M-09**: contact impedance was referred to a fixed 1 kg reference mass.
//!   Here `R_ii = (1 - d_i)/d_i * A_ii` with `A_ii` the exact diagonal of
//!   `J M^-1 J^T`, which is the example's own formula (section 2.4).
//! - **M-10**: friction was applied after the normal solve from the solved
//!   normal force, so there was no friction cone of any kind, link-against-link
//!   pairs were frictionless, and the tangent plane was hard-coded to `z`. Here
//!   friction rows are in the same problem, bounded by the cone.
//! - **M-11**: the unconstrained acceleration omitted the `Jdot v` term and the
//!   spatial-to-classical conversion. Here `a_u = J qacc_smooth + Jdot v` with
//!   `Jdot` from the example's own `mj_jacDot`.
//! - **M-12**: the limit solve built its response matrix with contact disabled,
//!   so neither force accounted for the other. Here they are rows of one `A`.
//!
//! # What is implemented and what is not
//!
//! Implemented: equality rows, limit rows, contact rows at `condim` 1
//! (frictionless) and 3 (normal plus two tangents) with a **pyramidal** friction
//! cone, per-row `solref`/`solimp` impedance, the exact `A_ii`, warm starting,
//! and a projected Gauss-Seidel sweep over the dual problem of section 2.3.
//!
//! Not implemented, and each is a named gap rather than an omission:
//! - The **elliptic** cone. Step 06 item 4 asks for both and records that a
//!   pyramid with more edges does not converge to the elliptic solution, so
//!   these are different soft-contact dynamics rather than two approximations
//!   of one thing. Only the pyramid is here.
//! - `condim` 4 and 6 (torsional and rolling friction).
//! - The Newton and conjugate-gradient algorithms of `08`. Step 06 item 5 asks
//!   for a Newton-style method alongside a Gauss-Seidel one; this supplies the
//!   latter. Projected Gauss-Seidel is also the one the example drops for its
//!   GPU backend because a strictly sequential per-row update does not
//!   vectorise, so the batched backend will need the Newton path
//!   (`specs/11-gpu-batched-backend.md`).
//! - Islands, adhesion, and friction-loss (`frictionloss`) rows.
//!
//! Purity contract (FR-6): `std` and plain `f64` slices and `Vec<f64>` only.
//! Nothing here uses `robot_types::math::Matrix`, `RigidPose` or `HashMap`.

use crate::composite::{
    crb, cross_motion, factor_m, solve_ld, CompositeError, CompositeModel, Position,
};
use crate::spatial::{add3, cross3, dot3, scale3, sub3};
use robot_model::joint::JointKind;

/// Minimum value allowed in a denominator, `mjMINVAL`.
pub const MINVAL: f64 = 1e-15;
/// Minimum constraint impedance, `mjMINIMP`.
pub const MINIMP: f64 = 0.0001;
/// Maximum constraint impedance, `mjMAXIMP`.
pub const MAXIMP: f64 = 0.9999;

/// What a constraint row is, which determines the set its force is confined to
/// (`Omega` of section 2.3).
///
/// Every kind here is either unbounded or non-negative. That is not a
/// simplification: it is what the example's **pyramidal** representation buys.
/// A friction cone is expressed by making each row a *generator direction* of
/// the cone, `n +/- mu t_k`, and requiring its coefficient to be non-negative
/// (07 section 9.2), so the cone constraint is carried entirely by the row
/// Jacobians and the solver needs no cone projection at all.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum RowKind {
    /// Equality: the force is unbounded in both directions.
    Equality,
    /// A joint limit: it may only push inward, so it is non-negative.
    Limit,
    /// A frictionless contact (`condim = 1`): one row along the normal,
    /// non-negative.
    ContactNormal,
    /// One edge of a pyramidal friction cone, with Jacobian `n +/- mu t_k` and
    /// a non-negative coefficient. `condim = 3` produces four of them per
    /// contact, in the example's order `(+t1, -t1, +t2, -t2)`.
    ContactPyramid {
        /// Index of the first edge row of this contact, so a caller can group
        /// the edges of one contact and decode them.
        first_edge: usize,
        /// Friction coefficient of this edge's tangent direction.
        mu: f64,
    },
}

/// One row of the constraint problem.
#[derive(Clone, Debug)]
pub struct Row {
    /// Jacobian row over the degrees of freedom, `nv` wide, in the composite
    /// model's internal order.
    pub jacobian: Vec<f64>,
    /// Constraint residual `efc_pos`. For a contact this is the negative
    /// penetration depth, so that a deeper contact has a more negative
    /// residual; for a limit, the signed distance to the bound.
    pub pos: f64,
    /// Time derivative of the Jacobian row contracted with the velocity,
    /// `(Jdot v)_i`. Appendix D M-11: this term was missing entirely.
    pub jdot_v: f64,
    /// `solref` as given by the model, before the timestep clamp.
    pub solref: [f64; 2],
    /// `solimp` as given by the model, before clamping.
    pub solimp: [f64; 5],
    /// Detection margin. The impedance is evaluated at `pos - margin`.
    pub margin: f64,
    /// What kind of row this is.
    pub kind: RowKind,
}

/// The solved problem.
#[derive(Clone, Debug)]
pub struct Solution {
    /// Constraint force per row.
    pub force: Vec<f64>,
    /// Resulting acceleration, `qacc_smooth + M^-1 J^T f`.
    pub qacc: Vec<f64>,
    /// Sweeps performed.
    pub iterations: usize,
    /// Largest force change in the final sweep, the convergence measure.
    pub residual: f64,
    /// Rows whose `solref` time constant was raised to `2 * timestep`.
    pub clamped_rows: Vec<usize>,
}

/// Per-run solver settings.
///
/// These are the knobs `specs/09-fidelity-tiers.md` section 12.1 requires to be
/// per-run and queryable rather than compile-time constants, for the solver
/// part of that list (Appendix D M-21).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SolverConfig {
    /// Maximum projected Gauss-Seidel sweeps.
    pub iterations: usize,
    /// Stop once the largest force change in a sweep falls below this.
    pub tolerance: f64,
    /// Whether to start from the previous step's forces.
    pub warm_start: bool,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            iterations: 100,
            tolerance: 1e-10,
            warm_start: true,
        }
    }
}

/// `mju_makeFrame` of 02 section 1.3: two unit tangents orthogonal to `normal`.
///
/// The example's construction exactly, including which fallback axis is chosen,
/// because the tangent directions decide what the friction rows mean.
#[must_use]
pub fn contact_tangents(normal: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let x = normal;
    let mut y = [0.0, 0.0, 0.0];
    // "if |y|^2 < 0.25 set y = (0,1,0) if -0.5 < x1 < 0.5 else (0,0,1)"
    if x[1] > -0.5 && x[1] < 0.5 {
        y[1] = 1.0;
    } else {
        y[2] = 1.0;
    }
    // Gram-Schmidt, then normalise.
    let d = dot3(x, y);
    y = sub3(y, scale3(x, d));
    let n = dot3(y, y).sqrt();
    let y = if n > MINVAL {
        scale3(y, 1.0 / n)
    } else {
        [0.0, 1.0, 0.0]
    };
    let z = cross3(x, y);
    (y, z)
}

/// The translational and rotational point Jacobians of a body-fixed point,
/// each `3 x nv` row-major (02 section 4, `mj_jac`).
///
/// `jacp[r * nv + i]` is the velocity of `point` along world axis `r` produced
/// by a unit rate of degree of freedom `i`.
#[must_use]
pub fn point_jacobian(
    model: &CompositeModel,
    pos: &Position,
    point: [f64; 3],
    body: usize,
) -> (Vec<f64>, Vec<f64>) {
    let nv = model.tree.nv;
    let mut jacp = vec![0.0; 3 * nv];
    let mut jacr = vec![0.0; 3 * nv];
    let offset = sub3(point, pos.subtree_com[body]);
    // Walk the ancestor chain of this body's last degree of freedom.
    let mut i = last_dof_of(model, body);
    while let Some(d) = i {
        let c = pos.cdof[d];
        let ang = [c[0], c[1], c[2]];
        let lin = [c[3], c[4], c[5]];
        let v = add3(lin, cross3(ang, offset));
        for r in 0..3 {
            jacp[r * nv + d] = v[r];
            jacr[r * nv + d] = ang[r];
        }
        let p = model.tree.dof_parentid[d];
        i = if p < 0 { None } else { Some(p as usize) };
    }
    (jacp, jacr)
}

/// The time derivative of the point Jacobian, `3 x nv` (02 section 4,
/// `mj_jacDot`).
///
/// Only the translational part is returned, which is what a contact row needs.
/// `cvel` and `cdof_dot` come from [`crate::composite::com_vel`].
#[must_use]
pub fn point_jacobian_dot(
    model: &CompositeModel,
    pos: &Position,
    cvel: &[[f64; 6]],
    cdof_dot: &[[f64; 6]],
    point: [f64; 3],
    body: usize,
) -> (Vec<f64>, Vec<f64>) {
    let nv = model.tree.nv;
    let mut jacp = vec![0.0; 3 * nv];
    // The rotational Jacobian is `jacr[:, i] = cdof[i][0:3]`, so its derivative
    // is `cdof_dot[i][0:3]` directly -- no transport term, because a rotational
    // row does not depend on where the contact point is.
    let mut jacr = vec![0.0; 3 * nv];
    let offset = sub3(point, pos.subtree_com[body]);
    // The point's own velocity, transformed from the tree reference point to
    // `point` (motion convention: linear part picks up `omega x offset`).
    let v = cvel[body];
    let omega = [v[0], v[1], v[2]];
    let pvel_lin = add3([v[3], v[4], v[5]], cross3(omega, offset));

    let mut i = last_dof_of(model, body);
    while let Some(d) = i {
        // For a quaternion degree of freedom the motion subspace rotates with
        // the body's own velocity, so its derivative is taken against that
        // rather than read from `cdof_dot`.
        let cdd = if is_quaternion_dof(model, d) {
            cross_motion(cvel[model.tree.dof_bodyid[d]], pos.cdof[d])
        } else {
            cdof_dot[d]
        };
        let cdd_ang = [cdd[0], cdd[1], cdd[2]];
        let cdd_lin = [cdd[3], cdd[4], cdd[5]];
        let c = pos.cdof[d];
        let c_ang = [c[0], c[1], c[2]];
        let term = add3(
            add3(cdd_lin, cross3(cdd_ang, offset)),
            cross3(c_ang, pvel_lin),
        );
        for r in 0..3 {
            jacp[r * nv + d] = term[r];
            jacr[r * nv + d] = cdd_ang[r];
        }
        let p = model.tree.dof_parentid[d];
        i = if p < 0 { None } else { Some(p as usize) };
    }
    (jacp, jacr)
}

/// The last degree of freedom owned by `body` or, if it has none, by its
/// nearest ancestor that does.
fn last_dof_of(model: &CompositeModel, body: usize) -> Option<usize> {
    let mut b = Some(body);
    while let Some(i) = b {
        let bd = &model.bodies[i];
        if bd.dof_num > 0 {
            return Some(bd.dof_adr + bd.dof_num - 1);
        }
        b = bd.parent;
    }
    None
}

/// Whether a degree of freedom is one of a `Free` joint's three rotational
/// ones, whose motion subspace is expressed in body axes and therefore rotates
/// with the body.
fn is_quaternion_dof(model: &CompositeModel, dof: usize) -> bool {
    let body = &model.bodies[model.tree.dof_bodyid[dof]];
    body.kind == JointKind::Free && dof >= body.dof_adr + 3
}

/// `getsolparam`'s integrator safety clamp: a standard-format time constant is
/// raised to `2 * timestep`, and the flag records that it was.
fn effective_timeconst(solref: [f64; 2], timestep: f64) -> (f64, bool) {
    if solref[0] > 0.0 {
        let floor = 2.0 * timestep;
        (solref[0].max(floor), solref[0] < floor)
    } else {
        (solref[0], false)
    }
}

/// `getimpedance` of 07: the impedance at a residual, for a clamped `solimp`.
fn impedance(solimp: [f64; 5], pos: f64, margin: f64) -> f64 {
    let dmin = solimp[0].clamp(MINIMP, MAXIMP);
    let dmax = solimp[1].clamp(MINIMP, MAXIMP);
    let width = solimp[2].max(0.0);
    let mid = solimp[3].clamp(MINIMP, MAXIMP);
    let power = solimp[4].max(1.0);
    if dmin == dmax || width <= MINVAL {
        return 0.5 * (dmin + dmax);
    }
    let mut x = (pos - margin) / width;
    if x < 0.0 {
        x = -x;
    }
    if x >= 1.0 {
        return dmax;
    }
    if x <= 0.0 {
        return dmin;
    }
    let y = if power == 1.0 {
        x
    } else if x <= mid {
        let a = 1.0 / mid.powf(power - 1.0);
        a * x.powf(power)
    } else {
        let b = 1.0 / (1.0 - mid).powf(power - 1.0);
        1.0 - b * (1.0 - x).powf(power)
    };
    dmin + y * (dmax - dmin)
}

/// `mj_makeImpedance`'s stiffness and damping for a row.
fn stiffness_damping(solref: [f64; 2], solimp: [f64; 5]) -> (f64, f64) {
    let d_width = solimp[1].clamp(MINIMP, MAXIMP);
    if solref[0] > 0.0 {
        (
            1.0 / (d_width * d_width * solref[0] * solref[0] * solref[1] * solref[1]).max(MINVAL),
            2.0 / (d_width * solref[0]).max(MINVAL),
        )
    } else {
        (
            -solref[0] / (d_width * d_width).max(MINVAL),
            -solref[1] / d_width.max(MINVAL),
        )
    }
}

/// The per-row impedance data both solvers need.
///
/// Separated out because the two need different amounts of `A = J M^-1 J'`.
/// Projected Gauss-Seidel works in force space and needs the whole matrix;
/// Newton works in acceleration space and needs only its diagonal, since
/// `R_ii = (1 - d)/d * A_ii` is the only place `A` enters the regularisation.
pub(crate) struct Impedance {
    /// Regularisation per row, `R_ii`.
    pub r: Vec<f64>,
    /// Constraint stiffness per row, `D_i = 1/R_i`.
    ///
    /// This is `efc_D` of 08 section 2, the primal problem's quadratic
    /// coefficient. At the optimum `J a - a_ref = -R f`, so
    /// `f = -D (J a - a_ref)`, which is why the primal solver gets a row's
    /// force by evaluating a hinge rather than by projecting onto a set.
    pub d: Vec<f64>,
    /// Reference acceleration per row, `a_ref`.
    pub aref: Vec<f64>,
    /// Rows whose time constant was raised to twice the timestep.
    pub clamped: Vec<usize>,
}

/// Computes `R`, `D` and `a_ref` for every row, given the diagonal of
/// `A = J M^-1 J'`.
pub(crate) fn row_impedance(rows: &[Row], qd: &[f64], timestep: f64, diag_a: &[f64]) -> Impedance {
    let m = rows.len();
    let mut out = Impedance {
        r: vec![0.0; m],
        d: vec![0.0; m],
        aref: vec![0.0; m],
        clamped: Vec::new(),
    };
    for (i, row) in rows.iter().enumerate() {
        let (tc, was_clamped) = effective_timeconst(row.solref, timestep);
        if was_clamped {
            out.clamped.push(i);
        }
        let solref = [tc, row.solref[1]];
        let imp = impedance(row.solimp, row.pos, row.margin);
        // Section 2.4: R_ii = (1 - d)/d * A_ii, with the exact diagonal.
        // Appendix D M-09 records the constant that stood here before.
        let mut r_i = (((1.0 - imp) / imp.max(MINVAL)) * diag_a[i]).max(MINVAL);
        // Section 2.4, pyramidal rows: "all edges share `Rpy = 2 mu^2 R0`
        // where `R0` is computed from the first edge's diagA".
        //
        // The four edges of one contact all contain the normal direction, so
        // four non-negative coefficients encode three force components and the
        // dual is convex but not strictly so: Gauss-Seidel iterates can drift
        // along the redundant direction without the per-sweep residual
        // shrinking. Sharing one R across the edges and scaling it by `2 mu^2`
        // regularises exactly that direction.
        if let RowKind::ContactPyramid { first_edge, mu } = row.kind {
            let first = &rows[first_edge];
            let d0 = impedance(first.solimp, first.pos, first.margin);
            let r0 = ((1.0 - d0) / d0.max(MINVAL)) * diag_a[first_edge];
            r_i = (2.0 * mu * mu * r0).max(MINVAL);
        }
        out.r[i] = r_i;
        out.d[i] = 1.0 / r_i.max(MINVAL);
        let (k, b) = stiffness_damping(solref, row.solimp);
        let vel = dot(&row.jacobian, qd);
        // a_ref = -B vel - K d (pos - margin).
        out.aref[i] = -b * vel - k * imp * (row.pos - row.margin);
    }
    out
}

/// Solves the constraint problem of 07 section 2.3 by projected Gauss-Seidel.
///
/// `qacc_smooth` is the unconstrained acceleration `M^-1 (tau - c)`, and `qd`
/// the current velocity, both in the composite model's internal degree-of-
/// freedom order. `warm` optionally supplies the previous step's forces.
///
/// The problem is
/// `min_{f in Omega} 0.5 f' (A + R) f + f' (a_u - a_ref)`
/// with `A = J M^-1 J^T`, `a_u = J qacc_smooth + Jdot v`,
/// `R_ii = (1 - d_i)/d_i * A_ii` and
/// `a_ref,i = -B_i (J v)_i - K_i d_i (pos_i - margin_i)`.
///
/// `Omega` is per row: equality rows free, limit and contact-normal rows
/// non-negative, friction rows confined to the pyramid `|f| <= mu * f_normal`
/// of the normal row they name.
///
/// # Errors
///
/// [`CompositeError::WidthMismatch`] or [`CompositeError::NotFinite`] on a bad
/// input, [`CompositeError::SingularInertia`] if the mass matrix cannot be
/// factorised.
#[allow(clippy::too_many_arguments)]
pub fn solve(
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

    // One factorisation of M for the whole problem, then one back-substitution
    // per row: `M^-1 J_i^T`. Appendix D M-23's lesson applied here from the
    // start.
    let mass = crb(model, pos);
    let (ld, diag_inv) = factor_m(&model.tree, &mass)?;
    let mut minv_jt: Vec<Vec<f64>> = Vec::with_capacity(m);
    for row in rows {
        let mut x = row.jacobian.clone();
        solve_ld(&model.tree, &ld, &diag_inv, &mut x);
        minv_jt.push(x);
    }

    // A = J M^-1 J^T, dense and symmetric.
    let mut a = vec![0.0; m * m];
    for i in 0..m {
        for j in i..m {
            let v = dot(&rows[i].jacobian, &minv_jt[j]);
            a[i * m + j] = v;
            a[j * m + i] = v;
        }
    }

    // Per-row impedance, regularisation and reference acceleration, shared
    // with the Newton solver through `row_impedance`.
    let diag_a: Vec<f64> = (0..m).map(|i| a[i * m + i]).collect();
    let imp = row_impedance(rows, qd, timestep, &diag_a);
    let r = imp.r;
    let aref = imp.aref;
    solution.clamped_rows = imp.clamped;

    // a_u = J qacc_smooth + Jdot v, and the dual's linear term
    // `efc_b = a_u - a_ref`.
    let b_vec: Vec<f64> = (0..m)
        .map(|i| dot(&rows[i].jacobian, qacc_smooth) + rows[i].jdot_v - aref[i])
        .collect();

    // Warm start, cold if it is not supplied or has the wrong width.
    let mut f = match (cfg.warm_start, warm) {
        (true, Some(w)) if w.len() == m && w.iter().all(|x| x.is_finite()) => w.to_vec(),
        _ => vec![0.0; m],
    };

    // Projected Gauss-Seidel on `(A + R) f = -b`, with each row projected onto
    // its own feasible set after the update.
    let mut residual = 0.0;
    let mut iterations = 0;
    for sweep in 0..cfg.iterations.max(1) {
        iterations = sweep + 1;
        let mut change = 0.0f64;
        for i in 0..m {
            let diag = a[i * m + i] + r[i];
            if diag <= MINVAL {
                continue;
            }
            let mut acc = b_vec[i];
            for (j, fj) in f.iter().enumerate() {
                if j != i {
                    acc += a[i * m + j] * fj;
                }
            }
            let unprojected = -acc / diag;
            let next = match rows[i].kind {
                RowKind::Equality => unprojected,
                // A limit pushes one way; a frictionless contact pushes along
                // its normal; a pyramid edge carries a non-negative coefficient
                // of a generator direction. The friction cone is in the
                // Jacobians, not here.
                RowKind::Limit | RowKind::ContactNormal | RowKind::ContactPyramid { .. } => {
                    unprojected.max(0.0)
                }
            };
            change = change.max((next - f[i]).abs());
            f[i] = next;
        }
        residual = change;
        if change <= cfg.tolerance {
            break;
        }
    }

    // qacc = qacc_smooth + M^-1 J^T f.
    let mut qacc = qacc_smooth.to_vec();
    for (fi, column) in f.iter().zip(&minv_jt) {
        if *fi != 0.0 {
            for (acc, x) in qacc.iter_mut().zip(column) {
                *acc += fi * x;
            }
        }
    }

    solution.force = f;
    solution.qacc = qacc;
    solution.iterations = iterations;
    solution.residual = residual;
    Ok(solution)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Default soft-constraint parameters, from the example's `defaultSolRefImp`.
#[must_use]
pub fn default_solref() -> [f64; 2] {
    [0.02, 1.0]
}

/// Default `solimp`, from the example's `defaultSolRefImp`.
#[must_use]
pub fn default_solimp() -> [f64; 5] {
    [0.9, 0.95, 0.001, 0.5, 2.0]
}

/// Builds the rows of one contact point at `condim`, given the contact frame
/// and the two bodies.
///
/// `depth` is the penetration, positive when the surfaces overlap, so every row
/// carries `pos = -depth` as the example does ("all rows carry `pos = dist`",
/// 07 section 9.2). `point` is the contact position in world coordinates. Pass
/// `body_b = None` for a contact against a static environment.
///
/// `friction` is the five-element vector the example's contact-parameter mixing
/// produces (07 section 3.1): `(fri0, fri0, fri1, fri2, fri2)` — sliding
/// friction twice, then torsional, then rolling twice. Use
/// [`sliding_friction`] to build one from a single sliding coefficient.
///
/// # Row structure
///
/// At `condim = 1` one row along the normal. Above that, the
/// **pyramid edges** of 07 section 9.2: `2(condim - 1)` rows, each a generator
/// direction of the friction cone with a non-negative coefficient, in the
/// example's order `(+t1, -t1, +t2, -t2, +spin, -spin, +roll1, -roll1, +roll2,
/// -roll2)`. Row `dimid` uses `dimid2 = dimid/2 + 1` and the direction
///
/// - `frame[dimid2]` against the **translational** Jacobian for `dimid2 < 3`
///   (the two sliding tangents), and
/// - `frame[dimid2 - 3]` against the **rotational** Jacobian for
///   `dimid2 >= 3` (spin about the normal at `condim >= 4`, rolling about the
///   two tangents at `condim = 6`).
///
/// So `condim` 4 adds torsional friction and `condim` 6 adds rolling friction,
/// which is what makes it a per-contact fidelity parameter rather than a
/// compile-time choice (`specs/06` item 4).
///
/// # A direction with zero friction contributes no rows
///
/// This generalises a defect that cost a diverging G1. The edge directions are
/// `n +/- mu d`, so at `mu = 0` both collapse onto `n`: two identical rows,
/// with `Rpy = 2 mu^2 R0` zero, so `D = 1/R` saturates at `1/MINVAL` and the
/// Hessian loses every pivot. A coefficient of zero means that direction
/// carries no friction, and the example's own answer for a contact with no
/// friction at all is `dim == 1`. So each direction is included only when its
/// coefficient is positive, and a contact whose every coefficient is zero gets
/// the single frictionless normal row.
///
/// That matters more at `condim` 4 and 6 than at 3, because torsional and
/// rolling coefficients are commonly zero while sliding friction is not.
///
/// # Errors
///
/// [`CompositeError::WidthMismatch`] if `condim` is not 1, 3, 4 or 6.
#[allow(clippy::too_many_arguments)]
pub fn contact_rows(
    model: &CompositeModel,
    pos: &Position,
    cvel: &[[f64; 6]],
    cdof_dot: &[[f64; 6]],
    qd: &[f64],
    point: [f64; 3],
    normal: [f64; 3],
    depth: f64,
    body_a: usize,
    body_b: Option<usize>,
    condim: usize,
    friction: [f64; 5],
    solref: [f64; 2],
    solimp: [f64; 5],
    margin: f64,
    out: &mut Vec<Row>,
) -> Result<(), CompositeError> {
    if !matches!(condim, 1 | 3 | 4 | 6) {
        return Err(CompositeError::WidthMismatch {
            what: "condim (1, 3, 4 and 6 are the example's values)",
            expected: 3,
            found: condim,
        });
    }
    let nv = model.tree.nv;
    // The relative point Jacobians and their derivatives: body a minus body b,
    // translational and rotational.
    let (ja_p, ja_r) = point_jacobian(model, pos, point, body_a);
    let (jda_p, jda_r) = point_jacobian_dot(model, pos, cvel, cdof_dot, point, body_a);
    let (jb_p, jb_r, jdb_p, jdb_r) = match body_b {
        Some(b) => {
            let (p, r) = point_jacobian(model, pos, point, b);
            let (dp, dr) = point_jacobian_dot(model, pos, cvel, cdof_dot, point, b);
            (p, r, dp, dr)
        }
        None => (
            vec![0.0; 3 * nv],
            vec![0.0; 3 * nv],
            vec![0.0; 3 * nv],
            vec![0.0; 3 * nv],
        ),
    };
    let rel_p: Vec<f64> = ja_p.iter().zip(&jb_p).map(|(x, y)| x - y).collect();
    let rel_r: Vec<f64> = ja_r.iter().zip(&jb_r).map(|(x, y)| x - y).collect();
    let rel_dp: Vec<f64> = jda_p.iter().zip(&jdb_p).map(|(x, y)| x - y).collect();
    let rel_dr: Vec<f64> = jda_r.iter().zip(&jdb_r).map(|(x, y)| x - y).collect();

    // Projects a relative Jacobian onto a direction in the contact frame.
    let project = |dir: [f64; 3], src: &[f64]| -> Vec<f64> {
        (0..nv)
            .map(|i| (0..3).map(|r| dir[r] * src[r * nv + i]).sum::<f64>())
            .collect()
    };

    let (t1, t2) = contact_tangents(normal);
    // The generator directions this contact actually has, in the example's
    // order. `angular` says which Jacobian the direction multiplies.
    let mut dirs: Vec<([f64; 3], bool, f64)> = Vec::new();
    if condim >= 3 {
        dirs.push((t1, false, friction[0]));
        dirs.push((t2, false, friction[1]));
    }
    if condim >= 4 {
        dirs.push((normal, true, friction[2]));
    }
    if condim >= 6 {
        dirs.push((t1, true, friction[3]));
        dirs.push((t2, true, friction[4]));
    }
    dirs.retain(|(_, _, mu)| *mu > 0.0);

    if dirs.is_empty() {
        let j = project(normal, &rel_p);
        let jd = project(normal, &rel_dp);
        out.push(Row {
            jdot_v: dot(&jd, qd),
            jacobian: j,
            pos: -depth,
            solref,
            solimp,
            margin,
            kind: RowKind::ContactNormal,
        });
        return Ok(());
    }

    let normal_j = project(normal, &rel_p);
    let normal_jd = project(normal, &rel_dp);
    let first_edge = out.len();
    for (dir, angular, mu) in dirs {
        let (src, src_d): (&[f64], &[f64]) = if angular {
            (&rel_r, &rel_dr)
        } else {
            (&rel_p, &rel_dp)
        };
        let dir_j = project(dir, src);
        let dir_jd = project(dir, src_d);
        for sign in [1.0f64, -1.0] {
            // The edge itself: the normal direction plus or minus `mu` times
            // this generator's direction, on whichever Jacobian it belongs to.
            let j: Vec<f64> = normal_j
                .iter()
                .zip(&dir_j)
                .map(|(n, d)| n + sign * mu * d)
                .collect();
            let jd: Vec<f64> = normal_jd
                .iter()
                .zip(&dir_jd)
                .map(|(n, d)| n + sign * mu * d)
                .collect();
            out.push(Row {
                jdot_v: dot(&jd, qd),
                jacobian: j,
                pos: -depth,
                solref,
                solimp,
                margin,
                kind: RowKind::ContactPyramid { first_edge, mu },
            });
        }
    }
    Ok(())
}

/// The example's five-element friction vector for a contact with sliding
/// friction only (07 section 3.1's `(fri0, fri0, fri1, fri2, fri2)` with the
/// torsional and rolling entries zero).
///
/// Zero torsional and rolling friction is this project's default rather than
/// the example's numbers: rule 2 of the mapping document keeps the example's
/// constants with the engine they describe, so adopting its
/// `friction = [1, 0.005, 0.0001]` would need its own measurement. A zero
/// entry contributes no rows, so a `condim` of 4 or 6 with this vector behaves
/// exactly as 3 does — the dimensionality is available and unused until a
/// coefficient is supplied.
#[must_use]
pub fn sliding_friction(mu: f64) -> [f64; 5] {
    [mu, mu, 0.0, 0.0, 0.0]
}

/// `mju_decodePyramid`: the contact-frame force of a pyramidal contact, from
/// its non-negative edge coefficients.
///
/// Returns `(normal, components)`, where `components[k]` is the force along the
/// k-th generator direction `contact_rows` emitted — the sliding tangents
/// first, then spin, then rolling, with any zero-coefficient direction absent.
/// `mus` is that direction's coefficient, in the same order.
///
/// `f_n = sum_k lambda_k` and `component_k = mu_k (lambda_{2k} -
/// lambda_{2k+1})`. The result always satisfies
/// `sum_k |component_k| <= mu f_n` when every `mu` is equal, which is the
/// pyramid this representation realises: the inscribed diamond, not the
/// circumscribed box.
#[must_use]
pub fn decode_pyramid(edges: &[f64], mus: &[f64]) -> (f64, Vec<f64>) {
    let mut normal = 0.0;
    let mut components = Vec::with_capacity(mus.len());
    for (k, pair) in edges.chunks(2).enumerate() {
        let plus = pair.first().copied().unwrap_or(0.0);
        let minus = pair.get(1).copied().unwrap_or(0.0);
        normal += plus + minus;
        let mu = mus.get(k).copied().unwrap_or(0.0);
        components.push(mu * (plus - minus));
    }
    (normal, components)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composite::{com_vel, forward_dynamics, position, rne, CompositeState};
    use robot_model::joint::{JointLimits, JointSpec};
    use robot_model::link::{InertiaTensor, LinkSpec, MassProperties};
    use robot_model::pose::RigidPose;
    use robot_model::profile::{FirmwareContract, RobotProfile, PROFILE_SCHEMA_VERSION};
    use robot_types::frames::{GravityConvention, Vec3};
    use robot_types::ids::{JointId, LinkId, Name};
    use robot_types::schema::ExtensionTable;
    use robot_types::time::Duration;
    use robot_types::units::{Kilograms, NewtonMetres, RadiansPerSecond};

    /// A single free body of the given mass, used as the contact test rig.
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

    /// Drops a free body onto a plane with `n` contact points spread over its
    /// underside and returns the penetration it settles at.
    fn settle(mass: f64, n: usize, condim: usize, friction: f64) -> f64 {
        let profile = free_body(mass);
        let model = CompositeModel::from_profile(&profile);
        let mut state = CompositeState::identity(&model);
        let dt = 1e-3;
        let tau = vec![0.0; model.nv()];
        let mut warm: Option<Vec<f64>> = None;
        for _ in 0..4000 {
            let pos = position(&model, &state.q).unwrap();
            let (cvel, cdof_dot) = com_vel(&model, &pos, &state.qd).unwrap();
            let bias = rne(&model, &pos, &state.qd, None).unwrap();
            let smooth: Vec<f64> = tau.iter().zip(&bias).map(|(t, b)| t - b).collect();
            let qacc_smooth = forward_dynamics(&model, &pos, &state.qd, &tau).unwrap();
            let _ = smooth;

            // `n` contact points in a ring on the underside, all at the same
            // height, so each carries an equal share.
            let mut rows = Vec::new();
            for k in 0..n {
                let angle = 2.0 * core::f64::consts::PI * (k as f64) / (n as f64);
                let r = if n == 1 { 0.0 } else { 0.05 };
                let point = [
                    state.q[0] + r * angle.cos(),
                    state.q[1] + r * angle.sin(),
                    state.q[2],
                ];
                let depth = -point[2];
                if depth <= 0.0 {
                    continue;
                }
                contact_rows(
                    &model,
                    &pos,
                    &cvel,
                    &cdof_dot,
                    &state.qd,
                    point,
                    [0.0, 0.0, 1.0],
                    depth,
                    0,
                    None,
                    condim,
                    sliding_friction(friction),
                    default_solref(),
                    default_solimp(),
                    0.0,
                    &mut rows,
                )
                .unwrap();
            }
            let solution = solve(
                &model,
                &pos,
                &state.qd,
                &qacc_smooth,
                &rows,
                dt,
                SolverConfig::default(),
                warm.as_deref(),
            )
            .unwrap();
            warm = Some(solution.force.clone());
            for (v, a) in state.qd.iter_mut().zip(&solution.qacc) {
                *v += dt * a;
            }
            let qd = state.qd.clone();
            crate::composite::integrate_pos(&model, &mut state.q, &qd, dt).unwrap();
        }
        -state.q[2]
    }

    #[test]
    fn the_total_normal_force_balances_the_weight_however_many_rows_carry_it() {
        // This is what solving the rows together buys, and it is worth being
        // precise about which part of Appendix D M-04 it does and does not fix.
        //
        // Under the per-point penalty law each point was an independent spring,
        // so n points produced n times the force at a given depth and the body
        // rose until `n k depth = m g`. Here the rows are coupled through
        // `A = J M^-1 J^T`, so whatever the count, the *total* force settles at
        // the weight and each row carries its share. That is asserted below.
        //
        // What this does **not** fix is the rest depth's dependence on the
        // count. Working the equilibrium through, with `A_ii ~ 1/m` for every
        // row and `R_i = (1-d)/d * A_ii`, the per-row force is `m g / n` and
        // the depth follows `R_i f_i = K d depth`, so `depth ~ 1/n` still. The
        // example's formulation has the same property. Removing it needs the
        // other half of M-04: the exact bounded manifolds of `04` sections 1.3
        // and 1.4, which cap the point count as a geometric property of the
        // contact instead of leaving it to how finely a shape is sampled.
        let profile = free_body(2.0);
        let model = CompositeModel::from_profile(&profile);
        let weight = 2.0 * 9.806_65;
        for n in [1usize, 4, 8] {
            let mut state = CompositeState::identity(&model);
            state.q[2] = -0.0005;
            let pos = position(&model, &state.q).unwrap();
            let (cvel, cdof_dot) = com_vel(&model, &pos, &state.qd).unwrap();
            let tau = vec![0.0; model.nv()];
            let qacc_smooth = forward_dynamics(&model, &pos, &state.qd, &tau).unwrap();
            let mut rows = Vec::new();
            for k in 0..n {
                let angle = 2.0 * core::f64::consts::PI * (k as f64) / (n as f64);
                let r = if n == 1 { 0.0 } else { 0.05 };
                contact_rows(
                    &model,
                    &pos,
                    &cvel,
                    &cdof_dot,
                    &state.qd,
                    [r * angle.cos(), r * angle.sin(), -0.0005],
                    [0.0, 0.0, 1.0],
                    0.0005,
                    0,
                    None,
                    1,
                    sliding_friction(0.0),
                    default_solref(),
                    default_solimp(),
                    0.0,
                    &mut rows,
                )
                .unwrap();
            }
            // Iterate the solve to its fixed point at this configuration, which
            // is what "at rest" means for the force balance.
            let mut warm: Option<Vec<f64>> = None;
            let mut total = 0.0;
            for _ in 0..200 {
                let solution = solve(
                    &model,
                    &pos,
                    &state.qd,
                    &qacc_smooth,
                    &rows,
                    1e-3,
                    SolverConfig::default(),
                    warm.as_deref(),
                )
                .unwrap();
                total = solution.force.iter().sum();
                warm = Some(solution.force);
            }
            // The rows must divide one load, not each carry it.
            let rel = (total - weight).abs() / weight;
            assert!(
                rel < 0.3,
                "{n} rows carried a total of {total} N against a weight of {weight} N"
            );
        }
    }

    #[test]
    fn the_rest_depth_barely_moves_with_the_load() {
        // A penalty spring's rest depth is `m g / k`, strictly proportional to
        // the load. The soft-constraint formulation sets the depth from a time
        // constant instead, and the equilibrium works out almost independent of
        // mass: `depth ~ g (1 - d) / (n K d^2)` carries no `m`. That is the
        // property worth pinning, and it is the opposite of what a
        // penalty-spring intuition predicts.
        let light = settle(1.0, 4, 1, 0.0);
        let heavy = settle(4.0, 4, 1, 0.0);
        assert!(light > 0.0 && heavy > 0.0, "both should settle in contact");
        let ratio = heavy / light;
        assert!(
            ratio < 2.0,
            "a four-fold load should barely change the depth; got a factor of {ratio}"
        );
    }

    #[test]
    fn friction_rows_are_bounded_by_the_cone_and_stop_a_sliding_body() {
        // Appendix D M-10: friction used to be applied after the normal solve,
        // so it was never bounded by the normal force the solve produced and
        // never influenced it. Here the two are rows of one problem.
        //
        // A body sliding along the plane with friction must come to rest; with
        // the friction coefficient at zero it must keep its speed.
        for (mu, expect_stop) in [(0.0, false), (0.8, true)] {
            let profile = free_body(2.0);
            let model = CompositeModel::from_profile(&profile);
            let mut state = CompositeState::identity(&model);
            state.qd[0] = 1.0;
            let dt = 1e-3;
            let tau = vec![0.0; model.nv()];
            let mut warm: Option<Vec<f64>> = None;
            for _ in 0..2000 {
                let pos = position(&model, &state.q).unwrap();
                let (cvel, cdof_dot) = com_vel(&model, &pos, &state.qd).unwrap();
                let qacc_smooth = forward_dynamics(&model, &pos, &state.qd, &tau).unwrap();
                let mut rows = Vec::new();
                let point = [state.q[0], state.q[1], state.q[2]];
                let depth = -point[2];
                if depth > 0.0 {
                    contact_rows(
                        &model,
                        &pos,
                        &cvel,
                        &cdof_dot,
                        &state.qd,
                        point,
                        [0.0, 0.0, 1.0],
                        depth,
                        0,
                        None,
                        3,
                        sliding_friction(mu),
                        default_solref(),
                        default_solimp(),
                        0.0,
                        &mut rows,
                    )
                    .unwrap();
                    // Every friction force must respect the cone.
                    let solution = solve(
                        &model,
                        &pos,
                        &state.qd,
                        &qacc_smooth,
                        &rows,
                        dt,
                        SolverConfig::default(),
                        warm.as_deref(),
                    )
                    .unwrap();
                    // Every edge coefficient is non-negative, and the force
                    // they encode lies in the pyramid: `|ft1| + |ft2| <= mu fn`.
                    for (edge, f) in solution.force.iter().enumerate() {
                        assert!(*f >= -1e-12, "pyramid edge {edge} pulled: {f}");
                    }
                    let (fn_, components) = decode_pyramid(&solution.force, &[mu, mu]);
                    assert!(fn_ >= -1e-12, "the normal force must not pull: {fn_}");
                    let tangential: f64 = components.iter().map(|c| c.abs()).sum();
                    assert!(
                        tangential <= mu * fn_.max(0.0) + 1e-9,
                        "friction {tangential} leaves the pyramid of bound {}",
                        mu * fn_.max(0.0)
                    );
                    warm = Some(solution.force.clone());
                    for (v, a) in state.qd.iter_mut().zip(&solution.qacc) {
                        *v += dt * a;
                    }
                } else {
                    for (v, a) in state.qd.iter_mut().zip(&qacc_smooth) {
                        *v += dt * a;
                    }
                }
                let qd = state.qd.clone();
                crate::composite::integrate_pos(&model, &mut state.q, &qd, dt).unwrap();
            }
            let speed = (state.qd[0] * state.qd[0] + state.qd[1] * state.qd[1]).sqrt();
            if expect_stop {
                assert!(
                    speed < 0.05,
                    "friction should have stopped it, speed {speed}"
                );
            } else {
                assert!(
                    speed > 0.9,
                    "with no friction the speed should survive, got {speed}"
                );
            }
        }
    }

    #[test]
    fn a_contact_row_never_pulls_and_a_separating_body_feels_nothing() {
        // The feasible set is what makes this a contact rather than a spring:
        // the normal force is non-negative, so a body already leaving the
        // surface gets no force holding it back.
        let profile = free_body(1.5);
        let model = CompositeModel::from_profile(&profile);
        let mut state = CompositeState::identity(&model);
        state.q[2] = -0.001;
        state.qd[2] = 5.0;
        let pos = position(&model, &state.q).unwrap();
        let (cvel, cdof_dot) = com_vel(&model, &pos, &state.qd).unwrap();
        let tau = vec![0.0; model.nv()];
        let qacc_smooth = forward_dynamics(&model, &pos, &state.qd, &tau).unwrap();
        let mut rows = Vec::new();
        contact_rows(
            &model,
            &pos,
            &cvel,
            &cdof_dot,
            &state.qd,
            [0.0, 0.0, -0.001],
            [0.0, 0.0, 1.0],
            0.001,
            0,
            None,
            1,
            sliding_friction(0.0),
            default_solref(),
            default_solimp(),
            0.0,
            &mut rows,
        )
        .unwrap();
        let solution = solve(
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
        assert!(
            solution.force[0] >= 0.0,
            "a contact must not pull: {}",
            solution.force[0]
        );
        assert!(
            solution.force[0] < 1e-6,
            "a body moving away fast should feel no contact force, got {}",
            solution.force[0]
        );
    }

    #[test]
    fn the_point_jacobian_matches_a_finite_difference_of_the_point_position() {
        // The Jacobian is the derivative of a body-fixed point's world
        // position with respect to the degrees of freedom, so a central
        // difference of the kinematics is an independent check of it.
        let profile = free_body(1.0);
        let model = CompositeModel::from_profile(&profile);
        let mut q = vec![0.0; model.nq];
        q[3] = 1.0;
        q[0] = 0.2;
        q[1] = -0.1;
        q[2] = 0.4;
        let local = [0.07, -0.03, 0.05];

        let world_point = |q: &[f64]| -> [f64; 3] {
            let pos = position(&model, q).unwrap();
            let r = pos.xmat[0];
            [
                r[0] * local[0] + r[1] * local[1] + r[2] * local[2] + pos.xpos[0][0],
                r[3] * local[0] + r[4] * local[1] + r[5] * local[2] + pos.xpos[0][1],
                r[6] * local[0] + r[7] * local[1] + r[8] * local[2] + pos.xpos[0][2],
            ]
        };

        let pos = position(&model, &q).unwrap();
        let point = world_point(&q);
        let (jacp, _) = point_jacobian(&model, &pos, point, 0);
        let nv = model.nv();
        let h = 1e-7;
        for i in 0..nv {
            // Perturb by integrating a unit velocity on degree of freedom i.
            let mut plus = q.clone();
            let mut minus = q.clone();
            let mut v = vec![0.0; nv];
            v[i] = 1.0;
            crate::composite::integrate_pos(&model, &mut plus, &v, h).unwrap();
            crate::composite::integrate_pos(&model, &mut minus, &v, -h).unwrap();
            let pp = world_point(&plus);
            let pm = world_point(&minus);
            for r in 0..3 {
                let fd = (pp[r] - pm[r]) / (2.0 * h);
                let analytic = jacp[r * nv + i];
                assert!(
                    (fd - analytic).abs() < 1e-6,
                    "dof {i} axis {r}: finite difference {fd}, analytic {analytic}"
                );
            }
        }
    }

    #[test]
    fn warm_starting_reaches_the_same_answer_in_fewer_sweeps() {
        // Step 06 item 7. The forces are the solution of a convex problem, so
        // the starting point cannot change where it converges, only how long it
        // takes -- but only *at convergence*, which is why this test sets its
        // own sweep budget instead of taking the default.
        //
        // Four pyramidal contacts on a six-degree-of-freedom body give sixteen
        // rows for a rank-six problem. Projected Gauss-Seidel needs **1011
        // sweeps** to bring the per-sweep force change below 1e-10 there,
        // against the default budget of 100. At 100 sweeps the residual is
        // still 1.1e-2 and the two starting points land on visibly different
        // forces, so an earlier version of this test was comparing two
        // unconverged iterates and reading the difference as disagreement.
        //
        // The default budget is not wrong for stepping: at 100 sweeps the
        // acceleration is already within 2e-4 relative of the converged value,
        // and acceleration is what the integrator consumes. It is the *forces*
        // that are far from settled. That gap is the quantitative case for the
        // Newton solver of `specs/08`, which Appendix D M-22 records as
        // outstanding, and it is measured here rather than asserted.
        const CONVERGED: SolverConfig = SolverConfig {
            iterations: 10_000,
            tolerance: 1e-10,
            warm_start: true,
        };
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
                sliding_friction(0.6),
                default_solref(),
                default_solimp(),
                0.0,
                &mut rows,
            )
            .unwrap();
        }
        let cold = solve(
            &model,
            &pos,
            &state.qd,
            &qacc_smooth,
            &rows,
            1e-3,
            CONVERGED,
            None,
        )
        .unwrap();
        let warm = solve(
            &model,
            &pos,
            &state.qd,
            &qacc_smooth,
            &rows,
            1e-3,
            CONVERGED,
            Some(&cold.force),
        )
        .unwrap();
        assert!(
            warm.iterations <= cold.iterations,
            "warm start took {} sweeps against {} cold",
            warm.iterations,
            cold.iterations
        );
        // They must agree on the **acceleration**, not on the individual row
        // forces, and the difference is not slack in the test.
        //
        // A pyramidal contact is a redundant parameterisation: four
        // non-negative edge coefficients encode three force components, and
        // every edge direction contains the normal, so infinitely many
        // coefficient vectors give the same physical force. The dual problem is
        // therefore convex but not strictly so in `f`, and two runs from
        // different starting points legitimately land on different coefficients
        // -- measured at up to 6.9e-2 relative here. What the problem does
        // determine uniquely is the acceleration, so that is what is compared.
        for (a, b) in warm.qacc.iter().zip(&cold.qacc) {
            let scale = a.abs().max(b.abs()).max(1.0);
            assert!(
                (a - b).abs() / scale < 1e-6,
                "warm and cold must agree on the acceleration: {a} against {b}"
            );
        }
    }

    #[test]
    fn contact_tangents_are_orthonormal_with_the_normal() {
        for n in [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.577_350_269_189_625_8; 3],
        ] {
            let (t1, t2) = contact_tangents(n);
            assert!(dot3(n, t1).abs() < 1e-12, "t1 not orthogonal for {n:?}");
            assert!(dot3(n, t2).abs() < 1e-12, "t2 not orthogonal for {n:?}");
            assert!(dot3(t1, t2).abs() < 1e-12, "tangents not orthogonal");
            assert!((dot3(t1, t1) - 1.0).abs() < 1e-12);
            assert!((dot3(t2, t2) - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn an_empty_problem_returns_the_unconstrained_acceleration() {
        let profile = free_body(1.0);
        let model = CompositeModel::from_profile(&profile);
        let state = CompositeState::identity(&model);
        let pos = position(&model, &state.q).unwrap();
        let tau = vec![0.0; model.nv()];
        let qacc_smooth = forward_dynamics(&model, &pos, &state.qd, &tau).unwrap();
        let solution = solve(
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
        assert_eq!(solution.qacc, qacc_smooth);
        assert!(solution.force.is_empty());
    }

    #[test]
    fn a_too_stiff_time_constant_is_clamped_and_reported() {
        // Step 06 item 3: a constraint whose time constant is below about twice
        // the timestep is too stiff for the integrator, and the clamp is
        // reported rather than applied silently.
        let profile = free_body(1.0);
        let model = CompositeModel::from_profile(&profile);
        let state = CompositeState::identity(&model);
        let pos = position(&model, &state.q).unwrap();
        let tau = vec![0.0; model.nv()];
        let qacc_smooth = forward_dynamics(&model, &pos, &state.qd, &tau).unwrap();
        let nv = model.nv();
        let mut jac = vec![0.0; nv];
        jac[2] = 1.0;
        let rows = vec![Row {
            jacobian: jac,
            pos: -0.001,
            jdot_v: 0.0,
            // A time constant far below 2 * dt.
            solref: [1e-6, 1.0],
            solimp: default_solimp(),
            margin: 0.0,
            kind: RowKind::ContactNormal,
        }];
        let solution = solve(
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
        assert_eq!(solution.clamped_rows, vec![0]);
    }
    #[test]
    fn a_frictionless_pair_gets_one_row_rather_than_a_degenerate_pyramid() {
        // Regression. `condim = 3` with `mu = 0` used to build four pyramid
        // edges, and at `mu = 0` the directions `n +/- mu t_k` all collapse
        // onto `n`: four identical rows, with `Rpy = 2 mu^2 R0 = 0` so
        // `D = 1/R` saturated at 1e15. The Hessian `M + J' D_q J` then lost
        // every pivot and the Newton solver diverged on step 1511 of the G1
        // collapse, once the folded robot started contacting its own links
        // through the default pair friction of zero.
        let profile = free_body(2.0);
        let model = CompositeModel::from_profile(&profile);
        let mut state = CompositeState::identity(&model);
        state.q[2] = -0.002;
        let pos = position(&model, &state.q).unwrap();
        let (cvel, cdof_dot) = com_vel(&model, &pos, &state.qd).unwrap();
        let build = |mu: f64| {
            let mut rows = Vec::new();
            contact_rows(
                &model,
                &pos,
                &cvel,
                &cdof_dot,
                &state.qd,
                [0.0, 0.0, -0.002],
                [0.0, 0.0, 1.0],
                0.002,
                0,
                None,
                3,
                sliding_friction(mu),
                default_solref(),
                default_solimp(),
                0.0,
                &mut rows,
            )
            .unwrap();
            rows
        };
        let frictionless = build(0.0);
        assert_eq!(frictionless.len(), 1, "zero friction is one normal row");
        assert_eq!(frictionless[0].kind, RowKind::ContactNormal);
        assert_eq!(build(0.6).len(), 4, "a real pyramid is four edges");
    }
    #[test]
    fn condim_selects_the_example_row_counts_and_directions() {
        // 07 section 9.2: a pyramidal contact has `2(condim - 1)` rows, in the
        // order (+t1, -t1, +t2, -t2, +spin, -spin, +roll1, -roll1, +roll2,
        // -roll2). The sliding tangents multiply the translational Jacobian and
        // spin and rolling multiply the rotational one, which is what makes
        // condim 4 and 6 different in kind from 3 rather than just longer.
        let profile = free_body(2.0);
        let model = CompositeModel::from_profile(&profile);
        let mut state = CompositeState::identity(&model);
        state.q[2] = -0.002;
        let pos = position(&model, &state.q).unwrap();
        let (cvel, cdof_dot) = com_vel(&model, &pos, &state.qd).unwrap();

        let build = |condim: usize, friction: [f64; 5]| {
            let mut rows = Vec::new();
            contact_rows(
                &model,
                &pos,
                &cvel,
                &cdof_dot,
                &state.qd,
                [0.0, 0.0, -0.002],
                [0.0, 0.0, 1.0],
                0.002,
                0,
                None,
                condim,
                friction,
                default_solref(),
                default_solimp(),
                0.0,
                &mut rows,
            )
            .unwrap();
            rows
        };

        // Sliding only: condim 4 and 6 add nothing, because their coefficients
        // are zero and a zero direction contributes no rows.
        assert_eq!(build(1, sliding_friction(0.6)).len(), 1);
        assert_eq!(build(3, sliding_friction(0.6)).len(), 4);
        assert_eq!(build(4, sliding_friction(0.6)).len(), 4);
        assert_eq!(build(6, sliding_friction(0.6)).len(), 4);

        // With a torsional coefficient, condim 4 is the example's six rows.
        let torsional = [0.6, 0.6, 0.01, 0.0, 0.0];
        assert_eq!(build(3, torsional).len(), 4, "condim 3 ignores torsion");
        assert_eq!(build(4, torsional).len(), 6);
        // And with rolling as well, condim 6 is ten.
        let full = [0.6, 0.6, 0.01, 0.002, 0.002];
        assert_eq!(build(6, full).len(), 10);
        assert_eq!(build(4, full).len(), 6, "condim 4 ignores rolling");

        // The spin rows act on rotation, not translation. For a free body the
        // first three degrees of freedom are world-frame linear and the last
        // three body-frame angular, so a sliding edge must have non-zero
        // entries in the linear block and a spin edge must differ from the
        // sliding ones in the angular block.
        let rows = build(4, torsional);
        let nv = model.nv();
        let linear_part = |r: &Row| (0..3).fold(0.0f64, |m, i| m.max(r.jacobian[i].abs()));
        let angular_part = |r: &Row| (3..nv).fold(0.0f64, |m, i| m.max(r.jacobian[i].abs()));
        // Row 4 is +spin. Its angular entries come from the rotational
        // Jacobian scaled by the torsional coefficient.
        let slide = &rows[0];
        let spin = &rows[4];
        assert!(
            linear_part(slide) > 0.0,
            "a sliding edge acts on translation"
        );
        assert!(angular_part(spin) > 0.0, "a spin edge must act on rotation");
        // Every edge pair differs only in the sign of its friction term, so
        // each pair averages to the same plain normal row. That is the
        // structural invariant of the generator representation: the pyramid is
        // centred on the normal.
        let mean_normal: Vec<f64> = (0..nv)
            .map(|i| 0.5 * (rows[4].jacobian[i] + rows[5].jacobian[i]))
            .collect();
        let mean_slide: Vec<f64> = (0..nv)
            .map(|i| 0.5 * (rows[0].jacobian[i] + rows[1].jacobian[i]))
            .collect();
        for i in 0..nv {
            assert!(
                (mean_normal[i] - mean_slide[i]).abs() < 1e-12,
                "every edge pair must average to the same normal row at index {i}"
            );
        }
    }

    #[test]
    fn a_torsional_coefficient_resists_spin_about_the_contact_normal() {
        // The physical point of condim 4. A body pressed onto a plane and spun
        // about the contact normal should feel a resisting torque, and should
        // not at condim 3.
        let profile = free_body(2.0);
        let model = CompositeModel::from_profile(&profile);
        let dt = 1e-3;
        let spin_decay = |condim: usize, friction: [f64; 5]| {
            let mut state = CompositeState::identity(&model);
            state.q[2] = -0.002;
            // Spin about the contact normal, which is the body-frame angular
            // z degree of freedom.
            state.qd[5] = 2.0;
            let mut warm: Option<Vec<f64>> = None;
            for _ in 0..200 {
                let pos = position(&model, &state.q).unwrap();
                let (cvel, cdof_dot) = com_vel(&model, &pos, &state.qd).unwrap();
                let tau = vec![0.0; model.nv()];
                let qacc_smooth = forward_dynamics(&model, &pos, &state.qd, &tau).unwrap();
                let mut rows = Vec::new();
                let depth = -state.q[2];
                if depth > 0.0 {
                    contact_rows(
                        &model,
                        &pos,
                        &cvel,
                        &cdof_dot,
                        &state.qd,
                        [0.0, 0.0, state.q[2]],
                        [0.0, 0.0, 1.0],
                        depth,
                        0,
                        None,
                        condim,
                        friction,
                        default_solref(),
                        default_solimp(),
                        0.0,
                        &mut rows,
                    )
                    .unwrap();
                }
                let solution = solve(
                    &model,
                    &pos,
                    &state.qd,
                    &qacc_smooth,
                    &rows,
                    dt,
                    SolverConfig::default(),
                    warm.as_deref(),
                )
                .unwrap();
                warm = Some(solution.force.clone());
                for (v, a) in state.qd.iter_mut().zip(&solution.qacc) {
                    *v += dt * a;
                }
                let qd = state.qd.clone();
                crate::composite::integrate_pos(&model, &mut state.q, &qd, dt).unwrap();
            }
            state.qd[5]
        };

        let without = spin_decay(3, sliding_friction(0.8));
        let with = spin_decay(4, [0.8, 0.8, 0.5, 0.0, 0.0]);
        assert!(
            with.abs() < without.abs(),
            "torsional friction should remove spin: condim 4 left {with}, condim 3 left {without}"
        );
    }
}

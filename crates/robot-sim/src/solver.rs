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
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum RowKind {
    /// Equality: the force is unbounded in both directions.
    Equality,
    /// A joint limit: the force may only push inward, so it is non-negative.
    Limit,
    /// A contact normal: non-negative, and it bounds the friction rows that
    /// name it.
    ContactNormal,
    /// A contact friction direction, bounded by the pyramidal cone
    /// `|f| <= mu * f_normal`.
    ContactFriction {
        /// Index of the normal row this direction belongs to.
        normal_row: usize,
        /// Friction coefficient for this direction.
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
) -> Vec<f64> {
    let nv = model.tree.nv;
    let mut jacp = vec![0.0; 3 * nv];
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
        }
        let p = model.tree.dof_parentid[d];
        i = if p < 0 { None } else { Some(p as usize) };
    }
    jacp
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

    // Per-row impedance, regularisation and reference acceleration.
    let mut r = vec![0.0; m];
    let mut aref = vec![0.0; m];
    for (i, row) in rows.iter().enumerate() {
        let (tc, clamped) = effective_timeconst(row.solref, timestep);
        if clamped {
            solution.clamped_rows.push(i);
        }
        let solref = [tc, row.solref[1]];
        let d = impedance(row.solimp, row.pos, row.margin);
        // Section 2.4: R_ii = (1 - d)/d * A_ii, with the exact diagonal.
        // Appendix D M-09 records the constant that stood here before.
        let diag = a[i * m + i];
        r[i] = (((1.0 - d) / d.max(MINVAL)) * diag).max(MINVAL);
        let (k, b) = stiffness_damping(solref, row.solimp);
        let vel = dot(&row.jacobian, qd);
        // a_ref = -B vel - K d (pos - margin).
        aref[i] = -b * vel - k * d * (row.pos - row.margin);
    }

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
                RowKind::Limit | RowKind::ContactNormal => unprojected.max(0.0),
                RowKind::ContactFriction { normal_row, mu } => {
                    // The pyramidal cone: this direction is bounded by the
                    // current normal force. Using the current value rather than
                    // a fixed one is what couples friction to the normal solve,
                    // which Appendix D M-10 records as absent.
                    let bound = (mu * f[normal_row]).max(0.0);
                    unprojected.clamp(-bound, bound)
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
/// `depth` is the penetration, positive when the surfaces overlap, so the
/// residual is `-depth`. `point` is the contact position in world coordinates.
/// Pass `body_b = None` for a contact against a static environment.
///
/// At `condim = 1` only the normal row is produced; at `condim = 3` two
/// friction rows follow it, bounded by the pyramid.
///
/// # Errors
///
/// [`CompositeError::WidthMismatch`] if `condim` is not 1 or 3.
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
    friction: f64,
    solref: [f64; 2],
    solimp: [f64; 5],
    margin: f64,
    out: &mut Vec<Row>,
) -> Result<(), CompositeError> {
    if condim != 1 && condim != 3 {
        return Err(CompositeError::WidthMismatch {
            what: "condim (only 1 and 3 are implemented)",
            expected: 3,
            found: condim,
        });
    }
    let nv = model.tree.nv;
    // The relative point Jacobian: body a minus body b.
    let (ja, _) = point_jacobian(model, pos, point, body_a);
    let jda = point_jacobian_dot(model, pos, cvel, cdof_dot, point, body_a);
    let (jb, jdb) = match body_b {
        Some(b) => (
            point_jacobian(model, pos, point, b).0,
            point_jacobian_dot(model, pos, cvel, cdof_dot, point, b),
        ),
        None => (vec![0.0; 3 * nv], vec![0.0; 3 * nv]),
    };
    let rel: Vec<f64> = ja.iter().zip(&jb).map(|(x, y)| x - y).collect();
    let rel_dot: Vec<f64> = jda.iter().zip(&jdb).map(|(x, y)| x - y).collect();

    // Project the relative Jacobian onto a contact-frame direction.
    let project = |dir: [f64; 3], src: &[f64]| -> Vec<f64> {
        (0..nv)
            .map(|i| (0..3).map(|r| dir[r] * src[r * nv + i]).sum::<f64>())
            .collect()
    };
    // `normal` points from b toward a, and a positive normal force must push
    // them apart, so the row measures the approach along it.
    let normal_row = out.len();
    let jn = project(normal, &rel);
    let jn_dot = project(normal, &rel_dot);
    let jdot_v_n = dot(&jn_dot, qd);
    out.push(Row {
        jacobian: jn,
        pos: -depth,
        jdot_v: jdot_v_n,
        solref,
        solimp,
        margin,
        kind: RowKind::ContactNormal,
    });
    if condim == 3 {
        let (t1, t2) = contact_tangents(normal);
        for t in [t1, t2] {
            let jt = project(t, &rel);
            let jt_dot = project(t, &rel_dot);
            out.push(Row {
                jacobian: jt,
                // Friction rows have no residual and no stiffness
                // (section 2.4: "friction loss and elliptic friction rows have
                // r = 0, K = 0").
                pos: 0.0,
                jdot_v: dot(&jt_dot, qd),
                solref,
                solimp,
                margin: 0.0,
                kind: RowKind::ContactFriction {
                    normal_row,
                    mu: friction,
                },
            });
        }
    }
    Ok(())
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
                    0.0,
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
                        mu,
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
                    let fn_ = solution.force[0];
                    assert!(fn_ >= -1e-12, "the normal force must not pull: {fn_}");
                    for ft in &solution.force[1..] {
                        assert!(
                            ft.abs() <= mu * fn_.max(0.0) + 1e-9,
                            "friction {ft} exceeds the cone bound {}",
                            mu * fn_.max(0.0)
                        );
                    }
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
            0.0,
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
        // takes.
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
        let cold = solve(
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
        let warm = solve(
            &model,
            &pos,
            &state.qd,
            &qacc_smooth,
            &rows,
            1e-3,
            SolverConfig::default(),
            Some(&cold.force),
        )
        .unwrap();
        assert!(
            warm.iterations <= cold.iterations,
            "warm start took {} sweeps against {} cold",
            warm.iterations,
            cold.iterations
        );
        // They must agree to within what the solver claims, not to machine
        // precision. The stopping rule bounds the force *change* per sweep, so
        // two runs from different starting points land at two nearby points of
        // the same convex problem; measured here at 2.6e-5 relative.
        for (a, b) in warm.force.iter().zip(&cold.force) {
            let scale = a.abs().max(b.abs()).max(1.0);
            assert!(
                (a - b).abs() / scale < 1e-3,
                "warm and cold must agree to the solver's tolerance: {a} against {b}"
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
}

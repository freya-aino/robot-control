//! Analytic velocity derivative of the bias forces — the term that turns
//! `implicitfast` into full `implicit` (example 11, `mjd_rne_vel`).
//!
//! `qfrc_bias(qvel)` is the recursive-Newton-Euler force at zero
//! acceleration: centripetal, Coriolis and gyroscopic terms (gravity is
//! velocity-independent). Its derivative `∂qfrc_bias/∂qvel` is computed
//! here by the example's own **dense variant** (`mjd_rne_vel_dense`,
//! example 11 section 2.4), which that document states is mathematically
//! identical to the sparse production recursion with a different rounding
//! order — the right fit for this crate's dense per-body arrays and
//! its at-most-35 degrees of freedom.
//!
//! The recursion differentiates each stage of the zero-acceleration RNE
//! of [`crate::composite`]: `Dcvel` (per body, 6 x nv) through the joint
//! model, `Dcdofdot` through the motion cross product, `Dcacc` forward,
//! `Dcfrcbody` through the composite inertia and the force cross product,
//! a backward accumulation over children, and a final projection through
//! each degree of freedom's `cdof`. The 6 x 6 derivative blocks
//! (sections 1.2, 1.3, 1.4 and 1.5 of example 11) are spelled out
//! entry for entry.
//!
//! Purity contract: `std` and plain arrays, like the rest of the dynamics
//! core (FR-6).

use crate::composite::{mul_inert_vec, CompositeError, CompositeModel, Position};
use robot_model::joint::JointKind;

/// `mjd_crossMotion_vel` (11 section 1.2): `∂(vel x_m v)/∂vel`, row-major
/// 6 x 6.
#[must_use]
pub fn cross_motion_vel_mat(v: [f64; 6]) -> [f64; 36] {
    let mut d = [0.0; 36];
    d[2] = -v[1];
    d[1] = v[2];
    d[8] = v[0];
    d[6] = -v[2];
    d[13] = -v[0];
    d[12] = v[1];
    d[20] = -v[4];
    d[19] = v[5];
    d[23] = -v[1];
    d[22] = v[2];
    d[26] = v[3];
    d[24] = -v[5];
    d[29] = v[0];
    d[27] = -v[2];
    d[31] = -v[3];
    d[30] = v[4];
    d[34] = -v[0];
    d[33] = v[1];
    d
}

/// `mjd_crossForce_vel` (11 section 1.3): `∂(vel x* f)/∂vel`.
#[must_use]
pub fn cross_force_vel_mat(f: [f64; 6]) -> [f64; 36] {
    let mut d = [0.0; 36];
    d[2] = -f[1];
    d[1] = f[2];
    d[5] = -f[4];
    d[4] = f[5];
    d[8] = f[0];
    d[6] = -f[2];
    d[11] = f[3];
    d[9] = -f[5];
    d[13] = -f[0];
    d[12] = f[1];
    d[16] = -f[3];
    d[15] = f[4];
    d[20] = -f[4];
    d[19] = f[5];
    d[26] = f[3];
    d[24] = -f[5];
    d[31] = -f[3];
    d[30] = f[4];
    d
}

/// `mjd_crossForce_frc` (11 section 1.4): `∂(vel x* f)/∂f`.
#[must_use]
pub fn cross_force_frc_mat(v: [f64; 6]) -> [f64; 36] {
    let mut d = [0.0; 36];
    d[1] = -v[2];
    d[2] = v[1];
    d[4] = -v[5];
    d[5] = v[4];
    d[6] = v[2];
    d[8] = -v[0];
    d[9] = v[5];
    d[11] = -v[3];
    d[12] = -v[1];
    d[13] = v[0];
    d[15] = -v[4];
    d[16] = v[3];
    d[22] = -v[2];
    d[23] = v[1];
    d[27] = v[2];
    d[29] = -v[0];
    d[33] = -v[1];
    d[34] = v[0];
    d
}

/// `mjd_mulInertVec_vel` (11 section 1.5): `∂(cinert . v)/∂v`.
#[must_use]
pub fn mul_inert_mat(i: &[f64; 10]) -> [f64; 36] {
    let mut d = [0.0; 36];
    d[0] = i[0];
    d[1] = i[3];
    d[2] = i[4];
    d[4] = -i[8];
    d[5] = i[7];
    d[6] = i[3];
    d[7] = i[1];
    d[8] = i[5];
    d[9] = i[8];
    d[11] = -i[6];
    d[12] = i[4];
    d[13] = i[5];
    d[14] = i[2];
    d[15] = -i[7];
    d[16] = i[6];
    d[19] = i[8];
    d[20] = -i[7];
    d[21] = i[9];
    d[26] = i[6];
    d[24] = -i[8];
    d[28] = i[9];
    d[30] = i[7];
    d[31] = -i[6];
    d[35] = i[9];
    d
}

fn mat6_mul_vec(m: &[f64; 36], v: [f64; 6]) -> [f64; 6] {
    let mut out = [0.0; 6];
    for (r, slot) in out.iter_mut().enumerate() {
        let row = &m[6 * r..6 * r + 6];
        *slot = row.iter().zip(&v).map(|(a, b)| a * b).sum();
    }
    out
}

fn mat6_add(a: &mut [f64; 36], b: &[f64; 36]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += *y;
    }
}

fn mat6_mul(a: &[f64; 36], b: &[f64; 36]) -> [f64; 36] {
    let mut out = [0.0; 36];
    for r in 0..6 {
        for k in 0..6 {
            let v = a[6 * r + k];
            if v != 0.0 {
                for c in 0..6 {
                    out[6 * r + c] += v * b[6 * k + c];
                }
            }
        }
    }
    out
}

/// A per-body (or per-dof) stack of 6 x nv derivative blocks: entry
/// `[row][6 * c + k]`... stored flat as `rows * 6 * nv`, row-major by
/// spatial component then column.
struct Blocks {
    data: Vec<f64>,
    nv: usize,
}

impl Blocks {
    fn zeros(rows: usize, nv: usize) -> Self {
        Self {
            data: vec![0.0; rows * 6 * nv],
            nv,
        }
    }
    fn row(&self, r: usize) -> &[f64] {
        &self.data[r * 6 * self.nv..(r + 1) * 6 * self.nv]
    }
    fn row_mut(&mut self, r: usize) -> &mut [f64] {
        &mut self.data[r * 6 * self.nv..(r + 1) * 6 * self.nv]
    }
    fn copy_row(&mut self, dst: usize, src: usize) {
        let (a, b) = (src.min(dst), src.max(dst));
        if a == b {
            return;
        }
        let (lo, hi) = self.data.split_at_mut(b * 6 * self.nv);
        let n = 6 * self.nv;
        if src < dst {
            hi[..n].copy_from_slice(&lo[src * n..src * n + n]);
        } else {
            lo[dst * n..dst * n + n].copy_from_slice(&hi[..n]);
        }
    }
    /// `column c` of row `r` as a spatial 6-vector.
    fn col(&self, r: usize, c: usize) -> [f64; 6] {
        let row = self.row(r);
        let mut out = [0.0; 6];
        for (k, slot) in out.iter_mut().enumerate() {
            *slot = row[k * self.nv + c];
        }
        out
    }
    fn set_col(&mut self, r: usize, c: usize, v: [f64; 6]) {
        let nv = self.nv;
        let row = self.row_mut(r);
        for (k, val) in v.iter().enumerate() {
            row[k * nv + c] = *val;
        }
    }
    fn add_col(&mut self, r: usize, c: usize, v: [f64; 6]) {
        let nv = self.nv;
        let row = self.row_mut(r);
        for (k, val) in v.iter().enumerate() {
            row[k * nv + c] += *val;
        }
    }
}

/// `∂qfrc_bias/∂qvel` as a dense row-major `nv x nv` matrix:
/// `out[r * nv + c] = ∂bias_r/∂qvel_c`.
///
/// `cvel` and `cdof_dot` are [`crate::composite::com_vel`]'s outputs at the
/// same state. The example's `qDeriv` is the derivative of the *force*
/// side and **subtracts** this quantity (11 section 2, sign convention);
/// returning the bias derivative with its own sign keeps this function
/// self-describing, and the integrator applies `M + h * D`.
///
/// # Errors
///
/// [`CompositeError::WidthMismatch`] when `qd` is the wrong width.
pub fn rne_vel(
    model: &CompositeModel,
    pos: &Position,
    cvel: &[[f64; 6]],
    cdof_dot: &[[f64; 6]],
    qd: &[f64],
) -> Result<Vec<f64>, CompositeError> {
    let nv = model.tree.nv;
    if qd.len() != nv {
        return Err(CompositeError::WidthMismatch {
            what: "qd",
            expected: nv,
            found: qd.len(),
        });
    }
    let nb = model.bodies.len();
    let mut dcvel = Blocks::zeros(nb, nv);
    let mut dcdofdot = Blocks::zeros(nv, nv);

    // Stage 1: Dcvel and Dcdofdot (11 section 2.2), in the same per-joint
    // order as `com_vel` itself: a Free joint's translations first (their
    // cdof_dot derivative stays zero), then its rotational block whose
    // cdof_dot derivatives see Dcvel before any of the block's own
    // columns, then scalar joints.
    for (i, body) in model.bodies.iter().enumerate() {
        if let Some(p) = body.parent {
            dcvel.copy_row(i, p);
        }
        let adr = body.dof_adr;
        match body.kind {
            JointKind::Fixed => {}
            JointKind::Free => {
                for k in 0..3 {
                    dcvel.add_col(i, adr + k, pos.cdof[adr + k]);
                }
                let snapshot: Vec<[f64; 6]> = (0..nv).map(|c| dcvel.col(i, c)).collect();
                for k in 0..3 {
                    let mat = cross_motion_vel_mat(pos.cdof[adr + 3 + k]);
                    for (c, column) in snapshot.iter().enumerate() {
                        dcdofdot.set_col(adr + 3 + k, c, mat6_mul_vec(&mat, *column));
                    }
                }
                for k in 0..3 {
                    dcvel.add_col(i, adr + 3 + k, pos.cdof[adr + 3 + k]);
                }
            }
            JointKind::Revolute | JointKind::Continuous | JointKind::Prismatic => {
                let mat = cross_motion_vel_mat(pos.cdof[adr]);
                for c in 0..nv {
                    let column = dcvel.col(i, c);
                    dcdofdot.set_col(adr, c, mat6_mul_vec(&mat, column));
                }
                dcvel.add_col(i, adr, pos.cdof[adr]);
            }
        }
    }

    // Stage 2 forward: Dcacc and Dcfrcbody (11 section 2.3 step 3).
    let mut dcacc = Blocks::zeros(nb, nv);
    let mut dcfrc = Blocks::zeros(nb, nv);
    for (i, body) in model.bodies.iter().enumerate() {
        if let Some(p) = body.parent {
            dcacc.copy_row(i, p);
        }
        for j in body.dof_adr..body.dof_adr + body.dof_num {
            dcacc.add_col(i, j, cdof_dot[j]);
            if qd[j] != 0.0 {
                let scale = qd[j];
                let n = 6 * nv;
                let src: Vec<f64> = dcdofdot.row(j).to_vec();
                let dst = dcacc.row_mut(i);
                for k in 0..n {
                    dst[k] += scale * src[k];
                }
            }
        }
        let dmul = mul_inert_mat(&pos.cinert[i]);
        let tmp = mul_inert_vec(&pos.cinert[i], cvel[i]);
        let mut mat = cross_force_vel_mat(tmp);
        mat6_add(&mut mat, &mat6_mul(&cross_force_frc_mat(cvel[i]), &dmul));
        for c in 0..nv {
            let acc_col = dcacc.col(i, c);
            let vel_col = dcvel.col(i, c);
            let mut f = mat6_mul_vec(&dmul, acc_col);
            let g = mat6_mul_vec(&mat, vel_col);
            for (a, b) in f.iter_mut().zip(&g) {
                *a += *b;
            }
            dcfrc.set_col(i, c, f);
        }
    }

    // Stage 3 backward: children accumulate into parents (step 5).
    for i in (0..nb).rev() {
        if let Some(p) = model.bodies[i].parent {
            let n = 6 * nv;
            let src: Vec<f64> = dcfrc.row(i).to_vec();
            let dst = dcfrc.row_mut(p);
            for k in 0..n {
                dst[k] += src[k];
            }
        }
    }

    // Stage 4: project through each dof's cdof (step 6).
    let mut out = vec![0.0; nv * nv];
    for (i, body) in model.bodies.iter().enumerate() {
        for j in body.dof_adr..body.dof_adr + body.dof_num {
            for c in 0..nv {
                let col = dcfrc.col(i, c);
                out[j * nv + c] = pos.cdof[j].iter().zip(&col).map(|(a, b)| a * b).sum();
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composite::{com_vel, position, rne};
    use robot_model::usd::with_floating_base;

    /// The analytic derivative against central finite differences of the
    /// zero-acceleration RNE bias itself — the check `mjd_smooth_velFD`
    /// exists to make possible, on the full floating G1 at a non-trivial
    /// state.
    #[test]
    fn the_rne_velocity_derivative_matches_finite_differences_on_the_g1() {
        let profile = with_floating_base(
            robot_model::fixtures::unitree_g1_29dof_usd().expect("fixture loads"),
        )
        .expect("floating wrap");
        let model = CompositeModel::from_profile(&profile);
        let nv = model.tree.nv;
        let mut q = vec![0.0; model.nq];
        for body in &model.bodies {
            if body.kind == JointKind::Free {
                q[body.qpos_adr + 3] = 1.0;
            }
        }
        // A non-trivial posture and velocity, deterministic.
        for (k, v) in q.iter_mut().enumerate().skip(7) {
            *v = 0.2 * ((k as f64) * 0.7).sin();
        }
        let qd: Vec<f64> = (0..nv).map(|k| 0.5 * ((k as f64) * 1.3).cos()).collect();

        let pos = position(&model, &q).unwrap();
        let (cvel, cdof_dot) = com_vel(&model, &pos, &qd).unwrap();
        let analytic = rne_vel(&model, &pos, &cvel, &cdof_dot, &qd).unwrap();

        let bias_at = |qd: &[f64]| -> Vec<f64> { rne(&model, &pos, qd, None).unwrap() };
        let h = 1e-6;
        let mut worst = 0.0f64;
        for c in 0..nv {
            let mut plus = qd.clone();
            plus[c] += h;
            let mut minus = qd.clone();
            minus[c] -= h;
            let bp = bias_at(&plus);
            let bm = bias_at(&minus);
            for r in 0..nv {
                let fd = (bp[r] - bm[r]) / (2.0 * h);
                let a = analytic[r * nv + c];
                let scale = fd.abs().max(a.abs()).max(1.0);
                worst = worst.max((fd - a).abs() / scale);
            }
        }
        assert!(
            worst < 1e-6,
            "worst relative disagreement with finite differences: {worst}"
        );
    }

    /// At zero velocity the bias is pure gravity, whose derivative is zero.
    #[test]
    fn the_derivative_vanishes_at_rest() {
        let profile = with_floating_base(
            robot_model::fixtures::unitree_g1_29dof_usd().expect("fixture loads"),
        )
        .expect("floating wrap");
        let model = CompositeModel::from_profile(&profile);
        let nv = model.tree.nv;
        let mut q = vec![0.0; model.nq];
        for body in &model.bodies {
            if body.kind == JointKind::Free {
                q[body.qpos_adr + 3] = 1.0;
            }
        }
        let qd = vec![0.0; nv];
        let pos = position(&model, &q).unwrap();
        let (cvel, cdof_dot) = com_vel(&model, &pos, &qd).unwrap();
        let d = rne_vel(&model, &pos, &cvel, &cdof_dot, &qd).unwrap();
        let max = d.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(max < 1e-12, "non-zero derivative at rest: {max}");
    }
}

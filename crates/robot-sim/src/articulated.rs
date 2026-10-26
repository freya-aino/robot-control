//! Multi-body forward and inverse dynamics over a `robot_model::RobotProfile`'s
//! joint tree, via the Articulated-Body Algorithm (forward) and the
//! Recursive Newton-Euler Algorithm (inverse) — Featherstone, *Rigid Body
//! Dynamics Algorithms*, chapters 4 and 7.
//!
//! # Scope: no ground contact; the base may now be fixed or free
//!
//! [`ArticulatedBody`] handles every joint `robot_model::joint::JointKind`
//! can express: the single-DOF kinds (`Revolute`/`Continuous`/`Prismatic`),
//! `Fixed` (zero DOF), and `Free` (six DOF — translation plus rotation, a
//! floating base). A profile's root link attaches to an explicit, typically
//! massless `world` link via a `Free` joint to let the attached body (e.g.
//! a humanoid's pelvis) fall and move under gravity and applied wrench;
//! `crates/robot-sim/SIMULATION_SPECS.md` Section 3 is the design record
//! for why this is a `robot-model`/`robot-sim` construct never importable
//! from URDF. Ground-contact force resolution is still out of scope,
//! named rather than silently assumed — see this crate's README.
//!
//! Every joint kind other than `Free` has at most one degree of freedom,
//! so most of the algebra below is genuinely scalar; `Free`'s six degrees
//! of freedom make the per-joint quantities Featherstone's algorithm calls
//! `U`/`D`/`u` into a 6x6-by-k matrix, a kxk matrix and a k-vector (k = 1 or
//! 6 here) instead of a 6-vector, a scalar and a scalar. The implementation
//! below is written once, generically over `k`, rather than duplicated —
//! every existing single-DOF test (the hand-derived pendulum closed form,
//! the double-pendulum round-trip, the real G1) therefore also exercises
//! and re-validates the general, multi-DOF code path at `k = 1`, not only
//! the new `k = 6` tests the `Free` joint adds.
//!
//! A `Free` joint also needs its own position representation distinct
//! from its velocity: translation plus a quaternion is seven numbers,
//! while its generalized velocity (and the `qdd`/`tau` it shares a width
//! with) is six — [`ArticulatedBody::position_width`] and
//! [`ArticulatedBody::dof`] are therefore genuinely different
//! quantities once any `Free` joint exists, matching
//! `SIMULATION_SPECS.md` Section 3 item 2's `nq`-vs-`nv` requirement.
//!
//! # Validation
//!
//! No part of this module trusts itself alone. [`ArticulatedBody::forward_dynamics`]
//! (ABA) and [`ArticulatedBody::inverse_dynamics`] (RNE) are independent
//! implementations of the same physics; `forward_dynamics_inverts_inverse_dynamics`
//! round-trips one through the other over the real vendored G1 profile
//! and a hand-built chain. `forward_dynamics_matches_the_textbook_point_mass_pendulum`
//! checks ABA against a closed form derived by hand, not by this module.
//! `free_motion_conserves_energy_like_the_single_pendulum_benchmark_does`
//! repeats `robot_sim::pendulum`'s own established conservation check over
//! a multi-link chain. The `Free` joint is validated the same way: a
//! torque-free floating body's translational motion is checked against
//! the closed-form projectile equation, and a spinning symmetric free
//! body's angular velocity is checked for conservation, before the real
//! G1 is ever given a floating base.

use crate::spatial::{
    cross_motion_force, cross_motion_motion, transform_force, transform_motion, SpatialInertia,
    Twist, Wrench,
};
use robot_model::joint::JointKind;
use robot_model::joint::JointSpec;
use robot_model::pose::RigidPose;
use robot_model::profile::RobotProfile;
use robot_types::frames::{Quat, Vec3};
use robot_types::ids::LinkId;
use robot_types::math::Matrix;
use std::collections::HashMap;
use std::ops::Mul;

/// Why a dynamics computation could not run.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum DynamicsError {
    /// A `q`/`qd`/`qdd`/`tau` vector did not have one entry per actuated
    /// joint.
    WidthMismatch {
        /// What the vector was for.
        what: &'static str,
        /// Degrees of freedom the profile declares.
        expected: usize,
        /// Entries supplied.
        found: usize,
    },
    /// A supplied value was not finite.
    NotFinite {
        /// What the value was for.
        what: &'static str,
    },
}

impl core::fmt::Display for DynamicsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::WidthMismatch {
                what,
                expected,
                found,
            } => {
                write!(
                    f,
                    "{what} has {found} entries, profile has {expected} degrees of freedom"
                )
            }
            Self::NotFinite { what } => write!(f, "{what} contains a non-finite value"),
        }
    }
}

impl core::error::Error for DynamicsError {}

/// One body in the articulated tree: a link together with the single
/// joint that attaches it to its parent (the profile's root link is not
/// itself a body here — it is the fixed base every body's chain of
/// parents eventually reaches, unless that root attaches to an explicit
/// `world` link via a `Free` joint, in which case the *former* root is
/// simply the first body in the tree like any other).
struct Body {
    joint: JointSpec,
    /// Index into `bodies` of this body's parent, or `None` when the
    /// parent is the fixed base (the profile's root link).
    parent: Option<usize>,
    /// Offset into the `qd`/`qdd`/`tau` vectors.
    dof_offset: usize,
    /// Width in the `qd`/`qdd`/`tau` vectors: 0 (`Fixed`), 1 (every
    /// single-axis kind) or 6 (`Free`).
    dof_width: usize,
    /// Offset into the `q` vector.
    position_offset: usize,
    /// Width in the `q` vector: 0 (`Fixed`), 1 (every single-axis kind)
    /// or 7 (`Free`: 3 translation + 4 quaternion).
    position_width: usize,
    inertia: SpatialInertia,
}

/// A profile's joint tree, preprocessed into the articulated structure
/// the dynamics algorithms recurse over.
pub struct ArticulatedBody {
    bodies: Vec<Body>,
    n_dof: usize,
    n_position: usize,
}

/// The motion subspace of `kind`: the basis twists (expressed in the
/// joint's own, constant local axes — see `JointSpec::transform_at`'s own
/// doc comment on why this is value-independent) that its generalized
/// velocity multiplies. Width `k` matches [`JointKind::dof_count`]: empty
/// for `Fixed`, one twist for every single-axis kind, six (three angular
/// then three linear, matching this module's `Free`-joint velocity
/// layout) for `Free`.
fn motion_subspace_basis(kind: JointKind, axis: Vec3) -> Vec<Twist> {
    match kind {
        JointKind::Fixed => Vec::new(),
        JointKind::Revolute | JointKind::Continuous => vec![Twist::angular_rate(axis, 1.0)],
        JointKind::Prismatic => vec![Twist::linear_rate(axis, 1.0)],
        JointKind::Free => vec![
            Twist::angular_rate(Vec3::unit_x(), 1.0),
            Twist::angular_rate(Vec3::unit_y(), 1.0),
            Twist::angular_rate(Vec3::unit_z(), 1.0),
            Twist::linear_rate(Vec3::unit_x(), 1.0),
            Twist::linear_rate(Vec3::unit_y(), 1.0),
            Twist::linear_rate(Vec3::unit_z(), 1.0),
        ],
    }
}

/// Width of `kind`'s own slice of the `q` vector: 0 (`Fixed`), 1 (every
/// single-axis kind) or 7 (`Free`: 3 translation then 4 quaternion,
/// `w, x, y, z` — matching `robot_types::frames::Quat`'s own field
/// order).
const fn position_width(kind: JointKind) -> usize {
    match kind {
        JointKind::Fixed => 0,
        JointKind::Revolute | JointKind::Continuous | JointKind::Prismatic => 1,
        JointKind::Free => 7,
    }
}

/// Combines a joint's basis twists with its own generalized velocity (or
/// acceleration) slice into one spatial motion vector — `sum_j basis[j] *
/// values[j]`. For a `Free` joint this is the standard "linear
/// combination of the six basis twists" reading of a 6-DOF joint's
/// motion; for a single-DOF joint it collapses to `basis[0] * values[0]`
/// exactly, and for `Fixed` (`basis` and `values` both empty) it is
/// `Twist::ZERO`.
fn combine(basis: &[Twist], values: &[f64]) -> Twist {
    basis
        .iter()
        .zip(values)
        .fold(Twist::ZERO, |acc, (s, &v)| acc + s.scaled(v))
}

/// The pose a `Free` joint's own seven-number position slice contributes,
/// relative to the joint's origin — translation then an (re-normalized)
/// orientation quaternion.
fn free_joint_pose(position: &[f64]) -> RigidPose {
    let translation = Vec3::new(position[0], position[1], position[2]);
    let rotation = Quat::from_components(position[3], position[4], position[5], position[6])
        .unwrap_or(Quat::IDENTITY);
    RigidPose::new(translation, rotation)
}

/// The pose this body's joint contributes at the current configuration
/// `q`, relative to the parent link's own frame — dispatching to
/// [`free_joint_pose`] for a `Free` joint (whose position
/// `JointSpec::transform_at` cannot represent) and to
/// `JointSpec::transform_at` otherwise.
fn joint_transform(body: &Body, q: &[f64]) -> RigidPose {
    let slice = &q[body.position_offset..body.position_offset + body.position_width];
    if body.joint.kind == JointKind::Free {
        body.joint.origin.compose(&free_joint_pose(slice))
    } else {
        let value = slice.first().copied().unwrap_or(0.0);
        body.joint.transform_at(value)
    }
}

impl ArticulatedBody {
    /// Preprocesses `profile` into the articulated structure the
    /// dynamics algorithms recurse over.
    ///
    /// Assumes `profile.validate()` has already passed, exactly as
    /// `robot_model::kinematics::forward_kinematics` assumes it — a tree
    /// structure and a single root are checked once at load, not on
    /// every dynamics call.
    #[must_use]
    pub fn from_profile(profile: &RobotProfile) -> Self {
        let mut dof_slots: HashMap<robot_types::ids::JointId, (usize, usize)> = HashMap::new();
        let mut position_slots: HashMap<robot_types::ids::JointId, (usize, usize)> = HashMap::new();
        let mut next_dof = 0usize;
        let mut next_position = 0usize;
        for j in profile.actuated_joints() {
            let dof_width = j.kind.dof_count();
            dof_slots.insert(j.id, (next_dof, dof_width));
            next_dof += dof_width;

            let pos_width = position_width(j.kind);
            position_slots.insert(j.id, (next_position, pos_width));
            next_position += pos_width;
        }

        let order = profile.topological_link_order();
        let mut body_index: HashMap<LinkId, usize> = HashMap::new();
        let mut bodies = Vec::with_capacity(order.len().saturating_sub(1));

        for link_id in order.into_iter().skip(1) {
            let joint = profile
                .parent_joint_of(link_id)
                .expect("every non-root link has exactly one parent joint in a validated profile");
            let link = profile
                .link(link_id)
                .expect("link exists in a validated profile");
            let parent = body_index.get(&joint.parent).copied();
            let index = bodies.len();
            let (dof_offset, dof_width) = dof_slots.get(&joint.id).copied().unwrap_or((0, 0));
            let (position_offset, pos_width) =
                position_slots.get(&joint.id).copied().unwrap_or((0, 0));
            bodies.push(Body {
                joint: joint.clone(),
                parent,
                dof_offset,
                dof_width,
                position_offset,
                position_width: pos_width,
                inertia: SpatialInertia::from_mass_properties(&link.mass),
            });
            body_index.insert(link_id, index);
        }

        Self {
            bodies,
            n_dof: next_dof,
            n_position: next_position,
        }
    }

    /// Degrees of freedom — the width every `qd`/`qdd`/`tau` vector must
    /// have. Matches `RobotProfile::dof` exactly.
    #[must_use]
    pub fn dof(&self) -> usize {
        self.n_dof
    }

    /// Width every `q` (position) vector must have. Equal to
    /// [`ArticulatedBody::dof`] unless the profile contains a `Free`
    /// joint, whose position (translation plus quaternion) needs seven
    /// numbers against its six velocity/force degrees of freedom.
    #[must_use]
    pub fn position_width(&self) -> usize {
        self.n_position
    }

    fn check_width(
        &self,
        what: &'static str,
        v: &[f64],
        expected: usize,
    ) -> Result<(), DynamicsError> {
        if v.len() != expected {
            return Err(DynamicsError::WidthMismatch {
                what,
                expected,
                found: v.len(),
            });
        }
        if v.iter().any(|x| !x.is_finite()) {
            return Err(DynamicsError::NotFinite { what });
        }
        Ok(())
    }

    /// Inverse dynamics (Recursive Newton-Euler): the joint torques
    /// required to realize acceleration `qdd` at configuration `q` and
    /// velocity `qd`, under `gravity` (e.g.
    /// `profile.gravity.acceleration()`).
    pub fn inverse_dynamics(
        &self,
        q: &[f64],
        qd: &[f64],
        qdd: &[f64],
        gravity: Vec3,
    ) -> Result<Vec<f64>, DynamicsError> {
        self.check_width("q", q, self.n_position)?;
        self.check_width("qd", qd, self.n_dof)?;
        self.check_width("qdd", qdd, self.n_dof)?;

        // Featherstone's trick: fold gravity into the base's acceleration
        // rather than adding a separate force everywhere it matters.
        let base_accel = Twist {
            angular: Vec3::ZERO,
            linear: -gravity,
        };

        let n = self.bodies.len();
        let mut v: Vec<Twist> = Vec::with_capacity(n);
        let mut a: Vec<Twist> = Vec::with_capacity(n);
        let mut f: Vec<Wrench> = Vec::with_capacity(n);
        let mut transforms = Vec::with_capacity(n);

        for body in &self.bodies {
            let t = joint_transform(body, q);
            let s = motion_subspace_basis(body.joint.kind, body.joint.axis);
            let dof = body.dof_offset..body.dof_offset + body.dof_width;
            let joint_v = combine(&s, &qd[dof.clone()]);
            let joint_a = combine(&s, &qdd[dof]);

            let v_parent = body.parent.map_or(Twist::ZERO, |p| v[p]);
            let a_parent = body.parent.map_or(base_accel, |p| a[p]);

            let v_i = transform_motion(&t, v_parent) + joint_v;
            let a_i = transform_motion(&t, a_parent) + joint_a + cross_motion_motion(v_i, joint_v);
            let f_i = body.inertia.apply(a_i) + cross_motion_force(v_i, body.inertia.apply(v_i));

            v.push(v_i);
            a.push(a_i);
            f.push(f_i);
            transforms.push(t);
        }

        let mut tau = vec![0.0; self.n_dof];
        for i in (0..n).rev() {
            let body = &self.bodies[i];
            let s = motion_subspace_basis(body.joint.kind, body.joint.axis);
            for (j, s_j) in s.iter().enumerate() {
                tau[body.dof_offset + j] = f[i].dot(*s_j);
            }
            if let Some(p) = body.parent {
                f[p] = f[p] + transform_force(&transforms[i], f[i]);
            }
        }
        Ok(tau)
    }

    /// Forward dynamics (Articulated-Body Algorithm): the joint
    /// accelerations produced by torques `tau` at configuration `q` and
    /// velocity `qd`, under `gravity`.
    pub fn forward_dynamics(
        &self,
        q: &[f64],
        qd: &[f64],
        tau: &[f64],
        gravity: Vec3,
    ) -> Result<Vec<f64>, DynamicsError> {
        self.check_width("q", q, self.n_position)?;
        self.check_width("qd", qd, self.n_dof)?;
        self.check_width("tau", tau, self.n_dof)?;

        let base_accel = Twist {
            angular: Vec3::ZERO,
            linear: -gravity,
        };
        let n = self.bodies.len();

        let mut v: Vec<Twist> = Vec::with_capacity(n);
        let mut c: Vec<Twist> = Vec::with_capacity(n);
        let mut transforms = Vec::with_capacity(n);
        let mut ia: Vec<SpatialInertia> = Vec::with_capacity(n);
        let mut pa: Vec<Wrench> = Vec::with_capacity(n);
        let mut bases: Vec<Vec<Twist>> = Vec::with_capacity(n);

        // Pass 1 (root -> leaf): velocities, bias accelerations, and each
        // body's own (not-yet-reduced) articulated inertia/bias force.
        for body in &self.bodies {
            let t = joint_transform(body, q);
            let s = motion_subspace_basis(body.joint.kind, body.joint.axis);
            let joint_v = combine(&s, &qd[body.dof_offset..body.dof_offset + body.dof_width]);

            let v_parent = body.parent.map_or(Twist::ZERO, |p| v[p]);
            let v_i = transform_motion(&t, v_parent) + joint_v;
            let c_i = cross_motion_motion(v_i, joint_v);
            let pa_i = cross_motion_force(v_i, body.inertia.apply(v_i));

            v.push(v_i);
            c.push(c_i);
            transforms.push(t);
            ia.push(body.inertia.clone());
            pa.push(pa_i);
            bases.push(s);
        }

        // Pass 2 (leaf -> root): eliminate each joint's own `k` degrees
        // of freedom (k = |basis|), folding its effective inertia/bias
        // force into its parent. U/D/u are retained per body (not just
        // the effective, parent-folded quantities) for pass 3 below,
        // exactly as the single-DOF version already had to.
        let mut u_list: Vec<Vec<Wrench>> = vec![Vec::new(); n];
        let mut u_scalar_list: Vec<Vec<f64>> = vec![Vec::new(); n];
        let mut d_mats: Vec<Matrix> = (0..n).map(|_| Matrix::zeros(0, 0)).collect();

        for i in (0..n).rev() {
            let body = &self.bodies[i];
            let s = &bases[i];
            let k = s.len();

            let (ia_eff, pa_eff) = if k == 0 {
                (ia[i].clone(), pa[i] + ia[i].apply(c[i]))
            } else {
                let u_wrenches: Vec<Wrench> = s.iter().map(|sj| ia[i].apply(*sj)).collect();
                let mut d_mat = Matrix::zeros(k, k);
                for (a_idx, s_a) in s.iter().enumerate() {
                    for (b_idx, u_b) in u_wrenches.iter().enumerate() {
                        d_mat[(a_idx, b_idx)] = u_b.dot(*s_a);
                    }
                }
                let tau_slice = &tau[body.dof_offset..body.dof_offset + body.dof_width];
                let u_scalar: Vec<f64> = (0..k)
                    .map(|a_idx| tau_slice[a_idx] - pa[i].dot(s[a_idx]))
                    .collect();

                let mut u_mat = Matrix::zeros(6, k);
                for (col, w) in u_wrenches.iter().enumerate() {
                    let wc = w.to_column();
                    for row in 0..6 {
                        u_mat[(row, col)] = wc[(row, 0)];
                    }
                }

                let d_inv_ut = d_mat
                    .solve(&u_mat.transposed())
                    .expect("joint inertia D is singular — a degenerate/zero inertia upstream of a real DOF");
                let reduction = u_mat
                    .mul(&d_inv_ut)
                    .expect("6xk * kx6 is always well-formed");
                let ia_eff = ia[i].sub(&SpatialInertia::from_matrix(reduction));

                let z = d_mat
                    .solve(&Matrix::column(&u_scalar))
                    .expect("same D as above");
                let u_dinv_u = u_mat.mul(&z).expect("6xk * kx1 is always well-formed");
                let pa_eff = pa[i] + ia_eff.apply(c[i]) + Wrench::from_column(&u_dinv_u);

                u_list[i] = u_wrenches;
                u_scalar_list[i] = u_scalar;
                d_mats[i] = d_mat;
                (ia_eff, pa_eff)
            };

            if let Some(p) = body.parent {
                ia[p] = ia[p].add(&ia_eff.transform_to_parent(&transforms[i]));
                pa[p] = pa[p] + transform_force(&transforms[i], pa_eff);
            }
        }

        // Pass 3 (root -> leaf): solve for qdd given the parent's now-known
        // true acceleration, using each body's own (non-reduced) U/D/u.
        let mut qdd = vec![0.0; self.n_dof];
        let mut a: Vec<Twist> = Vec::with_capacity(n);
        for (i, body) in self.bodies.iter().enumerate() {
            let s = &bases[i];
            let k = s.len();
            let a_parent = body.parent.map_or(base_accel, |p| a[p]);
            let a_prime = transform_motion(&transforms[i], a_parent) + c[i];

            let a_i = if k == 0 {
                a_prime
            } else {
                let u_wrenches = &u_list[i];
                let u_scalar = &u_scalar_list[i];
                let rhs: Vec<f64> = (0..k)
                    .map(|a_idx| u_scalar[a_idx] - u_wrenches[a_idx].dot(a_prime))
                    .collect();
                let qdd_col = d_mats[i]
                    .solve(&Matrix::column(&rhs))
                    .expect("same D already solved in pass 2");
                let mut a_i = a_prime;
                for a_idx in 0..k {
                    let qdd_val = qdd_col[(a_idx, 0)];
                    qdd[body.dof_offset + a_idx] = qdd_val;
                    a_i = a_i + s[a_idx].scaled(qdd_val);
                }
                a_i
            };
            a.push(a_i);
        }
        Ok(qdd)
    }

    /// Total kinetic energy at configuration `q`, velocity `qd`: `0.5 *
    /// sum_i v_i . (I_i v_i)` over every body's own spatial velocity and
    /// rigid-body inertia. Independent of [`ArticulatedBody::forward_dynamics`]
    /// and [`ArticulatedBody::inverse_dynamics`] beyond sharing the same
    /// velocity recursion — used (alongside potential energy from
    /// `robot_model::kinematics::forward_kinematics`, computed separately
    /// by callers) to check energy conservation under free dynamics.
    pub fn kinetic_energy(&self, q: &[f64], qd: &[f64]) -> Result<f64, DynamicsError> {
        self.check_width("q", q, self.n_position)?;
        self.check_width("qd", qd, self.n_dof)?;

        let mut v: Vec<Twist> = Vec::with_capacity(self.bodies.len());
        let mut total = 0.0;
        for body in &self.bodies {
            let t = joint_transform(body, q);
            let s = motion_subspace_basis(body.joint.kind, body.joint.axis);
            let joint_v = combine(&s, &qd[body.dof_offset..body.dof_offset + body.dof_width]);
            let v_parent = body.parent.map_or(Twist::ZERO, |p| v[p]);
            let v_i = transform_motion(&t, v_parent) + joint_v;
            total += 0.5 * body.inertia.apply(v_i).dot(v_i);
            v.push(v_i);
        }
        Ok(total)
    }
}

/// Configuration and velocity of every actuated joint, steppable in time
/// against an [`ArticulatedBody`] — the fixed-base-or-floating equivalent
/// of `robot_sim::pendulum::Pendulum`, generalized to a whole tree.
///
/// `q` and `qd` are different widths whenever the body contains a `Free`
/// joint ([`ArticulatedBody::position_width`] vs. [`ArticulatedBody::dof`])
/// — see this module's own doc comment.
#[derive(Clone, Debug)]
pub struct MultibodyState {
    /// Joint positions, in `RobotProfile::actuated_joints` order. Width
    /// is [`ArticulatedBody::position_width`].
    pub q: Vec<f64>,
    /// Joint velocities, same order. Width is [`ArticulatedBody::dof`].
    pub qd: Vec<f64>,
}

impl MultibodyState {
    /// A state at rest at `q`, with `dof` zero velocities (pass
    /// `body.dof()`; kept as an explicit parameter rather than derived
    /// from `q.len()` since the two widths differ once a `Free` joint
    /// exists).
    #[must_use]
    pub fn at_rest(q: Vec<f64>, dof: usize) -> Self {
        let qd = vec![0.0; dof];
        Self { q, qd }
    }

    /// Advances one step by semi-implicit (symplectic) Euler — update
    /// velocity from the acceleration first, then position from the
    /// updated velocity — the same integrator `robot_sim::pendulum::Pendulum::step`
    /// uses, generalized to every degree of freedom at once. A `Free`
    /// joint's position integrates differently from a scalar joint's
    /// (see [`Self::integrate_free_position`]); every other joint still
    /// just adds `qd * dt` to its own single `q` slot.
    ///
    /// Like any explicit integrator, this is only *conditionally* stable:
    /// too large a `dt` relative to the system's stiffest mode makes
    /// energy grow each step instead of staying bounded, ending in a
    /// non-finite state. This is a real, measured property of the real
    /// vendored G1 fixture, not a hypothetical — `robot sim multibody`
    /// (in `robot-cli`) diverges within about 1.3 simulated seconds at
    /// `dt = 5e-4` but runs stably at `dt = 1e-4`, which is why that
    /// command's own default is the smaller value. There is also no
    /// joint-limit or ground-contact constraint anywhere in this module
    /// (see its own doc comment): nothing stops an under-loaded joint
    /// from spinning through several full rotations once energy
    /// accumulates in it, which is a scope limitation, not a sign the
    /// integration itself is unstable.
    pub fn step(
        &mut self,
        body: &ArticulatedBody,
        tau: &[f64],
        gravity: Vec3,
        dt: robot_types::time::Duration,
    ) -> Result<(), DynamicsError> {
        let qdd = body.forward_dynamics(&self.q, &self.qd, tau, gravity)?;
        let dt_s = dt.as_secs_f64();
        for (v, a) in self.qd.iter_mut().zip(&qdd) {
            *v += a * dt_s;
        }
        for b in &body.bodies {
            if b.joint.kind == JointKind::Free {
                Self::integrate_free_position(
                    &mut self.q[b.position_offset..b.position_offset + 7],
                    &self.qd[b.dof_offset..b.dof_offset + 6],
                    dt_s,
                );
            } else if b.dof_width > 0 {
                self.q[b.position_offset] += self.qd[b.dof_offset] * dt_s;
            }
        }
        Ok(())
    }

    /// Integrates a `Free` joint's own seven-number position slice
    /// (translation then quaternion `w, x, y, z`) forward by `dt_s`
    /// given its six-number velocity slice (angular then linear,
    /// matching [`motion_subspace_basis`]'s `Free` ordering — both
    /// expressed in the body's own local frame, this module's spatial-
    /// algebra convention throughout).
    ///
    /// Translation integrates in the *world* frame
    /// (`translation += dt * rotation.rotate(local_linear_velocity)`,
    /// since the stored linear velocity is local); orientation
    /// integrates via the standard local-angular-velocity quaternion
    /// derivative `qdot = 0.5 * q * omega_local` (`omega_local` as a
    /// pure quaternion), a first-order step re-normalized afterward
    /// (`Quat::from_components` does the renormalizing). This is a
    /// stated simplification: a higher-order or exponential-map update
    /// would integrate orientation more accurately per step, and is an
    /// open option, not a claim that this is the only valid approach.
    fn integrate_free_position(position: &mut [f64], velocity: &[f64], dt_s: f64) {
        let angular = Vec3::new(velocity[0], velocity[1], velocity[2]);
        let linear_local = Vec3::new(velocity[3], velocity[4], velocity[5]);
        let translation = Vec3::new(position[0], position[1], position[2]);
        let rotation = Quat::from_components(position[3], position[4], position[5], position[6])
            .unwrap_or(Quat::IDENTITY);

        let new_translation = translation + rotation.rotate(linear_local) * dt_s;

        let omega_pure = Quat {
            w: 0.0,
            x: angular.x,
            y: angular.y,
            z: angular.z,
        };
        let qdot = rotation.mul(omega_pure);
        let half_dt = 0.5 * dt_s;
        let new_rotation = Quat::from_components(
            rotation.w + half_dt * qdot.w,
            rotation.x + half_dt * qdot.x,
            rotation.y + half_dt * qdot.y,
            rotation.z + half_dt * qdot.z,
        )
        .unwrap_or(rotation);

        position[0] = new_translation.x;
        position[1] = new_translation.y;
        position[2] = new_translation.z;
        position[3] = new_rotation.w;
        position[4] = new_rotation.x;
        position[5] = new_rotation.y;
        position[6] = new_rotation.z;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_model::joint::{JointLimits, JointSpec};
    use robot_model::kinematics::forward_kinematics;
    use robot_model::link::{InertiaTensor, LinkSpec, MassProperties};
    use robot_model::pose::RigidPose;
    use robot_model::profile::{FirmwareContract, RobotProfile, PROFILE_SCHEMA_VERSION};
    use robot_types::frames::{GravityConvention, Quat};
    use robot_types::ids::{JointId, LinkId, Name};
    use robot_types::schema::ExtensionTable;
    use robot_types::time::Duration;
    use robot_types::units::{NewtonMetres, Radians, RadiansPerSecond, Range};

    fn empty_link(id: u32, name: &str, mass: MassProperties) -> LinkSpec {
        LinkSpec {
            id: LinkId::new(id),
            name: Name::new(name).unwrap(),
            mass,
            collision: Vec::new(),
        }
    }

    fn revolute(
        id: u32,
        name: &str,
        parent: u32,
        child: u32,
        origin: RigidPose,
        axis: Vec3,
    ) -> JointSpec {
        JointSpec {
            id: JointId::new(id),
            name: Name::new(name).unwrap(),
            kind: JointKind::Revolute,
            parent: LinkId::new(parent),
            child: LinkId::new(child),
            origin,
            axis,
            limits: JointLimits::bounded(
                Range::<Radians>::new(
                    Radians(-core::f64::consts::PI),
                    Radians(core::f64::consts::PI),
                )
                .unwrap(),
                RadiansPerSecond(1000.0),
                NewtonMetres(1000.0),
            ),
            sign: 1.0,
            zero_offset: 0.0,
        }
    }

    fn wrap_profile(joints: Vec<JointSpec>, links: Vec<LinkSpec>) -> RobotProfile {
        RobotProfile {
            schema_version: PROFILE_SCHEMA_VERSION,
            name: Name::new("fixture").unwrap(),
            profile_version: "test".to_owned(),
            joints,
            links,
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

    /// A point mass `mass` at distance `length` along local +X from a
    /// single revolute joint (axis +Y) at the world-fixed origin.
    fn point_mass_pendulum(mass: f64, length: f64) -> RobotProfile {
        let root = empty_link(0, "anchor", MassProperties::MASSLESS);
        let joint = revolute(0, "pivot", 0, 1, RigidPose::IDENTITY, Vec3::unit_y());
        let bob = empty_link(
            1,
            "bob",
            MassProperties {
                mass: robot_types::units::Kilograms(mass),
                center_of_mass: Vec3::new(length, 0.0, 0.0),
                inertia: InertiaTensor::ZERO,
            },
        );
        wrap_profile(vec![joint], vec![root, bob])
    }

    /// A two-link planar chain: joint1 (axis +Y) at the origin carries
    /// link1 (mass m1, point mass at local +X offset l1); joint2 (axis
    /// +Y), at link1's local (l1, 0, 0), carries link2 (mass m2, point
    /// mass at local +X offset l2).
    fn double_pendulum(m1: f64, l1: f64, m2: f64, l2: f64) -> RobotProfile {
        let root = empty_link(0, "anchor", MassProperties::MASSLESS);
        let joint1 = revolute(0, "joint1", 0, 1, RigidPose::IDENTITY, Vec3::unit_y());
        let link1 = empty_link(
            1,
            "link1",
            MassProperties {
                mass: robot_types::units::Kilograms(m1),
                center_of_mass: Vec3::new(l1, 0.0, 0.0),
                inertia: InertiaTensor::ZERO,
            },
        );
        let joint2 = revolute(
            1,
            "joint2",
            1,
            2,
            RigidPose::from_translation(Vec3::new(l1, 0.0, 0.0)),
            Vec3::unit_y(),
        );
        let link2 = empty_link(
            2,
            "link2",
            MassProperties {
                mass: robot_types::units::Kilograms(m2),
                center_of_mass: Vec3::new(l2, 0.0, 0.0),
                inertia: InertiaTensor::ZERO,
            },
        );
        wrap_profile(vec![joint1, joint2], vec![root, link1, link2])
    }

    /// A single free-floating body (no children): `mass`/`inertia`
    /// attached to an explicit, massless `world` link via a `Free`
    /// joint — the minimal fixture for validating the `Free`-joint math
    /// in isolation, before the real G1 is ever given a floating base.
    fn free_body(mass: f64, inertia: InertiaTensor) -> RobotProfile {
        let world = empty_link(0, "world", MassProperties::MASSLESS);
        let joint = JointSpec {
            id: JointId::new(0),
            name: Name::new("base").unwrap(),
            kind: JointKind::Free,
            parent: LinkId::new(0),
            child: LinkId::new(1),
            origin: RigidPose::IDENTITY,
            axis: Vec3::ZERO,
            limits: JointLimits::unbounded(RadiansPerSecond(1000.0), NewtonMetres(1000.0)),
            sign: 1.0,
            zero_offset: 0.0,
        };
        let body_link = empty_link(
            1,
            "body",
            MassProperties {
                mass: robot_types::units::Kilograms(mass),
                center_of_mass: Vec3::ZERO,
                inertia,
            },
        );
        wrap_profile(vec![joint], vec![world, body_link])
    }

    /// `[tx, ty, tz, qw, qx, qy, qz]` at the origin with identity
    /// orientation — the canonical "at rest" `Free`-joint position.
    const FREE_IDENTITY: [f64; 7] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];

    const G: f64 = 9.806_65;
    const GRAVITY: Vec3 = Vec3::new(0.0, 0.0, -G);

    #[test]
    fn forward_dynamics_matches_the_textbook_point_mass_pendulum() {
        // I_pivot * qdd = tau + m*g*L*cos(theta), derived by hand (not by
        // this module) for a point mass at local (L,0,0) rotating about
        // +Y through the origin, under gravity (0,0,-g). See this
        // module's doc comment.
        let mass = 1.7;
        let length = 0.9;
        let profile = point_mass_pendulum(mass, length);
        profile.validate().expect("fixture is a valid profile");
        let body = ArticulatedBody::from_profile(&profile);
        assert_eq!(body.dof(), 1);

        let i_pivot = mass * length * length;
        for theta in [-1.2, -0.3, 0.0, 0.4, 1.1, 2.0] {
            for tau in [-2.0, 0.0, 3.5] {
                let qdd = body
                    .forward_dynamics(&[theta], &[0.0], &[tau], GRAVITY)
                    .unwrap();
                let expected = (tau + mass * G * length * theta.cos()) / i_pivot;
                assert!(
                    (qdd[0] - expected).abs() < 1e-9,
                    "theta={theta} tau={tau} got={} want={expected}",
                    qdd[0]
                );
            }
        }
    }

    #[test]
    fn inverse_dynamics_matches_the_textbook_point_mass_pendulum() {
        let mass = 1.7;
        let length = 0.9;
        let profile = point_mass_pendulum(mass, length);
        let body = ArticulatedBody::from_profile(&profile);
        let i_pivot = mass * length * length;

        for theta in [-1.0, 0.2, 1.5] {
            for qdd in [-3.0, 0.0, 2.0] {
                let tau = body
                    .inverse_dynamics(&[theta], &[0.0], &[qdd], GRAVITY)
                    .unwrap();
                let expected = i_pivot * qdd - mass * G * length * theta.cos();
                assert!(
                    (tau[0] - expected).abs() < 1e-9,
                    "got={} want={expected}",
                    tau[0]
                );
            }
        }
    }

    #[test]
    fn forward_dynamics_inverts_inverse_dynamics_on_a_double_pendulum() {
        let profile = double_pendulum(1.0, 0.5, 0.6, 0.35);
        profile.validate().expect("fixture is a valid profile");
        let body = ArticulatedBody::from_profile(&profile);

        let cases: [([f64; 2], [f64; 2], [f64; 2]); 3] = [
            ([0.3, -0.5], [0.1, 0.2], [0.0, 0.0]),
            ([-0.8, 1.1], [-0.3, 0.4], [1.5, -2.0]),
            ([0.0, 0.0], [0.0, 0.0], [0.5, -0.5]),
        ];
        for (q, qd, qdd) in cases {
            let tau = body.inverse_dynamics(&q, &qd, &qdd, GRAVITY).unwrap();
            let qdd_recovered = body.forward_dynamics(&q, &qd, &tau, GRAVITY).unwrap();
            for i in 0..2 {
                assert!(
                    (qdd_recovered[i] - qdd[i]).abs() < 1e-7,
                    "dof {i}: recovered={} want={}",
                    qdd_recovered[i],
                    qdd[i]
                );
            }
        }
    }

    #[test]
    fn forward_dynamics_inverts_inverse_dynamics_on_the_real_g1() {
        let profile = robot_model::fixtures::unitree_g1_29dof().unwrap();
        let body = ArticulatedBody::from_profile(&profile);
        let n = body.dof();

        // A deterministic, non-trivial posture/velocity/acceleration —
        // not all zero, so every term in both algorithms is exercised.
        let q: Vec<f64> = (0..n).map(|i| 0.1 * ((i as f64 * 1.37).sin())).collect();
        let qd: Vec<f64> = (0..n).map(|i| 0.05 * ((i as f64 * 0.91).cos())).collect();
        let qdd: Vec<f64> = (0..n).map(|i| 0.2 * ((i as f64 * 2.03).sin())).collect();

        let tau = body.inverse_dynamics(&q, &qd, &qdd, GRAVITY).unwrap();
        let qdd_recovered = body.forward_dynamics(&q, &qd, &tau, GRAVITY).unwrap();
        for i in 0..n {
            assert!(
                (qdd_recovered[i] - qdd[i]).abs() < 1e-6,
                "dof {i}: recovered={} want={}",
                qdd_recovered[i],
                qdd[i]
            );
        }
    }

    fn potential_energy(profile: &RobotProfile, q: &[f64], gravity: Vec3) -> f64 {
        let poses = forward_kinematics(profile, q).unwrap();
        profile
            .links
            .iter()
            .enumerate()
            .map(|(index, link)| {
                let pose = poses.by_index(index).unwrap();
                let com_world = pose.apply(link.mass.center_of_mass);
                -link.mass.mass.value() * gravity.dot(com_world)
            })
            .sum()
    }

    #[test]
    fn free_motion_conserves_energy_like_the_single_pendulum_benchmark_does() {
        let profile = double_pendulum(1.0, 0.5, 0.6, 0.35);
        let body = ArticulatedBody::from_profile(&profile);
        let mut state = MultibodyState {
            q: vec![0.6, -0.3],
            qd: vec![0.0, 0.0],
        };

        let e0 = body.kinetic_energy(&state.q, &state.qd).unwrap()
            + potential_energy(&profile, &state.q, GRAVITY);

        let dt = Duration::from_micros(200);
        for _ in 0..50_000 {
            state.step(&body, &[0.0, 0.0], GRAVITY, dt).unwrap();
        }

        let e1 = body.kinetic_energy(&state.q, &state.qd).unwrap()
            + potential_energy(&profile, &state.q, GRAVITY);
        let drift = (e1 - e0).abs() / e0.abs();
        assert!(
            drift < 0.05,
            "symplectic Euler energy drift too large: {drift} (e0={e0} e1={e1})"
        );
    }

    #[test]
    fn the_real_g1_simulates_without_diverging_when_bolted_at_the_pelvis() {
        // Not a balance/walking claim (no floating base, no ground
        // contact — see this module's doc comment): every joint except
        // the legs/arms is driven by gravity alone from a near-upright
        // posture, bolted at the pelvis, and must not blow up.
        let profile = robot_model::fixtures::unitree_g1_29dof().unwrap();
        let body = ArticulatedBody::from_profile(&profile);
        let mut state = MultibodyState::at_rest(vec![0.01; body.position_width()], body.dof());
        let tau = vec![0.0; body.dof()];
        let dt = Duration::from_micros(500);

        for step in 0..2_000 {
            state.step(&body, &tau, GRAVITY, dt).unwrap();
            for (i, &v) in state.q.iter().enumerate() {
                assert!(v.is_finite(), "step {step} dof {i} diverged: q={v}");
            }
        }
        let energy = body.kinetic_energy(&state.q, &state.qd).unwrap();
        assert!(
            energy.is_finite() && energy < 1.0e6,
            "implausible energy after simulating: {energy}"
        );
    }

    #[test]
    fn the_real_g1_with_a_floating_base_falls_under_gravity() {
        // The actual capability this session's "add a Free joint"
        // request was for: the pelvis is no longer bolted to world, so
        // a G1 with zero applied torque, resting under gravity alone,
        // must fall — its world-frame height must decrease, not stay
        // fixed the way the bolted-at-the-pelvis test above requires.
        let profile = robot_model::fixtures::unitree_g1_29dof_floating_base().unwrap();
        let body = ArticulatedBody::from_profile(&profile);
        assert_eq!(body.dof(), 29 + 6);
        assert_eq!(body.position_width(), 29 + 7);

        // q layout: this crate's own `from_profile` walks
        // `topological_link_order`, and the floating joint is the first
        // (and only) joint attached to the new root (`world`), so its
        // 7-wide position slice is q[0..7]; the G1's original 29 scalar
        // joints follow at q[7..36].
        let mut q = vec![0.0; body.position_width()];
        q[3] = 1.0; // identity quaternion (w, x, y, z)
        let mut state = MultibodyState {
            q,
            qd: vec![0.0; body.dof()],
        };
        let tau = vec![0.0; body.dof()];
        let dt = Duration::from_micros(100);

        let initial_height = state.q[2];
        for step in 0..2_000 {
            state.step(&body, &tau, GRAVITY, dt).unwrap();
            for (i, &v) in state.q.iter().enumerate() {
                assert!(v.is_finite(), "step {step} dof {i} diverged: q={v}");
            }
        }
        assert!(
            state.q[2] < initial_height - 1e-3,
            "pelvis height did not decrease: initial={initial_height} final={}",
            state.q[2]
        );
    }

    #[test]
    fn width_mismatches_are_reported() {
        let profile = point_mass_pendulum(1.0, 1.0);
        let body = ArticulatedBody::from_profile(&profile);
        assert!(matches!(
            body.forward_dynamics(&[0.0, 0.0], &[0.0], &[0.0], GRAVITY),
            Err(DynamicsError::WidthMismatch { what: "q", .. })
        ));
    }

    // --- Free joint: validated in isolation before the real G1 is ever
    // given a floating base. ---

    #[test]
    fn a_free_joint_has_six_dof_and_a_seven_wide_position() {
        let profile = free_body(1.0, InertiaTensor::diagonal(0.1, 0.1, 0.1));
        let body = ArticulatedBody::from_profile(&profile);
        assert_eq!(body.dof(), 6);
        assert_eq!(body.position_width(), 7);
    }

    #[test]
    fn a_free_body_at_rest_with_no_forces_stays_at_rest() {
        let profile = free_body(2.0, InertiaTensor::diagonal(0.1, 0.1, 0.1));
        let body = ArticulatedBody::from_profile(&profile);
        let mut state = MultibodyState::at_rest(FREE_IDENTITY.to_vec(), 6);
        let tau = vec![0.0; 6];
        let dt = Duration::from_millis(10);
        for _ in 0..100 {
            state.step(&body, &tau, Vec3::ZERO, dt).unwrap();
        }
        for (i, &want) in FREE_IDENTITY.iter().enumerate() {
            assert!(
                (state.q[i] - want).abs() < 1e-9,
                "q[{i}]={} want={want}",
                state.q[i]
            );
        }
    }

    #[test]
    fn a_free_body_with_no_forces_translates_at_constant_velocity() {
        let profile = free_body(1.5, InertiaTensor::diagonal(0.2, 0.2, 0.2));
        let body = ArticulatedBody::from_profile(&profile);
        let velocity = [0.0, 0.0, 0.0, 1.0, 2.0, -0.5]; // angular=0, local linear=(1,2,-0.5)
        let mut state = MultibodyState {
            q: FREE_IDENTITY.to_vec(),
            qd: velocity.to_vec(),
        };
        let tau = vec![0.0; 6];
        let dt = Duration::from_millis(5);
        let steps = 200;
        for _ in 0..steps {
            state.step(&body, &tau, Vec3::ZERO, dt).unwrap();
        }
        let t = f64::from(steps) * dt.as_secs_f64();
        for (axis, &v) in velocity[3..6].iter().enumerate() {
            let expected = v * t;
            assert!(
                (state.q[axis] - expected).abs() < 1e-6,
                "axis {axis}: got {} want {expected}",
                state.q[axis]
            );
        }
        // Orientation is untouched by pure translation.
        assert!((state.q[3] - 1.0).abs() < 1e-9);
        for &v in &velocity[3..6] {
            assert!(v.abs() > 0.0, "sanity: test actually moves");
        }
    }

    #[test]
    fn a_free_body_under_gravity_alone_matches_the_discrete_projectile_formula() {
        // Semi-implicit Euler under *constant* acceleration has a known,
        // exact discrete closed form distinct from the continuous
        // x(t) = 0.5*a*t^2: v_n = n*a*dt exactly, and summing v_k*dt from
        // k=1..n gives x_n = a*dt^2*n(n+1)/2 = 0.5*a*t*(t+dt). Checked
        // against that discrete formula, not the continuous one, so this
        // test validates the integrator and the dynamics together
        // without conflating a real bug with the integrator's own known,
        // bounded first-order bias.
        let profile = free_body(3.0, InertiaTensor::diagonal(0.05, 0.05, 0.05));
        let body = ArticulatedBody::from_profile(&profile);
        let mut state = MultibodyState::at_rest(FREE_IDENTITY.to_vec(), 6);
        let tau = vec![0.0; 6];
        let dt = Duration::from_micros(100);
        let steps = 5000u32;
        for _ in 0..steps {
            state.step(&body, &tau, GRAVITY, dt).unwrap();
        }
        let dt_s = dt.as_secs_f64();
        let t = f64::from(steps) * dt_s;
        let expected_z = 0.5 * (-G) * t * (t + dt_s);
        assert!(
            (state.q[2] - expected_z).abs() < 1e-6,
            "z={} want={expected_z}",
            state.q[2]
        );
        // No horizontal motion, no rotation: gravity is purely -Z.
        assert!(state.q[0].abs() < 1e-12 && state.q[1].abs() < 1e-12);
        assert!((state.q[3] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn a_spinning_free_body_with_isotropic_inertia_conserves_angular_velocity() {
        // omega x (I * omega) = omega x (I_scalar * omega) = I_scalar *
        // (omega x omega) = 0 for any omega when I is a scalar multiple
        // of the identity — a torque-free, gyroscopic-coupling-free
        // rotation, independent of this module's own formulas for I.
        let profile = free_body(1.0, InertiaTensor::diagonal(0.05, 0.05, 0.05));
        let body = ArticulatedBody::from_profile(&profile);
        let initial_angular = [0.3, -0.2, 0.5];
        let mut state = MultibodyState {
            q: FREE_IDENTITY.to_vec(),
            qd: vec![
                initial_angular[0],
                initial_angular[1],
                initial_angular[2],
                0.0,
                0.0,
                0.0,
            ],
        };
        let tau = vec![0.0; 6];
        let dt = Duration::from_micros(500);
        for _ in 0..2000 {
            state.step(&body, &tau, Vec3::ZERO, dt).unwrap();
        }
        for (i, &want) in initial_angular.iter().enumerate() {
            assert!(
                (state.qd[i] - want).abs() < 1e-6,
                "angular[{i}]={} want={want}",
                state.qd[i]
            );
        }
    }

    #[test]
    fn forward_dynamics_inverts_inverse_dynamics_on_a_free_body() {
        let profile = free_body(1.3, InertiaTensor::diagonal(0.05, 0.08, 0.03));
        let body = ArticulatedBody::from_profile(&profile);
        let rotation = Quat::from_axis_angle(Vec3::new(0.3, 0.5, 0.2), Radians(0.4)).unwrap();
        let q = [
            0.1, -0.2, 0.3, rotation.w, rotation.x, rotation.y, rotation.z,
        ];
        let qd = [0.1, -0.3, 0.2, 0.4, -0.1, 0.3];
        let qdd = [0.5, -0.2, 0.1, -0.3, 0.2, 0.4];

        let tau = body.inverse_dynamics(&q, &qd, &qdd, GRAVITY).unwrap();
        let qdd_recovered = body.forward_dynamics(&q, &qd, &tau, GRAVITY).unwrap();
        for i in 0..6 {
            assert!(
                (qdd_recovered[i] - qdd[i]).abs() < 1e-7,
                "dof {i}: recovered={} want={}",
                qdd_recovered[i],
                qdd[i]
            );
        }
    }
}

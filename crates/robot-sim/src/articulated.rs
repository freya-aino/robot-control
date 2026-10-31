//! Multi-body forward and inverse dynamics over a `robot_model::RobotProfile`'s
//! joint tree, via the Articulated-Body Algorithm (forward) and the
//! Recursive Newton-Euler Algorithm (inverse) — Featherstone, *Rigid Body
//! Dynamics Algorithms*, chapters 4 and 7.
//!
//! # Scope: sphere ground contact; the base may now be fixed or free
//!
//! [`ArticulatedBody`] handles every joint `robot_model::joint::JointKind`
//! can express: the single-DOF kinds (`Revolute`/`Continuous`/`Prismatic`),
//! `Fixed` (zero DOF), and `Free` (six DOF — translation plus rotation, a
//! floating base). A profile's root link attaches to an explicit, typically
//! massless `world` link via a `Free` joint to let the attached body (e.g.
//! a humanoid's pelvis) fall and move under gravity and applied wrench;
//! `crates/robot-sim/SIMULATION_SPECS.md` Section 3 is the design record
//! for why this is a `robot-model`/`robot-sim` construct never importable
//! from URDF. Ground contact is a penalty model applied to sphere collision
//! shapes only (see [`crate::contact::GroundContact`]); box, cylinder and
//! mesh shapes are counted but not modelled.
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

use crate::contact::GroundContact;
use crate::spatial::{
    add3, add6, compose_pose, cross3, cross_motion_force, cross_motion_motion, dot3, dot6,
    inertia_apply, inertia_from_mass, inertia_transform_to_parent, join6, mat3_mul_vec,
    mat3_transpose_mul_vec, matrix6_add, matrix6_sub, rotation_from_quat, scale3, scale6, split6,
    sub6, transform_force, transform_motion,
};
use robot_model::joint::JointKind;
use robot_model::link::CollisionGeometry;
use robot_model::profile::RobotProfile;
use robot_types::frames::Vec3;

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
    /// A joint's inertia matrix could not be inverted. A valid profile never
    /// does this; a state that has diverged to huge finite values can, once
    /// the intermediate products overflow.
    SingularJointInertia,
    /// A ground-contact parameter set was not physically valid.
    InvalidGroundContact,
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
            Self::SingularJointInertia => {
                write!(f, "a joint inertia matrix is singular or non-finite")
            }
            Self::InvalidGroundContact => {
                write!(f, "ground contact parameters are not physically valid")
            }
        }
    }
}

impl core::error::Error for DynamicsError {}

/// A rigid pose as (row-major rotation, translation): the algorithm-side
/// representation of what `robot_model::pose::RigidPose` holds at the crate
/// boundary.
type Pose = ([f64; 9], [f64; 3]);

const IDENTITY_ROTATION: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
const IDENTITY_POSE: Pose = (IDENTITY_ROTATION, [0.0; 3]);
const ZERO6: [f64; 6] = [0.0; 6];

/// One body in the articulated tree: a link together with the single
/// joint that attaches it to its parent (the profile's root link is not
/// itself a body here — it is the fixed base every body's chain of
/// parents eventually reaches, unless that root attaches to an explicit
/// `world` link via a `Free` joint, in which case the *former* root is
/// simply the first body in the tree like any other).
///
/// Every field is a plain scalar or array: the boundary adapter in
/// [`ArticulatedBody::from_profile`] converts the `robot-model` joint and
/// link once, so the recursions below never see a model type.
struct Body {
    kind: JointKind,
    /// Joint axis in the joint's local frame (unnormalized, as declared).
    axis: [f64; 3],
    /// The joint origin in the parent link's frame.
    origin: Pose,
    /// The joint's effort limit, used only by the stance hold.
    effort: f64,
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
    /// Row-major 6x6 spatial inertia about the link origin.
    inertia: [f64; 36],
    /// Hard position limits `(lower, upper)` in the joint's own unit, for a
    /// bounded hinge or slide; `None` for an unbounded or non-single-axis joint.
    limit: Option<(f64, f64)>,
    /// Index of this joint among the profile's actuated joints, the order of
    /// the `qd`, `qdd` and `tau` vectors' single-DOF entries.
    actuated: usize,
}

/// A profile's joint tree, preprocessed into the articulated structure
/// the dynamics algorithms recurse over.
pub struct ArticulatedBody {
    bodies: Vec<Body>,
    n_dof: usize,
    n_position: usize,
    n_actuated: usize,
    contact_spheres: Vec<ContactSphere>,
    unmodelled_collision_shapes: usize,
}

/// A sphere collision shape on one body, centred in that body's own link
/// frame, used as a ground-contact point.
struct ContactSphere {
    body: usize,
    center: [f64; 3],
    radius: f64,
}

/// The motion subspace of `kind`: the basis twists (expressed in the
/// joint's own, constant local axes) that its generalized velocity
/// multiplies, returned as a fixed-capacity array and the number `k` of
/// valid entries. `k` matches [`JointKind::dof_count`]: zero for `Fixed`,
/// one twist for every single-axis kind, six (three angular then three
/// linear, matching this module's `Free`-joint velocity layout) for `Free`.
fn motion_subspace_basis(kind: JointKind, axis: [f64; 3]) -> ([[f64; 6]; 6], usize) {
    let mut basis = [ZERO6; 6];
    match kind {
        JointKind::Fixed => (basis, 0),
        JointKind::Revolute | JointKind::Continuous => {
            basis[0] = join6(axis, [0.0; 3]);
            (basis, 1)
        }
        JointKind::Prismatic => {
            basis[0] = join6([0.0; 3], axis);
            (basis, 1)
        }
        JointKind::Free => {
            for (i, twist) in basis.iter_mut().enumerate() {
                twist[i] = 1.0;
            }
            (basis, 6)
        }
    }
}

/// Width of `kind`'s own slice of the `q` vector: 0 (`Fixed`), 1 (every
/// single-axis kind) or 7 (`Free`: 3 translation then 4 quaternion,
/// `w, x, y, z`).
const fn position_width(kind: JointKind) -> usize {
    match kind {
        JointKind::Fixed => 0,
        JointKind::Revolute | JointKind::Continuous | JointKind::Prismatic => 1,
        JointKind::Free => 7,
    }
}

/// Combines a joint's basis twists with its own generalized velocity (or
/// acceleration) slice into one spatial motion vector: `sum_j basis[j] *
/// values[j]`. For a single-DOF joint it collapses to `basis[0] *
/// values[0]` exactly, and for `Fixed` (`k = 0`) it is the zero motion.
fn combine(basis: &[[f64; 6]], values: &[f64]) -> [f64; 6] {
    let mut acc = ZERO6;
    for (s, &v) in basis.iter().zip(values) {
        acc = add6(acc, scale6(*s, v));
    }
    acc
}

/// Normalizes `(w, x, y, z)`; `None` if any entry is non-finite or the norm
/// is below `1e-9` (the same acceptance rule as
/// `robot_types::frames::Quat::from_components`).
fn normalize_quaternion(w: f64, x: f64, y: f64, z: f64) -> Option<[f64; 4]> {
    if !(w.is_finite() && x.is_finite() && y.is_finite() && z.is_finite()) {
        return None;
    }
    let n = (w * w + x * x + y * y + z * z).sqrt();
    if n < 1e-9 {
        return None;
    }
    Some([w / n, x / n, y / n, z / n])
}

/// The rotation of `angle` about `axis`: identity when the angle is not
/// finite or the axis has no usable direction (norm at most `1e-12`).
fn axis_angle_rotation(axis: [f64; 3], angle: f64) -> [f64; 9] {
    if !angle.is_finite() {
        return IDENTITY_ROTATION;
    }
    let n = dot3(axis, axis).sqrt();
    if !n.is_finite() || n <= 1e-12 {
        return IDENTITY_ROTATION;
    }
    let unit = scale3(axis, 1.0 / n);
    let half = angle * 0.5;
    let s = half.sin();
    rotation_from_quat(half.cos(), unit[0] * s, unit[1] * s, unit[2] * s)
}

/// The pose a `Free` joint's own seven-number position slice contributes,
/// relative to the joint's origin: translation, then a re-normalized
/// orientation quaternion (identity if the quaternion is unusable).
fn free_joint_pose(position: &[f64]) -> Pose {
    let translation = [position[0], position[1], position[2]];
    let [w, x, y, z] = normalize_quaternion(position[3], position[4], position[5], position[6])
        .unwrap_or([1.0, 0.0, 0.0, 0.0]);
    (rotation_from_quat(w, x, y, z), translation)
}

/// The pose this body's joint contributes at the current configuration
/// `q`, relative to the parent link's own frame: the joint origin composed
/// with the motion the joint's position produces.
fn joint_transform(body: &Body, q: &[f64]) -> Pose {
    let slice = &q[body.position_offset..body.position_offset + body.position_width];
    let value = slice.first().copied().unwrap_or(0.0);
    let (rot, trans) = match body.kind {
        JointKind::Fixed => IDENTITY_POSE,
        JointKind::Revolute | JointKind::Continuous => {
            (axis_angle_rotation(body.axis, value), [0.0; 3])
        }
        JointKind::Prismatic => {
            let n = dot3(body.axis, body.axis).sqrt();
            let shift = if n.is_finite() && n > 1e-12 {
                scale3(scale3(body.axis, 1.0 / n), value)
            } else {
                [0.0; 3]
            };
            (IDENTITY_ROTATION, shift)
        }
        JointKind::Free => free_joint_pose(slice),
    };
    compose_pose(body.origin.0, body.origin.1, rot, trans)
}

/// Solves `A X = B` by LU decomposition with partial pivoting, for a
/// row-major `n x n` matrix `a` (stride `n`) and a row-major `n x m`
/// right-hand side `b` (stride `m`), `n, m <= 6`. The pivot tolerance is
/// `max|A| * 1e-14` (`1e-300` for an all-zero `A`), a pivot magnitude at or
/// below it is singular, and any non-finite entry is singular. Returns
/// `None` on singularity.
#[allow(clippy::needless_range_loop)]
fn solve_dense(mut a: [f64; 36], mut b: [f64; 36], n: usize, m: usize) -> Option<[f64; 36]> {
    if a[..n * n].iter().any(|v| !v.is_finite()) || b[..n * m].iter().any(|v| !v.is_finite()) {
        return None;
    }
    let scale = a[..n * n].iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
    let tol = if scale > 0.0 { scale * 1e-14 } else { 1e-300 };

    for col in 0..n {
        let mut pivot_row = col;
        let mut pivot_mag = a[col * n + col].abs();
        for r in (col + 1)..n {
            let mag = a[r * n + col].abs();
            if mag > pivot_mag {
                pivot_mag = mag;
                pivot_row = r;
            }
        }
        if pivot_mag <= tol {
            return None;
        }
        if pivot_row != col {
            for c in 0..n {
                a.swap(col * n + c, pivot_row * n + c);
            }
            for c in 0..m {
                b.swap(col * m + c, pivot_row * m + c);
            }
        }
        let pivot = a[col * n + col];
        for r in (col + 1)..n {
            let factor = a[r * n + col] / pivot;
            if factor == 0.0 {
                continue;
            }
            for c in col..n {
                let v = a[col * n + c];
                a[r * n + c] -= factor * v;
            }
            for c in 0..m {
                let v = b[col * m + c];
                b[r * m + c] -= factor * v;
            }
        }
    }

    let mut x = [0.0; 36];
    for c in 0..m {
        for r in (0..n).rev() {
            let mut sum = b[r * m + c];
            for k in (r + 1)..n {
                sum -= a[r * n + k] * x[k * m + c];
            }
            x[r * m + c] = sum / a[r * n + r];
        }
    }
    Some(x)
}

impl ArticulatedBody {
    /// Preprocesses `profile` into the articulated structure the
    /// dynamics algorithms recurse over. This is the boundary adapter: it
    /// converts `robot-model` poses, vectors and mass properties into the
    /// plain arrays every later function uses.
    ///
    /// Assumes `profile.validate()` has already passed, exactly as
    /// `robot_model::kinematics::forward_kinematics` assumes it — a tree
    /// structure and a single root are checked once at load, not on
    /// every dynamics call.
    #[must_use]
    pub fn from_profile(profile: &RobotProfile) -> Self {
        let mut dof_slots = Vec::new();
        let mut position_slots = Vec::new();
        let mut next_dof = 0usize;
        let mut next_position = 0usize;
        for j in profile.actuated_joints() {
            let dof_width = j.kind.dof_count();
            dof_slots.push((j.id, (next_dof, dof_width)));
            next_dof += dof_width;

            let pos_width = position_width(j.kind);
            position_slots.push((j.id, (next_position, pos_width)));
            next_position += pos_width;
        }

        let order = profile.topological_link_order();
        let mut body_index = Vec::new();
        let mut bodies: Vec<Body> = Vec::with_capacity(order.len().saturating_sub(1));
        let mut contact_spheres = Vec::new();
        let mut unmodelled_collision_shapes = 0usize;

        for link_id in order.into_iter().skip(1) {
            let joint = profile
                .parent_joint_of(link_id)
                .expect("every non-root link has exactly one parent joint in a validated profile");
            let link = profile
                .link(link_id)
                .expect("link exists in a validated profile");
            let parent = body_index
                .iter()
                .find(|(id, _)| *id == joint.parent)
                .map(|(_, index)| *index);
            let index = bodies.len();
            let (dof_offset, dof_width) = dof_slots
                .iter()
                .find(|(id, _)| *id == joint.id)
                .map_or((0, 0), |(_, slot)| *slot);
            let (position_offset, pos_width) = position_slots
                .iter()
                .find(|(id, _)| *id == joint.id)
                .map_or((0, 0), |(_, slot)| *slot);

            let actuated = profile
                .actuated_joints()
                .position(|a| a.id == joint.id)
                .unwrap_or(0);
            let limit = match (joint.kind, joint.limits.position) {
                (JointKind::Revolute | JointKind::Prismatic, Some(range)) => {
                    Some((range.lo.value(), range.hi.value()))
                }
                _ => None,
            };
            let q = joint.origin.rotation;
            let t = joint.origin.translation;
            let props = &link.mass;
            let com = props.center_of_mass;
            let i = &props.inertia;
            bodies.push(Body {
                kind: joint.kind,
                axis: [joint.axis.x, joint.axis.y, joint.axis.z],
                origin: (rotation_from_quat(q.w, q.x, q.y, q.z), [t.x, t.y, t.z]),
                effort: joint.limits.effort.0,
                parent,
                dof_offset,
                dof_width,
                position_offset,
                position_width: pos_width,
                inertia: inertia_from_mass(
                    props.mass.value(),
                    [com.x, com.y, com.z],
                    [
                        i.ixx, i.ixy, i.ixz, i.ixy, i.iyy, i.iyz, i.ixz, i.iyz, i.izz,
                    ],
                ),
                limit,
                actuated,
            });
            for shape in &link.collision {
                match shape.geometry {
                    CollisionGeometry::Sphere { radius } => {
                        let c = shape.pose.translation;
                        contact_spheres.push(ContactSphere {
                            body: index,
                            center: [c.x, c.y, c.z],
                            radius,
                        });
                    }
                    _ => unmodelled_collision_shapes += 1,
                }
            }
            body_index.push((link_id, index));
        }

        Self {
            bodies,
            n_dof: next_dof,
            n_position: next_position,
            n_actuated: dof_slots.len(),
            contact_spheres,
            unmodelled_collision_shapes,
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

    /// Number of sphere collision shapes on non-root links, each a ground-
    /// contact point under [`ArticulatedBody::forward_dynamics_with_ground`].
    #[must_use]
    pub fn contact_sphere_count(&self) -> usize {
        self.contact_spheres.len()
    }

    /// World-frame height of the lowest contact sphere's bottom at
    /// configuration `q`, or `None` if the body has no contact spheres.
    /// Negative means that sphere is below the plane `z = 0` (penetration).
    pub fn lowest_contact_height(&self, q: &[f64]) -> Result<Option<f64>, DynamicsError> {
        self.check_width("q", q, self.n_position)?;
        let mut world: Vec<Pose> = Vec::with_capacity(self.bodies.len());
        let mut lowest: Option<f64> = None;
        for (i, body) in self.bodies.iter().enumerate() {
            let (rot, trans) = joint_transform(body, q);
            let world_i = body.parent.map_or((rot, trans), |p| {
                compose_pose(world[p].0, world[p].1, rot, trans)
            });
            for sphere in self.contact_spheres.iter().filter(|s| s.body == i) {
                let point = add3(mat3_mul_vec(world_i.0, sphere.center), world_i.1);
                let bottom = point[2] - sphere.radius;
                lowest = Some(lowest.map_or(bottom, |l| l.min(bottom)));
            }
            world.push(world_i);
        }
        Ok(lowest)
    }

    /// Number of non-root collision shapes (box, cylinder, mesh) this module
    /// does not model in ground contact.
    #[must_use]
    pub fn unmodelled_collision_shape_count(&self) -> usize {
        self.unmodelled_collision_shapes
    }

    /// Diagonal of the inverse joint-space mass matrix at configuration `q`,
    /// `(M^-1)_jj`, one entry per DOF. It is the joint-space response to a unit
    /// torque at `j` with zero velocity and zero gravity, and it is what an
    /// explicit velocity update on joint `j` actually divides by. It is not a
    /// dynamics output; stance damping uses it to bound the explicit step.
    pub fn inverse_inertia_diagonal(&self, q: &[f64]) -> Result<Vec<f64>, DynamicsError> {
        self.check_width("q", q, self.n_position)?;
        let zeros = vec![0.0; self.n_dof];
        let mut unit = vec![0.0; self.n_dof];
        let mut diagonal = Vec::with_capacity(self.n_dof);
        for j in 0..self.n_dof {
            unit[j] = 1.0;
            let column = self.forward_dynamics(q, &zeros, &unit, Vec3::ZERO)?;
            diagonal.push(column[j]);
            unit[j] = 0.0;
        }
        Ok(diagonal)
    }

    /// Per-DOF damping gains for an explicit hold at step `dt_seconds`: `kd_max`,
    /// capped per DOF so that the velocity update factor `1 - kd*dt*(M^-1)_jj`
    /// stays at or above one half. Without the cap, light joints (the G1 wrists)
    /// give `kd*dt*(M^-1)_jj` far above 2 and the explicit update diverges.
    pub fn stable_damping(
        &self,
        q: &[f64],
        kd_max: f64,
        dt_seconds: f64,
    ) -> Result<Vec<f64>, DynamicsError> {
        let inverse_inertia = self.inverse_inertia_diagonal(q)?;
        Ok(inverse_inertia
            .iter()
            .map(|m_inv| kd_max.min(0.5 / (m_inv * dt_seconds)))
            .collect())
    }

    /// Writes clamped PD hold torques toward zero posture into `tau`: for each
    /// single-DOF joint `j`, `tau[j] = clamp(-kp*q - kd[j]*qd, ±effort)` with
    /// `effort` the joint's own `JointLimits` effort. `kd` holds one damping
    /// gain per DOF. Every `Fixed` and `Free` slot of `tau` is written as zero,
    /// so the caller's buffer need not be pre-zeroed. This is a stance hold for
    /// qualification runs, not a controller with any balance objective.
    pub fn stance_torques(
        &self,
        q: &[f64],
        qd: &[f64],
        kp: f64,
        kd: &[f64],
        tau: &mut [f64],
    ) -> Result<(), DynamicsError> {
        self.check_width("q", q, self.n_position)?;
        self.check_width("qd", qd, self.n_dof)?;
        self.check_width("kd", kd, self.n_dof)?;
        self.check_width("tau", tau, self.n_dof)?;
        for body in &self.bodies {
            let slots = body.dof_offset..body.dof_offset + body.dof_width;
            if body.dof_width != 1 || body.position_width != 1 {
                tau[slots].fill(0.0);
                continue;
            }
            let effort = body.effort;
            let q_j = q[body.position_offset];
            let qd_j = qd[body.dof_offset];
            tau[body.dof_offset] = (-kp * q_j - kd[body.dof_offset] * qd_j).clamp(-effort, effort);
        }
        Ok(())
    }

    /// The world-frame contact wrench on body `body_index`, expressed in that
    /// body's own frame at its origin. Each sphere contributes the ground's
    /// force at its centre, and the torque that force produces about the
    /// body origin. `body_world` is the body's pose in the world and `v` its
    /// spatial velocity in its own frame. The `Vec3` conversion is confined
    /// to the call into [`GroundContact::force`], whose signature is public.
    fn contact_wrench(
        &self,
        body_index: usize,
        body_world: &Pose,
        v: [f64; 6],
        ground: &GroundContact,
    ) -> [f64; 6] {
        let (rot, trans) = *body_world;
        let (omega, linear) = split6(v);
        let mut total = ZERO6;
        for sphere in self.contact_spheres.iter().filter(|s| s.body == body_index) {
            let c = sphere.center;
            let point_world = add3(mat3_mul_vec(rot, c), trans);
            let point_velocity_body = add3(linear, cross3(omega, c));
            let velocity_world = mat3_mul_vec(rot, point_velocity_body);
            let force_world = ground.force(
                Vec3::new(point_world[0], point_world[1], point_world[2]),
                sphere.radius,
                Vec3::new(velocity_world[0], velocity_world[1], velocity_world[2]),
            );
            let force_body =
                mat3_transpose_mul_vec(rot, [force_world.x, force_world.y, force_world.z]);
            total = add6(total, join6(cross3(c, force_body), force_body));
        }
        total
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
        let base_accel = join6([0.0; 3], [-gravity.x, -gravity.y, -gravity.z]);

        let n = self.bodies.len();
        let mut v: Vec<[f64; 6]> = Vec::with_capacity(n);
        let mut a: Vec<[f64; 6]> = Vec::with_capacity(n);
        let mut f: Vec<[f64; 6]> = Vec::with_capacity(n);
        let mut transforms: Vec<Pose> = Vec::with_capacity(n);

        for body in &self.bodies {
            let (rot, trans) = joint_transform(body, q);
            let (s, k) = motion_subspace_basis(body.kind, body.axis);
            let dof = body.dof_offset..body.dof_offset + body.dof_width;
            let joint_v = combine(&s[..k], &qd[dof.clone()]);
            let joint_a = combine(&s[..k], &qdd[dof]);

            let v_parent = body.parent.map_or(ZERO6, |p| v[p]);
            let a_parent = body.parent.map_or(base_accel, |p| a[p]);

            let v_i = add6(transform_motion(rot, trans, v_parent), joint_v);
            let a_i = add6(
                add6(transform_motion(rot, trans, a_parent), joint_a),
                cross_motion_motion(v_i, joint_v),
            );
            let f_i = add6(
                inertia_apply(&body.inertia, a_i),
                cross_motion_force(v_i, inertia_apply(&body.inertia, v_i)),
            );

            v.push(v_i);
            a.push(a_i);
            f.push(f_i);
            transforms.push((rot, trans));
        }

        let mut tau = vec![0.0; self.n_dof];
        for i in (0..n).rev() {
            let body = &self.bodies[i];
            let (s, k) = motion_subspace_basis(body.kind, body.axis);
            for (j, s_j) in s[..k].iter().enumerate() {
                tau[body.dof_offset + j] = dot6(f[i], *s_j);
            }
            if let Some(p) = body.parent {
                let (rot, trans) = transforms[i];
                f[p] = add6(f[p], transform_force(rot, trans, f[i]));
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
        self.articulated_accelerations(q, qd, tau, gravity, None)
    }

    /// Forward dynamics with penalty ground contact on every sphere
    /// collision shape (see [`crate::contact::GroundContact`]). Contact
    /// forces enter the bias force of each body, so the same closed-form
    /// ABA recursion produces the accelerations; with `ground` absent the
    /// result is exactly [`ArticulatedBody::forward_dynamics`].
    pub fn forward_dynamics_with_ground(
        &self,
        q: &[f64],
        qd: &[f64],
        tau: &[f64],
        gravity: Vec3,
        ground: &GroundContact,
    ) -> Result<Vec<f64>, DynamicsError> {
        if !ground.is_valid() {
            return Err(DynamicsError::InvalidGroundContact);
        }
        self.articulated_accelerations(q, qd, tau, gravity, Some(ground))
    }

    /// Position of each bounded joint within its range, as a fraction in
    /// `[0, 1]` (0 at the lower limit, 1 at the upper). Indexed by the
    /// profile's actuated-joint order; `None` for joints without a limit.
    pub fn limit_fractions(&self, q: &[f64]) -> Result<Vec<Option<f64>>, DynamicsError> {
        self.check_width("q", q, self.n_position)?;
        let mut out = vec![None; self.n_actuated];
        for b in &self.bodies {
            if let Some((lo, hi)) = b.limit {
                let x = q[b.position_offset];
                out[b.actuated] = Some(if hi > lo { (x - lo) / (hi - lo) } else { 0.5 });
            }
        }
        Ok(out)
    }

    /// Applies hard joint limits to the unconstrained accelerations `qdd`.
    ///
    /// A bounded joint that is at a limit and accelerating into it is locked by
    /// a constraint torque chosen so that its acceleration becomes zero. The
    /// constraint torques of all locked joints solve one small linear system,
    /// `M⁻¹_SS c = -qdd_S`, with the columns `M⁻¹ e_j` taken from forward
    /// dynamics at zero velocity and no gravity. A limit can only push: a
    /// lower-limit torque must be non-negative and an upper-limit torque
    /// non-positive. A joint whose torque has the wrong sign is released for
    /// this step.
    ///
    /// Returns the constraint torque of every DOF, zero where no limit acted.
    fn apply_hard_limits(
        &self,
        q: &[f64],
        qd: &[f64],
        qdd: &mut [f64],
    ) -> Result<Vec<f64>, DynamicsError> {
        let _ = qd;
        let n = self.n_dof;
        let mut constraint = vec![0.0; n];
        let candidates: Vec<(usize, usize, f64, f64)> = self
            .bodies
            .iter()
            .filter_map(|b| {
                b.limit
                    .map(|(lo, hi)| (b.dof_offset, b.position_offset, lo, hi))
            })
            .collect();
        if candidates.is_empty() {
            return Ok(constraint);
        }
        let zeros = vec![0.0; n];
        let mut columns: Vec<Option<Vec<f64>>> = vec![None; n];
        let mut blocked: Vec<usize> = Vec::new();
        let tol = 1e-12;
        for _pass in 0..=candidates.len() {
            // Joints that must be locked now: at a limit, accelerating outward.
            let mut active: Vec<(usize, f64)> = Vec::new();
            for &(dof, pos, lo, hi) in &candidates {
                if blocked.contains(&dof) {
                    continue;
                }
                let x = q[pos];
                if x <= lo + tol && qdd[dof] < 0.0 {
                    active.push((dof, -1.0));
                } else if x >= hi - tol && qdd[dof] > 0.0 {
                    active.push((dof, 1.0));
                }
            }
            if active.is_empty() {
                break;
            }
            for &(dof, _) in &active {
                if columns[dof].is_none() {
                    let mut e = vec![0.0; n];
                    e[dof] = 1.0;
                    columns[dof] = Some(self.articulated_accelerations(
                        q,
                        &zeros,
                        &e,
                        Vec3::new(0.0, 0.0, 0.0),
                        None,
                    )?);
                }
            }
            let k = active.len();
            let mut a = vec![vec![0.0; k]; k];
            let mut rhs = vec![0.0; k];
            for (i, &(di, _)) in active.iter().enumerate() {
                rhs[i] = -qdd[di];
                for (j, &(dj, _)) in active.iter().enumerate() {
                    a[i][j] = columns[dj].as_ref().map_or(0.0, |col| col[di]);
                }
            }
            let Some(c) = solve_small_dense(a, rhs) else {
                break;
            };
            let mut released = false;
            for (i, &(dof, side)) in active.iter().enumerate() {
                let wrong_sign = (side < 0.0 && c[i] < 0.0) || (side > 0.0 && c[i] > 0.0);
                if wrong_sign {
                    blocked.push(dof);
                    released = true;
                }
            }
            if released {
                continue;
            }
            for (i, &(dof, _)) in active.iter().enumerate() {
                constraint[dof] = c[i];
                if let Some(col) = &columns[dof] {
                    for (acc, v) in qdd.iter_mut().zip(col) {
                        *acc += c[i] * v;
                    }
                }
            }
            break;
        }
        Ok(constraint)
    }

    #[allow(clippy::too_many_lines)]
    fn articulated_accelerations(
        &self,
        q: &[f64],
        qd: &[f64],
        tau: &[f64],
        gravity: Vec3,
        ground: Option<&GroundContact>,
    ) -> Result<Vec<f64>, DynamicsError> {
        self.check_width("q", q, self.n_position)?;
        self.check_width("qd", qd, self.n_dof)?;
        self.check_width("tau", tau, self.n_dof)?;

        let base_accel = join6([0.0; 3], [-gravity.x, -gravity.y, -gravity.z]);
        let n = self.bodies.len();

        let mut v: Vec<[f64; 6]> = Vec::with_capacity(n);
        let mut c: Vec<[f64; 6]> = Vec::with_capacity(n);
        let mut transforms: Vec<Pose> = Vec::with_capacity(n);
        let mut ia: Vec<[f64; 36]> = Vec::with_capacity(n);
        let mut pa: Vec<[f64; 6]> = Vec::with_capacity(n);
        let mut bases: Vec<([[f64; 6]; 6], usize)> = Vec::with_capacity(n);

        // Pass 1 (root -> leaf): velocities, bias accelerations, and each
        // body's own (not-yet-reduced) articulated inertia/bias force.
        let mut world: Vec<Pose> = Vec::with_capacity(n);
        for (i, body) in self.bodies.iter().enumerate() {
            let (rot, trans) = joint_transform(body, q);
            let (s, k) = motion_subspace_basis(body.kind, body.axis);
            let joint_v = combine(
                &s[..k],
                &qd[body.dof_offset..body.dof_offset + body.dof_width],
            );

            let v_parent = body.parent.map_or(ZERO6, |p| v[p]);
            let v_i = add6(transform_motion(rot, trans, v_parent), joint_v);
            let c_i = cross_motion_motion(v_i, joint_v);
            let world_i = body.parent.map_or((rot, trans), |p| {
                compose_pose(world[p].0, world[p].1, rot, trans)
            });
            let external = ground.map_or(ZERO6, |g| self.contact_wrench(i, &world_i, v_i, g));
            let pa_i = sub6(
                cross_motion_force(v_i, inertia_apply(&body.inertia, v_i)),
                external,
            );

            v.push(v_i);
            c.push(c_i);
            transforms.push((rot, trans));
            world.push(world_i);
            ia.push(body.inertia);
            pa.push(pa_i);
            bases.push((s, k));
        }

        // Pass 2 (leaf -> root): eliminate each joint's own `k` degrees
        // of freedom, folding its effective inertia/bias force into its
        // parent. U/D/u are retained per body (not just the effective,
        // parent-folded quantities) for pass 3 below.
        let mut u_list: Vec<[[f64; 6]; 6]> = vec![[ZERO6; 6]; n];
        let mut u_scalar_list: Vec<[f64; 6]> = vec![ZERO6; n];
        let mut d_mats: Vec<[f64; 36]> = vec![[0.0; 36]; n];

        for i in (0..n).rev() {
            let body = &self.bodies[i];
            let (s, k) = bases[i];

            let (ia_eff, pa_eff) = if k == 0 {
                (ia[i], add6(pa[i], inertia_apply(&ia[i], c[i])))
            } else {
                // U_a = I^A S_a, D_ab = S_a . U_b, u_a = tau_a - S_a . pA.
                let mut u_wrenches = [ZERO6; 6];
                for a_idx in 0..k {
                    u_wrenches[a_idx] = inertia_apply(&ia[i], s[a_idx]);
                }
                let mut d_mat = [0.0; 36];
                for a_idx in 0..k {
                    for b_idx in 0..k {
                        d_mat[a_idx * k + b_idx] = dot6(u_wrenches[b_idx], s[a_idx]);
                    }
                }
                let tau_slice = &tau[body.dof_offset..body.dof_offset + body.dof_width];
                let mut u_scalar = [0.0; 6];
                for a_idx in 0..k {
                    u_scalar[a_idx] = tau_slice[a_idx] - dot6(pa[i], s[a_idx]);
                }

                // U as a 6 x k matrix (stride k) and U^T as k x 6 (stride 6).
                let mut u_mat = [0.0; 36];
                let mut u_t = [0.0; 36];
                for col in 0..k {
                    for row in 0..6 {
                        u_mat[row * k + col] = u_wrenches[col][row];
                        u_t[col * 6 + row] = u_wrenches[col][row];
                    }
                }

                // I^a = I^A - U D^-1 U^T.
                let d_inv_ut =
                    solve_dense(d_mat, u_t, k, 6).ok_or(DynamicsError::SingularJointInertia)?;
                let mut reduction = [0.0; 36];
                for row in 0..6 {
                    for col in 0..6 {
                        let mut sum = 0.0;
                        for a_idx in 0..k {
                            sum += u_mat[row * k + a_idx] * d_inv_ut[a_idx * 6 + col];
                        }
                        reduction[row * 6 + col] = sum;
                    }
                }
                let ia_eff = matrix6_sub(&ia[i], &reduction);

                // p^a = pA + I^a c + U D^-1 u.
                let z = solve_dense(d_mat, u_scalar_as_column(&u_scalar), k, 1)
                    .ok_or(DynamicsError::SingularJointInertia)?;
                let mut u_dinv_u = ZERO6;
                for (row, out) in u_dinv_u.iter_mut().enumerate() {
                    let mut sum = 0.0;
                    for a_idx in 0..k {
                        sum += u_mat[row * k + a_idx] * z[a_idx];
                    }
                    *out = sum;
                }
                let pa_eff = add6(add6(pa[i], inertia_apply(&ia_eff, c[i])), u_dinv_u);

                u_list[i] = u_wrenches;
                u_scalar_list[i] = u_scalar;
                d_mats[i] = d_mat;
                (ia_eff, pa_eff)
            };

            if let Some(p) = body.parent {
                let (rot, trans) = transforms[i];
                ia[p] = matrix6_add(&ia[p], &inertia_transform_to_parent(&ia_eff, rot, trans));
                pa[p] = add6(pa[p], transform_force(rot, trans, pa_eff));
            }
        }

        // Pass 3 (root -> leaf): solve for qdd given the parent's now-known
        // true acceleration, using each body's own (non-reduced) U/D/u.
        let mut qdd = vec![0.0; self.n_dof];
        let mut a: Vec<[f64; 6]> = Vec::with_capacity(n);
        for (i, body) in self.bodies.iter().enumerate() {
            let (s, k) = bases[i];
            let a_parent = body.parent.map_or(base_accel, |p| a[p]);
            let (rot, trans) = transforms[i];
            let a_prime = add6(transform_motion(rot, trans, a_parent), c[i]);

            let a_i = if k == 0 {
                a_prime
            } else {
                let mut rhs = [0.0; 36];
                for a_idx in 0..k {
                    rhs[a_idx] = u_scalar_list[i][a_idx] - dot6(u_list[i][a_idx], a_prime);
                }
                let qdd_col =
                    solve_dense(d_mats[i], rhs, k, 1).ok_or(DynamicsError::SingularJointInertia)?;
                let mut a_i = a_prime;
                for a_idx in 0..k {
                    let qdd_val = qdd_col[a_idx];
                    qdd[body.dof_offset + a_idx] = qdd_val;
                    a_i = add6(a_i, scale6(s[a_idx], qdd_val));
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

        let mut v: Vec<[f64; 6]> = Vec::with_capacity(self.bodies.len());
        let mut total = 0.0;
        for body in &self.bodies {
            let (rot, trans) = joint_transform(body, q);
            let (s, k) = motion_subspace_basis(body.kind, body.axis);
            let joint_v = combine(
                &s[..k],
                &qd[body.dof_offset..body.dof_offset + body.dof_width],
            );
            let v_parent = body.parent.map_or(ZERO6, |p| v[p]);
            let v_i = add6(transform_motion(rot, trans, v_parent), joint_v);
            total += 0.5 * dot6(inertia_apply(&body.inertia, v_i), v_i);
            v.push(v_i);
        }
        Ok(total)
    }
}

/// A length-`k` scalar vector laid out as a `k x 1` right-hand side.
fn u_scalar_as_column(u: &[f64; 6]) -> [f64; 36] {
    let mut out = [0.0; 36];
    out[..6].copy_from_slice(u);
    out
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
        self.step_inner(body, tau, gravity, None, dt).map(|_| ())
    }

    /// As [`MultibodyState::step`], but with penalty ground contact on every
    /// sphere collision shape of `body` (see
    /// [`ArticulatedBody::forward_dynamics_with_ground`]).
    pub fn step_with_ground(
        &mut self,
        body: &ArticulatedBody,
        tau: &[f64],
        gravity: Vec3,
        ground: &GroundContact,
        dt: robot_types::time::Duration,
    ) -> Result<(), DynamicsError> {
        self.step_inner(body, tau, gravity, Some(ground), dt)
            .map(|_| ())
    }

    /// As [`MultibodyState::step_with_ground`], and returns the constraint torque
    /// of every DOF that the hard joint limits applied this step (zero where no
    /// limit acted).
    pub fn step_with_ground_recording(
        &mut self,
        body: &ArticulatedBody,
        tau: &[f64],
        gravity: Vec3,
        ground: &GroundContact,
        dt: robot_types::time::Duration,
    ) -> Result<Vec<f64>, DynamicsError> {
        self.step_inner(body, tau, gravity, Some(ground), dt)
    }

    fn step_inner(
        &mut self,
        body: &ArticulatedBody,
        tau: &[f64],
        gravity: Vec3,
        ground: Option<&GroundContact>,
        dt: robot_types::time::Duration,
    ) -> Result<Vec<f64>, DynamicsError> {
        let mut qdd = match ground {
            Some(g) => body.forward_dynamics_with_ground(&self.q, &self.qd, tau, gravity, g)?,
            None => body.forward_dynamics(&self.q, &self.qd, tau, gravity)?,
        };
        let constraint = body.apply_hard_limits(&self.q, &self.qd, &mut qdd)?;
        let dt_s = dt.as_secs_f64();
        for (v, a) in self.qd.iter_mut().zip(&qdd) {
            *v += a * dt_s;
        }
        for b in &body.bodies {
            if b.kind == JointKind::Free {
                Self::integrate_free_position(
                    &mut self.q[b.position_offset..b.position_offset + 7],
                    &self.qd[b.dof_offset..b.dof_offset + 6],
                    dt_s,
                );
            } else if b.dof_width > 0 {
                self.q[b.position_offset] += self.qd[b.dof_offset] * dt_s;
                // Hard limits: no overshoot. A joint that reaches a limit is held
                // there, and its outward velocity is removed.
                if let Some((lo, hi)) = b.limit {
                    let x = &mut self.q[b.position_offset];
                    let v = &mut self.qd[b.dof_offset];
                    if *x < lo {
                        *x = lo;
                        if *v < 0.0 {
                            *v = 0.0;
                        }
                    } else if *x > hi {
                        *x = hi;
                        if *v > 0.0 {
                            *v = 0.0;
                        }
                    }
                }
            }
        }
        Ok(constraint)
    }

    /// Integrates a `Free` joint's own seven-number position slice
    /// (translation then quaternion `w, x, y, z`) forward by `dt_s`
    /// given its six-number velocity slice (angular then linear,
    /// matching [`motion_subspace_basis`]'s `Free` ordering — both
    /// expressed in the body's own local frame, this module's spatial-
    /// algebra convention throughout).
    ///
    /// Translation integrates in the *world* frame
    /// (`translation += dt * R * local_linear_velocity`, since the stored
    /// linear velocity is local); orientation integrates via the standard
    /// local-angular-velocity quaternion derivative `qdot = 0.5 * q *
    /// omega_local` (`omega_local` as a pure quaternion), a first-order
    /// step re-normalized afterward ([`normalize_quaternion`] does the
    /// renormalizing). This is a stated simplification: a higher-order or
    /// exponential-map update would integrate orientation more accurately
    /// per step, and is an open option, not a claim that this is the only
    /// valid approach.
    fn integrate_free_position(position: &mut [f64], velocity: &[f64], dt_s: f64) {
        let angular = [velocity[0], velocity[1], velocity[2]];
        let linear_local = [velocity[3], velocity[4], velocity[5]];
        let translation = [position[0], position[1], position[2]];
        let [w, x, y, z] = normalize_quaternion(position[3], position[4], position[5], position[6])
            .unwrap_or([1.0, 0.0, 0.0, 0.0]);

        let rotation = rotation_from_quat(w, x, y, z);
        let new_translation = add3(
            translation,
            scale3(mat3_mul_vec(rotation, linear_local), dt_s),
        );

        // Hamilton product q * (0, omega).
        let qdot_w = -(x * angular[0] + y * angular[1] + z * angular[2]);
        let qdot_x = w * angular[0] + y * angular[2] - z * angular[1];
        let qdot_y = w * angular[1] - x * angular[2] + z * angular[0];
        let qdot_z = w * angular[2] + x * angular[1] - y * angular[0];
        let half_dt = 0.5 * dt_s;
        let [nw, nx, ny, nz] = normalize_quaternion(
            w + half_dt * qdot_w,
            x + half_dt * qdot_x,
            y + half_dt * qdot_y,
            z + half_dt * qdot_z,
        )
        .unwrap_or([w, x, y, z]);

        position[0] = new_translation[0];
        position[1] = new_translation[1];
        position[2] = new_translation[2];
        position[3] = nw;
        position[4] = nx;
        position[5] = ny;
        position[6] = nz;
    }
}

/// Solves `a x = b` by Gaussian elimination with partial pivoting. Returns
/// `None` if a pivot is numerically zero.
fn solve_small_dense(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-14 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..n {
            let f = a[row][col] / a[col][col];
            let pivot_row = a[col].clone();
            for (cell, p) in a[row][col..].iter_mut().zip(&pivot_row[col..]) {
                *cell -= f * p;
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut sum = b[row];
        for k in row + 1..n {
            sum -= a[row][k] * x[k];
        }
        x[row] = sum / a[row][row];
    }
    Some(x)
}

#[cfg(test)]
pub(crate) mod tests {
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
    pub(crate) fn double_pendulum(m1: f64, l1: f64, m2: f64, l2: f64) -> RobotProfile {
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

    /// As [`free_body`], but with one sphere collision shape of `radius`
    /// centred at the body origin: the minimal fixture for ground contact.
    fn free_sphere_body(mass: f64, radius: f64) -> RobotProfile {
        let mut profile = free_body(mass, InertiaTensor::diagonal(0.1, 0.1, 0.1));
        let body_link = profile
            .links
            .iter_mut()
            .find(|l| l.id == LinkId::new(1))
            .expect("free_body has a body link");
        body_link.collision.push(robot_model::link::CollisionShape {
            geometry: robot_model::link::CollisionGeometry::Sphere { radius },
            pose: RigidPose::IDENTITY,
        });
        profile
    }

    /// `[tx, ty, tz, qw, qx, qy, qz]` at the origin with identity
    /// orientation — the canonical "at rest" `Free`-joint position.
    const FREE_IDENTITY: [f64; 7] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];

    const G: f64 = 9.806_65;
    const GRAVITY: Vec3 = Vec3::new(0.0, 0.0, -G);

    #[test]
    fn dense_solver_solves_a_known_2x2_system() {
        let mut a = [0.0; 36];
        a[..4].copy_from_slice(&[3.0, 2.0, 1.0, 2.0]);
        let mut b = [0.0; 36];
        b[..2].copy_from_slice(&[4.0, 0.0]);

        let x = solve_dense(a, b, 2, 1).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-14);
        assert!((x[1] + 1.0).abs() < 1e-14);
    }

    #[test]
    fn dense_solver_pivots_when_the_leading_entry_is_zero() {
        let mut a = [0.0; 36];
        a[..4].copy_from_slice(&[0.0, 2.0, 1.0, 3.0]);
        let mut b = [0.0; 36];
        b[..2].copy_from_slice(&[-2.0, -1.0]);

        let x = solve_dense(a, b, 2, 1).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-14);
        assert!((x[1] + 1.0).abs() < 1e-14);
    }

    #[test]
    fn dense_solver_rejects_a_rank_deficient_matrix() {
        let mut a = [0.0; 36];
        a[..4].copy_from_slice(&[1.0, 2.0, 2.0, 4.0]);
        let mut b = [0.0; 36];
        b[..2].copy_from_slice(&[3.0, 6.0]);

        assert!(solve_dense(a, b, 2, 1).is_none());
    }

    #[test]
    fn dense_solver_uses_relative_pivot_tolerance() {
        // Case 1: diag(1e6, 1e-9) -> None
        let mut a1 = [0.0; 36];
        a1[0] = 1e6;
        a1[3] = 1e-9;
        let mut b1 = [0.0; 36];
        b1[0] = 1.0;
        b1[1] = 1.0;
        assert!(
            solve_dense(a1, b1, 2, 1).is_none(),
            "diag(1e6, 1e-9) should be singular"
        );

        // Case 2: diag(1e-6, 1e-19) -> Some
        // Tol = 1e-6 * 1e-14 = 1e-20. 1e-19 > 1e-20, so Some.
        let mut a2 = [0.0; 36];
        a2[0] = 1e-6;
        a2[3] = 1e-19;
        let mut b2 = [0.0; 36];
        b2[0] = 1.0;
        b2[1] = 1.0;
        assert!(
            solve_dense(a2, b2, 2, 1).is_some(),
            "diag(1e-6, 1e-19) should be non-singular"
        );
    }

    #[test]
    fn dense_solver_rejects_zero_and_non_finite_inputs() {
        let zero = [0.0; 36];
        assert!(solve_dense(zero, zero, 2, 1).is_none());

        let mut a = [0.0; 36];
        a[0] = f64::NAN;
        let mut b = [0.0; 36];
        b[0] = 1.0;
        assert!(solve_dense(a, b, 1, 1).is_none());

        let mut a_inf = [0.0; 36];
        a_inf[0] = 1.0;
        a_inf[4] = 1.0;
        a_inf[8] = 1.0;
        let mut b_inf = [0.0; 36];
        b_inf[0] = f64::INFINITY;
        assert!(solve_dense(a_inf, b_inf, 3, 1).is_none());
    }

    #[test]
    fn dense_solver_solves_a_dense_6x6_system_with_six_rhs_columns() {
        // A = [ 0.0, 2.0, 1.0, 1.0, 1.0, 1.0;
        //       1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
        //       1.0, 1.0, 2.0, 1.0, 1.0, 1.0;
        //       1.0, 1.0, 1.0, 2.0, 1.0, 1.0;
        //       1.0, 1.0, 1.0, 1.0, 2.0, 1.0;
        //       1.0, 1.0, 1.0, 1.0, 1.0, 2.0 ]
        // Row 0 has a 0.0 at (0,0), forcing a swap.
        let mut a = [0.0; 36];
        a[0] = 0.0;
        a[1] = 2.0;
        a[2] = 1.0;
        a[3] = 1.0;
        a[4] = 1.0;
        a[5] = 1.0;
        a[6] = 1.0;
        a[7] = 1.0;
        a[8] = 1.0;
        a[9] = 1.0;
        a[10] = 1.0;
        a[11] = 1.0;
        a[12] = 1.0;
        a[13] = 1.0;
        a[14] = 2.0;
        a[15] = 1.0;
        a[16] = 1.0;
        a[17] = 1.0;
        a[18] = 1.0;
        a[19] = 1.0;
        a[20] = 1.0;
        a[21] = 2.0;
        a[22] = 1.0;
        a[23] = 1.0;
        a[24] = 1.0;
        a[25] = 1.0;
        a[26] = 1.0;
        a[27] = 1.0;
        a[28] = 2.0;
        a[29] = 1.0;
        a[30] = 1.0;
        a[31] = 1.0;
        a[32] = 1.0;
        a[33] = 1.0;
        a[34] = 1.0;
        a[35] = 2.0;

        let mut x_expected = [0.0; 36];
        for i in 0..6 {
            x_expected[i * 6 + i] = 1.0;
        } // X is identity

        let mut b = [0.0; 36];
        // B = A * X_expected = A * I = A
        b.copy_from_slice(&a);

        let x = solve_dense(a, b, 6, 6).unwrap();
        for i in 0..36 {
            assert!((x[i] - x_expected[i]).abs() < 1e-12, "entry {i}");
        }
    }

    #[test]
    fn massless_revolute_joint_reports_singular_joint_inertia() {
        let profile = wrap_profile(
            vec![revolute(
                0,
                "massless_joint",
                0,
                1,
                RigidPose::IDENTITY,
                Vec3::unit_y(),
            )],
            vec![
                empty_link(0, "root", MassProperties::MASSLESS),
                empty_link(1, "massless", MassProperties::MASSLESS),
            ],
        );
        profile
            .validate()
            .expect("massless fixture has a valid tree");
        let body = ArticulatedBody::from_profile(&profile);

        assert!(matches!(
            body.forward_dynamics(&[0.0], &[0.0], &[0.0], GRAVITY),
            Err(DynamicsError::SingularJointInertia)
        ));
    }

    #[test]
    fn degenerate_axis_angle_uses_identity_rotation() {
        assert_eq!(axis_angle_rotation([0.0; 3], 0.7), IDENTITY_ROTATION);
        assert_eq!(
            axis_angle_rotation([1e-12, 0.0, 0.0], 0.7),
            IDENTITY_ROTATION
        );
        assert_eq!(
            axis_angle_rotation([0.0, 1.0, 0.0], f64::NAN),
            IDENTITY_ROTATION
        );
    }

    #[test]
    fn a_hard_limit_holds_a_swinging_pendulum_inside_its_range() {
        // Released from rest under gravity with a limit at ±0.3 rad, the joint
        // must stay inside the range, reach the limit, and carry a constraint
        // torque there. Without the limit the same pendulum swings past it.
        let mut limited = point_mass_pendulum(1.0, 1.0);
        limited.joints[0].limits = robot_model::joint::JointLimits::bounded(
            robot_types::units::Range {
                lo: robot_types::units::Radians(-0.3),
                hi: robot_types::units::Radians(0.3),
            },
            robot_types::units::RadiansPerSecond(100.0),
            robot_types::units::NewtonMetres(1.0e6),
        );
        let body = ArticulatedBody::from_profile(&limited);
        let mut state = MultibodyState {
            q: vec![0.0],
            qd: vec![0.0],
        };
        let gravity = Vec3::new(0.0, 0.0, -9.81);
        let dt = Duration::from_secs_f64(1e-3);
        let tau = vec![0.0];
        let ground = GroundContact {
            stiffness: 1.0,
            normal_damping: 0.0,
            friction: 0.0,
            friction_regularization: 1.0,
            ground_height: -100.0,
        };
        let mut max_q: f64 = 0.0;
        let mut max_constraint: f64 = 0.0;
        for _ in 0..3000 {
            let c = state
                .step_with_ground_recording(&body, &tau, gravity, &ground, dt)
                .unwrap();
            max_q = max_q.max(state.q[0].abs());
            max_constraint = max_constraint.max(c[0].abs());
            assert!(
                state.q[0] >= -0.3 - 1e-12 && state.q[0] <= 0.3 + 1e-12,
                "{}",
                state.q[0]
            );
        }
        assert!(
            (max_q - 0.3).abs() < 1e-9,
            "the pendulum should reach the limit, reached {max_q}"
        );
        assert!(max_constraint > 0.0, "a limit must have acted");

        let mut free = point_mass_pendulum(1.0, 1.0);
        free.joints[0].limits = robot_model::joint::JointLimits::unbounded(
            robot_types::units::RadiansPerSecond(100.0),
            robot_types::units::NewtonMetres(0.0),
        );
        let free_body = ArticulatedBody::from_profile(&free);
        let mut unlimited = MultibodyState {
            q: vec![0.0],
            qd: vec![0.0],
        };
        let mut max_free: f64 = 0.0;
        for _ in 0..3000 {
            unlimited
                .step_with_ground_recording(&free_body, &tau, gravity, &ground, dt)
                .unwrap();
            max_free = max_free.max(unlimited.q[0].abs());
        }
        assert!(
            max_free > 1.0,
            "without a limit the pendulum swings past 0.3 rad, reached {max_free}"
        );
    }

    #[test]
    fn a_hard_limit_does_not_inject_energy_into_a_pendulum() {
        // Without ground contact, total energy (kinetic plus potential) of a
        // pendulum that hits a limit must not rise above its starting value.
        let mut profile = point_mass_pendulum(1.0, 1.0);
        profile.joints[0].limits = robot_model::joint::JointLimits::bounded(
            robot_types::units::Range {
                lo: robot_types::units::Radians(-0.3),
                hi: robot_types::units::Radians(0.3),
            },
            robot_types::units::RadiansPerSecond(100.0),
            robot_types::units::NewtonMetres(1.0e6),
        );
        let body = ArticulatedBody::from_profile(&profile);
        let gravity = Vec3::new(0.0, 0.0, -9.81);
        let energy = |state: &MultibodyState| {
            // Bob at local +X, length 1, pivot at the origin: height = -cos(q) * ...
            // The bob's world height is x sin? For axis +Y, a rotation by q moves
            // +X to (cos q, 0, -sin q), so the bob's height is -sin(q).
            let h = -state.q[0].sin();
            body.kinetic_energy(&state.q, &state.qd).unwrap() + 9.81 * h
        };
        let mut state = MultibodyState {
            q: vec![0.25],
            qd: vec![0.0],
        };
        let e0 = energy(&state);
        let tau = vec![0.0];
        let dt = Duration::from_secs_f64(1e-4);
        let mut worst = f64::NEG_INFINITY;
        for _ in 0..20000 {
            state.step(&body, &tau, gravity, dt).unwrap();
            worst = worst.max(energy(&state) - e0);
        }
        assert!(
            worst < 1e-3 * e0.abs().max(1.0),
            "energy rose by {worst} above {e0}"
        );
    }

    #[test]
    fn limit_fraction_is_zero_at_the_lower_limit_and_one_at_the_upper() {
        let mut profile = point_mass_pendulum(1.0, 1.0);
        profile.joints[0].limits = robot_model::joint::JointLimits::bounded(
            robot_types::units::Range {
                lo: robot_types::units::Radians(-0.3),
                hi: robot_types::units::Radians(0.3),
            },
            robot_types::units::RadiansPerSecond(100.0),
            robot_types::units::NewtonMetres(1.0),
        );
        let body = ArticulatedBody::from_profile(&profile);
        let at = |x: f64| body.limit_fractions(&[x]).unwrap()[0].unwrap();
        assert!((at(-0.3) - 0.0).abs() < 1e-12);
        assert!((at(0.3) - 1.0).abs() < 1e-12);
        assert!((at(0.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn prismatic_joint_with_zero_axis_has_no_translation() {
        let body = Body {
            kind: JointKind::Prismatic,
            axis: [0.0; 3],
            origin: IDENTITY_POSE,
            effort: 0.0,
            parent: None,
            dof_offset: 0,
            dof_width: 1,
            position_offset: 0,
            position_width: 1,
            inertia: [0.0; 36],
            limit: None,
            actuated: 0,
        };

        assert_eq!(joint_transform(&body, &[2.5]), IDENTITY_POSE);
    }

    #[test]
    fn free_joint_with_zero_or_nan_quaternion_uses_identity_rotation() {
        let zero_quaternion = free_joint_pose(&[1.0, -2.0, 3.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(zero_quaternion.0, IDENTITY_ROTATION);
        assert_eq!(zero_quaternion.1, [1.0, -2.0, 3.0]);

        let nan_quaternion = free_joint_pose(&[3.0, 2.0, 1.0, f64::NAN, 0.0, 0.0, 0.0]);
        assert_eq!(nan_quaternion.0, IDENTITY_ROTATION);
        assert_eq!(nan_quaternion.1, [3.0, 2.0, 1.0]);
        assert_eq!(normalize_quaternion(0.0, 0.0, 0.0, 0.0), None);
        assert_eq!(normalize_quaternion(f64::NAN, 0.0, 0.0, 1.0), None);
    }

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

    /// Qualification stance for the floating G1, as run by
    /// `robot sim multibody --floating`: the pelvis starts so the lowest
    /// contact sphere clears the plane by 5 mm, then a clamped PD hold runs
    /// for 2 s. Returns the pelvis-height minimum and the state after the run.
    struct StanceRun {
        start_pelvis: f64,
        min_pelvis: f64,
        final_energy: f64,
    }

    fn run_g1_stance(steps: usize, hold: bool) -> StanceRun {
        let profile = robot_model::fixtures::unitree_g1_29dof_floating_base().unwrap();
        let body = ArticulatedBody::from_profile(&profile);
        let ground = test_ground(0.8);
        let mut q = vec![0.0; body.position_width()];
        q[3] = 1.0;
        let lowest = body.lowest_contact_height(&q).unwrap().unwrap();
        q[2] = 0.005 - lowest;
        let start_pelvis = q[2];
        let mut state = MultibodyState {
            q,
            qd: vec![0.0; body.dof()],
        };
        let mut tau = vec![0.0; body.dof()];
        let dt = Duration::from_micros(100);
        let kd = body
            .stable_damping(&state.q, 30.0, dt.as_secs_f64())
            .unwrap();
        let mut min_pelvis = f64::INFINITY;
        for step in 0..steps {
            if hold {
                body.stance_torques(&state.q, &state.qd, 300.0, &kd, &mut tau)
                    .unwrap();
            }
            state
                .step_with_ground(&body, &tau, GRAVITY, &ground, dt)
                .unwrap();
            assert!(
                state.q.iter().chain(&state.qd).all(|v| v.is_finite()),
                "step {step} diverged"
            );
            min_pelvis = min_pelvis.min(state.q[2]);
        }
        let final_energy = body.kinetic_energy(&state.q, &state.qd).unwrap();
        StanceRun {
            start_pelvis,
            min_pelvis,
            final_energy,
        }
    }

    /// The PD-hold stance at `dt = 5e-4` stays finite over 2 s. The failing
    /// zero-torque G1 probe is recorded in `fidelity.rs`, not here.
    #[test]
    fn the_g1_stance_hold_keeps_the_pelvis_above_the_plane_for_two_seconds() {
        // Qualification, not balance: a clamped PD hold toward the zero
        // posture. The pelvis minimum over 2 s must stay within 2 cm of its
        // start height, and the final kinetic energy must be bounded.
        let run = run_g1_stance(20_000, true);
        assert!(
            run.min_pelvis >= run.start_pelvis - 0.02,
            "pelvis fell to {:.4} m from start {:.4} m under the stance hold",
            run.min_pelvis,
            run.start_pelvis
        );
        assert!(
            run.final_energy <= 0.5,
            "stance kinetic energy {:.4} J not bounded",
            run.final_energy
        );
    }

    /// Known-limitation characterization, not a physical claim. The G1 has
    /// only the ten sphere shapes of the USD ankles, and the pelvis and torso have no
    /// collision geometry at all. With zero torque the legs fold and the trunk
    /// reaches the plane through those unmodelled shapes. Adding contact
    /// proxies (see ADR 0010) is expected to change this result; update the
    /// assertion then rather than treating it as a fixed fact.
    #[test]
    fn zero_torque_g1_collapse_is_a_known_limitation_of_the_sphere_only_model() {
        let run = run_g1_stance(10_000, false);
        assert!(
            run.min_pelvis < run.start_pelvis - 0.02,
            "zero torque did not collapse: min pelvis {:.4} m",
            run.min_pelvis
        );
    }

    fn test_ground(friction: f64) -> GroundContact {
        GroundContact {
            stiffness: 20_000.0,
            normal_damping: 200.0,
            friction,
            friction_regularization: 1e-3,
            ground_height: 0.0,
        }
    }

    #[test]
    fn a_dropped_sphere_settles_at_the_penalty_rest_height() {
        // Static balance of the penalty spring: the sphere sits where
        // k * penetration = m * g, so its centre rests at
        // radius - m*g/k above the plane. Damping only removes the bounce.
        let (mass, radius) = (2.0, 0.1);
        let profile = free_sphere_body(mass, radius);
        let body = ArticulatedBody::from_profile(&profile);
        assert_eq!(body.contact_sphere_count(), 1);
        assert_eq!(body.unmodelled_collision_shape_count(), 0);

        let ground = test_ground(0.0);
        let mut q = FREE_IDENTITY.to_vec();
        q[2] = 0.2;
        let mut state = MultibodyState::at_rest(q, 6);
        let tau = vec![0.0; 6];
        let dt = Duration::from_millis(1);
        for _ in 0..6_000 {
            state
                .step_with_ground(&body, &tau, GRAVITY, &ground, dt)
                .unwrap();
        }

        let expected = radius - mass * G / ground.stiffness;
        assert!(
            (state.q[2] - expected).abs() < 1e-6,
            "rest height {} want {expected}",
            state.q[2]
        );
        assert!(
            state.qd[2].abs() < 1e-6,
            "still moving vertically: {}",
            state.qd[2]
        );
    }

    #[test]
    fn frictionless_sliding_at_penalty_equilibrium_conserves_total_energy() {
        // Start at the static penalty equilibrium (k * penetration = m * g)
        // with zero vertical velocity, then slide with mu = 0. The only
        // forces are gravity and the normal spring, so total energy
        // (kinetic + m*g*z + k*pen^2/2) must not drift.
        let (mass, radius) = (1.0, 0.1);
        let profile = free_sphere_body(mass, radius);
        let body = ArticulatedBody::from_profile(&profile);
        let ground = test_ground(0.0);
        let penetration = mass * G / ground.stiffness;

        let mut q = FREE_IDENTITY.to_vec();
        q[2] = radius - penetration;
        let mut state = MultibodyState {
            q,
            qd: vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
        };
        let tau = vec![0.0; 6];
        let dt = Duration::from_micros(100);
        let total_energy = |s: &MultibodyState| {
            let v = &s.qd[3..6];
            let kinetic = 0.5 * mass * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            let pen = (radius - s.q[2]).max(0.0);
            kinetic + mass * G * s.q[2] + 0.5 * ground.stiffness * pen * pen
        };
        let e0 = total_energy(&state);
        for _ in 0..5_000 {
            state
                .step_with_ground(&body, &tau, GRAVITY, &ground, dt)
                .unwrap();
        }
        assert!(
            (state.qd[3] - 2.0).abs() < 1e-9,
            "frictionless contact changed sliding speed: {}",
            state.qd[3]
        );
        let e1 = total_energy(&state);
        assert!(
            (e1 - e0).abs() < 1e-9 * e0,
            "total energy drifted: {e0} -> {e1}"
        );
    }

    /// A rigid box as one free body with four bottom-corner sphere shapes at
    /// `(±half_x, ±half_y, -half_z)`. Unlike the single-sphere fixture, its
    /// corners are offset from the body origin, so the contact torque path
    /// (`r x F` about the origin, and `omega x r` in the velocity) is used.
    fn free_corner_box(mass: f64, half_extents: (f64, f64, f64), radius: f64) -> RobotProfile {
        let (half_x, half_y, half_z) = half_extents;
        let mut profile = free_body(mass, InertiaTensor::diagonal(0.1, 0.1, 0.1));
        let body_link = profile
            .links
            .iter_mut()
            .find(|l| l.id == LinkId::new(1))
            .expect("free_body has a body link");
        for (sx, sy) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
            body_link.collision.push(robot_model::link::CollisionShape {
                geometry: robot_model::link::CollisionGeometry::Sphere { radius },
                pose: RigidPose::from_translation(Vec3::new(sx * half_x, sy * half_y, -half_z)),
            });
        }
        profile
    }

    #[test]
    fn a_tilted_corner_box_settles_flat_at_the_analytic_rest_height() {
        // Four identical corner springs share the load equally once flat, so
        // each corner penetrates m*g/(4k) and the centre rests at
        // r + half_z - m*g/(4k). A start tilt about x makes the offset corner
        // torques do the work, so the box must come back flat with no residual
        // angular velocity.
        let (mass, radius, half_z) = (2.0, 0.02, 0.05);
        let profile = free_corner_box(mass, (0.1, 0.05, half_z), radius);
        let body = ArticulatedBody::from_profile(&profile);
        assert_eq!(body.contact_sphere_count(), 4);

        let ground = test_ground(0.5);
        let tilt = 0.05_f64;
        let mut q = FREE_IDENTITY.to_vec();
        q[2] = 0.2;
        q[3] = (tilt / 2.0).cos();
        q[4] = (tilt / 2.0).sin();
        let mut state = MultibodyState::at_rest(q, 6);
        let tau = vec![0.0; 6];
        let dt = Duration::from_micros(100);
        for _ in 0..30_000 {
            state
                .step_with_ground(&body, &tau, GRAVITY, &ground, dt)
                .unwrap();
        }

        let expected = radius + half_z - mass * G / (4.0 * ground.stiffness);
        assert!(
            (state.q[2] - expected).abs() < 1e-5,
            "rest height {} want {expected}",
            state.q[2]
        );
        assert!(
            state.q[4].abs() < 1e-5 && state.q[5].abs() < 1e-5,
            "box did not return flat: qx={} qy={}",
            state.q[4],
            state.q[5]
        );
        assert!(
            state.qd.iter().all(|v| v.abs() < 1e-4),
            "residual motion: {:?}",
            state.qd
        );
    }

    #[test]
    fn friction_dissipates_sliding_energy_and_never_reverses_it() {
        // Coulomb friction with a resting, penetrating sphere must slow the
        // slide monotonically and stop it, without pushing it backwards.
        let (mass, radius) = (1.0, 0.1);
        let profile = free_sphere_body(mass, radius);
        let body = ArticulatedBody::from_profile(&profile);
        let ground = test_ground(0.5);

        let mut q = FREE_IDENTITY.to_vec();
        q[2] = radius - 1e-3;
        let mut state = MultibodyState {
            q,
            qd: vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
        };
        let tau = vec![0.0; 6];
        let dt = Duration::from_micros(100);
        let mut previous = state.qd[3];
        for step in 0..20_000 {
            state
                .step_with_ground(&body, &tau, GRAVITY, &ground, dt)
                .unwrap();
            assert!(
                state.qd[3] <= previous + 1e-12 && state.qd[3] >= -1e-6,
                "step {step}: sliding speed went {previous} -> {}",
                state.qd[3]
            );
            previous = state.qd[3];
        }
        assert!(state.qd[3].abs() < 1e-3, "friction did not stop the slide");
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

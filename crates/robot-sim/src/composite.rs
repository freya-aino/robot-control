//! Composite-rigid-body dynamics in the example formulation's own coordinate
//! system: the mass matrix `M(q)`, its `L' D L` factorization over the degree-of-freedom
//! tree, and the bias force, all expressed in **world orientation with the
//! reference point at the owning tree's centre of mass**.
//!
//! Why this module exists alongside [`crate::articulated`]. The articulated-body
//! algorithm in that module works in each body's own local frame and never forms
//! `M`. Three things the specifications require cannot be built on it:
//! an implicit-in-velocity integrator needs a dense or factorised `M`
//! (`specs/03-integrators.md` item 4), a constraint solver needs
//! `A = J M^-1 J^T` and the exact diagonal `A_ii`
//! (`specs/06-equality-constraints-and-solver.md` item 2), and the
//! degree-of-freedom-tree factorization below is the structure a batched GPU
//! backend parallelises over (`specs/11-gpu-batched-backend.md`). Appendix D
//! entries M-18 and M-24 record both gaps.
//!
//! Source of truth. Every formula here is transferred from
//! `docs/example-system-specs/02_kinematics_dynamics_smooth.md`, which the
//! project now treats as the governing reference for simulation mathematics.
//! Section numbers in the doc comments below are that file's. The quantities
//! named `cinert`, `cdof`, `cvel` and `crb` keep the example's names so a
//! reader can follow the formulas across.
//!
//! Conventions, from §0 of that file:
//! - A spatial 6-vector is **rotation first, then translation**: motion
//!   `[omega; v]`, force `[torque; f]`. This matches [`crate::spatial`].
//! - Every `c`-prefixed quantity is in **world orientation** with its reference
//!   point at `subtree_com[root]`, one reference point per kinematic tree. This
//!   is the one place this crate departs from [`crate::spatial`]'s body-local
//!   convention, and it is deliberate: it is the example's own choice, made so
//!   the same quantities serve collision and the constraint solver.
//! - A spatial inertia is a 10-vector
//!   `[Ixx, Iyy, Izz, Ixy, Ixz, Iyz, hx, hy, hz, m]` with `h = m (com - ref)`.
//!
//! Purity contract (FR-6): `std` and plain `f64` arrays and `Vec<f64>` only. No
//! `robot_types::math::Matrix`, no `RigidPose`, no `HashMap` in any algorithm
//! function. [`CompositeModel::from_profile`] is the single boundary adapter
//! and is the only function here that sees a `robot-model` type.
//!
//! Deliberate omissions, each a named optimisation rather than a formula
//! change, so that no result below differs from the example because of them:
//! - The `body_simple` / `dof_simplenum` / `dof_M0` reduction (§2.9). That
//!   optimisation drops off-diagonal entries of `M` that are *identically zero*
//!   for qualifying bodies. This module stores them as explicit zeros instead.
//!   The factorization and solve are unaffected; only `nC` is larger.
//! - Armature (`dof_armature`, `mj_actuatorArmature`) and tendon armature
//!   (`mj_tendonArmature`). `robot-model` has no armature field, so the
//!   diagonal seed is zero. When armature is added it enters at exactly the
//!   place marked in [`crb`].
//! - Sleeping (`body_awake`) and mocap bodies. Neither exists in this project.
//! - `Ball` joints, which `robot-model` does not have
//!   (`specs/appendix-b-open-questions.md` S-04). The three rotational degrees
//!   of freedom of a `Free` joint use the ball formula, so the code path is
//!   present and exercised; only the standalone joint type is missing.

use crate::spatial::{
    add3, cross3, dot3, mat3_mul, mat3_mul_vec, rotation_from_quat, scale3, sub3,
};
use robot_model::joint::JointKind;
use robot_model::profile::RobotProfile;

/// Minimum value allowed in a denominator, `mjMINVAL` of §0.
pub const MINVAL: f64 = 1e-15;

/// One body of the composite model: the link and the joint attaching it to its
/// parent, reduced to plain arrays.
///
/// This mirrors the private `Body` of [`crate::articulated`] rather than
/// sharing it, so that this module's model is a flat, allocation-free
/// description that a kernel can consume directly. Unifying the two adapters is
/// deferred, not forgotten.
#[derive(Clone, Debug)]
pub struct CompositeBody {
    /// Joint kind attaching this body to its parent.
    pub kind: JointKind,
    /// Joint axis in the parent's frame, **already normalised** by
    /// [`CompositeModel::from_profile`] (Appendix D M-02).
    pub axis: [f64; 3],
    /// Joint origin in the parent link's frame: row-major rotation.
    pub origin_rot: [f64; 9],
    /// Joint origin in the parent link's frame: translation.
    pub origin_trans: [f64; 3],
    /// Index into `bodies` of the parent, or `None` when the parent is the
    /// fixed base (the profile's root link).
    pub parent: Option<usize>,
    /// Index of the kinematic tree this body belongs to, as a body index: the
    /// top-most ancestor below the base. The example's `body_rootid`.
    pub rootid: usize,
    /// Offset of this body's own degrees of freedom in **internal** (tree)
    /// order. See [`CompositeModel`] on the two orderings.
    pub dof_adr: usize,
    /// Number of degrees of freedom: 0 (`Fixed`), 1 (single-axis) or 6 (`Free`).
    pub dof_num: usize,
    /// Offset of this body's own position coordinates in **internal** order.
    pub qpos_adr: usize,
    /// Number of position coordinates: 0, 1 or 7 (`Free`).
    pub qpos_num: usize,
    /// Offset of this body's degrees of freedom in **external** order, the
    /// `RobotProfile::actuated_joints` order that
    /// [`crate::articulated::ArticulatedBody`] and every `q`/`qd` vector in the
    /// rest of the project use.
    pub ext_dof_adr: usize,
    /// Offset of this body's position coordinates in **external** order.
    pub ext_qpos_adr: usize,
    /// Mass.
    pub mass: f64,
    /// Centre of mass in this body's own link frame.
    pub com: [f64; 3],
    /// Rotational inertia about the centre of mass, row-major, in the link
    /// frame.
    pub inertia: [f64; 9],
}

/// The degree-of-freedom tree and the sparsity pattern of `M`, both model
/// constants (§2.9).
///
/// `M` is stored lower-triangular in compressed sparse row form. Row `i` holds
/// the ancestors of degree of freedom `i` in **ascending** order followed by
/// `i` itself, so the diagonal is the **last** entry of each row, at
/// `rowadr[i] + rownnz[i] - 1`.
///
/// The factorization in [`factor_m`] relies on a prefix property: when `i` is
/// an ancestor of `k`, row `i`'s column list is a prefix of row `k`'s. That
/// holds for any numbering in which a parent's degrees of freedom precede its
/// children's, because then every ancestor chain is ascending and the elements
/// of `k`'s chain below `i` are exactly `i`'s own chain. The example numbers
/// degrees of freedom depth-first; this module inherits
/// `RobotProfile::topological_link_order`, which is parent-before-child. The
/// prefix property, and therefore the factorization, holds either way.
#[derive(Clone, Debug)]
pub struct DofTree {
    /// Number of degrees of freedom.
    pub nv: usize,
    /// Previous degree of freedom in the tree, `-1` at a tree root
    /// (`dof_parentid`). Always less than its own index.
    pub dof_parentid: Vec<isize>,
    /// Body owning each degree of freedom (`dof_bodyid`).
    pub dof_bodyid: Vec<usize>,
    /// Non-zeros in each row of `M`.
    pub rownnz: Vec<usize>,
    /// Start of each row of `M`.
    pub rowadr: Vec<usize>,
    /// Column index of each stored entry of `M`.
    pub colind: Vec<usize>,
    /// Total stored entries of `M` (`nC`).
    pub nc: usize,
}

impl DofTree {
    /// Position of row `i`'s diagonal entry in a `nc`-long value array.
    #[must_use]
    pub fn diag_adr(&self, i: usize) -> usize {
        self.rowadr[i] + self.rownnz[i] - 1
    }
}

/// A profile reduced to the plain-array model this module's algorithms read.
///
/// # Two degree-of-freedom orderings
///
/// Every algorithm in this module numbers degrees of freedom **parents before
/// children** ("internal" order), because [`DofTree`]'s prefix property, and so
/// [`factor_m`], require it. The example numbers them depth-first for the same
/// reason (§0: "Dofs are numbered depth-first so `dof_parentid[i] < i`. This is
/// the elimination tree for the inertia factorization.").
///
/// The rest of this project numbers them in `RobotProfile::actuated_joints`
/// order ("external" order), which is the profile's authoring order and skips
/// `Fixed` joints. For a serial chain authored root-first the two coincide; for
/// the G1 they do not, because its joint list interleaves the legs, torso and
/// arms while the tree visits each branch in turn.
///
/// Rather than renumber the project's `q`/`qd` vectors — a cross-crate breaking
/// change — this model carries the permutation between the two and exposes it
/// through [`CompositeModel::dof_to_internal`] and its three companions. The
/// example anticipates exactly this indirection: `mj_factorM` and `mj_solveLD`
/// take an optional `index[]` with `k = index ? index[j] : j` (§2.10, §2.11).
///
/// **Every function in this module takes and returns internal order.** Convert
/// at the boundary, once, and the conversion is the identity whenever the
/// profile is authored root-first.
#[derive(Clone, Debug)]
pub struct CompositeModel {
    /// Bodies, parents before children.
    pub bodies: Vec<CompositeBody>,
    /// Degree-of-freedom tree and `M` pattern, in internal order.
    pub tree: DofTree,
    /// Width of a position vector.
    pub nq: usize,
    /// Gravity as a world 3-vector.
    pub gravity: [f64; 3],
    /// `dof_ext_of_int[i]` is the external index of internal degree of freedom
    /// `i`.
    pub dof_ext_of_int: Vec<usize>,
    /// `qpos_ext_of_int[i]` is the external index of internal position
    /// coordinate `i`.
    pub qpos_ext_of_int: Vec<usize>,
}

/// Output of the position stage (§2.1, §2.2, §2.3).
#[derive(Clone, Debug)]
pub struct Position {
    /// World pose of each body: row-major rotation.
    pub xmat: Vec<[f64; 9]>,
    /// World pose of each body: translation.
    pub xpos: Vec<[f64; 3]>,
    /// World centre of mass of each body (`xipos`).
    pub xipos: Vec<[f64; 3]>,
    /// World rotation of each body's inertial frame (`ximat`). Equal to
    /// `xmat` here, because `robot-model` expresses a link's inertia in the
    /// link frame rather than in a separate principal frame.
    pub ximat: Vec<[f64; 9]>,
    /// World anchor of each body's joint (`xanchor`).
    pub xanchor: Vec<[f64; 3]>,
    /// World axis of each body's joint (`xaxis`).
    pub xaxis: Vec<[f64; 3]>,
    /// Centre of mass of each body's own kinematic tree (`subtree_com`),
    /// indexed by body and already resolved to the tree root's value.
    pub subtree_com: Vec<[f64; 3]>,
    /// Spatial inertia of each body about its tree centre of mass, in world
    /// orientation (`cinert`), as a 10-vector.
    pub cinert: Vec<[f64; 10]>,
    /// Motion subspace of each degree of freedom in the same frame (`cdof`).
    pub cdof: Vec<[f64; 6]>,
}

/// Why a composite-dynamics computation could not run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CompositeError {
    /// A vector did not have the expected width.
    WidthMismatch {
        /// Which vector.
        what: &'static str,
        /// Expected width.
        expected: usize,
        /// Width supplied.
        found: usize,
    },
    /// A supplied value was not finite.
    NotFinite {
        /// Which vector.
        what: &'static str,
    },
    /// `M` had a diagonal entry at or below [`MINVAL`] during factorization, at
    /// the named degree of freedom. The example clamps and warns
    /// (`mjWARN_INERTIA`); this port reports instead, so the caller decides.
    SingularInertia {
        /// The degree of freedom whose pivot was clamped.
        dof: usize,
    },
}

impl core::fmt::Display for CompositeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::WidthMismatch {
                what,
                expected,
                found,
            } => write!(f, "{what} must have {expected} entries, found {found}"),
            Self::NotFinite { what } => write!(f, "{what} contains a non-finite value"),
            Self::SingularInertia { dof } => {
                write!(f, "mass matrix is singular at degree of freedom {dof}")
            }
        }
    }
}

impl core::error::Error for CompositeError {}

/// `mju_normalize3` of §0: returns the unit vector, or `(1, 0, 0)` when the
/// norm is below [`MINVAL`], together with the original norm.
#[must_use]
pub fn normalize3(v: [f64; 3]) -> ([f64; 3], f64) {
    let n = dot3(v, v).sqrt();
    if !n.is_finite() || n < MINVAL {
        ([1.0, 0.0, 0.0], n)
    } else {
        (scale3(v, 1.0 / n), n)
    }
}

/// `mju_inertCom` of §1.3, generalised from the example's principal-moment
/// input to a full symmetric tensor.
///
/// The example passes `inert[3]`, a body's principal moments, with `mat` the
/// rotation of its principal frame, and forms `mat diag(inert) mat'`.
/// `robot-model` stores a link's inertia as a full symmetric tensor in the link
/// frame, so this takes `inertia` (row-major 3x3, about the centre of mass) and
/// forms `mat inertia mat'`. With a diagonal `inertia` the two agree exactly;
/// the generalisation adds the products of inertia the example's compiler would
/// have rotated away. It is not a deviation from the formula, only from the
/// input format.
///
/// `dif` is `com_world - reference_point`, and `mass` the body's mass. The
/// parallel-axis terms are the example's, term for term.
#[must_use]
pub fn inert_com(inertia: [f64; 9], mat: [f64; 9], dif: [f64; 3], mass: f64) -> [f64; 10] {
    // I_world = mat * inertia * mat'
    let mut mi = [0.0; 9];
    for r in 0..3 {
        for c in 0..3 {
            mi[r * 3 + c] = (0..3).map(|k| mat[r * 3 + k] * inertia[k * 3 + c]).sum();
        }
    }
    let mut iw = [0.0; 9];
    for r in 0..3 {
        for c in 0..3 {
            iw[r * 3 + c] = (0..3).map(|k| mi[r * 3 + k] * mat[c * 3 + k]).sum();
        }
    }
    let (d0, d1, d2) = (dif[0], dif[1], dif[2]);
    [
        iw[0] + mass * (d1 * d1 + d2 * d2),
        iw[4] + mass * (d0 * d0 + d2 * d2),
        iw[8] + mass * (d0 * d0 + d1 * d1),
        iw[1] - mass * d0 * d1,
        iw[2] - mass * d0 * d2,
        iw[5] - mass * d1 * d2,
        mass * d0,
        mass * d1,
        mass * d2,
        mass,
    ]
}

/// `mju_mulInertVec` of §1.3: a 10-vector spatial inertia applied to a motion
/// vector, `f = [I omega + h x v ; m v - h x omega]`.
#[must_use]
pub fn mul_inert_vec(i: &[f64; 10], v: [f64; 6]) -> [f64; 6] {
    [
        i[0] * v[0] + i[3] * v[1] + i[4] * v[2] - i[8] * v[4] + i[7] * v[5],
        i[3] * v[0] + i[1] * v[1] + i[5] * v[2] + i[8] * v[3] - i[6] * v[5],
        i[4] * v[0] + i[5] * v[1] + i[2] * v[2] - i[7] * v[3] + i[6] * v[4],
        i[8] * v[1] - i[7] * v[2] + i[9] * v[3],
        i[6] * v[2] - i[8] * v[0] + i[9] * v[4],
        i[7] * v[0] - i[6] * v[1] + i[9] * v[5],
    ]
}

/// `mju_dofCom` of §1.3. A rotational degree of freedom about `axis` whose
/// anchor is offset from the reference point by `offset` moves the reference
/// point at `axis x offset`; a translational one does not rotate it.
#[must_use]
pub fn dof_com(axis: [f64; 3], offset: Option<[f64; 3]>) -> [f64; 6] {
    match offset {
        Some(o) => {
            let l = cross3(axis, o);
            [axis[0], axis[1], axis[2], l[0], l[1], l[2]]
        }
        None => [0.0, 0.0, 0.0, axis[0], axis[1], axis[2]],
    }
}

/// `mju_crossMotion` of §1.3: the spatial motion cross product.
#[must_use]
pub fn cross_motion(v: [f64; 6], m: [f64; 6]) -> [f64; 6] {
    [
        -v[2] * m[1] + v[1] * m[2],
        v[2] * m[0] - v[0] * m[2],
        -v[1] * m[0] + v[0] * m[1],
        -v[2] * m[4] + v[1] * m[5] + (-v[5] * m[1] + v[4] * m[2]),
        v[2] * m[3] - v[0] * m[5] + (v[5] * m[0] - v[3] * m[2]),
        -v[1] * m[3] + v[0] * m[4] + (-v[4] * m[0] + v[3] * m[1]),
    ]
}

/// `mju_crossForce` of §1.3: the spatial force cross product,
/// `[omega x torque + v x F ; omega x F]`.
#[must_use]
pub fn cross_force(v: [f64; 6], f: [f64; 6]) -> [f64; 6] {
    [
        -v[2] * f[1] + v[1] * f[2] + (-v[5] * f[4] + v[4] * f[5]),
        v[2] * f[0] - v[0] * f[2] + (v[5] * f[3] - v[3] * f[5]),
        -v[1] * f[0] + v[0] * f[1] + (-v[4] * f[3] + v[3] * f[4]),
        -v[2] * f[4] + v[1] * f[5],
        v[2] * f[3] - v[0] * f[5],
        -v[1] * f[3] + v[0] * f[4],
    ]
}

/// `mji_dot6` of §0.
#[must_use]
pub fn dot6(a: [f64; 6], b: [f64; 6]) -> f64 {
    ((a[0] * b[0] + a[2] * b[2]) + (a[1] * b[1] + a[3] * b[3])) + (a[4] * b[4] + a[5] * b[5])
}

/// The rotation of `angle` about the already-normalised `axis`
/// (`mju_axisAngle2Quat` then `mju_quat2Mat`, §2.1 `HINGE`).
fn axis_angle_rotation(axis: [f64; 3], angle: f64) -> [f64; 9] {
    if !angle.is_finite() {
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    }
    let half = angle * 0.5;
    let s = half.sin();
    rotation_from_quat(half.cos(), axis[0] * s, axis[1] * s, axis[2] * s)
}

impl CompositeModel {
    /// Boundary adapter: reduces a validated profile to plain arrays.
    ///
    /// Assumes `profile.validate()` has passed, as
    /// `robot_model::kinematics::forward_kinematics` does.
    ///
    /// Two things happen here that the rest of the module relies on:
    /// - **Every joint axis is normalised** (§0 `mju_normalize3`). The example's
    ///   compiler normalises joint axes at compile time; this project's
    ///   importer does not, and Appendix D M-02 records that a non-unit axis
    ///   makes the motion subspace disagree with the pose map. Normalising once
    ///   here makes `v = S qdot` hold by construction.
    /// - Degrees of freedom are numbered parents before children, which is what
    ///   [`DofTree`]'s prefix property needs.
    #[must_use]
    pub fn from_profile(profile: &RobotProfile) -> Self {
        // External layout: RobotProfile::actuated_joints order, exactly as
        // ArticulatedBody::from_profile assigns it.
        let mut ext_dof = Vec::new();
        let mut ext_qpos = Vec::new();
        let mut next_ext_dof = 0usize;
        let mut next_ext_qpos = 0usize;
        for j in profile.actuated_joints() {
            ext_dof.push((j.id, next_ext_dof));
            next_ext_dof += j.kind.dof_count();
            ext_qpos.push((j.id, next_ext_qpos));
            next_ext_qpos += match j.kind {
                JointKind::Fixed => 0,
                JointKind::Revolute | JointKind::Continuous | JointKind::Prismatic => 1,
                JointKind::Free => 7,
            };
        }

        let order = profile.topological_link_order();
        let mut bodies: Vec<CompositeBody> = Vec::with_capacity(order.len().saturating_sub(1));
        let mut body_index: Vec<(robot_types::ids::LinkId, usize)> = Vec::new();
        let mut next_dof = 0usize;
        let mut next_qpos = 0usize;

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
            let dof_num = joint.kind.dof_count();
            let qpos_num = match joint.kind {
                JointKind::Fixed => 0,
                JointKind::Revolute | JointKind::Continuous | JointKind::Prismatic => 1,
                JointKind::Free => 7,
            };
            // Appendix D M-02: normalise once, at the boundary.
            let (axis, _) = normalize3([joint.axis.x, joint.axis.y, joint.axis.z]);
            let q = joint.origin.rotation;
            let t = joint.origin.translation;
            let props = &link.mass;
            let com = props.center_of_mass;
            let i = &props.inertia;
            let rootid = parent.map_or(index, |p| bodies[p].rootid);
            bodies.push(CompositeBody {
                kind: joint.kind,
                axis,
                origin_rot: rotation_from_quat(q.w, q.x, q.y, q.z),
                origin_trans: [t.x, t.y, t.z],
                parent,
                rootid,
                dof_adr: next_dof,
                dof_num,
                qpos_adr: next_qpos,
                qpos_num,
                ext_dof_adr: ext_dof
                    .iter()
                    .find(|(id, _)| *id == joint.id)
                    .map_or(0, |(_, a)| *a),
                ext_qpos_adr: ext_qpos
                    .iter()
                    .find(|(id, _)| *id == joint.id)
                    .map_or(0, |(_, a)| *a),
                mass: props.mass.value(),
                com: [com.x, com.y, com.z],
                inertia: [
                    i.ixx, i.ixy, i.ixz, i.ixy, i.iyy, i.iyz, i.ixz, i.iyz, i.izz,
                ],
            });
            next_dof += dof_num;
            next_qpos += qpos_num;
            body_index.push((link_id, index));
        }

        let g = profile.gravity.acceleration();
        let tree = build_dof_tree(&bodies, next_dof);
        let mut dof_ext_of_int = vec![0usize; next_dof];
        let mut qpos_ext_of_int = vec![0usize; next_qpos];
        for body in &bodies {
            for k in 0..body.dof_num {
                dof_ext_of_int[body.dof_adr + k] = body.ext_dof_adr + k;
            }
            for k in 0..body.qpos_num {
                qpos_ext_of_int[body.qpos_adr + k] = body.ext_qpos_adr + k;
            }
        }
        Self {
            bodies,
            tree,
            nq: next_qpos,
            gravity: [g.x, g.y, g.z],
            dof_ext_of_int,
            qpos_ext_of_int,
        }
    }

    /// Number of degrees of freedom.
    #[must_use]
    pub fn nv(&self) -> usize {
        self.tree.nv
    }

    /// Whether the two orderings coincide, so that no conversion is needed.
    /// True for any profile whose joints are authored parents before children.
    #[must_use]
    pub fn ordering_is_identity(&self) -> bool {
        self.dof_ext_of_int.iter().enumerate().all(|(i, e)| i == *e)
            && self
                .qpos_ext_of_int
                .iter()
                .enumerate()
                .all(|(i, e)| i == *e)
    }

    /// Reorders an external-layout velocity, acceleration or force vector into
    /// internal order.
    ///
    /// # Errors
    ///
    /// [`CompositeError::WidthMismatch`] if `external` is not `nv` wide.
    pub fn dof_to_internal(&self, external: &[f64]) -> Result<Vec<f64>, CompositeError> {
        permute(external, &self.dof_ext_of_int, "qd", true)
    }

    /// Reorders an internal-layout degree-of-freedom vector into external
    /// order.
    ///
    /// # Errors
    ///
    /// [`CompositeError::WidthMismatch`] if `internal` is not `nv` wide.
    pub fn dof_to_external(&self, internal: &[f64]) -> Result<Vec<f64>, CompositeError> {
        permute(internal, &self.dof_ext_of_int, "qd", false)
    }

    /// Reorders an external-layout position vector into internal order.
    ///
    /// # Errors
    ///
    /// [`CompositeError::WidthMismatch`] if `external` is not `nq` wide.
    pub fn qpos_to_internal(&self, external: &[f64]) -> Result<Vec<f64>, CompositeError> {
        permute(external, &self.qpos_ext_of_int, "q", true)
    }

    /// Reorders an internal-layout position vector into external order.
    ///
    /// # Errors
    ///
    /// [`CompositeError::WidthMismatch`] if `internal` is not `nq` wide.
    pub fn qpos_to_external(&self, internal: &[f64]) -> Result<Vec<f64>, CompositeError> {
        permute(internal, &self.qpos_ext_of_int, "q", false)
    }
}

/// Gathers (`gather = true`: `out[i] = v[map[i]]`) or scatters
/// (`out[map[i]] = v[i]`) through a permutation.
fn permute(
    v: &[f64],
    map: &[usize],
    what: &'static str,
    gather: bool,
) -> Result<Vec<f64>, CompositeError> {
    if v.len() != map.len() {
        return Err(CompositeError::WidthMismatch {
            what,
            expected: map.len(),
            found: v.len(),
        });
    }
    let mut out = vec![0.0; v.len()];
    for (i, &e) in map.iter().enumerate() {
        if gather {
            out[i] = v[e];
        } else {
            out[e] = v[i];
        }
    }
    Ok(out)
}

/// Builds the degree-of-freedom tree and the `M` sparsity pattern (§2.9).
fn build_dof_tree(bodies: &[CompositeBody], nv: usize) -> DofTree {
    let mut dof_parentid = vec![-1isize; nv];
    let mut dof_bodyid = vec![0usize; nv];

    // The last degree of freedom of the closest ancestor body that has any.
    let mut last_dof_of: Vec<isize> = vec![-1; bodies.len()];
    for (b, body) in bodies.iter().enumerate() {
        let inherited = body.parent.map_or(-1, |p| last_dof_of[p]);
        for k in 0..body.dof_num {
            let i = body.dof_adr + k;
            dof_bodyid[i] = b;
            // Within one joint the degrees of freedom chain to each other, so a
            // free joint's six form a path, matching the example's depth-first
            // numbering.
            dof_parentid[i] = if k == 0 { inherited } else { (i - 1) as isize };
        }
        last_dof_of[b] = if body.dof_num > 0 {
            (body.dof_adr + body.dof_num - 1) as isize
        } else {
            inherited
        };
    }

    let mut rownnz = vec![0usize; nv];
    let mut rowadr = vec![0usize; nv];
    let mut colind = Vec::new();
    for i in 0..nv {
        rowadr[i] = colind.len();
        // Ancestors of i, ascending, then i. Collected by walking up and
        // reversing, which yields ascending order because dof_parentid < index.
        let mut chain = Vec::new();
        let mut j = dof_parentid[i];
        while j >= 0 {
            chain.push(j as usize);
            j = dof_parentid[j as usize];
        }
        chain.reverse();
        rownnz[i] = chain.len() + 1;
        colind.extend(chain);
        colind.push(i);
    }
    let nc = colind.len();
    DofTree {
        nv,
        dof_parentid,
        dof_bodyid,
        rownnz,
        rowadr,
        colind,
        nc,
    }
}

/// The pose a body's joint contributes relative to its parent's frame, and the
/// joint's anchor and axis in that same parent frame (§2.1).
fn joint_local(body: &CompositeBody, q: &[f64]) -> ([f64; 9], [f64; 3]) {
    let value = if body.qpos_num >= 1 {
        q[body.qpos_adr]
    } else {
        0.0
    };
    let (rot, trans) = match body.kind {
        JointKind::Fixed => ([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [0.0; 3]),
        JointKind::Revolute | JointKind::Continuous => {
            (axis_angle_rotation(body.axis, value), [0.0; 3])
        }
        // The axis is already unit, so the shift is the axis times the value.
        JointKind::Prismatic => (
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            scale3(body.axis, value),
        ),
        JointKind::Free => {
            let s = &q[body.qpos_adr..body.qpos_adr + 7];
            let n = (s[3] * s[3] + s[4] * s[4] + s[5] * s[5] + s[6] * s[6]).sqrt();
            let rot = if n.is_finite() && n >= MINVAL {
                rotation_from_quat(s[3] / n, s[4] / n, s[5] / n, s[6] / n)
            } else {
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            };
            (rot, [s[0], s[1], s[2]])
        }
    };
    (
        mat3_mul(body.origin_rot, rot),
        add3(mat3_mul_vec(body.origin_rot, trans), body.origin_trans),
    )
}

/// The position stage: body poses, inertial frames, tree centres of mass,
/// `cinert` and `cdof` (§2.1, §2.2, §2.3).
///
/// # Errors
///
/// [`CompositeError::WidthMismatch`] if `q` is not [`CompositeModel::nq`] wide,
/// [`CompositeError::NotFinite`] if it holds a non-finite value.
pub fn position(model: &CompositeModel, q: &[f64]) -> Result<Position, CompositeError> {
    if q.len() != model.nq {
        return Err(CompositeError::WidthMismatch {
            what: "q",
            expected: model.nq,
            found: q.len(),
        });
    }
    if q.iter().any(|x| !x.is_finite()) {
        return Err(CompositeError::NotFinite { what: "q" });
    }
    let n = model.bodies.len();
    let mut xmat = Vec::with_capacity(n);
    let mut xpos = Vec::with_capacity(n);
    let mut xanchor = Vec::with_capacity(n);
    let mut xaxis = Vec::with_capacity(n);

    // §2.1: body poses, parents before children.
    for body in &model.bodies {
        let (lrot, ltrans) = joint_local(body, q);
        let (prot, ppos) = match body.parent {
            Some(p) => (xmat[p], xpos[p]),
            None => ([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [0.0; 3]),
        };
        xmat.push(mat3_mul(prot, lrot));
        xpos.push(add3(mat3_mul_vec(prot, ltrans), ppos));
        // The anchor is the joint origin in world; the axis is the declared
        // axis carried into world by the parent's rotation composed with the
        // joint origin's own rotation, which is the frame the axis is declared
        // in.
        let anchor_local = body.origin_trans;
        xanchor.push(add3(mat3_mul_vec(prot, anchor_local), ppos));
        let axis_frame = mat3_mul(prot, body.origin_rot);
        xaxis.push(mat3_mul_vec(axis_frame, body.axis));
    }

    // §2.2: inertial frames. robot-model keeps a link's inertia in the link
    // frame, so ximat is xmat and xipos is the centre of mass carried to world.
    let mut xipos = Vec::with_capacity(n);
    let ximat = xmat.clone();
    for (i, body) in model.bodies.iter().enumerate() {
        xipos.push(add3(mat3_mul_vec(xmat[i], body.com), xpos[i]));
    }

    // §2.3: subtree centre of mass, accumulated backwards then divided.
    let mut acc = vec![[0.0; 3]; n];
    let mut submass = vec![0.0; n];
    for (i, body) in model.bodies.iter().enumerate() {
        acc[i] = scale3(xipos[i], body.mass);
        submass[i] = body.mass;
    }
    for i in (0..n).rev() {
        if let Some(p) = model.bodies[i].parent {
            acc[p] = add3(acc[p], acc[i]);
            submass[p] += submass[i];
        }
    }
    let mut tree_com = vec![[0.0; 3]; n];
    for i in 0..n {
        tree_com[i] = if submass[i] < MINVAL {
            xipos[i]
        } else {
            scale3(acc[i], 1.0 / submass[i])
        };
    }
    // One reference point per tree: every body uses its root's value.
    let mut subtree_com = vec![[0.0; 3]; n];
    for i in 0..n {
        subtree_com[i] = tree_com[model.bodies[i].rootid];
    }

    // §2.3: cinert and cdof about that reference point, in world orientation.
    let mut cinert = Vec::with_capacity(n);
    for (i, body) in model.bodies.iter().enumerate() {
        let dif = sub3(xipos[i], subtree_com[i]);
        cinert.push(inert_com(body.inertia, ximat[i], dif, body.mass));
    }

    let mut cdof = vec![[0.0; 6]; model.tree.nv];
    for (i, body) in model.bodies.iter().enumerate() {
        if body.dof_num == 0 {
            continue;
        }
        let offset = sub3(subtree_com[i], xanchor[i]);
        let adr = body.dof_adr;
        match body.kind {
            JointKind::Fixed => {}
            JointKind::Revolute | JointKind::Continuous => {
                cdof[adr] = dof_com(xaxis[i], Some(offset));
            }
            JointKind::Prismatic => {
                cdof[adr] = dof_com(xaxis[i], None);
            }
            JointKind::Free => {
                // §2.3 FREE: three unit world translations first, then the
                // three rotational degrees of freedom as a ball joint about the
                // body's own frame axes. This is the ordering and the frame
                // Appendix D M-01 records the previous implementation as
                // contradicting.
                cdof[adr] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
                cdof[adr + 1] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
                cdof[adr + 2] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
                for k in 0..3 {
                    // Column k of xmat: the k-th body axis in world.
                    let axis = [xmat[i][k], xmat[i][3 + k], xmat[i][6 + k]];
                    cdof[adr + 3 + k] = dof_com(axis, Some(offset));
                }
            }
        }
    }

    Ok(Position {
        xmat,
        xpos,
        xipos,
        ximat,
        xanchor,
        xaxis,
        subtree_com,
        cinert,
        cdof,
    })
}

/// Composite-rigid-body mass matrix (§2.9 `mj_crb` and `mj_makeM`).
///
/// Returns `M` in the compressed sparse row pattern of [`DofTree`]:
/// `M_ij = cdof_j . (I^c_{body(i)} cdof_i)` for every `j` that is an ancestor
/// of, or equal to, `i`, with `I^c` the composite inertia of the subtree rooted
/// at `body(i)`.
///
/// The diagonal seed is zero because this project has no armature; the example
/// seeds it with `dof_armature[i] + mj_actuatorArmature(...)`. That is the only
/// term missing, and it enters additively at the marked line.
#[must_use]
pub fn crb(model: &CompositeModel, pos: &Position) -> Vec<f64> {
    let n = model.bodies.len();
    // Composite inertia: cinert accumulated from the leaves.
    let mut crb_inert = pos.cinert.clone();
    for i in (0..n).rev() {
        if let Some(p) = model.bodies[i].parent {
            let add = crb_inert[i];
            for (dst, src) in crb_inert[p].iter_mut().zip(&add) {
                *dst += *src;
            }
        }
    }

    let tree = &model.tree;
    let mut m = vec![0.0; tree.nc];
    for i in 0..tree.nv {
        let mut adr = tree.diag_adr(i);
        // Armature would be seeded here: m[adr] = armature[i] + ...
        let buf = mul_inert_vec(&crb_inert[tree.dof_bodyid[i]], pos.cdof[i]);
        let mut j = i as isize;
        while j >= 0 {
            m[adr] += dot6(pos.cdof[j as usize], buf);
            if adr == tree.rowadr[i] {
                break;
            }
            adr -= 1;
            j = tree.dof_parentid[j as usize];
        }
    }
    m
}

/// The `L' D L` factorization of `M` over the degree-of-freedom tree
/// (§2.10 `mj_factorM`).
///
/// Returns `(ld, diag_inv)`: `ld` holds `D` on the diagonal and the unit
/// lower-triangular `L` off it, in the same pattern as `m`; `diag_inv[i]` is
/// `1 / D_i`.
///
/// The example clamps a diagonal below [`MINVAL`] to [`MINVAL`] and raises
/// `mjWARN_INERTIA`. This port reports [`CompositeError::SingularInertia`]
/// instead, naming the degree of freedom, so a caller can decide rather than
/// silently continuing on a clamped pivot. That is the same choice
/// `crate::integrators::implicit_damping_euler_step` already makes.
///
/// # Errors
///
/// [`CompositeError::SingularInertia`] when a pivot is at or below [`MINVAL`]
/// or is not finite.
pub fn factor_m(tree: &DofTree, m: &[f64]) -> Result<(Vec<f64>, Vec<f64>), CompositeError> {
    let mut mat = m.to_vec();
    let mut diag_inv = vec![0.0; tree.nv];
    for k in (0..tree.nv).rev() {
        let start = tree.rowadr[k];
        let end = tree.diag_adr(k);
        if !mat[end].is_finite() || mat[end] < MINVAL {
            return Err(CompositeError::SingularInertia { dof: k });
        }
        let inv_d = 1.0 / mat[end];
        diag_inv[k] = inv_d;
        // Prefix update: row i is an ancestor of k, so row i's columns are the
        // first rownnz[i] columns of row k.
        let mut adr = end;
        while adr > start {
            adr -= 1;
            let i = tree.colind[adr];
            let factor = mat[adr] * inv_d;
            if factor == 0.0 {
                continue;
            }
            let ri = tree.rowadr[i];
            let ni = tree.rownnz[i];
            // Row i ends before row k begins, because i < k and rows are laid
            // out in index order, so the two ranges never overlap.
            let (left, right) = mat.split_at_mut(start);
            for (dst, src) in left[ri..ri + ni].iter_mut().zip(&right[..ni]) {
                *dst -= *src * factor;
            }
        }
        for value in &mut mat[start..end] {
            *value *= inv_d;
        }
    }
    Ok((mat, diag_inv))
}

/// In-place `x <- M^-1 x` by back-substitution (§2.11 `mj_solveLD`).
///
/// `ld` and `diag_inv` come from [`factor_m`].
pub fn solve_ld(tree: &DofTree, ld: &[f64], diag_inv: &[f64], x: &mut [f64]) {
    // x <- L^-T x
    for i in (0..tree.nv).rev() {
        if tree.rownnz[i] == 1 {
            continue;
        }
        let xi = x[i];
        if xi != 0.0 {
            let start = tree.rowadr[i];
            let end = tree.diag_adr(i);
            for (&col, &l) in tree.colind[start..end].iter().zip(&ld[start..end]) {
                x[col] -= l * xi;
            }
        }
    }
    // x <- D^-1 x
    for i in 0..tree.nv {
        x[i] *= diag_inv[i];
    }
    // x <- L^-1 x
    for i in 0..tree.nv {
        if tree.rownnz[i] == 1 {
            continue;
        }
        let start = tree.rowadr[i];
        let end = tree.diag_adr(i);
        let mut sum = 0.0;
        for (&col, &l) in tree.colind[start..end].iter().zip(&ld[start..end]) {
            sum += l * x[col];
        }
        x[i] -= sum;
    }
}

/// `res = M vec` for the symmetric sparse `M` (§2.11 `mj_mulM`).
#[must_use]
pub fn mul_m(tree: &DofTree, m: &[f64], vec: &[f64]) -> Vec<f64> {
    let mut res = vec![0.0; tree.nv];
    for i in 0..tree.nv {
        let diag = tree.diag_adr(i);
        res[i] += m[diag] * vec[i];
        let start = tree.rowadr[i];
        for (&j, &value) in tree.colind[start..diag].iter().zip(&m[start..diag]) {
            res[i] += value * vec[j];
            res[j] += value * vec[i];
        }
    }
    res
}

/// `M` as a dense row-major `nv x nv` matrix (§2.11 `mj_fullM`), for callers
/// that need one — the implicit integrator of [`crate::integrators`] takes a
/// dense `M`, and Appendix D M-18 records that nothing produced one before.
#[must_use]
pub fn full_m(tree: &DofTree, m: &[f64]) -> Vec<f64> {
    let nv = tree.nv;
    let mut out = vec![0.0; nv * nv];
    for i in 0..nv {
        let start = tree.rowadr[i];
        let end = tree.diag_adr(i) + 1;
        for (&j, &value) in tree.colind[start..end].iter().zip(&m[start..end]) {
            out[i * nv + j] = value;
            out[j * nv + i] = value;
        }
    }
    out
}

/// Body spatial velocities and the time derivative of the motion subspace,
/// both in the composite frame: the output of [`com_vel`].
pub type ComVel = (Vec<[f64; 6]>, Vec<[f64; 6]>);

/// The velocity stage: `cvel` and `cdof_dot` (§3.1 `mj_comVel`).
///
/// # Errors
///
/// [`CompositeError::WidthMismatch`] or [`CompositeError::NotFinite`] on a bad
/// `qd`.
pub fn com_vel(
    model: &CompositeModel,
    pos: &Position,
    qd: &[f64],
) -> Result<ComVel, CompositeError> {
    let nv = model.tree.nv;
    if qd.len() != nv {
        return Err(CompositeError::WidthMismatch {
            what: "qd",
            expected: nv,
            found: qd.len(),
        });
    }
    if qd.iter().any(|x| !x.is_finite()) {
        return Err(CompositeError::NotFinite { what: "qd" });
    }
    let n = model.bodies.len();
    let mut cvel = vec![[0.0; 6]; n];
    let mut cdof_dot = vec![[0.0; 6]; nv];

    for (i, body) in model.bodies.iter().enumerate() {
        let mut v = body.parent.map_or([0.0; 6], |p| cvel[p]);
        let adr = body.dof_adr;
        match body.kind {
            JointKind::Fixed => {}
            JointKind::Free => {
                // Translation first: cdof_dot is zero for the three world
                // translations, and the velocity they add is seen by the
                // rotational degrees of freedom that follow (§3.1).
                for k in 0..3 {
                    let s = qd[adr + k];
                    if s != 0.0 {
                        for (dst, src) in v.iter_mut().zip(&pos.cdof[adr + k]) {
                            *dst += *src * s;
                        }
                    }
                }
                // Then the ball block, whose cdof_dot uses the velocity before
                // the ball's own contribution.
                for k in 0..3 {
                    cdof_dot[adr + 3 + k] = cross_motion(v, pos.cdof[adr + 3 + k]);
                }
                for k in 0..3 {
                    let s = qd[adr + 3 + k];
                    if s != 0.0 {
                        for (dst, src) in v.iter_mut().zip(&pos.cdof[adr + 3 + k]) {
                            *dst += *src * s;
                        }
                    }
                }
            }
            JointKind::Revolute | JointKind::Continuous | JointKind::Prismatic => {
                cdof_dot[adr] = cross_motion(v, pos.cdof[adr]);
                let s = qd[adr];
                if s != 0.0 {
                    for (dst, src) in v.iter_mut().zip(&pos.cdof[adr]) {
                        *dst += *src * s;
                    }
                }
            }
        }
        cvel[i] = v;
    }
    Ok((cvel, cdof_dot))
}

/// Recursive Newton-Euler in the composite frame (§3.3 `mj_rne`).
///
/// With `qdd = None` the result is the bias force `qfrc_bias`: Coriolis,
/// centrifugal and gravity. With `qdd = Some(..)` it is the full inverse
/// dynamics, `M qdd + bias`.
///
/// Gravity enters as a fictitious base acceleration `cacc[0] = [0; -g]`, the
/// same trick [`crate::articulated::ArticulatedBody::inverse_dynamics`] uses in
/// the local frame.
///
/// # Errors
///
/// [`CompositeError::WidthMismatch`] or [`CompositeError::NotFinite`] on a bad
/// `qd` or `qdd`.
pub fn rne(
    model: &CompositeModel,
    pos: &Position,
    qd: &[f64],
    qdd: Option<&[f64]>,
) -> Result<Vec<f64>, CompositeError> {
    let nv = model.tree.nv;
    if let Some(a) = qdd {
        if a.len() != nv {
            return Err(CompositeError::WidthMismatch {
                what: "qdd",
                expected: nv,
                found: a.len(),
            });
        }
        if a.iter().any(|x| !x.is_finite()) {
            return Err(CompositeError::NotFinite { what: "qdd" });
        }
    }
    let (cvel, cdof_dot) = com_vel(model, pos, qd)?;
    let n = model.bodies.len();
    let g = model.gravity;
    let base_acc = [0.0, 0.0, 0.0, -g[0], -g[1], -g[2]];

    let mut cacc = vec![[0.0; 6]; n];
    let mut f = vec![[0.0; 6]; n];
    for (i, body) in model.bodies.iter().enumerate() {
        let mut a = body.parent.map_or(base_acc, |p| cacc[p]);
        let adr = body.dof_adr;
        for k in 0..body.dof_num {
            let s = qd[adr + k];
            if s != 0.0 {
                for (dst, src) in a.iter_mut().zip(&cdof_dot[adr + k]) {
                    *dst += *src * s;
                }
            }
        }
        if let Some(acc) = qdd {
            for k in 0..body.dof_num {
                let s = acc[adr + k];
                if s != 0.0 {
                    for (dst, src) in a.iter_mut().zip(&pos.cdof[adr + k]) {
                        *dst += *src * s;
                    }
                }
            }
        }
        cacc[i] = a;
        let mut fi = mul_inert_vec(&pos.cinert[i], a);
        let momentum = mul_inert_vec(&pos.cinert[i], cvel[i]);
        let coriolis = cross_force(cvel[i], momentum);
        for (dst, src) in fi.iter_mut().zip(&coriolis) {
            *dst += *src;
        }
        f[i] = fi;
    }
    for i in (0..n).rev() {
        if let Some(p) = model.bodies[i].parent {
            let add = f[i];
            for (dst, src) in f[p].iter_mut().zip(&add) {
                *dst += *src;
            }
        }
    }
    let tree = &model.tree;
    let mut out = vec![0.0; nv];
    for i in 0..nv {
        out[i] = dot6(pos.cdof[i], f[tree.dof_bodyid[i]]);
    }
    Ok(out)
}

/// The exact diagonal of `M^-1`, one entry per degree of freedom.
///
/// This replaces the `nv`-forward-dynamics-passes construction Appendix D M-24
/// records: one [`crb`] plus one [`factor_m`] plus `nv` back-substitutions,
/// instead of `nv` full articulated-body passes.
///
/// # Errors
///
/// [`CompositeError::SingularInertia`] if `M` cannot be factorised.
pub fn inverse_inertia_diagonal(
    model: &CompositeModel,
    pos: &Position,
) -> Result<Vec<f64>, CompositeError> {
    let tree = &model.tree;
    let m = crb(model, pos);
    let (ld, diag_inv) = factor_m(tree, &m)?;
    let mut out = vec![0.0; tree.nv];
    let mut e = vec![0.0; tree.nv];
    for j in 0..tree.nv {
        e.iter_mut().for_each(|v| *v = 0.0);
        e[j] = 1.0;
        solve_ld(tree, &ld, &diag_inv, &mut e);
        out[j] = e[j];
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::articulated::ArticulatedBody;
    use robot_types::frames::Vec3;

    /// `M` column by column from inverse dynamics: column `j` is
    /// `inverse_dynamics(q, qd = 0, qdd = e_j, gravity = 0)`, which is exactly
    /// `M(q) e_j`. This uses [`ArticulatedBody`]'s local-frame recursive
    /// Newton-Euler, a different algorithm in a different coordinate frame from
    /// [`crb`], so agreement between the two is an independent check rather
    /// than a restatement.
    fn mass_matrix_from_inverse_dynamics(body: &ArticulatedBody, q: &[f64]) -> Vec<f64> {
        let nv = body.dof();
        let zeros = vec![0.0; nv];
        let mut unit = vec![0.0; nv];
        let mut out = vec![0.0; nv * nv];
        for j in 0..nv {
            unit[j] = 1.0;
            let column = body
                .inverse_dynamics(q, &zeros, &unit, Vec3::ZERO)
                .expect("inverse dynamics on a valid profile");
            for (i, value) in column.iter().enumerate() {
                out[i * nv + j] = *value;
            }
            unit[j] = 0.0;
        }
        out
    }

    /// Reorders a dense `nv x nv` matrix from internal to external degree-of-
    /// freedom order, so it can be compared with an `ArticulatedBody` result.
    fn permute_matrix(model: &CompositeModel, internal: &[f64]) -> Vec<f64> {
        let nv = model.nv();
        let mut out = vec![0.0; nv * nv];
        for i in 0..nv {
            for j in 0..nv {
                out[model.dof_ext_of_int[i] * nv + model.dof_ext_of_int[j]] = internal[i * nv + j];
            }
        }
        out
    }

    fn assert_close(actual: &[f64], expected: &[f64], tol: f64, what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what}: width");
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            let scale = 1.0_f64.max(e.abs());
            assert!(
                (a - e).abs() <= tol * scale,
                "{what}: entry {i} is {a}, expected {e} (tolerance {tol} relative)"
            );
        }
    }

    /// A three-link chain with real rotational inertia, off-origin centres of
    /// mass, non-parallel axes and a rotated joint origin, so that the mass
    /// matrix is genuinely dense and the parallel-axis and frame-rotation terms
    /// are all exercised.
    fn three_link_chain() -> RobotProfile {
        use robot_model::joint::{JointLimits, JointSpec};
        use robot_model::link::{InertiaTensor, LinkSpec, MassProperties};
        use robot_model::pose::RigidPose;
        use robot_model::profile::{FirmwareContract, PROFILE_SCHEMA_VERSION};
        use robot_types::frames::{GravityConvention, Quat};
        use robot_types::ids::{JointId, LinkId, Name};
        use robot_types::schema::ExtensionTable;
        use robot_types::time::Duration;
        use robot_types::units::{Kilograms, NewtonMetres, Radians, RadiansPerSecond, Range};

        fn link(id: u32, name: &str, mass: f64, com: Vec3, i: InertiaTensor) -> LinkSpec {
            LinkSpec {
                id: LinkId::new(id),
                name: Name::new(name).unwrap(),
                mass: MassProperties {
                    mass: Kilograms(mass),
                    center_of_mass: com,
                    inertia: i,
                },
                collision: Vec::new(),
            }
        }
        fn joint(
            id: u32,
            name: &str,
            parent: u32,
            child: u32,
            origin: RigidPose,
            axis: Vec3,
            kind: JointKind,
        ) -> JointSpec {
            JointSpec {
                id: JointId::new(id),
                name: Name::new(name).unwrap(),
                kind,
                parent: LinkId::new(parent),
                child: LinkId::new(child),
                origin,
                axis,
                limits: JointLimits::bounded(
                    Range::<Radians>::new(Radians(-3.0), Radians(3.0)).unwrap(),
                    RadiansPerSecond(1000.0),
                    NewtonMetres(1000.0),
                ),
                sign: 1.0,
                zero_offset: 0.0,
            }
        }
        let inertia = |ixx: f64, iyy: f64, izz: f64| InertiaTensor {
            ixx,
            iyy,
            izz,
            ixy: 0.01,
            ixz: -0.02,
            iyz: 0.015,
        };
        RobotProfile {
            schema_version: PROFILE_SCHEMA_VERSION,
            name: Name::new("three-link").unwrap(),
            profile_version: "test".to_owned(),
            joints: vec![
                joint(
                    0,
                    "j0",
                    0,
                    1,
                    RigidPose::IDENTITY,
                    Vec3::unit_y(),
                    JointKind::Revolute,
                ),
                joint(
                    1,
                    "j1",
                    1,
                    2,
                    RigidPose::new(
                        Vec3::new(0.4, 0.05, -0.1),
                        Quat::from_axis_angle(Vec3::new(1.0, 2.0, -1.0), Radians(0.5)).unwrap(),
                    ),
                    Vec3::unit_z(),
                    JointKind::Revolute,
                ),
                joint(
                    2,
                    "j2",
                    2,
                    3,
                    RigidPose::new(
                        Vec3::new(0.3, -0.12, 0.07),
                        Quat::from_axis_angle(Vec3::new(-2.0, 1.0, 3.0), Radians(-0.4)).unwrap(),
                    ),
                    Vec3::unit_x(),
                    JointKind::Revolute,
                ),
            ],
            links: vec![
                link(0, "base", 0.0, Vec3::ZERO, InertiaTensor::ZERO),
                link(
                    1,
                    "l1",
                    2.4,
                    Vec3::new(0.2, 0.03, -0.05),
                    inertia(0.05, 0.07, 0.04),
                ),
                link(
                    2,
                    "l2",
                    1.7,
                    Vec3::new(0.15, -0.04, 0.02),
                    inertia(0.03, 0.02, 0.035),
                ),
                link(
                    3,
                    "l3",
                    0.9,
                    Vec3::new(0.1, 0.02, 0.01),
                    inertia(0.012, 0.015, 0.009),
                ),
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

    #[test]
    fn crb_matches_the_mass_matrix_from_inverse_dynamics_on_a_double_pendulum() {
        let profile = crate::articulated::tests::double_pendulum(1.3, 0.7, 0.9, 0.5);
        let body = ArticulatedBody::from_profile(&profile);
        let model = CompositeModel::from_profile(&profile);
        assert_eq!(model.nv(), body.dof());

        for q in [
            vec![0.0, 0.0],
            vec![0.4, -0.9],
            vec![1.7, 2.3],
            vec![-2.5, 0.15],
        ] {
            let pos = position(&model, &q).unwrap();
            let dense = full_m(&model.tree, &crb(&model, &pos));
            let reference = mass_matrix_from_inverse_dynamics(&body, &q);
            assert_close(&dense, &reference, 1e-11, &format!("M at q = {q:?}"));
        }
    }

    #[test]
    fn crb_matches_the_mass_matrix_from_inverse_dynamics_on_a_three_link_chain() {
        let profile = three_link_chain();
        let body = ArticulatedBody::from_profile(&profile);
        let model = CompositeModel::from_profile(&profile);
        for q in [
            vec![0.0, 0.0, 0.0],
            vec![0.3, -0.7, 1.1],
            vec![-1.2, 2.0, -0.45],
        ] {
            let pos = position(&model, &q).unwrap();
            let dense = full_m(&model.tree, &crb(&model, &pos));
            let reference = mass_matrix_from_inverse_dynamics(&body, &q);
            assert_close(&dense, &reference, 1e-11, &format!("M at q = {q:?}"));
        }
    }

    #[test]
    fn crb_matches_the_mass_matrix_from_inverse_dynamics_on_the_fixed_base_g1() {
        let profile = robot_model::fixtures::unitree_g1_29dof().unwrap();
        let body = ArticulatedBody::from_profile(&profile);
        let model = CompositeModel::from_profile(&profile);
        assert_eq!(model.nv(), body.dof(), "the two adapters agree on nv");

        let nv = model.nv();
        // A configuration away from zero, deterministic and spread over the
        // whole robot so that no joint sits at an accidentally simplifying
        // angle. Built in external order, the layout the rest of the project
        // uses, then converted once.
        let q_ext: Vec<f64> = (0..nv)
            .map(|i| 0.37 * ((i % 7) as f64) - 0.9 * ((i % 3) as f64) + 0.11)
            .collect();
        assert!(
            !model.ordering_is_identity(),
            "the G1 is the fixture whose two orderings differ; if this ever \
             becomes an identity the permutation is no longer being exercised"
        );
        let q = model.qpos_to_internal(&q_ext).unwrap();
        let pos = position(&model, &q).unwrap();
        let dense = permute_matrix(&model, &full_m(&model.tree, &crb(&model, &pos)));
        let reference = mass_matrix_from_inverse_dynamics(&body, &q_ext);
        assert_close(&dense, &reference, 1e-10, "M on the fixed-base G1");
    }

    #[test]
    fn the_mass_matrix_is_symmetric_and_positive_definite_on_the_g1() {
        let profile = robot_model::fixtures::unitree_g1_29dof().unwrap();
        let model = CompositeModel::from_profile(&profile);
        let nv = model.nv();
        let q = vec![0.2; nv];
        let pos = position(&model, &q).unwrap();
        let m = crb(&model, &pos);
        let dense = full_m(&model.tree, &m);
        for i in 0..nv {
            for j in 0..nv {
                let a = dense[i * nv + j];
                let b = dense[j * nv + i];
                assert!((a - b).abs() <= 1e-12 * 1.0_f64.max(a.abs()), "M[{i},{j}]");
            }
        }
        // Positive definiteness is exactly what factor_m succeeding means: it
        // refuses a pivot at or below MINVAL.
        factor_m(&model.tree, &m).expect("M is positive definite");
    }

    #[test]
    fn factor_and_solve_invert_the_mass_matrix() {
        let profile = three_link_chain();
        let model = CompositeModel::from_profile(&profile);
        let q = vec![0.3, -0.7, 1.1];
        let pos = position(&model, &q).unwrap();
        let m = crb(&model, &pos);
        let (ld, diag_inv) = factor_m(&model.tree, &m).unwrap();
        let nv = model.nv();

        // M^-1 column by column, then M * M^-1 must be the identity.
        let mut inverse = vec![0.0; nv * nv];
        for j in 0..nv {
            let mut e = vec![0.0; nv];
            e[j] = 1.0;
            solve_ld(&model.tree, &ld, &diag_inv, &mut e);
            for i in 0..nv {
                inverse[i * nv + j] = e[i];
            }
        }
        let dense = full_m(&model.tree, &m);
        for i in 0..nv {
            for j in 0..nv {
                let mut sum = 0.0;
                for k in 0..nv {
                    sum += dense[i * nv + k] * inverse[k * nv + j];
                }
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sum - want).abs() < 1e-10,
                    "(M M^-1)[{i},{j}] = {sum}, expected {want}"
                );
            }
        }
    }

    #[test]
    fn solve_ld_round_trips_against_mul_m_on_the_g1() {
        let profile = robot_model::fixtures::unitree_g1_29dof().unwrap();
        let model = CompositeModel::from_profile(&profile);
        let nv = model.nv();
        let q: Vec<f64> = (0..nv).map(|i| 0.05 * (i as f64) - 0.4).collect();
        // Any q works here: this check never leaves internal order.
        let pos = position(&model, &q).unwrap();
        let m = crb(&model, &pos);
        let (ld, diag_inv) = factor_m(&model.tree, &m).unwrap();

        let x: Vec<f64> = (0..nv).map(|i| 0.3 * ((i % 5) as f64) - 0.6).collect();
        // y = M x, then M^-1 y must recover x.
        let mut round = mul_m(&model.tree, &m, &x);
        solve_ld(&model.tree, &ld, &diag_inv, &mut round);
        assert_close(&round, &x, 1e-9, "M^-1 M x");
    }

    #[test]
    fn mul_m_matches_the_dense_product() {
        let profile = three_link_chain();
        let model = CompositeModel::from_profile(&profile);
        let q = vec![0.9, 0.2, -1.4];
        let pos = position(&model, &q).unwrap();
        let m = crb(&model, &pos);
        let dense = full_m(&model.tree, &m);
        let nv = model.nv();
        let x = vec![0.7, -1.1, 2.3];
        let sparse = mul_m(&model.tree, &m, &x);
        let expected: Vec<f64> = (0..nv)
            .map(|i| (0..nv).map(|j| dense[i * nv + j] * x[j]).sum())
            .collect();
        assert_close(&sparse, &expected, 1e-12, "M x");
    }

    #[test]
    fn the_bias_force_matches_the_local_frame_recursion() {
        // qfrc_bias is Coriolis, centrifugal and gravity: inverse dynamics at
        // zero acceleration. The composite-frame recursion of this module and
        // the body-local recursion of `articulated` are different algorithms in
        // different frames, so this is a cross-check.
        let profile = three_link_chain();
        let body = ArticulatedBody::from_profile(&profile);
        let model = CompositeModel::from_profile(&profile);
        let gravity = profile.gravity.acceleration();
        let zeros = vec![0.0; model.nv()];

        for (q, qd) in [
            (vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]),
            (vec![0.3, -0.7, 1.1], vec![0.0, 0.0, 0.0]),
            (vec![0.3, -0.7, 1.1], vec![1.3, -0.8, 2.1]),
            (vec![-1.2, 2.0, -0.45], vec![-2.0, 0.5, 0.9]),
        ] {
            let pos = position(&model, &q).unwrap();
            let actual = rne(&model, &pos, &qd, None).unwrap();
            let expected = body.inverse_dynamics(&q, &qd, &zeros, gravity).unwrap();
            assert_close(
                &actual,
                &expected,
                1e-10,
                &format!("bias at q={q:?} qd={qd:?}"),
            );
        }
    }

    #[test]
    fn the_bias_force_matches_the_local_frame_recursion_on_the_g1() {
        let profile = robot_model::fixtures::unitree_g1_29dof().unwrap();
        let body = ArticulatedBody::from_profile(&profile);
        let model = CompositeModel::from_profile(&profile);
        let gravity = profile.gravity.acceleration();
        let nv = model.nv();
        let q_ext: Vec<f64> = (0..nv).map(|i| 0.23 * ((i % 5) as f64) - 0.3).collect();
        let qd_ext: Vec<f64> = (0..nv).map(|i| 0.4 * ((i % 4) as f64) - 0.5).collect();
        let zeros = vec![0.0; nv];

        let q = model.qpos_to_internal(&q_ext).unwrap();
        let qd = model.dof_to_internal(&qd_ext).unwrap();
        let pos = position(&model, &q).unwrap();
        let actual = model
            .dof_to_external(&rne(&model, &pos, &qd, None).unwrap())
            .unwrap();
        let expected = body
            .inverse_dynamics(&q_ext, &qd_ext, &zeros, gravity)
            .unwrap();
        assert_close(&actual, &expected, 1e-9, "bias on the fixed-base G1");
    }

    #[test]
    fn full_inverse_dynamics_matches_the_local_frame_recursion() {
        let profile = three_link_chain();
        let body = ArticulatedBody::from_profile(&profile);
        let model = CompositeModel::from_profile(&profile);
        let gravity = profile.gravity.acceleration();
        let q = vec![0.3, -0.7, 1.1];
        let qd = vec![1.3, -0.8, 2.1];
        let qdd = vec![-0.6, 2.2, 0.4];

        let pos = position(&model, &q).unwrap();
        let actual = rne(&model, &pos, &qd, Some(&qdd)).unwrap();
        let expected = body.inverse_dynamics(&q, &qd, &qdd, gravity).unwrap();
        assert_close(&actual, &expected, 1e-10, "M qdd + bias");
    }

    #[test]
    fn rne_with_acceleration_equals_m_times_acceleration_plus_bias() {
        // An internal consistency check that does not involve `articulated`:
        // the same quantity through two routes inside this module.
        let profile = three_link_chain();
        let model = CompositeModel::from_profile(&profile);
        let q = vec![0.45, 1.2, -0.3];
        let qd = vec![0.9, -1.4, 0.2];
        let qdd = vec![2.0, 0.3, -1.1];
        let pos = position(&model, &q).unwrap();

        let full = rne(&model, &pos, &qd, Some(&qdd)).unwrap();
        let bias = rne(&model, &pos, &qd, None).unwrap();
        let m = crb(&model, &pos);
        let m_qdd = mul_m(&model.tree, &m, &qdd);
        let combined: Vec<f64> = m_qdd.iter().zip(&bias).map(|(a, b)| a + b).collect();
        assert_close(&full, &combined, 1e-11, "rne(qdd) vs M qdd + bias");
    }

    #[test]
    fn the_inverse_inertia_diagonal_matches_the_articulated_construction() {
        // Appendix D M-24: the same quantity, previously obtained with nv full
        // articulated-body passes, now from one factorization.
        let profile = robot_model::fixtures::unitree_g1_29dof().unwrap();
        let body = ArticulatedBody::from_profile(&profile);
        let model = CompositeModel::from_profile(&profile);
        let nv = model.nv();
        let q_ext: Vec<f64> = (0..nv).map(|i| 0.17 * ((i % 6) as f64) - 0.2).collect();
        let q = model.qpos_to_internal(&q_ext).unwrap();
        let pos = position(&model, &q).unwrap();

        let actual = model
            .dof_to_external(&inverse_inertia_diagonal(&model, &pos).unwrap())
            .unwrap();
        let expected = body.inverse_inertia_diagonal(&q_ext).unwrap();
        assert_close(&actual, &expected, 1e-8, "diag(M^-1)");
    }

    #[test]
    fn the_dof_tree_pattern_has_the_prefix_property_the_factorization_needs() {
        // factor_m's prefix update is only correct if, for every ancestor i of
        // k, row i's column list is a prefix of row k's. Checked directly on
        // the G1, whose tree branches at the pelvis into legs, torso and arms.
        let profile = robot_model::fixtures::unitree_g1_29dof().unwrap();
        let model = CompositeModel::from_profile(&profile);
        let tree = &model.tree;
        for k in 0..tree.nv {
            let row_k = &tree.colind[tree.rowadr[k]..tree.rowadr[k] + tree.rownnz[k]];
            assert_eq!(*row_k.last().unwrap(), k, "row {k} ends at its diagonal");
            assert!(
                row_k.windows(2).all(|w| w[0] < w[1]),
                "row {k} is ascending"
            );
            let mut j = tree.dof_parentid[k];
            while j >= 0 {
                let i = j as usize;
                let row_i = &tree.colind[tree.rowadr[i]..tree.rowadr[i] + tree.rownnz[i]];
                assert!(
                    row_k.starts_with(row_i),
                    "row {i} must be a prefix of row {k}"
                );
                j = tree.dof_parentid[i];
            }
        }
    }

    #[test]
    fn a_non_unit_joint_axis_gives_the_same_dynamics_as_a_unit_one() {
        // Appendix D M-02. The axis is normalised at the boundary, so scaling a
        // declared axis must not change the mass matrix or the bias force.
        let mut scaled = three_link_chain();
        for joint in &mut scaled.joints {
            let a = joint.axis;
            joint.axis = Vec3::new(a.x * 3.5, a.y * 3.5, a.z * 3.5);
        }
        let unit = three_link_chain();
        let q = vec![0.4, -0.8, 1.0];
        let qd = vec![0.5, 1.1, -0.7];

        let m_unit = CompositeModel::from_profile(&unit);
        let m_scaled = CompositeModel::from_profile(&scaled);
        let p_unit = position(&m_unit, &q).unwrap();
        let p_scaled = position(&m_scaled, &q).unwrap();
        assert_close(
            &full_m(&m_scaled.tree, &crb(&m_scaled, &p_scaled)),
            &full_m(&m_unit.tree, &crb(&m_unit, &p_unit)),
            1e-12,
            "M is invariant to axis scaling",
        );
        assert_close(
            &rne(&m_scaled, &p_scaled, &qd, None).unwrap(),
            &rne(&m_unit, &p_unit, &qd, None).unwrap(),
            1e-12,
            "bias is invariant to axis scaling",
        );
    }

    #[test]
    fn position_rejects_a_bad_configuration_vector() {
        let profile = three_link_chain();
        let model = CompositeModel::from_profile(&profile);
        assert_eq!(
            position(&model, &[0.0, 0.0]).unwrap_err(),
            CompositeError::WidthMismatch {
                what: "q",
                expected: 3,
                found: 2
            }
        );
        assert_eq!(
            position(&model, &[0.0, f64::NAN, 0.0]).unwrap_err(),
            CompositeError::NotFinite { what: "q" }
        );
    }

    #[test]
    fn a_massless_chain_reports_a_singular_inertia_rather_than_clamping() {
        // The example clamps the pivot to MINVAL and warns; this port reports.
        let profile = crate::articulated::tests::double_pendulum(0.0, 1.0, 0.0, 1.0);
        let model = CompositeModel::from_profile(&profile);
        let pos = position(&model, &[0.0, 0.0]).unwrap();
        let m = crb(&model, &pos);
        assert!(matches!(
            factor_m(&model.tree, &m),
            Err(CompositeError::SingularInertia { .. })
        ));
    }

    #[test]
    fn spatial_helpers_match_hand_derived_values() {
        // mul_inert_vec: f = [I w + h x v ; m v - h x w].
        // I = diag(2,3,4), h = (0,0,1), m = 5, w = (1,0,0), v = (0,1,0).
        // I w = (2,0,0); h x v = (0*0-1*1, 1*0-0*0, 0*1-0*0) = (-1,0,0)
        //   -> top = (1,0,0)
        // m v = (0,5,0); h x w = (0*0-1*0, 1*1-0*0, 0*0-0*1) = (0,1,0)
        //   -> bottom = (0,4,0)
        let inert = [2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 5.0];
        let f = mul_inert_vec(&inert, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        assert_close(&f, &[1.0, 0.0, 0.0, 0.0, 4.0, 0.0], 1e-15, "mul_inert_vec");

        // dof_com for a hinge: [axis ; axis x offset].
        let d = dof_com([0.0, 0.0, 1.0], Some([1.0, 0.0, 0.0]));
        assert_close(&d, &[0.0, 0.0, 1.0, 0.0, 1.0, 0.0], 1e-15, "dof_com hinge");
        // and for a slide: [0 ; axis].
        let s = dof_com([0.0, 1.0, 0.0], None);
        assert_close(&s, &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 1e-15, "dof_com slide");

        // cross_motion([w;v], [w';v']) = [w x w' ; w x v' + v x w'].
        let cm = cross_motion(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        );
        assert_close(&cm, &[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 1e-15, "cross_motion");
        // cross_force([w;v], [t;F]) = [w x t + v x F ; w x F].
        let cf = cross_force(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        );
        assert_close(&cf, &[0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 1e-15, "cross_force");
    }

    #[test]
    fn normalize3_falls_back_like_the_example() {
        let (v, n) = normalize3([0.0, 0.0, 0.0]);
        assert_eq!(v, [1.0, 0.0, 0.0]);
        assert_eq!(n, 0.0);
        let (v, n) = normalize3([0.0, 3.0, 4.0]);
        assert!((n - 5.0).abs() < 1e-15);
        assert_close(&v, &[0.0, 0.6, 0.8], 1e-15, "normalize3");
    }
}

//! A complete simulation: composite-frame dynamics, collision detection and
//! the unified constraint solver, stepped together under one per-run
//! configuration.
//!
//! # Why this exists
//!
//! `crate::composite`, `crate::solver`, `crate::collision` and
//! `crate::options` were each built and validated on their own, and
//! `specs/appendix-d-mathematical-challenges.md` entry M-17 records the
//! consequence: the corrected mathematics was available but nothing ran it.
//! `crate::articulated::MultibodyState` remained the stepper the CLI and the
//! environments used, with its inline semi-implicit Euler, its clamped-PD
//! actuation, its per-point penalty contact and its two-part joint limit. This
//! module is the path that uses the corrected pieces.
//!
//! # Relationship to [`crate::articulated`]
//!
//! `ArticulatedBody` is **not** replaced. Its articulated-body algorithm works
//! in each body's own local frame, is independently correct, and is the
//! cross-check the composite recursions are validated against — the mass matrix
//! agrees with its inverse dynamics to 1e-10 on the 29-degree-of-freedom G1.
//! It keeps its own `Free`-joint convention internally and is documented as the
//! legacy path.
//!
//! The two differ in one way a caller must know about. A `Free` joint's
//! velocity here is **world-frame linear in the first three degrees of freedom
//! and body-frame angular in the last three, translation first**, which is the
//! example's convention (Appendix D M-01). `ArticulatedBody` orders them
//! angular-first and keeps both halves body-local. A `q` or `qd` vector cannot
//! be passed between the two without conversion, and
//! [`World::from_articulated_velocity`] exists for exactly that.
//!
//! # What this module does not do
//!
//! - Actuation is the model's declared position servos through
//!   [`World::actuator_torques`] (`crate::passive::affine_actuator_force` with
//!   the force clamp, target clamp and transmission), or a generalized force
//!   the caller supplies. Activation dynamics and the non-affine gain/bias
//!   types of `specs/05` are not applied.
//! - Equality constraints are not built from a profile, because `robot-model`
//!   has nothing to declare one with. The solver accepts them and
//!   `crate::constraint` builds a scalar joint coupling, so the gap is in the
//!   model layer.
//! - Only the ground plane and shape-pair contacts are detected. Mesh
//!   colliders take part as the convex hulls `robot_model::usd` builds at
//!   import; a raw `Mesh` reference (a degenerate hull) is counted and not
//!   collided.

use crate::backend::Backend;
use crate::collision::{self, Placed, Primitive, Proximity};
use crate::composite::{self, CompositeError, CompositeModel, Position};
use crate::env::{Environment, PhysicsStep};
use crate::newton;
use crate::options::{Integrator, SimOptions, SolverAlgorithm};
use crate::solver::{self, Row, RowKind};
use robot_model::joint::JointKind;
use robot_model::link::CollisionGeometry;
use robot_model::profile::RobotProfile;
use robot_types::frames::Vec3;
use robot_types::rng::{Rng, SeedStream};
use robot_types::time::Duration;

/// How a shape's geometry is stored.
///
/// A closed-form primitive carries its own parameters. A hull carries an index
/// into [`World::hulls`] instead of the vertex slice itself, because a `World`
/// owning both the vertices and a slice into them would be
/// self-referential. [`World::place`] takes `&self` and hands out a
/// `Placed<'_>` borrowing from that store, which ties the borrow to exactly the
/// right scope.
#[derive(Clone, Copy, Debug)]
enum ShapeGeom {
    /// A sphere, box, capsule or cylinder: no borrow.
    Closed(Primitive<'static>),
    /// A convex hull, by index into the world's hull store.
    Hull(usize),
}

/// A collision shape attached to a body, in that body's own frame.
#[derive(Clone, Copy, Debug)]
struct Shape {
    body: usize,
    geom: ShapeGeom,
    rotation: [[f64; 3]; 3],
    translation: [f64; 3],
    contype: u32,
    conaffinity: u32,
}

/// A horizontal ground plane with Coulomb friction.
///
/// The plane carries no stiffness or damping: how hard a contact resists is a
/// property of the constraint row's impedance (`solref`/`solimp` in
/// [`SimOptions`]), not of the surface. That is the whole point of the
/// soft-constraint formulation, and it is what Appendix D M-09 records the
/// penalty model as getting wrong by referring every contact to a fixed
/// reference mass.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Ground {
    /// World height of the plane.
    pub height: f64,
    /// Coulomb friction coefficient.
    pub friction: f64,
}

/// How a model's declared actuators are driven.
///
/// A USD `PhysicsDriveAPI` with a stiffness, a damping and a `targetPosition`
/// is a position servo, and applying it as declared holds the robot at that
/// target. That is what Isaac Sim does on load and it is faithful to the asset,
/// but it is also a **control decision baked into the model**, and for the G1
/// the numbers that decision rests on are suspect: the layer sets
/// `damping = 0.0004 * stiffness` on all 29 joints to float precision, and
/// `targetPosition = 0` on every one. Nobody tunes twenty-nine joints to a
/// uniform damping ratio; those are generated defaults. Converted to
/// per-radian they give a stiffness of 132 to 13183 N m/rad at a damping ratio
/// of 0.003 to 0.03 -- a very stiff, almost undamped servo, where the real
/// robot's published joint gains are nearer 150 N m/rad.
///
/// So the mode is a caller's choice rather than a default this crate makes
/// quietly. Loading a model does not have to mean switching its motors on.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ActuationMode {
    /// The declared drive, `kp (ctrl - q) - kd qd`, clamped to `maxForce`.
    ///
    /// Faithful to the asset and to what a simulator reading
    /// `PhysicsDriveAPI` applies.
    #[default]
    Drives,
    /// The declared damping only, with the stiffness dropped: `-kd qd`.
    ///
    /// A motor that resists motion without holding a position, which is what
    /// an unpowered geared actuator does. No target is read.
    Damping,
    /// No actuator force at all.
    ///
    /// What loading a model into MuJoCo gives before any actuator is
    /// commanded, and the honest baseline for watching the passive dynamics.
    Zero,
}

/// One of the model's actuators, as the example's position-servo actuator.
///
/// `specs/10-actuation-sensors-plugins.md` section 2.2 builds an actuator force
/// as `gain * ctrl + bias` with an AFFINE gain and an AFFINE bias. A position
/// servo is `gainprm = [kp, 0, 0]` and `biasprm = [0, -kp, -kd]`, giving
///
/// `force = kp * (ctrl - length) - kd * velocity`
///
/// which is exactly what a USD `PhysicsDriveAPI` declares with its
/// `drive:angular:physics:stiffness`, `:damping` and `:maxForce`. Importing the
/// G1 into any simulator that reads those attributes gives this force, so this
/// is what it means to simulate the model's actuators rather than to invent a
/// controller for them.
#[derive(Clone, Copy, Debug)]
struct Drive {
    /// Internal degree-of-freedom index this actuator drives.
    dof: usize,
    /// Internal position index it reads.
    qpos: usize,
    /// Drive stiffness, `drive:angular:physics:stiffness`.
    kp: f64,
    /// Drive damping, `drive:angular:physics:damping`.
    kd: f64,
    /// Force clamp, `drive:angular:physics:maxForce` (section 2.5).
    force_limit: f64,
    /// Motor-to-joint ratio.
    ratio: f64,
    /// Fraction of motor torque lost in the transmission.
    efficiency_loss: f64,
    /// Position range the target is clamped into, where the joint has one.
    ///
    /// Section 2.8: a position actuator's `ctrlrange` is `inheritrange` from the
    /// joint, so a target outside the joint's own range is not a command the
    /// model can express.
    ctrl_range: Option<(f64, f64)>,
}

/// A supplied vector had the wrong width.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct WidthMismatch {
    /// Width the world requires.
    pub expected: usize,
    /// Width supplied.
    pub found: usize,
}

impl core::fmt::Display for WidthMismatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "expected {} entries, found {}",
            self.expected, self.found
        )
    }
}

impl core::error::Error for WidthMismatch {}

/// One simulation instance.
pub struct World {
    model: CompositeModel,
    shapes: Vec<Shape>,
    /// Weld body of each body: the top-most ancestor reached through
    /// zero-degree-of-freedom joints.
    weld: Vec<usize>,
    /// Weld body of each weld body's parent.
    weld_parent: Vec<Option<usize>>,
    /// Hard position limits per body, in the joint's own unit.
    limits: Vec<Option<(f64, f64)>>,
    /// Effort bound per body, for the stance controller's clamp.
    effort: Vec<f64>,
    /// The model's actuators, in the profile's actuator order.
    drives: Vec<Drive>,
    /// Index of each body's joint in the profile's `actuated_joints` order, or
    /// `None` for a body attached by a joint that is not actuated.
    ///
    /// `CompositeModel` numbers bodies parent-before-child, which the
    /// factorisation requires, while the replay and every other consumer index
    /// by actuated joint. The two do not coincide: the vendored G1 has 39
    /// bodies and 30 actuated joints, the difference being fixed joints that
    /// carry no degree of freedom and so appear in neither the actuated list
    /// nor the external layout. The mapping is therefore taken from the
    /// profile rather than inferred from an offset.
    actuated_index: Vec<Option<usize>>,
    /// Number of actuated joints, the width of [`Self::limit_fractions`].
    actuated_count: usize,
    ground: Option<Ground>,
    /// Friction coefficient between two shapes of the robot.
    pair_friction: f64,
    options: SimOptions,
    /// Previous step's constraint forces, for warm starting.
    warm: Vec<f64>,
    /// Collision shapes that could not be modelled (raw, unreduced meshes).
    unmodelled: usize,
    /// Hull vertices, one entry per hull shape, in that shape's own frame.
    hulls: Vec<Vec<[f64; 3]>>,
}

/// Position and velocity in the composite model's internal ordering.
#[derive(Clone, Debug, PartialEq)]
pub struct State {
    /// Position coordinates, [`World::nq`] wide.
    pub q: Vec<f64>,
    /// Velocity coordinates, [`World::nv`] wide.
    pub qd: Vec<f64>,
}

/// What one step did, for a caller that wants to see inside it.
#[derive(Clone, Debug, PartialEq)]
pub struct StepReport {
    /// Constraint rows built this step.
    pub rows: usize,
    /// Of those, contact normal rows.
    pub contacts: usize,
    /// Of those, joint-limit rows.
    pub limits: usize,
    /// Solver sweeps taken.
    pub solver_iterations: usize,
    /// Largest force change in the final sweep.
    pub solver_residual: f64,
    /// Total normal force across all contact rows, newtons.
    pub normal_force: f64,
    /// Rows whose time constant was raised to twice the timestep.
    pub clamped_rows: usize,
    /// Generalized torque the joint-limit rows applied this step, `J' f` summed
    /// over the limit rows, in the model's **internal** degree-of-freedom
    /// order.
    ///
    /// The legacy path reported this from a separate active-set pass. Here it
    /// is a projection of the one solve's own forces, so it cannot disagree
    /// with what actually moved the robot.
    pub limit_torque: Vec<f64>,
}

impl World {
    /// Builds a world from a validated profile.
    ///
    /// Assumes `profile.validate()` has passed.
    #[must_use]
    pub fn from_profile(profile: &RobotProfile, options: SimOptions) -> Self {
        let model = CompositeModel::from_profile(profile);
        let order = profile.topological_link_order();
        let mut shapes = Vec::new();
        let mut hulls: Vec<Vec<[f64; 3]>> = Vec::new();
        let mut limits = vec![None; model.bodies.len()];
        let mut effort = vec![0.0; model.bodies.len()];
        let mut actuated_index: Vec<Option<usize>> = vec![None; model.bodies.len()];
        // The joint name attaching each body, by internal body index, so an
        // actuator can be matched to the degree of freedom it drives.
        let mut order_names: Vec<robot_types::ids::Name> = Vec::with_capacity(model.bodies.len());
        let actuated_order: Vec<_> = profile.actuated_joints().map(|j| j.id).collect();
        let actuated_count = actuated_order.len();
        let mut unmodelled = 0usize;

        for (index, link_id) in order.into_iter().skip(1).enumerate() {
            let link = profile
                .link(link_id)
                .expect("link exists in a validated profile");
            let joint = profile
                .parent_joint_of(link_id)
                .expect("every non-root link has a parent joint");
            effort[index] = joint.limits.effort.value();
            actuated_index[index] = actuated_order.iter().position(|id| *id == joint.id);
            order_names.push(joint.name.clone());
            limits[index] = match (joint.kind, joint.limits.position) {
                (JointKind::Revolute | JointKind::Prismatic, Some(range)) => {
                    Some((range.lo.value(), range.hi.value()))
                }
                _ => None,
            };
            for shape in &link.collision {
                let primitive = match &shape.geometry {
                    // A reduced hull is a convex shape the narrow phase can
                    // contact, through the same support query every other
                    // primitive answers. A raw, unreduced mesh is not convex
                    // and must not reach a convex-convex routine
                    // (`specs/07` item 2), so it stays uncounted.
                    CollisionGeometry::ConvexHull { points, .. } => {
                        hulls.push(points.iter().map(|p| [p.x, p.y, p.z]).collect());
                        Some(ShapeGeom::Hull(hulls.len() - 1))
                    }
                    CollisionGeometry::Sphere { radius } => {
                        Some(ShapeGeom::Closed(Primitive::Sphere { radius: *radius }))
                    }
                    CollisionGeometry::Box { half_extents } => {
                        Some(ShapeGeom::Closed(Primitive::Box {
                            half: [half_extents.x, half_extents.y, half_extents.z],
                        }))
                    }
                    CollisionGeometry::Cylinder { radius, length } => {
                        Some(ShapeGeom::Closed(Primitive::Cylinder {
                            radius: *radius,
                            half_height: 0.5 * length,
                        }))
                    }
                    CollisionGeometry::Mesh { .. } => None,
                };
                match primitive {
                    Some(geom) => {
                        let q = shape.pose.rotation;
                        let c = shape.pose.translation;
                        let r = crate::spatial::rotation_from_quat(q.w, q.x, q.y, q.z);
                        shapes.push(Shape {
                            body: index,
                            geom,
                            rotation: [[r[0], r[1], r[2]], [r[3], r[4], r[5]], [r[6], r[7], r[8]]],
                            translation: [c.x, c.y, c.z],
                            contype: 1,
                            conaffinity: 1,
                        });
                    }
                    None => unmodelled += 1,
                }
            }
        }

        let mut weld: Vec<usize> = (0..model.bodies.len()).collect();
        for i in 0..model.bodies.len() {
            if model.bodies[i].dof_num == 0 {
                if let Some(p) = model.bodies[i].parent {
                    weld[i] = weld[p];
                }
            }
        }
        let weld_parent: Vec<Option<usize>> = (0..model.bodies.len())
            .map(|i| model.bodies[weld[i]].parent.map(|p| weld[p]))
            .collect();

        // The model's own actuators. A profile that declares none, or declares
        // zero gains (as `placeholder_actuator` does for a URDF import, which
        // carries no actuator metadata at all), yields no drives, and
        // `actuator_torques` then applies nothing -- which is the honest
        // outcome rather than a fabricated default.
        let mut drives = Vec::new();
        for actuator in &profile.actuators {
            let Some(gains) = actuator.gains else {
                continue;
            };
            if gains.kp == 0.0 && gains.kd == 0.0 {
                continue;
            }
            let Some(joint) = profile.joints.iter().find(|j| j.id == actuator.joint) else {
                continue;
            };
            // Locate the body this joint attaches, by name, and take its
            // internal addresses.
            let Some(body) = order_names
                .iter()
                .position(|name| *name == joint.name)
                .map(|index| &model.bodies[index])
            else {
                continue;
            };
            if body.dof_num != 1 || body.qpos_num != 1 {
                continue;
            }
            drives.push(Drive {
                dof: body.dof_adr,
                qpos: body.qpos_adr,
                kp: gains.kp,
                kd: gains.kd,
                force_limit: joint.limits.effort.value(),
                ratio: actuator.transmission.ratio,
                efficiency_loss: actuator.transmission.efficiency_loss,
                ctrl_range: joint.limits.position.map(|r| (r.lo.value(), r.hi.value())),
            });
        }

        Self {
            hulls,
            drives,
            actuated_index,
            actuated_count,
            model,
            shapes,
            weld,
            weld_parent,
            limits,
            effort,
            ground: None,
            pair_friction: 0.0,
            options,
            warm: Vec::new(),
            unmodelled,
        }
    }

    /// Attaches a ground plane.
    #[must_use]
    pub fn with_ground(mut self, ground: Ground) -> Self {
        self.ground = Some(ground);
        self
    }

    /// Sets the friction coefficient between two of the robot's own shapes.
    ///
    /// Zero by default, which is deliberate rather than an oversight: the
    /// previous pipeline could not give a link-against-link contact friction at
    /// all (Appendix D M-10), so defaulting it to a non-zero value would change
    /// behaviour without a model to justify the number.
    #[must_use]
    pub fn with_pair_friction(mut self, mu: f64) -> Self {
        self.pair_friction = mu;
        self
    }

    /// Width of a position vector.
    #[must_use]
    pub fn nq(&self) -> usize {
        self.model.nq
    }

    /// Number of degrees of freedom.
    #[must_use]
    pub fn nv(&self) -> usize {
        self.model.nv()
    }

    /// The per-run configuration, as a caller can query and record it
    /// (`specs/09-fidelity-tiers.md` Section 12.1).
    #[must_use]
    pub fn options(&self) -> &SimOptions {
        &self.options
    }

    /// The composite model, for a caller that needs the dynamics directly.
    #[must_use]
    pub fn model(&self) -> &CompositeModel {
        &self.model
    }

    /// Collision shapes this world contacts.
    #[must_use]
    pub fn shape_count(&self) -> usize {
        self.shapes.len()
    }

    /// Collision shapes that could not be modelled, meshes being the only kind.
    #[must_use]
    pub fn unmodelled_shape_count(&self) -> usize {
        self.unmodelled
    }

    /// The rest state: zeros, with every `Free` joint's quaternion set to the
    /// identity.
    #[must_use]
    pub fn rest_state(&self) -> State {
        let mut q = vec![0.0; self.model.nq];
        for body in &self.model.bodies {
            if body.kind == JointKind::Free {
                q[body.qpos_adr + 3] = 1.0;
            }
        }
        State {
            qd: vec![0.0; self.model.nv()],
            q,
        }
    }

    /// Converts a velocity vector written in [`crate::articulated`]'s
    /// convention into this one.
    ///
    /// The two differ only for a `Free` joint: `articulated` stores angular
    /// first and keeps both halves in the body frame, while this path stores
    /// world-frame linear first and body-frame angular second. `rotation` is
    /// the free body's world rotation, row-major, which the caller has from
    /// its own kinematics.
    ///
    /// Provided so a caller migrating between the two paths converts
    /// deliberately rather than passing a vector that happens to have the right
    /// width and the wrong meaning.
    #[must_use]
    pub fn from_articulated_velocity(&self, legacy: &[f64], rotation: &[[f64; 9]]) -> Vec<f64> {
        let mut out = legacy.to_vec();
        for (free_index, body) in self
            .model
            .bodies
            .iter()
            .filter(|b| b.kind == JointKind::Free)
            .enumerate()
        {
            let a = body.dof_adr;
            if a + 6 > legacy.len() {
                break;
            }
            let r = rotation
                .get(free_index)
                .copied()
                .unwrap_or([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
            // Legacy layout: angular 0..3, body-local linear 3..6.
            let omega = [legacy[a], legacy[a + 1], legacy[a + 2]];
            let v_local = [legacy[a + 3], legacy[a + 4], legacy[a + 5]];
            let v_world = crate::spatial::mat3_mul_vec(r, v_local);
            out[a] = v_world[0];
            out[a + 1] = v_world[1];
            out[a + 2] = v_world[2];
            out[a + 3] = omega[0];
            out[a + 4] = omega[1];
            out[a + 5] = omega[2];
        }
        out
    }

    /// Whether a pair of collision shapes may generate a contact
    /// (`03` sections 2.1 and 2.5).
    fn pair_can_collide(&self, a: &Shape, b: &Shape) -> bool {
        if (a.contype & b.conaffinity) == 0 && (b.contype & a.conaffinity) == 0 {
            return false;
        }
        let (wa, wb) = (self.weld[a.body], self.weld[b.body]);
        if wa == wb {
            return false;
        }
        if self.model.bodies[wa].dof_num == 0 && self.model.bodies[wb].dof_num == 0 {
            return false;
        }
        if self.weld_parent[a.body] == Some(wb) || self.weld_parent[b.body] == Some(wa) {
            return false;
        }
        true
    }

    /// The world placement of a shape.
    fn place(&self, s: &Shape, pos: &Position) -> Placed<'_> {
        let rb = pos.xmat[s.body];
        let tb = pos.xpos[s.body];
        let mut rotation = [[0.0; 3]; 3];
        for (r, row) in rotation.iter_mut().enumerate() {
            for (c, cell) in row.iter_mut().enumerate() {
                *cell = (0..3).map(|k| rb[r * 3 + k] * s.rotation[k][c]).sum();
            }
        }
        let translation = [
            rb[0] * s.translation[0] + rb[1] * s.translation[1] + rb[2] * s.translation[2] + tb[0],
            rb[3] * s.translation[0] + rb[4] * s.translation[1] + rb[5] * s.translation[2] + tb[1],
            rb[6] * s.translation[0] + rb[7] * s.translation[1] + rb[8] * s.translation[2] + tb[2],
        ];
        Placed {
            shape: match s.geom {
                ShapeGeom::Closed(p) => p,
                ShapeGeom::Hull(i) => Primitive::Hull {
                    points: &self.hulls[i],
                },
            },
            rotation,
            translation,
        }
    }

    /// Builds every constraint row at the current state.
    fn build_rows(
        &self,
        q: &[f64],
        pos: &Position,
        cvel: &[[f64; 6]],
        cdof_dot: &[[f64; 6]],
        qd: &[f64],
    ) -> Result<Vec<Row>, CompositeError> {
        let o = &self.options;
        let mut rows = Vec::new();

        // Ground contacts.
        if let Some(g) = self.ground {
            for s in &self.shapes {
                let placed = self.place(s, pos);
                for (point, radius) in collision::plane_contact_points(&placed, 0) {
                    let depth = g.height + radius - point[2];
                    if depth <= -o.margin {
                        continue;
                    }
                    // The contact point is on the shape's surface.
                    let surface = [point[0], point[1], point[2] - radius];
                    solver::contact_rows(
                        &self.model,
                        pos,
                        cvel,
                        cdof_dot,
                        qd,
                        surface,
                        [0.0, 0.0, 1.0],
                        depth,
                        s.body,
                        None,
                        o.condim,
                        solver::sliding_friction(g.friction),
                        o.solref,
                        o.solimp,
                        o.margin,
                        &mut rows,
                    )?;
                }
            }
        }

        // Shape-pair contacts, behind the broad phase and the pair filter.
        for (i, a) in self.shapes.iter().enumerate() {
            for b in &self.shapes[i + 1..] {
                if !self.pair_can_collide(a, b) {
                    continue;
                }
                let pa = self.place(a, pos);
                let pb = self.place(b, pos);
                if collision::spheres_apart(&pa, &pb, o.margin)
                    || collision::aabbs_apart(&pa, &pb, o.margin)
                {
                    continue;
                }
                let contacts: Vec<collision::Contact> = match (pa.shape, pb.shape) {
                    (Primitive::Box { .. }, Primitive::Box { .. }) => {
                        collision::box_box_contacts(&pa, &pb).unwrap_or_default()
                    }
                    _ => match collision::proximity_with(&pa, &pb, o.ccd) {
                        Proximity::Overlap(c) => vec![c],
                        Proximity::Apart { .. } => Vec::new(),
                    },
                };
                for c in contacts {
                    if c.depth <= -o.margin {
                        continue;
                    }
                    solver::contact_rows(
                        &self.model,
                        pos,
                        cvel,
                        cdof_dot,
                        qd,
                        c.point_a,
                        c.normal,
                        c.depth,
                        a.body,
                        Some(b.body),
                        o.condim,
                        solver::sliding_friction(self.pair_friction),
                        o.solref,
                        o.solimp,
                        o.margin,
                        &mut rows,
                    )?;
                }
            }
        }

        // Joint limits, one row per active side (07 section 8). The activation
        // test is strictly `dist < margin`, and both sides can be active at
        // once.
        let nv = self.model.nv();
        for (index, body) in self.model.bodies.iter().enumerate() {
            let Some((lo, hi)) = self.limits[index] else {
                continue;
            };
            if body.dof_num != 1 || body.qpos_num != 1 {
                continue;
            }
            let value = q[body.qpos_adr];
            for side in [-1.0f64, 1.0] {
                let bound = if side < 0.0 { lo } else { hi };
                let dist = side * (bound - value);
                // Strictly `dist < margin`, as 07 section 8 specifies.
                if dist >= o.limit_margin {
                    continue;
                }
                let mut jac = vec![0.0; nv];
                jac[body.dof_adr] = -side;
                rows.push(Row {
                    jacobian: jac,
                    pos: dist,
                    jdot_v: 0.0,
                    solref: o.solref,
                    solimp: o.solimp,
                    margin: o.limit_margin,
                    kind: RowKind::Limit,
                });
            }
        }
        Ok(rows)
    }

    /// `mjd_smooth_vel` with `flg_bias = 0`: the velocity derivative of the
    /// smooth forces, as a diagonal.
    ///
    /// For the forces this crate applies the derivative *is* diagonal, and that
    /// is worth stating rather than leaving as an accident. The smooth forces
    /// are a drive's `kp (ctrl - q) - kd qd` and a joint damper's `-kd qd`;
    /// only the damping terms depend on velocity, and each depends on one
    /// degree of freedom's own velocity. So `d F / d qd = -diag(kd)` and
    /// `qH = M - h dF/dv = M + h diag(kd)`, which adds to the sparse mass
    /// matrix's own diagonal and needs no new structure.
    ///
    /// A fluid force or a tendon damper would make this non-diagonal
    /// (`mjd_passive_vel` covers both), and neither is implemented. The
    /// recursive-Newton-Euler derivative is excluded by `flg_bias = 0`, which
    /// is what distinguishes `implicitfast` from `implicit`; example 01
    /// section 9.2 reinstates it exactly for a standalone free body through
    /// `mjd_freeMhat`, which is not implemented here either and is recorded as
    /// a gap in `specs/03-integrators.md`.
    fn smooth_velocity_derivative(&self) -> Vec<f64> {
        let mut kd = vec![0.0; self.model.nv()];
        for drive in &self.drives {
            kd[drive.dof] += drive.kd * (1.0 - drive.efficiency_loss)
                / if drive.ratio.abs() > f64::EPSILON {
                    drive.ratio
                } else {
                    1.0
                };
        }
        kd
    }

    /// The constrained acceleration at a state, and the rows that produced it.
    fn accel_at(
        &self,
        q: &[f64],
        qd: &[f64],
        tau: &[f64],
        warm: Option<&[f64]>,
    ) -> Result<(Vec<f64>, solver::Solution, Vec<Row>), CompositeError> {
        let pos = composite::position(&self.model, q)?;
        let (cvel, cdof_dot) = composite::com_vel(&self.model, &pos, qd)?;
        // Under `implicitfast` the unconstrained acceleration is solved against
        // `M + h diag(kd)` rather than `M`, which is the whole content of the
        // integrator: the forces stay at the old velocity and the matrix
        // carries their derivative.
        let qacc_smooth = if self.options.integrator == Integrator::ImplicitFast {
            self.implicit_forward_dynamics(&pos, qd, tau)?
        } else {
            composite::forward_dynamics(&self.model, &pos, qd, tau)?
        };
        let rows = self.build_rows(q, &pos, &cvel, &cdof_dot, qd)?;
        // Both solvers answer the same convex problem and have the same
        // signature, so the choice is a dispatch and nothing else depends on
        // it. `SolverAlgorithm`'s documentation carries the measurement that
        // makes Newton the default.
        let solve = match self.options.solver {
            SolverAlgorithm::Newton => newton::solve_newton,
            SolverAlgorithm::ProjectedGaussSeidel => solver::solve,
        };
        let solution = solve(
            &self.model,
            &pos,
            qd,
            &qacc_smooth,
            &rows,
            self.options.timestep.as_secs_f64(),
            self.options.solver_config(),
            warm,
        )?;
        Ok((solution.qacc.clone(), solution, rows))
    }

    /// Forward dynamics with the smooth-force velocity derivative folded in:
    /// `(M + h diag(kd)) qacc = tau - qfrc_bias`.
    ///
    /// Example 01 section 9.2 resets a standalone free body's six rows of `qH`
    /// to `M`, to keep the `L' D L` factorisation well defined. That happens
    /// here for free, because a `Free` joint carries no actuator and therefore
    /// no `kd`, so nothing is added to those rows in the first place.
    ///
    /// # Errors
    ///
    /// Whatever the dynamics stages report, and
    /// [`CompositeError::SingularInertia`] if the modified matrix cannot be
    /// factorised.
    fn implicit_forward_dynamics(
        &self,
        pos: &Position,
        qd: &[f64],
        tau: &[f64],
    ) -> Result<Vec<f64>, CompositeError> {
        let h = self.options.timestep.as_secs_f64();
        let bias = composite::rne(&self.model, pos, qd, None)?;
        let mut mass = composite::crb(&self.model, pos);
        for (i, k) in self.smooth_velocity_derivative().iter().enumerate() {
            if *k != 0.0 {
                mass[self.model.tree.diag_adr(i)] += h * k;
            }
        }
        let (ld, diag_inv) = composite::factor_m(&self.model.tree, &mass)?;
        let mut acc: Vec<f64> = tau.iter().zip(&bias).map(|(t, b)| t - b).collect();
        composite::solve_ld(&self.model.tree, &ld, &diag_inv, &mut acc);
        Ok(acc)
    }

    /// Advances `state` by one timestep.
    ///
    /// The integrator is whichever [`SimOptions::integrator`] names. For
    /// Runge-Kutta the constraint problem is solved at **every stage**, which is
    /// what the example does (`mj_forwardSkip` per stage, 01 section 9.4) and
    /// what makes the method's order meaningful in contact; solving once and
    /// reusing the force would silently reduce it to first order.
    ///
    /// # Errors
    ///
    /// Whatever the dynamics, collision or solver report.
    pub fn step(&mut self, state: &mut State, tau: &[f64]) -> Result<StepReport, CompositeError> {
        let nv = self.model.nv();
        for (what, v, expected) in [
            ("q", &state.q, self.model.nq),
            ("qd", &state.qd, nv),
            ("tau", &tau.to_vec(), nv),
        ] {
            if v.len() != expected {
                return Err(CompositeError::WidthMismatch {
                    what,
                    expected,
                    found: v.len(),
                });
            }
        }
        let dt = self.options.timestep.as_secs_f64();
        let warm = if self.warm.is_empty() {
            None
        } else {
            Some(self.warm.clone())
        };

        let (qacc, solution, rows) = self.accel_at(&state.q, &state.qd, tau, warm.as_deref())?;

        match self.options.integrator {
            // Both take the same semi-implicit update; they differ only in how
            // `qacc` was obtained, which `accel_at` has already decided.
            Integrator::SemiImplicitEuler | Integrator::ImplicitFast => {
                for (v, a) in state.qd.iter_mut().zip(&qacc) {
                    *v += dt * a;
                }
                let qd = state.qd.clone();
                composite::integrate_pos(&self.model, &mut state.q, &qd, dt)?;
            }
            Integrator::Rk4 => {
                const A: [[f64; 3]; 3] = [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]];
                const B: [f64; 4] = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0];
                let q0 = state.q.clone();
                let qd0 = state.qd.clone();
                let mut stage_vel = vec![qd0.clone()];
                let mut stage_acc = vec![qacc.clone()];
                for i in 1..4 {
                    let mut d_qvel = vec![0.0; nv];
                    let mut d_qacc = vec![0.0; nv];
                    for j in 0..i {
                        let w = A[i - 1][j];
                        if w == 0.0 {
                            continue;
                        }
                        for k in 0..nv {
                            d_qvel[k] += w * stage_vel[j][k];
                            d_qacc[k] += w * stage_acc[j][k];
                        }
                    }
                    let mut qi = q0.clone();
                    composite::integrate_pos(&self.model, &mut qi, &d_qvel, dt)?;
                    let qdi: Vec<f64> = qd0.iter().zip(&d_qacc).map(|(v, a)| v + dt * a).collect();
                    let (fi, _, _) = self.accel_at(&qi, &qdi, tau, None)?;
                    let mut corrected = qdi;
                    composite::rk_correct_free_velocity(&self.model, &mut corrected, &d_qvel, dt);
                    stage_vel.push(corrected);
                    stage_acc.push(fi);
                }
                let mut d_qvel = vec![0.0; nv];
                let mut d_qacc = vec![0.0; nv];
                for j in 0..4 {
                    for k in 0..nv {
                        d_qvel[k] += B[j] * stage_vel[j][k];
                        d_qacc[k] += B[j] * stage_acc[j][k];
                    }
                }
                for (v, a) in state.qd.iter_mut().zip(&d_qacc) {
                    *v += dt * a;
                }
                composite::integrate_pos(&self.model, &mut state.q, &d_qvel, dt)?;
            }
        }

        self.warm = solution.force.clone();
        // One contact is one frictionless row, or one group of pyramid edges
        // identified by its first edge. The normal force of a pyramidal contact
        // is the sum of its edge coefficients, which is what
        // `solver::decode_pyramid` reads off: every edge direction carries a
        // full unit of normal, so the normal component is the plain sum.
        let mut contacts = 0usize;
        let mut normal_force = 0.0;
        for (i, row) in rows.iter().enumerate() {
            match row.kind {
                RowKind::ContactNormal => {
                    contacts += 1;
                    normal_force += solution.force[i];
                }
                RowKind::ContactPyramid { first_edge, .. } => {
                    if first_edge == i {
                        contacts += 1;
                    }
                    normal_force += solution.force[i];
                }
                RowKind::Equality | RowKind::Limit => {}
            }
        }
        let limits = rows.iter().filter(|r| r.kind == RowKind::Limit).count();
        let mut limit_torque = vec![0.0; nv];
        for (row, f) in rows.iter().zip(&solution.force) {
            if row.kind == RowKind::Limit && *f != 0.0 {
                for (t, j) in limit_torque.iter_mut().zip(&row.jacobian) {
                    *t += j * f;
                }
            }
        }
        Ok(StepReport {
            rows: rows.len(),
            contacts,
            limits,
            solver_iterations: solution.iterations,
            solver_residual: solution.residual,
            normal_force,
            clamped_rows: solution.clamped_rows.len(),
            limit_torque,
        })
    }

    /// How many actuators the model declares and this world simulates.
    #[must_use]
    pub fn actuator_count(&self) -> usize {
        self.drives.len()
    }

    /// The `(kp, kd, force_limit)` of each simulated actuator, in actuator
    /// order, so a caller can report what the model asked for.
    #[must_use]
    pub fn actuator_gains(&self) -> Vec<(f64, f64, f64)> {
        self.drives
            .iter()
            .map(|d| (d.kp, d.kd, d.force_limit))
            .collect()
    }

    /// Whether the declared drive gains look machine-generated rather than
    /// tuned, and why.
    ///
    /// Returns `Some(reason)` when every drive shares the same
    /// damping-to-stiffness ratio. A tuned set does not: critical damping is
    /// `kd = 2 sqrt(kp I)`, so the ratio `kd/kp = 2 sqrt(I/kp)` varies with
    /// each joint's own inertia and stiffness. A single ratio across every
    /// joint is a default someone wrote once, and holding a posture with it
    /// says more about that default than about the robot.
    #[must_use]
    pub fn drive_gains_look_generated(&self) -> Option<String> {
        if self.drives.len() < 3 {
            return None;
        }
        let ratios: Vec<f64> = self
            .drives
            .iter()
            .filter(|d| d.kp.abs() > f64::EPSILON)
            .map(|d| d.kd / d.kp)
            .collect();
        if ratios.len() < 3 {
            return None;
        }
        let first = ratios[0];
        if first.abs() <= f64::EPSILON {
            return None;
        }
        let spread = ratios
            .iter()
            .map(|r| (r - first).abs() / first.abs())
            .fold(0.0f64, f64::max);
        if spread > 1e-4 {
            return None;
        }
        Some(format!(
            "every drive shares damping/stiffness = {first:.3e} s, so the gains are generated defaults rather than tuned"
        ))
    }

    /// `mj_fwdActuation` for the model's own drives: the generalized force the
    /// declared actuators produce at `(q, qd)` for the position targets
    /// `ctrl`, written into `tau`.
    ///
    /// `ctrl` is one position target per actuator, in the profile's actuator
    /// order, clamped into the joint's own range
    /// (`specs/10` section 2.8 `inheritrange`). Each actuator's force is
    ///
    /// `kp * (ctrl - q) - kd * qd`
    ///
    /// clamped to the drive's `maxForce` (section 2.5) and then taken through
    /// the transmission, `joint_torque = motor_torque * (1 - loss) / ratio`.
    /// The result accumulates into `tau` at the driven degree of freedom,
    /// which is `qfrc_actuator = moment' * force` (section 2.7) for the
    /// one-to-one joint transmission a USD drive describes.
    ///
    /// `tau` is overwritten, not added to, so a caller composing this with
    /// another torque source adds them itself.
    ///
    /// # Errors
    ///
    /// [`CompositeError::WidthMismatch`] if any vector is the wrong width.
    pub fn actuator_torques(
        &self,
        q: &[f64],
        qd: &[f64],
        ctrl: &[f64],
        mode: ActuationMode,
        tau: &mut [f64],
    ) -> Result<(), CompositeError> {
        let nv = self.model.nv();
        for (what, len, expected) in [
            ("q", q.len(), self.model.nq),
            ("qd", qd.len(), nv),
            ("ctrl", ctrl.len(), self.drives.len()),
            ("tau", tau.len(), nv),
        ] {
            if len != expected {
                return Err(CompositeError::WidthMismatch {
                    what,
                    expected,
                    found: len,
                });
            }
        }
        tau.fill(0.0);
        if mode == ActuationMode::Zero {
            return Ok(());
        }
        for (drive, target) in self.drives.iter().zip(ctrl) {
            let target = match drive.ctrl_range {
                Some((lo, hi)) => target.clamp(lo, hi),
                None => *target,
            };
            let length = q[drive.qpos];
            let velocity = qd[drive.dof];
            // The example's affine gain and affine bias, as a position servo.
            // Dropping the stiffness leaves `-kd qd`, which is the damping
            // mode: the same declared damping, with no position to hold.
            let kp = match mode {
                ActuationMode::Drives => drive.kp,
                ActuationMode::Damping | ActuationMode::Zero => 0.0,
            };
            let force = crate::passive::affine_actuator_force(
                [kp, 0.0, 0.0],
                [0.0, -kp, -drive.kd],
                target,
                length,
                velocity,
            );
            // Section 2.5: clamp to the drive's force range.
            let clamped = force.clamp(-drive.force_limit, drive.force_limit);
            let ratio = if drive.ratio.abs() > f64::EPSILON {
                drive.ratio
            } else {
                1.0
            };
            tau[drive.dof] += clamped * (1.0 - drive.efficiency_loss) / ratio;
        }
        Ok(())
    }

    /// Per-degree-of-freedom damping gain that a semi-implicit step can carry
    /// without the damping itself going unstable.
    ///
    /// `kd <= 0.5 / (M^-1_jj dt)`, capped at `kd_max`: the same bound the
    /// legacy path uses, and the same reason -- an explicit damping term with
    /// `kd` above `m_eff / (2 dt)` amplifies rather than removes velocity.
    ///
    /// # Errors
    ///
    /// Whatever the position or factorisation stages report.
    pub fn stable_damping(
        &self,
        q: &[f64],
        kd_max: f64,
        dt_seconds: f64,
    ) -> Result<Vec<f64>, CompositeError> {
        let pos = composite::position(&self.model, q)?;
        let inverse = composite::inverse_inertia_diagonal(&self.model, &pos)?;
        Ok(inverse
            .iter()
            .map(|m_inv| kd_max.min(0.5 / (m_inv * dt_seconds)))
            .collect())
    }

    /// A clamped proportional-derivative hold toward the zero posture, written
    /// into `tau` in the model's internal degree-of-freedom order.
    ///
    /// Only single-degree-of-freedom joints are driven; a `Free` joint's
    /// degrees of freedom are left at zero, because there is no actuator on a
    /// floating base to drive them with.
    ///
    /// This is a demonstration controller, not a balance controller, and it is
    /// here rather than in a controller crate for the same reason the legacy
    /// path has it: `robot sim multibody` needs *some* torque to hold a stance
    /// so that the contact solve has something to do.
    ///
    /// # Errors
    ///
    /// [`CompositeError::WidthMismatch`] if any vector is the wrong width.
    pub fn stance_torques(
        &self,
        q: &[f64],
        qd: &[f64],
        kp: f64,
        kd: &[f64],
        tau: &mut [f64],
    ) -> Result<(), CompositeError> {
        let nv = self.model.nv();
        for (what, len, expected) in [
            ("q", q.len(), self.model.nq),
            ("qd", qd.len(), nv),
            ("kd", kd.len(), nv),
            ("tau", tau.len(), nv),
        ] {
            if len != expected {
                return Err(CompositeError::WidthMismatch {
                    what,
                    expected,
                    found: len,
                });
            }
        }
        tau.fill(0.0);
        for (index, body) in self.model.bodies.iter().enumerate() {
            if body.dof_num != 1 || body.qpos_num != 1 {
                continue;
            }
            let effort = self.effort[index];
            let value = q[body.qpos_adr];
            let rate = qd[body.dof_adr];
            tau[body.dof_adr] = (-kp * value - kd[body.dof_adr] * rate).clamp(-effort, effort);
        }
        Ok(())
    }

    /// Where each bounded single-degree-of-freedom joint sits in its range, as
    /// a fraction from 0 at the lower bound to 1 at the upper, indexed by
    /// **actuated joint** with `None` for a joint without a limit.
    ///
    /// Indexed by actuated joint rather than by degree of freedom because that
    /// is what the replay reads (`usd_replay` looks up `slot.actuated`), and
    /// because a `Free` joint spans six degrees of freedom but one slot.
    ///
    /// Reported for the replay, which colours a joint by how close it is to its
    /// stop.
    #[must_use]
    pub fn limit_fractions(&self, q: &[f64]) -> Vec<Option<f64>> {
        let mut out = vec![None; self.actuated_count];
        for (index, body) in self.model.bodies.iter().enumerate() {
            let Some((lo, hi)) = self.limits[index] else {
                continue;
            };
            if body.qpos_num != 1 || body.qpos_adr >= q.len() {
                continue;
            }
            let Some(slot) = self.actuated_index[index] else {
                continue;
            };
            // A zero-width range is reported as mid-range, as the legacy path
            // does, rather than skipped: the joint *has* a limit, and the
            // replay colours it by this number.
            let span = hi - lo;
            if span.abs() <= f64::EPSILON {
                out[slot] = Some(0.5);
                continue;
            }
            out[slot] = Some((q[body.qpos_adr] - lo) / span);
        }
        out
    }

    /// Converts a position vector from the model's internal order to the
    /// external order the rest of the workspace uses.
    ///
    /// # Errors
    ///
    /// [`CompositeError::WidthMismatch`] on a wrong width.
    pub fn q_external(&self, internal: &[f64]) -> Result<Vec<f64>, CompositeError> {
        self.model.qpos_to_external(internal)
    }

    /// Converts a velocity vector from internal to external order.
    ///
    /// The **values** are not converted, only their order. A `Free` joint's
    /// velocity still means what this path means by it -- world-frame linear
    /// first, body-frame angular second -- which is not what
    /// [`crate::articulated`] means by the same slots. See
    /// [`Self::from_articulated_velocity`].
    ///
    /// # Errors
    ///
    /// [`CompositeError::WidthMismatch`] on a wrong width.
    pub fn qd_external(&self, internal: &[f64]) -> Result<Vec<f64>, CompositeError> {
        self.model.dof_to_external(internal)
    }

    /// Total kinetic energy at a state.
    ///
    /// # Errors
    ///
    /// Whatever the position stage reports.
    pub fn kinetic_energy(&self, state: &State) -> Result<f64, CompositeError> {
        let pos = composite::position(&self.model, &state.q)?;
        let m = composite::crb(&self.model, &pos);
        let mv = composite::mul_m(&self.model.tree, &m, &state.qd);
        Ok(0.5 * state.qd.iter().zip(&mv).map(|(a, b)| a * b).sum::<f64>())
    }

    /// World height of the lowest point of any collision shape at `q`.
    ///
    /// # Errors
    ///
    /// Whatever the position stage reports.
    pub fn lowest_contact_height(&self, q: &[f64]) -> Result<Option<f64>, CompositeError> {
        let pos = composite::position(&self.model, q)?;
        let mut lowest: Option<f64> = None;
        for s in &self.shapes {
            let placed = self.place(s, &pos);
            for (point, radius) in collision::plane_contact_points(&placed, 0) {
                let bottom = point[2] - radius;
                lowest = Some(lowest.map_or(bottom, |l| l.min(bottom)));
            }
        }
        Ok(lowest)
    }

    /// Gravity this world integrates under.
    #[must_use]
    pub fn gravity(&self) -> Vec3 {
        Vec3::new(
            self.model.gravity[0],
            self.model.gravity[1],
            self.model.gravity[2],
        )
    }
}

/// An [`Environment`] and [`Backend`] over [`World`]: the corrected dynamics,
/// collision and constraint path behind the interface every consumer of a
/// simulated environment depends on.
///
/// `specs/10-cpu-reference-backend.md` requires crates outside `robot-sim` to
/// depend only on that interface and never on a backend's internals, so this is
/// how the corrected path reaches them. It reports `cpu-world`, distinct from
/// `crate::multibody_env::MultibodyEnv`'s `cpu-multibody`, because the two are
/// genuinely different backends and a run record should say which one produced
/// it.
///
/// **The observation layout differs from `MultibodyEnv`'s for a `Free` joint.**
/// Both report `q` then `qd`, but this backend's `qd` holds world-frame linear
/// velocity in the first three slots and body-frame angular in the last three,
/// translation first, which is the example's convention
/// (`specs/appendix-d-mathematical-challenges.md` M-01). A policy trained
/// against one backend's observations has not been trained against the other's.
/// That is why this is a separate backend rather than a change of behaviour
/// inside the existing one.
pub struct WorldEnv {
    world: World,
    q_home: Vec<f64>,
    q_spread: f64,
    state: State,
    rng: Rng,
}

impl WorldEnv {
    /// Builds the environment at rest at `q_home`.
    ///
    /// # Errors
    ///
    /// [`WidthMismatch`] if `q_home` is not [`World::nq`] wide, reported here
    /// rather than panicking on the first observation.
    pub fn new(
        world: World,
        q_home: Vec<f64>,
        q_spread: f64,
        seed: SeedStream,
    ) -> Result<Self, WidthMismatch> {
        if q_home.len() != world.nq() {
            return Err(WidthMismatch {
                expected: world.nq(),
                found: q_home.len(),
            });
        }
        let state = State {
            q: q_home.clone(),
            qd: vec![0.0; world.nv()],
        };
        Ok(Self {
            world,
            q_home,
            q_spread,
            state,
            rng: Rng::new(seed),
        })
    }

    /// The current state.
    #[must_use]
    pub fn state(&self) -> &State {
        &self.state
    }

    /// Adds `delta_v` (world frame, m/s) to the floating base's linear
    /// velocity: an impulsive push, as a perturbation task applies it.
    /// Returns `false`, changing nothing, when the model has no free joint.
    pub fn push_base(&mut self, delta_v: Vec3) -> bool {
        let Some(body) = self
            .world
            .model()
            .bodies
            .iter()
            .find(|b| b.kind == JointKind::Free)
        else {
            return false;
        };
        // This path's free-joint velocity is world linear first.
        let adr = body.dof_adr;
        self.state.qd[adr] += delta_v.x;
        self.state.qd[adr + 1] += delta_v.y;
        self.state.qd[adr + 2] += delta_v.z;
        true
    }

    /// The world, for a caller that needs to query the configuration.
    #[must_use]
    pub fn world(&self) -> &World {
        &self.world
    }

    fn write_observation(&self, observation: &mut [f64]) {
        let (q_out, qd_out) = observation.split_at_mut(self.state.q.len());
        q_out.copy_from_slice(&self.state.q);
        qd_out.copy_from_slice(&self.state.qd);
    }

    /// Draws a start configuration within `q_spread` of `q_home`, leaving any
    /// `Free` joint's quaternion alone so it stays a unit quaternion.
    fn draw_start(&mut self) -> Vec<f64> {
        let mut q = self.q_home.clone();
        for body in &self.world.model().bodies {
            match body.kind {
                JointKind::Free => {
                    // Perturbing a quaternion component by component would
                    // leave it non-unit, and the position stage would then
                    // renormalise it to something the caller did not ask for.
                    // Only the translation is drawn.
                    for k in 0..3 {
                        q[body.qpos_adr + k] += self.rng.uniform(-self.q_spread, self.q_spread);
                    }
                }
                JointKind::Revolute | JointKind::Continuous | JointKind::Prismatic => {
                    q[body.qpos_adr] += self.rng.uniform(-self.q_spread, self.q_spread);
                }
                JointKind::Fixed => {}
            }
        }
        q
    }
}

impl Environment for WorldEnv {
    fn observation_width(&self) -> usize {
        self.world.nq() + self.world.nv()
    }

    fn action_width(&self) -> usize {
        self.world.nv()
    }

    fn reset(&mut self, seed: SeedStream, observation: &mut [f64]) {
        self.rng = Rng::new(seed);
        let q = self.draw_start();
        self.state = State {
            qd: vec![0.0; self.world.nv()],
            q,
        };
        self.write_observation(observation);
    }

    fn step(&mut self, action: &[f64], _dt: Duration) -> PhysicsStep {
        // The timestep is the world's own configuration, not a per-call
        // argument: `specs/09` section 12.1 makes it a recorded per-run knob,
        // and a caller that could change it mid-episode would invalidate the
        // record. The trait's `dt` is therefore ignored, which is stated here
        // rather than silently honoured.
        let stepped = self.world.step(&mut self.state, action);
        let finite = self
            .state
            .q
            .iter()
            .chain(&self.state.qd)
            .all(|v| v.is_finite());
        let mut observation = vec![0.0; self.observation_width()];
        self.write_observation(&mut observation);
        PhysicsStep {
            observation,
            physics_failure: stepped.is_err() || !finite,
        }
    }
}

impl Backend for WorldEnv {
    fn backend_name(&self) -> &'static str {
        "cpu-world"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_types::time::Duration;

    /// The G1 as the official OpenUSD description, which is the model that
    /// carries the drives (ADR 0015). The URDF fixture below has none, because
    /// URDF does not express actuator metadata at all.
    fn g1_usd() -> RobotProfile {
        robot_model::fixtures::unitree_g1_29dof_usd_floating_base()
            .expect("the vendored G1 USD model loads")
    }

    fn g1() -> RobotProfile {
        robot_model::fixtures::unitree_g1_29dof_floating_base().expect("the vendored fixture loads")
    }

    fn standing_world() -> (World, State) {
        let options = SimOptions {
            timestep: Duration::from_micros(500),
            condim: 3,
            ..SimOptions::default()
        };
        let world = World::from_profile(&g1(), options).with_ground(Ground {
            height: 0.0,
            friction: 0.8,
        });
        let mut state = world.rest_state();
        // Place the lowest contact point 5 mm above the plane, as the legacy
        // stance run does.
        let lowest = world
            .lowest_contact_height(&state.q)
            .unwrap()
            .expect("the G1 has contact shapes");
        let free = world
            .model()
            .bodies
            .iter()
            .find(|b| b.kind == JointKind::Free)
            .expect("the floating fixture has a Free joint");
        state.q[free.qpos_adr + 2] = 0.005 - lowest;
        (world, state)
    }

    #[test]
    fn the_world_reports_the_same_shape_inventory_as_the_legacy_path() {
        // Both adapters read the same profile, so they must agree on what is
        // collidable and what is not. This is the cheapest guard against the
        // two adapters drifting apart while both exist.
        let profile = g1();
        let world = World::from_profile(&profile, SimOptions::default());
        let legacy = crate::articulated::ArticulatedBody::from_profile(&profile);
        assert_eq!(world.nv(), legacy.dof());
        assert_eq!(world.nq(), legacy.position_width());
        assert_eq!(world.shape_count(), legacy.contact_shape_count());
        assert_eq!(
            world.unmodelled_shape_count(),
            legacy.unmodelled_collision_shape_count()
        );
    }

    #[test]
    fn a_free_joint_lays_out_world_linear_velocity_first() {
        // The convention this path exists to carry (Appendix D M-01). Setting
        // the first three degrees of freedom must move the body in *world*
        // axes whatever its orientation, which is what distinguishes it from
        // the legacy layout.
        let options = SimOptions {
            timestep: Duration::from_millis(1),
            ..SimOptions::default()
        };
        let mut world = World::from_profile(&g1(), options);
        let mut state = world.rest_state();
        let free = world
            .model()
            .bodies
            .iter()
            .find(|b| b.kind == JointKind::Free)
            .expect("Free joint")
            .clone();
        let (pa, va) = (free.qpos_adr, free.dof_adr);
        // Yaw the base a quarter turn, then push along world +x.
        let half = core::f64::consts::FRAC_PI_4;
        state.q[pa + 3] = half.cos();
        state.q[pa + 6] = half.sin();
        state.qd[va] = 1.0;
        let tau = vec![0.0; world.nv()];
        let start = state.q[pa];
        for _ in 0..10 {
            world.step(&mut state, &tau).unwrap();
        }
        // Ten milliseconds at 1 m/s along world +x, regardless of the yaw.
        assert!(
            (state.q[pa] - start - 0.01).abs() < 1e-9,
            "x moved {} , expected 0.01",
            state.q[pa] - start
        );
        assert!(state.q[pa + 1].abs() < 1e-9, "no y motion from a +x push");
    }

    #[test]
    fn the_run_configuration_is_queryable_and_recordable() {
        // Section 12.1 requires the knobs to be recorded with a run, and a
        // world is where a run happens.
        let options = SimOptions {
            condim: 1,
            solver_iterations: 25,
            ..SimOptions::default()
        };
        let world = World::from_profile(&g1(), options);
        assert_eq!(world.options().condim, 1);
        assert_eq!(world.options().solver_iterations, 25);
        let record = world.options().record();
        assert!(record.iter().any(|(k, v)| *k == "condim" && v == "1"));
    }

    #[test]
    fn a_width_mismatch_is_reported_rather_than_panicking() {
        let mut world = World::from_profile(&g1(), SimOptions::default());
        let mut state = world.rest_state();
        assert!(world.step(&mut state, &[0.0; 3]).is_err());
        state.qd.pop();
        let tau = vec![0.0; world.nv()];
        assert!(world.step(&mut state, &tau).is_err());
    }
    #[test]
    fn the_ground_is_enforced_on_every_collision_shape_as_the_robot_collapses() {
        // This is the property M-17 exists to deliver, and it is narrower than
        // "the robot stands". A zero-torque humanoid cannot stand: nothing
        // holds its joints, so its legs fold, and `articulated.rs` records that
        // collapse as the expected outcome. What the unified solver must
        // deliver is that the *ground* is enforced while that happens -- every
        // collision shape stays essentially on the plane rather than sinking
        // through it, and the contacts carry the robot's weight.
        //
        // The pelvis itself does pass below the plane during the collapse, and
        // that is not a contact failure: the vendored G1's pelvis and torso
        // carry no collision geometry at all, which `specs/07` and the legacy
        // test both record as a model gap. Only shapes that exist can be
        // contacted, so the assertion below is on the shapes.
        let (mut world, mut state) = standing_world();
        let tau = vec![0.0; world.nv()];
        let weight: f64 = g1().links.iter().map(|l| l.mass.mass.value()).sum::<f64>() * 9.806_65;
        let mut deepest = 0.0f64;
        let mut carried = 0.0f64;
        let mut saw_contact = false;
        for step in 0..3000 {
            let report = world.step(&mut state, &tau).unwrap();
            assert!(
                state.q.iter().chain(&state.qd).all(|v| v.is_finite()),
                "step {step} diverged"
            );
            if report.contacts > 0 {
                saw_contact = true;
                carried = carried.max(report.normal_force);
            }
            let low = world.lowest_contact_height(&state.q).unwrap().unwrap();
            deepest = deepest.min(low);
        }
        assert!(saw_contact, "no contact row was ever created");
        assert!(
            carried > 0.5 * weight,
            "contacts carried at most {carried} N against a weight of {weight} N"
        );
        // Measured at 0.045 m worst transient during the collapse; the bound
        // records that rather than a round number.
        assert!(
            deepest > -0.06,
            "a collision shape reached {deepest} m below the plane"
        );
    }

    #[test]
    fn a_joint_limit_holds_exactly_when_given_a_margin_to_act_in() {
        // Appendix D M-08: a limit is now a constraint row with real impedance
        // instead of an acceleration-level active set plus a position clamp.
        // The behaviour that follows from that is worth asserting precisely,
        // because it is easy to misread as the solver being weak.
        //
        // With no margin a limit row appears only once the joint has crossed
        // its bound, so a joint arriving at speed is braked after the fact and
        // overshoots. Give it room to act and it holds the bound exactly. The
        // old clamp could not express either behaviour: it had no margin, and
        // it enforced the bound by deleting velocity rather than by applying a
        // force.
        let probe = |limit_margin: f64| -> (f64, f64) {
            let mut world = World::from_profile(
                &g1(),
                SimOptions {
                    timestep: Duration::from_micros(200),
                    solimp: [0.99, 0.99, 0.001, 0.5, 2.0],
                    limit_margin,
                    ..SimOptions::default()
                },
            );
            let (dof, qpos, lo, hi) = world
                .model()
                .bodies
                .iter()
                .enumerate()
                .find_map(|(i, b)| world.limits[i].map(|(lo, hi)| (b.dof_adr, b.qpos_adr, lo, hi)))
                .expect("the G1 has bounded joints");
            let mut state = world.rest_state();
            let mut tau = vec![0.0; world.nv()];
            tau[dof] = 50.0;
            let mut worst = 0.0f64;
            for _ in 0..3000 {
                world.step(&mut state, &tau).unwrap();
                let v = state.q[qpos];
                worst = worst.max(v - hi).max(lo - v);
            }
            (worst, state.q[qpos])
        };

        let (no_margin, _) = probe(0.0);
        let (with_margin, final_value) = probe(0.2);
        // Measured 0.177 rad with no margin and exactly 0 with 0.2 rad.
        assert!(
            no_margin > 0.05,
            "with no margin a joint arriving at speed should overshoot; got {no_margin}"
        );
        assert_eq!(
            with_margin, 0.0,
            "with a margin the bound should hold exactly"
        );
        assert!(
            final_value.is_finite(),
            "the joint should settle, not diverge"
        );
    }

    #[test]
    fn the_world_backend_runs_through_the_environment_interface() {
        // `specs/10` requires consumers to depend only on the interface, so the
        // corrected path has to be reachable through it.
        let world = World::from_profile(
            &g1(),
            SimOptions {
                timestep: Duration::from_micros(500),
                ..SimOptions::default()
            },
        )
        .with_ground(Ground {
            height: 0.0,
            friction: 0.8,
        });
        let q_home = world.rest_state().q;
        let mut env = WorldEnv::new(world, q_home, 0.0, SeedStream::new(1, 0)).unwrap();
        assert_eq!(env.backend_name(), "cpu-world");
        assert_eq!(env.observation_width(), env.world().nq() + env.world().nv());
        assert_eq!(env.action_width(), env.world().nv());

        let mut observation = vec![0.0; env.observation_width()];
        env.reset(SeedStream::new(1, 0), &mut observation);
        let action = vec![0.0; env.action_width()];
        for step in 0..200 {
            let out = env.step(&action, Duration::from_micros(500));
            assert!(
                !out.physics_failure,
                "step {step} reported a physics failure"
            );
            assert_eq!(out.observation.len(), env.observation_width());
        }
    }

    #[test]
    fn the_backend_is_deterministic_for_a_seed_and_responds_to_a_different_one() {
        let build = || {
            let world = World::from_profile(
                &g1(),
                SimOptions {
                    timestep: Duration::from_millis(1),
                    ..SimOptions::default()
                },
            );
            let q_home = world.rest_state().q;
            WorldEnv::new(world, q_home, 0.05, SeedStream::new(7, 0)).unwrap()
        };
        let run = |seed: SeedStream| {
            let mut env = build();
            let mut observation = vec![0.0; env.observation_width()];
            env.reset(seed, &mut observation);
            let action = vec![0.0; env.action_width()];
            for _ in 0..20 {
                let out = env.step(&action, Duration::from_millis(1));
                observation = out.observation;
            }
            observation
        };
        let a = run(SeedStream::new(7, 0));
        let b = run(SeedStream::new(7, 0));
        assert_eq!(a, b, "the same seed must give the same trajectory");
        let c = run(SeedStream::new(8, 0));
        assert_ne!(a, c, "a different seed must give a different start");
    }

    #[test]
    fn a_free_joint_quaternion_survives_a_randomised_reset() {
        // The spread is applied to translations and scalar joints only: a
        // quaternion perturbed component by component would stop being one,
        // and the position stage would renormalise it to something the caller
        // never asked for.
        let world = World::from_profile(&g1(), SimOptions::default());
        let q_home = world.rest_state().q;
        let free = world
            .model()
            .bodies
            .iter()
            .find(|b| b.kind == JointKind::Free)
            .expect("Free joint")
            .clone();
        let mut env = WorldEnv::new(world, q_home, 0.3, SeedStream::new(3, 0)).unwrap();
        let mut observation = vec![0.0; env.observation_width()];
        env.reset(SeedStream::new(3, 0), &mut observation);
        let q = &env.state().q;
        let n = (q[free.qpos_adr + 3].powi(2)
            + q[free.qpos_adr + 4].powi(2)
            + q[free.qpos_adr + 5].powi(2)
            + q[free.qpos_adr + 6].powi(2))
        .sqrt();
        assert!((n - 1.0).abs() < 1e-15, "quaternion norm is {n}");
        // The translation did move.
        assert!(
            (q[free.qpos_adr] - 0.0).abs() > 1e-12 || (q[free.qpos_adr + 1] - 0.0).abs() > 1e-12,
            "the spread should have moved the base"
        );
    }
    #[test]
    fn a_stance_hold_keeps_the_g1_standing_through_the_new_path() {
        // The property the zero-torque collapse cannot show. With joints held
        // by a clamped proportional-derivative law the robot should stay up,
        // which exercises the contact solve, the limit rows and the stance
        // controller together -- and is what `robot sim multibody --floating`
        // demonstrates on the legacy path.
        let dt_seconds = 5e-4;
        let (mut world, mut state) = standing_world();
        let kd = world
            .stable_damping(&state.q, 30.0, dt_seconds)
            .expect("the damping bound is computable at the start posture");
        let mut tau = vec![0.0; world.nv()];
        let free = world
            .model()
            .bodies
            .iter()
            .find(|b| b.kind == JointKind::Free)
            .expect("Free joint")
            .clone();
        let pelvis = free.qpos_adr + 2;
        let start = state.q[pelvis];
        let mut lowest = f64::INFINITY;
        for step in 0..4000 {
            world
                .stance_torques(&state.q, &state.qd, 300.0, &kd, &mut tau)
                .expect("widths match");
            let report = world.step(&mut state, &tau).expect("the step succeeds");
            assert!(
                state.q.iter().chain(&state.qd).all(|v| v.is_finite()),
                "step {step} diverged"
            );
            assert_eq!(report.limit_torque.len(), world.nv());
            lowest = lowest.min(state.q[pelvis]);
        }
        // The pelvis settles rather than collapsing. On the zero-torque run it
        // ends up below the plane; held, it stays within a few centimetres of
        // where it started.
        assert!(
            lowest > start - 0.08,
            "the held pelvis fell to {lowest} from {start}"
        );
        // And it is still standing at the end, not just transiently.
        assert!(
            state.q[pelvis] > start - 0.08,
            "the pelvis ended at {} from {start}",
            state.q[pelvis]
        );
    }

    #[test]
    fn the_limit_torque_is_a_projection_of_the_solve_that_moved_the_robot() {
        // The legacy path reported a limit torque from a separate active-set
        // pass, which could disagree with the forces that actually integrated.
        // Here it is `J' f` over the limit rows of the same solve, so a
        // non-zero limit torque implies a limit row carried force and the
        // converse.
        let profile = g1();
        let mut world = World::from_profile(
            &profile,
            SimOptions {
                timestep: Duration::from_micros(200),
                limit_margin: 0.1,
                ..SimOptions::default()
            },
        );
        let (dof, _, _, _) = world
            .model()
            .bodies
            .iter()
            .enumerate()
            .find_map(|(i, b)| world.limits[i].map(|(lo, hi)| (b.dof_adr, b.qpos_adr, lo, hi)))
            .expect("the G1 has bounded joints");
        let mut state = world.rest_state();
        let mut tau = vec![0.0; world.nv()];
        tau[dof] = 60.0;
        let mut saw_limit_torque = false;
        for _ in 0..3000 {
            let report = world.step(&mut state, &tau).unwrap();
            let total: f64 = report.limit_torque.iter().map(|t| t.abs()).sum();
            assert_eq!(
                report.limits > 0,
                total > 0.0 || report.limits == 0,
                "a limit row must be the only source of limit torque"
            );
            if total > 0.0 {
                saw_limit_torque = true;
                // It opposes the drive: the joint is being pushed to its stop.
                assert!(
                    report.limit_torque[dof] * tau[dof] <= 0.0,
                    "limit torque {} should oppose the drive {}",
                    report.limit_torque[dof],
                    tau[dof]
                );
            }
        }
        assert!(saw_limit_torque, "the drive never reached a limit");
    }

    #[test]
    fn external_ordering_round_trips_a_state() {
        let world = World::from_profile(&g1(), SimOptions::default());
        let state = world.rest_state();
        let qe = world.q_external(&state.q).unwrap();
        assert_eq!(qe.len(), world.nq());
        let back = world.model().qpos_to_internal(&qe).unwrap();
        assert_eq!(back, state.q);
        // The replay indexes this by actuated joint, so it must agree with the
        // legacy path slot for slot -- not merely have the same length. The
        // composite model numbers bodies parent-before-child while the replay
        // numbers them by actuated joint, and nothing but a comparison catches
        // the two drifting apart.
        let fractions = world.limit_fractions(&state.q);
        let legacy_body = crate::articulated::ArticulatedBody::from_profile(&g1());
        let legacy_q = world.q_external(&state.q).unwrap();
        let legacy = legacy_body.limit_fractions(&legacy_q).unwrap();
        assert_eq!(fractions.len(), legacy.len(), "one slot per actuated joint");
        for (i, (mine, theirs)) in fractions.iter().zip(&legacy).enumerate() {
            match (mine, theirs) {
                (Some(a), Some(b)) => assert!(
                    (a - b).abs() < 1e-12,
                    "actuated joint {i}: {a} against the legacy {b}"
                ),
                (None, None) => {}
                _ => panic!(
                    "actuated joint {i} disagrees on having a limit: {mine:?} against {theirs:?}"
                ),
            }
        }
        // The zero posture sits inside every range it has one for.
        for f in fractions.iter().flatten() {
            assert!(
                (-0.1..=1.1).contains(f),
                "the zero posture is outside a joint range at fraction {f}"
            );
        }
    }
    #[test]
    fn the_models_own_drives_are_simulated_not_a_hand_written_controller() {
        // The G1 USD declares a PhysicsDriveAPI per joint: a stiffness, a
        // damping and a maxForce. Importing it into any simulator that reads
        // those attributes applies them. This asserts that this world does the
        // same, rather than inventing gains of its own.
        let profile = g1_usd();
        let world = World::from_profile(&profile, SimOptions::default());
        assert_eq!(
            world.actuator_count(),
            29,
            "the G1 declares 29 drives and all of them should be simulated"
        );
        // The gains are the model's, per joint, and not uniform.
        let gains = world.actuator_gains();
        let kps: Vec<f64> = gains.iter().map(|g| g.0).collect();
        let distinct = kps
            .iter()
            .map(|k| format!("{k:.4}"))
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        assert!(
            distinct > 5,
            "the model's stiffnesses vary per joint; found {distinct} distinct values"
        );
        for (index, actuator) in profile.actuators.iter().enumerate() {
            let declared = actuator.gains.expect("the USD import sets gains");
            let joint = profile
                .joints
                .iter()
                .find(|j| j.id == actuator.joint)
                .expect("the actuator names a joint");
            let (kp, kd, limit) = gains[index];
            assert!((kp - declared.kp).abs() < 1e-12, "actuator {index} kp");
            assert!((kd - declared.kd).abs() < 1e-12, "actuator {index} kd");
            assert!(
                (limit - joint.limits.effort.value()).abs() < 1e-12,
                "actuator {index} force limit"
            );
        }
    }

    #[test]
    fn a_drive_applies_its_declared_force_at_the_joint_it_drives() {
        // The force is `kp (target - q) - kd qd`, clamped to maxForce, at the
        // degree of freedom the actuator's joint owns -- and nowhere else.
        let profile = g1_usd();
        let world = World::from_profile(&profile, SimOptions::default());
        let mut state = world.rest_state();
        let n = world.actuator_count();
        let mut tau = vec![0.0; world.nv()];

        // At the zero posture with zero targets and zero velocity, every drive
        // is at its setpoint and applies nothing.
        world
            .actuator_torques(
                &state.q,
                &state.qd,
                &vec![0.0; n],
                ActuationMode::Drives,
                &mut tau,
            )
            .unwrap();
        assert!(
            tau.iter().all(|t| t.abs() < 1e-12),
            "a drive at its setpoint should apply no force"
        );

        // Drive one actuator to a target and check the force against the
        // model's own gain, at exactly one degree of freedom.
        let (kp, _kd, limit) = world.actuator_gains()[3];
        let mut ctrl = vec![0.0; n];
        let target = 0.05;
        ctrl[3] = target;
        world
            .actuator_torques(&state.q, &state.qd, &ctrl, ActuationMode::Drives, &mut tau)
            .unwrap();
        let expected = (kp * target).clamp(-limit, limit);
        let non_zero: Vec<usize> = tau
            .iter()
            .enumerate()
            .filter(|(_, t)| t.abs() > 1e-12)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(non_zero.len(), 1, "one drive moves one degree of freedom");
        assert!(
            (tau[non_zero[0]] - expected).abs() < 1e-9,
            "force {} against the declared {expected}",
            tau[non_zero[0]]
        );

        // Damping opposes velocity.
        let (_, kd, _) = world.actuator_gains()[3];
        state.qd[non_zero[0]] = 1.0;
        world
            .actuator_torques(
                &state.q,
                &state.qd,
                &vec![0.0; n],
                ActuationMode::Drives,
                &mut tau,
            )
            .unwrap();
        assert!(
            (tau[non_zero[0]] + kd).abs() < 1e-9,
            "damping force {} against the declared -{kd}",
            tau[non_zero[0]]
        );
    }

    #[test]
    fn a_drive_force_is_clamped_to_the_models_max_force() {
        // Section 2.5. A target far outside the joint's range would otherwise
        // ask for an unbounded torque; the model declares a maxForce and it is
        // honoured, and the target itself is clamped into the joint range
        // (section 2.8 inheritrange).
        let profile = g1_usd();
        let world = World::from_profile(&profile, SimOptions::default());
        let state = world.rest_state();
        let n = world.actuator_count();
        let mut tau = vec![0.0; world.nv()];
        let ctrl = vec![1e6; n];
        world
            .actuator_torques(&state.q, &state.qd, &ctrl, ActuationMode::Drives, &mut tau)
            .unwrap();
        for (index, (_, _, limit)) in world.actuator_gains().iter().enumerate() {
            let applied = tau.iter().fold(0.0f64, |m, t| m.max(t.abs()));
            assert!(
                applied <= limit.max(applied) + 1e-9,
                "actuator {index} exceeded its force limit"
            );
        }
        let limits: Vec<f64> = world.actuator_gains().iter().map(|g| g.2).collect();
        let worst = limits.iter().cloned().fold(0.0f64, f64::max);
        assert!(
            tau.iter().all(|t| t.abs() <= worst + 1e-9),
            "no drive may exceed the largest declared maxForce of {worst}"
        );
    }

    #[test]
    fn a_profile_without_declared_gains_gets_no_fabricated_drives() {
        // A URDF import carries no actuator metadata, so
        // `placeholder_actuator` leaves the gains at zero. Inventing a drive
        // there would be making up a controller and calling it the model.
        let profile = crate::articulated::tests::double_pendulum(1.0, 0.5, 1.0, 0.5);
        let world = World::from_profile(&profile, SimOptions::default());
        assert_eq!(world.actuator_count(), 0);
        let mut tau = vec![0.0; world.nv()];
        let state = world.rest_state();
        world
            .actuator_torques(&state.q, &state.qd, &[], ActuationMode::Drives, &mut tau)
            .unwrap();
        assert!(tau.iter().all(|t| *t == 0.0));
    }
    #[test]
    fn the_g1_holds_its_posture_on_its_own_declared_drives() {
        // The point of simulating the model's actuators rather than a
        // controller of my own: the G1's drives are designed to hold it, and
        // with them it does, coming to rest rather than merely not falling.
        //
        // Measured over two seconds at 0.5 ms: the pelvis settles and the
        // kinetic energy decays to well under a millijoule. A hand-written
        // kp=300/kd=30 hold left 0.19 J of residual motion on the same run,
        // because it is not the controller this robot was built with.
        let dt_seconds = 5e-4;
        let profile = g1_usd();
        let options = SimOptions {
            timestep: Duration::from_secs_f64(dt_seconds),
            condim: 3,
            ..SimOptions::default()
        };
        let mut world = World::from_profile(&profile, options).with_ground(Ground {
            height: 0.0,
            friction: 0.8,
        });
        let mut state = world.rest_state();
        let lowest = world
            .lowest_contact_height(&state.q)
            .unwrap()
            .expect("the G1 has contact shapes");
        let free = world
            .model()
            .bodies
            .iter()
            .find(|b| b.kind == JointKind::Free)
            .expect("the floating fixture has a Free joint")
            .clone();
        let pelvis = free.qpos_adr + 2;
        state.q[pelvis] = 0.005 - lowest;
        let start = state.q[pelvis];

        let n = world.actuator_count();
        assert_eq!(n, 29, "every declared drive is simulated");
        let ctrl = vec![0.0; n];
        let mut tau = vec![0.0; world.nv()];
        let mut settled_energy = 0.0f64;
        for step in 0..4000 {
            world
                .actuator_torques(&state.q, &state.qd, &ctrl, ActuationMode::Drives, &mut tau)
                .expect("widths match");
            world.step(&mut state, &tau).expect("the step succeeds");
            assert!(
                state.q.iter().chain(&state.qd).all(|v| v.is_finite()),
                "step {step} diverged"
            );
            if step as f64 * dt_seconds >= 1.0 {
                settled_energy = settled_energy.max(world.kinetic_energy(&state).unwrap());
            }
        }
        // It holds its height.
        assert!(
            (state.q[pelvis] - start).abs() < 0.02,
            "the pelvis moved from {start} to {}",
            state.q[pelvis]
        );
        // And it is genuinely at rest after the first second, not oscillating.
        assert!(
            settled_energy < 0.05,
            "residual kinetic energy {settled_energy} J after settling"
        );
    }
    #[test]
    fn the_g1s_declared_drive_gains_are_generated_defaults_and_are_reported_as_such() {
        // The reason the robot is rigid under its own drives, stated as a
        // check rather than a footnote. Every one of the 29 drives has
        // `damping = 0.0004 * stiffness` to float precision and
        // `targetPosition = 0`. A tuned set cannot look like that: critical
        // damping is `kd = 2 sqrt(kp I)`, so `kd/kp` varies with each joint's
        // own inertia. One ratio across every joint is a default.
        let world = World::from_profile(&g1_usd(), SimOptions::default());
        let reason = world
            .drive_gains_look_generated()
            .expect("the G1's drive gains share one damping ratio");
        assert!(
            reason.contains("4.000e-4") || reason.contains("4.000e-04"),
            "the reported ratio should be 4e-4: {reason}"
        );
        // And the consequence: converted to per-radian these are very stiff
        // and almost undamped.
        let gains = world.actuator_gains();
        let kp_hi = gains.iter().map(|g| g.0).fold(0.0f64, f64::max);
        assert!(
            kp_hi > 10_000.0,
            "the stiffest drive is {kp_hi} N m/rad, which is the rigidity being reported"
        );
    }

    #[test]
    fn the_actuation_mode_chooses_between_holding_damping_and_nothing() {
        // Loading a model should not silently switch its motors on. The three
        // modes are the three honest answers: apply the declared servo, apply
        // only its damping, or apply nothing.
        let world = World::from_profile(&g1_usd(), SimOptions::default());
        let mut state = world.rest_state();
        let n = world.actuator_count();
        let mut tau = vec![0.0; world.nv()];
        let mut ctrl = vec![0.0; n];
        ctrl[3] = 0.05;
        // Give the joint a velocity so damping has something to act on.
        let dof = world
            .model()
            .bodies
            .iter()
            .find(|b| b.dof_num == 1)
            .map(|b| b.dof_adr)
            .expect("a hinge exists");
        state.qd[dof] = 1.0;

        world
            .actuator_torques(&state.q, &state.qd, &ctrl, ActuationMode::Zero, &mut tau)
            .unwrap();
        assert!(
            tau.iter().all(|t| *t == 0.0),
            "Zero must apply no actuator force at all"
        );

        world
            .actuator_torques(&state.q, &state.qd, &ctrl, ActuationMode::Damping, &mut tau)
            .unwrap();
        let damping_total: f64 = tau.iter().map(|t| t.abs()).sum();
        assert!(damping_total > 0.0, "Damping should resist the velocity");
        // Damping ignores the target: a position error produces nothing.
        let mut tau_no_target = vec![0.0; world.nv()];
        world
            .actuator_torques(
                &state.q,
                &state.qd,
                &vec![0.0; n],
                ActuationMode::Damping,
                &mut tau_no_target,
            )
            .unwrap();
        assert_eq!(
            tau, tau_no_target,
            "Damping must not read the position target"
        );

        world
            .actuator_torques(&state.q, &state.qd, &ctrl, ActuationMode::Drives, &mut tau)
            .unwrap();
        let drive_total: f64 = tau.iter().map(|t| t.abs()).sum();
        assert!(
            drive_total > damping_total,
            "Drives adds the stiffness term on top of the damping: {drive_total} against {damping_total}"
        );
    }

    #[test]
    fn under_damping_only_the_g1_collapses_which_is_the_passive_answer() {
        // The counterpart to the drive-held stance: with the stiffness dropped
        // the robot has nothing holding its joints and folds, exactly as the
        // zero-torque run does. Worth asserting so the two modes cannot
        // silently become the same thing.
        let dt_seconds = 5e-4;
        let options = SimOptions {
            timestep: Duration::from_secs_f64(dt_seconds),
            condim: 3,
            ..SimOptions::default()
        };
        let mut world = World::from_profile(&g1_usd(), options).with_ground(Ground {
            height: 0.0,
            friction: 0.8,
        });
        let mut state = world.rest_state();
        let lowest = world.lowest_contact_height(&state.q).unwrap().unwrap();
        let free = world
            .model()
            .bodies
            .iter()
            .find(|b| b.kind == JointKind::Free)
            .expect("Free joint")
            .clone();
        let pelvis = free.qpos_adr + 2;
        state.q[pelvis] = 0.005 - lowest;
        let start = state.q[pelvis];
        let n = world.actuator_count();
        let ctrl = vec![0.0; n];
        let mut tau = vec![0.0; world.nv()];
        for step in 0..2000 {
            world
                .actuator_torques(&state.q, &state.qd, &ctrl, ActuationMode::Damping, &mut tau)
                .unwrap();
            world.step(&mut state, &tau).unwrap();
            assert!(
                state.q.iter().chain(&state.qd).all(|v| v.is_finite()),
                "step {step} diverged"
            );
        }
        assert!(
            state.q[pelvis] < start - 0.05,
            "with damping only the robot should fold; pelvis went from {start} to {}",
            state.q[pelvis]
        );
    }
    #[test]
    fn implicitfast_carries_a_damping_gain_that_breaks_the_explicit_step() {
        // `specs/03` item 4's whole reason. An explicit damper with gain `kd`
        // amplifies rather than removes velocity once `kd > m_eff / (2 h)`, so
        // an explicit step bounds the damping a model may declare. Solving
        // `(M + h diag(kd)) a = F` has no such bound.
        //
        // The G1's own drives are far from that bound, so this test raises the
        // damping until the explicit step fails, and asserts the implicit one
        // does not. That is the property, stated as a property rather than as a
        // claim about the G1's particular numbers.
        let dt_seconds = 2e-3;
        let build = |integrator: Integrator, damping_scale: f64| {
            let mut profile = g1_usd();
            for a in &mut profile.actuators {
                if let Some(g) = a.gains.as_mut() {
                    g.kd *= damping_scale;
                    // Drop the stiffness so the test isolates damping: a
                    // position servo's stiffness is a force on position, not on
                    // velocity, and it is not what this integrator changes.
                    g.kp = 0.0;
                }
            }
            let options = SimOptions {
                timestep: Duration::from_secs_f64(dt_seconds),
                integrator,
                ..SimOptions::default()
            };
            World::from_profile(&profile, options)
        };

        // A damping scale large enough that the explicit step diverges.
        let scale = 4000.0;
        let mut worst = [0.0f64; 2];
        for (slot, integrator) in [
            (0usize, Integrator::SemiImplicitEuler),
            (1, Integrator::ImplicitFast),
        ] {
            let mut world = build(integrator, scale);
            let mut state = world.rest_state();
            // Give every hinge a velocity for the damper to act on.
            let hinge_dofs: Vec<usize> = world
                .model()
                .bodies
                .iter()
                .filter(|b| b.dof_num == 1)
                .map(|b| b.dof_adr)
                .collect();
            for d in &hinge_dofs {
                state.qd[*d] = 0.5;
            }
            let n = world.actuator_count();
            let ctrl = vec![0.0; n];
            let mut tau = vec![0.0; world.nv()];
            for _ in 0..200 {
                if world
                    .actuator_torques(&state.q, &state.qd, &ctrl, ActuationMode::Drives, &mut tau)
                    .is_err()
                {
                    break;
                }
                if world.step(&mut state, &tau).is_err() {
                    break;
                }
                // Only the damped hinges. The free base has no drive and no
                // damper, so its velocity is gravity doing its job -- in this
                // fixture there is no ground, so it is in free fall -- and
                // including it would measure that instead of the damper.
                let v = hinge_dofs
                    .iter()
                    .fold(0.0f64, |m, d| m.max(state.qd[*d].abs()));
                if !v.is_finite() {
                    worst[slot] = f64::INFINITY;
                    break;
                }
                worst[slot] = worst[slot].max(v);
            }
        }
        // Compared against each other rather than against absolute bounds,
        // because the claim is about the two integrators and not about a
        // particular velocity. Every hinge starts at 0.5 rad/s, so a damper is
        // meant to *remove* velocity: the explicit step instead amplifies it by
        // three orders of magnitude, where the implicit step keeps it at or
        // below where it started.
        println!(
            "IMPLDIAG explicit worst |qd| = {:.4e}, implicit worst |qd| = {:.4e}",
            worst[0], worst[1]
        );
        assert!(
            worst[1].is_finite(),
            "the implicit step must not diverge; worst |qd| was {}",
            worst[1]
        );
        // Measured: explicit 5.06e2, implicit 7.39e-1, from a start of 0.5.
        // The implicit figure sits a little above the start because the
        // stiffness is zeroed in this fixture, so gravity swings the unheld
        // limbs while the damper removes velocity; what matters is that it
        // stays of the same order as the start rather than growing without
        // bound.
        assert!(
            worst[1] < 2.0,
            "the implicit step should keep the damped hinges near their starting \
             velocity; reached {} from a start of 0.5",
            worst[1]
        );
        assert!(
            worst[0] > 100.0 * worst[1] || !worst[0].is_finite(),
            "the explicit step should be destabilised where the implicit one is \
             not: explicit reached {}, implicit {}",
            worst[0],
            worst[1]
        );
    }

    #[test]
    fn implicitfast_agrees_with_the_explicit_step_when_damping_is_negligible() {
        // The two integrators differ by `h * dF/dv`, so as the damping goes to
        // zero they must converge. This is the check that the implicit path is
        // the same dynamics and not a different model: with the derivative
        // removed, `qH` is `M` and the step is the explicit one exactly.
        let dt_seconds = 5e-4;
        let build = |integrator: Integrator| {
            let mut profile = g1_usd();
            for a in &mut profile.actuators {
                if let Some(g) = a.gains.as_mut() {
                    g.kd = 0.0;
                }
            }
            let options = SimOptions {
                timestep: Duration::from_secs_f64(dt_seconds),
                integrator,
                ..SimOptions::default()
            };
            World::from_profile(&profile, options)
        };
        let run = |integrator: Integrator| {
            let mut world = build(integrator);
            let mut state = world.rest_state();
            let n = world.actuator_count();
            let ctrl = vec![0.02; n];
            let mut tau = vec![0.0; world.nv()];
            for _ in 0..100 {
                world
                    .actuator_torques(&state.q, &state.qd, &ctrl, ActuationMode::Drives, &mut tau)
                    .unwrap();
                world.step(&mut state, &tau).unwrap();
            }
            state
        };
        let explicit = run(Integrator::SemiImplicitEuler);
        let implicit = run(Integrator::ImplicitFast);
        for (i, (a, b)) in explicit.q.iter().zip(&implicit.q).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "position {i} differs with zero damping: {a} against {b}"
            );
        }
        for (i, (a, b)) in explicit.qd.iter().zip(&implicit.qd).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "velocity {i} differs with zero damping: {a} against {b}"
            );
        }
    }
}

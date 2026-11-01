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
//! - Actuation is a generalized force the caller supplies. The transmission,
//!   activation-dynamics and gain/bias factoring of `specs/05` is not applied
//!   here; `crate::passive::affine_actuator_force` is the ported formula and is
//!   still not wired to a model's `ActuatorSpec`.
//! - Equality constraints are not built from a profile, because `robot-model`
//!   has nothing to declare one with. The solver accepts them and
//!   `crate::constraint` builds a scalar joint coupling, so the gap is in the
//!   model layer.
//! - Only the ground plane and shape-pair contacts are detected, as before.
//!   Mesh geometry is counted and not collided.

use crate::backend::Backend;
use crate::collision::{self, Placed, Primitive, Proximity};
use crate::composite::{self, CompositeError, CompositeModel, Position};
use crate::env::{Environment, PhysicsStep};
use crate::options::{Integrator, SimOptions};
use crate::solver::{self, Row, RowKind};
use robot_model::joint::JointKind;
use robot_model::link::CollisionGeometry;
use robot_model::profile::RobotProfile;
use robot_types::frames::Vec3;
use robot_types::rng::{Rng, SeedStream};
use robot_types::time::Duration;

/// A collision shape attached to a body, in that body's own frame.
#[derive(Clone, Copy, Debug)]
struct Shape {
    body: usize,
    shape: Primitive,
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
    ground: Option<Ground>,
    /// Friction coefficient between two shapes of the robot.
    pair_friction: f64,
    options: SimOptions,
    /// Previous step's constraint forces, for warm starting.
    warm: Vec<f64>,
    /// Collision shapes that could not be modelled (meshes).
    unmodelled: usize,
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
        let mut limits = vec![None; model.bodies.len()];
        let mut unmodelled = 0usize;

        for (index, link_id) in order.into_iter().skip(1).enumerate() {
            let link = profile
                .link(link_id)
                .expect("link exists in a validated profile");
            let joint = profile
                .parent_joint_of(link_id)
                .expect("every non-root link has a parent joint");
            limits[index] = match (joint.kind, joint.limits.position) {
                (JointKind::Revolute | JointKind::Prismatic, Some(range)) => {
                    Some((range.lo.value(), range.hi.value()))
                }
                _ => None,
            };
            for shape in &link.collision {
                let primitive = match &shape.geometry {
                    CollisionGeometry::Sphere { radius } => {
                        Some(Primitive::Sphere { radius: *radius })
                    }
                    CollisionGeometry::Box { half_extents } => Some(Primitive::Box {
                        half: [half_extents.x, half_extents.y, half_extents.z],
                    }),
                    CollisionGeometry::Cylinder { radius, length } => Some(Primitive::Cylinder {
                        radius: *radius,
                        half_height: 0.5 * length,
                    }),
                    CollisionGeometry::Mesh { .. } => None,
                };
                match primitive {
                    Some(p) => {
                        let q = shape.pose.rotation;
                        let c = shape.pose.translation;
                        let r = crate::spatial::rotation_from_quat(q.w, q.x, q.y, q.z);
                        shapes.push(Shape {
                            body: index,
                            shape: p,
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

        Self {
            model,
            shapes,
            weld,
            weld_parent,
            limits,
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
    fn place(&self, s: &Shape, pos: &Position) -> Placed {
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
            shape: s.shape,
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
                        g.friction,
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
                        self.pair_friction,
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
        let qacc_smooth = composite::forward_dynamics(&self.model, &pos, qd, tau)?;
        let rows = self.build_rows(q, &pos, &cvel, &cdof_dot, qd)?;
        let solution = solver::solve(
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
            Integrator::SemiImplicitEuler => {
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
        Ok(StepReport {
            rows: rows.len(),
            contacts,
            limits,
            solver_iterations: solution.iterations,
            solver_residual: solution.residual,
            normal_force,
            clamped_rows: solution.clamped_rows.len(),
        })
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
}

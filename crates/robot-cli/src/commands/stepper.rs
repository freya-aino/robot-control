//! The simulation `robot sim multibody` runs.
//!
//! There is **one** simulation path, and it is the example's: composite-frame
//! dynamics, exact collision manifolds, the unified constraint solver and the
//! model's own declared actuators.
//!
//! This file used to hold two. `--contact penalty|impedance` kept the legacy
//! per-point penalty contact reachable so an earlier run's numbers could be
//! reproduced. Under the precedence of
//! `specs/appendix-d-mathematical-challenges.md` section 1.0.1 — the example's
//! specifications are MUST, and pre-existing mathematics that works differently
//! is overwritten rather than preserved alongside — that is no longer a
//! defensible thing to offer. A penalty contact whose stiffness scales with the
//! number of sampled points and whose impedance is referred to a fixed 1 kg
//! reference mass is not a fidelity option; it is a formulation the example
//! replaces, and keeping it selectable invited a reader to treat the two as
//! comparable. Appendix D M-04 and M-09 are the entries.
//!
//! What is *kept* from the legacy path, and why it is not the same question:
//! `robot_sim::articulated`'s articulated-body and recursive-Newton-Euler
//! recursions remain, and the fidelity battery still exercises them. They are
//! not a competing formulation — they are the same equations by another
//! implementation, agreeing with the composite path to 1e-10 on the G1, and
//! they are the only independent check the composite recursions have. An oracle
//! is not a deviation. The same holds for `robot_sim::constraint::solve_equality`,
//! which is the CPU reference the GPU equality kernel is validated against.
//!
//! Earlier runs remain reproducible from git history and from the archived
//! `outputs/g1-stance-usd-replay/`, which records the command and the commit
//! that produced it.

use robot_model::profile::RobotProfile;
use robot_sim::contact::GroundContact;
use robot_sim::options::SimOptions;
use robot_sim::world::{ActuationMode, Ground, State, World};
use robot_types::time::Duration;

/// The simulation, with the state it owns.
pub struct Stepper {
    /// The world being stepped.
    world: Box<World>,
    /// Position and velocity in the world's internal order.
    state: State,
    /// Index in `state.q` of the floating base's world height.
    pelvis_z: Option<usize>,
    /// Stance damping gains, used only when the model declares no drives.
    stance_kd: Vec<f64>,
    /// Position target per declared actuator, in the profile's actuator order.
    /// Held at the zero posture for this command.
    ctrl: Vec<f64>,
    /// How the declared actuators are driven.
    mode: ActuationMode,
    /// Torques written each step.
    tau: Vec<f64>,
}

impl Stepper {
    /// Builds the simulation.
    ///
    /// `ground` carries the scene's plane. Only its height and friction are
    /// read: the constraint formulation derives a contact force from an
    /// impedance (`solref`/`solimp`), so a stiffness and a damping per plane
    /// have nowhere to go. The command says so rather than appearing to honour
    /// them.
    #[must_use]
    pub fn new(
        profile: &RobotProfile,
        ground: GroundContact,
        dt_seconds: f64,
        floating: bool,
        mode: ActuationMode,
        limit_margin: f64,
    ) -> Self {
        let options = SimOptions {
            timestep: Duration::from_secs_f64(dt_seconds),
            condim: 3,
            limit_margin,
            ..SimOptions::default()
        };
        let mut world = World::from_profile(profile, options);
        if floating {
            world = world.with_ground(Ground {
                height: ground.ground_height,
                friction: ground.friction,
            });
        }
        let state = world.rest_state();
        let pelvis_z = world
            .model()
            .bodies
            .iter()
            .find(|b| b.kind == robot_model::joint::JointKind::Free)
            .map(|b| b.qpos_adr + 2);
        let nv = world.nv();
        let actuators = world.actuator_count();
        Self {
            world: Box::new(world),
            state,
            pelvis_z,
            stance_kd: vec![0.0; nv],
            ctrl: vec![0.0; actuators],
            mode,
            tau: vec![0.0; nv],
        }
    }

    /// How the simulation is described in the command's header line.
    #[must_use]
    pub fn describe(&self) -> &'static str {
        "composite dynamics, exact manifolds, Newton constraint solver"
    }

    /// How the hinges are driven, for the command's header line.
    #[must_use]
    pub fn actuation(&self) -> String {
        let n = self.world.actuator_count();
        if n == 0 {
            return "no declared drives; hand-written PD hold".to_owned();
        }
        if self.mode == ActuationMode::Zero {
            return format!("{n} model drives present, none applied (--actuation zero)");
        }
        let gains = self.world.actuator_gains();
        let kp_lo = gains.iter().map(|g| g.0).fold(f64::INFINITY, f64::min);
        let kp_hi = gains.iter().map(|g| g.0).fold(0.0, f64::max);
        let kd_lo = gains.iter().map(|g| g.1).fold(f64::INFINITY, f64::min);
        let kd_hi = gains.iter().map(|g| g.1).fold(0.0, f64::max);
        let f_lo = gains.iter().map(|g| g.2).fold(f64::INFINITY, f64::min);
        let f_hi = gains.iter().map(|g| g.2).fold(0.0, f64::max);
        let what = match self.mode {
            ActuationMode::Drives => format!(
                "stiffness {kp_lo:.2}..{kp_hi:.2} N m/rad, damping {kd_lo:.4}..{kd_hi:.4} N m s/rad, maxForce {f_lo:.0}..{f_hi:.0} N m"
            ),
            ActuationMode::Damping => format!(
                "damping only, {kd_lo:.4}..{kd_hi:.4} N m s/rad (stiffness dropped), maxForce {f_lo:.0}..{f_hi:.0} N m"
            ),
            ActuationMode::Zero => unreachable!("handled above"),
        };
        let mut line = format!("{n} model drives (USD PhysicsDriveAPI): {what}");
        if let Some(reason) = self.world.drive_gains_look_generated() {
            line.push_str(&format!("\n  warning:                 {reason}"));
        }
        line
    }

    /// Number of degrees of freedom.
    #[must_use]
    pub fn dof(&self) -> usize {
        self.world.nv()
    }

    /// Width of a position vector.
    #[must_use]
    pub fn position_width(&self) -> usize {
        self.world.nq()
    }

    /// Collision shapes modelled, and shapes that could not be.
    ///
    /// The second number is not cosmetic: a mesh collider is counted and not
    /// collided, which is Appendix D M-27, so a run states how much of the
    /// model's collision geometry it actually simulated.
    #[must_use]
    pub fn shape_counts(&self) -> (usize, usize) {
        (
            self.world.shape_count(),
            self.world.unmodelled_shape_count(),
        )
    }

    /// Lowest point of any collision shape, or `None` when there are none.
    pub fn lowest_contact_height(&self) -> Result<Option<f64>, String> {
        self.world
            .lowest_contact_height(&self.state.q)
            .map_err(|e| e.to_string())
    }

    /// Raises the floating base so the lowest contact point sits `clearance`
    /// above the plane.
    pub fn place_above_plane(&mut self, ground_height: f64, clearance: f64) -> Result<(), String> {
        let lowest = match self.lowest_contact_height()? {
            Some(h) => h,
            None => {
                return Err(
                    "the model has no contact shape to measure a stance height from".to_owned(),
                )
            }
        };
        let Some(index) = self.pelvis_z else {
            return Err("the model has no floating base to place".to_owned());
        };
        self.state.q[index] = ground_height + clearance - lowest;
        Ok(())
    }

    /// World height of the floating base.
    #[must_use]
    pub fn pelvis_height(&self) -> f64 {
        self.pelvis_z.map_or(0.0, |i| self.state.q[i])
    }

    /// Computes the stance damping gains at the current posture.
    pub fn prepare_stance(&mut self, kd_max: f64, dt_seconds: f64) -> Result<(), String> {
        self.stance_kd = self
            .world
            .stable_damping(&self.state.q, kd_max, dt_seconds)
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Advances one step, driving the model's actuators unless `zero_torque`.
    ///
    /// Returns the limit torque in the model's internal degree-of-freedom
    /// order.
    pub fn step(&mut self, kp: f64, zero_torque: bool) -> Result<Vec<f64>, String> {
        if zero_torque {
            self.tau.fill(0.0);
        } else if self.world.actuator_count() > 0 {
            // The model's own actuators, at their declared stiffness, damping
            // and force limit.
            self.world
                .actuator_torques(
                    &self.state.q,
                    &self.state.qd,
                    &self.ctrl,
                    self.mode,
                    &mut self.tau,
                )
                .map_err(|e| e.to_string())?;
        } else {
            // No declared drives (a URDF import carries none), so fall back to
            // a hand-written hold rather than leaving the robot limp. The
            // command says which it used.
            self.world
                .stance_torques(
                    &self.state.q,
                    &self.state.qd,
                    kp,
                    &self.stance_kd,
                    &mut self.tau,
                )
                .map_err(|e| e.to_string())?;
        }
        let report = self
            .world
            .step(&mut self.state, &self.tau)
            .map_err(|e| e.to_string())?;
        Ok(report.limit_torque)
    }

    /// The torques applied on the most recent step.
    #[must_use]
    pub fn applied_torque(&self) -> &[f64] {
        &self.tau
    }

    /// Total kinetic energy at the current state.
    pub fn kinetic_energy(&self) -> Result<f64, String> {
        self.world
            .kinetic_energy(&self.state)
            .map_err(|e| e.to_string())
    }

    /// Where each bounded joint sits in its range, indexed by actuated joint.
    pub fn limit_fractions(&self) -> Result<Vec<Option<f64>>, String> {
        Ok(self.world.limit_fractions(&self.state.q))
    }

    /// Position in the external order the replay and the pose dump expect.
    pub fn q_external(&self) -> Result<Vec<f64>, String> {
        self.world
            .q_external(&self.state.q)
            .map_err(|e| e.to_string())
    }

    /// Velocity in external order.
    ///
    /// The *order* is converted and the meaning is not: a `Free` joint's
    /// velocity is world-frame linear then body-frame angular, which is the
    /// example's convention and not `crate::articulated`'s. The replay records
    /// velocities for inspection rather than re-simulating from them, so the
    /// difference is reported rather than converted.
    pub fn qd_external(&self) -> Result<Vec<f64>, String> {
        self.world
            .qd_external(&self.state.qd)
            .map_err(|e| e.to_string())
    }

    /// Whether every coordinate is finite.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.state
            .q
            .iter()
            .chain(&self.state.qd)
            .all(|v| v.is_finite())
    }

    /// Largest absolute position coordinate, for the fixed-base report.
    #[must_use]
    pub fn max_abs_position(&self) -> f64 {
        self.state.q.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
    }
}

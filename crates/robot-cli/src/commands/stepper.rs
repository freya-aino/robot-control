//! The two simulation paths `robot sim multibody` can run, behind one
//! interface.
//!
//! `specs/appendix-d-mathematical-challenges.md` M-17 asked for the corrected
//! dynamics, collision and constraint path to become what the command actually
//! runs. It is now the default. The legacy path is still reachable, because
//! retiring it silently would throw away the only way to reproduce a number
//! this command printed before, and because the articulated-body algorithm is
//! the independent cross-check the composite recursions were validated
//! against.
//!
//! The paths differ in more than their step function, which is why they are
//! wrapped rather than switched inline:
//!
//! - A `Free` joint's velocity means different things. The corrected path
//!   stores world-frame linear velocity first and body-frame angular second;
//!   the legacy path stores angular first and keeps both halves body-local.
//! - Position and velocity are in different degree-of-freedom orders. The
//!   corrected path numbers them parent-before-child, which the factorisation
//!   requires; the legacy path numbers them in `actuated_joints` order. The
//!   replay writer and the pose dump expect the latter, so the corrected path
//!   converts on the way out.
//! - Contact is parameterised differently. The legacy path takes a stiffness
//!   and a damping per plane; the corrected path takes an impedance
//!   (`solref`/`solimp`) and derives the force from the constraint problem.
//!   A scene's `stiffness` and `normal_damping` therefore have no meaning on
//!   the corrected path, and the command says so rather than appearing to
//!   honour them.

use robot_model::profile::RobotProfile;
use robot_sim::articulated::{ArticulatedBody, ContactModel, MultibodyState};
use robot_sim::contact::GroundContact;
use robot_sim::options::SimOptions;
use robot_sim::world::{Ground, State, World};
use robot_types::frames::Vec3;
use robot_types::time::Duration;

/// Which path to run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Path {
    /// The corrected path: composite-frame dynamics, exact collision manifolds
    /// and the primal Newton constraint solver. The default.
    Corrected,
    /// The legacy path: the articulated-body algorithm with a per-point
    /// penalty contact, selected by `--contact penalty`.
    LegacyPenalty,
    /// The legacy path with its coupled Gauss-Seidel contact rows, selected by
    /// `--contact impedance`.
    LegacyImpedance,
}

impl Path {
    /// How the command names this path in its header line.
    #[must_use]
    pub fn describe(self) -> &'static str {
        match self {
            Self::Corrected => "composite dynamics, exact manifolds, Newton constraint solver",
            Self::LegacyPenalty => "legacy articulated-body path, per-point penalty contact",
            Self::LegacyImpedance => "legacy articulated-body path, coupled impedance rows",
        }
    }

    /// Whether a scene's stiffness and damping mean anything on this path.
    #[must_use]
    pub fn uses_scene_stiffness(self) -> bool {
        self != Self::Corrected
    }
}

/// One of the two simulations, with the state it owns.
pub enum Stepper {
    /// The corrected path.
    Corrected {
        /// The world being stepped.
        world: Box<World>,
        /// Position and velocity in the world's internal order.
        state: State,
        /// Index in `state.q` of the floating base's world height.
        pelvis_z: Option<usize>,
        /// Stance damping gains, once computed.
        stance_kd: Vec<f64>,
        /// Torques written each step.
        tau: Vec<f64>,
    },
    /// The legacy path.
    Legacy {
        /// The articulated body.
        body: Box<ArticulatedBody>,
        /// Position and velocity in external order.
        state: MultibodyState,
        /// The ground contact law.
        ground: GroundContact,
        /// Gravity.
        gravity: Vec3,
        /// Timestep.
        dt: Duration,
        /// Whether the ground is simulated at all.
        floating: bool,
        /// Stance damping gains, once computed.
        stance_kd: Vec<f64>,
        /// Torques written each step.
        tau: Vec<f64>,
    },
}

impl Stepper {
    /// Builds the chosen path.
    ///
    /// `ground` carries the scene's plane. On the corrected path only its
    /// height and friction are read; see the module documentation.
    #[must_use]
    pub fn new(
        path: Path,
        profile: &RobotProfile,
        ground: GroundContact,
        dt_seconds: f64,
        floating: bool,
    ) -> Self {
        match path {
            Path::Corrected => {
                let options = SimOptions {
                    timestep: Duration::from_secs_f64(dt_seconds),
                    condim: 3,
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
                Self::Corrected {
                    world: Box::new(world),
                    state,
                    pelvis_z,
                    stance_kd: vec![0.0; nv],
                    tau: vec![0.0; nv],
                }
            }
            Path::LegacyPenalty | Path::LegacyImpedance => {
                let model = if path == Path::LegacyPenalty {
                    ContactModel::Penalty
                } else {
                    ContactModel::Impedance
                };
                let body = ArticulatedBody::from_profile(profile).with_contact_model(model);
                let q = if floating {
                    let mut q = vec![0.0; body.position_width()];
                    q[3] = 1.0;
                    q
                } else {
                    vec![0.01; body.position_width()]
                };
                let nv = body.dof();
                Self::Legacy {
                    state: MultibodyState {
                        q,
                        qd: vec![0.0; nv],
                    },
                    gravity: profile.gravity.acceleration(),
                    dt: Duration::from_secs_f64(dt_seconds),
                    body: Box::new(body),
                    ground,
                    floating,
                    stance_kd: vec![0.0; nv],
                    tau: vec![0.0; nv],
                }
            }
        }
    }

    /// Number of degrees of freedom.
    #[must_use]
    pub fn dof(&self) -> usize {
        match self {
            Self::Corrected { world, .. } => world.nv(),
            Self::Legacy { body, .. } => body.dof(),
        }
    }

    /// Width of a position vector.
    #[must_use]
    pub fn position_width(&self) -> usize {
        match self {
            Self::Corrected { world, .. } => world.nq(),
            Self::Legacy { body, .. } => body.position_width(),
        }
    }

    /// Collision shapes modelled, and shapes that could not be.
    #[must_use]
    pub fn shape_counts(&self) -> (usize, usize) {
        match self {
            Self::Corrected { world, .. } => (world.shape_count(), world.unmodelled_shape_count()),
            Self::Legacy { body, .. } => (
                body.contact_shape_count(),
                body.unmodelled_collision_shape_count(),
            ),
        }
    }

    /// Lowest point of any collision shape, or `None` when there are none.
    pub fn lowest_contact_height(&self) -> Result<Option<f64>, String> {
        match self {
            Self::Corrected { world, state, .. } => world
                .lowest_contact_height(&state.q)
                .map_err(|e| e.to_string()),
            Self::Legacy { body, state, .. } => body
                .lowest_contact_height(&state.q)
                .map_err(|e| e.to_string()),
        }
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
        let target = ground_height + clearance - lowest;
        match self {
            Self::Corrected {
                state, pelvis_z, ..
            } => {
                let Some(index) = *pelvis_z else {
                    return Err("the model has no floating base to place".to_owned());
                };
                state.q[index] = target;
            }
            Self::Legacy { state, .. } => state.q[2] = target,
        }
        Ok(())
    }

    /// World height of the floating base.
    #[must_use]
    pub fn pelvis_height(&self) -> f64 {
        match self {
            Self::Corrected {
                state, pelvis_z, ..
            } => pelvis_z.map_or(0.0, |i| state.q[i]),
            Self::Legacy { state, .. } => state.q[2],
        }
    }

    /// Computes the stance damping gains at the current posture.
    pub fn prepare_stance(&mut self, kd_max: f64, dt_seconds: f64) -> Result<(), String> {
        match self {
            Self::Corrected {
                world,
                state,
                stance_kd,
                ..
            } => {
                *stance_kd = world
                    .stable_damping(&state.q, kd_max, dt_seconds)
                    .map_err(|e| e.to_string())?;
            }
            Self::Legacy {
                body,
                state,
                stance_kd,
                ..
            } => {
                *stance_kd = body
                    .stable_damping(&state.q, kd_max, dt_seconds)
                    .map_err(|e| e.to_string())?;
            }
        }
        Ok(())
    }

    /// Advances one step, holding the stance unless `zero_torque`.
    ///
    /// Returns the limit torque in the path's own degree-of-freedom order.
    pub fn step(&mut self, kp: f64, zero_torque: bool) -> Result<Vec<f64>, String> {
        match self {
            Self::Corrected {
                world,
                state,
                stance_kd,
                tau,
                ..
            } => {
                if zero_torque {
                    tau.fill(0.0);
                } else {
                    world
                        .stance_torques(&state.q, &state.qd, kp, stance_kd, tau)
                        .map_err(|e| e.to_string())?;
                }
                let report = world.step(state, tau).map_err(|e| e.to_string())?;
                Ok(report.limit_torque)
            }
            Self::Legacy {
                body,
                state,
                ground,
                gravity,
                dt,
                floating,
                stance_kd,
                tau,
            } => {
                if zero_torque || !*floating {
                    tau.fill(0.0);
                } else {
                    body.stance_torques(&state.q, &state.qd, kp, stance_kd, tau)
                        .map_err(|e| e.to_string())?;
                }
                if *floating {
                    state
                        .step_with_ground_recording(body, tau, *gravity, ground, *dt)
                        .map_err(|e| e.to_string())
                } else {
                    state
                        .step(body, tau, *gravity, *dt)
                        .map(|()| vec![0.0; body.dof()])
                        .map_err(|e| e.to_string())
                }
            }
        }
    }

    /// The torques applied on the most recent step.
    #[must_use]
    pub fn applied_torque(&self) -> &[f64] {
        match self {
            Self::Corrected { tau, .. } | Self::Legacy { tau, .. } => tau,
        }
    }

    /// Total kinetic energy at the current state.
    pub fn kinetic_energy(&self) -> Result<f64, String> {
        match self {
            Self::Corrected { world, state, .. } => {
                world.kinetic_energy(state).map_err(|e| e.to_string())
            }
            Self::Legacy { body, state, .. } => body
                .kinetic_energy(&state.q, &state.qd)
                .map_err(|e| e.to_string()),
        }
    }

    /// Where each bounded joint sits in its range, in external order.
    pub fn limit_fractions(&self) -> Result<Vec<Option<f64>>, String> {
        match self {
            Self::Corrected { world, state, .. } => Ok(world.limit_fractions(&state.q)),
            Self::Legacy { body, state, .. } => {
                body.limit_fractions(&state.q).map_err(|e| e.to_string())
            }
        }
    }

    /// Position in the external order the replay and the pose dump expect.
    pub fn q_external(&self) -> Result<Vec<f64>, String> {
        match self {
            Self::Corrected { world, state, .. } => {
                world.q_external(&state.q).map_err(|e| e.to_string())
            }
            Self::Legacy { state, .. } => Ok(state.q.clone()),
        }
    }

    /// Velocity in external order.
    ///
    /// On the corrected path the *order* is converted and the meaning is not;
    /// a `Free` joint's velocity is still world-frame linear then body-frame
    /// angular. The replay records velocities for inspection rather than
    /// re-simulating from them, so the difference is reported rather than
    /// converted.
    pub fn qd_external(&self) -> Result<Vec<f64>, String> {
        match self {
            Self::Corrected { world, state, .. } => {
                world.qd_external(&state.qd).map_err(|e| e.to_string())
            }
            Self::Legacy { state, .. } => Ok(state.qd.clone()),
        }
    }

    /// Whether every coordinate is finite.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        match self {
            Self::Corrected { state, .. } => state.q.iter().chain(&state.qd).all(|v| v.is_finite()),
            Self::Legacy { state, .. } => state.q.iter().chain(&state.qd).all(|v| v.is_finite()),
        }
    }

    /// Largest absolute position coordinate, for the fixed-base report.
    #[must_use]
    pub fn max_abs_position(&self) -> f64 {
        let q = match self {
            Self::Corrected { state, .. } => &state.q,
            Self::Legacy { state, .. } => &state.q,
        };
        q.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
    }
}

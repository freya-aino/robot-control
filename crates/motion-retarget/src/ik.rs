//! Weighted task-space inverse kinematics via damped least squares.
//!
//! SPECS.md Section 8.3: "Implement weighted task-space IK followed by
//! temporal refinement... Qualify a Rust-native constrained solver before
//! selecting differential-QP IK or whole-body optimization." No QP solver
//! (Appendix B.2.8: Clarabel and the rest) has been qualified in this
//! revision, per `docs/decisions/0003-std-only-default-closure.md`; this is
//! the "feasible non-QP baseline" Appendix C's C-09 explicitly keeps open as
//! an acceptable interim position.
//!
//! The Jacobian is computed by central finite differences over
//! `robot_model::kinematics::forward_kinematics`, not derived analytically.
//! That is a legitimate choice for a first baseline — it needs no per-joint
//! derivative bookkeeping — and it is validated the same way every other
//! numerical routine in this workspace is: this module's own tests compare
//! the solver's result against an independently computable target, not
//! against itself.
//!
//! Scope: end-effector **position** targets only. SPECS.md Section 8.3 also
//! asks for orientation targets and posture priors; neither is implemented
//! here, stated rather than silently assumed.

use core::fmt;
use robot_model::kinematics::{forward_kinematics, KinematicsError};
use robot_model::profile::RobotProfile;
use robot_types::frames::Vec3;
use robot_types::math::{LinalgError, Matrix};

/// One end-effector position target.
#[derive(Clone, Copy, Debug)]
pub struct PositionTarget {
    /// Index into `profile.links` (not a [`robot_types::ids::LinkId`] —
    /// resolved once by the caller, since this solver runs many FK
    /// evaluations per call and a linear ID lookup each time would be
    /// wasted work).
    pub link_index: usize,
    /// Desired position, in the profile's root-relative frame (see
    /// `robot_model::kinematics`'s own note on what "root frame" means).
    pub target: Vec3,
    /// Relative weight of this target in the combined error.
    pub weight: f64,
}

/// Solver configuration.
#[derive(Clone, Copy, Debug)]
pub struct IkConfig {
    /// Maximum Gauss-Newton iterations.
    pub max_iterations: usize,
    /// Convergence tolerance on the weighted error norm.
    pub tolerance: f64,
    /// Damped-least-squares regularization.
    pub damping: f64,
    /// Finite-difference step for the numerical Jacobian.
    pub finite_difference_step: f64,
    /// Maximum per-iteration joint-value change (radians), to keep a step
    /// near a singularity from being physically implausible even though
    /// the damped solve already bounds it algebraically.
    pub max_step: f64,
}

impl Default for IkConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            tolerance: 1e-6,
            damping: 1e-3,
            finite_difference_step: 1e-6,
            max_step: 0.2,
        }
    }
}

/// The result of one solve.
#[derive(Clone, Debug)]
pub struct IkSolution {
    /// Final joint values, in `profile.actuated_joints()` order.
    pub joint_values: Vec<f64>,
    /// Iterations actually performed.
    pub iterations: usize,
    /// Weighted error norm at the final iterate.
    pub final_error: f64,
    /// Whether `tolerance` was reached before `max_iterations`.
    pub converged: bool,
}

/// Why a solve could not run.
#[derive(Clone, PartialEq, Debug)]
pub enum IkError {
    /// The initial joint-value vector had the wrong width.
    WidthMismatch {
        /// Expected width.
        expected: usize,
        /// Width supplied.
        found: usize,
    },
    /// Forward kinematics failed during the solve.
    Kinematics(KinematicsError),
    /// The damped least-squares step failed.
    Linalg(LinalgError),
}

impl fmt::Display for IkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WidthMismatch { expected, found } => {
                write!(
                    f,
                    "initial joint vector has width {found}, profile expects {expected}"
                )
            }
            Self::Kinematics(e) => write!(f, "{e}"),
            Self::Linalg(e) => write!(f, "{e}"),
        }
    }
}

impl core::error::Error for IkError {}

impl From<KinematicsError> for IkError {
    fn from(e: KinematicsError) -> Self {
        Self::Kinematics(e)
    }
}
impl From<LinalgError> for IkError {
    fn from(e: LinalgError) -> Self {
        Self::Linalg(e)
    }
}

/// Solves for joint values placing every target's link at its desired
/// position, starting from `initial`.
pub fn solve(
    profile: &RobotProfile,
    initial: &[f64],
    targets: &[PositionTarget],
    config: &IkConfig,
) -> Result<IkSolution, IkError> {
    let dof = profile.dof();
    if initial.len() != dof {
        return Err(IkError::WidthMismatch {
            expected: dof,
            found: initial.len(),
        });
    }

    let mut joints = initial.to_vec();
    let mut error_vec = error_vector(profile, &joints, targets)?;
    let mut error_norm = error_vec.norm();
    let mut iterations = 0;

    while iterations < config.max_iterations && error_norm > config.tolerance {
        let jacobian =
            numerical_jacobian(profile, &joints, targets, config.finite_difference_step)?;
        let weights: Vec<f64> = targets
            .iter()
            .flat_map(|t| [t.weight, t.weight, t.weight])
            .collect();
        // `jacobian` is d(error)/d(joint); a Newton step toward zero error
        // needs `jacobian * step ≈ -error_vec`, not `≈ error_vec`. An
        // earlier version of this solver passed `error_vec` directly, which
        // solves for the step that *increases* error fastest — the
        // regression test that caught this is this module's own
        // `converges_to_a_reachable_target_within_tolerance`.
        let step = Matrix::damped_least_squares(
            &jacobian,
            &error_vec.scaled(-1.0),
            config.damping,
            Some(&weights),
        )?;

        for (j, s) in joints.iter_mut().zip(step.as_slice()) {
            *j += s.clamp(-config.max_step, config.max_step);
        }
        clamp_to_limits(profile, &mut joints);

        error_vec = error_vector(profile, &joints, targets)?;
        error_norm = error_vec.norm();
        iterations += 1;
    }

    Ok(IkSolution {
        joint_values: joints,
        iterations,
        final_error: error_norm,
        converged: error_norm <= config.tolerance,
    })
}

fn error_vector(
    profile: &RobotProfile,
    joints: &[f64],
    targets: &[PositionTarget],
) -> Result<Matrix, IkError> {
    let poses = forward_kinematics(profile, joints)?;
    let mut rows = Vec::with_capacity(targets.len() * 3);
    for t in targets {
        let pose = poses
            .by_index(t.link_index)
            .expect("caller-supplied valid link index");
        let current = pose.translation;
        rows.push(t.target.x - current.x);
        rows.push(t.target.y - current.y);
        rows.push(t.target.z - current.z);
    }
    Ok(Matrix::column(&rows))
}

fn numerical_jacobian(
    profile: &RobotProfile,
    joints: &[f64],
    targets: &[PositionTarget],
    step: f64,
) -> Result<Matrix, IkError> {
    let rows = targets.len() * 3;
    let cols = joints.len();
    let mut jac = Matrix::zeros(rows, cols);
    for c in 0..cols {
        let mut plus = joints.to_vec();
        plus[c] += step;
        let mut minus = joints.to_vec();
        minus[c] -= step;
        let e_plus = error_vector(profile, &plus, targets)?;
        let e_minus = error_vector(profile, &minus, targets)?;
        for r in 0..rows {
            // d(error)/d(joint) = -d(position)/d(joint); error_vector already
            // returns (target - current), so this finite difference of the
            // error directly gives the Jacobian the damped solve expects.
            jac[(r, c)] = (e_plus[(r, 0)] - e_minus[(r, 0)]) / (2.0 * step);
        }
    }
    Ok(jac)
}

fn clamp_to_limits(profile: &RobotProfile, joints: &mut [f64]) {
    for (j, joint_spec) in joints.iter_mut().zip(profile.actuated_joints()) {
        if let Some(range) = joint_spec.limits.position {
            *j = range.clamp(robot_types::units::Radians(*j)).value();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_model::actuator::{
        ActuatorSpec, PdGains, ThermalLimits, TorqueSpeedEnvelope, Transmission,
    };
    use robot_model::joint::{JointKind, JointLimits, JointSpec};
    use robot_model::link::{LinkSpec, MassProperties};
    use robot_model::pose::RigidPose;
    use robot_model::profile::{FirmwareContract, PROFILE_SCHEMA_VERSION};
    use robot_types::frames::{FrameId, GravityConvention};
    use robot_types::ids::{ActuatorId, JointId, LinkId, Name};
    use robot_types::schema::ExtensionTable;
    use robot_types::time::Duration;
    use robot_types::units::{
        Amperes, Celsius, Kilograms, NewtonMetres, Radians, RadiansPerSecond, Range,
    };

    fn link(id: u32, name: &str) -> LinkSpec {
        let mut mp = MassProperties::MASSLESS;
        mp.mass = Kilograms(1.0);
        LinkSpec {
            id: LinkId::new(id),
            name: Name::new(name).unwrap(),
            mass: mp,
            collision: Vec::new(),
        }
    }

    fn joint(id: u32, parent: u32, child: u32, offset_x: f64) -> JointSpec {
        JointSpec {
            id: JointId::new(id),
            name: Name::new(&format!("j{id}")).unwrap(),
            kind: JointKind::Revolute,
            parent: LinkId::new(parent),
            child: LinkId::new(child),
            origin: RigidPose::from_translation(Vec3::new(offset_x, 0.0, 0.0)),
            axis: Vec3::unit_z(),
            limits: JointLimits::bounded(
                Range::<Radians>::new(Radians(-3.0), Radians(3.0)).unwrap(),
                RadiansPerSecond(10.0),
                NewtonMetres(50.0),
            ),
            sign: 1.0,
            zero_offset: 0.0,
        }
    }

    /// A genuine 2-DOF planar arm: shoulder rotates the whole arm, elbow
    /// rotates the forearm about its own pivot, and a fixed joint carries
    /// the hand a further unit length beyond the elbow — without that
    /// third link, the "lower" link's own position would depend on the
    /// shoulder angle alone (its origin *is* the elbow pivot), which was
    /// the actual cause of this test's first failure: the originally
    /// chosen target was not on the elbow's reachable circle at all.
    fn two_link_arm() -> RobotProfile {
        let actuator = |id: u32, joint_id: u32| ActuatorSpec {
            id: ActuatorId::new(id),
            name: Name::new(&format!("a{id}")).unwrap(),
            joint: JointId::new(joint_id),
            transmission: Transmission::DIRECT,
            envelope: TorqueSpeedEnvelope {
                stall_torque: NewtonMetres(1.0),
                corner_speed: RadiansPerSecond(1.0),
                no_load_speed: RadiansPerSecond(2.0),
            },
            gains: Some(PdGains { kp: 1.0, kd: 0.1 }),
            command_latency: Duration::from_millis(1),
            thermal: ThermalLimits {
                limit: Celsius(80.0),
                ambient: Celsius(25.0),
                time_constant: Duration::from_secs(10),
                current_at_limit: Amperes(5.0),
            },
            current_uncertainty: Amperes(0.01),
        };
        let hand_joint = JointSpec {
            id: JointId::new(2),
            name: Name::new("hand-fixed").unwrap(),
            kind: JointKind::Fixed,
            parent: LinkId::new(2),
            child: LinkId::new(3),
            origin: RigidPose::from_translation(Vec3::new(1.0, 0.0, 0.0)),
            axis: Vec3::unit_z(),
            limits: JointLimits::unbounded(RadiansPerSecond(0.0), NewtonMetres(0.0)),
            sign: 1.0,
            zero_offset: 0.0,
        };
        RobotProfile {
            schema_version: PROFILE_SCHEMA_VERSION,
            name: Name::new("fixture-two-link-arm").unwrap(),
            profile_version: "1.0".to_owned(),
            joints: vec![joint(0, 0, 1, 0.0), joint(1, 1, 2, 1.0), hand_joint],
            links: vec![
                link(0, "base"),
                link(1, "upper"),
                link(2, "lower"),
                link(3, "hand"),
            ],
            actuators: vec![actuator(0, 0), actuator(1, 1)],
            sensors: Vec::new(),
            root_link: LinkId::new(0),
            gravity: GravityConvention::minus_z_standard(FrameId::WORLD),
            firmware: FirmwareContract {
                version: "1.0".to_owned(),
                command_expiration: Duration::from_millis(100),
                watchdog_timeout: Duration::from_millis(200),
                has_independent_estop: true,
                allowed_modes: Vec::new(),
            },
            extensions: ExtensionTable::new(),
        }
    }

    #[test]
    fn converges_to_a_reachable_target_within_tolerance() {
        let profile = two_link_arm();
        // The hand (link 3) is reachable out to radius 2.0 (two unit links);
        // target a point known reachable with shoulder=30deg, elbow=60deg:
        // hand = Rz(30).rotate((1,0,0)) + Rz(30+60).rotate((1,0,0)).
        let shoulder = 30f64.to_radians();
        let elbow = 60f64.to_radians();
        let target_point = Vec3::new(shoulder.cos(), shoulder.sin(), 0.0)
            + Vec3::new((shoulder + elbow).cos(), (shoulder + elbow).sin(), 0.0);
        let target = PositionTarget {
            link_index: 3,
            target: target_point,
            weight: 1.0,
        };
        let solution = solve(&profile, &[0.0, 0.0], &[target], &IkConfig::default()).unwrap();
        assert!(
            solution.converged,
            "expected convergence, final error {}",
            solution.final_error
        );
        assert!(solution.final_error < 1e-4);

        // Verify independently via forward kinematics on the solved joints,
        // not by comparing against the (shoulder, elbow) angles used to
        // construct the target: a 2-DOF arm reaching a 2-D point has two
        // valid elbow-up/elbow-down solutions, and the solver is free to
        // find either.
        let poses = forward_kinematics(&profile, &solution.joint_values).unwrap();
        let achieved = poses.by_index(3).unwrap().translation;
        assert!((achieved - target.target).norm() < 1e-3);
    }

    #[test]
    fn an_unreachable_target_terminates_without_diverging() {
        let profile = two_link_arm();
        // Max reach is 2.0; this target is far outside it.
        let target = PositionTarget {
            link_index: 3,
            target: Vec3::new(10.0, 0.0, 0.0),
            weight: 1.0,
        };
        let solution = solve(&profile, &[0.0, 0.0], &[target], &IkConfig::default()).unwrap();
        assert!(!solution.converged);
        assert!(solution.joint_values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn joint_limits_are_respected_throughout() {
        let profile = two_link_arm();
        let target = PositionTarget {
            link_index: 3,
            target: Vec3::new(10.0, 10.0, 0.0),
            weight: 1.0,
        };
        let solution = solve(&profile, &[0.0, 0.0], &[target], &IkConfig::default()).unwrap();
        for (v, joint) in solution.joint_values.iter().zip(profile.actuated_joints()) {
            let range = joint.limits.position.unwrap();
            assert!(range.contains(Radians(*v)), "{v} outside {range:?}");
        }
    }

    #[test]
    fn a_mismatched_initial_width_is_rejected() {
        let profile = two_link_arm();
        let target = PositionTarget {
            link_index: 3,
            target: Vec3::ZERO,
            weight: 1.0,
        };
        assert!(matches!(
            solve(&profile, &[0.0], &[target], &IkConfig::default()),
            Err(IkError::WidthMismatch {
                expected: 2,
                found: 1
            })
        ));
    }
}

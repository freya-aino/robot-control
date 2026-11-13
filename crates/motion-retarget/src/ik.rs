//! Weighted multi-task inverse kinematics, per SPECS.md Section 8.3:
//! "weighted task-space IK ... Support end-effector positions/orientations,
//! posture priors, planted-foot constraints, joint limits and self-collision
//! penalties."
//!
//! The problem is a weighted nonlinear least-squares over the actuated
//! joints and, optionally, the six coordinates of a free base. Each
//! [`Task`] contributes residual rows scaled by the square root of its
//! weight; [`crate::collision::SelfCollision`] contributes one row per
//! link pair, nonzero only while the pair is closer than its margin. Joint
//! limits are hard: every iterate is projected back into the limits, so no
//! returned solution violates them.
//!
//! The method is Levenberg-Marquardt over a central finite-difference
//! Jacobian: a damped least-squares step, accepted only if it lowers the
//! cost, with the damping relaxed after an accepted step and raised after a
//! rejected one. SPECS.md Section 8.3: "Qualify a Rust-native constrained
//! solver before selecting differential-QP IK." No QP solver (Appendix
//! B.2.8) has been qualified (`docs/decisions/0003-std-only-default-closure.md`);
//! this is the non-QP baseline Appendix C's C-09 permits, with limits by
//! projection and collision as a penalty rather than as constraints.
//!
//! Section 8.3 also requires "bounded iterations/deadlines and a validated
//! response to infeasibility or timeout". Every solve is bounded by
//! `max_iterations` and an optional wall-clock `deadline`, and reports why
//! it stopped as a [`Termination`]; [`Termination::Stalled`] is the honest
//! outcome for conflicting tasks (a local minimum with nonzero cost), and a
//! caller decides from the per-task residuals whether it is acceptable.

use core::fmt;
use robot_model::kinematics::{forward_kinematics, KinematicsError, LinkPoses};
use robot_model::pose::RigidPose;
use robot_model::profile::RobotProfile;
use robot_types::frames::{Quat, Vec3};
use robot_types::math::{LinalgError, Matrix};

use crate::collision::SelfCollision;

/// One end-effector position target (kept as the simplest entry point).
#[derive(Clone, Copy, Debug)]
pub struct PositionTarget {
    /// Index into `profile.links` (not a [`robot_types::ids::LinkId`]:
    /// resolved once by the caller, since this solver runs many FK
    /// evaluations per call).
    pub link_index: usize,
    /// Desired position in the world frame (the profile's root frame when
    /// the base is fixed at the identity).
    pub target: Vec3,
    /// Relative weight of this target in the combined error.
    pub weight: f64,
}

/// One weighted objective.
#[derive(Clone, PartialEq, Debug)]
pub enum Task {
    /// A link origin at a world position (3 rows).
    Position {
        /// Index into `profile.links`.
        link: usize,
        /// World position.
        target: Vec3,
        /// Weight.
        weight: f64,
    },
    /// A link's world orientation (3 rows: the rotation vector of the
    /// error rotation, so the residual is in radians).
    Orientation {
        /// Index into `profile.links`.
        link: usize,
        /// World orientation.
        target: Quat,
        /// Weight.
        weight: f64,
    },
    /// A link's position relative to another link's (3 rows). This is the
    /// task that preserves intent: a punch is the hand's displacement from
    /// the shoulder, not the hand's absolute position.
    RelativePosition {
        /// Index into `profile.links` of the moving link.
        link: usize,
        /// Index into `profile.links` of the reference link.
        base: usize,
        /// Desired `link - base`, in world axes.
        target: Vec3,
        /// Weight.
        weight: f64,
    },
    /// Joint values near a nominal posture (one row per actuated joint).
    Posture {
        /// Nominal values in `profile.actuated_joints()` order.
        nominal: Vec<f64>,
        /// Weight per joint, same order.
        weights: Vec<f64>,
    },
}

impl From<PositionTarget> for Task {
    fn from(t: PositionTarget) -> Self {
        Self::Position {
            link: t.link_index,
            target: t.target,
            weight: t.weight,
        }
    }
}

impl Task {
    fn rows(&self) -> usize {
        match self {
            Self::Position { .. } | Self::Orientation { .. } | Self::RelativePosition { .. } => 3,
            Self::Posture { nominal, .. } => nominal.len(),
        }
    }
}

/// How the robot's base is placed.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Base {
    /// The base pose (world from root link) the solve starts at.
    pub pose: RigidPose,
    /// `Some` lets the solver move the base, penalizing its translation
    /// (metres) and rotation (radians) away from `pose` by these weights.
    pub free: Option<BasePrior>,
}

impl Base {
    /// The root link fixed at the world origin.
    pub const FIXED_AT_ORIGIN: Self = Self {
        pose: RigidPose::IDENTITY,
        free: None,
    };
}

/// Weights holding a free base near its starting pose.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct BasePrior {
    /// Weight on translation away from the starting pose.
    pub translation_weight: f64,
    /// Weight on rotation away from the starting pose.
    pub rotation_weight: f64,
}

/// Solver configuration.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct IkConfig {
    /// Maximum Levenberg-Marquardt iterations.
    pub max_iterations: usize,
    /// Convergence tolerance on the weighted residual norm.
    pub tolerance: f64,
    /// Initial damping (the `lambda` of the damped least-squares step).
    pub damping: f64,
    /// Finite-difference step for the numerical Jacobian.
    pub finite_difference_step: f64,
    /// Maximum per-iteration change of any variable (radians or metres).
    pub max_step: f64,
    /// An accepted step improving the cost by less than this fraction ends
    /// the solve as [`Termination::Stalled`].
    pub stall_fraction: f64,
    /// Wall-clock budget for one solve.
    pub deadline: Option<std::time::Duration>,
}

impl Default for IkConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            tolerance: 1e-6,
            damping: 1e-3,
            finite_difference_step: 1e-6,
            max_step: 0.2,
            stall_fraction: 1e-9,
            deadline: None,
        }
    }
}

/// Why a solve stopped.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Termination {
    /// The residual norm reached the tolerance.
    Converged,
    /// No step lowered the cost meaningfully: a local minimum, which for
    /// conflicting tasks is the expected answer.
    Stalled,
    /// `max_iterations` ran out while the cost was still falling.
    MaxIterations,
    /// The wall-clock deadline passed.
    Deadline,
}

/// The result of one solve.
#[derive(Clone, Debug)]
pub struct IkSolution {
    /// Final joint values, in `profile.actuated_joints()` order.
    pub joint_values: Vec<f64>,
    /// Final base pose (the starting pose when the base is fixed).
    pub base: RigidPose,
    /// Iterations actually performed.
    pub iterations: usize,
    /// Weighted residual norm at the final iterate.
    pub final_error: f64,
    /// Whether `tolerance` was reached.
    pub converged: bool,
    /// Why the solve stopped.
    pub termination: Termination,
    /// Unweighted residual norm per task, in task order.
    pub task_errors: Vec<f64>,
    /// Largest self-collision penetration of the margin, metres (zero when
    /// no collision model was given).
    pub max_penetration: f64,
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
    /// A task named a link that does not exist, or a posture of the wrong
    /// width.
    InvalidTask {
        /// Index of the task.
        task: usize,
    },
    /// A weight was negative or non-finite.
    InvalidWeight,
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
            Self::InvalidTask { task } => write!(f, "task {task} is invalid for this profile"),
            Self::InvalidWeight => f.write_str("weights must be finite and non-negative"),
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
/// position, starting from `initial`, with the base fixed at the origin.
pub fn solve(
    profile: &RobotProfile,
    initial: &[f64],
    targets: &[PositionTarget],
    config: &IkConfig,
) -> Result<IkSolution, IkError> {
    let tasks: Vec<Task> = targets.iter().map(|&t| t.into()).collect();
    solve_tasks(
        profile,
        initial,
        Base::FIXED_AT_ORIGIN,
        &tasks,
        None,
        config,
    )
}

/// The full problem: `tasks`, an optional self-collision penalty, and a
/// fixed or free base.
pub fn solve_tasks(
    profile: &RobotProfile,
    initial: &[f64],
    base: Base,
    tasks: &[Task],
    collision: Option<&SelfCollision>,
    config: &IkConfig,
) -> Result<IkSolution, IkError> {
    let started = std::time::Instant::now();
    let dof = profile.dof();
    if initial.len() != dof {
        return Err(IkError::WidthMismatch {
            expected: dof,
            found: initial.len(),
        });
    }
    validate_tasks(profile, tasks)?;
    if let Some(prior) = base.free {
        for w in [prior.translation_weight, prior.rotation_weight] {
            if !w.is_finite() || w < 0.0 {
                return Err(IkError::InvalidWeight);
            }
        }
    }
    let problem = Problem {
        profile,
        base,
        tasks,
        collision,
        limits: profile
            .actuated_joints()
            .map(|j| j.limits.position.map(|r| (r.lo.value(), r.hi.value())))
            .collect(),
    };

    let mut x = vec![0.0; problem.base_vars()];
    x.extend_from_slice(initial);
    problem.project(&mut x);
    let mut r = problem.residual(&x)?;
    let mut cost = r.norm();
    let mut lambda = config.damping.max(1e-12);
    let mut iterations = 0;
    let termination = loop {
        if cost <= config.tolerance {
            break Termination::Converged;
        }
        if iterations >= config.max_iterations {
            break Termination::MaxIterations;
        }
        if config.deadline.is_some_and(|d| started.elapsed() >= d) {
            break Termination::Deadline;
        }
        iterations += 1;
        let jac = problem.jacobian(&x, config.finite_difference_step)?;
        let rhs = r.scaled(-1.0);
        let mut accepted = None;
        for _ in 0..10 {
            let step = Matrix::damped_least_squares(&jac, &rhs, lambda, None)?;
            let mut trial = x.clone();
            for (v, s) in trial.iter_mut().zip(step.as_slice()) {
                *v += s.clamp(-config.max_step, config.max_step);
            }
            problem.project(&mut trial);
            let tr = problem.residual(&trial)?;
            let tc = tr.norm();
            if tc < cost {
                accepted = Some((trial, tr, tc));
                lambda = (lambda / 3.0).max(1e-12);
                break;
            }
            lambda *= 4.0;
        }
        let Some((nx, nr, nc)) = accepted else {
            break Termination::Stalled;
        };
        let improvement = (cost * cost - nc * nc) / (cost * cost);
        x = nx;
        r = nr;
        cost = nc;
        if improvement < config.stall_fraction && cost > config.tolerance {
            break Termination::Stalled;
        }
    };

    let (base_pose, poses) = problem.evaluate(&x)?;
    Ok(IkSolution {
        joint_values: x[problem.base_vars()..].to_vec(),
        base: base_pose,
        iterations,
        final_error: cost,
        converged: termination == Termination::Converged,
        termination,
        task_errors: tasks
            .iter()
            .map(|t| {
                let mut rows = Vec::new();
                problem.task_rows(
                    t,
                    &base_pose,
                    &poses,
                    &x[problem.base_vars()..],
                    1.0,
                    &mut rows,
                );
                rows.iter().map(|v| v * v).sum::<f64>().sqrt()
            })
            .collect(),
        max_penetration: collision.map_or(0.0, |c| {
            c.penetrations(&base_pose, &poses)
                .into_iter()
                .fold(0.0, f64::max)
        }),
    })
}

fn validate_tasks(profile: &RobotProfile, tasks: &[Task]) -> Result<(), IkError> {
    let links = profile.links.len();
    let dof = profile.dof();
    let ok_w = |w: f64| w.is_finite() && w >= 0.0;
    for (i, t) in tasks.iter().enumerate() {
        let valid = match t {
            Task::Position { link, weight, .. } | Task::Orientation { link, weight, .. } => {
                if !ok_w(*weight) {
                    return Err(IkError::InvalidWeight);
                }
                *link < links
            }
            Task::RelativePosition {
                link, base, weight, ..
            } => {
                if !ok_w(*weight) {
                    return Err(IkError::InvalidWeight);
                }
                *link < links && *base < links
            }
            Task::Posture { nominal, weights } => {
                if !weights.iter().all(|w| ok_w(*w)) {
                    return Err(IkError::InvalidWeight);
                }
                nominal.len() == dof && weights.len() == dof
            }
        };
        if !valid {
            return Err(IkError::InvalidTask { task: i });
        }
    }
    Ok(())
}

struct Problem<'a> {
    profile: &'a RobotProfile,
    base: Base,
    tasks: &'a [Task],
    collision: Option<&'a SelfCollision>,
    limits: Vec<Option<(f64, f64)>>,
}

impl Problem<'_> {
    fn base_vars(&self) -> usize {
        if self.base.free.is_some() {
            6
        } else {
            0
        }
    }

    fn project(&self, x: &mut [f64]) {
        let nb = self.base_vars();
        for (v, lim) in x[nb..].iter_mut().zip(&self.limits) {
            if let Some((lo, hi)) = lim {
                *v = v.clamp(*lo, *hi);
            }
        }
    }

    fn evaluate(&self, x: &[f64]) -> Result<(RigidPose, LinkPoses), IkError> {
        let nb = self.base_vars();
        let base = if nb == 6 {
            let dt = Vec3::new(x[0], x[1], x[2]);
            let dr = Quat::from_rotation_vector(Vec3::new(x[3], x[4], x[5]));
            RigidPose {
                translation: self.base.pose.translation + dt,
                rotation: (dr * self.base.pose.rotation).normalized(),
            }
        } else {
            self.base.pose
        };
        Ok((base, forward_kinematics(self.profile, &x[nb..])?))
    }

    fn residual(&self, x: &[f64]) -> Result<Matrix, IkError> {
        let (base, poses) = self.evaluate(x)?;
        let nb = self.base_vars();
        let mut rows = Vec::with_capacity(self.tasks.iter().map(Task::rows).sum::<usize>() + 6);
        if let Some(prior) = self.base.free {
            let (st, sr) = (
                prior.translation_weight.sqrt(),
                prior.rotation_weight.sqrt(),
            );
            rows.extend([
                x[0] * st,
                x[1] * st,
                x[2] * st,
                x[3] * sr,
                x[4] * sr,
                x[5] * sr,
            ]);
        }
        for t in self.tasks {
            self.task_rows(t, &base, &poses, &x[nb..], f64::NAN, &mut rows);
        }
        if let Some(c) = self.collision {
            let s = c.weight.sqrt();
            rows.extend(c.penetrations(&base, &poses).into_iter().map(|p| p * s));
        }
        Ok(Matrix::column(&rows))
    }

    /// Appends `t`'s residual rows. `scale` overrides the task weight's
    /// square root when finite (1.0 gives unweighted rows for reporting).
    fn task_rows(
        &self,
        t: &Task,
        base: &RigidPose,
        poses: &LinkPoses,
        q: &[f64],
        scale: f64,
        rows: &mut Vec<f64>,
    ) {
        let s = |w: f64| if scale.is_finite() { scale } else { w.sqrt() };
        let world = |link: usize| base.compose(&poses.by_index(link).expect("validated link"));
        match t {
            Task::Position {
                link,
                target,
                weight,
            } => {
                let e = world(*link).translation - *target;
                rows.extend(e.to_xyz().map(|v| v * s(*weight)));
            }
            Task::Orientation {
                link,
                target,
                weight,
            } => {
                let current = world(*link).rotation;
                let e = (current * target.conjugate())
                    .normalized()
                    .to_rotation_vector();
                rows.extend(e.to_xyz().map(|v| v * s(*weight)));
            }
            Task::RelativePosition {
                link,
                base: b,
                target,
                weight,
            } => {
                let e = world(*link).translation - world(*b).translation - *target;
                rows.extend(e.to_xyz().map(|v| v * s(*weight)));
            }
            Task::Posture { nominal, weights } => {
                // A zero-weight joint is not part of the task, so it does not
                // count toward the reported (unweighted) error either.
                for ((v, n), w) in q.iter().zip(nominal).zip(weights) {
                    rows.push(if *w > 0.0 { (v - n) * s(*w) } else { 0.0 });
                }
            }
        }
    }

    fn jacobian(&self, x: &[f64], h: f64) -> Result<Matrix, IkError> {
        let base = self.residual(x)?;
        let mut jac = Matrix::zeros(base.rows(), x.len());
        let mut probe = x.to_vec();
        for c in 0..x.len() {
            let v = x[c];
            probe[c] = v + h;
            let plus = self.residual(&probe)?;
            probe[c] = v - h;
            let minus = self.residual(&probe)?;
            probe[c] = v;
            for r in 0..base.rows() {
                jac[(r, c)] = (plus[(r, 0)] - minus[(r, 0)]) / (2.0 * h);
            }
        }
        Ok(jac)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::two_link_arm;
    use robot_types::units::Radians;

    #[test]
    fn an_orientation_target_is_reached() {
        let profile = two_link_arm();
        // The hand's orientation is the sum of both joint angles about z.
        let target = Quat::from_rz(Radians(1.2)).unwrap();
        let tasks = [Task::Orientation {
            link: 3,
            target,
            weight: 1.0,
        }];
        let s = solve_tasks(
            &profile,
            &[0.0, 0.0],
            Base::FIXED_AT_ORIGIN,
            &tasks,
            None,
            &IkConfig::default(),
        )
        .unwrap();
        assert_eq!(s.termination, Termination::Converged);
        assert!((s.joint_values[0] + s.joint_values[1] - 1.2).abs() < 1e-5);
    }

    #[test]
    fn a_posture_prior_picks_among_redundant_solutions() {
        // Orientation fixes only the sum of the two angles; the prior on the
        // shoulder pins the split.
        let profile = two_link_arm();
        let tasks = [
            Task::Orientation {
                link: 3,
                target: Quat::from_rz(Radians(1.0)).unwrap(),
                weight: 1.0,
            },
            Task::Posture {
                nominal: vec![0.25, 0.0],
                weights: vec![1e-2, 0.0],
            },
        ];
        let s = solve_tasks(
            &profile,
            &[0.0, 0.0],
            Base::FIXED_AT_ORIGIN,
            &tasks,
            None,
            &IkConfig::default(),
        )
        .unwrap();
        assert!(
            (s.joint_values[0] - 0.25).abs() < 1e-4,
            "{:?}",
            s.joint_values
        );
        assert!(
            (s.joint_values[1] - 0.75).abs() < 1e-4,
            "{:?}",
            s.joint_values
        );
        assert!(s.task_errors[1] < 1e-4);
    }

    #[test]
    fn a_relative_target_moves_with_the_base_link() {
        // Hand relative to the upper link's origin (the shoulder pivot):
        // fixing it at (1, 1, 0) is the elbow at 90 degrees with the
        // shoulder at 0 or the mirror solution.
        let profile = two_link_arm();
        let tasks = [Task::RelativePosition {
            link: 3,
            base: 1,
            target: Vec3::new(1.0, 1.0, 0.0),
            weight: 1.0,
        }];
        let s = solve_tasks(
            &profile,
            &[0.1, 0.5],
            Base::FIXED_AT_ORIGIN,
            &tasks,
            None,
            &IkConfig::default(),
        )
        .unwrap();
        assert_eq!(s.termination, Termination::Converged);
        let poses = forward_kinematics(&profile, &s.joint_values).unwrap();
        let d = poses.by_index(3).unwrap().translation - poses.by_index(1).unwrap().translation;
        assert!((d - Vec3::new(1.0, 1.0, 0.0)).norm() < 1e-4);
    }

    #[test]
    fn a_free_base_translates_to_reach_a_far_target() {
        let profile = two_link_arm();
        let tasks = [Task::Position {
            link: 3,
            target: Vec3::new(5.0, 0.0, 0.0),
            weight: 1.0,
        }];
        let base = Base {
            pose: RigidPose::IDENTITY,
            free: Some(BasePrior {
                translation_weight: 1e-6,
                rotation_weight: 1e-6,
            }),
        };
        let s = solve_tasks(
            &profile,
            &[0.0, 0.0],
            base,
            &tasks,
            None,
            &IkConfig {
                max_iterations: 200,
                max_step: 1.0,
                tolerance: 1e-3,
                ..IkConfig::default()
            },
        )
        .unwrap();
        assert!(
            s.task_errors[0] < 1e-3,
            "{:?} {:?}",
            s.task_errors,
            s.termination
        );
        assert!(s.base.translation.x > 2.9);
    }

    #[test]
    fn a_zero_deadline_stops_immediately_and_says_so() {
        let profile = two_link_arm();
        let tasks = [Task::Position {
            link: 3,
            target: Vec3::new(1.0, 1.0, 0.0),
            weight: 1.0,
        }];
        let config = IkConfig {
            deadline: Some(std::time::Duration::ZERO),
            ..IkConfig::default()
        };
        let s = solve_tasks(
            &profile,
            &[0.0, 0.0],
            Base::FIXED_AT_ORIGIN,
            &tasks,
            None,
            &config,
        )
        .unwrap();
        assert_eq!(s.termination, Termination::Deadline);
        assert_eq!(s.iterations, 0);
    }

    #[test]
    fn invalid_tasks_are_rejected() {
        let profile = two_link_arm();
        let bad_link = [Task::Position {
            link: 99,
            target: Vec3::ZERO,
            weight: 1.0,
        }];
        assert_eq!(
            solve_tasks(
                &profile,
                &[0.0, 0.0],
                Base::FIXED_AT_ORIGIN,
                &bad_link,
                None,
                &IkConfig::default()
            )
            .unwrap_err(),
            IkError::InvalidTask { task: 0 }
        );
        let bad_weight = [Task::Position {
            link: 3,
            target: Vec3::ZERO,
            weight: -1.0,
        }];
        assert_eq!(
            solve_tasks(
                &profile,
                &[0.0, 0.0],
                Base::FIXED_AT_ORIGIN,
                &bad_weight,
                None,
                &IkConfig::default()
            )
            .unwrap_err(),
            IkError::InvalidWeight
        );
    }

    #[test]
    fn an_unreachable_target_stalls_rather_than_running_out() {
        let profile = two_link_arm();
        let target = PositionTarget {
            link_index: 3,
            target: Vec3::new(10.0, 0.0, 0.0),
            weight: 1.0,
        };
        let s = solve(
            &profile,
            &[0.3, 0.3],
            &[target],
            &IkConfig {
                max_iterations: 500,
                ..IkConfig::default()
            },
        )
        .unwrap();
        assert_eq!(s.termination, Termination::Stalled);
        assert!((s.final_error - 8.0).abs() < 1e-3, "{}", s.final_error);
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

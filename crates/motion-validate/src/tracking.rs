//! Simulated tracking feasibility, per SPECS.md Section 8.2 item 8
//! ("simulated tracking feasibility") and Section 8.3 ("downstream tracking
//! success per skill and embodiment").
//!
//! The reference is played into `robot-sim`'s [`World`]: the robot starts at
//! the reference's first frame, and at every simulator step each joint is
//! driven toward the reference linearly interpolated to that instant, by
//! the model's declared drives when it has them and `use_declared_drives`
//! is set, and otherwise by a joint PD law plus an optional feedforward
//! torque (the inverse-dynamics estimate of [`crate::dynamics`],
//! interpolated).
//!
//! The PD law is written in acceleration units and scaled by each joint's
//! effective inertia `1 / (M^-1)_jj` at the frame's configuration:
//! `tau_j = m_j (w^2 (q_ref - q) + 2 zeta w (qd_ref - qd))`. Fixed gains in
//! N m/rad are unstable under an explicit step on a light wrist and
//! needlessly soft on a hip; a bandwidth `w` and damping ratio `zeta` mean
//! the same on every joint and stay stable while `2 zeta w dt` is small. With `floating` the base is free and stands on a ground
//! plane; without, the root link is bolted to the world.
//!
//! This measures one controller on one model, which is what "tracking
//! feasibility" can mean before a learned tracker exists: a reference this
//! controller tracks is not thereby feasible for every controller, and one
//! it fails is a reference to look at, not proof that no policy could.
//! Results are per frame and per task label, so a curation step can accept
//! the jab and reject the hook in the same clip.

use core::fmt;
use robot_model::profile::RobotProfile;
use robot_model::usd::with_floating_base;
use robot_sim::articulated::ArticulatedBody;
use robot_sim::options::SimOptions;
use robot_sim::world::{ActuationMode, Ground, State, World};
use robot_types::motion::{MotionRecord, RobotReference};

use crate::dynamics::FrameDynamics;
use crate::trajectory::{free_joint_q, Trajectory};

/// How the reference is tracked.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct TrackingConfig {
    /// Simulator options (the timestep among them).
    pub options: SimOptions,
    /// Closed-loop bandwidth of the PD law, Hz.
    pub bandwidth_hz: f64,
    /// Damping ratio of the PD law.
    pub damping_ratio: f64,
    /// Use the model's declared drives when it has them.
    pub use_declared_drives: bool,
    /// Free base on a ground plane, or base bolted to the world.
    pub floating: bool,
    /// The ground, when floating.
    pub ground: Ground,
    /// The robot has fallen when its base height drops below this
    /// fraction of the reference's.
    pub fall_height_ratio: f64,
}

/// Tracking of one task-labelled span.
#[derive(Clone, PartialEq, Debug)]
pub struct SkillTracking {
    /// The label.
    pub label: String,
    /// First frame.
    pub start: usize,
    /// One past the last frame.
    pub end: usize,
    /// RMS joint error over the span, radians.
    pub joint_rms: f64,
    /// Worst base position error over the span, metres.
    pub max_base_error: f64,
    /// Whether the robot fell before the span ended.
    pub fell: bool,
}

/// The result of tracking a reference.
#[derive(Clone, PartialEq, Debug)]
pub struct TrackingResult {
    /// RMS joint error at each reached frame, radians.
    pub joint_rms: Vec<f64>,
    /// Base position error at each reached frame, metres.
    pub base_error: Vec<f64>,
    /// First frame at which the robot had fallen.
    pub fell_at: Option<usize>,
    /// Per task label.
    pub per_label: Vec<SkillTracking>,
    /// Simulator steps taken.
    pub steps: usize,
    /// Which controller ran: `"declared-drives"` or `"pd"` / `"pd+feedforward"`.
    pub controller: &'static str,
}

impl TrackingResult {
    /// RMS joint error over every reached frame.
    #[must_use]
    pub fn overall_joint_rms(&self) -> f64 {
        rms(&self.joint_rms)
    }
}

fn rms(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
    }
}

/// Why tracking could not run.
#[derive(Clone, PartialEq, Debug)]
pub enum TrackingError {
    /// The floating profile could not be built.
    Profile(String),
    /// The simulator refused a vector or failed a step.
    Sim(String),
    /// A gain or ratio was out of range.
    InvalidConfig(&'static str),
}

impl fmt::Display for TrackingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Profile(e) => write!(f, "profile: {e}"),
            Self::Sim(e) => write!(f, "simulation: {e}"),
            Self::InvalidConfig(what) => write!(f, "invalid tracking config: {what}"),
        }
    }
}

impl core::error::Error for TrackingError {}

fn sim<E: fmt::Display>(e: E) -> TrackingError {
    TrackingError::Sim(e.to_string())
}

/// Tracks `reference` (with its extracted `trajectory`) on `profile`.
pub fn track(
    profile: &RobotProfile,
    reference: &MotionRecord<RobotReference>,
    trajectory: &Trajectory,
    feedforward: Option<&[FrameDynamics]>,
    config: &TrackingConfig,
) -> Result<TrackingResult, TrackingError> {
    for (v, what) in [
        (config.bandwidth_hz, "bandwidth_hz"),
        (config.damping_ratio, "damping_ratio"),
    ] {
        if !v.is_finite() || v < 0.0 {
            return Err(TrackingError::InvalidConfig(what));
        }
    }
    let sim_profile = if config.floating {
        with_floating_base(profile.clone()).map_err(|e| TrackingError::Profile(e.to_string()))?
    } else {
        profile.clone()
    };
    let mut world = World::from_profile(&sim_profile, config.options);
    if config.floating {
        world = world.with_ground(config.ground);
    }
    let nb = if config.floating { 6 } else { 0 };
    let nqb = if config.floating { 7 } else { 0 };
    let dof = profile.dof();

    let external_q = |f: usize| {
        let mut q = if config.floating {
            free_joint_q(&trajectory.base[f]).to_vec()
        } else {
            Vec::new()
        };
        q.extend_from_slice(&trajectory.q[f]);
        q
    };
    let mut qd0 = vec![0.0; nb];
    if config.floating {
        // This world's free-joint velocity is world linear, then body angular.
        let t = trajectory.base_twist[0];
        let v = trajectory.base[0]
            .rotation
            .rotate(robot_types::frames::Vec3::new(t[3], t[4], t[5]));
        qd0 = vec![v.x, v.y, v.z, t[0], t[1], t[2]];
    }
    qd0.extend_from_slice(&trajectory.qd[0]);
    let mut state = State {
        q: world
            .model()
            .qpos_to_internal(&external_q(0))
            .map_err(sim)?,
        qd: world.model().dof_to_internal(&qd0).map_err(sim)?,
    };

    let drives = config.use_declared_drives && world.actuator_count() > 0;
    let actuator_joint: Vec<usize> = sim_profile
        .actuators
        .iter()
        .map(|a| {
            profile
                .actuated_joints()
                .position(|j| j.id == a.joint)
                .unwrap_or(0)
        })
        .collect();
    let controller = if drives {
        "declared-drives"
    } else if feedforward.is_some() {
        "pd+feedforward"
    } else {
        "pd"
    };

    let dt = config.options.timestep.as_secs_f64();
    let body = ArticulatedBody::from_profile(&sim_profile);
    let w = 2.0 * core::f64::consts::PI * config.bandwidth_hz;
    let (kp, kd) = (w * w, 2.0 * config.damping_ratio * w);
    let nv = world.nv();
    let mut tau = vec![0.0; nv];
    let mut ctrl = vec![0.0; world.actuator_count()];
    let mut result = TrackingResult {
        joint_rms: vec![0.0],
        base_error: vec![0.0],
        fell_at: None,
        per_label: Vec::new(),
        steps: 0,
        controller,
    };
    'frames: for f in 0..trajectory.len() - 1 {
        let span = trajectory.times[f + 1] - trajectory.times[f];
        let steps = ((span / dt).round() as usize).max(1);
        let inertia: Vec<f64> = if drives {
            Vec::new()
        } else {
            let q_ext = world.model().qpos_to_external(&state.q).map_err(sim)?;
            body.inverse_inertia_diagonal(&q_ext)
                .map_err(sim)?
                .into_iter()
                .map(|v| if v > 0.0 { 1.0 / v } else { 0.0 })
                .collect()
        };
        for k in 0..steps {
            let s = (k as f64 + 1.0) / steps as f64;
            let q_ref: Vec<f64> = trajectory.q[f]
                .iter()
                .zip(&trajectory.q[f + 1])
                .map(|(a, b)| a + (b - a) * s)
                .collect();
            if drives {
                for (c, &j) in ctrl.iter_mut().zip(&actuator_joint) {
                    *c = q_ref[j];
                }
                world
                    .actuator_torques(&state.q, &state.qd, &ctrl, ActuationMode::Drives, &mut tau)
                    .map_err(sim)?;
            } else {
                let q_ext = world.model().qpos_to_external(&state.q).map_err(sim)?;
                let qd_ext = world.model().dof_to_external(&state.qd).map_err(sim)?;
                let mut t_ext = vec![0.0; nv];
                for j in 0..dof {
                    let qd_ref = (trajectory.q[f + 1][j] - trajectory.q[f][j]) / span;
                    let ff = feedforward.map_or(0.0, |d| {
                        d[f].joint_torque[j] + (d[f + 1].joint_torque[j] - d[f].joint_torque[j]) * s
                    });
                    let m = inertia[nb + j];
                    t_ext[nb + j] = m
                        * (kp * (q_ref[j] - q_ext[nqb + j]) + kd * (qd_ref - qd_ext[nb + j]))
                        + ff;
                }
                tau = world.model().dof_to_internal(&t_ext).map_err(sim)?;
            }
            world.step(&mut state, &tau).map_err(sim)?;
            result.steps += 1;
        }
        let q_ext = world.model().qpos_to_external(&state.q).map_err(sim)?;
        if q_ext.iter().any(|v| !v.is_finite()) {
            return Err(TrackingError::Sim(format!(
                "state became non-finite at frame {}",
                f + 1
            )));
        }
        let err: Vec<f64> = (0..dof)
            .map(|j| q_ext[nqb + j] - trajectory.q[f + 1][j])
            .collect();
        result.joint_rms.push(rms(&err));
        let base_err = if config.floating {
            let p = robot_types::frames::Vec3::new(q_ext[0], q_ext[1], q_ext[2]);
            let r = trajectory.base[f + 1].translation;
            if p.z < config.fall_height_ratio * r.z && result.fell_at.is_none() {
                result.fell_at = Some(f + 1);
            }
            (p - r).norm()
        } else {
            0.0
        };
        result.base_error.push(base_err);
        if result.fell_at.is_some() {
            break 'frames;
        }
    }

    let reached = result.joint_rms.len();
    for label in &reference.task_labels {
        let end = label.end.min(reached);
        if label.start >= end {
            result.per_label.push(SkillTracking {
                label: label.label.to_string(),
                start: label.start,
                end: label.end,
                joint_rms: f64::INFINITY,
                max_base_error: f64::INFINITY,
                fell: true,
            });
            continue;
        }
        result.per_label.push(SkillTracking {
            label: label.label.to_string(),
            start: label.start,
            end: label.end,
            joint_rms: rms(&result.joint_rms[label.start..end]),
            max_base_error: result.base_error[label.start..end]
                .iter()
                .copied()
                .fold(0.0, f64::max),
            fell: result.fell_at.is_some_and(|at| at < label.end),
        });
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::estimate;
    use crate::test_support::{g1, reference_from};
    use robot_types::frames::{Quat, Vec3};
    use robot_types::ids::Name;
    use robot_types::motion::TaskLabel;
    use robot_types::time::Duration;

    fn config() -> TrackingConfig {
        TrackingConfig {
            options: SimOptions {
                timestep: Duration::from_millis(1),
                ..SimOptions::default()
            },
            bandwidth_hz: 8.0,
            damping_ratio: 1.0,
            use_declared_drives: false,
            floating: false,
            ground: Ground {
                height: 0.0,
                friction: 0.8,
            },
            fall_height_ratio: 0.5,
        }
    }

    /// The bolted G1 swinging its right elbow through `amplitude` radians
    /// at `hz`.
    fn elbow_swing(amplitude: f64, hz: f64) -> MotionRecord<RobotReference> {
        let profile = g1();
        let elbow = profile
            .actuated_joints()
            .position(|j| j.name.as_str() == "right_elbow_joint")
            .unwrap();
        let mut r = reference_from(&profile, 26, |t| {
            let mut q = vec![0.0; profile.dof()];
            q[elbow] = 0.5 + amplitude * (2.0 * core::f64::consts::PI * hz * t).sin();
            (Vec3::ZERO, Quat::IDENTITY, q)
        });
        r.task_labels.push(TaskLabel {
            label: Name::new("swing").unwrap(),
            start: 0,
            end: 26,
        });
        r
    }

    #[test]
    fn a_gentle_swing_is_tracked_closely_with_feedforward() {
        let profile = g1();
        let r = elbow_swing(0.3, 1.0);
        let traj = Trajectory::from_reference(&r, profile.dof()).unwrap();
        let ff = estimate(&profile, &r, &traj, &[], 0.5, 0.1).unwrap();
        let with = track(&profile, &r, &traj, Some(&ff), &config()).unwrap();
        let without = track(&profile, &r, &traj, None, &config()).unwrap();
        assert_eq!(with.controller, "pd+feedforward");
        assert!(
            with.overall_joint_rms() < 0.01,
            "{}",
            with.overall_joint_rms()
        );
        assert!(with.overall_joint_rms() < without.overall_joint_rms());
        assert_eq!(with.per_label.len(), 1);
        assert!(!with.per_label[0].fell);
        assert_eq!(with.steps, 500);
    }

    #[test]
    fn a_floating_g1_standing_on_its_soles_does_not_fall() {
        let profile = g1();
        let (r, _) = crate::test_support::standing_reference(&profile, 3, true);
        let traj = Trajectory::from_reference(&r, profile.dof()).unwrap();
        // Put the ground under the lowest collision point of the first frame.
        let floating = with_floating_base(profile.clone()).unwrap();
        let world = World::from_profile(&floating, SimOptions::default());
        let mut q = free_joint_q(&traj.base[0]).to_vec();
        q.extend_from_slice(&traj.q[0]);
        let internal = world.model().qpos_to_internal(&q).unwrap();
        let lowest = world.lowest_contact_height(&internal).unwrap().unwrap();
        let config = TrackingConfig {
            options: SimOptions::default(),
            floating: true,
            ground: Ground {
                height: lowest,
                friction: 0.8,
            },
            ..config()
        };
        let t = track(&profile, &r, &traj, None, &config).unwrap();
        assert_eq!(t.fell_at, None);
        assert_eq!(t.steps, 400);
        assert!(t.base_error.iter().all(|e| *e < 0.02), "{:?}", t.base_error);
    }

    #[test]
    fn an_impossibly_fast_swing_is_not_tracked() {
        let profile = g1();
        let r = elbow_swing(1.0, 20.0);
        let traj = Trajectory::from_reference(&r, profile.dof()).unwrap();
        let t = track(&profile, &r, &traj, None, &config()).unwrap();
        assert!(t.overall_joint_rms() > 0.05, "{}", t.overall_joint_rms());
    }
}

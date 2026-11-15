//! Torque estimates and dynamic feasibility, per SPECS.md Section 8.2 item
//! 8 ("Validate temporal smoothness, velocity/acceleration, torque
//! estimates") and Section 8.3 ("joint-limit violations, torque-speed
//! violations").
//!
//! **Inverse dynamics with contact.** The reference's base and joint
//! trajectories, differentiated, go through `robot-sim`'s recursive
//! Newton-Euler inverse dynamics on the floating-base version of the
//! profile. That gives the generalized force the motion needs: six rows for
//! the base (a wrench no actuator can supply) and one per joint. The base
//! wrench must come from the ground through the feet the reference labels
//! as planted, so it is distributed over them as the least-norm solution of
//! `sum_i A_i w_i = W_base`, with each foot's moment weighted by
//! `1 / moment_length^2` against its force so a foot is not asked to
//! provide a large torque where a force would do. The joint torques are
//! then `tau = ID_joints - sum_i J_i' w_i`.
//!
//! With no planted foot the base wrench stays unsupported, which is the
//! test of a flight phase: a ballistic jump needs none, a reference that
//! hovers needs its whole weight, and [`FrameDynamics::unsupported_force`]
//! says which.
//!
//! The distribution is a minimum-norm estimate, not the robot's actual
//! contact forces (those depend on the controller), and it does not
//! enforce unilaterality or friction: [`crate::contact`] checks the result
//! against both and reports what it would take.

use core::fmt;
use robot_model::joint::JointKind;
use robot_model::kinematics::forward_kinematics;
use robot_model::profile::RobotProfile;
use robot_model::usd::with_floating_base;
use robot_sim::articulated::ArticulatedBody;
use robot_types::frames::Vec3;
use robot_types::ids::{ContactSiteId, LinkId};
use robot_types::math::Matrix;
use robot_types::motion::{MotionRecord, RobotReference};
use robot_types::units::{Radians, RadiansPerSecond};

use crate::trajectory::{free_joint_q, Trajectory};

/// A robot link that may carry a ground contact.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ContactSpec {
    /// Contact site the reference's labels use.
    pub site: ContactSiteId,
    /// Index into `profile.links`.
    pub link: usize,
}

/// Why an estimate could not be made.
#[derive(Clone, PartialEq, Debug)]
pub enum DynamicsError {
    /// The floating-base profile could not be built.
    Profile(String),
    /// Inverse dynamics failed.
    Sim(String),
    /// A contact link index was out of range.
    UnknownLink(usize),
}

impl fmt::Display for DynamicsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Profile(e) => write!(f, "floating-base profile: {e}"),
            Self::Sim(e) => write!(f, "inverse dynamics: {e}"),
            Self::UnknownLink(l) => write!(f, "link {l} does not exist"),
        }
    }
}

impl core::error::Error for DynamicsError {}

/// The estimated dynamics of one frame.
#[derive(Clone, PartialEq, Debug)]
pub struct FrameDynamics {
    /// Actuated-joint torques after contact distribution.
    pub joint_torque: Vec<f64>,
    /// Contact wrenches on the robot, world frame, about each foot link's
    /// origin: `(site, moment, force)`.
    pub contact_wrenches: Vec<(ContactSiteId, Vec3, Vec3)>,
    /// Norm of the base force no contact supplies, newtons.
    pub unsupported_force: f64,
    /// Norm of the base moment no contact supplies, newton-metres.
    pub unsupported_moment: f64,
}

/// One joint over its limits at one frame.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct TorqueExcess {
    /// Actuated-joint index.
    pub joint: usize,
    /// Torque required, N m.
    pub required: f64,
    /// Torque the actuator can supply at that speed, N m.
    pub available: f64,
}

/// Inverse dynamics for every frame of `reference`.
///
/// `profile` is the fixed-base profile the reference was retargeted onto;
/// a free joint is added for the estimate. A site counts as planted when the
/// reference labels it in contact with at least `contact_confidence`.
pub fn estimate(
    profile: &RobotProfile,
    reference: &MotionRecord<RobotReference>,
    trajectory: &Trajectory,
    contacts: &[ContactSpec],
    contact_confidence: f64,
    moment_length: f64,
) -> Result<Vec<FrameDynamics>, DynamicsError> {
    for c in contacts {
        if c.link >= profile.links.len() {
            return Err(DynamicsError::UnknownLink(c.link));
        }
    }
    let floating =
        with_floating_base(profile.clone()).map_err(|e| DynamicsError::Profile(e.to_string()))?;
    let body = ArticulatedBody::from_profile(&floating);
    let gravity = profile.gravity.acceleration();
    let joints: Vec<_> = profile.actuated_joints().collect();
    let paths: Vec<Vec<usize>> = contacts
        .iter()
        .map(|c| path_joints(profile, c.link))
        .collect();
    let l2 = moment_length * moment_length;

    let mut out = Vec::with_capacity(trajectory.len());
    for f in 0..trajectory.len() {
        let mut q = free_joint_q(&trajectory.base[f]).to_vec();
        q.extend_from_slice(&trajectory.q[f]);
        let mut qd = trajectory.base_twist[f].to_vec();
        qd.extend_from_slice(&trajectory.qd[f]);
        let mut qdd = trajectory.base_twist_rate[f].to_vec();
        qdd.extend_from_slice(&trajectory.qdd[f]);
        let tau = body
            .inverse_dynamics(&q, &qd, &qdd, gravity)
            .map_err(|e| DynamicsError::Sim(e.to_string()))?;
        let base_wrench = [tau[0], tau[1], tau[2], tau[3], tau[4], tau[5]];
        let mut joint_torque = tau[6..].to_vec();

        let planted: Vec<usize> = contacts
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                reference.frames[f]
                    .contacts
                    .iter()
                    .any(|l| l.site == c.site && l.in_contact && l.confidence >= contact_confidence)
            })
            .map(|(i, _)| i)
            .collect();
        let base = trajectory.base[f];
        let poses = forward_kinematics(profile, &trajectory.q[f])
            .map_err(|e| DynamicsError::Sim(e.to_string()))?;
        let world = |link: usize| base.compose(&poses.by_index(link).expect("validated link"));
        let rt = base.rotation.conjugate();

        let mut wrenches = Vec::new();
        let (mut unsupported_force, mut unsupported_moment) = (0.0, 0.0);
        if planted.is_empty() {
            unsupported_moment = Vec3::new(base_wrench[0], base_wrench[1], base_wrench[2]).norm();
            unsupported_force = Vec3::new(base_wrench[3], base_wrench[4], base_wrench[5]).norm();
        } else {
            // A (6 x 6k): world foot wrench [m; f] to base-frame wrench.
            let k = planted.len();
            let mut a = Matrix::zeros(6, 6 * k);
            for (slot, &ci) in planted.iter().enumerate() {
                let r = world(contacts[ci].link).translation - base.translation;
                for axis in 0..3 {
                    let mut e = [0.0; 3];
                    e[axis] = 1.0;
                    let unit = Vec3::from_xyz(e);
                    // A world moment maps to R' m; a world force to
                    // (R' (r x f), R' f).
                    let m_b = rt.rotate(unit).to_xyz();
                    let f_m = rt.rotate(r.cross(unit)).to_xyz();
                    let f_f = rt.rotate(unit).to_xyz();
                    for row in 0..3 {
                        a[(row, 6 * slot + axis)] = m_b[row];
                        a[(row, 6 * slot + 3 + axis)] = f_m[row];
                        a[(3 + row, 6 * slot + 3 + axis)] = f_f[row];
                    }
                }
            }
            let minv: Vec<f64> = (0..6 * k)
                .map(|i| if i % 6 < 3 { l2 } else { 1.0 })
                .collect();
            let mut am = a.clone();
            for c in 0..6 * k {
                for r in 0..6 {
                    am[(r, c)] *= minv[c];
                }
            }
            let gram = am
                .mul(&a.transposed())
                .map_err(|e| DynamicsError::Sim(e.to_string()))?;
            let y = gram
                .solve(&Matrix::column(&base_wrench))
                .map_err(|e| DynamicsError::Sim(e.to_string()))?;
            let w = am
                .transposed()
                .mul(&y)
                .map_err(|e| DynamicsError::Sim(e.to_string()))?;
            let w = w.as_slice();
            for (slot, &ci) in planted.iter().enumerate() {
                let m = Vec3::new(w[6 * slot], w[6 * slot + 1], w[6 * slot + 2]);
                let force = Vec3::new(w[6 * slot + 3], w[6 * slot + 4], w[6 * slot + 5]);
                let foot = world(contacts[ci].link).translation;
                for &j in &paths[ci] {
                    let spec = joints[j];
                    let child = profile
                        .links
                        .iter()
                        .position(|l| l.id == spec.child)
                        .expect("validated");
                    let frame = world(child);
                    let axis = frame.rotation.rotate(spec.axis);
                    joint_torque[j] -= match spec.kind {
                        JointKind::Prismatic => axis.dot(force),
                        _ => axis.dot(m + (foot - frame.translation).cross(force)),
                    };
                }
                wrenches.push((contacts[ci].site, m, force));
            }
        }
        out.push(FrameDynamics {
            joint_torque,
            contact_wrenches: wrenches,
            unsupported_force,
            unsupported_moment,
        });
    }
    Ok(out)
}

/// Actuated-joint indices on the path from the root to `link`.
fn path_joints(profile: &RobotProfile, link: usize) -> Vec<usize> {
    let actuated: Vec<_> = profile.actuated_joints().map(|j| j.id).collect();
    let mut out = Vec::new();
    let mut current: LinkId = profile.links[link].id;
    while let Some(joint) = profile.parent_joint_of(current) {
        if let Some(i) = actuated.iter().position(|id| *id == joint.id) {
            out.push(i);
        }
        current = joint.parent;
    }
    out
}

/// Joints whose required torque exceeds what their actuator supplies at the
/// frame's speed (the torque-speed envelope through the transmission). A
/// joint with no actuator is skipped: what it needs is reported in the
/// torques, but there is no envelope to compare with.
#[must_use]
pub fn torque_excess(profile: &RobotProfile, torque: &[f64], speed: &[f64]) -> Vec<TorqueExcess> {
    let joints: Vec<_> = profile.actuated_joints().collect();
    let mut out = Vec::new();
    for (i, joint) in joints.iter().enumerate() {
        let Some(act) = profile.actuators.iter().find(|a| a.joint == joint.id) else {
            continue;
        };
        let motor_speed = act
            .transmission
            .joint_velocity_to_motor(RadiansPerSecond(speed[i]));
        let available = act
            .transmission
            .motor_torque_to_joint(act.envelope.available_torque(motor_speed))
            .value()
            .abs();
        if torque[i].abs() > available {
            out.push(TorqueExcess {
                joint: i,
                required: torque[i],
                available,
            });
        }
    }
    out
}

/// Per-joint motion bounds derived from the profile's own declarations —
/// never invented numbers (SPECS.md Section 8.2's smoothness and
/// feasibility validation, gated by project-owner direction of 2026-11-15).
///
/// The acceleration bound is the joint's declared effort limit over its
/// diagonal effective inertia at the reference (zero) posture: `M_jj` is
/// read off exactly as one inverse-dynamics column (`qdd = e_j`, zero
/// velocity, zero gravity), using the same `ArticulatedBody` the torque
/// estimation already trusts. The jerk bound is the acceleration bound
/// over the driving actuator's declared `command_latency` — the fastest
/// the commanded torque, and so the acceleration, can meaningfully change.
///
/// A joint with no effort limit, no actuator, or a degenerate `M_jj`
/// yields `f64::INFINITY` in that slot: there is **no declared basis** for
/// a bound there, and an unbounded slot never rejects — stated rather
/// than silently inventing one.
#[derive(Clone, Debug, PartialEq)]
pub struct DerivedMotionLimits {
    /// Largest acceptable |acceleration| per joint, rad/s^2.
    pub max_acceleration: Vec<f64>,
    /// Largest acceptable RMS jerk per joint, rad/s^3.
    pub max_rms_jerk: Vec<f64>,
}

/// Derives [`DerivedMotionLimits`] for a profile. See the type's own
/// documentation for exactly where each number comes from.
///
/// # Errors
///
/// [`DynamicsError::Sim`] when the inverse-dynamics column cannot be
/// evaluated.
pub fn derived_motion_limits(profile: &RobotProfile) -> Result<DerivedMotionLimits, DynamicsError> {
    let body = ArticulatedBody::from_profile(profile);
    let dof = profile.dof();
    let q = vec![0.0; dof];
    let qd = vec![0.0; dof];
    let mut max_acceleration = Vec::with_capacity(dof);
    let mut max_rms_jerk = Vec::with_capacity(dof);
    for (j, joint) in profile.actuated_joints().enumerate() {
        let mut qdd = vec![0.0; dof];
        qdd[j] = 1.0;
        let tau = body
            .inverse_dynamics(&q, &qd, &qdd, Vec3::ZERO)
            .map_err(|e| DynamicsError::Sim(e.to_string()))?;
        let m_jj = tau[j];
        let effort = joint.limits.effort.value();
        let accel = if effort.is_finite() && effort > 0.0 && m_jj.is_finite() && m_jj > 0.0 {
            effort / m_jj
        } else {
            f64::INFINITY
        };
        let latency = profile
            .actuators
            .iter()
            .find(|a| a.joint == joint.id)
            .map(|a| a.command_latency.as_secs_f64());
        let jerk = match latency {
            Some(dt) if dt > 0.0 && accel.is_finite() => accel / dt,
            _ => f64::INFINITY,
        };
        max_acceleration.push(accel);
        max_rms_jerk.push(jerk);
    }
    Ok(DerivedMotionLimits {
        max_acceleration,
        max_rms_jerk,
    })
}

/// Position-limit excess per joint, as `(joint, excess)`.
#[must_use]
pub fn position_excess(profile: &RobotProfile, q: &[f64]) -> Vec<(usize, Radians)> {
    profile
        .actuated_joints()
        .zip(q)
        .enumerate()
        .filter_map(|(i, (j, &v))| {
            let range = j.limits.position?;
            (!range.contains(Radians(v))).then(|| (i, range.excess(Radians(v))))
        })
        .collect()
}

/// Velocity-limit excess per joint, as `(joint, speed)`.
#[must_use]
pub fn velocity_excess(profile: &RobotProfile, qd: &[f64]) -> Vec<(usize, f64)> {
    profile
        .actuated_joints()
        .zip(qd)
        .enumerate()
        .filter_map(|(i, (j, &v))| {
            let limit = j.limits.velocity.value();
            (limit > 0.0 && v.abs() > limit).then_some((i, v))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{g1, reference_from, standing_reference};
    use robot_types::frames::Quat;

    #[test]
    fn derived_limits_agree_with_the_composite_mass_matrix_diagonal() {
        // The bound is effort over M_jj, with M_jj read off one
        // inverse-dynamics column (RNE). The composite-rigid-body algorithm
        // builds the same matrix by a different recursion in a different
        // frame, so agreement is a cross-algorithm check, not a tautology.
        let profile = g1();
        let limits = derived_motion_limits(&profile).unwrap();
        let model = robot_sim::composite::CompositeModel::from_profile(&profile);
        let q = vec![0.0; model.nq];
        let pos = robot_sim::composite::position(&model, &q).unwrap();
        let mass = robot_sim::composite::crb(&model, &pos);
        // The composite model orders degrees of freedom internally; map its
        // diagonal into the profile's actuated order before comparing.
        let diag_internal: Vec<f64> = (0..model.nv())
            .map(|i| mass[model.tree.diag_adr(i)])
            .collect();
        let diag = model.dof_to_external(&diag_internal).unwrap();
        for (j, joint) in profile.actuated_joints().enumerate() {
            let effort = joint.limits.effort.value();
            let accel = limits.max_acceleration[j];
            assert!(accel.is_finite() && accel > 0.0, "joint {j}: {accel}");
            let m_jj = effort / accel;
            let crb_jj = diag[j];
            assert!(
                (m_jj - crb_jj).abs() <= 1e-9 * crb_jj.max(1.0),
                "joint {j}: RNE column gives {m_jj}, CRB gives {crb_jj}"
            );
            // The jerk bound is exactly the acceleration bound over the
            // actuator's declared command latency; a zero latency declares
            // no basis for a jerk bound, so the slot is unbounded.
            let act = profile
                .actuators
                .iter()
                .find(|a| a.joint == joint.id)
                .unwrap();
            let dt = act.command_latency.as_secs_f64();
            if dt > 0.0 {
                let expect = accel / dt;
                assert!((limits.max_rms_jerk[j] - expect).abs() <= 1e-9 * expect);
            } else {
                assert!(limits.max_rms_jerk[j].is_infinite(), "joint {j}");
            }
        }
    }

    #[test]
    fn standing_still_puts_the_whole_weight_on_the_feet() {
        let profile = g1();
        let (reference, feet) = standing_reference(&profile, 5, true);
        let traj = Trajectory::from_reference(&reference, profile.dof()).unwrap();
        let est = estimate(&profile, &reference, &traj, &feet, 0.5, 0.1).unwrap();
        let mass: f64 = profile.links.iter().map(|l| l.mass.mass.value()).sum();
        let g = profile.gravity.acceleration().norm();
        for f in &est {
            let fz: f64 = f.contact_wrenches.iter().map(|(_, _, force)| force.z).sum();
            assert!(
                (fz - mass * g).abs() < 1e-6 * mass * g,
                "{fz} vs {}",
                mass * g
            );
            let horizontal: f64 = f
                .contact_wrenches
                .iter()
                .map(|(_, _, force)| force.x.hypot(force.y))
                .sum();
            assert!(horizontal < 1e-6);
            assert_eq!(f.unsupported_force, 0.0);
            assert!(f.joint_torque.iter().all(|t| t.is_finite()));
        }
    }

    #[test]
    fn hovering_without_contact_is_unsupported_and_free_fall_is_not() {
        let profile = g1();
        let mass: f64 = profile.links.iter().map(|l| l.mass.mass.value()).sum();
        let g = profile.gravity.acceleration().norm();
        let (hover, feet) = standing_reference(&profile, 5, false);
        let traj = Trajectory::from_reference(&hover, profile.dof()).unwrap();
        let est = estimate(&profile, &hover, &traj, &feet, 0.5, 0.1).unwrap();
        assert!((est[2].unsupported_force - mass * g).abs() < 1e-6 * mass * g);

        let zero = vec![0.0; profile.dof()];
        let fall = reference_from(&profile, 7, |t| {
            (
                Vec3::new(0.0, 0.0, 1.0 - 0.5 * g * t * t),
                Quat::IDENTITY,
                zero.clone(),
            )
        });
        let traj = Trajectory::from_reference(&fall, profile.dof()).unwrap();
        let est = estimate(&profile, &fall, &traj, &feet, 0.5, 0.1).unwrap();
        // Central differences of a parabola are exact in the interior.
        for f in &est[2..5] {
            assert!(
                f.unsupported_force < 1e-6 * mass * g,
                "{}",
                f.unsupported_force
            );
        }
    }

    #[test]
    fn the_envelope_flags_a_torque_beyond_stall() {
        let profile = g1();
        let joints = profile.dof();
        let mut torque = vec![0.0; joints];
        let speed = vec![0.0; joints];
        assert!(torque_excess(&profile, &torque, &speed).is_empty());
        torque[3] = 1e6;
        let excess = torque_excess(&profile, &torque, &speed);
        assert_eq!(excess.len(), 1);
        assert_eq!(excess[0].joint, 3);
    }
}

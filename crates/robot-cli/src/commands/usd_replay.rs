//! Builds the USD replay of a floating-base run (ADR 0015).
//!
//! The simulator's state is one vector per step: the pelvis position and
//! quaternion (`w, x, y, z`), then the hinge positions; the velocity and torque
//! vectors follow the degrees of freedom, the free base first. The link poses
//! are not part of that state, so they are computed here, by forward kinematics
//! on a fixed-base copy of the profile and the pelvis pose from the state.
//! Shapes come from the USD collision prims of the profile, unchanged.

use robot_model::joint::JointKind;
use robot_model::kinematics::forward_kinematics;
use robot_model::link::CollisionGeometry;
use robot_model::pose::RigidPose;
use robot_model::profile::RobotProfile;
use robot_model::usd::VisualMesh;
use robot_types::frames::{Quat, Vec3};
use robot_usd::replay::{
    write_replay_usda, Replay, ReplayGround, ReplayJoint, ReplayLink, ReplayShape, ReplayShapeKind,
};

/// One simulated step as the replay needs it.
pub struct CapturedStep {
    /// Positions, in the layout of [`ArticulatedBody`](robot_sim::articulated::ArticulatedBody).
    pub q: Vec<f64>,
    /// Velocities, in degrees of freedom.
    pub qd: Vec<f64>,
    /// Applied torques, in degrees of freedom.
    pub tau: Vec<f64>,
    /// Hard-limit constraint torques, in degrees of freedom.
    pub limit_torque: Vec<f64>,
    /// Position within the range of each actuated joint that has a limit,
    /// in actuated order; `None` for a joint without one.
    pub limit_fraction: Vec<Option<f64>>,
}

/// Where a joint's state sits in the vectors of a step.
struct JointSlots {
    name: String,
    position: usize,
    dof: usize,
    actuated: usize,
    bounded: bool,
}

/// Builds the replay of `steps`, one frame per entry, written at
/// `frames_per_second` time codes per second, from the floating-base `profile`.
/// `visuals` are the display meshes of the USD layer, attached to their links.
///
/// # Errors
///
/// Returns a message if the profile has no free base to take the pelvis pose
/// from, or if a joint is not a hinge in the layout this module assumes.
pub fn build(
    profile: &RobotProfile,
    steps: &[CapturedStep],
    frames_per_second: f64,
    visuals: &[VisualMesh],
    ground: Option<ReplayGround>,
) -> Result<Replay, String> {
    let mut position = 0usize;
    let mut dof = 0usize;
    let mut slots = Vec::new();
    for joint in profile.actuated_joints() {
        let width = match joint.kind {
            JointKind::Free => 7,
            JointKind::Revolute | JointKind::Continuous | JointKind::Prismatic => 1,
            other => {
                return Err(format!(
                    "joint {} of kind {other:?} is not in the replay layout",
                    joint.name
                ))
            }
        };
        if joint.kind != JointKind::Free {
            slots.push(JointSlots {
                name: joint.name.to_string(),
                position,
                dof,
                actuated: slots.len(),
                bounded: joint.limits.position.is_some(),
            });
        }
        position += width;
        dof += joint.kind.dof_count();
    }
    let free = profile
        .joints
        .iter()
        .find(|j| j.kind == JointKind::Free)
        .ok_or("the replay needs a floating base")?;
    let world = profile
        .link(free.parent)
        .ok_or("the free joint has no parent link")?;

    // A fixed-base copy, for forward kinematics: the free joint becomes fixed
    // at its origin, and the pelvis pose is applied on top.
    let mut fixed = profile.clone();
    for joint in &mut fixed.joints {
        if joint.kind == JointKind::Free {
            joint.kind = JointKind::Fixed;
            joint.axis = Vec3::ZERO;
        }
    }
    let hinge_names: Vec<String> = fixed
        .actuated_joints()
        .map(|j| j.name.to_string())
        .collect();

    let mut link_tracks: Vec<ReplayLink> = profile
        .links
        .iter()
        .map(|link| {
            let name = link.name.to_string();
            let mut shapes: Vec<ReplayShape> = link
                .collision
                .iter()
                .enumerate()
                .map(|(i, shape)| replay_shape(i, shape))
                .collect();
            shapes.extend(
                visuals
                    .iter()
                    .filter(|v| v.link == name)
                    .enumerate()
                    .map(|(i, v)| replay_visual(i, v)),
            );
            ReplayLink {
                name,
                translation: Vec::with_capacity(steps.len()),
                orientation: Vec::with_capacity(steps.len()),
                shapes,
            }
        })
        .collect();
    let mut joint_tracks: Vec<ReplayJoint> = slots
        .iter()
        .map(|s| ReplayJoint {
            name: s.name.clone(),
            position: Vec::with_capacity(steps.len()),
            velocity: Vec::with_capacity(steps.len()),
            applied_torque: Vec::with_capacity(steps.len()),
            limit_fraction: if s.bounded {
                Some(Vec::with_capacity(steps.len()))
            } else {
                None
            },
            limit_torque: Vec::with_capacity(steps.len()),
        })
        .collect();

    for step in steps {
        let base = RigidPose::new(
            Vec3::new(step.q[0], step.q[1], step.q[2]),
            Quat {
                w: step.q[3],
                x: step.q[4],
                y: step.q[5],
                z: step.q[6],
            },
        );
        let hinge_values: Vec<f64> = hinge_names
            .iter()
            .map(|name| {
                slots
                    .iter()
                    .find(|s| &s.name == name)
                    .map(|s| step.q[s.position])
                    .unwrap_or(0.0)
            })
            .collect();
        let poses = forward_kinematics(&fixed, &hinge_values).map_err(|e| e.to_string())?;
        for track in &mut link_tracks {
            let pose = if track.name == world.name.as_str() {
                RigidPose::IDENTITY
            } else {
                let index = fixed
                    .links
                    .iter()
                    .position(|l| l.name.as_str() == track.name)
                    .ok_or_else(|| format!("link {} missing from the fixed copy", track.name))?;
                base.compose(&poses.as_slice()[index])
            };
            track
                .translation
                .push([pose.translation.x, pose.translation.y, pose.translation.z]);
            track.orientation.push([
                pose.rotation.x,
                pose.rotation.y,
                pose.rotation.z,
                pose.rotation.w,
            ]);
        }
        for (track, slot) in joint_tracks.iter_mut().zip(&slots) {
            track.position.push(step.q[slot.position]);
            track.velocity.push(step.qd[slot.dof]);
            track.applied_torque.push(step.tau[slot.dof]);
            track.limit_torque.push(step.limit_torque[slot.dof]);
            if let Some(fractions) = &mut track.limit_fraction {
                fractions.push(step.limit_fraction[slot.actuated].unwrap_or(0.5));
            }
        }
    }

    Ok(Replay {
        ground,
        frames_per_second,
        links: link_tracks,
        joints: joint_tracks,
    })
}

/// The largest distance, in radians, by which any bounded joint of `steps`
/// lies outside its range, and the number of steps in which one did. Zero for a
/// run that respects its limits. Measured from the captured positions and the
/// profile's limits, so it does not depend on the solver.
#[must_use]
pub fn limit_excursion(profile: &RobotProfile, steps: &[CapturedStep]) -> (f64, usize) {
    let mut position = 0usize;
    let mut worst = 0.0_f64;
    let mut outside_steps = 0usize;
    let mut bounded: Vec<(usize, f64, f64)> = Vec::new();
    for joint in profile.actuated_joints() {
        if let (Some(range), JointKind::Revolute | JointKind::Prismatic) =
            (joint.limits.position, joint.kind)
        {
            bounded.push((position, range.lo.value(), range.hi.value()));
        }
        position += if joint.kind == JointKind::Free { 7 } else { 1 };
    }
    for step in steps {
        let mut outside = false;
        for &(slot, lo, hi) in &bounded {
            let x = step.q[slot];
            let excess = (lo - x).max(x - hi).max(0.0);
            if excess > 0.0 {
                outside = true;
                worst = worst.max(excess);
            }
        }
        if outside {
            outside_steps += 1;
        }
    }
    (worst, outside_steps)
}

/// Writes the replay of `steps` as USDA text.
///
/// # Errors
///
/// Returns the message of the first failure, from [`build`] or the writer.
pub fn usda(
    profile: &RobotProfile,
    steps: &[CapturedStep],
    frames_per_second: f64,
    visuals: &[VisualMesh],
    ground: Option<ReplayGround>,
) -> Result<String, String> {
    let replay = build(profile, steps, frames_per_second, visuals, ground)?;
    write_replay_usda(&replay).map_err(|e| e.to_string())
}

/// A display mesh, with its geometry and pose in the link frame.
fn replay_visual(index: usize, visual: &VisualMesh) -> ReplayShape {
    ReplayShape {
        name: format!("visual_{index}_{}", visual.name),
        kind: ReplayShapeKind::Mesh {
            points: visual.points.clone(),
            face_vertex_counts: visual.face_vertex_counts.clone(),
            face_vertex_indices: visual.face_vertex_indices.clone(),
        },
        translation: [
            visual.pose.translation.x,
            visual.pose.translation.y,
            visual.pose.translation.z,
        ],
        orientation: [
            visual.pose.rotation.x,
            visual.pose.rotation.y,
            visual.pose.rotation.z,
            visual.pose.rotation.w,
        ],
        physics_only: false,
    }
}

fn replay_shape(index: usize, shape: &robot_model::link::CollisionShape) -> ReplayShape {
    let (kind, name) = match &shape.geometry {
        CollisionGeometry::Sphere { radius } => (
            ReplayShapeKind::Sphere { radius: *radius },
            format!("sphere_{index}"),
        ),
        CollisionGeometry::Box { half_extents } => (
            ReplayShapeKind::Cube {
                extents: [
                    2.0 * half_extents.x,
                    2.0 * half_extents.y,
                    2.0 * half_extents.z,
                ],
            },
            format!("box_{index}"),
        ),
        CollisionGeometry::Cylinder { radius, length } => (
            ReplayShapeKind::Cylinder {
                radius: *radius,
                height: *length,
            },
            format!("cylinder_{index}"),
        ),
        CollisionGeometry::Mesh { content_id, .. } => (
            ReplayShapeKind::MeshReference {
                path: content_id.clone(),
            },
            format!("mesh_{index}"),
        ),
    };
    ReplayShape {
        name,
        kind,
        translation: [
            shape.pose.translation.x,
            shape.pose.translation.y,
            shape.pose.translation.z,
        ],
        orientation: [
            shape.pose.rotation.x,
            shape.pose.rotation.y,
            shape.pose.rotation.z,
            shape.pose.rotation.w,
        ],
        physics_only: true,
    }
}

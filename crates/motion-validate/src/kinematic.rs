//! Kinematic error of a retargeted reference against its human source,
//! per SPECS.md Section 8.3: "Measure endpoint error normalized by body
//! size, root drift, planted-foot slip".
//!
//! The targets are rebuilt from the same [`RetargetMap`] the reference was
//! made with, so the error is measured against what was asked for:
//!
//! - **endpoint error**: for every mapped link with a position weight, the
//!   distance between the robot link and its target (absolute, or relative
//!   to its base link with the chain ratio), divided by `body_size`; the
//!   frame's value is the worst link;
//! - **root drift**: horizontal distance between the robot base and the
//!   scaled human root;
//! - **planted-foot slip**: for each mapped foot, while the *human* record
//!   labels it in contact, the horizontal distance the robot foot has moved
//!   since the contact began; the frame's value is the worst foot, `None`
//!   when no foot is planted.

use motion_process::kinematics::frame_poses;
use motion_retarget::retarget::{Anchor, RetargetMap};
use robot_model::kinematics::forward_kinematics;
use robot_model::profile::RobotProfile;
use robot_types::frames::Vec3;
use robot_types::motion::{HumanSkeleton, MotionRecord};

use crate::report::KinematicError;
use crate::trajectory::Trajectory;

/// Measures every frame. `human` and the reference behind `trajectory`
/// must have the same frame count.
#[must_use]
pub fn measure(
    human: &MotionRecord<HumanSkeleton>,
    profile: &RobotProfile,
    trajectory: &Trajectory,
    map: &RetargetMap,
    body_size: f64,
    contact_confidence: f64,
) -> Vec<KinematicError> {
    let mut plant_start: Vec<Option<Vec3>> = vec![None; map.feet.len()];
    let horizontal = |v: Vec3| v.x.hypot(v.y);
    (0..trajectory.len().min(human.frames.len()))
        .map(|f| {
            let hp = frame_poses(human, f);
            let base = trajectory.base[f];
            let world = forward_kinematics(profile, &trajectory.q[f]).map(|poses| {
                move |link: usize| {
                    base.compose(&poses.by_index(link).expect("link"))
                        .translation
                }
            });
            let mut endpoint = 0.0f64;
            let mut slip: Option<f64> = None;
            if let Ok(world) = &world {
                for m in map.links.iter().filter(|m| m.position_weight > 0.0) {
                    let Some((p, _)) = hp[m.human] else { continue };
                    let err = match m.anchor {
                        Anchor::Absolute => (world(m.robot) - p * map.scale).norm(),
                        Anchor::Relative {
                            human_base,
                            robot_base,
                            ratio,
                        } => {
                            let Some((pb, _)) = hp[human_base] else {
                                continue;
                            };
                            (world(m.robot) - world(robot_base) - (p - pb) * ratio).norm()
                        }
                    };
                    endpoint = endpoint.max(err);
                }
                for (k, foot) in map.feet.iter().enumerate() {
                    let planted = human.frames[f].contacts.iter().any(|c| {
                        c.site == foot.site && c.in_contact && c.confidence >= contact_confidence
                    });
                    if !planted {
                        plant_start[k] = None;
                        continue;
                    }
                    let now = world(foot.robot);
                    let start = *plant_start[k].get_or_insert(now);
                    let s = horizontal(now - start);
                    slip = Some(slip.map_or(s, |v: f64| v.max(s)));
                }
            }
            KinematicError {
                normalized_endpoint_error: endpoint / body_size,
                root_drift: horizontal(
                    base.translation - human.frames[f].root.translation * map.scale,
                ),
                planted_foot_slip: slip,
            }
        })
        .collect()
}

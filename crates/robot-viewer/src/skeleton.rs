//! Std-only skeleton rendering and declarative playback of a
//! [`PoseFrame`] log against its robot profile.
//!
//! [`render_skeleton`] draws every link of the robot as a bone from its
//! parent link's origin to its own, using the profile's own forward
//! kinematics placed at the frame's pelvis pose, in two orthographic views
//! side by side: side (x forward, z up) and front (y left, z up), with the
//! ground line. Contacts are not in a [`PoseFrame`] and are not drawn.
//!
//! [`render_playback`] puts a sequence of frames into one SVG and shows
//! each in turn with SMIL `<set>` elements at the frames' own times. It is
//! markup, not script: SPECS.md Section 2 requires every executable artifact
//! to be Rust, and a declarative animation in a data file is not one. It
//! plays once in any SVG viewer that implements SMIL, which every current
//! browser does.
//!
//! Output depends only on the inputs and every number is printed with
//! `{:.6}`, so the result is byte-identical across runs and checked against
//! a golden file (SPECS.md Section 14 row A-39).

use std::fmt::Write;

use robot_model::kinematics::forward_kinematics;
use robot_model::pose::RigidPose;
use robot_model::profile::RobotProfile;
use robot_types::frames::{Quat, Vec3};

use crate::pose::PoseFrame;

/// Width of one view panel.
const PANEL: f64 = 300.0;
/// Height of the canvas.
const HEIGHT: f64 = 360.0;
/// Metres shown across a panel.
const SPAN: f64 = 2.0;

/// Why a frame could not be drawn.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum SkeletonError {
    /// The frame's joint count differs from the profile's.
    Width {
        /// Profile joints.
        expected: usize,
        /// Frame joints.
        found: usize,
    },
    /// Forward kinematics refused the angles.
    Kinematics(String),
}

impl core::fmt::Display for SkeletonError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Width { expected, found } => {
                write!(f, "frame has {found} joints, profile {expected}")
            }
            Self::Kinematics(e) => write!(f, "{e}"),
        }
    }
}

impl core::error::Error for SkeletonError {}

/// Bones as `(parent origin, child origin)` in the world.
pub fn bones(
    profile: &RobotProfile,
    frame: &PoseFrame,
) -> Result<Vec<(Vec3, Vec3)>, SkeletonError> {
    if frame.joint_angles.len() != profile.dof() {
        return Err(SkeletonError::Width {
            expected: profile.dof(),
            found: frame.joint_angles.len(),
        });
    }
    let poses = forward_kinematics(profile, &frame.joint_angles)
        .map_err(|e| SkeletonError::Kinematics(e.to_string()))?;
    let [w, x, y, z] = frame.pelvis_quaternion;
    let base = RigidPose {
        translation: Vec3::from_xyz(frame.pelvis_position),
        rotation: Quat { w, x, y, z }.normalized(),
    };
    let index = |id| profile.links.iter().position(|l| l.id == id);
    let mut out = Vec::new();
    for j in &profile.joints {
        if let (Some(p), Some(c)) = (index(j.parent), index(j.child)) {
            let (Some(pp), Some(cp)) = (poses.by_index(p), poses.by_index(c)) else {
                continue;
            };
            out.push((base.compose(&pp).translation, base.compose(&cp).translation));
        }
    }
    Ok(out)
}

fn panel(
    out: &mut String,
    bones: &[(Vec3, Vec3)],
    offset_x: f64,
    horizontal: fn(Vec3) -> f64,
    centre: f64,
    label: &str,
) {
    let scale = PANEL / SPAN;
    let to = |h: f64, v: f64| {
        (
            offset_x + PANEL / 2.0 + (h - centre) * scale,
            HEIGHT - 20.0 - v * scale,
        )
    };
    let (gx0, gy) = to(centre - SPAN / 2.0, 0.0);
    let (gx1, _) = to(centre + SPAN / 2.0, 0.0);
    let _ = writeln!(out, "<line x1=\"{gx0:.6}\" y1=\"{gy:.6}\" x2=\"{gx1:.6}\" y2=\"{gy:.6}\" stroke=\"#888888\" stroke-width=\"2\"/>");
    let _ = writeln!(out, "<text x=\"{:.6}\" y=\"14.000000\" font-size=\"12\" font-family=\"monospace\">{label}</text>", offset_x + 6.0);
    for (a, b) in bones {
        let (x1, y1) = to(horizontal(*a), a.z);
        let (x2, y2) = to(horizontal(*b), b.z);
        let _ = writeln!(out, "<line x1=\"{x1:.6}\" y1=\"{y1:.6}\" x2=\"{x2:.6}\" y2=\"{y2:.6}\" stroke=\"#1f4e79\" stroke-width=\"3\" stroke-linecap=\"round\"/>");
    }
}

fn frame_body(profile: &RobotProfile, frame: &PoseFrame) -> Result<String, SkeletonError> {
    let b = bones(profile, frame)?;
    let [px, py, _] = frame.pelvis_position;
    let mut out = String::new();
    panel(&mut out, &b, 0.0, |v| v.x, px, "side (x, z)");
    panel(&mut out, &b, PANEL, |v| -v.y, -py, "front (-y, z)");
    let _ = writeln!(out, "<text x=\"6.000000\" y=\"{:.6}\" font-size=\"12\" font-family=\"monospace\">step {} t={:.6}s</text>", HEIGHT - 4.0, frame.step, frame.timestamp.nanos() as f64 * 1e-9);
    Ok(out)
}

fn header(out: &mut String) {
    let w = 2.0 * PANEL;
    let _ = writeln!(out, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w:.6}\" height=\"{HEIGHT:.6}\" viewBox=\"0 0 {w:.6} {HEIGHT:.6}\">");
    let _ = writeln!(out, "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>");
}

/// One frame's skeleton, side and front views.
pub fn render_skeleton(profile: &RobotProfile, frame: &PoseFrame) -> Result<String, SkeletonError> {
    let mut out = String::new();
    header(&mut out);
    out.push_str(&frame_body(profile, frame)?);
    out.push_str("</svg>\n");
    Ok(out)
}

/// Every `every`-th frame of `frames` in one SVG, each shown for its own
/// interval with SMIL, starting at the first frame's time.
pub fn render_playback(
    profile: &RobotProfile,
    frames: &[PoseFrame],
    every: usize,
) -> Result<String, SkeletonError> {
    let every = every.max(1);
    let shown: Vec<&PoseFrame> = frames.iter().step_by(every).collect();
    let t0 = shown.first().map_or(0, |f| f.timestamp.nanos());
    let mut out = String::new();
    header(&mut out);
    for (k, f) in shown.iter().enumerate() {
        let begin = (f.timestamp.nanos() - t0) as f64 * 1e-9;
        let end = shown
            .get(k + 1)
            .map_or(begin + 1.0, |n| (n.timestamp.nanos() - t0) as f64 * 1e-9);
        let initial = if k == 0 { "visible" } else { "hidden" };
        let _ = writeln!(out, "<g visibility=\"{initial}\">");
        if k > 0 {
            let _ = writeln!(
                out,
                "<set attributeName=\"visibility\" to=\"visible\" begin=\"{begin:.6}s\"/>"
            );
        }
        if k + 1 < shown.len() {
            let _ = writeln!(
                out,
                "<set attributeName=\"visibility\" to=\"hidden\" begin=\"{end:.6}s\"/>"
            );
        }
        out.push_str(&frame_body(profile, f)?);
        out.push_str("</g>\n");
    }
    out.push_str("</svg>\n");
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_model::fixtures::unitree_g1_29dof;
    use robot_types::time::{ClockId, Timestamp};

    fn frame(step: u64, angle: f64) -> PoseFrame {
        let profile = unitree_g1_29dof().unwrap();
        let mut q = vec![0.0; profile.dof()];
        q[0] = angle;
        PoseFrame {
            step,
            timestamp: Timestamp::sampling(step as i64 * 10_000_000, ClockId::Simulation),
            pelvis_position: [0.0, 0.0, 0.79],
            pelvis_quaternion: [1.0, 0.0, 0.0, 0.0],
            joint_angles: q,
        }
    }

    #[test]
    fn every_jointed_link_is_one_bone_and_the_feet_reach_the_ground() {
        let profile = unitree_g1_29dof().unwrap();
        let b = bones(&profile, &frame(0, 0.0)).unwrap();
        assert_eq!(b.len(), profile.joints.len());
        let lowest = b.iter().map(|(_, c)| c.z).fold(f64::INFINITY, f64::min);
        assert!(lowest > -0.05 && lowest < 0.15, "{lowest}");
    }

    #[test]
    fn the_skeleton_matches_its_golden_file() {
        let profile = unitree_g1_29dof().unwrap();
        let svg = render_skeleton(&profile, &frame(3, 0.3)).unwrap();
        assert_eq!(svg, render_skeleton(&profile, &frame(3, 0.3)).unwrap());
        let golden = include_str!("../tests/golden/g1_skeleton.svg");
        assert_eq!(
            svg, golden,
            "regenerate tests/golden/g1_skeleton.svg if the change is intended"
        );
    }

    #[test]
    fn playback_shows_each_frame_in_turn_with_smil_and_no_script() {
        let profile = unitree_g1_29dof().unwrap();
        let frames: Vec<_> = (0..5).map(|k| frame(k, 0.1 * k as f64)).collect();
        let svg = render_playback(&profile, &frames, 2).unwrap();
        assert_eq!(svg.matches("<g visibility=").count(), 3);
        assert!(svg.contains("begin=\"0.020000s\""));
        assert!(!svg.contains("<script"));
    }

    #[test]
    fn a_frame_of_the_wrong_width_is_refused() {
        let profile = unitree_g1_29dof().unwrap();
        let mut f = frame(0, 0.0);
        f.joint_angles.pop();
        assert!(matches!(
            render_skeleton(&profile, &f),
            Err(SkeletonError::Width { .. })
        ));
    }
}

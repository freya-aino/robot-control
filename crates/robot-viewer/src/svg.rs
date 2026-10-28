//! Std-only SVG rendering of one [`PoseFrame`].
//!
//! The picture is a fixed side view: x points right (forward), z points up,
//! and the ground line is at z = 0. The view window is fixed, so frames from
//! one run can be compared by eye. The output depends only on the frame: there
//! are no timestamps of the host, no paths and no random ids. Every number is
//! printed with `{:.6}`, which makes repeated runs byte-identical.
//!
//! The joint panel draws one horizontal bar per joint angle. A bar is
//! clamped to `±JOINT_DISPLAY_LIMIT` radians so a large angle cannot leave the
//! panel. The clamp is for display only and the number is printed beside each bar.

use crate::pose::PoseFrame;
use std::fmt::Write;

/// Canvas width in SVG user units.
pub const WIDTH: f64 = 480.0;
/// Canvas height for the side view, before the joint panel.
const VIEW_HEIGHT: f64 = 320.0;
/// Height of one joint bar row.
const ROW_HEIGHT: f64 = 14.0;
/// Canvas height for a run with `joints` joints.
pub fn height(joints: usize) -> f64 {
    VIEW_HEIGHT + ROW_HEIGHT * joints as f64 + 16.0
}
/// Fixed view window in metres: x from `X_MIN` to `X_MAX`, z from `Z_MIN` to `Z_MAX`.
const X_MIN: f64 = -1.0;
const X_MAX: f64 = 1.0;
const Z_MIN: f64 = -0.1;
const Z_MAX: f64 = 1.1;
/// Radius of the pelvis marker, in user units.
const PELVIS_RADIUS: f64 = 5.0;
/// Length of the orientation tick, in metres.
const TICK_METRES: f64 = 0.15;
/// Angle (rad) that fills half the joint bar panel.
const JOINT_DISPLAY_LIMIT: f64 = std::f64::consts::PI;
/// Half-width of the joint bar panel, in user units.
const BAR_HALF_WIDTH: f64 = 150.0;

/// Maps a world (x, z) in metres to SVG (x, y) in user units.
fn to_canvas(x: f64, z: f64) -> (f64, f64) {
    let sx = (x - X_MIN) / (X_MAX - X_MIN) * WIDTH;
    let sy = (Z_MAX - z) / (Z_MAX - Z_MIN) * VIEW_HEIGHT;
    (sx, sy)
}

/// The world x and z components of the pelvis' body x-axis, from its
/// unit quaternion `[w, x, y, z]`. This is the first column of the rotation
/// matrix, projected to the x-z plane.
fn forward_axis_xz(q: [f64; 4]) -> (f64, f64) {
    let [w, x, y, z] = q;
    let r00 = 1.0 - 2.0 * (y * y + z * z);
    let r20 = 2.0 * (x * z - w * y);
    (r00, r20)
}

/// Renders one frame as a complete SVG document.
pub fn render(frame: &PoseFrame) -> String {
    let joints = frame.joint_angles.len();
    let total_height = height(joints);
    let mut out = String::new();
    let _ = writeln!(
        out,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{WIDTH:.6}\" height=\"{total_height:.6}\" viewBox=\"0 0 {WIDTH:.6} {total_height:.6}\">"
    );
    let _ = writeln!(out, "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>");

    // Ground line at z = 0.
    let (gx0, gy) = to_canvas(X_MIN, 0.0);
    let (gx1, _) = to_canvas(X_MAX, 0.0);
    let _ = writeln!(
        out,
        "<line x1=\"{gx0:.6}\" y1=\"{gy:.6}\" x2=\"{gx1:.6}\" y2=\"{gy:.6}\" stroke=\"#888888\" stroke-width=\"2\"/>"
    );

    // Pelvis marker and orientation tick.
    let [px, _, pz] = frame.pelvis_position;
    let (cx, cy) = to_canvas(px, pz);
    let _ = writeln!(
        out,
        "<circle cx=\"{cx:.6}\" cy=\"{cy:.6}\" r=\"{PELVIS_RADIUS:.6}\" fill=\"#1f77b4\"/>"
    );
    let (ax, az) = forward_axis_xz(frame.pelvis_quaternion);
    let (tx, ty) = to_canvas(px + TICK_METRES * ax, pz + TICK_METRES * az);
    let _ = writeln!(
        out,
        "<line x1=\"{cx:.6}\" y1=\"{cy:.6}\" x2=\"{tx:.6}\" y2=\"{ty:.6}\" stroke=\"#d62728\" stroke-width=\"2\"/>"
    );

    // Step and time label.
    let seconds = frame.timestamp.nanos() as f64 / 1e9;
    let _ = writeln!(
        out,
        "<text x=\"8.000000\" y=\"16.000000\" font-family=\"monospace\" font-size=\"12\">step {} t={seconds:.6} s</text>",
        frame.step
    );

    // Joint panel: one bar per joint, centred on a zero line.
    let panel_top = VIEW_HEIGHT + 8.0;
    let centre = WIDTH / 2.0;
    for (i, angle) in frame.joint_angles.iter().enumerate() {
        let row_y = panel_top + ROW_HEIGHT * i as f64;
        let clamped = angle.clamp(-JOINT_DISPLAY_LIMIT, JOINT_DISPLAY_LIMIT);
        let length = clamped / JOINT_DISPLAY_LIMIT * BAR_HALF_WIDTH;
        let bar_x = if length >= 0.0 {
            centre
        } else {
            centre + length
        };
        let _ = writeln!(
            out,
            "<rect x=\"{bar_x:.6}\" y=\"{:.6}\" width=\"{:.6}\" height=\"{:.6}\" fill=\"#2ca02c\"/>",
            row_y + 2.0,
            length.abs(),
            ROW_HEIGHT - 4.0
        );
        let _ = writeln!(
            out,
            "<text x=\"{:.6}\" y=\"{:.6}\" font-family=\"monospace\" font-size=\"10\">{angle:.6}</text>",
            WIDTH - 110.0,
            row_y + ROW_HEIGHT - 3.0
        );
    }
    out.push_str("</svg>\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_types::time::{ClockId, Timestamp};

    fn fixture() -> PoseFrame {
        PoseFrame {
            step: 7,
            timestamp: Timestamp::sampling(700_000, ClockId::Simulation),
            pelvis_position: [0.1, 0.0, 0.79],
            pelvis_quaternion: [1.0, 0.0, 0.0, 0.0],
            joint_angles: vec![0.25, -0.5, 2.0],
        }
    }

    #[test]
    fn the_same_frame_renders_to_the_same_bytes() {
        assert_eq!(render(&fixture()), render(&fixture()));
    }

    #[test]
    fn a_frame_is_a_complete_svg_document() {
        let svg = render(&fixture());
        assert!(svg.starts_with("<svg xmlns=\"http://www.w3.org/2000/svg\""));
        assert!(svg.ends_with("</svg>\n"));
    }

    #[test]
    fn the_identity_orientation_points_forward() {
        let (ax, az) = forward_axis_xz([1.0, 0.0, 0.0, 0.0]);
        assert!((ax - 1.0).abs() < 1e-12);
        assert!(az.abs() < 1e-12);
    }

    #[test]
    fn a_half_turn_about_y_points_back_and_down_the_tick() {
        // Rotation of pi about y: w = cos(pi/2) = 0, y = sin(pi/2) = 1.
        let (ax, _) = forward_axis_xz([0.0, 0.0, 1.0, 0.0]);
        assert!((ax + 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_large_joint_angle_is_clamped_to_the_panel() {
        let svg = render(&fixture());
        // The 2.0 rad bar is below pi, so it is not clamped: width = 2/pi * 150.
        let expected = format!("{:.6}", 2.0 / JOINT_DISPLAY_LIMIT * BAR_HALF_WIDTH);
        assert!(svg.contains(&format!("width=\"{expected}\"")));
        let mut big = fixture();
        big.joint_angles = vec![100.0];
        let svg_big = render(&big);
        let full = format!("{:.6}", BAR_HALF_WIDTH);
        assert!(svg_big.contains(&format!("width=\"{full}\"")));
    }

    #[test]
    fn height_grows_with_the_joint_count() {
        assert!(height(3) > height(1));
    }
}

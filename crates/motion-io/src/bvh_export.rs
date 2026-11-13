//! BVH export, the "export" half of SPECS.md Section 6's "motion
//! import/export".
//!
//! The writer is the inverse of [`crate::bvh::parse`]: the root's rest
//! offset is its `OFFSET`, the frame's root translation its position
//! channels, and every joint's rotation is written as Euler angles in the
//! order its samples record as their `source_order` (ZXY, the most common
//! BVH order, when a channel has none). Joints are written depth-first, so
//! a record whose channels are not already in depth-first order comes back
//! from the parser reordered, with the same tree. Leaf joints get a zero
//! `End Site`, which the parser discards.
//!
//! Only a record the parser could have produced can be written: rotation
//! channels, a uniform frame interval, and no marker or scalar channels.

use core::fmt::{self, Write as _};
use robot_types::frames::{Axis, EulerOrder, Quat};
use robot_types::motion::{JointValue, MotionKind, MotionRecord};

/// Why a record could not be written as BVH.
#[derive(Clone, PartialEq, Debug)]
pub enum BvhWriteError {
    /// A channel holds markers or scalars, which BVH cannot carry.
    NotRotation {
        /// The channel.
        channel: usize,
    },
    /// Frame intervals differ by more than 1 % from the first.
    NonUniformFrames {
        /// The first frame whose interval differs.
        frame: usize,
    },
    /// The record has no frames, or one frame with no interval to state.
    TooShort,
    /// Channel 0 is not the only root.
    NotSingleRooted,
}

impl fmt::Display for BvhWriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotRotation { channel } => write!(f, "channel {channel} is not a rotation"),
            Self::NonUniformFrames { frame } => {
                write!(f, "frame interval changes at frame {frame}")
            }
            Self::TooShort => f.write_str("BVH needs at least two frames to state a frame time"),
            Self::NotSingleRooted => f.write_str("BVH needs exactly one root, at channel 0"),
        }
    }
}

impl core::error::Error for BvhWriteError {}

/// Intrinsic Euler angles `[a1, a2, a3]` with `q = R_a1 R_a2 R_a3`, for any
/// Tait-Bryan order; the middle angle lies in `[-pi/2, pi/2]`.
#[must_use]
pub fn quat_to_euler(q: Quat, order: EulerOrder) -> [f64; 3] {
    let m = q.to_rotation_matrix();
    let r = |i: usize, j: usize| m[(i, j)];
    let idx = |a: Axis| match a {
        Axis::X => 0,
        Axis::Y => 1,
        Axis::Z => 2,
    };
    let [a, b, c] = order.axes();
    let (i, j, k) = (idx(a), idx(b), idx(c));
    // +1 for the cyclic orders XYZ, YZX, ZXY.
    let s = if (j + 3 - i) % 3 == 1 { 1.0 } else { -1.0 };
    let beta = (s * r(i, k)).clamp(-1.0, 1.0).asin();
    if (1.0 - (s * r(i, k)).abs()) > 1e-12 {
        [
            (-s * r(j, k)).atan2(r(k, k)),
            beta,
            (-s * r(i, j)).atan2(r(i, i)),
        ]
    } else {
        // Gimbal lock: the first and third axes coincide; put it all in
        // the first angle.
        [(s * r(k, j)).atan2(r(j, j)), beta, 0.0]
    }
}

fn axis_name(a: Axis) -> &'static str {
    match a {
        Axis::X => "Xrotation",
        Axis::Y => "Yrotation",
        Axis::Z => "Zrotation",
    }
}

/// Writes `record` as BVH text.
pub fn write<K: MotionKind>(record: &MotionRecord<K>) -> Result<String, BvhWriteError> {
    let n = record.frames.len();
    if n < 2 {
        return Err(BvhWriteError::TooShort);
    }
    if record.channels.iter().skip(1).any(|c| c.parent.is_none())
        || record.channels[0].parent.is_some()
    {
        return Err(BvhWriteError::NotSingleRooted);
    }
    let dt = record.frames[1].timestamp.nanos() - record.frames[0].timestamp.nanos();
    if dt <= 0 {
        return Err(BvhWriteError::NonUniformFrames { frame: 1 });
    }
    if let Some(i) = record.frames.windows(2).position(|w| {
        let d = w[1].timestamp.nanos() - w[0].timestamp.nanos();
        (d - dt).abs() as f64 > 0.01 * dt as f64
    }) {
        return Err(BvhWriteError::NonUniformFrames { frame: i + 1 });
    }
    let orders: Vec<EulerOrder> = (0..record.channels.len())
        .map(|c| match &record.frames[0].values[c] {
            JointValue::Rotation { source_order, .. } => {
                Ok(source_order.unwrap_or(EulerOrder::Zxy))
            }
            _ => Err(BvhWriteError::NotRotation { channel: c }),
        })
        .collect::<Result<_, _>>()?;
    let children: Vec<Vec<usize>> = (0..record.channels.len())
        .map(|p| {
            (0..record.channels.len())
                .filter(|&c| record.channels[c].parent == Some(p))
                .collect()
        })
        .collect();

    let mut text = String::from("HIERARCHY\n");
    let mut order = Vec::new();
    write_joint(record, &children, &orders, 0, 0, &mut text, &mut order);
    let _ = writeln!(
        text,
        "MOTION\nFrames: {n}\nFrame Time: {:.9}",
        dt as f64 * 1e-9
    );
    for frame in &record.frames {
        let mut line = Vec::new();
        for &c in &order {
            let q = match &frame.values[c] {
                JointValue::Rotation { rotation, .. } => *rotation,
                _ => return Err(BvhWriteError::NotRotation { channel: c }),
            };
            if c == 0 {
                let t = frame.root.translation;
                line.extend([t.x, t.y, t.z]);
            }
            line.extend(quat_to_euler(q, orders[c]).map(f64::to_degrees));
        }
        let strs: Vec<String> = line.iter().map(|v| format!("{v:.9}")).collect();
        let _ = writeln!(text, "{}", strs.join(" "));
    }
    Ok(text)
}

fn write_joint<K: MotionKind>(
    record: &MotionRecord<K>,
    children: &[Vec<usize>],
    orders: &[EulerOrder],
    c: usize,
    depth: usize,
    text: &mut String,
    order: &mut Vec<usize>,
) {
    let pad = "  ".repeat(depth);
    let channel = &record.channels[c];
    let kind = if c == 0 { "ROOT" } else { "JOINT" };
    let o = channel.offset;
    let rot: Vec<&str> = orders[c].axes().into_iter().map(axis_name).collect();
    let _ = writeln!(text, "{pad}{kind} {}\n{pad}{{", channel.name);
    let _ = writeln!(text, "{pad}  OFFSET {:.9} {:.9} {:.9}", o.x, o.y, o.z);
    if c == 0 {
        let _ = writeln!(
            text,
            "{pad}  CHANNELS 6 Xposition Yposition Zposition {}",
            rot.join(" ")
        );
    } else {
        let _ = writeln!(text, "{pad}  CHANNELS 3 {}", rot.join(" "));
    }
    order.push(c);
    if children[c].is_empty() {
        let _ = writeln!(
            text,
            "{pad}  End Site\n{pad}  {{\n{pad}    OFFSET 0 0 0\n{pad}  }}"
        );
    }
    for &child in &children[c] {
        write_joint(record, children, orders, child, depth + 1, text, order);
    }
    let _ = writeln!(text, "{pad}}}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bvh::parse;
    use robot_types::frames::Vec3;
    use robot_types::motion::SourceLicense;
    use robot_types::units::Radians;

    const ORDERS: [EulerOrder; 6] = [
        EulerOrder::Xyz,
        EulerOrder::Xzy,
        EulerOrder::Yxz,
        EulerOrder::Yzx,
        EulerOrder::Zxy,
        EulerOrder::Zyx,
    ];

    #[test]
    fn euler_decomposition_inverts_composition_in_every_order() {
        let angles = [0.3, -1.1, 2.2];
        for order in ORDERS {
            let q = order.to_quat(angles.map(Radians)).unwrap();
            let back = quat_to_euler(q, order);
            let q2 = order.to_quat(back.map(Radians)).unwrap();
            assert!(q.angle_to(q2).value() < 1e-6, "{order:?}: {back:?}");
            for (a, b) in angles.iter().zip(back) {
                assert!((a - b).abs() < 1e-9, "{order:?}: {back:?}");
            }
        }
    }

    #[test]
    fn gimbal_lock_still_reproduces_the_rotation() {
        for order in ORDERS {
            let q = order
                .to_quat([
                    Radians(0.4),
                    Radians(core::f64::consts::FRAC_PI_2),
                    Radians(0.7),
                ])
                .unwrap();
            let back = quat_to_euler(q, order);
            let q2 = order.to_quat(back.map(Radians)).unwrap();
            assert!(q.angle_to(q2).value() < 1e-6, "{order:?}");
        }
    }

    const SOURCE: &str = "HIERARCHY
ROOT Hips
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Spine
  {
    OFFSET 0 10 0
    CHANNELS 3 Zrotation Yrotation Xrotation
    End Site
    {
      OFFSET 0 5 0
    }
  }
  JOINT Leg
  {
    OFFSET 3 -8 0
    CHANNELS 3 Xrotation Yrotation Zrotation
    End Site
    {
      OFFSET 0 -10 0
    }
  }
}
MOTION
Frames: 3
Frame Time: 0.0333333
0 90 0 10 20 30 5 -15 25 40 0 -60
1 91 2 12 18 33 0 0 0 -40 50 60
2 92 4 -170 80 -20 1 2 3 4 5 6
";

    #[test]
    fn a_parsed_file_written_and_parsed_again_is_the_same_motion() {
        let a = parse(
            SOURCE,
            "sha256:x",
            SourceLicense::unaudited("fixture-internal"),
        )
        .unwrap();
        let text = write(&a).unwrap();
        let b = parse(
            &text,
            "sha256:x",
            SourceLicense::unaudited("fixture-internal"),
        )
        .unwrap();
        assert_eq!(a.channels, b.channels);
        assert_eq!(a.frames.len(), b.frames.len());
        for (fa, fb) in a.frames.iter().zip(&b.frames) {
            assert!((fa.timestamp.nanos() - fb.timestamp.nanos()).abs() <= 1);
            assert!((fa.root.translation - fb.root.translation).norm() < 1e-6);
            for (va, vb) in fa.values.iter().zip(&fb.values) {
                let (
                    JointValue::Rotation {
                        rotation: qa,
                        source_order: oa,
                    },
                    JointValue::Rotation {
                        rotation: qb,
                        source_order: ob,
                    },
                ) = (va, vb)
                else {
                    panic!()
                };
                assert_eq!(oa, ob);
                assert!(qa.angle_to(*qb).value() < 1e-6);
            }
        }
        assert_eq!(b.frames[0].root.translation, Vec3::new(0.0, 90.0, 0.0));
    }

    #[test]
    fn markers_cannot_be_written() {
        let mut r = parse(
            SOURCE,
            "sha256:x",
            SourceLicense::unaudited("fixture-internal"),
        )
        .unwrap();
        r.frames[0].values[1] = JointValue::Position {
            position: Vec3::ZERO,
        };
        assert_eq!(write(&r), Err(BvhWriteError::NotRotation { channel: 1 }));
    }
}

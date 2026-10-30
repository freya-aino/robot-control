//! A BVH parser, with per-joint channel order preserved per SPECS.md
//! Section 7.2's warning: "Euler-angle conventions are source-specific;
//! never assume every BVH file uses the same rotation order."
//!
//! PLAN.md's first executable backlog lists "Implement BVH inspection and
//! canonical conversion" as item 5, immediately after the typed contracts —
//! this is that step. Per
//! `docs/decisions/0003-std-only-default-closure.md`, this is an authored
//! parser standing in for Appendix B.2.4's `bvh_anim`/`bvh_anim_parser`
//! candidates, both still `unverified`.
//!
//! Scope: the hierarchy block (`ROOT`/`JOINT`/`End Site`, `OFFSET`,
//! `CHANNELS`) and the motion block. Only the root joint may carry position
//! channels — every other joint declaring one is rejected by name rather
//! than silently reinterpreted, since the canonical schema's single
//! per-frame `root` transform has nowhere else to put it.

use core::fmt;
use robot_types::frames::{Axis, EulerOrder, Quat, Transform, Vec3};
use robot_types::ids::{Name, NameError};
use robot_types::motion::{
    Channel, HumanSkeleton, JointValue, MotionError, MotionFrame, MotionRecord, SourceLicense,
    Validity,
};
use robot_types::schema::SchemaError;
use robot_types::time::{ClockId, Duration, Timestamp};

/// One channel declared on a BVH joint.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ChannelKind {
    XPos,
    YPos,
    ZPos,
    XRot,
    YRot,
    ZRot,
}

impl ChannelKind {
    fn parse(text: &str) -> Option<Self> {
        match text {
            "Xposition" => Some(Self::XPos),
            "Yposition" => Some(Self::YPos),
            "Zposition" => Some(Self::ZPos),
            "Xrotation" => Some(Self::XRot),
            "Yrotation" => Some(Self::YRot),
            "Zrotation" => Some(Self::ZRot),
            _ => None,
        }
    }

    const fn is_rotation(self) -> bool {
        matches!(self, Self::XRot | Self::YRot | Self::ZRot)
    }

    const fn axis(self) -> Axis {
        match self {
            Self::XPos | Self::XRot => Axis::X,
            Self::YPos | Self::YRot => Axis::Y,
            Self::ZPos | Self::ZRot => Axis::Z,
        }
    }
}

struct BvhNode {
    name: String,
    offset: Vec3,
    channels: Vec<ChannelKind>,
    children: Vec<BvhNode>,
}

/// Why BVH parsing failed.
#[derive(Clone, PartialEq, Debug)]
pub enum BvhError {
    /// A syntax error at a line number (1-based).
    Syntax {
        /// What was expected.
        expected: &'static str,
        /// Line number.
        line: usize,
    },
    /// A construct outside the supported subset.
    Unsupported {
        /// The construct, e.g. `"position channel on a non-root joint"`.
        construct: String,
        /// Line number.
        line: usize,
    },
    /// The motion block's frame data did not match the declared channel
    /// count.
    ChannelCountMismatch {
        /// Frame index.
        frame: usize,
        /// Channels declared by the hierarchy.
        expected: usize,
        /// Values found on that frame's line.
        found: usize,
    },
    /// A declared size exceeds the importer's malformed-file limit
    /// (SPECS.md Section 8.1: "malformed-file limits"). The limit is checked
    /// before any allocation sized by the declaration.
    Limit {
        /// What was declared, e.g. `"frames"`.
        what: &'static str,
        /// The limit.
        limit: usize,
        /// The declared value.
        found: usize,
    },
    /// A name in the hierarchy failed the project's name rules.
    Name(NameError),
    /// The canonical record rejected the converted data.
    Motion(MotionError),
}

/// The most frames a BVH file may declare: about 2.3 hours at 120 Hz.
pub const MAX_FRAMES: usize = 1_000_000;
/// The most motion channels a BVH hierarchy may declare.
pub const MAX_CHANNELS: usize = 4096;

impl fmt::Display for BvhError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syntax { expected, line } => write!(f, "expected {expected} at line {line}"),
            Self::Unsupported { construct, line } => {
                write!(
                    f,
                    "{construct} at line {line} is outside the supported BVH subset"
                )
            }
            Self::ChannelCountMismatch {
                frame,
                expected,
                found,
            } => {
                write!(
                    f,
                    "frame {frame} has {found} values, hierarchy declares {expected} channels"
                )
            }
            Self::Limit { what, limit, found } => {
                write!(
                    f,
                    "declared {found} {what}, more than the importer limit of {limit}"
                )
            }
            Self::Name(e) => write!(f, "{e}"),
            Self::Motion(e) => write!(f, "{e}"),
        }
    }
}

impl core::error::Error for BvhError {}

impl From<NameError> for BvhError {
    fn from(e: NameError) -> Self {
        Self::Name(e)
    }
}
impl From<MotionError> for BvhError {
    fn from(e: MotionError) -> Self {
        Self::Motion(e)
    }
}
impl From<SchemaError> for BvhError {
    fn from(e: SchemaError) -> Self {
        Self::Motion(MotionError::Schema(e))
    }
}

struct LineParser<'a> {
    lines: core::iter::Peekable<core::slice::Iter<'a, &'a str>>,
    line_no: usize,
}

impl<'a> LineParser<'a> {
    fn next_nonblank(&mut self) -> Option<(&'a str, usize)> {
        loop {
            let line = self.lines.next()?;
            self.line_no += 1;
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                return Some((trimmed, self.line_no));
            }
        }
    }

    fn peek_nonblank(&mut self) -> Option<&'a str> {
        loop {
            {
                let line = self.lines.peek()?;
                if line.trim().is_empty() {
                    self.lines.next();
                    self.line_no += 1;
                } else {
                    return Some(line.trim());
                }
            }
        }
    }
}

/// Parses a BVH document's text into the canonical human-skeleton motion
/// record.
pub fn parse(
    text: &str,
    source_hash: &str,
    license: SourceLicense,
) -> Result<MotionRecord<HumanSkeleton>, BvhError> {
    let lines: Vec<&str> = text.lines().collect();
    let mut p = LineParser {
        lines: lines.iter().peekable(),
        line_no: 0,
    };

    let (header, line) = p.next_nonblank().ok_or(BvhError::Syntax {
        expected: "HIERARCHY",
        line: 0,
    })?;
    if header != "HIERARCHY" {
        return Err(BvhError::Syntax {
            expected: "HIERARCHY",
            line,
        });
    }

    let (root_line, line) = p.next_nonblank().ok_or(BvhError::Syntax {
        expected: "ROOT",
        line,
    })?;
    let root_name = root_line
        .strip_prefix("ROOT ")
        .ok_or(BvhError::Syntax {
            expected: "ROOT <name>",
            line,
        })?
        .trim()
        .to_owned();
    let root = parse_joint_body(&mut p, root_name)?;

    let (motion_header, line) = p.next_nonblank().ok_or(BvhError::Syntax {
        expected: "MOTION",
        line,
    })?;
    if motion_header != "MOTION" {
        return Err(BvhError::Syntax {
            expected: "MOTION",
            line,
        });
    }
    let (frames_line, line) = p.next_nonblank().ok_or(BvhError::Syntax {
        expected: "Frames:",
        line,
    })?;
    let frame_count: usize = frames_line
        .strip_prefix("Frames:")
        .ok_or(BvhError::Syntax {
            expected: "Frames:",
            line,
        })?
        .trim()
        .parse()
        .map_err(|_| BvhError::Syntax {
            expected: "an integer frame count",
            line,
        })?;
    let (time_line, line) = p.next_nonblank().ok_or(BvhError::Syntax {
        expected: "Frame Time:",
        line,
    })?;
    let frame_time_s: f64 = time_line
        .strip_prefix("Frame Time:")
        .ok_or(BvhError::Syntax {
            expected: "Frame Time:",
            line,
        })?
        .trim()
        .parse()
        .map_err(|_| BvhError::Syntax {
            expected: "a frame time",
            line,
        })?;
    let dt = Duration::from_secs_f64(frame_time_s);

    // Flatten the hierarchy depth-first, root first, rejecting a
    // non-root position channel explicitly rather than misplacing it.
    let mut flat: Vec<(&BvhNode, Option<usize>)> = Vec::new();
    flatten(&root, None, &mut flat);
    for (node, _) in &flat[1..] {
        if node.channels.iter().any(|c| !c.is_rotation()) {
            return Err(BvhError::Unsupported {
                construct: format!("position channel on non-root joint {:?}", node.name),
                line: 0,
            });
        }
    }

    let total_channels: usize = flat.iter().map(|(n, _)| n.channels.len()).sum();
    if total_channels > MAX_CHANNELS {
        return Err(BvhError::Limit {
            what: "channels",
            limit: MAX_CHANNELS,
            found: total_channels,
        });
    }
    if frame_count > MAX_FRAMES {
        return Err(BvhError::Limit {
            what: "frames",
            limit: MAX_FRAMES,
            found: frame_count,
        });
    }
    let channels: Vec<Channel> = flat
        .iter()
        .map(|(n, parent)| -> Result<Channel, BvhError> {
            Ok(Channel {
                name: Name::new(&n.name)?,
                parent: *parent,
                offset: n.offset,
            })
        })
        .collect::<Result<_, _>>()?;

    let mut frames = Vec::with_capacity(frame_count);
    for frame_index in 0..frame_count {
        let (data_line, _) = p.next_nonblank().ok_or(BvhError::Syntax {
            expected: "a motion data line",
            line: p.line_no,
        })?;
        let values: Vec<f64> = data_line
            .split_whitespace()
            .map(|t| t.parse::<f64>())
            .collect::<Result<_, _>>()
            .map_err(|_| BvhError::Syntax {
                expected: "a float",
                line: p.line_no,
            })?;
        if values.len() != total_channels {
            return Err(BvhError::ChannelCountMismatch {
                frame: frame_index,
                expected: total_channels,
                found: values.len(),
            });
        }

        let mut cursor = 0usize;
        let mut root_translation = Vec3::ZERO;
        let mut root_rotation = Quat::IDENTITY;
        let mut joint_values = Vec::with_capacity(flat.len());
        for (node_index, (node, _)) in flat.iter().enumerate() {
            let mut translation = Vec3::ZERO;
            let mut rot_axes = Vec::with_capacity(3);
            let mut rot_values = Vec::with_capacity(3);
            for &kind in &node.channels {
                let v = values[cursor];
                cursor += 1;
                if kind.is_rotation() {
                    rot_axes.push(kind.axis());
                    rot_values.push(robot_types::units::Radians::from_degrees(v));
                } else {
                    match kind {
                        ChannelKind::XPos => translation.x = v,
                        ChannelKind::YPos => translation.y = v,
                        ChannelKind::ZPos => translation.z = v,
                        _ => unreachable!(),
                    }
                }
            }
            let rotation = if rot_axes.is_empty() {
                Quat::IDENTITY
            } else if rot_axes.len() == 3 {
                let order = EulerOrder::from_axes([rot_axes[0], rot_axes[1], rot_axes[2]]).ok_or(
                    BvhError::Unsupported {
                        construct: "repeated rotation axis in CHANNELS".to_owned(),
                        line: 0,
                    },
                )?;
                order
                    .to_quat([rot_values[0], rot_values[1], rot_values[2]])
                    .map_err(|_| BvhError::Unsupported {
                        construct: "degenerate rotation channels".to_owned(),
                        line: 0,
                    })?
            } else {
                return Err(BvhError::Unsupported {
                    construct: format!("{}-axis partial rotation channels", rot_axes.len()),
                    line: 0,
                });
            };
            let order3 = if rot_axes.len() == 3 {
                EulerOrder::from_axes([rot_axes[0], rot_axes[1], rot_axes[2]])
            } else {
                None
            };
            if node_index == 0 {
                // The root's rotation is the root pose's rotation (SPECS.md
                // Section 7.2: one root pose per frame). It is also kept as
                // channel 0's value below, so the source channel order and
                // count are preserved; the two are the same quantity.
                root_rotation = rotation;
            }
            if !node.channels.iter().any(|c| !c.is_rotation()) {
                joint_values.push(JointValue::Rotation {
                    rotation,
                    source_order: order3,
                });
            } else {
                root_translation = translation;
                joint_values.push(JointValue::Rotation {
                    rotation,
                    source_order: order3,
                });
            }
        }

        let timestamp = Timestamp::sampling(
            dt.saturating_mul(frame_index as i64).nanos(),
            ClockId::Source(0),
        );
        frames.push(MotionFrame {
            timestamp,
            root: Transform::new(
                robot_types::frames::FrameId::WORLD,
                robot_types::frames::FrameId::WORLD,
                root_rotation,
                root_translation,
            ),
            validity: vec![Validity::Valid; joint_values.len()],
            values: joint_values,
            contacts: Vec::new(),
        });
    }

    let record = MotionRecord::new(
        HumanSkeleton,
        source_hash,
        license,
        "bvh-import-session",
        "bvh-skeleton",
        robot_types::frames::FrameId::WORLD,
        ClockId::Source(0),
        channels,
        frames,
    )?;
    Ok(record)
}

fn flatten<'a>(
    node: &'a BvhNode,
    parent: Option<usize>,
    out: &mut Vec<(&'a BvhNode, Option<usize>)>,
) {
    let my_index = out.len();
    out.push((node, parent));
    for child in &node.children {
        flatten(child, Some(my_index), out);
    }
}

fn parse_joint_body(p: &mut LineParser<'_>, name: String) -> Result<BvhNode, BvhError> {
    let (open, line) = p.next_nonblank().ok_or(BvhError::Syntax {
        expected: "'{'",
        line: 0,
    })?;
    if open != "{" {
        return Err(BvhError::Syntax {
            expected: "'{'",
            line,
        });
    }

    let (offset_line, line) = p.next_nonblank().ok_or(BvhError::Syntax {
        expected: "OFFSET",
        line,
    })?;
    let offset = parse_offset(offset_line, line)?;

    let mut channels = Vec::new();
    let mut children = Vec::new();

    if let Some(peek) = p.peek_nonblank() {
        if peek.starts_with("CHANNELS") {
            let (channels_line, line) = p.next_nonblank().expect("peeked");
            channels = parse_channels(channels_line, line)?;
        }
    }

    loop {
        let Some(peek) = p.peek_nonblank() else {
            return Err(BvhError::Syntax {
                expected: "'}' or JOINT/End Site",
                line: p.line_no,
            });
        };
        if peek == "}" {
            p.next_nonblank();
            break;
        }
        if let Some(joint_name) = peek.strip_prefix("JOINT ") {
            let joint_name = joint_name.trim().to_owned();
            p.next_nonblank();
            children.push(parse_joint_body(p, joint_name)?);
        } else if peek.starts_with("End Site") {
            p.next_nonblank();
            let (open, line) = p.next_nonblank().ok_or(BvhError::Syntax {
                expected: "'{'",
                line: p.line_no,
            })?;
            if open != "{" {
                return Err(BvhError::Syntax {
                    expected: "'{'",
                    line,
                });
            }
            let (_offset_line, _) = p.next_nonblank().ok_or(BvhError::Syntax {
                expected: "OFFSET",
                line,
            })?;
            let (close, line) = p.next_nonblank().ok_or(BvhError::Syntax {
                expected: "'}'",
                line,
            })?;
            if close != "}" {
                return Err(BvhError::Syntax {
                    expected: "'}'",
                    line,
                });
            }
            // End Site contributes no channel and is not stored as a joint.
        } else {
            return Err(BvhError::Syntax {
                expected: "'}' or JOINT/End Site",
                line: p.line_no,
            });
        }
    }

    Ok(BvhNode {
        name,
        offset,
        channels,
        children,
    })
}

fn parse_offset(line: &str, line_no: usize) -> Result<Vec3, BvhError> {
    let rest = line.strip_prefix("OFFSET").ok_or(BvhError::Syntax {
        expected: "OFFSET",
        line: line_no,
    })?;
    let parts: Vec<f64> = rest
        .split_whitespace()
        .map(|t| t.parse::<f64>())
        .collect::<Result<_, _>>()
        .map_err(|_| BvhError::Syntax {
            expected: "three OFFSET floats",
            line: line_no,
        })?;
    if parts.len() != 3 {
        return Err(BvhError::Syntax {
            expected: "three OFFSET floats",
            line: line_no,
        });
    }
    // BVH offsets are centimetres by the most common convention but the
    // unit is not self-describing in the file; this importer treats the
    // numeric value as metres-equivalent in the source's own units and
    // leaves unit reconciliation to the caller's provenance record, per
    // SPECS.md Section 7.2's requirement to preserve source-specific
    // conventions rather than silently assume one.
    Ok(Vec3::new(parts[0], parts[1], parts[2]))
}

fn parse_channels(line: &str, line_no: usize) -> Result<Vec<ChannelKind>, BvhError> {
    let rest = line.strip_prefix("CHANNELS").ok_or(BvhError::Syntax {
        expected: "CHANNELS",
        line: line_no,
    })?;
    let mut tokens = rest.split_whitespace();
    let count: usize = tokens
        .next()
        .ok_or(BvhError::Syntax {
            expected: "channel count",
            line: line_no,
        })?
        .parse()
        .map_err(|_| BvhError::Syntax {
            expected: "a channel count",
            line: line_no,
        })?;
    let kinds: Vec<ChannelKind> = tokens
        .map(|t| {
            ChannelKind::parse(t).ok_or(BvhError::Unsupported {
                construct: format!("channel {t:?}"),
                line: line_no,
            })
        })
        .collect::<Result<_, _>>()?;
    if kinds.len() != count {
        return Err(BvhError::Syntax {
            expected: "channel count to match listed channels",
            line: line_no,
        });
    }
    Ok(kinds)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn license() -> SourceLicense {
        SourceLicense {
            identifier: "fixture-internal".to_owned(),
            training_permitted: true,
            redistribution_permitted: false,
            commercial_permitted: false,
            retention: None,
            source_reference: "fixtures/motion/README.md".to_owned(),
        }
    }

    const SIMPLE_BVH: &str = "HIERARCHY
ROOT Hips
{
	OFFSET 0.00 0.00 0.00
	CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
	JOINT Spine
	{
		OFFSET 0.00 10.00 0.00
		CHANNELS 3 Zrotation Xrotation Yrotation
		End Site
		{
			OFFSET 0.00 5.00 0.00
		}
	}
}
MOTION
Frames: 2
Frame Time: 0.033333
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
1.0 2.0 3.0 90.0 0.0 0.0 0.0 0.0 0.0
";

    #[test]
    fn parses_hierarchy_and_two_frames() {
        let record = parse(SIMPLE_BVH, "sha256:abc", license()).unwrap();
        assert_eq!(record.channels.len(), 2); // Hips, Spine (End Site excluded)
        assert_eq!(record.channels[0].name.as_str(), "Hips");
        assert_eq!(record.channels[1].name.as_str(), "Spine");
        assert_eq!(record.channels[1].parent, Some(0));
        assert!((record.channels[1].offset.y - 10.0).abs() < 1e-9);
        assert_eq!(record.frame_count(), 2);
    }

    #[test]
    fn root_translation_lands_in_the_frame_root_not_a_joint_value() {
        let record = parse(SIMPLE_BVH, "sha256:abc", license()).unwrap();
        let second = &record.frames[1];
        assert!((second.root.translation.x - 1.0).abs() < 1e-9);
        assert!((second.root.translation.y - 2.0).abs() < 1e-9);
        assert!((second.root.translation.z - 3.0).abs() < 1e-9);
    }

    #[test]
    fn root_rotation_lands_in_the_frame_root_as_well_as_channel_zero() {
        let record = parse(SIMPLE_BVH, "sha256:abc", license()).unwrap();
        let second = &record.frames[1];
        // 90 degrees about Z: +X goes to +Y, through the root pose itself.
        let v = second.root.rotation.rotate(Vec3::unit_x());
        assert!((v.y - 1.0).abs() < 1e-6, "{v:?}");
        let JointValue::Rotation { rotation, .. } = &second.values[0] else {
            panic!("expected a rotation value");
        };
        assert_eq!(*rotation, second.root.rotation);
        assert_eq!(record.frames[0].root.rotation, Quat::IDENTITY);
    }

    #[test]
    fn a_frame_count_past_the_limit_is_rejected_before_allocation() {
        let huge = SIMPLE_BVH.replace("Frames: 2", "Frames: 99999999999");
        assert!(matches!(
            parse(&huge, "sha256:abc", license()),
            Err(BvhError::Limit { what: "frames", .. })
        ));
    }

    #[test]
    fn rotation_channel_order_is_preserved_as_zxy() {
        let record = parse(SIMPLE_BVH, "sha256:abc", license()).unwrap();
        let JointValue::Rotation {
            source_order,
            rotation,
        } = &record.frames[1].values[0]
        else {
            panic!("expected a rotation value");
        };
        assert_eq!(*source_order, Some(EulerOrder::Zxy));
        // 90 degrees about Z should rotate +X to +Y.
        let v = rotation.rotate(Vec3::unit_x());
        assert!((v.y - 1.0).abs() < 1e-6, "{v:?}");
    }

    #[test]
    fn a_channel_count_mismatch_is_rejected() {
        let bad = SIMPLE_BVH.replace("1.0 2.0 3.0 90.0 0.0 0.0 0.0 0.0 0.0\n", "1.0 2.0\n");
        assert!(matches!(
            parse(&bad, "sha256:abc", license()),
            Err(BvhError::ChannelCountMismatch { .. })
        ));
    }

    #[test]
    fn a_position_channel_on_a_non_root_joint_is_rejected_by_name() {
        let bad = SIMPLE_BVH.replace(
            "CHANNELS 3 Zrotation Xrotation Yrotation",
            "CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation",
        );
        let bad = bad.replace(
            "0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0",
            "0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0",
        );
        let bad = bad.replace(
            "1.0 2.0 3.0 90.0 0.0 0.0 0.0 0.0 0.0",
            "1.0 2.0 3.0 90.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0",
        );
        assert!(matches!(
            parse(&bad, "sha256:abc", license()),
            Err(BvhError::Unsupported { .. })
        ));
    }

    #[test]
    fn an_unknown_channel_name_is_rejected() {
        let bad = SIMPLE_BVH.replace(
            "Zrotation Xrotation Yrotation",
            "Wrotation Xrotation Yrotation",
        );
        assert!(matches!(
            parse(&bad, "sha256:abc", license()),
            Err(BvhError::Unsupported { .. })
        ));
    }
}

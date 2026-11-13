//! AMASS import: SMPL-family body-model parameters from an `.npz` file
//! (SPECS.md Section 8.1's "supported numerical human-body formats"; AMASS
//! is the first corpus Section 8.1 lists).
//!
//! An AMASS sequence stores, per frame, one axis-angle rotation per joint
//! of the body model (`poses`), a root translation in metres (`trans`), and
//! once per sequence the shape coefficients (`betas`), the subject gender
//! and the capture rate (`mocap_framerate`, or `mocap_frame_rate` in the
//! SMPL-X releases). The width of `poses` says which model it is: 72 for
//! SMPL (24 joints), 156 for SMPL-H (52) and 165 for SMPL-X (55).
//!
//! The record is a [`BodyModelParameters`] record with one rotation channel
//! per joint, named and parented as the model's public kinematic tree. The
//! rest offsets are **zero**: a joint's rest position comes from the body
//! model's template and joint regressor evaluated at `betas`, and those
//! model files are licensed separately from AMASS and are not part of this
//! repository. Forward kinematics on this record is therefore meaningless
//! until a caller supplies offsets from a model it is licensed to use; the
//! importer says so in the record's provenance rather than inventing a
//! skeleton. The coordinate frame is the dataset's own (AMASS is
//! Z-up), recorded as-is and not converted.

use core::fmt;
use robot_types::frames::{FrameId, Quat, Transform, Vec3};
use robot_types::ids::Name;
use robot_types::motion::{
    BodyModelParameters, Channel, JointValue, MotionError, MotionFrame, MotionRecord,
    ProvenanceStep, SourceLicense, Validity,
};
use robot_types::time::{ClockId, Timestamp};

use crate::npz::{read_npz, NpyArray, NpzError};

/// Which body model a pose vector belongs to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BodyModel {
    /// SMPL, 24 joints.
    Smpl,
    /// SMPL+H, 52 joints.
    SmplH,
    /// SMPL-X, 55 joints.
    SmplX,
}

const BODY: [(&str, i8); 22] = [
    ("pelvis", -1),
    ("left_hip", 0),
    ("right_hip", 0),
    ("spine1", 0),
    ("left_knee", 1),
    ("right_knee", 2),
    ("spine2", 3),
    ("left_ankle", 4),
    ("right_ankle", 5),
    ("spine3", 6),
    ("left_foot", 7),
    ("right_foot", 8),
    ("neck", 9),
    ("left_collar", 9),
    ("right_collar", 9),
    ("head", 12),
    ("left_shoulder", 13),
    ("right_shoulder", 14),
    ("left_elbow", 16),
    ("right_elbow", 17),
    ("left_wrist", 18),
    ("right_wrist", 19),
];

const FINGERS: [&str; 5] = ["index", "middle", "pinky", "ring", "thumb"];

impl BodyModel {
    /// The model with this many pose parameters.
    #[must_use]
    pub const fn from_pose_width(width: usize) -> Option<Self> {
        match width {
            72 => Some(Self::Smpl),
            156 => Some(Self::SmplH),
            165 => Some(Self::SmplX),
            _ => None,
        }
    }

    /// Joint names and parent indices, in pose-vector order.
    #[must_use]
    pub fn joints(self) -> Vec<(String, Option<usize>)> {
        let mut out: Vec<(String, Option<usize>)> = BODY
            .iter()
            .map(|(n, p)| ((*n).to_owned(), usize::try_from(*p).ok()))
            .collect();
        let hands = |out: &mut Vec<(String, Option<usize>)>| {
            for (side, wrist) in [("left", 20), ("right", 21)] {
                for finger in FINGERS {
                    for k in 1..=3 {
                        let parent = if k == 1 { wrist } else { out.len() - 1 };
                        out.push((format!("{side}_{finger}{k}"), Some(parent)));
                    }
                }
            }
        };
        match self {
            Self::Smpl => {
                out.push(("left_hand".into(), Some(20)));
                out.push(("right_hand".into(), Some(21)));
            }
            Self::SmplH => hands(&mut out),
            Self::SmplX => {
                out.push(("jaw".into(), Some(15)));
                out.push(("left_eye_smplhf".into(), Some(15)));
                out.push(("right_eye_smplhf".into(), Some(15)));
                hands(&mut out);
            }
        }
        out
    }
}

/// Why an AMASS file was refused.
#[derive(Clone, PartialEq, Debug)]
pub enum AmassError {
    /// The archive or an array could not be read.
    Npz(NpzError),
    /// A required array was missing or had the wrong shape.
    Field(String),
    /// The record was rejected.
    Motion(MotionError),
}

impl fmt::Display for AmassError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Npz(e) => write!(f, "{e}"),
            Self::Field(e) => write!(f, "AMASS field: {e}"),
            Self::Motion(e) => write!(f, "{e}"),
        }
    }
}

impl core::error::Error for AmassError {}

/// An imported sequence.
#[derive(Clone, PartialEq, Debug)]
pub struct AmassImport {
    /// The parameter record.
    pub record: MotionRecord<BodyModelParameters>,
    /// The body model.
    pub model: BodyModel,
    /// Shape coefficients.
    pub betas: Vec<f64>,
    /// Subject gender, as stored.
    pub gender: Option<String>,
    /// Capture rate, Hz.
    pub rate: f64,
}

/// Imports an AMASS `.npz`.
pub fn parse(
    bytes: &[u8],
    source_hash: &str,
    license: SourceLicense,
) -> Result<AmassImport, AmassError> {
    let arrays = read_npz(bytes).map_err(AmassError::Npz)?;
    let get = |k: &str| arrays.iter().find(|(n, _)| n == k).map(|(_, a)| a);
    let field = |k: &str| AmassError::Field(k.to_owned());
    let poses: &NpyArray = get("poses").ok_or_else(|| field("poses missing"))?;
    let trans: &NpyArray = get("trans").ok_or_else(|| field("trans missing"))?;
    let rate = get("mocap_framerate")
        .or_else(|| get("mocap_frame_rate"))
        .and_then(|a| a.numbers().and_then(|v| v.first().copied()))
        .ok_or_else(|| field("mocap_framerate missing"))?;
    if !(rate > 0.0 && rate.is_finite()) {
        return Err(field("mocap_framerate must be positive"));
    }
    let [frames, width] = poses.shape[..] else {
        return Err(field("poses must be 2-D"));
    };
    let model =
        BodyModel::from_pose_width(width).ok_or_else(|| field(&format!("poses width {width}")))?;
    if trans.shape != [frames, 3] {
        return Err(field("trans must be frames x 3"));
    }
    let p = poses
        .numbers()
        .ok_or_else(|| field("poses must be numeric"))?;
    let t = trans
        .numbers()
        .ok_or_else(|| field("trans must be numeric"))?;
    let joints = model.joints();
    let channels: Vec<Channel> = joints
        .iter()
        .map(|(n, parent)| {
            Ok(Channel {
                name: Name::new(n).map_err(|e| field(&e.to_string()))?,
                parent: *parent,
                offset: Vec3::ZERO,
            })
        })
        .collect::<Result<_, AmassError>>()?;
    let dt = 1e9 / rate;
    let out_frames: Vec<MotionFrame> = (0..frames)
        .map(|f| {
            let rot = |j: usize| {
                let a = &p[f * width + 3 * j..f * width + 3 * j + 3];
                Quat::from_rotation_vector(Vec3::new(a[0], a[1], a[2]))
            };
            let values: Vec<JointValue> = (0..joints.len())
                .map(|j| JointValue::Rotation {
                    rotation: rot(j),
                    source_order: None,
                })
                .collect();
            let valid = values.iter().all(
                |v| matches!(v, JointValue::Rotation { rotation, .. } if rotation.w.is_finite()),
            );
            MotionFrame {
                timestamp: Timestamp::sampling((f as f64 * dt).round() as i64, ClockId::Source(0)),
                root: Transform::new(
                    FrameId::WORLD,
                    FrameId::WORLD,
                    rot(0),
                    Vec3::new(t[3 * f], t[3 * f + 1], t[3 * f + 2]),
                ),
                validity: vec![
                    if valid {
                        Validity::Valid
                    } else {
                        Validity::Missing
                    };
                    values.len()
                ],
                values,
                contacts: Vec::new(),
            }
        })
        .collect();
    let betas = get("betas")
        .and_then(NpyArray::numbers)
        .map(<[f64]>::to_vec)
        .unwrap_or_default();
    let gender = get("gender").and_then(|a| a.string().map(str::to_owned));
    let mut record = MotionRecord::new(
        BodyModelParameters,
        source_hash,
        license,
        "amass-import-session",
        format!("{model:?}"),
        FrameId::WORLD,
        ClockId::Source(0),
        channels,
        out_frames,
    )
    .map_err(AmassError::Motion)?;
    record.record_step(
        ProvenanceStep::new(
            "import_amass",
            concat!("motion-io/", env!("CARGO_PKG_VERSION")),
        )
        .with_parameter("model", format!("{model:?}"))
        .with_parameter("rate_hz", rate.to_string())
        .with_parameter("rest_offsets", "zero: body-model template not available")
        .with_parameter("frame", "dataset (Z-up), not converted"),
    );
    Ok(AmassImport {
        record,
        model,
        betas,
        gender,
        rate,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::npz::tests::{npy, npy_f8, zip_stored};

    fn sequence(width: usize, frames: usize) -> Vec<u8> {
        let mut poses = vec![0.0; frames * width];
        poses[3 * 18 + 2] = 0.5; // left elbow about z in frame 0
        let trans: Vec<f64> = (0..frames)
            .flat_map(|f| [0.1 * f as f64, 0.0, 0.9])
            .collect();
        let g: Vec<u8> = "male"
            .chars()
            .flat_map(|c| (c as u32).to_le_bytes())
            .collect();
        zip_stored(&[
            ("poses.npy", npy_f8(&[frames, width], &poses)),
            ("trans.npy", npy_f8(&[frames, 3], &trans)),
            ("betas.npy", npy_f8(&[16], &[0.0; 16])),
            ("mocap_framerate.npy", npy_f8(&[], &[120.0])),
            ("gender.npy", npy("<U4", "()", &g)),
        ])
    }

    #[test]
    fn an_smplh_sequence_imports_with_its_tree_and_rate() {
        let a = parse(
            &sequence(156, 3),
            "sha256:x",
            SourceLicense::unaudited("fixture-internal"),
        )
        .unwrap();
        assert_eq!(a.model, BodyModel::SmplH);
        assert_eq!(a.record.channels.len(), 52);
        assert_eq!(a.record.channels[18].name.as_str(), "left_elbow");
        assert_eq!(a.record.channels[18].parent, Some(16));
        assert_eq!(a.record.channels[22].name.as_str(), "left_index1");
        assert_eq!(a.record.channels[22].parent, Some(20));
        assert_eq!(a.record.channels[23].parent, Some(22));
        assert_eq!(a.gender.as_deref(), Some("male"));
        assert_eq!(a.betas.len(), 16);
        let JointValue::Rotation { rotation, .. } = a.record.frames[0].values[18] else {
            panic!()
        };
        assert!((rotation.angle().value() - 0.5).abs() < 1e-12);
        assert!((a.record.frames[2].root.translation.x - 0.2).abs() < 1e-12);
        assert_eq!(a.record.frames[1].timestamp.nanos(), 8_333_333);
    }

    #[test]
    fn model_widths_select_the_tree() {
        assert_eq!(BodyModel::Smpl.joints().len(), 24);
        assert_eq!(BodyModel::SmplH.joints().len(), 52);
        assert_eq!(BodyModel::SmplX.joints().len(), 55);
        for model in [BodyModel::Smpl, BodyModel::SmplH, BodyModel::SmplX] {
            for (i, (_, p)) in model.joints().iter().enumerate() {
                assert!(p.is_none_or(|p| p < i), "{model:?} joint {i}");
            }
        }
    }

    #[test]
    fn an_unknown_width_or_missing_field_is_refused() {
        assert!(matches!(
            parse(&sequence(60, 2), "x", SourceLicense::unaudited("f")),
            Err(AmassError::Field(_))
        ));
        let z = zip_stored(&[("poses.npy", npy_f8(&[1, 72], &[0.0; 72]))]);
        assert!(matches!(
            parse(&z, "x", SourceLicense::unaudited("f")),
            Err(AmassError::Field(_))
        ));
    }
}

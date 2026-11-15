//! The retarget-map file format: a TOML description of a
//! [`RetargetConfig`], with human channels and robot links named rather
//! than indexed, so one file serves every record of a skeleton.
//!
//! It is read with `robot-artifacts`' bounded TOML subset (no inline
//! tables), and every table rejects keys it does not know, so a misspelt
//! weight is an error rather than a silently ignored line.
//!
//! ```toml
//! schema = "retarget-map/1"
//! scale = 0.8                       # human to robot lengths
//! root_offset = [0.0, 0.0, 0.0]     # rotation vector, radians
//! contact_confidence = 0.5
//!
//! [[link]]
//! human = "RightHand"
//! robot = "right_rubber_hand"
//! position_weight = 1.0
//! relative_human = "RightShoulder"  # omit both for an absolute target
//! relative_robot = "right_shoulder_pitch_link"
//! ratio = "auto"                    # or a number
//! orientation_weight = 0.1          # optional
//! orientation_offset = [0.0, 0.0, 0.0]
//!
//! [[foot]]
//! site = 0
//! human = "LeftFoot"
//! robot = "left_ankle_roll_link"
//! weight = 100.0
//! ground_height = 0.0
//!
//! [solver]      # all optional
//! max_iterations = 50
//! tolerance = 1e-4
//! deadline_ms = 20
//!
//! [base]        # present: the base is free at this cost
//! translation_weight = 0.01
//! rotation_weight = 0.1
//!
//! [posture]     # a prior toward the zero posture
//! weight = 0.001
//!
//! [collision]
//! margin = 0.01
//! weight = 1000.0
//!
//! [smoothing]
//! lambda = 5.0
//! refine_weight = 0.01
//! ```
//!
//! `ratio = "auto"` is [`chain_ratio`] between the two named pairs, with
//! the robot at its zero posture clamped into its limits.

use core::fmt;
use robot_artifacts::toml::{self, Table, TomlError, Value};
use robot_model::profile::RobotProfile;
use robot_types::frames::{Quat, Vec3};
use robot_types::ids::ContactSiteId;
use robot_types::motion::{HumanSkeleton, MotionRecord};

use crate::collision::SelfCollision;
use crate::ik::{BasePrior, IkConfig};
use crate::retarget::{
    chain_ratio, Anchor, FootMapping, LinkMapping, OrientationMapping, RetargetConfig, RetargetMap,
    Smoothing,
};

/// The schema string this module reads.
pub const SCHEMA: &str = "retarget-map/1";

/// Why a retarget map could not be read.
#[derive(Clone, PartialEq, Debug)]
pub enum ConfigError {
    /// The TOML itself was rejected.
    Toml(TomlError),
    /// The `schema` key was missing or another version.
    Schema(String),
    /// A table carried keys this format does not define.
    UnknownKeys {
        /// The table.
        table: String,
        /// The keys.
        keys: Vec<String>,
    },
    /// A named human channel does not exist in the record.
    UnknownChannel(String),
    /// A named robot link does not exist in the profile.
    UnknownLink(String),
    /// A value had the wrong type or range.
    Invalid(String),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Toml(e) => write!(f, "{e}"),
            Self::Schema(s) => write!(f, "expected schema {SCHEMA:?}, found {s:?}"),
            Self::UnknownKeys { table, keys } => {
                write!(f, "unknown keys in {table}: {}", keys.join(", "))
            }
            Self::UnknownChannel(n) => write!(f, "no human channel named {n:?}"),
            Self::UnknownLink(n) => write!(f, "no robot link named {n:?}"),
            Self::Invalid(what) => write!(f, "invalid value: {what}"),
        }
    }
}

impl core::error::Error for ConfigError {}

impl From<TomlError> for ConfigError {
    fn from(e: TomlError) -> Self {
        Self::Toml(e)
    }
}

fn check_keys(table: &Table, name: &str, known: &[&str]) -> Result<(), ConfigError> {
    let keys = table.unknown_keys(known);
    if keys.is_empty() {
        Ok(())
    } else {
        Err(ConfigError::UnknownKeys {
            table: name.to_owned(),
            keys,
        })
    }
}

fn float_or(table: &Table, key: &str, default: f64) -> Result<f64, ConfigError> {
    match table.get(key) {
        None => Ok(default),
        Some(v) => v
            .as_float()
            .ok_or_else(|| ConfigError::Invalid(format!("{key} must be a number"))),
    }
}

fn vec3(table: &Table, key: &str) -> Result<Vec3, ConfigError> {
    match table.get(key) {
        None => Ok(Vec3::ZERO),
        Some(v) => match v.as_float_array().as_deref() {
            Some([x, y, z]) => Ok(Vec3::new(*x, *y, *z)),
            _ => Err(ConfigError::Invalid(format!("{key} must be three numbers"))),
        },
    }
}

fn tables<'a>(root: &'a Table, key: &str) -> Result<Vec<&'a Table>, ConfigError> {
    match root.get(key) {
        None => Ok(Vec::new()),
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| match v {
                Value::Table(t) => Ok(t),
                _ => Err(ConfigError::Invalid(format!(
                    "[[{key}]] entries must be tables"
                ))),
            })
            .collect(),
        Some(_) => Err(ConfigError::Invalid(format!(
            "{key} must be an array of tables"
        ))),
    }
}

/// Parses `text` against the record's channel names and the profile's
/// link names.
pub fn parse(
    text: &str,
    record: &MotionRecord<HumanSkeleton>,
    profile: &RobotProfile,
) -> Result<RetargetConfig, ConfigError> {
    let root = toml::parse(text)?;
    check_keys(
        &root,
        "root",
        &[
            "schema",
            "scale",
            "root_offset",
            "contact_confidence",
            "link",
            "foot",
            "solver",
            "base",
            "posture",
            "collision",
            "smoothing",
        ],
    )?;
    let schema = root.require_str("schema")?;
    if schema != SCHEMA {
        return Err(ConfigError::Schema(schema.to_owned()));
    }
    let channel = |name: &str| {
        record
            .channel_index(name)
            .ok_or_else(|| ConfigError::UnknownChannel(name.to_owned()))
    };
    let link = |name: &str| {
        profile
            .links
            .iter()
            .position(|l| l.name.as_str() == name)
            .ok_or_else(|| ConfigError::UnknownLink(name.to_owned()))
    };
    let zero: Vec<f64> = profile
        .actuated_joints()
        .map(|j| {
            j.limits
                .position
                .map_or(0.0, |r| 0.0f64.clamp(r.lo.value(), r.hi.value()))
        })
        .collect();

    let mut links = Vec::new();
    for t in tables(&root, "link")? {
        check_keys(
            t,
            "[[link]]",
            &[
                "human",
                "robot",
                "position_weight",
                "relative_human",
                "relative_robot",
                "ratio",
                "orientation_weight",
                "orientation_offset",
            ],
        )?;
        let human = channel(t.require_str("human")?)?;
        let robot = link(t.require_str("robot")?)?;
        let anchor = match (t.get("relative_human"), t.get("relative_robot")) {
            (None, None) => Anchor::Absolute,
            (Some(h), Some(r)) => {
                let human_base = channel(
                    h.as_str()
                        .ok_or_else(|| ConfigError::Invalid("relative_human".into()))?,
                )?;
                let robot_base = link(
                    r.as_str()
                        .ok_or_else(|| ConfigError::Invalid("relative_robot".into()))?,
                )?;
                let ratio = match t.get("ratio") {
                    None => 1.0,
                    Some(v) if v.as_str() == Some("auto") => {
                        chain_ratio(record, human_base, human, profile, robot_base, robot, &zero)
                            .map_err(|e| ConfigError::Invalid(e.to_string()))?
                    }
                    Some(v) => v.as_float().ok_or_else(|| {
                        ConfigError::Invalid("ratio must be a number or \"auto\"".into())
                    })?,
                };
                Anchor::Relative {
                    human_base,
                    robot_base,
                    ratio,
                }
            }
            _ => {
                return Err(ConfigError::Invalid(
                    "relative_human and relative_robot go together".into(),
                ))
            }
        };
        let orientation = match t.get("orientation_weight") {
            None => None,
            Some(w) => Some(OrientationMapping {
                weight: w
                    .as_float()
                    .ok_or_else(|| ConfigError::Invalid("orientation_weight".into()))?,
                offset: Quat::from_rotation_vector(vec3(t, "orientation_offset")?),
            }),
        };
        links.push(LinkMapping {
            human,
            robot,
            anchor,
            position_weight: float_or(t, "position_weight", 1.0)?,
            orientation,
        });
    }

    let mut feet = Vec::new();
    for t in tables(&root, "foot")? {
        check_keys(
            t,
            "[[foot]]",
            &["site", "human", "robot", "weight", "ground_height"],
        )?;
        let site = u32::try_from(t.require_integer("site")?)
            .map_err(|_| ConfigError::Invalid("site must be a non-negative integer".into()))?;
        feet.push(FootMapping {
            site: ContactSiteId::new(site),
            human: channel(t.require_str("human")?)?,
            robot: link(t.require_str("robot")?)?,
            weight: float_or(t, "weight", 100.0)?,
            ground_height: float_or(t, "ground_height", 0.0)?,
        });
    }

    let mut ik = IkConfig::default();
    if let Some(Value::Table(t)) = root.get("solver") {
        check_keys(
            t,
            "[solver]",
            &[
                "max_iterations",
                "tolerance",
                "damping",
                "max_step",
                "deadline_ms",
                "method",
            ],
        )?;
        if let Some(v) = t.get("max_iterations") {
            let n = match v {
                Value::Integer(n) if *n > 0 => *n as usize,
                _ => {
                    return Err(ConfigError::Invalid(
                        "max_iterations must be a positive integer".into(),
                    ))
                }
            };
            ik.max_iterations = n;
        }
        ik.tolerance = float_or(t, "tolerance", ik.tolerance)?;
        ik.damping = float_or(t, "damping", ik.damping)?;
        ik.max_step = float_or(t, "max_step", ik.max_step)?;
        if t.get("deadline_ms").is_some() {
            let ms = float_or(t, "deadline_ms", 0.0)?;
            if !ms.is_finite() || ms < 0.0 {
                return Err(ConfigError::Invalid("deadline_ms".into()));
            }
            ik.deadline = Some(std::time::Duration::from_secs_f64(ms * 1e-3));
        }
        // The step solver, by name: the damped-least-squares baseline or the
        // differential-QP step of Section 8.3 (ADR 0018). Anything else is
        // refused, like every other key in this format.
        if let Some(v) = t.get("method") {
            ik.step_solver = match v {
                Value::String(s) if s == "lm" => crate::ik::StepSolver::DampedLeastSquares,
                Value::String(s) if s == "qp" => crate::ik::StepSolver::BoxQp,
                _ => {
                    return Err(ConfigError::Invalid(
                        "method must be \"lm\" or \"qp\"".into(),
                    ))
                }
            };
        }
    }
    let base_prior = match root.get("base") {
        Some(Value::Table(t)) => {
            check_keys(t, "[base]", &["translation_weight", "rotation_weight"])?;
            Some(BasePrior {
                translation_weight: float_or(t, "translation_weight", 1e-2)?,
                rotation_weight: float_or(t, "rotation_weight", 1e-1)?,
            })
        }
        _ => None,
    };
    let posture = match root.get("posture") {
        Some(Value::Table(t)) => {
            check_keys(t, "[posture]", &["weight"])?;
            Some((
                zero.clone(),
                vec![t.require_float("weight")?; profile.dof()],
            ))
        }
        _ => None,
    };
    let collision = match root.get("collision") {
        Some(Value::Table(t)) => {
            check_keys(t, "[collision]", &["margin", "weight"])?;
            Some(
                SelfCollision::from_profile(
                    profile,
                    &zero,
                    float_or(t, "margin", 0.01)?,
                    float_or(t, "weight", 1e3)?,
                )
                .map_err(|e| ConfigError::Invalid(e.to_string()))?,
            )
        }
        _ => None,
    };
    let smoothing = match root.get("smoothing") {
        Some(Value::Table(t)) => {
            check_keys(t, "[smoothing]", &["lambda", "refine_weight"])?;
            Some(Smoothing {
                lambda: t.require_float("lambda")?,
                refine_weight: float_or(t, "refine_weight", 1e-2)?,
            })
        }
        _ => None,
    };

    Ok(RetargetConfig {
        map: RetargetMap {
            scale: float_or(&root, "scale", 1.0)?,
            root_offset: Quat::from_rotation_vector(vec3(&root, "root_offset")?),
            links,
            feet,
        },
        ik,
        base_prior,
        posture,
        collision,
        smoothing,
        contact_confidence: float_or(&root, "contact_confidence", 0.5)?,
        initial: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retarget::retarget;
    use robot_model::fixtures::unitree_g1_29dof;
    use robot_types::frames::{FrameId, Transform};
    use robot_types::ids::Name;
    use robot_types::motion::{Channel, JointValue, MotionFrame, SourceLicense, Validity};
    use robot_types::time::{ClockId, Timestamp};

    const MAP: &str = r#"
schema = "retarget-map/1"
scale = 0.8

[[link]]
human = "RightHand"
robot = "right_rubber_hand"
relative_human = "RightShoulder"
relative_robot = "right_shoulder_pitch_link"
ratio = "auto"

[solver]
max_iterations = 40
tolerance = 1e-4
deadline_ms = 1000

[base]
translation_weight = 0.01
rotation_weight = 0.1

[posture]
weight = 0.001

[smoothing]
lambda = 2.0
"#;

    fn human() -> MotionRecord<HumanSkeleton> {
        let ch = |name: &str, parent, offset| Channel {
            name: Name::new(name).unwrap(),
            parent,
            offset,
        };
        let rot = JointValue::Rotation {
            rotation: Quat::IDENTITY,
            source_order: None,
        };
        let frames = (0..3)
            .map(|i| MotionFrame {
                timestamp: Timestamp::sampling(i * 20_000_000, ClockId::Source(0)),
                root: Transform::new(
                    FrameId::WORLD,
                    FrameId::WORLD,
                    Quat::IDENTITY,
                    Vec3::new(0.0, 0.0, 0.9),
                ),
                values: vec![rot.clone(), rot.clone(), rot.clone()],
                validity: vec![Validity::Valid; 3],
                contacts: Vec::new(),
            })
            .collect();
        MotionRecord::new(
            HumanSkeleton,
            "sha256:fixture",
            SourceLicense::unaudited("fixture-internal"),
            "s",
            "k",
            FrameId::WORLD,
            ClockId::Source(0),
            vec![
                ch("Hips", None, Vec3::ZERO),
                ch("RightShoulder", Some(0), Vec3::new(0.0, -0.2, 0.5)),
                ch("RightHand", Some(1), Vec3::new(0.0, 0.0, -0.6)),
            ],
            frames,
        )
        .unwrap()
    }

    #[test]
    fn a_map_file_resolves_names_and_runs() {
        let profile = unitree_g1_29dof().unwrap();
        let h = human();
        let config = parse(MAP, &h, &profile).unwrap();
        assert_eq!(config.map.links.len(), 1);
        let Anchor::Relative { ratio, .. } = config.map.links[0].anchor else {
            panic!()
        };
        assert!(ratio > 0.3 && ratio < 1.5, "{ratio}");
        assert_eq!(config.ik.max_iterations, 40);
        assert!(
            config.base_prior.is_some() && config.posture.is_some() && config.smoothing.is_some()
        );
        let out = retarget(&h, &profile, &config).unwrap();
        assert_eq!(out.reference.frame_count(), 3);
    }

    #[test]
    fn unknown_names_keys_and_schemas_are_errors() {
        let profile = unitree_g1_29dof().unwrap();
        let h = human();
        assert_eq!(
            parse(&MAP.replace("\"RightHand\"", "\"RHand\""), &h, &profile).unwrap_err(),
            ConfigError::UnknownChannel("RHand".into())
        );
        assert_eq!(
            parse(
                &MAP.replace("right_rubber_hand", "right_glove"),
                &h,
                &profile
            )
            .unwrap_err(),
            ConfigError::UnknownLink("right_glove".into())
        );
        assert!(matches!(
            parse(&MAP.replace("tolerance", "tolerence"), &h, &profile).unwrap_err(),
            ConfigError::UnknownKeys { .. }
        ));
        assert!(matches!(
            parse(
                &MAP.replace("retarget-map/1", "retarget-map/2"),
                &h,
                &profile
            )
            .unwrap_err(),
            ConfigError::Schema(_)
        ));
    }
}

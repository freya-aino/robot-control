//! The retargeting pipeline: a human motion record to a robot reference,
//! per SPECS.md Section 8.2 item 7 ("Retarget to target geometry with
//! contact, joint-limit and self-collision constraints") and Section 8.3.
//!
//! For every frame the [`RetargetMap`] turns the human's global joint poses
//! into IK tasks:
//!
//! - an **absolute** mapping targets a robot link at the human joint's
//!   position scaled by `scale` (for the feet and the root, whose world
//!   placement matters);
//! - a **relative** mapping targets a robot link's displacement from
//!   another robot link at the human's displacement between the matching
//!   joints, multiplied by the ratio of robot to human chain length. This
//!   is how "task intent, such as punch direction and target-relative
//!   reach" is preserved (Section 8.3): a jab reaching 90 % of the human's
//!   arm reaches 90 % of the robot's, in the same direction, instead of
//!   matching the human's absolute hand position;
//! - an optional **orientation** for either kind, as the human joint's
//!   world orientation followed by a fixed offset.
//!
//! **Planted feet.** For each [`FootMapping`], while the human record
//! labels its site in contact with at least `contact_confidence`, the robot
//! foot is held at the world point where that contact began (the mapped
//! human foot projected onto the ground height). The hold is a heavily
//! weighted position task, so slip is minimized rather than forbidden, and
//! the per-frame report says how far the solution is from the anchor.
//!
//! **Base.** The robot base starts each frame at the human root pose
//! (translation scaled, orientation followed by `root_offset`) and, when
//! `base_prior` is set, the solver may move it at that prior's cost. That
//! is what lets planted feet and a differently proportioned pelvis height
//! agree.
//!
//! **Temporal refinement.** With [`Smoothing`] set, joint and base
//! trajectories are smoothed (see [`crate::temporal`]) and every frame is
//! then solved again from its smoothed value, with a posture prior pulling
//! toward it: smoothing removes jitter, re-projection restores joint
//! limits, planted feet and collision clearance.
//!
//! Nothing here claims the reference is dynamically feasible: velocity
//! excess against each joint's declared limit is reported, and torque and
//! tracking feasibility are `motion-validate`'s job.

use core::fmt;
use motion_process::kinematics::frame_poses;
use robot_model::joint::JointKind;
use robot_model::kinematics::{forward_kinematics, KinematicsError};
use robot_model::pose::RigidPose;
use robot_model::profile::RobotProfile;
use robot_types::frames::{FrameId, Quat, Transform, Vec3};
use robot_types::ids::ContactSiteId;
use robot_types::motion::{
    Channel, HumanSkeleton, JointValue, MotionError, MotionFrame, MotionRecord, ProvenanceStep,
    RobotReference, Validity,
};

use crate::collision::SelfCollision;
use crate::ik::{solve_tasks, Base, BasePrior, IkConfig, IkError, Task, Termination};
use crate::temporal::{smooth, smooth_rotations, SmoothError};

/// Software revision recorded in provenance.
pub const REVISION: &str = concat!("motion-retarget/", env!("CARGO_PKG_VERSION"));

/// How a mapped link's position target is formed.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Anchor {
    /// Robot link at `scale` times the human joint's world position.
    Absolute,
    /// Robot link displaced from `robot_base` by `ratio` times the human
    /// joint's displacement from `human_base`.
    Relative {
        /// Human channel the displacement is measured from.
        human_base: usize,
        /// Index into `profile.links` the robot displacement is measured
        /// from.
        robot_base: usize,
        /// Robot chain length over human chain length.
        ratio: f64,
    },
}

/// Orientation target of a mapped link.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct OrientationMapping {
    /// Task weight.
    pub weight: f64,
    /// Robot link orientation = human joint world orientation * `offset`.
    pub offset: Quat,
}

/// One human joint driving one robot link.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct LinkMapping {
    /// Human channel index.
    pub human: usize,
    /// Index into `profile.links`.
    pub robot: usize,
    /// Position target construction.
    pub anchor: Anchor,
    /// Position task weight (zero for orientation only).
    pub position_weight: f64,
    /// Orientation task, if any.
    pub orientation: Option<OrientationMapping>,
}

/// A foot planted while the human record labels its site in contact.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct FootMapping {
    /// Contact site in the human record's labels.
    pub site: ContactSiteId,
    /// Human channel of the foot, for the anchor position.
    pub human: usize,
    /// Index into `profile.links` of the robot foot.
    pub robot: usize,
    /// Weight of the hold.
    pub weight: f64,
    /// World height of the robot foot link's origin when standing.
    pub ground_height: f64,
}

/// The human-to-robot correspondence.
#[derive(Clone, PartialEq, Debug)]
pub struct RetargetMap {
    /// Human-to-robot length scale for absolute targets and the root.
    pub scale: f64,
    /// Robot base orientation = human root orientation * `root_offset`.
    pub root_offset: Quat,
    /// Mapped links.
    pub links: Vec<LinkMapping>,
    /// Planted feet.
    pub feet: Vec<FootMapping>,
}

/// Temporal refinement settings.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Smoothing {
    /// Smoothing strength (see [`crate::temporal`]).
    pub lambda: f64,
    /// Weight of the posture prior toward the smoothed joints during
    /// re-projection.
    pub refine_weight: f64,
}

/// Pipeline configuration.
#[derive(Clone, PartialEq, Debug)]
pub struct RetargetConfig {
    /// The correspondence.
    pub map: RetargetMap,
    /// Per-frame solver settings.
    pub ik: IkConfig,
    /// `Some` frees the base, at this cost.
    pub base_prior: Option<BasePrior>,
    /// A posture prior (`nominal`, `weights`), in actuated-joint order.
    pub posture: Option<(Vec<f64>, Vec<f64>)>,
    /// Self-collision penalty.
    pub collision: Option<SelfCollision>,
    /// Temporal refinement.
    pub smoothing: Option<Smoothing>,
    /// Minimum contact-label confidence for a foot to be planted.
    pub contact_confidence: f64,
    /// Joint values the first frame starts from (`None`: zeros clamped to
    /// the limits).
    pub initial: Option<Vec<f64>>,
}

/// What happened at one frame.
#[derive(Clone, PartialEq, Debug)]
pub struct FrameReport {
    /// Why the (last) solve stopped.
    pub termination: Termination,
    /// Weighted residual norm.
    pub final_error: f64,
    /// Worst unweighted position error over the mapped links, metres.
    pub max_position_error: f64,
    /// Feet planted at this frame, with their distance from the anchor.
    pub planted: Vec<(ContactSiteId, f64)>,
    /// Largest self-collision penetration of the margin, metres.
    pub max_penetration: f64,
    /// Mapped links skipped because a human joint was missing.
    pub skipped_mappings: usize,
}

/// One joint exceeding its velocity limit between two frames.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct VelocityExcess {
    /// Frame index (the later of the pair).
    pub frame: usize,
    /// Actuated-joint index.
    pub joint: usize,
    /// Speed found, rad/s or m/s.
    pub speed: f64,
    /// The declared limit.
    pub limit: f64,
}

/// The pipeline's result.
#[derive(Clone, PartialEq, Debug)]
pub struct RetargetOutput {
    /// The robot reference: one scalar channel per actuated joint, the
    /// base pose as the root.
    pub reference: MotionRecord<RobotReference>,
    /// Per-frame reports.
    pub frames: Vec<FrameReport>,
    /// Joint velocity excess over the declared limits.
    pub velocity_excess: Vec<VelocityExcess>,
}

impl RetargetOutput {
    /// Frames whose solve hit its deadline or iteration cap, which the
    /// SPECS.md Section 8.3 timeout response must not pass silently.
    #[must_use]
    pub fn unfinished_frames(&self) -> Vec<usize> {
        self.frames
            .iter()
            .enumerate()
            .filter(|(_, f)| {
                matches!(
                    f.termination,
                    Termination::Deadline | Termination::MaxIterations
                )
            })
            .map(|(i, _)| i)
            .collect()
    }
}

/// Why retargeting failed.
#[derive(Clone, PartialEq, Debug)]
pub enum RetargetError {
    /// A mapping named a channel or link that does not exist.
    InvalidMapping(&'static str),
    /// A parameter was out of range.
    InvalidParameter(&'static str),
    /// The solver refused a frame.
    Ik {
        /// Frame index.
        frame: usize,
        /// The error.
        error: IkError,
    },
    /// Smoothing failed.
    Smooth(SmoothError),
    /// The output record failed validation.
    Motion(MotionError),
}

impl fmt::Display for RetargetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMapping(what) => write!(f, "invalid mapping: {what}"),
            Self::InvalidParameter(what) => write!(f, "invalid parameter: {what}"),
            Self::Ik { frame, error } => write!(f, "frame {frame}: {error}"),
            Self::Smooth(e) => write!(f, "{e}"),
            Self::Motion(e) => write!(f, "{e}"),
        }
    }
}

impl core::error::Error for RetargetError {}

/// The ratio of the robot's `robot_base`-to-`robot_tip` distance at
/// `robot_reference` to the human's `human_base`-to-`human_tip` distance in
/// the rest pose (every local rotation the identity).
pub fn chain_ratio(
    record: &MotionRecord<HumanSkeleton>,
    human_base: usize,
    human_tip: usize,
    profile: &RobotProfile,
    robot_base: usize,
    robot_tip: usize,
    robot_reference: &[f64],
) -> Result<f64, KinematicsError> {
    let rest = rest_positions(record);
    let human = (rest[human_tip] - rest[human_base]).norm();
    let poses = forward_kinematics(profile, robot_reference)?;
    let robot = (poses
        .by_index(robot_tip)
        .map(|p| p.translation)
        .unwrap_or(Vec3::ZERO)
        - poses
            .by_index(robot_base)
            .map(|p| p.translation)
            .unwrap_or(Vec3::ZERO))
    .norm();
    Ok(if human > 0.0 { robot / human } else { 1.0 })
}

fn rest_positions(record: &MotionRecord<HumanSkeleton>) -> Vec<Vec3> {
    let mut out: Vec<Vec3> = Vec::with_capacity(record.channels.len());
    for c in &record.channels {
        let parent = c.parent.map_or(Vec3::ZERO, |p| out[p]);
        out.push(parent + c.offset);
    }
    out
}

fn validate(
    record: &MotionRecord<HumanSkeleton>,
    profile: &RobotProfile,
    config: &RetargetConfig,
) -> Result<(), RetargetError> {
    let (channels, links) = (record.channels.len(), profile.links.len());
    let map = &config.map;
    if !map.scale.is_finite() || map.scale <= 0.0 {
        return Err(RetargetError::InvalidParameter("scale"));
    }
    if !(0.0..=1.0).contains(&config.contact_confidence) {
        return Err(RetargetError::InvalidParameter("contact_confidence"));
    }
    for m in &map.links {
        if m.human >= channels || m.robot >= links {
            return Err(RetargetError::InvalidMapping(
                "link mapping index out of range",
            ));
        }
        if let Anchor::Relative {
            human_base,
            robot_base,
            ratio,
        } = m.anchor
        {
            if human_base >= channels || robot_base >= links {
                return Err(RetargetError::InvalidMapping("relative base out of range"));
            }
            if !ratio.is_finite() || ratio <= 0.0 {
                return Err(RetargetError::InvalidParameter("ratio"));
            }
        }
    }
    for f in &map.feet {
        if f.human >= channels || f.robot >= links {
            return Err(RetargetError::InvalidMapping(
                "foot mapping index out of range",
            ));
        }
    }
    if let Some(s) = config.smoothing {
        if !s.refine_weight.is_finite() || s.refine_weight < 0.0 {
            return Err(RetargetError::InvalidParameter("refine_weight"));
        }
    }
    if let Some(init) = &config.initial {
        if init.len() != profile.dof() {
            return Err(RetargetError::InvalidParameter("initial width"));
        }
    }
    Ok(())
}

struct FrameTasks {
    tasks: Vec<Task>,
    base: RigidPose,
    planted: Vec<(ContactSiteId, usize, Vec3)>,
    skipped: usize,
}

fn frame_tasks(
    record: &MotionRecord<HumanSkeleton>,
    f: usize,
    config: &RetargetConfig,
    anchors: &mut [Option<Vec3>],
) -> FrameTasks {
    let map = &config.map;
    let frame = &record.frames[f];
    let poses = frame_poses(record, f);
    let base = RigidPose {
        translation: frame.root.translation * map.scale,
        rotation: (frame.root.rotation * map.root_offset).normalized(),
    };
    let mut tasks = Vec::new();
    let mut skipped = 0;
    for m in &map.links {
        let Some((p, q)) = poses[m.human] else {
            skipped += 1;
            continue;
        };
        if m.position_weight > 0.0 {
            match m.anchor {
                Anchor::Absolute => tasks.push(Task::Position {
                    link: m.robot,
                    target: p * map.scale,
                    weight: m.position_weight,
                }),
                Anchor::Relative {
                    human_base,
                    robot_base,
                    ratio,
                } => match poses[human_base] {
                    Some((pb, _)) => tasks.push(Task::RelativePosition {
                        link: m.robot,
                        base: robot_base,
                        target: (p - pb) * ratio,
                        weight: m.position_weight,
                    }),
                    None => skipped += 1,
                },
            }
        }
        if let Some(o) = m.orientation {
            tasks.push(Task::Orientation {
                link: m.robot,
                target: (q * o.offset).normalized(),
                weight: o.weight,
            });
        }
    }
    let mut planted = Vec::new();
    for (k, foot) in map.feet.iter().enumerate() {
        let in_contact = frame.contacts.iter().any(|c| {
            c.site == foot.site && c.in_contact && c.confidence >= config.contact_confidence
        });
        if !in_contact {
            anchors[k] = None;
            continue;
        }
        let anchor = match anchors[k] {
            Some(a) => a,
            None => {
                let Some((p, _)) = poses[foot.human] else {
                    continue;
                };
                let p = p * map.scale;
                let a = Vec3::new(p.x, p.y, foot.ground_height);
                anchors[k] = Some(a);
                a
            }
        };
        tasks.push(Task::Position {
            link: foot.robot,
            target: anchor,
            weight: foot.weight,
        });
        planted.push((foot.site, foot.robot, anchor));
    }
    if let Some((nominal, weights)) = &config.posture {
        tasks.push(Task::Posture {
            nominal: nominal.clone(),
            weights: weights.clone(),
        });
    }
    FrameTasks {
        tasks,
        base,
        planted,
        skipped,
    }
}

/// Retargets `record` onto `profile`.
pub fn retarget(
    record: &MotionRecord<HumanSkeleton>,
    profile: &RobotProfile,
    config: &RetargetConfig,
) -> Result<RetargetOutput, RetargetError> {
    validate(record, profile, config)?;
    let limits: Vec<Option<(f64, f64)>> = profile
        .actuated_joints()
        .map(|j| j.limits.position.map(|r| (r.lo.value(), r.hi.value())))
        .collect();
    let mut q: Vec<f64> = config.initial.clone().unwrap_or_else(|| {
        limits
            .iter()
            .map(|l| l.map_or(0.0, |(lo, hi)| 0.0f64.clamp(lo, hi)))
            .collect()
    });
    let n = record.frames.len();
    let mut anchors = vec![None; config.map.feet.len()];
    let mut all_tasks = Vec::with_capacity(n);
    let mut joints = Vec::with_capacity(n);
    let mut bases = Vec::with_capacity(n);
    let mut solutions = Vec::with_capacity(n);
    for f in 0..n {
        let ft = frame_tasks(record, f, config, &mut anchors);
        let base = Base {
            pose: ft.base,
            free: config.base_prior,
        };
        let s = solve_tasks(
            profile,
            &q,
            base,
            &ft.tasks,
            config.collision.as_ref(),
            &config.ik,
        )
        .map_err(|error| RetargetError::Ik { frame: f, error })?;
        q.clone_from(&s.joint_values);
        joints.push(s.joint_values.clone());
        bases.push(s.base);
        solutions.push(s);
        all_tasks.push(ft);
    }

    if let Some(sm) = config.smoothing {
        let smoothed = smooth(&joints, sm.lambda).map_err(RetargetError::Smooth)?;
        let translations: Vec<Vec<f64>> = bases
            .iter()
            .map(|b| b.translation.to_xyz().to_vec())
            .collect();
        let translations = smooth(&translations, sm.lambda).map_err(RetargetError::Smooth)?;
        let rotations: Vec<Quat> = bases.iter().map(|b| b.rotation).collect();
        let rotations = smooth_rotations(&rotations, sm.lambda).map_err(RetargetError::Smooth)?;
        let weights = vec![sm.refine_weight; profile.dof()];
        for f in 0..n {
            let mut tasks = all_tasks[f].tasks.clone();
            tasks.push(Task::Posture {
                nominal: smoothed[f].clone(),
                weights: weights.clone(),
            });
            let base = Base {
                pose: RigidPose {
                    translation: Vec3::new(
                        translations[f][0],
                        translations[f][1],
                        translations[f][2],
                    ),
                    rotation: rotations[f],
                },
                free: config.base_prior,
            };
            let s = solve_tasks(
                profile,
                &smoothed[f],
                base,
                &tasks,
                config.collision.as_ref(),
                &config.ik,
            )
            .map_err(|error| RetargetError::Ik { frame: f, error })?;
            joints[f].clone_from(&s.joint_values);
            bases[f] = s.base;
            solutions[f] = s;
        }
    }

    let mut frames = Vec::with_capacity(n);
    for f in 0..n {
        let s = &solutions[f];
        let poses = forward_kinematics(profile, &joints[f]).map_err(|e| RetargetError::Ik {
            frame: f,
            error: IkError::Kinematics(e),
        })?;
        let world = |link: usize| {
            bases[f]
                .compose(&poses.by_index(link).expect("validated"))
                .translation
        };
        let max_position_error = all_tasks[f]
            .tasks
            .iter()
            .zip(&s.task_errors)
            .filter(|(t, _)| matches!(t, Task::Position { .. } | Task::RelativePosition { .. }))
            .map(|(_, e)| *e)
            .fold(0.0, f64::max);
        frames.push(FrameReport {
            termination: s.termination,
            final_error: s.final_error,
            max_position_error,
            planted: all_tasks[f]
                .planted
                .iter()
                .map(|(site, link, anchor)| (*site, (world(*link) - *anchor).norm()))
                .collect(),
            max_penetration: s.max_penetration,
            skipped_mappings: all_tasks[f].skipped,
        });
    }

    let mut velocity_excess = Vec::new();
    let joint_specs: Vec<_> = profile.actuated_joints().collect();
    for f in 1..n {
        let dt = (record.frames[f].timestamp.nanos() - record.frames[f - 1].timestamp.nanos())
            as f64
            * 1e-9;
        if dt <= 0.0 {
            continue;
        }
        for (j, spec) in joint_specs.iter().enumerate() {
            let speed = (joints[f][j] - joints[f - 1][j]).abs() / dt;
            let limit = spec.limits.velocity.value();
            if limit > 0.0 && speed > limit {
                velocity_excess.push(VelocityExcess {
                    frame: f,
                    joint: j,
                    speed,
                    limit,
                });
            }
        }
    }

    let channels: Vec<Channel> = joint_specs
        .iter()
        .map(|j| Channel {
            name: j.name.clone(),
            parent: None,
            offset: Vec3::ZERO,
        })
        .collect();
    let out_frames: Vec<MotionFrame> = (0..n)
        .map(|f| MotionFrame {
            timestamp: record.frames[f].timestamp,
            root: Transform::new(
                FrameId::WORLD,
                record.reference_frame,
                bases[f].rotation,
                bases[f].translation,
            ),
            values: joint_specs
                .iter()
                .zip(&joints[f])
                .map(|(spec, v)| JointValue::Scalar {
                    value: *v,
                    unit: if spec.kind == JointKind::Prismatic {
                        "m"
                    } else {
                        "rad"
                    },
                })
                .collect(),
            validity: vec![Validity::Valid; joint_specs.len()],
            contacts: record.frames[f].contacts.clone(),
        })
        .collect();
    let mut reference = MotionRecord::new(
        RobotReference,
        record.source_hash.clone(),
        record.source_license.clone(),
        record.session_id.clone(),
        format!("{}@{}", profile.name, profile.profile_version),
        record.reference_frame,
        record.clock,
        channels,
        out_frames,
    )
    .map_err(RetargetError::Motion)?;
    reference.provenance = record.provenance.clone();
    reference.subject_id = record.subject_id.clone();
    reference.task_labels = record.task_labels.clone();
    reference.subject_height = record.subject_height;
    let unfinished = frames
        .iter()
        .filter(|f| {
            matches!(
                f.termination,
                Termination::Deadline | Termination::MaxIterations
            )
        })
        .count();
    reference.record_step(
        ProvenanceStep::new("retarget", REVISION)
            .with_parameter("profile", profile.name.to_string())
            .with_parameter("scale", config.map.scale.to_string())
            .with_parameter("mappings", config.map.links.len().to_string())
            .with_parameter("planted_feet", config.map.feet.len().to_string())
            .with_parameter(
                "smoothing_lambda",
                config
                    .smoothing
                    .map_or("none".to_owned(), |s| s.lambda.to_string()),
            )
            .with_parameter("unfinished_frames", unfinished.to_string())
            .with_parameter("velocity_excess", velocity_excess.len().to_string()),
    );
    Ok(RetargetOutput {
        reference,
        frames,
        velocity_excess,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_model::fixtures::unitree_g1_29dof;
    use robot_types::ids::Name;
    use robot_types::motion::{ContactLabel, SourceLicense};
    use robot_types::time::{ClockId, Timestamp};
    use robot_types::units::Radians;

    fn link(profile: &RobotProfile, name: &str) -> usize {
        profile
            .links
            .iter()
            .position(|l| l.name.as_str() == name)
            .unwrap_or_else(|| panic!("no link {name}"))
    }

    /// A human with hips, a right shoulder, elbow and hand (arm 0.6 m,
    /// held forward), and two feet, punching: the hand extends from 0.3 m
    /// to 0.6 m in front of the shoulder over the record. The feet are
    /// labelled in contact throughout.
    fn punching_human(frames: usize) -> MotionRecord<HumanSkeleton> {
        let ch = |name: &str, parent: Option<usize>, offset: Vec3| Channel {
            name: Name::new(name).unwrap(),
            parent,
            offset,
        };
        let channels = vec![
            ch("Hips", None, Vec3::ZERO),
            ch("RightShoulder", Some(0), Vec3::new(0.0, -0.2, 0.5)),
            ch("RightHand", Some(1), Vec3::new(0.6, 0.0, 0.0)),
            ch("LeftFoot", Some(0), Vec3::new(0.0, 0.1, -0.9)),
            ch("RightFoot", Some(0), Vec3::new(0.0, -0.1, -0.9)),
        ];
        let rot = |q: Quat| JointValue::Rotation {
            rotation: q,
            source_order: None,
        };
        let frames = (0..frames)
            .map(|i| {
                let s = i as f64 / (frames - 1).max(1) as f64;
                // Hand 0.3 + 0.3 s ahead along +x, via a marker channel.
                let hand = Vec3::new(0.3 + 0.3 * s, -0.2, 1.4);
                MotionFrame {
                    timestamp: Timestamp::sampling(i as i64 * 20_000_000, ClockId::Source(0)),
                    root: Transform::new(
                        FrameId::WORLD,
                        FrameId::WORLD,
                        Quat::IDENTITY,
                        Vec3::new(0.0, 0.0, 0.9),
                    ),
                    values: vec![
                        rot(Quat::IDENTITY),
                        rot(Quat::IDENTITY),
                        JointValue::Position { position: hand },
                        rot(Quat::IDENTITY),
                        rot(Quat::IDENTITY),
                    ],
                    validity: vec![Validity::Valid; 5],
                    contacts: vec![
                        ContactLabel {
                            site: ContactSiteId::new(0),
                            in_contact: true,
                            confidence: 0.95,
                        },
                        ContactLabel {
                            site: ContactSiteId::new(1),
                            in_contact: true,
                            confidence: 0.95,
                        },
                    ],
                }
            })
            .collect();
        MotionRecord::new(
            HumanSkeleton,
            "sha256:fixture",
            SourceLicense::unaudited("fixture-internal"),
            "session",
            "skeleton",
            FrameId::WORLD,
            ClockId::Source(0),
            channels,
            frames,
        )
        .unwrap()
    }

    fn g1_config(
        profile: &RobotProfile,
        human: &MotionRecord<HumanSkeleton>,
        smoothing: Option<Smoothing>,
    ) -> RetargetConfig {
        let zero = vec![0.0; profile.dof()];
        let shoulder = link(profile, "right_shoulder_pitch_link");
        let hand = link(profile, "right_rubber_hand");
        let ratio = chain_ratio(human, 1, 2, profile, shoulder, hand, &zero).unwrap();
        let ankle = link(profile, "left_ankle_roll_link");
        RetargetConfig {
            map: RetargetMap {
                scale: 0.8,
                root_offset: Quat::IDENTITY,
                links: vec![LinkMapping {
                    human: 2,
                    robot: hand,
                    anchor: Anchor::Relative {
                        human_base: 1,
                        robot_base: shoulder,
                        ratio,
                    },
                    position_weight: 1.0,
                    orientation: None,
                }],
                feet: vec![
                    FootMapping {
                        site: ContactSiteId::new(0),
                        human: 3,
                        robot: ankle,
                        weight: 100.0,
                        ground_height: 0.0,
                    },
                    FootMapping {
                        site: ContactSiteId::new(1),
                        human: 4,
                        robot: link(profile, "right_ankle_roll_link"),
                        weight: 100.0,
                        ground_height: 0.0,
                    },
                ],
            },
            ik: IkConfig {
                max_iterations: 60,
                tolerance: 1e-4,
                max_step: 0.3,
                ..IkConfig::default()
            },
            base_prior: Some(BasePrior {
                translation_weight: 1e-2,
                rotation_weight: 1e-1,
            }),
            posture: Some((zero.clone(), vec![1e-3; profile.dof()])),
            collision: None,
            smoothing,
            contact_confidence: 0.5,
            initial: None,
        }
    }

    #[test]
    fn a_punch_keeps_its_direction_and_relative_reach_on_the_g1() {
        let profile = unitree_g1_29dof().unwrap();
        let human = punching_human(6);
        let config = g1_config(&profile, &human, None);
        let out = retarget(&human, &profile, &config).unwrap();
        assert_eq!(out.reference.frame_count(), 6);
        out.reference.validate().unwrap();
        let shoulder = link(&profile, "right_shoulder_pitch_link");
        let hand = link(&profile, "right_rubber_hand");
        let Anchor::Relative { ratio, .. } = config.map.links[0].anchor else {
            unreachable!()
        };
        for (f, frame) in out.reference.frames.iter().enumerate() {
            let q: Vec<f64> = frame
                .values
                .iter()
                .map(|v| match v {
                    JointValue::Scalar { value, .. } => *value,
                    _ => unreachable!(),
                })
                .collect();
            let poses = forward_kinematics(&profile, &q).unwrap();
            let base = RigidPose {
                translation: frame.root.translation,
                rotation: frame.root.rotation,
            };
            let d = base.compose(&poses.by_index(hand).unwrap()).translation
                - base.compose(&poses.by_index(shoulder).unwrap()).translation;
            let s = f as f64 / 5.0;
            let human_d = Vec3::new(0.3 + 0.3 * s, 0.0, 0.0);
            let want = human_d * ratio;
            assert!((d - want).norm() < 0.02, "frame {f}: {d:?} vs {want:?}");
            // Joint limits hold in the output.
            for (v, j) in q.iter().zip(profile.actuated_joints()) {
                if let Some(r) = j.limits.position {
                    assert!(r.contains(Radians(*v)));
                }
            }
        }
        for f in &out.frames {
            for (_, slip) in &f.planted {
                assert!(*slip < 0.02, "{f:?}");
            }
        }
        assert_eq!(
            out.reference.provenance.last().unwrap().operation,
            "retarget"
        );
    }

    #[test]
    fn smoothing_then_reprojection_keeps_the_task_and_limits() {
        let profile = unitree_g1_29dof().unwrap();
        let human = punching_human(8);
        let config = g1_config(
            &profile,
            &human,
            Some(Smoothing {
                lambda: 5.0,
                refine_weight: 1e-2,
            }),
        );
        let out = retarget(&human, &profile, &config).unwrap();
        for f in &out.frames {
            assert!(f.max_position_error < 0.03, "{f:?}");
        }
        assert!(out.unfinished_frames().len() <= out.frames.len());
    }

    #[test]
    fn a_too_fast_reference_reports_velocity_excess() {
        let profile = unitree_g1_29dof().unwrap();
        let mut human = punching_human(4);
        // Squeeze the punch into 3 ms.
        for (i, f) in human.frames.iter_mut().enumerate() {
            f.timestamp = Timestamp::sampling(i as i64 * 1_000_000, ClockId::Source(0));
        }
        let config = g1_config(&profile, &human, None);
        let out = retarget(&human, &profile, &config).unwrap();
        assert!(!out.velocity_excess.is_empty());
        assert!(out.velocity_excess.iter().all(|v| v.speed > v.limit));
    }

    #[test]
    fn the_g1_collision_model_is_built_and_respected() {
        let profile = unitree_g1_29dof().unwrap();
        let zero = vec![0.0; profile.dof()];
        let model = SelfCollision::from_profile(&profile, &zero, 0.01, 1e3).unwrap();
        assert!(!model.capsules.is_empty());
        assert!(!model.pairs.is_empty());
        let human = punching_human(3);
        let mut config = g1_config(&profile, &human, None);
        config.collision = Some(model);
        let out = retarget(&human, &profile, &config).unwrap();
        for f in &out.frames {
            assert!(f.max_penetration < 0.01, "{f:?}");
        }
    }

    #[test]
    fn a_bad_mapping_is_refused_before_any_solve() {
        let profile = unitree_g1_29dof().unwrap();
        let human = punching_human(2);
        let mut config = g1_config(&profile, &human, None);
        config.map.links[0].human = 99;
        assert!(matches!(
            retarget(&human, &profile, &config),
            Err(RetargetError::InvalidMapping(_))
        ));
        let mut config = g1_config(&profile, &human, None);
        config.map.scale = 0.0;
        assert_eq!(
            retarget(&human, &profile, &config),
            Err(RetargetError::InvalidParameter("scale"))
        );
    }
}

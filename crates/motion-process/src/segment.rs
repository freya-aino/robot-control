//! Segmentation into stance, locomotion, strike, defense, recovery and
//! transition windows, per SPECS.md Section 8.2 item 6.
//!
//! The segmenter is a transparent rule set over kinematic features, not a
//! learned classifier: every frame gets one raw label from the first rule
//! that fires, in this order,
//!
//! 1. **strike**: a hand moves faster than `strike_speed` *and* away from
//!    the head faster than `strike_extension_rate`;
//! 2. **recovery**: within `recovery_window` after a strike frame, a hand
//!    moves faster than `recovery_speed` back toward the head faster than
//!    `recovery_extension_rate`;
//! 3. **defense**: the head moves relative to the root faster than
//!    `evasion_speed` (a slip or duck), or a hand closes on the head faster
//!    than `block_speed` while within `guard_radius` of it (a block);
//! 4. **locomotion**: the root's speed across the ground exceeds
//!    `locomotion_speed`;
//! 5. **stance**: none of the above.
//!
//! Runs of equal raw labels become segments. A segment shorter than its
//! kind's minimum (`min_action` for strike, defense and recovery,
//! `min_posture` for stance and locomotion) becomes a **transition**, and
//! adjacent transitions merge. A segment's confidence is the fraction of
//! its frames whose raw label agrees with it, which is one for every
//! segment that was not produced by merging; a transition has none.
//!
//! The rules encode a boxing-specific reading of motion and every threshold
//! is a field of [`SegmentConfig`], not a validated constant: whether these
//! windows match a human annotator's is an evaluation question this module
//! does not answer.

use core::fmt;
use robot_types::frames::Vec3;
use robot_types::ids::Name;
use robot_types::motion::{MotionKind, MotionRecord, ProvenanceStep, TaskLabel};
use robot_types::time::Duration;

use crate::kinematics::{all_positions, frame_times, velocities};
use crate::REVISION;

/// What a window is.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum SegmentKind {
    /// Standing, guard held, no fast motion.
    Stance,
    /// The root travelling across the ground.
    Locomotion,
    /// A hand driven outward fast.
    Strike,
    /// An evasive head movement or a block.
    Defense,
    /// A hand returning after a strike.
    Recovery,
    /// Too short or ambiguous to be any of the above.
    Transition,
}

impl SegmentKind {
    /// Every kind, in a stable order.
    pub const ALL: [Self; 6] = [
        Self::Stance,
        Self::Locomotion,
        Self::Strike,
        Self::Defense,
        Self::Recovery,
        Self::Transition,
    ];

    /// The task-label name written for this kind.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Stance => "stance",
            Self::Locomotion => "locomotion",
            Self::Strike => "strike",
            Self::Defense => "defense",
            Self::Recovery => "recovery",
            Self::Transition => "transition",
        }
    }

    const fn is_action(self) -> bool {
        matches!(self, Self::Strike | Self::Defense | Self::Recovery)
    }
}

/// The channels the segmenter reads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BodyMap {
    /// Head channel.
    pub head: usize,
    /// Left hand or wrist channel.
    pub left_hand: usize,
    /// Right hand or wrist channel.
    pub right_hand: usize,
}

/// Why segmentation failed.
#[derive(Clone, PartialEq, Debug)]
pub enum SegmentError {
    /// No channel matched any candidate name for this body part.
    MissingBodyPart(&'static str),
    /// A channel index was out of range.
    UnknownChannel(usize),
    /// A threshold was non-positive or non-finite.
    InvalidConfig(&'static str),
    /// The record had fewer than two frames.
    TooShort,
}

impl fmt::Display for SegmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingBodyPart(part) => write!(f, "no channel found for {part}"),
            Self::UnknownChannel(c) => write!(f, "channel {c} does not exist"),
            Self::InvalidConfig(field) => write!(f, "invalid segment config field {field}"),
            Self::TooShort => f.write_str("need at least two frames to segment"),
        }
    }
}

impl core::error::Error for SegmentError {}

const HEAD_NAMES: &[&str] = &["head", "headend", "head_end", "neck2"];
const LEFT_HAND_NAMES: &[&str] = &[
    "lefthand",
    "left_hand",
    "lhand",
    "l_hand",
    "leftwrist",
    "left_wrist",
    "l_wrist",
    "lwrist",
];
const RIGHT_HAND_NAMES: &[&str] = &[
    "righthand",
    "right_hand",
    "rhand",
    "r_hand",
    "rightwrist",
    "right_wrist",
    "r_wrist",
    "rwrist",
];

impl BodyMap {
    /// Finds each body part by case-insensitive exact match against a short
    /// list of names common in BVH and marker sets (`Head`, `LeftHand`,
    /// `l_wrist`, ...). It never guesses by substring: a skeleton with other
    /// names gets [`SegmentError::MissingBodyPart`] and an explicit map.
    pub fn find<K: MotionKind>(record: &MotionRecord<K>) -> Result<Self, SegmentError> {
        let find = |names: &[&str], part: &'static str| {
            names
                .iter()
                .find_map(|n| {
                    record
                        .channels
                        .iter()
                        .position(|c| c.name.as_str().eq_ignore_ascii_case(n))
                })
                .ok_or(SegmentError::MissingBodyPart(part))
        };
        Ok(Self {
            head: find(HEAD_NAMES, "head")?,
            left_hand: find(LEFT_HAND_NAMES, "left hand")?,
            right_hand: find(RIGHT_HAND_NAMES, "right hand")?,
        })
    }
}

/// Thresholds of the rule set. None of them is a validated constant.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SegmentConfig {
    /// Up direction of the record.
    pub up: Vec3,
    /// Hand speed above which an outward motion is a strike, m/s.
    pub strike_speed: f64,
    /// Rate of hand-head distance increase a strike needs, m/s.
    pub strike_extension_rate: f64,
    /// Hand speed above which an inward motion after a strike is recovery.
    pub recovery_speed: f64,
    /// Rate of hand-head distance decrease recovery needs, m/s.
    pub recovery_extension_rate: f64,
    /// How long after a strike frame recovery may start.
    pub recovery_window: Duration,
    /// Head speed relative to the root that counts as evasion, m/s.
    pub evasion_speed: f64,
    /// Inward hand speed near the head that counts as a block, m/s.
    pub block_speed: f64,
    /// Hand-head distance within which an inward hand is a block, m.
    pub guard_radius: f64,
    /// Root ground speed above which the subject is travelling, m/s.
    pub locomotion_speed: f64,
    /// Shortest strike, defense or recovery segment kept as such.
    pub min_action: Duration,
    /// Shortest stance or locomotion segment kept as such.
    pub min_posture: Duration,
}

impl SegmentConfig {
    fn validate(&self) -> Result<Vec3, SegmentError> {
        let fields = [
            (self.strike_speed, "strike_speed"),
            (self.strike_extension_rate, "strike_extension_rate"),
            (self.recovery_speed, "recovery_speed"),
            (self.recovery_extension_rate, "recovery_extension_rate"),
            (self.evasion_speed, "evasion_speed"),
            (self.block_speed, "block_speed"),
            (self.guard_radius, "guard_radius"),
            (self.locomotion_speed, "locomotion_speed"),
        ];
        for (v, name) in fields {
            if !v.is_finite() || v <= 0.0 {
                return Err(SegmentError::InvalidConfig(name));
            }
        }
        if self.min_action.nanos() < 0
            || self.min_posture.nanos() < 0
            || self.recovery_window.nanos() < 0
        {
            return Err(SegmentError::InvalidConfig("durations"));
        }
        self.up
            .normalized(1e-12)
            .ok_or(SegmentError::InvalidConfig("up"))
    }
}

/// One window.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Segment {
    /// What it is.
    pub kind: SegmentKind,
    /// First frame.
    pub start: usize,
    /// One past the last frame.
    pub end: usize,
    /// Fraction of frames whose raw label agrees; `None` for a transition.
    pub confidence: Option<f64>,
}

/// Per-frame raw labels from the rule set, before any smoothing.
pub fn raw_labels<K: MotionKind>(
    record: &MotionRecord<K>,
    body: BodyMap,
    config: &SegmentConfig,
) -> Result<Vec<SegmentKind>, SegmentError> {
    let up = config.validate()?;
    if record.frames.len() < 2 {
        return Err(SegmentError::TooShort);
    }
    for c in [body.head, body.left_hand, body.right_hand] {
        if c >= record.channels.len() {
            return Err(SegmentError::UnknownChannel(c));
        }
    }
    let times = frame_times(record);
    let positions = all_positions(record);
    let track = |c: usize| -> Vec<Option<Vec3>> { positions.iter().map(|p| p[c]).collect() };
    let head = track(body.head);
    let hands = [track(body.left_hand), track(body.right_hand)];
    let root: Vec<Option<Vec3>> = record
        .frames
        .iter()
        .map(|f| Some(f.root.translation))
        .collect();

    let root_vel = velocities(&root, &times);
    let head_rel: Vec<Option<Vec3>> = head
        .iter()
        .zip(&root)
        .map(|(h, r)| Some(h.as_ref().copied()? - r.as_ref().copied()?))
        .collect();
    let head_rel_vel = velocities(&head_rel, &times);
    let hand_vel: Vec<Vec<Option<Vec3>>> = hands.iter().map(|h| velocities(h, &times)).collect();
    let reach: Vec<Vec<Option<Vec3>>> = hands
        .iter()
        .map(|h| {
            h.iter()
                .zip(&head)
                .map(|(a, b)| {
                    Some(Vec3::new(
                        (a.as_ref().copied()? - b.as_ref().copied()?).norm(),
                        0.0,
                        0.0,
                    ))
                })
                .collect()
        })
        .collect();
    let reach_rate: Vec<Vec<Option<f64>>> = reach
        .iter()
        .map(|r| {
            velocities(r, &times)
                .into_iter()
                .map(|v| v.map(|v| v.x))
                .collect()
        })
        .collect();

    let n = record.frames.len();
    let window = config.recovery_window.as_secs_f64();
    let mut last_strike: Option<f64> = None;
    let mut labels = Vec::with_capacity(n);
    for f in 0..n {
        let hand = |k: usize| -> Option<(f64, f64, f64)> {
            Some((hand_vel[k][f]?.norm(), reach_rate[k][f]?, reach[k][f]?.x))
        };
        let striking = (0..2).any(|k| {
            hand(k).is_some_and(|(s, r, _)| {
                s > config.strike_speed && r > config.strike_extension_rate
            })
        });
        if striking {
            last_strike = Some(times[f]);
            labels.push(SegmentKind::Strike);
            continue;
        }
        let in_window = last_strike.is_some_and(|t| times[f] - t <= window);
        let recovering = in_window
            && (0..2).any(|k| {
                hand(k).is_some_and(|(s, r, _)| {
                    s > config.recovery_speed && r < -config.recovery_extension_rate
                })
            });
        if recovering {
            labels.push(SegmentKind::Recovery);
            continue;
        }
        let evading = head_rel_vel[f].is_some_and(|v| v.norm() > config.evasion_speed);
        let blocking = (0..2).any(|k| {
            hand(k).is_some_and(|(s, r, d)| {
                s > config.block_speed && r < 0.0 && d < config.guard_radius
            })
        });
        if evading || blocking {
            labels.push(SegmentKind::Defense);
            continue;
        }
        let ground_speed = root_vel[f].map_or(0.0, |v| (v - up * v.dot(up)).norm());
        labels.push(if ground_speed > config.locomotion_speed {
            SegmentKind::Locomotion
        } else {
            SegmentKind::Stance
        });
    }
    Ok(labels)
}

/// Segments `record`, replacing any existing task labels whose names are
/// segment kinds and leaving every other task label untouched.
pub fn segment<K: MotionKind>(
    record: &mut MotionRecord<K>,
    body: BodyMap,
    config: &SegmentConfig,
) -> Result<Vec<Segment>, SegmentError> {
    let raw = raw_labels(record, body, config)?;
    let times = frame_times(record);
    let dt = if times.len() > 1 {
        times[1] - times[0]
    } else {
        0.0
    };
    let duration = |s: usize, e: usize| times[e - 1] - times[s] + dt;

    let mut runs: Vec<(SegmentKind, usize, usize)> = Vec::new();
    for (f, &k) in raw.iter().enumerate() {
        match runs.last_mut() {
            Some((kind, _, end)) if *kind == k => *end = f + 1,
            _ => runs.push((k, f, f + 1)),
        }
    }
    for run in &mut runs {
        let min = if run.0.is_action() {
            config.min_action
        } else {
            config.min_posture
        };
        if duration(run.1, run.2) < min.as_secs_f64() {
            run.0 = SegmentKind::Transition;
        }
    }
    let mut merged: Vec<(SegmentKind, usize, usize)> = Vec::new();
    for run in runs {
        match merged.last_mut() {
            Some(last) if last.0 == run.0 => last.2 = run.2,
            _ => merged.push(run),
        }
    }
    let segments: Vec<Segment> = merged
        .into_iter()
        .map(|(kind, start, end)| Segment {
            kind,
            start,
            end,
            confidence: (kind != SegmentKind::Transition).then(|| {
                raw[start..end].iter().filter(|&&r| r == kind).count() as f64 / (end - start) as f64
            }),
        })
        .collect();

    record.task_labels.retain(|l| {
        !SegmentKind::ALL
            .iter()
            .any(|k| l.label.as_str() == k.label())
    });
    for s in &segments {
        record.task_labels.push(TaskLabel {
            label: Name::new(s.kind.label()).expect("static label names are valid"),
            start: s.start,
            end: s.end,
        });
    }
    record.record_step(
        ProvenanceStep::new("segment", REVISION)
            .with_parameter("segments", segments.len().to_string())
            .with_parameter("strike_speed_mps", config.strike_speed.to_string())
            .with_parameter("locomotion_speed_mps", config.locomotion_speed.to_string()),
    );
    Ok(segments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{channel, frame, record};
    use robot_types::frames::Quat;
    use robot_types::motion::{HumanSkeleton, JointValue};

    fn config() -> SegmentConfig {
        SegmentConfig {
            up: Vec3::unit_z(),
            strike_speed: 3.0,
            strike_extension_rate: 2.0,
            recovery_speed: 1.5,
            recovery_extension_rate: 1.0,
            recovery_window: Duration::from_millis(400),
            evasion_speed: 0.6,
            block_speed: 1.5,
            guard_radius: 0.3,
            locomotion_speed: 0.4,
            min_action: Duration::from_millis(40),
            min_posture: Duration::from_millis(100),
        }
    }

    /// A scripted 3 s bout at 100 Hz: stance, a right straight (0.5-0.6 s),
    /// its retraction (0.6-0.8 s), stance, a 1 m/s step-in (1.5-2.5 s), a
    /// lateral slip (2.5-2.7 s), stance.
    fn bout() -> MotionRecord<HumanSkeleton> {
        let channels = vec![
            channel("Hips", None, Vec3::ZERO),
            channel("Head", None, Vec3::ZERO),
            channel("LeftHand", None, Vec3::ZERO),
            channel("RightHand", None, Vec3::ZERO),
        ];
        let ramp = |t: f64, a: f64, b: f64| ((t - a) / (b - a)).clamp(0.0, 1.0);
        let frames = (0..300)
            .map(|i| {
                let t = i as f64 / 100.0;
                let root = Vec3::new(ramp(t, 1.5, 2.5), 0.0, 1.0);
                let slip = 0.2 * ramp(t, 2.5, 2.7);
                let head = root + Vec3::new(0.0, slip, 0.6);
                let punch = 0.5 * (ramp(t, 0.5, 0.6) - ramp(t, 0.6, 0.8));
                let lh = root + Vec3::new(0.15, 0.15, 0.5);
                let rh = root + Vec3::new(0.15 + punch, -0.15, 0.5);
                frame(
                    t,
                    root,
                    Quat::IDENTITY,
                    vec![
                        JointValue::Rotation {
                            rotation: Quat::IDENTITY,
                            source_order: None,
                        },
                        JointValue::Position { position: head },
                        JointValue::Position { position: lh },
                        JointValue::Position { position: rh },
                    ],
                )
            })
            .collect();
        record(channels, frames)
    }

    #[test]
    fn body_parts_are_found_by_exact_common_names_only() {
        let r = bout();
        assert_eq!(
            BodyMap::find(&r).unwrap(),
            BodyMap {
                head: 1,
                left_hand: 2,
                right_hand: 3
            }
        );
        let mut renamed = r.clone();
        renamed.channels[1].name = Name::new("Headset").unwrap();
        assert_eq!(
            BodyMap::find(&renamed),
            Err(SegmentError::MissingBodyPart("head"))
        );
    }

    #[test]
    fn a_scripted_bout_segments_in_order_at_the_scripted_times() {
        let mut r = bout();
        let body = BodyMap::find(&r).unwrap();
        let segments = segment(&mut r, body, &config()).unwrap();
        let kinds: Vec<_> = segments
            .iter()
            .filter(|s| s.kind != SegmentKind::Transition)
            .map(|s| s.kind)
            .collect();
        use SegmentKind::*;
        assert_eq!(
            kinds,
            [Stance, Strike, Recovery, Stance, Locomotion, Defense, Stance],
            "{segments:?}"
        );
        let strike = segments.iter().find(|s| s.kind == Strike).unwrap();
        assert!(
            strike.start.abs_diff(50) <= 2 && strike.end.abs_diff(60) <= 2,
            "{strike:?}"
        );
        let walk = segments.iter().find(|s| s.kind == Locomotion).unwrap();
        assert!(
            walk.start.abs_diff(150) <= 2 && walk.end.abs_diff(250) <= 2,
            "{walk:?}"
        );
        assert!(segments
            .iter()
            .all(|s| s.kind == Transition || s.confidence == Some(1.0)));
        // Segments tile the record.
        assert_eq!(segments[0].start, 0);
        assert!(segments.windows(2).all(|w| w[0].end == w[1].start));
        assert_eq!(segments.last().unwrap().end, 300);
        r.validate().unwrap();
        assert_eq!(r.task_labels.len(), segments.len());
    }

    #[test]
    fn resegmenting_replaces_only_segment_labels() {
        let mut r = bout();
        r.task_labels.push(TaskLabel {
            label: Name::new("jab").unwrap(),
            start: 50,
            end: 80,
        });
        let body = BodyMap::find(&r).unwrap();
        segment(&mut r, body, &config()).unwrap();
        let first = r.task_labels.len();
        segment(&mut r, body, &config()).unwrap();
        assert_eq!(r.task_labels.len(), first);
        assert_eq!(
            r.task_labels
                .iter()
                .filter(|l| l.label.as_str() == "jab")
                .count(),
            1
        );
    }

    #[test]
    fn a_short_flicker_becomes_a_transition() {
        let mut cfg = config();
        cfg.min_action = Duration::from_millis(500); // longer than any action here
        let mut r = bout();
        let body = BodyMap::find(&r).unwrap();
        let segments = segment(&mut r, body, &cfg).unwrap();
        assert!(segments.iter().all(|s| !s.kind.is_action()));
        assert!(segments.iter().any(|s| s.kind == SegmentKind::Transition));
    }

    #[test]
    fn a_non_positive_threshold_is_rejected() {
        let mut cfg = config();
        cfg.strike_speed = 0.0;
        let r = bout();
        assert_eq!(
            raw_labels(&r, BodyMap::find(&r).unwrap(), &cfg),
            Err(SegmentError::InvalidConfig("strike_speed"))
        );
    }
}

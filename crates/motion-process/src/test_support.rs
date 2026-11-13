//! Record builders shared by this crate's unit tests.

use robot_types::frames::{FrameId, Quat, Transform, Vec3};
use robot_types::ids::Name;
use robot_types::motion::{
    Channel, HumanSkeleton, JointValue, MotionFrame, MotionRecord, SourceLicense, Validity,
};
use robot_types::time::{ClockId, Timestamp};

pub fn license() -> SourceLicense {
    SourceLicense {
        identifier: "fixture-internal".to_owned(),
        training_permitted: true,
        redistribution_permitted: false,
        commercial_permitted: false,
        retention: None,
        source_reference: "fixtures/motion/README.md".to_owned(),
    }
}

pub fn uniform_times(n: usize, dt: f64) -> Vec<f64> {
    (0..n).map(|i| i as f64 * dt).collect()
}

pub fn channel(name: &str, parent: Option<usize>, offset: Vec3) -> Channel {
    Channel {
        name: Name::new(name).unwrap(),
        parent,
        offset,
    }
}

pub fn rotation(q: Quat) -> JointValue {
    JointValue::Rotation {
        rotation: q,
        source_order: None,
    }
}

pub fn frame(
    t: f64,
    root_translation: Vec3,
    root_rotation: Quat,
    values: Vec<JointValue>,
) -> MotionFrame {
    MotionFrame {
        timestamp: Timestamp::sampling((t * 1e9).round() as i64, ClockId::Source(0)),
        root: Transform::new(
            FrameId::WORLD,
            FrameId::WORLD,
            root_rotation,
            root_translation,
        ),
        validity: vec![Validity::Valid; values.len()],
        values,
        contacts: Vec::new(),
    }
}

pub fn record(channels: Vec<Channel>, frames: Vec<MotionFrame>) -> MotionRecord<HumanSkeleton> {
    MotionRecord::new(
        HumanSkeleton,
        "sha256:fixture",
        license(),
        "session",
        "skeleton",
        FrameId::WORLD,
        ClockId::Source(0),
        channels,
        frames,
    )
    .unwrap()
}

/// A root with one child offset 1 m along +x, one frame per root pose.
pub fn chain_record(poses: &[(Vec3, Quat)], rate_hz: f64) -> MotionRecord<HumanSkeleton> {
    let channels = vec![
        channel("root", None, Vec3::ZERO),
        channel("child", Some(0), Vec3::new(1.0, 0.0, 0.0)),
    ];
    let frames = poses
        .iter()
        .enumerate()
        .map(|(i, (t, q))| {
            frame(
                i as f64 / rate_hz,
                *t,
                *q,
                vec![rotation(*q), rotation(Quat::IDENTITY)],
            )
        })
        .collect();
    record(channels, frames)
}

/// A root whose translation follows `f(t)` sampled at `rate_hz`, plus the
/// one +x child of [`chain_record`].
pub fn root_path(n: usize, rate_hz: f64, f: impl Fn(f64) -> Vec3) -> MotionRecord<HumanSkeleton> {
    let poses: Vec<_> = (0..n)
        .map(|i| (f(i as f64 / rate_hz), Quat::IDENTITY))
        .collect();
    chain_record(&poses, rate_hz)
}

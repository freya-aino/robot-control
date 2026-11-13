//! Fixtures shared by this crate's unit tests.

use robot_model::fixtures::unitree_g1_29dof;
use robot_model::kinematics::forward_kinematics;
use robot_model::profile::RobotProfile;
use robot_types::frames::{FrameId, Quat, Transform, Vec3};
use robot_types::ids::ContactSiteId;
use robot_types::motion::{
    Channel, ContactLabel, JointValue, MotionFrame, MotionRecord, RobotReference, SourceLicense,
    Validity,
};
use robot_types::time::{ClockId, Timestamp};

use crate::dynamics::ContactSpec;

pub fn g1() -> RobotProfile {
    unitree_g1_29dof().unwrap()
}

pub fn link(profile: &RobotProfile, name: &str) -> usize {
    profile
        .links
        .iter()
        .position(|l| l.name.as_str() == name)
        .unwrap()
}

pub fn feet(profile: &RobotProfile) -> Vec<ContactSpec> {
    vec![
        ContactSpec {
            site: ContactSiteId::new(0),
            link: link(profile, "left_ankle_roll_link"),
        },
        ContactSpec {
            site: ContactSiteId::new(1),
            link: link(profile, "right_ankle_roll_link"),
        },
    ]
}

/// A reference at 50 Hz whose base pose and joints follow `f(t)`, with no
/// contact labels.
pub fn reference_from(
    profile: &RobotProfile,
    frames: usize,
    f: impl Fn(f64) -> (Vec3, Quat, Vec<f64>),
) -> MotionRecord<RobotReference> {
    let channels: Vec<Channel> = profile
        .actuated_joints()
        .map(|j| Channel {
            name: j.name.clone(),
            parent: None,
            offset: Vec3::ZERO,
        })
        .collect();
    let frames = (0..frames)
        .map(|i| {
            let t = i as f64 * 0.02;
            let (p, r, q) = f(t);
            MotionFrame {
                timestamp: Timestamp::sampling(i as i64 * 20_000_000, ClockId::Source(0)),
                root: Transform::new(FrameId::WORLD, FrameId::WORLD, r, p),
                validity: vec![Validity::Valid; q.len()],
                values: q
                    .into_iter()
                    .map(|v| JointValue::Scalar {
                        value: v,
                        unit: "rad",
                    })
                    .collect(),
                contacts: Vec::new(),
            }
        })
        .collect();
    MotionRecord::new(
        RobotReference,
        "sha256:fixture",
        SourceLicense::unaudited("fixture-internal"),
        "session",
        "g1",
        FrameId::WORLD,
        ClockId::Source(0),
        channels,
        frames,
    )
    .unwrap()
}

/// The G1 at its zero posture with its ankle links at height zero, both
/// feet labelled in contact when `labelled`.
pub fn standing_reference(
    profile: &RobotProfile,
    frames: usize,
    labelled: bool,
) -> (MotionRecord<RobotReference>, Vec<ContactSpec>) {
    let feet = feet(profile);
    let zero = vec![0.0; profile.dof()];
    let poses = forward_kinematics(profile, &zero).unwrap();
    let height = -poses.by_index(feet[0].link).unwrap().translation.z;
    let mut r = reference_from(profile, frames, |_| {
        (Vec3::new(0.0, 0.0, height), Quat::IDENTITY, zero.clone())
    });
    if labelled {
        for f in &mut r.frames {
            for c in &feet {
                f.contacts.push(ContactLabel {
                    site: c.site,
                    in_contact: true,
                    confidence: 0.9,
                });
            }
        }
    }
    (r, feet)
}

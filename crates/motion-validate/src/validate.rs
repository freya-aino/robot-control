//! The whole of SPECS.md Section 8.2 item 8 in one call: a retargeted
//! reference checked against its human source, its robot and (optionally)
//! a tracking simulation, producing a [`ValidationReport`].

use core::fmt;
use motion_retarget::retarget::RetargetMap;
use robot_model::profile::RobotProfile;
use robot_types::motion::{HumanSkeleton, MotionRecord, RobotReference};

use crate::contact::{check, ContactFrame, ContactTolerances};
use crate::dynamics::{
    estimate, position_excess, torque_excess, velocity_excess, ContactSpec, DynamicsError,
};
use crate::kinematic::measure;
use crate::report::{DynamicsSummary, DynamicsViolation, ValidationReport};
use crate::tracking::{track, TrackingConfig, TrackingError};
use crate::trajectory::{Trajectory, TrajectoryError};

/// What to check and how.
#[derive(Clone, PartialEq, Debug)]
pub struct ValidationConfig {
    /// The map the reference was retargeted with.
    pub map: RetargetMap,
    /// Body size the endpoint error is normalized by, metres (the robot's
    /// standing height, say).
    pub body_size: f64,
    /// Contact consistency tolerances (its `contact_confidence` also
    /// decides which feet are planted).
    pub contact: ContactTolerances,
    /// Moment weighting length of the contact distribution, metres.
    pub moment_length: f64,
    /// Simulated tracking, when wanted.
    pub tracking: Option<TrackingConfig>,
    /// Feed the inverse-dynamics torques forward while tracking.
    pub feedforward: bool,
}

/// Why validation could not run (as opposed to a rejection).
#[derive(Clone, PartialEq, Debug)]
pub enum ValidateError {
    /// The reference could not be read.
    Trajectory(TrajectoryError),
    /// Inverse dynamics failed.
    Dynamics(DynamicsError),
    /// Tracking failed to run.
    Tracking(TrackingError),
    /// The human and reference frame counts differ.
    FrameCountMismatch {
        /// Human frames.
        human: usize,
        /// Reference frames.
        reference: usize,
    },
    /// `body_size` or `moment_length` was not positive.
    InvalidConfig(&'static str),
}

impl fmt::Display for ValidateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Trajectory(e) => write!(f, "{e}"),
            Self::Dynamics(e) => write!(f, "{e}"),
            Self::Tracking(e) => write!(f, "{e}"),
            Self::FrameCountMismatch { human, reference } => {
                write!(f, "human has {human} frames, reference {reference}")
            }
            Self::InvalidConfig(what) => write!(f, "invalid validation config: {what}"),
        }
    }
}

impl core::error::Error for ValidateError {}

/// Validates `reference` (retargeted from `human` onto `profile`).
pub fn validate(
    human: &MotionRecord<HumanSkeleton>,
    reference: &MotionRecord<RobotReference>,
    profile: &RobotProfile,
    config: &ValidationConfig,
) -> Result<ValidationReport, ValidateError> {
    if !(config.body_size > 0.0 && config.body_size.is_finite()) {
        return Err(ValidateError::InvalidConfig("body_size"));
    }
    if !(config.moment_length > 0.0 && config.moment_length.is_finite()) {
        return Err(ValidateError::InvalidConfig("moment_length"));
    }
    if human.frames.len() != reference.frames.len() {
        return Err(ValidateError::FrameCountMismatch {
            human: human.frames.len(),
            reference: reference.frames.len(),
        });
    }
    let traj =
        Trajectory::from_reference(reference, profile.dof()).map_err(ValidateError::Trajectory)?;
    let contacts: Vec<ContactSpec> = config
        .map
        .feet
        .iter()
        .map(|f| ContactSpec {
            site: f.site,
            link: f.robot,
        })
        .collect();
    let dynamics = estimate(
        profile,
        reference,
        &traj,
        &contacts,
        config.contact.contact_confidence,
        config.moment_length,
    )
    .map_err(ValidateError::Dynamics)?;

    let violations: Vec<DynamicsViolation> = (0..traj.len())
        .map(|f| DynamicsViolation {
            position_limit_excess: position_excess(profile, &traj.q[f]),
            velocity_limit_excess: velocity_excess(profile, &traj.qd[f]),
            torque_excess: torque_excess(profile, &dynamics[f].joint_torque, &traj.qd[f]),
        })
        .collect();
    let dof = profile.dof();
    let peak = |rows: &dyn Fn(usize) -> f64| (0..traj.len()).map(rows).fold(0.0, f64::max);
    let summary = DynamicsSummary {
        peak_acceleration: (0..dof).map(|j| peak(&|f| traj.qdd[f][j].abs())).collect(),
        rms_jerk: (0..dof)
            .map(|j| {
                (traj.qddd.iter().map(|r| r[j] * r[j]).sum::<f64>() / traj.len() as f64).sqrt()
            })
            .collect(),
        peak_torque: (0..dof)
            .map(|j| peak(&|f| dynamics[f].joint_torque[j].abs()))
            .collect(),
    };
    let contact = check(
        profile,
        reference,
        &traj,
        &dynamics,
        &contacts,
        &config.contact,
    );
    let cop = crate::cop::check(
        profile,
        &traj,
        &dynamics,
        &contacts,
        config.contact.ground_height,
        config.contact.force_tolerance,
    );
    let contact_inconsistent_frames = contact
        .iter()
        .enumerate()
        .filter(|(_, c)| !ContactFrame::is_consistent(c))
        .map(|(i, _)| i)
        .collect();
    let kinematic = measure(
        human,
        profile,
        &traj,
        &config.map,
        config.body_size,
        config.contact.contact_confidence,
    );
    let tracking = match &config.tracking {
        Some(tc) => Some(
            track(
                profile,
                reference,
                &traj,
                config.feedforward.then_some(dynamics.as_slice()),
                tc,
            )
            .map_err(ValidateError::Tracking)?,
        ),
        None => None,
    };
    Ok(ValidationReport {
        kinematic,
        dynamics: violations,
        summary,
        contact,
        contact_inconsistent_frames,
        cop,
        requires_tracking_evaluation: tracking.is_none(),
        tracking,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::{AcceptanceThresholds, Verdict};
    use crate::test_support::g1;
    use motion_retarget::ik::{BasePrior, IkConfig};
    use motion_retarget::retarget::{retarget, Anchor, FootMapping, LinkMapping, RetargetConfig};
    use robot_types::frames::{FrameId, Quat, Transform, Vec3};
    use robot_types::ids::{ContactSiteId, Name};
    use robot_types::motion::{
        Channel, ContactLabel, JointValue, MotionFrame, SourceLicense, Validity,
    };
    use robot_types::time::{ClockId, Timestamp};

    /// A human standing with both feet planted, the right hand pushing
    /// slowly forward.
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
        let frames = (0..6)
            .map(|i| {
                let s = i as f64 / 5.0;
                MotionFrame {
                    timestamp: Timestamp::sampling(i * 40_000_000, ClockId::Source(0)),
                    root: Transform::new(
                        FrameId::WORLD,
                        FrameId::WORLD,
                        Quat::IDENTITY,
                        Vec3::new(0.0, 0.0, 0.9),
                    ),
                    values: vec![
                        rot.clone(),
                        rot.clone(),
                        JointValue::Position {
                            position: Vec3::new(0.25 + 0.1 * s, -0.2, 1.3),
                        },
                        rot.clone(),
                        rot.clone(),
                    ],
                    validity: vec![Validity::Valid; 5],
                    contacts: (0..2)
                        .map(|k| ContactLabel {
                            site: ContactSiteId::new(k),
                            in_contact: true,
                            confidence: 0.9,
                        })
                        .collect(),
                }
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
                ch("LeftFoot", Some(0), Vec3::new(0.0, 0.12, -0.9)),
                ch("RightFoot", Some(0), Vec3::new(0.0, -0.12, -0.9)),
            ],
            frames,
        )
        .unwrap()
    }

    #[test]
    fn a_retargeted_reach_is_validated_end_to_end() {
        let profile = g1();
        let link = |n: &str| {
            profile
                .links
                .iter()
                .position(|l| l.name.as_str() == n)
                .unwrap()
        };
        let h = human();
        let zero = vec![0.0; profile.dof()];
        let ratio = motion_retarget::retarget::chain_ratio(
            &h,
            1,
            2,
            &profile,
            link("right_shoulder_pitch_link"),
            link("right_rubber_hand"),
            &zero,
        )
        .unwrap();
        let map = motion_retarget::retarget::RetargetMap {
            scale: 0.8,
            root_offset: Quat::IDENTITY,
            links: vec![LinkMapping {
                human: 2,
                robot: link("right_rubber_hand"),
                anchor: Anchor::Relative {
                    human_base: 1,
                    robot_base: link("right_shoulder_pitch_link"),
                    ratio,
                },
                position_weight: 1.0,
                orientation: None,
            }],
            feet: vec![
                FootMapping {
                    site: ContactSiteId::new(0),
                    human: 3,
                    robot: link("left_ankle_roll_link"),
                    weight: 100.0,
                    ground_height: 0.0,
                },
                FootMapping {
                    site: ContactSiteId::new(1),
                    human: 4,
                    robot: link("right_ankle_roll_link"),
                    weight: 100.0,
                    ground_height: 0.0,
                },
            ],
        };
        let out = retarget(
            &h,
            &profile,
            &RetargetConfig {
                map: map.clone(),
                ik: IkConfig {
                    max_iterations: 60,
                    tolerance: 1e-5,
                    ..IkConfig::default()
                },
                base_prior: Some(BasePrior {
                    translation_weight: 1e-2,
                    rotation_weight: 1e-1,
                }),
                posture: Some((zero.clone(), vec![1e-3; profile.dof()])),
                collision: None,
                smoothing: None,
                contact_confidence: 0.5,
                initial: None,
            },
        )
        .unwrap();
        let config = ValidationConfig {
            map,
            body_size: 1.3,
            contact: ContactTolerances {
                ground_height: 0.0,
                sole_height: 0.0,
                height_tolerance: 0.02,
                force_tolerance: 5.0,
                friction: 0.8,
                contact_confidence: 0.5,
            },
            moment_length: 0.1,
            tracking: None,
            feedforward: false,
        };
        let report = validate(&h, &out.reference, &profile, &config).unwrap();
        assert_eq!(report.kinematic.len(), 6);
        assert!(report.requires_tracking_evaluation);
        for k in &report.kinematic {
            assert!(k.normalized_endpoint_error < 0.02, "{k:?}");
            assert!(k.planted_foot_slip.unwrap() < 0.01, "{k:?}");
        }
        // Standing reach: the feet carry the weight and nothing pulls.
        for c in &report.contact {
            assert!(
                c.pulling.is_empty() && c.unsupported_force.is_none(),
                "{c:?}"
            );
        }
        // The standing reach keeps its centre of pressure inside the feet,
        // so `min_cop_margin: 0.0` is a genuine check here, not a bypass.
        let thresholds = AcceptanceThresholds {
            max_normalized_endpoint_error: 0.02,
            max_planted_foot_slip: 0.01,
            max_root_drift: 0.2,
            allow_limit_violations: false,
            allow_torque_violations: true,
            allow_contact_inconsistency: true,
            max_tracking_rms: 0.05,
            require_tracking: false,
            min_cop_margin: 0.0,
            allow_unchecked_cop: false,
            // The derived envelope bounds are a genuine check here too: a
            // gentle standing reach must sit far inside what the G1's own
            // declared efforts and inertias permit.
            motion_limits: Some(crate::dynamics::derived_motion_limits(&profile).unwrap()),
        };
        assert_eq!(report.evaluate(&thresholds), Verdict::Accepted);
        let strict = AcceptanceThresholds {
            require_tracking: true,
            ..thresholds.clone()
        };
        assert!(matches!(report.evaluate(&strict), Verdict::Rejected(_)));
        // An absurd clip fails the derived gate: scale the report's peaks
        // past every joint's bound and the codes name the joints.
        let mut violent = report.clone();
        for (v, b) in violent
            .summary
            .peak_acceleration
            .iter_mut()
            .zip(&thresholds.motion_limits.as_ref().unwrap().max_acceleration)
        {
            *v = b * 2.0;
        }
        let Verdict::Rejected(reasons) = violent.evaluate(&thresholds) else {
            panic!("expected the over-violent clip to be rejected");
        };
        assert!(reasons.iter().any(|r| r.code() == "acceleration_excess"));
    }

    #[test]
    fn a_frame_count_mismatch_is_an_error_not_a_rejection() {
        let profile = g1();
        let h = human();
        let (r, _) = crate::test_support::standing_reference(&profile, 4, true);
        let config = ValidationConfig {
            map: motion_retarget::retarget::RetargetMap {
                scale: 1.0,
                root_offset: Quat::IDENTITY,
                links: Vec::new(),
                feet: Vec::new(),
            },
            body_size: 1.0,
            contact: ContactTolerances {
                ground_height: 0.0,
                sole_height: 0.0,
                height_tolerance: 0.02,
                force_tolerance: 5.0,
                friction: 0.8,
                contact_confidence: 0.5,
            },
            moment_length: 0.1,
            tracking: None,
            feedforward: false,
        };
        assert_eq!(
            validate(&h, &r, &profile, &config).unwrap_err(),
            ValidateError::FrameCountMismatch {
                human: 6,
                reference: 4
            }
        );
    }
}

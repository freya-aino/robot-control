//! Retargeting validation reports, kept to separate concerns.
//!
//! SPECS.md Section 8.3: "A validation report must separate kinematic
//! error, dynamics violations, contact consistency and task success."
//! [`ValidationReport`] has one field per concern, never a single pass/fail
//! bit, and "No universal fidelity percentage is a release criterion"
//! (same section): [`AcceptanceThresholds`] is a value the caller supplies
//! per robot and task, not a constant this crate bakes in.
//!
//! A rejection names every violating frame, not only the first, and every
//! [`RejectionReason`] has a stable [`RejectionReason::code`] so a curation
//! manifest can retain rejected examples "with machine-readable reasons"
//! (Section 8.2 item 9).

use core::fmt;
use robot_model::joint::JointSpec;
use robot_types::units::Radians;

use crate::contact::ContactFrame;
use crate::dynamics::TorqueExcess;
use crate::tracking::TrackingResult;

/// Kinematic error at one frame.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct KinematicError {
    /// Worst mapped-endpoint error divided by body size.
    pub normalized_endpoint_error: f64,
    /// Horizontal distance of the base from the scaled human root, metres.
    pub root_drift: f64,
    /// Worst planted-foot slip, metres; `None` with no foot planted.
    pub planted_foot_slip: Option<f64>,
}

/// Limit violations at one frame.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct DynamicsViolation {
    /// `(joint, excess)` beyond the position range.
    pub position_limit_excess: Vec<(usize, Radians)>,
    /// `(joint, speed)` beyond the velocity limit.
    pub velocity_limit_excess: Vec<(usize, f64)>,
    /// Torques beyond the actuator's torque-speed envelope.
    pub torque_excess: Vec<TorqueExcess>,
}

impl DynamicsViolation {
    /// Whether no limit was exceeded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.position_limit_excess.is_empty()
            && self.velocity_limit_excess.is_empty()
            && self.torque_excess.is_empty()
    }
}

/// Per-joint summaries over the whole reference.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct DynamicsSummary {
    /// Peak |acceleration| per joint.
    pub peak_acceleration: Vec<f64>,
    /// RMS jerk per joint: the temporal smoothness of SPECS.md Section 8.2
    /// item 8.
    pub rms_jerk: Vec<f64>,
    /// Peak |torque| per joint.
    pub peak_torque: Vec<f64>,
}

/// The full validation report.
#[derive(Clone, PartialEq, Debug)]
pub struct ValidationReport {
    /// Kinematic error per frame.
    pub kinematic: Vec<KinematicError>,
    /// Limit violations per frame.
    pub dynamics: Vec<DynamicsViolation>,
    /// Per-joint dynamics summaries.
    pub summary: DynamicsSummary,
    /// Contact consistency per frame.
    pub contact: Vec<ContactFrame>,
    /// Frames with any contact inconsistency.
    pub contact_inconsistent_frames: Vec<usize>,
    /// Simulated tracking, when it was run.
    pub tracking: Option<TrackingResult>,
    /// Whether the report still lacks a tracking evaluation.
    pub requires_tracking_evaluation: bool,
}

/// What the caller accepts. Every field is the caller's choice.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct AcceptanceThresholds {
    /// Largest normalized endpoint error.
    pub max_normalized_endpoint_error: f64,
    /// Largest planted-foot slip, metres.
    pub max_planted_foot_slip: f64,
    /// Largest root drift, metres.
    pub max_root_drift: f64,
    /// Accept position and velocity limit violations.
    pub allow_limit_violations: bool,
    /// Accept torque-speed envelope violations.
    pub allow_torque_violations: bool,
    /// Accept contact inconsistencies.
    pub allow_contact_inconsistency: bool,
    /// Largest RMS joint tracking error of any labelled skill, radians.
    pub max_tracking_rms: f64,
    /// Refuse a report with no tracking evaluation.
    pub require_tracking: bool,
}

/// Accept or reject.
#[derive(Clone, PartialEq, Debug)]
pub enum Verdict {
    /// Every check passed.
    Accepted,
    /// Every reason found.
    Rejected(Vec<RejectionReason>),
}

/// Why a reference was rejected.
#[derive(Clone, PartialEq, Debug)]
pub enum RejectionReason {
    /// Endpoint error over the threshold.
    EndpointError {
        /// Frame.
        frame: usize,
        /// Value.
        value: f64,
        /// Threshold.
        threshold: f64,
    },
    /// Planted-foot slip over the threshold.
    FootSlip {
        /// Frame.
        frame: usize,
        /// Value.
        value: f64,
        /// Threshold.
        threshold: f64,
    },
    /// Root drift over the threshold.
    RootDrift {
        /// Frame.
        frame: usize,
        /// Value.
        value: f64,
        /// Threshold.
        threshold: f64,
    },
    /// A position or velocity limit was exceeded.
    LimitViolation {
        /// Frame.
        frame: usize,
    },
    /// A torque-speed envelope was exceeded.
    TorqueViolation {
        /// Frame.
        frame: usize,
        /// Worst required over available ratio.
        value: f64,
    },
    /// Labels, geometry or dynamics disagreed about contact.
    ContactInconsistent {
        /// Frame.
        frame: usize,
    },
    /// A labelled skill was not tracked well enough.
    TrackingFailed {
        /// The label.
        label: String,
        /// Its RMS joint error.
        value: f64,
        /// Threshold.
        threshold: f64,
    },
    /// The robot fell while tracking.
    Fell {
        /// Frame.
        frame: usize,
    },
    /// Tracking was required and not run.
    TrackingMissing,
}

impl RejectionReason {
    /// A stable machine-readable code.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::EndpointError { .. } => "endpoint_error",
            Self::FootSlip { .. } => "foot_slip",
            Self::RootDrift { .. } => "root_drift",
            Self::LimitViolation { .. } => "limit_violation",
            Self::TorqueViolation { .. } => "torque_violation",
            Self::ContactInconsistent { .. } => "contact_inconsistent",
            Self::TrackingFailed { .. } => "tracking_failed",
            Self::Fell { .. } => "fell",
            Self::TrackingMissing => "tracking_missing",
        }
    }

    /// The frame the reason applies to, if it is frame-specific.
    #[must_use]
    pub const fn frame(&self) -> Option<usize> {
        match self {
            Self::EndpointError { frame, .. }
            | Self::FootSlip { frame, .. }
            | Self::RootDrift { frame, .. }
            | Self::LimitViolation { frame }
            | Self::TorqueViolation { frame, .. }
            | Self::ContactInconsistent { frame }
            | Self::Fell { frame } => Some(*frame),
            Self::TrackingFailed { .. } | Self::TrackingMissing => None,
        }
    }
}

impl fmt::Display for RejectionReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EndpointError {
                frame,
                value,
                threshold,
            } => write!(
                f,
                "frame {frame}: endpoint error {value} exceeds threshold {threshold}"
            ),
            Self::FootSlip {
                frame,
                value,
                threshold,
            } => write!(
                f,
                "frame {frame}: foot slip {value} exceeds threshold {threshold}"
            ),
            Self::RootDrift {
                frame,
                value,
                threshold,
            } => write!(
                f,
                "frame {frame}: root drift {value} exceeds threshold {threshold}"
            ),
            Self::LimitViolation { frame } => write!(f, "frame {frame}: joint-limit violation"),
            Self::TorqueViolation { frame, value } => {
                write!(f, "frame {frame}: torque {value}x the available envelope")
            }
            Self::ContactInconsistent { frame } => write!(f, "frame {frame}: contact inconsistent"),
            Self::TrackingFailed {
                label,
                value,
                threshold,
            } => write!(
                f,
                "skill {label}: tracking RMS {value} exceeds threshold {threshold}"
            ),
            Self::Fell { frame } => write!(f, "frame {frame}: fell while tracking"),
            Self::TrackingMissing => f.write_str("tracking evaluation required but not run"),
        }
    }
}

impl ValidationReport {
    /// Evaluates the report against `thresholds`.
    #[must_use]
    pub fn evaluate(&self, thresholds: &AcceptanceThresholds) -> Verdict {
        let mut reasons = Vec::new();
        for (i, k) in self.kinematic.iter().enumerate() {
            if k.normalized_endpoint_error > thresholds.max_normalized_endpoint_error {
                reasons.push(RejectionReason::EndpointError {
                    frame: i,
                    value: k.normalized_endpoint_error,
                    threshold: thresholds.max_normalized_endpoint_error,
                });
            }
            if let Some(slip) = k.planted_foot_slip {
                if slip > thresholds.max_planted_foot_slip {
                    reasons.push(RejectionReason::FootSlip {
                        frame: i,
                        value: slip,
                        threshold: thresholds.max_planted_foot_slip,
                    });
                }
            }
            if k.root_drift > thresholds.max_root_drift {
                reasons.push(RejectionReason::RootDrift {
                    frame: i,
                    value: k.root_drift,
                    threshold: thresholds.max_root_drift,
                });
            }
        }
        for (i, d) in self.dynamics.iter().enumerate() {
            if !thresholds.allow_limit_violations
                && !(d.position_limit_excess.is_empty() && d.velocity_limit_excess.is_empty())
            {
                reasons.push(RejectionReason::LimitViolation { frame: i });
            }
            if !thresholds.allow_torque_violations && !d.torque_excess.is_empty() {
                let worst = d
                    .torque_excess
                    .iter()
                    .map(|t| t.required.abs() / t.available.max(f64::MIN_POSITIVE))
                    .fold(0.0, f64::max);
                reasons.push(RejectionReason::TorqueViolation {
                    frame: i,
                    value: worst,
                });
            }
        }
        if !thresholds.allow_contact_inconsistency {
            for &frame in &self.contact_inconsistent_frames {
                reasons.push(RejectionReason::ContactInconsistent { frame });
            }
        }
        match &self.tracking {
            Some(t) => {
                if let Some(frame) = t.fell_at {
                    reasons.push(RejectionReason::Fell { frame });
                }
                for s in &t.per_label {
                    if s.joint_rms > thresholds.max_tracking_rms {
                        reasons.push(RejectionReason::TrackingFailed {
                            label: s.label.clone(),
                            value: s.joint_rms,
                            threshold: thresholds.max_tracking_rms,
                        });
                    }
                }
            }
            None if thresholds.require_tracking => reasons.push(RejectionReason::TrackingMissing),
            None => {}
        }
        if reasons.is_empty() {
            Verdict::Accepted
        } else {
            Verdict::Rejected(reasons)
        }
    }
}

/// Checks `values` against each joint's declared position range.
#[must_use]
pub fn check_position_limits(joints: &[JointSpec], values: &[f64]) -> DynamicsViolation {
    let mut violation = DynamicsViolation::default();
    for (i, (joint, &v)) in joints.iter().zip(values).enumerate() {
        if let Some(range) = joint.limits.position {
            if !range.contains(Radians(v)) {
                violation
                    .position_limit_excess
                    .push((i, range.excess(Radians(v))));
            }
        }
    }
    violation
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracking::SkillTracking;
    use robot_model::joint::{JointKind, JointLimits};
    use robot_model::pose::RigidPose;
    use robot_types::frames::Vec3;
    use robot_types::ids::{JointId, LinkId, Name};
    use robot_types::units::{NewtonMetres, RadiansPerSecond, Range};

    fn report(endpoint_errors: &[f64]) -> ValidationReport {
        ValidationReport {
            kinematic: endpoint_errors
                .iter()
                .map(|&e| KinematicError {
                    normalized_endpoint_error: e,
                    root_drift: 0.0,
                    planted_foot_slip: None,
                })
                .collect(),
            dynamics: Vec::new(),
            summary: DynamicsSummary::default(),
            contact: Vec::new(),
            contact_inconsistent_frames: Vec::new(),
            tracking: None,
            requires_tracking_evaluation: true,
        }
    }

    fn thresholds() -> AcceptanceThresholds {
        AcceptanceThresholds {
            max_normalized_endpoint_error: 0.05,
            max_planted_foot_slip: 0.02,
            max_root_drift: 0.1,
            allow_limit_violations: false,
            allow_torque_violations: false,
            allow_contact_inconsistency: false,
            max_tracking_rms: 0.05,
            require_tracking: false,
        }
    }

    #[test]
    fn within_threshold_is_accepted() {
        assert_eq!(
            report(&[0.01, 0.02]).evaluate(&thresholds()),
            Verdict::Accepted
        );
    }

    #[test]
    fn a_rejection_names_every_violating_frame_not_just_the_first() {
        let Verdict::Rejected(reasons) = report(&[0.1, 0.01, 0.2]).evaluate(&thresholds()) else {
            panic!("expected rejection");
        };
        let frames: Vec<_> = reasons.iter().filter_map(RejectionReason::frame).collect();
        assert_eq!(frames, vec![0, 2]);
        assert!(reasons.iter().all(|r| r.code() == "endpoint_error"));
    }

    #[test]
    fn tracking_is_judged_per_skill_and_its_absence_can_be_refused() {
        let mut r = report(&[0.0]);
        let strict = AcceptanceThresholds {
            require_tracking: true,
            ..thresholds()
        };
        assert_eq!(
            r.evaluate(&strict),
            Verdict::Rejected(vec![RejectionReason::TrackingMissing])
        );
        r.tracking = Some(TrackingResult {
            joint_rms: vec![0.0, 0.2],
            base_error: vec![0.0, 0.0],
            fell_at: None,
            per_label: vec![
                SkillTracking {
                    label: "jab".into(),
                    start: 0,
                    end: 1,
                    joint_rms: 0.01,
                    max_base_error: 0.0,
                    fell: false,
                },
                SkillTracking {
                    label: "hook".into(),
                    start: 1,
                    end: 2,
                    joint_rms: 0.2,
                    max_base_error: 0.0,
                    fell: false,
                },
            ],
            steps: 2,
            controller: "pd",
        });
        let Verdict::Rejected(reasons) = r.evaluate(&strict) else {
            panic!()
        };
        assert_eq!(reasons.len(), 1);
        assert!(
            matches!(&reasons[0], RejectionReason::TrackingFailed { label, .. } if label == "hook")
        );
    }

    #[test]
    fn limit_violations_are_checked_against_the_declared_range() {
        let joint = JointSpec {
            id: JointId::new(0),
            name: Name::new("j").unwrap(),
            kind: JointKind::Revolute,
            parent: LinkId::new(0),
            child: LinkId::new(1),
            origin: RigidPose::IDENTITY,
            axis: Vec3::unit_z(),
            limits: JointLimits::bounded(
                Range::<Radians>::new(Radians(-1.0), Radians(1.0)).unwrap(),
                RadiansPerSecond(5.0),
                NewtonMetres(10.0),
            ),
            sign: 1.0,
            zero_offset: 0.0,
        };
        let ok = check_position_limits(core::slice::from_ref(&joint), &[0.5]);
        assert!(ok.is_empty());
        let bad = check_position_limits(&[joint], &[1.5]);
        assert_eq!(bad.position_limit_excess.len(), 1);
        assert!((bad.position_limit_excess[0].1.value() - 0.5).abs() < 1e-12);
    }
}

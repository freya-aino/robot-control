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
use crate::cop::CopFrame;
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
    /// Centre-of-pressure support per frame (`crate::cop`): the combined
    /// CoP against the support polygon derived from the planted feet's
    /// declared collision geometry.
    pub cop: Vec<CopFrame>,
    /// Simulated tracking, when it was run.
    pub tracking: Option<TrackingResult>,
    /// Whether the report still lacks a tracking evaluation.
    pub requires_tracking_evaluation: bool,
}

/// What the caller accepts. Every field is the caller's choice.
///
/// No longer `Copy`: the optional per-joint motion limits carry vectors.
#[derive(Clone, PartialEq, Debug)]
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
    /// Smallest accepted centre-of-pressure margin, metres (negative values
    /// tolerate a CoP that far outside the support polygon; zero demands it
    /// stay inside). Unloaded frames are never judged by this.
    pub min_cop_margin: f64,
    /// Accept frames whose loaded feet declare no projectable collision
    /// geometry. `false` rejects them as `cop_unchecked`, so a missing
    /// support polygon can never silently pass.
    pub allow_unchecked_cop: bool,
    /// Per-joint motion bounds, from
    /// [`crate::dynamics::derived_motion_limits`] — the profile's own
    /// declarations, never invented numbers. `None` keeps acceleration and
    /// jerk report-only (the pre-gate behaviour); an `INFINITY` slot inside
    /// `Some` means that joint declares no basis for a bound and never
    /// rejects. A width mismatch against the report rejects as
    /// `motion_limits_width`, because judging joint `j` by joint `k`'s
    /// bound would be worse than not judging at all.
    pub motion_limits: Option<crate::dynamics::DerivedMotionLimits>,
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
    /// The combined centre of pressure left the support polygon by more
    /// than the threshold tolerates.
    CopOutside {
        /// Frame.
        frame: usize,
        /// The margin, metres (negative is outside).
        value: f64,
        /// Threshold.
        threshold: f64,
    },
    /// A loaded foot declared no projectable collision geometry, so the
    /// support polygon is unknown.
    CopUnchecked {
        /// Frame.
        frame: usize,
    },
    /// A joint's peak |acceleration| exceeds its derived bound (effort
    /// limit over the joint's effective inertia at the reference posture).
    AccelerationExcess {
        /// Joint index, in actuated order.
        joint: usize,
        /// Peak |acceleration|, rad/s^2.
        value: f64,
        /// The derived bound.
        threshold: f64,
    },
    /// A joint's RMS jerk exceeds its derived bound (acceleration bound
    /// over the actuator's declared command latency).
    JerkExcess {
        /// Joint index, in actuated order.
        joint: usize,
        /// RMS jerk, rad/s^3.
        value: f64,
        /// The derived bound.
        threshold: f64,
    },
    /// The supplied motion limits do not match the report's joint count,
    /// so no per-joint judgement would be about the joint it names.
    MotionLimitsWidth {
        /// Joints in the report.
        expected: usize,
        /// Bound slots supplied.
        found: usize,
    },
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
            Self::CopOutside { .. } => "cop_outside",
            Self::CopUnchecked { .. } => "cop_unchecked",
            Self::AccelerationExcess { .. } => "acceleration_excess",
            Self::JerkExcess { .. } => "jerk_excess",
            Self::MotionLimitsWidth { .. } => "motion_limits_width",
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
            | Self::Fell { frame }
            | Self::CopOutside { frame, .. }
            | Self::CopUnchecked { frame } => Some(*frame),
            Self::TrackingFailed { .. }
            | Self::TrackingMissing
            | Self::AccelerationExcess { .. }
            | Self::JerkExcess { .. }
            | Self::MotionLimitsWidth { .. } => None,
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
            Self::CopOutside {
                frame,
                value,
                threshold,
            } => write!(
                f,
                "frame {frame}: centre-of-pressure margin {value} m is below threshold {threshold} m"
            ),
            Self::CopUnchecked { frame } => write!(
                f,
                "frame {frame}: a loaded foot has no projectable collision geometry, so the \
                 support polygon is unknown"
            ),
            Self::AccelerationExcess {
                joint,
                value,
                threshold,
            } => write!(
                f,
                "joint {joint}: peak acceleration {value} rad/s^2 exceeds the derived bound \
                 {threshold} (effort limit over effective inertia)"
            ),
            Self::JerkExcess {
                joint,
                value,
                threshold,
            } => write!(
                f,
                "joint {joint}: RMS jerk {value} rad/s^3 exceeds the derived bound {threshold} \
                 (acceleration bound over command latency)"
            ),
            Self::MotionLimitsWidth { expected, found } => write!(
                f,
                "motion limits carry {found} joint bounds where the report has {expected} joints"
            ),
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
        for (i, frame) in self.cop.iter().enumerate() {
            match frame {
                CopFrame::Unloaded => {}
                CopFrame::NotCheckable { .. } => {
                    if !thresholds.allow_unchecked_cop {
                        reasons.push(RejectionReason::CopUnchecked { frame: i });
                    }
                }
                CopFrame::Checked { margin, .. } => {
                    if *margin < thresholds.min_cop_margin {
                        reasons.push(RejectionReason::CopOutside {
                            frame: i,
                            value: *margin,
                            threshold: thresholds.min_cop_margin,
                        });
                    }
                }
            }
        }
        if let Some(limits) = &thresholds.motion_limits {
            let dof = self.summary.peak_acceleration.len();
            if limits.max_acceleration.len() != dof || limits.max_rms_jerk.len() != dof {
                reasons.push(RejectionReason::MotionLimitsWidth {
                    expected: dof,
                    found: limits.max_acceleration.len().min(limits.max_rms_jerk.len()),
                });
            } else {
                for (joint, (&peak, &bound)) in self
                    .summary
                    .peak_acceleration
                    .iter()
                    .zip(&limits.max_acceleration)
                    .enumerate()
                {
                    // An INFINITY bound means the profile declares no basis
                    // for one; it never rejects, and never silently invents.
                    if peak > bound {
                        reasons.push(RejectionReason::AccelerationExcess {
                            joint,
                            value: peak,
                            threshold: bound,
                        });
                    }
                }
                for (joint, (&rms, &bound)) in self
                    .summary
                    .rms_jerk
                    .iter()
                    .zip(&limits.max_rms_jerk)
                    .enumerate()
                {
                    if rms > bound {
                        reasons.push(RejectionReason::JerkExcess {
                            joint,
                            value: rms,
                            threshold: bound,
                        });
                    }
                }
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
            cop: Vec::new(),
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
            min_cop_margin: 0.0,
            allow_unchecked_cop: false,
            motion_limits: None,
        }
    }

    #[test]
    fn derived_motion_limits_gate_acceleration_and_jerk_by_joint() {
        use crate::dynamics::DerivedMotionLimits;
        let mut r = report(&[0.0]);
        r.summary.peak_acceleration = vec![10.0, 500.0, 3.0];
        r.summary.rms_jerk = vec![1.0, 1.0, 9e9];
        let mut t = thresholds();
        // Joint 0's INFINITY slots are the no-declared-basis case and must
        // never reject; joint 1 exceeds its acceleration bound; joint 2
        // exceeds its jerk bound.
        t.motion_limits = Some(DerivedMotionLimits {
            max_acceleration: vec![f64::INFINITY, 100.0, 50.0],
            max_rms_jerk: vec![f64::INFINITY, f64::INFINITY, 1e6],
        });
        let Verdict::Rejected(reasons) = r.evaluate(&t) else {
            panic!("expected rejection");
        };
        let codes: Vec<_> = reasons.iter().map(RejectionReason::code).collect();
        assert_eq!(codes, vec!["acceleration_excess", "jerk_excess"]);
        assert!(matches!(
            reasons[0],
            RejectionReason::AccelerationExcess { joint: 1, .. }
        ));
        assert!(matches!(
            reasons[1],
            RejectionReason::JerkExcess { joint: 2, .. }
        ));

        // Under the bounds: accepted.
        r.summary.peak_acceleration = vec![10.0, 99.0, 3.0];
        r.summary.rms_jerk = vec![1.0, 1.0, 1e5];
        assert_eq!(r.evaluate(&t), Verdict::Accepted);

        // A width mismatch rejects by name rather than judging joint j by
        // joint k's bound.
        t.motion_limits = Some(DerivedMotionLimits {
            max_acceleration: vec![1.0],
            max_rms_jerk: vec![1.0],
        });
        let Verdict::Rejected(reasons) = r.evaluate(&t) else {
            panic!("expected rejection");
        };
        assert_eq!(reasons[0].code(), "motion_limits_width");
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
    fn the_cop_margin_and_the_unchecked_case_reject_by_their_own_codes() {
        let mut r = report(&[0.0, 0.0, 0.0]);
        r.cop = vec![
            CopFrame::Unloaded,
            CopFrame::Checked {
                cop: (0.0, 0.0),
                margin: -0.02,
            },
            CopFrame::NotCheckable {
                sites: vec![robot_types::ids::ContactSiteId::new(0)],
            },
        ];
        let Verdict::Rejected(reasons) = r.evaluate(&thresholds()) else {
            panic!("expected rejection");
        };
        let codes: Vec<_> = reasons.iter().map(RejectionReason::code).collect();
        assert_eq!(codes, vec!["cop_outside", "cop_unchecked"]);
        assert_eq!(reasons[0].frame(), Some(1));
        assert_eq!(reasons[1].frame(), Some(2));
        // Tolerating both makes the same report acceptable.
        let lax = AcceptanceThresholds {
            min_cop_margin: -0.05,
            allow_unchecked_cop: true,
            ..thresholds()
        };
        assert_eq!(r.evaluate(&lax), Verdict::Accepted);
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

//! Retargeting validation reports, kept to separate concerns.
//!
//! SPECS.md Section 8.3: "A validation report must separate kinematic
//! error, dynamics violations, contact consistency and task success."
//! [`ValidationReport`] has one field per concern — never a single pass/fail
//! bit — and "No universal fidelity percentage is a release criterion"
//! (same section): [`AcceptanceThresholds`] is a value the caller supplies
//! per robot and task, not a constant this crate bakes in.

use core::fmt;
use robot_model::joint::JointSpec;
use robot_types::units::Radians;

/// Per-sample kinematic error for one retargeted frame.
#[derive(Clone, Copy, Debug)]
pub struct KinematicError {
    /// Endpoint position error, normalized by a body-size reference length
    /// (SPECS.md Section 8.3: "Measure endpoint error normalized by body
    /// size").
    pub normalized_endpoint_error: f64,
    /// Root drift from the reference trajectory's root position, in metres.
    pub root_drift: f64,
    /// Planted-foot slip distance, in metres, where a foot was expected
    /// planted; `None` when no contact constraint applied this frame.
    pub planted_foot_slip: Option<f64>,
}

/// Per-sample joint-limit and actuator-feasibility violations.
#[derive(Clone, Debug, Default)]
pub struct DynamicsViolation {
    /// Joints whose commanded position exceeded their range, with the
    /// excess in radians.
    pub position_limit_excess: Vec<(usize, Radians)>,
    /// Joints whose implied velocity (finite-differenced between frames)
    /// exceeded their rated limit, with the excess in rad/s.
    pub velocity_limit_excess: Vec<(usize, f64)>,
}

impl DynamicsViolation {
    /// Whether any violation was recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.position_limit_excess.is_empty() && self.velocity_limit_excess.is_empty()
    }
}

/// The full report for one retargeted motion.
#[derive(Clone, Debug)]
pub struct ValidationReport {
    /// Per-frame kinematic error.
    pub kinematic: Vec<KinematicError>,
    /// Per-frame dynamics violations.
    pub dynamics: Vec<DynamicsViolation>,
    /// Frames where a contact constraint was inconsistent with the
    /// resulting pose (e.g. a planted foot slipped beyond tolerance).
    pub contact_inconsistent_frames: Vec<usize>,
    /// Whether downstream tracking (a separate, later measurement this
    /// report does not itself perform) is required before acceptance.
    /// Recorded as a reminder, not a result: SPECS.md Section 8.3 measures
    /// "downstream tracking success per skill and embodiment" separately
    /// from the kinematic/dynamics checks this report covers.
    pub requires_tracking_evaluation: bool,
}

/// Per-robot, per-task acceptance thresholds — a versioned artifact under
/// SPECS.md Section 14, supplied by the caller rather than assumed.
#[derive(Clone, Copy, Debug)]
pub struct AcceptanceThresholds {
    /// Maximum acceptable normalized endpoint error.
    pub max_normalized_endpoint_error: f64,
    /// Maximum acceptable planted-foot slip, in metres.
    pub max_planted_foot_slip: f64,
    /// Whether any joint-limit violation at all is acceptable.
    pub allow_limit_violations: bool,
}

/// The accept/reject outcome, with the specific reason(s) for a rejection —
/// never a bare boolean, so a rejected motion's report says why.
#[derive(Clone, Debug)]
pub enum Verdict {
    /// Every threshold was met.
    Accepted,
    /// At least one threshold was not met.
    Rejected(Vec<RejectionReason>),
}

/// One reason a motion was rejected.
#[derive(Clone, Debug)]
pub enum RejectionReason {
    /// A frame's normalized endpoint error exceeded the threshold.
    EndpointError {
        /// The frame.
        frame: usize,
        /// The measured value.
        value: f64,
        /// The threshold it exceeded.
        threshold: f64,
    },
    /// A frame's planted-foot slip exceeded the threshold.
    FootSlip {
        /// The frame.
        frame: usize,
        /// The measured value.
        value: f64,
        /// The threshold it exceeded.
        threshold: f64,
    },
    /// A frame had a joint-limit violation and the thresholds forbid any.
    LimitViolation {
        /// The frame.
        frame: usize,
    },
}

impl fmt::Display for RejectionReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EndpointError {
                frame,
                value,
                threshold,
            } => {
                write!(
                    f,
                    "frame {frame}: endpoint error {value} exceeds threshold {threshold}"
                )
            }
            Self::FootSlip {
                frame,
                value,
                threshold,
            } => {
                write!(
                    f,
                    "frame {frame}: foot slip {value} exceeds threshold {threshold}"
                )
            }
            Self::LimitViolation { frame } => write!(f, "frame {frame}: joint-limit violation"),
        }
    }
}

impl ValidationReport {
    /// Checks every frame against `thresholds`, returning the verdict with
    /// every violating frame named — not just the first.
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
        }
        if !thresholds.allow_limit_violations {
            for (i, d) in self.dynamics.iter().enumerate() {
                if !d.is_empty() {
                    reasons.push(RejectionReason::LimitViolation { frame: i });
                }
            }
        }
        if reasons.is_empty() {
            Verdict::Accepted
        } else {
            Verdict::Rejected(reasons)
        }
    }
}

/// Checks one frame's joint values against `joints`' declared limits,
/// building the [`DynamicsViolation`] record.
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
            dynamics: vec![DynamicsViolation::default(); endpoint_errors.len()],
            contact_inconsistent_frames: Vec::new(),
            requires_tracking_evaluation: true,
        }
    }

    #[test]
    fn within_threshold_is_accepted() {
        let r = report(&[0.01, 0.02, 0.015]);
        let thresholds = AcceptanceThresholds {
            max_normalized_endpoint_error: 0.05,
            max_planted_foot_slip: 0.01,
            allow_limit_violations: false,
        };
        assert!(matches!(r.evaluate(&thresholds), Verdict::Accepted));
    }

    #[test]
    fn a_rejection_names_every_violating_frame_not_just_the_first() {
        let r = report(&[0.01, 0.2, 0.3]);
        let thresholds = AcceptanceThresholds {
            max_normalized_endpoint_error: 0.05,
            max_planted_foot_slip: 0.01,
            allow_limit_violations: false,
        };
        let Verdict::Rejected(reasons) = r.evaluate(&thresholds) else {
            panic!("expected rejection")
        };
        assert_eq!(reasons.len(), 2);
    }

    #[test]
    fn limit_violations_are_checked_against_the_declared_range() {
        use robot_model::joint::{JointKind, JointLimits};
        use robot_model::pose::RigidPose;
        use robot_types::frames::Vec3;
        use robot_types::ids::{JointId, LinkId, Name};
        use robot_types::units::{NewtonMetres, RadiansPerSecond, Range};

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
                NewtonMetres(5.0),
            ),
            sign: 1.0,
            zero_offset: 0.0,
        };
        let violation = check_position_limits(&[joint], &[1.5]);
        assert_eq!(violation.position_limit_excess.len(), 1);
        assert!((violation.position_limit_excess[0].1.value() - 0.5).abs() < 1e-12);
        assert!(!violation.is_empty());
    }
}

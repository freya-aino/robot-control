//! A single inspection frame: skeleton poses, contacts and tracking error
//! at one timestamp.
//!
//! SPECS.md §6 describes `robot-viewer` as "inspection, synchronized
//! playback," and PLAN.md's Phase 2 names the content: "source skeleton,
//! target skeleton, ground contacts, limits and tracking errors." This
//! module is exactly that content, as plain data — no rendering. SPECS.md
//! §11's shortlist for a viewer (Rerun, egui, `urdf-viz`) sits behind the
//! dependency-qualification gate of Appendix B.6, and nothing has passed
//! it (Appendix B.1 marks every survey candidate `unverified`), so this
//! crate stops at the data a renderer would consume, per
//! `docs/decisions/0003-std-only-default-closure.md`.

use robot_model::kinematics::LinkPoses;
use robot_types::ids::ContactSiteId;
use robot_types::time::Timestamp;

/// Whether a contact site is active at this frame, alongside the measured
/// normal force magnitude that drove the classification — so a consumer
/// can distinguish "just crossed the threshold" from "solidly loaded"
/// without re-deriving it from raw sensor data.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ContactSample {
    /// Which contact site.
    pub site: ContactSiteId,
    /// Whether this site is currently classified as in contact.
    pub active: bool,
    /// Measured normal force magnitude, newtons.
    pub normal_force_newtons: f64,
}

/// One frame of inspection data: a snapshot of source/target skeleton
/// poses, contact state and per-joint tracking error, all at a single
/// timestamp.
///
/// SPECS.md §5's sampling-vs-arrival distinction applies here exactly as
/// everywhere else: `timestamp` names when this frame's data was sampled
/// (or says it's an arrival time), not when the viewer happens to render
/// it.
#[derive(Clone, PartialEq, Debug)]
pub struct ViewerFrame {
    /// When this frame's data was sampled.
    pub timestamp: Timestamp,
    /// The motion-capture or reference skeleton, if this frame has one
    /// (e.g. during retargeting inspection; absent during pure rollout
    /// playback).
    pub source_skeleton: Option<LinkPoses>,
    /// The robot's own link poses at this frame.
    pub target_skeleton: LinkPoses,
    /// Ground-contact state at every monitored site.
    pub contacts: Vec<ContactSample>,
    /// Per-joint tracking error (target minus commanded, or target minus
    /// achieved, as the producer documents), in the joint's own unit.
    /// Indexed the same way as the owning profile's actuated joints.
    pub tracking_error: Vec<f64>,
}

impl ViewerFrame {
    /// The largest-magnitude tracking error in this frame, or `None` if
    /// there is no tracking-error data at all (as opposed to all-zero
    /// error, which is a real, reportable value).
    #[must_use]
    pub fn worst_tracking_error(&self) -> Option<f64> {
        self.tracking_error
            .iter()
            .copied()
            .map(f64::abs)
            .fold(None, |acc, v| Some(acc.map_or(v, |a: f64| a.max(v))))
    }

    /// Contact sites currently active.
    pub fn active_contacts(&self) -> impl Iterator<Item = &ContactSample> {
        self.contacts.iter().filter(|c| c.active)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_types::ids::ContactSiteId;
    use robot_types::time::ClockId;

    fn frame(tracking_error: Vec<f64>) -> ViewerFrame {
        ViewerFrame {
            timestamp: Timestamp::sampling(0, ClockId::Simulation),
            source_skeleton: None,
            target_skeleton: LinkPoses::from_poses(vec![]),
            contacts: vec![
                ContactSample {
                    site: ContactSiteId::new(0),
                    active: true,
                    normal_force_newtons: 12.0,
                },
                ContactSample {
                    site: ContactSiteId::new(1),
                    active: false,
                    normal_force_newtons: 0.0,
                },
            ],
            tracking_error,
        }
    }

    #[test]
    fn worst_tracking_error_is_the_largest_magnitude_not_the_largest_signed_value() {
        let f = frame(vec![-0.5, 0.2, -0.05]);
        assert_eq!(f.worst_tracking_error(), Some(0.5));
    }

    #[test]
    fn no_tracking_error_data_is_distinct_from_zero_error() {
        let f = frame(vec![]);
        assert_eq!(f.worst_tracking_error(), None);
    }

    #[test]
    fn active_contacts_excludes_inactive_sites() {
        let f = frame(vec![0.0]);
        let active: Vec<_> = f.active_contacts().collect();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].site, ContactSiteId::new(0));
    }
}

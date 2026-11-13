//! Morphology conditioning for cross-embodiment few-shot transfer (SPECS.md
//! Section 11.3, row 9), and the scope rule on what such a result shows.
//!
//! [`descriptor`] summarizes a robot profile as a fixed-width vector a
//! policy can be conditioned on: total mass, standing height and arm span
//! at the zero posture, joint count, and the mean and spread of joint
//! velocity and effort limits. It reads only the profile, never simulator
//! state, so it is a deployable observation.
//!
//! The row's limitation is that "a held-out-robot result is evidence about
//! the pretraining set, and a single target robot is not a transfer
//! claim". [`TransferEvidence::claim`] encodes that: one held-out robot
//! yields [`TransferClaim::PretrainingSetEvidence`], and only two or more
//! distinct held-out robots yield [`TransferClaim::Transfer`].

use robot_model::kinematics::forward_kinematics;
use robot_model::profile::RobotProfile;

/// Width of [`descriptor`]'s output.
pub const DESCRIPTOR_WIDTH: usize = 8;

/// A morphology descriptor of `profile`.
#[must_use]
pub fn descriptor(profile: &RobotProfile) -> [f64; DESCRIPTOR_WIDTH] {
    let mass: f64 = profile.links.iter().map(|l| l.mass.mass.value()).sum();
    let zero = vec![0.0; profile.dof()];
    let (height, span) = forward_kinematics(profile, &zero).map_or((0.0, 0.0), |poses| {
        let pts: Vec<_> = poses.as_slice().iter().map(|p| p.translation).collect();
        let ext = |f: fn(&robot_types::frames::Vec3) -> f64| {
            let (lo, hi) = pts
                .iter()
                .map(f)
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), v| {
                    (a.min(v), b.max(v))
                });
            if hi > lo {
                hi - lo
            } else {
                0.0
            }
        };
        (ext(|p| p.z), ext(|p| p.y))
    });
    let joints: Vec<_> = profile.actuated_joints().collect();
    let stats = |v: Vec<f64>| {
        let n = v.len().max(1) as f64;
        let mean = v.iter().sum::<f64>() / n;
        let std = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();
        (mean, std)
    };
    let (vm, vs) = stats(joints.iter().map(|j| j.limits.velocity.value()).collect());
    let (em, es) = stats(joints.iter().map(|j| j.limits.effort.value()).collect());
    [mass, height, span, joints.len() as f64, vm, vs, em, es]
}

/// The scope a cross-embodiment result supports.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TransferClaim {
    /// One held-out robot: evidence about the pretraining set only.
    PretrainingSetEvidence,
    /// Two or more distinct held-out robots.
    Transfer,
}

/// Held-out robots a cross-embodiment result was measured on.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct TransferEvidence {
    /// Profile names of held-out robots (none in the pretraining set).
    pub held_out: Vec<String>,
    /// Profile names of the pretraining set.
    pub pretraining: Vec<String>,
}

impl TransferEvidence {
    /// The claim this evidence supports, or `None` if a "held-out" robot
    /// was in the pretraining set or nothing was held out.
    #[must_use]
    pub fn claim(&self) -> Option<TransferClaim> {
        if self.held_out.iter().any(|r| self.pretraining.contains(r)) {
            return None;
        }
        let mut distinct = self.held_out.clone();
        distinct.sort();
        distinct.dedup();
        match distinct.len() {
            0 => None,
            1 => Some(TransferClaim::PretrainingSetEvidence),
            _ => Some(TransferClaim::Transfer),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_model::fixtures::unitree_g1_29dof;

    #[test]
    fn the_g1_descriptor_is_human_scale() {
        let d = descriptor(&unitree_g1_29dof().unwrap());
        assert!(d[0] > 20.0 && d[0] < 60.0, "mass {}", d[0]);
        assert!(d[1] > 0.8 && d[1] < 1.6, "height {}", d[1]);
        assert_eq!(d[3], 29.0);
    }

    #[test]
    fn one_held_out_robot_is_not_a_transfer_claim() {
        let mut e = TransferEvidence {
            held_out: vec!["g1".into()],
            pretraining: vec!["h1".into(), "op3".into()],
        };
        assert_eq!(e.claim(), Some(TransferClaim::PretrainingSetEvidence));
        e.held_out.push("g1".into());
        assert_eq!(e.claim(), Some(TransferClaim::PretrainingSetEvidence));
        e.held_out.push("atlas".into());
        assert_eq!(e.claim(), Some(TransferClaim::Transfer));
        e.held_out.push("h1".into());
        assert_eq!(e.claim(), None);
    }
}

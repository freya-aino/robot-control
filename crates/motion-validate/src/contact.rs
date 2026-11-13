//! Contact consistency, the third of SPECS.md Section 8.3's separate
//! report categories.
//!
//! A reference is contact-consistent at a frame when its labels, its
//! geometry and its dynamics agree:
//!
//! - **geometry**: a foot labelled in contact is within `height_tolerance`
//!   of the ground, and a foot not labelled in contact is not below it by
//!   more than that;
//! - **support**: with no planted foot, the motion needs no base force
//!   beyond `force_tolerance` (otherwise something unlabelled is holding
//!   the robot up, or pushing it);
//! - **unilaterality and friction**: each estimated contact force pushes
//!   (`f_n >= -force_tolerance`) and lies in the friction cone
//!   (`|f_t| <= mu f_n + force_tolerance`).
//!
//! The force checks read [`crate::dynamics::estimate`]'s least-norm
//! distribution. A violation there means *that* distribution needs a pull
//! or a slide; with more than one foot a different split might not, so a
//! two-foot friction or pull violation is reported as found, not as proof
//! of infeasibility.

use robot_model::kinematics::forward_kinematics;
use robot_model::profile::RobotProfile;
use robot_types::ids::ContactSiteId;
use robot_types::motion::{MotionRecord, RobotReference};

use crate::dynamics::{ContactSpec, FrameDynamics};
use crate::trajectory::Trajectory;

/// Tolerances of the consistency checks.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ContactTolerances {
    /// World height of the ground plane.
    pub ground_height: f64,
    /// Height of a contact link's origin above the ground when standing.
    pub sole_height: f64,
    /// Allowed deviation from that height, metres.
    pub height_tolerance: f64,
    /// Force tolerance, newtons.
    pub force_tolerance: f64,
    /// Friction coefficient.
    pub friction: f64,
    /// Minimum label confidence for a site to count as labelled in contact.
    pub contact_confidence: f64,
}

/// What was inconsistent at one frame.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct ContactFrame {
    /// Labelled in contact but above the ground: `(site, height)`.
    pub labelled_but_lifted: Vec<(ContactSiteId, f64)>,
    /// Below the ground without a contact label: `(site, height)`.
    pub unlabelled_penetration: Vec<(ContactSiteId, f64)>,
    /// Base force needed with no planted foot, newtons.
    pub unsupported_force: Option<f64>,
    /// Estimated contact pulling on the ground: `(site, normal force)`.
    pub pulling: Vec<(ContactSiteId, f64)>,
    /// Estimated contact outside the friction cone: `(site, |f_t| / f_n)`.
    pub sliding: Vec<(ContactSiteId, f64)>,
}

impl ContactFrame {
    /// Whether nothing was inconsistent.
    #[must_use]
    pub fn is_consistent(&self) -> bool {
        self.labelled_but_lifted.is_empty()
            && self.unlabelled_penetration.is_empty()
            && self.unsupported_force.is_none()
            && self.pulling.is_empty()
            && self.sliding.is_empty()
    }
}

/// Checks every frame.
#[must_use]
pub fn check(
    profile: &RobotProfile,
    reference: &MotionRecord<RobotReference>,
    trajectory: &Trajectory,
    dynamics: &[FrameDynamics],
    contacts: &[ContactSpec],
    tol: &ContactTolerances,
) -> Vec<ContactFrame> {
    (0..trajectory.len())
        .map(|f| {
            let mut out = ContactFrame::default();
            let base = trajectory.base[f];
            let poses = forward_kinematics(profile, &trajectory.q[f]).ok();
            for c in contacts {
                let labelled = reference.frames[f].contacts.iter().any(|l| {
                    l.site == c.site && l.in_contact && l.confidence >= tol.contact_confidence
                });
                let Some(height) = poses
                    .as_ref()
                    .and_then(|p| p.by_index(c.link))
                    .map(|p| base.compose(&p).translation.z - tol.ground_height - tol.sole_height)
                else {
                    continue;
                };
                if labelled && height > tol.height_tolerance {
                    out.labelled_but_lifted.push((c.site, height));
                }
                if !labelled && height < -tol.height_tolerance {
                    out.unlabelled_penetration.push((c.site, height));
                }
            }
            let d = &dynamics[f];
            if d.contact_wrenches.is_empty() && d.unsupported_force > tol.force_tolerance {
                out.unsupported_force = Some(d.unsupported_force);
            }
            for (site, _, force) in &d.contact_wrenches {
                if force.z < -tol.force_tolerance {
                    out.pulling.push((*site, force.z));
                } else {
                    let tangential = force.x.hypot(force.y);
                    if tangential > tol.friction * force.z.max(0.0) + tol.force_tolerance {
                        out.sliding
                            .push((*site, tangential / force.z.max(f64::MIN_POSITIVE)));
                    }
                }
            }
            out
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::estimate;
    use crate::test_support::{g1, standing_reference};

    fn tol() -> ContactTolerances {
        ContactTolerances {
            ground_height: 0.0,
            sole_height: 0.0,
            height_tolerance: 0.01,
            force_tolerance: 1.0,
            friction: 0.8,
            contact_confidence: 0.5,
        }
    }

    #[test]
    fn standing_is_consistent() {
        let profile = g1();
        let (r, feet) = standing_reference(&profile, 4, true);
        let traj = Trajectory::from_reference(&r, profile.dof()).unwrap();
        let d = estimate(&profile, &r, &traj, &feet, 0.5, 0.1).unwrap();
        assert!(check(&profile, &r, &traj, &d, &feet, &tol())
            .iter()
            .all(ContactFrame::is_consistent));
    }

    #[test]
    fn a_lifted_labelled_foot_and_an_unlabelled_hover_are_flagged() {
        let profile = g1();
        let (mut r, feet) = standing_reference(&profile, 4, true);
        for f in &mut r.frames {
            f.root.translation.z += 0.05;
        }
        let traj = Trajectory::from_reference(&r, profile.dof()).unwrap();
        let d = estimate(&profile, &r, &traj, &feet, 0.5, 0.1).unwrap();
        let c = check(&profile, &r, &traj, &d, &feet, &tol());
        assert_eq!(c[1].labelled_but_lifted.len(), 2);

        let (r, feet) = standing_reference(&profile, 4, false);
        let traj = Trajectory::from_reference(&r, profile.dof()).unwrap();
        let d = estimate(&profile, &r, &traj, &feet, 0.5, 0.1).unwrap();
        let c = check(&profile, &r, &traj, &d, &feet, &tol());
        assert!(c[1].unsupported_force.is_some());
    }

    #[test]
    fn a_base_accelerating_sideways_on_one_foot_slides() {
        let profile = g1();
        let (mut r, feet) = standing_reference(&profile, 7, true);
        // 20 m/s^2 sideways: more than 0.8 g of friction can give.
        for (i, f) in r.frames.iter_mut().enumerate() {
            let t = i as f64 * 0.02;
            f.root.translation.y += 10.0 * t * t;
            f.contacts.retain(|l| l.site == feet[0].site);
        }
        let traj = Trajectory::from_reference(&r, profile.dof()).unwrap();
        let d = estimate(&profile, &r, &traj, &feet, 0.5, 0.1).unwrap();
        let c = check(
            &profile,
            &r,
            &traj,
            &d,
            &feet,
            &ContactTolerances {
                height_tolerance: 1.0,
                ..tol()
            },
        );
        assert!(!c[3].sliding.is_empty(), "{:?}", c[3]);
    }
}

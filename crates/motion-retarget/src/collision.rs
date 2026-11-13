//! Self-collision penalties for retargeting, per SPECS.md Section 8.3's
//! "self-collision penalties".
//!
//! Each collision shape of each link is bounded by one capsule (a segment
//! swept by a sphere): a sphere by itself, a box by a capsule spanning its
//! longest axis with the cross-section's half-diagonal as radius, a cylinder
//! by a capsule spanning its axis, and a convex hull by a capsule along its
//! principal axis (from the covariance of its vertices) that contains every
//! vertex. Every capsule contains its shape, so a zero penalty means the
//! shapes are separated by at least the margin; the converse does not hold,
//! and that conservatism is the price of a closed-form segment distance.
//!
//! The penalty for a pair is `max(0, r_a + r_b + margin - d)` where `d` is
//! the distance between their segments. Pairs are excluded when the links
//! are the same, joined directly by a joint, or already within the margin
//! in the reference posture: a capsule bound around a torso box often
//! overlaps the upper arm's at the shoulder by construction, and a penalty
//! there would only fight the kinematics. The excluded pairs are reported,
//! not hidden. A link whose shape is an unreduced [`CollisionGeometry::Mesh`]
//! has no geometry to bound and is counted in `unbounded_shapes`.
//!
//! This is the retargeting penalty only. Collision in simulation is
//! `robot-sim`'s exact narrow phase, which this does not replace.

use robot_model::kinematics::{forward_kinematics, KinematicsError, LinkPoses};
use robot_model::link::CollisionGeometry;
use robot_model::pose::RigidPose;
use robot_model::profile::RobotProfile;
use robot_types::frames::Vec3;
use robot_types::math::Matrix;

/// A capsule fixed to a link.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Capsule {
    /// Index into `profile.links`.
    pub link: usize,
    /// Segment start, in the link frame.
    pub a: Vec3,
    /// Segment end, in the link frame.
    pub b: Vec3,
    /// Radius in metres.
    pub radius: f64,
}

/// A self-collision penalty model.
#[derive(Clone, PartialEq, Debug)]
pub struct SelfCollision {
    /// Capsules of every bounded shape.
    pub capsules: Vec<Capsule>,
    /// Capsule index pairs checked.
    pub pairs: Vec<(usize, usize)>,
    /// Capsule index pairs excluded because they overlapped in the
    /// reference posture.
    pub excluded_at_reference: Vec<(usize, usize)>,
    /// Shapes with no geometry to bound (unreduced meshes).
    pub unbounded_shapes: usize,
    /// Clearance below which a pair is penalized, metres.
    pub margin: f64,
    /// Weight of the penalty rows.
    pub weight: f64,
}

impl SelfCollision {
    /// Builds the model from `profile`'s collision shapes, excluding pairs
    /// overlapping at the joint values `reference`.
    pub fn from_profile(
        profile: &RobotProfile,
        reference: &[f64],
        margin: f64,
        weight: f64,
    ) -> Result<Self, KinematicsError> {
        let mut capsules = Vec::new();
        let mut unbounded = 0;
        for (li, link) in profile.links.iter().enumerate() {
            for shape in &link.collision {
                match bound(&shape.geometry) {
                    Some((a, b, radius)) => capsules.push(Capsule {
                        link: li,
                        a: shape.pose.apply(a),
                        b: shape.pose.apply(b),
                        radius,
                    }),
                    None => unbounded += 1,
                }
            }
        }
        let index_of = |id| profile.links.iter().position(|l| l.id == id);
        let adjacent: Vec<(usize, usize)> = profile
            .joints
            .iter()
            .filter_map(|j| Some((index_of(j.parent)?, index_of(j.child)?)))
            .collect();
        let poses = forward_kinematics(profile, reference)?;
        let mut model = Self {
            capsules,
            pairs: Vec::new(),
            excluded_at_reference: Vec::new(),
            unbounded_shapes: unbounded,
            margin,
            weight,
        };
        for i in 0..model.capsules.len() {
            for j in i + 1..model.capsules.len() {
                let (la, lb) = (model.capsules[i].link, model.capsules[j].link);
                if la == lb || adjacent.contains(&(la, lb)) || adjacent.contains(&(lb, la)) {
                    continue;
                }
                if model.pair_penetration(i, j, &RigidPose::IDENTITY, &poses) > 0.0 {
                    model.excluded_at_reference.push((i, j));
                } else {
                    model.pairs.push((i, j));
                }
            }
        }
        Ok(model)
    }

    fn world_segment(&self, c: usize, base: &RigidPose, poses: &LinkPoses) -> (Vec3, Vec3) {
        let cap = &self.capsules[c];
        let pose = base.compose(&poses.by_index(cap.link).expect("capsule link exists"));
        (pose.apply(cap.a), pose.apply(cap.b))
    }

    fn pair_penetration(&self, i: usize, j: usize, base: &RigidPose, poses: &LinkPoses) -> f64 {
        let (a0, a1) = self.world_segment(i, base, poses);
        let (b0, b1) = self.world_segment(j, base, poses);
        let d = segment_distance(a0, a1, b0, b1);
        (self.capsules[i].radius + self.capsules[j].radius + self.margin - d).max(0.0)
    }

    /// Penetration of the margin for every checked pair, in `pairs` order.
    #[must_use]
    pub fn penetrations(&self, base: &RigidPose, poses: &LinkPoses) -> Vec<f64> {
        self.pairs
            .iter()
            .map(|&(i, j)| self.pair_penetration(i, j, base, poses))
            .collect()
    }
}

/// A bounding capsule `(a, b, radius)` in the shape frame.
fn bound(geometry: &CollisionGeometry) -> Option<(Vec3, Vec3, f64)> {
    match geometry {
        CollisionGeometry::Sphere { radius } => Some((Vec3::ZERO, Vec3::ZERO, *radius)),
        CollisionGeometry::Box { half_extents: h } => {
            let e = h.to_xyz();
            let long = (0..3)
                .max_by(|&i, &j| e[i].total_cmp(&e[j]))
                .expect("three axes");
            let (o1, o2) = ((long + 1) % 3, (long + 2) % 3);
            // The cross-section's corners must fit inside the radius.
            let radius = e[o1].hypot(e[o2]);
            // The segment spans the whole long axis: shortening it by the
            // radius, as for a cylinder, would leave the corners outside.
            let mut axis = [0.0; 3];
            axis[long] = e[long];
            let half = Vec3::from_xyz(axis);
            Some((-half, half, radius))
        }
        CollisionGeometry::Cylinder { radius, length } => {
            // The full axis again: the rim of each end face lies `radius`
            // off the axis at the axis end, which a shortened segment's cap
            // would not reach.
            let half = Vec3::new(0.0, 0.0, 0.5 * length);
            Some((-half, half, *radius))
        }
        CollisionGeometry::ConvexHull { points, .. } => hull_capsule(points),
        CollisionGeometry::Mesh { .. } => None,
    }
}

fn hull_capsule(points: &[Vec3]) -> Option<(Vec3, Vec3, f64)> {
    if points.is_empty() {
        return None;
    }
    let n = points.len() as f64;
    let centre = points.iter().fold(Vec3::ZERO, |a, p| a + *p) * (1.0 / n);
    let mut cov = [0.0; 9];
    for p in points {
        let d = (*p - centre).to_xyz();
        for r in 0..3 {
            for c in 0..3 {
                cov[r * 3 + c] += d[r] * d[c] / n;
            }
        }
    }
    let m = Matrix::from_row_major(3, 3, &cov).expect("3x3");
    let axis = match m.symmetric_eigen() {
        Ok((values, vectors)) => {
            let k = (0..3)
                .max_by(|&i, &j| values[i].total_cmp(&values[j]))
                .expect("three");
            Vec3::new(vectors[(0, k)], vectors[(1, k)], vectors[(2, k)])
                .normalized(1e-12)
                .unwrap_or(Vec3::unit_z())
        }
        Err(_) => Vec3::unit_z(),
    };
    let (mut lo, mut hi, mut radius) = (f64::INFINITY, f64::NEG_INFINITY, 0.0f64);
    for p in points {
        let d = *p - centre;
        let t = d.dot(axis);
        lo = lo.min(t);
        hi = hi.max(t);
        radius = radius.max((d - axis * t).norm());
    }
    // Shrink the segment by the radius where it is long enough, then grow
    // the radius until every vertex is inside: the cap of a shortened
    // segment must still reach the extreme vertices.
    let (s0, s1) = if hi - lo > 2.0 * radius {
        (lo + radius, hi - radius)
    } else {
        let mid = 0.5 * (lo + hi);
        (mid, mid)
    };
    let (a, b) = (centre + axis * s0, centre + axis * s1);
    let radius = points
        .iter()
        .map(|p| point_segment_distance(*p, a, b))
        .fold(0.0, f64::max);
    Some((a, b, radius))
}

fn point_segment_distance(p: Vec3, a: Vec3, b: Vec3) -> f64 {
    let ab = b - a;
    let len2 = ab.norm_squared();
    let t = if len2 > 0.0 {
        ((p - a).dot(ab) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (p - (a + ab * t)).norm()
}

/// Distance between segments `p0p1` and `q0q1` (Ericson, Real-Time
/// Collision Detection, section 5.1.9).
#[must_use]
pub fn segment_distance(p0: Vec3, p1: Vec3, q0: Vec3, q1: Vec3) -> f64 {
    let d1 = p1 - p0;
    let d2 = q1 - q0;
    let r = p0 - q0;
    let a = d1.norm_squared();
    let e = d2.norm_squared();
    let f = d2.dot(r);
    let eps = 1e-14;
    let (s, t) = if a <= eps && e <= eps {
        (0.0, 0.0)
    } else if a <= eps {
        (0.0, (f / e).clamp(0.0, 1.0))
    } else {
        let c = d1.dot(r);
        if e <= eps {
            ((-c / a).clamp(0.0, 1.0), 0.0)
        } else {
            let b = d1.dot(d2);
            let denom = a * e - b * b;
            let mut s = if denom > eps {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let mut t = (b * s + f) / e;
            if t < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            }
            (s, t)
        }
    };
    ((p0 + d1 * s) - (q0 + d2 * t)).norm()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ik::{solve_tasks, Base, IkConfig, Task};
    use crate::test_support::two_link_arm;
    use robot_model::link::CollisionShape;

    #[test]
    fn segment_distance_matches_closed_form_cases() {
        let d = segment_distance(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 1.0, -1.0),
            Vec3::new(0.5, 1.0, 1.0),
        );
        assert!((d - 1.0).abs() < 1e-12);
        // Parallel, overlapping in projection.
        let d = segment_distance(
            Vec3::ZERO,
            Vec3::unit_x(),
            Vec3::new(0.5, 2.0, 0.0),
            Vec3::new(3.0, 2.0, 0.0),
        );
        assert!((d - 2.0).abs() < 1e-12);
        // Endpoint to endpoint.
        let d = segment_distance(
            Vec3::ZERO,
            Vec3::unit_x(),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
        );
        assert!((d - 1.0).abs() < 1e-12);
    }

    #[test]
    fn every_bound_contains_its_shape() {
        let check = |g: CollisionGeometry, corners: Vec<Vec3>| {
            let (a, b, r) = bound(&g).unwrap();
            for c in corners {
                assert!(
                    point_segment_distance(c, a, b) <= r + 1e-12,
                    "{g:?} misses {c:?}"
                );
            }
        };
        let h = Vec3::new(0.3, 0.05, 0.1);
        let corners: Vec<Vec3> = (0..8)
            .map(|i| {
                Vec3::new(
                    if i & 1 == 0 { -h.x } else { h.x },
                    if i & 2 == 0 { -h.y } else { h.y },
                    if i & 4 == 0 { -h.z } else { h.z },
                )
            })
            .collect();
        check(CollisionGeometry::Box { half_extents: h }, corners.clone());
        check(
            CollisionGeometry::ConvexHull {
                points: corners.clone(),
                source: "test".to_owned(),
            },
            corners,
        );
        let rim: Vec<Vec3> = (0..16)
            .flat_map(|k| {
                let a = k as f64 * core::f64::consts::PI / 8.0;
                [
                    Vec3::new(0.05 * a.cos(), 0.05 * a.sin(), 0.2),
                    Vec3::new(0.05 * a.cos(), 0.05 * a.sin(), -0.2),
                ]
            })
            .collect();
        check(
            CollisionGeometry::Cylinder {
                radius: 0.05,
                length: 0.4,
            },
            rim,
        );
    }

    /// The two-link arm folded back on itself: spheres at the base and the
    /// hand collide when the elbow closes past about 150 degrees.
    fn arm_with_spheres() -> RobotProfile {
        let mut p = two_link_arm();
        let sphere = |r| CollisionShape {
            geometry: CollisionGeometry::Sphere { radius: r },
            pose: RigidPose::IDENTITY,
        };
        p.links[0].collision.push(sphere(0.3));
        p.links[3].collision.push(sphere(0.3));
        p
    }

    #[test]
    fn non_adjacent_pairs_are_checked_and_overlap_is_measured() {
        let profile = arm_with_spheres();
        let model = SelfCollision::from_profile(&profile, &[0.0, 0.0], 0.0, 1.0).unwrap();
        assert_eq!(model.pairs, vec![(0, 1)]);
        let folded = forward_kinematics(&profile, &[0.0, 2.9]).unwrap();
        let p = model.penetrations(&RigidPose::IDENTITY, &folded)[0];
        // Hand at distance 2 sin(0.12/2)*... computed directly:
        let hand = folded.by_index(3).unwrap().translation;
        assert!((p - (0.6 - hand.norm())).abs() < 1e-12);
        assert!(p > 0.0);
    }

    #[test]
    fn the_penalty_keeps_the_hand_away_from_the_base() {
        let profile = arm_with_spheres();
        let model = SelfCollision::from_profile(&profile, &[0.0, 0.0], 0.05, 1e4).unwrap();
        // Ask the hand to sit at the base: the penalty must hold it off.
        let tasks = [Task::Position {
            link: 3,
            target: Vec3::new(0.1, 0.0, 0.0),
            weight: 1.0,
        }];
        let config = IkConfig {
            max_iterations: 200,
            ..IkConfig::default()
        };
        let free = solve_tasks(
            &profile,
            &[0.0, 2.0],
            Base::FIXED_AT_ORIGIN,
            &tasks,
            None,
            &config,
        )
        .unwrap();
        let guarded = solve_tasks(
            &profile,
            &[0.0, 2.0],
            Base::FIXED_AT_ORIGIN,
            &tasks,
            Some(&model),
            &config,
        )
        .unwrap();
        let reach = |q: &[f64]| {
            forward_kinematics(&profile, q)
                .unwrap()
                .by_index(3)
                .unwrap()
                .translation
                .norm()
        };
        assert!(reach(&free.joint_values) < 0.2);
        assert!(
            reach(&guarded.joint_values) > 0.6,
            "{}",
            reach(&guarded.joint_values)
        );
        assert!(guarded.max_penetration < 0.01);
    }

    #[test]
    fn pairs_overlapping_at_the_reference_are_excluded_and_reported() {
        let mut profile = arm_with_spheres();
        profile.links[3].collision[0].geometry = CollisionGeometry::Sphere { radius: 1.8 };
        let model = SelfCollision::from_profile(&profile, &[0.0, 0.0], 0.0, 1.0).unwrap();
        assert!(model.pairs.is_empty());
        assert_eq!(model.excluded_at_reference, vec![(0, 1)]);
    }
}

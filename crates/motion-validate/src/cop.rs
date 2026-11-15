//! Centre-of-pressure support, the check SPECS.md Section 8.2 item 8's
//! "torque estimates" and Section 8.3's contact consistency imply but the
//! profile long had no polygon for: the total ground reaction must act at a
//! point the feet can actually provide.
//!
//! **The polygon comes from the declared collision geometry.** The profile
//! schema has no foot-support field, and does not need one: a foot's
//! support region is where its collision shapes meet the ground, so each
//! planted foot contributes the ground-plane projection of its declared
//! shapes (a box its eight corners, a hull its vertices, a sphere its
//! centre, a cylinder its two axis endpoints — the latter two conservative
//! rather than inflated by radius, since a curved shape genuinely supports
//! at a patch around a point), at the frame's own pose. The support polygon
//! is the convex hull of every planted foot's points, which in double
//! support is exactly the classical support region.
//!
//! **The centre of pressure is distribution-independent.** The per-foot
//! wrench split of [`crate::dynamics::estimate`] is a least-norm estimate,
//! so a per-foot CoP would judge the estimate rather than the motion. The
//! *combined* CoP — the point on the ground where the summed contact wrench
//! has no tangential moment — is fixed by the base wrench the motion needs,
//! whatever the split, so that is what is checked. A frame whose combined
//! CoP leaves the support polygon needs a ground that can pull somewhere,
//! which no unilateral contact provides.
//!
//! A foot with no projectable collision geometry makes the frame
//! [`CopFrame::NotCheckable`], named per site — reported, never a silent
//! pass. A frame with no planted vertical load is [`CopFrame::Unloaded`]
//! (a flight phase has no CoP to check).

use robot_model::kinematics::forward_kinematics;
use robot_model::link::CollisionGeometry;
use robot_model::profile::RobotProfile;
use robot_types::frames::Vec3;
use robot_types::ids::ContactSiteId;

use crate::dynamics::{ContactSpec, FrameDynamics};
use crate::trajectory::Trajectory;

/// The centre-of-pressure verdict of one frame.
#[derive(Clone, PartialEq, Debug)]
pub enum CopFrame {
    /// No planted foot carried vertical load, so there is no centre of
    /// pressure to check (a flight phase, or an unlabelled frame).
    Unloaded,
    /// At least one loaded foot declares no projectable collision geometry,
    /// so the support polygon is unknown. The sites are named.
    NotCheckable {
        /// The feet without usable geometry.
        sites: Vec<ContactSiteId>,
    },
    /// The combined centre of pressure, with its margin to the support
    /// boundary.
    Checked {
        /// Ground-plane position of the centre of pressure, metres.
        cop: (f64, f64),
        /// Signed distance to the support polygon's boundary, metres:
        /// positive inside, negative outside, zero on the edge. A
        /// degenerate support (a single point or a line of points) has no
        /// interior, so its margin is never positive.
        margin: f64,
    },
}

/// Checks every frame's combined centre of pressure against the support
/// polygon of its planted feet.
///
/// `ground_height` is the plane the feet stand on and `force_tolerance`
/// (newtons) the vertical load below which a frame counts as
/// [`CopFrame::Unloaded`] rather than dividing by a near-zero force.
#[must_use]
pub fn check(
    profile: &RobotProfile,
    trajectory: &Trajectory,
    dynamics: &[FrameDynamics],
    contacts: &[ContactSpec],
    ground_height: f64,
    force_tolerance: f64,
) -> Vec<CopFrame> {
    (0..trajectory.len())
        .map(|f| {
            let d = &dynamics[f];
            if d.contact_wrenches.is_empty() {
                return CopFrame::Unloaded;
            }
            let base = trajectory.base[f];
            let Ok(poses) = forward_kinematics(profile, &trajectory.q[f]) else {
                return CopFrame::Unloaded;
            };
            let world = |link: usize| base.compose(&poses.by_index(link).expect("validated link"));

            // Total wrench about the world origin, and the support points of
            // every loaded foot.
            let mut force = Vec3::ZERO;
            let mut moment = Vec3::ZERO;
            let mut points: Vec<(f64, f64)> = Vec::new();
            let mut unusable: Vec<ContactSiteId> = Vec::new();
            for (site, m, fo) in &d.contact_wrenches {
                let Some(spec) = contacts.iter().find(|c| c.site == *site) else {
                    continue;
                };
                let foot = world(spec.link);
                force = force + *fo;
                moment = moment + *m + foot.translation.cross(*fo);
                let before = points.len();
                for shape in &profile.links[spec.link].collision {
                    let placed = foot.compose(&shape.pose);
                    support_points(&shape.geometry, &placed, &mut points);
                }
                if points.len() == before {
                    unusable.push(*site);
                }
            }
            if force.z <= force_tolerance {
                return CopFrame::Unloaded;
            }
            if !unusable.is_empty() {
                return CopFrame::NotCheckable { sites: unusable };
            }
            // The point on z = ground_height where the tangential moment of
            // the total wrench vanishes: with the wrench (m, f) about the
            // origin, m_s = m - s x f = 0 in x and y gives
            // s_x = (z0 f_x - m_y) / f_z, s_y = (m_x + z0 f_y) / f_z.
            let cop = (
                (ground_height * force.x - moment.y) / force.z,
                (moment.x + ground_height * force.y) / force.z,
            );
            let hull = convex_hull(&points);
            CopFrame::Checked {
                cop,
                margin: signed_margin(&hull, cop),
            }
        })
        .collect()
}

/// Appends the ground-plane projection of one placed collision shape.
/// `placed` is the shape frame in world. A `Mesh` reference projects
/// nothing: it is the variant an importer uses for geometry it did not
/// reduce, and inventing support from it would be a silent pass.
fn support_points(
    geometry: &CollisionGeometry,
    placed: &robot_model::pose::RigidPose,
    out: &mut Vec<(f64, f64)>,
) {
    match geometry {
        CollisionGeometry::Sphere { .. } => {
            out.push((placed.translation.x, placed.translation.y));
        }
        CollisionGeometry::Box { half_extents } => {
            for sx in [-1.0, 1.0] {
                for sy in [-1.0, 1.0] {
                    for sz in [-1.0, 1.0] {
                        let corner = placed.apply(Vec3::new(
                            sx * half_extents.x,
                            sy * half_extents.y,
                            sz * half_extents.z,
                        ));
                        out.push((corner.x, corner.y));
                    }
                }
            }
        }
        CollisionGeometry::Cylinder { length, .. } => {
            for sz in [-0.5, 0.5] {
                let end = placed.apply(Vec3::new(0.0, 0.0, sz * length));
                out.push((end.x, end.y));
            }
        }
        CollisionGeometry::ConvexHull { points, .. } => {
            for p in points {
                let w = placed.apply(*p);
                out.push((w.x, w.y));
            }
        }
        CollisionGeometry::Mesh { .. } => {}
    }
}

/// Andrew's monotone chain: the counter-clockwise convex hull of a point
/// set. Fewer than three distinct points come back as they are (a point or
/// a segment), which [`signed_margin`] treats as a support with no
/// interior.
fn convex_hull(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut sorted: Vec<(f64, f64)> = points.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    sorted.dedup();
    if sorted.len() < 3 {
        return sorted;
    }
    let cross = |o: (f64, f64), a: (f64, f64), b: (f64, f64)| -> f64 {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };
    let mut hull: Vec<(f64, f64)> = Vec::with_capacity(sorted.len() + 1);
    for pass in 0..2 {
        let start = hull.len();
        let iter: Box<dyn Iterator<Item = &(f64, f64)>> = if pass == 0 {
            Box::new(sorted.iter())
        } else {
            Box::new(sorted.iter().rev())
        };
        for &p in iter {
            while hull.len() >= start + 2
                && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0
            {
                hull.pop();
            }
            hull.push(p);
        }
        hull.pop();
    }
    if hull.len() < 3 {
        // All collinear: the sorted extremes describe the segment.
        return vec![sorted[0], sorted[sorted.len() - 1]];
    }
    hull
}

/// Signed distance from `p` to the boundary of a counter-clockwise convex
/// polygon: positive inside, negative outside. A degenerate polygon (a
/// point or segment) has no interior, so its margin is `-distance`.
fn signed_margin(hull: &[(f64, f64)], p: (f64, f64)) -> f64 {
    match hull.len() {
        0 => f64::NEG_INFINITY,
        1 => -dist(p, hull[0]),
        2 => -segment_distance(p, hull[0], hull[1]),
        _ => {
            let mut inside = true;
            let mut min_edge = f64::INFINITY;
            for i in 0..hull.len() {
                let a = hull[i];
                let b = hull[(i + 1) % hull.len()];
                let edge = (b.0 - a.0, b.1 - a.1);
                let len = (edge.0 * edge.0 + edge.1 * edge.1).sqrt();
                if len <= 0.0 {
                    continue;
                }
                // Inward normal of a counter-clockwise edge.
                let signed = ((p.0 - a.0) * -edge.1 + (p.1 - a.1) * edge.0) / len;
                if signed < 0.0 {
                    inside = false;
                }
                min_edge = min_edge.min(signed);
            }
            if inside {
                min_edge
            } else {
                -(0..hull.len())
                    .map(|i| segment_distance(p, hull[i], hull[(i + 1) % hull.len()]))
                    .fold(f64::INFINITY, f64::min)
            }
        }
    }
}

fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - b.0).hypot(a.1 - b.1)
}

fn segment_distance(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let ab = (b.0 - a.0, b.1 - a.1);
    let len2 = ab.0 * ab.0 + ab.1 * ab.1;
    if len2 <= 0.0 {
        return dist(p, a);
    }
    let t = (((p.0 - a.0) * ab.0 + (p.1 - a.1) * ab.1) / len2).clamp(0.0, 1.0);
    dist(p, (a.0 + t * ab.0, a.1 + t * ab.1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::estimate;
    use crate::test_support::{g1, standing_reference};
    use robot_model::actuator::{
        ActuatorSpec, PdGains, ThermalLimits, TorqueSpeedEnvelope, Transmission,
    };
    use robot_model::joint::{JointKind, JointLimits, JointSpec};
    use robot_model::link::{CollisionShape, LinkSpec, MassProperties};
    use robot_model::pose::RigidPose;
    use robot_model::profile::{FirmwareContract, PROFILE_SCHEMA_VERSION};
    use robot_types::frames::{FrameId, GravityConvention};
    use robot_types::ids::{ActuatorId, JointId, LinkId, Name};
    use robot_types::motion::ContactLabel;
    use robot_types::schema::ExtensionTable;
    use robot_types::time::Duration;
    use robot_types::units::{
        Amperes, Celsius, Kilograms, NewtonMetres, Radians, RadiansPerSecond, Range,
    };

    /// A 10 kg body over one foot whose box sole is centred at
    /// `box_center_x`: the static CoP is the body origin's ground
    /// projection, so the margin is computable by hand.
    fn mass_over_box_foot(box_center_x: f64) -> robot_model::profile::RobotProfile {
        let mut body_mass = MassProperties::MASSLESS;
        body_mass.mass = Kilograms(10.0);
        let mut foot_mass = MassProperties::MASSLESS;
        foot_mass.mass = Kilograms(0.001);
        let body = LinkSpec {
            id: LinkId::new(0),
            name: Name::new("body").unwrap(),
            mass: body_mass,
            collision: Vec::new(),
        };
        let foot = LinkSpec {
            id: LinkId::new(1),
            name: Name::new("foot").unwrap(),
            mass: foot_mass,
            collision: vec![CollisionShape {
                geometry: CollisionGeometry::Box {
                    half_extents: Vec3::new(0.05, 0.04, 0.01),
                },
                pose: RigidPose::from_translation(Vec3::new(box_center_x, 0.0, 0.01)),
            }],
        };
        robot_model::profile::RobotProfile {
            schema_version: PROFILE_SCHEMA_VERSION,
            name: Name::new("fixture-mass-over-foot").unwrap(),
            profile_version: "1.0".to_owned(),
            joints: vec![JointSpec {
                id: JointId::new(0),
                name: Name::new("ankle").unwrap(),
                kind: JointKind::Revolute,
                parent: LinkId::new(0),
                child: LinkId::new(1),
                origin: RigidPose::from_translation(Vec3::new(0.0, 0.0, -0.5)),
                axis: Vec3::unit_z(),
                limits: JointLimits::bounded(
                    Range::<Radians>::new(Radians(-1.0), Radians(1.0)).unwrap(),
                    RadiansPerSecond(10.0),
                    NewtonMetres(50.0),
                ),
                sign: 1.0,
                zero_offset: 0.0,
            }],
            links: vec![body, foot],
            actuators: vec![ActuatorSpec {
                id: ActuatorId::new(0),
                name: Name::new("a0").unwrap(),
                joint: JointId::new(0),
                transmission: Transmission::DIRECT,
                envelope: TorqueSpeedEnvelope {
                    stall_torque: NewtonMetres(50.0),
                    corner_speed: RadiansPerSecond(1.0),
                    no_load_speed: RadiansPerSecond(2.0),
                },
                gains: Some(PdGains { kp: 1.0, kd: 0.1 }),
                command_latency: Duration::from_millis(1),
                thermal: ThermalLimits {
                    limit: Celsius(80.0),
                    ambient: Celsius(25.0),
                    time_constant: Duration::from_secs(10),
                    current_at_limit: Amperes(5.0),
                },
                current_uncertainty: Amperes(0.01),
            }],
            sensors: Vec::new(),
            root_link: LinkId::new(0),
            gravity: GravityConvention::minus_z_standard(FrameId::WORLD),
            firmware: FirmwareContract {
                version: "1.0".to_owned(),
                command_expiration: Duration::from_millis(100),
                watchdog_timeout: Duration::from_millis(200),
                has_independent_estop: true,
                allowed_modes: Vec::new(),
            },
            extensions: ExtensionTable::new(),
        }
    }

    /// The fixture standing still: foot sole on the ground, body above it.
    fn standing_frames(
        profile: &robot_model::profile::RobotProfile,
    ) -> (
        robot_types::motion::MotionRecord<robot_types::motion::RobotReference>,
        Vec<ContactSpec>,
    ) {
        let contacts = vec![ContactSpec {
            site: robot_types::ids::ContactSiteId::new(0),
            link: 1,
        }];
        // The foot origin is 0.5 m below the body; its sole box reaches
        // z = 0 when the body stands at 0.5.
        let mut r = crate::test_support::reference_from(profile, 4, |_| {
            (
                Vec3::new(0.0, 0.0, 0.5),
                robot_types::frames::Quat::IDENTITY,
                vec![0.0],
            )
        });
        for f in &mut r.frames {
            f.contacts.push(ContactLabel {
                site: robot_types::ids::ContactSiteId::new(0),
                in_contact: true,
                confidence: 0.9,
            });
        }
        (r, contacts)
    }

    fn cop_of(
        profile: &robot_model::profile::RobotProfile,
        r: &robot_types::motion::MotionRecord<robot_types::motion::RobotReference>,
        contacts: &[ContactSpec],
    ) -> Vec<CopFrame> {
        let traj = Trajectory::from_reference(r, profile.dof()).unwrap();
        let d = estimate(profile, r, &traj, contacts, 0.5, 0.1).unwrap();
        check(profile, &traj, &d, contacts, 0.0, 1.0)
    }

    #[test]
    fn a_mass_over_its_own_sole_has_the_hand_computed_margin() {
        // Sole x in [-0.05, 0.05], y in [-0.04, 0.04]; the CoP is the body
        // origin's projection (0, 0), so the margin is the nearest edge:
        // 0.04 in y.
        let profile = mass_over_box_foot(0.0);
        let (r, contacts) = standing_frames(&profile);
        for frame in cop_of(&profile, &r, &contacts) {
            let CopFrame::Checked { cop, margin } = frame else {
                panic!("{frame:?}");
            };
            assert!(cop.0.abs() < 1e-9 && cop.1.abs() < 1e-9, "{cop:?}");
            assert!((margin - 0.04).abs() < 1e-9, "margin {margin}");
        }
    }

    #[test]
    fn a_lean_past_the_sole_edge_fails_with_the_distance_outside() {
        // The sole moved to x in [0.05, 0.15]: the CoP stays at the body's
        // (0, 0), which is now 0.05 m outside the nearest edge.
        let profile = mass_over_box_foot(0.1);
        let (r, contacts) = standing_frames(&profile);
        for frame in cop_of(&profile, &r, &contacts) {
            let CopFrame::Checked { margin, .. } = frame else {
                panic!("{frame:?}");
            };
            assert!((margin + 0.05).abs() < 1e-9, "margin {margin}");
        }
    }

    #[test]
    fn a_foot_without_projectable_geometry_is_not_checkable_by_name() {
        let mut profile = mass_over_box_foot(0.0);
        profile.links[1].collision = vec![CollisionShape {
            geometry: CollisionGeometry::Mesh {
                content_id: "sha256:unreduced".to_owned(),
                scale: Vec3::new(1.0, 1.0, 1.0),
            },
            pose: RigidPose::IDENTITY,
        }];
        let (r, contacts) = standing_frames(&profile);
        for frame in cop_of(&profile, &r, &contacts) {
            let CopFrame::NotCheckable { sites } = frame else {
                panic!("{frame:?}");
            };
            assert_eq!(sites, vec![robot_types::ids::ContactSiteId::new(0)]);
        }
    }

    #[test]
    fn an_unlabelled_flight_phase_has_no_cop_to_check() {
        let profile = mass_over_box_foot(0.0);
        let (mut r, contacts) = standing_frames(&profile);
        for f in &mut r.frames {
            f.contacts.clear();
        }
        for frame in cop_of(&profile, &r, &contacts) {
            assert_eq!(frame, CopFrame::Unloaded);
        }
    }

    #[test]
    fn the_standing_g1_keeps_its_cop_inside_its_sole_spheres() {
        // The URDF G1's feet are four sole-corner spheres each, so the
        // support polygon is real; a static stance keeps the combined CoP
        // inside it.
        let profile = g1();
        let (r, feet) = standing_reference(&profile, 4, true);
        let traj = Trajectory::from_reference(&r, profile.dof()).unwrap();
        let d = estimate(&profile, &r, &traj, &feet, 0.5, 0.1).unwrap();
        let frames = check(&profile, &traj, &d, &feet, 0.0, 1.0);
        assert_eq!(frames.len(), 4);
        for frame in frames {
            let CopFrame::Checked { margin, .. } = frame else {
                panic!("{frame:?}");
            };
            assert!(margin > 0.0, "standing CoP left the feet: {margin}");
        }
    }

    #[test]
    fn a_hard_sideways_acceleration_pushes_the_cop_out_of_the_foot() {
        // 20 m/s^2 sideways at ~0.79 m of height needs the CoP about
        // h*a/g = 1.6 m to the side, far outside any foot.
        let profile = g1();
        let (mut r, feet) = standing_reference(&profile, 7, true);
        for (i, f) in r.frames.iter_mut().enumerate() {
            let t = i as f64 * 0.02;
            f.root.translation.y += 10.0 * t * t;
        }
        let traj = Trajectory::from_reference(&r, profile.dof()).unwrap();
        let d = estimate(&profile, &r, &traj, &feet, 0.5, 0.1).unwrap();
        let frames = check(&profile, &traj, &d, &feet, 0.0, 1.0);
        let CopFrame::Checked { margin, .. } = &frames[3] else {
            panic!("{:?}", frames[3]);
        };
        assert!(*margin < -0.5, "margin {margin}");
    }
}

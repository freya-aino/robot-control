//! Ground-plane and contact-interval estimation with uncertainty, per
//! SPECS.md Section 8.2 item 5: "Estimate ground plane and contact
//! intervals with uncertainty."
//!
//! [`estimate_ground_plane`] fits a plane to the positions of foot channels
//! at the frames where those feet are nearly stationary, by least squares
//! in the record's up direction with iterative 2.5-sigma outlier rejection.
//! Its uncertainty is the residual standard deviation (how flat the support
//! points are) and the standard error of the plane's height (how well that
//! many points pin it down).
//!
//! [`estimate_contacts`] turns height above that plane and foot speed into
//! a per-frame contact probability, the product of two logistic terms whose
//! height scale is widened by the plane's own uncertainty, so a poorly
//! determined floor produces less confident contacts rather than equally
//! confident ones. A two-threshold hysteresis turns probability into
//! intervals; each interval carries its mean probability and the duration,
//! at each boundary, over which the probability is ambiguous (between 0.25
//! and 0.75), which is the boundary's uncertainty. Labels are written into
//! the record as [`ContactLabel`]s with that probability as confidence; a
//! frame whose foot position is unknown is labelled with confidence 0.5.
//!
//! The record's lengths must already be metres and its up direction known
//! (see [`crate::condition`]); no threshold here is a validated constant,
//! which is why every one is a field of [`ContactConfig`].

use core::fmt;
use robot_types::frames::Vec3;
use robot_types::ids::ContactSiteId;
use robot_types::math::Matrix;
use robot_types::motion::{ContactLabel, MotionKind, MotionRecord, ProvenanceStep};
use robot_types::time::Duration;

use crate::kinematics::{channel_trajectory, frame_times, velocities};
use crate::REVISION;

/// A fitted ground plane.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct GroundPlane {
    /// Unit normal, on the side of `up`.
    pub normal: Vec3,
    /// A point on the plane.
    pub point: Vec3,
    /// Standard deviation of the support points' residuals, metres.
    pub residual_std: f64,
    /// Standard error of the plane's height at the support centroid,
    /// metres.
    pub height_std: f64,
    /// Support points used in the final fit.
    pub support: usize,
    /// Whether the tilt was fitted, or the support points were too
    /// clustered to determine it and the plane was taken perpendicular to
    /// `up`.
    pub tilt_fitted: bool,
}

impl GroundPlane {
    /// Signed height of `p` above the plane.
    #[must_use]
    pub fn height(&self, p: Vec3) -> f64 {
        self.normal.dot(p - self.point)
    }
}

/// Why estimation failed.
#[derive(Clone, PartialEq, Debug)]
pub enum ContactError {
    /// `up` was zero or non-finite.
    InvalidUp,
    /// A channel index was out of range.
    UnknownChannel(usize),
    /// Fewer than three stationary foot samples were found.
    InsufficientSupport {
        /// Samples found.
        found: usize,
    },
    /// A threshold was non-positive or non-finite, or the hysteresis
    /// thresholds were out of order.
    InvalidConfig(&'static str),
}

impl fmt::Display for ContactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidUp => f.write_str("up direction must be finite and non-zero"),
            Self::UnknownChannel(c) => write!(f, "channel {c} does not exist"),
            Self::InsufficientSupport { found } => {
                write!(
                    f,
                    "need at least three stationary foot samples, found {found}"
                )
            }
            Self::InvalidConfig(field) => write!(f, "invalid contact config field {field}"),
        }
    }
}

impl core::error::Error for ContactError {}

/// Fits a ground plane to `foot_channels` where each moves slower than
/// `stationary_speed` (metres per second).
pub fn estimate_ground_plane<K: MotionKind>(
    record: &MotionRecord<K>,
    foot_channels: &[usize],
    up: Vec3,
    stationary_speed: f64,
) -> Result<GroundPlane, ContactError> {
    let up = up.normalized(1e-12).ok_or(ContactError::InvalidUp)?;
    if !stationary_speed.is_finite() || stationary_speed <= 0.0 {
        return Err(ContactError::InvalidConfig("stationary_speed"));
    }
    let times = frame_times(record);
    let mut points = Vec::new();
    for &c in foot_channels {
        if c >= record.channels.len() {
            return Err(ContactError::UnknownChannel(c));
        }
        let traj = channel_trajectory(record, c);
        let vel = velocities(&traj, &times);
        for (p, v) in traj.iter().zip(&vel) {
            if let (Some(p), Some(v)) = (p, v) {
                if v.norm() < stationary_speed {
                    points.push(*p);
                }
            }
        }
    }
    fit_plane(&points, up)
}

fn tangent_basis(up: Vec3) -> (Vec3, Vec3) {
    let seed = if up.x.abs() < 0.9 {
        Vec3::unit_x()
    } else {
        Vec3::unit_y()
    };
    let e1 = (seed - up * seed.dot(up))
        .normalized(1e-12)
        .expect("seed not parallel to up");
    (e1, up.cross(e1))
}

fn fit_plane(points: &[Vec3], up: Vec3) -> Result<GroundPlane, ContactError> {
    if points.len() < 3 {
        return Err(ContactError::InsufficientSupport {
            found: points.len(),
        });
    }
    let (e1, e2) = tangent_basis(up);
    let coords: Vec<[f64; 3]> = points
        .iter()
        .map(|p| [p.dot(e1), p.dot(e2), p.dot(up)])
        .collect();
    let mut keep: Vec<bool> = vec![true; coords.len()];
    let mut result = None;
    for _ in 0..5 {
        let used: Vec<&[f64; 3]> = coords
            .iter()
            .zip(&keep)
            .filter(|(_, k)| **k)
            .map(|(c, _)| c)
            .collect();
        if used.len() < 3 {
            break;
        }
        let fit = fit_once(&used);
        let changed = coords
            .iter()
            .zip(keep.iter_mut())
            .fold(false, |changed, (c, k)| {
                let r = c[2] - (fit.a * c[0] + fit.b * c[1] + fit.c);
                let inlier = r.abs() <= 2.5 * fit.residual_std.max(1e-9);
                let flip = *k != inlier;
                *k = inlier;
                changed || flip
            });
        result = Some((fit, used.len()));
        if !changed {
            break;
        }
    }
    let (fit, support) = result.ok_or(ContactError::InsufficientSupport { found: 0 })?;
    let normal = (up - e1 * fit.a - e2 * fit.b)
        .normalized(1e-12)
        .expect("a finite slope gives a non-zero normal");
    let point = e1 * fit.cx + e2 * fit.cy + up * (fit.a * fit.cx + fit.b * fit.cy + fit.c);
    Ok(GroundPlane {
        normal,
        point,
        residual_std: fit.residual_std,
        height_std: fit.residual_std / (support as f64).sqrt(),
        support,
        tilt_fitted: fit.tilt_fitted,
    })
}

struct Fit {
    a: f64,
    b: f64,
    c: f64,
    cx: f64,
    cy: f64,
    residual_std: f64,
    tilt_fitted: bool,
}

fn fit_once(points: &[&[f64; 3]]) -> Fit {
    let n = points.len() as f64;
    let cx = points.iter().map(|p| p[0]).sum::<f64>() / n;
    let cy = points.iter().map(|p| p[1]).sum::<f64>() / n;
    let cz = points.iter().map(|p| p[2]).sum::<f64>() / n;
    let (mut sxx, mut sxy, mut syy, mut sxz, mut syz) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for p in points {
        let (x, y, z) = (p[0] - cx, p[1] - cy, p[2] - cz);
        sxx += x * x;
        sxy += x * y;
        syy += y * y;
        sxz += x * z;
        syz += y * z;
    }
    // The tilt is only determined when the support points span an area,
    // not a point or a line: require the smaller principal spread to be
    // at least 5 cm (standard deviation).
    let trace = sxx + syy;
    let det = sxx * syy - sxy * sxy;
    let min_eig = 0.5 * (trace - (trace * trace - 4.0 * det).max(0.0).sqrt());
    let spread_ok = min_eig / n >= 0.05 * 0.05;
    let (a, b) = if spread_ok {
        let m = Matrix::from_row_major(2, 2, &[sxx, sxy, sxy, syy]).expect("2x2");
        match m.solve(&Matrix::column(&[sxz, syz])) {
            Ok(s) => (s.as_slice()[0], s.as_slice()[1]),
            Err(_) => (0.0, 0.0),
        }
    } else {
        (0.0, 0.0)
    };
    let tilt_fitted = spread_ok && (a, b) != (0.0, 0.0);
    let c = cz - a * cx - b * cy;
    let dof = if tilt_fitted { 3.0 } else { 1.0 };
    let ss: f64 = points
        .iter()
        .map(|p| (p[2] - (a * p[0] + b * p[1] + c)).powi(2))
        .sum();
    Fit {
        a,
        b,
        c,
        cx,
        cy,
        residual_std: (ss / (n - dof).max(1.0)).sqrt(),
        tilt_fitted,
    }
}

/// A foot channel to label, and the contact site it represents.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct FootSite {
    /// Channel whose global position tracks the foot.
    pub channel: usize,
    /// Site the labels are written for.
    pub site: ContactSiteId,
    /// Height of the channel's origin above the sole when standing, metres.
    pub sole_offset: f64,
}

/// Thresholds of the contact model. None of them is a validated constant.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ContactConfig {
    /// Sole height at which the height term is one half, metres.
    pub height_threshold: f64,
    /// Logistic scale of the height term, metres.
    pub height_scale: f64,
    /// Foot speed at which the speed term is one half, metres per second.
    pub speed_threshold: f64,
    /// Logistic scale of the speed term, metres per second.
    pub speed_scale: f64,
    /// Probability above which a contact starts.
    pub enter: f64,
    /// Probability below which a contact ends.
    pub exit: f64,
    /// Contacts shorter than this are discarded.
    pub min_duration: Duration,
}

impl ContactConfig {
    fn validate(&self) -> Result<(), ContactError> {
        let positive = [
            (self.height_threshold, "height_threshold"),
            (self.height_scale, "height_scale"),
            (self.speed_threshold, "speed_threshold"),
            (self.speed_scale, "speed_scale"),
        ];
        for (v, name) in positive {
            if !v.is_finite() || v <= 0.0 {
                return Err(ContactError::InvalidConfig(name));
            }
        }
        if !(0.0 < self.exit && self.exit < self.enter && self.enter < 1.0) {
            return Err(ContactError::InvalidConfig("enter/exit"));
        }
        if self.min_duration.nanos() < 0 {
            return Err(ContactError::InvalidConfig("min_duration"));
        }
        Ok(())
    }
}

/// One contact interval.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ContactInterval {
    /// Site in contact.
    pub site: ContactSiteId,
    /// First frame in contact.
    pub start: usize,
    /// One past the last frame in contact.
    pub end: usize,
    /// Mean contact probability over the interval.
    pub mean_probability: f64,
    /// Span around the start over which the probability is ambiguous.
    pub start_uncertainty: Duration,
    /// Span around the end over which the probability is ambiguous.
    pub end_uncertainty: Duration,
}

/// Contact estimates for every site.
#[derive(Clone, PartialEq, Debug)]
pub struct ContactEstimate {
    /// Per site (in the order given), per frame contact probability; `None`
    /// where the foot position was unknown.
    pub probability: Vec<Vec<Option<f64>>>,
    /// Intervals, ordered by site then start.
    pub intervals: Vec<ContactInterval>,
}

fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Estimates contact for each of `sites` against `plane` and writes the
/// labels into `record`.
pub fn estimate_contacts<K: MotionKind>(
    record: &mut MotionRecord<K>,
    plane: &GroundPlane,
    sites: &[FootSite],
    config: &ContactConfig,
) -> Result<ContactEstimate, ContactError> {
    config.validate()?;
    let times = frame_times(record);
    let n = record.frames.len();
    let height_scale =
        (config.height_scale.powi(2) + plane.residual_std.powi(2) + plane.height_std.powi(2))
            .sqrt();
    let mut estimate = ContactEstimate {
        probability: Vec::with_capacity(sites.len()),
        intervals: Vec::new(),
    };
    for site in sites {
        if site.channel >= record.channels.len() {
            return Err(ContactError::UnknownChannel(site.channel));
        }
        let traj = channel_trajectory(record, site.channel);
        let vel = velocities(&traj, &times);
        let prob: Vec<Option<f64>> = traj
            .iter()
            .zip(&vel)
            .map(|(p, v)| {
                let h = plane.height(p.as_ref().copied()?) - site.sole_offset;
                let ph = logistic((config.height_threshold - h) / height_scale);
                let pv = v.map_or(0.5, |v| {
                    logistic((config.speed_threshold - v.norm()) / config.speed_scale)
                });
                Some(ph * pv)
            })
            .collect();

        let mut state = vec![false; n];
        let mut on = false;
        for (f, p) in prob.iter().enumerate() {
            if let Some(p) = *p {
                on = if on {
                    p >= config.exit
                } else {
                    p > config.enter
                };
            }
            state[f] = on;
        }
        let mut f = 0;
        while f < n {
            if !state[f] {
                f += 1;
                continue;
            }
            let start = f;
            while f < n && state[f] {
                f += 1;
            }
            let end = f;
            let span = times[end - 1] - times[start];
            if span * 1e9 < config.min_duration.nanos() as f64 {
                state[start..end].iter_mut().for_each(|s| *s = false);
                continue;
            }
            let known: Vec<f64> = prob[start..end].iter().flatten().copied().collect();
            let mean_probability = if known.is_empty() {
                0.5
            } else {
                known.iter().sum::<f64>() / known.len() as f64
            };
            estimate.intervals.push(ContactInterval {
                site: site.site,
                start,
                end,
                mean_probability,
                start_uncertainty: ambiguous_span(&prob, &times, start),
                end_uncertainty: ambiguous_span(&prob, &times, end - 1),
            });
        }

        for (f, frame) in record.frames.iter_mut().enumerate() {
            frame.contacts.retain(|c| c.site != site.site);
            let p = prob[f].unwrap_or(0.5);
            frame.contacts.push(ContactLabel {
                site: site.site,
                in_contact: state[f],
                confidence: if state[f] { p } else { 1.0 - p },
            });
        }
        estimate.probability.push(prob);
    }
    record.record_step(
        ProvenanceStep::new("estimate_contacts", REVISION)
            .with_parameter("sites", sites.len().to_string())
            .with_parameter("intervals", estimate.intervals.len().to_string())
            .with_parameter("height_threshold_m", config.height_threshold.to_string())
            .with_parameter("speed_threshold_mps", config.speed_threshold.to_string())
            .with_parameter("plane_residual_std_m", plane.residual_std.to_string()),
    );
    Ok(estimate)
}

/// Duration of the contiguous run of ambiguous frames (or unknown ones)
/// containing or adjacent to `at`.
fn ambiguous_span(prob: &[Option<f64>], times: &[f64], at: usize) -> Duration {
    let ambiguous = |i: usize| prob[i].is_none_or(|p| (0.25..=0.75).contains(&p));
    let seed = [Some(at), at.checked_sub(1), Some(at + 1)]
        .into_iter()
        .flatten()
        .find(|&i| i < prob.len() && ambiguous(i));
    let Some(seed) = seed else {
        return Duration::ZERO;
    };
    let (mut lo, mut hi) = (seed, seed);
    while lo > 0 && ambiguous(lo - 1) {
        lo -= 1;
    }
    while hi + 1 < prob.len() && ambiguous(hi + 1) {
        hi += 1;
    }
    // A single ambiguous sample still spans one frame interval.
    let dt = if prob.len() > 1 {
        times[1] - times[0]
    } else {
        0.0
    };
    Duration::from_secs_f64(times[hi] - times[lo] + dt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{channel, frame, record};
    use robot_types::frames::Quat;
    use robot_types::motion::{JointValue, Validity};

    /// Two marker feet: the left plants for the first half and swings in
    /// the second, the right the opposite, on a floor at z = `floor`.
    fn walking(floor: f64, tilt: f64) -> MotionRecord<robot_types::motion::HumanSkeleton> {
        let channels = vec![
            channel("root", None, Vec3::ZERO),
            channel("left_foot", None, Vec3::ZERO),
            channel("right_foot", None, Vec3::ZERO),
        ];
        let rate = 100.0;
        let ground = |x: f64| floor + tilt * x;
        let foot = |t: f64, phase: f64, y: f64| {
            // Stance for half the 1 s cycle, a 10 cm swing arc for the other.
            let u = (t + phase).rem_euclid(1.0);
            let step = (t + phase).div_euclid(1.0);
            if u < 0.5 {
                let x = step * 0.6;
                Vec3::new(x, y, ground(x))
            } else {
                let s = (u - 0.5) / 0.5;
                let x = step * 0.6 + 0.6 * s;
                Vec3::new(x, y, ground(x) + 0.1 * (core::f64::consts::PI * s).sin())
            }
        };
        let frames = (0..300)
            .map(|i| {
                let t = i as f64 / rate;
                frame(
                    t,
                    Vec3::new(0.0, 0.0, 1.0),
                    Quat::IDENTITY,
                    vec![
                        JointValue::Rotation {
                            rotation: Quat::IDENTITY,
                            source_order: None,
                        },
                        JointValue::Position {
                            position: foot(t, 0.0, 0.1),
                        },
                        JointValue::Position {
                            position: foot(t, 0.5, -0.1),
                        },
                    ],
                )
            })
            .collect();
        record(channels, frames)
    }

    fn config() -> ContactConfig {
        ContactConfig {
            height_threshold: 0.03,
            height_scale: 0.005,
            speed_threshold: 0.3,
            speed_scale: 0.05,
            enter: 0.6,
            exit: 0.4,
            min_duration: Duration::from_millis(50),
        }
    }

    fn sites() -> [FootSite; 2] {
        [
            FootSite {
                channel: 1,
                site: ContactSiteId::new(0),
                sole_offset: 0.0,
            },
            FootSite {
                channel: 2,
                site: ContactSiteId::new(1),
                sole_offset: 0.0,
            },
        ]
    }

    #[test]
    fn a_flat_floor_is_recovered_with_small_residual() {
        let r = walking(0.02, 0.0);
        let plane = estimate_ground_plane(&r, &[1, 2], Vec3::unit_z(), 0.1).unwrap();
        assert!(plane.tilt_fitted);
        assert!(
            (plane.normal - Vec3::unit_z()).norm() < 1e-6,
            "{:?}",
            plane.normal
        );
        assert!((plane.height(Vec3::ZERO) + 0.02).abs() < 1e-6);
        assert!(plane.residual_std < 1e-6);
    }

    #[test]
    fn a_tilted_floor_recovers_its_slope() {
        let r = walking(0.0, 0.05);
        let plane = estimate_ground_plane(&r, &[1, 2], Vec3::unit_z(), 0.1).unwrap();
        let expected = Vec3::new(-0.05, 0.0, 1.0).normalized(1e-12).unwrap();
        assert!(
            (plane.normal - expected).norm() < 1e-6,
            "{:?}",
            plane.normal
        );
    }

    #[test]
    fn a_single_footprint_does_not_fake_a_tilt() {
        let mut r = walking(0.0, 0.0);
        r.frames.truncate(40); // only the first stance of each foot
        let plane = estimate_ground_plane(&r, &[1], Vec3::unit_z(), 0.1).unwrap();
        assert!(!plane.tilt_fitted);
        assert_eq!(plane.normal, Vec3::unit_z());
    }

    #[test]
    fn stance_phases_become_intervals_and_labels_carry_confidence() {
        let mut r = walking(0.0, 0.0);
        let plane = estimate_ground_plane(&r, &[1, 2], Vec3::unit_z(), 0.1).unwrap();
        let est = estimate_contacts(&mut r, &plane, &sites(), &config()).unwrap();
        let left: Vec<_> = est
            .intervals
            .iter()
            .filter(|i| i.site == ContactSiteId::new(0))
            .collect();
        // Left foot stance: [0, 0.5), [1, 1.5), [2, 2.5) seconds.
        assert_eq!(left.len(), 3, "{left:?}");
        for (k, i) in left.iter().enumerate() {
            let start = k * 100;
            assert!(i.start.abs_diff(start) <= 3, "{i:?}");
            assert!(i.end.abs_diff(start + 50) <= 3, "{i:?}");
            assert!(i.mean_probability > 0.8);
        }
        r.validate().unwrap();
        let mid_stance = &r.frames[25].contacts;
        let mid_swing = &r.frames[75].contacts;
        assert!(mid_stance
            .iter()
            .any(|c| c.site == ContactSiteId::new(0) && c.in_contact && c.confidence > 0.9));
        assert!(mid_swing
            .iter()
            .any(|c| c.site == ContactSiteId::new(0) && !c.in_contact && c.confidence > 0.9));
        assert_eq!(r.provenance.last().unwrap().operation, "estimate_contacts");
    }

    #[test]
    fn a_rough_floor_lowers_confidence_and_a_hidden_foot_is_half_confident() {
        let mut r = walking(0.0, 0.0);
        let sharp = estimate_ground_plane(&r, &[1, 2], Vec3::unit_z(), 0.1).unwrap();
        let rough = GroundPlane {
            residual_std: 0.03,
            ..sharp
        };
        r.frames[25].validity[1] = Validity::Missing;
        let mut r2 = r.clone();
        let a = estimate_contacts(&mut r, &sharp, &sites(), &config()).unwrap();
        let b = estimate_contacts(&mut r2, &rough, &sites(), &config()).unwrap();
        assert!(b.probability[0][10].unwrap() < a.probability[0][10].unwrap());
        assert!(a.probability[0][25].is_none());
        let label = r.frames[25]
            .contacts
            .iter()
            .find(|c| c.site == ContactSiteId::new(0))
            .unwrap();
        assert!(
            label.in_contact,
            "hysteresis carries the state across one unknown frame"
        );
        assert!((label.confidence - 0.5).abs() < 1e-12);
    }

    #[test]
    fn inverted_hysteresis_and_too_few_points_are_rejected() {
        let mut r = walking(0.0, 0.0);
        let plane = estimate_ground_plane(&r, &[1, 2], Vec3::unit_z(), 0.1).unwrap();
        let bad = ContactConfig {
            enter: 0.3,
            exit: 0.6,
            ..config()
        };
        assert!(matches!(
            estimate_contacts(&mut r, &plane, &sites(), &bad),
            Err(ContactError::InvalidConfig(_))
        ));
        assert!(matches!(
            estimate_ground_plane(&r, &[], Vec3::unit_z(), 0.1),
            Err(ContactError::InsufficientSupport { .. })
        ));
    }
}

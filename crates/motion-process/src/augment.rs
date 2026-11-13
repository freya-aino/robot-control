//! Augmentation, the last responsibility SPECS.md Section 6 assigns this
//! crate: mirroring, time scaling, seeded sensor-like noise, and a rigid
//! root transform.
//!
//! Every function returns a new record and leaves the source untouched,
//! and records a provenance step naming the augmentation and its
//! parameters (with the seed, for noise), so an augmented example is never
//! mistaken for a captured one.
//!
//! An augmentation can make a reference physically wrong: a mirrored
//! southpaw jab is plausible, a 2x time-scaled one may exceed every
//! actuator's speed. Section 8.3's feasibility checks still apply
//! downstream; nothing here claims the result is feasible.

use core::fmt;
use robot_types::frames::{Quat, Vec3};
use robot_types::motion::{JointValue, MotionKind, MotionRecord, ProvenanceStep, Validity};
use robot_types::rng::{Rng, SeedStream};
use robot_types::time::Timestamp;

use crate::REVISION;

/// Why an augmentation was refused.
#[derive(Clone, PartialEq, Debug)]
pub enum AugmentError {
    /// A parameter was non-positive or non-finite.
    InvalidParameter(&'static str),
    /// A mirror pair referred to a channel that does not exist.
    UnknownChannel(usize),
    /// The skeleton is not symmetric about the mirror plane for this pair,
    /// so swapping their values would distort it.
    Asymmetric {
        /// The pair.
        pair: (usize, usize),
        /// Offset mismatch after reflection, in record length units.
        mismatch: f64,
    },
    /// A channel appeared in more than one pair.
    DuplicatePair(usize),
}

impl fmt::Display for AugmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(p) => write!(f, "invalid augmentation parameter {p}"),
            Self::UnknownChannel(c) => write!(f, "channel {c} does not exist"),
            Self::Asymmetric { pair, mismatch } => write!(
                f,
                "channels {} and {} are not mirror images (offset mismatch {mismatch})",
                pair.0, pair.1
            ),
            Self::DuplicatePair(c) => write!(f, "channel {c} appears in two mirror pairs"),
        }
    }
}

impl core::error::Error for AugmentError {}

fn reflect(p: Vec3, n: Vec3) -> Vec3 {
    p - n * (2.0 * p.dot(n))
}

/// The rotation `M R M` for the reflection `M` with unit normal `n`: the
/// axial part of a quaternion is a pseudo-vector, so it maps to `-M v`.
fn reflect_rotation(q: Quat, n: Vec3) -> Quat {
    let v = Vec3::new(q.x, q.y, q.z);
    let r = reflect(v, n) * -1.0;
    Quat {
        w: q.w,
        x: r.x,
        y: r.y,
        z: r.z,
    }
}

/// Left/right channel pairs found by name: `Left`/`Right`, `left_`/`right_`
/// and `l_`/`r_` prefixes (case-insensitive), matched exactly on the rest.
#[must_use]
pub fn mirror_pairs_by_name<K: MotionKind>(record: &MotionRecord<K>) -> Vec<(usize, usize)> {
    const PREFIXES: [(&str, &str); 4] = [
        ("left_", "right_"),
        ("left", "right"),
        ("l_", "r_"),
        ("l", "r"),
    ];
    let names: Vec<String> = record
        .channels
        .iter()
        .map(|c| c.name.as_str().to_ascii_lowercase())
        .collect();
    let mut pairs = Vec::new();
    let mut used = vec![false; names.len()];
    for (prefix_l, prefix_r) in PREFIXES {
        for (i, name) in names.iter().enumerate() {
            let Some(rest) = name.strip_prefix(prefix_l) else {
                continue;
            };
            if used[i] || rest.is_empty() {
                continue;
            }
            let partner = format!("{prefix_r}{rest}");
            if let Some(j) = names.iter().position(|n| *n == partner) {
                if !used[j] {
                    used[i] = true;
                    used[j] = true;
                    pairs.push((i, j));
                }
            }
        }
    }
    pairs
}

/// Mirrors `record` across the plane through the origin with normal
/// `normal`, swapping the values of each channel pair.
///
/// The channel layout is kept, so each pair must be mirror images in the
/// rest pose: the reflected offset of one must equal the other's within
/// `tolerance`. Contact labels are dropped because their sites cannot be
/// swapped without a site map; re-estimate contacts on the result.
pub fn mirror<K: MotionKind>(
    record: &MotionRecord<K>,
    normal: Vec3,
    pairs: &[(usize, usize)],
    tolerance: f64,
) -> Result<MotionRecord<K>, AugmentError> {
    let n = normal
        .normalized(1e-12)
        .ok_or(AugmentError::InvalidParameter("normal"))?;
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(AugmentError::InvalidParameter("tolerance"));
    }
    let count = record.channels.len();
    let mut partner: Vec<usize> = (0..count).collect();
    for &(a, b) in pairs {
        for c in [a, b] {
            if c >= count {
                return Err(AugmentError::UnknownChannel(c));
            }
            if partner[c] != c {
                return Err(AugmentError::DuplicatePair(c));
            }
        }
        let mismatch = (reflect(record.channels[a].offset, n) - record.channels[b].offset).norm();
        if mismatch > tolerance {
            return Err(AugmentError::Asymmetric {
                pair: (a, b),
                mismatch,
            });
        }
        partner[a] = b;
        partner[b] = a;
    }
    let mut out = record.clone();
    for (c, channel) in out.channels.iter_mut().enumerate() {
        if partner[c] == c {
            channel.offset = reflect(channel.offset, n);
        }
    }
    for (frame, source) in out.frames.iter_mut().zip(&record.frames) {
        frame.root.translation = reflect(source.root.translation, n);
        frame.root.rotation = reflect_rotation(source.root.rotation, n);
        for (c, &from) in partner.iter().enumerate() {
            frame.validity[c] = source.validity[from];
            frame.values[c] = match &source.values[from] {
                JointValue::Rotation {
                    rotation,
                    source_order,
                } => JointValue::Rotation {
                    rotation: reflect_rotation(*rotation, n),
                    source_order: *source_order,
                },
                JointValue::Position { position } => JointValue::Position {
                    position: reflect(*position, n),
                },
                other => other.clone(),
            };
        }
        frame.contacts.clear();
    }
    out.record_step(
        ProvenanceStep::new("augment_mirror", REVISION)
            .with_parameter("normal", format!("{} {} {}", n.x, n.y, n.z))
            .with_parameter("pairs", pairs.len().to_string()),
    );
    Ok(out)
}

/// Plays `record` `factor` times faster (`factor > 1` shortens it), by
/// rescaling timestamps about the first frame. Values are unchanged, so
/// every velocity scales by `factor` and every acceleration by its square.
pub fn time_scale<K: MotionKind>(
    record: &MotionRecord<K>,
    factor: f64,
) -> Result<MotionRecord<K>, AugmentError> {
    if !factor.is_finite() || factor <= 0.0 {
        return Err(AugmentError::InvalidParameter("factor"));
    }
    let mut out = record.clone();
    let t0 = record.frames.first().map_or(0, |f| f.timestamp.nanos());
    for f in &mut out.frames {
        let t = t0 + ((f.timestamp.nanos() - t0) as f64 / factor).round() as i64;
        f.timestamp = Timestamp::sampling(t, f.timestamp.clock());
    }
    out.record_step(
        ProvenanceStep::new("augment_time_scale", REVISION)
            .with_parameter("factor", factor.to_string()),
    );
    Ok(out)
}

/// Adds independent zero-mean Gaussian noise to every present sample and
/// the root pose: `position_std` (record length units) to translations and
/// positions, and a rotation of `rotation_std` radians per axis (as a
/// rotation vector) to rotations. Noised samples keep their validity.
pub fn add_noise<K: MotionKind>(
    record: &MotionRecord<K>,
    seed: SeedStream,
    position_std: f64,
    rotation_std: f64,
) -> Result<MotionRecord<K>, AugmentError> {
    if !position_std.is_finite() || position_std < 0.0 {
        return Err(AugmentError::InvalidParameter("position_std"));
    }
    if !rotation_std.is_finite() || rotation_std < 0.0 {
        return Err(AugmentError::InvalidParameter("rotation_std"));
    }
    let mut rng = Rng::new(seed);
    let vec = |rng: &mut Rng, s: f64| {
        Vec3::new(rng.normal(0.0, s), rng.normal(0.0, s), rng.normal(0.0, s))
    };
    let mut out = record.clone();
    for f in &mut out.frames {
        f.root.translation = f.root.translation + vec(&mut rng, position_std);
        f.root.rotation = (Quat::from_rotation_vector(vec(&mut rng, rotation_std))
            * f.root.rotation)
            .normalized();
        for (value, validity) in f.values.iter_mut().zip(&f.validity) {
            if *validity == Validity::Missing {
                continue;
            }
            match value {
                JointValue::Rotation { rotation, .. } => {
                    *rotation = (*rotation
                        * Quat::from_rotation_vector(vec(&mut rng, rotation_std)))
                    .normalized();
                }
                JointValue::Position { position } => {
                    *position = *position + vec(&mut rng, position_std)
                }
                JointValue::Scalar { value, unit } => {
                    let s = if *unit == "rad" {
                        rotation_std
                    } else {
                        position_std
                    };
                    *value += rng.normal(0.0, s);
                }
            }
        }
    }
    out.record_step(
        ProvenanceStep::new("augment_noise", REVISION)
            .with_parameter("seed", seed.seed.to_string())
            .with_parameter("stream", seed.stream.to_string())
            .with_parameter("position_std", position_std.to_string())
            .with_parameter("rotation_std_rad", rotation_std.to_string()),
    );
    Ok(out)
}

/// Applies a rigid transform to the whole performance: the root pose and
/// every marker position are rotated by `rotation` about the origin and then
/// translated. Local joint rotations are unchanged, so the performance
/// itself is identical; only where it happens moves.
#[must_use]
pub fn transform_root<K: MotionKind>(
    record: &MotionRecord<K>,
    rotation: Quat,
    translation: Vec3,
) -> MotionRecord<K> {
    let mut out = record.clone();
    for f in &mut out.frames {
        f.root.translation = rotation.rotate(f.root.translation) + translation;
        f.root.rotation = (rotation * f.root.rotation).normalized();
        for (value, validity) in f.values.iter_mut().zip(&f.validity) {
            match value {
                JointValue::Position { position } if *validity != Validity::Missing => {
                    *position = rotation.rotate(*position) + translation;
                }
                // The root channel's value mirrors the root rotation.
                _ => {}
            }
        }
        if let Some(JointValue::Rotation { rotation: r, .. }) = f.values.first_mut() {
            if record.channels.first().is_some_and(|c| c.parent.is_none()) {
                *r = f.root.rotation;
            }
        }
    }
    out.record_step(
        ProvenanceStep::new("augment_transform_root", REVISION)
            .with_parameter("rotation_vector", {
                let v = rotation.to_rotation_vector();
                format!("{} {} {}", v.x, v.y, v.z)
            })
            .with_parameter(
                "translation",
                format!("{} {} {}", translation.x, translation.y, translation.z),
            ),
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kinematics::{frame_poses, frame_positions};
    use crate::test_support::{channel, frame, record, rotation};
    use robot_types::motion::HumanSkeleton;
    use robot_types::units::Radians;

    /// Hips with a symmetric pair of arms (shoulder, hand) along +-y.
    fn arms(left_angle: f64, right_angle: f64) -> MotionRecord<HumanSkeleton> {
        let channels = vec![
            channel("Hips", None, Vec3::ZERO),
            channel("LeftArm", Some(0), Vec3::new(0.0, 0.2, 0.5)),
            channel("LeftHand", Some(1), Vec3::new(0.0, 0.3, 0.0)),
            channel("RightArm", Some(0), Vec3::new(0.0, -0.2, 0.5)),
            channel("RightHand", Some(3), Vec3::new(0.0, -0.3, 0.0)),
        ];
        let root_rot = Quat::from_rz(Radians(0.3)).unwrap();
        let frames = (0..3)
            .map(|i| {
                let t = i as f64 * 0.01;
                frame(
                    t,
                    Vec3::new(0.1, 0.4, 1.0),
                    root_rot,
                    vec![
                        rotation(root_rot),
                        rotation(Quat::from_rx(Radians(left_angle)).unwrap()),
                        rotation(Quat::IDENTITY),
                        rotation(Quat::from_ry(Radians(right_angle)).unwrap()),
                        rotation(Quat::IDENTITY),
                    ],
                )
            })
            .collect();
        record(channels, frames)
    }

    #[test]
    fn pairs_are_found_by_prefix() {
        let r = arms(0.0, 0.0);
        assert_eq!(mirror_pairs_by_name(&r), vec![(1, 3), (2, 4)]);
    }

    #[test]
    fn mirroring_reflects_every_global_position_and_swaps_sides() {
        let r = arms(0.4, -0.9);
        let pairs = mirror_pairs_by_name(&r);
        let n = Vec3::unit_y();
        let m = mirror(&r, n, &pairs, 1e-9).unwrap();
        let before = frame_positions(&r, 1);
        let after = frame_positions(&m, 1);
        let partner = [0, 3, 4, 1, 2];
        for c in 0..5 {
            let expected = reflect(before[partner[c]].unwrap(), n);
            assert!((after[c].unwrap() - expected).norm() < 1e-12, "channel {c}");
        }
        m.validate().unwrap();
    }

    #[test]
    fn mirroring_twice_is_the_identity() {
        let r = arms(0.4, -0.9);
        let pairs = mirror_pairs_by_name(&r);
        let m = mirror(
            &mirror(&r, Vec3::unit_y(), &pairs, 1e-9).unwrap(),
            Vec3::unit_y(),
            &pairs,
            1e-9,
        )
        .unwrap();
        for (a, b) in frame_poses(&r, 2).iter().zip(frame_poses(&m, 2)) {
            let (a, b) = (a.unwrap(), b.unwrap());
            assert!((a.0 - b.0).norm() < 1e-12);
            assert!(a.1.angle_to(b.1).value() < 1e-6);
        }
    }

    #[test]
    fn an_asymmetric_pair_is_refused() {
        let mut r = arms(0.0, 0.0);
        r.channels[4].offset = Vec3::new(0.0, -0.35, 0.0);
        assert!(matches!(
            mirror(&r, Vec3::unit_y(), &[(2, 4)], 1e-3),
            Err(AugmentError::Asymmetric { pair: (2, 4), .. })
        ));
    }

    #[test]
    fn time_scaling_halves_the_duration_at_double_speed() {
        let r = arms(0.0, 0.0);
        let fast = time_scale(&r, 2.0).unwrap();
        assert_eq!(fast.duration().nanos() * 2, r.duration().nanos());
        assert_eq!(fast.frames[0].timestamp, r.frames[0].timestamp);
        assert!(time_scale(&r, 0.0).is_err());
    }

    #[test]
    fn noise_is_seeded_and_has_the_requested_scale() {
        let channels = vec![channel("m", None, Vec3::ZERO)];
        let frames = (0..2000)
            .map(|i| {
                frame(
                    i as f64 * 0.01,
                    Vec3::ZERO,
                    Quat::IDENTITY,
                    vec![JointValue::Position {
                        position: Vec3::ZERO,
                    }],
                )
            })
            .collect();
        let r = record(channels, frames);
        let seed = SeedStream::new(7, 3);
        let a = add_noise(&r, seed, 0.01, 0.0).unwrap();
        let b = add_noise(&r, seed, 0.01, 0.0).unwrap();
        assert_eq!(a, b);
        let xs: Vec<f64> = a
            .frames
            .iter()
            .map(|f| match f.values[0] {
                JointValue::Position { position } => position.x,
                _ => unreachable!(),
            })
            .collect();
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        let std = (xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / xs.len() as f64).sqrt();
        assert!(mean.abs() < 0.001, "{mean}");
        assert!((std - 0.01).abs() < 0.001, "{std}");
        assert_eq!(a.provenance.last().unwrap().operation, "augment_noise");
    }

    #[test]
    fn a_root_transform_moves_the_performance_rigidly() {
        let r = arms(0.4, -0.9);
        let q = Quat::from_rz(Radians(1.1)).unwrap();
        let t = Vec3::new(2.0, -1.0, 0.0);
        let moved = transform_root(&r, q, t);
        for (a, b) in frame_positions(&r, 1)
            .iter()
            .zip(frame_positions(&moved, 1))
        {
            assert!((q.rotate(a.unwrap()) + t - b.unwrap()).norm() < 1e-12);
        }
    }
}

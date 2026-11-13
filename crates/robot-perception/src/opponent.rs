//! Opponent tracking, per SPECS.md Section 11.2: "Opponent perception
//! supplies tracked pose/range/velocity with uncertainty and age; a
//! tactical policy must handle occlusion and missing detections."
//!
//! [`OpponentTracker`] is a constant-velocity Kalman filter over the
//! opponent's position and velocity. A detection is a position with its
//! own covariance from whatever detector produced it; it is gated, so a
//! false detection far from the track is rejected and counted rather than
//! dragging the estimate. Between detections the track coasts: its mean
//! moves at the estimated velocity and its covariance grows, so an
//! occluded opponent is reported as *less certain*, not as frozen. Every
//! estimate carries its age (time since the last admitted detection), and
//! past `max_age` the tracker reports nothing: a tactical policy reading it
//! must then use its opponent-agnostic behavior, as Section 12.5 requires
//! of any opponent input.

use robot_types::frames::Vec3;
use robot_types::math::Matrix;

use crate::linear_kf::{FilterError, KalmanFilter};

/// The current estimate, with its uncertainty and age.
#[derive(Clone, PartialEq, Debug)]
pub struct TrackEstimate {
    /// Position.
    pub position: Vec3,
    /// Velocity.
    pub velocity: Vec3,
    /// Position standard deviation (largest axis), metres.
    pub position_sigma: f64,
    /// Seconds since the last admitted detection.
    pub age: f64,
}

/// Counters.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct TrackerStats {
    /// Detections admitted.
    pub admitted: u64,
    /// Detections rejected by the gate.
    pub rejected: u64,
    /// Tracks started.
    pub started: u64,
}

/// A single-opponent tracker.
pub struct OpponentTracker {
    kf: Option<KalmanFilter>,
    time: f64,
    last_detection: f64,
    q_acc: f64,
    max_age: f64,
    gate: f64,
    /// Counters.
    pub stats: TrackerStats,
}

impl OpponentTracker {
    /// A tracker with acceleration noise `q_acc` (m/s²) that drops its
    /// track after `max_age` seconds without a detection.
    #[must_use]
    pub fn new(q_acc: f64, max_age: f64) -> Self {
        Self {
            kf: None,
            time: 0.0,
            last_detection: 0.0,
            q_acc,
            max_age,
            gate: crate::base::CHI2_3_999,
            stats: TrackerStats::default(),
        }
    }

    /// Coasts the track to time `t` (seconds).
    pub fn predict_to(&mut self, t: f64) -> Result<(), FilterError> {
        let dt = t - self.time;
        self.time = t;
        let Some(kf) = &mut self.kf else {
            return Ok(());
        };
        if dt <= 0.0 {
            return Ok(());
        }
        let mut f = Matrix::identity(6);
        let mut q = Matrix::zeros(6, 6);
        let s2 = self.q_acc * self.q_acc;
        for i in 0..3 {
            f[(i, i + 3)] = dt;
            q[(i, i)] = s2 * dt.powi(4) / 4.0;
            q[(i, i + 3)] = s2 * dt.powi(3) / 2.0;
            q[(i + 3, i)] = s2 * dt.powi(3) / 2.0;
            q[(i + 3, i + 3)] = s2 * dt * dt;
        }
        kf.predict(&f, &q)
    }

    /// Feeds a detection at time `t` with position standard deviation
    /// `sigma`. Starts a track if none exists (or the old one aged out);
    /// returns whether the detection was admitted.
    pub fn detect(&mut self, t: f64, position: Vec3, sigma: f64) -> Result<bool, FilterError> {
        self.predict_to(t)?;
        let aged_out = t - self.last_detection > self.max_age;
        if self.kf.is_none() || aged_out {
            let x0 = Matrix::column(&[position.x, position.y, position.z, 0.0, 0.0, 0.0]);
            let mut p0 = Matrix::zeros(6, 6);
            for i in 0..3 {
                p0[(i, i)] = sigma * sigma;
                p0[(i + 3, i + 3)] = 4.0; // up to ~2 m/s unknown
            }
            self.kf = Some(KalmanFilter::new(x0, p0)?);
            self.last_detection = t;
            self.stats.started += 1;
            self.stats.admitted += 1;
            return Ok(true);
        }
        let kf = self.kf.as_mut().expect("present");
        let mut h = Matrix::zeros(3, 6);
        for i in 0..3 {
            h[(i, i)] = 1.0;
        }
        let r = Matrix::diagonal(&[sigma * sigma; 3]);
        let ok = kf.gated_update(&h, &r, &Matrix::column(&position.to_xyz()), self.gate)?;
        if ok {
            self.last_detection = t;
            self.stats.admitted += 1;
        } else {
            self.stats.rejected += 1;
        }
        Ok(ok)
    }

    /// The estimate at the current time, or `None` when there is no live
    /// track.
    #[must_use]
    pub fn estimate(&self) -> Option<TrackEstimate> {
        let kf = self.kf.as_ref()?;
        let age = self.time - self.last_detection;
        if age > self.max_age {
            return None;
        }
        let m = kf.mean();
        let p = kf.covariance();
        let sigma = (0..3).map(|i| p[(i, i)]).fold(0.0, f64::max).sqrt();
        Some(TrackEstimate {
            position: Vec3::new(m[(0, 0)], m[(1, 0)], m[(2, 0)]),
            velocity: Vec3::new(m[(3, 0)], m[(4, 0)], m[(5, 0)]),
            position_sigma: sigma,
            age,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_moving_opponent_is_tracked_and_coasts_through_occlusion() {
        let mut t = OpponentTracker::new(1.0, 0.5);
        let truth = |s: f64| Vec3::new(1.5 - 0.5 * s, 0.2 * s, 1.2);
        for k in 0..40 {
            let s = k as f64 * 0.02;
            t.detect(s, truth(s), 0.02).unwrap();
        }
        let e = t.estimate().unwrap();
        assert!(
            (e.velocity - Vec3::new(-0.5, 0.2, 0.0)).norm() < 0.1,
            "{e:?}"
        );
        let sigma_seen = e.position_sigma;
        // Occluded for 0.3 s: the track coasts forward and grows uncertain.
        t.predict_to(0.78 + 0.3).unwrap();
        let e = t.estimate().unwrap();
        assert!((e.position - truth(1.08)).norm() < 0.05, "{e:?}");
        assert!(e.position_sigma > sigma_seen);
        assert!((e.age - 0.3).abs() < 1e-9);
        // Past the maximum age: no estimate.
        t.predict_to(1.5).unwrap();
        assert!(t.estimate().is_none());
    }

    #[test]
    fn a_false_detection_is_rejected_and_a_lost_track_restarts() {
        let mut t = OpponentTracker::new(1.0, 0.5);
        for k in 0..20 {
            t.detect(k as f64 * 0.02, Vec3::new(1.0, 0.0, 1.2), 0.02)
                .unwrap();
        }
        assert!(!t.detect(0.4, Vec3::new(3.0, 2.0, 1.2), 0.02).unwrap());
        assert_eq!(t.stats.rejected, 1);
        assert!(t.detect(2.0, Vec3::new(3.0, 2.0, 1.2), 0.02).unwrap());
        assert_eq!(t.stats.started, 2);
    }
}

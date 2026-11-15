//! Low-pass filtering with a documented phase delay, per SPECS.md Section
//! 8.2 item 4: "Filter and resample with documented phase delay; preserve
//! real strike-speed characteristics."
//!
//! The filter is a second-order Butterworth section designed by the
//! bilinear transform with frequency prewarping, in one of two modes:
//!
//! - [`FilterMode::ZeroPhase`]: run forward then backward over the whole
//!   record (offline use). The phase response cancels exactly, so the
//!   recorded delay is zero. The two passes square the magnitude response,
//!   so the design cutoff is raised by Winter's correction factor
//!   `(2^(1/2) - 1)^(-1/4)` for two passes of a second-order section: the
//!   combined response is then -3 dB at the requested cutoff, not -6 dB.
//! - [`FilterMode::Causal`]: one forward pass (what a live pipeline can do).
//!   The recorded delay is the filter's exact group delay at DC, computed
//!   from its coefficients as `sum(k b_k)/sum(b_k) - sum(k a_k)/sum(a_k)`
//!   samples, not a nominal figure.
//!
//! Every pass starts from the steady state of its first sample, so a
//! record that begins at rest does not ring. A zero-phase pass also pads
//! each end by odd reflection, which keeps the end value and slope.
//!
//! Rotations are filtered **on the rotation group**: each sample is mapped
//! to a rotation vector relative to the sequence's first sample
//! (`log(q0^-1 q_i)`), unwrapped onto the `2 pi k` branch continuous with
//! its predecessor so an accumulated rotation past `pi` does not fold back,
//! filtered component-wise in that tangent space, and mapped back through
//! the exponential. The output is exactly a unit quaternion by
//! construction. An earlier revision filtered hemisphere-aligned quaternion
//! components and renormalized, which approximates the group only while the
//! per-sample rotation is small; the tangent-space path agrees with it in
//! that regime (worst divergence 7.3e-7 rad on a 0.2 rad, 1 Hz sway) and
//! stays correct where components alias (a 0.8 rad-per-sample spin filtered
//! at 2 Hz: tangent-space error 8.5e-3 rad against the true smooth
//! rotation, component filtering 0.64 rad — the tests measure both).
//!
//! "Preserve real strike-speed characteristics" is not something a
//! low-pass filter can guarantee, so this module measures it instead:
//! [`FilterReport::speed_retention`] is the peak speed of every channel's
//! global trajectory after filtering divided by the peak before, and a
//! caller curating strikes sets its own acceptance bound on it.

use core::fmt;
use robot_types::frames::{Quat, Vec3};
use robot_types::motion::{JointValue, MotionKind, MotionRecord, ProvenanceStep, Validity};
use robot_types::time::Duration;
use robot_types::units::Hertz;

use crate::kinematics::{all_positions, frame_times, velocities};
use crate::REVISION;

/// How the filter is applied.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FilterMode {
    /// Forward-backward: zero phase delay, offline only.
    ZeroPhase,
    /// Forward only: a real, recorded phase delay, usable on a live stream.
    Causal,
}

impl FilterMode {
    const fn name(self) -> &'static str {
        match self {
            Self::ZeroPhase => "zero_phase",
            Self::Causal => "causal",
        }
    }
}

/// Why filtering failed.
#[derive(Clone, PartialEq, Debug)]
pub enum FilterError {
    /// The cutoff was non-positive, non-finite, or at or above Nyquist.
    InvalidCutoff {
        /// The cutoff requested.
        cutoff: Hertz,
        /// The record's sampling rate.
        sample_rate: Hertz,
    },
    /// Frame intervals differ by more than 1 % from their median; resample
    /// first.
    NonUniformSampling {
        /// Index of the first interval out of tolerance.
        at_frame: usize,
    },
    /// The record had fewer than two frames.
    TooShort,
}

impl fmt::Display for FilterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCutoff {
                cutoff,
                sample_rate,
            } => write!(f, "cutoff {cutoff} invalid for sample rate {sample_rate}"),
            Self::NonUniformSampling { at_frame } => {
                write!(
                    f,
                    "non-uniform sampling at frame {at_frame}; resample first"
                )
            }
            Self::TooShort => f.write_str("need at least two frames to filter"),
        }
    }
}

impl core::error::Error for FilterError {}

/// A second-order low-pass section, `a0` normalized to one.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Biquad {
    /// Numerator coefficients.
    pub b: [f64; 3],
    /// Denominator coefficients `a1`, `a2`.
    pub a: [f64; 2],
}

impl Biquad {
    /// A Butterworth low-pass section with a -3 dB point at `cutoff` (in
    /// the same unit as `sample_rate`).
    #[must_use]
    pub fn butterworth_lowpass(cutoff: f64, sample_rate: f64) -> Self {
        Self::from_prewarped((core::f64::consts::PI * cutoff / sample_rate).tan())
    }

    fn from_prewarped(k: f64) -> Self {
        let r2 = core::f64::consts::SQRT_2;
        let norm = 1.0 / (1.0 + r2 * k + k * k);
        let b0 = k * k * norm;
        Self {
            b: [b0, 2.0 * b0, b0],
            a: [2.0 * (k * k - 1.0) * norm, (1.0 - r2 * k + k * k) * norm],
        }
    }

    /// Gain at DC (exactly one for a low-pass section, up to rounding).
    #[must_use]
    pub fn dc_gain(&self) -> f64 {
        (self.b[0] + self.b[1] + self.b[2]) / (1.0 + self.a[0] + self.a[1])
    }

    /// Group delay at DC, in samples.
    #[must_use]
    pub fn dc_group_delay_samples(&self) -> f64 {
        let nb = self.b[1] + 2.0 * self.b[2];
        let db = self.b[0] + self.b[1] + self.b[2];
        let na = self.a[0] + 2.0 * self.a[1];
        let da = 1.0 + self.a[0] + self.a[1];
        nb / db - na / da
    }

    /// Magnitude response at `freq` (same unit as `sample_rate`).
    #[must_use]
    pub fn magnitude(&self, freq: f64, sample_rate: f64) -> f64 {
        let w = 2.0 * core::f64::consts::PI * freq / sample_rate;
        let (c1, s1, c2, s2) = (w.cos(), w.sin(), (2.0 * w).cos(), (2.0 * w).sin());
        let num = (
            self.b[0] + self.b[1] * c1 + self.b[2] * c2,
            -self.b[1] * s1 - self.b[2] * s2,
        );
        let den = (
            1.0 + self.a[0] * c1 + self.a[1] * c2,
            -self.a[0] * s1 - self.a[1] * s2,
        );
        num.0.hypot(num.1) / den.0.hypot(den.1)
    }

    /// Filters `x` in place, forward, from the steady state of `x[0]`.
    pub fn run(&self, x: &mut [f64]) {
        let Some(&x0) = x.first() else { return };
        let y0 = x0 * self.dc_gain();
        let (mut x1, mut x2, mut y1, mut y2) = (x0, x0, y0, y0);
        for v in x.iter_mut() {
            let xn = *v;
            let yn =
                self.b[0] * xn + self.b[1] * x1 + self.b[2] * x2 - self.a[0] * y1 - self.a[1] * y2;
            x2 = x1;
            x1 = xn;
            y2 = y1;
            y1 = yn;
            *v = yn;
        }
    }

    /// Forward-backward filtering with odd-reflection padding at both ends.
    pub fn run_zero_phase(&self, x: &mut [f64]) {
        let n = x.len();
        if n < 2 {
            return;
        }
        let pad = (n - 1).min(12);
        let mut ext = Vec::with_capacity(n + 2 * pad);
        ext.extend((1..=pad).rev().map(|i| 2.0 * x[0] - x[i]));
        ext.extend_from_slice(x);
        ext.extend((1..=pad).map(|i| 2.0 * x[n - 1] - x[n - 1 - i]));
        self.run(&mut ext);
        ext.reverse();
        self.run(&mut ext);
        ext.reverse();
        x.copy_from_slice(&ext[pad..pad + n]);
    }
}

/// Winter's cutoff correction for two passes of a second-order Butterworth
/// section: the single-pass design cutoff is the requested one divided by
/// this, in the prewarped domain.
fn zero_phase_correction() -> f64 {
    (core::f64::consts::SQRT_2 - 1.0).powf(0.25)
}

/// What filtering did.
#[derive(Clone, PartialEq, Debug)]
pub struct FilterReport {
    /// Delay recorded in provenance.
    pub phase_delay: Duration,
    /// Per channel, peak speed after filtering over peak speed before;
    /// `None` for a channel that never moved or was never present.
    pub speed_retention: Vec<Option<f64>>,
    /// Root translation's peak-speed retention.
    pub root_speed_retention: Option<f64>,
}

impl FilterReport {
    /// The smallest per-channel retention, with its channel index.
    #[must_use]
    pub fn worst_retention(&self) -> Option<(usize, f64)> {
        self.speed_retention
            .iter()
            .enumerate()
            .filter_map(|(i, r)| r.map(|r| (i, r)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
    }
}

/// The sampling rate of a uniformly sampled record.
pub fn uniform_rate<K: MotionKind>(record: &MotionRecord<K>) -> Result<Hertz, FilterError> {
    if record.frames.len() < 2 {
        return Err(FilterError::TooShort);
    }
    let dts: Vec<i64> = record
        .frames
        .windows(2)
        .map(|w| w[1].timestamp.nanos() - w[0].timestamp.nanos())
        .collect();
    let mut sorted = dts.clone();
    sorted.sort_unstable();
    let median = sorted[sorted.len() / 2];
    if median <= 0 {
        return Err(FilterError::NonUniformSampling { at_frame: 0 });
    }
    if let Some(i) = dts
        .iter()
        .position(|&dt| (dt - median).abs() as f64 > 0.01 * median as f64)
    {
        return Err(FilterError::NonUniformSampling { at_frame: i });
    }
    Ok(Hertz(1e9 / median as f64))
}

/// Low-pass filters the root pose and every present sample of `record`.
///
/// Missing samples are never read or written: each channel is filtered
/// over its contiguous runs of present samples independently.
pub fn lowpass<K: MotionKind>(
    record: &mut MotionRecord<K>,
    cutoff: Hertz,
    mode: FilterMode,
) -> Result<FilterReport, FilterError> {
    let fs = uniform_rate(record)?;
    let fc = cutoff.value();
    if !fc.is_finite() || fc <= 0.0 || fc >= 0.5 * fs.value() {
        return Err(FilterError::InvalidCutoff {
            cutoff,
            sample_rate: fs,
        });
    }
    let k = (core::f64::consts::PI * fc / fs.value()).tan();
    let section = match mode {
        FilterMode::ZeroPhase => Biquad::from_prewarped(k / zero_phase_correction()),
        FilterMode::Causal => Biquad::from_prewarped(k),
    };
    let apply = |x: &mut [f64]| match mode {
        FilterMode::ZeroPhase => section.run_zero_phase(x),
        FilterMode::Causal => section.run(x),
    };

    let times = frame_times(record);
    let before = all_positions(record);
    let root_before: Vec<_> = record
        .frames
        .iter()
        .map(|f| Some(f.root.translation))
        .collect();

    // Root translation and rotation: always present.
    let n = record.frames.len();
    for axis in 0..3 {
        let mut x: Vec<f64> = record
            .frames
            .iter()
            .map(|f| f.root.translation.to_xyz()[axis])
            .collect();
        apply(&mut x);
        for (f, v) in record.frames.iter_mut().zip(x) {
            let mut t = f.root.translation.to_xyz();
            t[axis] = v;
            f.root.translation = Vec3::from_xyz(t);
        }
    }
    let mut roots: Vec<Quat> = record.frames.iter().map(|f| f.root.rotation).collect();
    filter_quats(&mut roots, &apply);
    for (f, q) in record.frames.iter_mut().zip(roots) {
        f.root.rotation = q;
    }

    for ci in 0..record.channels.len() {
        let mut start = 0;
        while start < n {
            if record.frames[start].validity[ci] == Validity::Missing {
                start += 1;
                continue;
            }
            let mut end = start;
            while end < n && record.frames[end].validity[ci] != Validity::Missing {
                end += 1;
            }
            filter_run(&mut record.frames[start..end], ci, &apply);
            start = end;
        }
    }

    let phase_delay = match mode {
        FilterMode::ZeroPhase => Duration::ZERO,
        FilterMode::Causal => {
            Duration::from_secs_f64(section.dc_group_delay_samples() / fs.value())
        }
    };
    let after = all_positions(record);
    let speed_retention = (0..record.channels.len())
        .map(|ci| {
            let b: Vec<_> = before.iter().map(|p| p[ci]).collect();
            let a: Vec<_> = after.iter().map(|p| p[ci]).collect();
            retention(&b, &a, &times)
        })
        .collect();
    let root_after: Vec<_> = record
        .frames
        .iter()
        .map(|f| Some(f.root.translation))
        .collect();
    let report = FilterReport {
        phase_delay,
        speed_retention,
        root_speed_retention: retention(&root_before, &root_after, &times),
    };
    let mut step = ProvenanceStep::new("lowpass", REVISION)
        .with_parameter("design", "butterworth2")
        .with_parameter("mode", mode.name())
        .with_parameter("cutoff_hz", fc.to_string())
        .with_phase_delay(phase_delay);
    if let Some((ci, r)) = report.worst_retention() {
        step = step
            .with_parameter("worst_speed_retention", r.to_string())
            .with_parameter(
                "worst_speed_retention_channel",
                record.channels[ci].name.to_string(),
            );
    }
    record.record_step(step);
    Ok(report)
}

fn filter_run(
    frames: &mut [robot_types::motion::MotionFrame],
    ci: usize,
    apply: &impl Fn(&mut [f64]),
) {
    if frames.len() < 3 {
        return;
    }
    match &frames[0].values[ci] {
        JointValue::Rotation { .. } => {
            let mut qs: Vec<Quat> = frames
                .iter()
                .map(|f| match f.values[ci] {
                    JointValue::Rotation { rotation, .. } => rotation,
                    _ => Quat::IDENTITY,
                })
                .collect();
            filter_quats(&mut qs, apply);
            for (f, q) in frames.iter_mut().zip(qs) {
                if let JointValue::Rotation { rotation, .. } = &mut f.values[ci] {
                    *rotation = q;
                }
            }
        }
        JointValue::Scalar { .. } => {
            let mut x: Vec<f64> = frames
                .iter()
                .map(|f| match f.values[ci] {
                    JointValue::Scalar { value, .. } => value,
                    _ => 0.0,
                })
                .collect();
            apply(&mut x);
            for (f, v) in frames.iter_mut().zip(x) {
                if let JointValue::Scalar { value, .. } = &mut f.values[ci] {
                    *value = v;
                }
            }
        }
        JointValue::Position { .. } => {
            for axis in 0..3 {
                let mut x: Vec<f64> = frames
                    .iter()
                    .map(|f| match f.values[ci] {
                        JointValue::Position { position } => position.to_xyz()[axis],
                        _ => 0.0,
                    })
                    .collect();
                apply(&mut x);
                for (f, v) in frames.iter_mut().zip(x) {
                    if let JointValue::Position { position } = &mut f.values[ci] {
                        let mut p = position.to_xyz();
                        p[axis] = v;
                        *position = Vec3::from_xyz(p);
                    }
                }
            }
        }
    }
}

/// Filters a quaternion sequence on the rotation group (see the module
/// documentation): tangent-space components relative to the first sample,
/// branch-unwrapped for continuity, filtered, and exponentiated back.
fn filter_quats(qs: &mut [Quat], apply: &impl Fn(&mut [f64])) {
    if qs.len() < 2 {
        return;
    }
    let reference = qs[0].normalized();
    let inverse = reference.conjugate();
    let mut tangent = Vec::with_capacity(qs.len());
    let mut prev = Vec3::ZERO;
    for q in qs.iter() {
        let v = unwrap_branch((inverse * *q).normalized().to_rotation_vector(), prev);
        tangent.push(v);
        prev = v;
    }
    let mut comps: [Vec<f64>; 3] = [
        tangent.iter().map(|v| v.x).collect(),
        tangent.iter().map(|v| v.y).collect(),
        tangent.iter().map(|v| v.z).collect(),
    ];
    for c in &mut comps {
        apply(c);
    }
    for (i, q) in qs.iter_mut().enumerate() {
        let v = Vec3::new(comps[0][i], comps[1][i], comps[2][i]);
        *q = (reference * Quat::from_rotation_vector(v)).normalized();
    }
}

/// The rotation-vector branch of the same rotation nearest `prev`.
///
/// `v` and `v + 2 pi k * v_unit` (all integers `k`, including the negative
/// branches through the opposite axis) name one rotation; the principal
/// value folds an accumulating rotation back at `pi`, which would put a
/// spurious reversal into the tangent curve. Choosing the branch nearest the
/// previous sample keeps the curve continuous, which is what makes filtering
/// its components meaningful.
fn unwrap_branch(v: Vec3, prev: Vec3) -> Vec3 {
    const TAU: f64 = 2.0 * core::f64::consts::PI;
    let theta = v.norm();
    if theta < 1e-9 {
        // The axis is undefined at the identity; every branch is a full turn
        // about some axis, and the one continuous with the curve keeps the
        // previous winding along the previous direction.
        let p = prev.norm();
        let turns = (p / TAU).round();
        if turns == 0.0 || p < 1e-9 {
            return v;
        }
        return prev * (turns * TAU / p);
    }
    let unit = v * (1.0 / theta);
    let k = ((unit.dot(prev) - theta) / TAU).round();
    v + unit * (TAU * k)
}

fn peak_speed(positions: &[Option<Vec3>], times: &[f64]) -> f64 {
    velocities(positions, times)
        .into_iter()
        .flatten()
        .map(Vec3::norm)
        .fold(0.0, f64::max)
}

fn retention(before: &[Option<Vec3>], after: &[Option<Vec3>], times: &[f64]) -> Option<f64> {
    let b = peak_speed(before, times);
    (b > 1e-12).then(|| peak_speed(after, times) / b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::root_path;
    use core::f64::consts::PI;

    #[test]
    fn the_design_is_minus_3_db_at_cutoff_with_unit_dc_gain() {
        let s = Biquad::butterworth_lowpass(10.0, 200.0);
        assert!((s.dc_gain() - 1.0).abs() < 1e-12);
        let m = s.magnitude(10.0, 200.0);
        assert!((m - core::f64::consts::FRAC_1_SQRT_2).abs() < 1e-9, "{m}");
    }

    #[test]
    fn the_zero_phase_correction_puts_the_two_pass_response_at_minus_3_db() {
        let fs = 200.0;
        let k = (PI * 10.0 / fs).tan();
        let s = Biquad::from_prewarped(k / zero_phase_correction());
        let m2 = s.magnitude(10.0, fs).powi(2);
        assert!((m2 - core::f64::consts::FRAC_1_SQRT_2).abs() < 1e-9, "{m2}");
    }

    #[test]
    fn the_recorded_causal_delay_matches_the_measured_lag_of_a_slow_ramp() {
        // A ramp is shifted by exactly the DC group delay once transients
        // die out, which makes it an independent check of the formula.
        let s = Biquad::butterworth_lowpass(5.0, 100.0);
        let mut x: Vec<f64> = (0..400).map(f64::from).collect();
        s.run(&mut x);
        let lag = 399.0 - x[399];
        assert!((lag - s.dc_group_delay_samples()).abs() < 1e-6, "{lag}");
    }

    #[test]
    fn zero_phase_filtering_has_no_lag_and_records_zero_delay() {
        let rate = 100.0;
        // 1 Hz sine root path, well below a 10 Hz cutoff.
        let mut record = root_path(300, rate, |t| Vec3::new((2.0 * PI * t).sin(), 0.0, 0.0));
        let original: Vec<f64> = record.frames.iter().map(|f| f.root.translation.x).collect();
        let report = lowpass(&mut record, Hertz(10.0), FilterMode::ZeroPhase).unwrap();
        assert_eq!(report.phase_delay, Duration::ZERO);
        assert_eq!(record.accumulated_phase_delay(), Duration::ZERO);
        let max_err = record
            .frames
            .iter()
            .zip(&original)
            .map(|(f, x)| (f.root.translation.x - x).abs())
            .fold(0.0, f64::max);
        assert!(max_err < 0.01, "{max_err}");
        assert!((report.root_speed_retention.unwrap() - 1.0).abs() < 0.02);
    }

    #[test]
    fn causal_filtering_records_its_delay_and_attenuates_noise() {
        let rate = 100.0;
        // Alternating (Nyquist-rate) noise riding on a constant.
        let mut record = root_path(200, rate, |t| {
            let i = (t * rate).round() as i64;
            Vec3::new(1.0 + if i % 2 == 0 { 0.1 } else { -0.1 }, 0.0, 0.0)
        });
        let report = lowpass(&mut record, Hertz(5.0), FilterMode::Causal).unwrap();
        let expected = Biquad::butterworth_lowpass(5.0, rate).dc_group_delay_samples() / rate;
        assert!((report.phase_delay.as_secs_f64() - expected).abs() < 1e-9);
        assert!(report.phase_delay.is_positive());
        assert_eq!(record.accumulated_phase_delay(), report.phase_delay);
        let tail_ripple = record.frames[150..]
            .iter()
            .map(|f| (f.root.translation.x - 1.0).abs())
            .fold(0.0, f64::max);
        assert!(tail_ripple < 0.01, "{tail_ripple}");
    }

    #[test]
    fn a_cutoff_at_nyquist_and_non_uniform_sampling_are_rejected() {
        let mut record = root_path(10, 100.0, |_| Vec3::ZERO);
        assert!(matches!(
            lowpass(&mut record, Hertz(50.0), FilterMode::ZeroPhase),
            Err(FilterError::InvalidCutoff { .. })
        ));
        record.frames.remove(4);
        assert!(matches!(
            lowpass(&mut record, Hertz(5.0), FilterMode::ZeroPhase),
            Err(FilterError::NonUniformSampling { at_frame: 3 })
        ));
    }

    #[test]
    fn a_fast_strike_reports_lost_peak_speed() {
        // A 50 ms Gaussian velocity burst: a 4 Hz cutoff must cost it speed,
        // and the report says so rather than passing it silently.
        let rate = 200.0;
        let mut record = root_path(200, rate, |t| {
            let s = (t - 0.5) / 0.02;
            Vec3::new(
                0.5 * (1.0 + libm_erf(s / core::f64::consts::SQRT_2)),
                0.0,
                0.0,
            )
        });
        let report = lowpass(&mut record, Hertz(4.0), FilterMode::ZeroPhase).unwrap();
        let r = report.root_speed_retention.unwrap();
        assert!(r < 0.6, "{r}");
        let mut record2 = root_path(200, rate, |t| {
            let s = (t - 0.5) / 0.02;
            Vec3::new(
                0.5 * (1.0 + libm_erf(s / core::f64::consts::SQRT_2)),
                0.0,
                0.0,
            )
        });
        let report2 = lowpass(&mut record2, Hertz(40.0), FilterMode::ZeroPhase).unwrap();
        assert!(report2.root_speed_retention.unwrap() > 0.95);
    }

    /// The retired component filter, kept here as the comparison reference:
    /// hemisphere-align, filter the four components, renormalize.
    fn component_filter_quats(qs: &mut [Quat], apply: &impl Fn(&mut [f64])) {
        for i in 1..qs.len() {
            let (p, q) = (qs[i - 1], qs[i]);
            if p.w * q.w + p.x * q.x + p.y * q.y + p.z * q.z < 0.0 {
                qs[i] = Quat {
                    w: -q.w,
                    x: -q.x,
                    y: -q.y,
                    z: -q.z,
                };
            }
        }
        let mut comps: [Vec<f64>; 4] = [
            qs.iter().map(|q| q.w).collect(),
            qs.iter().map(|q| q.x).collect(),
            qs.iter().map(|q| q.y).collect(),
            qs.iter().map(|q| q.z).collect(),
        ];
        for c in &mut comps {
            apply(c);
        }
        for (i, q) in qs.iter_mut().enumerate() {
            *q = Quat {
                w: comps[0][i],
                x: comps[1][i],
                y: comps[2][i],
                z: comps[3][i],
            }
            .normalized();
        }
    }

    fn angle_between(a: Quat, b: Quat) -> f64 {
        (a.conjugate() * b).normalized().to_rotation_vector().norm()
    }

    #[test]
    fn group_filtering_agrees_with_component_filtering_on_small_rotations() {
        // A 0.2 rad, 1 Hz sway at 100 Hz: the regime the component filter
        // was valid in. The tangent-space path must reproduce it closely,
        // or the change would alter every result the old approximation
        // produced well.
        let axis = Vec3::new(1.0, 1.0, 0.0).normalized(1e-12).unwrap();
        let qs: Vec<Quat> = (0..300)
            .map(|i| {
                let t = i as f64 / 100.0;
                Quat::from_rotation_vector(axis * (0.2 * (2.0 * PI * t).sin()))
            })
            .collect();
        let section = Biquad::butterworth_lowpass(10.0, 100.0);
        let apply = |x: &mut [f64]| section.run_zero_phase(x);
        let mut group = qs.clone();
        filter_quats(&mut group, &apply);
        let mut component = qs;
        component_filter_quats(&mut component, &apply);
        let worst = group
            .iter()
            .zip(&component)
            .map(|(a, b)| angle_between(*a, *b))
            .fold(0.0, f64::max);
        assert!(worst < 1e-3, "paths diverge by {worst} rad");
    }

    #[test]
    fn group_filtering_survives_large_rotations_where_components_alias() {
        // A steady 0.8 rad-per-sample spin about z with +/-0.05 rad jitter.
        // The quaternion components oscillate at about 6.4 Hz here, so a
        // 2 Hz low-pass collapses them toward their mean and the normalized
        // result is far from the true motion; the tangent curve is a clean
        // ramp plus jitter, which the same filter cleans up.
        let mut seed = 0x9e37u64;
        let mut noise = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.1
        };
        let truth: Vec<Quat> = (0..300)
            .map(|i| Quat::from_rotation_vector(Vec3::new(0.0, 0.0, 0.8 * i as f64)))
            .collect();
        let noisy: Vec<Quat> = truth
            .iter()
            .map(|q| (*q * Quat::from_rotation_vector(Vec3::new(0.0, 0.0, noise()))).normalized())
            .collect();
        let section = Biquad::butterworth_lowpass(2.0, 100.0);
        let apply = |x: &mut [f64]| section.run_zero_phase(x);
        let mut group = noisy.clone();
        filter_quats(&mut group, &apply);
        let mut component = noisy;
        component_filter_quats(&mut component, &apply);
        // Compare away from the ends: a 2 Hz section's response spans tens
        // of samples, so the boundary-padding transient reaches that far in.
        let err = |filtered: &[Quat]| {
            filtered
                .iter()
                .zip(&truth)
                .skip(80)
                .take(140)
                .map(|(a, b)| angle_between(*a, *b))
                .fold(0.0, f64::max)
        };
        let (group_err, component_err) = (err(&group), err(&component));
        assert!(
            group_err < 0.05,
            "group filtering drifted {group_err} rad from the true spin"
        );
        assert!(
            group_err < 0.1 * component_err,
            "group {group_err} rad should beat components {component_err} rad decisively"
        );
        // And the group path is exactly unit-norm, not merely renormalized
        // from a collapsed component vector.
        for q in &group {
            let n = (q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z).sqrt();
            assert!((n - 1.0).abs() < 1e-12);
        }
    }

    /// Abramowitz and Stegun 7.1.26, absolute error below 1.5e-7.
    fn libm_erf(x: f64) -> f64 {
        let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
        let y = 1.0
            - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736)
                * t
                + 0.254_829_592)
                * t
                * (-x * x).exp();
        y.copysign(x)
    }
}

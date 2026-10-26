//! Monotonic time, named clocks, and the sampling/arrival distinction.
//!
//! SPECS.md Section 5 requires that a timestamp declare whether it is a
//! *sampling instant* or an *arrival time*, "because an arrival time silently
//! relabeled as a sampling time corrupts every delay model trained on it".
//! That distinction is carried in the type, not in a comment, so a delay model
//! cannot consume an arrival time without the code saying so.
//!
//! Appendix A.3 fixes time as explicit nanosecond integers with a named clock.

use core::fmt;
use core::ops::{Add, AddAssign, Sub};

/// A signed nanosecond duration.
///
/// Signed because the difference of two timestamps is routinely negative when a
/// record is malformed, and a wrapping unsigned subtraction would hide it.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
pub struct Duration(i64);

impl Duration {
    /// Zero duration.
    pub const ZERO: Self = Self(0);

    /// From whole nanoseconds.
    #[inline]
    #[must_use]
    pub const fn from_nanos(nanos: i64) -> Self {
        Self(nanos)
    }

    /// From whole microseconds.
    #[inline]
    #[must_use]
    pub const fn from_micros(micros: i64) -> Self {
        Self(micros.saturating_mul(1_000))
    }

    /// From whole milliseconds.
    #[inline]
    #[must_use]
    pub const fn from_millis(millis: i64) -> Self {
        Self(millis.saturating_mul(1_000_000))
    }

    /// From whole seconds.
    #[inline]
    #[must_use]
    pub const fn from_secs(secs: i64) -> Self {
        Self(secs.saturating_mul(1_000_000_000))
    }

    /// From fractional seconds, rounding to the nearest nanosecond.
    ///
    /// Saturates rather than wrapping on a non-finite or out-of-range input, so
    /// a corrupt frame time in a source file becomes a clamped duration that
    /// validation rejects rather than a wrapped negative one.
    #[must_use]
    pub fn from_secs_f64(secs: f64) -> Self {
        if !secs.is_finite() {
            return Self(if secs > 0.0 { i64::MAX } else { i64::MIN });
        }
        let nanos = (secs * 1e9).round();
        if nanos >= i64::MAX as f64 {
            Self(i64::MAX)
        } else if nanos <= i64::MIN as f64 {
            Self(i64::MIN)
        } else {
            Self(nanos as i64)
        }
    }

    /// Whole nanoseconds.
    #[inline]
    #[must_use]
    pub const fn nanos(self) -> i64 {
        self.0
    }

    /// Fractional seconds, for reports and physics timesteps.
    #[inline]
    #[must_use]
    pub fn as_secs_f64(self) -> f64 {
        self.0 as f64 * 1e-9
    }

    /// Fractional milliseconds, for latency reports.
    #[inline]
    #[must_use]
    pub fn as_millis_f64(self) -> f64 {
        self.0 as f64 * 1e-6
    }

    /// Absolute value.
    #[inline]
    #[must_use]
    pub const fn abs(self) -> Self {
        Self(self.0.abs())
    }

    /// True for a strictly positive duration.
    #[inline]
    #[must_use]
    pub const fn is_positive(self) -> bool {
        self.0 > 0
    }

    /// Saturating multiplication by an integer factor.
    #[inline]
    #[must_use]
    pub const fn saturating_mul(self, factor: i64) -> Self {
        Self(self.0.saturating_mul(factor))
    }
}

impl fmt::Debug for Duration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ns", self.0)
    }
}

impl fmt::Display for Duration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let abs = self.0.abs();
        if abs >= 1_000_000_000 {
            write!(f, "{:.3} s", self.as_secs_f64())
        } else if abs >= 1_000_000 {
            write!(f, "{:.3} ms", self.as_millis_f64())
        } else if abs >= 1_000 {
            write!(f, "{:.3} us", self.0 as f64 * 1e-3)
        } else {
            write!(f, "{} ns", self.0)
        }
    }
}

impl Add for Duration {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }
}

impl AddAssign for Duration {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 = self.0.saturating_add(rhs.0);
    }
}

impl Sub for Duration {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0.saturating_sub(rhs.0))
    }
}

/// Identity of the clock a timestamp was read from.
///
/// Two timestamps may only be subtracted when their clocks agree; a host
/// monotonic reading and a device counter are not comparable without a recorded
/// mapping, and [`Timestamp::since`] enforces that.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum ClockId {
    /// Host monotonic clock; the project's default control-path clock.
    HostMonotonic,
    /// A simulated clock advanced by the integrator, not by wall time.
    Simulation,
    /// A clock internal to a device, identified by the device's stable index.
    ///
    /// Device counters wrap and drift; a mapping to [`ClockId::HostMonotonic`]
    /// is a recorded artifact, never an assumption.
    Device(u16),
    /// A clock belonging to an imported source file, such as a motion capture
    /// session. Never comparable to a host clock without a recorded mapping.
    Source(u16),
}

impl fmt::Display for ClockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HostMonotonic => write!(f, "host-monotonic"),
            Self::Simulation => write!(f, "simulation"),
            Self::Device(i) => write!(f, "device:{i}"),
            Self::Source(i) => write!(f, "source:{i}"),
        }
    }
}

/// Whether a timestamp names the instant a quantity was *measured* or the
/// instant the host *received* it.
///
/// SPECS.md Section 5: where a device does not provide a sampling instant, the
/// record must say that the timestamp is an arrival time and carry the
/// resulting uncertainty.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum StampKind {
    /// The device reported the instant of measurement.
    Sampling,
    /// The host observed the sample at this instant; the true sampling instant
    /// is earlier by an unknown amount bounded by `uncertainty`.
    Arrival,
}

/// A point in time on a named clock, with its kind and uncertainty.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Timestamp {
    nanos: i64,
    clock: ClockId,
    kind: StampKind,
    uncertainty: Duration,
}

/// Why two timestamps could not be compared.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TimeError {
    /// The timestamps came from different clocks and no mapping was supplied.
    ClockMismatch {
        /// Clock of the later timestamp.
        lhs: ClockId,
        /// Clock of the earlier timestamp.
        rhs: ClockId,
    },
}

impl fmt::Display for TimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ClockMismatch { lhs, rhs } => {
                write!(f, "cannot subtract timestamps from clocks {lhs} and {rhs}")
            }
        }
    }
}

impl core::error::Error for TimeError {}

impl Timestamp {
    /// A sampling instant reported by the source itself.
    #[inline]
    #[must_use]
    pub const fn sampling(nanos: i64, clock: ClockId) -> Self {
        Self {
            nanos,
            clock,
            kind: StampKind::Sampling,
            uncertainty: Duration::ZERO,
        }
    }

    /// An arrival time, with the bound on how much earlier the true sampling
    /// instant may have been.
    #[inline]
    #[must_use]
    pub const fn arrival(nanos: i64, clock: ClockId, uncertainty: Duration) -> Self {
        Self {
            nanos,
            clock,
            kind: StampKind::Arrival,
            uncertainty,
        }
    }

    /// Nanoseconds on this timestamp's clock.
    #[inline]
    #[must_use]
    pub const fn nanos(self) -> i64 {
        self.nanos
    }

    /// The clock this timestamp was read from.
    #[inline]
    #[must_use]
    pub const fn clock(self) -> ClockId {
        self.clock
    }

    /// Whether this is a sampling instant or an arrival time.
    #[inline]
    #[must_use]
    pub const fn kind(self) -> StampKind {
        self.kind
    }

    /// Bound on the error between this timestamp and the true sampling instant.
    #[inline]
    #[must_use]
    pub const fn uncertainty(self) -> Duration {
        self.uncertainty
    }

    /// True when this timestamp may be used as a measurement instant in a delay
    /// model without widening its error bars.
    #[inline]
    #[must_use]
    pub const fn is_sampling(self) -> bool {
        matches!(self.kind, StampKind::Sampling)
    }

    /// Elapsed time since `earlier`, rejecting a clock mismatch.
    ///
    /// The combined uncertainty of the two endpoints is returned alongside the
    /// interval, so an age computed from two arrival times carries its own error
    /// bound instead of reading as exact.
    pub fn since(self, earlier: Self) -> Result<(Duration, Duration), TimeError> {
        if self.clock != earlier.clock {
            return Err(TimeError::ClockMismatch {
                lhs: self.clock,
                rhs: earlier.clock,
            });
        }
        Ok((
            Duration(self.nanos.saturating_sub(earlier.nanos)),
            self.uncertainty + earlier.uncertainty,
        ))
    }

    /// Advances by `delta`, preserving clock, kind and uncertainty.
    #[inline]
    #[must_use]
    pub const fn advanced(self, delta: Duration) -> Self {
        Self {
            nanos: self.nanos.saturating_add(delta.nanos()),
            ..self
        }
    }

    /// Re-stamps on another clock through a recorded offset.
    ///
    /// `offset_uncertainty` is added to this timestamp's own uncertainty,
    /// because a mapping between clocks is itself measured and never exact.
    #[must_use]
    pub fn mapped_to(self, clock: ClockId, offset: Duration, offset_uncertainty: Duration) -> Self {
        Self {
            nanos: self.nanos.saturating_add(offset.nanos()),
            clock,
            kind: self.kind,
            uncertainty: self.uncertainty + offset_uncertainty,
        }
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let k = match self.kind {
            StampKind::Sampling => "sampled",
            StampKind::Arrival => "arrived",
        };
        write!(f, "{}ns@{} ({k}", self.nanos, self.clock)?;
        if self.uncertainty != Duration::ZERO {
            write!(f, " +/-{}", self.uncertainty)?;
        }
        write!(f, ")")
    }
}

/// A monotonic clock source.
///
/// The control runtime takes its time through this trait so that a deterministic
/// replay can substitute a stepped clock without the runtime behaving
/// differently (SPECS.md Section 4 `offline-replay`).
pub trait MonotonicClock {
    /// Current instant. Must never decrease between calls.
    fn now(&self) -> Timestamp;

    /// The clock's identity, written into every record it stamps.
    fn id(&self) -> ClockId;
}

/// A clock advanced explicitly by the caller.
///
/// Used by simulation, replay and tests. Because it is stepped rather than read,
/// a run driven by it is reproducible.
#[derive(Clone, Debug)]
pub struct SteppedClock {
    nanos: i64,
    id: ClockId,
}

impl SteppedClock {
    /// A clock starting at `start_nanos` on `id`.
    #[must_use]
    pub const fn new(start_nanos: i64, id: ClockId) -> Self {
        Self {
            nanos: start_nanos,
            id,
        }
    }

    /// Advances the clock. Panics on a negative delta: a monotonic clock that
    /// can be stepped backwards is not monotonic, and silently clamping would
    /// hide the caller's bug.
    pub fn advance(&mut self, delta: Duration) {
        assert!(delta.nanos() >= 0, "SteppedClock cannot move backwards");
        self.nanos = self.nanos.saturating_add(delta.nanos());
    }
}

impl MonotonicClock for SteppedClock {
    fn now(&self) -> Timestamp {
        Timestamp::sampling(self.nanos, self.id)
    }

    fn id(&self) -> ClockId {
        self.id
    }
}

/// The host monotonic clock.
///
/// `std::time::Instant` is monotonic but has no public epoch, so the first
/// reading is captured at construction and subsequent readings are expressed as
/// an offset from it. The resulting integers are comparable within one process,
/// which is exactly the guarantee Appendix A.3 asks for.
#[derive(Clone, Debug)]
pub struct HostClock {
    origin: std::time::Instant,
}

impl HostClock {
    /// Captures the origin instant.
    #[must_use]
    pub fn new() -> Self {
        Self {
            origin: std::time::Instant::now(),
        }
    }
}

impl Default for HostClock {
    fn default() -> Self {
        Self::new()
    }
}

impl MonotonicClock for HostClock {
    fn now(&self) -> Timestamp {
        let elapsed = self.origin.elapsed();
        let nanos = i64::try_from(elapsed.as_nanos()).unwrap_or(i64::MAX);
        Timestamp::sampling(nanos, ClockId::HostMonotonic)
    }

    fn id(&self) -> ClockId {
        ClockId::HostMonotonic
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_conversions_round_trip() {
        assert_eq!(Duration::from_millis(10).nanos(), 10_000_000);
        assert_eq!(Duration::from_secs_f64(0.01).nanos(), 10_000_000);
        assert!((Duration::from_micros(1500).as_millis_f64() - 1.5).abs() < 1e-12);
    }

    #[test]
    fn nonfinite_frame_time_saturates_instead_of_wrapping() {
        assert_eq!(Duration::from_secs_f64(f64::NAN).nanos(), i64::MIN);
        assert_eq!(Duration::from_secs_f64(f64::INFINITY).nanos(), i64::MAX);
        assert_eq!(Duration::from_secs_f64(1e30).nanos(), i64::MAX);
    }

    #[test]
    fn subtracting_across_clocks_is_rejected() {
        let a = Timestamp::sampling(10, ClockId::HostMonotonic);
        let b = Timestamp::sampling(5, ClockId::Device(0));
        assert!(matches!(a.since(b), Err(TimeError::ClockMismatch { .. })));
    }

    #[test]
    fn arrival_uncertainty_accumulates_through_subtraction() {
        let clock = ClockId::Device(3);
        let a = Timestamp::arrival(2_000_000, clock, Duration::from_micros(200));
        let b = Timestamp::arrival(1_000_000, clock, Duration::from_micros(300));
        let (delta, unc) = a.since(b).unwrap();
        assert_eq!(delta, Duration::from_millis(1));
        assert_eq!(unc, Duration::from_micros(500));
        assert!(!a.is_sampling());
    }

    #[test]
    fn clock_mapping_widens_uncertainty() {
        let device = Timestamp::sampling(1_000, ClockId::Device(1));
        let host = device.mapped_to(
            ClockId::HostMonotonic,
            Duration::from_millis(5),
            Duration::from_micros(50),
        );
        assert_eq!(host.clock(), ClockId::HostMonotonic);
        assert_eq!(host.nanos(), 5_001_000);
        assert_eq!(host.uncertainty(), Duration::from_micros(50));
        assert!(host.is_sampling());
    }

    #[test]
    #[should_panic(expected = "cannot move backwards")]
    fn stepped_clock_refuses_to_go_backwards() {
        let mut c = SteppedClock::new(0, ClockId::Simulation);
        c.advance(Duration::from_nanos(-1));
    }

    #[test]
    fn host_clock_is_nondecreasing() {
        let c = HostClock::new();
        let a = c.now();
        let b = c.now();
        assert!(b.nanos() >= a.nanos());
        assert_eq!(c.id(), ClockId::HostMonotonic);
    }
}

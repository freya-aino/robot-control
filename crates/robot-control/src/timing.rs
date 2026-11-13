//! The four timing quantities of SPECS.md Section 5, kept distinct.
//!
//! "These quantities are distinct and a report MUST NOT collapse them":
//! compute duration, sensor-to-command age, decision age, deadline margin.
//! [`TickTiming`] carries exactly these four, named, so a report built from
//! it cannot accidentally conflate two of them into one number.

use robot_types::time::Duration;

/// The timing record for one control tick.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TickTiming {
    /// Policy-path entry to command-ready.
    pub compute_duration: Duration,
    /// Sensor sampling instant to command accepted by the actuator
    /// interface. `None` when no feedback was available this tick.
    pub sensor_to_command_age: Option<Duration>,
    /// Age of the most recent tactical-tier decision applied this tick, if
    /// any crossed a tier boundary (SPECS.md Section 5.1 item 4).
    pub decision_age: Option<Duration>,
    /// Budget minus `compute_duration`; negative means an overrun.
    pub deadline_margin: Duration,
}

impl TickTiming {
    /// Whether this tick missed its deadline.
    #[must_use]
    pub fn missed_deadline(&self) -> bool {
        self.deadline_margin.nanos() < 0
    }
}

/// A fixed-capacity ring of recent tick timings, for a p50/p95/p99/p99.9/max
/// report (SPECS.md Section 5) without unbounded growth.
pub struct TimingHistory<const N: usize> {
    entries: [Option<TickTiming>; N],
    next: usize,
    filled: usize,
}

impl<const N: usize> Default for TimingHistory<N> {
    fn default() -> Self {
        Self {
            entries: [None; N],
            next: 0,
            filled: 0,
        }
    }
}

impl<const N: usize> TimingHistory<N> {
    /// An empty history.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one tick's timing, overwriting the oldest entry once full.
    pub fn record(&mut self, timing: TickTiming) {
        self.entries[self.next] = Some(timing);
        self.next = (self.next + 1) % N;
        self.filled = (self.filled + 1).min(N);
    }

    /// Number of entries currently held.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.filled
    }

    /// True when no tick has been recorded.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.filled == 0
    }

    /// The compute duration at the given percentile (0–100), by nearest-rank
    /// on the sorted sample. Returns `None` for an empty history.
    #[must_use]
    pub fn compute_duration_percentile(&self, percentile: f64) -> Option<Duration> {
        let mut samples: Vec<i64> = self
            .entries
            .iter()
            .flatten()
            .map(|t| t.compute_duration.nanos())
            .collect();
        if samples.is_empty() {
            return None;
        }
        samples.sort_unstable();
        let rank = ((percentile / 100.0) * (samples.len() as f64 - 1.0)).round() as usize;
        Some(Duration::from_nanos(samples[rank.min(samples.len() - 1)]))
    }

    /// Worst observed compute duration.
    #[must_use]
    pub fn max_compute_duration(&self) -> Option<Duration> {
        self.entries
            .iter()
            .flatten()
            .map(|t| t.compute_duration)
            .max()
    }

    /// Count of ticks that missed their deadline.
    #[must_use]
    pub fn missed_deadline_count(&self) -> usize {
        self.entries
            .iter()
            .flatten()
            .filter(|t| t.missed_deadline())
            .count()
    }
}

/// p50, p95, p99, p99.9 and maximum of one quantity.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Tails {
    /// Samples.
    pub n: usize,
    /// Median.
    pub p50: Duration,
    /// 95th percentile.
    pub p95: Duration,
    /// 99th percentile.
    pub p99: Duration,
    /// 99.9th percentile.
    pub p999: Duration,
    /// Maximum.
    pub max: Duration,
}

impl Tails {
    /// Nearest-rank tails of `samples` (nanoseconds); `None` when empty.
    #[must_use]
    pub fn of(mut samples: Vec<i64>) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }
        samples.sort_unstable();
        let at = |p: f64| {
            // Nearest rank; the epsilon keeps 99.9 % of 1000 at rank 999
            // when the product rounds to 999.0000000000001.
            let rank = ((p / 100.0) * samples.len() as f64 - 1e-9).ceil() as usize;
            Duration::from_nanos(samples[rank.clamp(1, samples.len()) - 1])
        };
        Some(Self {
            n: samples.len(),
            p50: at(50.0),
            p95: at(95.0),
            p99: at(99.0),
            p999: at(99.9),
            max: Duration::from_nanos(*samples.last().expect("non-empty")),
        })
    }
}

/// The report SPECS.md Section 5 requires: each of its four quantities
/// separately, never collapsed, with deadline misses.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TimingReport {
    /// Compute duration tails.
    pub compute: Option<Tails>,
    /// Sensor-to-command age tails (ticks that had feedback).
    pub sensor_to_command: Option<Tails>,
    /// Decision age tails (ticks that applied an upper-tier decision).
    pub decision_age: Option<Tails>,
    /// Deadline margin tails.
    pub deadline_margin: Option<Tails>,
    /// Ticks whose compute exceeded the budget.
    pub missed_deadlines: usize,
    /// Ticks recorded.
    pub ticks: usize,
}

impl<const N: usize> TimingHistory<N> {
    /// The four-quantity report over the recorded ticks. Builds vectors:
    /// call it off the control thread.
    #[must_use]
    pub fn report(&self) -> TimingReport {
        let all = || self.entries.iter().flatten();
        TimingReport {
            compute: Tails::of(all().map(|t| t.compute_duration.nanos()).collect()),
            sensor_to_command: Tails::of(
                all()
                    .filter_map(|t| t.sensor_to_command_age.map(|d| d.nanos()))
                    .collect(),
            ),
            decision_age: Tails::of(
                all()
                    .filter_map(|t| t.decision_age.map(|d| d.nanos()))
                    .collect(),
            ),
            deadline_margin: Tails::of(all().map(|t| t.deadline_margin.nanos()).collect()),
            missed_deadlines: self.missed_deadline_count(),
            ticks: self.filled,
        }
    }
}

#[cfg(test)]
mod report_tests {
    use super::*;

    #[test]
    fn the_report_keeps_the_four_quantities_apart() {
        let mut h: TimingHistory<1000> = TimingHistory::new();
        for i in 0..1000 {
            h.record(TickTiming {
                compute_duration: Duration::from_micros(i + 1),
                sensor_to_command_age: (i % 2 == 0).then(|| Duration::from_micros(2 * i)),
                decision_age: None,
                deadline_margin: Duration::from_micros(8000 - i),
            });
        }
        let r = h.report();
        let c = r.compute.unwrap();
        assert_eq!(c.p50, Duration::from_micros(500));
        assert_eq!(c.p999, Duration::from_micros(999));
        assert_eq!(c.max, Duration::from_micros(1000));
        assert_eq!(r.sensor_to_command.unwrap().n, 500);
        assert!(r.decision_age.is_none());
        assert_eq!(r.ticks, 1000);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn timing(compute_ns: i64, budget_ns: i64) -> TickTiming {
        TickTiming {
            compute_duration: Duration::from_nanos(compute_ns),
            sensor_to_command_age: None,
            decision_age: None,
            deadline_margin: Duration::from_nanos(budget_ns - compute_ns),
        }
    }

    #[test]
    fn missed_deadline_is_a_negative_margin() {
        assert!(!timing(5, 10).missed_deadline());
        assert!(timing(15, 10).missed_deadline());
    }

    #[test]
    fn history_wraps_without_growing_past_capacity() {
        let mut history: TimingHistory<4> = TimingHistory::new();
        for i in 1..=6 {
            history.record(timing(i * 100, 1000));
        }
        assert_eq!(history.len(), 4);
        // The oldest two (100, 200) have been overwritten; max remaining is 600.
        assert_eq!(
            history.max_compute_duration(),
            Some(Duration::from_nanos(600))
        );
    }

    #[test]
    fn percentile_and_missed_count_are_reported_separately() {
        let mut history: TimingHistory<8> = TimingHistory::new();
        for i in 1..=5 {
            history.record(timing(i * 10, 1000));
        }
        history.record(timing(2000, 1000)); // one overrun
        assert_eq!(history.missed_deadline_count(), 1);
        let p99 = history.compute_duration_percentile(99.0).unwrap();
        assert_eq!(p99, Duration::from_nanos(2000));
        assert!(history.compute_duration_percentile(0.0).unwrap() <= p99);
    }

    #[test]
    fn an_empty_history_reports_none() {
        let history: TimingHistory<4> = TimingHistory::new();
        assert!(history.compute_duration_percentile(50.0).is_none());
        assert!(history.max_compute_duration().is_none());
        assert!(history.is_empty());
    }
}

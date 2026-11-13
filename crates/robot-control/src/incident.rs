//! The reserved safety-event channel of SPECS.md Section 5: "Safety-relevant
//! incidents require a reserved event channel and bounded pre/post-event
//! recording buffer", and "Log overflow must never block actuation; emit
//! dropped-record counts."
//!
//! [`IncidentRecorder`] keeps the last `PRE` tick records in a fixed ring.
//! When an incident is triggered it freezes that ring as the incident's
//! pre-event window and collects the next `POST` records, then holds the
//! completed capture until [`IncidentRecorder::take`] drains it off the
//! control thread. A trigger that arrives while a capture is in progress or
//! still waiting to be taken is not queued and does not block: it is
//! counted in [`IncidentRecorder::dropped`]. Every array is fixed at
//! compile time, so recording never allocates.

use robot_supervisor::policy::Response;

use crate::timing::TickTiming;

/// One tick, as the incident channel keeps it.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct TickRecord {
    /// Tick index.
    pub tick: u64,
    /// Its timing.
    pub timing: TickTiming,
    /// The supervisor's response.
    pub response: Response,
}

/// A completed capture.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Incident<const PRE: usize, const POST: usize> {
    /// What triggered it.
    pub reason: &'static str,
    /// The tick it was triggered on.
    pub at_tick: u64,
    /// Up to `PRE` records before the trigger, oldest first.
    pub before: [Option<TickRecord>; PRE],
    /// `POST` records from the trigger on.
    pub after: [Option<TickRecord>; POST],
}

/// The bounded incident recorder.
pub struct IncidentRecorder<const PRE: usize, const POST: usize> {
    ring: [Option<TickRecord>; PRE],
    next: usize,
    capture: Option<(Incident<PRE, POST>, usize)>,
    done: Option<Incident<PRE, POST>>,
    dropped: u64,
}

impl<const PRE: usize, const POST: usize> Default for IncidentRecorder<PRE, POST> {
    fn default() -> Self {
        Self {
            ring: [None; PRE],
            next: 0,
            capture: None,
            done: None,
            dropped: 0,
        }
    }
}

impl<const PRE: usize, const POST: usize> IncidentRecorder<PRE, POST> {
    /// An empty recorder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one tick.
    pub fn record(&mut self, r: TickRecord) {
        if let Some((inc, filled)) = &mut self.capture {
            inc.after[*filled] = Some(r);
            *filled += 1;
            if *filled == POST {
                let (inc, _) = self.capture.take().expect("present");
                self.done = Some(inc);
            }
        }
        if PRE > 0 {
            self.ring[self.next] = Some(r);
            self.next = (self.next + 1) % PRE;
        }
    }

    /// Starts a capture at tick `at_tick`, before that tick is recorded.
    /// Returns `false` (and counts a drop) if one is already in progress or
    /// unread.
    pub fn trigger(&mut self, reason: &'static str, at_tick: u64) -> bool {
        if self.capture.is_some() || self.done.is_some() {
            self.dropped += 1;
            return false;
        }
        let mut before = [None; PRE];
        for (k, slot) in before.iter_mut().enumerate() {
            *slot = self.ring[(self.next + k) % PRE.max(1)];
        }
        let inc = Incident {
            reason,
            at_tick,
            before,
            after: [None; POST],
        };
        if POST == 0 {
            self.done = Some(inc);
        } else {
            self.capture = Some((inc, 0));
        }
        true
    }

    /// Takes a completed capture, freeing the channel.
    pub fn take(&mut self) -> Option<Incident<PRE, POST>> {
        self.done.take()
    }

    /// Triggers that were dropped.
    #[must_use]
    pub const fn dropped(&self) -> u64 {
        self.dropped
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_types::time::Duration;

    fn rec(tick: u64) -> TickRecord {
        TickRecord {
            tick,
            timing: TickTiming {
                compute_duration: Duration::from_micros(tick as i64),
                sensor_to_command_age: None,
                decision_age: None,
                deadline_margin: Duration::ZERO,
            },
            response: Response::ClampAndContinue,
        }
    }

    #[test]
    fn a_capture_holds_the_window_around_the_trigger() {
        let mut r: IncidentRecorder<3, 2> = IncidentRecorder::new();
        for t in 0..10 {
            if t == 6 {
                assert!(r.trigger("over-temperature", 6));
            }
            r.record(rec(t));
        }
        let inc = r.take().unwrap();
        let before: Vec<u64> = inc.before.iter().flatten().map(|x| x.tick).collect();
        let after: Vec<u64> = inc.after.iter().flatten().map(|x| x.tick).collect();
        assert_eq!(before, [3, 4, 5]);
        assert_eq!(after, [6, 7]);
        assert!(r.take().is_none());
    }

    #[test]
    fn a_second_trigger_is_dropped_and_counted_not_queued() {
        let mut r: IncidentRecorder<2, 4> = IncidentRecorder::new();
        assert!(r.trigger("a", 0));
        r.record(rec(0));
        assert!(!r.trigger("b", 1));
        assert_eq!(r.dropped(), 1);
    }
}

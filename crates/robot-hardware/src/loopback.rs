//! A loopback device: commands become feedback, optionally after a fixed
//! number of ticks of delay.
//!
//! PLAN.md Phase 5 names this the first adapter to implement, "before a
//! replay device, then an actual documented protocol" — it is what the
//! control runtime's own plumbing is tested against before any real
//! transport exists, and it is the mechanism SPECS.md Section 4's
//! `hw-shadow` entry precondition means by "the disable verified by loopback
//! readback": send a command with transmission disabled, then poll and
//! confirm nothing echoes back.
//!
//! This device allocates per `send_command` call (`Vec::to_vec` and a
//! `Vec::remove`), which the deadline-critical control path of SPECS.md
//! Section 5 does not permit. That is acceptable here specifically because a
//! loopback device stands in for hardware during development and testing of
//! the *runtime's own plumbing* — it is not itself a certified zero-allocation
//! implementation, and a protocol adapter built against real hardware is
//! where that guarantee must actually be met.

use crate::device::{AdapterError, FeedbackHeader, HardwareAdapter};
use robot_types::time::{Duration, Timestamp};

/// A command in flight, counting down the polls remaining before it
/// surfaces as feedback.
struct Pending {
    values: Vec<f64>,
    timestamp: Timestamp,
    remaining_polls: usize,
}

/// A loopback adapter with a fixed command-to-feedback delay, measured in
/// `poll_feedback` calls rather than wall time.
///
/// `delay_ticks = 0` means a command is visible on the very next
/// `poll_feedback` call after it was sent; `delay_ticks = k` means that call
/// and `k` further calls must happen first — `k + 1` polls in total. The
/// queue of pending commands is reserved once, at construction, to its
/// expected depth; normal use (one send per control cycle) never grows it.
pub struct LoopbackDevice {
    width: usize,
    transmit_enabled: bool,
    delay_ticks: usize,
    pending: Vec<Pending>,
    last_liveness: Option<Timestamp>,
}

impl LoopbackDevice {
    /// Builds a loopback device of the given width and delay.
    #[must_use]
    pub fn new(width: usize, delay_ticks: usize) -> Self {
        Self {
            width,
            transmit_enabled: true,
            delay_ticks,
            pending: Vec::with_capacity(delay_ticks + 2),
            last_liveness: None,
        }
    }
}

impl HardwareAdapter for LoopbackDevice {
    fn width(&self) -> usize {
        self.width
    }

    fn send_command(&mut self, command: &[f64], timestamp: Timestamp) -> Result<(), AdapterError> {
        if command.len() != self.width {
            return Err(AdapterError::WidthMismatch {
                expected: self.width,
                found: command.len(),
            });
        }
        if !self.transmit_enabled {
            return Err(AdapterError::TransmitDisabled);
        }
        self.pending.push(Pending {
            values: command.to_vec(),
            timestamp,
            remaining_polls: self.delay_ticks + 1,
        });
        self.last_liveness = Some(timestamp);
        Ok(())
    }

    fn poll_feedback(
        &mut self,
        positions: &mut [f64],
    ) -> Result<Option<FeedbackHeader>, AdapterError> {
        if positions.len() != self.width {
            return Err(AdapterError::WidthMismatch {
                expected: self.width,
                found: positions.len(),
            });
        }
        if self.pending.is_empty() {
            return Ok(None);
        }
        // Every pending command's delay clock runs in parallel: a tick
        // elapses for all of them on each poll, regardless of queue
        // position, so two commands sent back to back surface independently
        // rather than one blocking the other's countdown.
        for p in &mut self.pending {
            p.remaining_polls = p.remaining_polls.saturating_sub(1);
        }
        if self.pending[0].remaining_polls > 0 {
            return Ok(None);
        }
        let due = self.pending.remove(0);
        positions.copy_from_slice(&due.values);
        Ok(Some(FeedbackHeader {
            timestamp: due.timestamp,
            is_sampling_instant: true,
        }))
    }

    fn transmit_enabled(&self) -> bool {
        self.transmit_enabled
    }

    fn set_transmit_enabled(&mut self, enabled: bool) {
        self.transmit_enabled = enabled;
    }

    fn link_age(&self, now: Timestamp) -> Option<Duration> {
        self.last_liveness
            .and_then(|t| now.since(t).ok())
            .map(|(age, _)| age)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_types::time::ClockId;

    #[test]
    fn a_zero_delay_loopback_echoes_on_the_very_next_poll() {
        let mut dev = LoopbackDevice::new(2, 0);
        let t = Timestamp::sampling(0, ClockId::HostMonotonic);
        dev.send_command(&[1.0, 2.0], t).unwrap();
        let mut out = [0.0; 2];
        let header = dev.poll_feedback(&mut out).unwrap().unwrap();
        assert_eq!(out, [1.0, 2.0]);
        assert_eq!(header.timestamp, t);
    }

    #[test]
    fn a_delayed_loopback_withholds_feedback_until_the_delay_elapses() {
        let mut dev = LoopbackDevice::new(1, 2);
        let t0 = Timestamp::sampling(0, ClockId::HostMonotonic);
        dev.send_command(&[9.0], t0).unwrap();
        let mut out = [0.0];

        // Not due on the first two polls.
        assert!(dev.poll_feedback(&mut out).unwrap().is_none());
        assert!(dev.poll_feedback(&mut out).unwrap().is_none());
        // Due on the third poll.
        let header = dev.poll_feedback(&mut out).unwrap().unwrap();
        assert_eq!(out, [9.0]);
        assert_eq!(header.timestamp, t0);
    }

    #[test]
    fn two_sends_surface_in_order_each_after_its_own_delay() {
        let mut dev = LoopbackDevice::new(1, 1);
        let clock = ClockId::HostMonotonic;
        let t0 = Timestamp::sampling(0, clock);
        let t1 = Timestamp::sampling(1, clock);
        dev.send_command(&[1.0], t0).unwrap();
        dev.send_command(&[2.0], t1).unwrap();
        let mut out = [0.0];
        assert!(dev.poll_feedback(&mut out).unwrap().is_none());
        let h0 = dev.poll_feedback(&mut out).unwrap().unwrap();
        assert_eq!((out[0], h0.timestamp), (1.0, t0));
        let h1 = dev.poll_feedback(&mut out).unwrap().unwrap();
        assert_eq!((out[0], h1.timestamp), (2.0, t1));
    }

    #[test]
    fn disabling_transmit_blocks_the_command_and_readback_confirms_it() {
        let mut dev = LoopbackDevice::new(1, 0);
        dev.set_transmit_enabled(false);
        let t = Timestamp::sampling(0, ClockId::HostMonotonic);
        assert_eq!(
            dev.send_command(&[5.0], t),
            Err(AdapterError::TransmitDisabled)
        );
        let mut out = [0.0];
        // Nothing was ever written, so there is nothing to read back.
        assert!(dev.poll_feedback(&mut out).unwrap().is_none());
    }

    #[test]
    fn width_mismatches_are_rejected() {
        let mut dev = LoopbackDevice::new(2, 0);
        let t = Timestamp::sampling(0, ClockId::HostMonotonic);
        assert!(matches!(
            dev.send_command(&[1.0], t),
            Err(AdapterError::WidthMismatch {
                expected: 2,
                found: 1
            })
        ));
        let mut out = [0.0; 3];
        assert!(matches!(
            dev.poll_feedback(&mut out),
            Err(AdapterError::WidthMismatch {
                expected: 2,
                found: 3
            })
        ));
    }

    #[test]
    fn link_age_tracks_the_last_successful_send() {
        let mut dev = LoopbackDevice::new(1, 0);
        let clock = ClockId::HostMonotonic;
        assert!(dev.link_age(Timestamp::sampling(0, clock)).is_none());
        dev.send_command(&[1.0], Timestamp::sampling(1_000_000, clock))
            .unwrap();
        let age = dev.link_age(Timestamp::sampling(5_000_000, clock)).unwrap();
        assert_eq!(age, Duration::from_millis(4));
    }
}

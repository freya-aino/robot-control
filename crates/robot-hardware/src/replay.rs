//! A replay device: deterministic playback of recorded feedback, never
//! actuating.
//!
//! PLAN.md Phase 5 names this the second adapter to implement, after
//! loopback and before "an actual documented protocol." It is what
//! SPECS.md Section 4's `offline-replay` mode runs against: "None; no device
//! attached" for actuation authority, with the fault response "Abort with a
//! diagnostic; never fabricate missing samples" — this device returns
//! [`crate::device::AdapterError::ReplayExhausted`] rather than repeating or
//! interpolating once its recorded frames run out.

use crate::device::{AdapterError, FeedbackHeader, HardwareAdapter};
use robot_types::time::{Duration, Timestamp};

/// One recorded feedback sample.
#[derive(Clone, PartialEq, Debug)]
pub struct RecordedFrame {
    /// Per-component feedback values.
    pub values: Vec<f64>,
    /// When this sample was originally taken.
    pub timestamp: Timestamp,
    /// Whether `timestamp` was a true sampling instant in the original
    /// recording.
    pub is_sampling_instant: bool,
}

/// A device that plays back a fixed, pre-loaded sequence of frames.
///
/// The frame sequence is supplied whole at construction (loaded by the
/// caller from a `robot_types::episode::EpisodeRecord` or a fixture); this
/// device performs no I/O and allocates nothing beyond that one load.
#[derive(PartialEq, Debug)]
pub struct ReplayDevice {
    width: usize,
    frames: Vec<RecordedFrame>,
    cursor: usize,
}

/// Why a replay device could not be built.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ReplayBuildError {
    /// A frame's width did not match the declared width.
    FrameWidthMismatch {
        /// Index of the offending frame.
        frame: usize,
        /// Declared width.
        expected: usize,
        /// Width found.
        found: usize,
    },
    /// The sequence had no frames.
    Empty,
}

impl core::fmt::Display for ReplayBuildError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::FrameWidthMismatch {
                frame,
                expected,
                found,
            } => {
                write!(f, "frame {frame} has width {found}, expected {expected}")
            }
            Self::Empty => write!(f, "replay sequence has no frames"),
        }
    }
}

impl core::error::Error for ReplayBuildError {}

impl ReplayDevice {
    /// Builds a replay device from a loaded frame sequence.
    pub fn new(width: usize, frames: Vec<RecordedFrame>) -> Result<Self, ReplayBuildError> {
        if frames.is_empty() {
            return Err(ReplayBuildError::Empty);
        }
        for (i, frame) in frames.iter().enumerate() {
            if frame.values.len() != width {
                return Err(ReplayBuildError::FrameWidthMismatch {
                    frame: i,
                    expected: width,
                    found: frame.values.len(),
                });
            }
        }
        Ok(Self {
            width,
            frames,
            cursor: 0,
        })
    }

    /// Total frames loaded.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// How many frames remain to be read.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.frames.len() - self.cursor
    }

    /// Rewinds to the first frame, for a repeated deterministic run.
    pub fn rewind(&mut self) {
        self.cursor = 0;
    }
}

impl HardwareAdapter for ReplayDevice {
    fn width(&self) -> usize {
        self.width
    }

    /// Always refuses: a replay device never actuates.
    fn send_command(&mut self, command: &[f64], _timestamp: Timestamp) -> Result<(), AdapterError> {
        if command.len() != self.width {
            return Err(AdapterError::WidthMismatch {
                expected: self.width,
                found: command.len(),
            });
        }
        Err(AdapterError::TransmitDisabled)
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
        if self.cursor >= self.frames.len() {
            return Err(AdapterError::ReplayExhausted);
        }
        let frame = &self.frames[self.cursor];
        positions.copy_from_slice(&frame.values);
        let header = FeedbackHeader {
            timestamp: frame.timestamp,
            is_sampling_instant: frame.is_sampling_instant,
        };
        self.cursor += 1;
        Ok(Some(header))
    }

    /// Always `false`: a replay device has no transmit path to enable.
    fn transmit_enabled(&self) -> bool {
        false
    }

    /// A no-op: a replay device's transmit path cannot be enabled.
    fn set_transmit_enabled(&mut self, _enabled: bool) {}

    fn link_age(&self, now: Timestamp) -> Option<Duration> {
        let frame = self.frames.get(self.cursor.saturating_sub(1))?;
        now.since(frame.timestamp).ok().map(|(age, _)| age)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_types::time::ClockId;

    fn frame(value: f64, nanos: i64) -> RecordedFrame {
        RecordedFrame {
            values: vec![value],
            timestamp: Timestamp::sampling(nanos, ClockId::Source(0)),
            is_sampling_instant: true,
        }
    }

    #[test]
    fn frames_play_back_in_recorded_order() {
        let mut dev =
            ReplayDevice::new(1, vec![frame(1.0, 0), frame(2.0, 10), frame(3.0, 20)]).unwrap();
        let mut out = [0.0];
        for expected in [1.0, 2.0, 3.0] {
            dev.poll_feedback(&mut out).unwrap();
            assert_eq!(out[0], expected);
        }
    }

    #[test]
    fn exhaustion_is_a_distinct_terminal_error_not_a_repeat() {
        let mut dev = ReplayDevice::new(1, vec![frame(1.0, 0)]).unwrap();
        let mut out = [0.0];
        dev.poll_feedback(&mut out).unwrap();
        assert_eq!(
            dev.poll_feedback(&mut out),
            Err(AdapterError::ReplayExhausted)
        );
        // It does not silently repeat the last frame.
        assert_eq!(out[0], 1.0);
    }

    #[test]
    fn rewind_makes_a_run_reproducible() {
        let mut dev = ReplayDevice::new(1, vec![frame(1.0, 0), frame(2.0, 10)]).unwrap();
        let mut out = [0.0];
        dev.poll_feedback(&mut out).unwrap();
        dev.poll_feedback(&mut out).unwrap();
        assert_eq!(dev.remaining(), 0);
        dev.rewind();
        assert_eq!(dev.remaining(), 2);
        dev.poll_feedback(&mut out).unwrap();
        assert_eq!(out[0], 1.0);
    }

    #[test]
    fn replay_never_actuates() {
        let mut dev = ReplayDevice::new(1, vec![frame(1.0, 0)]).unwrap();
        let t = Timestamp::sampling(0, ClockId::Source(0));
        assert_eq!(
            dev.send_command(&[9.0], t),
            Err(AdapterError::TransmitDisabled)
        );
        assert!(!dev.transmit_enabled());
        dev.set_transmit_enabled(true); // no-op
        assert!(!dev.transmit_enabled());
    }

    #[test]
    fn an_empty_or_mismatched_frame_set_is_rejected_at_construction() {
        assert_eq!(ReplayDevice::new(1, vec![]), Err(ReplayBuildError::Empty));
        assert_eq!(
            ReplayDevice::new(2, vec![frame(1.0, 0)]),
            Err(ReplayBuildError::FrameWidthMismatch {
                frame: 0,
                expected: 2,
                found: 1
            })
        );
    }
}

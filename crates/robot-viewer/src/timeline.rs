//! Synchronized playback over a sequence of [`ViewerFrame`]s.
//!
//! PLAN.md's Phase 2 asks for "synchronized playback," which this module
//! treats as a scrubbing/seek operation over an ordered frame sequence —
//! again, data, not a rendered timeline widget. [`Timeline::nearest`] is
//! the operation an actual player (or `robot-cli`'s own text output) would
//! call once per displayed instant.

use crate::frame::ViewerFrame;
use robot_types::time::{ClockId, TimeError, Timestamp};

/// Anything that carries one timestamp and can be sequenced on a [`Timeline`].
pub trait Timed {
    /// The instant this record describes.
    fn timestamp(&self) -> Timestamp;
}

impl Timed for ViewerFrame {
    fn timestamp(&self) -> Timestamp {
        self.timestamp
    }
}

/// Why a frame could not be appended to a [`Timeline`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TimelineError {
    /// The frame's timestamp is on a different clock than the timeline's.
    ClockMismatch {
        /// This timeline's clock.
        expected: ClockId,
        /// The frame's clock.
        found: ClockId,
    },
    /// The frame's timestamp is not later than the last frame already
    /// appended — a timeline is append-only in time order, not a general
    /// collection.
    OutOfOrder,
}

impl core::fmt::Display for TimelineError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ClockMismatch { expected, found } => {
                write!(
                    f,
                    "frame clock {found} does not match timeline clock {expected}"
                )
            }
            Self::OutOfOrder => write!(f, "frame timestamp is not later than the previous frame"),
        }
    }
}

impl core::error::Error for TimelineError {}

/// An ordered, single-clock sequence of inspection frames, supporting
/// nearest-frame lookup by timestamp.
///
/// Every frame shares one [`ClockId`] — SPECS.md §5 never treats two
/// different clocks' readings as directly comparable, and a playback
/// timeline that silently mixed them would be exactly that mistake with
/// extra steps.
#[derive(Clone, Debug)]
pub struct Timeline<F: Timed = ViewerFrame> {
    clock: ClockId,
    frames: Vec<F>,
}

impl Timeline<ViewerFrame> {
    /// An empty timeline of [`ViewerFrame`]s on the given clock.
    #[must_use]
    pub const fn new(clock: ClockId) -> Self {
        Self::with_clock(clock)
    }
}

impl<F: Timed> Timeline<F> {
    /// An empty timeline of any [`Timed`] record on the given clock.
    #[must_use]
    pub const fn with_clock(clock: ClockId) -> Self {
        Self {
            clock,
            frames: Vec::new(),
        }
    }

    /// Appends a frame. The frame's timestamp must be on this timeline's
    /// clock and strictly later than the last frame already present.
    pub fn push(&mut self, frame: F) -> Result<(), TimelineError> {
        let stamp = frame.timestamp();
        if stamp.clock() != self.clock {
            return Err(TimelineError::ClockMismatch {
                expected: self.clock,
                found: stamp.clock(),
            });
        }
        if let Some(last) = self.frames.last() {
            if stamp.nanos() <= last.timestamp().nanos() {
                return Err(TimelineError::OutOfOrder);
            }
        }
        self.frames.push(frame);
        Ok(())
    }

    /// Number of frames.
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// True when the timeline has no frames.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// The frame whose timestamp is closest to `at`, breaking ties toward
    /// the earlier frame.
    ///
    /// `at` must be on this timeline's clock, enforced the same way a
    /// [`Timestamp::since`] comparison would be — a scrub position from the
    /// wrong clock is a bug to surface, not a value to silently coerce.
    pub fn nearest(&self, at: Timestamp) -> Result<Option<&F>, TimeError> {
        let Some(first) = self.frames.first() else {
            return Ok(None);
        };
        // Any frame shares the timeline's clock already; use the first
        // frame purely to get a `since` clock-mismatch check against `at`.
        first.timestamp().since(at)?;

        let index = self
            .frames
            .partition_point(|f| f.timestamp().nanos() < at.nanos());
        let candidate = match (index.checked_sub(1), self.frames.get(index)) {
            (Some(prev_index), Some(_)) => {
                let prev = &self.frames[prev_index];
                let next = &self.frames[index];
                let prev_distance = at.nanos() - prev.timestamp().nanos();
                let next_distance = next.timestamp().nanos() - at.nanos();
                if next_distance < prev_distance {
                    next
                } else {
                    prev
                }
            }
            (Some(prev_index), None) => &self.frames[prev_index],
            (None, Some(_)) => &self.frames[index],
            (None, None) => unreachable!("checked non-empty above"),
        };
        Ok(Some(candidate))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_model::kinematics::LinkPoses;

    fn frame_at(nanos: i64) -> ViewerFrame {
        ViewerFrame {
            timestamp: Timestamp::sampling(nanos, ClockId::Simulation),
            source_skeleton: None,
            target_skeleton: LinkPoses::from_poses(vec![]),
            contacts: vec![],
            tracking_error: vec![],
        }
    }

    #[test]
    fn frames_must_be_appended_in_strictly_increasing_time_order() {
        let mut t = Timeline::new(ClockId::Simulation);
        t.push(frame_at(10)).unwrap();
        assert_eq!(t.push(frame_at(10)), Err(TimelineError::OutOfOrder));
        assert_eq!(t.push(frame_at(5)), Err(TimelineError::OutOfOrder));
    }

    #[test]
    fn a_frame_on_the_wrong_clock_is_rejected() {
        let mut t = Timeline::new(ClockId::Simulation);
        let wrong_clock = ViewerFrame {
            timestamp: Timestamp::sampling(0, ClockId::HostMonotonic),
            ..frame_at(0)
        };
        assert_eq!(
            t.push(wrong_clock),
            Err(TimelineError::ClockMismatch {
                expected: ClockId::Simulation,
                found: ClockId::HostMonotonic,
            })
        );
    }

    #[test]
    fn nearest_picks_the_closer_neighbor_and_breaks_ties_early() {
        let mut t = Timeline::new(ClockId::Simulation);
        t.push(frame_at(0)).unwrap();
        t.push(frame_at(10)).unwrap();
        t.push(frame_at(20)).unwrap();

        assert_eq!(
            t.nearest(Timestamp::sampling(4, ClockId::Simulation))
                .unwrap()
                .unwrap()
                .timestamp
                .nanos(),
            0
        );
        assert_eq!(
            t.nearest(Timestamp::sampling(6, ClockId::Simulation))
                .unwrap()
                .unwrap()
                .timestamp
                .nanos(),
            10
        );
        // Exact tie between 0 and 10 at t=5 breaks toward the earlier frame.
        assert_eq!(
            t.nearest(Timestamp::sampling(5, ClockId::Simulation))
                .unwrap()
                .unwrap()
                .timestamp
                .nanos(),
            0
        );
        assert_eq!(
            t.nearest(Timestamp::sampling(1000, ClockId::Simulation))
                .unwrap()
                .unwrap()
                .timestamp
                .nanos(),
            20
        );
    }

    #[test]
    fn nearest_on_an_empty_timeline_is_none_not_an_error() {
        let t = Timeline::new(ClockId::Simulation);
        assert_eq!(
            t.nearest(Timestamp::sampling(0, ClockId::Simulation))
                .unwrap(),
            None
        );
    }

    #[test]
    fn nearest_with_the_wrong_clock_is_an_error() {
        let mut t = Timeline::new(ClockId::Simulation);
        t.push(frame_at(0)).unwrap();
        assert!(t
            .nearest(Timestamp::sampling(0, ClockId::HostMonotonic))
            .is_err());
    }
}

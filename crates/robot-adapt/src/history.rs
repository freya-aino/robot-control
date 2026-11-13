//! The deployable observation history an on-robot adaptation module
//! reads, per SPECS.md Section 11.3 obligations 2 and 3.
//!
//! A [`HistoryWindow`] is declared with the exact observation names it
//! consumes, so obligation 2 ("MUST declare the observation set available
//! at deployment and MUST NOT consume privileged simulator state") is a
//! constructor check against [`PRIVILEGED_PREFIX`] rather than a review
//! comment. It holds a fixed number of samples at a declared period and
//! knows when its content is usable: obligation 3 requires that a module
//! whose history is "short, stale or discontinuous" produce "the declared
//! default context, not an extrapolation", and [`HistoryWindow::state`]
//! says which of those holds, so the caller can do exactly that.

use core::fmt;
use std::collections::VecDeque;

/// Observation names starting with this are simulator-only (true
/// friction, true mass, contact ground truth) and may not enter a
/// deployable window.
pub const PRIVILEGED_PREFIX: &str = "privileged.";

/// Why a window could not be built or fed.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum HistoryError {
    /// A declared observation is privileged.
    Privileged(String),
    /// A sample had the wrong width.
    Width {
        /// Expected.
        expected: usize,
        /// Found.
        found: usize,
    },
    /// Length or period was zero.
    Invalid(&'static str),
}

impl fmt::Display for HistoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Privileged(n) => write!(f, "observation {n:?} is privileged simulator state"),
            Self::Width { expected, found } => {
                write!(f, "sample width {found}, expected {expected}")
            }
            Self::Invalid(what) => write!(f, "invalid history window: {what}"),
        }
    }
}

impl core::error::Error for HistoryError {}

/// Whether a window's content may be used.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WindowState {
    /// Full, fresh and continuous.
    Ready,
    /// Fewer samples than the window length.
    Short {
        /// Samples held.
        have: usize,
    },
    /// The newest sample is older than allowed.
    Stale,
    /// A gap between consecutive samples exceeded the allowed jitter.
    Discontinuous,
}

/// A fixed-length history of deployable observations.
#[derive(Clone, Debug)]
pub struct HistoryWindow {
    names: Vec<String>,
    length: usize,
    period_ns: i64,
    jitter_ns: i64,
    samples: VecDeque<(i64, Vec<f64>)>,
}

impl HistoryWindow {
    /// A window of `length` samples, nominally `period_ns` apart, allowing
    /// `jitter_ns` of deviation per interval.
    pub fn new(
        names: Vec<String>,
        length: usize,
        period_ns: i64,
        jitter_ns: i64,
    ) -> Result<Self, HistoryError> {
        if let Some(n) = names.iter().find(|n| n.starts_with(PRIVILEGED_PREFIX)) {
            return Err(HistoryError::Privileged(n.clone()));
        }
        if length == 0 || period_ns <= 0 || jitter_ns < 0 || names.is_empty() {
            return Err(HistoryError::Invalid("length, period, jitter or names"));
        }
        Ok(Self {
            names,
            length,
            period_ns,
            jitter_ns,
            samples: VecDeque::with_capacity(length),
        })
    }

    /// The declared observation names.
    #[must_use]
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Width of the flattened window.
    #[must_use]
    pub fn flat_width(&self) -> usize {
        self.names.len() * self.length
    }

    /// Adds a sample stamped `t_ns`. A sample not newer than the last one
    /// clears the window: time going backwards is a discontinuity no
    /// reordering can repair.
    pub fn push(&mut self, t_ns: i64, sample: &[f64]) -> Result<(), HistoryError> {
        if sample.len() != self.names.len() {
            return Err(HistoryError::Width {
                expected: self.names.len(),
                found: sample.len(),
            });
        }
        if self.samples.back().is_some_and(|(t, _)| *t >= t_ns) {
            self.samples.clear();
        }
        if self.samples.len() == self.length {
            self.samples.pop_front();
        }
        self.samples.push_back((t_ns, sample.to_vec()));
        Ok(())
    }

    /// Whether the content is usable at time `now_ns`.
    #[must_use]
    pub fn state(&self, now_ns: i64) -> WindowState {
        if self.samples.len() < self.length {
            return WindowState::Short {
                have: self.samples.len(),
            };
        }
        let gap_bad = self
            .samples
            .iter()
            .zip(self.samples.iter().skip(1))
            .any(|(a, b)| (b.0 - a.0 - self.period_ns).abs() > self.jitter_ns);
        if gap_bad {
            return WindowState::Discontinuous;
        }
        let newest = self.samples.back().map_or(i64::MIN, |(t, _)| *t);
        if now_ns - newest > self.period_ns + self.jitter_ns {
            return WindowState::Stale;
        }
        WindowState::Ready
    }

    /// The window flattened oldest first, if [`WindowState::Ready`].
    #[must_use]
    pub fn flatten(&self, now_ns: i64) -> Option<Vec<f64>> {
        (self.state(now_ns) == WindowState::Ready).then(|| {
            self.samples
                .iter()
                .flat_map(|(_, s)| s.iter().copied())
                .collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window() -> HistoryWindow {
        HistoryWindow::new(vec!["joint.q0".into(), "imu.gyro_z".into()], 3, 10, 2).unwrap()
    }

    #[test]
    fn privileged_observations_are_refused_at_construction() {
        assert_eq!(
            HistoryWindow::new(vec!["privileged.friction".into()], 3, 10, 2).unwrap_err(),
            HistoryError::Privileged("privileged.friction".into())
        );
    }

    #[test]
    fn short_stale_and_discontinuous_windows_are_not_ready() {
        let mut w = window();
        w.push(0, &[1.0, 2.0]).unwrap();
        assert_eq!(w.state(0), WindowState::Short { have: 1 });
        w.push(10, &[3.0, 4.0]).unwrap();
        w.push(20, &[5.0, 6.0]).unwrap();
        assert_eq!(w.state(25), WindowState::Ready);
        assert_eq!(w.flatten(25).unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(w.state(40), WindowState::Stale);
        w.push(45, &[7.0, 8.0]).unwrap();
        assert_eq!(w.state(45), WindowState::Discontinuous);
        assert!(w.flatten(45).is_none());
    }

    #[test]
    fn time_going_backwards_clears_the_window() {
        let mut w = window();
        for t in [0, 10, 20] {
            w.push(t, &[0.0, 0.0]).unwrap();
        }
        w.push(5, &[0.0, 0.0]).unwrap();
        assert_eq!(w.state(5), WindowState::Short { have: 1 });
    }
}

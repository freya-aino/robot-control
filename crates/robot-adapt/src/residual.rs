//! A bounded robot-side residual (SPECS.md Section 11.3, row 4) and the
//! authority bound obligation 4 requires.
//!
//! "Adaptation authority MUST be bounded and the bound enforced outside the
//! adapted component." [`ResidualBound`] is that outside enforcement: the
//! residual network proposes, the bound clamps each dimension to its
//! declared authority and limits how fast the correction may change, and
//! the result is added to the base policy's action. The supervisor of
//! Section 12.1 then clamps the sum like any other command; this bound does
//! not replace it.
//!
//! The row's limitation is that a residual "can destabilize the base policy
//! or hide sensor errors". A residual pinned at its bound is the observable
//! sign of both, so [`ResidualBound`] counts, per dimension, how often the
//! clamp engaged over a sliding window, and [`ResidualBound::saturated`]
//! names every dimension above the caller's fraction.

use std::collections::VecDeque;

/// The authority bound around a learned residual.
#[derive(Clone, PartialEq, Debug)]
pub struct ResidualBound {
    authority: Vec<f64>,
    max_rate: Vec<f64>,
    last: Vec<f64>,
    window: usize,
    history: VecDeque<Vec<bool>>,
}

impl ResidualBound {
    /// Per-dimension authority (absolute bound) and maximum change per
    /// call, with a saturation window of `window` calls.
    ///
    /// # Panics
    ///
    /// Never; widths are checked and a mismatch returns `None`.
    #[must_use]
    pub fn new(authority: Vec<f64>, max_rate: Vec<f64>, window: usize) -> Option<Self> {
        let ok = authority.len() == max_rate.len()
            && window > 0
            && authority
                .iter()
                .chain(&max_rate)
                .all(|v| v.is_finite() && *v >= 0.0);
        ok.then(|| Self {
            last: vec![0.0; authority.len()],
            authority,
            max_rate,
            window,
            history: VecDeque::with_capacity(window),
        })
    }

    /// The command `base + bounded(residual)`. A non-finite residual entry
    /// contributes nothing and counts as saturated.
    pub fn apply(&mut self, base: &[f64], residual: &[f64]) -> Vec<f64> {
        let mut clamped = Vec::with_capacity(base.len());
        let mut hit = Vec::with_capacity(base.len());
        for (i, b) in base.iter().enumerate() {
            let raw = residual.get(i).copied().unwrap_or(0.0);
            let (target, bad) = if raw.is_finite() {
                (raw, false)
            } else {
                (0.0, true)
            };
            let a = self.authority.get(i).copied().unwrap_or(0.0);
            let bounded = target.clamp(-a, a);
            let r = self.max_rate.get(i).copied().unwrap_or(0.0);
            let prev = self.last.get(i).copied().unwrap_or(0.0);
            let limited = bounded.clamp(prev - r, prev + r);
            if let Some(l) = self.last.get_mut(i) {
                *l = limited;
            }
            hit.push(bad || bounded != target || limited != bounded);
            clamped.push(b + limited);
        }
        if self.history.len() == self.window {
            self.history.pop_front();
        }
        self.history.push_back(hit);
        clamped
    }

    /// Dimensions whose clamp engaged on more than `fraction` of the recent
    /// window.
    #[must_use]
    pub fn saturated(&self, fraction: f64) -> Vec<usize> {
        let n = self.history.len().max(1) as f64;
        (0..self.authority.len())
            .filter(|&i| {
                let count = self
                    .history
                    .iter()
                    .filter(|h| h.get(i).copied().unwrap_or(false))
                    .count();
                count as f64 / n > fraction
            })
            .collect()
    }

    /// Resets the rate limiter and history (after a mode change or fault).
    pub fn reset(&mut self) {
        self.last.iter_mut().for_each(|v| *v = 0.0);
        self.history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authority_and_rate_limit_hold_whatever_the_residual_proposes() {
        let mut b = ResidualBound::new(vec![0.1, 0.5], vec![0.05, 1.0], 10).unwrap();
        let out = b.apply(&[1.0, 1.0], &[10.0, -0.2]);
        assert_eq!(out, vec![1.05, 0.8]); // rate-limited from 0, then within authority
        let out = b.apply(&[1.0, 1.0], &[10.0, -0.2]);
        assert_eq!(out, vec![1.1, 0.8]); // reaches the 0.1 authority
        let out = b.apply(&[1.0, 1.0], &[f64::NAN, -0.2]);
        assert!((out[0] - 1.05).abs() < 1e-12); // NaN means zero, approached at the rate limit
    }

    #[test]
    fn a_pinned_dimension_is_reported_as_saturated() {
        let mut b = ResidualBound::new(vec![0.1, 0.1], vec![1.0, 1.0], 20).unwrap();
        for _ in 0..20 {
            b.apply(&[0.0, 0.0], &[5.0, 0.01]);
        }
        assert_eq!(b.saturated(0.5), vec![0]);
        b.reset();
        assert!(b.saturated(0.0).is_empty());
    }

    #[test]
    fn mismatched_widths_are_refused() {
        assert!(ResidualBound::new(vec![0.1], vec![], 5).is_none());
        assert!(ResidualBound::new(vec![-0.1], vec![0.1], 5).is_none());
    }
}

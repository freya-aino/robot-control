//! Simulator correction (SPECS.md Section 11.3, row 3): a model of the
//! mismatch between real and simulated transitions, fitted offline, used
//! to correct the simulator a policy is retrained in.
//!
//! The model is linear in caller-chosen features of `(state, action)`:
//! `delta = W phi(s, a)`, one ridge regression per state dimension, where
//! `delta = s'_real - s'_sim` for a simulated step taken from the *same*
//! state and action the real step started from. That pairing is the
//! "well-defined alignment" the row's limitation demands, and
//! [`AlignedTransition`] makes it the only input the fit accepts. The
//! "validation of correction" is [`CorrectionModel::evaluate`]: held-out
//! next-state error with and without the correction, reported side by
//! side so a correction that makes things worse is visible.

use crate::sysid::{identify, IdentificationError, Sample};

/// One real step and the simulator's prediction from the same start.
#[derive(Clone, PartialEq, Debug)]
pub struct AlignedTransition {
    /// Features `phi(s, a)` of the shared start state and action.
    pub features: Vec<f64>,
    /// The real next state.
    pub real_next: Vec<f64>,
    /// The simulated next state from the same start and action.
    pub sim_next: Vec<f64>,
}

/// A fitted correction.
#[derive(Clone, PartialEq, Debug)]
pub struct CorrectionModel {
    /// One coefficient row per state dimension.
    pub weights: Vec<Vec<f64>>,
}

/// Held-out next-state error with and without correction.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct CorrectionEvaluation {
    /// RMS error of the uncorrected simulator.
    pub baseline_rms: f64,
    /// RMS error after correction.
    pub corrected_rms: f64,
}

impl CorrectionEvaluation {
    /// Whether the correction helped on held-out data.
    #[must_use]
    pub fn improves(&self) -> bool {
        self.corrected_rms < self.baseline_rms
    }
}

impl CorrectionModel {
    /// Fits the correction to `data` with ridge `ridge`.
    pub fn fit(data: &[AlignedTransition], ridge: f64) -> Result<Self, IdentificationError> {
        let dims = data.first().map_or(0, |t| t.real_next.len());
        let mut weights = Vec::with_capacity(dims);
        for d in 0..dims {
            let samples: Vec<Sample> = data
                .iter()
                .map(|t| Sample {
                    features: t.features.clone(),
                    response: t.real_next[d] - t.sim_next[d],
                })
                .collect();
            weights.push(identify(&samples, ridge)?.coefficients);
        }
        Ok(Self { weights })
    }

    /// The corrected next state for a simulated `sim_next`.
    #[must_use]
    pub fn correct(&self, features: &[f64], sim_next: &[f64]) -> Vec<f64> {
        sim_next
            .iter()
            .zip(&self.weights)
            .map(|(s, w)| s + w.iter().zip(features).map(|(a, b)| a * b).sum::<f64>())
            .collect()
    }

    /// Errors on held-out transitions.
    #[must_use]
    pub fn evaluate(&self, held_out: &[AlignedTransition]) -> CorrectionEvaluation {
        let rms = |pairs: &mut dyn Iterator<Item = (Vec<f64>, &Vec<f64>)>| {
            let (mut sum, mut n) = (0.0, 0usize);
            for (pred, real) in pairs {
                for (p, r) in pred.iter().zip(real) {
                    sum += (p - r).powi(2);
                    n += 1;
                }
            }
            if n == 0 {
                0.0
            } else {
                (sum / n as f64).sqrt()
            }
        };
        CorrectionEvaluation {
            baseline_rms: rms(&mut held_out.iter().map(|t| (t.sim_next.clone(), &t.real_next))),
            corrected_rms: rms(&mut held_out
                .iter()
                .map(|t| (self.correct(&t.features, &t.sim_next), &t.real_next))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real: `x' = x + dt (v)`, `v' = v + dt (u - 0.8 v)`; simulator misses
    /// the damping.
    fn transitions(n: usize, phase: f64) -> Vec<AlignedTransition> {
        let dt = 0.01;
        (0..n)
            .map(|i| {
                let t = i as f64 * 0.1 + phase;
                let (x, v, u) = (t.sin(), (1.3 * t).cos(), (0.7 * t).sin());
                AlignedTransition {
                    features: vec![x, v, u, 1.0],
                    real_next: vec![x + dt * v, v + dt * (u - 0.8 * v)],
                    sim_next: vec![x + dt * v, v + dt * u],
                }
            })
            .collect()
    }

    #[test]
    fn a_missing_damping_term_is_learned_and_validated_on_held_out_data() {
        let model = CorrectionModel::fit(&transitions(200, 0.0), 1e-9).unwrap();
        assert!(
            (model.weights[1][1] + 0.008).abs() < 1e-9,
            "{:?}",
            model.weights
        );
        let e = model.evaluate(&transitions(50, 100.0));
        assert!(e.improves());
        assert!(e.corrected_rms < 1e-9 && e.baseline_rms > 1e-3, "{e:?}");
    }

    #[test]
    fn a_correction_that_does_not_generalize_is_reported_as_such() {
        // Fit on data where the mismatch is a constant; evaluate where it
        // has flipped sign: the correction makes things worse.
        let mut fit_data = transitions(50, 0.0);
        for t in &mut fit_data {
            t.real_next[0] = t.sim_next[0] + 0.1;
        }
        let mut held = transitions(50, 10.0);
        for t in &mut held {
            t.real_next[0] = t.sim_next[0] - 0.1;
        }
        let model = CorrectionModel::fit(&fit_data, 1e-6).unwrap();
        assert!(!model.evaluate(&held).improves());
    }
}

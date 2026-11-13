//! Sensor noise models, at the injection point Appendix B S-06 of
//! `crates/robot-sim/specs` defaults to: `robot-sim` produces noise-free
//! readings, and this module perturbs them.
//!
//! `specs/08` section 9 item 2: noise must be "generated at a declared
//! point in the pipeline, recorded as such". A [`NoisyChannel`] carries its
//! model and seed, and [`NoisyChannel::record`] states both with the
//! injection point, so a log of noisy readings says how they were made.
//!
//! The model, per scalar channel, applied in this order:
//!
//! 1. **bias**: a random walk, `b += N(0, walk^2 dt)` per sample, starting
//!    at a fixed initial bias;
//! 2. **white noise**: `N(0, white^2)`;
//! 3. **quantization**: rounding to the nearest multiple of `quantum`;
//! 4. **delay**: the output is the sample from `delay` samples earlier
//!    (the oldest available while the line fills).
//!
//! Section 9.1's simulator qualification lists "sensor delay/noise" among
//! what a humanoid simulator needs; this is that, on the perception side of
//! the boundary.

use std::collections::VecDeque;

use robot_types::rng::{Rng, SeedStream};

/// One channel's noise model.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct NoiseModel {
    /// White-noise standard deviation.
    pub white: f64,
    /// Bias random-walk intensity, units per square-root second.
    pub bias_walk: f64,
    /// Initial bias.
    pub initial_bias: f64,
    /// Quantization step (0: none).
    pub quantum: f64,
    /// Delay in samples.
    pub delay: usize,
}

impl NoiseModel {
    /// No noise at all.
    pub const NONE: Self = Self {
        white: 0.0,
        bias_walk: 0.0,
        initial_bias: 0.0,
        quantum: 0.0,
        delay: 0,
    };
}

/// A vector of identically modelled channels with independent noise.
pub struct NoisyChannel {
    model: NoiseModel,
    seed: SeedStream,
    rng: Rng,
    bias: Vec<f64>,
    line: VecDeque<Vec<f64>>,
}

impl NoisyChannel {
    /// `width` channels under `model`, drawing from `seed`.
    #[must_use]
    pub fn new(width: usize, model: NoiseModel, seed: SeedStream) -> Self {
        Self {
            model,
            seed,
            rng: Rng::new(seed),
            bias: vec![model.initial_bias; width],
            line: VecDeque::with_capacity(model.delay + 1),
        }
    }

    /// Perturbs `truth` (sampled `dt` seconds after the previous call) into
    /// `out`.
    pub fn apply(&mut self, truth: &[f64], dt: f64, out: &mut [f64]) {
        let m = self.model;
        let mut sample = vec![0.0; truth.len()];
        for (i, (s, t)) in sample.iter_mut().zip(truth).enumerate() {
            if let Some(b) = self.bias.get_mut(i) {
                *b += self.rng.normal(0.0, m.bias_walk * dt.max(0.0).sqrt());
                let mut v = t + *b + self.rng.normal(0.0, m.white);
                if m.quantum > 0.0 {
                    v = (v / m.quantum).round() * m.quantum;
                }
                *s = v;
            }
        }
        if self.line.len() == m.delay + 1 {
            self.line.pop_front();
        }
        self.line.push_back(sample);
        let delayed = self.line.front().expect("just pushed");
        out.copy_from_slice(&delayed[..out.len().min(delayed.len())]);
    }

    /// The record that makes a noisy log reconstructible.
    #[must_use]
    pub fn record(&self) -> String {
        let m = self.model;
        format!(
            "noise point: robot-perception::noise after robot-sim::sensors; seed {} stream {}; white {} bias_walk {} initial_bias {} quantum {} delay {}",
            self.seed.seed, self.seed.stream, m.white, m.bias_walk, m.initial_bias, m.quantum, m.delay
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(v: &[f64]) -> (f64, f64) {
        let n = v.len() as f64;
        let mean = v.iter().sum::<f64>() / n;
        (
            mean,
            (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt(),
        )
    }

    #[test]
    fn white_noise_has_its_stated_spread_and_none_is_exact() {
        let mut c = NoisyChannel::new(
            1,
            NoiseModel {
                white: 0.05,
                ..NoiseModel::NONE
            },
            SeedStream::new(1, 0),
        );
        let mut out = [0.0];
        let xs: Vec<f64> = (0..5000)
            .map(|_| {
                c.apply(&[2.0], 0.01, &mut out);
                out[0]
            })
            .collect();
        let (m, s) = stats(&xs);
        assert!(
            (m - 2.0).abs() < 0.005 && (s - 0.05).abs() < 0.003,
            "{m} {s}"
        );
        let mut exact = NoisyChannel::new(2, NoiseModel::NONE, SeedStream::new(1, 0));
        let mut o2 = [0.0; 2];
        exact.apply(&[1.5, -2.0], 0.01, &mut o2);
        assert_eq!(o2, [1.5, -2.0]);
    }

    #[test]
    fn the_bias_walk_grows_like_the_square_root_of_time() {
        // Across many independent channels, the bias spread after T seconds
        // is walk * sqrt(T).
        let model = NoiseModel {
            bias_walk: 0.1,
            ..NoiseModel::NONE
        };
        let mut c = NoisyChannel::new(2000, model, SeedStream::new(2, 0));
        let truth = vec![0.0; 2000];
        let mut out = vec![0.0; 2000];
        for _ in 0..400 {
            c.apply(&truth, 0.01, &mut out);
        }
        let (_, s) = stats(&out);
        assert!((s - 0.1 * 4.0f64.sqrt()).abs() < 0.015, "{s}");
    }

    #[test]
    fn quantization_and_delay_behave_as_stated() {
        let model = NoiseModel {
            quantum: 0.25,
            delay: 2,
            ..NoiseModel::NONE
        };
        let mut c = NoisyChannel::new(1, model, SeedStream::new(3, 0));
        let mut out = [0.0];
        let mut seen = Vec::new();
        for x in [0.1, 0.3, 0.6, 0.9] {
            c.apply(&[x], 0.01, &mut out);
            seen.push(out[0]);
        }
        // Quantized 0.0, 0.25, 0.5, 1.0; delayed two samples behind once full.
        assert_eq!(seen, vec![0.0, 0.0, 0.0, 0.25]);
        assert!(c.record().contains("seed 3"));
    }
}

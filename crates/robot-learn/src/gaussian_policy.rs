//! A diagonal Gaussian policy: a mean network plus a state-independent
//! log-standard-deviation vector.
//!
//! This is the standard continuous-action PPO parameterization. The log-std
//! is state-independent (one vector of `action_width` parameters, not a
//! second network) deliberately: it is the smallest choice that still lets
//! the policy learn how much to explore per action dimension, and it keeps
//! the gradient derivation below a few closed-form lines rather than a
//! second backward pass.

use robot_nn::mlp::{
    frozen_mask_for_count, ForwardTrace, Mlp, MlpError, ParamGroup, ParamGroupError, Scratch,
};
use robot_nn::policy::{Policy, StochasticPolicy};
use robot_types::rng::Rng;

const LOG_STD_MIN: f64 = -5.0;
const LOG_STD_MAX: f64 = 2.0;

/// The policy: a mean network and a log-std vector.
#[derive(Clone, Debug)]
pub struct GaussianPolicy {
    /// The mean network.
    pub mean_net: Mlp,
    /// Log standard deviation, one per action dimension.
    pub log_std: Vec<f64>,
}

/// One sampled action together with the information needed to compute its
/// log-probability and the gradients of the PPO loss with respect to the
/// policy's parameters.
#[derive(Clone, Debug)]
pub struct ActionSample {
    /// The sampled action.
    pub action: Vec<f64>,
    /// Log-probability of `action` under the policy at the time it was
    /// sampled.
    pub log_prob: f64,
    /// The forward trace of the mean network, needed for backpropagation.
    pub trace: ForwardTrace,
}

impl GaussianPolicy {
    /// Builds a policy from a mean network and an initial log-std.
    #[must_use]
    pub fn new(mean_net: Mlp, initial_log_std: f64) -> Self {
        let width = mean_net.output_width();
        Self {
            mean_net,
            log_std: vec![initial_log_std; width],
        }
    }

    /// Rebuilds a policy from its mean network and a per-dimension log-std
    /// vector, as stored in a checkpoint. Returns `None` when the vector's
    /// length does not match the network's output width.
    #[must_use]
    pub fn from_parts(mean_net: Mlp, log_std: Vec<f64>) -> Option<Self> {
        (log_std.len() == mean_net.output_width()).then_some(Self { mean_net, log_std })
    }

    /// Samples an action at `observation`, using `rng` for the Gaussian
    /// noise.
    #[must_use]
    pub fn sample(&self, observation: &[f64], rng: &mut Rng) -> ActionSample {
        let trace = self.mean_net.forward_with_trace(observation);
        let mean = trace.output();
        let action: Vec<f64> = mean
            .iter()
            .zip(&self.log_std)
            .map(|(m, log_s)| m + log_s.exp() * rng.standard_normal())
            .collect();
        let log_prob = gaussian_log_prob(&action, mean, &self.log_std);
        ActionSample {
            action,
            log_prob,
            trace,
        }
    }

    /// Log-probability of `action` under the policy's current mean (from
    /// `trace`) and log-std, used when recomputing the policy's distribution
    /// during a PPO update on data collected by an earlier parameter set.
    #[must_use]
    pub fn log_prob_from_trace(&self, trace: &ForwardTrace, action: &[f64]) -> f64 {
        gaussian_log_prob(action, trace.output(), &self.log_std)
    }

    /// Entropy of the current (state-independent-variance) distribution.
    #[must_use]
    pub fn entropy(&self) -> f64 {
        self.log_std
            .iter()
            .map(|log_s| 0.5 * (core::f64::consts::TAU * core::f64::consts::E).ln() + log_s)
            .sum()
    }

    /// Gradient of `-log_prob` (the negative log-likelihood) with respect to
    /// the mean network's output and the log-std vector, scaled by
    /// `upstream` (the PPO loss's own gradient with respect to `log_prob`).
    ///
    /// Returns `(d_mean, d_log_std)`.
    #[must_use]
    pub fn log_prob_gradient(
        &self,
        trace: &ForwardTrace,
        action: &[f64],
        upstream: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let mean = trace.output();
        let mut d_mean = vec![0.0; mean.len()];
        let mut d_log_std = vec![0.0; self.log_std.len()];
        for i in 0..mean.len() {
            let std = self.log_std[i].exp();
            let diff = action[i] - mean[i];
            // d(log_prob)/d(mean) = diff / std^2
            d_mean[i] = upstream * (diff / (std * std));
            // d(log_prob)/d(log_std) = (diff^2 / std^2) - 1
            d_log_std[i] = upstream * ((diff * diff) / (std * std) - 1.0);
        }
        (d_mean, d_log_std)
    }

    /// Total number of trainable policy parameters in the flat order used by
    /// [`GaussianPolicy::frozen_mask`]: the mean network's parameters first,
    /// then one log-std entry per action dimension.
    #[must_use]
    pub fn param_count(&self) -> usize {
        self.mean_net.param_count() + self.log_std.len()
    }

    /// Checks `groups` against the full policy parameter space (mean network
    /// then log-std) and returns one frozen flag per entry.
    pub fn frozen_mask(&self, groups: &[ParamGroup]) -> Result<Vec<bool>, ParamGroupError> {
        frozen_mask_for_count(self.param_count(), groups)
    }

    /// Applies a gradient step to the log-std vector directly (a plain
    /// gradient descent step; `robot-learn`'s Adam machinery is reserved for
    /// the network parameters, and a single small vector does not need the
    /// same bookkeeping).
    pub fn apply_log_std_gradient(&mut self, grad: &[f64], learning_rate: f64) {
        for (s, g) in self.log_std.iter_mut().zip(grad) {
            *s = (*s - learning_rate * g).clamp(LOG_STD_MIN, LOG_STD_MAX);
        }
    }

    /// Like [`GaussianPolicy::apply_log_std_gradient`], but entries whose
    /// `frozen` flag is set are left untouched, bit for bit. `grad` and
    /// `frozen` must each have one entry per log-std value; otherwise nothing
    /// changes and an error is returned.
    pub fn apply_log_std_gradient_masked(
        &mut self,
        grad: &[f64],
        learning_rate: f64,
        frozen: &[bool],
    ) -> Result<(), &'static str> {
        if grad.len() != self.log_std.len() || frozen.len() != self.log_std.len() {
            return Err("log-std gradient or mask length does not match log-std");
        }
        for ((s, g), skip) in self.log_std.iter_mut().zip(grad).zip(frozen) {
            if !*skip {
                *s = (*s - learning_rate * g).clamp(LOG_STD_MIN, LOG_STD_MAX);
            }
        }
        Ok(())
    }
}

impl Policy for GaussianPolicy {
    fn input_width(&self) -> usize {
        self.mean_net.input_width()
    }

    fn output_width(&self) -> usize {
        self.mean_net.output_width()
    }

    fn scratch_len(&self) -> usize {
        self.mean_net.scratch_len()
    }

    /// The deterministic (mean) action.
    fn act(
        &self,
        observation: &[f64],
        action: &mut [f64],
        scratch: &mut Scratch,
    ) -> Result<(), MlpError> {
        self.mean_net.forward(observation, action, scratch)
    }
}

impl StochasticPolicy for GaussianPolicy {
    /// Allocation-free counterpart of [`GaussianPolicy::sample`]: it draws
    /// the same noise in the same order, so the two agree on any RNG stream.
    fn sample_action(
        &self,
        observation: &[f64],
        rng: &mut Rng,
        action: &mut [f64],
        scratch: &mut Scratch,
    ) -> Result<f64, MlpError> {
        self.mean_net.forward(observation, action, scratch)?;
        let mut log_prob = 0.0;
        for (a, log_s) in action.iter_mut().zip(&self.log_std) {
            let mean = *a;
            let std = log_s.exp();
            *a = mean + std * rng.standard_normal();
            let z = (*a - mean) / std;
            log_prob += -0.5 * z * z - log_s - 0.5 * (core::f64::consts::TAU).ln();
        }
        Ok(log_prob)
    }
}

fn gaussian_log_prob(x: &[f64], mean: &[f64], log_std: &[f64]) -> f64 {
    let mut total = 0.0;
    for ((xi, mi), log_si) in x.iter().zip(mean).zip(log_std) {
        let std = log_si.exp();
        let z = (xi - mi) / std;
        total += -0.5 * z * z - log_si - 0.5 * (core::f64::consts::TAU).ln();
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_nn::layer::{Activation, Linear};
    use robot_types::rng::SeedStream;

    fn policy() -> GaussianPolicy {
        let layer = Linear::new(2, 1, vec![0.5, -0.5], vec![0.1]).unwrap();
        let net = Mlp::new(vec![layer], vec![Activation::Identity]).unwrap();
        GaussianPolicy::new(net, -0.5)
    }

    #[test]
    fn sampled_actions_scatter_around_the_mean() {
        let p = policy();
        let mut rng = Rng::new(SeedStream::new(1, 0));
        let obs = [1.0, 0.0];
        let trace = p.mean_net.forward_with_trace(&obs);
        let mean = trace.output()[0];
        let samples: Vec<f64> = (0..2000)
            .map(|_| p.sample(&obs, &mut rng).action[0])
            .collect();
        let avg = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(
            (avg - mean).abs() < 0.05,
            "sample mean {avg} should track the policy mean {mean}"
        );
    }

    #[test]
    fn sample_action_matches_sample_on_the_same_stream() {
        let p = policy();
        let obs = [1.0, 0.0];
        let mut rng_a = Rng::new(SeedStream::new(9, 2));
        let mut rng_b = Rng::new(SeedStream::new(9, 2));
        let mut scratch = Scratch::new(Policy::scratch_len(&p));
        let mut action = [0.0; 1];
        for _ in 0..50 {
            let reference = p.sample(&obs, &mut rng_a);
            let lp = p
                .sample_action(&obs, &mut rng_b, &mut action, &mut scratch)
                .unwrap();
            assert_eq!(action[0], reference.action[0]);
            assert_eq!(lp, reference.log_prob);
        }
    }

    #[test]
    fn act_is_the_mean_action() {
        let p = policy();
        let obs = [1.0, 0.0];
        let mut scratch = Scratch::new(Policy::scratch_len(&p));
        let mut action = [0.0; 1];
        Policy::act(&p, &obs, &mut action, &mut scratch).unwrap();
        let mean = p.mean_net.forward_with_trace(&obs).output()[0];
        assert_eq!(action[0], mean);
    }

    #[test]
    fn log_prob_gradient_matches_central_finite_differences_on_mean() {
        let p = policy();
        let obs = [1.0, 0.0];
        let trace = p.mean_net.forward_with_trace(&obs);
        let action = [0.3];
        let (d_mean, _) = p.log_prob_gradient(&trace, &action, 1.0);

        let eps = 1e-6;
        let mean = trace.output()[0];
        let lp = |m: f64| gaussian_log_prob(&action, &[m], &p.log_std);
        let numeric = (lp(mean + eps) - lp(mean - eps)) / (2.0 * eps);
        assert!(
            (numeric - d_mean[0]).abs() < 1e-4,
            "numeric {numeric} vs analytic {}",
            d_mean[0]
        );
    }

    #[test]
    fn log_prob_gradient_matches_central_finite_differences_on_log_std() {
        let p = policy();
        let obs = [1.0, 0.0];
        let trace = p.mean_net.forward_with_trace(&obs);
        let action = [0.3];
        let (_, d_log_std) = p.log_prob_gradient(&trace, &action, 1.0);

        let eps = 1e-6;
        let mean = trace.output();
        let lp = |ls: f64| gaussian_log_prob(&action, mean, &[ls]);
        let numeric = (lp(p.log_std[0] + eps) - lp(p.log_std[0] - eps)) / (2.0 * eps);
        assert!(
            (numeric - d_log_std[0]).abs() < 1e-4,
            "numeric {numeric} vs analytic {}",
            d_log_std[0]
        );
    }

    #[test]
    fn entropy_increases_with_log_std() {
        let mut p = policy();
        let low = p.entropy();
        p.log_std[0] += 1.0;
        let high = p.entropy();
        assert!(high > low);
    }

    #[test]
    fn log_std_gradient_step_is_clamped_to_bounds() {
        let mut p = policy();
        p.apply_log_std_gradient(&[1e6], 1.0);
        assert!(p.log_std[0] >= LOG_STD_MIN - 1e-9);
        p.log_std[0] = 0.0;
        p.apply_log_std_gradient(&[-1e6], 1.0);
        assert!(p.log_std[0] <= LOG_STD_MAX + 1e-9);
    }

    #[test]
    fn legacy_log_std_gradient_truncates_mismatched_lengths() {
        let mut p = policy();
        p.log_std[0] = 0.0;
        p.apply_log_std_gradient(&[0.25, 1e6], 0.5);
        assert_eq!(p.log_std[0], -0.125);

        p.apply_log_std_gradient(&[], 0.5);
        assert_eq!(p.log_std[0], -0.125);
    }
}

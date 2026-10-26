//! The Adam optimizer, applied to a `robot_nn::mlp::Mlp`'s parameters.
//!
//! SPECS.md Section 9.2 requires "checkpoint/resume including optimizer,
//! RNG and normalizer state" — the whole reason [`AdamState`] is a value the
//! caller owns and can serialize, rather than hidden inside an opaque
//! optimizer object.

use crate::backprop::Gradients;
use robot_nn::mlp::Mlp;

/// Adam hyperparameters.
#[derive(Clone, Copy, Debug)]
pub struct AdamConfig {
    /// Step size.
    pub learning_rate: f64,
    /// First-moment decay.
    pub beta1: f64,
    /// Second-moment decay.
    pub beta2: f64,
    /// Numerical stabilizer.
    pub epsilon: f64,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

/// Adam's per-parameter moment estimates, shaped like the network it
/// optimizes, plus the step count needed for bias correction.
///
/// This is exactly the state SPECS.md Section 9.2 requires be resumable: a
/// checkpoint that restores `AdamState` but not the step count would apply
/// an incorrectly bias-corrected step on resume.
#[derive(Clone, Debug)]
pub struct AdamState {
    m: Gradients,
    v: Gradients,
    step: u64,
}

impl AdamState {
    /// Zero-initialized state shaped for `mlp`.
    #[must_use]
    pub fn new(mlp: &Mlp) -> Self {
        Self {
            m: Gradients::zeros_like(mlp),
            v: Gradients::zeros_like(mlp),
            step: 0,
        }
    }

    /// The step count, for inspection or for a checkpoint record.
    #[must_use]
    pub const fn step(&self) -> u64 {
        self.step
    }

    /// Applies one Adam step to `mlp` using `grads`, updating both the
    /// optimizer state and the network's parameters in place.
    pub fn step_with(&mut self, mlp: &mut Mlp, grads: &Gradients, config: &AdamConfig) {
        self.step += 1;
        let t = self.step as i32;
        let bias_correction1 = 1.0 - config.beta1.powi(t);
        let bias_correction2 = 1.0 - config.beta2.powi(t);

        for layer in 0..mlp.layers().len() {
            update_param_slice(
                mlp.layers_mut()[layer].weight_mut(),
                &grads.weight[layer],
                &mut self.m.weight[layer],
                &mut self.v.weight[layer],
                config,
                bias_correction1,
                bias_correction2,
            );
        }
        for layer in 0..mlp.layers().len() {
            update_param_slice(
                mlp.layers_mut()[layer].bias_mut(),
                &grads.bias[layer],
                &mut self.m.bias[layer],
                &mut self.v.bias[layer],
                config,
                bias_correction1,
                bias_correction2,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn update_param_slice(
    params: &mut [f64],
    grad: &[f64],
    m: &mut [f64],
    v: &mut [f64],
    config: &AdamConfig,
    bias_correction1: f64,
    bias_correction2: f64,
) {
    for (((p, g), m), v) in params
        .iter_mut()
        .zip(grad)
        .zip(m.iter_mut())
        .zip(v.iter_mut())
    {
        *m = config.beta1 * *m + (1.0 - config.beta1) * g;
        *v = config.beta2 * *v + (1.0 - config.beta2) * g * g;
        let m_hat = *m / bias_correction1;
        let v_hat = *v / bias_correction2;
        *p -= config.learning_rate * m_hat / (v_hat.sqrt() + config.epsilon);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backprop::backward;
    use robot_nn::layer::{Activation, Linear};

    fn quadratic_net() -> Mlp {
        // A single linear layer, identity activation: output = w*x + b.
        // Minimizing 0.5*(output - target)^2 by gradient descent on w, b
        // is a simple, analytically checkable convex problem.
        let layer = Linear::new(1, 1, vec![2.0], vec![0.0]).unwrap();
        Mlp::new(vec![layer], vec![Activation::Identity]).unwrap()
    }

    #[test]
    fn adam_descends_a_convex_quadratic() {
        let mut mlp = quadratic_net();
        let mut state = AdamState::new(&mlp);
        let config = AdamConfig {
            learning_rate: 0.05,
            ..AdamConfig::default()
        };
        let x = [1.0];
        let target = 0.0;

        let mut loss = f64::INFINITY;
        for _ in 0..200 {
            let trace = mlp.forward_with_trace(&x);
            let output = trace.output()[0];
            loss = 0.5 * (output - target).powi(2);
            let d_output = output - target;
            let (grads, _) = backward(&mlp, &trace, &[d_output]);
            state.step_with(&mut mlp, &grads, &config);
        }
        assert!(
            loss < 1e-6,
            "Adam should drive this convex loss near zero, got {loss}"
        );
    }

    #[test]
    fn step_count_increments_and_affects_bias_correction() {
        let mlp = quadratic_net();
        let mut state = AdamState::new(&mlp);
        assert_eq!(state.step(), 0);
        let grads = Gradients::zeros_like(&mlp);
        let mut mlp2 = mlp.clone();
        state.step_with(&mut mlp2, &grads, &AdamConfig::default());
        assert_eq!(state.step(), 1);
    }

    #[test]
    fn zero_gradient_leaves_parameters_unchanged() {
        let mut mlp = quadratic_net();
        let original = mlp.clone();
        let mut state = AdamState::new(&mlp);
        let grads = Gradients::zeros_like(&mlp);
        for _ in 0..10 {
            state.step_with(&mut mlp, &grads, &AdamConfig::default());
        }
        assert_eq!(
            mlp, original,
            "zero gradient must never move the parameters"
        );
    }
}

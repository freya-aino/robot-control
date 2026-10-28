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

    /// First-moment estimates, for checkpointing.
    #[must_use]
    pub const fn first_moment(&self) -> &Gradients {
        &self.m
    }

    /// Second-moment estimates, for checkpointing.
    #[must_use]
    pub const fn second_moment(&self) -> &Gradients {
        &self.v
    }

    /// Applies one Adam step to `mlp` using `grads`, updating both the
    /// optimizer state and the network's parameters in place.
    pub fn step_with(&mut self, mlp: &mut Mlp, grads: &Gradients, config: &AdamConfig) {
        self.apply_step(mlp, grads, config, &[]);
    }

    /// Like [`AdamState::step_with`], but parameters flagged in `frozen` are
    /// left exactly as they are: their value, first moment and second moment
    /// do not change, so no momentum from earlier steps can move them.
    ///
    /// `frozen` uses the flat order of `robot_nn::mlp::Mlp::frozen_mask` and
    /// must have one entry per parameter.
    pub fn step_masked(
        &mut self,
        mlp: &mut Mlp,
        grads: &Gradients,
        config: &AdamConfig,
        frozen: &[bool],
    ) -> Result<(), &'static str> {
        if frozen.len() != mlp.param_count() {
            return Err("frozen mask length does not match the network");
        }
        self.apply_step(mlp, grads, config, frozen);
        Ok(())
    }

    /// `frozen` is either empty (nothing frozen) or one flag per parameter.
    fn apply_step(
        &mut self,
        mlp: &mut Mlp,
        grads: &Gradients,
        config: &AdamConfig,
        frozen: &[bool],
    ) {
        self.step += 1;
        let t = self.step as i32;
        let bias_correction1 = 1.0 - config.beta1.powi(t);
        let bias_correction2 = 1.0 - config.beta2.powi(t);

        let mut offset = 0;
        for layer in 0..mlp.layers().len() {
            let len = self.m.weight[layer].len();
            update_param_slice(
                mlp.layers_mut()[layer].weight_mut(),
                &grads.weight[layer],
                &mut self.m.weight[layer],
                &mut self.v.weight[layer],
                frozen.get(offset..offset + len).unwrap_or(&[]),
                config,
                bias_correction1,
                bias_correction2,
            );
            offset += len;

            let len = self.m.bias[layer].len();
            update_param_slice(
                mlp.layers_mut()[layer].bias_mut(),
                &grads.bias[layer],
                &mut self.m.bias[layer],
                &mut self.v.bias[layer],
                frozen.get(offset..offset + len).unwrap_or(&[]),
                config,
                bias_correction1,
                bias_correction2,
            );
            offset += len;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn update_param_slice(
    params: &mut [f64],
    grad: &[f64],
    m: &mut [f64],
    v: &mut [f64],
    frozen: &[bool],
    config: &AdamConfig,
    bias_correction1: f64,
    bias_correction2: f64,
) {
    for (i, (((p, g), m), v)) in params
        .iter_mut()
        .zip(grad)
        .zip(m.iter_mut())
        .zip(v.iter_mut())
        .enumerate()
    {
        if frozen.get(i).copied().unwrap_or(false) {
            continue;
        }
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

    fn unit_grads(mlp: &Mlp, weight: f64, bias: f64) -> Gradients {
        let mut grads = Gradients::zeros_like(mlp);
        grads.weight[0][0] = weight;
        grads.bias[0][0] = bias;
        grads
    }

    #[test]
    fn frozen_weight_stays_bit_identical_after_momentum_has_built_up() {
        let mut mlp = quadratic_net();
        let mut state = AdamState::new(&mlp);
        let config = AdamConfig {
            learning_rate: 0.05,
            ..AdamConfig::default()
        };
        for _ in 0..5 {
            let grads = unit_grads(&mlp, 1.0, 1.0);
            state.step_with(&mut mlp, &grads, &config);
        }
        let weight_before = mlp.layers()[0].weight()[0];
        let m_before = state.first_moment().weight[0][0];
        let v_before = state.second_moment().weight[0][0];
        let bias_before = mlp.layers()[0].bias()[0];
        assert!(
            m_before != 0.0 && v_before != 0.0,
            "moments must be nonzero"
        );

        let frozen = [true, false];
        let grads = unit_grads(&mlp, 5.0, 5.0);
        state
            .step_masked(&mut mlp, &grads, &config, &frozen)
            .unwrap();

        assert_eq!(
            mlp.layers()[0].weight()[0].to_bits(),
            weight_before.to_bits()
        );
        assert_eq!(
            state.first_moment().weight[0][0].to_bits(),
            m_before.to_bits()
        );
        assert_eq!(
            state.second_moment().weight[0][0].to_bits(),
            v_before.to_bits()
        );
        assert_ne!(mlp.layers()[0].bias()[0], bias_before);
    }

    #[test]
    fn frozen_bias_stays_bit_identical_while_weight_moves() {
        let mut mlp = quadratic_net();
        let mut state = AdamState::new(&mlp);
        let config = AdamConfig {
            learning_rate: 0.05,
            ..AdamConfig::default()
        };
        for _ in 0..5 {
            let grads = unit_grads(&mlp, 1.0, 1.0);
            state.step_with(&mut mlp, &grads, &config);
        }
        let weight_before = mlp.layers()[0].weight()[0];
        let bias_before = mlp.layers()[0].bias()[0];
        let bias_m_before = state.first_moment().bias[0][0];

        let grads = unit_grads(&mlp, 5.0, 5.0);
        state
            .step_masked(&mut mlp, &grads, &config, &[false, true])
            .unwrap();

        assert_eq!(mlp.layers()[0].bias()[0].to_bits(), bias_before.to_bits());
        assert_eq!(
            state.first_moment().bias[0][0].to_bits(),
            bias_m_before.to_bits()
        );
        assert_ne!(mlp.layers()[0].weight()[0], weight_before);
    }

    #[test]
    fn mask_length_mismatch_is_rejected_without_touching_state() {
        let mut mlp = quadratic_net();
        let original = mlp.clone();
        let mut state = AdamState::new(&mlp);
        let config = AdamConfig::default();
        let grads = unit_grads(&mlp, 1.0, 1.0);

        assert!(state
            .step_masked(&mut mlp, &grads, &config, &[true])
            .is_err());
        assert!(state
            .step_masked(&mut mlp, &grads, &config, &[false, false, false])
            .is_err());
        assert_eq!(state.step(), 0);
        assert_eq!(mlp, original);
    }

    #[test]
    fn gradient_mask_length_mismatch_is_rejected() {
        let mlp = quadratic_net();
        let mut grads = unit_grads(&mlp, 1.0, 1.0);
        assert!(grads.zero_frozen(&[true]).is_err());
        assert!(grads.zero_frozen(&[true, false, false]).is_err());
        assert_eq!(grads.weight[0][0], 1.0);
        assert_eq!(grads.bias[0][0], 1.0);
    }
}

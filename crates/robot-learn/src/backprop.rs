//! Manual backpropagation through a `robot_nn::mlp::Mlp`.
//!
//! SPECS.md Section 6 places "autodiff" inside `robot-learn`; this is the
//! authored form of it, per
//! `docs/decisions/0003-std-only-default-closure.md`. It is not a general
//! autodiff engine — it is the closed-form backward pass for exactly the
//! operator set `robot-nn` defines (dense layers, elementwise activations),
//! which is the standard sequential-network chain rule and needs no
//! computation graph: `robot_nn::mlp::Mlp::forward_with_trace` already
//! records every intermediate value this needs.

use robot_nn::layer::Linear;
use robot_nn::mlp::{ForwardTrace, Mlp};

/// Gradients for one network, in the same shape as its parameters.
#[derive(Clone, Debug)]
pub struct Gradients {
    /// Per-layer weight gradients, row-major, same shape as the layer's own
    /// weight matrix.
    pub weight: Vec<Vec<f64>>,
    /// Per-layer bias gradients.
    pub bias: Vec<Vec<f64>>,
}

impl Gradients {
    /// Zero gradients shaped for `mlp`.
    #[must_use]
    pub fn zeros_like(mlp: &Mlp) -> Self {
        Self {
            weight: mlp
                .layers()
                .iter()
                .map(|l| vec![0.0; l.weight().len()])
                .collect(),
            bias: mlp
                .layers()
                .iter()
                .map(|l| vec![0.0; l.bias().len()])
                .collect(),
        }
    }

    /// Accumulates `other` into `self` in place (for minibatch summation).
    pub fn add_assign(&mut self, other: &Self) {
        for (a, b) in self.weight.iter_mut().zip(&other.weight) {
            for (x, y) in a.iter_mut().zip(b) {
                *x += y;
            }
        }
        for (a, b) in self.bias.iter_mut().zip(&other.bias) {
            for (x, y) in a.iter_mut().zip(b) {
                *x += y;
            }
        }
    }

    /// Scales every entry by `factor` in place (e.g. `1/batch_size`).
    pub fn scale(&mut self, factor: f64) {
        for v in self.weight.iter_mut().flatten() {
            *v *= factor;
        }
        for v in self.bias.iter_mut().flatten() {
            *v *= factor;
        }
    }
}

/// Backpropagates a loss gradient `d_output` (dL/d(network output)) through
/// `mlp` given the forward `trace`, returning the gradient with respect to
/// every weight and bias, and (optionally useful to a caller chaining
/// networks) the gradient with respect to the network's input.
///
/// This is a training-time path; it allocates freely, unlike
/// `robot_nn::mlp::Mlp::forward`.
#[must_use]
pub fn backward(mlp: &Mlp, trace: &ForwardTrace, d_output: &[f64]) -> (Gradients, Vec<f64>) {
    let mut grads = Gradients::zeros_like(mlp);
    let mut d_activation_out = d_output.to_vec();

    for layer_index in (0..mlp.layers().len()).rev() {
        let layer: &Linear = &mlp.layers()[layer_index];
        let activation = mlp.activations()[layer_index];
        let z = &trace.pre_activations[layer_index];
        let a = &trace.post_activations[layer_index];
        let input = if layer_index == 0 {
            &trace.input
        } else {
            &trace.post_activations[layer_index - 1]
        };

        // d(loss)/d(pre-activation) = d(loss)/d(post-activation) * activation'(z).
        let d_pre: Vec<f64> = d_activation_out
            .iter()
            .zip(z)
            .zip(a)
            .map(|((d, &zi), &ai)| d * activation.derivative_from_output(zi, ai))
            .collect();

        // d(loss)/d(weight[o][i]) = d_pre[o] * input[i]; d(loss)/d(bias[o]) = d_pre[o].
        let in_features = layer.in_features();
        for (o, &d_pre_o) in d_pre.iter().enumerate() {
            grads.bias[layer_index][o] += d_pre_o;
            let row = &mut grads.weight[layer_index][o * in_features..(o + 1) * in_features];
            for (w, &x) in row.iter_mut().zip(input) {
                *w += d_pre_o * x;
            }
        }

        // d(loss)/d(input) = W^T d_pre, propagated to the previous layer.
        let mut d_input = vec![0.0; in_features];
        for (o, &d_pre_o) in d_pre.iter().enumerate() {
            let w_row = &layer.weight()[o * in_features..(o + 1) * in_features];
            for (i, w) in w_row.iter().enumerate() {
                d_input[i] += w * d_pre_o;
            }
        }
        d_activation_out = d_input;
    }

    (grads, d_activation_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_nn::layer::Activation;
    use robot_nn::mlp::Scratch;

    fn net() -> Mlp {
        let l0 = Linear::new(
            2,
            3,
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![0.1, 0.0, -0.1],
        )
        .unwrap();
        let l1 = Linear::new(3, 1, vec![0.5, -0.5, 1.0], vec![0.0]).unwrap();
        Mlp::new(vec![l0, l1], vec![Activation::Tanh, Activation::Identity]).unwrap()
    }

    /// Computes the scalar loss `L = 0.5 * output[0]^2` and its gradient via
    /// this module's `backward`, for a one-dimensional output.
    fn loss_and_grad(mlp: &Mlp, input: &[f64]) -> (f64, Gradients) {
        let trace = mlp.forward_with_trace(input);
        let output = trace.output()[0];
        let loss = 0.5 * output * output;
        let (grads, _) = backward(mlp, &trace, &[output]);
        (loss, grads)
    }

    #[test]
    fn weight_gradient_matches_central_finite_differences() {
        let mlp = net();
        let input = [0.3, -0.7];
        let (_, analytic) = loss_and_grad(&mlp, &input);

        let eps = 1e-6;
        for layer in 0..mlp.layers().len() {
            for idx in 0..mlp.layers()[layer].weight().len() {
                let mut plus = mlp.clone();
                plus.layers_mut()[layer].weight_mut()[idx] += eps;
                let (loss_plus, _) = loss_and_grad(&plus, &input);

                let mut minus = mlp.clone();
                minus.layers_mut()[layer].weight_mut()[idx] -= eps;
                let (loss_minus, _) = loss_and_grad(&minus, &input);

                let numeric = (loss_plus - loss_minus) / (2.0 * eps);
                let got = analytic.weight[layer][idx];
                assert!(
                    (numeric - got).abs() < 1e-4,
                    "layer {layer} weight {idx}: numeric {numeric} vs analytic {got}"
                );
            }
        }
    }

    #[test]
    fn bias_gradient_matches_central_finite_differences() {
        let mlp = net();
        let input = [0.3, -0.7];
        let (_, analytic) = loss_and_grad(&mlp, &input);
        let eps = 1e-6;
        for layer in 0..mlp.layers().len() {
            for idx in 0..mlp.layers()[layer].bias().len() {
                let mut plus = mlp.clone();
                plus.layers_mut()[layer].bias_mut()[idx] += eps;
                let (loss_plus, _) = loss_and_grad(&plus, &input);

                let mut minus = mlp.clone();
                minus.layers_mut()[layer].bias_mut()[idx] -= eps;
                let (loss_minus, _) = loss_and_grad(&minus, &input);

                let numeric = (loss_plus - loss_minus) / (2.0 * eps);
                let got = analytic.bias[layer][idx];
                assert!((numeric - got).abs() < 1e-4, "layer {layer} bias {idx}");
            }
        }
    }

    #[test]
    fn input_gradient_matches_central_finite_differences() {
        let mlp = net();
        let base_input = [0.3, -0.7];
        let eps = 1e-6;
        for i in 0..2 {
            let mut plus = base_input;
            plus[i] += eps;
            let trace_plus = mlp.forward_with_trace(&plus);
            let loss_plus = 0.5 * trace_plus.output()[0].powi(2);

            let mut minus = base_input;
            minus[i] -= eps;
            let trace_minus = mlp.forward_with_trace(&minus);
            let loss_minus = 0.5 * trace_minus.output()[0].powi(2);

            let trace = mlp.forward_with_trace(&base_input);
            let (_, d_input) = backward(&mlp, &trace, &[trace.output()[0]]);
            let numeric = (loss_plus - loss_minus) / (2.0 * eps);
            assert!(
                (numeric - d_input[i]).abs() < 1e-4,
                "input {i}: numeric {numeric} vs {}",
                d_input[i]
            );
        }
    }

    #[test]
    fn backward_trace_matches_plain_forward() {
        let mlp = net();
        let input = [0.1, 0.2];
        let trace = mlp.forward_with_trace(&input);
        let mut scratch = Scratch::new(mlp.scratch_len());
        let mut out = [0.0];
        mlp.forward(&input, &mut out, &mut scratch).unwrap();
        assert!((trace.output()[0] - out[0]).abs() < 1e-12);
    }

    #[test]
    fn gradients_accumulate_and_scale_correctly() {
        let mlp = net();
        let mut total = Gradients::zeros_like(&mlp);
        for input in [[0.1, 0.2], [0.3, -0.1]] {
            let trace = mlp.forward_with_trace(&input);
            let (g, _) = backward(&mlp, &trace, &[1.0]);
            total.add_assign(&g);
        }
        total.scale(0.5);
        // Spot check: the averaged bias gradient for the last layer should
        // equal the mean of the two individual bias gradients.
        let trace0 = mlp.forward_with_trace(&[0.1, 0.2]);
        let (g0, _) = backward(&mlp, &trace0, &[1.0]);
        let trace1 = mlp.forward_with_trace(&[0.3, -0.1]);
        let (g1, _) = backward(&mlp, &trace1, &[1.0]);
        let expected = (g0.bias[1][0] + g1.bias[1][0]) / 2.0;
        assert!((total.bias[1][0] - expected).abs() < 1e-12);
    }
}

//! Weight initialization.
//!
//! Scaled-normal, fan-in-dependent initialization (a restricted form of the
//! common "Xavier/He" family) is small enough to belong next to the operator
//! definitions it initializes, keeping `robot-learn` free to own the
//! optimizer and update rule without also owning this. The initializer takes
//! a [`robot_types::rng::Rng`] explicitly rather than seeding its own, so a
//! training run's full seed stream — network init included — is
//! reproducible from one recorded [`robot_types::rng::SeedStream`].

use crate::layer::{Activation, Linear};
use crate::mlp::{Mlp, MlpError};
use robot_types::rng::Rng;

/// Builds an [`Mlp`] of the given layer widths, initializing each layer's
/// weights from a zero-mean normal distribution scaled by `1/sqrt(fan_in)`
/// and zeroing every bias.
///
/// `widths` has one more entry than there are layers: `widths[0]` is the
/// input width and `widths[widths.len()-1]` is the output width. `activations`
/// has `widths.len() - 1` entries, one per layer.
pub fn initialize(
    widths: &[usize],
    activations: Vec<Activation>,
    rng: &mut Rng,
) -> Result<Mlp, MlpError> {
    if widths.len() < 2 {
        return Err(MlpError::Empty);
    }
    let mut layers = Vec::with_capacity(widths.len() - 1);
    for pair in widths.windows(2) {
        let (in_features, out_features) = (pair[0], pair[1]);
        let scale = 1.0 / (in_features.max(1) as f64).sqrt();
        let weight: Vec<f64> = (0..in_features * out_features)
            .map(|_| rng.normal(0.0, scale))
            .collect();
        let bias = vec![0.0; out_features];
        layers.push(Linear::new(in_features, out_features, weight, bias)?);
    }
    Mlp::new(layers, activations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_types::rng::SeedStream;

    #[test]
    fn initialized_network_has_the_requested_shape() {
        let mut rng = Rng::new(SeedStream::new(1, 0));
        let net = initialize(
            &[4, 8, 2],
            vec![Activation::Tanh, Activation::Identity],
            &mut rng,
        )
        .unwrap();
        assert_eq!(net.input_width(), 4);
        assert_eq!(net.output_width(), 2);
        assert_eq!(net.depth(), 2);
    }

    #[test]
    fn same_seed_stream_reproduces_identical_weights() {
        let mut rng_a = Rng::new(SeedStream::new(7, 3));
        let mut rng_b = Rng::new(SeedStream::new(7, 3));
        let a = initialize(
            &[3, 5, 1],
            vec![Activation::Tanh, Activation::Identity],
            &mut rng_a,
        )
        .unwrap();
        let b = initialize(
            &[3, 5, 1],
            vec![Activation::Tanh, Activation::Identity],
            &mut rng_b,
        )
        .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn weights_are_not_degenerately_zero() {
        let mut rng = Rng::new(SeedStream::new(5, 0));
        let net = initialize(&[10, 10], vec![Activation::Identity], &mut rng).unwrap();
        let nonzero = net.layers()[0]
            .weight()
            .iter()
            .filter(|w| w.abs() > 1e-6)
            .count();
        assert!(
            nonzero > 50,
            "expected most of 100 weights to be nonzero, got {nonzero}"
        );
    }

    #[test]
    fn too_few_widths_is_rejected() {
        let mut rng = Rng::new(SeedStream::new(1, 0));
        assert!(matches!(
            initialize(&[4], vec![], &mut rng),
            Err(MlpError::Empty)
        ));
    }
}

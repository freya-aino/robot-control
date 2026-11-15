//! Context encoders for history-conditioned adaptation and the RMA-class
//! two-phase latent identification (SPECS.md Section 11.3, rows 2 and 6).
//!
//! Both rows deploy the same thing: a small network that maps a window of
//! deployable observations to a context vector the policy reads. Row 2
//! trains it to regress the randomized dynamics parameters; row 6's phase 2
//! trains it to regress the latent a phase-1 privileged policy was
//! conditioned on. The deployed half is [`ContextEncoder::context`]:
//!
//! - when the window is not [`WindowState::Ready`], it returns the
//!   **declared default context** and says why (obligation 3);
//! - when the window is ready, it encodes it and counts the inputs outside
//!   the range the encoder was trained over. Row 6's limitation is that a
//!   regressed latent "fails silently outside" its randomization ranges;
//!   this count is what makes that failure loud.
//!
//! The fitting half, [`fit`], needs `robot-learn` and sits behind the
//! default `training` feature, so a deployable build can turn it off and
//! keep Section 6's "deployment does not depend on training autograd code".

use robot_nn::mlp::Mlp;

use crate::history::{HistoryWindow, WindowState};

/// Per-input bounds seen during training.
#[derive(Clone, PartialEq, Debug)]
pub struct Envelope {
    /// Minimum per input.
    pub min: Vec<f64>,
    /// Maximum per input.
    pub max: Vec<f64>,
}

impl Envelope {
    /// The bounds of `inputs`.
    #[must_use]
    pub fn of(inputs: &[Vec<f64>]) -> Self {
        let width = inputs.first().map_or(0, Vec::len);
        let mut min = vec![f64::INFINITY; width];
        let mut max = vec![f64::NEG_INFINITY; width];
        for x in inputs {
            for (i, v) in x.iter().enumerate() {
                min[i] = min[i].min(*v);
                max[i] = max[i].max(*v);
            }
        }
        Self { min, max }
    }

    /// Inputs of `x` outside the bounds.
    #[must_use]
    pub fn outside(&self, x: &[f64]) -> usize {
        x.iter()
            .zip(self.min.iter().zip(&self.max))
            .filter(|(v, (lo, hi))| *v < lo || *v > hi)
            .count()
    }
}

/// Where a context came from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ContextSource {
    /// The encoder, with this many inputs outside its training envelope.
    Encoded {
        /// Inputs outside the envelope.
        out_of_range: usize,
    },
    /// The declared default, because the window was not ready.
    Default(WindowState),
}

/// A deployable context encoder.
#[derive(Clone, PartialEq, Debug)]
pub struct ContextEncoder {
    /// The network.
    pub network: Mlp,
    /// The declared default context.
    pub default_context: Vec<f64>,
    /// The training-input envelope.
    pub envelope: Envelope,
}

impl ContextEncoder {
    /// The context for `window` at `now_ns`.
    #[must_use]
    pub fn context(&self, window: &HistoryWindow, now_ns: i64) -> (Vec<f64>, ContextSource) {
        match window.flatten(now_ns) {
            Some(x) if x.len() == self.network.input_width() => {
                let out_of_range = self.envelope.outside(&x);
                let y = self
                    .network
                    .forward_with_trace(&x)
                    .expect("width checked by the match arm")
                    .output()
                    .to_vec();
                (y, ContextSource::Encoded { out_of_range })
            }
            _ => (
                self.default_context.clone(),
                ContextSource::Default(window.state(now_ns)),
            ),
        }
    }
}

/// Training settings.
#[cfg(feature = "training")]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct FitConfig {
    /// Passes over the data.
    pub epochs: usize,
    /// Minibatch size.
    pub batch: usize,
    /// Adam learning rate.
    pub learning_rate: f64,
    /// Seed for initialization and shuffling.
    pub seed: u64,
}

/// Mean squared error of `network` over `data`.
#[must_use]
pub fn mse(network: &Mlp, data: &[(Vec<f64>, Vec<f64>)]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let total: f64 = data
        .iter()
        .map(|(x, y)| {
            let out = network
                .forward_with_trace(x)
                .expect("training pairs share the network's input width");
            out.output()
                .iter()
                .zip(y)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                / y.len().max(1) as f64
        })
        .sum();
    total / data.len() as f64
}

/// Trains `network` on `(input, target)` pairs by minibatch Adam on mean
/// squared error, leaving parameters flagged in `frozen` (flat order of
/// `Mlp::frozen_mask`, empty for none) unchanged.
#[cfg(feature = "training")]
pub fn train(
    network: &mut Mlp,
    data: &[(Vec<f64>, Vec<f64>)],
    config: &FitConfig,
    frozen: &[bool],
) -> Result<(), &'static str> {
    use robot_learn::backprop::{backward, Gradients};
    use robot_learn::optimizer::{AdamConfig, AdamState};
    use robot_types::rng::{Rng, SeedStream};

    if data.is_empty() || config.batch == 0 {
        return Err("no data or zero batch");
    }
    let adam = AdamConfig {
        learning_rate: config.learning_rate,
        ..AdamConfig::default()
    };
    let mut state = AdamState::new(network);
    let mut rng = Rng::new(SeedStream::new(config.seed, 1));
    let mut order: Vec<usize> = (0..data.len()).collect();
    for _ in 0..config.epochs {
        rng.shuffle(&mut order);
        for chunk in order.chunks(config.batch) {
            let mut grads = Gradients::zeros_like(network);
            for &i in chunk {
                let (x, y) = &data[i];
                let trace = network
                    .forward_with_trace(x)
                    .expect("training pairs share the network's input width");
                let n = y.len().max(1) as f64;
                let d: Vec<f64> = trace
                    .output()
                    .iter()
                    .zip(y)
                    .map(|(a, b)| 2.0 * (a - b) / n)
                    .collect();
                let (g, _) = backward(network, &trace, &d);
                grads.add_assign(&g);
            }
            grads.scale(1.0 / chunk.len() as f64);
            if frozen.is_empty() {
                state.step_with(network, &grads, &adam);
            } else {
                state.step_masked(network, &grads, &adam, frozen)?;
            }
        }
    }
    Ok(())
}

/// Fits a fresh encoder of the given hidden widths (tanh hidden layers,
/// linear output) to `(window, target)` pairs, recording the training
/// envelope.
#[cfg(feature = "training")]
pub fn fit(
    data: &[(Vec<f64>, Vec<f64>)],
    hidden: &[usize],
    default_context: Vec<f64>,
    config: &FitConfig,
) -> Result<ContextEncoder, &'static str> {
    use robot_nn::init::initialize;
    use robot_nn::layer::Activation;
    use robot_types::rng::{Rng, SeedStream};

    let (inw, outw) = match data.first() {
        Some((x, y)) => (x.len(), y.len()),
        None => return Err("no data"),
    };
    if default_context.len() != outw {
        return Err("default context width differs from the target width");
    }
    let mut widths = vec![inw];
    widths.extend_from_slice(hidden);
    widths.push(outw);
    let mut acts = vec![Activation::Tanh; hidden.len()];
    acts.push(Activation::Identity);
    let mut rng = Rng::new(SeedStream::new(config.seed, 0));
    let mut network = initialize(&widths, acts, &mut rng).map_err(|_| "invalid widths")?;
    train(&mut network, data, config, &[])?;
    let inputs: Vec<Vec<f64>> = data.iter().map(|(x, _)| x.clone()).collect();
    Ok(ContextEncoder {
        network,
        default_context,
        envelope: Envelope::of(&inputs),
    })
}

#[cfg(all(test, feature = "training"))]
mod tests {
    use super::*;
    use crate::history::HistoryWindow;

    /// A first-order joint `qd' = -b qd + u` with unknown damping `b`:
    /// the window holds (qd, u) pairs, the target is `b`.
    fn episode(b: f64, start: f64) -> Vec<[f64; 2]> {
        let dt = 0.02;
        let mut qd = start;
        (0..6)
            .map(|k| {
                let u = (k as f64 * 0.9).sin();
                let s = [qd, u];
                qd += dt * (-b * qd + u);
                s
            })
            .collect()
    }

    #[test]
    fn an_encoder_learns_damping_from_history_and_flags_extrapolation() {
        let mut data = Vec::new();
        for i in 0..200 {
            let b = 0.5 + 2.5 * (i % 20) as f64 / 19.0;
            let start = 1.0 + 0.5 * ((i / 20) as f64 / 9.0);
            let w: Vec<f64> = episode(b, start).iter().flatten().copied().collect();
            data.push((w, vec![b]));
        }
        let config = FitConfig {
            epochs: 400,
            batch: 20,
            learning_rate: 3e-3,
            seed: 7,
        };
        let encoder = fit(&data, &[16], vec![1.75], &config).unwrap();
        assert!(
            mse(&encoder.network, &data) < 0.05,
            "{}",
            mse(&encoder.network, &data)
        );

        let mut window =
            HistoryWindow::new(vec!["qd".into(), "u".into()], 6, 20_000_000, 1_000_000).unwrap();
        for (k, s) in episode(2.0, 1.2).iter().enumerate() {
            window.push(k as i64 * 20_000_000, s).unwrap();
        }
        let (c, src) = encoder.context(&window, 100_000_000);
        assert!((c[0] - 2.0).abs() < 0.4, "{c:?}");
        assert_eq!(src, ContextSource::Encoded { out_of_range: 0 });

        // A start velocity far outside training: encoded, but flagged.
        let mut far =
            HistoryWindow::new(vec!["qd".into(), "u".into()], 6, 20_000_000, 1_000_000).unwrap();
        for (k, s) in episode(2.0, 10.0).iter().enumerate() {
            far.push(k as i64 * 20_000_000, s).unwrap();
        }
        let (_, src) = encoder.context(&far, 100_000_000);
        assert!(matches!(src, ContextSource::Encoded { out_of_range } if out_of_range > 0));

        // A short window: the declared default, not an extrapolation.
        let short =
            HistoryWindow::new(vec!["qd".into(), "u".into()], 6, 20_000_000, 1_000_000).unwrap();
        let (c, src) = encoder.context(&short, 0);
        assert_eq!(c, vec![1.75]);
        assert_eq!(src, ContextSource::Default(WindowState::Short { have: 0 }));
    }
}

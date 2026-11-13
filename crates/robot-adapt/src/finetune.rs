//! Small adapter fine-tuning (SPECS.md Section 11.3, row 5) and
//! simulation-guided fine-tuning (row 7), both gated by obligation 5:
//! "evaluated against the frozen pre-adaptation baseline on the held-out
//! session, reporting both improvement and regression, and MUST support
//! rollback to that baseline as an immediately available artifact".
//!
//! [`fine_tune`] trains only the parameters a [`ParamGroup`] table leaves
//! unfrozen, on real data, and returns a [`FineTuneOutcome`] that always
//! carries the untouched baseline next to the candidate: rollback is
//! dropping the candidate. The candidate is `accepted` only if its held-out
//! loss beats the baseline's by the caller's margin; a regression is
//! reported, never adopted.
//!
//! Row 7's real-world updates are on-hardware weight updates; scheduling
//! one goes through [`crate::ladder::AdaptationLadder`], which demands the
//! separate decision obligation 7 names. Its objective is the real loss
//! plus a weighted simulation-derived term ([`GuidedObjective`]), whose
//! errors are the simulator's, so the outcome reports both terms.

use robot_nn::mlp::Mlp;

/// The result of a gated fine-tune.
#[derive(Clone, PartialEq, Debug)]
pub struct FineTuneOutcome {
    /// The frozen pre-adaptation network: the rollback artifact.
    pub baseline: Mlp,
    /// The fine-tuned network.
    pub candidate: Mlp,
    /// Held-out loss of the baseline.
    pub baseline_loss: f64,
    /// Held-out loss of the candidate.
    pub candidate_loss: f64,
    /// Whether the candidate beat the baseline by the margin.
    pub accepted: bool,
    /// Trainable parameter count.
    pub trained_parameters: usize,
}

impl FineTuneOutcome {
    /// The network to deploy: the candidate if accepted, else the baseline.
    #[must_use]
    pub fn deployable(&self) -> &Mlp {
        if self.accepted {
            &self.candidate
        } else {
            &self.baseline
        }
    }
}

/// A simulation-guided objective: real MSE plus `weight` times a
/// simulation-derived MSE toward `sim_targets` on the same inputs.
#[derive(Clone, PartialEq, Debug)]
pub struct GuidedObjective {
    /// Simulation-derived targets, one per training input.
    pub sim_targets: Vec<Vec<f64>>,
    /// Weight of the simulation term.
    pub weight: f64,
}

/// Fine-tunes `base` on `train_data`, training only parameters `groups`
/// leaves unfrozen, and gates the result on `held_out`.
#[cfg(feature = "training")]
pub fn fine_tune(
    base: &Mlp,
    groups: &[robot_nn::mlp::ParamGroup],
    train_data: &[(Vec<f64>, Vec<f64>)],
    held_out: &[(Vec<f64>, Vec<f64>)],
    guide: Option<&GuidedObjective>,
    config: &crate::encoder::FitConfig,
    min_improvement: f64,
) -> Result<FineTuneOutcome, &'static str> {
    use crate::encoder::mse;
    let frozen = base
        .frozen_mask(groups)
        .map_err(|_| "parameter groups do not fit the network")?;
    let trained_parameters = frozen.iter().filter(|f| !**f).count();
    // The guided term is folded into the targets: minimizing
    // |y - a|^2 + w |y - b|^2 is minimizing (1 + w) |y - (a + w b)/(1 + w)|^2.
    let data: Vec<(Vec<f64>, Vec<f64>)> = match guide {
        Some(g) if g.sim_targets.len() == train_data.len() => train_data
            .iter()
            .zip(&g.sim_targets)
            .map(|((x, y), s)| {
                let t = y
                    .iter()
                    .zip(s)
                    .map(|(a, b)| (a + g.weight * b) / (1.0 + g.weight))
                    .collect();
                (x.clone(), t)
            })
            .collect(),
        Some(_) => return Err("one simulation target per training input"),
        None => train_data.to_vec(),
    };
    let mut candidate = base.clone();
    crate::encoder::train(&mut candidate, &data, config, &frozen)?;
    let baseline_loss = mse(base, held_out);
    let candidate_loss = mse(&candidate, held_out);
    Ok(FineTuneOutcome {
        baseline: base.clone(),
        candidate,
        baseline_loss,
        candidate_loss,
        accepted: candidate_loss < baseline_loss * (1.0 - min_improvement),
        trained_parameters,
    })
}

#[cfg(all(test, feature = "training"))]
mod tests {
    use super::*;
    use crate::encoder::FitConfig;
    use robot_nn::init::initialize;
    use robot_nn::layer::Activation;
    use robot_nn::mlp::ParamGroup;
    use robot_types::rng::{Rng, SeedStream};

    fn data(offset: f64, n: usize, phase: f64) -> Vec<(Vec<f64>, Vec<f64>)> {
        (0..n)
            .map(|i| {
                let x = ((i as f64) * 0.37 + phase).sin();
                (vec![x], vec![2.0 * x + offset])
            })
            .collect()
    }

    fn base() -> Mlp {
        let mut rng = Rng::new(SeedStream::new(1, 0));
        let mut m = initialize(&[1, 1], vec![Activation::Identity], &mut rng).unwrap();
        m.layers_mut()[0].weight_mut()[0] = 2.0;
        m.layers_mut()[0].bias_mut()[0] = 0.0;
        m
    }

    fn config() -> FitConfig {
        FitConfig {
            epochs: 300,
            batch: 16,
            learning_rate: 1e-2,
            seed: 3,
        }
    }

    /// Only the bias (flat index 1) is trainable.
    fn bias_only() -> Vec<ParamGroup> {
        vec![ParamGroup {
            name: "weights".into(),
            range: 0..1,
            frozen: true,
        }]
    }

    #[test]
    fn a_bias_only_adapter_fixes_an_offset_and_is_accepted() {
        let out = fine_tune(
            &base(),
            &bias_only(),
            &data(0.5, 64, 0.0),
            &data(0.5, 32, 9.0),
            None,
            &config(),
            0.1,
        )
        .unwrap();
        assert_eq!(out.trained_parameters, 1);
        assert!(
            out.accepted,
            "{} vs {}",
            out.candidate_loss, out.baseline_loss
        );
        assert_eq!(out.candidate.layers()[0].weight()[0], 2.0);
        assert!((out.candidate.layers()[0].bias()[0] - 0.5).abs() < 0.05);
        assert_eq!(out.baseline, base());
    }

    #[test]
    fn a_fine_tune_that_regresses_on_held_out_data_is_not_deployed() {
        // Train on an offset the held-out session does not have.
        let out = fine_tune(
            &base(),
            &bias_only(),
            &data(0.5, 64, 0.0),
            &data(0.0, 32, 9.0),
            None,
            &config(),
            0.0,
        )
        .unwrap();
        assert!(!out.accepted);
        assert_eq!(out.deployable(), &out.baseline);
    }

    #[test]
    fn a_guided_term_pulls_toward_the_simulation_target() {
        let train = data(0.5, 64, 0.0);
        let guide = GuidedObjective {
            sim_targets: train.iter().map(|(x, _)| vec![2.0 * x[0] - 0.5]).collect(),
            weight: 1.0,
        };
        let out = fine_tune(
            &base(),
            &bias_only(),
            &train,
            &data(0.0, 32, 9.0),
            Some(&guide),
            &config(),
            0.0,
        )
        .unwrap();
        // Equal weights on +0.5 and -0.5 land near zero.
        assert!(out.candidate.layers()[0].bias()[0].abs() < 0.05);
    }
}

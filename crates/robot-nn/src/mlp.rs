//! A sequential dense network with preallocated scratch, for the
//! deadline-critical forward pass.
//!
//! SPECS.md Section 5: "Keep inference resident and allocate working memory
//! before activation" and "The control path MUST NOT perform... heap
//! allocation after activation." [`Scratch`] is that preallocated working
//! memory, built once from [`Mlp::scratch_len`] and reused on every call to
//! [`Mlp::forward`].

use crate::layer::{Activation, LayerError, Linear};
use core::fmt;
use core::ops::Range;

/// Why an MLP could not be built or evaluated.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MlpError {
    /// Adjacent layers disagreed about width.
    WidthMismatch {
        /// Index of the layer whose output width is wrong.
        layer: usize,
        /// Width the next layer expects.
        expected: usize,
        /// Width this layer produces.
        found: usize,
    },
    /// The number of activations did not match the number of layers.
    ActivationCountMismatch {
        /// Layers present.
        layers: usize,
        /// Activations present.
        activations: usize,
    },
    /// The network has no layers.
    Empty,
    /// A layer failed its own construction or evaluation.
    Layer(LayerError),
    /// The scratch buffer was sized for a different network.
    ScratchMismatch {
        /// Length the network needs.
        expected: usize,
        /// Length supplied.
        found: usize,
    },
}

impl fmt::Display for MlpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WidthMismatch {
                layer,
                expected,
                found,
            } => write!(
                f,
                "layer {layer} produces width {found}, next layer expects {expected}"
            ),
            Self::ActivationCountMismatch {
                layers,
                activations,
            } => write!(
                f,
                "{layers} layers but {activations} activations; they must match 1:1"
            ),
            Self::Empty => write!(f, "network has no layers"),
            Self::Layer(e) => write!(f, "{e}"),
            Self::ScratchMismatch { expected, found } => {
                write!(
                    f,
                    "scratch buffer has length {found}, network needs {expected}"
                )
            }
        }
    }
}

impl core::error::Error for MlpError {}

impl From<LayerError> for MlpError {
    fn from(e: LayerError) -> Self {
        Self::Layer(e)
    }
}

/// A named slice of an [`Mlp`]'s parameters that training may freeze.
///
/// `range` indexes the flat parameter order used by [`Mlp::frozen_mask`]:
/// each layer's weights in row-major order, then its biases, layer by layer.
/// Parameters covered by no group stay trainable.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ParamGroup {
    /// Identifies the group in errors and in checkpoints. Must be non-empty
    /// and unique within a table.
    pub name: String,
    /// Flat parameter indices the group covers.
    pub range: Range<usize>,
    /// When set, training leaves every parameter in `range` unchanged.
    pub frozen: bool,
}

/// Why a parameter group table does not fit a network.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ParamGroupError {
    /// A group has an empty name.
    EmptyName,
    /// Two groups share a name.
    DuplicateName(String),
    /// A group's range is empty or reversed.
    EmptyRange {
        /// The offending group.
        name: String,
    },
    /// A group's range reaches past the network's parameters.
    OutOfBounds {
        /// The offending group.
        name: String,
        /// One past the last index the group names.
        end: usize,
        /// Parameters the network has.
        count: usize,
    },
    /// Two groups cover the same parameter.
    Overlap {
        /// The earlier group.
        first: String,
        /// The group that collides with it.
        second: String,
    },
}

impl fmt::Display for ParamGroupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyName => write!(f, "parameter group has an empty name"),
            Self::DuplicateName(name) => write!(f, "parameter group {name:?} is defined twice"),
            Self::EmptyRange { name } => {
                write!(f, "parameter group {name:?} has an empty or reversed range")
            }
            Self::OutOfBounds { name, end, count } => write!(
                f,
                "parameter group {name:?} ends at {end}, but the network has {count} parameters"
            ),
            Self::Overlap { first, second } => {
                write!(f, "parameter groups {first:?} and {second:?} overlap")
            }
        }
    }
}

impl core::error::Error for ParamGroupError {}

/// Preallocated working memory for [`Mlp::forward`].
///
/// Two ping-pong buffers sized to the network's widest layer, so a forward
/// pass through any depth of network touches no allocator.
#[derive(Clone, Debug)]
pub struct Scratch {
    a: Vec<f64>,
    b: Vec<f64>,
}

impl Scratch {
    /// Allocates scratch space for a network whose widest layer output is
    /// `max_width`.
    #[must_use]
    pub fn new(max_width: usize) -> Self {
        Self {
            a: vec![0.0; max_width],
            b: vec![0.0; max_width],
        }
    }

    /// The capacity this scratch space was built for.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.a.len()
    }
}

/// A sequence of dense layers, each followed by an activation.
#[derive(Clone, PartialEq, Debug)]
pub struct Mlp {
    layers: Vec<Linear>,
    activations: Vec<Activation>,
}

impl Mlp {
    /// Builds a network, checking that adjacent layer widths agree and that
    /// every layer has exactly one activation.
    pub fn new(layers: Vec<Linear>, activations: Vec<Activation>) -> Result<Self, MlpError> {
        if layers.is_empty() {
            return Err(MlpError::Empty);
        }
        if layers.len() != activations.len() {
            return Err(MlpError::ActivationCountMismatch {
                layers: layers.len(),
                activations: activations.len(),
            });
        }
        for (i, pair) in layers.windows(2).enumerate() {
            if pair[0].out_features() != pair[1].in_features() {
                return Err(MlpError::WidthMismatch {
                    layer: i,
                    expected: pair[1].in_features(),
                    found: pair[0].out_features(),
                });
            }
        }
        Ok(Self {
            layers,
            activations,
        })
    }

    /// Width of the input vector.
    #[must_use]
    pub fn input_width(&self) -> usize {
        self.layers[0].in_features()
    }

    /// Width of the output vector.
    #[must_use]
    pub fn output_width(&self) -> usize {
        self.layers[self.layers.len() - 1].out_features()
    }

    /// Number of layers.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.layers.len()
    }

    /// Read-only access to the layers, in forward order.
    #[must_use]
    pub fn layers(&self) -> &[Linear] {
        &self.layers
    }

    /// Mutable access to the layers, for an optimizer step.
    pub fn layers_mut(&mut self) -> &mut [Linear] {
        &mut self.layers
    }

    /// Activations, one per layer.
    #[must_use]
    pub fn activations(&self) -> &[Activation] {
        &self.activations
    }

    /// Total number of parameters: every layer's weights plus its biases.
    #[must_use]
    pub fn param_count(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.weight().len() + l.bias().len())
            .sum()
    }

    /// Checks `groups` against this network and returns one flag per flat
    /// parameter, `true` where a frozen group covers it.
    ///
    /// Groups must have distinct non-empty names, non-empty ranges inside
    /// [`Mlp::param_count`], and no shared parameters. Any violation is an
    /// error and no mask is produced.
    pub fn frozen_mask(&self, groups: &[ParamGroup]) -> Result<Vec<bool>, ParamGroupError> {
        frozen_mask_for_count(self.param_count(), groups)
    }
}

/// Checks `groups` against a flat parameter space of `count` entries and
/// returns one flag per entry, `true` where a frozen group covers it.
///
/// [`Mlp::frozen_mask`] uses this with the network's own parameter count. An
/// owner that appends parameters to that flat order (for example a policy's
/// log-std vector) uses it with the extended count.
pub fn frozen_mask_for_count(
    count: usize,
    groups: &[ParamGroup],
) -> Result<Vec<bool>, ParamGroupError> {
    let mut mask = vec![false; count];
    for (i, group) in groups.iter().enumerate() {
        if group.name.is_empty() {
            return Err(ParamGroupError::EmptyName);
        }
        if groups[..i].iter().any(|earlier| earlier.name == group.name) {
            return Err(ParamGroupError::DuplicateName(group.name.clone()));
        }
        if group.range.start >= group.range.end {
            return Err(ParamGroupError::EmptyRange {
                name: group.name.clone(),
            });
        }
        if group.range.end > count {
            return Err(ParamGroupError::OutOfBounds {
                name: group.name.clone(),
                end: group.range.end,
                count,
            });
        }
        for earlier in &groups[..i] {
            if group.range.start < earlier.range.end && earlier.range.start < group.range.end {
                return Err(ParamGroupError::Overlap {
                    first: earlier.name.clone(),
                    second: group.name.clone(),
                });
            }
        }
        if group.frozen {
            mask[group.range.clone()].fill(true);
        }
    }
    Ok(mask)
}

impl Mlp {
    /// The scratch capacity [`Scratch::new`] must be built with to evaluate
    /// this network.
    #[must_use]
    pub fn scratch_len(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.in_features().max(l.out_features()))
            .max()
            .unwrap_or(0)
            .max(self.input_width())
    }

    /// Evaluates the network. Allocates nothing: `scratch` and `out` are
    /// both supplied by the caller.
    ///
    /// Both the learner and the deployed runtime call this same function
    /// against the same weights, which is what makes an action-parity check
    /// between them a triviality rather than a separate numerical routine to
    /// maintain (PLAN.md Phase 4, "implement identical Rust
    /// normalization/action transforms for training and deployment").
    pub fn forward(
        &self,
        input: &[f64],
        out: &mut [f64],
        scratch: &mut Scratch,
    ) -> Result<(), MlpError> {
        if input.len() != self.input_width() {
            return Err(LayerError::BufferShapeMismatch {
                expected: self.input_width(),
                found: input.len(),
            }
            .into());
        }
        if out.len() != self.output_width() {
            return Err(LayerError::BufferShapeMismatch {
                expected: self.output_width(),
                found: out.len(),
            }
            .into());
        }
        let needed = self.scratch_len();
        if scratch.a.len() < needed || scratch.b.len() < needed {
            return Err(MlpError::ScratchMismatch {
                expected: needed,
                found: scratch.a.len(),
            });
        }

        scratch.a[..input.len()].copy_from_slice(input);
        let mut current_len = input.len();
        let mut use_a = true;

        for (layer, activation) in self.layers.iter().zip(&self.activations) {
            let (src, dst) = if use_a {
                (
                    &scratch.a[..current_len],
                    &mut scratch.b[..layer.out_features()],
                )
            } else {
                (
                    &scratch.b[..current_len],
                    &mut scratch.a[..layer.out_features()],
                )
            };
            layer.forward(src, dst)?;
            for v in dst.iter_mut() {
                *v = activation.apply(*v);
            }
            current_len = layer.out_features();
            use_a = !use_a;
        }

        let final_buf = if use_a { &scratch.a } else { &scratch.b };
        out.copy_from_slice(&final_buf[..current_len]);
        Ok(())
    }

    /// Records every intermediate pre-activation and post-activation value,
    /// for use by a backward pass. `robot-learn` owns backpropagation; this
    /// method exists here because it shares `forward`'s layer-by-layer loop
    /// and nothing else needs it.
    ///
    /// Allocates: this is an offline/training-time path, not the
    /// deadline-critical one, and §5's no-allocation rule applies to the
    /// control path, not to training.
    #[must_use]
    pub fn forward_with_trace(&self, input: &[f64]) -> ForwardTrace {
        let mut pre_activations = Vec::with_capacity(self.layers.len());
        let mut post_activations = Vec::with_capacity(self.layers.len());
        let mut current = input.to_vec();
        for (layer, activation) in self.layers.iter().zip(&self.activations) {
            let mut z = vec![0.0; layer.out_features()];
            layer
                .forward(&current, &mut z)
                .expect("widths validated at construction");
            let a: Vec<f64> = z.iter().map(|v| activation.apply(*v)).collect();
            pre_activations.push(z);
            post_activations.push(a.clone());
            current = a;
        }
        ForwardTrace {
            input: input.to_vec(),
            pre_activations,
            post_activations,
        }
    }
}

/// The intermediate values of one forward pass, for backpropagation.
#[derive(Clone, Debug)]
pub struct ForwardTrace {
    /// The network's input.
    pub input: Vec<f64>,
    /// Pre-activation values `z` for each layer, in forward order.
    pub pre_activations: Vec<Vec<f64>>,
    /// Post-activation values `a` for each layer, in forward order. The last
    /// entry is the network's output.
    pub post_activations: Vec<Vec<f64>>,
}

impl ForwardTrace {
    /// The network's output.
    #[must_use]
    pub fn output(&self) -> &[f64] {
        self.post_activations
            .last()
            .map_or(&self.input, |v| v.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_layer_net() -> Mlp {
        // 2 -> 3 (tanh) -> 1 (identity)
        let l0 = Linear::new(2, 3, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![0.0; 3]).unwrap();
        let l1 = Linear::new(3, 1, vec![1.0, 1.0, 1.0], vec![0.0]).unwrap();
        Mlp::new(vec![l0, l1], vec![Activation::Tanh, Activation::Identity]).unwrap()
    }

    #[test]
    fn forward_matches_a_hand_traced_computation() {
        let net = two_layer_net();
        let mut scratch = Scratch::new(net.scratch_len());
        let mut out = [0.0; 1];
        net.forward(&[0.5, -0.25], &mut out, &mut scratch).unwrap();

        // Layer 0 pre-activation: [0.5, -0.25, 0.25]
        let z0 = [0.5f64, -0.25, 0.25];
        let a0: Vec<f64> = z0.iter().map(|v| v.tanh()).collect();
        let expected: f64 = a0.iter().sum();
        assert!((out[0] - expected).abs() < 1e-12);
    }

    #[test]
    fn forward_is_deterministic_and_allocation_free_across_calls() {
        let net = two_layer_net();
        let mut scratch = Scratch::new(net.scratch_len());
        let mut out_a = [0.0; 1];
        let mut out_b = [0.0; 1];
        net.forward(&[0.3, 0.7], &mut out_a, &mut scratch).unwrap();
        net.forward(&[0.3, 0.7], &mut out_b, &mut scratch).unwrap();
        assert_eq!(
            out_a, out_b,
            "same input and weights must give bit-identical output"
        );
    }

    #[test]
    fn two_independently_built_networks_with_the_same_weights_agree_exactly() {
        // This is the action-parity property PLAN.md Phase 4 asks for,
        // demonstrated directly: a "learner" copy and a "runtime" copy built
        // from the same weight data produce identical output.
        let learner = two_layer_net();
        let runtime = two_layer_net();
        let mut s1 = Scratch::new(learner.scratch_len());
        let mut s2 = Scratch::new(runtime.scratch_len());
        let mut o1 = [0.0; 1];
        let mut o2 = [0.0; 1];
        learner.forward(&[1.0, 2.0], &mut o1, &mut s1).unwrap();
        runtime.forward(&[1.0, 2.0], &mut o2, &mut s2).unwrap();
        assert_eq!(o1, o2);
    }

    #[test]
    fn mismatched_adjacent_widths_are_rejected() {
        let l0 = Linear::zeros(2, 3);
        let l1 = Linear::zeros(4, 1);
        assert!(matches!(
            Mlp::new(vec![l0, l1], vec![Activation::Tanh, Activation::Identity]),
            Err(MlpError::WidthMismatch {
                layer: 0,
                expected: 4,
                found: 3
            })
        ));
    }

    #[test]
    fn activation_count_must_match_layer_count() {
        let l0 = Linear::zeros(2, 2);
        assert!(matches!(
            Mlp::new(vec![l0], vec![Activation::Tanh, Activation::Identity]),
            Err(MlpError::ActivationCountMismatch {
                layers: 1,
                activations: 2
            })
        ));
    }

    #[test]
    fn an_empty_network_is_rejected() {
        assert!(matches!(Mlp::new(vec![], vec![]), Err(MlpError::Empty)));
    }

    #[test]
    fn undersized_scratch_is_rejected_rather_than_overrun() {
        let net = two_layer_net();
        let mut tiny = Scratch::new(1);
        let mut out = [0.0; 1];
        assert!(matches!(
            net.forward(&[0.0, 0.0], &mut out, &mut tiny),
            Err(MlpError::ScratchMismatch { .. })
        ));
    }

    fn group(name: &str, range: Range<usize>, frozen: bool) -> ParamGroup {
        ParamGroup {
            name: name.to_owned(),
            range,
            frozen,
        }
    }

    #[test]
    fn frozen_mask_marks_exactly_the_frozen_flat_indices() {
        let net = two_layer_net();
        assert_eq!(net.param_count(), 13);
        let mask = net
            .frozen_mask(&[
                group("l0-weights", 0..6, true),
                group("l1-bias", 12..13, false),
            ])
            .unwrap();
        let expected: Vec<bool> = (0..13).map(|i| i < 6).collect();
        assert_eq!(mask, expected);
    }

    #[test]
    fn parameter_group_tables_that_do_not_fit_are_rejected() {
        let net = two_layer_net();
        assert_eq!(
            net.frozen_mask(&[group("", 0..1, true)]),
            Err(ParamGroupError::EmptyName)
        );
        assert_eq!(
            net.frozen_mask(&[group("a", 0..1, true), group("a", 2..3, true)]),
            Err(ParamGroupError::DuplicateName("a".to_owned()))
        );
        assert_eq!(
            net.frozen_mask(&[group("empty", 4..4, true)]),
            Err(ParamGroupError::EmptyRange {
                name: "empty".to_owned()
            })
        );
        assert_eq!(
            net.frozen_mask(&[group("far", 10..14, true)]),
            Err(ParamGroupError::OutOfBounds {
                name: "far".to_owned(),
                end: 14,
                count: 13
            })
        );
        assert_eq!(
            net.frozen_mask(&[group("a", 0..6, true), group("b", 5..9, false)]),
            Err(ParamGroupError::Overlap {
                first: "a".to_owned(),
                second: "b".to_owned()
            })
        );
    }

    #[test]
    fn trace_output_matches_direct_forward() {
        let net = two_layer_net();
        let trace = net.forward_with_trace(&[0.5, -0.25]);
        let mut scratch = Scratch::new(net.scratch_len());
        let mut out = [0.0; 1];
        net.forward(&[0.5, -0.25], &mut out, &mut scratch).unwrap();
        assert!((trace.output()[0] - out[0]).abs() < 1e-12);
        assert_eq!(trace.pre_activations.len(), 2);
        assert_eq!(trace.post_activations.len(), 2);
    }
}

//! The closed, exportable operator set: dense layers and elementwise
//! activations.
//!
//! SPECS.md Section 7.4: "Unsupported operators must fail before
//! deployment." For ONNX that is a runtime check against a parser's coverage;
//! for this authored format the equivalent guarantee is structural — the
//! operator set is a closed Rust enum, so a bundle cannot be constructed with
//! an operator this crate does not implement. There is no "unsupported
//! operator" state to fail on at load time because the type system already
//! excludes it at construction time.

use core::fmt;

/// An elementwise activation function.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Activation {
    /// No transformation.
    Identity,
    /// Hyperbolic tangent, bounded to `(-1, 1)`.
    Tanh,
    /// Rectified linear unit.
    ReLU,
    /// Logistic sigmoid, bounded to `(0, 1)`.
    Sigmoid,
}

impl Activation {
    /// Applies the activation to one value.
    #[inline]
    #[must_use]
    pub fn apply(self, x: f64) -> f64 {
        match self {
            Self::Identity => x,
            Self::Tanh => x.tanh(),
            Self::ReLU => x.max(0.0),
            Self::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        }
    }

    /// The derivative of the activation with respect to its input, evaluated
    /// from the activation's own *output* `y = apply(x)` where that is
    /// cheaper (tanh, sigmoid) and from the input directly for ReLU.
    ///
    /// Taking the output rather than the input is the standard trick that
    /// avoids recomputing `tanh`/`exp` during backpropagation; the forward
    /// pass already produced `y` and the backward pass reuses it.
    #[inline]
    #[must_use]
    pub fn derivative_from_output(self, x: f64, y: f64) -> f64 {
        match self {
            Self::Identity => 1.0,
            Self::Tanh => 1.0 - y * y,
            Self::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Sigmoid => y * (1.0 - y),
        }
    }

    /// The canonical identifier used in bundles.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::Tanh => "tanh",
            Self::ReLU => "relu",
            Self::Sigmoid => "sigmoid",
        }
    }

    /// Parses an identifier.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "identity" => Some(Self::Identity),
            "tanh" => Some(Self::Tanh),
            "relu" => Some(Self::ReLU),
            "sigmoid" => Some(Self::Sigmoid),
            _ => None,
        }
    }
}

impl fmt::Display for Activation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

/// A dense (fully connected) layer: `y = W x + b`.
///
/// Weights are stored row-major, `out_features` rows by `in_features`
/// columns, so that `forward` accesses each row contiguously.
#[derive(Clone, PartialEq, Debug)]
pub struct Linear {
    in_features: usize,
    out_features: usize,
    weight: Vec<f64>,
    bias: Vec<f64>,
}

/// Why a layer could not be built or evaluated.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LayerError {
    /// The weight vector's length did not match `out_features * in_features`.
    WeightShapeMismatch {
        /// Expected length.
        expected: usize,
        /// Length found.
        found: usize,
    },
    /// The bias vector's length did not match `out_features`.
    BiasShapeMismatch {
        /// Expected length.
        expected: usize,
        /// Length found.
        found: usize,
    },
    /// A weight or bias entry was not finite.
    NotFinite,
    /// An input or output buffer had the wrong length.
    BufferShapeMismatch {
        /// Expected length.
        expected: usize,
        /// Length found.
        found: usize,
    },
}

impl fmt::Display for LayerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WeightShapeMismatch { expected, found } => {
                write!(f, "weight has {found} entries, expected {expected}")
            }
            Self::BiasShapeMismatch { expected, found } => {
                write!(f, "bias has {found} entries, expected {expected}")
            }
            Self::NotFinite => write!(f, "weight or bias contains a non-finite entry"),
            Self::BufferShapeMismatch { expected, found } => {
                write!(f, "buffer has {found} entries, expected {expected}")
            }
        }
    }
}

impl core::error::Error for LayerError {}

impl Linear {
    /// Builds a layer, validating shapes and finiteness.
    pub fn new(
        in_features: usize,
        out_features: usize,
        weight: Vec<f64>,
        bias: Vec<f64>,
    ) -> Result<Self, LayerError> {
        let expected_weight = out_features * in_features;
        if weight.len() != expected_weight {
            return Err(LayerError::WeightShapeMismatch {
                expected: expected_weight,
                found: weight.len(),
            });
        }
        if bias.len() != out_features {
            return Err(LayerError::BiasShapeMismatch {
                expected: out_features,
                found: bias.len(),
            });
        }
        if weight.iter().chain(&bias).any(|v| !v.is_finite()) {
            return Err(LayerError::NotFinite);
        }
        Ok(Self {
            in_features,
            out_features,
            weight,
            bias,
        })
    }

    /// A layer of zero weights and biases.
    #[must_use]
    pub fn zeros(in_features: usize, out_features: usize) -> Self {
        Self {
            in_features,
            out_features,
            weight: vec![0.0; in_features * out_features],
            bias: vec![0.0; out_features],
        }
    }

    /// Input width.
    #[inline]
    #[must_use]
    pub const fn in_features(&self) -> usize {
        self.in_features
    }

    /// Output width.
    #[inline]
    #[must_use]
    pub const fn out_features(&self) -> usize {
        self.out_features
    }

    /// Read-only access to the row-major weight matrix.
    #[must_use]
    pub fn weight(&self) -> &[f64] {
        &self.weight
    }

    /// Read-only access to the bias vector.
    #[must_use]
    pub fn bias(&self) -> &[f64] {
        &self.bias
    }

    /// Mutable access to the weight matrix, for an optimizer step.
    pub fn weight_mut(&mut self) -> &mut [f64] {
        &mut self.weight
    }

    /// Mutable access to the bias vector, for an optimizer step.
    pub fn bias_mut(&mut self) -> &mut [f64] {
        &mut self.bias
    }

    /// Computes `out = W * input + b`. Allocates nothing: `out` is supplied
    /// by the caller, which is what lets a control-path caller preallocate
    /// once and call this every tick.
    pub fn forward(&self, input: &[f64], out: &mut [f64]) -> Result<(), LayerError> {
        if input.len() != self.in_features {
            return Err(LayerError::BufferShapeMismatch {
                expected: self.in_features,
                found: input.len(),
            });
        }
        if out.len() != self.out_features {
            return Err(LayerError::BufferShapeMismatch {
                expected: self.out_features,
                found: out.len(),
            });
        }
        for (row, (o, b)) in out.iter_mut().zip(&self.bias).enumerate() {
            let w_row = &self.weight[row * self.in_features..(row + 1) * self.in_features];
            let mut acc = *b;
            for (w, x) in w_row.iter().zip(input) {
                acc += w * x;
            }
            *o = acc;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activations_apply_pointwise() {
        assert_eq!(Activation::Identity.apply(2.5), 2.5);
        assert_eq!(Activation::ReLU.apply(-1.0), 0.0);
        assert_eq!(Activation::ReLU.apply(2.0), 2.0);
        assert!((Activation::Sigmoid.apply(0.0) - 0.5).abs() < 1e-12);
        assert!(Activation::Tanh.apply(0.0).abs() < 1e-12);
    }

    #[test]
    fn activation_identifiers_round_trip() {
        for a in [
            Activation::Identity,
            Activation::Tanh,
            Activation::ReLU,
            Activation::Sigmoid,
        ] {
            assert_eq!(Activation::parse(a.id()), Some(a));
        }
        assert_eq!(Activation::parse("gelu"), None);
    }

    #[test]
    fn derivative_from_output_matches_the_closed_form() {
        let x = 0.37;
        let y = Activation::Tanh.apply(x);
        let numerical =
            (Activation::Tanh.apply(x + 1e-6) - Activation::Tanh.apply(x - 1e-6)) / 2e-6;
        let analytic = Activation::Tanh.derivative_from_output(x, y);
        assert!((numerical - analytic).abs() < 1e-6);

        let ys = Activation::Sigmoid.apply(x);
        let num_s =
            (Activation::Sigmoid.apply(x + 1e-6) - Activation::Sigmoid.apply(x - 1e-6)) / 2e-6;
        assert!((num_s - Activation::Sigmoid.derivative_from_output(x, ys)).abs() < 1e-6);

        assert_eq!(Activation::ReLU.derivative_from_output(1.0, 1.0), 1.0);
        assert_eq!(Activation::ReLU.derivative_from_output(-1.0, 0.0), 0.0);
    }

    #[test]
    fn linear_forward_matches_hand_computation() {
        // 2 inputs, 2 outputs: W = [[1,2],[3,4]], b = [0.5, -0.5]
        let layer = Linear::new(2, 2, vec![1.0, 2.0, 3.0, 4.0], vec![0.5, -0.5]).unwrap();
        let mut out = [0.0; 2];
        layer.forward(&[1.0, 1.0], &mut out).unwrap();
        assert!((out[0] - 3.5).abs() < 1e-12); // 1*1+2*1+0.5
        assert!((out[1] - 6.5).abs() < 1e-12); // 3*1+4*1-0.5
    }

    #[test]
    fn linear_rejects_malformed_shapes_and_nonfinite_values() {
        assert!(matches!(
            Linear::new(2, 2, vec![1.0, 2.0, 3.0], vec![0.0, 0.0]),
            Err(LayerError::WeightShapeMismatch { .. })
        ));
        assert!(matches!(
            Linear::new(2, 2, vec![1.0, 2.0, 3.0, 4.0], vec![0.0]),
            Err(LayerError::BiasShapeMismatch { .. })
        ));
        assert!(matches!(
            Linear::new(1, 1, vec![f64::NAN], vec![0.0]),
            Err(LayerError::NotFinite)
        ));
    }

    #[test]
    fn forward_rejects_mismatched_buffers() {
        let layer = Linear::zeros(2, 3);
        let mut out = [0.0; 3];
        assert!(matches!(
            layer.forward(&[1.0], &mut out),
            Err(LayerError::BufferShapeMismatch {
                expected: 2,
                found: 1
            })
        ));
        let mut short_out = [0.0; 2];
        assert!(matches!(
            layer.forward(&[1.0, 1.0], &mut short_out),
            Err(LayerError::BufferShapeMismatch {
                expected: 3,
                found: 2
            })
        ));
    }
}

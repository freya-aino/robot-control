//! `robot-mlp/1`, the Rust-loadable versioned weights format of SPECS.md
//! Section 7.4 ("model weights in a Rust-loadable versioned format,
//! graph/architecture version"), and the place where "Unsupported operators
//! must fail before deployment" is enforced for a file read from disk.
//!
//! # Layout (little-endian)
//!
//! | Field | Size |
//! | --- | --- |
//! | magic `b"RMLP"` | 4 |
//! | format version, `1` | 2 |
//! | layer count `L` | 2 |
//! | per layer: input width `u32`, output width `u32`, activation code `u8` | 9 each |
//! | per layer: weights `out * in` then biases `out`, `f64` | 8 each |
//!
//! Activation codes: `0` identity, `1` tanh, `2` ReLU, `3` sigmoid. Any
//! other code is an unsupported operator and the load fails, naming it.
//! Widths must chain (each layer's input equals the previous output), every
//! value must be finite, and the declared sizes must match the bytes
//! exactly, so a truncated or padded file is refused rather than partially
//! read. [`MAX_PARAMETERS`] bounds the allocation a header can request.

use crate::layer::{Activation, Linear};
use crate::mlp::Mlp;

/// Largest parameter count a file may declare.
pub const MAX_PARAMETERS: usize = 1 << 26;

/// Why a weights file was refused.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum FormatError {
    /// Not `RMLP`.
    Magic,
    /// A format version this reader does not know.
    Version(u16),
    /// An activation code this crate does not implement.
    UnsupportedOperator {
        /// The layer.
        layer: usize,
        /// The code.
        code: u8,
    },
    /// Byte count disagrees with the header.
    Size,
    /// Widths do not chain, or are zero.
    Shape,
    /// A non-finite parameter.
    NotFinite,
    /// More parameters than [`MAX_PARAMETERS`].
    TooLarge,
}

impl core::fmt::Display for FormatError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Magic => f.write_str("not a robot-mlp file"),
            Self::Version(v) => write!(f, "unsupported robot-mlp version {v}"),
            Self::UnsupportedOperator { layer, code } => {
                write!(
                    f,
                    "layer {layer}: unsupported operator (activation code {code})"
                )
            }
            Self::Size => f.write_str("file size disagrees with its header"),
            Self::Shape => f.write_str("layer widths do not chain"),
            Self::NotFinite => f.write_str("non-finite parameter"),
            Self::TooLarge => f.write_str("parameter count over the limit"),
        }
    }
}

impl core::error::Error for FormatError {}

const fn code(a: Activation) -> u8 {
    match a {
        Activation::Identity => 0,
        Activation::Tanh => 1,
        Activation::ReLU => 2,
        Activation::Sigmoid => 3,
    }
}

const fn activation(code: u8) -> Option<Activation> {
    Some(match code {
        0 => Activation::Identity,
        1 => Activation::Tanh,
        2 => Activation::ReLU,
        3 => Activation::Sigmoid,
        _ => return None,
    })
}

/// Serializes `mlp`.
#[must_use]
pub fn encode(mlp: &Mlp) -> Vec<u8> {
    let mut out = b"RMLP".to_vec();
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&(mlp.layers().len() as u16).to_le_bytes());
    for (layer, act) in mlp.layers().iter().zip(mlp.activations()) {
        out.extend_from_slice(&(layer.in_features() as u32).to_le_bytes());
        out.extend_from_slice(&(layer.out_features() as u32).to_le_bytes());
        out.push(code(*act));
    }
    for layer in mlp.layers() {
        for v in layer.weight().iter().chain(layer.bias()) {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// Deserializes and validates a network.
pub fn decode(bytes: &[u8]) -> Result<Mlp, FormatError> {
    if bytes.get(..4) != Some(b"RMLP") {
        return Err(FormatError::Magic);
    }
    let u16_at = |o: usize| {
        bytes
            .get(o..o + 2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .ok_or(FormatError::Size)
    };
    let version = u16_at(4)?;
    if version != 1 {
        return Err(FormatError::Version(version));
    }
    let count = usize::from(u16_at(6)?);
    let mut shapes = Vec::with_capacity(count);
    let mut total = 0usize;
    let mut previous: Option<usize> = None;
    for k in 0..count {
        let o = 8 + 9 * k;
        let h = bytes.get(o..o + 9).ok_or(FormatError::Size)?;
        let inw = u32::from_le_bytes([h[0], h[1], h[2], h[3]]) as usize;
        let outw = u32::from_le_bytes([h[4], h[5], h[6], h[7]]) as usize;
        let act = activation(h[8]).ok_or(FormatError::UnsupportedOperator {
            layer: k,
            code: h[8],
        })?;
        if inw == 0 || outw == 0 || previous.is_some_and(|p| p != inw) {
            return Err(FormatError::Shape);
        }
        previous = Some(outw);
        total = inw
            .checked_mul(outw)
            .and_then(|w| w.checked_add(outw))
            .and_then(|n| total.checked_add(n))
            .filter(|t| *t <= MAX_PARAMETERS)
            .ok_or(FormatError::TooLarge)?;
        shapes.push((inw, outw, act));
    }
    let data_start = 8 + 9 * count;
    if bytes.len() != data_start + 8 * total {
        return Err(FormatError::Size);
    }
    let mut cursor = data_start;
    let mut next = || {
        let b = &bytes[cursor..cursor + 8];
        cursor += 8;
        f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
    };
    let mut layers = Vec::with_capacity(count);
    let mut acts = Vec::with_capacity(count);
    for (inw, outw, act) in shapes {
        let weight: Vec<f64> = (0..inw * outw).map(|_| next()).collect();
        let bias: Vec<f64> = (0..outw).map(|_| next()).collect();
        if weight.iter().chain(&bias).any(|v| !v.is_finite()) {
            return Err(FormatError::NotFinite);
        }
        layers.push(Linear::new(inw, outw, weight, bias).map_err(|_| FormatError::Shape)?);
        acts.push(act);
    }
    Mlp::new(layers, acts).map_err(|_| FormatError::Shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::init::initialize;
    use crate::mlp::Scratch;
    use crate::policy::Policy;
    use robot_types::rng::{Rng, SeedStream};

    fn net() -> Mlp {
        let mut rng = Rng::new(SeedStream::new(4, 0));
        initialize(
            &[5, 8, 3],
            vec![Activation::Tanh, Activation::Identity],
            &mut rng,
        )
        .unwrap()
    }

    #[test]
    fn a_network_round_trips_bit_exactly_and_acts_identically() {
        let a = net();
        let b = decode(&encode(&a)).unwrap();
        assert_eq!(a, b);
        let (mut ya, mut yb) = ([0.0; 3], [0.0; 3]);
        let mut s = Scratch::new(a.scratch_len());
        let x = [0.1, -0.2, 0.3, 0.4, -0.5];
        a.act(&x, &mut ya, &mut s).unwrap();
        b.act(&x, &mut yb, &mut s).unwrap();
        assert_eq!(ya, yb);
    }

    #[test]
    fn an_unknown_operator_fails_at_load_naming_it() {
        let mut bytes = encode(&net());
        bytes[8 + 9 + 8] = 9; // second layer's activation code
        assert_eq!(
            decode(&bytes),
            Err(FormatError::UnsupportedOperator { layer: 1, code: 9 })
        );
    }

    #[test]
    fn truncation_padding_and_non_finite_values_are_refused() {
        let bytes = encode(&net());
        assert_eq!(decode(&bytes[..bytes.len() - 1]), Err(FormatError::Size));
        let mut padded = bytes.clone();
        padded.push(0);
        assert_eq!(decode(&padded), Err(FormatError::Size));
        let mut nan = bytes.clone();
        let at = 8 + 18;
        nan[at..at + 8].copy_from_slice(&f64::NAN.to_le_bytes());
        assert_eq!(decode(&nan), Err(FormatError::NotFinite));
        let mut big = bytes;
        big[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        big[17..21].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            decode(&big),
            Err(FormatError::TooLarge | FormatError::Shape)
        ));
    }
}

//! A recurrent policy: one GRU cell and a linear head, for SPECS.md
//! Section 9.2's "recurrent policy support when needed" and Section 7.4's
//! "recurrent-state specification".
//!
//! The cell is the standard gated recurrent unit:
//!
//! ```text
//! z  = sigmoid(Wz x + Uz h + bz)
//! r  = sigmoid(Wr x + Ur h + br)
//! n  = tanh(Wn x + bn + r * (Un h + bun))
//! h' = (1 - z) * n + z * h
//! y  = Wo h' + bo
//! ```
//!
//! State is explicit and owned by the policy. [`GruPolicy::reset`] zeroes it
//! (Section 10.3: "Recurrent state must be reset or transformed by an
//! explicit contract"); [`GruPolicy::set_state`] carries state in only at
//! the exact declared width. [`GruPolicy::spec`] reports the
//! `robot_types::bundle::RecurrentStateSpec` a bundle records. Every buffer
//! is sized at construction, so [`GruPolicy::step`] does not allocate.
//!
//! This is inference only: training a recurrent policy needs
//! backpropagation through time, which `robot-learn` does not implement,
//! because no task in this workspace has yet needed memory.

use robot_types::bundle::{RecurrentStatePolicy, RecurrentStateSpec};

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Why a GRU could not be built or stepped.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GruError {
    /// A parameter array has the wrong length.
    Shape(&'static str),
    /// A parameter is non-finite.
    NotFinite,
    /// An input, output or state slice has the wrong width.
    Width {
        /// Expected.
        expected: usize,
        /// Found.
        found: usize,
    },
}

impl core::fmt::Display for GruError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Shape(w) => write!(f, "GRU parameter {w} has the wrong length"),
            Self::NotFinite => f.write_str("non-finite GRU parameter"),
            Self::Width { expected, found } => write!(f, "width {found}, expected {expected}"),
        }
    }
}

impl core::error::Error for GruError {}

/// GRU parameters, row-major, gates stacked `[z; r; n]`.
#[derive(Clone, PartialEq, Debug)]
pub struct GruParams {
    /// Input width.
    pub input: usize,
    /// Hidden width.
    pub hidden: usize,
    /// Output width.
    pub output: usize,
    /// `3H x I` input weights.
    pub w: Vec<f64>,
    /// `3H x H` recurrent weights.
    pub u: Vec<f64>,
    /// `3H` input biases.
    pub b: Vec<f64>,
    /// `3H` recurrent biases.
    pub bu: Vec<f64>,
    /// `O x H` head weights.
    pub wo: Vec<f64>,
    /// `O` head biases.
    pub bo: Vec<f64>,
}

/// A GRU policy with its state.
#[derive(Clone, PartialEq, Debug)]
pub struct GruPolicy {
    p: GruParams,
    h: Vec<f64>,
    gx: Vec<f64>,
    gh: Vec<f64>,
    on_activation: RecurrentStatePolicy,
}

impl GruPolicy {
    /// Validates `params`; the state starts at zero.
    pub fn new(params: GruParams, on_activation: RecurrentStatePolicy) -> Result<Self, GruError> {
        let (i, h, o) = (params.input, params.hidden, params.output);
        for (len, want, name) in [
            (params.w.len(), 3 * h * i, "w"),
            (params.u.len(), 3 * h * h, "u"),
            (params.b.len(), 3 * h, "b"),
            (params.bu.len(), 3 * h, "bu"),
            (params.wo.len(), o * h, "wo"),
            (params.bo.len(), o, "bo"),
        ] {
            if len != want {
                return Err(GruError::Shape(name));
            }
        }
        let all = [
            &params.w, &params.u, &params.b, &params.bu, &params.wo, &params.bo,
        ];
        if all.iter().any(|v| v.iter().any(|x| !x.is_finite())) {
            return Err(GruError::NotFinite);
        }
        Ok(Self {
            h: vec![0.0; h],
            gx: vec![0.0; 3 * h],
            gh: vec![0.0; 3 * h],
            p: params,
            on_activation,
        })
    }

    /// The recurrent-state specification a bundle records.
    #[must_use]
    pub fn spec(&self) -> RecurrentStateSpec {
        RecurrentStateSpec::Recurrent {
            width: self.p.hidden,
            on_activation: self.on_activation,
        }
    }

    /// Zeroes the state.
    pub fn reset(&mut self) {
        self.h.iter_mut().for_each(|v| *v = 0.0);
    }

    /// The current state.
    #[must_use]
    pub fn state(&self) -> &[f64] {
        &self.h
    }

    /// Carries in a predecessor's state of exactly the declared width.
    pub fn set_state(&mut self, h: &[f64]) -> Result<(), GruError> {
        if h.len() != self.h.len() {
            return Err(GruError::Width {
                expected: self.h.len(),
                found: h.len(),
            });
        }
        self.h.copy_from_slice(h);
        Ok(())
    }

    /// One step: updates the state from `x` and writes the action.
    pub fn step(&mut self, x: &[f64], out: &mut [f64]) -> Result<(), GruError> {
        let (i, h, o) = (self.p.input, self.p.hidden, self.p.output);
        if x.len() != i {
            return Err(GruError::Width {
                expected: i,
                found: x.len(),
            });
        }
        if out.len() != o {
            return Err(GruError::Width {
                expected: o,
                found: out.len(),
            });
        }
        for r in 0..3 * h {
            self.gx[r] = self.p.b[r] + (0..i).map(|c| self.p.w[r * i + c] * x[c]).sum::<f64>();
            self.gh[r] =
                self.p.bu[r] + (0..h).map(|c| self.p.u[r * h + c] * self.h[c]).sum::<f64>();
        }
        for k in 0..h {
            let z = sigmoid(self.gx[k] + self.gh[k]);
            let r = sigmoid(self.gx[h + k] + self.gh[h + k]);
            let n = (self.gx[2 * h + k] + r * self.gh[2 * h + k]).tanh();
            self.h[k] = (1.0 - z) * n + z * self.h[k];
        }
        for (k, y) in out.iter_mut().enumerate() {
            *y = self.p.bo[k]
                + (0..h)
                    .map(|c| self.p.wo[k * h + c] * self.h[c])
                    .sum::<f64>();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One unit that integrates its input: the update gate is driven near
    /// zero (take the candidate) and the candidate is `tanh(x + h)`.
    fn integrator() -> GruPolicy {
        GruPolicy::new(
            GruParams {
                input: 1,
                hidden: 1,
                output: 1,
                w: vec![0.0, 0.0, 1.0],
                u: vec![0.0, 0.0, 1.0],
                b: vec![-20.0, 20.0, 0.0],
                bu: vec![0.0, 0.0, 0.0],
                wo: vec![1.0],
                bo: vec![0.0],
            },
            RecurrentStatePolicy::ResetToZero,
        )
        .unwrap()
    }

    #[test]
    fn the_output_remembers_past_inputs_and_reset_forgets_them() {
        let mut g = integrator();
        let mut y = [0.0];
        g.step(&[0.2], &mut y).unwrap();
        let first = y[0];
        g.step(&[0.0], &mut y).unwrap();
        // Zero input, but the state carries the earlier 0.2.
        assert!((y[0] - first.tanh()).abs() < 1e-6, "{y:?}");
        g.reset();
        g.step(&[0.0], &mut y).unwrap();
        assert!(y[0].abs() < 1e-6);
    }

    #[test]
    fn the_spec_and_state_carry_are_width_checked() {
        let mut g = integrator();
        assert_eq!(g.spec().width(), 1);
        assert!(g.set_state(&[0.5]).is_ok());
        assert_eq!(g.state(), &[0.5]);
        assert_eq!(
            g.set_state(&[0.5, 0.1]),
            Err(GruError::Width {
                expected: 1,
                found: 2
            })
        );
    }

    #[test]
    fn malformed_parameters_are_refused() {
        let mut p = integrator().p;
        p.u.pop();
        assert_eq!(
            GruPolicy::new(p, RecurrentStatePolicy::ResetToZero),
            Err(GruError::Shape("u"))
        );
        let mut p = integrator().p;
        p.b[0] = f64::NAN;
        assert_eq!(
            GruPolicy::new(p, RecurrentStatePolicy::ResetToZero),
            Err(GruError::NotFinite)
        );
    }
}

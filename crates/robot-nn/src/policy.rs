//! The plug-in seam for policies: anything that maps a normalized
//! observation to an action can drive [`ControlLoop`] without that loop
//! knowing its concrete type.
//!
//! [`Policy`] is the deterministic contract every policy satisfies;
//! [`Mlp`] is its first implementor. [`StochasticPolicy`] is an extension
//! for policies that can also draw an action from a distribution. It is a
//! separate trait so that deterministic inference never has to carry a
//! sampling path, and so that a policy with no notion of randomness does not
//! have to pretend to have one.
//!
//! Both traits write into caller-supplied buffers and a caller-supplied
//! [`Scratch`], so an implementation that honours them touches no allocator
//! on the control path (SPECS.md Section 5).
//!
//! [`ControlLoop`]: https://docs.rs/robot-control

use crate::mlp::{Mlp, MlpError, Scratch};
use robot_types::rng::Rng;

/// A deterministic map from observation to action.
pub trait Policy {
    /// Width of the observation vector this policy consumes.
    fn input_width(&self) -> usize;

    /// Width of the action vector this policy produces.
    fn output_width(&self) -> usize;

    /// The scratch capacity [`Scratch::new`] must be built with before
    /// [`Policy::act`] is called.
    fn scratch_len(&self) -> usize;

    /// Writes the action for `observation` into `action`.
    fn act(
        &self,
        observation: &[f64],
        action: &mut [f64],
        scratch: &mut Scratch,
    ) -> Result<(), MlpError>;
}

/// A [`Policy`] that can also sample an action from its distribution.
pub trait StochasticPolicy: Policy {
    /// Writes a sampled action for `observation` into `action`, drawing
    /// noise from `rng`, and returns the log-probability of that action.
    fn sample_action(
        &self,
        observation: &[f64],
        rng: &mut Rng,
        action: &mut [f64],
        scratch: &mut Scratch,
    ) -> Result<f64, MlpError>;
}

impl Policy for Mlp {
    fn input_width(&self) -> usize {
        Mlp::input_width(self)
    }

    fn output_width(&self) -> usize {
        Mlp::output_width(self)
    }

    fn scratch_len(&self) -> usize {
        Mlp::scratch_len(self)
    }

    fn act(
        &self,
        observation: &[f64],
        action: &mut [f64],
        scratch: &mut Scratch,
    ) -> Result<(), MlpError> {
        self.forward(observation, action, scratch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::{Activation, Linear};

    fn identity_mlp() -> Mlp {
        let layer = Linear::new(2, 2, vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 0.0]).unwrap();
        Mlp::new(vec![layer], vec![Activation::Identity]).unwrap()
    }

    #[test]
    fn mlp_act_matches_forward() {
        let net = identity_mlp();
        let obs = [0.25, -0.5];
        let mut via_forward = [0.0; 2];
        let mut via_trait = [0.0; 2];
        let mut s1 = Scratch::new(net.scratch_len());
        let mut s2 = Scratch::new(Policy::scratch_len(&net));
        net.forward(&obs, &mut via_forward, &mut s1).unwrap();
        Policy::act(&net, &obs, &mut via_trait, &mut s2).unwrap();
        assert_eq!(via_forward, via_trait);
    }

    #[test]
    fn mlp_reports_its_widths_through_the_trait() {
        let net = identity_mlp();
        assert_eq!(Policy::input_width(&net), 2);
        assert_eq!(Policy::output_width(&net), 2);
        assert_eq!(Policy::scratch_len(&net), net.scratch_len());
    }
}

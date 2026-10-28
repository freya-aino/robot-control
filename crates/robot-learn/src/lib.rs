//! Rollouts, authored autodiff, the PPO baseline and the experience stores
//! of SPECS.md Sections 7.6 and 9.2.
//!
//! [`backprop`] is the authored backward pass for `robot_nn`'s closed
//! operator set (see `docs/decisions/0003-std-only-default-closure.md`).
//! [`optimizer`] is Adam, with its moment state exposed as a value so it can
//! be checkpointed (SPECS.md Section 9.2: "checkpoint/resume including
//! optimizer, RNG and normalizer state"). [`gae`] computes advantages
//! honoring the termination/truncation distinction. [`rollout`] is the
//! bounded on-policy buffer of Section 7.6. [`gaussian_policy`] and [`ppo`]
//! tie all of the above into a trainable agent.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod backprop;
pub mod checkpoint;
pub mod gae;
pub mod gaussian_policy;
pub mod optimizer;
pub mod ppo;
pub mod rollout;

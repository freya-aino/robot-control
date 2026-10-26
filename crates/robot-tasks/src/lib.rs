//! Tracking, balance and reference tasks with a versioned reward registry,
//! per SPECS.md Section 9.2 and Section 9.3's reward-term registry
//! extension point.
//!
//! [`reward::RewardRegistry`] is a name-keyed, build-time-constructed
//! registry over plain function pointers — Section 9.3's "no control-path
//! lookup" requirement needs no dynamic dispatch here, since the term set
//! is fixed per task and linked in. [`balance::BalanceTask`] is the first
//! concrete task, built on `robot-sim::env::PendulumEnv`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod balance;
pub mod reward;

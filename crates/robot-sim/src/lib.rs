//! Simulation interface, authored articulated physics, batching and
//! randomization, per SPECS.md Section 9.1.
//!
//! This revision delivers the two smallest qualification benchmarks Section
//! 9.1 and Appendix B.4 name first — "pendulum, actuator step response" and
//! "foot slip, push recovery" — as [`pendulum`] and [`contact`], plus the
//! [`env::Environment`] trait and a [`env::PendulumEnv`] that PLAN.md's
//! first executable backlog item 8 ("Train, checkpoint and reload a tiny
//! PPO policy in a Rust simulator task") can train against.
//!
//! # What this crate does not provide
//!
//! There is no multi-body articulated-dynamics engine here. SPECS.md
//! Section 9.1 requires qualifying Rapier's existing multibody-joint and
//! collision capabilities, or a verified alternative, before authoring one;
//! Appendix B.6 admits no candidate until its gates pass
//! (`docs/decisions/0003-std-only-default-closure.md`), and that
//! qualification work has not been performed in this revision. A humanoid
//! with coupled multi-joint dynamics is therefore out of scope here — this
//! is the honest statement PLAN.md's Phase 0 exit gate requires
//! ("Simulator throughput and fidelity limitations are documented"), not an
//! oversight. [`pendulum::Pendulum`] and [`contact::ContactBody`] are
//! single-body benchmarks, each qualified against an analytic or
//! energy-conservation reference, exactly as far as that honesty extends.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod contact;
pub mod env;
pub mod pendulum;

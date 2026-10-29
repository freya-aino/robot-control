//! Simulation interface, authored articulated physics, batching and
//! randomization, per SPECS.md Section 9.1.
//!
//! **See `SIMULATION_SPECS.md` at the root of this crate for the current
//! target**: GPU-bound, batched execution across thousands of
//! environments, variable fidelity as a first-class interface, a
//! Rust-native GPU-compute path, and a backend-agnostic boundary so
//! every other crate depends on [`env::Environment`] and never on a
//! physics-backend-specific type. That document is grounded in direct
//! research against MuJoCo's own source/docs, MJX, Brax, CubeCL and
//! `wgpu`, not a restatement of this module's existing scope.
//!
//! This revision delivers the two smallest qualification benchmarks Section
//! 9.1 and Appendix B.4 name first — "pendulum, actuator step response" and
//! "foot slip, push recovery" — as [`pendulum`] and [`contact`], plus the
//! [`env::Environment`] trait and a [`env::PendulumEnv`] that PLAN.md's
//! first executable backlog item 8 ("Train, checkpoint and reload a tiny
//! PPO policy in a Rust simulator task") can train against. [`spatial`]
//! and [`articulated`] add an authored multi-body dynamics solver (the
//! Articulated-Body and Recursive Newton-Euler algorithms) over any
//! `robot_model::RobotProfile`'s joint tree, including the vendored
//! Unitree G1 fixture.
//!
//! # What this crate does not provide
//!
//! [`articulated::ArticulatedBody`] now handles a floating base too:
//! `robot_model::joint::JointKind::Free` lets a profile's root attach to
//! an explicit `world` link with six unconstrained degrees of freedom,
//! and `robot_model::fixtures::unitree_g1_29dof_floating_base` is a real
//! floating-base G1 that genuinely falls under gravity (`robot sim
//! multibody --floating` in `robot-cli` demonstrates it). What is still
//! missing is **any ground-contact force resolution at all** — nothing
//! stops that falling body from passing through a floor, because there
//! is no floor, no collision geometry, and no contact solver yet. A
//! humanoid can fall here; it cannot stand, balance, or walk, because
//! nothing yet resists it touching the ground. That is a deliberate,
//! documented limitation (see [`articulated`]'s own doc comment), not an
//! oversight — resolving ground contact is the named next increment.
//!
//! SPECS.md Section 9.1 separately asks to qualify Rapier's multibody and
//! collision capabilities, or a verified alternative, before authoring
//! one; Appendix B.6 admits no candidate until its gates pass
//! (`docs/decisions/0003-std-only-default-closure.md`), and that
//! qualification work has not been performed. [`articulated`] is this
//! project's established answer to exactly that situation elsewhere —
//! `robot_types::math`'s authored linear algebra, `motion-retarget`'s
//! damped-least-squares IK baseline — an authored, independently
//! validated interim, not a claim that qualification is unnecessary.
//! [`pendulum::Pendulum`] and [`contact::ContactBody`] remain the
//! simpler single-body benchmarks they always were, each qualified
//! against its own analytic or energy-conservation reference.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod articulated;
pub mod backend;
pub mod contact;
pub mod env;
pub mod fidelity;
pub mod multibody_env;
pub mod oracle;
pub mod pendulum;
pub mod spatial;

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
//! [`articulated::ArticulatedBody`] is **fixed-base**: the profile's root
//! link is rigidly welded to an inertial world frame, and there is no
//! ground-contact force resolution. A humanoid cannot yet fall over,
//! balance, or walk here — only its joints can move under gravity and
//! applied torque while its base stays put. That is a deliberate,
//! documented limitation (see [`articulated`]'s own doc comment for why
//! it is also exactly what an imported `RobotProfile`'s root already
//! represents, since the URDF importer rejects a `floating` root joint
//! by name), not an oversight: letting the base itself move freely and
//! resolving foot contact are the named next increments.
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
pub mod contact;
pub mod env;
pub mod pendulum;
pub mod spatial;

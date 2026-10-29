//! CubeCL GPU backend for robot-sim simulation algorithms.
//!
//! This crate provides GPU compute kernels using CubeCL (WGPU backend) mirroring
//! the pure CPU simulation algorithms in `robot-sim`. Implemented: semi-implicit
//! Euler, implicit-damping Euler and equality-only dense solves (one thread
//! per environment, `nv,nrow <= 6`), joint spring/damper forces, affine
//! actuator force, joint-equality residual/Jacobian, `solimp` impedance with
//! its derivative `impP`, a named pendulum RK4 step and plane-sphere contact.
//! Not implemented: the general RK4 (closure-based), the contact/limit/friction
//! solver or systems above 6 degrees of freedom/rows. Each launch runs one
//! thread per element (or environment for dense solves) in `ceil(n / 256)` workgroups.
//!
//! # Unsafe Policy
//!
//! The crate enforces `#![deny(unsafe_code)]`. The only `unsafe` is the
//! CubeCL buffer binding inside a private launch module, reached only through
//! the public wrappers, which validate every slice length and the device buffer
//! size before the `unsafe` block runs. Default builds without the `cubecl`
//! feature contain zero unsafe code and no external GPU dependencies.
//!
//! The launch module is crate-private, so callers cannot skip those checks:
//!
//! ```compile_fail
//! let _ = robot_sim_gpu::launch::backend::launch_joint_spring;
//! ```

#![deny(unsafe_code)]

pub mod constraint;
pub mod contact;
pub mod error;
pub mod integrators;
pub mod passive;

mod launch;

pub use constraint::{
    gpu_constraint_impedance, gpu_constraint_impedance_with_derivative, gpu_joint_equality_eval,
    gpu_solve_equality,
};
pub use contact::{gpu_plane_sphere_contact, GpuPlaneSphereContact};
pub use error::GpuSimError;
pub use integrators::{
    gpu_implicit_damping_euler_step, gpu_rk4_pendulum_step, gpu_semi_implicit_euler_step,
};
pub use launch::{
    check_batch_size, check_equality_batch, check_implicit_batch, equality_resources,
    rk4_resources, EqualityResources, Rk4Resources, EQUALITY_BINDINGS, MAX_DOF, RK4_BINDINGS,
};
pub use passive::{gpu_affine_actuator_forces, gpu_joint_damper_forces, gpu_joint_spring_forces};

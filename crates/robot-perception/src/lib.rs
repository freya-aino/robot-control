//! Floating-base state estimation with covariance consistency, per
//! SPECS.md Section 11.2.
//!
//! "The estimation module in `robot-perception` must specify
//! orientation/error-state conventions, IMU bias dynamics, contact
//! assumptions, covariance propagation and measurement gating." This crate
//! states each of those explicitly rather than leaving it implicit:
//!
//! | Requirement | Where |
//! | --- | --- |
//! | Orientation/error-state convention | [`attitude`]'s module docs: world-from-body quaternion, multiplicative body-frame error |
//! | IMU bias dynamics | [`attitude::AttitudeFilter`]'s bias state, modeled as a random walk |
//! | Contact assumptions | [`contact::ContactClassifier`] (site-level hysteresis) and [`attitude`]'s explicit specific-force magnitude gate, documented as a proxy rather than true contact-conditioned gating |
//! | Covariance propagation | [`linear_kf::KalmanFilter::predict`] |
//! | Measurement gating | [`linear_kf::KalmanFilter::gated_update`], a chi-squared Mahalanobis gate |
//!
//! [`linear_kf::KalmanFilter`] is the authored baseline this project runs
//! until an Appendix B.2.9 candidate (`multicalc`, `numeris`, `typed_kalman`,
//! `kfilter`) clears the Appendix B.6 gates — see
//! `docs/decisions/0003-std-only-default-closure.md`. Its hand-derived
//! measurement Jacobian is validated against central finite differences in
//! `attitude`'s test suite, per Section 11.2's requirement that "Autodiff
//! Jacobians are optional and require validation, as do hand-derived
//! Jacobians."

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod attitude;
pub mod contact;
pub mod linear_kf;

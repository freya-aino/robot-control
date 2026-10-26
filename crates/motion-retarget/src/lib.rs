//! Robot mapping, constrained IK and temporal refinement, per SPECS.md
//! Section 8.3.
//!
//! [`ik::solve`] is the "feasible non-QP baseline" Appendix C's C-09 keeps
//! open while no QP solver has cleared Appendix B.6
//! (`docs/decisions/0003-std-only-default-closure.md`). It handles
//! end-effector position targets only; orientation targets, posture
//! priors, planted-foot constraints and temporal refinement — all named in
//! Section 8.3 — are not implemented in this revision.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod ik;

//! Retargeting, per SPECS.md Section 8.2 item 7 and Section 8.3.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod collision;
pub mod config;
pub mod ik;
pub mod qp;
pub mod retarget;
pub mod temporal;

#[cfg(test)]
mod test_support;

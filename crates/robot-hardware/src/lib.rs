//! Device traits, loopback and replay devices, and the protocol-adapter
//! boundary of SPECS.md Section 7.1.
//!
//! PLAN.md Phase 5: "Implement a hardware adapter first against a loopback
//! device, then a replay device, then an actual documented protocol."
//! [`device::HardwareAdapter`] is the trait all three share;
//! [`loopback::LoopbackDevice`] and [`replay::ReplayDevice`] are the first
//! two. No protocol-specific adapter exists in this revision — there is no
//! designated first hardware target yet (C-02 is open), and SPECS.md
//! Section 1.1 requires that any such adapter be "authorship against a cited
//! historical reference" from the retired `unitree/` commit, with every
//! field treated as "a hypothesis requiring hardware verification," not a
//! port.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod device;
pub mod loopback;
pub mod replay;

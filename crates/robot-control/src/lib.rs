//! The deterministic control runtime of SPECS.md Section 5, built from the
//! independently qualified pieces this workspace already has:
//! `robot-nn` for allocation-free inference, `robot-supervisor` for the
//! last-stage safety check, and `robot-hardware` for the device boundary.
//!
//! [`control_loop::ControlLoop`] is the one structure: every allocation
//! happens in [`control_loop::ControlLoop::new`], and
//! [`control_loop::ControlLoop::tick`] touches only buffer *contents*, never
//! their capacity. [`timing::TickTiming`] and [`timing::TimingHistory`] keep
//! the four timing quantities of Section 5 — compute duration,
//! sensor-to-command age, decision age, deadline margin — as separate
//! fields, per that section's explicit prohibition on collapsing them.
//!
//! This revision implements the policy tier only. The tactical tier
//! (`robot-skills`) is a declared dependency but not yet wired into
//! `tick`'s decision path — seeing this stated plainly here is the point of
//! `control_loop`'s own module documentation.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod control_loop;
pub mod operation;
pub mod timing;

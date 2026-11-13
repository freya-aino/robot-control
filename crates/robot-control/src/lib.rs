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
//! Two tiers run in one loop: the policy tier in `tick`, and the tactical
//! tier of [`tactical`] as a `LoopOperation` at its own rate, whose
//! proposals reach the policy tier as expiring, timestamped records.
//! Temperature and contact feedback reach the supervisor through
//! `ControlLoop::with_sensors`, contacts classified by `robot-perception`.
//! [`incident`] is the reserved safety-event channel.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod control_loop;
pub mod incident;
pub mod operation;
pub mod tactical;
pub mod timing;

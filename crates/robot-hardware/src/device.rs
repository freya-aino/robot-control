//! The hardware adapter trait every device — loopback, replay, or an actual
//! protocol — implements.
//!
//! PLAN.md Phase 5: "Implement a hardware adapter first against a loopback
//! device, then a replay device, then an actual documented protocol." All
//! three share this one trait, so the runtime code that drives them is
//! written once and is itself untested-by-construction against which
//! concrete device is behind it.
//!
//! SPECS.md Section 5 binds every method here: "no blocking external
//! dependencies," "no heap allocation after activation." Every signature
//! takes caller-owned buffers rather than returning an owned `Vec`.

use core::fmt;
use robot_types::time::Timestamp;

/// Why an adapter call failed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AdapterError {
    /// A buffer's length did not match the adapter's configured width.
    WidthMismatch {
        /// Width the adapter was configured with.
        expected: usize,
        /// Width supplied.
        found: usize,
    },
    /// A command was sent while transmission is disabled.
    ///
    /// SPECS.md Section 4, `hw-shadow`: "Transmit path disabled and the
    /// disable verified by loopback readback." This is the error a caller
    /// sees if it tries to transmit anyway, and the loopback device's
    /// readback is what confirms no command actually reached "the motor."
    TransmitDisabled,
    /// No new feedback was available (not itself a fault — a normal poll
    /// result between updates).
    NotReady,
    /// A replay device has no further recorded frames.
    ///
    /// Distinct from [`AdapterError::NotReady`], which is a transient "ask
    /// again next tick" condition: exhaustion is terminal for this device,
    /// and SPECS.md Section 4's `offline-replay` fault response is "Abort
    /// with a diagnostic; never fabricate missing samples" — this error is
    /// that diagnostic, not a retriable state.
    ReplayExhausted,
    /// The underlying transport reported a fault.
    TransportFault {
        /// A short, stable fault code.
        code: &'static str,
    },
}

impl fmt::Display for AdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WidthMismatch { expected, found } => {
                write!(f, "buffer width {found}, adapter configured for {expected}")
            }
            Self::TransmitDisabled => write!(f, "transmit path is disabled"),
            Self::NotReady => write!(f, "no new feedback is available"),
            Self::ReplayExhausted => write!(f, "replay has no further recorded frames"),
            Self::TransportFault { code } => write!(f, "transport fault: {code}"),
        }
    }
}

impl core::error::Error for AdapterError {}

/// One feedback sample from a device: the quantities `robot_types::episode`
/// needs to build a `JointSample` per joint, but produced by the adapter in
/// flat per-joint arrays rather than the richer episode record (which
/// `robot-control` assembles from this).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct FeedbackHeader {
    /// Sampling or arrival instant this feedback was stamped at.
    pub timestamp: Timestamp,
    /// Whether `timestamp` is a true sampling instant.
    pub is_sampling_instant: bool,
}

/// A hardware adapter: the boundary between the control process and a real
/// or simulated transport.
///
/// Every method is synchronous and non-blocking by contract — SPECS.md
/// Section 5 forbids blocking on the control path, and an implementation
/// that cannot honor that is not a conforming adapter, independent of
/// whether the trait itself can express the constraint.
pub trait HardwareAdapter {
    /// Number of command/feedback components this adapter exchanges.
    fn width(&self) -> usize;

    /// Sends a command. `positions` is in the semantics the adapter was
    /// configured for (position/velocity/torque); the adapter does not
    /// reinterpret it.
    ///
    /// Returns [`AdapterError::TransmitDisabled`] without touching any
    /// physical output when transmission is disabled — this is the behavior
    /// `hw-shadow` mode depends on.
    fn send_command(&mut self, command: &[f64], timestamp: Timestamp) -> Result<(), AdapterError>;

    /// Polls for new feedback, writing it into `positions` and returning the
    /// feedback's header.
    ///
    /// Returns `Ok(None)` when no new sample is available since the last
    /// poll (not an error: a device may run slower than the control loop).
    fn poll_feedback(
        &mut self,
        positions: &mut [f64],
    ) -> Result<Option<FeedbackHeader>, AdapterError>;

    /// Whether the transmit path is currently enabled.
    fn transmit_enabled(&self) -> bool;

    /// Enables or disables the transmit path. Disabling takes effect
    /// immediately for the next `send_command` call.
    fn set_transmit_enabled(&mut self, enabled: bool);

    /// Age since the last confirmed liveness signal from the transport
    /// (a feedback sample, an acknowledgement, or an explicit heartbeat),
    /// for the supervisor's communications-health check.
    fn link_age(&self, now: Timestamp) -> Option<robot_types::time::Duration>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapter_error_messages_name_the_specific_mismatch() {
        let e = AdapterError::WidthMismatch {
            expected: 3,
            found: 1,
        };
        assert!(e.to_string().contains('3'));
        assert!(e.to_string().contains('1'));
    }
}

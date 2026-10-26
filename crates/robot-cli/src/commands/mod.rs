//! One module per `robot` subcommand group that has a real implementation.

pub mod data;
pub mod eval;
pub mod motion;
pub mod sim;

/// Exit code for a command that ran and found a problem (a rejected
/// motion, a failed qualification run, bad input).
pub const EXIT_FAILURE: i32 = 1;
/// Exit code for a command this build does not implement yet.
pub const EXIT_NOT_IMPLEMENTED: i32 = 2;
/// Exit code for a malformed invocation (missing argument, bad flag).
pub const EXIT_USAGE: i32 = 64;

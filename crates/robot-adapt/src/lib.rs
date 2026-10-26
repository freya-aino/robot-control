//! System identification, history conditioning and bounded residuals, per
//! SPECS.md Section 11.3.
//!
//! [`sysid::identify`] implements the first, least-exposed row of Section
//! 11.3's ordered table: calibrated linear system identification.
//! History-conditioned adaptation, the RMA-class two-phase latent
//! identification, simulator correction and bounded residuals — every row
//! below system identification in that table — are not implemented here;
//! Section 11.3's own ordering requires system identification to be
//! "measured and found insufficient on the same robot" before any later
//! row is scheduled at all, so this crate's scope matches where the
//! project's own evidence currently stands.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod sysid;

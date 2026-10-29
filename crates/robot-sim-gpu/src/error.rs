//! Error types for GPU simulation operations and kernel launches.

/// Error conditions encountered during GPU operations or input validation.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum GpuSimError {
    /// A required feature (e.g. `cubecl`) is not enabled in this build.
    GpuDisabled,
    /// Vector or slice lengths do not match expected dimensions.
    DimensionMismatch {
        /// What buffer/slice had the mismatch.
        what: &'static str,
        /// Expected element count.
        expected: usize,
        /// Actual element count found.
        found: usize,
    },
    /// A floating point input is not finite (NaN or Inf).
    NonFinite {
        /// Name of the input containing non-finite values.
        what: &'static str,
    },
    /// An invalid parameter was passed (e.g. negative radius or non-positive dt).
    InvalidParameter {
        /// Description of the invalid parameter.
        what: &'static str,
    },
    /// Hardware / driver device acquisition or kernel launch failed.
    LaunchError(String),
}

impl core::fmt::Display for GpuSimError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::GpuDisabled => write!(
                f,
                "GPU execution requested but the 'cubecl' feature is not enabled"
            ),
            Self::DimensionMismatch {
                what,
                expected,
                found,
            } => {
                write!(
                    f,
                    "dimension mismatch for {what}: expected {expected}, found {found}"
                )
            }
            Self::NonFinite { what } => write!(f, "non-finite value found in {what}"),
            Self::InvalidParameter { what } => write!(f, "invalid parameter: {what}"),
            Self::LaunchError(msg) => write!(f, "GPU launch error: {msg}"),
        }
    }
}

impl core::error::Error for GpuSimError {}

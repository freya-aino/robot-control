//! GPU passive joint forces and affine actuators mirroring the reference simulator 3.17.0.
//!
//! Mirrors [`robot_sim::passive`].

use crate::error::GpuSimError;
use crate::launch::{backend, check_batch_size};

/// Computes linear joint spring forces on GPU: `out[i] = -(q[i] - q_ref[i]) * stiffness[i]`.
///
/// Mirrors [`robot_sim::passive::joint_spring_force`].
pub fn gpu_joint_spring_forces(
    q: &[f32],
    q_ref: &[f32],
    stiffness: &[f32],
    out: &mut [f32],
) -> Result<(), GpuSimError> {
    let n = q.len();
    if q_ref.len() != n {
        return Err(GpuSimError::DimensionMismatch {
            what: "q_ref",
            expected: n,
            found: q_ref.len(),
        });
    }
    if stiffness.len() != n {
        return Err(GpuSimError::DimensionMismatch {
            what: "stiffness",
            expected: n,
            found: stiffness.len(),
        });
    }
    if out.len() != n {
        return Err(GpuSimError::DimensionMismatch {
            what: "out",
            expected: n,
            found: out.len(),
        });
    }
    check_batch_size(n, 1)?;

    if q.iter().any(|x| !x.is_finite()) {
        return Err(GpuSimError::NonFinite { what: "q" });
    }
    if q_ref.iter().any(|x| !x.is_finite()) {
        return Err(GpuSimError::NonFinite { what: "q_ref" });
    }
    if stiffness.iter().any(|x| !x.is_finite()) {
        return Err(GpuSimError::NonFinite { what: "stiffness" });
    }

    backend::launch_joint_spring(q, q_ref, stiffness, out)
}

/// Computes linear joint damping forces on GPU: `out[i] = -qd[i] * damping[i]`.
///
/// Mirrors [`robot_sim::passive::joint_damper_force`].
pub fn gpu_joint_damper_forces(
    qd: &[f32],
    damping: &[f32],
    out: &mut [f32],
) -> Result<(), GpuSimError> {
    let n = qd.len();
    if damping.len() != n {
        return Err(GpuSimError::DimensionMismatch {
            what: "damping",
            expected: n,
            found: damping.len(),
        });
    }
    if out.len() != n {
        return Err(GpuSimError::DimensionMismatch {
            what: "out",
            expected: n,
            found: out.len(),
        });
    }
    check_batch_size(n, 1)?;

    if qd.iter().any(|x| !x.is_finite()) {
        return Err(GpuSimError::NonFinite { what: "qd" });
    }
    if damping.iter().any(|x| !x.is_finite()) {
        return Err(GpuSimError::NonFinite { what: "damping" });
    }

    backend::launch_joint_damper(qd, damping, out)
}

/// Computes affine actuator forces on GPU: `gain * ctrl + bias`.
///
/// Mirrors [`robot_sim::passive::affine_actuator_force`].
pub fn gpu_affine_actuator_forces(
    gain_prm: &[[f32; 3]],
    bias_prm: &[[f32; 3]],
    ctrl: &[f32],
    length: &[f32],
    velocity: &[f32],
    out: &mut [f32],
) -> Result<(), GpuSimError> {
    let n = ctrl.len();
    if gain_prm.len() != n {
        return Err(GpuSimError::DimensionMismatch {
            what: "gain_prm",
            expected: n,
            found: gain_prm.len(),
        });
    }
    if bias_prm.len() != n {
        return Err(GpuSimError::DimensionMismatch {
            what: "bias_prm",
            expected: n,
            found: bias_prm.len(),
        });
    }
    if length.len() != n {
        return Err(GpuSimError::DimensionMismatch {
            what: "length",
            expected: n,
            found: length.len(),
        });
    }
    if velocity.len() != n {
        return Err(GpuSimError::DimensionMismatch {
            what: "velocity",
            expected: n,
            found: velocity.len(),
        });
    }
    if out.len() != n {
        return Err(GpuSimError::DimensionMismatch {
            what: "out",
            expected: n,
            found: out.len(),
        });
    }
    check_batch_size(n, 3)?;

    if ctrl.iter().any(|x| !x.is_finite()) {
        return Err(GpuSimError::NonFinite { what: "ctrl" });
    }
    if length.iter().any(|x| !x.is_finite()) {
        return Err(GpuSimError::NonFinite { what: "length" });
    }
    if velocity.iter().any(|x| !x.is_finite()) {
        return Err(GpuSimError::NonFinite { what: "velocity" });
    }
    for (i, gp) in gain_prm.iter().enumerate() {
        if gp.iter().any(|x| !x.is_finite()) {
            return Err(GpuSimError::NonFinite { what: "gain_prm" });
        }
        if bias_prm[i].iter().any(|x| !x.is_finite()) {
            return Err(GpuSimError::NonFinite { what: "bias_prm" });
        }
    }

    let mut gain_flat = Vec::with_capacity(n * 3);
    let mut bias_flat = Vec::with_capacity(n * 3);
    for i in 0..n {
        gain_flat.extend_from_slice(&gain_prm[i]);
        bias_flat.extend_from_slice(&bias_prm[i]);
    }

    backend::launch_affine_actuator(&gain_flat, &bias_flat, ctrl, length, velocity, out)
}

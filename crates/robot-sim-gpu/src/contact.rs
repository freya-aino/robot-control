//! GPU plane-sphere collision contact detection mirroring MuJoCo 3.17.0.
//!
//! Mirrors [`robot_sim::contact::plane_sphere_contact`].

use crate::error::GpuSimError;
use crate::launch::{backend, check_batch_size, check_lengths};

/// One plane-sphere contact point produced on GPU.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct GpuPlaneSphereContact {
    /// Distance (negative for penetration, zero for exact touch, positive for margin zone).
    pub dist: f32,
    /// Normal pointing out of the plane (+z = [0, 0, 1]).
    pub normal: [f32; 3],
    /// Contact position in world frame.
    pub pos: [f32; 3],
}

/// Detects plane-sphere collision contacts on GPU for an array of spheres.
///
/// Mirrors [`robot_sim::contact::plane_sphere_contact`].
///
/// # Errors
///
/// Returns [`GpuSimError::DimensionMismatch`] if slice lengths differ,
/// [`GpuSimError::NonFinite`] if any float is NaN/Inf, or
/// [`GpuSimError::InvalidParameter`] if any radius is negative.
pub fn gpu_plane_sphere_contact(
    centers: &[[f32; 3]],
    radii: &[f32],
    plane_height: f32,
    margin: f32,
    contacts: &mut [Option<GpuPlaneSphereContact>],
) -> Result<(), GpuSimError> {
    let n = radii.len();
    check_lengths(
        n,
        &[("centers", centers.len()), ("contacts", contacts.len())],
    )?;
    check_batch_size(n, 3)?;

    if !plane_height.is_finite() {
        return Err(GpuSimError::NonFinite {
            what: "plane_height",
        });
    }
    if !margin.is_finite() {
        return Err(GpuSimError::NonFinite { what: "margin" });
    }
    for c in centers {
        if c.iter().any(|x| !x.is_finite()) {
            return Err(GpuSimError::NonFinite { what: "centers" });
        }
    }
    for &r in radii {
        if !r.is_finite() {
            return Err(GpuSimError::NonFinite { what: "radii" });
        }
        if r < 0.0 {
            return Err(GpuSimError::InvalidParameter {
                what: "sphere radius cannot be negative",
            });
        }
    }

    let mut centers_flat = Vec::with_capacity(n * 3);
    for c in centers {
        centers_flat.extend_from_slice(c);
    }

    let mut valid = vec![0u32; n];
    let mut dists = vec![0.0f32; n];
    let mut pos_flat = vec![0.0f32; 3 * n];

    backend::launch_plane_sphere_contact(
        &centers_flat,
        radii,
        plane_height,
        margin,
        &mut valid,
        &mut dists,
        &mut pos_flat,
    )?;

    for i in 0..n {
        if valid[i] != 0 {
            contacts[i] = Some(GpuPlaneSphereContact {
                dist: dists[i],
                normal: [0.0, 0.0, 1.0],
                pos: [pos_flat[i * 3], pos_flat[i * 3 + 1], pos_flat[i * 3 + 2]],
            });
        } else {
            contacts[i] = None;
        }
    }

    Ok(())
}

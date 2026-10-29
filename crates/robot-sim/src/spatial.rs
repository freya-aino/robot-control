//! Spatial vector algebra: the 6D motion/force representation the
//! multi-body dynamics solver in [`crate::articulated`] is built on.
//!
//! Follows the standard (Featherstone, *Rigid Body Dynamics Algorithms*)
//! convention: a 6D *motion* vector packs (angular, linear) velocity or
//! acceleration of a body-fixed frame's own origin, expressed in that
//! frame's own axes; a 6D *force* vector packs (torque, force) acting
//! about that same origin, in the same axes. Both are plain `[f64; 6]`
//! arrays laid out as `[angular_x, angular_y, angular_z, linear_x,
//! linear_y, linear_z]`, so a motion value and a force value share a type
//! but are transformed by different functions ([`transform_motion`] and
//! [`transform_force`]), which is where the sign conventions differ.
//!
//! Every value in this module is a plain array with an explicit formula:
//! 3-vectors are `[f64; 3]`, rotations are row-major `[f64; 9]`, and a
//! spatial inertia is a row-major symmetric `[f64; 36]`. Nothing here
//! imports `robot_types` or `robot_model`; the boundary adapters in
//! [`crate::articulated`] convert those types into these arrays once.
//!
//! The 6x6 spatial inertia is kept as a general matrix rather than a
//! (mass, COM, inertia) triple: the articulated-body backward pass
//! produces "effective" inertias that are no longer expressible in that
//! simpler form once a joint's own degrees of freedom have been eliminated
//! from it.

/// `a + b` for 3-vectors.
#[must_use]
pub fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// `a - b` for 3-vectors.
#[must_use]
pub fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// `s * a` for a 3-vector.
#[must_use]
pub fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Dot product of two 3-vectors.
#[must_use]
pub fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product `a x b`.
#[must_use]
pub fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// `R v` for a row-major 3x3 rotation.
#[must_use]
pub fn mat3_mul_vec(r: [f64; 9], v: [f64; 3]) -> [f64; 3] {
    [
        r[0] * v[0] + r[1] * v[1] + r[2] * v[2],
        r[3] * v[0] + r[4] * v[1] + r[5] * v[2],
        r[6] * v[0] + r[7] * v[1] + r[8] * v[2],
    ]
}

/// `R^T v` for a row-major 3x3 rotation: the inverse rotation applied to `v`.
#[must_use]
pub fn mat3_transpose_mul_vec(r: [f64; 9], v: [f64; 3]) -> [f64; 3] {
    [
        r[0] * v[0] + r[3] * v[1] + r[6] * v[2],
        r[1] * v[0] + r[4] * v[1] + r[7] * v[2],
        r[2] * v[0] + r[5] * v[1] + r[8] * v[2],
    ]
}

/// Splits a 6-vector into its first three (angular/torque) and last three
/// (linear/force) components.
#[must_use]
pub fn split6(v: [f64; 6]) -> ([f64; 3], [f64; 3]) {
    ([v[0], v[1], v[2]], [v[3], v[4], v[5]])
}

/// Joins an angular/torque part and a linear/force part into a 6-vector.
#[must_use]
pub fn join6(top: [f64; 3], bottom: [f64; 3]) -> [f64; 6] {
    [top[0], top[1], top[2], bottom[0], bottom[1], bottom[2]]
}

/// `a + b` for 6-vectors (motions or forces).
#[must_use]
pub fn add6(a: [f64; 6], b: [f64; 6]) -> [f64; 6] {
    [
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
        a[3] + b[3],
        a[4] + b[4],
        a[5] + b[5],
    ]
}

/// `a - b` for 6-vectors.
#[must_use]
pub fn sub6(a: [f64; 6], b: [f64; 6]) -> [f64; 6] {
    [
        a[0] - b[0],
        a[1] - b[1],
        a[2] - b[2],
        a[3] - b[3],
        a[4] - b[4],
        a[5] - b[5],
    ]
}

/// `s * a` for a 6-vector.
#[must_use]
pub fn scale6(a: [f64; 6], s: f64) -> [f64; 6] {
    [a[0] * s, a[1] * s, a[2] * s, a[3] * s, a[4] * s, a[5] * s]
}

/// Dot product of two 6-vectors. Between a [`transform_force`]-style force
/// and a motion it is the instantaneous power, which is frame-invariant.
#[must_use]
pub fn dot6(a: [f64; 6], b: [f64; 6]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3] + a[4] * b[4] + a[5] * b[5]
}

/// Transforms a motion vector expressed in a parent frame into the same
/// motion vector expressed in a child frame. The child's pose in the
/// parent frame is given as rotation `rot` (row-major) and translation
/// `trans`.
///
/// Derivation: for a point rigidly fixed in the parent body at the child's
/// origin (position `p` in parent axes), the rigid-body velocity field gives
/// `v_parent_at_p = v_linear + v_angular x p`; rotating both parts into the
/// child's axes by `R^T` gives the result.
#[must_use]
pub fn transform_motion(rot: [f64; 9], trans: [f64; 3], v: [f64; 6]) -> [f64; 6] {
    let (w, l) = split6(v);
    join6(
        mat3_transpose_mul_vec(rot, w),
        mat3_transpose_mul_vec(rot, add3(l, cross3(w, trans))),
    )
}

/// Transforms a force vector expressed in a child frame into the same force
/// vector expressed in the parent frame, given the child's pose in the
/// parent frame (`rot`, `trans`).
///
/// The dual of [`transform_motion`]: for any motion `v` in the parent frame
/// and force `f` in the child frame, `dot6(transform_motion(t, v), f) ==
/// dot6(v, transform_force(t, f))` (power is frame-invariant).
/// `spatial_duality_preserves_power` checks this numerically.
#[must_use]
pub fn transform_force(rot: [f64; 9], trans: [f64; 3], f: [f64; 6]) -> [f64; 6] {
    let (n, force) = split6(f);
    let rotated_force = mat3_mul_vec(rot, force);
    join6(
        add3(mat3_mul_vec(rot, n), cross3(trans, rotated_force)),
        rotated_force,
    )
}

/// The spatial "motion cross product" `v x* w` of two motion vectors, itself
/// a motion vector (Featherstone's `ad` operator). Used for the Coriolis and
/// bias acceleration term contributed by a joint's own velocity.
#[must_use]
pub fn cross_motion_motion(v: [f64; 6], w: [f64; 6]) -> [f64; 6] {
    let (va, vl) = split6(v);
    let (wa, wl) = split6(w);
    join6(cross3(va, wa), add3(cross3(va, wl), cross3(vl, wa)))
}

/// The spatial "force cross product" `v x*f f`, the dual of
/// [`cross_motion_motion`]: a motion vector crossed with a force vector
/// produces a force vector. This is the velocity-product (Coriolis and
/// centrifugal) bias force of a rotating, translating rigid body.
#[must_use]
pub fn cross_motion_force(v: [f64; 6], f: [f64; 6]) -> [f64; 6] {
    let (va, vl) = split6(v);
    let (fa, fl) = split6(f);
    join6(add3(cross3(va, fa), cross3(vl, fl)), cross3(va, fl))
}

/// Row-major 3x3 matrix product `a * b`.
#[must_use]
pub fn mat3_mul(a: [f64; 9], b: [f64; 9]) -> [f64; 9] {
    let mut out = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            out[i * 3 + j] = (0..3).map(|k| a[i * 3 + k] * b[k * 3 + j]).sum();
        }
    }
    out
}

/// Composes two rigid poses, each given as (rotation, translation): the
/// result maps a point through `b` first and then through `a`.
#[must_use]
pub fn compose_pose(
    rot_a: [f64; 9],
    trans_a: [f64; 3],
    rot_b: [f64; 9],
    trans_b: [f64; 3],
) -> ([f64; 9], [f64; 3]) {
    (
        mat3_mul(rot_a, rot_b),
        add3(mat3_mul_vec(rot_a, trans_b), trans_a),
    )
}

/// Row-major rotation matrix of the unit quaternion `(w, x, y, z)`:
/// `(w^2 - |u|^2) E + 2 u u^T + 2 w [u]x` with `u = (x, y, z)`. This is the
/// same map `robot_types::frames::Quat::rotate` applies to a vector.
#[must_use]
pub fn rotation_from_quat(w: f64, x: f64, y: f64, z: f64) -> [f64; 9] {
    let u = [x, y, z];
    let d = w * w - (x * x + y * y + z * z);
    let cross = skew(u);
    let mut out = [0.0; 9];
    for r in 0..3 {
        for c in 0..3 {
            let identity = if r == c { d } else { 0.0 };
            out[r * 3 + c] = identity + 2.0 * u[r] * u[c] + 2.0 * w * cross[r * 3 + c];
        }
    }
    out
}

/// Skew-symmetric matrix `[v]x`, row-major, so that `[v]x w = v x w`.
#[must_use]
pub fn skew(v: [f64; 3]) -> [f64; 9] {
    [0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0]
}

/// Row-major 6x6 spatial inertia applied to a motion vector: `I v`.
#[must_use]
pub fn inertia_apply(inertia: &[f64; 36], v: [f64; 6]) -> [f64; 6] {
    let mut out = [0.0; 6];
    for (r, out_r) in out.iter_mut().enumerate() {
        let row = &inertia[r * 6..r * 6 + 6];
        *out_r = row.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
    }
    out
}

/// Element-wise sum of two row-major 6x6 matrices.
#[must_use]
pub fn matrix6_add(a: &[f64; 36], b: &[f64; 36]) -> [f64; 36] {
    let mut out = [0.0; 36];
    for i in 0..36 {
        out[i] = a[i] + b[i];
    }
    out
}

/// Element-wise difference of two row-major 6x6 matrices.
#[must_use]
pub fn matrix6_sub(a: &[f64; 36], b: &[f64; 36]) -> [f64; 36] {
    let mut out = [0.0; 36];
    for i in 0..36 {
        out[i] = a[i] - b[i];
    }
    out
}

/// The spatial inertia of a rigid body about its own frame origin, from mass,
/// centre-of-mass offset `center_of_mass` (in this frame) and the rotational
/// inertia `inertia_about_com` (row-major 3x3, about the centre of mass, in
/// this frame's axes).
///
/// Featherstone eq. 2.63: with `c` the centre of mass relative to the
/// reference origin, `m` the mass and `I_c` the rotational inertia about the
/// centre of mass, `I = [[I_c + m*skew(c)*skew(c)^T, m*skew(c)],
/// [m*skew(c)^T, m*E]]`.
#[must_use]
pub fn inertia_from_mass(
    mass: f64,
    center_of_mass: [f64; 3],
    inertia_about_com: [f64; 9],
) -> [f64; 36] {
    let c = center_of_mass;
    let sc = skew(c);
    // skew(c) * skew(c)^T, computed as the explicit product rather than the
    // closed form |c|^2 E - c c^T, so `parallel_axis_matches_the_closed_form`
    // checks the product against the closed form and does not assume it.
    let mut sc_sc_t = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            sc_sc_t[i * 3 + j] = (0..3).map(|k| sc[i * 3 + k] * sc[j * 3 + k]).sum();
        }
    }
    let mut out = [0.0; 36];
    for r in 0..3 {
        for col in 0..3 {
            let top_left = inertia_about_com[r * 3 + col] + mass * sc_sc_t[r * 3 + col];
            let top_right = mass * sc[r * 3 + col];
            let bottom_left = mass * sc[col * 3 + r];
            let bottom_right = if r == col { mass } else { 0.0 };
            out[r * 6 + col] = top_left;
            out[r * 6 + col + 3] = top_right;
            out[(r + 3) * 6 + col] = bottom_left;
            out[(r + 3) * 6 + col + 3] = bottom_right;
        }
    }
    out
}

/// Transforms a spatial inertia expressed about a child frame's origin into
/// the same inertia expressed about the parent frame's origin, given the
/// child's pose in the parent frame (`rot`, `trans`).
///
/// Computed by congruence: evaluated column by column on the six standard
/// basis twists, so it reuses [`transform_motion`] and [`transform_force`]
/// rather than re-deriving the algebra in closed form.
#[must_use]
pub fn inertia_transform_to_parent(
    inertia: &[f64; 36],
    rot: [f64; 9],
    trans: [f64; 3],
) -> [f64; 36] {
    let mut out = [0.0; 36];
    for col in 0..6 {
        let mut basis = [0.0; 6];
        basis[col] = 1.0;
        let v_child = transform_motion(rot, trans, basis);
        let f_child = inertia_apply(inertia, v_child);
        let f_parent = transform_force(rot, trans, f_child);
        for row in 0..6 {
            out[row * 6 + col] = f_parent[row];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_model::pose::RigidPose;
    use robot_types::frames::{Quat, Vec3};
    use robot_types::units::Radians;

    const IDENTITY_ROT: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    fn rot_x(angle: f64) -> [f64; 9] {
        let (s, c) = angle.sin_cos();
        [1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c]
    }

    fn rot_y(angle: f64) -> [f64; 9] {
        let (s, c) = angle.sin_cos();
        [c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c]
    }

    fn diagonal_inertia(ixx: f64, iyy: f64, izz: f64) -> [f64; 9] {
        [ixx, 0.0, 0.0, 0.0, iyy, 0.0, 0.0, 0.0, izz]
    }

    #[test]
    fn spatial_duality_preserves_power() {
        let rot = rot_x(0.7);
        let trans = [0.3, -0.1, 0.5];
        let v_parent = [0.1, 0.2, -0.3, 1.0, -2.0, 0.5];
        let f_child = [0.4, 0.1, 0.0, -0.2, 0.3, 0.6];
        let v_child = transform_motion(rot, trans, v_parent);
        let lhs = dot6(v_child, f_child);

        let f_parent = transform_force(rot, trans, f_child);
        let rhs = dot6(v_parent, f_parent);
        assert!((lhs - rhs).abs() < 1e-9, "lhs={lhs} rhs={rhs}");
    }

    #[test]
    fn transform_motion_round_trips_through_its_own_inverse() {
        let rot = rot_y(1.1);
        let trans = [1.0, 0.0, -0.5];
        let v = [0.2, -0.4, 0.1, 2.0, 1.0, -1.0];
        let v_child = transform_motion(rot, trans, v);
        // The inverse pose: rotation R^T, translation -R^T t.
        let rot_inv = [
            rot[0], rot[3], rot[6], rot[1], rot[4], rot[7], rot[2], rot[5], rot[8],
        ];
        let trans_inv = scale3(mat3_transpose_mul_vec(rot, trans), -1.0);
        let back = transform_motion(rot_inv, trans_inv, v_child);
        for i in 0..6 {
            assert!((back[i] - v[i]).abs() < 1e-9, "component {i}");
        }
    }

    #[test]
    fn parallel_axis_matches_the_closed_form() {
        let m = 2.5;
        let c = [0.3, -0.1, 0.2];
        let ic = diagonal_inertia(0.01, 0.02, 0.03);
        let inertia = inertia_from_mass(m, c, ic);

        // Hand-computed top-left block via the closed form I_c + m(|c|^2 E - c c^T).
        let c2 = dot3(c, c);
        let expected_top_left = [
            [
                0.01 + m * (c2 - c[0] * c[0]),
                -m * c[0] * c[1],
                -m * c[0] * c[2],
            ],
            [
                -m * c[1] * c[0],
                0.02 + m * (c2 - c[1] * c[1]),
                -m * c[1] * c[2],
            ],
            [
                -m * c[2] * c[0],
                -m * c[2] * c[1],
                0.03 + m * (c2 - c[2] * c[2]),
            ],
        ];
        for (r, row) in expected_top_left.iter().enumerate() {
            for (col, want) in row.iter().enumerate() {
                assert!(
                    (inertia[r * 6 + col] - want).abs() < 1e-9,
                    "r={r} col={col} got={} want={want}",
                    inertia[r * 6 + col],
                );
            }
        }
        // Bottom-right block is m * identity.
        for r in 0..3 {
            for col in 0..3 {
                let want = if r == col { m } else { 0.0 };
                assert!((inertia[(r + 3) * 6 + col + 3] - want).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn a_rigid_inertia_computes_known_kinetic_energy() {
        // Point mass at distance L from the reference origin, spinning about
        // an axis through the origin perpendicular to the offset:
        // KE = 0.5 * m * (omega * L)^2 (pure circular motion), independent of
        // this module's own formulas for I.
        let m = 1.3;
        let l = 0.7;
        let omega = 2.0;
        let c = [l, 0.0, 0.0];
        let inertia = inertia_from_mass(m, c, [0.0; 9]);
        let v = [0.0, omega, 0.0, 0.0, 0.0, 0.0];
        let momentum = inertia_apply(&inertia, v);
        let ke = 0.5 * dot6(momentum, v);
        let expected = 0.5 * m * (omega * l).powi(2);
        assert!((ke - expected).abs() < 1e-9, "ke={ke} expected={expected}");
    }

    #[test]
    fn transforming_an_inertia_to_its_own_frame_is_a_no_op() {
        let inertia = inertia_from_mass(1.0, [0.1, 0.2, -0.1], diagonal_inertia(0.01, 0.02, 0.03));
        let transformed = inertia_transform_to_parent(&inertia, IDENTITY_ROT, [0.0; 3]);
        for i in 0..36 {
            assert!((transformed[i] - inertia[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn quaternion_rotation_matrix_matches_robot_types_rotation() {
        let vectors = [
            Vec3::new(0.2, -0.7, 1.1),
            Vec3::new(-1.3, 0.4, 0.9),
            Vec3::new(0.0, 2.0, -0.5),
        ];
        let quaternions = [
            Quat::from_components(0.5, 0.5, 0.5, 0.5).unwrap(),
            Quat::from_axis_angle(Vec3::new(1.0, -2.0, 3.0), Radians(0.73)).unwrap(),
            Quat::from_components(-0.2, 0.3, 0.4, -0.8).unwrap(),
        ];

        for q in quaternions {
            let rotation = rotation_from_quat(q.w, q.x, q.y, q.z);
            for v in vectors {
                let actual = mat3_mul_vec(rotation, [v.x, v.y, v.z]);
                let expected = q.rotate(v);
                assert!((actual[0] - expected.x).abs() < 1e-12);
                assert!((actual[1] - expected.y).abs() < 1e-12);
                assert!((actual[2] - expected.z).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn array_pose_composition_matches_rigid_pose_compose() {
        let outer = RigidPose::new(
            Vec3::new(0.7, -1.2, 0.4),
            Quat::from_axis_angle(Vec3::new(1.0, 2.0, -1.0), Radians(0.61)).unwrap(),
        );
        let inner = RigidPose::new(
            Vec3::new(-0.3, 0.8, 1.4),
            Quat::from_axis_angle(Vec3::new(-2.0, 1.0, 3.0), Radians(-0.47)).unwrap(),
        );

        let outer_rotation = rotation_from_quat(
            outer.rotation.w,
            outer.rotation.x,
            outer.rotation.y,
            outer.rotation.z,
        );
        let inner_rotation = rotation_from_quat(
            inner.rotation.w,
            inner.rotation.x,
            inner.rotation.y,
            inner.rotation.z,
        );
        let (actual_rotation, actual_translation) = compose_pose(
            outer_rotation,
            [
                outer.translation.x,
                outer.translation.y,
                outer.translation.z,
            ],
            inner_rotation,
            [
                inner.translation.x,
                inner.translation.y,
                inner.translation.z,
            ],
        );
        let expected = outer.compose(&inner);
        let expected_rotation = rotation_from_quat(
            expected.rotation.w,
            expected.rotation.x,
            expected.rotation.y,
            expected.rotation.z,
        );

        for i in 0..9 {
            assert!((actual_rotation[i] - expected_rotation[i]).abs() < 1e-12);
        }
        assert_eq!(actual_rotation, mat3_mul(outer_rotation, inner_rotation));
        assert!((actual_translation[0] - expected.translation.x).abs() < 1e-12);
        assert!((actual_translation[1] - expected.translation.y).abs() < 1e-12);
        assert!((actual_translation[2] - expected.translation.z).abs() < 1e-12);
    }
}

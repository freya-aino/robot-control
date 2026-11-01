//! Narrow-phase collision for convex primitives: sphere, capsule, box and
//! cylinder, placed in the world by a rigid transform.
//!
//! Two algorithms are used, and both are standard in production physics code:
//!
//! - **Capsule and sphere pairs** use the analytic closest points of two line
//!   segments (Ericson, *Real-Time Collision Detection*, §5.1.9). A capsule is a
//!   segment swept by a radius, so the distance between two capsules is the
//!   distance between their segments minus the sum of the radii.
//! - **Every other convex pair** uses GJK (Gilbert, Johnson and Keerthi, 1988)
//!   on the Minkowski difference `A - B`. GJK gives the distance and the closest
//!   points when the shapes are apart. When they overlap, the simplex that
//!   encloses the origin seeds EPA (Expanding Polytope Algorithm, van den Bergen,
//!   2001), which returns the penetration depth and the direction of least
//!   penetration.
//!
//! Support functions are exact for all four shapes, so GJK and EPA need no shape
//! specific code. Planes are handled by [`plane_contact_points`], which reduces a
//! shape to the points that describe its contact with the plane.

/// A 3-vector in world or body coordinates.
pub type V3 = [f64; 3];

fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn norm2(a: V3) -> f64 {
    dot(a, a)
}
fn norm(a: V3) -> f64 {
    norm2(a).sqrt()
}
fn normalized_or(a: V3, fallback: V3) -> V3 {
    let n = norm(a);
    if n > 1e-300 {
        scale(a, 1.0 / n)
    } else {
        fallback
    }
}

/// A row-major 3×3 rotation.
pub type Mat3 = [[f64; 3]; 3];

fn mat_vec(m: &Mat3, v: V3) -> V3 {
    [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
}
fn mat_t_vec(m: &Mat3, v: V3) -> V3 {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

/// A convex primitive in its own frame. Capsules and cylinders are along Z.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Primitive {
    /// A sphere of this radius.
    Sphere {
        /// Radius in metres.
        radius: f64,
    },
    /// A capsule: the segment `[-half_height, +half_height]` on Z, swept by
    /// `radius`. Its total length is `2 * (half_height + radius)`.
    Capsule {
        /// Half the length of the core segment, metres.
        half_height: f64,
        /// Radius of the swept sphere, metres.
        radius: f64,
    },
    /// A box with these half extents.
    Box {
        /// Half extents along the local axes, metres.
        half: V3,
    },
    /// A cylinder of this radius and half height, along Z.
    Cylinder {
        /// Radius, metres.
        radius: f64,
        /// Half height, metres.
        half_height: f64,
    },
}

/// A primitive placed in the world.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placed {
    /// The shape.
    pub shape: Primitive,
    /// Rotation from the shape frame to the world.
    pub rotation: Mat3,
    /// Position of the shape frame in the world.
    pub translation: V3,
}

impl Placed {
    /// The point of the shape farthest in a direction given in the shape frame.
    fn local_support(&self, d: V3) -> V3 {
        match self.shape {
            Primitive::Sphere { radius } => scale(normalized_or(d, [1.0, 0.0, 0.0]), radius),
            Primitive::Capsule {
                half_height,
                radius,
            } => {
                let end = [
                    0.0,
                    0.0,
                    if d[2] >= 0.0 {
                        half_height
                    } else {
                        -half_height
                    },
                ];
                add(end, scale(normalized_or(d, [1.0, 0.0, 0.0]), radius))
            }
            Primitive::Box { half } => [
                if d[0] >= 0.0 { half[0] } else { -half[0] },
                if d[1] >= 0.0 { half[1] } else { -half[1] },
                if d[2] >= 0.0 { half[2] } else { -half[2] },
            ],
            Primitive::Cylinder {
                radius,
                half_height,
            } => {
                let xy = [d[0], d[1], 0.0];
                let rim = if norm2(xy) > 1e-300 {
                    scale(normalized_or(xy, [1.0, 0.0, 0.0]), radius)
                } else {
                    [0.0, 0.0, 0.0]
                };
                [
                    rim[0],
                    rim[1],
                    if d[2] >= 0.0 {
                        half_height
                    } else {
                        -half_height
                    },
                ]
            }
        }
    }

    /// The point of the shape farthest in a direction given in the world frame.
    #[must_use]
    pub fn support(&self, d_world: V3) -> V3 {
        let d_local = mat_t_vec(&self.rotation, d_world);
        add(
            mat_vec(&self.rotation, self.local_support(d_local)),
            self.translation,
        )
    }

    /// The world position of a point given in the shape frame.
    #[must_use]
    pub fn to_world(&self, p: V3) -> V3 {
        add(mat_vec(&self.rotation, p), self.translation)
    }

    /// World axis-aligned bounding box as `(centre, half_extents)`.
    ///
    /// Exact for each primitive: a sphere is its radius on every axis, a
    /// capsule the box of its swept segment, a box the absolute row sums of its
    /// rotation times its half extents, and a cylinder its rim circle swept
    /// along its axis.
    ///
    /// This and [`Placed::bounding_sphere`] are the two cheap rejection tests
    /// the example's broad phase runs before any narrow-phase call
    /// (`mj_filterSphere` and `filterBox`, 03 sections 2.2 and 2.4). Appendix D
    /// M-15 records that neither existed on the pipeline: every shape pair went
    /// straight to a full GJK query, and `bounding_sphere` was written but
    /// called from nowhere in the workspace.
    #[must_use]
    pub fn world_aabb(&self) -> (V3, V3) {
        let r = &self.rotation;
        // Column j of the rotation is shape axis j in world.
        let axis_abs = |k: usize| [r[0][k].abs(), r[1][k].abs(), r[2][k].abs()];
        match self.shape {
            Primitive::Sphere { radius } => (self.translation, [radius, radius, radius]),
            Primitive::Capsule {
                half_height,
                radius,
            } => {
                let up = axis_abs(2);
                (
                    self.translation,
                    [
                        up[0] * half_height + radius,
                        up[1] * half_height + radius,
                        up[2] * half_height + radius,
                    ],
                )
            }
            Primitive::Box { half } => {
                let (ax, ay, az) = (axis_abs(0), axis_abs(1), axis_abs(2));
                (
                    self.translation,
                    [
                        ax[0] * half[0] + ay[0] * half[1] + az[0] * half[2],
                        ax[1] * half[0] + ay[1] * half[1] + az[1] * half[2],
                        ax[2] * half[0] + ay[2] * half[1] + az[2] * half[2],
                    ],
                )
            }
            Primitive::Cylinder {
                radius,
                half_height,
            } => {
                // The extent along world axis i is |n_z,i| * h plus the radius
                // of the rim projected onto that axis, which is
                // r * sqrt(1 - n_z,i^2).
                let up = axis_abs(2);
                let half = |i: usize| {
                    let c = up[i].min(1.0);
                    up[i] * half_height + radius * (1.0 - c * c).max(0.0).sqrt()
                };
                (self.translation, [half(0), half(1), half(2)])
            }
        }
    }

    /// A bounding sphere `(centre, radius)` in the world.
    #[must_use]
    pub fn bounding_sphere(&self) -> (V3, f64) {
        let r = match self.shape {
            Primitive::Sphere { radius } => radius,
            Primitive::Capsule {
                half_height,
                radius,
            } => half_height + radius,
            Primitive::Box { half } => norm(half),
            Primitive::Cylinder {
                radius,
                half_height,
            } => (radius * radius + half_height * half_height).sqrt(),
        };
        (self.translation, r)
    }
}

/// `mj_filterSphere` of 03 section 2.4, bounding-sphere form: `true` when the
/// two shapes are far enough apart that no contact within `margin` is possible.
///
/// The cheapest useful rejection, and exact for spheres.
#[must_use]
pub fn spheres_apart(a: &Placed, b: &Placed, margin: f64) -> bool {
    let (ca, ra) = a.bounding_sphere();
    let (cb, rb) = b.bounding_sphere();
    let bound = ra + rb + margin;
    norm2(sub(cb, ca)) > bound * bound
}

/// `filterBox` of 03 section 2.2: `true` when the world axis-aligned bounding
/// boxes are separated on some axis. Touching counts as overlapping, as the
/// example's strict comparison does.
#[must_use]
pub fn aabbs_apart(a: &Placed, b: &Placed, margin: f64) -> bool {
    let (c1, h1) = a.world_aabb();
    let (c2, h2) = b.world_aabb();
    (0..3).any(|k| c1[k] + h1[k] + margin < c2[k] - h2[k] || c2[k] + h2[k] + margin < c1[k] - h1[k])
}

/// The result of a narrow-phase test between two convex shapes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contact {
    /// Penetration depth, metres. Zero when the shapes are apart.
    pub depth: f64,
    /// Unit direction in which shape `a` must move to separate from `b`.
    pub normal: V3,
    /// Contact point on shape `a`, world frame.
    pub point_a: V3,
    /// Contact point on shape `b`, world frame.
    pub point_b: V3,
}

/// Result of a narrow-phase query that found the shapes apart.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Proximity {
    /// The shapes are separated by `distance` (metres, positive).
    Apart {
        /// Gap between the surfaces, metres.
        distance: f64,
        /// Closest point on `a`, world frame.
        point_a: V3,
        /// Closest point on `b`, world frame.
        point_b: V3,
        /// Unit direction from `b` toward `a`.
        normal: V3,
    },
    /// The shapes overlap.
    Overlap(Contact),
}

#[derive(Clone, Copy, Debug)]
struct Vertex {
    /// Minkowski difference point `a_point - b_point`.
    w: V3,
    a: V3,
    b: V3,
}

fn msupport(a: &Placed, b: &Placed, d: V3) -> Vertex {
    let pa = a.support(d);
    let pb = b.support(scale(d, -1.0));
    Vertex {
        w: sub(pa, pb),
        a: pa,
        b: pb,
    }
}

/// Closest point to the origin of triangle `(a, b, c)`, as barycentric weights
/// `(u, v, w)` of the vertices (Ericson §5.1.5, with `p` the origin).
fn closest_point_triangle_origin(a: V3, b: V3, c: V3) -> [f64; 3] {
    let p = [0.0; 3];
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return [1.0, 0.0, 0.0];
    }
    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return [0.0, 1.0, 0.0];
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return [1.0 - v, v, 0.0];
    }
    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return [0.0, 0.0, 1.0];
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return [1.0 - w, 0.0, w];
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return [0.0, 1.0 - w, w];
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    [1.0 - v - w, v, w]
}

// ---------------------------------------------------------------------------
// Signed-volume subdistance (Montanari, Petrinic and Barbieri 2017), as
// specified in `docs/example-system-specs/05_collision_convex_gjk_epa.md`
// section 5.4.
//
// This replaces the Ericson closest-feature region tests the simplex reduction
// used before, and the reason is structural rather than a matter of accuracy.
// The expanding-polytope algorithm is only defined when the origin lies
// strictly inside the polytope it expands, and the example never has to check
// that: its GJK signals penetration exactly when this routine returns four
// non-zero barycentric weights, and four non-zero weights *are* strict
// enclosure. Appendix D entries M-05 and M-06 record the alternative --
// region tests give a correct closest feature but no such invariant, and
// building the polytope constructions of section 6.1 on top of them trades a
// false overlap for a missed one and back.
// ---------------------------------------------------------------------------

/// `sameSign2` of section 5.4: `1` when both arguments are positive, `-1` when
/// both are negative, `0` otherwise. Zero counts as "not the same sign", which
/// is what makes the enclosure test strict.
fn same_sign2(a: f64, b: f64) -> i32 {
    if a > 0.0 && b > 0.0 {
        1
    } else if a < 0.0 && b < 0.0 {
        -1
    } else {
        0
    }
}

/// `det3` of section 5.4, in the example's term order.
fn det3(a: V3, b: V3, c: V3) -> f64 {
    a[0] * (b[1] * c[2] - b[2] * c[1])
        + a[1] * (b[2] * c[0] - b[0] * c[2])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
}

/// `projectOriginLine` of section 5.4.
fn project_origin_line(s1: V3, s2: V3) -> V3 {
    let diff = sub(s2, s1);
    let dd = dot(diff, diff);
    let scl = if dd > 0.0 { -(dot(s2, diff) / dd) } else { 0.0 };
    add(s2, scale(diff, scl))
}

/// `projectOriginPlane` of section 5.4: the origin's projection onto the plane
/// through three points, with the example's three attempts for robustness.
/// `None` corresponds to the example's non-zero return, meaning the points are
/// collinear.
fn project_origin_plane(v1: V3, v2: V3, v3: V3) -> Option<V3> {
    let d21 = sub(v2, v1);
    let d31 = sub(v3, v1);
    let d32 = sub(v3, v2);

    let n = cross(d32, d21);
    let nv = dot(n, v2);
    let nn = dot(n, n);
    if nn == 0.0 {
        return None;
    }
    if nv != 0.0 && nn > 1e-15 {
        return Some(scale(n, nv / nn));
    }

    let n = cross(d21, d31);
    let nv = dot(n, v1);
    let nn = dot(n, n);
    if nn == 0.0 {
        return None;
    }
    if nv != 0.0 && nn > 1e-15 {
        return Some(scale(n, nv / nn));
    }

    let n = cross(d31, d32);
    let nv = dot(n, v3);
    let nn = dot(n, n);
    if nn == 0.0 {
        return None;
    }
    Some(scale(n, nv / nn))
}

/// `S1D` of section 5.4: barycentric weights of the origin's closest point on
/// the segment `s1..s2`.
fn s1d(s1: V3, s2: V3) -> [f64; 2] {
    let p_o = project_origin_line(s1, s2);
    let mut idx = 0usize;
    let mut mu_max = s1[0] - s2[0];
    if (s1[1] - s2[1]).abs() >= mu_max.abs() {
        idx = 1;
        mu_max = s1[1] - s2[1];
    }
    if (s1[2] - s2[2]).abs() >= mu_max.abs() {
        idx = 2;
        mu_max = s1[2] - s2[2];
    }
    let c1 = p_o[idx] - s2[idx];
    let c2 = s1[idx] - p_o[idx];
    if same_sign2(mu_max, c1) != 0 && same_sign2(mu_max, c2) != 0 {
        [c1 / mu_max, c2 / mu_max]
    } else {
        // Outside the segment: the example always takes `s2`, the newest
        // vertex in GJK ordering.
        [0.0, 1.0]
    }
}

/// `S2D` of section 5.4: barycentric weights over the triangle `s1, s2, s3`.
fn s2d(s1: V3, s2: V3, s3: V3) -> [f64; 3] {
    let Some(p_o) = project_origin_plane(s1, s2, s3) else {
        let l = s1d(s1, s2);
        return [l[0], l[1], 0.0];
    };

    // Exact term order from section 5.4.
    let m14 = s2[1] * s3[2] - s2[2] * s3[1] - s1[1] * s3[2] + s1[2] * s3[1] + s1[1] * s2[2]
        - s1[2] * s2[1];
    let m24 = s2[0] * s3[2] - s2[2] * s3[0] - s1[0] * s3[2] + s1[2] * s3[0] + s1[0] * s2[2]
        - s1[2] * s2[0];
    let m34 = s2[0] * s3[1] - s2[1] * s3[0] - s1[0] * s3[1] + s1[1] * s3[0] + s1[0] * s2[1]
        - s1[1] * s2[0];

    // Drop the axis whose minor is largest, keeping the best-conditioned
    // two-dimensional projection.
    let (ax, ay, m_max) = if m14.abs() >= m24.abs() && m14.abs() >= m34.abs() {
        (1usize, 2usize, m14)
    } else if m24.abs() >= m34.abs() {
        (0usize, 2usize, m24)
    } else {
        (0usize, 1usize, m34)
    };

    let p = [p_o[ax], p_o[ay]];
    let a = [s1[ax], s1[ay]];
    let bb = [s2[ax], s2[ay]];
    let c = [s3[ax], s3[ay]];

    let c31 = p[0] * bb[1] + p[1] * c[0] + bb[0] * c[1] - p[0] * c[1] - p[1] * bb[0] - c[0] * bb[1];
    let c32 = p[0] * c[1] + p[1] * a[0] + c[0] * a[1] - p[0] * a[1] - p[1] * c[0] - a[0] * c[1];
    let c33 = p[0] * a[1] + p[1] * bb[0] + a[0] * bb[1] - p[0] * bb[1] - p[1] * a[0] - bb[0] * a[1];

    let comp = [
        same_sign2(m_max, c31),
        same_sign2(m_max, c32),
        same_sign2(m_max, c33),
    ];
    if comp.iter().all(|c| *c != 0) {
        return [c31 / m_max, c32 / m_max, c33 / m_max];
    }

    // Fall back to the edges, in the example's order, with the first candidate
    // taken unconditionally and the rest on a strict improvement.
    let mut best = [0.0; 3];
    let mut dmin = f64::INFINITY;
    let mut consider = |l: [f64; 2], p1: V3, p2: V3, slot: [usize; 2]| {
        let x = add(scale(p1, l[0]), scale(p2, l[1]));
        let d = norm2(x);
        if d < dmin {
            dmin = d;
            best = [0.0; 3];
            best[slot[0]] = l[0];
            best[slot[1]] = l[1];
        }
    };
    if comp[0] == 0 {
        consider(s1d(s2, s3), s2, s3, [1, 2]);
    }
    if comp[1] == 0 {
        consider(s1d(s1, s3), s1, s3, [0, 2]);
    }
    if comp[2] == 0 {
        consider(s1d(s1, s2), s1, s2, [0, 1]);
    }
    best
}

/// `S3D` of section 5.4: barycentric weights over the tetrahedron.
///
/// All four weights come back non-zero exactly when the origin is strictly
/// inside, which is the invariant the expansion depends on.
fn s3d(s1: V3, s2: V3, s3: V3, s4: V3) -> [f64; 4] {
    let c41 = -det3(s2, s3, s4);
    let c42 = det3(s1, s3, s4);
    let c43 = -det3(s1, s2, s4);
    let c44 = det3(s1, s2, s3);
    let m_det = c41 + c42 + c43 + c44;
    let cs = [c41, c42, c43, c44];
    let comp = [
        same_sign2(m_det, c41),
        same_sign2(m_det, c42),
        same_sign2(m_det, c43),
        same_sign2(m_det, c44),
    ];
    if comp.iter().all(|c| *c != 0) {
        return [cs[0] / m_det, cs[1] / m_det, cs[2] / m_det, cs[3] / m_det];
    }

    let mut best = [0.0; 4];
    let mut dmin = f64::INFINITY;
    let mut consider = |l: [f64; 3], p: [V3; 3], slot: [usize; 3]| {
        let x = add(add(scale(p[0], l[0]), scale(p[1], l[1])), scale(p[2], l[2]));
        let d = norm2(x);
        if d < dmin {
            dmin = d;
            best = [0.0; 4];
            best[slot[0]] = l[0];
            best[slot[1]] = l[1];
            best[slot[2]] = l[2];
        }
    };
    if comp[0] == 0 {
        consider(s2d(s2, s3, s4), [s2, s3, s4], [1, 2, 3]);
    }
    if comp[1] == 0 {
        consider(s2d(s1, s3, s4), [s1, s3, s4], [0, 2, 3]);
    }
    if comp[2] == 0 {
        consider(s2d(s1, s2, s4), [s1, s2, s4], [0, 1, 3]);
    }
    if comp[3] == 0 {
        consider(s2d(s1, s2, s3), [s1, s2, s3], [0, 1, 2]);
    }
    best
}

/// `subdistance` of section 5.4: barycentric weights of the origin's closest
/// point on a simplex of one to four Minkowski-difference points.
fn subdistance(simplex: &[Vertex]) -> [f64; 4] {
    let mut lambda = [0.0; 4];
    match simplex.len() {
        4 => s3d(simplex[0].w, simplex[1].w, simplex[2].w, simplex[3].w),
        3 => {
            let l = s2d(simplex[0].w, simplex[1].w, simplex[2].w);
            lambda[..3].copy_from_slice(&l);
            lambda
        }
        2 => {
            let l = s1d(simplex[0].w, simplex[1].w);
            lambda[..2].copy_from_slice(&l);
            lambda
        }
        _ => {
            lambda[0] = 1.0;
            lambda
        }
    }
}

/// `sameSide(p0, p1, p2, p3)` of 05 section 6.1: whether `p3` and the origin
/// lie strictly on the same side of the plane through `p0, p1, p2`.
fn same_side(p0: V3, p1: V3, p2: V3, p3: V3) -> bool {
    let n = cross(sub(p1, p0), sub(p2, p0));
    let d1 = dot(n, sub(p3, p0));
    let d2 = dot(n, scale(p0, -1.0));
    (d1 > 0.0 && d2 > 0.0) || (d1 < 0.0 && d2 < 0.0)
}

/// `testTetra` of 05 section 6.1: whether the tetrahedron strictly contains the
/// origin.
fn test_tetra(p: [V3; 4]) -> bool {
    same_side(p[0], p[1], p[2], p[3])
        && same_side(p[1], p[2], p[3], p[0])
        && same_side(p[2], p[3], p[0], p[1])
        && same_side(p[3], p[0], p[1], p[2])
}

/// The initial polytope for the expansion: its vertices and the index triples
/// of its faces, from 05 section 6.1.
///
/// Two constructions, as the example has them. A four-vertex simplex gives a
/// tetrahedron directly (`polytope4`); a three-vertex one gives a **five-vertex
/// hexahedron** (`polytope3`), taking support points on both sides of the
/// triangle. The hexahedron is not an optimisation: when the triangle's plane
/// passes through the origin, no tetrahedron on those three vertices plus one
/// more can strictly contain the origin without being degenerate, which is why
/// an earlier "grow to four affinely independent points" step could not
/// establish the invariant whatever its tolerances (Appendix D section 1.5.1).
fn build_polytope(
    a: &Placed,
    b: &Placed,
    simplex: &[Vertex],
    gjk_distance: f64,
) -> Option<(Vec<Vertex>, Vec<[usize; 3]>)> {
    match simplex.len() {
        4 => {
            let w = [simplex[0].w, simplex[1].w, simplex[2].w, simplex[3].w];
            if test_tetra(w) {
                return Some((
                    simplex.to_vec(),
                    vec![[0, 1, 2], [0, 3, 1], [0, 2, 3], [3, 2, 1]],
                ));
            }
            // "replaceSimplex3": fall back to the triangle construction on one
            // of the faces.
            for skip in 0..4 {
                let face: Vec<Vertex> = (0..4).filter(|&i| i != skip).map(|i| simplex[i]).collect();
                if let Some(built) = build_polytope(a, b, &face, gjk_distance) {
                    return Some(built);
                }
            }
            None
        }
        3 => {
            let (v1, v2, v3) = (simplex[0].w, simplex[1].w, simplex[2].w);
            let n = cross(sub(v2, v1), sub(v3, v1));
            let n_norm = norm(n);
            if n_norm <= 1e-300 || !n_norm.is_finite() {
                // "P3_BAD_NORMAL": the three points are collinear.
                return None;
            }
            let dir = scale(n, 1.0 / n_norm);
            let v5 = msupport(a, b, scale(dir, -1.0));
            let v4 = msupport(a, b, dir);
            // "P3_ORIGIN_ON_FACE": neither support point may lie in the
            // triangle's own plane, or the hexahedron has no volume on that
            // side.
            const MIN_EPA_TOL: f64 = 1e-12;
            if dot(dir, sub(v4.w, v1)) < MIN_EPA_TOL
                || dot(scale(dir, -1.0), sub(v5.w, v1)) < MIN_EPA_TOL
            {
                return None;
            }
            // "P3_MISSING_ORIGIN", guarded by the example with
            // `dist[0] > 10*mjMINVAL`. That guard is load-bearing: when the
            // origin lies on the triangle it is on the shared face of both
            // half-tetrahedra, so neither `testTetra` can pass, and applying
            // the test unconditionally rejects every such overlap. The
            // hexahedron does contain the origin in that case.
            if gjk_distance > 10.0 * 1e-15
                && !test_tetra([v1, v2, v3, v4.w])
                && !test_tetra([v1, v2, v3, v5.w])
            {
                return None;
            }
            // Insertion order follows the example: v5 at index 3, v4 at 4.
            let verts = vec![simplex[0], simplex[1], simplex[2], v5, v4];
            let faces = vec![
                [4, 0, 1],
                [4, 2, 0],
                [4, 1, 2],
                [3, 1, 0],
                [3, 0, 2],
                [3, 2, 1],
            ];
            Some((verts, faces))
        }
        _ => None,
    }
}

/// Iteration cap for the expanding-polytope loop. The example caps at
/// `min(max_iterations, 1000)` (05 section 6.3).
const EPA_MAX_ITERATIONS: usize = 128;
/// Relative gap between the penetration depth's lower and upper bounds at
/// which the expansion is considered converged (05 section 6.3).
const EPA_TOLERANCE: f64 = 1e-8;
/// Maximum faces the expanding polytope may hold. The example carries the same
/// bound (`maxfaces`) and breaks out of the loop when a horizon would exceed
/// it, keeping the best face found rather than failing: "if nedges > maxfaces -
/// nfaces: warning out of memory for faces; break" (05 section 6.3).
///
/// Omitting it is not a missing optimisation, it is a missing termination
/// condition. Each pass replaces the faces the new support point can see with
/// one face per horizon edge, so a polytope whose faces are not a proper convex
/// hull -- which the triangle construction can produce -- grows its face count
/// geometrically instead of refining it. Measured without this bound, a single
/// narrow-phase query on a collapsed G1 took 2.9 seconds while reporting a
/// normal four GJK iterations, because the entire cost was a face list that had
/// run away.
const EPA_MAX_FACES: usize = 256;

/// EPA: expands the polytope around the origin until it reaches the Minkowski
/// boundary, and returns the closest face's depth and normal.
///
/// Returns `None` when the polytope degenerates: a face's plane strictly
/// outside the origin, a horizon that does not close, or the iteration budget
/// exhausted with the bound gap still open. The example reports no contact in
/// each of those cases rather than a fabricated one (`face = NULL` gives
/// `nx = 0`), and so does this. Appendix D M-05 records the previous
/// behaviour: a fixed iteration count, the result overwritten each pass, and
/// `depth.max(0.0)` turning a failure into a zero depth with an arbitrary
/// normal and no diagnostic.
fn epa(
    a: &Placed,
    b: &Placed,
    initial: Vec<Vertex>,
    initial_faces: &[[usize; 3]],
) -> Option<Contact> {
    let mut verts: Vec<Vertex> = initial;
    // A face is (indices, outward unit normal, distance from origin). The origin
    // may lie on a face of the starting tetrahedron, so faces are oriented with
    // respect to an interior point (the tetrahedron's centroid), not the origin.
    let interior = scale(
        verts.iter().fold([0.0; 3], |acc, v| add(acc, v.w)),
        1.0 / verts.len() as f64,
    );
    let make_face = |verts: &[Vertex], i: usize, j: usize, k: usize| -> ([usize; 3], V3, f64) {
        let mut n = normalized_or(
            cross(sub(verts[j].w, verts[i].w), sub(verts[k].w, verts[i].w)),
            [0.0, 0.0, 1.0],
        );
        // Keep the winding consistent with the normal: a flipped face is
        // traversed the other way, so horizon edges of neighbours cancel.
        if dot(n, sub(verts[i].w, interior)) < 0.0 {
            n = scale(n, -1.0);
            let d = dot(n, verts[i].w).abs();
            return ([i, k, j], n, d);
        }
        // The distance is the magnitude of the origin's projection onto the
        // face plane, as the example's `face.dist2 = |face.v|^2` is. The
        // direction lives in the outward normal, which the centroid test above
        // fixes, so the distance itself is never negative. Taking it signed
        // instead -- which an earlier version did -- makes a face plane that
        // happens to fall on the far side of the origin look closer than every
        // real face, and the expansion then measures a depth in the wrong
        // direction. That state is reachable because the initial face set of
        // the triangle construction is not guaranteed to be exactly the convex
        // hull; the horizon expansion repairs it, given a usable distance.
        let d = dot(n, verts[i].w).abs();
        ([i, j, k], n, d)
    };
    let mut faces: Vec<([usize; 3], V3, f64)> = Vec::new();
    for &[i, j, k] in initial_faces {
        faces.push(make_face(&verts, i, j, k));
    }
    let mut result_normal = [0.0, 0.0, 1.0];
    let mut result_depth = 0.0;
    let mut result_points = (verts[0].a, verts[0].b);
    // The example's stopping rule (05 section 6.3): `lower` is the distance to
    // the closest face, a lower bound on the penetration depth, and `upper` the
    // best bound the support points have given. Stopping on the gap yields a
    // measured accuracy rather than an unquantified one.
    let mut upper = f64::INFINITY;
    let mut converged = false;
    let mut horizon: Vec<(usize, usize)> = Vec::new();
    let mut visible: Vec<usize> = Vec::new();
    for _ in 0..EPA_MAX_ITERATIONS {
        let (fi, _) = faces
            .iter()
            .enumerate()
            .min_by(|x, y| x.1 .2.total_cmp(&y.1 .2))
            .expect("the polytope always has faces");
        let (idx, n, dist) = faces[fi];
        // A negative distance means the origin is outside this face's plane, so
        // the polytope does not contain it and the expansion has no direction
        // to measure a depth along. Zero is allowed: the face passes through
        // the origin and growing it is exactly what the expansion does next.
        if !dist.is_finite() {
            return None;
        }
        let w = msupport(a, b, n);
        let upper_k = dot(w.w, n);
        if upper_k < upper {
            upper = upper_k;
        }
        result_normal = n;
        result_depth = dist;
        let on_boundary = upper - dist < EPA_TOLERANCE * (1.0 + dist);
        // Barycentric point of the origin's projection on the face, for the
        // contact points.
        let proj = scale(n, dist);
        let bary = closest_point_triangle_origin(
            sub(verts[idx[0]].w, proj),
            sub(verts[idx[1]].w, proj),
            sub(verts[idx[2]].w, proj),
        );
        result_points = (
            (0..3).fold([0.0; 3], |acc, i| add(acc, scale(verts[idx[i]].a, bary[i]))),
            (0..3).fold([0.0; 3], |acc, i| add(acc, scale(verts[idx[i]].b, bary[i]))),
        );
        if on_boundary {
            converged = true;
            break;
        }
        // Remove the faces the new support point can see; the horizon edges of
        // the removed region connect to it.
        //
        // The two scratch buffers live outside the loop and are cleared rather
        // than reallocated. That is not micro-optimisation: draining the face
        // list into a freshly allocated one each pass costs an allocation and a
        // full copy per iteration, and with the face count growing as the
        // polytope refines, a collapsing G1 spent minutes in the allocator.
        let new_index = verts.len();
        verts.push(w);
        horizon.clear();
        visible.clear();
        for (i, (_, fnrm, fd)) in faces.iter().enumerate() {
            if dot(*fnrm, w.w) - *fd > 1e-12 {
                visible.push(i);
            }
        }
        if visible.is_empty() {
            // No face can see the new support point, so the polytope already
            // reaches the Minkowski boundary in this direction. This is
            // convergence, not failure: it is the polyhedral analogue of the
            // example's repeated-support-vertex test, which breaks while
            // keeping the current face.
            converged = true;
            break;
        }
        for &i in &visible {
            let f = faces[i].0;
            for (e0, e1) in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
                if let Some(pos) = horizon.iter().position(|&(x, y)| x == e1 && y == e0) {
                    horizon.swap_remove(pos);
                } else {
                    horizon.push((e0, e1));
                }
            }
        }
        if horizon.len() < 3 {
            // Faces were removed but their boundary does not form a closed
            // ring: the example's "if horizon.nedges < 3: face = NULL".
            return None;
        }
        if faces.len() - visible.len() + horizon.len() > EPA_MAX_FACES {
            // The example's out-of-faces break, which keeps the current best
            // face rather than discarding the result.
            break;
        }
        // Descending order keeps the remaining indices valid as each visible
        // face is swap-removed.
        for &i in visible.iter().rev() {
            faces.swap_remove(i);
        }
        for &(e0, e1) in &horizon {
            let face = make_face(&verts, e0, e1, new_index);
            faces.push(face);
        }
    }
    // Running out of the iteration budget is **not** a failure. The example
    // leaves `face` set when its loop exits on `kmax` and uses it
    // (`if face: dist[0] = epaWitness(...)`); only the specific degeneracies
    // above null it. The closest face found is a valid lower bound on the
    // penetration depth, and reporting it is far better for the contact model
    // than dropping the contact, which is what returning `None` here did --
    // and which made deep overlaps of curved shapes, exactly the regime the
    // G1 collapse test exercises, both wrong and extremely slow as the budget
    // was raised to compensate.
    //
    // `converged` is kept so that the distinction is visible to a reader: a
    // converged result is accurate to `EPA_TOLERANCE`, an unconverged one only
    // to whatever gap the budget reached.
    let _ = converged;
    Some(Contact {
        depth: result_depth,
        normal: scale(result_normal, -1.0),
        point_a: result_points.0,
        point_b: result_points.1,
    })
}

/// Closest points of two segments `p1..q1` and `p2..q2` (Ericson §5.1.9).
fn closest_segment_segment(p1: V3, q1: V3, p2: V3, q2: V3) -> (V3, V3) {
    let d1 = sub(q1, p1);
    let d2 = sub(q2, p2);
    let r = sub(p1, p2);
    let a = dot(d1, d1);
    let e = dot(d2, d2);
    let f = dot(d2, r);
    let (s, t) = if a <= 1e-300 && e <= 1e-300 {
        (0.0, 0.0)
    } else if a <= 1e-300 {
        (0.0, (f / e).clamp(0.0, 1.0))
    } else {
        let c = dot(d1, r);
        if e <= 1e-300 {
            ((-c / a).clamp(0.0, 1.0), 0.0)
        } else {
            let b = dot(d1, d2);
            let denom = a * e - b * b;
            let mut s = if denom > 1e-300 {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let mut t = (b * s + f) / e;
            if t < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            }
            (s, t)
        }
    };
    (add(p1, scale(d1, s)), add(p2, scale(d2, t)))
}

/// The axis segment of a capsule in the world: its two end centres.
fn capsule_segment(p: &Placed, half_height: f64) -> (V3, V3) {
    let up = [p.rotation[0][2], p.rotation[1][2], p.rotation[2][2]];
    (
        add(p.translation, scale(up, -half_height)),
        add(p.translation, scale(up, half_height)),
    )
}

/// Analytic narrow phase for two capsules or spheres (a sphere is a capsule with
/// zero half height). Returns the proximity or overlap of their swept segments.
fn capsule_pair(a: &Placed, a_h: f64, a_r: f64, b: &Placed, b_h: f64, b_r: f64) -> Proximity {
    let (a0, a1) = capsule_segment(a, a_h);
    let (b0, b1) = capsule_segment(b, b_h);
    let (pa, pb) = closest_segment_segment(a0, a1, b0, b1);
    let delta = sub(pa, pb);
    let gap = norm(delta);
    let radii = a_r + b_r;
    let normal = normalized_or(delta, [0.0, 0.0, 1.0]);
    if gap > radii {
        Proximity::Apart {
            distance: gap - radii,
            point_a: add(pa, scale(normal, -a_r)),
            point_b: add(pb, scale(normal, b_r)),
            normal,
        }
    } else {
        // Overlap: the surface points lie on the segment-to-segment line.
        Proximity::Overlap(Contact {
            depth: radii - gap,
            normal,
            point_a: add(pa, scale(normal, -a_r)),
            point_b: add(pb, scale(normal, b_r)),
        })
    }
}

/// The narrow phase of two placed convex primitives.
///
/// Capsule and sphere pairs use the analytic segment test. All other pairs use
/// GJK, with EPA for overlap. `Overlap.normal` pushes `a` away from `b`.
#[must_use]
pub fn proximity(a: &Placed, b: &Placed) -> Proximity {
    if let (Some((ah, ar)), Some((bh, br))) = (as_capsule(a.shape), as_capsule(b.shape)) {
        return capsule_pair(a, ah, ar, b, bh, br);
    }
    gjk_epa(a, b)
}

/// A sphere is a capsule with no core; a capsule is itself.
fn as_capsule(shape: Primitive) -> Option<(f64, f64)> {
    match shape {
        Primitive::Sphere { radius } => Some((0.0, radius)),
        Primitive::Capsule {
            half_height,
            radius,
        } => Some((half_height, radius)),
        _ => None,
    }
}

/// GJK distance and overlap for any pair of convex primitives, with EPA for the
/// penetration. Exposed so that the analytic fast paths can be checked against it.
#[must_use]
pub fn gjk_epa(a: &Placed, b: &Placed) -> Proximity {
    // The loop follows 05 section 5.3. The important structural point is how
    // penetration is decided: not by a separate containment test, but by the
    // subdistance returning four non-zero barycentric weights, which is strict
    // enclosure of the origin by construction. That is the invariant the
    // expansion in [`epa`] needs, and establishing it here rather than
    // recovering it afterwards is what Appendix D M-05 and M-06 asked for.
    const MAX_ITERATIONS: usize = 64;
    const TOLERANCE: f64 = 1e-12;

    let mut d = sub(b.translation, a.translation);
    d = normalized_or(d, [1.0, 0.0, 0.0]);
    // The simplex never exceeds four vertices, so it lives in a fixed array.
    // This matters: an earlier version held it in a `Vec` and allocated twice
    // per iteration during the compaction step, which on a contact-heavy run
    // is on the order of a million heap allocations and dominated everything
    // else. Plain fixed-size arrays are also what FR-6 asks of an algorithm
    // module and what a kernel can use.
    let mut simplex = [msupport(a, b, scale(d, -1.0)); 4];
    let mut nsimplex = 1usize;
    let mut lambda = [1.0, 0.0, 0.0, 0.0];
    let mut x_k = simplex[0].w;
    let mut x_norm = norm(x_k);
    let mut x_norm_prev = 0.0;
    let mut separated = false;
    let mut enclosed = false;

    for _ in 0..MAX_ITERATIONS {
        // Both tests are relative to the current scale. The example writes the
        // stall test as an absolute `mjMINVAL`, which works there because its
        // tolerance comes from the model and its geometry is normalised; here
        // an absolute threshold is the same mistake as the duality gap below,
        // and it shows up in the same place -- deep penetration, where `x_k`
        // stops improving at a magnitude far above 1e-15 and the loop then
        // runs its whole budget on every query.
        if x_norm < TOLERANCE || (x_norm_prev - x_norm).abs() < 1e-12 * (1.0 + x_norm) {
            break;
        }
        // Support toward the origin from the current closest point.
        let s_k = msupport(a, b, scale(x_k, -1.0 / x_norm));
        // Frank-Wolfe duality gap: no point of the difference lies closer to
        // the origin along `x_k`, so `x_k` is the closest point.
        //
        // The example writes this as an absolute `0.5 * tolerance^2`, with the
        // tolerance taken from the model (`ccd_tolerance`, default 1e-6). Ours
        // is a compile-time constant, so the test is made relative to the
        // current distance instead. The difference is not cosmetic: an
        // absolute threshold derived from a tight tolerance is of order 1e-24,
        // which a converged iteration never reaches, so the loop runs its full
        // iteration budget on every single query. That showed up as the whole
        // contact-bearing test suite slowing by more than an order of
        // magnitude.
        // The gap `dot(x_k, x_k - s_k)` has units of length squared. Dividing
        // by `|x_k|` turns it into the distance by which the new support point
        // improves on the current closest point, which is what a length
        // tolerance can be compared against.
        //
        // Getting this scaling wrong does not produce a wrong answer, it
        // produces a slow one, and only in one regime. Compared against an
        // absolute `1e-12`, the test never fires once `|x_k|` is small -- which
        // is exactly the deep-penetration case -- so the loop runs its whole
        // iteration budget on every query. A collapsing G1 went from four
        // seconds to over a minute on that alone, while shallow contact looked
        // fine at under five iterations per call.
        if dot(x_k, sub(x_k, s_k.w)) / x_norm < 1e-10 * (1.0 + x_norm) {
            break;
        }
        simplex[nsimplex] = s_k;
        nsimplex += 1;
        lambda = subdistance(&simplex[..nsimplex]);
        // Compact in place, dropping exact-zero weights and preserving order,
        // as the example does.
        let mut compact = [simplex[0]; 4];
        let mut compact_lambda = [0.0; 4];
        let mut n = 0usize;
        for i in 0..nsimplex {
            if lambda[i] != 0.0 {
                compact[n] = simplex[i];
                compact_lambda[n] = lambda[i];
                n += 1;
            }
        }
        if n == 0 {
            // Should not occur; treat as separated rather than proceed.
            separated = true;
            break;
        }
        simplex = compact;
        lambda = compact_lambda;
        nsimplex = n;
        x_k = (0..nsimplex).fold([0.0; 3], |acc, i| add(acc, scale(simplex[i].w, lambda[i])));
        x_norm_prev = x_norm;
        x_norm = norm(x_k);
        if nsimplex == 4 {
            // Four non-zero weights: the origin is strictly inside.
            enclosed = true;
            break;
        }
    }

    let witness = |simplex: &[Vertex], lambda: &[f64; 4]| -> (V3, V3) {
        let total: f64 = lambda[..simplex.len()].iter().sum();
        if simplex.is_empty() || total <= 0.0 || !total.is_finite() {
            return (a.translation, b.translation);
        }
        let mut pa = [0.0; 3];
        let mut pb = [0.0; 3];
        for (i, v) in simplex.iter().enumerate() {
            let w = lambda[i] / total;
            pa = add(pa, scale(v.a, w));
            pb = add(pb, scale(v.b, w));
        }
        (pa, pb)
    };

    // The separation test is the example's, and it belongs *after* the loop,
    // not inside it. In distance mode (no cutoff) section 5.3 never sets
    // `separated` while iterating -- the in-loop `lower > 0` checks are the
    // contacts-only and cutoff paths -- and uses a single final support query
    // instead. Setting the flag inside the loop reports a deep overlap as
    // separated, because an intermediate simplex can sit entirely on one side
    // of the origin on the way to enclosing it.
    if x_norm > 0.0 {
        let tmp = msupport(a, b, scale(x_k, -1.0 / x_norm));
        if dot(x_k, tmp.w) > 0.0 {
            separated = true;
        }
    }

    // `dist = (n == 4 && !separated) ? 0 : x_norm`. The extra clause covers a
    // simplex of fewer than four vertices that nonetheless has the origin on
    // it, which the triangle construction in `build_polytope` handles.
    let penetrating = (enclosed && !separated) || (!separated && x_norm < TOLERANCE);
    if penetrating {
        if let Some((pverts, pfaces)) = build_polytope(a, b, &simplex[..nsimplex], x_norm) {
            if let Some(contact) = epa(a, b, pverts, &pfaces) {
                return Proximity::Overlap(contact);
            }
        }
        // The polytope could not be built, or the expansion could not
        // establish a depth. Reporting `Apart` at zero distance is the truthful
        // statement -- the surfaces coincide to within what this routine
        // resolves, and there is no penetration to generate force from -- and
        // it is what the example does, whose `face = NULL` path reports no
        // contact at all. Appendix D M-05 records the previous behaviour, an
        // `Overlap` with zero depth and a normal taken from the difference of
        // the two shapes' frame origins.
        let (pa, pb) = witness(&simplex[..nsimplex], &lambda);
        return Proximity::Apart {
            distance: 0.0,
            point_a: pa,
            point_b: pb,
            normal: normalized_or(
                sub(pa, pb),
                normalized_or(sub(a.translation, b.translation), [0.0, 0.0, 1.0]),
            ),
        };
    }

    let (pa, pb) = witness(&simplex[..nsimplex], &lambda);
    let normal = normalized_or(
        sub(pa, pb),
        normalized_or(sub(a.translation, b.translation), [0.0, 0.0, 1.0]),
    );
    Proximity::Apart {
        distance: x_norm,
        point_a: pa,
        point_b: pb,
        normal,
    }
}

/// The contact points of a convex primitive with a horizontal plane `z = height`
/// (normal +Z), as `(centre, radius)` samples. Each sample is a sphere-like point
/// with its penetration `height - (centre.z - radius)`.
///
/// A sphere gives its centre. A capsule gives its two end centres, each with the
/// capsule radius. A box gives its eight corners and a cylinder its two rims
/// sampled at `rim_samples` points, all with zero radius. Corners and rim samples
/// are what let a flat-lying box or cylinder rest on several points.
#[must_use]
pub fn plane_contact_points(placed: &Placed, rim_samples: usize) -> Vec<(V3, f64)> {
    match placed.shape {
        Primitive::Sphere { radius } => vec![(placed.translation, radius)],
        Primitive::Capsule {
            half_height,
            radius,
        } => {
            let (p0, p1) = capsule_segment(placed, half_height);
            vec![(p0, radius), (p1, radius)]
        }
        Primitive::Box { half } => {
            let mut out = Vec::with_capacity(8);
            for sx in [-1.0, 1.0] {
                for sy in [-1.0, 1.0] {
                    for sz in [-1.0, 1.0] {
                        let local = [sx * half[0], sy * half[1], sz * half[2]];
                        out.push((placed.to_world(local), 0.0));
                    }
                }
            }
            out
        }
        Primitive::Cylinder {
            radius,
            half_height,
        } => {
            let n = rim_samples.max(3);
            let mut out = Vec::with_capacity(2 * n);
            for k in 0..n {
                let angle = 2.0 * core::f64::consts::PI * (k as f64) / (n as f64);
                let (s, c) = angle.sin_cos();
                for z in [-half_height, half_height] {
                    out.push((placed.to_world([radius * c, radius * s, z]), 0.0));
                }
            }
            out
        }
    }
}

fn box_parts(p: &Placed) -> Option<(V3, V3, [V3; 3])> {
    let Primitive::Box { half } = p.shape else {
        return None;
    };
    let r = &p.rotation;
    let axes = [
        [r[0][0], r[1][0], r[2][0]],
        [r[0][1], r[1][1], r[2][1]],
        [r[0][2], r[1][2], r[2][2]],
    ];
    Some((p.translation, half, axes))
}

/// Separating-axis test for two boxes (Gottschalk, Lin and Manocha 1996 for the
/// 15 candidate axes). Returns the least overlap over all candidate axes and the
/// unit axis that attains it, oriented from `b` toward `a`. Returns `None` if a
/// separating axis exists, if either shape is not a box, or if the boxes are
/// too far apart to overlap.
#[must_use]
pub fn box_sat(a: &Placed, b: &Placed) -> Option<(f64, V3)> {
    let (ca, ha, axes_a) = box_parts(a)?;
    let (cb, hb, axes_b) = box_parts(b)?;
    let mut candidates: Vec<V3> = Vec::with_capacity(15);
    candidates.extend(axes_a);
    candidates.extend(axes_b);
    for ea in axes_a {
        for eb in axes_b {
            let c = cross(ea, eb);
            if norm2(c) > 1e-18 {
                candidates.push(scale(c, 1.0 / norm(c)));
            }
        }
    }
    let d = sub(ca, cb);
    let mut best: Option<(f64, V3)> = None;
    for u in candidates {
        let ra: f64 = (0..3).map(|i| ha[i] * dot(axes_a[i], u).abs()).sum();
        let rb: f64 = (0..3).map(|i| hb[i] * dot(axes_b[i], u).abs()).sum();
        let s = dot(d, u);
        let overlap = ra + rb - s.abs();
        if overlap < 0.0 {
            return None;
        }
        if best.is_none_or(|(o, _)| overlap < o) {
            let oriented = if s >= 0.0 { u } else { scale(u, -1.0) };
            best = Some((overlap, oriented));
        }
    }
    best
}

/// Contact manifold of two boxes, up to eight points.
///
/// The normal and depth come from [`box_sat`]. A corner of one box that lies
/// inside the other is a contact point, and its depth is its penetration along
/// the normal. Corners of both boxes are used, and a corner of the second set
/// that coincides laterally with one already taken is dropped. This gives the
/// four corner contacts of a box resting flat on a box, which a single
/// closest-point contact cannot give. When no corner lies inside the other
/// box (edges crossing), the single deepest support pair is returned.
///
/// The normal points from `b` toward `a`, so that `a` is pushed away from `b`.
/// Both points of each contact are the corner itself.
#[must_use]
pub fn box_box_contacts(a: &Placed, b: &Placed) -> Option<Vec<Contact>> {
    let (depth, n) = box_sat(a, b)?;
    let (ca, ha, axes_a) = box_parts(a)?;
    let (cb, hb, axes_b) = box_parts(b)?;
    let reach = |c: V3, h: V3, axes: &[V3; 3]| -> (f64, f64) {
        let centre = dot(c, n);
        let r: f64 = (0..3).map(|i| h[i] * dot(axes[i], n).abs()).sum();
        (centre - r, centre + r)
    };
    let (min_a, _) = reach(ca, ha, &axes_a);
    let (_, max_b) = reach(cb, hb, &axes_b);
    let corners = |c: V3, h: V3, axes: &[V3; 3]| -> [V3; 8] {
        let mut out = [[0.0; 3]; 8];
        for (k, slot) in out.iter_mut().enumerate() {
            let sx = if k & 1 == 0 { -1.0 } else { 1.0 };
            let sy = if k & 2 == 0 { -1.0 } else { 1.0 };
            let sz = if k & 4 == 0 { -1.0 } else { 1.0 };
            let local = [sx * h[0], sy * h[1], sz * h[2]];
            *slot = add(
                c,
                add(
                    add(scale(axes[0], local[0]), scale(axes[1], local[1])),
                    scale(axes[2], local[2]),
                ),
            );
        }
        out
    };
    let inside = |p: V3, c: V3, h: V3, axes: &[V3; 3]| -> bool {
        let d = sub(p, c);
        (0..3).all(|i| dot(d, axes[i]).abs() <= h[i] + 1e-12)
    };
    let mut points: Vec<Contact> = Vec::new();
    let push = |p: V3, pen: f64, points: &mut Vec<Contact>| {
        let lateral_dup = points.iter().any(|q| {
            let diff = sub(p, q.point_a);
            norm(sub(diff, scale(n, dot(diff, n)))) < 1e-7
        });
        if pen > 0.0 && !lateral_dup {
            points.push(Contact {
                depth: pen,
                normal: n,
                point_a: p,
                point_b: p,
            });
        }
    };
    for p in corners(cb, hb, &axes_b) {
        if inside(p, ca, ha, &axes_a) {
            push(p, dot(n, p) - min_a, &mut points);
        }
    }
    for p in corners(ca, ha, &axes_a) {
        if inside(p, cb, hb, &axes_b) {
            push(p, max_b - dot(n, p), &mut points);
        }
    }
    if points.is_empty() {
        // Edge-edge: no corner of either box lies inside the other, so the
        // boxes meet along two crossing edges. The separating-axis minimum
        // gives the right depth and normal, but the contact *point* is in the
        // interior of each edge, not at a corner.
        //
        // Appendix D M-07: this used the most negative corner of `a` and the
        // most positive corner of `b` along the normal. Those are generally
        // metres away from where the boxes actually touch, so the depth was
        // right and the moment arm was wrong, and `wrench_at` produced a
        // torque that did not correspond to the contact. A box resting on an
        // edge received a correct push and an incorrect spin.
        //
        // The two edges are found as the ones most nearly perpendicular to the
        // normal and most extreme along it, which is the standard
        // construction; their closest points are then the contact. Falling
        // back to the previous corner pair would reintroduce the defect, so a
        // failure to identify an edge reports no manifold instead.
        // The contacting feature of a box along a direction is its support
        // set: the corners extreme along that direction. Two such corners mean
        // an edge, which is the case here. This is more robust than picking
        // the edge from axis heuristics, because it makes no assumption about
        // how the box is oriented relative to the normal.
        let support_feature = |c: V3, h: V3, axes: &[V3; 3], dir: V3| -> Vec<V3> {
            let cs = corners(c, h, axes);
            let best = cs
                .iter()
                .map(|p| dot(*p, dir))
                .fold(f64::NEG_INFINITY, f64::max);
            // Tolerance relative to the box's own extent, so the set is stable
            // for a box lying flat against the direction.
            let extent = h[0].abs() + h[1].abs() + h[2].abs();
            let tol = 1e-9 * extent.max(1.0);
            cs.into_iter()
                .filter(|p| dot(*p, dir) >= best - tol)
                .collect()
        };
        // `n` points from b toward a, so a's contacting feature is extreme
        // along -n and b's along +n.
        let fa = support_feature(ca, ha, &axes_a, scale(n, -1.0));
        let fb = support_feature(cb, hb, &axes_b, n);
        let segment = |f: &[V3]| -> (V3, V3) {
            match f.len() {
                0 => ([0.0; 3], [0.0; 3]),
                1 => (f[0], f[0]),
                // Two corners are an edge; more than two is a face, and its
                // two most separated corners span it well enough for the
                // closest-point query below.
                _ => {
                    let mut best = (f[0], f[1]);
                    let mut far = -1.0;
                    for i in 0..f.len() {
                        for j in (i + 1)..f.len() {
                            let d = norm2(sub(f[i], f[j]));
                            if d > far {
                                far = d;
                                best = (f[i], f[j]);
                            }
                        }
                    }
                    best
                }
            }
        };
        if fa.is_empty() || fb.is_empty() {
            return None;
        }
        let (a0, a1) = segment(&fa);
        let (b0, b1) = segment(&fb);
        let (pa, pb) = closest_segment_segment(a0, a1, b0, b1);
        points.push(Contact {
            depth,
            normal: n,
            point_a: pa,
            point_b: pb,
        });
    }
    points.truncate(8);
    Some(points)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> Mat3 {
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    }

    fn at(shape: Primitive, t: V3) -> Placed {
        Placed {
            shape,
            rotation: identity(),
            translation: t,
        }
    }

    fn rot_z(angle: f64) -> Mat3 {
        let (s, c) = angle.sin_cos();
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    }

    #[test]
    fn two_spheres_overlap_by_the_sum_of_radii_minus_the_distance() {
        let a = at(Primitive::Sphere { radius: 0.5 }, [0.0, 0.0, 0.0]);
        let b = at(Primitive::Sphere { radius: 0.5 }, [0.8, 0.0, 0.0]);
        match proximity(&a, &b) {
            Proximity::Overlap(c) => {
                assert!((c.depth - 0.2).abs() < 1e-12, "{c:?}");
                assert!(
                    (c.normal[0] + 1.0).abs() < 1e-12,
                    "a is pushed away from b: {c:?}"
                );
            }
            other => panic!("expected overlap, got {other:?}"),
        }
    }

    #[test]
    fn two_capsules_match_the_analytic_segment_distance() {
        // Parallel capsules side by side: the segments are 1.2 apart, radii 0.4.
        let a = at(
            Primitive::Capsule {
                half_height: 0.5,
                radius: 0.4,
            },
            [0.0, 0.0, 0.0],
        );
        let b = at(
            Primitive::Capsule {
                half_height: 0.5,
                radius: 0.4,
            },
            [1.2, 0.0, 0.0],
        );
        match proximity(&a, &b) {
            Proximity::Apart { distance, .. } => {
                assert!((distance - 0.4).abs() < 1e-12, "{distance}")
            }
            other => panic!("expected apart, got {other:?}"),
        }
    }

    #[test]
    fn a_capsule_end_touching_a_sphere_is_a_point_contact() {
        // Capsule along Z, its upper end centre at z = 0.5. Sphere radius 0.25
        // above it, touching the capsule's end cap exactly.
        let capsule = at(
            Primitive::Capsule {
                half_height: 0.5,
                radius: 0.25,
            },
            [0.0, 0.0, 0.0],
        );
        let sphere = at(Primitive::Sphere { radius: 0.25 }, [0.0, 0.0, 1.0]);
        match proximity(&capsule, &sphere) {
            Proximity::Apart { distance, .. } => assert!(distance.abs() < 1e-12, "{distance}"),
            Proximity::Overlap(c) => assert!(c.depth.abs() < 1e-12, "{c:?}"),
        }
    }

    #[test]
    fn gjk_agrees_with_the_analytic_capsule_test() {
        let a = Placed {
            shape: Primitive::Capsule {
                half_height: 0.6,
                radius: 0.3,
            },
            rotation: rot_z(0.4),
            translation: [0.1, -0.2, 0.0],
        };
        let b = Placed {
            shape: Primitive::Capsule {
                half_height: 0.4,
                radius: 0.2,
            },
            rotation: rot_z(-1.1),
            translation: [1.3, 0.4, 0.2],
        };
        let analytic = proximity(&a, &b);
        let general = gjk_epa(&a, &b);
        let dist = |p: Proximity| match p {
            Proximity::Apart { distance, .. } => distance,
            Proximity::Overlap(c) => -c.depth,
        };
        assert!(
            (dist(analytic) - dist(general)).abs() < 1e-7,
            "{analytic:?} vs {general:?}"
        );
    }

    #[test]
    fn boxes_apart_report_their_gap() {
        let a = at(
            Primitive::Box {
                half: [0.5, 0.5, 0.5],
            },
            [0.0, 0.0, 0.0],
        );
        let b = at(
            Primitive::Box {
                half: [0.5, 0.5, 0.5],
            },
            [2.0, 0.0, 0.0],
        );
        match proximity(&a, &b) {
            Proximity::Apart {
                distance, normal, ..
            } => {
                assert!((distance - 1.0).abs() < 1e-9, "{distance}");
                // Apart normals point from b toward a: a sits at the origin, b at +2.
                assert!((normal[0] + 1.0).abs() < 1e-9, "{normal:?}");
            }
            other => panic!("expected apart, got {other:?}"),
        }
    }

    #[test]
    fn overlapping_boxes_give_the_face_depth_and_normal() {
        // Along X the boxes overlap by 0.1 (a spans [0.4, 1.4], b spans [-0.5, 0.5]);
        // along Y and Z the overlap is 0.9 and 1.0. The least penetration is X.
        let a = at(
            Primitive::Box {
                half: [0.5, 0.5, 0.5],
            },
            [0.9, 0.1, 0.0],
        );
        let b = at(
            Primitive::Box {
                half: [0.5, 0.5, 0.5],
            },
            [0.0, 0.0, 0.0],
        );
        match proximity(&a, &b) {
            Proximity::Overlap(c) => {
                assert!((c.depth - 0.1).abs() < 1e-6, "{c:?}");
                assert!((c.normal[0] - 1.0).abs() < 1e-6, "{c:?}");
            }
            other => panic!("expected overlap, got {other:?}"),
        }
    }

    #[test]
    fn a_rotated_box_on_a_box_is_resolved_along_the_smallest_axis() {
        // A box rotated 45° about Z dips 0.1 below the top face of a larger box.
        let top = at(
            Primitive::Box {
                half: [1.0, 1.0, 0.5],
            },
            [0.0, 0.0, 0.0],
        );
        let tilted = Placed {
            shape: Primitive::Box {
                half: [0.3, 0.3, 0.3],
            },
            rotation: rot_z(0.7),
            // Rotation about Z keeps the vertical half extent, so the bottom face is
            // at centre - 0.3. Put it 0.1 below the top face at z = 0.5.
            translation: [0.0, 0.0, 0.5 + 0.3 - 0.1],
        };
        match proximity(&tilted, &top) {
            Proximity::Overlap(c) => {
                assert!(c.depth > 0.0 && c.depth < 0.3, "{c:?}");
                assert!(c.normal[2] > 0.9, "the push-out direction is up: {c:?}");
            }
            other => panic!("expected overlap, got {other:?}"),
        }
    }

    #[test]
    fn a_cylinder_standing_on_a_box_overlaps_by_the_stated_amount() {
        let base = at(
            Primitive::Box {
                half: [1.0, 1.0, 0.5],
            },
            [0.0, 0.0, 0.0],
        );
        // Cylinder half height 0.4, centre at 0.5 + 0.4 - 0.05: 0.05 into the box.
        let cyl = at(
            Primitive::Cylinder {
                radius: 0.3,
                half_height: 0.4,
            },
            [0.0, 0.0, 0.85],
        );
        match proximity(&cyl, &base) {
            Proximity::Overlap(c) => {
                assert!((c.depth - 0.05).abs() < 1e-6, "{c:?}");
                assert!((c.normal[2] - 1.0).abs() < 1e-6, "{c:?}");
            }
            other => panic!("expected overlap, got {other:?}"),
        }
    }

    #[test]
    fn plane_points_of_a_box_are_its_eight_corners() {
        let b = at(
            Primitive::Box {
                half: [0.5, 0.25, 0.1],
            },
            [0.0, 0.0, 1.0],
        );
        let pts = plane_contact_points(&b, 8);
        assert_eq!(pts.len(), 8);
        let lowest = pts.iter().map(|(c, _)| c[2]).fold(f64::INFINITY, f64::min);
        assert!((lowest - 0.9).abs() < 1e-12);
    }

    #[test]
    fn a_capsule_yields_its_two_end_spheres_for_the_plane() {
        let c = at(
            Primitive::Capsule {
                half_height: 0.5,
                radius: 0.1,
            },
            [0.0, 0.0, 0.0],
        );
        let pts = plane_contact_points(&c, 8);
        assert_eq!(pts.len(), 2);
        assert!(pts.iter().all(|(_, r)| (*r - 0.1).abs() < 1e-12));
    }
    #[test]
    fn an_edge_edge_box_contact_is_placed_on_the_edges_not_at_a_corner() {
        // Two long thin boxes crossing like an X, one above the other, rotated
        // 45 degrees about z relative to each other so that no corner of
        // either lies inside the other. They touch near the world origin,
        // where their edges cross.
        //
        // Appendix D M-07: the contact used to be reported at the extreme
        // corners along the normal, which for these boxes are about 0.7 m from
        // where they actually meet. The force was therefore applied with a
        // moment arm that does not exist.
        let lower = Placed {
            shape: Primitive::Box {
                half: [1.0, 0.05, 0.05],
            },
            rotation: identity(),
            translation: [0.0, 0.0, 0.0],
        };
        let upper = Placed {
            shape: Primitive::Box {
                half: [1.0, 0.05, 0.05],
            },
            rotation: rot_z(core::f64::consts::FRAC_PI_2),
            translation: [0.0, 0.0, 0.09],
        };
        let contacts = box_box_contacts(&upper, &lower).expect("the boxes overlap");
        assert!(!contacts.is_empty());
        for c in &contacts {
            // The normal is +z (the least-overlap axis is the thin one).
            assert!(
                c.normal[2].abs() > 0.99,
                "normal should be vertical: {:?}",
                c.normal
            );
            // The contact must be where the edges cross, near the origin in x
            // and y, not out at a corner about 0.7 m away.
            assert!(
                c.point_a[0].abs() < 0.2 && c.point_a[1].abs() < 0.2,
                "point_a {:?} is not near the crossing",
                c.point_a
            );
            assert!(
                c.point_b[0].abs() < 0.2 && c.point_b[1].abs() < 0.2,
                "point_b {:?} is not near the crossing",
                c.point_b
            );
            // and the two witness points are close together, since the
            // surfaces are 0.01 m apart at the crossing.
            let gap = norm(sub(c.point_a, c.point_b));
            assert!(gap < 0.15, "witness points {gap} apart");
        }
    }
    #[test]
    fn the_subdistance_gives_four_non_zero_weights_only_when_the_origin_is_inside() {
        // This is the invariant the expansion depends on, and the reason the
        // simplex reduction was replaced (Appendix D M-05, M-06). A
        // tetrahedron strictly containing the origin must produce four
        // non-zero barycentric weights; one that does not must produce at
        // least one exact zero.
        let v = |w: V3| Vertex {
            w,
            a: w,
            b: [0.0; 3],
        };

        // Origin strictly inside.
        let inside = [
            v([1.0, 1.0, 1.0]),
            v([-1.0, -1.0, 1.0]),
            v([-1.0, 1.0, -1.0]),
            v([1.0, -1.0, -1.0]),
        ];
        let l = subdistance(&inside);
        assert!(
            l.iter().all(|x| *x != 0.0),
            "origin inside must give four non-zero weights, got {l:?}"
        );
        assert!(
            (l.iter().sum::<f64>() - 1.0).abs() < 1e-12,
            "weights must sum to one: {l:?}"
        );

        // The same tetrahedron translated so the origin is well outside.
        let outside: Vec<Vertex> = inside
            .iter()
            .map(|x| v(add(x.w, [10.0, 0.0, 0.0])))
            .collect();
        let l = subdistance(&outside);
        assert!(
            l.contains(&0.0),
            "origin outside must zero at least one weight, got {l:?}"
        );

        // A flat tetrahedron has no interior, so it cannot contain the origin
        // however close the origin is to its plane. This is the case that used
        // to be reported as containment.
        let flat = [
            v([1.0, 1.0, 0.0]),
            v([-1.0, -1.0, 0.0]),
            v([-1.0, 1.0, 0.0]),
            v([1.0, -1.0, 0.0]),
        ];
        let l = subdistance(&flat);
        assert!(
            l.contains(&0.0),
            "a degenerate simplex must not report containment, got {l:?}"
        );
    }

    #[test]
    fn the_subdistance_recovers_the_closest_point_on_a_segment_and_a_triangle() {
        let v = |w: V3| Vertex {
            w,
            a: w,
            b: [0.0; 3],
        };
        // Segment from (1,1,0) to (1,-1,0): closest point to the origin is
        // (1,0,0), so the weights are (0.5, 0.5).
        let seg = [v([1.0, 1.0, 0.0]), v([1.0, -1.0, 0.0])];
        let l = subdistance(&seg);
        assert!(
            (l[0] - 0.5).abs() < 1e-12 && (l[1] - 0.5).abs() < 1e-12,
            "{l:?}"
        );

        // Segment entirely on one side: the closest point is an endpoint, and
        // the example always keeps the newest vertex when outside.
        let seg = [v([1.0, 2.0, 0.0]), v([1.0, 3.0, 0.0])];
        let l = subdistance(&seg);
        assert!(l[0] == 0.0 || l[1] == 0.0, "an endpoint must win: {l:?}");

        // Triangle in the plane x = 1 containing (1,0,0): all three weights
        // non-zero and the combination is the closest point.
        let tri = [v([1.0, 1.0, 1.0]), v([1.0, -1.0, 1.0]), v([1.0, 0.0, -1.0])];
        let l = subdistance(&tri);
        assert!(l[..3].iter().all(|x| *x != 0.0), "{l:?}");
        let closest = (0..3).fold([0.0; 3], |acc, i| add(acc, scale(tri[i].w, l[i])));
        assert!((closest[0] - 1.0).abs() < 1e-12, "{closest:?}");
        assert!(
            closest[1].abs() < 1e-12 && closest[2].abs() < 1e-12,
            "{closest:?}"
        );
    }

    #[test]
    fn a_deeply_overlapping_pair_of_boxes_reports_the_face_depth_through_gjk() {
        // Deep overlap, where the origin is well inside the Minkowski
        // difference and the simplex ends on a full tetrahedron. The least
        // overlap is along z (0.2 against 1.8 and 1.8).
        let a = at(
            Primitive::Box {
                half: [1.0, 1.0, 1.0],
            },
            [0.1, 0.1, 1.8],
        );
        let b = at(
            Primitive::Box {
                half: [1.0, 1.0, 1.0],
            },
            [0.0, 0.0, 0.0],
        );
        match gjk_epa(&a, &b) {
            Proximity::Overlap(c) => {
                assert!((c.depth - 0.2).abs() < 1e-6, "{c:?}");
                assert!(c.normal[2] > 0.99, "a is pushed up: {c:?}");
            }
            other => panic!("expected overlap, got {other:?}"),
        }
    }

    #[test]
    fn two_coincident_boxes_do_not_fabricate_a_contact_normal() {
        // Exactly coincident shapes are the degenerate case: the Minkowski
        // difference is centred on the origin and no separating direction is
        // meaningful. Appendix D M-05 records that this used to come back as an
        // `Overlap` with zero depth and a normal invented from the difference
        // of the two frame origins, which for coincident shapes is the zero
        // vector. Whatever is reported must not claim a penetration depth it
        // cannot support.
        let a = at(
            Primitive::Box {
                half: [0.5, 0.5, 0.5],
            },
            [0.0, 0.0, 0.0],
        );
        let b = a;
        match gjk_epa(&a, &b) {
            // Either answer is acceptable; a fabricated depth is not.
            Proximity::Overlap(c) => {
                assert!(c.depth.is_finite() && c.depth >= 0.0, "{c:?}");
                let n = norm(c.normal);
                assert!((n - 1.0).abs() < 1e-9, "normal must be unit: {c:?}");
            }
            Proximity::Apart { distance, .. } => {
                assert!(distance.abs() < 1e-9, "coincident shapes are not apart");
            }
        }
    }
    #[test]
    fn the_world_aabb_bounds_every_primitive_it_is_asked_about() {
        // Checked against the support function, which is independent of the
        // bounding-box formulas: the box must contain the extreme point in
        // each of the six axis directions, and must not be loose by more than
        // a tolerance in any of them.
        let shapes = [
            Primitive::Sphere { radius: 0.3 },
            Primitive::Capsule {
                half_height: 0.4,
                radius: 0.15,
            },
            Primitive::Box {
                half: [0.2, 0.35, 0.1],
            },
            Primitive::Cylinder {
                radius: 0.25,
                half_height: 0.3,
            },
        ];
        let rotations = [identity(), rot_z(0.6), rot_z(-1.2)];
        for shape in shapes {
            for rotation in rotations {
                let placed = Placed {
                    shape,
                    rotation,
                    translation: [0.7, -0.2, 0.4],
                };
                let (c, h) = placed.world_aabb();
                for k in 0..3 {
                    for sign in [-1.0, 1.0] {
                        let mut dir = [0.0; 3];
                        dir[k] = sign;
                        let extreme = dot(placed.support(dir), dir);
                        let face = sign * c[k] + h[k];
                        assert!(
                            extreme <= face + 1e-9,
                            "{shape:?} axis {k} sign {sign}: support {extreme} beyond face {face}"
                        );
                        assert!(
                            face - extreme < 1e-6,
                            "{shape:?} axis {k} sign {sign}: box loose by {}",
                            face - extreme
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_broad_phase_rejects_most_distant_pairs() {
        // The point of the filters is that they are cheap and reject a lot.
        // Two shapes a metre apart must be rejected by the first test.
        let a = at(Primitive::Sphere { radius: 0.1 }, [0.0, 0.0, 0.0]);
        let b = at(Primitive::Sphere { radius: 0.1 }, [1.0, 0.0, 0.0]);
        assert!(spheres_apart(&a, &b, 0.0));
        assert!(aabbs_apart(&a, &b, 0.0));
        // Just touching: neither may reject.
        let c = at(Primitive::Sphere { radius: 0.1 }, [0.2, 0.0, 0.0]);
        assert!(!spheres_apart(&a, &c, 0.0));
        assert!(!aabbs_apart(&a, &c, 0.0));
        // The margin widens the test, so a gap below it is kept.
        let d = at(Primitive::Sphere { radius: 0.1 }, [0.25, 0.0, 0.0]);
        assert!(spheres_apart(&a, &d, 0.0));
        assert!(!spheres_apart(&a, &d, 0.1));
    }
}

#[cfg(test)]
mod fuzz_tests {
    use super::*;

    fn at(shape: Primitive, t: V3) -> Placed {
        Placed {
            shape,
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: t,
        }
    }

    fn rot_z(angle: f64) -> Mat3 {
        let (s, c) = angle.sin_cos();
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    }

    /// A deterministic LCG in [0, 1).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        fn range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.next()
        }
        fn rotation(&mut self) -> Mat3 {
            // Rotation from a random unit quaternion.
            let q = [
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
            ];
            let n = q.iter().map(|x| x * x).sum::<f64>().sqrt();
            let (w, x, y, z) = (q[0] / n, q[1] / n, q[2] / n, q[3] / n);
            [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - w * z),
                    2.0 * (x * z + w * y),
                ],
                [
                    2.0 * (x * y + w * z),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - w * x),
                ],
                [
                    2.0 * (x * z - w * y),
                    2.0 * (y * z + w * x),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ]
        }
    }

    fn signed_distance(p: Proximity) -> f64 {
        match p {
            Proximity::Apart { distance, .. } => distance,
            Proximity::Overlap(c) => -c.depth,
        }
    }

    #[test]
    fn random_capsule_pairs_agree_between_the_analytic_test_and_gjk() {
        let mut rng = Lcg(7);
        for case in 0..300 {
            let a = Placed {
                shape: Primitive::Capsule {
                    half_height: rng.range(0.05, 0.8),
                    radius: rng.range(0.05, 0.4),
                },
                rotation: rng.rotation(),
                translation: [
                    rng.range(-1.0, 1.0),
                    rng.range(-1.0, 1.0),
                    rng.range(-1.0, 1.0),
                ],
            };
            let b = Placed {
                shape: Primitive::Capsule {
                    half_height: rng.range(0.05, 0.8),
                    radius: rng.range(0.05, 0.4),
                },
                rotation: rng.rotation(),
                translation: [
                    rng.range(-1.0, 1.0),
                    rng.range(-1.0, 1.0),
                    rng.range(-1.0, 1.0),
                ],
            };
            let analytic = signed_distance(proximity(&a, &b));
            let general = signed_distance(gjk_epa(&a, &b));
            assert!(
                (analytic - general).abs() < 1e-6,
                "case {case}: analytic {analytic} vs gjk {general}"
            );
        }
    }

    /// The exact penetration of two boxes by the separating axis theorem: the
    /// least overlap over the 15 candidate axes (three face normals of each box
    /// and the nine edge cross products).
    fn sat_depth(a: &Placed, b: &Placed) -> Option<f64> {
        let Primitive::Box { half: ha } = a.shape else {
            return None;
        };
        let Primitive::Box { half: hb } = b.shape else {
            return None;
        };
        let axes_a = [
            [a.rotation[0][0], a.rotation[1][0], a.rotation[2][0]],
            [a.rotation[0][1], a.rotation[1][1], a.rotation[2][1]],
            [a.rotation[0][2], a.rotation[1][2], a.rotation[2][2]],
        ];
        let axes_b = [
            [b.rotation[0][0], b.rotation[1][0], b.rotation[2][0]],
            [b.rotation[0][1], b.rotation[1][1], b.rotation[2][1]],
            [b.rotation[0][2], b.rotation[1][2], b.rotation[2][2]],
        ];
        let mut candidates: Vec<V3> = Vec::new();
        candidates.extend(axes_a);
        candidates.extend(axes_b);
        for ea in axes_a {
            for eb in axes_b {
                candidates.push(cross(ea, eb));
            }
        }
        let d = sub(b.translation, a.translation);
        let mut best = f64::INFINITY;
        for axis in candidates {
            let len = norm(axis);
            if len < 1e-9 {
                continue;
            }
            let u = scale(axis, 1.0 / len);
            let radius = |axes: &[V3; 3], h: V3| -> f64 {
                (0..3).map(|i| h[i] * dot(axes[i], u).abs()).sum()
            };
            let ra = radius(&axes_a, ha);
            let rb = radius(&axes_b, hb);
            let overlap = ra + rb - dot(d, u).abs();
            // A separating axis: report its negative overlap, so the sign is
            // negative exactly when the boxes are apart.
            if overlap < 0.0 {
                return Some(overlap);
            }
            best = best.min(overlap);
        }
        Some(best)
    }

    #[test]
    fn a_box_resting_flat_on_a_box_gives_its_four_bottom_corners() {
        // Lower box top at z = 0.1; upper box bottom 0.01 below it.
        let lower = at(
            Primitive::Box {
                half: [0.5, 0.5, 0.1],
            },
            [0.0, 0.0, 0.0],
        );
        let upper = at(
            Primitive::Box {
                half: [0.5, 0.5, 0.1],
            },
            [0.0, 0.0, 0.19],
        );
        let contacts = box_box_contacts(&upper, &lower).expect("overlapping");
        assert_eq!(contacts.len(), 4, "{contacts:?}");
        for c in &contacts {
            assert!((c.depth - 0.01).abs() < 1e-9, "{c:?}");
            assert!((c.normal[2] - 1.0).abs() < 1e-9, "{c:?}");
        }
        let total: f64 = contacts.iter().map(|c| c.depth).sum();
        assert!((total - 0.04).abs() < 1e-9);
    }

    #[test]
    fn a_tilted_box_on_a_box_still_gets_contacts_pushing_up() {
        let lower = at(
            Primitive::Box {
                half: [0.5, 0.5, 0.1],
            },
            [0.0, 0.0, 0.0],
        );
        let upper = Placed {
            shape: Primitive::Box {
                half: [0.3, 0.3, 0.1],
            },
            rotation: rot_z(0.4),
            translation: [0.05, 0.0, 0.19],
        };
        let contacts = box_box_contacts(&upper, &lower).expect("overlapping");
        assert!(!contacts.is_empty());
        assert!(contacts.iter().all(|c| c.normal[2] > 0.99), "{contacts:?}");
        assert!(contacts.iter().all(|c| c.depth > 0.0 && c.depth < 0.05));
    }

    #[test]
    fn separated_boxes_give_no_manifold() {
        let a = at(
            Primitive::Box {
                half: [0.5, 0.5, 0.5],
            },
            [0.0, 0.0, 0.0],
        );
        let b = at(
            Primitive::Box {
                half: [0.5, 0.5, 0.5],
            },
            [0.0, 0.0, 1.2],
        );
        assert!(box_box_contacts(&a, &b).is_none());
        assert!(box_sat(&a, &b).is_none());
    }

    #[test]
    fn the_manifold_normal_agrees_with_the_separating_axis_on_random_boxes() {
        let mut rng = Lcg(23);
        let mut checked = 0;
        for case in 0..300 {
            let a = Placed {
                shape: Primitive::Box {
                    half: [
                        rng.range(0.1, 0.6),
                        rng.range(0.1, 0.6),
                        rng.range(0.1, 0.6),
                    ],
                },
                rotation: rng.rotation(),
                translation: [
                    rng.range(-0.6, 0.6),
                    rng.range(-0.6, 0.6),
                    rng.range(-0.6, 0.6),
                ],
            };
            let b = Placed {
                shape: Primitive::Box {
                    half: [
                        rng.range(0.1, 0.6),
                        rng.range(0.1, 0.6),
                        rng.range(0.1, 0.6),
                    ],
                },
                rotation: rng.rotation(),
                translation: [0.0, 0.0, 0.0],
            };
            let exact = sat_depth(&a, &b).unwrap();
            match box_box_contacts(&a, &b) {
                Some(contacts) => {
                    assert!(exact > 0.0, "case {case}: manifold for separated boxes");
                    let (depth, normal) = box_sat(&a, &b).unwrap();
                    assert!((depth - exact).abs() < 1e-9, "case {case}");
                    for c in &contacts {
                        assert!(dot(c.normal, normal) > 1.0 - 1e-9, "case {case}");
                        assert!(c.depth > 0.0, "case {case}");
                    }
                    checked += 1;
                }
                None => assert!(
                    exact <= 1e-9,
                    "case {case}: separated but SAT overlaps {exact}"
                ),
            }
        }
        assert!(checked > 50, "{checked} overlapping cases");
    }

    #[test]
    fn random_overlapping_boxes_match_the_separating_axis_depth() {
        let mut rng = Lcg(11);
        let mut overlapping = 0;
        for case in 0..300 {
            let a = Placed {
                shape: Primitive::Box {
                    half: [
                        rng.range(0.1, 0.6),
                        rng.range(0.1, 0.6),
                        rng.range(0.1, 0.6),
                    ],
                },
                rotation: rng.rotation(),
                translation: [
                    rng.range(-0.6, 0.6),
                    rng.range(-0.6, 0.6),
                    rng.range(-0.6, 0.6),
                ],
            };
            let b = Placed {
                shape: Primitive::Box {
                    half: [
                        rng.range(0.1, 0.6),
                        rng.range(0.1, 0.6),
                        rng.range(0.1, 0.6),
                    ],
                },
                rotation: rng.rotation(),
                translation: [0.0, 0.0, 0.0],
            };
            let exact = sat_depth(&a, &b).unwrap();
            match proximity(&a, &b) {
                Proximity::Overlap(c) => {
                    overlapping += 1;
                    assert!(
                        exact > 0.0,
                        "case {case}: GJK overlap but SAT separates by {exact}"
                    );
                    assert!(
                        (c.depth - exact).abs() < 1e-5,
                        "case {case}: EPA depth {} vs SAT {exact} A={a:?} B={b:?}",
                        c.depth
                    );
                    assert!((norm(c.normal) - 1.0).abs() < 1e-9);
                }
                Proximity::Apart { distance, .. } => {
                    assert!(
                        exact <= 1e-9,
                        "case {case}: GJK apart at {distance} but SAT overlaps by {exact}"
                    );
                }
            }
        }
        assert!(
            overlapping > 50,
            "the sample must contain overlaps ({overlapping})"
        );
    }
    #[test]
    fn the_broad_phase_filters_never_reject_a_touching_pair() {
        // A filter that discards a pair the narrow phase would have reported is
        // a silent loss of contact, so the only property that matters is this
        // one: whenever the shapes actually overlap, neither filter rejects
        // them. Checked over randomised pairs.
        let mut rng = Lcg(23);
        let mut overlaps = 0;
        for case in 0..400 {
            let a = Placed {
                shape: Primitive::Box {
                    half: [
                        rng.range(0.1, 0.4),
                        rng.range(0.1, 0.4),
                        rng.range(0.1, 0.4),
                    ],
                },
                rotation: rng.rotation(),
                translation: [
                    rng.range(-0.6, 0.6),
                    rng.range(-0.6, 0.6),
                    rng.range(-0.6, 0.6),
                ],
            };
            let b = Placed {
                shape: Primitive::Cylinder {
                    radius: rng.range(0.1, 0.3),
                    half_height: rng.range(0.1, 0.4),
                },
                rotation: rng.rotation(),
                translation: [0.0, 0.0, 0.0],
            };
            if let Proximity::Overlap(_) = proximity(&a, &b) {
                overlaps += 1;
                assert!(
                    !spheres_apart(&a, &b, 0.0),
                    "case {case}: bounding spheres rejected an overlapping pair"
                );
                assert!(
                    !aabbs_apart(&a, &b, 0.0),
                    "case {case}: AABBs rejected an overlapping pair"
                );
            }
        }
        assert!(overlaps > 40, "only {overlaps} overlapping cases generated");
    }
}

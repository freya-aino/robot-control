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

/// Signed volume of the tetrahedron `(p, q, r, s)` times six.
fn orient(p: V3, q: V3, r: V3, s: V3) -> f64 {
    dot(cross(sub(q, p), sub(r, p)), sub(s, p))
}

/// A reduced simplex: its vertices with their barycentric weights, the closest
/// point to the origin, and the squared distance of that point.
type SimplexFeature = (Vec<(Vertex, f64)>, V3, f64);

/// Reduces a simplex of 1 to 4 vertices to the feature closest to the origin.
/// Returns the reduced vertices with the barycentric weight of each, the closest
/// point, and whether the origin lies inside the simplex.
fn reduce(vs: &[Vertex]) -> (Vec<(Vertex, f64)>, V3, bool) {
    match vs.len() {
        1 => (vec![(vs[0], 1.0)], vs[0].w, false),
        2 => {
            let (p0, p1) = (vs[0].w, vs[1].w);
            let e = sub(p1, p0);
            let ee = norm2(e);
            let t = if ee > 0.0 { -dot(p0, e) / ee } else { 0.0 };
            if t <= 0.0 {
                (vec![(vs[0], 1.0)], p0, false)
            } else if t >= 1.0 {
                (vec![(vs[1], 1.0)], p1, false)
            } else {
                let q = add(p0, scale(e, t));
                (vec![(vs[0], 1.0 - t), (vs[1], t)], q, false)
            }
        }
        3 => {
            let w = closest_point_triangle_origin(vs[0].w, vs[1].w, vs[2].w);
            let q = [0, 1, 2]
                .iter()
                .fold([0.0; 3], |acc, &i| add(acc, scale(vs[i].w, w[i])));
            let kept: Vec<(Vertex, f64)> = (0..3)
                .filter(|&i| w[i] > 1e-14)
                .map(|i| (vs[i], w[i]))
                .collect();
            (kept, q, norm2(q) < 1e-24)
        }
        _ => {
            // Tetrahedron: origin inside if it is on the inner side of every face.
            let mut best: Option<SimplexFeature> = None;
            let mut inside = true;
            for skip in 0..4 {
                let idx: Vec<usize> = (0..4).filter(|&i| i != skip).collect();
                let (p, q, r) = (vs[idx[0]].w, vs[idx[1]].w, vs[idx[2]].w);
                let opposite = vs[skip].w;
                let s_opp = orient(p, q, r, opposite);
                let s_origin = orient(p, q, r, [0.0; 3]);
                if s_origin * s_opp < 0.0 {
                    inside = false;
                    let tri = [vs[idx[0]], vs[idx[1]], vs[idx[2]]];
                    let (kept, c, _) = reduce(&tri);
                    let d = norm2(c);
                    if best.as_ref().is_none_or(|(_, _, bd)| d < *bd) {
                        best = Some((kept, c, d));
                    }
                }
            }
            if inside {
                let kept = vs.iter().map(|v| (*v, 0.25)).collect();
                return (kept, [0.0; 3], true);
            }
            let (kept, c, _) = best.expect("an outside face exists when the origin is outside");
            (kept, c, false)
        }
    }
}

/// Grows a degenerate simplex to a tetrahedron that encloses the origin, by
/// probing along axes and their cross products.
fn complete_simplex(a: &Placed, b: &Placed, mut vs: Vec<Vertex>) -> Vec<Vertex> {
    let axes: [V3; 6] = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    let mut candidates: Vec<V3> = axes.to_vec();
    if vs.len() == 2 {
        let e = sub(vs[1].w, vs[0].w);
        for ax in &axes {
            candidates.push(cross(e, *ax));
        }
    }
    if vs.len() == 3 {
        let n = cross(sub(vs[1].w, vs[0].w), sub(vs[2].w, vs[0].w));
        candidates.push(n);
        candidates.push(scale(n, -1.0));
    }
    for d in candidates {
        if vs.len() == 4 {
            break;
        }
        if norm2(d) < 1e-24 {
            continue;
        }
        let v = msupport(a, b, d);
        let independent = match vs.len() {
            1 => norm2(sub(v.w, vs[0].w)) > 1e-18,
            2 => norm2(cross(sub(vs[1].w, vs[0].w), sub(v.w, vs[0].w))) > 1e-18,
            _ => orient(vs[0].w, vs[1].w, vs[2].w, v.w).abs() > 1e-18,
        };
        if independent {
            vs.push(v);
        }
    }
    vs
}

/// EPA: expands the polytope around the origin until it reaches the Minkowski
/// boundary, and returns the closest face's depth and normal.
fn epa(a: &Placed, b: &Placed, tetra: [Vertex; 4]) -> Contact {
    let mut verts: Vec<Vertex> = tetra.to_vec();
    // A face is (indices, outward unit normal, distance from origin). The origin
    // may lie on a face of the starting tetrahedron, so faces are oriented with
    // respect to an interior point (the tetrahedron's centroid), not the origin.
    let interior = scale(tetra.iter().fold([0.0; 3], |acc, v| add(acc, v.w)), 0.25);
    let make_face = |verts: &[Vertex], i: usize, j: usize, k: usize| -> ([usize; 3], V3, f64) {
        let mut n = normalized_or(
            cross(sub(verts[j].w, verts[i].w), sub(verts[k].w, verts[i].w)),
            [0.0, 0.0, 1.0],
        );
        // Keep the winding consistent with the normal: a flipped face is
        // traversed the other way, so horizon edges of neighbours cancel.
        if dot(n, sub(verts[i].w, interior)) < 0.0 {
            n = scale(n, -1.0);
            let d = dot(n, verts[i].w);
            return ([i, k, j], n, d);
        }
        let d = dot(n, verts[i].w);
        ([i, j, k], n, d)
    };
    let mut faces: Vec<([usize; 3], V3, f64)> = Vec::new();
    for &(i, j, k) in &[(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)] {
        faces.push(make_face(&verts, i, j, k));
    }
    let mut result_normal = [0.0, 0.0, 1.0];
    let mut result_depth = 0.0;
    let mut result_points = (tetra[0].a, tetra[0].b);
    for _ in 0..128 {
        let (fi, _) = faces
            .iter()
            .enumerate()
            .min_by(|x, y| x.1 .2.total_cmp(&y.1 .2))
            .expect("the polytope always has faces");
        let (idx, n, dist) = faces[fi];
        let w = msupport(a, b, n);
        result_normal = n;
        result_depth = dist;
        let on_boundary = dot(w.w, n) - dist <= 1e-10 * (1.0 + dist);
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
            break;
        }
        // Remove the faces the new support point can see; the horizon edges of
        // the removed region connect to it.
        let new_index = verts.len();
        verts.push(w);
        let mut horizon: Vec<(usize, usize)> = Vec::new();
        let mut kept = Vec::with_capacity(faces.len() + 2);
        for face in faces.drain(..) {
            let (f, fnrm, fd) = face;
            let visible = dot(fnrm, w.w) - fd > 1e-12;
            if !visible {
                kept.push((f, fnrm, fd));
                continue;
            }
            for (e0, e1) in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
                if let Some(pos) = horizon.iter().position(|&(x, y)| x == e1 && y == e0) {
                    horizon.swap_remove(pos);
                } else {
                    horizon.push((e0, e1));
                }
            }
        }
        if horizon.is_empty() {
            break;
        }
        for (e0, e1) in horizon {
            kept.push(make_face(&verts, e0, e1, new_index));
        }
        faces = kept;
    }
    Contact {
        depth: result_depth.max(0.0),
        normal: scale(result_normal, -1.0),
        point_a: result_points.0,
        point_b: result_points.1,
    }
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
    let mut d = sub(b.translation, a.translation);
    d = normalized_or(d, [1.0, 0.0, 0.0]);
    let mut simplex: Vec<(Vertex, f64)> = vec![(msupport(a, b, d), 1.0)];
    let mut verts: Vec<Vertex> = vec![simplex[0].0];
    let mut overlap = false;
    let mut closest = verts[0].w;
    for _ in 0..128 {
        let c2 = norm2(closest);
        if c2 < 1e-24 {
            overlap = true;
            break;
        }
        let w = msupport(a, b, scale(closest, -1.0));
        // No point of the difference is closer to the origin: `closest` is the
        // distance.
        if dot(w.w, closest) - c2 >= -1e-12 * (1.0 + c2) {
            break;
        }
        verts.push(w);
        let (reduced, q, contains) = reduce(&verts);
        verts = reduced.iter().map(|(v, _)| *v).collect();
        simplex = reduced;
        closest = q;
        if contains {
            overlap = true;
            break;
        }
    }
    if overlap {
        let tetra_verts = complete_simplex(a, b, verts);
        if tetra_verts.len() == 4 {
            let t = [
                tetra_verts[0],
                tetra_verts[1],
                tetra_verts[2],
                tetra_verts[3],
            ];
            return Proximity::Overlap(epa(a, b, t));
        }
        // Degenerate even after completion: report touching with zero depth.
        let c = scale(add(a.translation, b.translation), 0.5);
        let n = normalized_or(sub(a.translation, b.translation), [0.0, 0.0, 1.0]);
        return Proximity::Overlap(Contact {
            depth: 0.0,
            normal: n,
            point_a: c,
            point_b: c,
        });
    }
    // Apart: the closest point of the difference is `closest`, and the weights of
    // the reduced simplex give the closest points on each shape.
    let mut pa = [0.0; 3];
    let mut pb = [0.0; 3];
    let total: f64 = simplex.iter().map(|(_, w)| w).sum();
    for (v, w) in &simplex {
        let wn = if total > 0.0 { w / total } else { 0.0 };
        pa = add(pa, scale(v.a, wn));
        pb = add(pb, scale(v.b, wn));
    }
    let distance = norm(closest);
    let normal = normalized_or(
        sub(pa, pb),
        normalized_or(sub(a.translation, b.translation), [0.0, 0.0, 1.0]),
    );
    Proximity::Apart {
        distance,
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
}

#[cfg(test)]
mod fuzz_tests {
    use super::*;

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
}

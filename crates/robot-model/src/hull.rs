//! Convex hull of a point cloud, for reducing a collision mesh before any
//! narrow-phase routine sees it.
//!
//! `specs/08-model-compilation-and-sensors.md` item 3 requires mesh collision
//! geometry to be reduced to a convex hull **at compile time, never at
//! simulation time**, and the vendored G1 is why: its collision meshes carry
//! 4266 to 154230 vertices each, median 19116. A support query is
//! `argmax_i (v_i . d)`, so scanning a raw mesh would cost tens of thousands of
//! dot products per GJK iteration, per pair, per step.
//!
//! The support function of a point set equals the support function of its
//! convex hull, so reducing a cloud to its hull vertices changes no collision
//! answer at all — it only removes points that could never win an `argmax`.
//! That is what makes this a pure optimisation rather than an approximation,
//! and it is why this module returns **vertices** and not faces: the narrow
//! phase never asks for a face.
//!
//! The reference engine delegates this to `qhull` at compile time, so the
//! construction is not specified by the example and any correct hull is
//! faithful. This is the incremental (quickhull-style) construction: seed a
//! tetrahedron, then repeatedly add the point farthest outside some face,
//! delete the faces it can see, and stitch the horizon. It is the same
//! operation `robot_sim::collision`'s expanding-polytope step performs, for the
//! same reason.
//!
//! Degenerate input is handled by dimension rather than by refusal, because a
//! real asset contains flat and near-flat collision shells: a cloud that is
//! coplanar, collinear or a single point has no three-dimensional hull, and the
//! extreme points of the shape it does have are returned instead. Those are
//! still exactly the points a support query can select.

/// Smallest value treated as non-zero, relative to the cloud's own extent.
const RELATIVE_EPS: f64 = 1e-12;

/// One triangular face of the growing hull, as indices into the point list and
/// an outward plane.
#[derive(Clone, Debug)]
struct Face {
    vertex: [usize; 3],
    normal: [f64; 3],
    offset: f64,
    live: bool,
    /// Points outside this face, with the farthest first in `furthest`.
    ///
    /// This conflict list is what makes the construction tractable rather than
    /// quadratic. Without it each round rescans every point against every live
    /// face: on a G1 collision mesh that is 19000 points times some hundreds of
    /// faces times some hundreds of rounds, about 3e9 dot products for one
    /// shape. With it, a round only redistributes the points belonging to the
    /// faces it just deleted.
    outside: Vec<usize>,
    /// The farthest outside point and its height above the plane.
    furthest: (usize, f64),
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// The farthest point from the line through `a` and `b`, and its distance.
fn farthest_from_line(points: &[[f64; 3]], a: usize, b: usize) -> (usize, f64) {
    let base = points[a];
    let axis = sub(points[b], base);
    let len = norm(axis);
    if len <= 0.0 {
        return (a, 0.0);
    }
    let unit = [axis[0] / len, axis[1] / len, axis[2] / len];
    let mut best = a;
    let mut best_d = 0.0;
    for (i, p) in points.iter().enumerate() {
        let v = sub(*p, base);
        let along = dot(v, unit);
        let perp = [
            v[0] - along * unit[0],
            v[1] - along * unit[1],
            v[2] - along * unit[2],
        ];
        let d = norm(perp);
        if d > best_d {
            best_d = d;
            best = i;
        }
    }
    (best, best_d)
}

/// Builds a face with an outward normal, given a fourth point known to be
/// inside.
fn face_of(points: &[[f64; 3]], v: [usize; 3], inside: [f64; 3]) -> Option<Face> {
    let n = cross(
        sub(points[v[1]], points[v[0]]),
        sub(points[v[2]], points[v[0]]),
    );
    let len = norm(n);
    if len <= 0.0 {
        return None;
    }
    let mut normal = [n[0] / len, n[1] / len, n[2] / len];
    let mut offset = dot(normal, points[v[0]]);
    // Flip so the known-inside point is below the plane.
    if dot(normal, inside) - offset > 0.0 {
        normal = [-normal[0], -normal[1], -normal[2]];
        offset = -offset;
        return Some(Face {
            vertex: [v[0], v[2], v[1]],
            normal,
            offset,
            live: true,
            outside: Vec::new(),
            furthest: (usize::MAX, 0.0),
        });
    }
    Some(Face {
        vertex: v,
        normal,
        offset,
        live: true,
        outside: Vec::new(),
        furthest: (usize::MAX, 0.0),
    })
}

/// Assigns each candidate to one face it lies outside of, updating that face's
/// farthest point. A point outside several faces is assigned to the one it is
/// farthest above, which is the face that will certainly see it.
fn assign(points: &[[f64; 3]], faces: &mut [Face], candidates: &[usize], eps: f64) {
    for &i in candidates {
        let p = points[i];
        let mut best = usize::MAX;
        let mut best_h = eps;
        for (fi, f) in faces.iter().enumerate() {
            if !f.live {
                continue;
            }
            let h = dot(f.normal, p) - f.offset;
            if h > best_h {
                best_h = h;
                best = fi;
            }
        }
        if best != usize::MAX {
            faces[best].outside.push(i);
            if best_h > faces[best].furthest.1 {
                faces[best].furthest = (i, best_h);
            }
        }
    }
}

/// What the incremental construction produced: either the live faces of a
/// genuinely three-dimensional hull, or the extreme points of a cloud with
/// less than full dimension (a point, a segment or a planar polygon).
enum Reduced {
    Degenerate(Vec<usize>),
    Solid(Vec<Face>),
}

/// The indices of `points` that lie on the convex hull.
///
/// The result is a subset of the input, in ascending index order, and is never
/// empty for a non-empty input. For a cloud with no three-dimensional extent
/// the extreme points of its actual dimension are returned, which is still
/// exactly the set a support query can select.
#[must_use]
pub fn hull_vertices(points: &[[f64; 3]]) -> Vec<usize> {
    if points.len() <= 4 {
        return (0..points.len()).collect();
    }
    match reduce(points) {
        Reduced::Degenerate(out) => out,
        Reduced::Solid(faces) => {
            // The hull's vertices are those still referenced by a live face.
            let mut used = vec![false; points.len()];
            for f in faces.iter().filter(|f| f.live) {
                for v in f.vertex {
                    used[v] = true;
                }
            }
            let out: Vec<usize> = (0..points.len()).filter(|i| used[*i]).collect();
            if out.is_empty() {
                (0..points.len().min(4)).collect()
            } else {
                out
            }
        }
    }
}

/// The outward-oriented triangular faces of the convex hull, as index triples
/// into `points` wound counter-clockwise seen from outside, or `None` for a
/// cloud with no three-dimensional hull (fewer than four points, or collinear
/// or coplanar input).
///
/// [`hull_vertices`] deliberately returns vertices only, because a support
/// query never asks for a face. This entry point exists for the consumer that
/// *does* need the surface: the plane-hull contact manifold and the
/// hill-climbing support walk of `specs/07-collision-pipeline.md` both need
/// the hull's polygon and adjacency structure, and the construction here
/// already builds every face before discarding it.
#[must_use]
pub fn hull_faces(points: &[[f64; 3]]) -> Option<Vec<[usize; 3]>> {
    if points.len() < 4 {
        return None;
    }
    match reduce(points) {
        Reduced::Degenerate(_) => None,
        Reduced::Solid(faces) => {
            let out: Vec<[usize; 3]> = faces.iter().filter(|f| f.live).map(|f| f.vertex).collect();
            if out.len() < 4 {
                None
            } else {
                Some(out)
            }
        }
    }
}

/// The incremental (quickhull-style) construction shared by [`hull_vertices`]
/// and [`hull_faces`]: seed a tetrahedron, insert the farthest outside point,
/// delete the faces it sees, stitch the horizon.
fn reduce(points: &[[f64; 3]]) -> Reduced {
    // Scale the tolerance to the cloud, so a millimetre-scale part and a
    // metre-scale one behave the same way.
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in points {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let extent = (0..3).fold(0.0f64, |m, k| m.max(hi[k] - lo[k]));
    if !extent.is_finite() || extent <= 0.0 {
        return Reduced::Degenerate(vec![0]);
    }
    let eps = extent * RELATIVE_EPS;

    // Seed: the two points farthest apart along the widest axis, then the
    // farthest from that line, then the farthest from that plane.
    let axis = (0..3)
        .max_by(|a, b| {
            (hi[*a] - lo[*a])
                .partial_cmp(&(hi[*b] - lo[*b]))
                .unwrap_or(core::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    let mut a = 0;
    let mut b = 0;
    for (i, p) in points.iter().enumerate() {
        if p[axis] < points[a][axis] {
            a = i;
        }
        if p[axis] > points[b][axis] {
            b = i;
        }
    }
    if a == b {
        return Reduced::Degenerate(vec![a]);
    }
    let (c, line_d) = farthest_from_line(points, a, b);
    if line_d <= eps {
        // Collinear: only the two ends can ever be selected.
        let mut out = [a, b];
        out.sort_unstable();
        return Reduced::Degenerate(out.to_vec());
    }
    let plane_n = cross(sub(points[b], points[a]), sub(points[c], points[a]));
    let plane_len = norm(plane_n);
    let plane_unit = [
        plane_n[0] / plane_len,
        plane_n[1] / plane_len,
        plane_n[2] / plane_len,
    ];
    let plane_off = dot(plane_unit, points[a]);
    let mut d = a;
    let mut best = 0.0;
    for (i, p) in points.iter().enumerate() {
        let h = (dot(plane_unit, *p) - plane_off).abs();
        if h > best {
            best = h;
            d = i;
        }
    }
    if best <= eps {
        // Coplanar. The hull is a polygon; return its vertices by a
        // two-dimensional pass in the plane's own basis.
        return Reduced::Degenerate(planar_hull(points, plane_unit, eps));
    }

    // Seed tetrahedron, every face oriented outward.
    let centroid = [
        (points[a][0] + points[b][0] + points[c][0] + points[d][0]) / 4.0,
        (points[a][1] + points[b][1] + points[c][1] + points[d][1]) / 4.0,
        (points[a][2] + points[b][2] + points[c][2] + points[d][2]) / 4.0,
    ];
    let mut faces: Vec<Face> = Vec::new();
    for v in [[a, b, c], [a, b, d], [a, c, d], [b, c, d]] {
        if let Some(f) = face_of(points, v, centroid) {
            faces.push(f);
        }
    }
    if faces.len() < 4 {
        return Reduced::Degenerate(planar_hull(points, plane_unit, eps));
    }

    // Incremental insertion with conflict lists. Each round takes the point
    // farthest outside any live face, deletes the faces that can see it,
    // stitches the horizon, and redistributes only the points that belonged to
    // the deleted faces. The loop ends when no face has an outside point, so it
    // is bounded by the number of hull vertices.
    let candidates: Vec<usize> = (0..points.len())
        .filter(|i| ![a, b, c, d].contains(i))
        .collect();
    assign(points, &mut faces, &candidates, eps);

    while let Some(fi) = faces
        .iter()
        .enumerate()
        .filter(|(_, f)| f.live && f.furthest.0 != usize::MAX)
        .max_by(|(_, x), (_, y)| {
            x.furthest
                .1
                .partial_cmp(&y.furthest.1)
                .unwrap_or(core::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
    {
        let apex = faces[fi].furthest.0;
        let p = points[apex];

        // Every face the apex can see dies; the horizon is the set of edges
        // shared by exactly one dying face.
        let mut horizon: Vec<(usize, usize)> = Vec::new();
        let mut orphaned: Vec<usize> = Vec::new();
        for f in &mut faces {
            if !f.live {
                continue;
            }
            if dot(f.normal, p) - f.offset > eps {
                f.live = false;
                orphaned.append(&mut f.outside);
                for k in 0..3 {
                    horizon.push((f.vertex[k], f.vertex[(k + 1) % 3]));
                }
            }
        }
        // Edges appearing in both directions are interior to the visible
        // region and are not on the horizon.
        let mut boundary: Vec<(usize, usize)> = Vec::new();
        for &(s, t) in &horizon {
            if let Some(pos) = boundary.iter().position(|&(x, y)| x == t && y == s) {
                boundary.swap_remove(pos);
            } else {
                boundary.push((s, t));
            }
        }
        orphaned.retain(|i| *i != apex);
        if boundary.is_empty() {
            // Numerically the apex saw nothing coherent. Drop it rather than
            // looping forever, and put its companions back in play.
            faces.retain(|f| f.live);
            assign(points, &mut faces, &orphaned, eps);
            continue;
        }
        let first_new = faces.len();
        for (s, t) in boundary {
            if let Some(f) = face_of(points, [s, t, apex], centroid) {
                faces.push(f);
            }
        }
        // Only the new faces can own the orphans, so assign against those.
        let mut fresh: Vec<Face> = faces.split_off(first_new);
        assign(points, &mut fresh, &orphaned, eps);
        faces.append(&mut fresh);
        faces.retain(|f| f.live);
    }

    Reduced::Solid(faces)
}

/// Hull of a cloud that lies in one plane: a two-dimensional gift-wrap in the
/// plane's own basis.
fn planar_hull(points: &[[f64; 3]], normal: [f64; 3], eps: f64) -> Vec<usize> {
    // Any two directions orthogonal to the normal.
    let helper = if normal[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let u = cross(normal, helper);
    let ulen = norm(u);
    if ulen <= 0.0 {
        return (0..points.len().min(1)).collect();
    }
    let u = [u[0] / ulen, u[1] / ulen, u[2] / ulen];
    let v = cross(normal, u);
    let flat: Vec<(f64, f64)> = points.iter().map(|p| (dot(*p, u), dot(*p, v))).collect();

    // Start at the lexicographically smallest point and wrap counter-clockwise.
    let mut start = 0;
    for i in 1..flat.len() {
        if flat[i] < flat[start] {
            start = i;
        }
    }
    let mut out = Vec::new();
    let mut current = start;
    loop {
        out.push(current);
        let mut next = (current + 1) % flat.len();
        for candidate in 0..flat.len() {
            if candidate == current {
                continue;
            }
            let (cx, cy) = flat[current];
            let (nx, ny) = flat[next];
            let (tx, ty) = flat[candidate];
            let turn = (nx - cx) * (ty - cy) - (ny - cy) * (tx - cx);
            if turn < -eps {
                next = candidate;
            } else if turn.abs() <= eps {
                // Collinear: keep the farther point, so interior points of an
                // edge are dropped.
                let dn = (nx - cx) * (nx - cx) + (ny - cy) * (ny - cy);
                let dt = (tx - cx) * (tx - cx) + (ty - cy) * (ty - cy);
                if dt > dn {
                    next = candidate;
                }
            }
        }
        current = next;
        if current == start || out.len() > flat.len() {
            break;
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force check that every returned point is genuinely extreme and
    /// every dropped point is genuinely not, by testing the support function
    /// over the subset against the support function over the whole cloud in
    /// many directions.
    ///
    /// This is the property that matters: the hull is a valid reduction exactly
    /// when it answers every support query identically.
    fn support_agrees(points: &[[f64; 3]], keep: &[usize], tol: f64) {
        let dirs = {
            let mut d = Vec::new();
            let n = 12;
            for i in 0..n {
                for j in 0..n {
                    let theta = core::f64::consts::PI * (i as f64 + 0.5) / n as f64;
                    let phi = 2.0 * core::f64::consts::PI * (j as f64) / n as f64;
                    d.push([
                        theta.sin() * phi.cos(),
                        theta.sin() * phi.sin(),
                        theta.cos(),
                    ]);
                }
            }
            d
        };
        for dir in dirs {
            let full = points
                .iter()
                .map(|p| dot(*p, dir))
                .fold(f64::NEG_INFINITY, f64::max);
            let reduced = keep
                .iter()
                .map(|i| dot(points[*i], dir))
                .fold(f64::NEG_INFINITY, f64::max);
            assert!(
                (full - reduced).abs() <= tol,
                "support differs along {dir:?}: full {full}, reduced {reduced}"
            );
        }
    }

    #[test]
    fn a_cube_with_interior_points_reduces_to_its_eight_corners() {
        let mut points = Vec::new();
        for x in [-1.0, 1.0] {
            for y in [-1.0, 1.0] {
                for z in [-1.0, 1.0] {
                    points.push([x, y, z]);
                }
            }
        }
        // Interior noise that cannot be extreme.
        for i in 0..200 {
            let t = i as f64 / 200.0;
            points.push([0.9 * (t - 0.5), 0.8 * (0.5 - t), 0.7 * t]);
        }
        let keep = hull_vertices(&points);
        assert_eq!(keep.len(), 8, "a cube has eight hull vertices");
        assert!(
            keep.iter().all(|i| *i < 8),
            "the corners are the first eight"
        );
        support_agrees(&points, &keep, 1e-12);
    }

    #[test]
    fn points_on_a_sphere_are_all_kept_and_interior_ones_are_not() {
        let mut points = Vec::new();
        let n = 14;
        for i in 0..n {
            for j in 0..n {
                let theta = core::f64::consts::PI * (i as f64 + 0.5) / n as f64;
                let phi = 2.0 * core::f64::consts::PI * (j as f64) / n as f64;
                points.push([
                    theta.sin() * phi.cos(),
                    theta.sin() * phi.sin(),
                    theta.cos(),
                ]);
            }
        }
        let on_sphere = points.len();
        // A dense interior cloud, all of which must be dropped.
        for i in 0..500 {
            let t = i as f64 / 500.0;
            points.push([0.3 * t, 0.2 - 0.4 * t, 0.1 * (t - 0.5)]);
        }
        let keep = hull_vertices(&points);
        assert!(
            keep.iter().all(|i| *i < on_sphere),
            "no interior point may be kept"
        );
        // Not a correctness requirement, but the reduction must actually
        // reduce: a sphere's sampled surface is nearly all hull.
        assert!(
            keep.len() > on_sphere / 2,
            "kept {} of {on_sphere}",
            keep.len()
        );
        support_agrees(&points, &keep, 1e-12);
    }

    #[test]
    fn a_coplanar_cloud_reduces_to_its_polygon_and_drops_edge_interiors() {
        // A square in the z = 2 plane, with points along its edges and inside.
        let mut points = vec![
            [0.0, 0.0, 2.0],
            [4.0, 0.0, 2.0],
            [4.0, 4.0, 2.0],
            [0.0, 4.0, 2.0],
        ];
        for i in 1..4 {
            points.push([i as f64, 0.0, 2.0]);
            points.push([4.0, i as f64, 2.0]);
        }
        points.push([2.0, 2.0, 2.0]);
        let keep = hull_vertices(&points);
        assert_eq!(
            keep.len(),
            4,
            "a square's hull is four corners, got {keep:?}"
        );
        support_agrees(&points, &keep, 1e-12);
    }

    #[test]
    fn a_collinear_cloud_reduces_to_its_two_ends() {
        let points: Vec<[f64; 3]> = (0..30)
            .map(|i| {
                let t = i as f64;
                [t, 2.0 * t, -3.0 * t]
            })
            .collect();
        let keep = hull_vertices(&points);
        assert_eq!(keep, vec![0, 29]);
        support_agrees(&points, &keep, 1e-12);
    }

    #[test]
    fn degenerate_clouds_are_handled_by_dimension_rather_than_refused() {
        assert!(hull_vertices(&[]).is_empty());
        assert_eq!(hull_vertices(&[[1.0, 2.0, 3.0]]), vec![0]);
        // All points identical.
        let same = vec![[1.0, 1.0, 1.0]; 50];
        let keep = hull_vertices(&same);
        assert!(!keep.is_empty() && keep.len() <= same.len());
        support_agrees(&same, &keep, 1e-12);
        // A tetrahedron is already its own hull.
        let tet = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        assert_eq!(hull_vertices(&tet).len(), 4);
    }

    #[test]
    fn hull_faces_of_a_cube_triangulate_its_surface_with_outward_winding() {
        let mut points = Vec::new();
        for x in [-1.0, 1.0] {
            for y in [-1.0, 1.0] {
                for z in [-1.0, 1.0] {
                    points.push([x, y, z]);
                }
            }
        }
        points.push([0.0, 0.0, 0.0]); // interior, must appear in no face
        let faces = hull_faces(&points).expect("a cube has a solid hull");
        // Euler: V - E + F = 2 for a triangulated closed surface, so a cube's
        // 8 vertices give 12 triangles and 18 edges.
        assert_eq!(faces.len(), 12, "a triangulated cube has twelve faces");
        let mut edges = Vec::new();
        for f in &faces {
            assert!(f.iter().all(|v| *v < 8), "an interior point was used");
            for k in 0..3 {
                let (s, t) = (f[k], f[(k + 1) % 3]);
                edges.push((s.min(t), s.max(t)));
                // Outward winding: the face normal points away from the centre.
                let n = cross(
                    sub(points[f[1]], points[f[0]]),
                    sub(points[f[2]], points[f[0]]),
                );
                assert!(dot(n, points[f[0]]) > 0.0, "face {f:?} winds inward");
            }
        }
        edges.sort_unstable();
        edges.dedup();
        assert_eq!(edges.len(), 18, "a triangulated cube has eighteen edges");
        // Every directed edge must appear in exactly two faces (a closed,
        // consistently wound surface).
        let mut directed = Vec::new();
        for f in &faces {
            for k in 0..3 {
                directed.push((f[k], f[(k + 1) % 3]));
            }
        }
        for &(s, t) in &directed {
            assert!(
                directed.iter().filter(|e| **e == (t, s)).count() == 1,
                "edge ({s},{t}) is not matched by exactly one reverse edge"
            );
        }
    }

    #[test]
    fn hull_faces_refuses_clouds_without_a_solid_hull() {
        assert!(hull_faces(&[]).is_none());
        assert!(hull_faces(&[[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).is_none());
        // Coplanar: a square has a polygon, not a solid.
        let square = [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.5, 0.5, 1.0],
        ];
        assert!(hull_faces(&square).is_none());
        // A tetrahedron is the smallest solid.
        let tet = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        assert_eq!(hull_faces(&tet).map(|f| f.len()), Some(4));
    }

    #[test]
    fn the_hull_is_scale_invariant() {
        // The tolerance is relative to the cloud's extent, so a millimetre-
        // scale part and a kilometre-scale one must reduce identically.
        let base: Vec<[f64; 3]> = {
            let mut v = Vec::new();
            for x in [-1.0, 1.0] {
                for y in [-1.0, 1.0] {
                    for z in [-1.0, 1.0] {
                        v.push([x, y, z]);
                    }
                }
            }
            v.push([0.0, 0.0, 0.0]);
            v
        };
        let reference = hull_vertices(&base);
        for scale in [1e-4, 1e-2, 1.0, 1e3] {
            let scaled: Vec<[f64; 3]> = base
                .iter()
                .map(|p| [p[0] * scale, p[1] * scale, p[2] * scale])
                .collect();
            assert_eq!(
                hull_vertices(&scaled),
                reference,
                "scale {scale} changed the hull"
            );
        }
    }
    #[test]
    fn a_large_pseudo_mesh_reduces_quickly_and_exactly() {
        // The shape of the real problem: the G1's collision meshes carry 4266
        // to 154230 vertices each, median 19116. This builds a cloud of that
        // size on a smooth surface with dense interior filling, which is the
        // hard case for the conflict lists -- most points are interior and must
        // be discarded without ever being rescanned.
        let mut points = Vec::with_capacity(20_000);
        // Surface of an ellipsoid, finely sampled.
        let n = 70;
        for i in 0..n {
            for j in 0..n {
                let theta = core::f64::consts::PI * (i as f64 + 0.5) / n as f64;
                let phi = 2.0 * core::f64::consts::PI * (j as f64) / n as f64;
                points.push([
                    0.30 * theta.sin() * phi.cos(),
                    0.12 * theta.sin() * phi.sin(),
                    0.45 * theta.cos(),
                ]);
            }
        }
        let on_surface = points.len();
        // Interior fill, none of which may survive.
        let mut seed = 12_345u64;
        let mut next = || {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((seed >> 33) as f64) / ((1u64 << 31) as f64) - 0.5
        };
        while points.len() < 20_000 {
            points.push([0.20 * next(), 0.08 * next(), 0.30 * next()]);
        }
        let keep = hull_vertices(&points);
        assert!(
            keep.iter().all(|i| *i < on_surface),
            "an interior point was kept"
        );
        // The reduction is the point of the exercise.
        assert!(
            keep.len() < points.len() / 3,
            "reduced {} of {} -- not enough of a reduction to be worth it",
            keep.len(),
            points.len()
        );
        support_agrees(&points, &keep, 1e-9);
    }
}

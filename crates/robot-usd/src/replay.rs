//! Writes a recorded robot run as a USDA layer with time samples (ADR 0015).
//!
//! The layer is an ordinary OpenUSD text file. Each link is a flat `Xform`
//! under `/replay/links` whose world pose is time-sampled per frame; its
//! collision and primitive shapes are children with static local poses, so a
//! viewer moves them with the link. Each joint is an `Xform` under
//! `/replay/joints` carrying the UsdPhysics state attributes
//! (`state:angular:physics:position` in degrees, `state:angular:physics:velocity`
//! in degrees per second) and the custom `robot:appliedTorque` in N·m. Time codes
//! are frame indices, and `timeCodesPerSecond` is the frame rate.
//!
//! Quaternions are written in USD text order, `(w, x, y, z)`. The reader in this
//! crate stores them as `(x, y, z, w)`; the two conventions are not interchanged.

use core::fmt::Write as _;

/// The geometry of one shape, in the frame of its link.
#[derive(Clone, Debug, PartialEq)]
pub enum ReplayShapeKind {
    /// A sphere of this radius (metres).
    Sphere {
        /// Radius in metres.
        radius: f64,
    },
    /// A box with these full edge lengths (metres).
    Cube {
        /// Full edge lengths along the shape's axes.
        extents: [f64; 3],
    },
    /// A cylinder along Z with this radius and height (metres).
    Cylinder {
        /// Radius in metres.
        radius: f64,
        /// Full height in metres.
        height: f64,
    },
    /// A reference to a mesh prim of the source asset; no geometry is written.
    MeshReference {
        /// The source prim path.
        path: String,
    },
    /// A mesh written with its geometry, in the shape's frame.
    Mesh {
        /// Vertex positions, metres.
        points: Vec<[f64; 3]>,
        /// Vertices per face.
        face_vertex_counts: Vec<i32>,
        /// Vertex indices of the faces, flattened.
        face_vertex_indices: Vec<i32>,
    },
}

/// A shape attached to a link.
#[derive(Clone, Debug, PartialEq)]
pub struct ReplayShape {
    /// The prim name.
    pub name: String,
    /// What it is.
    pub kind: ReplayShapeKind,
    /// Translation in the link frame (metres).
    pub translation: [f64; 3],
    /// Orientation in the link frame, `(x, y, z, w)`.
    pub orientation: [f64; 4],
    /// `true` for a physics-only shape (`purpose = "guide"` in the source).
    pub physics_only: bool,
}

/// One link's world pose at every frame.
#[derive(Clone, Debug, PartialEq)]
pub struct ReplayLink {
    /// The prim name.
    pub name: String,
    /// World translation per frame (metres).
    pub translation: Vec<[f64; 3]>,
    /// World orientation per frame, `(x, y, z, w)`.
    pub orientation: Vec<[f64; 4]>,
    /// Shapes attached to the link.
    pub shapes: Vec<ReplayShape>,
}

/// One joint's state and torque at every frame.
#[derive(Clone, Debug, PartialEq)]
pub struct ReplayJoint {
    /// The prim name.
    pub name: String,
    /// Joint position per frame, radians.
    pub position: Vec<f64>,
    /// Joint velocity per frame, radians per second.
    pub velocity: Vec<f64>,
    /// Applied torque per frame, N·m.
    pub applied_torque: Vec<f64>,
    /// Position within the joint's range per frame, `0` at the lower limit and
    /// `1` at the upper; `None` for a joint without limits.
    pub limit_fraction: Option<Vec<f64>>,
    /// Hard-limit constraint torque per frame, N·m; zero where no limit acted.
    pub limit_torque: Vec<f64>,
}

/// A joint is in proximity to a limit when its fraction of range is at or below
/// this value, or at or above [`PROXIMITY_UPPER`] (SPECS 9.1: the outer 5 % of
/// each end of the range).
pub const PROXIMITY_LOWER: f64 = 0.05;
/// The upper proximity bound; see [`PROXIMITY_LOWER`].
pub const PROXIMITY_UPPER: f64 = 0.95;

/// A recorded run.
#[derive(Clone, Debug, PartialEq)]
pub struct Replay {
    /// Frames per second of simulated time.
    pub frames_per_second: f64,
    /// Links, in a fixed order.
    pub links: Vec<ReplayLink>,
    /// Joints, in a fixed order.
    pub joints: Vec<ReplayJoint>,
    /// The ground collider of the scene the run used, if any.
    pub ground: Option<ReplayGround>,
}

/// The ground collider of a run, written as a `Plane` prim.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReplayGround {
    /// Height along Z, metres.
    pub height: f64,
    /// Penalty normal stiffness, N per metre.
    pub stiffness: f64,
    /// Penalty normal damping, N per metre per second.
    pub normal_damping: f64,
    /// Coulomb friction coefficient.
    pub friction: f64,
}

/// Why a replay could not be written.
#[derive(Debug, Clone, PartialEq)]
pub enum ReplayError {
    /// A value was NaN or infinite, which USD cannot store.
    NotFinite {
        /// What the value belongs to.
        what: String,
    },
    /// Two tracks have different numbers of frames.
    FrameCount {
        /// The track name.
        track: String,
        /// Frames the replay has.
        expected: usize,
        /// Frames the track has.
        found: usize,
    },
}

impl core::fmt::Display for ReplayError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotFinite { what } => {
                write!(f, "{what} is not finite and cannot be stored in USD")
            }
            Self::FrameCount {
                track,
                expected,
                found,
            } => write!(
                f,
                "track {track} has {found} frames, the replay has {expected}"
            ),
        }
    }
}

impl core::error::Error for ReplayError {}

/// A finite number as USD text. Values below 1e-6 in magnitude use exponent
/// notation, which USD accepts, so that near-zero samples do not bloat the file.
fn number(value: f64) -> String {
    if value != 0.0 && value.abs() < 1e-6 {
        format!("{value:e}")
    } else {
        format!("{value}")
    }
}

fn finite(value: f64, what: impl FnOnce() -> String) -> Result<f64, ReplayError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ReplayError::NotFinite { what: what() })
    }
}

fn vec3(v: [f64; 3], what: impl Fn() -> String) -> Result<String, ReplayError> {
    Ok(format!(
        "({}, {}, {})",
        number(finite(v[0], &what)?),
        number(finite(v[1], &what)?),
        number(finite(v[2], &what)?)
    ))
}

/// A quaternion in USD text order from storage order.
fn quat_text(q: [f64; 4], what: impl Fn() -> String) -> Result<String, ReplayError> {
    Ok(format!(
        "({}, {}, {}, {})",
        number(finite(q[3], &what)?),
        number(finite(q[0], &what)?),
        number(finite(q[1], &what)?),
        number(finite(q[2], &what)?)
    ))
}

/// The `timeSamples` block of one attribute.
fn samples(values: impl Iterator<Item = String>) -> String {
    let body: Vec<String> = values
        .enumerate()
        .map(|(frame, v)| format!("{frame}: {v}"))
        .collect();
    format!("{{ {} }}", body.join(", "))
}

fn check_frames(track: &str, expected: usize, found: usize) -> Result<(), ReplayError> {
    if expected == found {
        Ok(())
    } else {
        Err(ReplayError::FrameCount {
            track: track.to_owned(),
            expected,
            found,
        })
    }
}

fn write_shape(out: &mut String, indent: &str, shape: &ReplayShape) -> Result<(), ReplayError> {
    let what = || format!("shape {}", shape.name);
    let mut attrs = String::new();
    match &shape.kind {
        ReplayShapeKind::Sphere { radius } => {
            let _ = writeln!(
                attrs,
                "{indent}    double radius = {}",
                finite(*radius, what)?
            );
        }
        ReplayShapeKind::Cube { extents } => {
            let _ = writeln!(attrs, "{indent}    double size = 1");
            let _ = writeln!(
                attrs,
                "{indent}    double3 xformOp:scale = {}",
                vec3(*extents, what)?
            );
        }
        ReplayShapeKind::Cylinder { radius, height } => {
            let _ = writeln!(
                attrs,
                "{indent}    double radius = {}",
                finite(*radius, what)?
            );
            let _ = writeln!(
                attrs,
                "{indent}    double height = {}",
                finite(*height, what)?
            );
            let _ = writeln!(attrs, "{indent}    uniform token axis = \"Z\"");
        }
        ReplayShapeKind::MeshReference { .. } => {}
        ReplayShapeKind::Mesh {
            points,
            face_vertex_counts,
            face_vertex_indices,
        } => {
            let pts: Vec<String> = points
                .iter()
                .map(|p| vec3(*p, what))
                .collect::<Result<_, _>>()?;
            let _ = writeln!(attrs, "{indent}    point3f[] points = [{}]", pts.join(", "));
            let counts: Vec<String> = face_vertex_counts.iter().map(i32::to_string).collect();
            let _ = writeln!(
                attrs,
                "{indent}    int[] faceVertexCounts = [{}]",
                counts.join(", ")
            );
            let indices: Vec<String> = face_vertex_indices.iter().map(i32::to_string).collect();
            let _ = writeln!(
                attrs,
                "{indent}    int[] faceVertexIndices = [{}]",
                indices.join(", ")
            );
            let _ = writeln!(
                attrs,
                "{indent}    uniform token subdivisionScheme = \"none\""
            );
        }
    }
    let kind = match shape.kind {
        ReplayShapeKind::Sphere { .. } => "Sphere",
        ReplayShapeKind::Cube { .. } => "Cube",
        ReplayShapeKind::Cylinder { .. } => "Cylinder",
        ReplayShapeKind::MeshReference { .. } => "Xform",
        ReplayShapeKind::Mesh { .. } => "Mesh",
    };
    let order = match shape.kind {
        ReplayShapeKind::Cube { .. } => {
            "[\"xformOp:translate\", \"xformOp:orient\", \"xformOp:scale\"]"
        }
        _ => "[\"xformOp:translate\", \"xformOp:orient\"]",
    };
    let purpose = if shape.physics_only {
        "    uniform token purpose = \"guide\"\n"
    } else {
        ""
    };
    let reference = match &shape.kind {
        ReplayShapeKind::MeshReference { path } => {
            format!("{indent}    string robot:meshReference = \"{path}\"\n")
        }
        _ => String::new(),
    };
    let _ = writeln!(out, "{indent}def {kind} \"{}\"\n{indent}{{", shape.name);
    out.push_str(&attrs);
    out.push_str(&reference);
    let _ = writeln!(
        out,
        "{indent}    double3 xformOp:translate = {}",
        vec3(shape.translation, what)?
    );
    let _ = writeln!(
        out,
        "{indent}    quatd xformOp:orient = {}",
        quat_text(shape.orientation, what)?
    );
    let _ = writeln!(out, "{indent}    uniform token[] xformOpOrder = {order}");
    out.push_str(purpose);
    let _ = writeln!(out, "{indent}}}");
    Ok(())
}

/// Writes `replay` as a USDA layer.
///
/// # Errors
///
/// Returns [`ReplayError::NotFinite`] for any NaN or infinite value and
/// [`ReplayError::FrameCount`] for a track whose length differs from the
/// replay's. Nothing is returned partially written.
pub fn write_replay_usda(replay: &Replay) -> Result<String, ReplayError> {
    let frames = replay
        .links
        .first()
        .map(|l| l.translation.len())
        .or_else(|| replay.joints.first().map(|j| j.position.len()))
        .unwrap_or(0);
    for link in &replay.links {
        check_frames(&link.name, frames, link.translation.len())?;
        check_frames(&link.name, frames, link.orientation.len())?;
    }
    for joint in &replay.joints {
        check_frames(&joint.name, frames, joint.position.len())?;
        check_frames(&joint.name, frames, joint.velocity.len())?;
        check_frames(&joint.name, frames, joint.applied_torque.len())?;
        check_frames(&joint.name, frames, joint.limit_torque.len())?;
        if let Some(fraction) = &joint.limit_fraction {
            check_frames(&joint.name, frames, fraction.len())?;
        }
    }
    let last = frames.saturating_sub(1);
    let mut out = String::new();
    let _ = writeln!(out, "#usda 1.0");
    let _ = writeln!(out, "(");
    let _ = writeln!(out, "    defaultPrim = \"replay\"");
    let _ = writeln!(out, "    metersPerUnit = 1");
    let _ = writeln!(out, "    upAxis = \"Z\"");
    let _ = writeln!(out, "    timeCodesPerSecond = {}", replay.frames_per_second);
    let _ = writeln!(out, "    startTimeCode = 0");
    let _ = writeln!(out, "    endTimeCode = {last}");
    let _ = writeln!(
        out,
        "    doc = \"Replay of a recorded robot-sim run (ADR 0015). Time codes are frame indices.\""
    );
    let _ = writeln!(out, ")");
    let _ = writeln!(out, "def Xform \"replay\" (kind = \"assembly\")");
    let _ = writeln!(out, "{{");
    let _ = writeln!(out, "    def Scope \"links\"");
    let _ = writeln!(out, "    {{");
    for link in &replay.links {
        let what = || format!("link {}", link.name);
        let _ = writeln!(out, "        def Xform \"{}\"", link.name);
        let _ = writeln!(out, "        {{");
        let t = link
            .translation
            .iter()
            .map(|v| vec3(*v, what))
            .collect::<Result<Vec<_>, _>>()?;
        let q = link
            .orientation
            .iter()
            .map(|v| quat_text(*v, what))
            .collect::<Result<Vec<_>, _>>()?;
        let _ = writeln!(
            out,
            "            double3 xformOp:translate.timeSamples = {}",
            samples(t.into_iter())
        );
        let _ = writeln!(
            out,
            "            quatd xformOp:orient.timeSamples = {}",
            samples(q.into_iter())
        );
        let _ = writeln!(
            out,
            "            uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:orient\"]"
        );
        for shape in &link.shapes {
            write_shape(&mut out, "            ", shape)?;
        }
        let _ = writeln!(out, "        }}");
    }
    let _ = writeln!(out, "    }}");
    let _ = writeln!(out, "    def Scope \"joints\"");
    let _ = writeln!(out, "    {{");
    for joint in &replay.joints {
        let what = || format!("joint {}", joint.name);
        let mut pos = Vec::new();
        let mut vel = Vec::new();
        let mut tau = Vec::new();
        let mut limit_tau = Vec::new();
        for frame in 0..frames {
            pos.push(number(finite(joint.position[frame].to_degrees(), what)?));
            vel.push(number(finite(joint.velocity[frame].to_degrees(), what)?));
            tau.push(number(finite(joint.applied_torque[frame], what)?));
            limit_tau.push(number(finite(joint.limit_torque[frame], what)?));
        }
        let _ = writeln!(out, "        def Xform \"{}\"", joint.name);
        let _ = writeln!(out, "        {{");
        let _ = writeln!(
            out,
            "            double state:angular:physics:position.timeSamples = {}",
            samples(pos.into_iter())
        );
        let _ = writeln!(
            out,
            "            double state:angular:physics:velocity.timeSamples = {}",
            samples(vel.into_iter())
        );
        let _ = writeln!(
            out,
            "            double robot:appliedTorque.timeSamples = {}",
            samples(tau.into_iter())
        );
        let _ = writeln!(
            out,
            "            double robot:limitTorque.timeSamples = {}",
            samples(limit_tau.into_iter())
        );
        if let Some(fraction) = &joint.limit_fraction {
            let mut frac = Vec::new();
            let mut near = Vec::new();
            for &f in fraction.iter().take(frames) {
                frac.push(number(finite(f, what)?));
                let proximity = f <= PROXIMITY_LOWER || f >= PROXIMITY_UPPER;
                near.push(if proximity { "1" } else { "0" }.to_owned());
            }
            let _ = writeln!(
                out,
                "            double robot:limitFraction.timeSamples = {}",
                samples(frac.into_iter())
            );
            let _ = writeln!(
                out,
                "            double robot:limitProximity.timeSamples = {}",
                samples(near.into_iter())
            );
        }
        let _ = writeln!(out, "        }}");
    }
    let _ = writeln!(out, "    }}");
    if let Some(ground) = replay.ground {
        let what = || "ground".to_owned();
        let _ = writeln!(out, "    def Plane \"ground\" (");
        let _ = writeln!(out, "        apiSchemas = [\"PhysicsCollisionAPI\"]");
        let _ = writeln!(out, "    )");
        let _ = writeln!(out, "    {{");
        let _ = writeln!(out, "        uniform token axis = \"Z\"");
        let _ = writeln!(
            out,
            "        double3 xformOp:translate = {}",
            vec3([0.0, 0.0, finite(ground.height, what)?], what)?
        );
        let _ = writeln!(
            out,
            "        uniform token[] xformOpOrder = [\"xformOp:translate\"]"
        );
        let _ = writeln!(
            out,
            "        float physics:dynamicFriction = {}",
            number(finite(ground.friction, what)?)
        );
        let _ = writeln!(
            out,
            "        double sim:contactStiffness = {}",
            number(finite(ground.stiffness, what)?)
        );
        let _ = writeln!(
            out,
            "        double sim:contactDamping = {}",
            number(finite(ground.normal_damping, what)?)
        );
        let _ = writeln!(
            out,
            "        uniform token[] sim:tags = [\"environment\", \"collider\"]"
        );
        let _ = writeln!(out, "    }}");
    }
    let _ = writeln!(out, "}}");
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> Replay {
        Replay {
            frames_per_second: 1000.0,
            links: vec![ReplayLink {
                name: "pelvis".to_owned(),
                translation: vec![[0.0, 0.0, 0.79], [0.0, 0.0, 0.78]],
                orientation: vec![[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
                shapes: vec![ReplayShape {
                    name: "contact_0".to_owned(),
                    kind: ReplayShapeKind::Sphere { radius: 0.005 },
                    translation: [0.1, 0.0, -0.03],
                    orientation: [0.0, 0.0, 0.0, 1.0],
                    physics_only: true,
                }],
            }],
            ground: None,
            joints: vec![ReplayJoint {
                name: "left_hip_pitch_joint".to_owned(),
                position: vec![0.0, core::f64::consts::FRAC_PI_2],
                velocity: vec![0.0, 1.0],
                applied_torque: vec![0.0, -2.5],
                limit_fraction: Some(vec![0.02, 0.5]),
                limit_torque: vec![0.0, 0.0],
            }],
        }
    }

    #[test]
    fn writes_time_samples_with_frame_codes_and_usd_quaternion_order() {
        let text = write_replay_usda(&tiny()).unwrap();
        assert!(text.starts_with("#usda 1.0"));
        assert!(text.contains("timeCodesPerSecond = 1000"));
        assert!(text.contains("endTimeCode = 1"));
        assert!(text.contains(
            "double3 xformOp:translate.timeSamples = { 0: (0, 0, 0.79), 1: (0, 0, 0.78) }"
        ));
        // Storage order (x, y, z, w) = (0, 0, 0, 1) is written as (w, x, y, z) = (1, 0, 0, 0).
        assert!(text
            .contains("quatd xformOp:orient.timeSamples = { 0: (1, 0, 0, 0), 1: (1, 0, 0, 0) }"));
    }

    #[test]
    fn joint_states_are_written_in_the_usd_physics_units() {
        let text = write_replay_usda(&tiny()).unwrap();
        // Position is radians in the replay and degrees in the layer.
        assert!(
            text.contains("state:angular:physics:position.timeSamples = { 0: 0, 1: 90 }"),
            "{text}"
        );
        assert!(text.contains("robot:appliedTorque.timeSamples = { 0: 0, 1: -2.5 }"));
    }

    #[test]
    fn physics_only_shapes_are_marked_as_guides() {
        let text = write_replay_usda(&tiny()).unwrap();
        assert!(text.contains("def Sphere \"contact_0\""));
        assert!(text.contains("uniform token purpose = \"guide\""));
    }

    #[test]
    fn a_non_finite_value_is_refused_before_any_text_is_returned() {
        let mut bad = tiny();
        bad.joints[0].applied_torque[1] = f64::NAN;
        assert!(matches!(
            write_replay_usda(&bad),
            Err(ReplayError::NotFinite { .. })
        ));
    }

    #[test]
    fn a_track_with_the_wrong_frame_count_is_refused() {
        let mut bad = tiny();
        bad.joints[0].velocity.pop();
        assert!(matches!(
            write_replay_usda(&bad),
            Err(ReplayError::FrameCount { .. })
        ));
    }
}

//! Integration tests against the four vendored Unitree G1 29-DOF USD layers.
//!
//! Every pinned number below was measured with this crate on the vendored
//! files (OpenUSD commit 84d8ed1ae0337ce30dca8d91a3fbe7f04eebc358 layouts)
//! on 2026-10-30. Recount them with `cargo run -p robot-usd --example usd_dump`
//! if the assets change.

use robot_usd::crate_file::CrateFile;
use robot_usd::{usda, Prim, Stage, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const ASSETS: &str = "../../assets/robots/unitree_g1_usd";

fn asset(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(ASSETS).join(rel)
}

fn root_layer() -> PathBuf {
    asset("g1_29dof_rev_1_0.usd")
}

/// The composed stage, built once per test binary.
fn stage() -> &'static Stage {
    static STAGE: OnceLock<Stage> = OnceLock::new();
    STAGE.get_or_init(|| Stage::compose(&root_layer()).expect("the G1 root layer composes"))
}

fn prim<'a>(stage: &'a Stage, path: &str) -> &'a Prim {
    stage
        .prim(path)
        .unwrap_or_else(|| panic!("no prim at {path}"))
}

fn float_of(v: &Option<Value>) -> f64 {
    match v {
        Some(Value::Float(x)) => *x as f64,
        Some(Value::Double(x)) => *x,
        other => panic!("expected a float, found {other:?}"),
    }
}

#[test]
fn vendored_files_have_provenance_sizes_and_usdc_header() {
    // Sizes from assets/robots/unitree_g1_usd/PROVENANCE.md.
    let expected = [
        ("g1_29dof_rev_1_0.usd", 1603u64),
        ("configuration/g1_29dof_rev_1_0_base.usd", 28_337_485),
        ("configuration/g1_29dof_rev_1_0_physics.usd", 10_366),
        ("configuration/g1_29dof_rev_1_0_sensor.usd", 656),
    ];
    for (rel, bytes) in expected {
        let data = std::fs::read(asset(rel)).unwrap();
        assert_eq!(data.len() as u64, bytes, "{rel}");
        assert_eq!(&data[..8], b"PXR-USDC", "{rel}");
        assert_eq!(&data[8..11], &[0, 8, 0], "{rel} version");
    }
}

#[test]
fn every_vendored_layer_decodes_completely() {
    // Measured spec counts per file: every field of every spec decodes.
    let expected = [
        ("g1_29dof_rev_1_0.usd", 13usize),
        ("configuration/g1_29dof_rev_1_0_base.usd", 1181),
        ("configuration/g1_29dof_rev_1_0_physics.usd", 850),
        ("configuration/g1_29dof_rev_1_0_sensor.usd", 2),
    ];
    for (rel, specs) in expected {
        let data = std::fs::read(asset(rel)).unwrap();
        let file = CrateFile::parse(data).unwrap_or_else(|e| panic!("{rel}: {e}"));
        let decoded = file.specs().unwrap_or_else(|e| panic!("{rel}: {e}"));
        assert_eq!(decoded.len(), specs, "{rel}");
    }
}

#[test]
fn composed_stage_counts_by_type() {
    // Measured 2026-10-30: 275 prims in total.
    let counts = stage().count_by_type();
    let want: BTreeMap<String, usize> = [
        ("Cube", 2),
        ("Cylinder", 6),
        ("Material", 4),
        ("Mesh", 50),
        ("PhysicsRevoluteJoint", 29),
        ("Scope", 2),
        ("Shader", 4),
        ("Sphere", 10),
        ("Xform", 168),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v))
    .collect();
    assert_eq!(counts, want);
    assert_eq!(counts.values().sum::<usize>(), 275);
}

#[test]
fn composition_pulls_in_payloads_variants_and_internal_references() {
    let s = stage();
    // The base layer defines the meshes; they are reached through internal references.
    assert!(s.layers[0].ends_with("g1_29dof_rev_1_0.usd"));
    let visual = prim(s, "/g1_29dof_rev_1_0/pelvis/visuals/pelvis/mesh");
    assert_eq!(visual.type_name.as_deref(), Some("Mesh"));
    // The physics payload (selected by the variant) supplies joints and masses.
    assert!(prim(s, "/g1_29dof_rev_1_0/joints/left_hip_pitch_joint")
        .property("physics:body0")
        .is_some());
}

#[test]
fn joint_example_left_hip_pitch() {
    let s = stage();
    let j = prim(s, "/g1_29dof_rev_1_0/joints/left_hip_pitch_joint");
    assert_eq!(j.type_name.as_deref(), Some("PhysicsRevoluteJoint"));
    assert_eq!(
        j.property("physics:axis").unwrap().default,
        Some(Value::Token("Y".into()))
    );
    // Limits in degrees, as authored: measured -144.99841 and 165.00038.
    assert!(
        (float_of(&j.property("physics:lowerLimit").unwrap().default) + 144.99841).abs() < 1e-4
    );
    assert!(
        (float_of(&j.property("physics:upperLimit").unwrap().default) - 165.00038).abs() < 1e-4
    );
    assert_eq!(
        j.property("physics:body0").unwrap().targets,
        vec!["/g1_29dof_rev_1_0/pelvis".to_string()]
    );
    assert_eq!(
        j.property("physics:body1").unwrap().targets,
        vec!["/g1_29dof_rev_1_0/left_hip_pitch_link".to_string()]
    );
}

#[test]
fn link_mass_examples() {
    let s = stage();
    // Measured: pelvis 3.814 kg, left_hip_pitch_link 1.35 kg.
    let pelvis = prim(s, "/g1_29dof_rev_1_0/pelvis");
    assert!((float_of(&pelvis.property("physics:mass").unwrap().default) - 3.814).abs() < 1e-6);
    let hip = prim(s, "/g1_29dof_rev_1_0/left_hip_pitch_link");
    assert!((float_of(&hip.property("physics:mass").unwrap().default) - 1.35).abs() < 1e-6);
    // Every physical link carries a positive mass.
    let mut masses = 0;
    s.for_each_prim(|p| {
        if let Some(prop) = p.property("physics:mass") {
            masses += 1;
            assert!(float_of(&prop.default) > 0.0, "{}", p.path);
        }
    });
    assert_eq!(masses, 30);
}

#[test]
fn collision_sphere_example() {
    let s = stage();
    let sphere = prim(
        s,
        "/g1_29dof_rev_1_0/left_ankle_roll_link/collisions/mesh_1/sphere",
    );
    assert_eq!(sphere.type_name.as_deref(), Some("Sphere"));
    // Measured radius 0.004999999888241291 (float32 0.005).
    assert!((float_of(&sphere.property("radius").unwrap().default) - 0.005).abs() < 1e-8);
}

#[test]
fn mesh_points_agree_with_their_extents_and_topology() {
    // 50 meshes: every extent equals the bounding box of its points, and face
    // counts and indices are consistent. This checks the compressed array decoders.
    let s = stage();
    let mut meshes = 0;
    s.for_each_prim(|p| {
        if p.type_name.as_deref() != Some("Mesh") {
            return;
        }
        meshes += 1;
        let default = |n: &str| p.property(n).and_then(|x| x.default.clone());
        let (Some(Value::Array(points)), Some(Value::Array(extent))) =
            (default("points"), default("extent"))
        else {
            panic!("{} has no points or extent", p.path);
        };
        let mut lo = [f32::MAX; 3];
        let mut hi = [f32::MIN; 3];
        for v in &points {
            let Value::Vec3f(a) = v else {
                panic!("point is not a float3")
            };
            for k in 0..3 {
                lo[k] = lo[k].min(a[k]);
                hi[k] = hi[k].max(a[k]);
            }
        }
        let (Value::Vec3f(e0), Value::Vec3f(e1)) = (&extent[0], &extent[1]) else {
            panic!("extent")
        };
        for k in 0..3 {
            assert!(
                (lo[k] - e0[k]).abs() < 1e-4 && (hi[k] - e1[k]).abs() < 1e-4,
                "{} axis {k}",
                p.path
            );
        }
        let (Some(Value::Array(counts)), Some(Value::Array(indices))) =
            (default("faceVertexCounts"), default("faceVertexIndices"))
        else {
            panic!("{} has no topology", p.path);
        };
        let total: i64 = counts
            .iter()
            .map(|c| {
                if let Value::Int(x) = c {
                    *x as i64
                } else {
                    panic!("count")
                }
            })
            .sum();
        assert_eq!(total as usize, indices.len(), "{}", p.path);
        for i in &indices {
            let Value::Int(x) = i else { panic!("index") };
            assert!(*x >= 0 && (*x as usize) < points.len(), "{}", p.path);
        }
    });
    assert_eq!(meshes, 50);
}

#[test]
fn usda_round_trip_of_composed_subtrees() {
    // The whole composed stage round-trips too (checked by hand: about 170 MB of
    // text, too slow for the default debug test run). Here the root prim keeps
    // its joints and the pelvis collisions, which include mesh normals.
    let s = stage();
    let mut root = prim(s, "/g1_29dof_rev_1_0").clone();
    let mut pelvis = prim(s, "/g1_29dof_rev_1_0/pelvis").clone();
    pelvis.children.retain(|c| c.name == "collisions");
    root.children = vec![prim(s, "/g1_29dof_rev_1_0/joints").clone(), pelvis];
    let subset = Stage {
        layers: s.layers.clone(),
        metadata: s.metadata.clone(),
        prims: vec![root],
    };
    let text = usda::write_usda(&subset);
    let back = usda::parse_stage(&text).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(back.prims, subset.prims);
    assert_eq!(back.metadata, subset.metadata);
}

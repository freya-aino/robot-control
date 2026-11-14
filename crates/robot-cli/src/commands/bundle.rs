//! `robot skills` and `robot eval --bundle`: the two commands the
//! `skill-manifest/1` and `robot-bundle/1` formats (ADR 0016) unblocked.
//!
//! `robot skills list|search` reads a skill library's manifest and answers
//! from the typed descriptors — the same structures the in-loop router
//! consults, so what the operator sees is what the selector selects over.
//!
//! `robot eval --bundle` is the load-time half of the "wrong artifact
//! deployed" hazard control (SPECS.md Section 12.2): the bundle's digest and
//! every weight blob verify, and `check_activatable` runs against a named
//! vendored robot's real joint limits in `SimEval` mode. Exit 0 means
//! "verified and activatable against that envelope", not "good": quality is
//! `eval run`'s job, against an acceptance profile.

use super::{EXIT_FAILURE, EXIT_USAGE};
use robot_artifacts::bundle::read_bundle;
use robot_model::fixtures;
use robot_model::joint::JointKind;
use robot_model::profile::RobotProfile;
use robot_skills::catalog::Catalog;
use robot_skills::manifest::read_manifest;
use robot_types::bundle::HostEnvelope;
use robot_types::mode::OperatingMode;
use std::path::Path;

/// `robot skills <list|search> --manifest <toml> [query...]`.
pub fn skills(args: &[String]) -> i32 {
    let usage = "robot skills <list|search> --manifest <skill-manifest.toml> [query...]";
    let Some(manifest_at) = args.iter().position(|a| a == "--manifest") else {
        eprintln!("usage: {usage}");
        return EXIT_USAGE;
    };
    let Some(path) = args.get(manifest_at + 1) else {
        eprintln!("error: --manifest needs a path");
        return EXIT_USAGE;
    };
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: {path}: {e}");
            return EXIT_FAILURE;
        }
    };
    let skills = match read_manifest(&text) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {path}: {e}");
            return EXIT_FAILURE;
        }
    };
    match args.first().map(String::as_str) {
        Some("list") => {
            println!("{} skills in {path}", skills.len());
            for d in &skills {
                let tags: Vec<&str> = d.tags.iter().map(|t| t.as_str()).collect();
                println!(
                    "  [{}] {} tags={} duration={}ms memory={}B transitions={}",
                    d.id.0,
                    d.name.as_str(),
                    tags.join(","),
                    d.expected_duration.nanos() / 1_000_000,
                    d.inference_memory_bytes,
                    d.outgoing_transitions.len(),
                );
            }
            0
        }
        Some("search") => {
            let query = args[..manifest_at]
                .iter()
                .skip(1)
                .cloned()
                .collect::<Vec<_>>()
                .join(" ");
            if query.is_empty() {
                eprintln!("usage: robot skills search <query...> --manifest <toml>");
                return EXIT_USAGE;
            }
            // The catalog is the Section 10.1 prose index: name and tags,
            // never a feasibility check.
            let entries: Vec<(robot_skills::descriptor::SkillId, String)> = skills
                .iter()
                .map(|d| {
                    let tags: Vec<&str> = d.tags.iter().map(|t| t.as_str()).collect();
                    (d.id, format!("{} {}", d.name.as_str(), tags.join(" ")))
                })
                .collect();
            let refs: Vec<(robot_skills::descriptor::SkillId, &str)> =
                entries.iter().map(|(id, s)| (*id, s.as_str())).collect();
            let catalog = Catalog::build(&refs);
            let hits = catalog.search(&query);
            if hits.is_empty() {
                println!("no skill matches {query:?}");
                return EXIT_FAILURE;
            }
            for (id, score) in hits {
                let name = skills
                    .iter()
                    .find(|d| d.id == id)
                    .map_or("?", |d| d.name.as_str());
                println!("  [{:>3}] {name} score={score:.3}", id.0);
            }
            0
        }
        _ => {
            eprintln!("usage: {usage}");
            EXIT_USAGE
        }
    }
}

fn host_envelope(profile: &RobotProfile) -> HostEnvelope {
    let actuated = profile.joints.iter().filter(|j| {
        matches!(
            j.kind,
            JointKind::Revolute | JointKind::Continuous | JointKind::Prismatic
        )
    });
    let mut position = Vec::new();
    let mut velocity = Vec::new();
    let mut torque = Vec::new();
    for j in actuated {
        let (lo, hi) = j
            .limits
            .position
            .map_or((f64::NEG_INFINITY, f64::INFINITY), |r| {
                (r.lo.value(), r.hi.value())
            });
        position.push((lo, hi));
        velocity.push(j.limits.velocity.value());
        torque.push(j.limits.effort.value());
    }
    HostEnvelope {
        robot_profile: format!("{}/{}", profile.name.as_str(), profile.profile_version),
        position_limits: position,
        velocity_limits: velocity,
        torque_limits: torque,
    }
}

/// `robot eval --bundle <dir> --robot <g1-usd|go2-usd|h1-usd>`.
pub fn eval_bundle(args: &[String]) -> i32 {
    let usage = "robot eval --bundle <dir> --robot <g1-usd|go2-usd|h1-usd>";
    let Some(dir) = args.first() else {
        eprintln!("usage: {usage}");
        return EXIT_USAGE;
    };
    let robot = match args.iter().position(|a| a == "--robot") {
        Some(i) => match args.get(i + 1) {
            Some(r) => r.clone(),
            None => {
                eprintln!("error: --robot needs a value");
                return EXIT_USAGE;
            }
        },
        None => {
            eprintln!("usage: {usage}");
            return EXIT_USAGE;
        }
    };
    let profile = match robot.as_str() {
        "g1-usd" => fixtures::unitree_g1_29dof_usd(),
        "go2-usd" => fixtures::unitree_go2_12dof_usd(),
        "h1-usd" => fixtures::unitree_h1_19dof_usd(),
        other => {
            eprintln!("error: unknown robot {other:?} (g1-usd, go2-usd or h1-usd)");
            return EXIT_USAGE;
        }
    };
    let profile = match profile {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: vendored profile: {e}");
            return EXIT_FAILURE;
        }
    };

    // Integrity first: digest-sealed manifest, size- and content-checked
    // blobs, before any field of it is trusted.
    let manifest = match read_bundle(Path::new(dir)) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("bundle {dir}: REFUSED: {e}");
            return EXIT_FAILURE;
        }
    };
    println!("bundle {} at {dir}", manifest.bundle_id);
    println!("  digest   {}", manifest.digest);
    println!(
        "  policy   obs width {} -> action width {} at {} Hz, recurrent width {}",
        manifest.observation.width(),
        manifest.action.width,
        manifest.rate.0,
        manifest.recurrent.width(),
    );
    println!(
        "  weights  {} component(s), {} bytes total, all content-verified",
        manifest.components.len(),
        manifest.total_weight_bytes(),
    );
    println!("  trained for {}", manifest.envelope.robot_profile);

    let host = host_envelope(&profile);
    println!(
        "  host     {} ({} actuated joints)",
        host.robot_profile,
        host.velocity_limits.len()
    );
    match manifest.check_activatable(&host, OperatingMode::SimEval, manifest.rate) {
        Ok(()) => {
            println!("ACTIVATABLE in sim-eval against {}", host.robot_profile);
            0
        }
        Err(e) => {
            println!("REFUSED for sim-eval against {}: {e:?}", host.robot_profile);
            EXIT_FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_artifacts::bundle::write_bundle;
    use robot_artifacts::hash::ContentId;
    use robot_skills::descriptor::{EmbeddingVersion, SkillDescriptor, SkillId};
    use robot_skills::manifest::write_manifest;
    use robot_types::bundle::{
        ComponentManifest, DataLineage, PolicyBundleManifest, RecurrentStateSpec,
        SafetyEnvelopeAssumptions, BUNDLE_SCHEMA_VERSION,
    };
    use robot_types::ids::Name;
    use robot_types::obs::{ActionSemantics, ActionSpec, FieldSpec, ObservationSpec};
    use robot_types::schema::SchemaVersion;
    use robot_types::tier::{
        MeasuredCompute, MissingUpperTierBehavior, Tier, TierAssignment, TierSchedule,
    };
    use robot_types::time::Duration;
    use robot_types::units::Hertz;

    fn tmp(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("robot-cli-bundle-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn skills_list_and_search_read_a_manifest() {
        let dir = tmp("skills");
        let d = SkillDescriptor {
            id: SkillId(1),
            name: Name::new("jab").unwrap(),
            tags: vec![Name::new("strike").unwrap()],
            embedding: vec![0.0],
            embedding_version: EmbeddingVersion(1),
            embodiment_compatibility: vec!["g1/1.0".to_owned()],
            initiation_guards: vec![Name::new("double-support").unwrap()],
            termination_condition: Name::new("strike-complete").unwrap(),
            expected_duration: Duration::from_millis(400),
            assumes_ground_contact: true,
            supported_disturbances: vec![],
            inference_memory_bytes: 1024,
            outgoing_transitions: vec![],
        };
        let path = dir.join("skills.toml");
        std::fs::write(&path, write_manifest(&[d])).unwrap();
        let p = path.display().to_string();
        let a = |v: &[&str]| v.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
        assert_eq!(skills(&a(&["list", "--manifest", &p])), 0);
        assert_eq!(skills(&a(&["search", "jab", "--manifest", &p])), 0);
        assert_eq!(
            skills(&a(&["search", "zzz", "--manifest", &p])),
            EXIT_FAILURE
        );
        assert_eq!(skills(&a(&["list"])), EXIT_USAGE);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn eval_bundle_activates_a_matching_bundle_and_refuses_a_mismatch() {
        let profile = fixtures::unitree_go2_12dof_usd().unwrap();
        let host = host_envelope(&profile);
        let n = host.velocity_limits.len();
        let blob = vec![7u8; 64];
        let mut tiers = TierSchedule::new();
        tiers.declare(TierAssignment {
            component: "actor".to_owned(),
            tier: Tier::Policy,
            rate: Hertz(100.0),
            budget: Duration::from_millis(8),
            measured_compute: MeasuredCompute {
                percentile: 99.9,
                duration: Duration::from_micros(500),
                max: Duration::from_micros(900),
                run_id: "bench-1".to_owned(),
            },
            input_freshness_limit: Duration::from_millis(100),
            on_missing_upper: MissingUpperTierBehavior::UseDeclaredDefault,
        });
        let manifest = PolicyBundleManifest {
            schema_version: BUNDLE_SCHEMA_VERSION,
            bundle_id: "go2-stand/1".to_owned(),
            observation: ObservationSpec::new(
                SchemaVersion::new(1, 0),
                vec![FieldSpec::deployable("joint_position", n, "rad").unwrap()],
            )
            .unwrap(),
            normalizer_mean: vec![0.0; n],
            normalizer_variance: vec![1.0; n],
            normalizer_count: 100,
            normalizer_clip: 5.0,
            action: ActionSpec::identity(
                SchemaVersion::new(1, 0),
                ActionSemantics::JointPositionTarget,
                n,
                1.0,
            )
            .unwrap(),
            recurrent: RecurrentStateSpec::Stateless,
            rate: Hertz(100.0),
            tiers,
            components: vec![ComponentManifest {
                name: Name::new("actor").unwrap(),
                architecture_version: SchemaVersion::new(1, 0),
                weights_digest: ContentId::of(&blob).to_string(),
                weights_bytes: blob.len() as u64,
                decodes_latent_space: None,
            }],
            envelope: SafetyEnvelopeAssumptions {
                robot_profile: host.robot_profile.clone(),
                safety_profile: "none-declared/0".to_owned(),
                position_limits: host.position_limits.clone(),
                velocity_limits: host.velocity_limits.clone(),
                torque_limits: host.torque_limits.clone(),
                qualified_modes: vec![robot_types::mode::OperatingMode::SimEval],
            },
            lineage: DataLineage {
                inputs: vec![],
                config_hash: "sha256:cfg".to_owned(),
                environment_lock: "devenv.lock@test".to_owned(),
                source_revision: "git:test".to_owned(),
                held_out: vec![],
            },
            transitions: vec![],
            benchmarks: vec![],
            digest: String::new(),
            signature: None,
        };
        let dir = tmp("eval");
        write_bundle(&dir, &manifest, &[("actor", &blob)]).unwrap();
        let d = dir.display().to_string();
        let a = |v: &[&str]| v.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
        assert_eq!(eval_bundle(&a(&[&d, "--robot", "go2-usd"])), 0);
        // The same bundle against a different robot is a profile mismatch.
        assert_eq!(eval_bundle(&a(&[&d, "--robot", "h1-usd"])), EXIT_FAILURE);
        // A tampered manifest is refused before any activation question.
        let mpath = dir.join("manifest.toml");
        let text = std::fs::read_to_string(&mpath)
            .unwrap()
            .replace("go2-stand/1", "sneaky/1");
        std::fs::write(&mpath, text).unwrap();
        assert_eq!(eval_bundle(&a(&[&d, "--robot", "go2-usd"])), EXIT_FAILURE);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

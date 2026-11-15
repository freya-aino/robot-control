//! The PLAN.md §4 commands built on the motion, evaluation, adaptation,
//! learning and viewer crates:
//!
//! - `robot data import <source> --store <dir> [--license <id>] [--training]`
//! - `robot motion retarget <bvh> --map <file> [--robot g1|g1-usd] --out <csv>`
//! - `robot motion validate <bvh> --reference <csv> --map <file> [--robot ...]`
//! - `robot eval run --profile <toml> --results <csv>`
//! - `robot adapt identify <csv> [--dead-band <rad/s>]`
//! - `robot train [--iterations N] [--seed S] --out <dir>`
//! - `robot replay <run-dir>`
//!
//! Every one reports what it did and exits 0, or names what failed and
//! exits [`EXIT_FAILURE`]. Thresholds the commands need and the user does
//! not give are printed with their values, never applied silently.

use std::fmt::Write as _;
use std::path::Path;

use motion_io::import::{import, inspect_imported, UpAxis};
use motion_process::condition::{convert_axes, AxisConversion};
use motion_process::contact::{estimate_contacts, estimate_ground_plane, ContactConfig, FootSite};
use motion_retarget::retarget::retarget;
use motion_validate::contact::ContactTolerances;
use motion_validate::package::{manifest, reference_id, CurationEntry};
use motion_validate::report::AcceptanceThresholds;
use motion_validate::validate::{validate, ValidationConfig};
use robot_artifacts::store::{BlobStore, StoreQuota};
use robot_model::profile::RobotProfile;
use robot_types::frames::{FrameId, Quat, Transform, Vec3};
use robot_types::motion::{
    Channel, HumanSkeleton, JointValue, MotionFrame, MotionRecord, RobotReference, SourceLicense,
    Validity,
};
use robot_types::time::{Duration, Timestamp};

use super::EXIT_FAILURE;

fn fail(message: impl std::fmt::Display) -> i32 {
    eprintln!("error: {message}");
    EXIT_FAILURE
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn license(args: &[String], source: &Path) -> SourceLicense {
    let id = flag(args, "--license")
        .map_or_else(|| format!("cli-import:{}", source.display()), str::to_owned);
    let mut l = SourceLicense::unaudited(id);
    if args.iter().any(|a| a == "--training") {
        l.training_permitted = true;
    }
    l
}

fn robot(args: &[String]) -> Result<RobotProfile, String> {
    match flag(args, "--robot").unwrap_or("g1") {
        "g1" => robot_model::fixtures::unitree_g1_29dof().map_err(|e| e.to_string()),
        "g1-usd" => robot_model::fixtures::unitree_g1_29dof_usd().map_err(|e| e.to_string()),
        other => Err(format!("unknown robot {other:?}; known: g1, g1-usd")),
    }
}

/// `robot data import`.
pub fn data_import(source: &Path, args: &[String]) -> i32 {
    let Some(store_dir) = flag(args, "--store") else {
        return fail("data import needs --store <dir>");
    };
    let bytes = match std::fs::read(source) {
        Ok(b) => b,
        Err(e) => return fail(format!("cannot read {}: {e}", source.display())),
    };
    let store = match BlobStore::open(store_dir, StoreQuota::default()) {
        Ok(s) => s,
        Err(e) => return fail(format!("cannot open store {store_dir}: {e}")),
    };
    match import(&store, &bytes, license(args, source)) {
        Ok((id, imported)) => {
            let i = inspect_imported(&imported);
            println!("{}: stored original as {id}", source.display());
            println!("  frames:          {}", i.frames);
            println!("  channels:        {}", i.channels);
            println!(
                "  rate:            {}",
                i.rate.map_or("undefined".into(), |r| format!("{r:.3} Hz"))
            );
            println!("  valid fraction:  {:.4}", i.valid_fraction);
            println!(
                "  licence:         {} (training {})",
                i.license,
                if i.training_permitted {
                    "permitted"
                } else {
                    "not permitted"
                }
            );
            println!("  up axis (guess): {:?}", i.up_axis_guess);
            println!(
                "  unit (guess):    {}",
                i.length_unit_guess.unwrap_or("unknown")
            );
            0
        }
        Err(e) => fail(e),
    }
}

fn load_bvh(path: &Path, args: &[String]) -> Result<MotionRecord<HumanSkeleton>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let hash = robot_artifacts::hash::ContentId::of(text.as_bytes()).to_string();
    let mut r =
        motion_io::bvh::parse(&text, &hash, license(args, path)).map_err(|e| e.to_string())?;
    let i = motion_io::import::inspect(&r);
    let scale = match flag(args, "--unit").or(i.length_unit_guess) {
        Some("m") => 1.0,
        Some("mm") => 0.001,
        Some("cm") => 0.01,
        other => {
            return Err(format!(
                "cannot tell the length unit ({other:?}); pass --unit m|cm|mm"
            ))
        }
    };
    let up = match flag(args, "--up") {
        Some("y") => UpAxis::Y,
        Some("z") => UpAxis::Z,
        Some(o) => return Err(format!("--up must be y or z, not {o:?}")),
        None => i.up_axis_guess.unwrap_or(UpAxis::Y),
    };
    let conversion = match up {
        UpAxis::Y => AxisConversion::y_up_to_z_up(),
        _ => AxisConversion {
            basis: Quat::IDENTITY,
            scale: 1.0,
            name: "identity",
        },
    }
    .with_scale(scale);
    eprintln!("note: converting {up:?}-up, {scale} m per unit, to z-up metres");
    convert_axes(&mut r, conversion).map_err(|e| e.to_string())?;
    Ok(r)
}

fn label_contacts(
    record: &mut MotionRecord<HumanSkeleton>,
    config: &motion_retarget::retarget::RetargetConfig,
) -> Result<(), String> {
    if config.map.feet.is_empty() {
        return Ok(());
    }
    let feet: Vec<usize> = config.map.feet.iter().map(|f| f.human).collect();
    let plane =
        estimate_ground_plane(record, &feet, Vec3::unit_z(), 0.15).map_err(|e| e.to_string())?;
    let contact = ContactConfig {
        height_threshold: 0.06,
        height_scale: 0.01,
        speed_threshold: 0.3,
        speed_scale: 0.05,
        enter: 0.6,
        exit: 0.4,
        min_duration: Duration::from_millis(60),
    };
    eprintln!(
        "note: contact model (unvalidated defaults): sole height < 0.06 m, foot speed < 0.3 m/s; ground plane residual {:.4} m",
        plane.residual_std
    );
    let sites: Vec<FootSite> = config
        .map
        .feet
        .iter()
        .map(|f| FootSite {
            channel: f.human,
            site: f.site,
            sole_offset: 0.0,
        })
        .collect();
    estimate_contacts(record, &plane, &sites, &contact).map_err(|e| e.to_string())?;
    Ok(())
}

fn prepare(
    source: &Path,
    args: &[String],
) -> Result<
    (
        MotionRecord<HumanSkeleton>,
        RobotProfile,
        motion_retarget::retarget::RetargetConfig,
    ),
    String,
> {
    let Some(map) = flag(args, "--map") else {
        return Err("needs --map <retarget-map/1 file>".into());
    };
    let profile = robot(args)?;
    let mut human = load_bvh(source, args)?;
    let map_text = std::fs::read_to_string(map).map_err(|e| format!("cannot read {map}: {e}"))?;
    let config =
        motion_retarget::config::parse(&map_text, &human, &profile).map_err(|e| e.to_string())?;
    label_contacts(&mut human, &config)?;
    Ok((human, profile, config))
}

/// `robot motion retarget`.
pub fn motion_retarget(source: &Path, args: &[String]) -> i32 {
    let Some(out) = flag(args, "--out") else {
        return fail("motion retarget needs --out <csv>");
    };
    let (human, profile, config) = match prepare(source, args) {
        Ok(x) => x,
        Err(e) => return fail(e),
    };
    let result = match retarget(&human, &profile, &config) {
        Ok(r) => r,
        Err(e) => return fail(e),
    };
    if let Err(e) = std::fs::write(out, write_reference_csv(&result.reference)) {
        return fail(format!("cannot write {out}: {e}"));
    }
    let worst = result
        .frames
        .iter()
        .map(|f| f.max_position_error)
        .fold(0.0, f64::max);
    let slip = result
        .frames
        .iter()
        .flat_map(|f| f.planted.iter().map(|p| p.1))
        .fold(0.0, f64::max);
    println!("{}: retargeted onto {}", source.display(), profile.name);
    println!("  frames:              {}", result.reference.frame_count());
    println!("  reference id:        {}", reference_id(&result.reference));
    println!("  worst task error:    {worst:.4} m");
    println!("  worst planted slip:  {slip:.4} m");
    println!(
        "  unfinished frames:   {}",
        result.unfinished_frames().len()
    );
    println!("  velocity excess:     {}", result.velocity_excess.len());
    println!("  written:             {out}");
    0
}

/// The `robot-reference/1` CSV: a header line naming the profile and the
/// joints, then one row per frame of `t_ns, px, py, pz, qw, qx, qy, qz,
/// joints...` at full `f64` precision.
#[must_use]
pub fn write_reference_csv(r: &MotionRecord<RobotReference>) -> String {
    let mut s = format!("# robot-reference/1 {}\n", r.skeleton_hash);
    let names: Vec<String> = r.channels.iter().map(|c| c.name.to_string()).collect();
    let _ = writeln!(s, "t_ns,px,py,pz,qw,qx,qy,qz,{}", names.join(","));
    for f in &r.frames {
        let (t, q) = (f.root.translation, f.root.rotation);
        let mut row = vec![
            f.timestamp.nanos().to_string(),
            format!("{:e}", t.x),
            format!("{:e}", t.y),
            format!("{:e}", t.z),
            format!("{:e}", q.w),
            format!("{:e}", q.x),
            format!("{:e}", q.y),
            format!("{:e}", q.z),
        ];
        for v in &f.values {
            if let JointValue::Scalar { value, .. } = v {
                row.push(format!("{value:e}"));
            }
        }
        let _ = writeln!(s, "{}", row.join(","));
    }
    s
}

/// Reads a `robot-reference/1` CSV against `profile`'s joints.
pub fn read_reference_csv(
    text: &str,
    profile: &RobotProfile,
    source: &MotionRecord<HumanSkeleton>,
) -> Result<MotionRecord<RobotReference>, String> {
    let mut lines = text.lines();
    let head = lines.next().ok_or("empty reference")?;
    let skeleton = head
        .strip_prefix("# robot-reference/1 ")
        .ok_or("not a robot-reference/1 file")?
        .to_owned();
    let cols: Vec<&str> = lines.next().ok_or("no column line")?.split(',').collect();
    let joints: Vec<_> = profile.actuated_joints().collect();
    if cols.len() != 8 + joints.len()
        || cols[8..]
            .iter()
            .zip(&joints)
            .any(|(c, j)| *c != j.name.as_str())
    {
        return Err("reference columns do not match the robot's joints".into());
    }
    let mut frames = Vec::new();
    for (k, line) in lines.enumerate() {
        let v: Vec<f64> = line
            .split(',')
            .map(|x| {
                x.trim()
                    .parse()
                    .map_err(|_| format!("row {k}: bad number {x:?}"))
            })
            .collect::<Result<_, _>>()?;
        if v.len() != cols.len() {
            return Err(format!(
                "row {k}: {} values, expected {}",
                v.len(),
                cols.len()
            ));
        }
        frames.push(MotionFrame {
            timestamp: Timestamp::sampling(v[0] as i64, source.clock),
            root: Transform::new(
                FrameId::WORLD,
                source.reference_frame,
                Quat {
                    w: v[4],
                    x: v[5],
                    y: v[6],
                    z: v[7],
                }
                .normalized(),
                Vec3::new(v[1], v[2], v[3]),
            ),
            values: v[8..]
                .iter()
                .map(|x| JointValue::Scalar {
                    value: *x,
                    unit: "rad",
                })
                .collect(),
            validity: vec![Validity::Valid; joints.len()],
            contacts: source
                .frames
                .get(k)
                .map(|f| f.contacts.clone())
                .unwrap_or_default(),
        });
    }
    let channels = joints
        .iter()
        .map(|j| Channel {
            name: j.name.clone(),
            parent: None,
            offset: Vec3::ZERO,
        })
        .collect();
    MotionRecord::new(
        RobotReference,
        source.source_hash.clone(),
        source.source_license.clone(),
        source.session_id.clone(),
        skeleton,
        source.reference_frame,
        source.clock,
        channels,
        frames,
    )
    .map_err(|e| e.to_string())
}

/// `robot motion validate <bvh> --reference <csv> --map <file>`.
pub fn motion_validate_reference(source: &Path, args: &[String]) -> i32 {
    let Some(reference_path) = flag(args, "--reference") else {
        return fail("needs --reference <csv>");
    };
    let (human, profile, config) = match prepare(source, args) {
        Ok(x) => x,
        Err(e) => return fail(e),
    };
    let text = match std::fs::read_to_string(reference_path) {
        Ok(t) => t,
        Err(e) => return fail(format!("cannot read {reference_path}: {e}")),
    };
    let reference = match read_reference_csv(&text, &profile, &human) {
        Ok(r) => r,
        Err(e) => return fail(e),
    };
    let vc = ValidationConfig {
        map: config.map.clone(),
        body_size: 1.3,
        contact: ContactTolerances {
            ground_height: 0.0,
            sole_height: 0.0,
            height_tolerance: 0.05,
            force_tolerance: 20.0,
            friction: 0.8,
            contact_confidence: config.contact_confidence,
        },
        moment_length: 0.1,
        tracking: None,
        feedforward: false,
    };
    let thresholds = AcceptanceThresholds {
        max_normalized_endpoint_error: 0.05,
        max_planted_foot_slip: 0.03,
        max_root_drift: 0.25,
        allow_limit_violations: false,
        allow_torque_violations: false,
        allow_contact_inconsistency: false,
        max_tracking_rms: 0.1,
        require_tracking: false,
        // CoP inside the support polygon whenever it is checkable; a foot
        // whose declared geometry cannot be projected reports rather than
        // rejects, since this command runs on arbitrary imported robots.
        min_cop_margin: 0.0,
        allow_unchecked_cop: true,
        // Acceleration and jerk bounds derived from the profile's own
        // declared effort limits, effective inertias and command latencies
        // (never invented numbers); a joint declaring no basis gets an
        // unbounded slot that never rejects.
        motion_limits: match motion_validate::dynamics::derived_motion_limits(&profile) {
            Ok(limits) => Some(limits),
            Err(e) => return fail(e),
        },
    };
    eprintln!("note: thresholds (unvalidated defaults): endpoint 0.05 of 1.3 m body size, slip 0.03 m, drift 0.25 m, CoP margin 0 m where checkable, accel/jerk from the profile's declared envelopes");
    let report = match validate(&human, &reference, &profile, &vc) {
        Ok(r) => r,
        Err(e) => return fail(e),
    };
    let verdict = report.evaluate(&thresholds);
    let entry = CurationEntry {
        reference: reference_id(&reference),
        source_hash: human.source_hash.clone(),
        session: human.session_id.clone(),
        verdict: verdict.clone(),
    };
    println!("{}", manifest(&[entry]));
    match verdict {
        motion_validate::report::Verdict::Accepted => 0,
        motion_validate::report::Verdict::Rejected(_) => EXIT_FAILURE,
    }
}

/// `robot eval run --profile <toml> --results <csv>`, the results CSV being
/// `seed,metric,value` rows.
pub fn eval_run(args: &[String]) -> i32 {
    let (Some(profile_path), Some(results_path)) =
        (flag(args, "--profile"), flag(args, "--results"))
    else {
        return fail("eval run needs --profile <toml> and --results <csv>");
    };
    let read = |p: &str| std::fs::read_to_string(p).map_err(|e| format!("cannot read {p}: {e}"));
    let (profile_text, results_text) = match (read(profile_path), read(results_path)) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => return fail(e),
    };
    let mut profile = match robot_eval::acceptance::AcceptanceProfile::from_toml(&profile_text) {
        Ok(p) => p,
        Err(e) => return fail(e),
    };
    let mut results: Vec<robot_eval::evaluate::SeedResult> = Vec::new();
    for (k, line) in results_text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty() && !l.starts_with('#'))
    {
        let f: Vec<&str> = line.split(',').map(str::trim).collect();
        let (Some(seed), Some(metric), Some(value)) = (
            f.first().and_then(|s| s.parse::<u64>().ok()),
            f.get(1),
            f.get(2).and_then(|v| v.parse::<f64>().ok()),
        ) else {
            return fail(format!(
                "results line {}: expected seed,metric,value",
                k + 1
            ));
        };
        match results.iter_mut().find(|r| r.seed == seed) {
            Some(r) => r.metrics.push(((*metric).to_owned(), value)),
            None => results.push(robot_eval::evaluate::SeedResult {
                seed,
                metrics: vec![((*metric).to_owned(), value)],
            }),
        }
    }
    let record = robot_eval::evaluate::evaluate(&mut profile, &results);
    println!(
        "profile {} v{} ({})",
        record.profile, record.version, record.profile_id
    );
    for o in &record.outcomes {
        let stats = o.stats.map_or("fewer than two seeds".to_owned(), |s| {
            format!(
                "mean {:.6} ± {:.6} (t95)",
                s.mean, s.t_confidence_95_halfwidth
            )
        });
        println!(
            "  {:<24} {:?} vs {} {:?}: {stats}",
            o.threshold.metric, o.outcome, o.threshold.threshold, o.threshold.direction
        );
    }
    if !record.missing_seeds.is_empty() {
        println!("  missing seeds: {:?}", record.missing_seeds);
    }
    if !record.unlisted_seeds.is_empty() {
        println!("  excluded unlisted seeds: {:?}", record.unlisted_seeds);
    }
    println!("  accepted: {}", record.accepted());
    if record.accepted() {
        0
    } else {
        EXIT_FAILURE
    }
}

/// `robot adapt identify <csv>`: columns `t_s,position_rad,torque_nm`.
pub fn adapt_identify(path: &Path, args: &[String]) -> i32 {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => return fail(format!("cannot read {}: {e}", path.display())),
    };
    let mut rows: Vec<[f64; 3]> = Vec::new();
    for (k, line) in text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty() && !l.starts_with('#'))
    {
        let v: Vec<f64> = line
            .split(',')
            .filter_map(|x| x.trim().parse().ok())
            .collect();
        if v.len() != 3 {
            return fail(format!(
                "line {}: expected t_s,position_rad,torque_nm",
                k + 1
            ));
        }
        rows.push([v[0], v[1], v[2]]);
    }
    if rows.len() < 5 {
        return fail("need at least five samples");
    }
    let dead_band = flag(args, "--dead-band")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.05);
    let samples: Vec<robot_adapt::sysid::JointSample> = (1..rows.len() - 1)
        .map(|i| {
            let (a, b, c) = (rows[i - 1], rows[i], rows[i + 1]);
            let (h1, h2) = (b[0] - a[0], c[0] - b[0]);
            let v = (c[1] - a[1]) / (h1 + h2);
            let acc = 2.0 * ((c[1] - b[1]) / h2 - (b[1] - a[1]) / h1) / (h1 + h2);
            robot_adapt::sysid::JointSample {
                velocity: v,
                acceleration: acc,
                torque: b[2],
            }
        })
        .collect();
    match robot_adapt::sysid::identify_joint(&samples, dead_band, 0.0) {
        Ok(j) => {
            let se = &j.fit.standard_errors;
            println!(
                "{}: {} samples (dead band {dead_band} rad/s)",
                path.display(),
                samples.len()
            );
            println!("  inertia  {:.6} ± {:.6} kg m^2", j.inertia, se[0]);
            println!("  viscous  {:.6} ± {:.6} N m s/rad", j.viscous, se[1]);
            println!("  coulomb  {:.6} ± {:.6} N m", j.coulomb, se[2]);
            println!("  offset   {:.6} ± {:.6} N m", j.offset, se[3]);
            println!(
                "  rms residual {:.6} N m, condition ratio {:.3e}",
                j.fit.rms_residual,
                j.fit.condition_ratio()
            );
            0
        }
        Err(e) => fail(e),
    }
}

/// `robot train`: PPO on the pendulum balance reference task (SPECS.md
/// Section 9.2: "PPO solves small reference tasks ... before humanoid
/// results are trusted"), checkpointed through `robot-learn`.
pub fn train(args: &[String]) -> i32 {
    use robot_learn::gaussian_policy::GaussianPolicy;
    use robot_learn::ppo::{PpoAgent, PpoConfig, RolloutContext};
    use robot_learn::rollout::RolloutBuffer;
    use robot_nn::init::initialize;
    use robot_nn::layer::Activation;
    use robot_sim::env::PendulumEnv;
    use robot_sim::pendulum::PendulumParams;
    use robot_tasks::balance::{BalanceTask, BalanceTaskConfig};
    use robot_types::ids::EnvId;
    use robot_types::rng::{Rng, SeedStream};

    let Some(out) = flag(args, "--out") else {
        return fail("train needs --out <dir>");
    };
    let iterations: u32 = flag(args, "--iterations")
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let seed: u64 = flag(args, "--seed")
        .and_then(|v| v.parse().ok())
        .unwrap_or(42);
    let mut rng = Rng::new(SeedStream::new(seed, 1));
    let (Ok(policy), Ok(value)) = (
        initialize(
            &[2, 16, 1],
            vec![Activation::Tanh, Activation::Identity],
            &mut rng,
        ),
        initialize(
            &[2, 16, 1],
            vec![Activation::Tanh, Activation::Identity],
            &mut rng,
        ),
    ) else {
        return fail("network initialization failed");
    };
    let mut agent = PpoAgent::new(
        GaussianPolicy::new(policy, -1.0),
        value,
        PpoConfig::default(),
        2,
    );
    let params = PendulumParams {
        length: 1.0,
        mass: 1.0,
        gravity: 9.80665,
        damping: 0.05,
    };
    let pi = core::f64::consts::PI;
    let mut env = PendulumEnv::with_reset_range(
        params,
        4.0,
        (pi - 0.3, pi + 0.3),
        (-0.3, 0.3),
        SeedStream::new(seed, 7),
    );
    let task = BalanceTask::new(BalanceTaskConfig {
        max_steps: 100,
        fall_angle: 1.0,
        ..BalanceTaskConfig::default()
    });
    let task_fn =
        |o: &[f64], a: &[f64], n: &[f64], step: u32, fail: bool| task.evaluate(o, a, n, step, fail);
    let mut buffer = RolloutBuffer::new(agent.transition_schema("pendulum-fixture/1.0"), 2000);
    let mut rollout_rng = Rng::new(SeedStream::new(seed, 5));
    for i in 0..iterations {
        buffer.clear();
        agent.collect_rollout(
            &mut env,
            &task_fn,
            &mut buffer,
            2000,
            RolloutContext {
                env_id: EnvId::new(0),
                robot_profile: "pendulum-fixture/1.0",
                episode_seed_base: 9000 + u64::from(i),
            },
            &mut rollout_rng,
        );
        let mean_reward = buffer
            .as_slice()
            .iter()
            .map(|t| t.reward.total())
            .sum::<f64>()
            / buffer.len().max(1) as f64;
        agent.update(&buffer);
        println!("iteration {:>4}: mean step reward {mean_reward:.4}", i + 1);
    }
    match robot_learn::checkpoint::save(Path::new(out), &agent, &rollout_rng) {
        Ok(()) => {
            println!(
                "checkpoint written to {out} (generation {:?})",
                agent.generation()
            );
            0
        }
        Err(e) => fail(e),
    }
}

/// `robot replay <run-dir> [--svg <out> --robot g1-usd|g1 [--every N]]`:
/// reads a `robot sim multibody --out` frame dump back through
/// `robot-viewer`, checks its timeline and optionally writes a skeleton
/// playback SVG.
pub fn replay(dir: &Path, args: &[String]) -> i32 {
    let log = match robot_viewer::pose::load_committed(dir) {
        Ok(l) => l,
        Err(e) => return fail(e),
    };
    if let Some(out) = flag(args, "--svg") {
        let mut a = args.to_vec();
        if flag(args, "--robot").is_none() {
            a.extend(["--robot".to_owned(), "g1-usd".to_owned()]);
        }
        let profile = match robot(&a) {
            Ok(p) => p,
            Err(e) => return fail(e),
        };
        let every = flag(args, "--every")
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);
        match robot_viewer::skeleton::render_playback(&profile, &log.frames, every) {
            Ok(svg) => {
                if let Err(e) = std::fs::write(out, svg) {
                    return fail(format!("cannot write {out}: {e}"));
                }
                println!("playback written to {out} (every {every}th frame)");
            }
            Err(e) => return fail(e),
        }
    }
    match log.into_timeline() {
        Ok(t) => {
            println!("{}: {} frames, timeline valid", dir.display(), t.len());
            0
        }
        Err(e) => fail(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reference_csv_round_trips() {
        let profile = robot_model::fixtures::unitree_g1_29dof().unwrap();
        let human_text = "HIERARCHY\nROOT Hips\n{\n  OFFSET 0 0 0\n  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n  End Site\n  {\n    OFFSET 0 10 0\n  }\n}\nMOTION\nFrames: 2\nFrame Time: 0.02\n0 90 0 0 0 0\n0 90 0 0 0 0\n";
        let human = motion_io::bvh::parse(human_text, "x", SourceLicense::unaudited("t")).unwrap();
        let channels = profile
            .actuated_joints()
            .map(|j| Channel {
                name: j.name.clone(),
                parent: None,
                offset: Vec3::ZERO,
            })
            .collect();
        let frames = (0..2)
            .map(|i| MotionFrame {
                timestamp: Timestamp::sampling(i * 20_000_000, human.clock),
                root: Transform::new(
                    FrameId::WORLD,
                    FrameId::WORLD,
                    Quat::IDENTITY,
                    Vec3::new(0.1, 0.0, 0.7),
                ),
                values: (0..profile.dof())
                    .map(|k| JointValue::Scalar {
                        value: 0.01 * k as f64,
                        unit: "rad",
                    })
                    .collect(),
                validity: vec![Validity::Valid; profile.dof()],
                contacts: Vec::new(),
            })
            .collect();
        let r = MotionRecord::new(
            RobotReference,
            "x",
            SourceLicense::unaudited("t"),
            "s",
            "g1",
            FrameId::WORLD,
            human.clock,
            channels,
            frames,
        )
        .unwrap();
        let back = read_reference_csv(&write_reference_csv(&r), &profile, &human).unwrap();
        assert_eq!(reference_id(&back), reference_id(&r));
    }
}

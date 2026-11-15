//! The `robot` command, per PLAN.md §4.
//!
//! PLAN.md states its own proposed CLI surface is a set of "interface
//! requirements, not currently implemented tools." This binary wires a
//! subset of that surface to real crate code — `data inspect`, `sim
//! qualify`, `sim multibody`, `eval summarize`, `motion validate` — and reports every
//! other named command as not yet implemented, with the specific reason,
//! rather than omitting it silently.

mod commands;

use commands::{EXIT_NOT_IMPLEMENTED, EXIT_USAGE};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    std::process::exit(run(&args));
}

fn run(args: &[String]) -> i32 {
    match args.first().map(String::as_str) {
        Some("data") => run_data(&args[1..]),
        Some("motion") => run_motion(&args[1..]),
        Some("sim") => run_sim(&args[1..]),
        Some("eval") => run_eval(&args[1..]),
        Some("train") => commands::pipeline::train(&args[1..]),
        Some("replay") => match args.get(1) {
            Some(dir) => commands::pipeline::replay(&PathBuf::from(dir), &args[2..]),
            None => usage_error("robot replay <run-dir> [--svg <out> [--robot g1-usd|g1] [--every N]]"),
        },
        Some("adapt") => match (args.get(1).map(String::as_str), args.get(2)) {
            (Some("identify"), Some(csv)) => commands::pipeline::adapt_identify(&PathBuf::from(csv), &args[3..]),
            _ => usage_error("robot adapt identify <csv> [--dead-band RAD_PER_S]"),
        },
        Some("deploy") => not_implemented(
            "deploy",
            Some("no hardware target is chosen (C-02); a verified robot-bundle/1 directory (`robot eval --bundle`) and robot_hardware::link::WireAdapter are the pieces it will wire"),
        ),
        Some("deps") => not_implemented(
            "deps",
            Some("the dependency knowledge base is developer tooling and this binary deliberately does not link robot-devtools; run `cargo run -p robot-devtools --bin devtools -- kb-sweep|kb-search|ledger-diff`"),
        ),
        Some("skills") => commands::bundle::skills(&args[1..]),
        Some(other) => {
            eprintln!("error: unknown command {other:?}");
            print_usage();
            EXIT_USAGE
        }
        None => {
            print_usage();
            EXIT_USAGE
        }
    }
}

fn run_data(args: &[String]) -> i32 {
    match args.first().map(String::as_str) {
        Some("inspect") => match args.get(1) {
            Some(path) => commands::data::inspect(&PathBuf::from(path)),
            None => usage_error("robot data inspect <path>"),
        },
        Some("import") => match args.get(1) {
            Some(path) => commands::pipeline::data_import(&PathBuf::from(path), &args[2..]),
            None => usage_error(
                "robot data import <source> --store <dir> [--license <id>] [--training]",
            ),
        },
        Some(other) => unknown_subcommand("data", other),
        None => usage_error("robot data <inspect|import> ..."),
    }
}

fn run_motion(args: &[String]) -> i32 {
    match args.first().map(String::as_str) {
        Some("validate") => match args.get(1) {
            Some(path) if args.iter().any(|a| a == "--reference") => {
                commands::pipeline::motion_validate_reference(&PathBuf::from(path), &args[2..])
            }
            Some(path) => commands::motion::validate(&PathBuf::from(path)),
            None => usage_error("robot motion validate <bvh> [--reference <csv> --map <file> [--robot g1]]"),
        },
        Some("retarget") => match args.get(1) {
            Some(path) => commands::pipeline::motion_retarget(&PathBuf::from(path), &args[2..]),
            None => usage_error("robot motion retarget <bvh> --map <file> --out <csv> [--robot g1|g1-usd] [--unit m|cm|mm] [--up y|z]"),
        },
        Some(other) => unknown_subcommand("motion", other),
        None => usage_error("robot motion <validate|retarget> ..."),
    }
}

fn run_sim(args: &[String]) -> i32 {
    match args.first().map(String::as_str) {
        Some("qualify") => run_qualify(&args[1..]),
        Some("multibody") => run_multibody(&args[1..]),
        Some("fidelity") => {
            let out = match parse_flag::<PathBuf>(&args[1..], "--out") {
                Ok(out) => out,
                Err(message) => return invalid_flag(&message),
            };
            commands::sim::fidelity(out.as_deref())
        }
        Some(other) => unknown_subcommand("sim", other),
        None => usage_error(
            "robot sim <qualify|multibody|fidelity> [--steps N] [--dt SECONDS] [--floating] [--zero-torque] [--out DIR] [--scene FILE] [--actuation drives|damping|zero] [--limit-margin RAD] [--integrator euler|implicitfast|rk4] [--tier rollout|reference|acceptance]",
        ),
    }
}

const QUALIFY_DEFAULT_STEPS: u32 = 2_000;
const QUALIFY_DEFAULT_DT: f64 = 0.001;
const MULTIBODY_DEFAULT_DT: f64 = 0.0001;
/// Fixed-base runs keep their short default; floating runs cover the full
/// qualification duration (see `commands::sim::default_steps`).
const MULTIBODY_FIXED_BASE_STEPS: u32 = 2_000;

fn run_qualify(flags: &[String]) -> i32 {
    let steps = match parse_flag::<u32>(flags, "--steps") {
        Ok(steps) => steps.unwrap_or(QUALIFY_DEFAULT_STEPS),
        Err(message) => return invalid_flag(&message),
    };
    let dt_seconds = match parse_flag::<f64>(flags, "--dt") {
        Ok(dt) => dt.unwrap_or(QUALIFY_DEFAULT_DT),
        Err(message) => return invalid_flag(&message),
    };
    commands::sim::qualify(steps, dt_seconds)
}

fn run_multibody(flags: &[String]) -> i32 {
    let floating = flags.iter().any(|a| a == "--floating");
    let zero_torque = flags.iter().any(|a| a == "--zero-torque");
    // `specs/09` section 12.2: a named tier is a bundle of knobs a run starts
    // from, and any explicit knob overrides it. `--tier rollout --dt 5e-4` is
    // therefore the rollout tier at a finer timestep, exactly as the section's
    // closing requirement asks.
    let tier = match flags
        .iter()
        .position(|a| a == "--tier")
        .map(|i| flags.get(i + 1))
    {
        None => None,
        Some(Some(v)) => match robot_sim::options::Tier::from_name(v) {
            Some(t) => Some(t),
            None => return invalid_flag("--tier expects rollout, reference or acceptance"),
        },
        Some(None) => return invalid_flag("--tier needs a value"),
    };
    let tier_options = tier.map(robot_sim::options::Tier::options);
    let dt_seconds = match parse_flag::<f64>(flags, "--dt") {
        Ok(dt) => dt.unwrap_or_else(|| {
            tier_options
                .as_ref()
                .map_or(MULTIBODY_DEFAULT_DT, |o| o.timestep.as_secs_f64())
        }),
        Err(message) => return invalid_flag(&message),
    };
    let steps = match parse_flag::<u32>(flags, "--steps") {
        Ok(Some(steps)) => steps,
        Ok(None) if floating => match commands::sim::default_steps(dt_seconds) {
            Ok(steps) => steps,
            Err(message) => return invalid_flag(&message),
        },
        Ok(None) => MULTIBODY_FIXED_BASE_STEPS,
        Err(message) => return invalid_flag(&message),
    };
    let out = match parse_flag::<PathBuf>(flags, "--out") {
        Ok(out) => out,
        Err(message) => return invalid_flag(&message),
    };
    let scene = match parse_flag::<PathBuf>(flags, "--scene") {
        Ok(scene) => scene,
        Err(message) => return invalid_flag(&message),
    };
    // `--contact penalty|impedance` selected a legacy contact formulation that
    // the example's specifications replace, so it no longer exists. Rejecting
    // it explicitly rather than ignoring it: an old invocation that silently
    // ran a different simulation than it asked for would be worse than one
    // that fails.
    if flags.iter().any(|a| a == "--contact") {
        return invalid_flag(
            "--contact was removed: the per-point penalty contact it selected is a formulation \
             the example's specifications replace, not a fidelity option. There is one \
             simulation path. Use --actuation to choose how the model's declared drives are \
             applied",
        );
    }
    // `specs/09` section 12.1 makes every fidelity knob a per-run parameter.
    // `limit_margin` was reachable only from `SimOptions`, and the zero-torque
    // collapse is why it matters: with no margin a joint-limit row appears only
    // after the joint has crossed its stop, so 17 of the G1's 29 joints leave
    // their range under collapse loads, by up to 0.31 rad.
    let limit_margin = match parse_flag::<f64>(flags, "--limit-margin") {
        Ok(v) => v.unwrap_or(0.0),
        Err(message) => return invalid_flag(&message),
    };
    if !limit_margin.is_finite() || limit_margin < 0.0 {
        return invalid_flag("--limit-margin must be finite and non-negative");
    }
    // `specs/09` item 1: the integrator is a per-run knob.
    let integrator = match flags
        .iter()
        .position(|a| a == "--integrator")
        .map(|i| flags.get(i + 1))
    {
        None => tier_options
            .as_ref()
            .map_or(robot_sim::options::Integrator::SemiImplicitEuler, |o| {
                o.integrator
            }),
        Some(Some(v)) if v == "euler" => robot_sim::options::Integrator::SemiImplicitEuler,
        Some(Some(v)) if v == "implicitfast" => robot_sim::options::Integrator::ImplicitFast,
        Some(Some(v)) if v == "rk4" => robot_sim::options::Integrator::Rk4,
        Some(_) => {
            return invalid_flag("--integrator expects euler, implicitfast or rk4");
        }
    };
    let actuation = match flags
        .iter()
        .position(|a| a == "--actuation")
        .map(|i| flags.get(i + 1))
    {
        None => robot_sim::world::ActuationMode::Drives,
        Some(Some(v)) if v == "drives" => robot_sim::world::ActuationMode::Drives,
        Some(Some(v)) if v == "damping" => robot_sim::world::ActuationMode::Damping,
        Some(Some(v)) if v == "zero" => robot_sim::world::ActuationMode::Zero,
        Some(_) => {
            return invalid_flag("--actuation expects drives, damping or zero");
        }
    };
    commands::sim::multibody_scene_to(commands::sim::MultibodyRun {
        steps,
        dt_seconds,
        floating,
        zero_torque,
        out: out.as_deref(),
        scene: scene.as_deref(),
        actuation,
        limit_margin,
        integrator,
    })
}

fn run_eval(args: &[String]) -> i32 {
    match args.first().map(String::as_str) {
        Some("summarize") => commands::eval::summarize_cli(&args[1..]),
        Some("run") => commands::pipeline::eval_run(&args[1..]),
        Some("--bundle") => commands::bundle::eval_bundle(&args[1..]),
        Some(other) => unknown_subcommand("eval", other),
        None => usage_error("robot eval summarize <values...>"),
    }
}

/// Reads `flag`'s value. `Ok(None)` means the flag is absent, so a default may
/// apply. A present flag with a missing or unparsable value is an error, never a
/// silent default.
fn parse_flag<T: std::str::FromStr>(args: &[String], flag: &str) -> Result<Option<T>, String> {
    let Some(index) = args.iter().position(|a| a == flag) else {
        return Ok(None);
    };
    let Some(raw) = args.get(index + 1) else {
        return Err(format!("{flag} needs a value"));
    };
    raw.parse::<T>()
        .map(Some)
        .map_err(|_| format!("invalid {flag} value {raw:?}"))
}

fn invalid_flag(message: &str) -> i32 {
    eprintln!("error: {message}");
    EXIT_USAGE
}

fn not_implemented(command: &str, reason: Option<&str>) -> i32 {
    eprint!("`robot {command}` is not implemented in this revision");
    match reason {
        Some(r) => eprintln!(": {r}."),
        None => eprintln!("."),
    }
    EXIT_NOT_IMPLEMENTED
}

fn unknown_subcommand(group: &str, sub: &str) -> i32 {
    eprintln!("error: unknown {group} subcommand {sub:?}");
    EXIT_USAGE
}

fn usage_error(usage: &str) -> i32 {
    eprintln!("usage: {usage}");
    EXIT_USAGE
}

fn print_usage() {
    eprintln!("usage: robot <command> ...");
    eprintln!();
    eprintln!("implemented:");
    eprintln!("  robot data inspect <path>");
    eprintln!("  robot motion validate <path>");
    eprintln!("  robot sim qualify [--steps N] [--dt SECONDS]");
    eprintln!("  robot sim multibody [--steps N] [--dt SECONDS] [--floating] [--zero-torque]\n           [--out DIR] [--scene FILE] [--actuation drives|damping|zero]\n           [--limit-margin RAD] [--integrator euler|implicitfast|rk4]\n           [--tier rollout|reference|acceptance]");
    eprintln!("  robot eval summarize <values...>");
    eprintln!("  robot eval --bundle <dir> --robot <g1-usd|go2-usd|h1-usd>");
    eprintln!("  robot skills <list|search> --manifest <skill-manifest.toml> [query...]");
    eprintln!("  robot data import, robot motion retarget, robot eval run,");
    eprintln!("  robot train, robot adapt, robot replay (see each command's --help)");
    eprintln!();
    eprintln!("named in PLAN.md but not implemented yet (each refuses with its reason):");
    eprintln!("  robot deps, robot deploy");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_arguments_is_a_usage_error() {
        assert_eq!(run(&[]), EXIT_USAGE);
    }

    #[test]
    fn an_unknown_top_level_command_is_a_usage_error() {
        assert_eq!(run(&["frobnicate".to_owned()]), EXIT_USAGE);
    }

    #[test]
    fn a_named_but_unbuilt_command_reports_not_implemented_not_unknown() {
        assert_eq!(run(&["skills".to_owned()]), EXIT_USAGE);
        assert_eq!(run(&["deploy".to_owned()]), EXIT_NOT_IMPLEMENTED);
        assert_eq!(
            run(&["deps".to_owned(), "inventory".to_owned()]),
            EXIT_NOT_IMPLEMENTED
        );
    }

    #[test]
    fn data_inspect_without_a_path_is_a_usage_error() {
        assert_eq!(run(&["data".to_owned(), "inspect".to_owned()]), EXIT_USAGE);
    }

    #[test]
    fn sim_qualify_with_default_flags_runs() {
        assert_eq!(run(&["sim".to_owned(), "qualify".to_owned()]), 0);
    }

    fn sim(args: &[&str]) -> i32 {
        let mut full = vec!["sim".to_owned()];
        full.extend(args.iter().map(|s| (*s).to_owned()));
        run(&full)
    }

    #[test]
    fn sim_fidelity_runs_the_declared_battery_and_passes() {
        assert_eq!(sim(&["fidelity"]), 0);
    }

    #[test]
    fn multibody_rejects_an_unknown_tier_and_a_missing_tier_value() {
        assert_eq!(sim(&["multibody", "--tier", "coarse"]), EXIT_USAGE);
        assert_eq!(sim(&["multibody", "--tier"]), EXIT_USAGE);
    }

    #[test]
    fn multibody_rejects_a_dt_that_is_not_finite_and_positive() {
        for dt in ["0", "-0.001", "NaN", "inf", "abc"] {
            assert_eq!(sim(&["multibody", "--dt", dt]), EXIT_USAGE, "dt {dt}");
        }
    }

    #[test]
    fn multibody_rejects_an_unparsable_or_zero_or_negative_steps() {
        for steps in ["0", "-5", "abc"] {
            assert_eq!(
                sim(&["multibody", "--steps", steps, "--dt", "0.0001"]),
                EXIT_USAGE,
                "steps {steps}"
            );
        }
    }

    #[test]
    fn multibody_rejects_a_flag_with_no_value() {
        assert_eq!(sim(&["multibody", "--dt"]), EXIT_USAGE);
        assert_eq!(sim(&["multibody", "--steps"]), EXIT_USAGE);
    }

    #[test]
    fn multibody_rejects_a_timestep_that_quantizes_to_zero_nanoseconds() {
        assert_eq!(
            sim(&["multibody", "--floating", "--dt", "1e-12"]),
            EXIT_USAGE
        );
        assert_eq!(
            sim(&["multibody", "--floating", "--dt", "1e-300"]),
            EXIT_USAGE
        );
    }

    #[test]
    fn multibody_rejects_a_floating_step_count_above_the_cap() {
        assert_eq!(
            sim(&["multibody", "--floating", "--dt", "1e-7"]),
            EXIT_USAGE
        );
    }

    #[test]
    fn qualify_rejects_invalid_flags_before_simulating() {
        assert_eq!(sim(&["qualify", "--dt", "0"]), EXIT_USAGE);
        assert_eq!(sim(&["qualify", "--steps", "-5"]), EXIT_USAGE);
        assert_eq!(sim(&["qualify", "--steps", "0"]), EXIT_USAGE);
    }

    #[test]
    fn parse_flag_distinguishes_absent_from_malformed() {
        let args: Vec<String> = ["--dt", "0.5", "--steps", "x"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        assert_eq!(parse_flag::<f64>(&args, "--dt"), Ok(Some(0.5)));
        assert_eq!(parse_flag::<f64>(&args, "--other"), Ok(None));
        assert!(parse_flag::<u32>(&args, "--steps").is_err());
    }

    #[test]
    fn eval_summarize_runs_end_to_end() {
        let args = [
            "eval".to_owned(),
            "summarize".to_owned(),
            "1.0".to_owned(),
            "2.0".to_owned(),
        ];
        assert_eq!(run(&args), 0);
    }
}

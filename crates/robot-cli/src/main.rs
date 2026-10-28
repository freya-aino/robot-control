//! The `robot` command, per PLAN.md §4.
//!
//! PLAN.md states its own proposed CLI surface is a set of "interface
//! requirements, not currently implemented tools." This binary wires a
//! subset of that surface to real crate code — `data inspect`, `sim
//! qualify`, `eval summarize`, `motion validate` — and reports every
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
        Some(group @ ("train" | "skills" | "deps" | "adapt" | "deploy" | "replay")) => {
            not_implemented(group, args.get(1).map(String::as_str))
        }
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
        Some("import") => not_implemented("data import", None),
        Some(other) => unknown_subcommand("data", other),
        None => usage_error("robot data <inspect|import> ..."),
    }
}

fn run_motion(args: &[String]) -> i32 {
    match args.first().map(String::as_str) {
        Some("validate") => match args.get(1) {
            Some(path) => commands::motion::validate(&PathBuf::from(path)),
            None => usage_error("robot motion validate <path>"),
        },
        Some("retarget") => not_implemented(
            "motion retarget",
            Some("no generic source-skeleton-to-robot joint-mapping config format exists yet; motion_retarget::ik::solve is real but needs targets this CLI cannot derive generically"),
        ),
        Some(other) => unknown_subcommand("motion", other),
        None => usage_error("robot motion <validate|retarget> ..."),
    }
}

fn run_sim(args: &[String]) -> i32 {
    match args.first().map(String::as_str) {
        Some("qualify") => run_qualify(&args[1..]),
        Some("multibody") => run_multibody(&args[1..]),
        Some(other) => unknown_subcommand("sim", other),
        None => usage_error(
            "robot sim <qualify|multibody> [--steps N] [--dt SECONDS] [--floating] [--out DIR]",
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
    let dt_seconds = match parse_flag::<f64>(flags, "--dt") {
        Ok(dt) => dt.unwrap_or(MULTIBODY_DEFAULT_DT),
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
    commands::sim::multibody_to(steps, dt_seconds, floating, zero_torque, out.as_deref())
}

fn run_eval(args: &[String]) -> i32 {
    match args.first().map(String::as_str) {
        Some("summarize") => commands::eval::summarize_cli(&args[1..]),
        Some("--bundle") | Some("run") => not_implemented(
            "eval run",
            Some("no bundle/suite format or suite runner exists yet; robot_eval::stats::summarize is real and reachable via `robot eval summarize <values...>`"),
        ),
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
    eprintln!("  robot sim multibody [--steps N] [--dt SECONDS] [--floating] [--zero-torque]");
    eprintln!("  robot eval summarize <values...>");
    eprintln!();
    eprintln!("named in PLAN.md but not implemented yet:");
    eprintln!("  robot data import, robot motion retarget, robot eval run,");
    eprintln!("  robot train, robot skills, robot deps, robot adapt, robot deploy, robot replay");
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
        assert_eq!(run(&["train".to_owned()]), EXIT_NOT_IMPLEMENTED);
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

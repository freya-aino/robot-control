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
        Some("qualify") => {
            let steps = flag_value(&args[1..], "--steps")
                .and_then(|v| v.parse().ok())
                .unwrap_or(2_000);
            let dt_seconds = flag_value(&args[1..], "--dt")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.001);
            commands::sim::qualify(steps, dt_seconds)
        }
        Some("multibody") => {
            let steps = flag_value(&args[1..], "--steps")
                .and_then(|v| v.parse().ok())
                .unwrap_or(2_000);
            let dt_seconds = flag_value(&args[1..], "--dt")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0001);
            commands::sim::multibody(steps, dt_seconds)
        }
        Some(other) => unknown_subcommand("sim", other),
        None => usage_error("robot sim <qualify|multibody> [--steps N] [--dt SECONDS]"),
    }
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

fn flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
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
    eprintln!("  robot sim multibody [--steps N] [--dt SECONDS]");
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

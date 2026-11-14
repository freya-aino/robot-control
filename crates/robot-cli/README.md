# robot-cli

The `robot` binary, per PLAN.md §4's proposed CLI surface. Depends on
every other crate in the workspace except `robot-devtools` (deliberately —
see below) and the feature-gated `robot-sim-gpu`.

PLAN.md states its own proposed surface is "interface requirements, not
currently implemented tools." This crate wires a real, honestly-scoped
subset of that surface to actual crate code, and reports every other
named command as not implemented with the specific reason, rather than
omitting it silently or faking a stub that looks finished.

## What's really implemented

| Command | Backed by |
| --- | --- |
| `robot data inspect <path>` | `robot_artifacts::chunk::Chunk::decode` / `robot_artifacts::toml::parse`, auto-detected |
| `robot motion validate <path>` | `motion_io::bvh::parse` + `robot_types::motion::MotionRecord::validate` |
| `robot sim qualify [--steps N] [--dt SECONDS]` | `robot_sim::pendulum`'s undamped energy-conservation benchmark (SPECS.md §9.1, Appendix B.4 item 2) |
| `robot sim multibody [--floating] [--zero-torque] [--dt S] [--steps N] [--out DIR] [--scene FILE] [--actuation drives\|damping\|zero] [--limit-margin RAD] [--integrator euler\|implicitfast\|rk4]` | `robot_sim::world::World` (composite dynamics, exact manifolds, Newton constraint solver) on the vendored G1 USD model with the scene's ground plane (ADR 0015). `--out` writes `robot_viewer::pose` frames (CSV, JSON Lines, SVG) and two `robot_usd::replay` USDA files. A floating run is judged over 2 s on pelvis drop, penetration and settled energy; a shorter run is INCONCLUSIVE |
| `robot eval summarize <values...>` | `robot_eval::stats::summarize` |

## The motion, evaluation, adaptation and training commands

| Command | What it does |
| --- | --- |
| `robot data import <source> --store <dir> [--license <id>] [--training]` | Detects BVH, C3D or AMASS by content, stores the original in a content-addressed `BlobStore` before parsing, and prints the content ID and inspection (rate, validity, licence, guessed up axis and unit) |
| `robot motion retarget <bvh> --map <file> --out <csv> [--robot g1\|g1-usd] [--unit m\|cm\|mm] [--up y\|z]` | Converts axes and units (guessed and stated, or given). Estimates the ground plane and foot contacts for the map's feet with printed default thresholds. Retargets with the `retarget-map/1` file and writes a `robot-reference/1` CSV. Prints task error, planted slip, unfinished frames and velocity excess |
| `robot motion validate <bvh> --reference <csv> --map <file>` | Runs `motion_validate::validate` on the source and the reference with printed default thresholds, prints the `motion-curation/1` manifest entry, and exits 1 on rejection. Without `--reference` it validates the source record, as before |
| `robot eval run --profile <toml> --results <csv>` | Evaluates `seed,metric,value` rows against an `acceptance-profile/1` file: pass, fail or inconclusive per metric by t interval, with missing and unlisted seeds |
| `robot eval --bundle <dir> --robot <g1-usd\|go2-usd\|h1-usd>` | Reads a `robot-bundle/1` directory (ADR 0016): the digest-sealed manifest and every content-addressed weight blob verify, then `check_activatable` runs against the named vendored robot's real joint limits in sim-eval mode. Exit 0 is "verified and activatable", not a quality claim |
| `robot skills <list\|search> --manifest <toml> [query...]` | Reads a `skill-manifest/1` library (ADR 0016) and lists the typed descriptors, or searches them with the §10.1 lexical catalog (TF-IDF, no model call) |
| `robot adapt identify <csv> [--dead-band RAD_PER_S]` | Identifies one joint's inertia, viscous and Coulomb friction, and offset from `t_s,position_rad,torque_nm` rows, with standard errors and the condition ratio |
| `robot train [--iterations N] [--seed S] --out <dir>` | PPO on the pendulum balance reference task (§9.2: reference tasks before humanoids) with the configuration `robot-learn` measured to stabilize it (150 updates of 2000 steps by default), printing mean step reward per iteration and writing a `robot-learn` checkpoint |
| `robot replay <run-dir> [--svg <out> [--robot g1-usd\|g1] [--every N]]` | Reads a `robot sim multibody --out` frame dump back through `robot-viewer` and checks its timeline; with `--svg`, writes a side-and-front skeleton playback (SMIL, no script) |

A smoke run on a 30-frame cm/Y-up BVH arm reach, imported, retargeted
onto the G1 and validated, gave 2.2 mm task error, zero planted-foot slip
and an accepted manifest entry. That run is a check that the pipeline is
wired, not an evaluation.

## What's named but not implemented, and why

- `robot deploy`: waits on a hardware target (C-02); the bundle half of
  its job now exists (`robot eval --bundle` verifies a `robot-bundle/1`
  directory). The library
  pieces they would wire are `robot_control`, `robot_hardware::link` and
  `robot_skills::selection`.
- `robot deps`: deliberately not here (see below). Run
  `cargo run -p robot-devtools --bin devtools -- kb-sweep|kb-search|ledger-diff`.

Each of these exits with code `2` and a one-line reason on stderr, distinct
from code `64` for a genuine usage error (bad flags, unknown command, `--out`
without `--floating`), code `1` for a command that ran and found a real
problem (a rejected motion, a failed qualification run) and code `3` for a
floating-base `sim multibody` run shorter than the 2 s qualification
duration (INCONCLUSIVE).

## Why this crate does not depend on `robot-devtools`

SPECS.md §2.3.1 draws a boundary around development-time automation
("no shipping crate depends on it"); `robot-cli` is PLAN.md's shipped
operator surface, not that boundary's tooling, so it does not add
`robot-devtools` to its dependencies even though a `robot deps` command
is named in PLAN.md's proposed surface. That command is listed above as
not implemented rather than implemented against the wrong side of that
boundary.

## Testing

`cargo test -p robot-cli` — 46 tests (one is the `robot-reference/1` CSV round trip in `commands::pipeline`) across `main.rs` (dispatch: no
arguments, unknown command, named-but-unimplemented reporting, missing
required argument) and each command module (chunk/TOML auto-detection and
rejection of garbage input, BVH parse-and-validate on a well-formed and a
malformed file, the pendulum benchmark passing at a reasonable timestep
and failing at a deliberately unstable one, numeric-argument parsing and
the distinct usage-vs-failure exit codes for `eval summarize`, multibody
flag validation (`dt`, `steps`, `--out` requires `--floating`, frame and
timestamp caps), the PASS/FAIL/INCONCLUSIVE stance verdict and
byte-identical repeated frame dumps).

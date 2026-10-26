# robot-cli

The `robot` binary, per PLAN.md §4's proposed CLI surface. Depends on
every other crate in the workspace except `robot-devtools` (deliberately —
see below).

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
| `robot eval summarize <values...>` | `robot_eval::stats::summarize` |

## What's named but not implemented, and why

- `robot data import` — no import-pipeline config format exists yet.
- `robot motion retarget` — `motion_retarget::ik::solve` is real, but it
  needs Cartesian link targets; deriving those generically from an
  arbitrary source skeleton needs a joint-mapping config format this
  workspace doesn't have. `robot motion validate` (above) validates the
  *source* motion record, which is as much of this pipeline as is wired.
- `robot eval run` (the full `--bundle <id> --suite <path>` form) — no
  bundle or suite-runner format exists yet; the statistic itself is
  reachable via `robot eval summarize`.
- `robot train`, `robot skills`, `robot deps`, `robot adapt`, `robot
  deploy`, `robot replay` — named in PLAN.md §4, not built in this
  revision.

Each of these exits with code `2` and a one-line reason on stderr, distinct
from code `64` for a genuine usage error (bad flags, unknown command) and
code `1` for a command that ran and found a real problem (a rejected
motion, a failed qualification run).

## Why this crate does not depend on `robot-devtools`

SPECS.md §2.3.1 draws a boundary around development-time automation
("no shipping crate depends on it"); `robot-cli` is PLAN.md's shipped
operator surface, not that boundary's tooling, so it does not add
`robot-devtools` to its dependencies even though a `robot deps` command
is named in PLAN.md's proposed surface. That command is listed above as
not implemented rather than implemented against the wrong side of that
boundary.

## Testing

`cargo test -p robot-cli` — 17 tests across `main.rs` (dispatch: no
arguments, unknown command, named-but-unimplemented reporting, missing
required argument) and each command module (chunk/TOML auto-detection and
rejection of garbage input, BVH parse-and-validate on a well-formed and a
malformed file, the pendulum benchmark passing at a reasonable timestep
and failing at a deliberately unstable one, numeric-argument parsing and
the distinct usage-vs-failure exit codes for `eval summarize`).

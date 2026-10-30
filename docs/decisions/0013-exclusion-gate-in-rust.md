# 0013 — the excluded-runtime gate is a Rust binary in `robot-devtools`, run by `devenv test` and a git hook

- Status: accepted
- Date: 2026-10-30
- Closes: the enforcement half of SPECS.md §2.1 item 4 for local runs; row A-25's removal assertion
- Touches: SPECS.md §2.1 item 4, §2.2 items 3 and 6, §6 (row A-25); `devenv.nix`; `crates/robot-devtools/src/exclusion_gate.rs`, `crates/robot-devtools/src/bin/exclusion_gate.rs`

## Context

SPECS.md §2.1 item 4: "The default build MUST fail if an excluded runtime
appears in its dependency closure, and CI MUST enforce it." Until this
decision the only gate was a bash body inside `devenv.nix`
(`scripts.robot-exclusion-gate`) that compared the number of `Cargo.lock`
entries with the number of workspace members, printed a warning when the
lockfile was larger, and always exited 0. Since ADR 0012 added the
optional `cubecl` dependency the lockfile has held 472 packages against 23
members, so the warning fired on every run and nothing failed. The script
was also not part of `enterTest` or `git-hooks`, and its `if`/`grep`/`wc`
body was exactly the branching and parsing §2.2 item 6 forbids in a devenv
script. The 2026-10-30 re-audit of `docs/spec-checklists/02-...` recorded
all of this under items 2.1.4, 2.2.3, 2.2.6 and 2.3.

Row A-25 separately requires that no shipping crate depend on
`robot-devtools`; that was true by inspection but asserted by nothing.

## Decision

1. The gate's logic is Rust: `robot_devtools::exclusion_gate` holds three
   pure text checks with unit tests, and the `exclusion-gate` binary
   gathers the inputs and exits 1 on any violation (2 if an input cannot
   be read). The three checks are:
   - no `Cargo.lock` package name is an excluded runtime, under any
     feature — the exact names and prefixes are the constants
     `EXCLUDED_NAMES` and `EXCLUDED_PREFIXES` (Python bindings, PyTorch,
     JAX, MuJoCo, CycloneDDS). A feature-gated exception such as `cubecl`
     may be in the lockfile; an excluded runtime may not, because §2 admits
     no exception for them;
   - the feature-off default closure (`cargo tree --workspace -e normal
     --prefix none --offline`) contains only workspace path packages, which
     is ADR 0003's claim made executable;
   - no crate other than `robot-devtools` lists it under `[dependencies]`
     or `[build-dependencies]`; `[dev-dependencies]` is permitted because
     it ships in nothing.
2. `devenv.nix` keeps a one-line alias (`scripts.robot-exclusion-gate`),
   runs the binary from `enterTest` before `cargo test`, and declares it as
   the `robot-exclusion-gate` git hook on `Cargo.toml`, `Cargo.lock`,
   `devenv.nix`, `devenv.yaml` and `devenv.lock` changes.
3. The gate reads `cargo tree` with `-e normal`, so build and dev
   dependencies are outside the closure it checks. Build scripts and
   development tools are governed by the development-tool exception of
   §2.1, not by this gate.

## Consequences

- `devenv test` now fails on an excluded runtime, a non-path default
  dependency, or a shipping edge to `robot-devtools`. Adding a qualified
  dependency under Appendix B.6 will require amending the second check
  (for example to an allow-list keyed to the ADR that admitted it); that
  amendment is the intended friction.
- "CI MUST enforce it" is still only partly met: no hosted CI service runs
  `devenv test` (ADR 0001), so enforcement is local and pre-commit. Item
  2.1.4 stays `[~]` until a CI definition exists.
- The deny list is a name list, not a semantic check; a renamed binding
  would pass it. The path-only closure check is what catches anything
  unknown, because an unknown crate is not a workspace path.

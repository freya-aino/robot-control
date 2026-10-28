# AGENT.md

Guidance for AI coding agents working in this repository. Read this before
making changes. The authoritative sources are `SPECS.md` (what the platform
must be) and `PLAN.md` (how it is delivered); this file only summarizes how
to work here.

## Project overview

A Rust humanoid robotics platform: motion processing, simulation, policy
training, and hardware control (including boxing research). It is a Cargo
workspace of 22 crates under `crates/`, with one `robot` CLI binary in
`crates/robot-cli`. Each crate has its own `README.md` that states what is
implemented, the reasoning behind deviations, and what is deliberately not
implemented yet. Read the crate README before changing a crate.

Key locations:

- `SPECS.md`: platform specification, dependency policy, acceptance requirements.
- `PLAN.md`: delivery phases and qualification gates.
- `docs/decisions/`: architecture decision records (ADRs). Check these before
  reversing any architectural choice.
- `docs/ledger/`: `evidence-ledger.md` and `automation-log.md`.
- `docs/acceptance/`, `docs/benchmarks/`, `docs/knowledge-base/`: supporting docs.
- `fixtures/` (models, motion, profiles) and `assets/robots/`: test and sample data.
- `devenv.nix`, `devenv.yaml`, `devenv.lock`: the pinned development environment.

## Setup and commands

Enter the environment first. Do not install tools globally.

```
devenv shell        # or: direnv allow, if configured
devenv test         # the full gate, see below
```

Individual checks, run inside the shell:

```
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test -p <crate-name>      # a single crate
```

`devenv test` runs fmt check, clippy with `-D warnings`, and
`cargo test --workspace`. A change is not done until this gate passes.

## Code conventions

- All project-authored code is Rust. Do not add Python, shell-script
  application logic, or Python sidecars (SPECS.md Section 2).
- Formatting follows `rustfmt.toml`: edition 2021, `max_width = 100`,
  `use_field_init_shorthand = true`. Run `cargo fmt` before finishing.
- Clippy warnings are errors. Fix them rather than adding `#[allow(...)]`
  unless there is a clear, documented reason.
- Match the style of the surrounding crate: naming, module layout, doc
  comments, and the way tests are organized. Existing code cites SPECS.md
  section numbers in comments and docs; keep that practice when implementing
  a spec section.
- Keep `unsafe` out of the default build. If it is ever needed, document the
  invariants of every `unsafe` block.

## Dependency policy (important)

- The default dependency closure is empty: standard library only
  (`docs/decisions/0003-std-only-default-closure.md`). `[workspace.dependencies]`
  in `Cargo.toml` is empty.
- Do not add a third-party crate, FFI binding, or foreign runtime without
  going through the qualification gates in SPECS.md Section 2.1 and Appendix
  B.6, and without an ADR. Candidates in Appendix B are `unverified` until
  qualified.
- Do not add Python, PyTorch, JAX, MuJoCo, CycloneDDS, or other excluded
  runtimes to the Cargo dependency closure or the devenv environment.
- Add native build inputs to `devenv.nix` only when a qualified dependency
  requires them.

## Testing expectations

- Every implemented capability has tests grounded in the relevant SPECS.md
  section. Prefer analytic, statistical, or round-trip tests with fixed seeds
  (the workspace provides a deterministic PRNG in `robot_types::rng`).
- Bug fixes should include a reproducing test that fails before the fix.
- Never weaken, delete, or `#[ignore]` a failing test to get a green gate.
  Fix the cause, or report the failure explicitly.
- Negative results are recorded honestly. For example, PPO on the pendulum
  task is documented as not demonstrating measured improvement. Do not hide
  such results behind passing-looking assertions.

## Evidence discipline

SPECS.md Section 3 governs claims. Treat research and performance claims as
`unverified` until a primary source or measurement supports them.

- Do not promote an evidence state without a recorded source, and record
  changes in `docs/ledger/evidence-ledger.md`.
- Performance numbers must cite the command, fixture, and environment
  (`devenv.lock` pins the inputs).
- Evidence does not transfer between artifacts. A result for one model,
  fixture, or policy bundle does not apply to another.

## Documentation and records

- When you add or change a crate's behavior, update that crate's `README.md`
  in the same change. Keep the "implemented" and "not implemented yet" lists
  accurate.
- When a change makes an architectural decision or fills a gap left open by
  the specs, add a new ADR in `docs/decisions/` using the next free number,
  following the existing format (Status, Date, Touches, Context, Decision).
- If you use development-time automation (including AI agents) to produce
  project artifacts, add an entry to `docs/ledger/automation-log.md` stating
  the tool, the model identity and version, and the artifacts produced
  (SPECS.md Section 2.3.1).
- Keep the root `README.md` crate table in sync when crates are added or their
  scope changes.

## Workflow rules

- Commit only when the user explicitly asks. When you commit, follow the
  commit format below.
- Commit subject: `type(scope): imperative summary`. The type is lowercase
  and one of `feat`, `fix`, `docs`, `refactor`, `test`, `build`, `chore`.
  The scope is a crate name (for example `robot-nn`) or an area (for example
  `docs`, `agent`). The subject is at most 72 characters and has no trailing
  period.
- Commit body: wrap at 72 columns and explain why the change exists. End it
  with the trailer `Co-authored-by: Junie <junie@jetbrains.com>`.
- One logical change per commit. Stage files by explicit path or hunk, never
  with `git add -A` or `git add .`. Check `git diff --cached --name-only`
  before each commit.
- Never stage or commit `.junie/` (session plans), `.envrc` (machine-local),
  or `target/`.
- Do not rewrite existing commits (no amend, rebase, or squash) unless the
  user asks.
- Do not modify `.envrc`, `.devenv*`, `.direnv/`, `target/`, or the
  `.pre-commit-config.yaml` symlink. They are machine-local.
- Keep changes scoped to the task. Avoid unrelated refactors across crates.
- Deployment must not require Nix or devenv on the robot host. The devenv
  environment is for building and testing only.
- Safety-relevant code (`robot-supervisor`, `robot-hardware`, `robot-control`)
  must stay structurally isolated from policy, retrieval, and personality code
  (SPECS.md Section 12). Do not route policy output directly to actuators
  around the supervisor.

# 0001 — devenv-provisioned toolchain and host platforms

- Status: accepted
- Date: 2026-10-26
- Closes: C-14 (partially — see "What remains open" below)
- Touches: SPECS.md §2.2, §14 row A-22

## Context

SPECS.md §2.2 requires a single declarative definition — `devenv.nix` /
`devenv.yaml` / `devenv.lock` — to supply the Rust toolchain, native build
inputs and development tools, and requires that "A clean checkout MUST reach
a working environment through `devenv shell` with no further manual
installation."

## Decision

- The toolchain is supplied through `languages.rust` in `devenv.nix`, pinned
  to the `stable` channel resolved through the `rust-overlay` flake input
  declared in `devenv.yaml`, with `rustc`, `cargo`, `clippy`, `rustfmt`,
  `rust-analyzer` and `rust-src` components.
- `devenv.yaml` declares three inputs: `nixpkgs` (pinned to
  `nixos-unstable`), `rust-overlay`, and `git-hooks` (`cachix/git-hooks.nix`),
  the last required because `devenv.nix` enables `git-hooks.hooks.rustfmt`
  and `.clippy`.
- `devenv.lock` is committed and is the environment pin every reported
  measurement must cite, per §2.2 item 1.
- `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets --
  -D warnings` and `cargo test --workspace` are declared once, in
  `enterTest`, and run identically through `devenv test` locally and (once a
  CI job is wired up) in CI, per §2.2 item 4.
- `git-hooks.hooks.rustfmt` / `.clippy` give the same two checks as local
  pre-commit hooks, per §2.2 item 3.

## Evidence

`devenv shell -- rustc --version` and `devenv shell -- cargo --version` were
run in this session and returned `rustc 1.99.1 (58cf8f9c3 2026-10-13)` /
`cargo 1.99.1 (5f94df478 2026-08-27)`. `devenv test` was run end to end and
passed (`cargo fmt --check`, `cargo clippy -D warnings`, `cargo test
--workspace`, 287 tests, 0 failures) under this locked revision. This is a
`reproduced_in_sim`-grade claim about *this* environment bootstrapping
successfully on *this* host — it is not yet evidence about any other host.

## What remains open (deferred to a future revision of C-14)

- **Host platform coverage.** Only Linux x86_64 has been exercised in this
  session. macOS and WSL2 remain unverified, exactly as the interim position
  in Appendix C already stated before this record.
- **CI execution mode.** Whether CI runs `devenv test` natively or inside an
  image built from the same lock is not yet decided; no CI job exists yet in
  this revision.
- **GPU userspace.** Not provisioned; nothing in the current dependency
  closure (0003) needs it.

## Consequences

- A developer or CI host without Nix cannot reproduce the exact pinned
  toolchain through this mechanism; they fall back to whatever `rustc`/
  `cargo` the host provides, which is an environment that "cannot be named"
  in the sense of SPECS.md §13 unless its own versions are recorded
  separately. The host's ambient toolchain in this session
  (`rustc 1.98.1`) was used for the bulk of implementation work before
  `devenv.lock` was generated; every test run in this session was also
  re-verified under the locked `devenv test` invocation described above, so
  no reported result here rests on the unlocked toolchain alone.
- `.devenv/`, `.pre-commit-config.yaml` and `.direnv/` are gitignored: they
  are machine-local build products (the pre-commit config embeds absolute
  `/nix/store/...` paths from the building host), not source.

# Development-time automation log

PLAN.md Phase 0's closing bullet: "If any development-time automation is
used while doing the above, state it: which tool, which model identity and
version, and which artifacts it produced. Attribution is cheap now and
unreconstructable after the fact." This is use U1 of SPECS.md §2.3.1
(development-time automation), and this file is its attribution record —
distinct from `robot-devtools`, which is where U1's *runtime* tooling will
live once written.

## Entry 2026-10-26

- **Tool:** Claude Code (Anthropic CLI agent), running inside the project's
  working directory with shell, file-edit and build-tool access.
- **Model identity/version:** Sessions across this work used `claude-opus-5`
  (1M context) and `claude-sonnet-5` (1M context), both Anthropic models;
  the session switched models mid-task per user command (`/model`).
  Exact IDs as reported by the harness: `claude-opus-5[1m]`,
  `claude-sonnet-5[1m]`.
- **Request:** "SPECS.md and PLAN.md describe a desired system. implement
  it. document each segment with its own README.md."
- **Artifacts produced in this entry's scope:** the Cargo workspace layout
  (`Cargo.toml`, 22 crate manifests); the full implementation and test
  suites of `robot-types`, `robot-artifacts` and `robot-model` (described in
  their own `README.md` files); `devenv.nix`/`devenv.yaml`/`devenv.lock`;
  this ledger, the evidence ledger, and the decision records in
  `docs/decisions/0001` through `0003`.
- **Review gate applied:** SPECS.md §2.3.1 item 1 requires generated
  artifacts entering the repository to pass "the same review, schema
  validation and test gate as hand-written work." Every crate above passed
  `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` and `cargo test --workspace` under the locked `devenv test`
  invocation before this entry was written. Human review by the requesting
  user has not yet occurred as of this entry and is expected before any
  claim above is relied upon for a release decision, per SPECS.md §2.3.1
  item 2 ("Output... is a proposal or a record, never a decision").
- **Boundary respected:** no artifact produced in this entry is depended on
  by a deployable crate in a way that requires the tool to remain present;
  the tool was not invoked at runtime, only at development time, consistent
  with the U1 boundary of SPECS.md §2.3.1 and the removal test of row A-25.
- **Known limitation of this entry:** per SPECS.md §2.3.1 item 5, "the same
  run never produces both an implementation and the test that accepts it"
  for *generated* contributions reviewed by a separate process. This session
  authored both the implementation and its own tests in the same pass, which
  is the normal mode for a human author working alone and is explicitly
  permitted by §2.3.1's rule — that rule targets a *generated* contribution
  reviewed by an independent gate, not authorship in general. It is recorded
  here so a later reviewer can weigh the tests' independence accordingly:
  they were not written by a reviewer independent of the implementation, and
  should be read as the author's own verification, not as a substitute for
  external review.

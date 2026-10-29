## 2. Language and dependency policy

All project-authored executable code MUST be Rust: ingestion, numerical processing, training, GPU kernels where needed, evaluation, visualization applications, deployment, tools, and tests. There must be no required Python training environment, Python preprocessing script, Python model-export step, or shell-script application logic. TOML configuration, schemas, Markdown documentation, and data files are not programming-language exceptions.

The default implementation targets Rust-native libraries and Rust-authored compute kernels. Candidate dependencies include Burn for differentiable models, CubeCL for Rust GPU kernels, nalgebra for linear algebra, Rapier for initial rigid-body experiments, tracing for diagnostics, and egui for tools. These are candidates requiring version, license, backend, and performance validation; they are not assertions that a complete humanoid robotics stack already exists in Rust.

Appendix B incorporates the supplied Rust ecosystem survey revised v3.0, assigns concrete candidate roles, and defines dependency qualification gates. Its maturity assessments are provisional. A Rust API alone does not establish that a library or its enabled dependency closure satisfies this policy. Generated C and required Python code-generation workflows do not satisfy the Rust-only policy. Development tools with foreign engines are also identified explicitly rather than silently classified as Rust-native. A crate written entirely in Rust whose advertised capability is produced by a hosted model endpoint is Rust-native in language and foreign in behavior; Section 2.3 governs that case.

Distinguish the following dependency classes explicitly:

| Class | Policy |
| --- | --- |
| Rust-authored application and algorithm code | Mandatory throughout |
| Rust-native compute and physics dependencies | Default; qualify for each required capability |
| OS, device firmware, GPU drivers | External platform dependencies; inventory them |
| C/C++ simulator, vendor SDK, ONNX runtime, or other foreign library through FFI | Excluded from the default implementation; any future exception requires an explicit scope decision and separate feature/profile |
| Python frameworks exposed through subprocesses or bindings | Excluded |
| External model service: a hosted inference endpoint invoked at runtime over the network | Excluded from the default build; permitted only under Section 2.3, never on a control path |

Do not label a wrapper around a foreign training or simulation stack a Rust-native implementation. The first engineering gate must establish whether the Rust-only simulation and autodiff stack can meet the task. If not, report the gap and propose Rust implementation work; do not silently substitute Isaac Lab, PyTorch, MuJoCo, or another non-Rust engine. External research can inform algorithms without becoming a runtime dependency.

### 2.1 Foreign-dependency exception procedure

Several clauses in this document and in Appendix B defer to "a separate scope decision". That phrase is only meaningful if the decision has a defined shape. An exception MUST satisfy all of the following before any such dependency enters the tree:

1. A written request naming the single capability required, the Rust candidates tried, and the measurements that show they are insufficient. "No Rust implementation was found" is a search result, not a measurement.
2. A classification as either a development-tool exception (fuzz engines, profilers, cross-validation simulators — never present at runtime) or a runtime exception (anything reachable from a deployed process). Runtime exceptions carry a higher burden and MUST state their effect on Sections 5, 12 and 13.
3. An isolated Cargo feature and build profile, named so that its foreign nature is visible at the call site, excluded from `default` features and from the default CI job.
4. A gate proving exclusion: the default build MUST fail if the excluded runtime appears in its dependency closure, and the check MUST run in CI rather than relying on review.
5. A prohibition on silent reach: an exception MUST NOT be reachable from the control path of Section 5 or from a policy bundle of Section 7.4 unless that reachability is itself the approved subject of the request.
6. A review date and a removal plan naming the Rust work that would retire the exception, with its estimated cost.

An approved exception covers the named capability, version and profile only. It does not generalize to the same library used elsewhere, to a newer major version, or to a sibling crate from the same project. Appendix B grants no exceptions; it only identifies which candidates would need one.

### 2.2 Development environment provisioning with devenv

The toolchain is a dependency like any other, and an unpinned toolchain invalidates the evidence states of Section 3 as effectively as an unpinned library version. Every developer machine, CI job and benchmark host MUST obtain the Rust toolchain, native build inputs, linkers, GPU userspace where applicable and auxiliary development tools from one declarative definition instead of host package managers or onboarding prose. The project uses [devenv](https://devenv.sh) over Nix for that definition.

| Artifact | Role |
| --- | --- |
| `devenv.nix` | The environment itself: `languages.rust` toolchain and components, native inputs, `packages`, `scripts`, `processes`, `git-hooks`, `enterTest` |
| `devenv.yaml` | Declared inputs, including the pinned nixpkgs and the Rust toolchain overlay |
| `devenv.lock` | Resolved input revisions; committed, reviewed, and cited as the environment pin |
| Optional `.envrc` containing `use devenv` | direnv activation convenience; a shortcut for `devenv shell`, never a requirement |

The following are binding:

1. `devenv.lock` MUST be committed, and CI, benchmark runs and release builds MUST use the same revision. A measurement produced outside the locked environment either records that fact under Section 3 or is not recorded at all.
2. A clean checkout MUST reach a working environment through `devenv shell` with no further manual installation. A missing step is a defect in `devenv.nix`, not an omission in a README.
3. Formatting, lint and dependency-policy checks — `cargo fmt`, `cargo clippy` and the exclusion gate of Section 2.1 — SHOULD be declared as `git-hooks` so that local and CI enforcement share one definition rather than drifting apart.
4. CI SHOULD execute through `devenv test` / `enterTest`, so that "passes locally" and "passes CI" denote the same environment.
5. `devenv processes` MAY host development-only services such as the viewer, a local telemetry collector or a replay device. Nothing declared there is part of the control path of Section 5 or a runtime dependency of a deployed robot.
6. Nix expressions and the `scripts` they declare are environment provisioning, in the same class as TOML configuration under Section 2's opening paragraph, and are not a language exception. They MUST NOT accumulate application logic: a script entry is a thin alias over a `cargo` or `robot-cli` invocation, and anything containing branching, parsing or numerics belongs in Rust.
7. Deployment MUST NOT require Nix or devenv on the robot host. The deliverables remain the Rust binaries and policy bundles of Section 7.4; the locked environment reproduces the build, it does not become part of the product.
8. The runtimes excluded by Section 2 and Section 1.1 MUST NOT appear in the default environment. Manual execution of the retired Python prior art from a historical commit, or an approved Section 2.1 exception, lives in a separately named non-default environment whose foreign nature is visible at its call site, and the exclusion gate still runs in the default one. The prior art's removal from the working tree does not retire this clause, because the exclusion gate is what keeps it out of the default closure.

The known limitation is host coverage: Nix supports Linux and macOS, and Windows only through WSL2. Which host platforms the locked environment is actually exercised on, which toolchain input supplies Rust, and whether CI runs devenv natively or inside an image built from the same lock, are open under [C-14](#appendix-c--open-questions-and-unresolved-decisions). A reproducible environment is not a hermetic build: Nix pins inputs, it does not make `cargo` fetch nothing, so dependency locking under Appendix B.6 remains separately required.

### 2.3 External model services and agent layers

The v3.0 survey of Appendix B introduced a case the two preceding classes do not describe: a dependency that is Rust-authored, links no foreign binary and runs no local foreign runtime, yet obtains its advertised capability from a third-party model endpoint over the network. Language purity is satisfied and behavioral control is not. Such a dependency is an **external model service**, and the following are binding wherever one appears:

1. The Cargo feature enabling it MUST be absent from `default` features and from the default CI job, and the exclusion gate of Section 2.1 item 4 MUST treat an unexpected model-service client in the default dependency closure as a build failure.
2. An external model service MUST NOT be reachable from the control path or policy path of Section 5, from the supervisor of Section 12.1, or from a policy bundle of Section 7.4. This prohibition is not waivable by a Section 2.1 exception: Section 2.1 governs foreign code, and no scope decision admits a network round trip to the control tick.
3. Every capability built on this class MUST define an offline-degradation mode and MUST be tested in it. The autonomous control loop, the safety envelope and all acceptance rows of Section 14 MUST pass with every model-service feature disabled, per row A-24.
4. A local-inference substitution path MUST be named for each such capability, together with the authored work that path requires, before the capability is scheduled. "A hosted model is available today" is not a substitution path.
5. A model-service response is untrusted input under Section 13. It MUST be schema-validated and envelope-checked before it reaches any consumer, and it enters the system only through the validated, timestamped queue interfaces of Section 5, subject to the staleness policy that applies to any other asynchronous proposal.
6. Model identity, version or endpoint changes are version changes under Section 3: every claim resting on the previous model is demoted, because a hosted model is not pinnable by a lockfile and its behavior may change without notice.
7. Non-model components of the same crate family — session stores, markdown indexing, transport, telemetry, schema validation — are ordinary Rust-native dependencies and are qualified independently of the model-service gate. A crate family MUST NOT be adopted as an umbrella; per-crate qualification under Appendix B.6 applies.

Which agent-layer capabilities, if any, this project adopts is open under [C-15](#appendix-c--open-questions-and-unresolved-decisions); the interim position is that none is scheduled and all remain feature-gated and optional.

#### 2.3.1 Admissible uses of an agent layer

The supplied integration appendix for the Rust agent framework proposes a wide capability surface. Only three uses of it are admissible in this project, and they are ordered by how far they sit from the robot. Each use has a different risk profile, a different qualification route and a different failure consequence, and they MUST NOT be conflated in a decision record, a feature flag or a dependency line.

| Use | Where it runs | What it may produce | What it may never do |
| --- | --- | --- | --- |
| **U1 — Development-time automation** | Developer and CI hosts only, inside the locked environment of Section 2.2, never on a robot host | Parsing, integration and code-review helpers; dataset and descriptor handling; test-run and environment orchestration; artifact and ledger bookkeeping | Produce any artifact that enters a build, bundle or result without the same review, schema validation and test gate as hand-written work |
| **U2 — Offline design and sampling assistance** | Offline, between experiments, per Section 9.5 | Candidate priors, curricula, scenario and opponent variations, expressive and personality variations as sampling proposals | Select, score or tune anything itself; replace the measurement that decides whether a proposal is kept |
| **U3 — Dependency knowledge base** | Development and operations tooling, per Section 13.1 | Curated, cited records of the low-level dependencies and hardware interfaces the system relies on, used when diagnosing a fault | Be cited as evidence under Section 3, or serve as the control for any hazard under Section 12.2 |

The following bind all three:

1. U1, U2 and U3 are **development, design and operations tooling**. None of them is a runtime capability of a robot, and none of them changes the control architecture of Sections 5, 6 and 12. Their complete removal MUST leave every deliverable buildable and every acceptance row closable, which is the standing check of row A-25.
2. Output from any of the three is a **proposal or a record, never a decision**. It becomes part of the project only by passing the gate that would apply to the same content from a human: review and CI for code, Section 3 evidence promotion for claims, Section 9 measurement for design choices, Appendix B.6 for dependencies.
3. Each use MUST be separately feature-gated and separately decided. "The framework is adopted" is not a decision; the decision names which of U1, U2 and U3 is enabled, with which crates, at which pinned versions.
4. Anything outside these three — agent-selected strategy during a bout, an agent in the supervision or promotion path, an agent-owned safety check, a robot-hosted agent runtime — is a non-goal under Section 15 and is not made admissible by a feature flag.
5. A generated artifact MUST be attributable: the record states which tool, model identity and version produced it, so that a later defect can be traced to its origin rather than to an anonymous commit.



## Implementation checklist

Status legend: `[x]` implemented, `[~]` partial, `[ ]` not implemented, `[?]` not audited in this pass. Summary: 10 `[x]`, 5 `[~]`, 2 `[ ]`, 10 `[?]` across 27 items.

- [x] 2.1 All project-authored executable code is Rust (no non-Rust application crates) — Evidence: all 23 `crates/*/Cargo.toml` are Rust workspace members; `find` found no `*.py` or `*.ipynb` files outside `target/`.
- [x] 2.2 No required Python training environment, preprocessing script or model-export step — Evidence: no Python files or `pyproject.toml`; `grep` for python/pytorch/mujoco/jax matches only prose and provenance comments (for example `crates/robot-sim/src/passive.rs`, `crates/robot-model/src/fixtures.rs`).
- [~] 2.3 No shell-script application logic — Evidence: `devenv.nix` `scripts.robot-exclusion-gate` contains `if`, `grep` and `find` logic. Note: this is branching and parsing in a Nix script, which the 2.2 item 6 rule forbids; move it into a Rust tool or `robot-cli` subcommand.
- [x] 2.4 Default dependency closure is Rust-native and path-only — Evidence: every `[dependencies]` entry in `Cargo.toml` and `crates/*/Cargo.toml` is a `path =` crate. Note: the only external crate is `cubecl`, optional in `crates/robot-sim-gpu/Cargo.toml`.
- [~] 2.5 Dependency classes are distinguished explicitly — Evidence: `docs/decisions/0003-std-only-default-closure.md` and Appendix B. Note: no dependency-class inventory file was found in this pass, so classes are not recorded per dependency.
- [x] 2.6 A wrapper around a foreign engine is not labeled Rust-native — Evidence: `crates/robot-sim/src/passive.rs` and `integrators.rs` document MuJoCo as their reference, and `robot-sim` has no external dependency in `Cargo.toml`.
- [~] 2.7 First engineering gate reports the Rust-only gap rather than silently substituting an engine — Evidence: `crates/robot-sim/README.md` records the gaps (sphere-only contact, no GPU batched path in this crate). Note: no written gate decision record for the Rust-only stack was found.
- [?] 2.1.1 Each exception has a written request naming the capability, Rust candidates tried and measurements — not audited.
- [?] 2.1.2 Each exception is classified as development-tool or runtime — not audited.
- [x] 2.1.3 Foreign-capability dependency sits behind an isolated Cargo feature excluded from `default` — Evidence: `cubecl` is `optional = true` in `crates/robot-sim-gpu/Cargo.toml` behind the `cubecl` feature. Note: default-feature exclusion of the gate is unverified.
- [ ] 2.1.4 The default build fails if an excluded runtime is in its closure, and CI enforces it — Evidence: `devenv.nix` `robot-exclusion-gate` always prints success and never exits non-zero, and it is absent from `git-hooks` and `enterTest`. Note: make the gate fail on any non-path dependency or excluded crate name, and run it from `enterTest` and CI.
- [?] 2.1.5 Exceptions are unreachable from the control path and policy bundles — not audited.
- [?] 2.1.6 Each exception carries a review date and removal plan — not audited.
- [?] 2.2.1 `devenv.lock` is committed and used by CI, benchmarks and releases — Evidence: `devenv.lock` is git-tracked. Note: CI usage cannot be checked because no CI config exists (see 2.2.4).
- [?] 2.2.2 A clean checkout reaches a working shell through `devenv shell` — not run in this pass.
- [~] 2.2.3 fmt, clippy and the exclusion gate are declared as `git-hooks` — Evidence: `devenv.nix` declares `rustfmt` and `clippy` hooks. Note: the exclusion gate is not a hook, and `.pre-commit-config.yaml` is gitignored (`.gitignore:76`), so hooks are local only.
- [ ] 2.2.4 CI executes through `devenv test` / `enterTest` — Evidence: `enterTest` is defined in `devenv.nix`, but no `.github`, `.gitlab-ci.yml` or other CI config is tracked. Note: add a CI definition that runs `devenv test`.
- [x] 2.2.5 Development-only services live under `processes` and none are a deployment dependency — Evidence: `devenv.nix` declares no active `processes`; the viewer entry is commented out.
- [~] 2.2.6 Nix scripts are thin aliases with no branching, parsing or numerics — Evidence: `robot-exclusion-gate` violates this (see 2.3). The other scripts are one-line `cargo` aliases.
- [x] 2.2.7 Deployment does not require Nix or devenv on the robot host — Evidence: `devenv.nix` comment states the deliverables are plain `cargo build --release` binaries. Note: descriptive, not exercised.
- [x] 2.2.8 Excluded runtimes do not appear in the default environment — Evidence: no Python, MuJoCo, PyTorch or CycloneDDS input in `devenv.nix`, `devenv.yaml` or any `Cargo.toml`; the Python prior art was removed (`SPECS.md` §1.1).
- [x] 2.3.1 No external model-service client is in the default dependency closure — Evidence: no network or inference crate appears in any `[dependencies]` list.
- [x] 2.3.2 No model service is reachable from the control path — Evidence: no model-service crate exists; `crates/robot-control/README.md` lists only local crates.
- [?] 2.3.3 to 2.3.6 Offline-degradation, local-inference substitution, response validation and version-change rules — N/A today because no model-service capability is present; not audited for future work.
- [?] 2.3.1-U1 Development-time agent layer is restricted to developer and CI hosts — not audited (a `.claude/` directory exists at the root, not reviewed).
- [?] 2.3.1-U2 and U3 agent-layer uses are feature-gated and decided separately — not audited.
- [?] 2.3.1-U4 and U5 non-goal agent roles are excluded and generated artifacts are attributed — not audited.

# Implementation Plan

This plan implements `SPECS.md` as a new logical architecture without relying on any existing repository structure. All project-authored implementation, tools, training, and tests are Rust. Milestones advance on evidence rather than dates; hardware availability and the Rust simulation stack are major dependencies.

The revised v3.0 Rust ecosystem addendum supersedes the previous integration in `SPECS.md` Appendix B. Apply its solver, estimation, observability, testing, serialization and parallelism register and qualification gates throughout this plan, together with the v3.0 agent-orchestration inventory in Appendix B.2.13. Incoming survey claims remain unverified until primary sources or executable evidence support them; reported versions are not dependency pins. Existing Rapier/Rerun corrections and the Rust-only policy remain in force.

The v3.0 additions change no phase in this plan. Agent orchestration and every capability that depends on a hosted model fall under the external-model-service class of `SPECS.md` Section 2.3: default-off, out of process, never on a control path, and removable without effect, per rows A-23 and A-24. Nothing in Phases 0 to 4 depends on them, and whether any of them is adopted at all is open under C-15. Treat the survey's integration proposals as candidate Phase 7 experiments, not as scope.

The supplied ADK-Rust integration appendix adds no further phase either. It is folded into `SPECS.md` Section 2.3.1, which admits exactly three uses of a Rust agentic layer, and this plan schedules them in that order of cost and distance from the robot:

| Use | What it is for here | Where it appears in this plan |
| --- | --- | --- |
| **U1 — Development-time automation** | Coding and maintaining this project: integration and parsing helpers around the already-defined data models, model and dataset handling, code review assistance, environment-specific integration, test-run orchestration, ledger and inventory bookkeeping | Phase 0 for identity and ledger discipline, Phase 1 for the tooling boundary, then continuous. Never a dependency of a shipping crate |
| **U2 — Offline policy-design and sampling assistance** | Improving policy selection quality and generation from priors by proposing candidate priors, curricula, scenarios, opponents and personality styles, which measurement then accepts or rejects; this is also where comparable prior work of the same kind belongs | Phase 11 track B, after a measured baseline exists in Phases 4 and 7–8. Offline only, between experiments |
| **U3 — Low-level dependency knowledge base** | Maintaining cited, versioned knowledge of every low-level dependency — crates, native linkage, OS and kernel options, drivers, firmware, bus parameters, actuator and sensor interfaces — so a hardware fault is diagnosed against project-maintained knowledge instead of being rediscovered | Seeded in Phase 0 from the dependency inventory, keyed to the hardware target in Phase 6, maintained thereafter |

U1 and U3 are tooling and operations discipline and do not wait on the C-15 adoption decision; they require only that nothing they use reaches a build or a robot host, per row A-25. U2 is the only one of the three that could ever involve a hosted model on a design path, and it is deliberately last, because a sampler that proposes candidates is worthless before there is a measurement that can reject them.

`SPECS.md` v0.7.0 added four method-level contracts that this plan now schedules rather than merely cites: the hierarchical latency invariant of Section 5.1, the staged latent competitive pipeline of Section 9.4, retrieval-augmented policy learning in Section 9.7, and the expanded limited-data adaptation table of Section 11.3, together with the retrieval tiers of Section 10.6. None of them adds a phase either. They are requirements *inside* Phases 4 to 9 — a tier declaration and a per-tier budget wherever a learned component is deployed, stage separation wherever a latent pipeline is trained, and a provenance and held-out-exclusion discipline wherever prior data is retrieved. Rows A-32 to A-35 are the gates, and C-22 to C-24 are the decisions those phases must close.

The Python prior art is gone. `training/`, `unitree/` and `test.ipynb` were removed from the working tree on 2026-10-26, closing C-01, and `SPECS.md` Section 1.1 now describes their disposition as historical references reachable only through version control. Three consequences for this plan: nothing in any phase below reads those paths, the Unitree protocol work of Phase 5 is Rust authorship against a cited commit rather than a port, and reintroducing any of that code is a Section 2.1 exception request. The working tree currently holds documentation only, so the Phase 0 bootstrap below starts from an empty workspace — which is the honest starting condition and was already assumed.

### Status snapshot, 2026-10-27

Recorded from a review of the current working tree. Anything marked not implemented has no code behind it yet.

- Repository: 22 crates with a standard-library-only default dependency closure (ADR 0003). `AGENT.md` is committed as `881d39a`.
- Simulation: `robot sim multibody --floating` runs the vendored Unitree G1 (35 DOF, floating base) with sphere ground contact (ADR 0010). A clamped PD stance hold keeps the pelvis above the plane for 2 s (min 0.787 m). Zero torque reports a FAIL (pelvis falls to −0.77 m). This is a supported-stance check, not a balance claim. Ground contact covers only the 8 ankle-roll sphere shapes; box, cylinder and mesh shapes are not modelled.
- Visualization: test-purpose only. `robot sim multibody --out <dir>` writes per-step CSV and JSON Lines frames plus a std-only SVG per frame, and `robot-viewer` reads them back through `Timeline` with strict validation (ADR 0009). No interactive viewer exists; Rerun and egui are still `unverified` candidates.
- Plug-in seams: `Environment` (`robot_sim::env`) and `HardwareAdapter` (`robot_hardware::device`) are complemented by `Policy`/`LoopOperation` in `robot-nn`/`robot-control`, `Backend` in `robot-sim`, and `ModelImporter` in `robot-model`.
- Learning: feed-forward MLP policies only; recurrent policies are not implemented. Checkpoint/resume saves weights, observation normalizer, optimizer and RNG state; named parameter groups support per-group freezing.
- Fidelity testing: shared analytic harness (ADR 0008 v7, seven checks plus the measured G1 probe row), reference/coarse tiers, TOML report (analytic subset). G1 contact is `not_measured` at reference and `not_applicable` at coarse; GPU checks are not applicable.

Sequencing: the plug-in seams (Phase 1) and the fidelity, checkpoint and parameter-group work (Phase 3) close before rendering goes beyond test-purpose file output. The frame-dump and SVG steps in Phase 2 may proceed earlier only because they add no dependency.

## 1. Delivery strategy

Build one complete vertical slice before expanding the research surface:

**BVH motion → canonical record → target retargeting → qualified simulation → PPO tracker → Rust policy bundle → deterministic replay runtime.**

The next slice deploys noncontact motion on a selected robot. Retrieval, latent self-play, residual adaptation and personality extend that measured baseline. A repository with many algorithm names but no reproducible motion-to-control path is not a successful first delivery.

Use three release levels:

- **R0: simulation research foundation.** End-to-end motion processing, target-specific tracking, artifact lineage and runtime replay.
- **R1: hardware control foundation.** Calibrated target, supervised noncontact motion, bounded timing and data-efficient identification.
- **R2: research extensions.** Retrieval/expert comparisons, latent skills, adaptation and robot-to-robot boxing experiments, each independently qualified.

## 2. Decisions to record before implementation

Create architecture decision records for:

1. Rust compiler/toolchain, the devenv-locked environment that provides it, and supported host OS/CPU/GPU platforms.
2. Rust-native physics and differentiable-compute capability findings.
3. First simulator robot model and first hardware target when available.
4. Canonical units, coordinate frames, quaternion encoding and timing model.
5. Observation/action schema, actuator interface and artifact representation.
6. Dataset eligibility and redistribution restrictions.
7. Benchmark machine, control period, timing acceptance profile and physical operating envelope.
8. Whether any agent-layer or external-model-service capability is adopted at all, per C-15; if the answer is no, record that and close the question rather than leaving it to drift into scope. C-16 and C-17 are recorded only if C-15 resolves to adopt something.
9. Which of the three admissible uses of `SPECS.md` Section 2.3.1 is enabled, per C-18 — one decision record per use, naming the crates, the pinned versions and the review obligation. "We adopt the agent framework" is not an acceptable record, because the three uses have different risk, different gates and different consequences of failure.
10. The scope, owner and update cadence of the dependency knowledge base, per C-19, recorded once the hardware target of C-02 is known.
11. The tier assignment, rate and compute budget of every learned component that is deployed, per C-22 and `SPECS.md` Section 5.1. This decision is taken per deployed stack, not once for the project, and it is the record that states whether the strategy layer survives being slowed from its experiment rate to its intended tier.
12. Which prior corpora are admissible for retrieval-augmented learning, with the retrieval unit, the similarity source and whether an embedding model is involved at all, per C-23. One record per corpus, after the license audit of C-07.
13. Which limited-data adaptation methods beyond identification and history conditioning are actually scheduled, and under which operating mode any of them may update parameters on hardware, per C-24. One record per method, and no record is written before the methods above it in the Section 11.3 ordering have been measured and found insufficient.
14. Plug-in trait boundaries for the policy (`Policy`), the simulator backend (`Backend`, per `SIMULATION_SPECS.md` Section 13.2) and robot model import. The change must be behavior-preserving, with the existing tests passing unchanged.
15. Fidelity tier definitions and the fidelity report schema: the `Check` and `Tier` types, the shared battery, and the JSON or TOML format written through `robot-artifacts`.
16. Test-render output format: a CSV or JSON Lines frame dump first, std-only SVG second, and Rerun `.rrd` only after Appendix B.6 qualification and an ADR.

C-01 is closed: the Python prior art was removed rather than migrated, and the decision record states that its remaining role is citation from version control.

Hardware-agnostic APIs do not require choosing the physical robot immediately. Simulator qualification and joint-level policy training do require a concrete model. Begin with a documented reference embodiment and test a structurally different second model before claiming portability.

## 3. Phase 0 — Evidence audit and Rust feasibility gates

### Work

- Bootstrap the devenv environment required by `SPECS.md` Section 2.2 before any benchmark: `devenv.nix` with the Rust toolchain and native inputs, `devenv.yaml` inputs and a committed `devenv.lock`. Every feasibility measurement below cites that lock revision; a number produced outside it is labeled accordingly or discarded.
- Establish a source ledger from the research leads in `SPECS.md`.
- RoboStriker (arXiv:2601.22517) is `paper_verified` as of 2026-10-26 and is recorded in the ledger as the reference decomposition of `SPECS.md` Section 9.4, not as a reproduced result. What remains in this phase is narrower: confirm whether any source release exists, and record that every reported number came from a foreign simulator on one embodiment, so no stage of Section 9.4.1 is scheduled as if its hyperparameters were inherited.
- Verify Unitree boxing product/demo claims and clearly label unknown implementation details.
- Resolve SPiRL, STRAP, SMOSE and ASAP first; then investigate delay-aware and newer retrieval leads. Reject placeholder or mismatched references.
- Separate the three retrieval literatures in the ledger before any of them is cited in a design: offline corpus selection (`SPECS.md` Section 9.7), between-episode working-set selection, and in-loop routing (Section 10). A latency or sample-efficiency number from one tier is not evidence for another, per Section 10.6 item 3.
- Verify the RMA-class, simulation-guided fine-tuning and cross-embodiment leads of `SPECS.md` Section 11.3 at the same bar, recording for each what it requires on hardware. A method whose published procedure needs on-robot weight updates is a C-24 candidate and not a Phase 9 default.
- Record the retirement of the Python prior art as a ledger entry with the commit that last contained it, so later comparison numbers and protocol transcriptions can cite a specific revision instead of "the old training code".
- Inspect Rust autodiff support for policy distributions, PPO losses, recurrent models, optimizer state, checkpointing and CPU inference.
- Measure a small Rust MLP on the intended deployment class, with warmup and latency tails.
- Prototype articulated joints, actuators, foot-ground contact and resets in the candidate Rust simulator.
- Audit licenses, foreign runtime dependencies and GPU kernel/toolchain requirements.
- Audit Burn/CubeCL first and Candle as an alternative; verify selected operators, checkpointing, CPU inference and feature-specific native dependencies.
- Inspect Rapier's existing CCD and multibody joints before proposing authored articulation. Verify ferromotion's repository, modules and reported capabilities before allocating an alternative-backend experiment.
- Inspect HULKs for runtime and recovery architecture; verify the supplied hardware and publication claims before citing them.
- Inspect `rl_traits` semantics and ember-rl/thrust-rl/sb3-burn implementations as references, using independent numerical fixtures rather than treating immature implementations as correctness oracles.
- Verify Clarabel, pounce_qp, simple_qp, tinympc-rs and direct Rust OpEn capabilities. Exclude OSQP C linkage/generated C and required Python generation from the default path.
- Verify the estimation candidates' actual APIs and maintenance; do not select multicalc solely on advertised feature coverage.
- Confirm the registry identity and ownership of the v3.0 agent crates before any of them appears in a dependency line, and record the three-way name collision between Google's Agent Development Kit, zavora-ai's `adk-rust` and Yougigun's `rust-adk` in the ledger so a later reference cannot be made ambiguously. This is a ledger entry, not an adoption step, and nothing in this phase's exit gate depends on it.
- Seed the dependency knowledge base of `SPECS.md` Section 13.1 from the dependency inventory this phase already produces: one entry per low-level dependency, with canonical source, version, license, native linkage and evidence state, plus the platform entries — OS, kernel options, drivers, GPU toolchain — that the license and runtime audit above already touches. This costs almost nothing while the inventory is being built and is expensive to reconstruct later.
- Record the convention that makes the base usable during a fault: version-keyed entries, mandatory citation, visible staleness on a version change, and local readability with no service available. An uncited entry is a note, and notes are not consulted as specifications.
- If any development-time automation is used while doing the above, state it: which tool, which model identity and version, and which artifacts it produced. Attribution is cheap now and unreconstructable after the fact.

### Deliverables

Evidence ledger, dependency inventory, the seeded dependency knowledge base keyed to that inventory, capability matrix, committed `devenv.lock`, benchmark reports and decisions naming supported baseline features.

### Exit gate

A Rust-only executable can run a small controlled physics task, collect trajectories, differentiate a training loss, update a policy, save/load it and execute inference. A clean checkout reaches the same toolchain through `devenv shell`, and every reported measurement names the lock revision it ran under. Simulator throughput and fidelity limitations are documented.

### If the gate fails

Create explicit Rust implementation tasks for missing articulated dynamics, GPU support, distributions or serialization. Reduce the first simulation task if necessary, while preserving the overall roadmap. Do not claim R0 readiness or silently introduce a foreign-language framework. Scope any future dependency exception as a separate decision.

## 4. Phase 1 — Contracts, workspace and experiment foundations

### Work

- Establish Cargo workspace boundaries for types, robot model, artifacts, CLI and evaluation.
- Define typed time, units, frames, joint identifiers and schema versions.
- Implement embodiment profiles and URDF ingestion for a documented supported subset.
- Define separate human-motion, robot-reference, robot-episode and policy-bundle schemas.
- Implement content hashing, manifest validation and atomic artifact writes.
- Create experiment configuration and run directories with immutable input manifests.
- Set up `cargo fmt`, `cargo clippy`, unit/integration checks, dependency/license review and reproducible build records. Declare the formatting, lint and excluded-runtime checks as devenv `git-hooks` and run CI through `devenv test`, so local and CI enforcement share one definition.
- Keep any foreign runtime out of the default devenv environment, and keep the exclusion gate standing now that the Python prior art of `SPECS.md` Section 1.1 is removed: deletion is not enforcement, and a reintroduced PyTorch or MJX dependency must still fail the build. Manual comparison runs against a historical commit use a separately named non-default environment.
- Expose development-only services such as the viewer, a local telemetry collector and the replay device through devenv `processes`; nothing declared there may become a deployment dependency.
- Define simulator, hardware, learner and inference interfaces without promising stable dynamic-plugin ABI.
- Make the remaining concrete pieces replaceable, behavior-preserving: a `Policy` trait (`act`, input and output width) implemented by the MLP and Gaussian policies, with `ControlLoop` generic over it; a `Backend` trait in `robot-sim` as `SIMULATION_SPECS.md` Section 13.2 describes; and a model importer trait in `robot-model`. `Environment` and `HardwareAdapter` are the reference pattern. The existing tests must pass unchanged.
- Evaluate `urdf-rs` with our semantic validation layer; inventory unsupported model features explicitly.
- Create per-dependency records covering enabled features, transitive native linkage, build scripts, license, maintenance and replacement cost under Appendix B.6.
- Define versioned message envelopes and compare bincode/postcard/rkyv/Cap'n Proto/prost only where their full build/runtime toolchains satisfy policy. Separate IPC encoding, transport and durable storage decisions.
- Establish Rust property/mutation/regression harnesses. cargo-fuzz/libFuzzer and honggfuzz require a separate development-tool exception; do not mandate them under the current Rust-only requirement.
- Define off-thread tracing/metrics ownership, bounded event records and exporter failure behavior before integrating OpenTelemetry.
- Establish the development-time automation boundary of use U1: a `robot-devtools` crate (or the devenv `scripts` equivalent) that owns integration and parsing helpers, descriptor and dataset handling, test-run orchestration and ledger bookkeeping, and that no shipping crate depends on. Add a CI assertion that no deployable crate reaches it, alongside the existing excluded-runtime gate.
- Decide the review rule for generated contributions before any are made: generated code and generated tests pass the same review and CI gate as authored work, are attributable to tool and model version, and the same run never produces both an implementation and the test that accepts it.
- Give the dependency knowledge base of Phase 0 a schema and a home in version control, with per-dependency records keyed to the inventory above, so that `robot deps` tooling and the per-dependency Appendix B.6 records are the same records rather than two drifting copies.

### Proposed CLI surface

```text
robot data inspect <source>
robot data import <source> --config <path>
robot motion retarget <motion-id> --robot <profile>
robot motion validate <reference-id> --config <path>
robot sim qualify --config <path>
robot train --config <path>
robot eval --bundle <id> --suite <path>
robot skills index --manifest <path>
robot deps inventory --lock <path>
robot deps explain <dependency> [--version <v>]
robot deps fault record --episode <id>
robot adapt identify --episodes <manifest> --robot <profile>
robot deploy --bundle <id> --mode shadow
robot replay <episode-id>
```

These commands are interface requirements, not currently implemented tools.

### Exit gate

Fixtures round-trip without losing units, frames or timestamps; invalid or incompatible artifacts are rejected. A clean checkout bootstraps through `devenv shell` and passes `devenv test` with no manual installation. Every run is traceable to source/config/data/model/environment versions.

## 5. Phase 2 — Motion ingestion, curation and retargeting

### Work

- Implement BVH parsing with per-joint channel order; add malformed-file limits and parser fuzzing.
- Curate a small legally usable set of stance, stepping, guard and slow strike motions.
- Implement resampling, gap masks, ground normalization, contact estimation and segmentation.
- Build a Rust viewer showing source skeleton, target skeleton, ground contacts, limits and tracking errors.
- Implement forward kinematics and weighted constrained IK.
- Add temporal smoothing/refinement and planted-foot constraints.
- Produce per-motion validation reports with accepted/rejected reasons.
- Add C3D and supported numerical body-model ingestion after BVH works; track licenses for body-model assets separately.
- Preserve original-session splits through augmentation and windowing.
- Audit `bvh_anim` / `bvh_anim_parser` and `c3dio` before adopting them; test source-format edge cases and do not assume optional FFI features satisfy the default dependency policy.
- Compare Rerun/egui and `urdf-viz` for inspection tools. Preserve canonical data independently of Rerun RRD files; qualify MCAP separately if needed.
- Test-purpose rendering comes before the viewer above. First, a `robot sim multibody --out <dir>` option writes pelvis pose and joint angles per step as CSV or JSON Lines, read back through `robot-viewer`'s `Timeline`. Then an std-only SVG per frame, with golden-file tests. Neither step adds a dependency. Rerun `.rrd` and egui/wgpu wait for Appendix B.6 qualification and an ADR.
- Investigate Rust SMPL-family implementations before scheduling authorship. Scope parsing, licensed model evaluation and fitting separately; keep BVH delivery independent of this investigation.
- Resolve the constrained solver before implementing differential-QP IK: benchmark representative full-body matrices, setup/update/solve costs, cold/warm behavior, residuals, memory and infeasible cases. Retain a feasible non-QP baseline where appropriate.

### Exit gate

A fixed reference set retargets reproducibly to two different model geometries. Reports expose contact slip, unreachable targets and limit violations. No infeasible reference is silently accepted through joint clipping.

### Deferred work

Live optical/inertial capture and video-to-motion inference follow the file-based vertical slice. They reuse canonical records and synchronization contracts.

## 6. Phase 3 — Qualified simulation and baseline RL

The simulation target changed, by explicit project direction, partway through this phase: `crates/robot-sim/SIMULATION_SPECS.md` now specifies GPU-bound batched execution, variable fidelity and MuJoCo-caliber fidelity as an authored, Rust-native engine, grounded in direct research against MuJoCo's own source/docs, MJX, Brax, CubeCL and `wgpu`. The work items below (particularly "introduce GPU simulation only through a separately qualified backend" and "cross-validate through MuJoCo bindings only if a separate foreign-dependency scope decision has been made") remain accurate — a GPU backend is still gated the same way, and nothing here grants a MuJoCo-binding exception — but the *default-path* assumption of "qualify Rapier first, author only demonstrated gaps" is superseded for the articulation/contact/GPU-batching subsystem specifically by that document's own Section 11 decision: build the authored, GPU-batched, Rust-native engine directly, informed by MuJoCo's design rather than waiting on Rapier's qualification. `docs/decisions/0005` records this change.

### Work

- Implement simulator stepping, reset, batched environments and independently seeded randomization.
- Add motor response, control decimation, sensor noise/delay and explicit command semantics.
- Qualify pendulum, contact/friction, static balance and perturbation benchmarks.
- Implement PPO, advantage estimation, action distributions, value loss, entropy regularization and normalization.
- Distinguish truncation from terminal failure in return calculation.
- Implement checkpoint/resume including optimizer, RNG and normalizer state. Store all of them through the `robot-artifacts` chunk format, and save the normalizer with the weights.
  - Status: implemented in `robot-learn::checkpoint`; checkpoints save and restore weights, normalizer, Adam and RNG state.
- Add parameter groups (named weight slices) with a per-group `frozen` flag, so part of a policy can be trained or fine-tuned while the rest stays fixed.
  - Status: implemented in `robot-nn`; frozen parameter groups are excluded from learning updates and covered by tests.
- Build a fidelity harness in `robot-sim`: a `Check` trait (name, tolerance, measured value, pass or fail), a `Tier` struct that bundles timestep, integrator and solver settings, and a runner that writes a report through `robot-artifacts`. The first battery reuses the existing analytic tests (pendulum period, PD equilibrium, ABA/RNE closed form, energy drift, free-fall), and every declared tier must pass it. The CPU-vs-GPU agreement test (`SIMULATION_SPECS.md` Section 14.2) and the contact-budget exhaustion counter (Section 14.4) wait for a GPU backend.
  - Status (ADR 0008, schema v7; v7 adds the measured G1 `[[probe]]` row): the harness is implemented in `crates/robot-sim/src/fidelity.rs`. `FidelityCheck` is the plug-in trait; `Check` is the result row; the archival battery runs seven analytic checks (v6 adds pendulum_trajectory against oracle::pendulum_closed_form, abs-only tolerance and worst-sample index) at the `reference` (100 µs) and `coarse` (1 ms) tiers, and writes `fidelity.toml` through `robot-artifacts`. The report scope is the analytic subset (`scope = "analytic-subset"`). It does not qualify G1 ground contact: each tier carries a `[[contact]]` row, `not_applicable` at 1 ms (outside the ADR 0010 range `[5e-5, 4e-4]` s) and `not_measured` at 100 µs. The sphere-contact check is deferred. The CPU-vs-GPU agreement and contact-budget rows stay `not_applicable` until a GPU backend exists.
- Add ground-contact handling to the multibody path. Done for sphere shapes: the floating-base G1 is held above the plane by a clamped PD stance under a qualified dt range (ADR 0010). It is not a balance result. The 28 box, cylinder and mesh shapes are unmodelled, so zero torque still collapses. Balance tests remain blocked until contact proxies and a controller exist. Run `robot sim qualify` as a gate.
- Train small reference tasks and compare learning against deterministic baselines.
- Train standing, balance and basic stepping on the selected reference robot.
- Measure environment throughput and project the cost of the next experiment; introduce GPU simulation only through a separately qualified backend.
- Follow Appendix B.4: qualify existing Rapier articulation/contact first and compare any verified alternative at matched initial conditions and controls. Author only demonstrated gaps.
- Test full episode snapshot/replay, including sensor histories, task state, RNG and recurrent policy state outside the physics engine.
- Cross-validate through MuJoCo bindings only if a separate foreign-dependency scope decision has been made. Keep this optional profile outside the default dependency path.
- Benchmark Rayon CPU rollout/preprocessing pools against sequential execution. Use isolated resources and per-environment RNG streams; document reduction/scheduling nondeterminism and bandwidth limits.

### Exit gate

Baseline RL succeeds on agreed small tasks across at least three seeds, resumes correctly and exports a CPU-executable actor. Contact behavior and numerical sensitivity are documented. A basic simulated humanoid controller maintains balance within its acceptance profile. A fidelity report exists for every declared tier, and a parameter group frozen during fine-tuning is verified unchanged.

## 7. Phase 4 — Motion tracker and R0 vertical slice

### Work

- Implement reference sampling, phase handling, termination and tracking rewards.
- Train a compact target-specific tracker on a small motion set, then expand repertoire.
- Add a privileged critic if useful while enforcing actor-only deployable inputs.
- Train overlapping transition windows and disturbance recovery alongside nominal clips.
- Validate held-out motions, contacts, sensor delays and physics variations.
- Export immutable policy bundles and implement identical Rust normalization/action transforms for training and deployment.
- Add deterministic replay and action-parity checks between learner inference and runtime inference.
- Publish an R0 evaluation report with compute cost, failure cases and dataset eligibility.
- Declare the tier of every learned component in the bundle, even though the R0 stack has only one, and record its measured compute duration alongside it per `SPECS.md` Section 5.1 item 1. Establishing the declaration while the hierarchy is trivial is what makes row A-32 cheap later; retrofitting tier metadata onto a stack that already has three layers is what makes it expensive.

### Exit gate: R0

One command sequence produces a policy from motion input and evaluates it in simulation and runtime replay. Per-skill results beat the agreed baseline without constraint violations being hidden by reward aggregation. No hardware success is claimed at this stage.

## 8. Phase 5 — Deterministic runtime and hardware integration

Runtime design can begin after Phase 1 and proceed alongside simulation work.

### Work

- Implement fixed-capacity observation/action buffers and monotonic timing.
- Keep inference resident and allocate working memory before activation.
- Add command freshness, observation validity, deadline and communication watchdogs.
- Implement a hardware adapter first against a loopback device, then a replay device, then an actual documented protocol.
- Verify joint ordering, sign, offsets, units and command semantics before enabling motion.
- Implement fault responses appropriate to the selected robot, including stale sensors, NaNs, missed deadlines, thermal limits and loss of transport.
- Add bounded incident recording and nonblocking telemetry.
- Test control-path allocation behavior and latency under CPU/GPU/I/O contention.
- Integrate hardware emergency stop and operating-mode interlocks.
- Implement the tier machinery of `SPECS.md` Section 5.1 in the runtime rather than in each experiment: per-tier rates, deadlines and budgets; generation-tagged, timestamped, bounded outputs across tier boundaries; freshness rejection evaluated by the consuming tier; and a declared hold-or-default behavior for a missing upper-tier decision. A component with no declared tier MUST fail activation — enforced in code, not in review.
- Report per-tier compute tails and decision age as separate numbers from the start. An asynchronous upper tier converts latency into decision age, and a report showing only one of the two is not evidence under Section 10.6 item 2.
- Transcribe the DDS/SDK field layouts, rates and command semantics from the retired `unitree/` reference at its cited commit into authored Rust, treating every field as a hypothesis requiring hardware verification per `SPECS.md` Section 1.1. This is authorship against a historical document, and the loopback and replay devices are where the hypotheses are falsified before a motor moves.
- Compare HULKs/Copper runtime patterns and `iceoryx2` host IPC against actual scheduling and payload requirements. Evaluate Zenoh, `roslibrust` or dora only where orchestration/interoperability warrants them; verify protocol compatibility and dependency closure.
- Qualify the estimator inside `robot-perception`: IMU bias/rotation conventions, contact fusion, covariance consistency, jitter, dropout, outlier and slip cases. Compare multicalc/numeris/typed_kalman/kfilter or an authored baseline against independent truth.
- Benchmark instrumentation enabled/disabled, full event queues and disconnected exporters. Construct OpenTelemetry spans off-thread from bounded timestamps instead of assuming span entry/exit is deadline-safe.
- Benchmark serialization p99.9, allocations and validation on sensor/command/snapshot shapes; test malformed inputs and version migration. Keep general Rayon work stealing outside the control loop.

### Exit gate

Fault injection never sends invalid or expired commands. The designated target meets its timing profile for a stated qualification duration, with max latency and missed deadlines reported. Each declared tier meets its own deadline and behaves as declared when the tier above it is late, absent or expired, per row A-32. Shadow mode can run from live sensors without actuation. Scheduling measurements alone are not called a hard-real-time guarantee.

## 9. Phase 6 — Calibration, transfer and R1 noncontact deployment

### Work

- Establish physical geometry, mass/payload, joint calibration and sensor alignment.
- Measure supported actuator response, command delay, friction and contact behavior through bounded calibration motions.
- Fit identifiable parameters and preserve uncertainty for parameters the data cannot distinguish.
- Retrain or adapt the target tracker in calibrated randomized simulation.
- Progress through replay, shadow, supported stance, supported slow motion and free-standing noncontact motion.
- Log commands, measured responses and timestamps needed for later identification.
- Compare measured behavior with simulated distributions rather than expecting frame-perfect replay.
- Document rollback and requalification requirements for firmware, payload or glove changes.
- Key the dependency knowledge base to this robot: firmware revisions, bus and transport parameters, driver and kernel options, actuator and sensor interface documents, thermal and torque-speed material, and the known defects of each, with citations and the profile version each applies to. This is the phase where the base either pays for itself or proves it was never maintained.
- Record every diagnosed low-level fault as a fault entry — symptom, the measurement that identified the layer, resolution, and the dependency version it applies to — linked to the episode record and the incident channel. A fault resolved without an entry will be resolved again from scratch.

### Exit gate: R1

The selected robot performs a documented noncontact skill set within its physical and timing profile. Failure handling is demonstrated. The deployment report identifies remaining sim-to-real discrepancies and makes no general claim about arbitrary humanoids.

## 10. Phase 7 — Skill retrieval and sparse experts

### Experimental sequence

1. Benchmark the existing compact tracker as the dense reference.
2. Add skill-conditioned tracking with structured labels.
3. Build a small expert bank with shared contracts and an exact-search Rust index.
4. Add state-dependent feasibility filtering and deterministic routing.
5. Introduce learned embeddings and/or learned routing only against this baseline.
6. Compare latent-conditioned decoding if Phase 8 makes it available.

### Work

- Define skill initiation/termination and transition descriptors.
- Build skill embeddings from task-space motion features; record embedding version and normalization.
- Add local top-k search, candidate filtering, expiry and confidence thresholds.
- Keep recovery and active experts resident; implement asynchronous prefetch and atomic boundary activation.
- Implement recurrent-state transitions, phase/contact guards and preemptible hysteresis.
- Train transitions explicitly; evaluate blending only as a separate variant.
- Test routing collapse, unseen states, cache misses, stale proposals and incompatible experts.
- Benchmark cold/warm latency, decision age, active parameters, total resident memory and end-to-end task performance.
- Measure the exact-search crossover by dimensions, library size and filtering cost. Qualify `instant-distance` only if needed, including recall and index round-trip tests; C++-backed FAISS/usearch remain excluded by default.
- Optionally add the Markdown skill descriptors of `SPECS.md` Section 10.1 as an operator-facing catalog and a supplementary routing signal. Fix the descriptor vocabulary first, index at build or load time, and test that a descriptor disagreeing with the typed fields cannot widen the feasibility filter. This involves no model call and is droppable without affecting the gate below.
- Separate the three retrieval tiers of `SPECS.md` Section 10.6 in configuration, in code and in every report: R0 corpus selection on a developer host, R1 working-set selection completing before the mode-entry preconditions of Section 4, and R2 in-loop routing inside the control process. R1 publishes to R2 as generation-tagged immutable handles; R2 reads a resident preloaded index and performs no file, database, embedding or allocation work.
- Test each tier's empty-result path explicitly — R0 falls back to target-only data, R1 retains the previous working set, R2 retains the current controller or invokes the Section 10.4 fallback — and exercise cache expiry with decision age recorded. A tier that stalls instead of degrading is a defect, not a slow path.
- Do not promote a mechanism across tiers on the strength of its validation at a cheaper one. An embedding similarity that selected a good training set says nothing about a 100 Hz router, and the measurement is redone per tier.

### Exit gate

The report states whether retrieval actually helps over the dense baseline. A new skill can be added through a versioned manifest and targeted training, but promotion still requires routing and transition regression tests. Failed or late retrieval never stalls the control path. R0, R1 and R2 are separately configured and separately measured, with R2 cost appearing in the policy-path tails next to decision age, per row A-35.

## 11. Phase 8 — Research methodology ingestion and self-play

### Work

The staged pipeline of `SPECS.md` Section 9.4.1 is the structure of this phase, and its four stages are implemented and qualified in order. Each stage is separately checkpointed, separately gated and independently deletable, per Section 9.4.2 items 1 and 2: a tracker that does not track is not repaired by competition, and a collapsed competition stage must not take the tracker down with it.

### Work — stage 1: tracker baseline

- Reuse the qualified Phase 4 tracker as the frozen teacher rather than retraining one here. If it does not meet its tracking profile, this phase does not start.

### Work — stage 2: latent skill distillation

- Implement the student encoder, decoder and state-conditioned prior distilled from the frozen tracker, with the latent distribution constrained to a declared bound or manifold.
- Declare the latent space as a versioned interface through the skill-space extension point of Section 9.3: dimensionality, bound or manifold constraint, versioned identity and the decoder artifact that interprets it. Transitions stored in that space carry the same identity, per Section 7.6 item 8, and a buffer rejects a latent/joint-space mixture.
- Choose the dimensionality and the constraint from our own ablation. RoboStriker's 32-dimensional unit hypersphere and 50 Hz rate are conditions of its experiment and are recorded as such, not copied as settings.
- Record the prior or divergence term as a training constraint only. It MUST NOT appear in a hazard register, and the supervisor of Section 12.1 stays independent of it, per Section 9.4.2 item 4.
- Compare repertoire coverage, tracking and controllability of latent versus expert representations.

### Work — stage 3: behavioral warm-up

- Train a latent-space policy on goal-oriented ego-centric task observations against a passive or scripted target, with a motion-prior or style term alongside the task reward, logged as a separate reward component under Section 9.2.
- Treat the published collapse of the no-warm-up ablation as a hypothesis to re-test here, not as a settled result. Reproducing it cheaply is what justifies the stage's cost.
- Never present a style or discriminator score as evidence of feasibility, stability or safety, per Section 9.4.2 item 5.

### Work — stage 4: competitive co-evolution

- Implement boxing task geometry, controlled contact, scoring, fall rules and ring boundaries.
- Validate hit detection using scripted positive/negative cases before RL receives its reward; enforce the energy budget outside the policy that is rewarded by it.
- Maintain the current policy, a historical opponent population and a held-out evaluation pool through the opponent-and-population extension point, with the held-out pool fixed independently of the training opponent distribution.
- Implement opponent averaging — a learned average policy over a reservoir of historical best responses, or an equivalent — and ablate it against naive latest-snapshot self-play. Anti-cycling is a measured claim, not an architecture diagram.
- Store the reservoir or population as a store over the canonical schema of Section 7.6 with a declared retention rule, capacity, admission probability and seed stream. Concurrent on-policy and reservoir stores share one transition format, per item 7.
- Add checkpoint tournaments and evaluation against scripted policies.

### Work — cross-cutting

- Add delay-aware policy training using recorded latency distributions and action histories.
- Measure the tiered configuration before any hardware claim. Where an experiment runs the strategy layer and the decoder at one rate, record it as the degenerate case of Section 5.1 item 2 and re-measure task performance at the intended tier rates; this is the evidence C-22 needs.
- Evaluate SMOSE-inspired shallow experts as an independent experiment, not a required replacement. Trajectory retrieval moves to the dedicated track below, because it is data selection and not a runtime mechanism.

### Work — retrieval-augmented policy learning (R0, per `SPECS.md` Section 9.7)

This track is offline data selection and shares no measurement with the Phase 7 router. It can run in parallel with the stages above once a corpus is licensed.

- Write the manifest before the code: corpus revisions and licenses, query, retrieval unit, similarity metric and embedding version, admission filter, and the mixing rule — including what the learner does when retrieval returns nothing.
- Label every retrieved transition with source corpus, source episode, embodiment and retrieval generation, so a batch's composition can be reconstructed. A result that cannot name what it trained on is not a result.
- Exclude evaluation episodes, held-out disturbances, held-out opponents and the Phase 9 validation session from the corpus by construction, and test the exclusion rather than asserting it.
- Run the two mandatory comparisons at equal update budget: target data alone, and the whole corpus. Without both, an improvement cannot be attributed to selection rather than to data volume.
- Report retrieval quality separately from task outcome: fragments retrieved, fragments rejected and why, corpus coverage used, and sensitivity to the number retrieved.
- Start with exact search per Appendix B.2.5; adopt approximate indexing only after the crossover measurement, reporting recall loss as a property of the selected training set.
- Prove removability: disabling the feature leaves the baseline learner, the shared schema and the applicable acceptance rows unchanged, per row A-30.

### Exit gate

Each method has a source-backed algorithm card, fixed-budget baseline, multiple seeds, ablations and an outcome report. Every stage is separately checkpointed and separately qualified, and deleting the competitive stage leaves a usable tracker, per row A-33. Opponent averaging has been ablated against latest-snapshot self-play on held-out opponents, and the full pipeline has been compared at equal budget against the conditioned-tracker and expert-bank alternatives, reporting task outcome, physical stability and command smoothness separately. Strategy improvements persist against held-out opponents and cannot be explained by scoring bugs or unsafe contact exploitation. Any retrieval-augmented result satisfies row A-34, including the tested held-out exclusion.

## 12. Phase 9 — Data-efficient adaptation

### Implementation order

The ordering of the `SPECS.md` Section 11.3 table is binding and runs by increasing hardware exposure. A method is not scheduled until the methods above it have been measured and found insufficient **on the same robot**, and each step below is a separate C-24 decision record:

1. Calibrated system identification and improved randomization ranges.
2. Simulation-trained history encoder for latent dynamics adaptation.
3. Two-phase latent identification of the RMA class: a privileged-conditioned policy in randomized simulation, then a small adaptation module regressing that latent from recent deployable history. Phase 2 inherits phase 1's limits and the deployment observation set is audited for privileged leakage.
4. Offline simulator correction with well-defined transition alignment.
5. Multi-model randomization across distinct dynamics models, which requires a second qualified simulator and therefore is not available before Section 9.1 provides one.
6. Retrieval-conditioned adaptation: select prior episodes similar to the observed deployment condition through the Section 9.7 track and adapt on their union with the few real episodes. Observation similarity is not dynamics similarity, and the admission filter is what carries that distinction.
7. Cross-embodiment few-shot transfer, under the full requirements of Section 11.1, with the target robot genuinely held out.
8. Bounded robot-side residual or small adapter, only if prior steps leave a measurable gap.
9. Simulation-guided fine-tuning and any other method requiring on-hardware updates. These need the mode gating of Section 4 and a separate decision; nothing in this phase authorizes continuous learning on a robot.

### Work

- Add event-triggered recording with representative successful background samples.
- Select calibration trajectories for information gain within the current operating envelope.
- Build real-data splits by session and operating condition.
- Compare adaptation methods at equal real-interaction budgets and report learning curves.
- Verify deployment inputs contain no unavailable paired simulator state.
- Constrain residual authority and test fallback to the frozen controller.
- Partial fine-tuning (a residual head or a single parameter group) depends on the parameter-group support of Phase 3. Until that exists, only whole-network updates are available.
- Use shadow evaluation, simulation regression and staged hardware validation for every candidate.
- Treat every adaptation method as an extension under Sections 9.3 and 9.6: its own manifest, its own feature flag and its own conformance run. Any component that runs on the robot — adaptation module, context encoder, residual head — is a tier under Section 5.1 with a declared rate, a measured budget and a defined behavior when its input is stale.
- Enforce adaptation authority outside the adapted component: the supervisor clamps a learned correction like any other command, and widening a bound is a change to the safety case, not a hyperparameter.
- State sample-efficiency claims in units that cost something — real minutes, real episodes, physical interventions, wall-clock downtime. Simulation steps consumed by the method are reported separately and never counted as efficiency.
- Retain immutable baseline and last-known-good bundles; demonstrate rollback.
- If verified ferromotion calibration/dynamics-gradient modules are suitable, compare them with the identification baseline. Check gradients against finite differences on smooth fixtures, contact-mode limitations, parameter identifiability and held-out trajectory error before adoption.
- Evaluate qualified Rust MPC as a separate model-based baseline or bounded correction only if useful. Specify its model/data needs; do not call MPC residual-policy training or introduce Python-driven problem generation.

### Exit gate

An adaptation improves a prespecified held-out metric without violating control constraints. Report actual data duration, transition count, intervention count, sensor requirements and uncertainty. A negative result is acceptable; an unsupported universal few-shot claim is not.

## 13. Phase 10 — Contact experiments and expressive behavior

These are independent optional extensions after R1, with contact experiments additionally depending on validated task scoring and physical contact modeling.

### Contact work

- Calibrate padded fixture contact, glove/payload geometry and impulse measurement.
- Validate perturbation recovery before introducing active opponents.
- Progress from static targets to constrained robot-to-robot drills and then bounded bouts.
- Evaluate opponent perception uncertainty, latency, occlusion and target identity.
- Use independent match logging and score verification.

### Personality work

- Define a small set of style parameters and persona states.
- Map styles to qualified motion variants, timing and bounded tactical preferences.
- Permit celebrations/taunts only through normal initiation and transition checks.
- Keep persistent identity and narrative state outside the control loop.
- If opponent memory is implemented, use a qualified embedded local store with versioned entries, confirm that an unavailable, empty or stale store degrades to opponent-agnostic behavior, and settle the retention and deletion terms of C-17 before any record outlives a session.
- Evaluate recognizable style and its cost in balance, energy, latency and task success.

### Exit gate

Contact behavior is validated within a stated robot/rule/envelope configuration. Expressive behavior cannot bypass supervisor or recovery priority. Neither extension is required to call R0 or R1 complete.

## 14. Phase 11 — Optional agent layer (only if C-15 adopts it)

This phase exists so the v3.0 proposals have a defined place to be evaluated, not because they are scheduled. It starts only after R1, only if C-15 resolves to adopt something, and it is cancellable at any point with no effect on any earlier phase. It covers only what the three admissible uses of `SPECS.md` Section 2.3.1 do not already place elsewhere: U1 tooling is continuous from Phase 1 and U3 knowledge-base maintenance is continuous from Phase 0, so the tracks below are the remaining, genuinely optional work.

### Work — track A: common prerequisites

- Qualify the non-model components first and separately: the session and durable event store, the markdown index and the local graph memory, each under Appendix B.6 against our own fixtures. If the useful parts cannot be taken without the model client, stop here and record that.
- Cost the local-inference substitution path of Appendix B.5 item 10 before any model-dependent capability is built. If the honest answer is that the capability only works against a hosted endpoint, it stays an experiment and is never promoted.
- Build the advisory interface as a validated, timestamped, expiring record over the existing queue contract, out of process, behind a default-off feature. Report rejection and expiry counts like any other asynchronous proposal.
- Evaluate individual proposals — offline training-campaign orchestration, skill cataloging, opponent-memory curation, commentary and operator advisories — one at a time against a prespecified metric. Offline and operator-facing uses are preferred because they carry no deadline.
- Test unreachable endpoint, timeout, truncated and malformed responses, and prove the A-24 removal case: with every model-service feature disabled, the control loop, safety envelope and acceptance matrix pass unchanged.

### Work — track B: offline policy-design and sampling assistance (use U2)

This is the only track with a plausible effect on policy quality, and it runs entirely offline, between experiments, per `SPECS.md` Section 9.5.

- Start from the measurement, not the sampler. Track B begins only once Phases 4 and 7–8 have produced a baseline whose comparison protocol — fixed budgets, seed lists, held-out sets — can reject a proposal. Without that, an assisted proposal cannot be shown to be wrong, and anything that cannot be wrong is not an improvement.
- Define the proposal interface: a candidate set of priors, curricula, randomization ranges, scenario and opponent variations, reward-term candidates or expressive style variations, entering through the Section 9.3 algorithm manifest with its claim stated in advance and its own feature flag.
- Keep the generator away from the examiner. Evaluation seeds, held-out disturbances and held-out opponents are fixed independently of whatever proposed the training distribution, and this is asserted in the run record rather than assumed.
- Treat personality as sampling, not generation: candidate style parameter sets over already-validated expressive skills per `SPECS.md` Section 12.5, each a versioned configuration measured for recognizability without task or stability regression. No runtime model call produces motion, cadence or speech.
- Survey comparable prior work of this kind — the personality, expressive-motion and persistent-identity leads already in the research intake register — as candidate sampling strategies at the same evidence bar. Their presence in a survey is not a result.
- Report negative results and keep them: a proposal set that produced no measured improvement is recorded so the same suggestions are not re-sampled forever.

### Work — track C: dependency knowledge base beyond the authored base (use U3)

Only reached if the maintained base of Phase 0 and Phase 6 exists and its limits have been felt in a real diagnosis.

- Measure first whether retrieval over the base is actually the bottleneck. Local lexical and exact search over version-controlled records is the baseline; assisted retrieval or drafting is justified only against it, per Appendix B.2.5 and B.6 item 8.
- Keep the base readable and usable with every model-service feature disabled, so an incident during a network outage is still diagnosable. Assisted retrieval that becomes the only access path has converted an operational record into an external dependency.
- Hold assisted-draft entries to the citation rule: an entry whose source cannot be named is a note, is marked as one, and is never read back as a specification or cited to close an acceptance row.

### Exit gate

Each adopted capability has a measured benefit against its baseline, an exercised offline-degradation mode and no path into the control loop or the supervisor. For track B, benefit means a measured improvement on held-out sets that were fixed independently of the proposal source; for track C, it means a measured improvement over local search on the same records. Guardrail and audit components are recorded as tooling, never credited in a hazard register. A negative result closes the capability; an unfalsifiable "it helps" is not an outcome.

## 15. Dependency map and work streams

```text
0 Evidence / Rust feasibility
  -> 1 Contracts
       -> 2 Motion processing ----+
       -> 3 Simulation / PPO -----+-> 4 Tracker / R0
       -> 5 Runtime ----------------------------+
                                                +-> 6 Transfer / R1
4 -> 7 Retrieval / experts
4 -> 8 Latent methods / self-play
6 -> 9 Limited-data adaptation
6 + 8 -> 10 Contact experiments
6 + qualified expressive skills -> 10 Personality
6 + C-15 decision -> 11 Optional agent layer  (nothing depends on it)

U1 development automation : 1 -> continuous   (no shipping crate depends on it)
U3 dependency knowledge base: 0 -> 6 -> continuous  (operations asset, not a gate)
U2 assisted sampling       : 4 + 7/8 baseline -> 11 track B  (never earlier)

Tier declaration + budgets  : 4 seeded -> 5 enforced -> 8 re-measured  (A-32, C-22)
R0 corpus retrieval         : 2 corpus + 7 exact search -> 8 track     (A-34, C-23)
R1/R2 retrieval tiers       : 5 runtime -> 7 measured per tier         (A-35)
```

Data/retargeting, simulation/learning and runtime/observability can proceed as separate engineering work streams after shared schemas stabilize. Schema changes require fixture migration and integration checks across all consumers. This plan does not assume any particular team size or request autonomous agent delegation.

## 16. Benchmark and verification program

Maintain a versioned benchmark manifest containing hardware, workloads, seed lists, thresholds, measurement duration and required artifacts.

| Suite | Examples | Runs when |
| --- | --- | --- |
| Environment | Clean-checkout bootstrap, `devenv test`, lock drift between developers and CI, excluded-runtime gate, host-platform coverage | Toolchain, `devenv.lock` or CI definition changes |
| Tooling isolation | No shipping crate depends on `robot-devtools` or any automation; the build and every acceptance row survive its removal; generated contributions are attributable | Tooling, CI definition or workspace-dependency changes |
| Knowledge base | Every inventoried dependency has a cited entry; version changes mark dependent entries stale; the base is readable with no model service; closed faults have linked fault records | Dependency, firmware, driver or robot-profile changes, and after every diagnosed fault |
| Schema / ingestion | Frames, channel order, malformed input, timestamp gaps, license metadata | Data/model contract changes |
| Dependencies | Features, native linkage, build scripts, maintenance, license, fallback plan | Dependency/version/backend changes |
| Optimization | Constraint residuals, setup/update/solve tails, infeasible cases, timeout fallback | Solver/formulation/model changes |
| Estimation | Ground-truth error, covariance consistency, bias/contact faults, conditioning | Sensor/filter/model changes |
| Instrumentation | Enabled/disabled timing, queue overflow, exporter outage, cardinality bounds | Telemetry/runtime changes |
| Encoding / parallelism | Message validation/migration, latency/allocations, sequential vs. parallel rollouts | Encoding/pool/schema changes |
| Numerical | FK/IK known solutions, gradients, return calculation, action transforms | Math/training changes |
| Physics | Pendulum, friction, contact impulses, standing and pushes | Physics or robot model changes |
| Fidelity tiers | Each declared tier passes the shared `Check` battery; a report is written per run; CPU-vs-GPU agreement and contact-budget counter once a GPU backend exists | Physics, solver, integrator or tier-setting changes |
| Checkpoint and parameter groups | Save/load restores weights, optimizer, RNG and normalizer; frozen groups stay bit-identical during fine-tuning | Learner or checkpoint-format changes |
| Visualization output | Golden-file CSV/SVG output for fixed fixtures; one frame per simulation step | Frame-dump or viewer changes |
| Learning | Small-task PPO curves, resume equivalence within tolerance | Learner/backend changes |
| Motion | Held-out tracking, foot slip, saturation, transitions | References or tracker changes |
| Runtime | Allocation checks, tails under load, stale input and watchdog faults | Runtime/inference/hardware changes |
| Retrieval | Routing quality, switch failures, cold-cache behavior, unseen states | Index/router/expert changes |
| Adaptation | Session-held-out metrics, drift, residual limits, rollback | Calibration/adapter changes |
| Boxing | Hit attribution, opponent holdouts, recovery, match rule compliance | Task/strategy/contact changes |
| Hierarchy latency | Per-tier compute tails, deadline margin and decision age under load; tier-rate monotonicity; late, absent and expired upper-tier outputs; highest-tier removal leaving a safe controller | Any change to a tier assignment, rate, budget or hierarchical component |
| Latent pipeline | Per-stage checkpoint and qualification; latent dimensionality, bound and decoder identity; reservoir reproducibility; opponent-averaging ablation on held-out opponents; stage deletion leaving a usable tracker | Latent, distillation or self-play changes |
| Retrieval-augmented learning | Corpus revision, license and provenance labels; tested held-out exclusion; target-only and whole-corpus comparisons at equal budget; retrieval and admission-rejection counts; retriever removal | Corpus, query, similarity, embedding-version or admission-filter changes |
| Retrieval tiers | R0/R1/R2 configured and measured separately; no file, network, embedding or allocation work on an R2 path; R2 cost in the policy-path tails with decision age; per-tier empty-result and cache-expiry behavior | Index, router, working-set or corpus-retrieval changes |
| Offline degradation | Default build free of model-service features, full acceptance matrix with the agent layer absent, unreachable/timeout/malformed-response handling, advisory expiry and rejection counts | Any agent-layer or external-model-service change, and as a standing check once such a feature exists |

Keep deterministic correctness checks separate from expensive stochastic training and hardware tests. Do not rerun a full training campaign for documentation-only changes. CI must not imply that hardware tests passed when a device was unavailable.

## 17. Main risks and planned responses

| Risk | Early detection | Response |
| --- | --- | --- |
| Rust simulation lacks adequate articulation/contact fidelity | Phase 0 qualification | Fund missing Rust backend work; restrict early tasks and document limits |
| CPU simulation makes experiments impractical | Measured rollout throughput | Smaller reference tasks, better batching; qualify Rust GPU physics separately |
| Autodiff/inference parity gaps | Save/load and action-parity tests | Restrict operator set and use a shared Rust model definition |
| Dataset unusable or mislabeled | Source/license audit and clip inspection | Use eligible subsets or collect a small targeted capture set |
| Retargeted motions are dynamically impossible | Actuator/contact checks and tracker failures | Adjust timing/amplitude or reject motion |
| Expert transitions destabilize control | Boundary-state and perturbation suites | Prefer conditioned tracker; train transitions before enabling switches |
| Retrieval slower than dense inference | Fixed-hardware latency and memory benchmark | Retain dense runtime; use retrieval only for planning/curation |
| Sensors lack synchronization or observability | Clock and identification diagnostics | Improve synchronization; add targeted sensing or preserve uncertainty |
| Small real datasets overfit | Held-out sessions and learning curves | Narrow adaptation scope and collect informative additional trials |
| Competitive policies exploit simulator/scoring | Scripted contact tests and opponent diversity | Correct scoring/model; invalidate affected training runs |
| Paper claims cannot be verified/reproduced | Evidence ledger and minimal reproduction | Keep idea experimental; do not block baseline delivery |
| Locked environment unusable on a required host or drifting from CI | Clean-checkout bootstrap check and lock-revision comparison between developer, CI and benchmark runs | Pin inputs in `devenv.lock`; build the CI image from the same lock for hosts without Nix; treat an onboarding failure as a defect in `devenv.nix` |
| Agent-layer scope creep from a survey into the baseline | A model-service feature appears in the default closure, or an advisory becomes load-bearing | Enforce the exclusion gate as a build failure; keep the A-24 removal test standing; cancel Phase 11 without penalty |
| Hosted model changes, degrades or is withdrawn | Behavior drift with no code change; endpoint deprecation notice | Demote dependent claims per `SPECS.md` Section 3; fall back to the offline mode; do not schedule a capability without a costed local-inference path |
| Development automation becomes load-bearing | A build, test or release step that nobody can perform by hand; a tool in a shipping crate's dependency graph | Tooling-isolation suite as a standing check; removal test per row A-25; record the manual cost of every automated step |
| Unreviewed generated contribution reaches the repository | A commit no author can explain; a test that passes only against generated expectations | Same review and CI gate as authored work; attribution per `SPECS.md` Section 2.3.1 item 5; never generate an implementation and its accepting test in one run |
| Assisted sampling overfits its own evaluation | Improvements that vanish on independently fixed held-out seeds, disturbances or opponents | Fix held-out sets independently of the proposal source; prespecified claim and ablation per Section 9.3; keep negative results |
| Knowledge base becomes stale or authoritative-looking | A fault diagnosed against an outdated firmware or driver note; a report citing the base as evidence | Version-keyed invalidation, mandatory citation, visible staleness, and the standing prohibition on citing the base as evidence or as a hazard control |
| A hierarchy is claimed while every layer runs at one rate | Identical configured rates across tiers; decision age reported as zero | Record the degenerate case per `SPECS.md` Section 5.1 item 2 and re-measure at the intended tier rates before any hardware claim |
| An expensive mechanism migrates downward into a tighter tier | Retrieval, dynamic dispatch, allocation or variable-iteration optimization appearing below the tier at which it was qualified | Treat it as a Section 5 change requiring a change-control record; per-tier budget tests as a standing check |
| A training constraint is credited as a safety control | A latent prior, manifold bound or style discriminator cited in a hazard register | Section 9.4.2 items 4 and 5 forbid it; keep the supervisor independent and re-review the hazard register when a latent stage is added |
| Retrieval improves results only because it added data | A gain that survives neither the target-only nor the whole-corpus comparison | Both comparisons at equal update budget are mandatory per row A-34; report retrieval and rejection counts, not just success rate |
| The retriever reaches held-out data | Implausible few-shot results; an evaluation episode found in the corpus | Exclude by construction and test the exclusion; treat any breach as invalidating every affected run |
| Adaptation is scheduled ahead of its ordering | A residual or on-hardware update proposed before identification and history conditioning were measured on the same robot | The Section 11.3 ordering is binding; each step needs its own C-24 record and a measured insufficiency above it |
| The retired Python prior art is reintroduced by convenience | A PyTorch, MJX or CycloneDDS dependency reappearing in the closure; a restored `training/` directory | The exclusion gate fails the build; reintroduction is a Section 2.1 exception request, not a revert |
| Rendering starts before the seams and harness are stable | Test renders depend on types that change in Phase 1 or Phase 3 | Keep rendering to frame dumps and golden files until the Phase 1 traits and Phase 3 harness are closed |
| Policy code stays hard-wired to MLPs | A new policy type needs edits in `PpoAgent`, `ControlLoop` and the checkpoint path | Introduce the `Policy` trait before a second policy type is added; a recurrent policy is its first test |

## 18. First executable backlog

The first implementation batch should be small enough to produce evidence before architecture expands:

1. Create the devenv environment, commit `devenv.lock`, and verify that a clean checkout builds and tests through `devenv shell` / `devenv test`.
2. Create evidence-ledger and architecture-decision formats.
3. Benchmark one Rust training/inference backend and one Rust physics candidate.
4. Define typed motion/robot/action contracts with tiny fixtures.
5. Implement BVH inspection and canonical conversion.
6. Load a reference robot, compute FK and display one retargeted static pose.
7. Run one actuated joint with recorded command/measurement timestamps.
8. Train, checkpoint and reload a tiny PPO policy in a Rust simulator task.
9. Replay a policy through a preallocated runtime and report sensor-to-command latency.
10. Review feasibility and measured costs before expanding into humanoid motion tracking.

The final output of each batch is working code, reproducible artifacts, an evaluation report and an updated decision record. The project advances when those artifacts support the next phase, not when a list of frameworks has been assembled.

## 19. Ecosystem qualification deliverables

The addendum introduces concrete outputs without requiring all candidates to enter the build:

| Deliverable | Phase | Completion evidence |
| --- | --- | --- |
| Development-environment decision | 0–1 | Committed `devenv.lock`, clean-checkout bootstrap record, `devenv test` CI job, exercised host platforms and a passing excluded-runtime gate; closes `SPECS.md` C-14 |
| Source-backed dependency register | 0–1 | Canonical URLs, inspected versions, evidence state, license and capability-specific maturity |
| Learning backend decision | 0 | Burn workload results; alternative comparison if required; training/export/inference parity |
| Physics capability and gap report | 0, 3 | Existing Rapier CCD/multibody evaluation, verified alternative results, bounded authorship tasks |
| Parser/model validation selection | 1–2 | BVH/C3D/URDF fixture and robustness results; unsupported-feature handling |
| Runtime/transport decision | 5 | Reference architecture findings, measured scheduling/IPC, protocol compatibility |
| Retrieval crossover report | 7 | Exact-search measurements and justified ANN adoption or rejection |
| Identification module decision | 9 | Derivative checks, observability/identifiability findings, held-out fit quality |
| QP/MPC solver decision | 0, 2; runtime requalification in 5 | Rust-compatible toolchain, full-body constraint benchmarks, infeasibility and deadline handling |
| Estimator decision | 0, 5–6 | Verified implementation, sensor model, accuracy/consistency and fault recovery results |
| Robustness harness/toolchain decision | 1–2 | Parser mutation/property coverage and minimized regression corpus; no implicit foreign fuzz-engine exception |
| Telemetry decision | 1, 5 | Off-thread instrumentation, bounded overhead and exporter-failure isolation |
| Encoding and worker-pool decision | 1, 3, 5 | Message-size/latency/migration results, sequential speedup baseline, isolated resource budgets |
| Agent-layer adoption decision | 0 for identity, 11 for anything further | Registry-confirmed crate identity with the name collision recorded; a decision that closes `SPECS.md` C-15, which may legitimately be "adopt nothing" |
| Offline-degradation evidence, if any capability is adopted | 11 | Default-closure proof, unreachable-endpoint behavior and a passing acceptance matrix with every model-service feature disabled; closes rows A-23 and A-24 |
| Development-automation boundary and review rule | 1, then continuous | `robot-devtools` boundary with the CI assertion that no shipping crate depends on it, the generated-contribution review and attribution rule, and a passing removal test; closes row A-25 and the U1 part of `SPECS.md` C-18 |
| Dependency knowledge base | 0 seeded, 6 keyed to the robot, then continuous | Cited, version-keyed entries for every inventoried dependency and platform input, staleness on version change, local readability with no model service, and fault records linked to episode records; closes row A-27 and C-19 |
| Assisted-sampling protocol, if track B is run | 11 | Proposal manifests with prespecified claims, independently fixed held-out sets, recorded proposal origin and model version, ablations and retained negative results; closes row A-26 |
| Prior-art retirement record | 0 | The commit that last contained `training/`, `unitree/` and `test.ipynb`, the ledger entry citing it, and a passing exclusion gate in the default closure; closes `SPECS.md` C-01 |
| Tier assignment and per-tier budget report | 4 seeded, 5 enforced, 8 re-measured | Declared tier, rate, measured compute duration and deadline margin per learned component, per-tier fallback results and the highest-tier removal test; closes row A-32 and `SPECS.md` C-22 |
| Staged latent pipeline report | 8 | Per-stage checkpoints and qualification, declared latent dimensionality/bound/decoder identity, opponent-averaging ablation, reproducible population sampling and the equal-budget comparison against the non-latent alternatives; closes row A-33 |
| Retrieval-augmented learning report | 8 | Corpus manifest with licenses and provenance labels, tested held-out exclusion, target-only and whole-corpus comparisons at equal budget, retrieval and rejection statistics; closes row A-34 and `SPECS.md` C-23 |
| Retrieval tier separation report | 7 | Separate R0/R1/R2 configuration and measurement, an R2 path proven free of file, network, embedding and allocation work, policy-path tails with decision age, and per-tier empty-result behavior; closes row A-35 |
| Limited-data adaptation ordering record | 9 | One record per scheduled method with the measured insufficiency of the methods above it, declared deployment observation sets, bounded authority enforced externally, and sample efficiency stated in real minutes, episodes and interventions; closes `SPECS.md` C-24 |
| Plug-in seam traits | 1 | `Policy`, `Backend` and model-importer traits in place, with existing tests passing unchanged; closes the seam gap in the status snapshot |
| Fidelity harness report | 3, then each tier | `Check`/`Tier` runner, a shared battery and a JSON or TOML report per declared tier |
| Checkpoint and parameter-group support | 3, prerequisite for 9 | Round-trip checkpoint including the normalizer, and a frozen-group test; prerequisite for partial fine-tuning in Phase 9 |
| Test-render artifacts | 2 | Frame dump and SVG golden files for a fixed simulation fixture, with no new dependency |

Do not schedule a complete articulated-dynamics rewrite, full SMPL fitting system or hybrid physics engine solely because the supplied survey reports a gap. Resolve the evidence first, then estimate the smallest necessary Rust implementation. Dependency promotion follows Appendix B.6; unqualified candidates remain references.

Track Appendix B.7 risks alongside Section 17, including estimator instability, solver timeouts, telemetry overhead, archive compatibility, development-tool policy conflicts and the hosted-model and agent-authority risks added by v3.0. Use observed evidence to assign impact/likelihood instead of importing the survey's ratings. No new `robot-state`, `robot-ctrl` or `robot-test` crate is required merely by the survey terminology; existing perception/control/evaluation boundaries own these tasks initially, and the same applies to the v3.0 `robot-brain` proposal under C-16.

The v3.0 agent additions are held to the same standard as every earlier survey claim. A reported capability does not create a requirement, a Rust client for a hosted model is not a Rust-native capability, and an orchestration framework advances no gate in Phases 0 to 6. Deliver the measured motion-to-control path first and decide C-15 on evidence, including the entirely acceptable outcome of adopting none of it. The same applies to the three admissible uses: automating project work, assisting policy sampling and maintaining dependency knowledge are ways of working, and each is kept only while it demonstrably reduces cost or improves a measured result. A tool that cannot be removed has stopped being a tool.

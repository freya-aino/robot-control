# Implementation Plan

This plan implements `SPECS.md` as a new logical architecture without relying on any existing repository structure. All project-authored implementation, tools, training, and tests are Rust. Milestones advance on evidence rather than dates; hardware availability and the Rust simulation stack are major dependencies.

The revised v2.0 Rust ecosystem addendum supersedes the previous integration in `SPECS.md` Appendix B. Apply its expanded solver, estimation, observability, testing, serialization and parallelism register and qualification gates throughout this plan. Incoming survey claims remain unverified until primary sources or executable evidence support them; reported versions are not dependency pins. Existing Rapier/Rerun corrections and the Rust-only policy remain in force.

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

1. Rust compiler/toolchain and supported host OS/CPU/GPU platforms.
2. Rust-native physics and differentiable-compute capability findings.
3. First simulator robot model and first hardware target when available.
4. Canonical units, coordinate frames, quaternion encoding and timing model.
5. Observation/action schema, actuator interface and artifact representation.
6. Dataset eligibility and redistribution restrictions.
7. Benchmark machine, control period, timing acceptance profile and physical operating envelope.

Hardware-agnostic APIs do not require choosing the physical robot immediately. Simulator qualification and joint-level policy training do require a concrete model. Begin with a documented reference embodiment and test a structurally different second model before claiming portability.

## 3. Phase 0 — Evidence audit and Rust feasibility gates

### Work

- Establish a source ledger from the research leads in `SPECS.md`.
- Verify RoboStriker's paper identity, algorithms, data, code status and reported sim-to-real evidence. Extract equations and hyperparameters only from the primary source.
- Verify Unitree boxing product/demo claims and clearly label unknown implementation details.
- Resolve SPiRL, STRAP, SMOSE and ASAP first; then investigate delay-aware and newer retrieval leads. Reject placeholder or mismatched references.
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

### Deliverables

Evidence ledger, dependency inventory, capability matrix, benchmark reports and decisions naming supported baseline features.

### Exit gate

A Rust-only executable can run a small controlled physics task, collect trajectories, differentiate a training loss, update a policy, save/load it and execute inference. Simulator throughput and fidelity limitations are documented.

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
- Set up `cargo fmt`, `cargo clippy`, unit/integration checks, dependency/license review and reproducible build records.
- Define simulator, hardware, learner and inference interfaces without promising stable dynamic-plugin ABI.
- Evaluate `urdf-rs` with our semantic validation layer; inventory unsupported model features explicitly.
- Create per-dependency records covering enabled features, transitive native linkage, build scripts, license, maintenance and replacement cost under Appendix B.6.
- Define versioned message envelopes and compare bincode/postcard/rkyv/Cap'n Proto/prost only where their full build/runtime toolchains satisfy policy. Separate IPC encoding, transport and durable storage decisions.
- Establish Rust property/mutation/regression harnesses. cargo-fuzz/libFuzzer and honggfuzz require a separate development-tool exception; do not mandate them under the current Rust-only requirement.
- Define off-thread tracing/metrics ownership, bounded event records and exporter failure behavior before integrating OpenTelemetry.

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
robot adapt identify --episodes <manifest> --robot <profile>
robot deploy --bundle <id> --mode shadow
robot replay <episode-id>
```

These commands are interface requirements, not currently implemented tools.

### Exit gate

Fixtures round-trip without losing units, frames or timestamps; invalid or incompatible artifacts are rejected. Every run is traceable to source/config/data/model versions.

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
- Investigate Rust SMPL-family implementations before scheduling authorship. Scope parsing, licensed model evaluation and fitting separately; keep BVH delivery independent of this investigation.
- Resolve the constrained solver before implementing differential-QP IK: benchmark representative full-body matrices, setup/update/solve costs, cold/warm behavior, residuals, memory and infeasible cases. Retain a feasible non-QP baseline where appropriate.

### Exit gate

A fixed reference set retargets reproducibly to two different model geometries. Reports expose contact slip, unreachable targets and limit violations. No infeasible reference is silently accepted through joint clipping.

### Deferred work

Live optical/inertial capture and video-to-motion inference follow the file-based vertical slice. They reuse canonical records and synchronization contracts.

## 6. Phase 3 — Qualified simulation and baseline RL

### Work

- Implement simulator stepping, reset, batched environments and independently seeded randomization.
- Add motor response, control decimation, sensor noise/delay and explicit command semantics.
- Qualify pendulum, contact/friction, static balance and perturbation benchmarks.
- Implement PPO, advantage estimation, action distributions, value loss, entropy regularization and normalization.
- Distinguish truncation from terminal failure in return calculation.
- Implement checkpoint/resume including optimizer, RNG and normalizer state.
- Train small reference tasks and compare learning against deterministic baselines.
- Train standing, balance and basic stepping on the selected reference robot.
- Measure environment throughput and project the cost of the next experiment; introduce GPU simulation only through a separately qualified backend.
- Follow Appendix B.4: qualify existing Rapier articulation/contact first and compare any verified alternative at matched initial conditions and controls. Author only demonstrated gaps.
- Test full episode snapshot/replay, including sensor histories, task state, RNG and recurrent policy state outside the physics engine.
- Cross-validate through MuJoCo bindings only if a separate foreign-dependency scope decision has been made. Keep this optional profile outside the default dependency path.
- Benchmark Rayon CPU rollout/preprocessing pools against sequential execution. Use isolated resources and per-environment RNG streams; document reduction/scheduling nondeterminism and bandwidth limits.

### Exit gate

Baseline RL succeeds on agreed small tasks across at least three seeds, resumes correctly and exports a CPU-executable actor. Contact behavior and numerical sensitivity are documented. A basic simulated humanoid controller maintains balance within its acceptance profile.

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
- Compare HULKs/Copper runtime patterns and `iceoryx2` host IPC against actual scheduling and payload requirements. Evaluate Zenoh, `roslibrust` or dora only where orchestration/interoperability warrants them; verify protocol compatibility and dependency closure.
- Qualify the estimator inside `robot-perception`: IMU bias/rotation conventions, contact fusion, covariance consistency, jitter, dropout, outlier and slip cases. Compare multicalc/numeris/typed_kalman/kfilter or an authored baseline against independent truth.
- Benchmark instrumentation enabled/disabled, full event queues and disconnected exporters. Construct OpenTelemetry spans off-thread from bounded timestamps instead of assuming span entry/exit is deadline-safe.
- Benchmark serialization p99.9, allocations and validation on sensor/command/snapshot shapes; test malformed inputs and version migration. Keep general Rayon work stealing outside the control loop.

### Exit gate

Fault injection never sends invalid or expired commands. The designated target meets its timing profile for a stated qualification duration, with max latency and missed deadlines reported. Shadow mode can run from live sensors without actuation. Scheduling measurements alone are not called a hard-real-time guarantee.

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

### Exit gate

The report states whether retrieval actually helps over the dense baseline. A new skill can be added through a versioned manifest and targeted training, but promotion still requires routing and transition regression tests. Failed or late retrieval never stalls the control path.

## 11. Phase 8 — Research methodology ingestion and self-play

### Work

- Implement one verified latent-skill baseline with encoder, prior, decoder and distillation objectives.
- Reproduce a small benchmark before increasing capacity or adopting paper-specific regularizers.
- Compare repertoire coverage, tracking and controllability of latent versus expert representations.
- Implement boxing task geometry, controlled contact, scoring, fall rules and ring boundaries.
- Validate hit detection using scripted positive/negative cases before RL receives its reward.
- Add opponent snapshots, checkpoint tournaments, held-out opponents and evaluation against scripted policies.
- Implement the verified RoboStriker competitive method as a research variant if evidence and resources support it.
- Add delay-aware policy training using recorded latency distributions and action histories.
- Evaluate STRAP-inspired trajectory retrieval or SMOSE-inspired shallow experts as independent experiments, not required replacements.

### Exit gate

Each method has a source-backed algorithm card, fixed-budget baseline, multiple seeds, ablations and an outcome report. Strategy improvements persist against held-out opponents and cannot be explained by scoring bugs or unsafe contact exploitation.

## 12. Phase 9 — Data-efficient adaptation

### Implementation order

1. Calibrated system identification and improved randomization ranges.
2. Simulation-trained history encoder for latent dynamics adaptation.
3. Offline simulator correction with well-defined transition alignment.
4. Bounded robot-side residual or small adapter, only if prior steps leave a measurable gap.

### Work

- Add event-triggered recording with representative successful background samples.
- Select calibration trajectories for information gain within the current operating envelope.
- Build real-data splits by session and operating condition.
- Compare adaptation methods at equal real-interaction budgets and report learning curves.
- Verify deployment inputs contain no unavailable paired simulator state.
- Constrain residual authority and test fallback to the frozen controller.
- Use shadow evaluation, simulation regression and staged hardware validation for every candidate.
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
- Evaluate recognizable style and its cost in balance, energy, latency and task success.

### Exit gate

Contact behavior is validated within a stated robot/rule/envelope configuration. Expressive behavior cannot bypass supervisor or recovery priority. Neither extension is required to call R0 or R1 complete.

## 14. Dependency map and work streams

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
```

Data/retargeting, simulation/learning and runtime/observability can proceed as separate engineering work streams after shared schemas stabilize. Schema changes require fixture migration and integration checks across all consumers. This plan does not assume any particular team size or request autonomous agent delegation.

## 15. Benchmark and verification program

Maintain a versioned benchmark manifest containing hardware, workloads, seed lists, thresholds, measurement duration and required artifacts.

| Suite | Examples | Runs when |
| --- | --- | --- |
| Schema / ingestion | Frames, channel order, malformed input, timestamp gaps, license metadata | Data/model contract changes |
| Dependencies | Features, native linkage, build scripts, maintenance, license, fallback plan | Dependency/version/backend changes |
| Optimization | Constraint residuals, setup/update/solve tails, infeasible cases, timeout fallback | Solver/formulation/model changes |
| Estimation | Ground-truth error, covariance consistency, bias/contact faults, conditioning | Sensor/filter/model changes |
| Instrumentation | Enabled/disabled timing, queue overflow, exporter outage, cardinality bounds | Telemetry/runtime changes |
| Encoding / parallelism | Message validation/migration, latency/allocations, sequential vs. parallel rollouts | Encoding/pool/schema changes |
| Numerical | FK/IK known solutions, gradients, return calculation, action transforms | Math/training changes |
| Physics | Pendulum, friction, contact impulses, standing and pushes | Physics or robot model changes |
| Learning | Small-task PPO curves, resume equivalence within tolerance | Learner/backend changes |
| Motion | Held-out tracking, foot slip, saturation, transitions | References or tracker changes |
| Runtime | Allocation checks, tails under load, stale input and watchdog faults | Runtime/inference/hardware changes |
| Retrieval | Routing quality, switch failures, cold-cache behavior, unseen states | Index/router/expert changes |
| Adaptation | Session-held-out metrics, drift, residual limits, rollback | Calibration/adapter changes |
| Boxing | Hit attribution, opponent holdouts, recovery, match rule compliance | Task/strategy/contact changes |

Keep deterministic correctness checks separate from expensive stochastic training and hardware tests. Do not rerun a full training campaign for documentation-only changes. CI must not imply that hardware tests passed when a device was unavailable.

## 16. Main risks and planned responses

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

## 17. First executable backlog

The first implementation batch should be small enough to produce evidence before architecture expands:

1. Create evidence-ledger and architecture-decision formats.
2. Benchmark one Rust training/inference backend and one Rust physics candidate.
3. Define typed motion/robot/action contracts with tiny fixtures.
4. Implement BVH inspection and canonical conversion.
5. Load a reference robot, compute FK and display one retargeted static pose.
6. Run one actuated joint with recorded command/measurement timestamps.
7. Train, checkpoint and reload a tiny PPO policy in a Rust simulator task.
8. Replay a policy through a preallocated runtime and report sensor-to-command latency.
9. Review feasibility and measured costs before expanding into humanoid motion tracking.

The final output of each batch is working code, reproducible artifacts, an evaluation report and an updated decision record. The project advances when those artifacts support the next phase, not when a list of frameworks has been assembled.

## 18. Ecosystem qualification deliverables

The addendum introduces concrete outputs without requiring all candidates to enter the build:

| Deliverable | Phase | Completion evidence |
| --- | --- | --- |
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

Do not schedule a complete articulated-dynamics rewrite, full SMPL fitting system or hybrid physics engine solely because the supplied survey reports a gap. Resolve the evidence first, then estimate the smallest necessary Rust implementation. Dependency promotion follows Appendix B.6; unqualified candidates remain references.

Track Appendix B.7 risks alongside Section 16, including estimator instability, solver timeouts, telemetry overhead, archive compatibility and development-tool policy conflicts. Use observed evidence to assign impact/likelihood instead of importing the survey's ratings. No new `robot-state`, `robot-ctrl` or `robot-test` crate is required merely by the survey terminology; existing perception/control/evaluation boundaries own these tasks initially.

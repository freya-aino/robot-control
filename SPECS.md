# Humanoid Robotics Platform Specification

Status: proposed architecture and acceptance contract. Not ratified; no clause below is evidence that the described system exists.

| Field | Value |
| --- | --- |
| Document version | 0.3.1-draft |
| Last revised | 2026-10-24 |
| Companion documents | `PLAN.md` (phasing, gates, backlog); evidence ledger; decision records; per-robot acceptance and safety profiles; `devenv.nix`, `devenv.yaml` and `devenv.lock` as the executable form of Section 2.2 |
| Normative scope | Sections 1–15 bind the implementation. Appendix A is definitional, Appendix B is advisory until its gates pass, Appendix C records what is still undecided |
| Change control | Changes to Sections 2, 4, 5, 7, 12 or 14 require a decision record and a review of every dependent acceptance profile, fixture and consumer crate |

### Revision history

| Version | Change | Date |
| --- | --- | --- |
| 0.1 | Initial specification, Sections 1–15 | — |
| 0.2 | Appendix B replaced with the supplied Rust ecosystem survey revised v2.0 | — |
| 0.3.0-draft | Document control, normative-language convention, glossary (Appendix A), operating-mode and evidence-promotion tables, foreign-dependency exception procedure, safety case (Section 12), acceptance traceability, open-questions register (Appendix C) | 2026-10-24 |
| 0.3.1-draft | devenv-provisioned development environment (Section 2.2), locked-environment recording in Section 13, acceptance row A-22, glossary and open-question entries. Touches Sections 2 and 14, so the change-control rule applies and the decision record is tracked as [C-14](#appendix-c--open-questions-and-unresolved-decisions) until written | 2026-10-24 |

### Normative language

Key words are used in the sense of RFC 2119/8174 when capitalized:

| Keyword | Meaning |
| --- | --- |
| MUST / MUST NOT | Binding. A violation blocks promotion and must be recorded as a defect, not a known limitation |
| SHOULD / SHOULD NOT | Strong default. Deviation requires a written rationale in the relevant decision record |
| MAY | Permitted option carrying no obligation |

Uncapitalized prose in this document is descriptive context, rationale or a correction to an earlier premise. Where descriptive prose and a capitalized requirement conflict, the capitalized requirement governs. Tables in Sections 7, 11 and 14 are requirements; tables in Appendix B are candidate registers, not approvals.

### How to read this document

Sections 1–5 fix policy: what is built, in what language, on what evidence, in which operating modes, under which timing constraints. Sections 6–13 fix contracts: components, artifacts, pipelines, runtime behavior and operations. Section 14 is the acceptance contract and Section 15 bounds the first release. Appendix A defines terms, Appendix B qualifies dependencies, Appendix C lists unresolved decisions that currently block requirements.

### Contents

1. [Purpose and scope](#1-purpose-and-scope)
2. [Language and dependency policy](#2-language-and-dependency-policy)
3. [Evidence policy and corrections to the research premise](#3-evidence-policy-and-corrections-to-the-research-premise)
4. [Operating modes and boundaries](#4-operating-modes-and-boundaries)
5. [Timing and execution model](#5-timing-and-execution-model)
6. [Proposed workspace and component responsibilities](#6-proposed-workspace-and-component-responsibilities)
7. [Robot and artifact contracts](#7-robot-and-artifact-contracts)
8. [Motion acquisition and retargeting](#8-motion-acquisition-and-retargeting)
9. [Simulation, training and methodology extension](#9-simulation-training-and-methodology-extension)
10. [Retrieval-indexed policy bank and low latency](#10-retrieval-indexed-policy-bank-and-low-latency)
11. [Transfer and data-efficient improvement](#11-transfer-and-data-efficient-improvement)
12. [Safety, supervision, perception and personality](#12-safety-supervision-perception-and-personality)
13. [Reproducibility and operational requirements](#13-reproducibility-and-operational-requirements)
14. [Acceptance and evaluation matrix](#14-acceptance-and-evaluation-matrix)
15. [Explicit non-goals for the first release](#15-explicit-non-goals-for-the-first-release)

- [Appendix A — Terminology, units and conventions](#appendix-a--terminology-units-and-conventions)
- [Appendix B — Rust ecosystem survey and dependency qualification](#appendix-b--rust-ecosystem-survey-and-dependency-qualification-revised-v20)
- [Appendix C — Open questions and unresolved decisions](#appendix-c--open-questions-and-unresolved-decisions)

## 1. Purpose and scope

Build a Rust platform for acquiring human motion, learning physically feasible humanoid behavior, transferring it to different robot embodiments, and improving deployment from limited real-world experience. Humanoid robot boxing is the initial demanding application: stance, footwork, guard, strikes, avoidance, controlled contact, recovery, opponent modeling, and expressive behavior.

The platform must also support noncombat tasks without rebuilding the data pipeline, training infrastructure, or hardware integration. Boxing is a task package, not an assumption embedded in every controller.

The principal architecture is hierarchical:

1. Acquire and curate motion and robot observations.
2. Retarget human motion to a specific robot and validate feasibility.
3. Train a stable motion tracker and reusable skills in simulation.
4. Select skills using task policy, retrieval, or sparse expert routing.
5. Execute through a deterministic control runtime and hardware adapter.
6. Record selected deployment evidence, identify model mismatch, and validate adaptations before promotion.

Deliverables include reusable crates, command-line tools, declarative experiment configurations, versioned artifacts, simulator and hardware interfaces, evaluation suites, and operational documentation.

### 1.1 Relationship to existing repository contents

This repository already contains a Python stack: `training/` (PyTorch, JAX, MuJoCo MJX, Warp, Hydra, MLflow, ONNX export) and `unitree/` (CycloneDDS configuration and a Unitree SDK connector). Section 2 excludes every one of those runtimes from the default implementation. The specification therefore MUST state their disposition rather than silently describing a system that does not match the working tree.

| Existing asset | Disposition |
| --- | --- |
| `training/` Python training code | Prior art and behavioral reference. Not a dependency, not a build target, and not permitted in any deployable path. It MAY be read for algorithm intent and MAY be executed manually, outside CI, to produce comparison numbers recorded as `unverified` baselines |
| `training/conf/` experiment configurations | Requirements input. The task, reward and randomization intent expressed there SHOULD be migrated into the configuration schema of Section 9.3; the YAML files themselves are not the canonical format |
| `unitree/` DDS/SDK connector | Protocol reference for `robot-hardware`. Field layouts, rates and command semantics observed here are hypotheses requiring the hardware verification of Section 7.1, not a validated protocol specification |
| `test.ipynb` and notebook workflows | Scratch artifacts. Excluded from acceptance, reproducibility and artifact-lineage requirements |

Three consequences are binding. The Rust workspace of Section 6 MUST NOT import, subprocess, or link against any of the above. Numbers produced by the Python stack MUST NOT be reported as platform results without restating their provenance. Retiring or relocating this code is a migration decision requiring a decision record; until that record exists, its continued presence is a known divergence from Section 2 and is tracked as [C-01](#appendix-c--open-questions-and-unresolved-decisions), not an implicit exception.

## 2. Language and dependency policy

All project-authored executable code MUST be Rust: ingestion, numerical processing, training, GPU kernels where needed, evaluation, visualization applications, deployment, tools, and tests. There must be no required Python training environment, Python preprocessing script, Python model-export step, or shell-script application logic. TOML configuration, schemas, Markdown documentation, and data files are not programming-language exceptions.

The default implementation targets Rust-native libraries and Rust-authored compute kernels. Candidate dependencies include Burn for differentiable models, CubeCL for Rust GPU kernels, nalgebra for linear algebra, Rapier for initial rigid-body experiments, tracing for diagnostics, and egui for tools. These are candidates requiring version, license, backend, and performance validation; they are not assertions that a complete humanoid robotics stack already exists in Rust.

Appendix B incorporates the supplied Rust ecosystem survey revised v2.0, assigns concrete candidate roles, and defines dependency qualification gates. Its maturity assessments are provisional. A Rust API alone does not establish that a library or its enabled dependency closure satisfies this policy. Generated C and required Python code-generation workflows do not satisfy the Rust-only policy. Development tools with foreign engines are also identified explicitly rather than silently classified as Rust-native.

Distinguish two dependency classes explicitly:

| Class | Policy |
| --- | --- |
| Rust-authored application and algorithm code | Mandatory throughout |
| Rust-native compute and physics dependencies | Default; qualify for each required capability |
| OS, device firmware, GPU drivers | External platform dependencies; inventory them |
| C/C++ simulator, vendor SDK, ONNX runtime, or other foreign library through FFI | Excluded from the default implementation; any future exception requires an explicit scope decision and separate feature/profile |
| Python frameworks exposed through subprocesses or bindings | Excluded |

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
8. The runtimes excluded by Section 2 and Section 1.1 MUST NOT appear in the default environment. Manual execution of the Python prior art, or an approved Section 2.1 exception, lives in a separately named non-default environment whose foreign nature is visible at its call site, and the exclusion gate still runs in the default one.

The known limitation is host coverage: Nix supports Linux and macOS, and Windows only through WSL2. Which host platforms the locked environment is actually exercised on, which toolchain input supplies Rust, and whether CI runs devenv natively or inside an image built from the same lock, are open under [C-14](#appendix-c--open-questions-and-unresolved-decisions). A reproducible environment is not a hermetic build: Nix pins inputs, it does not make `cargo` fetch nothing, so dependency locking under Appendix B.6 remains separately required.

## 3. Evidence policy and corrections to the research premise

The preceding conversation is a source of requirements and hypotheses, not a verified bibliography. No live source verification is claimed in this specification. Record source URLs, title, authors, version/date, accessed date, code commit, license, evidence type, and the exact supported claim before relying on research in an implementation decision.

Required evidence states: `unverified`, `paper_verified`, `code_inspected`, `reproduced_in_sim`, and `validated_on_hardware`. A project demo, paper description, public implementation, and reproducible benchmark are separate kinds of evidence.

The states are ordered, and the order is the point: each one licenses a narrow class of claim and nothing above it. Promotion requires the admitting evidence in the following table and a dated ledger entry naming the artifact that supplies it.

| State | Admitted by | Licenses | Does not license |
| --- | --- | --- | --- |
| `unverified` | Any mention: conversation, blog post, product demo, unread citation | Scheduling an investigation | Any implementation decision, estimate or citation |
| `paper_verified` | Primary source retrieved, identity confirmed, claim located in text with version and date | Quoting the stated method and its stated conditions | Assuming the method works, reproduces, or applies to our embodiment |
| `code_inspected` | Published implementation read at a pinned commit, license recorded, dependency closure inventoried | Statements about what the implementation does and what it requires | Statements about its numerical correctness or performance |
| `reproduced_in_sim` | Our Rust implementation reaches the stated behavior on a declared fixture, multiple seeds, with the comparison budget recorded | Simulation claims within the tested operating region | Any hardware claim, and any claim outside the tested region |
| `validated_on_hardware` | Measured on the designated robot under a versioned acceptance profile, with failures and interventions reported | Claims about that robot, that profile, that envelope | Claims about other embodiments, payloads, firmware or envelopes |

Evidence does not transfer between artifacts: a reproduction of an algorithm does not promote the paper that proposed it, and a hardware validation of one bundle does not promote its successor. Demotion is also required — if a dependency version, robot profile, firmware or acceptance profile changes, every claim resting on it returns to the highest state its surviving evidence still supports. Cross-simulator agreement is recorded as a result attached to a `reproduced_in_sim` entry; it is not a state of its own.

Specific corrections that constrain this design:

- A normalized humanoid skeleton does not make a learned torque or joint-target policy embodiment independent. Dynamics, sensing, action semantics, and morphology remain robot specific.
- A semantic match does not establish dynamic feasibility or a safe transition between policies. Even two stable policies can become unstable when switched or blended.
- Sparse activation reduces active parameter count; it does not guarantee lower wall-clock latency than a small dense policy. Routing, transfers, dispatch, synchronization, and cache misses must be measured.
- Asynchronous retrieval removes blocking work from a control thread, but still introduces decision age and potential stale intent. It is not zero-latency decision-making.
- Retrieval routing does not eliminate selection collapse, skill starvation, or dataset imbalance. Adding an expert can change routing and transition behavior and requires regression evaluation.
- A unit-norm latent vector bounds its representation, not joint torques, balance, or contact forces.
- Cross-simulator agreement is useful evidence, not a guarantee of hardware success. Real contact calibration remains necessary.
- No fixed percentage of motion transferability, minimum number of real rollouts, adaptation duration, or fraction of success attributable to retargeting is assumed.
- Do not infer proprietary Unitree training architecture, exact datasets, curriculum, or autonomous capabilities from public demonstrations. Separate documented features from engineering reconstruction.
- PHC commonly refers to Perpetual Humanoid Control. Mink is a differential inverse-kinematics library built around MuJoCo, not inherently a GPU batched SMPL pipeline. These names must not become misleading implementation requirements.
- The earlier residual-policy sketch requires privileged simulated state at inference and has no derived correction targets. It is not an executable adaptation algorithm. Simulator correction and robot-side action correction must have distinct contracts.

### 3.1 Research intake register

Maintain the following leads; verify identity, claims, implementation availability, and licensing before scheduling a reproduction.

| Lead from the conversation | Intended investigation | Initial disposition |
| --- | --- | --- |
| RoboStriker, arXiv:2601.22517 | Motion tracking, latent distillation, competitive self-play | Verify paper and exact architecture, dimensions, rates, losses, and release status |
| Unitree G1 boxing / Iron Fist King | Public product behavior, hardware constraints, demonstrations | Treat training internals as unknown unless directly documented |
| GMR, PHC, H2O/OmniH2O, ExBody, ProtoMotions | Retargeting, imitation, whole-body control | Reference algorithms; Rust implementations required |
| ASAP, arXiv:2502.01143 | Learning simulator/action mismatch from real trajectories | Verify where correction operates during training and deployment |
| SPiRL, NPMP, ControlVAE, PhysicsVAE, NCP | Skill priors and latent controllers | Establish a simple reproducible baseline before extensions |
| STRAP, arXiv:2412.15182; SMOSE | Retrieval and sparse expert control | Verify objective, retrieval unit, benchmark and actual latency evidence |
| ARLI, DARP, ReGIL, MABR, COLLIE, SafeFlow, InstantMimic, H-Zero, R²S², LATENT | Delay-aware control, retrieval, reusable representations, transfer | Unverified leads; no core dependency |
| PAMoR, SocialHumanoid, StreamTalk, LiveGesture, PersonaGestor, Animus | Optional personality and expressive motion | Unverified leads; separate from motion safety and baseline delivery |
| “JitRL”, “Learn-to-Teach 2507.XXXXX”, claimed percentage improvements | Possible topics rather than usable citations | Require an identifiable primary source; placeholder identifiers are rejected |

A failed verification is a result: retain a correction record instead of preserving an unsupported claim.

## 4. Operating modes and boundaries

Support deterministic offline replay, simulation training, simulation evaluation, hardware shadow mode, supported hardware motion, free-standing noncontact motion, and separately enabled robot-to-robot contact experiments. Shadow mode consumes live sensor data and computes actions without transmitting motor commands. Teleoperation and scripted controllers are first-class baselines.

Each mode has explicit permitted actions, hardware interlocks, timeout behavior, and artifact requirements. The mode is a property of the running process, not a configuration comment: it MUST be recorded in every episode record, MUST be reported by the supervisor on request, and MUST NOT change without passing back through the entry preconditions of the target mode.

| Mode | Actuation authority | Entry preconditions | Required interlocks | Loss-of-input or fault response |
| --- | --- | --- | --- | --- |
| `offline-replay` | None; no device attached | Artifact hashes resolve; bundle schema matches recorded observations | None | Abort with a diagnostic; never fabricate missing samples |
| `sim-train` | Simulated only | Seeds, config and data manifest pinned | None | Fail the run; partial results are labeled as such |
| `sim-eval` | Simulated only | Frozen bundle; evaluation seeds disjoint from training | None | Fail the suite rather than reporting a partial pass |
| `hw-shadow` | None; commands computed, logged, and discarded at the adapter | Transmit path disabled and the disable verified by loopback readback; e-stop armed | E-stop; transmit-disable assertion checked every tick | Continue computing; record the fault; never enable transmit as a fallback |
| `hw-supported` | Motion with the robot mechanically supported | Joint sign/offset/units verified per Section 7.1; reduced torque and velocity envelope loaded | E-stop; support-fixture present; reduced envelope enforced by the supervisor | Robot-specific hold or controlled stop per its safety profile |
| `hw-freestanding-noncontact` | Full motion, no contact targets | `hw-supported` qualification passed; area cleared; exclusion zone enforced | E-stop; area interlock; envelope; watchdog independent of the policy process | Validated fallback of Section 10.4; falling is a designed outcome, not a fault path |
| `hw-contact` | Motion against an instrumented fixture or robot | Separate enablement; noncontact qualification passed; scoring validated; contact energy budget set | All of the above plus per-target contact permission and an independent match logger | Terminate the bout, not just the skill; record the full pre/post-event buffer |
| `teleop` | Operator-commanded, mode-scoped | Same preconditions as the hardware mode it runs within | Same interlocks; operator presence control where the robot requires one | Command expiry stops motion; stale operator input is not held |
| `scripted` | Deterministic scripted controller | Same preconditions as the hardware mode it runs within | Same interlocks | Same as the enclosing hardware mode |

Teleoperation and scripted control are modifiers on a hardware mode, not escapes from it: they are subject to the same envelope, supervisor and artifact requirements, and an operator MUST NOT be able to command outside the envelope that a learned policy would be clamped to.

Autonomous contact with people is outside this project's boxing scope. Contact experiments involve instrumented robots or test fixtures in a controlled area. This boundary affects task definitions, target identification, and deployment interlocks. A human entering the area is an exclusion-zone violation handled by Section 12.3, never a target.

## 5. Timing and execution model

The following are initial design targets, not measured guarantees. Each robot profile must establish its actual rates and deadlines.

| Loop | Initial target | Execution requirements |
| --- | --- | --- |
| Embedded actuator servo | 500–1,000 Hz where supported | Usually firmware owned; document host/firmware split |
| Host state estimation and command supervision | 500–1,000 Hz if hardware supports it | Bounded work, no blocking external dependencies |
| Learned whole-body policy | 50–200 Hz; start at 100 Hz | Resident model, preallocated tensors and buffers |
| Fast recovery / contact reflex | Every host control tick | Can interrupt the normal skill schedule |
| Tactical router and opponent forecast | 10–50 Hz, task dependent | Bounded local computation; stale-output rejection |
| Semantic search and repertoire planning | 1–10 Hz | Asynchronous; produces proposals, not motor commands |
| Training and adaptation | Offline or isolated background process | Never contend with the deployed control process |

At an initial 100 Hz policy rate, the system has a 10 ms period. Establish a hardware-specific budget for sensor acquisition, estimation, inference, supervision, and command transport. Initial engineering objective: p99.9 policy-path completion below 8 ms under representative load, with remaining margin, and no deadline overruns during a specified qualification run. This is a target to validate, not a hard-real-time proof.

Measure sensor-to-command age separately from compute duration. Report p50/p95/p99/p99.9/max, deadline misses, input age, clock uncertainty, and fallback events. A hard-real-time claim additionally requires bounded worst-case execution and scheduling analysis for the selected OS, transport, and devices.

These quantities are distinct and a report MUST NOT collapse them:

| Quantity | Measured from | Measured to | Typical failure it exposes |
| --- | --- | --- | --- |
| Compute duration | Policy path entry | Command ready | Inference regression, allocation, contention |
| Sensor-to-command age | Sensor sampling instant | Command accepted by the actuator interface | Transport delay, buffering, clock skew, missed sampling |
| Decision age | Production of the last accepted high-level proposal | The tick on which it is still being applied | Stale intent from asynchronous retrieval or planning |
| Deadline margin | Command ready | Period boundary | Tail growth before it becomes an overrun |

A system can hold compute duration well inside budget while sensor-to-command age drifts out of it. Sampling instants come from the device where the device provides them; where it does not, the record MUST say that the timestamp is an arrival time and carry the resulting uncertainty, because an arrival time silently relabeled as a sampling time corrupts every delay model trained on it.

The control path MUST NOT perform file reads, network database queries, model downloads, heap allocation after activation, locks with unbounded contention, GPU compilation, logging flushes, or model loading. Tokio or similar asynchronous runtimes may serve orchestration, but not the deadline-critical loop.

Rayon work stealing and general-purpose telemetry subscribers are not presumed suitable for this loop. Use dedicated bounded resources for control and separate pools for rollout/data processing. Capture preallocated timestamp/event records in the loop and construct/export telemetry off-thread; benchmark enabled instrumentation against a disabled baseline.

Use monotonic timestamps, fixed-capacity queues, explicit freshness limits, CPU affinity and scheduling configuration where available, and a watchdog independent of the policy process. Log overflow must never block actuation; emit dropped-record counts. Safety-relevant incidents require a reserved event channel and bounded pre/post-event recording buffer.

## 6. Proposed workspace and component responsibilities

These are planned logical boundaries; they do not describe or depend on existing repository contents.

| Crate or package family | Responsibility |
| --- | --- |
| `robot-types` | Units, frames, time, joint IDs, typed observations/actions, schemas |
| `robot-model` | Kinematic tree, inertias, actuator limits, geometry, sensor placement |
| `motion-io` | Motion import/export and synchronized capture adapters |
| `motion-process` | Resampling, filtering, contacts, segmentation, augmentation |
| `motion-retarget` | Robot mapping, constrained IK, temporal optimization |
| `motion-validate` | Kinematic and simulated feasibility reports |
| `robot-sim` | Simulation interface, Rust physics backend, batching, randomization |
| `robot-tasks` | Tracking, balance, reach, footwork, boxing and perturbation tasks |
| `robot-learn` | Rollouts, autodiff, PPO baseline, losses, optimizers, checkpoints |
| `robot-skills` | Latent models, expert bank, transition contracts, retrieval index |
| `robot-adapt` | Identification, adaptation encoders, correction models, calibration |
| `robot-perception` | State estimation, contacts, opponent tracking and uncertainty |
| `robot-control` | Deterministic runtime, inference, arbitration, supervision |
| `robot-hardware` | Device traits, protocol adapters, replay and loopback devices |
| `robot-eval` | Benchmarks, statistical reports, regression gates |
| `robot-artifacts` | Content hashes, manifests, lineage, signing and promotion |
| `robot-persona` | Optional identity and bounded expressive intent |
| `robot-cli`, `robot-viewer` | Experiment tooling, inspection, synchronized playback |

Training and deployment share observation/action schemas and tested normalization logic. Deployment does not depend on training autograd code. Avoid a plugin design that permits arbitrary dynamic code loading in the control process: build-time Rust implementations are the default, and optional experimental workers run outside that process.

Use Appendix B.3 to select initial dependency candidates for these crate families. Candidate selection does not authorize dependencies that conflict with Section 2.

## 7. Robot and artifact contracts

### 7.1 Embodiment description

Each versioned robot profile describes:

- Stable joint/link names and explicit indices, units, axes, limits and zero offsets.
- Link masses, center of mass, inertia, collision meshes and available articulated degrees of freedom.
- Motor transmission, command interface, torque-speed envelope, gains, saturation, latency, thermal limits and uncertainty bounds.
- Sensor frames, timestamps, rates, covariance, calibration and missing-data behavior.
- Root convention, gravity, contact surfaces, feet and hand geometry, payload and glove changes.
- Firmware/protocol version, watchdog semantics, command expiration, emergency stop interface, and allowed operating modes.

URDF import is an initial interchange target, not a complete dynamics specification. Store missing actuator/contact metadata separately. Unsupported model features must be rejected explicitly rather than silently dropped.

### 7.2 Canonical motion record

Store `schema_version`, source hash, source license, subject/session ID, skeleton/model hash, frame timestamps, validity masks, root pose, named joint rotations or positions, optional velocities, contact labels with confidence, task labels, and provenance for every transformation.

Use SI units, a documented right-handed coordinate system, explicit world/body frames, and named quaternion ordering. Prefer typed quaternion serialization with named components over ambiguous four-element arrays. Euler-angle conventions are source-specific; never assume every BVH file uses the same rotation order.

Human skeleton motion, parametric human-body parameters, and robot joint trajectories are distinct record types. BVH, C3D, and SMPL-family data are not interchangeable. SMPL assets and source datasets may have independent license restrictions.

### 7.3 Robot episode record

Record monotonic sensor timestamps, arrival times, synchronized wall-time mapping where needed, raw and estimated state, uncertainty, desired and measured joint motion, commanded actions, actuator acknowledgments if available, contacts, temperatures, power, policy/expert IDs, transition state, task outcome, and supervisor interventions.

Store both raw commands and applied/clamped commands so training cannot confuse policy intent with actuator input. Record validity and provenance for estimated torque; motor current is not automatically calibrated joint torque. Preserve observation and action history needed to model delays.

### 7.4 Policy bundle

A deployable bundle contains model weights in a Rust-loadable versioned format, graph/architecture version, observation schema, normalization statistics, action semantics and scaling, recurrent-state specification, rates, robot-profile compatibility, training data lineage, safety-envelope assumptions, transition contracts, benchmark results, and checksum/signature.

ONNX is an optional interchange format only if a qualifying Rust-native path supports the required operators. It is not the default training/export dependency. Unsupported operators must fail before deployment.

### 7.5 Storage and splits

Use chunked binary arrays for high-rate numerical data and columnar records for searchable metadata; evaluate Arrow/Parquet implementations for Rust. TOML manifests refer to content-addressed blobs. Never require loading an entire corpus into memory. Do not load executable pickle objects; supported legacy formats need a constrained Rust parser or a documented non-pickle export from the data provider.

Select IPC and persistent-record encoding separately using Appendix B's serialization candidates. Every wire/archive format requires explicit versioning, size limits, framing, endianness/alignment rules where applicable and validation before access. Serialization is not a transport or a schema-migration strategy. Durable episode records must remain readable across supported software upgrades.

Split datasets by subject, recording session, and original motion before generating windows or augmentations. For competitive learning, separate evaluation opponent pools. Track license eligibility, retention policy, consent where relevant, and dataset revisions.

## 8. Motion acquisition and retargeting

### 8.1 Capture inputs

Deliver file-based BVH ingestion first, followed by C3D and supported numerical human-body formats. Add live capture through documented Rust protocol adapters for optical or inertial systems. Keep timestamps and confidence through every transform.

Video-to-motion is an optional later module: require a model that can be executed through the permitted Rust stack and validate camera calibration, root translation, occlusion and contact ambiguity. A pose estimated from ordinary video is not automatically a dynamically valid reference.

Candidate corpora include AMASS and its source datasets, CMU boxing-related sequences, LAFAN1, the martial-arts motion dataset referenced as PMC7813879, and Bandai Namco motion collections. Confirm exact action coverage, access terms, commercial/training permissions, and redistribution rules before ingestion. Do not assume BEAT-style co-speech corpora contain useful boxing demonstrations. Existing robot-retargeted corpora are tied to their source embodiment and conventions.

### 8.2 Processing pipeline

1. Inspect format, license, skeleton, rates and coordinate conventions.
2. Preserve immutable originals and assign content IDs.
3. Correct axes/units, validate finite values and timestamps, and mark gaps.
4. Filter and resample with documented phase delay; preserve real strike-speed characteristics.
5. Estimate ground plane and contact intervals with uncertainty.
6. Segment into stance, locomotion, strike, defense, recovery and transition windows.
7. Retarget to target geometry with contact, joint-limit and self-collision constraints.
8. Validate temporal smoothness, velocity/acceleration, torque estimates and simulated tracking feasibility.
9. Package accepted references and retain rejected examples with machine-readable reasons.

### 8.3 Retargeting requirements

Implement weighted task-space IK followed by temporal refinement. Support end-effector positions/orientations, posture priors, planted-foot constraints, joint limits and self-collision penalties. Preserve task intent, such as punch direction and target-relative reach, rather than matching all human coordinates equally.

Qualify a Rust-native constrained solver before selecting differential-QP IK or whole-body optimization. Measure setup, updates, solve time, allocations, residuals and failure detection on representative full-body constraints. Runtime uses require bounded iterations/deadlines and a validated response to infeasibility or timeout. MPC and constraint filters are optional control methods, not automatic substitutes for a learned tracker or proof of safety.

Morphology changes may require altering stance, stride, punch amplitude, torso rotation and timing. Motion time scaling is constrained by actuator and task requirements; it cannot repair every infeasible reference. A validation report must separate kinematic error, dynamics violations, contact consistency and task success.

No universal fidelity percentage is a release criterion. Measure endpoint error normalized by body size, root drift, planted-foot slip, joint-limit violations, torque-speed violations, and downstream tracking success per skill and embodiment.

## 9. Simulation, training and methodology extension

### 9.1 Simulator qualification

A useful humanoid simulator needs articulated dynamics, joint actuators, stable foot contact, collision filtering, configurable integration, sensor delay/noise, perturbations, resets, reproducible seeds and sufficient rollout throughput. A generic rigid-body engine is only a starting point.

Qualify the Rust backend on pendulum, actuator step response, static standing, foot slip, push recovery and controlled impact benchmarks. Separate physics timestep, actuator update rate and policy timestep. Record numerical sensitivity and the operating region where the model is credible.

Start with CPU vectorized environments. GPU batching is a separate engineering deliverable and cannot be assumed from GPU neural-network support. Estimate training duration from measured environment steps per second and required experiment sizes before expanding scale.

Qualify Rapier's existing multibody-joint and collision capabilities before proposing a replacement articulation layer. Compare credible alternatives, including ferromotion modules if their reported implementation can be verified, using Appendix B.4. Foreign-engine cross-validation remains optional and requires the separate scope decision in Section 2; this specification does not grant that exception.

### 9.2 Baseline training

Implement PPO with generalized advantage estimation, bounded actions, observation normalization, checkpoint/resume, recurrent policy support when needed, and explicit handling of termination versus time-limit truncation. An asymmetric critic may use simulator-only state; exported actors must pass an observation audit proving no privileged inputs remain.

Train balance and locomotion first, then target-specific motion tracking. Rewards and costs should cover task completion, reference tracking, balance, energy, action smoothness, joint/actuator constraints, and undesirable contacts. Log reward terms independently to expose reward exploitation. Persistent torque violations cannot be hidden by a good aggregate return.

Domain randomization must be calibrated and sensitivity-driven. Include plausible mass/inertia, friction, motor response, backlash where relevant, delay, sensor bias and contact parameters. Reserve held-out disturbances and parameter combinations for evaluation.

### 9.3 Algorithm interface

Every algorithm supplies a manifest and implementations for initialization, observation requirements, action contract, experience collection, update, checkpoint, export, and evaluation. Configuration includes compute budget, seeds, prerequisites and expected scientific claim.

Support on-policy rollout buffers initially; add replay only for algorithms that need it. A shared learner interface must not pretend that PPO, offline imitation, supervised distillation, model identification and self-play have identical update semantics.

Research intake workflow: verify source → extract executable assumptions → implement minimal Rust baseline → reproduce a small benchmark → compare fixed budgets → promote or reject with evidence. Pin paper versions and commits. Imported mathematical ideas must have ablation tests and separate feature flags.

### 9.4 Hierarchical skills and competitive training

After tracking works, compare:

- A single compact tracker conditioned on skill ID or reference motion.
- A latent-conditioned tracker with learned encoder/prior and distillation losses.
- A bank of specialized compact experts with shared observation/action contracts.

Use the verified RoboStriker methodology, if available, as one candidate for latent competitive learning. Do not fix a 32-dimensional unit sphere, exact control frequency, or a particular self-play algorithm solely from the conversation.

Boxing self-play requires robust hit attribution, contact filtering, target zones, stance/fall rules, episode termination, ring boundaries and task-specific force/energy budgets. Validate scoring against scripted scenarios before optimizing it. Maintain historical opponent snapshots, varied scripted opponents, held-out evaluation opponents and anti-cycling measurements. Victories against the latest self are insufficient evidence of progress.

## 10. Retrieval-indexed policy bank and low latency

### 10.1 Two kinds of similarity

Semantic similarity organizes intent: “jab”, “retreat”, “guard”, “celebrate”. Physical compatibility determines what may execute now: stance, contact mode, joint velocity, available support, reach, actuator health and opponent-relative geometry.

The retriever therefore produces candidate skills; a state-conditioned feasibility filter and fast router select executable candidates. Text embeddings may be computed offline with a Rust model. Start with structured skill tags and numerical features, and add learned embeddings only when they improve a measured outcome.

For small libraries, benchmark exact vector search in Rust before adopting approximate nearest-neighbor indexing. A remote vector database is not part of the control path.

### 10.2 Skill descriptor and execution state

Each skill has semantic tags, embedding/version, embodiment compatibility, initiation set, termination condition, expected duration, contact assumptions, supported disturbances, inference memory requirement and outgoing transition contracts.

The runtime tracks current skill, phase, active contacts, last accepted proposal, proposal timestamp/expiry, transition status and emergency override. Candidate selection must include uncertainty and out-of-distribution detection. Log rejection reasons and routing frequency.

### 10.3 Memory and scheduling

Safety, recovery and active working-set experts stay resident. Stage optional experts asynchronously into a bounded cache; validate schema, checksum, normalization and device memory before activation. Cold-load failure or late retrieval retains a compatible current controller or invokes the robot-specific fallback.

Switch at a defined control boundary using generation-tagged immutable handles. Recurrent state must be reset or transformed by an explicit contract. No expert may be activated while its input/output schema or state is incompatible. Runtime hot swapping is tested separately from offline model loading.

### 10.4 Transitions and fallback

Use contact- and phase-aware initiation guards, hysteresis, minimum dwell times where appropriate and interruptible transitions. Recovery can preempt dwell times. Initial implementation should favor hard selection at validated transition states or a shared conditioned tracker.

Do not assume weighted averaging of motor commands is safe. Blending requires explicit transition training and testing, or a validated constrained controller. Guard posture is not a universal fallback when already falling; fallback behavior depends on state and hardware, and may involve controlled stopping, support seeking or a configured protective response.

### 10.5 Required comparison

Compare dense, conditioned, retrieval-only, learned-router and hybrid designs at equal hardware and training budgets. Report repertoire coverage, task success, transition failures, disturbance recovery, active parameters, resident memory, energy where measurable, latency tails and decision age. Measure retrieval amortization and cold-cache behavior explicitly.

Parallel expert training is useful but not necessarily independent: shared transitions, normalization, training distributions and safety assumptions couple the experts. Running several experts simultaneously may increase single-step latency on small devices.

## 11. Transfer and data-efficient improvement

### 11.1 What can transfer

Human motion representations, task descriptions, motion segmentation, skill labels, strategy structure and some learned priors can transfer across robots. Low-level control typically requires target-specific retargeting and simulated training or adaptation. Cross-embodiment policies require explicit morphology conditioning, joint masks/graphs, multiple training embodiments and held-out robot evaluation.

The default transfer path is:

1. Train or curate generalized human motion representations.
2. Build and calibrate the target robot description.
3. Retarget references and reject or alter infeasible motions.
4. Train a target-specific tracker in calibrated, randomized simulation.
5. Qualify its inference artifact and observation pipeline.
6. Run replay, shadow mode, supported motion and noncontact deployment.
7. Collect targeted evidence of model mismatch.
8. Update the simulator or train a bounded adapter offline.
9. Re-evaluate against the frozen baseline and promote with rollback.

### 11.2 Sensor use

Joint encoders and IMU support state estimation, contact inference and history-conditioned adaptation. Current, temperature, measured torque and foot force sensors improve identification where available. External tracking can provide independent root-motion ground truth but is optional for deployment. Opponent perception supplies tracked pose/range/velocity with uncertainty and age; a tactical policy must handle occlusion and missing detections.

The estimation module in `robot-perception` must specify orientation/error-state conventions, IMU bias dynamics, contact assumptions, covariance propagation and measurement gating. Qualify filters under timestamp jitter, dropouts, outliers, slip and poorly conditioned observations. Compare observable-state error and covariance consistency against analytic or synthetic truth; no universal percentage-divergence threshold is assumed. Autodiff Jacobians are optional and require validation, as do hand-derived Jacobians.

Closed-loop control uses sensor feedback every tick without changing model weights. This is distinct from online adaptation of latent state, parameter estimation and gradient-based policy updates.

### 11.3 Limited-data adaptation strategies

| Method | Data / deployment contract | Principal limitation |
| --- | --- | --- |
| System identification | Time-synchronized commands and measured responses; update simulator parameters offline | Parameters may not be identifiable from passive logs |
| History-conditioned adaptation | Train encoder in randomized simulation; infer latent dynamics from recent real history | Can extrapolate poorly outside training ranges |
| Simulator correction model | Learn a model explaining real-vs-sim transition mismatch; retrain policy in corrected simulation | Requires well-defined alignment and validation of correction |
| Bounded robot-side residual | Train correction offline using deployable observations only; constrain authority | Can destabilize the base policy or hide sensor errors |
| Small adapter fine-tuning | Update selected policy parameters with held-out real-data evaluation | Few parameters do not imply few required samples |

Implement identification first, then history adaptation; treat residual RL and online weight updates as later experiments. Do not require paired perfect simulated/real state at execution. Do not assume a unique target action correction can be inferred from state error alone.

Use event-triggered logging of uncertainty, slip, saturation, tracking error and contact anomalies, plus representative successful background samples. This avoids retaining every high-rate stream while limiting failure-selection bias. Select calibration motions for information gain subject to the operating envelope. Keep a held-out session for validation and compare learning curves versus real minutes, transitions and interventions.

Stop collecting or adapting when improvement saturates or uncertainty criteria are met; do not promise that 20–30 minutes is universally sufficient. Real interaction count, physical interventions, data storage and simulator compute are separate efficiency metrics.

## 12. Safety, supervision, perception and personality

A multi-kilogram humanoid that throws punches stores enough energy to injure people and destroy itself. The requirements below are the minimum structure; the numeric limits are per-robot and live in a versioned safety profile alongside the acceptance profile of Section 14.

### 12.1 Supervisor

A supervisor independent of strategy and personality checks observation freshness, finite commands, position/velocity/torque-speed limits, temperatures, communications health, contact permissions and current operating mode. Hardware emergency stop remains independent of the learning stack. Clamp-and-continue is not sufficient for every fault; each fault has a validated response appropriate to the robot.

The supervisor MUST be the last stage before the actuator interface, MUST be reachable on every path including teleoperation and scripted control, and MUST NOT import the policy, retrieval or personality crates. Its checks run within the control tick and are subject to the same prohibitions as Section 5: no allocation, no blocking, no logging flush. It MUST be testable in isolation against recorded and synthetic fault traces, and its rejection decisions MUST be recorded with enough context to reconstruct why a command was refused.

### 12.2 Hazard register

Maintain a hazard register per robot and per operating mode. Each entry names the hazard, its plausible causes, how it is detected, the control that mitigates it, and the residual risk accepted by whom. The following are the initial entries; they are a starting structure, not a completed analysis, and a hazard analysis for a specific robot MUST be performed before that robot first moves under `hw-freestanding-noncontact`.

| Hazard | Principal causes | Detection | Control |
| --- | --- | --- | --- |
| Uncommanded motion | Stale command held by firmware, watchdog misconfiguration, mode confusion | Command expiry counters, independent watchdog, transmit-disable readback | Short command expiry in firmware; watchdog outside the policy process; mode asserted every tick |
| Fall | Loss of balance, actuator saturation, late or failed skill transition | Attitude and contact estimation, deadline misses, recovery invocation count | Protective response from the robot's safety profile; area clear of obstacles and people |
| Excess contact energy | Reward exploitation, scoring bug, mis-scaled action, opponent model error | Impulse and force measurement, energy budget accounting, independent match logging | Per-target permissions, energy budget enforced outside the policy, padded fixtures |
| Actuator overload | Infeasible reference, torque-speed envelope violation, thermal accumulation | Temperature, current, saturation duration, torque-speed margin | Envelope enforcement in the supervisor; reference rejection in Section 8.3; thermal derating |
| Pinch and entanglement | Cables, gloves, straps, support fixtures within swept volume | Pre-run checklist, collision geometry including attachments | Swept-volume check with attachments modeled per Section 12.4 |
| Human enters the area | Procedural failure, unexpected access, observer approach | Area interlock, observer role, perception detection as a secondary signal | Immediate controlled stop under Section 12.3; perception is never the primary control |
| Loss of transport | Cable, radio, switch, host stall | Link heartbeat, arrival-time gaps, deadline misses | Firmware-side command expiry; the robot must be safe when the host disappears |
| Wrong artifact deployed | Bundle/profile mismatch, firmware change, payload change | Compatibility validation at load; checksum and signature | Refuse to activate; last-known-good bundle available immediately |

A hazard whose only control is "the policy learned not to do that" is not controlled. Controls MUST be independent of the learned component they constrain.

### 12.3 Area, access and human exclusion

Hardware motion beyond `hw-shadow` requires a defined physical area with controlled access, a nominated operator holding the emergency stop, and no person inside the robot's reachable volume plus a fall margin. Entry during motion is a stop condition, not a task input. Perception-based human detection MAY contribute as a secondary signal; it MUST NOT be the primary control, because the perception stack is exactly the component whose failure modes this control exists to survive.

Emergency stop MUST remain functional with the host powered off, the policy process hung, and the transport disconnected. Test this, rather than assuming it from a wiring diagram.

### 12.4 Equipment, payload and contact envelope

Account for gloves, padding, cables and payloads in collision geometry and dynamics. Match rules and allowed contact envelopes are configuration, with separate permissions for motion demonstrations and contact trials. A change of glove, padding, payload or firmware changes the dynamics and the swept volume: it invalidates the affected qualification and requires requalification under Section 11 before the robot returns to its previous mode.

### 12.5 Personality

Personality is optional and operates through bounded intent: posture style, cadence, idle motion, celebrations, speech timing and tactical preferences. Persistent identity and opponent memory remain outside the motor loop. Expressive actions use the same validated skills and transitions; they cannot override balance recovery or supervisor decisions. Test whether style is recognizable without degrading task and stability metrics.

Expression MUST NOT widen any envelope, extend any dwell time against a recovery request, or introduce a motion that has not passed the same validation as a task skill. A celebration is a skill with a transition contract, not an exemption.

## 13. Reproducibility and operational requirements

Every run records source revision, compiler/toolchain, the `devenv.lock` revision that supplied it, dependency lock, hardware/driver details, seeds, data hashes, robot model, simulator configuration, reward definition and all model versions. A run whose environment cannot be named is not reproducible, whatever else it recorded. Determinism is a documented mode and tolerance, not an assumed property of GPU execution.

Provide restartable training, atomic checkpoints, bounded storage quotas, artifact retention, structured metrics and replayable failure episodes. Production bundles are immutable. Promotion includes compatibility validation, evaluation report and an immediately available last-known-good artifact.

Treat downloaded datasets and model files as untrusted input: enforce size limits, reject malformed dimensions, fuzz parsers and verify checksums. Restrict command-network access and authenticate remote control where the transport allows it. This is operational integrity, not a dependency on a cloud control plane.

## 14. Acceptance and evaluation matrix

Each row states the evidence that closes it and the section that defines the requirement. A row is closed by an artifact in the evidence ledger, not by an assertion in a report.

| ID | Area | Requirement source | Required evidence |
| --- | --- | --- | --- |
| A-01 | Rust constraint | §2, §1.1 | All authored executable source is Rust; dependency audit identifies native components and default build excludes foreign training/simulator runtimes; the CI exclusion gate of §2.1 passes |
| A-02 | Dependency promotion | B.6 | Appendix B gates pass for the selected versions, features and target platforms; maturity claims and enabled transitive dependencies are recorded |
| A-03 | Evidence discipline | §3 | Every claim in a release report carries an evidence state and ledger reference; demotions after version or profile changes are recorded |
| A-04 | Parser robustness | §7.1, §8.1, B.2.11 | Motion and model importers pass malformed-input and fuzz checks, including units, rotation order and unsupported URDF features |
| A-05 | Optimization | §8.3, B.2.8 | Qualified Rust solver meets full-body constraint residual, resource and timing profiles; infeasibility/timeout behavior is tested |
| A-06 | Estimation | §11.2, B.2.9 | Filter error, covariance consistency and recovery are measured under sensor faults and conditioning stress against declared thresholds |
| A-07 | Telemetry | §5, B.2.10 | Enabled/disabled instrumentation comparison, bounded queues, drop counters and exporter-failure isolation pass |
| A-08 | Encoding and parallelism | §7.5, B.2.12 | Representative-message benchmarks, malformed input/migration checks and sequential/parallel rollout comparisons pass |
| A-09 | Cross-engine comparison, if authorized | §2.1, §9.1 | Publish benchmark divergence and configuration details; agreement is recorded as evidence, not hardware validation |
| A-10 | Data correctness | §7.2, §8.2 | Known motion fixtures preserve units, frame transforms, joint names, timestamps and contact metadata; invalid records fail explicitly |
| A-11 | Retargeting | §8.3 | Per-skill reports show geometric error, contact slip and actuator feasibility against target-specific thresholds |
| A-12 | Simulator | §9.1 | Recorded qualification benchmarks and throughput justify the selected tasks and training budgets |
| A-13 | Learning | §9.2 | PPO solves small reference tasks across multiple seeds before humanoid results are trusted |
| A-14 | Tracker | §9.2, §9.4 | Held-out motion and disturbance performance compared with scripted/PD baselines; privileged-observation audit passes |
| A-15 | Runtime | §5, §10.3 | Measured end-to-end deadline/freshness results on designated hardware under load; compute duration, sensor-to-command age and decision age reported separately; allocation and fault-injection checks pass |
| A-16 | Retrieval | §10.5 | Beats or meaningfully trades off against a compact dense baseline; every accepted transition passes compatibility checks |
| A-17 | Adaptation | §11.3 | Improvement on held-out real sessions without regressions in safety constraints; real-data learning curve and rollback demonstrated |
| A-18 | Operating modes | §4 | Mode is recorded, enforced and reported; transitions pass entry preconditions; teleoperation and scripted control are envelope-bound |
| A-19 | Safety case | §12 | Per-robot hazard register with independent controls; supervisor isolation tests; emergency stop verified with host down; area interlock exercised |
| A-20 | Hardware | §4, §12 | Mode-gated progression, watchdog and emergency-stop checks; no simulation-only result labeled hardware validation |
| A-21 | Boxing | §9.4, §12.2 | Scoring validated independently; held-out opponents and disturbance scenarios; contact tests follow noncontact qualification |
| A-22 | Environment reproducibility | §2.2, §13 | Clean-checkout `devenv shell` bootstrap and `devenv test` pass at the committed `devenv.lock`; every reported result names that revision; the default environment contains no excluded runtime |

Numeric physical thresholds depend on robot mass, reach, actuator specifications, sensing and task constraints. Set them in a versioned acceptance profile before evaluating a candidate; do not tune thresholds after observing the result. Use multiple seeds and confidence intervals for stochastic outcomes, and include negative results.

An acceptance profile is itself an artifact and MUST record: the robot profile and firmware version it applies to, the control period and timing budget, the physical operating envelope, per-metric thresholds with their justification, the seed list and run count, the measurement duration, and the date and author. Profiles are versioned and immutable once a run has been evaluated against them. Changing a threshold after observing a result produces a new profile version and invalidates the comparison, which MUST be stated rather than quietly re-baselined.

## 15. Explicit non-goals for the first release

The first release does not promise universal zero-shot robot control, production autonomous boxing, full video-to-motion reconstruction, an unverified Rust GPU simulator, arbitrary online policy learning on hardware, a language model in the servo loop, or reproduction of every named paper. It delivers the contracts and measured foundations needed to add those capabilities deliberately.

## Appendix A — Terminology, units and conventions

This appendix is definitional. It creates no requirements, but a term used in Sections 1–15 means what it says here.

### A.1 Abbreviations

| Term | Expansion | Note in this document |
| --- | --- | --- |
| ABA | Articulated-body algorithm | Forward-dynamics method |
| AMASS | Archive of Motion Capture as Surface Shapes | Corpus aggregating source datasets with independent licenses |
| ANN | Approximate nearest neighbor | Search method, never "artificial neural network" here |
| BVH | Biovision Hierarchy | Text skeletal motion format; first-release input |
| C3D | Coordinate 3D | Binary marker/analog biomechanics format |
| CBF | Control barrier function | Constraint-filter method, not a safety proof |
| CCD | Continuous collision detection | Physics-engine capability |
| DDP / iLQR | Differential dynamic programming / iterative LQR | Trajectory-optimization methods |
| EKF / UKF | Extended / unscented Kalman filter | Estimator variants |
| FK / IK | Forward / inverse kinematics | |
| GAE | Generalized advantage estimation | PPO component |
| HNSW | Hierarchical navigable small world | ANN index structure |
| IMU | Inertial measurement unit | |
| IPC | Inter-process communication | |
| MCAP | Message capture container format | Distinct from Rerun's RRD |
| MPC | Model-predictive control | Baseline or correction, not an adaptation algorithm |
| MPPI / CEM | Model-predictive path integral / cross-entropy method | Sampling-based control |
| OOD | Out of distribution | |
| PD | Proportional-derivative control | Baseline controller |
| PHC | Perpetual Humanoid Control | See the correction in §3 |
| PPO | Proximal policy optimization | Authored baseline learner |
| QP / SQP | Quadratic program / sequential quadratic programming | |
| RNEA | Recursive Newton-Euler algorithm | Inverse-dynamics method |
| RRD | Rerun recording format | Viewer artifact, never the canonical record |
| SMPL | Skinned Multi-Person Linear model | Parametric human-body model family with licensed assets |
| SRBD | Single rigid-body dynamics | Reduced model for MPC |
| URDF | Unified Robot Description Format | Interchange input, not a dynamics specification |
| WBC | Whole-body control | |

### A.2 Project-specific terms

| Term | Definition |
| --- | --- |
| Authored | Written for this project in Rust, as opposed to depended upon |
| Qualified | Having passed the Appendix B.6 gates for a named version, feature set and platform. A library is qualified for a capability, never in general |
| Control path | The code executed every control tick between sensor intake and command emission, subject to the prohibitions of §5 |
| Policy path | The subset of the control path from observation assembly through inference to a proposed action, before supervision |
| Control boundary | A tick at which switching the active controller is defined and tested; the only place expert activation may occur |
| Resident | Loaded, initialized and memory-committed before activation, so that use costs no allocation or I/O |
| Compute duration, sensor-to-command age, decision age, deadline margin | As defined in the timing table of §5; they are not interchangeable |
| Embodiment / robot profile | A versioned description of one physical or simulated robot per §7.1 |
| Policy bundle | A deployable, immutable, checksummed artifact per §7.4 |
| Skill | A named behavior with an initiation set, termination condition and transition contracts per §10.2 |
| Expert | A model implementing one or more skills, interchangeable only where schemas and contracts match |
| Transition contract | The explicit statement of which states permit a switch and how recurrent state is reset or transformed |
| Initiation set | The set of states in which a skill may legally begin |
| Operating envelope | The physical limits a robot is permitted to operate within for a given mode, enforced by the supervisor |
| Acceptance profile | The versioned threshold set a candidate is evaluated against, per §14 |
| Safety profile | The versioned per-robot limits, fault responses and hazard controls of §12 |
| Evidence state | One of the five states of §3, with the promotion rules given there |
| Shadow mode | Computing commands from live sensors with the transmit path disabled and the disable verified, per §4 |
| Locked environment | The toolchain, native inputs and development tools resolved by the committed `devenv.lock`, per §2.2. "Locally" is not an environment |
| Default environment | The devenv environment entered by `devenv shell` with no profile selection, from which Section 2's excluded runtimes are absent and the §2.1 exclusion gate runs |

### A.3 Units, frames and timing conventions

SI units throughout: metres, kilograms, seconds, radians, newtons, newton-metres, amperes, kelvin or degrees Celsius where stated explicitly. Angles are radians in stored records regardless of what a source format uses; a converted field records the source convention in its provenance.

The canonical coordinate frame, gravity direction, root convention and quaternion component ordering are fixed by decision record and recorded here once ratified; see [C-04](#appendix-c--open-questions-and-unresolved-decisions). Until then, no code may assume a project-wide convention, and every transform MUST name the frames it maps between. Quaternions are serialized with named components, never as a bare four-element array, precisely because the ordering question above is still open.

Time is monotonic within a process and stored as explicit nanosecond integers with a named clock. Wall-clock mapping is recorded separately where correlation with external systems is required. A timestamp field MUST declare whether it is a sampling instant or an arrival time, per Section 5.

## Appendix B — Rust Ecosystem Survey and Dependency Qualification (Revised v2.0)

### B.1 Provenance, maturity and corrections

This appendix supersedes the previous survey integration with the user-supplied revised v2.0, adding optimization, state estimation, observability, testing, serialization and parallelism. Its reported versions, maintenance status, paper references, performance figures and inspection claims have not been independently checked for this document. All incoming entries start at `unverified` under Section 3 until a primary-source record establishes a higher evidence state. In particular, the supplied `paper_verified` label for HULKs and `code_inspected` label for Burn are not inherited automatically. The register retains local subsection numbering for existing references; the new capability inventories follow B.2.7.

Maturity is assessed per capability and enabled backend, not simply per project. Record release history, maintainers, issue response, tests, deployment evidence, license and dependency closure. Distinguish an established general-purpose library from a qualified humanoid-control implementation. Stars, downloads and project benchmarks do not replace workload testing.

Corrections and qualifications to the supplied survey:

1. Rapier has continuous collision detection and a multibody-joint API. The blanket claims that it lacks CCD and any articulation subsystem are incorrect. Whether its existing implementation meets floating-base humanoid contact, actuator, stability and throughput requirements remains a qualification question.
2. Rerun's native recording format is RRD; MCAP is a separate format. Evaluate MCAP import/export independently rather than calling Rerun recordings MCAP files.
3. “No Rust SMPL toolchain was found” is a survey result, not proof that none exists. Verify available implementations and asset licensing before committing to authorship.
4. A URDF parser does not itself establish physical model validity or guarantee rejection of every unsupported feature. The platform owns semantic validation.
5. Other immature RL implementations are comparison references, not numerical correctness oracles. Use analytic fixtures and independently derived checks as well.
6. Identical random seeds do not make trajectories from different physics engines equivalent. Compare matched initial conditions, parameters, inputs and observables with stated tolerances.
7. Analytic derivatives of smooth rigid-body dynamics do not imply globally exact gradients through contact-mode changes. Validate identification gradients within and across the intended operating regimes.
8. OSQP's Rust wrapper uses a C solver; generating C instead of linking a prebuilt library does not make it Rust-native. OpEn's Python-driven generation path conflicts with Section 2 even when its output is Rust. Evaluate a separately usable Rust solver core only if it needs no Python workflow.
9. `cargo-fuzz` uses the foreign libFuzzer engine; honggfuzz also has a foreign engine. Rust harnesses do not make those toolchains Rust-only. They are optional development-tool exception candidates, not mandatory dependencies under the current policy.
10. `tracing` is not inherently asynchronous or allocation-free, nor does OpenTelemetry integration automatically meet control deadlines. A subscriber/exporter can add blocking or allocating work.
11. Zero-copy archives do not imply stable Rust ABI, validated input or automatic schema evolution. Cap'n Proto and Protobuf generators may require external tools; generated Rust alone does not settle the full toolchain audit.
12. The specification has no “no hand-derived gradients” principle. Estimator feature lists and const generics do not establish numerical accuracy or reduced runtime overhead. A blanket “divergence < 0.1%” is undefined without a metric and reference.

Maintain Section 3's evidence states. Record cross-engine agreement as an evaluation result attached to a simulator reproduction; do not introduce an incomparable promotion level called `cross_sim_agreement`.

### B.2 Dependency register

#### B.2.1 Learning and numerical compute

| Candidate | Provisional maturity / role | Required qualification |
| --- | --- | --- |
| Burn / CubeCL (`tracel-ai`) | Established Rust ML candidate; default shortlist for `robot-learn` | Verify reported v0.22 and backend claims; test our policy distributions, autodiff, CPU/GPU operators, optimizer state, checkpoint and inference parity |
| `burn-onnx` | Candidate Rust-native ONNX import/code generation | Test actual operator coverage, shapes and numerical parity; do not assume arbitrary ONNX models import |
| Candle (Hugging Face) | Established alternative tensor/model ecosystem | Compare required training and inference operators with Burn; inspect selected backend dependencies |
| nalgebra | Established numerical foundation | Pin version, precision and SIMD features; validate conventions and solver behavior |
| `tch-rs`, `ort` | Bindings to libtorch / ONNX Runtime | Foreign-runtime class; excluded from default build |

The survey reports Burn features including `flex`, multiple GPU backends and fast incremental compilation. Treat these as version-specific claims to verify, not acceptance evidence. GPU driver dependencies remain external platform dependencies under Section 2.

#### B.2.2 Reinforcement learning

| Candidate | Reported capability / maturity | Decision |
| --- | --- | --- |
| `ember-rl` | Small Burn DQN/PPO/SAC implementation | Reference-only until identity, code and numerical behavior are audited |
| `thrust-rl` | Burn algorithms and separated inference; reported removal of tch | Verify current feature/dependency closure; inspect deployable-inference patterns |
| `sb3-burn` | Partial Stable-Baselines3-inspired implementation | Verify completeness per algorithm; no reliance on unfinished PPO/SAC |
| `r2l-burn` | PPO with reported Python Gymnasium environment path | Python-dependent execution is excluded; inspect algorithm code independently |
| `rl_traits` | Environment/agent interfaces with reported terminated/truncated distinction | Audit semantics before adopting traits or documenting divergence |

Retain the authored PPO baseline. Audit GAE, truncation bootstrapping, action log probabilities, entropy, clipping, advantage normalization, optimizer state and checkpoint behavior. Compare small deterministic fixtures and learning curves; matching seeds alone does not guarantee matching training across implementations.

#### B.2.3 Physics and robotics mathematics

| Candidate | Provisional maturity / role | Decision |
| --- | --- | --- |
| Rapier (`rapier3d`, `rapier3d-f64`) | Established Rust rigid-body engine, with CCD and multibody joints; humanoid workload unqualified | First simulator candidate; test existing articulation, motors, contacts, snapshots and determinism before authoring missing functionality |
| `ferromotion` family | Survey reports a young, broad pure-Rust robotics ecosystem; identity and implementation unverified here | Investigate module by module; not a mandatory dependency or architectural foundation |
| MuJoCo Rust bindings, including `mujoco-rs`, `rusty_mujoco`, `mujoco-rust`, `mujoco` | Foreign-engine interfaces; survey maintenance/version claims need checking | Excluded by default; optional cross-validation only following a separate scope decision |

The ferromotion survey reports FK, analytic Jacobians, RNEA, ABA, mass matrices, dynamics derivatives, LM IK, Paden–Kahan subproblems, trajectory optimization, contact solving, retargeting and identification. It also reports `ferromotion-learn`, calibration modules and wgpu rollout examples. Verify repository ownership, published crate/module names, licenses, versions and working code before evaluating these capabilities. Reported arm or cartpole demonstrations do not qualify humanoid contact dynamics or GPU humanoid training.

Candidate implementations of IK may use nalgebra, qualified Jacobians and a Rust QP solver. Mink remains a Python/MuJoCo algorithm reference. Select a solver only after testing feasibility detection, constraint accuracy, iteration limits and timing; LM IK and constrained differential-QP IK are not interchangeable.

Do not combine two physics solvers by casually assigning one dynamics and the other contact impulses. Any hybrid must specify state ownership, integration, impulse application and solver coupling, and pass energy/momentum and contact tests. Prefer evaluating complete candidate pipelines first.

#### B.2.4 Motion and model formats

| Candidate | Provisional maturity / role | Decision |
| --- | --- | --- |
| `c3dio` | Survey reports C3D reader/writer and biomechanics use | Primary C3D shortlist; verify institutional/maintenance claims and test analog channels, scaling, units and round-trip behavior |
| `bvh_anim` | Reported read-oriented BVH implementation | Audit source and features, including any FFI module; test per-joint channel ordering |
| `bvh_anim_parser` | Reported smaller BVH parser | Secondary comparison candidate; disagreements require a known-correct fixture |
| `c3d-rs` | Reported stale/WIP implementation | Reference-only pending maintenance review, not an unconditional abandonment claim |
| `urdf-rs` (OpenRR) | Established Rust URDF parser candidate | Pair parsing with platform semantic validation and explicit unsupported-feature rejection |
| Rust SMPL-family implementations | No qualified implementation identified in the supplied survey | Search/qualification task; plan a minimal authored path or provider numeric export if needed |

BVH remains the first-release input. SMPL support must separately specify parameter parsing, model assets, blend shapes/skinning, joint regression and optional fitting; implementing one does not imply the others. Provider export avoids unsafe pickle loading but may not eliminate the need for licensed model evaluation.

#### B.2.5 Retrieval

| Candidate | Provisional maturity / role | Decision |
| --- | --- | --- |
| Rust exact vector search | Simple measurable baseline | Default; benchmark scalar/SIMD implementation, metric and dimensions |
| `instant-distance` | Pure-Rust HNSW candidate; maintenance unverified | Qualify only when measured library scale justifies approximate search |
| `usearch`, `faiss` Rust interfaces | Bindings over foreign implementations | Excluded from default build despite Rust APIs |

Measure the exact/approximate crossover on target hardware. Include search recall, filtering, index build/update cost, resident memory, serialization and worst observed timing. A fast neighbor lookup that returns infeasible skills does not satisfy the routing contract.

#### B.2.6 Robot frameworks and transport

| Candidate | Provisional maturity / role | Decision |
| --- | --- | --- |
| HULKs `hulk` | Rust humanoid soccer architecture; survey reports competition deployment and a RoboCup paper | High-priority architectural reference; verify cited paper, platforms, timing methodology and dependency boundaries |
| `rust_robotics` (`rsasaki0109`) | Educational/reference planning, estimation and control algorithms | Audit each component; algorithm count is not production evidence |
| `humanoid-rs` | Reported early personal framework | Monitor/reference-only until implementation and validation are inspected |
| Copper | Rust robotics runtime candidate | Compare logging, scheduling and replay contracts before choosing custom orchestration |
| Zenoh | Established Rust communication ecosystem | External orchestration candidate; audit enabled transports/plugins and keep blocking operations off control path |
| `roslibrust` | Rust asynchronous ROS interoperability candidate | Verify backend/protocol coverage and mock support; Zenoh support alone does not establish compatibility with every ROS 2 graph |
| `rclrs` / `ros2_rust` | ROS 2 interface using C rcl | Foreign dependency class; requires a separate scope decision |
| `iceoryx2` | Rust shared-memory IPC candidate | Benchmark payload, ownership, recovery and latency on target; marketing latency is not a deadline guarantee |
| `dora-rs` | Rust dataflow orchestration candidate | Audit runtime/dependency closure and graph overhead; Python-capable ecosystem does not require selecting Python nodes |

HULKs may inform real-time data flow, fall recovery, configuration and testing. Do not transfer its tuned parameters or infer whole-stack Rust purity solely from Rust application code. Verify the supplied NAO/Booster K1 and publication details before citing them as evidence.

#### B.2.7 Visualization and storage

| Candidate | Provisional maturity / role | Decision |
| --- | --- | --- |
| Rerun | Established Rust visualization/logging ecosystem | Viewer shortlist; use Rust SDK, inspect enabled native dependencies, and isolate logging from control |
| `urdf-viz` | OpenRR model inspection candidate | Evaluate supported models and control interface for inspection only |
| egui / wgpu | Established Rust GUI/graphics foundations | Retain for custom tools; audit platform/backend feature closure |
| Polars / Arrow Rust | Established data processing candidates | Evaluate chunked metadata/array workflows and selected dependencies; no Python runtime required by our application |
| Rust MCAP tooling | Separate episode interchange candidate | Qualify independently of Rerun's RRD storage, including compression dependencies |

Preserve canonical episode artifacts independently of viewer-specific recordings. Playback and training must not depend on the continued availability of a GUI application.

#### B.2.8 Optimization solvers and model-predictive control

| Candidate | Reported capability / maturity | Decision |
| --- | --- | --- |
| `clarabel` | Rust conic optimization solver supporting convex QPs | First Rust-native QP shortlist; inspect selected linear algebra features and test our sparse constraints without requiring ferromotion |
| `pounce_qp` | Survey reports sparse parametric active-set QP, warm starts and indefinite-Hessian handling | Unverified candidate; confirm supported problem class and convergence guarantees before any MPC/SQP use |
| `simple_qp` | QP modeling layer with multiple backends | Audit actual backend and expression/setup costs; a Rust frontend does not make its backend Rust-native |
| `osqp` | Rust wrapper around established C OSQP; reported version/downloads unverified | Foreign class, excluded by default; neither C code generation nor build-time linkage creates an exemption |
| `optimization_engine` (OpEn) | Rust optimization core with a Python-driven problem-generation workflow | Investigate a direct Rust-only API separately; exclude required Python generation, including offline build steps |
| `tinympc-rs` | Reported embedded/no_std linear MPC, ADMM and const-generic horizons | Verify implementation language, problem class, bounds and maintenance; do not infer purity from the name |
| `ferromotion-control` | Reported MPC, iLQR/DDP, MPPI, CEM, centroidal/SRBD models, WBC and CBF-QP filters | Verify each module and solver dependency; no umbrella adoption or claimed safety guarantee |

Benchmark qualified Rust candidates on representative dimensions, sparsity, contact modes and ill-conditioned/infeasible cases. Include setup/update time, cold and warm solve time, peak memory, allocations, primal/dual residuals and deadline exits. Convex-QP success does not establish general nonconvex optimization correctness. Analytic IK may be useful for a restricted subproblem but is not a drop-in fallback that preserves full-body constraints.

MPC can provide a baseline, reference generator or constrained correction. It is not itself a residual-policy training algorithm or evidence of sample-efficient adaptation. Define model identification, controller state and deployment data requirements separately.

#### B.2.9 State estimation

| Candidate | Survey-reported scope | Qualification decision |
| --- | --- | --- |
| `multicalc` | EKF/UKF/error-state, attitude and particle filters; Lie groups and autodiff | Broad candidate, identity/features/maintenance unverified; compare actual humanoid IMU/contact formulation before selecting |
| `numeris` | EKF/UKF/square-root and cubature variants, smoothing, some no_std support | Alternative pending code audit; test conditioning and covariance behavior |
| `typed_kalman` | Typed dimensions, linear/square-root/EKF variants and Joseph update | Inspect numerical and dimension contracts; type safety is not estimator consistency |
| `kfilter` | Reported no_std Kalman/EKF/UKF implementation | Minimal candidate; verify supported models and fault behavior |
| `ferromotion-control` | Reported filters, complementary estimator and momentum observer | Evaluate independently of dynamics adoption; an observer is not a complete floating-base estimator |
| `rust_robotics_control` | Reported estimation/control algorithms | Verify canonical identity and relationship to `rust_robotics`; reference-only until qualified |

No primary filter library is selected solely on advertised breadth. Test rotation conventions, bias estimation, covariance positive semidefiniteness, observable-state accuracy, innovation consistency and recovery under dropouts/outliers/contact switches. Use matched models and independent ground truth; finite differences or analytic fixtures cross-check Jacobians. Keep state estimation within `robot-perception` unless a later architecture decision justifies extracting a crate.

#### B.2.10 Observability and instrumentation

| Candidate | Role / provisional maturity | Qualification decision |
| --- | --- | --- |
| `tracing`, `tracing-subscriber` | Established Rust structured instrumentation | Preferred orchestration instrumentation; validate subscribers and keep formatting/export off control thread |
| `opentelemetry`, SDK and exporters | Standard distributed telemetry ecosystem | Verify stability per selected signal/version; optional off-thread integration, no cloud/exporter dependency for control |
| `tracing-opentelemetry`, `opentelemetry-appender-tracing` | Trace/span bridge and log bridge respectively | Distinguish tracing from log export; audit enabled versions and overhead |
| `prometheus` / `prometheus-client` | Rust metrics exposition candidates | Resolve exact crate/repository; compare cardinality, histogram/exemplar support and maintenance rather than conflating clients |
| `rosql` | Survey reports ROS-aware queries, CLI/server/WASM | Unverified post-mortem analysis candidate; not a required runtime dependency |
| dora observability tools | Reported topic inspection, resource metrics and distributed traces | Verify commands/features and actual setup; do not promise zero-configuration observability |

Use bounded timestamped event records, sequence numbers and drop counts at runtime boundaries. Off-thread workers aggregate metrics and construct spans with robot/episode/policy correlation IDs. Bound label cardinality and retention. Test full queues, exporter disconnection and telemetry storms, and compare latency distributions with instrumentation disabled. External collectors such as Tempo or Prometheus are optional infrastructure, not part of a Rust-native controller or a prerequisite for local reports.

#### B.2.11 Testing, fuzzing and fault injection

| Candidate | Role | Decision |
| --- | --- | --- |
| Rust property-test frameworks, e.g. `proptest` | Generated structured cases and shrinking | Default shortlist subject to dependency audit; add bounded Rust mutation/replay harnesses for parser robustness |
| `cargo-fuzz` / `libfuzzer-sys` | Coverage-guided fuzzing with libFuzzer; typically nightly/sanitizer toolchain | Foreign-engine development tool; requires a separate exception before adoption, not mandated by this revision |
| `honggfuzz` / Rust bindings | Feedback-driven foreign fuzz engine | Same exception boundary; do not assume faster results without workload measurements |
| ROBOFUZZ, R2D2, ROFER | Survey-reported robotics semantic/stateful fuzz research | Unverified methodology references; verify papers and claimed results before citation or implementation |

Require malformed-input and mutation/property coverage for BVH, C3D, URDF and custom binary readers, with size/time bounds and regression replay. BVH and URDF are text formats; robustness requirements are not limited to binary formats. Track minimized failure fixtures and larger corpora in appropriate artifact storage. Replay minimized regressions in ordinary CI and run budgeted long campaigns separately. If a foreign fuzz-engine exception is later approved, pin its toolchain and keep it outside default-build execution.

Extend fault injection to stale/reordered messages, invalid dimensions, contact-mode transitions, callback ordering, watchdog behavior and actuator acknowledgment loss. Robotics-specific oracles test bounded commands and state-machine invariants, not only process crashes. Prefer reusable harnesses in `robot-eval` and parser crates before creating a separate `robot-test` package.

#### B.2.12 Parallelism and serialization

| Candidate | Role / provisional maturity | Qualification decision |
| --- | --- | --- |
| Rayon | Established CPU data parallelism / custom pools | Default rollout and preprocessing shortlist; isolate from control, benchmark speedup and memory bandwidth |
| Crossbeam | Established queues/channels/concurrency tools | Select bounded primitives for a demonstrated need; test backpressure and allocation rather than assume every API is real-time-safe |
| `rkyv` | Archived access with reduced decoding/copying | Candidate when profiling warrants it; validate untrusted bytes, alignment, format version and feature/endian configuration |
| Cap'n Proto Rust (`capnp` ecosystem) | Schema-based encoding/RPC | Qualify message handling and compiler/toolchain separately; an external schema compiler is not automatically permitted |
| `bincode` 2.x | Rust binary encoding with optional Serde integration | Candidate, not a durable schema by itself; verify current maintenance, limits, configuration and migration story |
| `postcard` | Compact Serde-based embedded encoding | Embedded shortlist; test bounded buffers/framing and size. Varint encoding is not zlib compression |
| `prost` | Protocol Buffers messages and code generation | Qualify runtime and any `protoc` requirement separately; not an automatic ROS DDS/CDR bridge |

Do not adopt separate formats for every component before measuring. Compare representative sensor frames, state snapshots, commands and episode chunks across payload sizes, using p99.9 latency, throughput, allocations and encoded size. Compare validation cost and schema evolution as well as raw speed; zero-copy is not universally faster. No unqualified benchmark ranking or stable Rust-ABI claim is accepted.

Use bounded dedicated worker pools, per-environment RNG streams and stable environment IDs for parallel rollouts. Record scheduling/reduction nondeterminism where present; compare sequential and parallel results within declared tolerances. Rayon belongs in offline work unless a distinct bounded scheduling design proves suitability for a particular runtime task.

`glam`, additionally named by the v2.0 survey, is a graphics/vector-math candidate for tools; avoid introducing duplicate physics/frame conventions alongside nalgebra without a measured benefit and tested conversions.

### B.3 Concrete workspace anchors

| Crate family | Initial shortlist / authored work |
| --- | --- |
| `robot-types` | Typed units/frames/time; evaluate a units crate such as `uom` |
| `robot-model` | `urdf-rs` plus authored semantic validation and actuator metadata |
| `motion-io` | BVH candidates, `c3dio`; qualified or authored numeric human-body format support |
| `motion-retarget` | nalgebra, qualified Jacobians and constrained solver; authored task mapping |
| `robot-sim` | Rapier existing multibody/contact path first; compare verified alternative modules |
| `robot-learn` | Burn shortlist, authored PPO, audited RL references |
| `robot-skills` | Exact Rust search; `instant-distance` only after crossover measurement |
| `robot-hardware` | Protocol-specific Rust adapters; qualify `roslibrust`/Zenoh if interoperability is needed |
| `robot-control` | Authored deterministic runtime; compare HULKs/Copper patterns and qualified IPC |
| `robot-adapt` | Authored identification baseline; investigate verified dynamics-gradient modules |
| `robot-viewer` | Rerun or egui; `urdf-viz` for model inspection |
| `robot-artifacts` | Rust columnar/binary tooling; canonical schemas separate from visualization formats |
| `robot-perception` estimation module | Qualify multicalc/numeris/typed_kalman/kfilter or authored estimator against the same fixtures |
| `motion-retarget`, `robot-control` optimization modules | Clarabel and other verified Rust solver candidates; qualified MPC/WBC modules where justified |
| `robot-eval`, parser test targets | Rust property/mutation/fault harnesses; optional foreign-engine fuzzing only after separate exception |
| Orchestration telemetry | tracing; optional off-thread OpenTelemetry and metrics exporter |
| Offline rollout/data workers | Rayon/custom pools and bounded Crossbeam primitives where needed |

These anchors are a shortlist, not permission to add every library. Prefer the smallest qualified dependency set.

The survey's proposed `robot-state`, `robot-ctrl` and `robot-test` names map to the existing estimation, control and evaluation boundaries above. Extract new crates only if ownership or dependency isolation warrants it; do not maintain duplicate responsibilities.

### B.4 Simulator and adaptation qualification

1. Verify existing Rapier functionality and any alternative's actual code before designing missing subsystems.
2. Test analytic pendulum and actuator fixtures, articulated mass/inertia behavior, floating-base dynamics, joint limits and motor constraints.
3. Test standing, slip, contact impulses, pushes and impacts. Report penetration, energy behavior, tracking error and sensitivity to timestep/solver settings.
4. Measure CPU environment throughput before budgeting GPU implementation. A GPU network backend or cartpole rollout is insufficient evidence for GPU humanoid physics.
5. Test snapshot completeness: engine state alone may omit policy recurrence, RNG, sensors, pending commands and task state required for replay.
6. If foreign cross-validation is separately authorized, compare matched fixtures through a non-default profile and publish divergence. Do not make this unavailable optional engine a prerequisite for default-build progress.
7. For gradient-based identification, compare derivatives with finite differences on smooth fixtures, document behavior at contact changes and test parameter identifiability. Use held-out trajectories and calibrated torque estimates.

### B.5 Capability gaps and priority

1. **Qualified humanoid simulation:** establish whether existing Rust articulation/contact implementations suffice; author only demonstrated gaps.
2. **Training throughput:** qualify CPU batching, then budget any necessary Rust GPU physics work.
3. **Reliable RL:** implement and validate PPO; inspect existing traits and algorithms before duplicating useful interfaces.
4. **Constrained retargeting:** qualify solver and Jacobians; implement robot-specific task mapping and temporal/contact constraints.
5. **Human-body formats:** qualify existing Rust code or implement the minimum required SMPL-family subset; keep it off the BVH-first critical path.
6. **Low-data identification:** qualify gradients and sensing assumptions before relying on a calibration library.
7. **Estimator validation:** define the floating-base sensor model and qualification fixtures before adopting broad filter suites.
8. **Wire/archive contracts:** benchmark versioned bounded encodings and migration behavior; author application schemas regardless of serializer choice.
9. **Robustness and observability:** build parser/property/fault harnesses and nonblocking instrumentation with a Rust-compatible toolchain.

The survey's claimed absences become verification tasks, not automatic instructions to rewrite large subsystems.

### B.6 Gates for dependency promotion

For each selected candidate:

1. Record canonical repository/crate URL, version/commit, access date, license, maintainers and actual inspected source. Versions reported in the survey are not pins.
2. Inventory features, target-specific dependencies, build scripts, native linkage and runtime requirements. A lockfile alone does not prove Rust-only execution.
3. Reproduce the required capability on our fixture and hardware, with measured resource use and latency where relevant.
4. Validate serialization/checkpoint/index/snapshot behavior and determinism guarantees where applicable.
5. Fuzz external-data boundaries and test malformed inputs, dimensions, limits and failure behavior.
6. Record fallback/replacement cost and ownership if maintenance stops.
7. Promote only the qualified modules/backends; retain untested features as experimental.
8. Compare the previous baseline or credible alternative at a fixed workload/budget; where no alternative exists, record authorship and maintenance cost estimates.

A failed gate leaves the candidate reference-only. Update the gap register and plan rather than assuming the dependency is usable. No claim in this appendix constitutes permission for foreign-runtime exceptions or confirmation of hardware validation.

### B.7 Risk register

Likelihood and impact are project-specific assessments to make after qualification; the supplied ratings are not measured probabilities.

| Risk | Detection / implication | Mitigation |
| --- | --- | --- |
| Young dynamics/control library stalls | Missing maintainers or unverified broad feature surface | Keep module boundaries and qualified Rapier path; estimate targeted replacement, not a mandatory parallel rewrite |
| Solver or generator violates Rust-only policy | C solver, generated C, Python workflow or external compiler in build chain | Prefer qualifying Rust paths; foreign use requires a separate scope decision |
| Burn or other pre-1.0 API changes | Checkpoint/backend compatibility breaks | Pin toolchain/features and preserve readable model contracts; freezing one training run does not remove migration cost |
| Fuzz-engine/toolchain conflict | libFuzzer/honggfuzz or nightly requirements | Keep Rust-native property/mutation/regression tests available; optional exception and isolated runner only if approved |
| Parallelism has poor speedup or hurts deadlines | Bandwidth saturation, oversubscription, tail growth | Profile sequential baseline, isolate pools, bound work and keep work stealing off control |
| SMPL support remains unavailable | No qualified implementation/assets | Implement only required subset or accept provider export; keep BVH-first delivery independent |
| Estimator diverges under stress | Inconsistent covariance, bias drift, outliers/contact loss | Test alternative filters and model assumptions; detect uncertainty and degrade gracefully |
| QP/MPC misses deadline or is infeasible | Tail measurements, solver statuses, residual failures | Bound iterations, simplify qualified task and invoke validated fallback; do not assume analytic IK preserves constraints |
| Telemetry changes control behavior | Enabled/disabled timing divergence, full queues | Bounded event capture, drop counters and isolated exporters |
| Archive/wire incompatibility or malformed data | Upgrade replay failures, invalid archived pointers/dimensions | Versioned envelopes, validation, migration fixtures and input limits |

## Appendix C — Open questions and unresolved decisions

This specification deliberately leaves decisions open rather than inventing defaults that later read as requirements. Each entry below blocks something concrete. An entry is closed by a decision record, after which the decision is cited from the relevant section and the row moves to the revision history — it is not deleted silently.

| ID | Question | Blocks | Resolution mechanism | Interim position |
| --- | --- | --- | --- | --- |
| C-01 | What happens to the existing Python `training/` and `unitree/` code? | §1.1; any claim that the repository satisfies §2 | Migration decision record | Retained as prior art and protocol reference; excluded from build, CI and all deployable paths |
| C-02 | Which reference robot model, and which first hardware target? | Simulator qualification, joint-level training, every numeric threshold, the per-robot safety profile | Decision record after availability is known | Hardware-agnostic APIs proceed; a documented reference embodiment is used for simulation |
| C-03 | Can the Rust-only physics and autodiff stack meet humanoid contact and throughput requirements? | R0 feasibility, the scale of authored work, any GPU physics commitment | `PLAN.md` Phase 0 exit gate; B.4 | Assumed insufficient until measured; authored work is scoped only against demonstrated gaps |
| C-04 | Canonical frames, gravity direction, root convention, quaternion ordering and timing model | Every schema in §7, every transform, all fixture data | Decision record; recorded into A.3 | No project-wide convention may be assumed; transforms name their frames explicitly |
| C-05 | Learning backend: Burn/CubeCL, Candle, or authored | §9.2, policy bundle format, export and inference parity | Phase 0 workload comparison under B.6 | Burn shortlisted, not selected |
| C-06 | Observation and action schema, and the actuator command interface | Training/deployment sharing, §7.4 bundles, supervisor limits | Decision record after C-02 | Schemas are versioned from the start so the decision is migratable |
| C-07 | Which corpora are licensed for our training and redistribution? | §8.1 ingestion, any published result derived from them | License audit per source dataset, not per aggregate | No corpus is assumed usable; AMASS component licenses are checked individually |
| C-08 | Benchmark machine, control period and the first acceptance profile | §5 budgets, §14 thresholds, all timing claims | Decision record before the first evaluation run | 100 Hz is a starting target, not a commitment |
| C-09 | Which Rust QP/optimization solver, if any, qualifies for full-body IK? | §8.3, B.2.8, runtime use in §10 | Benchmark on representative full-body constraints | Clarabel shortlisted; a feasible non-QP baseline is retained |
| C-10 | Which estimator implementation, or an authored one? | §11.2, B.2.9, state estimation in `robot-perception` | Common-fixture comparison against independent truth | No selection on advertised feature breadth |
| C-11 | Durable record encoding and IPC encoding | §7.5, upgrade-readability guarantee | Measured comparison on representative payloads | Decided separately from each other; neither is chosen by default |
| C-12 | Is GPU batched simulation required, and who authors it? | Training budgets, Phase 3 scope | Measured CPU throughput first, then a costed proposal | Not assumed from GPU neural-network support |
| C-13 | License and redistribution terms for this project's own outputs | Publication of bundles, datasets and reports | Decision record | Unresolved; no external distribution until it is |
| C-14 | Which Rust toolchain input and host platforms does the locked devenv environment support, and does CI run devenv natively or in an image built from the same lock? | §2.2 onboarding, the A-22 row, the benchmark-host decision in C-08, and the change-record obligation for this §2/§14 revision | Decision record after the environment is bootstrapped and exercised in `PLAN.md` Phase 0–1 | nixpkgs-pinned stable Rust through `devenv.yaml` inputs; Linux x86_64 is the only host assumed exercised; macOS and WSL2 are unverified |

A blocked requirement is not a deferred requirement. Where an open question prevents a section from being testable, the corresponding Section 14 row cannot be closed, and a release report MUST say so rather than omitting the row.

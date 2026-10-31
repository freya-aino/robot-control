# Humanoid Robotics Platform Specification

Status: proposed architecture and acceptance contract. Not ratified; no clause below is evidence that the described system exists.

| Field | Value |
| --- | --- |
| Document version | 0.8.5-draft |
| Last revised | 2026-10-31 |
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
| 0.4.0-draft | Appendix B replaced with the supplied Rust ecosystem survey revised v3.0. Adds the external-model-service dependency class (Section 2.3), the agent-advisory boundary in Sections 5 and 12.6, the optional agent layer in Section 6, acceptance rows A-23 and A-24, glossary entries and open questions C-15 to C-17. Touches Sections 2, 5, 12 and 14, so the change-control rule applies and the decision record is tracked as [C-15](#appendix-c--open-questions-and-unresolved-decisions) until written | 2026-10-25 |
| 0.5.0-draft | Supplied ADK-Rust integration appendix folded into the existing structure instead of being added as an appendix. Specifies the three admissible uses of an agent layer — development-time automation, offline policy and personality sampling assistance, and a maintained low-level dependency knowledge base — in Section 2.3.1, with the knowledge base in Section 13.1, sampling assistance in Sections 9.5 and 12.5, acceptance rows A-25 to A-27, qualification gate 10, risk rows and open questions C-18 and C-19. Touches Sections 2, 12, 13 and 14, so the change-control rule applies and the decision record remains tracked as [C-15](#appendix-c--open-questions-and-unresolved-decisions) | 2026-10-25 |
| 0.6.0-draft | Adds extensible robot descriptor and shared experience/replay contracts in Sections 7.1 and 7.6, RL extension points and conformance requirements in Sections 9.3 and 9.6, Rust mechanism candidates in Appendix B, implementation guidance in Appendix D, acceptance rows A-28 to A-31, and open questions C-20 and C-21. Touches Sections 7 and 14, so the change-control rule applies and the decision record is tracked as [C-20](#appendix-c--open-questions-and-unresolved-decisions) and [C-21](#appendix-c--open-questions-and-unresolved-decisions) until written | 2026-10-26 |
| 0.7.0-draft | Folds the surveyed low-data humanoid learning methods into the existing structure: the hierarchical latency invariant in Section 5.1, the staged latent competitive pipeline of Section 9.4 following the now `paper_verified` RoboStriker method, retrieval-augmented policy learning in Section 9.7, retrieval tiers and deadline attribution in Section 10.6, expanded limited-data adaptation methods in Section 11.3, three further extension points and two storage clauses in Sections 9.3 and 7.6, research-register and Appendix A/B entries, acceptance rows A-32 to A-35, and open questions C-22 to C-24. Touches Sections 5, 7 and 14, so the change-control rule applies and the decision record is tracked as [C-22](#appendix-c--open-questions-and-unresolved-decisions) until written | 2026-10-26 |
| 0.7.1-draft | Records the removal of the Python prior art from the working tree: Section 1.1 is restated as the disposition of retired assets reachable only in version control, Section 2.2 item 8 points manual comparison runs at a historical commit, and C-01 is resolved with reintroduction routed through the Section 2.1 exception procedure. Touches Sections 1 and 2, so the change-control rule applies and the migration decision record is the resolution of [C-01](#appendix-c--open-questions-and-unresolved-decisions) | 2026-10-26 |
| 0.7.2-draft | Revises the simulation target, by explicit project direction, to require GPU-bound batched execution, variable fidelity and high-fidelity simulation, detailed in a new companion document rather than in this file: `crates/robot-sim/SIMULATION_SPECS.md`. Section 9.1 cross-references it without relaxing the Rust-native requirement; Appendix B.2.1, B.2.3 and B.4 gain cross-references and one new qualification item (B.4 item 8, GPU-vs-CPU-reference agreement before any throughput claim). Does not touch Sections 2, 4, 5, 7, 12 or 14, so the Section-level change-control rule is not triggered; the new open question is tracked as [C-25](#appendix-c--open-questions-and-unresolved-decisions) and the decision itself is `docs/decisions/0005-gpu-batched-simulation-target.md` | 2026-10-26 |
| 0.7.3-draft | Revises the draft against the state of the working tree on 2026-10-27: Section 9.1 records that the floating-base multibody path has no ground contact, Section 9.2 requires checkpoints to carry the observation normalizer and adds parameter-group freezing for fine-tuning, Section 9.3 states that the policy, backend and model-importer seams are not yet traits and must be introduced behavior-preservingly, and test-purpose rendering is specified as a frame dump before any viewer dependency. Adds acceptance rows A-36 to A-39 and open questions C-26 to C-28. Touches Section 14, so the change-control rule applies and the decision record is tracked as [C-26](#appendix-c--open-questions-and-unresolved-decisions) until written | 2026-10-27 |
| 0.7.4-draft | Updates Section 9.1 to record that the floating-base multibody path now has sphere-only ground contact and a qualified supported-stance hold, with its dt range and limits, replacing the earlier statement that the path has no ground contact. Balance, standing and push-recovery claims remain forbidden until contact proxies and a controller exist. Decision record: `docs/decisions/0010-ground-contact-penalty-model.md`. Touches Section 9.1 only; the Section-level change-control rule is met through that record | 2026-10-27 |
| 0.8.5-draft | Adds the attached-agent discovery method of Section 7.7: an agent attaches to a robot and assembles its motion, joint ranges, inertial parameters, available tests and external sources into a candidate profile and a discovery report. A commissioning probe is limited to declared limits, web-sourced claims are proposals, and no agent path activates a profile. Adds acceptance rows A-47 to A-50 and open question C-29. Touches Sections 7 and 14, so the change-control rule applies and the decision record is tracked as [C-29](#appendix-c--open-questions-and-unresolved-decisions) until written | 2026-10-31 |
| 0.8.4-draft | Makes environment collision universal across scenes through the tags `environment` and `collider` in `sim:tags` (Section 9.1). Any prim so tagged, in any scene, is a static collider; robots are recognised by their physics, not by tags; a tagged shape the simulator cannot contact is refused by name. Change control: Sections 7 and 9; decision record `docs/decisions/0015-usd-isaac-simulation-model.md`. | 2026-10-30 |
| 0.8.3-draft | Makes the ground a collider of the scene, not a constant of the simulator: the scene layer `assets/scenes/g1_ground.usda` sublayers the unmodified Unitree G1 layer and adds a `Plane` with `PhysicsCollisionAPI` and its contact parameters (Section 9.1). The ground is written to the replay as a `Plane` prim. Change control: Sections 7 and 9; decision record `docs/decisions/0015-usd-isaac-simulation-model.md`. | 2026-10-30 |
| 0.8.2-draft | Requires hard joint limits in simulation with a proximity signal (Section 9.1), two selectable contact modes (primitives for pretraining, convex hulls for fine-tuning), authored pelvis and torso proxies, adjacency rules at joints, and all-pairs self-collision (Section 9.1). Adds acceptance rows A-43 to A-46 (Section 14). Environment collision is governed by tags, the same in every scene. A prim whose `sim:tags` contain both `environment` and `collider` is a static collider of the scene wherever it sits in the stage. A prim tagged `environment` alone is visual only. Robots are recognised by their physics (mass, joints), never by a tag, and no tag other than these two changes how collision is computed. A tagged collider of a shape the simulator cannot contact MUST cause the run to be refused, naming the prim. The ground plane MUST be a collider of the simulated scene, described in USD with its normal, height and contact parameters. The simulator MUST take its ground from that description. The contact law applied to it is the same as for every other collider. The vendored robot layer MUST NOT be modified to add a collider; scene additions are made in a separate layer that sublayers it.

Performance is not budgeted on the current development machine; it is deferred to a GPU-bound machine (Section 9.1). Change control: Sections 7 and 9 and 14, decision record `docs/decisions/0015-usd-isaac-simulation-model.md`. | 2026-10-30 |
| 0.8.1-draft | Makes the OpenUSD description the source of truth for the robot model and its collision geometry (Sections 7.1 and 7.3), requires contact for every declared collision shape in simulation (Section 9.1), adds acceptance rows A-40 to A-42 (Section 14), and records OpenUSD as a development tool in the devenv (Section 2.2). Change control: Sections 2, 7 and 14 changed; decision record `docs/decisions/0015-usd-isaac-simulation-model.md`. | 2026-10-30 |
| 0.8.0-draft | Replaces the earlier simulation model reference and removes every other scene-description importer from the specification and the repository. The simulation model is the OpenUSD description of the Unitree G1 used by Isaac Sim: links, joints, collision and primitive shapes are read from USD, and recorded replays are written as USD time samples, per ADR 0015. Change control: Sections 2, 7 and 14 are affected and are covered by `docs/decisions/0015-usd-isaac-simulation-model.md`. | 2026-10-30 |

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
- [Appendix B — Rust ecosystem survey and dependency qualification](#appendix-b--rust-ecosystem-survey-and-dependency-qualification-revised-v30)
- [Appendix C — Open questions and unresolved decisions](#appendix-c--open-questions-and-unresolved-decisions)
- [Appendix D — Rust extensibility patterns](#appendix-d--rust-extensibility-patterns)

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

### 1.1 Relationship to prior repository contents

This repository previously contained a Python stack: `training/` (PyTorch, JAX, Warp, Hydra, MLflow, ONNX export), `unitree/` (CycloneDDS configuration and a Unitree SDK connector) and `test.ipynb`. That code was removed from the working tree on 2026-10-26, which closes the divergence from Section 2 that [C-01](#appendix-c--open-questions-and-unresolved-decisions) tracked. The working tree now contains documentation only, and the Rust workspace of Section 6 is unwritten; the specification states the disposition of the retired assets because they remain reachable in version control and are still cited as prior art.

| Retired asset | Disposition after removal |
| --- | --- |
| `training/` Python training code | Prior art and behavioral reference, retained only in history. It MAY be read at a named commit for algorithm intent. It MUST NOT be restored into the working tree, and any manual comparison run happens from a checkout of that commit in the separately named non-default environment of Section 2.2 item 8, with its numbers recorded as `unverified` baselines |
| `training/conf/` experiment configurations | Requirements input. The task, reward and randomization intent expressed there SHOULD be migrated into the configuration schema of Section 9.3, with the source commit cited; the YAML files themselves are not the canonical format and are not reintroduced |
| `unitree/` DDS/SDK connector | Protocol reference for `robot-hardware`. Field layouts, rates and command semantics observed there are hypotheses requiring the hardware verification of Section 7.1. Transcribing them into Rust is authorship against a cited historical reference, not a port |
| `test.ipynb` and notebook workflows | Scratch artifacts, deleted. Excluded from acceptance, reproducibility and artifact-lineage requirements, and not recreated in any form |

Four consequences are binding. The Rust workspace of Section 6 MUST NOT import, subprocess, or link against any of the above, and the removal does not weaken that prohibition — a deleted dependency and an excluded dependency are the same requirement. Numbers produced by the retired Python stack MUST NOT be reported as platform results without restating their provenance, including the commit they were produced at. Reintroducing any of this code, in whole or in part, is a Section 2.1 exception request and not a revert. Removal is not evidence: nothing about the Rust stack became more verified by deleting the prior art, and every evidence state of Section 3 is unchanged by it.

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

Do not label a wrapper around a foreign training or simulation stack a Rust-native implementation. The first engineering gate must establish whether the Rust-only simulation and autodiff stack can meet the task. If not, report the gap and propose Rust implementation work; do not silently substitute Isaac Lab, PyTorch, the reference simulator, or another non-Rust engine. External research can inform algorithms without becoming a runtime dependency.

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

The devenv also provides the OpenUSD command-line tools (`openusd`, which supplies `usdcat` and `usdchecker`). They validate replays and layers the project writes. They are development tools: no shipping crate depends on them, and the Rust USD reader of Section 7.1 does not call them.

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
- A unit-norm latent vector bounds its representation, not joint torques, balance, or contact forces. A latent action space restricts *where exploration can go*; the decoder that realizes it is still an ordinary learned controller subject to Sections 5 and 12.
- Retrieval-augmented policy learning retrieves **data**, not motor commands. The published few-shot results are offline data-selection results: a retriever chooses which prior trajectories join a training batch. Reusing the word "retrieval" for the runtime skill router of Section 10 conflates an offline corpus search with a deadline-bound selection and must not be done in a report.
- A hierarchical decomposition reduces the *difficulty* of the upper layers; it does not reduce the latency obligation of the lower ones. Published hierarchies frequently run every layer at one rate — RoboStriker runs both its latent strategy policy and its decoder at 50 Hz — so a tiered-rate architecture is this project's requirement, not an inherited result.
- "Low-data" in the surveyed literature usually means low *real-robot* data on top of large-scale simulation or a large prior corpus. It is not a claim that the total training budget is small, and it must not be reported as one.
- Cross-simulator agreement is useful evidence, not a guarantee of hardware success. Real contact calibration remains necessary.
- No fixed percentage of motion transferability, minimum number of real rollouts, adaptation duration, or fraction of success attributable to retargeting is assumed.
- Do not infer proprietary Unitree training architecture, exact datasets, curriculum, or autonomous capabilities from public demonstrations. Separate documented features from engineering reconstruction.
- PHC commonly refers to Perpetual Humanoid Control. Mink is a differential inverse-kinematics library built around its own simulator model format, not inherently a GPU batched SMPL pipeline. These names must not become misleading implementation requirements.
- The earlier residual-policy sketch requires privileged simulated state at inference and has no derived correction targets. It is not an executable adaptation algorithm. Simulator correction and robot-side action correction must have distinct contracts.

### 3.1 Research intake register

Maintain the following leads; verify identity, claims, implementation availability, and licensing before scheduling a reproduction.

| Lead from the conversation | Intended investigation | Initial disposition |
| --- | --- | --- |
| RoboStriker, arXiv:2601.22517 (v1, submitted 2026-01-30) | Motion tracking, latent distillation, competitive self-play | `paper_verified` as of 2026-10-26: primary source retrieved and the method located in text. Licenses quoting the staged method of Section 9.4 and its stated conditions only. No source release was identified, so nothing is `code_inspected`, and every reported number was produced in a foreign simulator on one robot |
| NFSP (Heinrich and Silver, arXiv:1603.01121); fictitious play | Opponent averaging, anti-cycling, reservoir-sampled average policy | Cited by RoboStriker as the basis of LS-NFSP; verify the original convergence conditions separately from the humanoid result |
| AMP (arXiv:2104.02180), ASE, DeepMimic | Adversarial motion priors and reusable skill embeddings as style regularizers | Reference algorithms for the style term of Section 9.4; a discriminator reward is a style constraint, never a safety or feasibility constraint |
| BehaviorRetrieval, FlowRetrieval, RAEA, RAG-Diff | Retrieval-augmented imitation and policy adaptation | Verify the retrieval unit, embedding source and whether retrieval occurs offline, between episodes or in the loop; each choice has a different cost under Section 10.6 |
| RMA and A-RMA (rapid motor adaptation) | Two-phase latent system identification from recent proprioceptive history | The reference form of the history-conditioned row of Section 11.3; verify the adaptation-module training procedure and its deployment observation set |
| SGFT (simulation-guided fine-tuning, ICLR 2025) | Real-world fine-tuning guided by a simulation-derived value or shaping term | Candidate for the adapter row of Section 11.3; verify what the method requires on hardware before any on-robot update is scheduled |
| MOSAIC, H-Zero, PolySim, ZEST and related 2025–2026 sim-to-real humanoid leads | Residual adaptation, cross-embodiment few-shot transfer, multi-simulator randomization | Unverified leads collected during the Section 9.7/11.3 survey; investigation order follows Appendix B.5, and none is a dependency |
| Unitree G1 boxing / Iron Fist King | Public product behavior, hardware constraints, demonstrations | Treat training internals as unknown unless directly documented |
| GMR, PHC, H2O/OmniH2O, ExBody, ProtoMotions | Retargeting, imitation, whole-body control | Reference algorithms; Rust implementations required |
| ASAP, arXiv:2502.01143 | Learning simulator/action mismatch from real trajectories | Verify where correction operates during training and deployment |
| SPiRL, NPMP, ControlVAE, PhysicsVAE, NCP | Skill priors and latent controllers | Establish a simple reproducible baseline before extensions |
| STRAP, arXiv:2412.15182; SMOSE | Retrieval and sparse expert control | Verify objective, retrieval unit, benchmark and actual latency evidence |
| ARLI, DARP, ReGIL, MABR, COLLIE, SafeFlow, InstantMimic, H-Zero, R²S², LATENT | Delay-aware control, retrieval, reusable representations, transfer | Unverified leads; no core dependency |
| PAMoR, SocialHumanoid, StreamTalk, LiveGesture, PersonaGestor, Animus | Optional personality and expressive motion | Unverified leads; separate from motion safety and baseline delivery. The persistent-identity sketch has a candidate Rust mechanism in Appendix B.2.13; a mechanism is not a requirement |
| `adk-rust` (zavora-ai) and the ADK name collision | Agent orchestration: workflow agents, session lifecycle, markdown skill indexing, graph memory, agent-to-agent transport, guardrails | `unverified`; three unrelated projects share the "ADK" name and every reference MUST name the project and crate. Non-model crates are qualified separately from the model-service gate of Section 2.3, and any use is read against the three admissible uses of Section 2.3.1 |
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
| Optional agent advisory, if ever enabled | 1 Hz or slower, or between rounds | Out of process, feature-gated per Section 2.3; produces expiring advisory records, never commands; the loop above it runs unchanged when it is absent |
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

No external model service of Section 2.3 and no agent runtime built on one may execute inside a deadline-critical section, in any mode, under any approved exception. Where an agent layer exists at all it runs outside the control process and its output enters only as an expiring advisory record through the same validated queue interface, freshness limit and rejection accounting as any other asynchronous proposal, measured as decision age in the table above. An advisory that has expired is dropped, not held: a stale directive is indistinguishable from a wrong one. Removing the agent layer entirely MUST leave the control loop functionally unchanged, which is the point of row A-24.

Rayon work stealing and general-purpose telemetry subscribers are not presumed suitable for this loop. Use dedicated bounded resources for control and separate pools for rollout/data processing. Capture preallocated timestamp/event records in the loop and construct/export telemetry off-thread; benchmark enabled instrumentation against a disabled baseline.

Use monotonic timestamps, fixed-capacity queues, explicit freshness limits, CPU affinity and scheduling configuration where available, and a watchdog independent of the policy process. Log overflow must never block actuation; emit dropped-record counts. Safety-relevant incidents require a reserved event channel and bounded pre/post-event recording buffer.

### 5.1 Hierarchical decomposition and the latency invariant

The learned stack is hierarchical by construction, and the hierarchy is ordered by deadline rather than by conceptual importance. The governing invariant is that **the lower a component sits, the tighter its latency obligation and the smaller its permitted variability**. A design that places a large or variable-cost network below a small one violates this section regardless of its accuracy.

The following bind every hierarchical policy, including the staged pipeline of Section 9.4 and any router of Section 10:

1. Every learned component MUST be assigned to exactly one tier of the table above, and that assignment MUST be recorded in the policy bundle of Section 7.4 alongside the component's measured compute duration. A component with no declared tier MUST NOT be activated.
2. Tier rates MUST be non-increasing with height: no component may run faster than a component below it, and the lowest learned tier MUST run at the policy rate or faster. Two tiers MAY share a rate, which is the degenerate case and MUST be recorded as such rather than described as a hierarchy.
3. Each tier carries its own deadline, its own worst-case budget and its own fallback. A tier MUST remain functional when the tier above it is late, absent or has produced only an expired output; the response to a missing upper-tier decision is the declared hold-or-default behavior of that tier, never an extended deadline below.
4. Outputs crossing a tier boundary MUST be bounded, validated and timestamped, and consumers MUST reject them on the freshness limit of their own tier. Decision age is attributed to the tier that produced the decision, not to the tier that applied it.
5. Lower tiers MUST NOT acquire dependencies that upper tiers have: no retrieval query, corpus access, dynamic dispatch, allocation or variable-iteration optimization may be introduced below the tier at which it was qualified. Pushing an expensive mechanism downward is a change to Section 5 and requires the change-control record.
6. The per-tier budget MUST be measured, not inferred from parameter count. Section 3's correction about sparse activation applies to every tier: a smaller network is a hypothesis about latency until the tail is measured under representative load.
7. Removing the highest tier MUST leave a controller that is still safe, even if it is no longer tactical. This is the hierarchical form of the removal tests in rows A-24 and A-30, and it is what makes strategic layers optional rather than load-bearing.

The frequency separation is also an evidence boundary. A result obtained with every layer running at one rate does not establish that the same decomposition holds when the strategy layer is slowed to its intended tier, because decision age that was zero in the published setting becomes nonzero here. Any tiered deployment of a method validated at a single rate MUST re-measure task performance at the tier rates actually used.

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
| `robot-devtools` | Development-time automation and the dependency knowledge base of §13.1: descriptor and dataset handling, integration and parsing helpers, test-run orchestration, ledger and inventory bookkeeping. Developer and CI hosts only; never a dependency of any deployable crate |

Training and deployment share observation/action schemas and tested normalization logic. Deployment does not depend on training autograd code. Avoid a plugin design that permits arbitrary dynamic code loading in the control process: build-time Rust implementations are the default, and optional experimental workers run outside that process.

The v3.0 survey proposes an additional `robot-brain` package hosting agent orchestration, opponent memory and operator-facing advisory agents. No such package is created by this revision. Its responsibilities, where they are adopted at all, are an optional out-of-process extension of `robot-persona` and the tooling crates, and they are subject to Section 2.3 in full: feature-gated, absent from the default build, unreachable from `robot-control`, `robot-hardware` and the supervisor, and removable without affecting them. Whether the boundary justifies its own package is open under [C-16](#appendix-c--open-questions-and-unresolved-decisions); a survey's proposed crate name is not an architectural requirement, as already stated for `robot-state`, `robot-ctrl` and `robot-test` in Appendix B.3.

The three admissible uses of Section 2.3.1 attach to this layout as follows: development-time automation (U1) lives in `robot-devtools` and the `devenv` scripts of Section 2.2 and is a `dev-dependency` of nothing that ships; assisted sampling (U2) is an offline producer of configurations consumed by `robot-learn`, `robot-tasks` and `robot-persona`, with no reverse dependency; and the dependency knowledge base (U3) is records in this repository with tooling in `robot-devtools`. A dependency edge from `robot-control`, `robot-hardware`, `robot-skills` or any bundle-producing crate to any of the three is a policy violation, detectable by the exclusion gate of Section 2.1 and asserted by row A-25.

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

Every profile MUST carry a `schema_version` and typed, stable IDs for joints, links, actuators and sensors. IDs MUST NOT be reused for a different entity within a profile lineage. The required core schema MUST remain small and explicit; additional declarative metadata belongs in open, namespaced extension tables with declared value types and capability tags. Unknown core fields and unknown extensions MUST be rejected before a profile is activated on a control path. Non-control tooling MUST preserve unknown namespaced extensions without interpreting them when it reads and writes a profile. A capability tag is a claim to validate against the profile's typed fields, not a substitute for them. Schema changes MUST provide explicit migrations, and each migration MUST preserve or report loss of meaning.

The OpenUSD robot description is the reference embodiment (ADR 0015). The Unitree G1 OpenUSD layers are read by the std-only reader in `crates/robot-usd` and converted into a profile by `robot_model::usd`. The import MUST take link masses, centres of mass and principal inertias from `physics:*`; joint frames, axes and limits from the `PhysicsRevoluteJoint` prims; drive effort, stiffness and damping from `drive:angular:physics:*`; and the velocity limit from `physxJoint:maxJointVelocity`. Every collision prim under a link's `collisions` scope (sphere, box, cylinder or mesh) MUST be imported with its pose in the link frame, and no collision shape may be dropped. Mesh collision shapes are kept as references to the asset prim, and their vertices MUST be available to contact (Section 9.1). The USD model has no primitive collision shape for the pelvis or the torso. The simulator MUST therefore use primitive proxies for those links. Their dimensions MUST be fitted to the bounding box of the USD pelvis and torso collision meshes, and recorded in the profile as derived from those meshes, with the fitting method stated. Anything the importer cannot represent MUST be refused by name. Content that exists only in another application, such as Omniverse MDL materials, MUST be reported as a warning and MUST NOT stop the import.

URDF import is an initial interchange target, not a complete dynamics specification. Store missing actuator/contact metadata separately. Unsupported model features must be rejected explicitly rather than silently dropped.

### 7.2 Canonical motion record

Store `schema_version`, source hash, source license, subject/session ID, skeleton/model hash, frame timestamps, validity masks, root pose, named joint rotations or positions, optional velocities, contact labels with confidence, task labels, and provenance for every transformation.

Use SI units, a documented right-handed coordinate system, explicit world/body frames, and named quaternion ordering. Prefer typed quaternion serialization with named components over ambiguous four-element arrays. Euler-angle conventions are source-specific; never assume every BVH file uses the same rotation order.

Human skeleton motion, parametric human-body parameters, and robot joint trajectories are distinct record types. BVH, C3D, and SMPL-family data are not interchangeable. SMPL assets and source datasets may have independent license restrictions.

### 7.3 Robot episode record

Record monotonic sensor timestamps, arrival times, synchronized wall-time mapping where needed, raw and estimated state, uncertainty, desired and measured joint motion, commanded actions, actuator acknowledgments if available, contacts, temperatures, power, policy/expert IDs, transition state, task outcome, and supervisor interventions.

A simulated run that is recorded for inspection MUST be written as an OpenUSD layer with time samples (ADR 0015). The layer MUST carry each link's world pose per step, each joint's position, velocity and applied torque per step, and every collision and primitive shape of the model with its link-frame pose. Contact points and contact forces MUST be recorded once the simulator exposes them; until then the layer MUST say so rather than imply that forces are present.

Store both raw commands and applied/clamped commands so training cannot confuse policy intent with actuator input. Record validity and provenance for estimated torque; motor current is not automatically calibrated joint torque. Preserve observation and action history needed to model delays.

### 7.4 Policy bundle

A deployable bundle contains model weights in a Rust-loadable versioned format, graph/architecture version, observation schema, normalization statistics, action semantics and scaling, recurrent-state specification, rates, robot-profile compatibility, training data lineage, safety-envelope assumptions, transition contracts, benchmark results, and checksum/signature.

ONNX is an optional interchange format only if a qualifying Rust-native path supports the required operators. It is not the default training/export dependency. Unsupported operators must fail before deployment.

### 7.5 Storage and splits

Use chunked binary arrays for high-rate numerical data and columnar records for searchable metadata; evaluate Arrow/Parquet implementations for Rust. TOML manifests refer to content-addressed blobs. Never require loading an entire corpus into memory. Do not load executable pickle objects; supported legacy formats need a constrained Rust parser or a documented non-pickle export from the data provider.

Select IPC and persistent-record encoding separately using Appendix B's serialization candidates. Every wire/archive format requires explicit versioning, size limits, framing, endianness/alignment rules where applicable and validation before access. Serialization is not a transport or a schema-migration strategy. Durable episode records must remain readable across supported software upgrades.

Split datasets by subject, recording session, and original motion before generating windows or augmentations. For competitive learning, separate evaluation opponent pools. Track license eligibility, retention policy, consent where relevant, and dataset revisions.

### 7.6 Experience and replay storage contract

On-policy rollouts and off-policy replay MUST use the same versioned transition and episode schema. A transition MUST identify its episode, timestep, environment and robot-profile version, and contain the observation, action, reward terms, next observation, termination and time-limit-truncation state, validity masks, and any task data required to reproduce reward or goal relabeling. Preserve timestamps, provenance and random-stream identity where they affect the target or replay result. An episode MUST also carry its configuration, seed manifest and completion reason. This schema is distinct from the robot episode record in Section 7.3: a training transition is a typed view of the required training evidence, not a replacement for raw hardware records.

1. Training storage MUST expose bounded chunks in a structure-of-arrays (SoA) layout or an equivalently measured layout; it MUST NOT require materializing the entire corpus or a complete replay buffer as one allocation. The initial in-memory replay implementation SHOULD use preallocated fixed-capacity ring storage with explicit overwrite and generation-ID semantics. On-policy rollout storage MUST report capacity exhaustion rather than silently overwrite samples before the learner consumes them.
2. Storage and sampling MUST be separate interfaces. A sampler MAY provide uniform, prioritized (including a sum-tree implementation), sequence/window sampling, or goal relabeling such as HER-style relabeling. Sampling returns MUST identify source generations and, where applicable, probabilities and importance weights; stale or overwritten references MUST be rejected.
3. N-step targets, generalized advantage estimation (GAE), masks and other learner-specific targets MUST be derived views or batches over canonical transitions, not incompatible serialized transition formats. Derived caches MUST identify the source transition generations and computation/configuration version.
4. Each environment and sampler MUST have an independently recorded seed stream. Replaying the same stored transitions and sampler seed MUST reproduce indices, weights and relabeled goals, subject to explicitly recorded parallel-reduction nondeterminism. Parallel rollout workers MUST have stable environment IDs and independent RNG streams, consistent with Appendix B.2.12.
5. Persistent chunks MUST use the versioned, bounded encodings of Section 7.5. The in-memory view SHOULD permit zero-copy access where alignment, validation and versioning allow it; zero-copy is not permission to trust unvalidated bytes or to make Rust ABI the durable schema. Migration and malformed-input behavior remain explicit.
6. Buffer policy, sampler, target/view construction and persistence MUST be independently replaceable. A new algorithm MUST be able to reuse the canonical transition schema without editing the robot model, simulator, or control runtime. These interfaces are training/offline interfaces; they MUST NOT be reachable from the deadline-critical control path.
7. An algorithm MAY require more than one concurrent store over the same schema — a competitive learner of Section 9.4 maintains both an on-policy rollout buffer and a reservoir-sampled behavior buffer, and a retrieval-augmented learner of Section 9.7 reads a prior corpus alongside the current rollout. Multiple stores MUST therefore be composable without a second transition format: each store declares its retention policy (bounded ring, reservoir, append-only corpus), its admission rule, and whether its contents are drawn from the current policy, a historical policy or an external corpus. Transitions MUST record which policy or corpus produced them and at which generation, because a record whose origin is unknown cannot be used as a behavior target.
8. Where a transition stores an action in a learned latent space rather than the actuator action space, it MUST record the latent-space identity and version and the decoder artifact required to interpret it. A latent transition is not interchangeable with a joint-space transition, and a buffer MUST reject a mixture unless the schema explicitly carries both.

### 7.7 Attached-agent robot discovery

This section defines a method by which an agent attaches to a robot and assembles, automatically, what the project needs to know about it: its motion, joint ranges, inertial parameters, available tests, and what external sources state. The method produces a **candidate profile** under Section 7.1 and a **discovery report**. It activates nothing.

The offline, simulated and web parts of the method are development tooling under U1 of Section 2.3.1. The commissioning probe of item 3 is a separate use that is not admissible until [C-29](#appendix-c--open-questions-and-unresolved-decisions) is decided. No part of the method runs in the control path, in the supervisor of Section 12.1, or on a robot host outside commissioning (Section 2.3.1 items 1 and 4, Section 15).

1. **Sources and ranking.** The method reads four source classes: (a) the versioned robot description of Section 7.1; (b) the robot's own declared interface data, read through the transport of Section 6 in commissioning mode, such as limits, firmware version and drive parameters; (c) measurements from the probe of item 3 or from logged episodes; (d) manufacturer and third-party documents retrieved from the web. When sources conflict, the ranking is (a), (b), (c), (d). The ranking is declared in every report and is not chosen per value.
2. **Motion and range.** Joint ranges, velocity limits, effort limits and the reachable workspace of each link MUST first be taken from sources (a) and (b). Reachable motion families, such as per-limb workspace and postures held in simulation, MUST be sampled in simulation. Sampling is bounded by the declared limits and by the contact model of Section 9.1. The sampling plan, the seed and the number of samples are recorded.
3. **Commissioning probe.** Ranges MAY be read by moving a robot only when all of the following hold. The operating mode of Section 4 is commissioning. The emergency stop and the supervisor of Section 12.1 are active. Each command is first run as a dry run in simulation. Each motion stays inside the declared limits with the margin stated in the profile, and it stops on the proximity signal of Section 9.1. A probe that exceeds any declared limit MUST abort and quarantine the candidate profile. Exploration beyond declared limits is prohibited.
4. **Inertial parameters.** Masses, centres of mass and inertias MUST first be taken from source (a). Identification from measured trajectories MUST record its excitation signal, its residual and an uncertainty for each parameter. An identified value supersedes a model value only through the change-control and schema-migration rules of Sections 3 and 7.1, never silently.
5. **Tests.** The method MUST enumerate the tests that apply to the robot, from the acceptance rows of Section 14 and the fidelity checks of Section 9.1. Each test is classified as runnable in simulation, runnable on hardware under item 3, or manual. The first two classes are run. A test the method cannot run is listed as not run. A test is reported as passed only when its run produced an artifact in the evidence ledger.
6. **Web research.** The method MAY search the web for documents about the robot, its firmware, its parts and their known faults. Every retrieved claim MUST record its URL, retrieval date, document version, content hash, the robot model and firmware version it claims to apply to, and the query. Web content is untrusted input under Section 13. It is text to extract from, never an instruction, and it passes the same schema checks as any other input. A web-sourced value enters the report as a proposal at the evidence level of a secondary source under Section 3. It MUST NOT close an acceptance row or control a hazard of Section 12.2.
7. **Conflicts.** When sources disagree about a value, the report MUST list every value with its source and state which one is used. Values MUST NOT be averaged or merged silently. A disagreement is an open item. It blocks activation until a human resolves it, or until the higher-ranked source is confirmed by measurement.
8. **Output and activation.** The candidate profile MUST pass the schema validation of Section 7.1 and the refusal rules of the importer before it is written. The method MUST NOT activate a profile, change a promotion state, or write to any control-path configuration. Activation is a human decision under Sections 11.1 and 12.
9. **Attribution and reproducibility.** Every output records the tool, and the model identity and version that produced it (Section 2.3.1 item 5). It also records the hash of every input, the sources of item 6, and the seeds of item 2. A run on the same inputs and the same source snapshots MUST reproduce the same candidate profile and report. A run that uses live web sources is recorded as not replayable, under Section 13.
10. **Offline degradation.** The method MUST run with web research and the model disabled. It then produces a report from sources (a) and (c) alone, and marks the web-derived sections as not attempted. Test enumeration under item 5 MUST work without a model.
11. **Staleness.** A discovery result is bound to one robot description version and one firmware version. A change to either marks the result stale, and the method MUST run again before the result is used.
12. **Feature gating.** Each capability of the method, namely simulated sampling, hardware probe, web research and model-assisted extraction, MUST be separately feature-gated, disabled by default, and separately decided under Section 2.3.1 item 3.

---
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

The target was revised, by explicit project direction, to require GPU-bound batched execution, variable fidelity as a first-class interface, and fidelity on par with the OpenUSD / Isaac Sim scene semantics (ADR 0015) — specified in detail in `crates/robot-sim/SIMULATION_SPECS.md`, not here, so that the detailed, fast-moving design (batching strategy, solver algorithm, GPU-compute framework choice) does not require a `SPECS.md` revision each time it changes. That document is grounded in direct research against the reference's own source/documentation and the Rust-ecosystem GPU-compute candidates of Appendix B.2.1 (Burn/CubeCL, Candle, and `wgpu` directly), not a restatement of Rapier qualification. It does not grant the the reference simulator-binding exception this section's closing sentence withholds: the Rust-native requirement stands, and the reference's own implementation is a design reference, never a dependency. `docs/decisions/0005` records this as a decision, and C-25 tracks what the GPU-batched simulation specification still leaves open.

Joint limits are hard constraints in the simulator: a hinge MUST NOT exceed its USD lower or upper limit, and no soft-limit overshoot is accepted. Each hinge MUST also report a proximity signal as it approaches a limit. The signal is defined on the fraction of the hinge's total range (upper minus lower limit): it is raised when the joint is in the lowest 5 % or the highest 95 % of that range, so that downstream consumers see a limit approaching before it is reached. The 5 % and 95 % values are declared in the profile. Effort and velocity limits are recorded alongside.

Collision in simulation has two selectable modes, chosen per run. The primitive mode uses spheres, boxes and cylinders, with pelvis and torso proxies (Section 7.1), and is for general pretraining. The convex-hull mode uses one convex hull per collision mesh, and is for detailed fine-tuning. The ground-contact law is the penalty normal force with regularised Coulomb friction of ADR 0010, applied at each contact point, and each point MUST report its penetration and force.

Self-collision MUST be modelled between every pair of links that are not directly connected by a joint, honouring the collision filters of the USD model. Primitive pairs are implemented first. Two adjacent links MUST NOT collide at the joint they share; a limited overlap at that joint is allowed, with its allowed depth declared in the profile. The initial value is 3 mm. This value is a placeholder: for primitive shapes the overlap problem remains at any depth, and a per-joint algorithm may be needed to find the balance between joint overlap and real contact. That algorithm is not yet specified.

Performance is not budgeted on the current development machine. Runtime targets, optimisation and the convex-hull and self-collision costs MUST be measured later on a GPU-bound machine; no run on the current machine may be reported as meeting a runtime target.

The simulator's ground contact MUST cover every collision shape that the USD model declares for a link (Section 7.1). Sphere shapes are contact points at their centres. Box and cylinder shapes contribute contact points at their corners and rim samples. Mesh shapes contribute their vertices as contact points, sampled to a stated budget. The contact law is the penalty normal force with regularised Coulomb friction of ADR 0010, and each contact point MUST report its penetration and force, so that a run can show which shape touched the plane.

Until that is implemented and measured, the floating-base result is a limitation, not physics. The current `robot sim multibody --floating` run uses sphere contact only, and its zero-torque collapse through the plane MUST be reported as that limitation. A supported-stance result (a clamped PD hold, judged over 2 s on pelvis drop, penetration and settled energy, valid only for the dt range of ADR 0010) MUST NOT be reported as balance. Static-standing, foot-slip and push-recovery results MUST NOT be reported as capability until contact for all declared shapes, a controller and those benchmarks are implemented and qualified. A free-fall run is not evidence of contact behavior.

### 9.2 Baseline training

Implement PPO with generalized advantage estimation, bounded actions, observation normalization, checkpoint/resume, recurrent policy support when needed, and explicit handling of termination versus time-limit truncation. An asymmetric critic may use simulator-only state; exported actors must pass an observation audit proving no privileged inputs remain.

Train balance and locomotion first, then target-specific motion tracking. Rewards and costs should cover task completion, reference tracking, balance, energy, action smoothness, joint/actuator constraints, and undesirable contacts. Log reward terms independently to expose reward exploitation. Persistent torque violations cannot be hidden by a good aggregate return.

Domain randomization must be calibrated and sensitivity-driven. Include plausible mass/inertia, friction, motor response, backlash where relevant, delay, sensor bias and contact parameters. Reserve held-out disturbances and parameter combinations for evaluation.

Checkpoint/resume MUST store, through the chunked format of Section 7.5, the weights, optimizer state, RNG state and observation normalizer statistics, so that the normalizer is always restored with the weights it was trained with. Fine-tuning MUST be able to restrict updates to a named parameter group (a named slice of weights) that is marked `frozen`. Until parameter groups exist, only whole-network updates are available, and every partial-adaptation method of Section 11.3 depends on this capability.

### 9.3 Algorithm interface

Every algorithm supplies a capability manifest and implementations for initialization, observation requirements, action contract, experience collection, update, checkpoint, export, and evaluation. Configuration includes compute budget, seeds, prerequisites and expected scientific claim. The manifest MUST declare stable method ID and version, supported interfaces/capabilities, observation and action schema versions, required data, feature gate, determinism limits, and artifact compatibility.

The following are separate extension points. An implementation MUST be addable behind the named interface and a Cargo feature without requiring edits to unrelated core crates. The dispatch column states the permitted form; Section 9.6 defines the common requirements.

| Extension point | Contract boundary | Dispatch and scope |
| --- | --- | --- |
| Algorithm / learner | Initialization, collection/update semantics, checkpoint and evaluation capabilities declared in the manifest | The learner is offline and MAY use dynamic dispatch; any implementation shared with deployed inference MUST use static dispatch on the control path |
| Environment and vectorization backend | Reset/step, termination vs truncation, snapshots, stable environment IDs and per-environment RNG | Dynamic dispatch MAY be used offline; never a control-path dependency |
| Rollout and replay storage | Shared transition/episode schema and bounded storage from Section 7.6 | Dynamic dispatch MAY be used offline; sampler is a separate interface |
| Policy / model module | Observation/action contract, recurrence, inference state and exportable operator set | Compiled implementation and static dispatch on a control path; offline training MAY compose modules dynamically, but exported actor must satisfy Sections 7.4 and 10.3 |
| Reward-term registry | Named, versioned reward/cost terms and their logged components | Name-keyed registry built before an offline run; no control-path lookup |
| Curriculum / randomization provider | Scenario proposals, parameter distributions, bounds and held-out-set exclusions | Name-keyed registry built before an offline run; dynamic dispatch offline only |
| Evaluation hook | Metrics, fixtures, baselines, statistical report and acceptance-profile reference | Dynamic dispatch MAY be used offline; evaluation cannot modify held-out sets |
| Export path | Validated conversion from training state to versioned policy artifact and parity checks | Dynamic dispatch MAY be used in tooling; deployed loading follows the fixed, validated bundle contract and Section 5 |
| Skill-space / latent interface | Encoder, decoder and prior over a learned action space: dimensionality, bounds or manifold constraint, versioned identity, and the decoder artifact a latent transition depends on | Offline composition is free; a deployed decoder is a tier of Section 5.1 with static dispatch and the exportable operator set of Sections 7.4 and 10.3 |
| Opponent and population provider | Which policies an agent trains against: latest snapshot, sampled historical population, learned average policy, scripted opponent or held-out evaluation pool | Name-keyed registry built before an offline run; the held-out pool is defined independently of the provider, per Sections 9.4 and 9.5 |
| Experience-retrieval provider | Query construction, similarity metric, retrieval unit, admission filter and the provenance labels written onto retrieved data | Offline and between-episode only; never a control-path dependency, and subject to the tier rules of Section 10.6 |

Support on-policy rollout buffers initially; add replay only for algorithms that need it, using Section 7.6. A shared learner interface MUST NOT pretend that PPO, offline imitation, supervised distillation, model identification and self-play have identical update semantics. A name-keyed registry MAY select among implementations already linked into the offline executable, but it MUST be constructed once before work begins and MUST NOT support register/unregister or lookup from a control tick. This is build-time composition, not dynamic code loading; Section 6's prohibition remains binding.

Research intake workflow: verify source → extract executable assumptions → implement minimal Rust baseline → reproduce a small benchmark → compare fixed budgets → promote or reject with evidence. Pin paper versions and commits. Imported mathematical ideas must have ablation tests and separate feature flags.

The policy, simulator backend and robot-model importer are required extension points, but their current implementations are concrete types rather than traits: the MLP and Gaussian policies, the multibody solver and the robot-model loaders. Each MUST be introduced behavior-preservingly, as a trait with the existing implementation behind it and the existing tests unchanged, before a second policy type, backend or importer is added. The `Environment` and hardware-adapter traits are the reference pattern for this change.

### 9.4 Hierarchical skills and competitive training

After tracking works, compare:

- A single compact tracker conditioned on skill ID or reference motion.
- A latent-conditioned tracker with learned encoder/prior and distillation losses.
- A bank of specialized compact experts with shared observation/action contracts.

#### 9.4.1 The staged latent pipeline

RoboStriker (arXiv:2601.22517) is `paper_verified` as of 2026-10-26 and is adopted as the **reference decomposition** for latent competitive learning, not as a result this project has reproduced. The method it describes, and which Section 9.4.2 turns into requirements, proceeds in stages that are separately trainable and separately falsifiable:

| Stage | What is learned | Why it is a separate stage |
| --- | --- | --- |
| 1. Motion tracker | A single-agent policy tracking retargeted capture references from proprioception, a short reference window and the previous action, trained with weighted exponential tracking terms, regularization and early termination | Produces physically consistent motor competence before any task or opponent exists; failures here are tracking failures, not strategy failures |
| 2. Latent skill distillation | A student encoder, decoder and state-conditioned prior distilled from the frozen tracker, with the latent distribution constrained to a bounded manifold and regularized toward the prior | Converts a reference-driven tracker into a sampleable action space; the prior is what later keeps a strategy inside physically plausible regions |
| 3. Behavioral warm-up | A latent-space policy driven by goal-oriented, ego-centric task observations against a passive or scripted target, with a motion-prior style term alongside the task reward | Overcomes reward sparsity before a reactive opponent exists; the published ablation that skipped it collapsed |
| 4. Competitive co-evolution | Self-play in the latent space with opponent averaging: a best-response learner and a supervised average policy over a reservoir of the agent's own historical best-response behavior, mixed by an anticipatory parameter | Opponent averaging is what addresses strategy cycling; self-play directly in the actuator action space was the weakest published variant |

The transferable claims are the ordering and the separation of concerns: skills before tactics, a bounded action space before competition, a scripted target before a reactive one, and averaged opponents before a claim of progress. The specific values the paper reports — a 32-dimensional unit hypersphere, a 50 Hz policy rate, a 29-degree-of-freedom robot, the reward weights and the reservoir capacity — are conditions of that experiment, not settings this project inherits. They were obtained in a foreign simulator excluded by Section 2, on one embodiment, with no inspected source release.

#### 9.4.2 Requirements for a staged latent pipeline

1. Stages MUST be separately trainable, separately checkpointed and separately evaluated. Each stage declares what it consumes, what artifact it produces and the measurement that qualifies it. A later stage MUST NOT be used to excuse an unqualified earlier one: a tracker that does not track is not repaired by competition.
2. Each stage MUST be a distinct capability manifest under Section 9.3, behind its own feature flag. Deleting the competitive stage MUST leave a usable tracker, and deleting the latent stage MUST leave the conditioned-tracker alternative of the comparison above buildable.
3. A learned latent action space MUST declare its dimensionality, its bound or manifold constraint, its versioned identity and the decoder artifact that interprets it, per Section 7.6 item 8 and the skill-space extension point of Section 9.3. The dimensionality and the constraint are measured choices from the ablation required below, never copied from a paper.
4. A prior or divergence penalty that keeps a strategy near the distilled manifold is a **training constraint**. It MUST NOT be recorded as a hazard control under Section 12.2, and the supervisor of Section 12.1 MUST remain independent of it. Section 3 already states that a bounded latent bounds a representation and not a contact force.
5. A style or motion-prior term, including an adversarial discriminator reward, is likewise a style objective. It MAY be used to keep behavior near the capture distribution; it MUST be logged as a separate reward component per Section 9.2, and a report MUST NOT present a style score as evidence of feasibility, stability or safety.
6. Competitive training MUST maintain, at minimum, the current policy, a historical opponent population and a held-out evaluation pool, through the opponent-provider extension point. The held-out pool MUST be defined independently of the training opponent distribution, as Section 9.5 item 3 already requires of proposed distributions.
7. An opponent-averaging mechanism — a learned average policy over a reservoir of historical best responses, uniform sampling from a snapshot population, or an equivalent — MUST be implemented and MUST be ablated against naive latest-snapshot self-play. Anti-cycling is a claim that requires the comparison, not an architecture diagram.
8. The reservoir or population store is a store over the canonical schema of Section 7.6, not a private format. Its retention rule, capacity, admission probability and seed stream are recorded, and a run that cannot reproduce its sampled population is not reproducible.
9. The staged pipeline MUST satisfy Section 5.1 when deployed. The strategy layer runs at the tactical tier and the decoder at the policy tier or faster; where an experiment runs both at one rate, that is recorded and the tiered configuration is re-measured before any hardware claim.
10. The required comparison is against the two non-latent alternatives listed at the head of this section, at equal budget, reporting task outcome, physical stability and command smoothness separately. A latent pipeline that wins only on the tactical metric has not justified its additional stages.

Boxing self-play requires robust hit attribution, contact filtering, target zones, stance/fall rules, episode termination, ring boundaries and task-specific force/energy budgets. Validate scoring against scripted scenarios before optimizing it. Maintain historical opponent snapshots, varied scripted opponents, held-out evaluation opponents and anti-cycling measurements. Victories against the latest self are insufficient evidence of progress. Contact-derived rewards and terminal outcome conditions are scoring definitions under Section 12.2's energy budget, and the budget is enforced outside the policy that is rewarded by it.

### 9.5 Assisted sampling in policy design

Policy quality depends on what is sampled: which priors are tried, which curricula and randomization ranges are explored, which scenarios and opponents are held out, and which expressive variations are offered to Section 12.5. Proposing candidates for those sets is the second admissible use of an agent layer (U2 of Section 2.3.1). It is offline design assistance, operating between experiments on a developer host, and the following are binding:

1. Assisted sampling MAY propose candidate priors, curricula, randomization ranges, scenario and opponent variations, reward-term candidates and expressive style variations. It MUST NOT select among them, score them, weight them or tune a threshold. Selection is the measurement defined by Sections 9.2, 9.4, 10.5 and 14, unchanged.
2. A proposal is admitted only through the algorithm interface of Section 9.3: a manifest, an ablation and a separate feature flag, with the expected claim stated before the run. An imported idea whose only support is that a model suggested it has evidence state `unverified` and licenses nothing, per Section 3.
3. Proposals MUST NOT be allowed to narrow the held-out set. Evaluation seeds, held-out disturbances and held-out opponents are defined independently of whatever proposed the training distribution; otherwise the generator and the examiner are the same process.
4. Every accepted proposal records its origin — tool, model identity and version, request parameters, date — in the run record of Section 13, and a model change demotes the claims resting on it per Section 3. The accepted proposal itself, once reproduced in our code and measured, stands on that measurement rather than on its origin.
5. Assisted sampling runs only offline. It is not a training-time component, not a reward term, and not a runtime sampler inside a policy. Nothing in the deployed bundle of Section 7.4 may call it.
6. Negative results are retained. A proposal set that produced no measured improvement is recorded as such, so the same suggestions are not re-sampled indefinitely.

Durable orchestration of long training campaigns — resumable runs, parked approvals, event replay — belongs to the same offline class, under the experimental-status restriction of Appendix B.2.13 and the promotion procedure of Section 11.1: a tool that records an approval does not grant it.

### 9.6 Extensibility requirements

The interfaces in Section 9.3 and the data contract in Section 7.6 exist to make methodology replaceable without turning the learner or deployed controller into a plugin host. The following requirements apply to every extension:

1. Public extension traits MUST evolve additively. A new optional capability MUST use a new subtrait, adapter, or method with a safe default; it MUST NOT add a new required method to an existing implementor-facing trait or silently change established semantics. Breaking changes require an explicit schema/API version and migration plan.
2. Every implementation MUST publish a machine-readable capability manifest with a unique stable name, implementation/schema versions, supported observation/action/data contracts, declared limits, and feature gate. Capability names MUST be validated for uniqueness before a run starts. Cargo features select code at build time; they MUST NOT enable loading executable code at runtime.
3. Each optional algorithm or methodology MUST have its own feature flag. The default feature set remains the qualified baseline, and disabling an extension MUST leave the baseline buildable and its applicable acceptance rows closable.
4. Generic parameters or closed enums SHOULD be preferred where the concrete implementation is known at build time, particularly for bounded or deadline-critical paths. Trait objects and name-keyed registries MAY be used for offline experiment composition only. A registry is constructed once at process startup from linked implementations; it MUST NOT mutate during a run. No registry lookup, dynamic dispatch, or deserialization of tagged trait objects is permitted inside a deadline-critical path.
5. Every extension implementation MUST pass the common conformance suite before it is enabled in an experiment: schema/capability validation; termination and truncation semantics; deterministic fixture and seed handling; checkpoint/resume; malformed-input and failure behavior; and, where export is claimed, training/inference parity and artifact validation. Tests MUST include at least one negative case and be independent of implementation-specific expected outputs.
6. A feature-removal test MUST demonstrate that disabling or removing an optional method does not break the baseline learner, shared data schema, or deployable controller. An extension test MUST demonstrate that a conforming implementation can be added without edits to the control loop or unrelated core crates.

Registry implementations and other dependency candidates are listed in Appendix B and remain subject to B.6. Appendix D explains the Rust mechanisms and their limits; it does not approve a dependency or relax Sections 2, 5 or 6.

### 9.7 Retrieval-augmented policy learning

Retrieval-augmented policy learning addresses the data problem directly: when the target behavior has only a handful of demonstrations or real episodes, a retriever selects the most relevant fragments of a large prior corpus and those fragments join the training set. The published few-shot results in this family — sub-trajectory retrieval matched by subsequence alignment over foundation-model embeddings, behavior- and flow-similarity retrieval, and experience retrieval for adaptation — share one structure and one caveat. The structure is query, corpus, retrieval unit, admission filter, and a learner that trains on the union of target data and retrieved data. The caveat is that this is **offline data selection**, and Section 3 forbids reporting it as a runtime capability.

This section governs retrieval that produces *training data*. Retrieval that produces a *runtime decision* is the policy bank of Section 10, and the two MUST NOT be described as one mechanism or share a measurement.

#### 9.7.1 Contract

A retrieval-augmented method is admitted through the experience-retrieval provider of Section 9.3 and MUST declare each of the following in its manifest:

| Element | Requirement |
| --- | --- |
| Corpus | The exact dataset revisions, their licenses under Section 7.5 and their embodiment provenance. A corpus assembled from robots other than the target is a cross-embodiment claim under Section 11.1 |
| Query | What is matched: a target demonstration, a sub-trajectory, a goal, a state window or a task descriptor; and the observation fields it may legally read |
| Retrieval unit | The granularity returned — full episode, fixed window, aligned sub-trajectory or single transition — recorded as a view over the canonical schema of Section 7.6, never as a new format |
| Similarity | The metric, the embedding or feature source, its version, and whether alignment is exact, approximate or time-warped. Semantic similarity is not physical compatibility; Section 10.1's distinction applies here too |
| Admission filter | The feasibility, embodiment-compatibility and license checks a retrieved fragment passes before it may enter a batch |
| Mixing | How retrieved and target data are weighted or scheduled in the update, and what the learner does when retrieval returns nothing |

#### 9.7.2 Requirements

1. Retrieval MUST run offline or between episodes. No corpus query, index lookup or embedding computation may occur on the control path, under Section 5 and the tiers of Section 10.6.
2. Every retrieved transition MUST be labeled with its source corpus, source episode, embodiment and retrieval generation, per Section 7.6 item 7. A batch whose composition cannot be reconstructed is not a reproducible experiment, and a result that cannot name what it trained on is not a result.
3. The retriever MUST NOT be allowed to reach held-out data. Evaluation episodes, held-out disturbances, held-out opponents and the validation session of Section 11.3 MUST be excluded from the corpus by construction and the exclusion MUST be tested, not asserted. A retriever that can reach the test set is the Section 9.5 item 3 failure in a new form.
4. A retrieval-augmented result MUST be compared against the same learner trained on the target data alone and against the same learner trained on the whole corpus, at equal update budget. Without both, an improvement cannot be attributed to selection rather than to data volume.
5. Retrieval quality MUST be reported separately from task outcome: how many fragments were retrieved, how many the admission filter rejected and why, the corpus coverage actually used, and the sensitivity of the outcome to the number retrieved. A single success rate hides a retriever that is doing nothing.
6. Embedding models used to build a corpus index are offline artifacts subject to Sections 2 and 2.3. A hosted embedding service is an external model service; an index built with one records the model identity and version, and a model change demotes dependent claims under Section 3.
7. Exact search over the corpus is the default implementation, per Appendix B.2.5. Approximate indexing is adopted only after the crossover measurement, and recall loss MUST be reported as a property of the training set it selected, not hidden in a build step.
8. The retriever MUST be removable. Disabling the feature leaves the baseline learner, the shared schema and the applicable acceptance rows unchanged, which is row A-30 applied to this extension point.

## 10. Retrieval-indexed policy bank and low latency

### 10.1 Two kinds of similarity

Semantic similarity organizes intent: “jab”, “retreat”, “guard”, “celebrate”. Physical compatibility determines what may execute now: stance, contact mode, joint velocity, available support, reach, actuator health and opponent-relative geometry.

The retriever therefore produces candidate skills; a state-conditioned feasibility filter and fast router select executable candidates. Text embeddings may be computed offline with a Rust model. Start with structured skill tags and numerical features, and add learned embeddings only when they improve a measured outcome.

For small libraries, benchmark exact vector search in Rust before adopting approximate nearest-neighbor indexing. A remote vector database is not part of the control path.

A skill MAY additionally carry a human-readable descriptor — capabilities, preconditions, failure modes and counter-indications as Markdown — indexed at build or load time and matched lexically. Such a descriptor is documentation and operator-facing catalog, and its indexing involves no model call; it is data under Section 2's opening paragraph, not code. It MAY contribute a routing signal that composes with the numerical retrieval of Section 10.2, and it MUST NOT replace it or relax any feasibility check: prose asserting that a skill is applicable is not a statement about joint velocity, support or reach. The descriptor's authoritative fields remain the typed ones of Section 10.2, and a disagreement between the two is a defect in the descriptor.

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

### 10.6 Retrieval tiers and deadline attribution

"Retrieval" names three different operations in this project, at three different costs. They MUST be distinguished in design, in configuration and in every report, because the published latency numbers for one say nothing about another.

| Tier | Operation | Where it runs | Binding constraint |
| --- | --- | --- | --- |
| R0 — corpus retrieval | Selecting prior trajectories or sub-trajectories to train on, per Section 9.7 | Offline, on a developer or CI host | No deadline; subject to license, provenance and held-out exclusion rules. Never present in a deployed bundle |
| R1 — between-episode retrieval | Choosing a working set of experts, an opponent model or an adaptation context before a bout or between rounds | Offline-equivalent, outside the control process | Completes before the mode entry preconditions of Section 4 are satisfied; a failure degrades to the previous working set, never to a late load |
| R2 — in-loop selection | The skill router and feasibility filter of Sections 10.1 to 10.4 | Inside the control process, at the tactical tier of Section 5.1 | Bounded local computation over resident data only, with stale-output rejection and measured tails; this is the only tier with a deadline |

The following are binding:

1. An R2 lookup MUST operate on a resident, preloaded index. It MUST NOT read a file, query a database, compute an embedding, allocate, or block on a staging operation. Index updates occur at R1 and are published to R2 as generation-tagged immutable handles under Section 10.3.
2. R2 cost MUST be measured as part of the policy-path compute duration and reported in the tails of Section 5, separately from decision age. An asynchronous retrieval that merely moves work off the control thread converts latency into decision age; Section 3 already records this and the report MUST show both numbers.
3. A method validated at R0 MUST NOT be promoted to R1 or R2 on the strength of that validation. Each tier requires its own measurement, and moving a mechanism downward is the Section 5.1 item 5 prohibition.
4. Published inference-latency figures for retrieval-augmented systems are evidence about their setting only. A reported millisecond figure obtained on other hardware, at another rate, with another index size, licenses nothing here; Appendix B.2.5 requires the crossover and worst-observed timing on target hardware.
5. Caching a previous retrieval result is permitted at R1 and R2 only with an explicit expiry and a recorded decision age. A cache without an expiry is a stale decision with better latency, which is not an improvement.
6. Every tier MUST have a defined empty-result behavior. R0 falls back to target data alone, R1 retains the current working set, and R2 retains the current controller or invokes the fallback of Section 10.4. No tier may stall a decision while waiting for a retrieval.

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
| Two-phase latent identification (RMA-class) | Phase 1 trains a privileged-conditioned policy in randomized simulation; phase 2 trains a small adaptation module to regress that latent from a recent proprioceptive window, and only the module's output enters the deployed observation | A regressed latent is only as good as the randomization ranges it was fitted over; it reports a plausible context, not a measured parameter, and it fails silently outside them |
| Simulation-guided fine-tuning | Real-world updates shaped by a value or objective term derived from simulation rather than by unguided real exploration | Requires on-hardware updating, which Section 4 permits only under an explicitly enabled mode; the guiding term inherits every error of the simulator that produced it |
| Retrieval-conditioned adaptation | Select prior episodes similar to the observed deployment condition, per Section 9.7, and adapt on the union with the few real episodes | Similarity in observation space is not similarity in dynamics; the admission filter, not the metric, is what makes it safe |
| Cross-embodiment few-shot transfer | Pretrain over several embodiments with morphology conditioning, then adapt to a held-out robot from a small sample | Section 11.1's requirements apply in full; a held-out-robot result is evidence about the pretraining set, and a single target robot is not a transfer claim |
| Multi-simulator or multi-model randomization | Randomize across distinct dynamics models rather than only over parameters of one | Requires a second qualified simulator, which Section 9.1 does not currently grant; otherwise it is parameter randomization with extra vocabulary |

Implement identification first, then history adaptation; treat residual RL and online weight updates as later experiments. Do not require paired perfect simulated/real state at execution. Do not assume a unique target action correction can be inferred from state error alone.

The rows above are ordered by increasing hardware exposure, and that ordering is binding: a method MUST NOT be scheduled before the methods above it have been measured and found insufficient on the same robot. Each method additionally carries the following obligations.

1. Each adaptation method is an extension under Sections 9.3 and 9.6, with its own manifest, feature flag and conformance run. Adaptation is not a special case exempt from the extension rules.
2. Every method MUST declare the observation set available at deployment and MUST NOT consume privileged simulator state at inference. This is the existing Section 9.2 audit applied to adapters, and it is the specific defect Section 3 records in the earlier residual sketch.
3. Any component that runs on the robot — an adaptation module, a context encoder, a residual head — is a tier under Section 5.1 with a declared rate, a measured budget and a defined behavior when its input history is short, stale or discontinuous. An adaptation module that has not yet filled its history window MUST produce the declared default context, not an extrapolation.
4. Adaptation authority MUST be bounded and the bound enforced outside the adapted component. A learned correction is clamped by the supervisor of Section 12.1 like any other command, and widening a bound to accommodate an adapter is a safety-profile change, not a tuning step.
5. Every method MUST be evaluated against the frozen pre-adaptation baseline on the held-out session, reporting both improvement and regression, and MUST support rollback to that baseline as an immediately available artifact under Section 13.
6. Sample-efficiency claims MUST be stated in the units that cost something: real minutes, real episodes, physical interventions and wall-clock downtime. Simulation steps consumed by the method belong in the same report, because a method that needs a week of simulation to save ten minutes of hardware time has moved the cost rather than removed it.
7. On-hardware weight updates remain a later experiment and require the mode gating of Section 4 plus a separate decision; nothing in this table authorizes continuous learning on a robot.

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

Opponent memory, where it is implemented, is a persistent store written between rounds or between bouts and read at strategy rate. Appendix B.2.13 identifies a candidate mechanism — a versioned entity/observation graph — for what Section 3 previously recorded only as an unverified persistent-identity sketch. Any such store is subject to three constraints: its read path MUST NOT depend on network availability or on a hosted backend, so an embedded or local backend is required wherever it is consulted during operation; its contents are an input to strategy, never to the supervisor or the motor loop; and a store that is unavailable, empty or stale MUST degrade to the opponent-agnostic behavior rather than stalling a decision. Retention, consent and deletion obligations for recorded opponent data follow Section 7.5.

Where personality is expressed as a set of style parameters and expressive-skill choices, the candidates for that set MAY be proposed by the offline assisted sampling of Section 9.5. A personality is then a sampled, versioned configuration over already-validated skills, recognizable and testable under the metrics above. It is not a generated behavior: no runtime model call produces posture, cadence, celebration or tactical preference, and a style variation that has not passed the same validation as a task skill is not available to be expressed.

### 12.6 Agent-layer governance

If the optional agent layer of Section 6 is ever enabled, it is governed as a supervised input source and not as a controller:

1. A guardrail, schema validation or audit-logging component MAY be an additional, earlier check on an advisory record. It MUST NOT be described or relied upon as a safety control, and it MUST NOT be placed in the position of the supervisor of Section 12.1, which remains the last stage before the actuator interface and MUST NOT import it.
2. Under Section 12.2, a hazard whose only control is a model-produced judgment is not controlled. Controls MUST remain independent of the agent layer, exactly as they must remain independent of the learned policy.
3. Agent-layer outputs that reach an operator — advisories, commentary, speech — MUST be distinguishable in the record from measured state, so that a generated statement is never read back as an observation.
4. Every agent-layer decision, guardrail intervention and approval gate is a logged artifact under Section 13, with the model identity and version that produced it.
5. Human approval gates implemented through an agent runtime are a convenience over a procedure, not a substitute for one. Promotion to hardware follows Section 11.1 and Section 14 regardless of which tool records the approval.

## 13. Reproducibility and operational requirements

Every run records source revision, compiler/toolchain, the `devenv.lock` revision that supplied it, dependency lock, hardware/driver details, seeds, data hashes, robot model, simulator configuration, reward definition and all model versions. A run whose environment cannot be named is not reproducible, whatever else it recorded. Determinism is a documented mode and tolerance, not an assumed property of GPU execution.

Provide restartable training, atomic checkpoints, bounded storage quotas, artifact retention, structured metrics and replayable failure episodes. Production bundles are immutable. Promotion includes compatibility validation, evaluation report and an immediately available last-known-good artifact.

Treat downloaded datasets and model files as untrusted input: enforce size limits, reject malformed dimensions, fuzz parsers and verify checksums. Restrict command-network access and authenticate remote control where the transport allows it. This is operational integrity, not a dependency on a cloud control plane. A response from an external model service of Section 2.3 is untrusted input of the same kind and is validated before use.

A hosted model is not reproducible by recording its name. Where any part of a reported result passed through an external model service, the record MUST name the provider, model identifier, version or snapshot, request parameters and date, and MUST state that the result is not replayable if the provider does not guarantee a pinned snapshot. A durable event log produced by an agent runtime is an operational record, and it becomes a reproducibility artifact only once that runtime itself is out of experimental status and qualified under Appendix B.6.

### 13.1 Low-level dependency knowledge base

A humanoid fails at its lowest layers: a firmware revision, a bus timing parameter, a driver quirk, an actuator thermal curve, a kernel option, a crate version with a known defect. Diagnosing such a fault by rediscovering the layer each time is the normal and the expensive outcome. The third admissible use of an agent layer (U3 of Section 2.3.1) is to help assemble and maintain a **dependency knowledge base**: a project-owned, versioned record of every low-level dependency the system relies on, so that a hardware fault is resolved against maintained project knowledge rather than against a search engine.

The knowledge base is an artifact of this project, not a service. The following are binding:

1. Scope is the full low-level inventory: Rust dependencies and their native linkage, the OS, kernel options, drivers and firmware of Section 2's platform class, bus and transport parameters, actuator and sensor interface documents, and the per-robot profile versions they apply to. The inventory required by Section 2 and Appendix B.6 item 1 is its primary key, so an entry cannot exist for something the dependency audit does not know about.
2. Every entry MUST be **cited and attributed**: source document, version and date, who or what added it, and the evidence state of each claim under Section 3. An uncited entry is a note, and a note MUST NOT be read as a specification.
3. The knowledge base MUST NOT be cited as evidence in a release report, and it MUST NOT be the control for any hazard under Section 12.2. It shortens diagnosis; it does not close an acceptance row. A claim that matters is promoted through Section 3 against the primary source, and the knowledge-base entry then references the ledger rather than substituting for it.
4. Entries MUST be invalidated by version change. A firmware, driver, crate or robot-profile change marks every dependent entry stale, exactly as it demotes a claim under Section 3. A stale entry is visibly stale, not silently wrong.
5. The knowledge base MUST be readable and usable without any model service: it is text and structured records under version control in this repository, indexed locally. Assisted retrieval or assisted drafting over it is a convenience under U1, subject to Section 2.3 in full, and its unavailability degrades to reading the records directly.
6. Fault records close the loop: a diagnosed hardware fault produces an entry naming the symptom, the measurement that identified the layer, the resolution and the dependency version it applies to, linked to the episode records and incident channel of Section 5. This is what makes the base a project asset rather than a collected reading list.
7. The knowledge base is operational knowledge and MAY contain supplier or robot-specific material; its retention and distribution follow Section 7.5 and the licensing decision of [C-13](#appendix-c--open-questions-and-unresolved-decisions).

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
| A-23 | Agent-layer qualification, if adopted | §2.3, §6, §12.6, B.6 | Only the per-crate qualified, non-model subset enters the build at a pinned version; experimental crates carry no production duty; advisory records are schema-validated, expiring and rejected when stale, with rejection counts reported; no agent component is reachable from the control path or the supervisor |
| A-24 | Offline degradation | §2.3, §12.6 | With every external-model-service feature disabled, the full autonomous control loop, safety envelope and all applicable rows of this matrix pass unchanged; each dependent capability either continues or fails safe with the service unreachable |
| A-25 | Development-time automation hygiene | §2.3.1, §2.2 | Automation runs only on developer and CI hosts inside the locked environment, never on a robot host; every generated artifact that entered the repository is attributable and passed the same review, schema validation and CI gate as authored work; removing the automation leaves every deliverable buildable and every row of this matrix closable |
| A-26 | Assisted sampling discipline | §9.3, §9.5 | Each accepted proposal has a manifest, an ablation and a prespecified claim; held-out seeds, disturbances and opponents were defined independently of the proposal source; proposal origin and model version are recorded, and negative results are retained |
| A-27 | Dependency knowledge base | §13.1, B.6 | Every low-level dependency in the Section 2 inventory has an entry with citation, version and evidence state; version changes mark dependent entries stale; the base is readable with no model service available; closed hardware faults have fault records linked to their episode records; no acceptance row or hazard control cites the base as its evidence |
| A-28 | RL extension conformance | §9.3, §9.6 | Every enabled learner, environment, storage/sampler, policy/model, reward, curriculum, evaluation and export implementation passes the shared conformance suite, including schema/capability validation, deterministic fixtures, checkpoint/resume, malformed-input behavior and applicable export parity |
| A-29 | Experience and replay storage | §7.3, §7.5, §7.6 | On-policy and off-policy paths consume the same versioned transition/episode schema; chunk bounds, ring wrap/overwrite, stale-generation rejection, sampler probabilities/weights, target views, migrations and seeded replay are tested on representative data |
| A-30 | Extension and removal isolation | §6, §9.3, §9.6 | A new conforming method can be added behind its existing interface/feature without edits to unrelated core crates or the control loop; disabling/removing it leaves the baseline build, shared schemas and applicable controller acceptance rows unchanged |
| A-31 | Robot profile extensibility | §7.1, §7.5 | Typed stable IDs, namespaced extension tables, capability validation, unknown-field rejection on activation, tooling round-trip preservation and version-to-version migrations are tested with valid, unknown and malformed profile fixtures |
| A-32 | Hierarchical latency tiers | §5.1, §7.4 | Every learned component declares a tier; tier rates are non-increasing with height; per-tier compute duration, deadline margin and decision age are measured separately under representative load; each tier's behavior with a late, absent or expired upper-tier output is tested; removing the highest tier leaves a safe controller |
| A-33 | Staged latent pipeline | §9.4, §5.1 | Each stage is separately checkpointed and separately qualified; the latent space declares dimensionality, bound and decoder identity; the style and prior terms are logged as separate components and cited as no hazard control; opponent averaging is ablated against latest-snapshot self-play on held-out opponents; the pipeline is compared at equal budget against the conditioned tracker and the expert bank |
| A-34 | Retrieval-augmented learning | §9.7, §7.6 | Corpus revisions, licenses, retrieval unit and similarity version are recorded; every retrieved transition carries its provenance; held-out exclusion from the corpus is tested rather than asserted; the method is compared against target-only and whole-corpus training at equal update budget; retrieval and admission-rejection counts are reported; disabling the retriever leaves the baseline unchanged |
| A-35 | Retrieval tier discipline | §10.6, §5 | R0, R1 and R2 are separately configured and separately measured; no R2 path performs file, network, embedding or allocation work; R2 cost appears in the policy-path tails and decision age is reported alongside it; each tier's empty-result behavior and cache expiry are exercised |
| A-36 | Simulation fidelity tiers | §9.1, `SIMULATION_SPECS.md` §12.2 and §14 | Every declared fidelity tier passes one shared battery of analytic references, energy-drift bounds and round-trip checks, with a stated tolerance for each check; each run writes a versioned report through `robot-artifacts`; a failed check is reported as failed, never omitted. The CPU-vs-GPU agreement and contact-budget counter checks are recorded as pending until a GPU backend exists |
| A-37 | Checkpoint and partial fine-tuning | §9.2, §9.3, §7.5 | A checkpoint round-trip restores weights, optimizer state, RNG and observation normalizer; resume equivalence holds within a recorded tolerance; named parameter groups with `frozen` flags leave frozen weights bit-identical after a fine-tuning run |
| A-38 | Plug-in seam traits | §6, §9.3, §9.6 | The policy, simulator backend and robot-model importer boundaries are traits with their existing implementations behind them; the pre-existing test suite passes unchanged; a second policy type can be added without edits to the learner or control loop, shown by a test |
| A-39 | Test-purpose frame output | §5, §14 | `robot sim` output can be written as a versioned per-step frame dump (CSV or JSON Lines) that reads back with identical frame count and values; any std-only rendering is checked by golden-file tests; no rendering dependency enters the default closure before Appendix B.6 qualification and an ADR |
| A-40 | USD model import | §7.1 | The vendored G1 layers import with the measured counts of links, joints, actuators and collision shapes of each primitive kind; joint limits, effort and velocity agree with the URDF of the same revision within the stated tolerance; zero-pose link positions agree with it to 1e-6 m; the Omniverse-only materials are reported as warnings and the import completes |
| A-41 | USD replay validity | §7.3 | A recorded run's `replay.usda` passes OpenUSD's `usdchecker` and imports in a USD viewer with the link animation present (frame count and the pelvis track checked against the simulator output) |
| A-42 | Contact coverage | §9.1 | Every collision shape the USD model declares for a link takes part in ground contact: a probe that drops each shape onto the plane shows penetration and force from that shape; the zero-torque run reports the shape that first touches the plane |
| A-43 | Hard joint limits | §9.1 | No hinge exceeds its USD lower or upper limit in any simulated run; the limit-proximity signal is raised inside the declared margin band before the limit is reached |
| A-44 | Primitive contact | §9.1 | Each primitive proxy and USD primitive shape (sphere, box, cylinder, including pelvis and torso proxies) produces penetration and force when dropped on the plane |
| A-45 | Convex-hull contact | §9.1 | Each collision mesh's convex hull produces contact that matches the primitive-mode result to a stated tolerance on a shared test, and the mode is selectable per run |
| A-46 | Self-collision | §9.1 | Non-adjacent link pairs collide when they touch; adjacent links do not collide at their shared joint beyond the declared overlap |
| A-47 | Discovery candidate profile | §7.7 | With no hardware and no network, the method run on the vendored G1 produces a candidate profile that passes §7.1 validation and a report listing limits, inertia, sampled workspace and enumerated tests, each with its source; a second run on the same inputs reproduces both exactly |
| A-48 | Probe refusal | §7.7 item 3 | A commissioning probe commanded beyond a declared limit aborts and quarantines the candidate profile; no probe runs without a preceding simulated dry run |
| A-49 | Web claims are proposals | §7.7 items 6 and 7 | A web-sourced value cannot close an acceptance row or change a hazard control; a conflicting value appears with both sources and blocks activation |
| A-50 | Agent cannot activate | §7.7 item 8, §2.3.1 | The discovery method has no write path to an activated profile or control configuration; a candidate profile is rejected by the activation check until a human decision record exists |

Numeric physical thresholds depend on robot mass, reach, actuator specifications, sensing and task constraints. Set them in a versioned acceptance profile before evaluating a candidate; do not tune thresholds after observing the result. Use multiple seeds and confidence intervals for stochastic outcomes, and include negative results.

An acceptance profile is itself an artifact and MUST record: the robot profile and firmware version it applies to, the control period and timing budget, the physical operating envelope, per-metric thresholds with their justification, the seed list and run count, the measurement duration, and the date and author. Profiles are versioned and immutable once a run has been evaluated against them. Changing a threshold after observing a result produces a new profile version and invalidates the comparison, which MUST be stated rather than quietly re-baselined.

## 15. Explicit non-goals for the first release

The first release does not promise universal zero-shot robot control, production autonomous boxing, full video-to-motion reconstruction, an unverified Rust GPU simulator, arbitrary online policy learning on hardware, a language model in the servo loop, an agent-orchestrated strategy, personality or operations layer, a robot-hosted agent runtime, an agent-selected action or an agent-owned safety or promotion gate, durable agent-run training campaigns, or reproduction of every named paper. It delivers the contracts and measured foundations needed to add those capabilities deliberately. Development-time automation, offline sampling assistance and the dependency knowledge base of Section 2.3.1 are tooling rather than release capabilities: no release claim rests on them, and none of them is a promise of this release either.

## Appendix A — Terminology, units and conventions

This appendix is definitional. It creates no requirements, but a term used in Sections 1–15 means what it says here.

### A.1 Abbreviations

| Term | Expansion | Note in this document |
| --- | --- | --- |
| A2A | Agent-to-agent protocol | Inter-agent transport named in B.2.13; orchestration only, never a control transport |
| ABA | Articulated-body algorithm | Forward-dynamics method |
| ADK | Agent Development Kit | Ambiguous: at least three unrelated projects use the name. Never write "ADK" unqualified; name the project and crate per B.2.13 |
| AMASS | Archive of Motion Capture as Surface Shapes | Corpus aggregating source datasets with independent licenses |
| AMP | Adversarial motion prior | A discriminator-based style objective per §9.4; a style term, never a feasibility or safety constraint |
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
| MARL | Multi-agent reinforcement learning | The class containing the self-play of §9.4 |
| MCAP | Message capture container format | Distinct from Rerun's RRD |
| MPC | Model-predictive control | Baseline or correction, not an adaptation algorithm |
| MPPI / CEM | Model-predictive path integral / cross-entropy method | Sampling-based control |
| NFSP / LS-NFSP | Neural fictitious self-play / its latent-space variant | Opponent-averaging self-play; LS-NFSP is the RoboStriker variant described in §9.4.1 |
| OOD | Out of distribution | |
| PD | Proportional-derivative control | Baseline controller |
| PHC | Perpetual Humanoid Control | See the correction in §3 |
| PPO | Proximal policy optimization | Authored baseline learner |
| QP / SQP | Quadratic program / sequential quadratic programming | |
| RMA | Rapid motor adaptation | The two-phase latent identification row of §11.3 |
| RNEA | Recursive Newton-Euler algorithm | Inverse-dynamics method |
| RRD | Rerun recording format | Viewer artifact, never the canonical record |
| S-DTW | Subsequence dynamic time warping | Candidate alignment metric for sub-trajectory retrieval under §9.7 |
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
| Tier | One level of the hierarchical decomposition of §5.1, defined by its rate, deadline, budget and fallback. Lower tiers have tighter deadlines; the assignment is recorded, not implied by a diagram |
| Latent action space | A learned, bounded action space produced by the distillation stage of §9.4, identified by version and inseparable from the decoder artifact that interprets it |
| Corpus retrieval (R0), between-episode retrieval (R1), in-loop selection (R2) | The three retrieval tiers of §10.6. Only R2 has a deadline, and evidence does not transfer between them |
| Prior corpus | A dataset of prior trajectories available to the retrieval-augmented learning of §9.7, distinct from the target data and from any held-out set |
| Transition contract | The explicit statement of which states permit a switch and how recurrent state is reset or transformed |
| Initiation set | The set of states in which a skill may legally begin |
| Operating envelope | The physical limits a robot is permitted to operate within for a given mode, enforced by the supervisor |
| Acceptance profile | The versioned threshold set a candidate is evaluated against, per §14 |
| Safety profile | The versioned per-robot limits, fault responses and hazard controls of §12 |
| Evidence state | One of the five states of §3, with the promotion rules given there |
| Shadow mode | Computing commands from live sensors with the transmit path disabled and the disable verified, per §4 |
| Locked environment | The toolchain, native inputs and development tools resolved by the committed `devenv.lock`, per §2.2. "Locally" is not an environment |
| Default environment | The devenv environment entered by `devenv shell` with no profile selection, from which Section 2's excluded runtimes are absent and the §2.1 exclusion gate runs |
| External model service | A dependency class under §2.3: Rust-authored, linking no foreign binary, whose advertised capability is produced by a hosted inference endpoint reached over the network. Rust-native in language, foreign in behavior |
| Agent layer | The optional, out-of-process, feature-gated orchestration described in §6 and §12.6. It proposes; it never commands |
| Advisory record | A validated, timestamped, expiring proposal emitted by the agent layer. Dropped when stale; never a motor command, never a safety control |
| Offline degradation | Required behavior of a capability when its external model service is disabled or unreachable: continue in reduced form or fail safe, per row A-24, never stall a decision |
| Skill descriptor | The human-readable Markdown catalog entry of §10.1. Data, not code; a routing signal, never a feasibility check |
| Development-time automation | Use U1 of §2.3.1: tooling on developer and CI hosts only, whose output passes the same gates as authored work and whose removal changes no deliverable |
| Assisted sampling | Use U2 of §2.3.1 and §9.5: offline proposal of design candidates. It proposes a candidate set; measurement selects from it |
| Dependency knowledge base | Use U3 of §2.3.1 and §13.1: the project-owned, cited, version-invalidated record of low-level dependencies used when diagnosing faults. Not evidence under §3 and not a hazard control |

### A.3 Units, frames and timing conventions

SI units throughout: metres, kilograms, seconds, radians, newtons, newton-metres, amperes, kelvin or degrees Celsius where stated explicitly. Angles are radians in stored records regardless of what a source format uses; a converted field records the source convention in its provenance.

The canonical coordinate frame, gravity direction, root convention and quaternion component ordering are fixed by decision record and recorded here once ratified; see [C-04](#appendix-c--open-questions-and-unresolved-decisions). Until then, no code may assume a project-wide convention, and every transform MUST name the frames it maps between. Quaternions are serialized with named components, never as a bare four-element array, precisely because the ordering question above is still open.

Time is monotonic within a process and stored as explicit nanosecond integers with a named clock. Wall-clock mapping is recorded separately where correlation with external systems is required. A timestamp field MUST declare whether it is a sampling instant or an arrival time, per Section 5.

## Appendix B — Rust Ecosystem Survey and Dependency Qualification (Revised v3.0)

### B.1 Provenance, maturity and corrections

This appendix supersedes the previous survey integration with the user-supplied revised v3.0, which retains the v2.0 content and adds agent orchestration, session and memory stores, markdown skill indexing, inter-agent transport and guardrail tooling. Its reported versions, maintenance status, paper references, performance figures and inspection claims have not been independently checked for this document. All incoming entries start at `unverified` under Section 3 until a primary-source record establishes a higher evidence state. In particular, the supplied `paper_verified` label for HULKs and `code_inspected` label for Burn are not inherited automatically, and the v3.0 agent-layer entries of B.2.13 are `unverified` without exception. The register retains local subsection numbering for existing references; the v3.0 agent inventory is B.2.13.

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
13. “ADK” identifies at least three unrelated projects: Google's Agent Development Kit, this survey's `adk-rust` by zavora-ai, and `rust-adk` by Yougigun. A citation naming only “ADK” supports nothing. Every reference MUST name the owning project and the exact crate, and registry ownership MUST be confirmed before any dependency line is written, because a plausible crate name is a supply-chain surface and not an identity.
14. A pure-Rust agent framework that calls a hosted model is Rust-native in language only. The survey's own “zero Python” framing describes the client, not the capability: the behavior lives on a third party's server, is not pinned by `Cargo.lock`, and may change without notice. This is the external-model-service class of Section 2.3, not the Rust-native class.
15. Self-reported “experimental” status is a disqualification for production duty, not a caveat. Where the survey assigns an experimental crate a role such as hardware-safety guardrails, that assignment is rejected outright; see also correction 17. Experimental components may be used for offline tooling only.
16. Reported interface counts — numbers of agent types, tools, memory backends or protocol features — are inventory, not capability. None of them evidences latency, failure behavior under an unreachable endpoint, or correctness of a single decision.
17. A guardrail, policy filter or audit log in an agent framework is not a safety control under Section 12. It operates on text or structured proposals, carries no authority over actuators, and cannot be credited in a hazard register. The supervisor of Section 12.1 remains the control, and the agent layer's checks are at best an earlier, additional filter per Section 12.6.
18. “Sub-millisecond” or “microsecond” figures quoted for agent-layer components are not control-path qualifications and MUST NOT be compared with the budgets of Section 5. Any component that may reach the network has unbounded worst-case latency regardless of its median, and an agent layer is excluded from the deadline-critical loop by Section 2.3 rather than admitted on a benchmark.
19. Markdown descriptors indexed by keyword search are a catalog and a routing signal, never a feasibility check. Prose asserting that a skill applies is not a statement about joint velocity, support polygon or reach, and MUST NOT relax the compatibility filter of Section 10. Lexical search quality also degrades with vocabulary drift, so descriptor conventions must be fixed before the index is relied upon.
20. A graph memory with temporal versioning records what was believed and when; it does not make the belief true. Opponent-model entries are evidence-bearing records under Section 3 with their own provenance, and a store that is unavailable or stale MUST degrade to opponent-agnostic behavior per Section 12.5. Its read path MUST NOT require a network service or a hosted backend.

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

CubeCL (the GPU-kernel-compute layer Burn is built on) and `wgpu` (the graphics/compute API CubeCL's own GPU backends sit on, also usable standalone via hand-written WGSL) are candidates for `robot-sim`'s GPU-batched physics kernels as well as for `robot-learn`'s neural-network compute — see `crates/robot-sim/SIMULATION_SPECS.md` Section 11.3 for the detailed, source-grounded assessment of both for that specific, non-neural-network use. Qualification for the two roles is not assumed to transfer: a backend qualified for `robot-learn`'s tensor operations is not thereby qualified for `robot-sim`'s kernel-shaped, iterative, tree-recursive workload, and each is gated separately under B.6.

#### B.2.2 Reinforcement learning

| Candidate | Reported capability / maturity | Decision |
| --- | --- | --- |
| `ember-rl` | Small Burn DQN/PPO/SAC implementation | Reference-only until identity, code and numerical behavior are audited |
| `thrust-rl` | Burn algorithms and separated inference; reported removal of tch | Verify current feature/dependency closure; inspect deployable-inference patterns |
| `sb3-burn` | Partial Stable-Baselines3-inspired implementation | Verify completeness per algorithm; no reliance on unfinished PPO/SAC |
| `r2l-burn` | PPO with reported Python Gymnasium environment path | Python-dependent execution is excluded; inspect algorithm code independently |
| `rl_traits` | Environment/agent interfaces with reported terminated/truncated distinction | Audit semantics before adopting traits or documenting divergence |

Retain the authored PPO baseline. Audit GAE, truncation bootstrapping, action log probabilities, entropy, clipping, advantage normalization, optimizer state and checkpoint behavior. Compare small deterministic fixtures and learning curves; matching seeds alone does not guarantee matching training across implementations.

No surveyed Rust crate provides the multi-agent, opponent-averaging or distillation machinery that Section 9.4 requires, and none provides the retrieval-augmented data selection of Section 9.7. Those are authored work over the extension points of Section 9.3 and the storage contract of Section 7.6: a supervised average-policy learner, a reservoir store, an opponent/population provider and a retrieval provider. Treat the reference crates above as sources of interface semantics for the single-agent case only; a trait set that has no notion of an opponent or of a population generation is not evidence that one is unnecessary.

#### B.2.3 Physics and robotics mathematics

| Candidate | Provisional maturity / role | Decision |
| --- | --- | --- |
| Rapier (`rapier3d`, `rapier3d-f64`) | Established Rust rigid-body engine, with CCD and multibody joints; humanoid workload unqualified | First simulator candidate; test existing articulation, motors, contacts, snapshots and determinism before authoring missing functionality |
| `ferromotion` family | Survey reports a young, broad pure-Rust robotics ecosystem; identity and implementation unverified here | Investigate module by module; not a mandatory dependency or architectural foundation |
| Rust bindings to foreign physics engines, including `rusty_mujoco` | Foreign-engine interfaces; survey maintenance/version claims need checking | Excluded by default; optional cross-validation only following a separate scope decision |

The ferromotion survey reports FK, analytic Jacobians, RNEA, ABA, mass matrices, dynamics derivatives, LM IK, Paden–Kahan subproblems, trajectory optimization, contact solving, retargeting and identification. It also reports `ferromotion-learn`, calibration modules and wgpu rollout examples. Verify repository ownership, published crate/module names, licenses, versions and working code before evaluating these capabilities. Reported arm or cartpole demonstrations do not qualify humanoid contact dynamics or GPU humanoid training.

Candidate implementations of IK may use nalgebra, qualified Jacobians and a Rust QP solver. Mink remains a Python algorithm reference. Select a solver only after testing feasibility detection, constraint accuracy, iteration limits and timing; LM IK and constrained differential-QP IK are not interchangeable.

Do not combine two physics solvers by casually assigning one dynamics and the other contact impulses. Any hybrid must specify state ownership, integration, impulse application and solver coupling, and pass energy/momentum and contact tests. Prefer evaluating complete candidate pipelines first.

`crates/robot-sim/SIMULATION_SPECS.md` specifies the GPU-batched, variable-fidelity target this row's qualification work is now measured against, including the batching-strategy choice (one GPU thread per environment, running the existing authored ABA/RNE recursion largely unchanged, rather than restructuring it into per-tree-level batched tensor operations) and why: see that document's Section 11.1. This does not relax the Rust-native requirement above it — the reference simulator itself is a design reference for that document's Sections 4-10, never a dependency.

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
| Authored sequence alignment for sub-trajectory retrieval | No qualified Rust implementation of subsequence dynamic time warping identified for this use | Authored candidate for the offline retrieval unit of §9.7; benchmark cost against corpus size and window length, and test the degenerate and tied-cost cases before trusting an alignment |
| Embedding sources for a retrieval index | Structured skill tags and numerical features first, per §10.1; a learned or foundation-model embedding is a separate, versioned artifact | A hosted embedding endpoint is an external model service under §2.3 and is excluded from the default build; a locally executed Rust model is qualified like any other §2 dependency |

Measure the exact/approximate crossover on target hardware. Include search recall, filtering, index build/update cost, resident memory, serialization and worst observed timing. A fast neighbor lookup that returns infeasible skills does not satisfy the routing contract.

The three retrieval tiers of Section 10.6 are qualified separately. A candidate measured for offline corpus selection is not thereby qualified for the in-loop router, and an index structure that is acceptable at R0 may be inadmissible at R2 because it allocates, memory-maps a file or rebuilds incrementally.

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

#### B.2.13 Agent orchestration and model-backed tooling

Added by the v3.0 survey. Every entry here is `unverified`, and the class distinction of Section 2.3 applies per crate: a crate that reaches a hosted endpoint is an external model service, while a session store, index or transport is an ordinary Rust-native candidate qualified on its own merits. Nothing in this subsection is scheduled, and nothing here may appear in a default build. Each candidate below is additionally read against the three admissible uses of Section 2.3.1 — development-time automation (U1), offline sampling assistance (U2), dependency knowledge base (U3) — and a candidate that serves none of them is reference-only regardless of its qualification state.

| Candidate | Survey-reported scope | Class and qualification decision |
| --- | --- | --- |
| Agent and runner core, reported as `adk-agent` and `adk-runner` (zavora-ai `adk-rust`) | Agent runtime: sequential, parallel, loop and conditional workflow agents; tool invocation; typed tool interfaces | External model service wherever a model call is involved. Confirm project and crate ownership per correction 13 before any dependency line; no control-path use under any circumstance |
| Model-provider clients in the same family | Clients for several hosted model providers | External model service. Default-off feature, offline-degradation test required, provider/model/version recorded per Section 13 |
| Session and durable event store, with SQLite or server-backed options | Session lifecycle, durable event log, resumable runs | Rust-native candidate; qualify the chosen backend, schema evolution, growth bounds and crash recovery. An operational log, not a reproducibility artifact until the runtime itself is qualified |
| Markdown skill indexing, reported as `adk-skill` | Build/load-time indexing of Markdown capability descriptors, lexical matching | Rust-native candidate; the descriptor tier of Section 10.1. Routing signal only, never a feasibility check, per correction 19 |
| Graph memory with temporal versioning | Entity/observation graph, bi-temporal history, several storage backends | Rust-native candidate for the opponent memory of Section 12.5, provided an embedded local backend is qualified. Network-dependent read path is excluded per correction 20 |
| Agent-to-agent transport | Inter-agent messaging and discovery | Rust-native candidate for orchestration only; it is not a robot control transport and is not compared against Section 5 budgets |
| Guardrail, policy-filter and audit components | Input/output filtering, policy checks, audit logging | Rust-native candidate for tooling. Explicitly **not** a safety control per correction 17 and Section 12.6; any survey assignment of hardware-safety duty is rejected |
| Telemetry component, reported as `adk-telemetry` | Agent-run instrumentation | Rust-native candidate, qualified under B.2.10 like any other subscriber; keep construction and export off any control thread |
| Components the survey labels experimental, including its managed-run and enterprise crates | Durable managed runs, enterprise features | Offline tooling only; no production duty per correction 15, including the durable-run log the survey cites for training campaigns |
| A local model-serving path, authored on the Burn shortlist or a qualified Rust inference crate | The substitution that would remove the external-model-service class entirely | Rust-native, and the only route by which a model-dependent capability could ever reach a default build. Unscoped; see B.5 item 10 |

Before any part of this family is adopted, three things must hold: the crate identity and ownership are confirmed from the registry; the non-model subset is separable, so that the useful stores and indexes can be taken without the model client; and a named local-inference substitution path exists for each model-dependent capability per Section 2.3 item 4. A reported ability to swap model providers is not a substitution path, because every option in the list is still a hosted endpoint.

The survey's eight integration proposals — including agent-run training campaigns, markdown skill catalogs, opponent-memory graphs, multi-agent commentary and operator advisories — are ideas to evaluate individually against Section 14, not a roadmap. Their adoption status is open under [C-15](#appendix-c--open-questions-and-unresolved-decisions).

#### B.2.14 Extensible Rust interfaces and replay storage

The mechanisms below are candidates for implementing Sections 7.1, 7.6, 9.3 and 9.6, not requirements to adopt a crate. Qualify any selected dependency under B.6 and keep registration, dynamic dispatch and deserialization outside the deadline-critical path.

| Mechanism | Candidate use | Qualification boundary |
| --- | --- | --- |
| Generics and monomorphization | Static dispatch for concrete learner, model and storage implementations | Preferred where bounded execution or control-path use matters; measure code size and compile-time costs when many combinations are instantiated |
| Closed enums with `enum_dispatch` | A finite, build-known set of implementations behind a trait-shaped API | Compare with a hand-authored enum; implementations are closed at build time and no dynamic plugins are implied |
| `inventory` or `linkme` | Build-linked name/capability catalog for offline method discovery | `inventory` registrations use constructor-based initialization; `linkme` collects const static entries into linker sections. Verify target/linker support, startup behavior and failure modes; neither is a runtime code loader |
| `typetag` with `inventory`, and `erased-serde` | Offline serialization of heterogeneous trait objects where a closed schema cannot use an enum | `typetag` relies on `inventory`/constructors and builds a tag-to-deserializer map on first trait-object deserialization; erased serialization does not provide schema evolution. Exclude from control-path configuration, policy bundles and replay records unless separately encoded by explicit versioned schemas |
| SoA or tiled AoSoA chunks | Batch-friendly transition and rollout layout | Treat layout as an authored data-model choice; benchmark fields, batch sizes, vectorization, memory use and indexed sampling before choosing SoA over AoS/AoSoA |
| Fixed-capacity ring buffers | Bounded in-memory replay or rollout storage | Decide overwrite/full behavior, generation IDs, wraparound, sampling access and snapshot semantics. A FIFO/SPSC queue is not automatically a random-access replay buffer |

Primary mechanism documentation: [Rust generics](https://doc.rust-lang.org/book/ch10-01-syntax.html), [`enum_dispatch`](https://docs.rs/enum_dispatch), [`inventory`](https://docs.rs/inventory), [`linkme`](https://docs.rs/linkme), [`typetag`](https://docs.rs/typetag), [`erased-serde`](https://docs.rs/erased-serde), and the [`ringbuffer` crate](https://docs.rs/ringbuffer). These links establish mechanism descriptions only; they do not establish suitability, maintenance, license, platform support or performance for this project.

### B.3 Concrete workspace anchors

| Crate family | Initial shortlist / authored work |
| --- | --- |
| `robot-types` | Typed units/frames/time and versioned observation/action/transition schemas; evaluate a units crate such as `uom` |
| `robot-model` | `urdf-rs` plus authored semantic validation, stable typed IDs, namespaced profile extensions and actuator metadata |
| `motion-io` | BVH candidates, `c3dio`; qualified or authored numeric human-body format support |
| `motion-retarget` | nalgebra, qualified Jacobians and constrained solver; authored task mapping |
| `robot-sim` | Rapier existing multibody/contact path first; compare verified alternative modules |
| `robot-learn` | Burn shortlist, authored PPO, capability manifests and additive extension traits; audited RL references |
| `robot-skills` | Exact Rust search; `instant-distance` only after crossover measurement; the R0/R1/R2 tiers of §10.6 kept as separate, separately qualified code paths |
| `robot-hardware` | Protocol-specific Rust adapters; qualify `roslibrust`/Zenoh if interoperability is needed |
| `robot-control` | Authored deterministic runtime; compare HULKs/Copper patterns and qualified IPC |
| `robot-adapt` | Authored identification baseline; investigate verified dynamics-gradient modules |
| `robot-viewer` | Rerun or egui; `urdf-viz` for model inspection |
| `robot-artifacts` | Rust columnar/binary tooling; canonical schemas separate from visualization formats; versioned episode and replay chunks |
| `robot-perception` estimation module | Qualify multicalc/numeris/typed_kalman/kfilter or authored estimator against the same fixtures |
| `motion-retarget`, `robot-control` optimization modules | Clarabel and other verified Rust solver candidates; qualified MPC/WBC modules where justified |
| `robot-eval`, parser test targets | Rust property/mutation/fault harnesses; optional foreign-engine fuzzing only after separate exception |
| Orchestration telemetry | tracing; optional off-thread OpenTelemetry and metrics exporter |
| Offline rollout/data workers | Rayon/custom pools and bounded Crossbeam primitives where needed |
| `robot-learn`, `robot-artifacts` | Authored sampler/storage traits, SoA or measured chunk layouts, bounded ring storage; qualify registry/dispatch crates only for offline composition |
| Optional agent layer, if adopted | Per-crate qualified non-model components from B.2.13, default-off and out of process; no model client in any default build |
| `robot-devtools` and the knowledge base of §13.1 | Authored Rust tooling over version-controlled records; local lexical and exact search first per B.2.5; optional assisted drafting or retrieval strictly under §2.3. Developer and CI hosts only |

These anchors are a shortlist, not permission to add every library. Prefer the smallest qualified dependency set.

The survey's proposed `robot-state`, `robot-ctrl` and `robot-test` names map to the existing estimation, control and evaluation boundaries above. Extract new crates only if ownership or dependency isolation warrants it; do not maintain duplicate responsibilities. The v3.0 survey's proposed `robot-brain` is treated the same way: its responsibilities attach to `robot-persona` and the tooling crates per Section 6, and whether it becomes a package is open under [C-16](#appendix-c--open-questions-and-unresolved-decisions).

### B.4 Simulator and adaptation qualification

1. Verify existing Rapier functionality and any alternative's actual code before designing missing subsystems.
2. Test analytic pendulum and actuator fixtures, articulated mass/inertia behavior, floating-base dynamics, joint limits and motor constraints.
3. Test standing, slip, contact impulses, pushes and impacts. Report penetration, energy behavior, tracking error and sensitivity to timestep/solver settings.
4. Measure CPU environment throughput before budgeting GPU implementation. A GPU network backend or cartpole rollout is insufficient evidence for GPU humanoid physics.
5. Test snapshot completeness: engine state alone may omit policy recurrence, RNG, sensors, pending commands and task state required for replay.
6. If foreign cross-validation is separately authorized, compare matched fixtures through a non-default profile and publish divergence. Do not make this unavailable optional engine a prerequisite for default-build progress.
7. For gradient-based identification, compare derivatives with finite differences on smooth fixtures, document behavior at contact changes and test parameter identifiability. Use held-out trajectories and calibrated torque estimates.
8. For the GPU-batched backend `crates/robot-sim/SIMULATION_SPECS.md` Section 11 requires: check its output against the CPU reference backend of that document's Section 11.4 on shared fixtures at matched configuration before item 4's throughput measurement is trusted as evidence of anything beyond raw speed. A fast, unverified-for-correctness GPU backend is not a qualified one.

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
10. **Local inference, if the agent layer is ever wanted:** the real gap behind every v3.0 proposal is a qualified Rust local-inference path, not a client for a hosted endpoint. Scope it against the Burn shortlist of B.2.1 before scheduling any model-dependent capability. Lowest priority; it blocks nothing in the baseline.

11. **Low-level dependency knowledge, as a maintained artifact:** the inventory required by B.6 item 1 exists per candidate but is not yet a maintained, cited, version-invalidated base usable during a hardware fault, per Section 13.1. The gap is curation and tooling discipline, not a missing library; a dependency a fault traces to is a dependency whose interface documents, firmware revisions and known defects should already be recorded. Medium priority once hardware work begins, and it does not depend on any agent capability.

The survey's claimed absences become verification tasks, not automatic instructions to rewrite large subsystems. A newly surveyed capability does not change this priority order: the baseline of items 1 to 4 is unaffected by the existence of an agent framework.

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
9. For a candidate in the external-model-service class of Section 2.3, additionally: confirm registry ownership of the exact crate; demonstrate that the feature is absent from the default dependency closure; record provider, model identifier, version and request parameters; test the unreachable-endpoint, timeout, truncated-response and malformed-response paths; and show the offline-degradation behavior required by row A-24. A gate passed against one hosted model does not carry to another model or to a changed snapshot.

10. For a candidate used only as development-time automation under use U1 of Section 2.3.1, gates 3 to 5 apply to the artifacts it produces rather than to a runtime path, and two additional conditions hold: it MUST NOT appear as a dependency of any crate that ships, and the work it automates MUST remain performable without it at a recorded cost. A tool whose absence blocks the build is not tooling, it is a dependency, and it is gated as one.

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
| Hosted model changes or is withdrawn | Behavior drift with no code or lockfile change; endpoint deprecation | Keep every model-dependent capability default-off and offline-degradable; demote dependent claims on model change per §3; prefer a local-inference path before scheduling the capability |
| Agent layer acquires implicit authority | An advisory becomes load-bearing; a guardrail is cited in a hazard register | Supervisor independence tests, §12.6, and the A-24 removal test as a standing check |
| Dependency confusion on an ambiguously named agent crate | A plausible crate name resolves to an unrelated or hostile package | Confirm registry ownership before the dependency line, pin exact versions, and audit the closure per B.6 item 1 |
| Single-maintainer agent framework stalls or churns | Concentrated commit history and frequent breaking releases reported by the survey; wide API surface | Keep every use optional and behind our own traits; pin exact versions; prefer the thin components whose authored replacement is bounded and whose cost is recorded under B.6 item 6 |
| Generated development artifact enters the repository unreviewed | A commit no author can explain; a test passing against generated expectations | U1 output passes the same review and CI gate as authored work, is attributable per Section 2.3.1 item 5, and never generates both an implementation and the test that accepts it |
| Assisted sampling narrows its own evaluation | Improvements that do not reproduce on held-out seeds, disturbances or opponents | Held-out sets defined independently of the proposal source, per Section 9.5 item 3; ablation and prespecified claim per Section 9.3 |
| A retrieval corpus overlaps the evaluation set | Few-shot improvements that vanish on a genuinely unseen session; suspiciously exact retrieved matches | Construct the corpus by exclusion and test the exclusion, per Section 9.7 item 3; record provenance on every retrieved transition |
| A hierarchy is claimed but every layer runs at one rate | Identical configured rates across tiers; decision age reported as zero | Section 5.1 item 2 requires the degenerate case to be recorded as such, and item 1 requires a declared tier per component |
| An upper tier becomes load-bearing for safety | A fall or envelope violation when the strategy layer is late or disabled | Per-tier fallback testing under row A-32 and the highest-tier removal test of Section 5.1 item 7 |
| A training constraint is cited as a safety control | A latent prior, style discriminator or manifold bound appearing in a hazard register | Section 9.4.2 items 4 and 5 forbid it; Section 12.2 already requires controls independent of the learned component |
| Knowledge-base entry drifts from the dependency it describes | A fault diagnosed against a stale firmware or driver note; confident wrong answers during an incident | Version-keyed invalidation per Section 13.1 item 4, mandatory citation, and the prohibition on citing the base as evidence or as a hazard control |

## Appendix C — Open questions and unresolved decisions

This specification deliberately leaves decisions open rather than inventing defaults that later read as requirements. Each entry below blocks something concrete. An entry is closed by a decision record, after which the decision is cited from the relevant section and the row moves to the revision history — it is not deleted silently.

| ID | Question | Blocks | Resolution mechanism | Interim position |
| --- | --- | --- | --- | --- |
| C-01 | What happens to the existing Python `training/` and `unitree/` code? | §1.1; any claim that the repository satisfies §2 | Migration decision record | **Resolved 2026-10-26.** Removed from the working tree; retained in version control as cited prior art and protocol reference, and excluded from build, CI and all deployable paths. Reintroduction requires a §2.1 exception |
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
| C-15 | Does this project adopt any agent-layer capability at all, and if so which of the v3.0 proposals, under what offline-degradation and local-inference commitment? | §2.3, §6, §12.6, B.2.13, rows A-23 and A-24, and the change-record obligation for this §2/§5/§12/§14 revision | Decision record after the non-model subset is separately qualified under B.6 and a local-inference path is costed per B.5 item 10 | None adopted and none scheduled; every component stays default-off, out of process and removable, and the baseline of Phases 0–4 proceeds without it |
| C-16 | Is a separate `robot-brain` package justified, or do agent responsibilities stay inside `robot-persona` and the tooling crates? | §6 workspace layout, B.3 anchors | Decision record, only after C-15 resolves to adopt something | No package is created; the survey's crate name is not an architectural requirement |
| C-17 | What are the retention, consent and deletion terms for persistent opponent-memory records, and which embedded local backend holds them? | §12.5, §7.5, B.2.13 graph-memory row | Decision record alongside the data-handling terms of C-13 | No persistent opponent record is retained beyond a session; no hosted backend is permitted on any read path |
| C-18 | Which of the three admissible uses of §2.3.1 — U1 development-time automation, U2 offline sampling assistance, U3 dependency knowledge base — is actually enabled, with which crates and under whose review? | §2.3.1, §9.5, §13.1, rows A-25 to A-27, and the change-record obligation for this §2/§12/§13/§14 revision | Three separate decision records, one per use; a single combined "adopt the framework" record is not accepted | None enabled. U1 and U3 are the cheapest and least coupled and are the expected first candidates; U2 is admissible but idle until there is a baseline to improve on |
| C-19 | What is the required scope, owner and update cadence of the dependency knowledge base, and does it live in this repository or alongside the evidence ledger? | §13.1, row A-27, the fault-record loop with §5 incident recording | Decision record once the first hardware target of C-02 is known, because the inventory that matters is that robot's | Version-controlled records in this repository, keyed to the §2 dependency inventory, updated whenever a dependency version or a diagnosed fault changes; no separate service |
| C-20 | Which static-dispatch, closed-enum or build-linked registry strategy is selected for RL extensions, and which target platforms support it without startup or control-path hazards? | §9.3, §9.6, Appendix D, rows A-28 and A-30, and the change-record obligation for this §14 revision | Decision record after a minimal offline prototype and platform/linker qualification under B.6 | Generic/static dispatch is the default for bounded paths; registry candidates remain offline-only and unselected; no dynamic code loading is permitted |
| C-21 | Which in-memory and persistent backends implement the shared transition/episode schema, bounded chunks, replay ring and sampler interfaces? | §7.6, §9.3, B.2.14, rows A-28 and A-29, and the change-record obligation for this §7/§14 revision | Decision record after representative workload benchmarks, migration tests and determinism checks under B.6 | Canonical schema and interface boundaries are fixed; an authored preallocated ring and chunked storage are the initial design, but no backend crate or encoding is selected |
| C-22 | What are the tier assignment, rate and budget for each learned component of the deployed hierarchy, and does the strategy layer survive being slowed from the experiment rate to its intended tier? | §5.1, §9.4, §10.6, rows A-32 and A-33, and the change-record obligation for this §5/§14 revision | Decision record after per-tier measurement on the benchmark host of C-08, once the first embodiment of C-02 is known | Two tiers are assumed — a tactical layer at 10–50 Hz over a policy-rate decoder — but no rate is committed, and a single-rate experiment is recorded as such rather than reported as a hierarchy |
| C-23 | Which prior corpora are admissible for retrieval-augmented learning, what is the retrieval unit and similarity source, and is an embedding model involved at all? | §9.7, §7.5, §10.6, row A-34, and the license audit of C-07 | Decision record per corpus, after the license audit and a measured comparison against target-only and whole-corpus training | No corpus is assumed usable; structured features and exact search are the default, and no embedding service is adopted |
| C-24 | Which limited-data adaptation methods beyond identification and history conditioning are actually scheduled, and under which operating mode may any of them update parameters on hardware? | §11.3, §4, row A-17 | Decision record per method, after the preceding rows of the Section 11.3 table have been measured and found insufficient on the target robot | Identification first, then history conditioning; the latent-identification, fine-tuning, retrieval-conditioned and cross-embodiment rows are unscheduled, and no on-hardware weight update is authorized |
| C-25 | Does the one-GPU-thread-per-environment batching strategy actually outperform the per-tree-level vectorized alternative on our own fixtures; is CubeCL's control-flow performance acceptable for an iterative constraint solver in practice; what are the named fidelity tiers and their per-knob settings; is a `Ball` joint required for the first floating-base milestone or deferrable? | §9.1, `crates/robot-sim/SIMULATION_SPECS.md` Sections 11-12, Appendix B of that document | Working CPU-reference-backend prototype and a first GPU-batched prototype, measured against each other per `SIMULATION_SPECS.md` §14 | One-thread-per-environment is the current default (no Rust-native precedent exists for either strategy applied to a tree-recursive rigid-body solver, per that document's Appendix A); `Free` joint required, `Ball` deferrable; no fidelity tier names are fixed |
| C-26 | Which trait boundaries, and which static or dynamic form, do the policy (`Policy`), simulator backend (`Backend`) and robot-model importer take in the Rust workspace? | §6, §9.3, §9.6, rows A-30 and A-38, and the change-record obligation for this §14 revision | Decision record after the behavior-preserving change is measured against the existing test suite | Generic parameters on the control path and trait objects offline only, per Appendix D.1; the three traits are proposed, not yet implemented |
| C-27 | What is the fidelity report schema, and which tier settings (timestep, integrator, solver) does the shared battery cover? | §9.1, row A-36, `SIMULATION_SPECS.md` §12.2 and §14, and the change-record obligation for this §14 revision | Decision record after the first two tiers have run the shared battery on the benchmark host of C-08 | No tier names are fixed; the report is proposed as JSON or TOML written through `robot-artifacts` |
| C-28 | Which test-purpose render format comes first: a frame dump (CSV or JSON Lines), std-only SVG, or a qualified viewer format such as Rerun `.rrd`? | §14 row A-39, Appendix B.6, §15 | Decision record; a viewer format also requires a B.6 qualification and an ADR | Frame dump first, SVG golden files second; no viewer dependency is selected |
| C-29 | Is attached-agent discovery of Section 7.7 an admissible use beyond U1 to U3, and which of its capabilities (commissioning probe, web research, model-assisted extraction) are enabled? | §2.3.1, §7.7, §15 | Decision record and a separate feature gate per capability; the commissioning probe also needs a Section 12 safety case before it is enabled | Offline, simulated and web parts are not scheduled; no hardware probe is enabled |

A blocked requirement is not a deferred requirement. Where an open question prevents a section from being testable, the corresponding Section 14 row cannot be closed, and a release report MUST say so rather than omitting the row.

## Appendix D — Rust extensibility patterns

This appendix describes implementation choices that may realize Sections 7 and 9. It is advisory: Section 9.6 and the decision records in Appendix C govern, and no pattern below authorizes dynamic code loading or changes the timing rules in Section 5.

### D.1 Dispatch and composition

Rust generics are monomorphized for concrete types and provide compile-time method dispatch. This is the preferred default for a control-path implementation and shared interfaces whose concrete type is known when the executable is built; it trades runtime indirection for compile-time and binary-size growth. Benchmark both effects rather than assuming the resulting binary is smaller or faster.

A closed enum is appropriate when the implementations are known and finite at build time. `enum_dispatch` is a candidate macro to forward a trait-like API across enum variants; compare it with a manually authored enum and keep the set auditable. It is not an open plugin registry.

Trait objects can simplify heterogeneous experiment composition, but their use is limited to offline training, evaluation and tooling. Name-keyed registries may resolve a manifest to an implementation once at process startup from code already linked into that executable. They MUST NOT register or unregister methods during a run and MUST NOT be queried on a control tick. `inventory` distributes registrations from linked crates through constructor-based initialization; `linkme` gathers const static entries into linker sections and therefore depends on supported linker/platform behavior. Qualify initialization, dead-stripping, target support and deterministic duplicate-name rejection before selecting either mechanism.

`typetag` combines trait-object serialization with distributed registration, relying on `inventory` and constructor initialization; it constructs a tag-to-deserializer map on first trait-object deserialization. `erased-serde` supplies object-safe Serde interfaces. These are possible offline interchange mechanisms, not canonical schema or migration systems. Prefer a versioned explicit enum for a stable set of serialized variants, and never deserialize tagged trait objects on a control path or treat one as executable plugin loading.

### D.2 Robot and experience data layout

The robot descriptor and transition schema are authored Rust types independent of the chosen serializer. Stable typed IDs, explicit schema versions, namespaced extensions and migration rules preserve meaning as embodiments and methods grow. Registries describe capabilities; they do not replace validation of typed fields or grant authority to unknown metadata.

SoA chunks place each field in a contiguous column and can make batched sampling and vectorized field access efficient; AoS or tiled AoSoA may be better for other access patterns. Benchmark representative observation/action dimensions, batch sizes, cache behavior, gather cost and memory use before freezing a layout. Keep serialization chunks bounded and versioned as required by Sections 7.5 and 7.6.

A fixed-capacity ring gives bounded memory and predictable overwrite behavior, but the buffer contract must additionally define capacity exhaustion, generation IDs, wraparound and access by sampled transition index. A simple FIFO/SPSC ring is not a complete replay store: prioritized or sequence sampling needs separate index structures, and sampler state/RNG must be checkpointed if exact continuation is required. Keep sampler traits separate from storage so uniform, prioritized, sequence and goal-relabeling methods can evolve without changing the transition format.

### D.3 Qualification references

The mechanism descriptions in Appendix B.2.14 link to the Rust Book and crate documentation for `enum_dispatch`, `inventory`, `linkme`, `typetag`, `erased-serde` and `ringbuffer`. Those references are research starting points, not endorsements. Before adoption, record the exact version, license, platform and linker behavior, feature/dependency closure, startup behavior, benchmark results, schema compatibility and removal cost under Appendix B.6.

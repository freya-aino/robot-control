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

Numeric physical thresholds depend on robot mass, reach, actuator specifications, sensing and task constraints. Set them in a versioned acceptance profile before evaluating a candidate; do not tune thresholds after observing the result. Use multiple seeds and confidence intervals for stochastic outcomes, and include negative results.

An acceptance profile is itself an artifact and MUST record: the robot profile and firmware version it applies to, the control period and timing budget, the physical operating envelope, per-metric thresholds with their justification, the seed list and run count, the measurement duration, and the date and author. Profiles are versioned and immutable once a run has been evaluated against them. Changing a threshold after observing a result produces a new profile version and invalidates the comparison, which MUST be stated rather than quietly re-baselined.


## Implementation checklist

Status legend: `[x]` implemented, `[~]` partial, `[ ]` not implemented, `[?]` not audited in this pass. Closure means an artifact in `docs/ledger/evidence-ledger.md` or `docs/acceptance/`, which is currently empty. Summary: 1 `[x]`, 17 `[~]`, 19 `[ ]`, 2 `[?]` across 39 items.

- [~] A-01 Rust constraint — Evidence: `docs/ledger/evidence-ledger.md` "No MuJoCo runtime" row; `devenv.nix` gate. Note: the exclusion gate always passes (see 2.1.4), so the gate requirement is not met.
- [~] A-02 Dependency promotion — Evidence: `docs/decisions/0003-std-only-default-closure.md`; ledger shows no Appendix B candidate promoted. Note: no candidate has been qualified, so the default closure is the authored baseline.
- [x] A-03 Evidence discipline — Evidence: `crates/robot-types/src/evidence.rs` and `docs/ledger/evidence-ledger.md` (states, admitting artifacts, dates). Note: demotion log is empty, as expected.
- [~] A-04 Parser robustness — Evidence: `crates/robot-artifacts/README.md` lines 49-55 (adversarial TOML sweep); `crates/motion-io/src/bvh.rs` tests. Note: no fuzz run for URDF or motion importers; rotation order is covered.
- [ ] A-05 Optimization — Evidence: `crates/motion-retarget/README.md` says no QP solver is qualified. Note: a qualified solver and its residual, resource and timing profiles are required.
- [~] A-06 Estimation — Evidence: `crates/robot-perception/README.md` lines 54-61 (analytic and finite-difference tests). Note: no covariance-consistency or sensor-fault stress measurement exists.
- [ ] A-07 Telemetry — Evidence: none found; no enabled/disabled comparison or exporter-isolation test. Note: add the instrumentation comparison and drop counters.
- [ ] A-08 Encoding and parallelism — Evidence: `crates/robot-artifacts/src/chunk.rs` tests exist; no representative-message benchmark or sequential/parallel rollout comparison found. Note: add the benchmark.
- [?] A-09 Cross-engine comparison — not authorized in the current evidence; no cross-engine run exists in the ledger.
- [~] A-10 Data correctness — Evidence: `crates/motion-io/src/bvh.rs` and its tests. Note: units, frame transforms, timestamps and contact metadata were not checked against fixtures.
- [ ] A-11 Retargeting — Evidence: no per-skill report exists in `crates/motion-validate`. Note: produce per-skill reports against thresholds.
- [ ] A-12 Simulator — Evidence: ledger has no throughput entry; `crates/robot-sim/README.md` records none. Note: add a recorded qualification benchmark.
- [ ] A-13 Learning — Evidence: `crates/robot-learn` has PPO unit tests only; no multi-seed reference-task result is in the ledger. Note: run PPO on small reference tasks over several seeds.
- [ ] A-14 Tracker — Evidence: no tracker exists (see 9.4.1). Note: implement and compare with PD baselines.
- [~] A-15 Runtime — Evidence: `crates/robot-control/src/timing.rs` reports the four quantities separately. Note: no measured end-to-end hardware results, and the GPU latency numbers are not a control-path measurement.
- [ ] A-16 Retrieval — Evidence: no experience retrieval exists (see 9.7.1). Note: implement and compare with a dense baseline.
- [ ] A-17 Adaptation — Evidence: only system identification exists (`crates/robot-adapt`). Note: real held-out session evaluation is not present.
- [~] A-18 Operating modes — Evidence: `crates/robot-types/src/mode.rs`. Note: transitions do not run entry preconditions (see 4.5); teleoperation is not verified as envelope-bound.
- [~] A-19 Safety case — Evidence: `crates/robot-supervisor` (21 tests, fault policy refuses clamp for uncontrolled faults, `robot-supervisor/README.md`). Note: no hazard register per robot, no emergency-stop test with host down, and no area-interlock exercise.
- [ ] A-20 Hardware — Evidence: no hardware validation in the ledger; the ledger states no row is `validated_on_hardware`. Note: none done.
- [ ] A-21 Boxing — Evidence: `crates/robot-tasks` has balance only; no scoring, opponents or disturbance scenarios. Note: implement the boxing task and its independent scoring.
- [~] A-22 Environment reproducibility — Evidence: `docs/ledger/evidence-ledger.md` records `devenv test` passing under the committed `devenv.lock` (host-local). Note: no clean-checkout run on a second host, and the exclusion gate is not part of `devenv test` (see 2.1.4).
- [?] A-23 Agent-layer qualification — no agent component is present in the build (see 2.3.1), so the row is not yet applicable.
- [~] A-24 Offline degradation — Evidence: no external model-service dependency is in the closure (see 2.3.1). Note: no test run with all external features disabled was recorded.
- [~] A-25 Development-time automation hygiene — Evidence: `docs/ledger/automation-log.md` exists. Note: the automation gate is not enforced in CI (see 2.2.4).
- [ ] A-26 Assisted sampling discipline — Evidence: no assisted sampling (see 9.5). Note: implement the manifest, ablation and origin recording.
- [ ] A-27 Dependency knowledge base — Evidence: `docs/knowledge-base/dependencies/` and `docs/knowledge-base/faults/` are empty. Note: create cited entries for each low-level dependency.
- [~] A-28 RL extension conformance — Evidence: per-module unit tests in `robot-learn` and `robot-types`. Note: no shared conformance suite (see 9.6.5).
- [~] A-29 Experience and replay storage — Evidence: `crates/robot-learn/src/rollout.rs` fixed-capacity buffer; `crates/robot-skills/src/tiers.rs` stale-generation rejection test. Note: no sampler probability or importance-weight tests, and no on-policy/off-policy shared schema tests.
- [ ] A-30 Extension and removal isolation — Evidence: no optional extension exists to remove (see 9.6.6). Note: add the removal test once an extension exists.
- [~] A-31 Robot profile extensibility — Evidence: `crates/robot-model/src/profile.rs` (`schema_version`, `validate`, `activate` with `Strictness::Activating`). Note: no version-to-version migration tests were found.
- [ ] A-32 Hierarchical latency tiers — Evidence: no tier-rate declarations or late-upper-tier tests found. Note: implement tier declarations and the late-output tests.
- [ ] A-33 Staged latent pipeline — Evidence: no latent pipeline exists (see 9.4.2). Note: none until the tracker exists.
- [ ] A-34 Retrieval-augmented learning — Evidence: no experience retrieval exists (see 9.7.1). Note: implement with the corpus and admission rules.
- [ ] A-35 Retrieval tier discipline — Evidence: `crates/robot-skills` tiers exist, but the R2 path is not called from `ControlLoop::tick` (see 10.6.3). Note: wire the tiers and measure each.
- [~] A-36 Simulation fidelity tiers — Evidence: `crates/robot-sim/src/fidelity.rs` battery with v6/v7 reports, tolerances at 100 us and 1 ms, and `docs/ledger/evidence-ledger.md` rows. Note: the report is not written through `robot-artifacts`, and the energy-drift check is only one of the shared battery checks.
- [~] A-37 Checkpoint and partial fine-tuning — Evidence: `crates/robot-learn/src/checkpoint.rs` (weights, optimizer, RNG, normalizer); frozen-group test in `ppo.rs`. Note: a bit-identical frozen-weight check across a fine-tuning run was not found.
- [~] A-38 Plug-in seam traits — Evidence: `Policy` (`crates/robot-nn/src/policy.rs`), `Backend` (`crates/robot-sim/src/backend.rs`), `ModelImporter` (`crates/robot-model/src/importer.rs`). Note: no test adds a second policy type without touching the learner.
- [ ] A-39 Test-purpose frame output — Evidence: no versioned per-step frame dump found in `crates/robot-cli`. Note: implement the frame writer with a read-back test.

Numeric physical thresholds and acceptance profiles: no acceptance profile artifact exists in `docs/acceptance/` (empty), so the immutability and versioning requirements of Section 14 are not yet implemented.

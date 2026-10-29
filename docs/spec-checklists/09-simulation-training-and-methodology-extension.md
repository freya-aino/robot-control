## 9. Simulation, training and methodology extension

### 9.1 Simulator qualification

A useful humanoid simulator needs articulated dynamics, joint actuators, stable foot contact, collision filtering, configurable integration, sensor delay/noise, perturbations, resets, reproducible seeds and sufficient rollout throughput. A generic rigid-body engine is only a starting point.

Qualify the Rust backend on pendulum, actuator step response, static standing, foot slip, push recovery and controlled impact benchmarks. Separate physics timestep, actuator update rate and policy timestep. Record numerical sensitivity and the operating region where the model is credible.

Start with CPU vectorized environments. GPU batching is a separate engineering deliverable and cannot be assumed from GPU neural-network support. Estimate training duration from measured environment steps per second and required experiment sizes before expanding scale.

Qualify Rapier's existing multibody-joint and collision capabilities before proposing a replacement articulation layer. Compare credible alternatives, including ferromotion modules if their reported implementation can be verified, using Appendix B.4. Foreign-engine cross-validation remains optional and requires the separate scope decision in Section 2; this specification does not grant that exception.

The target was revised, by explicit project direction, to require GPU-bound batched execution, variable fidelity as a first-class interface, and fidelity on par with MuJoCo's own engine — specified in detail in `crates/robot-sim/SIMULATION_SPECS.md`, not here, so that the detailed, fast-moving design (batching strategy, solver algorithm, GPU-compute framework choice) does not require a `SPECS.md` revision each time it changes. That document is grounded in direct research against MuJoCo's own source/documentation and the Rust-ecosystem GPU-compute candidates of Appendix B.2.1 (Burn/CubeCL, Candle, and `wgpu` directly), not a restatement of Rapier qualification. It does not grant the MuJoCo-binding exception this section's closing sentence withholds: the Rust-native requirement stands, and MuJoCo's own implementation is a design reference, never a dependency. `docs/decisions/0005` records this as a decision, and C-25 tracks what the GPU-batched simulation specification still leaves open.

The floating-base multibody path has sphere-only ground contact (penalty normal force with regularised Coulomb friction, ADR 0010). The 28 box, cylinder and mesh collision shapes of the vendored G1 are not in contact, and there is no joint-limit or actuator model. The `robot sim multibody --floating` result is a supported-stance qualification: a clamped PD hold toward the zero posture, judged over 2 s on pelvis drop, contact penetration and settled kinetic energy, and valid only for the dt range recorded in ADR 0010. Zero torque collapses through the plane via unmodelled shapes and is recorded as a known limitation, not as physics. Static-standing, foot-slip and push-recovery results MUST NOT be reported as balance or standing capability for that path until contact proxies, a controller and those benchmarks are implemented and qualified. A free-fall run is not evidence of contact behavior.

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



## Implementation checklist

Status legend: `[x]` implemented, `[~]` partial, `[ ]` not implemented, `[?]` not audited in this pass. Summary: 2 `[x]`, 13 `[~]`, 19 `[ ]`, 5 `[?]` across 39 items.

- [~] 9.1.1 Simulator covers articulated dynamics, joint actuators, foot contact, collision filtering, integration choice, sensor delay/noise, perturbations, resets and seeds — Evidence: `crates/robot-sim/README.md` (integrators, passive forces, constraint solver, `Environment` trait). Note: sphere-only ground contact and no joint-limit or actuator model (see 9.1.9); sensor delay/noise and perturbations not confirmed.
- [~] 9.1.2 Qualify the Rust backend on pendulum, actuator step, static standing, foot slip, push recovery and controlled impact — Evidence: `crates/robot-sim/src/fidelity.rs` (fidelity check battery, `FidelityCheck` trait). Note: the individual checks were not enumerated in this pass; foot slip, push recovery and impact benchmarks were not found by search.
- [~] 9.1.3 Physics timestep, actuator rate and policy timestep are separate — Evidence: `crates/robot-control/src/control_loop.rs` tick and `robot-sim` integrators. Note: separate actuator-rate handling was not confirmed.
- [ ] 9.1.4 Record numerical sensitivity and the credible operating region — Evidence: ADR 0010 records the valid dt range for the floating-base path only. Note: add a sensitivity sweep and a region record for each backend.
- [~] 9.1.5 CPU vectorized environments are the first deliverable — Evidence: `crates/robot-sim-gpu/README.md` covers the GPU track. Note: no CPU vectorized batch runner was located in this pass; locate it or add one.
- [ ] 9.1.6 Environment steps per second measured to estimate training duration — Evidence: none found; no throughput benchmark artifact was located in this pass. Note: add a throughput benchmark artifact.
- [x] 9.1.7 Pure Rust simulator; no foreign MuJoCo binding is a dependency — Evidence: all dependencies path-only (see 2.4); `crates/robot-sim/README.md` lines 32-40 record authored ports of MuJoCo passive/integrator/constraint logic.
- [ ] 9.1.8 Rapier multibody qualification before any replacement articulation layer — Evidence: `grep` for `rapier` in `Cargo.toml` and `crates/*/Cargo.toml` found no matches. Note: qualify Rapier or record a decision to skip it (ADR).
- [ ] 9.1.9 Joint-limit and actuator model on the floating-base multibody path — Evidence: `crates/robot-sim/README.md` and ADR 0010 state no joint-limit or actuator model. Note: implement limits and actuator dynamics.
- [?] 9.1.10 Static standing, foot slip and push recovery results are not reported as balance capability before contact proxies and controller exist — Evidence: the rule is stated in SPECS.md Section 9.1; no report-generation code was audited in this pass.
- [~] 9.2.1 PPO with GAE, bounded actions, observation normalization, checkpoint/resume, recurrent support and termination-vs-truncation — Evidence: `crates/robot-learn/src/ppo.rs` (PPO), `gae.rs` (GAE with termination/truncation test per `robot-learn/README.md`), `checkpoint.rs` (resume), `gaussian_policy.rs` (diagonal Gaussian). Note: recurrent policy support was not found; bounded action handling was not confirmed.
- [x] 9.2.2 Checkpoint/resume stores weights, optimizer state, RNG state and normalizer statistics through the chunked format — Evidence: `crates/robot-learn/src/checkpoint.rs` and `robot-learn/README.md` lines 60-72 (schema v2, digest-verified chunks).
- [ ] 9.2.3 Asymmetric critic that may read simulator-only state — Evidence: `grep` for `asymmetric` and `privileged` in `crates/` finds only the bundle's privileged-observation rejection (`crates/robot-types/src/bundle.rs` lines 170-212). Note: implement a privileged critic input path.
- [~] 9.2.4 Exported actor passes an observation audit proving no privileged input remains — Evidence: `crates/robot-types/src/bundle.rs` `BundleError::PrivilegedObservation` (lines 207-212). Note: the check is a field declaration check, not a runtime audit of the exported actor's inputs.
- [~] 9.2.5 Balance and locomotion trained first, then motion tracking — Evidence: `crates/robot-tasks/src/balance.rs` exists. Note: locomotion and tracking tasks were not searched for in this pass.
- [~] 9.2.6 Rewards and costs cover task, tracking, balance, energy, smoothness, constraints and contacts, with each term logged independently — Evidence: `crates/robot-tasks/src/reward.rs` (reward terms). Note: independent logging per term and the full term set were not confirmed.
- [ ] 9.2.7 Domain randomization calibrated by sensitivity, with held-out disturbances — Evidence: `grep` for `domain_random` and `curriculum` found no provider. Note: implement a randomization provider with held-out sets.
- [~] 9.2.8 Checkpoint/resume with frozen parameter-group updates — Evidence: `crates/robot-learn/src/checkpoint.rs` and `robot-learn/README.md` lines 74-84 (`ParamGroup` with `frozen`). Note: `zero_frozen` mask in `backprop.rs` line 72 and the frozen-group test in `ppo.rs` are present.
- [~] 9.3.1 Every algorithm supplies a capability manifest with method ID, version, schemas, data, feature gate and determinism limits — Evidence: `crates/robot-types/src/schema.rs` and `crates/robot-artifacts/src/manifest.rs` provide manifests for artifacts. Note: no learner capability manifest type exists.
- [ ] 9.3.2 Learner, environment, rollout storage, model, reward, curriculum, evaluation, export, skill-space, opponent and retrieval extension points as traits — Evidence: public traits found are `Environment`, `Backend`, `Policy`, `StochasticPolicy`, `ModelImporter`, `HardwareAdapter`, `MotionKind`, `Migration`, `FidelityCheck`. Note: no learner, reward registry, curriculum, opponent, export, latent or retrieval trait exists.
- [~] 9.3.3 Concrete policy, backend and importer types are made traits behavior-preservingly before a second implementation is added — Evidence: `Policy` (`crates/robot-nn/src/policy.rs`), `Backend` (`crates/robot-sim/src/backend.rs`) and `ModelImporter` (`crates/robot-model/src/importer.rs`) traits exist. Note: the multibody solver has no trait; see 9.3.2 for the missing extension points.
- [ ] 9.4.1 Comparison of single compact tracker, latent tracker and bank of experts — Evidence: no tracker or expert-bank module found. Note: implement after tracking exists.
- [ ] 9.4.2 Staged latent pipeline with separately trainable and checkpointed stages — Evidence: no latent stage found (see 9.3.2). Note: implement after the tracker.
- [ ] 9.4.3 Opponent averaging and anti-cycling ablation against latest-snapshot self-play — Evidence: `grep` for `opponent` and `self-play` returns only `crates/robot-persona/src/lib.rs` lines 7-11, which states opponent memory is not implemented. Note: implement the opponent provider and the ablation.
- [ ] 9.5.1 Assisted sampling proposes candidates only, never selects or scores them — Evidence: no assisted-sampling module found. Note: implement as an offline tool with the run-record origin fields of 9.5.4.
- [ ] 9.5.2 Proposals are admitted only through the algorithm interface, with a manifest, ablation and separate feature flag — Evidence: no assisted-sampling module exists. Note: enforce the admission path when the module is added.
- [ ] 9.5.3 Held-out pool defined independently of any proposer — Evidence: no held-out pool definition found. Note: create a held-out set generator independent of proposals.
- [ ] 9.5.4 Every accepted proposal records its origin (tool, model identity and version, request parameters, date) in the run record — Evidence: no run record for proposals exists. Note: add origin fields to the run record.
- [~] 9.6.4 Closed enums or generics on deadline-critical paths; registries constructed once and never mutated during a run — Evidence: `crates/robot-types/src/bundle.rs` validates the bundle contract (`PrivilegedObservation`, `UndeclaredTier`); `crates/robot-nn/src/mlp.rs` lines 485-497 test action parity. Note: no registry exists, so the registry rule is untested.
- [~] 9.6.5 Common conformance suite for every extension (schema, termination/truncation, seeds, checkpoint, malformed input, export parity) — Evidence: per-module unit tests exist in `robot-learn` and `robot-types`. Note: no shared conformance suite was found.
- [ ] 9.6.6 Feature-removal test showing the baseline still builds without an optional method — Evidence: no such test found by `grep` for `feature` removal tests. Note: add a CI job that builds each optional feature off.
- [ ] 9.7.1 Retrieval-augmented learner with corpus, query, retrieval unit, similarity, admission filter and mixing declared in a manifest — Evidence: `crates/robot-skills/src/index.rs` retrieves runtime skills (Section 10), not training data. Note: implement an experience-retrieval provider; it is distinct from skill retrieval.
- [ ] 9.7.2 Retrieved transitions labelled with corpus, episode, embodiment and generation — Evidence: no experience-retrieval label path exists. Note: depends on 9.7.1.
- [ ] 9.7.3 Held-out data excluded from the retrieval corpus by construction and tested — Evidence: no corpus exists. Note: depends on 9.7.1.
- [?] 9.7.4 Retrieval compared against target-only and whole-corpus training at equal budget — not applicable until 9.7.1 exists.
- [?] 9.7.5 Retrieval quality reported separately from task outcome — not applicable until 9.7.1 exists.
- [?] 9.7.6 Embedding models for a corpus index record identity and version — not applicable until 9.7.1 exists.
- [ ] 9.7.7 Exact search as the default for experience retrieval — Evidence: `robot-skills` uses `ExactIndex` for skills only. Note: apply the same exact-search default to the experience provider.
- [?] 9.7.8 The retriever is removable without changing baseline acceptance — not audited; depends on 9.7.1.

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
| the reference simulator Rust bindings, including `reference-rs`, `rusty_mujoco`, `reference-rust`, `reference` | Foreign-engine interfaces; survey maintenance/version claims need checking | Excluded by default; optional cross-validation only following a separate scope decision |

The ferromotion survey reports FK, analytic Jacobians, RNEA, ABA, mass matrices, dynamics derivatives, LM IK, Paden–Kahan subproblems, trajectory optimization, contact solving, retargeting and identification. It also reports `ferromotion-learn`, calibration modules and wgpu rollout examples. Verify repository ownership, published crate/module names, licenses, versions and working code before evaluating these capabilities. Reported arm or cartpole demonstrations do not qualify humanoid contact dynamics or GPU humanoid training.

Candidate implementations of IK may use nalgebra, qualified Jacobians and a Rust QP solver. Mink remains a Python/the reference simulator algorithm reference. Select a solver only after testing feasibility detection, constraint accuracy, iteration limits and timing; LM IK and constrained differential-QP IK are not interchangeable.

Do not combine two physics solvers by casually assigning one dynamics and the other contact impulses. Any hybrid must specify state ownership, integration, impulse application and solver coupling, and pass energy/momentum and contact tests. Prefer evaluating complete candidate pipelines first.

`crates/robot-sim/SIMULATION_SPECS.md` specifies the GPU-batched, variable-fidelity, high-fidelity target this row's qualification work is now measured against, including the batching-strategy choice (one GPU thread per environment, running the existing authored ABA/RNE recursion largely unchanged, rather than restructuring it into per-tree-level batched tensor operations) and why: see that document's Section 11.1. This does not relax the Rust-native requirement above it — the reference simulator itself is a design reference for that document's Sections 4-10, never a dependency.

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


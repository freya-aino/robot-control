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


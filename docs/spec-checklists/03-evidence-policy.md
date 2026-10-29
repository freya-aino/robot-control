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
- PHC commonly refers to Perpetual Humanoid Control. Mink is a differential inverse-kinematics library built around MuJoCo, not inherently a GPU batched SMPL pipeline. These names must not become misleading implementation requirements.
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



## Implementation checklist

Status legend: `[x]` implemented, `[~]` partial, `[ ]` not implemented, `[?]` not audited in this pass. Summary: 5 `[x]`, 2 `[~]`, 0 `[ ]`, 3 `[?]` across 10 items.

- [x] 3.1 The five evidence states exist and are ordered — Evidence: `crates/robot-types/src/evidence.rs` lines 16-34 (enum), 48-56 (`rank`).
- [x] 3.2 Only states above `unverified` may be cited in an implementation decision — Evidence: `crates/robot-types/src/evidence.rs` `is_citable` (lines 106-111).
- [x] 3.3 Promotion requires the admitting evidence and a dated ledger entry — Evidence: `docs/ledger/evidence-ledger.md` (each row has state, admitting artifact and date; promotion rules in `crates/robot-types/src/evidence.rs`).
- [?] 3.4 Demotion when a dependency, profile, firmware or acceptance version changes — `evidence.rs` module docs claim demotion is an operation; the function body was not read in this pass.
- [x] 3.5 Cross-simulator agreement is not a state — Evidence: `evidence.rs` header (lines 8-9) and the enum (lines 16-34) has no such variant.
- [~] 3.6 Source metadata (URL, title, authors, version, accessed date, commit, license, evidence type, claim) is recorded before research is relied on — Evidence: `docs/decisions/0012-cubecl-b6-qualification-gpu-backend.md` records authors, repository, accessed date and source inspected. Note: the research intake register in this segment lists no URLs or accessed dates for most leads.
- [x] 3.7 The research intake register is maintained — Evidence: the section 3.1 table in SPECS.md, with RoboStriker dispositioned as `paper_verified` on 2026-10-26.
- [~] 3.8 A failed verification is recorded as a correction record — Evidence: `crates/robot-learn/src/ppo.rs` lines 925-929 record a corrected test assertion. Note: no repo-wide correction-record store exists.
- [?] 3.9 Design corrections are respected (for example, retrieval is not described as runtime routing, decision age is not treated as zero latency) — not audited across all crates.
- [?] 3.10 Leads are verified for identity, claims and licensing before a reproduction is scheduled — not audited.

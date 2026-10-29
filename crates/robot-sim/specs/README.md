# robot-sim Simulation Specifications — Index

This directory replaces the former single-file `crates/robot-sim/SIMULATION_SPECS.md` (263 lines). That file is now a pointer to this index, so existing references to it (ADR 0005, `README.md`, `PLAN.md`, `lib.rs`) remain valid.

Status: proposed architecture for `robot-sim`'s physics layer. Not ratified; no clause below is evidence that the described system exists. This document specializes `SPECS.md` Section 9.1 and Appendix B.4 for one crate; it does not amend them.

| Field | Value |
| --- | --- |
| Document version | 0.1.1-draft |
| Last revised | 2026-10-29 |
| Companion documents | `SPECS.md` (platform specification; this document inherits its normative-language and evidence-state conventions without restating them), `PLAN.md` Phase 3, `docs/decisions/0005-mujoco-caliber-gpu-batched-simulation-target.md`, `crates/robot-sim/README.md` |
| Scope | `robot-sim`'s physics engine only: dynamics, contact, collision, actuation, integration and their GPU-batched execution. Motion capture, retargeting, learning algorithms and hardware are out of scope and unaffected |
| Normative language | MUST/SHOULD/MAY are used exactly as defined in `SPECS.md`'s "Normative language" section (RFC 2119/8174) |
| Evidence language | `unverified`/`paper_verified`/`code_inspected`/`reproduced_in_sim`/`validated_on_hardware` are used exactly as defined in `SPECS.md` Section 3. Per that table, `code_inspected` licenses statements about what an implementation does and requires; it does **not** license statements about its numerical correctness or performance. Every MuJoCo/MJX/Brax claim below is tagged against that table, not asserted as a performance guarantee |

## Step template

Every step file uses the same header fields:

- **Status** — current implementation state, updated as implementation steps land. Not a claim of numerical correctness or parity.
- **Goal** — what the step specifies.
- **Depends on** — other step files whose output this step needs.
- **MuJoCo baseline** — the pinned MuJoCo 3.17.0 source file(s) and function name(s) that this step's algorithm is checked against (Appendix A pin, commit `800a109a4bdd02cb60404d565db4ae16fb04a53b`). Each function is located by name at that commit. The field states the evidence tag: `code_inspected` only where the function body was read, otherwise `unverified`. A step with no algorithm function says so explicitly.
- **Purity contract** — the constraints from FR-6 (`std` and plain `f64` arrays only in algorithm modules; no `robot_types::math::Matrix`, `RigidPose` or `HashMap` in algorithm functions).
- **Inputs/outputs** — the data the step consumes and produces.
- **Done when** — the observable completion criteria.
- **Open questions** — Appendix B entries that block the step, or `none blocking`. Non-blocking items are not listed here.

Each step file links back to this index in its first line. Cross-references to other steps and to appendices are Markdown links. The old spec's requirements and content are mapped below into step files and appendices; wording is adjusted where implementation status or superseded history requires clarification. Step 04 adds the source-faithful oracle plan.

## Step index

| Step file | Step | Status | Depends on |
| --- | --- | --- | --- |
| [00-target-and-constraints.md](00-target-and-constraints.md) | Target, constraints, evidence base | Specification only | — |
| [01-state-and-coordinates.md](01-state-and-coordinates.md) | Generalized coordinates, `Free` / `Ball` joints | Partially implemented (`Free` done; `Ball` deferred) | 00 |
| [02-pure-dynamics-core.md](02-pure-dynamics-core.md) | Mass matrix (CRBA), bias forces (RNE), frame convention | Partially implemented (spatial and ABA/RNE paths use plain arrays; CRBA / M(q) not implemented) | 01 |
| [03-integrators.md](03-integrators.md) | Integrator family (`implicitfast`, `implicit`, RK4) | Partially implemented (semi-implicit Euler, implicit-damping Euler and RK4 CPU subsets) | 02 |
| [04-mujoco-fidelity-loop.md](04-mujoco-fidelity-loop.md) | Source-faithful oracle plan: oracle definitions, tolerance table per tier, derivation rules | Partially implemented (`oracle.rs` and fidelity schema v7) | 02, 03 |
| [05-passive-forces-and-actuators.md](05-passive-forces-and-actuators.md) | Passive forces, actuators, tendons (deferred) | Partially implemented (linear scalar-joint spring/damper and affine direct-joint actuator subset) | 02, 03 |
| [06-equality-constraints-and-solver.md](06-equality-constraints-and-solver.md) | Contact and equality constraint solver | Partially implemented (`mjEQ_JOINT` equality-only solve and plane-sphere geometry; no shared contact solver) | 03, 05 |
| [07-collision-pipeline.md](07-collision-pipeline.md) | Broad, mid and narrow phase collision | Partially implemented (standalone plane-sphere primitive; collision pipeline not implemented) | 01 |
| [08-model-compilation-and-sensors.md](08-model-compilation-and-sensors.md) | Model compilation, asset import, sensors | Not started | 01, 02 |
| [09-fidelity-tiers.md](09-fidelity-tiers.md) | Enumerated fidelity knobs and named tiers | Not started | 03, 06, 07 |
| [10-cpu-reference-backend.md](10-cpu-reference-backend.md) | CPU reference backend and backend-agnostic interface | Partially implemented (single-instance backend, `Environment`) | 02, 03 |
| [11-gpu-batched-backend.md](11-gpu-batched-backend.md) | GPU-batched execution architecture | Partial: CubeCL f32 kernel subset in `crates/robot-sim-gpu` (`cubecl` feature; ADR 0012); G1/ABA and full contact solver not ported | 10 |
| [12-cpu-gpu-agreement.md](12-cpu-gpu-agreement.md) | Validation, evidence and cross-backend agreement | Partial: feature-gated CubeCL f32 CPU/GPU agreement subset (ADR 0012); no full G1/ABA agreement | 10, 11 |

Dependencies are derived from the cross-references in the old spec; they are a starting point to be confirmed at each step's review.

Appendices:

| File | Content |
| --- | --- |
| [appendix-a-citations.md](appendix-a-citations.md) | Appendix A — Citation ledger |
| [appendix-b-open-questions.md](appendix-b-open-questions.md) | Appendix B — Open questions (S-01 to S-06) |
| [appendix-c-existing-code.md](appendix-c-existing-code.md) | Appendix C — Relationship to existing `robot-sim` code |

## Old-to-new section map

Every top-level section and appendix of the old spec appears in exactly one primary location. Sections marked **split** have items or subsections that belong to different steps; each part is listed, and every item appears exactly once.

| Old section | Title | New location |
| --- | --- | --- |
| Header (lines 1–13) | Status, document metadata | This index (above) |
| §0 | Purpose and how this document was built | [00](00-target-and-constraints.md) |
| §1 | Target and explicit non-goals | [00](00-target-and-constraints.md) |
| §2 | Evidence base at a glance | [00](00-target-and-constraints.md) |
| §3 | State representation and coordinates | [01](01-state-and-coordinates.md) |
| §4 | Dynamics: mass matrix, bias forces, integration — **split** | items 1–2 in [02](02-pure-dynamics-core.md); items 3–5 in [03](03-integrators.md) |
| §5 | Contact and constraint solver | [06](06-equality-constraints-and-solver.md) |
| §6 | Collision detection pipeline | [07](07-collision-pipeline.md) |
| §7 | Actuators, tendons, equality constraints — **split** | items 1 and 4 in [05](05-passive-forces-and-actuators.md); items 2–3 in [06](06-equality-constraints-and-solver.md) |
| §8 | Passive forces | [05](05-passive-forces-and-actuators.md) |
| §9 | Sensors | [08](08-model-compilation-and-sensors.md) |
| §10 | Model compilation and asset import | [08](08-model-compilation-and-sensors.md) |
| §11 | GPU-batched execution architecture — **split** | §11.1–11.3 in [11](11-gpu-batched-backend.md); §11.4 in [10](10-cpu-reference-backend.md) |
| §12 | Variable fidelity as a first-class interface | [09](09-fidelity-tiers.md) |
| §13 | Backend-agnostic architecture | [10](10-cpu-reference-backend.md) |
| §14 | Validation and evidence requirements | [12](12-cpu-gpu-agreement.md) |
| §15 | Explicit non-goals (restated) | [00](00-target-and-constraints.md) |
| Appendix A | Citation ledger | [appendix-a-citations.md](appendix-a-citations.md) |
| Appendix B | Open questions | [appendix-b-open-questions.md](appendix-b-open-questions.md) |
| Appendix C | Relationship to existing code | [appendix-c-existing-code.md](appendix-c-existing-code.md) |

Step 04 (`04-mujoco-fidelity-loop.md`) has no old-spec section; it is new content owned by Step 2 of the plan.

## Cross-references

Bodies of the step files keep the old `Section N` numbering, so references such as "Section 13.2" or "Section 11.4" resolve through the map above. Appendix B IDs (S-01 to S-06) are cited from the "Open questions" fields of the step files.

# robot-sim Simulation Specifications — Index

This directory replaces the former single-file `crates/robot-sim/SIMULATION_SPECS.md` (263 lines). That file is now a pointer to this index, so existing references to it (ADR 0005, `README.md`, `PLAN.md`, `lib.rs`) remain valid.

Status: proposed architecture for `robot-sim`'s physics layer. Not ratified; no clause below is evidence that the described system exists. This document specializes `SPECS.md` Section 9.1 and Appendix B.4 for one crate; it does not amend them.

| Field | Value |
| --- | --- |
| Document version | 0.1.9-draft |
| Last revised | 2026-11-01 |
| Change in 0.1.9-draft | **`condim` 4 and 6 implemented** ([06](06-equality-constraints-and-solver.md) item 4, a MUST): torsional friction about the contact normal and rolling friction about the two tangents, as the example's `2(condim - 1)` generator rows. The split that matters is which Jacobian a generator multiplies — sliding tangents the translational one, spin and rolling the rotational one — so these differ from `condim` 3 in kind rather than in length, and `point_jacobian_dot` now returns the rotational derivative they need. A direction with a zero coefficient contributes no rows, which matters more here than at 3 because torsional and rolling coefficients are commonly zero; `solver::sliding_friction` builds the example's five-element vector with those entries zero, so 4 and 6 behave as 3 until a coefficient is supplied |
| Change in 0.1.8-draft | **`implicitfast` implemented** ([03](03-integrators.md) item 4, a MUST): `mj_implicitSkip` with `flg_bias = 0`, solving `(M - h dF/dv) qacc = tau - qfrc_bias`. For the forces this crate applies the derivative is diagonal, so `qH = M + h diag(kd)` adds to the sparse mass matrix's own diagonal. Measured: with the G1's drive damping scaled up and its stiffness zeroed at `dt = 2 ms`, the explicit step drives the damped hinges to `\|qd\| = 5.1e2` from a start of 0.5 where `implicitfast` keeps them at 0.74; with the damping removed the two agree to 1e-12. On the G1's *declared* gains it changes nothing measurable, which is the point — it removes a ceiling rather than changing an answer. `--integrator euler\|implicitfast\|rk4` is a per-run knob. Full `implicit` and `mjd_freeMhat`'s free-body gyroscopic block remain unported |
| Change in 0.1.7-draft | **M-27 resolved**: mesh colliders are reduced to their convex hull at import (`robot_model::hull`) and contacted through the support query `argmax_i (v_i . d)` that `mjc_meshSupport` computes. The G1 reports 33 collision shapes modelled and 0 not modelled, exactly what its layer declares, and `usd::PROXY_LINKS` is gone with the two boxes it fabricated. The reduction is exact rather than an approximation, because the support function of a point set equals the support function of its convex hull — every hull test asserts that directly, in 144 directions, rather than asserting a vertex count. Two costs remain and are stated as such: the support query still scans (the example hill-climbs a vertex adjacency graph above ten vertices), and a plane-hull contact is one point because the three extra contacts of `mjc_PlaneConvex` need polygon topology a vertex hull does not carry |
| Change in 0.1.6-draft | The example's specifications are **MUST**, in the normative sense of `SPECS.md`: where pre-existing mathematics works differently it is **overwritten**, not recorded as a deviation. [Appendix D](appendix-d-mathematical-challenges.md) section 1.0.2 states the rule and the three categories a reader must tell apart — a competing formulation (defect, overwritten), a second implementation used as an oracle (kept, and not a deviation), and a stricter input contract (kept, re-labelled validation). Overwritten under it: the per-point penalty contact and the `--contact` selector that reached it are gone, and `contact::plane_sphere_contact` takes the plane's own frame instead of assuming `+z`. `articulated`'s articulated-body recursions and `constraint::solve_equality` are kept and labelled in their own module documentation as oracles, because removing them would leave the primary path unvalidatable |
| Change in 0.1.5-draft | Reworks every step file against the example's implementation of these dynamics and against the problems now measured. The switch-over is done: `world::World` runs composite dynamics, exact collision manifolds and the unified constraint solver, and `robot sim multibody` runs it by default (M-17 closed). `newton` adds the primal solver of example `08` section 6 and is the **default**, on a measured convergence argument rather than a preference — Gauss-Seidel needs 1011 sweeps on sixteen pyramidal rows where Newton needs one. The friction cone is corrected to the example's generator-direction pyramid, which is the inscribed diamond, not the circumscribed square an earlier pass built (41 per cent too much diagonal friction). The model's declared actuators are simulated (M-25) in the model's own units (M-26). Four entries open: **M-22** GPU f32 and degree-of-freedom cap, **M-27** 15 of the G1's 33 colliders unsimulated plus two fabricated bounding-box proxies, **M-28** the declared drive gains are generated defaults. [Appendix D](appendix-d-mathematical-challenges.md) section 1.7 is the current status and supersedes 1.6 |
| Change in 0.1.4-draft | Records the second correction pass. Fourteen of Appendix D's twenty-four entries are now closed; [Appendix D](appendix-d-mathematical-challenges.md) section 1.6 is the current status table and supersedes 1.5, with 1.6.1 and 1.6.2 stating what the two remaining entries need. New in `robot-sim`: `solver` (the unified constraint problem of `07` and `08`) and `options` (the per-run fidelity knobs of Section 12.1). The battery moves to schema v8 with a plane force-balance, a friction-cone and a narrow-phase row (ADR 0008 amendment) |
| Change in 0.1.3-draft | Records the precedence change of [Appendix D](appendix-d-mathematical-challenges.md) section 1.0.1: `docs/example-system-specs/` is now the single source of truth for the mathematics and execution model of multi-body simulation, while this project's goals, scope, language, dependency policy, evidence discipline and GPU-bound target are unchanged. Appendix D gains a status section (1.5) recording which of M-01 to M-24 are fixed, and 1.5.1 recording what M-05 and M-06 turned out to need. Appendix B **S-07** and **S-08** are resolved |
| Change in 0.1.2-draft | Adds [Appendix D](appendix-d-mathematical-challenges.md), the record of a full implementation audit against these step files and against the non-normative `docs/example-system-specs/`, and adds a **Challenges** field to the steps it found entries against (01–07, 09, 11, 12). Opens Appendix B **S-07** to **S-09** for the three entries needing a specification decision. No requirement, acceptance row or deviations list is changed by this revision; the audit records where the code does not meet the text already here |
| Companion documents | `SPECS.md` (platform specification; this document inherits its normative-language and evidence-state conventions without restating them), `PLAN.md` Phase 3, `docs/decisions/0005-gpu-batched-simulation-target.md`, `crates/robot-sim/README.md` |
| Scope | `robot-sim`'s physics engine only: dynamics, contact, collision, actuation, integration and their GPU-batched execution. Motion capture, retargeting, learning algorithms and hardware are out of scope and unaffected |
| Normative language | MUST/SHOULD/MAY are used exactly as defined in `SPECS.md`'s "Normative language" section (RFC 2119/8174) |
| Evidence language | `unverified`/`paper_verified`/`code_inspected`/`reproduced_in_sim`/`validated_on_hardware` are used exactly as defined in `SPECS.md` Section 3. Per that table, `code_inspected` licenses statements about what an implementation does and requires; it does **not** license statements about its numerical correctness or performance. Every the reference simulator/batched backend/Brax claim below is tagged against that table, not asserted as a performance guarantee |

## Step template

Every step file uses the same header fields:

- **Status** — current implementation state, updated as implementation steps land. Not a claim of numerical correctness or parity.
- **Goal** — what the step specifies.
- **Depends on** — other step files whose output this step needs.
- **Reference formulation** — the pinned reference formulation source file(s) and function name(s) that this step's algorithm is checked against (Appendix A pin, commit `800a109a4bdd02cb60404d565db4ae16fb04a53b`). Each function is located by name at that commit. The field states the evidence tag: `code_inspected` only where the function body was read, otherwise `unverified`. A step with no algorithm function says so explicitly.
- **Purity contract** — the constraints from FR-6 (`std` and plain `f64` arrays only in algorithm modules; no `robot_types::math::Matrix`, `RigidPose` or `HashMap` in algorithm functions).
- **Inputs/outputs** — the data the step consumes and produces.
- **Done when** — the observable completion criteria.
- **Open questions** — Appendix B entries that block the step, or `none blocking`. Non-blocking items are not listed here.
- **Challenges** — present only on a step the implementation audit of [Appendix D](appendix-d-mathematical-challenges.md) found something in. It names the Appendix D entries that step owns, with their severity, and is a statement about the code, never a change to the requirement above it. A step with no such field had nothing found against it; `10` is in that position, and `00` states the target rather than an algorithm. `08` acquired one in 0.1.5-draft, when the USD import was audited against the layer.

Each step file links back to this index in its first line. Cross-references to other steps and to appendices are Markdown links. The old spec's requirements and content are mapped below into step files and appendices; wording is adjusted where implementation status or superseded history requires clarification. Step 04 adds the source-faithful oracle plan.

## Current priorities

Four Appendix D entries are open. In the order the dependencies run:

1. **Narrow-phase support cost** ([07](07-collision-pipeline.md)). M-27 is
   resolved and every one of the G1's 33 colliders is now simulated, but the
   support query over a hull is still an exhaustive scan and the pelvis hull is
   5583 vertices. Example 04 section 1.5 answers this with
   `mjc_hillclimbSupport`, walking a vertex adjacency graph above a ten-vertex
   threshold. A cost, not a different answer: a 2-second G1 stance at 0.5 ms
   runs in 8.5 s of wall clock today, so this is the first thing that will stop
   being affordable. The same change would supply the polygon topology
   `mjc_PlaneConvex` needs for its three extra plane-hull contacts, so a hull
   lying flat on the plane stops getting a single contact it cannot resist
   tipping about.
2. **M-28 — the declared drive gains** ([05](05-passive-forces-and-actuators.md)).
   Not a code defect: the actuators are simulated correctly, at correct units,
   and the gains they carry are generated defaults. Open decision S-10 (what a
   named tier pins, and whether to carry a published gain set as a declared
   override).
3. **Full `implicit`** ([03](03-integrators.md)). `implicitfast` has landed and
   meets item 4's MUST; what remains is the `qLU` path with the
   recursive-Newton-Euler derivative, and `mjd_freeMhat`'s free-body gyroscopic
   block — which is what makes `implicitfast` and `implicit` identical for a
   standalone free body, so a tumbling free body is the case that notices.
4. **M-22 — the GPU backend** ([11](11-gpu-batched-backend.md),
   [12](12-cpu-gpu-agreement.md)). The solver prerequisite has landed; what
   remains is a sparse Hessian, the sparse tree factorization of example `12`
   section 7.2, and the `f32`/`MAX_DOF` questions against the G1's 35 degrees of
   freedom at `cond(M) ~ 1e5`.

One methodological point earned the hard way and recorded in
[Appendix D](appendix-d-mathematical-challenges.md) sections 1.7.4 and 1.7.6,
because it bears on how any of the above should be validated: **a simulation
that looks plausible is weak evidence about the component you are testing.** A
hand-written joint hold with invented gains produced a passing stance for weeks
while masking a factor-of-57 unit error in the model import, and the dual
solver's failure to converge was acting as damping that hid four degenerate
constraint rows. Both were found by making the thing under test actually run and
converge, not by reading it.

## Step index

| Step file | Step | Status | Depends on |
| --- | --- | --- | --- |
| [00-target-and-constraints.md](00-target-and-constraints.md) | Target, constraints, evidence base | Specification only | — |
| [01-state-and-coordinates.md](01-state-and-coordinates.md) | Generalized coordinates, `Free` / `Ball` joints | `Free` implemented **in the example's convention** (world linear first, body angular second), momentum exact to 1e-14; axes normalised at one accessor. `Ball` deferred. D/M-01, M-02 closed; B/S-07 resolved. The layout is not interchangeable with `articulated`'s | 00 |
| [02-pure-dynamics-core.md](02-pure-dynamics-core.md) | Mass matrix (CRBA), bias forces (RNE), frame convention | **Implemented**: CRBA, `L' D L` factorization, back-substitution and RNE in `composite`, world-orientation frame. Cross-validated against the independent local-frame recursion to 1e-10 on the G1. The example has no articulated-body algorithm; it forms `M` and solves. D/M-18, M-24 closed | 01 |
| [03-integrators.md](03-integrators.md) | Integrator family (`implicitfast`, `implicit`, RK4) | **Item 4's MUST met.** Semi-implicit Euler, `implicitfast` (`qH = M + h diag(kd)`, carrying a damping gain that destabilises the explicit step by 685x) and RK4 with the Munthe-Kaas correction, all per-run selectable (D/M-17, M-18, M-19 closed). Full `implicit` and the free-body gyroscopic block of `mjd_freeMhat` remain unported | 02 |
| [04-fidelity-loop.md](04-fidelity-loop.md) | Source-faithful oracle plan: oracle definitions, tolerance table per tier, derivation rules | Schema v8: ten checks including a plane force-balance, a friction-cone bound and a narrow-phase depth row (D/M-20 closed, ADR 0008 amended). Writing them found a real collision defect on the first sample | 02, 03 |
| [05-passive-forces-and-actuators.md](05-passive-forces-and-actuators.md) | Passive forces, actuators, tendons (deferred) | **The model's own actuators are simulated and run** — `mj_fwdActuation` as the example's position servo, with the force clamp, the target clamp and the transmission (D/M-17, M-25, M-26 closed). D/M-28 open: the G1's declared gains are generated defaults, so `ActuationMode` makes applying them a choice. Activation dynamics, tendons and the non-spring passive terms unported | 02, 03 |
| [06-equality-constraints-and-solver.md](06-equality-constraints-and-solver.md) | Contact and equality constraint solver | **Implemented, both algorithms, running.** One convex problem over contact, limit and equality rows; the example's generator-direction pyramid cone; primal Newton (example `08` §6) as the **default** with projected Gauss-Seidel as its cross-check, agreeing to 1e-8. All four `condim` values, including torsional and rolling friction on the rotational Jacobian. D/M-08 to M-13 and M-23 closed; B/S-09 resolved. Missing: elliptic cone, islands | 03, 05 |
| [07-collision-pipeline.md](07-collision-pipeline.md) | Broad, mid and narrow phase collision | Broad phase, pair filter, signed-volume GJK with the polytope2/3/4 constructions, exact plane manifolds and the edge-edge fix all landed (D/M-04 to M-07, M-15, M-16 closed). **D/M-27 open and it is a MUST**: 15 of the G1's 33 colliders are meshes and are not collided, and two bounding-box proxies the model does not declare are fabricated. The fix is a support-query convex hull | 01 |
| [08-model-compilation-and-sensors.md](08-model-compilation-and-sensors.md) | Model compilation, asset import, sensors | **USD import exists and is audited exact** on masses (33.341142 kg to the digit), all 30 inertia tensors through `principalAxes` to 1e-6, and all 29 joint limits, velocities, efforts and axes with zero mismatches. **Item 3 violated** (D/M-27): mesh collision is dropped rather than hulled. Sensors not started | 01, 02 |
| [09-fidelity-tiers.md](09-fidelity-tiers.md) | Enumerated fidelity knobs and named tiers | Knobs implemented as `options::SimOptions`: per-run, validated rather than clamped, recorded per run (D/M-21 closed). Three were added by measurement — `limit_margin`, `max_faces` as a termination condition, and the solver algorithm. Named tiers remain open under B/S-03 | 03, 06, 07 |
| [10-cpu-reference-backend.md](10-cpu-reference-backend.md) | CPU reference backend and backend-agnostic interface | **One simulation path, two backends behind one interface**: `cpu-world` (corrected path) and `cpu-multibody` (legacy). Both names reported, because their observation layouts differ for a `Free` joint. `ArticulatedBody` retained deliberately as the independent oracle | 02, 03 |
| [11-gpu-batched-backend.md](11-gpu-batched-backend.md) | GPU-batched execution architecture | Partial: CubeCL f32 kernel subset (ADR 0012). **The CPU prerequisite has landed** — the primal Newton solver the batched path needs now exists, for two independent reasons (Gauss-Seidel neither vectorises nor converges). Still needed: a sparse Hessian and the sparse tree factorization of example `12` §7.2. D/M-22 open: `MAX_DOF = 6` against 35, f32 against `cond(M) ~ 1e5` | 10 |
| [12-cpu-gpu-agreement.md](12-cpu-gpu-agreement.md) | Validation, evidence and cross-backend agreement | Partial: feature-gated CubeCL f32 agreement subset (ADR 0012). **The CPU reference is no longer unoracled for contact** (D/M-20 closed) and the two CPU solvers cross-check to 1e-8, so a GPU backend now has something to agree with. D/M-22 sets the f32 tolerance; D/M-27 means a contact agreement test has no geometry to use | 10, 11 |

Dependencies are derived from the cross-references in the old spec; they are a starting point to be confirmed at each step's review.

Appendices:

| File | Content |
| --- | --- |
| [appendix-a-citations.md](appendix-a-citations.md) | Appendix A — Citation ledger |
| [appendix-b-open-questions.md](appendix-b-open-questions.md) | Appendix B — Open questions (S-01 to S-06; S-07 to S-09 resolved) |
| [appendix-c-existing-code.md](appendix-c-existing-code.md) | Appendix C — Relationship to existing `robot-sim` code |
| [appendix-d-mathematical-challenges.md](appendix-d-mathematical-challenges.md) | Appendix D — Mathematical and algorithmic challenges to the current implementation (M-01 to M-28; 26 closed, M-22 and M-28 open) |

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

Appendix D (`appendix-d-mathematical-challenges.md`) has no old-spec section either; it is the record of the 2026-10-31 implementation audit against the step files and against `docs/example-system-specs/`.

Step 04 (`04-fidelity-loop.md`) has no old-spec section; it is new content owned by Step 2 of the plan.

## Cross-references

Bodies of the step files keep the old `Section N` numbering, so references such as "Section 13.2" or "Section 11.4" resolve through the map above. Appendix B IDs (S-01 to S-06) are cited from the "Open questions" fields of the step files.

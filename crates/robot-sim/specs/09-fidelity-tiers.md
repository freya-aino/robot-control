# Step 09 — Variable fidelity as a first-class interface

[Index](README.md) · [Appendix A](appendix-a-citations.md) · [Appendix B](appendix-b-open-questions.md)

**Status**: **Knobs implemented; named tiers still open.** `options::SimOptions`
carries every knob old Section 12 enumerates as a per-run parameter —
`timestep`, `integrator`, `solver`, `solver_iterations`, `solver_tolerance`,
`condim`, `contact_slots`, `margin`, `gap`, `limit_margin`, `warm_start`,
`solref`, `solimp` and the convex-collision budget (`CcdConfig`: tolerance,
iterations, `max_faces`). They are **validated rather than clamped**: an invalid
combination is refused, not silently adjusted. `SimOptions::record()` emits them
as key/value pairs so a run carries the configuration that produced it, which is
Section 12.1's requirement. Appendix D M-21 is closed.

Three knobs were added by measurement rather than by enumeration, and each has
its number in its own documentation: `limit_margin`, separate from the collision
`margin` as the example keeps `jnt_margin` separate from `geom_margin` (0.177
rad of overshoot at zero margin against exactly zero at 0.2 rad);
`CcdConfig::max_faces`, which is a **termination condition** and not a budget
(one EPA query took 2.9 s without it, because the face count of a non-hull
polytope grows geometrically); and the solver algorithm, where the default is
Newton for the convergence reason recorded in
[06](06-equality-constraints-and-solver.md).

An actuation mode is a fourth such knob and is **not** in `SimOptions`:
`ActuationMode` is a parameter of the actuation call rather than of the run,
because it selects how a model's declared drives are read rather than how
accurately they are integrated. Whether it belongs in a named tier is open; see
[05](05-passive-forces-and-actuators.md) and Appendix D M-28.

The analytic fidelity harness (`fidelity.rs`) is at schema v8 and runs ten
checks, now including a plane force balance, a friction-cone bound and a
narrow-phase depth row (Appendix D M-20 closed, ADR 0008 amended). The named
tiers themselves remain open under Appendix B S-03.

**Goal**: Make each fidelity knob (timestep, integrator, solver algorithm, iteration budget, `condim`, contact-slot budget, collision margin/gap) an explicit per-run parameter, and define named tiers as bundles of those knobs. Covers old §12 in full.

**Depends on**: [03](03-integrators.md) (integrator and timestep), [06](06-equality-constraints-and-solver.md) (solver and `condim`), [07](07-collision-pipeline.md) (margin and gap).

**Reference formulation**: Tag `3.17.0`, commit `800a109a4bdd02cb60404d565db4ae16fb04a53b` (Appendix A, entries 3, 5, 12, 15). This step defines no algorithm and pins no engine function. Each knob's function pin is inherited from its owning step: integrator from [03](03-integrators.md) (`mj_Euler`, `mj_implicit`, `mj_RungeKutta`), solver from [06](06-equality-constraints-and-solver.md) (`mj_solNewton`, `mj_solCG`, `mj_solPGS`), and margin/gap from [07](07-collision-pipeline.md) (`mj_collision`). the reference simulator exposes these as unnamed knobs (entry 15), so the named tiers are this project's own design. Evidence tag: `unverified` for the tier mapping.

**Purity contract**: Tier definitions are plain data (a named bundle of knob values). They MUST NOT carry `Matrix`, `RigidPose` or `HashMap` (FR-6). Tier lookup, if any, is a `match`, not a map.

**Inputs/outputs**: Input: a tier name or explicit knob values. Output: a resolved configuration used by the backend, and the knob values recorded with each run.

**Done when**: Every knob in the §12.1 table is a queryable run parameter. Every named tier is a pure bundle of knobs. A caller can start from a tier and override one knob, and the resulting configuration is reported.

**Open questions**: Appendix B S-03 (exact tier names and settings, grounded in a CPU fidelity sweep) is blocking for the named-tier item of this step. The knob enumeration is not blocked.

**Status (updated 2026-11-01)**: the knobs of Section 12.1 **are** a per-run configuration, `robot_sim::options::SimOptions`. All seven are explicit fields with documented defaults, invalid values are rejected rather than clamped (`condim` 4 and 6 are refused because they are unimplemented, not silently treated as 3), and `record` returns every knob as text so a run can carry its own configuration -- which Section 12.1 requires and without which a fidelity claim cannot be audited. A tier is a plain bundle a caller can override one field of, which is the section's closing requirement. Appendix D M-21 is closed. The *named* tiers and their per-knob settings remain open under Appendix B S-03, which wants them grounded in a measured sweep rather than guessed.

**Challenges (as first recorded)**: [Appendix D](appendix-d-mathematical-challenges.md) M-21 (**gap**). Section 12.1 requires each of seven knobs to be "an explicit, per-run, queryable parameter ... never a compile-time constant". Today only the timestep is one. Integrator and solver exist as `Tier` fields but are `String`s validated to accept a single value each; the solver iteration/tolerance budget is the private constants `CONTACT_PGS_SWEEPS = 500` and `CONTACT_PGS_TOLERANCE = 1e-10`; `condim`, the contact-slot budget and collision margin/gap do not exist on the CPU path at all. Appendix D adds three further constants that belong on the enumeration because they change the physics and not only the cost: `RIM_SAMPLES = 8` (it scales contact stiffness — M-04), `ADJACENT_OVERLAP_ALLOWANCE = 0.003` (M-16) and `CONTACT_REFERENCE_MASS = 1.0` (M-09).

So `Tier` currently distinguishes runs by timestep alone, and this step's closing requirement — every tier expressible purely as a named bundle of the 12.1 knobs — has four of seven knobs with nothing to bundle. This does not block the design work S-03 owns, but it does mean a tier cannot yet be *recorded* faithfully with a run, which is what makes a fidelity claim auditable at all.

---

## 12. Variable fidelity as a first-class interface

### 12.1 Enumerated knobs

The following MUST each be an explicit, per-run, queryable parameter of a simulation configuration — never a compile-time constant and never silently defaulted without the default being documented:

| Knob | Range of effect | Research basis |
| --- | --- | --- |
| Timestep | Smaller = more accurate, proportionally slower; too large relative to constraint stiffness = unstable (Section 5.3) | `[the reference simulator §15]` |
| Integrator | Euler-equivalent (cheapest, needs implicit damping per Section 4.3-4.4 to be usable at all) through RK4-equivalent (costliest, best long-horizon energy behavior for tumbling free bodies) | `[the reference simulator §5, §15]` |
| Solver algorithm | Newton-equivalent (few iterations, more per-iteration cost) vs. CG-equivalent (more iterations, cheaper each) | `[the reference simulator §3, §15]` |
| Solver iteration/tolerance budget | the reference's own documentation endorses "often just one" Newton iteration as sufficient for RL-training fidelity specifically — i.e. this knob's acceptable range is *task-dependent*, not a universal accuracy dial | `[batched backend §15]` |
| Contact dimensionality (`condim`) | 1 (frictionless) through 6 (+torsional +rolling); strictly more solver cost for strictly more physical realism | `[the reference simulator §3, §15]` |
| Contact-slot budget (GPU backend only) | Trades "contacts silently dropped under load" against memory/throughput (Section 11.2) | `[batched backend §12, §15]` |
| Collision margin/gap | How early a near-contact is detected/reported vs. force-generating | `[the reference simulator §4, §15]` |

### 12.2 Named fidelity tiers (this project's addition)

the reference simulator itself exposes none of the above as named tiers — research confirmed only an unnamed, independently-tunable knob pile, with its own documentation reading as "here is a direction to turn each knob," not "here are three presets" [the reference simulator §15]. NVIDIA's Isaac Lab comes closer with a named, documented two-way solver choice (PGS vs. TGS) and a whole family of named, swappable solver *backends* (PhysX, MJWarp, Kamino, VBD, MPM), each with its own tuning guide [IsaacLab §15]. Brax's four pipelines (batched backend, Generalized, Positional, Spring) are the closest found precedent for a single engine exposing an explicitly speed/fidelity-ordered set of named choices behind one interface [Brax §15, §17].

Following Brax's precedent rather than the reference's, `robot-sim` SHOULD define named fidelity tiers as a first-class, documented part of its configuration surface (exact names and per-tier knob settings are an implementation decision, tracked open in Appendix B, not fixed by this document) — for example, a cheap tier suited to early-training rollout volume, a tier matching this session's own validated CPU reference settings, and a tier enabling the costlier implicit/RK4/elliptic-cone/high-iteration settings for final acceptance-style evaluation. Whatever the eventual names, every tier MUST be expressible purely as a named bundle of the Section 12.1 knobs — no tier may introduce behavior unreachable through the individual knobs, so a caller can always start from a named tier and override one knob without losing the ability to reason about what changed.

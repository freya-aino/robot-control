# Step 09 — Variable fidelity as a first-class interface

[Index](README.md) · [Appendix A](appendix-a-citations.md) · [Appendix B](appendix-b-open-questions.md)

**Status**: Not started. No fidelity knob is queryable at run time today. The analytic fidelity harness (`fidelity.rs`, schema v7) runs seven checks at fixed tiers, including the 1 rad pendulum trajectory against the closed form, which is the nearest existing mechanism.

**Goal**: Make each fidelity knob (timestep, integrator, solver algorithm, iteration budget, `condim`, contact-slot budget, collision margin/gap) an explicit per-run parameter, and define named tiers as bundles of those knobs. Covers old §12 in full.

**Depends on**: [03](03-integrators.md) (integrator and timestep), [06](06-equality-constraints-and-solver.md) (solver and `condim`), [07](07-collision-pipeline.md) (margin and gap).

**MuJoCo baseline**: Tag `3.17.0`, commit `800a109a4bdd02cb60404d565db4ae16fb04a53b` (Appendix A, entries 3, 5, 12, 15). This step defines no algorithm and pins no engine function. Each knob's function pin is inherited from its owning step: integrator from [03](03-integrators.md) (`mj_Euler`, `mj_implicit`, `mj_RungeKutta`), solver from [06](06-equality-constraints-and-solver.md) (`mj_solNewton`, `mj_solCG`, `mj_solPGS`), and margin/gap from [07](07-collision-pipeline.md) (`mj_collision`). MuJoCo exposes these as unnamed knobs (entry 15), so the named tiers are this project's own design. Evidence tag: `unverified` for the tier mapping.

**Purity contract**: Tier definitions are plain data (a named bundle of knob values). They MUST NOT carry `Matrix`, `RigidPose` or `HashMap` (FR-6). Tier lookup, if any, is a `match`, not a map.

**Inputs/outputs**: Input: a tier name or explicit knob values. Output: a resolved configuration used by the backend, and the knob values recorded with each run.

**Done when**: Every knob in the §12.1 table is a queryable run parameter. Every named tier is a pure bundle of knobs. A caller can start from a tier and override one knob, and the resulting configuration is reported.

**Open questions**: Appendix B S-03 (exact tier names and settings, grounded in a CPU fidelity sweep) is blocking for the named-tier item of this step. The knob enumeration is not blocked.

---

## 12. Variable fidelity as a first-class interface

### 12.1 Enumerated knobs

The following MUST each be an explicit, per-run, queryable parameter of a simulation configuration — never a compile-time constant and never silently defaulted without the default being documented:

| Knob | Range of effect | Research basis |
| --- | --- | --- |
| Timestep | Smaller = more accurate, proportionally slower; too large relative to constraint stiffness = unstable (Section 5.3) | `[MuJoCo §15]` |
| Integrator | Euler-equivalent (cheapest, needs implicit damping per Section 4.3-4.4 to be usable at all) through RK4-equivalent (costliest, best long-horizon energy behavior for tumbling free bodies) | `[MuJoCo §5, §15]` |
| Solver algorithm | Newton-equivalent (few iterations, more per-iteration cost) vs. CG-equivalent (more iterations, cheaper each) | `[MuJoCo §3, §15]` |
| Solver iteration/tolerance budget | MuJoCo's own documentation endorses "often just one" Newton iteration as sufficient for RL-training fidelity specifically — i.e. this knob's acceptable range is *task-dependent*, not a universal accuracy dial | `[MJX §15]` |
| Contact dimensionality (`condim`) | 1 (frictionless) through 6 (+torsional +rolling); strictly more solver cost for strictly more physical realism | `[MuJoCo §3, §15]` |
| Contact-slot budget (GPU backend only) | Trades "contacts silently dropped under load" against memory/throughput (Section 11.2) | `[MJX §12, §15]` |
| Collision margin/gap | How early a near-contact is detected/reported vs. force-generating | `[MuJoCo §4, §15]` |

### 12.2 Named fidelity tiers (this project's addition)

MuJoCo itself exposes none of the above as named tiers — research confirmed only an unnamed, independently-tunable knob pile, with its own documentation reading as "here is a direction to turn each knob," not "here are three presets" [MuJoCo §15]. NVIDIA's Isaac Lab comes closer with a named, documented two-way solver choice (PGS vs. TGS) and a whole family of named, swappable solver *backends* (PhysX, MJWarp, Kamino, VBD, MPM), each with its own tuning guide [IsaacLab §15]. Brax's four pipelines (MJX, Generalized, Positional, Spring) are the closest found precedent for a single engine exposing an explicitly speed/fidelity-ordered set of named choices behind one interface [Brax §15, §17].

Following Brax's precedent rather than MuJoCo's, `robot-sim` SHOULD define named fidelity tiers as a first-class, documented part of its configuration surface (exact names and per-tier knob settings are an implementation decision, tracked open in Appendix B, not fixed by this document) — for example, a cheap tier suited to early-training rollout volume, a tier matching this session's own validated CPU reference settings, and a tier enabling the costlier implicit/RK4/elliptic-cone/high-iteration settings for final acceptance-style evaluation. Whatever the eventual names, every tier MUST be expressible purely as a named bundle of the Section 12.1 knobs — no tier may introduce behavior unreachable through the individual knobs, so a caller can always start from a named tier and override one knob without losing the ability to reason about what changed.

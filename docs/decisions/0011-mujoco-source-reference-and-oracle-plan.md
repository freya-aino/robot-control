# 0011. MuJoCo as a source reference: oracle plan, pure CPU core, CubeCL gate

Status: accepted for the source-reference decision, the oracle approach, the
pure-core rule and the gating of the GPU stage. The CubeCL GPU backend is not
accepted: it is gated by the Appendix B.6 qualification, which has not run.
No GPU dependency is added by this decision.

## Context

`crates/robot-sim/SIMULATION_SPECS.md` (263 lines) set the target of a
GPU-batched simulator "on par with MuJoCo" and named MuJoCo 3.17.0 at commit
`800a109a4bdd02cb60404d565db4ae16fb04a53b` as the research reference
(Appendix A). An earlier draft of step 04 proposed running MuJoCo in a
non-default `mujoco-baseline` devenv profile and committing its trace files
under `fixtures/robot-sim/mujoco/`, with the Rust gate replaying them. That
draft was superseded. It would have put a foreign runtime and generated data
into the repository, against `SPECS.md` Section 2 and ADR 0005, which keeps
MuJoCo as a design reference, not a dependency.

The user's direction is to transfer MuJoCo's simulation algorithms into
native Rust, check them with analytic oracles and conservation laws, and
port the same algorithms to a GPU backend later. MuJoCo must not enter the
repository in any form.

## Decision

1. **MuJoCo is a source reference, not a runtime.** Each ported function
   cites the pinned MuJoCo 3.17.0 file and function (commit `800a109a`).
   No MuJoCo binding, FFI, Python sidecar, trace file, fixture, profile or
   devenv package is added. An uncommitted `profiles.mujoco-baseline` block
   in `devenv.nix` was removed; `devenv.nix` matches the commit again.

2. **Fidelity is checked by an oracle plan, not by parity.** The checks are
   (a) analytic closed forms (pendulum), (b) conservation laws (energy,
   momentum), and (c) reference values derived by hand from MuJoCo's
   documented equations, with the derivation written in the test. Each
   tolerance is recorded with its scenario and tier. Every report describes
   the result as a **source-faithful port with oracle checks**. No report
   claims measured parity with MuJoCo, because no MuJoCo output is in the
   loop. The oracle plan and the tolerance table are in
   `crates/robot-sim/specs/04-mujoco-fidelity-loop.md`.

3. **Strict pure core.** Algorithm modules (`spatial`, `articulated`,
   integrators, passive forces, constraints, the oracle module) use `std`
   and plain `f64` arrays with explicit formulas. They do not use
   `robot_types::math::Matrix`, `RigidPose` or `HashMap`. Boundary adapters
   convert between `robot-model` types and plain arrays at the crate edge.
   The public `ArticulatedBody` API is kept.

4. **GPU stage: CubeCL, after qualification.** CubeCL is the primary
   candidate for the GPU backend, and `wgpu`/WGSL is the fallback. Neither
   is added before the Appendix B.6 qualification has run and recorded its
   result. If qualification fails, the GPU stage stops, or a new ADR selects
   `wgpu`/WGSL. The CPU core is the reference for CPU-vs-GPU agreement tests.

5. **Schema.** The fidelity report moves from schema v5 to schema v6 with
   per-check tolerance and sample-index reporting (ADR 0008). That change is
   part of the later oracle implementation step, not this decision.

## Consequences

- Passing oracle checks does not prove MuJoCo parity. Any document that
  states otherwise is wrong and must be corrected.
- Hand-derived reference values may be wrong. Each one needs a second
  derivation in its test before it is relied on.
- The G1 floating-base fall at `dt = 5e-4` (outside the ADR 0010 range) is a
  failing fidelity row once measured. It is not `#[ignore]`d.
- Any binding-based cross-engine comparison against MuJoCo itself stays
  under the `SPECS.md` Section 2.1 exception procedure. This decision does not
  request it.
- The specs under `crates/robot-sim/specs/` and `SIMULATION_SPECS.md` (now a
  pointer) are the implementation plan. This ADR does not change ADR 0005.

## Tests

No code changed in this decision. The validation for this step is the spec
review and the devenv check described in the ledger entry. The oracle tests
and the pure-core refactor are later steps of the plan.

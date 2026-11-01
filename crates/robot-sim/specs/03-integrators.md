# Step 03 — Integrators

[Index](README.md) · [Appendix A](appendix-a-citations.md)

**Status**: **Running, selectable per run; the implicit family is the gap.**
`World::step` integrates with whichever integrator `options::SimOptions` names,
so item 5 is met, and it is the stepper `robot sim multibody` runs — Appendix D
M-17 is closed.

Implemented: semi-implicit Euler (`mj_advance` order, velocity first then the
position update with the new velocity) and RK4 (`mj_RungeKutta`, N = 4). RK4
takes a `Free` joint, with the Munthe-Kaas correction
(`rk_correct_free_velocity`, `mju_derivExpmap`, the Bortz coefficients), which is
the configuration item 4 specifies it *for* — M-19 closed. `M(q)` now exists
from CRBA, so `implicit_damping_euler_step` is reachable — M-18 closed.

Under RK4 the constraint problem is solved at **every stage**, as
`mj_forwardSkip` is per stage in
[example 01](../../../docs/example-system-specs/01_pipeline_integrators_datamodel.md)
section 9.4. Solving once and reusing the force would quietly reduce the method
to first order, which is worth stating as specification: an integrator's order
is a claim about the whole right-hand side, and in contact the constraint force
is part of it.

**Not ported, and now the substantive gap**: `mj_implicit` / `mj_implicitSkip`
and the sparse `implicit`/`implicitfast` Jacobian path — non-diagonal
derivatives and the Coriolis derivative. Item 4 makes `implicitfast` a MUST
before any claim of "stable at a usable timestep on a real humanoid", and that
claim is not yet available.

Two things changed what this gap costs. A drive's damping is applied explicitly
rather than folded into the `qH` term of `mj_EulerSkip`
([05](05-passive-forces-and-actuators.md)), so a stiffer gain set than the G1's
declared one would need this path first. And the `dt = 5e-4` divergence this
step was partly written around no longer reproduces: the G1 holds a stance for
2 s at `dt = 1e-4` and at `dt = 5e-4` on the corrected path, with the contact
constraint rather than a penalty spring and the model's own drives rather than a
hand-written hold. The original divergence is therefore not attributable to the
integrator alone, and the step should stop implying it is.

**Goal**: Provide an integrator family (at least `implicitfast`-equivalent, with `implicit`-equivalent and RK4 as options) selected per run. Covers old §4, items 3–5. Items 1–2 are in step 02.

**Depends on**: [02](02-pure-dynamics-core.md).

**Reference formulation**: Tag `3.17.0`, commit `800a109a4bdd02cb60404d565db4ae16fb04a53b` (Appendix A, entry 5). Source file `src/engine/engine_forward.c`. The functions transferred in Step 5, with evidence tag `code_inspected` (bodies read as text at the pin):

- `mj_advance` (L1449): velocity update `qvel += h*qacc` first, then the position update with the new `qvel`. Ported as `semi_implicit_euler_step`.
- `mj_EulerSkip(m, d, skipfactor)` (L1492), damping branch: `qH = M`, `qH[i,i] += h*damp_deriv`, then the solve of `(M + h*B) * qacc = qfrc_smooth + qfrc_constraint`, then `mj_advance`. Ported as `implicit_damping_euler_step`. `mj_Euler` (L1578) is `mj_EulerSkip(m, d, 0)`, so the reference's default Euler is this semi-implicit step with implicit diagonal damping, not plain explicit Euler.
- `mj_RungeKutta` (L1605), N = 4: stage states `q + A*h*v` and `qd + A*h*F`, B-weighted final `dX`, then `mj_advance(m, d, dX + nv, dX)`. Ported as `rk4_step`.

Not transferred: `mj_implicit` (L1904, `mj_implicitSkip`) and the sparse `implicit`/`implicitfast` Jacobian path (non-diagonal derivatives, Coriolis derivative). Those are still open for this step.

Documentation: `doc/computation.rst` §Numerical integration (location only, not read).

**Purity contract**: Integrator algorithms use `std` and `[f64]` arrays with explicit formulas only. They MUST NOT use `robot_types::math::Matrix`, `RigidPose` or `HashMap` (FR-6). The implicit-in-velocity solve is written out explicitly.

**Inputs/outputs**: Input: state `(q, qd)`, timestep `dt`, the integrator choice, and passive-force derivatives (step 05). Output: the next state.

**Done when**: Each integrator option is selectable per run, the `dt = 5e-4`
divergence is reproduced as a measured result rather than hidden, and the
implicit variant has a documented damping-derivative path.

**Status against Done when**: selectable per run, yes. The `dt = 5e-4` row is
measured, and the measurement has moved: it no longer diverges, so what is
recorded is a passing stance rather than a failure, and the earlier failure is
attributed to the contact model and the actuation rather than to the integrator
(see Status). The implicit damping-derivative path exists for diagonal damping
only and is not wired to a drive's damping, so this clause is not met.

**Step 5 status (pure CPU subset)**: the ported functions have hand-derived
tests. The 1-DOF stiff-damping oracle shows the implicit step stays bounded
where the explicit step with the same damper diverges (`M = 1`, `B = 1000`,
`h = 0.1`: implicit `qd1 = 1/101`, explicit `qd1 = -99`). Input contract: `dt`
and every `damping` entry finite and non-negative; every `bias` and `mass` entry
finite; `M` symmetric, since only the lower triangle is read.

**Open questions**: none blocking. The integrator-per-tier mapping is owned by
[09](09-fidelity-tiers.md) (Appendix B S-03).

**Challenges**: [Appendix D](appendix-d-mathematical-challenges.md) confirms
`semi_implicit_euler_step`, `implicit_damping_euler_step`, `rk4_step` and
`cholesky_solve` all correct as written (section 1.1), including that RK4's
position update uses the `B`-weighted mean velocity.

**M-17, M-18 and M-19 are closed**: the ported functions are reachable and are
what runs, `M(q)` exists, and RK4 integrates a `Free` joint. The 1.9 per cent
energy gain Appendix D measured on an anisotropic spinning free body — the
quantity RK4 was wanted for — is now takeable with it, and is measured at more
than 100 times smaller under RK4 than under the semi-implicit step.

Nothing in this step is open as a defect. What remains is the unported implicit
family, which is a gap against item 4 and is stated in the Status above.

---

## 4. Dynamics: mass matrix, bias forces, integration

3. The semi-implicit Euler integrator this session built and validated (`MultibodyState::step`) MUST NOT remain the only integrator. the reference's own "simple" default integrator is not plain semi-implicit Euler — it already folds joint-damping derivatives into an implicit solve by default — and the reference simulator offers `implicit`/`implicitfast`/`discrete` variants specifically because stiff systems (small distal masses, large effective torque ratios — exactly what the real G1's wrists are) need more than explicit integration can stably provide at a usable timestep [the reference simulator §5]. This gives a precise, textbook name and mechanism for the dt-sensitivity this session found empirically (stable at `dt=1e-4`, diverging by ~1.3 simulated seconds at `dt=5e-4`): the fix is an implicit-in-velocity term, not merely a smaller `dt`.
4. `robot-sim` MUST implement, at minimum, an integrator equivalent to the reference's `implicitfast` (semi-implicit position update, implicit-in-velocity for joint damping and passive forces, dropping only the most expensive Coriolis/RNE-derivative term) before any claim of "stable at a usable timestep on a real humanoid" is made. A full `implicit`-equivalent (including RNE derivatives) and an RK4 option SHOULD exist for long-horizon energy-conservation-sensitive use (free-floating tumbling bodies specifically; the reference's own docs note `implicitfast`'s dropped gyroscopic term "mildly damps" a standalone spinning free body, which is now directly relevant once `Free` joints exist) [the reference simulator §5].
5. Integrator choice MUST be a per-run, queryable fidelity knob (Section 12), not a compile-time constant.

# Step 03 — Integrators

[Index](README.md) · [Appendix A](appendix-a-citations.md)

**Status**: **Running, selectable per run, and `implicitfast` has landed.**
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

**`implicitfast` is implemented** (`Integrator::ImplicitFast`,
`--integrator implicitfast`), which is item 4's MUST. It is `mj_implicitSkip`
with `flg_bias = 0` (example 01 section 9.2): the smooth forces stay at the old
velocity and their velocity derivative goes into the matrix, so the
unconstrained acceleration solves

`(M - h dF/dv) qacc = tau - qfrc_bias`

instead of `M qacc = tau - qfrc_bias`. For the forces this crate applies the
derivative **is diagonal**, and that is a property of the force set rather than
a simplification: the smooth forces are a drive's `kp (ctrl - q) - kd qd` and a
joint damper's `-kd qd`, only the damping terms depend on velocity, and each
depends on one degree of freedom's own velocity. So `qH = M + h diag(kd)`, which
adds to the sparse mass matrix's own diagonal and needs no new structure. A
fluid force or a tendon damper would make it non-diagonal (`mjd_passive_vel`
covers both) and neither is implemented.

Example 01 section 9.2 resets a standalone free body's six rows of `qH` to `M`
to keep the factorisation well defined; that happens here for free, because a
`Free` joint carries no actuator and so no `kd` is added to those rows at all.

**What it buys, and what it does not.** The property is that an explicit damper
with gain `kd` amplifies rather than removes velocity once
`kd > m_eff / (2 h)`, so an explicit step puts a **ceiling** on the damping a
model may declare. Measured on the G1 with its drive damping scaled up and its
stiffness zeroed, at `dt = 2 ms`: the explicit step drives the damped hinges to
`|qd| = 5.1e2` from a start of 0.5, while `implicitfast` keeps them at 0.74.
With the damping removed the two agree to 1e-12, which is the check that this is
the same dynamics and not a different model.

On the G1's **declared** gains it changes nothing measurable — a 2-second stance
gives a pelvis minimum of 0.7882 m under Euler and 0.7881 m under
`implicitfast`, and both PASS. That is the expected and correct outcome, worth
stating so the feature is not oversold: the layer's damping (0.05 to 5.3
N m s/rad) is far below the explicit ceiling, so the integrator removes a limit
rather than changing an answer. It is what makes a stiffer gain set usable, and
[05](05-passive-forces-and-actuators.md) no longer has to record a drive's
damping as explicit-only.

**Not ported**: full `implicit` — `mj_implicit`'s `qLU` path with the
recursive-Newton-Euler derivative (`mjd_smooth_vel` with `flg_bias = 1`) and the
unsymmetric LU solve — and `mjd_freeMhat`, the free-body gyroscopic block that
reinstates the bias derivative for a standalone free body and is what makes
`implicitfast` and `implicit` identical for one. Without it a tumbling free body
is integrated as `implicitfast` proper rather than as `implicit`, which is the
example's own behaviour for a non-free-body system and a divergence only for a
free one.

One correction to this step's own history: the `dt = 5e-4` divergence it was
partly written around no longer reproduces. The G1 holds a stance for 2 s at
`dt = 1e-4` and at `dt = 5e-4`, with the contact constraint rather than a
penalty spring and the model's own drives rather than a hand-written hold. The
original divergence is therefore not attributable to the integrator, and this
step should stop implying it is.

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

**Status against Done when**: all three clauses are met. Each integrator is
selectable per run (`--integrator euler|implicitfast|rk4`). The `dt = 5e-4` row
is measured and the measurement has moved — it no longer diverges, so what is
recorded is a passing stance, with the earlier failure attributed to the contact
model and the actuation rather than to the integrator. And the implicit variant
has a documented damping-derivative path that is wired to a drive's damping.

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

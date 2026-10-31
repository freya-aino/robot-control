# Step 03 — Integrators

[Index](README.md) · [Appendix A](appendix-a-citations.md)

**Status**: Partially implemented (Step 5). Pure ports exist in `robot_sim::integrators` for semi-implicit Euler (`mj_advance` order), the `mj_EulerSkip` damping branch (`implicit_damping_euler_step`, diagonal damping only, `None` on invalid input or a non-SPD system) and RK4 (`mj_RungeKutta`, N=4). `MultibodyState::step` in `robot_sim::articulated` is still the original stepper and is not wired to these functions. Full `implicit` and `implicitfast` (sparse `M`, non-diagonal derivatives) are not ported. The zero-torque floating G1 at `dt=5e-4` is recorded as a measured schema-v7 `[[probe]]` failure in `fidelity.rs` (`SingularJointInertia`, step 2236); it is not a passing analytic check or a claim of the reference simulator parity. The probe's attribution includes missing pelvis and torso collision geometry and does not isolate the integrator's contribution.

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

**Done when**: Each integrator option is selectable per run, the `dt = 5e-4` divergence is reproduced as a measured result rather than hidden, and the implicit variant has a documented damping-derivative path.

**Step 5 status (pure CPU subset)**: The three functions above are ported with hand-derived tests. The 1-DOF stiff-damping oracle shows the implicit step stays bounded where the explicit step with the same damper diverges (`M = 1`, `B = 1000`, `h = 0.1`: the implicit `qd1 = 1/101`, the explicit `qd1 = -99`). Input contract: `dt` and every `damping` entry must be finite and non-negative; every `bias` and `mass` entry must be finite; `M` must be symmetric (only the lower triangle is read, so an asymmetric `M` is refused and `None` is returned). Not yet wired into `MultibodyState::step`. The G1 `dt = 5e-4` row is measured and recorded as `fail` in the fidelity report, step 2236, `SingularJointInertia`.

**Open questions**: none blocking. The integrator-per-tier mapping is owned by [09](09-fidelity-tiers.md) (Appendix B S-03).

---

## 4. Dynamics: mass matrix, bias forces, integration

3. The semi-implicit Euler integrator this session built and validated (`MultibodyState::step`) MUST NOT remain the only integrator. the reference's own "simple" default integrator is not plain semi-implicit Euler — it already folds joint-damping derivatives into an implicit solve by default — and the reference simulator offers `implicit`/`implicitfast`/`discrete` variants specifically because stiff systems (small distal masses, large effective torque ratios — exactly what the real G1's wrists are) need more than explicit integration can stably provide at a usable timestep [the reference simulator §5]. This gives a precise, textbook name and mechanism for the dt-sensitivity this session found empirically (stable at `dt=1e-4`, diverging by ~1.3 simulated seconds at `dt=5e-4`): the fix is an implicit-in-velocity term, not merely a smaller `dt`.
4. `robot-sim` MUST implement, at minimum, an integrator equivalent to the reference's `implicitfast` (semi-implicit position update, implicit-in-velocity for joint damping and passive forces, dropping only the most expensive Coriolis/RNE-derivative term) before any claim of "stable at a usable timestep on a real humanoid" is made. A full `implicit`-equivalent (including RNE derivatives) and an RK4 option SHOULD exist for long-horizon energy-conservation-sensitive use (free-floating tumbling bodies specifically; the reference's own docs note `implicitfast`'s dropped gyroscopic term "mildly damps" a standalone spinning free body, which is now directly relevant once `Free` joints exist) [the reference simulator §5].
5. Integrator choice MUST be a per-run, queryable fidelity knob (Section 12), not a compile-time constant.

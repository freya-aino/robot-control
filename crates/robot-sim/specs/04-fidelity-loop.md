# Step 04 — Source-faithful oracle plan (fidelity loop)

[Index](README.md) · [Appendix A](appendix-a-citations.md) · [Appendix B](appendix-b-open-questions.md)

**Status**: Partially implemented in plan Step 4 and Step 5. `crates/robot-sim/src/oracle.rs` provides `pendulum_closed_form`, `energy_drift` and `compare_series` (std only, unit-tested). The archival battery in `crates/robot-sim/src/fidelity.rs` uses schema v7 (`shared-analytic-v7`): its seven analytic checks include the 1-rad pendulum trajectory; rows carry real sample indices; v5 and v6 reports are refused. Schema v7 adds the measured G1 zero-torque probe rows (section 2.6), which are recorded as `fail` when they fail and do not change `all_pass`. Scalar tolerances retain their original meanings, and the trajectory's pointwise absolute/relative tolerances are recorded by tier (section 2.5). No the reference simulator output, trace, binding or profile exists in this repository, and none is part of this spec.

**Numbering in this file.** "plan Step N" refers to the implementation plan (Steps 1 to 7). "spec NN" refers to the step file `NN-*.md` in this directory. Section numbers (1 to 5) refer to this file.

**Goal**: Define the oracle plan that checks the source-faithful ports (plan Steps 3 to 7) without running the reference simulator. The oracles are (a) closed forms, (b) conservation laws, and (c) reference values derived by hand from the reference's documented equations and the pinned source, with the derivation written in the test. This spec records the v5 battery as implemented, specifies the v6 oracle definitions, and gives the derivation rules. It does not claim measured parity with the reference simulator.

**Depends on**: [spec 02](02-pure-dynamics-core.md), [spec 03](03-integrators.md). The oracles check the dynamics and integrators pinned there.

**Reference formulation**: reference formulation at commit `800a109a4bdd02cb60404d565db4ae16fb04a53b` (Appendix A). the reference simulator is a source reference only. This spec defines no algorithm and pins no engine function. Function pins are inherited from [spec 02](02-pure-dynamics-core.md), [spec 03](03-integrators.md), [spec 05](05-passive-forces-and-actuators.md), [spec 06](06-equality-constraints-and-solver.md) and [spec 07](07-collision-pipeline.md). Evidence per `SPECS.md` section 3: `code_inspected` for the existence and location of those functions at the pin (Appendix A); specs 03 and 05 record `code_inspected` bodies for their ported functions. This spec itself inspected no function bodies; bodies not read remain `unverified` in specs 02, 06 and 07.

**Purity contract**: The oracle module (`crates/robot-sim/src/oracle.rs`, plan Step 4) uses `std` only, with plain `f64` slices. It contains no `Matrix`, `RigidPose` or `HashMap`, and no the reference simulator runtime, binding, trace reader or fixture.

**Inputs/outputs**: Input: the candidate series produced by the CPU port (and, after plan Step 7, the GPU backend), plus the closed-form or conservation reference computed inside the oracle. Output: an `OracleReport` for the pendulum trajectory (max error, index, pass), written as a v6 `fidelity.toml` check row with `measured`, `sample`, `tolerance_abs` and `tolerance_rel`. Other checks retain scalar `tolerance` fields.

## 1. Historical v5 battery (recorded for comparison with v6)

These are the historical v5 metrics, recorded so that v6 changes remain visible. All six continue in v6; the standalone `energy_drift` now measures the maximum over all samples, and each check reports the actual step/sample of its metric. v6 adds the pendulum trajectory row in section 2.5. Each check runs at both declared tiers: `reference` (100 µs) and `coarse` (1 ms). Both use semi-implicit Euler and the explicit articulated-body solver (`SOLVER_EXPLICIT_ABA`).

| Check (battery name) | System | Historical v5 metric | Unit | Horizon | Tolerance | Constant |
| --- | --- | --- | --- | --- | --- | --- |
| `pendulum_period` | standalone pendulum, `theta0 = 1 rad`, undamped | `\|(t2 − t1) − T\| / T` where `t1, t2` are two successive upward zero crossings (linear interpolation in the step) and `T = 4 K(k) / ω0`, `k = sin(θ0/2)`, `ω0 = sqrt(g/L)` | dimensionless (relative) | 4 s | 1e-4 | `PERIOD_TOLERANCE` |
| `pd_equilibrium` | standalone pendulum with PD and gravity compensation, start 1.5 rad, target 0.3 rad, `Kp = 20`, `Kd = 4` | `\|θ(T) − 0.3\|` at the final sample | rad (absolute) | 20 s | 1e-4 | `PD_TOLERANCE` |
| `aba_rne_closed_form` | point-mass pendulum (articulated), no time integration | Max over 16 samples of `\|τ_RNE − τ_closed\|` and `\|qdd_ABA − qdd_closed\|`, with `qd = 0.5`, `qdd = 1.2`, `q` spread over `[−π, π)`. Closed forms: `τ = m L² qdd − m g L cos q`, `qdd = τ/(m L²) + (g/L) cos q` | N·m for `τ`, rad/s² for `qdd` (mixed; one tolerance) | none (static) | 1e-9 | `ABA_RNE_TOLERANCE` |
| `energy_drift` | standalone pendulum, `theta0 = 1 rad`, undamped, unforced | `\|E(T) − E(0)\| / \|E(0)\|` at the **final sample only** | dimensionless (relative) | 10 s | 0.02 | `ENERGY_TOLERANCE` |
| `articulated_energy_drift` | two-link planar chain released from rest, no torque | `\|E(T) − E(0)\| / \|E(0)\|` at the **final sample only**; `E` = articulated kinetic energy + potential from forward kinematics | dimensionless (relative) | 10 s | 1e-2 | `ARTICULATED_ENERGY_TOLERANCE` |
| `free_fall` | free body, articulated, dropped from rest | `\|z(T) − z_exact\| / \|z_exact\|` at the final sample, `z_exact = −g t²/2`, `t = steps · dt`. Semi-implicit Euler gives relative error `dt / t` | dimensionless (relative) | 1 s | 1e-2 | `FREE_FALL_TOLERANCE` |

Rules that follow from the table:

- The battery pass rule is `measured.is_finite() && measured <= tolerance` (`Check::new`). It is a single-number test; it has no separate absolute and relative terms.
- Three v5 metrics looked at the final sample only. v6 `energy_drift` uses the guarded oracle's maximum over samples (and records its index); `articulated_energy_drift` and `free_fall` still look at the final step. A divergence that recovers by the end is not seen by those final-step metrics.
- `aba_rne_closed_form` is a closed-form comparison, not the RNE-to-ABA round trip defined in section 2. Its two terms are in different units under one tolerance; a v6 change would split them.
- The battery's `energy_drift` check now uses the guarded `energy_drift` in `oracle.rs` (section 2.2), and reports its worst-step index.
- The G1 floating-base ground-contact path is not in this battery. ADR 0010 qualifies the timestep range `[5e-5, 4e-4]` s for the 2 s stance. The `reference` tier (100 µs) is inside that range, so its G1 contact status is `not_measured`. The `coarse` tier (1 ms) is outside it, so its status is `not_applicable`. No status here says that G1 contact is qualified.
- Known failing row: the G1 floating-base fall at `dt = 5e-4` lies above the validated range. It is measured by the probe in section 2.6 and reported as a `fail` row with its step and error kind. It is not `#[ignore]`d and not removed, and the battery's `all_pass` is unaffected.

## 2. Oracle definitions (v6, specified here, implemented in plan Step 4)

Each oracle is a pure function of plain `f64` data. The reference is never read from a simulator.

### 2.1 Oracles and their metrics

| Oracle | Reference | Metric and comparison | Used by |
| --- | --- | --- | --- |
| Pendulum closed form | Exact `theta(t)` (section 2.3) | `compare_series` against the simulated `theta` series, sample by sample | plan Step 4 (standalone pendulum); articulated port pending |
| Energy drift | `E(0)` of the conservative system | max over samples of `\|E(t_i) − E(0)\| / \|E(0)\|`, subject to the guard in section 2.2 | plan Step 4 (standalone pendulum); additional systems pending |
| Momentum conservation | Initial linear and angular momentum of a free body with no external force | max over samples of `\|p(t_i) − p(0)\|` (kg·m/s) and of `\|L(t_i) − L(0)\|` (kg·m²/s), reported separately | pending |
| ABA/RNE round trip | `tau = RNE(q, qd, qdd)`, then `qdd' = ABA(q, qd, tau)` | max over samples of `\|qdd' − qdd\|` (rad/s²) | pending future work; not a v6 battery row |
| Hand-derived value | Number derived by hand from a stated the reference simulator equation (section 3) | `compare_series` or scalar comparison with the tolerance in section 2.4 | plan Steps 5, 6, 7 |

### 2.2 Zero and near-zero energy

- If `E(0) = 0` exactly, `energy_drift` returns NaN, not a named error (a deviation from the original wording). `compare_series` treats NaN as a failure, so it never reports a pass.
- No `E_floor` constant was added in Step 4. Only exact zero and non-finite inputs are guarded. A near-zero `E(0)` is not guarded and gives a large relative drift. The absolute fallback row is not set.
- The battery's standalone `energy_drift` check uses this oracle; its `E(0)` is nonzero at its amplitude.
- Test reference: `energy_drift_guards_only_exact_zero_and_non_finite_energy` in `oracle.rs` checks the policy: `E(0) = 1e-300` gives its true drift, `E(0) = 5e-324` with no change gives 0, a negative `E(0)` uses `|E(0)|`, and a non-finite energy gives NaN.

### 2.3 Exact pendulum reference

- The period in section 1 is a scalar; it is not a trajectory. The trajectory reference is `theta(t)`.
- Exact form (point-mass pendulum, undamped, released from rest at `theta0`): `theta(t) = 2 · asin( k · sn(K(k) − ω0 · t | k) )`, with `k = sin(theta0/2)`, `k' = cos(theta0/2)`, `ω0 = sqrt(g/L)`, and `sn` the Jacobi elliptic function of modulus `k`. The code passes `k'` (not `m = k²`) to `sn` and `K`, so that the near-separatrix limit is not formed from `sqrt(1 − k²)`. Checks: at `t = 0`, `sn(K) = 1` gives `theta = theta0`; at `t = K/ω0`, `sn(0) = 0` gives `theta = 0`; for `k → 0` it tends to `theta0 · cos(ω0 t)`.
- `oracle.rs` provides `pendulum_period` (AGM) and `pendulum_closed_form` using the Landen (AGM) descending recursion of Abramowitz and Stegun 16.4. Tests check identities at `t = 0`, `K` and `2K`, a hand-derived power series of `K`, an independent RK4 integration over one period at 1 rad, and an independent RK4 integration at 3.0 rad (tolerance 1e-6 over 4000 steps). The domain is `|theta0| < pi`; outside it the function returns NaN.
- Precision near the separatrix. With gap `g = pi − |theta0|`, `k' ≈ g/2` and `K ≈ ln(4/k')`. Forming `k'` as `sqrt(1 − k²)` loses every digit once `g` is near `2e-8`, so `k'` is taken from `cos(theta0/2)`. The rounding of `theta0` then gives an absolute error of about `eps / g` in `K`. The period asymptote `4K ≈ 4(L + (k'²/4)(L − 1))`, with `L = ln(4/k')`, is checked for `g` in {1e-3, 1e-6, 1e-9} against a tolerance of `1e-9 + 16·eps/g`. The largest `f64` below `pi` is admissible and returns a finite period and the release angle at `t = 0`. No input inside the domain is rejected for its distance from `pi`.
- Restricted linear regime (allowed only for small amplitude): `theta(t) = theta0 · cos(ω0 t)`. Its relative period error is `theta0² / 16` (from `T ≈ T0 (1 + theta0² / 16)`). For a relative period error within `1e-4`, `theta0 <= 0.04 rad`. The trajectory error at time `t` is about `theta0 · (theta0²/16) · ω0 · t` from accumulated phase drift. Plan Step 4 must check this against the horizon and the chosen tolerance before using the linear form. It is never used for the 1 rad battery amplitude.

### 2.4 Tolerances and the abs+rel rule

- For `compare_series(reference, candidate, tol)` with `tol = { abs, rel }`: a sample `i` passes when `|candidate_i − reference_i| <= abs + rel · |reference_i|`. The series passes when every sample passes. `max_error` is `max_i |candidate_i − reference_i|`, and `at_sample` is the index where it occurs. This general oracle API permits relative tolerance, but every v6 serialized series row (built-in or plug-in) requires `rel = 0`: a row storing only the maximum raw error and its index cannot reconstruct whether every sample passed a relative, reference-dependent bound.
- The battery's single-number checks in section 1 keep their existing rule (`measured <= tolerance`). They do not use `abs` and `rel`.
- `run_checks` rejects a series tolerance unless `rel` is exactly `0.0` and `abs` is finite and non-negative, returning `BatteryError::CheckFailed`; this also rejects non-finite relative values. A plug-in cannot serialize a v6 row that violates this abs-only contract. A `FidelityCheck` that returns no `series_tolerance` remains a scalar check.
- A length mismatch between `reference` and `candidate` returns a named error (variant name fixed in plan Step 4). It never truncates silently.

### 2.5 Tolerances for the new oracles (status after plan Step 4)

Step 4 runs the pendulum trajectory oracle at both declared tiers. Its v6 check row records `tolerance_abs`, `tolerance_rel`, the maximum error in radians and its sample index. Measured maxima for the current semi-implicit Euler path over four seconds are 1.5018532e-4 rad at sample 37435 (100 µs) and 1.5040877e-3 rad at sample 3744 (1 ms). The bounds allow roughly 3.3 times those errors and scale with the first-order global error: tenfold larger timestep receives a tenfold larger absolute allowance. These are tests of the Rust integrator against an analytic oracle, not evidence of the reference simulator parity. The trajectory uses `rel = 0`, as every v6 series row must, because only its maximum raw error and index are persisted; the runner requires exactly zero relative tolerance for both built-in and plug-in series checks. Scalar checks keep `tolerance` and their previous semantics. The reader rejects noncanonical tier tolerances and contradictory pass flags. A row marked `pending` has no battery tolerance yet.

| Scenario | Tier | Oracle | Metric and unit | Tolerance | Status |
| --- | --- | --- | --- | --- | --- |
| Pendulum trajectory `theta(t)`, 1 rad release, 4 s (section 2.3) | `reference` (100 µs) | `compare_series` | max abs error (rad), index 0 to 40000 | abs = 5e-4 rad, rel = 0 | v6 battery row |
| Pendulum trajectory `theta(t)`, 1 rad release, 4 s (section 2.3) | `coarse` (1 ms) | `compare_series` | max abs error (rad), index 0 to 4000 | abs = 5e-3 rad, rel = 0 | v6 battery row |
| Free-body momentum (no external force) | `reference`, `coarse` | momentum conservation | max abs deviation (kg·m/s, kg·m²/s) | pending | not implemented in Step 4 |
| Free-body spin energy bound | `reference`, `coarse` | energy drift | relative, final and max over samples | pending | `energy_drift` implemented in `oracle.rs`; no battery row yet |
| ABA/RNE round trip | `reference`, `coarse` | round trip | max abs `\|qdd' − qdd\|` (rad/s²) | pending | not implemented in Step 4 |
| Hand-derived values | per test | hand-derived | per test | set in each test | rule in section 3; these are test assertions, not battery rows |
| G1 zero-torque fall, floating base, sphere ground, `dt = 5e-4` s (section 2.6) | one fixed scenario (no tier) | measured probe: `SingularJointInertia` at a step, or pass | first failing step (recorded window 2100 to 2400, 4000-step cap) and error kind | none; recorded `fail` row | measured probe row, not a battery check |

### 2.6 G1 zero-torque probe (schema v7)

- Scenario: the floating G1 (`robot_model::fixtures::unitree_g1_29dof_floating_base`), zero applied torque, sphere ground with the `robot-cli` `FLOATING_GROUND` constants, start pose with the pelvis 5 mm above the first contact height, `dt = 5e-4` s, at most 4000 steps. The scenario runner is `fidelity.rs` (`g1_zero_torque_probe`, which calls the private `g1_zero_torque_fall`). The algorithm module `articulated.rs` holds only the array-based dynamics. The row's `scenario` key is `g1-zero-torque-dt-5e-4`. It is one fixed scenario, not a per-tier run, so the report does not imply the coarse tier was measured.
- Setup failures (fixture load, no contact sphere, a dynamics error during setup) are `ProbeError` values and fail the battery. They are never recorded as a `pass` row. A non-finite `q` or `qd` during the fall is `NotFinite`, not `SingularJointInertia`.
- Measured result: the first failure is `SingularJointInertia` at step `2236` (about 1.12 s). The test asserts a failing step inside the window 2100 to 2400, so a regression that fails at step 1 is caught.
- Reporting: one `[[probe]]` row with keys `scenario`, `system`, `status`, `step`, `error`, `reason`. `status = "fail"` carries `step` and an `error` that names a `DynamicsError` variant, with `step` below 4000. `status = "pass"` carries neither. `reason` must equal the canonical text (`G1_PROBE_REASON`).
- Reader contract (ADR 0008 section 8): the strict reader validates the probe row structurally and never simulates. `reproduce(max_steps)` re-runs the probe, counts its 4000 steps against `max_steps` before running, and refuses a stored row that differs from the fresh measurement. The row is not a battery check and does not change `all_pass`.
- Attribution: the failure is not attributed to the integrator alone. The pelvis and torso have no collision geometry, so once the legs fold the trunk reaches the plane through unmodelled shapes (ADR 0010). The row records this model-coverage limit; the integrator's share is not isolated here.
- This is a measured probe, not parity with the reference simulator.

## 3. Derivation rules for hand-derived values

1. A hand-derived reference value MUST come from a closed form, or from a the reference simulator equation whose function, file and commit (`800a109a`) are named.
2. The derivation MUST be written as comments in the test, one step per line, and the final number MUST be reproduced to at least the tolerance precision.
3. Each value MUST be cross-checked by a second derivation in the same test (for example, the closed form and a numerical integral of the same equation). If they disagree, the test fails and the derivation is corrected. The test is never relaxed.
4. A hand-derived value MUST NOT be taken from a simulator run of this code. A value produced by the port under test is never its own reference.
5. Evidence state of a hand-derived value (`SPECS.md` section 3): it is `unverified` with respect to the reference simulator behaviour, because no the reference simulator output is compared. Before the second derivation in rule 3 agrees, the value MUST NOT be used as a test reference. After agreement it MAY be used as a test reference, which is a test-internal check of the port. That use licenses no implementation claim about the reference simulator, and agreement of two derivations does not promote any the reference simulator claim.
6. Evidence state of a pinned the reference simulator function: `code_inspected` only for what was read at the pin (existence, location, and what the function does). Per `SPECS.md` section 3, `code_inspected` does not license statements about numerical correctness. Functions whose bodies have not been read stay `unverified`, as in Appendix A.

## 4. Evidence and reporting

- Every result is described as a **source-faithful port with oracle checks**. No report, ADR, ledger entry or test name claims measured parity with the reference simulator.
- Passing a tolerance establishes that the oracle check passes within that tolerance. It does not establish parity with the reference simulator.
- A passing oracle does not promote a claim about the reference simulator behaviour. A the reference simulator citation does not promote a claim about this code (see [spec 12](12-cpu-gpu-agreement.md)).
- Each step that adds an oracle records in the evidence ledger which functions are source-faithful and which oracles they passed.

## 5. Scenarios and refresh

- Scenario list: the six historical checks in section 1, plus the v6 pendulum trajectory and pending rows in section 2.5, plus the hand-derived references of plan Steps 5 to 7. Each scenario names its system (the `CHECK_SYSTEMS` labels).
- Refresh: changing a scenario, horizon, metric or tolerance requires a new `BATTERY` identifier, a new `SCHEMA_VERSION`, and an ADR 0008 update. The reader rejects reports with an old identifier.

## Open questions

- Resolved in Step 4: no `E_floor` constant was added (section 2.2). Only exact zero and non-finite `E(0)` are guarded, and they return NaN. The standalone pendulum energy metric changed to maximum drift; the articulated energy and free-fall final-step metrics remain. Battery tolerances for the `pending` rows of section 2.5 remain for later steps.
- Resolved in Step 4: `sn` is the Landen recursion in `oracle.rs`, checked against an RK4 integration and a hand-derived power series of `K(m)`. The linear regime of section 2.3 was not used and has no tolerance.
- No cross-engine comparison against the reference simulator itself is requested here. Any such comparison is subject to the `SPECS.md` Section 2.1 exception procedure.

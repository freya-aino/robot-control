# 0010. Ground contact for the floating-base multibody path: penalty sphere model

Status: accepted for the contact law, the integration, the stance-damping rule
and the tests. The floating-base G1 **stance** is a supported-stance
qualification with a stated dt range. It is not a balance claim, and the
zero-torque collapse is recorded as a known limitation of the sphere-only model.

## Context

`robot_sim::articulated` had no ground contact, so the floating-base G1
(`robot sim multibody --floating`) fell through the floor. PLAN.md Phase 3 and
SPECS.md Sections 5, 7 and 14 require that a change to contact or integration
is recorded in an ADR, that numerical claims carry their tolerances, and that
the physical limits are stated.

## Decision

1. **Contact law (`robot_sim::contact::GroundContact`).** For a contact sphere
   of radius `r` centred at `c`, penetration is `p = ground_height + r - c.z`.
   The normal force is `N = max(0, k*p + d*v_closing)`, with stiffness `k` and
   damping `d`. Friction is regularised Coulomb,
   `F_t = -mu * N * v_t / max(|v_t|, eps)`. It is bounded by `mu*N`, and below
   the regularisation speed `eps` it acts as a viscous stick of slope `mu*N/eps`.
2. **Shapes.** Only `CollisionGeometry::Sphere` shapes are modelled. Box,
   cylinder and mesh shapes are counted (`unmodelled_collision_shape_count`)
   and do not touch the ground. A rigid box is modelled as four corner spheres
   on one free body (see the box test). Box and mesh shapes need a new ADR.
3. **Integration.** Contact enters the articulated-body bias force in pass 1
   of `articulated_accelerations` as `pA - f_ext`, each body in its own frame.
   With no ground the result is exactly the previous `forward_dynamics`.
   `MultibodyState::step_with_ground` uses the same semi-implicit Euler as
   `step`.
4. **Validation.** `DynamicsError::InvalidGroundContact` rejects non-finite or
   non-physical parameters (`stiffness > 0`, `eps > 0`, `damping >= 0`,
   `friction >= 0`). `MultibodyEnv` reports this as `physics_failure`.
5. **Stance-damping stability rule (`ArticulatedBody::stable_damping`).**
   The stance uses `kd_j = min(kd_max, 0.5 / ((M^-1)_jj * dt))`, with
   `kd_max = 30` and `(M^-1)_jj` from `inverse_inertia_diagonal`. This caps the
   explicit damping update factor `1 - kd*dt*(M^-1)_jj` at 0.5 or above.
   Without the cap, the light G1 wrist joints can get `kd*dt*(M^-1)_jj` far
   above 2. The cap is computed once at the start pose.
6. **Stance hold (qualification only).** `ArticulatedBody::stance_torques`
   writes `tau_j = clamp(-kp*q_j - kd_j*qd_j, ±effort_j)` for each single-DOF
   joint, using each joint's own `JointLimits::effort` from the URDF,
   with `kp = 300 N·m/rad`. Every `Fixed` and `Free` slot of `tau` is written
   as zero, so the caller's buffer does not need to be pre-zeroed. The hold has
   no pelvis or centre-of-mass feedback. It is **not balance control**.
7. **Verdict (`robot sim multibody --floating`).** The run is judged only over
   the qualification duration of 2.0 s. Otherwise it is INCONCLUSIVE and exits 3.
   Within 2 s, the run passes only if all three hold:
   - pelvis minimum ≥ start height − 0.02 m;
   - lowest contact-sphere bottom ≥ −0.01 m;
   - kinetic energy after 0.5 s ≤ 0.5 J (and a sample exists).
   The floating default `--steps` is `ceil(2.0 / dt)`, so a bare `--floating`
   run is at least 2 s. The fixed-base default stays at 2000 steps. A pelvis
   below the plane is reported as FAIL with exit 1. The verdict is in
   `judge_stance`, and it has unit tests for each branch.
8. **Argument and step-count validation (`robot-cli`).** A present `--dt` or
   `--steps` must parse; a missing value, a malformed value, or a value that is
   not finite and greater than zero is a usage error (exit 64) before any
   simulation runs. `--steps` must be between 1 and `MAX_STEPS` = 10 000 000.
   `dt` must be representable as a nanosecond `Duration` to within 0.1 %, so
   `1e-12` is rejected. The floating default is computed with checked
   arithmetic: a count above `MAX_STEPS` is an error, not a saturated `u32`.
   Duration is printed to microsecond precision with the step count.

## G1 contact geometry (verified against the URDF)

Contact spheres are on the two `*_ankle_roll_link` bodies only, 4 per foot,
radius 0.005 m. Centres in the ankle_roll link frame:

- `(-0.05, ±0.025, -0.03)` and `(0.12, ±0.03, -0.03)`, rounded from
  `(-0.05 0.025 -0.03)` and the matching right-side entries.
- The foot spans x in about [-0.05, 0.12] and y in about ±0.03 m, a support
  polygon, not a point.

`pelvis` has no collision shape. The shoulders have 4 cylinders. All other
links are mesh shapes. Mesh files are not vendored. The CLI reports these
counts as `8 sphere, 28 not modelled`.

## Tolerances and analytic checks

| Test | Expected | Tolerance |
|---|---|---|
| Sphere settles (`a_dropped_sphere_settles_at_the_penalty_rest_height`) | `r - m*g/k` | 1e-6 m; vertical speed < 1e-6 m/s |
| Tilted 4-corner box (`a_tilted_corner_box_settles_flat_at_the_analytic_rest_height`) | centre `r + h - m*g/(4k)`, flat, no residual motion | 1e-5 m, 1e-5 rad, 1e-4 rates |
| Frictionless slide at penalty equilibrium (`frictionless_sliding_at_penalty_equilibrium_conserves_total_energy`) | total energy `½mv² + mgz + ½k·pen²` | relative 1e-9, 5000 steps at dt = 1e-4 |
| Friction (`friction_dissipates_sliding_energy_and_never_reverses_it`) | slide speed non-increasing to 1e-12, never below −1e-6 m/s, 20 000 steps | final speed < 1e-3 m/s |
| `GroundContact::force` | zero when separated; normal clamped to 0 in fast separation; `\|F_t\| <= mu*N`; `is_valid` rejects bad parameters | exact / 1e-12 |
| G1 stance, 2 s, dt = 1e-4 (`the_g1_stance_hold_keeps_the_pelvis_above_the_plane_for_two_seconds`) | pelvis drop ≤ 0.02 m; final KE ≤ 0.5 J | as stated |
| G1 zero-torque characterization (`zero_torque_g1_collapse_is_a_known_limitation_of_the_sphere_only_model`) | pelvis drops more than 0.02 m below start | known-limitation test |
| Grounded `MultibodyEnv` stance (`the_floating_g1_with_ground_is_stepped_through_the_contact_path`) | pelvis drop ≤ 0.02 m over 2000 steps through the trait | as stated |

## Stability and the measured dt range

Measured with the default stance (`kp = 300`, `kd_max = 30` with the cap above,
`k = 2e4`, `d = 200`, `mu = 0.8`, `eps = 1e-3`), 2.0 s each, release build:

| dt (s) | steps | pelvis minimum (m) | lowest contact (m) | max KE after 0.5 s (J) | Verdict |
|---|---|---|---|---|---|
| 5e-5 | 40 000 | 0.7874 (t=0.059 s) | −0.0050 | 0.077 | PASS |
| 1e-4 | 20 000 | 0.7873 (t=0.060 s) | −0.0052 | 0.089 | PASS |
| 2e-4 | 10 000 | 0.7872 (t=0.060 s) | −0.0053 | 0.097 | PASS |
| 4e-4 | 5 000 | 0.7873 (t=1.621 s) | −0.0054 | 0.116 | PASS |
| 8e-4 | 2 500 | 0.7821 (t=2.000 s) | −0.0051 | 8.29 | FAIL (settled energy bound), exit 1 |
| 1e-3 | 2 000 | 0.7545 (t=2.000 s) | −0.0099 | 21.06 | FAIL (pelvis drop and settled energy), exit 1 |

All rows are the release-build output of `robot sim multibody --floating --dt D`,
this session. At 4e-4 the pelvis minimum occurs late (t = 1.621 s), so the
stance is still moving there, and its settled energy is larger than at 1e-4.
The 8e-4 and 1e-3 rows show the drift growing with dt, and the verdict catches
it through the measured checks, not a dt guard.

Long run at dt = 1e-4 over 40 s (`--steps 400000`, measured this session):
kinetic energy 0.0044 J at 4 s, 0.0006 J at 8 s, then 0.0002 J from 12 s to
40 s. The pelvis holds at 0.7889 m from 8 s on. The energy does not grow after
settling.

**Validated range: dt ∈ [5e-5, 4e-4] s for the 2 s qualification, default
1e-4.** Values above 4e-4 are not validated. dt = 4e-4 passes the verdict but
is at the edge of the validated range, as the late minimum shows. The 40 s
long run was measured only at dt = 1e-4; it was not run at other dt values.

Two observations about the mechanism are **not instrumented**:
- The earlier instability (energy growing at dt = 1e-4 over 40 s, collapse at
  dt = 2e-4, from the pre-cap `kd = 30` stance) is consistent with the explicit
  damping on the light wrist joints being above the stability limit. The
  pre-cap run was not re-instrumented, and which joints the cap binds on was
  not recorded. The post-cap sweep above is the measured result.
- The dt = 8e-4 failure is consistent with the explicit spring term
  `kp*dt²*(M^-1)_jj` approaching 1 on light joints. This was not measured.

The regularised friction stick (`mu*N/eps`) is not separately bounded here.
The stance results are measured, not a proof of stability for other
parameters.

## Measured qualification (`robot sim multibody --floating`, release build)

- **Stance hold** (`--dt 0.0001`, default steps, 2.0 s, about 1.9 s wall time):
  pelvis starts at 0.7969 m. Minimum 0.7873 m at t = 0.060 s, a drop of 9.6 mm.
  Ends at 0.7883 m. Lowest contact-sphere bottom −0.0052 m. Max KE after 0.5 s
  0.089 J. Exit 0 with `PASS (supported stance within tolerances; no balance claim)`.
- **Zero torque** (`--zero-torque`, same duration): pelvis minimum −0.7708 m at
  t = 0.807 s. Lowest contact −0.0420 m. KE after 0.5 s up to 330 J. Exit 1 with
  `FAIL (pelvis fell below the ground plane)`.
- **Short run** (`--steps 1000 --dt 1e-4`, 0.1 s): INCONCLUSIVE, exit 3. It is not
  a pass. The settled-energy line reports `n/a` because no sample exists before 0.5 s.
- Wall time: the 2 s stance run (`--steps 20000 --dt 1e-4`) takes about 1.9 s in
  release; the 40 s run took about 34 s in the independent review (not re-timed here).

## Limitations

- **Not balance.** The stance holds the posture toward zero with clamped
  torques. The verdict measures the supported stance on 8 sphere shapes, nothing
  more.
- **Zero-torque collapse is a known limitation, not physics.** The pelvis and
  torso have no collision geometry, and 28 shapes are unmodelled. The trunk
  reaches the plane through those missing shapes. Adding contact proxies is
  expected to change this result. The characterization test is named and
  documented that way.
- Penalty contact penetrates by about `N/k` at rest for a free body. The
  stance feet sink about 5 mm. The cause is not separated: the stance torques
  may add load through the legs, but no sweep isolates that.
- Starting embedded in the penalty floor is unsupported. The CLI derives the
  start height from the geometry.
- No joint-limit penalty, no actuator model, and joint damping only through the
  stance `kd`. Zero torque lets the legs fold.
- No self-collision, no box, cylinder or mesh contact, no friction model beyond
  regularised Coulomb, and no energy-conservation guarantee through impacts.
- Not validated against a reference simulator.

## Consequences

- `ArticulatedBody` gains `contact_sphere_count`, `unmodelled_collision_shape_count`,
  `lowest_contact_height`, `forward_dynamics_with_ground`, `inverse_inertia_diagonal`,
  `stable_damping`, `stance_torques` (per-DOF `kd`), and `MultibodyState::step_with_ground`.
  Existing methods are unchanged, and the fixed-base path is unchanged.
- `MultibodyEnv` gains an opt-in `with_ground`. Without it, the free path is
  unchanged. The backend name is `cpu-multibody-ground` when ground is attached.
- `robot sim multibody` gains `--zero-torque`. The floating verdict is INCONCLUSIVE
  (exit 3) for runs shorter than 2 s, and FAIL (exit 1) otherwise on a failed check.
- The workspace `Cargo.toml` sets `[profile.test.package.robot-sim] opt-level = 2`
  to keep the 20 000-step G1 tests quicker. No test is weakened.

## Touches

- `crates/robot-sim/src/contact.rs`
- `crates/robot-sim/src/articulated.rs`
- `crates/robot-sim/src/spatial.rs` (`Sub for Wrench`)
- `crates/robot-sim/src/multibody_env.rs`
- `crates/robot-cli/src/commands/sim.rs`, `crates/robot-cli/src/main.rs`
- `Cargo.toml` (test profile override)
- `docs/decisions/0010-ground-contact-penalty-model.md`

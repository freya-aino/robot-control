# 0008. Fidelity report schema and battery contract

- Status: accepted; format selected before the writer; schema v5 (`scope`
  key, per-tier `[[contact]]` applicability rows, battery identifier
  `shared-analytic-v5`, `BatteryError::CheckFailed`) amended 2026-10-28.
  Schema v4 (per-check `system` attribution, `FidelityCheck` runner) and v3
  (`production_pass`, dt-independent free-fall tolerance, articulated energy
  drift, step budget) are the baselines.
- Date: 2026-10-28 (v5 amendment).
- Scope: the archival battery is an **analytic subset**. It contains no
  ground-contact check, so no report qualifies the floating-base G1 contact path.
  The sphere/box contact check is deferred until it can be measured inside the
  ADR 0010 range.
- Touches: `crates/robot-sim/src/fidelity.rs`, `crates/robot-sim/tests/fidelity_extension.rs`,
  `crates/robot-sim/Cargo.toml`, `crates/robot-sim/README.md`, `PLAN.md`,
  `docs/decisions/0008-fidelity-report-schema.md`.

The TOML format was selected before the writer in `robot_sim::fidelity` was
implemented, as PLAN.md Section 2 item 15 requires. Version 2 superseded the
ambiguous version 1 reader contract. Version 3 added the articulated energy-drift
check and replaced the dt-scaled free-fall tolerance. Version 4 adds the
`system` label on every check row, validated strictly by the reader against the
canonical system of each check name, and a `FidelityCheck` trait that the battery
runner iterates over. Version 5 adds the top-level `scope = "analytic-subset"`
key, a `[[contact]]` table that states the G1 contact applicability of every
tier from its timestep alone, and `BatteryError::CheckFailed` for plug-in
failures. Versions 1 through 4 are rejected rather than silently interpreted as
a complete production report.

## Context

PLAN.md Phase 3 asks for a fidelity harness in `robot-sim`: a `Check` (name,
tolerance, measured value, pass or fail), a `Tier` (timestep, integrator, solver),
and a runner that writes a report through `robot-artifacts`. SPECS.md Sections 5, 7
and 14 require that numerical claims carry their tolerances, that manifests are TOML,
and that artifacts are immutable. The GPU-only checks (`SIMULATION_SPECS.md`
Section 14.2 CPU-vs-GPU agreement, Section 14.4 contact-budget exhaustion) cannot run
until a GPU backend exists and must still appear in the report.

## Decision

1. **Format: TOML through `robot_artifacts::toml`.** The report is written with
   `toml::to_string` and stored with `robot_artifacts::store::write_immutable`. The
   writer is deterministic: no timestamps, no host paths, and floats use the
   shortest round-trip representation. The same battery run twice gives identical
   bytes. A second write of different bytes to the same path is an error.
2. **File name:** `fidelity.toml` in the directory the caller provides.
3. **Top-level keys:**
   - `schema = "robot-sim.fidelity-report"` (string, fixed)
   - `schema_version = 5` (integer)
   - `scope = "analytic-subset"` (string, fixed). The report covers the analytic
     battery only. It is not a G1 or floating-base qualification; any other value
     is rejected.
   - `battery = "shared-analytic-v5"` (string). The suffix versions the battery
     contents (check set, check definitions, horizons, canonical tolerances). It
     changes together with `schema_version` whenever those change, so the
     identifier always names what was measured. Earlier battery identifiers
     (v1–v4) are rejected, not reinterpreted.
   - `tier_set = "declared"` or `"experimental"` (string). `declared` means
     **exactly** `declared_tiers()` in order, including each name, timestep,
     integrator and solver. `experimental` means a nonempty custom tier list
     that is not the declared list; it is never interpreted as a production
     tier result, even when all its checks pass. `run_battery` selects the mode
     by exact tier equality. Both modes use the same check battery and tolerance
     rules. Neither mode permits relabeling the other's tier set.
   - `all_pass = <bool>`: true only if every applicable check in every tier passed.
     Not-applicable entries do not count; for experimental reports this is
     **only an experimental result**, not a production qualification.
   - **Production qualification is `FidelityReport::production_pass()`**, defined as
     `tier_set == declared && all_pass()`. It is the only predicate that may gate
     production use. `all_pass()` alone does not consider `tier_set` and must not
     be used for qualification.
4. **Tier table, one `[[tier]]` per declared tier:** `name` (string),
   `timestep_ns` (integer, exact), `integrator` (string), `solver` (string).
5. **Check table, one flat `[[check]]` per (tier, check) pair:** `tier` (string
   naming a `[[tier]]`), `name`, `system` (string, the physical system the check
   exercised: `standalone_pendulum`, `point_mass_pendulum_articulated`,
   `double_pendulum_articulated` or `free_body_articulated`; the reader rejects any
   value that differs from the canonical system of `name`), `tolerance` (float, same
   units as the measured value and stated in the check's doc), `measured` (float),
   `pass` (bool). A
   non-finite measured value is written as `nan` or `inf` and always fails. A flat
   array is used rather than nested arrays of tables so the subset parser and writer
   stay simple.
5a. **Contact table, one `[[contact]]` per declared tier (v5).** Fields: `tier`,
   `system` (always `g1_floating_base_ground_contact`), `status` and `reason`.
   `status` is `not_applicable` or `not_measured`; it is never a qualified value.
   The status is derived only from the tier timestep against the ADR 0010 validated
   G1 stance range `[5e-5, 4e-4]` s (`[50_000, 400_000]` ns). A timestep outside
   that range is `not_applicable`, for example the 1 ms coarse tier. A timestep
   inside it is `not_measured`, because the analytic battery has no G1 contact
   check. The reader recomputes the expected rows from the tiers and rejects any
   mismatch, so a report cannot claim G1 contact at any tier.
6. **Not-applicable table, one `[[not_applicable]]` per declared check that cannot
   run:** `name` and `reason`. No `measured` or `pass` is written for these, so a
   reader cannot mistake them for a pass.
7. **Pass rule:** `pass = measured.is_finite() && measured <= tolerance`. Every
   check's measured value is a non-negative error metric, so a single comparison
   applies to all checks.
8. **Reader:** `read_report` validates the structure of the stored report only.
   It reads at most 1 MiB plus one byte and rejects larger files before parsing;
   no timestep or report contents can trigger simulation during an archival read.
   It rejects malformed TOML, unknown keys at every table level, mismatched
   schema/version/battery/tier-set/scope, missing or changed production tiers,
   absent, extra or out-of-order battery checks for any tier, wrong scope,
   [[contact]] rows that differ from the canonical applicability,
   noncanonical tolerances (the tolerance for each check name is fixed;
   see `canonical_tolerance`), contradictory `pass` or `all_pass`, and
   absent or changed not-applicable entries. It returns no partially loaded
   report. The reader **does not re-run simulation**: stored measurements are
   archival claims, not bit-for-bit outputs to recompute with a later solver,
   toolchain or libm. This version has no source-commit/toolchain/platform
   provenance; callers must not claim cross-platform bit-identical replay.
   `FidelityReport::reproduce(max_steps)` is an explicit opt-in same-build,
   same-platform comparison; it validates the report and refuses any run whose
   summed integration steps exceed the caller's cap. Its failure does not make
   a structurally sound archival report unreadable.
   `write_report_with_id` returns `robot_artifacts::hash::ContentId` for the
   exact stored bytes; `read_report_verified` checks against an independently
   retained expected ID before parsing. A digest stored only alongside an
   editable report is **not** authenticity/provenance or tamper-proof evidence.
9. **Battery and tiers.** The battery is the same set of checks for every declared
   tier:
   - `pendulum_period`: relative error of the measured period against the exact
     elliptic period `T = 4 K(sin(theta0/2)) / omega0` (K evaluated by the AGM),
     with theta0 = 1 rad, the period taken between two upward zero crossings with
     linear interpolation, tolerance 1e-4.
   - `pd_equilibrium`: `|theta_end - theta_target|` for a gravity-compensated PD
     pendulum after 20 s, tolerance 1e-4 rad.
   - `aba_rne_closed_form`: max absolute error of the articulated point-mass pendulum
     forward and inverse dynamics against the textbook closed form, tolerance 1e-9.
     This check does not integrate, so its value does not depend on the tier.
   - `energy_drift`: relative energy drift of the undamped pendulum over 10 s,
     tolerance 0.02.
   - `articulated_energy_drift`: relative mechanical-energy drift of a two-link
     planar revolute chain (point masses 1.0 and 0.6 kg, lengths 0.5 and 0.35 m,
     initial angles 0.6 and -0.3 rad, released from rest, no torque) over 10 s.
     Kinetic energy is `ArticulatedBody::kinetic_energy`; potential energy is
     `-sum(m g . p_com)` from forward kinematics of the same profile. Tolerance
     1e-2. A divergent integration (a dynamics error) is recorded as a non-finite
     measurement and therefore fails.
   - `free_fall`: relative error `|z_end - z_exact| / |z_exact|` for a free
     articulated body over 1 s, where `z_exact = -g t^2 / 2` and `t` is the
     simulated horizon (a whole number of steps). Tolerance 1e-2, independent of
     `dt`. The semi-implicit integrator's first-order offset is `g t dt / 2`
     against the exact drop `g t^2 / 2`, so the relative error is exactly `dt / t`:
     1e-4 at 100 us and 1e-3 at 1 ms. The check therefore passes up to
     dt = 10 ms over the 1 s horizon and fails above it (2e-2 at 20 ms, 0.2 at
     200 ms, 1.0 for a 3 s step).
   - Declared tiers: `reference` (dt = 100 us) and `coarse` (dt = 1 ms). Both use
     `integrator = "semi-implicit-euler"` and `solver = "explicit-articulated-body"`.
   - **Step budget.** `run_battery` sums the integration steps of every simulated
     check over every tier (`ceil(horizon / dt)` per check) and refuses the call
     with `BatteryError::StepBudget` if the total exceeds `MAX_BATTERY_STEPS`
     (2 000 000). The check runs after tier validation and before any
     simulation, so a pathological tier (for example dt = 1 ns, about 4.5e10
     steps) fails immediately instead of running for hours. The declared tiers
     need 495 000 steps. `run_battery_with_limit` takes an explicit limit, and
     `reproduce(max_steps)` uses it with the caller's cap.
 10. **Not applicable (recorded, not run):** `cpu_gpu_agreement`
    (`SIMULATION_SPECS.md` Section 14.2) and `contact_budget_exhaustion`
    (Section 14.4). Each has a `[[not_applicable]]` entry naming the missing GPU
    backend.

## Production qualification scope

`production_pass()` qualifies the analytic subset only: the declared tier set ran
and all six analytic checks passed. It does not qualify floating-base G1 ground
contact at any tier. The `contact` rows are the statement of that scope, and a
`coarse` `all_pass = true` must not be read as contact evidence.

## Check plug-in contract

`FidelityCheck` is the extension point for standalone numerical checks. Its
`steps(tier)` value is the check's upper bound on integration steps; both
`run_checks` and the archival battery add those values with checked `u128`
arithmetic before calling `measure`. `run_checks` and default `run_battery` use
the 2 000 000-step limit; `run_battery_with_limit` uses its explicitly supplied
limit. `run_checks` also validates the tier's positive timestep, integrator and
solver before measuring. A custom `measure` result of `Some(value)` yields a result row;
`None` means only that this standalone invocation skips that check. Plug-ins report
their own failures as `BatteryError::CheckFailed { name, message }`; the enum is
`#[non_exhaustive]`. It does not
create a `[[not_applicable]]` report entry. Returning `None` from a built-in
check is an error, because the archival battery requires every check at every
tier. The static `not_applicable_checks()` list is the source of archived N/A
rows (currently GPU-only checks).

Custom results returned by `run_checks` are not themselves archival reports,
and the report reader rejects check names outside the canonical battery. To add
a check to archival reports, the check must be added to the battery registry
and the canonical name/system metadata, its tolerance and definition must be
documented here, and both `SCHEMA_VERSION` and `BATTERY` must be bumped. The
built-in registry consistency test keeps its ordered names, physical systems
and tolerances aligned with the archival reader's canonical tables.

## Consequences

- Tolerances are stated in the report next to each measured value. A check cannot
  pass without a measured value.
- Schema changes require a new `schema_version` and an explicit decision record
  or amendment (as with the v2 tier-set change and the v3 check change recorded here).
- Changing a canonical tolerance or a check's physical definition is a schema
  change, not a parameter tweak.
- The writer validates the same structure and one-MiB size limit before creating
  a report file; invalid production tiers cannot be emitted as valid archives.
- The writer and reader live in `robot-sim`, which gains a dependency on
  `robot-artifacts`. `robot-artifacts` depends only on `robot-types`, so no cycle
  exists.
- No new third-party crates are added (ADR 0003).
- The battery re-creates the analytic setups of the existing pendulum and
  articulated tests (same masses, lengths and start angles) and measures them
  through the tier's timestep. It does not assert the existing test results.
  The articulated builders (`double_pendulum`, `free_body` and helpers) are
  currently a copy of the `articulated.rs` test fixtures in `fidelity.rs`, not a
  shared module; consolidating them into one `pub(crate)` fixture module is an
  open follow-up.

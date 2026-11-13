# motion-validate

Validation of a retargeted robot reference against its human source, its
robot and a tracking simulation, per SPECS.md §8.2 items 8 and 9 and §8.3.
Depends on `robot-types`, `robot-model`, `robot-sim` (inverse dynamics and
the tracking world), `motion-process` (human forward kinematics),
`motion-retarget` (the map the reference was made with) and
`robot-artifacts` (content IDs and the manifest's TOML).

## What's here

| Module | Contents |
| --- | --- |
| `trajectory` | A `RobotReference` record as arrays, with central-difference velocity, acceleration and jerk, and the base's body-frame twist and its rate |
| `kinematic` | Per-frame normalized endpoint error (against the map's own targets), root drift, planted-foot slip |
| `dynamics` | Floating-base RNE inverse dynamics through `robot-sim`, the base wrench distributed least-norm over the planted feet, joint torques after distribution, unsupported base force with no contact; position, velocity and torque-speed envelope checks |
| `contact` | Contact consistency: labelled-but-lifted and unlabelled-penetrating feet, unsupported base force, pulling contacts, contacts outside the friction cone |
| `tracking` | Simulated tracking in `robot-sim`'s `World`: declared drives, or an inertia-scaled PD with optional inverse-dynamics feedforward; fixed or floating base on a ground plane; per-frame error, fall detection, per-task-label results |
| `report` | `ValidationReport` (kinematic, dynamics, summaries, contact, tracking as separate fields), `AcceptanceThresholds`, `Verdict`, `RejectionReason` with stable codes |
| `validate` | `validate`: the whole of §8.2 item 8 in one call |
| `package` | `reference_id` (content ID of a canonical text form), `manifest` (the `motion-curation/1` TOML: accepted and rejected references, rejections with codes, frames and messages) |

## Decisions worth knowing

**No single pass/fail field.** §8.3: "A validation report must separate
kinematic error, dynamics violations, contact consistency and task
success... No universal fidelity percentage is a release criterion." The
verdict comes from `ValidationReport::evaluate` against thresholds the
caller supplies, and a rejection names every violating frame.

**Torques need the contacts.** Inverse dynamics alone gives the torques
for a robot bolted at the pelvis. The base wrench the motion needs has to
come from the ground, so it is distributed over the feet the reference
labels as planted. The split is least-norm, with moments weighted by
`1 / moment_length²` against forces. Tests check that a standing G1's
estimated foot forces sum to its weight to 1e-6. A hovering reference with
no contact needs its full weight from nowhere, and a ballistic one needs
no force at all.

**The distribution is an estimate.** It does not enforce unilaterality or
friction. `contact` checks the estimate against both, and a two-foot
violation is reported as found, not as proof that no split works.

**Tracking measures one controller.** A reference this PD (or the model's
drives) tracks is not thereby feasible for every controller. One it fails
is a reference to look at, not proof that no policy could track it.
Results are per task label, so a clip's jab can pass while its hook fails.

**PD gains are bandwidths.** Fixed N m/rad gains under an explicit step
are unstable on a light wrist and soft on a hip. The PD acts in
acceleration units scaled by each joint's effective inertia
`1 / (M⁻¹)ⱼⱼ`, so a bandwidth and damping ratio mean the same on every
joint.

**Rejected examples are kept.** The curation manifest lists rejected
references with every reason's code, frame (−1 when not frame-specific)
and message, so a threshold change can revisit them.

## Limits

- The torque-speed check uses each actuator's envelope through its
  transmission. A joint with no actuator gets its torque reported but no
  envelope check.
- Centre-of-pressure limits are not checked: the profile does not describe
  a foot's support polygon.
- Acceleration and jerk are reported as summaries. No limit is declared for
  them, so the caller decides what is too much.

## Testing

`cargo test -p motion-validate`: 17 tests. They cover:
- the G1's standing weight carried exactly by its feet;
- hover flagged and free fall not;
- the torque envelope;
- standing consistent, a lifted labelled foot flagged, a hover flagged, and
  sideways acceleration on one foot leaving the friction cone;
- a bolted G1 elbow swing tracked to under 0.01 rad RMS with feedforward,
  and an impossible 20 Hz swing not tracked;
- a floating G1 standing 400 steps without falling;
- verdicts per frame and per skill, with missing tracking refused when
  required, and limit excess against the declared range;
- reference IDs that are stable and sensitive to every sample, and the
  manifest round-tripping with codes;
- a retargeted standing reach validated end to end and accepted.

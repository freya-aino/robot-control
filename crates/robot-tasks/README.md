# robot-tasks

The tasks of SPECS.md §6's `robot-tasks` row (tracking, balance, reach,
footwork, boxing and perturbation) with §9.2's reward coverage,
termination/truncation distinction, privileged-observation audit and
calibrated domain randomization. Depends on `robot-types`, `robot-model`
and `robot-sim`.

## What's here

| Module | Contents |
| --- | --- |
| `reward` | `RewardRegistry`, `RewardTermSpec`: a name-keyed, build-time, function-pointer registry over flat observations (§9.3's "no control-path lookup") |
| `balance` | `BalanceTask` on `robot-sim::env::PendulumEnv`: upright, alive (+1 per non-falling step; without it the task paid the agent to fall, see `robot-learn`'s README) and effort terms |
| `humanoid` | `HumanoidState` (base pose and velocities, joints, torques, tracked links, foot and forbidden contacts), `StateExtractor` from `robot-sim`'s `World` through the profile's own kinematics, `ObservationSpec` with `actor_audit`, `Status` (`Terminated` vs `Truncated`), `HumanoidRegistry`, the `HumanoidTask` trait, `step_task` |
| `terms` | Shared terms: upright, height, power, action rate, joint speed, undesirable contacts, base drift |
| `tasks` | `StandTask`, `PerturbationTask` (push schedule, recovery after a window), `TrackingTask` (interpolated reference, terminated when lost, truncated at the end), `ReachTask`, `FootworkTask` (velocity command, landing reward), `BoxingTask` (pad strikes scored once per entry above a speed, guard) |
| `env` | `HumanoidEnv`: the world stepped under the model's declared drives at a control rate (`substeps` physics steps per held action), observed through a task, as `robot-sim`'s `Environment`; the outcome channel (`outcome_task_fn`) that hands the registry's reward and the task's status to `robot-learn`'s task closure; `g1_stand`, the reference G1 wiring at the measured rollout-tier settings (`implicitfast`, 1 ms, 5 substeps) |
| `randomize` | `Randomization`: ranges with a calibration source, held-out sub-intervals never drawn in training, sensitivity ranking |

`robot-sim`'s `WorldEnv` gains `push_base`, the impulsive base velocity
change `PerturbationTask` schedules.

## Decisions worth knowing

**Tasks read a robot-agnostic state.** Rewards and terminations are
written against `HumanoidState`, not simulator vectors. The same code
scores a simulated step, a recorded episode or hardware telemetry. A G1
test extracts a standing state from the real world and checks height,
foot contact and a push.

**Privileged inputs are declared and audited.** §9.2: "exported actors
must pass an observation audit proving no privileged inputs remain."
Base linear velocity and height, which an IMU does not measure, are marked
privileged in every task. `actor_audit` names any privileged entry an
actor reads.

**Termination is not truncation.** A fall or lost tracking is
`Terminated(reason)` (zero bootstrap). A time limit or the end of a
reference is `Truncated`.

**Boxing scoring is separate from the reward.** Row A-21 asks for scoring
"validated independently". `BoxingTask::strikes` and `advance` hold the
scoring rule (inside the pad, approaching faster than a minimum, once per
entry), tested on their own. The reward reads the scorer's count.

**Held-out means never trained on.** A parameter's held-out interval is
cut out of its training distribution, not merely drawn with other seeds.

## Why rewards are never a bare float

SPECS.md §9.2: "Log reward terms independently to expose reward
exploitation. Persistent torque violations cannot be hidden by a good
aggregate return." `RewardRegistry::evaluate` always returns the typed
`robot_types::transition::RewardTerms`, which carries every term's name,
value and weight; `.total()` is derived from that, never stored
independently, so there is no path where an aggregate can silently disagree
with its own terms.

## A termination bug the test suite caught

The first version of `BalanceTask`'s fall check was backwards: it measured
distance from the *downward* vertical and failed only near the bottom,
which meant a pendulum hanging sideways — clearly fallen — was never
flagged as terminal. Rerunning the already-written tests surfaced the
mismatch (`falling_past_the_configured_angle_terminates` disagreed with the
fixed `fall_angle` semantics once that doc comment was corrected to measure
distance from *upright*), and the fix re-expresses `fall_angle` as "angular
distance from upright, beyond which the episode fails" — matching what a
balance task's config should mean in the first place.

## Testing

`cargo test -p robot-tasks`: 32 tests (including the alive bonus). The
`env` module adds four: the G1 standing through the environment under its
declared drives for 25 control steps at the rollout-tier physics settings,
seed-deterministic resets that genuinely differ across seeds, the outcome
channel terminating with zero reward on a missing outcome or a physics
failure, and the malformed constructions refused by name. 11 are the original ones: the registry's per-term reporting
and duplicate-name rejection, the upright term's periodicity and maximum at
`theta = pi`, and every termination path (fall, step-limit truncation,
physics failure, and the corrected "within the fall angle, continues" case).
The new ones cover:
- the observation audit, tilt and yaw;
- shared terms at rest and power counting negative work;
- termination and truncation;
- observation width matching the spec;
- reach success;
- tracking interpolation, loss and end;
- footwork velocity and landing;
- strike scoring once, and only when fast;
- recovery after the window;
- training never drawing held-out values;
- refused uncalibrated or degenerate ranges;
- sensitivity ranking;
- the G1 world integration.

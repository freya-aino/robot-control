# robot-tasks

Tracking, balance and reference tasks with a versioned reward registry, per
SPECS.md §9.2 and the reward-term extension point of §9.3. Depends on
`robot-types`, `robot-model` (not yet used — reserved for a profile-aware
task beyond the pendulum fixture) and `robot-sim`.

## What's here

| Module | Contents |
| --- | --- |
| `reward` | `RewardRegistry`, `RewardTermSpec`: a name-keyed, build-time, function-pointer registry — §9.3's "no control-path lookup" needs no dynamic dispatch for a fixed, linked-in term set |
| `balance` | `BalanceTask`: upright + effort reward terms and termination/truncation logic on top of `robot-sim::env::PendulumEnv` |

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

`cargo test -p robot-tasks` — 11 tests: the registry's per-term reporting
and duplicate-name rejection, the upright term's periodicity and maximum at
`theta = pi`, and every termination path (fall, step-limit truncation,
physics failure, and the corrected "within the fall angle, continues" case).

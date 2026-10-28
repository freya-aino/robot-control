# robot-learn

Rollouts, authored backpropagation, the PPO baseline and the bounded
on-policy buffer of SPECS.md §7.6 and §9.2. Depends on `robot-types`,
`robot-nn`, `robot-sim`, `robot-artifacts` (and `robot-tasks` as a
dev-dependency, for the test suite only).

## What's here

| Module | Contents |
| --- | --- |
| `backprop` | Manual backward pass through a `robot_nn::mlp::Mlp` (the authored autodiff `docs/decisions/0003-std-only-default-closure.md` describes) |
| `optimizer` | Adam, with its moment state (`AdamState`) exposed as an owned value so it can be checkpointed |
| `gae` | Generalized advantage estimation, taking a correct per-step bootstrap value — see the bug below |
| `gaussian_policy` | A diagonal Gaussian policy: a mean network plus a state-independent log-std vector |
| `rollout` | `RolloutBuffer`: a bounded on-policy store over `robot_types::transition::Transition` itself, per §7.6 |
| `ppo` | `PpoAgent`: ties everything above into one trainable agent against any `robot_sim::env::Environment` |

## A real bug `gae.rs`'s signature now prevents

The first version of `compute_gae` took one scalar `bootstrap_value` for the
segment's last step and inferred every other step's bootstrap as
`values[t + 1]`. That is wrong whenever a rollout chunk holds more than one
episode — which SPECS.md §7.6 explicitly permits — because the step right
after a truncated episode is the *first step of an unrelated, freshly reset
episode*, not a continuation of the one that just ended. `compute_gae` now
takes an explicit `next_values: &[f64]`, one entry per step, each evaluated
directly from that step's own recorded `next_observation`. The bug and its
fix are documented in `gae.rs`'s own doc comment, and
`a_truncation_in_the_middle_of_a_buffer_bootstraps_from_its_own_next_value_not_a_neighbor`
is the regression test that would have caught it.

## Why the pendulum training test doesn't claim convergence

`ppo::tests::training_pipeline_runs_without_diverging` runs the full
pipeline — environment, task, rollout buffer, GAE, backprop, Adam, for both
networks — for twenty update cycles and checks that nothing diverges
(non-finite parameters, generation counter, buffer capacity). It does
**not** assert that training beats a deterministic zero-action baseline.
An earlier version of this test did, found and motivated the GAE fix above,
and still failed to show reliable improvement afterward across several
reset-range, fall-angle and iteration-count settings. Rather than keep
adjusting hyperparameters until one seed happened to pass, the test now
reports both numbers via `eprintln!` (visible under `--nocapture`) and
states the situation plainly in its own doc comment: stabilizing an
inverted pendulum with a co-trained value function is a harder tuning
problem than this baseline's defaults currently solve in a small compute
budget, and that is recorded as a negative result rather than hidden.
`ppo_drives_a_one_step_bandit_policy_mean_toward_the_target` is the
decisive test that the gradient machinery itself — clipped surrogate,
entropy bonus, Adam, backprop — is correct: it isolates the policy-gradient
update from every difficulty specific to multi-step credit assignment and
converges cleanly.

This is SPECS.md §3's evidence discipline applied to this crate's own test
suite: "A failed verification is a result: retain a correction record
instead of preserving an unsupported claim."

## Checkpointing

SPECS.md §9.2 requires "checkpoint/resume including optimizer, RNG and
normalizer state." `AdamState` exposes its moment estimates and step count
as plain owned data for exactly this reason; wiring that into
`robot-artifacts`' chunk/blob format for an actual save/load round trip is
the next piece of this crate's own backlog, not yet implemented here.

## Parameter groups

`robot_nn::mlp::ParamGroup { name, range, frozen }` names a flat slice of a
network's parameters (layer weights, then biases, layer by layer). `PpoAgent`
keeps one table per network (`set_policy_groups`, `set_value_groups`). The
policy table indexes the mean network's parameters followed by the log-std
vector, so `log_std` can be frozen like any other slice. Each table is checked
against its network before it is stored: names must be non-empty and unique,
ranges non-empty, inside the parameter count, and non-overlapping. A frozen
group keeps its weights, first moment and second moment unchanged on every
update, so momentum cannot move it; frozen `log_std` entries are skipped bit
for bit.

## Testing

`cargo test -p robot-learn` — 24 tests: every gradient (backprop through
`Mlp`, the Gaussian policy's log-probability with respect to both mean and
log-std) checked against central finite differences; Adam checked against
a convex quadratic; GAE checked against a hand-computed trajectory, the
truncation/termination distinction, and the cross-episode bootstrap case
above; the rollout buffer's capacity-exhaustion and schema-rejection paths;
and the two PPO-level tests described above.

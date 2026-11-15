# robot-learn

Rollouts, authored backpropagation, the PPO baseline and the bounded
on-policy buffer of SPECS.md §7.6 and §9.2. Depends on `robot-types`,
`robot-nn`, `robot-sim`, `robot-artifacts` (and `robot-tasks` as a
dev-dependency, for the test suite only).

## What's here

| Module | Contents |
| --- | --- |
| `backprop` | Manual backward pass through a `robot_nn::mlp::Mlp` (authored against `std` under the policy of `docs/decisions/0003-std-only-default-closure.md`; Burn and CubeCL remain unqualified for training) |
| `optimizer` | Adam, with its moment state (`AdamState`) exposed as an owned value so it can be checkpointed |
| `gae` | Generalized advantage estimation, taking a correct per-step bootstrap value — see the bug below |
| `gaussian_policy` | A diagonal Gaussian policy: a mean network plus a state-independent log-std vector |
| `checkpoint` | Save and resume for `PpoAgent` and `Rng`: manifest-based directory layout, atomic write, schema and digest-verified chunks (ADR 0007). |
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

## The pendulum negative result, and its correction

`ppo::tests::training_pipeline_runs_without_diverging` runs the full
pipeline for twenty update cycles and checks only that nothing diverges.
An earlier version asserted improvement over a zero-action baseline and
could not show it across reset ranges, fall angles and iteration counts.
That was recorded as a negative result rather than tuned away.

The cause was the task, not PPO. `robot-tasks`' `BalanceTask` paid only
negative per-step rewards (`-distance²`), and a fall ended the episode
with zero future value. Near the fall angle, falling was therefore worth
more to the critic than staying up imperfectly: the task paid the agent to
fall. Measured before the fix, at 100 updates of 2000 steps over three
seeds, the deterministic return was −9.1 to −9.4 against −9.7 for zero
action, the same within noise.

`BalanceTask` now has an `alive-v1` term: +1 per step that does not end in
a fall, withheld on the falling step. With it, PPO at learning rate 3e-3,
initial log-std 0, and 150 updates of 2000 steps holds the pendulum up.
Deterministic returns over 20 held-out resets are 90.8, 96.2 and 97.1 on
seeds 1, 2 and 3, against 34.2 for zero action, out of a possible ~100.
These were measured with this crate's test at the current revision in the
devenv toolchain. `ppo_stabilizes_the_pendulum_on_two_seeds` asserts it
for seeds 1 and 2 (above 85, and above baseline + 40); it is the evidence
SPECS.md §14 row A-13 asks for ("PPO solves small reference tasks across
multiple seeds"). `ppo_drives_a_one_step_bandit_policy_mean_toward_the_target`
still isolates the gradient machinery.

This is SPECS.md §3 applied to this crate's own history: the failed
verification was kept as a record, and its correction is recorded with
the measurement that supports it.

## The humanoid training path exists, and what that does and does not claim

`tests/humanoid_training_path.rs` runs the whole humanoid seam end to end:
the vendored G1 in `robot-sim`'s world under its own declared drives,
wrapped by `robot-tasks`' `HumanoidEnv` (`robot_tasks::env::g1_stand`, the
rollout-tier physics settings), rewarded by the stand task's registry
through the outcome channel, collected by `collect_rollout` and optimized
by `update`. The test asserts plumbing facts only: every control step
yields a schema-valid transition with a finite registry reward, widths
match the 29-drive action space, and the update genuinely moves the policy
parameters.

**Nothing humanoid is trained.** Two smoke-scale iterations verify the
seams; a policy that stands is a compute-scale run (hours of CPU at the
rollout tier, or the GPU-batched backend `SIMULATION_SPECS.md` targets)
that this workspace has not performed. The claim this section makes is
narrower and new: the path from the vendored robot to a PPO update is
built, tested, and runs.

## Checkpointing

SPECS.md §9.2 requires "checkpoint/resume including optimizer, RNG and
normalizer state." A checkpoint is a directory: one chunk file per piece of
state (networks, log-std, both Adam moment sets, normalizer, config, counters,
RNG, parameter groups) plus a `manifest.toml` written last as the commit point.
The current schema is version 2 (ADR 0007); version-1 checkpoints are rejected.

- `save(dir, &agent, &rng)` records the source revision as `"unknown"`.
- `save_with_metadata(dir, &agent, &rng, Some(revision))` records a caller-supplied revision. An empty revision is rejected before anything is written.
- `load(dir)` returns `(PpoAgent, Rng)`. `load_with_metadata(dir)` also returns a `CheckpointMetadata` with the recorded `source_revision`. Manifests written before the field existed report `"unknown"`.

Loading is all-or-nothing. Every chunk is checked against its manifest digest and schema, and the policy and value topologies are checked against the observation width and the scalar value output. Parameter-group tables are checked against their networks. The agent is built only after all checks pass. `save` refuses to overwrite an existing manifest. The generation counter lives in the `counters` chunk, not the manifest.

## Parameter groups

`robot_nn::mlp::ParamGroup { name, range, frozen }` names a flat slice of a
network's parameters (layer weights, then biases, layer by layer). `PpoAgent`
keeps one table per network (`set_policy_groups`, `set_value_groups`). The policy table indexes the mean network's parameters followed by the log-std vector, so `log_std` can be frozen like any other slice. Each
table is checked against its network before it is stored: names must be
non-empty and unique, ranges non-empty, inside the parameter count, and
non-overlapping. A frozen group keeps its weights, first moment and second
moment unchanged on every update, so momentum cannot move it; frozen `log_std`
entries are skipped bit for bit. Tables are stored in the `param-groups` chunk,
checked again on save, and re-validated on load.

## Testing

`cargo test -p robot-learn`: 62 tests (one of them, `child_process_reads_checkpoint`, is a helper that only does work when spawned by `checkpoint_is_loaded_by_a_separate_process`). They cover: every gradient (backprop through `Mlp`, the Gaussian policy's log-probability with respect to both mean and log-std) checked against central finite differences; Adam checked against a convex quadratic; GAE checked against a hand-computed trajectory, the truncation/termination distinction, and the cross-episode bootstrap case above; the rollout buffer's capacity-exhaustion and schema-rejection paths; the checkpoint round trip, the resumed-training bit-equivalence test, the separate-process load, and the rejection tests for schema version, missing chunk, swapped digest, corrupt chunk, truncated manifest and topology mismatch; parameter-group round trip and out-of-bounds rejection on load; a PPO update in which a frozen policy group stays bit-identical while unfrozen parameters change; invalid group tables leaving the agent unchanged; and the two PPO-level tests described above. `GaussianPolicy`'s deterministic trait action and allocation-free sampled action are checked against their existing reference paths. `robot-nn` adds mask tests for `Mlp::frozen_mask`.

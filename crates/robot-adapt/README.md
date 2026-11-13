# robot-adapt

Limited-data adaptation, per SPECS.md §11.3: every row of its method
table that the specification permits, the table's binding order enforced
as code, and the obligations every method carries. Depends on
`robot-types`, `robot-model`, `robot-nn`, `robot-artifacts`, and
`robot-learn` behind the default `training` feature.

## What's here

| Module | §11.3 | Contents |
| --- | --- | --- |
| `ladder` | ordering rule, obligation 7 | `AdaptationLadder`: a method may be scheduled only after every method above it has an evidence-backed *insufficient* measurement on the same robot. Simulation-guided fine-tuning also needs a `HardwareUpdateDecision`. Multi-simulator randomization is refused because §9.1 grants no second simulator |
| `sysid` | row 1 | Ridge least squares with singular values and per-parameter standard errors; `identify_joint` (inertia, viscous, Coulomb, offset) |
| `history` | obligations 2, 3 | `HistoryWindow`: declared deployable observation names (privileged names refused), fixed length, `Short`/`Stale`/`Discontinuous`/`Ready` |
| `encoder` | rows 2, 6 | `ContextEncoder`: the declared default context when the window is not ready, and a count of inputs outside the training envelope. `fit`/`train` (feature `training`) |
| `correction` | row 3 | `CorrectionModel` fitted only on `AlignedTransition`s (real and simulated step from the same start), with held-out error with and without correction |
| `residual` | row 4, obligation 4 | `ResidualBound`: per-dimension authority and rate limits enforced outside the residual, NaN treated as zero, saturation counted per dimension |
| `finetune` | rows 5, 7, obligation 5 | `fine_tune` (feature `training`): only unfrozen `ParamGroup`s train. The outcome always carries the frozen baseline (the rollback artifact) and is accepted only if held-out loss beats it. `GuidedObjective` for the simulation-derived term |
| `retrieval` | row 8 | `retrieve`: an admission filter is required and runs before ranking. Refusals are counted |
| `morphology` | row 9 | A profile-only morphology descriptor, and `TransferEvidence::claim`: one held-out robot is pretraining-set evidence, two or more distinct ones are a transfer claim |
| `report` | obligations 5, 6 | `AdaptationReport`: baseline and adapted held-out metrics with regressions listed, rollback artifact, and cost in real minutes, real episodes, interventions, downtime and simulation steps; rendered as `adaptation-report/1` TOML |

## Decisions worth knowing

**The order is a gate, not advice.** §11.3: "a method MUST NOT be
scheduled before the methods above it have been measured and found
insufficient on the same robot." `may_schedule` returns the blocking
method. A method found *sufficient* closes the ladder below it.

**Deployment carries no autograd.** §6: "Deployment does not depend on
training autograd code." The deployable parts (history window, encoder
inference, residual bound, ladder, report) build without `robot-learn`.
A robot-side consumer depends on this crate with
`default-features = false`, and both feature sets are linted.

**Out of range is loud.** Row 6's limitation is that a regressed latent
"fails silently outside" its randomization ranges. `ContextEncoder`
records the per-input envelope it was trained on and reports how many
inputs of each window fall outside it.

**Rollback is always there.** `FineTuneOutcome` keeps the untouched
baseline next to the candidate, `accepted` is decided on held-out data,
and `deployable()` returns the baseline whenever the candidate did not
beat it.

## What is not here, and why

- **Multi-simulator randomization** (row 10) needs a second qualified
  simulator. §9.1 does not grant one, so the ladder refuses the method.
- **Running adaptation on a robot** is not this crate's job. These are
  components and gates. Each method is an extension under §§9.3 and 9.6
  with its own manifest and conformance run, which belong to the training
  and control crates that wire them.

## Testing

`cargo test -p robot-adapt`: 27 tests. They cover:
- the ladder's order, per-robot scoping, closure by a sufficient method,
  the hardware decision, the refused second simulator, and evidence
  required;
- exact recovery, unidentifiability and joint dynamics with standard
  errors that shrink with data;
- window states and privileged names refused;
- an encoder that learns damping from history, flags extrapolation, and
  falls back to the default when short;
- a correction learned and validated on held-out data, and one that does
  not generalize, reported as such;
- residual authority, rate and saturation;
- a bias-only adapter accepted, a regressing fine-tune not deployed, and
  a guided objective;
- retrieval filtering before ranking;
- the G1 descriptor and transfer-claim scope;
- the report's regressions and TOML.

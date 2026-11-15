# isaac-gr00t

Runs NVIDIA Isaac GR00T N-series foundation-policy checkpoints
**offline** to generate demonstration episodes and distillation targets
for training under SPECS.md Section 9.

Status: `unverified`, unpinned, no `Containerfile` yet. See
[`../README.md`](../README.md) for what qualification requires.

## The boundary this record exists to state

GR00T is a *policy-class* model, so SPECS.md §2.4 item 7 binds it
harder than the video-to-SMPL-X estimator:

- Its output enters this workspace **as data only** — episodes, labels,
  distillation targets — through a validated ingestion path, as
  untrusted input under §13.
- It is **never a controller**. Nothing derived from it reaches a robot
  except as Rust-inference weights (`robot-mlp/1`, run by `robot-nn`)
  inside a §7.4 policy bundle that passed the ordinary activation
  gates. A GR00T-shaped capability on the control path would require
  the Rust reimplementation route of §2, not this record.
- It runs on development hosts only; a robot host never has a container
  runtime (§2.4 item 3).

## Open at qualification

- Which N-series checkpoint, at which pinned weights identity, under
  which exact NVIDIA license text (recorded in the evidence ledger).
- The concrete episode file schema: the in-memory episode and
  transition records exist in `robot-types`, but no on-disk episode
  format is committed yet, so this service's `output_format` cannot be
  more specific than it is. Specifying that format is part of this
  service's qualification, not a precondition smuggled past it.
- Whether episode generation needs simulator-in-the-loop (which would
  couple this record to the §9.1 simulator's batch interface) or runs
  open-loop over recorded observations.

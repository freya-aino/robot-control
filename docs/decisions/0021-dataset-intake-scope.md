# 0021. Dataset intake joins the external-source scope, motion data first

- Date: 2026-11-15
- Touches: scope of `services/` / external-source registration (SPECS.md
  §2.4, §8.1, C-07). No code changes in this record.

Status: accepted (scope); every individual dataset remains blocked on its
own license audit (C-07) before ingestion.

## Context

SPECS.md §8.1 names candidate corpora (AMASS and its sources, CMU
boxing-related sequences, LAFAN1, the martial-arts set PMC7813879, Bandai
Namco collections) and C-07 requires a per-source license audit before any
is ingested. The §2.4 registry (`services/`, ADR 0017) so far registers
only containerized *producers* (video-to-SMPL-X, GR00T). Public motion
datasets — files someone downloads under a license, not services that run —
had no registration home.

## Decision

The project owner has directed that the external-source scope **include
the most popular freely available humanoid and motion-capture datasets**,
and that the initial scope needs **no sensor data** — motion/pose content
only (joint trajectories, SMPL-X/BVH/C3D poses), not IMU streams, raw
video or other sensor channels those corpora may carry.

Mechanically:

1. Datasets are registered like services but as what they are: a
   `data-source/1` record kind alongside `robot-service/1` (name,
   capability, license disposition, access route, evidence state,
   content identity of whatever is fetched), validated by the same gate.
   The record format is the next implementation step, mirroring ADR 0017.
2. Each dataset enters as `unverified` until its own license audit (C-07)
   is recorded in the evidence ledger; registration never implies
   permission to ingest, and the datasets themselves are never vendored.
3. Everything fetched enters through `robot data import` as untrusted,
   content-addressed input — the same file-only boundary §2.4 draws for
   services.
4. "No sensor data" is recorded as the initial scope so that pipelines
   (conditioning, retargeting, validation) are built and judged against
   motion content only; widening to sensor channels is a later, separate
   scope decision.

## Consequences

- C-07's per-corpus audits become the live blocking item for intake; the
  shortlist to audit first is a project-owner choice (several of the
  popular sources, AMASS included, require individual registration that
  only the owner can perform).
- The `services/` README and gate grow the `data-source/1` kind when the
  format lands.

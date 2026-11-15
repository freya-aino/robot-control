# video-to-smplx

Estimates SMPL-X body-model parameters from monocular video and exports
them as **AMASS-layout `.npz`**, the format this workspace already
ingests end to end: `motion_io::amass` parses SMPL-X 165-value pose
vectors with `trans`, `betas`, `gender` and `mocap_framerate`, and
`robot data import` stores the original content-addressed before
parsing. **The Rust side of this seam needed no new code** — choosing
AMASS layout as the output contract is the whole point of this record.

Status: `unverified`, unpinned, no `Containerfile` yet. See
[`../README.md`](../README.md) for what qualification requires.

## Candidate upstream implementations

All `unverified` (named in conversation and public listings; no primary
source retrieved, no code inspected — SPECS.md §3 licenses nothing
beyond scheduling an investigation):

| Candidate | Claimed capability | Noted concern to check |
| --- | --- | --- |
| GVHMR (zju3dv) | World-grounded human motion recovery from monocular video | Output is SMPL-X-compatible? License of code and of required body models |
| 4D-Humans / HMR 2.0 (shubham-goel) | Tracking and 3D human reconstruction from video | Native output is SMPL, not SMPL-X; a conversion step would need its own validation |
| SMPLer-X | Large-scale expressive pose and shape estimation | Model-zoo license terms; per-frame (no temporal coherence) output may need smoothing before retargeting |

Qualification picks one, reads it at a pinned revision, records its
license closure in the evidence ledger, writes the `Containerfile`
(sources pinned, SMPL-X assets fetched under the user's own MPI
registration at build/run — never baked into a shared image), builds,
pins the digest, and round-trips a declared video fixture through
`robot data import`.

## Output contract

One `.npz` per processed clip, AMASS layout:

- `poses`: frames × 165 (SMPL-X), radians, axis-angle per joint
- `trans`: frames × 3, metres
- `betas`: shape coefficients
- `gender`: string scalar
- `mocap_framerate` (or `mocap_frame_rate`): Hz

Anything else the estimator produces (vertices, camera, confidence) is
not part of the contract; if it becomes needed, the contract version
bumps and the importer grows an explicit, tested field. Note the
standing importer caveat: AMASS-layout records carry zero rest offsets
because the SMPL-X template is licensed separately; forward kinematics
on the result needs the caller to supply model offsets
(`crates/motion-io/README.md`).

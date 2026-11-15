# video-to-smplx

Estimates SMPL-X body-model parameters from monocular video and exports
them as **AMASS-layout `.npz`**, the format this workspace already
ingests end to end: `motion_io::amass` parses SMPL-X 165-value pose
vectors with `trans`, `betas`, `gender` and `mocap_framerate`, and
`robot data import` stores the original content-addressed before
parsing. **The Rust side of this seam needed no new code** — choosing
AMASS layout as the output contract is the whole point of this record.

Status: `paper_verified`, unpinned, no `Containerfile` yet. See
[`../README.md`](../README.md) for what qualification requires.

## Selected candidate: GVHMR (zju3dv)

Promoted to `paper_verified` on 2026-11-15 (evidence ledger, §2.4
candidate table): repository, `docs/INSTALL.md` and `LICENSE` retrieved
at pinned commit `ee960bb6e2ea2d381aa97f08e9b71ef320b624b1`; identity
confirmed as "GVHMR: World-Grounded Human Motion Recovery via
Gravity-View Coordinates", SIGGRAPH Asia 2024 / TPAMI 2026. The claim
this record rests on is located in text: "We predict SMPLX params"
(`docs/INSTALL.md`), with SMPL/SMPL-X body models downloaded under the
user's own MPI registration — matching this record's
`license_disposition`.

**License constraint, recorded before any image exists:** the GVHMR
license (ZJU 3D Vision Group) permits educational, research and
non-profit use only; derivative modifications must be open-source;
commercial use requires emailed permission. Any future commercial use
of this project revisits this service first.

What `paper_verified` does **not** license (SPECS.md §3): that the
implementation works, reproduces, or that its native dump format is
usable as-is. The demo pipeline's output format is unverified; the
AMASS-layout contract below is produced by an export step **owned by
the container**, written and read at `code_inspected`.

Remaining to `code_inspected` and a pinned digest: read the
implementation and its export path at the pinned commit, inventory the
dependency closure, write the `Containerfile` (sources pinned, SMPL-X
assets fetched at build/run under the user's registration — never baked
into a shared image), build, pin, and round-trip a declared video
fixture through `robot data import`. That change also adds rootless
`podman` to `devenv.nix` (C-30).

## Other candidates (still `unverified`)

| Candidate | Claimed capability | Noted concern to check |
| --- | --- | --- |
| 4D-Humans / HMR 2.0 (shubham-goel) | Tracking and 3D human reconstruction from video | Native output is SMPL, not SMPL-X; a conversion step would need its own validation |
| SMPLer-X | Large-scale expressive pose and shape estimation | Model-zoo license terms; per-frame (no temporal coherence) output may need smoothing before retargeting |

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

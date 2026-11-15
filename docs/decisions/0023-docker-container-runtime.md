# 0023. Docker is the development-host container runtime

- Status: accepted
- Date: 2026-11-15
- Tracks: C-30 (runtime half resolved; per-service qualification stays open)
- Touches: SPECS.md Section 2.4 item 8 and the C-30 row; `services/README.md`.

## Context

Section 2.4 item 8 designated rootless `podman`, to be added to
`devenv.nix` when the first service record reached `code_inspected`. The
development host in actual use runs a system-level Docker daemon
(29.x), and the project owner has chosen to use it rather than
provision a second runtime.

## Decision

**Docker replaces rootless podman as the designated runtime** for the
containerized foreign services of Section 2.4, with the boundary rules
unchanged and restated here because the runtime is the least important
part of them:

1. Images are still pinned by digest in their `robot-service/1` records;
   a record with an empty digest still fails qualification whatever
   runtime would run it.
2. Services still exchange **files only**, their output still enters
   through `robot data import` as untrusted content-addressed input, and
   nothing container-shaped touches the Cargo closure, a robot host or a
   control path.
3. The runtime is **not** added to `devenv.nix`: Docker is system-level
   on the host, and Section 2.2's rule against speculative inputs now
   cuts the other way — there is nothing to provision. The services
   gate and records do not name a runtime; only qualification procedure
   docs do.
4. Rootless podman remains the recorded fallback for a host without
   Docker; choosing it there is configuration, not a new decision.

## Consequences

- The remaining C-30 question is the per-service half only: which
  services qualify first, from which pinned upstream revisions, with the
  license closure recorded and the image digest-pinned.
- The practical unblock: `services/video-to-smplx` (already
  `paper_verified` on GVHMR) can proceed to image build and digest
  pinning as soon as the model-asset licenses (the owner's SMPL-X
  registration) are in hand.

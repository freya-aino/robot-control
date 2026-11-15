# 0017. Containerized foreign services at the data boundary

- Date: 2026-11-15
- Touches: `SPECS.md` (new Section 2.4, dependency-class table, rows A-51
  and A-52, glossary, C-30), `services/`,
  `crates/robot-devtools/src/services_gate.rs`,
  `crates/robot-devtools/src/bin/exclusion_gate.rs`, `devenv.nix`
  (git-hook trigger pattern only).

Status: accepted.

## Context

The project needs capabilities whose best current implementations are
foreign stacks: monocular video to SMPL-X body-model parameters, and
pretrained humanoid foundation policies (NVIDIA Isaac GR00T N-series) as
offline demonstration and distillation sources. SPECS.md classified two
integration routes for such a capability — linking it through FFI
(Section 2.1, excluded by default) and calling a hosted endpoint
(Section 2.3, excluded from the default build and unpinnable under the
Section 3 evidence policy) — but not the route this project already uses
informally for AMASS: the foreign producer runs elsewhere, writes files,
and a std-only Rust importer validates and content-addresses them.

That route was a habit, not an artifact. There was no registry of which
foreign producers the project relies on, no pinning discipline, no gate,
and no stated rule keeping a foreign *policy* model's output on the data
side of the control boundary.

## Decision

1. **A third named dependency class**, specified in SPECS.md Section 2.4:
   a *containerized foreign service* is a foreign stack packaged as an
   OCI container, run on development hosts only, exchanging data
   exclusively through files that a Rust ingestion path validates
   (untrusted input per Section 13, content-addressed per Section 8.2).
   It is not an FFI exception (nothing links) and not an external model
   service (nothing is hosted); containers make the foreign runtime
   pinnable by digest, which is exactly what Section 2.3 item 6 says a
   hosted endpoint is not.

2. **A registry, `services/<name>/`**, one directory per service, each
   holding a `service.toml` in the **`robot-service/1`** format: the
   bounded TOML subset of `robot-artifacts`, flat keys only
   (`format`, `name`, `capability`, `class`, `evidence_state`,
   `output_format`, `ingest_command`, `license_disposition`,
   `image_reference`, `image_digest`), unknown keys refused. The only
   admissible class is `development-tool`; a runtime class does not
   exist in v1 and is refused by name, which is this format's way of
   making Section 2.4 item 3 structural rather than reviewed.

3. **A services gate in `robot-devtools`**
   (`services_gate::check_services`), run by the same `exclusion-gate`
   binary that enforces Section 2.1 item 4, so `devenv test` and the
   git hook enforce rows A-51 and A-52 without a new entry point. The
   pinning rule is typed: `image_digest` must be empty below
   `code_inspected`, and at `code_inspected` or better it must be a
   well-formed `sha256:` digest accompanied by an `image_reference` and
   a `Containerfile` in the service directory.

4. **Two candidate records, both `unverified`, both unpinned**:
   `services/video-to-smplx/` (contract: AMASS-layout `.npz` with
   SMPL-X pose vectors, which `motion_io::amass` and `robot data
   import` already ingest end to end — the Rust side of this seam
   needed no new code) and `services/isaac-gr00t/` (offline episode
   generation and distillation targets only, per Section 2.4 item 7).
   Candidate upstream implementations are recorded in the evidence
   ledger at `unverified`, consistent with Appendix B's treatment of
   unqualified survey entries.

5. **No speculative provisioning.** Rootless `podman` is the designated
   runtime but does *not* enter `devenv.nix` with this change, because
   no record has reached `code_inspected` and `devenv.nix` adds inputs
   only when a qualified dependency requires them. The only `devenv.nix`
   change is the git-hook trigger pattern, so editing a service record
   re-runs the gate. C-30 tracks first qualification.

## Consequences

- The default Cargo closure, the default environment and every shipping
  crate are untouched; deleting `services/` leaves the workspace green
  (row A-52). The class cannot silently grow a runtime foothold: the
  record format has no field that could express one.
- A foreign result becomes replayable evidence: image digest plus
  weights identity plus invocation, recorded per Section 2.4 item 5,
  where a hosted endpoint result is permanently `not replayable`.
- Qualifying a service is now a defined procedure (C-30's admitting
  artifact list) instead of an ad-hoc judgement, and the first
  qualification carries known work: write the `Containerfile` at pinned
  revisions, record the license closure, build, pin, round-trip a
  fixture through `robot data import`, and add `podman` to `devenv.nix`
  in the same change.
- SMPL-family and GR00T weights never enter this repository or a shared
  image; the record's `license_disposition` is gate-required to be
  nonempty, so the question cannot be skipped silently.

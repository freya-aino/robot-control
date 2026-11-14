# 0016. The `robot-bundle/1` and `skill-manifest/1` file formats

- Date: 2026-11-15
- Touches: `crates/robot-artifacts/src/bundle.rs`, `crates/robot-skills/src/manifest.rs`, `crates/robot-cli/src/commands/bundle.rs` (`robot skills`, `robot eval --bundle`).

Status: accepted.

## Context

SPECS.md Section 7.4 defines the policy bundle and Section 10.2 the skill
descriptor, and both existed as in-memory types
(`robot_types::bundle::PolicyBundleManifest`,
`robot_skills::descriptor::SkillDescriptor`) with no on-disk form. That gap
was the stated reason `robot skills`, `robot eval --bundle` and half of
`robot deploy` exited with code 2. The specs name the artifact discipline
(Section 7.5: TOML manifests; Section 13: treat everything read back from
disk as untrusted; Section 12.2: "wrong artifact deployed" → refuse to
activate) but not a concrete file layout. This ADR fills that gap.

## Decision

1. **`skill-manifest/1`** (`robot_skills::manifest`): one TOML document,
   `format = "skill-manifest/1"`, one `[[skill]]` table per descriptor with
   every Section 10.2 field. Unknown keys, duplicate ids or names, and
   invalid fields are refused by name. Read and write round-trip exactly.

2. **`robot-bundle/1`** (`robot_artifacts::bundle`): a directory of
   `manifest.toml` plus one opaque blob per learned component under
   `weights/<name>`. The manifest is the full `PolicyBundleManifest` in the
   bounded TOML subset of `robot_artifacts::toml`. Its `digest` seals the
   manifest: `digest_parts` of the manifest serialized with an empty digest
   and no signature. Every blob is named, sized and content-addressed by its
   `ComponentManifest` and verified — size before content, content before
   interpretation — on both write and read. `write_bundle` refuses to seal a
   manifest over blobs it does not describe. Activation remains
   `check_activatable`, run by the caller against a real host envelope.

3. **What v1 refuses, by name** rather than encoding half-heartedly:
   `ActionSemantics::Latent`, `RecurrentStatePolicy::TransformedBy`, and a
   component with `decodes_latent_space` — all carry compile-time artifact
   identities and no latent decoder exists in this workspace — and any
   observation unit outside a fixed SI symbol table
   (`robot_artifacts::bundle::UNIT_SYMBOLS`), because `FieldSpec.unit` is a
   compile-time symbol and this reader interns rather than leaks.

4. **CLI**: `robot skills list|search --manifest <toml>` answers from the
   typed descriptors; `robot eval --bundle <dir> --robot <fixture>` verifies
   integrity and runs the activation gate against the named vendored robot's
   joint limits in `SimEval` mode. Exit 0 is "verified and activatable",
   not a quality claim. `robot deploy` still refuses on C-02 alone.

## Consequences

- A bundle on disk can be reviewed, diffed, content-verified and refused
  before a byte of its weights is interpreted; the Section 12.2 load-time
  control is now executable from the CLI.
- The weight blobs stay format-agnostic here; `robot-mlp/1`
  (`robot_nn::format`) is the first concrete component format.
- When a latent decoder or a recurrent-state transformation artifact ever
  exists, this format needs a v2, and the v1 reader will name the refusal
  rather than misread the file.

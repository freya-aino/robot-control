# services/ — containerized foreign services

The registry of SPECS.md Section 2.4 and
[ADR 0017](../docs/decisions/0017-containerized-foreign-services.md):
foreign stacks (Python pipelines, pretrained models, preprocessing
engines) that this project uses as **development tools in OCI
containers**, exchanging data with the Rust workspace **through files
only**. Nothing in this directory is a Cargo dependency, runs on a robot
host, or is reachable from a control path. Deleting this directory
leaves every crate building and every test passing (row A-52).

## Layout

One directory per service:

```
services/<name>/
  service.toml     # the robot-service/1 record (required, gate-checked)
  README.md        # candidates, contract detail, qualification status
  Containerfile    # required at code_inspected or better; sources pinned
```

Every `service.toml` is validated by the services gate
(`robot-devtools`, run from the `exclusion-gate` binary in `devenv
test` and the git hook). The gate enforces row A-51: unknown keys, an
inadmissible class, a digest the evidence state does not license, a
qualified record without its `Containerfile`, or a directory without a
record each fail the build by name.

## The `robot-service/1` record

Written in the bounded TOML subset of `robot-artifacts`, flat keys only:

| Key | Meaning |
| --- | --- |
| `format` | Exactly `"robot-service/1"` |
| `name` | The directory name: lowercase, digits, hyphens |
| `capability` | The single capability provided (SPECS.md §2.1 item 1 discipline) |
| `class` | Exactly `"development-tool"`; v1 admits no other class |
| `evidence_state` | One of the five SPECS.md §3 states |
| `output_format` | The documented file format the container writes |
| `ingest_command` | The Rust ingestion path that validates that output |
| `license_disposition` | How licensed models/assets arrive; never vendored (§2.4 item 6) |
| `image_reference` | The image name/tag; may name an upstream candidate before qualification |
| `image_digest` | Empty below `code_inspected`; a `sha256:` digest once pinned |

## Lifecycle

1. **`unverified`** — the record names the capability, the output
   contract and the ingestion seam. No digest, no `Containerfile`
   required. This is where every service starts.
2. **`code_inspected`** — the upstream implementation has been read at a
   pinned revision and its license closure recorded in
   `docs/ledger/evidence-ledger.md`; the `Containerfile` exists with
   pinned sources; the image is built and `image_digest` pinned. The
   designated runtime is the host's system-level **Docker** (§2.4
   item 8, ADR 0023); nothing enters `devenv.nix`, and rootless
   `podman` is the recorded fallback for a host without Docker.
3. **Rebuild or retag** — a version change under SPECS.md §3: dependent
   claims demote, the ledger records the new digest.

A result that passed through a service records the image digest, the
model-weights identity and the invocation, which is what makes it
replayable (§2.4 item 5).

## Rules that do not bend

- **Files, not calls.** A long-running endpoint called from workspace
  code is an external model service under §2.3, not this class.
- **Output is untrusted input** (§13): it enters only through the named
  `ingest_command`, which validates, bounds and content-addresses it.
- **Licensed assets are fetched under the user's own license** at image
  build or container run — never committed, never baked into a shared
  image.
- **Policy-class models produce data, not control** (§2.4 item 7): a
  foundation policy's output enters training as data; a robot runs only
  Rust-inference weights from a §7.4 bundle.

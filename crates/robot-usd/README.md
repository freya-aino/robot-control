# robot-usd

A std-only reader for OpenUSD layers: the binary USDC "crate" container, a
minimal layer composer, and a USDA text reader and writer. No third-party crate,
no FFI, no `unsafe`.

## What's here

| Module | Contents |
| --- | --- |
| `compression` | LZ4 block decompression, the `TfFastCompression` container, and the integer codec of `integerCoding.cpp` |
| `crate_file` | `PXR-USDC` reader: bootstrap, table of contents, TOKENS, STRINGS, FIELDS, FIELDSETS, PATHS, SPECS, and every value representation the G1 files use |
| `diagnostics` | Non-fatal findings: `shader_warnings` reports MDL material sources that exist only in Omniverse (the four G1 materials, all `OmniPBR.mdl`) |
| `layer` | `Layer` and `Spec`: the format-independent view of one file, with format detection |
| `stage` | Composition: sublayers, variant selections, internal and external references, payloads |
| `replay` | `write_replay_usda(replay)`: a recorded robot run as a USDA layer with `timeSamples` on link world poses (per frame), UsdPhysics state and `robot:appliedTorque` on joints, and collision and primitive shapes as static children of their links (ADR 0015) |
| `usda` | `write_usda(stage)` and `parse_layer` / `parse_stage` for the subset the writer emits |
| `value` | `Value`, `ListOp`, `Arc`, `Specifier`, `Variability` |

Byte layouts follow OpenUSD commit `84d8ed1ae0337ce30dca8d91a3fbe7f04eebc358`
(`pxr/usd/sdf/crateFile.cpp`, `crateFile.h`, `crateDataTypes.h`,
`crateValueInliners.h`, `integerCoding.cpp`, `pxr/base/tf/fastCompression.cpp`).

## Conventions

Quaternions are stored as `(x, y, z, w)`, as the crate stores them. USDA text
writes `(w, x, y, z)`, which the replay writer converts. The reader's
`Value::Quatf` documentation that says `(real, i, j, k)` is wrong about the
storage order; the G1 principal axes and identity joint frames confirm `(x, y, z, w)`.

## Not implemented (reported by name, never skipped)

- Crate formats older than 0.4.0 and newer than 0.8.x.
- Value types outside those the G1 files use (`UnregisteredValue`, `Spline`,
  `Relocates`, `PathExpression`, `AnimationBlock`, `Permission`) error with
  their type id.
- Composition: inherits, specializes, non-identity layer offsets, deleted or
  ordered list items, variant sets beyond the selection of each prim, and
  standalone connection specs. Each makes composition fail with `Unsupported`.
- Relationship targets are kept as authored; they are not remapped through
  references or payloads.
- USDA: the writer emits a subset (prims, typed attributes, relationships,
  connections, time samples, metadata with a type keyword). Halves are read as
  `float`.

## Verification

Unit tests cover LZ4, the container, the integer codec, the TOC and inline
values, and the USDA reader. `tests/g1.rs` checks the four vendored layers
against values measured with this crate: all specs decode, 275 composed prims
by type, the joint and link examples, collision geometry, mesh extents against
their points, and a USDA round trip.

`cargo run -p robot-usd --example usd_dump -- <root layer> [prim path] [depth]`
prints the composed tree.

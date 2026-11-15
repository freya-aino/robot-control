# robot-usd

A std-only reader for OpenUSD layers: the binary USDC "crate" container, a
minimal layer composer, and a USDA text reader and writer. No third-party crate,
no FFI, no `unsafe`.

## What's here

| Module | Contents |
| --- | --- |
| `compression` | LZ4 block decompression, the `TfFastCompression` container, and the integer codec of `integerCoding.cpp` |
| `crate_file` | `PXR-USDC` reader: bootstrap, table of contents, TOKENS, STRINGS, FIELDS, FIELDSETS, PATHS, SPECS, and every value representation the G1 files use |
| `error` | `UsdError` (`Malformed`, `Unsupported`, I/O) and the crate `Result` |
| `diagnostics` | Non-fatal findings: `shader_warnings` reports MDL material sources that exist only in Omniverse (the four G1 materials, all `OmniPBR.mdl`) |
| `layer` | `Layer` and `Spec`: the format-independent view of one file, with format detection |
| `stage` | Composition in LIVRPS order: sublayers, inherits, variant selections, internal and external references, payloads, specializes; layer offsets applied to time samples; full `SdfListOp` edits; relationship and connection targets remapped through arcs |
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

Half-precision values (`half`, `half2..4`, `quath`, arrays) are decoded from
their binary16 bit patterns — subnormals, signed zeros, infinities and NaN —
and widened to the `float` family, which is exact: binary32 represents every
finite binary16 value without rounding. `Value` therefore has no half
variants, and the only information not preserved is the authored type name,
so a USDA round trip re-emits halves as `float`. The decode is pinned by
exact-equality tests on the edge patterns and a strict-monotonicity sweep
over all 31744 positive finite patterns.

## Not implemented (reported by name, never skipped)

- Crate formats older than 0.4.0 and newer than 0.8.x.
- Value types outside those the G1 files use (`UnregisteredValue`, `Spline`,
  `Relocates`, `PathExpression`, `AnimationBlock`, `Permission`) error with
  their type id.
- Composition: variant sets beyond the authored selection of each prim (no
  fallback-driven selection), and standalone connection or relationship-target
  specs that author fields of their own. Each makes composition fail with
  `Unsupported`.
- USDA: the writer emits a subset (prims, typed attributes, relationships,
  connections, time samples, metadata with a type keyword). The authored
  `half` type *name* is not round-tripped (see Conventions: the values
  themselves are decoded exactly and widened).

## Verification

Unit tests cover LZ4, the container, the integer codec, the TOC and inline
values, and the USDA reader. `tests/composition.rs` exercises the arcs the
vendored files don't: inherits, specializes, list edits (delete, reorder,
explicit), sublayer and reference layer offsets on time samples, target
remapping through a reference, and fieldless standalone target specs.
`tests/g1.rs` checks the four vendored layers
against values measured with this crate: all specs decode, 275 composed prims
by type, the joint and link examples, collision geometry, mesh extents against
their points, and a USDA round trip.

`cargo run -p robot-usd --example usd_dump -- <root layer> [prim path] [depth]`
prints the composed tree.

# motion-io

Motion import and export, per SPECS.md §8.1 and §8.2 items 1–2. Depends on
`robot-types` and `robot-artifacts` (the blob store that keeps originals).

## What's here

| Module | Contents |
| --- | --- |
| `bvh` | BVH parser: hierarchy (`ROOT`/`JOINT`/`End Site`, `OFFSET`, `CHANNELS`) and motion block into `MotionRecord<HumanSkeleton>` |
| `bvh_export` | BVH writer, the parser's inverse (rotation order kept per joint), and `quat_to_euler` for every Tait-Bryan order |
| `c3d` | C3D reader: Intel, DEC (VAX floats) and MIPS files, integer and float data, `POINT:UNITS` to metres, invalid points as `Missing`, residuals kept as per-sample confidence, scaled analog channels, the full parameter section |
| `npz` / `inflate` | ZIP (stored and DEFLATE members, CRC-32 checked) and NPY (v1-3, numeric and unicode dtypes) readers; a bounded RFC 1951 decoder |
| `amass` | AMASS `.npz`: SMPL (72), SMPL-H (156) and SMPL-X (165) pose vectors with their public kinematic trees, `trans`, `betas`, gender and rate, as `MotionRecord<BodyModelParameters>` |
| `import` | Content-based format detection; `import` stores the original in a `BlobStore` before parsing so `source_hash` names bytes on disk; `inspect` (frames, rate, licence, validity, and labelled up-axis and length-unit guesses) |
| `capture` | Live capture: the documented `robot-capture/1` line protocol, `LineAdapter` and non-blocking `UdpAdapter`, and a `Recorder` that keeps capture-clock timestamps, per-marker confidence, dropped and reordered frame counts, and a clock-offset estimate |

All parsers are authored against `std` under
[`docs/decisions/0003-std-only-default-closure.md`](../../docs/decisions/0003-std-only-default-closure.md).

The BVH parser stands in for Appendix B.2.4's still-`unverified`
`bvh_anim`/`bvh_anim_parser` candidates.

## Where the root pose goes, and why non-root position channels are rejected

A BVH root joint's position channels go into the canonical record's
per-frame `root: Transform`, not into a `JointValue` — SPECS.md §7.2's
schema has exactly one root pose per frame. The root's rotation channels
go to `root.rotation` too, and are also kept as channel 0's
`JointValue::Rotation` so the source channel order and count are
preserved; the two are the same quantity, and a consumer reads the pose
from `root`. A non-root joint declaring a
position channel (legal BVH, essentially never used) has nowhere correct to
go in that schema, so the importer reports
`BvhError::Unsupported { construct: "position channel on non-root joint ..." }`
rather than silently misplacing it into the wrong field.

## Malformed-file limits

SPECS.md §8.1 asks for malformed-file limits. A declared `Frames:` count
above `MAX_FRAMES` (1,000,000) or a hierarchy above `MAX_CHANNELS` (4096)
is rejected with `BvhError::Limit` before any allocation sized by the
declaration. Offsets are not unit-reconciled (BVH does not declare its
length unit); the caller's provenance record carries the unit.

## Rotation order is recorded, not assumed

SPECS.md §7.2: "Euler-angle conventions are source-specific; never assume
every BVH file uses the same rotation order." Each joint's `CHANNELS` line
is read in whatever order it lists its rotation axes, and that order is
stored as `JointValue::Rotation`'s `source_order` field on every frame —
not normalized away during import.

## AMASS rest offsets are zero, on purpose

A joint's rest position comes from the SMPL-family template and joint
regressor evaluated at `betas`. Those model files are licensed separately
from AMASS and are not in this repository. The importer therefore records
zero offsets and says so in provenance, rather than inventing a skeleton.
Forward kinematics on an AMASS record needs offsets from a model the caller
is licensed to use.

## Live capture links no vendor SDK

`robot-capture/1` is a one-line-per-frame text protocol (see `capture`'s
module docs). A vendor system speaks it through a bridge process outside
this repository. The recorder keeps the capture clock's sampling stamps as
the timeline, so network jitter never moves a frame. Sequence numbers make
a dropped frame a counted fact, not a guess.

## Testing

`cargo test -p motion-io`: 35 tests. They cover:
- BVH hierarchy, root handling, rotation order and rejection cases (8);
- BVH export round-tripping through the parser in every Euler order,
  including gimbal lock;
- C3D in integer Intel and float MIPS layouts, residuals, analog scaling,
  VAX float conversion and truncation;
- stored, fixed-Huffman and dynamic-Huffman DEFLATE streams, the output
  limit, and the CRC-32 check value;
- NPY and ZIP round trips, corruption, Fortran order;
- AMASS SMPL-H import with its tree and rate, and refused widths;
- content-based detection, originals stored before parsing, and Y-up
  centimetre inspection;
- capture gaps, confidence, sampling time, malformed lines and UDP
  delivery.

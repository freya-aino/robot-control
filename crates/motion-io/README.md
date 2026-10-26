# motion-io

Motion import and export, per SPECS.md §8.1. Depends on `robot-types` and
`robot-artifacts` (the latter not yet used — reserved for storing imported
motion as a content-addressed blob).

## What's here

| Module | Contents |
| --- | --- |
| `bvh` | A BVH parser: hierarchy (`ROOT`/`JOINT`/`End Site`, `OFFSET`, `CHANNELS`) and motion block, converting into `robot_types::motion::MotionRecord<HumanSkeleton>` |

PLAN.md's first executable backlog lists "Implement BVH inspection and
canonical conversion" as item 5, right after the typed contracts — this is
that step, authored per
[`docs/decisions/0003-std-only-default-closure.md`](../../docs/decisions/0003-std-only-default-closure.md)
in place of Appendix B.2.4's still-`unverified` `bvh_anim`/`bvh_anim_parser`
candidates.

## Where root translation goes, and why non-root position channels are rejected

A BVH root joint's position channels go into the canonical record's
per-frame `root: Transform`, not into a `JointValue` — SPECS.md §7.2's
schema has exactly one root pose per frame. A non-root joint declaring a
position channel (legal BVH, essentially never used) has nowhere correct to
go in that schema, so the importer reports
`BvhError::Unsupported { construct: "position channel on non-root joint ..." }`
rather than silently misplacing it into the wrong field.

## Rotation order is recorded, not assumed

SPECS.md §7.2: "Euler-angle conventions are source-specific; never assume
every BVH file uses the same rotation order." Each joint's `CHANNELS` line
is read in whatever order it lists its rotation axes, and that order is
stored as `JointValue::Rotation`'s `source_order` field on every frame —
not normalized away during import.

## Testing

`cargo test -p motion-io` — 6 tests: hierarchy flattening and parent
indices, root translation landing in `frame.root` rather than a joint
value, rotation-order preservation, and three rejection cases (channel
count mismatch, a non-root position channel, an unknown channel name).

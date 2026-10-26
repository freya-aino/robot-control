# robot-model

The kinematic tree, inertias, actuator limits, sensor placement and the robot
profile of SPECS.md Section 7.1. Depends only on `robot-types`.

## What's here

| Module | SPECS.md anchor | Contents |
| --- | --- | --- |
| `pose` | — | `RigidPose`: a local, un-named translation+rotation, deliberately distinct from `robot_types::frames::Transform` (see below) |
| `joint` | §7.1 | `JointKind` (revolute/continuous/prismatic/fixed), `JointLimits`, `JointSpec` with explicit sign/zero-offset conversion |
| `link` | §7.1 | `MassProperties`, `InertiaTensor` with a physical-plausibility check, `CollisionGeometry` (sphere/box/cylinder/mesh-by-reference) |
| `actuator` | §7.1 | `Transmission`, a two-corner `TorqueSpeedEnvelope`, `PdGains`, a single-pole `ThermalLimits` model |
| `sensor` | §7.1 | `SensorKind`, `DiagonalCovariance`, `MissingDataBehavior` as an explicit per-sensor declaration |
| `profile` | §7.1 | `RobotProfile`: identity uniqueness, dangling-reference checks, joint-graph-is-a-tree validation, and the stricter `activate()` gate (extension strictness + independent e-stop) required before a control path |
| `kinematics` | §6 | `forward_kinematics`: one unambiguous computation over the validated tree |
| `urdf` | §7.1 | A minimal authored XML reader plus the URDF kinematic/mass subset this project supports |

## Why `RigidPose` instead of `robot_types::frames::Transform`

`Transform` exists so that genuinely cross-cutting, named project frames (the
world frame, a sensor's output convention — the things C-04 leaves open)
can't be silently composed across a mismatch. A joint's origin, a sensor's
mounting offset and a collision shape's placement aren't that: each is
structurally local to exactly one link. Inventing a synthetic `FrameId` per
link to satisfy `Transform`'s checking would only relabel "the parent link's
own frame" without adding information — and it very nearly produced a subtle
bug during development (an early version of `forward_kinematics` assumed a
link's index coincided with `FrameId::WORLD`'s index, which is false in
general). `RigidPose` is the deliberately un-named equivalent for the local
case, and `forward_kinematics` composes `RigidPose`s, not `Transform`s.

## URDF import: what's supported and what's named as unsupported

Supported: `<link>` with `<inertial>`/`<collision><geometry>`
(box/sphere/cylinder/mesh-by-reference), `<joint type="revolute | continuous |
prismatic | fixed">` with `<parent>`/`<child>`/`<origin>`/`<axis>`/`<limit>`.

Rejected **by name**, never silently dropped (SPECS.md §7.1: "Unsupported
model features must be rejected explicitly rather than silently dropped"):
`floating`/`planar` joint types, a `mimic` element, and any `<transmission>`
block. A real actuator's transmission ratio, gains, thermal curve and command
latency are not in URDF at all — `urdf::placeholder_actuator` exists only for
fixtures and says so in its own doc comment; a real profile authors
`ActuatorSpec` values from the vendor's documentation.

`urdf-rs` (Appendix B.2.4) remains the shortlisted long-term candidate; this
module is the authored interim path required because Appendix B.6 admits no
dependency until it clears qualification, and because "the platform owns
semantic validation" regardless of which parser eventually supplies the raw
XML tree.

## Testing

`cargo test -p robot-model` — 66 tests: joint/link/actuator/sensor validation
boundaries, profile graph checks (dangling references, non-tree in-degrees, a
disconnected-but-locally-valid cycle), forward-kinematics composition checked
against hand-computed rotations, and URDF import including every explicit
rejection case.

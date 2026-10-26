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
| `fixtures` | §3 | Vendored real-robot descriptions built into validated `RobotProfile`s — currently Unitree's G1 humanoid, 29-DOF rev 1.0 |

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

## The vendored Unitree G1 fixture

`fixtures::unitree_g1_29dof()` builds a validated `RobotProfile` for
Unitree's G1 humanoid, 29-DOF revision 1.0, from a vendored URDF — see
`assets/robots/unitree_g1/PROVENANCE.md` at the repository root for the
exact source commit, license (BSD-3-Clause, Unitree Robotics), SHA-256,
and what was checked (no `mimic`/`transmission`/`floating`/`planar`
constructs) before vendoring it.

This is a kinematics-only fixture, deliberately. Its actuators are all
`urdf::placeholder_actuator` (joint limits standing in for a real
envelope — URDF has no transmission/gain/thermal/latency data at all),
and its `FirmwareContract` declares `has_independent_estop: false`
because this crate has no cited, verified source for the real robot's
e-stop circuit — not because the real G1 lacks one. `.validate()`
succeeds; `.activate()` correctly fails with `NoIndependentEstop` until
a real, cited firmware contract replaces this one. Use it for forward
kinematics, IK retargeting (`motion-retarget`) and inspection
(`robot-viewer`), not as a claim this profile is safe to drive hardware.

No mesh geometry is vendored — the URDF's `.STL` references are kept as
opaque strings, same as any other imported URDF, since this project has
neither a renderer nor a mesh-consuming physics engine yet.

## Testing

`cargo test -p robot-model` — 70 tests: joint/link/actuator/sensor validation
boundaries, profile graph checks (dangling references, non-tree in-degrees, a
disconnected-but-locally-valid cycle), forward-kinematics composition checked
against hand-computed rotations, URDF import including every explicit
rejection case, and the vendored G1 fixture (profile validation, the
29-DOF count, the expected joint names, the activation refusal above, and
forward kinematics running over the full real tree at the zero posture).

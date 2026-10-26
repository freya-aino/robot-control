# robot-model

The kinematic tree, inertias, actuator limits, sensor placement and the robot
profile of SPECS.md Section 7.1. Depends only on `robot-types`.

## What's here

| Module | SPECS.md anchor | Contents |
| --- | --- | --- |
| `pose` | — | `RigidPose`: a local, un-named translation+rotation, deliberately distinct from `robot_types::frames::Transform` (see below) |
| `joint` | §7.1 | `JointKind` (revolute/continuous/prismatic/fixed/free), `JointLimits`, `JointSpec` with explicit sign/zero-offset conversion |
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

`fixtures::unitree_g1_29dof_floating_base()` builds on the same import:
it adds an explicit, massless `world` link and a `JointKind::Free` joint
(see below) re-rooting the profile there instead of at `pelvis`, so the
pelvis gains six unconstrained degrees of freedom against world rather
than being implicitly fixed. It still only `.validate()`s, for the same
reason — no cited firmware contract exists either way.

## `JointKind::Free`: the floating base, as a simulation-time construct

`JointKind` gained a fifth variant, `Free` (six degrees of freedom:
translation plus rotation), specifically to let a profile's root attach
to an explicit `world` link rather than being implicitly welded to
whatever frame holds it. It is **never importable from URDF** — the
importer's rejection of a `floating` joint type by name is unchanged —
because `crates/robot-sim/SIMULATION_SPECS.md` Section 1.6 treats a
floating base as a decision the simulation layer makes, not a fact a
robot's own description file asserts about itself.

A `Free` joint has no single axis and no single scalar value, unlike
every other kind here, so two things had to change rather than just
grow a new `match` arm: `JointSpec::validate`'s degenerate-axis check
now only applies to the three single-axis kinds
(`JointKind::has_single_axis`), and `JointSpec::transform_at` —
which takes one `f64` — panics loudly on a `Free` joint rather than
guessing, with a message pointing at `robot_sim::articulated`'s own,
richer position representation. `robot_model::kinematics::
forward_kinematics` is built on `transform_at` and therefore inherits
that same loud panic rather than silently producing a wrong pose if
ever called on a floating-base profile — it has not been generalized to
support `Free` in this revision, and does not need to be: the dynamics
solver that actually needs a floating base
(`robot_sim::articulated::ArticulatedBody`) reads `Free`'s position
directly, never through `transform_at`.

`RobotProfile::dof()` also had a real, if currently invisible, bug this
surfaced: it counted actuated *joints*, not their *degrees of freedom*,
which was silently correct only because every joint kind before `Free`
contributed exactly one. It now sums `JointKind::dof_count()`, which is
numerically identical to the old behavior for every profile without a
`Free` joint (i.e. every profile that existed before this change) and
correct for the ones that do.

## Testing

`cargo test -p robot-model` — 76 tests: joint/link/actuator/sensor validation
boundaries (including `Free`'s own degenerate-axis exemption, its rejection
of position limits, and `transform_at`'s documented panic), profile graph
checks (dangling references, non-tree in-degrees, a
disconnected-but-locally-valid cycle), forward-kinematics composition checked
against hand-computed rotations, URDF import including every explicit
rejection case, and the vendored G1 fixture in both its fixed-base form
(profile validation, the 29-DOF count, the expected joint names, the
activation refusal above, forward kinematics at the zero posture) and its
floating-base form (the added `world` link and `Free` joint, the 35-DOF
count, the same activation refusal).

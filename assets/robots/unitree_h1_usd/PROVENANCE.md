# Provenance: the official H1 USD layers

| Field | Value |
| --- | --- |
| Robot | Unitree H1 humanoid, 19 actuated joints |
| Source repository | `unitreerobotics/unitree_model` |
| Source path | `H1/h1/usd/` (the root layer and `configuration/`) |
| Pinned commit | `b6a8942b0803b6c137e58cef12beb4b03e4a2fa7` (committed 2026-06-01T15:08:48Z) — the same commit the G1 layers in `../unitree_g1_usd/` are pinned to |
| Fetched | 2026-11-15 |
| License | BSD 3-Clause (`LICENSE` in this directory, SHA-256 `d62cbfaf8906d8d8d2315dc43f375165a2b76a8fb56cfedecc28929ec2e83752`, byte-identical to the file vendored with the G1 layers) |

Each file is a Git LFS object in the source repository. The pointer files at
the pinned commit name these object IDs, and the SHA-256 of each downloaded
object equals its object ID:

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `h1.usd` | 1600 | `3f128fc0f9d66c4225f03d8a5f80570514c0b25cf2da580ddb12e536e61fd14a` |
| `configuration/h1_base.usd` | 26283174 | `73dea11b425beeeb28234fd63861e6ad3ed58ad66c6dfb28990fa946848f2b3c` |
| `configuration/h1_physics.usd` | 7877 | `58026cc629b43f8f2f84c62e2f21ecfd3b3e60173f5e7f5fd3f61aefb9ef5137` |
| `configuration/h1_sensor.usd` | 654 | `ae888ecf7f281ef2628ad88024a78ae77c6d3cc1c5de41fa56074aba717d3b9f` |

## What the importer verified at vendoring time

Measured by `robot_model::usd::import_fixed_base` on these exact files (the
assertions live in `robot_model::fixtures`' tests):

- 20 links, root `pelvis` (the only link with no incoming joint).
- 19 joints, all `PhysicsRevoluteJoint`: 5 per leg (hip yaw, hip roll, hip
  pitch, knee, ankle), 1 torso, 4 per arm (shoulder pitch/roll/yaw, elbow).
  No wrist or hand joints in this variant.
- Total declared link mass 51.601 kg. This is the sum of the file's
  `physics:mass` attributes, not a verified fact about any physical robot.

## What these files do *not* give this project

Same as the G1 layers: drive stiffness/damping and torque limits are read
from the physics layer, but no firmware version, command latency, watchdog
behavior, or emergency-stop circuit information exists here. The fixture's
`FirmwareContract` therefore declares no independent e-stop, and
`RobotProfile::activate` refuses it — kinematics, simulation and
retargeting use only.

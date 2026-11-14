# Provenance: the official Go2 USD layers

| Field | Value |
| --- | --- |
| Robot | Unitree Go2 quadruped, 12 actuated joints |
| Source repository | `unitreerobotics/unitree_model` |
| Source path | `Go2/usd/` (the root layer and `configuration/`) |
| Pinned commit | `b6a8942b0803b6c137e58cef12beb4b03e4a2fa7` (committed 2026-06-01T15:08:48Z) — the same commit the G1 layers in `../unitree_g1_usd/` are pinned to |
| Fetched | 2026-11-15 |
| License | BSD 3-Clause (`LICENSE` in this directory, SHA-256 `d62cbfaf8906d8d8d2315dc43f375165a2b76a8fb56cfedecc28929ec2e83752`, byte-identical to the file vendored with the G1 layers) |

Each file is a Git LFS object in the source repository. The pointer files at
the pinned commit name these object IDs, and the SHA-256 of each downloaded
object equals its object ID:

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `go2.usd` | 1601 | `fd3c8c23c97cea3208fae1ab5eeacba4ffa616fe5f1397be6c7edf4feb93f180` |
| `configuration/go2_description_base.usd` | 18806122 | `eeed42fb9e41395c47bc3fac7ff5d58625ac62f44bfa5a09cacebff83229c7a0` |
| `configuration/go2_description_physics.usd` | 6829 | `43800f436b3533c869f0b7774e5c70bc4291579df70a62263f808fb1a8f05505` |
| `configuration/go2_description_sensor.usd` | 655 | `d63cfa5964eb5a72b8e2a02dc89448881497717f21d76b1ec9683a92cb765a16` |

## What the importer verified at vendoring time

Measured by `robot_model::usd::import_fixed_base` on these exact files (the
assertions live in `robot_model::fixtures`' tests):

- 19 links, root `base` (the only link with no incoming joint).
- 18 joints: 12 `PhysicsRevoluteJoint` (3 per leg: hip, thigh, calf) and 6
  `PhysicsFixedJoint` (the four foot spheres and the two head segments).
  The fixed joints are why this robot, not the G1, first exercised the
  importer's `PhysicsFixedJoint` support.
- Total declared link mass 16.087 kg. This is the sum of the file's
  `physics:mass` attributes, not a verified fact about any physical robot.

## What these files do *not* give this project

Same as the G1 layers: drive stiffness/damping and torque limits are read
from the physics layer, but no firmware version, command latency, watchdog
behavior, or emergency-stop circuit information exists here. The fixture's
`FirmwareContract` therefore declares no independent e-stop, and
`RobotProfile::activate` refuses it — kinematics, simulation and
retargeting use only.

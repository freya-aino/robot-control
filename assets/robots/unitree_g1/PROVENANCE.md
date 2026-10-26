# Provenance: `g1_29dof_rev_1_0.urdf`

| Field | Value |
| --- | --- |
| Robot | Unitree G1 humanoid, 29-DOF, revision 1.0 |
| Source repository | `unitreerobotics/unitree_rl_gym` |
| Source path | `resources/robots/g1_description/g1_29dof_rev_1_0.urdf` |
| Pinned commit | `276801e46c5d433564f24658bac64f254b7d2d4b` (2025-07-25T10:20:21Z) |
| Fetched | 2026-10-26, verified byte-identical between `main`'s tip at fetch time and the pinned commit above |
| SHA-256 of the vendored file | `824ce02c2c1e489aa8dece47a10fe02d1872289c0e6d01ba51abd291e66b7b2c` |
| License | BSD 3-Clause, Copyright (c) 2016-2023 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics") — full text in `LICENSE` in this directory, copied verbatim from the same commit |

## Why this exact variant

The user asked for "the same version [...] used by
[HansZ8/RoboJuDo](https://github.com/HansZ8/RoboJuDo)". RoboJuDo does not
vendor a URDF at all — it vendors the MuJoCo MJCF sibling,
`assets/robots/g1/g1_29dof_rev_1_0.xml`, and
`robojudo/config/g1/env/g1_env_cfg.py`'s `G1EnvCfg` (the active/default
config in `robojudo/config/g1/g1_cfg.py` — the 23-DOF and 12-DOF configs
are present but commented out) sets:

```python
xml: str = (ASSETS_DIR / "robots/g1/g1_29dof_rev_1_0.xml").as_posix()
```

i.e. the same `g1_29dof_rev_1_0` name and DOF count (29: 2×6 leg joints,
3 waist joints, 2×7 arm joints) this project's URDF importer needs an
actual URDF for, not MJCF. `unitree_rl_gym` carries the identically-named
URDF and MJCF side by side in the same directory (same joint names, same
`_rev_1_0` revision tag), which is the standard pairing Unitree publishes
for this robot — this URDF is that pairing's URDF half.

## What was and wasn't checked before vendoring

Verified directly against the file (not assumed):

- Root link: `pelvis` (unique, no other link lacks an incoming joint).
- 39 links, 38 real joints: 29 `revolute` + 9 `fixed`. The one `type="floating"`
  joint and `world` link in the file are inside an XML comment
  (`<!-- [CAUTION] uncomment when convert to mujoco -->`) and are not part
  of the real structure.
- No `mimic` or `<transmission>` elements — both are constructs this
  project's importer rejects by name.
- The `<mujoco><compiler .../></mujoco>` element at the top of the file is
  an unrecognized child of `<robot>` to this project's importer and is
  silently skipped, same as any other unknown tag.

## What this file does *not* give this project

- **Mesh geometry.** The URDF references `.STL` files under `meshes/` for
  visual and collision geometry. Those binary mesh files are **not**
  vendored here — this project has no rendering toolkit to show them
  (see `robot-viewer`'s README) and no physics engine that currently
  consumes collision meshes (see `robot-sim`'s README). The filenames are
  retained as opaque reference strings by the importer, exactly as
  `robot_model::urdf::parse_link`'s own doc comment describes: resolving
  them to a content-addressed blob is `robot-artifacts`' job, not this
  importer's, and is not done in this revision.
- **Actuator, firmware or safety facts.** URDF carries joint kinematics
  and mass properties only — no transmission ratio, gains, thermal
  limits, command latency, firmware version, watchdog behavior, or
  emergency-stop circuit information. `robot_model::fixtures::unitree_g1_29dof`
  builds its actuators from `robot_model::urdf::placeholder_actuator`
  (joint limits standing in for a real envelope) and a `FirmwareContract`
  with `has_independent_estop: false`, since nothing here is a cited,
  verified claim about the real robot's safety hardware. See that
  function's doc comment for why this is deliberate, not an oversight.

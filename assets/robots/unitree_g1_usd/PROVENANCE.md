# Provenance: the official G1 USD layers

| Field | Value |
| --- | --- |
| Robot | Unitree G1 humanoid, 29-DOF, revision 1.0 |
| Source repository | `unitreerobotics/unitree_model` |
| Source path | `G1/29dof/usd/g1_29dof_rev_1_0/` (the root layer and `configuration/`) |
| Pinned commit | `b6a8942b0803b6c137e58cef12beb4b03e4a2fa7` (committed 2026-06-01T15:08:48Z) |
| License | BSD 3-Clause (`LICENSE` in this directory, SHA-256 `d62cbfaf8906d8d8d2315dc43f375165a2b76a8fb56cfedecc28929ec2e83752`) |

Each file is a Git LFS object in the source repository. The pointer files at
the pinned commit name these object IDs, and the SHA-256 of each downloaded
object equals its object ID:

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `g1_29dof_rev_1_0.usd` | 1603 | `cfed730bb043c42708260f25bf3bdf49ac80deed5d5a49384721b8e01059a96e` |
| `configuration/g1_29dof_rev_1_0_base.usd` | 28337485 | `d9768c942783ae0932f0c3db3558d5283b9a93ba0e561d15d8984f548aa2a65d` |
| `configuration/g1_29dof_rev_1_0_physics.usd` | 10366 | `cde4ff0378183e4e2063df1fdcad4060df8247eaf1da777f45e2adea0b6dde22` |
| `configuration/g1_29dof_rev_1_0_sensor.usd` | 656 | `5115364b53bffbb37007af0fd1038336679b66abdae134bc87d0fca9829f9b60` |

## Status

These are binary USD crate (`PXR-USDC`) files and are the reference G1 model
(ADR 0015). The USD reader in `crates/robot-usd` is the project's own std-only
decoder. Its progress is recorded in that crate's README, not here.

The joint names and link names are stored in compressed tokens inside the
container, so they cannot be read with `strings`; they are read by the reader.

## Scene

The ground plane is not part of this layer. It was added in a separate scene
layer, `assets/scenes/g1_ground.usda`, which sublayers this file unchanged. The
layers in this directory are the byte-identical files whose SHA-256 is listed
above; `assets/scenes/README.md` documents the added ground collider.


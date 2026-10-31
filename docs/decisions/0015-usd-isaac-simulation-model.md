# 0015 — the simulation model is the OpenUSD description used by Isaac Sim

- Status: accepted. Implemented on 2026-10-30: the reader (`crates/robot-usd`), the importer (`robot_model::usd`), the replay writer (`robot_usd::replay`), and the simulator's use of the USD model. Their evidence state is `code_inspected` for the layouts and `reproduced_in_sim` for the G1 runs listed in the replay folder README; no hardware evidence exists.
- Date: 2026-10-30
- Supersedes: ADR 0005 (the target wording), ADR 0011 (the reference-oracle plan) and ADR 0014 (the withdrawn G1 import path). ADR 0014 is retained as a tombstone.
- Touches: SPECS.md §2 (dependency policy), §7.1 (embodiment description), §7.3 (episode record), §7.5 (storage), §14 (acceptance); `crates/robot-usd` (new), `crates/robot-model`, `crates/robot-sim`, `assets/robots/unitree_g1_usd/`.

## Context

The simulation needs a complete description of the robot and its environment: links, joints, collision and primitive shapes, masses, drive limits and sensors. The project's first robot description was a URDF, which carries kinematics only. A second description, a MuJoCo scene, was imported for a time. Both are removed from the repository by this decision. The reference for the robot model is now the official Unitree G1 OpenUSD asset, which Unitree publishes for Isaac Sim.

OpenUSD is the scene format Isaac Sim uses. It stores time-sampled attributes, so the same format can describe the static robot model and a recorded replay. Isaac Sim's physics runs on a USD stage, and its results are written back to the stage as time samples. Blender and other USD viewers can open the files without a project-specific renderer.

## Decision

1. **The source of truth for the robot and the environment is OpenUSD.** The model used by `robot-model` and `robot-sim` is built from the USD files, not from a hand-written profile. The vendored G1 layers under `assets/robots/unitree_g1_usd/` are the reference asset. The structure is read from the layers, and the USD `UsdPhysics` and `UsdGeom` schemas define the links, joints, collision shapes, masses and limits.
2. **The Rust simulator is the physics engine.** `robot-sim` integrates the multibody dynamics, applies the motor torques, and computes the contacts. USD carries the description; it does not run the physics.
3. **Capture is complete.** A run records, for every step: joint positions and velocities, commanded and applied torques, the pose of every link, the pose and dimensions of every collision and primitive shape, contact points and forces, and the environment's static geometry. Anything the simulator knows about the robot must be writable to USD.
4. **Replays are USD files.** A replay is a USD stage with time samples on link transforms, joint attributes and custom physics attributes. Any USD viewer can open it, including Blender. The project's own SVG frames remain as a quick diagnostic and are not the replay.
5. **The reader is authored in Rust, with no third-party crate.** `crates/robot-usd` reads the USDC binary container and the USDA text form. It is std-only, consistent with ADR 0003. OpenUSD itself is not a build dependency.
6. **Isaac Sim is the fidelity reference, used when it is available.** Fidelity claims are measured against analytic checks first. Runs in Isaac Sim, when a user installs it, are a second check and are not part of the build.
7. **Every earlier simulator reference is removed.** No code, asset, active specification or active document names the removed simulator. The exception is the exclusion gate's deny list, which keeps excluded foreign runtimes out of the build (ADR 0013). Historical records keep their text and are marked superseded.

## Consequences

- The default build gains a reader and no dependency.
- The contact model covers spheres and the ground plane. USD primitive shapes (boxes, cylinders, capsules) are captured, but they do not yet take part in contact. This gap is recorded in the robot-sim README and remains open.
- Mesh geometry is captured as a reference to the asset and is not rendered by the project.
- Replays for the zero-torque run show the physical result as recorded. Until primitive contacts exist, the torso can pass through the ground, as the earlier run showed.

## Implementation record

- `crates/robot-usd`: decodes all four vendored G1 layers, composes them with variant selections, payloads and references, and writes and reads USDA. OpenUSD commit `84d8ed1ae0337ce30dca8d91a3fbe7f04eebc358` is the byte-layout reference.
- `robot_model::usd`: builds the G1 profile from the layer. The zero-pose kinematics agree with the URDF of the same revision to 1.5e-8 m, and the joint limits, effort and velocity agree with it too.
- The simulator (`robot sim multibody`) loads the USD profile and writes `replay.usda` next to its frame dump. The replay has link world poses per frame, joint position, velocity and applied torque, and the collision and primitive shapes of every link.

## Not captured, and why

- Contact forces and contact points are not exposed by `robot-sim`. The replay carries the collision shapes and the link poses from which contact can be read, but not the forces. Exposing them is a `robot-sim` change and is not made here.
- Mesh geometry is referenced by prim path and not written. The asset's visual meshes stay in the vendored layers.
- The contact model still covers the spheres and the ground plane only. The USD foot boxes and the knee and shoulder cylinders are captured, not simulated.

## Known limits

- The replay of a 2 s run at 10 kHz is about 137 MB of USDA text. A stride option would reduce it; none is implemented.
- The replay passes OpenUSD's `usdchecker` (package `openusd` 26.05, from the Nix store) and `usdcat` reads it back. The G1 root layer does not pass `usdchecker`: its MDL materials (`OmniPBR.mdl`) are Omniverse-only and cannot be resolved outside it. The layer is read by `robot-usd` regardless.
- Blender (installed by the user) imports the replay headlessly: 66 objects, frame range 0 to 19999, and the pelvis samples match the simulator (0.7969 m at frame 0, -0.6458 m at frame 19999).
- Omniverse-only MDL shaders in the G1 layer are reported as warnings (`robot_usd::diagnostics`), not errors. The simulator prints them on load and continues.

## Playback in Blender (2026-10-30)

Blender maps each USD time code to a frame and keeps its own frame rate (24 fps by default); it does not read `timeCodesPerSecond` from the layer. A full-rate replay therefore plays too slowly to review. The simulator also writes `replay_playback.usda` at 100 time codes per second, and the Blender scene frame rate has to be set to 100 fps for real-time playback. The replay writes each visual mesh's geometry as a static `Mesh` prim under its link, so the G1 body appears in Blender (36 meshes). Materials are not written: they are Omniverse MDL shaders.

## Ground collider in the scene (2026-10-30)

The ground is a `Plane` collider in `assets/scenes/g1_ground.usda`, which sublayers the unmodified Unitree layer. The simulator reads the plane and its contact parameters from that layer; the constant that was in the simulator is removed. The contact parameters are the same values (stiffness 20 000 N/m, damping 200 N s/m, friction 0.8). Because USD stores `physics:dynamicFriction` as a 32-bit float, the friction is 0.800000011920929 in the scene, which changes chaotic outcomes (the zero-torque fall) slightly from about 0.4 s on. The PD stance run is unchanged to the printed precision. The replay writes the ground as a `Plane` prim.

## Universal environment tags (2026-10-30)

Environment collision is decided by tags, not by file. A prim whose `sim:tags` contains `environment` and `collider` is a static collider wherever it sits in any scene; `environment` alone is visual. The importer reads them with `robot_model::usd::import_environment`, and the simulator takes its environment from the scene named by `--scene`. The simulator contacts planes only; a tagged box, sphere, cylinder or mesh is refused by name until the contact work of the plan lands. The robot is placed on the scene's plane height, not on z = 0.


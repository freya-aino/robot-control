# Appendix A — Citation ledger

[Index](README.md) · [Step 00](00-target-and-constraints.md) · [Step 02](02-pure-dynamics-core.md) · [Step 04](04-fidelity-loop.md)

This appendix is the citation ledger for the whole spec. It is referenced by the `[the reference simulator §N]`, `[batched backend §N]`, `[Brax §N]`, `[IsaacLab §N]` and `[Rust-GPU §16]` tags in the step files. Source: old `SIMULATION_SPECS.md` lines 221–246, moved verbatim.


All entries below reflect two research passes performed on 2026-10-26 by an agent with live web/source access, reading primary sources directly (via `curl` against raw repository content where precision mattered, not third-party summarization). Every entry is `code_inspected` per `SPECS.md` §3's table (source/implementation read at a pinned commit) unless marked `secondary` (general knowledge, not independently re-verified this pass) or `unverified` (the agent explicitly could not confirm it).

| # | Subject | Source | Pin | Notes |
| --- | --- | --- | --- | --- |
| 1 | Coordinates, joint types, free-joint velocity convention | `google-deepmind/reference`, `include/reference/mjtype.h`, `doc/overview.rst` | tag `3.17.0`, commit `800a109a4bdd02cb60404d565db4ae16fb04a53b` | — |
| 2 | CRBA/RNE, `mj_forward` pipeline stages | same repo, `src/engine/engine_core_smooth.c`, `engine_forward.c` | same pin | Gravity-as-fictitious-base-acceleration trick confirmed identical sign convention to this session's own code |
| 3 | Constraint solver, `solref`/`solimp`, friction cones, PGS/CG/Newton | same repo, `src/engine/engine_solver.c`, `doc/computation.rst` | same pin | Exact version Newton became default: `unverified` (doc states current default, not changelog-dated) |
| 4 | Collision pipeline, GJK/EPA, convex hulls | same repo, `src/engine/engine_collision_driver.c` and docs | same pin | — |
| 5 | Integrators (Euler/RK4/implicit/implicitfast/discrete) | same repo, `doc/computation.rst` §Numerical integration | same pin | Exact introducing version of each non-Euler/RK4 integrator: `unverified` |
| 6 | Actuators, tendons, equality constraints | same repo, `doc/computation.rst` §Actuation, `mjtype.h` | same pin | — |
| 7 | Passive forces | same repo, `doc/computation.rst` §Passive forces | same pin | — |
| 8 | Sensors | same repo, `include/reference/mjtype.h`, `mjmodel.h` | same pin | Auto-application of `sensor_noise`: `unverified` (field confirmed, injection point not traced into `engine_sensor.c`) |
| 9 | Model compilation (`inertiafromgeom`, `fusestatic`, mesh→hull) | same repo, `doc/XMLreference.rst` compiler section | same pin | — |
| 10 | Determinism, native `mjbatch`, threading | same repo, `doc/computation.rst` §Reproducibility, `include/reference/mjbatch.h` | same pin | `mjbatch.h` copyright-dated 2026, shipped in the pinned 3.17.0 release itself |
| 11 | the reference simulator 3.x feature history | same repo, `doc/changelog.rst` | same pin | — |
| 12 | batched backend architecture, static shapes, top-k contact selection, unsupported-feature enums | same repo, `mjx/reference/mjx/_src/{types,collision_driver,solver}.py`, `doc/mjx.rst` | same pin | Published throughput numbers (A100/TPU-v5/M3-Max/3995WX steps-per-sec): `secondary` — the reference's own reported figures for specific models/hardware, not independently reproduced |
| 13/16 | CubeCL capabilities, maturity, issues | `tracel-ai/cubecl` README, `crates/cubecl-core/src/frontend/{branch,synchronization}.rs`, issue tracker | repo state as of 2026-10-26, release v0.11.3 | Suitability for non-NN/physics workloads specifically: `unverified` — zero found precedent either way |
| 13/16 | Candle `cutile` custom-kernel path | `huggingface/candle` README, `candle-book/src/guide/cutile.md` | repo state as of 2026-10-26 | — |
| 13/16 | `wgpu` capabilities, Bevy non-rendering-compute precedent | `gfx-rs/wgpu` README; `bevyengine/bevy` examples `compute_shader_game_of_life.rs`, `gpu_readback.rs`; `rakeshbs/nbody_bevy_compute_shader` | repo states as of 2026-10-26 | All found precedent is embarrassingly-parallel, not tree-recursive |
| 16 | Search for an existing Rust-native GPU rigid-body engine | GitHub search, `Far-Beyond-Pulsar/RapierGPU` and broader query | as of 2026-10-26 | `RapierGPU`: single-day repo, 1 star, 8 months stale — judged not a usable precedent |
| 14 | Batching-strategy comparison (vmap-per-level vs. per-thread SIMT) | synthesis from entries 3/12 above | — | The specific claim "this is *why* batched backend-Warp was built" is `secondary` (reasonable inference from both systems' documented designs, not a quoted statement from either team) |
| 15 | Variable-fidelity precedent: the reference simulator (none named), PhysX/IsaacLab (TGS/PGS + solver-backend family) | `doc/mjx.rst` performance-tuning section; `isaac-sim/IsaacLab` `docs/source/how-to/solver_tuning_physx.rst`, `docs/source/concepts/solver_differences.rst` | repo states as of 2026-10-26 | Why TGS improves on PGS mechanically: `secondary` (general robotics-simulation background, not found explained in the read IsaacLab docs) |
| 17 | Brax `Env`/`PipelineEnv` boundary, pipeline list, batched backend leak points | `google/brax`, `brax/envs/base.py`; `doc/mjx.rst` | repo/pin states as of 2026-10-26 | — |
| 17 | Rapier internal module separation | `dimforge/rapier`, `src/dynamics/solver/*`, `src/geometry/broad_phase` | repo state as of 2026-10-26 | Whether this separation is trait-abstracted specifically for backend-swapping: `unverified` |

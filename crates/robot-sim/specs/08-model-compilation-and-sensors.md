# Step 08 — Model compilation, asset import and sensors

[Index](README.md) · [Appendix A](appendix-a-citations.md)

**Status**: Not started. `robot_model::urdf::parse_link` treats a missing `<inertial>` element as `MassProperties::MASSLESS` (see §10 item 1). The sensor-noise injection point is undecided (Appendix B S-06).

**Goal**: Geometry-based inertia inference as an opt-in import mode, an optional static-body fusing pass, convex-hull reduction for meshes, a sensor type set, and an explicit sensor-noise policy. Covers old §9 and §10.

**Depends on**: [01](01-state-and-coordinates.md) (joint and link topology), [02](02-pure-dynamics-core.md) (mass properties feed inertia inference).

**Reference formulation**: Tag `3.17.0`, commit `800a109a4bdd02cb60404d565db4ae16fb04a53b` (Appendix A, entries 8 and 9). Function definitions, located by name at the pin:
- `src/engine/engine_sensor.c`: `mj_sensorPos`, `mj_sensorVel`, `mj_sensorAcc` (sensor evaluation stages by position, velocity and acceleration).
- `doc/XMLreference.rst` compiler section for `inertiafromgeom` and `fusestatic`. Inertia inference and static fusing are compiler behaviours, documented there rather than in a single named C function. Convex-hull generation is in the compiler and is not pinned to a function in Step 1.
- `include/reference/mjtype.h` and `mjmodel.h` for the sensor catalog.

Evidence tag: `unverified` for every behavioural claim in this step, including the `sensor_noise` injection point (entry 8). The sensor function definitions are located at the pin, but their bodies were not read in Step 1.

**Purity contract**: Inertia inference (volume and mass from geometry) uses `std` and `f64` arithmetic only. No `Matrix`, `RigidPose` or `HashMap` in algorithm functions (FR-6). Import adapters remain in `robot-model`.

**Inputs/outputs**: Input: URDF link inertial and collision geometry, compiler options. Output: a compiled model with inferred or explicit mass properties, optional fused bodies, and convex hulls for mesh collision.

**Done when**: Inertia inference is opt-in and never changes the default import behavior. Static fusing is a caller-selected transform. Mesh collision is always reduced to a convex hull at compile time. Sensor noise is generated at a declared point and recorded as such.

**Open questions**: Appendix B S-05 (exact inference algorithm) and S-06 (sensor-noise location) are both blocking for this step's sensor-noise and inference items. Both are listed in [Appendix B](appendix-b-open-questions.md).

---

## 9. Sensors

1. The sensor types already named across `robot-types`/`robot-model`/`robot-perception` (joint position/velocity, IMU-style accelerometer/gyro, force/torque, contact) cover the robotics-relevant subset of the reference's 55-type catalog [the reference simulator §8]; this revision does not require matching the full catalog (frame sensors on arbitrary objects, subtree center-of-mass/momentum sensors, geometric-relationship sensors) unless a specific task needs one.
2. Whether a per-sensor noise model is applied by the physics engine itself or left to the caller MUST be an explicit, documented choice — not left ambiguous the way the reference's own `sensor_noise` field is (the research agent confirmed the field exists but could not confirm, without reading `engine_sensor.c` further, whether it is auto-applied [the reference simulator §8, flagged unverified]). This project's own existing convention (`StampKind`, explicit sampling-vs-arrival timestamps) already treats delay/uncertainty as data the producer must declare, not something the consumer infers; sensor noise should follow the same discipline — generated at a declared point in the pipeline, recorded as such.

## 10. Model compilation and asset import

1. `robot_model::urdf::parse_link`'s current behavior — a missing `<inertial>` element silently becomes `MassProperties::MASSLESS` — is a real, now-named gap against the reference's own URDF-import compiler behavior: the reference's `inertiafromgeom` defaults to `"auto"` (infer mass/inertia from the attached collision geometry's volume and a density when `<inertial>` is absent) specifically because, in its own documentation's words, "a number of publicly available URDF models [have] seemingly arbitrary inertias" [the reference simulator §9]. `robot-model` SHOULD add geometry-based inertia inference as an explicit, opt-in import mode (never a silent default change to existing behavior, consistent with `SPECS.md` §7.1's "reject unsupported constructs explicitly" ethos already governing this importer) rather than leaving every under-specified link massless.
2. A static-body-fusing pass (collapsing a chain of links connected only by `Fixed` joints into their parent, matching the reference's own `fusestatic` default for URDF import specifically [the reference simulator §9]) MAY be added as a separate, explicit transform a caller opts into — the vendored G1 fixture's 39 links/38 joints (versus what the reference's own compiler would produce from the identical URDF) is the concrete, already-present example this would change. This MUST remain optional: `robot-viewer`'s inspection use case benefits from every named link surviving import; a GPU-batched dynamics pass benefits from fewer, fused bodies.
3. Mesh collision geometry MUST be reduced to a convex hull before any narrow-phase collision routine sees it, matching the reference's own compile-time `qhull` step [the reference simulator §4, §9] — never at simulation time, and never passing a full concave mesh into a convex-convex collision routine.

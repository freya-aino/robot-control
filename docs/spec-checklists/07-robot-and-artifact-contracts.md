## 7. Robot and artifact contracts

### 7.1 Embodiment description

Each versioned robot profile describes:

- Stable joint/link names and explicit indices, units, axes, limits and zero offsets.
- Link masses, center of mass, inertia, collision meshes and available articulated degrees of freedom.
- Motor transmission, command interface, torque-speed envelope, gains, saturation, latency, thermal limits and uncertainty bounds.
- Sensor frames, timestamps, rates, covariance, calibration and missing-data behavior.
- Root convention, gravity, contact surfaces, feet and hand geometry, payload and glove changes.
- Firmware/protocol version, watchdog semantics, command expiration, emergency stop interface, and allowed operating modes.

Every profile MUST carry a `schema_version` and typed, stable IDs for joints, links, actuators and sensors. IDs MUST NOT be reused for a different entity within a profile lineage. The required core schema MUST remain small and explicit; additional declarative metadata belongs in open, namespaced extension tables with declared value types and capability tags. Unknown core fields and unknown extensions MUST be rejected before a profile is activated on a control path. Non-control tooling MUST preserve unknown namespaced extensions without interpreting them when it reads and writes a profile. A capability tag is a claim to validate against the profile's typed fields, not a substitute for them. Schema changes MUST provide explicit migrations, and each migration MUST preserve or report loss of meaning.

URDF import is an initial interchange target, not a complete dynamics specification. Store missing actuator/contact metadata separately. Unsupported model features must be rejected explicitly rather than silently dropped.

### 7.2 Canonical motion record

Store `schema_version`, source hash, source license, subject/session ID, skeleton/model hash, frame timestamps, validity masks, root pose, named joint rotations or positions, optional velocities, contact labels with confidence, task labels, and provenance for every transformation.

Use SI units, a documented right-handed coordinate system, explicit world/body frames, and named quaternion ordering. Prefer typed quaternion serialization with named components over ambiguous four-element arrays. Euler-angle conventions are source-specific; never assume every BVH file uses the same rotation order.

Human skeleton motion, parametric human-body parameters, and robot joint trajectories are distinct record types. BVH, C3D, and SMPL-family data are not interchangeable. SMPL assets and source datasets may have independent license restrictions.

### 7.3 Robot episode record

Record monotonic sensor timestamps, arrival times, synchronized wall-time mapping where needed, raw and estimated state, uncertainty, desired and measured joint motion, commanded actions, actuator acknowledgments if available, contacts, temperatures, power, policy/expert IDs, transition state, task outcome, and supervisor interventions.

Store both raw commands and applied/clamped commands so training cannot confuse policy intent with actuator input. Record validity and provenance for estimated torque; motor current is not automatically calibrated joint torque. Preserve observation and action history needed to model delays.

### 7.4 Policy bundle

A deployable bundle contains model weights in a Rust-loadable versioned format, graph/architecture version, observation schema, normalization statistics, action semantics and scaling, recurrent-state specification, rates, robot-profile compatibility, training data lineage, safety-envelope assumptions, transition contracts, benchmark results, and checksum/signature.

ONNX is an optional interchange format only if a qualifying Rust-native path supports the required operators. It is not the default training/export dependency. Unsupported operators must fail before deployment.

### 7.5 Storage and splits

Use chunked binary arrays for high-rate numerical data and columnar records for searchable metadata; evaluate Arrow/Parquet implementations for Rust. TOML manifests refer to content-addressed blobs. Never require loading an entire corpus into memory. Do not load executable pickle objects; supported legacy formats need a constrained Rust parser or a documented non-pickle export from the data provider.

Select IPC and persistent-record encoding separately using Appendix B's serialization candidates. Every wire/archive format requires explicit versioning, size limits, framing, endianness/alignment rules where applicable and validation before access. Serialization is not a transport or a schema-migration strategy. Durable episode records must remain readable across supported software upgrades.

Split datasets by subject, recording session, and original motion before generating windows or augmentations. For competitive learning, separate evaluation opponent pools. Track license eligibility, retention policy, consent where relevant, and dataset revisions.

### 7.6 Experience and replay storage contract

On-policy rollouts and off-policy replay MUST use the same versioned transition and episode schema. A transition MUST identify its episode, timestep, environment and robot-profile version, and contain the observation, action, reward terms, next observation, termination and time-limit-truncation state, validity masks, and any task data required to reproduce reward or goal relabeling. Preserve timestamps, provenance and random-stream identity where they affect the target or replay result. An episode MUST also carry its configuration, seed manifest and completion reason. This schema is distinct from the robot episode record in Section 7.3: a training transition is a typed view of the required training evidence, not a replacement for raw hardware records.

1. Training storage MUST expose bounded chunks in a structure-of-arrays (SoA) layout or an equivalently measured layout; it MUST NOT require materializing the entire corpus or a complete replay buffer as one allocation. The initial in-memory replay implementation SHOULD use preallocated fixed-capacity ring storage with explicit overwrite and generation-ID semantics. On-policy rollout storage MUST report capacity exhaustion rather than silently overwrite samples before the learner consumes them.
2. Storage and sampling MUST be separate interfaces. A sampler MAY provide uniform, prioritized (including a sum-tree implementation), sequence/window sampling, or goal relabeling such as HER-style relabeling. Sampling returns MUST identify source generations and, where applicable, probabilities and importance weights; stale or overwritten references MUST be rejected.
3. N-step targets, generalized advantage estimation (GAE), masks and other learner-specific targets MUST be derived views or batches over canonical transitions, not incompatible serialized transition formats. Derived caches MUST identify the source transition generations and computation/configuration version.
4. Each environment and sampler MUST have an independently recorded seed stream. Replaying the same stored transitions and sampler seed MUST reproduce indices, weights and relabeled goals, subject to explicitly recorded parallel-reduction nondeterminism. Parallel rollout workers MUST have stable environment IDs and independent RNG streams, consistent with Appendix B.2.12.
5. Persistent chunks MUST use the versioned, bounded encodings of Section 7.5. The in-memory view SHOULD permit zero-copy access where alignment, validation and versioning allow it; zero-copy is not permission to trust unvalidated bytes or to make Rust ABI the durable schema. Migration and malformed-input behavior remain explicit.
6. Buffer policy, sampler, target/view construction and persistence MUST be independently replaceable. A new algorithm MUST be able to reuse the canonical transition schema without editing the robot model, simulator, or control runtime. These interfaces are training/offline interfaces; they MUST NOT be reachable from the deadline-critical control path.
7. An algorithm MAY require more than one concurrent store over the same schema — a competitive learner of Section 9.4 maintains both an on-policy rollout buffer and a reservoir-sampled behavior buffer, and a retrieval-augmented learner of Section 9.7 reads a prior corpus alongside the current rollout. Multiple stores MUST therefore be composable without a second transition format: each store declares its retention policy (bounded ring, reservoir, append-only corpus), its admission rule, and whether its contents are drawn from the current policy, a historical policy or an external corpus. Transitions MUST record which policy or corpus produced them and at which generation, because a record whose origin is unknown cannot be used as a behavior target.
8. Where a transition stores an action in a learned latent space rather than the actuator action space, it MUST record the latent-space identity and version and the decoder artifact required to interpret it. A latent transition is not interchangeable with a joint-space transition, and a buffer MUST reject a mixture unless the schema explicitly carries both.



## Implementation checklist

Status legend: `[x]` implemented, `[~]` partial, `[ ]` not implemented, `[?]` not audited in this pass. Summary: 4 `[x]`, 6 `[~]`, 1 `[ ]`, 8 `[?]` across 19 items.

- [x] 7.1.1 Every robot profile carries a `schema_version` — Evidence: `crates/robot-model/src/profile.rs` line 4 and `PROFILE_SCHEMA_VERSION` set in `crates/robot-model/src/fixtures.rs` line 101 and `kinematics.rs` line 207.
- [~] 7.1.2 Typed stable IDs for joints, links, actuators and sensors — Evidence: `JointId`, `LinkId`, `ActuatorId` imported in `kinematics.rs` line 148 and `fixtures.rs` line 22. Note: sensor ID typing was not confirmed in this pass.
- [~] 7.1.3 IDs are not reused for different entities within a lineage — Evidence: `RobotProfile::validate` checks uniqueness within one profile (`profile.rs` module docs). Note: no cross-version lineage check was found.
- [~] 7.1.4 Unknown core fields and extensions are rejected before activation — Evidence: `profile.rs` `activate` applies the `Strictness::Activating` extension-table check (module docs, lines 8-11). Note: unknown-field rejection at deserialization was not verified.
- [?] 7.1.5 Tooling preserves unknown namespaced extensions without interpreting them — not audited.
- [?] 7.1.6 Schema migrations preserve meaning or report loss — no `migrate` function was found in `robot-model` (`grep` of `migrat` returned nothing); `robot-artifacts` not checked.
- [?] 7.1.7 URDF unsupported features are rejected explicitly — `crates/robot-model/src/urdf.rs` exists; error behaviour not read.
- [?] 7.2 Canonical motion record stores source hash, license, validity masks and provenance — `crates/motion-io/src/bvh.rs` exists; record fields not audited.
- [?] 7.3 Robot episode record stores raw and applied commands, estimated-torque provenance and timestamps — `crates/robot-types/src/episode.rs` exists; fields not audited.
- [?] 7.4 Policy bundle contains weights, normalization, recurrent-state spec, rates and checksum — checkpoint layout is in ADR 0007; bundle contents not audited.
- [x] 7.5.1 Content-addressed blobs with TOML manifests — Evidence: `crates/robot-artifacts/src/hash.rs`, `manifest.rs`, `toml.rs`, `chunk.rs`.
- [x] 7.5.2 No executable pickle objects are loaded — Evidence: no Python code exists in the repository (see 1.1).
- [~] 7.5.3 Arrow/Parquet evaluation and explicit versioning of every wire/archive format — Note: no Arrow/Parquet evaluation record was found; `chunk.rs` versioning was not audited.
- [x] 7.6.7 Stores declare a retention policy — Evidence: `crates/robot-types/src/transition.rs` `RetentionPolicy` with a `BoundedRing` variant (line 609-613) and `Reservoir` in `crates/robot-types/src/rng.rs` line 277.
- [~] 7.6.1 Bounded fixed-capacity storage that reports capacity exhaustion instead of overwriting — Evidence: `crates/robot-learn/src/rollout.rs` `RolloutBuffer` with `capacity()` (lines 48-70). Note: overflow-reporting behaviour and generation IDs were not confirmed.
- [ ] 7.6.2 Prioritized sampling via a sum-tree, with stale-reference rejection — Evidence: `grep` for `sum.?tree` in `crates/` found no match. Note: implement a sampler interface that returns source generations and importance weights.
- [~] 7.6.4 Each environment and sampler has an independent seed stream — Evidence: `SeedStream` in `crates/robot-types/src/rng.rs` and `SeedStream::new(42, 7)` usage in `robot-learn/src/ppo.rs`. Note: the stable-env-ID and replay-determinism requirements were not verified.
- [?] 7.6.8 Latent-space transitions record latent identity and decoder artifact — not audited.
- [?] 7.6.3 and 7.6.6 Derived views (n-step, GAE) and independently replaceable stores — not audited.

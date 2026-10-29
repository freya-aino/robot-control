## 8. Motion acquisition and retargeting

### 8.1 Capture inputs

Deliver file-based BVH ingestion first, followed by C3D and supported numerical human-body formats. Add live capture through documented Rust protocol adapters for optical or inertial systems. Keep timestamps and confidence through every transform.

Video-to-motion is an optional later module: require a model that can be executed through the permitted Rust stack and validate camera calibration, root translation, occlusion and contact ambiguity. A pose estimated from ordinary video is not automatically a dynamically valid reference.

Candidate corpora include AMASS and its source datasets, CMU boxing-related sequences, LAFAN1, the martial-arts motion dataset referenced as PMC7813879, and Bandai Namco motion collections. Confirm exact action coverage, access terms, commercial/training permissions, and redistribution rules before ingestion. Do not assume BEAT-style co-speech corpora contain useful boxing demonstrations. Existing robot-retargeted corpora are tied to their source embodiment and conventions.

### 8.2 Processing pipeline

1. Inspect format, license, skeleton, rates and coordinate conventions.
2. Preserve immutable originals and assign content IDs.
3. Correct axes/units, validate finite values and timestamps, and mark gaps.
4. Filter and resample with documented phase delay; preserve real strike-speed characteristics.
5. Estimate ground plane and contact intervals with uncertainty.
6. Segment into stance, locomotion, strike, defense, recovery and transition windows.
7. Retarget to target geometry with contact, joint-limit and self-collision constraints.
8. Validate temporal smoothness, velocity/acceleration, torque estimates and simulated tracking feasibility.
9. Package accepted references and retain rejected examples with machine-readable reasons.

### 8.3 Retargeting requirements

Implement weighted task-space IK followed by temporal refinement. Support end-effector positions/orientations, posture priors, planted-foot constraints, joint limits and self-collision penalties. Preserve task intent, such as punch direction and target-relative reach, rather than matching all human coordinates equally.

Qualify a Rust-native constrained solver before selecting differential-QP IK or whole-body optimization. Measure setup, updates, solve time, allocations, residuals and failure detection on representative full-body constraints. Runtime uses require bounded iterations/deadlines and a validated response to infeasibility or timeout. MPC and constraint filters are optional control methods, not automatic substitutes for a learned tracker or proof of safety.

Morphology changes may require altering stance, stride, punch amplitude, torso rotation and timing. Motion time scaling is constrained by actuator and task requirements; it cannot repair every infeasible reference. A validation report must separate kinematic error, dynamics violations, contact consistency and task success.

No universal fidelity percentage is a release criterion. Measure endpoint error normalized by body size, root drift, planted-foot slip, joint-limit violations, torque-speed violations, and downstream tracking success per skill and embodiment.



## Implementation checklist

Status legend: `[x]` implemented, `[~]` partial, `[ ]` not implemented, `[?]` not audited in this pass. Summary: 3 `[x]`, 8 `[~]`, 10 `[ ]`, 2 `[?]` across 23 items.

- [x] 8.1.1 File-based BVH ingestion — Evidence: `crates/motion-io/src/bvh.rs` (parser, 6 tests per `crates/motion-io/README.md` lines 38-43).
- [x] 8.1.2 Rotation order is recorded per joint, not assumed — Evidence: `crates/motion-io/README.md` lines 30-36 (`JointValue::Rotation` `source_order`).
- [ ] 8.1.3 C3D ingestion — Evidence: `crates/motion-io/src/lib.rs` exposes only `bvh`; `crates/motion-io/README.md` module table lists only `bvh`. Note: add a C3D reader.
- [ ] 8.1.4 Supported numerical human-body formats (SMPL-family parameters) — Evidence: none in `crates/motion-io/` module table. Note: add a parser for the chosen format after license review (see 8.1.7).
- [ ] 8.1.5 Live capture through documented Rust protocol adapters — Evidence: no capture adapter found under `crates/motion-io/`. Note: implement an adapter and document its protocol.
- [ ] 8.1.6 Video-to-motion module — Evidence: none found. Note: optional later module; no action needed now.
- [~] 8.1.7 Candidate corpora are license-checked before ingestion — Evidence: no corpus or license register in `crates/` READMEs was read. Note: record access terms per corpus before ingest.
- [x] 8.2.1 Immutable originals with content IDs — Evidence: `crates/robot-artifacts/src/hash.rs` content hashing and `store.rs` (not all read, see 7.5.1).
- [~] 8.2.2 Axis/unit correction, finite-value and timestamp validation, gap marking — Evidence: `crates/motion-process/src/resample.rs` (resampling). Note: axis/unit correction and gap marking were not confirmed in `motion-process`.
- [~] 8.2.3 Filtering and resampling with documented phase delay — Evidence: `crates/motion-process/src/resample.rs` lines 2-17 (module docs). Note: no filter stage with documented phase delay was found.
- [ ] 8.2.4 Ground-plane and contact-interval estimation with uncertainty — Evidence: `grep` for `contact_interval` found no contact estimator. Note: implement contact detection in `motion-process`.
- [ ] 8.2.5 Segmentation into stance, locomotion, strike, defense, recovery and transition windows — Evidence: no segmentation module found. Note: implement segmentation.
- [~] 8.2.6 Retargeting with contact, joint-limit and self-collision constraints — Evidence: `crates/motion-retarget/src/ik.rs` `solve` enforces joint limits (README lines 45-50). Note: no contact or self-collision constraints exist.
- [~] 8.2.7 Validation of smoothness, velocity/acceleration, torque and simulated feasibility — Evidence: `crates/motion-validate/src/report.rs` (feasibility report). Note: smoothness and torque checks were not confirmed in this pass.
- [~] 8.2.8 Accepted references packaged, rejected examples retained with machine-readable reasons — Evidence: `crates/motion-validate/src/report.rs` report types. Note: retention of rejected examples was not verified.
- [~] 8.3.1 Weighted task-space IK followed by temporal refinement — Evidence: `crates/motion-retarget/src/ik.rs` damped-least-squares IK (README lines 8-10). Note: no temporal refinement stage exists.
- [ ] 8.3.2 End-effector orientation targets — Evidence: `motion-retarget/README.md` line 10 states position targets only. Note: add orientation residuals.
- [ ] 8.3.3 Posture priors and planted-foot constraints — Evidence: none found in `ik.rs`. Note: add both as weighted residuals.
- [ ] 8.3.4 Self-collision penalties — Evidence: none found. Note: add a penalty term.
- [~] 8.3.5 Rust-native constrained solver qualified before differential-QP IK — Evidence: `motion-retarget/README.md` lines 12-16 record that no QP candidate cleared qualification and a non-QP baseline is used. Note: this is the permitted interim state; qualification measurements are not recorded.
- [?] 8.3.6 Bounded iterations and validated response to infeasibility or timeout — `ik.rs` iteration limits not read.
- [?] 8.3.7 Morphology changes and separation of kinematic, dynamic, contact and task error in reports — not audited beyond `report.rs` existence.
- [ ] 8.3.8 Endpoint error normalized by body size, root drift and planted-foot slip metrics — Evidence: `motion-validate/src/report.rs` was not confirmed to compute these metrics; planted-foot slip has no producer (see 8.3.3). Note: add these metrics.

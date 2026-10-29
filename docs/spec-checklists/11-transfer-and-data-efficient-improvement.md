## 11. Transfer and data-efficient improvement

### 11.1 What can transfer

Human motion representations, task descriptions, motion segmentation, skill labels, strategy structure and some learned priors can transfer across robots. Low-level control typically requires target-specific retargeting and simulated training or adaptation. Cross-embodiment policies require explicit morphology conditioning, joint masks/graphs, multiple training embodiments and held-out robot evaluation.

The default transfer path is:

1. Train or curate generalized human motion representations.
2. Build and calibrate the target robot description.
3. Retarget references and reject or alter infeasible motions.
4. Train a target-specific tracker in calibrated, randomized simulation.
5. Qualify its inference artifact and observation pipeline.
6. Run replay, shadow mode, supported motion and noncontact deployment.
7. Collect targeted evidence of model mismatch.
8. Update the simulator or train a bounded adapter offline.
9. Re-evaluate against the frozen baseline and promote with rollback.

### 11.2 Sensor use

Joint encoders and IMU support state estimation, contact inference and history-conditioned adaptation. Current, temperature, measured torque and foot force sensors improve identification where available. External tracking can provide independent root-motion ground truth but is optional for deployment. Opponent perception supplies tracked pose/range/velocity with uncertainty and age; a tactical policy must handle occlusion and missing detections.

The estimation module in `robot-perception` must specify orientation/error-state conventions, IMU bias dynamics, contact assumptions, covariance propagation and measurement gating. Qualify filters under timestamp jitter, dropouts, outliers, slip and poorly conditioned observations. Compare observable-state error and covariance consistency against analytic or synthetic truth; no universal percentage-divergence threshold is assumed. Autodiff Jacobians are optional and require validation, as do hand-derived Jacobians.

Closed-loop control uses sensor feedback every tick without changing model weights. This is distinct from online adaptation of latent state, parameter estimation and gradient-based policy updates.

### 11.3 Limited-data adaptation strategies

| Method | Data / deployment contract | Principal limitation |
| --- | --- | --- |
| System identification | Time-synchronized commands and measured responses; update simulator parameters offline | Parameters may not be identifiable from passive logs |
| History-conditioned adaptation | Train encoder in randomized simulation; infer latent dynamics from recent real history | Can extrapolate poorly outside training ranges |
| Simulator correction model | Learn a model explaining real-vs-sim transition mismatch; retrain policy in corrected simulation | Requires well-defined alignment and validation of correction |
| Bounded robot-side residual | Train correction offline using deployable observations only; constrain authority | Can destabilize the base policy or hide sensor errors |
| Small adapter fine-tuning | Update selected policy parameters with held-out real-data evaluation | Few parameters do not imply few required samples |
| Two-phase latent identification (RMA-class) | Phase 1 trains a privileged-conditioned policy in randomized simulation; phase 2 trains a small adaptation module to regress that latent from a recent proprioceptive window, and only the module's output enters the deployed observation | A regressed latent is only as good as the randomization ranges it was fitted over; it reports a plausible context, not a measured parameter, and it fails silently outside them |
| Simulation-guided fine-tuning | Real-world updates shaped by a value or objective term derived from simulation rather than by unguided real exploration | Requires on-hardware updating, which Section 4 permits only under an explicitly enabled mode; the guiding term inherits every error of the simulator that produced it |
| Retrieval-conditioned adaptation | Select prior episodes similar to the observed deployment condition, per Section 9.7, and adapt on the union with the few real episodes | Similarity in observation space is not similarity in dynamics; the admission filter, not the metric, is what makes it safe |
| Cross-embodiment few-shot transfer | Pretrain over several embodiments with morphology conditioning, then adapt to a held-out robot from a small sample | Section 11.1's requirements apply in full; a held-out-robot result is evidence about the pretraining set, and a single target robot is not a transfer claim |
| Multi-simulator or multi-model randomization | Randomize across distinct dynamics models rather than only over parameters of one | Requires a second qualified simulator, which Section 9.1 does not currently grant; otherwise it is parameter randomization with extra vocabulary |

Implement identification first, then history adaptation; treat residual RL and online weight updates as later experiments. Do not require paired perfect simulated/real state at execution. Do not assume a unique target action correction can be inferred from state error alone.

The rows above are ordered by increasing hardware exposure, and that ordering is binding: a method MUST NOT be scheduled before the methods above it have been measured and found insufficient on the same robot. Each method additionally carries the following obligations.

1. Each adaptation method is an extension under Sections 9.3 and 9.6, with its own manifest, feature flag and conformance run. Adaptation is not a special case exempt from the extension rules.
2. Every method MUST declare the observation set available at deployment and MUST NOT consume privileged simulator state at inference. This is the existing Section 9.2 audit applied to adapters, and it is the specific defect Section 3 records in the earlier residual sketch.
3. Any component that runs on the robot — an adaptation module, a context encoder, a residual head — is a tier under Section 5.1 with a declared rate, a measured budget and a defined behavior when its input history is short, stale or discontinuous. An adaptation module that has not yet filled its history window MUST produce the declared default context, not an extrapolation.
4. Adaptation authority MUST be bounded and the bound enforced outside the adapted component. A learned correction is clamped by the supervisor of Section 12.1 like any other command, and widening a bound to accommodate an adapter is a safety-profile change, not a tuning step.
5. Every method MUST be evaluated against the frozen pre-adaptation baseline on the held-out session, reporting both improvement and regression, and MUST support rollback to that baseline as an immediately available artifact under Section 13.
6. Sample-efficiency claims MUST be stated in the units that cost something: real minutes, real episodes, physical interventions and wall-clock downtime. Simulation steps consumed by the method belong in the same report, because a method that needs a week of simulation to save ten minutes of hardware time has moved the cost rather than removed it.
7. On-hardware weight updates remain a later experiment and require the mode gating of Section 4 plus a separate decision; nothing in this table authorizes continuous learning on a robot.

Use event-triggered logging of uncertainty, slip, saturation, tracking error and contact anomalies, plus representative successful background samples. This avoids retaining every high-rate stream while limiting failure-selection bias. Select calibration motions for information gain subject to the operating envelope. Keep a held-out session for validation and compare learning curves versus real minutes, transitions and interventions.

Stop collecting or adapting when improvement saturates or uncertainty criteria are met; do not promise that 20–30 minutes is universally sufficient. Real interaction count, physical interventions, data storage and simulator compute are separate efficiency metrics.

## Implementation checklist

Status legend: `[x]` implemented, `[~]` partial, `[ ]` not implemented, `[?]` not audited in this pass. Summary: 4 `[x]`, 9 `[~]`, 4 `[?]`, 14 `[ ]` across 31 items.

- [~] 11.1.1 Explicit morphology conditioning, joint masks/graphs and held-out robot evaluation for cross-embodiment policies — Evidence: `crates/robot-model/src/profile.rs` declares per-robot profiles. Note: no morphology-conditioning input or held-out-robot evaluation found.
- [x] 11.1.2 Default transfer path steps 1 to 4: representation, target description calibration, retargeting with infeasible-motion rejection, target-specific tracker training — Evidence: `crates/robot-model` (description), `crates/motion-retarget` (retargeting, `crates/motion-validate` rejection report), `crates/robot-learn` (training). Note: steps 1 and 4 are partial (see 8.1 and 9.2).
- [ ] 11.1.3 Qualify the inference artifact and observation pipeline before deployment (step 5) — Evidence: no qualification gate for the inference artifact found; `robot-artifacts` holds manifests but no qualification record. Note: add a qualification step with a recorded result.
- [~] 11.1.4 Replay, shadow mode, supported motion and noncontact deployment (step 6) — Evidence: `crates/robot-hardware/README.md` covers loopback and replay devices; `hw-shadow` in `crates/robot-control/src/control_loop.rs` (see 4.10). Note: supported-motion and noncontact deployment staging not confirmed.
- [ ] 11.1.5 Collect targeted evidence of model mismatch (step 7) — Evidence: no mismatch-evidence collector found. Note: implement event-triggered logging per the closing paragraph.
- [~] 11.1.6 Update the simulator or train a bounded adapter offline (step 8) — Evidence: `crates/robot-adapt/README.md` identifies only the first row of the table is implemented. Note: no bounded adapter exists.
- [ ] 11.1.7 Re-evaluate against the frozen baseline and promote with rollback (step 9) — Evidence: no frozen-baseline re-evaluation or rollback artifact found. Note: implement rollback as an artifact under Section 13.
- [~] 11.2.1 Joint encoders and IMU used for state estimation, contact inference and history-conditioned adaptation — Evidence: `crates/robot-perception/src/attitude.rs` (IMU attitude filter), `crates/robot-perception/src/contact.rs` (contact classifier). Note: history-conditioned adaptation is absent (see 11.3.3); encoder use not confirmed.
- [~] 11.2.2 Estimation module specifies orientation/error-state conventions, IMU bias dynamics, contact assumptions, covariance propagation and measurement gating — Evidence: `crates/robot-perception/README.md` lines 14-18 and `attitude.rs` module docs; the README says the module names each requirement. Note: contact assumption is explicitly absent (see 11.2.4).
- [x] 11.2.3 Covariance propagation and Mahalanobis measurement gating — Evidence: `crates/robot-perception/src/linear_kf.rs` (predict, Mahalanobis gating, Joseph-form update, `robot-perception/README.md` line 10).
- [~] 11.2.4 Contact assumption specified and wired into the estimator's gate — Evidence: `crates/robot-perception/README.md` lines 40-50 state the attitude gate uses specific-force magnitude, not contact. Note: `ContactClassifier` in `contact.rs` is not wired into the attitude gate.
- [?] 11.2.5 Filter qualification under timestamp jitter, dropouts, outliers, slip and ill-conditioned observations — not audited; `robot-perception` tests were described (15 tests) but the robustness cases were not read.
- [~] 11.2.6 Observable-state error and covariance consistency compared against analytic or synthetic truth — Evidence: `robot-perception/README.md` lines 54-61 (constant-rate and convergence checks). Note: covariance consistency (NEES/NIS-style) is not checked; only analytic and finite-difference checks are present.
- [x] 11.2.7 Jacobians validated by central finite differences — Evidence: `crates/robot-perception/src/attitude.rs` `measurement_jacobian_matches_central_finite_differences` (README lines 26-38).
- [ ] 11.3.1 Closed-loop sensor feedback every tick without changing model weights, distinct from online adaptation — Evidence: no closed-loop feedback path distinct from weight updates was located. Note: state the distinction in the control module and confirm the tick path.
- [x] 11.3.2 System identification as the first adaptation method — Evidence: `crates/robot-adapt/src/sysid.rs` `identify` (ridge least squares, singular values, `robot-adapt/README.md` lines 12, 14-15).
- [ ] 11.3.3 History-conditioned adaptation (encoder trained in randomized simulation, latent inferred from recent history) — Evidence: `robot-adapt/README.md` lines 16-17 states it is unimplemented. Note: implement after identification is measured insufficient.
- [ ] 11.3.4 Simulator correction model, retraining policy in corrected simulation — Evidence: `robot-adapt/README.md` line 16-17. Note: implement after row 2 is measured insufficient.
- [ ] 11.3.5 Bounded robot-side residual trained offline with deployable observations only — Evidence: `robot-adapt/README.md` lines 16-17 (bounded residuals listed as unimplemented). Note: implement with a supervisor-enforced bound.
- [ ] 11.3.6 Small adapter fine-tuning with held-out real-data evaluation — Evidence: `robot-adapt/README.md` lines 16-17. Note: implement after earlier rows.
- [ ] 11.3.7 Two-phase latent identification (RMA-class) with the adaptation output as the only privileged-derived input — Evidence: `robot-adapt/README.md` lines 16-17. Note: implement after earlier rows.
- [?] 11.3.8 Simulation-guided fine-tuning on hardware under an enabled mode — not applicable in this pass; depends on mode gating of Section 4.
- [?] 11.3.9 Retrieval-conditioned adaptation — not applicable until 9.7.1 exists.
- [?] 11.3.10 Cross-embodiment few-shot transfer and multi-simulator randomization — not applicable; requires morphology conditioning (11.1.1) and a second qualified simulator (Section 9.1).
- [ ] 11.4.1 Each adaptation method declares its deployment observation set and consumes no privileged state at inference — Evidence: no adaptation method other than sysid exists to audit. Note: add the declaration to each new method.
- [ ] 11.4.2 A module that has not filled its history window produces the declared default context, not an extrapolation — Evidence: no history-window module exists. Note: add with 11.3.3.
- [~] 11.4.3 Adaptation authority is bounded, with the bound enforced outside the adapted component by the supervisor — Evidence: the supervisor of Section 12.1 exists in `crates/robot-supervisor/src/supervisor.rs`. Note: no learned-correction clamp is wired to the supervisor.
- [ ] 11.4.4 Every method evaluated against the frozen baseline on the held-out session, with rollback to that baseline as an artifact — Evidence: no evaluation hook or rollback artifact for adaptation found. Note: implement with 11.1.7.
- [ ] 11.4.5 Sample-efficiency claims reported in real minutes, episodes, interventions and wall-clock downtime — Evidence: no adaptation report with these units found. Note: add a report template.
- [ ] 11.4.6 On-hardware weight updates gated by Section 4 and a separate decision — Evidence: no on-hardware weight-update path exists. Note: none needed now; maintain this as a prohibition.
- [~] 11.4.7 Event-triggered logging of uncertainty, slip, saturation and contact anomalies with representative background samples — Evidence: `crates/robot-types/src/episode.rs` and `crates/robot-artifacts` provide the record format. Note: no event-trigger logic was found.


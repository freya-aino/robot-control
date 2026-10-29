## 12. Safety, supervision, perception and personality

A multi-kilogram humanoid that throws punches stores enough energy to injure people and destroy itself. The requirements below are the minimum structure; the numeric limits are per-robot and live in a versioned safety profile alongside the acceptance profile of Section 14.

### 12.1 Supervisor

A supervisor independent of strategy and personality checks observation freshness, finite commands, position/velocity/torque-speed limits, temperatures, communications health, contact permissions and current operating mode. Hardware emergency stop remains independent of the learning stack. Clamp-and-continue is not sufficient for every fault; each fault has a validated response appropriate to the robot.

The supervisor MUST be the last stage before the actuator interface, MUST be reachable on every path including teleoperation and scripted control, and MUST NOT import the policy, retrieval or personality crates. Its checks run within the control tick and are subject to the same prohibitions as Section 5: no allocation, no blocking, no logging flush. It MUST be testable in isolation against recorded and synthetic fault traces, and its rejection decisions MUST be recorded with enough context to reconstruct why a command was refused.

### 12.2 Hazard register

Maintain a hazard register per robot and per operating mode. Each entry names the hazard, its plausible causes, how it is detected, the control that mitigates it, and the residual risk accepted by whom. The following are the initial entries; they are a starting structure, not a completed analysis, and a hazard analysis for a specific robot MUST be performed before that robot first moves under `hw-freestanding-noncontact`.

| Hazard | Principal causes | Detection | Control |
| --- | --- | --- | --- |
| Uncommanded motion | Stale command held by firmware, watchdog misconfiguration, mode confusion | Command expiry counters, independent watchdog, transmit-disable readback | Short command expiry in firmware; watchdog outside the policy process; mode asserted every tick |
| Fall | Loss of balance, actuator saturation, late or failed skill transition | Attitude and contact estimation, deadline misses, recovery invocation count | Protective response from the robot's safety profile; area clear of obstacles and people |
| Excess contact energy | Reward exploitation, scoring bug, mis-scaled action, opponent model error | Impulse and force measurement, energy budget accounting, independent match logging | Per-target permissions, energy budget enforced outside the policy, padded fixtures |
| Actuator overload | Infeasible reference, torque-speed envelope violation, thermal accumulation | Temperature, current, saturation duration, torque-speed margin | Envelope enforcement in the supervisor; reference rejection in Section 8.3; thermal derating |
| Pinch and entanglement | Cables, gloves, straps, support fixtures within swept volume | Pre-run checklist, collision geometry including attachments | Swept-volume check with attachments modeled per Section 12.4 |
| Human enters the area | Procedural failure, unexpected access, observer approach | Area interlock, observer role, perception detection as a secondary signal | Immediate controlled stop under Section 12.3; perception is never the primary control |
| Loss of transport | Cable, radio, switch, host stall | Link heartbeat, arrival-time gaps, deadline misses | Firmware-side command expiry; the robot must be safe when the host disappears |
| Wrong artifact deployed | Bundle/profile mismatch, firmware change, payload change | Compatibility validation at load; checksum and signature | Refuse to activate; last-known-good bundle available immediately |

A hazard whose only control is "the policy learned not to do that" is not controlled. Controls MUST be independent of the learned component they constrain.

### 12.3 Area, access and human exclusion

Hardware motion beyond `hw-shadow` requires a defined physical area with controlled access, a nominated operator holding the emergency stop, and no person inside the robot's reachable volume plus a fall margin. Entry during motion is a stop condition, not a task input. Perception-based human detection MAY contribute as a secondary signal; it MUST NOT be the primary control, because the perception stack is exactly the component whose failure modes this control exists to survive.

Emergency stop MUST remain functional with the host powered off, the policy process hung, and the transport disconnected. Test this, rather than assuming it from a wiring diagram.

### 12.4 Equipment, payload and contact envelope

Account for gloves, padding, cables and payloads in collision geometry and dynamics. Match rules and allowed contact envelopes are configuration, with separate permissions for motion demonstrations and contact trials. A change of glove, padding, payload or firmware changes the dynamics and the swept volume: it invalidates the affected qualification and requires requalification under Section 11 before the robot returns to its previous mode.

### 12.5 Personality

Personality is optional and operates through bounded intent: posture style, cadence, idle motion, celebrations, speech timing and tactical preferences. Persistent identity and opponent memory remain outside the motor loop. Expressive actions use the same validated skills and transitions; they cannot override balance recovery or supervisor decisions. Test whether style is recognizable without degrading task and stability metrics.

Expression MUST NOT widen any envelope, extend any dwell time against a recovery request, or introduce a motion that has not passed the same validation as a task skill. A celebration is a skill with a transition contract, not an exemption.

Opponent memory, where it is implemented, is a persistent store written between rounds or between bouts and read at strategy rate. Appendix B.2.13 identifies a candidate mechanism — a versioned entity/observation graph — for what Section 3 previously recorded only as an unverified persistent-identity sketch. Any such store is subject to three constraints: its read path MUST NOT depend on network availability or on a hosted backend, so an embedded or local backend is required wherever it is consulted during operation; its contents are an input to strategy, never to the supervisor or the motor loop; and a store that is unavailable, empty or stale MUST degrade to the opponent-agnostic behavior rather than stalling a decision. Retention, consent and deletion obligations for recorded opponent data follow Section 7.5.

Where personality is expressed as a set of style parameters and expressive-skill choices, the candidates for that set MAY be proposed by the offline assisted sampling of Section 9.5. A personality is then a sampled, versioned configuration over already-validated skills, recognizable and testable under the metrics above. It is not a generated behavior: no runtime model call produces posture, cadence, celebration or tactical preference, and a style variation that has not passed the same validation as a task skill is not available to be expressed.

### 12.6 Agent-layer governance

If the optional agent layer of Section 6 is ever enabled, it is governed as a supervised input source and not as a controller:

1. A guardrail, schema validation or audit-logging component MAY be an additional, earlier check on an advisory record. It MUST NOT be described or relied upon as a safety control, and it MUST NOT be placed in the position of the supervisor of Section 12.1, which remains the last stage before the actuator interface and MUST NOT import it.
2. Under Section 12.2, a hazard whose only control is a model-produced judgment is not controlled. Controls MUST remain independent of the agent layer, exactly as they must remain independent of the learned policy.
3. Agent-layer outputs that reach an operator — advisories, commentary, speech — MUST be distinguishable in the record from measured state, so that a generated statement is never read back as an observation.
4. Every agent-layer decision, guardrail intervention and approval gate is a logged artifact under Section 13, with the model identity and version that produced it.
5. Human approval gates implemented through an agent runtime are a convenience over a procedure, not a substitute for one. Promotion to hardware follows Section 11.1 and Section 14 regardless of which tool records the approval.


## Implementation checklist

Status legend: `[x]` implemented, `[~]` partial, `[ ]` not implemented, `[?]` not audited in this pass. Summary: 5 `[x]`, 10 `[~]`, 9 `[ ]`, 2 `[?]` across 26 items.

- [~] 12.1.1 Supervisor checks observation freshness, finite commands, position/velocity/torque-speed limits, temperatures, communications health, contact permissions and operating mode — Evidence: `crates/robot-supervisor/README.md` lines 17-21 (ten fault categories), `supervisor.rs` `check_tick`. Note: temperature and torque-speed checks were not confirmed among the ten categories.
- [x] 12.1.2 Each fault has a validated response; clamp-and-continue is refused for safety-critical categories — Evidence: `crates/robot-supervisor/src/policy.rs` `FaultPolicy::new` constructor check (`robot-supervisor/README.md` lines 29-33).
- [x] 12.1.3 Supervisor is the last stage before the actuator interface — Evidence: `crates/robot-control/src/control_loop.rs` line 338 gates `send_command`; the supervisor check runs in the tick path (`control_loop.rs` `supervisor` call sites).
- [x] 12.1.4 Supervisor does not import the policy, retrieval or personality crates — Evidence: `crates/robot-supervisor/Cargo.toml` `[dependencies]` lists only `robot-types` and `robot-model`; `grep` for `robot-(learn|skills|persona|nn)` in the manifest found no matches.
- [~] 12.1.5 Supervisor checks run within the tick with no allocation or blocking — Evidence: `robot-supervisor/README.md` lines 37-47 (all `Vec`s sized in `Supervisor::new`; `check_tick` writes into caller slices). Note: no-blocking and no-logging-flush were not confirmed by reading the logging path.
- [~] 12.1.6 Supervisor testable in isolation against recorded and synthetic fault traces — Evidence: `robot-supervisor/README.md` lines 61-67 (21 tests). Note: no recorded fault-trace replay harness was found.
- [~] 12.1.7 Rejection decisions recorded with enough context to reconstruct the refusal — Evidence: `crates/robot-supervisor/src/fault.rs` `FaultLog<N>` (records each fault with its measurement and limit). Note: overflow counting exists; the completeness of per-command rejection context was not verified.
- [ ] 12.1.8 Hardware emergency stop independent of the learning stack — Evidence: no hardware E-stop interface found in `crates/robot-hardware`. Note: add an E-stop path that does not pass through the software loop.
- [ ] 12.2.1 Hazard register maintained per robot and per operating mode, with cause, detection, control and accepted residual risk — Evidence: no hazard register file found (`grep`/glob for `hazard` in `docs/` returned no register). Note: create the register.
- [~] 12.2.2 Hazard controls are independent of the learned component they constrain — Evidence: supervisor independence per 12.1.4; watchdog outside the policy process is not confirmed (see 5.9).
- [x] 12.2.3 Transmit-disable readback in hw-shadow — Evidence: `crates/robot-control/src/control_loop.rs` line 338 transmit gate and `robot-hardware/README.md` loopback device (see 4.10).
- [~] 12.2.4 Command expiry counters and firmware-side expiry — Evidence: `control_loop.rs` `command_expiration` (lines 565-566). Note: firmware-side expiry is not implemented in this repository; it is a robot-firmware requirement.
- [ ] 12.2.5 Watchdog independent of the policy process — Evidence: no watchdog module found. Note: implement a watchdog with its own timer.
- [ ] 12.2.6 Energy budget accounting enforced outside the policy — Evidence: no energy-budget module found. Note: implement enforcement in the supervisor.
- [ ] 12.2.7 Area interlock and observer-role handling — Evidence: no interlock or role module found. Note: operational control; record in the hardware setup document.
- [~] 12.2.8 Compatibility validation at load, checksum and signature, refusal to activate on mismatch, last-known-good bundle — Evidence: `crates/robot-artifacts/src/hash.rs` (checksum) and `robot-types/src/bundle.rs` bundle contract checks. Note: no signature scheme and no last-known-good retention were found.
- [ ] 12.3.1 Defined physical area with controlled access, nominated E-stop operator and no-person-inside check — Evidence: operational requirement; no area-interlock artifact in the repository. Note: document the procedure.
- [ ] 12.3.2 Emergency stop functional with host off, policy hung and transport disconnected — Evidence: no hardware E-stop found (see 12.1.8). Note: test on hardware; cannot be verified from the repository.
- [~] 12.3.3 Human-in-area entry handled as a stop condition, not a task input — Evidence: `crates/robot-types/src/mode.rs` `Interlock` enum lists area-related interlocks. Note: entry during motion was not confirmed as a stop trigger.
- [?] 12.4 Equipment, payload and contact envelope accounted for in collision geometry and requalification on change — not audited.
- [~] 12.5.1 Personality uses bounded intent over validated skills and cannot override recovery or the supervisor — Evidence: `crates/robot-persona/src/lib.rs` lines 7-11 (opponent memory not implemented, personality is a declared crate). Note: persona runtime behavior was not audited.
- [x] 12.5.2 Persistent identity and opponent memory kept outside the motor loop — Evidence: `robot-persona` is not a dependency of `robot-control` (see 6.4).
- [~] 12.5.3 Opponent store, where implemented, degrades to opponent-agnostic behaviour when unavailable — Evidence: no opponent store exists (see 9.4.3). Note: implement the degrade path with the store.
- [ ] 12.5.4 Personality parameter sets sampled offline under Section 9.5 as versioned configuration — Evidence: no personality-configuration sampler found (see 9.5.1). Note: implement after 9.5.
- [ ] 12.6.1 Optional agent layer is never placed in the supervisor position and does not import the supervisor — Evidence: no agent-layer crate found in `crates/`. Note: enforce in the dependency gate when added.
- [?] 12.6.2 to 12.6.5 Agent-layer advisory distinction, logging, approval gates — not applicable; no agent layer is present in the crates.

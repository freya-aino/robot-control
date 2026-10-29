## 4. Operating modes and boundaries

Support deterministic offline replay, simulation training, simulation evaluation, hardware shadow mode, supported hardware motion, free-standing noncontact motion, and separately enabled robot-to-robot contact experiments. Shadow mode consumes live sensor data and computes actions without transmitting motor commands. Teleoperation and scripted controllers are first-class baselines.

Each mode has explicit permitted actions, hardware interlocks, timeout behavior, and artifact requirements. The mode is a property of the running process, not a configuration comment: it MUST be recorded in every episode record, MUST be reported by the supervisor on request, and MUST NOT change without passing back through the entry preconditions of the target mode.

| Mode | Actuation authority | Entry preconditions | Required interlocks | Loss-of-input or fault response |
| --- | --- | --- | --- | --- |
| `offline-replay` | None; no device attached | Artifact hashes resolve; bundle schema matches recorded observations | None | Abort with a diagnostic; never fabricate missing samples |
| `sim-train` | Simulated only | Seeds, config and data manifest pinned | None | Fail the run; partial results are labeled as such |
| `sim-eval` | Simulated only | Frozen bundle; evaluation seeds disjoint from training | None | Fail the suite rather than reporting a partial pass |
| `hw-shadow` | None; commands computed, logged, and discarded at the adapter | Transmit path disabled and the disable verified by loopback readback; e-stop armed | E-stop; transmit-disable assertion checked every tick | Continue computing; record the fault; never enable transmit as a fallback |
| `hw-supported` | Motion with the robot mechanically supported | Joint sign/offset/units verified per Section 7.1; reduced torque and velocity envelope loaded | E-stop; support-fixture present; reduced envelope enforced by the supervisor | Robot-specific hold or controlled stop per its safety profile |
| `hw-freestanding-noncontact` | Full motion, no contact targets | `hw-supported` qualification passed; area cleared; exclusion zone enforced | E-stop; area interlock; envelope; watchdog independent of the policy process | Validated fallback of Section 10.4; falling is a designed outcome, not a fault path |
| `hw-contact` | Motion against an instrumented fixture or robot | Separate enablement; noncontact qualification passed; scoring validated; contact energy budget set | All of the above plus per-target contact permission and an independent match logger | Terminate the bout, not just the skill; record the full pre/post-event buffer |
| `teleop` | Operator-commanded, mode-scoped | Same preconditions as the hardware mode it runs within | Same interlocks; operator presence control where the robot requires one | Command expiry stops motion; stale operator input is not held |
| `scripted` | Deterministic scripted controller | Same preconditions as the hardware mode it runs within | Same interlocks | Same as the enclosing hardware mode |

Teleoperation and scripted control are modifiers on a hardware mode, not escapes from it: they are subject to the same envelope, supervisor and artifact requirements, and an operator MUST NOT be able to command outside the envelope that a learned policy would be clamped to.

Autonomous contact with people is outside this project's boxing scope. Contact experiments involve instrumented robots or test fixtures in a controlled area. This boundary affects task definitions, target identification, and deployment interlocks. A human entering the area is an exclusion-zone violation handled by Section 12.3, never a target.



## Implementation checklist

Status legend: `[x]` implemented, `[~]` partial, `[ ]` not implemented, `[?]` not audited in this pass. Summary: 2 `[x]`, 5 `[~]`, 0 `[ ]`, 5 `[?]` across 12 items.

- [x] 4.1 The seven operating modes are enumerated — Evidence: `crates/robot-types/src/mode.rs` lines 17-32 (`OperatingMode`).
- [~] 4.2 Each mode has permitted actions, interlocks, timeout behaviour and artifact requirements — Evidence: `mode.rs` `authority()` (line 271), `required_interlocks()` (line 284) and `fault_response()` (line 344). Note: artifact requirements per mode were not found in `mode.rs`; confirm they are encoded or add them.
- [?] 4.3 The mode is recorded in every episode record — `crates/robot-types/src/episode.rs` references the mode but was not read in this pass.
- [~] 4.4 The supervisor reports the current mode on request — Evidence: `crates/robot-supervisor/src/supervisor.rs` lines 47-51 (`TickInput.mode`). Note: no on-request report query was found; the mode reaches the supervisor only as tick input.
- [~] 4.5 The mode changes only after the target mode's entry preconditions pass — Evidence: `mode.rs` `check` (lines 481-483) enforces prerequisites. Note: `crates/robot-control/src/control_loop.rs` sets `mode` once at construction (lines 593-595, 696-698) and has no transition function that calls `check`; wire the entry check into any mode-change path.
- [x] 4.6 Teleoperation and scripted control are modelled as modifiers, not modes — Evidence: `mode.rs` header (lines 9-11) and the separate `ControlSource` type.
- [?] 4.7 Teleoperation and scripted control are bound by the same envelope and supervisor as the enclosing hardware mode — not audited in this pass.
- [?] 4.8 Teleoperation command expiry stops motion and stale operator input is not held — `control_loop.rs` sets `command_expiration` (lines 565-566); the expiry path for teleop was not read.
- [?] 4.9 A human entering the exclusion zone is handled as a Section 12.3 violation, not as a target — not audited in this pass.
- [~] 4.10 `hw-shadow` disables transmit and verifies it by loopback readback each tick — Evidence: `crates/robot-control/src/control_loop.rs` line 338 gates `send_command` on `mode.authority() != None`. Note: no per-tick loopback readback of the transmit-disable assertion was found in this pass; add the readback check and a test for it.
- [?] 4.11 A `hw-shadow` fault continues computation, records the fault and never enables transmit as a fallback — not audited in this pass.
- [~] 4.12 `hw-supported` enforces a reduced torque and velocity envelope — Evidence: `Interlock::ReducedEnvelope` and `EntryPrecondition::ReducedEnvelopeLoaded` exist in `crates/robot-types/src/mode.rs` (lines 73, 186). Note: `allowed_modes` at `crates/robot-control/src/control_loop.rs` line 568 is a test fixture only, so no runtime envelope enforcement was found; wire the reduced envelope into the supervisor's limit check.

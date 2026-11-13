# robot-control

The deterministic control runtime of SPECS.md §5, built from this
workspace's already-qualified pieces. Depends on `robot-types`,
`robot-model`, `robot-nn`, `robot-hardware`, `robot-supervisor`,
`robot-perception` and `robot-skills` (now both used).

## What's here

| Module | Contents |
| --- | --- |
| `control_loop` | `ControlLoop<A, C, N, P = Mlp, O = NoOp>`: poll → infer → supervise → transmit, built once, ticked repeatedly; generic over a `Policy` and a `LoopOperation`, with construction-time width validation (`ConfigError`) |
| `operation` | `LoopOperation<P>` hooks (`check_widths`, `before_tick`, `select_action`, `after_tick`), `NoOp`, `SamplingToggle` (deterministic or sampled action), `ExperienceRecorder` (fixed-capacity ordered transition buffer), and tuple composition |
| `timing` | `TickTiming` (the four distinct quantities of §5), `TimingHistory<N>` (fixed-capacity ring), and `TimingReport`: p50/p95/p99/p99.9/max of compute duration, sensor-to-command age, decision age and deadline margin, each separately, with misses |
| `tactical` | The tactical tier: `TacticalOperation` runs `robot_skills::selection::Selector` at its own period and publishes timestamped, expiring `Proposal`s. `SkillBank` holds resident experts and runs the proposed skill's expert, or the declared default with no valid proposal. `TacticalConfig::validate` enforces §5.1's tier rules |
| `incident` | `IncidentRecorder<PRE, POST>`: §5's reserved safety-event channel, fixed pre/post windows, triggers during a capture dropped and counted, never blocking |

## Allocation discipline

Every `Vec` `ControlLoop` owns (`feedback`, `normalized`, `raw_action`,
`command`) is sized once in `ControlLoop::new`, from the policy network's
own input/output widths. `ControlLoop::tick` writes into these buffers —
taking the `Policy::act` path (or the installed
`LoopOperation::select_action` path) and then calling
`robot_supervisor::supervisor::Supervisor::check_tick`; the loop itself
allocates nothing on the tick path. Custom policies and operations must
also honor the allocation-free contract. `ControlLoop::new`
takes one `ControlLoopConfig` struct rather than eight positional
parameters; that refactor was clippy's `too_many_arguments` lint, not
hand-wringing about style, but the result is also just a better call site.

## The two tiers

The **policy tier** handles inference, the supervisor check and
transmission, with real sensor-to-command age and deadline-margin
accounting, exercised end to end against
`robot_hardware::loopback::LoopbackDevice`.

`with_sensors` wires **temperature and contact feedback** into the
supervisor. Temperature taps index actuator temperatures in the feedback
vector. Contact taps index each site's normal force, classified with
hysteresis by `robot_perception::contact::ContactClassifier`. Before this,
those two supervisor checks were passed empty lists. A test shows an
over-temperature reading reaching the supervisor's fault log.

The **tactical tier** is `TacticalOperation`, a `LoopOperation` over a
`SkillBank`. The loop gains two default hooks, `observe_time` and
`decision_age`, so the operation schedules its router on the loop's own
clock and the tick's `TickTiming::decision_age` reports the age of the
proposal applied, attributed to the tier that produced it (§5.1 item 4).
§5.1's rules are checked at construction:
- **Rates:** the tactical period may not be shorter than the policy period.
- **Proposal lifetime:** it may not be shorter than one tactical period.
- **Degradation:** a late, absent or expired proposal runs the declared
  default expert. That is item 3's hold-or-default behaviour, and item 7's
  "removing the highest tier leaves a safe controller".

In `hw-shadow` the loop disables the adapter's transmit at construction,
verifies by readback every tick that it is still disabled, and sends
nothing. A tick that finds transmit enabled fails with
`TickError::TransmitEnabledInShadow` and never re-enables or disables the
flag itself (SPECS.md §4).

## What is not here

- **The opponent-forecast half of the tactical tier.** It belongs to the
  strategy work of §9.4.
- **A measured qualification run** on designated hardware (row A-15).
  The report type is here; the numbers need the robot.

## A transmit-gating bug caught before it shipped

An early version of `tick` decided whether to call `send_command` with a
condition built from `Response != EmergencyStop || mode.authority()...` —
backwards from what the accompanying comment described, and not actually
equivalent to "transmit unless no device is attached." It's now a direct
check on `mode.authority() != ActuationAuthority::None`, matching SPECS.md
§12.2's point that a zero-torque command is itself the safe state and
should still be sent, rather than withheld, so the actuator never holds a
stale prior command.

## Testing

`cargo test -p robot-control`: 46 tests. 36 are the original ones: hw-shadow transmit readback (disabled at construction, nothing sent, a re-enabled flag is a tick fault), loopback feedback and deterministic
timing, policy/operation plug-in and composition behavior, buffer reuse,
construction-time width validation, safe supervisor escalation for
non-finite observations and policy errors, and adapter-error precedence.
The new ones cover:
- the router running at its own rate and selecting the expert;
- an expired proposal falling back to the default expert;
- tier rules checked at construction;
- the bank refusing mismatched experts;
- the tactical tier running inside the loop with a decision age;
- a temperature tap reaching the supervisor;
- contact hysteresis and a refused tap;
- the four-quantity report;
- incident windows and dropped triggers.

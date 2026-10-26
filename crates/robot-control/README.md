# robot-control

The deterministic control runtime of SPECS.md §5, built from this
workspace's already-qualified pieces. Depends on `robot-types`,
`robot-model`, `robot-nn`, `robot-hardware`, `robot-supervisor`,
`robot-perception` and `robot-skills`.

## What's here

| Module | Contents |
| --- | --- |
| `control_loop` | `ControlLoop<A, C, N>`: poll → infer → supervise → transmit, built once, ticked repeatedly |
| `timing` | `TickTiming` (the four distinct quantities of §5) and `TimingHistory<N>`, a fixed-capacity ring for a percentile report |

## Allocation discipline

Every `Vec` `ControlLoop` owns (`feedback`, `normalized`, `raw_action`,
`command`) is sized once in `ControlLoop::new`, from the policy network's
own input/output widths. `ControlLoop::tick` writes into these buffers —
via `robot_nn::mlp::Mlp::forward` and `robot_supervisor::supervisor::
Supervisor::check_tick`, both already zero-allocation by their own
crates' design — and allocates nothing of its own. `ControlLoop::new`
takes one `ControlLoopConfig` struct rather than eight positional
parameters; that refactor was clippy's `too_many_arguments` lint, not
hand-wringing about style, but the result is also just a better call site.

## What this revision does and does not implement

Implemented: the **policy tier** — inference, the full supervisor check
(finite/limit/mode/contact), and transmission — with real sensor-to-command
age and deadline-margin accounting, exercised end to end against
`robot_hardware::loopback::LoopbackDevice`.

Not implemented, and said so in `control_loop`'s own module doc rather than
left to be discovered: the **tactical tier**. `robot-skills` is a declared
dependency, but `ControlLoop::tick` does not yet query a
`robot_skills::tiers::Router` to select among skills, so there is no
two-tier `decision_age` to report yet — `TickTiming::decision_age` is
always `None` in this revision. SPECS.md §5.1's tier-rate and
per-tier-budget machinery lives in `robot_types::tier` and is exercised
there; wiring a second tier into this loop is the next integration step,
not a claim this crate makes today.

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

`cargo test -p robot-control` — 7 tests: a tick with no feedback yet still
produces a command without panicking, feedback through a loopback device is
consumed on the following tick, the deadline margin matches the configured
budget under a stepped (deterministic) clock, and `TimingHistory`'s ring
wrap, percentile and missed-deadline-count behavior.

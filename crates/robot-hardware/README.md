# robot-hardware

Device traits, loopback and replay devices, and the protocol-adapter
boundary of SPECS.md §7.1. Depends only on `robot-types`.

## What's here

| Module | Contents |
| --- | --- |
| `device` | `HardwareAdapter`: the trait every device (simulated or real) implements; `AdapterError`; `FeedbackHeader` |
| `loopback` | `LoopbackDevice`: commands become feedback after a configurable, per-command, independently-counted poll delay |
| `replay` | `ReplayDevice`: deterministic playback of a pre-loaded frame sequence; never actuates; exhaustion is a distinct terminal error, never a repeat |

PLAN.md Phase 5: "Implement a hardware adapter first against a loopback
device, then a replay device, then an actual documented protocol." This
crate delivers the first two. No protocol-specific adapter exists yet: C-02
(first hardware target) is open, and SPECS.md §1.1 requires any future
adapter transcribing the retired `unitree/` DDS/SDK reference to treat every
field as "a hypothesis requiring hardware verification" rather than a port.

## The loopback delay model

Each `send_command` call enqueues an entry with its own countdown
(`delay_ticks + 1`). Every `poll_feedback` call ticks *every* pending
entry's countdown in parallel — not just the one at the front — so that two
commands sent back to back surface independently rather than the second
waiting for the first to fully clear the queue. This was not the first
implementation: an earlier version decremented only the front entry and
failed a two-in-flight test outright (`two_sends_surface_in_order_each_after_its_own_delay`),
which is why that test stays in the suite as the regression it caught.

`LoopbackDevice` is explicitly *not* a zero-allocation implementation
(`send_command` copies into an owned `Vec` each call): it exists to exercise
the runtime's own plumbing during development, and its own doc comment says
so. The allocation-free guarantee SPECS.md §5 requires belongs to the
eventual protocol adapter built against real hardware, not to this test
double.

## What `hw-shadow` actually depends on here

SPECS.md §4's `hw-shadow` entry precondition is "Transmit path disabled and
the disable verified by loopback readback." Concretely: call
`set_transmit_enabled(false)`, then `send_command` — which must return
`AdapterError::TransmitDisabled` without touching any stored state — then
`poll_feedback`, which must return `Ok(None)` because nothing was ever
written. `disabling_transmit_blocks_the_command_and_readback_confirms_it`
exercises exactly that sequence.

## Testing

`cargo test -p robot-hardware` — 12 tests covering zero- and multi-tick
delay, two commands in flight at once, transmit-disable readback, width
validation, link-age tracking, deterministic replay with rewind, and replay
exhaustion as a distinct terminal state.

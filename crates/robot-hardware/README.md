# robot-hardware

Device traits, loopback and replay devices, and the protocol-adapter
boundary named in SPECS.md §6's workspace table and exercised by §4's
`hw-shadow` mode. Depends only on `robot-types`.

## What's here

| Module | Contents |
| --- | --- |
| `device` | `HardwareAdapter`: the trait every device (simulated or real) implements; `AdapterError`; `FeedbackHeader` |
| `loopback` | `LoopbackDevice`: commands become feedback after a configurable, per-command, independently-counted poll delay |
| `replay` | `ReplayDevice`: deterministic playback of a pre-loaded frame sequence; never actuates; exhaustion is a distinct terminal error, never a repeat |
| `wire` | `robot-wire/1`, this project's documented device protocol: 24-byte header (magic, version, kind, sequence, sender timestamp, command validity, flags, width), `f64` values, CRC-32 trailer. Allocation-free encode/decode into caller buffers |
| `link` | `WireAdapter`: the host side as a `HardwareAdapter` over any `Transport` (UDP, or the in-process `MemoryLink`). Preallocated buffers, newest-frame feedback, counted corrupt/stale/lost frames, device sampling times mapped onto the host clock with a stated uncertainty, transmit disabled at start. `FirmwarePeer`: the reference device side, where commands are valid for their stated period and the watchdog holds position and flags itself when none is |

PLAN.md Phase 5: "Implement a hardware adapter first against a loopback
device, then a replay device, then an actual documented protocol." This
crate delivers all three. The protocol is `robot-wire/1`, documented in
`wire`'s module docs and specified by its tests. It is this project's own
protocol, not a vendor transcription. C-02 (first hardware target) is
open, and SPECS.md §1.1 treats every field of the retired `unitree/`
DDS/SDK reference as "a hypothesis requiring hardware verification". A
vendor bridge speaking `robot-wire/1` on one side and the vendor protocol
on the other is where those hypotheses get verified. This crate links no
vendor SDK.

## Clocks across the link

Feedback carries the device's sampling instant on the device clock. The
loop measures ages on the host clock, so `WireAdapter` maps each stamp.
The offset is the smallest `arrival - sampling` seen: every transport
delay is non-negative, so the least-delayed frame bounds it most tightly.
The mapped stamp carries the configured transport-delay bound as its
uncertainty. A device that does not flag its timestamps as sampling
instants gets arrival stamps, labelled as such (§5).

## The device watchdog

`FirmwarePeer` implements the device side of the command contract. A
command is valid for its stated period from receipt. With no valid
command, the watchdog trips: the device holds position and sets
`FLAG_WATCHDOG` in every feedback frame until a valid command arrives. The
adapter exposes those flags as `device_flags()`. This is SPECS.md §5's
"watchdog independent of the policy process", on the device side of the
link.

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

`cargo test -p robot-hardware`: 20 tests. 12 are the original ones, covering zero- and multi-tick
delay, two commands in flight at once, transmit-disable readback, width
validation, link-age tracking, deterministic replay with rewind, and replay
exhaustion as a distinct terminal state.

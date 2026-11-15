# 0025. The robot bridge is a configuration artifact; the G1 is the first

- Date: 2026-11-15
- Tracks: C-02 (direction set by the project owner: "any robot really —
  the plugin system should allow us to quickly configure the necessary
  elements for any new robot. we are starting with the G1")
- Touches: `crates/robot-hardware` (`bridge` module, robot-model and
  robot-artifacts dependencies), `assets/bridges/unitree_g1.toml`,
  SPECS.md C-02 row.

Status: accepted.

## Context

`robot-hardware` had the pieces — device traits, the `robot-wire/1`
adapter with CRC, sequencing and clock mapping, loopback and replay
devices, a reference firmware peer — but nothing said how a *particular*
robot plugs into them, and C-02 (which robot?) kept that question frozen.
The owner's direction unfreezes it sideways: the target is not one robot
but a configuration mechanism, exercised first on the G1.

## Decision

A robot is described to the hardware layer by a **`robot-bridge/1`
file** (bounded TOML subset), never by robot-specific code:

1. The file names the robot family, the committed protocol
   (`robot-wire/1` is the only admissible value), the endpoint, the link
   timing, and — the heart of it — the mapping from every actuated joint
   of the robot's profile to a wire index.
2. `BridgeConfig::load(text, &profile)` validates the whole artifact
   against the profile before any transport exists: every actuated joint
   mapped exactly once, indices dense, unknown joints and keys refused by
   name, and the command validity cross-checked against the profile's
   own firmware contract (a profile that declares a positive
   `command_expiration` is the authority; an undeclared contract leaves
   the bridge's declaration standing).
3. The validated bridge builds the existing `WireAdapter` and permutes
   values between profile order and wire order, so `robot-control` and
   the supervisor never learn a robot-specific ordering.

**The G1 configuration** (`assets/bridges/unitree_g1.toml`) maps the 29
joints onto the G1's own low-level motor order, read from the vendor
SDK's example source at a pinned revision (unitree_sdk2 @ `aaf69cb5`,
BSD-3-Clause; the evidence ledger row records the surface: DDS topics
`rt/lowcmd`/`rt/lowstate`, IDL `unitree_hg::msg::dds_::LowCmd_`/
`LowState_`, `G1_NUM_MOTOR = 29` with the exact `G1JointIndex` enum,
PD-plus-feedforward per-motor command, CRC32). The mapping turns out to
be the identity against the vendored profile's declaration order — now
an explicit, validated fact instead of an assumption.

**What this deliberately does not do.** The G1's native transport is DDS
over CycloneDDS, an excluded runtime (SPECS.md §2). The bridge therefore
stays on `robot-wire/1`, and the missing piece is a firmware-side shim
translating between the two on the robot's own computer — hardware-bound
work that waits for a robot, named in the ledger rather than pretended
away. Until then the bridge drives the loopback and replay devices,
which is exactly what its tests do, and `robot deploy` keeps its honest
exit 2.

## Consequences

- A second robot is a second `assets/bridges/*.toml` plus its profile —
  no new code unless its protocol genuinely differs.
- C-02's remaining open half is purchase and commissioning, not design.
- The supervisor/safety isolation of SPECS.md §12 is untouched: the
  bridge is below the supervisor, and nothing routes around it.

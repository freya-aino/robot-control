# Robot Control

A proposed Rust humanoid robotics platform for motion processing, simulation,
policy training, and hardware control, including boxing research.

- [Platform specification](SPECS.md): architecture, dependency policy, and acceptance requirements.
- [Implementation plan](PLAN.md): delivery phases and qualification gates.

The earlier Python prior art (`training/`, `unitree/`, `test.ipynb`) was removed
from the working tree on 2026-10-26 and remains available only in version
control; see [SPECS.md Section 1.1](SPECS.md#11-relationship-to-prior-repository-contents)
for its disposition.

## Vendor SDK integration recommendations

Rewriting an SDK is possible, but its value depends on the implementation,
documented interfaces, maintenance burden, and applicable rights. For a project
that permits native vendor dependencies, safe Rust bindings or a hybrid design
are usually the practical starting point.

**Project policy:** [SPECS.md Section 2](SPECS.md#2-language-and-dependency-policy)
currently excludes foreign SDKs, Python sidecars, and foreign simulation engines
from the default Rust implementation. The options below are architectural
recommendations, not approved dependencies. Using them requires the explicit
scope decision and isolated feature/profile described in Section 2.1.

### Choosing an approach

| Factor | Implication |
| --- | --- |
| Thin wrapper over a documented protocol or standard algorithms | A Rust implementation may be viable after checking compatibility and licensing. |
| Accessible hardware specifications and firmware interfaces | Make independent implementation easier; owning hardware alone does not establish protocol access or legal permission. |
| Proprietary computation, calibration, or control algorithms | Prefer retaining the vendor implementation unless there is a validated replacement. |
| Frequent vendor updates | Bindings reduce duplicated maintenance, but upgrades still need ABI and behavior checks. |
| Performance-critical native code | Measure latency and throughput before deciding to replace it. |
| License or interoperability restrictions | Review the applicable terms and jurisdiction before reverse engineering or distributing a replacement. |

### Three integration patterns

1. **Safe Rust bindings to the vendor library.** Generate low-level bindings from
   supported C headers with `bindgen` during the build or a controlled generation
   step, then expose a small, manually reviewed safe API. A C++ SDK may need a C
   shim or an appropriate bridge. Encapsulate handles, initialization, shutdown,
   ownership, lifetimes, callbacks, thread restrictions, and error conversion.
   Check target OS, architecture, ABI, and library versions. Rust safety at the
   boundary depends on valid invariants; it does not make the vendor binary safe.

2. **Hybrid Rust control with vendor hardware I/O.** Keep state machines,
   planning, RL policy execution, motion selection, opponent modeling, and
   recovery in Rust. Retain the vendor layer only for necessary driver setup,
   calibration access, low-level I/O, and firmware communication. Confirm which
   responsibilities actually reside in the SDK versus the device firmware.

3. **Full Rust implementation.** Consider this when source or sufficiently
   complete protocol documentation is available, applicable rights permit it,
   and hardware access supports validation. Budget for compatibility testing,
   calibration, timing, recovery behavior, and future firmware changes. A rewrite
   takes on maintenance responsibility and does not automatically receive vendor
   fixes.

### Candidate systems

These are investigation directions; SDK internals and demo implementations must
be verified against the exact product and version.

| System | Recommendation |
| --- | --- |
| Unitree G1 SDK | Inspect the actual transport, supported APIs, licenses, and firmware requirements. Consider bindings for essential hardware access while implementing project control logic in Rust. Do not assume the SDK is merely USB/serial wrappers or that a boxing demo establishes how its policy was trained. |
| CHINGMU motion capture | Determine whether the SDK streams already reconstructed poses or performs reconstruction itself. Investigate open motion-recovery tools such as GVHMR and retargeting tools such as GMR as alternatives, validating accuracy, latency, licensing, and hardware needs; they are not automatic drop-in replacements. |
| NVIDIA Isaac Lab / Isaac Sim | If separately authorized, use as an external simulation or validation substrate rather than attempting a rewrite. Check component-specific source availability and licenses; Isaac Lab being open source does not establish that the entire simulator is. The project's default remains the qualified Rust simulation stack. |
| Generic servo/controller SDK | A direct Rust CAN/UART/TCP implementation may be suitable when the protocol is documented. Verify device-specific sequencing, calibration, timing, and fault handling rather than assuming the DLL only wraps PID and transport. |

### Practical recommendations for the boxing stack

- Prefer a narrow vendor hardware adapter over rewriting undocumented control
  behavior when an SDK exception is justified. Keep project strategy and control
  logic in Rust.
- Evaluate open motion-capture and retargeting algorithms before committing to a
  proprietary pipeline. A Rust port needs numerical validation. A Python sidecar
  over gRPC or Zenoh is an optional exception candidate, with explicit timing and
  failure behavior, rather than part of the default architecture.
- Document every `unsafe` block's invariants. Validate command ranges, buffer
  lengths, return codes, and response contents; test error propagation, malformed
  responses, initialization failures, and disconnect/reconnect sequences. Use
  property and mutation testing; foreign fuzz engines require the development-tool
  exception described in the specification.
- Validate hardware integration early through the project's supervised,
  noncontact qualification stages. Record required initialization and shutdown
  sequences, check timing and watchdog behavior, and reproduce stateful vendor
  sequencing before optimizing.
- Pin SDK/header versions and rerun compatibility checks for vendor updates.
  Bindings preserve access to vendor fixes only when those updates are integrated
  and validated.

### Licensing and independent implementations

Review SDK licenses, redistribution terms, and applicable interoperability rules.
Reverse-engineering and anti-circumvention rules vary by jurisdiction; hardware
ownership does not settle them. Patents may apply independently of source-code
copyright, so rewriting alone does not remove that concern.

Where appropriate, a clean-room process can separate analysis from implementation
using a reviewed interface specification and documented provenance. It is not a
blanket exemption from license, patent, or anti-circumvention obligations.

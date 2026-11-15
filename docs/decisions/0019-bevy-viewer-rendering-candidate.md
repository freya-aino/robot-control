# 0019. Bevy selected as the viewer rendering candidate

- Date: 2026-11-15
- Touches: `SPECS.md` (Appendix B survey row, §11.2 viewer shortlist),
  `crates/robot-viewer/README.md`. No dependency is added by this record.

Status: accepted (direction); the dependency itself remains `unverified`
until it passes the Appendix B.6 qualification gate.

## Context

`robot-viewer` deliberately stops at a data layer plus std-only SVG,
because SPECS.md §11.2's shortlist (Rerun, egui, `urdf-viz`) was entirely
`unverified` and no B.6 gate had been run (ADR 0003 keeps the default
closure empty). The project owner has now directed that viewer rendering
be achieved with the **Bevy engine**, which was not previously in the
Appendix B survey.

## Decision

Bevy replaces Rerun/egui as the *first* candidate to take through B.6 for
the viewer. The integration shape follows the one precedent this
workspace already has for a heavy foreign-adjacent dependency,
`robot-sim-gpu`'s CubeCL feature (ADR 0012):

1. A separate, unpublished crate (working name `robot-viewer-app`,
   `publish = false`) depends on Bevy; `robot-viewer` itself stays
   std-only as the data layer the app consumes. Nothing in the default
   workspace build compiles Bevy, and no shipping or control-path crate
   may depend on the app crate (SPECS.md §2.5: a viewer is a development
   service, never a robot-host dependency).
2. B.6 must be run before the crate lands: pinned version, license sweep
   of the enabled feature closure (Bevy pulls a large tree; `wgpu` is
   already surveyed), build time and binary impact measured, and the
   exclusion gate taught the new allowed subtree — exactly as cubecl was
   gated.
3. First deliverable: an **offline replay viewer** over the data the
   project already records (`viewer-frames/1` runs, the USD replays),
   because it exercises rendering without touching any live loop. Live
   inspection hooks come after.

## Consequences

- SPECS.md §11.2's shortlist and the Appendix B survey gain Bevy as the
  designated candidate; Rerun and egui remain surveyed alternatives.
- `robot-viewer`'s "no rendering toolkit" rationale changes from "no
  candidate chosen" to "candidate chosen, qualification pending", and its
  data layer is explicitly the contract the Bevy app renders from.
- The B.6 run is the next actionable step and is implementation work, not
  a further decision.

# 0002 — canonical frames, gravity direction and quaternion ordering remain open

- Status: accepted (an explicit non-decision)
- Date: 2026-10-26
- Tracks: C-04
- Touches: SPECS.md Appendix A.3, every schema in §7

## Context

SPECS.md Appendix A.3 states: "The canonical coordinate frame, gravity
direction, root convention and quaternion component ordering are fixed by
decision record and recorded here once ratified... Until then, no code may
assume a project-wide convention, and every transform MUST name the frames it
maps between."

## Decision

This record exists to state explicitly that **C-04 is not resolved by this
revision**, and to record how the type system keeps that non-decision from
drifting into an accidental one:

1. `robot_types::frames::Quat` has no four-element array constructor or
   accessor. Components are reached by name (`w`, `x`, `y`, `z`) only.
2. `robot_types::frames::Transform` carries a `from: FrameId` and
   `to: FrameId` and its `compose` refuses to compose across a mismatch
   (`FrameError::Mismatch`). `FrameId::WORLD` fixes only an *index*
   (`FrameId(0)`) as a convenience for test fixtures and the profile's
   per-link reference choice in `robot-model::kinematics` — it fixes no
   *orientation*, *handedness* or *gravity direction*.
3. `robot_types::frames::GravityConvention::minus_z_standard` is documented
   as "a configuration default for tests and fixtures... not a project
   convention": every caller that uses it in non-test code must state why.
4. `robot_types::frames::EulerOrder` exists specifically because "Euler-angle
   conventions are source-specific; never assume every BVH file uses the same
   rotation order" (§7.2) — every importer records the order it read, rather
   than normalizing silently to an assumed one.
5. URDF's `rpy` attribute is the one exception, and it is documented as such
   in `robot_model::urdf::parse_origin`: URDF itself fixes an extrinsic
   fixed-axis roll-pitch-yaw convention (`Rz * Ry * Rx`), which is a fact
   about the *file format*, not a project-wide convention this codebase
   adopts elsewhere.

## Consequences

- No benchmark, fixture or profile in this revision may hard-code "the"
  world frame's orientation or "the" gravity direction as if settled.
- Every future PR touching `§7` schemas must either preserve this
  non-decision or supersede this record together with a real C-04 decision
  record, per the change-control rule in SPECS.md's document-control table
  ("Changes to Sections 2, 4, 5, 7, 12 or 14 require a decision record").
- `robot-model::kinematics::forward_kinematics` computes link poses relative
  to the *profile's own root link*, explicitly documented as "a choice local
  to this type, not an assertion about a project-wide world frame" — so that
  when C-04 is eventually ratified, only the call sites that need a true
  world frame (not this module) are affected.

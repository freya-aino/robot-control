# 0022. The canonical convention: Z-up, right-handed, −Z gravity, w-first

- Status: accepted (supersedes ADR 0002's explicit non-decision)
- Date: 2026-11-15
- Tracks: C-04 (resolved)
- Touches: SPECS.md Appendix A.3 and the C-04 row; doc comments in
  `robot-types`, `robot-model` that hedged "until C-04 resolves".

## Context

ADR 0002 deliberately kept C-04 open and built type-system discipline so
the non-decision could not drift into an accidental one: frames named on
every transform, quaternion components reached by name, gravity carried
explicitly per profile. Since then the entire implementation has in
practice converged on one convention — the vendored G1's USD layer, the
ground scene, `GravityConvention::minus_z_standard`, the simulator's
`Ground` at a `z` height, the viewer's side/front projections and every
fixture agree. The project owner has now directed that it be ratified.

## Decision

The project-wide canonical convention, recorded into Appendix A.3:

1. **World frame**: right-handed, **Z up**. The ground plane is `z =
   height`; "forward" is a per-robot statement in its profile, not a
   world convention.
2. **Gravity**: **−Z** at standard magnitude unless a profile declares
   otherwise (`GravityConvention::minus_z_standard` is now the project
   convention it previously only stood in for; a profile may still
   declare a different convention explicitly, and every consumer keeps
   reading the declared one rather than assuming).
3. **Quaternions**: unit, **w-first** in documentation and constructor
   order (`w, x, y, z`), Hamilton product, `q.rotate(v)` mapping body to
   world for a world-from-body attitude. Serialization stays **named
   components** — that rule was never about the open question alone; it
   is what makes records self-describing, and it remains.
4. **Root convention**: a profile's declared `root_link` is the
   kinematic reference; a floating base is the explicit `world` link +
   `Free` joint wrap (`with_floating_base`), never an implicit re-rooting.
5. **Units and time**: SI throughout and the per-clock timing model of
   Appendix A.3, unchanged — they were never in question, only grouped
   under C-04's row.

What does **not** change: every enforcement mechanism of ADR 0002 stays.
Transforms still name their frames, quaternion components are still
reached by name, importers still record source conventions (BVH up-axis
and Euler orders, URDF's rpy) and convert explicitly. The convention
makes the target of those conversions normative; it does not license
dropping the bookkeeping.

## Consequences

- Doc comments that hedged "until C-04 resolves" now cite this record as
  the convention instead.
- Fixture data and scenes need no changes — the ratified convention is
  the one they already use, which is the point of ratifying it.
- A future robot whose vendor data is Y-up continues to be converted at
  import, exactly as BVH already is.

# motion-process

Resampling, filtering, contact estimation and segmentation, per SPECS.md
§8.2. Depends on `robot-types`.

## What's here

| Module | Contents |
| --- | --- |
| `resample` | `resample`: fixed-rate resampling by linear interpolation (lerp for translation, slerp for rotation) |

## Why this resampler has zero phase delay, and what that doesn't cover

SPECS.md §8.2 item 4 asks for resampling "with documented phase delay."
Linear interpolation at exact output sample times, drawing only from the
two bracketing input frames, introduces no phase shift — each output
sample is evaluated at its own true time, not smoothed across a window —
so the recorded delay is `Duration::ZERO`, and
`resampling_records_zero_phase_delay` checks that this is what actually
gets written to provenance rather than assumed.

What this does **not** cover: anti-aliasing. Downsampling fast motion
(a boxing strike) with no pre-filter can alias, which is exactly the "real
strike-speed characteristics" §8.2 item 4 warns about preserving. This
module's own doc comment says so; a qualified low-pass stage is future
work, not implemented here.

## Every resampled sample is marked `Interpolated`

Every output frame is synthesized from two inputs (or, at `frac == 0`,
numerically coincides with one) — never a direct carry-through of a single
source sample — so every channel's validity is unconditionally
`Validity::Interpolated`, never silently promoted back to `Valid`. An
earlier draft tracked the two bracketing samples' validity and branched on
whether either was `Missing`, but both branches of that condition produced
the same result; the dead branch was removed rather than left as
decoration implying a distinction that wasn't there.

## Testing

`cargo test -p motion-process` — 4 tests: upsampling frame count and
rotation interpolation, the zero-phase-delay provenance record, rejection
of a non-positive or non-finite target rate, and preservation of source
identity/session/provenance metadata across the resample.

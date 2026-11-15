# motion-process

The motion processing pipeline of SPECS.md §8.2, items 3 to 6: conditioning,
filtering and resampling, ground-plane and contact estimation, and
segmentation, plus the augmentation SPECS.md §6 assigns this crate.
Depends on `robot-types`.

Items 1 and 2 (inspection, immutable originals with content IDs) belong to
`motion-io` and `robot-artifacts`; items 7 to 9 to `motion-retarget` and
`motion-validate`. Every operation records a provenance step on the record
it changes, with its parameters and, for a filter, its phase delay.

## What's here

| Module | §8.2 item | Contents |
| --- | --- | --- |
| `kinematics` | (support) | Forward kinematics (positions and orientations) in the BVH convention `motion-io` imports; finite-difference velocities that never read a missing sample |
| `condition` | 3 | `convert_axes` (basis change and length scale, e.g. Y-up centimetres to Z-up metres), `mark_non_finite` (bad samples become `Missing`, frames with a bad root are dropped and reported), `find_timing_gaps`, `fill_gaps` (bounded, marked `Interpolated`) |
| `filter` | 4 | Second-order Butterworth low-pass, zero-phase (forward-backward, Winter-corrected cutoff) or causal (exact DC group delay recorded); rotations filtered on the rotation group (tangent-space components about the first sample, branch-unwrapped, exponentiated back); per-channel peak-speed retention |
| `resample` | 4 | `resample` (lerp/slerp at exact output times, zero phase delay), `resample_anti_aliased` (zero-phase pre-filter at 0.4 × the target rate when downsampling) and `lowpass_resampled` (filters a possibly non-uniform record, resampling it to its own median rate first when the filter would refuse it) |
| `contact` | 5 | `estimate_ground_plane` (least squares on stationary foot samples, 2.5σ rejection, residual and height standard error), `estimate_contacts` (logistic height × speed probability, hysteresis, per-boundary ambiguity spans, `ContactLabel`s written with confidence) |
| `augment` | §6 | `mirror` (reflection plus left/right swap, refused for an asymmetric skeleton), `mirror_pairs_by_name`, `time_scale`, `add_noise` (seeded, recorded in provenance), `transform_root` |
| `segment` | 6 | Rule-based stance / locomotion / strike / defense / recovery / transition windows with per-segment confidence, written as task labels; with `step_window` set, foot-contact alternation (from the contact stage's labels) reads as locomotion, so stepping in place is no longer stance |

## Decisions worth knowing

**Phase delay is computed, not asserted.** Zero-phase filtering and
interpolating resampling record `Duration::ZERO` because their phase
cancels or never arises. The causal filter records
`sum(k b_k)/sum(b_k) - sum(k a_k)/sum(a_k)` samples, the exact DC group
delay of its coefficients; a test checks it against the measured lag of a
filtered ramp.

**Strike speed is measured, not promised.** SPECS.md asks to "preserve real
strike-speed characteristics". No low-pass filter can guarantee that, so
`FilterReport::speed_retention` gives each channel's peak speed after
filtering over before. A curation step sets its own bound. A test shows a
4 Hz cutoff costing a 50 ms burst more than 40 % of its peak speed and a
40 Hz cutoff costing it under 5 %.

**Uncertainty reaches the labels.** The plane's residual and height
standard error widen the contact model's height scale, so a rough or
poorly supported floor yields less confident contacts. A frame with an
unknown foot position is labelled with confidence 0.5, and hysteresis
carries the previous state across it rather than inventing a transition.

**A plane tilt is fitted only when the support spans an area.** If the
stationary foot samples are clustered (smaller principal spread under
5 cm), the plane is taken perpendicular to `up` and `tilt_fitted` is
false, rather than fitting a slope to noise.

**Rotations are filtered on the group, validated against the retired
approximation.** The earlier revision filtered hemisphere-aligned
quaternion components and renormalized. The tangent-space path reproduces
it where it was valid (worst divergence 7.3e-7 rad on a 0.2 rad, 1 Hz
sway at 100 Hz) and stays correct where components alias: on a 0.8
rad-per-sample spin with ±0.05 rad jitter filtered at 2 Hz, the group
path's worst interior error against the true smooth rotation is 8.5e-3
rad where the component path's is 0.64 rad. Both numbers are asserted in
the tests, with the retired filter kept in the test module as the
comparison reference.

**Body parts are found by exact name or not at all.** `BodyMap::find`
matches a short list of common names case-insensitively and never by
substring. Any other skeleton passes an explicit `BodyMap`.

**No threshold is a validated constant.** Contact and segmentation
thresholds are required fields of `ContactConfig` and `SegmentConfig`. The
values in the tests are test values. Whether the segmenter's windows match
a human annotator's is an evaluation question this crate does not answer.

## Limits

- `lowpass` itself still needs uniform sampling (intervals within 1 % of
  the median); `lowpass_resampled` takes a non-uniform record and
  resamples it to its own median rate first.
- Segmentation needs head and both hand channels. The stepping rule reads
  the contact labels `estimate_contacts` wrote; a record without them, or
  `step_window = 0`, keeps the prior root-speed-only behaviour.

## Testing

`cargo test -p motion-process`: 47 tests. They cover forward kinematics and
missing-sample propagation, rigid axis conversion, non-finite marking and
dropped-frame gaps, bounded gap filling, the filter's −3 dB point (single
and two-pass), its DC gain, its recorded delay against a measured ramp lag,
zero-lag zero-phase filtering, strike-speed retention, anti-aliasing of a
component above the new Nyquist, flat and tilted plane recovery, refusing
to fit a tilt to one footprint, contact intervals and label confidence on
scripted gait, the effect of plane roughness and a hidden foot, a
scripted boxing bout segmented in order at the scripted times, and the
augmentations (mirroring reflects every global position and is its own
inverse, asymmetric pairs are refused, time scaling, seeded noise of the
requested scale, rigid root transform).

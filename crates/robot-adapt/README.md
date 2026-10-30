# robot-adapt

System identification only (history conditioning and bounded residuals
are not implemented), per
SPECS.md §11.3. Depends on `robot-types` and `robot-model` (the latter not
yet used — reserved for mapping identified parameters back onto a specific
robot profile's actuators).

## What's here

| Module | Contents |
| --- | --- |
| `sysid` | `identify`: ridge-regularized linear least squares over a caller-defined feature model, reporting singular values for identifiability |

This implements only the **first row** of SPECS.md §11.3's ordered
adaptation table: calibrated system identification. Every later row —
history-conditioned adaptation, the RMA-class two-phase latent
identification, simulator correction, bounded residuals, fine-tuning — is
unimplemented, and that is consistent with the table's own ordering rule:
"A method MUST NOT be scheduled before the methods above it have been
measured and found insufficient on the same robot." Nothing in this
project has reached that measurement yet.

## Identifiability is a number, not a caveat

SPECS.md §11.3: "Parameters may not be identifiable from passive logs" and
"preserve uncertainty for parameters the data cannot distinguish."
`IdentificationResult::singular_values` and `condition_ratio()` make that
checkable: `an_unidentifiable_parameter_combination_is_flagged_by_a_small_singular_value`
constructs a regressor where one feature is an exact multiple of another
(a textbook unidentifiable pair) and confirms the condition ratio collapses
toward zero, rather than the fit silently returning confident-looking
coefficients for an unidentifiable combination.

## Testing

`cargo test -p robot-adapt` — 4 tests: exact coefficient recovery on
noiseless synthetic data, the identifiability check above, and two
input-validation rejections (inconsistent feature width, empty sample set).

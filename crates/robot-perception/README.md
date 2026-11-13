# robot-perception

Floating-base state estimation with covariance consistency, per SPECS.md
§11.2. Depends only on `robot-types`.

## What's here

| Module | Contents |
| --- | --- |
| `linear_kf` | `KalmanFilter`: generic discrete-time linear-Gaussian KF — predict, innovation, Mahalanobis gating, Joseph-form update |
| `attitude` | `AttitudeFilter`: multiplicative error-state attitude filter with gyro bias, built on `linear_kf` |
| `contact` | `ContactClassifier`: per-site hysteresis contact detection with confidence |
| `base` | `BaseEstimator`: base position and velocity in the world frame, propagated by world-frame IMU acceleration over the measured `dt`, corrected by stance-leg velocity (`leg_velocity`, checked against a rigidly moving base) and flat-ground height. Mahalanobis-gated, so a slipping foot is rejected and counted, not believed. Dropouts only predict |
| `opponent` | `OpponentTracker`: a constant-velocity track with gated detections, coasting through occlusion with growing uncertainty, reporting its age, and dropping the track (returning `None`) past the maximum age |
| `noise` | `NoisyChannel`: bias random walk, white noise, quantization and delay applied to `robot-sim`'s ideal sensor readings, with `record()` naming the injection point and seed (`crates/robot-sim/specs` Appendix B S-06's default location) |
| `consistency` | `Consistency`: NEES accumulation with the mean and the fraction inside the chi-square 95 % interval, the §11.2 comparison against synthetic truth |

Every one of SPECS.md §11.2's required specifications — "orientation/error-
state conventions, IMU bias dynamics, contact assumptions, covariance
propagation and measurement gating" — is named explicitly in `attitude`'s
module documentation rather than left implicit; see that file's doc comment
for the table mapping each requirement to its implementation.

This is an authored baseline under the policy of
`docs/decisions/0003-std-only-default-closure.md`: SPECS.md Appendix B.2.9's candidates (`multicalc`, `numeris`,
`typed_kalman`, `kfilter`) are all still `unverified`, and this crate is
what the project runs, and what any future candidate must be compared
against, until one of them clears Appendix B.6.

## A Jacobian bug the test suite caught

SPECS.md §11.2: "Autodiff Jacobians are optional and require validation, as
do hand-derived Jacobians." `measurement_jacobian`'s first draft had the
wrong sign — `measurement_jacobian_matches_central_finite_differences`
failed immediately, and two downstream tests
(`accelerometer_correction_reduces_a_known_tilt_error` and one gating test)
failed with it, because a wrong-signed Jacobian drives a Kalman update in
the wrong direction. This is recorded here because it is exactly the
scenario the finite-difference check exists for: a plausible-looking
hand derivation that is simply wrong, caught before it reached anything
downstream of this crate. The fixed derivation is explained in the function's
own doc comment.

## Where this estimator stops

The accelerometer correction gates on specific-force *magnitude* as a proxy
for "the body is approximately static," not on ground contact. SPECS.md
§11.2 asks the module to specify its contact assumption explicitly, and the
explicit answer here is: this filter does not use one. `ContactClassifier`
exists and is tested, but wiring its output into the attitude filter's gate
(a strictly better assumption for a legged robot standing on one foot vs.
airborne) is future work for whichever crate integrates a specific robot
profile — most likely `robot-control`, once a reference embodiment is
chosen (C-02).

## Qualification against synthetic truth

§11.2: "Qualify filters under timestamp jitter, dropouts, outliers, slip
and poorly conditioned observations. Compare observable-state error and
covariance consistency against analytic or synthetic truth." The base
estimator's tests run 2000 steps with:
- ±20 % timestamp jitter;
- accelerometer and leg-velocity noise;
- a dropout every tenth step;
- in one run, 50 steps of a sliding foot.

They check the final velocity error and the mean NEES of the six-state
error, which must lie within a factor of two of its ideal. With the slip,
they check that at least 40 slipped measurements were rejected and the
estimate stayed accurate. No percentage threshold is assumed. The NEES
statistic itself is tested against errors drawn from a known covariance,
and against the same errors with an overconfident one.

## Testing

`cargo test -p robot-perception`: 25 tests. The original 15 cover the Kalman filter checked
against analytic constant-velocity tracking, covariance collapse under a
near-perfect sensor, Mahalanobis-gate outlier rejection, and Joseph-form
positive-semidefiniteness after 200 updates (via eigenvalue check); the
attitude filter checked against exact rigid-body integration for constant
angular rate, the Jacobian validated by central finite differences, tilt-
error convergence under repeated correction, the specific-force magnitude
gate, and bias correctly offsetting the integrated rotation; the contact
classifier checked for hysteresis (no chatter inside the band) and for
confidence ordering. The new ones cover the leg-velocity formula, base
estimation accuracy and consistency, slip gating, NEES against known and
overconfident covariances, the chi-square interval, opponent coasting
and aging, and false-detection rejection and track restart, and the noise model (white spread, square-root bias growth, quantization and delay).

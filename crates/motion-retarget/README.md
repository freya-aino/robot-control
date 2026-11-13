# motion-retarget

Retargeting a human motion record onto a robot profile, per SPECS.md §8.2
item 7 and §8.3: weighted task-space IK with position, orientation,
relative-position and posture tasks, planted feet, hard joint limits, a
self-collision penalty and a free base, followed by temporal refinement.
Depends on `robot-types`, `robot-model`, `motion-process` (forward
kinematics of the human record) and `robot-artifacts` (the TOML subset of
the map file).

## What's here

| Module | Contents |
| --- | --- |
| `ik` | `solve_tasks`: Levenberg-Marquardt over `Task::{Position, Orientation, RelativePosition, Posture}`, an optional free `Base` with a prior, an optional `SelfCollision` penalty; limits by projection; bounded by `max_iterations` and a wall-clock `deadline`, with a `Termination` (`Converged`, `Stalled`, `MaxIterations`, `Deadline`) and per-task residuals. `solve` keeps the position-only entry point |
| `collision` | `SelfCollision`: one bounding capsule per collision shape (sphere, box, cylinder, convex hull), non-adjacent pairs, pairs overlapping in the reference posture excluded and reported, exact segment-segment distance |
| `temporal` | `smooth`, `smooth_rotations`: a discrete smoothing spline solved by banded Cholesky in O(n); straight lines pass through unchanged |
| `retarget` | `retarget`: `RetargetMap` (absolute and chain-relative links, orientation offsets, planted feet from contact labels), per-frame solve warm-started from the previous frame, smoothing then re-projection, a `RobotReference` record with per-frame reports and velocity excess. `chain_ratio` measures robot over human chain length |
| `config` | The `retarget-map/1` TOML file: names instead of indices, `ratio = "auto"`, unknown keys rejected |

## Decisions worth knowing

**Intent over coordinates.** §8.3 asks to "preserve task intent, such as
punch direction and target-relative reach, rather than matching all human
coordinates equally". A relative mapping targets the hand's displacement
from the shoulder, scaled by the robot/human arm-length ratio, so a jab
reaching 90 % of the human's arm reaches 90 % of the robot's in the same
direction. A test retargets a forward punch onto the G1 and checks the
shoulder-to-hand vector at every frame within 2 cm.

**Limits are hard, collision is soft.** Every iterate is projected into the
joint limits, so no output violates them. Self-collision is a penalty
(a zero penalty guarantees clearance because every capsule contains its
shape, but the solver may trade some penetration for task error). The
per-frame report gives the penetration that remains.

**Planted feet are held, and the slip is reported.** While a foot's
contact label is confident, its robot foot is held at the ground point
where the contact began, with a heavy weight. The report gives the
distance from the anchor at every planted frame, which is
`motion-validate`'s planted-foot slip.

**Infeasibility and timeouts are explicit.** §8.3 requires "bounded
iterations/deadlines and a validated response to infeasibility or
timeout". Each frame reports why its solve stopped, and
`RetargetOutput::unfinished_frames` lists frames that hit a deadline or
the iteration cap. An unreachable target ends as `Stalled` at its true
residual (a test checks 8.0 m for a target 8 m beyond reach), not as
`MaxIterations`.

**The reference is not claimed feasible.** Joint velocity above each
joint's declared limit is listed in `velocity_excess`. Torque and
simulated tracking feasibility are `motion-validate`'s job.

**No QP solver.** §8.3: "Qualify a Rust-native constrained solver before
selecting differential-QP IK." No QP candidate (Appendix B.2.8) has
cleared qualification (`docs/decisions/0003-std-only-default-closure.md`).
This is the non-QP baseline Appendix C's C-09 permits. The Jacobian is
central finite differences over `robot_model::kinematics`.

## Two bugs the original test suite caught

**A fixture that assumed the wrong kinematics.** The first version of
`converges_to_a_reachable_target_within_tolerance` targeted a point derived
from a textbook two-link-arm formula, but this project's joint composition
(documented in `robot-model`'s own `kinematics.rs`) places a link's origin
exactly at its parent joint's pivot — so the "lower" link's position in the
test fixture depended on the shoulder angle *only*, and the chosen target
wasn't on the circle that link can actually reach. The fix added a third,
fixed-joint "hand" link beyond the elbow, so the target genuinely depends on
both joint angles, matching the classic two-link arm the test intends to
exercise.

**A Newton-step sign error.** Even after fixing the target, the solver
diverged. The Jacobian here is `d(error)/d(joint)`, computed directly by
finite-differencing the error vector; a Newton step toward zero error needs
`jacobian · step ≈ -error`, not `≈ +error`. The original code passed
`error_vec` straight into `damped_least_squares`, which solves for the step
that *increases* error fastest. The fix negates the right-hand side, and
the comment at the call site explains why in the same terms as here.

Both fixes are recorded in the code's own comments, not only here, so a
future reader hits the explanation at the point of the bug rather than
only in this file.

## Testing

`cargo test -p motion-retarget`: 28 tests. Solver: convergence to a reachable
target (checked by forward kinematics), orientation and relative targets, a
posture prior choosing among redundant solutions, a free base reaching a far
target, a zero deadline, stalling at the true residual of an unreachable
target, hard limits, invalid tasks and widths. Collision: closed-form segment
distances, every bound containing its shape, a measured overlap, the penalty
holding a hand off the base, pairs excluded at the reference. Smoothing:
lines unchanged, normal equations satisfied, jitter attenuated, rotation
hemisphere flips. Pipeline on the vendored G1: punch direction and reach
preserved with planted feet slipping under 2 cm, smoothing with
re-projection, velocity excess reported for a 3 ms punch, a collision model
built from the G1's shapes and respected, invalid maps refused. Map file:
names resolved and run, unknown names, keys and schemas refused.

# motion-retarget

Joint-limit-constrained IK, per SPECS.md §8.3. Skeleton-to-robot mapping,
contact and collision constraints and temporal optimization are not
implemented. Depends on `robot-types` and `robot-model`.

## What's here

| Module | Contents |
| --- | --- |
| `ik` | `solve`: damped-least-squares task-space IK for end-effector position targets, with a numerical (finite-difference) Jacobian |

SPECS.md §8.3: "Qualify a Rust-native constrained solver before selecting
differential-QP IK." No QP candidate (Appendix B.2.8) has cleared
qualification (`docs/decisions/0003-std-only-default-closure.md`); this is
the "feasible non-QP baseline" Appendix C's C-09 explicitly permits in the
interim.

## Two bugs the test suite caught

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

`cargo test -p motion-retarget` — 4 tests: convergence to a reachable
two-joint target (verified independently via forward kinematics on the
solved angles, not by comparing against the angles used to construct the
target — a 2-DOF arm reaching a 2-D point has two valid solutions), an
unreachable target terminating without diverging to non-finite values,
joint limits holding throughout, and a width-mismatch rejection.

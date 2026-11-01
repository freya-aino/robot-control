# Example system: mapping to the parent specifications

Status: **non-normative guidance**. This document introduces no requirement, changes no acceptance row and does not alter any parent specification. Every parent file named here keeps its current text. Nothing below is evidence under `SPECS.md` Section 3.

## 1. What the example folder is, and how it is used

`docs/example-system-specs/` is a set of 17 long documents. Each one describes, in equations and algorithm steps, how a complete rigid-body and contact simulator is built: its integrators, kinematics, collision routines, constraint solver, and its GPU execution. Its index (`00_INDEX.md`) gives the reading order. The folder is a worked example of how such a simulator is put together. It is not a specification to implement.

Three rules govern its use here:

1. The folder is **inspiration for mathematics and algorithms**. A parent section may point to an example section that shows a method worth considering. The parent requirement itself stays as written.
2. Its numerical constants, defaults and tolerances belong to the engine it describes. They are not transplanted into this project. Any constant adopted here needs its own measurement under `SPECS.md` Section 9.
3. Where the example and the parent disagree, the parent governs. Where the example describes a behaviour the parent does not require, the parent is not changed to match it.

## 2. Mapping

Each row names a parent area, the example sections worth reading, what to take from them, the current state of this repository, and a candidate piece of work. The candidate work is an option for a later decision, not a commitment.

### 2.1 Integrators and time stepping (parent `03-integrators.md`, `SPECS.md` Section 5)

| Example | What to take | Current state | Candidate work |
| --- | --- | --- | --- |
| `01` §9.1 Euler, §9.2 implicit and implicitfast | The Euler update with the velocity sub-step, and implicit treatment of joint damping in the velocity equation | Semi-implicit Euler with implicit joint damping in `integrators.rs` | Compare the two damping treatments on the G1 at the step sizes used |
| `01` §9.4 Runge-Kutta 4 | Stage order, and that time advances only at the full step | Not implemented | Only if an integrator with higher order is needed for a fidelity tier (`09-fidelity-tiers.md`) |
| `11` §3.4 `mjd_freeMhat`, §3.3 `mjd_freeBias_vel` | The effective mass `M + hD` and the velocity-derivative blocks used by implicit integration of a free body | Not used | Use as the reference for an implicit free-base update |

### 2.2 State, coordinates and kinematics (parent `01-state-and-coordinates.md`, `02-pure-dynamics-core.md`)

| Example | What to take | Current state | Candidate work |
| --- | --- | --- | --- |
| `02` §1.2 quaternion algebra, §1.3 spatial algebra | Quaternion and spatial conventions, with the sign rules for normalisation | Free-base quaternion stored as `(w, x, y, z)` in `articulated.rs`, integrated by `integrate_free_position` | Check the integration against `11` §1.7 `mjd_quatIntegrate` on a spinning free body |
| `02` §2.1 body poses, §2.3 centre-of-mass frames | How the composite inertial frame is built from body poses | Body poses from `joint_transform` and `compose_pose` | None needed now |
| `02` §2.9 composite rigid body, §2.10 factorisation, §2.11 back-substitution | Mass matrix by composite rigid body and sparse factorisation | The dynamics use the articulated-body algorithm, with no mass matrix formed | The hard-limit solve (see `06-equality-constraints-and-solver.md` below) takes its columns from forward dynamics. A factorised `M` would avoid repeated solves if the number of limited joints grows |
| `02` §3.3 inverse dynamics (`mj_rne`) | Recursive Newton-Euler as the bias-force reference | Implemented as `inverse_dynamics` | Check against the example's bias-force convention on a model with gravity |
| `13` (inertia from geoms and meshes, frame conventions) | How the inertia of a body is derived from its geometry and expressed in its principal frame | The G1 inertia comes from the USD `physics:*` attributes | Check that the principal-axes convention in the importer agrees with the example's, on one link |

### 2.3 Actuation, passive forces and sensors (parent `05-passive-forces-and-actuators.md`)

| Example | What to take | Current state | Candidate work |
| --- | --- | --- | --- |
| `10` actuator gain and bias types (the position servo) | Separation of an actuator into gain, bias and dynamics, which makes the servo a composition of three choices | Joint PD gains `PdGains` and effort limits | Describe the G1 actuators in the same three-part form when the actuator model is extended |
| `09` springs and dampers | Passive spring and damper forces as functions of position and velocity only | Not implemented as passive joint forces in this crate | Use as the form of any passive joint spring or damper |
| `10` sensors (touch and limit sensors) | Sensor outputs defined from the constraint rows and contact list | Proximity signal in outer 5 and 95 percent of range (`SPECS.md` Section 9.1) | The example defines a limit sensor from the same geometry, which is a useful check on the proximity signal |

### 2.4 Contact and constraint solving (parent `06-equality-constraints-and-solver.md`, `SPECS.md` Sections 9.1 and 9.3)

This is the area the example treats most fully, and the one where this repository is weakest.

| Example | What to take | Current state | Candidate work |
| --- | --- | --- | --- |
| `07` §2.1 to §2.4, the convex problem and impedance | Contact and limit forces as the solution of one convex problem, with each row having a reference acceleration `a_ref` and an impedance `d` that sets how much of the row is enforced | Ground and pair contacts are penalty springs with fixed stiffness and damping (ADR 0010). Joint limits are an active-set solve (`articulated.rs`) | Test a soft-constraint form for contact whose stiffness comes from a time constant and damping ratio. This may remove the energy growth seen in the zero-torque run, which is the open question recorded under `docs/ledger` |
| `07` §12.3 impedance sigmoid | A smooth transition from the soft to the hard value over a width | Not used; the penalty is linear in depth | Use the sigmoid as the penetration-to-stiffness map if a smooth contact onset is needed |
| `07` §8 limit rows | A limit is active when its distance is below a margin, one row per side, and the activation test is strictly below the margin | Hard limits with a proximity band of 5 and 95 percent of range | The margin in the example is the same idea as the proximity band. Keep the two definitions aligned in documentation |
| `08` §6 primal Newton and conjugate gradient, §6.8 exact line search | Solving the convex problem by a primal method with an exact line search | Not implemented | Only if a joint-coupled solve replaces the active-set limit solve |
| `08` §8 islands | Splitting a problem into independent groups of coupled constraints | Not implemented | Useful once many bodies sleep or contact groups separate, so that only active groups are solved |

### 2.5 Collision (parent `07-collision-pipeline.md`, `SPECS.md` Section 9.1)

| Example | What to take | Current state | Candidate work |
| --- | --- | --- | --- |
| `03` §2 filters (AABB, sphere, body pair) | Cheap rejection of pairs before any narrow-phase call | All pairs of contact shapes are tested (`contact_wrenches`) | Add an AABB filter before `collision::proximity` if the shape count grows |
| `03` §3 sweep-and-prune with principal axis | A sorted sweep along the axis of largest spread, to produce candidate pairs | Not implemented | Use when all-pairs cost becomes a measured bottleneck on the GPU machine |
| `03` §5 midphase tree | Bounding-volume hierarchy for meshes | Not implemented; meshes are not in contact | Needed for mesh contact (the convex-hull mode of `SPECS.md` Section 9.1) |
| `03` §6.4 greedy farthest-point reduction | Reducing a large contact set to a fixed number of points that cover its extent | Not implemented | Use when a convex-hull contact produces more points than the solver budget |
| `03` §6.6 and §6.7 contact parameter mixing | The rule that combines friction and solver parameters of the two geoms in a pair | Friction uses the scene value; pair contact is frictionless | Define the mixing rule explicitly before adding pair friction |

### 2.6 Primitive contacts (parent `07-collision-pipeline.md`, `SPECS.md` Section 9.1 primitive mode)

| Example | What to take | Current state | Candidate work |
| --- | --- | --- | --- |
| `04` §1.3 plane-cylinder, up to four contacts | An exact rule that picks the cylinder rim points nearest the plane, with a stated order of operations | Cylinder rim sampled at eight points (`collision::plane_contact_points`) | Replace the sampling with the exact four-point rule, which gives a stable flat rest |
| `04` §1.4 plane-box, up to four contacts | Corner selection for a box on a plane, lower half only | Box corners, all eight | Compare with the four-point rule for the lower corners only |
| `04` §2.4 capsule-capsule, up to two contacts | Two contacts along a capsule pair, from the segment closest points | One contact from the closest points (`capsule_pair`) | Return the two-contact manifold when a capsule lies along another |
| `04` §3.3 capsule-box | A heuristic that places up to two contacts on the capsule axis | Covered by the general GJK path | Use if a capsule-box pair is common enough to need its own routine |
| `04` §3.4 box-box, separating axis theorem with face clipping, up to eight contacts | A complete contact manifold for two boxes, from the 15 candidate axes and clipping of the incident face | GJK and EPA give one contact point (`gjk_epa`) | Important for stable stacking of boxes. Add the SAT and clipping path |

### 2.7 Convex collision: GJK, EPA and multiple contacts (parent `07-collision-pipeline.md`, `SPECS.md` Section 9.1 convex-hull mode)

| Example | What to take | Current state | Candidate work |
| --- | --- | --- | --- |
| `05` §5.2 Minkowski support and §5.3 GJK loop | The support map of the difference body, and the loop that reduces the simplex | `gjk_epa` in `collision.rs`, with a simplex reduction written from first principles | Keep. Check against `05` §5.3 for iteration limits |
| `05` §5.4 signed-volume subdistance (Montanari and others, 2017) | A robust closest-feature computation for the simplex, with exact sign tests | The reduction uses the triangle and tetrahedron region tests of Ericson | Replace only if a degenerate-simplex failure is measured |
| `05` §6.3 EPA main loop | The stopping rule on the gap between the upper and lower bounds of the penetration, with an iteration cap | EPA stops when the new support point lies on the current face (tolerance `1e-10`), with a cap of 128 iterations | Adopt the bound-gap stopping rule, which gives a measured accuracy |
| `05` §8 MultiCCD | Multiple contact points for one convex pair, from clipped face polygons | One contact per pair | This is the manifold needed by the convex-hull mode, and the same role as `04` §3.4 for boxes |
| `05` §4.3 contact budget per pair | A stated maximum number of contacts for a pair, which sizes the buffers | Not applicable | Relevant to the GPU buffers (see Section 2.9) |

### 2.8 Fidelity, validation and agreement (parent `04-fidelity-loop.md`, `12-cpu-gpu-agreement.md`, `SPECS.md` Section 14)

| Example | What to take | Current state | Candidate work |
| --- | --- | --- | --- |
| `11` §10.3 finite-difference step and §10.5 inverse finite differences | Checking analytic derivatives against central finite differences with a stated step | The fidelity loop checks closed forms and conservation laws (`oracle.rs`) | Add a derivative check against finite differences where an analytic derivative is used |
| `15` §11 validation tolerances | Per-stage tolerances for a float32 port, separate from a float64 reference | Agreement tolerances in `tolerances.toml` for the GPU backend | Record the stage at which each tolerance applies, as the example does |
| `01` (verification at every stage) and `00_INDEX.md` §3 build order | Validate one stage at a time, before the next stage is added | The fidelity loop already runs stage by stage | No change |

### 2.9 GPU and batched execution (parent `11-gpu-batched-backend.md`)

| Example | What to take | Current state | Candidate work |
| --- | --- | --- | --- |
| `15` batching model and fixed buffers with overflow flags | Fixed-size per-world buffers, and an overflow flag that records dropped rows or contacts rather than failing | Static contact budgets are a stated fidelity cost (`11-gpu-batched-backend.md` 11.2) | Adopt the overflow flag, so that a dropped contact is visible in the run record |
| `08` §10.5 one tile per world for the line search | Work assignment by world, with threads striding over rows | Not implemented on GPU | Relevant when the solver moves to the GPU |
| `15` CUDA graph capture | Capturing a fixed launch sequence and replaying it | Not implemented | Relevant to the GPU machine; no current action |
| `15` (float32 policy) | Single precision on the GPU, with the cost of that choice recorded | The GPU backend is `robot-sim-gpu` | Record the precision of every stage in the run record |

### 2.10 Model compilation and scene import (parent `08-model-compilation-and-sensors.md`)

| Example | What to take | Current state | Candidate work |
| --- | --- | --- | --- |
| `13` (compile order, element numbering, fusing of static bodies) | Fusing a chain of bodies joined by fixed joints into one body, before simulation | Not implemented. The G1 has no fixed-joint chains of this kind in its simulated links | Keep as a candidate for other robots |
| `13` (mesh convex hull and qhull) | Computing the convex hull of a collision mesh once, at compile time | Mesh collision is not in contact (`unmodelled_collision_shape_count`) | Required for the convex-hull mode. Compute the hull in the importer, not per step |

### 2.11 Out of scope for this repository

These example sections describe features this project does not plan to use in the current specifications. They are listed so that a reader does not look for them in the mapped areas.

- `06` signed-distance-field collision and `16` flexible bodies. Deformables are not in `SPECS.md` Section 9.1.
- `14` ray casting, rendering and the batch API. The viewer and replay are written as USD and do not need them.
- `09` fluid models. No fluid is simulated.

## 3. Divergences to keep in mind

- The example describes a simulator whose defaults and tolerances were chosen for that engine. Our penalty stiffness of 2e4 N/m and damping of 200 N s/m are scene values, not defaults borrowed from it.
- The example's GPU sections describe a float32 implementation with specific algorithm variants. Our GPU backend (`robot-sim-gpu`) is a separate design, and the example's variants are not required to match it.
- Where the example gives an exact rule, such as the four-point cylinder rule or the eight-face box clipping, our current code approximates it. The approximation is a known gap, recorded here and not treated as a defect of the parent specification.

## 4. What this document does not do

It does not change `SPECS.md`, any file under `crates/robot-sim/specs/`, any acceptance row, or any decision record. Adopting an item from Section 2 requires its own decision record under the change-control rule of `SPECS.md`, and its own measurement before it is claimed.

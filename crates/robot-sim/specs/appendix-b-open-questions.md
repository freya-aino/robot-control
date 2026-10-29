# Appendix B — Open questions

[Index](README.md) · [Step 09](09-fidelity-tiers.md) · [Step 11](11-gpu-batched-backend.md) · [Step 08](08-model-compilation-and-sensors.md)

Source: old `SIMULATION_SPECS.md` lines 248–259, moved verbatim. The step files cross-reference these IDs in their "Open questions" fields (see the index).

Follows `SPECS.md` Appendix C's convention: each entry blocks something concrete and is closed by a decision record, not deleted silently.

| ID | Question | Blocks | Interim position |
| --- | --- | --- | --- |
| S-01 | Does strategy (b) (one GPU thread per environment, Section 11.1) actually outperform strategy (a) on our own fixtures, once both are prototyped? | Committing further engineering to either batching strategy beyond a first prototype | Build the smallest possible (b)-strategy prototype first (Section 11.1's decision), measure, revisit if it underperforms |
| S-02 | Is CubeCL's control-flow performance acceptable for an iterative constraint solver in practice, given zero found precedent either way? | Whether `robot-sim`'s GPU backend can be CubeCL-only, or needs a hand-written-WGSL fallback for the solver specifically (11.3) | Prototype the solver kernel in CubeCL first; fall back to `wgpu`/WGSL only for whatever specific kernel measurement shows it can't handle |
| S-03 | Exact named fidelity tiers and their per-knob settings (Section 12.2) | Any concrete tier implementation | Define tiers once a working CPU-reference-backend fidelity sweep exists to ground the choice in measured cost/accuracy, not a guess |
| S-04 | `Ball` joint (Section 3 item 1) — required for a first floating-base humanoid milestone, or deferrable? | Scope of the first `Free`-joint-enabled `robot-model` change | **Partially resolved 2026-10-26.** `JointKind::Free` is implemented (`robot_model::joint`), `ArticulatedBody`'s ABA/RNE are generalized to any DOF count and validated on it (closed-form, conservation and round-trip checks, plus the real G1 via `fixtures::unitree_g1_29dof_floating_base`, which now genuinely falls under gravity). `Ball` remains deferred; no near-term task needs a spherical joint specifically |
| S-05 | Geometry-based inertia inference (Section 10 item 1) — exact algorithm (uniform-density convex hull volume? a simpler bounding-shape approximation?) | Whether `robot-model`'s URDF importer gains this in the same change that adds `Free`/`Ball` joints or separately | Track as a separate, smaller change; not required for the floating-base milestone |
| S-06 | Whether `robot-sim`'s own sensor-noise injection point should live in the physics step or in `robot-perception` (Section 9 item 2) | Where noise-model code is authored | Default to `robot-perception` (existing crate boundary), revisit only if a sensor's noise is itself physics-coupled (e.g. contact-dependent tactile noise) |

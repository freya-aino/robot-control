# Scenes

`g1_ground.usda` is the G1 scene used by the simulator. It is an OpenUSD layer
that sublayers the vendored Unitree G1 layer
(`../robots/unitree_g1_usd/g1_29dof_rev_1_0.usd`, unmodified) and adds one
collider:

- `/ground` is a `Plane` with `PhysicsCollisionAPI`, normal along Z, at the
  height of its `xformOp:translate` (zero). `physics:dynamicFriction` is its
  Coulomb friction coefficient; `robot:contactStiffness` (N/m) and
  `robot:contactDamping` (N per m/s) are the penalty contact law of ADR 0010.

The ground was added here, in a separate layer, and not written into the
Unitree layer. The Unitree layer is pinned by SHA-256 in its PROVENANCE.md, and
changing it would break that provenance. The composed stage is the Unitree robot
plus this ground.

## Tags

Environment collision is decided by two tags in `sim:tags`, the same in every scene:

- `environment` and `collider`: the prim is a static collider, wherever it is in the
  stage. The simulator reads its shape, pose and contact parameters.
- `environment` alone: visual only, not in contact.
- No tag, or any other tag: not an environment collider.

Robots are recognised by their physics (mass, joints), not by tags. A tagged
collider of a shape the simulator cannot contact yet (box, sphere, cylinder, mesh)
is refused by name. Today the simulator contacts planes; a scene must contain
exactly one. `examples/tagging_example.usda` shows each case and
`examples/raised_plane.usda` a plane at 0.25 m. Run a scene with
`robot sim multibody --floating --scene <file>`.


# Using the suite

A working guide to this workspace: what is here, how to run it, and how the
pieces chain into a pipeline. Every command in this guide was run against the
revision that introduced it; where a command prints "unvalidated defaults",
that is the tool telling you the same thing this guide does — the pipeline
works end to end, but no threshold in it has been validated against a real
robot.

For *what the system is supposed to become*, read [SPECS.md](../SPECS.md);
for the delivery order, [PLAN.md](../PLAN.md); for what is true today, each
crate's `README.md` and this guide. Where they disagree, the crate README is
the more specific document and wins.

## 1. Setup

The workspace is devenv-managed Rust, std-only by default (ADR 0003): no
third-party crate is in the default build, so `cargo build` needs nothing
but the toolchain.

```sh
devenv shell        # or: direnv allow, if configured
devenv test         # the full gate: fmt check, clippy -D warnings,
                    # dependency exclusion gate, cargo test --workspace
```

Day-to-day:

```sh
cargo build -p robot-cli --release   # the `robot` binary -> target/release/robot
cargo test -p <crate>                # one crate's tests
cargo run -p robot-cli -- <command>  # run without installing
```

The one optional feature is the GPU backend: `cargo test -p robot-sim-gpu
--features cubecl` needs a Vulkan adapter and is off by default (ADR 0012).

## 2. The map of the workspace

Twenty-four crates, in dependency order. `robot-types` is the bottom;
`robot-cli` is the top and the program a user runs.

| Layer | Crates | What lives there |
| --- | --- | --- |
| Vocabulary | `robot-types` | Units, frames, time, typed ids, observation/action specs, the policy-bundle and episode schemas, evidence states |
| Artifacts | `robot-artifacts` | Hashing, the bounded TOML subset, atomic writes, blob store, manifests, the `robot-bundle/1` format, promotion/rollback |
| Robot description | `robot-model`, `robot-usd` | URDF and OpenUSD import, profiles, kinematics, convex hulls, robot discovery; a std-only USD reader/composer/writer |
| Simulation | `robot-sim`, `robot-sim-gpu` | Articulated dynamics (ABA/RNE), contacts, constraint solver, integrators, ideal sensors, the fidelity battery; a partial GPU port |
| Motion | `motion-io`, `motion-process`, `motion-retarget`, `motion-validate` | BVH/C3D/AMASS in, BVH out, capture protocol; filtering, contacts, segmentation; multi-task IK retargeting; kinematic/dynamic/tracking validation |
| Learning | `robot-nn`, `robot-learn`, `robot-tasks`, `robot-adapt` | MLP/GRU inference and the `robot-mlp/1` weights file; PPO; tasks and reward registries; the §11.3 adaptation ladder |
| Runtime | `robot-control`, `robot-perception`, `robot-supervisor`, `robot-hardware`, `robot-skills`, `robot-persona` | The deterministic control loop and tactical tier; state estimation; the independent safety supervisor; device transports and `robot-wire/1`; the skill router; bounded style |
| Evaluation | `robot-eval`, `robot-viewer` | Acceptance profiles, seed statistics, the A-01..A-50 matrix; SVG skeleton rendering and SMIL playback |
| Tooling | `robot-devtools`, `robot-cli` | The exclusion gate, dependency knowledge base, ledger diff; the `robot` command |

Three rules explain most of what you will notice while reading the code:

- **Nothing third-party without qualification.** The TOML parser, SHA-256,
  LZ4, DEFLATE, the USD crate-file reader, PCG random numbers and the linear
  algebra are all authored here, each against its reference specification
  (SPECS.md Appendix B.6 is the gate a real dependency must pass).
- **Refuse by name, never skip silently.** Parsers and importers error with
  the name of the unsupported construct. Expect `Unsupported: ...` rather
  than a half-read file.
- **Claims carry evidence.** Anything stated as a measurement names its run;
  anything unmeasured is labeled a default. The five evidence states and
  what each licenses are in `robot_types::evidence`.

## 3. Robot models

Three robots are vendored, each from Unitree's official OpenUSD release at
one pinned commit of `unitreerobotics/unitree_model` (BSD 3-Clause), each
with a `PROVENANCE.md` recording the commit, the SHA-256 of every layer, and
exactly what was verified at vendoring time:

| Robot | Assets | Fixture (in `robot_model::fixtures`) |
| --- | --- | --- |
| G1 humanoid, 29 DOF | `assets/robots/unitree_g1_usd/` (and the URDF in `assets/robots/unitree_g1/`) | `unitree_g1_29dof_usd()`, `unitree_g1_29dof()` |
| Go2 quadruped, 12 DOF | `assets/robots/unitree_go2_usd/` | `unitree_go2_12dof_usd()` |
| H1 humanoid, 19 DOF | `assets/robots/unitree_h1_usd/` | `unitree_h1_19dof_usd()` |

Each fixture has a `_floating_base` variant (free joint first) which is what
the simulator runs. All of them **refuse activation** — the files carry
masses, limits and drive gains, but no cited firmware or e-stop facts, and
SPECS.md §12.1 makes a verified independent e-stop a precondition of any
hardware-facing mode. They are for kinematics, simulation and retargeting.

To import a robot of your own:

- **USD**: `robot_model::usd::import_fixed_base(path)` composes the stage
  with the project's own reader (`robot-usd`: USDC and USDA, sublayers,
  references, payloads, variants, inherits/specializes, layer offsets) and
  reads links with `physics:mass`, `PhysicsRevoluteJoint` and
  `PhysicsFixedJoint` prims, collision scopes, and angular drives. Isaac-style
  robot layers (what Unitree ships) are the tested shape.
- **URDF**: `robot_model::urdf::import_from_str` — revolute, continuous,
  prismatic and fixed joints; `mimic` and `<transmission>` are refused by
  name; actuators come from `placeholder_actuator` because URDF carries no
  actuator truth.
- Inspect any USD layer with
  `cargo run -p robot-usd --example usd_dump -- <layer> [prim] [depth]`.

Vendoring discipline, if you add assets: pin the commit, record each file's
SHA-256, copy the license verbatim, and write the `PROVENANCE.md` before the
first use. The three existing files are the template.

## 4. Simulating

```sh
robot sim qualify                      # analytic pendulum energy-drift check
robot sim multibody                    # fixed-base G1, 2000 steps
robot sim multibody --floating --out /tmp/g1   # free base on the ground scene,
                                       # 2 s, writes CSV/JSONL/SVG frames + USDA replays
```

The floating run is judged on pelvis drop, ground penetration and settled
energy, and prints PASS, FAIL or INCONCLUSIVE with the numbers. Useful
flags: `--integrator euler|implicitfast|rk4`, `--actuation
drives|damping|zero` (the model's own USD drive gains are the default),
`--scene <usda>` for a different environment, `--dt`, `--steps`,
`--limit-margin`.

Replay a frame dump and render it:

```sh
robot replay /tmp/g1 --svg /tmp/g1/playback.svg --robot g1-usd --every 50
```

The SVG is side-and-front skeleton playback animated with SMIL — markup,
not script — and opens in any browser. The two `replay*.usda` files a
`--out` run writes are USD layers with `timeSamples`, loadable by anything
that reads USD.

In library code, the loop is `World::from_profile(&profile, SimOptions)`,
then `world.step(&mut state, &tau)` — see
`crates/robot-sim/tests/vendored_robots.rs` for the minimal version using
the Go2 and H1. Ideal sensors (encoders, IMU by finite differences) are
`robot_sim::sensors::SensorSuite`; noise is deliberately *not* there but in
`robot_perception::noise`, applied at a point the caller records.

## 5. The motion pipeline

The chain is: **import → retarget → validate**, and each stage's output is
content-addressed so the next stage can name its input exactly.

A minimal, runnable example is vendored at `fixtures/motion/guide-reach/`:
`reach.bvh`, a synthetic Y-up centimeter standing reach with
`RightShoulder`/`RightHand` and two feet (any BVH goes through the same
commands), and this `map.toml`:

```toml
schema = "retarget-map/1"
scale = 0.8

[[link]]
human = "RightHand"
robot = "right_rubber_hand"
relative_human = "RightShoulder"       # reach measured from the shoulder,
relative_robot = "right_shoulder_pitch_link"
ratio = "auto"                         # arm-length ratio from the skeletons

[[foot]]
site = 0
human = "LeftFoot"
robot = "left_ankle_roll_link"
weight = 100.0
ground_height = 0.0

[[foot]]
site = 1
human = "RightFoot"
robot = "right_ankle_roll_link"
weight = 100.0
ground_height = 0.0

[solver]
max_iterations = 40
tolerance = 1e-4
deadline_ms = 1000

[base]                                 # the pelvis may move, at this cost
translation_weight = 0.01
rotation_weight = 0.1

[posture]                              # prior toward the zero posture
weight = 0.001

[smoothing]
lambda = 2.0
```

Then:

```sh
cd fixtures/motion/guide-reach

# 1. Store the original before parsing a byte of it; print what it is.
robot data import reach.bvh --store store/
#    -> content id, rate, channels, validity, guessed up-axis and unit

# 2. Retarget onto the G1. Axes and units are converted (guessed or given
#    with --unit/--up), the ground plane and foot contacts estimated.
robot motion retarget reach.bvh --map map.toml --out reference.csv --robot g1
#    -> worst task error, planted-foot slip, unfinished frames; a
#       robot-reference/1 CSV

# 3. Validate the pair: kinematic limits, estimated torques, contact
#    consistency, simulated tracking. Exit 1 on rejection.
robot motion validate reach.bvh --reference reference.csv --map map.toml
#    -> a motion-curation/1 manifest entry, accepted or rejected with codes
```

On this guide's example the retarget reports a 2.2 mm worst task error and
zero slip, and validation accepts. Every threshold used is printed and
labeled an unvalidated default.

The same machinery as a library: `motion_io::bvh::parse`,
`motion_process::{condition, filter, contact, segment}`,
`motion_retarget::config::read_map` + `retarget::retarget`,
`motion_validate::validate`. C3D and AMASS `.npz` files go through
`motion_io::c3d` / `motion_io::amass`; live capture speaks the
`robot-capture/1` protocol in `motion_io::capture`.

## 6. Training and evaluation

```sh
robot train --out run/ [--iterations N] [--seed S]
```

trains PPO on the pendulum balance reference task (SPECS.md §9.2 orders
reference tasks before humanoids) with the configuration `robot-learn`
measured to stabilize it, and writes a checkpoint. No humanoid policy has
been trained in this workspace yet; the humanoid tasks (stand, push
recovery, tracking, reach, footwork, boxing pads in `robot-tasks`) are
implemented and tested, but a humanoid training run is open work.

Evaluation is split in two, and the distinction matters:

```sh
# Quality: seed results against an acceptance profile. Small-sample t
# intervals; pass, fail or inconclusive per metric.
robot eval run --profile profile.toml --results results.csv

# Integrity + activation: a robot-bundle/1 directory. The digest-sealed
# manifest and every content-addressed weight blob verify, then the
# activation gate runs against a vendored robot's real joint limits.
robot eval --bundle my-bundle/ --robot go2-usd
```

Exit 0 from `--bundle` means *verified and activatable in sim-eval against
that envelope* — the "wrong artifact deployed" control of SPECS.md §12.2 —
never a quality claim.

A bundle is built in code (`robot_artifacts::bundle::write_bundle`): the
full `PolicyBundleManifest` — observation and action specs, frozen
normalizer, tier declarations with *measured* compute, safety-envelope
assumptions, data lineage, transition contracts, benchmarks — plus one
weight blob per component (`robot-mlp/1` from `robot_nn::format` is the
first format). The writer refuses to seal a manifest over blobs it does not
describe; the reader verifies sizes before content and content before
interpretation. ADR 0016 records the format.

Acceptance at the project level is the 50-row matrix in `robot-eval`
(`matrix.rs`, SPECS.md §14): a row closes only through the evidence ledger's
"Acceptance rows" table, and all 50 are currently open.

## 7. Skills

A skill library lives in one reviewable file, `skill-manifest/1`:

```sh
robot skills list --manifest skills.toml
robot skills search jab --manifest skills.toml
```

The file holds the §10.2 typed descriptors — tags, embedding, embodiment
compatibility, initiation guards, termination, duration, contact
assumptions, memory, allowed transitions (`robot_skills::manifest`). The
search is the §10.1 lexical catalog (TF-IDF, no model call) and can only
rank; feasibility is the in-loop router's job (`robot_skills::selection`),
which checks guards, out-of-distribution distance, dwell and transition
contracts, and fails closed on an unknown guard.

## 8. Adaptation to a real robot's data

```sh
robot adapt identify joints.csv [--dead-band RAD_PER_S]
```

fits per-joint friction and damping (with standard errors) from a
`time,position,velocity,torque` log — method 2 of the SPECS.md §11.3
ladder. The full ladder (residual correction, history encoders, bounded
fine-tuning, retrieval, morphology randomization) is in `robot-adapt`,
which enforces the ladder's order: a later method is refused until the
earlier one's report exists.

## 9. The safety model, in one place

Worth understanding even if you never touch hardware, because it shapes
every API:

- **Modes** (`robot_types::mode`): `OfflineReplay → SimTrain → SimEval →
  HwShadow → HwSupported → HwFreestandingNoncontact → HwContact`. Each has
  entry preconditions; nothing hardware-facing activates from a profile
  without a cited independent e-stop.
- **The supervisor** (`robot-supervisor`) is independent of everything
  learned: fixed limits, torque-speed and thermal envelopes, saturation and
  contact-energy budgets, interlocks, a hazard register, and a fault table
  where the must-escalate set is not configurable.
- **The control loop** (`robot-control`) is allocation-free on its hot
  path, two-tiered (policy + tactical with decision age), feeds
  temperatures and contacts to the supervisor, and records incidents.
- **Activation gates** are everywhere deliberate: profiles refuse
  activation without firmware facts; bundles refuse activation on digest,
  privileged observations, tier or envelope mismatch; experts refuse
  admission on checksum or memory. If something refuses to start, it is
  usually telling you which claim has no evidence yet.

## 10. Disciplines that apply to any change

- Run `cargo fmt` and workspace clippy with `-D warnings`; the pre-commit
  hooks enforce both plus the dependency exclusion gate.
- Update the crate README you touched, the root README table if scope
  changed, and add a dated entry to `docs/ledger/automation-log.md` if a
  tool did the work.
- A new file format, dependency or spec-gap decision gets an ADR in
  `docs/decisions/` (sixteen exist; 0003 std-only, 0012 GPU, 0015 USD
  model source, 0016 bundle/skill formats are the load-bearing ones).
- Claims about behavior cite a run or a test; negative results are recorded
  rather than deleted (see `robot-learn`'s README for the worked example:
  the PPO failure that turned out to be a reward bug, and its correction).
- `robot deps` is deliberately not in the robot binary:
  `cargo run -p robot-devtools --bin devtools -- kb-sweep|kb-search|ledger-diff`
  sweeps the dependency knowledge base and diffs the evidence ledger.

## 11. What does not work yet

Honest edges, as of 2026-11-15:

- **`robot deploy`** exits with code 2: no hardware target is chosen
  (decision C-02). The wire protocol (`robot-wire/1`), a reference firmware
  peer with a watchdog, and the bundle verification it would compose exist.
- **robot-sim-gpu** is a qualified f32 subset (≤ 6 DOF dense kernels,
  integrators, plane-sphere contact) — not batched whole-robot dynamics.
  M-22 (f32 conditioning) and the missing hardware are recorded blockers.
- **No humanoid policy has been trained**; `robot train` is the pendulum
  reference task.
- **No acceptance row (A-01..A-50) is closed**; the matrix and ledger
  machinery to close them exist.
- The interactive viewer is SVG/SMIL only (no approved GUI dependency);
  C-17 keeps opponent memory session-scoped; S-08 (inertia realisability)
  and C-29 (commissioning probes) are open.

Exit codes everywhere: `0` success, `1` ran-and-found-a-problem, `2` not
implemented (with the reason on stderr), `64` usage.

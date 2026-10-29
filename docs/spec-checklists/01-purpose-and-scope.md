## 1. Purpose and scope

Build a Rust platform for acquiring human motion, learning physically feasible humanoid behavior, transferring it to different robot embodiments, and improving deployment from limited real-world experience. Humanoid robot boxing is the initial demanding application: stance, footwork, guard, strikes, avoidance, controlled contact, recovery, opponent modeling, and expressive behavior.

The platform must also support noncombat tasks without rebuilding the data pipeline, training infrastructure, or hardware integration. Boxing is a task package, not an assumption embedded in every controller.

The principal architecture is hierarchical:

1. Acquire and curate motion and robot observations.
2. Retarget human motion to a specific robot and validate feasibility.
3. Train a stable motion tracker and reusable skills in simulation.
4. Select skills using task policy, retrieval, or sparse expert routing.
5. Execute through a deterministic control runtime and hardware adapter.
6. Record selected deployment evidence, identify model mismatch, and validate adaptations before promotion.

Deliverables include reusable crates, command-line tools, declarative experiment configurations, versioned artifacts, simulator and hardware interfaces, evaluation suites, and operational documentation.

### 1.1 Relationship to prior repository contents

This repository previously contained a Python stack: `training/` (PyTorch, JAX, MuJoCo MJX, Warp, Hydra, MLflow, ONNX export), `unitree/` (CycloneDDS configuration and a Unitree SDK connector) and `test.ipynb`. That code was removed from the working tree on 2026-10-26, which closes the divergence from Section 2 that [C-01](#appendix-c--open-questions-and-unresolved-decisions) tracked. The working tree now contains documentation only, and the Rust workspace of Section 6 is unwritten; the specification states the disposition of the retired assets because they remain reachable in version control and are still cited as prior art.

| Retired asset | Disposition after removal |
| --- | --- |
| `training/` Python training code | Prior art and behavioral reference, retained only in history. It MAY be read at a named commit for algorithm intent. It MUST NOT be restored into the working tree, and any manual comparison run happens from a checkout of that commit in the separately named non-default environment of Section 2.2 item 8, with its numbers recorded as `unverified` baselines |
| `training/conf/` experiment configurations | Requirements input. The task, reward and randomization intent expressed there SHOULD be migrated into the configuration schema of Section 9.3, with the source commit cited; the YAML files themselves are not the canonical format and are not reintroduced |
| `unitree/` DDS/SDK connector | Protocol reference for `robot-hardware`. Field layouts, rates and command semantics observed there are hypotheses requiring the hardware verification of Section 7.1. Transcribing them into Rust is authorship against a cited historical reference, not a port |
| `test.ipynb` and notebook workflows | Scratch artifacts, deleted. Excluded from acceptance, reproducibility and artifact-lineage requirements, and not recreated in any form |

Four consequences are binding. The Rust workspace of Section 6 MUST NOT import, subprocess, or link against any of the above, and the removal does not weaken that prohibition — a deleted dependency and an excluded dependency are the same requirement. Numbers produced by the retired Python stack MUST NOT be reported as platform results without restating their provenance, including the commit they were produced at. Reintroducing any of this code, in whole or in part, is a Section 2.1 exception request and not a revert. Removal is not evidence: nothing about the Rust stack became more verified by deleting the prior art, and every evidence state of Section 3 is unchanged by it.



## Implementation checklist

Status legend: `[x]` implemented, `[~]` partial, `[ ]` not implemented, `[?]` not audited in this pass. Summary: 6 `[x]`, 2 `[~]`, 0 `[ ]`, 3 `[?]` across 11 items.

- [x] 1.1 Retired `training/` Python code is absent from the working tree — Evidence: no `training/` directory in the project root listing; `find` found no `*.py` or `*.ipynb` files.
- [x] 1.2 Retired `unitree/` DDS/SDK connector is absent from the working tree — Evidence: no `unitree/` directory. `robot-model` vendors only a URDF asset under `assets/robots/unitree_g1/`, which is a model file, not the connector.
- [x] 1.3 `test.ipynb` is absent from the working tree — Evidence: no notebook files found.
- [x] 1.4 The Rust workspace does not import, subprocess or link the retired stack — Evidence: all `[dependencies]` are path crates (see 2.4); `grep` matches are comments only.
- [~] 1.5 Hierarchical architecture: acquire, retarget, train tracker, select skills, execute through control runtime, record evidence — Evidence: crates exist for each stage (`motion-io`, `motion-retarget`, `robot-learn`, `robot-skills`, `robot-control`, `robot-hardware`, `robot-artifacts`). Note: `crates/robot-control/README.md` says the tactical tier that selects skills is not wired into `ControlLoop::tick`.
- [~] 1.6 Deliverables include reusable crates and command-line tools — Evidence: 23 crates and a `robot-cli` crate exist. Note: the CLI commands read are `sim` subcommands; other subcommands were not audited.
- [?] 1.7 Declarative experiment configurations — not audited (no config schema crate was read in this pass).
- [x] 1.8 Versioned artifacts — Evidence: `crates/robot-artifacts/README.md` (manifests, lineage, promotion ledger, atomic writes).
- [x] 1.9 Simulator and hardware interfaces — Evidence: `crates/robot-sim/README.md` (`Environment` trait) and `crates/robot-hardware/README.md` (`HardwareAdapter` trait, loopback and replay devices).
- [?] 1.10 Evaluation suites — `crates/robot-eval` exists; its README was not read in this pass.
- [?] 1.11 Noncombat tasks run without rebuilding the pipeline; boxing is a task package — `crates/robot-tasks` exists; its README was not read in this pass.

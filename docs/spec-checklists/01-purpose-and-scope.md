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

This repository previously contained a Python stack: `training/` (PyTorch, JAX, the reference simulator batched backend, Warp, Hydra, MLflow, ONNX export), `unitree/` (CycloneDDS configuration and a Unitree SDK connector) and `test.ipynb`. That code was removed from the working tree on 2026-10-26, which closes the divergence from Section 2 that [C-01](#appendix-c--open-questions-and-unresolved-decisions) tracked. The working tree now contains documentation only, and the Rust workspace of Section 6 is unwritten; the specification states the disposition of the retired assets because they remain reachable in version control and are still cited as prior art.

| Retired asset | Disposition after removal |
| --- | --- |
| `training/` Python training code | Prior art and behavioral reference, retained only in history. It MAY be read at a named commit for algorithm intent. It MUST NOT be restored into the working tree, and any manual comparison run happens from a checkout of that commit in the separately named non-default environment of Section 2.2 item 8, with its numbers recorded as `unverified` baselines |
| `training/conf/` experiment configurations | Requirements input. The task, reward and randomization intent expressed there SHOULD be migrated into the configuration schema of Section 9.3, with the source commit cited; the YAML files themselves are not the canonical format and are not reintroduced |
| `unitree/` DDS/SDK connector | Protocol reference for `robot-hardware`. Field layouts, rates and command semantics observed there are hypotheses requiring the hardware verification of Section 7.1. Transcribing them into Rust is authorship against a cited historical reference, not a port |
| `test.ipynb` and notebook workflows | Scratch artifacts, deleted. Excluded from acceptance, reproducibility and artifact-lineage requirements, and not recreated in any form |

Four consequences are binding. The Rust workspace of Section 6 MUST NOT import, subprocess, or link against any of the above, and the removal does not weaken that prohibition — a deleted dependency and an excluded dependency are the same requirement. Numbers produced by the retired Python stack MUST NOT be reported as platform results without restating their provenance, including the commit they were produced at. Reintroducing any of this code, in whole or in part, is a Section 2.1 exception request and not a revert. Removal is not evidence: nothing about the Rust stack became more verified by deleting the prior art, and every evidence state of Section 3 is unchanged by it.



## Implementation checklist

Status legend: `[x]` implemented, `[~]` partial, `[ ]` not implemented, `[?]` not audited in this pass. Summary: 6 `[x]`, 5 `[~]`, 0 `[ ]`, 0 `[?]` across 11 items.

- [x] 1.1 Retired `training/` Python code is absent from the working tree — Evidence: `ls` of the project root shows no `training/`; `find . -name '*.py' -o -name '*.ipynb'` (excluding `target/`) finds nothing.
- [x] 1.2 Retired `unitree/` DDS/SDK connector is absent from the working tree — Evidence: no `unitree/` directory; `assets/robots/unitree_g1/` holds only a URDF model asset, not the connector.
- [x] 1.3 `test.ipynb` is absent from the working tree — Evidence: the same `find` finds no notebook.
- [x] 1.4 The Rust workspace does not import, subprocess or link the retired stack — Evidence: `[workspace.dependencies]` is empty (`Cargo.toml:40`); every crate manifest lists only `path` dependencies except `crates/robot-sim-gpu/Cargo.toml:16-17` (optional `cubecl`, behind the non-default `cubecl` feature); `grep -rn "Command::new\|python\|pyo3\|extern \"C\"" crates` finds one `std::process::Command` use, which re-executes the test binary itself inside `#[cfg(test)]` (`crates/robot-learn/src/checkpoint.rs:1403`).
- [~] 1.5 Hierarchical architecture: acquire, retarget, train tracker, select skills, execute through control runtime, record evidence — Evidence: one crate per stage exists (`motion-io`, `motion-retarget`, `robot-learn`, `robot-skills`, `robot-control`, `robot-hardware`, `robot-artifacts`) and `ControlLoop::tick` runs poll, infer, supervise, transmit (`crates/robot-control/README.md:12`). Note: skill selection is not wired into the loop; `crates/robot-control/README.md:38-39` states that `robot-skills` is a declared dependency but `ControlLoop::tick` does not query a tactical tier, and no trained motion tracker exists (`robot-learn` trains the pendulum fixture only).
- [~] 1.6 Deliverables include reusable crates and command-line tools — Evidence: 23 crates under `crates/`; the `robot` binary dispatches `motion`, `sim` and `eval` (`crates/robot-cli/src/main.rs:23-25`), while `train`, `skills`, `deps`, `adapt`, `deploy` and `replay` return `EXIT_NOT_IMPLEMENTED` (`main.rs:26-28`) and `motion retarget` is a stub (`main.rs:59`); `robot eval` covers only `summarize` over command-line values (`crates/robot-cli/src/commands/eval.rs:1-8`).
- [~] 1.7 Declarative experiment configurations — Evidence: configurations are Rust structs (`PpoConfig` at `crates/robot-learn/src/ppo.rs:50`, `BalanceTaskConfig` at `crates/robot-tasks/src/balance.rs:13`, `ControlLoopConfig` at `crates/robot-control/src/control_loop.rs:146`) and `PpoConfig` is serialized into the checkpoint manifest (ADR 0007). Note: no CLI command reads an experiment configuration file, and no declarative config schema (the §9.3 migration target for `training/conf/`) exists; `grep -rn "config" crates/robot-cli/src/commands/eval.rs` finds none.
- [x] 1.8 Versioned artifacts — Evidence: `crates/robot-artifacts/src/{hash,manifest,store,chunk}.rs` (content hashing, `ArtifactManifest`, `Lineage`, `PromotionLedger`, `write_immutable`) and the checkpoint manifest of `docs/decisions/0007-checkpoint-manifest-layout.md`.
- [x] 1.9 Simulator and hardware interfaces — Evidence: `pub trait Environment` (`crates/robot-sim/src/env.rs:26`), `pub trait Backend: Environment` (`crates/robot-sim/src/backend.rs:14`), `pub trait HardwareAdapter` (`crates/robot-hardware/src/device.rs:87`) with loopback and replay devices (`crates/robot-hardware/Cargo.toml:8`).
- [~] 1.10 Evaluation suites — Evidence: `crates/robot-eval` provides `AcceptanceProfile` (immutable once evaluated) and `stats::summarize` with 6 tests (`crates/robot-eval/README.md`). Note: no suite runner exists; `crates/robot-cli/src/commands/eval.rs:4-5` says "the full bundle/suite-driven evaluation needs a bundle format and suite runner this workspace does not have yet".
- [~] 1.11 Noncombat tasks run without rebuilding the pipeline; boxing is a task package — Evidence: `crates/robot-tasks` implements a `RewardRegistry` and a `BalanceTask` on `PendulumEnv` (`crates/robot-tasks/README.md:10-13`), so the one existing task is noncombat and separate from the learner. Note: no boxing task package exists (`grep -rli boxing crates/robot-tasks crates/robot-skills` finds nothing), so the "task package, not an assumption" property is untested with a second task.

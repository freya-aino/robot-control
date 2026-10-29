## 6. Proposed workspace and component responsibilities

These are planned logical boundaries; they do not describe or depend on existing repository contents.

| Crate or package family | Responsibility |
| --- | --- |
| `robot-types` | Units, frames, time, joint IDs, typed observations/actions, schemas |
| `robot-model` | Kinematic tree, inertias, actuator limits, geometry, sensor placement |
| `motion-io` | Motion import/export and synchronized capture adapters |
| `motion-process` | Resampling, filtering, contacts, segmentation, augmentation |
| `motion-retarget` | Robot mapping, constrained IK, temporal optimization |
| `motion-validate` | Kinematic and simulated feasibility reports |
| `robot-sim` | Simulation interface, Rust physics backend, batching, randomization |
| `robot-tasks` | Tracking, balance, reach, footwork, boxing and perturbation tasks |
| `robot-learn` | Rollouts, autodiff, PPO baseline, losses, optimizers, checkpoints |
| `robot-skills` | Latent models, expert bank, transition contracts, retrieval index |
| `robot-adapt` | Identification, adaptation encoders, correction models, calibration |
| `robot-perception` | State estimation, contacts, opponent tracking and uncertainty |
| `robot-control` | Deterministic runtime, inference, arbitration, supervision |
| `robot-hardware` | Device traits, protocol adapters, replay and loopback devices |
| `robot-eval` | Benchmarks, statistical reports, regression gates |
| `robot-artifacts` | Content hashes, manifests, lineage, signing and promotion |
| `robot-persona` | Optional identity and bounded expressive intent |
| `robot-cli`, `robot-viewer` | Experiment tooling, inspection, synchronized playback |
| `robot-devtools` | Development-time automation and the dependency knowledge base of §13.1: descriptor and dataset handling, integration and parsing helpers, test-run orchestration, ledger and inventory bookkeeping. Developer and CI hosts only; never a dependency of any deployable crate |

Training and deployment share observation/action schemas and tested normalization logic. Deployment does not depend on training autograd code. Avoid a plugin design that permits arbitrary dynamic code loading in the control process: build-time Rust implementations are the default, and optional experimental workers run outside that process.

The v3.0 survey proposes an additional `robot-brain` package hosting agent orchestration, opponent memory and operator-facing advisory agents. No such package is created by this revision. Its responsibilities, where they are adopted at all, are an optional out-of-process extension of `robot-persona` and the tooling crates, and they are subject to Section 2.3 in full: feature-gated, absent from the default build, unreachable from `robot-control`, `robot-hardware` and the supervisor, and removable without affecting them. Whether the boundary justifies its own package is open under [C-16](#appendix-c--open-questions-and-unresolved-decisions); a survey's proposed crate name is not an architectural requirement, as already stated for `robot-state`, `robot-ctrl` and `robot-test` in Appendix B.3.

The three admissible uses of Section 2.3.1 attach to this layout as follows: development-time automation (U1) lives in `robot-devtools` and the `devenv` scripts of Section 2.2 and is a `dev-dependency` of nothing that ships; assisted sampling (U2) is an offline producer of configurations consumed by `robot-learn`, `robot-tasks` and `robot-persona`, with no reverse dependency; and the dependency knowledge base (U3) is records in this repository with tooling in `robot-devtools`. A dependency edge from `robot-control`, `robot-hardware`, `robot-skills` or any bundle-producing crate to any of the three is a policy violation, detectable by the exclusion gate of Section 2.1 and asserted by row A-25.

Use Appendix B.3 to select initial dependency candidates for these crate families. Candidate selection does not authorize dependencies that conflict with Section 2.



## Implementation checklist

Status legend: `[x]` implemented, `[~]` partial, `[ ]` not implemented, `[?]` not audited in this pass. Summary: 5 `[x]`, 1 `[~]`, 0 `[ ]`, 3 `[?]` across 9 items.

- [x] 6.1 Every crate or package family in the responsibility table exists — Evidence: all 20 named crates present under `crates/` (`crates/*/Cargo.toml` glob, 23 results including `robot-nn` and `robot-sim-gpu`).
- [~] 6.2 Each crate implements only its listed responsibility — Note: only boundaries were checked, not the content of each crate; not audited crate by crate in this pass.
- [x] 6.3 No deployable crate depends on `robot-devtools` — Evidence: `grep` of `crates/*/Cargo.toml` finds `robot-devtools` in its own manifest only.
- [x] 6.4 `robot-persona` is not a dependency of the control path — Evidence: `robot-persona` appears only in `crates/robot-cli/Cargo.toml` line 33; `robot-control` does not depend on it.
- [?] 6.5 Deployment does not depend on training autograd code — `robot-control` and `robot-hardware` dependency lists were not read in full.
- [?] 6.6 No dynamic code loading in the control process — not audited.
- [x] 6.7 No `robot-brain` package is created — Evidence: no `crates/robot-brain` directory.
- [x] 6.8 `robot-control`, `robot-hardware` and `robot-skills` have no dependency edge to the U1, U2 or U3 tooling — Evidence: `grep` shows `robot-devtools` is depended on by nothing, and `robot-persona` is absent from these three manifests.
- [?] 6.9 Training and deployment share observation/action schemas through `robot-types` — `robot-types` exists; the shared-use claim was not traced.

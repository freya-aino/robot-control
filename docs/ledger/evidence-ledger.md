# Evidence ledger

SPECS.md §3: "Record source URLs, title, authors, version/date, accessed
date, code commit, license, evidence type, and the exact supported claim
before relying on research in an implementation decision." This ledger is
the dated record that promotion and demotion in `robot_types::evidence` is
checked against; the type system enforces the *rules*, this file is the
*data*.

Entries below transcribe SPECS.md §3.1's research intake register and record
the evidence state each lead starts this project at. No entry here has been
promoted beyond what is stated; promotion requires a dated artifact
reference, which is added to this file when (and only when) that artifact
exists.

## Convention

Each entry: `lead` (identity), `claim` (the exact thing it is cited for),
`state` (one of the five SPECS.md §3 states), `admitting artifact`
(what licenses the current state), `accessed` (date), `disposition`.

## Primary methodology references

| Lead | Claim | State | Admitting artifact | Accessed | Disposition |
| --- | --- | --- | --- | --- | --- |
| RoboStriker, arXiv:2601.22517 (v1, 2026-01-30) | Staged latent competitive-learning decomposition (SPECS.md §9.4.1): tracker → distillation → warm-up → co-evolution, as a reference ordering | `paper_verified` | Primary source retrieved, method located in text | 2026-10-26 | Reference decomposition only. No source release identified. Every reported number (32-dim hypersphere, 50 Hz, 29-DoF robot, reward weights, reservoir capacity) is a condition of that foreign-simulator experiment, not inherited as a setting (§3, §9.4.1) |
| NFSP (Heinrich & Silver, arXiv:1603.01121) | Basis of opponent-averaging via fictitious play | `unverified` | — | 2026-10-26 | Cited by RoboStriker; original convergence conditions not independently verified in this revision |
| AMP (arXiv:2104.02180), ASE, DeepMimic | Reference algorithms for the style/motion-prior term of §9.4.2 item 5 | `unverified` | — | 2026-10-26 | A discriminator reward is a style constraint only; never a feasibility/safety claim per §9.4.2 item 5 |
| BehaviorRetrieval, FlowRetrieval, RAEA, RAG-Diff | Retrieval-augmented imitation/adaptation structure for §9.7 | `unverified` | — | 2026-10-26 | Retrieval unit, embedding source and offline/between-episode/in-loop tier (§10.6) not yet individually verified per lead |
| RMA / A-RMA | Reference form of the two-phase latent identification row of §11.3 | `unverified` | — | 2026-10-26 | Adaptation-module training procedure and deployment observation set not yet verified |
| SGFT (ICLR 2025) | Candidate for the simulation-guided fine-tuning row of §11.3 | `unverified` | — | 2026-10-26 | Requires on-hardware updates; not scheduled absent the Section 4 mode gating and a separate C-24 decision |
| MOSAIC, H-Zero, PolySim, ZEST | Residual adaptation / cross-embodiment / multi-simulator leads | `unverified` | — | 2026-10-26 | Collected during §9.7/§11.3 survey; none is a dependency |
| Unitree G1 boxing / Iron Fist King | Public product behavior only | `unverified` (product demo) | — | 2026-10-26 | Training internals treated as unknown; a demo does not establish architecture |
| GMR, PHC, H2O/OmniH2O, ExBody, ProtoMotions | Retargeting/imitation/whole-body-control reference algorithms | `unverified` | — | 2026-10-26 | Rust implementations required; none exists in this workspace yet |
| ASAP, arXiv:2502.01143 | Learning simulator/action mismatch from real trajectories | `unverified` | — | 2026-10-26 | Correction-operates-where (train vs. deploy) not yet verified |
| SPiRL, NPMP, ControlVAE, PhysicsVAE, NCP | Skill-prior / latent-controller references | `unverified` | — | 2026-10-26 | A simple reproducible baseline is required before any extension |
| STRAP (arXiv:2412.15182), SMOSE | Retrieval / sparse-expert control references | `unverified` | — | 2026-10-26 | Objective, retrieval unit, benchmark and latency evidence not yet verified |
| ARLI, DARP, ReGIL, MABR, COLLIE, SafeFlow, InstantMimic, H-Zero, R²S², LATENT | Delay-aware control / retrieval / reusable-representation leads | `unverified` | — | 2026-10-26 | No core dependency |
| PAMoR, SocialHumanoid, StreamTalk, LiveGesture, PersonaGestor, Animus | Optional personality/expressive-motion leads | `unverified` | — | 2026-10-26 | Separate from motion safety and baseline delivery |
| `adk-rust` (zavora-ai), `rust-adk` (Yougigun), Google ADK | Agent orchestration — name-collision warning | `unverified` | — | 2026-10-26 | Three unrelated projects share "ADK"; every future reference MUST name the owning project and exact crate (§3.1, Appendix B.1 correction 13) |
| "JitRL", "Learn-to-Teach 2507.XXXXX" | Possible topics | n/a — rejected | — | 2026-10-26 | Placeholder identifiers; not usable citations |

## Rust ecosystem survey candidates (Appendix B)

Every Appendix B candidate starts `unverified` per Appendix B.1 ("All
incoming entries start at `unverified` under Section 3 until a primary-source
record establishes a higher evidence state... the supplied `paper_verified`
label for HULKs and `code_inspected` label for Burn are not inherited
automatically"). None has been promoted in this revision. See
`docs/decisions/0003-std-only-default-closure.md` for the consequence this
has for the current dependency closure, and `docs/knowledge-base/dependencies/`
for the per-candidate inventory entries as they are opened.

## This project's own artifacts

| Artifact | Claim | State | Admitting artifact | Date |
| --- | --- | --- | --- | --- |
| `robot-types` crate | Unit tests pass; module-level invariants hold | `reproduced_in_sim`-adjacent (code-level, not a simulation claim) | `cargo test -p robot-types`: 158 passed, 0 failed | 2026-10-26 |
| `robot-artifacts` crate | Unit tests pass | same | `cargo test -p robot-artifacts`: 63 passed, 0 failed | 2026-10-26 |
| `robot-model` crate | Unit tests pass | same | `cargo test -p robot-model`: 66 passed, 0 failed | 2026-10-26 |
| devenv-locked environment bootstraps | `devenv shell` reaches a working Rust toolchain with no manual steps; `devenv test` passes | `reproduced_in_sim`-adjacent (host-local, not yet cross-host) | This session's `devenv test` run: fmt/clippy/test all green under the committed `devenv.lock` | 2026-10-26 |

None of the rows above is a `validated_on_hardware` claim, and none involves
a robot. No row in this table licenses a statement about any other host,
embodiment, firmware or acceptance profile (SPECS.md §3).

## Demotion log

(Empty. No dependency version, robot profile, firmware or acceptance profile
has changed since this ledger was opened.)

# Humanoid Robotics Platform Specification

Status: proposed architecture and acceptance contract. Not ratified; no clause below is evidence that the described system exists.

| Field | Value |
| --- | --- |
| Document version | 0.7.3-draft |
| Last revised | 2026-10-27 |
| Companion documents | `PLAN.md` (phasing, gates, backlog); evidence ledger; decision records; per-robot acceptance and safety profiles; `devenv.nix`, `devenv.yaml` and `devenv.lock` as the executable form of Section 2.2 |
| Normative scope | Sections 1–15 bind the implementation. Appendix A is definitional, Appendix B is advisory until its gates pass, Appendix C records what is still undecided |
| Change control | Changes to Sections 2, 4, 5, 7, 12 or 14 require a decision record and a review of every dependent acceptance profile, fixture and consumer crate |

### Revision history

| Version | Change | Date |
| --- | --- | --- |
| 0.1 | Initial specification, Sections 1–15 | — |
| 0.2 | Appendix B replaced with the supplied Rust ecosystem survey revised v2.0 | — |
| 0.3.0-draft | Document control, normative-language convention, glossary (Appendix A), operating-mode and evidence-promotion tables, foreign-dependency exception procedure, safety case (Section 12), acceptance traceability, open-questions register (Appendix C) | 2026-10-24 |
| 0.3.1-draft | devenv-provisioned development environment (Section 2.2), locked-environment recording in Section 13, acceptance row A-22, glossary and open-question entries. Touches Sections 2 and 14, so the change-control rule applies and the decision record is tracked as [C-14](#appendix-c--open-questions-and-unresolved-decisions) until written | 2026-10-24 |
| 0.4.0-draft | Appendix B replaced with the supplied Rust ecosystem survey revised v3.0. Adds the external-model-service dependency class (Section 2.3), the agent-advisory boundary in Sections 5 and 12.6, the optional agent layer in Section 6, acceptance rows A-23 and A-24, glossary entries and open questions C-15 to C-17. Touches Sections 2, 5, 12 and 14, so the change-control rule applies and the decision record is tracked as [C-15](#appendix-c--open-questions-and-unresolved-decisions) until written | 2026-10-25 |
| 0.5.0-draft | Supplied ADK-Rust integration appendix folded into the existing structure instead of being added as an appendix. Specifies the three admissible uses of an agent layer — development-time automation, offline policy and personality sampling assistance, and a maintained low-level dependency knowledge base — in Section 2.3.1, with the knowledge base in Section 13.1, sampling assistance in Sections 9.5 and 12.5, acceptance rows A-25 to A-27, qualification gate 10, risk rows and open questions C-18 and C-19. Touches Sections 2, 12, 13 and 14, so the change-control rule applies and the decision record remains tracked as [C-15](#appendix-c--open-questions-and-unresolved-decisions) | 2026-10-25 |
| 0.6.0-draft | Adds extensible robot descriptor and shared experience/replay contracts in Sections 7.1 and 7.6, RL extension points and conformance requirements in Sections 9.3 and 9.6, Rust mechanism candidates in Appendix B, implementation guidance in Appendix D, acceptance rows A-28 to A-31, and open questions C-20 and C-21. Touches Sections 7 and 14, so the change-control rule applies and the decision record is tracked as [C-20](#appendix-c--open-questions-and-unresolved-decisions) and [C-21](#appendix-c--open-questions-and-unresolved-decisions) until written | 2026-10-26 |
| 0.7.0-draft | Folds the surveyed low-data humanoid learning methods into the existing structure: the hierarchical latency invariant in Section 5.1, the staged latent competitive pipeline of Section 9.4 following the now `paper_verified` RoboStriker method, retrieval-augmented policy learning in Section 9.7, retrieval tiers and deadline attribution in Section 10.6, expanded limited-data adaptation methods in Section 11.3, three further extension points and two storage clauses in Sections 9.3 and 7.6, research-register and Appendix A/B entries, acceptance rows A-32 to A-35, and open questions C-22 to C-24. Touches Sections 5, 7 and 14, so the change-control rule applies and the decision record is tracked as [C-22](#appendix-c--open-questions-and-unresolved-decisions) until written | 2026-10-26 |
| 0.7.1-draft | Records the removal of the Python prior art from the working tree: Section 1.1 is restated as the disposition of retired assets reachable only in version control, Section 2.2 item 8 points manual comparison runs at a historical commit, and C-01 is resolved with reintroduction routed through the Section 2.1 exception procedure. Touches Sections 1 and 2, so the change-control rule applies and the migration decision record is the resolution of [C-01](#appendix-c--open-questions-and-unresolved-decisions) | 2026-10-26 |
| 0.7.2-draft | Revises the simulation target, by explicit project direction, to require GPU-bound batched execution, variable fidelity and high-fidelity fidelity, detailed in a new companion document rather than in this file: `crates/robot-sim/SIMULATION_SPECS.md`. Section 9.1 cross-references it without relaxing the Rust-native requirement; Appendix B.2.1, B.2.3 and B.4 gain cross-references and one new qualification item (B.4 item 8, GPU-vs-CPU-reference agreement before any throughput claim). Does not touch Sections 2, 4, 5, 7, 12 or 14, so the Section-level change-control rule is not triggered; the new open question is tracked as [C-25](#appendix-c--open-questions-and-unresolved-decisions) and the decision itself is `docs/decisions/0005-gpu-batched-simulation-target.md` | 2026-10-26 |
| 0.7.3-draft | Revises the draft against the state of the working tree on 2026-10-27: Section 9.1 records that the floating-base multibody path has no ground contact, Section 9.2 requires checkpoints to carry the observation normalizer and adds parameter-group freezing for fine-tuning, Section 9.3 states that the policy, backend and model-importer seams are not yet traits and must be introduced behavior-preservingly, and test-purpose rendering is specified as a frame dump before any viewer dependency. Adds acceptance rows A-36 to A-39 and open questions C-26 to C-28. Touches Section 14, so the change-control rule applies and the decision record is tracked as [C-26](#appendix-c--open-questions-and-unresolved-decisions) until written | 2026-10-27 |
| 0.7.4-draft | Updates Section 9.1 to record that the floating-base multibody path now has sphere-only ground contact and a qualified supported-stance hold, with its dt range and limits, replacing the earlier statement that the path has no ground contact. Balance, standing and push-recovery claims remain forbidden until contact proxies and a controller exist. Decision record: `docs/decisions/0010-ground-contact-penalty-model.md`. Touches Section 9.1 only; the Section-level change-control rule is met through that record | 2026-10-27 |

### Normative language

Key words are used in the sense of RFC 2119/8174 when capitalized:

| Keyword | Meaning |
| --- | --- |
| MUST / MUST NOT | Binding. A violation blocks promotion and must be recorded as a defect, not a known limitation |
| SHOULD / SHOULD NOT | Strong default. Deviation requires a written rationale in the relevant decision record |
| MAY | Permitted option carrying no obligation |

Uncapitalized prose in this document is descriptive context, rationale or a correction to an earlier premise. Where descriptive prose and a capitalized requirement conflict, the capitalized requirement governs. Tables in Sections 7, 11 and 14 are requirements; tables in Appendix B are candidate registers, not approvals.

### How to read this document

Sections 1–5 fix policy: what is built, in what language, on what evidence, in which operating modes, under which timing constraints. Sections 6–13 fix contracts: components, artifacts, pipelines, runtime behavior and operations. Section 14 is the acceptance contract and Section 15 bounds the first release. Appendix A defines terms, Appendix B qualifies dependencies, Appendix C lists unresolved decisions that currently block requirements.

### Contents

1. [Purpose and scope](#1-purpose-and-scope)
2. [Language and dependency policy](#2-language-and-dependency-policy)
3. [Evidence policy and corrections to the research premise](#3-evidence-policy-and-corrections-to-the-research-premise)
4. [Operating modes and boundaries](#4-operating-modes-and-boundaries)
5. [Timing and execution model](#5-timing-and-execution-model)
6. [Proposed workspace and component responsibilities](#6-proposed-workspace-and-component-responsibilities)
7. [Robot and artifact contracts](#7-robot-and-artifact-contracts)
8. [Motion acquisition and retargeting](#8-motion-acquisition-and-retargeting)
9. [Simulation, training and methodology extension](#9-simulation-training-and-methodology-extension)
10. [Retrieval-indexed policy bank and low latency](#10-retrieval-indexed-policy-bank-and-low-latency)
11. [Transfer and data-efficient improvement](#11-transfer-and-data-efficient-improvement)
12. [Safety, supervision, perception and personality](#12-safety-supervision-perception-and-personality)
13. [Reproducibility and operational requirements](#13-reproducibility-and-operational-requirements)
14. [Acceptance and evaluation matrix](#14-acceptance-and-evaluation-matrix)
15. [Explicit non-goals for the first release](#15-explicit-non-goals-for-the-first-release)

- [Appendix A — Terminology, units and conventions](#appendix-a--terminology-units-and-conventions)
- [Appendix B — Rust ecosystem survey and dependency qualification](#appendix-b--rust-ecosystem-survey-and-dependency-qualification-revised-v30)
- [Appendix C — Open questions and unresolved decisions](#appendix-c--open-questions-and-unresolved-decisions)
- [Appendix D — Rust extensibility patterns](#appendix-d--rust-extensibility-patterns)



## Implementation checklist

Status legend: `[x]` implemented, `[~]` partial, `[ ]` not implemented, `[?]` not audited in this pass. Summary: 4 `[x]`, 3 `[~]`, 1 `[ ]`, 0 `[?]` across 8 items.

- [~] 0.1 Changes to Sections 2, 4, 5, 7, 12 or 14 are recorded in a decision record — Evidence: `docs/decisions/` holds 12 records; `0006-policy-and-loop-operation-traits.md:6` touches §5 and §12, `0007-checkpoint-manifest-layout.md:6` touches §7.4, `0010-ground-contact-penalty-model.md` is cited by revision 0.7.4-draft. Note: the revision-history rows 0.3.1 to 0.7.3 each say their change-control record is "tracked as C-14/C-15/C-20/C-21/C-22/C-26 until written"; only C-14 has a (partial) record (`0001-devenv-toolchain-and-platforms.md:5`), and `grep -n "C-[0-9][0-9]" docs/decisions/*.md` finds no record for C-15, C-20, C-21, C-22 or C-26.
- [~] 0.2 Dependent acceptance profiles, fixtures and consumer crates are reviewed on each such change — Evidence: every ADR has a `Touches:` line naming spec sections and crate files (for example `0006-policy-and-loop-operation-traits.md:6`, `0009-test-render-format.md:7-11`), and ADR 0006 "Consequences" (line 43) records that the consumer crates' tests are unchanged. Note: no ADR names a dependent acceptance profile because none exists (`docs/acceptance/` is empty; `robot-eval` has the `AcceptanceProfile` type at `crates/robot-eval/src/acceptance.rs:36` but no profile instance is checked in).
- [x] 0.3 Revision history is maintained — Evidence: `SPECS.md:17-28` lists versions 0.1 to 0.7.4-draft with dates and change-control notes.
- [x] 0.4 Document version matches the latest revision — Evidence: `SPECS.md:7` reads `0.7.4-draft`, matching the last revision-history row (`SPECS.md:28`, dated 2026-10-27, the same date as the "Last revised" field); corrected on 2026-10-30.
- [x] 0.5 Normative-language convention (RFC 2119/8174 keywords) is stated — Evidence: the "Normative language" table at `SPECS.md:30-38`.
- [~] 0.6 Companion documents exist: PLAN.md, evidence ledger, decision records, per-robot acceptance and safety profiles, devenv files — Evidence: `PLAN.md`, `docs/ledger/evidence-ledger.md`, `docs/decisions/0001`–`0012`, `devenv.nix`, `devenv.yaml` and `devenv.lock` are present (`ls`). Note: no per-robot acceptance or safety profile exists; `docs/acceptance/` is empty and `grep -rn "SafetyProfile" crates` finds no type.
- [ ] 0.7 Any MUST violation is recorded as a defect, not a known limitation — Evidence: `grep -rni defect docs/ledger docs/decisions` finds no defect record, while known MUST gaps are recorded as limitations (for example ADR 0010 records the zero-torque collapse "as a known limitation of the sphere-only model", `docs/decisions/0010-ground-contact-penalty-model.md:5-6`; the ledger's "Demotion log" is empty). Note: no defect register exists; the process rule has no mechanism in the repository.
- [x] 0.8 Open decisions are tracked in Appendix C — Evidence: `SPECS.md:1281-1308` holds the register C-01 to C-28; the revision history cites C-14, C-15, C-20 to C-22 and C-25 to C-28; per-row status is audited in `appendix-c-open-questions-and-unresolved-decisions.md`.

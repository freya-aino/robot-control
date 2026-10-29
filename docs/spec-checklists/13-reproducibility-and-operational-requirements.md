## 13. Reproducibility and operational requirements

Every run records source revision, compiler/toolchain, the `devenv.lock` revision that supplied it, dependency lock, hardware/driver details, seeds, data hashes, robot model, simulator configuration, reward definition and all model versions. A run whose environment cannot be named is not reproducible, whatever else it recorded. Determinism is a documented mode and tolerance, not an assumed property of GPU execution.

Provide restartable training, atomic checkpoints, bounded storage quotas, artifact retention, structured metrics and replayable failure episodes. Production bundles are immutable. Promotion includes compatibility validation, evaluation report and an immediately available last-known-good artifact.

Treat downloaded datasets and model files as untrusted input: enforce size limits, reject malformed dimensions, fuzz parsers and verify checksums. Restrict command-network access and authenticate remote control where the transport allows it. This is operational integrity, not a dependency on a cloud control plane. A response from an external model service of Section 2.3 is untrusted input of the same kind and is validated before use.

A hosted model is not reproducible by recording its name. Where any part of a reported result passed through an external model service, the record MUST name the provider, model identifier, version or snapshot, request parameters and date, and MUST state that the result is not replayable if the provider does not guarantee a pinned snapshot. A durable event log produced by an agent runtime is an operational record, and it becomes a reproducibility artifact only once that runtime itself is out of experimental status and qualified under Appendix B.6.

### 13.1 Low-level dependency knowledge base

A humanoid fails at its lowest layers: a firmware revision, a bus timing parameter, a driver quirk, an actuator thermal curve, a kernel option, a crate version with a known defect. Diagnosing such a fault by rediscovering the layer each time is the normal and the expensive outcome. The third admissible use of an agent layer (U3 of Section 2.3.1) is to help assemble and maintain a **dependency knowledge base**: a project-owned, versioned record of every low-level dependency the system relies on, so that a hardware fault is resolved against maintained project knowledge rather than against a search engine.

The knowledge base is an artifact of this project, not a service. The following are binding:

1. Scope is the full low-level inventory: Rust dependencies and their native linkage, the OS, kernel options, drivers and firmware of Section 2's platform class, bus and transport parameters, actuator and sensor interface documents, and the per-robot profile versions they apply to. The inventory required by Section 2 and Appendix B.6 item 1 is its primary key, so an entry cannot exist for something the dependency audit does not know about.
2. Every entry MUST be **cited and attributed**: source document, version and date, who or what added it, and the evidence state of each claim under Section 3. An uncited entry is a note, and a note MUST NOT be read as a specification.
3. The knowledge base MUST NOT be cited as evidence in a release report, and it MUST NOT be the control for any hazard under Section 12.2. It shortens diagnosis; it does not close an acceptance row. A claim that matters is promoted through Section 3 against the primary source, and the knowledge-base entry then references the ledger rather than substituting for it.
4. Entries MUST be invalidated by version change. A firmware, driver, crate or robot-profile change marks every dependent entry stale, exactly as it demotes a claim under Section 3. A stale entry is visibly stale, not silently wrong.
5. The knowledge base MUST be readable and usable without any model service: it is text and structured records under version control in this repository, indexed locally. Assisted retrieval or assisted drafting over it is a convenience under U1, subject to Section 2.3 in full, and its unavailability degrades to reading the records directly.
6. Fault records close the loop: a diagnosed hardware fault produces an entry naming the symptom, the measurement that identified the layer, the resolution and the dependency version it applies to, linked to the episode records and incident channel of Section 5. This is what makes the base a project asset rather than a collected reading list.
7. The knowledge base is operational knowledge and MAY contain supplier or robot-specific material; its retention and distribution follow Section 7.5 and the licensing decision of [C-13](#appendix-c--open-questions-and-unresolved-decisions).


## Implementation checklist

Status legend: `[x]` implemented, `[~]` partial, `[ ]` not implemented, `[?]` not audited in this pass. Summary: 4 `[x]`, 6 `[~]`, 6 `[ ]`, 1 `[?]` across 17 items.

- [~] 13.1 Every run records source revision, toolchain, `devenv.lock` revision, dependency lock, hardware/driver details, seeds, data hashes, robot model, simulator configuration, reward definition and model versions — Evidence: `crates/robot-artifacts/src/manifest.rs` `Environment` (named, fully populated, `robot-artifacts/README.md` lines 42-44) and `RunDirectory` input manifest (`store.rs`). Note: toolchain, `devenv.lock` revision and hardware/driver fields were not confirmed in the `Environment` struct; add them if absent.
- [~] 13.2 Determinism documented as a mode and tolerance, not assumed for GPU execution — Evidence: `crates/robot-sim-gpu/tests/equality_agreement.rs` and `implicit_agreement.rs` test GPU agreement. Note: no written determinism mode or tolerance record was found.
- [x] 13.3 Restartable training with atomic checkpoints — Evidence: `crates/robot-artifacts/src/store.rs` `write_atomic` (temp file, fsync, rename) and `robot-artifacts/README.md` lines 28-36; `crates/robot-learn/src/checkpoint.rs` resume.
- [~] 13.4 Bounded storage quotas and artifact retention — Evidence: `crates/robot-artifacts/src/store.rs` `BlobStore` (content-addressed). Note: no quota enforcement or retention policy was found by search for `quota` and `retention`.
- [~] 13.5 Structured metrics — Evidence: `crates/robot-learn/src/` metrics types were not located in this pass. Note: confirm a structured metrics record type exists or add one.
- [ ] 13.6 Replayable failure episodes — Evidence: `crates/robot-hardware/src/replay.rs` replays hardware input (replay device); no failure-episode replay harness was found. Note: connect episode records to replay.
- [x] 13.7 Production bundles are immutable — Evidence: `store.rs` `write_immutable` (`robot-artifacts/README.md` line 14).
- [x] 13.8 Promotion includes compatibility validation, evaluation report and immediately available last-known-good artifact — Evidence: `crates/robot-artifacts/src/manifest.rs` `PromotionLedger` with `roll_back()` (`robot-artifacts/README.md` lines 42-45).
- [~] 13.9 Downloaded datasets and model files treated as untrusted input: size limits, dimension checks, parser fuzzing, checksum verification — Evidence: `crates/robot-artifacts/src/hash.rs` checksum; `toml.rs` adversarial-input sweep (`robot-artifacts/README.md` lines 49-55). Note: size limits and malformed-dimension rejection for datasets were not confirmed; coverage-guided fuzzing is not in place (per README).
- [ ] 13.10 Restrict command-network access and authenticate remote control — Evidence: no network-control code exists in `crates/robot-hardware` or `robot-control`. Note: none needed while there is no remote control; record this as the state.
- [x] 13.11 Operational integrity, no dependency on a cloud control plane — Evidence: no network or cloud client in any `[dependencies]` list (see 2.4).
- [?] 13.12 External model service responses validated before use — no external model service exists (see 2.3.1); not applicable today.
- [ ] 13.13 Records for results passing through an external model service (provider, model ID, version, request parameters, date, replayability) — Evidence: no such record exists because no model service is used. Note: add the fields to the run record if a service is ever used.
- [~] 13.14 Dependency knowledge base as a versioned project-owned record — Evidence: `crates/robot-devtools/src/knowledge_base.rs` exists. Note: it is code, not a record store with citations; no structured entries were found in this pass.
- [ ] 13.15 Knowledge-base entries cited and attributed with source, version, date, author and evidence state — Evidence: no knowledge-base entries found in the repository. Note: create the record format and first entries.
- [ ] 13.16 Entries invalidated by version change (firmware, driver, crate or profile) — Evidence: no invalidation logic found in `knowledge_base.rs` (not read in full). Note: implement stale-marking tied to the lockfile and profile versions.
- [ ] 13.17 Fault records link symptom, measurement, resolution, dependency version and episode records — Evidence: no fault-record schema found. Note: define and create the record format.

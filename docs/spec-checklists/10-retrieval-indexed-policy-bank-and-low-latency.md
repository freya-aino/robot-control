## 10. Retrieval-indexed policy bank and low latency

### 10.1 Two kinds of similarity

Semantic similarity organizes intent: “jab”, “retreat”, “guard”, “celebrate”. Physical compatibility determines what may execute now: stance, contact mode, joint velocity, available support, reach, actuator health and opponent-relative geometry.

The retriever therefore produces candidate skills; a state-conditioned feasibility filter and fast router select executable candidates. Text embeddings may be computed offline with a Rust model. Start with structured skill tags and numerical features, and add learned embeddings only when they improve a measured outcome.

For small libraries, benchmark exact vector search in Rust before adopting approximate nearest-neighbor indexing. A remote vector database is not part of the control path.

A skill MAY additionally carry a human-readable descriptor — capabilities, preconditions, failure modes and counter-indications as Markdown — indexed at build or load time and matched lexically. Such a descriptor is documentation and operator-facing catalog, and its indexing involves no model call; it is data under Section 2's opening paragraph, not code. It MAY contribute a routing signal that composes with the numerical retrieval of Section 10.2, and it MUST NOT replace it or relax any feasibility check: prose asserting that a skill is applicable is not a statement about joint velocity, support or reach. The descriptor's authoritative fields remain the typed ones of Section 10.2, and a disagreement between the two is a defect in the descriptor.

### 10.2 Skill descriptor and execution state

Each skill has semantic tags, embedding/version, embodiment compatibility, initiation set, termination condition, expected duration, contact assumptions, supported disturbances, inference memory requirement and outgoing transition contracts.

The runtime tracks current skill, phase, active contacts, last accepted proposal, proposal timestamp/expiry, transition status and emergency override. Candidate selection must include uncertainty and out-of-distribution detection. Log rejection reasons and routing frequency.

### 10.3 Memory and scheduling

Safety, recovery and active working-set experts stay resident. Stage optional experts asynchronously into a bounded cache; validate schema, checksum, normalization and device memory before activation. Cold-load failure or late retrieval retains a compatible current controller or invokes the robot-specific fallback.

Switch at a defined control boundary using generation-tagged immutable handles. Recurrent state must be reset or transformed by an explicit contract. No expert may be activated while its input/output schema or state is incompatible. Runtime hot swapping is tested separately from offline model loading.

### 10.4 Transitions and fallback

Use contact- and phase-aware initiation guards, hysteresis, minimum dwell times where appropriate and interruptible transitions. Recovery can preempt dwell times. Initial implementation should favor hard selection at validated transition states or a shared conditioned tracker.

Do not assume weighted averaging of motor commands is safe. Blending requires explicit transition training and testing, or a validated constrained controller. Guard posture is not a universal fallback when already falling; fallback behavior depends on state and hardware, and may involve controlled stopping, support seeking or a configured protective response.

### 10.5 Required comparison

Compare dense, conditioned, retrieval-only, learned-router and hybrid designs at equal hardware and training budgets. Report repertoire coverage, task success, transition failures, disturbance recovery, active parameters, resident memory, energy where measurable, latency tails and decision age. Measure retrieval amortization and cold-cache behavior explicitly.

Parallel expert training is useful but not necessarily independent: shared transitions, normalization, training distributions and safety assumptions couple the experts. Running several experts simultaneously may increase single-step latency on small devices.

### 10.6 Retrieval tiers and deadline attribution

"Retrieval" names three different operations in this project, at three different costs. They MUST be distinguished in design, in configuration and in every report, because the published latency numbers for one say nothing about another.

| Tier | Operation | Where it runs | Binding constraint |
| --- | --- | --- | --- |
| R0 — corpus retrieval | Selecting prior trajectories or sub-trajectories to train on, per Section 9.7 | Offline, on a developer or CI host | No deadline; subject to license, provenance and held-out exclusion rules. Never present in a deployed bundle |
| R1 — between-episode retrieval | Choosing a working set of experts, an opponent model or an adaptation context before a bout or between rounds | Offline-equivalent, outside the control process | Completes before the mode entry preconditions of Section 4 are satisfied; a failure degrades to the previous working set, never to a late load |
| R2 — in-loop selection | The skill router and feasibility filter of Sections 10.1 to 10.4 | Inside the control process, at the tactical tier of Section 5.1 | Bounded local computation over resident data only, with stale-output rejection and measured tails; this is the only tier with a deadline |

The following are binding:

1. An R2 lookup MUST operate on a resident, preloaded index. It MUST NOT read a file, query a database, compute an embedding, allocate, or block on a staging operation. Index updates occur at R1 and are published to R2 as generation-tagged immutable handles under Section 10.3.
2. R2 cost MUST be measured as part of the policy-path compute duration and reported in the tails of Section 5, separately from decision age. An asynchronous retrieval that merely moves work off the control thread converts latency into decision age; Section 3 already records this and the report MUST show both numbers.
3. A method validated at R0 MUST NOT be promoted to R1 or R2 on the strength of that validation. Each tier requires its own measurement, and moving a mechanism downward is the Section 5.1 item 5 prohibition.
4. Published inference-latency figures for retrieval-augmented systems are evidence about their setting only. A reported millisecond figure obtained on other hardware, at another rate, with another index size, licenses nothing here; Appendix B.2.5 requires the crossover and worst-observed timing on target hardware.
5. Caching a previous retrieval result is permitted at R1 and R2 only with an explicit expiry and a recorded decision age. A cache without an expiry is a stale decision with better latency, which is not an improvement.
6. Every tier MUST have a defined empty-result behavior. R0 falls back to target data alone, R1 retains the current working set, and R2 retains the current controller or invokes the fallback of Section 10.4. No tier may stall a decision while waiting for a retrieval.

## Implementation checklist

Status legend: `[x]` implemented, `[~]` partial, `[ ]` not implemented, `[?]` not audited in this pass. Summary: 4 `[x]`, 10 `[~]`, 9 `[ ]`, 1 `[?]` across 24 items.

- [x] 10.1.1 Structured skill tags and numerical features are the first retrieval signal — Evidence: `crates/robot-skills/src/descriptor.rs` lines 32-35 (`tags`, `embedding`) and `crates/robot-skills/README.md` lines 28-30.
- [ ] 10.1.2 Physical feasibility filter (stance, contact, joint velocity, support, reach, actuator health, opponent geometry) before routing — Evidence: `robot-skills/README.md` lines 31-35 states no feasibility filter exists. Note: add the state-conditioned filter in the crate that owns robot state.
- [x] 10.1.3 Exact vector search is the default before any approximate index — Evidence: `crates/robot-skills/src/index.rs` `ExactIndex` (brute-force, `robot-skills/README.md` lines 15-19).
- [ ] 10.1.4 Exact-search benchmark run before adopting approximate indexing — Evidence: `robot-skills/README.md` line 18 states no benchmark has been run. Note: run and record the benchmark on target hardware.
- [x] 10.1.5 No remote vector database on the control path — Evidence: no remote-database client in any `[dependencies]` list (see 2.4).
- [ ] 10.1.6 Optional Markdown skill descriptor indexed lexically, non-authoritative and not replacing typed fields — Evidence: `robot-skills/README.md` lines 30-31 states no Markdown catalog exists.
- [~] 10.2.1 Skill descriptor fields: tags, embedding/version, embodiment, initiation set, termination, duration, contacts, disturbances, inference memory, outgoing transitions — Evidence: `crates/robot-skills/src/descriptor.rs` `SkillDescriptor` (lines 19-36 show tags, embedding and embedding version). Note: the remaining fields were not read in this pass; confirm each is present.
- [~] 10.2.2 Runtime tracks current skill, phase, active contacts, last accepted proposal, expiry, transition status and emergency override — Evidence: `crates/robot-skills/src/tiers.rs` `WorkingSet` and `Router` track generation and residency. Note: phase, active contacts, expiry and emergency override were not confirmed.
- [ ] 10.2.3 Candidate selection includes uncertainty and out-of-distribution detection — Evidence: no uncertainty or OOD code found in `robot-skills`. Note: implement both.
- [~] 10.2.4 Rejection reasons and routing frequency are logged — Evidence: `Router` rejects stale generations (`robot-skills/README.md` lines 39-43). Note: the rejection reason is not logged as a structured record and routing frequency is not counted.
- [~] 10.3.1 Safety, recovery and working-set experts stay resident; optional experts staged asynchronously into a bounded cache — Evidence: `crates/robot-skills/src/tiers.rs` `WorkingSet` (R1, immutable). Note: the bounded cache and the asynchronous staging path were not found.
- [~] 10.3.2 Schema, checksum, normalization and device memory validated before activation — Evidence: `crates/robot-artifacts/src/hash.rs` (checksum). Note: schema and normalization checks at skill activation are not wired in `robot-skills`.
- [ ] 10.3.3 Cold-load failure or late retrieval retains the current controller or invokes the robot-specific fallback — Evidence: `robot-skills/README.md` line 42 covers only the empty-result degrade path. Note: add cold-load failure handling.
- [x] 10.3.4 Switch at a defined boundary with generation-tagged immutable handles — Evidence: `crates/robot-skills/src/tiers.rs` (`Router` rejects stale generations; test "equal generation is not newer", `robot-skills/README.md` lines 40-41).
- [ ] 10.3.5 Recurrent state reset or transformed by an explicit contract at a switch — Evidence: `grep` for `recurrent` in `robot-skills` found no contract. Note: define the reset contract.
- [~] 10.4.1 Contact- and phase-aware initiation guards, hysteresis, dwell times and interruptible transitions — Evidence: `Router::switch_to` residency check (`robot-skills/README.md` lines 31-32). Note: hysteresis and dwell times are explicitly not implemented per the README.
- [ ] 10.4.2 Blending of motor commands only with explicit transition training or a validated constrained controller — Evidence: no blending code found. Note: keep blending disabled until validated; record the rule in the controller.
- [~] 10.4.3 Fallback behaviour depends on state and hardware (controlled stop, support seeking or protective response) — Evidence: `Router` falls back to the current controller on empty result (`robot-skills/README.md` line 42). Note: state-dependent fallback variants are not implemented.
- [ ] 10.5.1 Comparison of dense, conditioned, retrieval-only, learned-router and hybrid designs at equal budget — Evidence: none found. Note: implement the comparison after the policies exist.
- [ ] 10.6.1 R0 corpus retrieval is offline, has no deadline and is never in a deployed bundle — Evidence: no R0 corpus-retrieval module exists (see 9.7.1). Note: implement with 9.7.
- [~] 10.6.2 R1 between-episode working-set selection that retains the previous set on failure — Evidence: `crates/robot-skills/src/tiers.rs` `WorkingSet` (generation-tagged, immutable). Note: the R1 failure-degrade path and mode-entry timing link were not confirmed.
- [~] 10.6.3 R2 in-loop selection on a resident preloaded index with no file, database, embedding or allocation — Evidence: `ExactIndex::nearest_into` writes into a caller-supplied buffer (`robot-skills/README.md` lines 23-24). Note: the tiers are not yet called from `ControlLoop::tick` (see `robot-control/README.md` lines 37-41).
- [~] 10.6.4 R2 cost reported in policy-path compute tails separately from decision age — Evidence: `crates/robot-control/src/timing.rs` `TickTiming` has separate `compute_duration` and `decision_age` fields (lines 11-22). Note: no R2-specific attribution exists because R2 is not wired in.
- [?] 10.6.5 Caching with explicit expiry and recorded decision age at R1 and R2 — not audited; no cache code was found in `robot-skills` by reading the README.


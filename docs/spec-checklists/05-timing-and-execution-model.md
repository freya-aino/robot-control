## 5. Timing and execution model

The following are initial design targets, not measured guarantees. Each robot profile must establish its actual rates and deadlines.

| Loop | Initial target | Execution requirements |
| --- | --- | --- |
| Embedded actuator servo | 500–1,000 Hz where supported | Usually firmware owned; document host/firmware split |
| Host state estimation and command supervision | 500–1,000 Hz if hardware supports it | Bounded work, no blocking external dependencies |
| Learned whole-body policy | 50–200 Hz; start at 100 Hz | Resident model, preallocated tensors and buffers |
| Fast recovery / contact reflex | Every host control tick | Can interrupt the normal skill schedule |
| Tactical router and opponent forecast | 10–50 Hz, task dependent | Bounded local computation; stale-output rejection |
| Semantic search and repertoire planning | 1–10 Hz | Asynchronous; produces proposals, not motor commands |
| Optional agent advisory, if ever enabled | 1 Hz or slower, or between rounds | Out of process, feature-gated per Section 2.3; produces expiring advisory records, never commands; the loop above it runs unchanged when it is absent |
| Training and adaptation | Offline or isolated background process | Never contend with the deployed control process |

At an initial 100 Hz policy rate, the system has a 10 ms period. Establish a hardware-specific budget for sensor acquisition, estimation, inference, supervision, and command transport. Initial engineering objective: p99.9 policy-path completion below 8 ms under representative load, with remaining margin, and no deadline overruns during a specified qualification run. This is a target to validate, not a hard-real-time proof.

Measure sensor-to-command age separately from compute duration. Report p50/p95/p99/p99.9/max, deadline misses, input age, clock uncertainty, and fallback events. A hard-real-time claim additionally requires bounded worst-case execution and scheduling analysis for the selected OS, transport, and devices.

These quantities are distinct and a report MUST NOT collapse them:

| Quantity | Measured from | Measured to | Typical failure it exposes |
| --- | --- | --- | --- |
| Compute duration | Policy path entry | Command ready | Inference regression, allocation, contention |
| Sensor-to-command age | Sensor sampling instant | Command accepted by the actuator interface | Transport delay, buffering, clock skew, missed sampling |
| Decision age | Production of the last accepted high-level proposal | The tick on which it is still being applied | Stale intent from asynchronous retrieval or planning |
| Deadline margin | Command ready | Period boundary | Tail growth before it becomes an overrun |

A system can hold compute duration well inside budget while sensor-to-command age drifts out of it. Sampling instants come from the device where the device provides them; where it does not, the record MUST say that the timestamp is an arrival time and carry the resulting uncertainty, because an arrival time silently relabeled as a sampling time corrupts every delay model trained on it.

The control path MUST NOT perform file reads, network database queries, model downloads, heap allocation after activation, locks with unbounded contention, GPU compilation, logging flushes, or model loading. Tokio or similar asynchronous runtimes may serve orchestration, but not the deadline-critical loop.

No external model service of Section 2.3 and no agent runtime built on one may execute inside a deadline-critical section, in any mode, under any approved exception. Where an agent layer exists at all it runs outside the control process and its output enters only as an expiring advisory record through the same validated queue interface, freshness limit and rejection accounting as any other asynchronous proposal, measured as decision age in the table above. An advisory that has expired is dropped, not held: a stale directive is indistinguishable from a wrong one. Removing the agent layer entirely MUST leave the control loop functionally unchanged, which is the point of row A-24.

Rayon work stealing and general-purpose telemetry subscribers are not presumed suitable for this loop. Use dedicated bounded resources for control and separate pools for rollout/data processing. Capture preallocated timestamp/event records in the loop and construct/export telemetry off-thread; benchmark enabled instrumentation against a disabled baseline.

Use monotonic timestamps, fixed-capacity queues, explicit freshness limits, CPU affinity and scheduling configuration where available, and a watchdog independent of the policy process. Log overflow must never block actuation; emit dropped-record counts. Safety-relevant incidents require a reserved event channel and bounded pre/post-event recording buffer.

### 5.1 Hierarchical decomposition and the latency invariant

The learned stack is hierarchical by construction, and the hierarchy is ordered by deadline rather than by conceptual importance. The governing invariant is that **the lower a component sits, the tighter its latency obligation and the smaller its permitted variability**. A design that places a large or variable-cost network below a small one violates this section regardless of its accuracy.

The following bind every hierarchical policy, including the staged pipeline of Section 9.4 and any router of Section 10:

1. Every learned component MUST be assigned to exactly one tier of the table above, and that assignment MUST be recorded in the policy bundle of Section 7.4 alongside the component's measured compute duration. A component with no declared tier MUST NOT be activated.
2. Tier rates MUST be non-increasing with height: no component may run faster than a component below it, and the lowest learned tier MUST run at the policy rate or faster. Two tiers MAY share a rate, which is the degenerate case and MUST be recorded as such rather than described as a hierarchy.
3. Each tier carries its own deadline, its own worst-case budget and its own fallback. A tier MUST remain functional when the tier above it is late, absent or has produced only an expired output; the response to a missing upper-tier decision is the declared hold-or-default behavior of that tier, never an extended deadline below.
4. Outputs crossing a tier boundary MUST be bounded, validated and timestamped, and consumers MUST reject them on the freshness limit of their own tier. Decision age is attributed to the tier that produced the decision, not to the tier that applied it.
5. Lower tiers MUST NOT acquire dependencies that upper tiers have: no retrieval query, corpus access, dynamic dispatch, allocation or variable-iteration optimization may be introduced below the tier at which it was qualified. Pushing an expensive mechanism downward is a change to Section 5 and requires the change-control record.
6. The per-tier budget MUST be measured, not inferred from parameter count. Section 3's correction about sparse activation applies to every tier: a smaller network is a hypothesis about latency until the tail is measured under representative load.
7. Removing the highest tier MUST leave a controller that is still safe, even if it is no longer tactical. This is the hierarchical form of the removal tests in rows A-24 and A-30, and it is what makes strategic layers optional rather than load-bearing.

The frequency separation is also an evidence boundary. A result obtained with every layer running at one rate does not establish that the same decomposition holds when the strategy layer is slowed to its intended tier, because decision age that was zero in the published setting becomes nonzero here. Any tiered deployment of a method validated at a single rate MUST re-measure task performance at the tier rates actually used.



## Implementation checklist

Status legend: `[x]` implemented, `[~]` partial, `[ ]` not implemented, `[?]` not audited in this pass. Summary: 5 `[x]`, 2 `[~]`, 0 `[ ]`, 6 `[?]` across 13 items.

- [x] 5.1 Compute duration, sensor-to-command age, decision age and deadline margin are four distinct, named fields — Evidence: `crates/robot-control/src/timing.rs` lines 11-22 (`TickTiming`).
- [x] 5.2 Timing reports cannot collapse the four quantities into one number — Evidence: `timing.rs` lines 1-6 and the `TickTiming` struct; no combined field exists.
- [x] 5.3 Compute duration is reported as p50/p95/p99/p99.9/max with deadline misses — Evidence: `timing.rs` `compute_duration_percentile` (lines 79-93), `max_compute_duration` (lines 96-103), `missed_deadline_count` (lines 106-113). Note: p99.9 is supported through the percentile argument; no report function fixes these five percentiles.
- [~] 5.4 Sensor-to-command age, input age, clock uncertainty and fallback events are reported — Evidence: `timing.rs` carries `sensor_to_command_age: Option<Duration>` (line 17). Note: no clock-uncertainty field, input-age field, fallback counter or percentile API for sensor-to-command age exists.
- [~] 5.5 Decision age is measured per tier boundary — Evidence: `timing.rs` `decision_age` (lines 18-20). Note: production-to-application timestamps are not computed in the crate.
- [x] 5.6 Timing history uses a fixed-capacity ring with no unbounded growth — Evidence: `timing.rs` `TimingHistory<const N: usize>` (lines 35-63), tests at lines 135-147.
- [x] 5.7 Deadline miss is defined as negative margin — Evidence: `timing.rs` `missed_deadline` (lines 28-30), test at lines 129-133.
- [?] 5.8 Control path performs no file reads, network queries, model downloads, post-activation allocation, unbounded locks, GPU compilation, log flushes or model loading — not audited against `control_loop.rs` in this pass.
- [?] 5.9 Monotonic timestamps, fixed-capacity queues and watchdog independent of the policy process — `robot-hardware` loopback/replay and `robot-supervisor` were not read for the watchdog and queue rules.
- [?] 5.10 Overflow of log records never blocks actuation and dropped-record counts are emitted — not audited.
- [?] 5.11 A reserved event channel and bounded pre/post-event buffer exist for safety incidents — not audited.
- [?] 5.12 Per-tier budgets, decision-age attribution and tier-boundary validation (Section 5.1 items 1 to 7) — not audited beyond the `decision_age` field.
- [?] 5.13 Initial engineering objective (p99.9 under 8 ms on representative load) is measured by a qualification run — no measurement artifact was read in this pass.

# robot-persona

Bounded expressive intent over validated skills, per SPECS.md §12.5.
Depends on `robot-skills` and `robot-types`.

## What's here

| Module | Contents |
| --- | --- |
| `style` | `StyleConfig`: a sampled, versioned configuration of expressive-skill and tactical preferences plus bounded parameters. `validate` refuses unvalidated skills, unknown tactics, bad weights, duplicate dimensions and version 0. Also `select_expressive_skill` and `sample_preference` |
| `params` | The §12.5 dimensions (posture, cadence, idle amplitude, celebration duration, speech delay, aggression, patience) as positions in `[0, 1]` inside a range the caller already validated. `resolve` cannot land outside that range for any input, NaN included |
| `arbiter` | `Arbiter::decide`: `Yield` whenever recovery is requested or the supervisor vetoes, even mid-celebration. `Refuse` for unvalidated, out-of-style, or unreachable skills (the current skill's `outgoing_transitions` is the transition contract). `Express` otherwise |
| `memory` | `SessionMemory`: opponent observations committed between rounds and read at strategy rate as a bounded `TacticalBias`. `None` (opponent-agnostic) when unknown, sparse or stale |
| `evaluation` | `recognizability`: leave-one-out nearest-centroid accuracy against chance. `degradation`: task and stability metrics that got worse against the neutral configuration |

## Decisions worth knowing

**No style can widen an envelope.** §12.5: "Expression MUST NOT widen any
envelope." A style never states a value in physical units. It states a
position inside a range the task validation supplies. Widening a range
is a validation change made elsewhere, never a style edit.

**Recovery wins mid-skill.** Expression MUST NOT "extend any dwell time
against a recovery request". The arbiter yields as soon as recovery is
requested, including while an expressive skill is running.

**A celebration needs a transition.** "A celebration is a skill with a
transition contract, not an exemption." An expressive skill is reachable
only along the current skill's declared outgoing transitions.

**Opponent memory follows C-17's interim position.** C-17 (retention,
consent, deletion, backend) is open. Its interim position is "No persistent
opponent record is retained beyond a session; no hosted backend is
permitted on any read path." `SessionMemory` has no serialization, file or
network path, and is dropped with the session. An unavailable, empty or
stale record degrades to opponent-agnostic behavior, as §12.5 requires.
Its output is a tactical bias. No type in it reaches the supervisor or the
motor loop.

## What is not here

- **Persistent opponent memory across sessions.** It waits on the C-17
  decision record.
- **Speech generation.** A speech line is an expressive skill with a
  timing position, not generated text. §12.6 governs any agent layer
  separately.

## Testing

`cargo test -p robot-persona`: 17 tests. They cover:
- selection, recovery refusal, the validated allowlist and preference
  sampling (6 original tests);
- style validation and neutral positions;
- range-bounded resolution, NaN included;
- the arbiter yielding mid-skill and enforcing transitions;
- memory degrading when unknown, sparse or stale, clamping corrupt
  observations, and forgetting;
- recognizability of distinct styles and not of identical ones;
- degradation listing only metrics that got worse.

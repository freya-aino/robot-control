# robot-persona

Bounded expressive intent over validated skills, per SPECS.md §12.5.
Depends on `robot-types` and `robot-skills`.

## What's here

| Module | Contents |
| --- | --- |
| `style` | `StyleConfig` (sampled, versioned skill preferences), `select_expressive_skill`, `sample_preference` |

## Why "cannot widen an envelope" is a type, not a check

SPECS.md §12.5: "Expression MUST NOT widen any envelope, extend any dwell
time against a recovery request, or introduce a motion that has not
passed the same validation as a task skill." `StyleConfig` has no torque,
velocity, position or dwell-time field at all — there is nothing in the
type for a style to widen. `select_expressive_skill`'s signature reflects
the same discipline: its only possible successful return value is a
`SkillId` that was already present in *both* the caller-supplied validated
allowlist and the style's own preference list. It cannot synthesize a new
skill or relax either check; recovery takes priority unconditionally,
checked first and before either list lookup.

## What's not implemented

Persistent identity and opponent memory — also named in §12.5 — are not
in this crate. C-17 (retention, consent and deletion terms for any such
store) is explicitly open in SPECS.md's own open-questions appendix, and
this revision does not get ahead of that decision.

## Testing

`cargo test -p robot-persona` — 6 tests: a valid selection, recovery
refusing expression unconditionally (checked even when the skill is
otherwise valid), an unvalidated skill refused despite being in the
style's own list, a validated skill outside the style refused, weighted
sampling matching its declared ratio under a deterministic stratified
sweep, and a style with no positive weights sampling nothing rather than
an arbitrary default.

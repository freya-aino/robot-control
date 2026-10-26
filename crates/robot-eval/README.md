# robot-eval

Acceptance profiles, statistics and regression gates, per SPECS.md §14.
Depends on `robot-types` and `robot-artifacts` (the latter not yet used —
reserved for publishing a profile as a content-addressed artifact).

## What's here

| Module | Contents |
| --- | --- |
| `acceptance` | `AcceptanceProfile`: every field §14 requires, immutable-once-evaluated by construction (`revise` always returns a new version) |
| `stats` | `summarize`: mean/std-dev/95%-confidence-halfwidth over seeds; `confidence_intervals_overlap` |

## Why there is no `set_threshold` method

SPECS.md §14: "Profiles are versioned and immutable once a run has been
evaluated against them. Changing a threshold after observing a result
produces a new profile version and invalidates the comparison, which MUST
be stated rather than quietly re-baselined." `AcceptanceProfile` has no
method that mutates its own `thresholds` field. `revise` is the only path
to a different threshold, and it always increments `version`, resets
`evaluated` to `false`, and leaves the original profile — and whatever run
was scored against it — untouched. `revision_increments_version_and_resets_evaluated_status`
checks all three of those in one test.

## What `confidence_intervals_overlap` does not claim

It reports whether two 95%-confidence intervals (normal approximation)
overlap. Non-overlapping is reasonably strong evidence of a real
difference; *overlapping is not evidence of equivalence* — the function's
own doc comment says so, because a proper equivalence test is a different,
unimplemented statistic, and conflating "we didn't detect a difference"
with "they're the same" is a common and specifically-named failure mode in
exactly this kind of report.

## Testing

`cargo test -p robot-eval` — 6 tests: mean/standard-deviation computation
checked against a textbook worked example, rejection of fewer-than-two
samples, confidence-interval overlap in both the separated and identical
cases, and the acceptance-profile revision/immutability behavior above.

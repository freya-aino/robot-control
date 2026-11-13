# robot-eval

Acceptance and evaluation, per SPECS.md §14: acceptance profiles as
artifacts, seed statistics with honest small-sample intervals,
pass/fail by interval, the regression gate, and the acceptance matrix.
Depends on `robot-types` and `robot-artifacts`.

## What's here

| Module | Contents |
| --- | --- |
| `acceptance` | `AcceptanceProfile`: every field §14 requires (robot profile and firmware version, control period and budget, envelope, thresholds with direction and justification, seed list and run count, duration, date, author). Immutable once evaluated (`revise` is the only path to a new threshold). `to_toml`/`from_toml` (`acceptance-profile/1`) and `content_id` |
| `stats` | `summarize` (mean, standard deviation, normal and Student-t 95 % half-widths), `welch` (unequal-variance t test), `tost` (two one-sided tests for equivalence within a margin), `confidence_intervals_overlap` |
| `distribution` | Student's t CDF and quantile via log-gamma and the regularized incomplete beta function, std-only |
| `evaluate` | `evaluate`: a metric passes only if its whole t interval clears the threshold, fails if the whole interval misses, and is otherwise inconclusive. Missing and unlisted seeds are recorded, and the profile is marked evaluated. `regression_gate`: Welch per metric between two evaluations of the same profile, refusing a different profile with both IDs named |
| `matrix` | The fifty §14 rows (a test keeps them in step with SPECS.md), and `status`, which closes a row only where the evidence ledger's "Acceptance rows" table names an artifact |

## Decisions worth knowing

**No `set_threshold` method.** §14: "Changing a threshold after observing
a result produces a new profile version and invalidates the comparison,
which MUST be stated rather than quietly re-baselined." `revise` always
returns a new version. A profile's identity is the content ID of its TOML.
`regression_gate` refuses to compare evaluations of different IDs and
names both, which is that statement.

**A mean that clears the line is not a pass.** Five noisy seeds whose
mean is under the threshold but whose interval is not are *inconclusive*.
A test covers exactly that case.

**t, not 1.96.** With five seeds the normal half-width understates the
95 % interval by a factor of 2.776/1.96. Both are reported. Evaluation
uses t.

**Equivalence has its own test.** Overlapping intervals are not evidence
of equivalence. `tost` is: it rejects "the difference is at least the
margin" in both directions.

**Rows close only through the ledger.** `docs/ledger/evidence-ledger.md`
now has an "Acceptance rows" table. It is empty, so every row reads
`Open`. Nothing in this crate can mark a row closed except a ledger line
naming an artifact.

## Testing

`cargo test -p robot-eval`: 18 tests. They cover:
- the textbook mean and standard deviation, sample-count rejection and
  interval overlap (original tests);
- profile revision and lookup, the TOML round trip, and the content ID
  changing with any threshold;
- log-gamma against factorials, t quantiles against published tables,
  and the normal limit;
- the t/normal width ratio, Welch separating means, and TOST confirming
  equivalence only within a sensible margin;
- pass, fail and inconclusive by interval, missing and unlisted seeds, and
  the gate flagging a regression and refusing a revised profile;
- matrix rows matching SPECS.md, closure only by a ledger artifact, and
  the real ledger parsing.

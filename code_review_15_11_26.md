# Code review — 2026-11-15

> **Fix status (same day).** All 7 high and all 15 medium findings are fixed
> and tested in the commits that follow `b500baf` on this branch (`05bafc4`
> activation gate H1/H2/M11 · `a5f6771` selector/supervisor H3/H4/M7/M9/M10 ·
> `84c3312` hostile-file readers H5–H7/M12–M14 · `c047620` robot-usd M1–M3 ·
> `42138f4` bundle seal M4–M6 · `672f2cb` eval/resample M8/M15), plus a
> maintenance commit covering most lows. Lows deliberately left open: EPA
> convergence surfacing on `Contact`/`StepReport`, the `implicitfast`
> saturation note, C3D label dedup and `POINT:SCALE == 0`, `filter_run`'s
> small-run reporting, `chain_ratio` link errors, the TOML `[[a]]` header
> bound, `Digest::from_hex` case strictness, duplicate bundle component
> names, the ledger differ's cross-table keys, the PPO normalizer note, the
> ESKF reset-Jacobian note, and the USDA arc refusal-by-name wording.

- Revision reviewed: `b500baf` (branch `chore/tidy-docs`), working tree clean.
- Scope: all 24 workspace crates, source and tests (~94k lines of Rust, 1338 passing tests).
- Method: six parallel inspection passes (foundation; model/USD; motion pipeline; sim/tasks;
  learning stack; runtime/safety), every reported finding checked against the source with
  file:line evidence. All seven high-severity findings were then re-verified independently
  against the code before this report was written. Finding ids (H1…, M1…, L…) are local to
  this review and unrelated to the M-xx milestone ids in PLAN.md.

## Verdict

The workspace is in genuinely good shape for its size: the refusal-by-name discipline is
real and mostly followed, `unsafe` is confined to one feature-gated kernel-launch module
(22 of 24 crates forbid it outright), there are zero TODO/FIXME markers, error paths
return typed errors in almost every public entry point, and several crates
(robot-control, robot-persona, robot-viewer, motion-validate, robot-adapt) came through
inspection with no findings at all. The heavy mathematics — the CRB/`L'DL` dynamics, both
contact solvers, GJK/EPA with its degeneracy handling, Welch/TOST statistics, Joseph-form
Kalman updates, GAE/PPO/Adam — was checked deliberately and held up.

The defects that matter cluster at **trust boundaries**, and five of the seven highs share
one shape: a gate or fallback written as "refuse if `x > y`" **fails open** when the input
is absent, truncated, or NaN. The single most important consequence: as of this revision,
the §12.2 "wrong artifact deployed" gate (`check_activatable`, exercised by
`robot eval --bundle`) can be passed by a bundle with empty or NaN envelope vectors, and
— because every USD import hard-codes the G1's profile identity — by a G1-trained bundle
presented to the Go2 or H1 host.

Counts: **7 high, 15 medium, ~24 low**. Nothing found requires an architectural change;
every high has a localized fix.

## High-severity findings (all independently re-verified)

### H1 — `check_activatable` is fail-open on envelope shape and NaN
`crates/robot-types/src/bundle.rs:398-452` (reached from `robot eval --bundle` via
`crates/robot-cli/src/commands/bundle.rs:209`). Found independently by two review passes.

All three envelope-containment loops iterate
`self.envelope.X.iter().zip(&host.X)`; `zip` stops at the shorter side, and the
comparisons (`bundle.0 < host_limit.0`, `bundle > host_limit`) are false for NaN. The
bounded TOML subset deliberately parses `inf`/`nan`, and `robot_artifacts::bundle::from_toml`
checks only that `position_limits_lo/hi` agree with *each other* (`bundle.rs:598`), not
with the host width, and imposes no finiteness requirement.

**Failure:** a mis-built or hostile bundle whose envelope arrays are empty, truncated
(12 entries against a 29-joint host), or `nan` passes the §12.2 gate vacuously;
`robot eval --bundle` prints ACTIVATABLE with exit 0 while the bundle assumes unlimited
authority on the unchecked joints.

**Fix:** in `check_activatable`, require each envelope vector's length to equal the
host's and reject any non-finite limit (named errors). Optionally also refuse non-finite
envelope floats at `read_bundle` time. Fixing the gate itself is the stronger cut — it
also covers bundles built in memory. Add the missing tests: mismatched lengths, NaN.

### H2 — every USD import carries the G1's profile identity
`crates/robot-model/src/usd.rs:720-721`; inherited unchanged by the Go2/H1 fixtures
(`crates/robot-model/src/fixtures.rs:62-65, 87-90`).

`import_fixed_base` builds every profile with
`name: Name::new("unitree-g1-29dof-usd")` and the same `profile_version`, regardless of
which robot was imported. `crates/robot-cli/src/commands/bundle.rs:134` constructs the
host's `robot_profile` string as `"{name}/{profile_version}"`, so g1-usd, go2-usd and
h1-usd present the **identical** profile string to the gate.

**Failure:** a bundle trained for the G1 passes the profile-match ("wrong artifact
deployed") check against the Go2 or H1 host — the exact confusion the check exists to
refuse.

**Fix:** derive the name from the layer (defaultPrim / source file) or pass it as a
parameter from each fixture; add a test asserting the three vendored profiles have three
distinct `robot_profile` strings.

### H3 — a falling robot with a non-resident recovery skill gets `Keep`, not `Protective`
`crates/robot-skills/src/selection.rs` — falling branch of `decide` (~:294) calling
`commit` (~:256).

In the falling/emergency branch, when the configured recovery skill's guards pass, the
selector calls `commit`; `commit` returns `Decision::Keep` when `router.switch_to` fails
(skill not resident in the working set). The module contract says non-resident recovery
must fall through to the protective response.

**Failure:** `fs.falling = true` with a recovery skill configured but not resident (empty
router, or an R1 working set omitting it) → `Decision::Keep` every tick; the ordinary
skill keeps executing during a fall and the caller is never told to run the protective
stop. (If the guards *fail*, the code correctly returns `Protective`; only the
residency-failure path is wrong.)

**Fix:** in the falling branch, on residency failure return `Decision::Protective` (and
count the fallback) instead of surfacing `commit`'s `Keep`.

### H4 — fault-log overflow can downgrade the supervisor's tick response
`crates/robot-supervisor/src/supervisor.rs:394-400` with
`crates/robot-supervisor/src/fault.rs:260-267`.

`most_severe_response` folds over *recorded* faults only; `FaultLog::push` silently drops
faults once `len == N` (incrementing a counter), and the context faults — over-temperature,
unauthorized contact, contact energy, **interlock open** — are recorded last in the tick.

**Failure:** a `Supervisor<16>` on a 29-joint robot where every joint trips its position
limit fills the log with clamp-category faults; an e-stop interlock found open the same
tick is dropped, the response stays `ClampAndContinue`, and torque keeps flowing with the
interlock open. The active-contact list is unbounded, so no fixed `N` is provably safe.

**Fix:** track the maximum severity over every *detected* fault before capacity is
consulted (fold at push time), or escalate to at least `HoldLastSafeCommand` whenever
`overflowed() > 0`.

### H5 — BVH hierarchy parser recurses without a depth bound
`crates/motion-io/src/bvh.rs:460` (`parse_joint_body`; the `flatten` helper above it is
also recursive).

Nested `JOINT` blocks recurse once per level; `MAX_CHANNELS` does not bound depth (a
joint may omit `CHANNELS`) and is checked only after parsing.

**Failure:** a ~500 KB file with ~20k nested joints overflows the stack and **aborts the
process** through the public `import()` — the hostile-file corner the crate explicitly
claims to cover.

**Fix:** carry a depth counter and refuse past a named limit (e.g. 256) with
`BvhError::Limit`.

### H6 — zero-width npz dtype panics the AMASS reader
`crates/motion-io/src/npz.rs:265-303` (`parse_npy`).

A dtype such as `<U0` or `<f0` parses to `width == 0`, so `elem == 0`;
`need = count * 0 = 0` passes the truncation check and `body.chunks(elem)` panics
("chunk size must be non-zero").

**Failure:** one crafted member in an `.npz` archive crashes `import()`.

**Fix:** refuse `width == 0` as `Unsupported` immediately after parsing `descr`.

### H7 — USDC integer counts reach arithmetic and allocation unchecked
`crates/robot-usd/src/compression.rs:131-144` (`decode_ints`) and its callers
`crates/robot-usd/src/crate_file.rs:1093-1101` (`read_compressed_ints`: `(n * 2)`,
`n * width`), plus the uncapped `Vec::with_capacity(n)` in `read_tokens`
(crate_file.rs:358) and `n * 8` in `read_fields` (:378).

`count` comes straight from the file (`c.u64()? as usize`). The workspace sets
`overflow-checks = true` and `panic = "abort"` for release, so `count * 2` on a declared
`count = 2^63` **aborts the process** instead of returning `Malformed`; smaller
still-huge counts attempt multi-GB allocations; and had overflow wrapped, the shortened
`need` would pass the length check and `codes[group]` would index out of bounds. The rest
of the file already uses the right idioms (checked adds, `n.min(1 << 20)` caps) — these
sites just missed them.

**Fix:** bound `n` against bytes remaining (`n > c.d.len()` → `Malformed`) and use
checked arithmetic at all four sites.

## Medium-severity findings

| # | Location | Defect |
|---|---|---|
| M1 | `robot-usd/src/stage.rs:336-361` | Variant selections do not compose strongest-wins per set: every authored `variantSelection` pushes a source, so a stronger layer's `color=red` does not *replace* a weaker layer's `color=blue` — the deselected variant's opinions leak into the composed prim. Merge selection dictionaries strongest-wins per set first. |
| M2 | `robot-usd/src/stage.rs:697-724` | `arc_roots` infers arc roots by stripping the longest common *trailing* path run; when referenced and composed prims share a name (`/tpl/arm` → `/robot/arm`) it computes `/tpl` → `/robot` and rewrites sibling targets like `/tpl/base` that the module doc promises are "kept as authored". Derive roots from the actual arc, not path suffixes. |
| M3 | `robot-usd/src/usda.rs:658-730`, `stage.rs:527-579` | `prim` and `compose_prim` recurse per nesting level with no depth bound (the crate-file path decoder is iterative; these walks are not). Deeply nested prims abort via stack overflow. Apply the existing `MAX_NESTING` idea in both places. |
| M4 | `robot-artifacts/src/bundle.rs:442` | `from_toml` never checks unknown keys, so the digest seal does not cover injected content: add `[extra]` tables to a sealed `manifest.toml` and `read_bundle` still reports the digest valid — against ADR 0016's "reviewed, diffed, content-verified" claim and the crate's own `manifest.rs:348` practice. Call `unknown_keys` on the root and every sub-table; add an injection test. |
| M5 | `robot-artifacts/src/bundle.rs:804-816` | `load_weights` re-reads the blob *after* verification and returns the second, unverified read — a swapped file between the two reads is handed out as "verified". Hash the bytes actually returned against `weights_digest`. |
| M6 | `robot-types/src/obs.rs:163` + `robot-artifacts/src/bundle.rs:460` | Field widths from an untrusted manifest are unbounded and `ObservationSpec::width()` is an unchecked sum: two `2^63` widths abort on parse (overflow-checks). Enforce a `MAX_FIELD_WIDTH` and use `checked_add`. |
| M7 | `robot-skills/src/index.rs:126,130` + `selection.rs:314` | `sort_by(partial_cmp().unwrap())` panics on a NaN distance (one NaN query feature — e.g. an estimator fault — crashes the R2 selector mid-decision), and a NaN best distance *passes* the OOD gate (`NaN > ood_distance` is false). Use `total_cmp` and reject non-finite query features at `decide` entry. |
| M8 | `robot-eval/src/evaluate.rs:103-113` | Missing or non-finite per-seed metric values are silently dropped and duplicate seed entries double-counted: a metric can `Pass` from 2 of 5 declared seeds while `missing_seeds` stays empty. Track per-metric missing/non-finite seeds; dedupe or refuse repeated seeds. |
| M9 | `robot-supervisor/src/envelope.rs:123-133` + `supervisor.rs:351` | A missing or NaN actuator temperature yields derating factor 1.0 forever and no fault (`!t.is_finite() → 1.0`; `NaN >= limit` false): a failed thermistor grants sustained full torque. Treat non-finite temperature as a fault and derate to 0. |
| M10 | `robot-supervisor/src/supervisor.rs:366-379` | Mode-required interlocks and contact-energy budgets are checked only inside `if let Some(m) = &self.envelope` — a supervisor built without an `EnvelopeMonitor` silently skips them although the requirement comes from the *mode*. Move the interlock check out of the envelope branch, or refuse construction for modes with non-empty `required_interlocks()`. |
| M11 | `robot-cli/src/commands/bundle.rs:209` | `eval_bundle` passes `manifest.rate` as `host_policy_rate`, making the tier-schedule leg of the gate self-referential. Defensible for SimEval, but say so in the output or derive the host rate independently. |
| M12 | `motion-io/src/npz.rs:90-92` | The EOCD scan indexes `data[i..i+4]` even when `data.len() < 4` — panics on 0–3-byte input through the public `read_zip`/`read_npz` (not reachable via `import()`). Use `data.get(..)`. |
| M13 | `motion-io/src/c3d.rs:240-245` | `dims.iter().product()` and `count * width` are unchecked; with workspace overflow-checks a parameter declaring ≥9 dims of 255 aborts `import()`. Use `try_fold`/`checked_mul` as `npz.rs` already does. |
| M14 | `motion-io/src/capture.rs` (`Recorder::push`) | `arrival.nanos() - source_time.nanos()` with `source_time` parsed from the datagram as any `i64`: `i64::MIN` aborts the live-capture recorder on one malformed datagram. Use `saturating_sub` or bound at parse. |
| M15 | `motion-process/src/resample.rs` (`interpolate_frame`) | `Missing` samples' placeholder values are lerped/slerped and relabeled `Interpolated`: an occluded C3D marker re-enters the record as fake data at the world origin, which ground-plane and contact estimation then consume. If either bracketing sample is `Missing`, emit `Missing`. |

## Low-severity findings (by crate)

**robot-types** — `Name` accepts `"."`/`".."` (a component named `..` makes the bundle
blob path escape `weights/`; no deeper traversal is possible, but refuse it by name in
`Name::new`). `NormalizerWidth` reports the mean length even when the variance vector
mismatched. `TierSchedule::declare` accepts duplicate component assignments and
`component()` silently answers with the first.

**robot-artifacts** — No `schema_version` compatibility check in the bundle reader
(unlike `manifest.rs:364`). `[[a]]` array-table headers bounded by neither `max_entries`
nor `max_array_len` (a 4 MB file of `[[a]]` lines builds ~600k empty tables; capped by
`max_bytes` only). Minor undocumented TOML-subset leniencies (table redefinition reopens,
no newline required after `]`, `1_`/`_1` numbers, raw control chars in basic strings) —
fine internally, but external TOML tooling would reject some sealed documents.
`Digest::from_hex` accepts uppercase despite "lowercase" docs; `chunk.rs:484` rebuilds a
digest via hex round-trip where a `Digest::from_bytes` constructor belongs. Duplicate
`[[component]]` names verify; `load_weights` answers with the first.

**robot-devtools** — `depends_on` silently drops non-string array elements
(`inventory.rs:216-220`) instead of erroring like neighboring fields, which could hide an
edge from the staleness sweep. The ledger differ keys rows by first cell across all
tables (`ledger.rs:110-146`); add a duplicate-key guard if tables grow.

**robot-model / robot-usd** — `ImportReport.joints` doc still says "Revolute joints" but
counts fixed joints since the Go2 change. `ListOp::apply` pushes duplicate
`prepended`/`appended` items verbatim where OpenUSD dedupes. USDA arc syntax
(`references = @file@`, variant sets) dies with a generic parse error instead of the
crate's `Unsupported`-by-name contract (arcs are only composed from USDC today). Test
gaps: sublayer-cycle guard, `MAX_NESTING` reference-cycle guard, variant override across
layers (would have caught M1), offsets through two nested arcs.

**robot-sim / robot-sim-gpu / robot-tasks** — A mixed-format `solref` (`[0.02, -1.0]`) is
refused by name in `constraint.rs` and the GPU wrapper but silently reinterpreted by
`SimOptions::validate`/`solver.rs:309` — same invalid input, three paths, one silent.
EPA's convergence flag is computed and discarded (`collision.rs:1056`); an unconverged
depth is indistinguishable from a converged one — surface it on `Contact`/`StepReport`.
`TrackingTask::new` accepts empty references and panics on first `reference()` call
(`tasks.rs:375`); validate in the constructor. The GPU equality kernel's damping branch
disagrees with the CPU reference for mixed-sign `solref` (`launch.rs:525-534`) — masked
today by host validation, one refactor away from silent divergence; fix the kernel branch.
`implicitfast`'s damping derivative ignores force-limit saturation (`world.rs:737-748`) —
same approximation the reference simulator makes, but unnamed in rustdoc that otherwise
enumerates exclusions. (Info: single-point hull-on-ground contact is documented in place;
it matters more now that Go2/H1 make hull feet likely — known missing half of M-04.)

**robot-nn / robot-learn** — `forward_with_trace` panics on wrong-width input instead of
returning the existing `LayerError` (`mlp.rs:401-409`). `gaussian_policy.rs:98-104` doc
says "gradient of −log π" but returns `upstream · ∂logπ/∂θ` (sign-consistent in PPO;
fix the doc before it invites a sign bug). The observation normalizer keeps updating
during collection, so epoch-0 importance ratios are systematically ≠ 1 — standard
practice, but unstated. `gae.rs:46-48` length agreement is `debug_assert!` only.

**robot-skills** — `cache.rs:107-112`: `admit` checks the memory budget before retiring
the same skill's old entry, so replacing a resident expert can spuriously fail `Memory`.
`write_manifest` validates nothing (duplicate ids, NaN embeddings produce a file
`read_manifest` then refuses) — asymmetric against the round-trip contract.

**robot-perception** — `AttitudeFilter::propagate` documents rejecting bad `dt` but only
`debug_assert`s (`attitude.rs:152-157`); in release a NaN-derived `dt` corrupts
process-noise scaling. The standard ESKF covariance reset Jacobian after injection is
omitted — a known, small approximation at these error magnitudes; worth one doc sentence.

**robot-hardware** — design notes, not defects (documented, C-02 open): the wire protocol
has no authentication (CRC only) and `FirmwarePeer` dates command validity from receipt,
so a delayed datagram with a plausible sequence number is accepted as fresh — must be
revisited for a real bridge. The min-filtered clock offset can make successive mapped
sampling stamps non-monotonic by nanoseconds; ages use saturating subtraction, but a
consumer assuming monotone sampling times should know.

**robot-cli** — usage text still lists `data import`, `motion retarget`, `eval run`,
`train`, `adapt`, `replay` as "not implemented yet" (`main.rs:271-273`) though all six
dispatch to real implementations. `host_envelope` re-derives "actuated" with a different
filter (`Revolute | Continuous | Prismatic`, `commands/bundle.rs:113-117`) than
`RobotProfile::actuated_joints` ("everything except Fixed") — agreement is coincidental
on today's fixtures; build the envelope from `actuated_joints()`.

**motion-io** — the C3D label-dedup scheme can still collide (`["A","A_2",…,"A"]` → two
`A_2`), and `channel_index` silently resolves to the first; `POINT:SCALE == 0.0` zeroes
all integer-coded positions without an error. **motion-process** — runs of fewer than 3
present samples skip filtering silently and don't appear in the `FilterReport`.
**motion-retarget** — `chain_ratio` turns unknown link indices into `Vec3::ZERO` via
`unwrap_or` (plausible wrong scale, no error) and a zero-length human chain silently
returns 1.0 (`retarget.rs:272-284`); the foot-anchor projection hardcodes Z-up without
saying so at the call site.

## Clean by inspection

robot-control (fail-safe observation-age fallback, NaN-command conversion so the
supervisor's declared response is transmitted, readback-verified shadow mode),
robot-persona (recovery/veto checked first, no path from arbiter to command),
robot-viewer (no panic paths in any non-test module), motion-validate (every division
guarded, refusals by name), robot-adapt (ladder ordering, evidence demands and
`ResidualBound` authority/rate enforcement all verified), robot-eval's statistics
(Welch–Satterthwaite, TOST as max-of-one-sided, t-quantiles vs published tables), and
robot-hardware's frame layer (CRC-before-trust, every-single-bit-flip test, width
bounds-checked before use, wraparound-safe sequence accounting). The acceptance matrix
cannot close a row without a ledger artifact. robot-sim-gpu's tolerance record cannot
silently rot (a feature-off test re-asserts every measured value). robot-tasks'
privileged-observation audit has no bypass found, and held-out randomization ranges
cannot leak into training by construction.

## Cross-cutting themes

1. **NaN and shape fail-open at gates.** H1, H3's sibling (M7's OOD gate), M8, M9: every
   comparison gate written as `if x > y { refuse }` silently passes NaN, and `zip`
   truncates length mismatches. A workspace-wide sweep for `partial_cmp().unwrap()`,
   bare float comparisons in refusal positions, and un-length-checked `zip` in
   validators would close the class; the finiteness discipline that the wire codec and
   `MotionRecord::validate` already enforce is the model to copy.
2. **Newer code skips disciplines older code established.** `bundle.rs` vs
   `manifest.rs` (unknown keys, schema compatibility — M4); `SimOptions::validate` vs
   `constraint.rs` (solref format); the USDC count sites vs the checked idioms in the
   same file (H7). Review checklists beat memory here: the disciplines exist, they
   just were not re-applied.
3. **Recursion depth is bounded in some parsers and not others.** `MAX_NESTING` exists
   for reference arcs but not for BVH joints (H5), USDA prims, or the composed-tree
   walk (M3).
4. **`overflow-checks = true` + `panic = "abort"` turns unchecked arithmetic on
   untrusted counts into process aborts** (H7, M6, M13, M14). The setting is right —
   it converts silent wraparound into loud failure — but it raises the bar: every
   `count * k` on a file-supplied count must be checked arithmetic with a named error.
5. **Capacity edge cases can degrade safety instead of escalating** (H4 fault log,
   robot-skills cache admit order). Fixed-capacity safety structures should fold
   severity before capacity is consulted, never after.

## Suggested fix order

1. H1 + H2 + M11 together — they are the three legs of the same activation-gate story,
   and the new `robot eval --bundle` tests make regression coverage cheap.
2. H3 + H4 + M9 + M10 — the two safety-path crates (skills selection, supervisor);
   each fix is a few lines plus a test.
3. H5–H7 + M12–M14 — the hostile-file hardening sweep in motion-io and robot-usd
   (depth counters, checked arithmetic, zero-width refusal).
4. M1–M5, M15 and the lows as ordinary maintenance.

None of the fixes needs a new dependency or an ADR beyond possibly one line in ADR 0016's
consequences (unknown-key coverage of the digest seal, M4).

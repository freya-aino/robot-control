# 0009. Test-purpose frame dump and SVG render format

- Status: accepted (decided before the writers were implemented).
- Date: 2026-10-28.
- Scope: `robot sim multibody --floating --out <dir>` frame files and per-frame
  SVG renders. Not a viewer, not a qualification claim, not a balance result.
- Touches: `crates/robot-viewer/src/pose.rs`, `crates/robot-viewer/src/csv.rs`,
  `crates/robot-viewer/src/jsonl.rs`, `crates/robot-viewer/src/svg.rs`,
  `crates/robot-viewer/src/timeline.rs`, `crates/robot-viewer/src/lib.rs`,
  `crates/robot-viewer/README.md`, `crates/robot-cli/src/commands/sim.rs`,
  `crates/robot-cli/src/main.rs`, `docs/decisions/0009-test-render-format.md`.

## Context

PLAN.md Phase 2 asks for inspection output that does not depend on a viewer.
SPECS.md §6 places inspection in `robot-viewer`, and Appendix B.6 blocks any
rendering toolkit (Rerun, egui, `urdf-viz`) until it is qualified. So the render
path must be std-only (ADR 0003). The existing `ViewerFrame` needs a full
`LinkPoses` for every link, which a floating-base run does not produce as a
row-per-step record. The frame dump therefore uses its own record type,
`PoseFrame`, holding the pelvis pose and the single-DOF joint angles, and
`Timeline` is generalised over a `Timed` trait so it can hold either type.

## Decision

1. **Record: `PoseFrame`.** One record per simulation step with:
   - `timestamp`: `Timestamp::sampling(nanos, ClockId::Simulation)`.
   - `step`: 1-based step index.
   - `pelvis_position`: `[x, y, z]` in metres, from `q[0..3]`.
   - `pelvis_quaternion`: `[w, x, y, z]`, from `q[3..7]` (the order used by
     `robot_types::frames::Quat`).
   - `joint_angles`: one value per single-DOF actuated joint (`Revolute`,
     `Continuous`, `Prismatic`), in profile order, from `q[7..]`.
2. **Step convention.** Row `k` (k = 1..=N) is the state after the k-th step.
   Its time is `k * dt` computed as the integer `k * dt_nanos`, where
   `dt_nanos = Duration::from_secs_f64(dt).nanos()`. Times are never accumulated
   in floating point. The initial state is not written as a row.
3. **Files in `<dir>`:**
   - `frames.csv`: leading schema marker `# robot-viewer.pose-frames v1`, header `step,time_ns,pelvis_x,pelvis_y,pelvis_z,pelvis_qw,pelvis_qx,pelvis_qy,pelvis_qz,<joint names>`, one row per step, and end footer `# end steps=N`.
   - `frames.jsonl`: first line is a header object
     `{"schema":"robot-viewer.pose-frames","schema_version":1,"joints":[...]}`.
     Each following line is one step:
     `{"step":k,"time_ns":t,"pelvis_position":[x,y,z],"pelvis_quaternion":[w,x,y,z],"joint_angles":[...]}`.
     The final line is a footer object:
     `{"end":true,"steps":N}`.
   - `svg/frame_NNNNNN.svg`: one SVG per step, `NNNNNN` the zero-padded step index.
   - `run.commit`: the commit manifest, written last. Its two lines are
     `# robot-viewer.pose-commit v1` and `steps N`. A directory is a completed run
     only if this file exists and matches the frame files (see §8).
   - Staging and lock: while a run is in progress, all of the above are written
     under `<dir>/.pose-staging/`. The staging directory is also the output lock
     (§10). Nothing else in `<dir>` is touched until commit.
4. **Number format.** Floats use Rust's `Display` for `f64` (shortest
   round-trip, never exponent form). The writer refuses non-finite values, so no
   `NaN` or `inf` reaches the file. Both formats are therefore deterministic on a
   given toolchain. This is the same reproducibility contract as ADR 0008.
5. **Readers are strict.** `csv::read_file` and `jsonl::read_file` reject: a wrong or
   missing header or schema marker; a wrong schema name or version; a missing,
   malformed or mismatched end footer; extra records after the end footer; a
   field count or joint count that disagrees with the header; a non-finite or
   ill-formed number (JSONL number tokens must follow RFC 8259 number syntax); a
   step index that is not 1, 2, 3, ... in order; a
   timestamp that is not strictly increasing; a timestamp on a clock other than
   `ClockId::Simulation`; a timestamp outside `[0, 2^53]` ns; or a JSONL integer
   field (`step`, `time_ns`, `schema_version`, or footer `steps`) whose token is
   not an exact unsigned decimal integer (fractional and exponent forms are
   rejected, and leading zeros are rejected). Both readers validate the decoded
   frame through the same contract. Any error returns no partial result. The
   JSONL parser is a small std-only parser that accepts only the value shapes
   this format uses.
   - Bounds on pathological input. A header line is at most 64 KiB. Before
     creating or changing the output directory, `FrameDump::create` counts the
     exact encoded CSV and JSON Lines header lengths (including newlines) and
     rejects either one over this limit with `HeaderTooLong`. A step line is at
     most `4096 + 340 * joints` bytes, where `joints` is the count the header
     declares; this allows up to 340 bytes for each joint's finite `f64` Display
     value and separator, plus a 4096-byte allowance for fixed fields, seven pose
     values, integers and JSON punctuation. Longer lines are rejected before
     they are buffered in full. A dump declares at most 256 joints (`MAX_JOINTS`),
     which covers the G1 with margin. JSON nesting is at most 8 levels
     (`MAX_DEPTH`), so the recursive value parser cannot grow the stack on a
     hostile line. Each reader bound returns an explicit `Malformed` error with
     the line number.
   - Step limit. A dump holds at most 999 999 steps (`MAX_FRAME_STEPS`), because
     SVG names are six digits wide (`frame_NNNNNN.svg`). `FrameDump::push` refuses a
     larger step with `TooManySteps`, and the CLI refuses `--out` with a longer
     `--steps` before any simulation runs. Names therefore round-trip and sort in
     step order for every step a dump can hold.
6. **Timeline.** `Timeline` is generic over a `Timed` trait (`fn timestamp(&self)
   -> Timestamp`). The default type is `ViewerFrame`, so existing call sites and
   tests keep compiling. Reading back gives `Timeline<PoseFrame>`.
7. **SVG format.** A fixed-size side view (x forward, z up) with: the ground line
   at `z = 0`, the pelvis as a marker at `(x, z)`, an orientation tick from the
   quaternion, and a bar for each joint angle. Numbers are printed with fixed
   `{:.6}` precision. The output is a function of the `PoseFrame` alone, with no
   timestamps or host paths. Golden files in `crates/robot-viewer/tests/golden/`
   pin the output for a fixed fixture.
8. **Transactional publication.**
   - `FrameDump::create` rejects an empty path with `PoseError::EmptyDirectory`
     (CLI: `EXIT_USAGE`, before any filesystem call). It validates joint names,
     count and the encoded header bounds in memory first; an oversized header
     returns `PoseError::HeaderTooLong` without creating `<dir>`. It then takes
     the output lock (§10), then preflights: it creates `<dir>` and `<dir>/svg`, probes that each accepts
     a new file, refuses a directory where `frames.csv`, `frames.jsonl` or
     `run.commit` would go, and refuses a directory at any per-step SVG name
     already in `svg/`. It then opens the staged files with their headers. Any
     failure before any step runs leaves every existing file in `<dir>` intact.
     The CLI also rejects `--out` without `--floating` with `EXIT_USAGE`.
   - `--out` requires `--floating`. A fixed base has no pelvis pose, so the
     command returns `EXIT_USAGE` otherwise.
   - `FrameDump::finish` writes the footers, checks the staged CSV and JSONL with
     the strict readers as described in §9, and then runs a publish preflight: every target that a commit will rename onto (the
     two frame files, `run.commit`, and `svg/frame_NNNNNN.svg` for steps
     `1..=N`) must not be a directory. Only after that check does it remove the
     old `run.commit`, rename the staged files and SVGs into place, remove SVG
     names past the new last step, and write `run.commit` last through a staged
     file.
   - Safety and preflight: `FrameDump::push` validates the clock is
     `ClockId::Simulation`, the timestamp is in `[0, 2^53]` ns, and is strictly
     increasing. The CLI preflights the total run time (`steps * dt <= 2^53` ns)
     before creating the output directory.
   - Dropping a `FrameDump` before a successful `finish` removes only its own
     staging directory. A failed or abandoned run therefore leaves any earlier
     committed run intact, and a failed run that already removed `run.commit`
     is not loadable.
   - Readers: `pose::load_committed(dir)` is the only accepted entry point for a
     dump directory. It requires `run.commit`, a matching marker and `steps N`,
     CSV and JSONL that agree, and exactly the SVGs for steps `1..=N`. Any other
     result is an error with no partial log.
   - A repeated run with the same seed, dt and step count yields byte-identical
     `frames.csv`, `frames.jsonl`, `run.commit` and SVG files.
9. **Verification happens before publish, under the lock, streaming.** There is no
   readback after the commit. `FrameDump::finish` checks the staged files before
   anything is published. `pose::verify_pair` reads `frames.csv` and
   `frames.jsonl` side by side, one frame from each at a time. It uses the same
   line rules as the readers in §5: the same schema and header checks, field
   counts, number syntax, step order, strictly rising times, and footer. Each
   pair must match bit for bit, including every pelvis value and joint angle, so
   `0.0` and `-0.0` differ. Both files must hold the written step count, and the
   last frame they hold must equal the last frame pushed. Memory is one frame per
   file plus one line buffer per file, not the run length. The explicit
   `load_committed` uses the same verifier and keeps the frames; `finish` does not.
   A failed check returns an error and leaves any committed run in `<dir>`
   untouched. The CLI does not read the committed directory back, and it removes
   no outputs after `finish` returns. This means no code runs after the lock is
   released, so a second writer's committed run can never be deleted by this
   process.
10. **Output lock and stale-stage recovery.**
   - The lock is the directory `<dir>/.pose-staging/`, created with
     `fs::create_dir`, which is atomic: exactly one `create` succeeds. The owner
     file records `pid <n>`, and on Linux also `namespace <pid-ns link>` and
     `boot <boot_id>`.
   - The lock is held from `create` until `finish` returns or the `FrameDump` is
     dropped. `FrameDump` ensures the lock is released exactly once: `commit()`
     takes the staging path from `self`, removing it, and `Drop` does nothing if
     `self.staging` is `None`. Every check, publish and cleanup of the staging
     directory happens inside that window. Invariant, tested by
     `commit_releases_lock_exactly_once`: after `commit()`, `self.staging` is
     `None`, and dropping the dump never touches a staging directory that another
     writer created after the release. The test simulates that writer by creating
     `.pose-staging/marker` after `commit()` and checks that the marker survives
     the drop.
   - An existing staging directory is reclaimed only if all of these hold. Its
     owner file parses and records a namespace and boot. Both match this process's
     pid namespace and boot id. The owner pid is absent from `/proc`. Reclaim
     renames the directory aside, then deletes it only if the owner file is the
     same one that was checked; otherwise it restores it. A pid is meaningful only
     inside the namespace and boot that issued it, so a foreign or unidentified
     owner is never reclaimed. Non-Linux platforms never reclaim.
   - A live owner, this process's own pid, a foreign namespace or boot, a missing
     or unreadable owner file, or an unknown platform gives `PoseError::Locked`
     with the path and reason. Nothing is removed. The message tells the user to
     remove the directory by hand only when no run is active.
   - A `SIGKILL` skips `Drop`, so the staging directory stays until a later run in
     the same namespace and boot finds the owner dead.
   - A kill between `create_dir` and the owner write leaves a staging directory
     with no owner. That case is refused, not reclaimed, so it needs manual removal
     when no run is active. This is the safe failure chosen over guessing.
   - Dropping a `FrameDump` before `finish` removes only the staging directory.

## Consequences

- No new third-party crates. `robot-viewer` stays std-only apart from its existing
  `robot-types` and `robot-model` dependencies.
- Floating-base G1 runs write one SVG per step. Measured with the release build of
  `robot sim multibody --floating --dt 0.0001 --out <dir>` after the streaming
  verifier landed: the 20 000-step run writes 20 000 SVGs, `frames.csv` of 16.7 MB
  and `frames.jsonl` of 18.3 MB, exits 0 (PASS), and takes about 3.0 s wall time on
  two repeated runs. The 2 000-step run writes 2 000 SVGs and exits 3, which means
  INCONCLUSIVE because it is shorter than the qualification duration. It is not a
  failure, and its files are committed. Peak RSS, sampled from `VmHWM` in
  `/proc/<pid>/status` about every 20 ms, was 3 720 kB for 2 000 steps and
  3 708 kB for 20 000 steps. The two lengths therefore show no growth with run
  length, which is what streaming verification is meant to give. The sampler can
  miss a short peak, so these are sampled values, not an exact allocator
  high-water mark. The earlier whole-file verifier was not measured, so no memory
  comparison is claimed against it.
- The commit is not atomic against power loss or `SIGKILL` during the renames. A
  crash in that window can leave a removed manifest with a partly renamed
  directory. Such a directory is rejected by `load_committed`, and the next run
  overwrites it. No test injects a failure during the renames.
- Stale staging: a staging directory left by a killed run is reclaimed by the next
  `create` only when its owner process is provably dead in the same pid namespace
  and boot (§10). Otherwise `create` refuses and keeps it, so a `SIGINT` or
  `SIGKILL` can leave up to one run's staged data in `<dir>/.pose-staging/` until
  that run is removed or its owner is gone. Tests cover the empty path, preflight,
  premature drop, a rejected create, concurrent create, a failed `finish` that keeps
  the committed run and releases the lock, stale reclaim, a live, own, foreign-boot,
  bare-pid or unidentified owner, and a committed run that is missing its manifest,
  an SVG or has a bad manifest.
- Residual risk: the lock is only as reliable as the owner identity. A writer in
  another container or on another host is refused, not reclaimed, because its
  namespace or boot differs. Two hosts that share a boot id would still need manual
  care; boot ids are per-kernel, so this is not expected in practice.
- The SVG shows the pelvis and joint angles, not the full skeleton. Link poses for
  every body are not written, so the render is not a kinematic reconstruction.
- The frame record is not a qualification artifact. A frame dump proves only that
  the run was written and read back.

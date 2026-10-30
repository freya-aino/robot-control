# Motion fixtures

This directory holds the motion-capture fixtures that `motion-io` and
`robot-types` tests may cite as a `SourceLicense.source_reference`. SPECS.md
Section 8.1 item 7 requires every motion source to carry an explicit license
record (training, redistribution and commercial permissions, and retention),
and Appendix C question C-07 (licensed corpora) is still open.

## Contents

No motion files are committed yet. The BVH test inputs of
`crates/motion-io/src/bvh.rs` are inline strings authored for this project,
and the `fixture-internal` license they cite is defined here:

| Identifier | Training | Redistribution | Commercial | Retention | Source |
| --- | --- | --- | --- | --- | --- |
| `fixture-internal` | permitted | not permitted | not permitted | none | authored inline in this repository's tests; no external capture session |

## Adding a fixture

A file added here needs a row in the table above with its origin, the
license text or URL, the accessed date and the permissions it grants; the
`robot motion validate` command imports any file without such a row as
`unaudited`, which is the correct conservative default. A fixture does not
make a corpus admissible: that is a C-07 decision record.

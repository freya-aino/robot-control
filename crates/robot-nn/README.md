# robot-nn

Forward-only inference and the exportable operator set shared by `robot-learn`
(training) and `robot-control` (deployment). Depends only on `robot-types`.

This crate is **not** in SPECS.md §6's workspace table; its extraction is
justified and recorded in
[`docs/decisions/0004-nn-and-supervisor-crate-extraction.md`](../../docs/decisions/0004-nn-and-supervisor-crate-extraction.md)
under the Appendix B.3 rule that a new crate needs "ownership or dependency
isolation" to warrant existing.

## What's here

| Module | Contents |
| --- | --- |
| `layer` | `Activation` (identity/tanh/ReLU/sigmoid) and `Linear`, a dense `y = Wx + b` layer |
| `mlp` | `Mlp`: a validated sequence of `Linear` + `Activation` pairs; `Scratch`: preallocated ping-pong buffers; `forward_with_trace` for backpropagation |
| `init` | Fan-in-scaled normal initialization driven by a `robot_types::rng::Rng`, so network init is part of one reproducible seed stream |

## Why the operator set needs no "unsupported operator" check

SPECS.md §7.4 requires that "Unsupported operators must fail before
deployment" — written with ONNX import in mind, where a parser might accept a
graph using an operator the runtime doesn't implement. This crate has no
analogous gap: `Activation` is a four-variant closed enum and `Linear` is the
only layer kind. A bundle referencing an operator this crate doesn't support
is not *representable*, so there's nothing to validate at load time that the
type system hasn't already excluded at construction time.

## Why inference is allocation-free

SPECS.md §5: "Keep inference resident and allocate working memory before
activation... The control path MUST NOT perform... heap allocation after
activation." `Scratch::new(mlp.scratch_len())` is the one allocation, done
once at startup; every subsequent `Mlp::forward` call writes into caller-owned
buffers and touches no allocator. `forward_with_trace`, used only by
`robot-learn`'s backward pass, is the deliberate exception — SPECS.md §5's
no-allocation rule binds the control path, not the training path.

## Action parity is structural, not a separate check

PLAN.md Phase 4 asks for "identical Rust normalization/action transforms for
training and deployment" and "action-parity checks between learner inference
and runtime inference." Because `robot-learn` and `robot-control` both call
the *same* `Mlp::forward` on the *same* weight data, two builds that load an
identical bundle necessarily produce bit-identical output — there is no
second implementation to drift out of sync with the first. The test
`two_independently_built_networks_with_the_same_weights_agree_exactly` in
`mlp.rs` demonstrates this directly rather than asserting it.

## Testing

`cargo test -p robot-nn` — 18 tests: activation derivatives checked against
central-difference numerics, forward passes checked against hand-traced
arithmetic, determinism across repeated calls, and rejection of every
malformed shape (mismatched adjacent widths, wrong activation count,
undersized scratch).

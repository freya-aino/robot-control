//! Forward-only inference and the exportable operator set shared by the
//! learner and the deployed runtime.
//!
//! SPECS.md Section 6: "Training and deployment share observation/action
//! schemas and tested normalization logic. Deployment does not depend on
//! training autograd code." This crate is the shared forward pass that
//! guarantee requires: `robot-learn` depends on it to run rollouts and to
//! evaluate the network during backpropagation (via
//! [`mlp::Mlp::forward_with_trace`]), and `robot-control` depends on it to
//! run deployed inference — without `robot-control` depending on
//! `robot-learn` at all. See
//! `docs/decisions/0004-nn-and-supervisor-crate-extraction.md` for why this
//! is a crate rather than a module of `robot-learn`.
//!
//! The operator set ([`layer::Activation`], [`layer::Linear`]) is closed by
//! construction: SPECS.md Section 7.4 requires that "Unsupported operators
//! must fail before deployment," and here there is no unsupported-operator
//! state to fail on, because the enum admits nothing else.
//!
//! [`mlp::Scratch`] is the preallocated working memory SPECS.md Section 5
//! requires: "Keep inference resident and allocate working memory before
//! activation... The control path MUST NOT perform... heap allocation after
//! activation." [`mlp::Mlp::forward`] touches no allocator once `Scratch` has
//! been built.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod init;
pub mod layer;
pub mod mlp;

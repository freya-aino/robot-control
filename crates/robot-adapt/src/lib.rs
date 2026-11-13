//! Limited-data adaptation, per SPECS.md Section 11.3.
//!
//! | Section 11.3 row | Module |
//! | --- | --- |
//! | 1. System identification | [`sysid`] |
//! | 2. History-conditioned adaptation | [`history`], [`encoder`] |
//! | 3. Simulator correction model | [`correction`] |
//! | 4. Bounded robot-side residual | [`residual`] |
//! | 5. Small adapter fine-tuning | [`finetune`] |
//! | 6. Two-phase latent identification | [`encoder`] (phase 2), [`history`] |
//! | 7. Simulation-guided fine-tuning | [`finetune`] (`GuidedObjective`), gated by [`ladder`] |
//! | 8. Retrieval-conditioned adaptation | [`retrieval`] |
//! | 9. Cross-embodiment few-shot transfer | [`morphology`] |
//! | 10. Multi-simulator randomization | refused by [`ladder`]: Section 9.1 grants no second simulator |
//!
//! The table's ordering rule is enforced by [`ladder::AdaptationLadder`];
//! obligations 5 and 6 are [`report::AdaptationReport`]. Training code
//! (`encoder::fit`, `encoder::train`, `finetune::fine_tune`) is behind the
//! default `training` feature so a deployable consumer can depend on this
//! crate with `default-features = false` and get no autograd code.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod correction;
pub mod encoder;
pub mod finetune;
pub mod history;
pub mod ladder;
pub mod morphology;
pub mod report;
pub mod residual;
pub mod retrieval;
pub mod sysid;

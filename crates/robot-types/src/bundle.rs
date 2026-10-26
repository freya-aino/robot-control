//! The policy bundle manifest of SPECS.md Section 7.4.
//!
//! "A deployable bundle contains model weights in a Rust-loadable versioned
//! format, graph/architecture version, observation schema, normalization
//! statistics, action semantics and scaling, recurrent-state specification,
//! rates, robot-profile compatibility, training data lineage, safety-envelope
//! assumptions, transition contracts, benchmark results, and
//! checksum/signature."
//!
//! Every one of those is a required field below. The manifest describes the
//! bundle; the weights themselves are a separate content-addressed blob, which
//! is what lets `robot-control` validate compatibility before it loads anything
//! large.
//!
//! The manifest is also where SPECS.md Section 5.1 item 1 lands: the tier
//! assignment of every learned component is recorded here, and
//! [`PolicyBundleManifest::check_activatable`] refuses a bundle that omits one.

use crate::evidence::EvidenceState;
use crate::ids::Name;
use crate::obs::{ActionSpec, ObservationSpec, VectorError};
use crate::schema::{SchemaError, SchemaVersion};
use crate::tier::{TierError, TierSchedule};
use crate::time::Duration;
use crate::transition::LatentSpaceId;
use crate::units::Hertz;
use core::fmt;

/// The schema version this crate writes for bundle manifests.
pub const BUNDLE_SCHEMA_VERSION: SchemaVersion = SchemaVersion::new(1, 0);

/// How recurrent state is handled across a control boundary.
///
/// SPECS.md Section 10.3: "Recurrent state must be reset or transformed by an
/// explicit contract." The contract is declared here, not inferred at the switch.
#[derive(Clone, PartialEq, Debug)]
pub enum RecurrentStateSpec {
    /// The policy is feed-forward and carries no state.
    Stateless,
    /// The policy carries recurrent state of the given width.
    Recurrent {
        /// Number of scalars.
        width: usize,
        /// What happens to the state when this policy is activated.
        on_activation: RecurrentStatePolicy,
    },
}

impl RecurrentStateSpec {
    /// Width of the recurrent state, zero when stateless.
    #[must_use]
    pub const fn width(&self) -> usize {
        match self {
            Self::Stateless => 0,
            Self::Recurrent { width, .. } => *width,
        }
    }
}

/// What a policy does with recurrent state when it is activated.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RecurrentStatePolicy {
    /// Zero the state.
    ResetToZero,
    /// Carry the predecessor's state, permitted only when the predecessor
    /// declares an identical recurrent specification.
    CarryFromCompatiblePredecessor,
    /// Apply a named, tested transformation.
    TransformedBy {
        /// Identity of the transformation artifact.
        artifact: &'static str,
    },
}

/// A declared transition contract for one outgoing switch.
///
/// SPECS.md Appendix A.2: "The explicit statement of which states permit a
/// switch and how recurrent state is reset or transformed."
#[derive(Clone, PartialEq, Debug)]
pub struct TransitionContract {
    /// Skill or bundle this contract permits switching to.
    pub target: String,
    /// Named initiation guard that must hold at the switch.
    pub guard: String,
    /// Minimum dwell time before the switch may be taken, where one applies.
    /// Recovery may preempt it (SPECS.md Section 10.4).
    pub minimum_dwell: Option<Duration>,
    /// Whether recovery can preempt the dwell time.
    pub preemptible_by_recovery: bool,
    /// Identity of the regression run that tested this transition.
    pub tested_by: String,
}

/// The safety envelope a bundle was trained and evaluated against.
///
/// These are *assumptions the bundle makes*, not limits it enforces. SPECS.md
/// Section 12.2 requires controls independent of the learned component, so the
/// supervisor holds the authoritative envelope and this record is what gets
/// compared against it at load time.
#[derive(Clone, PartialEq, Debug)]
pub struct SafetyEnvelopeAssumptions {
    /// Robot profile identity and version.
    pub robot_profile: String,
    /// Safety profile identity and version.
    pub safety_profile: String,
    /// Per-joint position limits assumed, in radians.
    pub position_limits: Vec<(f64, f64)>,
    /// Per-joint velocity limits assumed, in radians per second.
    pub velocity_limits: Vec<f64>,
    /// Per-joint torque limits assumed, in newton-metres.
    pub torque_limits: Vec<f64>,
    /// Operating modes this bundle is qualified for.
    pub qualified_modes: Vec<crate::mode::OperatingMode>,
}

/// A benchmark result recorded in the bundle.
#[derive(Clone, PartialEq, Debug)]
pub struct BenchmarkResult {
    /// Metric name.
    pub metric: String,
    /// Measured value.
    pub value: f64,
    /// Unit symbol.
    pub unit: String,
    /// Acceptance profile version the value was measured against.
    pub acceptance_profile: String,
    /// Evidence state of this number.
    pub evidence: EvidenceState,
    /// Identity of the run that produced it.
    pub run_id: String,
}

/// The training data lineage of a bundle.
#[derive(Clone, PartialEq, Debug)]
pub struct DataLineage {
    /// Content identities of every motion or corpus input.
    pub inputs: Vec<String>,
    /// Content identity of the configuration used.
    pub config_hash: String,
    /// `devenv.lock` revision the run executed under (SPECS.md Section 2.2
    /// item 1 and Section 13).
    pub environment_lock: String,
    /// Source revision of this repository.
    pub source_revision: String,
    /// Episodes held out from training, named so a later retrieval cannot reach
    /// them (SPECS.md Section 9.7.2 item 3).
    pub held_out: Vec<String>,
}

/// A learned component inside a bundle.
#[derive(Clone, PartialEq, Debug)]
pub struct ComponentManifest {
    /// Stable component name.
    pub name: Name,
    /// Architecture/graph version of this component.
    pub architecture_version: SchemaVersion,
    /// Content identity of the weight blob.
    pub weights_digest: String,
    /// Size of the weight blob in bytes, checked before loading.
    pub weights_bytes: u64,
    /// Latent space this component decodes, where it is a decoder.
    pub decodes_latent_space: Option<LatentSpaceId>,
}

/// A deployable bundle's manifest.
#[derive(Clone, PartialEq, Debug)]
pub struct PolicyBundleManifest {
    /// Schema version of the manifest itself.
    pub schema_version: SchemaVersion,
    /// Stable bundle identity.
    pub bundle_id: String,
    /// Observation schema the actor reads. Must contain no privileged field.
    pub observation: ObservationSpec,
    /// Normalization statistics, frozen.
    pub normalizer_mean: Vec<f64>,
    /// Normalization variance.
    pub normalizer_variance: Vec<f64>,
    /// Samples the statistics were estimated from.
    pub normalizer_count: u64,
    /// Clip applied to normalized observations.
    pub normalizer_clip: f64,
    /// Action semantics and scaling.
    pub action: ActionSpec,
    /// Recurrent state specification.
    pub recurrent: RecurrentStateSpec,
    /// Rate the bundle expects to be invoked at.
    pub rate: Hertz,
    /// Tier assignments for every learned component.
    pub tiers: TierSchedule,
    /// Learned components.
    pub components: Vec<ComponentManifest>,
    /// Safety envelope assumptions.
    pub envelope: SafetyEnvelopeAssumptions,
    /// Training data lineage.
    pub lineage: DataLineage,
    /// Outgoing transition contracts.
    pub transitions: Vec<TransitionContract>,
    /// Benchmark results.
    pub benchmarks: Vec<BenchmarkResult>,
    /// Digest over the manifest and every referenced blob.
    pub digest: String,
    /// Detached signature over `digest`, where promotion requires one.
    pub signature: Option<String>,
}

/// Why a bundle could not be activated.
#[derive(Clone, PartialEq, Debug)]
pub enum BundleError {
    /// The actor reads a privileged observation field.
    PrivilegedObservation(VectorError),
    /// A learned component declared no tier.
    UndeclaredTier {
        /// The component.
        component: String,
    },
    /// The tier schedule itself is invalid.
    Tier(TierError),
    /// Normalization statistics did not match the observation width.
    NormalizerWidth {
        /// Width the observation declares.
        expected: usize,
        /// Width the statistics have.
        found: usize,
    },
    /// The bundle targets a different robot profile.
    ProfileMismatch {
        /// Profile the bundle declares.
        bundle: String,
        /// Profile presented.
        host: String,
    },
    /// The bundle is not qualified for the requested mode.
    ModeNotQualified {
        /// The mode requested.
        mode: crate::mode::OperatingMode,
    },
    /// The bundle's assumed envelope is wider than the host's enforced one.
    ///
    /// Section 11.3 item 4: "widening a bound to accommodate an adapter is a
    /// safety-profile change, not a tuning step." A bundle that assumes more
    /// authority than the supervisor grants is refused at load.
    EnvelopeWiderThanHost {
        /// Which limit.
        limit: &'static str,
        /// Joint index.
        joint: usize,
        /// Value the bundle assumes.
        bundle: f64,
        /// Value the host enforces.
        host: f64,
    },
    /// A latent action has no decoder in the bundle.
    MissingDecoder {
        /// Latent space name.
        space: String,
    },
    /// The manifest had no digest.
    Unsigned,
    /// A schema problem.
    Schema(SchemaError),
    /// A benchmark result rests on evidence too weak for the claim.
    InsufficientEvidence {
        /// The metric.
        metric: String,
        /// State it carries.
        state: EvidenceState,
        /// State required.
        required: EvidenceState,
    },
}

impl fmt::Display for BundleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PrivilegedObservation(e) => write!(f, "{e}"),
            Self::UndeclaredTier { component } => write!(
                f,
                "component {component:?} declares no tier; Section 5.1 item 1 \
                 forbids activating it"
            ),
            Self::Tier(e) => write!(f, "{e}"),
            Self::NormalizerWidth { expected, found } => write!(
                f,
                "normalization statistics have width {found} for a {expected}-wide observation"
            ),
            Self::ProfileMismatch { bundle, host } => {
                write!(f, "bundle targets {bundle:?}, host is {host:?}")
            }
            Self::ModeNotQualified { mode } => {
                write!(f, "bundle is not qualified for mode {mode}")
            }
            Self::EnvelopeWiderThanHost {
                limit,
                joint,
                bundle,
                host,
            } => write!(
                f,
                "bundle assumes {limit} {bundle} on joint {joint}, host enforces {host}"
            ),
            Self::MissingDecoder { space } => {
                write!(
                    f,
                    "latent space {space:?} has no decoder component in the bundle"
                )
            }
            Self::Unsigned => write!(f, "manifest has no digest"),
            Self::Schema(e) => write!(f, "{e}"),
            Self::InsufficientEvidence {
                metric,
                state,
                required,
            } => write!(
                f,
                "benchmark {metric:?} carries {state} but the claim requires {required}"
            ),
        }
    }
}

impl core::error::Error for BundleError {}

impl From<TierError> for BundleError {
    fn from(e: TierError) -> Self {
        Self::Tier(e)
    }
}

impl From<SchemaError> for BundleError {
    fn from(e: SchemaError) -> Self {
        Self::Schema(e)
    }
}

/// The limits a host actually enforces, presented at load time.
#[derive(Clone, PartialEq, Debug)]
pub struct HostEnvelope {
    /// Robot profile identity and version.
    pub robot_profile: String,
    /// Per-joint position limits, in radians.
    pub position_limits: Vec<(f64, f64)>,
    /// Per-joint velocity limits, in radians per second.
    pub velocity_limits: Vec<f64>,
    /// Per-joint torque limits, in newton-metres.
    pub torque_limits: Vec<f64>,
}

impl PolicyBundleManifest {
    /// Checks everything that must hold before a bundle may be activated.
    ///
    /// This is the load-time gate referenced by the "Wrong artifact deployed"
    /// hazard in SPECS.md Section 12.2, whose control is "Refuse to activate".
    pub fn check_activatable(
        &self,
        host: &HostEnvelope,
        mode: crate::mode::OperatingMode,
        host_policy_rate: Hertz,
    ) -> Result<(), BundleError> {
        if self.digest.trim().is_empty() {
            return Err(BundleError::Unsigned);
        }

        // Section 9.2: an exported actor reads no privileged input.
        self.observation
            .audit_deployable()
            .map_err(BundleError::PrivilegedObservation)?;

        let width = self.observation.width();
        if self.normalizer_mean.len() != width || self.normalizer_variance.len() != width {
            return Err(BundleError::NormalizerWidth {
                expected: width,
                found: self.normalizer_mean.len(),
            });
        }

        // Section 5.1 item 1: every learned component declares a tier.
        for component in &self.components {
            if self.tiers.component(component.name.as_str()).is_none() {
                return Err(BundleError::UndeclaredTier {
                    component: component.name.to_string(),
                });
            }
        }
        self.tiers.validate(host_policy_rate)?;

        if self.envelope.robot_profile != host.robot_profile {
            return Err(BundleError::ProfileMismatch {
                bundle: self.envelope.robot_profile.clone(),
                host: host.robot_profile.clone(),
            });
        }
        if !self.envelope.qualified_modes.contains(&mode) {
            return Err(BundleError::ModeNotQualified { mode });
        }

        // The bundle may assume a *narrower* envelope than the host enforces,
        // never a wider one.
        for (j, (bundle, host_limit)) in self
            .envelope
            .position_limits
            .iter()
            .zip(&host.position_limits)
            .enumerate()
        {
            if bundle.0 < host_limit.0 {
                return Err(BundleError::EnvelopeWiderThanHost {
                    limit: "position lower bound",
                    joint: j,
                    bundle: bundle.0,
                    host: host_limit.0,
                });
            }
            if bundle.1 > host_limit.1 {
                return Err(BundleError::EnvelopeWiderThanHost {
                    limit: "position upper bound",
                    joint: j,
                    bundle: bundle.1,
                    host: host_limit.1,
                });
            }
        }
        for (j, (bundle, host_limit)) in self
            .envelope
            .velocity_limits
            .iter()
            .zip(&host.velocity_limits)
            .enumerate()
        {
            if bundle > host_limit {
                return Err(BundleError::EnvelopeWiderThanHost {
                    limit: "velocity",
                    joint: j,
                    bundle: *bundle,
                    host: *host_limit,
                });
            }
        }
        for (j, (bundle, host_limit)) in self
            .envelope
            .torque_limits
            .iter()
            .zip(&host.torque_limits)
            .enumerate()
        {
            if bundle > host_limit {
                return Err(BundleError::EnvelopeWiderThanHost {
                    limit: "torque",
                    joint: j,
                    bundle: *bundle,
                    host: *host_limit,
                });
            }
        }

        // A latent action is meaningless without its decoder.
        if let crate::obs::ActionSemantics::Latent { space_id } = self.action.semantics {
            let has_decoder = self.components.iter().any(|c| {
                c.decodes_latent_space
                    .as_ref()
                    .is_some_and(|s| s.name == space_id)
            });
            if !has_decoder {
                return Err(BundleError::MissingDecoder {
                    space: space_id.to_owned(),
                });
            }
        }

        Ok(())
    }

    /// Checks that a hardware claim rests on hardware evidence.
    ///
    /// SPECS.md Section 3: `reproduced_in_sim` does not license "any hardware
    /// claim". A release report calls this before quoting a benchmark as a
    /// hardware result.
    pub fn check_hardware_claims(&self) -> Result<(), BundleError> {
        for b in &self.benchmarks {
            if b.evidence < EvidenceState::ValidatedOnHardware {
                return Err(BundleError::InsufficientEvidence {
                    metric: b.metric.clone(),
                    state: b.evidence,
                    required: EvidenceState::ValidatedOnHardware,
                });
            }
        }
        Ok(())
    }

    /// Total weight bytes across components, used to preallocate before
    /// activation (SPECS.md Section 5: no allocation after activation).
    #[must_use]
    pub fn total_weight_bytes(&self) -> u64 {
        self.components.iter().map(|c| c.weights_bytes).sum()
    }

    /// The transition contract to a named target, if declared.
    #[must_use]
    pub fn contract_to(&self, target: &str) -> Option<&TransitionContract> {
        self.transitions.iter().find(|t| t.target == target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mode::OperatingMode;
    use crate::obs::{ActionSemantics, Availability, FieldSpec};
    use crate::tier::{MeasuredCompute, MissingUpperTierBehavior, Tier, TierAssignment};

    fn observation(privileged: bool) -> ObservationSpec {
        let mut fields = vec![FieldSpec::deployable("joint_position", 2, "rad").unwrap()];
        if privileged {
            fields.push(FieldSpec::privileged("friction", 1, "1").unwrap());
        }
        ObservationSpec::new(SchemaVersion::new(1, 0), fields).unwrap()
    }

    fn tiers() -> TierSchedule {
        let mut s = TierSchedule::new();
        s.declare(TierAssignment {
            component: "actor".to_owned(),
            tier: Tier::Policy,
            rate: Hertz(100.0),
            budget: Duration::from_millis(8),
            measured_compute: MeasuredCompute {
                percentile: 99.9,
                duration: Duration::from_micros(900),
                max: Duration::from_micros(1500),
                run_id: "bench-7".to_owned(),
            },
            input_freshness_limit: Duration::from_millis(100),
            on_missing_upper: MissingUpperTierBehavior::UseDeclaredDefault,
        });
        s
    }

    fn manifest() -> PolicyBundleManifest {
        let obs = observation(false);
        let width = obs.width();
        PolicyBundleManifest {
            schema_version: BUNDLE_SCHEMA_VERSION,
            bundle_id: "tracker/1".to_owned(),
            observation: obs,
            normalizer_mean: vec![0.0; width],
            normalizer_variance: vec![1.0; width],
            normalizer_count: 1000,
            normalizer_clip: 5.0,
            action: ActionSpec::identity(
                SchemaVersion::new(1, 0),
                ActionSemantics::JointPositionTarget,
                2,
                1.0,
            )
            .unwrap(),
            recurrent: RecurrentStateSpec::Stateless,
            rate: Hertz(100.0),
            tiers: tiers(),
            components: vec![ComponentManifest {
                name: Name::new("actor").unwrap(),
                architecture_version: SchemaVersion::new(1, 0),
                weights_digest: "sha256:aa".to_owned(),
                weights_bytes: 4096,
                decodes_latent_space: None,
            }],
            envelope: SafetyEnvelopeAssumptions {
                robot_profile: "fixture-arm/1.0".to_owned(),
                safety_profile: "fixture-arm-safety/1.0".to_owned(),
                position_limits: vec![(-1.0, 1.0), (-1.0, 1.0)],
                velocity_limits: vec![2.0, 2.0],
                torque_limits: vec![10.0, 10.0],
                qualified_modes: vec![OperatingMode::SimEval, OperatingMode::HwShadow],
            },
            lineage: DataLineage {
                inputs: vec!["sha256:motion".to_owned()],
                config_hash: "sha256:cfg".to_owned(),
                environment_lock: "devenv.lock@abc123".to_owned(),
                source_revision: "git:d67a672".to_owned(),
                held_out: vec!["session-9".to_owned()],
            },
            transitions: Vec::new(),
            benchmarks: Vec::new(),
            digest: "sha256:bundle".to_owned(),
            signature: None,
        }
    }

    fn host() -> HostEnvelope {
        HostEnvelope {
            robot_profile: "fixture-arm/1.0".to_owned(),
            position_limits: vec![(-1.5, 1.5), (-1.5, 1.5)],
            velocity_limits: vec![3.0, 3.0],
            torque_limits: vec![12.0, 12.0],
        }
    }

    #[test]
    fn a_well_formed_bundle_activates() {
        assert!(manifest()
            .check_activatable(&host(), OperatingMode::HwShadow, Hertz(100.0))
            .is_ok());
    }

    #[test]
    fn a_component_without_a_tier_cannot_be_activated() {
        let mut m = manifest();
        m.components.push(ComponentManifest {
            name: Name::new("strategy").unwrap(),
            architecture_version: SchemaVersion::new(1, 0),
            weights_digest: "sha256:bb".to_owned(),
            weights_bytes: 8192,
            decodes_latent_space: None,
        });
        assert!(matches!(
            m.check_activatable(&host(), OperatingMode::HwShadow, Hertz(100.0)),
            Err(BundleError::UndeclaredTier { .. })
        ));
    }

    #[test]
    fn a_privileged_observation_is_refused_at_load() {
        let mut m = manifest();
        m.observation = observation(true);
        m.normalizer_mean = vec![0.0; m.observation.width()];
        m.normalizer_variance = vec![1.0; m.observation.width()];
        assert!(matches!(
            m.check_activatable(&host(), OperatingMode::HwShadow, Hertz(100.0)),
            Err(BundleError::PrivilegedObservation(
                VectorError::PrivilegedInputInActor { .. }
            ))
        ));
        assert_eq!(
            m.observation.fields()[1].availability,
            Availability::Privileged
        );
    }

    #[test]
    fn a_wider_assumed_envelope_is_refused() {
        let mut m = manifest();
        m.envelope.torque_limits = vec![20.0, 10.0];
        assert!(matches!(
            m.check_activatable(&host(), OperatingMode::HwShadow, Hertz(100.0)),
            Err(BundleError::EnvelopeWiderThanHost {
                limit: "torque",
                joint: 0,
                ..
            })
        ));

        let mut m2 = manifest();
        m2.envelope.position_limits = vec![(-2.0, 1.0), (-1.0, 1.0)];
        assert!(matches!(
            m2.check_activatable(&host(), OperatingMode::HwShadow, Hertz(100.0)),
            Err(BundleError::EnvelopeWiderThanHost {
                limit: "position lower bound",
                ..
            })
        ));
    }

    #[test]
    fn a_narrower_assumed_envelope_is_accepted() {
        let mut m = manifest();
        m.envelope.torque_limits = vec![5.0, 5.0];
        assert!(m
            .check_activatable(&host(), OperatingMode::HwShadow, Hertz(100.0))
            .is_ok());
    }

    #[test]
    fn a_profile_or_mode_mismatch_is_refused() {
        let mut other = host();
        other.robot_profile = "other/2.0".to_owned();
        assert!(matches!(
            manifest().check_activatable(&other, OperatingMode::HwShadow, Hertz(100.0)),
            Err(BundleError::ProfileMismatch { .. })
        ));
        assert!(matches!(
            manifest().check_activatable(
                &host(),
                OperatingMode::HwFreestandingNoncontact,
                Hertz(100.0)
            ),
            Err(BundleError::ModeNotQualified { .. })
        ));
    }

    #[test]
    fn a_latent_action_without_its_decoder_is_refused() {
        let mut m = manifest();
        m.action = ActionSpec::identity(
            SchemaVersion::new(1, 0),
            ActionSemantics::Latent {
                space_id: "boxing-latent",
            },
            2,
            1.0,
        )
        .unwrap();
        assert!(matches!(
            m.check_activatable(&host(), OperatingMode::HwShadow, Hertz(100.0)),
            Err(BundleError::MissingDecoder { .. })
        ));

        m.components[0].decodes_latent_space = Some(LatentSpaceId {
            name: "boxing-latent".to_owned(),
            version: SchemaVersion::new(1, 0),
            decoder_artifact: "sha256:dec".to_owned(),
            dimension: 2,
            constraint: crate::transition::LatentConstraint::UnitHypersphere,
        });
        assert!(m
            .check_activatable(&host(), OperatingMode::HwShadow, Hertz(100.0))
            .is_ok());
    }

    #[test]
    fn an_undigested_manifest_is_refused() {
        let mut m = manifest();
        m.digest = "  ".to_owned();
        assert!(matches!(
            m.check_activatable(&host(), OperatingMode::HwShadow, Hertz(100.0)),
            Err(BundleError::Unsigned)
        ));
    }

    #[test]
    fn mismatched_normalization_width_is_refused() {
        let mut m = manifest();
        m.normalizer_mean.pop();
        assert!(matches!(
            m.check_activatable(&host(), OperatingMode::HwShadow, Hertz(100.0)),
            Err(BundleError::NormalizerWidth { .. })
        ));
    }

    #[test]
    fn a_simulation_benchmark_does_not_license_a_hardware_claim() {
        let mut m = manifest();
        m.benchmarks.push(BenchmarkResult {
            metric: "tracking_error".to_owned(),
            value: 0.03,
            unit: "m".to_owned(),
            acceptance_profile: "fixture-arm-accept/1.0".to_owned(),
            evidence: EvidenceState::ReproducedInSim,
            run_id: "run-3".to_owned(),
        });
        assert!(matches!(
            m.check_hardware_claims(),
            Err(BundleError::InsufficientEvidence { .. })
        ));
    }

    #[test]
    fn weight_bytes_are_summed_for_preallocation() {
        assert_eq!(manifest().total_weight_bytes(), 4096);
    }

    #[test]
    fn transition_contracts_are_looked_up_by_target() {
        let mut m = manifest();
        m.transitions.push(TransitionContract {
            target: "guard".to_owned(),
            guard: "double-support".to_owned(),
            minimum_dwell: Some(Duration::from_millis(200)),
            preemptible_by_recovery: true,
            tested_by: "regression-12".to_owned(),
        });
        assert!(m.contract_to("guard").is_some());
        assert!(m.contract_to("absent").is_none());
        assert_eq!(RecurrentStateSpec::Stateless.width(), 0);
        assert_eq!(
            RecurrentStateSpec::Recurrent {
                width: 64,
                on_activation: RecurrentStatePolicy::ResetToZero
            }
            .width(),
            64
        );
    }
}

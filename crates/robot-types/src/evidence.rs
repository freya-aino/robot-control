//! The five evidence states of SPECS.md Section 3, with promotion and demotion.
//!
//! The states are ordered and "the order is the point: each one licenses a narrow
//! class of claim and nothing above it". This module makes the ordering a type,
//! the admitting evidence a required argument, and demotion an operation that
//! exists rather than a convention nobody applies.
//!
//! Section 3 also forbids two specific mistakes, both encoded here: evidence does
//! not transfer between artifacts, and cross-simulator agreement is not a state.

use core::cmp::Ordering;
use core::fmt;

/// How well supported a claim is.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum EvidenceState {
    /// Any mention: conversation, blog post, product demo, unread citation.
    /// Licenses scheduling an investigation and nothing else.
    Unverified,
    /// Primary source retrieved, identity confirmed, claim located in text.
    /// Licenses quoting the stated method and its stated conditions.
    PaperVerified,
    /// Published implementation read at a pinned commit, licence recorded,
    /// dependency closure inventoried. Licenses statements about what the
    /// implementation does and requires — not about its correctness.
    CodeInspected,
    /// Our Rust implementation reaches the stated behaviour on a declared
    /// fixture across multiple seeds. Licenses simulation claims inside the
    /// tested operating region only.
    ReproducedInSim,
    /// Measured on the designated robot under a versioned acceptance profile.
    /// Licenses claims about that robot, that profile, that envelope.
    ValidatedOnHardware,
}

impl EvidenceState {
    /// All states in ascending order.
    pub const ALL: [Self; 5] = [
        Self::Unverified,
        Self::PaperVerified,
        Self::CodeInspected,
        Self::ReproducedInSim,
        Self::ValidatedOnHardware,
    ];

    /// The rank used for ordering.
    #[must_use]
    pub const fn rank(self) -> u8 {
        match self {
            Self::Unverified => 0,
            Self::PaperVerified => 1,
            Self::CodeInspected => 2,
            Self::ReproducedInSim => 3,
            Self::ValidatedOnHardware => 4,
        }
    }

    /// The canonical identifier used in the ledger.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::Unverified => "unverified",
            Self::PaperVerified => "paper_verified",
            Self::CodeInspected => "code_inspected",
            Self::ReproducedInSim => "reproduced_in_sim",
            Self::ValidatedOnHardware => "validated_on_hardware",
        }
    }

    /// Parses a ledger identifier.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|s| s.id() == text)
    }

    /// What this state licenses, quoted from the table in SPECS.md Section 3.
    #[must_use]
    pub const fn licenses(self) -> &'static str {
        match self {
            Self::Unverified => "scheduling an investigation",
            Self::PaperVerified => "quoting the stated method and its stated conditions",
            Self::CodeInspected => {
                "statements about what the implementation does and what it requires"
            }
            Self::ReproducedInSim => "simulation claims within the tested operating region",
            Self::ValidatedOnHardware => "claims about that robot, that profile, that envelope",
        }
    }

    /// What this state explicitly does not license.
    #[must_use]
    pub const fn does_not_license(self) -> &'static str {
        match self {
            Self::Unverified => "any implementation decision, estimate or citation",
            Self::PaperVerified => {
                "assuming the method works, reproduces, or applies to our embodiment"
            }
            Self::CodeInspected => "statements about numerical correctness or performance",
            Self::ReproducedInSim => "any hardware claim, and any claim outside the tested region",
            Self::ValidatedOnHardware => {
                "claims about other embodiments, payloads, firmware or envelopes"
            }
        }
    }

    /// True when a claim in this state may be cited in an implementation
    /// decision at all. Only [`EvidenceState::Unverified`] may not.
    #[must_use]
    pub const fn is_citable(self) -> bool {
        self.rank() > Self::Unverified.rank()
    }
}

impl PartialOrd for EvidenceState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for EvidenceState {
    fn cmp(&self, other: &Self) -> Ordering {
        self.rank().cmp(&other.rank())
    }
}

impl fmt::Display for EvidenceState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

/// The artifact that admits a claim to a state.
///
/// SPECS.md Section 3: "Promotion requires the admitting evidence in the
/// following table and a dated ledger entry naming the artifact that supplies
/// it." An empty artifact reference is therefore not constructible.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct AdmittingArtifact {
    /// What kind of artifact this is.
    pub kind: ArtifactKind,
    /// Stable reference: a URL, a DOI, a commit hash, a run identifier.
    pub reference: String,
    /// ISO-8601 date the evidence was obtained.
    pub dated: String,
}

/// The kinds of artifact that can admit a claim.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ArtifactKind {
    /// A primary source document at a named version.
    PrimarySource,
    /// A published implementation at a pinned commit.
    PinnedCommit,
    /// A reproduction run of ours, by run identifier.
    ReproductionRun,
    /// A hardware evaluation run against a versioned acceptance profile.
    HardwareRun,
}

impl ArtifactKind {
    /// The highest state this kind of artifact can admit.
    ///
    /// A primary source cannot admit `code_inspected`; a reproduction run
    /// cannot admit `validated_on_hardware`.
    #[must_use]
    pub const fn admits_up_to(self) -> EvidenceState {
        match self {
            Self::PrimarySource => EvidenceState::PaperVerified,
            Self::PinnedCommit => EvidenceState::CodeInspected,
            Self::ReproductionRun => EvidenceState::ReproducedInSim,
            Self::HardwareRun => EvidenceState::ValidatedOnHardware,
        }
    }

    /// The canonical identifier used in the ledger.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::PrimarySource => "primary_source",
            Self::PinnedCommit => "pinned_commit",
            Self::ReproductionRun => "reproduction_run",
            Self::HardwareRun => "hardware_run",
        }
    }
}

/// Why a promotion or demotion was refused.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum EvidenceError {
    /// The artifact cannot admit the requested state.
    InsufficientArtifact {
        /// State requested.
        requested: EvidenceState,
        /// Highest state the artifact can admit.
        admits_up_to: EvidenceState,
    },
    /// The requested state is not above the current one.
    NotAPromotion {
        /// Current state.
        current: EvidenceState,
        /// State requested.
        requested: EvidenceState,
    },
    /// The artifact reference or date was empty.
    UnnamedArtifact,
    /// An attempt to carry evidence from one artifact to another.
    ///
    /// SPECS.md Section 3: "a reproduction of an algorithm does not promote the
    /// paper that proposed it".
    TransferBetweenArtifacts {
        /// Subject the evidence belongs to.
        held_by: String,
        /// Subject the caller tried to promote.
        requested_for: String,
    },
}

impl fmt::Display for EvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientArtifact {
                requested,
                admits_up_to,
            } => write!(
                f,
                "artifact admits at most {admits_up_to}, cannot admit {requested}"
            ),
            Self::NotAPromotion { current, requested } => {
                write!(f, "{requested} is not above the current state {current}")
            }
            Self::UnnamedArtifact => write!(f, "admitting artifact has no reference or date"),
            Self::TransferBetweenArtifacts {
                held_by,
                requested_for,
            } => write!(
                f,
                "evidence held by {held_by:?} does not transfer to {requested_for:?}"
            ),
        }
    }
}

impl core::error::Error for EvidenceError {}

/// Why a claim was demoted.
///
/// SPECS.md Section 3: "if a dependency version, robot profile, firmware or
/// acceptance profile changes, every claim resting on it returns to the highest
/// state its surviving evidence still supports."
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum DemotionCause {
    /// A dependency version changed.
    DependencyVersion {
        /// Which dependency.
        dependency: String,
    },
    /// The robot profile changed.
    RobotProfile {
        /// Profile identity.
        profile: String,
    },
    /// Device firmware changed.
    Firmware {
        /// Firmware identity.
        firmware: String,
    },
    /// The acceptance profile changed.
    AcceptanceProfile {
        /// Profile identity.
        profile: String,
    },
    /// A hosted model's identity, version or endpoint changed.
    ///
    /// SPECS.md Section 2.3 item 6 makes this a version change, "because a
    /// hosted model is not pinnable by a lockfile".
    ModelService {
        /// Model identity.
        model: String,
    },
    /// The locked development environment changed.
    LockedEnvironment {
        /// `devenv.lock` revision.
        revision: String,
    },
}

impl fmt::Display for DemotionCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DependencyVersion { dependency } => {
                write!(f, "dependency version changed: {dependency}")
            }
            Self::RobotProfile { profile } => write!(f, "robot profile changed: {profile}"),
            Self::Firmware { firmware } => write!(f, "firmware changed: {firmware}"),
            Self::AcceptanceProfile { profile } => {
                write!(f, "acceptance profile changed: {profile}")
            }
            Self::ModelService { model } => write!(f, "model service changed: {model}"),
            Self::LockedEnvironment { revision } => {
                write!(f, "locked environment changed: {revision}")
            }
        }
    }
}

/// A claim about one named subject, with the evidence that supports it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Claim {
    /// The subject the evidence attaches to: a paper, a crate at a version, a
    /// bundle, a robot profile. Evidence does not transfer off this subject.
    pub subject: String,
    /// The exact claim being supported, stated before the evidence is sought.
    pub statement: String,
    state: EvidenceState,
    history: Vec<EvidenceEvent>,
}

/// One promotion or demotion in a claim's history.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum EvidenceEvent {
    /// A promotion admitted by an artifact.
    Promoted {
        /// State reached.
        to: EvidenceState,
        /// The admitting artifact.
        artifact: AdmittingArtifact,
    },
    /// A demotion forced by a version change.
    Demoted {
        /// State fallen back to.
        to: EvidenceState,
        /// Why.
        cause: DemotionCause,
    },
}

impl Claim {
    /// A new claim at `unverified`, which is where Appendix B.1 says every
    /// incoming survey entry starts.
    #[must_use]
    pub fn new(subject: impl Into<String>, statement: impl Into<String>) -> Self {
        Self {
            subject: subject.into(),
            statement: statement.into(),
            state: EvidenceState::Unverified,
            history: Vec::new(),
        }
    }

    /// The current state.
    #[inline]
    #[must_use]
    pub const fn state(&self) -> EvidenceState {
        self.state
    }

    /// The promotion and demotion history, oldest first.
    #[must_use]
    pub fn history(&self) -> &[EvidenceEvent] {
        &self.history
    }

    /// Promotes to `target`, if `artifact` admits it and `subject` matches.
    pub fn promote(
        &mut self,
        target: EvidenceState,
        artifact: AdmittingArtifact,
        artifact_subject: &str,
    ) -> Result<(), EvidenceError> {
        if artifact.reference.trim().is_empty() || artifact.dated.trim().is_empty() {
            return Err(EvidenceError::UnnamedArtifact);
        }
        if artifact_subject != self.subject {
            return Err(EvidenceError::TransferBetweenArtifacts {
                held_by: artifact_subject.to_owned(),
                requested_for: self.subject.clone(),
            });
        }
        if target <= self.state {
            return Err(EvidenceError::NotAPromotion {
                current: self.state,
                requested: target,
            });
        }
        let ceiling = artifact.kind.admits_up_to();
        if target > ceiling {
            return Err(EvidenceError::InsufficientArtifact {
                requested: target,
                admits_up_to: ceiling,
            });
        }
        self.state = target;
        self.history.push(EvidenceEvent::Promoted {
            to: target,
            artifact,
        });
        Ok(())
    }

    /// Demotes to the highest state the surviving evidence still supports.
    ///
    /// `surviving` is what remains valid after the change; the claim falls to
    /// the minimum of its current state and that.
    pub fn demote(&mut self, cause: DemotionCause, surviving: EvidenceState) {
        if surviving < self.state {
            self.state = surviving;
            self.history.push(EvidenceEvent::Demoted {
                to: surviving,
                cause,
            });
        }
    }

    /// True when this claim may be cited to support an implementation decision.
    #[must_use]
    pub const fn is_citable(&self) -> bool {
        self.state.is_citable()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact(kind: ArtifactKind) -> AdmittingArtifact {
        AdmittingArtifact {
            kind,
            reference: "arXiv:2601.22517v1".to_owned(),
            dated: "2026-10-26".to_owned(),
        }
    }

    #[test]
    fn states_are_ordered() {
        assert!(EvidenceState::Unverified < EvidenceState::PaperVerified);
        assert!(EvidenceState::ReproducedInSim < EvidenceState::ValidatedOnHardware);
        assert!(!EvidenceState::Unverified.is_citable());
        assert!(EvidenceState::PaperVerified.is_citable());
        assert_eq!(
            EvidenceState::parse("code_inspected"),
            Some(EvidenceState::CodeInspected)
        );
        assert_eq!(EvidenceState::parse("cross_sim_agreement"), None);
    }

    #[test]
    fn a_paper_cannot_admit_a_hardware_claim() {
        let mut claim = Claim::new("arXiv:2601.22517", "staged latent pipeline ordering");
        let err = claim
            .promote(
                EvidenceState::ValidatedOnHardware,
                artifact(ArtifactKind::PrimarySource),
                "arXiv:2601.22517",
            )
            .unwrap_err();
        assert_eq!(
            err,
            EvidenceError::InsufficientArtifact {
                requested: EvidenceState::ValidatedOnHardware,
                admits_up_to: EvidenceState::PaperVerified,
            }
        );
        assert_eq!(claim.state(), EvidenceState::Unverified);
    }

    #[test]
    fn robostriker_is_paper_verified_and_nothing_more() {
        let mut claim = Claim::new("arXiv:2601.22517", "staged latent pipeline ordering");
        claim
            .promote(
                EvidenceState::PaperVerified,
                artifact(ArtifactKind::PrimarySource),
                "arXiv:2601.22517",
            )
            .unwrap();
        assert_eq!(claim.state(), EvidenceState::PaperVerified);
        assert!(claim.is_citable());
        // No source release was identified, so code_inspected is unreachable.
        assert!(claim
            .promote(
                EvidenceState::CodeInspected,
                artifact(ArtifactKind::PrimarySource),
                "arXiv:2601.22517"
            )
            .is_err());
    }

    #[test]
    fn evidence_does_not_transfer_between_artifacts() {
        let mut paper = Claim::new("arXiv:2601.22517", "method works on our embodiment");
        let err = paper
            .promote(
                EvidenceState::ReproducedInSim,
                AdmittingArtifact {
                    kind: ArtifactKind::ReproductionRun,
                    reference: "run-2026-10-26-001".to_owned(),
                    dated: "2026-10-26".to_owned(),
                },
                "robot-learn/latent-stage-2",
            )
            .unwrap_err();
        assert!(matches!(
            err,
            EvidenceError::TransferBetweenArtifacts { .. }
        ));
    }

    #[test]
    fn unnamed_artifacts_are_refused() {
        let mut claim = Claim::new("x", "y");
        let err = claim
            .promote(
                EvidenceState::PaperVerified,
                AdmittingArtifact {
                    kind: ArtifactKind::PrimarySource,
                    reference: "   ".to_owned(),
                    dated: "2026-10-26".to_owned(),
                },
                "x",
            )
            .unwrap_err();
        assert_eq!(err, EvidenceError::UnnamedArtifact);
    }

    #[test]
    fn sideways_promotion_is_refused() {
        let mut claim = Claim::new("x", "y");
        claim
            .promote(
                EvidenceState::PaperVerified,
                artifact(ArtifactKind::PrimarySource),
                "x",
            )
            .unwrap();
        assert!(matches!(
            claim.promote(
                EvidenceState::PaperVerified,
                artifact(ArtifactKind::PrimarySource),
                "x"
            ),
            Err(EvidenceError::NotAPromotion { .. })
        ));
    }

    #[test]
    fn firmware_change_demotes_a_hardware_claim() {
        let mut claim = Claim::new("bundle:tracker-v3", "tracks held-out motions on robot A");
        claim
            .promote(
                EvidenceState::ValidatedOnHardware,
                AdmittingArtifact {
                    kind: ArtifactKind::HardwareRun,
                    reference: "hw-run-7".to_owned(),
                    dated: "2026-10-26".to_owned(),
                },
                "bundle:tracker-v3",
            )
            .unwrap();
        claim.demote(
            DemotionCause::Firmware {
                firmware: "1.4.2 -> 1.5.0".to_owned(),
            },
            EvidenceState::ReproducedInSim,
        );
        assert_eq!(claim.state(), EvidenceState::ReproducedInSim);
        assert_eq!(claim.history().len(), 2);
    }

    #[test]
    fn demotion_never_raises_a_state() {
        let mut claim = Claim::new("x", "y");
        claim.demote(
            DemotionCause::ModelService {
                model: "some-model".to_owned(),
            },
            EvidenceState::ValidatedOnHardware,
        );
        assert_eq!(claim.state(), EvidenceState::Unverified);
        assert!(claim.history().is_empty());
    }
}

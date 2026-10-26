//! Operating modes, their actuation authority and their entry preconditions.
//!
//! SPECS.md Section 4: "The mode is a property of the running process, not a
//! configuration comment: it MUST be recorded in every episode record, MUST be
//! reported by the supervisor on request, and MUST NOT change without passing
//! back through the entry preconditions of the target mode."
//!
//! The table in Section 4 is reproduced here as data, so that the runtime can
//! check authority rather than a reviewer checking a comment. Teleoperation and
//! scripted control are *modifiers* on a hardware mode, not modes of their own,
//! which is why they are a separate [`ControlSource`].

use core::fmt;

/// An operating mode of the platform.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum OperatingMode {
    /// Deterministic replay from recorded artifacts; no device attached.
    OfflineReplay,
    /// Simulated training.
    SimTrain,
    /// Simulated evaluation against a frozen bundle.
    SimEval,
    /// Live sensors, commands computed and discarded at the adapter.
    HwShadow,
    /// Motion with the robot mechanically supported.
    HwSupported,
    /// Full motion, no contact targets.
    HwFreestandingNoncontact,
    /// Motion against an instrumented fixture or robot.
    HwContact,
}

/// What a mode is permitted to do with actuators.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ActuationAuthority {
    /// No device attached at all.
    None,
    /// Simulated actuation only.
    Simulated,
    /// Commands computed, logged and discarded at the adapter.
    ComputedAndDiscarded,
    /// Motion within a reduced envelope, robot mechanically supported.
    SupportedMotion,
    /// Full motion within the normal envelope, no contact permitted.
    FreeMotionNoncontact,
    /// Full motion with per-target contact permission.
    ContactPermitted,
}

impl ActuationAuthority {
    /// True when this authority can move a physical actuator.
    #[must_use]
    pub const fn moves_hardware(self) -> bool {
        matches!(
            self,
            Self::SupportedMotion | Self::FreeMotionNoncontact | Self::ContactPermitted
        )
    }
}

/// An interlock a mode requires to be armed before and during motion.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub enum Interlock {
    /// Hardware emergency stop, armed and verified.
    EmergencyStop,
    /// The transmit path is disabled, and the disable is confirmed by loopback
    /// readback every tick (SPECS.md Section 4, `hw-shadow`).
    TransmitDisableAsserted,
    /// A mechanical support fixture is present.
    SupportFixture,
    /// The reduced envelope is loaded and enforced by the supervisor.
    ReducedEnvelope,
    /// The physical area interlock is closed.
    AreaInterlock,
    /// A watchdog running outside the policy process.
    IndependentWatchdog,
    /// Per-target contact permission is granted.
    ContactPermission,
    /// An independent match logger is recording.
    IndependentMatchLogger,
    /// Operator presence control, where the robot requires one.
    OperatorPresence,
}

impl Interlock {
    /// The canonical identifier used in records.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::EmergencyStop => "e-stop",
            Self::TransmitDisableAsserted => "transmit-disable-asserted",
            Self::SupportFixture => "support-fixture",
            Self::ReducedEnvelope => "reduced-envelope",
            Self::AreaInterlock => "area-interlock",
            Self::IndependentWatchdog => "independent-watchdog",
            Self::ContactPermission => "contact-permission",
            Self::IndependentMatchLogger => "independent-match-logger",
            Self::OperatorPresence => "operator-presence",
        }
    }
}

/// What the mode does when input is lost or a fault is detected.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FaultResponse {
    /// Abort with a diagnostic; never fabricate missing samples.
    AbortWithDiagnostic,
    /// Fail the run; partial results are labelled as such.
    FailRun,
    /// Fail the suite rather than reporting a partial pass.
    FailSuite,
    /// Keep computing, record the fault, never enable transmit as a fallback.
    ContinueComputingRecordFault,
    /// Robot-specific hold or controlled stop per its safety profile.
    RobotSpecificHoldOrStop,
    /// The validated fallback of SPECS.md Section 10.4. Falling is a designed
    /// outcome here, not a fault path.
    ValidatedFallback,
    /// Terminate the bout, not just the skill; record the full event buffer.
    TerminateBout,
}

/// Who is producing the commands inside a mode.
///
/// SPECS.md Section 4: teleoperation and scripted control "are subject to the
/// same envelope, supervisor and artifact requirements, and an operator MUST NOT
/// be able to command outside the envelope that a learned policy would be
/// clamped to".
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ControlSource {
    /// A learned policy bundle.
    Policy,
    /// A human operator, scoped to the enclosing hardware mode.
    Teleoperation,
    /// A deterministic scripted controller.
    Scripted,
}

impl ControlSource {
    /// The canonical identifier used in records.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::Policy => "policy",
            Self::Teleoperation => "teleop",
            Self::Scripted => "scripted",
        }
    }
}

/// A mode together with the source of its commands.
///
/// This pair is what a record carries, because "teleop in `hw-supported`" and
/// "policy in `hw-supported`" are different runs with the same envelope.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ModeContext {
    /// The enclosing mode.
    pub mode: OperatingMode,
    /// Who produced the commands.
    pub source: ControlSource,
}

impl fmt::Display for ModeContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.mode.id(), self.source.id())
    }
}

/// A precondition that must hold before a mode may be entered.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub enum EntryPrecondition {
    /// Artifact hashes resolve.
    ArtifactHashesResolve,
    /// The bundle schema matches the recorded observations.
    BundleSchemaMatchesRecords,
    /// Seeds, config and data manifest are pinned.
    RunInputsPinned,
    /// The bundle is frozen and evaluation seeds are disjoint from training.
    FrozenBundleDisjointSeeds,
    /// The transmit path is disabled and the disable verified by loopback.
    TransmitDisableVerified,
    /// Joint sign, offset and units verified per SPECS.md Section 7.1.
    JointConventionsVerified,
    /// A reduced torque and velocity envelope is loaded.
    ReducedEnvelopeLoaded,
    /// The `hw-supported` qualification has passed.
    SupportedQualificationPassed,
    /// The area is cleared and the exclusion zone enforced.
    AreaClearedAndEnforced,
    /// The noncontact qualification has passed.
    NoncontactQualificationPassed,
    /// Contact experiments are separately enabled.
    ContactSeparatelyEnabled,
    /// Scoring has been validated against scripted scenarios.
    ScoringValidated,
    /// The contact energy budget is set and enforced outside the policy.
    ContactEnergyBudgetSet,
    /// Every learned component in the bundle declares a tier (SPECS.md
    /// Section 5.1 item 1: "A component with no declared tier MUST NOT be
    /// activated").
    AllComponentsDeclareTier,
}

impl EntryPrecondition {
    /// The canonical identifier used in records.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::ArtifactHashesResolve => "artifact-hashes-resolve",
            Self::BundleSchemaMatchesRecords => "bundle-schema-matches-records",
            Self::RunInputsPinned => "run-inputs-pinned",
            Self::FrozenBundleDisjointSeeds => "frozen-bundle-disjoint-seeds",
            Self::TransmitDisableVerified => "transmit-disable-verified",
            Self::JointConventionsVerified => "joint-conventions-verified",
            Self::ReducedEnvelopeLoaded => "reduced-envelope-loaded",
            Self::SupportedQualificationPassed => "supported-qualification-passed",
            Self::AreaClearedAndEnforced => "area-cleared-and-enforced",
            Self::NoncontactQualificationPassed => "noncontact-qualification-passed",
            Self::ContactSeparatelyEnabled => "contact-separately-enabled",
            Self::ScoringValidated => "scoring-validated",
            Self::ContactEnergyBudgetSet => "contact-energy-budget-set",
            Self::AllComponentsDeclareTier => "all-components-declare-tier",
        }
    }
}

impl OperatingMode {
    /// All modes, in increasing order of actuation authority.
    pub const ALL: [Self; 7] = [
        Self::OfflineReplay,
        Self::SimTrain,
        Self::SimEval,
        Self::HwShadow,
        Self::HwSupported,
        Self::HwFreestandingNoncontact,
        Self::HwContact,
    ];

    /// The canonical identifier used in records and on the command line.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::OfflineReplay => "offline-replay",
            Self::SimTrain => "sim-train",
            Self::SimEval => "sim-eval",
            Self::HwShadow => "hw-shadow",
            Self::HwSupported => "hw-supported",
            Self::HwFreestandingNoncontact => "hw-freestanding-noncontact",
            Self::HwContact => "hw-contact",
        }
    }

    /// Parses a mode identifier.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.id() == text)
    }

    /// True when the mode attaches to physical hardware at all.
    #[must_use]
    pub const fn is_hardware(self) -> bool {
        matches!(
            self,
            Self::HwShadow | Self::HwSupported | Self::HwFreestandingNoncontact | Self::HwContact
        )
    }

    /// The mode's actuation authority.
    #[must_use]
    pub const fn authority(self) -> ActuationAuthority {
        match self {
            Self::OfflineReplay => ActuationAuthority::None,
            Self::SimTrain | Self::SimEval => ActuationAuthority::Simulated,
            Self::HwShadow => ActuationAuthority::ComputedAndDiscarded,
            Self::HwSupported => ActuationAuthority::SupportedMotion,
            Self::HwFreestandingNoncontact => ActuationAuthority::FreeMotionNoncontact,
            Self::HwContact => ActuationAuthority::ContactPermitted,
        }
    }

    /// The interlocks this mode requires.
    #[must_use]
    pub fn required_interlocks(self) -> &'static [Interlock] {
        match self {
            Self::OfflineReplay | Self::SimTrain | Self::SimEval => &[],
            Self::HwShadow => &[Interlock::EmergencyStop, Interlock::TransmitDisableAsserted],
            Self::HwSupported => &[
                Interlock::EmergencyStop,
                Interlock::SupportFixture,
                Interlock::ReducedEnvelope,
            ],
            Self::HwFreestandingNoncontact => &[
                Interlock::EmergencyStop,
                Interlock::AreaInterlock,
                Interlock::IndependentWatchdog,
            ],
            Self::HwContact => &[
                Interlock::EmergencyStop,
                Interlock::AreaInterlock,
                Interlock::IndependentWatchdog,
                Interlock::ContactPermission,
                Interlock::IndependentMatchLogger,
            ],
        }
    }

    /// The preconditions that must be satisfied to enter this mode.
    #[must_use]
    pub fn entry_preconditions(self) -> &'static [EntryPrecondition] {
        match self {
            Self::OfflineReplay => &[
                EntryPrecondition::ArtifactHashesResolve,
                EntryPrecondition::BundleSchemaMatchesRecords,
            ],
            Self::SimTrain => &[EntryPrecondition::RunInputsPinned],
            Self::SimEval => &[EntryPrecondition::FrozenBundleDisjointSeeds],
            Self::HwShadow => &[
                EntryPrecondition::TransmitDisableVerified,
                EntryPrecondition::AllComponentsDeclareTier,
            ],
            Self::HwSupported => &[
                EntryPrecondition::JointConventionsVerified,
                EntryPrecondition::ReducedEnvelopeLoaded,
                EntryPrecondition::AllComponentsDeclareTier,
            ],
            Self::HwFreestandingNoncontact => &[
                EntryPrecondition::SupportedQualificationPassed,
                EntryPrecondition::AreaClearedAndEnforced,
                EntryPrecondition::AllComponentsDeclareTier,
            ],
            Self::HwContact => &[
                EntryPrecondition::ContactSeparatelyEnabled,
                EntryPrecondition::NoncontactQualificationPassed,
                EntryPrecondition::ScoringValidated,
                EntryPrecondition::ContactEnergyBudgetSet,
                EntryPrecondition::AllComponentsDeclareTier,
            ],
        }
    }

    /// What happens on loss of input or a fault.
    #[must_use]
    pub const fn fault_response(self) -> FaultResponse {
        match self {
            Self::OfflineReplay => FaultResponse::AbortWithDiagnostic,
            Self::SimTrain => FaultResponse::FailRun,
            Self::SimEval => FaultResponse::FailSuite,
            Self::HwShadow => FaultResponse::ContinueComputingRecordFault,
            Self::HwSupported => FaultResponse::RobotSpecificHoldOrStop,
            Self::HwFreestandingNoncontact => FaultResponse::ValidatedFallback,
            Self::HwContact => FaultResponse::TerminateBout,
        }
    }

    /// The mode whose qualification this mode builds on, if any.
    ///
    /// This is the progression of SPECS.md Section 4 and PLAN.md Phase 6:
    /// replay, shadow, supported, free-standing, contact.
    #[must_use]
    pub const fn prerequisite(self) -> Option<Self> {
        match self {
            Self::OfflineReplay | Self::SimTrain => None,
            Self::SimEval => Some(Self::SimTrain),
            Self::HwShadow => Some(Self::OfflineReplay),
            Self::HwSupported => Some(Self::HwShadow),
            Self::HwFreestandingNoncontact => Some(Self::HwSupported),
            Self::HwContact => Some(Self::HwFreestandingNoncontact),
        }
    }
}

impl fmt::Display for OperatingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

/// Why a mode could not be entered.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ModeError {
    /// A required precondition was not satisfied.
    PreconditionUnmet {
        /// Target mode.
        mode: OperatingMode,
        /// The unmet precondition.
        precondition: EntryPrecondition,
    },
    /// A required interlock was not armed.
    InterlockNotArmed {
        /// Target mode.
        mode: OperatingMode,
        /// The unarmed interlock.
        interlock: Interlock,
    },
    /// The prerequisite mode had not been qualified.
    PrerequisiteNotQualified {
        /// Target mode.
        mode: OperatingMode,
        /// The mode that must be qualified first.
        prerequisite: OperatingMode,
    },
}

impl fmt::Display for ModeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PreconditionUnmet { mode, precondition } => {
                write!(
                    f,
                    "cannot enter {mode}: precondition {} unmet",
                    precondition.id()
                )
            }
            Self::InterlockNotArmed { mode, interlock } => {
                write!(
                    f,
                    "cannot enter {mode}: interlock {} not armed",
                    interlock.id()
                )
            }
            Self::PrerequisiteNotQualified { mode, prerequisite } => {
                write!(f, "cannot enter {mode}: {prerequisite} is not qualified")
            }
        }
    }
}

impl core::error::Error for ModeError {}

/// The evidence a caller presents when requesting a mode transition.
#[derive(Clone, Default, Debug)]
pub struct ModeEntryEvidence {
    /// Preconditions the caller has satisfied.
    pub satisfied: Vec<EntryPrecondition>,
    /// Interlocks the caller has armed.
    pub armed: Vec<Interlock>,
    /// Modes already qualified for this robot.
    pub qualified: Vec<OperatingMode>,
}

impl ModeEntryEvidence {
    /// Empty evidence, which satisfies only the modes with no requirements.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a satisfied precondition.
    #[must_use]
    pub fn with_precondition(mut self, p: EntryPrecondition) -> Self {
        if !self.satisfied.contains(&p) {
            self.satisfied.push(p);
        }
        self
    }

    /// Records an armed interlock.
    #[must_use]
    pub fn with_interlock(mut self, i: Interlock) -> Self {
        if !self.armed.contains(&i) {
            self.armed.push(i);
        }
        self
    }

    /// Records a qualified mode.
    #[must_use]
    pub fn with_qualified(mut self, m: OperatingMode) -> Self {
        if !self.qualified.contains(&m) {
            self.qualified.push(m);
        }
        self
    }

    /// Checks the evidence against a target mode.
    ///
    /// All three classes are checked, and the *first* failure is returned with
    /// the specific item that was missing, so an operator sees what to fix
    /// rather than "entry refused".
    pub fn check(&self, mode: OperatingMode) -> Result<(), ModeError> {
        if let Some(prereq) = mode.prerequisite() {
            if mode.is_hardware() && !self.qualified.contains(&prereq) {
                return Err(ModeError::PrerequisiteNotQualified {
                    mode,
                    prerequisite: prereq,
                });
            }
        }
        for p in mode.entry_preconditions() {
            if !self.satisfied.contains(p) {
                return Err(ModeError::PreconditionUnmet {
                    mode,
                    precondition: *p,
                });
            }
        }
        for i in mode.required_interlocks() {
            if !self.armed.contains(i) {
                return Err(ModeError::InterlockNotArmed {
                    mode,
                    interlock: *i,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shadow_mode_has_no_actuation_authority() {
        assert_eq!(
            OperatingMode::HwShadow.authority(),
            ActuationAuthority::ComputedAndDiscarded
        );
        assert!(!OperatingMode::HwShadow.authority().moves_hardware());
        assert!(OperatingMode::HwSupported.authority().moves_hardware());
    }

    #[test]
    fn shadow_mode_never_enables_transmit_as_a_fallback() {
        assert_eq!(
            OperatingMode::HwShadow.fault_response(),
            FaultResponse::ContinueComputingRecordFault
        );
    }

    #[test]
    fn every_hardware_mode_requires_an_emergency_stop() {
        for mode in OperatingMode::ALL {
            if mode.is_hardware() {
                assert!(
                    mode.required_interlocks()
                        .contains(&Interlock::EmergencyStop),
                    "{mode} must require an e-stop"
                );
            }
        }
    }

    #[test]
    fn every_hardware_mode_requires_a_declared_tier() {
        for mode in OperatingMode::ALL {
            if mode.is_hardware() {
                assert!(
                    mode.entry_preconditions()
                        .contains(&EntryPrecondition::AllComponentsDeclareTier),
                    "{mode} must require tier declarations"
                );
            }
        }
    }

    #[test]
    fn hardware_progression_is_ordered() {
        assert_eq!(
            OperatingMode::HwContact.prerequisite(),
            Some(OperatingMode::HwFreestandingNoncontact)
        );
        assert_eq!(
            OperatingMode::HwFreestandingNoncontact.prerequisite(),
            Some(OperatingMode::HwSupported)
        );
        assert_eq!(
            OperatingMode::HwSupported.prerequisite(),
            Some(OperatingMode::HwShadow)
        );
    }

    #[test]
    fn entering_contact_without_noncontact_qualification_is_refused() {
        let evidence = ModeEntryEvidence::new()
            .with_precondition(EntryPrecondition::ContactSeparatelyEnabled)
            .with_precondition(EntryPrecondition::ScoringValidated)
            .with_precondition(EntryPrecondition::ContactEnergyBudgetSet)
            .with_precondition(EntryPrecondition::AllComponentsDeclareTier)
            .with_precondition(EntryPrecondition::NoncontactQualificationPassed)
            .with_interlock(Interlock::EmergencyStop)
            .with_interlock(Interlock::AreaInterlock)
            .with_interlock(Interlock::IndependentWatchdog)
            .with_interlock(Interlock::ContactPermission)
            .with_interlock(Interlock::IndependentMatchLogger);
        assert!(matches!(
            evidence.check(OperatingMode::HwContact),
            Err(ModeError::PrerequisiteNotQualified { .. })
        ));
        let ok = evidence.with_qualified(OperatingMode::HwFreestandingNoncontact);
        assert!(ok.check(OperatingMode::HwContact).is_ok());
    }

    #[test]
    fn a_missing_interlock_names_itself() {
        let evidence = ModeEntryEvidence::new()
            .with_qualified(OperatingMode::OfflineReplay)
            .with_precondition(EntryPrecondition::TransmitDisableVerified)
            .with_precondition(EntryPrecondition::AllComponentsDeclareTier)
            .with_interlock(Interlock::EmergencyStop);
        assert_eq!(
            evidence.check(OperatingMode::HwShadow),
            Err(ModeError::InterlockNotArmed {
                mode: OperatingMode::HwShadow,
                interlock: Interlock::TransmitDisableAsserted,
            })
        );
    }

    #[test]
    fn simulation_modes_need_no_interlocks() {
        let evidence =
            ModeEntryEvidence::new().with_precondition(EntryPrecondition::RunInputsPinned);
        assert!(evidence.check(OperatingMode::SimTrain).is_ok());
    }

    #[test]
    fn mode_identifiers_round_trip() {
        for mode in OperatingMode::ALL {
            assert_eq!(OperatingMode::parse(mode.id()), Some(mode));
        }
        assert_eq!(OperatingMode::parse("hw-autonomous"), None);
    }

    #[test]
    fn mode_context_names_the_command_source() {
        let ctx = ModeContext {
            mode: OperatingMode::HwSupported,
            source: ControlSource::Teleoperation,
        };
        assert_eq!(ctx.to_string(), "hw-supported/teleop");
    }
}

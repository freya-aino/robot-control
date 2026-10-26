//! Artifact manifests, lineage and promotion with a last-known-good artifact.
//!
//! SPECS.md Section 13: "Promotion includes compatibility validation, evaluation
//! report and an immediately available last-known-good artifact." The hazard
//! register in Section 12.2 names "Wrong artifact deployed" with the control
//! "Refuse to activate; last-known-good bundle available immediately".
//!
//! [`PromotionLedger`] is that mechanism: it records which artifact is current,
//! keeps the previous one reachable, and can roll back without consulting
//! anything external.

use crate::hash::ContentId;
use crate::toml::{self, Table, TomlError, Value};
use core::fmt;
use robot_types::evidence::EvidenceState;
use robot_types::schema::SchemaVersion;

/// The schema version this crate writes for artifact manifests.
pub const MANIFEST_SCHEMA_VERSION: SchemaVersion = SchemaVersion::new(1, 0);

/// What kind of artifact a manifest describes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ArtifactKind {
    /// A canonical motion record.
    MotionRecord,
    /// A retargeted robot reference motion.
    RobotReference,
    /// A recorded robot or simulator episode.
    Episode,
    /// A deployable policy bundle.
    PolicyBundle,
    /// A training checkpoint.
    Checkpoint,
    /// An evaluation report.
    EvaluationReport,
    /// A robot profile.
    RobotProfile,
    /// An acceptance profile.
    AcceptanceProfile,
    /// A skill manifest.
    SkillManifest,
}

impl ArtifactKind {
    /// All kinds.
    pub const ALL: [Self; 9] = [
        Self::MotionRecord,
        Self::RobotReference,
        Self::Episode,
        Self::PolicyBundle,
        Self::Checkpoint,
        Self::EvaluationReport,
        Self::RobotProfile,
        Self::AcceptanceProfile,
        Self::SkillManifest,
    ];

    /// The canonical identifier used in manifests.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::MotionRecord => "motion-record",
            Self::RobotReference => "robot-reference",
            Self::Episode => "episode",
            Self::PolicyBundle => "policy-bundle",
            Self::Checkpoint => "checkpoint",
            Self::EvaluationReport => "evaluation-report",
            Self::RobotProfile => "robot-profile",
            Self::AcceptanceProfile => "acceptance-profile",
            Self::SkillManifest => "skill-manifest",
        }
    }

    /// Parses an identifier.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.id() == text)
    }

    /// Whether promotion of this kind requires an evaluation report.
    #[must_use]
    pub const fn promotion_requires_report(self) -> bool {
        matches!(self, Self::PolicyBundle)
    }
}

/// Where a run executed, recorded so a result can be reproduced.
///
/// SPECS.md Section 13: "Every run records source revision, compiler/toolchain,
/// the `devenv.lock` revision that supplied it, dependency lock, hardware/driver
/// details, seeds, data hashes … A run whose environment cannot be named is not
/// reproducible, whatever else it recorded."
#[derive(Clone, PartialEq, Debug)]
pub struct Environment {
    /// Source revision of this repository.
    pub source_revision: String,
    /// `devenv.lock` revision, per SPECS.md Section 2.2 item 1.
    pub environment_lock: String,
    /// Rust toolchain identity.
    pub toolchain: String,
    /// Host platform triple.
    pub host: String,
    /// Dependency lock digest.
    pub dependency_lock: String,
}

impl Environment {
    /// Whether every field is populated.
    ///
    /// A run whose environment cannot be named is not reproducible, so this is
    /// checked rather than assumed.
    #[must_use]
    pub fn is_named(&self) -> bool {
        ![
            &self.source_revision,
            &self.environment_lock,
            &self.toolchain,
            &self.host,
            &self.dependency_lock,
        ]
        .iter()
        .any(|f| f.trim().is_empty())
    }
}

/// The inputs an artifact was derived from.
#[derive(Clone, Default, PartialEq, Debug)]
pub struct Lineage {
    /// Content identities of the artifacts this one was derived from.
    pub inputs: Vec<ContentId>,
    /// Identity of the configuration used.
    pub config: Option<ContentId>,
    /// Identifier of the run that produced the artifact.
    pub run_id: String,
    /// Named held-out sets excluded from the inputs.
    ///
    /// SPECS.md Section 9.7.2 item 3 requires held-out exclusion to be "tested,
    /// not asserted"; naming them here is what the test checks against.
    pub held_out: Vec<String>,
}

impl Lineage {
    /// Whether `candidate` is among this artifact's inputs, transitively
    /// through `resolve`.
    ///
    /// Used to test held-out exclusion: a corpus that reaches an evaluation
    /// episode through two hops is still reaching it.
    pub fn depends_on(
        &self,
        candidate: &ContentId,
        resolve: &dyn Fn(&ContentId) -> Option<Lineage>,
        max_depth: usize,
    ) -> bool {
        if max_depth == 0 {
            return false;
        }
        if self.inputs.contains(candidate) {
            return true;
        }
        self.inputs.iter().any(|input| {
            resolve(input).is_some_and(|l| l.depends_on(candidate, resolve, max_depth - 1))
        })
    }
}

/// A manifest describing one artifact.
#[derive(Clone, PartialEq, Debug)]
pub struct ArtifactManifest {
    /// Schema version of the manifest.
    pub schema_version: SchemaVersion,
    /// What the artifact is.
    pub kind: ArtifactKind,
    /// Identity of the artifact's content blob.
    pub content: ContentId,
    /// Size of the blob in bytes.
    pub bytes: u64,
    /// Schema version of the *content*, which differs from the manifest's own.
    pub content_schema_version: SchemaVersion,
    /// Where it came from.
    pub lineage: Lineage,
    /// Where it was produced.
    pub environment: Environment,
    /// Evidence state of the claims this artifact carries.
    pub evidence: EvidenceState,
    /// Free-form, namespaced annotations preserved across a round trip.
    pub annotations: Vec<(String, String)>,
}

/// Why a manifest was rejected.
#[derive(Clone, PartialEq, Debug)]
pub enum ManifestError {
    /// A TOML-level problem.
    Toml(TomlError),
    /// A content identity did not parse.
    BadContentId {
        /// The field.
        field: &'static str,
        /// The text.
        text: String,
    },
    /// An artifact kind was not recognized.
    UnknownKind {
        /// The identifier found.
        found: String,
    },
    /// An evidence state was not recognized.
    UnknownEvidence {
        /// The identifier found.
        found: String,
    },
    /// The environment was not fully named.
    UnnamedEnvironment,
    /// A manifest version this build cannot read.
    IncompatibleSchema {
        /// Version this build reads.
        reader: SchemaVersion,
        /// Version found.
        found: SchemaVersion,
    },
    /// An unknown core field was present.
    UnknownField {
        /// The field name.
        field: String,
    },
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Toml(e) => write!(f, "{e}"),
            Self::BadContentId { field, text } => {
                write!(f, "field {field:?} is not a content identity: {text:?}")
            }
            Self::UnknownKind { found } => write!(f, "unknown artifact kind {found:?}"),
            Self::UnknownEvidence { found } => write!(f, "unknown evidence state {found:?}"),
            Self::UnnamedEnvironment => write!(
                f,
                "environment is not fully named; such a run is not reproducible"
            ),
            Self::IncompatibleSchema { reader, found } => {
                write!(f, "manifest version {found}, this build reads {reader}")
            }
            Self::UnknownField { field } => write!(f, "unknown manifest field {field:?}"),
        }
    }
}

impl core::error::Error for ManifestError {}

impl From<TomlError> for ManifestError {
    fn from(e: TomlError) -> Self {
        Self::Toml(e)
    }
}

const KNOWN_FIELDS: &[&str] = &[
    "schema_version",
    "kind",
    "content",
    "bytes",
    "content_schema_version",
    "evidence",
    "lineage",
    "environment",
    "annotations",
];

impl ArtifactManifest {
    /// Serializes to the manifest TOML subset.
    #[must_use]
    pub fn to_toml(&self) -> String {
        let mut root = Table::new();
        root.insert(
            "schema_version",
            Value::String(self.schema_version.to_string()),
        );
        root.insert("kind", Value::String(self.kind.id().to_owned()));
        root.insert("content", Value::String(self.content.to_string()));
        root.insert("bytes", Value::Integer(self.bytes as i64));
        root.insert(
            "content_schema_version",
            Value::String(self.content_schema_version.to_string()),
        );
        root.insert("evidence", Value::String(self.evidence.id().to_owned()));

        let mut lineage = Table::new();
        lineage.insert(
            "inputs",
            Value::Array(
                self.lineage
                    .inputs
                    .iter()
                    .map(|i| Value::String(i.to_string()))
                    .collect(),
            ),
        );
        if let Some(cfg) = &self.lineage.config {
            lineage.insert("config", Value::String(cfg.to_string()));
        }
        lineage.insert("run_id", Value::String(self.lineage.run_id.clone()));
        lineage.insert(
            "held_out",
            Value::Array(
                self.lineage
                    .held_out
                    .iter()
                    .map(|h| Value::String(h.clone()))
                    .collect(),
            ),
        );
        root.insert("lineage", Value::Table(lineage));

        let mut env = Table::new();
        env.insert(
            "source_revision",
            Value::String(self.environment.source_revision.clone()),
        );
        env.insert(
            "environment_lock",
            Value::String(self.environment.environment_lock.clone()),
        );
        env.insert(
            "toolchain",
            Value::String(self.environment.toolchain.clone()),
        );
        env.insert("host", Value::String(self.environment.host.clone()));
        env.insert(
            "dependency_lock",
            Value::String(self.environment.dependency_lock.clone()),
        );
        root.insert("environment", Value::Table(env));

        if !self.annotations.is_empty() {
            let mut annotations = Table::new();
            for (k, v) in &self.annotations {
                annotations.insert(k.clone(), Value::String(v.clone()));
            }
            root.insert("annotations", Value::Table(annotations));
        }

        toml::to_string(&root)
    }

    /// Parses a manifest, rejecting unknown core fields.
    pub fn from_toml(text: &str) -> Result<Self, ManifestError> {
        let root = toml::parse(text)?;

        let unknown = root.unknown_keys(KNOWN_FIELDS);
        if let Some(field) = unknown.first() {
            return Err(ManifestError::UnknownField {
                field: field.clone(),
            });
        }

        let schema_version =
            SchemaVersion::parse(root.require_str("schema_version")?).map_err(|_| {
                ManifestError::Toml(TomlError::WrongType {
                    key: "schema_version".to_owned(),
                    expected: "MAJOR.MINOR",
                    found: "string",
                })
            })?;
        if !MANIFEST_SCHEMA_VERSION.can_read(schema_version) {
            return Err(ManifestError::IncompatibleSchema {
                reader: MANIFEST_SCHEMA_VERSION,
                found: schema_version,
            });
        }

        let kind_text = root.require_str("kind")?;
        let kind = ArtifactKind::parse(kind_text).ok_or_else(|| ManifestError::UnknownKind {
            found: kind_text.to_owned(),
        })?;

        let content_text = root.require_str("content")?;
        let content = ContentId::parse(content_text).map_err(|_| ManifestError::BadContentId {
            field: "content",
            text: content_text.to_owned(),
        })?;

        let bytes = u64::try_from(root.require_integer("bytes")?).map_err(|_| {
            ManifestError::Toml(TomlError::WrongType {
                key: "bytes".to_owned(),
                expected: "non-negative integer",
                found: "integer",
            })
        })?;

        let content_schema_version =
            SchemaVersion::parse(root.require_str("content_schema_version")?).map_err(|_| {
                ManifestError::Toml(TomlError::WrongType {
                    key: "content_schema_version".to_owned(),
                    expected: "MAJOR.MINOR",
                    found: "string",
                })
            })?;

        let evidence_text = root.require_str("evidence")?;
        let evidence =
            EvidenceState::parse(evidence_text).ok_or_else(|| ManifestError::UnknownEvidence {
                found: evidence_text.to_owned(),
            })?;

        let lineage_table = root.require_table("lineage")?;
        let mut inputs = Vec::new();
        for item in lineage_table.require_array("inputs")? {
            let text = item.as_str().ok_or_else(|| {
                ManifestError::Toml(TomlError::WrongType {
                    key: "lineage.inputs".to_owned(),
                    expected: "string",
                    found: item.type_name(),
                })
            })?;
            inputs.push(
                ContentId::parse(text).map_err(|_| ManifestError::BadContentId {
                    field: "lineage.inputs",
                    text: text.to_owned(),
                })?,
            );
        }
        let config = match lineage_table.get("config") {
            Some(v) => {
                let text = v.as_str().ok_or_else(|| {
                    ManifestError::Toml(TomlError::WrongType {
                        key: "lineage.config".to_owned(),
                        expected: "string",
                        found: v.type_name(),
                    })
                })?;
                Some(
                    ContentId::parse(text).map_err(|_| ManifestError::BadContentId {
                        field: "lineage.config",
                        text: text.to_owned(),
                    })?,
                )
            }
            None => None,
        };
        let held_out = lineage_table
            .get("held_out")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|i| i.as_str().map(str::to_owned))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let env_table = root.require_table("environment")?;
        let environment = Environment {
            source_revision: env_table.require_str("source_revision")?.to_owned(),
            environment_lock: env_table.require_str("environment_lock")?.to_owned(),
            toolchain: env_table.require_str("toolchain")?.to_owned(),
            host: env_table.require_str("host")?.to_owned(),
            dependency_lock: env_table.require_str("dependency_lock")?.to_owned(),
        };
        if !environment.is_named() {
            return Err(ManifestError::UnnamedEnvironment);
        }

        let annotations = root
            .get("annotations")
            .and_then(Value::as_table)
            .map(|t| {
                t.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.to_owned(), s.to_owned())))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        Ok(Self {
            schema_version,
            kind,
            content,
            bytes,
            content_schema_version,
            lineage: Lineage {
                inputs,
                config,
                run_id: lineage_table.require_str("run_id")?.to_owned(),
                held_out,
            },
            environment,
            evidence,
            annotations,
        })
    }

    /// The manifest's own identity, over its serialized form.
    #[must_use]
    pub fn manifest_id(&self) -> ContentId {
        ContentId::of(self.to_toml().as_bytes())
    }
}

/// Why a promotion was refused.
#[derive(Clone, PartialEq, Debug)]
pub enum PromotionError {
    /// The artifact kind requires an evaluation report and none was supplied.
    MissingEvaluationReport {
        /// The artifact.
        artifact: ContentId,
    },
    /// The artifact's evidence state is below the requirement for its mode.
    InsufficientEvidence {
        /// State carried.
        found: EvidenceState,
        /// State required.
        required: EvidenceState,
    },
    /// Compatibility validation was not recorded as passing.
    CompatibilityNotValidated,
    /// There is no previous artifact to roll back to.
    NothingToRollBackTo,
    /// The environment of the producing run was not fully named.
    UnnamedEnvironment,
}

impl fmt::Display for PromotionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingEvaluationReport { artifact } => {
                write!(f, "promotion of {artifact} requires an evaluation report")
            }
            Self::InsufficientEvidence { found, required } => {
                write!(f, "artifact carries {found}, promotion requires {required}")
            }
            Self::CompatibilityNotValidated => {
                write!(
                    f,
                    "compatibility validation has not been recorded as passing"
                )
            }
            Self::NothingToRollBackTo => write!(f, "no previous artifact to roll back to"),
            Self::UnnamedEnvironment => {
                write!(f, "producing run has an unnamed environment")
            }
        }
    }
}

impl core::error::Error for PromotionError {}

/// One promotion entry.
#[derive(Clone, PartialEq, Debug)]
pub struct Promotion {
    /// The promoted artifact.
    pub artifact: ContentId,
    /// Its manifest.
    pub manifest: ContentId,
    /// The evaluation report that justified it.
    pub evaluation_report: Option<ContentId>,
    /// Evidence state at promotion.
    pub evidence: EvidenceState,
    /// Free-form note, such as the operator who approved it.
    pub note: String,
}

/// A record of which artifact is current and which is last known good.
///
/// Rollback is a local operation over this ledger: the previous artifact's
/// identity is already here, so recovering does not require re-running anything.
#[derive(Clone, Default, Debug)]
pub struct PromotionLedger {
    history: Vec<Promotion>,
}

impl PromotionLedger {
    /// An empty ledger.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The currently promoted artifact.
    #[must_use]
    pub fn current(&self) -> Option<&Promotion> {
        self.history.last()
    }

    /// The artifact promoted before the current one.
    ///
    /// This is the "immediately available last-known-good artifact" of SPECS.md
    /// Section 13: available because it is a lookup, not a rebuild.
    #[must_use]
    pub fn last_known_good(&self) -> Option<&Promotion> {
        if self.history.len() < 2 {
            None
        } else {
            self.history.get(self.history.len() - 2)
        }
    }

    /// Every promotion, oldest first.
    #[must_use]
    pub fn history(&self) -> &[Promotion] {
        &self.history
    }

    /// Promotes an artifact.
    ///
    /// `compatibility_validated` is the caller's statement that the bundle was
    /// checked against the host profile and envelope. It is a required argument
    /// so that promoting without the check is a visible choice at the call site
    /// rather than an omission.
    pub fn promote(
        &mut self,
        manifest: &ArtifactManifest,
        manifest_id: ContentId,
        evaluation_report: Option<ContentId>,
        compatibility_validated: bool,
        required_evidence: EvidenceState,
        note: impl Into<String>,
    ) -> Result<(), PromotionError> {
        if !manifest.environment.is_named() {
            return Err(PromotionError::UnnamedEnvironment);
        }
        if !compatibility_validated {
            return Err(PromotionError::CompatibilityNotValidated);
        }
        if manifest.kind.promotion_requires_report() && evaluation_report.is_none() {
            return Err(PromotionError::MissingEvaluationReport {
                artifact: manifest.content.clone(),
            });
        }
        if manifest.evidence < required_evidence {
            return Err(PromotionError::InsufficientEvidence {
                found: manifest.evidence,
                required: required_evidence,
            });
        }
        self.history.push(Promotion {
            artifact: manifest.content.clone(),
            manifest: manifest_id,
            evaluation_report,
            evidence: manifest.evidence,
            note: note.into(),
        });
        Ok(())
    }

    /// Rolls back to the last known good artifact, returning it.
    ///
    /// The rolled-back-from entry is removed, so a second rollback goes back
    /// another step rather than oscillating.
    pub fn roll_back(&mut self) -> Result<&Promotion, PromotionError> {
        if self.history.len() < 2 {
            return Err(PromotionError::NothingToRollBackTo);
        }
        self.history.pop();
        Ok(self.history.last().expect("checked length"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn environment() -> Environment {
        Environment {
            source_revision: "git:d67a672".to_owned(),
            environment_lock: "devenv.lock@abc123".to_owned(),
            toolchain: "rustc 1.85.0".to_owned(),
            host: "x86_64-unknown-linux-gnu".to_owned(),
            dependency_lock: "sha256:emptyclosure".to_owned(),
        }
    }

    fn manifest(kind: ArtifactKind, evidence: EvidenceState) -> ArtifactManifest {
        ArtifactManifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            kind,
            content: ContentId::of(b"artifact bytes"),
            bytes: 14,
            content_schema_version: SchemaVersion::new(1, 0),
            lineage: Lineage {
                inputs: vec![ContentId::of(b"input-a"), ContentId::of(b"input-b")],
                config: Some(ContentId::of(b"config")),
                run_id: "run-001".to_owned(),
                held_out: vec!["session-9".to_owned()],
            },
            environment: environment(),
            evidence,
            annotations: vec![("operator".to_owned(), "freya".to_owned())],
        }
    }

    #[test]
    fn manifests_round_trip_through_toml() {
        let m = manifest(ArtifactKind::PolicyBundle, EvidenceState::ReproducedInSim);
        let text = m.to_toml();
        let back = ArtifactManifest::from_toml(&text).unwrap();
        assert_eq!(back, m, "round trip changed the manifest:\n{text}");
        assert_eq!(m.manifest_id(), back.manifest_id());
    }

    #[test]
    fn unknown_core_fields_are_rejected() {
        // Prepended, so the key lands in the root table rather than inside the
        // last sub-table the writer emitted.
        let text = format!(
            "surprise = 1\n{}",
            manifest(ArtifactKind::Episode, EvidenceState::Unverified).to_toml()
        );
        assert!(matches!(
            ArtifactManifest::from_toml(&text),
            Err(ManifestError::UnknownField { .. })
        ));
    }

    #[test]
    fn an_unnamed_environment_is_rejected() {
        let mut m = manifest(ArtifactKind::Episode, EvidenceState::Unverified);
        m.environment.environment_lock = String::new();
        assert!(!m.environment.is_named());
        assert!(matches!(
            ArtifactManifest::from_toml(&m.to_toml()),
            Err(ManifestError::UnnamedEnvironment)
        ));
    }

    #[test]
    fn unknown_kinds_and_states_are_rejected() {
        let text = manifest(ArtifactKind::Episode, EvidenceState::Unverified)
            .to_toml()
            .replace("\"episode\"", "\"something-else\"");
        assert!(matches!(
            ArtifactManifest::from_toml(&text),
            Err(ManifestError::UnknownKind { .. })
        ));

        let text = manifest(ArtifactKind::Episode, EvidenceState::Unverified)
            .to_toml()
            .replace("\"unverified\"", "\"cross_sim_agreement\"");
        assert!(matches!(
            ArtifactManifest::from_toml(&text),
            Err(ManifestError::UnknownEvidence { .. })
        ));
    }

    #[test]
    fn a_malformed_content_identity_is_rejected() {
        let text = manifest(ArtifactKind::Episode, EvidenceState::Unverified)
            .to_toml()
            .replace("content = \"sha256:", "content = \"md5:");
        assert!(matches!(
            ArtifactManifest::from_toml(&text),
            Err(ManifestError::BadContentId {
                field: "content",
                ..
            })
        ));
    }

    #[test]
    fn a_future_manifest_version_is_refused() {
        let text = manifest(ArtifactKind::Episode, EvidenceState::Unverified)
            .to_toml()
            .replace("schema_version = \"1.0\"", "schema_version = \"1.9\"");
        assert!(matches!(
            ArtifactManifest::from_toml(&text),
            Err(ManifestError::IncompatibleSchema { .. })
        ));
    }

    #[test]
    fn a_bundle_cannot_be_promoted_without_a_report() {
        let mut ledger = PromotionLedger::new();
        let m = manifest(ArtifactKind::PolicyBundle, EvidenceState::ReproducedInSim);
        assert!(matches!(
            ledger.promote(
                &m,
                m.manifest_id(),
                None,
                true,
                EvidenceState::ReproducedInSim,
                "first"
            ),
            Err(PromotionError::MissingEvaluationReport { .. })
        ));
    }

    #[test]
    fn promotion_requires_compatibility_validation() {
        let mut ledger = PromotionLedger::new();
        let m = manifest(ArtifactKind::PolicyBundle, EvidenceState::ReproducedInSim);
        assert!(matches!(
            ledger.promote(
                &m,
                m.manifest_id(),
                Some(ContentId::of(b"report")),
                false,
                EvidenceState::ReproducedInSim,
                "first"
            ),
            Err(PromotionError::CompatibilityNotValidated)
        ));
    }

    #[test]
    fn promotion_to_hardware_requires_hardware_evidence() {
        let mut ledger = PromotionLedger::new();
        let m = manifest(ArtifactKind::PolicyBundle, EvidenceState::ReproducedInSim);
        assert!(matches!(
            ledger.promote(
                &m,
                m.manifest_id(),
                Some(ContentId::of(b"report")),
                true,
                EvidenceState::ValidatedOnHardware,
                "to hardware"
            ),
            Err(PromotionError::InsufficientEvidence { .. })
        ));
    }

    #[test]
    fn rollback_returns_the_previous_artifact() {
        let mut ledger = PromotionLedger::new();
        assert!(ledger.current().is_none());
        assert!(ledger.last_known_good().is_none());
        assert!(matches!(
            ledger.roll_back(),
            Err(PromotionError::NothingToRollBackTo)
        ));

        let mut first = manifest(ArtifactKind::PolicyBundle, EvidenceState::ReproducedInSim);
        first.content = ContentId::of(b"bundle-v1");
        ledger
            .promote(
                &first,
                first.manifest_id(),
                Some(ContentId::of(b"report-1")),
                true,
                EvidenceState::ReproducedInSim,
                "v1",
            )
            .unwrap();
        assert_eq!(
            ledger.current().unwrap().artifact,
            ContentId::of(b"bundle-v1")
        );
        assert!(ledger.last_known_good().is_none());

        let mut second = manifest(ArtifactKind::PolicyBundle, EvidenceState::ReproducedInSim);
        second.content = ContentId::of(b"bundle-v2");
        ledger
            .promote(
                &second,
                second.manifest_id(),
                Some(ContentId::of(b"report-2")),
                true,
                EvidenceState::ReproducedInSim,
                "v2",
            )
            .unwrap();
        assert_eq!(
            ledger.current().unwrap().artifact,
            ContentId::of(b"bundle-v2")
        );
        assert_eq!(
            ledger.last_known_good().unwrap().artifact,
            ContentId::of(b"bundle-v1")
        );

        let restored = ledger.roll_back().unwrap().artifact.clone();
        assert_eq!(restored, ContentId::of(b"bundle-v1"));
        assert_eq!(ledger.history().len(), 1);
    }

    #[test]
    fn transitive_lineage_finds_a_held_out_episode_two_hops_away() {
        let evaluation = ContentId::of(b"evaluation-episode");
        let intermediate = ContentId::of(b"intermediate");
        let intermediate_lineage = Lineage {
            inputs: vec![evaluation.clone()],
            config: None,
            run_id: "run-mid".to_owned(),
            held_out: Vec::new(),
        };
        let top = Lineage {
            inputs: vec![intermediate.clone()],
            config: None,
            run_id: "run-top".to_owned(),
            held_out: Vec::new(),
        };
        let resolve = |id: &ContentId| -> Option<Lineage> {
            if *id == intermediate {
                Some(intermediate_lineage.clone())
            } else {
                None
            }
        };
        assert!(top.depends_on(&evaluation, &resolve, 8));
        // A depth bound of one only sees direct inputs.
        assert!(!top.depends_on(&evaluation, &resolve, 1));
        assert!(!top.depends_on(&ContentId::of(b"unrelated"), &resolve, 8));
    }

    #[test]
    fn only_bundles_require_a_report() {
        assert!(ArtifactKind::PolicyBundle.promotion_requires_report());
        assert!(!ArtifactKind::Episode.promotion_requires_report());
        for kind in ArtifactKind::ALL {
            assert_eq!(ArtifactKind::parse(kind.id()), Some(kind));
        }
        assert_eq!(ArtifactKind::parse("nonsense"), None);
    }
}

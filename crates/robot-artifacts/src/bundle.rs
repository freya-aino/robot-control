//! The `robot-bundle/1` directory format: a deployable policy bundle on
//! disk.
//!
//! SPECS.md Section 7.4 defines the bundle's contents and
//! `robot_types::bundle::PolicyBundleManifest` is their in-memory form; this
//! module is their on-disk form (ADR 0016), the missing piece `robot eval
//! --bundle` and `robot deploy` named. The layout is:
//!
//! ```text
//! <dir>/manifest.toml      this format, digest-sealed
//! <dir>/weights/<name>     one opaque blob per learned component
//! ```
//!
//! The weight blobs are opaque here: their own format (for example
//! `robot-mlp/1`, `robot_nn::format`) is the component's concern, and each
//! blob is named, sized and content-addressed by its `ComponentManifest`, so
//! it is verified *before* a byte of it is interpreted (SPECS.md Section 13).
//!
//! The manifest's `digest` seals the manifest itself: it is the
//! [`digest_parts`](crate::hash::digest_parts) of the manifest serialized
//! with an empty digest and no signature. [`read_bundle`] recomputes and
//! compares it, verifies every blob's size and content identity, and then
//! runs no activation check of its own — activation is
//! `PolicyBundleManifest::check_activatable`, which needs the host envelope
//! and stays with the caller.
//!
//! ## What this format refuses, by name
//!
//! - A component with `decodes_latent_space`: no latent decoder exists in
//!   this workspace, so v1 does not define its encoding.
//! - `RecurrentStatePolicy::TransformedBy` and `ActionSemantics::Latent`:
//!   both carry compile-time artifact identities with the same problem.
//! - A `FieldSpec` unit outside [`UNIT_SYMBOLS`]: units are compile-time
//!   symbols (`&'static str`), so reading interns against this fixed table
//!   rather than leaking arbitrary strings.

use crate::hash::{digest_parts, ContentId};
use crate::store::{write_atomic, StoreError};
use crate::toml::{self, Table, TomlError, Value};
use robot_types::bundle::{
    BenchmarkResult, ComponentManifest, DataLineage, PolicyBundleManifest, RecurrentStatePolicy,
    RecurrentStateSpec, SafetyEnvelopeAssumptions, TransitionContract,
};
use robot_types::evidence::EvidenceState;
use robot_types::ids::Name;
use robot_types::mode::OperatingMode;
use robot_types::obs::{ActionSemantics, ActionSpec, Availability, FieldSpec, ObservationSpec};
use robot_types::schema::SchemaVersion;
use robot_types::tier::{
    MeasuredCompute, MissingUpperTierBehavior, Tier, TierAssignment, TierSchedule,
};
use robot_types::time::Duration;
use robot_types::units::Hertz;
use std::path::Path;

/// The format identity this module writes and the only one it reads.
pub const FORMAT: &str = "robot-bundle/1";

/// The unit symbols a bundle's observation fields may use. Units are
/// compile-time symbols in `robot_types::obs::FieldSpec`; reading interns
/// against this table and rejects anything else by name.
pub const UNIT_SYMBOLS: &[&str] = &[
    "1", "rad", "rad/s", "rad/s^2", "m", "m/s", "m/s^2", "N", "N m", "s", "Hz", "kg", "bool",
];

/// Why a bundle could not be written or read.
#[derive(Clone, PartialEq, Debug)]
pub enum BundleFileError {
    /// The TOML itself was rejected.
    Toml(TomlError),
    /// The `format` key is missing or names another format.
    Format(String),
    /// A construct `robot-bundle/1` does not encode.
    Unsupported(String),
    /// A field failed validation.
    Invalid(String),
    /// The manifest digest did not match its contents.
    DigestMismatch {
        /// Digest the manifest states.
        stated: String,
        /// Digest of the manifest as read.
        computed: String,
    },
    /// A weight blob is missing, has the wrong size, or the wrong identity.
    Blob {
        /// The component.
        component: String,
        /// What was wrong.
        what: String,
    },
    /// An I/O or atomic-write failure.
    Store(String),
}

impl core::fmt::Display for BundleFileError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Toml(e) => write!(f, "{e}"),
            Self::Format(found) => write!(f, "not a {FORMAT} file (format = {found:?})"),
            Self::Unsupported(what) => write!(f, "{FORMAT} does not encode {what}"),
            Self::Invalid(what) => write!(f, "{what}"),
            Self::DigestMismatch { stated, computed } => {
                write!(f, "manifest digest {stated} but contents digest {computed}")
            }
            Self::Blob { component, what } => write!(f, "weights of {component}: {what}"),
            Self::Store(e) => write!(f, "{e}"),
        }
    }
}

impl core::error::Error for BundleFileError {}

impl From<TomlError> for BundleFileError {
    fn from(e: TomlError) -> Self {
        Self::Toml(e)
    }
}

impl From<StoreError> for BundleFileError {
    fn from(e: StoreError) -> Self {
        Self::Store(e.to_string())
    }
}

impl From<std::io::Error> for BundleFileError {
    fn from(e: std::io::Error) -> Self {
        Self::Store(e.to_string())
    }
}

type Result<T> = core::result::Result<T, BundleFileError>;

fn floats(v: &[f64]) -> Value {
    Value::Array(v.iter().map(|x| Value::Float(*x)).collect())
}

fn strings(v: &[String]) -> Value {
    Value::Array(v.iter().map(|s| Value::String(s.clone())).collect())
}

/// Serializes a manifest. `digest` and `signature` are written as stored;
/// [`seal`] computes the digest.
pub fn to_toml(m: &PolicyBundleManifest) -> Result<String> {
    let mut root = Table::new();
    root.insert("format", Value::String(FORMAT.to_owned()));
    root.insert(
        "schema_version",
        Value::String(m.schema_version.to_string()),
    );
    root.insert("bundle_id", Value::String(m.bundle_id.clone()));
    root.insert("rate_hz", Value::Float(m.rate.0));
    root.insert("normalizer_mean", floats(&m.normalizer_mean));
    root.insert("normalizer_variance", floats(&m.normalizer_variance));
    root.insert(
        "normalizer_count",
        Value::Integer(m.normalizer_count as i64),
    );
    root.insert("normalizer_clip", Value::Float(m.normalizer_clip));

    let mut obs = Table::new();
    obs.insert(
        "schema_version",
        Value::String(m.observation.schema_version.to_string()),
    );
    let mut fields = Vec::new();
    for f in m.observation.fields() {
        let mut t = Table::new();
        t.insert("name", Value::String(f.name.as_str().to_owned()));
        t.insert("width", Value::Integer(f.width as i64));
        t.insert("unit", Value::String(f.unit.to_owned()));
        t.insert(
            "availability",
            Value::String(
                match f.availability {
                    Availability::Deployable => "deployable",
                    Availability::Privileged => "privileged",
                }
                .to_owned(),
            ),
        );
        fields.push(Value::Table(t));
    }
    obs.insert("field", Value::Array(fields));
    root.insert("observation", Value::Table(obs));

    let mut action = Table::new();
    action.insert(
        "schema_version",
        Value::String(m.action.schema_version.to_string()),
    );
    let semantics = match m.action.semantics {
        ActionSemantics::JointPositionTarget => "joint-position-target",
        ActionSemantics::JointVelocityTarget => "joint-velocity-target",
        ActionSemantics::JointTorque => "joint-torque",
        ActionSemantics::JointPositionResidual => "joint-position-residual",
        ActionSemantics::Latent { .. } => {
            return Err(BundleFileError::Unsupported(
                "ActionSemantics::Latent (no latent decoder exists in this workspace)".to_owned(),
            ))
        }
    };
    action.insert("semantics", Value::String(semantics.to_owned()));
    action.insert("width", Value::Integer(m.action.width as i64));
    action.insert("scale", floats(&m.action.scale));
    action.insert("offset", floats(&m.action.offset));
    action.insert("output_limit", floats(&m.action.output_limit));
    root.insert("action", Value::Table(action));

    let mut rec = Table::new();
    match &m.recurrent {
        RecurrentStateSpec::Stateless => {
            rec.insert("kind", Value::String("stateless".to_owned()));
        }
        RecurrentStateSpec::Recurrent {
            width,
            on_activation,
        } => {
            rec.insert("kind", Value::String("recurrent".to_owned()));
            rec.insert("width", Value::Integer(*width as i64));
            let policy = match on_activation {
                RecurrentStatePolicy::ResetToZero => "reset-to-zero",
                RecurrentStatePolicy::CarryFromCompatiblePredecessor => {
                    "carry-from-compatible-predecessor"
                }
                RecurrentStatePolicy::TransformedBy { .. } => {
                    return Err(BundleFileError::Unsupported(
                        "RecurrentStatePolicy::TransformedBy".to_owned(),
                    ))
                }
            };
            rec.insert("on_activation", Value::String(policy.to_owned()));
        }
    }
    root.insert("recurrent", Value::Table(rec));

    let mut tiers = Vec::new();
    for a in m.tiers.assignments() {
        let mut t = Table::new();
        t.insert("component", Value::String(a.component.clone()));
        t.insert("tier", Value::String(a.tier.id().to_owned()));
        t.insert("rate_hz", Value::Float(a.rate.0));
        t.insert("budget_nanos", Value::Integer(a.budget.nanos()));
        t.insert(
            "measured_percentile",
            Value::Float(a.measured_compute.percentile),
        );
        t.insert(
            "measured_nanos",
            Value::Integer(a.measured_compute.duration.nanos()),
        );
        t.insert(
            "measured_max_nanos",
            Value::Integer(a.measured_compute.max.nanos()),
        );
        t.insert(
            "measured_run_id",
            Value::String(a.measured_compute.run_id.clone()),
        );
        t.insert(
            "input_freshness_nanos",
            Value::Integer(a.input_freshness_limit.nanos()),
        );
        t.insert(
            "on_missing_upper",
            Value::String(a.on_missing_upper.id().to_owned()),
        );
        tiers.push(Value::Table(t));
    }
    root.insert("tier", Value::Array(tiers));

    let mut components = Vec::new();
    for c in &m.components {
        if c.decodes_latent_space.is_some() {
            return Err(BundleFileError::Unsupported(format!(
                "component {} decodes a latent space (no latent decoder exists in this workspace)",
                c.name.as_str()
            )));
        }
        let mut t = Table::new();
        t.insert("name", Value::String(c.name.as_str().to_owned()));
        t.insert(
            "architecture_version",
            Value::String(c.architecture_version.to_string()),
        );
        t.insert("weights_digest", Value::String(c.weights_digest.clone()));
        t.insert("weights_bytes", Value::Integer(c.weights_bytes as i64));
        components.push(Value::Table(t));
    }
    root.insert("component", Value::Array(components));

    let mut env = Table::new();
    env.insert(
        "robot_profile",
        Value::String(m.envelope.robot_profile.clone()),
    );
    env.insert(
        "safety_profile",
        Value::String(m.envelope.safety_profile.clone()),
    );
    env.insert(
        "position_limits_lo",
        floats(
            &m.envelope
                .position_limits
                .iter()
                .map(|(lo, _)| *lo)
                .collect::<Vec<_>>(),
        ),
    );
    env.insert(
        "position_limits_hi",
        floats(
            &m.envelope
                .position_limits
                .iter()
                .map(|(_, hi)| *hi)
                .collect::<Vec<_>>(),
        ),
    );
    env.insert("velocity_limits", floats(&m.envelope.velocity_limits));
    env.insert("torque_limits", floats(&m.envelope.torque_limits));
    env.insert(
        "qualified_modes",
        Value::Array(
            m.envelope
                .qualified_modes
                .iter()
                .map(|q| Value::String(q.id().to_owned()))
                .collect(),
        ),
    );
    root.insert("envelope", Value::Table(env));

    let mut lineage = Table::new();
    lineage.insert("inputs", strings(&m.lineage.inputs));
    lineage.insert("config_hash", Value::String(m.lineage.config_hash.clone()));
    lineage.insert(
        "environment_lock",
        Value::String(m.lineage.environment_lock.clone()),
    );
    lineage.insert(
        "source_revision",
        Value::String(m.lineage.source_revision.clone()),
    );
    lineage.insert("held_out", strings(&m.lineage.held_out));
    root.insert("lineage", Value::Table(lineage));

    let mut transitions = Vec::new();
    for c in &m.transitions {
        let mut t = Table::new();
        t.insert("target", Value::String(c.target.clone()));
        t.insert("guard", Value::String(c.guard.clone()));
        if let Some(d) = c.minimum_dwell {
            t.insert("minimum_dwell_nanos", Value::Integer(d.nanos()));
        }
        t.insert(
            "preemptible_by_recovery",
            Value::Boolean(c.preemptible_by_recovery),
        );
        t.insert("tested_by", Value::String(c.tested_by.clone()));
        transitions.push(Value::Table(t));
    }
    root.insert("transition", Value::Array(transitions));

    let mut benchmarks = Vec::new();
    for b in &m.benchmarks {
        let mut t = Table::new();
        t.insert("metric", Value::String(b.metric.clone()));
        t.insert("value", Value::Float(b.value));
        t.insert("unit", Value::String(b.unit.clone()));
        t.insert(
            "acceptance_profile",
            Value::String(b.acceptance_profile.clone()),
        );
        t.insert("evidence", Value::String(b.evidence.id().to_owned()));
        t.insert("run_id", Value::String(b.run_id.clone()));
        benchmarks.push(Value::Table(t));
    }
    root.insert("benchmark", Value::Array(benchmarks));

    root.insert("digest", Value::String(m.digest.clone()));
    if let Some(sig) = &m.signature {
        root.insert("signature", Value::String(sig.clone()));
    }
    Ok(toml::to_string(&root))
}

fn version_of(text: &str, what: &str) -> Result<SchemaVersion> {
    SchemaVersion::parse(text)
        .map_err(|e| BundleFileError::Invalid(format!("{what} {text:?}: {e:?}")))
}

fn name_of(text: &str, what: &str) -> Result<Name> {
    Name::new(text).map_err(|e| BundleFileError::Invalid(format!("{what} {text:?}: {e:?}")))
}

fn duration_of(v: i64, what: &str) -> Result<Duration> {
    if v < 0 {
        return Err(BundleFileError::Invalid(format!("negative {what}")));
    }
    Ok(Duration::from_nanos(v))
}

fn float_list(t: &Table, key: &str) -> Result<Vec<f64>> {
    t.require_array(key)?
        .iter()
        .map(|v| {
            v.as_float()
                .ok_or_else(|| BundleFileError::Invalid(format!("{key} holds a non-number")))
        })
        .collect()
}

fn string_list(t: &Table, key: &str) -> Result<Vec<String>> {
    t.require_array(key)?
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .ok_or_else(|| BundleFileError::Invalid(format!("{key} holds a non-string")))
        })
        .collect()
}

fn tables_of<'a>(root: &'a Table, key: &str) -> Result<Vec<&'a Table>> {
    match root.get(key) {
        None => Ok(Vec::new()),
        Some(v) => v
            .as_array()
            .ok_or_else(|| BundleFileError::Invalid(format!("{key} is not an array of tables")))?
            .iter()
            .map(|v| match v {
                Value::Table(t) => Ok(t),
                _ => Err(BundleFileError::Invalid(format!("{key} holds a non-table"))),
            })
            .collect(),
    }
}

/// Deserializes a manifest. The digest is read as stored, not verified;
/// [`read_bundle`] verifies it.
pub fn from_toml(text: &str) -> Result<PolicyBundleManifest> {
    let root = toml::parse(text)?;
    let format = root.require_str("format")?;
    if format != FORMAT {
        return Err(BundleFileError::Format(format.to_owned()));
    }

    let obs_table = root.require_table("observation")?;
    let mut fields = Vec::new();
    for t in tables_of(obs_table, "field")? {
        let unit_text = t.require_str("unit")?;
        let unit = UNIT_SYMBOLS
            .iter()
            .find(|u| **u == unit_text)
            .copied()
            .ok_or_else(|| {
                BundleFileError::Unsupported(format!(
                    "unit {unit_text:?} (known: {UNIT_SYMBOLS:?})"
                ))
            })?;
        let width = usize::try_from(t.require_integer("width")?)
            .map_err(|_| BundleFileError::Invalid("negative field width".to_owned()))?;
        let name = t.require_str("name")?;
        let field = match t.require_str("availability")? {
            "deployable" => FieldSpec::deployable(name, width, unit),
            "privileged" => FieldSpec::privileged(name, width, unit),
            other => {
                return Err(BundleFileError::Invalid(format!(
                    "availability {other:?} on field {name}"
                )))
            }
        }
        .map_err(|e| BundleFileError::Invalid(format!("field {name}: {e:?}")))?;
        fields.push(field);
    }
    let observation = ObservationSpec::new(
        version_of(
            obs_table.require_str("schema_version")?,
            "observation schema",
        )?,
        fields,
    )
    .map_err(|e| BundleFileError::Invalid(format!("observation: {e:?}")))?;

    let action_table = root.require_table("action")?;
    let semantics = match action_table.require_str("semantics")? {
        "joint-position-target" => ActionSemantics::JointPositionTarget,
        "joint-velocity-target" => ActionSemantics::JointVelocityTarget,
        "joint-torque" => ActionSemantics::JointTorque,
        "joint-position-residual" => ActionSemantics::JointPositionResidual,
        other => {
            return Err(BundleFileError::Unsupported(format!(
                "action semantics {other:?}"
            )))
        }
    };
    let action = ActionSpec::new(
        version_of(action_table.require_str("schema_version")?, "action schema")?,
        semantics,
        float_list(action_table, "scale")?,
        float_list(action_table, "offset")?,
        float_list(action_table, "output_limit")?,
    )
    .map_err(|e| BundleFileError::Invalid(format!("action: {e:?}")))?;
    let stated_width = usize::try_from(action_table.require_integer("width")?)
        .map_err(|_| BundleFileError::Invalid("negative action width".to_owned()))?;
    if action.width != stated_width {
        return Err(BundleFileError::Invalid(format!(
            "action width {} disagrees with its arrays ({})",
            stated_width, action.width
        )));
    }

    let rec_table = root.require_table("recurrent")?;
    let recurrent = match rec_table.require_str("kind")? {
        "stateless" => RecurrentStateSpec::Stateless,
        "recurrent" => RecurrentStateSpec::Recurrent {
            width: usize::try_from(rec_table.require_integer("width")?)
                .map_err(|_| BundleFileError::Invalid("negative recurrent width".to_owned()))?,
            on_activation: match rec_table.require_str("on_activation")? {
                "reset-to-zero" => RecurrentStatePolicy::ResetToZero,
                "carry-from-compatible-predecessor" => {
                    RecurrentStatePolicy::CarryFromCompatiblePredecessor
                }
                other => {
                    return Err(BundleFileError::Unsupported(format!(
                        "recurrent on_activation {other:?}"
                    )))
                }
            },
        },
        other => {
            return Err(BundleFileError::Invalid(format!(
                "recurrent kind {other:?}"
            )))
        }
    };

    let mut tiers = TierSchedule::new();
    for t in tables_of(&root, "tier")? {
        let tier_text = t.require_str("tier")?;
        tiers.declare(TierAssignment {
            component: t.require_str("component")?.to_owned(),
            tier: Tier::parse(tier_text)
                .ok_or_else(|| BundleFileError::Invalid(format!("unknown tier {tier_text:?}")))?,
            rate: Hertz(t.require_float("rate_hz")?),
            budget: duration_of(t.require_integer("budget_nanos")?, "budget_nanos")?,
            measured_compute: MeasuredCompute {
                percentile: t.require_float("measured_percentile")?,
                duration: duration_of(t.require_integer("measured_nanos")?, "measured_nanos")?,
                max: duration_of(
                    t.require_integer("measured_max_nanos")?,
                    "measured_max_nanos",
                )?,
                run_id: t.require_str("measured_run_id")?.to_owned(),
            },
            input_freshness_limit: duration_of(
                t.require_integer("input_freshness_nanos")?,
                "input_freshness_nanos",
            )?,
            on_missing_upper: match t.require_str("on_missing_upper")? {
                "hold-until-expiry" => MissingUpperTierBehavior::HoldUntilExpiry,
                "use-declared-default" => MissingUpperTierBehavior::UseDeclaredDefault,
                "invoke-fallback" => MissingUpperTierBehavior::InvokeFallback,
                other => {
                    return Err(BundleFileError::Invalid(format!(
                        "unknown on_missing_upper {other:?}"
                    )))
                }
            },
        });
    }

    let mut components = Vec::new();
    for t in tables_of(&root, "component")? {
        let bytes = t.require_integer("weights_bytes")?;
        if bytes < 0 {
            return Err(BundleFileError::Invalid(
                "negative weights_bytes".to_owned(),
            ));
        }
        components.push(ComponentManifest {
            name: name_of(t.require_str("name")?, "component name")?,
            architecture_version: version_of(
                t.require_str("architecture_version")?,
                "architecture_version",
            )?,
            weights_digest: t.require_str("weights_digest")?.to_owned(),
            weights_bytes: bytes as u64,
            decodes_latent_space: None,
        });
    }

    let env_table = root.require_table("envelope")?;
    let lo = float_list(env_table, "position_limits_lo")?;
    let hi = float_list(env_table, "position_limits_hi")?;
    if lo.len() != hi.len() {
        return Err(BundleFileError::Invalid(
            "position_limits_lo and _hi lengths disagree".to_owned(),
        ));
    }
    let envelope = SafetyEnvelopeAssumptions {
        robot_profile: env_table.require_str("robot_profile")?.to_owned(),
        safety_profile: env_table.require_str("safety_profile")?.to_owned(),
        position_limits: lo.into_iter().zip(hi).collect(),
        velocity_limits: float_list(env_table, "velocity_limits")?,
        torque_limits: float_list(env_table, "torque_limits")?,
        qualified_modes: string_list(env_table, "qualified_modes")?
            .iter()
            .map(|s| {
                OperatingMode::parse(s).ok_or_else(|| {
                    BundleFileError::Invalid(format!("unknown operating mode {s:?}"))
                })
            })
            .collect::<Result<_>>()?,
    };

    let lin_table = root.require_table("lineage")?;
    let lineage = DataLineage {
        inputs: string_list(lin_table, "inputs")?,
        config_hash: lin_table.require_str("config_hash")?.to_owned(),
        environment_lock: lin_table.require_str("environment_lock")?.to_owned(),
        source_revision: lin_table.require_str("source_revision")?.to_owned(),
        held_out: string_list(lin_table, "held_out")?,
    };

    let mut transitions = Vec::new();
    for t in tables_of(&root, "transition")? {
        transitions.push(TransitionContract {
            target: t.require_str("target")?.to_owned(),
            guard: t.require_str("guard")?.to_owned(),
            minimum_dwell: match t.get("minimum_dwell_nanos") {
                None => None,
                Some(Value::Integer(n)) => Some(duration_of(*n, "minimum_dwell_nanos")?),
                Some(_) => {
                    return Err(BundleFileError::Invalid(
                        "minimum_dwell_nanos is not an integer".to_owned(),
                    ))
                }
            },
            preemptible_by_recovery: t.require_bool("preemptible_by_recovery")?,
            tested_by: t.require_str("tested_by")?.to_owned(),
        });
    }

    let mut benchmarks = Vec::new();
    for t in tables_of(&root, "benchmark")? {
        let evidence_text = t.require_str("evidence")?;
        benchmarks.push(BenchmarkResult {
            metric: t.require_str("metric")?.to_owned(),
            value: t.require_float("value")?,
            unit: t.require_str("unit")?.to_owned(),
            acceptance_profile: t.require_str("acceptance_profile")?.to_owned(),
            evidence: EvidenceState::parse(evidence_text).ok_or_else(|| {
                BundleFileError::Invalid(format!("unknown evidence state {evidence_text:?}"))
            })?,
            run_id: t.require_str("run_id")?.to_owned(),
        });
    }

    let count = root.require_integer("normalizer_count")?;
    if count < 0 {
        return Err(BundleFileError::Invalid(
            "negative normalizer_count".to_owned(),
        ));
    }
    Ok(PolicyBundleManifest {
        schema_version: version_of(root.require_str("schema_version")?, "schema_version")?,
        bundle_id: root.require_str("bundle_id")?.to_owned(),
        observation,
        normalizer_mean: float_list(&root, "normalizer_mean")?,
        normalizer_variance: float_list(&root, "normalizer_variance")?,
        normalizer_count: count as u64,
        normalizer_clip: root.require_float("normalizer_clip")?,
        action,
        recurrent,
        rate: Hertz(root.require_float("rate_hz")?),
        tiers,
        components,
        envelope,
        lineage,
        transitions,
        benchmarks,
        digest: root.require_str("digest")?.to_owned(),
        signature: match root.get("signature") {
            None => None,
            Some(v) => Some(
                v.as_str()
                    .ok_or_else(|| {
                        BundleFileError::Invalid("signature is not a string".to_owned())
                    })?
                    .to_owned(),
            ),
        },
    })
}

/// The digest that seals a manifest: the manifest serialized with an empty
/// digest and no signature.
pub fn manifest_digest(m: &PolicyBundleManifest) -> Result<ContentId> {
    let mut unsealed = m.clone();
    unsealed.digest = String::new();
    unsealed.signature = None;
    Ok(digest_parts(&[to_toml(&unsealed)?.as_bytes()]))
}

/// Writes `<dir>/manifest.toml` and `<dir>/weights/<name>` atomically, after
/// checking every component against its blob and sealing the digest.
///
/// `blobs` must carry exactly the manifest's components, by name; each blob's
/// size and content identity must equal the component's declaration, so a
/// manifest can never be written sealed over blobs it does not describe.
pub fn write_bundle(
    dir: &Path,
    manifest: &PolicyBundleManifest,
    blobs: &[(&str, &[u8])],
) -> Result<ContentId> {
    for c in &manifest.components {
        let blob = blobs
            .iter()
            .find(|(n, _)| *n == c.name.as_str())
            .map(|(_, b)| *b)
            .ok_or_else(|| BundleFileError::Blob {
                component: c.name.as_str().to_owned(),
                what: "no blob supplied".to_owned(),
            })?;
        if blob.len() as u64 != c.weights_bytes {
            return Err(BundleFileError::Blob {
                component: c.name.as_str().to_owned(),
                what: format!("{} bytes, manifest says {}", blob.len(), c.weights_bytes),
            });
        }
        let id = ContentId::of(blob).to_string();
        if id != c.weights_digest {
            return Err(BundleFileError::Blob {
                component: c.name.as_str().to_owned(),
                what: format!("content {id}, manifest says {}", c.weights_digest),
            });
        }
    }
    for (name, _) in blobs {
        if !manifest.components.iter().any(|c| c.name.as_str() == *name) {
            return Err(BundleFileError::Blob {
                component: (*name).to_owned(),
                what: "blob supplied for no component".to_owned(),
            });
        }
    }
    let digest = manifest_digest(manifest)?;
    let mut sealed = manifest.clone();
    sealed.digest = digest.to_string();
    std::fs::create_dir_all(dir.join("weights"))?;
    for (name, blob) in blobs {
        write_atomic(&dir.join("weights").join(name), blob)?;
    }
    write_atomic(&dir.join("manifest.toml"), to_toml(&sealed)?.as_bytes())?;
    Ok(digest)
}

/// Reads and verifies a bundle: the manifest parses, its digest matches its
/// contents, and every component's blob exists with the declared size and
/// content identity. Activation is the caller's
/// `PolicyBundleManifest::check_activatable`.
pub fn read_bundle(dir: &Path) -> Result<PolicyBundleManifest> {
    let manifest = from_toml(&std::fs::read_to_string(dir.join("manifest.toml"))?)?;
    let computed = manifest_digest(&manifest)?.to_string();
    if manifest.digest != computed {
        return Err(BundleFileError::DigestMismatch {
            stated: manifest.digest.clone(),
            computed,
        });
    }
    for c in &manifest.components {
        let path = dir.join("weights").join(c.name.as_str());
        let meta = std::fs::metadata(&path).map_err(|e| BundleFileError::Blob {
            component: c.name.as_str().to_owned(),
            what: e.to_string(),
        })?;
        // Size first, so an oversized blob is refused before it is read.
        if meta.len() != c.weights_bytes {
            return Err(BundleFileError::Blob {
                component: c.name.as_str().to_owned(),
                what: format!(
                    "{} bytes on disk, manifest says {}",
                    meta.len(),
                    c.weights_bytes
                ),
            });
        }
        let bytes = std::fs::read(&path)?;
        let id = ContentId::of(&bytes).to_string();
        if id != c.weights_digest {
            return Err(BundleFileError::Blob {
                component: c.name.as_str().to_owned(),
                what: format!("content {id}, manifest says {}", c.weights_digest),
            });
        }
    }
    Ok(manifest)
}

/// Reads one component's verified weight blob.
pub fn load_weights(dir: &Path, component: &str) -> Result<Vec<u8>> {
    let manifest = read_bundle(dir)?;
    let c = manifest
        .components
        .iter()
        .find(|c| c.name.as_str() == component)
        .ok_or_else(|| BundleFileError::Blob {
            component: component.to_owned(),
            what: "not in the manifest".to_owned(),
        })?;
    let bytes = std::fs::read(dir.join("weights").join(c.name.as_str()))?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_types::bundle::BUNDLE_SCHEMA_VERSION;

    fn weights() -> Vec<u8> {
        (0u16..2048).flat_map(u16::to_le_bytes).collect()
    }

    fn manifest() -> PolicyBundleManifest {
        let blob = weights();
        let obs = ObservationSpec::new(
            SchemaVersion::new(1, 0),
            vec![
                FieldSpec::deployable("joint_position", 2, "rad").unwrap(),
                FieldSpec::privileged("friction", 1, "1").unwrap(),
            ],
        )
        .unwrap();
        let mut tiers = TierSchedule::new();
        tiers.declare(TierAssignment {
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
        PolicyBundleManifest {
            schema_version: BUNDLE_SCHEMA_VERSION,
            bundle_id: "tracker/1".to_owned(),
            observation: obs,
            normalizer_mean: vec![0.0; 3],
            normalizer_variance: vec![1.0; 3],
            normalizer_count: 1000,
            normalizer_clip: 5.0,
            action: ActionSpec::identity(
                SchemaVersion::new(1, 0),
                ActionSemantics::JointPositionTarget,
                2,
                1.0,
            )
            .unwrap(),
            recurrent: RecurrentStateSpec::Recurrent {
                width: 8,
                on_activation: RecurrentStatePolicy::ResetToZero,
            },
            rate: Hertz(100.0),
            tiers,
            components: vec![ComponentManifest {
                name: Name::new("actor").unwrap(),
                architecture_version: SchemaVersion::new(1, 0),
                weights_digest: ContentId::of(&blob).to_string(),
                weights_bytes: blob.len() as u64,
                decodes_latent_space: None,
            }],
            envelope: SafetyEnvelopeAssumptions {
                robot_profile: "fixture-arm/1.0".to_owned(),
                safety_profile: "fixture-arm-safety/1.0".to_owned(),
                position_limits: vec![(-1.0, 1.0), (-1.0, 1.0)],
                velocity_limits: vec![2.0, 2.0],
                torque_limits: vec![10.0, 10.0],
                qualified_modes: vec![OperatingMode::SimEval],
            },
            lineage: DataLineage {
                inputs: vec!["sha256:motion".to_owned()],
                config_hash: "sha256:cfg".to_owned(),
                environment_lock: "devenv.lock@abc123".to_owned(),
                source_revision: "git:d67a672".to_owned(),
                held_out: vec!["session-9".to_owned()],
            },
            transitions: vec![TransitionContract {
                target: "guard".to_owned(),
                guard: "double-support".to_owned(),
                minimum_dwell: Some(Duration::from_millis(200)),
                preemptible_by_recovery: true,
                tested_by: "reg-3".to_owned(),
            }],
            benchmarks: vec![BenchmarkResult {
                metric: "tracking_error".to_owned(),
                value: 0.012,
                unit: "rad".to_owned(),
                acceptance_profile: "profile/1".to_owned(),
                evidence: EvidenceState::ReproducedInSim,
                run_id: "run-1".to_owned(),
            }],
            digest: String::new(),
            signature: None,
        }
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("robot-bundle-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn a_manifest_round_trips_exactly_through_toml() {
        let m = manifest();
        let text = to_toml(&m).unwrap();
        assert!(text.starts_with("format = \"robot-bundle/1\""));
        let back = from_toml(&text).unwrap();
        assert_eq!(back, m);
        assert_eq!(to_toml(&back).unwrap(), text);
    }

    #[test]
    fn write_then_read_verifies_digest_and_blobs() {
        let dir = tmp("roundtrip");
        let blob = weights();
        let digest = write_bundle(&dir, &manifest(), &[("actor", &blob)]).unwrap();
        let back = read_bundle(&dir).unwrap();
        assert_eq!(back.digest, digest.to_string());
        assert_eq!(back.bundle_id, "tracker/1");
        assert_eq!(load_weights(&dir, "actor").unwrap(), blob);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_tampered_manifest_or_blob_is_refused() {
        let dir = tmp("tamper");
        let blob = weights();
        write_bundle(&dir, &manifest(), &[("actor", &blob)]).unwrap();

        // Flip a byte of the weights: the content identity no longer matches.
        let path = dir.join("weights").join("actor");
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[0] ^= 1;
        std::fs::write(&path, &bytes).unwrap();
        assert!(matches!(
            read_bundle(&dir),
            Err(BundleFileError::Blob { .. })
        ));
        bytes[0] ^= 1;
        std::fs::write(&path, &bytes).unwrap();
        assert!(read_bundle(&dir).is_ok());

        // Change a manifest field: the sealed digest no longer matches.
        let mpath = dir.join("manifest.toml");
        let text = std::fs::read_to_string(&mpath)
            .unwrap()
            .replace("bundle_id = \"tracker/1\"", "bundle_id = \"other/1\"");
        std::fs::write(&mpath, text).unwrap();
        assert!(matches!(
            read_bundle(&dir),
            Err(BundleFileError::DigestMismatch { .. })
        ));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn mismatched_blobs_cannot_be_written() {
        let dir = tmp("mismatch");
        let blob = weights();
        let short = &blob[..16];
        assert!(matches!(
            write_bundle(&dir, &manifest(), &[("actor", short)]),
            Err(BundleFileError::Blob { .. })
        ));
        assert!(matches!(
            write_bundle(&dir, &manifest(), &[]),
            Err(BundleFileError::Blob { .. })
        ));
        assert!(!dir.exists(), "a refused bundle must write nothing");
    }

    #[test]
    fn latent_constructs_and_unknown_units_are_refused_by_name() {
        let mut m = manifest();
        m.components[0].decodes_latent_space = Some(robot_types::transition::LatentSpaceId {
            name: "s".to_owned(),
            version: SchemaVersion::new(1, 0),
            decoder_artifact: "sha256:d".to_owned(),
            dimension: 4,
            constraint: robot_types::transition::LatentConstraint::UnitHypersphere,
        });
        assert!(matches!(to_toml(&m), Err(BundleFileError::Unsupported(_))));
        let text = to_toml(&manifest())
            .unwrap()
            .replace("unit = \"rad\"", "unit = \"furlong\"");
        assert!(matches!(
            from_toml(&text),
            Err(BundleFileError::Unsupported(_))
        ));
    }
}

//! Observation and action contracts, and the normalization both training and
//! deployment share.
//!
//! SPECS.md Section 6: "Training and deployment share observation/action schemas
//! and tested normalization logic." PLAN.md Phase 4 asks for "identical Rust
//! normalization/action transforms for training and deployment", so the
//! transform lives *here*, in the schema crate, and both the learner and the
//! runtime call this one implementation. A second copy is the defect this
//! placement exists to prevent.
//!
//! Section 9.2 additionally requires an audit "proving no privileged inputs
//! remain" in an exported actor. That is [`ObservationSpec::audit_deployable`],
//! and it works because each field declares its [`Availability`] rather than
//! having it inferred from a name.

use crate::ids::{Name, NameError};
use crate::schema::{SchemaError, SchemaVersion};
use core::fmt;

/// Where an observation field's value can come from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Availability {
    /// Measurable on the robot at inference time.
    Deployable,
    /// Available only inside the simulator.
    ///
    /// Permitted in an asymmetric critic (SPECS.md Section 9.2) and forbidden in
    /// an exported actor.
    Privileged,
}

/// A named, dimensioned block of an observation vector.
#[derive(Clone, PartialEq, Debug)]
pub struct FieldSpec {
    /// Field name, stable across schema minor versions.
    pub name: Name,
    /// Number of scalars this field occupies.
    pub width: usize,
    /// Canonical unit symbol, for the record and the report.
    pub unit: &'static str,
    /// Where the value comes from.
    pub availability: Availability,
}

impl FieldSpec {
    /// A deployable field.
    pub fn deployable(name: &str, width: usize, unit: &'static str) -> Result<Self, NameError> {
        Ok(Self {
            name: Name::new(name)?,
            width,
            unit,
            availability: Availability::Deployable,
        })
    }

    /// A privileged, simulator-only field.
    pub fn privileged(name: &str, width: usize, unit: &'static str) -> Result<Self, NameError> {
        Ok(Self {
            name: Name::new(name)?,
            width,
            unit,
            availability: Availability::Privileged,
        })
    }
}

/// The layout of an observation vector.
#[derive(Clone, PartialEq, Debug)]
pub struct ObservationSpec {
    /// Schema version of this layout.
    pub schema_version: SchemaVersion,
    fields: Vec<FieldSpec>,
}

/// Why an observation or action vector was rejected.
#[derive(Clone, PartialEq, Debug)]
pub enum VectorError {
    /// The vector length did not match the spec.
    WidthMismatch {
        /// Width the spec declares.
        expected: usize,
        /// Width supplied.
        found: usize,
    },
    /// A component was NaN or infinite.
    ///
    /// SPECS.md Section 12.1 requires the supervisor to check "finite commands";
    /// the same check applies on the way in, because a NaN observation produces
    /// a NaN action that is harder to attribute.
    NotFinite {
        /// Index of the offending component.
        index: usize,
    },
    /// An exported actor declared a privileged input.
    PrivilegedInputInActor {
        /// The offending field.
        field: String,
    },
    /// Two fields claimed the same name.
    DuplicateField {
        /// The repeated name.
        field: String,
    },
    /// A field declared zero width.
    ZeroWidthField {
        /// The offending field.
        field: String,
    },
}

impl fmt::Display for VectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WidthMismatch { expected, found } => {
                write!(f, "expected width {expected}, found {found}")
            }
            Self::NotFinite { index } => write!(f, "component {index} is not finite"),
            Self::PrivilegedInputInActor { field } => write!(
                f,
                "exported actor reads privileged field {field:?}; \
                 Section 9.2 requires a deployable observation set"
            ),
            Self::DuplicateField { field } => write!(f, "duplicate field name {field:?}"),
            Self::ZeroWidthField { field } => write!(f, "field {field:?} has zero width"),
        }
    }
}

impl core::error::Error for VectorError {}

impl ObservationSpec {
    /// Builds a spec, rejecting duplicate names and zero-width fields.
    pub fn new(schema_version: SchemaVersion, fields: Vec<FieldSpec>) -> Result<Self, VectorError> {
        for (i, f) in fields.iter().enumerate() {
            if f.width == 0 {
                return Err(VectorError::ZeroWidthField {
                    field: f.name.to_string(),
                });
            }
            if fields[..i].iter().any(|g| g.name == f.name) {
                return Err(VectorError::DuplicateField {
                    field: f.name.to_string(),
                });
            }
        }
        Ok(Self {
            schema_version,
            fields,
        })
    }

    /// The declared fields, in vector order.
    #[must_use]
    pub fn fields(&self) -> &[FieldSpec] {
        &self.fields
    }

    /// Total width of the vector.
    #[must_use]
    pub fn width(&self) -> usize {
        self.fields.iter().map(|f| f.width).sum()
    }

    /// Width of the deployable fields only.
    #[must_use]
    pub fn deployable_width(&self) -> usize {
        self.fields
            .iter()
            .filter(|f| f.availability == Availability::Deployable)
            .map(|f| f.width)
            .sum()
    }

    /// The `[start, end)` span of a named field, if present.
    #[must_use]
    pub fn span_of(&self, name: &str) -> Option<(usize, usize)> {
        let mut offset = 0;
        for f in &self.fields {
            if f.name.as_str() == name {
                return Some((offset, offset + f.width));
            }
            offset += f.width;
        }
        None
    }

    /// Checks a vector against this spec.
    pub fn validate(&self, values: &[f64]) -> Result<(), VectorError> {
        let expected = self.width();
        if values.len() != expected {
            return Err(VectorError::WidthMismatch {
                expected,
                found: values.len(),
            });
        }
        if let Some(index) = values.iter().position(|v| !v.is_finite()) {
            return Err(VectorError::NotFinite { index });
        }
        Ok(())
    }

    /// The audit of SPECS.md Section 9.2: an exported actor must read no
    /// privileged field.
    pub fn audit_deployable(&self) -> Result<(), VectorError> {
        match self
            .fields
            .iter()
            .find(|f| f.availability == Availability::Privileged)
        {
            Some(f) => Err(VectorError::PrivilegedInputInActor {
                field: f.name.to_string(),
            }),
            None => Ok(()),
        }
    }

    /// The deployable projection of this spec, which is what an actor exports.
    ///
    /// The critic keeps the full spec; the actor keeps this one. Deriving it
    /// rather than maintaining two specs is what keeps them from diverging.
    pub fn deployable_projection(&self) -> Result<Self, VectorError> {
        Self::new(
            self.schema_version,
            self.fields
                .iter()
                .filter(|f| f.availability == Availability::Deployable)
                .cloned()
                .collect(),
        )
    }

    /// Copies the deployable components of a full observation into `out`.
    pub fn project_deployable(&self, values: &[f64], out: &mut [f64]) -> Result<(), VectorError> {
        self.validate(values)?;
        let expected = self.deployable_width();
        if out.len() != expected {
            return Err(VectorError::WidthMismatch {
                expected,
                found: out.len(),
            });
        }
        let mut src = 0;
        let mut dst = 0;
        for f in &self.fields {
            if f.availability == Availability::Deployable {
                out[dst..dst + f.width].copy_from_slice(&values[src..src + f.width]);
                dst += f.width;
            }
            src += f.width;
        }
        Ok(())
    }
}

/// What an action vector means to the actuator interface.
///
/// SPECS.md Section 7.4 requires a bundle to record "action semantics and
/// scaling"; Section 7.1 requires the profile to declare the "command
/// interface". A mismatch between the two is what
/// [`ActionSpec::check_compatible_with`] exists to catch before motion.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ActionSemantics {
    /// Target joint positions in radians.
    JointPositionTarget,
    /// Target joint velocities in radians per second.
    JointVelocityTarget,
    /// Joint torques in newton-metres.
    JointTorque,
    /// Position targets added to a reference pose, in radians.
    JointPositionResidual,
    /// A latent action interpreted by a named decoder.
    ///
    /// SPECS.md Section 7.6 item 8: a latent action "MUST record the
    /// latent-space identity and version and the decoder artifact required to
    /// interpret it", so the decoder identity travels with the semantics.
    Latent {
        /// Versioned identity of the latent space.
        space_id: &'static str,
    },
}

impl ActionSemantics {
    /// The canonical identifier used in bundles.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::JointPositionTarget => "joint-position-target",
            Self::JointVelocityTarget => "joint-velocity-target",
            Self::JointTorque => "joint-torque",
            Self::JointPositionResidual => "joint-position-residual",
            Self::Latent { .. } => "latent",
        }
    }

    /// True when this action is directly interpretable by an actuator.
    ///
    /// A latent action is not: it requires its decoder, and a runtime that
    /// cannot load that decoder must refuse to activate rather than send the
    /// latent vector to a motor.
    #[must_use]
    pub const fn is_actuator_ready(self) -> bool {
        !matches!(self, Self::Latent { .. })
    }
}

/// The layout and semantics of an action vector.
#[derive(Clone, PartialEq, Debug)]
pub struct ActionSpec {
    /// Schema version of this layout.
    pub schema_version: SchemaVersion,
    /// What the components mean.
    pub semantics: ActionSemantics,
    /// Number of components.
    pub width: usize,
    /// Per-component scale applied to the policy output.
    pub scale: Vec<f64>,
    /// Per-component offset added after scaling.
    pub offset: Vec<f64>,
    /// Per-component clamp applied by the *bundle* before the supervisor sees
    /// the command. This is a scaling bound, not a safety control: SPECS.md
    /// Section 12.2 requires controls independent of the learned component, and
    /// the supervisor remains the last stage.
    pub output_limit: Vec<f64>,
}

impl ActionSpec {
    /// Builds a spec, rejecting inconsistent widths and unusable scales.
    pub fn new(
        schema_version: SchemaVersion,
        semantics: ActionSemantics,
        scale: Vec<f64>,
        offset: Vec<f64>,
        output_limit: Vec<f64>,
    ) -> Result<Self, VectorError> {
        let width = scale.len();
        if offset.len() != width {
            return Err(VectorError::WidthMismatch {
                expected: width,
                found: offset.len(),
            });
        }
        if output_limit.len() != width {
            return Err(VectorError::WidthMismatch {
                expected: width,
                found: output_limit.len(),
            });
        }
        if let Some(index) = scale
            .iter()
            .chain(&offset)
            .chain(&output_limit)
            .position(|v| !v.is_finite())
        {
            return Err(VectorError::NotFinite { index });
        }
        if let Some(index) = output_limit.iter().position(|v| *v < 0.0) {
            return Err(VectorError::NotFinite { index });
        }
        Ok(Self {
            schema_version,
            semantics,
            width,
            scale,
            offset,
            output_limit,
        })
    }

    /// An identity transform of the given width, for tests and simple tasks.
    pub fn identity(
        schema_version: SchemaVersion,
        semantics: ActionSemantics,
        width: usize,
        limit: f64,
    ) -> Result<Self, VectorError> {
        Self::new(
            schema_version,
            semantics,
            vec![1.0; width],
            vec![0.0; width],
            vec![limit; width],
        )
    }

    /// Applies scale, offset and the bundle's own clamp.
    ///
    /// This is the transform shared by the learner and the runtime. Both call it
    /// so that an action-parity check (PLAN.md Phase 4) compares inference, not
    /// two transcriptions of the same arithmetic.
    pub fn apply(&self, raw: &[f64], out: &mut [f64]) -> Result<(), VectorError> {
        if raw.len() != self.width {
            return Err(VectorError::WidthMismatch {
                expected: self.width,
                found: raw.len(),
            });
        }
        if out.len() != self.width {
            return Err(VectorError::WidthMismatch {
                expected: self.width,
                found: out.len(),
            });
        }
        if let Some(index) = raw.iter().position(|v| !v.is_finite()) {
            return Err(VectorError::NotFinite { index });
        }
        for i in 0..self.width {
            let v = raw[i] * self.scale[i] + self.offset[i];
            out[i] = v.clamp(-self.output_limit[i], self.output_limit[i]);
        }
        Ok(())
    }

    /// Checks that this action spec matches what a robot profile accepts.
    pub fn check_compatible_with(
        &self,
        profile_semantics: ActionSemantics,
        profile_width: usize,
    ) -> Result<(), SchemaError> {
        if self.semantics != profile_semantics {
            return Err(SchemaError::InvalidValue {
                field: "action_semantics",
                reason: format!(
                    "bundle declares {} but profile accepts {}",
                    self.semantics.id(),
                    profile_semantics.id()
                ),
            });
        }
        if self.width != profile_width {
            return Err(SchemaError::InvalidValue {
                field: "action_width",
                reason: format!(
                    "bundle declares {} components, profile has {profile_width}",
                    self.width
                ),
            });
        }
        Ok(())
    }
}

/// Running mean and variance by Welford's algorithm.
///
/// Exported in a bundle as frozen statistics (SPECS.md Section 7.4) and used
/// unchanged at deployment. The count is part of the state so that resuming a
/// training run continues the same estimate rather than restarting it, which
/// SPECS.md Section 9.2 requires of checkpoint/resume.
#[derive(Clone, PartialEq, Debug)]
pub struct Normalizer {
    mean: Vec<f64>,
    m2: Vec<f64>,
    count: u64,
    clip: f64,
    epsilon: f64,
    frozen: bool,
}

impl Normalizer {
    /// A normalizer over `width` components.
    ///
    /// `clip` bounds the normalized magnitude; SPECS.md Section 9.2 requires
    /// observation normalization but the clip is what keeps one outlier
    /// observation from producing an enormous activation.
    #[must_use]
    pub fn new(width: usize, clip: f64) -> Self {
        Self {
            mean: vec![0.0; width],
            m2: vec![0.0; width],
            count: 0,
            clip,
            epsilon: 1e-8,
            frozen: false,
        }
    }

    /// Reconstructs frozen statistics from a bundle.
    pub fn from_statistics(
        mean: Vec<f64>,
        variance: Vec<f64>,
        count: u64,
        clip: f64,
    ) -> Result<Self, VectorError> {
        if variance.len() != mean.len() {
            return Err(VectorError::WidthMismatch {
                expected: mean.len(),
                found: variance.len(),
            });
        }
        if let Some(index) = mean.iter().chain(&variance).position(|v| !v.is_finite()) {
            return Err(VectorError::NotFinite { index });
        }
        if let Some(index) = variance.iter().position(|v| *v < 0.0) {
            return Err(VectorError::NotFinite { index });
        }
        let denom = count.max(1) as f64;
        Ok(Self {
            m2: variance.iter().map(|v| v * denom).collect(),
            mean,
            count,
            clip,
            epsilon: 1e-8,
            frozen: true,
        })
    }

    /// Number of components.
    #[must_use]
    pub fn width(&self) -> usize {
        self.mean.len()
    }

    /// Samples incorporated so far.
    #[must_use]
    pub const fn count(&self) -> u64 {
        self.count
    }

    /// True when the statistics no longer update.
    #[must_use]
    pub const fn is_frozen(&self) -> bool {
        self.frozen
    }

    /// Freezes the statistics for export.
    pub fn freeze(&mut self) {
        self.frozen = true;
    }

    /// The per-component mean.
    #[must_use]
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// The per-component population variance.
    #[must_use]
    pub fn variance(&self) -> Vec<f64> {
        let denom = self.count.max(1) as f64;
        self.m2.iter().map(|m| m / denom).collect()
    }

    /// Incorporates one sample. A frozen normalizer ignores it, which is the
    /// deployment behaviour: statistics come from training and do not drift on
    /// the robot.
    pub fn observe(&mut self, values: &[f64]) -> Result<(), VectorError> {
        if values.len() != self.mean.len() {
            return Err(VectorError::WidthMismatch {
                expected: self.mean.len(),
                found: values.len(),
            });
        }
        if let Some(index) = values.iter().position(|v| !v.is_finite()) {
            return Err(VectorError::NotFinite { index });
        }
        if self.frozen {
            return Ok(());
        }
        self.count += 1;
        let n = self.count as f64;
        for ((mean, m2), value) in self.mean.iter_mut().zip(&mut self.m2).zip(values) {
            let delta = *value - *mean;
            *mean += delta / n;
            *m2 += delta * (*value - *mean);
        }
        Ok(())
    }

    /// Normalizes into `out`.
    ///
    /// With no samples the transform is the identity rather than a division by
    /// a fabricated variance; a policy evaluated before any observation is seen
    /// should behave as if normalization were absent, not receive zeros.
    pub fn normalize(&self, values: &[f64], out: &mut [f64]) -> Result<(), VectorError> {
        if values.len() != self.mean.len() || out.len() != self.mean.len() {
            return Err(VectorError::WidthMismatch {
                expected: self.mean.len(),
                found: values.len().max(out.len()),
            });
        }
        if let Some(index) = values.iter().position(|v| !v.is_finite()) {
            return Err(VectorError::NotFinite { index });
        }
        if self.count == 0 {
            out.copy_from_slice(values);
            return Ok(());
        }
        let denom = self.count as f64;
        for i in 0..values.len() {
            let var = self.m2[i] / denom;
            let std = (var + self.epsilon).sqrt();
            out[i] = ((values[i] - self.mean[i]) / std).clamp(-self.clip, self.clip);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> ObservationSpec {
        ObservationSpec::new(
            SchemaVersion::new(1, 0),
            vec![
                FieldSpec::deployable("joint_position", 3, "rad").unwrap(),
                FieldSpec::privileged("ground_friction", 1, "1").unwrap(),
                FieldSpec::deployable("imu_gyro", 3, "rad/s").unwrap(),
            ],
        )
        .unwrap()
    }

    #[test]
    fn widths_and_spans_are_derived_from_the_fields() {
        let s = spec();
        assert_eq!(s.width(), 7);
        assert_eq!(s.deployable_width(), 6);
        assert_eq!(s.span_of("ground_friction"), Some((3, 4)));
        assert_eq!(s.span_of("imu_gyro"), Some((4, 7)));
        assert_eq!(s.span_of("absent"), None);
    }

    #[test]
    fn duplicate_and_zero_width_fields_are_rejected() {
        let dup = ObservationSpec::new(
            SchemaVersion::new(1, 0),
            vec![
                FieldSpec::deployable("a", 1, "1").unwrap(),
                FieldSpec::deployable("a", 1, "1").unwrap(),
            ],
        );
        assert!(matches!(dup, Err(VectorError::DuplicateField { .. })));
        let zero = ObservationSpec::new(
            SchemaVersion::new(1, 0),
            vec![FieldSpec::deployable("a", 0, "1").unwrap()],
        );
        assert!(matches!(zero, Err(VectorError::ZeroWidthField { .. })));
    }

    #[test]
    fn privileged_audit_fails_for_an_actor_and_passes_for_its_projection() {
        let s = spec();
        assert!(matches!(
            s.audit_deployable(),
            Err(VectorError::PrivilegedInputInActor { .. })
        ));
        let actor = s.deployable_projection().unwrap();
        assert!(actor.audit_deployable().is_ok());
        assert_eq!(actor.width(), 6);
    }

    #[test]
    fn projection_copies_only_deployable_components() {
        let s = spec();
        let full = [1.0, 2.0, 3.0, 99.0, 4.0, 5.0, 6.0];
        let mut out = [0.0; 6];
        s.project_deployable(&full, &mut out).unwrap();
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut wrong = [0.0; 5];
        assert!(s.project_deployable(&full, &mut wrong).is_err());
    }

    #[test]
    fn nonfinite_observations_are_rejected() {
        let s = spec();
        let mut v = vec![0.0; 7];
        v[4] = f64::NAN;
        assert_eq!(s.validate(&v), Err(VectorError::NotFinite { index: 4 }));
        assert_eq!(
            s.validate(&[0.0; 3]),
            Err(VectorError::WidthMismatch {
                expected: 7,
                found: 3
            })
        );
    }

    #[test]
    fn action_transform_scales_offsets_and_clamps() {
        let a = ActionSpec::new(
            SchemaVersion::new(1, 0),
            ActionSemantics::JointPositionTarget,
            vec![2.0, 0.5],
            vec![0.1, -0.1],
            vec![1.0, 0.2],
        )
        .unwrap();
        let mut out = [0.0; 2];
        a.apply(&[0.3, 1.0], &mut out).unwrap();
        assert!((out[0] - 0.7).abs() < 1e-12);
        assert!(
            (out[1] - 0.2).abs() < 1e-12,
            "second component should clamp: {}",
            out[1]
        );
    }

    #[test]
    fn action_spec_rejects_bad_shapes_and_negative_limits() {
        assert!(ActionSpec::new(
            SchemaVersion::new(1, 0),
            ActionSemantics::JointTorque,
            vec![1.0, 1.0],
            vec![0.0],
            vec![1.0, 1.0],
        )
        .is_err());
        assert!(ActionSpec::new(
            SchemaVersion::new(1, 0),
            ActionSemantics::JointTorque,
            vec![1.0],
            vec![0.0],
            vec![-1.0],
        )
        .is_err());
        let a = ActionSpec::identity(
            SchemaVersion::new(1, 0),
            ActionSemantics::JointTorque,
            2,
            1.0,
        )
        .unwrap();
        let mut out = [0.0; 2];
        assert!(a.apply(&[f64::NAN, 0.0], &mut out).is_err());
    }

    #[test]
    fn a_latent_action_is_not_actuator_ready() {
        let latent = ActionSemantics::Latent {
            space_id: "boxing-latent/1.0",
        };
        assert!(!latent.is_actuator_ready());
        assert!(ActionSemantics::JointTorque.is_actuator_ready());
    }

    #[test]
    fn mismatched_action_semantics_are_refused() {
        let a = ActionSpec::identity(
            SchemaVersion::new(1, 0),
            ActionSemantics::JointTorque,
            12,
            1.0,
        )
        .unwrap();
        assert!(a
            .check_compatible_with(ActionSemantics::JointPositionTarget, 12)
            .is_err());
        assert!(a
            .check_compatible_with(ActionSemantics::JointTorque, 11)
            .is_err());
        assert!(a
            .check_compatible_with(ActionSemantics::JointTorque, 12)
            .is_ok());
    }

    #[test]
    fn normalizer_matches_the_batch_statistics() {
        let mut n = Normalizer::new(1, 10.0);
        let samples = [1.0, 2.0, 3.0, 4.0, 5.0];
        for s in samples {
            n.observe(&[s]).unwrap();
        }
        assert!((n.mean()[0] - 3.0).abs() < 1e-12);
        assert!((n.variance()[0] - 2.0).abs() < 1e-12);
        assert_eq!(n.count(), 5);
    }

    #[test]
    fn normalizer_round_trips_through_frozen_statistics() {
        let mut n = Normalizer::new(2, 5.0);
        for i in 0..50 {
            n.observe(&[f64::from(i), f64::from(i) * -0.5]).unwrap();
        }
        let mut before = [0.0; 2];
        n.normalize(&[10.0, -3.0], &mut before).unwrap();

        let restored =
            Normalizer::from_statistics(n.mean().to_vec(), n.variance(), n.count(), 5.0).unwrap();
        let mut after = [0.0; 2];
        restored.normalize(&[10.0, -3.0], &mut after).unwrap();
        assert!((before[0] - after[0]).abs() < 1e-12);
        assert!((before[1] - after[1]).abs() < 1e-12);
        assert!(restored.is_frozen());
    }

    #[test]
    fn a_frozen_normalizer_does_not_drift_on_the_robot() {
        let mut n = Normalizer::new(1, 5.0);
        n.observe(&[1.0]).unwrap();
        n.observe(&[3.0]).unwrap();
        n.freeze();
        let mean_before = n.mean()[0];
        n.observe(&[1000.0]).unwrap();
        assert_eq!(n.mean()[0], mean_before);
        assert_eq!(n.count(), 2);
    }

    #[test]
    fn normalization_clips_outliers() {
        let mut n = Normalizer::new(1, 2.0);
        for i in 0..100 {
            n.observe(&[f64::from(i % 3)]).unwrap();
        }
        let mut out = [0.0];
        n.normalize(&[1e6], &mut out).unwrap();
        assert_eq!(out[0], 2.0);
    }

    #[test]
    fn an_unobserved_normalizer_is_the_identity() {
        let n = Normalizer::new(3, 5.0);
        let mut out = [0.0; 3];
        n.normalize(&[1.0, -2.0, 3.0], &mut out).unwrap();
        assert_eq!(out, [1.0, -2.0, 3.0]);
    }
}

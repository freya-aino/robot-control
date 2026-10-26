//! Bounded expressive intent: style as sampled configuration over
//! already-validated skills, never generated motion.
//!
//! SPECS.md Section 12.5: "Expression MUST NOT widen any envelope, extend
//! any dwell time against a recovery request, or introduce a motion that
//! has not passed the same validation as a task skill. A celebration is a
//! skill with a transition contract, not an exemption."
//!
//! This is enforced structurally, not by a runtime check that could be
//! bypassed: [`StyleConfig`] has no field that could widen a numeric
//! envelope (no torque, velocity or position limit lives here at all), and
//! [`select_expressive_skill`] can only ever return a [`SkillId`] already
//! present in the caller-supplied allowlist of validated skills — it
//! cannot synthesize one.

use core::fmt;
use robot_skills::descriptor::SkillId;

/// A versioned, sampled style configuration.
///
/// SPECS.md Section 12.5's closing paragraph: candidates for this set "MAY
/// be proposed by the offline assisted sampling of Section 9.5," but "a
/// personality is then a sampled, versioned configuration over
/// already-validated skills... not a generated behavior." Nothing in this
/// struct can call a model or synthesize a new skill; it only names which
/// already-approved skills this persona prefers and how strongly.
#[derive(Clone, Debug)]
pub struct StyleConfig {
    /// Stable name for this style (e.g. `"aggressive-v1"`).
    pub name: String,
    /// Version, bumped whenever the preference weights change — a style is
    /// itself a versioned artifact, not a live-tunable parameter set.
    pub version: u32,
    /// Expressive skills this style may select from, each with a relative
    /// preference weight. Every entry must already exist in the caller's
    /// validated-skill allowlist; this struct does not grant existence.
    pub expressive_preferences: Vec<(SkillId, f64)>,
    /// Tactical preference weights, keyed by a named tactical dimension
    /// (e.g. `"aggression"`, `"patience"`) rather than free-form, so a
    /// style cannot invent an unbounded-effect knob.
    pub tactical_preferences: Vec<(String, f64)>,
}

/// Why an expressive skill could not be selected.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExpressionError {
    /// A recovery is active; expression never takes priority over it.
    RecoveryActive,
    /// The requested skill is not in the validated allowlist.
    NotValidated,
    /// The requested skill is not one this style expresses at all.
    NotInStyle,
}

impl fmt::Display for ExpressionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RecoveryActive => write!(f, "a recovery is active; expression is refused"),
            Self::NotValidated => write!(f, "skill is not in the validated allowlist"),
            Self::NotInStyle => write!(f, "skill is not part of this style"),
        }
    }
}

impl core::error::Error for ExpressionError {}

/// Selects an expressive skill, refusing if a recovery is active or the
/// skill is not both validated and part of this style.
///
/// This function cannot widen an envelope or extend a dwell time: it has
/// no parameter through which either could be expressed. Its only output
/// is a [`SkillId`] already present in both `validated_allowlist` and
/// `style.expressive_preferences` — a value the caller already trusts.
pub fn select_expressive_skill(
    style: &StyleConfig,
    requested: SkillId,
    validated_allowlist: &[SkillId],
    recovery_active: bool,
) -> Result<SkillId, ExpressionError> {
    if recovery_active {
        return Err(ExpressionError::RecoveryActive);
    }
    if !validated_allowlist.contains(&requested) {
        return Err(ExpressionError::NotValidated);
    }
    if !style
        .expressive_preferences
        .iter()
        .any(|(id, _)| *id == requested)
    {
        return Err(ExpressionError::NotInStyle);
    }
    Ok(requested)
}

/// Samples an expressive skill from `style`'s preference weights, via a
/// caller-supplied uniform draw in `[0, 1)` — kept explicit rather than
/// owning an RNG, so the draw is reproducible from whatever seed stream
/// the caller is already tracking.
///
/// Returns `None` if every weight is non-positive or the style has no
/// expressive preferences, rather than falling back to an arbitrary skill.
#[must_use]
pub fn sample_preference(style: &StyleConfig, uniform_draw: f64) -> Option<SkillId> {
    let total: f64 = style
        .expressive_preferences
        .iter()
        .map(|(_, w)| w.max(0.0))
        .sum();
    if total <= 0.0 {
        return None;
    }
    let mut target = uniform_draw.clamp(0.0, 1.0) * total;
    for (id, w) in &style.expressive_preferences {
        let w = w.max(0.0);
        if w <= 0.0 {
            continue;
        }
        target -= w;
        if target <= 0.0 {
            return Some(*id);
        }
    }
    style.expressive_preferences.last().map(|(id, _)| *id)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn style() -> StyleConfig {
        StyleConfig {
            name: "celebratory-v1".to_owned(),
            version: 1,
            expressive_preferences: vec![(SkillId(10), 0.9), (SkillId(11), 0.1)],
            tactical_preferences: vec![("aggression".to_owned(), 0.6)],
        }
    }

    #[test]
    fn a_validated_in_style_skill_is_selected() {
        let allowlist = [SkillId(10), SkillId(11)];
        assert_eq!(
            select_expressive_skill(&style(), SkillId(10), &allowlist, false),
            Ok(SkillId(10))
        );
    }

    #[test]
    fn recovery_always_refuses_expression_regardless_of_validity() {
        let allowlist = [SkillId(10)];
        assert_eq!(
            select_expressive_skill(&style(), SkillId(10), &allowlist, true),
            Err(ExpressionError::RecoveryActive)
        );
    }

    #[test]
    fn an_unvalidated_skill_is_refused_even_if_the_style_lists_it() {
        let allowlist: [SkillId; 0] = [];
        assert_eq!(
            select_expressive_skill(&style(), SkillId(10), &allowlist, false),
            Err(ExpressionError::NotValidated)
        );
    }

    #[test]
    fn a_validated_skill_outside_the_style_is_refused() {
        let allowlist = [SkillId(99)];
        assert_eq!(
            select_expressive_skill(&style(), SkillId(99), &allowlist, false),
            Err(ExpressionError::NotInStyle)
        );
    }

    #[test]
    fn sampling_respects_preference_weights_in_aggregate() {
        let s = style();
        let mut counts = [0u32; 2];
        for i in 0..1000 {
            let draw = (i as f64 + 0.5) / 1000.0; // deterministic stratified sweep
            match sample_preference(&s, draw) {
                Some(SkillId(10)) => counts[0] += 1,
                Some(SkillId(11)) => counts[1] += 1,
                _ => panic!("unexpected sample"),
            }
        }
        // Weight ratio is 9:1; the stratified sweep should land close to that.
        let ratio = f64::from(counts[0]) / f64::from(counts[1]);
        assert!((ratio - 9.0).abs() < 0.5, "ratio {ratio}");
    }

    #[test]
    fn a_style_with_no_positive_weights_samples_nothing() {
        let empty = StyleConfig {
            name: "none".to_owned(),
            version: 1,
            expressive_preferences: vec![],
            tactical_preferences: vec![],
        };
        assert_eq!(sample_preference(&empty, 0.5), None);
    }
}

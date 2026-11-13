//! Style parameters that cannot widen an envelope, per SPECS.md Section
//! 12.5: "Expression MUST NOT widen any envelope".
//!
//! A style never states a cadence, an idle amplitude or a speech delay in
//! physical units. It states a **position inside** a range that the task
//! or skill validation already approved, as a number in `[0, 1]`, and
//! [`BoundedParameter::resolve`] maps it into whatever range the caller
//! currently holds. Out-of-range inputs are clamped into `[0, 1]` before
//! resolution, so no stored value, however corrupted, can produce a
//! resolved value outside the validated range: widening the range is a
//! validation change made elsewhere, never a style edit.
//!
//! The dimensions are the ones Section 12.5 lists for bounded intent:
//! posture style, cadence, idle motion, celebrations and speech timing,
//! plus tactical preferences, as a closed enum so a style cannot invent an
//! unbounded-effect knob.

/// An expressive dimension a style may position within its validated range.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Dimension {
    /// Stance width or guard height within the validated posture range.
    Posture,
    /// Tempo of rhythmic motion (bouncing, stepping) within its range.
    Cadence,
    /// Amplitude of idle motion.
    IdleAmplitude,
    /// How long a celebration lasts within its validated duration range.
    CelebrationDuration,
    /// Delay before a speech line within its allowed window.
    SpeechDelay,
    /// Readiness to attack, within the strategy layer's range.
    Aggression,
    /// Willingness to wait for an opening.
    Patience,
}

impl Dimension {
    /// Every dimension.
    pub const ALL: [Self; 7] = [
        Self::Posture,
        Self::Cadence,
        Self::IdleAmplitude,
        Self::CelebrationDuration,
        Self::SpeechDelay,
        Self::Aggression,
        Self::Patience,
    ];

    /// Stable name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Posture => "posture",
            Self::Cadence => "cadence",
            Self::IdleAmplitude => "idle_amplitude",
            Self::CelebrationDuration => "celebration_duration",
            Self::SpeechDelay => "speech_delay",
            Self::Aggression => "aggression",
            Self::Patience => "patience",
        }
    }

    /// Whether the dimension acts at strategy rate (tactical) rather than
    /// on motion style.
    #[must_use]
    pub const fn is_tactical(self) -> bool {
        matches!(self, Self::Aggression | Self::Patience)
    }
}

/// A position inside a validated range.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct BoundedParameter {
    /// The dimension.
    pub dimension: Dimension,
    /// Position in `[0, 1]`; anything else is clamped.
    pub position: f64,
}

impl BoundedParameter {
    /// The value inside `[lo, hi]` this position names. A non-finite
    /// position or range resolves to `lo` (the conservative end the caller
    /// chose), and a reversed range is treated as its sorted form.
    #[must_use]
    pub fn resolve(&self, lo: f64, hi: f64) -> f64 {
        if !lo.is_finite() || !hi.is_finite() {
            return lo;
        }
        let (a, b) = if lo <= hi { (lo, hi) } else { (hi, lo) };
        let p = if self.position.is_finite() {
            self.position.clamp(0.0, 1.0)
        } else {
            0.0
        };
        a + (b - a) * p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_position_resolves_outside_the_validated_range() {
        for position in [-5.0, 0.0, 0.3, 1.0, 7.0, f64::NAN, f64::INFINITY] {
            let p = BoundedParameter {
                dimension: Dimension::Cadence,
                position,
            };
            let v = p.resolve(1.2, 2.0);
            assert!((1.2..=2.0).contains(&v), "{position} -> {v}");
        }
        let p = BoundedParameter {
            dimension: Dimension::Cadence,
            position: 0.25,
        };
        assert!((p.resolve(1.0, 3.0) - 1.5).abs() < 1e-12);
        assert!((p.resolve(3.0, 1.0) - 1.5).abs() < 1e-12);
    }

    #[test]
    fn tactical_dimensions_are_marked() {
        let tactical: Vec<_> = Dimension::ALL.iter().filter(|d| d.is_tactical()).collect();
        assert_eq!(tactical, [&Dimension::Aggression, &Dimension::Patience]);
    }
}

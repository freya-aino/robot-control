//! Per-site contact classification from force measurements, with hysteresis.
//!
//! SPECS.md Section 11.2 names "contact assumptions" as something the
//! estimation module must specify explicitly. This is the minimal
//! classifier that turns a noisy force reading into a `(in_contact,
//! confidence)` pair without chattering across the threshold — the
//! hysteresis band is the explicit answer to "what counts as a contact
//! transition," rather than a bare `force > 0` comparison that would flip on
//! every noise sample near the boundary.

use robot_types::ids::ContactSiteId;
use robot_types::units::Newtons;

/// Hysteresis thresholds for one contact site.
#[derive(Clone, Copy, Debug)]
pub struct ContactThresholds {
    /// Force above which contact is asserted if not already in contact.
    pub engage: Newtons,
    /// Force below which contact is cleared if currently in contact.
    /// Must be less than `engage` for the hysteresis band to be non-empty.
    pub release: Newtons,
}

/// Why thresholds were rejected.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct InvertedThresholds;

impl core::fmt::Display for InvertedThresholds {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "release threshold must be below engage threshold")
    }
}

impl core::error::Error for InvertedThresholds {}

impl ContactThresholds {
    /// Builds thresholds, rejecting an inverted or degenerate band.
    pub fn new(engage: Newtons, release: Newtons) -> Result<Self, InvertedThresholds> {
        if release.value() >= engage.value() {
            return Err(InvertedThresholds);
        }
        Ok(Self { engage, release })
    }
}

/// A classifier for one contact site, holding the hysteresis state between
/// calls.
pub struct ContactClassifier {
    site: ContactSiteId,
    thresholds: ContactThresholds,
    in_contact: bool,
}

impl ContactClassifier {
    /// Builds a classifier, starting out of contact.
    #[must_use]
    pub const fn new(site: ContactSiteId, thresholds: ContactThresholds) -> Self {
        Self {
            site,
            thresholds,
            in_contact: false,
        }
    }

    /// The site this classifier tracks.
    #[must_use]
    pub const fn site(&self) -> ContactSiteId {
        self.site
    }

    /// Updates the classifier with a new force magnitude and returns the
    /// resulting `(in_contact, confidence)` pair.
    ///
    /// Confidence is a simple linear measure of how far the reading is past
    /// the threshold that governs the current state, saturating at 1: well
    /// past the engage threshold while in contact is high confidence; just
    /// past it is low confidence. It is a reporting aid, never a substitute
    /// for the hysteresis decision itself.
    pub fn update(&mut self, force: Newtons) -> (bool, f64) {
        if self.in_contact {
            if force.value() < self.thresholds.release.value() {
                self.in_contact = false;
            }
        } else if force.value() > self.thresholds.engage.value() {
            self.in_contact = true;
        }

        // Saturates at twice the relevant threshold rather than exactly at
        // it, so "just past the threshold" and "far past it" are
        // distinguishable confidences instead of both reading 1.0.
        let confidence = if self.in_contact {
            let span = (2.0 * self.thresholds.engage.value()).max(1e-9);
            (force.value() / span).clamp(0.0, 1.0)
        } else {
            let span = (2.0 * self.thresholds.release.value()).max(1e-9);
            (1.0 - force.value() / span).clamp(0.0, 1.0)
        };
        (self.in_contact, confidence)
    }

    /// Current state without a new measurement.
    #[must_use]
    pub const fn in_contact(&self) -> bool {
        self.in_contact
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classifier() -> ContactClassifier {
        ContactClassifier::new(
            ContactSiteId::new(0),
            ContactThresholds::new(Newtons(20.0), Newtons(5.0)).unwrap(),
        )
    }

    #[test]
    fn inverted_thresholds_are_rejected() {
        assert!(ContactThresholds::new(Newtons(5.0), Newtons(20.0)).is_err());
        assert!(ContactThresholds::new(Newtons(5.0), Newtons(5.0)).is_err());
    }

    #[test]
    fn engages_above_and_releases_below_with_hysteresis_in_between() {
        let mut c = classifier();
        assert!(!c.in_contact());
        let (in_contact, _) = c.update(Newtons(3.0));
        assert!(!in_contact, "below both thresholds: no contact");

        let (in_contact, _) = c.update(Newtons(10.0));
        assert!(
            !in_contact,
            "in the hysteresis band while not yet in contact: stays out"
        );

        let (in_contact, _) = c.update(Newtons(25.0));
        assert!(in_contact, "above engage: contact asserted");

        let (in_contact, _) = c.update(Newtons(10.0));
        assert!(
            in_contact,
            "in the hysteresis band while already in contact: stays in"
        );

        let (in_contact, _) = c.update(Newtons(3.0));
        assert!(!in_contact, "below release: contact cleared");
    }

    #[test]
    fn a_reading_oscillating_in_the_hysteresis_band_does_not_chatter() {
        let mut c = classifier();
        c.update(Newtons(25.0)); // engage
        for _ in 0..20 {
            let (in_contact, _) = c.update(Newtons(10.0)); // inside the band
            assert!(
                in_contact,
                "must not chatter while inside the hysteresis band"
            );
        }
    }

    #[test]
    fn confidence_increases_with_distance_past_the_relevant_threshold() {
        let mut c = classifier();
        let (_, low) = c.update(Newtons(21.0));
        let mut c2 = classifier();
        let (_, high) = c2.update(Newtons(100.0));
        assert!(high > low);
    }
}

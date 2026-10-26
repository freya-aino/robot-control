//! The dependency knowledge base's data model, per SPECS.md Section 13.1.
//!
//! "Every entry MUST be cited and attributed... An uncited entry is a
//! note, and a note MUST NOT be read as a specification." and "Entries
//! MUST be invalidated by version change... A stale entry is visibly
//! stale, not silently wrong." This module is the typed form of exactly
//! those two rules: [`KnowledgeBaseEntry::require_citation`] is the gate
//! between a note and a specification, and [`KnowledgeBaseEntry::update_version`]
//! is the only way an entry's version changes, and it always marks the
//! entry stale.
//!
//! This crate (`robot-devtools`) is developer/CI-only tooling under use U1
//! and U3 of SPECS.md Section 2.3.1; no shipping crate depends on it, and
//! `PLAN.md`'s own CI-assertion requirement for that boundary is tracked,
//! not yet enforced here as an actual lint.

use core::fmt;
use robot_types::evidence::EvidenceState;

/// A citation: where a claim came from, required before it is anything
/// more than a note.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Citation {
    /// Source document, URL or standard reference.
    pub source: String,
    /// Version or revision of the source.
    pub source_version: String,
    /// Date the citation was recorded.
    pub date: String,
    /// Who or what added the entry (a person's role, or
    /// "tool:<name>:<version>" for development-time automation — SPECS.md
    /// Section 2.3.1 item 5 requires generated artifacts to be
    /// attributable).
    pub added_by: String,
}

/// One entry in the dependency knowledge base.
#[derive(Clone, Debug)]
pub struct KnowledgeBaseEntry {
    /// The dependency, firmware, driver or platform item this entry is
    /// about.
    pub subject: String,
    /// Version this entry's claims apply to.
    pub version: String,
    /// The claim itself.
    pub claim: String,
    /// Citation, if any. `None` means this is a note, not a specification.
    pub citation: Option<Citation>,
    /// Evidence state of the underlying claim, per SPECS.md Section 3.
    pub evidence: EvidenceState,
    /// Whether a version change has made this entry stale since it was
    /// written.
    pub stale: bool,
}

/// Why an entry could not be used as a specification.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NotASpecification {
    /// The entry has no citation at all.
    Uncited,
    /// The entry is marked stale.
    Stale,
}

impl fmt::Display for NotASpecification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Uncited => write!(
                f,
                "entry has no citation; it is a note, not a specification"
            ),
            Self::Stale => write!(f, "entry is marked stale since a version change"),
        }
    }
}

impl core::error::Error for NotASpecification {}

impl KnowledgeBaseEntry {
    /// Checks whether this entry may be cited as a specification (SPECS.md
    /// Section 13.1 item 3 separately forbids citing the base as evidence
    /// under Section 3 or as a hazard control — this check is upstream of
    /// that: it only confirms the entry is not an uncited or stale note in
    /// the first place).
    pub fn require_citation(&self) -> Result<&Citation, NotASpecification> {
        if self.stale {
            return Err(NotASpecification::Stale);
        }
        self.citation.as_ref().ok_or(NotASpecification::Uncited)
    }

    /// Updates the entry's version, marking it stale.
    ///
    /// SPECS.md Section 13.1 item 4: "A firmware, driver, crate or
    /// robot-profile change marks every dependent entry stale, exactly as
    /// it demotes a claim under Section 3." This is the only method that
    /// changes `version`, and it never leaves `stale` as it was — the
    /// caller re-validates and calls [`KnowledgeBaseEntry::refresh`] to
    /// clear it once the entry has actually been re-checked against the
    /// new version.
    pub fn update_version(&mut self, new_version: impl Into<String>) {
        self.version = new_version.into();
        self.stale = true;
    }

    /// Clears the stale flag after the entry has been re-verified against
    /// its current version, recording the new citation.
    pub fn refresh(&mut self, citation: Citation) {
        self.citation = Some(citation);
        self.stale = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cited_entry() -> KnowledgeBaseEntry {
        KnowledgeBaseEntry {
            subject: "rapier3d".to_owned(),
            version: "0.22.0".to_owned(),
            claim: "supports continuous collision detection".to_owned(),
            citation: Some(Citation {
                source: "https://docs.rs/rapier3d/0.22.0".to_owned(),
                source_version: "0.22.0".to_owned(),
                date: "2026-10-26".to_owned(),
                added_by: "freya".to_owned(),
            }),
            evidence: EvidenceState::CodeInspected,
            stale: false,
        }
    }

    #[test]
    fn an_uncited_entry_is_a_note_not_a_specification() {
        let mut entry = cited_entry();
        entry.citation = None;
        assert_eq!(entry.require_citation(), Err(NotASpecification::Uncited));
    }

    #[test]
    fn a_cited_fresh_entry_may_be_used() {
        let entry = cited_entry();
        assert!(entry.require_citation().is_ok());
    }

    #[test]
    fn a_version_change_marks_the_entry_stale_and_refuses_citation() {
        let mut entry = cited_entry();
        entry.update_version("0.23.0");
        assert!(entry.stale);
        assert_eq!(entry.require_citation(), Err(NotASpecification::Stale));
    }

    #[test]
    fn refreshing_after_a_version_change_restores_usability() {
        let mut entry = cited_entry();
        entry.update_version("0.23.0");
        entry.refresh(Citation {
            source: "https://docs.rs/rapier3d/0.23.0".to_owned(),
            source_version: "0.23.0".to_owned(),
            date: "2026-11-01".to_owned(),
            added_by: "freya".to_owned(),
        });
        assert!(!entry.stale);
        assert!(entry.require_citation().is_ok());
        assert_eq!(entry.version, "0.23.0");
    }
}

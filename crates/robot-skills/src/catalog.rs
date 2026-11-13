//! The optional Markdown skill catalog of SPECS.md Section 10.1: a
//! human-readable descriptor (capabilities, preconditions, failure modes,
//! counter-indications) "indexed at build or load time and matched
//! lexically", which "MAY contribute a routing signal that composes with
//! the numerical retrieval" and "MUST NOT replace it or relax any
//! feasibility check".
//!
//! [`Catalog`] indexes descriptors by TF-IDF over lowercase word tokens,
//! with no model call. [`Catalog::compose`] is the only way its signal
//! reaches routing, and it can only **reorder** the candidates numerical
//! retrieval already produced, never add one; feasibility runs afterwards
//! on whatever order results. [`Catalog::disagreements`] reports where a
//! descriptor's prose contradicts the typed descriptor's contact
//! assumption, which the section calls "a defect in the descriptor".

use crate::descriptor::{SkillDescriptor, SkillId};

fn tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| t.len() > 1)
        .map(str::to_ascii_lowercase)
        .collect()
}

/// One indexed descriptor: its skill, its text and its term weights.
type Indexed = (SkillId, String, Vec<(String, f64)>);

/// A lexical index of skill descriptors.
#[derive(Clone, Debug, Default)]
pub struct Catalog {
    docs: Vec<Indexed>,
}

impl Catalog {
    /// Indexes `(skill, markdown)` pairs.
    #[must_use]
    pub fn build(entries: &[(SkillId, &str)]) -> Self {
        let n = entries.len().max(1) as f64;
        let docs_tokens: Vec<Vec<String>> = entries.iter().map(|(_, t)| tokens(t)).collect();
        let df = |term: &str| {
            docs_tokens
                .iter()
                .filter(|d| d.iter().any(|t| t == term))
                .count() as f64
        };
        let docs = entries
            .iter()
            .zip(&docs_tokens)
            .map(|((id, text), toks)| {
                let mut terms: Vec<(String, f64)> = Vec::new();
                for t in toks {
                    match terms.iter_mut().find(|(u, _)| u == t) {
                        Some(e) => e.1 += 1.0,
                        None => terms.push((t.clone(), 1.0)),
                    }
                }
                let len = toks.len().max(1) as f64;
                for (t, w) in &mut terms {
                    *w = (*w / len) * (1.0 + (n / df(t)).ln());
                }
                (*id, (*text).to_owned(), terms)
            })
            .collect();
        Self { docs }
    }

    /// Lexical scores for `query`, highest first.
    #[must_use]
    pub fn search(&self, query: &str) -> Vec<(SkillId, f64)> {
        let q = tokens(query);
        let mut out: Vec<(SkillId, f64)> = self
            .docs
            .iter()
            .map(|(id, _, terms)| {
                let s = q
                    .iter()
                    .filter_map(|t| terms.iter().find(|(u, _)| u == t))
                    .map(|(_, w)| w)
                    .sum();
                (*id, s)
            })
            .filter(|(_, s)| *s > 0.0)
            .collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1));
        out
    }

    /// Reorders numerical candidates `(skill, squared distance)` by
    /// subtracting `weight` times each one's lexical score for `query`.
    /// The set of candidates is unchanged.
    #[must_use]
    pub fn compose(
        &self,
        numeric: &[(SkillId, f64)],
        query: &str,
        weight: f64,
    ) -> Vec<(SkillId, f64)> {
        let lexical = self.search(query);
        let mut out: Vec<(SkillId, f64)> = numeric
            .iter()
            .map(|(id, d)| {
                let s = lexical
                    .iter()
                    .find(|(l, _)| l == id)
                    .map_or(0.0, |(_, s)| *s);
                (*id, d - weight.max(0.0) * s)
            })
            .collect();
        out.sort_by(|a, b| a.1.total_cmp(&b.1));
        out
    }

    /// Skills whose prose states a contact assumption that contradicts the
    /// typed descriptor (a line `Contact: none` against
    /// `assumes_ground_contact = true`, or `Contact: ground` against false).
    #[must_use]
    pub fn disagreements(&self, typed: &[SkillDescriptor]) -> Vec<SkillId> {
        self.docs
            .iter()
            .filter_map(|(id, text, _)| {
                let d = typed.iter().find(|d| d.id == *id)?;
                let line = text
                    .lines()
                    .find_map(|l| l.trim().strip_prefix("Contact:"))?
                    .trim()
                    .to_ascii_lowercase();
                let prose_contact = match line.as_str() {
                    "none" | "airborne" => false,
                    "ground" | "feet" => true,
                    _ => return None,
                };
                (prose_contact != d.assumes_ground_contact).then_some(*id)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const JAB: &str = "# Jab\nCapabilities: fast straight lead-hand strike.\nContact: ground\n";
    const HOOK: &str = "# Hook\nCapabilities: curved power strike to the side.\nContact: none\n";

    #[test]
    fn lexical_search_ranks_by_tf_idf() {
        let c = Catalog::build(&[(SkillId(1), JAB), (SkillId(2), HOOK)]);
        let r = c.search("straight strike");
        assert_eq!(r[0].0, SkillId(1));
    }

    #[test]
    fn composing_reorders_but_never_adds_candidates() {
        let c = Catalog::build(&[
            (SkillId(1), JAB),
            (SkillId(2), HOOK),
            (SkillId(3), "# Guard\nhands up"),
        ]);
        let numeric = [(SkillId(2), 1.0), (SkillId(1), 1.05)];
        let out = c.compose(&numeric, "straight", 1.0);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].0, SkillId(1));
        assert!(!out.iter().any(|(id, _)| *id == SkillId(3)));
    }

    #[test]
    fn prose_contradicting_the_typed_contact_field_is_a_defect() {
        use crate::descriptor::EmbeddingVersion;
        use robot_types::ids::Name;
        let typed = |id: u32| SkillDescriptor {
            id: SkillId(id),
            name: Name::new("x").unwrap(),
            tags: Vec::new(),
            embedding: vec![0.0],
            embedding_version: EmbeddingVersion(1),
            embodiment_compatibility: Vec::new(),
            initiation_guards: Vec::new(),
            termination_condition: Name::new("done").unwrap(),
            expected_duration: robot_types::time::Duration::from_millis(1),
            assumes_ground_contact: true,
            supported_disturbances: Vec::new(),
            inference_memory_bytes: 0,
            outgoing_transitions: Vec::new(),
        };
        let c = Catalog::build(&[(SkillId(1), JAB), (SkillId(2), HOOK)]);
        assert_eq!(c.disagreements(&[typed(1), typed(2)]), vec![SkillId(2)]);
    }
}

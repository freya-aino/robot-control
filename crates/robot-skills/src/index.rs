//! Exact nearest-neighbor search over skill embeddings.
//!
//! SPECS.md Section 10.1: "For small libraries, benchmark exact vector
//! search in Rust before adopting approximate nearest-neighbor indexing."
//! Appendix B.2.5 makes this the default and names `instant-distance` (an
//! approximate HNSW candidate) as qualified "only when measured library
//! scale justifies approximate search" — unmeasured and unqualified here,
//! per `docs/decisions/0003-std-only-default-closure.md`.
//!
//! [`ExactIndex::nearest_into`] is the zero-allocation form SPECS.md
//! Section 10.6 requires for the R2 tier: "An R2 lookup MUST operate on a
//! resident, preloaded index. It MUST NOT read a file, query a database,
//! compute an embedding, allocate, or block."

use crate::descriptor::SkillId;
use core::fmt;

/// A flat, resident array of (skill, embedding) pairs.
///
/// Built once (allocating freely) and then queried with
/// [`ExactIndex::nearest_into`], which allocates nothing.
pub struct ExactIndex {
    ids: Vec<SkillId>,
    embeddings: Vec<Vec<f64>>,
    dimension: usize,
}

/// Why an index could not be built or queried.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IndexError {
    /// An embedding's width did not match the index's declared dimension.
    DimensionMismatch {
        /// Expected width.
        expected: usize,
        /// Width found.
        found: usize,
    },
    /// The index holds no entries.
    Empty,
}

impl fmt::Display for IndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, found } => {
                write!(f, "embedding has width {found}, index expects {expected}")
            }
            Self::Empty => write!(f, "index has no entries"),
        }
    }
}

impl core::error::Error for IndexError {}

impl ExactIndex {
    /// Builds an index from (id, embedding) pairs, all of the same width.
    pub fn build(dimension: usize, entries: Vec<(SkillId, Vec<f64>)>) -> Result<Self, IndexError> {
        for (_, e) in &entries {
            if e.len() != dimension {
                return Err(IndexError::DimensionMismatch {
                    expected: dimension,
                    found: e.len(),
                });
            }
        }
        let (ids, embeddings) = entries.into_iter().unzip();
        Ok(Self {
            ids,
            embeddings,
            dimension,
        })
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// True when the index holds no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Embedding dimension this index was built for.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Finds the `out.len()` nearest entries to `query` by squared Euclidean
    /// distance, writing `(SkillId, distance)` pairs into `out` in
    /// ascending-distance order and returning how many were filled (fewer
    /// than `out.len()` only when the index itself has fewer entries).
    ///
    /// Allocates nothing: this is the R2-tier form. A linear scan over a
    /// resident array is exactly what "small libraries" (Section 10.1)
    /// calls for; the crossover to an approximate index is a measurement
    /// this module does not yet have evidence to make (Section 10.6 item 4).
    pub fn nearest_into(
        &self,
        query: &[f64],
        out: &mut [(SkillId, f64)],
    ) -> Result<usize, IndexError> {
        if query.len() != self.dimension {
            return Err(IndexError::DimensionMismatch {
                expected: self.dimension,
                found: query.len(),
            });
        }
        if self.is_empty() {
            return Err(IndexError::Empty);
        }
        let k = out.len().min(self.ids.len());
        // Selection over a fixed-size output slice: no heap allocation,
        // O(n*k) which is fine for the "small library" regime this tier is
        // qualified for.
        let mut filled = 0usize;
        for (id, embedding) in self.ids.iter().zip(&self.embeddings) {
            let distance = squared_distance(query, embedding);
            if filled < k {
                out[filled] = (*id, distance);
                filled += 1;
                if filled == k {
                    out[..filled].sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                }
            } else if distance < out[k - 1].1 {
                out[k - 1] = (*id, distance);
                out[..k].sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            }
        }
        if filled < k {
            out[..filled].sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        }
        Ok(filled)
    }
}

fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index() -> ExactIndex {
        ExactIndex::build(
            2,
            vec![
                (SkillId(1), vec![0.0, 0.0]),
                (SkillId(2), vec![1.0, 0.0]),
                (SkillId(3), vec![5.0, 5.0]),
            ],
        )
        .unwrap()
    }

    #[test]
    fn nearest_returns_closest_entries_in_order() {
        let idx = index();
        let mut out = [(SkillId(0), 0.0); 2];
        let filled = idx.nearest_into(&[0.1, 0.0], &mut out).unwrap();
        assert_eq!(filled, 2);
        assert_eq!(out[0].0, SkillId(1));
        assert_eq!(out[1].0, SkillId(2));
    }

    #[test]
    fn requesting_more_than_available_returns_what_exists() {
        let idx = index();
        let mut out = [(SkillId(0), 0.0); 10];
        let filled = idx.nearest_into(&[0.0, 0.0], &mut out).unwrap();
        assert_eq!(filled, 3);
    }

    #[test]
    fn dimension_mismatch_is_rejected() {
        let idx = index();
        let mut out = [(SkillId(0), 0.0); 1];
        assert!(matches!(
            idx.nearest_into(&[0.0, 0.0, 0.0], &mut out),
            Err(IndexError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            ExactIndex::build(2, vec![(SkillId(1), vec![1.0])]),
            Err(IndexError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn an_empty_index_is_reported() {
        let idx = ExactIndex::build(2, vec![]).unwrap();
        let mut out = [(SkillId(0), 0.0); 1];
        assert!(matches!(
            idx.nearest_into(&[0.0, 0.0], &mut out),
            Err(IndexError::Empty)
        ));
        assert!(idx.is_empty());
    }
}

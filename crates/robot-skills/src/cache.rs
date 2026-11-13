//! Staging optional experts, per SPECS.md Section 10.3: "Stage optional
//! experts asynchronously into a bounded cache; validate schema, checksum,
//! normalization and device memory before activation. Cold-load failure or
//! late retrieval retains a compatible current controller."
//!
//! [`ExpertCache::validate`] is the activation check, in the section's
//! order, and refuses with the first failure. [`Stager`] runs it off the
//! control thread: [`Stager::request`] hands an artifact to a worker and
//! [`Stager::poll`] collects results without blocking, so a slow or failed
//! load never stalls a decision; until a result arrives, the caller simply
//! keeps its current controller. Admitted experts are resident and their
//! memory is charged to the cache's budget until evicted.

use std::sync::mpsc::{channel, Receiver, Sender};

use robot_artifacts::hash::ContentId;

use crate::descriptor::SkillId;

/// An expert as loaded from storage.
#[derive(Clone, PartialEq, Debug)]
pub struct ExpertArtifact {
    /// The skill it implements.
    pub skill: SkillId,
    /// Input/output schema identifier it was exported with.
    pub schema: String,
    /// The bytes.
    pub bytes: Vec<u8>,
    /// Their content ID, from the manifest.
    pub checksum: ContentId,
    /// Whether observation normalizer statistics are present.
    pub has_normalizer: bool,
    /// Device memory needed when active, bytes.
    pub memory_bytes: u64,
}

/// Why an expert was not admitted.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum StageError {
    /// The schema differs from the controller's.
    Schema {
        /// Expected.
        expected: String,
        /// Found.
        found: String,
    },
    /// The bytes do not match their checksum.
    Checksum,
    /// No normalizer statistics: the weights cannot be used as trained.
    NoNormalizer,
    /// Not enough memory left in the budget.
    Memory {
        /// Needed.
        needed: u64,
        /// Free.
        free: u64,
    },
}

impl core::fmt::Display for StageError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Schema { expected, found } => {
                write!(f, "schema {found}, controller expects {expected}")
            }
            Self::Checksum => f.write_str("checksum mismatch"),
            Self::NoNormalizer => f.write_str("no observation normalizer"),
            Self::Memory { needed, free } => write!(f, "needs {needed} bytes, {free} free"),
        }
    }
}

impl core::error::Error for StageError {}

/// Resident experts within a memory budget.
#[derive(Clone, Debug)]
pub struct ExpertCache {
    schema: String,
    budget: u64,
    resident: Vec<(SkillId, u64)>,
}

impl ExpertCache {
    /// A cache for experts of `schema` within `budget` bytes.
    #[must_use]
    pub fn new(schema: impl Into<String>, budget: u64) -> Self {
        Self {
            schema: schema.into(),
            budget,
            resident: Vec::new(),
        }
    }

    /// Free bytes.
    #[must_use]
    pub fn free(&self) -> u64 {
        self.budget - self.resident.iter().map(|(_, m)| m).sum::<u64>()
    }

    /// The activation checks of Section 10.3, against the current budget.
    pub fn validate(&self, a: &ExpertArtifact) -> Result<(), StageError> {
        validate(&self.schema, self.free(), a)
    }

    /// Admits a validated expert (re-checking memory, which may have
    /// changed since the worker validated it).
    pub fn admit(&mut self, a: &ExpertArtifact) -> Result<(), StageError> {
        self.validate(a)?;
        self.resident.retain(|(s, _)| *s != a.skill);
        self.resident.push((a.skill, a.memory_bytes));
        Ok(())
    }

    /// Evicts an expert, freeing its memory.
    pub fn evict(&mut self, skill: SkillId) -> bool {
        let before = self.resident.len();
        self.resident.retain(|(s, _)| *s != skill);
        before != self.resident.len()
    }

    /// Resident skills.
    #[must_use]
    pub fn resident(&self) -> Vec<SkillId> {
        self.resident.iter().map(|(s, _)| *s).collect()
    }
}

fn validate(schema: &str, free: u64, a: &ExpertArtifact) -> Result<(), StageError> {
    if a.schema != schema {
        return Err(StageError::Schema {
            expected: schema.to_owned(),
            found: a.schema.clone(),
        });
    }
    if a.checksum.verify(&a.bytes).is_err() {
        return Err(StageError::Checksum);
    }
    if !a.has_normalizer {
        return Err(StageError::NoNormalizer);
    }
    if a.memory_bytes > free {
        return Err(StageError::Memory {
            needed: a.memory_bytes,
            free,
        });
    }
    Ok(())
}

/// Background validation of staged experts.
pub struct Stager {
    tx: Sender<(ExpertArtifact, String, u64)>,
    rx: Receiver<(ExpertArtifact, Result<(), StageError>)>,
}

impl Default for Stager {
    fn default() -> Self {
        Self::new()
    }
}

impl Stager {
    /// Starts the worker.
    #[must_use]
    pub fn new() -> Self {
        let (tx, work) = channel::<(ExpertArtifact, String, u64)>();
        let (done, rx) = channel();
        std::thread::spawn(move || {
            for (a, schema, free) in work {
                let r = validate(&schema, free, &a);
                if done.send((a, r)).is_err() {
                    break;
                }
            }
        });
        Self { tx, rx }
    }

    /// Queues `artifact` for validation against `cache`'s schema and its
    /// free memory now. Never blocks.
    pub fn request(&self, cache: &ExpertCache, artifact: ExpertArtifact) -> bool {
        self.tx
            .send((artifact, cache.schema.clone(), cache.free()))
            .is_ok()
    }

    /// A finished validation, if any. Never blocks.
    pub fn poll(&self) -> Option<(ExpertArtifact, Result<(), StageError>)> {
        self.rx.try_recv().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expert(skill: u32, mem: u64) -> ExpertArtifact {
        let bytes = vec![skill as u8; 16];
        ExpertArtifact {
            skill: SkillId(skill),
            schema: "obs-v3".into(),
            checksum: ContentId::of(&bytes),
            bytes,
            has_normalizer: true,
            memory_bytes: mem,
        }
    }

    #[test]
    fn each_activation_check_refuses_in_turn() {
        let cache = ExpertCache::new("obs-v3", 100);
        assert_eq!(cache.validate(&expert(1, 50)), Ok(()));
        let mut e = expert(1, 50);
        e.schema = "obs-v2".into();
        assert!(matches!(cache.validate(&e), Err(StageError::Schema { .. })));
        let mut e = expert(1, 50);
        e.bytes[0] ^= 1;
        assert_eq!(cache.validate(&e), Err(StageError::Checksum));
        let mut e = expert(1, 50);
        e.has_normalizer = false;
        assert_eq!(cache.validate(&e), Err(StageError::NoNormalizer));
        assert_eq!(
            cache.validate(&expert(1, 500)),
            Err(StageError::Memory {
                needed: 500,
                free: 100
            })
        );
    }

    #[test]
    fn admission_charges_the_budget_and_eviction_frees_it() {
        let mut cache = ExpertCache::new("obs-v3", 100);
        cache.admit(&expert(1, 60)).unwrap();
        assert_eq!(cache.free(), 40);
        assert!(cache.admit(&expert(2, 60)).is_err());
        assert!(cache.evict(SkillId(1)));
        cache.admit(&expert(2, 60)).unwrap();
        assert_eq!(cache.resident(), vec![SkillId(2)]);
    }

    #[test]
    fn staging_runs_off_thread_and_polling_never_blocks() {
        let cache = ExpertCache::new("obs-v3", 100);
        let stager = Stager::new();
        assert!(stager.request(&cache, expert(3, 10)));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let (a, r) = loop {
            if let Some(x) = stager.poll() {
                break x;
            }
            assert!(std::time::Instant::now() < deadline);
            std::thread::yield_now();
        };
        assert_eq!(a.skill, SkillId(3));
        assert_eq!(r, Ok(()));
        assert!(stager.poll().is_none());
    }
}

//! Opponent memory, per SPECS.md Section 12.5, within the interim
//! position of open question C-17.
//!
//! Section 12.5's constraints on any opponent store: its read path MUST
//! NOT depend on network availability or a hosted backend; its contents are
//! an input to strategy, never to the supervisor or the motor loop; and a
//! store that is "unavailable, empty or stale MUST degrade to the
//! opponent-agnostic behavior rather than stalling a decision". C-17
//! (retention, consent, deletion, which backend) is open, and its interim
//! position is "No persistent opponent record is retained beyond a
//! session; no hosted backend is permitted on any read path."
//!
//! [`SessionMemory`] is exactly that interim: an in-process store with no
//! serialization and no file or network path, created per session and
//! dropped with it. It is written between rounds ([`SessionMemory::commit_round`])
//! and read at strategy rate ([`SessionMemory::bias`]), and the read
//! returns `None` (opponent-agnostic) when the opponent is unknown, has too
//! few observations, or was last observed longer ago than the caller's
//! staleness limit. The only thing it yields is a [`TacticalBias`], a pair
//! of bounded tactical positions; it has no type that reaches a command.

use std::collections::BTreeMap;

use crate::params::{BoundedParameter, Dimension};

/// What was seen of an opponent in one round.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RoundObservation {
    /// Fraction of the round the opponent spent attacking, `[0, 1]`.
    pub attack_fraction: f64,
    /// Fraction of our attacks the opponent evaded, `[0, 1]`.
    pub evasion_fraction: f64,
}

/// A strategy-rate bias derived from memory.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct TacticalBias {
    /// Position for [`Dimension::Aggression`].
    pub aggression: BoundedParameter,
    /// Position for [`Dimension::Patience`].
    pub patience: BoundedParameter,
}

#[derive(Clone, Debug, Default)]
struct OpponentRecord {
    rounds: Vec<RoundObservation>,
    last_round_ns: i64,
}

/// Session-scoped opponent memory. Not persisted, by construction.
#[derive(Debug, Default)]
pub struct SessionMemory {
    opponents: BTreeMap<String, OpponentRecord>,
}

impl SessionMemory {
    /// An empty memory for a new session.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a round against `opponent`, ending at `t_ns`. Observations
    /// outside `[0, 1]` are clamped; non-finite ones are dropped.
    pub fn commit_round(&mut self, opponent: &str, t_ns: i64, obs: RoundObservation) {
        if !(obs.attack_fraction.is_finite() && obs.evasion_fraction.is_finite()) {
            return;
        }
        let r = self.opponents.entry(opponent.to_owned()).or_default();
        r.rounds.push(RoundObservation {
            attack_fraction: obs.attack_fraction.clamp(0.0, 1.0),
            evasion_fraction: obs.evasion_fraction.clamp(0.0, 1.0),
        });
        r.last_round_ns = r.last_round_ns.max(t_ns);
    }

    /// The bias for `opponent` at `now_ns`, or `None` for the
    /// opponent-agnostic behavior: unknown, fewer than `min_rounds`
    /// observations, or last seen more than `stale_after_ns` ago.
    ///
    /// The mapping: against an opponent who attacks a lot, be more patient
    /// (counter); against one who evades a lot, be less aggressive. Both
    /// are positions in `[0, 1]`, resolved by the strategy layer inside its
    /// own validated ranges.
    #[must_use]
    pub fn bias(
        &self,
        opponent: &str,
        now_ns: i64,
        min_rounds: usize,
        stale_after_ns: i64,
    ) -> Option<TacticalBias> {
        let r = self.opponents.get(opponent)?;
        if r.rounds.len() < min_rounds.max(1) || now_ns - r.last_round_ns > stale_after_ns {
            return None;
        }
        let n = r.rounds.len() as f64;
        let attack = r.rounds.iter().map(|o| o.attack_fraction).sum::<f64>() / n;
        let evade = r.rounds.iter().map(|o| o.evasion_fraction).sum::<f64>() / n;
        Some(TacticalBias {
            aggression: BoundedParameter {
                dimension: Dimension::Aggression,
                position: 1.0 - evade,
            },
            patience: BoundedParameter {
                dimension: Dimension::Patience,
                position: attack,
            },
        })
    }

    /// Forgets `opponent` (a deletion request inside the session).
    pub fn forget(&mut self, opponent: &str) -> bool {
        self.opponents.remove(opponent).is_some()
    }

    /// Opponents currently held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.opponents.len()
    }

    /// Whether nothing is held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.opponents.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEC: i64 = 1_000_000_000;

    #[test]
    fn unknown_sparse_and_stale_opponents_degrade_to_agnostic() {
        let mut m = SessionMemory::new();
        assert!(m.bias("blue", 0, 1, SEC).is_none());
        m.commit_round(
            "blue",
            10 * SEC,
            RoundObservation {
                attack_fraction: 0.8,
                evasion_fraction: 0.2,
            },
        );
        assert!(m.bias("blue", 10 * SEC, 2, 60 * SEC).is_none()); // too few rounds
        m.commit_round(
            "blue",
            20 * SEC,
            RoundObservation {
                attack_fraction: 0.6,
                evasion_fraction: 0.4,
            },
        );
        let b = m.bias("blue", 30 * SEC, 2, 60 * SEC).unwrap();
        assert!((b.patience.position - 0.7).abs() < 1e-12);
        assert!((b.aggression.position - 0.7).abs() < 1e-12);
        assert!(m.bias("blue", 200 * SEC, 2, 60 * SEC).is_none()); // stale
    }

    #[test]
    fn corrupt_observations_cannot_push_a_bias_out_of_range() {
        let mut m = SessionMemory::new();
        m.commit_round(
            "red",
            0,
            RoundObservation {
                attack_fraction: 9.0,
                evasion_fraction: -3.0,
            },
        );
        m.commit_round(
            "red",
            0,
            RoundObservation {
                attack_fraction: f64::NAN,
                evasion_fraction: 0.0,
            },
        );
        let b = m.bias("red", 0, 1, SEC).unwrap();
        assert_eq!(b.patience.position, 1.0);
        assert_eq!(b.aggression.position, 1.0);
    }

    #[test]
    fn forgetting_removes_the_record() {
        let mut m = SessionMemory::new();
        m.commit_round(
            "x",
            0,
            RoundObservation {
                attack_fraction: 0.5,
                evasion_fraction: 0.5,
            },
        );
        assert!(m.forget("x"));
        assert!(m.is_empty());
    }
}

//! The bounded on-policy rollout buffer of SPECS.md Section 7.6.
//!
//! "Training storage MUST expose bounded chunks... it MUST NOT require
//! materializing the entire corpus or a complete replay buffer as one
//! allocation. ... On-policy rollout storage MUST report capacity
//! exhaustion rather than silently overwrite samples before the learner
//! consumes them." This buffer is sized once at construction and
//! [`RolloutBuffer::push`] refuses a transition past that capacity rather
//! than wrapping.
//!
//! It stores `robot_types::transition::Transition` directly — the same type
//! an eventual off-policy store would use — rather than a PPO-specific
//! record, which is what Section 7.6's "same versioned transition and
//! episode schema" for on-policy and off-policy data actually means in
//! practice.

use core::fmt;
use robot_types::transition::{Transition, TransitionError, TransitionSchema};

/// Why a transition could not be stored.
#[derive(Clone, PartialEq, Debug)]
pub enum RolloutError {
    /// The buffer is already at capacity.
    CapacityExhausted {
        /// The configured capacity.
        capacity: usize,
    },
    /// The transition did not match this buffer's schema.
    SchemaRejected(TransitionError),
}

impl fmt::Display for RolloutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CapacityExhausted { capacity } => {
                write!(
                    f,
                    "rollout buffer is at capacity ({capacity}); consume it before pushing more"
                )
            }
            Self::SchemaRejected(e) => write!(f, "{e}"),
        }
    }
}

impl core::error::Error for RolloutError {}

/// A fixed-capacity on-policy rollout buffer.
pub struct RolloutBuffer {
    schema: TransitionSchema,
    capacity: usize,
    transitions: Vec<Transition>,
}

impl RolloutBuffer {
    /// Builds an empty buffer, reserving its full capacity up front.
    #[must_use]
    pub fn new(schema: TransitionSchema, capacity: usize) -> Self {
        Self {
            schema,
            capacity,
            transitions: Vec::with_capacity(capacity),
        }
    }

    /// The buffer's capacity.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Transitions currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    /// True when the buffer holds no transitions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }

    /// Appends one transition, checking it against the schema.
    pub fn push(&mut self, transition: Transition) -> Result<(), RolloutError> {
        if self.transitions.len() >= self.capacity {
            return Err(RolloutError::CapacityExhausted {
                capacity: self.capacity,
            });
        }
        self.schema
            .accepts(&transition)
            .map_err(RolloutError::SchemaRejected)?;
        self.transitions.push(transition);
        Ok(())
    }

    /// The stored transitions, in collection order.
    #[must_use]
    pub fn as_slice(&self) -> &[Transition] {
        &self.transitions
    }

    /// Empties the buffer for reuse, without releasing its reserved
    /// capacity — the next collection phase allocates nothing new.
    pub fn clear(&mut self) {
        self.transitions.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_types::ids::EnvId;
    use robot_types::rng::SeedStream;
    use robot_types::tier::Generation;
    use robot_types::transition::{
        ActionRepresentation, RewardTerms, Termination, TransitionOrigin,
    };

    fn schema() -> TransitionSchema {
        TransitionSchema {
            schema_version: robot_types::transition::TRANSITION_SCHEMA_VERSION,
            observation_width: 2,
            action_width: 1,
            action_representation: ActionRepresentation::ActuatorSpace,
            robot_profile: "pendulum-fixture/1.0".to_owned(),
        }
    }

    fn transition() -> Transition {
        Transition {
            schema_version: robot_types::transition::TRANSITION_SCHEMA_VERSION,
            episode_id: "ep-0".to_owned(),
            timestep: 0,
            env_id: EnvId::new(0),
            robot_profile: "pendulum-fixture/1.0".to_owned(),
            observation: vec![0.0, 0.0],
            action: vec![0.1],
            action_representation: ActionRepresentation::ActuatorSpace,
            reward: RewardTerms::new(),
            next_observation: vec![0.0, 0.1],
            termination: Termination::Continuing,
            observation_validity: vec![true, true],
            task_data: Vec::new(),
            origin: TransitionOrigin::CurrentPolicy {
                generation: Generation(0),
            },
            seeds: SeedStream::new(1, 0),
            behavior_log_prob: Some(-0.5),
        }
    }

    #[test]
    fn pushing_past_capacity_reports_exhaustion_rather_than_overwriting() {
        let mut buf = RolloutBuffer::new(schema(), 2);
        buf.push(transition()).unwrap();
        buf.push(transition()).unwrap();
        assert_eq!(buf.len(), 2);
        assert_eq!(
            buf.push(transition()),
            Err(RolloutError::CapacityExhausted { capacity: 2 })
        );
        assert_eq!(buf.len(), 2, "the rejected push must not have been stored");
    }

    #[test]
    fn a_schema_mismatched_transition_is_rejected() {
        let mut buf = RolloutBuffer::new(schema(), 4);
        let mut bad = transition();
        bad.action.push(99.0); // now width 2, schema wants 1
        assert!(matches!(
            buf.push(bad),
            Err(RolloutError::SchemaRejected(_))
        ));
        assert!(buf.is_empty());
    }

    #[test]
    fn clear_empties_without_losing_capacity() {
        let mut buf = RolloutBuffer::new(schema(), 4);
        buf.push(transition()).unwrap();
        buf.clear();
        assert!(buf.is_empty());
        assert_eq!(buf.capacity(), 4);
        for _ in 0..4 {
            buf.push(transition()).unwrap();
        }
        assert_eq!(buf.len(), 4);
    }
}

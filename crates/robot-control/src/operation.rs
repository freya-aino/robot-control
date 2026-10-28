//! Hook points around the policy call in [`crate::control_loop::ControlLoop`].
//!
//! A [`LoopOperation`] is how behaviour is added to the loop without editing
//! it: sampling control, seeding, and online-training hooks are all operations
//! in their own crates. The loop is generic over its operation, so dispatch is
//! compile-time and the tick path never goes through a trait object.
//!
//! Every method either writes into a buffer the loop already owns or updates
//! state the operation allocated itself at construction. An operation that
//! keeps to that rule keeps the control path allocation-free (SPECS.md
//! Section 5).
//!
//! Operations compose: `(A, B)` is itself a [`LoopOperation`]. The first
//! element selects the action; both elements observe every tick.

use robot_nn::mlp::{MlpError, Scratch};
use robot_nn::policy::{Policy, StochasticPolicy};
use robot_supervisor::policy::Response;
use robot_types::rng::{Rng, SeedStream};

/// A width an operation was built for that does not match the loop it was
/// installed into.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct WidthMismatch {
    /// Which width disagreed, for example `"observation"` or `"action"`.
    pub component: &'static str,
    /// The width the loop provides.
    pub expected: usize,
    /// The width the operation was built for.
    pub found: usize,
}

impl core::fmt::Display for WidthMismatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "operation {} width is {}, loop provides {}",
            self.component, self.found, self.expected
        )
    }
}

impl core::error::Error for WidthMismatch {}

/// Everything one tick produced, handed to [`LoopOperation::after_tick`].
///
/// `raw_action` is the policy's (or selecting operation's) output before
/// supervision. `command` is what the supervisor let through and what was
/// transmitted. Recording training data normally wants `raw_action`, since
/// the log-probability belongs to it; `command` is the physical action.
#[derive(Clone, Copy, Debug)]
pub struct TickView<'a> {
    /// The normalized observation the policy saw.
    pub observation: &'a [f64],
    /// The action before supervision.
    pub raw_action: &'a [f64],
    /// The action after supervision, as transmitted.
    pub command: &'a [f64],
    /// The supervisor's response to this tick.
    pub response: Response,
    /// Log-probability of `raw_action`, when a stochastic selector produced
    /// it. `None` for deterministic selection.
    pub log_prob: Option<f64>,
}

/// A hook the control loop calls around every policy evaluation.
///
/// Every method has a default, so an operation implements only the hooks it
/// needs. [`LoopOperation::select_action`] is the one that decides which
/// action the policy produces; the others observe.
///
/// # When a tick fails
///
/// The hooks are not guaranteed to come in pairs. A tick ends at the first
/// stage that fails, and only the hooks of stages that ran have fired:
///
/// - Failure before the observation is normalized (adapter poll, a
///   non-finite or mis-sized observation): no hook is called. A non-finite
///   observation is escalated to the supervisor instead; see
///   `ControlLoop::tick`.
/// - Failure in `select_action`, the supervisor or the transmission:
///   `before_tick` has been called and `after_tick` will not be. A sampled
///   action whose transmission failed is therefore not recorded.
/// - `after_tick` is called only for a tick that completed.
///
/// An operation that needs to match `before_tick` to `after_tick` should
/// treat a second `before_tick` with no `after_tick` in between as an
/// abandoned tick, not as a protocol error.
pub trait LoopOperation<P: Policy> {
    /// Checks that the operation fits a loop with the given widths. Called
    /// once when the loop is constructed, so a mismatch is reported then
    /// rather than on the first tick.
    fn check_widths(
        &self,
        _observation_width: usize,
        _action_width: usize,
    ) -> Result<(), WidthMismatch> {
        Ok(())
    }

    /// Called once per tick, after the observation is normalized and before
    /// the policy runs.
    fn before_tick(&mut self, _observation: &[f64]) {}

    /// Produces this tick's raw action and returns its log-probability when
    /// the selection is stochastic. The default evaluates the policy
    /// deterministically and returns `None`.
    fn select_action(
        &mut self,
        policy: &P,
        observation: &[f64],
        action: &mut [f64],
        scratch: &mut Scratch,
    ) -> Result<Option<f64>, MlpError> {
        policy.act(observation, action, scratch)?;
        Ok(None)
    }

    /// Called once per tick after supervision.
    fn after_tick(&mut self, _tick: &TickView<'_>) {}
}

/// The operation that does nothing. It is the default for
/// [`crate::control_loop::ControlLoop`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NoOp;

impl<P: Policy> LoopOperation<P> for NoOp {}

/// Two operations run together. The first selects the action; both observe.
///
/// Only the first element's [`LoopOperation::select_action`] is called. Put
/// the selecting operation first: `(SamplingToggle, ExperienceRecorder)`
/// samples and records, while `(ExperienceRecorder, SamplingToggle)` records
/// the mean action and never samples.
impl<P: Policy, A: LoopOperation<P>, B: LoopOperation<P>> LoopOperation<P> for (A, B) {
    fn check_widths(
        &self,
        observation_width: usize,
        action_width: usize,
    ) -> Result<(), WidthMismatch> {
        self.0.check_widths(observation_width, action_width)?;
        self.1.check_widths(observation_width, action_width)
    }

    fn before_tick(&mut self, observation: &[f64]) {
        self.0.before_tick(observation);
        self.1.before_tick(observation);
    }

    fn select_action(
        &mut self,
        policy: &P,
        observation: &[f64],
        action: &mut [f64],
        scratch: &mut Scratch,
    ) -> Result<Option<f64>, MlpError> {
        self.0.select_action(policy, observation, action, scratch)
    }

    fn after_tick(&mut self, tick: &TickView<'_>) {
        self.0.after_tick(tick);
        self.1.after_tick(tick);
    }
}

/// Switches between the deterministic mean action and a sampled action.
///
/// With sampling enabled, the action comes from
/// [`StochasticPolicy::sample_action`] drawing from this operation's own RNG
/// stream, and the log-probability is returned to the loop. With sampling
/// disabled, it is the policy's mean action. Toggle it at any time through
/// `ControlLoop::operation_mut`.
#[derive(Clone, Debug)]
pub struct SamplingToggle {
    enabled: bool,
    rng: Rng,
}

impl SamplingToggle {
    /// Builds the toggle with its own RNG stream and initial state.
    #[must_use]
    pub fn new(stream: SeedStream, enabled: bool) -> Self {
        Self {
            enabled,
            rng: Rng::new(stream),
        }
    }

    /// Whether the next tick samples rather than taking the mean action.
    #[must_use]
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Enables or disables sampling for subsequent ticks.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

impl<P: StochasticPolicy> LoopOperation<P> for SamplingToggle {
    fn select_action(
        &mut self,
        policy: &P,
        observation: &[f64],
        action: &mut [f64],
        scratch: &mut Scratch,
    ) -> Result<Option<f64>, MlpError> {
        if self.enabled {
            policy
                .sample_action(observation, &mut self.rng, action, scratch)
                .map(Some)
        } else {
            policy.act(observation, action, scratch)?;
            Ok(None)
        }
    }
}

/// One recorded tick, read back from an [`ExperienceRecorder`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transition<'a> {
    /// The normalized observation the policy saw.
    pub observation: &'a [f64],
    /// The action before supervision.
    pub raw_action: &'a [f64],
    /// The action after supervision, as transmitted.
    pub command: &'a [f64],
    /// The log-probability of `raw_action`, when the selection was stochastic.
    pub log_prob: Option<f64>,
}

/// A training hook that keeps the last `capacity` transitions in
/// fixed-capacity storage.
///
/// All storage is allocated once in [`ExperienceRecorder::new`]. Each tick
/// writes into the next slot and overwrites the oldest entry when full, so
/// recording never allocates. A learner reads the transitions between ticks,
/// oldest first, with [`ExperienceRecorder::iter`], and drains them with
/// [`ExperienceRecorder::clear`].
#[derive(Clone, Debug)]
pub struct ExperienceRecorder {
    observation_width: usize,
    action_width: usize,
    capacity: usize,
    next: usize,
    len: usize,
    observations: Vec<f64>,
    raw_actions: Vec<f64>,
    commands: Vec<f64>,
    log_probs: Vec<Option<f64>>,
}

impl ExperienceRecorder {
    /// Allocates a recorder for observations of `observation_width`, actions
    /// of `action_width`, keeping the last `capacity` transitions (at least
    /// one).
    #[must_use]
    pub fn new(observation_width: usize, action_width: usize, capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            observation_width,
            action_width,
            capacity,
            next: 0,
            len: 0,
            observations: vec![0.0; capacity * observation_width],
            raw_actions: vec![0.0; capacity * action_width],
            commands: vec![0.0; capacity * action_width],
            log_probs: vec![None; capacity],
        }
    }

    /// Number of transitions currently held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether no transition is held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Maximum number of transitions held.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Forgets every held transition. Storage is kept.
    pub fn clear(&mut self) {
        self.next = 0;
        self.len = 0;
    }

    /// The transition at `index`, where 0 is the oldest held transition and
    /// `len() - 1` the newest.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<Transition<'_>> {
        if index >= self.len {
            return None;
        }
        let slot = (self.next + self.capacity - self.len + index) % self.capacity;
        Some(Transition {
            observation: &self.observations[slot * self.observation_width..]
                [..self.observation_width],
            raw_action: &self.raw_actions[slot * self.action_width..][..self.action_width],
            command: &self.commands[slot * self.action_width..][..self.action_width],
            log_prob: self.log_probs[slot],
        })
    }

    /// Iterates the held transitions oldest first.
    pub fn iter(&self) -> impl Iterator<Item = Transition<'_>> + '_ {
        (0..self.len).filter_map(move |i| self.get(i))
    }

    /// The most recently recorded transition, if any.
    #[must_use]
    pub fn last(&self) -> Option<Transition<'_>> {
        self.len.checked_sub(1).and_then(|i| self.get(i))
    }
}

impl<P: Policy> LoopOperation<P> for ExperienceRecorder {
    fn check_widths(
        &self,
        observation_width: usize,
        action_width: usize,
    ) -> Result<(), WidthMismatch> {
        if self.observation_width != observation_width {
            return Err(WidthMismatch {
                component: "observation",
                expected: observation_width,
                found: self.observation_width,
            });
        }
        if self.action_width != action_width {
            return Err(WidthMismatch {
                component: "action",
                expected: action_width,
                found: self.action_width,
            });
        }
        Ok(())
    }

    fn after_tick(&mut self, tick: &TickView<'_>) {
        let slot = self.next;
        self.observations[slot * self.observation_width..][..self.observation_width]
            .copy_from_slice(tick.observation);
        self.raw_actions[slot * self.action_width..][..self.action_width]
            .copy_from_slice(tick.raw_action);
        self.commands[slot * self.action_width..][..self.action_width]
            .copy_from_slice(tick.command);
        self.log_probs[slot] = tick.log_prob;
        self.next = (self.next + 1) % self.capacity;
        self.len = (self.len + 1).min(self.capacity);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use robot_nn::layer::{Activation, Linear};
    use robot_nn::mlp::Mlp;
    use robot_nn::policy::{Policy, StochasticPolicy};

    /// A stochastic policy whose mean is an identity `Mlp` and whose noise
    /// is Gaussian with a fixed standard deviation. Shared with the loop
    /// tests so they can exercise the stochastic path through `ControlLoop`.
    pub(crate) struct TestGaussian {
        pub(crate) mean: Mlp,
        pub(crate) std: f64,
    }

    impl Policy for TestGaussian {
        fn input_width(&self) -> usize {
            self.mean.input_width()
        }

        fn output_width(&self) -> usize {
            self.mean.output_width()
        }

        fn scratch_len(&self) -> usize {
            self.mean.scratch_len()
        }

        fn act(
            &self,
            observation: &[f64],
            action: &mut [f64],
            scratch: &mut Scratch,
        ) -> Result<(), MlpError> {
            self.mean.forward(observation, action, scratch)
        }
    }

    impl StochasticPolicy for TestGaussian {
        fn sample_action(
            &self,
            observation: &[f64],
            rng: &mut Rng,
            action: &mut [f64],
            scratch: &mut Scratch,
        ) -> Result<f64, MlpError> {
            self.mean.forward(observation, action, scratch)?;
            let mut log_prob = 0.0;
            for a in action.iter_mut() {
                let z = rng.standard_normal();
                *a += self.std * z;
                log_prob += -0.5 * z * z;
            }
            Ok(log_prob)
        }
    }

    pub(crate) fn identity_gaussian() -> TestGaussian {
        let layer = Linear::new(2, 2, vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 0.0]).unwrap();
        TestGaussian {
            mean: Mlp::new(vec![layer], vec![Activation::Identity]).unwrap(),
            std: 0.5,
        }
    }

    fn view<'a>(
        observation: &'a [f64],
        raw_action: &'a [f64],
        command: &'a [f64],
        log_prob: Option<f64>,
    ) -> TickView<'a> {
        TickView {
            observation,
            raw_action,
            command,
            response: Response::ClampAndContinue,
            log_prob,
        }
    }

    #[test]
    fn noop_select_action_is_the_deterministic_policy() {
        let net = identity_gaussian();
        let obs = [0.5, -0.25];
        let mut direct = [0.0; 2];
        let mut scratch = Scratch::new(net.scratch_len());
        net.act(&obs, &mut direct, &mut scratch).unwrap();

        let mut via_op = [0.0; 2];
        let mut op = NoOp;
        let lp = LoopOperation::<TestGaussian>::select_action(
            &mut op,
            &net,
            &obs,
            &mut via_op,
            &mut scratch,
        )
        .unwrap();
        assert_eq!(direct, via_op);
        assert_eq!(lp, None);
    }

    #[test]
    fn sampling_toggle_disabled_takes_the_mean_action_without_a_log_prob() {
        let net = identity_gaussian();
        let obs = [0.5, -0.25];
        let mut scratch = Scratch::new(net.scratch_len());
        let mut op = SamplingToggle::new(SeedStream::new(7, 0), false);
        let mut action = [0.0; 2];
        let lp = op
            .select_action(&net, &obs, &mut action, &mut scratch)
            .unwrap();
        assert_eq!(action, obs);
        assert_eq!(lp, None);
    }

    #[test]
    fn sampling_toggle_enabled_returns_a_log_prob_and_a_different_action() {
        let net = identity_gaussian();
        let obs = [0.5, -0.25];
        let mut scratch = Scratch::new(net.scratch_len());
        let mut op = SamplingToggle::new(SeedStream::new(7, 0), true);
        assert!(op.enabled());
        let mut action = [0.0; 2];
        let lp = op
            .select_action(&net, &obs, &mut action, &mut scratch)
            .unwrap();
        assert_ne!(action, obs);
        assert!(lp.is_some_and(f64::is_finite));
    }

    #[test]
    fn sampling_toggle_can_be_switched_between_ticks() {
        let net = identity_gaussian();
        let obs = [0.5, -0.25];
        let mut scratch = Scratch::new(net.scratch_len());
        let mut op = SamplingToggle::new(SeedStream::new(7, 0), false);
        let mut action = [0.0; 2];
        op.select_action(&net, &obs, &mut action, &mut scratch)
            .unwrap();
        assert_eq!(action, obs);
        op.set_enabled(true);
        op.select_action(&net, &obs, &mut action, &mut scratch)
            .unwrap();
        assert_ne!(action, obs);
    }

    #[test]
    fn sampling_toggle_is_reproducible_for_a_fixed_seed() {
        let net = identity_gaussian();
        let obs = [0.5, -0.25];
        let mut scratch = Scratch::new(net.scratch_len());
        let mut a = SamplingToggle::new(SeedStream::new(3, 1), true);
        let mut b = SamplingToggle::new(SeedStream::new(3, 1), true);
        let mut x = [0.0; 2];
        let mut y = [0.0; 2];
        for _ in 0..10 {
            let la = a.select_action(&net, &obs, &mut x, &mut scratch).unwrap();
            let lb = b.select_action(&net, &obs, &mut y, &mut scratch).unwrap();
            assert_eq!(x, y);
            assert_eq!(la, lb);
        }
    }

    #[test]
    fn recorder_reads_back_the_last_capacity_transitions_oldest_first() {
        let mut rec = ExperienceRecorder::new(2, 1, 3);
        assert!(rec.is_empty());
        assert_eq!(rec.last(), None);
        assert_eq!(rec.get(0), None);

        for t in 0..5 {
            let obs = [t as f64, -(t as f64)];
            let raw = [20.0 + t as f64];
            let cmd = [10.0 + t as f64];
            LoopOperation::<Mlp>::after_tick(&mut rec, &view(&obs, &raw, &cmd, Some(-(t as f64))));
        }
        assert_eq!(rec.len(), 3);
        assert_eq!(rec.capacity(), 3);

        let observations: Vec<f64> = rec.iter().map(|tr| tr.observation[0]).collect();
        assert_eq!(observations, vec![2.0, 3.0, 4.0]);

        let newest = rec.last().unwrap();
        assert_eq!(newest.observation, &[4.0, -4.0][..]);
        assert_eq!(newest.raw_action, &[24.0][..]);
        assert_eq!(newest.command, &[14.0][..]);
        assert_eq!(newest.log_prob, Some(-4.0));
    }

    #[test]
    fn recorder_clear_empties_it_and_keeps_storage() {
        let mut rec = ExperienceRecorder::new(2, 1, 2);
        let obs_ptr = rec.observations.as_ptr();
        LoopOperation::<Mlp>::after_tick(&mut rec, &view(&[1.0, 2.0], &[3.0], &[3.0], None));
        rec.clear();
        assert!(rec.is_empty());
        assert_eq!(rec.iter().count(), 0);
        assert_eq!(rec.observations.as_ptr(), obs_ptr);
    }

    #[test]
    fn recorder_storage_is_not_reallocated_by_recording() {
        let mut rec = ExperienceRecorder::new(2, 1, 4);
        let obs_ptr = rec.observations.as_ptr();
        let raw_ptr = rec.raw_actions.as_ptr();
        let cmd_ptr = rec.commands.as_ptr();
        let lp_ptr = rec.log_probs.as_ptr();
        for t in 0..20 {
            LoopOperation::<Mlp>::after_tick(
                &mut rec,
                &view(&[t as f64, 0.0], &[t as f64], &[t as f64], None),
            );
        }
        assert_eq!(rec.observations.as_ptr(), obs_ptr);
        assert_eq!(rec.raw_actions.as_ptr(), raw_ptr);
        assert_eq!(rec.commands.as_ptr(), cmd_ptr);
        assert_eq!(rec.log_probs.as_ptr(), lp_ptr);
    }

    #[test]
    fn recorder_rejects_a_width_mismatch_at_construction_time() {
        let rec = ExperienceRecorder::new(3, 1, 4);
        let err = LoopOperation::<Mlp>::check_widths(&rec, 2, 1).unwrap_err();
        assert_eq!(err.component, "observation");
        assert_eq!((err.expected, err.found), (2, 3));
        let err = LoopOperation::<Mlp>::check_widths(&rec, 3, 2).unwrap_err();
        assert_eq!(err.component, "action");
        assert!(LoopOperation::<Mlp>::check_widths(&rec, 3, 1).is_ok());
    }

    #[test]
    fn a_pair_records_the_action_that_the_first_element_selected() {
        let net = identity_gaussian();
        let obs = [0.5, -0.25];
        let mut scratch = Scratch::new(net.scratch_len());
        let mut pair = (
            SamplingToggle::new(SeedStream::new(9, 0), true),
            ExperienceRecorder::new(2, 2, 4),
        );
        let mut action = [0.0; 2];
        let lp = pair
            .select_action(&net, &obs, &mut action, &mut scratch)
            .unwrap();
        assert_ne!(action, obs);
        assert!(lp.is_some());
        LoopOperation::<TestGaussian>::after_tick(&mut pair, &view(&obs, &action, &action, lp));
        assert_eq!(pair.1.len(), 1);
        assert_eq!(pair.1.last().unwrap().raw_action, &action[..]);
    }
}

//! Generalized advantage estimation, respecting the
//! termination/truncation distinction.
//!
//! SPECS.md Section 9.2: "Distinguish truncation from terminal failure in
//! return calculation." [`compute_gae`] takes a per-step
//! `robot_types::transition::Termination` for exactly that reason:
//! `Termination::bootstraps()` controls whether the value of the state after
//! a step contributes to that step's TD error, and a time-limit truncation
//! bootstraps while a true terminal state does not.

use robot_types::transition::Termination;

/// Computes per-step advantages and value targets over one trajectory
/// segment, using the standard GAE(λ) recursion.
///
/// `rewards`, `values`, `next_values` and `terminations` must have equal
/// length `n`. `next_values[t]` MUST be `V(next_observation)` evaluated
/// directly from step `t`'s own recorded next observation — **not**
/// inferred as `values[t + 1]`.
///
/// This is deliberate, and it corrects a real bug an earlier version of
/// this function had: within one bounded rollout chunk (SPECS.md Section
/// 7.6 allows a chunk to span more than one episode), a step that ends an
/// episode by truncation is very often immediately followed, at index
/// `t + 1`, by the *first step of the next, freshly reset episode* — a
/// different trajectory entirely. Using `values[t + 1]` as that step's
/// bootstrap silently substituted the value of an unrelated state for the
/// genuine next state, corrupting the advantage estimate for exactly the
/// steps most likely to matter (an episode that survived to truncation
/// rather than failing early). Requiring the caller to supply `next_values`
/// explicitly removes the chance of reintroducing that mistake.
///
/// Returns `(advantages, returns)`, both of length `n`, where
/// `returns[i] = advantages[i] + values[i]` is the value-function
/// regression target.
#[must_use]
pub fn compute_gae(
    rewards: &[f64],
    values: &[f64],
    next_values: &[f64],
    terminations: &[Termination],
    gamma: f64,
    lambda: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = rewards.len();
    debug_assert_eq!(values.len(), n);
    debug_assert_eq!(next_values.len(), n);
    debug_assert_eq!(terminations.len(), n);

    let mut advantages = vec![0.0; n];
    let mut running = 0.0;
    for t in (0..n).rev() {
        let next_value = if terminations[t].bootstraps() {
            next_values[t]
        } else {
            0.0
        };
        // Once an episode ends (terminal or truncated), the advantage chain
        // does not carry across the boundary into the next episode's steps.
        // This also guards the *next* step's bootstrap: the step at t+1, if
        // it starts a new episode, was never part of this `running` chain,
        // but that boundary is enforced by the caller supplying a correct
        // `next_values[t]` for step t's own transition, not by this loop
        // inferring neighbors from `values`.
        let next_running = if terminations[t].ends_episode() {
            0.0
        } else {
            running
        };
        let delta = rewards[t] + gamma * next_value - values[t];
        running = delta + gamma * lambda * next_running;
        advantages[t] = running;
    }

    let returns: Vec<f64> = advantages.iter().zip(values).map(|(a, v)| a + v).collect();
    (advantages, returns)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_a_hand_computed_three_step_trajectory() {
        // gamma=1, lambda=1 reduces GAE to the plain Monte Carlo advantage:
        // advantage[t] = sum_{k>=t} reward[k] - value[t] (no bootstrap, all terminal).
        let rewards = [1.0, 2.0, 3.0];
        let values = [0.0, 0.0, 0.0];
        let next_values = [0.0, 0.0, 0.0];
        let terms = [
            Termination::Continuing,
            Termination::Continuing,
            Termination::Terminal,
        ];
        let (adv, ret) = compute_gae(&rewards, &values, &next_values, &terms, 1.0, 1.0);
        assert!((adv[2] - 3.0).abs() < 1e-12);
        assert!((adv[1] - 5.0).abs() < 1e-12);
        assert!((adv[0] - 6.0).abs() < 1e-12);
        assert!((ret[0] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn truncation_bootstraps_but_terminal_does_not() {
        let rewards = [1.0];
        let values = [0.0];
        let next_values = [100.0];
        let gamma = 0.9;

        let (adv_term, _) = compute_gae(
            &rewards,
            &values,
            &next_values,
            &[Termination::Terminal],
            gamma,
            1.0,
        );
        let (adv_trunc, _) = compute_gae(
            &rewards,
            &values,
            &next_values,
            &[Termination::TimeLimit],
            gamma,
            1.0,
        );

        // Terminal: advantage = reward + gamma*0 - value = 1.0.
        assert!((adv_term[0] - 1.0).abs() < 1e-12);
        // Truncated: advantage = reward + gamma*next_value - value = 1 + 0.9*100 = 91.
        assert!((adv_trunc[0] - (1.0 + gamma * 100.0)).abs() < 1e-9);
    }

    #[test]
    fn advantage_does_not_leak_across_an_episode_boundary() {
        // Two one-step episodes back to back; the second episode's return
        // must not be influenced by the first episode's reward.
        let rewards = [10.0, 1.0];
        let values = [0.0, 0.0];
        let next_values = [0.0, 0.0];
        let terms = [Termination::Terminal, Termination::Terminal];
        let (adv, _) = compute_gae(&rewards, &values, &next_values, &terms, 0.99, 0.95);
        assert!(
            (adv[1] - 1.0).abs() < 1e-12,
            "second episode's advantage must be its own reward only"
        );
    }

    #[test]
    fn a_truncation_in_the_middle_of_a_buffer_bootstraps_from_its_own_next_value_not_a_neighbor() {
        // Step 0 is a one-step truncated episode; step 1 is the *first* step
        // of a different, freshly reset episode. Using values[1] (a totally
        // different state) as step 0's bootstrap would be the bug this
        // signature exists to prevent; next_values[0] is what must be used.
        let rewards = [1.0, 50.0]; // step 1's huge reward must not leak into step 0.
        let values = [0.0, 0.0];
        let next_values = [7.0, 0.0]; // V(step 0's own next observation) = 7.0
        let terms = [Termination::TimeLimit, Termination::Terminal];
        let gamma = 0.9;
        let (adv, _) = compute_gae(&rewards, &values, &next_values, &terms, gamma, 1.0);
        // advantage[0] = reward[0] + gamma*next_values[0] - values[0] = 1 + 0.9*7 = 7.3,
        // emphatically not influenced by reward[1] = 50 or values[1].
        assert!(
            (adv[0] - (1.0 + gamma * 7.0)).abs() < 1e-9,
            "got {}",
            adv[0]
        );
    }

    #[test]
    fn returns_equal_advantage_plus_value() {
        let rewards = [0.5, -0.2, 0.9];
        let values = [0.1, 0.2, 0.3];
        let next_values = [0.2, 0.3, 0.4];
        let terms = [
            Termination::Continuing,
            Termination::Continuing,
            Termination::TimeLimit,
        ];
        let (adv, ret) = compute_gae(&rewards, &values, &next_values, &terms, 0.95, 0.9);
        for i in 0..3 {
            assert!((ret[i] - (adv[i] + values[i])).abs() < 1e-12);
        }
    }
}

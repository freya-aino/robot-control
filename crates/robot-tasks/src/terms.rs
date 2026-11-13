//! Reward terms shared by the humanoid tasks, covering SPECS.md Section
//! 9.2's list: "task completion, reference tracking, balance, energy,
//! action smoothness, joint/actuator constraints, and undesirable
//! contacts".
//!
//! Every term reads a [`HumanoidContext`]. Tracked links follow one fixed
//! layout ([`LEFT_FOOT`], [`RIGHT_FOOT`], [`RIGHT_HAND`], [`LEFT_HAND`],
//! [`HEAD`]), which a [`crate::humanoid::StateExtractor`] must be built to
//! produce. Positive terms are rewards in `[0, 1]`; penalty terms are
//! non-negative and carry a negative weight in the task's registry, so the
//! sign of every logged term is unambiguous.

use crate::humanoid::HumanoidContext;

/// Tracked-link index of the left foot.
pub const LEFT_FOOT: usize = 0;
/// Tracked-link index of the right foot.
pub const RIGHT_FOOT: usize = 1;
/// Tracked-link index of the right hand.
pub const RIGHT_HAND: usize = 2;
/// Tracked-link index of the left hand.
pub const LEFT_HAND: usize = 3;
/// Tracked-link index of the head.
pub const HEAD: usize = 4;

/// `exp(-(e / scale)^2)`: one at zero error, falling smoothly.
#[must_use]
pub fn kernel(error: f64, scale: f64) -> f64 {
    (-(error / scale).powi(2)).exp()
}

/// Balance: upright base, `exp(-(tilt / 0.3)^2)`.
#[must_use]
pub fn upright(ctx: &HumanoidContext<'_>) -> f64 {
    kernel(ctx.next.tilt(), 0.3)
}

/// Balance: base height near `target[0]`, 5 cm scale.
#[must_use]
pub fn height(ctx: &HumanoidContext<'_>) -> f64 {
    kernel(
        ctx.next.base_position.z - ctx.target.first().copied().unwrap_or(0.0),
        0.05,
    )
}

/// Energy (penalty): mechanical power `sum |tau qd|`, watts.
#[must_use]
pub fn power(ctx: &HumanoidContext<'_>) -> f64 {
    ctx.next
        .torque
        .iter()
        .zip(&ctx.next.qd)
        .map(|(t, v)| (t * v).abs())
        .sum()
}

/// Action smoothness (penalty): `|a - a_prev|^2`.
#[must_use]
pub fn action_rate(ctx: &HumanoidContext<'_>) -> f64 {
    ctx.action
        .iter()
        .zip(ctx.prev_action)
        .map(|(a, b)| (a - b).powi(2))
        .sum()
}

/// Joint velocity (penalty): `|qd|^2`.
#[must_use]
pub fn joint_speed(ctx: &HumanoidContext<'_>) -> f64 {
    ctx.next.qd.iter().map(|v| v * v).sum()
}

/// Undesirable contacts (penalty): their count.
#[must_use]
pub fn bad_contacts(ctx: &HumanoidContext<'_>) -> f64 {
    ctx.next.undesired_contacts as f64
}

/// Base drift (penalty): horizontal base speed while standing.
#[must_use]
pub fn base_drift(ctx: &HumanoidContext<'_>) -> f64 {
    ctx.next
        .base_linear_velocity
        .x
        .hypot(ctx.next.base_linear_velocity.y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::humanoid::tests::standing;

    #[test]
    fn standing_still_scores_full_balance_and_no_penalty() {
        let s = standing(3);
        let a = [0.0; 3];
        let ctx = HumanoidContext {
            prev: &s,
            action: &a,
            prev_action: &a,
            next: &s,
            target: &[0.75],
        };
        assert!((upright(&ctx) - 1.0).abs() < 1e-12);
        assert!((height(&ctx) - 1.0).abs() < 1e-12);
        assert_eq!(power(&ctx), 0.0);
        assert_eq!(action_rate(&ctx), 0.0);
        assert_eq!(bad_contacts(&ctx), 0.0);
    }

    #[test]
    fn power_counts_negative_work_too() {
        let mut s = standing(2);
        s.torque = vec![2.0, -3.0];
        s.qd = vec![1.0, 1.0];
        let ctx = HumanoidContext {
            prev: &s,
            action: &[],
            prev_action: &[],
            next: &s,
            target: &[],
        };
        assert_eq!(power(&ctx), 5.0);
    }
}

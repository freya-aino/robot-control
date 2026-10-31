//! Passive joint forces and affine actuator gain/bias, ported from the reference simulator 3.17.0
//! (commit `800a109a`).
//!
//! Pure algorithm module: `std` and plain `f64` only (FR-6). Scope is the
//! linear, scalar-joint subset. The polynomial coefficients (`*poly`), wrapped
//! rotational setpoints, muscle/DC-motor/PID gains, tendon and flex forces,
//! gravity compensation, fluid and contact passive forces, and user callbacks
//! are not ported; they are listed in `crates/robot-sim/specs/05-passive-forces-and-actuators.md`.

/// Joint spring force for a hinge or slide joint with a linear stiffness.
///
/// Formulation: `src/engine/engine_passive.c`, `springdamper`
/// (a `static` function), hinge/slide branch: `x = qpos - qpos_spring`,
/// `qfrc_spring = -x * polyForce(stiffness, spoly, x, NPOLY, 0)`. With
/// the polynomial coefficients zero, `polyForce` returns `stiffness`, so
/// the force is `-stiffness * (q - q_ref)`.
pub fn joint_spring_force(q: f64, q_ref: f64, stiffness: f64) -> f64 {
    -(q - q_ref) * stiffness
}

/// Joint damper force for a dof with a linear damping coefficient.
///
/// Formulation: `src/engine/engine_passive.c`, `springdamper`,
/// dof-level damper: `qfrc_damper = -v * polyForce(damping, poly, v, NPOLY, 1)`.
/// With the polynomial coefficients zero and `flg_odd = 1`, `polyForce`
/// returns `damping`, so the force is `-damping * v` (`polyForce` in
/// `src/engine/engine_util_misc.c`).
pub fn joint_damper_force(qd: f64, damping: f64) -> f64 {
    -qd * damping
}

/// Affine actuator force `gain * input + bias`, with `gain` and `bias` both
/// affine in length and velocity.
///
/// Formulation: `src/engine/engine_forward.c`, `fwdActuation`:
/// `GAIN_AFFINE` gives `gain = prm[0] + prm[1]*length + prm[2]*velocity`,
/// `BIAS_AFFINE` gives `bias = biasprm[0] + biasprm[1]*length +
/// biasprm[2]*velocity`, and the output is `force = gain * ctrl + bias`
/// (`force[oadr] += bias`). A fixed-gain, affine-bias position servo is the
/// case `gain_prm = [kp, 0, 0]`, `bias_prm = [0, -kp, -kv]`.
///
/// `ctrl` is the control input. In the same function the input is `ctrl` only
/// when the actuator has a control (`actuator_ctrlnum != 0`), and `0`
/// otherwise; this port takes `ctrl` as the caller's input, so pass `0.0` for
/// an actuator without a control.
pub fn affine_actuator_force(
    gain_prm: [f64; 3],
    bias_prm: [f64; 3],
    ctrl: f64,
    length: f64,
    velocity: f64,
) -> f64 {
    let gain = gain_prm[0] + gain_prm[1] * length + gain_prm[2] * velocity;
    let bias = bias_prm[0] + bias_prm[1] * length + bias_prm[2] * velocity;
    gain * ctrl + bias
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spring_force_is_minus_stiffness_times_displacement() {
        // Hand derivation: q - q_ref = 0.3 - 0.1 = 0.2, k = 5,
        // so F = -5 * 0.2 = -1.0. The sign is restoring: F < 0 for q > q_ref.
        assert!((joint_spring_force(0.3, 0.1, 5.0) + 1.0).abs() < 1e-15);
        // At the reference the spring exerts no force.
        assert_eq!(joint_spring_force(0.1, 0.1, 5.0), 0.0);
        // Zero stiffness gives zero force regardless of displacement.
        assert_eq!(joint_spring_force(7.0, -2.0, 0.0), 0.0);
    }

    #[test]
    fn damper_force_opposes_velocity_with_the_damping_coefficient() {
        // Hand derivation: v = 2, b = 0.5, so F = -b*v = -1.0, opposite to v.
        assert!((joint_damper_force(2.0, 0.5) + 1.0).abs() < 1e-15);
        assert!((joint_damper_force(-2.0, 0.5) - 1.0).abs() < 1e-15);
        assert_eq!(joint_damper_force(3.0, 0.0), 0.0);
    }

    #[test]
    fn position_servo_is_kp_times_error_minus_kv_times_velocity() {
        // Hand derivation: the reference simulator position servo with kp = 10, kv = 2,
        // gainprm = [10, 0, 0], biasprm = [0, -10, -2], ctrl (setpoint) = 0.5,
        // length = 0.2, velocity = 1.0.
        // force = 10*0.5 + (0 - 10*0.2 - 2*1.0) = 5 - 2 - 2 = 1.0.
        // Equivalently kp*(setpoint - length) - kv*velocity = 10*0.3 - 2 = 1.0.
        let f = affine_actuator_force([10.0, 0.0, 0.0], [0.0, -10.0, -2.0], 0.5, 0.2, 1.0);
        assert!((f - 1.0).abs() < 1e-14, "force = {f}");
    }

    #[test]
    fn velocity_servo_is_kv_times_velocity_error() {
        // Hand derivation: velocity servo with kv = 4, gainprm = [4, 0, 0],
        // biasprm = [0, 0, -4], ctrl (setpoint velocity) = 0.5, length = 0.7 (ignored
        // because the length coefficients are zero), velocity = 0.25.
        // gain = 4, bias = 0 + 0*0.7 + (-4)*0.25 = -1, force = 4*0.5 - 1 = 1.0.
        // Equivalently kv*(v_set - v) = 4*(0.5 - 0.25) = 1.0.
        let f = affine_actuator_force([4.0, 0.0, 0.0], [0.0, 0.0, -4.0], 0.5, 0.7, 0.25);
        assert!((f - 1.0).abs() < 1e-14, "force = {f}");
    }

    #[test]
    fn affine_gain_and_bias_both_depend_on_length_and_velocity() {
        // Hand derivation: gainprm = [1, 2, 3], biasprm = [4, 5, 6], ctrl = 2,
        // length = 1, velocity = -1.
        // gain = 1 + 2*1 + 3*(-1) = 0; bias = 4 + 5*1 + 6*(-1) = 3;
        // force = 0*2 + 3 = 3. The gain term vanishes here, so the result checks
        // that gain and bias are both evaluated at the same length and velocity.
        let f = affine_actuator_force([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], 2.0, 1.0, -1.0);
        assert!((f - 3.0).abs() < 1e-15, "force = {f}");
    }

    #[test]
    fn zero_control_leaves_only_the_bias() {
        // With ctrl = 0 the output is the bias alone: bias = 1 + 0.5*2 + 0 = 2.
        let f = affine_actuator_force([9.0, 9.0, 9.0], [1.0, 0.5, 0.0], 0.0, 2.0, 4.0);
        assert!((f - 2.0).abs() < 1e-15, "force = {f}");
    }
}

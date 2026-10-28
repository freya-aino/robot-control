//! `robot sim qualify` and `robot sim multibody` — PLAN.md §4's proposed
//! `robot sim qualify` command, plus a second one this revision adds for
//! the multi-body solver in `robot_sim::articulated`.
//!
//! `qualify` is SPECS.md §9.1's "analytic pendulum" energy-conservation
//! check (Appendix B.4 item 2). `multibody` runs the vendored Unitree G1
//! fixture's real joint tree through the Articulated-Body Algorithm. Without
//! `--floating` the pelvis is bolted to the world. With `--floating` it is a
//! free base on a sphere ground plane (ADR 0010), driven by a clamped PD
//! stance hold, or by zero torque with `--zero-torque`. Neither is a standing
//! or walking claim. The floating verdict is only whether the pelvis stays
//! above the plane. A full qualification run (actuator step response, foot
//! slip, push recovery, controlled impact) is not implemented here.

use super::{EXIT_FAILURE, EXIT_USAGE};
use robot_model::joint::JointKind;
use robot_model::profile::RobotProfile;
use robot_sim::articulated::{ArticulatedBody, MultibodyState};
use robot_sim::contact::GroundContact;
use robot_sim::pendulum::{Pendulum, PendulumParams, PendulumState};
use robot_types::time::{ClockId, Duration, Timestamp};
use robot_viewer::pose::{FrameDump, PoseFrame};
use std::path::Path;

/// Relative energy drift this benchmark tolerates before reporting failure.
/// Loose enough to pass on any reasonable step count while still catching
/// a broken integrator (e.g. plain explicit Euler on this system drifts by
/// orders of magnitude more over the same run).
const MAX_RELATIVE_ENERGY_DRIFT: f64 = 0.05;

/// Runs the undamped pendulum for `steps` steps of `dt_seconds` each, with
/// zero actuator torque, and reports the worst relative energy drift from
/// the initial value. Returns the process exit code.
pub fn qualify(steps: u32, dt_seconds: f64) -> i32 {
    if let Err(message) = check_timing(steps, dt_seconds) {
        eprintln!("error: {message}");
        return EXIT_USAGE;
    }
    let params = PendulumParams {
        length: 1.0,
        mass: 1.0,
        gravity: 9.81,
        damping: 0.0,
    };
    let mut pendulum = Pendulum::new(
        params,
        PendulumState {
            theta: 1.0,
            omega: 0.0,
        },
    );
    let initial_energy = pendulum.energy();
    let dt = Duration::from_secs_f64(dt_seconds);

    let mut worst_drift = 0.0_f64;
    for step in 0..steps {
        if let Err(e) = pendulum.step(0.0, dt) {
            eprintln!("error: step {step}: {e}");
            return EXIT_FAILURE;
        }
        let drift = ((pendulum.energy() - initial_energy) / initial_energy).abs();
        worst_drift = worst_drift.max(drift);
    }

    println!("pendulum qualification: {steps} steps at {dt_seconds} s");
    println!("  initial energy:          {initial_energy:.6} J");
    println!("  final energy:            {:.6} J", pendulum.energy());
    println!("  worst relative drift:    {worst_drift:.6}");
    println!("  tolerance:               {MAX_RELATIVE_ENERGY_DRIFT:.6}");

    if worst_drift <= MAX_RELATIVE_ENERGY_DRIFT {
        println!("  result:                  PASS");
        0
    } else {
        println!("  result:                  FAIL (energy drift exceeds tolerance)");
        EXIT_FAILURE
    }
}

/// Ground plane used by `multibody --floating`: a penalty model whose
/// stiffness and damping were chosen for the G1's ~35 kg, with sphere
/// collision shapes only (see `robot_sim::contact`). Not tuned for balance.
const FLOATING_GROUND: GroundContact = GroundContact {
    stiffness: 2.0e4,
    normal_damping: 200.0,
    friction: 0.8,
    friction_regularization: 1e-3,
    ground_height: 0.0,
};

/// PD gains for the `multibody --floating` stance: a hold toward the zero
/// posture with the torque clamped to each joint's effort limit. This is a
/// supported-stance qualification, not balance control: there is no feedback
/// on the pelvis or the centre of mass. `STANCE_KD` is an upper bound; each
/// joint's damping is further capped by its inertia (see
/// `ArticulatedBody::stable_damping`).
const STANCE_KP: f64 = 300.0;
const STANCE_KD: f64 = 30.0;
/// Clearance (m) between the lowest contact sphere's bottom and the plane at
/// t = 0. The pelvis height is derived from this, not hard-coded.
const STANCE_CLEARANCE: f64 = 0.005;
/// Duration (s) that the floating-base verdict covers. `robot sim multibody`
/// defaults its step count to this divided by `dt`.
pub const QUALIFICATION_SECONDS: f64 = 2.0;
/// Time (s) allowed for the stance to settle before the energy bound applies.
const SETTLE_SECONDS: f64 = 0.5;
/// Largest allowed drop (m) of the pelvis below its start height.
const PELVIS_DROP_TOLERANCE: f64 = 0.02;
/// Deepest allowed penetration (m) of the lowest contact sphere.
const CONTACT_PENETRATION_TOLERANCE: f64 = 0.01;
/// Largest allowed kinetic energy (J) after `SETTLE_SECONDS`.
const SETTLED_KINETIC_ENERGY_BOUND: f64 = 0.5;
/// Exit code for a floating run too short to judge.
const EXIT_INCONCLUSIVE: i32 = 3;
/// Largest step count the CLI accepts, given or derived from `dt`. A run at
/// this count takes minutes of wall time, so anything above it is a usage error.
pub const MAX_STEPS: u32 = 10_000_000;

/// Checks a timestep: finite and greater than zero, and representable as a
/// nanosecond `Duration` to within 0.1 %. `Duration::from_secs_f64` rounds and
/// saturates, so without this check `1e-12` would silently become a zero step.
pub fn validate_dt(dt_seconds: f64) -> Result<f64, String> {
    if !dt_seconds.is_finite() || dt_seconds <= 0.0 {
        return Err(format!(
            "invalid dt {dt_seconds}: expected a finite number of seconds greater than zero"
        ));
    }
    let quantized = Duration::from_secs_f64(dt_seconds).as_secs_f64();
    if (quantized - dt_seconds).abs() > dt_seconds * 1e-3 {
        return Err(format!(
            "invalid dt {dt_seconds}: not representable at nanosecond resolution"
        ));
    }
    Ok(dt_seconds)
}

/// Checks a step count: at least one and at most [`MAX_STEPS`].
pub fn validate_steps(steps: u32) -> Result<u32, String> {
    if steps == 0 {
        return Err("invalid steps 0: expected at least one step".to_owned());
    }
    if steps > MAX_STEPS {
        return Err(format!(
            "invalid steps {steps}: above the cap of {MAX_STEPS}"
        ));
    }
    Ok(steps)
}

/// Checks a timestep and a step count together before a run starts.
pub fn check_timing(steps: u32, dt_seconds: f64) -> Result<(), String> {
    validate_dt(dt_seconds)?;
    validate_steps(steps)?;
    Ok(())
}

/// Step count that covers `QUALIFICATION_SECONDS` at `dt_seconds`, rounded up
/// so the run is never shorter than the qualification duration. The count is
/// checked before it is converted to `u32`, so a tiny timestep is an error
/// rather than a saturated count.
pub fn default_steps(dt_seconds: f64) -> Result<u32, String> {
    let dt = validate_dt(dt_seconds)?;
    let steps = (QUALIFICATION_SECONDS / dt).ceil();
    if steps > f64::from(MAX_STEPS) {
        return Err(format!(
            "{QUALIFICATION_SECONDS} s at dt {dt} s needs {steps} steps, above the cap of {MAX_STEPS}; use a larger dt or pass --steps"
        ));
    }
    Ok((steps as u32).max(1))
}

/// Runs the vendored Unitree G1 through `robot_sim::articulated`'s
/// Articulated-Body Algorithm for `steps` steps of `dt_seconds` each, printing
/// a progress trace. Returns the process exit code.
///
/// `floating = false` bolts the pelvis to world (fixed base, no ground
/// contact). That path is unchanged and reports PASS when every state is
/// finite. `floating = true` uses the floating-base G1 with sphere ground
/// contact and is judged by `judge_stance`. The default is a clamped PD stance
/// hold (see `ArticulatedBody::stance_torques`). `zero_torque = true` applies
/// no torque, as the collapse reference. Neither is a balance claim. A
/// floating run shorter than `QUALIFICATION_SECONDS` is INCONCLUSIVE.
///
/// `out = Some(dir)` writes one pose frame per step to `dir` (ADR 0009). It is
/// only valid with `floating = true`. The dump takes the directory's lock and
/// checks its output paths before any step runs. A run that fails before
/// `finish` leaves the directory's earlier committed run intact.
#[cfg(test)]
fn multibody(steps: u32, dt_seconds: f64, floating: bool, zero_torque: bool) -> i32 {
    multibody_to(steps, dt_seconds, floating, zero_torque, None)
}

/// [`multibody`] with an optional frame directory.
pub fn multibody_to(
    steps: u32,
    dt_seconds: f64,
    floating: bool,
    zero_torque: bool,
    out: Option<&Path>,
) -> i32 {
    if out.is_some_and(|dir| dir.as_os_str().is_empty()) {
        eprintln!("error: --out needs a directory path, not an empty string");
        return EXIT_USAGE;
    }
    if out.is_some() && !floating {
        eprintln!("error: --out needs --floating: a fixed base has no pelvis pose to record");
        return EXIT_USAGE;
    }
    if let Err(message) = check_timing(steps, dt_seconds) {
        eprintln!("error: {message}");
        return EXIT_USAGE;
    }
    if out.is_some() && u64::from(steps) > robot_viewer::pose::MAX_FRAME_STEPS {
        eprintln!(
            "error: --out holds at most {} steps (six-digit SVG names), got {steps}",
            robot_viewer::pose::MAX_FRAME_STEPS
        );
        return EXIT_USAGE;
    }
    if out.is_some() {
        let dt = Duration::from_secs_f64(dt_seconds);
        let dt_nanos = dt.nanos();
        if let Some(total_time) = dt_nanos.checked_mul(i64::from(steps)) {
            if total_time > robot_viewer::pose::MAX_TIMESTAMP_NS {
                eprintln!(
                    "error: run time {total_time} ns exceeds maximum allowed {} ns",
                    robot_viewer::pose::MAX_TIMESTAMP_NS
                );
                return EXIT_USAGE;
            }
        } else {
            eprintln!("error: run time overflow");
            return EXIT_USAGE;
        }
    }
    let profile = if floating {
        robot_model::fixtures::unitree_g1_29dof_floating_base()
    } else {
        robot_model::fixtures::unitree_g1_29dof()
    };
    let profile = match profile {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: could not build the vendored G1 fixture: {e}");
            return EXIT_FAILURE;
        }
    };
    let body = ArticulatedBody::from_profile(&profile);
    let gravity = profile.gravity.acceleration();
    let dt = Duration::from_secs_f64(dt_seconds);
    let dt_nanos = dt.nanos();
    let mut frames = match out {
        Some(dir) => match open_frames(dir, &profile, body.position_width()) {
            Ok(opened) => Some(opened),
            Err(message) => {
                eprintln!("error: {message}");
                return EXIT_FAILURE;
            }
        },
        None => None,
    };
    let out_dir = out.map(Path::to_path_buf);

    // A small displacement from the zero posture, not a claim about any
    // real G1 standing pose: this is a demonstration of the solver, not
    // a balance controller. The floating base starts at an identity
    // quaternion (q[3] = w = 1) at the origin.
    let mut q = if floating {
        vec![0.0; body.position_width()]
    } else {
        vec![0.01; body.position_width()]
    };
    if floating {
        q[3] = 1.0;
        // Place the pelvis so the lowest contact sphere's bottom sits
        // STANCE_CLEARANCE above the plane. The height is measured from the
        // model, not assumed.
        let lowest_at_origin = match body.lowest_contact_height(&q) {
            Ok(Some(h)) => h,
            Ok(None) => {
                eprintln!("error: the floating G1 has no contact spheres");
                return EXIT_FAILURE;
            }
            Err(e) => {
                eprintln!("error: {e}");
                return EXIT_FAILURE;
            }
        };
        q[2] = STANCE_CLEARANCE - lowest_at_origin;
    }
    let mut state = MultibodyState {
        q,
        qd: vec![0.0; body.dof()],
    };
    let stance_kd = match body.stable_damping(&state.q, STANCE_KD, dt_seconds) {
        Ok(kd) => kd,
        Err(e) => {
            eprintln!("error: {e}");
            return EXIT_FAILURE;
        }
    };
    let start_pelvis = state.q[2];
    let mut tau = vec![0.0; body.dof()];

    let mode = match (floating, zero_torque) {
        (true, false) => "floating base, PD stance hold, sphere ground contact",
        (true, true) => "floating base, zero torque (collapse reference), sphere ground contact",
        _ => "fixed base, no ground contact",
    };
    println!("G1 multibody simulation ({mode}): {steps} steps at {dt_seconds} s");
    println!("  degrees of freedom:      {}", body.dof());
    if floating {
        println!(
            "  contact shapes:          {} sphere, {} not modelled",
            body.contact_sphere_count(),
            body.unmodelled_collision_shape_count()
        );
    }

    let report_every = (steps / 10).max(1);
    let mut min_pelvis = f64::INFINITY;
    let mut min_pelvis_time = 0.0;
    let mut min_contact = f64::INFINITY;
    let mut max_settled_energy: Option<f64> = None;
    for step in 0..steps {
        if floating && !zero_torque {
            if let Err(e) =
                body.stance_torques(&state.q, &state.qd, STANCE_KP, &stance_kd, &mut tau)
            {
                eprintln!("error: step {step}: {e}");
                return EXIT_FAILURE;
            }
        }
        let stepped = if floating {
            state.step_with_ground(&body, &tau, gravity, &FLOATING_GROUND, dt)
        } else {
            state.step(&body, &tau, gravity, dt)
        };
        if let Err(e) = stepped {
            eprintln!("error: step {step}: {e}");
            return EXIT_FAILURE;
        }
        if let Some((dump, layout)) = frames.as_mut() {
            let time_ns = dt_nanos * (i64::from(step) + 1);
            let frame = pose_frame(u64::from(step) + 1, time_ns, &state.q, layout);
            if let Err(e) = dump.push(&frame) {
                eprintln!("error: step {step}: {e}");
                return EXIT_FAILURE;
            }
        }
        if floating {
            let t = (step + 1) as f64 * dt_seconds;
            if state.q[2] < min_pelvis {
                min_pelvis = state.q[2];
                min_pelvis_time = t;
            }
            match body.lowest_contact_height(&state.q) {
                Ok(Some(h)) => min_contact = min_contact.min(h),
                Ok(None) => {}
                Err(e) => {
                    eprintln!("error: step {step}: {e}");
                    return EXIT_FAILURE;
                }
            }
            if t >= SETTLE_SECONDS {
                match body.kinetic_energy(&state.q, &state.qd) {
                    Ok(e) => max_settled_energy = Some(max_settled_energy.map_or(e, |m| m.max(e))),
                    Err(e) => {
                        eprintln!("error: step {step}: {e}");
                        return EXIT_FAILURE;
                    }
                }
            }
        }
        if step % report_every == 0 || step + 1 == steps {
            let energy = match body.kinetic_energy(&state.q, &state.qd) {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("error: step {step}: {e}");
                    return EXIT_FAILURE;
                }
            };
            let t = (step + 1) as f64 * dt_seconds;
            if floating {
                println!(
                    "  t={t:>7.3}s  kinetic_energy={energy:>10.4} J  pelvis_height={:.4} m",
                    state.q[2]
                );
            } else {
                let max_q = state.q.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                println!("  t={t:>7.3}s  kinetic_energy={energy:>10.4} J  max|q|={max_q:.4} rad");
            }
        }
    }

    let finite = state.q.iter().chain(&state.qd).all(|v| v.is_finite());
    if !finite {
        println!("  result:                  FAIL (non-finite state)");
        return EXIT_FAILURE;
    }
    if !floating {
        println!("  result:                  PASS (ran to completion, finite throughout)");
        return 0;
    }
    if let (Some((dump, _)), Some(dir)) = (frames.take(), out_dir) {
        let written = match dump.finish() {
            Ok(written) => written,
            Err(e) => {
                eprintln!("error: {e}");
                return EXIT_FAILURE;
            }
        };
        println!(
            "  frames:                  {written} steps committed to {}",
            dir.display()
        );
    }
    judge_stance(&StanceTrace {
        duration: steps as f64 * dt_seconds,
        start_pelvis,
        min_pelvis,
        min_pelvis_time,
        min_contact,
        max_settled_energy,
    })
}

/// Where the floating base and each single-DOF joint sit in `q`, in the
/// order `ArticulatedBody::from_profile` lays them out.
struct FrameLayout {
    joint_names: Vec<String>,
    joint_slots: Vec<usize>,
}

/// Works out which `q` slots a frame reads. The floating base must be the
/// first slot, so the pelvis pose is `q[0..7]` (three translation, then
/// `w, x, y, z`). Every other actuated joint is single-DOF and contributes one
/// angle. The total width must equal the body's, so a mismatch is an error
/// before any step runs.
fn frame_layout(profile: &RobotProfile, position_width: usize) -> Result<FrameLayout, String> {
    let mut offset = 0;
    let mut has_base = false;
    let mut joint_names = Vec::new();
    let mut joint_slots = Vec::new();
    for joint in profile.actuated_joints() {
        match joint.kind {
            JointKind::Free => {
                if offset != 0 {
                    return Err("the floating base must be the first position slot".to_owned());
                }
                has_base = true;
                offset += 7;
            }
            JointKind::Revolute | JointKind::Continuous | JointKind::Prismatic => {
                joint_names.push(joint.name.as_str().to_owned());
                joint_slots.push(offset);
                offset += 1;
            }
            JointKind::Fixed => {}
        }
    }
    if !has_base {
        return Err("--out needs a floating base, and this profile has none".to_owned());
    }
    if offset != position_width {
        return Err(format!(
            "joint layout is {offset} wide, the body expects {position_width}"
        ));
    }
    Ok(FrameLayout {
        joint_names,
        joint_slots,
    })
}

/// Opens the frame dump and its layout. This runs before any step, so an
/// unwritable `--out` directory fails before the simulation starts.
fn open_frames(
    dir: &Path,
    profile: &RobotProfile,
    position_width: usize,
) -> Result<(FrameDump, FrameLayout), String> {
    let layout = frame_layout(profile, position_width)?;
    let dump = FrameDump::create(dir, &layout.joint_names).map_err(|e| e.to_string())?;
    Ok((dump, layout))
}

/// The frame for step `step` (1-based), stamped `time_ns`, read from `q`.
fn pose_frame(step: u64, time_ns: i64, q: &[f64], layout: &FrameLayout) -> PoseFrame {
    PoseFrame {
        step,
        timestamp: Timestamp::sampling(time_ns, ClockId::Simulation),
        pelvis_position: [q[0], q[1], q[2]],
        pelvis_quaternion: [q[3], q[4], q[5], q[6]],
        joint_angles: layout.joint_slots.iter().map(|&slot| q[slot]).collect(),
    }
}

/// What a floating-base run measured, judged by [`judge_stance`].
struct StanceTrace {
    duration: f64,
    start_pelvis: f64,
    min_pelvis: f64,
    min_pelvis_time: f64,
    min_contact: f64,
    max_settled_energy: Option<f64>,
}

/// Prints the floating-base checks and returns the exit code. A supported
/// stance passes only if the pelvis drop, the contact penetration and the
/// settled kinetic energy are all within their tolerances.
fn judge_stance(trace: &StanceTrace) -> i32 {
    println!(
        "  duration:                {:.6} s (qualification duration {QUALIFICATION_SECONDS:.1} s)",
        trace.duration
    );
    println!("  pelvis start height:     {:.4} m", trace.start_pelvis);
    println!(
        "  pelvis height minimum:   {:.4} m at t={:.3} s",
        trace.min_pelvis, trace.min_pelvis_time
    );
    println!("  lowest contact minimum:  {:.4} m", trace.min_contact);
    match trace.max_settled_energy {
        Some(energy) => println!("  max kinetic energy after {SETTLE_SECONDS:.1} s: {energy:.4} J"),
        None => println!("  max kinetic energy after {SETTLE_SECONDS:.1} s: n/a (no samples)"),
    }
    if trace.duration + 1e-9 < QUALIFICATION_SECONDS {
        println!("  result:                  INCONCLUSIVE (run is shorter than the qualification duration)");
        return EXIT_INCONCLUSIVE;
    }
    let pelvis_held = trace.min_pelvis >= trace.start_pelvis - PELVIS_DROP_TOLERANCE;
    let contact_held = trace.min_contact >= -CONTACT_PENETRATION_TOLERANCE;
    let energy_bounded = trace
        .max_settled_energy
        .is_some_and(|e| e <= SETTLED_KINETIC_ENERGY_BOUND);
    println!(
        "  check pelvis drop <= {PELVIS_DROP_TOLERANCE:.3} m:        {}",
        word(pelvis_held)
    );
    println!(
        "  check penetration <= {CONTACT_PENETRATION_TOLERANCE:.3} m:      {}",
        word(contact_held)
    );
    println!(
        "  check settled energy <= {SETTLED_KINETIC_ENERGY_BOUND:.2} J: {}",
        word(energy_bounded)
    );
    if pelvis_held && contact_held && energy_bounded {
        println!("  result:                  PASS (supported stance within tolerances; no balance claim)");
        0
    } else if trace.min_pelvis < FLOATING_GROUND.ground_height {
        println!("  result:                  FAIL (pelvis fell below the ground plane)");
        EXIT_FAILURE
    } else {
        println!("  result:                  FAIL (stance not held within tolerances)");
        EXIT_FAILURE
    }
}

fn word(ok: bool) -> &'static str {
    if ok {
        "PASS"
    } else {
        "FAIL"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_viewer::pose;

    #[test]
    fn a_reasonable_step_count_and_timestep_passes() {
        assert_eq!(qualify(2_000, 0.001), 0);
    }

    #[test]
    fn a_wildly_large_timestep_can_fail() {
        assert_eq!(qualify(2_000, 0.5), EXIT_FAILURE);
    }

    #[test]
    fn multibody_runs_the_real_g1_without_diverging() {
        assert_eq!(multibody(200, 0.0005, false, false), 0);
    }

    #[test]
    fn multibody_with_a_floating_base_short_run_is_inconclusive() {
        assert_eq!(multibody(200, 0.0001, true, false), EXIT_INCONCLUSIVE);
    }

    fn trace(min_pelvis: f64, min_contact: f64, energy: Option<f64>, duration: f64) -> StanceTrace {
        StanceTrace {
            duration,
            start_pelvis: 0.8,
            min_pelvis,
            min_pelvis_time: 0.06,
            min_contact,
            max_settled_energy: energy,
        }
    }

    #[test]
    fn a_held_stance_over_the_qualification_duration_passes() {
        assert_eq!(judge_stance(&trace(0.79, -0.005, Some(0.1), 2.0)), 0);
    }

    #[test]
    fn a_run_shorter_than_the_qualification_duration_is_inconclusive_not_a_pass() {
        assert_eq!(
            judge_stance(&trace(0.79, -0.005, Some(0.1), 0.2)),
            EXIT_INCONCLUSIVE
        );
    }

    #[test]
    fn a_pelvis_below_the_plane_fails_with_the_collapse_verdict() {
        assert_eq!(
            judge_stance(&trace(-0.77, -0.04, Some(300.0), 2.0)),
            EXIT_FAILURE
        );
    }

    #[test]
    fn a_pelvis_drop_past_tolerance_fails_even_above_the_plane() {
        assert_eq!(
            judge_stance(&trace(0.70, -0.005, Some(0.1), 2.0)),
            EXIT_FAILURE
        );
    }

    #[test]
    fn unbounded_settled_energy_fails_even_when_the_pelvis_is_held() {
        assert_eq!(
            judge_stance(&trace(0.79, -0.005, Some(8.0), 2.0)),
            EXIT_FAILURE
        );
    }

    #[test]
    fn a_missing_settled_energy_sample_fails_rather_than_passing_silently() {
        assert_eq!(judge_stance(&trace(0.79, -0.005, None, 2.0)), EXIT_FAILURE);
    }

    #[test]
    fn a_fine_timestep_derives_the_full_two_second_step_count() {
        assert_eq!(default_steps(0.0001), Ok(20000));
        assert_eq!(default_steps(0.00015), Ok(13334));
    }

    #[test]
    fn default_steps_rejects_timesteps_that_are_not_finite_and_positive() {
        assert!(default_steps(0.0).is_err());
        assert!(default_steps(-0.001).is_err());
        assert!(default_steps(f64::NAN).is_err());
        assert!(default_steps(f64::INFINITY).is_err());
    }

    #[test]
    fn default_steps_rejects_a_count_above_the_cap_instead_of_saturating() {
        assert!(default_steps(1e-7).is_err());
        assert!(default_steps(1e-12).is_err());
        assert!(default_steps(1e-300).is_err());
    }

    #[test]
    fn validate_dt_rejects_a_timestep_that_rounds_to_zero_nanoseconds() {
        assert!(validate_dt(4e-10).is_err());
        assert!(validate_dt(1e-9).is_ok());
    }

    #[test]
    fn validate_steps_rejects_zero_and_counts_above_the_cap() {
        assert!(validate_steps(0).is_err());
        assert_eq!(validate_steps(MAX_STEPS), Ok(MAX_STEPS));
        assert!(validate_steps(MAX_STEPS + 1).is_err());
    }

    #[test]
    fn qualify_and_multibody_refuse_invalid_timing_before_simulating() {
        assert_eq!(qualify(0, 0.001), EXIT_USAGE);
        assert_eq!(qualify(2_000, -1.0), EXIT_USAGE);
        assert_eq!(multibody(0, 0.0001, true, false), EXIT_USAGE);
        assert_eq!(multibody(200, f64::NAN, false, false), EXIT_USAGE);
    }

    fn frame_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("robot-cli-sim-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn out_on_a_fixed_base_is_a_usage_error_and_creates_nothing() {
        let dir = frame_dir("fixed-base");
        assert_eq!(
            multibody_to(200, 0.0001, false, false, Some(&dir)),
            EXIT_USAGE
        );
        assert!(!dir.exists());
    }

    #[test]
    fn multibody_refuses_out_of_range_time_before_creating_dir() {
        let dir = frame_dir("too-much-time");
        // 1e16 ns per step * 2 steps exceeds the 2^53 ns timestamp limit.
        assert_eq!(multibody_to(2, 1e7, true, false, Some(&dir)), EXIT_USAGE);
        assert!(!dir.exists());
    }

    #[test]
    fn out_over_the_frame_cap_is_a_usage_error_before_creating_dir() {
        let dir = frame_dir("over-frame-cap");
        assert_eq!(
            multibody_to(1_000_000, 0.0001, true, false, Some(&dir)),
            EXIT_USAGE
        );
        assert!(!dir.exists());
    }

    #[test]
    fn an_empty_out_path_is_a_usage_error_and_writes_nothing() {
        assert_eq!(
            multibody_to(200, 0.0001, true, false, Some(Path::new(""))),
            EXIT_USAGE
        );
    }

    #[test]
    fn out_writes_one_frame_per_step_and_reads_them_back_even_when_inconclusive() {
        let dir = frame_dir("one-per-step");
        // A 200-step floating run is shorter than the qualification duration,
        // so it reports INCONCLUSIVE, but the frames are still written.
        assert_eq!(
            multibody_to(200, 0.0001, true, false, Some(&dir)),
            EXIT_INCONCLUSIVE
        );
        let log = robot_viewer::csv::read_file(&dir.join(pose::CSV_FILE)).unwrap();
        assert_eq!(log.frames.len(), 200);
        assert_eq!(log.frames.last().unwrap().step, 200);
        assert_eq!(log.frames.last().unwrap().timestamp.nanos(), 200 * 100_000);
        assert_eq!(
            std::fs::read_dir(dir.join(pose::SVG_DIR)).unwrap().count(),
            200
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_repeated_run_with_the_same_settings_is_byte_identical() {
        let a = frame_dir("repeat-a");
        let b = frame_dir("repeat-b");
        multibody_to(200, 0.0001, true, false, Some(&a));
        multibody_to(200, 0.0001, true, false, Some(&b));
        for file in [pose::CSV_FILE, pose::JSONL_FILE] {
            assert_eq!(
                std::fs::read(a.join(file)).unwrap(),
                std::fs::read(b.join(file)).unwrap(),
                "{file} differs between identical runs"
            );
        }
        let _ = std::fs::remove_dir_all(&a);
        let _ = std::fs::remove_dir_all(&b);
    }

    #[test]
    fn an_unwritable_out_directory_fails_before_any_step() {
        let blocker = frame_dir("blocker");
        std::fs::create_dir_all(&blocker).unwrap();
        let file_in_the_way = blocker.join("not-a-dir");
        std::fs::write(&file_in_the_way, b"x").unwrap();
        assert_eq!(
            multibody_to(200, 0.0001, true, false, Some(&file_in_the_way.join("out"))),
            EXIT_FAILURE
        );
        assert!(!file_in_the_way.join("out").exists());
        let _ = std::fs::remove_dir_all(&blocker);
    }
}

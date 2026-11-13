//! The humanoid tasks of SPECS.md Section 6's `robot-tasks` row:
//! tracking, balance, reach, footwork, boxing and perturbation.
//!
//! Each declares its observation (privileged entries marked), its reward
//! terms through a [`HumanoidRegistry`], and the distinction Section 9.2
//! requires between termination (a failure: the robot fell, tracking was
//! lost) and truncation (a time limit or the end of a reference).
//!
//! | Task | Target | Completion terms |
//! | --- | --- | --- |
//! | [`StandTask`] | base height | upright, height |
//! | [`PerturbationTask`] | base height, push schedule | upright, height, recovery |
//! | [`TrackingTask`] | a reference trajectory | joint and base tracking |
//! | [`ReachTask`] | a hand target point | hand distance, success |
//! | [`FootworkTask`] | a base velocity command | velocity tracking, alternation |
//! | [`BoxingTask`] | a pad position | scored strikes, guard |
//!
//! Every task also carries the shared regularizers of [`crate::terms`]:
//! power, action rate, joint speed and undesirable contacts.

use robot_types::frames::Vec3;

use crate::humanoid::{
    HumanoidContext, HumanoidRegistry, HumanoidState, HumanoidTask, HumanoidTerm, ObservationEntry,
    ObservationSpec, Status,
};
use crate::terms::{
    action_rate, bad_contacts, base_drift, joint_speed, kernel, power, upright, HEAD, LEFT_FOOT,
    LEFT_HAND, RIGHT_FOOT, RIGHT_HAND,
};

fn regularizers() -> Vec<HumanoidTerm> {
    vec![
        HumanoidTerm {
            name: "power",
            weight: -1e-4,
            compute: power,
        },
        HumanoidTerm {
            name: "action_rate",
            weight: -1e-2,
            compute: action_rate,
        },
        HumanoidTerm {
            name: "joint_speed",
            weight: -1e-4,
            compute: joint_speed,
        },
        HumanoidTerm {
            name: "bad_contacts",
            weight: -1.0,
            compute: bad_contacts,
        },
    ]
}

fn registry(mut terms: Vec<HumanoidTerm>) -> HumanoidRegistry {
    terms.extend(regularizers());
    HumanoidRegistry::new(terms).expect("task term names are distinct")
}

fn proprioception(dof: usize) -> Vec<ObservationEntry> {
    vec![
        ObservationEntry {
            name: "base_rotation",
            width: 4,
            privileged: false,
        },
        ObservationEntry {
            name: "base_angular_velocity",
            width: 3,
            privileged: false,
        },
        ObservationEntry {
            name: "q",
            width: dof,
            privileged: false,
        },
        ObservationEntry {
            name: "qd",
            width: dof,
            privileged: false,
        },
        // An IMU does not measure linear velocity or height directly: only
        // the critic may read them.
        ObservationEntry {
            name: "base_linear_velocity",
            width: 3,
            privileged: true,
        },
        ObservationEntry {
            name: "base_height",
            width: 1,
            privileged: true,
        },
    ]
}

fn write_proprioception(s: &HumanoidState, out: &mut Vec<f64>) {
    let r = s.base_rotation;
    out.extend([r.w, r.x, r.y, r.z]);
    out.extend(s.base_angular_velocity.to_xyz());
    out.extend(&s.q);
    out.extend(&s.qd);
    out.extend(s.base_linear_velocity.to_xyz());
    out.push(s.base_position.z);
}

/// Fall limits shared by every task.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct FallLimits {
    /// Base height below which the robot has fallen, metres.
    pub min_height: f64,
    /// Tilt beyond which the robot has fallen, radians.
    pub max_tilt: f64,
    /// Steps after which the episode is truncated.
    pub max_steps: usize,
}

impl FallLimits {
    fn status(&self, s: &HumanoidState, step: usize) -> Status {
        if s.base_position.z < self.min_height {
            Status::Terminated("base below minimum height")
        } else if s.tilt() > self.max_tilt {
            Status::Terminated("base tilted past limit")
        } else if !s.base_position.is_finite() {
            Status::Terminated("non-finite state")
        } else if step >= self.max_steps {
            Status::Truncated
        } else {
            Status::Running
        }
    }
}

/// Stand still at a height (the humanoid balance task).
pub struct StandTask {
    /// Desired base height.
    pub height: f64,
    /// Fall limits.
    pub limits: FallLimits,
    /// Joint count.
    pub dof: usize,
    registry: HumanoidRegistry,
}

impl StandTask {
    /// A standing task.
    #[must_use]
    pub fn new(dof: usize, height: f64, limits: FallLimits) -> Self {
        Self {
            height,
            limits,
            dof,
            registry: registry(vec![
                HumanoidTerm {
                    name: "upright",
                    weight: 1.0,
                    compute: upright,
                },
                HumanoidTerm {
                    name: "height",
                    weight: 1.0,
                    compute: crate::terms::height,
                },
                HumanoidTerm {
                    name: "base_drift",
                    weight: -0.5,
                    compute: base_drift,
                },
            ]),
        }
    }
}

impl HumanoidTask for StandTask {
    fn id(&self) -> &'static str {
        "stand/1"
    }
    fn observation_spec(&self) -> ObservationSpec {
        ObservationSpec {
            entries: proprioception(self.dof),
        }
    }
    fn observe(&self, s: &HumanoidState, out: &mut Vec<f64>) {
        write_proprioception(s, out);
    }
    fn target(&self, _s: &HumanoidState) -> Vec<f64> {
        vec![self.height]
    }
    fn registry(&self) -> &HumanoidRegistry {
        &self.registry
    }
    fn status(&self, s: &HumanoidState, step: usize) -> Status {
        self.limits.status(s, step)
    }
}

/// A push: a base velocity change at a step.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Push {
    /// Step at which it is applied.
    pub step: usize,
    /// Base velocity change, world frame, m/s.
    pub delta_v: Vec3,
}

/// Stand through a schedule of pushes, and recover.
pub struct PerturbationTask {
    /// The standing task underneath.
    pub stand: StandTask,
    /// The pushes, applied by the environment driver.
    pub pushes: Vec<Push>,
    /// Seconds after a push within which the base must be still again.
    pub recovery_window: f64,
    last_push_time: Option<f64>,
    registry: HumanoidRegistry,
}

fn recovered(ctx: &HumanoidContext<'_>) -> f64 {
    // target: [height, seconds since last push (or -1), window]
    let since = ctx.target.get(1).copied().unwrap_or(-1.0);
    let window = ctx.target.get(2).copied().unwrap_or(1.0);
    if since < 0.0 || since < window {
        0.0
    } else {
        kernel(ctx.next.base_linear_velocity.norm(), 0.2)
    }
}

impl PerturbationTask {
    /// A perturbation task over `pushes`.
    #[must_use]
    pub fn new(stand: StandTask, pushes: Vec<Push>, recovery_window: f64) -> Self {
        Self {
            stand,
            pushes,
            recovery_window,
            last_push_time: None,
            registry: registry(vec![
                HumanoidTerm {
                    name: "upright",
                    weight: 1.0,
                    compute: upright,
                },
                HumanoidTerm {
                    name: "height",
                    weight: 1.0,
                    compute: crate::terms::height,
                },
                HumanoidTerm {
                    name: "recovered",
                    weight: 1.0,
                    compute: recovered,
                },
            ]),
        }
    }

    /// The push due at `step`, if any; records its time.
    pub fn push_at(&mut self, step: usize, time: f64) -> Option<Vec3> {
        let p = self.pushes.iter().find(|p| p.step == step)?;
        self.last_push_time = Some(time);
        Some(p.delta_v)
    }
}

impl HumanoidTask for PerturbationTask {
    fn id(&self) -> &'static str {
        "perturbation/1"
    }
    fn observation_spec(&self) -> ObservationSpec {
        self.stand.observation_spec()
    }
    fn observe(&self, s: &HumanoidState, out: &mut Vec<f64>) {
        write_proprioception(s, out);
    }
    fn target(&self, s: &HumanoidState) -> Vec<f64> {
        let since = self.last_push_time.map_or(-1.0, |t| s.time - t);
        vec![self.stand.height, since, self.recovery_window]
    }
    fn registry(&self) -> &HumanoidRegistry {
        &self.registry
    }
    fn status(&self, s: &HumanoidState, step: usize) -> Status {
        self.stand.limits.status(s, step)
    }
}

/// Track a reference trajectory (joint positions and base position).
pub struct TrackingTask {
    /// Frame times, seconds.
    pub times: Vec<f64>,
    /// Joint references per frame.
    pub joints: Vec<Vec<f64>>,
    /// Base position references per frame.
    pub base: Vec<Vec3>,
    /// Joint error (RMS, rad) beyond which tracking is lost.
    pub max_joint_error: f64,
    /// Base error (m) beyond which tracking is lost.
    pub max_base_error: f64,
    /// Fall limits.
    pub limits: FallLimits,
    registry: HumanoidRegistry,
}

fn joint_tracking(ctx: &HumanoidContext<'_>) -> f64 {
    let n = ctx.next.q.len();
    let err = ctx
        .next
        .q
        .iter()
        .zip(ctx.target)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / n.max(1) as f64;
    kernel(err.sqrt(), 0.25)
}

fn base_tracking(ctx: &HumanoidContext<'_>) -> f64 {
    let n = ctx.next.q.len();
    let Some(b) = ctx.target.get(n..n + 3) else {
        return 0.0;
    };
    kernel(
        (ctx.next.base_position - Vec3::new(b[0], b[1], b[2])).norm(),
        0.2,
    )
}

impl TrackingTask {
    /// A tracking task over a reference.
    #[must_use]
    pub fn new(
        times: Vec<f64>,
        joints: Vec<Vec<f64>>,
        base: Vec<Vec3>,
        limits: FallLimits,
    ) -> Self {
        Self {
            times,
            joints,
            base,
            max_joint_error: 0.5,
            max_base_error: 0.5,
            limits,
            registry: registry(vec![
                HumanoidTerm {
                    name: "joint_tracking",
                    weight: 1.0,
                    compute: joint_tracking,
                },
                HumanoidTerm {
                    name: "base_tracking",
                    weight: 1.0,
                    compute: base_tracking,
                },
                HumanoidTerm {
                    name: "upright",
                    weight: 0.2,
                    compute: upright,
                },
            ]),
        }
    }

    /// The reference linearly interpolated at `t`.
    #[must_use]
    pub fn reference(&self, t: f64) -> (Vec<f64>, Vec3) {
        let k = self
            .times
            .partition_point(|&x| x <= t)
            .clamp(1, self.times.len().max(2) - 1);
        let (a, b) = (k - 1, k.min(self.times.len() - 1));
        let span = (self.times[b] - self.times[a]).max(f64::MIN_POSITIVE);
        let s = ((t - self.times[a]) / span).clamp(0.0, 1.0);
        let q = self.joints[a]
            .iter()
            .zip(&self.joints[b])
            .map(|(x, y)| x + (y - x) * s)
            .collect();
        (q, self.base[a].lerp(self.base[b], s))
    }
}

impl HumanoidTask for TrackingTask {
    fn id(&self) -> &'static str {
        "tracking/1"
    }
    fn observation_spec(&self) -> ObservationSpec {
        let dof = self.joints.first().map_or(0, Vec::len);
        let mut e = proprioception(dof);
        e.push(ObservationEntry {
            name: "reference_q",
            width: dof,
            privileged: false,
        });
        e.push(ObservationEntry {
            name: "reference_phase",
            width: 1,
            privileged: false,
        });
        ObservationSpec { entries: e }
    }
    fn observe(&self, s: &HumanoidState, out: &mut Vec<f64>) {
        write_proprioception(s, out);
        let (q, _) = self.reference(s.time);
        out.extend(q);
        let end = self
            .times
            .last()
            .copied()
            .unwrap_or(1.0)
            .max(f64::MIN_POSITIVE);
        out.push((s.time / end).clamp(0.0, 1.0));
    }
    fn target(&self, s: &HumanoidState) -> Vec<f64> {
        let (mut q, b) = self.reference(s.time);
        q.extend(b.to_xyz());
        q
    }
    fn registry(&self) -> &HumanoidRegistry {
        &self.registry
    }
    fn status(&self, s: &HumanoidState, step: usize) -> Status {
        match self.limits.status(s, step) {
            Status::Running => {}
            other => return other,
        }
        let (q, b) = self.reference(s.time);
        let n = q.len().max(1) as f64;
        let rms =
            (s.q.iter()
                .zip(&q)
                .map(|(a, r)| (a - r).powi(2))
                .sum::<f64>()
                / n)
                .sqrt();
        if rms > self.max_joint_error || (s.base_position - b).norm() > self.max_base_error {
            Status::Terminated("tracking lost")
        } else if s.time >= self.times.last().copied().unwrap_or(0.0) {
            Status::Truncated
        } else {
            Status::Running
        }
    }
}

/// Reach a target point with the right hand.
pub struct ReachTask {
    /// Target, world frame.
    pub target: Vec3,
    /// Distance counted as reached, metres.
    pub tolerance: f64,
    /// Standing height for the balance term.
    pub height: f64,
    /// Fall limits.
    pub limits: FallLimits,
    /// Joint count.
    pub dof: usize,
    registry: HumanoidRegistry,
}

fn hand_distance(ctx: &HumanoidContext<'_>) -> f64 {
    let t = Vec3::new(ctx.target[1], ctx.target[2], ctx.target[3]);
    ctx.next
        .links
        .get(RIGHT_HAND)
        .map_or(0.0, |h| kernel((*h - t).norm(), 0.15))
}

fn reached(ctx: &HumanoidContext<'_>) -> f64 {
    let t = Vec3::new(ctx.target[1], ctx.target[2], ctx.target[3]);
    let tol = ctx.target[4];
    f64::from(u8::from(
        ctx.next
            .links
            .get(RIGHT_HAND)
            .is_some_and(|h| (*h - t).norm() < tol),
    ))
}

impl ReachTask {
    /// A reach task.
    #[must_use]
    pub fn new(dof: usize, target: Vec3, tolerance: f64, height: f64, limits: FallLimits) -> Self {
        Self {
            target,
            tolerance,
            height,
            limits,
            dof,
            registry: registry(vec![
                HumanoidTerm {
                    name: "hand_distance",
                    weight: 1.0,
                    compute: hand_distance,
                },
                HumanoidTerm {
                    name: "reached",
                    weight: 1.0,
                    compute: reached,
                },
                HumanoidTerm {
                    name: "upright",
                    weight: 0.5,
                    compute: upright,
                },
                HumanoidTerm {
                    name: "height",
                    weight: 0.5,
                    compute: crate::terms::height,
                },
            ]),
        }
    }
}

impl HumanoidTask for ReachTask {
    fn id(&self) -> &'static str {
        "reach/1"
    }
    fn observation_spec(&self) -> ObservationSpec {
        let mut e = proprioception(self.dof);
        e.push(ObservationEntry {
            name: "target_in_base",
            width: 3,
            privileged: false,
        });
        ObservationSpec { entries: e }
    }
    fn observe(&self, s: &HumanoidState, out: &mut Vec<f64>) {
        write_proprioception(s, out);
        let rel = s
            .base_rotation
            .conjugate()
            .rotate(self.target - s.base_position);
        out.extend(rel.to_xyz());
    }
    fn target(&self, _s: &HumanoidState) -> Vec<f64> {
        vec![
            self.height,
            self.target.x,
            self.target.y,
            self.target.z,
            self.tolerance,
        ]
    }
    fn registry(&self) -> &HumanoidRegistry {
        &self.registry
    }
    fn status(&self, s: &HumanoidState, step: usize) -> Status {
        self.limits.status(s, step)
    }
}

/// Follow a commanded base velocity with alternating steps.
pub struct FootworkTask {
    /// Command `[vx, vy, yaw_rate]`, base heading frame.
    pub command: [f64; 3],
    /// Standing height.
    pub height: f64,
    /// Fall limits.
    pub limits: FallLimits,
    /// Joint count.
    pub dof: usize,
    registry: HumanoidRegistry,
}

fn velocity_tracking(ctx: &HumanoidContext<'_>) -> f64 {
    let yaw = ctx.next.yaw();
    let v = ctx.next.base_linear_velocity;
    let (c, s) = (yaw.cos(), yaw.sin());
    let (vx, vy) = (c * v.x + s * v.y, -s * v.x + c * v.y);
    kernel((vx - ctx.target[1]).hypot(vy - ctx.target[2]), 0.25)
}

fn yaw_tracking(ctx: &HumanoidContext<'_>) -> f64 {
    let wz = ctx
        .next
        .base_rotation
        .rotate(ctx.next.base_angular_velocity)
        .z;
    kernel(wz - ctx.target[3], 0.25)
}

/// Rewards a foot that just landed, so steps alternate rather than one foot
/// shuffling; zero when the command is to stand.
fn foot_landing(ctx: &HumanoidContext<'_>) -> f64 {
    let moving = ctx.target[1].hypot(ctx.target[2]) + ctx.target[3].abs() > 0.05;
    if !moving {
        return 0.0;
    }
    [LEFT_FOOT, RIGHT_FOOT]
        .iter()
        .filter(|&&f| {
            !ctx.prev.foot_contact.get(f).copied().unwrap_or(true)
                && ctx.next.foot_contact.get(f).copied().unwrap_or(false)
        })
        .count() as f64
}

impl FootworkTask {
    /// A footwork task.
    #[must_use]
    pub fn new(dof: usize, command: [f64; 3], height: f64, limits: FallLimits) -> Self {
        Self {
            command,
            height,
            limits,
            dof,
            registry: registry(vec![
                HumanoidTerm {
                    name: "velocity_tracking",
                    weight: 1.0,
                    compute: velocity_tracking,
                },
                HumanoidTerm {
                    name: "yaw_tracking",
                    weight: 0.5,
                    compute: yaw_tracking,
                },
                HumanoidTerm {
                    name: "foot_landing",
                    weight: 0.2,
                    compute: foot_landing,
                },
                HumanoidTerm {
                    name: "upright",
                    weight: 0.5,
                    compute: upright,
                },
                HumanoidTerm {
                    name: "height",
                    weight: 0.5,
                    compute: crate::terms::height,
                },
            ]),
        }
    }
}

impl HumanoidTask for FootworkTask {
    fn id(&self) -> &'static str {
        "footwork/1"
    }
    fn observation_spec(&self) -> ObservationSpec {
        let mut e = proprioception(self.dof);
        e.push(ObservationEntry {
            name: "command",
            width: 3,
            privileged: false,
        });
        e.push(ObservationEntry {
            name: "foot_contact",
            width: 2,
            privileged: false,
        });
        ObservationSpec { entries: e }
    }
    fn observe(&self, s: &HumanoidState, out: &mut Vec<f64>) {
        write_proprioception(s, out);
        out.extend(self.command);
        out.extend(
            s.foot_contact
                .iter()
                .take(2)
                .map(|c| f64::from(u8::from(*c))),
        );
    }
    fn target(&self, _s: &HumanoidState) -> Vec<f64> {
        vec![
            self.height,
            self.command[0],
            self.command[1],
            self.command[2],
        ]
    }
    fn registry(&self) -> &HumanoidRegistry {
        &self.registry
    }
    fn status(&self, s: &HumanoidState, step: usize) -> Status {
        self.limits.status(s, step)
    }
}

/// Strike a pad and keep a guard: the noncontact boxing drill.
///
/// A strike **scores** when a hand enters the pad's radius moving toward the
/// pad faster than `min_strike_speed`; one entry scores once, and the hand
/// must leave the pad before it can score again. Scoring lives in
/// [`BoxingTask::advance`], separately from the reward, so it can be
/// validated on its own (Section 14 row A-21: "Scoring validated
/// independently").
pub struct BoxingTask {
    /// Pad centre, world frame.
    pub pad: Vec3,
    /// Pad radius, metres.
    pub radius: f64,
    /// Minimum approach speed for a scored strike, m/s.
    pub min_strike_speed: f64,
    /// Guard: the non-striking hand within this distance of the head.
    pub guard_distance: f64,
    /// Standing height.
    pub height: f64,
    /// Fall limits.
    pub limits: FallLimits,
    /// Joint count.
    pub dof: usize,
    /// Strikes scored so far.
    pub score: u32,
    in_pad: [bool; 2],
    registry: HumanoidRegistry,
}

fn approach(ctx: &HumanoidContext<'_>) -> f64 {
    let pad = Vec3::new(ctx.target[1], ctx.target[2], ctx.target[3]);
    [RIGHT_HAND, LEFT_HAND]
        .iter()
        .filter_map(|&h| Some(kernel((*ctx.next.links.get(h)? - pad).norm(), 0.3)))
        .fold(0.0, f64::max)
}

fn guard(ctx: &HumanoidContext<'_>) -> f64 {
    let (Some(head), Some(l), Some(r)) = (
        ctx.next.links.get(HEAD),
        ctx.next.links.get(LEFT_HAND),
        ctx.next.links.get(RIGHT_HAND),
    ) else {
        return 0.0;
    };
    let d = (*l - *head).norm().min((*r - *head).norm());
    f64::from(u8::from(d < ctx.target[5]))
}

fn scored(ctx: &HumanoidContext<'_>) -> f64 {
    // target[6] is the number of strikes that scored on this transition,
    // computed by the task's own scorer before the reward is evaluated.
    ctx.target.get(6).copied().unwrap_or(0.0)
}

impl BoxingTask {
    /// A pad drill.
    #[must_use]
    pub fn new(dof: usize, pad: Vec3, height: f64, limits: FallLimits) -> Self {
        Self {
            pad,
            radius: 0.12,
            min_strike_speed: 2.0,
            guard_distance: 0.3,
            height,
            limits,
            dof,
            score: 0,
            in_pad: [false; 2],
            registry: registry(vec![
                HumanoidTerm {
                    name: "scored",
                    weight: 5.0,
                    compute: scored,
                },
                HumanoidTerm {
                    name: "approach",
                    weight: 0.2,
                    compute: approach,
                },
                HumanoidTerm {
                    name: "guard",
                    weight: 0.2,
                    compute: guard,
                },
                HumanoidTerm {
                    name: "upright",
                    weight: 0.5,
                    compute: upright,
                },
                HumanoidTerm {
                    name: "height",
                    weight: 0.5,
                    compute: crate::terms::height,
                },
            ]),
        }
    }

    /// Strikes that score on the transition `prev -> next`, without
    /// changing state.
    #[must_use]
    pub fn strikes(&self, prev: &HumanoidState, next: &HumanoidState) -> u32 {
        let dt = (next.time - prev.time).max(f64::MIN_POSITIVE);
        let mut n = 0;
        for (k, &h) in [RIGHT_HAND, LEFT_HAND].iter().enumerate() {
            let (Some(a), Some(b)) = (prev.links.get(h), next.links.get(h)) else {
                continue;
            };
            let inside = (*b - self.pad).norm() <= self.radius;
            let toward = ((*a - self.pad).norm() - (*b - self.pad).norm()) / dt;
            if inside && !self.in_pad[k] && toward >= self.min_strike_speed {
                n += 1;
            }
        }
        n
    }
}

impl HumanoidTask for BoxingTask {
    fn id(&self) -> &'static str {
        "boxing-pad/1"
    }
    fn observation_spec(&self) -> ObservationSpec {
        let mut e = proprioception(self.dof);
        e.push(ObservationEntry {
            name: "pad_in_base",
            width: 3,
            privileged: false,
        });
        ObservationSpec { entries: e }
    }
    fn observe(&self, s: &HumanoidState, out: &mut Vec<f64>) {
        write_proprioception(s, out);
        out.extend(
            s.base_rotation
                .conjugate()
                .rotate(self.pad - s.base_position)
                .to_xyz(),
        );
    }
    fn target(&self, _s: &HumanoidState) -> Vec<f64> {
        vec![
            self.height,
            self.pad.x,
            self.pad.y,
            self.pad.z,
            self.radius,
            self.guard_distance,
            0.0,
        ]
    }
    fn registry(&self) -> &HumanoidRegistry {
        &self.registry
    }
    fn status(&self, s: &HumanoidState, step: usize) -> Status {
        self.limits.status(s, step)
    }
    fn advance(&mut self, prev: &HumanoidState, next: &HumanoidState) {
        self.score += self.strikes(prev, next);
        for (k, &h) in [RIGHT_HAND, LEFT_HAND].iter().enumerate() {
            self.in_pad[k] = next
                .links
                .get(h)
                .is_some_and(|p| (*p - self.pad).norm() <= self.radius);
        }
    }
}

/// Steps a boxing task, putting the transition's scored strikes into the
/// reward context (the generic step cannot know them in advance).
pub fn step_boxing(
    task: &mut BoxingTask,
    prev: &HumanoidState,
    prev_action: &[f64],
    action: &[f64],
    next: &HumanoidState,
    step: usize,
) -> (robot_types::transition::RewardTerms, Status) {
    let mut target = task.target(prev);
    target[6] = f64::from(task.strikes(prev, next));
    let terms = task.registry().evaluate(&HumanoidContext {
        prev,
        action,
        prev_action,
        next,
        target: &target,
    });
    task.advance(prev, next);
    (terms, task.status(next, step))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::humanoid::step_task;
    use crate::humanoid::tests::standing;

    fn limits() -> FallLimits {
        FallLimits {
            min_height: 0.4,
            max_tilt: 0.8,
            max_steps: 100,
        }
    }

    #[test]
    fn falling_terminates_and_the_time_limit_truncates() {
        let mut t = StandTask::new(3, 0.75, limits());
        let s = standing(3);
        let (r, st) = step_task(&mut t, &s, &[0.0; 3], &[0.0; 3], &s, 1);
        assert_eq!(st, Status::Running);
        assert!((r.get("upright").unwrap().value - 1.0).abs() < 1e-12);
        let mut fallen = s.clone();
        fallen.base_position.z = 0.2;
        assert!(matches!(t.status(&fallen, 1), Status::Terminated(_)));
        assert_eq!(t.status(&s, 100), Status::Truncated);
    }

    #[test]
    fn the_observation_matches_its_spec_and_hides_privileged_state_from_the_audit() {
        let t = ReachTask::new(3, Vec3::new(0.4, 0.0, 1.0), 0.05, 0.75, limits());
        let mut obs = Vec::new();
        t.observe(&standing(3), &mut obs);
        assert_eq!(obs.len(), t.observation_spec().width());
        assert_eq!(
            t.observation_spec().actor_audit(&["q", "base_height"]),
            ["base_height"]
        );
    }

    #[test]
    fn a_reached_target_scores_and_a_far_one_does_not() {
        let mut t = ReachTask::new(2, Vec3::new(0.2, -0.2, 1.0), 0.05, 0.75, limits());
        let s = standing(2);
        let (r, _) = step_task(&mut t, &s, &[0.0; 2], &[0.0; 2], &s, 1);
        assert_eq!(r.get("reached").unwrap().value, 1.0);
        t.target = Vec3::new(1.0, 1.0, 1.0);
        let (r, _) = step_task(&mut t, &s, &[0.0; 2], &[0.0; 2], &s, 1);
        assert_eq!(r.get("reached").unwrap().value, 0.0);
    }

    #[test]
    fn tracking_interpolates_and_terminates_when_lost() {
        let t = TrackingTask::new(
            vec![0.0, 1.0],
            vec![vec![0.0], vec![1.0]],
            vec![Vec3::new(0.0, 0.0, 0.75), Vec3::new(1.0, 0.0, 0.75)],
            limits(),
        );
        let (q, b) = t.reference(0.25);
        assert!((q[0] - 0.25).abs() < 1e-12 && (b.x - 0.25).abs() < 1e-12);
        let mut s = standing(1);
        s.time = 0.25;
        s.q = vec![0.25];
        s.base_position.x = 0.25;
        assert_eq!(t.status(&s, 1), Status::Running);
        s.q = vec![1.5];
        assert_eq!(t.status(&s, 1), Status::Terminated("tracking lost"));
        s.q = vec![1.0];
        s.time = 1.0;
        s.base_position.x = 1.0;
        assert_eq!(t.status(&s, 1), Status::Truncated);
    }

    #[test]
    fn footwork_rewards_the_commanded_velocity_and_a_landing() {
        let mut t = FootworkTask::new(1, [0.5, 0.0, 0.0], 0.75, limits());
        let mut prev = standing(1);
        prev.foot_contact = vec![false, true];
        let mut next = prev.clone();
        next.base_linear_velocity = Vec3::new(0.5, 0.0, 0.0);
        next.foot_contact = vec![true, true];
        let (r, _) = step_task(&mut t, &prev, &[0.0], &[0.0], &next, 1);
        assert!((r.get("velocity_tracking").unwrap().value - 1.0).abs() < 1e-12);
        assert_eq!(r.get("foot_landing").unwrap().value, 1.0);
    }

    #[test]
    fn a_strike_scores_once_per_entry_and_only_when_fast() {
        let pad = Vec3::new(0.6, -0.2, 1.2);
        let mut t = BoxingTask::new(1, pad, 0.75, limits());
        let mut prev = standing(1);
        prev.links[RIGHT_HAND] = Vec3::new(0.3, -0.2, 1.2);
        let mut hit = prev.clone();
        hit.time = 0.1;
        hit.links[RIGHT_HAND] = Vec3::new(0.55, -0.2, 1.2); // 2.5 m/s toward the pad
        let (r, _) = step_boxing(&mut t, &prev, &[0.0], &[0.0], &hit, 1);
        assert_eq!(r.get("scored").unwrap().value, 1.0);
        assert_eq!(t.score, 1);
        // Staying inside the pad does not score again.
        let mut stay = hit.clone();
        stay.time = 0.2;
        stay.links[RIGHT_HAND] = Vec3::new(0.6, -0.2, 1.2);
        let (r, _) = step_boxing(&mut t, &hit, &[0.0], &[0.0], &stay, 2);
        assert_eq!(r.get("scored").unwrap().value, 0.0);
        // A slow touch does not score.
        let mut t2 = BoxingTask::new(1, pad, 0.75, limits());
        let mut slow = prev.clone();
        slow.time = 1.0;
        slow.links[RIGHT_HAND] = Vec3::new(0.55, -0.2, 1.2);
        assert_eq!(t2.strikes(&prev, &slow), 0);
        t2.advance(&prev, &slow);
        assert_eq!(t2.score, 0);
    }

    #[test]
    fn recovery_is_rewarded_only_after_the_window() {
        let mut t = PerturbationTask::new(
            StandTask::new(1, 0.75, limits()),
            vec![Push {
                step: 5,
                delta_v: Vec3::new(0.5, 0.0, 0.0),
            }],
            0.5,
        );
        assert!(t.push_at(4, 0.4).is_none());
        assert_eq!(t.push_at(5, 0.5), Some(Vec3::new(0.5, 0.0, 0.0)));
        let mut s = standing(1);
        s.time = 0.6;
        let (r, _) = step_task(&mut t, &s, &[0.0], &[0.0], &s, 6);
        assert_eq!(r.get("recovered").unwrap().value, 0.0);
        s.time = 1.2;
        let (r, _) = step_task(&mut t, &s, &[0.0], &[0.0], &s, 12);
        assert!((r.get("recovered").unwrap().value - 1.0).abs() < 1e-12);
    }
}

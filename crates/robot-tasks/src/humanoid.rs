//! The robot-agnostic state humanoid tasks read, the task interface, and
//! the reward registry over it.
//!
//! Every task in this crate is written against [`HumanoidState`] rather
//! than against a simulator's coordinate vectors, so the same reward and
//! termination code runs on `robot-sim`'s world, on a recorded episode, or
//! on hardware telemetry. [`StateExtractor`] builds it from `robot-sim`'s
//! [`World`] using the robot profile's own forward kinematics.
//!
//! SPECS.md Section 9.2: "an asymmetric critic may use simulator-only state;
//! exported actors must pass an observation audit proving no privileged
//! inputs remain." Every task declares its observation as an
//! [`ObservationSpec`] whose entries say whether they are privileged, and
//! [`ObservationSpec::actor_audit`] is that audit.

use robot_model::kinematics::forward_kinematics;
use robot_model::pose::RigidPose;
use robot_model::profile::RobotProfile;
use robot_sim::world::{State, World};
use robot_types::frames::{Quat, Vec3};
use robot_types::transition::{RewardTerm, RewardTerms};

/// What a humanoid task can see of the robot at one instant.
#[derive(Clone, PartialEq, Debug)]
pub struct HumanoidState {
    /// Seconds since episode start.
    pub time: f64,
    /// Base (pelvis) position, world frame.
    pub base_position: Vec3,
    /// Base orientation, world from base.
    pub base_rotation: Quat,
    /// Base linear velocity, world frame.
    pub base_linear_velocity: Vec3,
    /// Base angular velocity, base frame.
    pub base_angular_velocity: Vec3,
    /// Actuated joint positions.
    pub q: Vec<f64>,
    /// Actuated joint velocities.
    pub qd: Vec<f64>,
    /// Applied joint torques.
    pub torque: Vec<f64>,
    /// World positions of the tracked links, in the extractor's order.
    pub links: Vec<Vec3>,
    /// Whether each tracked foot is in contact.
    pub foot_contact: Vec<bool>,
    /// Contacts with the ground by links that should not touch it.
    pub undesired_contacts: usize,
}

impl HumanoidState {
    /// The base's tilt from upright, radians (angle between the base's up
    /// axis and world up).
    #[must_use]
    pub fn tilt(&self) -> f64 {
        let up = self.base_rotation.rotate(Vec3::unit_z());
        up.z.clamp(-1.0, 1.0).acos()
    }

    /// Unit-free yaw of the base about world up.
    #[must_use]
    pub fn yaw(&self) -> f64 {
        let fwd = self.base_rotation.rotate(Vec3::unit_x());
        fwd.y.atan2(fwd.x)
    }
}

/// Builds [`HumanoidState`] from a `robot-sim` world.
#[derive(Clone, Debug)]
pub struct StateExtractor {
    /// The fixed-base profile (the world runs its floating version).
    pub profile: RobotProfile,
    /// Indices into `profile.links` to report positions for.
    pub tracked_links: Vec<usize>,
    /// Indices into `tracked_links` that are feet.
    pub feet: Vec<usize>,
    /// Indices into `profile.links` that must not touch the ground.
    pub forbidden_contact: Vec<usize>,
    /// Height below which a link origin counts as touching the ground.
    pub contact_height: f64,
    /// World height of the ground.
    pub ground_height: f64,
}

impl StateExtractor {
    /// Reads `state` of a floating-base `world` built from this extractor's
    /// profile, with `torque` the last applied actuated torques.
    pub fn extract(
        &self,
        world: &World,
        state: &State,
        torque: &[f64],
        time: f64,
    ) -> Result<HumanoidState, String> {
        let q = world
            .model()
            .qpos_to_external(&state.q)
            .map_err(|e| e.to_string())?;
        let qd = world
            .model()
            .dof_to_external(&state.qd)
            .map_err(|e| e.to_string())?;
        if q.len() < 7 || qd.len() < 6 {
            return Err("the world has no floating base".into());
        }
        let base = RigidPose {
            translation: Vec3::new(q[0], q[1], q[2]),
            rotation: Quat {
                w: q[3],
                x: q[4],
                y: q[5],
                z: q[6],
            }
            .normalized(),
        };
        let joints = q[7..].to_vec();
        let poses = forward_kinematics(&self.profile, &joints).map_err(|e| e.to_string())?;
        let world_of = |l: usize| {
            base.compose(&poses.by_index(l).unwrap_or(RigidPose::IDENTITY))
                .translation
        };
        let links: Vec<Vec3> = self.tracked_links.iter().map(|&l| world_of(l)).collect();
        let touching = |p: Vec3| p.z - self.ground_height < self.contact_height;
        Ok(HumanoidState {
            time,
            base_position: base.translation,
            base_rotation: base.rotation,
            // This world's free-joint velocity: world linear, then body angular.
            base_linear_velocity: Vec3::new(qd[0], qd[1], qd[2]),
            base_angular_velocity: Vec3::new(qd[3], qd[4], qd[5]),
            torque: if torque.len() == joints.len() {
                torque.to_vec()
            } else {
                vec![0.0; joints.len()]
            },
            qd: qd[6..].to_vec(),
            q: joints,
            foot_contact: self
                .feet
                .iter()
                .map(|&k| links.get(k).is_some_and(|p| touching(*p)))
                .collect(),
            undesired_contacts: self
                .forbidden_contact
                .iter()
                .filter(|&&l| touching(world_of(l)))
                .count(),
            links,
        })
    }
}

/// One observation entry.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ObservationEntry {
    /// Name.
    pub name: &'static str,
    /// Width.
    pub width: usize,
    /// Simulator-only (critic) input.
    pub privileged: bool,
}

/// A task's declared observation layout.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ObservationSpec {
    /// Entries, in order.
    pub entries: Vec<ObservationEntry>,
}

impl ObservationSpec {
    /// Total width.
    #[must_use]
    pub fn width(&self) -> usize {
        self.entries.iter().map(|e| e.width).sum()
    }

    /// The width of the actor's observation (privileged entries removed).
    #[must_use]
    pub fn actor_width(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| !e.privileged)
            .map(|e| e.width)
            .sum()
    }

    /// The SPECS.md Section 9.2 audit: the privileged entries among
    /// `actor_inputs` (names an exported actor reads). Empty means the
    /// actor passes.
    #[must_use]
    pub fn actor_audit(&self, actor_inputs: &[&str]) -> Vec<&'static str> {
        self.entries
            .iter()
            .filter(|e| e.privileged && actor_inputs.contains(&e.name))
            .map(|e| e.name)
            .collect()
    }
}

/// Why an episode ended, or that it has not.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Status {
    /// Still running.
    Running,
    /// A terminal failure (bootstrap with zero value).
    Terminated(&'static str),
    /// A time limit or reference end (bootstrap from the value estimate).
    Truncated,
}

/// What a reward term sees.
#[derive(Clone, Copy, Debug)]
pub struct HumanoidContext<'a> {
    /// State before the step.
    pub prev: &'a HumanoidState,
    /// The action applied.
    pub action: &'a [f64],
    /// The action before it.
    pub prev_action: &'a [f64],
    /// State after the step.
    pub next: &'a HumanoidState,
    /// Task-specific target values the term reads (a target point, a
    /// commanded velocity, a reference frame), laid out by the task.
    pub target: &'a [f64],
}

/// One reward term over [`HumanoidContext`].
#[derive(Clone, Copy)]
pub struct HumanoidTerm {
    /// Stable name.
    pub name: &'static str,
    /// Weight.
    pub weight: f64,
    /// The value.
    pub compute: fn(&HumanoidContext<'_>) -> f64,
}

/// A fixed set of humanoid reward terms, evaluated into typed per-term
/// rewards (never a bare float; SPECS.md Section 9.2).
#[derive(Clone)]
pub struct HumanoidRegistry {
    terms: Vec<HumanoidTerm>,
}

impl HumanoidRegistry {
    /// A registry, refusing duplicate names.
    pub fn new(terms: Vec<HumanoidTerm>) -> Result<Self, crate::reward::DuplicateTermName> {
        for (i, t) in terms.iter().enumerate() {
            if terms[..i].iter().any(|u| u.name == t.name) {
                return Err(crate::reward::DuplicateTermName(t.name.to_owned()));
            }
        }
        Ok(Self { terms })
    }

    /// Evaluates every term.
    #[must_use]
    pub fn evaluate(&self, ctx: &HumanoidContext<'_>) -> RewardTerms {
        let mut out = RewardTerms::new();
        for t in &self.terms {
            out.push(RewardTerm {
                name: t.name.to_owned(),
                value: (t.compute)(ctx),
                weight: t.weight,
                is_style_term: false,
            })
            .expect("names were checked unique at registry construction");
        }
        out
    }

    /// Term names, in order.
    #[must_use]
    pub fn names(&self) -> Vec<&'static str> {
        self.terms.iter().map(|t| t.name).collect()
    }
}

/// A humanoid task: observation, reward, status and target.
pub trait HumanoidTask {
    /// Stable task name and version, e.g. `"stand/1"`.
    fn id(&self) -> &'static str;
    /// The observation layout.
    fn observation_spec(&self) -> ObservationSpec;
    /// Writes the observation of `s` (full layout, privileged included).
    fn observe(&self, s: &HumanoidState, out: &mut Vec<f64>);
    /// The task's current target values, for the reward context.
    fn target(&self, s: &HumanoidState) -> Vec<f64>;
    /// The reward terms.
    fn registry(&self) -> &HumanoidRegistry;
    /// Episode status after `step` steps at state `s`.
    fn status(&self, s: &HumanoidState, step: usize) -> Status;
    /// Per-step bookkeeping after a transition (scoring, phase changes).
    fn advance(&mut self, _prev: &HumanoidState, _next: &HumanoidState) {}
}

/// Steps the reward and status of `task` for one transition.
pub fn step_task(
    task: &mut dyn HumanoidTask,
    prev: &HumanoidState,
    prev_action: &[f64],
    action: &[f64],
    next: &HumanoidState,
    step: usize,
) -> (RewardTerms, Status) {
    let target = task.target(prev);
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
pub(crate) mod tests {
    use super::*;

    /// A standing state for tests: pelvis at 0.75 m, upright, still.
    pub(crate) fn standing(dof: usize) -> HumanoidState {
        HumanoidState {
            time: 0.0,
            base_position: Vec3::new(0.0, 0.0, 0.75),
            base_rotation: Quat::IDENTITY,
            base_linear_velocity: Vec3::ZERO,
            base_angular_velocity: Vec3::ZERO,
            q: vec![0.0; dof],
            qd: vec![0.0; dof],
            torque: vec![0.0; dof],
            links: vec![
                Vec3::new(0.0, 0.1, 0.03),
                Vec3::new(0.0, -0.1, 0.03),
                Vec3::new(0.2, -0.2, 1.0),
                Vec3::new(0.2, 0.2, 1.0),
                Vec3::new(0.0, 0.0, 1.25),
            ],
            foot_contact: vec![true, true],
            undesired_contacts: 0,
        }
    }

    #[test]
    fn the_audit_names_privileged_actor_inputs() {
        let spec = ObservationSpec {
            entries: vec![
                ObservationEntry {
                    name: "q",
                    width: 3,
                    privileged: false,
                },
                ObservationEntry {
                    name: "friction",
                    width: 1,
                    privileged: true,
                },
            ],
        };
        assert_eq!(spec.width(), 4);
        assert_eq!(spec.actor_width(), 3);
        assert!(spec.actor_audit(&["q"]).is_empty());
        assert_eq!(spec.actor_audit(&["q", "friction"]), ["friction"]);
    }

    #[test]
    fn tilt_and_yaw_read_the_base_rotation() {
        let mut s = standing(1);
        assert!(s.tilt().abs() < 1e-12);
        s.base_rotation = Quat::from_rx(robot_types::units::Radians(0.3)).unwrap();
        assert!((s.tilt() - 0.3).abs() < 1e-12);
        s.base_rotation = Quat::from_rz(robot_types::units::Radians(1.0)).unwrap();
        assert!((s.yaw() - 1.0).abs() < 1e-12);
    }
}

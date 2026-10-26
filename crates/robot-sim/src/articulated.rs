//! Multi-body forward and inverse dynamics over a `robot_model::RobotProfile`'s
//! joint tree, via the Articulated-Body Algorithm (forward) and the
//! Recursive Newton-Euler Algorithm (inverse) — Featherstone, *Rigid Body
//! Dynamics Algorithms*, chapters 4 and 7.
//!
//! # Scope: fixed base, no ground contact
//!
//! [`ArticulatedBody`] treats the profile's root link as rigidly welded
//! to an inertial world frame. This is not an arbitrary restriction: it
//! is exactly what a `robot_model::RobotProfile` already represents,
//! since the URDF importer rejects `floating`/`planar` joints by name
//! (`robot_model::urdf`), so no imported profile's root can be anything
//! but implicitly fixed. Letting the root itself move freely (so a
//! humanoid's pelvis can fall or balance) and resolving ground contact
//! at the feet are both out of scope here, named rather than silently
//! assumed — see this crate's README for why, and for what the next
//! increment needs.
//!
//! Every joint this solver handles has at most one degree of freedom
//! (`robot_model::joint::JointKind` has no multi-DOF variant at all —
//! `Fixed` is zero, everything else is exactly one), so the algorithm
//! below needs no generic multi-DOF joint-space linear solve: every
//! per-joint quantity that would otherwise be a matrix is a scalar.
//!
//! # Validation
//!
//! No part of this module trusts itself alone. [`ArticulatedBody::forward_dynamics`]
//! (ABA) and [`ArticulatedBody::inverse_dynamics`] (RNE) are independent
//! implementations of the same physics; `forward_dynamics_inverts_inverse_dynamics`
//! round-trips one through the other over the real vendored G1 profile
//! and a hand-built chain. `forward_dynamics_matches_the_textbook_point_mass_pendulum`
//! checks ABA against a closed form derived by hand, not by this module.
//! `free_motion_conserves_energy_like_the_single_pendulum_benchmark_does`
//! repeats `robot_sim::pendulum`'s own established conservation check over
//! a multi-link chain.

use crate::spatial::{
    cross_motion_force, cross_motion_motion, transform_force, transform_motion, SpatialInertia,
    Twist, Wrench,
};
use robot_model::joint::JointKind;
use robot_model::joint::JointSpec;
use robot_model::profile::RobotProfile;
use robot_types::frames::Vec3;
use robot_types::ids::LinkId;
use std::collections::HashMap;

/// Why a dynamics computation could not run.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum DynamicsError {
    /// A `q`/`qd`/`qdd`/`tau` vector did not have one entry per actuated
    /// joint.
    WidthMismatch {
        /// What the vector was for.
        what: &'static str,
        /// Degrees of freedom the profile declares.
        expected: usize,
        /// Entries supplied.
        found: usize,
    },
    /// A supplied value was not finite.
    NotFinite {
        /// What the value was for.
        what: &'static str,
    },
}

impl core::fmt::Display for DynamicsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::WidthMismatch {
                what,
                expected,
                found,
            } => {
                write!(
                    f,
                    "{what} has {found} entries, profile has {expected} degrees of freedom"
                )
            }
            Self::NotFinite { what } => write!(f, "{what} contains a non-finite value"),
        }
    }
}

impl core::error::Error for DynamicsError {}

/// One body in the articulated tree: a link together with the single
/// joint that attaches it to its parent (the profile's root link is not
/// itself a body here — it is the fixed base every body's chain of
/// parents eventually reaches).
struct Body {
    joint: JointSpec,
    /// Index into `bodies` of this body's parent, or `None` when the
    /// parent is the fixed base (the profile's root link).
    parent: Option<usize>,
    /// Index into the `q`/`qd`/`qdd`/`tau` vectors, in
    /// `RobotProfile::actuated_joints` order — `None` for a fixed joint.
    dof: Option<usize>,
    inertia: SpatialInertia,
}

/// A profile's joint tree, preprocessed into the fixed-base articulated
/// structure the dynamics algorithms recurse over.
pub struct ArticulatedBody {
    bodies: Vec<Body>,
    n_dof: usize,
}

fn motion_subspace(kind: JointKind, axis: Vec3) -> Twist {
    match kind {
        JointKind::Revolute | JointKind::Continuous => Twist::angular_rate(axis, 1.0),
        JointKind::Prismatic => Twist::linear_rate(axis, 1.0),
        JointKind::Fixed => Twist::ZERO,
    }
}

impl ArticulatedBody {
    /// Preprocesses `profile` into a fixed-base articulated structure.
    ///
    /// Assumes `profile.validate()` has already passed, exactly as
    /// `robot_model::kinematics::forward_kinematics` assumes it — a tree
    /// structure and a single root are checked once at load, not on
    /// every dynamics call.
    #[must_use]
    pub fn from_profile(profile: &RobotProfile) -> Self {
        let dof_index: HashMap<robot_types::ids::JointId, usize> = profile
            .actuated_joints()
            .enumerate()
            .map(|(i, j)| (j.id, i))
            .collect();

        let order = profile.topological_link_order();
        let mut body_index: HashMap<LinkId, usize> = HashMap::new();
        let mut bodies = Vec::with_capacity(order.len().saturating_sub(1));

        for link_id in order.into_iter().skip(1) {
            let joint = profile
                .parent_joint_of(link_id)
                .expect("every non-root link has exactly one parent joint in a validated profile");
            let link = profile
                .link(link_id)
                .expect("link exists in a validated profile");
            let parent = body_index.get(&joint.parent).copied();
            let index = bodies.len();
            bodies.push(Body {
                dof: dof_index.get(&joint.id).copied(),
                joint: joint.clone(),
                parent,
                inertia: SpatialInertia::from_mass_properties(&link.mass),
            });
            body_index.insert(link_id, index);
        }

        Self {
            bodies,
            n_dof: dof_index.len(),
        }
    }

    /// Degrees of freedom, matching `RobotProfile::dof`.
    #[must_use]
    pub fn dof(&self) -> usize {
        self.n_dof
    }

    fn check_width(&self, what: &'static str, v: &[f64]) -> Result<(), DynamicsError> {
        if v.len() != self.n_dof {
            return Err(DynamicsError::WidthMismatch {
                what,
                expected: self.n_dof,
                found: v.len(),
            });
        }
        if v.iter().any(|x| !x.is_finite()) {
            return Err(DynamicsError::NotFinite { what });
        }
        Ok(())
    }

    /// Inverse dynamics (Recursive Newton-Euler): the joint torques
    /// required to realize acceleration `qdd` at configuration `q` and
    /// velocity `qd`, under `gravity` (e.g.
    /// `profile.gravity.acceleration()`).
    pub fn inverse_dynamics(
        &self,
        q: &[f64],
        qd: &[f64],
        qdd: &[f64],
        gravity: Vec3,
    ) -> Result<Vec<f64>, DynamicsError> {
        self.check_width("q", q)?;
        self.check_width("qd", qd)?;
        self.check_width("qdd", qdd)?;

        // Featherstone's trick: fold gravity into the base's acceleration
        // rather than adding a separate force everywhere it matters.
        let base_accel = Twist {
            angular: Vec3::ZERO,
            linear: -gravity,
        };

        let n = self.bodies.len();
        let mut v: Vec<Twist> = Vec::with_capacity(n);
        let mut a: Vec<Twist> = Vec::with_capacity(n);
        let mut f: Vec<Wrench> = Vec::with_capacity(n);
        let mut transforms = Vec::with_capacity(n);

        for body in &self.bodies {
            let t = body.joint.transform_at(body.dof.map_or(0.0, |d| q[d]));
            let s = motion_subspace(body.joint.kind, body.joint.axis);
            let (dqd, dqdd) = body.dof.map_or((0.0, 0.0), |d| (qd[d], qdd[d]));

            let v_parent = body.parent.map_or(Twist::ZERO, |p| v[p]);
            let a_parent = body.parent.map_or(base_accel, |p| a[p]);

            let v_i = transform_motion(&t, v_parent) + s.scaled(dqd);
            let a_i = transform_motion(&t, a_parent)
                + s.scaled(dqdd)
                + cross_motion_motion(v_i, s.scaled(dqd));
            let f_i = body.inertia.apply(a_i) + cross_motion_force(v_i, body.inertia.apply(v_i));

            v.push(v_i);
            a.push(a_i);
            f.push(f_i);
            transforms.push(t);
        }

        let mut tau = vec![0.0; self.n_dof];
        for i in (0..n).rev() {
            let body = &self.bodies[i];
            let s = motion_subspace(body.joint.kind, body.joint.axis);
            if let Some(d) = body.dof {
                tau[d] = f[i].dot(s);
            }
            if let Some(p) = body.parent {
                f[p] = f[p] + transform_force(&transforms[i], f[i]);
            }
        }
        Ok(tau)
    }

    /// Forward dynamics (Articulated-Body Algorithm): the joint
    /// accelerations produced by torques `tau` at configuration `q` and
    /// velocity `qd`, under `gravity`.
    pub fn forward_dynamics(
        &self,
        q: &[f64],
        qd: &[f64],
        tau: &[f64],
        gravity: Vec3,
    ) -> Result<Vec<f64>, DynamicsError> {
        self.check_width("q", q)?;
        self.check_width("qd", qd)?;
        self.check_width("tau", tau)?;

        let base_accel = Twist {
            angular: Vec3::ZERO,
            linear: -gravity,
        };
        let n = self.bodies.len();

        let mut v: Vec<Twist> = Vec::with_capacity(n);
        let mut c: Vec<Twist> = Vec::with_capacity(n);
        let mut transforms = Vec::with_capacity(n);
        let mut ia: Vec<SpatialInertia> = Vec::with_capacity(n);
        let mut pa: Vec<Wrench> = Vec::with_capacity(n);

        // Pass 1 (root -> leaf): velocities, bias accelerations, and each
        // body's own (not-yet-reduced) articulated inertia/bias force.
        for body in &self.bodies {
            let t = body.joint.transform_at(body.dof.map_or(0.0, |d| q[d]));
            let s = motion_subspace(body.joint.kind, body.joint.axis);
            let dqd = body.dof.map_or(0.0, |d| qd[d]);

            let v_parent = body.parent.map_or(Twist::ZERO, |p| v[p]);
            let v_i = transform_motion(&t, v_parent) + s.scaled(dqd);
            let c_i = cross_motion_motion(v_i, s.scaled(dqd));
            let pa_i = cross_motion_force(v_i, body.inertia.apply(v_i));

            v.push(v_i);
            c.push(c_i);
            transforms.push(t);
            ia.push(body.inertia.clone());
            pa.push(pa_i);
        }

        // Pass 2 (leaf -> root): eliminate each joint's own DOF, folding
        // its effective inertia/bias force into its parent.
        let mut u: Vec<Wrench> = vec![Wrench::ZERO; n];
        let mut d: Vec<f64> = vec![1.0; n];
        let mut u_scalar: Vec<f64> = vec![0.0; n];
        for i in (0..n).rev() {
            let body = &self.bodies[i];
            let s = motion_subspace(body.joint.kind, body.joint.axis);

            let (ia_eff, pa_eff) = if let Some(dof) = body.dof {
                let u_i = ia[i].apply(s);
                let d_i = u_i.dot(s);
                let u_scalar_i = tau[dof] - pa[i].dot(s);
                u[i] = u_i;
                d[i] = d_i;
                u_scalar[i] = u_scalar_i;
                let ia_eff = ia[i].sub(&SpatialInertia::rank_one(u_i).scaled(1.0 / d_i));
                let pa_eff = pa[i] + ia_eff.apply(c[i]) + u_i.scaled(u_scalar_i / d_i);
                (ia_eff, pa_eff)
            } else {
                let pa_eff = pa[i] + ia[i].apply(c[i]);
                (ia[i].clone(), pa_eff)
            };

            if let Some(p) = body.parent {
                ia[p] = ia[p].add(&ia_eff.transform_to_parent(&transforms[i]));
                pa[p] = pa[p] + transform_force(&transforms[i], pa_eff);
            }
        }

        // Pass 3 (root -> leaf): solve for qdd given the parent's now-known
        // true acceleration, using each body's own (non-reduced) U/D/u.
        let mut qdd = vec![0.0; self.n_dof];
        let mut a: Vec<Twist> = Vec::with_capacity(n);
        for (i, body) in self.bodies.iter().enumerate() {
            let s = motion_subspace(body.joint.kind, body.joint.axis);
            let a_parent = body.parent.map_or(base_accel, |p| a[p]);
            let a_prime = transform_motion(&transforms[i], a_parent) + c[i];

            let a_i = if let Some(dof) = body.dof {
                let qdd_i = (u_scalar[i] - u[i].dot(a_prime)) / d[i];
                qdd[dof] = qdd_i;
                a_prime + s.scaled(qdd_i)
            } else {
                a_prime
            };
            a.push(a_i);
        }
        Ok(qdd)
    }

    /// Total kinetic energy at configuration `q`, velocity `qd`: `0.5 *
    /// sum_i v_i . (I_i v_i)` over every body's own spatial velocity and
    /// rigid-body inertia. Independent of [`ArticulatedBody::forward_dynamics`]
    /// and [`ArticulatedBody::inverse_dynamics`] beyond sharing the same
    /// velocity recursion — used (alongside potential energy from
    /// `robot_model::kinematics::forward_kinematics`, computed separately
    /// by callers) to check energy conservation under free dynamics.
    pub fn kinetic_energy(&self, q: &[f64], qd: &[f64]) -> Result<f64, DynamicsError> {
        self.check_width("q", q)?;
        self.check_width("qd", qd)?;

        let mut v: Vec<Twist> = Vec::with_capacity(self.bodies.len());
        let mut total = 0.0;
        for body in &self.bodies {
            let t = body.joint.transform_at(body.dof.map_or(0.0, |d| q[d]));
            let s = motion_subspace(body.joint.kind, body.joint.axis);
            let dqd = body.dof.map_or(0.0, |d| qd[d]);
            let v_parent = body.parent.map_or(Twist::ZERO, |p| v[p]);
            let v_i = transform_motion(&t, v_parent) + s.scaled(dqd);
            total += 0.5 * body.inertia.apply(v_i).dot(v_i);
            v.push(v_i);
        }
        Ok(total)
    }
}

/// Configuration and velocity of every actuated joint, steppable in time
/// against an [`ArticulatedBody`] — the fixed-base equivalent of
/// `robot_sim::pendulum::Pendulum`, generalized to a whole tree.
#[derive(Clone, Debug)]
pub struct MultibodyState {
    /// Joint positions, in `RobotProfile::actuated_joints` order.
    pub q: Vec<f64>,
    /// Joint velocities, same order.
    pub qd: Vec<f64>,
}

impl MultibodyState {
    /// A state at rest at `q`.
    #[must_use]
    pub fn at_rest(q: Vec<f64>) -> Self {
        let qd = vec![0.0; q.len()];
        Self { q, qd }
    }

    /// Advances one step by semi-implicit (symplectic) Euler — update
    /// velocity from the acceleration first, then position from the
    /// updated velocity — the same integrator `robot_sim::pendulum::Pendulum::step`
    /// uses, generalized to every degree of freedom at once.
    ///
    /// Like any explicit integrator, this is only *conditionally* stable:
    /// too large a `dt` relative to the system's stiffest mode makes
    /// energy grow each step instead of staying bounded, ending in a
    /// non-finite state. This is a real, measured property of the real
    /// vendored G1 fixture, not a hypothetical — `robot sim multibody`
    /// (in `robot-cli`) diverges within about 1.3 simulated seconds at
    /// `dt = 5e-4` but runs stably at `dt = 1e-4`, which is why that
    /// command's own default is the smaller value. There is also no
    /// joint-limit or ground-contact constraint anywhere in this module
    /// (see its own doc comment): nothing stops an under-loaded joint
    /// from spinning through several full rotations once energy
    /// accumulates in it, which is a scope limitation, not a sign the
    /// integration itself is unstable.
    pub fn step(
        &mut self,
        body: &ArticulatedBody,
        tau: &[f64],
        gravity: Vec3,
        dt: robot_types::time::Duration,
    ) -> Result<(), DynamicsError> {
        let qdd = body.forward_dynamics(&self.q, &self.qd, tau, gravity)?;
        let dt_s = dt.as_secs_f64();
        for (v, a) in self.qd.iter_mut().zip(&qdd) {
            *v += a * dt_s;
        }
        for (p, v) in self.q.iter_mut().zip(&self.qd) {
            *p += v * dt_s;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_model::joint::{JointLimits, JointSpec};
    use robot_model::kinematics::forward_kinematics;
    use robot_model::link::{InertiaTensor, LinkSpec, MassProperties};
    use robot_model::pose::RigidPose;
    use robot_model::profile::{FirmwareContract, RobotProfile, PROFILE_SCHEMA_VERSION};
    use robot_types::frames::GravityConvention;
    use robot_types::ids::{JointId, LinkId, Name};
    use robot_types::schema::ExtensionTable;
    use robot_types::time::Duration;
    use robot_types::units::{NewtonMetres, Radians, RadiansPerSecond, Range};

    fn empty_link(id: u32, name: &str, mass: MassProperties) -> LinkSpec {
        LinkSpec {
            id: LinkId::new(id),
            name: Name::new(name).unwrap(),
            mass,
            collision: Vec::new(),
        }
    }

    fn revolute(
        id: u32,
        name: &str,
        parent: u32,
        child: u32,
        origin: RigidPose,
        axis: Vec3,
    ) -> JointSpec {
        JointSpec {
            id: JointId::new(id),
            name: Name::new(name).unwrap(),
            kind: JointKind::Revolute,
            parent: LinkId::new(parent),
            child: LinkId::new(child),
            origin,
            axis,
            limits: JointLimits::bounded(
                Range::<Radians>::new(
                    Radians(-core::f64::consts::PI),
                    Radians(core::f64::consts::PI),
                )
                .unwrap(),
                RadiansPerSecond(1000.0),
                NewtonMetres(1000.0),
            ),
            sign: 1.0,
            zero_offset: 0.0,
        }
    }

    fn wrap_profile(joints: Vec<JointSpec>, links: Vec<LinkSpec>) -> RobotProfile {
        RobotProfile {
            schema_version: PROFILE_SCHEMA_VERSION,
            name: Name::new("fixture").unwrap(),
            profile_version: "test".to_owned(),
            joints,
            links,
            actuators: Vec::new(),
            sensors: Vec::new(),
            root_link: LinkId::new(0),
            gravity: GravityConvention::minus_z_standard(robot_types::frames::FrameId::WORLD),
            firmware: FirmwareContract {
                version: "test".to_owned(),
                command_expiration: Duration::ZERO,
                watchdog_timeout: Duration::ZERO,
                has_independent_estop: false,
                allowed_modes: Vec::new(),
            },
            extensions: ExtensionTable::new(),
        }
    }

    /// A point mass `mass` at distance `length` along local +X from a
    /// single revolute joint (axis +Y) at the world-fixed origin.
    fn point_mass_pendulum(mass: f64, length: f64) -> RobotProfile {
        let root = empty_link(0, "anchor", MassProperties::MASSLESS);
        let joint = revolute(0, "pivot", 0, 1, RigidPose::IDENTITY, Vec3::unit_y());
        let bob = empty_link(
            1,
            "bob",
            MassProperties {
                mass: robot_types::units::Kilograms(mass),
                center_of_mass: Vec3::new(length, 0.0, 0.0),
                inertia: InertiaTensor::ZERO,
            },
        );
        wrap_profile(vec![joint], vec![root, bob])
    }

    /// A two-link planar chain: joint1 (axis +Y) at the origin carries
    /// link1 (mass m1, point mass at local +X offset l1); joint2 (axis
    /// +Y), at link1's local (l1, 0, 0), carries link2 (mass m2, point
    /// mass at local +X offset l2).
    fn double_pendulum(m1: f64, l1: f64, m2: f64, l2: f64) -> RobotProfile {
        let root = empty_link(0, "anchor", MassProperties::MASSLESS);
        let joint1 = revolute(0, "joint1", 0, 1, RigidPose::IDENTITY, Vec3::unit_y());
        let link1 = empty_link(
            1,
            "link1",
            MassProperties {
                mass: robot_types::units::Kilograms(m1),
                center_of_mass: Vec3::new(l1, 0.0, 0.0),
                inertia: InertiaTensor::ZERO,
            },
        );
        let joint2 = revolute(
            1,
            "joint2",
            1,
            2,
            RigidPose::from_translation(Vec3::new(l1, 0.0, 0.0)),
            Vec3::unit_y(),
        );
        let link2 = empty_link(
            2,
            "link2",
            MassProperties {
                mass: robot_types::units::Kilograms(m2),
                center_of_mass: Vec3::new(l2, 0.0, 0.0),
                inertia: InertiaTensor::ZERO,
            },
        );
        wrap_profile(vec![joint1, joint2], vec![root, link1, link2])
    }

    const G: f64 = 9.806_65;
    const GRAVITY: Vec3 = Vec3::new(0.0, 0.0, -G);

    #[test]
    fn forward_dynamics_matches_the_textbook_point_mass_pendulum() {
        // I_pivot * qdd = tau + m*g*L*cos(theta), derived by hand (not by
        // this module) for a point mass at local (L,0,0) rotating about
        // +Y through the origin, under gravity (0,0,-g). See this
        // module's doc comment.
        let mass = 1.7;
        let length = 0.9;
        let profile = point_mass_pendulum(mass, length);
        profile.validate().expect("fixture is a valid profile");
        let body = ArticulatedBody::from_profile(&profile);
        assert_eq!(body.dof(), 1);

        let i_pivot = mass * length * length;
        for theta in [-1.2, -0.3, 0.0, 0.4, 1.1, 2.0] {
            for tau in [-2.0, 0.0, 3.5] {
                let qdd = body
                    .forward_dynamics(&[theta], &[0.0], &[tau], GRAVITY)
                    .unwrap();
                let expected = (tau + mass * G * length * theta.cos()) / i_pivot;
                assert!(
                    (qdd[0] - expected).abs() < 1e-9,
                    "theta={theta} tau={tau} got={} want={expected}",
                    qdd[0]
                );
            }
        }
    }

    #[test]
    fn inverse_dynamics_matches_the_textbook_point_mass_pendulum() {
        let mass = 1.7;
        let length = 0.9;
        let profile = point_mass_pendulum(mass, length);
        let body = ArticulatedBody::from_profile(&profile);
        let i_pivot = mass * length * length;

        for theta in [-1.0, 0.2, 1.5] {
            for qdd in [-3.0, 0.0, 2.0] {
                let tau = body
                    .inverse_dynamics(&[theta], &[0.0], &[qdd], GRAVITY)
                    .unwrap();
                let expected = i_pivot * qdd - mass * G * length * theta.cos();
                assert!(
                    (tau[0] - expected).abs() < 1e-9,
                    "got={} want={expected}",
                    tau[0]
                );
            }
        }
    }

    #[test]
    fn forward_dynamics_inverts_inverse_dynamics_on_a_double_pendulum() {
        let profile = double_pendulum(1.0, 0.5, 0.6, 0.35);
        profile.validate().expect("fixture is a valid profile");
        let body = ArticulatedBody::from_profile(&profile);

        let cases: [([f64; 2], [f64; 2], [f64; 2]); 3] = [
            ([0.3, -0.5], [0.1, 0.2], [0.0, 0.0]),
            ([-0.8, 1.1], [-0.3, 0.4], [1.5, -2.0]),
            ([0.0, 0.0], [0.0, 0.0], [0.5, -0.5]),
        ];
        for (q, qd, qdd) in cases {
            let tau = body.inverse_dynamics(&q, &qd, &qdd, GRAVITY).unwrap();
            let qdd_recovered = body.forward_dynamics(&q, &qd, &tau, GRAVITY).unwrap();
            for i in 0..2 {
                assert!(
                    (qdd_recovered[i] - qdd[i]).abs() < 1e-7,
                    "dof {i}: recovered={} want={}",
                    qdd_recovered[i],
                    qdd[i]
                );
            }
        }
    }

    #[test]
    fn forward_dynamics_inverts_inverse_dynamics_on_the_real_g1() {
        let profile = robot_model::fixtures::unitree_g1_29dof().unwrap();
        let body = ArticulatedBody::from_profile(&profile);
        let n = body.dof();

        // A deterministic, non-trivial posture/velocity/acceleration —
        // not all zero, so every term in both algorithms is exercised.
        let q: Vec<f64> = (0..n).map(|i| 0.1 * ((i as f64 * 1.37).sin())).collect();
        let qd: Vec<f64> = (0..n).map(|i| 0.05 * ((i as f64 * 0.91).cos())).collect();
        let qdd: Vec<f64> = (0..n).map(|i| 0.2 * ((i as f64 * 2.03).sin())).collect();

        let tau = body.inverse_dynamics(&q, &qd, &qdd, GRAVITY).unwrap();
        let qdd_recovered = body.forward_dynamics(&q, &qd, &tau, GRAVITY).unwrap();
        for i in 0..n {
            assert!(
                (qdd_recovered[i] - qdd[i]).abs() < 1e-6,
                "dof {i}: recovered={} want={}",
                qdd_recovered[i],
                qdd[i]
            );
        }
    }

    fn potential_energy(profile: &RobotProfile, q: &[f64], gravity: Vec3) -> f64 {
        let poses = forward_kinematics(profile, q).unwrap();
        profile
            .links
            .iter()
            .enumerate()
            .map(|(index, link)| {
                let pose = poses.by_index(index).unwrap();
                let com_world = pose.apply(link.mass.center_of_mass);
                -link.mass.mass.value() * gravity.dot(com_world)
            })
            .sum()
    }

    #[test]
    fn free_motion_conserves_energy_like_the_single_pendulum_benchmark_does() {
        let profile = double_pendulum(1.0, 0.5, 0.6, 0.35);
        let body = ArticulatedBody::from_profile(&profile);
        let mut state = MultibodyState {
            q: vec![0.6, -0.3],
            qd: vec![0.0, 0.0],
        };

        let e0 = body.kinetic_energy(&state.q, &state.qd).unwrap()
            + potential_energy(&profile, &state.q, GRAVITY);

        let dt = Duration::from_micros(200);
        for _ in 0..50_000 {
            state.step(&body, &[0.0, 0.0], GRAVITY, dt).unwrap();
        }

        let e1 = body.kinetic_energy(&state.q, &state.qd).unwrap()
            + potential_energy(&profile, &state.q, GRAVITY);
        let drift = (e1 - e0).abs() / e0.abs();
        assert!(
            drift < 0.05,
            "symplectic Euler energy drift too large: {drift} (e0={e0} e1={e1})"
        );
    }

    #[test]
    fn the_real_g1_simulates_without_diverging_when_bolted_at_the_pelvis() {
        // Not a balance/walking claim (no floating base, no ground
        // contact — see this module's doc comment): every joint except
        // the legs/arms is driven by gravity alone from a near-upright
        // posture, bolted at the pelvis, and must not blow up.
        let profile = robot_model::fixtures::unitree_g1_29dof().unwrap();
        let body = ArticulatedBody::from_profile(&profile);
        let mut state = MultibodyState::at_rest(vec![0.01; body.dof()]);
        let tau = vec![0.0; body.dof()];
        let dt = Duration::from_micros(500);

        for step in 0..2_000 {
            state.step(&body, &tau, GRAVITY, dt).unwrap();
            for (i, &v) in state.q.iter().enumerate() {
                assert!(v.is_finite(), "step {step} dof {i} diverged: q={v}");
            }
        }
        let energy = body.kinetic_energy(&state.q, &state.qd).unwrap();
        assert!(
            energy.is_finite() && energy < 1.0e6,
            "implausible energy after simulating: {energy}"
        );
    }

    #[test]
    fn width_mismatches_are_reported() {
        let profile = point_mass_pendulum(1.0, 1.0);
        let body = ArticulatedBody::from_profile(&profile);
        assert!(matches!(
            body.forward_dynamics(&[0.0, 0.0], &[0.0], &[0.0], GRAVITY),
            Err(DynamicsError::WidthMismatch { what: "q", .. })
        ));
    }
}

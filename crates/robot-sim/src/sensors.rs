//! Ideal sensors on the simulated robot, the sensor half of
//! `specs/08-model-compilation-and-sensors.md` section 9.
//!
//! Section 9 item 1 asks for the robotics subset of a sensor catalog:
//! joint position and velocity, and an IMU-style accelerometer and gyro.
//! Item 2 asks that the noise injection point be an explicit choice; this
//! module takes Appendix B S-06's stated default: **this module produces
//! noise-free readings, and noise models live in `robot-perception`**
//! (`robot_perception::noise`), applied by the caller at a point it records.
//! A reading here is therefore the ground truth a noise model perturbs, and
//! it carries the simulation time it was sampled at.
//!
//! The IMU sits at a point fixed to any link. Its readings come from that
//! site's world pose at successive samples, by finite differences: angular
//! velocity from the relative rotation of the last two samples, linear
//! acceleration from the last three positions. The accelerometer reports
//! specific force `R' (a - g)` in the site frame, so a resting IMU reads
//! `+g` upward and a free-falling one reads zero. Because the differences
//! are backward, the readings lag the state by up to one sample interval
//! (stated, not hidden), and the first two samples after construction or
//! [`SensorSuite::reset`] carry no IMU reading.

use robot_model::kinematics::forward_kinematics;
use robot_model::pose::RigidPose;
use robot_model::profile::RobotProfile;
use robot_types::frames::{Quat, Vec3};

use crate::world::{State, World};

/// Where the IMU is.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ImuSite {
    /// Index into the fixed-base profile's `links`.
    pub link: usize,
    /// Site position in the link frame.
    pub offset: Vec3,
}

/// One IMU sample.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ImuReading {
    /// Specific force in the site frame, m/s².
    pub specific_force: Vec3,
    /// Angular velocity in the site frame, rad/s.
    pub angular_velocity: Vec3,
}

/// One sample of every sensor.
#[derive(Clone, PartialEq, Debug)]
pub struct SensorReadings {
    /// Simulation time, seconds.
    pub time: f64,
    /// Joint positions, actuated-joint order.
    pub joint_positions: Vec<f64>,
    /// Joint velocities, actuated-joint order.
    pub joint_velocities: Vec<f64>,
    /// The IMU, once three samples exist.
    pub imu: Option<ImuReading>,
}

/// The robot's sensors.
pub struct SensorSuite {
    profile: RobotProfile,
    imu: Option<ImuSite>,
    gravity: Vec3,
    history: Vec<(f64, RigidPose)>,
}

impl SensorSuite {
    /// Sensors for a world built from `profile`'s floating-base version.
    #[must_use]
    pub fn new(profile: RobotProfile, imu: Option<ImuSite>) -> Self {
        let gravity = profile.gravity.acceleration();
        Self {
            profile,
            imu,
            gravity,
            history: Vec::with_capacity(3),
        }
    }

    /// Forgets the IMU history (after a reset of the world).
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Samples every sensor at `state`, time `time` seconds.
    pub fn read(
        &mut self,
        world: &World,
        state: &State,
        time: f64,
    ) -> Result<SensorReadings, String> {
        let q = world
            .model()
            .qpos_to_external(&state.q)
            .map_err(|e| e.to_string())?;
        let qd = world
            .model()
            .dof_to_external(&state.qd)
            .map_err(|e| e.to_string())?;
        let dof = self.profile.dof();
        if q.len() != 7 + dof || qd.len() != 6 + dof {
            return Err("the world is not the floating-base version of this profile".into());
        }
        let joints = q[7..].to_vec();
        let imu = match self.imu {
            None => None,
            Some(site) => {
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
                let poses =
                    forward_kinematics(&self.profile, &joints).map_err(|e| e.to_string())?;
                let link = base.compose(&poses.by_index(site.link).ok_or("IMU link out of range")?);
                let pose = RigidPose {
                    translation: link.apply(site.offset),
                    rotation: link.rotation,
                };
                if self.history.len() == 3 {
                    self.history.remove(0);
                }
                self.history.push((time, pose));
                self.imu_reading()
            }
        };
        Ok(SensorReadings {
            time,
            joint_positions: joints,
            joint_velocities: qd[6..].to_vec(),
            imu,
        })
    }

    fn imu_reading(&self) -> Option<ImuReading> {
        let [(t0, p0), (t1, p1), (t2, p2)] = self.history[..] else {
            return None;
        };
        let (h1, h2) = (t1 - t0, t2 - t1);
        if h1 <= 0.0 || h2 <= 0.0 {
            return None;
        }
        let v1 = (p1.translation - p0.translation) * (1.0 / h1);
        let v2 = (p2.translation - p1.translation) * (1.0 / h2);
        let a = (v2 - v1) * (2.0 / (h1 + h2));
        let r = p2.rotation;
        let omega = (p1.rotation.conjugate() * p2.rotation)
            .normalized()
            .to_rotation_vector()
            * (1.0 / h2);
        Some(ImuReading {
            specific_force: r.conjugate().rotate(a - self.gravity),
            angular_velocity: omega,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::SimOptions;
    use robot_model::fixtures::unitree_g1_29dof;
    use robot_model::usd::with_floating_base;

    fn setup() -> (World, RobotProfile) {
        let fixed = unitree_g1_29dof().unwrap();
        let world = World::from_profile(
            &with_floating_base(fixed.clone()).unwrap(),
            SimOptions::default(),
        );
        (world, fixed)
    }

    fn pelvis(p: &RobotProfile) -> usize {
        p.links
            .iter()
            .position(|l| l.name.as_str() == "pelvis")
            .unwrap()
    }

    #[test]
    fn a_resting_imu_reads_gravity_up_and_no_rotation() {
        let (world, p) = setup();
        let mut s = SensorSuite::new(
            p.clone(),
            Some(ImuSite {
                link: pelvis(&p),
                offset: Vec3::ZERO,
            }),
        );
        let state = world.rest_state();
        assert!(s.read(&world, &state, 0.0).unwrap().imu.is_none());
        s.read(&world, &state, 0.01).unwrap();
        let r = s.read(&world, &state, 0.02).unwrap();
        let imu = r.imu.unwrap();
        assert!(
            (imu.specific_force - Vec3::new(0.0, 0.0, 9.80665)).norm() < 1e-9,
            "{imu:?}"
        );
        assert!(imu.angular_velocity.norm() < 1e-12);
        assert_eq!(r.joint_positions.len(), p.dof());
    }

    #[test]
    fn a_free_falling_imu_reads_no_specific_force() {
        let (mut world, p) = setup();
        let mut s = SensorSuite::new(
            p.clone(),
            Some(ImuSite {
                link: pelvis(&p),
                offset: Vec3::ZERO,
            }),
        );
        let mut state = world.rest_state();
        let tau = vec![0.0; world.nv()];
        let dt = world.options().timestep.as_secs_f64();
        let mut last = None;
        for k in 0..30 {
            world.step(&mut state, &tau).unwrap();
            last = s.read(&world, &state, (k + 1) as f64 * dt).unwrap().imu;
        }
        let f = last.unwrap().specific_force;
        assert!(f.norm() < 0.2, "{f:?}");
    }

    #[test]
    fn encoders_report_the_state_and_a_wrong_world_is_refused() {
        let (world, p) = setup();
        let mut s = SensorSuite::new(p.clone(), None);
        let r = s.read(&world, &world.rest_state(), 0.0).unwrap();
        assert!(r.joint_positions.iter().all(|v| *v == 0.0) && r.imu.is_none());
        let fixed_world = World::from_profile(&p, SimOptions::default());
        assert!(s
            .read(&fixed_world, &fixed_world.rest_state(), 0.0)
            .is_err());
    }
}

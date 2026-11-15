//! The envelope checks of SPECS.md Section 12.2's hazard register that a
//! fixed per-joint limit cannot express, enforced inside the supervisor
//! and outside any learned component.
//!
//! | Hazard (Section 12.2) | Check |
//! | --- | --- |
//! | Actuator overload: "torque-speed envelope violation, thermal accumulation" | the torque each actuator can supply at the measured joint speed, through its transmission, scaled down linearly from `derate_start` of its thermal limit to zero at the limit |
//! | Actuator overload: "saturation duration" | consecutive ticks a joint's command was clamped |
//! | Excess contact energy: "energy budget enforced outside the policy" | per-site energy integrated from measured contact power |
//! | Human enters the area, uncommanded motion: interlocks | every interlock the current mode requires must be reported closed |
//!
//! [`EnvelopeMonitor`] is built from the robot profile with every buffer
//! sized at construction; [`EnvelopeMonitor::measure`] copies the tick's
//! measurements into those buffers, and the supervisor's `check_tick` reads
//! them. No step allocates.

use robot_model::actuator::{ThermalLimits, TorqueSpeedEnvelope, Transmission};
use robot_model::profile::RobotProfile;
use robot_types::ids::ContactSiteId;
use robot_types::mode::Interlock;
use robot_types::units::{Celsius, RadiansPerSecond};

/// Envelope settings.
#[derive(Clone, Debug)]
pub struct EnvelopeConfig {
    /// Fraction of each actuator's thermal limit at which derating starts.
    pub derate_start: f64,
    /// Consecutive clamped ticks that raise a sustained-saturation fault.
    pub max_saturated_ticks: u32,
    /// Energy budget per contact site, joules: `(site, budget)`.
    pub contact_budgets: Vec<(ContactSiteId, f64)>,
}

/// Per-joint actuator data and per-tick measurements.
#[derive(Clone, Debug)]
pub struct EnvelopeMonitor {
    config: EnvelopeConfig,
    actuators: Vec<Option<(TorqueSpeedEnvelope, Transmission, ThermalLimits, usize)>>,
    speeds: Vec<f64>,
    temperatures: Vec<Celsius>,
    saturated: Vec<u32>,
    energy: Vec<f64>,
    closed: Vec<Interlock>,
}

impl EnvelopeMonitor {
    /// Builds from `profile`'s actuated joints and actuators.
    #[must_use]
    pub fn new(profile: &RobotProfile, config: EnvelopeConfig) -> Self {
        let joints: Vec<_> = profile.actuated_joints().collect();
        let actuators = joints
            .iter()
            .map(|j| {
                profile
                    .actuators
                    .iter()
                    .enumerate()
                    .find(|(_, a)| a.joint == j.id)
                    .map(|(i, a)| (a.envelope, a.transmission, a.thermal, i))
            })
            .collect();
        let n = joints.len();
        let sites = config.contact_budgets.len();
        Self {
            actuators,
            speeds: vec![0.0; n],
            temperatures: vec![Celsius(f64::NAN); profile.actuators.len()],
            saturated: vec![0; n],
            energy: vec![0.0; sites],
            closed: Vec::with_capacity(16),
            config,
        }
    }

    /// Records this tick's measurements: joint speeds (actuated-joint
    /// order), actuator temperatures (profile actuator order, may be
    /// shorter), contact power per site over `dt` seconds, and the
    /// interlocks reported closed.
    pub fn measure(
        &mut self,
        speeds: &[f64],
        temperatures: &[Celsius],
        contact_power: &[(ContactSiteId, f64)],
        dt: f64,
        closed: &[Interlock],
    ) {
        for (s, v) in self.speeds.iter_mut().zip(speeds) {
            *s = *v;
        }
        for (t, v) in self.temperatures.iter_mut().zip(temperatures) {
            *t = *v;
        }
        for (site, power) in contact_power {
            if let Some(i) = self
                .config
                .contact_budgets
                .iter()
                .position(|(s, _)| s == site)
            {
                self.energy[i] += power.max(0.0) * dt;
            }
        }
        self.closed.clear();
        for i in closed.iter().take(self.closed.capacity()) {
            self.closed.push(*i);
        }
    }

    /// Resets the contact energy account (a new bout or trial).
    pub fn reset_energy(&mut self) {
        self.energy.iter_mut().for_each(|e| *e = 0.0);
    }

    /// The torque joint `j` may produce now, if it has an actuator.
    #[must_use]
    pub fn available_torque(&self, j: usize) -> Option<f64> {
        let (env, tx, thermal, a) = self.actuators.get(j).copied().flatten()?;
        let motor_speed = tx.joint_velocity_to_motor(RadiansPerSecond(self.speeds[j]));
        let base = tx
            .motor_torque_to_joint(env.available_torque(motor_speed))
            .value()
            .abs();
        let t = self.temperatures.get(a).map_or(f64::NAN, |t| t.value());
        let limit = thermal.limit.value();
        let start =
            thermal.ambient.value() + self.config.derate_start * (limit - thermal.ambient.value());
        // A temperature that is not a finite number — never measured, or a
        // failed thermistor — derates to zero: granting full torque on a
        // missing reading would fail open on exactly the path the thermal
        // envelope exists to close.
        let factor = if !t.is_finite() {
            0.0
        } else if t <= start {
            1.0
        } else if t >= limit {
            0.0
        } else {
            (limit - t) / (limit - start)
        };
        Some(base * factor)
    }

    /// Updates joint `j`'s saturation count; returns it when it reached the
    /// limit.
    pub fn saturation(&mut self, j: usize, clamped: bool) -> Option<u32> {
        let c = &mut self.saturated[j];
        *c = if clamped { c.saturating_add(1) } else { 0 };
        (*c >= self.config.max_saturated_ticks).then_some(*c)
    }

    /// Sites over budget: `(site, energy, budget)`.
    pub fn over_budget(&self) -> impl Iterator<Item = (ContactSiteId, f64, f64)> + '_ {
        self.config
            .contact_budgets
            .iter()
            .zip(&self.energy)
            .filter(|((_, b), e)| **e > *b)
            .map(|((s, b), e)| (*s, *e, *b))
    }

    /// Whether `interlock` was reported closed.
    #[must_use]
    pub fn is_closed(&self, interlock: Interlock) -> bool {
        self.closed.contains(&interlock)
    }
}

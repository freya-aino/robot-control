//! The hazard register of SPECS.md Section 12.2, as checked data.
//!
//! "Maintain a hazard register per robot and per operating mode. Each entry
//! names the hazard, its plausible causes, how it is detected, the control
//! that mitigates it, and the residual risk accepted by whom." And: "A
//! hazard whose only control is 'the policy learned not to do that' is not
//! controlled. Controls MUST be independent of the learned component they
//! constrain."
//!
//! [`HazardRegister::validate`] enforces both: every entry needs a cause,
//! a detection, at least one control, and a named person accepting the
//! residual risk; and every entry needs at least one control whose
//! [`ControlKind`] is not [`ControlKind::Learned`]. [`HazardRegister::ready_for`]
//! is the gate before a mode: a register for that mode exists, validates,
//! and records that the robot-specific analysis was performed, which
//! Section 12.2 requires "before that robot first moves under
//! `hw-freestanding-noncontact`". [`HazardRegister::initial`] is the
//! section's own starting table, explicitly marked as not yet analysed.

use robot_types::mode::OperatingMode;

/// What a control is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ControlKind {
    /// Hardware: e-stop circuit, interlock, padding.
    Hardware,
    /// Firmware: command expiry, watchdog.
    Firmware,
    /// The supervisor or another non-learned software check.
    Software,
    /// A procedure: checklist, operator, area control.
    Procedural,
    /// A learned component's behaviour. Never sufficient alone.
    Learned,
}

/// One control.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Control {
    /// What it is.
    pub kind: ControlKind,
    /// What it does.
    pub description: String,
}

/// One hazard.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Hazard {
    /// The hazard.
    pub name: String,
    /// Plausible causes.
    pub causes: String,
    /// How it is detected.
    pub detection: String,
    /// Mitigating controls.
    pub controls: Vec<Control>,
    /// Who accepted the residual risk (empty: nobody yet).
    pub accepted_by: String,
}

/// A register for one robot in one mode.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct HazardRegister {
    /// Robot profile name and version.
    pub robot: String,
    /// The mode it covers.
    pub mode: OperatingMode,
    /// Whether a robot-specific hazard analysis was performed.
    pub analysis_performed: bool,
    /// Entries.
    pub hazards: Vec<Hazard>,
}

/// Why a register is not acceptable.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum HazardError {
    /// A field is empty.
    Incomplete {
        /// The hazard.
        hazard: String,
        /// The empty field.
        field: &'static str,
    },
    /// Every control is learned.
    OnlyLearnedControl {
        /// The hazard.
        hazard: String,
    },
    /// No register covers the robot and mode.
    NoRegister,
    /// The analysis has not been performed.
    NotAnalysed,
}

impl core::fmt::Display for HazardError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Incomplete { hazard, field } => write!(f, "hazard {hazard:?}: {field} is empty"),
            Self::OnlyLearnedControl { hazard } => write!(
                f,
                "hazard {hazard:?} is controlled only by a learned component, which is not a control (SPECS.md 12.2)"
            ),
            Self::NoRegister => f.write_str("no hazard register for this robot and mode"),
            Self::NotAnalysed => f.write_str("the robot-specific hazard analysis has not been performed"),
        }
    }
}

impl core::error::Error for HazardError {}

fn control(kind: ControlKind, description: &str) -> Control {
    Control {
        kind,
        description: description.to_owned(),
    }
}

impl HazardRegister {
    /// Section 12.2's initial entries for `robot` in `mode`, not analysed
    /// and with no risk accepted: "a starting structure, not a completed
    /// analysis".
    #[must_use]
    pub fn initial(robot: &str, mode: OperatingMode) -> Self {
        use ControlKind::{Firmware, Hardware, Procedural, Software};
        let h = |name: &str, causes: &str, detection: &str, controls: Vec<Control>| Hazard {
            name: name.to_owned(),
            causes: causes.to_owned(),
            detection: detection.to_owned(),
            controls,
            accepted_by: String::new(),
        };
        Self {
            robot: robot.to_owned(),
            mode,
            analysis_performed: false,
            hazards: vec![
                h("uncommanded motion", "stale command held by firmware, watchdog misconfiguration, mode confusion", "command expiry counters, independent watchdog, transmit-disable readback", vec![control(Firmware, "short command expiry"), control(Firmware, "watchdog outside the policy process"), control(Software, "mode asserted every tick")]),
                h("fall", "loss of balance, actuator saturation, late or failed skill transition", "attitude and contact estimation, deadline misses, recovery invocation count", vec![control(Software, "protective response from the safety profile"), control(Procedural, "area clear of obstacles and people")]),
                h("excess contact energy", "reward exploitation, scoring bug, mis-scaled action, opponent model error", "impulse and force measurement, energy budget accounting, independent match logging", vec![control(Software, "energy budget enforced in the supervisor"), control(Hardware, "padded fixtures")]),
                h("actuator overload", "infeasible reference, torque-speed envelope violation, thermal accumulation", "temperature, current, saturation duration, torque-speed margin", vec![control(Software, "envelope enforcement and thermal derating in the supervisor")]),
                h("pinch and entanglement", "cables, gloves, straps, support fixtures within swept volume", "pre-run checklist, collision geometry including attachments", vec![control(Procedural, "swept-volume check with attachments modeled")]),
                h("human enters the area", "procedural failure, unexpected access, observer approach", "area interlock, observer role", vec![control(Hardware, "area interlock: immediate controlled stop"), control(Procedural, "nominated operator holding the emergency stop")]),
                h("loss of transport", "cable, radio, switch, host stall", "link heartbeat, arrival-time gaps, deadline misses", vec![control(Firmware, "firmware-side command expiry")]),
                h("wrong artifact deployed", "bundle/profile mismatch, firmware change, payload change", "compatibility validation at load; checksum and signature", vec![control(Software, "refuse to activate; last-known-good bundle available")]),
            ],
        }
    }

    /// Checks every entry.
    pub fn validate(&self) -> Result<(), HazardError> {
        for h in &self.hazards {
            for (v, field) in [
                (&h.name, "name"),
                (&h.causes, "causes"),
                (&h.detection, "detection"),
                (&h.accepted_by, "accepted_by"),
            ] {
                if v.trim().is_empty() {
                    return Err(HazardError::Incomplete {
                        hazard: h.name.clone(),
                        field,
                    });
                }
            }
            if h.controls.is_empty() {
                return Err(HazardError::Incomplete {
                    hazard: h.name.clone(),
                    field: "controls",
                });
            }
            if h.controls.iter().all(|c| c.kind == ControlKind::Learned) {
                return Err(HazardError::OnlyLearnedControl {
                    hazard: h.name.clone(),
                });
            }
        }
        Ok(())
    }

    /// Whether `robot` may enter `mode` as far as the registers say.
    pub fn ready_for(
        registers: &[Self],
        robot: &str,
        mode: OperatingMode,
    ) -> Result<(), HazardError> {
        if !mode.is_hardware() {
            return Ok(());
        }
        let r = registers
            .iter()
            .find(|r| r.robot == robot && r.mode == mode)
            .ok_or(HazardError::NoRegister)?;
        if !r.analysis_performed {
            return Err(HazardError::NotAnalysed);
        }
        r.validate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_initial_register_is_a_starting_structure_not_an_analysis() {
        let r = HazardRegister::initial("g1@1.0", OperatingMode::HwFreestandingNoncontact);
        assert_eq!(r.hazards.len(), 8);
        assert!(matches!(
            r.validate(),
            Err(HazardError::Incomplete {
                field: "accepted_by",
                ..
            })
        ));
        assert_eq!(
            HazardRegister::ready_for(&[r], "g1@1.0", OperatingMode::HwFreestandingNoncontact),
            Err(HazardError::NotAnalysed)
        );
    }

    #[test]
    fn a_learned_only_control_is_refused() {
        let mut r = HazardRegister::initial("g1@1.0", OperatingMode::HwContact);
        r.analysis_performed = true;
        for h in &mut r.hazards {
            h.accepted_by = "safety lead".into();
        }
        assert_eq!(r.validate(), Ok(()));
        r.hazards[1].controls = vec![control(
            ControlKind::Learned,
            "the policy learned to stay upright",
        )];
        assert_eq!(
            r.validate(),
            Err(HazardError::OnlyLearnedControl {
                hazard: "fall".into()
            })
        );
    }

    #[test]
    fn simulation_needs_no_register_and_hardware_needs_one_for_its_mode() {
        assert_eq!(
            HazardRegister::ready_for(&[], "g1@1.0", OperatingMode::SimTrain),
            Ok(())
        );
        assert_eq!(
            HazardRegister::ready_for(&[], "g1@1.0", OperatingMode::HwShadow),
            Err(HazardError::NoRegister)
        );
    }
}

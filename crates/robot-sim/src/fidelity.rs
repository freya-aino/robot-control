//! Fidelity harness: a shared analytic battery run at every declared
//! simulation [`Tier`], reported as an immutable TOML artifact.
//!
//! The report format is fixed by `docs/decisions/0008-fidelity-report-schema.md`.
//! Scalar checks use `measured.is_finite() && measured <= tolerance`;
//! v6 series checks record the maximum absolute error and its sample index,
//! with zero relative tolerance so the pass decision is auditable.
//! GPU-only checks are listed as not applicable rather than omitted, so a
//! reader cannot mistake their absence for a pass.

use std::fmt;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use crate::articulated::{ArticulatedBody, DynamicsError, MultibodyState};
use crate::contact::GroundContact;
use crate::oracle::{self, OracleError};
use crate::pendulum::{NonFiniteInput, Pendulum, PendulumParams, PendulumState};
use robot_artifacts::hash::{ContentId, HashError};
use robot_artifacts::store::{write_immutable, StoreError};
use robot_artifacts::toml::{parse, to_string, Table, TomlError, Value};
use robot_model::joint::{JointKind, JointLimits, JointSpec};
use robot_model::kinematics::{forward_kinematics, KinematicsError};
use robot_model::link::{InertiaTensor, LinkSpec, MassProperties};
use robot_model::pose::RigidPose;
use robot_model::profile::{FirmwareContract, RobotProfile, PROFILE_SCHEMA_VERSION};
use robot_types::frames::{FrameId, GravityConvention, Vec3};
use robot_types::ids::{JointId, LinkId, Name};
use robot_types::schema::ExtensionTable;
use robot_types::time::Duration;
use robot_types::units::{Kilograms, NewtonMetres, Radians, RadiansPerSecond, Range};

/// Battery identifier written into every report. Its version suffix tracks
/// the battery contents (check set, canonical tolerances, horizons) and is
/// bumped together with [`SCHEMA_VERSION`] whenever those change.
pub const BATTERY: &str = "shared-analytic-v7";
/// Scope written into every report. The archival battery covers the analytic
/// subset only: it contains no ground-contact check, so no report qualifies
/// the floating-base G1 contact path (ADR 0010). See [`contact_applicability`].
pub const SCOPE: &str = "analytic-subset";
/// Schema identifier written into every report.
pub const SCHEMA: &str = "robot-sim.fidelity-report";
/// Schema version written into every report.
pub const SCHEMA_VERSION: i64 = 7;
/// File name the report is written under.
pub const REPORT_FILE: &str = "fidelity.toml";
/// Largest total number of integration steps [`run_battery`] accepts for one
/// call. [`run_checks`] applies the same cap to its custom checks at one tier.
/// The declared analytic tiers need 539 000 steps. The archival battery and
/// `reproduce` also budget the G1 probe's 4 000-step cap, for 543 000 total;
/// anything above this cap is refused before any simulation runs.
pub const MAX_BATTERY_STEPS: u64 = 2_000_000;
/// Maximum archival report size accepted by the reader (one MiB).
pub const MAX_REPORT_BYTES: usize = 1024 * 1024;
/// The only integrator the battery can run.
pub const INTEGRATOR_SEMI_IMPLICIT_EULER: &str = "semi-implicit-euler";
/// The only solver the battery can run.
pub const SOLVER_EXPLICIT_ABA: &str = "explicit-articulated-body";
/// The checks the battery runs at every tier, in run order.
pub const CHECK_NAMES: [&str; 7] = [
    "pendulum_period",
    "pendulum_trajectory",
    "pd_equilibrium",
    "aba_rne_closed_form",
    "energy_drift",
    "articulated_energy_drift",
    "free_fall",
];
/// The physical system each battery check exercises, in [`CHECK_NAMES`] order.
/// A reader rejects any row whose `system` differs from this table, so a
/// standalone pendulum result cannot be relabelled as articulated evidence.
const CHECK_SYSTEMS: [&str; 7] = [
    "standalone_pendulum",
    "standalone_pendulum",
    "standalone_pendulum",
    "point_mass_pendulum_articulated",
    "standalone_pendulum",
    "double_pendulum_articulated",
    "free_body_articulated",
];
/// Top-level keys a report may carry. Anything else is rejected.
const REPORT_KEYS: [&str; 11] = [
    "schema",
    "schema_version",
    "battery",
    "scope",
    "tier_set",
    "all_pass",
    "tier",
    "check",
    "not_applicable",
    "contact",
    "probe",
];
const PROBE_KEYS: [&str; 6] = ["scenario", "system", "status", "step", "error", "reason"];
const TIER_KEYS: [&str; 4] = ["name", "timestep_ns", "integrator", "solver"];
const CHECK_KEYS: [&str; 9] = [
    "tier",
    "name",
    "system",
    "tolerance",
    "tolerance_abs",
    "tolerance_rel",
    "measured",
    "sample",
    "pass",
];
const NOT_APPLICABLE_KEYS: [&str; 2] = ["name", "reason"];
const CONTACT_KEYS: [&str; 4] = ["tier", "system", "status", "reason"];

/// The system label of the floating-base G1 ground-contact path.
pub const G1_CONTACT_SYSTEM: &str = "g1_floating_base_ground_contact";
/// Shortest timestep, in nanoseconds, of the validated G1 stance range
/// (ADR 0010: dt in [5e-5, 4e-4] s for the 2 s qualification).
pub const G1_CONTACT_MIN_TIMESTEP_NANOS: i64 = 50_000;
/// Longest timestep, in nanoseconds, of the validated G1 stance range.
pub const G1_CONTACT_MAX_TIMESTEP_NANOS: i64 = 400_000;

const GRAVITY: f64 = 9.81;
const PENDULUM_LENGTH: f64 = 1.0;
const PENDULUM_MASS: f64 = 1.0;
const PERIOD_AMPLITUDE: f64 = 1.0;
const PERIOD_HORIZON_NANOS: u64 = 4_000_000_000;
const PERIOD_TOLERANCE: f64 = 1e-4;
const TRAJECTORY_HORIZON_NANOS: u64 = 4_000_000_000;
// Semi-implicit Euler accumulates O(dt) phase error over four seconds.
// The absolute budgets scale by ten between 100 us and 1 ms; relative
// tolerance is zero so the stored maximum error determines the pass flag.
const TRAJECTORY_REFERENCE_TOLERANCE: f64 = 5e-4;
const TRAJECTORY_COARSE_TOLERANCE: f64 = 5e-3;
const PD_START: f64 = 1.5;
const PD_TARGET: f64 = 0.3;
const PD_KP: f64 = 20.0;
const PD_KD: f64 = 4.0;
const PD_HORIZON_NANOS: u64 = 20_000_000_000;
const PD_TOLERANCE: f64 = 1e-4;
const ABA_RNE_TOLERANCE: f64 = 1e-9;
const ENERGY_HORIZON_NANOS: u64 = 10_000_000_000;
const ENERGY_TOLERANCE: f64 = 0.02;
const ARTICULATED_ENERGY_HORIZON_NANOS: u64 = 10_000_000_000;
const ARTICULATED_ENERGY_TOLERANCE: f64 = 1e-2;
const CHAIN_MASS_1: f64 = 1.0;
const CHAIN_LENGTH_1: f64 = 0.5;
const CHAIN_MASS_2: f64 = 0.6;
const CHAIN_LENGTH_2: f64 = 0.35;
const CHAIN_START: [f64; 2] = [0.6, -0.3];
const FREE_FALL_HORIZON_NANOS: u64 = 1_000_000_000;
const FREE_FALL_TOLERANCE: f64 = 1e-2;

/// One numerical claim: the measured error against its tolerance, labelled
/// with the physical system it exercised.
#[derive(Clone, Debug, PartialEq)]
pub struct Check {
    /// Stable check identifier.
    pub name: String,
    /// The physical system the check exercised. Must match the canonical
    /// system for `name` (see the report reader).
    pub system: String,
    /// Largest accepted scalar measured value, in the check's own units.
    /// For a series check this is the absolute component of `series_tolerance`.
    pub tolerance: f64,
    /// Pointwise tolerance for a series check. `None` for legacy scalar
    /// metrics; TOML writes only `tolerance_abs` and `tolerance_rel` when set.
    pub series_tolerance: Option<oracle::Tolerance>,
    /// The measured error. Non-finite values always fail.
    pub measured: f64,
    /// Index of the sample (or step) where `measured` is attained. Zero when
    /// the check is a single scalar with no per-sample series.
    pub sample: usize,
    /// For scalars, `measured.is_finite() && measured <= tolerance`.
    /// The built-in series check has zero relative tolerance, so the same
    /// rule applies to its maximum absolute pointwise error.
    pub pass: bool,
}

impl Check {
    /// Builds a scalar check at sample zero and derives its pass flag from the
    /// single pass rule.
    #[must_use]
    pub fn new(name: &str, tolerance: f64, measured: f64) -> Self {
        Self::for_system(name, system_of(name), tolerance, measured, 0)
    }

    fn for_system(name: &str, system: &str, tolerance: f64, measured: f64, sample: usize) -> Self {
        let pass = measured.is_finite() && measured <= tolerance;
        Self {
            name: name.to_owned(),
            system: system.to_owned(),
            tolerance,
            series_tolerance: None,
            measured,
            sample,
            pass,
        }
    }

    fn for_series(
        name: &str,
        system: &str,
        tolerance: oracle::Tolerance,
        measured: Measurement,
    ) -> Self {
        Self {
            name: name.to_owned(),
            system: system.to_owned(),
            tolerance: tolerance.abs,
            series_tolerance: Some(tolerance),
            measured: measured.value,
            sample: measured.sample,
            // The v6 battery uses rel=0: maximum raw error is sufficient to
            // reproduce the pointwise pass rule from the serialized report.
            pass: measured.value.is_finite() && measured.value <= tolerance.abs,
        }
    }
}

/// The canonical physical system for a battery check name, or the empty
/// string for an unknown name.
fn system_of(name: &str) -> &'static str {
    CHECK_NAMES
        .iter()
        .position(|n| *n == name)
        .map_or("", |i| CHECK_SYSTEMS[i])
}

/// What a check measured: the worst error and the sample index where it is
/// attained. For a per-sample series, `sample` is the index of that sample
/// (for the trajectory and energy series, the number of integration steps
/// taken before it). For a scalar measured at the end of a run, `sample` is
/// the final step count.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Measurement {
    /// The measured error, in the check's own units. Non-finite values fail.
    pub value: f64,
    /// Index of the sample or step at which `value` is attained.
    pub sample: usize,
}

impl Measurement {
    /// A measurement attained at `sample`.
    #[must_use]
    pub const fn new(value: f64, sample: usize) -> Self {
        Self { value, sample }
    }
}

/// One fidelity check the battery can run. The battery runner iterates over
/// `&dyn FidelityCheck`, so a standalone check can be added without editing
/// the runner. Built-in checks are unit structs; [`run_checks`] evaluates any
/// slice of them at one tier.
pub trait FidelityCheck {
    /// Stable check identifier, unique within a battery.
    fn name(&self) -> &'static str;

    /// Physical system the check exercises.
    fn system(&self) -> &'static str;

    /// Largest accepted measured value, in the check's own units.
    fn tolerance(&self) -> f64;

    /// Optional tier-specific pointwise tolerance for a series check. `None`
    /// denotes a scalar check. V6 maximum-error rows require finite, non-
    /// negative `abs` and `rel = 0`; `run_checks` rejects any other series
    /// tolerance because the serialized maximum raw error cannot audit a
    /// pointwise relative-tolerance decision.
    fn series_tolerance(&self, _tier: &Tier) -> Option<oracle::Tolerance> {
        None
    }

    /// Measures the check's error at `tier`. Returning `None` skips the check
    /// in the standalone [`run_checks`] result; it does not create an archival
    /// not-applicable entry. Built-in battery checks must return `Some`.
    fn measure(&self, tier: &Tier) -> Result<Option<Measurement>, BatteryError>;

    /// Upper bound on the integration steps the check needs at `tier`.
    /// [`run_checks`] includes this value in its budget before measuring.
    fn steps(&self, tier: &Tier) -> u64;
}

/// Runs `checks` at `tier`, subject to tier validation and
/// [`MAX_BATTERY_STEPS`]. Inapplicable (`None`) checks are skipped, not failed
/// or recorded as archival not-applicable entries. The declared archival
/// battery has a stricter contract: all of its checks must return `Some`.
pub fn run_checks(checks: &[&dyn FidelityCheck], tier: &Tier) -> Result<Vec<Check>, BatteryError> {
    run_checks_inner(checks, tier, true, MAX_BATTERY_STEPS)
}

fn run_checks_inner(
    checks: &[&dyn FidelityCheck],
    tier: &Tier,
    allow_inapplicable: bool,
    max_steps: u64,
) -> Result<Vec<Check>, BatteryError> {
    validate_tiers(std::slice::from_ref(tier))?;
    validate_check_names(checks)?;
    let required = check_steps(checks, std::slice::from_ref(tier))?;
    if required > u128::from(max_steps) {
        return Err(BatteryError::StepBudget {
            required,
            limit: max_steps,
        });
    }

    let mut out = Vec::new();
    for check in checks {
        let Some(measured) = check.measure(tier)? else {
            if allow_inapplicable {
                continue;
            }
            return Err(BatteryError::InapplicableCheck {
                name: check.name().to_owned(),
                tier: tier.name.clone(),
            });
        };
        out.push(if let Some(tolerance) = check.series_tolerance(tier) {
            if tolerance.rel != 0.0 || !tolerance.abs.is_finite() || tolerance.abs < 0.0 {
                return Err(BatteryError::CheckFailed {
                    name: check.name().to_owned(),
                    message: "v6 maximum-error rows require finite non-negative absolute and zero relative tolerance".to_owned(),
                });
            }
            Check::for_series(check.name(), check.system(), tolerance, measured)
        } else {
            Check::for_system(
                check.name(),
                check.system(),
                check.tolerance(),
                measured.value,
                measured.sample,
            )
        });
    }
    Ok(out)
}

/// A declared simulation tier: the timestep, integrator and solver a set of
/// checks was run with.
#[derive(Clone, Debug, PartialEq)]
pub struct Tier {
    /// Tier identifier, unique within a report.
    pub name: String,
    /// Fixed integration timestep.
    pub timestep: Duration,
    /// Integrator identifier. Must be [`INTEGRATOR_SEMI_IMPLICIT_EULER`].
    pub integrator: String,
    /// Solver identifier. Must be [`SOLVER_EXPLICIT_ABA`].
    pub solver: String,
}

impl Tier {
    /// A tier using the only integrator and solver the battery implements.
    #[must_use]
    pub fn semi_implicit(name: &str, timestep: Duration) -> Self {
        Self {
            name: name.to_owned(),
            timestep,
            integrator: INTEGRATOR_SEMI_IMPLICIT_EULER.to_owned(),
            solver: SOLVER_EXPLICIT_ABA.to_owned(),
        }
    }
}

/// The two tiers the battery declares: a reference and a coarse timestep.
#[must_use]
pub fn declared_tiers() -> Vec<Tier> {
    vec![
        Tier::semi_implicit("reference", Duration::from_micros(100)),
        Tier::semi_implicit("coarse", Duration::from_millis(1)),
    ]
}

/// Whether the exact production tier declaration was run or a custom experiment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TierSet {
    /// All production tiers, with their exact timestep, integrator and solver.
    Declared,
    /// Custom tiers, never interpreted as the production battery's result.
    Experimental,
}

impl TierSet {
    fn as_str(self) -> &'static str {
        match self {
            Self::Declared => "declared",
            Self::Experimental => "experimental",
        }
    }
}

/// A check that cannot run in this build, recorded with its reason.
#[derive(Clone, Debug, PartialEq)]
pub struct NotApplicable {
    /// Check identifier.
    pub name: String,
    /// Why it cannot run.
    pub reason: String,
}

/// The checks that cannot run until a GPU backend exists.
#[must_use]
pub fn not_applicable_checks() -> Vec<NotApplicable> {
    vec![
        NotApplicable {
            name: "cpu_gpu_agreement".to_owned(),
            reason: "the battery does not run on the feature-gated robot-sim-gpu subset (ADR 0012); specs/12 agreement is not measurable here".to_owned(),
        },
        NotApplicable {
            name: "contact_budget_exhaustion".to_owned(),
            reason: "robot-sim-gpu has no contact-slot budget (specs/11); exhaustion is not measurable".to_owned(),
        },
    ]
}

/// Whether the floating-base G1 ground-contact path is qualified at a tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContactStatus {
    /// The tier's timestep is outside the ADR 0010 validated range, so G1
    /// contact is not qualified there and no contact result is expected.
    NotApplicable,
    /// The timestep is inside the validated range, but the archival battery
    /// has no G1 contact check, so the contact path is still not measured.
    NotMeasured,
}

impl ContactStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::NotApplicable => "not_applicable",
            Self::NotMeasured => "not_measured",
        }
    }

    fn parse(text: &str) -> Option<Self> {
        match text {
            "not_applicable" => Some(Self::NotApplicable),
            "not_measured" => Some(Self::NotMeasured),
            _ => None,
        }
    }
}

/// Measured outcome of the floating-base G1 zero-torque fall at one tier.
/// Unlike [`ContactApplicability`], this is a measurement, not a status. A
/// `Fail` row is a recorded failure: it does not change [`FidelityReport::all_pass`]
/// and it does not make the default gate fail.
#[derive(Clone, Debug, PartialEq)]
pub struct ContactProbe {
    /// Scenario label, always [`G1_PROBE_LABEL`].
    pub scenario: String,
    /// Always [`G1_CONTACT_SYSTEM`].
    pub system: String,
    /// `Fail` when the fall stopped with a [`DynamicsError`] within the step
    /// window; `Pass` when it stayed finite and solvable for the whole window.
    pub status: ProbeStatus,
    /// Index of the first failing step, or `None` for a pass.
    pub step: Option<u64>,
    /// The [`DynamicsError`] variant name at the failing step, or `None` for a pass.
    pub error: Option<String>,
    /// Why the row says what it says, including the model attribution.
    pub reason: String,
}

/// Outcome of a [`ContactProbe`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeStatus {
    /// The probe ran the full window without a dynamics error.
    Pass,
    /// The probe stopped at a dynamics error. Recorded, not hidden.
    Fail,
}

impl ProbeStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fail => "fail",
        }
    }

    fn parse(text: &str) -> Option<Self> {
        match text {
            "pass" => Some(Self::Pass),
            "fail" => Some(Self::Fail),
            _ => None,
        }
    }
}

/// Timestep of the G1 zero-torque probe row, in microseconds (`5e-4` s).
pub const G1_PROBE_DT_MICROS: i64 = 500;
/// Number of steps the probe runs at most. Two seconds at [`G1_PROBE_DT_MICROS`].
pub const G1_PROBE_MAX_STEPS: u64 = 4_000;

/// Scenario label of the G1 probe row. The probe is one fixed scenario at
/// `dt = 5e-4` s, not a per-tier run, so it is recorded once per report.
pub const G1_PROBE_LABEL: &str = "g1-zero-torque-dt-5e-4";

/// Start clearance of the G1 probe above the ground plane, in metres.
const G1_PROBE_START_CLEARANCE: f64 = 0.005;
/// Gravity magnitude of the G1 probe, in m/s^2 (the `robot-cli` value).
const G1_PROBE_GRAVITY: f64 = 9.806_65;
/// Ground contact of the G1 probe (the `robot-cli` `FLOATING_GROUND` values).
const G1_PROBE_GROUND: GroundContact = GroundContact {
    stiffness: 2.0e4,
    normal_damping: 200.0,
    friction: 0.8,
    friction_regularization: 1e-3,
    ground_height: 0.0,
};
/// Canonical reason text of the G1 probe row. The reader requires this exact text.
pub const G1_PROBE_REASON: &str = "zero-torque floating G1 fall on the sphere ground at dt = 500 us, run for at most 4000 steps; the pelvis and torso have no collision geometry, so the trunk reaches the plane through unmodelled shapes once the legs fold (ADR 0010)";
/// Error names a `fail` probe row may carry: the [`DynamicsError`] variants.
const DYNAMICS_ERROR_NAMES: [&str; 4] = [
    "WidthMismatch",
    "NotFinite",
    "SingularJointInertia",
    "InvalidGroundContact",
];

/// Setup failures of the G1 probe. A setup failure is an error, never a `pass` row.
#[derive(Debug, PartialEq)]
pub enum ProbeError {
    /// The vendored G1 fixture did not load.
    Fixture(String),
    /// The G1 floating base has no contact sphere to place on the ground.
    NoContactSpheres,
    /// A dynamics call failed during setup.
    Dynamics(DynamicsError),
}

impl fmt::Display for ProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fixture(message) => write!(f, "the G1 fixture did not load: {message}"),
            Self::NoContactSpheres => write!(f, "the G1 floating base has no contact sphere"),
            Self::Dynamics(e) => write!(f, "{e}"),
        }
    }
}

impl core::error::Error for ProbeError {}

/// The zero-torque floating-G1 fall on the sphere ground, run for at most
/// `max_steps` steps at `dt`. Returns the first failing step and its error,
/// or `None` if the fall stays finite and solvable for the whole window.
/// Setup failures are returned as [`ProbeError`], never as a pass.
///
/// The ground and start pose match the `robot-cli` `sim multibody --floating`
/// path. The failure is attributed to the model as well as to `dt`: the pelvis
/// and torso carry no collision geometry, so once the legs fold the trunk
/// reaches the plane through unmodelled shapes (ADR 0010).
fn g1_zero_torque_fall(
    dt: Duration,
    max_steps: u64,
) -> Result<Option<(u64, DynamicsError)>, ProbeError> {
    let profile = robot_model::fixtures::unitree_g1_29dof_floating_base()
        .map_err(|e| ProbeError::Fixture(e.to_string()))?;
    let body = ArticulatedBody::from_profile(&profile);
    let mut q = vec![0.0; body.position_width()];
    q[3] = 1.0;
    let lowest = body
        .lowest_contact_height(&q)
        .map_err(ProbeError::Dynamics)?
        .ok_or(ProbeError::NoContactSpheres)?;
    q[2] = G1_PROBE_START_CLEARANCE - lowest;
    let mut state = MultibodyState::at_rest(q, body.dof());
    let tau = vec![0.0; body.dof()];
    let gravity = Vec3::new(0.0, 0.0, -G1_PROBE_GRAVITY);
    for step in 0..max_steps {
        if let Err(error) = state.step_with_ground(&body, &tau, gravity, &G1_PROBE_GROUND, dt) {
            return Ok(Some((step, error)));
        }
        if state.q.iter().chain(&state.qd).any(|v| !v.is_finite()) {
            return Ok(Some((
                step,
                DynamicsError::NotFinite {
                    what: "G1 joint state",
                },
            )));
        }
    }
    Ok(None)
}

/// Runs the G1 zero-torque probe at `dt = 5e-4` and records the outcome.
/// Called by the battery and by [`FidelityReport::reproduce`], never by the reader.
pub fn g1_zero_torque_probe() -> Result<ContactProbe, ProbeError> {
    let outcome = g1_zero_torque_fall(
        Duration::from_micros(G1_PROBE_DT_MICROS),
        G1_PROBE_MAX_STEPS,
    )?;
    let (status, step, error) = match outcome {
        None => (ProbeStatus::Pass, None, None),
        Some((step, error)) => (
            ProbeStatus::Fail,
            Some(step),
            Some(dynamics_error_name(&error).to_owned()),
        ),
    };
    Ok(ContactProbe {
        scenario: G1_PROBE_LABEL.to_owned(),
        system: G1_CONTACT_SYSTEM.to_owned(),
        status,
        step,
        error,
        reason: G1_PROBE_REASON.to_owned(),
    })
}

/// Structural check of one probe row, run by the archival reader. It never
/// simulates: the row must carry the canonical scenario, system and reason,
/// a `fail` row a step inside the window and a known error name, and a
/// `pass` row neither.
fn validate_probe_row(row: &ContactProbe) -> Result<(), ReportError> {
    if row.scenario != G1_PROBE_LABEL {
        return Err(invalid(format!(
            "probe scenario {:?} is not {G1_PROBE_LABEL:?}",
            row.scenario
        )));
    }
    if row.system != G1_CONTACT_SYSTEM {
        return Err(invalid(format!(
            "probe system {:?} is not {G1_CONTACT_SYSTEM:?}",
            row.system
        )));
    }
    if row.reason != G1_PROBE_REASON {
        return Err(invalid(
            "probe reason is not the canonical G1 probe reason".to_owned(),
        ));
    }
    match (row.status, row.step, row.error.as_deref()) {
        (ProbeStatus::Pass, None, None) => Ok(()),
        (ProbeStatus::Fail, Some(step), Some(error)) => {
            if step >= G1_PROBE_MAX_STEPS {
                return Err(invalid(format!(
                    "probe step {step} is outside the window of {G1_PROBE_MAX_STEPS} steps"
                )));
            }
            if !DYNAMICS_ERROR_NAMES.contains(&error) {
                return Err(invalid(format!(
                    "probe error {error:?} is not a known DynamicsError"
                )));
            }
            Ok(())
        }
        _ => Err(invalid(
            "probe status, step and error do not agree".to_owned(),
        )),
    }
}

fn dynamics_error_name(error: &DynamicsError) -> &'static str {
    match error {
        DynamicsError::WidthMismatch { .. } => "WidthMismatch",
        DynamicsError::NotFinite { .. } => "NotFinite",
        DynamicsError::SingularJointInertia => "SingularJointInertia",
        DynamicsError::InvalidGroundContact => "InvalidGroundContact",
    }
}

/// The G1 ground-contact applicability of one declared tier.
#[derive(Clone, Debug, PartialEq)]
pub struct ContactApplicability {
    /// Name of the [`Tier`].
    pub tier: String,
    /// Always [`G1_CONTACT_SYSTEM`].
    pub system: String,
    /// Applicability at this tier.
    pub status: ContactStatus,
    /// Why the status holds.
    pub reason: String,
}

/// The G1 contact applicability of each tier, derived only from its timestep.
/// A coarse tier outside the ADR 0010 range is `NotApplicable`; a tier inside
/// it is `NotMeasured`, because the analytic battery never runs a G1 contact
/// check. No status here says that G1 contact is qualified.
#[must_use]
pub fn contact_applicability(tiers: &[Tier]) -> Vec<ContactApplicability> {
    tiers
        .iter()
        .map(|tier| {
            let nanos = tier.timestep.nanos();
            let range = format!(
                "[{}, {}] ns (ADR 0010)",
                G1_CONTACT_MIN_TIMESTEP_NANOS, G1_CONTACT_MAX_TIMESTEP_NANOS
            );
            let (status, reason) = if (G1_CONTACT_MIN_TIMESTEP_NANOS..=G1_CONTACT_MAX_TIMESTEP_NANOS)
                .contains(&nanos)
            {
                (
                    ContactStatus::NotMeasured,
                    format!("timestep {nanos} ns is inside the validated G1 contact range {range}, but the analytic battery has no G1 contact check; G1 contact is not measured at this tier"),
                )
            } else {
                (
                    ContactStatus::NotApplicable,
                    format!("timestep {nanos} ns is outside the validated G1 contact range {range}; G1 contact is not qualified at this tier"),
                )
            };
            ContactApplicability {
                tier: tier.name.clone(),
                system: G1_CONTACT_SYSTEM.to_owned(),
                status,
                reason,
            }
        })
        .collect()
}

/// A check together with the tier it was run at.
#[derive(Clone, Debug, PartialEq)]
pub struct TierCheck {
    /// Name of the [`Tier`] the check ran at.
    pub tier: String,
    /// The check result.
    pub check: Check,
}

/// A fidelity report: every declared tier, every check at every tier, and
/// the checks that are not applicable.
#[derive(Clone, Debug, PartialEq)]
pub struct FidelityReport {
    /// Battery identifier.
    pub battery: String,
    /// Production tier set or an explicitly labeled experiment.
    pub tier_set: TierSet,
    /// Declared tiers, in run order.
    pub tiers: Vec<Tier>,
    /// Every check at every tier.
    pub checks: Vec<TierCheck>,
    /// Checks recorded as not applicable.
    pub not_applicable: Vec<NotApplicable>,
    /// G1 ground-contact applicability of each tier. Not a pass or fail.
    pub contact: Vec<ContactApplicability>,
    /// Measured G1 zero-torque probe, one row for the fixed `dt = 5e-4` scenario.
    /// A `fail` row is a recorded failure and does not change [`Self::all_pass`].
    pub probe: Vec<ContactProbe>,
}

impl FidelityReport {
    /// True only if there is at least one check and every check passed.
    /// Not-applicable entries do not count. Says nothing about `tier_set`;
    /// use [`Self::production_pass`] to qualify a result for production.
    #[must_use]
    pub fn all_pass(&self) -> bool {
        !self.checks.is_empty() && self.checks.iter().all(|c| c.check.pass)
    }

    /// The only production qualification predicate: the exact declared tier
    /// set was run and every check passed. Experimental reports never qualify.
    /// It qualifies the analytic subset only (`scope`): it says nothing about
    /// G1 ground contact, which is reported separately in `contact`.
    #[must_use]
    pub fn production_pass(&self) -> bool {
        self.tier_set == TierSet::Declared && self.all_pass()
    }

    /// Serializes the report to the ADR 0008 TOML layout.
    #[must_use]
    pub fn to_toml(&self) -> String {
        let mut root = Table::new();
        root.insert("schema", Value::String(SCHEMA.to_owned()));
        root.insert("schema_version", Value::Integer(SCHEMA_VERSION));
        root.insert("battery", Value::String(self.battery.clone()));
        root.insert("scope", Value::String(SCOPE.to_owned()));
        root.insert("tier_set", Value::String(self.tier_set.as_str().to_owned()));
        root.insert("all_pass", Value::Boolean(self.all_pass()));
        if !self.tiers.is_empty() {
            root.insert(
                "tier",
                Value::Array(
                    self.tiers
                        .iter()
                        .map(tier_table)
                        .map(Value::Table)
                        .collect(),
                ),
            );
        }
        if !self.checks.is_empty() {
            root.insert(
                "check",
                Value::Array(
                    self.checks
                        .iter()
                        .map(check_table)
                        .map(Value::Table)
                        .collect(),
                ),
            );
        }
        if !self.not_applicable.is_empty() {
            root.insert(
                "not_applicable",
                Value::Array(
                    self.not_applicable
                        .iter()
                        .map(not_applicable_table)
                        .map(Value::Table)
                        .collect(),
                ),
            );
        }
        if !self.contact.is_empty() {
            root.insert(
                "contact",
                Value::Array(
                    self.contact
                        .iter()
                        .map(contact_table)
                        .map(Value::Table)
                        .collect(),
                ),
            );
        }
        if !self.probe.is_empty() {
            root.insert(
                "probe",
                Value::Array(
                    self.probe
                        .iter()
                        .map(probe_table)
                        .map(Value::Table)
                        .collect(),
                ),
            );
        }
        to_string(&root)
    }

    /// Validates the v7 report layout without executing any simulation.
    /// Declared reports require exactly [`declared_tiers`]; custom experiments
    /// must be explicitly labeled. Both modes require canonical check rows,
    /// tolerances, pass flags and not-applicable entries.
    pub fn from_toml(text: &str) -> Result<Self, ReportError> {
        let root = parse(text).map_err(ReportError::Toml)?;
        reject_unknown_keys(&root, &REPORT_KEYS, "report")?;
        let schema = root.require_str("schema").map_err(ReportError::Toml)?;
        if schema != SCHEMA {
            return Err(invalid(format!(
                "schema is {schema:?}, expected {SCHEMA:?}"
            )));
        }
        let version = root
            .require_integer("schema_version")
            .map_err(ReportError::Toml)?;
        if version != SCHEMA_VERSION {
            return Err(invalid(format!(
                "schema_version is {version}, expected {SCHEMA_VERSION}"
            )));
        }
        let battery = root.require_str("battery").map_err(ReportError::Toml)?;
        if battery != BATTERY {
            return Err(invalid(format!(
                "battery is {battery:?}, expected {BATTERY:?}"
            )));
        }
        let scope = root.require_str("scope").map_err(ReportError::Toml)?;
        if scope != SCOPE {
            return Err(invalid(format!("scope is {scope:?}, expected {SCOPE:?}")));
        }
        let tier_set = match root.require_str("tier_set").map_err(ReportError::Toml)? {
            "declared" => TierSet::Declared,
            "experimental" => TierSet::Experimental,
            other => return Err(invalid(format!("unknown tier_set {other:?}"))),
        };
        let stored_all_pass = root.require_bool("all_pass").map_err(ReportError::Toml)?;

        let mut tiers = Vec::new();
        for row in tables(&root, "tier")? {
            reject_unknown_keys(row, &TIER_KEYS, "tier")?;
            tiers.push(read_tier(row)?);
        }
        if tiers.is_empty() {
            return Err(invalid("report declares no tiers".to_owned()));
        }
        reject_duplicates(tiers.iter().map(|t| t.name.as_str()), "tier")?;
        if (tier_set == TierSet::Declared) != (tiers == declared_tiers()) {
            return Err(invalid(
                "tier_set must distinguish the exact declared tiers from experimental tiers"
                    .to_owned(),
            ));
        }
        for tier in &tiers {
            if tier.integrator != INTEGRATOR_SEMI_IMPLICIT_EULER
                || tier.solver != SOLVER_EXPLICIT_ABA
            {
                return Err(invalid(format!(
                    "tier {:?} names an integrator or solver the battery does not implement",
                    tier.name
                )));
            }
        }

        let mut checks = Vec::new();
        for row in tables(&root, "check")? {
            reject_unknown_keys(row, &CHECK_KEYS, "check")?;
            checks.push(read_check(row)?);
        }
        let expected: Vec<(&str, &str)> = tiers
            .iter()
            .flat_map(|t| CHECK_NAMES.iter().map(move |name| (t.name.as_str(), *name)))
            .collect();
        let found: Vec<(&str, &str)> = checks
            .iter()
            .map(|c| (c.tier.as_str(), c.check.name.as_str()))
            .collect();
        if found != expected {
            return Err(invalid(format!(
                "check rows do not match the battery: expected the {} checks {:?} at each declared tier in order, found {} rows",
                CHECK_NAMES.len(),
                CHECK_NAMES,
                found.len()
            )));
        }
        for tc in &checks {
            if !tiers.iter().any(|t| t.name == tc.tier) {
                return Err(invalid(format!(
                    "check names undeclared tier {:?}",
                    tc.tier
                )));
            }
            let tier = tiers
                .iter()
                .find(|t| t.name == tc.tier)
                .expect("tier checked");
            let builtin = BUILTIN_CHECKS
                .iter()
                .find(|c| c.name() == tc.check.name)
                .ok_or_else(|| {
                    invalid(format!("check {:?} is not a battery check", tc.check.name))
                })?;
            let canonical = builtin.series_tolerance(tier).map_or_else(
                || canonical_tolerance(&tc.check.name).expect("known battery check"),
                |tol| tol.abs,
            );
            if tc.check.tolerance.to_bits() != canonical.to_bits() {
                return Err(invalid(format!(
                    "check {:?} at tier {:?} has tolerance {} but the battery declares {}",
                    tc.check.name, tc.tier, tc.check.tolerance, canonical
                )));
            }
            if tc.check.series_tolerance != builtin.series_tolerance(tier) {
                return Err(invalid(format!(
                    "check {:?} at tier {:?} has non-canonical series tolerance",
                    tc.check.name, tc.tier
                )));
            }
            let rule = tc.check.measured.is_finite() && tc.check.measured <= canonical;
            if tc.check.pass != rule {
                return Err(invalid(format!(
                    "check {:?} at tier {:?} has pass={} but measured={} tolerance={}",
                    tc.check.name, tc.tier, tc.check.pass, tc.check.measured, tc.check.tolerance
                )));
            }
        }

        let mut not_applicable = Vec::new();
        for row in tables(&root, "not_applicable")? {
            reject_unknown_keys(row, &NOT_APPLICABLE_KEYS, "not_applicable")?;
            not_applicable.push(read_not_applicable(row)?);
        }
        if not_applicable != not_applicable_checks() {
            return Err(invalid(
                "not_applicable rows do not match the battery's declared list".to_owned(),
            ));
        }

        let mut contact = Vec::new();
        for row in tables(&root, "contact")? {
            reject_unknown_keys(row, &CONTACT_KEYS, "contact")?;
            contact.push(read_contact(row)?);
        }
        if contact != contact_applicability(&tiers) {
            return Err(invalid(
                "contact rows do not match the canonical G1 contact applicability of the tiers"
                    .to_owned(),
            ));
        }

        let mut probe = Vec::new();
        for row in tables(&root, "probe")? {
            reject_unknown_keys(row, &PROBE_KEYS, "probe")?;
            let parsed = read_probe(row)?;
            validate_probe_row(&parsed)?;
            probe.push(parsed);
        }
        if probe.len() != 1 {
            return Err(invalid(format!(
                "a report carries exactly one probe row, found {}",
                probe.len()
            )));
        }

        let report = Self {
            battery: battery.to_owned(),
            tier_set,
            tiers,
            checks,
            not_applicable,
            contact,
            probe,
        };
        if report.all_pass() != stored_all_pass {
            return Err(invalid(format!(
                "all_pass is {stored_all_pass} but the checks give {}",
                report.all_pass()
            )));
        }
        Ok(report)
    }

    /// Explicit same-build, same-platform comparison, not archival validation.
    /// Refuses to run if the total requested integration steps exceed `max_steps`.
    pub fn reproduce(&self, max_steps: u64) -> Result<(), ReportError> {
        Self::from_toml(&self.to_toml())?;
        let fresh = run_battery_with_limit(&self.tiers, max_steps)
            .map_err(|e| invalid(format!("the declared tiers cannot be re-run: {e}")))?;
        if self.probe != fresh.probe {
            return Err(invalid(
                "stored probe row does not match a fresh G1 probe run".to_owned(),
            ));
        }
        if self.checks.len() != fresh.checks.len() {
            return Err(invalid("reproduction check count differs".to_owned()));
        }
        for (stored, rerun) in self.checks.iter().zip(&fresh.checks) {
            if stored.check.measured.to_bits() != rerun.check.measured.to_bits()
                || stored.check.sample != rerun.check.sample
            {
                return Err(invalid(format!(
                    "check {:?} at tier {:?} stores measured={} (sample {}) but re-running gives {} (sample {})",
                    stored.check.name,
                    stored.tier,
                    stored.check.measured,
                    stored.check.sample,
                    rerun.check.measured,
                    rerun.check.sample
                )));
            }
        }
        Ok(())
    }
}

/// Why a report could not be written or read.
#[derive(Debug)]
pub enum ReportError {
    /// The file is not valid TOML in the supported subset.
    Toml(TomlError),
    /// The file is valid TOML but not a valid report.
    Invalid(String),
    /// The file could not be read.
    Io {
        /// The path involved.
        path: PathBuf,
        /// The underlying error.
        source: io::Error,
    },
    /// The immutable write failed, including an attempt to overwrite a
    /// report with different bytes.
    Store(StoreError),
    /// File bytes differ from a separately retained expected identity.
    Identity(HashError),
}

impl fmt::Display for ReportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Toml(e) => write!(f, "fidelity report is malformed TOML: {e}"),
            Self::Invalid(reason) => write!(f, "fidelity report is invalid: {reason}"),
            Self::Io { path, source } => {
                write!(f, "reading fidelity report {}: {source}", path.display())
            }
            Self::Store(e) => write!(f, "writing fidelity report: {e}"),
            Self::Identity(e) => write!(f, "fidelity report identity mismatch: {e}"),
        }
    }
}

impl core::error::Error for ReportError {}

/// Writes the report as `dir/fidelity.toml` through `robot-artifacts`'
/// immutable write. Writing the same bytes again succeeds; different bytes
/// at the same path are rejected.
pub fn write_report(dir: &Path, report: &FidelityReport) -> Result<PathBuf, ReportError> {
    write_report_with_id(dir, report).map(|(path, _)| path)
}

/// Writes immutable report bytes and returns their content identity for
/// separately retained verification. An unkeyed digest alone is not provenance.
pub fn write_report_with_id(
    dir: &Path,
    report: &FidelityReport,
) -> Result<(PathBuf, ContentId), ReportError> {
    let path = dir.join(REPORT_FILE);
    let text = report.to_toml();
    if text.len() > MAX_REPORT_BYTES {
        return Err(invalid(format!("report exceeds {MAX_REPORT_BYTES} bytes")));
    }
    FidelityReport::from_toml(&text)?;
    write_immutable(&path, text.as_bytes()).map_err(ReportError::Store)?;
    Ok((path, ContentId::of(text.as_bytes())))
}

/// Reads and structurally validates a report without running any simulation.
pub fn read_report(path: &Path) -> Result<FidelityReport, ReportError> {
    read_report_bytes(path, None)
}

/// Reads and validates a report against a separately trusted expected content ID.
pub fn read_report_verified(
    path: &Path,
    expected: &ContentId,
) -> Result<FidelityReport, ReportError> {
    read_report_bytes(path, Some(expected))
}

fn read_report_bytes(
    path: &Path,
    expected: Option<&ContentId>,
) -> Result<FidelityReport, ReportError> {
    let file = std::fs::File::open(path).map_err(|source| ReportError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut bytes = Vec::new();
    file.take((MAX_REPORT_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|source| ReportError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    if bytes.len() > MAX_REPORT_BYTES {
        return Err(invalid(format!("report exceeds {MAX_REPORT_BYTES} bytes")));
    }
    if let Some(id) = expected {
        id.verify(&bytes).map_err(ReportError::Identity)?;
    }
    let text =
        std::str::from_utf8(&bytes).map_err(|_| invalid("report is not UTF-8".to_owned()))?;
    FidelityReport::from_toml(text)
}

/// Why the battery could not run. Non-exhaustive: new failure kinds may be
/// added without a breaking change.
#[derive(Debug, PartialEq)]
#[non_exhaustive]
pub enum BatteryError {
    /// No tiers were declared.
    NoTiers,
    /// Two tiers share a name.
    DuplicateTier(String),
    /// Two checks in one runner share a name.
    DuplicateCheck(String),
    /// A tier's timestep is not strictly positive.
    NonPositiveTimestep(String),
    /// A tier names an integrator the battery does not implement.
    UnsupportedIntegrator {
        /// The tier.
        tier: String,
        /// The requested integrator.
        integrator: String,
    },
    /// A tier names a solver the battery does not implement.
    UnsupportedSolver {
        /// The tier.
        tier: String,
        /// The requested solver.
        solver: String,
    },
    /// A pendulum step was rejected.
    Pendulum(NonFiniteInput),
    /// An articulated-body evaluation was rejected.
    Dynamics(DynamicsError),
    /// A forward-kinematics evaluation was rejected.
    Kinematics(KinematicsError),
    /// The tiers would need more integration steps than the limit allows.
    /// Raised before any simulation runs.
    StepBudget {
        /// Total steps the tiers would need.
        required: u128,
        /// The step limit in force.
        limit: u64,
    },
    /// A plugged-in check's step counts overflowed the budget accumulator.
    StepCountOverflow,
    /// A built-in archival battery check returned `None` for a tier.
    InapplicableCheck {
        /// The check that did not produce a measurement.
        name: String,
        /// The tier where it was inapplicable.
        tier: String,
    },
    /// A check failed to measure for its own reason. Plug-in checks report
    /// their own failures here instead of borrowing a built-in variant.
    CheckFailed {
        /// The check that failed.
        name: String,
        /// The check's own description of the failure.
        message: String,
    },
    /// The pure oracle comparator rejected its inputs, for example a series
    /// length mismatch. Never converted to a silent pass or truncation.
    Oracle(OracleError),
    /// The G1 probe could not be set up. A setup failure is an error, never a
    /// `pass` row.
    Probe(ProbeError),
}

impl fmt::Display for BatteryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoTiers => write!(f, "no simulation tiers declared"),
            Self::DuplicateTier(name) => write!(f, "tier {name:?} is declared twice"),
            Self::DuplicateCheck(name) => write!(f, "check {name:?} is declared twice"),
            Self::NonPositiveTimestep(name) => {
                write!(f, "tier {name:?} has a non-positive timestep")
            }
            Self::UnsupportedIntegrator { tier, integrator } => write!(
                f,
                "tier {tier:?} requests integrator {integrator:?}; only {INTEGRATOR_SEMI_IMPLICIT_EULER:?} is implemented"
            ),
            Self::UnsupportedSolver { tier, solver } => write!(
                f,
                "tier {tier:?} requests solver {solver:?}; only {SOLVER_EXPLICIT_ABA:?} is implemented"
            ),
            Self::Pendulum(e) => write!(f, "{e}"),
            Self::Dynamics(e) => write!(f, "{e}"),
            Self::Kinematics(e) => write!(f, "{e}"),
            Self::StepBudget { required, limit } => write!(
                f,
                "the tiers need {required} integration steps, limit is {limit}"
            ),
            Self::StepCountOverflow => write!(f, "the integration step count overflowed u128"),
            Self::InapplicableCheck { name, tier } => write!(
                f,
                "built-in check {name:?} returned no measurement at tier {tier:?}"
            ),
            Self::CheckFailed { name, message } => {
                write!(f, "check {name:?} failed: {message}")
            }
            Self::Oracle(e) => write!(f, "oracle comparison rejected: {e}"),
            Self::Probe(e) => write!(f, "G1 probe could not be set up: {e}"),
        }
    }
}

impl core::error::Error for BatteryError {}

/// Runs the shared battery at every tier and returns the report.
/// Refuses tiers needing more than [`MAX_BATTERY_STEPS`] steps in total.
pub fn run_battery(tiers: &[Tier]) -> Result<FidelityReport, BatteryError> {
    run_battery_with_limit(tiers, MAX_BATTERY_STEPS)
}

/// Runs the shared battery under an explicit total step limit. Every tier is
/// validated and the step total checked against `max_steps` before any
/// simulation runs.
pub fn run_battery_with_limit(
    tiers: &[Tier],
    max_steps: u64,
) -> Result<FidelityReport, BatteryError> {
    validate_tiers(tiers)?;
    let required = battery_steps(tiers)? + u128::from(G1_PROBE_MAX_STEPS);
    if required > u128::from(max_steps) {
        return Err(BatteryError::StepBudget {
            required,
            limit: max_steps,
        });
    }

    let mut checks = Vec::new();
    for tier in tiers {
        for check in tier_checks(tier, max_steps)? {
            checks.push(TierCheck {
                tier: tier.name.clone(),
                check,
            });
        }
    }
    let probe = g1_zero_torque_probe().map_err(BatteryError::Probe)?;
    Ok(FidelityReport {
        battery: BATTERY.to_owned(),
        tier_set: if tiers == declared_tiers() {
            TierSet::Declared
        } else {
            TierSet::Experimental
        },
        tiers: tiers.to_vec(),
        checks,
        not_applicable: not_applicable_checks(),
        contact: contact_applicability(tiers),
        probe: vec![probe],
    })
}

fn validate_tiers(tiers: &[Tier]) -> Result<(), BatteryError> {
    if tiers.is_empty() {
        return Err(BatteryError::NoTiers);
    }
    for (i, tier) in tiers.iter().enumerate() {
        if tiers[..i].iter().any(|t| t.name == tier.name) {
            return Err(BatteryError::DuplicateTier(tier.name.clone()));
        }
        if !tier.timestep.is_positive() {
            return Err(BatteryError::NonPositiveTimestep(tier.name.clone()));
        }
        if tier.integrator != INTEGRATOR_SEMI_IMPLICIT_EULER {
            return Err(BatteryError::UnsupportedIntegrator {
                tier: tier.name.clone(),
                integrator: tier.integrator.clone(),
            });
        }
        if tier.solver != SOLVER_EXPLICIT_ABA {
            return Err(BatteryError::UnsupportedSolver {
                tier: tier.name.clone(),
                solver: tier.solver.clone(),
            });
        }
    }
    Ok(())
}

fn validate_check_names(checks: &[&dyn FidelityCheck]) -> Result<(), BatteryError> {
    for (i, check) in checks.iter().enumerate() {
        if checks[..i].iter().any(|other| other.name() == check.name()) {
            return Err(BatteryError::DuplicateCheck(check.name().to_owned()));
        }
    }
    Ok(())
}

fn check_steps(checks: &[&dyn FidelityCheck], tiers: &[Tier]) -> Result<u128, BatteryError> {
    let mut total = 0_u128;
    for tier in tiers {
        for check in checks {
            total = total
                .checked_add(u128::from(check.steps(tier)))
                .ok_or(BatteryError::StepCountOverflow)?;
        }
    }
    Ok(total)
}

/// The built-in checks, in [`CHECK_NAMES`] order.
const BUILTIN_CHECKS: [&dyn FidelityCheck; 7] = [
    &PendulumPeriod,
    &PendulumTrajectory,
    &PdEquilibrium,
    &AbaRneClosedForm,
    &EnergyDrift,
    &ArticulatedEnergyDrift,
    &FreeFall,
];

fn tier_checks(tier: &Tier, max_steps: u64) -> Result<Vec<Check>, BatteryError> {
    run_checks_inner(&BUILTIN_CHECKS, tier, false, max_steps)
}

struct PendulumPeriod;

impl FidelityCheck for PendulumPeriod {
    fn name(&self) -> &'static str {
        "pendulum_period"
    }
    fn system(&self) -> &'static str {
        system_of(self.name())
    }
    fn tolerance(&self) -> f64 {
        PERIOD_TOLERANCE
    }
    fn measure(&self, tier: &Tier) -> Result<Option<Measurement>, BatteryError> {
        Ok(Some(pendulum_period(tier.timestep)?))
    }
    fn steps(&self, tier: &Tier) -> u64 {
        steps_for(PERIOD_HORIZON_NANOS, tier.timestep)
    }
}

struct PendulumTrajectory;

impl FidelityCheck for PendulumTrajectory {
    fn name(&self) -> &'static str {
        "pendulum_trajectory"
    }
    fn system(&self) -> &'static str {
        system_of(self.name())
    }
    fn tolerance(&self) -> f64 {
        TRAJECTORY_REFERENCE_TOLERANCE
    }
    fn series_tolerance(&self, tier: &Tier) -> Option<oracle::Tolerance> {
        Some(oracle::Tolerance {
            abs: if tier.timestep.nanos() <= 100_000 {
                TRAJECTORY_REFERENCE_TOLERANCE
            } else {
                TRAJECTORY_COARSE_TOLERANCE
            },
            rel: 0.0,
        })
    }
    fn measure(&self, tier: &Tier) -> Result<Option<Measurement>, BatteryError> {
        Ok(Some(pendulum_trajectory(
            tier.timestep,
            &self.series_tolerance(tier).expect("series check"),
        )?))
    }
    fn steps(&self, tier: &Tier) -> u64 {
        steps_for(TRAJECTORY_HORIZON_NANOS, tier.timestep)
    }
}

struct PdEquilibrium;

impl FidelityCheck for PdEquilibrium {
    fn name(&self) -> &'static str {
        "pd_equilibrium"
    }
    fn system(&self) -> &'static str {
        system_of(self.name())
    }
    fn tolerance(&self) -> f64 {
        PD_TOLERANCE
    }
    fn measure(&self, tier: &Tier) -> Result<Option<Measurement>, BatteryError> {
        Ok(Some(pd_equilibrium(tier.timestep)?))
    }
    fn steps(&self, tier: &Tier) -> u64 {
        steps_for(PD_HORIZON_NANOS, tier.timestep)
    }
}

struct AbaRneClosedForm;

impl FidelityCheck for AbaRneClosedForm {
    fn name(&self) -> &'static str {
        "aba_rne_closed_form"
    }
    fn system(&self) -> &'static str {
        system_of(self.name())
    }
    fn tolerance(&self) -> f64 {
        ABA_RNE_TOLERANCE
    }
    fn measure(&self, _tier: &Tier) -> Result<Option<Measurement>, BatteryError> {
        Ok(Some(aba_rne_closed_form()))
    }
    fn steps(&self, _tier: &Tier) -> u64 {
        0
    }
}

struct EnergyDrift;

impl FidelityCheck for EnergyDrift {
    fn name(&self) -> &'static str {
        "energy_drift"
    }
    fn system(&self) -> &'static str {
        system_of(self.name())
    }
    fn tolerance(&self) -> f64 {
        ENERGY_TOLERANCE
    }
    fn measure(&self, tier: &Tier) -> Result<Option<Measurement>, BatteryError> {
        Ok(Some(energy_drift(tier.timestep)?))
    }
    fn steps(&self, tier: &Tier) -> u64 {
        steps_for(ENERGY_HORIZON_NANOS, tier.timestep)
    }
}

struct ArticulatedEnergyDrift;

impl FidelityCheck for ArticulatedEnergyDrift {
    fn name(&self) -> &'static str {
        "articulated_energy_drift"
    }
    fn system(&self) -> &'static str {
        system_of(self.name())
    }
    fn tolerance(&self) -> f64 {
        ARTICULATED_ENERGY_TOLERANCE
    }
    fn measure(&self, tier: &Tier) -> Result<Option<Measurement>, BatteryError> {
        Ok(Some(articulated_energy_drift(tier.timestep)?))
    }
    fn steps(&self, tier: &Tier) -> u64 {
        steps_for(ARTICULATED_ENERGY_HORIZON_NANOS, tier.timestep)
    }
}

struct FreeFall;

impl FidelityCheck for FreeFall {
    fn name(&self) -> &'static str {
        "free_fall"
    }
    fn system(&self) -> &'static str {
        system_of(self.name())
    }
    fn tolerance(&self) -> f64 {
        FREE_FALL_TOLERANCE
    }
    fn measure(&self, tier: &Tier) -> Result<Option<Measurement>, BatteryError> {
        Ok(Some(free_fall(tier.timestep)?))
    }
    fn steps(&self, tier: &Tier) -> u64 {
        steps_for(FREE_FALL_HORIZON_NANOS, tier.timestep)
    }
}

fn pendulum_params() -> PendulumParams {
    PendulumParams {
        length: PENDULUM_LENGTH,
        mass: PENDULUM_MASS,
        gravity: GRAVITY,
        damping: 0.0,
    }
}

/// Total integration steps the built-in checks need at `tiers`. Timesteps
/// must already be validated as positive.
fn battery_steps(tiers: &[Tier]) -> Result<u128, BatteryError> {
    check_steps(&BUILTIN_CHECKS, tiers)
}

/// Whole steps needed to cover `horizon_nanos`, rounded up.
fn steps_for(horizon_nanos: u64, dt: Duration) -> u64 {
    horizon_nanos.div_ceil(dt.nanos() as u64)
}

/// Relative error of the measured period against the exact elliptic period
/// from [`oracle::pendulum_period`]. The period is taken between two
/// successive upward zero crossings of the angle, with linear interpolation
/// inside the step. The sample index is the step of the second crossing.
fn pendulum_period(dt: Duration) -> Result<Measurement, BatteryError> {
    let exact = oracle::pendulum_period(PERIOD_AMPLITUDE) * (PENDULUM_LENGTH / GRAVITY).sqrt();
    let mut pendulum = Pendulum::new(
        pendulum_params(),
        PendulumState {
            theta: PERIOD_AMPLITUDE,
            omega: 0.0,
        },
    );
    let dt_s = dt.as_secs_f64();
    let mut crossings = Vec::with_capacity(2);
    let mut second_step = 0_usize;
    for k in 0..steps_for(PERIOD_HORIZON_NANOS, dt) {
        let before = pendulum.state().theta;
        pendulum.step(0.0, dt).map_err(BatteryError::Pendulum)?;
        let after = pendulum.state().theta;
        if before < 0.0 && after >= 0.0 && crossings.len() < 2 {
            let fraction = -before / (after - before);
            crossings.push((k as f64 + fraction) * dt_s);
            second_step = k as usize + 1;
        }
    }
    // A horizon that holds fewer than two crossings yields an infinite
    // measurement, which fails the check rather than passing silently.
    let measured = match crossings.as_slice() {
        [first, second] => ((second - first) - exact).abs() / exact,
        _ => f64::INFINITY,
    };
    Ok(Measurement::new(measured, second_step))
}

/// Compares every angle of a free 1-rad pendulum with the elliptic solution.
/// The reference uses dimensionless time `t * sqrt(g/L)`; index zero is the
/// initial state, index `n` is the state after `n` integration steps.
fn pendulum_trajectory(
    dt: Duration,
    tolerance: &oracle::Tolerance,
) -> Result<Measurement, BatteryError> {
    let mut pendulum = Pendulum::new(
        pendulum_params(),
        PendulumState {
            theta: PERIOD_AMPLITUDE,
            omega: 0.0,
        },
    );
    let count = steps_for(TRAJECTORY_HORIZON_NANOS, dt) as usize;
    let mut reference = Vec::with_capacity(count + 1);
    let mut candidate = Vec::with_capacity(count + 1);
    let dt_s = dt.as_secs_f64();
    let scale = (GRAVITY / PENDULUM_LENGTH).sqrt();
    for sample in 0..=count {
        reference.push(oracle::pendulum_closed_form(
            PERIOD_AMPLITUDE,
            sample as f64 * dt_s * scale,
        ));
        candidate.push(pendulum.state().theta);
        if sample < count {
            pendulum.step(0.0, dt).map_err(BatteryError::Pendulum)?;
        }
    }
    let result =
        oracle::compare_series(&reference, &candidate, tolerance).map_err(BatteryError::Oracle)?;
    // The battery's series bound is absolute (rel=0), so this measured
    // maximum and index reproduce the oracle pass flag in the TOML reader.
    debug_assert_eq!(
        result.pass,
        result.max_error.is_finite() && result.max_error <= tolerance.abs
    );
    Ok(Measurement::new(result.max_error, result.at_sample))
}

/// A PD-held pendulum with gravity compensation must settle at its target.
fn pd_equilibrium(dt: Duration) -> Result<Measurement, BatteryError> {
    let params = pendulum_params();
    let mut pendulum = Pendulum::new(
        params,
        PendulumState {
            theta: PD_START,
            omega: 0.0,
        },
    );
    for _ in 0..steps_for(PD_HORIZON_NANOS, dt) {
        let state = pendulum.state();
        let gravity_compensation = params.mass * params.gravity * params.length * state.theta.sin();
        let torque = PD_KP * (PD_TARGET - state.theta) - PD_KD * state.omega + gravity_compensation;
        pendulum.step(torque, dt).map_err(BatteryError::Pendulum)?;
    }
    let measured = (pendulum.state().theta - PD_TARGET).abs();
    Ok(Measurement::new(
        measured,
        steps_for(PD_HORIZON_NANOS, dt) as usize,
    ))
}

/// Joint forward and inverse dynamics of a point-mass pendulum against the
/// closed form `qdd = tau / (m L^2) + (g / L) cos q` and its inverse
/// `tau = m L^2 qdd - m g L cos q`. No time integration is involved, so the
/// value is identical at every tier.
fn aba_rne_closed_form() -> Measurement {
    let mass = PENDULUM_MASS;
    let length = PENDULUM_LENGTH;
    let profile = point_mass_pendulum(mass, length);
    let body = ArticulatedBody::from_profile(&profile);
    let gravity = Vec3::new(0.0, 0.0, -GRAVITY);
    let inertia = mass * length * length;

    let mut worst = 0.0_f64;
    let mut worst_sample = 0_usize;
    for sample in 0..16 {
        let q = -std::f64::consts::PI + sample as f64 * std::f64::consts::TAU / 16.0;
        let qd = 0.5;
        let qdd = 1.2;
        let tau_closed = inertia * qdd - mass * GRAVITY * length * q.cos();
        match body.inverse_dynamics(&[q], &[qd], &[qdd], gravity) {
            Ok(tau) => {
                let error = (tau[0] - tau_closed).abs();
                if error > worst {
                    worst = error;
                    worst_sample = sample;
                }
            }
            Err(_) => return Measurement::new(f64::INFINITY, sample),
        }
        let qdd_closed = tau_closed / inertia + (GRAVITY / length) * q.cos();
        match body.forward_dynamics(&[q], &[qd], &[tau_closed], gravity) {
            Ok(acc) => {
                let error = (acc[0] - qdd_closed).abs();
                if error > worst {
                    worst = error;
                    worst_sample = sample;
                }
            }
            Err(_) => return Measurement::new(f64::INFINITY, sample),
        }
    }
    Measurement::new(worst, worst_sample)
}

/// Maximum relative mechanical-energy drift over every step of an undamped,
/// unforced pendulum, computed by [`oracle::energy_drift`]. The sample is the
/// step where the largest drift is attained.
fn energy_drift(dt: Duration) -> Result<Measurement, BatteryError> {
    let mut pendulum = Pendulum::new(
        pendulum_params(),
        PendulumState {
            theta: PERIOD_AMPLITUDE,
            omega: 0.0,
        },
    );
    let mut energies = Vec::with_capacity(steps_for(ENERGY_HORIZON_NANOS, dt) as usize + 1);
    energies.push(pendulum.energy());
    for _ in 0..steps_for(ENERGY_HORIZON_NANOS, dt) {
        pendulum.step(0.0, dt).map_err(BatteryError::Pendulum)?;
        energies.push(pendulum.energy());
    }
    Ok(series_drift(&energies))
}

/// Largest energy drift over a series and the step where it is attained. A
/// non-finite or empty series yields `NaN`, which fails the check.
fn series_drift(energies: &[f64]) -> Measurement {
    let measured = oracle::energy_drift(energies);
    let sample = energies
        .iter()
        .enumerate()
        .skip(1)
        .max_by(|(_, a), (_, b)| {
            ((**a) - energies[0])
                .abs()
                .total_cmp(&((**b) - energies[0]).abs())
        })
        .map_or(0, |(i, _)| i);
    Measurement::new(measured, sample)
}

/// Relative mechanical-energy drift of a two-link planar chain released from
/// rest, with no torque applied. Kinetic energy comes from the articulated
/// body and potential energy from forward kinematics of the same profile.
fn articulated_energy_drift(dt: Duration) -> Result<Measurement, BatteryError> {
    let profile = double_pendulum(CHAIN_MASS_1, CHAIN_LENGTH_1, CHAIN_MASS_2, CHAIN_LENGTH_2);
    let body = ArticulatedBody::from_profile(&profile);
    let gravity = Vec3::new(0.0, 0.0, -GRAVITY);
    let mut state = MultibodyState::at_rest(CHAIN_START.to_vec(), body.dof());
    let energy = |state: &MultibodyState| -> Result<f64, BatteryError> {
        let kinetic = body
            .kinetic_energy(&state.q, &state.qd)
            .map_err(BatteryError::Dynamics)?;
        Ok(kinetic + potential_energy(&profile, &state.q, gravity)?)
    };
    let e0 = energy(&state)?;
    let tau = vec![0.0; body.dof()];
    let steps = steps_for(ARTICULATED_ENERGY_HORIZON_NANOS, dt);
    for k in 0..steps {
        if state.step(&body, &tau, gravity, dt).is_err() {
            return Ok(Measurement::new(f64::NAN, k as usize + 1));
        }
    }
    let e1 = energy(&state)?;
    let measured = (e1 - e0).abs() / e0.abs();
    Ok(Measurement::new(measured, steps as usize))
}

/// Sum over links of `-m g . p_com`, with each centre of mass placed by
/// forward kinematics.
fn potential_energy(profile: &RobotProfile, q: &[f64], gravity: Vec3) -> Result<f64, BatteryError> {
    let poses = forward_kinematics(profile, q).map_err(BatteryError::Kinematics)?;
    let mut energy = 0.0;
    for (index, link) in profile.links.iter().enumerate() {
        let pose = poses
            .by_index(index)
            .expect("forward_kinematics returns one pose per link");
        let com_world = pose.apply(link.mass.center_of_mass);
        energy -= link.mass.mass.value() * gravity.dot(com_world);
    }
    Ok(energy)
}

/// A free body dropped from rest falls `g t^2 / 2`. Semi-implicit Euler gives
/// `g t^2 / 2 + g t dt / 2` after `t` seconds, so the relative error is
/// `dt / t`. It shrinks with the timestep but is not fixed by it: a 3 s step
/// over the 1 s horizon gives a relative error of 1.
fn free_fall(dt: Duration) -> Result<Measurement, BatteryError> {
    let profile = free_body(1.0, InertiaTensor::diagonal(1.0, 1.0, 1.0));
    let body = ArticulatedBody::from_profile(&profile);
    let gravity = Vec3::new(0.0, 0.0, -GRAVITY);
    let mut q = vec![0.0; body.position_width()];
    q[3] = 1.0;
    let mut state = MultibodyState::at_rest(q, body.dof());
    let tau = vec![0.0; body.dof()];
    let steps = steps_for(FREE_FALL_HORIZON_NANOS, dt);
    for _ in 0..steps {
        state
            .step(&body, &tau, gravity, dt)
            .map_err(BatteryError::Dynamics)?;
    }
    let t = steps as f64 * dt.as_secs_f64();
    let z_exact = -0.5 * GRAVITY * t * t;
    let measured = (state.q[2] - z_exact).abs() / z_exact.abs();
    Ok(Measurement::new(measured, steps as usize))
}

/// A revolute two-link planar chain: joint1 at the origin carries link1 (point
/// mass at local +X offset `l1`); joint2 at link1's tip carries link2 (point
/// mass at local +X offset `l2`). Both axes are +Y.
fn double_pendulum(m1: f64, l1: f64, m2: f64, l2: f64) -> RobotProfile {
    let root = empty_link(0, "anchor", MassProperties::MASSLESS);
    let joint1 = revolute(0, "joint1", 0, 1, RigidPose::IDENTITY);
    let link1 = empty_link(
        1,
        "link1",
        MassProperties {
            mass: Kilograms(m1),
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
    );
    let link2 = empty_link(
        2,
        "link2",
        MassProperties {
            mass: Kilograms(m2),
            center_of_mass: Vec3::new(l2, 0.0, 0.0),
            inertia: InertiaTensor::ZERO,
        },
    );
    wrap_profile(vec![joint1, joint2], vec![root, link1, link2])
}

/// The scalar tolerance the battery declares for `name`, or `None` for a name
/// the battery does not run. The series check uses a tier-specific tolerance
/// which the reader validates separately.
fn canonical_tolerance(name: &str) -> Option<f64> {
    BUILTIN_CHECKS
        .iter()
        .find(|check| check.name() == name)
        .map(|check| check.tolerance())
}

/// A revolute point-mass pendulum: one joint on the world, bob at local +X.
fn point_mass_pendulum(mass: f64, length: f64) -> RobotProfile {
    let root = empty_link(0, "anchor", MassProperties::MASSLESS);
    let joint = revolute(0, "pivot", 0, 1, RigidPose::IDENTITY);
    let bob = empty_link(
        1,
        "bob",
        MassProperties {
            mass: Kilograms(mass),
            center_of_mass: Vec3::new(length, 0.0, 0.0),
            inertia: InertiaTensor::ZERO,
        },
    );
    wrap_profile(vec![joint], vec![root, bob])
}

/// A single free body attached to a massless `world` link via a `Free` joint.
fn free_body(mass: f64, inertia: InertiaTensor) -> RobotProfile {
    let world = empty_link(0, "world", MassProperties::MASSLESS);
    let joint = JointSpec {
        id: JointId::new(0),
        name: Name::new("base").expect("valid literal"),
        kind: JointKind::Free,
        parent: LinkId::new(0),
        child: LinkId::new(1),
        origin: RigidPose::IDENTITY,
        axis: Vec3::ZERO,
        limits: JointLimits::unbounded(RadiansPerSecond(1000.0), NewtonMetres(1000.0)),
        sign: 1.0,
        zero_offset: 0.0,
    };
    let body = empty_link(
        1,
        "body",
        MassProperties {
            mass: Kilograms(mass),
            center_of_mass: Vec3::ZERO,
            inertia,
        },
    );
    wrap_profile(vec![joint], vec![world, body])
}

fn empty_link(id: u32, name: &str, mass: MassProperties) -> LinkSpec {
    LinkSpec {
        id: LinkId::new(id),
        name: Name::new(name).expect("valid literal"),
        mass,
        collision: Vec::new(),
    }
}

fn revolute(id: u32, name: &str, parent: u32, child: u32, origin: RigidPose) -> JointSpec {
    JointSpec {
        id: JointId::new(id),
        name: Name::new(name).expect("valid literal"),
        kind: JointKind::Revolute,
        parent: LinkId::new(parent),
        child: LinkId::new(child),
        origin,
        axis: Vec3::unit_y(),
        limits: JointLimits::bounded(
            Range::<Radians>::new(
                Radians(-core::f64::consts::PI),
                Radians(core::f64::consts::PI),
            )
            .expect("valid literal range"),
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
        name: Name::new("fidelity-battery").expect("valid literal"),
        profile_version: "fidelity-battery".to_owned(),
        joints,
        links,
        actuators: Vec::new(),
        sensors: Vec::new(),
        root_link: LinkId::new(0),
        gravity: GravityConvention::minus_z_standard(FrameId::WORLD),
        firmware: FirmwareContract {
            version: "fidelity-battery".to_owned(),
            command_expiration: Duration::ZERO,
            watchdog_timeout: Duration::ZERO,
            has_independent_estop: false,
            allowed_modes: Vec::new(),
        },
        extensions: ExtensionTable::new(),
    }
}

fn tier_table(tier: &Tier) -> Table {
    let mut t = Table::new();
    t.insert("name", Value::String(tier.name.clone()));
    t.insert("timestep_ns", Value::Integer(tier.timestep.nanos()));
    t.insert("integrator", Value::String(tier.integrator.clone()));
    t.insert("solver", Value::String(tier.solver.clone()));
    t
}

fn check_table(tc: &TierCheck) -> Table {
    let mut t = Table::new();
    t.insert("tier", Value::String(tc.tier.clone()));
    t.insert("name", Value::String(tc.check.name.clone()));
    t.insert("system", Value::String(tc.check.system.clone()));
    if let Some(tol) = tc.check.series_tolerance {
        t.insert("tolerance_abs", Value::Float(tol.abs));
        t.insert("tolerance_rel", Value::Float(tol.rel));
    } else {
        t.insert("tolerance", Value::Float(tc.check.tolerance));
    }
    t.insert("measured", Value::Float(tc.check.measured));
    t.insert("sample", Value::Integer(tc.check.sample as i64));
    t.insert("pass", Value::Boolean(tc.check.pass));
    t
}

fn not_applicable_table(na: &NotApplicable) -> Table {
    let mut t = Table::new();
    t.insert("name", Value::String(na.name.clone()));
    t.insert("reason", Value::String(na.reason.clone()));
    t
}

fn invalid(reason: String) -> ReportError {
    ReportError::Invalid(reason)
}

/// Rejects any key of `t` that is not in `allowed`.
fn reject_unknown_keys(t: &Table, allowed: &[&str], kind: &str) -> Result<(), ReportError> {
    for (key, _) in t.iter() {
        if !allowed.contains(&key) {
            return Err(invalid(format!("{kind} has unknown key {key:?}")));
        }
    }
    Ok(())
}

/// The `[[key]]` array of tables under `key`, empty when absent.
fn tables<'a>(root: &'a Table, key: &str) -> Result<Vec<&'a Table>, ReportError> {
    let Some(value) = root.get(key) else {
        return Ok(Vec::new());
    };
    let items = value
        .as_array()
        .ok_or_else(|| invalid(format!("{key} is not an array")))?;
    items
        .iter()
        .map(|item| match item {
            Value::Table(t) => Ok(t),
            _ => Err(invalid(format!("{key} entry is not a table"))),
        })
        .collect()
}

fn read_tier(t: &Table) -> Result<Tier, ReportError> {
    let nanos = t
        .require_integer("timestep_ns")
        .map_err(ReportError::Toml)?;
    if nanos <= 0 {
        return Err(invalid(format!("timestep_ns is {nanos}, must be positive")));
    }
    Ok(Tier {
        name: t.require_str("name").map_err(ReportError::Toml)?.to_owned(),
        timestep: Duration::from_nanos(nanos),
        integrator: t
            .require_str("integrator")
            .map_err(ReportError::Toml)?
            .to_owned(),
        solver: t
            .require_str("solver")
            .map_err(ReportError::Toml)?
            .to_owned(),
    })
}

/// The `sample` field of a check row. Must be a non-negative integer.
fn read_sample(t: &Table) -> Result<usize, ReportError> {
    let sample = t.require_integer("sample").map_err(ReportError::Toml)?;
    usize::try_from(sample)
        .map_err(|_| invalid(format!("sample is {sample}, must be non-negative")))
}

fn read_check(t: &Table) -> Result<TierCheck, ReportError> {
    let name = t.require_str("name").map_err(ReportError::Toml)?;
    let system = t.require_str("system").map_err(ReportError::Toml)?;
    let canonical = system_of(name);
    if canonical.is_empty() {
        return Err(invalid(format!("check {name:?} is not a battery check")));
    }
    if system != canonical {
        return Err(invalid(format!(
            "check {name:?} is labelled {system:?}, its system is {canonical:?}"
        )));
    }
    let series_tolerance = if name == "pendulum_trajectory" {
        if t.get("tolerance").is_some() {
            return Err(invalid(
                "series check must not have scalar tolerance".to_owned(),
            ));
        }
        let abs = t
            .require_float("tolerance_abs")
            .map_err(ReportError::Toml)?;
        let rel = t
            .require_float("tolerance_rel")
            .map_err(ReportError::Toml)?;
        if !(abs.is_finite() && abs >= 0.0 && rel.is_finite() && rel >= 0.0) {
            return Err(invalid(
                "series tolerance must be finite and non-negative".to_owned(),
            ));
        }
        Some(oracle::Tolerance { abs, rel })
    } else {
        if t.get("tolerance_abs").is_some() || t.get("tolerance_rel").is_some() {
            return Err(invalid(
                "scalar check must not have series tolerance".to_owned(),
            ));
        }
        None
    };
    Ok(TierCheck {
        tier: t.require_str("tier").map_err(ReportError::Toml)?.to_owned(),
        check: Check {
            name: name.to_owned(),
            system: system.to_owned(),
            tolerance: if let Some(tol) = series_tolerance {
                tol.abs
            } else {
                t.require_float("tolerance").map_err(ReportError::Toml)?
            },
            series_tolerance,
            measured: t.require_float("measured").map_err(ReportError::Toml)?,
            sample: read_sample(t)?,
            pass: t.require_bool("pass").map_err(ReportError::Toml)?,
        },
    })
}

fn contact_table(c: &ContactApplicability) -> Table {
    let mut t = Table::new();
    t.insert("tier", Value::String(c.tier.clone()));
    t.insert("system", Value::String(c.system.clone()));
    t.insert("status", Value::String(c.status.as_str().to_owned()));
    t.insert("reason", Value::String(c.reason.clone()));
    t
}

fn read_contact(t: &Table) -> Result<ContactApplicability, ReportError> {
    let status_text = t.require_str("status").map_err(ReportError::Toml)?;
    let status = ContactStatus::parse(status_text)
        .ok_or_else(|| invalid(format!("unknown contact status {status_text:?}")))?;
    Ok(ContactApplicability {
        tier: t.require_str("tier").map_err(ReportError::Toml)?.to_owned(),
        system: t
            .require_str("system")
            .map_err(ReportError::Toml)?
            .to_owned(),
        status,
        reason: t
            .require_str("reason")
            .map_err(ReportError::Toml)?
            .to_owned(),
    })
}

fn probe_table(p: &ContactProbe) -> Table {
    let mut t = Table::new();
    t.insert("scenario", Value::String(p.scenario.clone()));
    t.insert("system", Value::String(p.system.clone()));
    t.insert("status", Value::String(p.status.as_str().to_owned()));
    if let Some(step) = p.step {
        t.insert("step", Value::Integer(step as i64));
    }
    if let Some(error) = &p.error {
        t.insert("error", Value::String(error.clone()));
    }
    t.insert("reason", Value::String(p.reason.clone()));
    t
}

fn read_probe(t: &Table) -> Result<ContactProbe, ReportError> {
    let status_text = t.require_str("status").map_err(ReportError::Toml)?;
    let status = ProbeStatus::parse(status_text)
        .ok_or_else(|| invalid(format!("unknown probe status {status_text:?}")))?;
    let step = match t.get("step") {
        None => None,
        Some(_) => {
            let value = t.require_integer("step").map_err(ReportError::Toml)?;
            Some(
                u64::try_from(value)
                    .map_err(|_| invalid(format!("probe step {value} is negative")))?,
            )
        }
    };
    let error = match t.get("error") {
        None => None,
        Some(_) => Some(
            t.require_str("error")
                .map_err(ReportError::Toml)?
                .to_owned(),
        ),
    };
    Ok(ContactProbe {
        scenario: t
            .require_str("scenario")
            .map_err(ReportError::Toml)?
            .to_owned(),
        system: t
            .require_str("system")
            .map_err(ReportError::Toml)?
            .to_owned(),
        status,
        step,
        error,
        reason: t
            .require_str("reason")
            .map_err(ReportError::Toml)?
            .to_owned(),
    })
}

fn read_not_applicable(t: &Table) -> Result<NotApplicable, ReportError> {
    Ok(NotApplicable {
        name: t.require_str("name").map_err(ReportError::Toml)?.to_owned(),
        reason: t
            .require_str("reason")
            .map_err(ReportError::Toml)?
            .to_owned(),
    })
}

fn reject_duplicates<'a>(
    names: impl Iterator<Item = &'a str>,
    kind: &str,
) -> Result<(), ReportError> {
    let mut seen: Vec<&str> = Vec::new();
    for name in names {
        if seen.contains(&name) {
            return Err(invalid(format!("{kind} {name:?} appears twice")));
        }
        seen.push(name);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    struct InapplicableCheck;

    impl FidelityCheck for InapplicableCheck {
        fn name(&self) -> &'static str {
            "test_inapplicable"
        }
        fn system(&self) -> &'static str {
            "test_system"
        }
        fn tolerance(&self) -> f64 {
            0.1
        }
        fn measure(&self, _tier: &Tier) -> Result<Option<Measurement>, BatteryError> {
            Ok(None)
        }
        fn steps(&self, _tier: &Tier) -> u64 {
            0
        }
    }

    static DECLARED: OnceLock<FidelityReport> = OnceLock::new();

    /// The declared battery, run once per test process. Tests that only need
    /// a valid declared report to serialize or tamper with share this run;
    /// determinism and reproduction tests still run the battery afresh.
    fn cached() -> &'static FidelityReport {
        DECLARED.get_or_init(|| run_battery(&declared_tiers()).unwrap())
    }

    fn declared() -> FidelityReport {
        cached().clone()
    }

    fn declared_check(tier: &str, name: &str) -> Check {
        cached()
            .checks
            .iter()
            .find(|c| c.tier == tier && c.check.name == name)
            .unwrap()
            .check
            .clone()
    }

    #[test]
    fn built_in_registry_matches_archival_name_system_and_tolerance_tables() {
        assert_eq!(BUILTIN_CHECKS.len(), CHECK_NAMES.len());
        assert_eq!(BUILTIN_CHECKS.len(), CHECK_SYSTEMS.len());
        for (index, check) in BUILTIN_CHECKS.iter().enumerate() {
            assert_eq!(check.name(), CHECK_NAMES[index]);
            assert_eq!(check.system(), CHECK_SYSTEMS[index]);
            assert_eq!(canonical_tolerance(check.name()), Some(check.tolerance()));
        }
        assert!(canonical_tolerance("custom_check").is_none());
    }

    #[test]
    fn built_in_archival_runner_rejects_an_inapplicable_check() {
        let tier = Tier::semi_implicit("test", Duration::from_millis(1));
        let checks: [&dyn FidelityCheck; 1] = [&InapplicableCheck];
        assert_eq!(
            run_checks_inner(&checks, &tier, false, MAX_BATTERY_STEPS),
            Err(BatteryError::InapplicableCheck {
                name: "test_inapplicable".to_owned(),
                tier: "test".to_owned(),
            })
        );
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("fidelity-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn check_pass_rule_rejects_non_finite_and_over_tolerance() {
        assert!(Check::new("a", 1.0, 0.5).pass);
        assert!(Check::new("a", 1.0, 1.0).pass);
        assert!(!Check::new("a", 1.0, 1.5).pass);
        assert!(!Check::new("a", 1.0, f64::NAN).pass);
        assert!(!Check::new("a", 1.0, f64::INFINITY).pass);
    }

    #[test]
    fn v7_report_round_trips_the_per_check_sample_index() {
        let report = declared();
        let text = report.to_toml();
        assert!(text.contains("schema_version = 7"));
        assert!(text.contains("battery = \"shared-analytic-v7\""));
        let parsed = FidelityReport::from_toml(&text).unwrap();
        assert_eq!(parsed, report);
        let stored = declared_check("reference", "aba_rne_closed_form");
        assert!(stored.sample < 16, "sample={}", stored.sample);
        let period = declared_check("reference", "pendulum_period");
        assert!(period.sample > 0, "period sample={}", period.sample);
        for (tier, expected) in [
            ("reference", TRAJECTORY_REFERENCE_TOLERANCE),
            ("coarse", TRAJECTORY_COARSE_TOLERANCE),
        ] {
            let trajectory = declared_check(tier, "pendulum_trajectory");
            let (low, high, sample) = if tier == "reference" {
                (1e-4, 2e-4, 37_435)
            } else {
                (1e-3, 2e-3, 3_744)
            };
            assert_eq!(trajectory.sample, sample, "{tier} worst sample");
            assert!(
                (low..high).contains(&trajectory.measured),
                "{tier} max error {} at {}",
                trajectory.measured,
                trajectory.sample
            );
            assert_eq!(
                trajectory.series_tolerance,
                Some(oracle::Tolerance {
                    abs: expected,
                    rel: 0.0
                })
            );
            assert!(
                trajectory.pass,
                "{tier} max error {} at {} exceeds {expected}",
                trajectory.measured, trajectory.sample
            );
        }
    }

    #[test]
    fn trajectory_tolerance_round_trips_and_rejects_tampering() {
        let report = declared();
        let text = report.to_toml();
        assert!(text.contains("tolerance_abs = 0.0005"));
        assert!(text.contains("tolerance_rel = 0.0"));
        assert_eq!(FidelityReport::from_toml(&text).unwrap(), report);
        let mut changed = report;
        let trajectory = changed
            .checks
            .iter_mut()
            .find(|c| c.check.name == "pendulum_trajectory")
            .unwrap();
        trajectory.check.series_tolerance.as_mut().unwrap().abs *= 2.0;
        assert_rejected(&changed.to_toml());
        let missing = text.replacen("tolerance_rel = 0.0\n", "", 1);
        assert!(FidelityReport::from_toml(&missing).is_err());
        let contradictory = text.replacen("tolerance_rel = 0.0", "tolerance_rel = 0.1", 1);
        assert_rejected(&contradictory);
    }

    #[test]
    fn trajectory_oracle_failure_is_not_silently_accepted() {
        let tier = declared_tiers().remove(0);
        let check = PendulumTrajectory;
        let mut tolerance = check.series_tolerance(&tier).unwrap();
        tolerance.abs = 1e-10;
        let measured = pendulum_trajectory(tier.timestep, &tolerance).unwrap();
        assert!(measured.sample > 0);
        assert!(!Check::for_series(check.name(), check.system(), tolerance, measured).pass);
    }

    #[test]
    fn g1_probe_fail_row_round_trips_strictly_without_changing_the_gate() {
        // The probe re-runs the G1 zero-torque fall at dt = 5e-4 and must record
        // the measured failure: status fail, step inside the recorded window, and
        // error SingularJointInertia. The row round-trips through the strict reader,
        // and all_pass() stays a function of the analytic checks only.
        let report = declared();
        let probe = g1_zero_torque_probe().expect("the G1 fixture must load");
        assert_eq!(probe.status, ProbeStatus::Fail);
        assert_eq!(probe.error.as_deref(), Some("SingularJointInertia"));
        let step = probe.step.unwrap();
        assert!((2_100..=2_400).contains(&step), "step={step}");
        let text = report.to_toml();
        assert!(text.contains("status = \"fail\""));
        assert!(text.contains("error = \"SingularJointInertia\""));
        let parsed = FidelityReport::from_toml(&text).unwrap();
        assert_eq!(parsed.probe, report.probe);
        assert!(parsed.all_pass());
    }

    #[test]
    fn tampered_probe_row_is_refused_by_the_strict_reader() {
        let text = declared().to_toml();
        let forged = text.replace("status = \"fail\"", "status = \"pass\"");
        assert!(matches!(
            FidelityReport::from_toml(&forged),
            Err(ReportError::Invalid(_))
        ));
    }

    #[test]
    fn probe_row_with_wrong_scenario_unknown_error_or_window_step_is_refused() {
        let text = declared().to_toml();
        let wrong_scenario = text.replace(
            "scenario = \"g1-zero-torque-dt-5e-4\"",
            "scenario = \"other\"",
        );
        let unknown_error = text.replace("error = \"SingularJointInertia\"", "error = \"Bogus\"");
        let step_outside = text.replace("step = 2236", "step = 4000");
        for forged in [wrong_scenario, unknown_error, step_outside] {
            assert!(matches!(
                FidelityReport::from_toml(&forged),
                Err(ReportError::Invalid(_))
            ));
        }
    }

    #[test]
    fn a_fresh_probe_run_matches_the_archived_row_and_respects_the_budget() {
        let report = declared();
        assert!(matches!(
            report.reproduce(1_000),
            Err(ReportError::Invalid(_))
        ));
        assert!(report.reproduce(1_000_000).is_ok());
    }

    #[test]
    fn reproduction_refuses_a_probe_row_that_differs_from_a_fresh_run() {
        let mut report = declared();
        report.probe[0].step = Some(2_101);
        assert!(matches!(
            report.reproduce(1_000_000),
            Err(ReportError::Invalid(_))
        ));
    }

    #[test]
    fn v7_reader_refuses_a_v6_report() {
        let text = declared()
            .to_toml()
            .replace("schema_version = 7", "schema_version = 6")
            .replace("shared-analytic-v7", "shared-analytic-v6");
        assert!(matches!(
            FidelityReport::from_toml(&text),
            Err(ReportError::Invalid(_))
        ));
    }

    #[test]
    fn v6_reader_refuses_a_negative_sample_index() {
        let text = declared()
            .to_toml()
            .lines()
            .map(|line| {
                if line.trim_start().starts_with("sample = ") {
                    "sample = -1".to_owned()
                } else {
                    line.to_owned()
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(matches!(
            FidelityReport::from_toml(&text),
            Err(ReportError::Invalid(_))
        ));
    }

    #[test]
    fn reproduction_rejects_a_tampered_trajectory_sample_index() {
        let mut report = declared();
        let trajectory = report
            .checks
            .iter_mut()
            .find(|row| row.tier == "reference" && row.check.name == "pendulum_trajectory")
            .unwrap();
        assert_ne!(trajectory.check.sample, 0);
        trajectory.check.sample = 0;

        let archived = FidelityReport::from_toml(&report.to_toml())
            .expect("a structurally valid report may contain an incorrect sample index");
        assert_eq!(
            archived
                .checks
                .iter()
                .find(|row| row.tier == "reference" && row.check.name == "pendulum_trajectory")
                .unwrap()
                .check
                .sample,
            0
        );
        assert!(matches!(
            archived.reproduce(1_000_000),
            Err(ReportError::Invalid(_))
        ));
    }

    #[test]
    fn battery_runs_every_check_at_every_declared_tier() {
        let report = declared();
        assert_eq!(report.tiers.len(), 2);
        assert_eq!(report.checks.len(), 2 * CHECK_NAMES.len());
        for tier in &report.tiers {
            let names: Vec<&str> = report
                .checks
                .iter()
                .filter(|c| c.tier == tier.name)
                .map(|c| c.check.name.as_str())
                .collect();
            assert_eq!(names, CHECK_NAMES);
        }
        let na: Vec<&str> = report
            .not_applicable
            .iter()
            .map(|n| n.name.as_str())
            .collect();
        assert_eq!(na, ["cpu_gpu_agreement", "contact_budget_exhaustion"]);
    }

    #[test]
    fn measured_pendulum_period_matches_the_elliptic_reference() {
        let exact = oracle::pendulum_period(PERIOD_AMPLITUDE) * (PENDULUM_LENGTH / GRAVITY).sqrt();
        // Small-amplitude limit is 2 pi sqrt(L/g); the finite-amplitude period is larger.
        let small = 2.0 * std::f64::consts::PI * (PENDULUM_LENGTH / GRAVITY).sqrt();
        assert!(exact > small);
        // Series T = T0 (1 + th^2/16 + 11 th^4/3072 + ...) at th = 1 rad gives 2.1393 s.
        assert!((exact - 2.1393).abs() < 1e-3, "exact {exact}");
        let measured = pendulum_period(Duration::from_micros(100)).unwrap();
        assert!(
            measured.value <= PERIOD_TOLERANCE,
            "period error {}",
            measured.value
        );
    }

    #[test]
    fn free_fall_and_aba_rne_pass_at_both_declared_tiers() {
        for tier in declared_tiers() {
            for name in ["free_fall", "aba_rne_closed_form"] {
                let c = declared_check(&tier.name, name);
                assert!(
                    c.pass,
                    "{} {name} measured {} tol {}",
                    tier.name, c.measured, c.tolerance
                );
            }
        }
    }

    #[test]
    fn coarser_timestep_has_larger_free_fall_error() {
        let fine = declared_check("reference", "free_fall");
        let coarse = declared_check("coarse", "free_fall");
        assert!(coarse.measured > fine.measured);
        // Semi-implicit Euler free fall has relative error exactly dt / t.
        assert!((fine.measured - 1e-4).abs() < 1e-9, "{}", fine.measured);
        assert!((coarse.measured - 1e-3).abs() < 1e-9, "{}", coarse.measured);
    }

    #[test]
    fn report_round_trips_through_toml_without_changing_any_value() {
        let report = declared();
        let text = report.to_toml();
        let back = FidelityReport::from_toml(&text).unwrap();
        assert_eq!(back, report);
        assert_eq!(back.to_toml(), text);
    }

    #[test]
    fn declared_report_rejects_a_removed_entire_tier() {
        let mut report = declared();
        report.tiers.pop();
        report.checks.retain(|c| c.tier == "reference");
        report.contact = contact_applicability(&report.tiers);
        report.probe = vec![g1_zero_torque_probe().unwrap()];
        assert!(FidelityReport::from_toml(&report.to_toml()).is_err());
        let switched = report
            .to_toml()
            .replace("tier_set = \"declared\"", "tier_set = \"experimental\"");
        assert_eq!(
            FidelityReport::from_toml(&switched).unwrap().tier_set,
            TierSet::Experimental
        );
    }

    #[test]
    fn declared_report_rejects_a_changed_timestep_even_with_canonical_checks() {
        let mut report = declared();
        report.tiers[1].timestep = Duration::from_micros(400);
        for c in &mut report.checks {
            if c.tier == "coarse" && c.check.name == "free_fall" {
                c.check = Check::new("free_fall", FREE_FALL_TOLERANCE, 0.0);
            }
        }
        assert!(FidelityReport::from_toml(&report.to_toml()).is_err());
    }

    #[test]
    fn report_is_byte_identical_across_runs() {
        let fresh = run_battery(&declared_tiers()).unwrap().to_toml();
        assert_eq!(fresh, cached().to_toml());
    }

    #[test]
    fn write_then_read_gives_the_same_report() {
        let dir = temp_dir("roundtrip");
        let report = declared();
        let path = write_report(&dir, &report).unwrap();
        assert_eq!(read_report(&path).unwrap(), report);
        // Writing identical bytes again is allowed; the file is unchanged.
        write_report(&dir, &report).unwrap();
    }

    #[test]
    fn different_bytes_cannot_overwrite_an_existing_report() {
        let dir = temp_dir("immutable");
        let report = declared();
        write_report(&dir, &report).unwrap();
        let mut changed = report.clone();
        changed.checks[0].check.measured += 1.0;
        changed.checks[0].check.pass = false;
        assert!(matches!(
            write_report(&dir, &changed),
            Err(ReportError::Store(_))
        ));
    }

    #[test]
    fn reader_rejects_a_wrong_schema_version() {
        let text = declared().to_toml().replace(
            &format!("schema_version = {SCHEMA_VERSION}"),
            "schema_version = 1",
        );
        assert!(matches!(
            FidelityReport::from_toml(&text),
            Err(ReportError::Invalid(_))
        ));
    }

    #[test]
    fn reader_rejects_a_wrong_schema_name() {
        let text = declared()
            .to_toml()
            .replace("robot-sim.fidelity-report", "other.report");
        assert!(matches!(
            FidelityReport::from_toml(&text),
            Err(ReportError::Invalid(_))
        ));
    }

    #[test]
    fn reader_rejects_a_pass_flag_that_contradicts_its_numbers() {
        let mut report = declared();
        report.checks[0].check.pass = !report.checks[0].check.pass;
        assert!(matches!(
            FidelityReport::from_toml(&report.to_toml()),
            Err(ReportError::Invalid(_))
        ));
    }

    #[test]
    fn reader_rejects_a_check_row_relabelled_to_another_system() {
        let text = honest_text().replacen(
            "system = \"standalone_pendulum\"",
            "system = \"double_pendulum_articulated\"",
            1,
        );
        assert_rejected(&text);
    }

    #[test]
    fn reader_rejects_a_check_row_with_an_unknown_name() {
        let text = honest_text().replacen(
            "name = \"pendulum_period\"",
            "name = \"sphere_contact_rest_height\"",
            1,
        );
        assert_rejected(&text);
    }

    #[test]
    fn reader_rejects_an_all_pass_that_contradicts_the_checks() {
        let text = declared()
            .to_toml()
            .replace("all_pass = true", "all_pass = false");
        assert!(matches!(
            FidelityReport::from_toml(&text),
            Err(ReportError::Invalid(_))
        ));
    }

    #[test]
    fn reader_rejects_a_truncated_file_without_partial_load() {
        let text = declared().to_toml();
        let cut = &text[..text.len() / 2];
        assert!(matches!(
            FidelityReport::from_toml(cut),
            Err(ReportError::Toml(_))
        ));
    }

    fn honest_text() -> String {
        cached().to_toml()
    }

    fn assert_rejected(text: &str) {
        assert!(
            matches!(
                FidelityReport::from_toml(text),
                Err(ReportError::Invalid(_))
            ),
            "accepted a tampered report"
        );
    }

    #[test]
    fn reader_rejects_a_report_with_a_removed_check_row() {
        let mut report = declared();
        report.checks.remove(0);
        assert_rejected(&report.to_toml());
        let mut failing =
            run_battery(&[Tier::semi_implicit("huge", Duration::from_millis(200))]).unwrap();
        failing.checks.retain(|c| c.check.pass);
        assert_rejected(&failing.to_toml());
    }

    #[test]
    fn reader_rejects_a_check_row_whose_tolerance_was_raised() {
        let mut report = declared();
        report.checks[0].check.tolerance = 1e9;
        assert_rejected(&report.to_toml());
    }

    #[test]
    fn reader_rejects_a_free_fall_tolerance_that_differs_from_the_battery_rule() {
        let mut report = declared();
        let free_fall = report
            .checks
            .iter_mut()
            .find(|c| c.check.name == "free_fall")
            .unwrap();
        free_fall.check.tolerance *= 2.0;
        assert_rejected(&report.to_toml());
    }

    #[test]
    fn reader_rejects_a_declared_tier_with_no_check_rows() {
        let mut report = declared();
        report
            .tiers
            .push(Tier::semi_implicit("empty", Duration::from_micros(50)));
        assert_rejected(&report.to_toml());
    }

    #[test]
    fn reader_rejects_a_report_with_no_tiers() {
        let mut report = declared();
        report.tiers.clear();
        report.checks.clear();
        assert_rejected(&report.to_toml());
    }

    #[test]
    fn reader_rejects_a_tier_with_an_undeclared_integrator_or_solver() {
        let mut report = declared();
        report.tiers[1].solver = "rk4-articulated".to_owned();
        assert_rejected(&report.to_toml());
    }

    #[test]
    fn reader_rejects_a_missing_not_applicable_entry() {
        let mut report = declared();
        report.not_applicable.pop();
        assert_rejected(&report.to_toml());
    }

    #[test]
    fn reader_rejects_a_wrong_battery_name() {
        let text = honest_text().replace(BATTERY, "other-battery");
        assert_rejected(&text);
        let stale = honest_text().replace(BATTERY, "shared-analytic-v1");
        assert_rejected(&stale);
    }

    #[test]
    fn reader_rejects_unknown_top_level_and_row_keys() {
        let text = honest_text();
        let top = text.replacen(
            &format!("schema_version = {SCHEMA_VERSION}\n"),
            &format!("schema_version = {SCHEMA_VERSION}\nextra = 1\n"),
            1,
        );
        assert_ne!(top, text);
        assert_rejected(&top);
        let row = text.replacen("[[check]]\n", "[[check]]\nextra = 1\n", 1);
        assert_ne!(row, text);
        assert_rejected(&row);
        let tier = text.replacen("[[tier]]\n", "[[tier]]\nextra = 1\n", 1);
        assert_ne!(tier, text);
        assert_rejected(&tier);
    }

    #[test]
    fn reader_rejects_pass_flags_forced_true_over_failing_numbers() {
        let mut report = declared();
        for c in &mut report.checks {
            c.check.pass = true;
        }
        report.checks[0].check.measured = 1.0;
        assert_rejected(&report.to_toml());
    }

    #[test]
    fn archival_reader_accepts_consistent_numbers_without_replaying_simulation() {
        let mut report = declared();
        let idx = report
            .checks
            .iter()
            .position(|c| c.tier == "coarse" && c.check.name == "energy_drift")
            .unwrap();
        report.checks[idx].check = Check::new("energy_drift", ENERGY_TOLERANCE, 5.0);
        let forged = FidelityReport::from_toml(&report.to_toml()).unwrap();
        assert!(!forged.all_pass());
        let dir = temp_dir("archival-read");
        let path = write_report(&dir, &forged).unwrap();
        assert_eq!(read_report(&path).unwrap(), forged);
        assert!(matches!(
            forged.reproduce(1_000_000),
            Err(ReportError::Invalid(_))
        ));
    }

    #[test]
    fn an_honest_report_passes_explicit_reproduction() {
        let report = declared();
        let back = FidelityReport::from_toml(&report.to_toml()).unwrap();
        back.reproduce(1_000_000).unwrap();
    }

    #[test]
    fn one_nanosecond_experimental_report_reads_without_running_and_reproduction_is_capped() {
        let mut report = declared();
        report.tier_set = TierSet::Experimental;
        report.tiers.truncate(1);
        report.tiers[0].name = "nano".to_owned();
        report.tiers[0].timestep = Duration::from_nanos(1);
        report.checks.retain(|c| c.tier == "reference");
        for c in &mut report.checks {
            c.tier = "nano".to_owned();
            if c.check.name == "free_fall" {
                c.check = Check::new("free_fall", FREE_FALL_TOLERANCE, 0.0);
            }
        }
        report.contact = contact_applicability(&report.tiers);
        report.probe = vec![g1_zero_torque_probe().unwrap()];
        let dir = temp_dir("nano-read");
        let path = write_report(&dir, &report).unwrap();
        assert_eq!(read_report(&path).unwrap(), report);
        assert!(matches!(
            report.reproduce(1_000_000),
            Err(ReportError::Invalid(_))
        ));
    }

    #[test]
    fn report_identity_is_external_and_rejects_even_structurally_valid_byte_edits() {
        let dir = temp_dir("identity");
        let report = declared();
        let (path, id) = write_report_with_id(&dir, &report).unwrap();
        assert_eq!(read_report_verified(&path, &id).unwrap(), report);
        let mut edited = report.clone();
        edited.checks[0].check = Check::new("pendulum_period", PERIOD_TOLERANCE, 0.0);
        std::fs::write(&path, edited.to_toml()).unwrap();
        assert_eq!(read_report(&path).unwrap(), edited);
        assert!(matches!(
            read_report_verified(&path, &id),
            Err(ReportError::Identity(_))
        ));
    }

    #[test]
    fn tier_set_metadata_must_be_present_and_cannot_relabel_declared_as_experimental() {
        let text = honest_text();
        assert_rejected(&text.replace("tier_set = \"declared\"", "tier_set = \"experimental\""));
        assert!(FidelityReport::from_toml(&text.replace("tier_set = \"declared\"\n", "")).is_err());
    }

    #[test]
    fn writer_rejects_incomplete_production_report_before_creating_a_file() {
        let dir = temp_dir("incomplete-write");
        let mut report = declared();
        report.tiers.pop();
        report.checks.retain(|c| c.tier == "reference");
        assert!(matches!(
            write_report(&dir, &report),
            Err(ReportError::Invalid(_))
        ));
        assert!(!dir.join(REPORT_FILE).exists());
    }

    #[test]
    fn oversized_archive_is_rejected_without_simulation() {
        let dir = temp_dir("oversized");
        let path = dir.join("oversized.toml");
        std::fs::write(&path, vec![b'x'; MAX_REPORT_BYTES + 1]).unwrap();
        assert!(matches!(read_report(&path), Err(ReportError::Invalid(_))));
    }

    #[test]
    fn non_finite_measurements_round_trip_as_failing_checks() {
        let mut report = declared();
        report.checks[0].check = Check::new("pendulum_period", PERIOD_TOLERANCE, f64::NAN);
        let pd = report
            .checks
            .iter_mut()
            .find(|c| c.tier == "reference" && c.check.name == "pd_equilibrium")
            .unwrap();
        pd.check = Check::new("pd_equilibrium", PD_TOLERANCE, f64::INFINITY);
        let back = FidelityReport::from_toml(&report.to_toml()).unwrap();
        assert!(back.checks[0].check.measured.is_nan());
        assert!(!back.checks[0].check.pass);
        let pd = back
            .checks
            .iter()
            .find(|c| c.tier == "reference" && c.check.name == "pd_equilibrium")
            .unwrap();
        assert!(pd.check.measured.is_infinite());
        assert!(!pd.check.pass);
    }

    #[test]
    fn production_pass_requires_the_declared_tier_set_and_every_check() {
        let declared = declared();
        assert!(declared.production_pass());
        let mut experimental = declared.clone();
        experimental.tier_set = TierSet::Experimental;
        assert!(experimental.all_pass());
        assert!(!experimental.production_pass());
        let mut failing = declared;
        failing.checks[0].check = Check::new("pendulum_period", PERIOD_TOLERANCE, 1.0);
        assert!(!failing.production_pass());
    }

    #[test]
    fn free_fall_fails_a_gross_three_second_step_and_passes_declared_tiers() {
        let coarse = free_fall(Duration::from_secs(3)).unwrap();
        assert!(
            coarse.value > FREE_FALL_TOLERANCE,
            "3 s step measured {}",
            coarse.value
        );
        let over = free_fall(Duration::from_millis(20)).unwrap();
        assert!(
            over.value > FREE_FALL_TOLERANCE,
            "20 ms step measured {}",
            over.value
        );
        for tier in declared_tiers() {
            let c = declared_check(&tier.name, "free_fall");
            assert!(c.pass, "{} measured {}", tier.name, c.measured);
        }
    }

    #[test]
    fn articulated_energy_drift_passes_declared_tiers_and_fails_when_divergent() {
        for tier in declared_tiers() {
            let c = declared_check(&tier.name, "articulated_energy_drift");
            assert!(c.pass, "{} measured {}", tier.name, c.measured);
        }
        let divergent = articulated_energy_drift(Duration::from_millis(200)).unwrap();
        assert!(divergent.value.is_nan() || divergent.value > ARTICULATED_ENERGY_TOLERANCE);
    }

    #[test]
    fn articulated_energy_drift_fails_a_finite_accuracy_loss_not_only_divergence() {
        let c = articulated_energy_drift(Duration::from_millis(5)).unwrap();
        assert!(c.value.is_finite(), "measured {}", c.value);
        assert!(
            c.value > ARTICULATED_ENERGY_TOLERANCE,
            "5 ms measured {}",
            c.value
        );
    }

    #[test]
    fn declared_tiers_fit_the_battery_step_budget() {
        let required = battery_steps(&declared_tiers()).unwrap();
        assert_eq!(required, 539_000);
        assert_eq!(required + u128::from(G1_PROBE_MAX_STEPS), 543_000);
        assert!(required <= u128::from(MAX_BATTERY_STEPS));
    }

    #[test]
    fn battery_refuses_a_pathological_tier_before_simulating() {
        // Without the guard this call would integrate about 4e10 steps.
        let nano = Tier::semi_implicit("nano", Duration::from_nanos(1));
        let started = std::time::Instant::now();
        let err = run_battery(&[nano, declared_tiers()[1].clone()]).unwrap_err();
        assert!(
            matches!(err, BatteryError::StepBudget { limit, .. } if limit == MAX_BATTERY_STEPS),
            "{err:?}"
        );
        if let BatteryError::StepBudget { required, .. } = err {
            assert!(required > 4 * 10_u128.pow(10));
        }
        // Refused during validation: no tier, not even the cheap coarse one, ran.
        assert!(started.elapsed() < std::time::Duration::from_secs(1));
    }

    #[test]
    fn explicit_step_limit_is_enforced_exactly() {
        let coarse = [declared_tiers()[1].clone()];
        let required = 49_000 + G1_PROBE_MAX_STEPS;
        assert!(matches!(
            run_battery_with_limit(&coarse, required - 1),
            Err(BatteryError::StepBudget {
                required: r,
                limit
            }) if r == u128::from(required) && limit == required - 1
        ));
        let report = run_battery_with_limit(&coarse, required).unwrap();
        assert_eq!(report.checks, cached().checks[CHECK_NAMES.len()..]);
    }

    #[test]
    fn battery_rejects_an_unsupported_solver() {
        let mut tier = Tier::semi_implicit("solver", Duration::from_micros(100));
        tier.solver = "contact-solver-v2".to_owned();
        assert!(matches!(
            run_battery(&[tier]),
            Err(BatteryError::UnsupportedSolver { .. })
        ));
    }

    #[test]
    fn an_over_coarse_tier_fails_its_checks_and_the_report_says_so() {
        let report =
            run_battery(&[Tier::semi_implicit("huge", Duration::from_millis(200))]).unwrap();
        assert_eq!(report.tier_set, TierSet::Experimental);
        let failing: Vec<&str> = report
            .checks
            .iter()
            .filter(|c| !c.check.pass)
            .map(|c| c.check.name.as_str())
            .collect();
        assert_eq!(
            failing,
            [
                "pendulum_period",
                "pendulum_trajectory",
                "energy_drift",
                "articulated_energy_drift",
                "free_fall"
            ]
        );
        assert!(!report.all_pass());
        assert!(!report.production_pass());
        let back = FidelityReport::from_toml(&report.to_toml()).unwrap();
        assert_eq!(back.tier_set, TierSet::Experimental);
        assert!(!back.all_pass());
        let mut inflated = back;
        inflated.checks[0].check.tolerance = 1e9;
        assert!(matches!(
            FidelityReport::from_toml(&inflated.to_toml()),
            Err(ReportError::Invalid(_))
        ));
    }

    #[test]
    fn contact_applicability_is_per_tier_and_never_claims_g1_qualification() {
        let contact = &declared().contact;
        assert_eq!(contact.len(), 2);
        let reference = contact.iter().find(|c| c.tier == "reference").unwrap();
        let coarse = contact.iter().find(|c| c.tier == "coarse").unwrap();
        assert_eq!(reference.status, ContactStatus::NotMeasured);
        assert_eq!(coarse.status, ContactStatus::NotApplicable);
        assert!(coarse.reason.contains("outside"), "{}", coarse.reason);
        for c in contact {
            assert_eq!(c.system, G1_CONTACT_SYSTEM);
            assert!(!c.reason.contains("qualified at") || c.reason.contains("not qualified"));
        }
        assert_eq!(report_scope_text(), SCOPE);
        assert!(CHECK_NAMES.iter().all(|name| !name.contains("contact")));
    }

    fn report_scope_text() -> String {
        honest_text()
            .lines()
            .find_map(|l| {
                l.strip_prefix("scope = \"")
                    .and_then(|r| r.strip_suffix('"'))
            })
            .unwrap()
            .to_owned()
    }

    #[test]
    fn contact_range_bounds_follow_adr_0010() {
        let at = |nanos: i64| {
            contact_applicability(&[Tier::semi_implicit("t", Duration::from_nanos(nanos))])[0]
                .status
        };
        assert_eq!(
            at(G1_CONTACT_MIN_TIMESTEP_NANOS - 1),
            ContactStatus::NotApplicable
        );
        assert_eq!(
            at(G1_CONTACT_MIN_TIMESTEP_NANOS),
            ContactStatus::NotMeasured
        );
        assert_eq!(
            at(G1_CONTACT_MAX_TIMESTEP_NANOS),
            ContactStatus::NotMeasured
        );
        assert_eq!(
            at(G1_CONTACT_MAX_TIMESTEP_NANOS + 1),
            ContactStatus::NotApplicable
        );
    }

    #[test]
    fn reader_rejects_contact_rows_that_disagree_with_the_tier_timesteps() {
        let mut report = declared();
        report.contact[1].status = ContactStatus::NotMeasured;
        assert_rejected(&report.to_toml());
        let mut dropped = declared();
        dropped.contact.clear();
        assert_rejected(&dropped.to_toml());
        let text = honest_text().replace("status = \"not_applicable\"", "status = \"qualified\"");
        assert_rejected(&text);
    }

    #[test]
    fn reader_rejects_a_wrong_scope() {
        let text = honest_text().replace(SCOPE, "g1-floating-base");
        assert_rejected(&text);
    }

    #[test]
    fn battery_rejects_invalid_tier_declarations() {
        assert_eq!(run_battery(&[]).unwrap_err(), BatteryError::NoTiers);
        let zero = Tier::semi_implicit("zero", Duration::ZERO);
        assert_eq!(
            run_battery(&[zero]).unwrap_err(),
            BatteryError::NonPositiveTimestep("zero".to_owned())
        );
        let mut rk = Tier::semi_implicit("rk", Duration::from_micros(100));
        rk.integrator = "rk4".to_owned();
        assert!(matches!(
            run_battery(&[rk]),
            Err(BatteryError::UnsupportedIntegrator { .. })
        ));
        let dup = [declared_tiers()[0].clone(), declared_tiers()[0].clone()];
        assert!(matches!(
            run_battery(&dup),
            Err(BatteryError::DuplicateTier(_))
        ));
    }
}

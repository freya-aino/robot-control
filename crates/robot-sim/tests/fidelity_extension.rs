//! Public API tests for third-party fidelity check implementations.

use std::cell::Cell;

use robot_sim::fidelity::{
    run_checks, BatteryError, FidelityCheck, Measurement, Tier, MAX_BATTERY_STEPS,
};
use robot_sim::oracle::Tolerance;
use robot_types::time::Duration;

#[derive(Clone, Copy)]
enum Outcome {
    Value(f64),
    Inapplicable,
    Error,
}

struct PlugInCheck {
    name: &'static str,
    outcome: Outcome,
    series: Option<Tolerance>,
    horizon_nanos: u64,
    measured: Cell<bool>,
}

impl FidelityCheck for PlugInCheck {
    fn name(&self) -> &'static str {
        self.name
    }

    fn system(&self) -> &'static str {
        "external_test_system"
    }

    fn tolerance(&self) -> f64 {
        0.5
    }

    fn series_tolerance(&self, _tier: &Tier) -> Option<Tolerance> {
        self.series
    }

    fn measure(&self, _tier: &Tier) -> Result<Option<Measurement>, BatteryError> {
        self.measured.set(true);
        match self.outcome {
            Outcome::Value(value) => Ok(Some(Measurement::new(value, 0))),
            Outcome::Inapplicable => Ok(None),
            Outcome::Error => Err(BatteryError::CheckFailed {
                name: self.name.to_owned(),
                message: "injected measurement failure".to_owned(),
            }),
        }
    }

    fn steps(&self, tier: &Tier) -> u64 {
        self.horizon_nanos.div_ceil(tier.timestep.nanos() as u64)
    }
}

fn tier(dt: Duration) -> Tier {
    Tier::semi_implicit("plugin-test", dt)
}

#[test]
fn external_plugin_measurement_preserves_its_metadata_and_pass_rule() {
    let check = PlugInCheck {
        name: "external_measurement",
        outcome: Outcome::Value(0.75),
        series: None,
        horizon_nanos: 12,
        measured: Cell::new(false),
    };
    let checks: [&dyn FidelityCheck; 1] = [&check];
    let result = run_checks(&checks, &tier(Duration::from_millis(1))).unwrap();

    assert!(check.measured.get());
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].name, "external_measurement");
    assert_eq!(result[0].system, "external_test_system");
    assert_eq!(result[0].tolerance, 0.5);
    assert_eq!(result[0].measured, 0.75);
    assert!(!result[0].pass);
}

#[test]
fn standalone_runner_skips_an_inapplicable_plugin_result() {
    let check = PlugInCheck {
        name: "external_optional",
        outcome: Outcome::Inapplicable,
        series: None,
        horizon_nanos: 0,
        measured: Cell::new(false),
    };
    let checks: [&dyn FidelityCheck; 1] = [&check];
    assert!(run_checks(&checks, &tier(Duration::from_millis(1)))
        .unwrap()
        .is_empty());
    assert!(check.measured.get());
}

#[test]
fn standalone_runner_propagates_plugin_measurement_errors() {
    let check = PlugInCheck {
        name: "external_error",
        outcome: Outcome::Error,
        series: None,
        horizon_nanos: 0,
        measured: Cell::new(false),
    };
    let checks: [&dyn FidelityCheck; 1] = [&check];
    assert_eq!(
        run_checks(&checks, &tier(Duration::from_millis(1))),
        Err(BatteryError::CheckFailed {
            name: "external_error".to_owned(),
            message: "injected measurement failure".to_owned(),
        })
    );
    assert!(check.measured.get());
}

#[test]
fn standalone_runner_rejects_duplicate_plugin_names_before_measuring() {
    let first = PlugInCheck {
        name: "external_twin",
        outcome: Outcome::Value(0.0),
        series: None,
        horizon_nanos: 0,
        measured: Cell::new(false),
    };
    let second = PlugInCheck {
        name: "external_twin",
        outcome: Outcome::Value(0.0),
        series: None,
        horizon_nanos: 0,
        measured: Cell::new(false),
    };
    let checks: [&dyn FidelityCheck; 2] = [&first, &second];
    assert_eq!(
        run_checks(&checks, &tier(Duration::from_millis(1))),
        Err(BatteryError::DuplicateCheck("external_twin".to_owned()))
    );
    assert!(!first.measured.get());
    assert!(!second.measured.get());
}

#[test]
fn custom_step_budget_uses_plugin_steps_and_refuses_one_nanosecond_tiers_first() {
    const HORIZON_NANOS: u64 = 45_000_000_000;
    let check = PlugInCheck {
        name: "expensive_external",
        outcome: Outcome::Value(0.0),
        series: None,
        horizon_nanos: HORIZON_NANOS,
        measured: Cell::new(false),
    };
    let checks: [&dyn FidelityCheck; 1] = [&check];
    let result = run_checks(&checks, &tier(Duration::from_nanos(1)));

    assert_eq!(
        result,
        Err(BatteryError::StepBudget {
            required: u128::from(HORIZON_NANOS),
            limit: MAX_BATTERY_STEPS,
        })
    );
    assert!(!check.measured.get());
}

#[test]
fn standalone_runner_rejects_an_invalid_tier_before_measuring() {
    let check = PlugInCheck {
        name: "external_invalid_tier",
        outcome: Outcome::Value(0.0),
        series: None,
        horizon_nanos: 0,
        measured: Cell::new(false),
    };
    let checks: [&dyn FidelityCheck; 1] = [&check];
    assert_eq!(
        run_checks(&checks, &tier(Duration::ZERO)),
        Err(BatteryError::NonPositiveTimestep("plugin-test".to_owned()))
    );
    assert!(!check.measured.get());
}

#[test]
fn standalone_runner_rejects_plugin_series_relative_tolerance() {
    let check = PlugInCheck {
        name: "external_relative_series",
        outcome: Outcome::Value(0.25),
        series: Some(Tolerance { abs: 0.5, rel: 0.1 }),
        horizon_nanos: 0,
        measured: Cell::new(false),
    };
    let checks: [&dyn FidelityCheck; 1] = [&check];

    assert_eq!(
        run_checks(&checks, &tier(Duration::from_millis(1))),
        Err(BatteryError::CheckFailed {
            name: "external_relative_series".to_owned(),
            message: "v6 maximum-error rows require finite non-negative absolute and zero relative tolerance"
                .to_owned(),
        })
    );
    assert!(check.measured.get());
}

#[test]
fn standalone_runner_rejects_plugin_series_nan_absolute_tolerance() {
    let check = PlugInCheck {
        name: "external_nan_series",
        outcome: Outcome::Value(0.25),
        series: Some(Tolerance {
            abs: f64::NAN,
            rel: 0.0,
        }),
        horizon_nanos: 0,
        measured: Cell::new(false),
    };
    let checks: [&dyn FidelityCheck; 1] = [&check];

    assert_eq!(
        run_checks(&checks, &tier(Duration::from_millis(1))),
        Err(BatteryError::CheckFailed {
            name: "external_nan_series".to_owned(),
            message: "v6 maximum-error rows require finite non-negative absolute and zero relative tolerance"
                .to_owned(),
        })
    );
    assert!(check.measured.get());
}

#[test]
fn standalone_runner_accepts_plugin_series_with_zero_relative_tolerance() {
    let tolerance = Tolerance { abs: 0.5, rel: 0.0 };
    let check = PlugInCheck {
        name: "external_absolute_series",
        outcome: Outcome::Value(0.25),
        series: Some(tolerance),
        horizon_nanos: 0,
        measured: Cell::new(false),
    };
    let checks: [&dyn FidelityCheck; 1] = [&check];

    let result = run_checks(&checks, &tier(Duration::from_millis(1))).unwrap();
    assert!(check.measured.get());
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].series_tolerance, Some(tolerance));
    assert_eq!(result[0].tolerance, tolerance.abs);
    assert_eq!(result[0].measured, 0.25);
    assert!(result[0].pass);
}

//! Checks the recorded CPU-vs-GPU agreement tolerances in `tolerances.toml`.
//!
//! Runs with the feature off. Every scenario must be complete. Absolute
//! records (no `tolerance_rel`) must have every measured maximum within
//! `tolerance_abs`. Combined records must have every measured worst ratio
//! within 1, since their bound is `tolerance_abs + tolerance_rel * |cpu|`.
//! The GPU tests read their tolerances from the same file through `common`.

mod common;

use common::{number, parse_scenarios, RECORDS};

const REQUIRED_KEYS: [&str; 12] = [
    "name",
    "kernel",
    "tier",
    "envs",
    "steps",
    "dt",
    "tolerance_abs",
    "measured_max_intel",
    "measured_max_lavapipe",
    "measured_index_intel",
    "measured_index_lavapipe",
    "test",
];

#[test]
fn tolerance_records_are_complete_and_hold() {
    let scenarios = parse_scenarios(RECORDS);
    assert!(
        !scenarios.is_empty(),
        "tolerances.toml has no [[scenario]] blocks"
    );

    for scenario in &scenarios {
        for key in REQUIRED_KEYS {
            assert!(
                scenario.contains_key(key),
                "scenario {:?} is missing `{key}`",
                scenario.get("name")
            );
        }
        let name = &scenario["name"];
        let tolerance = number(scenario, "tolerance_abs");
        assert!(tolerance > 0.0, "{name}: tolerance must be positive");
        if scenario.contains_key("tolerance_rel") {
            assert!(
                number(scenario, "tolerance_rel") >= 0.0,
                "{name}: tolerance_rel must be non-negative"
            );
            for device in [
                "measured_worst_ratio_intel",
                "measured_worst_ratio_lavapipe",
            ] {
                let ratio = number(scenario, device);
                assert!(
                    ratio.is_finite() && ratio <= 1.0,
                    "{name}: {device} = {ratio} exceeds the combined bound"
                );
            }
        } else {
            for device in ["measured_max_intel", "measured_max_lavapipe"] {
                let measured = number(scenario, device);
                assert!(
                    measured.is_finite() && measured <= tolerance,
                    "{name}: {device} = {measured} exceeds tolerance_abs = {tolerance}"
                );
            }
        }
    }
}

#[test]
fn tolerance_parser_rejects_a_key_outside_a_scenario() {
    let result = std::panic::catch_unwind(|| parse_scenarios("name = \"orphan\"\n"));
    assert!(result.is_err());
}

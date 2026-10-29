//! Shared reader for `tolerances.toml`. The file is a flat list of `key = value`
//! pairs under `[[scenario]]` headers, parsed with `std` only so that the
//! feature-off build and the GPU tests read the same records.

#![allow(dead_code)]

use std::collections::BTreeMap;

pub const RECORDS: &str = include_str!("../../tolerances.toml");

pub type Record = BTreeMap<String, String>;

pub fn parse_scenarios(text: &str) -> Vec<Record> {
    let mut scenarios: Vec<Record> = Vec::new();
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line == "[[scenario]]" {
            scenarios.push(BTreeMap::new());
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .unwrap_or_else(|| panic!("malformed line: {line}"));
        let scenario = scenarios
            .last_mut()
            .unwrap_or_else(|| panic!("key outside a [[scenario]] block: {line}"));
        scenario.insert(
            key.trim().to_string(),
            value.trim().trim_matches('"').to_string(),
        );
    }
    scenarios
}

pub fn record(name: &str) -> Record {
    parse_scenarios(RECORDS)
        .into_iter()
        .find(|r| r.get("name").map(String::as_str) == Some(name))
        .unwrap_or_else(|| panic!("no tolerance record named {name}"))
}

pub fn number(record: &Record, key: &str) -> f64 {
    record[key]
        .parse::<f64>()
        .unwrap_or_else(|_| panic!("{key} is not a number: {}", record[key]))
}

pub fn tolerance_abs(name: &str) -> f32 {
    number(&record(name), "tolerance_abs") as f32
}

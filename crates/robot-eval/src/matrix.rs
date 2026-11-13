//! The acceptance matrix of SPECS.md Section 14, and its closure rule:
//! "A row is closed by an artifact in the evidence ledger, not by an
//! assertion in a report."
//!
//! [`ROWS`] is the matrix's rows (ID, area, requirement source), kept in
//! step with SPECS.md by a test that reads the specification. [`status`]
//! reads `docs/ledger/evidence-ledger.md` and closes a row only where the
//! ledger's "Acceptance rows" table names an artifact for it; anything
//! else, including a row the ledger mentions without an artifact, stays
//! open. A report built from this cannot claim a row the ledger does not
//! support.

/// `(id, area, requirement source)` for every row.
pub const ROWS: [(&str, &str, &str); 50] = [
    ("A-01", "Rust constraint", "§2, §1.1"),
    ("A-02", "Dependency promotion", "B.6"),
    ("A-03", "Evidence discipline", "§3"),
    ("A-04", "Parser robustness", "§7.1, §8.1, B.2.11"),
    ("A-05", "Optimization", "§8.3, B.2.8"),
    ("A-06", "Estimation", "§11.2, B.2.9"),
    ("A-07", "Telemetry", "§5, B.2.10"),
    ("A-08", "Encoding and parallelism", "§7.5, B.2.12"),
    (
        "A-09",
        "Cross-engine comparison, if authorized",
        "§2.1, §9.1",
    ),
    ("A-10", "Data correctness", "§7.2, §8.2"),
    ("A-11", "Retargeting", "§8.3"),
    ("A-12", "Simulator", "§9.1"),
    ("A-13", "Learning", "§9.2"),
    ("A-14", "Tracker", "§9.2, §9.4"),
    ("A-15", "Runtime", "§5, §10.3"),
    ("A-16", "Retrieval", "§10.5"),
    ("A-17", "Adaptation", "§11.3"),
    ("A-18", "Operating modes", "§4"),
    ("A-19", "Safety case", "§12"),
    ("A-20", "Hardware", "§4, §12"),
    ("A-21", "Boxing", "§9.4, §12.2"),
    ("A-22", "Environment reproducibility", "§2.2, §13"),
    (
        "A-23",
        "Agent-layer qualification, if adopted",
        "§2.3, §6, §12.6, B.6",
    ),
    ("A-24", "Offline degradation", "§2.3, §12.6"),
    (
        "A-25",
        "Development-time automation hygiene",
        "§2.3.1, §2.2",
    ),
    ("A-26", "Assisted sampling discipline", "§9.3, §9.5"),
    ("A-27", "Dependency knowledge base", "§13.1, B.6"),
    ("A-28", "RL extension conformance", "§9.3, §9.6"),
    ("A-29", "Experience and replay storage", "§7.3, §7.5, §7.6"),
    ("A-30", "Extension and removal isolation", "§6, §9.3, §9.6"),
    ("A-31", "Robot profile extensibility", "§7.1, §7.5"),
    ("A-32", "Hierarchical latency tiers", "§5.1, §7.4"),
    ("A-33", "Staged latent pipeline", "§9.4, §5.1"),
    ("A-34", "Retrieval-augmented learning", "§9.7, §7.6"),
    ("A-35", "Retrieval tier discipline", "§10.6, §5"),
    (
        "A-36",
        "Simulation fidelity tiers",
        "§9.1, `SIMULATION_SPECS.md` §12.2 and §14",
    ),
    (
        "A-37",
        "Checkpoint and partial fine-tuning",
        "§9.2, §9.3, §7.5",
    ),
    ("A-38", "Plug-in seam traits", "§6, §9.3, §9.6"),
    ("A-39", "Test-purpose frame output", "§5, §14"),
    ("A-40", "USD model import", "§7.1"),
    ("A-41", "USD replay validity", "§7.3"),
    ("A-42", "Contact coverage", "§9.1"),
    ("A-43", "Hard joint limits", "§9.1"),
    ("A-44", "Primitive contact", "§9.1"),
    ("A-45", "Convex-hull contact", "§9.1"),
    ("A-46", "Self-collision", "§9.1"),
    ("A-47", "Discovery candidate profile", "§7.7"),
    ("A-48", "Probe refusal", "§7.7 item 3"),
    ("A-49", "Web claims are proposals", "§7.7 items 6 and 7"),
    ("A-50", "Agent cannot activate", "§7.7 item 8, §2.3.1"),
];

/// A row's state.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum RowState {
    /// No admitting artifact recorded.
    Open,
    /// Closed by this artifact on this date.
    Closed {
        /// The artifact reference.
        artifact: String,
        /// The date recorded.
        date: String,
    },
}

/// Every row's state according to `ledger` (the evidence ledger's text).
///
/// The ledger's closure table is any Markdown table row of the form
/// `| A-NN | <artifact> | <date> | ... |` under a heading containing
/// "Acceptance rows". An artifact of `—`, `-` or empty does not close.
#[must_use]
pub fn status(ledger: &str) -> Vec<(&'static str, RowState)> {
    let mut in_section = false;
    let mut closed: Vec<(String, String, String)> = Vec::new();
    for line in ledger.lines() {
        if line.starts_with('#') {
            in_section = line.contains("Acceptance rows");
            continue;
        }
        if !in_section || !line.trim_start().starts_with('|') {
            continue;
        }
        let cells: Vec<&str> = line.split('|').map(str::trim).collect();
        if cells.len() < 4 {
            continue;
        }
        let (id, artifact, date) = (cells[1], cells[2], cells[3]);
        if ROWS.iter().any(|r| r.0 == id) && !matches!(artifact, "" | "-" | "—") {
            closed.push((id.to_owned(), artifact.to_owned(), date.to_owned()));
        }
    }
    ROWS.iter()
        .map(|(id, _, _)| {
            let state = closed.iter().rev().find(|(c, _, _)| c == id).map_or(
                RowState::Open,
                |(_, a, d)| RowState::Closed {
                    artifact: a.clone(),
                    date: d.clone(),
                },
            );
            (*id, state)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rows_match_the_specification() {
        let spec = include_str!("../../../SPECS.md");
        let ids: Vec<&str> = spec
            .lines()
            .filter_map(|l| l.strip_prefix("| A-"))
            .filter_map(|l| l.split('|').next())
            .map(str::trim)
            .collect();
        assert_eq!(ids.len(), ROWS.len());
        for ((id, _, _), n) in ROWS.iter().zip(&ids) {
            assert_eq!(id[2..], **n);
        }
    }

    #[test]
    fn only_a_ledger_artifact_closes_a_row() {
        let ledger = "# Evidence ledger

| A-01 | mentioned in passing | x |

## Acceptance rows

| Row | Artifact | Date |
| --- | --- | --- |
| A-01 | runs/2026-11-13/exclusion-gate.txt | 2026-11-13 |
| A-02 | — | 2026-11-13 |
";
        let s = status(ledger);
        assert!(
            matches!(&s[0].1, RowState::Closed { artifact, .. } if artifact.starts_with("runs/"))
        );
        assert_eq!(s[1].1, RowState::Open);
        assert!(s[2..].iter().all(|(_, st)| *st == RowState::Open));
    }

    #[test]
    fn the_real_ledger_parses() {
        let ledger = include_str!("../../../docs/ledger/evidence-ledger.md");
        assert_eq!(status(ledger).len(), 50);
    }
}

//! Evidence-ledger diffing, the ledger tooling SPECS.md Section 13.1 names
//! and Section 3 governs: "Do not promote an evidence state without a
//! recorded source".
//!
//! [`rows`] reads every Markdown table of a ledger that has a `State`
//! column, keyed by its first cell. [`diff`] compares two versions of the
//! ledger: rows added, rows removed, and rows whose state changed, each
//! change classed as a promotion or a demotion by the order of Section 3's
//! states. A promotion whose admitting-artifact cell is empty or `—` is an
//! [`Unsupported`] promotion, which is exactly the change the rule forbids,
//! and the `devtools ledger-diff` command fails on one.

use robot_types::evidence::EvidenceState;

/// One ledger row.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Row {
    /// The first cell (the lead or claim key).
    pub key: String,
    /// The `State` cell, backquotes removed.
    pub state: String,
    /// The admitting-artifact cell, if the table has one.
    pub artifact: Option<String>,
}

/// Rows of every table with a `State` column.
#[must_use]
pub fn rows(ledger: &str) -> Vec<Row> {
    let mut out = Vec::new();
    let mut header: Option<Vec<String>> = None;
    for line in ledger.lines() {
        let t = line.trim();
        if !t.starts_with('|') {
            header = None;
            continue;
        }
        let cells: Vec<String> = t
            .trim_matches('|')
            .split('|')
            .map(|c| c.trim().to_owned())
            .collect();
        if cells
            .iter()
            .all(|c| c.chars().all(|ch| ch == '-' || ch == ':' || ch == ' '))
        {
            continue;
        }
        match &header {
            None => header = Some(cells),
            Some(h) => {
                let col = |name: &str| h.iter().position(|c| c.eq_ignore_ascii_case(name));
                let Some(si) = col("State") else { continue };
                let state = cells
                    .get(si)
                    .map(|c| c.replace('`', ""))
                    .unwrap_or_default();
                let artifact = col("Admitting artifact").and_then(|i| cells.get(i).cloned());
                out.push(Row {
                    key: cells.first().cloned().unwrap_or_default(),
                    state,
                    artifact,
                });
            }
        }
    }
    out
}

/// A state change.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Change {
    /// Row key.
    pub key: String,
    /// State before.
    pub from: String,
    /// State after.
    pub to: String,
    /// Whether the new state ranks above the old.
    pub promotion: bool,
}

/// A promotion with no admitting artifact.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Unsupported {
    /// Row key.
    pub key: String,
    /// The new state.
    pub to: String,
}

/// The difference between two ledger versions.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct LedgerDiff {
    /// Keys only in the new version.
    pub added: Vec<String>,
    /// Keys only in the old version.
    pub removed: Vec<String>,
    /// State changes.
    pub changed: Vec<Change>,
    /// Promotions without an admitting artifact.
    pub unsupported: Vec<Unsupported>,
}

fn rank(state: &str) -> Option<usize> {
    let s = EvidenceState::parse(state.split_whitespace().next().unwrap_or(""))?;
    EvidenceState::ALL.iter().position(|x| *x == s)
}

/// Compares `old` with `new`.
#[must_use]
pub fn diff(old: &str, new: &str) -> LedgerDiff {
    let (a, b) = (rows(old), rows(new));
    let mut d = LedgerDiff::default();
    for r in &b {
        match a.iter().find(|x| x.key == r.key) {
            None => d.added.push(r.key.clone()),
            Some(o) if o.state != r.state => {
                let promotion =
                    matches!((rank(&o.state), rank(&r.state)), (Some(x), Some(y)) if y > x);
                if promotion
                    && r.artifact
                        .as_deref()
                        .is_none_or(|a| matches!(a.trim(), "" | "—" | "-"))
                {
                    d.unsupported.push(Unsupported {
                        key: r.key.clone(),
                        to: r.state.clone(),
                    });
                }
                d.changed.push(Change {
                    key: r.key.clone(),
                    from: o.state.clone(),
                    to: r.state.clone(),
                    promotion,
                });
            }
            Some(_) => {}
        }
    }
    d.removed = a
        .iter()
        .filter(|o| !b.iter().any(|r| r.key == o.key))
        .map(|o| o.key.clone())
        .collect();
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    const OLD: &str = "## Leads

| Lead | Claim | State | Admitting artifact |
| --- | --- | --- | --- |
| RMA | two-phase latent | `unverified` | — |
| ASAP | mismatch learning | `unverified` | — |
";

    #[test]
    fn rows_are_read_from_tables_with_a_state_column() {
        let r = rows(OLD);
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].state, "unverified");
        assert_eq!(r[0].artifact.as_deref(), Some("—"));
    }

    #[test]
    fn a_promotion_without_an_artifact_is_unsupported() {
        let new = OLD
            .replace("| RMA | two-phase latent | `unverified` | — |", "| RMA | two-phase latent | `paper_verified` | — |")
            .replace("| ASAP | mismatch learning | `unverified` | — |", "| ASAP | mismatch learning | `paper_verified` | arXiv:2502.01143 read 2026-11-13 |")
            + "| NEW | x | `unverified` | — |\n";
        let d = diff(OLD, &new);
        assert_eq!(d.added, vec!["NEW".to_owned()]);
        assert_eq!(d.changed.len(), 2);
        assert!(d.changed.iter().all(|c| c.promotion));
        assert_eq!(
            d.unsupported,
            vec![Unsupported {
                key: "RMA".into(),
                to: "paper_verified".into()
            }]
        );
    }

    #[test]
    fn a_demotion_needs_no_artifact_and_removals_are_listed() {
        let new = "| Lead | Claim | State | Admitting artifact |\n| --- | --- | --- | --- |\n| RMA | x | `unverified` | — |\n";
        let promoted = OLD.replace(
            "`unverified` | — |\n| ASAP",
            "`code_inspected` | review | |\n| ASAP",
        );
        let d = diff(&promoted, new);
        assert_eq!(d.removed, vec!["ASAP".to_owned()]);
        assert!(d.unsupported.is_empty());
        assert!(!d.changed[0].promotion);
    }
}

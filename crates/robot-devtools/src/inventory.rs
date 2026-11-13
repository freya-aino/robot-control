//! The dependency inventory, knowledge-base records and the staleness
//! sweep of SPECS.md Section 13.1.
//!
//! - **Inventory as primary key** (item 1): [`lock_inventory`] reads
//!   `Cargo.lock`; [`Store::orphans`] lists crate entries that name a
//!   package the lockfile does not contain, so "an entry cannot exist for
//!   something the dependency audit does not know about" is a check.
//! - **Invalidated by version change** (item 4): [`Store::sweep`] marks
//!   every crate entry whose version differs from the lockfile's stale, and
//!   then every entry that depends on a stale one, transitively. A stale
//!   entry stays in the file, marked, until a person refreshes it.
//! - **Readable without a model service** (item 5): the store is the
//!   `kb-store/1` TOML file under version control, and [`Store::search`]
//!   is a local word match.
//! - **Fault records** (item 6): [`FaultRecord`] names symptom,
//!   identifying measurement, resolution, the dependency version it applies
//!   to, and the episode and incident records it links.
//! - **Never evidence** (item 3): [`ledger_cites_kb`] finds any line of an
//!   evidence ledger that cites the knowledge base, which the gate refuses.

use robot_artifacts::toml::{self, Table, Value};
use robot_types::evidence::EvidenceState;

use crate::knowledge_base::{Citation, KnowledgeBaseEntry};

/// One package of the lockfile.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Package {
    /// Name.
    pub name: String,
    /// Version.
    pub version: String,
    /// Whether it comes from a registry or git (not a workspace path).
    pub external: bool,
}

/// Every package of a `Cargo.lock`.
#[must_use]
pub fn lock_inventory(lock: &str) -> Vec<Package> {
    let mut out = Vec::new();
    let mut current: Option<Package> = None;
    for line in lock.lines().map(str::trim) {
        if line == "[[package]]" {
            out.extend(current.take());
            current = Some(Package {
                name: String::new(),
                version: String::new(),
                external: false,
            });
            continue;
        }
        let Some(p) = current.as_mut() else { continue };
        let value = |l: &str, k: &str| {
            l.strip_prefix(k)
                .map(str::trim_start)
                .and_then(|r| r.strip_prefix('='))
                .map(|r| r.trim().trim_matches('"').to_owned())
        };
        if let Some(v) = value(line, "name") {
            p.name = v;
        } else if let Some(v) = value(line, "version") {
            p.version = v;
        } else if value(line, "source").is_some() {
            p.external = true;
        }
    }
    out.extend(current);
    out
}

/// What an entry describes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// A Cargo package (keyed by the lockfile).
    Crate,
    /// Firmware.
    Firmware,
    /// A driver.
    Driver,
    /// OS or kernel option.
    Os,
    /// A bus or transport parameter.
    Transport,
    /// A robot profile.
    Profile,
}

impl Kind {
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "crate" => Self::Crate,
            "firmware" => Self::Firmware,
            "driver" => Self::Driver,
            "os" => Self::Os,
            "transport" => Self::Transport,
            "profile" => Self::Profile,
            _ => return None,
        })
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Crate => "crate",
            Self::Firmware => "firmware",
            Self::Driver => "driver",
            Self::Os => "os",
            Self::Transport => "transport",
            Self::Profile => "profile",
        }
    }
}

/// A stored entry: the knowledge-base entry plus its kind and dependencies.
#[derive(Clone, Debug)]
pub struct StoredEntry {
    /// What it describes.
    pub kind: Kind,
    /// The entry.
    pub entry: KnowledgeBaseEntry,
    /// Subjects this entry's claim depends on.
    pub depends_on: Vec<String>,
}

/// A diagnosed hardware fault (Section 13.1 item 6).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FaultRecord {
    /// What was observed.
    pub symptom: String,
    /// The measurement that identified the layer.
    pub measurement: String,
    /// What fixed it.
    pub resolution: String,
    /// `subject@version` it applies to.
    pub applies_to: String,
    /// Episode record reference.
    pub episode: String,
    /// Incident channel reference.
    pub incident: String,
}

/// The `kb-store/1` file.
#[derive(Clone, Debug, Default)]
pub struct Store {
    /// Entries.
    pub entries: Vec<StoredEntry>,
    /// Fault records.
    pub faults: Vec<FaultRecord>,
}

/// A store parse failure.
#[derive(Clone, PartialEq, Debug)]
pub struct StoreError(pub String);

impl core::fmt::Display for StoreError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "knowledge base: {}", self.0)
    }
}

impl core::error::Error for StoreError {}

fn tables<'a>(t: &'a Table, key: &str) -> Result<Vec<&'a Table>, StoreError> {
    match t.get(key) {
        None => Ok(Vec::new()),
        Some(Value::Array(a)) => a
            .iter()
            .map(|v| match v {
                Value::Table(t) => Ok(t),
                _ => Err(StoreError(format!("[[{key}]] must hold tables"))),
            })
            .collect(),
        Some(_) => Err(StoreError(format!("{key} must be an array of tables"))),
    }
}

impl Store {
    /// Parses a `kb-store/1` file.
    pub fn parse(text: &str) -> Result<Self, StoreError> {
        let t = toml::parse(text).map_err(|e| StoreError(e.to_string()))?;
        let e = |m: &str| StoreError(m.to_owned());
        if t.require_str("schema")
            .map_err(|x| StoreError(x.to_string()))?
            != "kb-store/1"
        {
            return Err(e("schema must be kb-store/1"));
        }
        let s = |t: &Table, k: &str| {
            t.require_str(k)
                .map(str::to_owned)
                .map_err(|x| StoreError(x.to_string()))
        };
        let mut store = Self::default();
        for en in tables(&t, "entry")? {
            let citation = match en.get("citation_source") {
                None => None,
                Some(_) => Some(Citation {
                    source: s(en, "citation_source")?,
                    source_version: s(en, "citation_version")?,
                    date: s(en, "citation_date")?,
                    added_by: s(en, "added_by")?,
                }),
            };
            store.entries.push(StoredEntry {
                kind: Kind::parse(&s(en, "kind")?).ok_or_else(|| e("unknown kind"))?,
                entry: KnowledgeBaseEntry {
                    subject: s(en, "subject")?,
                    version: s(en, "version")?,
                    claim: s(en, "claim")?,
                    citation,
                    evidence: EvidenceState::parse(&s(en, "evidence")?)
                        .ok_or_else(|| e("unknown evidence state"))?,
                    stale: matches!(en.get("stale"), Some(Value::Boolean(true))),
                },
                depends_on: match en.get("depends_on") {
                    None => Vec::new(),
                    Some(Value::Array(a)) => a
                        .iter()
                        .filter_map(|v| v.as_str().map(str::to_owned))
                        .collect(),
                    Some(_) => return Err(e("depends_on must be an array")),
                },
            });
        }
        for f in tables(&t, "fault")? {
            store.faults.push(FaultRecord {
                symptom: s(f, "symptom")?,
                measurement: s(f, "measurement")?,
                resolution: s(f, "resolution")?,
                applies_to: s(f, "applies_to")?,
                episode: s(f, "episode")?,
                incident: s(f, "incident")?,
            });
        }
        Ok(store)
    }

    /// The store as `kb-store/1` text.
    #[must_use]
    pub fn to_toml(&self) -> String {
        let mut t = Table::new();
        t.insert("schema", Value::String("kb-store/1".into()));
        let str_v = |s: &str| Value::String(s.to_owned());
        t.insert(
            "entry",
            Value::Array(
                self.entries
                    .iter()
                    .map(|e| {
                        let mut r = Table::new();
                        r.insert("kind", str_v(e.kind.name()));
                        r.insert("subject", str_v(&e.entry.subject));
                        r.insert("version", str_v(&e.entry.version));
                        r.insert("claim", str_v(&e.entry.claim));
                        r.insert("evidence", str_v(e.entry.evidence.id()));
                        r.insert("stale", Value::Boolean(e.entry.stale));
                        if let Some(c) = &e.entry.citation {
                            r.insert("citation_source", str_v(&c.source));
                            r.insert("citation_version", str_v(&c.source_version));
                            r.insert("citation_date", str_v(&c.date));
                            r.insert("added_by", str_v(&c.added_by));
                        }
                        r.insert(
                            "depends_on",
                            Value::Array(e.depends_on.iter().map(|d| str_v(d)).collect()),
                        );
                        Value::Table(r)
                    })
                    .collect(),
            ),
        );
        t.insert(
            "fault",
            Value::Array(
                self.faults
                    .iter()
                    .map(|f| {
                        let mut r = Table::new();
                        r.insert("symptom", str_v(&f.symptom));
                        r.insert("measurement", str_v(&f.measurement));
                        r.insert("resolution", str_v(&f.resolution));
                        r.insert("applies_to", str_v(&f.applies_to));
                        r.insert("episode", str_v(&f.episode));
                        r.insert("incident", str_v(&f.incident));
                        Value::Table(r)
                    })
                    .collect(),
            ),
        );
        toml::to_string(&t)
    }

    /// Crate entries naming a package the inventory lacks.
    #[must_use]
    pub fn orphans(&self, inventory: &[Package]) -> Vec<&str> {
        self.entries
            .iter()
            .filter(|e| {
                e.kind == Kind::Crate && !inventory.iter().any(|p| p.name == e.entry.subject)
            })
            .map(|e| e.entry.subject.as_str())
            .collect()
    }

    /// Marks crate entries whose version changed in `inventory` stale (and
    /// records the new version), then every entry depending on a stale one,
    /// transitively. Returns the subjects newly marked.
    pub fn sweep(&mut self, inventory: &[Package]) -> Vec<String> {
        let mut marked = Vec::new();
        for e in &mut self.entries {
            if e.kind != Kind::Crate {
                continue;
            }
            if let Some(p) = inventory.iter().find(|p| p.name == e.entry.subject) {
                if p.version != e.entry.version && !e.entry.stale {
                    e.entry.update_version(p.version.clone());
                    marked.push(e.entry.subject.clone());
                }
            }
        }
        loop {
            let stale: Vec<String> = self
                .entries
                .iter()
                .filter(|e| e.entry.stale)
                .map(|e| e.entry.subject.clone())
                .collect();
            let mut changed = false;
            for e in &mut self.entries {
                if !e.entry.stale && e.depends_on.iter().any(|d| stale.contains(d)) {
                    e.entry.stale = true;
                    marked.push(e.entry.subject.clone());
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        marked
    }

    /// Entries whose subject or claim contains every word of `query`
    /// (case-insensitive).
    #[must_use]
    pub fn search(&self, query: &str) -> Vec<&StoredEntry> {
        let words: Vec<String> = query
            .split_whitespace()
            .map(str::to_ascii_lowercase)
            .collect();
        self.entries
            .iter()
            .filter(|e| {
                let text = format!("{} {}", e.entry.subject, e.entry.claim).to_ascii_lowercase();
                words.iter().all(|w| text.contains(w.as_str()))
            })
            .collect()
    }
}

/// Lines of an evidence ledger that cite the knowledge base (by its path
/// or `kb:` scheme). Section 13.1 item 3: the base "MUST NOT be cited as
/// evidence".
#[must_use]
pub fn ledger_cites_kb(ledger: &str) -> Vec<usize> {
    ledger
        .lines()
        .enumerate()
        .filter(|(_, l)| l.contains("docs/kb/") || l.contains("kb:"))
        .map(|(i, _)| i + 1)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOCK: &str = r#"
version = 4

[[package]]
name = "robot-types"
version = "0.1.0"

[[package]]
name = "libm"
version = "0.2.9"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "abc"
dependencies = [
 "x",
]
"#;

    const STORE: &str = r#"schema = "kb-store/1"

[[entry]]
kind = "crate"
subject = "libm"
version = "0.2.8"
claim = "sqrt is correctly rounded on x86_64"
evidence = "code_inspected"
citation_source = "https://docs.rs/libm/0.2.8"
citation_version = "0.2.8"
citation_date = "2026-11-01"
added_by = "freya"
depends_on = []

[[entry]]
kind = "firmware"
subject = "imu-fw"
version = "3.1"
claim = "gyro bias drifts under 0.01 rad/s per hour"
evidence = "unverified"
depends_on = ["libm"]

[[fault]]
symptom = "pelvis attitude drifts after 10 minutes"
measurement = "gyro z mean at rest 0.02 rad/s"
resolution = "bias calibration on boot"
applies_to = "imu-fw@3.1"
episode = "runs/2026-11-02/ep-17"
incident = "incident-0003"
"#;

    #[test]
    fn the_lockfile_inventory_is_read_with_its_origin() {
        let inv = lock_inventory(LOCK);
        assert_eq!(inv.len(), 2);
        assert!(!inv[0].external);
        assert_eq!(
            inv[1],
            Package {
                name: "libm".into(),
                version: "0.2.9".into(),
                external: true
            }
        );
    }

    #[test]
    fn a_version_bump_marks_the_entry_and_its_dependents_stale() {
        let mut s = Store::parse(STORE).unwrap();
        let marked = s.sweep(&lock_inventory(LOCK));
        assert_eq!(marked, vec!["libm".to_owned(), "imu-fw".to_owned()]);
        assert_eq!(s.entries[0].entry.version, "0.2.9");
        assert!(s.entries[0].entry.require_citation().is_err());
        // The stale marks survive a round trip through the file.
        let back = Store::parse(&s.to_toml()).unwrap();
        assert!(back.entries.iter().all(|e| e.entry.stale));
        assert_eq!(back.faults.len(), 1);
    }

    #[test]
    fn orphans_search_and_ledger_citations_are_found() {
        let s = Store::parse(&STORE.replace("subject = \"libm\"", "subject = \"not-in-lock\""))
            .unwrap();
        assert_eq!(s.orphans(&lock_inventory(LOCK)), vec!["not-in-lock"]);
        let s = Store::parse(STORE).unwrap();
        assert_eq!(s.search("gyro bias").len(), 1);
        assert_eq!(
            ledger_cites_kb("| x | y |\n| z | docs/kb/store.toml |\n"),
            vec![2]
        );
    }
}

//! Developer tooling over the knowledge base and the evidence ledger
//! (SPECS.md Sections 3 and 13.1). Developer and CI hosts only.
//!
//! ```text
//! devtools kb-sweep [--write]        mark entries stale against Cargo.lock
//! devtools kb-search <words...>      local search, no model service
//! devtools kb-publish <store-dir>    snapshot the base into a blob store,
//!                                    printing the content identity
//! devtools ledger-diff <old> <new>   state changes; fails on an
//!                                    unsupported promotion
//! ```
//!
//! Paths are relative to the workspace root: `Cargo.lock`,
//! `docs/kb/knowledge-base.toml` and `docs/ledger/evidence-ledger.md`.
//! Exit status: 0 clean, 1 a finding (stale entries without `--write`,
//! orphans, an unsupported promotion, the ledger citing the base), 2 an
//! input could not be read.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use robot_devtools::inventory::{ledger_cites_kb, lock_inventory, Store};
use robot_devtools::ledger::diff;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(path: &Path) -> Result<String, ExitCode> {
    std::fs::read_to_string(path).map_err(|e| {
        eprintln!("devtools: cannot read {}: {e}", path.display());
        ExitCode::from(2)
    })
}

fn run() -> Result<ExitCode, ExitCode> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let root = root();
    let kb_path = root.join("docs/kb/knowledge-base.toml");
    match args.first().map(String::as_str) {
        Some("kb-sweep") => {
            let inventory = lock_inventory(&read(&root.join("Cargo.lock"))?);
            let mut store = Store::parse(&read(&kb_path)?).map_err(|e| {
                eprintln!("devtools: {e}");
                ExitCode::from(2)
            })?;
            let orphans: Vec<String> = store
                .orphans(&inventory)
                .iter()
                .map(|s| (*s).to_owned())
                .collect();
            let marked = store.sweep(&inventory);
            let cites = ledger_cites_kb(&read(&root.join("docs/ledger/evidence-ledger.md"))?);
            for s in &marked {
                println!("stale: {s}");
            }
            for o in &orphans {
                println!("orphan (not in Cargo.lock): {o}");
            }
            for l in &cites {
                println!(
                    "evidence-ledger.md:{l} cites the knowledge base, which is never evidence"
                );
            }
            if args.get(1).map(String::as_str) == Some("--write") && !marked.is_empty() {
                std::fs::write(&kb_path, store.to_toml()).map_err(|e| {
                    eprintln!("devtools: cannot write {}: {e}", kb_path.display());
                    ExitCode::from(2)
                })?;
                println!("wrote {} stale marks", marked.len());
            }
            let finding = !orphans.is_empty()
                || !cites.is_empty()
                || (!marked.is_empty() && args.get(1).is_none());
            println!(
                "kb-sweep: {} entries, {} packages, {} newly stale, {} orphans",
                store.entries.len(),
                inventory.len(),
                marked.len(),
                orphans.len()
            );
            Ok(ExitCode::from(u8::from(finding)))
        }
        Some("kb-search") => {
            let store = Store::parse(&read(&kb_path)?).map_err(|e| {
                eprintln!("devtools: {e}");
                ExitCode::from(2)
            })?;
            for e in store.search(&args[1..].join(" ")) {
                let mark = if e.entry.stale { " [STALE]" } else { "" };
                println!(
                    "{}@{}{mark}: {}",
                    e.entry.subject, e.entry.version, e.entry.claim
                );
            }
            Ok(ExitCode::SUCCESS)
        }
        Some("kb-publish") if args.len() == 2 => {
            let store_root = Path::new(&args[1]);
            match robot_devtools::publish::publish_knowledge_base(store_root, &kb_path) {
                Ok(id) => {
                    println!("published {} as {id}", kb_path.display());
                    Ok(ExitCode::SUCCESS)
                }
                Err(e) => {
                    eprintln!("devtools: {e}");
                    Err(ExitCode::from(2))
                }
            }
        }
        Some("ledger-diff") if args.len() == 3 => {
            let d = diff(&read(Path::new(&args[1]))?, &read(Path::new(&args[2]))?);
            for k in &d.added {
                println!("added: {k}");
            }
            for k in &d.removed {
                println!("removed: {k}");
            }
            for c in &d.changed {
                let kind = if c.promotion { "promoted" } else { "changed" };
                println!("{kind}: {} {} -> {}", c.key, c.from, c.to);
            }
            for u in &d.unsupported {
                println!(
                    "UNSUPPORTED promotion: {} -> {} with no admitting artifact",
                    u.key, u.to
                );
            }
            Ok(ExitCode::from(u8::from(!d.unsupported.is_empty())))
        }
        _ => {
            eprintln!(
                "usage: devtools kb-sweep [--write] | kb-search <words> | \
                 kb-publish <store-dir> | ledger-diff <old> <new>"
            );
            Err(ExitCode::from(2))
        }
    }
}

fn main() -> ExitCode {
    run().unwrap_or_else(|c| c)
}

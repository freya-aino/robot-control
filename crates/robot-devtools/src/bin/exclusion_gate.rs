//! `exclusion-gate`: the executable form of SPECS.md Section 2.1 item 4 and
//! row A-25. Exit code 0 means every check passed; 1 means at least one
//! violation, each printed on its own line to stderr; 2 means the gate
//! could not gather its inputs.
//!
//! Usage: `exclusion-gate [workspace-root]`. The root defaults to the
//! workspace containing this crate. The checks themselves are in
//! `robot_devtools::exclusion_gate`; this binary only reads `Cargo.lock`,
//! runs `cargo tree --workspace -e normal --prefix none --offline` and
//! reads every `crates/*/Cargo.toml`.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use robot_devtools::exclusion_gate::{
    devtools_dependents, excluded_names_in_lock, foreign_packages_in_tree, Violation,
};

fn main() -> ExitCode {
    let root = match std::env::args().nth(1) {
        Some(arg) => PathBuf::from(arg),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."),
    };
    let root = match root.canonicalize() {
        Ok(root) => root,
        Err(error) => {
            eprintln!(
                "exclusion gate: cannot resolve workspace root {}: {error}",
                root.display()
            );
            return ExitCode::from(2);
        }
    };

    let lock = match std::fs::read_to_string(root.join("Cargo.lock")) {
        Ok(text) => text,
        Err(error) => {
            eprintln!("exclusion gate: cannot read Cargo.lock: {error}");
            return ExitCode::from(2);
        }
    };

    let tree = match Command::new("cargo")
        .args([
            "tree",
            "--workspace",
            "-e",
            "normal",
            "--prefix",
            "none",
            "--offline",
        ])
        .current_dir(&root)
        .output()
    {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).into_owned()
        }
        Ok(output) => {
            eprintln!(
                "exclusion gate: `cargo tree` failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            return ExitCode::from(2);
        }
        Err(error) => {
            eprintln!("exclusion gate: cannot run cargo: {error}");
            return ExitCode::from(2);
        }
    };

    let manifests = match read_manifests(&root.join("crates")) {
        Ok(manifests) => manifests,
        Err(error) => {
            eprintln!("exclusion gate: cannot read crate manifests: {error}");
            return ExitCode::from(2);
        }
    };

    let mut violations: Vec<Violation> = excluded_names_in_lock(&lock);
    violations.extend(foreign_packages_in_tree(&tree, &root));
    violations.extend(devtools_dependents(&manifests));

    if violations.is_empty() {
        let packages = tree.lines().filter(|l| !l.trim().is_empty()).count();
        let lock_entries = lock
            .lines()
            .filter(|l| l.trim_start().starts_with("name = "))
            .count();
        println!(
            "exclusion gate: ok ({} default-closure entries, all workspace paths; {} lockfile packages, none an excluded runtime; {} manifests, none reach robot-devtools)",
            packages,
            lock_entries,
            manifests.len()
        );
        ExitCode::SUCCESS
    } else {
        for violation in &violations {
            eprintln!("exclusion gate: {violation}");
        }
        eprintln!("exclusion gate: {} violation(s)", violations.len());
        ExitCode::from(1)
    }
}

fn read_manifests(crates_dir: &Path) -> std::io::Result<Vec<(String, String)>> {
    let mut manifests = Vec::new();
    for entry in std::fs::read_dir(crates_dir)? {
        let entry = entry?;
        let manifest = entry.path().join("Cargo.toml");
        if manifest.is_file() {
            let name = entry.file_name().to_string_lossy().into_owned();
            manifests.push((name, std::fs::read_to_string(manifest)?));
        }
    }
    manifests.sort();
    Ok(manifests)
}

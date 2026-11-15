//! `exclusion-gate`: the executable form of SPECS.md Section 2.1 item 4,
//! row A-25, and the Section 2.4 services gate (row A-51). Exit code 0
//! means every check passed; 1 means at least one violation, each printed
//! on its own line to stderr; 2 means the gate could not gather its
//! inputs.
//!
//! Usage: `exclusion-gate [workspace-root]`. The root defaults to the
//! workspace containing this crate. The checks themselves are in
//! `robot_devtools::exclusion_gate` and `robot_devtools::services_gate`;
//! this binary only reads `Cargo.lock`, runs `cargo tree --workspace -e
//! normal --prefix none --offline`, reads every `crates/*/Cargo.toml`,
//! and lists `services/*/` (an absent `services/` directory is an empty,
//! passing registry).

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use robot_devtools::exclusion_gate::{
    devtools_dependents, excluded_names_in_lock, foreign_packages_in_tree, Violation,
};
use robot_devtools::services_gate::{check_services, ServiceEntry};
use robot_devtools::sources_gate;

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

    let services = match read_services(&root.join("services")) {
        Ok(services) => services,
        Err(error) => {
            eprintln!("exclusion gate: cannot read services registry: {error}");
            return ExitCode::from(2);
        }
    };

    // One registry, two record kinds (ADR 0021): entries whose record
    // declares data-source/1 go to the sources gate, everything else -
    // including a missing or unparsable record - stays with the services
    // gate, whose refusals cover those cases.
    let (source_entries, service_entries): (Vec<ServiceEntry>, Vec<ServiceEntry>) =
        services.into_iter().partition(|e| {
            e.record
                .as_deref()
                .is_some_and(sources_gate::is_data_source)
        });

    let mut violations: Vec<Violation> = excluded_names_in_lock(&lock);
    violations.extend(foreign_packages_in_tree(&tree, &root));
    violations.extend(devtools_dependents(&manifests));
    let service_violations = check_services(&service_entries);
    let source_violations: Vec<sources_gate::Violation> = source_entries
        .iter()
        .filter_map(|e| e.record.as_deref().map(|text| (e, text)))
        .flat_map(|(e, text)| sources_gate::check_record(&e.name, text))
        .collect();

    if violations.is_empty() && service_violations.is_empty() && source_violations.is_empty() {
        let packages = tree.lines().filter(|l| !l.trim().is_empty()).count();
        let lock_entries = lock
            .lines()
            .filter(|l| l.trim_start().starts_with("name = "))
            .count();
        println!(
            "exclusion gate: ok ({} default-closure entries, all workspace paths; {} lockfile packages, none an excluded runtime; {} manifests, none reach robot-devtools; {} service record(s), all valid robot-service/1; {} data source record(s), all valid data-source/1)",
            packages,
            lock_entries,
            manifests.len(),
            service_entries.len(),
            source_entries.len()
        );
        ExitCode::SUCCESS
    } else {
        for violation in &violations {
            eprintln!("exclusion gate: {violation}");
        }
        for violation in &source_violations {
            eprintln!("exclusion gate: {violation}");
        }
        for violation in &service_violations {
            eprintln!("exclusion gate: {violation}");
        }
        eprintln!(
            "exclusion gate: {} violation(s)",
            violations.len() + service_violations.len() + source_violations.len()
        );
        ExitCode::from(1)
    }
}

fn read_services(services_dir: &Path) -> std::io::Result<Vec<ServiceEntry>> {
    let mut services = Vec::new();
    if !services_dir.is_dir() {
        return Ok(services);
    }
    for entry in std::fs::read_dir(services_dir)? {
        let entry = entry?;
        if !entry.path().is_dir() {
            continue;
        }
        let record = match std::fs::read_to_string(entry.path().join("service.toml")) {
            Ok(text) => Some(text),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e),
        };
        services.push(ServiceEntry {
            name: entry.file_name().to_string_lossy().into_owned(),
            record,
            has_containerfile: entry.path().join("Containerfile").is_file(),
        });
    }
    services.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(services)
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

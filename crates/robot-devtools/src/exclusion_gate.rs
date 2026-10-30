//! The excluded-runtime and tooling-isolation gate of SPECS.md Section 2.1
//! item 4 and Section 6 (row A-25), as pure text checks.
//!
//! SPECS.md Section 2.1 item 4: "The default build MUST fail if an excluded
//! runtime appears in its dependency closure, and CI MUST enforce it."
//! Section 2.2 item 6 forbids branching or parsing in the devenv scripts,
//! so the logic lives here, in Rust, and `devenv.nix` only invokes the
//! `exclusion-gate` binary of this crate. Three checks are made:
//!
//! 1. [`excluded_names_in_lock`]: no package in `Cargo.lock`, under any
//!    feature, is one of the excluded runtimes of Section 2 and Section 1.1
//!    (Python bindings, PyTorch, JAX, MuJoCo, CycloneDDS). A feature-gated
//!    exception such as `cubecl` (ADR 0012) is allowed to appear in the
//!    lockfile; an excluded runtime is not, because Section 2 admits no
//!    exception for them at all.
//! 2. [`foreign_packages_in_tree`]: the default (feature-off, normal
//!    dependency) closure printed by `cargo tree` contains only workspace
//!    path packages, which is ADR 0003's standing claim.
//! 3. [`devtools_dependents`]: no crate reaches `robot-devtools` through a
//!    normal or build dependency, which is the row A-25 assertion that the
//!    development-time automation of Section 2.3.1 ships in nothing.
//!
//! Every function here takes text and returns the violations it found; the
//! binary gathers the text and exits non-zero on any violation.

use std::fmt;
use std::path::Path;

/// Exact package names that are excluded runtimes or their bindings.
pub const EXCLUDED_NAMES: &[&str] = &[
    "pyo3",
    "cpython",
    "rust-cpython",
    "python3-sys",
    "inline-python",
    "numpy",
    "tch",
    "torch-sys",
    "mujoco",
    "mujoco-rs",
    "mujoco-rust",
    "mujoco-sys",
    "cyclonedds",
    "cyclonedds-rs",
    "cyclonedds-sys",
    "jax",
];

/// Name prefixes that denote a crate family of an excluded runtime.
pub const EXCLUDED_PREFIXES: &[&str] = &["pyo3-", "mujoco-", "cyclonedds-", "torch-", "tch-"];

/// One reason the gate fails.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Violation {
    /// An excluded runtime is in the lockfile.
    ExcludedRuntime {
        /// The lockfile package name.
        package: String,
    },
    /// A package in the default closure is not a workspace path package.
    ForeignPackage {
        /// The `cargo tree` line, trimmed.
        entry: String,
    },
    /// A crate other than `robot-devtools` depends on it outside
    /// `[dev-dependencies]`.
    DevtoolsDependent {
        /// The depending crate.
        krate: String,
        /// The manifest section the dependency appears in.
        section: String,
    },
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExcludedRuntime { package } => write!(
                f,
                "excluded runtime `{package}` is in Cargo.lock (SPECS.md Section 2.1 item 4)"
            ),
            Self::ForeignPackage { entry } => write!(
                f,
                "default closure contains a non-workspace package: {entry} (ADR 0003)"
            ),
            Self::DevtoolsDependent { krate, section } => write!(
                f,
                "`{krate}` depends on robot-devtools in `{section}` (SPECS.md Section 6, row A-25)"
            ),
        }
    }
}

/// Whether a lockfile package name is an excluded runtime.
#[must_use]
pub fn is_excluded_name(name: &str) -> bool {
    EXCLUDED_NAMES.contains(&name) || EXCLUDED_PREFIXES.iter().any(|p| name.starts_with(p))
}

/// Every `name = "..."` entry of a `Cargo.lock` text that names an excluded
/// runtime, in file order, without duplicates.
#[must_use]
pub fn excluded_names_in_lock(lock: &str) -> Vec<Violation> {
    let mut found: Vec<Violation> = Vec::new();
    for line in lock.lines() {
        let Some(rest) = line.trim().strip_prefix("name = ") else {
            continue;
        };
        let name = rest.trim().trim_matches('"');
        if is_excluded_name(name) {
            let violation = Violation::ExcludedRuntime {
                package: name.to_owned(),
            };
            if !found.contains(&violation) {
                found.push(violation);
            }
        }
    }
    found
}

/// Every line of a `cargo tree --prefix none` listing that does not name a
/// package whose manifest directory lies under `workspace_root`.
///
/// `cargo tree` prints a path package as `name vX.Y.Z (/abs/path)` and a
/// registry package without a path. A `(*)` de-duplication marker and a
/// `(proc-macro)` marker are ignored.
#[must_use]
pub fn foreign_packages_in_tree(tree: &str, workspace_root: &Path) -> Vec<Violation> {
    let mut found: Vec<Violation> = Vec::new();
    for raw in tree.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let in_workspace = parenthesised(line)
            .filter(|inner| inner.starts_with('/'))
            .any(|inner| Path::new(inner).starts_with(workspace_root));
        if !in_workspace {
            let violation = Violation::ForeignPackage {
                entry: line.to_owned(),
            };
            if !found.contains(&violation) {
                found.push(violation);
            }
        }
    }
    found
}

/// Every crate whose manifest declares `robot-devtools` under a normal,
/// build or target-specific (non-dev) dependency table.
///
/// `manifests` pairs a crate name with its `Cargo.toml` text. The crate
/// named `robot-devtools` itself is skipped.
#[must_use]
pub fn devtools_dependents(manifests: &[(String, String)]) -> Vec<Violation> {
    let mut found = Vec::new();
    for (krate, text) in manifests {
        if krate == "robot-devtools" {
            continue;
        }
        let mut section = String::new();
        for raw in text.lines() {
            let line = raw.trim();
            if line.starts_with('[') {
                section = line.trim_matches(|c| c == '[' || c == ']').to_owned();
                continue;
            }
            if !names_devtools(line) {
                continue;
            }
            if is_shipping_dependency_table(&section) {
                found.push(Violation::DevtoolsDependent {
                    krate: krate.clone(),
                    section: section.clone(),
                });
            }
        }
    }
    found
}

fn names_devtools(line: &str) -> bool {
    let Some(rest) = line.strip_prefix("robot-devtools") else {
        return false;
    };
    rest.trim_start().starts_with('=') || rest.trim_start().starts_with('.')
}

fn is_shipping_dependency_table(section: &str) -> bool {
    let last = section.rsplit('.').next().unwrap_or(section);
    matches!(last, "dependencies" | "build-dependencies")
        || section.starts_with("dependencies.")
        || section.starts_with("build-dependencies.")
}

fn parenthesised(line: &str) -> impl Iterator<Item = &str> {
    line.match_indices('(').filter_map(move |(start, _)| {
        let rest = &line[start + 1..];
        rest.find(')').map(|end| &rest[..end])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_excluded_runtime_in_the_lockfile_is_reported_once() {
        let lock = "[[package]]\nname = \"serde\"\nversion = \"1.0.0\"\n\n[[package]]\nname = \"pyo3\"\nversion = \"0.21.0\"\n\n[[package]]\nname = \"pyo3\"\nversion = \"0.22.0\"\n\n[[package]]\nname = \"mujoco-sys\"\nversion = \"0.1.0\"\n";
        let found = excluded_names_in_lock(lock);
        assert_eq!(
            found,
            vec![
                Violation::ExcludedRuntime {
                    package: "pyo3".to_owned()
                },
                Violation::ExcludedRuntime {
                    package: "mujoco-sys".to_owned()
                },
            ]
        );
    }

    #[test]
    fn allowed_names_that_share_letters_with_excluded_ones_pass() {
        for name in [
            "cubecl",
            "wgpu",
            "tchannel",
            "jaxb",
            "python-like",
            "numpy-ish",
        ] {
            assert!(!is_excluded_name(name), "{name}");
        }
        for name in ["pyo3-ffi", "tch", "torch-sys", "cyclonedds-rs", "jax"] {
            assert!(is_excluded_name(name), "{name}");
        }
    }

    #[test]
    fn only_workspace_path_packages_pass_the_tree_check() {
        let root = Path::new("/work/robot-control");
        let tree = "\
robot-cli v0.1.0 (/work/robot-control/crates/robot-cli)
robot-types v0.1.0 (/work/robot-control/crates/robot-types) (*)

robot-macros v0.1.0 (/work/robot-control/crates/robot-macros) (proc-macro)
";
        assert!(foreign_packages_in_tree(tree, root).is_empty());

        let bad = "robot-cli v0.1.0 (/work/robot-control/crates/robot-cli)\nserde v1.0.0\nlocal v0.1.0 (/elsewhere/local)\n";
        let found = foreign_packages_in_tree(bad, root);
        assert_eq!(found.len(), 2);
        assert!(
            matches!(&found[0], Violation::ForeignPackage { entry } if entry == "serde v1.0.0")
        );
        assert!(
            matches!(&found[1], Violation::ForeignPackage { entry } if entry == "local v0.1.0 (/elsewhere/local)")
        );
    }

    #[test]
    fn a_normal_or_build_dependency_on_devtools_fails_and_a_dev_dependency_passes() {
        let manifests = vec![
            (
                "robot-control".to_owned(),
                "[package]\nname = \"robot-control\"\n\n[dependencies]\nrobot-types = { path = \"../robot-types\" }\nrobot-devtools = { path = \"../robot-devtools\" }\n".to_owned(),
            ),
            (
                "robot-eval".to_owned(),
                "[dev-dependencies]\nrobot-devtools = { path = \"../robot-devtools\" }\n".to_owned(),
            ),
            (
                "robot-model".to_owned(),
                "[build-dependencies]\nrobot-devtools.path = \"../robot-devtools\"\n".to_owned(),
            ),
            (
                "robot-devtools".to_owned(),
                "[dependencies]\nrobot-devtools = { path = \".\" }\n".to_owned(),
            ),
        ];
        let found = devtools_dependents(&manifests);
        assert_eq!(
            found,
            vec![
                Violation::DevtoolsDependent {
                    krate: "robot-control".to_owned(),
                    section: "dependencies".to_owned()
                },
                Violation::DevtoolsDependent {
                    krate: "robot-model".to_owned(),
                    section: "build-dependencies".to_owned()
                },
            ]
        );
    }

    #[test]
    fn violations_name_the_governing_clause() {
        let text = Violation::ExcludedRuntime {
            package: "tch".to_owned(),
        }
        .to_string();
        assert!(text.contains("Section 2.1 item 4"), "{text}");
    }
}

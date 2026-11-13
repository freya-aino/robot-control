//! The attached-agent robot discovery method of SPECS.md Section 7.7, as
//! the deterministic, offline core rows A-47 to A-50 test.
//!
//! - **Sources and ranking** (item 1): every value is a [`Claim`] from a
//!   [`Source`]; the ranking is description, device interface, measurement,
//!   web, and [`resolve`] uses the highest-ranked value.
//! - **Conflicts** (item 7): values that disagree beyond a tolerance are all
//!   listed with their sources in a [`Conflict`]; nothing is averaged, and
//!   any conflict blocks activation until a human resolves it.
//! - **Web claims are proposals** (items 6, A-49): a web value is used only
//!   when nothing ranked higher exists, and then as a *proposal*, with its
//!   URL, date, version, content hash, model, firmware and query; a
//!   proposal can never be the value an acceptance row or hazard control
//!   cites.
//! - **Motion and range** (item 2): [`sample_workspace`] samples each
//!   link's reachable positions by forward kinematics with joint values
//!   drawn inside the declared limits, recording seed and sample count.
//! - **Tests** (item 5): the caller's applicable tests, each classified as
//!   simulation, hardware or manual; a test is passed only with an artifact.
//! - **Probe refusal** (item 3, A-48): [`check_probe`] refuses a commanded
//!   position without a prior simulated dry run or outside the declared
//!   limits shrunk by the profile's margin.
//! - **The agent cannot activate** (item 8, A-50): [`DiscoveryReport`]
//!   carries no activation path, and [`DiscoveryReport::blocks_activation`]
//!   is true until a human decision record is attached.
//!
//! The report renders to text deterministically ([`DiscoveryReport::render`]),
//! so a second run on the same inputs reproduces it byte for byte (A-47).

use std::fmt::Write as _;

use robot_types::frames::Vec3;
use robot_types::rng::{Rng, SeedStream};

use crate::kinematics::forward_kinematics;
use crate::profile::RobotProfile;

/// Where a value came from, highest rank first.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Source {
    /// The versioned robot description (Section 7.1).
    Description {
        /// Profile name and version.
        profile: String,
    },
    /// The robot's declared interface data, read in commissioning.
    DeviceInterface {
        /// Firmware version reported.
        firmware: String,
    },
    /// A probe or logged-episode measurement.
    Measurement {
        /// The run or episode.
        run: String,
    },
    /// A retrieved document.
    Web {
        /// URL.
        url: String,
        /// Retrieval date.
        retrieved: String,
        /// Document version.
        version: String,
        /// Content hash.
        hash: String,
        /// Robot model and firmware the document claims to apply to.
        applies_to: String,
        /// The query that found it.
        query: String,
    },
}

impl Source {
    /// Rank: 0 is highest.
    #[must_use]
    pub const fn rank(&self) -> u8 {
        match self {
            Self::Description { .. } => 0,
            Self::DeviceInterface { .. } => 1,
            Self::Measurement { .. } => 2,
            Self::Web { .. } => 3,
        }
    }

    fn label(&self) -> String {
        match self {
            Self::Description { profile } => format!("description {profile}"),
            Self::DeviceInterface { firmware } => format!("device firmware {firmware}"),
            Self::Measurement { run } => format!("measurement {run}"),
            Self::Web {
                url,
                retrieved,
                hash,
                ..
            } => format!("web {url} ({retrieved}, {hash})"),
        }
    }
}

/// One value of one quantity from one source.
#[derive(Clone, PartialEq, Debug)]
pub struct Claim {
    /// The quantity, e.g. `joint/left_knee_joint/position_max`.
    pub quantity: String,
    /// The value.
    pub value: f64,
    /// Its source.
    pub source: Source,
}

/// Disagreeing values of one quantity.
#[derive(Clone, PartialEq, Debug)]
pub struct Conflict {
    /// The quantity.
    pub quantity: String,
    /// Every value with its source label.
    pub values: Vec<(f64, String)>,
    /// The value used meanwhile (highest rank).
    pub used: f64,
}

/// The value used for one quantity.
#[derive(Clone, PartialEq, Debug)]
pub struct Resolved {
    /// The quantity.
    pub quantity: String,
    /// The value.
    pub value: f64,
    /// Its source label.
    pub source: String,
    /// Only a web source supplied it: a proposal, never evidence.
    pub proposal: bool,
}

/// Resolves claims by rank; values within `tolerance` of the used one
/// agree, any other value is a conflict.
#[must_use]
pub fn resolve(claims: &[Claim], tolerance: f64) -> (Vec<Resolved>, Vec<Conflict>) {
    let mut quantities: Vec<&str> = claims.iter().map(|c| c.quantity.as_str()).collect();
    quantities.sort_unstable();
    quantities.dedup();
    let mut used = Vec::new();
    let mut conflicts = Vec::new();
    for q in quantities {
        let mut these: Vec<&Claim> = claims.iter().filter(|c| c.quantity == q).collect();
        these.sort_by_key(|c| c.source.rank());
        let best = these[0];
        used.push(Resolved {
            quantity: q.to_owned(),
            value: best.value,
            source: best.source.label(),
            proposal: matches!(best.source, Source::Web { .. }),
        });
        if these
            .iter()
            .any(|c| (c.value - best.value).abs() > tolerance)
        {
            conflicts.push(Conflict {
                quantity: q.to_owned(),
                values: these.iter().map(|c| (c.value, c.source.label())).collect(),
                used: best.value,
            });
        }
    }
    (used, conflicts)
}

/// A sampled reachable region of one link.
#[derive(Clone, PartialEq, Debug)]
pub struct Workspace {
    /// Link name.
    pub link: String,
    /// Componentwise minimum, root frame.
    pub min: Vec3,
    /// Componentwise maximum, root frame.
    pub max: Vec3,
}

/// Samples `samples` joint configurations uniformly inside the declared
/// position limits (a joint without limits spans ±pi) and records each
/// listed link's bounding box.
pub fn sample_workspace(
    profile: &RobotProfile,
    links: &[usize],
    samples: usize,
    seed: SeedStream,
) -> Vec<Workspace> {
    let mut rng = Rng::new(seed);
    let ranges: Vec<(f64, f64)> = profile
        .actuated_joints()
        .map(|j| {
            j.limits
                .position
                .map_or((-core::f64::consts::PI, core::f64::consts::PI), |r| {
                    (r.lo.value(), r.hi.value())
                })
        })
        .collect();
    let mut out: Vec<Workspace> = links
        .iter()
        .map(|&l| Workspace {
            link: profile.links[l].name.to_string(),
            min: Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
            max: Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
        })
        .collect();
    for _ in 0..samples {
        let q: Vec<f64> = ranges
            .iter()
            .map(|(lo, hi)| rng.uniform(*lo, *hi))
            .collect();
        let Ok(poses) = forward_kinematics(profile, &q) else {
            continue;
        };
        for (w, &l) in out.iter_mut().zip(links) {
            if let Some(p) = poses.by_index(l) {
                let t = p.translation;
                w.min = Vec3::new(w.min.x.min(t.x), w.min.y.min(t.y), w.min.z.min(t.z));
                w.max = Vec3::new(w.max.x.max(t.x), w.max.y.max(t.y), w.max.z.max(t.z));
            }
        }
    }
    out
}

/// How a test can be run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TestClass {
    /// In simulation.
    Simulation,
    /// On hardware, under the commissioning probe.
    Hardware,
    /// By a person.
    Manual,
}

/// One applicable test and its outcome.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TestEntry {
    /// The test (an acceptance row or fidelity check).
    pub id: String,
    /// How it runs.
    pub class: TestClass,
    /// The evidence-ledger artifact its run produced, if it was run.
    pub artifact: Option<String>,
}

/// Why a probe command was refused.
#[derive(Clone, PartialEq, Debug)]
pub enum ProbeAbort {
    /// No simulated dry run preceded it.
    NoDryRun,
    /// A joint target lies outside its limits shrunk by the margin.
    OutsideMargin {
        /// Actuated-joint index.
        joint: usize,
        /// The target.
        target: f64,
    },
    /// The width is wrong.
    Width,
}

/// Checks one commissioning probe command (Section 7.7 item 3). An abort
/// quarantines the candidate profile; the caller records it.
pub fn check_probe(
    profile: &RobotProfile,
    target: &[f64],
    dry_run_passed: bool,
    margin: f64,
) -> Result<(), ProbeAbort> {
    if !dry_run_passed {
        return Err(ProbeAbort::NoDryRun);
    }
    if target.len() != profile.dof() {
        return Err(ProbeAbort::Width);
    }
    for (i, (j, &t)) in profile.actuated_joints().zip(target).enumerate() {
        if let Some(r) = j.limits.position {
            if t < r.lo.value() + margin || t > r.hi.value() - margin {
                return Err(ProbeAbort::OutsideMargin {
                    joint: i,
                    target: t,
                });
            }
        }
    }
    Ok(())
}

/// The discovery report.
#[derive(Clone, PartialEq, Debug)]
pub struct DiscoveryReport {
    /// Robot profile name and version.
    pub robot: String,
    /// Values used.
    pub values: Vec<Resolved>,
    /// Open conflicts.
    pub conflicts: Vec<Conflict>,
    /// Sampled workspaces.
    pub workspaces: Vec<Workspace>,
    /// Workspace seed and sample count.
    pub sampling: (SeedStream, usize),
    /// Applicable tests.
    pub tests: Vec<TestEntry>,
    /// The human decision record approving activation, if any.
    pub decision_record: Option<String>,
}

impl DiscoveryReport {
    /// Builds the report for `profile` from its own description plus
    /// `extra` claims.
    #[must_use]
    pub fn build(
        profile: &RobotProfile,
        extra: &[Claim],
        links: &[usize],
        samples: usize,
        seed: SeedStream,
        tests: Vec<TestEntry>,
    ) -> Self {
        let desc = Source::Description {
            profile: format!("{}@{}", profile.name, profile.profile_version),
        };
        let mut claims: Vec<Claim> = Vec::new();
        for j in profile.actuated_joints() {
            let mut add = |what: &str, v: f64| {
                claims.push(Claim {
                    quantity: format!("joint/{}/{what}", j.name),
                    value: v,
                    source: desc.clone(),
                });
            };
            if let Some(r) = j.limits.position {
                add("position_min", r.lo.value());
                add("position_max", r.hi.value());
            }
            add("velocity_max", j.limits.velocity.value());
            add("effort_max", j.limits.effort.value());
        }
        for l in &profile.links {
            claims.push(Claim {
                quantity: format!("link/{}/mass", l.name),
                value: l.mass.mass.value(),
                source: desc.clone(),
            });
        }
        claims.extend_from_slice(extra);
        let (values, conflicts) = resolve(&claims, 1e-9);
        Self {
            robot: format!("{}@{}", profile.name, profile.profile_version),
            values,
            conflicts,
            workspaces: sample_workspace(profile, links, samples, seed),
            sampling: (seed, samples),
            tests,
            decision_record: None,
        }
    }

    /// Activation stays blocked while any conflict is open or no human
    /// decision record is attached.
    #[must_use]
    pub fn blocks_activation(&self) -> bool {
        !self.conflicts.is_empty() || self.decision_record.is_none()
    }

    /// Deterministic text.
    #[must_use]
    pub fn render(&self) -> String {
        let mut s = format!("discovery report: {}\n", self.robot);
        let _ = writeln!(
            s,
            "workspace sampling: seed {} stream {}, {} samples",
            self.sampling.0.seed, self.sampling.0.stream, self.sampling.1
        );
        for v in &self.values {
            let tag = if v.proposal { " [PROPOSAL]" } else { "" };
            let _ = writeln!(
                s,
                "value {} = {:.9} from {}{tag}",
                v.quantity, v.value, v.source
            );
        }
        for c in &self.conflicts {
            let _ = writeln!(s, "CONFLICT {} (using {:.9}):", c.quantity, c.used);
            for (v, src) in &c.values {
                let _ = writeln!(s, "  {v:.9} from {src}");
            }
        }
        for w in &self.workspaces {
            let _ = writeln!(
                s,
                "workspace {}: x [{:.6}, {:.6}] y [{:.6}, {:.6}] z [{:.6}, {:.6}]",
                w.link, w.min.x, w.max.x, w.min.y, w.max.y, w.min.z, w.max.z
            );
        }
        for t in &self.tests {
            let status = t
                .artifact
                .as_deref()
                .map_or("not run".to_owned(), |a| format!("run, artifact {a}"));
            let _ = writeln!(s, "test {} ({:?}): {status}", t.id, t.class);
        }
        let _ = writeln!(
            s,
            "activation: {}",
            if self.blocks_activation() {
                "blocked"
            } else {
                "permitted by decision record"
            }
        );
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::unitree_g1_29dof;

    fn web(value: f64) -> Claim {
        Claim {
            quantity: "joint/left_knee_joint/position_max".into(),
            value,
            source: Source::Web {
                url: "https://example.org/g1-manual.pdf".into(),
                retrieved: "2026-11-13".into(),
                version: "1.2".into(),
                hash: "sha256:00".into(),
                applies_to: "G1 29dof, fw 1.0".into(),
                query: "unitree g1 knee range".into(),
            },
        }
    }

    fn hand(p: &RobotProfile) -> usize {
        p.links
            .iter()
            .position(|l| l.name.as_str() == "right_rubber_hand")
            .unwrap()
    }

    #[test]
    fn the_report_is_reproducible_offline_and_blocks_activation() {
        let p = unitree_g1_29dof().unwrap();
        let tests = vec![TestEntry {
            id: "A-43".into(),
            class: TestClass::Simulation,
            artifact: None,
        }];
        let a = DiscoveryReport::build(
            &p,
            &[],
            &[hand(&p)],
            500,
            SeedStream::new(9, 0),
            tests.clone(),
        );
        let b = DiscoveryReport::build(&p, &[], &[hand(&p)], 500, SeedStream::new(9, 0), tests);
        assert_eq!(a.render(), b.render());
        assert!(a.conflicts.is_empty());
        assert!(a.blocks_activation());
        assert!(a.render().contains("test A-43 (Simulation): not run"));
        let w = &a.workspaces[0];
        assert!(w.max.x > w.min.x && w.max.z > w.min.z);
    }

    #[test]
    fn a_conflicting_web_value_is_listed_not_used_or_averaged() {
        let p = unitree_g1_29dof().unwrap();
        let r = DiscoveryReport::build(&p, &[web(3.0)], &[], 0, SeedStream::new(1, 0), Vec::new());
        assert_eq!(r.conflicts.len(), 1);
        let c = &r.conflicts[0];
        assert_eq!(c.values.len(), 2);
        assert!(c.used < 3.0, "the description outranks the web");
        assert!(r
            .render()
            .contains("CONFLICT joint/left_knee_joint/position_max"));
    }

    #[test]
    fn a_web_only_value_is_a_proposal() {
        let claims = [Claim {
            quantity: "actuator/knee/backlash".into(),
            ..web(0.01)
        }];
        let (used, conflicts) = resolve(&claims, 1e-9);
        assert!(conflicts.is_empty());
        assert!(used[0].proposal);
    }

    #[test]
    fn a_probe_without_a_dry_run_or_past_the_margin_aborts() {
        let p = unitree_g1_29dof().unwrap();
        let zero = vec![0.0; p.dof()];
        assert_eq!(
            check_probe(&p, &zero, false, 0.05),
            Err(ProbeAbort::NoDryRun)
        );
        assert_eq!(check_probe(&p, &zero, true, 0.05), Ok(()));
        let knee = p
            .actuated_joints()
            .position(|j| j.name.as_str() == "left_knee_joint")
            .unwrap();
        let hi = p
            .actuated_joints()
            .nth(knee)
            .unwrap()
            .limits
            .position
            .unwrap()
            .hi
            .value();
        let mut over = zero;
        over[knee] = hi - 0.01;
        assert!(matches!(
            check_probe(&p, &over, true, 0.05),
            Err(ProbeAbort::OutsideMargin { .. })
        ));
    }
}

//! Per-run simulation options: the fidelity knobs
//! `specs/09-fidelity-tiers.md` section 12.1 requires to be explicit,
//! per-run and queryable rather than compile-time constants.
//!
//! That section is emphatic about the form as well as the content: each knob
//! must be "an explicit, per-run, queryable parameter of a simulation
//! configuration -- never a compile-time constant and never silently defaulted
//! without the default being documented".
//! `specs/appendix-d-mathematical-challenges.md` entry M-21 records how far
//! from that the code was: only the timestep qualified, the integrator and
//! solver existed as strings validated to accept one value each, the solver
//! budget was two private constants, and `condim`, the contact-slot budget and
//! collision margin/gap did not exist at all. It also names three further
//! constants that belonged on the list because they changed the physics rather
//! than only the cost.
//!
//! Every default below is stated with its source. Where a default comes from
//! the example it is marked as such and is still this project's own choice to
//! adopt: the mapping document's rule 2 stands, and a constant taken from the
//! example needs its own measurement before any claim rests on it
//! (`SPECS.md` Section 9).
//!
//! Purity contract (FR-6): plain scalars and arrays. No `Matrix`, no
//! `RigidPose`, no `HashMap`.

use robot_types::time::Duration;

/// Which integrator a run uses.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Integrator {
    /// Semi-implicit (symplectic) Euler: velocity first, then position with
    /// the new velocity. The cheapest, and the only one the legacy
    /// `MultibodyState::step` path offers.
    SemiImplicitEuler,
    /// Classical fourth-order Runge-Kutta, including the
    /// Runge-Kutta-Munthe-Kaas correction that makes it higher order on a
    /// rotating free body (`crate::composite::CompositeState::rk4_step`).
    Rk4,
}

/// Which algorithm solves the constraint problem.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SolverAlgorithm {
    /// Projected Gauss-Seidel on the dual problem
    /// (`crate::solver`). Step 06 item 5 asks for a Newton-style method
    /// alongside this one, and
    /// `specs/11-gpu-batched-backend.md` records that a strictly sequential
    /// per-row update is the reason the example's own GPU backend drops this
    /// algorithm, so the batched path will need the Newton variant. Only this
    /// one exists today, and the enum says so rather than pretending otherwise.
    ProjectedGaussSeidel,
}

/// Convex-collision-detection budget: the knobs the narrow phase trades
/// accuracy against cost with.
///
/// These were compile-time constants until the performance investigation
/// recorded in the GJK commit, which is the clearest possible argument for
/// section 12.1's rule: the right values depend on the shapes and the regime,
/// and a single choice baked into the binary was wrong for both shallow contact
/// and deep penetration at once. The example carries the same two as model
/// options (`ccd_tolerance`, `ccd_iterations`).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct CcdConfig {
    /// Relative gap between the penetration depth's lower and upper bounds at
    /// which the expansion stops.
    pub tolerance: f64,
    /// Maximum expansion iterations.
    pub iterations: usize,
    /// Maximum faces the expanding polytope may hold. A termination condition,
    /// not a performance guard: without it the face count grows geometrically
    /// for a polytope that is not a proper convex hull.
    pub max_faces: usize,
}

impl Default for CcdConfig {
    /// Defaults chosen to agree with the analytic capsule result to better than
    /// 1e-6 on the randomised fuzz set while keeping a collapsing G1 at the
    /// cost it had before the narrow phase was rewritten.
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            iterations: 128,
            max_faces: 256,
        }
    }
}

impl CcdConfig {
    /// Whether every field is usable.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.tolerance.is_finite()
            && self.tolerance > 0.0
            && self.iterations > 0
            && self.max_faces >= 4
    }
}

/// The complete per-run configuration.
///
/// The seven knobs of section 12.1 are the first seven fields, in that
/// section's own order, followed by the narrow-phase budget and the default
/// soft-constraint parameters a contact row is built with.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SimOptions {
    /// Fixed integration timestep.
    pub timestep: Duration,
    /// Integrator.
    pub integrator: Integrator,
    /// Constraint-solver algorithm.
    pub solver: SolverAlgorithm,
    /// Maximum solver sweeps.
    pub solver_iterations: usize,
    /// Solver convergence tolerance, on the largest force change in a sweep.
    pub solver_tolerance: f64,
    /// Contact dimensionality: 1 for frictionless, 3 for a normal and two
    /// friction directions. 4 and 6 (torsional and rolling friction) are not
    /// implemented and are rejected by [`SimOptions::is_valid`].
    pub condim: usize,
    /// Contact-slot budget. `None` means unbounded, which is what the CPU
    /// backend does; the GPU backend needs a fixed budget and an overflow
    /// flag (`specs/11-gpu-batched-backend.md`).
    pub contact_slots: Option<usize>,
    /// Collision margin: contacts are detected within this distance.
    pub margin: f64,
    /// Collision gap: detected within the margin but generating force only
    /// beyond the gap.
    pub gap: f64,
    /// Narrow-phase budget.
    pub ccd: CcdConfig,
    /// Whether the solver starts from the previous step's forces.
    pub warm_start: bool,
    /// Default `solref` for a contact row.
    pub solref: [f64; 2],
    /// Default `solimp` for a contact row.
    pub solimp: [f64; 5],
}

impl Default for SimOptions {
    /// Every default, with its source:
    ///
    /// - `timestep` 1e-4 s: this project's own measured choice. The CLI's
    ///   multibody command already defaults to it because the G1 diverges at
    ///   5e-4 under semi-implicit Euler (`specs/03-integrators.md`).
    /// - `integrator` semi-implicit Euler: the cheapest, and the only one the
    ///   legacy stepper offers.
    /// - `solver` projected Gauss-Seidel: the only one implemented.
    /// - `solver_iterations` 100 and `solver_tolerance` 1e-10: this project's
    ///   choice. The example's own documentation notes that one Newton
    ///   iteration is often enough for reinforcement-learning fidelity, which
    ///   makes this knob task-dependent rather than a universal accuracy dial
    ///   (section 12.1).
    /// - `condim` 3: friction on, which is the useful default for a walking
    ///   robot.
    /// - `contact_slots` `None`: unbounded on the CPU.
    /// - `margin` and `gap` 0: no early detection, which is the conservative
    ///   default and what the pipeline did before they existed.
    /// - `ccd`: see [`CcdConfig::default`].
    /// - `warm_start` true: step 06 item 7.
    /// - `solref` and `solimp`: the example's `defaultSolRefImp`. Adopted, not
    ///   measured here.
    fn default() -> Self {
        Self {
            timestep: Duration::from_micros(100),
            integrator: Integrator::SemiImplicitEuler,
            solver: SolverAlgorithm::ProjectedGaussSeidel,
            solver_iterations: 100,
            solver_tolerance: 1e-10,
            condim: 3,
            contact_slots: None,
            margin: 0.0,
            gap: 0.0,
            ccd: CcdConfig::default(),
            warm_start: true,
            solref: [0.02, 1.0],
            solimp: [0.9, 0.95, 0.001, 0.5, 2.0],
        }
    }
}

impl SimOptions {
    /// Whether every knob holds a usable value, including that `condim` is one
    /// this crate implements.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.timestep.as_secs_f64() > 0.0
            && self.timestep.as_secs_f64().is_finite()
            && self.solver_iterations > 0
            && self.solver_tolerance.is_finite()
            && self.solver_tolerance >= 0.0
            && (self.condim == 1 || self.condim == 3)
            && self.margin.is_finite()
            && self.margin >= 0.0
            && self.gap.is_finite()
            && self.gap >= 0.0
            && self.ccd.is_valid()
            && self.solref.iter().all(|v| v.is_finite())
            && self.solimp.iter().all(|v| v.is_finite())
            && self.contact_slots.is_none_or(|n| n > 0)
    }

    /// The solver settings this configuration implies.
    #[must_use]
    pub fn solver_config(&self) -> crate::solver::SolverConfig {
        crate::solver::SolverConfig {
            iterations: self.solver_iterations,
            tolerance: self.solver_tolerance,
            warm_start: self.warm_start,
        }
    }

    /// A record of every knob as `(name, value)` text, for writing into a run
    /// record.
    ///
    /// Section 12.1 requires the knobs to be *recorded with each run*, not
    /// merely settable. Without this a fidelity claim cannot be audited,
    /// because a reader cannot tell what the run was configured with.
    #[must_use]
    pub fn record(&self) -> Vec<(&'static str, String)> {
        vec![
            ("timestep_s", format!("{:e}", self.timestep.as_secs_f64())),
            ("integrator", format!("{:?}", self.integrator)),
            ("solver", format!("{:?}", self.solver)),
            ("solver_iterations", self.solver_iterations.to_string()),
            ("solver_tolerance", format!("{:e}", self.solver_tolerance)),
            ("condim", self.condim.to_string()),
            (
                "contact_slots",
                self.contact_slots
                    .map_or_else(|| "unbounded".to_owned(), |n| n.to_string()),
            ),
            ("margin", format!("{:e}", self.margin)),
            ("gap", format!("{:e}", self.gap)),
            ("ccd_tolerance", format!("{:e}", self.ccd.tolerance)),
            ("ccd_iterations", self.ccd.iterations.to_string()),
            ("ccd_max_faces", self.ccd.max_faces.to_string()),
            ("warm_start", self.warm_start.to_string()),
            ("solref", format!("{:?}", self.solref)),
            ("solimp", format!("{:?}", self.solimp)),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_configuration_is_valid_and_records_every_knob() {
        let o = SimOptions::default();
        assert!(o.is_valid());
        let record = o.record();
        // Section 12.1 enumerates seven knobs; all seven must appear by name,
        // along with the narrow-phase budget this crate needs.
        for name in [
            "timestep_s",
            "integrator",
            "solver",
            "solver_iterations",
            "solver_tolerance",
            "condim",
            "contact_slots",
            "margin",
            "gap",
            "ccd_tolerance",
            "ccd_iterations",
        ] {
            assert!(
                record.iter().any(|(k, _)| *k == name),
                "{name} is missing from the run record"
            );
        }
    }

    #[test]
    fn invalid_knobs_are_rejected_rather_than_silently_clamped() {
        let base = SimOptions::default();
        let bad = [
            SimOptions {
                timestep: Duration::ZERO,
                ..base
            },
            SimOptions {
                solver_iterations: 0,
                ..base
            },
            SimOptions {
                solver_tolerance: f64::NAN,
                ..base
            },
            // condim 4 and 6 are real in the example but not implemented here,
            // so they must be refused rather than quietly treated as 3.
            SimOptions { condim: 4, ..base },
            SimOptions { condim: 6, ..base },
            SimOptions { condim: 0, ..base },
            SimOptions {
                margin: -1.0,
                ..base
            },
            SimOptions {
                contact_slots: Some(0),
                ..base
            },
            SimOptions {
                ccd: CcdConfig {
                    iterations: 0,
                    ..base.ccd
                },
                ..base
            },
            SimOptions {
                ccd: CcdConfig {
                    max_faces: 3,
                    ..base.ccd
                },
                ..base
            },
        ];
        for (i, o) in bad.iter().enumerate() {
            assert!(!o.is_valid(), "case {i} should have been rejected: {o:?}");
        }
    }

    #[test]
    fn a_tier_is_a_named_bundle_a_caller_can_override_one_knob_of() {
        // Section 12.1's closing requirement: no tier may introduce behaviour
        // unreachable through the individual knobs, so a caller must be able to
        // start from one and change a single value.
        let coarse = SimOptions {
            timestep: Duration::from_millis(1),
            solver_iterations: 10,
            condim: 1,
            ..SimOptions::default()
        };
        assert!(coarse.is_valid());
        let one_knob_changed = SimOptions {
            condim: 3,
            ..coarse
        };
        assert!(one_knob_changed.is_valid());
        assert_eq!(one_knob_changed.solver_iterations, 10);
        assert_eq!(one_knob_changed.timestep, Duration::from_millis(1));
        assert_eq!(one_knob_changed.condim, 3);
    }

    #[test]
    fn the_solver_configuration_follows_the_options() {
        let o = SimOptions {
            solver_iterations: 7,
            solver_tolerance: 1e-3,
            warm_start: false,
            ..SimOptions::default()
        };
        let c = o.solver_config();
        assert_eq!(c.iterations, 7);
        assert!((c.tolerance - 1e-3).abs() < 1e-18);
        assert!(!c.warm_start);
    }
}

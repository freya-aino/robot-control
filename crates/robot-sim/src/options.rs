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
    /// `implicitfast`: a semi-implicit position update with the velocity
    /// derivative of the smooth forces folded into the solve
    /// (`mj_implicitSkip` with `flg_bias = 0`, example 01 section 9.2).
    ///
    /// `specs/03-integrators.md` item 4 makes this a MUST before any claim of
    /// "stable at a usable timestep on a real humanoid", and the reason is
    /// damping. An explicit damper with gain `kd` amplifies rather than removes
    /// velocity once `kd > m_eff / (2 h)`, so an explicit step bounds the
    /// damping a model may declare. Solving `(M - h dF/dv) a = F` instead has no
    /// such bound: the stiffer the damper, the more the matrix dominates.
    ///
    /// This matters for the G1 specifically because its drives carry their
    /// damping as a force, so under [`Self::SemiImplicitEuler`] that damping is
    /// explicit and a stiffer gain set than the layer's declared one could not
    /// be used.
    ImplicitFast,
    /// Classical fourth-order Runge-Kutta, including the
    /// Runge-Kutta-Munthe-Kaas correction that makes it higher order on a
    /// rotating free body (`crate::composite::CompositeState::rk4_step`).
    Rk4,
}

/// Which algorithm solves the constraint problem.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SolverAlgorithm {
    /// The primal Newton method of `specs/08-solvers-islands.md` section 6
    /// (`crate::newton`), on the acceleration.
    ///
    /// The default, because it is the only one of the two that converges on a
    /// pyramidal contact problem at a sane budget. Measured on sixteen
    /// pyramidal rows over a rank-six problem: **one** iteration to a scaled
    /// gradient norm of 4.9e-14, against 1011 Gauss-Seidel sweeps for the same
    /// answer. The reason is structural -- its variable is the acceleration,
    /// one per degree of freedom, where the dual has one redundant coefficient
    /// per pyramid edge.
    Newton,
    /// Projected Gauss-Seidel on the dual problem (`crate::solver`).
    ///
    /// Kept as the independent cross-check on [`Self::Newton`]: the two solve
    /// the same convex problem by different routes and agree on the
    /// acceleration to 1e-8, which is how the Newton path was validated.
    ///
    /// Not suitable as the default. Its per-row update is strictly sequential,
    /// which `specs/11-gpu-batched-backend.md` records as the reason the
    /// example's own GPU backend drops the algorithm, and on pyramidal rows it
    /// needs roughly a thousand sweeps where Newton needs one.
    ProjectedGaussSeidel,
}

/// A named fidelity tier: a pure bundle of the section 12.1 knobs, nothing
/// more (section 12.2's closing requirement — no tier may introduce behavior
/// unreachable through the individual knobs, so a caller can start from a
/// tier and override one knob and still reason about what changed).
///
/// The names and settings resolve Appendix B **S-03**, grounded in the
/// 2026-11-15 G1 stance sweep (`robot sim multibody --floating`, 2 s, every
/// integrator at `dt` 1e-4 to 1e-3; the numbers are in
/// `specs/09-fidelity-tiers.md`): semi-implicit Euler passes the stance
/// battery up to 5e-4 and fails its settled-energy check at 1e-3; RK4
/// diverges outright at 1e-3 (singular mass matrix at 0.58 s); and
/// `implicitfast` passes at every tested timestep including 1e-3, which is
/// the textbook implicit-damping payoff measured rather than assumed.
///
/// Following Brax's named-pipeline precedent (section 12.2) rather than the
/// example's unnamed knob pile.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tier {
    /// Early-training rollout volume: the cheapest configuration that still
    /// passes the full supported-stance battery. `implicitfast` at 1 ms —
    /// ten times fewer steps than [`Self::Reference`] for the same 2 s of
    /// simulated time, and the only integrator that holds the battery there.
    Rollout,
    /// The validated CPU reference settings every existing claim in this
    /// crate is measured at: semi-implicit Euler at 100 µs, Newton, the
    /// documented defaults of [`SimOptions::default`]. This tier *is* that
    /// default, by construction.
    Reference,
    /// Final acceptance-style evaluation: RK4 at 100 µs, for the
    /// long-horizon energy behaviour RK4 is kept for (measured over 100
    /// times below the semi-implicit step's drift on a tumbling free body).
    Acceptance,
}

impl Tier {
    /// The tier's knob bundle. A caller overrides individual fields from
    /// here; the bundle itself is a plain `match`, per the purity contract.
    #[must_use]
    pub fn options(self) -> SimOptions {
        match self {
            Self::Rollout => SimOptions {
                timestep: Duration::from_millis(1),
                integrator: Integrator::ImplicitFast,
                ..SimOptions::default()
            },
            Self::Reference => SimOptions::default(),
            Self::Acceptance => SimOptions {
                integrator: Integrator::Rk4,
                ..SimOptions::default()
            },
        }
    }

    /// The tier's stable name, as runs record it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Rollout => "rollout",
            Self::Reference => "reference",
            Self::Acceptance => "acceptance",
        }
    }

    /// Parses a tier by its stable name; anything else is `None`, refused
    /// rather than defaulted, like every other knob.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "rollout" => Some(Self::Rollout),
            "reference" => Some(Self::Reference),
            "acceptance" => Some(Self::Acceptance),
            _ => None,
        }
    }
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
    /// Contact dimensionality: 1 frictionless, 3 adding two sliding friction
    /// directions, 4 adding torsional friction about the normal, 6 adding
    /// rolling friction about the two tangents.
    ///
    /// All four of the example's values are implemented, which is `specs/06`
    /// item 4's MUST. A direction whose friction coefficient is zero
    /// contributes no constraint rows, so 4 and 6 behave exactly as 3 does
    /// until a torsional or rolling coefficient is supplied — the
    /// dimensionality is available rather than forced.
    pub condim: usize,
    /// Contact-slot budget: the number of contact points a step may carry.
    /// `None` means unbounded. The CPU path enforces the budget in
    /// `World::build_rows`, counting a discarded point in
    /// `StepReport::dropped_contacts` rather than dropping it silently; the
    /// GPU backend needs the same fixed budget plus an overflow flag
    /// (`specs/11-gpu-batched-backend.md`).
    pub contact_slots: Option<usize>,
    /// Collision margin: contacts are detected within this distance.
    pub margin: f64,
    /// Collision gap: detected within the margin but generating force only
    /// beyond the gap.
    pub gap: f64,
    /// Joint-limit margin: a limit row is created once the joint comes within
    /// this distance of a bound, rather than only after crossing it.
    ///
    /// Separate from `margin` because the example keeps `jnt_margin` and
    /// `geom_margin` separate, and because they do different jobs. Its effect
    /// is large and easy to misread as a solver weakness: with no margin a
    /// limit row appears only once `dist < 0`, so a joint arriving at speed is
    /// braked only after it has crossed, and overshoots. Measured on a G1 joint
    /// driven at 50 N m, the worst violation is 0.18 rad at zero margin and
    /// **exactly zero** at 0.2 rad, because the constraint then has room to
    /// decelerate the joint before the bound. The cost is that the limit starts
    /// resisting early, which is the trade the knob exists to expose.
    pub limit_margin: f64,
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
    /// - `contact_slots` `None`: unbounded, so nothing is ever dropped unless
    ///   a run asks for a budget.
    /// - `margin`, `gap` and `limit_margin` 0: no early detection, which is
    ///   the conservative default, what the pipeline did before they existed,
    ///   and what the example's own `geom_margin` and `jnt_margin` default to.
    /// - `ccd`: see [`CcdConfig::default`].
    /// - `warm_start` true: step 06 item 7.
    /// - `solref` and `solimp`: the example's `defaultSolRefImp`. Adopted, not
    ///   measured here.
    fn default() -> Self {
        Self {
            timestep: Duration::from_micros(100),
            integrator: Integrator::SemiImplicitEuler,
            solver: SolverAlgorithm::Newton,
            solver_iterations: 100,
            solver_tolerance: 1e-10,
            condim: 3,
            contact_slots: None,
            margin: 0.0,
            gap: 0.0,
            limit_margin: 0.0,
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
            && (matches!(self.condim, 1 | 3 | 4 | 6))
            && self.margin.is_finite()
            && self.margin >= 0.0
            && self.gap.is_finite()
            && self.gap >= 0.0
            && self.limit_margin.is_finite()
            && self.limit_margin >= 0.0
            && self.ccd.is_valid()
            && self.solref.iter().all(|v| v.is_finite())
            // Mixed solref formats (one positive time-constant entry, one
            // negative direct-stiffness entry) are refused by name in
            // `constraint::effective_timeconst`; the same input must not be
            // silently reinterpreted on this path.
            && (self.solref[0] > 0.0) == (self.solref[1] > 0.0)
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
            ("limit_margin", format!("{:e}", self.limit_margin)),
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
            "limit_margin",
        ] {
            assert!(
                record.iter().any(|(k, _)| *k == name),
                "{name} is missing from the run record"
            );
        }
    }

    #[test]
    fn every_condim_the_example_defines_is_accepted() {
        // `specs/06` item 4 makes contact dimensionality a per-contact
        // parameter, and all four of the example's values are implemented:
        // 1 frictionless, 3 sliding, 4 adding torsional, 6 adding rolling.
        let base = SimOptions::default();
        for condim in [1, 3, 4, 6] {
            assert!(
                SimOptions { condim, ..base }.is_valid(),
                "condim {condim} is one of the example's and must be accepted"
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
            // 1, 3, 4 and 6 are the example's values and all are implemented;
            // anything else is refused rather than quietly treated as the
            // nearest one.
            SimOptions { condim: 0, ..base },
            SimOptions { condim: 2, ..base },
            SimOptions { condim: 5, ..base },
            SimOptions { condim: 7, ..base },
            SimOptions {
                margin: -1.0,
                ..base
            },
            SimOptions {
                limit_margin: -1.0,
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

    #[test]
    fn every_tier_is_a_valid_bundle_and_reference_is_the_default() {
        for tier in [Tier::Rollout, Tier::Reference, Tier::Acceptance] {
            let o = tier.options();
            assert!(o.is_valid(), "{tier:?} bundles an invalid option");
            assert_eq!(Tier::from_name(tier.name()), Some(tier));
        }
        // Section 12.2: the reference tier is exactly the validated default,
        // so every existing measurement is a measurement of that tier.
        assert_eq!(Tier::Reference.options(), SimOptions::default());
        assert_eq!(
            Tier::Rollout.options().integrator,
            Integrator::ImplicitFast,
            "the rollout tier carries the only integrator that passes the stance battery at 1 ms"
        );
        assert_eq!(Tier::from_name("coarse"), None, "unknown names are refused");
    }

    #[test]
    fn a_tier_override_changes_one_knob_and_keeps_the_rest() {
        // The section's closing requirement: start from a tier, override one
        // knob, and the rest of the bundle is untouched.
        let o = SimOptions {
            timestep: Duration::from_micros(500),
            ..Tier::Rollout.options()
        };
        assert_eq!(o.integrator, Integrator::ImplicitFast);
        assert_eq!(o.timestep, Duration::from_micros(500));
        assert!(o.is_valid());
    }
}

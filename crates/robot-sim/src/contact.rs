//! The foot/ground contact qualification benchmark: a point mass falling
//! onto a compliant ground plane.
//!
//! Appendix B.4 item 3: "Test standing, slip, contact impulses, pushes and
//! impacts. Report penetration, energy behavior, tracking error and
//! sensitivity to timestep/solver settings." This module is the smallest
//! instance of that: a point mass under gravity, with ground contact
//! modeled as a penalty spring-damper along the normal and Coulomb friction
//! along the tangent — the standard soft-contact formulation, chosen
//! because it needs no constraint solver (Appendix B.2.8's QP candidates
//! remain unqualified per `docs/decisions/0003-std-only-default-closure.md`)
//! and because its penetration and energy behavior are exactly the
//! quantities the benchmark is supposed to report.

use core::fmt;
use robot_types::frames::Vec3;
use robot_types::time::Duration;

/// Parameters of the compliant ground-contact model.
#[derive(Clone, Copy, Debug)]
pub struct ContactParams {
    /// Mass of the point.
    pub mass: f64,
    /// Gravitational acceleration magnitude.
    pub gravity: f64,
    /// Normal spring stiffness (N/m of penetration).
    pub stiffness: f64,
    /// Normal damping (N per m/s of closing velocity), active only while
    /// penetrating.
    pub normal_damping: f64,
    /// Coulomb friction coefficient applied to the tangential velocity
    /// while in contact.
    pub friction: f64,
    /// Height of the ground plane (world Z).
    pub ground_height: f64,
}

/// State of the point mass: position and velocity.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ContactState {
    /// Position in world coordinates.
    pub position: Vec3,
    /// Velocity in world coordinates.
    pub velocity: Vec3,
}

/// Why a contact step was rejected.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct NonFiniteStep;

impl fmt::Display for NonFiniteStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "contact step received a non-finite or non-positive dt")
    }
}

impl core::error::Error for NonFiniteStep {}

/// A point mass with compliant ground contact.
#[derive(Clone, Copy, Debug)]
pub struct ContactBody {
    params: ContactParams,
    state: ContactState,
    /// Maximum penetration depth observed, for the qualification report.
    max_penetration: f64,
}

impl ContactBody {
    /// Builds a body at the given initial state.
    #[must_use]
    pub const fn new(params: ContactParams, initial: ContactState) -> Self {
        Self {
            params,
            state: initial,
            max_penetration: 0.0,
        }
    }

    /// Current state.
    #[must_use]
    pub const fn state(&self) -> ContactState {
        self.state
    }

    /// Penetration depth at the current state (positive when below the
    /// ground plane, zero or negative otherwise).
    #[must_use]
    pub fn penetration(&self) -> f64 {
        (self.params.ground_height - self.state.position.z).max(0.0)
    }

    /// Deepest penetration observed over the body's history.
    #[must_use]
    pub const fn max_penetration(&self) -> f64 {
        self.max_penetration
    }

    /// Mechanical energy (kinetic + gravitational potential), for the
    /// energy-behavior report Appendix B.4 item 3 asks for. The contact
    /// spring's stored energy is not included, so this quantity is expected
    /// to dip while penetrating and should not be mistaken for a
    /// conservation law — a lossy contact model is physically expected to
    /// remove energy on impact.
    #[must_use]
    pub fn energy(&self) -> f64 {
        let kinetic = 0.5 * self.params.mass * self.state.velocity.norm_squared();
        let potential = self.params.mass * self.params.gravity * self.state.position.z;
        kinetic + potential
    }

    /// Advances the state by `dt`, applying gravity, ground contact and an
    /// external force.
    pub fn step(&mut self, external_force: Vec3, dt: Duration) -> Result<(), NonFiniteStep> {
        let dt_s = dt.as_secs_f64();
        if dt_s <= 0.0 || !dt_s.is_finite() {
            return Err(NonFiniteStep);
        }
        let ContactParams {
            mass,
            gravity,
            stiffness,
            normal_damping,
            friction,
            ..
        } = self.params;

        let mut force = external_force + Vec3::new(0.0, 0.0, -mass * gravity);
        let penetration = self.penetration();
        if penetration > 0.0 {
            self.max_penetration = self.max_penetration.max(penetration);
            let closing_speed = -self.state.velocity.z; // positive while sinking
            let normal_force = (stiffness * penetration + normal_damping * closing_speed).max(0.0);
            force.z += normal_force;

            // Coulomb friction opposing tangential velocity, capped by
            // `friction * normal_force` so it cannot reverse the motion it
            // is damping (which would be a non-physical oscillation).
            let tangential = Vec3::new(self.state.velocity.x, self.state.velocity.y, 0.0);
            if let Some(direction) = tangential.normalized(1e-9) {
                let max_friction = friction * normal_force;
                let needed = tangential.norm() * mass / dt_s;
                let applied = max_friction.min(needed);
                force = force - direction * applied;
            }
        }

        if !force.is_finite() {
            return Err(NonFiniteStep);
        }

        let acceleration = force * (1.0 / mass);
        self.state.velocity = self.state.velocity + acceleration * dt_s;
        self.state.position = self.state.position + self.state.velocity * dt_s;
        Ok(())
    }
}

/// Penalty ground plane for articulated bodies (see `robot_sim::articulated`).
///
/// A contact sphere of radius `r` centred at `c` touches the plane `z =
/// ground_height` when `c.z = ground_height + r`. The normal force is the
/// same spring-damper law [`ContactBody`] uses, `max(0, k * pen + d *
/// closing_speed)`, and the tangential force is Coulomb friction
/// `-mu * N * v_t / max(|v_t|, eps)`. That form is bounded by `mu * N`
/// (the Coulomb cap) and reduces to a viscous stick of slope `mu * N / eps`
/// below the regularization speed `eps`, which keeps the explicit integrator
/// smooth at zero slip. Limits: this is a soft-contact penalty model, so it
/// penetrates by `N / k` at rest and its stability depends on the timestep
/// (roughly `omega * dt < 1` with `omega = sqrt(k / m_eff)`). It is not a
/// constraint solver and does not reproduce hard impacts exactly.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct GroundContact {
    /// Normal spring stiffness (N/m of penetration).
    pub stiffness: f64,
    /// Normal damping (N per m/s of closing velocity).
    pub normal_damping: f64,
    /// Coulomb friction coefficient.
    pub friction: f64,
    /// Tangential speed (m/s) below which friction is regularized to a
    /// viscous stick.
    pub friction_regularization: f64,
    /// Height of the ground plane (world Z).
    pub ground_height: f64,
}

impl GroundContact {
    /// Whether every parameter is finite, with positive stiffness and
    /// regularization speed and non-negative damping and friction.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.stiffness.is_finite()
            && self.stiffness > 0.0
            && self.normal_damping.is_finite()
            && self.normal_damping >= 0.0
            && self.friction.is_finite()
            && self.friction >= 0.0
            && self.friction_regularization.is_finite()
            && self.friction_regularization > 0.0
            && self.ground_height.is_finite()
    }

    /// The world-frame force on a contact sphere centred at `center` with
    /// radius `radius`, moving with world-frame point velocity `velocity`.
    /// Zero whenever the sphere is not penetrating the plane.
    #[must_use]
    pub fn force(&self, center: Vec3, radius: f64, velocity: Vec3) -> Vec3 {
        let penetration = self.ground_height + radius - center.z;
        if penetration <= 0.0 {
            return Vec3::ZERO;
        }
        let closing_speed = -velocity.z;
        let normal = (self.stiffness * penetration + self.normal_damping * closing_speed).max(0.0);
        let tangential = Vec3::new(velocity.x, velocity.y, 0.0);
        let speed = tangential.norm().max(self.friction_regularization);
        let friction = tangential * (-(self.friction * normal) / speed);
        Vec3::new(friction.x, friction.y, normal)
    }
}

/// Why a plane-sphere contact query was rejected.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PlaneSphereError {
    /// An input is NaN or infinite.
    NonFinite,
    /// The sphere radius is negative.
    NegativeRadius,
}

impl fmt::Display for PlaneSphereError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite => write!(f, "plane-sphere contact input is non-finite"),
            Self::NegativeRadius => write!(f, "plane-sphere contact radius is negative"),
        }
    }
}

impl core::error::Error for PlaneSphereError {}

/// A plane-sphere contact in the reference's convention: `dist` is the signed
/// separation (negative when penetrating), `normal` points from the plane
/// toward the sphere, and `pos` is the contact point midway between the
/// plane and the sphere surface.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct PlaneSphereContact {
    /// Signed distance between the surfaces (`con->dist`).
    pub dist: f64,
    /// Contact normal, unit length (`con->normal`).
    pub normal: [f64; 3],
    /// Contact position (`con->pos`).
    pub pos: [f64; 3],
}

/// Plane-sphere contact for a horizontal plane `z = plane_height` with normal
/// `+z`. Source: `PlaneSphere` (static) and its public wrapper
/// `PlaneSphere`, the reference formulation,
/// `src/engine/engine_collision_primitive.c`, lines 28-49 and 53-62.
///
/// The source takes `normal` from the plane's rotation matrix column `mat1[2,5,8]`,
/// computes `cdist = dot(pos2 - pos1, normal)`, rejects the pair when
/// `cdist > margin + radius`, and otherwise sets `dist = cdist - radius` and
/// `pos = pos2 - normal * (dist / 2 + radius)`, where `pos2` is the sphere
/// centre. This port fixes `normal = +z` (the identity-frame plane) and
/// returns `pos` without the `tangent` field the source also zeroes, because
/// the contact struct here has no tangent.
///
/// For a plane at `plane_height` the contact point sits at height
/// `center.z - radius - dist / 2`, midway between the plane and the sphere's
/// lowest point when the sphere touches or penetrates.
///
/// Returns `Ok(None)` when the sphere is farther than `margin` from the plane.
/// Only the plane-sphere geometry is ported. The penalty force stays in
/// [`GroundContact::force`], which is the baseline this function does not change.
pub fn plane_sphere_contact(
    center: [f64; 3],
    radius: f64,
    plane_height: f64,
    margin: f64,
) -> Result<Option<PlaneSphereContact>, PlaneSphereError> {
    if !center.iter().all(|x| x.is_finite())
        || !radius.is_finite()
        || !plane_height.is_finite()
        || !margin.is_finite()
    {
        return Err(PlaneSphereError::NonFinite);
    }
    if radius < 0.0 {
        return Err(PlaneSphereError::NegativeRadius);
    }
    let normal = [0.0, 0.0, 1.0];
    let cdist = center[2] - plane_height;
    if cdist > margin + radius {
        return Ok(None);
    }
    let dist = cdist - radius;
    let offset = -dist / 2.0 - radius;
    let pos = [
        center[0] + normal[0] * offset,
        center[1] + normal[1] * offset,
        center[2] + normal[2] * offset,
    ];
    Ok(Some(PlaneSphereContact { dist, normal, pos }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> ContactParams {
        ContactParams {
            mass: 1.0,
            gravity: 9.80665,
            stiffness: 20_000.0,
            normal_damping: 50.0,
            friction: 0.6,
            ground_height: 0.0,
        }
    }

    #[test]
    fn a_resting_body_on_the_ground_has_bounded_penetration() {
        let mut body = ContactBody::new(
            params(),
            ContactState {
                position: Vec3::new(0.0, 0.0, 0.001),
                velocity: Vec3::ZERO,
            },
        );
        let dt = Duration::from_micros(200);
        for _ in 0..5000 {
            body.step(Vec3::ZERO, dt).unwrap();
        }
        // At steady state the spring force balances gravity: penetration ≈
        // mass*gravity/stiffness.
        let expected = params().mass * params().gravity / params().stiffness;
        assert!(
            (body.penetration() - expected).abs() < expected * 0.5 + 1e-4,
            "expected penetration near {expected}, got {}",
            body.penetration()
        );
        assert!(body.state().position.z.is_finite());
    }

    #[test]
    fn a_falling_body_loses_energy_on_impact() {
        let mut body = ContactBody::new(
            params(),
            ContactState {
                position: Vec3::new(0.0, 0.0, 1.0),
                velocity: Vec3::ZERO,
            },
        );
        let e0 = body.energy();
        let dt = Duration::from_micros(100);
        // Fall, bounce, and settle; run long enough to pass through contact.
        for _ in 0..20_000 {
            body.step(Vec3::ZERO, dt).unwrap();
        }
        let e_final = body.energy();
        assert!(
            e_final < e0,
            "a lossy contact model must remove energy: {e_final} vs {e0}"
        );
        assert!(
            body.max_penetration() > 0.0,
            "the body must have actually contacted the ground"
        );
    }

    #[test]
    fn friction_resists_but_does_not_reverse_sliding() {
        let mut body = ContactBody::new(
            params(),
            ContactState {
                position: Vec3::new(0.0, 0.0, 0.001),
                velocity: Vec3::new(2.0, 0.0, 0.0),
            },
        );
        let dt = Duration::from_micros(200);
        let mut previous_vx = body.state().velocity.x;
        for _ in 0..2000 {
            body.step(Vec3::ZERO, dt).unwrap();
            let vx = body.state().velocity.x;
            assert!(
                vx <= previous_vx + 1e-6,
                "friction must not accelerate the slide"
            );
            assert!(
                vx >= -1e-6,
                "friction must not reverse the slide's direction"
            );
            previous_vx = vx;
        }
        assert!(
            previous_vx.abs() < 1.0,
            "sliding should have slowed substantially: {previous_vx}"
        );
    }

    #[test]
    fn a_push_recovers_after_the_force_is_removed() {
        let mut body = ContactBody::new(
            params(),
            ContactState {
                position: Vec3::new(0.0, 0.0, 0.001),
                velocity: Vec3::ZERO,
            },
        );
        let dt = Duration::from_micros(200);
        // Push sideways briefly.
        for _ in 0..50 {
            body.step(Vec3::new(10.0, 0.0, 0.0), dt).unwrap();
        }
        let pushed_speed = body.state().velocity.x;
        assert!(
            pushed_speed > 0.0,
            "the push should have imparted sideways velocity"
        );
        // Remove the push; friction should bring it back to rest.
        for _ in 0..5000 {
            body.step(Vec3::ZERO, dt).unwrap();
        }
        assert!(
            body.state().velocity.norm() < 0.05,
            "should settle after the push is removed"
        );
    }

    #[test]
    fn nonfinite_or_nonpositive_dt_is_rejected() {
        let mut body = ContactBody::new(
            params(),
            ContactState {
                position: Vec3::ZERO,
                velocity: Vec3::ZERO,
            },
        );
        assert!(body.step(Vec3::ZERO, Duration::ZERO).is_err());
    }

    fn ground() -> GroundContact {
        GroundContact {
            stiffness: 20_000.0,
            normal_damping: 50.0,
            friction: 0.6,
            friction_regularization: 1e-3,
            ground_height: 0.0,
        }
    }

    #[test]
    fn ground_force_is_zero_when_the_sphere_is_separated() {
        let g = ground();
        let f = g.force(Vec3::new(0.0, 0.0, 0.2), 0.1, Vec3::new(1.0, 0.0, -3.0));
        assert_eq!(f, Vec3::ZERO);
    }

    #[test]
    fn ground_normal_force_is_clamped_to_zero_during_fast_separation() {
        let g = ground();
        // Barely penetrating but moving up quickly: the damping term would
        // be negative, and the clamp must keep the normal force at zero.
        let f = g.force(
            Vec3::new(0.0, 0.0, 0.1 - 1e-4),
            0.1,
            Vec3::new(0.0, 0.0, 5.0),
        );
        assert_eq!(f.z, 0.0);
        assert_eq!(f.x, 0.0);
    }

    #[test]
    fn ground_friction_never_exceeds_the_coulomb_bound() {
        let g = ground();
        let center = Vec3::new(0.0, 0.0, 0.1 - 1e-3);
        for &vx in &[1e-6, 1e-3, 0.1, 1.0, 10.0] {
            let f = g.force(center, 0.1, Vec3::new(vx, 0.0, 0.0));
            let tangential = (f.x * f.x + f.y * f.y).sqrt();
            assert!(f.z > 0.0);
            assert!(
                tangential <= g.friction * f.z * (1.0 + 1e-12),
                "vx={vx}: |F_t|={tangential} exceeds mu*N={}",
                g.friction * f.z
            );
            assert!(f.x < 0.0, "friction must oppose the sliding direction");
        }
    }

    #[test]
    fn ground_contact_validation_rejects_invalid_parameters() {
        assert!(ground().is_valid());
        let bad = [
            GroundContact {
                stiffness: 0.0,
                ..ground()
            },
            GroundContact {
                stiffness: f64::NAN,
                ..ground()
            },
            GroundContact {
                normal_damping: -1.0,
                ..ground()
            },
            GroundContact {
                friction: -0.1,
                ..ground()
            },
            GroundContact {
                friction_regularization: 0.0,
                ..ground()
            },
            GroundContact {
                friction_regularization: f64::INFINITY,
                ..ground()
            },
            GroundContact {
                ground_height: f64::NAN,
                ..ground()
            },
        ];
        for g in bad {
            assert!(!g.is_valid(), "{g:?} should be invalid");
        }
    }

    #[test]
    fn plane_sphere_penetration_matches_the_hand_derived_values() {
        // Hand derivation: plane z = 0, sphere centre (0.3, -0.2, 0.08), radius 0.1,
        // margin 0. cdist = 0.08, so dist = 0.08 - 0.1 = -0.02. The position offset
        // is -dist/2 - radius = 0.01 - 0.1 = -0.09, so pos.z = 0.08 - 0.09 = -0.01,
        // which is the midpoint of the plane (0) and the sphere's lowest point (-0.02).
        // x and y are unchanged because the normal is +z.
        let c = plane_sphere_contact([0.3, -0.2, 0.08], 0.1, 0.0, 0.0)
            .unwrap()
            .unwrap();
        assert!((c.dist + 0.02).abs() < 1e-15, "dist = {}", c.dist);
        assert_eq!(c.normal, [0.0, 0.0, 1.0]);
        assert!((c.pos[0] - 0.3).abs() < 1e-15 && (c.pos[1] + 0.2).abs() < 1e-15);
        assert!((c.pos[2] + 0.01).abs() < 1e-15, "pos.z = {}", c.pos[2]);
    }

    #[test]
    fn plane_sphere_margin_admits_a_separated_sphere() {
        // Hand derivation: centre height 0.15, radius 0.1, margin 0.1. cdist = 0.15 is
        // not greater than margin + radius = 0.2, so the pair is reported with
        // dist = 0.15 - 0.1 = 0.05 > 0 (separated, inside the margin).
        let c = plane_sphere_contact([0.0, 0.0, 0.15], 0.1, 0.0, 0.1)
            .unwrap()
            .unwrap();
        assert!((c.dist - 0.05).abs() < 1e-15, "dist = {}", c.dist);
    }

    #[test]
    fn plane_sphere_beyond_the_margin_is_no_contact() {
        // Hand derivation: centre height 0.25, radius 0.1, margin 0. cdist = 0.25
        // exceeds margin + radius = 0.1, so no contact is returned.
        assert_eq!(
            plane_sphere_contact([0.0, 0.0, 0.25], 0.1, 0.0, 0.0),
            Ok(None)
        );
    }

    #[test]
    fn plane_sphere_refuses_invalid_inputs() {
        assert_eq!(
            plane_sphere_contact([0.0, 0.0, f64::NAN], 0.1, 0.0, 0.0),
            Err(PlaneSphereError::NonFinite)
        );
        assert_eq!(
            plane_sphere_contact([0.0, 0.0, 0.0], -0.1, 0.0, 0.0),
            Err(PlaneSphereError::NegativeRadius)
        );
    }
}

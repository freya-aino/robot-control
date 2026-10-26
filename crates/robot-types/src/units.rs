//! SI quantities as newtypes.
//!
//! SPECS.md Appendix A.3 fixes SI units throughout and requires stored angles in
//! radians regardless of the source convention. These newtypes exist so a
//! degrees-valued float cannot be assigned to a radians-valued field without an
//! explicit conversion that the provenance record can name.
//!
//! The arithmetic is deliberately narrow: a quantity may be added to and
//! subtracted from its own kind and scaled by a dimensionless factor. Products
//! that change dimension (force times distance, for example) are written out as
//! named constructors so the resulting dimension is visible at the call site.

use core::fmt;
use core::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

/// Declares a dimensioned `f64` newtype with its canonical unit symbol.
macro_rules! quantity {
    ($(#[$meta:meta])* $name:ident, $symbol:literal) => {
        $(#[$meta])*
        #[derive(Clone, Copy, PartialEq, PartialOrd, Default)]
        pub struct $name(pub f64);

        impl $name {
            /// The canonical unit symbol, as written in records and reports.
            pub const SYMBOL: &'static str = $symbol;
            /// Zero in the canonical unit.
            pub const ZERO: Self = Self(0.0);

            /// The underlying value in the canonical unit.
            #[inline]
            #[must_use]
            pub const fn value(self) -> f64 {
                self.0
            }

            /// True when the value is finite; records reject non-finite samples.
            #[inline]
            #[must_use]
            pub fn is_finite(self) -> bool {
                self.0.is_finite()
            }

            /// Absolute value in the canonical unit.
            #[inline]
            #[must_use]
            pub fn abs(self) -> Self {
                Self(self.0.abs())
            }

            /// Clamps into `[lo, hi]`. Panics in debug builds if `lo > hi`.
            #[inline]
            #[must_use]
            pub fn clamp(self, lo: Self, hi: Self) -> Self {
                debug_assert!(lo.0 <= hi.0, "clamp bounds inverted");
                Self(self.0.clamp(lo.0, hi.0))
            }

            /// Linear interpolation, `t` dimensionless and unclamped.
            #[inline]
            #[must_use]
            pub fn lerp(self, other: Self, t: f64) -> Self {
                Self(self.0 + (other.0 - self.0) * t)
            }

            /// Largest of two values.
            #[inline]
            #[must_use]
            pub fn max(self, other: Self) -> Self {
                Self(self.0.max(other.0))
            }

            /// Smallest of two values.
            #[inline]
            #[must_use]
            pub fn min(self, other: Self) -> Self {
                Self(self.0.min(other.0))
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{} {}", self.0, $symbol)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{} {}", self.0, $symbol)
            }
        }

        impl Add for $name {
            type Output = Self;
            #[inline]
            fn add(self, rhs: Self) -> Self {
                Self(self.0 + rhs.0)
            }
        }

        impl AddAssign for $name {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                self.0 += rhs.0;
            }
        }

        impl Sub for $name {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: Self) -> Self {
                Self(self.0 - rhs.0)
            }
        }

        impl SubAssign for $name {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                self.0 -= rhs.0;
            }
        }

        impl Neg for $name {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self {
                Self(-self.0)
            }
        }

        impl Mul<f64> for $name {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: f64) -> Self {
                Self(self.0 * rhs)
            }
        }

        impl Mul<$name> for f64 {
            type Output = $name;
            #[inline]
            fn mul(self, rhs: $name) -> $name {
                $name(self * rhs.0)
            }
        }

        impl Div<f64> for $name {
            type Output = Self;
            #[inline]
            fn div(self, rhs: f64) -> Self {
                Self(self.0 / rhs)
            }
        }

        /// Ratio of two like quantities is dimensionless.
        impl Div for $name {
            type Output = f64;
            #[inline]
            fn div(self, rhs: Self) -> f64 {
                self.0 / rhs.0
            }
        }
    };
}

quantity!(
    /// Length in metres.
    Metres,
    "m"
);
quantity!(
    /// Plane angle in radians. Stored records are always radians (Appendix A.3).
    Radians,
    "rad"
);
quantity!(
    /// Mass in kilograms.
    Kilograms,
    "kg"
);
quantity!(
    /// Force in newtons.
    Newtons,
    "N"
);
quantity!(
    /// Torque in newton-metres.
    NewtonMetres,
    "N*m"
);
quantity!(
    /// Electric current in amperes.
    Amperes,
    "A"
);
quantity!(
    /// Temperature in degrees Celsius. Recorded explicitly, per Appendix A.3.
    Celsius,
    "degC"
);
quantity!(
    /// Linear speed in metres per second.
    MetresPerSecond,
    "m/s"
);
quantity!(
    /// Angular speed in radians per second.
    RadiansPerSecond,
    "rad/s"
);
quantity!(
    /// Angular acceleration in radians per second squared.
    RadiansPerSecondSquared,
    "rad/s^2"
);
quantity!(
    /// Linear acceleration in metres per second squared.
    MetresPerSecondSquared,
    "m/s^2"
);
quantity!(
    /// Rate in hertz. Tier rates of Section 5.1 are expressed here.
    Hertz,
    "Hz"
);
quantity!(
    /// Energy in joules; the contact energy budget of Section 12.2 is in these.
    Joules,
    "J"
);
quantity!(
    /// Electrical power in watts.
    Watts,
    "W"
);

impl Radians {
    /// Converts from degrees, which only source formats use.
    ///
    /// The caller is responsible for recording the source convention in the
    /// provenance of the record being built (SPECS.md Appendix A.3).
    #[inline]
    #[must_use]
    pub fn from_degrees(degrees: f64) -> Self {
        Self(degrees * core::f64::consts::PI / 180.0)
    }

    /// Converts to degrees for human-facing output only.
    #[inline]
    #[must_use]
    pub fn to_degrees(self) -> f64 {
        self.0 * 180.0 / core::f64::consts::PI
    }

    /// Wraps into `(-pi, pi]`.
    #[inline]
    #[must_use]
    pub fn wrapped(self) -> Self {
        let two_pi = core::f64::consts::TAU;
        let mut v = self.0 % two_pi;
        if v > core::f64::consts::PI {
            v -= two_pi;
        } else if v <= -core::f64::consts::PI {
            v += two_pi;
        }
        Self(v)
    }
}

impl Hertz {
    /// The period of this rate.
    ///
    /// Returns `None` for a non-positive or non-finite rate, because a tier
    /// deadline cannot be derived from one (SPECS.md Section 5.1 item 1).
    #[must_use]
    pub fn period(self) -> Option<crate::time::Duration> {
        if self.0 > 0.0 && self.0.is_finite() {
            Some(crate::time::Duration::from_secs_f64(1.0 / self.0))
        } else {
            None
        }
    }
}

/// A closed inclusive interval in one quantity, used for every limit in a
/// robot profile and every envelope the supervisor enforces.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Range<T> {
    /// Inclusive lower bound.
    pub lo: T,
    /// Inclusive upper bound.
    pub hi: T,
}

macro_rules! range_impl {
    ($name:ident) => {
        impl Range<$name> {
            /// Builds a range, rejecting inverted or non-finite bounds.
            pub fn new(lo: $name, hi: $name) -> Result<Self, &'static str> {
                if !lo.is_finite() || !hi.is_finite() {
                    return Err("range bound is not finite");
                }
                if lo.0 > hi.0 {
                    return Err("range lower bound exceeds upper bound");
                }
                Ok(Self { lo, hi })
            }

            /// Symmetric range `[-magnitude, +magnitude]`.
            pub fn symmetric(magnitude: $name) -> Result<Self, &'static str> {
                Self::new(-magnitude.abs(), magnitude.abs())
            }

            /// True when `value` lies inside the closed interval.
            #[inline]
            #[must_use]
            pub fn contains(&self, value: $name) -> bool {
                value.0 >= self.lo.0 && value.0 <= self.hi.0
            }

            /// Clamps `value` into the interval.
            #[inline]
            #[must_use]
            pub fn clamp(&self, value: $name) -> $name {
                value.clamp(self.lo, self.hi)
            }

            /// Width of the interval.
            #[inline]
            #[must_use]
            pub fn width(&self) -> $name {
                self.hi - self.lo
            }

            /// Midpoint of the interval.
            #[inline]
            #[must_use]
            pub fn mid(&self) -> $name {
                $name((self.lo.0 + self.hi.0) * 0.5)
            }

            /// Signed distance outside the interval: negative below `lo`,
            /// positive above `hi`, zero inside. The supervisor reports this so
            /// a rejection record says how far out of bounds a command was.
            #[inline]
            #[must_use]
            pub fn excess(&self, value: $name) -> $name {
                if value.0 < self.lo.0 {
                    value - self.lo
                } else if value.0 > self.hi.0 {
                    value - self.hi
                } else {
                    $name::ZERO
                }
            }
        }
    };
}

range_impl!(Metres);
range_impl!(Radians);
range_impl!(Kilograms);
range_impl!(Newtons);
range_impl!(NewtonMetres);
range_impl!(Amperes);
range_impl!(Celsius);
range_impl!(MetresPerSecond);
range_impl!(RadiansPerSecond);
range_impl!(Hertz);
range_impl!(Joules);
range_impl!(Watts);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn like_quantities_add_and_scale() {
        assert_eq!(Metres(1.5) + Metres(0.5), Metres(2.0));
        assert_eq!(Metres(2.0) * 0.5, Metres(1.0));
        assert_eq!(2.0 * Metres(2.0), Metres(4.0));
        assert_eq!(Metres(4.0) / Metres(2.0), 2.0);
    }

    #[test]
    fn degrees_convert_and_wrap() {
        let r = Radians::from_degrees(180.0);
        assert!((r.value() - core::f64::consts::PI).abs() < 1e-12);
        assert!(
            (Radians(3.0 * core::f64::consts::PI).wrapped().value() - core::f64::consts::PI).abs()
                < 1e-12
        );
        assert!((Radians::from_degrees(-90.0).to_degrees() + 90.0).abs() < 1e-12);
    }

    #[test]
    fn range_rejects_inverted_and_nonfinite_bounds() {
        assert!(Range::<Metres>::new(Metres(1.0), Metres(0.0)).is_err());
        assert!(Range::<Metres>::new(Metres(f64::NAN), Metres(0.0)).is_err());
        let r = Range::<Metres>::new(Metres(-1.0), Metres(1.0)).unwrap();
        assert!(r.contains(Metres(0.0)));
        assert!(!r.contains(Metres(1.5)));
        assert_eq!(r.clamp(Metres(1.5)), Metres(1.0));
        assert_eq!(r.excess(Metres(1.5)), Metres(0.5));
        assert_eq!(r.excess(Metres(-2.0)), Metres(-1.0));
        assert_eq!(r.excess(Metres(0.0)), Metres(0.0));
    }

    #[test]
    fn rate_period_rejects_nonpositive() {
        assert!(Hertz(0.0).period().is_none());
        assert!(Hertz(-1.0).period().is_none());
        let p = Hertz(100.0).period().unwrap();
        assert_eq!(p.nanos(), 10_000_000);
    }
}

//! Stable typed identifiers for joints, links, actuators and sensors.
//!
//! SPECS.md Section 7.1: "Every profile MUST carry a `schema_version` and typed,
//! stable IDs for joints, links, actuators and sensors. IDs MUST NOT be reused
//! for a different entity within a profile lineage."
//!
//! The types below are distinct so that a joint index cannot be passed where an
//! actuator index is expected. They are *not* interchangeable even when their
//! numeric values agree, because a robot with a geared or coupled transmission
//! has a different count of each (SPECS.md Section 7.1, motor transmission).

use core::fmt;

/// Declares a stable, typed, non-reusable identifier.
macro_rules! typed_id {
    ($(#[$meta:meta])* $name:ident, $prefix:literal) => {
        $(#[$meta])*
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(u32);

        impl $name {
            /// Wraps a raw index.
            ///
            /// Assignment of indices is the robot profile's responsibility; this
            /// constructor does not validate that the entity exists.
            #[inline]
            #[must_use]
            pub const fn new(index: u32) -> Self {
                Self(index)
            }

            /// The raw index.
            #[inline]
            #[must_use]
            pub const fn index(self) -> u32 {
                self.0
            }

            /// The index as a `usize` for array addressing.
            #[inline]
            #[must_use]
            pub const fn as_usize(self) -> usize {
                self.0 as usize
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}{}", $prefix, self.0)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}{}", $prefix, self.0)
            }
        }
    };
}

typed_id!(
    /// A joint of a robot profile.
    JointId,
    "joint#"
);
typed_id!(
    /// A link of a robot profile.
    LinkId,
    "link#"
);
typed_id!(
    /// An actuator of a robot profile. Not interchangeable with [`JointId`]:
    /// a transmission may drive several joints from one actuator or vice versa.
    ActuatorId,
    "actuator#"
);
typed_id!(
    /// A sensor of a robot profile.
    SensorId,
    "sensor#"
);
typed_id!(
    /// A contact surface or site declared by a robot profile.
    ContactSiteId,
    "site#"
);
typed_id!(
    /// A parallel environment inside a batched simulator.
    ///
    /// SPECS.md Section 7.6 item 4 requires stable environment IDs with
    /// independent RNG streams; the ID is the key for that stream.
    EnvId,
    "env#"
);

/// A short, stable, human-meaningful name.
///
/// Names are the key used across files; indices are the key used in memory. A
/// profile owns the mapping between them, and the constraints below exist so a
/// name cannot silently carry whitespace, case variants or path separators that
/// make two records disagree about the same entity.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Name(String);

/// Why a name was rejected.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NameError {
    /// The name was empty.
    Empty,
    /// The name exceeded [`Name::MAX_LEN`] bytes.
    TooLong {
        /// Length supplied.
        len: usize,
    },
    /// The name contained a character outside the permitted set.
    InvalidCharacter {
        /// The offending character.
        found: char,
    },
}

impl fmt::Display for NameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "name is empty"),
            Self::TooLong { len } => {
                write!(f, "name is {len} bytes, limit is {}", Name::MAX_LEN)
            }
            Self::InvalidCharacter { found } => {
                write!(f, "name contains {found:?}; permitted: a-z A-Z 0-9 _ - .")
            }
        }
    }
}

impl core::error::Error for NameError {}

impl Name {
    /// Maximum length in bytes. Bounded because names appear in fixed-capacity
    /// records on the control path (SPECS.md Section 5).
    pub const MAX_LEN: usize = 64;

    /// Validates and wraps a name.
    pub fn new(text: &str) -> Result<Self, NameError> {
        if text.is_empty() {
            return Err(NameError::Empty);
        }
        if text.len() > Self::MAX_LEN {
            return Err(NameError::TooLong { len: text.len() });
        }
        if let Some(bad) = text
            .chars()
            .find(|c| !(c.is_ascii_alphanumeric() || *c == '_' || *c == '-' || *c == '.'))
        {
            return Err(NameError::InvalidCharacter { found: bad });
        }
        Ok(Self(text.to_owned()))
    }

    /// The name as a string slice.
    #[inline]
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl fmt::Display for Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl core::str::FromStr for Name {
    type Err = NameError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

/// A namespaced key for the extension tables of SPECS.md Section 7.1.
///
/// "additional declarative metadata belongs in open, namespaced extension tables
/// with declared value types and capability tags". The namespace is what lets
/// non-control tooling preserve an extension it does not understand without
/// guessing whether the key is a core field.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExtensionKey {
    namespace: Name,
    key: Name,
}

impl ExtensionKey {
    /// Builds a key from a namespace and a leaf name.
    pub fn new(namespace: &str, key: &str) -> Result<Self, NameError> {
        Ok(Self {
            namespace: Name::new(namespace)?,
            key: Name::new(key)?,
        })
    }

    /// Parses `namespace:key`.
    pub fn parse(text: &str) -> Result<Self, NameError> {
        match text.split_once(':') {
            Some((ns, k)) => Self::new(ns, k),
            None => Err(NameError::InvalidCharacter { found: ':' }),
        }
    }

    /// The namespace.
    #[inline]
    #[must_use]
    pub fn namespace(&self) -> &Name {
        &self.namespace
    }

    /// The leaf key.
    #[inline]
    #[must_use]
    pub fn key(&self) -> &Name {
        &self.key
    }
}

impl fmt::Display for ExtensionKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.namespace, self.key)
    }
}

impl fmt::Debug for ExtensionKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_types_do_not_interchange() {
        // This is a compile-time property; the runtime check is that the
        // Display prefixes differ so a record is unambiguous.
        assert_eq!(JointId::new(3).to_string(), "joint#3");
        assert_eq!(ActuatorId::new(3).to_string(), "actuator#3");
        assert_eq!(EnvId::new(0).as_usize(), 0);
    }

    #[test]
    fn names_reject_separators_and_whitespace() {
        assert!(Name::new("left_hip_pitch").is_ok());
        assert!(Name::new("imu.0-raw").is_ok());
        assert_eq!(Name::new(""), Err(NameError::Empty));
        assert_eq!(
            Name::new("left hip"),
            Err(NameError::InvalidCharacter { found: ' ' })
        );
        assert_eq!(
            Name::new("a/b"),
            Err(NameError::InvalidCharacter { found: '/' })
        );
        assert!(matches!(
            Name::new(&"x".repeat(Name::MAX_LEN + 1)),
            Err(NameError::TooLong { .. })
        ));
    }

    #[test]
    fn extension_keys_require_a_namespace() {
        let k = ExtensionKey::parse("vendor.acme:thermal_curve").unwrap();
        assert_eq!(k.namespace().as_str(), "vendor.acme");
        assert_eq!(k.key().as_str(), "thermal_curve");
        assert_eq!(k.to_string(), "vendor.acme:thermal_curve");
        assert!(ExtensionKey::parse("no_namespace").is_err());
    }
}

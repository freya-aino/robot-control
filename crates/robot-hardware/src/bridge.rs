//! The pluggable robot bridge of ADR 0025 (C-02 direction): a new robot is
//! a **configuration artifact**, not code.
//!
//! A `robot-bridge/1` file names the robot it serves, maps every actuated
//! joint of that robot's profile onto a wire index, and declares the link
//! timing it expects. Loading it against the profile validates the whole
//! mapping before any transport exists: every actuated joint mapped exactly
//! once, indices dense, the command validity agreeing with the profile's
//! own firmware contract — a mismatch is refused by name, never averaged.
//! The validated bridge then builds the existing [`crate::link::WireAdapter`]
//! and permutes values between the profile's joint order and the wire's,
//! so `robot-control` never learns a robot-specific ordering.
//!
//! The protocol stays `robot-wire/1`: this module adds no transport and no
//! vendor code. For a robot whose firmware speaks something else (the G1's
//! own low-level interface is DDS, see the committed
//! `assets/bridges/unitree_g1.toml` and its evidence-ledger row), the
//! bridge file records that surface and the adapter shim that would
//! translate it remains firmware-side work, outside this crate's closure.

use core::fmt;
use robot_artifacts::toml::{self, Table, TomlError};
use robot_model::profile::RobotProfile;
use robot_types::time::{Duration, MonotonicClock};

/// The format this module reads.
pub const FORMAT: &str = "robot-bridge/1";

/// Every key a `robot-bridge/1` file may contain at the root.
pub const KNOWN_KEYS: &[&str] = &[
    "format",
    "robot",
    "protocol",
    "endpoint",
    "command_validity_us",
    "delay_bound_us",
    "interface_source",
    "joints",
];

/// Why a bridge file was refused.
#[derive(Clone, PartialEq, Debug)]
pub enum BridgeError {
    /// The file did not parse, or a key is missing or mistyped.
    Parse(String),
    /// The `format` is not `robot-bridge/1`.
    WrongFormat(String),
    /// A root key outside [`KNOWN_KEYS`].
    UnknownKey(String),
    /// The `protocol` is not `robot-wire/1`, the only committed protocol.
    UnknownProtocol(String),
    /// The profile has an actuated joint the mapping does not name.
    UnmappedJoint(String),
    /// The mapping names a joint the profile does not have.
    UnknownJoint(String),
    /// Two joints map to the same wire index, or an index is out of range.
    BadIndex {
        /// The joint whose index is wrong.
        joint: String,
        /// The index found.
        index: i64,
    },
    /// The declared command validity disagrees with the profile's firmware
    /// contract. A profile that declares a positive expiration is the
    /// authority and the bridge must match it; a profile whose contract is
    /// undeclared (zero) leaves the bridge's own declaration standing.
    ValidityMismatch {
        /// Microseconds the bridge declares.
        bridge_us: i64,
        /// Microseconds the profile's `command_expiration` declares.
        profile_us: i64,
    },
    /// A declared duration is not positive.
    BadDuration(&'static str),
}

impl fmt::Display for BridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "bridge file: {e}"),
            Self::WrongFormat(found) => {
                write!(f, "bridge format is {found:?}, not {FORMAT:?}")
            }
            Self::UnknownKey(key) => write!(f, "bridge file has unknown key `{key}`"),
            Self::UnknownProtocol(p) => {
                write!(f, "bridge protocol {p:?} is not \"robot-wire/1\"")
            }
            Self::UnmappedJoint(j) => {
                write!(f, "actuated joint {j:?} has no wire index in the bridge")
            }
            Self::UnknownJoint(j) => {
                write!(
                    f,
                    "bridge maps joint {j:?}, which the profile does not declare"
                )
            }
            Self::BadIndex { joint, index } => write!(
                f,
                "joint {joint:?} maps to wire index {index}, which is out of range or taken"
            ),
            Self::ValidityMismatch {
                bridge_us,
                profile_us,
            } => write!(
                f,
                "bridge command validity {bridge_us} us disagrees with the profile's \
                 firmware command_expiration {profile_us} us; the profile is the authority"
            ),
            Self::BadDuration(what) => write!(f, "bridge `{what}` must be positive"),
        }
    }
}

impl core::error::Error for BridgeError {}

/// A validated bridge: the mapping between a profile's actuated-joint order
/// and the wire's index order, plus the link timing.
#[derive(Clone, Debug, PartialEq)]
pub struct BridgeConfig {
    /// The robot family this bridge serves, as the file declares it.
    pub robot: String,
    /// Where the peer lives (free-form; a UDP address once one exists).
    pub endpoint: String,
    /// Where the wire surface was read from, for the evidence trail.
    pub interface_source: String,
    /// Command validity, equal to the profile's `command_expiration`.
    pub command_validity: Duration,
    /// Worst-case one-way transport delay the link budget assumes.
    pub delay_bound: Duration,
    /// `wire_of[profile_actuated_index]` is the wire index.
    wire_of: Vec<usize>,
}

impl BridgeConfig {
    /// Parses a bridge file and validates it against the profile it will
    /// drive.
    ///
    /// # Errors
    ///
    /// [`BridgeError`] naming the first inconsistency.
    pub fn load(text: &str, profile: &RobotProfile) -> Result<Self, BridgeError> {
        let table = toml::parse(text).map_err(|e: TomlError| BridgeError::Parse(e.to_string()))?;
        if let Some(key) = table.unknown_keys(KNOWN_KEYS).into_iter().next() {
            return Err(BridgeError::UnknownKey(key));
        }
        let get = |key: &str| -> Result<String, BridgeError> {
            table
                .require_str(key)
                .map(str::to_owned)
                .map_err(|e| BridgeError::Parse(e.to_string()))
        };
        let format = get("format")?;
        if format != FORMAT {
            return Err(BridgeError::WrongFormat(format));
        }
        let protocol = get("protocol")?;
        if protocol != "robot-wire/1" {
            return Err(BridgeError::UnknownProtocol(protocol));
        }
        let robot = get("robot")?;
        let endpoint = get("endpoint")?;
        let interface_source = get("interface_source")?;
        let int = |key: &'static str| -> Result<i64, BridgeError> {
            table
                .require_integer(key)
                .map_err(|e| BridgeError::Parse(e.to_string()))
        };
        let validity_us = int("command_validity_us")?;
        let delay_us = int("delay_bound_us")?;
        if validity_us <= 0 {
            return Err(BridgeError::BadDuration("command_validity_us"));
        }
        if delay_us <= 0 {
            return Err(BridgeError::BadDuration("delay_bound_us"));
        }
        let profile_us = profile.firmware.command_expiration.nanos() / 1000;
        if profile_us > 0 && validity_us != profile_us {
            return Err(BridgeError::ValidityMismatch {
                bridge_us: validity_us,
                profile_us,
            });
        }

        let joints: &Table = table
            .require_table("joints")
            .map_err(|e| BridgeError::Parse(e.to_string()))?;
        let actuated: Vec<&str> = profile.actuated_joints().map(|j| j.name.as_str()).collect();
        for (key, _) in joints.iter() {
            if !actuated.contains(&key) {
                return Err(BridgeError::UnknownJoint(key.to_owned()));
            }
        }
        let width = actuated.len();
        let mut wire_of = vec![usize::MAX; width];
        let mut taken = vec![false; width];
        for (slot, name) in actuated.iter().enumerate() {
            let index = joints
                .require_integer(name)
                .map_err(|_| BridgeError::UnmappedJoint((*name).to_owned()))?;
            let valid = usize::try_from(index)
                .ok()
                .filter(|i| *i < width && !taken[*i]);
            let Some(index) = valid else {
                return Err(BridgeError::BadIndex {
                    joint: (*name).to_owned(),
                    index,
                });
            };
            taken[index] = true;
            wire_of[slot] = index;
        }

        Ok(Self {
            robot,
            endpoint,
            interface_source,
            command_validity: Duration::from_nanos(validity_us * 1000),
            delay_bound: Duration::from_nanos(delay_us * 1000),
            wire_of,
        })
    }

    /// Number of mapped joints, the wire width.
    #[must_use]
    pub fn width(&self) -> usize {
        self.wire_of.len()
    }

    /// Reorders values from the profile's actuated order into wire order.
    ///
    /// # Panics
    ///
    /// Panics on a width mismatch, which is a caller bug: both vectors are
    /// sized by this bridge's own [`Self::width`].
    pub fn to_wire(&self, profile_order: &[f64], wire_order: &mut [f64]) {
        assert_eq!(profile_order.len(), self.wire_of.len());
        assert_eq!(wire_order.len(), self.wire_of.len());
        for (slot, &index) in self.wire_of.iter().enumerate() {
            wire_order[index] = profile_order[slot];
        }
    }

    /// Reorders values from wire order into the profile's actuated order.
    ///
    /// # Panics
    ///
    /// Panics on a width mismatch, as [`Self::to_wire`] does.
    pub fn from_wire(&self, wire_order: &[f64], profile_order: &mut [f64]) {
        assert_eq!(profile_order.len(), self.wire_of.len());
        assert_eq!(wire_order.len(), self.wire_of.len());
        for (slot, &index) in self.wire_of.iter().enumerate() {
            profile_order[slot] = wire_order[index];
        }
    }

    /// Builds the wire adapter this bridge configures, over any transport
    /// and clock — the loopback and replay devices in tests, a UDP socket
    /// on a host with a robot.
    pub fn adapter<T: crate::link::Transport, C: MonotonicClock>(
        &self,
        transport: T,
        clock: C,
    ) -> crate::link::WireAdapter<T, C> {
        crate::link::WireAdapter::new(
            transport,
            clock,
            self.width(),
            self.command_validity,
            self.delay_bound,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The committed G1 bridge against the vendored USD profile.
    fn g1() -> (String, RobotProfile) {
        let text = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../assets/bridges/unitree_g1.toml"),
        )
        .expect("the committed G1 bridge exists");
        let profile = robot_model::fixtures::unitree_g1_29dof_usd().expect("the vendored G1 loads");
        (text, profile)
    }

    #[test]
    fn the_committed_g1_bridge_validates_against_the_vendored_profile() {
        let (text, profile) = g1();
        let bridge = BridgeConfig::load(&text, &profile).unwrap();
        assert_eq!(bridge.width(), 29, "the G1 declares 29 drives");
        // The USD asset declares no firmware contract (zero expiration), so
        // the bridge's own declaration stands.
        assert_eq!(bridge.command_validity, Duration::from_micros(100_000));
        // The SDK's joint order and the profile's declaration order agree,
        // which the mapping makes explicit rather than assumed: the
        // permutation is the identity.
        let vals: Vec<f64> = (0..29).map(f64::from).collect();
        let mut wire = vec![0.0; 29];
        bridge.to_wire(&vals, &mut wire);
        assert_eq!(wire, vals);
        let mut back = vec![0.0; 29];
        bridge.from_wire(&wire, &mut back);
        assert_eq!(back, vals);
    }

    #[test]
    fn a_command_round_trips_through_the_adapter_and_firmware_peer() {
        use crate::device::HardwareAdapter;
        use crate::link::{FirmwarePeer, MemoryLink};
        use robot_types::time::{ClockId, SteppedClock};
        let (text, profile) = g1();
        let bridge = BridgeConfig::load(&text, &profile).unwrap();
        let (host, device) = MemoryLink::pair();
        let clock = SteppedClock::new(1_000_000, ClockId::HostMonotonic);
        let now = robot_types::time::MonotonicClock::now(&clock);
        let mut adapter = bridge.adapter(host, clock);
        let mut peer = FirmwarePeer::new(device, vec![0.0; bridge.width()], 0.0);
        adapter.set_transmit_enabled(true);
        let command: Vec<f64> = (0..bridge.width()).map(|i| i as f64 * 0.01).collect();
        let mut wire = vec![0.0; bridge.width()];
        bridge.to_wire(&command, &mut wire);
        adapter.send_command(&wire, now).unwrap();
        peer.step(2_000_000, 0.001).unwrap();
        assert_eq!(peer.position, wire, "the peer received the frame");
    }

    #[test]
    fn inconsistencies_are_refused_by_name() {
        let (text, profile) = g1();
        // The USD asset declares no firmware contract, so the mismatch rule
        // needs a profile that does: declare 50 ms and the bridge's 100 ms
        // is refused, while agreeing values pass.
        let mut contracted = profile.clone();
        contracted.firmware.command_expiration = Duration::from_micros(50_000);
        assert!(matches!(
            BridgeConfig::load(&text, &contracted),
            Err(BridgeError::ValidityMismatch {
                bridge_us: 100_000,
                profile_us: 50_000
            })
        ));
        let agreeing = text.replace(
            "command_validity_us = 100000",
            "command_validity_us = 50000",
        );
        assert!(BridgeConfig::load(&agreeing, &contracted).is_ok());
        let missing = text.replace("left_knee_joint = 3\n", "");
        assert!(matches!(
            BridgeConfig::load(&missing, &profile),
            Err(BridgeError::UnmappedJoint(j)) if j == "left_knee_joint"
        ));
        let duplicate = text.replace("left_knee_joint = 3", "left_knee_joint = 4");
        assert!(matches!(
            BridgeConfig::load(&duplicate, &profile),
            Err(BridgeError::BadIndex { .. })
        ));
        let foreign = text.replace(
            "left_knee_joint = 3",
            "left_knee_joint = 3\nnot_a_joint = 99",
        );
        assert!(matches!(
            BridgeConfig::load(&foreign, &profile),
            Err(BridgeError::UnknownJoint(_)) | Err(BridgeError::BadIndex { .. })
        ));
        // Appended text would land inside [joints]; an unknown ROOT key has
        // to come before the table.
        let unknown = format!("extra = 1\n{text}");
        assert!(matches!(
            BridgeConfig::load(&unknown, &profile),
            Err(BridgeError::UnknownKey(_))
        ));
        let bad_protocol = text.replace("robot-wire/1", "dds");
        assert!(matches!(
            BridgeConfig::load(&bad_protocol, &profile),
            Err(BridgeError::UnknownProtocol(_))
        ));
    }
}

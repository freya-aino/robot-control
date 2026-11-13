//! A [`HardwareAdapter`] over [`crate::wire`], and the reference device
//! side of the protocol.
//!
//! [`WireAdapter`] is the host side. It encodes commands into a buffer
//! sized at construction and sends them over a [`Transport`]; it drains
//! feedback without blocking, keeps the newest valid frame, and counts
//! every frame it refuses (CRC failure, stale or repeated sequence number)
//! and every frame lost in transit (sequence gaps).
//!
//! Feedback timestamps are the device's **sampling** instants on the
//! device's clock. The control loop measures ages on the host clock, so the
//! adapter maps them: the host-minus-device offset is estimated as the
//! smallest `arrival - sampling` seen so far (every transport delay is
//! non-negative, so the least-delayed frame bounds the offset most
//! tightly), and the mapped stamp carries the configured transport-delay
//! bound as its uncertainty. SPECS.md Section 5: a timestamp is never
//! "an arrival time silently relabeled as a sampling time".
//!
//! [`FirmwarePeer`] is the device side as this project specifies it: a
//! command is valid for its stated period from receipt, and when no valid
//! command is held the **watchdog** trips and the device holds its current
//! position, reporting [`wire::FLAG_WATCHDOG`] until a valid command
//! arrives. It drives a first-order position follower so tests can close
//! the loop; a real bridge replaces the follower with the vendor's servo.

use std::io;
use std::net::UdpSocket;

use robot_types::time::{ClockId, Duration, MonotonicClock, Timestamp};

use crate::device::{AdapterError, FeedbackHeader, HardwareAdapter};
use crate::wire::{self, Header, Kind};

/// A datagram transport.
pub trait Transport {
    /// Sends one datagram.
    fn send(&mut self, bytes: &[u8]) -> io::Result<()>;
    /// Receives one datagram into `buf` without blocking; `Ok(None)` when
    /// nothing is waiting.
    fn recv(&mut self, buf: &mut [u8]) -> io::Result<Option<usize>>;
}

impl Transport for UdpSocket {
    fn send(&mut self, bytes: &[u8]) -> io::Result<()> {
        UdpSocket::send(self, bytes).map(|_| ())
    }

    fn recv(&mut self, buf: &mut [u8]) -> io::Result<Option<usize>> {
        match UdpSocket::recv(self, buf) {
            Ok(n) => Ok(Some(n)),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => Ok(None),
            Err(e) => Err(e),
        }
    }
}

/// An in-process datagram pair, for tests and simulation.
pub struct MemoryLink {
    outbox: std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<Vec<u8>>>>,
    inbox: std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<Vec<u8>>>>,
}

impl MemoryLink {
    /// Two connected ends.
    #[must_use]
    pub fn pair() -> (Self, Self) {
        let a = std::sync::Arc::default();
        let b = std::sync::Arc::default();
        (
            Self {
                outbox: std::sync::Arc::clone(&a),
                inbox: std::sync::Arc::clone(&b),
            },
            Self {
                outbox: b,
                inbox: a,
            },
        )
    }
}

impl Transport for MemoryLink {
    fn send(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.outbox
            .lock()
            .map_err(|_| io::Error::other("poisoned"))?
            .push_back(bytes.to_vec());
        Ok(())
    }

    fn recv(&mut self, buf: &mut [u8]) -> io::Result<Option<usize>> {
        let Some(d) = self
            .inbox
            .lock()
            .map_err(|_| io::Error::other("poisoned"))?
            .pop_front()
        else {
            return Ok(None);
        };
        let n = d.len().min(buf.len());
        buf[..n].copy_from_slice(&d[..n]);
        Ok(Some(n))
    }
}

/// Link counters.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct LinkStats {
    /// Frames refused for a CRC or format error.
    pub corrupt: u64,
    /// Frames refused as not newer than the last accepted.
    pub stale: u64,
    /// Frames missing from sequence gaps.
    pub lost: u64,
    /// Feedback frames accepted.
    pub accepted: u64,
}

/// The host side of `robot-wire/1`.
pub struct WireAdapter<T: Transport, C: MonotonicClock> {
    transport: T,
    clock: C,
    width: usize,
    tx: Vec<u8>,
    rx: Vec<u8>,
    scratch: Vec<f64>,
    seq: u32,
    command_valid_us: u32,
    delay_bound: Duration,
    transmit: bool,
    last_seq: Option<u32>,
    last_arrival: Option<Timestamp>,
    offset_ns: Option<i64>,
    device_flags: u16,
    /// Counters.
    pub stats: LinkStats,
}

impl<T: Transport, C: MonotonicClock> WireAdapter<T, C> {
    /// An adapter of `width` values whose commands stay valid for
    /// `command_validity`, with `delay_bound` the transport's worst-case
    /// one-way delay. Transmit starts disabled.
    pub fn new(
        transport: T,
        clock: C,
        width: usize,
        command_validity: Duration,
        delay_bound: Duration,
    ) -> Self {
        Self {
            transport,
            clock,
            width,
            tx: vec![0; wire::frame_len(width)],
            rx: vec![0; wire::frame_len(width) + 64],
            scratch: vec![0.0; width],
            seq: 0,
            command_valid_us: u32::try_from(command_validity.nanos() / 1000).unwrap_or(u32::MAX),
            delay_bound,
            transmit: false,
            last_seq: None,
            last_arrival: None,
            offset_ns: None,
            device_flags: 0,
            stats: LinkStats::default(),
        }
    }

    /// The device's flags from the newest accepted feedback.
    #[must_use]
    pub const fn device_flags(&self) -> u16 {
        self.device_flags
    }
}

fn fault(code: &'static str) -> AdapterError {
    AdapterError::TransportFault { code }
}

impl<T: Transport, C: MonotonicClock> HardwareAdapter for WireAdapter<T, C> {
    fn width(&self) -> usize {
        self.width
    }

    fn send_command(&mut self, command: &[f64], timestamp: Timestamp) -> Result<(), AdapterError> {
        if command.len() != self.width {
            return Err(AdapterError::WidthMismatch {
                expected: self.width,
                found: command.len(),
            });
        }
        if !self.transmit {
            return Err(AdapterError::TransmitDisabled);
        }
        self.seq = self.seq.wrapping_add(1);
        let header = Header {
            kind: Kind::Command,
            seq: self.seq,
            time_ns: timestamp.nanos(),
            valid_us: self.command_valid_us,
            flags: 0,
            width: u16::try_from(self.width).map_err(|_| fault("width over u16"))?,
        };
        let n = wire::encode(&header, command, &mut self.tx).map_err(|_| fault("encode"))?;
        self.transport
            .send(&self.tx[..n])
            .map_err(|_| fault("send"))
    }

    fn poll_feedback(
        &mut self,
        positions: &mut [f64],
    ) -> Result<Option<FeedbackHeader>, AdapterError> {
        if positions.len() != self.width {
            return Err(AdapterError::WidthMismatch {
                expected: self.width,
                found: positions.len(),
            });
        }
        let mut newest: Option<(i64, u16, Timestamp)> = None;
        loop {
            let n = match self.transport.recv(&mut self.rx) {
                Ok(Some(n)) => n,
                Ok(None) => break,
                Err(_) => return Err(fault("recv")),
            };
            let arrival = self.clock.now();
            let Ok(h) = wire::decode(&self.rx[..n], &mut self.scratch) else {
                self.stats.corrupt += 1;
                continue;
            };
            if h.kind != Kind::Feedback {
                self.stats.corrupt += 1;
                continue;
            }
            if let Some(last) = self.last_seq {
                let ahead = h.seq.wrapping_sub(last);
                if ahead == 0 || ahead > u32::MAX / 2 {
                    self.stats.stale += 1;
                    continue;
                }
                self.stats.lost += u64::from(ahead - 1);
            }
            self.last_seq = Some(h.seq);
            self.stats.accepted += 1;
            let lag = arrival.nanos() - h.time_ns;
            self.offset_ns = Some(self.offset_ns.map_or(lag, |o| o.min(lag)));
            positions.copy_from_slice(&self.scratch);
            newest = Some((h.time_ns, h.flags, arrival));
        }
        let Some((device_ns, flags, arrival)) = newest else {
            return Ok(None);
        };
        self.device_flags = flags;
        self.last_arrival = Some(arrival);
        let offset = Duration::from_nanos(self.offset_ns.unwrap_or(0));
        let sampling = flags & wire::FLAG_SAMPLING != 0;
        let stamp = if sampling {
            Timestamp::sampling(device_ns, ClockId::Device(0)).mapped_to(
                self.clock.id(),
                offset,
                self.delay_bound,
            )
        } else {
            Timestamp::arrival(arrival.nanos(), self.clock.id(), self.delay_bound)
        };
        Ok(Some(FeedbackHeader {
            timestamp: stamp,
            is_sampling_instant: sampling,
        }))
    }

    fn transmit_enabled(&self) -> bool {
        self.transmit
    }

    fn set_transmit_enabled(&mut self, enabled: bool) {
        self.transmit = enabled;
    }

    fn link_age(&self, now: Timestamp) -> Option<Duration> {
        self.last_arrival
            .and_then(|t| now.since(t).ok())
            .map(|(d, _)| d)
    }
}

/// The reference device side: commands with validity, a watchdog, and a
/// first-order position follower.
pub struct FirmwarePeer<T: Transport> {
    transport: T,
    /// Current positions.
    pub position: Vec<f64>,
    target: Vec<f64>,
    valid_until_ns: Option<i64>,
    last_seq: Option<u32>,
    time_constant: f64,
    seq: u32,
    buf: Vec<u8>,
    /// Whether the watchdog was tripped on the last step.
    pub watchdog_tripped: bool,
}

impl<T: Transport> FirmwarePeer<T> {
    /// A device at `initial` positions, following valid targets with time
    /// constant `time_constant` seconds.
    pub fn new(transport: T, initial: Vec<f64>, time_constant: f64) -> Self {
        let n = initial.len();
        Self {
            transport,
            target: initial.clone(),
            position: initial,
            valid_until_ns: None,
            last_seq: None,
            time_constant,
            seq: 0,
            buf: vec![0; wire::frame_len(n) + 64],
            watchdog_tripped: true,
        }
    }

    /// Advances the device to `now_ns` on its own clock by `dt` seconds:
    /// takes commands, servos or holds, and sends a feedback frame sampled
    /// at `now_ns`.
    pub fn step(&mut self, now_ns: i64, dt: f64) -> io::Result<()> {
        let n = self.position.len();
        let mut values = vec![0.0; n];
        while let Some(len) = self.transport.recv(&mut self.buf)? {
            let Ok(h) = wire::decode(&self.buf[..len], &mut values) else {
                continue;
            };
            if h.kind != Kind::Command
                || self.last_seq.is_some_and(|s| {
                    h.seq.wrapping_sub(s) == 0 || h.seq.wrapping_sub(s) > u32::MAX / 2
                })
            {
                continue;
            }
            self.last_seq = Some(h.seq);
            self.target.copy_from_slice(&values);
            self.valid_until_ns = Some(now_ns + i64::from(h.valid_us) * 1000);
        }
        self.watchdog_tripped = self.valid_until_ns.is_none_or(|t| now_ns > t);
        if !self.watchdog_tripped {
            let a = (dt / self.time_constant).min(1.0);
            for (p, t) in self.position.iter_mut().zip(&self.target) {
                *p += a * (t - *p);
            }
        }
        self.seq = self.seq.wrapping_add(1);
        let mut flags = wire::FLAG_SAMPLING;
        if self.watchdog_tripped {
            flags |= wire::FLAG_WATCHDOG;
        }
        let header = Header {
            kind: Kind::Feedback,
            seq: self.seq,
            time_ns: now_ns,
            valid_us: 0,
            flags,
            width: n as u16,
        };
        let len = wire::encode(&header, &self.position, &mut self.buf)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        self.transport.send(&self.buf[..len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_types::time::SteppedClock;

    fn setup() -> (
        WireAdapter<MemoryLink, SteppedClock>,
        FirmwarePeer<MemoryLink>,
    ) {
        let (host, dev) = MemoryLink::pair();
        let mut a = WireAdapter::new(
            host,
            SteppedClock::new(1_000_000_000, ClockId::HostMonotonic),
            2,
            Duration::from_millis(30),
            Duration::from_millis(2),
        );
        a.set_transmit_enabled(true);
        (a, FirmwarePeer::new(dev, vec![0.0, 0.0], 0.05))
    }

    #[test]
    fn a_command_is_followed_and_fed_back_with_a_mapped_sampling_time() {
        let (mut a, mut d) = setup();
        // A loop commands every tick; each command is valid for 30 ms.
        for k in 0..10 {
            a.send_command(&[1.0, -1.0], a.clock.now()).unwrap();
            d.step(k * 10_000_000, 0.01).unwrap();
        }
        let mut pos = [0.0; 2];
        let h = a.poll_feedback(&mut pos).unwrap().unwrap();
        assert!(pos[0] > 0.5 && pos[1] < -0.5, "{pos:?}");
        assert!(h.is_sampling_instant);
        assert_eq!(h.timestamp.clock(), ClockId::HostMonotonic);
        assert_eq!(h.timestamp.uncertainty(), Duration::from_millis(2));
        assert_eq!(a.stats.accepted, 10);
        assert_eq!(a.device_flags() & wire::FLAG_WATCHDOG, 0);
    }

    #[test]
    fn an_expired_command_trips_the_watchdog_and_the_device_holds() {
        let (mut a, mut d) = setup();
        a.send_command(&[1.0, 1.0], a.clock.now()).unwrap();
        d.step(0, 0.01).unwrap();
        let held = d.position.clone();
        d.step(100_000_000, 0.01).unwrap(); // 100 ms later, past 30 ms validity
        assert!(d.watchdog_tripped);
        assert_eq!(d.position, held);
        let mut pos = [0.0; 2];
        a.poll_feedback(&mut pos).unwrap();
        assert_ne!(a.device_flags() & wire::FLAG_WATCHDOG, 0);
    }

    #[test]
    fn corrupt_stale_and_lost_frames_are_counted() {
        let (mut a, mut d) = setup();
        d.step(0, 0.01).unwrap();
        // Corrupt: inject garbage into the adapter's inbox via the device.
        d.transport.send(&[0x52, 0x57, 1, 2, 0, 0]).unwrap();
        // A gap: the device skips a sequence number.
        d.seq += 1;
        d.step(10_000_000, 0.01).unwrap();
        let mut pos = [0.0; 2];
        a.poll_feedback(&mut pos).unwrap();
        assert_eq!(a.stats.corrupt, 1);
        assert_eq!(a.stats.lost, 1);
        assert_eq!(a.stats.accepted, 2);
    }

    #[test]
    fn transmit_starts_disabled_and_the_width_is_checked() {
        let (host, _) = MemoryLink::pair();
        let mut a = WireAdapter::new(
            host,
            SteppedClock::new(0, ClockId::HostMonotonic),
            2,
            Duration::from_millis(30),
            Duration::ZERO,
        );
        assert_eq!(
            a.send_command(&[0.0, 0.0], a.clock.now()),
            Err(AdapterError::TransmitDisabled)
        );
        a.set_transmit_enabled(true);
        assert!(matches!(
            a.send_command(&[0.0], a.clock.now()),
            Err(AdapterError::WidthMismatch { .. })
        ));
    }

    #[test]
    fn the_protocol_runs_over_udp() {
        let host = UdpSocket::bind("127.0.0.1:0").unwrap();
        let dev = UdpSocket::bind("127.0.0.1:0").unwrap();
        host.connect(dev.local_addr().unwrap()).unwrap();
        dev.connect(host.local_addr().unwrap()).unwrap();
        host.set_nonblocking(true).unwrap();
        dev.set_nonblocking(true).unwrap();
        let mut a = WireAdapter::new(
            host,
            SteppedClock::new(0, ClockId::HostMonotonic),
            1,
            Duration::from_millis(30),
            Duration::from_millis(1),
        );
        a.set_transmit_enabled(true);
        let mut d = FirmwarePeer::new(dev, vec![0.0], 0.01);
        a.send_command(&[0.5], a.clock.now()).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let mut pos = [0.0];
        let mut k = 0;
        loop {
            d.step(k * 10_000_000, 0.01).unwrap();
            k += 1;
            if a.poll_feedback(&mut pos).unwrap().is_some() && pos[0] > 0.0 {
                break;
            }
            assert!(std::time::Instant::now() < deadline);
            std::thread::yield_now();
        }
        assert!(pos[0] > 0.0);
    }
}

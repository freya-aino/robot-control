//! Live capture through a documented Rust protocol adapter, per SPECS.md
//! Section 8.1: "Add live capture through documented Rust protocol
//! adapters for optical or inertial systems. Keep timestamps and confidence
//! through every transform."
//!
//! # The `robot-capture/1` protocol
//!
//! One frame per line (a UDP datagram, or a line of any byte stream), ASCII,
//! space-separated:
//!
//! ```text
//! F <seq> <source_time_ns> <count> (<name> <x> <y> <z> <confidence>){count}
//! ```
//!
//! `seq` is a `u64` that increases by one per frame at the source, so a gap
//! is a dropped frame, not a guess; `source_time_ns` is the capture
//! system's own sampling clock; positions are metres in the capture
//! system's frame; `confidence` is in `[0, 1]` (a vendor residual or
//! tracking quality, mapped by the vendor-side bridge, which documents its
//! mapping). A line that does not parse is counted and skipped, never
//! guessed at. A vendor SDK speaks this protocol through a bridge process
//! outside this repository; this crate does not link any vendor SDK.
//!
//! # Timestamps
//!
//! Each sample carries two: the **sampling** stamp on the capture clock,
//! and an **arrival** stamp on the host clock with an uncertainty (the
//! adapter's poll granularity). [`Recorder`] keeps the sampling stamps as
//! the record's timeline, so network jitter never moves a frame, and keeps
//! the running minimum of `arrival - source` as the clock-offset estimate
//! (the least-delayed frame bounds the offset most tightly).

use core::fmt;
use std::io::BufRead;
use std::net::UdpSocket;

use robot_types::frames::{FrameId, Transform, Vec3};
use robot_types::ids::Name;
use robot_types::motion::{
    Channel, HumanSkeleton, JointValue, MotionError, MotionFrame, MotionRecord, ProvenanceStep,
    SourceLicense, Validity,
};
use robot_types::time::{ClockId, Duration, Timestamp};

/// One marker in a captured frame.
#[derive(Clone, PartialEq, Debug)]
pub struct CapturedMarker {
    /// Marker name.
    pub name: String,
    /// Position, metres.
    pub position: Vec3,
    /// Confidence in `[0, 1]`.
    pub confidence: f64,
}

/// One captured frame.
#[derive(Clone, PartialEq, Debug)]
pub struct CaptureSample {
    /// Source sequence number.
    pub seq: u64,
    /// Sampling instant on the capture clock.
    pub source_time: Timestamp,
    /// Arrival instant on the host clock.
    pub arrival: Timestamp,
    /// Markers.
    pub markers: Vec<CapturedMarker>,
}

/// Why a line was not a frame.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum CaptureError {
    /// The line did not match the protocol.
    Malformed(String),
    /// The transport failed.
    Io(String),
}

impl fmt::Display for CaptureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Malformed(why) => write!(f, "malformed capture line: {why}"),
            Self::Io(e) => write!(f, "capture transport: {e}"),
        }
    }
}

impl core::error::Error for CaptureError {}

/// The capture clock's identifier in records this module produces.
pub const CAPTURE_CLOCK: ClockId = ClockId::Source(1);

/// Parses one `robot-capture/1` line received at `arrival`.
pub fn parse_line(line: &str, arrival: Timestamp) -> Result<CaptureSample, CaptureError> {
    let bad = |why: &str| CaptureError::Malformed(why.to_owned());
    let mut t = line.split_ascii_whitespace();
    if t.next() != Some("F") {
        return Err(bad("missing F tag"));
    }
    let seq: u64 = t
        .next()
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| bad("seq"))?;
    let time: i64 = t
        .next()
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| bad("source_time_ns"))?;
    let count: usize = t
        .next()
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| bad("count"))?;
    if count > 4096 {
        return Err(bad("count over 4096"));
    }
    let mut markers = Vec::with_capacity(count);
    for _ in 0..count {
        let name = t.next().ok_or_else(|| bad("marker name"))?.to_owned();
        let mut num = || -> Result<f64, CaptureError> {
            t.next()
                .and_then(|v| v.parse::<f64>().ok())
                .filter(|v| v.is_finite())
                .ok_or_else(|| bad("marker value"))
        };
        let position = Vec3::new(num()?, num()?, num()?);
        let confidence = num()?;
        if !(0.0..=1.0).contains(&confidence) {
            return Err(bad("confidence outside [0, 1]"));
        }
        markers.push(CapturedMarker {
            name,
            position,
            confidence,
        });
    }
    if t.next().is_some() {
        return Err(bad("trailing tokens"));
    }
    Ok(CaptureSample {
        seq,
        source_time: Timestamp::sampling(time, CAPTURE_CLOCK),
        arrival,
        markers,
    })
}

/// A source of captured frames.
pub trait CaptureAdapter {
    /// The next frame if one is available now, without blocking.
    fn poll(&mut self) -> Result<Option<CaptureSample>, CaptureError>;
    /// Lines that did not parse so far.
    fn malformed(&self) -> u64;
}

/// The host clock as nanoseconds since the adapter was created.
#[derive(Clone, Copy, Debug)]
struct HostClock {
    start: std::time::Instant,
}

impl HostClock {
    fn now(&self, uncertainty: Duration) -> Timestamp {
        let n = i64::try_from(self.start.elapsed().as_nanos()).unwrap_or(i64::MAX);
        Timestamp::arrival(n, ClockId::HostMonotonic, uncertainty)
    }
}

/// `robot-capture/1` over any line source (a file replay, a TCP stream).
pub struct LineAdapter<R: BufRead> {
    reader: R,
    clock: HostClock,
    malformed: u64,
    line: String,
}

impl<R: BufRead> LineAdapter<R> {
    /// Wraps `reader`.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            clock: HostClock {
                start: std::time::Instant::now(),
            },
            malformed: 0,
            line: String::new(),
        }
    }
}

impl<R: BufRead> CaptureAdapter for LineAdapter<R> {
    fn poll(&mut self) -> Result<Option<CaptureSample>, CaptureError> {
        loop {
            self.line.clear();
            let n = self
                .reader
                .read_line(&mut self.line)
                .map_err(|e| CaptureError::Io(e.to_string()))?;
            if n == 0 {
                return Ok(None);
            }
            if self.line.trim().is_empty() {
                continue;
            }
            match parse_line(&self.line, self.clock.now(Duration::ZERO)) {
                Ok(s) => return Ok(Some(s)),
                Err(_) => self.malformed += 1,
            }
        }
    }

    fn malformed(&self) -> u64 {
        self.malformed
    }
}

/// `robot-capture/1` over UDP, one frame per datagram, non-blocking.
pub struct UdpAdapter {
    socket: UdpSocket,
    clock: HostClock,
    malformed: u64,
    buf: Vec<u8>,
}

impl UdpAdapter {
    /// Binds `addr` (for example `"0.0.0.0:9763"`).
    pub fn bind(addr: &str) -> Result<Self, CaptureError> {
        let socket = UdpSocket::bind(addr).map_err(|e| CaptureError::Io(e.to_string()))?;
        socket
            .set_nonblocking(true)
            .map_err(|e| CaptureError::Io(e.to_string()))?;
        Ok(Self {
            socket,
            clock: HostClock {
                start: std::time::Instant::now(),
            },
            malformed: 0,
            buf: vec![0; 65_536],
        })
    }

    /// The bound address.
    pub fn local_addr(&self) -> Result<std::net::SocketAddr, CaptureError> {
        self.socket
            .local_addr()
            .map_err(|e| CaptureError::Io(e.to_string()))
    }
}

impl CaptureAdapter for UdpAdapter {
    fn poll(&mut self) -> Result<Option<CaptureSample>, CaptureError> {
        loop {
            match self.socket.recv(&mut self.buf) {
                Ok(n) => {
                    let arrival = self.clock.now(Duration::ZERO);
                    match core::str::from_utf8(&self.buf[..n])
                        .map_err(|e| CaptureError::Malformed(e.to_string()))
                        .and_then(|line| parse_line(line, arrival))
                    {
                        Ok(s) => return Ok(Some(s)),
                        Err(_) => self.malformed += 1,
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => return Ok(None),
                Err(e) => return Err(CaptureError::Io(e.to_string())),
            }
        }
    }

    fn malformed(&self) -> u64 {
        self.malformed
    }
}

/// Accumulates samples into a marker record.
#[derive(Clone, Debug)]
pub struct Recorder {
    names: Vec<String>,
    min_confidence: f64,
    frames: Vec<MotionFrame>,
    confidence: Vec<Vec<Option<f64>>>,
    last_seq: Option<u64>,
    dropped: u64,
    out_of_order: u64,
    offset: Option<i64>,
}

impl Recorder {
    /// Records the named markers; a marker below `min_confidence` or absent
    /// from a frame is marked missing in that frame.
    #[must_use]
    pub fn new(names: Vec<String>, min_confidence: f64) -> Self {
        Self {
            names,
            min_confidence,
            frames: Vec::new(),
            confidence: Vec::new(),
            last_seq: None,
            dropped: 0,
            out_of_order: 0,
            offset: None,
        }
    }

    /// Adds a sample. A sequence number at or below the last one is a
    /// duplicate or reordered datagram and is counted and discarded, so
    /// the timeline stays monotonic.
    pub fn push(&mut self, s: &CaptureSample) {
        if let Some(last) = self.last_seq {
            if s.seq <= last {
                self.out_of_order += 1;
                return;
            }
            self.dropped += s.seq - last - 1;
        }
        self.last_seq = Some(s.seq);
        let lag = s.arrival.nanos() - s.source_time.nanos();
        self.offset = Some(self.offset.map_or(lag, |o| o.min(lag)));
        let mut values = Vec::with_capacity(self.names.len());
        let mut validity = Vec::with_capacity(self.names.len());
        let mut conf = Vec::with_capacity(self.names.len());
        for name in &self.names {
            match s.markers.iter().find(|m| &m.name == name) {
                Some(m) if m.confidence >= self.min_confidence => {
                    values.push(JointValue::Position {
                        position: m.position,
                    });
                    validity.push(Validity::Valid);
                    conf.push(Some(m.confidence));
                }
                other => {
                    values.push(JointValue::Position {
                        position: Vec3::ZERO,
                    });
                    validity.push(Validity::Missing);
                    conf.push(other.map(|m| m.confidence));
                }
            }
        }
        self.frames.push(MotionFrame {
            timestamp: s.source_time,
            root: Transform::identity(FrameId::WORLD),
            values,
            validity,
            contacts: Vec::new(),
        });
        self.confidence.push(conf);
    }

    /// Frames lost in transit, from sequence gaps.
    #[must_use]
    pub const fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Datagrams discarded as duplicates or reordered.
    #[must_use]
    pub const fn out_of_order(&self) -> u64 {
        self.out_of_order
    }

    /// The estimated host-minus-capture clock offset (an upper bound on
    /// it, since every delay is non-negative).
    #[must_use]
    pub fn clock_offset(&self) -> Option<Duration> {
        self.offset.map(Duration::from_nanos)
    }

    /// Per frame, per marker: the confidence the source reported.
    #[must_use]
    pub fn confidence(&self) -> &[Vec<Option<f64>>] {
        &self.confidence
    }

    /// The record so far.
    pub fn finish(
        &self,
        session: &str,
        license: SourceLicense,
    ) -> Result<MotionRecord<HumanSkeleton>, MotionError> {
        let channels = self
            .names
            .iter()
            .map(|n| {
                Ok(Channel {
                    name: Name::new(n)?,
                    parent: None,
                    offset: Vec3::ZERO,
                })
            })
            .collect::<Result<Vec<_>, MotionError>>()?;
        let mut r = MotionRecord::new(
            HumanSkeleton,
            format!("live:{session}"),
            license,
            session,
            "capture-markers",
            FrameId::WORLD,
            CAPTURE_CLOCK,
            channels,
            self.frames.clone(),
        )?;
        r.record_step(
            ProvenanceStep::new("capture", concat!("motion-io/", env!("CARGO_PKG_VERSION")))
                .with_parameter("protocol", "robot-capture/1")
                .with_parameter("dropped", self.dropped.to_string())
                .with_parameter("out_of_order", self.out_of_order.to_string())
                .with_parameter(
                    "clock_offset_ns",
                    self.offset.map_or("unknown".to_owned(), |o| o.to_string()),
                ),
        );
        Ok(r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STREAM: &str = "F 1 1000000 2 lwr 0.1 0.2 0.3 0.9 rwr -0.1 0.2 0.3 0.95
garbage line
F 2 11000000 2 lwr 0.1 0.2 0.4 0.2 rwr -0.1 0.2 0.3 0.95
F 2 11000000 2 lwr 0 0 0 1 rwr 0 0 0 1
F 5 41000000 1 rwr -0.1 0.2 0.35 0.9
";

    #[test]
    fn a_line_stream_records_gaps_confidence_and_sampling_time() {
        let mut a = LineAdapter::new(STREAM.as_bytes());
        let mut rec = Recorder::new(vec!["lwr".into(), "rwr".into()], 0.5);
        while let Some(s) = a.poll().unwrap() {
            rec.push(&s);
        }
        assert_eq!(a.malformed(), 1);
        assert_eq!(rec.dropped(), 2); // seq 3 and 4
        assert_eq!(rec.out_of_order(), 1); // the repeated seq 2
        let r = rec
            .finish("live-test", SourceLicense::unaudited("fixture-internal"))
            .unwrap();
        assert_eq!(r.frames.len(), 3);
        assert_eq!(r.frames[1].timestamp.nanos(), 11_000_000);
        assert_eq!(r.frames[1].validity[0], Validity::Missing); // 0.2 < 0.5
        assert_eq!(rec.confidence()[1][0], Some(0.2));
        assert_eq!(r.frames[2].validity[0], Validity::Missing); // absent
        assert_eq!(rec.confidence()[2][0], None);
        assert!(rec.clock_offset().is_some());
        r.validate().unwrap();
    }

    #[test]
    fn malformed_lines_are_refused_with_a_reason() {
        let t = Timestamp::arrival(0, ClockId::HostMonotonic, Duration::ZERO);
        assert!(parse_line("F 1 0 1 m 0 0 0 1.5", t).is_err());
        assert!(parse_line("F 1 0 2 m 0 0 0 1", t).is_err());
        assert!(parse_line("F 1 0 1 m 0 0 nan 1", t).is_err());
        assert!(parse_line("F 1 0 0 extra", t).is_err());
        assert!(parse_line("F 1 0 0", t).is_ok());
    }

    #[test]
    fn frames_arrive_over_udp() {
        let mut a = UdpAdapter::bind("127.0.0.1:0").unwrap();
        let addr = a.local_addr().unwrap();
        let tx = UdpSocket::bind("127.0.0.1:0").unwrap();
        tx.send_to(b"F 7 123 1 m 1 2 3 1", addr).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        let sample = loop {
            if let Some(s) = a.poll().unwrap() {
                break s;
            }
            assert!(std::time::Instant::now() < deadline, "no datagram");
            std::thread::yield_now();
        };
        assert_eq!(sample.seq, 7);
        assert_eq!(sample.markers[0].position, Vec3::new(1.0, 2.0, 3.0));
    }
}

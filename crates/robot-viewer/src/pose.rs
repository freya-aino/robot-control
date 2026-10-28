//! Per-step pose frames for a floating-base run: pelvis pose and joint angles,
//! written as CSV, JSON Lines and one SVG per step.
//!
//! This is the record described in `docs/decisions/0009-test-render-format.md`.
//! It is separate from [`crate::frame::ViewerFrame`], which needs full link
//! poses and contacts. [`FrameDump`] streams one step at a time, so memory
//! does not grow with the run length. A run is published only when it commits:
//! a failed or abandoned run leaves any previously committed outputs as they were.

use crate::csv;
use crate::jsonl;
use crate::svg;
use crate::timeline::{Timed, Timeline, TimelineError};
use robot_types::time::{ClockId, Timestamp};
use std::fmt;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Name of the CSV file inside a dump directory.
pub const CSV_FILE: &str = "frames.csv";
/// Name of the JSON Lines file inside a dump directory.
pub const JSONL_FILE: &str = "frames.jsonl";
/// Name of the subdirectory holding one SVG per step.
pub const SVG_DIR: &str = "svg";
/// Name of the commit manifest inside a dump directory. It is written last and
/// removed first, so its presence is what makes the frame files loadable.
pub const COMMIT_FILE: &str = "run.commit";
/// First line of the commit manifest.
pub const COMMIT_MARKER: &str = "# robot-viewer.pose-commit v1";
/// Name of the staging directory a dump writes into until it is committed. Its
/// existence is the output lock: while it exists, only its owner writes to `dir`.
const STAGING_DIR: &str = ".pose-staging";
/// File inside the staging directory that records the owning process id.
const OWNER_FILE: &str = "owner";
/// How many times [`FrameDump::create`] retries the lock after reclaiming a stale
/// staging directory, in case another process reclaims it at the same moment.
const RECLAIM_ATTEMPTS: usize = 3;
/// Largest step a dump can hold. The per-step SVG names are `frame_NNNNNN.svg`,
/// six digits wide, so a longer run would need wider names and is refused before
/// anything is written.
pub const MAX_FRAME_STEPS: u64 = 999_999;
/// Largest timestamp, in nanoseconds, allowed in a frame dump.
pub const MAX_TIMESTAMP_NS: i64 = 1 << 53;
/// Longest header line a frame file may hold. The writer checks the encoded
/// header against this limit before creating the output directory.
const MAX_HEADER_BYTES: usize = 64 * 1024;
/// Most joint angles one frame may carry. Covers the G1 with room to spare.
pub const MAX_JOINTS: usize = 256;
/// Fixed room for row keys, integers, seven pose scalars, separators and JSON
/// punctuation, based on the finite `f64` Display bound below.
const FRAME_LINE_BASE_BYTES: usize = 4096;
/// Room for a joint scalar and its separator. A finite `f64` Display value is
/// shorter than 330 bytes, including its sign; 340 leaves formatting margin.
const FRAME_LINE_BYTES_PER_JOINT: usize = 340;

/// One simulated step: the pelvis pose and every single-DOF joint angle.
#[derive(Clone, PartialEq, Debug)]
pub struct PoseFrame {
    /// 1-based index of the step that produced this state.
    pub step: u64,
    /// Simulation time of this state, `step * dt`, on [`ClockId::Simulation`].
    pub timestamp: Timestamp,
    /// Pelvis position, metres, world frame.
    pub pelvis_position: [f64; 3],
    /// Pelvis orientation as `[w, x, y, z]`.
    pub pelvis_quaternion: [f64; 4],
    /// One angle per single-DOF joint, in the same order as the dump's joint names.
    pub joint_angles: Vec<f64>,
}

impl Timed for PoseFrame {
    fn timestamp(&self) -> Timestamp {
        self.timestamp
    }
}

impl PoseFrame {
    /// Checks that every value is finite, the timestamp is within bounds
    /// and matches the clock, and that there is one angle per joint.
    pub fn validate(&self, joints: usize) -> Result<(), PoseError> {
        if self.joint_angles.len() != joints {
            return Err(PoseError::JointCount {
                step: self.step,
                expected: joints,
                found: self.joint_angles.len(),
            });
        }
        if self.timestamp.clock() != ClockId::Simulation {
            return Err(PoseError::InvalidClock {
                step: self.step,
                expected: ClockId::Simulation,
                found: self.timestamp.clock(),
            });
        }
        if self.timestamp.nanos() < 0 || self.timestamp.nanos() > MAX_TIMESTAMP_NS {
            return Err(PoseError::TimestampOutOfBounds {
                step: self.step,
                found: self.timestamp.nanos(),
                limit: MAX_TIMESTAMP_NS,
            });
        }
        let finite = self.pelvis_position.iter().all(|v| v.is_finite())
            && self.pelvis_quaternion.iter().all(|v| v.is_finite())
            && self.joint_angles.iter().all(|v| v.is_finite());
        if !finite {
            return Err(PoseError::NotFinite { step: self.step });
        }
        Ok(())
    }
}

/// A frame file read back and validated, before it is placed on a timeline.
#[derive(Clone, PartialEq, Debug)]
pub struct PoseLog {
    /// Joint names, in the order each frame's `joint_angles` uses.
    pub joints: Vec<String>,
    /// Frames in step order, with steps 1, 2, 3, ... and strictly rising times.
    pub frames: Vec<PoseFrame>,
}

impl PoseLog {
    /// Places the frames on a simulation-clock [`Timeline`].
    pub fn into_timeline(self) -> Result<Timeline<PoseFrame>, PoseError> {
        let mut timeline = Timeline::with_clock(ClockId::Simulation);
        for frame in self.frames {
            timeline.push(frame).map_err(PoseError::Timeline)?;
        }
        Ok(timeline)
    }
}

/// Why pose frames could not be written, read or rendered.
#[derive(Debug)]
pub enum PoseError {
    /// A file could not be created, written or removed.
    Io {
        /// The file or directory involved.
        path: PathBuf,
        /// The underlying error.
        source: io::Error,
    },
    /// A file did not match the format. Nothing from it is used.
    Malformed {
        /// 1-based line number, or 0 for a file-level problem.
        line: usize,
        /// What was wrong.
        reason: String,
    },
    /// A value was NaN or infinite.
    NotFinite {
        /// The step that held the value.
        step: u64,
    },
    /// A frame had the wrong number of joint angles.
    JointCount {
        /// The step that held the frame.
        step: u64,
        /// Joints declared by the dump.
        expected: usize,
        /// Angles supplied.
        found: usize,
    },
    /// A frame arrived out of step order.
    OutOfSequence {
        /// The step that was expected next.
        expected: u64,
        /// The step that was supplied.
        found: u64,
    },
    /// A joint name would not survive the CSV and JSON Lines formats.
    InvalidJointName {
        /// The rejected name.
        name: String,
    },
    /// The dump would need more joint angles per frame than a frame file allows.
    TooManyJoints {
        /// Joints requested.
        found: usize,
        /// The most a frame file may declare.
        limit: usize,
    },
    /// An encoded CSV or JSONL header exceeds the reader's bounded line size.
    HeaderTooLong {
        /// Format whose header is too large.
        format: &'static str,
        /// Encoded line length, including its newline.
        found: usize,
        /// Maximum encoded line length.
        limit: usize,
    },
    /// The run is longer than the six-digit SVG names can hold.
    TooManySteps {
        /// The step that was refused.
        step: u64,
        /// The largest step a dump can hold.
        limit: u64,
    },
    /// The frames could not be placed on a timeline.
    Timeline(TimelineError),
    /// The output path was empty, so there is no directory to write into.
    EmptyDirectory,
    /// The directory holds no commit manifest, so it is not a completed run.
    NotCommitted {
        /// The directory that was read.
        path: PathBuf,
    },
    /// Another run holds the output directory, or its staging directory cannot
    /// be shown to be stale. Nothing was written or removed.
    Locked {
        /// The staging directory that holds the lock.
        path: PathBuf,
        /// Why the lock was not taken.
        reason: String,
    },
    /// A timestamp was on a clock other than `Simulation`.
    InvalidClock {
        /// The step that had the wrong clock.
        step: u64,
        /// Expected `ClockId::Simulation`.
        expected: ClockId,
        /// The clock found.
        found: ClockId,
    },
    /// A timestamp was negative or exceeded the `MAX_TIMESTAMP_NS` limit.
    TimestampOutOfBounds {
        /// The step that had the invalid timestamp.
        step: u64,
        /// The timestamp found.
        found: i64,
        /// The limit applied.
        limit: i64,
    },
    /// A frame arrived with a timestamp that is not strictly increasing.
    TimeNotIncreasing {
        /// The step that violated monotonicity.
        step: u64,
        /// The previous timestamp.
        previous: i64,
        /// The found timestamp.
        found: i64,
    },
}

impl fmt::Display for PoseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "{}: {source}", path.display()),
            Self::Malformed { line, reason } => write!(f, "line {line}: {reason}"),
            Self::NotFinite { step } => write!(f, "step {step} contains a non-finite value"),
            Self::JointCount {
                step,
                expected,
                found,
            } => write!(
                f,
                "step {step} has {found} joint angles, the dump declares {expected}"
            ),
            Self::OutOfSequence { expected, found } => {
                write!(f, "expected step {expected}, found step {found}")
            }
            Self::InvalidJointName { name } => write!(f, "joint name {name:?} is not allowed"),
            Self::TooManyJoints { found, limit } => {
                write!(
                    f,
                    "{found} joints requested, a frame file allows at most {limit}"
                )
            }
            Self::HeaderTooLong {
                format,
                found,
                limit,
            } => write!(
                f,
                "{format} header is {found} bytes, exceeding the {limit}-byte limit"
            ),
            Self::TooManySteps { step, limit } => write!(
                f,
                "step {step} is past the {limit}-step limit of six-digit SVG names"
            ),
            Self::Timeline(e) => write!(f, "{e}"),
            Self::EmptyDirectory => write!(f, "the output directory path is empty"),
            Self::NotCommitted { path } => write!(
                f,
                "{}: no {COMMIT_FILE} manifest, so this is not a completed run",
                path.display()
            ),
            Self::Locked { path, reason } => write!(f, "{}: {reason}", path.display()),
            Self::InvalidClock {
                step,
                expected,
                found,
            } => write!(f, "step {step} uses clock {found:?}, expected {expected:?}"),
            Self::TimestampOutOfBounds { step, found, limit } => {
                write!(
                    f,
                    "step {step} timestamp {found} ns is out of bounds [0, {limit}]"
                )
            }
            Self::TimeNotIncreasing {
                step,
                previous,
                found,
            } => write!(
                f,
                "step {step} timestamp {found} ns is not strictly after {previous} ns"
            ),
        }
    }
}

impl std::error::Error for PoseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Timeline(e) => Some(e),
            _ => None,
        }
    }
}

/// Whether a joint name can appear unescaped in both the CSV and the JSON
/// Lines output: non-empty ASCII letters, digits, `_`, `-` or `.`.
pub fn is_valid_joint_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.')
}

#[derive(Default)]
struct LineLengthCounter {
    current: usize,
    max: usize,
}

impl Write for LineLengthCounter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        for &byte in bytes {
            self.current += 1;
            if byte == b'\n' {
                self.max = self.max.max(self.current);
                self.current = 0;
            }
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn max_written_line_length(write: impl FnOnce(&mut LineLengthCounter) -> io::Result<()>) -> usize {
    let mut counter = LineLengthCounter::default();
    write(&mut counter).expect("writing a frame header to a counting sink cannot fail");
    counter.max.max(counter.current)
}

/// Builds a [`PoseError::Malformed`] for `line`.
pub(crate) fn malformed(line: usize, reason: impl Into<String>) -> PoseError {
    PoseError::Malformed {
        line,
        reason: reason.into(),
    }
}

/// Parses one decimal float token. Only digits, sign, point and exponent are
/// accepted, so `NaN`, `inf` and hexadecimal forms are rejected before parsing,
/// and the parsed value must be finite.
pub(crate) fn parse_float(token: &str, line: usize) -> Result<f64, PoseError> {
    let charset = !token.is_empty()
        && token
            .bytes()
            .all(|b| b.is_ascii_digit() || matches!(b, b'+' | b'-' | b'.' | b'e' | b'E'));
    if !charset {
        return Err(malformed(line, format!("not a decimal number: {token:?}")));
    }
    let value: f64 = token
        .parse()
        .map_err(|_| malformed(line, format!("not a decimal number: {token:?}")))?;
    if !value.is_finite() {
        return Err(malformed(line, format!("non-finite value {token:?}")));
    }
    Ok(value)
}

/// Parses one non-negative integer token made only of ASCII digits.
pub(crate) fn parse_unsigned(token: &str, line: usize) -> Result<u64, PoseError> {
    if token.is_empty() || !token.bytes().all(|b| b.is_ascii_digit()) {
        return Err(malformed(
            line,
            format!("not an unsigned integer: {token:?}"),
        ));
    }
    token
        .parse()
        .map_err(|_| malformed(line, format!("integer out of range: {token:?}")))
}

/// One decoded line of a frame file: a step, or the end footer.
pub(crate) enum Record {
    /// A step line, already checked for field count and number syntax.
    Frame(PoseFrame),
    /// The end footer and the step count it declares.
    Footer(u64),
}

/// The line rules of one frame file format. [`FrameStream`] applies the rules
/// that span lines; a format says only what its own lines look like.
pub(crate) trait FrameFormat {
    /// Reads the opening lines and returns the joint names they declare.
    fn open<R: BufRead>(lines: &mut LineReader<R>) -> Result<Vec<String>, PoseError>;

    /// Decodes one line that follows the opening lines.
    fn decode(line: usize, text: &str, joints: &[String]) -> Result<Record, PoseError>;
}

/// Yields the lines of a frame file one at a time, without their line endings.
/// Every line must end in `\n`, so a file cut mid-line is rejected.
pub(crate) struct LineReader<R> {
    input: R,
    path: Option<PathBuf>,
    line: usize,
    limit: usize,
    text: String,
}

impl<R: BufRead> LineReader<R> {
    fn new(input: R, path: Option<PathBuf>) -> Self {
        Self {
            input,
            path,
            line: 0,
            limit: MAX_HEADER_BYTES,
            text: String::new(),
        }
    }

    /// Sets the longest line, in bytes, that [`Self::next_line`] will accept.
    pub(crate) fn set_limit(&mut self, bytes: usize) {
        self.limit = bytes;
    }

    /// The next line with its 1-based number, or `None` at the end of input.
    /// A line longer than the limit is rejected before it is buffered in full.
    pub(crate) fn next_line(&mut self) -> Result<Option<(usize, &str)>, PoseError> {
        self.text.clear();
        let limit = self.limit as u64;
        let read = Read::take(&mut self.input, limit + 1)
            .read_line(&mut self.text)
            .map_err(|e| self.read_error(e))?;
        if read == 0 {
            return Ok(None);
        }
        self.line += 1;
        if self.text.len() > self.limit {
            return Err(malformed(
                self.line,
                format!("line is longer than {} bytes", self.limit),
            ));
        }
        if !self.text.ends_with('\n') {
            return Err(malformed(0, "missing final newline"));
        }
        self.text.pop();
        if self.text.ends_with('\r') {
            self.text.pop();
        }
        Ok(Some((self.line, &self.text)))
    }

    fn read_error(&self, source: io::Error) -> PoseError {
        match &self.path {
            Some(path) => io_error(path, source),
            None => malformed(self.line + 1, source.to_string()),
        }
    }
}

/// A frame file read one line at a time. The format decodes each line, and this
/// applies the rules that span lines: steps 1, 2, 3, ..., strictly rising times,
/// nothing after the footer, and a footer whose count matches the steps read.
/// Each frame is handed out once it is checked, so the caller decides what to keep.
pub(crate) struct FrameStream<R, F> {
    lines: LineReader<R>,
    joints: Vec<String>,
    steps: u64,
    previous_time: Option<i64>,
    footer: Option<(usize, u64)>,
    format: PhantomData<F>,
}

impl<R: BufRead, F: FrameFormat> FrameStream<R, F> {
    /// Reads the opening lines of `input`.
    pub(crate) fn open(input: R, path: Option<PathBuf>) -> Result<Self, PoseError> {
        let mut lines = LineReader::new(input, path);
        let joints = F::open(&mut lines)?;
        if joints.len() > MAX_JOINTS {
            return Err(malformed(
                0,
                format!("{} joints, the limit is {MAX_JOINTS}", joints.len()),
            ));
        }
        lines.set_limit(FRAME_LINE_BASE_BYTES + FRAME_LINE_BYTES_PER_JOINT * joints.len());
        Ok(Self {
            lines,
            joints,
            steps: 0,
            previous_time: None,
            footer: None,
            format: PhantomData,
        })
    }

    /// Joint names, in the order each frame's angles use.
    pub(crate) fn joints(&self) -> &[String] {
        &self.joints
    }

    /// The next frame in step order. Returns `None` only after the end of the
    /// input has been reached and the footer has been checked.
    pub(crate) fn next_frame(&mut self) -> Result<Option<PoseFrame>, PoseError> {
        loop {
            let Some((line, text)) = self.lines.next_line()? else {
                return self.end().map(|()| None);
            };
            if self.footer.is_some() {
                return Err(malformed(line, "extra record after end footer"));
            }
            match F::decode(line, text, &self.joints)? {
                Record::Footer(steps) => self.footer = Some((line, steps)),
                Record::Frame(frame) => {
                    self.check_order(&frame, line)?;
                    return Ok(Some(frame));
                }
            }
        }
    }

    fn check_order(&mut self, frame: &PoseFrame, line: usize) -> Result<(), PoseError> {
        frame
            .validate(self.joints.len())
            .map_err(|error| malformed(line, error.to_string()))?;
        let expected = self.steps + 1;
        if frame.step != expected {
            return Err(malformed(
                line,
                format!("step {} out of sequence, expected {expected}", frame.step),
            ));
        }
        let nanos = frame.timestamp.nanos();
        if self.previous_time.is_some_and(|prev| nanos <= prev) {
            return Err(malformed(line, "time_ns is not strictly increasing"));
        }
        self.previous_time = Some(nanos);
        self.steps = expected;
        Ok(())
    }

    fn end(&self) -> Result<(), PoseError> {
        let Some((footer_line, footer_steps)) = self.footer else {
            return Err(malformed(0, "missing end footer"));
        };
        if footer_steps != self.steps {
            return Err(malformed(
                footer_line,
                format!("footer declares {footer_steps} steps, found {}", self.steps),
            ));
        }
        Ok(())
    }
}

/// Opens a frame file on disk as a [`FrameStream`].
pub(crate) fn open_file<F: FrameFormat>(
    path: &Path,
) -> Result<FrameStream<BufReader<File>, F>, PoseError> {
    let file = File::open(path).map_err(|e| io_error(path, e))?;
    FrameStream::open(BufReader::new(file), Some(path.to_path_buf()))
}

/// Reads every frame of a stream into a [`PoseLog`]. Nothing is returned unless
/// the whole input passes.
pub(crate) fn collect_log<R: BufRead, F: FrameFormat>(
    mut stream: FrameStream<R, F>,
) -> Result<PoseLog, PoseError> {
    let mut frames = Vec::new();
    while let Some(frame) = stream.next_frame()? {
        frames.push(frame);
    }
    Ok(PoseLog {
        joints: stream.joints().to_vec(),
        frames,
    })
}

/// What a streaming check of one run's two frame files found.
pub(crate) struct PairCheck {
    /// Joint names both files declare.
    pub(crate) joints: Vec<String>,
    /// Number of steps in both files.
    pub(crate) steps: u64,
    /// The last frame, if any.
    pub(crate) last: Option<PoseFrame>,
    /// Every frame, in order. Filled only when the caller asked to keep them.
    pub(crate) frames: Vec<PoseFrame>,
}

/// Whether two frames are the same bit for bit, so `0.0` and `-0.0` differ.
fn same_frame(a: &PoseFrame, b: &PoseFrame) -> bool {
    fn same_floats(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }
    a.step == b.step
        && a.timestamp.nanos() == b.timestamp.nanos()
        && same_floats(&a.pelvis_position, &b.pelvis_position)
        && same_floats(&a.pelvis_quaternion, &b.pelvis_quaternion)
        && same_floats(&a.joint_angles, &b.joint_angles)
}

/// Reads `frames.csv` and `frames.jsonl` side by side, one frame from each at a
/// time, with the same strict line rules as the readers. Each pair must match
/// bit for bit, including every pelvis value and joint angle. Memory is one frame
/// per file plus one line buffer per file, not the run length. Frames are kept
/// only when `keep` is set, which is the explicit load.
pub(crate) fn verify_pair(
    csv_path: &Path,
    jsonl_path: &Path,
    keep: bool,
) -> Result<PairCheck, PoseError> {
    let mut csv = open_file::<csv::Csv>(csv_path)?;
    let mut jsonl = open_file::<jsonl::Jsonl>(jsonl_path)?;
    if csv.joints() != jsonl.joints() {
        return Err(malformed(0, "frames.csv and frames.jsonl disagree"));
    }
    let mut check = PairCheck {
        joints: csv.joints().to_vec(),
        steps: 0,
        last: None,
        frames: Vec::new(),
    };
    loop {
        match (csv.next_frame()?, jsonl.next_frame()?) {
            (None, None) => return Ok(check),
            (Some(a), Some(b)) => {
                if !same_frame(&a, &b) {
                    return Err(malformed(
                        0,
                        format!("frames.csv and frames.jsonl disagree at step {}", a.step),
                    ));
                }
                check.steps += 1;
                if keep {
                    check.frames.push(a.clone());
                }
                check.last = Some(a);
            }
            _ => {
                return Err(malformed(
                    0,
                    "frames.csv and frames.jsonl hold different numbers of steps",
                ))
            }
        }
    }
}

/// The SVG file name for a step, `frame_NNNNNN.svg`.
pub fn svg_file_name(step: u64) -> String {
    format!("frame_{step:06}.svg")
}

fn io_error(path: &Path, source: io::Error) -> PoseError {
    PoseError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// Whether `name` is a file name this module writes into the SVG directory.
fn is_svg_frame_name(name: &str) -> bool {
    name.len() == "frame_000000.svg".len()
        && name.starts_with("frame_")
        && name.ends_with(".svg")
        && name.as_bytes()[6..12].iter().all(u8::is_ascii_digit)
}

/// The step a `frame_NNNNNN.svg` name belongs to, or `None` for any other name.
fn svg_frame_step(name: &str) -> Option<u64> {
    if !is_svg_frame_name(name) {
        return None;
    }
    name[6..12].parse().ok()
}

/// Streams one run's frames to `frames.csv`, `frames.jsonl` and `svg/`, with
/// transactional publication.
///
/// [`FrameDump::create`] writes into a private staging directory inside `dir`
/// and leaves every existing file in `dir` alone. [`FrameDump::finish`] commits
/// the run: it renames the staged files into place and writes the
/// [`COMMIT_FILE`] manifest last. Only a committed directory is accepted by
/// [`load_committed`]. A dump dropped before a successful `finish` removes its
/// staging directory and nothing else, so a failed or abandoned run never
/// exposes partial output and never removes the outputs of an earlier run.
/// Frames must arrive as steps 1, 2, 3, ... Only one dump may hold a directory
/// at a time: [`FrameDump::create`] fails with [`PoseError::Locked`] while another
/// run's staging directory exists and that run is still alive.
pub struct FrameDump {
    dir: PathBuf,
    staging: Option<PathBuf>,
    joints: usize,
    csv: BufWriter<File>,
    jsonl: BufWriter<File>,
    written: u64,
    last: Option<PoseFrame>,
}

impl FrameDump {
    /// Takes the output lock, checks every output path, and opens the two staged
    /// frame files with their headers. The lock is the staging directory: it is
    /// created atomically, and an existing one is reclaimed only when its owner
    /// process is provably dead. A live or unidentifiable owner yields
    /// [`PoseError::Locked`]. Nothing already committed in `dir` is removed or
    /// replaced here, so any error leaves a committed run untouched.
    pub fn create(dir: &Path, joints: &[String]) -> Result<Self, PoseError> {
        if dir.as_os_str().is_empty() {
            return Err(PoseError::EmptyDirectory);
        }
        if let Some(bad) = joints.iter().find(|n| !is_valid_joint_name(n)) {
            return Err(PoseError::InvalidJointName { name: bad.clone() });
        }
        if joints.len() > MAX_JOINTS {
            return Err(PoseError::TooManyJoints {
                found: joints.len(),
                limit: MAX_JOINTS,
            });
        }
        for (format, header_bytes) in [
            (
                "CSV",
                max_written_line_length(|out| csv::write_header(out, joints)),
            ),
            (
                "JSON Lines",
                max_written_line_length(|out| jsonl::write_header(out, joints)),
            ),
        ] {
            if header_bytes > MAX_HEADER_BYTES {
                return Err(PoseError::HeaderTooLong {
                    format,
                    found: header_bytes,
                    limit: MAX_HEADER_BYTES,
                });
            }
        }
        fs::create_dir_all(dir).map_err(|e| io_error(dir, e))?;
        let staging = acquire_staging(dir)?;
        match prepare(dir, &staging, joints) {
            Ok((csv, jsonl)) => Ok(Self {
                dir: dir.to_path_buf(),
                staging: Some(staging),
                joints: joints.len(),
                csv,
                jsonl,
                written: 0,
                last: None,
            }),
            Err(e) => {
                let _ = fs::remove_dir_all(&staging);
                Err(e)
            }
        }
    }

    /// Appends one step to every output.
    pub fn push(&mut self, frame: &PoseFrame) -> Result<(), PoseError> {
        let expected = self.written + 1;
        if frame.step > MAX_FRAME_STEPS {
            return Err(PoseError::TooManySteps {
                step: frame.step,
                limit: MAX_FRAME_STEPS,
            });
        }
        if frame.step != expected {
            return Err(PoseError::OutOfSequence {
                expected,
                found: frame.step,
            });
        }
        if let Some(last) = &self.last {
            if frame.timestamp.nanos() <= last.timestamp.nanos() {
                return Err(PoseError::TimeNotIncreasing {
                    step: frame.step,
                    previous: last.timestamp.nanos(),
                    found: frame.timestamp.nanos(),
                });
            }
        }
        frame.validate(self.joints)?;
        let csv_path = self
            .staging
            .as_ref()
            .expect("staging missing")
            .join(CSV_FILE);
        let jsonl_path = self
            .staging
            .as_ref()
            .expect("staging missing")
            .join(JSONL_FILE);
        csv::write_row(&mut self.csv, frame).map_err(|source| PoseError::Io {
            path: csv_path,
            source,
        })?;
        jsonl::write_row(&mut self.jsonl, frame).map_err(|source| PoseError::Io {
            path: jsonl_path,
            source,
        })?;
        let svg_path = self
            .staging
            .as_ref()
            .expect("staging missing")
            .join(SVG_DIR)
            .join(svg_file_name(frame.step));
        let text = svg::render(frame);
        fs::write(&svg_path, text).map_err(|source| PoseError::Io {
            path: svg_path,
            source,
        })?;
        self.written += 1;
        self.last = Some(frame.clone());
        Ok(())
    }

    /// Completes the run and publishes it. Writes the footers, then checks the
    /// staged files before anything is published: the CSV and JSON Lines must
    /// parse with the strict readers, agree with each other, hold every step
    /// written, and end with the last frame pushed. A failed check returns an
    /// error and leaves a committed run in `dir` untouched. Otherwise the staged
    /// files are moved into `dir` and the [`COMMIT_FILE`] manifest is written
    /// last. Returns the number of steps committed. Call once after the last frame.
    ///
    /// The output lock is held until this returns, and nothing is read back or
    /// removed after the commit, so no later step of this process can touch a
    /// run that another writer has since committed.
    ///
    /// The earlier run's manifest is removed before anything is published, so a
    /// failure part way through leaves a directory that [`load_committed`]
    /// refuses. Besides that manifest, this removes only per-step SVG names past
    /// this run's last step.
    pub fn finish(mut self) -> Result<u64, PoseError> {
        self.commit()?;
        Ok(self.written)
    }

    fn commit(&mut self) -> Result<(), PoseError> {
        let staging = self
            .staging
            .as_ref()
            .expect("staging directory missing in commit");
        let staged_csv = staging.join(CSV_FILE);
        let staged_jsonl = staging.join(JSONL_FILE);
        csv::write_footer(&mut self.csv, self.written).map_err(|e| io_error(&staged_csv, e))?;
        jsonl::write_footer(&mut self.jsonl, self.written)
            .map_err(|e| io_error(&staged_jsonl, e))?;
        self.csv.flush().map_err(|e| io_error(&staged_csv, e))?;
        self.jsonl.flush().map_err(|e| io_error(&staged_jsonl, e))?;
        check_staged(&staged_csv, &staged_jsonl, self.written, self.last.as_ref())?;

        preflight_publish(&self.dir, self.written)?;
        remove_if_present(&self.dir.join(COMMIT_FILE))?;
        publish(&staged_csv, &self.dir.join(CSV_FILE))?;
        publish(&staged_jsonl, &self.dir.join(JSONL_FILE))?;
        let svg_target = self.dir.join(SVG_DIR);
        for step in 1..=self.written {
            let name = svg_file_name(step);
            publish(&staging.join(SVG_DIR).join(&name), &svg_target.join(&name))?;
        }
        remove_stale_svgs(&svg_target, self.written)?;
        let manifest = staging.join(COMMIT_FILE);
        fs::write(
            &manifest,
            format!("{COMMIT_MARKER}\nsteps {}\n", self.written),
        )
        .map_err(|e| io_error(&manifest, e))?;
        publish(&manifest, &self.dir.join(COMMIT_FILE))?;

        if let Some(staging) = self.staging.take() {
            let _ = fs::remove_dir_all(&staging);
        }
        Ok(())
    }
}

impl Drop for FrameDump {
    /// Removes the staging directory. Nothing outside it is touched, so a dump
    /// that never committed leaves any earlier committed run as it was.
    fn drop(&mut self) {
        if let Some(staging) = &self.staging {
            let _ = fs::remove_dir_all(staging);
        }
    }
}

/// Creates the staging directory's SVG subdirectory and the two staged frame
/// files, each with its header.
fn open_staged(
    staging: &Path,
    joints: &[String],
) -> Result<(BufWriter<File>, BufWriter<File>), PoseError> {
    let svg = staging.join(SVG_DIR);
    fs::create_dir_all(&svg).map_err(|e| io_error(&svg, e))?;
    let csv_path = staging.join(CSV_FILE);
    let jsonl_path = staging.join(JSONL_FILE);
    let csv_file = File::create(&csv_path).map_err(|e| io_error(&csv_path, e))?;
    let jsonl_file = File::create(&jsonl_path).map_err(|e| io_error(&jsonl_path, e))?;
    let mut csv = BufWriter::new(csv_file);
    let mut jsonl = BufWriter::new(jsonl_file);
    csv::write_header(&mut csv, joints).map_err(|e| io_error(&csv_path, e))?;
    jsonl::write_header(&mut jsonl, joints).map_err(|e| io_error(&jsonl_path, e))?;
    Ok((csv, jsonl))
}

/// Runs every preflight check under the output lock, then opens the staged
/// files. The caller removes the staging directory if this fails.
fn prepare(
    dir: &Path,
    staging: &Path,
    joints: &[String],
) -> Result<(BufWriter<File>, BufWriter<File>), PoseError> {
    preflight_targets(dir)?;
    probe_writable(dir)?;
    let svg_dir = dir.join(SVG_DIR);
    fs::create_dir_all(&svg_dir).map_err(|e| io_error(&svg_dir, e))?;
    probe_writable(&svg_dir)?;
    preflight_svg_entries(&svg_dir)?;
    open_staged(staging, joints)
}

/// Takes the output lock for `dir` by creating its staging directory, which
/// fails if another run holds it, and records this process as the owner. A
/// staging directory left by a dead run is reclaimed. One whose owner cannot be
/// shown to be dead is refused and never removed.
fn acquire_staging(dir: &Path) -> Result<PathBuf, PoseError> {
    let staging = dir.join(STAGING_DIR);
    for _ in 0..RECLAIM_ATTEMPTS {
        match fs::create_dir(&staging) {
            Ok(()) => {
                let owner = staging.join(OWNER_FILE);
                if let Err(e) = fs::write(&owner, owner_record()) {
                    let _ = fs::remove_dir_all(&staging);
                    return Err(io_error(&owner, e));
                }
                return Ok(staging);
            }
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                reclaim_or_refuse(dir, &staging)?;
            }
            Err(e) => return Err(io_error(&staging, e)),
        }
    }
    Err(locked(
        &staging,
        "the staging directory kept changing; try again",
    ))
}

/// Decides what to do with an existing staging directory. It is removed only
/// when its recorded owner is provably dead. Any other case returns an error,
/// so the caller never takes a lock that a live run holds.
fn reclaim_or_refuse(dir: &Path, staging: &Path) -> Result<(), PoseError> {
    let Some(owner) = owner_of(staging) else {
        return Err(locked(
            staging,
            "held by an unidentified run (no owner recorded); it is not reclaimed, so remove this directory if no run is active",
        ));
    };
    let pid = owner.pid;
    if pid == std::process::id() {
        return Err(locked(staging, format!("held by pid {pid}, this process")));
    }
    let Some(here) = host_identity() else {
        return Err(locked(
            staging,
            format!("held by pid {pid}; this platform cannot tell whether it is still running, so it is not reclaimed"),
        ));
    };
    if owner.namespace.as_deref() != Some(here.namespace.as_str())
        || owner.boot.as_deref() != Some(here.boot.as_str())
    {
        return Err(locked(
            staging,
            format!("held by pid {pid} from another pid namespace or boot, so its liveness cannot be checked here; it is not reclaimed"),
        ));
    }
    if owner_is_alive(pid) {
        return Err(locked(
            staging,
            format!("held by pid {pid}, which is still running"),
        ));
    }
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let reclaimed = dir.join(format!(
        "{STAGING_DIR}-reclaimed-{}-{nanos}",
        std::process::id()
    ));
    match fs::rename(staging, &reclaimed) {
        Ok(()) => {
            if owner_of(&reclaimed).as_ref() == Some(&owner) {
                return fs::remove_dir_all(&reclaimed).map_err(|e| io_error(&reclaimed, e));
            }
            // Another run created a live lock between our check and the rename.
            // Put it back; the next attempt will see its live owner and refuse.
            fs::rename(&reclaimed, staging).map_err(|_| {
                locked(
                    staging,
                    "the lock changed hands during reclaim and could not be restored; retry",
                )
            })
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(io_error(staging, e)),
    }
}

/// The owner recorded in a staging directory's owner file. A pid is only
/// meaningful inside the pid namespace and boot that issued it, so both are
/// recorded too. A missing field means the owner cannot be checked.
#[derive(Clone, PartialEq, Debug)]
struct Owner {
    pid: u32,
    namespace: Option<String>,
    boot: Option<String>,
}

/// Reads a staging directory's owner file. `None` if it is unreadable or holds
/// anything other than the recognised fields.
fn owner_of(staging: &Path) -> Option<Owner> {
    let text = fs::read_to_string(staging.join(OWNER_FILE)).ok()?;
    let mut pid = None;
    let mut namespace = None;
    let mut boot = None;
    for line in text.lines() {
        if let Some(value) = line.strip_prefix("pid ") {
            pid = Some(value.trim().parse::<u32>().ok().filter(|p| *p > 0)?);
        } else if let Some(value) = line.strip_prefix("namespace ") {
            namespace = Some(value.to_owned());
        } else {
            let value = line.strip_prefix("boot ")?;
            boot = Some(value.to_owned());
        }
    }
    Some(Owner {
        pid: pid?,
        namespace,
        boot,
    })
}

/// The pid namespace and boot id this process runs in, read from Linux procfs.
/// `None` where they cannot be read, in which case no lock is ever reclaimed.
struct HostIdentity {
    namespace: String,
    boot: String,
}

fn host_identity() -> Option<HostIdentity> {
    let namespace = fs::read_link("/proc/self/ns/pid")
        .ok()?
        .to_string_lossy()
        .into_owned();
    let boot = fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .ok()?
        .trim()
        .to_owned();
    Some(HostIdentity { namespace, boot })
}

/// The owner file contents for this process: its pid, plus its namespace and
/// boot when those can be read.
fn owner_record() -> String {
    let pid = std::process::id();
    match host_identity() {
        Some(here) => format!(
            "pid {pid}\nnamespace {}\nboot {}\n",
            here.namespace, here.boot
        ),
        None => format!("pid {pid}\n"),
    }
}

/// Whether process `pid` exists in this pid namespace. Only meaningful for an
/// owner whose namespace and boot match this process's.
fn owner_is_alive(pid: u32) -> bool {
    Path::new("/proc").join(pid.to_string()).exists()
}

fn locked(path: &Path, reason: impl Into<String>) -> PoseError {
    PoseError::Locked {
        path: path.to_path_buf(),
        reason: reason.into(),
    }
}

/// Refuses a directory where a file must be published. A commit that met one
/// would fail after it had already removed the earlier manifest.
fn reject_directory(path: &Path) -> Result<(), PoseError> {
    if path.is_dir() {
        return Err(io_error(
            path,
            io::Error::other("is a directory, so a published file cannot replace it"),
        ));
    }
    Ok(())
}

/// Refuses targets that a commit would replace but cannot: a directory where a
/// frame file or the manifest must go.
fn preflight_targets(dir: &Path) -> Result<(), PoseError> {
    for name in [CSV_FILE, JSONL_FILE, COMMIT_FILE] {
        reject_directory(&dir.join(name))?;
    }
    Ok(())
}

/// Refuses a directory with a per-step SVG name in the SVG directory. Other
/// entries there are not frame files and are left alone.
fn preflight_svg_entries(svg_dir: &Path) -> Result<(), PoseError> {
    for entry in fs::read_dir(svg_dir).map_err(|e| io_error(svg_dir, e))? {
        let entry = entry.map_err(|e| io_error(svg_dir, e))?;
        if entry.file_name().to_str().is_some_and(is_svg_frame_name) {
            reject_directory(&entry.path())?;
        }
    }
    Ok(())
}

/// Checks every path a commit of `steps` steps will rename onto. This runs
/// before the earlier manifest is removed, so a directory in the way leaves a
/// previously committed run intact.
fn preflight_publish(dir: &Path, steps: u64) -> Result<(), PoseError> {
    preflight_targets(dir)?;
    let svg_dir = dir.join(SVG_DIR);
    for step in 1..=steps {
        reject_directory(&svg_dir.join(svg_file_name(step)))?;
    }
    Ok(())
}

/// Proves that `dir` accepts a new file, by creating and removing a probe.
fn probe_writable(dir: &Path) -> Result<(), PoseError> {
    let probe = dir.join(format!(".pose-probe-{}", std::process::id()));
    let _ = fs::remove_file(&probe);
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&probe)
        .map_err(|e| io_error(&probe, e))?;
    fs::remove_file(&probe).map_err(|e| io_error(&probe, e))
}

/// Removes `path`, treating an absent file as already removed.
fn remove_if_present(path: &Path) -> Result<(), PoseError> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(io_error(path, e)),
    }
}

/// Moves a staged file to its final name, replacing any file already there.
fn publish(from: &Path, to: &Path) -> Result<(), PoseError> {
    fs::rename(from, to).map_err(|e| io_error(to, e))
}

/// Removes per-step SVG files whose step is past `steps`. Other files in the
/// directory are not frame files and are left alone.
fn remove_stale_svgs(svg_dir: &Path, steps: u64) -> Result<(), PoseError> {
    for entry in fs::read_dir(svg_dir).map_err(|e| io_error(svg_dir, e))? {
        let entry = entry.map_err(|e| io_error(svg_dir, e))?;
        let past_end = entry
            .file_name()
            .to_str()
            .and_then(svg_frame_step)
            .is_some_and(|step| step > steps);
        if past_end {
            let path = entry.path();
            fs::remove_file(&path).map_err(|e| io_error(&path, e))?;
        }
    }
    Ok(())
}

/// Verifies both staged files with the same strict line rules as the readers,
/// streaming, so a file the readers reject is never published and memory does
/// not grow with the run length. Each file must hold `steps` frames and end with
/// `last`.
fn check_staged(
    csv_path: &Path,
    jsonl_path: &Path,
    steps: u64,
    last: Option<&PoseFrame>,
) -> Result<(), PoseError> {
    let check = verify_pair(csv_path, jsonl_path, false)?;
    if check.steps != steps {
        return Err(malformed(
            0,
            format!("staged outputs do not hold the {steps} steps written"),
        ));
    }
    let matches_last = match (check.last.as_ref(), last) {
        (None, None) => true,
        (Some(staged), Some(pushed)) => same_frame(staged, pushed),
        _ => false,
    };
    if !matches_last {
        return Err(malformed(
            0,
            "the staged final frame is not the last frame pushed",
        ));
    }
    Ok(())
}

/// Reads a committed dump directory. It is accepted only when the
/// [`COMMIT_FILE`] manifest exists and matches. The CSV and JSON Lines files must
/// agree with each other, and every per-step SVG up to the committed step must be
/// present, with none past it. Nothing is returned unless every check passes.
pub fn load_committed(dir: &Path) -> Result<PoseLog, PoseError> {
    let manifest = dir.join(COMMIT_FILE);
    let text = match fs::read_to_string(&manifest) {
        Ok(text) => text,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            return Err(PoseError::NotCommitted {
                path: dir.to_path_buf(),
            })
        }
        Err(e) => return Err(io_error(&manifest, e)),
    };
    let steps = parse_commit(&text)?;
    let check = verify_pair(&dir.join(CSV_FILE), &dir.join(JSONL_FILE), true)?;
    if check.steps != steps {
        return Err(malformed(
            0,
            format!(
                "the commit manifest declares {steps} steps but the frames hold {}",
                check.steps
            ),
        ));
    }
    let svg_dir = dir.join(SVG_DIR);
    for step in 1..=steps {
        let path = svg_dir.join(svg_file_name(step));
        if !path.is_file() {
            return Err(malformed(0, format!("missing {}", path.display())));
        }
    }
    for entry in fs::read_dir(&svg_dir).map_err(|e| io_error(&svg_dir, e))? {
        let entry = entry.map_err(|e| io_error(&svg_dir, e))?;
        let past_end = entry
            .file_name()
            .to_str()
            .and_then(svg_frame_step)
            .is_some_and(|step| step > steps);
        if past_end {
            return Err(malformed(
                0,
                format!("{} is past the committed run", entry.path().display()),
            ));
        }
    }
    Ok(PoseLog {
        joints: check.joints,
        frames: check.frames,
    })
}

/// Parses the commit manifest: the schema marker, then `steps N`, with nothing else.
fn parse_commit(text: &str) -> Result<u64, PoseError> {
    if !text.ends_with('\n') {
        return Err(malformed(0, "missing final newline"));
    }
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() != 2 {
        return Err(malformed(
            0,
            format!("expected 2 lines, found {}", lines.len()),
        ));
    }
    if lines[0] != COMMIT_MARKER {
        return Err(malformed(
            1,
            format!("expected schema marker {COMMIT_MARKER:?}"),
        ));
    }
    let Some(value) = lines[1].strip_prefix("steps ") else {
        return Err(malformed(2, "expected `steps N`"));
    };
    parse_unsigned(value, 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commit_releases_lock_exactly_once() {
        let dir = std::env::temp_dir().join(format!(
            "robot-viewer-pose-{}-commit-once",
            std::process::id()
        ));
        let joints = vec!["j1".to_string()];

        // 1. Create dump
        let mut dump = FrameDump::create(&dir, &joints).unwrap();
        let staging = dump.staging.clone().unwrap();

        // 2. Commit
        dump.commit().unwrap();

        // 3. Verify the committed dump no longer owns the staging path.
        assert!(dump.staging.is_none());
        assert!(!staging.exists());

        // 4. A new writer acquires the released lock. Dropping the old dump
        // must not remove data owned by that writer.
        fs::create_dir(&staging).unwrap();
        let marker = staging.join("marker");
        fs::write(&marker, b"contender").unwrap();
        drop(dump);
        assert_eq!(fs::read(&marker).unwrap(), b"contender");

        // 5. Cleanup
        let _ = fs::remove_dir_all(&dir);
    }
    fn frame(step: u64, joints: usize) -> PoseFrame {
        PoseFrame {
            step,
            timestamp: Timestamp::sampling(step as i64 * 1_000, ClockId::Simulation),
            pelvis_position: [0.0, 0.0, 0.8],
            pelvis_quaternion: [1.0, 0.0, 0.0, 0.0],
            joint_angles: vec![0.1; joints],
        }
    }

    #[test]
    fn validate_rejects_a_wrong_joint_count() {
        assert!(matches!(
            frame(1, 2).validate(3),
            Err(PoseError::JointCount {
                step: 1,
                expected: 3,
                found: 2
            })
        ));
    }

    #[test]
    fn validate_rejects_a_non_finite_value() {
        let mut f = frame(4, 1);
        f.pelvis_position[1] = f64::NAN;
        assert!(matches!(
            f.validate(1),
            Err(PoseError::NotFinite { step: 4 })
        ));
    }

    #[test]
    fn joint_names_must_be_plain_ascii() {
        assert!(is_valid_joint_name("left_hip_pitch_joint"));
        assert!(!is_valid_joint_name(""));
        assert!(!is_valid_joint_name("a,b"));
        assert!(!is_valid_joint_name("a\"b"));
        assert!(!is_valid_joint_name("hip joint"));
    }

    #[test]
    fn svg_file_names_are_zero_padded() {
        assert_eq!(svg_file_name(1), "frame_000001.svg");
        assert_eq!(svg_file_name(20_000), "frame_020000.svg");
        assert!(is_svg_frame_name("frame_000123.svg"));
        assert!(!is_svg_frame_name("frame_12.svg"));
        assert!(!is_svg_frame_name("notes.txt"));
    }

    #[test]
    fn svg_names_stay_six_digits_and_sort_in_step_order_up_to_the_cap() {
        let last = svg_file_name(MAX_FRAME_STEPS);
        assert_eq!(last, "frame_999999.svg");
        assert!(is_svg_frame_name(&last));
        assert!(svg_file_name(9).as_str() < svg_file_name(10).as_str());
        assert!(svg_file_name(99_999).as_str() < svg_file_name(100_000).as_str());
    }

    #[test]
    fn push_refuses_a_step_past_the_six_digit_limit_before_writing() {
        let dir = std::env::temp_dir().join(format!("pose-cap-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let joints = vec!["hip".to_owned()];
        let mut dump = FrameDump::create(&dir, &joints).unwrap();
        let frame = PoseFrame {
            step: MAX_FRAME_STEPS + 1,
            timestamp: Timestamp::sampling(0, ClockId::Simulation),
            pelvis_position: [0.0, 0.0, 0.79],
            pelvis_quaternion: [1.0, 0.0, 0.0, 0.0],
            joint_angles: vec![0.0],
        };
        assert!(matches!(
            dump.push(&frame),
            Err(PoseError::TooManySteps { .. })
        ));
        drop(dump);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn create_refuses_more_joints_than_a_frame_file_allows() {
        let dir = std::env::temp_dir().join(format!("pose-joints-{}", std::process::id()));
        let joints: Vec<String> = (0..=MAX_JOINTS).map(|i| format!("j{i}")).collect();
        assert!(matches!(
            FrameDump::create(&dir, &joints),
            Err(PoseError::TooManyJoints { .. })
        ));
        assert!(!dir.exists());
    }

    #[test]
    fn finite_float_extremes_round_trip_through_a_committed_dump() {
        let dir = std::env::temp_dir().join(format!("pose-float-bounds-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let joints = vec![
            "a".to_owned(),
            "b".to_owned(),
            "c".to_owned(),
            "d".to_owned(),
        ];
        let extremes = [f64::MAX, -f64::MIN_POSITIVE, f64::from_bits(1)];
        let mut next_extreme = extremes.iter().cycle();
        let pelvis_position = std::array::from_fn(|_| *next_extreme.next().unwrap());
        let pelvis_quaternion = std::array::from_fn(|_| *next_extreme.next().unwrap());
        let joint_angles = (0..joints.len())
            .map(|_| *next_extreme.next().unwrap())
            .collect();
        let frame = PoseFrame {
            step: 1,
            timestamp: Timestamp::sampling(1_000, ClockId::Simulation),
            pelvis_position,
            pelvis_quaternion,
            joint_angles,
        };

        let mut dump = FrameDump::create(&dir, &joints).unwrap();
        dump.push(&frame).unwrap();
        assert_eq!(dump.finish().unwrap(), 1);

        let loaded = load_committed(&dir).unwrap();
        assert_eq!(loaded.joints, joints);
        assert_eq!(loaded.frames, vec![frame]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn create_rejects_a_valid_joint_name_set_with_an_oversized_header_before_creating_dir() {
        let dir = std::env::temp_dir().join(format!("pose-header-limit-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let joints: Vec<String> = (0..MAX_JOINTS).map(|_| "a".repeat(300)).collect();

        assert!(matches!(
            FrameDump::create(&dir, &joints),
            Err(PoseError::HeaderTooLong { .. })
        ));
        assert!(!dir.exists());
    }

    #[test]
    fn a_line_past_the_joint_scaled_limit_is_rejected_before_it_is_buffered() {
        let mut lines = LineReader::new(std::io::Cursor::new(vec![b'x'; 100]), None);
        lines.set_limit(16);
        assert!(matches!(
            lines.next_line(),
            Err(PoseError::Malformed { line: 1, .. })
        ));
    }

    fn write_pair(name: &str, csv: &str, jsonl: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("robot-viewer-pose-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(CSV_FILE), csv).unwrap();
        fs::write(dir.join(JSONL_FILE), jsonl).unwrap();
        dir
    }

    fn encode_pair(frames: &[PoseFrame], joints: &[String]) -> (String, String) {
        let mut csv = Vec::new();
        csv::write_header(&mut csv, joints).unwrap();
        let mut jsonl = Vec::new();
        jsonl::write_header(&mut jsonl, joints).unwrap();
        for f in frames {
            csv::write_row(&mut csv, f).unwrap();
            jsonl::write_row(&mut jsonl, f).unwrap();
        }
        csv::write_footer(&mut csv, frames.len() as u64).unwrap();
        jsonl::write_footer(&mut jsonl, frames.len() as u64).unwrap();
        (
            String::from_utf8(csv).unwrap(),
            String::from_utf8(jsonl).unwrap(),
        )
    }

    #[test]
    fn verify_pair_accepts_matching_files_and_counts_steps() {
        let joints = vec!["hip".to_owned()];
        let frames = [frame(1, 1), frame(2, 1), frame(3, 1)];
        let (csv_text, jsonl_text) = encode_pair(&frames, &joints);
        let dir = write_pair("pair-ok", &csv_text, &jsonl_text);
        let check = verify_pair(&dir.join(CSV_FILE), &dir.join(JSONL_FILE), true).unwrap();
        assert_eq!(check.steps, 3);
        assert_eq!(check.frames, frames.to_vec());
        assert_eq!(check.joints, joints);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn verify_pair_rejects_a_single_differing_angle() {
        let joints = vec!["hip".to_owned()];
        let (csv_text, jsonl_text) = encode_pair(&[frame(1, 1), frame(2, 1)], &joints);
        let jsonl_text = jsonl_text.replacen("0.1", "0.2", 2);
        let dir = write_pair("pair-angle", &csv_text, &jsonl_text);
        assert!(matches!(
            verify_pair(&dir.join(CSV_FILE), &dir.join(JSONL_FILE), false),
            Err(PoseError::Malformed { .. })
        ));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn verify_pair_distinguishes_negative_zero_from_zero() {
        let joints = vec!["hip".to_owned()];
        let mut negative = frame(1, 1);
        negative.joint_angles[0] = -0.0;
        let mut positive = frame(1, 1);
        positive.joint_angles[0] = 0.0;
        let (csv_text, _) = encode_pair(&[negative], &joints);
        let (_, jsonl_text) = encode_pair(&[positive], &joints);
        assert!(csv_text.lines().any(|l| l.ends_with(",-0")));
        let dir = write_pair("pair-zero", &csv_text, &jsonl_text);
        assert!(verify_pair(&dir.join(CSV_FILE), &dir.join(JSONL_FILE), false).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn verify_pair_rejects_different_step_counts_without_keeping_frames() {
        let joints = vec!["hip".to_owned()];
        let (csv_text, _) = encode_pair(&[frame(1, 1), frame(2, 1)], &joints);
        let (_, jsonl_text) = encode_pair(&[frame(1, 1)], &joints);
        let dir = write_pair("pair-count", &csv_text, &jsonl_text);
        assert!(verify_pair(&dir.join(CSV_FILE), &dir.join(JSONL_FILE), false).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn verify_pair_rejects_a_malformed_csv_row_even_when_jsonl_is_valid() {
        let joints = vec!["hip".to_owned()];
        let (csv_text, jsonl_text) = encode_pair(&[frame(1, 1)], &joints);
        let csv_text = csv_text.replace("# end steps=1", "2,2000,0,0,0.8,1,0,0,0\n# end steps=1");
        let dir = write_pair("pair-bad-csv", &csv_text, &jsonl_text);
        assert!(verify_pair(&dir.join(CSV_FILE), &dir.join(JSONL_FILE), false).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn csv_reader_rejects_a_timestamp_above_the_exact_integer_bound() {
        let joints = vec!["hip".to_owned()];
        let (mut csv_text, _) = encode_pair(&[frame(1, 1)], &joints);
        csv_text = csv_text.replace("1000", "9007199254740993");
        assert!(matches!(
            csv::parse(&csv_text),
            Err(PoseError::Malformed { .. })
        ));
    }

    #[test]
    fn jsonl_reader_rejects_a_timestamp_above_the_exact_integer_bound() {
        let joints = vec!["hip".to_owned()];
        let (_, jsonl_text) = encode_pair(&[frame(1, 1)], &joints);
        let jsonl_text = jsonl_text.replace("\"time_ns\":1000", "\"time_ns\":9007199254740993");
        assert!(matches!(
            jsonl::parse(&jsonl_text),
            Err(PoseError::Malformed { .. })
        ));
    }

    #[test]
    fn jsonl_reader_rejects_fractional_and_exponent_timestamp_tokens() {
        let joints = vec!["hip".to_owned()];
        let (_, jsonl_text) = encode_pair(&[frame(1, 1)], &joints);
        for invalid in ["1.5e3", "1500.0"] {
            let text = jsonl_text.replace("\"time_ns\":1000", &format!("\"time_ns\":{invalid}"));
            assert!(matches!(
                jsonl::parse(&text),
                Err(PoseError::Malformed { .. })
            ));
        }
    }

    #[test]
    fn dump_rejects_an_out_of_order_step_before_writing_it() {
        let dir =
            std::env::temp_dir().join(format!("robot-viewer-pose-{}-order", std::process::id()));
        let joints = vec!["hip".to_owned()];
        let mut dump = FrameDump::create(&dir, &joints).unwrap();
        assert!(matches!(
            dump.push(&frame(2, 1)),
            Err(PoseError::OutOfSequence {
                expected: 1,
                found: 2
            })
        ));
        drop(dump);
        let _ = fs::remove_dir_all(&dir);
    }

    fn rejected_push(name: &str, frame: PoseFrame) -> PoseError {
        let dir = std::env::temp_dir().join(format!(
            "robot-viewer-pose-{}-timestamp-{name}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        let joints = vec!["j1".to_string()];
        let mut dump = FrameDump::create(&dir, &joints).unwrap();
        let error = dump.push(&frame).unwrap_err();
        assert_eq!(dump.finish().unwrap(), 0);
        assert!(load_committed(&dir).unwrap().frames.is_empty());
        let _ = fs::remove_dir_all(&dir);
        error
    }

    #[test]
    fn push_rejects_a_wrong_clock_without_writing() {
        let mut frame = frame(1, 1);
        frame.timestamp = Timestamp::sampling(1, ClockId::HostMonotonic);
        assert!(matches!(
            rejected_push("wrong-clock", frame),
            PoseError::InvalidClock { .. }
        ));
    }

    #[test]
    fn push_rejects_negative_and_out_of_range_timestamps_without_writing() {
        for (name, nanos) in [("negative", -1), ("too-large", MAX_TIMESTAMP_NS + 1)] {
            let mut frame = frame(1, 1);
            frame.timestamp = Timestamp::sampling(nanos, ClockId::Simulation);
            assert!(matches!(
                rejected_push(name, frame),
                PoseError::TimestampOutOfBounds { .. }
            ));
        }
    }

    #[test]
    fn push_rejects_non_increasing_timestamp_without_writing_the_bad_frame() {
        let dir = std::env::temp_dir().join(format!(
            "robot-viewer-pose-{}-timestamp-nonincreasing",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        let joints = vec!["j1".to_string()];
        let mut dump = FrameDump::create(&dir, &joints).unwrap();
        let mut first = frame(1, 1);
        first.timestamp = Timestamp::sampling(100, ClockId::Simulation);
        dump.push(&first).unwrap();
        let mut second = frame(2, 1);
        second.timestamp = Timestamp::sampling(50, ClockId::Simulation);
        assert!(matches!(
            dump.push(&second),
            Err(PoseError::TimeNotIncreasing { .. })
        ));
        assert_eq!(dump.finish().unwrap(), 1);
        let log = load_committed(&dir).unwrap();
        assert_eq!(log.frames, vec![first]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn push_accepts_the_exact_timestamp_bound() {
        let dir = std::env::temp_dir().join(format!(
            "robot-viewer-pose-{}-timestamp-boundary",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        let joints = vec!["j1".to_string()];
        let mut dump = FrameDump::create(&dir, &joints).unwrap();
        let mut frame = frame(1, 1);
        frame.timestamp = Timestamp::sampling(MAX_TIMESTAMP_NS, ClockId::Simulation);
        dump.push(&frame).unwrap();
        assert_eq!(dump.finish().unwrap(), 1);
        assert_eq!(load_committed(&dir).unwrap().frames, vec![frame]);
        let _ = fs::remove_dir_all(&dir);
    }
}

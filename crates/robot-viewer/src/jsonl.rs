//! JSON Lines form of the pose frames.
//!
//! The first line is a header object, each following line is one step, and
//! the final line is a footer object with the step count.
//! Layout is fixed by `docs/decisions/0009-test-render-format.md`:
//!
//! ```text
//! {"schema":"robot-viewer.pose-frames","schema_version":1,"joints":["hip"]}
//! {"step":1,"time_ns":100000,"pelvis_position":[x,y,z],"pelvis_quaternion":[w,x,y,z],"joint_angles":[a]}
//! {"end":true,"steps":1}
//! ```
//!
//! The reader is a small std-only parser that accepts only the value shapes
//! this format writes. Keys must appear in the documented order, and numbers
//! must use JSON number syntax and be finite.

use crate::pose::{
    self, collect_log, malformed, FrameFormat, FrameStream, LineReader, PoseError, PoseFrame,
    PoseLog, Record,
};
use robot_types::time::{ClockId, Timestamp};
use std::io::{self, BufRead, Write};
use std::path::Path;

/// The `schema` value written in the header line.
pub const SCHEMA: &str = "robot-viewer.pose-frames";
/// The `schema_version` value written in the header line.
pub const SCHEMA_VERSION: u64 = 1;

/// Writes the header line for `joints`.
pub fn write_header<W: Write>(out: &mut W, joints: &[String]) -> io::Result<()> {
    write!(
        out,
        "{{\"schema\":\"{SCHEMA}\",\"schema_version\":{SCHEMA_VERSION},\"joints\":["
    )?;
    for (i, name) in joints.iter().enumerate() {
        if i > 0 {
            write!(out, ",")?;
        }
        write!(out, "\"{name}\"")?;
    }
    writeln!(out, "]}}")
}

/// Writes one step as a line.
pub fn write_row<W: Write>(out: &mut W, frame: &PoseFrame) -> io::Result<()> {
    write!(
        out,
        "{{\"step\":{},\"time_ns\":{},\"pelvis_position\":[{},{},{}],\"pelvis_quaternion\":[{},{},{},{}],\"joint_angles\":[",
        frame.step,
        frame.timestamp.nanos(),
        frame.pelvis_position[0],
        frame.pelvis_position[1],
        frame.pelvis_position[2],
        frame.pelvis_quaternion[0],
        frame.pelvis_quaternion[1],
        frame.pelvis_quaternion[2],
        frame.pelvis_quaternion[3],
    )?;
    for (i, angle) in frame.joint_angles.iter().enumerate() {
        if i > 0 {
            write!(out, ",")?;
        }
        write!(out, "{angle}")?;
    }
    writeln!(out, "]}}")
}

/// Writes the footer line with the total step count.
pub fn write_footer<W: Write>(out: &mut W, steps: u64) -> io::Result<()> {
    writeln!(out, "{{\"end\":true,\"steps\":{steps}}}")
}

/// A parsed JSON value, limited to what this format uses.
#[derive(Debug, PartialEq)]
enum Value<'a> {
    Bool(bool),
    Number(f64, &'a str),
    Text(&'a str),
    Array(Vec<Value<'a>>),
    Object(Vec<(&'a str, Value<'a>)>),
}

/// Deepest nesting a line may use. The format needs two levels (an object
/// holding arrays); anything deeper is rejected before the recursion grows.
const MAX_DEPTH: usize = 8;

/// A cursor over one line.
struct Cursor<'a> {
    text: &'a str,
    pos: usize,
    line: usize,
    depth: usize,
}

impl<'a> Cursor<'a> {
    fn peek(&self) -> Option<u8> {
        self.text.as_bytes().get(self.pos).copied()
    }

    /// Counts one more open container, failing past [`MAX_DEPTH`].
    fn enter(&mut self) -> Result<(), PoseError> {
        if self.depth == MAX_DEPTH {
            return Err(malformed(
                self.line,
                format!("nesting is deeper than {MAX_DEPTH} levels"),
            ));
        }
        self.depth += 1;
        Ok(())
    }

    fn expect(&mut self, byte: u8) -> Result<(), PoseError> {
        if self.peek() == Some(byte) {
            self.pos += 1;
            Ok(())
        } else {
            Err(malformed(self.line, format!("expected {:?}", byte as char)))
        }
    }

    /// Reads a string with no escapes. Escapes are rejected, since joint names
    /// and keys never need them.
    fn string(&mut self) -> Result<&'a str, PoseError> {
        self.expect(b'"')?;
        let start = self.pos;
        while let Some(b) = self.peek() {
            match b {
                b'"' => {
                    let s = &self.text[start..self.pos];
                    self.pos += 1;
                    return Ok(s);
                }
                b'\\' => return Err(malformed(self.line, "escape sequences are not allowed")),
                _ => self.pos += 1,
            }
        }
        Err(malformed(self.line, "unterminated string"))
    }

    /// Reads a JSON number token and requires it to be finite.
    fn number(&mut self) -> Result<(f64, &'a str), PoseError> {
        let start = self.pos;
        while let Some(b) = self.peek() {
            if b.is_ascii_digit() || matches!(b, b'+' | b'-' | b'.' | b'e' | b'E') {
                self.pos += 1;
            } else {
                break;
            }
        }
        let token = &self.text[start..self.pos];
        if !is_json_number(token) {
            return Err(malformed(self.line, "invalid JSON number"));
        }
        Ok((pose::parse_float(token, self.line)?, token))
    }

    fn value(&mut self) -> Result<Value<'a>, PoseError> {
        match self.peek() {
            Some(b'"') => Ok(Value::Text(self.string()?)),
            Some(b'[') => {
                self.enter()?;
                self.pos += 1;
                let mut items = Vec::new();
                if self.peek() == Some(b']') {
                    self.pos += 1;
                    self.depth -= 1;
                    return Ok(Value::Array(items));
                }
                loop {
                    items.push(self.value()?);
                    match self.peek() {
                        Some(b',') => self.pos += 1,
                        Some(b']') => {
                            self.pos += 1;
                            self.depth -= 1;
                            return Ok(Value::Array(items));
                        }
                        _ => return Err(malformed(self.line, "expected ',' or ']'")),
                    }
                }
            }
            Some(b'{') => {
                self.enter()?;
                self.pos += 1;
                let mut members = Vec::new();
                if self.peek() == Some(b'}') {
                    self.pos += 1;
                    self.depth -= 1;
                    return Ok(Value::Object(members));
                }
                loop {
                    let key = self.string()?;
                    self.expect(b':')?;
                    members.push((key, self.value()?));
                    match self.peek() {
                        Some(b',') => self.pos += 1,
                        Some(b'}') => {
                            self.pos += 1;
                            self.depth -= 1;
                            return Ok(Value::Object(members));
                        }
                        _ => return Err(malformed(self.line, "expected ',' or '}'")),
                    }
                }
            }
            Some(b't') => {
                if self.text[self.pos..].starts_with("true") {
                    self.pos += 4;
                    Ok(Value::Bool(true))
                } else {
                    Err(malformed(self.line, "expected 'true'"))
                }
            }
            Some(b'f') => {
                if self.text[self.pos..].starts_with("false") {
                    self.pos += 5;
                    Ok(Value::Bool(false))
                } else {
                    Err(malformed(self.line, "expected 'false'"))
                }
            }
            Some(_) => {
                let (number, token) = self.number()?;
                Ok(Value::Number(number, token))
            }
            None => Err(malformed(self.line, "unexpected end of line")),
        }
    }
}

/// Returns whether `token` follows the JSON number grammar from RFC 8259.
fn is_json_number(token: &str) -> bool {
    let bytes = token.as_bytes();
    let mut pos = 0;

    if bytes.get(pos) == Some(&b'-') {
        pos += 1;
    }

    match bytes.get(pos) {
        Some(b'0') => pos += 1,
        Some(b'1'..=b'9') => {
            pos += 1;
            while bytes.get(pos).is_some_and(u8::is_ascii_digit) {
                pos += 1;
            }
        }
        _ => return false,
    }

    if bytes.get(pos) == Some(&b'.') {
        pos += 1;
        let fraction_start = pos;
        while bytes.get(pos).is_some_and(u8::is_ascii_digit) {
            pos += 1;
        }
        if pos == fraction_start {
            return false;
        }
    }

    if matches!(bytes.get(pos), Some(b'e' | b'E')) {
        pos += 1;
        if matches!(bytes.get(pos), Some(b'+' | b'-')) {
            pos += 1;
        }
        let exponent_start = pos;
        while bytes.get(pos).is_some_and(u8::is_ascii_digit) {
            pos += 1;
        }
        if pos == exponent_start {
            return false;
        }
    }

    pos == bytes.len()
}

/// Parses one whole line into a value, rejecting trailing bytes.
fn parse_line(text: &str, line: usize) -> Result<Value<'_>, PoseError> {
    let mut cursor = Cursor {
        text,
        pos: 0,
        line,
        depth: 0,
    };
    let value = cursor.value()?;
    if cursor.pos != text.len() {
        return Err(malformed(line, "trailing characters after value"));
    }
    Ok(value)
}

/// Returns the members of an object whose keys are exactly `keys`, in order.
fn members<'v, 'a>(
    value: &'v Value<'a>,
    keys: &[&str],
    line: usize,
) -> Result<&'v [(&'a str, Value<'a>)], PoseError> {
    let Value::Object(members) = value else {
        return Err(malformed(line, "expected an object"));
    };
    let found: Vec<&str> = members.iter().map(|(k, _)| *k).collect();
    if found != keys {
        return Err(malformed(
            line,
            format!("keys {found:?} do not match the expected {keys:?}"),
        ));
    }
    Ok(members)
}

fn number_array(value: &Value<'_>, len: usize, line: usize) -> Result<Vec<f64>, PoseError> {
    let Value::Array(items) = value else {
        return Err(malformed(line, "expected an array of numbers"));
    };
    if items.len() != len {
        return Err(malformed(
            line,
            format!("array has {} entries, expected {len}", items.len()),
        ));
    }
    items
        .iter()
        .map(|item| match item {
            Value::Number(n, _) => Ok(*n),
            _ => Err(malformed(line, "array entry is not a number")),
        })
        .collect()
}

fn unsigned(value: &Value<'_>, line: usize) -> Result<u64, PoseError> {
    match value {
        Value::Number(_, token) => parse_json_unsigned(token, line),
        _ => Err(malformed(line, "expected a non-negative integer")),
    }
}

fn parse_json_unsigned(token: &str, line: usize) -> Result<u64, PoseError> {
    if token.is_empty() || !token.bytes().all(|b| b.is_ascii_digit()) {
        return Err(malformed(line, "expected unsigned integer"));
    }
    if token.len() > 1 && token.starts_with('0') {
        return Err(malformed(line, "leading zero"));
    }
    token
        .parse::<u64>()
        .map_err(|_| malformed(line, "integer out of range"))
}

/// The JSON Lines line rules, applied by the shared streaming reader in [`pose`].
pub(crate) struct Jsonl;

const STEP_KEYS: [&str; 5] = [
    "step",
    "time_ns",
    "pelvis_position",
    "pelvis_quaternion",
    "joint_angles",
];

impl FrameFormat for Jsonl {
    fn open<R: BufRead>(lines: &mut LineReader<R>) -> Result<Vec<String>, PoseError> {
        let (header_line, header_text) = lines
            .next_line()?
            .ok_or_else(|| malformed(1, "missing header line"))?;
        let header = parse_line(header_text, header_line)?;
        let header_members = members(
            &header,
            &["schema", "schema_version", "joints"],
            header_line,
        )?;
        let Value::Text(schema) = header_members[0].1 else {
            return Err(malformed(header_line, "schema is not a string"));
        };
        if schema != SCHEMA {
            return Err(malformed(
                header_line,
                format!("schema {schema:?} is not {SCHEMA:?}"),
            ));
        }
        let version = unsigned(&header_members[1].1, header_line)?;
        if version != SCHEMA_VERSION {
            return Err(malformed(
                header_line,
                format!("schema_version {version} is not {SCHEMA_VERSION}"),
            ));
        }
        let Value::Array(names) = &header_members[2].1 else {
            return Err(malformed(header_line, "joints is not an array"));
        };
        let mut joints = Vec::with_capacity(names.len());
        for name in names {
            let Value::Text(name) = name else {
                return Err(malformed(header_line, "joint name is not a string"));
            };
            if !pose::is_valid_joint_name(name) {
                return Err(malformed(
                    header_line,
                    format!("invalid joint name {name:?}"),
                ));
            }
            joints.push((*name).to_owned());
        }
        Ok(joints)
    }

    fn decode(line: usize, text: &str, joints: &[String]) -> Result<Record, PoseError> {
        let value = parse_line(text, line)?;
        let Value::Object(ref members_list) = value else {
            return Err(malformed(line, "expected an object"));
        };
        let first_key = members_list.first().map(|(k, _)| *k);
        if first_key == Some("end") {
            let footer_members = members(&value, &["end", "steps"], line)?;
            let Value::Bool(end_flag) = footer_members[0].1 else {
                return Err(malformed(line, "end is not a boolean"));
            };
            if !end_flag {
                return Err(malformed(line, "end must be true"));
            }
            return Ok(Record::Footer(unsigned(&footer_members[1].1, line)?));
        }

        let fields = members(&value, &STEP_KEYS, line)?;
        let step = unsigned(&fields[0].1, line)?;
        let time = unsigned(&fields[1].1, line)?;
        let time_ns = i64::try_from(time).map_err(|_| malformed(line, "time_ns out of range"))?;
        let position = number_array(&fields[2].1, 3, line)?;
        let quaternion = number_array(&fields[3].1, 4, line)?;
        let angles = number_array(&fields[4].1, joints.len(), line)?;
        Ok(Record::Frame(PoseFrame {
            step,
            timestamp: Timestamp::sampling(time_ns, ClockId::Simulation),
            pelvis_position: [position[0], position[1], position[2]],
            pelvis_quaternion: [quaternion[0], quaternion[1], quaternion[2], quaternion[3]],
            joint_angles: angles,
        }))
    }
}

/// Parses JSON Lines text written by [`write_header`], [`write_row`], and [`write_footer`].
///
/// Rejects a missing final newline, a header with the wrong schema or version,
/// any object whose keys differ from the documented ones or are out of order,
/// any wrong array length, any non-finite number, steps that are not 1, 2, 3, ...,
/// times that are not strictly increasing, a missing or mismatched end footer,
/// or any trailing records after the footer. Nothing is returned unless every
/// line passes.
pub fn parse(text: &str) -> Result<PoseLog, PoseError> {
    collect_log(FrameStream::<&[u8], Jsonl>::open(text.as_bytes(), None)?)
}

/// Reads and parses a JSON Lines file. The whole file is read before parsing.
pub fn read_file(path: &Path) -> Result<PoseLog, PoseError> {
    let text = std::fs::read_to_string(path).map_err(|source| PoseError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    parse(&text)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn joints() -> Vec<String> {
        vec!["hip".to_owned(), "knee".to_owned()]
    }

    fn frame(step: u64) -> PoseFrame {
        PoseFrame {
            step,
            timestamp: Timestamp::sampling(step as i64 * 100_000, ClockId::Simulation),
            pelvis_position: [0.0, 0.0, 0.79],
            pelvis_quaternion: [1.0, 0.0, 0.0, 0.0],
            joint_angles: vec![0.01, -0.25],
        }
    }

    fn encode(frames: &[PoseFrame]) -> String {
        let mut out = Vec::new();
        write_header(&mut out, &joints()).unwrap();
        for f in frames {
            write_row(&mut out, f).unwrap();
        }
        write_footer(&mut out, frames.len() as u64).unwrap();
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn header_line_is_the_documented_one() {
        let text = encode(&[]);
        assert_eq!(
            text,
            "{\"schema\":\"robot-viewer.pose-frames\",\"schema_version\":1,\"joints\":[\"hip\",\"knee\"]}\n{\"end\":true,\"steps\":0}\n"
        );
    }

    #[test]
    fn round_trips_exactly() {
        let frames = vec![frame(1), frame(2), frame(3)];
        let log = parse(&encode(&frames)).unwrap();
        assert_eq!(log.joints, joints());
        assert_eq!(log.frames, frames);
    }

    #[test]
    fn rejects_a_wrong_schema_name() {
        let text = encode(&[frame(1)]).replace("robot-viewer.pose-frames", "other");
        assert!(matches!(
            parse(&text),
            Err(PoseError::Malformed { line: 1, .. })
        ));
    }

    #[test]
    fn rejects_a_wrong_schema_version() {
        let text = encode(&[frame(1)]).replace("\"schema_version\":1", "\"schema_version\":2");
        assert!(matches!(
            parse(&text),
            Err(PoseError::Malformed { line: 1, .. })
        ));
    }

    #[test]
    fn rejects_nesting_past_the_depth_limit_without_recursing_further() {
        let deep = format!("{}{}", "[".repeat(MAX_DEPTH + 1), "]".repeat(MAX_DEPTH + 1));
        assert!(matches!(
            parse_line(&deep, 4),
            Err(PoseError::Malformed { line: 4, .. })
        ));
        let at_limit = format!("{}{}", "[".repeat(MAX_DEPTH), "]".repeat(MAX_DEPTH));
        assert!(parse_line(&at_limit, 4).is_ok());
    }

    #[test]
    fn rejects_a_truncated_line_without_partial_load() {
        let text = encode(&[frame(1), frame(2)]);
        let cut = &text[..text.len() - 5];
        assert!(parse(cut).is_err());
    }

    #[test]
    fn rejects_a_line_boundary_truncation_missing_footer() {
        let text = encode(&[frame(1), frame(2)]);
        let lines: Vec<&str> = text.lines().collect();
        let cut = lines[..lines.len() - 1].join("\n") + "\n";
        assert!(matches!(
            parse(&cut),
            Err(PoseError::Malformed { line: 0, reason }) if reason.contains("missing end footer")
        ));
    }

    #[test]
    fn rejects_a_footer_with_wrong_step_count() {
        let text = encode(&[frame(1), frame(2)]).replace("\"steps\":2", "\"steps\":3");
        assert!(matches!(
            parse(&text),
            Err(PoseError::Malformed { line: 4, .. })
        ));
    }

    #[test]
    fn rejects_extra_records_after_footer() {
        let mut text = encode(&[frame(1)]);
        text.push_str("{\"step\":2,\"time_ns\":200000,\"pelvis_position\":[0.0,0.0,0.79],\"pelvis_quaternion\":[1.0,0.0,0.0,0.0],\"joint_angles\":[0.01,-0.25]}\n");
        assert!(matches!(
            parse(&text),
            Err(PoseError::Malformed { line: 4, .. })
        ));
    }

    #[test]
    fn rejects_a_missing_final_newline() {
        let text = encode(&[frame(1)]);
        assert!(parse(&text[..text.len() - 1]).is_err());
    }

    #[test]
    fn rejects_reordered_keys() {
        let text = encode(&[frame(1)]).replace(
            "\"step\":1,\"time_ns\"",
            "\"time_ns\":100000,\"step\":1,\"x\"",
        );
        assert!(matches!(
            parse(&text),
            Err(PoseError::Malformed { line: 2, .. })
        ));
    }

    #[test]
    fn rejects_an_angle_count_that_disagrees_with_the_header() {
        let text = encode(&[frame(1)]).replace("[0.01,-0.25]", "[0.01]");
        assert!(matches!(
            parse(&text),
            Err(PoseError::Malformed { line: 2, .. })
        ));
    }

    #[test]
    fn rejects_nan_and_infinity() {
        let good = encode(&[frame(1)]);
        assert!(parse(&good.replace("0.79", "NaN")).is_err());
        assert!(parse(&good.replace("0.79", "Infinity")).is_err());
    }

    #[test]
    fn rejects_non_json_number_syntax() {
        let good = encode(&[frame(1)]);
        for invalid in ["01.5", "+1", ".5", "1."] {
            assert!(
                parse(&good.replace("0.79", invalid)).is_err(),
                "accepted invalid JSON number {invalid:?}"
            );
        }
    }

    #[test]
    fn accepts_json_exponent_number_syntax() {
        let good = encode(&[frame(1)]);
        for exponent in ["7.9e-1", "7.9E-1"] {
            let parsed = parse(&good.replace("0.79", exponent)).unwrap();
            assert_eq!(parsed.frames[0].pelvis_position[2], 0.79);
        }
    }

    #[test]
    fn rejects_a_skipped_step() {
        let text = encode(&[frame(1), frame(3)]);
        assert!(matches!(
            parse(&text),
            Err(PoseError::Malformed { line: 3, .. })
        ));
    }

    #[test]
    fn rejects_trailing_bytes_after_a_row() {
        let good = encode(&[frame(1)]);
        let text = good.replace("{\"end\":true", "x{\"end\":true");
        assert!(matches!(
            parse(&text),
            Err(PoseError::Malformed { line: 3, .. })
        ));
    }

    #[test]
    fn rejects_leading_zero_integers() {
        let good = encode(&[frame(1)]);
        assert!(parse(&good.replace("\"step\":1", "\"step\":01")).is_err());
        assert!(parse(&good.replace("\"time_ns\":100000", "\"time_ns\":0100000")).is_err());
        assert!(parse(&good.replace("\"schema_version\":1", "\"schema_version\":01")).is_err());
        assert!(parse(&good.replace("\"steps\":1", "\"steps\":01")).is_err());
    }

    #[test]
    fn accepts_zero_footer() {
        let text = "{\"schema\":\"robot-viewer.pose-frames\",\"schema_version\":1,\"joints\":[]}\n{\"end\":true,\"steps\":0}\n";
        assert!(parse(text).is_ok());
    }
}

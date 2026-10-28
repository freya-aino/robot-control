//! CSV form of the pose frames: a schema marker, one header row, one row per
//! step, and an end footer with the step count.
//!
//! Layout is fixed by `docs/decisions/0009-test-render-format.md`:
//!
//! ```text
//! # robot-viewer.pose-frames v1
//! step,time_ns,pelvis_x,pelvis_y,pelvis_z,pelvis_qw,pelvis_qx,pelvis_qy,pelvis_qz,<joints...>
//! 1,100000,0,0,0.79,1,0,0,0,0.01,-0.25
//! # end steps=1
//! ```

use crate::pose::{
    self, collect_log, malformed, parse_float, parse_unsigned, FrameFormat, FrameStream,
    LineReader, PoseError, PoseFrame, PoseLog, Record,
};
use robot_types::time::{ClockId, Timestamp};
use std::io::{self, BufRead, Write};
use std::path::Path;

/// The schema marker line written at the start of the CSV file.
pub const SCHEMA_MARKER: &str = "# robot-viewer.pose-frames v1";

/// The prefix written before the step count in the CSV footer line.
pub const FOOTER_PREFIX: &str = "# end steps=";

const FIXED_COLUMNS: [&str; 9] = [
    "step",
    "time_ns",
    "pelvis_x",
    "pelvis_y",
    "pelvis_z",
    "pelvis_qw",
    "pelvis_qx",
    "pelvis_qy",
    "pelvis_qz",
];

/// Writes the schema marker and header row for `joints`.
pub fn write_header<W: Write>(out: &mut W, joints: &[String]) -> io::Result<()> {
    writeln!(out, "{SCHEMA_MARKER}")?;
    write!(out, "{}", FIXED_COLUMNS.join(","))?;
    for name in joints {
        write!(out, ",{name}")?;
    }
    writeln!(out)
}

/// Writes one row. Floats use `Display`, which is the shortest form that
/// round-trips, and never uses exponent notation for the values this format
/// holds. Non-finite values are refused by [`PoseFrame::validate`] before this.
pub fn write_row<W: Write>(out: &mut W, frame: &PoseFrame) -> io::Result<()> {
    write!(
        out,
        "{},{},{},{},{},{},{},{},{}",
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
    for angle in &frame.joint_angles {
        write!(out, ",{angle}")?;
    }
    writeln!(out)
}

/// Writes the footer line with the total step count.
pub fn write_footer<W: Write>(out: &mut W, steps: u64) -> io::Result<()> {
    writeln!(out, "{FOOTER_PREFIX}{steps}")
}

/// The CSV line rules, applied by the shared streaming reader in [`pose`].
pub(crate) struct Csv;

impl FrameFormat for Csv {
    fn open<R: BufRead>(lines: &mut LineReader<R>) -> Result<Vec<String>, PoseError> {
        let (marker_line, marker) = lines
            .next_line()?
            .ok_or_else(|| malformed(1, "missing schema marker line"))?;
        if marker != SCHEMA_MARKER {
            return Err(malformed(
                marker_line,
                format!("expected schema marker {SCHEMA_MARKER:?}, found {marker:?}"),
            ));
        }
        let (header_line, header) = lines
            .next_line()?
            .ok_or_else(|| malformed(2, "missing column header row"))?;
        let fields: Vec<&str> = header.split(',').collect();
        if fields.len() < FIXED_COLUMNS.len() || fields[..FIXED_COLUMNS.len()] != FIXED_COLUMNS {
            return Err(malformed(
                header_line,
                "header does not start with the fixed columns",
            ));
        }
        let joints: Vec<String> = fields[FIXED_COLUMNS.len()..]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        if let Some(bad) = joints.iter().find(|n| !pose::is_valid_joint_name(n)) {
            return Err(malformed(
                header_line,
                format!("invalid joint name {bad:?}"),
            ));
        }
        Ok(joints)
    }

    fn decode(line: usize, text: &str, joints: &[String]) -> Result<Record, PoseError> {
        if let Some(rest) = text.strip_prefix(FOOTER_PREFIX) {
            return Ok(Record::Footer(parse_unsigned(rest, line)?));
        }
        if text.starts_with('#') {
            return Err(malformed(line, "unrecognized comment or metadata line"));
        }
        let parts: Vec<&str> = text.split(',').collect();
        if parts.len() != FIXED_COLUMNS.len() + joints.len() {
            return Err(malformed(
                line,
                format!(
                    "{} fields, header declares {}",
                    parts.len(),
                    FIXED_COLUMNS.len() + joints.len()
                ),
            ));
        }
        let step = parse_unsigned(parts[0], line)?;
        let time = parse_unsigned(parts[1], line)?;
        let time_ns = i64::try_from(time).map_err(|_| malformed(line, "time_ns out of range"))?;

        let mut angles = Vec::with_capacity(joints.len());
        for token in &parts[9..] {
            angles.push(parse_float(token, line)?);
        }
        Ok(Record::Frame(PoseFrame {
            step,
            timestamp: Timestamp::sampling(time_ns, ClockId::Simulation),
            pelvis_position: [
                parse_float(parts[2], line)?,
                parse_float(parts[3], line)?,
                parse_float(parts[4], line)?,
            ],
            pelvis_quaternion: [
                parse_float(parts[5], line)?,
                parse_float(parts[6], line)?,
                parse_float(parts[7], line)?,
                parse_float(parts[8], line)?,
            ],
            joint_angles: angles,
        }))
    }
}

/// Parses CSV text written by [`write_header`], [`write_row`], and [`write_footer`].
///
/// Rejects a missing final newline, a missing or invalid schema marker, a header
/// that does not match the fixed columns, a field count that disagrees with the
/// header, any non-decimal or non-finite number, steps that are not 1, 2, 3, ...,
/// times that are not strictly increasing, a missing or mismatched end footer,
/// or any trailing records after the footer. Nothing is returned unless every line passes.
pub fn parse(text: &str) -> Result<PoseLog, PoseError> {
    collect_log(FrameStream::<&[u8], Csv>::open(text.as_bytes(), None)?)
}

/// Reads and parses a CSV file. The whole file is read before parsing, so a
/// failure never yields a partial log.
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
    fn writes_the_documented_header_and_footer() {
        let text = encode(&[]);
        assert_eq!(
            text,
            "# robot-viewer.pose-frames v1\nstep,time_ns,pelvis_x,pelvis_y,pelvis_z,pelvis_qw,pelvis_qx,pelvis_qy,pelvis_qz,hip,knee\n# end steps=0\n"
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
    fn rejects_a_missing_schema_marker() {
        let text = "step,time_ns,pelvis_x,pelvis_y,pelvis_z,pelvis_qw,pelvis_qx,pelvis_qy,pelvis_qz,hip,knee\n# end steps=0\n";
        assert!(matches!(
            parse(text),
            Err(PoseError::Malformed { line: 1, .. })
        ));
    }

    #[test]
    fn rejects_a_wrong_schema_marker() {
        let text = encode(&[frame(1)]).replace(SCHEMA_MARKER, "# robot-viewer.pose-frames v2");
        assert!(matches!(
            parse(&text),
            Err(PoseError::Malformed { line: 1, .. })
        ));
    }

    #[test]
    fn rejects_a_missing_final_newline() {
        let text = encode(&[frame(1)]);
        let truncated = &text[..text.len() - 1];
        assert!(matches!(parse(truncated), Err(PoseError::Malformed { .. })));
    }

    #[test]
    fn rejects_a_wrong_header() {
        let text = encode(&[frame(1)]).replace("pelvis_qw", "qw");
        assert!(matches!(
            parse(&text),
            Err(PoseError::Malformed { line: 2, .. })
        ));
    }

    #[test]
    fn rejects_a_short_row_without_partial_load() {
        let mut text = encode(&[frame(1), frame(2)]);
        text.insert_str(text.find("# end").unwrap(), "3,300000,0,0,0.79\n");
        assert!(matches!(
            parse(&text),
            Err(PoseError::Malformed { line: 5, .. })
        ));
    }

    #[test]
    fn rejects_nan_and_infinity_tokens() {
        let good = encode(&[frame(1)]);
        let nan = good.replace("0.79", "NaN");
        let inf = good.replace("0.79", "inf");
        assert!(parse(&nan).is_err());
        assert!(parse(&inf).is_err());
    }

    #[test]
    fn rejects_a_skipped_step() {
        let text = encode(&[frame(1), frame(3)]);
        assert!(matches!(
            parse(&text),
            Err(PoseError::Malformed { line: 4, .. })
        ));
    }

    #[test]
    fn rejects_non_increasing_time() {
        let mut f2 = frame(2);
        f2.timestamp = Timestamp::sampling(100_000, ClockId::Simulation);
        let text = encode(&[frame(1), f2]);
        assert!(matches!(
            parse(&text),
            Err(PoseError::Malformed { line: 4, .. })
        ));
    }

    #[test]
    fn rejects_a_line_boundary_truncated_file_missing_footer() {
        let text = encode(&[frame(1), frame(2)]);
        // Drop the last line (the footer)
        let lines: Vec<&str> = text.lines().collect();
        let cut = lines[..lines.len() - 1].join("\n") + "\n";
        assert!(matches!(
            parse(&cut),
            Err(PoseError::Malformed { line: 0, reason }) if reason.contains("missing end footer")
        ));
    }

    #[test]
    fn rejects_a_footer_with_wrong_step_count() {
        let text = encode(&[frame(1), frame(2)]).replace("# end steps=2", "# end steps=3");
        assert!(matches!(
            parse(&text),
            Err(PoseError::Malformed { line: 5, .. })
        ));
    }

    #[test]
    fn rejects_extra_records_after_footer() {
        let mut text = encode(&[frame(1)]);
        text.push_str("2,200000,0,0,0.79,1,0,0,0,0.01,-0.25\n");
        assert!(matches!(
            parse(&text),
            Err(PoseError::Malformed { line: 5, .. })
        ));
    }
}

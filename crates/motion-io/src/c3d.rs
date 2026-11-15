//! C3D import, the second file format SPECS.md Section 8.1 names after BVH.
//!
//! A C3D file is 512-byte blocks: a header block, a parameter section of
//! groups and typed parameters, and a data section of per-frame 3D points
//! and analog samples. This reader handles all three processor types the
//! format defines (Intel little-endian, DEC with VAX floats, MIPS
//! big-endian) and both data encodings (scaled 16-bit integers, or 32-bit
//! floats when `POINT:SCALE` is negative).
//!
//! Points become marker channels ([`JointValue::Position`], no parent) in
//! metres, converted from `POINT:UNITS`. A point whose residual word is
//! negative was not reconstructed in that frame and is marked
//! [`Validity::Missing`], not placed at the zero the file stores. The
//! residual itself, C3D's per-sample confidence, is kept in
//! [`C3dImport::residuals`] in metres, so "timestamps and confidence through
//! every transform" (Section 8.1) starts with the importer not dropping it.
//! Analog channels (force plates, EMG) are returned scaled per
//! `ANALOG:SCALE`, `ANALOG:OFFSET` and `ANALOG:GEN_SCALE`.
//!
//! Limits: a file declaring more than [`MAX_POINTS`] points, [`MAX_FRAMES`]
//! frames or a parameter section beyond the file is refused before any
//! allocation sized by those declarations.

use core::fmt;
use robot_types::frames::{FrameId, Transform, Vec3};
use robot_types::ids::Name;
use robot_types::motion::{
    Channel, HumanSkeleton, JointValue, MotionError, MotionFrame, MotionRecord, ProvenanceStep,
    SourceLicense, Validity,
};
use robot_types::time::{ClockId, Timestamp};

/// Largest point count accepted.
pub const MAX_POINTS: usize = 4096;
/// Largest frame count accepted.
pub const MAX_FRAMES: usize = 1_000_000;

/// Why a C3D file was refused.
#[derive(Clone, PartialEq, Debug)]
pub enum C3dError {
    /// The file ended before a declared structure.
    Truncated(&'static str),
    /// The header key byte or processor type was wrong.
    NotC3d(&'static str),
    /// A declared size exceeded a limit.
    Limit(&'static str),
    /// A parameter was malformed or missing.
    Parameter(String),
    /// The resulting record was rejected.
    Motion(MotionError),
}

impl fmt::Display for C3dError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated(what) => write!(f, "C3D truncated in {what}"),
            Self::NotC3d(what) => write!(f, "not a C3D file: {what}"),
            Self::Limit(what) => write!(f, "C3D limit exceeded: {what}"),
            Self::Parameter(what) => write!(f, "C3D parameter: {what}"),
            Self::Motion(e) => write!(f, "{e}"),
        }
    }
}

impl core::error::Error for C3dError {}

/// The three processor types of the C3D specification.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Processor {
    /// Intel: little-endian integers, IEEE floats.
    Intel,
    /// DEC: little-endian integers, VAX F floats.
    Dec,
    /// MIPS/SGI: big-endian integers and IEEE floats.
    Mips,
}

impl Processor {
    fn i16(self, b: [u8; 2]) -> i16 {
        match self {
            Self::Mips => i16::from_be_bytes(b),
            _ => i16::from_le_bytes(b),
        }
    }

    fn u16(self, b: [u8; 2]) -> u16 {
        self.i16(b) as u16
    }

    fn f32(self, b: [u8; 4]) -> f32 {
        match self {
            Self::Intel => f32::from_le_bytes(b),
            Self::Mips => f32::from_be_bytes(b),
            // VAX F: the two 16-bit halves swapped relative to IEEE, and an
            // exponent bias two larger, so the IEEE reading of the swapped
            // bytes is four times the value.
            Self::Dec => {
                if b == [0; 4] {
                    0.0
                } else {
                    f32::from_le_bytes([b[2], b[3], b[0], b[1]]) / 4.0
                }
            }
        }
    }
}

/// A parameter's data.
#[derive(Clone, PartialEq, Debug)]
pub enum ParamData {
    /// Characters, with the dimensions (first dimension is string length).
    Chars(Vec<u8>, Vec<usize>),
    /// Bytes.
    Bytes(Vec<u8>, Vec<usize>),
    /// 16-bit integers.
    Ints(Vec<i16>, Vec<usize>),
    /// Floats.
    Floats(Vec<f32>, Vec<usize>),
}

impl ParamData {
    /// The values as floats (characters are refused).
    #[must_use]
    pub fn floats(&self) -> Option<Vec<f64>> {
        match self {
            Self::Chars(..) => None,
            Self::Bytes(v, _) => Some(v.iter().map(|&b| f64::from(b)).collect()),
            Self::Ints(v, _) => Some(v.iter().map(|&i| f64::from(i)).collect()),
            Self::Floats(v, _) => Some(v.iter().map(|&x| f64::from(x)).collect()),
        }
    }

    /// The values as trimmed strings, one per column of a 2-D character
    /// array (or one for a 1-D array).
    #[must_use]
    pub fn strings(&self) -> Option<Vec<String>> {
        let Self::Chars(bytes, dims) = self else {
            return None;
        };
        let width = dims.first().copied().unwrap_or(bytes.len()).max(1);
        Some(
            bytes
                .chunks(width)
                .map(|c| String::from_utf8_lossy(c).trim().to_owned())
                .collect(),
        )
    }
}

/// A parsed parameter section: `GROUP:PARAMETER` to data.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct Parameters {
    entries: Vec<(String, ParamData)>,
}

impl Parameters {
    /// Looks up `group:name` (case-insensitive).
    #[must_use]
    pub fn get(&self, group: &str, name: &str) -> Option<&ParamData> {
        let key = format!(
            "{}:{}",
            group.to_ascii_uppercase(),
            name.to_ascii_uppercase()
        );
        self.entries.iter().find(|(k, _)| *k == key).map(|(_, v)| v)
    }

    /// Every key, in file order.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|(k, _)| k.as_str())
    }
}

/// An imported analog channel.
#[derive(Clone, PartialEq, Debug)]
pub struct AnalogChannel {
    /// Label.
    pub label: String,
    /// Unit, as declared.
    pub unit: String,
    /// Samples, `ANALOG:RATE` per second.
    pub samples: Vec<f64>,
}

/// Everything imported from one file.
#[derive(Clone, PartialEq, Debug)]
pub struct C3dImport {
    /// The marker record.
    pub record: MotionRecord<HumanSkeleton>,
    /// Per frame, per point: the residual in metres, `None` when invalid.
    pub residuals: Vec<Vec<Option<f64>>>,
    /// Analog channels.
    pub analog: Vec<AnalogChannel>,
    /// Analog sample rate, Hz.
    pub analog_rate: f64,
    /// The full parameter section.
    pub parameters: Parameters,
    /// The file's processor type.
    pub processor: Processor,
}

fn slice<'a>(
    data: &'a [u8],
    at: usize,
    len: usize,
    what: &'static str,
) -> Result<&'a [u8], C3dError> {
    data.get(at..at + len).ok_or(C3dError::Truncated(what))
}

fn parse_parameters(data: &[u8], start: usize, proc_: Processor) -> Result<Parameters, C3dError> {
    let mut groups: Vec<(i8, String)> = Vec::new();
    let mut params: Vec<(i8, String, ParamData)> = Vec::new();
    let mut p = start + 4;
    loop {
        let head = slice(data, p, 2, "parameter entry")?;
        let name_len = (head[0] as i8).unsigned_abs() as usize;
        let id = head[1] as i8;
        if name_len == 0 || id == 0 {
            break;
        }
        let name = String::from_utf8_lossy(slice(data, p + 2, name_len, "parameter name")?)
            .to_ascii_uppercase();
        let off_at = p + 2 + name_len;
        let next = proc_.u16(
            slice(data, off_at, 2, "parameter offset")?
                .try_into()
                .expect("2 bytes"),
        ) as usize;
        let body = off_at + 2;
        if id < 0 {
            groups.push((-id, name));
        } else {
            let ty = slice(data, body, 1, "parameter type")?[0] as i8;
            let ndims = slice(data, body + 1, 1, "parameter dims")?[0] as usize;
            let dims: Vec<usize> = slice(data, body + 2, ndims, "parameter dims")?
                .iter()
                .map(|&d| usize::from(d))
                .collect();
            // Checked: the workspace aborts on overflow, and each dimension
            // byte can be up to 255, so nine of them already overflow usize
            // multiplication on crafted input.
            let count: usize = dims
                .iter()
                .try_fold(1usize, |a, &d| a.checked_mul(d))
                .ok_or(C3dError::Limit("parameter dimensions overflow"))?
                .max(if ndims == 0 { 1 } else { 0 });
            let width = usize::from(ty.unsigned_abs());
            let bytes = count
                .checked_mul(width)
                .ok_or(C3dError::Limit("parameter size overflows"))?;
            let raw = slice(data, body + 2 + ndims, bytes, "parameter data")?;
            let value = match ty {
                -1 => ParamData::Chars(raw.to_vec(), dims),
                1 => ParamData::Bytes(raw.to_vec(), dims),
                2 => ParamData::Ints(
                    raw.chunks(2).map(|c| proc_.i16([c[0], c[1]])).collect(),
                    dims,
                ),
                4 => ParamData::Floats(
                    raw.chunks(4)
                        .map(|c| proc_.f32([c[0], c[1], c[2], c[3]]))
                        .collect(),
                    dims,
                ),
                _ => return Err(C3dError::Parameter(format!("{name}: data type {ty}"))),
            };
            params.push((id, name, value));
        }
        if next == 0 {
            break;
        }
        p = off_at + next;
    }
    let entries = params
        .into_iter()
        .filter_map(|(gid, name, value)| {
            let group = groups.iter().find(|(g, _)| *g == gid)?;
            Some((format!("{}:{name}", group.1), value))
        })
        .collect();
    Ok(Parameters { entries })
}

fn length_scale(unit: &str) -> f64 {
    match unit.to_ascii_lowercase().as_str() {
        "m" => 1.0,
        "cm" => 0.01,
        "in" => 0.0254,
        _ => 0.001, // "mm", the format's near-universal default
    }
}

/// Imports a C3D file.
pub fn parse(
    bytes: &[u8],
    source_hash: &str,
    license: SourceLicense,
) -> Result<C3dImport, C3dError> {
    let header = slice(bytes, 0, 512, "header")?;
    if header[1] != 0x50 {
        return Err(C3dError::NotC3d("header key byte is not 0x50"));
    }
    let param_start = (usize::from(header[0]).max(1) - 1) * 512;
    let pblock = slice(bytes, param_start, 4, "parameter block")?;
    let proc_ = match pblock[3] {
        84 => Processor::Intel,
        85 => Processor::Dec,
        86 => Processor::Mips,
        _ => return Err(C3dError::NotC3d("unknown processor type")),
    };
    let word = |i: usize| proc_.u16([header[2 * i], header[2 * i + 1]]);
    let points = usize::from(word(1));
    let analog_per_frame = usize::from(word(2));
    let first = usize::from(word(3));
    let last = usize::from(word(4));
    let scale = proc_.f32([header[12], header[13], header[14], header[15]]);
    let data_start = (usize::from(word(8)).max(1) - 1) * 512;
    let header_rate = proc_.f32([header[20], header[21], header[22], header[23]]);
    if points > MAX_POINTS {
        return Err(C3dError::Limit("point count"));
    }
    let frames = (last + 1).saturating_sub(first);
    if frames == 0 || frames > MAX_FRAMES {
        return Err(C3dError::Limit("frame count"));
    }
    let parameters = parse_parameters(bytes, param_start, proc_)?;

    let rate = parameters
        .get("POINT", "RATE")
        .and_then(|p| p.floats())
        .and_then(|v| v.first().copied())
        .unwrap_or(f64::from(header_rate));
    if !(rate > 0.0 && rate.is_finite()) {
        return Err(C3dError::Parameter("POINT:RATE must be positive".into()));
    }
    let unit = parameters
        .get("POINT", "UNITS")
        .and_then(|p| p.strings())
        .and_then(|s| s.into_iter().next())
        .unwrap_or_else(|| "mm".to_owned());
    let to_m = length_scale(&unit);
    let mut labels: Vec<String> = Vec::new();
    for key in ["LABELS", "LABELS2", "LABELS3"] {
        if let Some(l) = parameters.get("POINT", key).and_then(|p| p.strings()) {
            labels.extend(l);
        }
    }
    labels.resize_with(points, String::new);
    let float_data = scale < 0.0;
    let abs_scale = f64::from(scale.abs());
    let word_size = if float_data { 4 } else { 2 };
    let frame_size = (4 * points + analog_per_frame) * word_size;
    let needed = frames
        .checked_mul(frame_size)
        .ok_or(C3dError::Limit("data size"))?;
    let data = slice(bytes, data_start, needed, "data section")?;
    let read = |at: usize| -> f64 {
        if float_data {
            f64::from(proc_.f32([data[at], data[at + 1], data[at + 2], data[at + 3]]))
        } else {
            f64::from(proc_.i16([data[at], data[at + 1]]))
        }
    };

    let mut used_names: Vec<String> = Vec::new();
    let channels: Vec<Channel> = labels
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let cleaned: String = l
                .chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                        c
                    } else {
                        '_'
                    }
                })
                .collect();
            let mut name = if cleaned.is_empty() {
                format!("point{i}")
            } else {
                cleaned
            };
            if used_names.contains(&name) {
                name = format!("{name}_{i}");
            }
            used_names.push(name.clone());
            Ok(Channel {
                name: Name::new(&name)
                    .map_err(|e| C3dError::Parameter(format!("label {l:?}: {e}")))?,
                parent: None,
                offset: Vec3::ZERO,
            })
        })
        .collect::<Result<_, C3dError>>()?;

    let dt_nanos = 1e9 / rate;
    let mut out_frames = Vec::with_capacity(frames);
    let mut residuals = Vec::with_capacity(frames);
    let analog_count = analog_per_frame;
    let mut analog_raw: Vec<f64> = Vec::with_capacity(frames * analog_count);
    for f in 0..frames {
        let base = f * frame_size;
        let mut values = Vec::with_capacity(points);
        let mut validity = Vec::with_capacity(points);
        let mut res = Vec::with_capacity(points);
        for p in 0..points {
            let at = base + 4 * p * word_size;
            let (x, y, z, r) = (
                read(at),
                read(at + word_size),
                read(at + 2 * word_size),
                read(at + 3 * word_size),
            );
            // The residual word: negative means invalid; otherwise its low
            // byte is the residual in units of |scale|.
            let r_word = r as i32 as i16;
            let k = if float_data { 1.0 } else { abs_scale };
            let position = Vec3::new(x * k * to_m, y * k * to_m, z * k * to_m);
            if r_word < 0 || !position.is_finite() {
                values.push(JointValue::Position {
                    position: Vec3::ZERO,
                });
                validity.push(Validity::Missing);
                res.push(None);
            } else {
                values.push(JointValue::Position { position });
                validity.push(Validity::Valid);
                res.push(Some(f64::from(r_word & 0xff) * abs_scale * to_m));
            }
        }
        for a in 0..analog_count {
            analog_raw.push(read(base + (4 * points + a) * word_size));
        }
        out_frames.push(MotionFrame {
            timestamp: Timestamp::sampling(
                (f as f64 * dt_nanos).round() as i64,
                ClockId::Source(0),
            ),
            root: Transform::identity(FrameId::WORLD),
            values,
            validity,
            contacts: Vec::new(),
        });
        residuals.push(res);
    }

    let (analog, analog_rate) =
        analog_channels(&parameters, &analog_raw, frames, analog_count, rate);
    let mut record = MotionRecord::new(
        HumanSkeleton,
        source_hash,
        license,
        "c3d-import-session",
        "c3d-markers",
        FrameId::WORLD,
        ClockId::Source(0),
        channels,
        out_frames,
    )
    .map_err(C3dError::Motion)?;
    record.record_step(
        ProvenanceStep::new(
            "import_c3d",
            concat!("motion-io/", env!("CARGO_PKG_VERSION")),
        )
        .with_parameter("processor", format!("{proc_:?}"))
        .with_parameter("units", unit)
        .with_parameter("rate_hz", rate.to_string())
        .with_parameter("float_data", float_data.to_string()),
    );
    Ok(C3dImport {
        record,
        residuals,
        analog,
        analog_rate,
        parameters,
        processor: proc_,
    })
}

fn analog_channels(
    params: &Parameters,
    raw: &[f64],
    frames: usize,
    per_frame: usize,
    point_rate: f64,
) -> (Vec<AnalogChannel>, f64) {
    let used = params
        .get("ANALOG", "USED")
        .and_then(|p| p.floats())
        .and_then(|v| v.first().copied())
        .map_or(0, |v| v.max(0.0) as usize);
    if used == 0 || per_frame == 0 || per_frame % used != 0 {
        return (Vec::new(), 0.0);
    }
    let per_point = per_frame / used;
    let labels = params
        .get("ANALOG", "LABELS")
        .and_then(|p| p.strings())
        .unwrap_or_default();
    let units = params
        .get("ANALOG", "UNITS")
        .and_then(|p| p.strings())
        .unwrap_or_default();
    let scales = params
        .get("ANALOG", "SCALE")
        .and_then(|p| p.floats())
        .unwrap_or_default();
    let offsets = params
        .get("ANALOG", "OFFSET")
        .and_then(|p| p.floats())
        .unwrap_or_default();
    let gen = params
        .get("ANALOG", "GEN_SCALE")
        .and_then(|p| p.floats())
        .and_then(|v| v.first().copied())
        .unwrap_or(1.0);
    let channels = (0..used)
        .map(|c| {
            let scale = scales.get(c).copied().unwrap_or(1.0);
            let offset = offsets.get(c).copied().unwrap_or(0.0);
            let samples = (0..frames)
                .flat_map(|f| (0..per_point).map(move |s| (f, s)))
                .map(|(f, s)| (raw[f * per_frame + s * used + c] - offset) * scale * gen)
                .collect();
            AnalogChannel {
                label: labels
                    .get(c)
                    .cloned()
                    .unwrap_or_else(|| format!("analog{c}")),
                unit: units.get(c).cloned().unwrap_or_default(),
                samples,
            }
        })
        .collect();
    (channels, point_rate * per_point as f64)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Assembles a minimal Intel or MIPS C3D file: `points` per frame as
    /// `[x, y, z, residual]`, `analog` raw samples per frame.
    pub(crate) fn build(
        big_endian: bool,
        float: bool,
        labels: &[&str],
        frames: &[Vec<[f32; 4]>],
        analog: &[Vec<i16>],
        rate: f32,
    ) -> Vec<u8> {
        let i16b = |v: i16| {
            if big_endian {
                v.to_be_bytes()
            } else {
                v.to_le_bytes()
            }
        };
        let f32b = |v: f32| {
            if big_endian {
                v.to_be_bytes()
            } else {
                v.to_le_bytes()
            }
        };
        let points = labels.len();
        let analog_per_frame = analog.first().map_or(0, Vec::len);
        let mut header = vec![0u8; 512];
        header[0] = 2;
        header[1] = 0x50;
        header[2..4].copy_from_slice(&i16b(points as i16));
        header[4..6].copy_from_slice(&i16b(analog_per_frame as i16));
        header[6..8].copy_from_slice(&i16b(1));
        header[8..10].copy_from_slice(&i16b(frames.len() as i16));
        let scale = if float { -0.5f32 } else { 0.5 };
        header[12..16].copy_from_slice(&f32b(scale));
        header[16..18].copy_from_slice(&i16b(3));
        header[20..24].copy_from_slice(&f32b(rate));

        // Parameter section: POINT group (id 1), ANALOG group (id 2).
        let mut p = vec![0u8, 0, 1, if big_endian { 86 } else { 84 }];
        let mut entry = |name: &str, id: i8, body: Vec<u8>| {
            p.push(name.len() as u8);
            p.push(id as u8);
            p.extend_from_slice(name.as_bytes());
            p.extend_from_slice(&i16b((body.len() + 2) as i16));
            p.extend(body);
        };
        entry("POINT", -1, vec![0]);
        entry("ANALOG", -2, vec![0]);
        let mut units = vec![(-1i8) as u8, 1, 2];
        units.extend_from_slice(b"mm");
        units.push(0);
        entry("UNITS", 1, units);
        let width = 4;
        let mut lab = vec![(-1i8) as u8, 2, width as u8, points as u8];
        for l in labels {
            let mut s = l.as_bytes().to_vec();
            s.resize(width, b' ');
            lab.extend(s);
        }
        lab.push(0);
        entry("LABELS", 1, lab);
        if analog_per_frame > 0 {
            let mut used = vec![2u8, 0];
            used.extend_from_slice(&i16b(1));
            used.push(0);
            entry("USED", 2, used);
            let mut sc = vec![4u8, 1, 1];
            sc.extend_from_slice(&f32b(2.0));
            sc.push(0);
            entry("SCALE", 2, sc);
        }
        p.extend_from_slice(&[0, 0]);
        p.resize(512, 0);

        let mut data = Vec::new();
        for (f, frame) in frames.iter().enumerate() {
            for pt in frame {
                for &v in pt {
                    if float {
                        data.extend_from_slice(&f32b(v));
                    } else {
                        data.extend_from_slice(&i16b(v as i16));
                    }
                }
            }
            for &a in analog.get(f).map_or(&[][..], Vec::as_slice) {
                if float {
                    data.extend_from_slice(&f32b(f32::from(a)));
                } else {
                    data.extend_from_slice(&i16b(a));
                }
            }
        }
        let mut out = header;
        out.extend(p);
        out.extend(data);
        out
    }

    fn frames() -> Vec<Vec<[f32; 4]>> {
        vec![
            vec![[100.0, 200.0, 300.0, 4.0], [10.0, 0.0, 0.0, -1.0]],
            vec![[102.0, 200.0, 300.0, 2.0], [12.0, 0.0, 0.0, 6.0]],
        ]
    }

    #[test]
    fn integer_intel_points_scale_to_metres_and_invalid_points_go_missing() {
        let bytes = build(false, false, &["LHD", "RHD"], &frames(), &[], 100.0);
        let c = parse(
            &bytes,
            "sha256:x",
            SourceLicense::unaudited("fixture-internal"),
        )
        .unwrap();
        assert_eq!(c.processor, Processor::Intel);
        assert_eq!(c.record.channels[0].name.as_str(), "LHD");
        let JointValue::Position { position } = c.record.frames[0].values[0] else {
            panic!()
        };
        // 100 counts * 0.5 scale = 50 mm = 0.05 m.
        assert!((position - Vec3::new(0.05, 0.1, 0.15)).norm() < 1e-12);
        assert_eq!(c.record.frames[0].validity[1], Validity::Missing);
        assert!((c.residuals[0][0].unwrap() - 0.002).abs() < 1e-12);
        assert_eq!(c.residuals[0][1], None);
        assert_eq!(c.record.frames[1].timestamp.nanos(), 10_000_000);
        c.record.validate().unwrap();
    }

    #[test]
    fn float_mips_points_read_the_same_values() {
        let bytes = build(true, true, &["LHD", "RHD"], &frames(), &[], 100.0);
        let c = parse(
            &bytes,
            "sha256:x",
            SourceLicense::unaudited("fixture-internal"),
        )
        .unwrap();
        assert_eq!(c.processor, Processor::Mips);
        let JointValue::Position { position } = c.record.frames[1].values[0] else {
            panic!()
        };
        // Float data is not scaled by POINT:SCALE.
        assert!((position - Vec3::new(0.102, 0.2, 0.3)).norm() < 1e-9);
    }

    #[test]
    fn analog_samples_are_scaled() {
        let bytes = build(
            false,
            false,
            &["A", "B"],
            &frames(),
            &[vec![7], vec![-3]],
            100.0,
        );
        let c = parse(
            &bytes,
            "sha256:x",
            SourceLicense::unaudited("fixture-internal"),
        )
        .unwrap();
        assert_eq!(c.analog.len(), 1);
        assert_eq!(c.analog[0].samples, vec![14.0, -6.0]);
        assert!((c.analog_rate - 100.0).abs() < 1e-12);
    }

    #[test]
    fn a_vax_float_reads_as_its_ieee_value() {
        // 1.0 in VAX F is 0x4080_0000 with the 16-bit halves swapped:
        // bytes [0x80, 0x40, 0x00, 0x00].
        assert_eq!(Processor::Dec.f32([0x80, 0x40, 0x00, 0x00]), 1.0);
        assert_eq!(Processor::Dec.f32([0; 4]), 0.0);
    }

    #[test]
    fn a_wrong_key_byte_and_a_truncated_file_are_refused() {
        let mut bytes = build(
            false,
            false,
            &["A"],
            &[vec![[0.0; 4]], vec![[0.0; 4]]],
            &[],
            100.0,
        );
        bytes.truncate(1030); // inside the 16-byte data section at 1024
        assert!(matches!(
            parse(&bytes, "x", SourceLicense::unaudited("f")),
            Err(C3dError::Truncated(_))
        ));
        bytes[1] = 0;
        assert!(matches!(
            parse(&bytes, "x", SourceLicense::unaudited("f")),
            Err(C3dError::NotC3d(_))
        ));
    }

    #[test]
    fn parameter_dimensions_that_overflow_are_refused_not_an_abort() {
        // A parameter declaring ten dimensions of 255 overflows the usize
        // product; the workspace aborts on overflow, so it must be checked.
        let mut data = vec![0u8; 64];
        data[4] = 1; // name length
        data[5] = 1; // parameter in group 1
        data[6] = b'A';
        // next-entry offset 0 at 7..9: last entry.
        data[9] = 1; // type: bytes, width 1
        data[10] = 10; // ten dimensions...
        for d in &mut data[11..21] {
            *d = 255; // ...of 255 each: 255^10 > usize::MAX
        }
        assert!(matches!(
            parse_parameters(&data, 0, Processor::Intel),
            Err(C3dError::Limit(_))
        ));
    }
}

//! NumPy `.npz` archives: a ZIP container of `.npy` arrays, the format the
//! AMASS corpus ships in (SPECS.md Section 8.1's "supported numerical
//! human-body formats").
//!
//! The ZIP reader walks the central directory, accepts stored (method 0)
//! and DEFLATE (method 8, via [`crate::inflate`]) members, and verifies each
//! member's CRC-32; ZIP64, encryption and other methods are refused by name.
//! The NPY reader accepts format versions 1 to 3, little- and big-endian
//! `f4`/`f8`/`i4`/`i8`/`u1` arrays in C order, and the fixed-width unicode
//! strings (`<U*`) AMASS uses for its `gender` field. Fortran-ordered
//! arrays are refused rather than silently transposed.
//!
//! Every size read from the file is checked against [`MAX_MEMBER_BYTES`]
//! before allocation.

use core::fmt;

use crate::inflate::{inflate, InflateError};

/// Largest decompressed member accepted, bytes.
pub const MAX_MEMBER_BYTES: usize = 1 << 30;

/// Why an archive or array was refused.
#[derive(Clone, PartialEq, Debug)]
pub enum NpzError {
    /// A structure ran past the end of the data.
    Truncated(&'static str),
    /// A signature or magic number was wrong.
    BadSignature(&'static str),
    /// A construct this reader does not implement.
    Unsupported(String),
    /// A member's CRC-32 did not match.
    Checksum {
        /// The member.
        name: String,
    },
    /// Decompression failed.
    Inflate(InflateError),
    /// The NPY header could not be read.
    Header(String),
}

impl fmt::Display for NpzError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated(what) => write!(f, "truncated {what}"),
            Self::BadSignature(what) => write!(f, "bad {what} signature"),
            Self::Unsupported(what) => write!(f, "unsupported: {what}"),
            Self::Checksum { name } => write!(f, "CRC-32 mismatch in {name}"),
            Self::Inflate(e) => write!(f, "{e}"),
            Self::Header(e) => write!(f, "NPY header: {e}"),
        }
    }
}

impl core::error::Error for NpzError {}

/// CRC-32 (IEEE 802.3, reflected, as ZIP uses it).
#[must_use]
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &b in data {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xedb8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

fn le16(d: &[u8], at: usize) -> Result<usize, NpzError> {
    d.get(at..at + 2)
        .map(|b| usize::from(u16::from_le_bytes([b[0], b[1]])))
        .ok_or(NpzError::Truncated("zip field"))
}

fn le32(d: &[u8], at: usize) -> Result<u32, NpzError> {
    d.get(at..at + 4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .ok_or(NpzError::Truncated("zip field"))
}

/// Reads every member of a ZIP archive as `(name, bytes)`.
pub fn read_zip(data: &[u8]) -> Result<Vec<(String, Vec<u8>)>, NpzError> {
    // An EOCD record is 22 bytes; shorter input cannot hold one, and the
    // checked `get` keeps a 0–3 byte input from panicking the scan.
    if data.len() < 22 {
        return Err(NpzError::BadSignature("end of central directory"));
    }
    let min = data.len().saturating_sub(22 + 65_535);
    let eocd = (min..=data.len() - 22)
        .rev()
        .find(|&i| data.get(i..i + 4) == Some(&[0x50, 0x4b, 0x05, 0x06]))
        .ok_or(NpzError::BadSignature("end of central directory"))?;
    let count = le16(data, eocd + 10)?;
    let mut p = le32(data, eocd + 16)? as usize;
    if count == 0xffff || p == 0xffff_ffff {
        return Err(NpzError::Unsupported("ZIP64".into()));
    }
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        if le32(data, p)? != 0x0201_4b50 {
            return Err(NpzError::BadSignature("central directory entry"));
        }
        let flags = le16(data, p + 8)?;
        let method = le16(data, p + 10)?;
        let crc = le32(data, p + 16)?;
        let csize = le32(data, p + 20)?;
        let usize_ = le32(data, p + 24)?;
        let (nlen, xlen, clen) = (
            le16(data, p + 28)?,
            le16(data, p + 30)?,
            le16(data, p + 32)?,
        );
        let local = le32(data, p + 42)? as usize;
        let name = String::from_utf8_lossy(
            data.get(p + 46..p + 46 + nlen)
                .ok_or(NpzError::Truncated("name"))?,
        )
        .into_owned();
        p += 46 + nlen + xlen + clen;
        if csize == 0xffff_ffff || usize_ == 0xffff_ffff || local == 0xffff_ffff {
            return Err(NpzError::Unsupported(format!("ZIP64 member {name}")));
        }
        if flags & 1 != 0 {
            return Err(NpzError::Unsupported(format!("encrypted member {name}")));
        }
        if usize_ as usize > MAX_MEMBER_BYTES {
            return Err(NpzError::Unsupported(format!(
                "member {name} larger than the limit"
            )));
        }
        if le32(data, local)? != 0x0403_4b50 {
            return Err(NpzError::BadSignature("local file header"));
        }
        let start = local + 30 + le16(data, local + 26)? + le16(data, local + 28)?;
        let raw = data
            .get(start..start + csize as usize)
            .ok_or(NpzError::Truncated("member data"))?;
        let bytes = match method {
            0 => raw.to_vec(),
            8 => inflate(raw, usize_ as usize).map_err(NpzError::Inflate)?,
            m => {
                return Err(NpzError::Unsupported(format!(
                    "compression method {m} in {name}"
                )))
            }
        };
        if bytes.len() != usize_ as usize {
            return Err(NpzError::Truncated("member (size mismatch)"));
        }
        if crc32(&bytes) != crc {
            return Err(NpzError::Checksum { name });
        }
        out.push((name, bytes));
    }
    Ok(out)
}

/// An array's elements.
#[derive(Clone, PartialEq, Debug)]
pub enum NpyData {
    /// Numbers, converted to `f64`.
    Numbers(Vec<f64>),
    /// Fixed-width unicode strings.
    Strings(Vec<String>),
}

/// One `.npy` array.
#[derive(Clone, PartialEq, Debug)]
pub struct NpyArray {
    /// Shape; empty for a scalar.
    pub shape: Vec<usize>,
    /// Elements in C order.
    pub data: NpyData,
}

impl NpyArray {
    /// The elements as numbers, if numeric.
    #[must_use]
    pub fn numbers(&self) -> Option<&[f64]> {
        match &self.data {
            NpyData::Numbers(v) => Some(v),
            NpyData::Strings(_) => None,
        }
    }

    /// The first element as a string, if a string array.
    #[must_use]
    pub fn string(&self) -> Option<&str> {
        match &self.data {
            NpyData::Strings(v) => v.first().map(String::as_str),
            NpyData::Numbers(_) => None,
        }
    }
}

fn header_value<'a>(header: &'a str, key: &str) -> Result<&'a str, NpzError> {
    let k = format!("'{key}':");
    let at = header
        .find(&k)
        .ok_or_else(|| NpzError::Header(format!("no {key}")))?
        + k.len();
    Ok(header[at..].trim_start())
}

/// Parses one `.npy` file.
pub fn parse_npy(bytes: &[u8]) -> Result<NpyArray, NpzError> {
    if bytes.get(..6) != Some(b"\x93NUMPY") {
        return Err(NpzError::BadSignature("NPY"));
    }
    let major = *bytes.get(6).ok_or(NpzError::Truncated("NPY version"))?;
    let (hlen, hstart) = match major {
        1 => (le16(bytes, 8)?, 10),
        2 | 3 => (le32(bytes, 8)? as usize, 12),
        v => return Err(NpzError::Unsupported(format!("NPY version {v}"))),
    };
    let header = core::str::from_utf8(
        bytes
            .get(hstart..hstart + hlen)
            .ok_or(NpzError::Truncated("NPY header"))?,
    )
    .map_err(|e| NpzError::Header(e.to_string()))?;
    let descr_v = header_value(header, "descr")?;
    let descr = descr_v
        .strip_prefix('\'')
        .and_then(|s| s.split('\'').next())
        .ok_or_else(|| NpzError::Header("descr is not a string".into()))?;
    if header_value(header, "fortran_order")?.starts_with("True") {
        return Err(NpzError::Unsupported("Fortran-ordered array".into()));
    }
    let shape_v = header_value(header, "shape")?;
    let inner = shape_v
        .strip_prefix('(')
        .and_then(|s| s.split(')').next())
        .ok_or_else(|| NpzError::Header("shape is not a tuple".into()))?;
    let shape: Vec<usize> = inner
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse()
                .map_err(|_| NpzError::Header(format!("shape entry {s:?}")))
        })
        .collect::<Result<_, _>>()?;
    let count = shape
        .iter()
        .try_fold(1usize, |a, &d| a.checked_mul(d))
        .ok_or_else(|| NpzError::Header("shape overflows".into()))?;
    let body = &bytes[hstart + hlen..];
    let (order, kind, width) = {
        let mut c = descr.chars();
        let order = c
            .next()
            .ok_or_else(|| NpzError::Header("empty descr".into()))?;
        let kind = c
            .next()
            .ok_or_else(|| NpzError::Header("descr kind".into()))?;
        let width: usize = c
            .as_str()
            .parse()
            .map_err(|_| NpzError::Header(format!("descr {descr}")))?;
        // A zero-width element (`<U0`, `<f0`) would make every chunking
        // step below panic ("chunk size must be non-zero").
        if width == 0 {
            return Err(NpzError::Unsupported(format!("dtype {descr}")));
        }
        (order, kind, width)
    };
    let big = order == '>';
    let elem = if kind == 'U' { 4 * width } else { width };
    let need = count
        .checked_mul(elem)
        .ok_or_else(|| NpzError::Header("size overflows".into()))?;
    let body = body.get(..need).ok_or(NpzError::Truncated("NPY data"))?;
    let num = |chunk: &[u8]| -> Result<f64, NpzError> {
        let mut b = chunk.to_vec();
        if big {
            b.reverse();
        }
        Ok(match (kind, width) {
            ('f', 8) => f64::from_le_bytes(b.try_into().expect("8")),
            ('f', 4) => f64::from(f32::from_le_bytes(b.try_into().expect("4"))),
            ('i', 8) => i64::from_le_bytes(b.try_into().expect("8")) as f64,
            ('i', 4) => f64::from(i32::from_le_bytes(b.try_into().expect("4"))),
            ('u', 1) | ('b', 1) => f64::from(b[0]),
            _ => return Err(NpzError::Unsupported(format!("dtype {descr}"))),
        })
    };
    let data = if kind == 'U' {
        NpyData::Strings(
            body.chunks(elem)
                .map(|c| {
                    c.chunks(4)
                        .map(|u| {
                            let v = if big {
                                u32::from_be_bytes([u[0], u[1], u[2], u[3]])
                            } else {
                                u32::from_le_bytes([u[0], u[1], u[2], u[3]])
                            };
                            char::from_u32(v).unwrap_or('\u{fffd}')
                        })
                        .take_while(|&c| c != '\0')
                        .collect()
                })
                .collect(),
        )
    } else {
        NpyData::Numbers(body.chunks(elem).map(num).collect::<Result<_, _>>()?)
    };
    Ok(NpyArray { shape, data })
}

/// Reads every array of an `.npz` archive, keyed by name without `.npy`.
pub fn read_npz(data: &[u8]) -> Result<Vec<(String, NpyArray)>, NpzError> {
    read_zip(data)?
        .into_iter()
        .map(|(name, bytes)| {
            let key = name.strip_suffix(".npy").unwrap_or(&name).to_owned();
            Ok((key, parse_npy(&bytes)?))
        })
        .collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A version-1 NPY of `f8` values.
    pub(crate) fn npy_f8(shape: &[usize], values: &[f64]) -> Vec<u8> {
        let dims: Vec<String> = shape.iter().map(ToString::to_string).collect();
        let shape_s = if shape.len() == 1 {
            format!("({},)", dims[0])
        } else {
            format!("({})", dims.join(", "))
        };
        npy(
            "<f8",
            &shape_s,
            &values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
    }

    pub(crate) fn npy(descr: &str, shape: &str, body: &[u8]) -> Vec<u8> {
        let mut h = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
        while (10 + h.len() + 1) % 64 != 0 {
            h.push(' ');
        }
        h.push('\n');
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend_from_slice(&(h.len() as u16).to_le_bytes());
        out.extend_from_slice(h.as_bytes());
        out.extend_from_slice(body);
        out
    }

    /// A stored (uncompressed) ZIP of the given members, as `np.savez` writes.
    pub(crate) fn zip_stored(members: &[(&str, Vec<u8>)]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut central = Vec::new();
        for (name, bytes) in members {
            let offset = out.len() as u32;
            let crc = crc32(bytes);
            out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
            out.extend_from_slice(&[20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            out.extend_from_slice(&crc.to_le_bytes());
            out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(bytes);
            central.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
            central.extend_from_slice(&[20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            central.extend_from_slice(&crc.to_le_bytes());
            central.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            central.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            central.extend_from_slice(&(name.len() as u16).to_le_bytes());
            central.extend_from_slice(&[0; 12]);
            central.extend_from_slice(&offset.to_le_bytes());
            central.extend_from_slice(name.as_bytes());
        }
        let cd_offset = out.len() as u32;
        out.extend_from_slice(&central);
        out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
        out.extend_from_slice(&[0, 0, 0, 0]);
        out.extend_from_slice(&(members.len() as u16).to_le_bytes());
        out.extend_from_slice(&(members.len() as u16).to_le_bytes());
        out.extend_from_slice(&(central.len() as u32).to_le_bytes());
        out.extend_from_slice(&cd_offset.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out
    }

    #[test]
    fn crc32_matches_the_standard_check_value() {
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
    }

    #[test]
    fn arrays_round_trip_through_a_stored_archive() {
        let a = npy_f8(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let s: Vec<u8> = "female"
            .chars()
            .flat_map(|c| (c as u32).to_le_bytes())
            .collect();
        let g = npy("<U6", "()", &s);
        let z = zip_stored(&[("poses.npy", a), ("gender.npy", g)]);
        let arrays = read_npz(&z).unwrap();
        assert_eq!(arrays[0].0, "poses");
        assert_eq!(arrays[0].1.shape, vec![2, 3]);
        assert_eq!(arrays[0].1.numbers().unwrap()[5], 6.0);
        assert_eq!(arrays[1].1.string(), Some("female"));
    }

    #[test]
    fn a_corrupted_member_fails_its_checksum() {
        let mut z = zip_stored(&[("x.npy", npy_f8(&[1], &[1.0]))]);
        let at = z.len() / 3;
        z[at] ^= 0xff;
        assert!(read_npz(&z).is_err());
    }

    #[test]
    fn fortran_order_and_unknown_dtypes_are_refused() {
        let mut f = npy_f8(&[1], &[1.0]);
        let at = f.windows(5).position(|w| w == b"False").unwrap();
        f[at..at + 5].copy_from_slice(b"True ");
        assert!(matches!(parse_npy(&f), Err(NpzError::Unsupported(_))));
        assert!(matches!(
            parse_npy(&npy("<c16", "(1,)", &[0; 16])),
            Err(NpzError::Unsupported(_))
        ));
    }

    #[test]
    fn a_zero_width_dtype_is_refused_not_a_panic() {
        // `<U0` and `<f0` would make the element size 0 and panic the
        // body chunking ("chunk size must be non-zero").
        assert!(matches!(
            parse_npy(&npy("<U0", "(1,)", &[])),
            Err(NpzError::Unsupported(_))
        ));
        assert!(matches!(
            parse_npy(&npy("<f0", "(1,)", &[])),
            Err(NpzError::Unsupported(_))
        ));
    }

    #[test]
    fn a_tiny_zip_input_errors_instead_of_panicking() {
        for len in 0..4 {
            assert!(read_zip(&vec![0x50; len]).is_err());
        }
    }
}

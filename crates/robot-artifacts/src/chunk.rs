//! Versioned, bounded, framed binary chunks for high-rate numerical data.
//!
//! SPECS.md Section 7.5: "Use chunked binary arrays for high-rate numerical data
//! … Every wire/archive format requires explicit versioning, size limits,
//! framing, endianness/alignment rules where applicable and validation before
//! access. Serialization is not a transport or a schema-migration strategy."
//!
//! Each of those words is a field or a check below. The format is:
//!
//! ```text
//! magic    8 bytes   "RBTCHNK\0"
//! version  2 bytes   format version, little-endian
//! layout   1 byte    0 = structure-of-arrays, 1 = array-of-structures
//! dtype    1 byte    0 = f32, 1 = f64, 2 = i32, 3 = u8
//! rows     4 bytes   little-endian
//! cols     4 bytes   little-endian
//! schema  32 bytes   digest of the schema this chunk was written against
//! payload  rows*cols*width bytes, little-endian
//! digest  32 bytes   SHA-256 of everything above
//! ```
//!
//! Endianness is fixed little-endian on the wire regardless of host, so an
//! archive written on one machine reads on another. The trailing digest is
//! checked before any payload byte is interpreted, which is what SPECS.md
//! Section 13 means by validating untrusted input: an archive from a shared
//! drive is untrusted input too.
//!
//! Zero-copy access is deliberately *not* offered. Appendix B.1 correction 11
//! warns that "Zero-copy archives do not imply stable Rust ABI, validated input
//! or automatic schema evolution"; decoding into owned, validated vectors costs
//! a copy and removes a class of defect that is very hard to test for.

use crate::hash::{ContentId, Digest, Sha256};
use core::fmt;

/// Magic bytes identifying a chunk.
pub const MAGIC: [u8; 8] = *b"RBTCHNK\0";

/// The format version this build writes.
pub const FORMAT_VERSION: u16 = 1;

/// Fixed header size in bytes.
const HEADER_LEN: usize = 8 + 2 + 1 + 1 + 4 + 4 + 32;

/// Trailing digest size in bytes.
const DIGEST_LEN: usize = 32;

/// How values are arranged in the payload.
///
/// SPECS.md Appendix D.2 asks for SoA chunks to be *benchmarked* rather than
/// assumed optimal, so both layouts are representable and the chunk records
/// which one it used.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Layout {
    /// Each field contiguous: column `c` occupies `[c*rows, (c+1)*rows)`.
    StructureOfArrays,
    /// Each record contiguous: row `r` occupies `[r*cols, (r+1)*cols)`.
    ArrayOfStructures,
}

impl Layout {
    const fn code(self) -> u8 {
        match self {
            Self::StructureOfArrays => 0,
            Self::ArrayOfStructures => 1,
        }
    }

    const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::StructureOfArrays),
            1 => Some(Self::ArrayOfStructures),
            _ => None,
        }
    }

    /// The canonical identifier used in manifests.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::StructureOfArrays => "soa",
            Self::ArrayOfStructures => "aos",
        }
    }
}

/// The element type of a chunk's payload.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DType {
    /// 32-bit float.
    F32,
    /// 64-bit float.
    F64,
    /// 32-bit signed integer.
    I32,
    /// Unsigned byte.
    U8,
}

impl DType {
    const fn code(self) -> u8 {
        match self {
            Self::F32 => 0,
            Self::F64 => 1,
            Self::I32 => 2,
            Self::U8 => 3,
        }
    }

    const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::F32),
            1 => Some(Self::F64),
            2 => Some(Self::I32),
            3 => Some(Self::U8),
            _ => None,
        }
    }

    /// Bytes per element.
    #[must_use]
    pub const fn width(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F64 => 8,
            Self::U8 => 1,
        }
    }

    /// The canonical identifier used in manifests.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::I32 => "i32",
            Self::U8 => "u8",
        }
    }
}

/// Limits applied when decoding untrusted input.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ChunkLimits {
    /// Maximum encoded size in bytes.
    pub max_bytes: usize,
    /// Maximum rows.
    pub max_rows: u32,
    /// Maximum columns.
    pub max_cols: u32,
}

impl Default for ChunkLimits {
    /// Bounds sized so that one chunk stays well inside a cache-friendly
    /// working set; SPECS.md Section 7.6 item 1 requires bounded chunks and
    /// forbids "materializing the entire corpus … as one allocation".
    fn default() -> Self {
        Self {
            max_bytes: 256 * 1024 * 1024,
            max_rows: 4_000_000,
            max_cols: 4096,
        }
    }
}

/// A decoded chunk.
#[derive(Clone, PartialEq, Debug)]
pub struct Chunk {
    /// Row count.
    pub rows: u32,
    /// Column count.
    pub cols: u32,
    /// Payload layout.
    pub layout: Layout,
    /// Element type.
    pub dtype: DType,
    /// Digest of the schema this chunk was written against.
    ///
    /// Checked on decode: a chunk whose schema digest does not match the
    /// reader's schema is a migration problem, and SPECS.md Section 7.5 says
    /// serialization is not a migration strategy, so it is reported.
    pub schema_digest: Digest,
    /// Values as `f64`, widened from the stored type.
    pub values: Vec<f64>,
}

/// Why a chunk could not be decoded.
#[derive(Clone, PartialEq, Debug)]
pub enum ChunkError {
    /// The magic bytes did not match.
    BadMagic,
    /// The format version is not readable by this build.
    UnsupportedVersion {
        /// Version found.
        found: u16,
        /// Version this build writes.
        supported: u16,
    },
    /// The layout byte was not a known layout.
    UnknownLayout {
        /// Byte found.
        code: u8,
    },
    /// The dtype byte was not a known type.
    UnknownDType {
        /// Byte found.
        code: u8,
    },
    /// The input was shorter than the header or the declared payload.
    Truncated {
        /// Bytes expected.
        expected: usize,
        /// Bytes available.
        found: usize,
    },
    /// The declared dimensions exceeded the limits.
    DimensionsOutOfRange {
        /// Rows declared.
        rows: u32,
        /// Columns declared.
        cols: u32,
    },
    /// The encoded chunk exceeded the size limit.
    TooLarge {
        /// Size found.
        found: usize,
        /// Limit.
        limit: usize,
    },
    /// The trailing digest did not match the content.
    DigestMismatch {
        /// Digest stored.
        expected: String,
        /// Digest computed.
        actual: String,
    },
    /// The chunk was written against a different schema.
    SchemaMismatch {
        /// Digest stored in the chunk.
        chunk: String,
        /// Digest the reader expects.
        reader: String,
    },
    /// A decoded floating-point value was NaN or infinite.
    NotFinite {
        /// Index into the value vector.
        index: usize,
    },
    /// The value count did not match the declared dimensions.
    ValueCountMismatch {
        /// Count expected.
        expected: usize,
        /// Count supplied.
        found: usize,
    },
    /// An integer value did not fit the target type.
    ValueOutOfRange {
        /// Index into the value vector.
        index: usize,
        /// The value.
        value: f64,
        /// Target type.
        dtype: DType,
    },
}

impl fmt::Display for ChunkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic => write!(f, "not a chunk: magic bytes do not match"),
            Self::UnsupportedVersion { found, supported } => {
                write!(
                    f,
                    "chunk format version {found}, this build writes {supported}"
                )
            }
            Self::UnknownLayout { code } => write!(f, "unknown layout code {code}"),
            Self::UnknownDType { code } => write!(f, "unknown dtype code {code}"),
            Self::Truncated { expected, found } => {
                write!(f, "truncated: expected {expected} bytes, found {found}")
            }
            Self::DimensionsOutOfRange { rows, cols } => {
                write!(f, "dimensions {rows}x{cols} exceed the decode limits")
            }
            Self::TooLarge { found, limit } => {
                write!(f, "chunk is {found} bytes, limit is {limit}")
            }
            Self::DigestMismatch { expected, actual } => {
                write!(
                    f,
                    "chunk digest mismatch: stored {expected}, computed {actual}"
                )
            }
            Self::SchemaMismatch { chunk, reader } => write!(
                f,
                "chunk was written against schema {chunk}, reader expects {reader}"
            ),
            Self::NotFinite { index } => write!(f, "value {index} is not finite"),
            Self::ValueCountMismatch { expected, found } => {
                write!(f, "expected {expected} values, found {found}")
            }
            Self::ValueOutOfRange {
                index,
                value,
                dtype,
            } => {
                write!(f, "value {index} ({value}) does not fit {}", dtype.id())
            }
        }
    }
}

impl core::error::Error for ChunkError {}

impl Chunk {
    /// Builds a chunk from values, checking the count and finiteness.
    pub fn new(
        rows: u32,
        cols: u32,
        layout: Layout,
        dtype: DType,
        schema_digest: Digest,
        values: Vec<f64>,
    ) -> Result<Self, ChunkError> {
        let expected = rows as usize * cols as usize;
        if values.len() != expected {
            return Err(ChunkError::ValueCountMismatch {
                expected,
                found: values.len(),
            });
        }
        if matches!(dtype, DType::F32 | DType::F64) {
            if let Some(index) = values.iter().position(|v| !v.is_finite()) {
                return Err(ChunkError::NotFinite { index });
            }
        }
        if matches!(dtype, DType::I32 | DType::U8) {
            for (index, v) in values.iter().enumerate() {
                let fits = match dtype {
                    DType::I32 => {
                        v.fract() == 0.0 && *v >= f64::from(i32::MIN) && *v <= f64::from(i32::MAX)
                    }
                    DType::U8 => v.fract() == 0.0 && *v >= 0.0 && *v <= 255.0,
                    _ => true,
                };
                if !fits {
                    return Err(ChunkError::ValueOutOfRange {
                        index,
                        value: *v,
                        dtype,
                    });
                }
            }
        }
        Ok(Self {
            rows,
            cols,
            layout,
            dtype,
            schema_digest,
            values,
        })
    }

    /// The value at `(row, col)`, honouring the layout.
    #[must_use]
    pub fn get(&self, row: u32, col: u32) -> Option<f64> {
        if row >= self.rows || col >= self.cols {
            return None;
        }
        let index = match self.layout {
            Layout::StructureOfArrays => col as usize * self.rows as usize + row as usize,
            Layout::ArrayOfStructures => row as usize * self.cols as usize + col as usize,
        };
        self.values.get(index).copied()
    }

    /// Encodes the chunk.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let width = self.dtype.width();
        let payload_len = self.values.len() * width;
        let mut out = Vec::with_capacity(HEADER_LEN + payload_len + DIGEST_LEN);
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.push(self.layout.code());
        out.push(self.dtype.code());
        out.extend_from_slice(&self.rows.to_le_bytes());
        out.extend_from_slice(&self.cols.to_le_bytes());
        out.extend_from_slice(self.schema_digest.bytes());
        for v in &self.values {
            match self.dtype {
                DType::F32 => out.extend_from_slice(&(*v as f32).to_le_bytes()),
                DType::F64 => out.extend_from_slice(&v.to_le_bytes()),
                DType::I32 => out.extend_from_slice(&(*v as i32).to_le_bytes()),
                DType::U8 => out.push(*v as u8),
            }
        }
        let mut h = Sha256::new();
        h.update(&out);
        out.extend_from_slice(h.finalize().bytes());
        out
    }

    /// The chunk's content identity.
    #[must_use]
    pub fn content_id(&self) -> ContentId {
        ContentId::of(&self.encode())
    }

    /// Decodes a chunk, validating framing, limits, digest and schema.
    ///
    /// `reader_schema` is the digest of the schema the caller expects. Passing
    /// `None` skips that check and is appropriate only for tooling that inspects
    /// an archive without interpreting it.
    pub fn decode(
        bytes: &[u8],
        reader_schema: Option<Digest>,
        limits: ChunkLimits,
    ) -> Result<Self, ChunkError> {
        if bytes.len() > limits.max_bytes {
            return Err(ChunkError::TooLarge {
                found: bytes.len(),
                limit: limits.max_bytes,
            });
        }
        if bytes.len() < HEADER_LEN + DIGEST_LEN {
            return Err(ChunkError::Truncated {
                expected: HEADER_LEN + DIGEST_LEN,
                found: bytes.len(),
            });
        }
        if bytes[..8] != MAGIC {
            return Err(ChunkError::BadMagic);
        }
        let version = u16::from_le_bytes([bytes[8], bytes[9]]);
        if version != FORMAT_VERSION {
            return Err(ChunkError::UnsupportedVersion {
                found: version,
                supported: FORMAT_VERSION,
            });
        }
        let layout =
            Layout::from_code(bytes[10]).ok_or(ChunkError::UnknownLayout { code: bytes[10] })?;
        let dtype =
            DType::from_code(bytes[11]).ok_or(ChunkError::UnknownDType { code: bytes[11] })?;
        let rows = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        let cols = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
        if rows > limits.max_rows || cols > limits.max_cols {
            return Err(ChunkError::DimensionsOutOfRange { rows, cols });
        }
        let mut schema = [0u8; 32];
        schema.copy_from_slice(&bytes[20..52]);
        let schema_digest = Digest::from_hex(&hex(&schema)).expect("32 bytes are 64 hex digits");

        let count = rows as usize * cols as usize;
        let payload_len = count
            .checked_mul(dtype.width())
            .ok_or(ChunkError::DimensionsOutOfRange { rows, cols })?;
        let expected = HEADER_LEN + payload_len + DIGEST_LEN;
        if bytes.len() != expected {
            return Err(ChunkError::Truncated {
                expected,
                found: bytes.len(),
            });
        }

        // Validate the digest before interpreting a single payload byte.
        let mut h = Sha256::new();
        h.update(&bytes[..HEADER_LEN + payload_len]);
        let computed = h.finalize();
        let stored = &bytes[HEADER_LEN + payload_len..];
        if computed.bytes() != stored {
            return Err(ChunkError::DigestMismatch {
                expected: hex(stored),
                actual: computed.to_hex(),
            });
        }

        if let Some(expected_schema) = reader_schema {
            if expected_schema != schema_digest {
                return Err(ChunkError::SchemaMismatch {
                    chunk: schema_digest.to_hex(),
                    reader: expected_schema.to_hex(),
                });
            }
        }

        let payload = &bytes[HEADER_LEN..HEADER_LEN + payload_len];
        let mut values = Vec::with_capacity(count);
        for i in 0..count {
            let v = match dtype {
                DType::F32 => f64::from(f32::from_le_bytes([
                    payload[i * 4],
                    payload[i * 4 + 1],
                    payload[i * 4 + 2],
                    payload[i * 4 + 3],
                ])),
                DType::F64 => f64::from_le_bytes([
                    payload[i * 8],
                    payload[i * 8 + 1],
                    payload[i * 8 + 2],
                    payload[i * 8 + 3],
                    payload[i * 8 + 4],
                    payload[i * 8 + 5],
                    payload[i * 8 + 6],
                    payload[i * 8 + 7],
                ]),
                DType::I32 => f64::from(i32::from_le_bytes([
                    payload[i * 4],
                    payload[i * 4 + 1],
                    payload[i * 4 + 2],
                    payload[i * 4 + 3],
                ])),
                DType::U8 => f64::from(payload[i]),
            };
            if matches!(dtype, DType::F32 | DType::F64) && !v.is_finite() {
                return Err(ChunkError::NotFinite { index: i });
            }
            values.push(v);
        }

        Ok(Self {
            rows,
            cols,
            layout,
            dtype,
            schema_digest,
            values,
        })
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(char::from_digit(u32::from(b >> 4), 16).expect("nibble"));
        s.push(char::from_digit(u32::from(b & 0x0f), 16).expect("nibble"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema() -> Digest {
        ContentId::of(b"transition/1.0").digest()
    }

    fn chunk(layout: Layout, dtype: DType) -> Chunk {
        Chunk::new(
            3,
            2,
            layout,
            dtype,
            schema(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap()
    }

    #[test]
    fn round_trips_every_dtype_and_layout() {
        for layout in [Layout::StructureOfArrays, Layout::ArrayOfStructures] {
            for dtype in [DType::F32, DType::F64, DType::I32, DType::U8] {
                let c = chunk(layout, dtype);
                let bytes = c.encode();
                let back = Chunk::decode(&bytes, Some(schema()), ChunkLimits::default()).unwrap();
                assert_eq!(back, c, "{layout:?}/{dtype:?} did not round trip");
            }
        }
    }

    #[test]
    fn layout_changes_indexing_not_storage() {
        let soa = chunk(Layout::StructureOfArrays, DType::F64);
        let aos = chunk(Layout::ArrayOfStructures, DType::F64);
        // Same payload, different addressing.
        assert_eq!(soa.get(0, 0), Some(1.0));
        assert_eq!(soa.get(1, 0), Some(2.0));
        assert_eq!(soa.get(0, 1), Some(4.0));
        assert_eq!(aos.get(0, 0), Some(1.0));
        assert_eq!(aos.get(0, 1), Some(2.0));
        assert_eq!(aos.get(1, 0), Some(3.0));
        assert_eq!(soa.get(3, 0), None);
        assert_eq!(soa.get(0, 2), None);
    }

    #[test]
    fn a_flipped_payload_bit_is_caught_before_interpretation() {
        let mut bytes = chunk(Layout::StructureOfArrays, DType::F64).encode();
        let target = HEADER_LEN + 3;
        bytes[target] ^= 0x01;
        assert!(matches!(
            Chunk::decode(&bytes, Some(schema()), ChunkLimits::default()),
            Err(ChunkError::DigestMismatch { .. })
        ));
    }

    #[test]
    fn a_flipped_header_bit_is_caught_too() {
        let mut bytes = chunk(Layout::StructureOfArrays, DType::F64).encode();
        bytes[12] ^= 0x80; // rows
        let result = Chunk::decode(&bytes, Some(schema()), ChunkLimits::default());
        assert!(result.is_err(), "corrupt dimensions must not decode");
    }

    #[test]
    fn truncation_is_detected() {
        let bytes = chunk(Layout::StructureOfArrays, DType::F64).encode();
        for cut in [0usize, 1, HEADER_LEN - 1, HEADER_LEN, bytes.len() - 1] {
            assert!(
                Chunk::decode(&bytes[..cut], Some(schema()), ChunkLimits::default()).is_err(),
                "a {cut}-byte prefix must not decode"
            );
        }
    }

    #[test]
    fn foreign_bytes_are_rejected_by_magic() {
        let junk = vec![0u8; HEADER_LEN + DIGEST_LEN + 8];
        assert_eq!(
            Chunk::decode(&junk, None, ChunkLimits::default()),
            Err(ChunkError::BadMagic)
        );
    }

    #[test]
    fn a_future_version_is_refused_not_guessed() {
        let mut bytes = chunk(Layout::StructureOfArrays, DType::F64).encode();
        bytes[8..10].copy_from_slice(&999u16.to_le_bytes());
        assert!(matches!(
            Chunk::decode(&bytes, None, ChunkLimits::default()),
            Err(ChunkError::UnsupportedVersion { found: 999, .. })
        ));
    }

    #[test]
    fn unknown_layout_and_dtype_codes_are_refused() {
        let mut bytes = chunk(Layout::StructureOfArrays, DType::F64).encode();
        bytes[10] = 7;
        assert!(matches!(
            Chunk::decode(&bytes, None, ChunkLimits::default()),
            Err(ChunkError::UnknownLayout { code: 7 })
        ));
        let mut bytes = chunk(Layout::StructureOfArrays, DType::F64).encode();
        bytes[11] = 9;
        assert!(matches!(
            Chunk::decode(&bytes, None, ChunkLimits::default()),
            Err(ChunkError::UnknownDType { code: 9 })
        ));
    }

    #[test]
    fn a_chunk_from_another_schema_is_reported_not_migrated() {
        let bytes = chunk(Layout::StructureOfArrays, DType::F64).encode();
        let other = ContentId::of(b"transition/2.0").digest();
        assert!(matches!(
            Chunk::decode(&bytes, Some(other), ChunkLimits::default()),
            Err(ChunkError::SchemaMismatch { .. })
        ));
        // Tooling may inspect without interpreting.
        assert!(Chunk::decode(&bytes, None, ChunkLimits::default()).is_ok());
    }

    #[test]
    fn limits_are_enforced_before_allocation() {
        let bytes = chunk(Layout::StructureOfArrays, DType::F64).encode();
        let tight = ChunkLimits {
            max_bytes: 8,
            ..ChunkLimits::default()
        };
        assert!(matches!(
            Chunk::decode(&bytes, None, tight),
            Err(ChunkError::TooLarge { .. })
        ));
        let narrow = ChunkLimits {
            max_cols: 1,
            ..ChunkLimits::default()
        };
        assert!(matches!(
            Chunk::decode(&bytes, None, narrow),
            Err(ChunkError::DimensionsOutOfRange { .. })
        ));
    }

    #[test]
    fn a_declared_size_that_would_overflow_is_refused() {
        // rows and cols chosen so rows*cols*8 overflows a 32-bit product and
        // the declared payload could not possibly be present.
        let mut bytes = chunk(Layout::StructureOfArrays, DType::F64).encode();
        bytes[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        bytes[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            Chunk::decode(&bytes, None, ChunkLimits::default()),
            Err(ChunkError::DimensionsOutOfRange { .. })
        ));
    }

    #[test]
    fn nonfinite_and_out_of_range_values_are_refused_at_construction() {
        assert!(matches!(
            Chunk::new(
                1,
                1,
                Layout::StructureOfArrays,
                DType::F64,
                schema(),
                vec![f64::NAN]
            ),
            Err(ChunkError::NotFinite { index: 0 })
        ));
        assert!(matches!(
            Chunk::new(
                1,
                1,
                Layout::StructureOfArrays,
                DType::U8,
                schema(),
                vec![300.0]
            ),
            Err(ChunkError::ValueOutOfRange { .. })
        ));
        assert!(matches!(
            Chunk::new(
                1,
                1,
                Layout::StructureOfArrays,
                DType::I32,
                schema(),
                vec![0.5]
            ),
            Err(ChunkError::ValueOutOfRange { .. })
        ));
        assert!(matches!(
            Chunk::new(
                2,
                2,
                Layout::StructureOfArrays,
                DType::F64,
                schema(),
                vec![1.0]
            ),
            Err(ChunkError::ValueCountMismatch {
                expected: 4,
                found: 1
            })
        ));
    }

    #[test]
    fn an_empty_chunk_is_representable() {
        let c = Chunk::new(
            0,
            0,
            Layout::StructureOfArrays,
            DType::F64,
            schema(),
            vec![],
        )
        .unwrap();
        let back = Chunk::decode(&c.encode(), Some(schema()), ChunkLimits::default()).unwrap();
        assert_eq!(back.rows, 0);
        assert!(back.values.is_empty());
    }

    #[test]
    fn f32_storage_is_lossy_and_says_so() {
        let c = Chunk::new(
            1,
            1,
            Layout::StructureOfArrays,
            DType::F32,
            schema(),
            vec![0.1],
        )
        .unwrap();
        let back = Chunk::decode(&c.encode(), Some(schema()), ChunkLimits::default()).unwrap();
        // The round trip is exact for the *stored* precision, not for f64.
        assert_eq!(back.values[0], f64::from(0.1f32));
        assert_ne!(back.values[0], 0.1f64);
    }

    #[test]
    fn content_id_is_stable_across_encodes() {
        let c = chunk(Layout::StructureOfArrays, DType::F64);
        assert_eq!(c.content_id(), c.content_id());
        let mut other = c.clone();
        other.values[0] += 1.0;
        assert_ne!(c.content_id(), other.content_id());
    }
}

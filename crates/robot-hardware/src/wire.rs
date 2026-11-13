//! `robot-wire/1`: the documented protocol PLAN.md Phase 5 asks for after
//! the loopback and replay devices ("then an actual documented protocol").
//!
//! It is this project's own protocol, not a transcription of a vendor's:
//! C-02 (the first hardware target) is open, and SPECS.md Section 1.1
//! treats every field of the retired vendor reference as a hypothesis
//! requiring hardware verification. A vendor bridge (firmware or a
//! process next to the vendor SDK) speaks `robot-wire/1` on one side and the
//! vendor protocol on the other; that bridge is where the hypotheses get
//! verified, and this crate never links a vendor SDK.
//!
//! # Frame layout (little-endian)
//!
//! | Offset | Size | Field |
//! | --- | --- | --- |
//! | 0 | 2 | magic `b"RW"` |
//! | 2 | 1 | version, `1` |
//! | 3 | 1 | kind: `1` command, `2` feedback |
//! | 4 | 4 | sequence number, `u32`, +1 per frame per sender |
//! | 8 | 8 | timestamp, `i64` ns on the sender's monotonic clock: for a command the instant it was issued, for feedback the **sampling** instant |
//! | 16 | 4 | command validity, `u32` µs (commands; 0 in feedback) |
//! | 20 | 2 | flags: bit 0 sampling instant, bit 1 watchdog tripped, bit 2 e-stop active |
//! | 22 | 2 | width `n`, `u16` |
//! | 24 | 8n | values, `f64`: command targets, or measured positions |
//! | 24+8n | 4 | CRC-32 (IEEE) of every byte before it |
//!
//! A command is valid from its timestamp for its validity period; the
//! firmware side MUST treat an expired command as absent (its watchdog
//! then holds the safe state). Encoding and decoding work on caller-owned
//! buffers and never allocate (SPECS.md Section 5).

/// Frame header length, bytes.
pub const HEADER: usize = 24;
/// CRC trailer length, bytes.
pub const TRAILER: usize = 4;

/// Bytes of a frame of `width` values.
#[must_use]
pub const fn frame_len(width: usize) -> usize {
    HEADER + 8 * width + TRAILER
}

/// Frame kind.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// Host to device.
    Command = 1,
    /// Device to host.
    Feedback = 2,
}

/// Flag: the timestamp is a sampling instant (not an arrival time).
pub const FLAG_SAMPLING: u16 = 1;
/// Flag: the device watchdog has tripped (no valid command in time).
pub const FLAG_WATCHDOG: u16 = 1 << 1;
/// Flag: the device's emergency stop is active.
pub const FLAG_ESTOP: u16 = 1 << 2;

/// A frame's header fields.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Header {
    /// Kind.
    pub kind: Kind,
    /// Sequence number.
    pub seq: u32,
    /// Timestamp, ns.
    pub time_ns: i64,
    /// Command validity, µs.
    pub valid_us: u32,
    /// Flags.
    pub flags: u16,
    /// Value count.
    pub width: u16,
}

/// Why a frame was refused.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WireError {
    /// Shorter than its header says.
    Short,
    /// Not `b"RW"` version 1.
    Magic,
    /// Unknown kind.
    Kind,
    /// CRC mismatch.
    Crc,
    /// The width differs from the caller's buffer, or the output buffer is
    /// too small.
    Width,
    /// A value was non-finite.
    NotFinite,
}

impl core::fmt::Display for WireError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::Short => "short frame",
            Self::Magic => "bad magic or version",
            Self::Kind => "unknown frame kind",
            Self::Crc => "CRC mismatch",
            Self::Width => "width mismatch",
            Self::NotFinite => "non-finite value",
        })
    }
}

impl core::error::Error for WireError {}

/// CRC-32 (IEEE 802.3), table-free.
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

/// Encodes a frame into `out`, returning its length.
pub fn encode(header: &Header, values: &[f64], out: &mut [u8]) -> Result<usize, WireError> {
    let n = values.len();
    if usize::from(header.width) != n || out.len() < frame_len(n) {
        return Err(WireError::Width);
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(WireError::NotFinite);
    }
    out[0..2].copy_from_slice(b"RW");
    out[2] = 1;
    out[3] = header.kind as u8;
    out[4..8].copy_from_slice(&header.seq.to_le_bytes());
    out[8..16].copy_from_slice(&header.time_ns.to_le_bytes());
    out[16..20].copy_from_slice(&header.valid_us.to_le_bytes());
    out[20..22].copy_from_slice(&header.flags.to_le_bytes());
    out[22..24].copy_from_slice(&header.width.to_le_bytes());
    for (i, v) in values.iter().enumerate() {
        out[HEADER + 8 * i..HEADER + 8 * i + 8].copy_from_slice(&v.to_le_bytes());
    }
    let body = HEADER + 8 * n;
    let crc = crc32(&out[..body]);
    out[body..body + 4].copy_from_slice(&crc.to_le_bytes());
    Ok(body + 4)
}

/// Decodes `frame`, writing its values into `values` (which must be exactly
/// the frame's width).
pub fn decode(frame: &[u8], values: &mut [f64]) -> Result<Header, WireError> {
    if frame.len() < HEADER + TRAILER {
        return Err(WireError::Short);
    }
    if &frame[0..2] != b"RW" || frame[2] != 1 {
        return Err(WireError::Magic);
    }
    let width = u16::from_le_bytes([frame[22], frame[23]]);
    let n = usize::from(width);
    if frame.len() < frame_len(n) {
        return Err(WireError::Short);
    }
    let body = HEADER + 8 * n;
    let crc = u32::from_le_bytes([
        frame[body],
        frame[body + 1],
        frame[body + 2],
        frame[body + 3],
    ]);
    if crc != crc32(&frame[..body]) {
        return Err(WireError::Crc);
    }
    let kind = match frame[3] {
        1 => Kind::Command,
        2 => Kind::Feedback,
        _ => return Err(WireError::Kind),
    };
    if values.len() != n {
        return Err(WireError::Width);
    }
    for (i, v) in values.iter_mut().enumerate() {
        let b = &frame[HEADER + 8 * i..HEADER + 8 * i + 8];
        *v = f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
        if !v.is_finite() {
            return Err(WireError::NotFinite);
        }
    }
    let u32_at =
        |o: usize| u32::from_le_bytes([frame[o], frame[o + 1], frame[o + 2], frame[o + 3]]);
    Ok(Header {
        kind,
        seq: u32_at(4),
        time_ns: i64::from_le_bytes([
            frame[8], frame[9], frame[10], frame[11], frame[12], frame[13], frame[14], frame[15],
        ]),
        valid_us: u32_at(16),
        flags: u16::from_le_bytes([frame[20], frame[21]]),
        width,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(width: u16) -> Header {
        Header {
            kind: Kind::Command,
            seq: 7,
            time_ns: -5,
            valid_us: 20_000,
            flags: FLAG_SAMPLING,
            width,
        }
    }

    #[test]
    fn a_frame_round_trips_bit_exactly() {
        let mut buf = [0u8; frame_len(3)];
        let values = [0.1, -2.5, 1e-300];
        let n = encode(&header(3), &values, &mut buf).unwrap();
        assert_eq!(n, frame_len(3));
        let mut out = [0.0; 3];
        assert_eq!(decode(&buf, &mut out).unwrap(), header(3));
        assert_eq!(out, values);
    }

    #[test]
    fn every_single_bit_flip_is_caught() {
        let mut buf = [0u8; frame_len(2)];
        encode(&header(2), &[1.0, 2.0], &mut buf).unwrap();
        for byte in 0..buf.len() {
            for bit in 0..8 {
                let mut bad = buf;
                bad[byte] ^= 1 << bit;
                assert!(
                    decode(&bad, &mut [0.0; 2]).is_err(),
                    "byte {byte} bit {bit}"
                );
            }
        }
    }

    #[test]
    fn short_wrong_width_and_non_finite_frames_are_refused() {
        let mut buf = [0u8; frame_len(2)];
        encode(&header(2), &[1.0, 2.0], &mut buf).unwrap();
        assert_eq!(decode(&buf[..10], &mut [0.0; 2]), Err(WireError::Short));
        assert_eq!(decode(&buf, &mut [0.0; 3]), Err(WireError::Width));
        assert_eq!(
            encode(&header(2), &[1.0, f64::NAN], &mut buf),
            Err(WireError::NotFinite)
        );
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
    }
}

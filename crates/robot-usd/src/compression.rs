//! Decompression used by the USDC crate format.
//!
//! Three layers, all read from OpenUSD at the pinned commit (see `lib.rs`):
//!
//! * [`lz4_decompress_block`]: the LZ4 *block* format, the layout OpenUSD's
//!   vendored `pxr/base/tf/pxrLZ4/lz4.cpp` decodes.
//! * [`fast_decompress`]: the `TfFastCompression` container from
//!   `pxr/base/tf/fastCompression.cpp`. The first byte is a chunk count. Zero
//!   means one LZ4 block follows and fills the rest of the buffer. A nonzero
//!   count `n` means `n` chunks follow, each as a little-endian `i32` size and
//!   an LZ4 block.
//! * [`decode_ints`]: the integer codec of `pxr/usd/sdf/integerCoding.cpp`,
//!   which turns a decompressed buffer back into a list of 32- or 64-bit ints.

use crate::error::{malformed, Result};

/// Decompresses one LZ4 block. `max_out` bounds the output size, as the
/// `maxOutputSize` argument of `LZ4_decompress_safe` does.
pub fn lz4_decompress_block(src: &[u8], max_out: usize) -> Result<Vec<u8>> {
    let mut out: Vec<u8> = Vec::with_capacity(max_out.min(src.len().saturating_mul(4)));
    let mut i = 0usize;
    let n = src.len();
    loop {
        let token = *src.get(i).ok_or_else(|| trunc("token"))?;
        i += 1;

        // Literal run.
        let mut lit = (token >> 4) as usize;
        if lit == 15 {
            loop {
                let b = *src.get(i).ok_or_else(|| trunc("literal length"))?;
                i += 1;
                lit += b as usize;
                if b != 255 {
                    break;
                }
            }
        }
        let lit_end = i
            .checked_add(lit)
            .filter(|&e| e <= n)
            .ok_or_else(|| trunc("literals"))?;
        if out.len() + lit > max_out {
            return malformed("LZ4 output exceeds the stated size");
        }
        out.extend_from_slice(&src[i..lit_end]);
        i = lit_end;

        // The last sequence of a block carries literals only.
        if i == n {
            break;
        }

        // Match.
        if i + 2 > n {
            return Err(trunc("match offset"));
        }
        let offset = u16::from_le_bytes([src[i], src[i + 1]]) as usize;
        i += 2;
        if offset == 0 || offset > out.len() {
            return malformed("LZ4 match offset outside the decoded window");
        }
        let mut mlen = (token & 0x0f) as usize + 4;
        if (token & 0x0f) == 15 {
            loop {
                let b = *src.get(i).ok_or_else(|| trunc("match length"))?;
                i += 1;
                mlen += b as usize;
                if b != 255 {
                    break;
                }
            }
        }
        if out.len() + mlen > max_out {
            return malformed("LZ4 output exceeds the stated size");
        }
        let start = out.len() - offset;
        // Byte-by-byte copy: matches may overlap their own output.
        for k in 0..mlen {
            let b = out[start + k];
            out.push(b);
        }
    }
    Ok(out)
}

fn trunc(what: &str) -> crate::error::UsdError {
    crate::error::UsdError::Malformed(format!("truncated LZ4 block while reading {what}"))
}

/// Decompresses a `TfFastCompression` buffer. `max_out` is the buffer size
/// the caller allocated (OpenUSD passes the same bound).
pub fn fast_decompress(src: &[u8], max_out: usize) -> Result<Vec<u8>> {
    let (&chunks, rest) = src.split_first().ok_or_else(|| trunc("container header"))?;
    if chunks == 0 {
        let out = lz4_decompress_block(rest, max_out)?;
        return Ok(out);
    }
    let mut out = Vec::new();
    let mut p = rest;
    for _ in 0..chunks {
        if p.len() < 4 {
            return Err(trunc("chunk size"));
        }
        let size = i32::from_le_bytes([p[0], p[1], p[2], p[3]]);
        p = &p[4..];
        if size < 0 || size as usize > p.len() {
            return malformed("chunk size outside the buffer");
        }
        let (chunk, tail) = p.split_at(size as usize);
        let room = max_out.saturating_sub(out.len());
        out.extend_from_slice(&lz4_decompress_block(chunk, room)?);
        p = tail;
    }
    Ok(out)
}

/// Reads the integers stored as `Sdf_IntegerCompression` (`width == 4`) or
/// `Sdf_IntegerCompression64` (`width == 8`) from an already decompressed
/// buffer, as `_DecodeIntegers` does: a common value, then one 2-bit code
/// per integer (four per byte, least significant bits first), then the
/// variable-length deltas. Values are returned sign-extended; callers that
/// hold unsigned data truncate them.
pub fn decode_ints(data: &[u8], count: usize, width: usize) -> Result<Vec<i64>> {
    if width != 4 && width != 8 {
        return malformed("integer width must be 4 or 8");
    }
    let small = width / 4;
    let medium = width / 2;
    let common_len = width;
    let codes_len = (count * 2).div_ceil(8);
    let need = common_len + codes_len;
    if data.len() < need {
        return malformed("integer block shorter than its header");
    }
    let common = read_signed(&data[..common_len], width);
    let codes = &data[common_len..need];
    let mut vints = &data[need..];
    let mut out = Vec::with_capacity(count);
    let mut prev: i64 = 0;
    let mut i = 0usize;
    let mut group = 0usize;
    while i < count {
        let code_byte = codes[group];
        group += 1;
        let in_group = (count - i).min(4);
        for j in 0..in_group {
            let delta = match (code_byte >> (2 * j)) & 3 {
                0 => common,
                1 => take_signed(&mut vints, small)?,
                2 => take_signed(&mut vints, medium)?,
                _ => take_signed(&mut vints, width)?,
            };
            prev = wrap(prev.wrapping_add(delta), width);
            out.push(prev);
            i += 1;
        }
    }
    Ok(out)
}

fn take_signed(p: &mut &[u8], n: usize) -> Result<i64> {
    if p.len() < n {
        return malformed("integer block ends inside its variable-length section");
    }
    let v = read_signed(&p[..n], n);
    *p = &p[n..];
    Ok(v)
}

fn read_signed(b: &[u8], n: usize) -> i64 {
    match n {
        1 => b[0] as i8 as i64,
        2 => i16::from_le_bytes([b[0], b[1]]) as i64,
        4 => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64,
        _ => i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]),
    }
}

/// Keeps the arithmetic at the element width, as the C++ `SInt` does.
fn wrap(v: i64, width: usize) -> i64 {
    if width == 4 {
        v as i32 as i64
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lz4_literals_only_block() {
        // Token 0x30: three literals, no match (last sequence).
        let block = [0x30, b'a', b'b', b'c'];
        assert_eq!(lz4_decompress_block(&block, 16).unwrap(), b"abc");
    }

    #[test]
    fn lz4_overlapping_match() {
        // "a" literal, then a match of length 4+4=8 at offset 1 (run of 'a'),
        // then a final literal "z". Token 0x14 = 1 literal, match nibble 4.
        // Final sequence: token 0x10 = 1 literal, no match.
        let block = [0x14, b'a', 0x01, 0x00, 0x10, b'z'];
        let out = lz4_decompress_block(&block, 32).unwrap();
        assert_eq!(out, b"aaaaaaaaaz");
    }

    #[test]
    fn lz4_rejects_bad_offset_and_overflow() {
        // Match offset 5 with only 1 byte of history.
        let block = [0x10, b'a', 0x05, 0x00, 0x00];
        assert!(lz4_decompress_block(&block, 64).is_err());
        // Literal run longer than the buffer.
        assert!(lz4_decompress_block(&[0x50, b'a'], 64).is_err());
        // Output larger than the stated bound.
        assert!(lz4_decompress_block(&[0x30, b'a', b'b', b'c'], 2).is_err());
    }

    #[test]
    fn fast_container_single_block() {
        // Leading zero byte, then an LZ4 literal-only block.
        let buf = [0u8, 0x20, b'h', b'i'];
        assert_eq!(fast_decompress(&buf, 8).unwrap(), b"hi");
    }

    #[test]
    fn fast_container_chunked() {
        // Two chunks, each with an i32 size prefix.
        let mut buf = vec![2u8];
        buf.extend_from_slice(&3i32.to_le_bytes());
        buf.extend_from_slice(&[0x20, b'o', b'k']);
        buf.extend_from_slice(&3i32.to_le_bytes());
        buf.extend_from_slice(&[0x20, b'!', b'!']);
        assert_eq!(fast_decompress(&buf, 16).unwrap(), b"ok!!");
    }

    #[test]
    fn ints_common_and_small_deltas() {
        // Input [5, 6, 7, 8] encodes as deltas [5, 1, 1, 1] with common value 1.
        // Element 0 is Small (01, one int8 delta); elements 1..3 are Common (00).
        let mut data = Vec::new();
        data.extend_from_slice(&1i32.to_le_bytes()); // common
        data.push(0x01); // codes for four elements (LSB first)
        data.push(5); // int8 delta for element 0
        assert_eq!(decode_ints(&data, 4, 4).unwrap(), vec![5, 6, 7, 8]);
    }

    #[test]
    fn ints_large_and_negative_wrap() {
        // [100000, -3]: deltas [100000, -100003]. Both need 4 bytes (Large).
        // Codes: both Large (11) -> 0b1111 = 0x0f for two elements.
        let mut data = Vec::new();
        data.extend_from_slice(&0i32.to_le_bytes()); // common 0
        data.push(0b0000_1111);
        data.extend_from_slice(&100000i32.to_le_bytes());
        data.extend_from_slice(&(-100003i32).to_le_bytes());
        assert_eq!(decode_ints(&data, 2, 4).unwrap(), vec![100000, -3]);
    }

    #[test]
    fn ints_64_bit_medium() {
        // [70000, 70001] as 64-bit: first delta 70000 needs Medium (int32 in
        // the 64-bit codec), second delta 1 is common.
        let mut data = Vec::new();
        data.extend_from_slice(&1i64.to_le_bytes()); // common 1
        data.push(0b0000_0010); // element0 = Medium (10), element1 = Common (00)
        data.extend_from_slice(&70000i32.to_le_bytes());
        assert_eq!(decode_ints(&data, 2, 8).unwrap(), vec![70000, 70001]);
    }

    #[test]
    fn ints_reject_truncated_header() {
        assert!(decode_ints(&[1, 2], 4, 4).is_err());
    }
}

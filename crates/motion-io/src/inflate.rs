//! A DEFLATE (RFC 1951) decoder, for the compressed members of a `.npz`
//! archive.
//!
//! Authored against `std` under `docs/decisions/0003-std-only-default-closure.md`
//! rather than taking a compression crate. It implements the whole format
//! (stored, fixed-Huffman and dynamic-Huffman blocks) with canonical Huffman
//! decoding by code length, and bounds its output: a stream that would
//! decompress past `max_output` bytes is refused, which is the
//! decompression-bomb limit SPECS.md Section 13's "enforce size limits"
//! asks of untrusted input.

use core::fmt;

/// Why a stream could not be decoded.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum InflateError {
    /// The input ended inside a block.
    Truncated,
    /// A block type of 3, or another malformed construct.
    Malformed(&'static str),
    /// The output would exceed the caller's limit.
    TooLarge {
        /// The limit.
        limit: usize,
    },
}

impl fmt::Display for InflateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => f.write_str("deflate stream truncated"),
            Self::Malformed(what) => write!(f, "malformed deflate stream: {what}"),
            Self::TooLarge { limit } => write!(f, "deflate output exceeds {limit} bytes"),
        }
    }
}

impl core::error::Error for InflateError {}

struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    bit: u32,
    nbits: u32,
}

impl<'a> Bits<'a> {
    const fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            bit: 0,
            nbits: 0,
        }
    }

    fn need(&mut self, n: u32) -> Result<(), InflateError> {
        while self.nbits < n {
            let byte = *self.data.get(self.pos).ok_or(InflateError::Truncated)?;
            self.pos += 1;
            self.bit |= u32::from(byte) << self.nbits;
            self.nbits += 8;
        }
        Ok(())
    }

    fn bits(&mut self, n: u32) -> Result<u32, InflateError> {
        if n == 0 {
            return Ok(0);
        }
        self.need(n)?;
        let v = self.bit & ((1u32 << n) - 1);
        self.bit >>= n;
        self.nbits -= n;
        Ok(v)
    }

    fn align(&mut self) {
        self.bit = 0;
        self.nbits = 0;
    }
}

/// A canonical Huffman code: symbol counts per length and symbols sorted
/// by code.
struct Huffman {
    counts: [u16; 16],
    symbols: Vec<u16>,
}

impl Huffman {
    fn new(lengths: &[u8]) -> Result<Self, InflateError> {
        let mut counts = [0u16; 16];
        for &l in lengths {
            counts[usize::from(l)] += 1;
        }
        counts[0] = 0;
        let mut left: i32 = 1;
        for &c in &counts[1..] {
            left = left * 2 - i32::from(c);
            if left < 0 {
                return Err(InflateError::Malformed("over-subscribed code"));
            }
        }
        let mut offs = [0u16; 16];
        for len in 1..15 {
            offs[len + 1] = offs[len] + counts[len];
        }
        let mut symbols = vec![0u16; lengths.len()];
        for (sym, &l) in lengths.iter().enumerate() {
            if l != 0 {
                symbols[usize::from(offs[usize::from(l)])] = sym as u16;
                offs[usize::from(l)] += 1;
            }
        }
        Ok(Self { counts, symbols })
    }

    fn decode(&self, bits: &mut Bits<'_>) -> Result<u16, InflateError> {
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for len in 1..16 {
            code |= bits.bits(1)? as i32;
            let count = i32::from(self.counts[len]);
            if code - count < first {
                return Ok(self.symbols[(index + (code - first)) as usize]);
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err(InflateError::Malformed("invalid Huffman code"))
    }
}

const LEN_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LEN_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

/// Decodes a raw DEFLATE stream (no zlib or gzip header), refusing output
/// beyond `max_output` bytes.
pub fn inflate(data: &[u8], max_output: usize) -> Result<Vec<u8>, InflateError> {
    let mut bits = Bits::new(data);
    let mut out: Vec<u8> = Vec::new();
    loop {
        let last = bits.bits(1)? == 1;
        match bits.bits(2)? {
            0 => {
                bits.align();
                let p = bits.pos;
                let header = data.get(p..p + 4).ok_or(InflateError::Truncated)?;
                let len = usize::from(u16::from_le_bytes([header[0], header[1]]));
                let nlen = u16::from_le_bytes([header[2], header[3]]);
                if nlen != !(len as u16) {
                    return Err(InflateError::Malformed("stored length check"));
                }
                let block = data
                    .get(p + 4..p + 4 + len)
                    .ok_or(InflateError::Truncated)?;
                if out.len() + len > max_output {
                    return Err(InflateError::TooLarge { limit: max_output });
                }
                out.extend_from_slice(block);
                bits.pos = p + 4 + len;
            }
            1 => {
                let mut lengths = [0u8; 288];
                lengths[..144].fill(8);
                lengths[144..256].fill(9);
                lengths[256..280].fill(7);
                lengths[280..].fill(8);
                let lit = Huffman::new(&lengths)?;
                let dist = Huffman::new(&[5u8; 30])?;
                block(&mut bits, &lit, &dist, &mut out, max_output)?;
            }
            2 => {
                let hlit = bits.bits(5)? as usize + 257;
                let hdist = bits.bits(5)? as usize + 1;
                let hclen = bits.bits(4)? as usize + 4;
                const ORDER: [usize; 19] = [
                    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
                ];
                let mut cl = [0u8; 19];
                for &i in ORDER.iter().take(hclen) {
                    cl[i] = bits.bits(3)? as u8;
                }
                let clh = Huffman::new(&cl)?;
                let mut lengths = vec![0u8; hlit + hdist];
                let mut i = 0;
                while i < hlit + hdist {
                    let sym = clh.decode(&mut bits)?;
                    let (value, repeat) = match sym {
                        0..=15 => (sym as u8, 1),
                        16 => {
                            let prev = *lengths[..i]
                                .last()
                                .ok_or(InflateError::Malformed("repeat with no previous length"))?;
                            (prev, 3 + bits.bits(2)? as usize)
                        }
                        17 => (0, 3 + bits.bits(3)? as usize),
                        _ => (0, 11 + bits.bits(7)? as usize),
                    };
                    if i + repeat > hlit + hdist {
                        return Err(InflateError::Malformed("code lengths overrun"));
                    }
                    lengths[i..i + repeat].fill(value);
                    i += repeat;
                }
                if lengths[256] == 0 {
                    return Err(InflateError::Malformed("no end-of-block code"));
                }
                let lit = Huffman::new(&lengths[..hlit])?;
                let dist = Huffman::new(&lengths[hlit..])?;
                block(&mut bits, &lit, &dist, &mut out, max_output)?;
            }
            _ => return Err(InflateError::Malformed("block type 3")),
        }
        if last {
            return Ok(out);
        }
    }
}

fn block(
    bits: &mut Bits<'_>,
    lit: &Huffman,
    dist: &Huffman,
    out: &mut Vec<u8>,
    max_output: usize,
) -> Result<(), InflateError> {
    loop {
        let sym = lit.decode(bits)?;
        match sym {
            0..=255 => {
                if out.len() >= max_output {
                    return Err(InflateError::TooLarge { limit: max_output });
                }
                out.push(sym as u8);
            }
            256 => return Ok(()),
            257..=285 => {
                let i = usize::from(sym - 257);
                let len = usize::from(LEN_BASE[i]) + bits.bits(u32::from(LEN_EXTRA[i]))? as usize;
                let d = usize::from(dist.decode(bits)?);
                if d >= 30 {
                    return Err(InflateError::Malformed("distance symbol"));
                }
                let distance =
                    usize::from(DIST_BASE[d]) + bits.bits(u32::from(DIST_EXTRA[d]))? as usize;
                if distance > out.len() {
                    return Err(InflateError::Malformed("distance beyond output"));
                }
                if out.len() + len > max_output {
                    return Err(InflateError::TooLarge { limit: max_output });
                }
                let start = out.len() - distance;
                for k in 0..len {
                    out.push(out[start + k]);
                }
            }
            _ => return Err(InflateError::Malformed("literal/length symbol")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct BitWriter {
        out: Vec<u8>,
        acc: u64,
        n: u32,
    }

    impl BitWriter {
        /// Header fields and extra bits: least-significant bit first.
        fn put(&mut self, v: u32, bits: u32) {
            self.acc |= u64::from(v) << self.n;
            self.n += bits;
            while self.n >= 8 {
                self.out.push(self.acc as u8);
                self.acc >>= 8;
                self.n -= 8;
            }
        }

        /// Huffman codes: most-significant bit first.
        fn code(&mut self, code: u32, len: u32) {
            for k in (0..len).rev() {
                self.put((code >> k) & 1, 1);
            }
        }

        fn finish(mut self) -> Vec<u8> {
            if self.n > 0 {
                self.out.push(self.acc as u8);
            }
            self.out
        }
    }

    #[test]
    fn a_stored_block_round_trips() {
        let payload = b"hello, stored block";
        let mut s = vec![0b001]; // final, type 0
        s.extend_from_slice(&(payload.len() as u16).to_le_bytes());
        s.extend_from_slice(&(!(payload.len() as u16)).to_le_bytes());
        s.extend_from_slice(payload);
        assert_eq!(inflate(&s, 1024).unwrap(), payload);
    }

    #[test]
    fn a_fixed_huffman_block_with_an_overlapping_match_decodes() {
        // "abc", then (length 12, distance 3): the copy overlaps itself.
        let mut w = BitWriter::default();
        w.put(1, 1);
        w.put(1, 2);
        for c in b"abc" {
            w.code(0x30 + u32::from(*c), 8); // literals 0..143: 8 bits from 0x30
        }
        w.code(265 - 256, 7); // length symbol 265: base 11, 1 extra bit
        w.put(1, 1); // 11 + 1 = 12
        w.code(2, 5); // distance symbol 2: distance 3, no extra bits
        w.code(0, 7); // end of block
        assert_eq!(inflate(&w.finish(), 1024).unwrap(), b"abcabcabcabcabc");
    }

    #[test]
    fn a_dynamic_huffman_block_decodes() {
        // Literals 0..63, 256 and length symbol 265 get 7-bit codes; distance
        // symbol 2 gets the single 1-bit code. Data: 0..63, then
        // (length 12, distance 3).
        let (hlit, hdist) = (266usize, 3usize);
        let mut lengths = vec![0u8; hlit + hdist];
        lengths[..64].fill(7);
        lengths[256] = 7;
        lengths[265] = 7;
        lengths[hlit + 2] = 1;

        let mut w = BitWriter::default();
        w.put(1, 1);
        w.put(2, 2);
        w.put((hlit - 257) as u32, 5);
        w.put((hdist - 1) as u32, 5);
        w.put(19 - 4, 4);
        // Code-length code: symbols 0, 1, 7 and 18 at length 2, so their
        // canonical codes in symbol order are 00, 01, 10, 11.
        const ORDER: [usize; 19] = [
            16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
        ];
        for &i in &ORDER {
            w.put(if matches!(i, 0 | 1 | 7 | 18) { 2 } else { 0 }, 3);
        }
        let mut i = 0;
        while i < lengths.len() {
            match lengths[i] {
                0 => {
                    let run = lengths[i..]
                        .iter()
                        .take(138)
                        .take_while(|&&l| l == 0)
                        .count();
                    if run >= 11 {
                        w.code(0b11, 2);
                        w.put((run - 11) as u32, 7);
                        i += run;
                        continue;
                    }
                    w.code(0b00, 2);
                }
                1 => w.code(0b01, 2),
                _ => w.code(0b10, 2),
            }
            i += 1;
        }
        // Literal/length codes, canonical by symbol: 0..63 -> 0..63,
        // 256 -> 64, 265 -> 65.
        for s in 0..64 {
            w.code(s, 7);
        }
        w.code(65, 7);
        w.put(1, 1); // length 12
        w.code(0, 1); // distance symbol 2: distance 3
        w.code(64, 7);
        let mut expected: Vec<u8> = (0..64).collect();
        for k in 0..12 {
            expected.push(61 + (k % 3) as u8);
        }
        assert_eq!(inflate(&w.finish(), 1024).unwrap(), expected);
    }

    #[test]
    fn output_beyond_the_limit_is_refused() {
        let payload = [7u8; 20];
        let mut s = vec![0b001];
        s.extend_from_slice(&20u16.to_le_bytes());
        s.extend_from_slice(&(!20u16).to_le_bytes());
        s.extend_from_slice(&payload);
        assert_eq!(inflate(&s, 5), Err(InflateError::TooLarge { limit: 5 }));
    }

    #[test]
    fn truncation_and_block_type_3_are_errors() {
        assert_eq!(inflate(&[], 10), Err(InflateError::Truncated));
        assert!(matches!(
            inflate(&[0b111], 10),
            Err(InflateError::Malformed(_))
        ));
    }
}

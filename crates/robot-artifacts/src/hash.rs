//! SHA-256, authored, and the content identity built on it.
//!
//! SPECS.md Section 7.5 requires TOML manifests that "refer to content-addressed
//! blobs", and Section 13 requires treating downloaded data as untrusted input
//! and verifying checksums. A content hash is therefore not optional
//! infrastructure here, it is how an artifact is named.
//!
//! The implementation follows FIPS 180-4 directly. It is about ninety lines, has
//! published test vectors, and needs no dependency — which matters because
//! Appendix B.6 admits no dependency until its gates pass.

use core::fmt;

const K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

/// An incremental SHA-256 state.
#[derive(Clone)]
pub struct Sha256 {
    state: [u32; 8],
    buffer: [u8; 64],
    buffered: usize,
    length_bits: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    /// A fresh hasher.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: [
                0x6a09_e667,
                0xbb67_ae85,
                0x3c6e_f372,
                0xa54f_f53a,
                0x510e_527f,
                0x9b05_688c,
                0x1f83_d9ab,
                0x5be0_cd19,
            ],
            buffer: [0; 64],
            buffered: 0,
            length_bits: 0,
        }
    }

    /// Feeds bytes.
    pub fn update(&mut self, mut data: &[u8]) {
        self.length_bits = self
            .length_bits
            .wrapping_add((data.len() as u64).wrapping_mul(8));
        if self.buffered > 0 {
            let take = (64 - self.buffered).min(data.len());
            self.buffer[self.buffered..self.buffered + take].copy_from_slice(&data[..take]);
            self.buffered += take;
            data = &data[take..];
            if self.buffered == 64 {
                let block = self.buffer;
                self.compress(&block);
                self.buffered = 0;
            }
        }
        while data.len() >= 64 {
            let mut block = [0u8; 64];
            block.copy_from_slice(&data[..64]);
            self.compress(&block);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.buffer[..data.len()].copy_from_slice(data);
            self.buffered = data.len();
        }
    }

    /// Finishes and returns the digest.
    #[must_use]
    pub fn finalize(mut self) -> Digest {
        let bits = self.length_bits;
        self.update_raw(&[0x80]);
        while self.buffered != 56 {
            self.update_raw(&[0x00]);
        }
        self.update_raw(&bits.to_be_bytes());
        debug_assert_eq!(self.buffered, 0);

        let mut out = [0u8; 32];
        for (i, word) in self.state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        Digest(out)
    }

    /// Feeds padding bytes without counting them in the length.
    fn update_raw(&mut self, data: &[u8]) {
        for byte in data {
            self.buffer[self.buffered] = *byte;
            self.buffered += 1;
            if self.buffered == 64 {
                let block = self.buffer;
                self.compress(&block);
                self.buffered = 0;
            }
        }
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        for (s, v) in self.state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *s = s.wrapping_add(v);
        }
    }
}

/// A 256-bit digest.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Digest([u8; 32]);

impl Digest {
    /// The raw bytes.
    #[inline]
    #[must_use]
    pub const fn bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// The lowercase hex form.
    #[must_use]
    pub fn to_hex(self) -> String {
        let mut s = String::with_capacity(64);
        for b in self.0 {
            s.push(char::from_digit(u32::from(b >> 4), 16).expect("nibble is hex"));
            s.push(char::from_digit(u32::from(b & 0x0f), 16).expect("nibble is hex"));
        }
        s
    }

    /// Parses a lowercase hex digest.
    pub fn from_hex(text: &str) -> Result<Self, HashError> {
        if text.len() != 64 {
            return Err(HashError::BadLength { len: text.len() });
        }
        let mut out = [0u8; 32];
        let bytes = text.as_bytes();
        for (i, slot) in out.iter_mut().enumerate() {
            let hi = (bytes[i * 2] as char)
                .to_digit(16)
                .ok_or(HashError::NotHex { position: i * 2 })?;
            let lo = (bytes[i * 2 + 1] as char)
                .to_digit(16)
                .ok_or(HashError::NotHex {
                    position: i * 2 + 1,
                })?;
            *slot = ((hi << 4) | lo) as u8;
        }
        Ok(Self(out))
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl fmt::Debug for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sha256:{}", self.to_hex())
    }
}

/// Why a digest could not be parsed or verified.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum HashError {
    /// The hex string was not 64 characters.
    BadLength {
        /// Length supplied.
        len: usize,
    },
    /// A character was not a hex digit.
    NotHex {
        /// Position in the string.
        position: usize,
    },
    /// The identity string did not start with a supported algorithm prefix.
    UnknownAlgorithm {
        /// What was found before the colon.
        found: String,
    },
    /// Verification failed.
    Mismatch {
        /// Digest expected.
        expected: String,
        /// Digest computed.
        actual: String,
    },
}

impl fmt::Display for HashError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadLength { len } => write!(f, "digest has {len} characters, expected 64"),
            Self::NotHex { position } => write!(f, "character at {position} is not hex"),
            Self::UnknownAlgorithm { found } => write!(f, "unknown hash algorithm {found:?}"),
            Self::Mismatch { expected, actual } => {
                write!(f, "digest mismatch: expected {expected}, computed {actual}")
            }
        }
    }
}

impl core::error::Error for HashError {}

/// A content identity: an algorithm name and a digest.
///
/// Written as `sha256:<hex>`. The prefix is present so that a later algorithm
/// change is a parse-time distinction rather than a silent reinterpretation of
/// 64 hex characters.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ContentId {
    digest: Digest,
}

impl ContentId {
    /// The algorithm prefix this build writes.
    pub const ALGORITHM: &'static str = "sha256";

    /// Hashes bytes into a content identity.
    #[must_use]
    pub fn of(bytes: &[u8]) -> Self {
        let mut h = Sha256::new();
        h.update(bytes);
        Self {
            digest: h.finalize(),
        }
    }

    /// Wraps an existing digest.
    #[must_use]
    pub const fn from_digest(digest: Digest) -> Self {
        Self { digest }
    }

    /// The digest.
    #[inline]
    #[must_use]
    pub const fn digest(&self) -> Digest {
        self.digest
    }

    /// Parses `sha256:<hex>`.
    pub fn parse(text: &str) -> Result<Self, HashError> {
        let (algorithm, hex) = text
            .split_once(':')
            .ok_or_else(|| HashError::UnknownAlgorithm {
                found: text.to_owned(),
            })?;
        if algorithm != Self::ALGORITHM {
            return Err(HashError::UnknownAlgorithm {
                found: algorithm.to_owned(),
            });
        }
        Ok(Self {
            digest: Digest::from_hex(hex)?,
        })
    }

    /// Verifies that `bytes` hash to this identity.
    pub fn verify(&self, bytes: &[u8]) -> Result<(), HashError> {
        let actual = Self::of(bytes);
        if actual == *self {
            Ok(())
        } else {
            Err(HashError::Mismatch {
                expected: self.to_string(),
                actual: actual.to_string(),
            })
        }
    }

    /// A short prefix for human-facing output. Never used as an identity: a
    /// truncated hash is not a hash.
    #[must_use]
    pub fn short(&self) -> String {
        self.digest.to_hex()[..12].to_owned()
    }
}

impl fmt::Display for ContentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", Self::ALGORITHM, self.digest.to_hex())
    }
}

impl fmt::Debug for ContentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

/// Hashes a sequence of byte slices as if they were concatenated.
///
/// Used to derive a manifest digest over several blobs without materializing the
/// concatenation. Each part is length-prefixed so that `["ab", "c"]` and
/// `["a", "bc"]` produce different digests; without that, a manifest could be
/// rearranged without changing its identity.
#[must_use]
pub fn digest_parts(parts: &[&[u8]]) -> ContentId {
    let mut h = Sha256::new();
    for part in parts {
        h.update(&(part.len() as u64).to_be_bytes());
        h.update(part);
    }
    ContentId::from_digest(h.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_fips_180_4_vectors() {
        // The three standard vectors from the FIPS 180-4 examples.
        assert_eq!(
            ContentId::of(b"").digest().to_hex(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            ContentId::of(b"abc").digest().to_hex(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            ContentId::of(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")
                .digest()
                .to_hex(),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn long_input_vector() {
        // One million 'a' characters, the fourth FIPS example.
        let mut h = Sha256::new();
        let chunk = vec![b'a'; 1000];
        for _ in 0..1000 {
            h.update(&chunk);
        }
        assert_eq!(
            h.finalize().to_hex(),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn incremental_updates_match_a_single_update() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i % 251) as u8).collect();
        let whole = ContentId::of(&data);
        for split in [1usize, 63, 64, 65, 127, 128, 500] {
            let mut h = Sha256::new();
            for piece in data.chunks(split) {
                h.update(piece);
            }
            assert_eq!(
                ContentId::from_digest(h.finalize()),
                whole,
                "chunking by {split} changed the digest"
            );
        }
    }

    #[test]
    fn content_ids_round_trip_through_text() {
        let id = ContentId::of(b"robot");
        let text = id.to_string();
        assert!(text.starts_with("sha256:"));
        assert_eq!(ContentId::parse(&text).unwrap(), id);
        assert_eq!(id.short().len(), 12);
    }

    #[test]
    fn a_missing_or_wrong_algorithm_prefix_is_rejected() {
        let hex = ContentId::of(b"x").digest().to_hex();
        assert!(matches!(
            ContentId::parse(&hex),
            Err(HashError::UnknownAlgorithm { .. })
        ));
        assert!(matches!(
            ContentId::parse(&format!("md5:{hex}")),
            Err(HashError::UnknownAlgorithm { .. })
        ));
    }

    #[test]
    fn malformed_hex_is_rejected() {
        assert!(matches!(
            Digest::from_hex("abc"),
            Err(HashError::BadLength { len: 3 })
        ));
        let bad = format!("{}z", &ContentId::of(b"x").digest().to_hex()[..63]);
        assert!(matches!(
            Digest::from_hex(&bad),
            Err(HashError::NotHex { position: 63 })
        ));
    }

    #[test]
    fn verification_reports_a_mismatch() {
        let id = ContentId::of(b"expected");
        assert!(id.verify(b"expected").is_ok());
        let err = id.verify(b"tampered").unwrap_err();
        assert!(matches!(err, HashError::Mismatch { .. }));
    }

    #[test]
    fn part_lengths_are_committed_to() {
        // Without length prefixes these two would collide.
        assert_ne!(
            digest_parts(&[b"ab", b"c"]),
            digest_parts(&[b"a", b"bc"]),
            "parts must not be ambiguously concatenated"
        );
        assert_eq!(digest_parts(&[b"ab", b"c"]), digest_parts(&[b"ab", b"c"]));
    }
}

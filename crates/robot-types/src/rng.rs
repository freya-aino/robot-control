//! Deterministic, splittable random streams.
//!
//! SPECS.md Section 7.6 item 4: "Each environment and sampler MUST have an
//! independently recorded seed stream. Replaying the same stored transitions and
//! sampler seed MUST reproduce indices, weights and relabeled goals."
//!
//! The generator is PCG-XSH-RR 64/32 (O'Neill 2014), authored here because it is
//! twenty lines, has a published derivation and a documented stream parameter,
//! and because Appendix B.6 admits no dependency until its gates pass. The
//! stream parameter is what makes per-environment independence structural rather
//! than a hope about distant seeds.
//!
//! This is *not* a cryptographic generator and must not be used as one.

use core::fmt;

/// A recorded seed stream identity.
///
/// Carried in episode and run records so a replay can reconstruct exactly the
/// stream that produced a result.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct SeedStream {
    /// The root seed of the experiment.
    pub seed: u64,
    /// The stream selector, unique per consumer: environment index, sampler,
    /// randomization draw, noise source.
    pub stream: u64,
}

impl SeedStream {
    /// A stream from a root seed and selector.
    #[must_use]
    pub const fn new(seed: u64, stream: u64) -> Self {
        Self { seed, stream }
    }

    /// A derived stream for a named purpose under the same root seed.
    ///
    /// The selector is mixed with a hash of `purpose` so that adding a new
    /// consumer does not shift the streams of existing ones, which would
    /// silently invalidate every stored comparison.
    #[must_use]
    pub fn derive(self, purpose: &str) -> Self {
        Self {
            seed: self.seed,
            stream: self.stream ^ fnv1a64(purpose.as_bytes()),
        }
    }

    /// A per-environment stream, as required for parallel rollout workers.
    #[must_use]
    pub const fn for_env(self, env_index: u32) -> Self {
        Self {
            seed: self.seed,
            stream: self.stream.wrapping_add((env_index as u64) << 1),
        }
    }
}

impl fmt::Display for SeedStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "seed={:#018x} stream={:#018x}", self.seed, self.stream)
    }
}

/// 64-bit FNV-1a, used only to mix a name into a stream selector.
#[must_use]
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        hash ^= u64::from(*b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// A PCG-XSH-RR 64/32 generator.
///
/// The state is fully recoverable through [`Rng::snapshot`] and
/// [`Rng::restore`], which is what makes checkpoint/resume reproduce a run
/// rather than merely continue it (SPECS.md Section 9.2).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Rng {
    state: u64,
    increment: u64,
    origin: SeedStream,
}

/// A recoverable generator state.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RngSnapshot {
    /// Internal state word.
    pub state: u64,
    /// Stream increment, always odd.
    pub increment: u64,
    /// The stream this generator was constructed from.
    pub origin: SeedStream,
}

impl Rng {
    /// A generator for a recorded seed stream.
    #[must_use]
    pub fn new(origin: SeedStream) -> Self {
        // The increment must be odd for the LCG to have full period.
        let increment = (origin.stream << 1) | 1;
        let mut rng = Self {
            state: 0,
            increment,
            origin,
        };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(origin.seed);
        rng.next_u32();
        rng
    }

    /// The stream this generator was constructed from.
    #[inline]
    #[must_use]
    pub const fn origin(&self) -> SeedStream {
        self.origin
    }

    /// Captures the state for a checkpoint.
    #[inline]
    #[must_use]
    pub const fn snapshot(&self) -> RngSnapshot {
        RngSnapshot {
            state: self.state,
            increment: self.increment,
            origin: self.origin,
        }
    }

    /// Restores a captured state.
    ///
    /// Rejects an even increment, which cannot come from this generator and
    /// would silently halve the period.
    pub fn restore(snapshot: RngSnapshot) -> Result<Self, &'static str> {
        if snapshot.increment % 2 == 0 {
            return Err("PCG increment must be odd");
        }
        Ok(Self {
            state: snapshot.state,
            increment: snapshot.increment,
            origin: snapshot.origin,
        })
    }

    /// Next 32-bit output.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(self.increment);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Next 64-bit output.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        (u64::from(self.next_u32()) << 32) | u64::from(self.next_u32())
    }

    /// Uniform in `[0, 1)`.
    ///
    /// Built from 53 bits so every representable double in the interval is
    /// reachable with the correct probability.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / (1u64 << 53) as f64;
        ((self.next_u64() >> 11) as f64) * SCALE
    }

    /// Uniform in `[lo, hi)`. Returns `lo` when the interval is empty or
    /// either bound is NaN.
    #[inline]
    #[allow(clippy::neg_cmp_op_on_partial_ord)] // deliberately catches NaN too; see doc above
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        if !(hi > lo) {
            return lo;
        }
        lo + (hi - lo) * self.next_f64()
    }

    /// Uniform integer in `[0, bound)`, without modulo bias.
    ///
    /// Returns 0 when `bound` is 0.
    pub fn below(&mut self, bound: u32) -> u32 {
        if bound == 0 {
            return 0;
        }
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let r = self.next_u32();
            if r >= threshold {
                return r % bound;
            }
        }
    }

    /// Standard normal by the Marsaglia polar method.
    ///
    /// The polar method draws two variates at a time; only one is returned and
    /// the other is discarded, so that the number of generator calls per sample
    /// does not depend on how many samples preceded it. That costs some speed
    /// and buys reproducibility under a different call order.
    pub fn standard_normal(&mut self) -> f64 {
        loop {
            let u = self.uniform(-1.0, 1.0);
            let v = self.uniform(-1.0, 1.0);
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                return u * (-2.0 * s.ln() / s).sqrt();
            }
        }
    }

    /// Normal variate with the given mean and standard deviation.
    pub fn normal(&mut self, mean: f64, std_dev: f64) -> f64 {
        mean + std_dev * self.standard_normal()
    }

    /// A Bernoulli draw.
    pub fn bernoulli(&mut self, p: f64) -> bool {
        self.next_f64() < p
    }

    /// Fisher-Yates shuffle.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        let n = items.len();
        if n < 2 {
            return;
        }
        for i in (1..n).rev() {
            let j = self.below(u32::try_from(i + 1).unwrap_or(u32::MAX)) as usize;
            items.swap(i, j);
        }
    }

    /// Samples an index from unnormalized non-negative weights.
    ///
    /// Returns `None` when the weights are empty or sum to zero, which the
    /// prioritized sampler of SPECS.md Section 7.6 item 2 must report rather
    /// than resolve by picking index zero.
    pub fn weighted_index(&mut self, weights: &[f64]) -> Option<usize> {
        // `total` sums only values the filter already proved finite and
        // positive, so it can never be NaN here; a plain `<=` is exact.
        let total: f64 = weights.iter().filter(|w| w.is_finite() && **w > 0.0).sum();
        if total <= 0.0 {
            return None;
        }
        let mut target = self.next_f64() * total;
        for (i, w) in weights.iter().enumerate() {
            if !w.is_finite() || *w <= 0.0 {
                continue;
            }
            target -= w;
            if target <= 0.0 {
                return Some(i);
            }
        }
        // Reachable only through floating-point accumulation at the very end.
        weights.iter().rposition(|w| w.is_finite() && *w > 0.0)
    }
}

/// Reservoir sampling over a stream, with a recorded admission rule.
///
/// SPECS.md Section 9.4.2 item 8: the reservoir's "retention rule, capacity,
/// admission probability and seed stream are recorded, and a run that cannot
/// reproduce its sampled population is not reproducible."
#[derive(Clone, Debug)]
pub struct Reservoir<T> {
    capacity: usize,
    seen: u64,
    items: Vec<T>,
    rng: Rng,
}

impl<T> Reservoir<T> {
    /// A reservoir of the given capacity drawing from `stream`.
    #[must_use]
    pub fn new(capacity: usize, stream: SeedStream) -> Self {
        Self {
            capacity,
            seen: 0,
            items: Vec::with_capacity(capacity),
            rng: Rng::new(stream),
        }
    }

    /// Offers an item; returns whether it was retained.
    ///
    /// Classic algorithm R: the first `capacity` items are always retained, and
    /// item `n` thereafter is retained with probability `capacity / n`, giving
    /// every item seen an equal chance of being in the reservoir.
    pub fn offer(&mut self, item: T) -> bool {
        self.seen += 1;
        if self.items.len() < self.capacity {
            self.items.push(item);
            return true;
        }
        if self.capacity == 0 {
            return false;
        }
        let j = self.rng.below(u32::try_from(self.seen).unwrap_or(u32::MAX)) as usize;
        if j < self.capacity {
            self.items[j] = item;
            true
        } else {
            false
        }
    }

    /// The retained items.
    #[must_use]
    pub fn items(&self) -> &[T] {
        &self.items
    }

    /// How many items have been offered.
    #[must_use]
    pub const fn seen(&self) -> u64 {
        self.seen
    }

    /// The declared capacity.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// The probability the next offered item is admitted, which a run record
    /// reports alongside the capacity.
    #[must_use]
    pub fn admission_probability(&self) -> f64 {
        let next = self.seen + 1;
        if next <= self.capacity as u64 {
            1.0
        } else {
            self.capacity as f64 / next as f64
        }
    }

    /// The generator state, for a checkpoint.
    #[must_use]
    pub const fn rng_snapshot(&self) -> RngSnapshot {
        self.rng.snapshot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_stream_reproduces_the_same_sequence() {
        let s = SeedStream::new(42, 7);
        let a: Vec<u32> = (0..8).map(|_| Rng::new(s).next_u32()).collect();
        let mut rng = Rng::new(s);
        let b: Vec<u32> = (0..8).map(|_| rng.next_u32()).collect();
        assert_eq!(a[0], b[0]);
        let mut rng2 = Rng::new(s);
        let c: Vec<u32> = (0..8).map(|_| rng2.next_u32()).collect();
        assert_eq!(b, c);
    }

    #[test]
    fn different_streams_diverge_under_one_seed() {
        let mut a = Rng::new(SeedStream::new(1, 0));
        let mut b = Rng::new(SeedStream::new(1, 1));
        let va: Vec<u32> = (0..16).map(|_| a.next_u32()).collect();
        let vb: Vec<u32> = (0..16).map(|_| b.next_u32()).collect();
        assert_ne!(va, vb);
    }

    #[test]
    fn env_streams_are_independent() {
        let root = SeedStream::new(99, 0);
        let mut seqs = Vec::new();
        for env in 0..4u32 {
            let mut rng = Rng::new(root.for_env(env));
            seqs.push((0..8).map(|_| rng.next_u32()).collect::<Vec<_>>());
        }
        for i in 0..seqs.len() {
            for j in (i + 1)..seqs.len() {
                assert_ne!(seqs[i], seqs[j], "env {i} and {j} share a stream");
            }
        }
    }

    #[test]
    fn named_derivation_is_stable_and_distinct() {
        let root = SeedStream::new(5, 0);
        assert_eq!(root.derive("sampler"), root.derive("sampler"));
        assert_ne!(root.derive("sampler"), root.derive("randomization"));
        assert_eq!(root.derive("x").seed, root.seed);
    }

    #[test]
    fn snapshot_restores_the_exact_sequence() {
        let mut rng = Rng::new(SeedStream::new(3, 3));
        for _ in 0..17 {
            rng.next_u64();
        }
        let snap = rng.snapshot();
        let expect: Vec<u64> = (0..5).map(|_| rng.next_u64()).collect();
        let mut restored = Rng::restore(snap).unwrap();
        let got: Vec<u64> = (0..5).map(|_| restored.next_u64()).collect();
        assert_eq!(expect, got);
    }

    #[test]
    fn restoring_an_even_increment_is_rejected() {
        let snap = RngSnapshot {
            state: 1,
            increment: 2,
            origin: SeedStream::new(0, 0),
        };
        assert!(Rng::restore(snap).is_err());
    }

    #[test]
    fn uniform_stays_in_range() {
        let mut rng = Rng::new(SeedStream::new(11, 0));
        for _ in 0..10_000 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v));
            let u = rng.uniform(-2.0, 5.0);
            assert!((-2.0..5.0).contains(&u));
        }
        assert_eq!(rng.uniform(3.0, 3.0), 3.0);
    }

    #[test]
    fn below_is_unbiased_enough_and_bounded() {
        let mut rng = Rng::new(SeedStream::new(13, 0));
        let mut counts = [0u32; 5];
        for _ in 0..50_000 {
            let i = rng.below(5) as usize;
            counts[i] += 1;
        }
        for c in counts {
            assert!(
                (9_000..11_000).contains(&c),
                "bucket counts skewed: {counts:?}"
            );
        }
        assert_eq!(rng.below(0), 0);
        assert_eq!(rng.below(1), 0);
    }

    #[test]
    fn normal_has_the_right_moments() {
        let mut rng = Rng::new(SeedStream::new(17, 0));
        let n = 100_000;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for _ in 0..n {
            let v = rng.standard_normal();
            sum += v;
            sum_sq += v * v;
        }
        let mean = sum / f64::from(n);
        let var = sum_sq / f64::from(n) - mean * mean;
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((var - 1.0).abs() < 0.03, "variance {var}");
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut rng = Rng::new(SeedStream::new(19, 0));
        let mut v: Vec<u32> = (0..64).collect();
        rng.shuffle(&mut v);
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..64).collect::<Vec<_>>());
        assert_ne!(v, sorted, "shuffle left the slice in order");
        let mut one = [1];
        rng.shuffle(&mut one);
        assert_eq!(one, [1]);
    }

    #[test]
    fn weighted_index_reports_an_empty_distribution() {
        let mut rng = Rng::new(SeedStream::new(23, 0));
        assert_eq!(rng.weighted_index(&[]), None);
        assert_eq!(rng.weighted_index(&[0.0, 0.0]), None);
        assert_eq!(rng.weighted_index(&[f64::NAN, 0.0]), None);
        assert_eq!(rng.weighted_index(&[0.0, 1.0, 0.0]), Some(1));
    }

    #[test]
    fn weighted_index_follows_the_weights() {
        let mut rng = Rng::new(SeedStream::new(29, 0));
        let weights = [1.0, 3.0];
        let mut hits = [0u32; 2];
        for _ in 0..20_000 {
            hits[rng.weighted_index(&weights).unwrap()] += 1;
        }
        let ratio = f64::from(hits[1]) / f64::from(hits[0]);
        assert!((ratio - 3.0).abs() < 0.2, "ratio {ratio}");
    }

    #[test]
    fn reservoir_is_uniform_and_reproducible() {
        let stream = SeedStream::new(31, 1);
        let mut a = Reservoir::new(4, stream);
        let mut b = Reservoir::new(4, stream);
        for i in 0..100u32 {
            a.offer(i);
            b.offer(i);
        }
        assert_eq!(
            a.items(),
            b.items(),
            "same stream must reproduce the population"
        );
        assert_eq!(a.items().len(), 4);
        assert_eq!(a.seen(), 100);
        assert_eq!(a.capacity(), 4);
        assert!((a.admission_probability() - 4.0 / 101.0).abs() < 1e-12);
    }

    #[test]
    fn reservoir_retains_everything_below_capacity() {
        let mut r = Reservoir::new(8, SeedStream::new(37, 0));
        for i in 0..3u32 {
            assert!(r.offer(i));
        }
        assert_eq!(r.items(), &[0, 1, 2]);
        assert_eq!(r.admission_probability(), 1.0);
    }

    #[test]
    fn zero_capacity_reservoir_admits_nothing() {
        let mut r: Reservoir<u32> = Reservoir::new(0, SeedStream::new(41, 0));
        assert!(!r.offer(1));
        assert!(r.items().is_empty());
    }
}

use rand::distr::{Bernoulli, Distribution};
use rand::Rng;

/// The highest rate that draws the gaps between mutated genes; a higher one draws once per gene.
///
/// A gap costs one draw and a few multiplications per bit of the chromosome's length, so it pays
/// until most genes mutate. Flipping bits or adding Gaussian steps to 2000 individuals of 500
/// genes on 4 threads (Apple M5), drawing the gaps takes about 0.1 times as long as one draw per
/// gene at a rate of 0.01, 0.3 times at 0.05, 0.45 at 0.1, 0.65 at 0.2, 0.75 at 0.3 and 0.85 at
/// 0.5, and as long at 0.6. The threshold keeps a margin below where they cross.
///
/// Part of the seeded stream, like [`MIN_GAP_RATE`] and both bounds being included: changing it
/// changes every seeded run at a rate between the old and the new value, which is a breaking
/// change (see `tests/golden.rs`).
const MAX_GAP_RATE: f64 = 0.3;

/// The lowest rate, other than zero, that draws the gaps; a lower one draws once per gene.
///
/// The gaps are drawn from `1.0 - rate`, which is rounded to a multiple of `2^-53`: the rate it
/// stands for can be off by up to `2^-54`, about `5.6e-8` of `1e-9`, and below `2^-54` it rounds
/// to `1.0`, which would never mutate a gene.
///
/// Part of the seeded stream: see [`MAX_GAP_RATE`].
const MIN_GAP_RATE: f64 = 1e-9;

/// One step of the ladder per bit of a `usize`, so it covers any chromosome.
const STEPS: usize = usize::BITS as usize;

/// The genes a per-gene mutation changes: each gene of a chromosome, independently, with
/// probability `mutation_rate`. Build it once per `mutate`, then call [`MutatedGenes::for_each`]
/// once per individual, with that individual's generator.
///
/// From [`MIN_GAP_RATE`] to [`MAX_GAP_RATE`], both included, it draws how many genes are left
/// alone before the next mutated one, instead of drawing for every gene, so a chromosome takes
/// one draw per mutated gene, and one more to find there are none left. The genes mutated are
/// distributed the same way: in a row of independent trials that each succeed with probability
/// `p`, the number of failures before the next success is `k` or more with probability
/// `(1 - p)^k`, whatever happened before it. [`GapLadder`] says how a gap is drawn from that,
/// with exact floating-point operations only. Other rates draw once per gene, as 0.5.0 did,
/// except a zero rate, which draws nothing at all.
// One per `mutate` call, on the stack, shared by every individual: boxing the ladder would only
// add an allocation and an indirection.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Copy, Debug)]
pub(crate) enum MutatedGenes {
    /// No gene, without a draw: a zero rate.
    Nothing,
    /// The gap to the next mutated gene.
    Gaps(GapLadder),
    /// One draw per gene.
    EachGene(Bernoulli),
}

impl MutatedGenes {
    /// The genes `operator` mutates at `mutation_rate`.
    ///
    /// # Panics
    ///
    /// If `mutation_rate` is not within `[0, 1]`, with a message naming `operator`.
    pub(crate) fn new(operator: &str, mutation_rate: f64) -> Self {
        // `contains` is false for NaN too.
        if !(0.0..=1.0).contains(&mutation_rate) {
            panic!("{operator} needs a mutation_rate within [0, 1], got {mutation_rate}");
        }

        if mutation_rate == 0.0 {
            MutatedGenes::Nothing
        } else if (MIN_GAP_RATE..=MAX_GAP_RATE).contains(&mutation_rate) {
            MutatedGenes::Gaps(GapLadder::new(mutation_rate))
        } else {
            // Built once: `Rng::random_bool` builds the same distribution again at every call.
            MutatedGenes::EachGene(Bernoulli::new(mutation_rate).unwrap())
        }
    }

    /// Calls `mutate` with the index of each mutated gene of a chromosome of `genes` genes, in
    /// increasing order, and with `rng`, for `mutate` to draw the mutation itself from.
    pub(crate) fn for_each<R: Rng>(
        &self,
        genes: usize,
        rng: &mut R,
        mut mutate: impl FnMut(usize, &mut R),
    ) {
        match self {
            MutatedGenes::Nothing => {}
            MutatedGenes::Gaps(ladder) => {
                let mut next = 0;
                while next < genes {
                    let left = genes - next;
                    let skipped = ladder.gap(left, rng);
                    if skipped == left {
                        break;
                    }
                    next += skipped;
                    mutate(next, rng);
                    next += 1;
                }
            }
            MutatedGenes::EachGene(mutates) => {
                for gene in 0..genes {
                    if mutates.sample(rng) {
                        mutate(gene, rng);
                    }
                }
            }
        }
    }
}

/// Draws the gap to the next mutated gene, with exact operations only: one uniform
/// draw, multiplications and comparisons, each rounded the same way on every platform. No
/// `powi`, `ln` or `exp`, whose last bit can differ from one platform to another, and nothing
/// from another crate's sampler, which could change in a patch release.
///
/// `steps[j]` is `q^(2^j)`, with `q = 1 - rate`, built by squaring `q` again and again. The gap
/// takes one `u = rng.random::<f64>()`: in rand 0.9, the top 53 bits of a `u64` times `2^-53`,
/// one of the `2^53` multiples of `2^-53` in `[0, 1)`, all equally likely, built exactly. The gap
/// is the largest `g` up to the genes left with `q^g > u`, up to the rounding of the products,
/// found by binary lifting: from the highest bit of the genes left down, it adds `2^j` while the
/// product of the steps taken so far and `steps[j]` stays above `u`. So the gap is `k` or more
/// when `u < q^k`, with probability `q^k` rounded up to a multiple of `2^-53`: the distribution
/// of the failures before a success, as closely as `u` can tell. The ends of `[0, 1)` work out:
/// as `u < 1`, a gap of zero or more is certain, and `u = 0`, one chance in `2^53`, gives the
/// longest gap whose product is still above zero. A gap of all the genes left means none of them
/// mutates.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GapLadder {
    steps: [f64; STEPS],
}

impl GapLadder {
    fn new(mutation_rate: f64) -> Self {
        let mut steps = [0.0; STEPS];
        let mut power = 1.0 - mutation_rate;
        for step in steps.iter_mut() {
            *step = power;
            power *= power;
        }
        GapLadder { steps }
    }

    /// The genes skipped before the next mutated one, out of `left`: `left` when none mutates.
    fn gap<R: Rng>(&self, left: usize, rng: &mut R) -> usize {
        let u: f64 = rng.random();
        let mut kept = 1.0;
        let mut gap = 0;
        for bit in (0..usize::BITS - left.leading_zeros()).rev() {
            let step = 1 << bit;
            if gap + step <= left {
                let longer = kept * self.steps[bit as usize];
                if longer > u {
                    kept = longer;
                    gap += step;
                }
            }
        }
        gap
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rngs;

    /// How many times each of `genes` genes mutates over `chromosomes` chromosomes, checking
    /// that each chromosome gets its mutated genes in increasing order, each once.
    fn mutations_per_gene(mutation_rate: f64, genes: usize, chromosomes: usize) -> Vec<u32> {
        let mutated = MutatedGenes::new("test", mutation_rate);
        let rngs = Rngs::new(13);
        let mut counts = vec![0; genes];

        for index in 0..chromosomes {
            let mut last = None;
            mutated.for_each(genes, &mut rngs.rng(index), |gene, _| {
                assert!(last < Some(gene), "{gene} after {last:?}");
                last = Some(gene);
                counts[gene] += 1;
            });
        }
        counts
    }

    /// The first mutated genes of a chromosome of `genes` genes, with the generator of
    /// `Rngs::new(2026)` for index 0.
    fn first_mutated(mutation_rate: f64, genes: usize) -> Vec<usize> {
        let mut mutated = Vec::new();
        MutatedGenes::new("test", mutation_rate).for_each(
            genes,
            &mut Rngs::new(2026).rng(0),
            |gene, _| {
                if mutated.len() < 8 {
                    mutated.push(gene);
                }
            },
        );
        mutated
    }

    #[test]
    fn draws_the_gaps_between_the_thresholds_and_once_per_gene_outside() {
        let gaps = |rate| matches!(MutatedGenes::new("test", rate), MutatedGenes::Gaps(_));

        assert!(matches!(
            MutatedGenes::new("test", 0.0),
            MutatedGenes::Nothing
        ));
        assert!(gaps(MIN_GAP_RATE) && gaps(0.05) && gaps(MAX_GAP_RATE));
        assert!(!gaps(MIN_GAP_RATE / 2.0) && !gaps(0.300001) && !gaps(0.5) && !gaps(1.0));
    }

    #[test]
    fn mutates_every_gene_with_the_rate_either_way() {
        // 20 000 chromosomes of 20 genes: each gene is a binomial of 20 000 trials, so the first
        // and the last gene must mutate as often as the others, and all 20 together a binomial
        // of 400 000 trials, which also catches a small bias in the rate.
        for rate in [0.001, 0.01, 0.05, 0.1, MAX_GAP_RATE, 0.5] {
            let counts = mutations_per_gene(rate, 20, 20_000);

            let expected = rate * 20_000.0;
            let deviation = (expected * (1.0 - rate)).sqrt();
            for (gene, &count) in counts.iter().enumerate() {
                assert!(
                    (f64::from(count) - expected).abs() < 4.5 * deviation,
                    "rate {rate}, gene {gene}: {count}"
                );
            }

            let total = f64::from(counts.iter().sum::<u32>());
            let expected = rate * 400_000.0;
            let deviation = (expected * (1.0 - rate)).sqrt();
            assert!(
                (total - expected).abs() < 4.0 * deviation,
                "rate {rate}: {total} in all"
            );
        }
    }

    #[test]
    fn mutates_the_expected_share_of_a_long_chromosome() {
        // 300 chromosomes of 300 genes at 5%: 4 500 expected, with a standard deviation of
        // about 65.
        let mutated: u32 = mutations_per_gene(0.05, 300, 300).iter().sum();

        assert!(
            (f64::from(mutated) - 4500.0).abs() < 4.0 * 65.0,
            "{mutated}"
        );
    }

    /// The mutated genes are part of every seeded run: if this fails, the gaps are drawn
    /// differently, which is a breaking change (see `tests/golden.rs`). It pins the deep steps of
    /// the ladder, which the golden runs, with their few genes, never reach: at one gene in 500,
    /// at a tiny rate and at the lowest one, and at the highest, with a gene after another. And
    /// once per gene above it.
    #[test]
    fn gaps_are_stable_across_versions() {
        assert_eq!(
            first_mutated(0.002, 4_000),
            [107, 195, 367, 1173, 2302, 3047, 3126, 3804]
        );
        assert_eq!(
            first_mutated(1e-6, 4_000_000),
            [215956, 391249, 733940, 2345697]
        );
        assert_eq!(
            first_mutated(MIN_GAP_RATE, 4_000_000_000),
            [215956303, 391248829, 733939067, 2345696793]
        );
        assert_eq!(
            first_mutated(MAX_GAP_RATE, 30),
            [0, 1, 2, 7, 14, 19, 20, 24]
        );
        assert_eq!(first_mutated(0.5, 20), [3, 4, 5, 7, 8, 10, 12, 14]);
    }

    #[test]
    fn a_zero_rate_mutates_nothing_and_draws_nothing() {
        assert!(mutations_per_gene(0.0, 50, 100)
            .iter()
            .all(|&count| count == 0));

        let rngs = Rngs::new(13);
        let mut rng = rngs.rng(0);
        MutatedGenes::new("test", 0.0).for_each(50, &mut rng, |_, _| {});
        assert_eq!(rng.random::<u64>(), rngs.rng(0).random::<u64>());
    }

    #[test]
    fn a_rate_of_one_mutates_every_gene() {
        assert!(mutations_per_gene(1.0, 50, 100)
            .iter()
            .all(|&count| count == 100));
    }

    #[test]
    fn a_rate_below_the_floor_still_mutates() {
        // `1.0 - 1e-17` is `1.0`: a ladder would never mutate a gene.
        assert_eq!(1.0 - 1e-17, 1.0);
        assert!(matches!(
            MutatedGenes::new("test", 1e-17),
            MutatedGenes::EachGene(_)
        ));
    }

    #[test]
    fn an_empty_chromosome_mutates_nothing() {
        for rate in [0.05, 0.5, 1.0] {
            assert!(mutations_per_gene(rate, 0, 10).is_empty());
        }
    }

    #[test]
    #[should_panic(expected = "test needs a mutation_rate within [0, 1], got NaN")]
    fn a_nan_rate_panics() {
        MutatedGenes::new("test", f64::NAN);
    }

    #[test]
    #[should_panic(expected = "test needs a mutation_rate within [0, 1], got -0.1")]
    fn a_negative_rate_panics() {
        MutatedGenes::new("test", -0.1);
    }
}

use rand::distr::{Bernoulli, Distribution};
use rand::Rng;
use rand_distr::Geometric;

/// The highest rate that draws the gaps between mutated genes; a higher one draws once per gene.
///
/// A gap costs a few draws, so it only pays when it skips enough genes. Flipping bits or adding
/// Gaussian steps to 2000 individuals of 500 genes on 4 threads (Apple M5), drawing the gaps
/// takes about 0.1 times as long as one draw per gene at a rate of 0.01, 0.4 times at 0.05,
/// 0.75 times at 0.1 and as long at 0.15, then longer: 1.2 times at 0.2, 1.5 to 1.9 at 0.5. The
/// threshold keeps a margin below where they cross.
const MAX_GAP_RATE: f64 = 0.1;

/// The lowest rate, other than zero, that draws the gaps. Below it `1 - rate` keeps too few of
/// the rate's digits for `Geometric` to be accurate, and from about `1e-16`, where it rounds to
/// `1.0`, `Geometric::new` never returns. Such a rate draws once per gene instead.
const MIN_GAP_RATE: f64 = 1e-9;

/// The genes a per-gene mutation changes: each gene of a chromosome, independently, with
/// probability `mutation_rate`. Build it once per `mutate`, then call [`MutatedGenes::for_each`]
/// once per individual, with that individual's generator.
///
/// Up to [`MAX_GAP_RATE`] it draws how many genes are left alone before the next mutated one,
/// instead of drawing for every gene, so a chromosome of `n` genes takes a few draws per mutated
/// gene instead of `n`. The genes mutated are distributed the same way: in a row of independent
/// trials that each succeed with probability `p`, the number of failures before the next success
/// follows the geometric distribution of that `p`, whatever happened before it. A zero rate
/// draws nothing at all.
#[derive(Clone, Copy, Debug)]
pub(crate) enum MutatedGenes {
    /// The gap to the next mutated gene, drawn from `Geometric(mutation_rate)`.
    Gaps(Geometric),
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

        if mutation_rate == 0.0 || (MIN_GAP_RATE..=MAX_GAP_RATE).contains(&mutation_rate) {
            MutatedGenes::Gaps(Geometric::new(mutation_rate).unwrap())
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
            MutatedGenes::Gaps(gap) => {
                let mut next = 0;
                while next < genes {
                    // `u64::MAX`, without a draw, at a zero rate.
                    let skipped = gap.sample(rng);
                    if skipped >= (genes - next) as u64 {
                        break;
                    }
                    next += skipped as usize;
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

    #[test]
    fn draws_the_gaps_up_to_the_threshold_and_once_per_gene_above() {
        let gaps = |rate| matches!(MutatedGenes::new("test", rate), MutatedGenes::Gaps(_));

        assert!(gaps(0.0) && gaps(MIN_GAP_RATE) && gaps(0.05) && gaps(MAX_GAP_RATE));
        assert!(!gaps(MIN_GAP_RATE / 2.0) && !gaps(0.100001) && !gaps(0.5) && !gaps(1.0));
    }

    #[test]
    fn mutates_every_gene_with_the_rate_either_way() {
        // 20 000 chromosomes of 20 genes: each gene is a binomial of 20 000 trials, so the first
        // and the last gene must mutate as often as the others.
        for rate in [0.01, 0.05, MAX_GAP_RATE, 0.3] {
            let chromosomes = 20_000.0;
            let expected = rate * chromosomes;
            let deviation = (chromosomes * rate * (1.0 - rate)).sqrt();

            for (gene, &count) in mutations_per_gene(rate, 20, 20_000).iter().enumerate() {
                assert!(
                    (f64::from(count) - expected).abs() < 4.5 * deviation,
                    "rate {rate}, gene {gene}: {count}"
                );
            }
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
    fn a_tiny_rate_still_returns() {
        // `Geometric::new` would never return here.
        mutations_per_gene(1e-18, 50, 100);
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

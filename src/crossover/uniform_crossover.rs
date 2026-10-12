use rand::{distr::Bernoulli, distr::Distribution, Rng};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::ParallelSliceMut;

use crate::population::{Bin, Int, Real};
use crate::{Crossover, Individual, Rngs};

/// # Uniform Crossover
///
/// For each gene, it selects whether to swap them between the parents based on the `toss_probability`.
///
/// It works for [`Bin`], [`Int`] and [`Real`], but not for permuted individuals, as swapping
/// genes between two permutations rarely yields permutations.
///
/// # Panics
///
/// `crossover` panics if `crossover_rate` or `toss_probability` is not within `[0, 1]`.
#[derive(Clone)]
pub struct UniformCrossover {
    /// The probability of crossover occurring.
    pub crossover_rate: f64,
    /// The probability of swapping the genes.
    pub toss_probability: f64,
}

impl Default for UniformCrossover {
    fn default() -> Self {
        UniformCrossover {
            crossover_rate: 0.8,
            toss_probability: 0.5,
        }
    }
}

impl UniformCrossover {
    /// The crossover is the same for every individual whose genes can be swapped one by one,
    /// only the gene type changes.
    fn crossover_population<T: Individual>(&self, population: &mut [T], rngs: &Rngs) {
        let toss = Bernoulli::new(self.toss_probability).unwrap();

        population
            .par_chunks_exact_mut(2)
            .enumerate()
            .for_each(|(pair, chunk)| {
                let mut rng = rngs.rng(pair);
                if rng.random_bool(self.crossover_rate) {
                    let (first, second) = chunk.split_at_mut(1);

                    for (gene1, gene2) in first[0]
                        .get_mut_chromosome()
                        .iter_mut()
                        .zip(second[0].get_mut_chromosome())
                    {
                        if toss.sample(&mut rng) {
                            std::mem::swap(gene1, gene2);
                        }
                    }
                }
            });
    }
}

impl Crossover<Bin> for UniformCrossover {
    fn crossover(&self, population: &mut [Bin], rngs: &Rngs) {
        self.crossover_population(population, rngs);
    }
}

impl Crossover<Int> for UniformCrossover {
    fn crossover(&self, population: &mut [Int], rngs: &Rngs) {
        self.crossover_population(population, rngs);
    }
}

impl Crossover<Real> for UniformCrossover {
    fn crossover(&self, population: &mut [Real], rngs: &Rngs) {
        self.crossover_population(population, rngs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pairs of parents, the first all zeros and the second all ones.
    fn crossed(crossover: &UniformCrossover) -> Vec<Int> {
        let mut population: Vec<Int> = (0..200)
            .map(|i| Int::new(vec![i % 2; 300], (0, 1)))
            .collect();
        crossover.crossover(&mut population, &Rngs::new(9));
        population
    }

    /// How many genes the first parent of each pair took from the second.
    fn swapped(population: &[Int]) -> usize {
        population
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| {
                let (first, second) = (pair[0].get_chromosome(), pair[1].get_chromosome());
                // Each position keeps one gene of each parent.
                assert!(first.iter().zip(second).all(|(a, b)| a + b == 1));
                first.iter().filter(|&&gene| gene == 1).count()
            })
            .sum()
    }

    #[test]
    fn swaps_the_expected_share_of_genes() {
        let crossover = UniformCrossover {
            crossover_rate: 1.0,
            toss_probability: 0.3,
        };

        // 30 000 positions at 30%: 9 000 expected, with a standard deviation of about 80.
        let swapped = swapped(&crossed(&crossover)) as f64;
        assert!((swapped - 9000.0).abs() < 4.0 * 80.0, "{swapped}");
    }

    #[test]
    fn a_certain_toss_swaps_whole_chromosomes() {
        let crossover = UniformCrossover {
            crossover_rate: 1.0,
            toss_probability: 1.0,
        };

        assert_eq!(swapped(&crossed(&crossover)), 100 * 300);
    }

    #[test]
    fn a_zero_crossover_rate_changes_nothing() {
        let crossover = UniformCrossover {
            crossover_rate: 0.0,
            toss_probability: 1.0,
        };

        assert_eq!(swapped(&crossed(&crossover)), 0);
    }
}

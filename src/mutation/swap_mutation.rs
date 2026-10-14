use rand::Rng;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::mutation::MutatedGenes;
use crate::{Individual, Mutation, Rngs};

/// # Swap Mutation
///
/// For each gene, with probability `mutation_rate`, swaps it with a gene at a random position
/// of the same individual, so a permutation stays one.
///
/// # Panics
///
/// `mutate` panics if `mutation_rate` is not within `[0, 1]`.
#[derive(Clone)]
pub struct SwapMutation {
    pub mutation_rate: f64,
}

impl SwapMutation {
    pub fn new(mutation_rate: f64) -> Self {
        Self { mutation_rate }
    }
}

impl Default for SwapMutation {
    fn default() -> Self {
        SwapMutation {
            mutation_rate: 0.05,
        }
    }
}

impl<T: Individual> Mutation<T> for SwapMutation {
    fn mutate(&self, population: &mut [T], rngs: &Rngs) {
        let mutated = MutatedGenes::new("SwapMutation", self.mutation_rate);

        population
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, individual)| {
                let mut rng = rngs.rng(index);
                let genes = individual.get_chromosome().len();
                mutated.for_each(genes, &mut rng, |j, rng| {
                    let swap_with = rng.random_range(0..genes);

                    let temp = individual.get_gene(j);
                    individual.set_gene(j, individual.get_gene(swap_with));
                    individual.set_gene(swap_with, temp);
                });
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::population::{is_permutation, IntPerm};

    fn identity() -> Vec<i64> {
        (1..=50).collect()
    }

    fn mutated(mutation_rate: f64) -> Vec<IntPerm> {
        let mut population = vec![IntPerm::new(identity()); 100];
        SwapMutation::new(mutation_rate).mutate(&mut population, &Rngs::new(5));
        population
    }

    #[test]
    fn keeps_every_individual_a_permutation() {
        for rate in [0.05, 0.2] {
            let population = mutated(rate);

            assert!(population
                .iter()
                .all(|individual| is_permutation(individual.get_chromosome())));
            assert!(population
                .iter()
                .any(|individual| *individual.get_chromosome() != identity()));
        }
    }

    #[test]
    fn swaps_the_expected_share_of_two_gene_individuals() {
        // Each of the two genes mutates with probability `rate` and then swaps with the other
        // one half the time, so an individual ends up swapped (one effective swap, not zero or
        // two) with probability `rate * (1 - rate / 2)`.
        for rate in [0.05, 0.3] {
            let mut population = vec![IntPerm::new(vec![1, 2]); 20_000];
            SwapMutation::new(rate).mutate(&mut population, &Rngs::new(5));

            let swapped = population
                .iter()
                .filter(|individual| individual.get_chromosome() == [2, 1])
                .count() as f64;
            let share = rate * (1.0 - rate / 2.0);
            let expected = share * 20_000.0;
            let deviation = (expected * (1.0 - share)).sqrt();
            assert!(
                (swapped - expected).abs() < 4.0 * deviation,
                "rate {rate}: {swapped}"
            );
        }
    }

    #[test]
    fn a_zero_rate_changes_nothing() {
        assert!(mutated(0.0)
            .iter()
            .all(|individual| *individual.get_chromosome() == identity()));
    }

    #[test]
    #[should_panic(expected = "SwapMutation needs a mutation_rate within [0, 1], got 1.5")]
    fn a_rate_above_one_panics() {
        mutated(1.5);
    }
}

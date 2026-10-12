use rand::distr::{Bernoulli, Distribution};
use rand::Rng;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

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
        // Built once: `Rng::random_bool` builds the same distribution again at every call.
        let mutates = Bernoulli::new(self.mutation_rate).unwrap();

        population
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, individual)| {
                let mut rng = rngs.rng(index);
                for j in 0..individual.get_chromosome().len() {
                    if mutates.sample(&mut rng) {
                        let swap_with = rng.random_range(0..individual.get_chromosome().len());

                        let temp = individual.get_gene(j);
                        individual.set_gene(j, individual.get_gene(swap_with));
                        individual.set_gene(swap_with, temp);
                    }
                }
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
        let population = mutated(0.2);

        assert!(population
            .iter()
            .all(|individual| is_permutation(individual.get_chromosome())));
        assert!(population
            .iter()
            .any(|individual| *individual.get_chromosome() != identity()));
    }

    #[test]
    fn a_zero_rate_changes_nothing() {
        assert!(mutated(0.0)
            .iter()
            .all(|individual| *individual.get_chromosome() == identity()));
    }
}

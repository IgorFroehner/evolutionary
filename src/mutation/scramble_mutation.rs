use crate::{Individual, Mutation, Rngs};
use rand::seq::SliceRandom;
use rand::Rng;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

/// # Scramble Mutation
///
/// For each individual in the population it has `mutation_rate` probability of scrambling a random
/// subsequence of the chromosome.
#[derive(Clone)]
pub struct ScrambleMutation {
    pub mutation_rate: f64,
}

impl Default for ScrambleMutation {
    fn default() -> Self {
        Self {
            mutation_rate: 0.05,
        }
    }
}

impl<T: Individual> Mutation<T> for ScrambleMutation {
    fn mutate(&self, population: &mut [T], rngs: &Rngs) {
        population
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, individual)| {
                let mut rng = rngs.rng(index);
                if rng.random_bool(self.mutation_rate) {
                    let from = rng.random_range(0..individual.get_chromosome().len());
                    let to = rng.random_range(from..individual.get_chromosome().len());

                    if from != to {
                        individual.get_mut_chromosome()[from..=to].shuffle(&mut rng);
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
        ScrambleMutation { mutation_rate }.mutate(&mut population, &Rngs::new(5));
        population
    }

    #[test]
    fn keeps_every_individual_a_permutation() {
        let population = mutated(1.0);

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

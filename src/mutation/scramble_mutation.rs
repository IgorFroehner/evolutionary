use rand::{Rng, thread_rng};
use rand::seq::SliceRandom;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use crate::{Individual, Mutation};

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
    fn mutate(&self, population: &mut Vec<T>) {
        population.par_iter_mut().for_each_init(
            || thread_rng(),
            |rng, individual| {
                if rng.gen_bool(self.mutation_rate) {
                    let from = rng.gen_range(0..individual.get_chromosome().len());
                    let to = rng.gen_range(from..individual.get_chromosome().len());

                    if from != to {
                        individual.get_mut_chromosome()[from..=to].shuffle(rng);
                    }
                }
            }
        );
    }
}

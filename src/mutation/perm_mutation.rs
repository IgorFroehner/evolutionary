use rand::{thread_rng, Rng};
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};

use crate::population::IntPerm;

use super::Mutation;

#[derive(Clone)]
pub struct PermMutation {
    mutation_rate: f64,
}

impl PermMutation {
    pub fn new(mutation_rate: f64) -> Self {
        PermMutation { mutation_rate }
    }
}

impl Default for PermMutation {
    fn default() -> Self {
        PermMutation {
            mutation_rate: 0.05,
        }
    }
}

impl Mutation<IntPerm> for PermMutation {
    fn mutate(&self, population: &mut Vec<IntPerm>) {
        population.par_iter_mut().for_each_init(
            || thread_rng(),
            |rng, individual| {
                for j in 0..individual.chromosome.len() {
                    if rng.gen_bool(self.mutation_rate) {
                        let swap_with = rng.gen_range(0..individual.chromosome.len());

                        let temp = individual.chromosome[j];
                        individual.chromosome[j] = individual.chromosome[swap_with];
                        individual.chromosome[swap_with] = temp;
                    }
                }
            },
        );
    }
}

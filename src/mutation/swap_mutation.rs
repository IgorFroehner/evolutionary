use rand::{thread_rng, Rng};
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};

use crate::{Mutation, population::Real};

#[derive(Clone)]
pub struct SwapMutation {
    pub mutation_rate: f64,
}

impl Default for SwapMutation {
    fn default() -> Self {
        SwapMutation {
            mutation_rate: 0.05,
        }
    }
}

impl Mutation<Real> for SwapMutation {
    fn mutate(&self, population: &mut Vec<Real>) {
        population
            .par_iter_mut()
            .for_each_init(|| thread_rng(), |rng, individual| {
                for j in 0..individual.chromosome.len() {
                    if rng.gen_bool(self.mutation_rate) {
                        let swap_with = rng.gen_range(0..individual.chromosome.len());

                        let temp = individual.chromosome[j];
                        individual.chromosome[j] = individual.chromosome[swap_with];
                        individual.chromosome[swap_with] = temp;
                    }
                }
            });
    }
}

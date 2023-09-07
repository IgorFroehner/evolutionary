use rand::{thread_rng, Rng};
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};

use crate::population::IntPerm;

use super::Mutation;

#[derive(Clone)]
pub struct PermMutation {
    pub mutation_rate: f64,
}

impl Default for PermMutation {
    fn default() -> Self {
        PermMutation { mutation_rate: 0.05 }
    }
}

impl Mutation<IntPerm> for PermMutation {
    fn mutate(&self, population: &mut Vec<IntPerm>) {
        population.par_iter_mut().for_each_init(
            || thread_rng(),
            |rng, individual| {
                for j in 0..individual.0.len() {
                    if rng.gen_bool(self.mutation_rate) {
                        let swap_with = rng.gen_range(0..individual.0.len());

                        let temp = individual.0[j];
                        individual.0[j] = individual.0[swap_with];
                        individual.0[swap_with] = temp;
                    }
                }
            },
        );
    }
}

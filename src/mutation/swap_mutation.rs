use rand::{thread_rng, Rng};
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};

use crate::{Individual, Mutation};

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
    fn mutate(&self, population: &mut Vec<T>) {
        population
            .par_iter_mut()
            .for_each_init(|| thread_rng(), |rng, individual| {
                for j in 0..individual.get_chromosome().len() {
                    if rng.gen_bool(self.mutation_rate) {
                        let swap_with = rng.gen_range(0..individual.get_chromosome().len());

                        let temp = individual.get_chromosome()[j];
                        individual.set_gene(j, individual.get_chromosome()[swap_with]);
                        individual.set_gene(swap_with, temp);
                    }
                }
            });
    }
}

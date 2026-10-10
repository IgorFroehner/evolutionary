use rand::Rng;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::{Individual, Mutation, Rngs};

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
    fn mutate(&self, population: &mut Vec<T>, rngs: &Rngs) {
        population
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, individual)| {
                let mut rng = rngs.rng(index);
                for j in 0..individual.get_chromosome().len() {
                    if rng.random_bool(self.mutation_rate) {
                        let swap_with = rng.random_range(0..individual.get_chromosome().len());

                        let temp = individual.get_gene(j);
                        individual.set_gene(j, individual.get_gene(swap_with));
                        individual.set_gene(swap_with, temp);
                    }
                }
            });
    }
}

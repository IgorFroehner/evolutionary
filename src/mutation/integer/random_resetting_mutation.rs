use rand::Rng;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::mutation::random_resetting_mutation::RandomResettingMutation;
use crate::{population::Int, Individual, Mutation};
use crate::Rngs;

impl Mutation<Int> for RandomResettingMutation {
    fn mutate(&self, population: &mut Vec<Int>, rngs: &Rngs) {
        population
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, individual)| {
                let mut rng = rngs.rng(index);
                for j in 0..individual.get_chromosome().len() {
                    if rng.random_bool(self.mutation_rate) {
                        individual
                            .set_gene(j, rng.random_range(individual.range.0..individual.range.1));
                    }
                }
            });
    }
}

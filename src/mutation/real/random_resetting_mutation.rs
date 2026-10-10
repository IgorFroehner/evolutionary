use rand::Rng;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::{population::Real, Individual, Mutation};
use crate::mutation::random_resetting_mutation::RandomResettingMutation;
use crate::Rngs;

impl Mutation<Real> for RandomResettingMutation {
    fn mutate(&self, population: &mut Vec<Real>, rngs: &Rngs) {
        population
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, individual)| {
                let mut rng = rngs.rng(index);
                for j in 0..individual.chromosome.len() {
                    if rng.random_bool(self.mutation_rate) {
                        individual
                            .set_gene(j, rng.random_range(individual.range.0..individual.range.1));
                    }
                }
            });
    }
}

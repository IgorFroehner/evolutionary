use rand::distr::{Bernoulli, Distribution};
use rand::Rng;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::{population::Real, Individual, Mutation};
use crate::mutation::random_resetting_mutation::RandomResettingMutation;
use crate::Rngs;

impl Mutation<Real> for RandomResettingMutation {
    fn mutate(&self, population: &mut [Real], rngs: &Rngs) {
        // Built once: `Rng::random_bool` builds the same distribution again at every call.
        let mutates = Bernoulli::new(self.mutation_rate).unwrap();

        population
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, individual)| {
                let mut rng = rngs.rng(index);
                for j in 0..individual.chromosome.len() {
                    if mutates.sample(&mut rng) {
                        individual
                            .set_gene(j, rng.random_range(individual.range.0..individual.range.1));
                    }
                }
            });
    }
}

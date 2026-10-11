use rand::distr::{Bernoulli, Distribution};
use rand::Rng;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::mutation::random_resetting_mutation::RandomResettingMutation;
use crate::{population::Int, Individual, Mutation};
use crate::Rngs;

impl Mutation<Int> for RandomResettingMutation {
    fn mutate(&self, population: &mut [Int], rngs: &Rngs) {
        // Built once: `Rng::random_bool` builds the same distribution again at every call.
        let mutates = Bernoulli::new(self.mutation_rate).unwrap();

        population
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, individual)| {
                let mut rng = rngs.rng(index);
                for j in 0..individual.get_chromosome().len() {
                    if mutates.sample(&mut rng) {
                        // Inclusive, like `Int::generate_member`.
                        individual
                            .set_gene(j, rng.random_range(individual.range.0..=individual.range.1));
                    }
                }
            });
    }
}

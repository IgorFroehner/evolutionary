use rand::{thread_rng, Rng};
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};

use crate::mutation::random_resetting_mutation::RandomResettingMutation;
use crate::{population::Int, Individual, Mutation};

impl Mutation<Int> for RandomResettingMutation {
    fn mutate(&self, population: &mut Vec<Int>) {
        population.par_iter_mut().for_each_init(
            || thread_rng(),
            |rng, individual| {
                for j in 0..individual.get_chromosome().len() {
                    if rng.gen_bool(self.mutation_rate) {
                        individual
                            .set_gene(j, rng.gen_range(individual.range.0..individual.range.1));
                    }
                }
            },
        );
    }
}

use rand::Rng;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::mutation::random_resetting_mutation::RandomResettingMutation;
use crate::mutation::MutatedGenes;
use crate::Rngs;
use crate::{population::Int, Individual, Mutation};

impl Mutation<Int> for RandomResettingMutation {
    fn mutate(&self, population: &mut [Int], rngs: &Rngs) {
        let mutated = MutatedGenes::new("RandomResettingMutation", self.mutation_rate);

        population
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, individual)| {
                let mut rng = rngs.rng(index);
                let genes = individual.get_chromosome().len();
                mutated.for_each(genes, &mut rng, |j, rng| {
                    // Inclusive, like `Int::generate_member`.
                    individual
                        .set_gene(j, rng.random_range(individual.range.0..=individual.range.1));
                });
            });
    }
}

use rand::Rng;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::mutation::random_resetting_mutation::RandomResettingMutation;
use crate::mutation::MutatedGenes;
use crate::Rngs;
use crate::{population::Real, Individual, Mutation};

impl Mutation<Real> for RandomResettingMutation {
    fn mutate(&self, population: &mut [Real], rngs: &Rngs) {
        let mutated = MutatedGenes::new("RandomResettingMutation", self.mutation_rate);

        population
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, individual)| {
                let mut rng = rngs.rng(index);
                let genes = individual.chromosome.len();
                mutated.for_each(genes, &mut rng, |j, rng| {
                    individual
                        .set_gene(j, rng.random_range(individual.range.0..individual.range.1));
                });
            });
    }
}

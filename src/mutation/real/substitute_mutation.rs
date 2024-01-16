use rand::{thread_rng, Rng};
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};

use crate::{population::Real, Individual, Mutation};

/// # Substitute Mutation
///
/// For each gene in the real representation it has `mutation_rate` probability of substituting
/// the gene with a random value.
#[derive(Clone)]
pub struct SubstituteMutation {
    pub mutation_rate: f64,
}

impl Default for SubstituteMutation {
    fn default() -> Self {
        SubstituteMutation {
            mutation_rate: 0.05,
        }
    }
}

impl Mutation<Real> for SubstituteMutation {
    fn mutate(&self, population: &mut Vec<Real>) {
        population.par_iter_mut().for_each_init(
            || thread_rng(),
            |rng, individual| {
                for j in 0..individual.chromosome.len() {
                    if rng.gen_bool(self.mutation_rate) {
                        individual
                            .set_gene(j, rng.gen_range(individual.range.0..individual.range.1));
                    }
                }
            },
        );
    }
}

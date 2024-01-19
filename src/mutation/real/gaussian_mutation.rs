use crate::prelude::Real;
use crate::{Individual, Mutation};
use rand::{thread_rng, Rng};
use rand_distr::Normal;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

/// # Gaussian Mutation
///
/// For each gene in each individual in the population it has `mutation_rate` probability of
/// mutating the gene. The mutation is done by adding a random number from a Normal (Gaussian)
/// distribution with mean 0 and standard deviation `sigma`.
#[derive(Clone)]
pub struct GaussianMutation {
    pub mutation_rate: f64,
    pub sigma: f64,
}

impl Default for GaussianMutation {
    fn default() -> Self {
        Self {
            mutation_rate: 0.05,
            sigma: 1.0,
        }
    }
}

impl Mutation<Real> for GaussianMutation {
    fn mutate(&self, population: &mut Vec<Real>) {
        let gaussian_distribution = Normal::new(0.0, self.sigma).unwrap();

        population.par_iter_mut().for_each_init(
            || thread_rng(),
            |rng, individual| {
                for j in 0..individual.get_chromosome().len() {
                    if rng.gen_bool(self.mutation_rate) {
                        individual.set_gene(
                            j,
                            individual.get_gene(j) + rng.sample(gaussian_distribution),
                        );
                    }
                }
            },
        );
    }
}

use rand::distr::{Bernoulli, Distribution};
use rand::Rng;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::{population::Real, Mutation};
use crate::Rngs;

#[derive(Clone)]
pub struct DeltaMutation {
    pub mutation_rate: f64,
    pub frac: f64,
}

impl Default for DeltaMutation {
    fn default() -> Self {
        DeltaMutation {
            mutation_rate: 0.05,
            frac: 10.0,
        }
    }
}

impl Mutation<Real> for DeltaMutation {
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
                        let delta =
                            rng.random_range(individual.range.0..individual.range.1) / self.frac;

                        let result = if rng.random_bool(0.5) {
                            individual.chromosome[j] - delta
                        } else {
                            individual.chromosome[j] + delta
                        };

                        if result < individual.range.0 {
                            individual.chromosome[j] = individual.range.0;
                        } else if result > individual.range.1 {
                            individual.chromosome[j] = individual.range.1;
                        } else {
                            individual.chromosome[j] = result;
                        }
                    }
                }
            });
    }
}

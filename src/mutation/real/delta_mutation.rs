use rand::{thread_rng, Rng};
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};

use crate::{population::Real, Mutation};

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
    fn mutate(&self, population: &mut Vec<Real>) {
        population
            .par_iter_mut()
            .for_each_init(|| thread_rng(), |rng, individual| {
                for j in 0..individual.chromosome.len() {
                    if rng.gen_bool(self.mutation_rate) {
                        let delta = rng.gen_range(individual.range.0..individual.range.1) / self.frac;
                        
                        let result = if rng.gen_bool(0.5) {
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

use rand::{distributions::Bernoulli, prelude::Distribution, thread_rng, Rng};
use rayon::{prelude::ParallelIterator, slice::ParallelSliceMut};

use crate::population::Bin;

use super::Crossover;

#[derive(Clone)]
pub struct UniformCrossover {
    pub crossover_rate: f64,
}

impl Default for UniformCrossover {
    fn default() -> Self {
        UniformCrossover {
            crossover_rate: 0.5,
        }
    }
}

impl Crossover<Bin> for UniformCrossover {
    fn crossover(&self, population: &mut Vec<Bin>) {
        let distribution = Bernoulli::new(0.5).unwrap();

        population.par_chunks_mut(2).for_each_init(
            || thread_rng(),
            |mut rng, chunk| {
                if rng.gen_bool(self.crossover_rate) {
                    let mut parent1 = chunk[0].clone();
                    let mut parent2 = chunk[1].clone();

                    let len = parent1.chromosome.len();

                    for i in 0..len {
                        let toss = distribution.sample(&mut rng);

                        if toss {
                            let temp = parent1.chromosome[i];
                            parent1.chromosome[i] = parent2.chromosome[i];
                            parent2.chromosome[i] = temp;
                        }
                    }

                    chunk[0] = parent1;
                    chunk[1] = parent2;
                }
            },
        )
    }
}

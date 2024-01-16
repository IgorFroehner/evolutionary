use rand::{Rng, thread_rng, distributions::Bernoulli, prelude::Distribution};
use rayon::iter::ParallelIterator;
use rayon::prelude::ParallelSliceMut;
use crate::{Crossover, Individual};
use crate::crossover::UniformCrossover;

use crate::population::Bin;

impl Crossover<Bin> for UniformCrossover {
    fn crossover(&self, population: &mut Vec<Bin>) {
        let distribution = Bernoulli::new(self.toss_probability).unwrap();

        population.par_chunks_mut(2).for_each_init(
            || thread_rng(),
            |mut rng, chunk| {
                if rng.gen_bool(self.crossover_rate) {
                    let mut parent1 = chunk[0].clone();
                    let mut parent2 = chunk[1].clone();

                    let len = parent1.get_chromosome().len();

                    for i in 0..len {
                        if distribution.sample(&mut rng) {
                            let temp = parent1.get_chromosome()[i];
                            parent1.set_gene(i, parent2.get_chromosome()[i]);
                            parent2.set_gene(i, temp);
                        }
                    }

                    chunk[0] = parent1;
                    chunk[1] = parent2;
                }
            },
        )
    }
}

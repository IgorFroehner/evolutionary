use rand::{distr::Bernoulli, distr::Distribution, Rng};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::ParallelSliceMut;
use crate::{Crossover, Individual, Rngs};
use crate::crossover::UniformCrossover;

use crate::population::Int;

impl Crossover<Int> for UniformCrossover {
    fn crossover(&self, population: &mut [Int], rngs: &Rngs) {
        let distribution = Bernoulli::new(self.toss_probability).unwrap();

        population
            .par_chunks_exact_mut(2)
            .enumerate()
            .for_each(|(pair, chunk)| {
                let mut rng = rngs.rng(pair);
                if rng.random_bool(self.crossover_rate) {
                    let mut parent1 = chunk[0].clone();
                    let mut parent2 = chunk[1].clone();

                    let len = parent1.get_chromosome().len();

                    for i in 0..len {
                        if distribution.sample(&mut rng) {
                            let temp = parent1.get_gene(i);
                            parent1.set_gene(i, parent2.get_gene(i));
                            parent2.set_gene(i, temp);
                        }
                    }

                    chunk[0] = parent1;
                    chunk[1] = parent2;
                }
            });
    }
}

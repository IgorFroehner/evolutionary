use rand::{distributions::Bernoulli, prelude::Distribution, thread_rng, Rng};
use rayon::{prelude::ParallelIterator, slice::ParallelSliceMut};

use crate::Individual;

use super::Crossover;

/// # Uniform Crossover
/// 
/// For each gene, it selects whether to swap them between the parents based on the `toss_probability`. 
#[derive(Clone)]
pub struct UniformCrossover {
    /// The probability of crossover occurring.
    pub crossover_rate: f64,
    /// The probability of swaping the genes.
    pub toss_probability: f64
}

impl Default for UniformCrossover {
    fn default() -> Self {
        UniformCrossover {
            crossover_rate: 0.8,
            toss_probability: 0.5,
        }
    }
}

impl<T: Individual> Crossover<T> for UniformCrossover {
    fn crossover(&self, population: &mut Vec<T>) {
        let distribution = Bernoulli::new(self.toss_probability).unwrap();

        population.par_chunks_mut(2).for_each_init(
            || thread_rng(),
            |mut rng, chunk| {
                if rng.gen_bool(self.crossover_rate) {
                    let mut parent1 = chunk[0].clone();
                    let mut parent2 = chunk[1].clone();

                    let len = parent1.get_chromossome().len();

                    for i in 0..len {
                        if distribution.sample(&mut rng) {
                            let temp = parent1.get_chromossome()[i];
                            parent1.set_gene(i, parent2.get_chromossome()[i]);
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

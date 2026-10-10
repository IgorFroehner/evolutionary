use rand::distr::{Bernoulli, Distribution};
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use crate::{Individual, Mutation, Rngs};

use crate::population::Bin;

/// # Bit Swap Mutation
///
/// For each gene in the binary representation it has `mutation_rate` probability of swapping the bit (negating it).
///
/// Example:
/// ```rust
/// use evolutionary::prelude::*;
/// let mut population = vec![Bin::new(vec![true, false, true, false, true, false, true, false])];
///
/// let mut mutation = BitFlipMutation {
///    mutation_rate: 1.0,
/// };
///
/// mutation.mutate(&mut population, &Rngs::new(0));
///
/// assert_eq!(*population[0].get_chromosome(), vec![false, true, false, true, false, true, false, true]);
/// ```
#[derive(Clone)]
pub struct BitFlipMutation {
    pub mutation_rate: f64,
}

impl Default for BitFlipMutation {
    fn default() -> Self {
        BitFlipMutation {
            mutation_rate: 0.05,
        }
    }
}

impl Mutation<Bin> for BitFlipMutation {
    fn mutate(&self, population: &mut [Bin], rngs: &Rngs) {
        // Built once: `Rng::random_bool` builds the same distribution again at every call.
        let mutates = Bernoulli::new(self.mutation_rate).unwrap();

        population
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, member)| {
                let mut rng = rngs.rng(index);
                for gene in member.get_mut_chromosome().iter_mut() {
                    if mutates.sample(&mut rng) {
                        *gene = !*gene;
                    }
                }
            });
    }
}

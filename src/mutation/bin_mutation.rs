use rand::{thread_rng, Rng};
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};

use crate::population::Bin;

use super::Mutation;

/// # Bit Swap Mutation
/// 
/// For each gene in the binary representation it has `mutation_rate` probabity of swaping the bit (negating it).
/// 
/// Example:
/// ```rust
/// use evolutionary::prelude::*;
/// let mut population = vec![Bin::new(vec![true, false, true, false, true, false, true, false])];
/// 
/// let mut mutation = BitSwapMutation {
///    mutation_rate: 1.0,
/// };
/// 
/// mutation.mutate(&mut population);
/// 
/// assert_eq!(population[0].chromosome, vec![false, true, false, true, false, true, false, true]);
/// ```
#[derive(Clone)]
pub struct BitSwapMutation {
    pub mutation_rate: f64,
}

impl Default for BitSwapMutation {
    fn default() -> Self {
        BitSwapMutation {
            mutation_rate: 0.05,
        }
    }
}

impl Mutation<Bin> for BitSwapMutation {
    fn mutate(&self, population: &mut Vec<Bin>) {
        population.par_iter_mut().for_each_init(
            || thread_rng(),
            |rng, member| {
                for i in 0..member.chromosome.len() {
                    if rng.gen_bool(self.mutation_rate) {
                        member.chromosome[i] = !member.chromosome[i];
                    }
                }
            },
        );
    }
}

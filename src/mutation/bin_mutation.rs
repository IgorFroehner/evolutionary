use rand::{thread_rng, Rng};
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};

use crate::population::Bin;

use super::Mutation;

#[derive(Clone)]
pub struct BinMutation {
    pub mutation_rate: f64,
}

impl Default for BinMutation {
    fn default() -> Self {
        BinMutation {
            mutation_rate: 0.05,
        }
    }
}

impl Mutation<Bin> for BinMutation {
    fn mutate(&self, population: &mut Vec<Bin>) {
        population.par_iter_mut().for_each_init(|| thread_rng(), |rng, member| {
            for i in 0..member.0.len() {
                if rng.gen_bool(self.mutation_rate) {
                    member.0[i] = !member.0[i];
                }
            }
        });
    }
}

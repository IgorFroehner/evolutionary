use rand::{thread_rng, Rng};

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
        let mut rng = thread_rng();

        for i in 0..population.len() {
            let mut member = population[i].clone();

            for j in 0..member.0.len() {
                if rng.gen_bool(self.mutation_rate) {
                    member.0[j] = !member.0[j];
                }
            }

            population[i] = member;
        }
    }
}

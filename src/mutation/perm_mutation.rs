use rand::{thread_rng, Rng};

use crate::population::IntPerm;

use super::Mutation;

#[derive(Clone)]
pub struct PermMutation {
    pub mutation_rate: f64,
}

impl Default for PermMutation {
    fn default() -> Self {
        PermMutation {
            mutation_rate: 0.05,
        }
    }
}

impl Mutation<IntPerm> for PermMutation {
    fn mutate(&self, population: &mut Vec<IntPerm>) {
        let mut rng = thread_rng();

        for i in 0..population.len() {
            let mut member = population[i].clone();

            for j in 0..member.0.len() {
                if rng.gen_bool(self.mutation_rate) {
                    let swap_with = rng.gen_range(0..member.0.len());

                    let temp = member.0[j];
                    member.0[j] = member.0[swap_with];
                    member.0[swap_with] = temp;
                }
            }

            population[i] = member;
        }
    }
}

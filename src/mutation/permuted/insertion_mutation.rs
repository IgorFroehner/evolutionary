use rand::{Rng, thread_rng};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use crate::{Individual, Mutation};
use crate::population::IntPerm;

/// # Insertion Mutation
///
/// For each individual in the population it has `mutation_rate` probability of removing a random
/// gene from one place and inserting it back in other random position. If the gene is inserted in
/// the same position, nothing happens.
#[derive(Clone, Debug)]
pub struct InsertionMutation {
    pub mutation_rate: f64,
}

impl Default for InsertionMutation {
    fn default() -> Self {
        Self {
            mutation_rate: 0.05,
        }
    }
}

impl InsertionMutation {
    fn insert(chromosome: &mut Vec<i64>, from: usize, to: usize) {
        let gene = chromosome.remove(from);
        chromosome.insert(to, gene);
    }
}

impl Mutation<IntPerm> for InsertionMutation {
    fn mutate(&self, population: &mut Vec<IntPerm>) {
        population.par_iter_mut().for_each_init(
            || thread_rng(),
            |rng, individual| {
                if rng.gen_bool(self.mutation_rate) {
                    let from = rng.gen_range(0..individual.chromosome.len());
                    let to = rng.gen_range(0..individual.chromosome.len());

                    if from != to {
                        Self::insert(&mut individual.get_mut_chromosome(), from, to);
                    }
                }
        });
    }
}

#[cfg(test)]
mod tests {
    use crate::mutation::InsertionMutation;

    #[test]
    fn test_insertion() {
        let mut chromosome = vec![1, 2, 3, 4, 5, 6, 7, 9];

        InsertionMutation::insert(&mut chromosome, 0, 3);

        assert_eq!(chromosome, vec![2, 3, 4, 1, 5, 6, 7, 9]);
    }
}

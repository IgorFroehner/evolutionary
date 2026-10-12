use crate::population::IntPerm;
use crate::{Individual, Mutation, Rngs};
use rand::Rng;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

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
    /// Moves the gene at `from` to `to`, shifting the genes in between by one: the same as
    /// removing it and inserting it back, without changing the length.
    fn insert(chromosome: &mut [i64], from: usize, to: usize) {
        if from < to {
            chromosome[from..=to].rotate_left(1);
        } else {
            chromosome[to..=from].rotate_right(1);
        }
    }
}

impl Mutation<IntPerm> for InsertionMutation {
    fn mutate(&self, population: &mut [IntPerm], rngs: &Rngs) {
        population
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, individual)| {
                let mut rng = rngs.rng(index);
                if rng.random_bool(self.mutation_rate) {
                    let from = rng.random_range(0..individual.chromosome.len());
                    let to = rng.random_range(0..individual.chromosome.len());

                    if from != to {
                        Self::insert(individual.get_mut_chromosome(), from, to);
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

    #[test]
    fn insert_is_remove_then_insert_both_ways() {
        let original: Vec<i64> = (1..=6).collect();

        for from in 0..original.len() {
            for to in 0..original.len() {
                let mut expected = original.clone();
                let gene = expected.remove(from);
                expected.insert(to, gene);

                let mut chromosome = original.clone();
                InsertionMutation::insert(&mut chromosome, from, to);

                assert_eq!(chromosome, expected, "from {from} to {to}");
            }
        }
    }
}

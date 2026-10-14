use rand::Rng;
use rayon::{
    prelude::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

use crate::{population::IntPerm, Individual};

use super::{assert_permutations, Crossover};
use crate::Rngs;

/// # Ordered Crossover (OX)
///
/// The ordered crossover operator (OX) is a crossover operator for permutation-based chromosomes.
/// It maintains the relative order of the elements in the parent chromosomes. With the `crossover_rate`
/// probability it chooses two points in the parents, maintain the elements between the points and
/// fill the remaining positions with the elements of the other parent in the order they appear.
///
/// # Panics
///
/// `crossover` panics if `crossover_rate` is not within `[0, 1]`, or if two parents it crosses
/// are not permutations of `1..=n`.
#[derive(Clone)]
pub struct OrderedCrossover {
    pub crossover_rate: f64,
}

impl Default for OrderedCrossover {
    fn default() -> Self {
        OrderedCrossover {
            crossover_rate: 0.8,
        }
    }
}

impl OrderedCrossover {
    /// Crosses the parents in place, each keeping its genes within the section `start..=end`,
    /// with `start <= end + 1` and `start <= n`. An `end` at or past `n - 1` keeps every gene
    /// from `start` on.
    ///
    /// # Panics
    ///
    /// If the parents are not permutations of `1..=n`, or if `start > end + 1` or `start > n`.
    pub fn apply_ox(parent1: &mut [i64], parent2: &mut [i64], start: usize, end: usize) {
        assert_permutations(parent1, parent2);

        let len = parent1.len();

        // The genes each parent has outside of the section, which it takes back from the other
        // parent in the order they appear there: missing[gene - 1].
        let mut missing1 = vec![false; len];
        let mut missing2 = vec![false; len];

        for i in (0..start).chain(end + 1..len) {
            missing1[(parent1[i] - 1) as usize] = true;
            missing2[(parent2[i] - 1) as usize] = true;
        }

        // Parent2 is refilled first, from parent1 while it is still unchanged, so the genes
        // parent1 takes from parent2 are read before that.
        let from_parent2: Vec<i64> = parent2
            .iter()
            .copied()
            .filter(|&gene| missing1[(gene - 1) as usize])
            .collect();

        OrderedCrossover::fill_outside(
            parent2,
            start,
            end,
            parent1
                .iter()
                .copied()
                .filter(|&gene| missing2[(gene - 1) as usize]),
        );
        OrderedCrossover::fill_outside(parent1, start, end, from_parent2.into_iter());
    }

    /// Writes `genes`, in order, to the positions of `child` outside of `start..=end`.
    fn fill_outside(
        child: &mut [i64],
        start: usize,
        end: usize,
        mut genes: impl Iterator<Item = i64>,
    ) {
        let len = child.len();

        for i in (0..start).chain(end + 1..len) {
            child[i] = genes.next().unwrap();
        }
    }
}

impl Crossover<IntPerm> for OrderedCrossover {
    fn crossover(&self, population: &mut [IntPerm], rngs: &Rngs) {
        population
            .par_chunks_exact_mut(2)
            .enumerate()
            .for_each(|(pair, chunk)| {
                let mut rng = rngs.rng(pair);
                if rng.random_bool(self.crossover_rate) {
                    let (first, second) = chunk.split_at_mut(1);

                    let len = first[0].get_chromosome().len();

                    let start = rng.random_range(0..len);
                    let end = rng.random_range(start..len);

                    OrderedCrossover::apply_ox(
                        &mut first[0].chromosome,
                        &mut second[0].chromosome,
                        start,
                        end,
                    );
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use crate::crossover::OrderedCrossover;

    #[test]
    fn ox_crossover() {
        //                                     |        |
        let mut parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut parent2 = vec![5, 7, 4, 9, 1, 3, 6, 2, 8];

        OrderedCrossover::apply_ox(&mut parent1, &mut parent2, 2, 5);

        assert_eq!(parent1, vec![7, 9, 3, 4, 5, 6, 1, 2, 8]);
        assert_eq!(parent2, vec![2, 5, 4, 9, 1, 3, 6, 7, 8]);
    }

    #[test]
    #[should_panic(expected = "the chromosome must be a permutation of 1..=3, got [1, 1, 3]")]
    fn ox_rejects_a_repeated_gene() {
        // It used to give the second child a repeated gene too: [1, 2, 1].
        OrderedCrossover::apply_ox(&mut [1, 1, 3], &mut [1, 2, 3], 1, 1);
    }
}

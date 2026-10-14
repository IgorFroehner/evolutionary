use rand::Rng;
use rayon::{
    prelude::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

use crate::population::IntPerm;

use super::{assert_permutations, Crossover};
use crate::Rngs;

/// # Cycle Crossover (CX)
///
/// With the `crossover_rate` probability, finds the cycle of positions that starts at the first
/// gene of the parents, keeps the genes of that cycle in place and swaps all the others between
/// the parents, so both children stay permutations.
///
/// # Panics
///
/// `crossover` panics if `crossover_rate` is not within `[0, 1]`, or if two parents it crosses
/// are not permutations of `1..=n`.
#[derive(Clone)]
pub struct CycleCrossover {
    pub crossover_rate: f64,
}

impl Default for CycleCrossover {
    fn default() -> Self {
        CycleCrossover {
            crossover_rate: 0.8,
        }
    }
}

impl CycleCrossover {
    fn cx_crossover(parent1: &mut [i64], parent2: &mut [i64]) {
        assert_permutations(parent1, parent2);

        let len = parent1.len();
        // Nothing to cross, and no first position to start the cycle at.
        if len == 0 {
            return;
        }

        // The position of each gene in parent1: positions[gene - 1].
        let mut positions = vec![0; len];
        for (i, &gene) in parent1.iter().enumerate() {
            positions[(gene - 1) as usize] = i;
        }

        // Follows the cycle from the first position until it comes back to it.
        let mut in_cycle = vec![false; len];
        let mut pos = 0;
        while !in_cycle[pos] {
            in_cycle[pos] = true;

            pos = positions[(parent2[pos] - 1) as usize];
        }

        for i in 0..len {
            if !in_cycle[i] {
                std::mem::swap(&mut parent1[i], &mut parent2[i]);
            }
        }
    }
}

impl Crossover<IntPerm> for CycleCrossover {
    fn crossover(&self, population: &mut [IntPerm], rngs: &Rngs) {
        population
            .par_chunks_exact_mut(2)
            .enumerate()
            .for_each(|(pair, chunk)| {
                let mut rng = rngs.rng(pair);
                if rng.random_bool(self.crossover_rate) {
                    let (first, second) = chunk.split_at_mut(1);

                    CycleCrossover::cx_crossover(
                        &mut first[0].chromosome,
                        &mut second[0].chromosome,
                    );
                }
            });
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_cx_crossover1() {
        let mut parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut parent2 = vec![9, 3, 7, 8, 2, 6, 5, 1, 4];

        super::CycleCrossover::cx_crossover(&mut parent1, &mut parent2);

        assert_eq!(parent1, vec![1, 3, 7, 4, 2, 6, 5, 8, 9]);
        assert_eq!(parent2, vec![9, 2, 3, 8, 5, 6, 7, 1, 4]);
    }

    #[test]
    fn test_cx_crossover2() {
        let mut parent1 = vec![9, 8, 2, 1, 7, 4, 5, 10, 6, 3];
        let mut parent2 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        super::CycleCrossover::cx_crossover(&mut parent1, &mut parent2);

        assert_eq!(parent1, vec![9, 2, 3, 1, 5, 4, 7, 8, 6, 10]);
        assert_eq!(parent2, vec![1, 8, 2, 4, 7, 6, 5, 10, 9, 3]);
    }

    #[test]
    fn test_cx_crossover_full_cycle() {
        let mut parent1 = vec![1, 2, 3, 4, 5];
        let mut parent2 = vec![3, 5, 2, 1, 4];

        super::CycleCrossover::cx_crossover(&mut parent1, &mut parent2);

        assert_eq!(parent1, vec![1, 2, 3, 4, 5]);
        assert_eq!(parent2, vec![3, 5, 2, 1, 4]);
    }

    #[test]
    #[should_panic(expected = "the chromosome must be a permutation of 1..=3, got [2, 2, 2]")]
    fn cx_rejects_a_repeated_gene() {
        // The cycle from the first position used to never come back to it.
        super::CycleCrossover::cx_crossover(&mut [1, 2, 3], &mut [2, 2, 2]);
    }
}

use std::collections::HashMap;

use rand::Rng;
use rayon::{
    prelude::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

use crate::population::IntPerm;

use super::Crossover;
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
        let mut aux_hash1 = HashMap::new();
        let mut aux_hash2 = HashMap::new();

        for i in 0..parent1.len() {
            aux_hash1.insert(parent1[i] - 1, i);
            aux_hash2.insert(parent2[i] - 1, i);
        }

        let mut cycle = vec![0];

        let mut pos = *aux_hash1.get(&(parent2[0] - 1)).unwrap();
        while pos != 0 {
            cycle.push(pos);

            pos = *aux_hash1.get(&(parent2[pos] - 1)).unwrap();
        }

        for i in 0..parent1.len() {
            if !cycle.contains(&i) {
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
                    let mut parent1 = chunk[0].clone();
                    let mut parent2 = chunk[1].clone();

                    CycleCrossover::cx_crossover(&mut parent1.chromosome, &mut parent2.chromosome);

                    chunk[0] = parent1;
                    chunk[1] = parent2;
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
}

use std::collections::HashSet;

use rand::{thread_rng, Rng};
use rayon::{prelude::ParallelIterator, slice::ParallelSliceMut};

use crate::population::IntPerm;

use super::Crossover;

#[derive(Clone)]
pub struct OXCrossover {
    pub crossover_rate: f64,
}

impl Default for OXCrossover {
    fn default() -> Self {
        OXCrossover {
            crossover_rate: 0.8,
        }
    }
}

impl OXCrossover {
    pub fn apply_ox(parent1: &mut Vec<i64>, parent2: &mut Vec<i64>, start: usize, end: usize) {
        let len = parent1.len();

        let mut child1 = vec![-1; len];
        let mut child2 = vec![-1; len];

        let mut set = HashSet::new();
        for i in start..=end {
            child1[i] = parent1[i];
            child2[i] = parent2[i];
            set.insert(parent1[i]);
        }

        let mut j = (end + 1) % len;
        let mut k = j;
        while j != start {
            if !set.contains(&parent2[j]) {
                child1[k] = parent2[j];
                k = (k + 1) % len;
            }
            j = (j + 1) % len;
        }

        let mut set = HashSet::new();
        for i in start..=end {
            set.insert(parent2[i]);
        }

        let mut j = (end + 1) % len;
        let mut k = j;
        while j != start {
            if !set.contains(&parent1[j]) {
                child2[k] = parent1[j];
                k = (k + 1) % len;
            }
            j = (j + 1) % len;
        }

        for i in 0..len {
            if child1[i] == -1 {
                child1[i] = parent2[i];
            }
            if child2[i] == -1 {
                child2[i] = parent1[i];
            }
        }

        parent1.clone_from_slice(&child1);
        parent2.clone_from_slice(&child2);
    }
}

impl Crossover<IntPerm> for OXCrossover {
    fn crossover(&self, population: &mut Vec<IntPerm>) {
        population.par_chunks_mut(2).for_each_init(
            || thread_rng(),
            |rng, chunk| {
                if rng.gen_bool(self.crossover_rate) {
                    let mut parent1 = chunk[0].clone();
                    let mut parent2 = chunk[1].clone();

                    let len = parent1.chromosome.len();

                    let start = rng.gen_range(0..len);
                    let end = rng.gen_range(start..len);

                    OXCrossover::apply_ox(&mut parent1.chromosome, &mut parent2.chromosome, start, end);

                    chunk[0] = parent1;
                    chunk[1] = parent2;
                }
            },
        );
    }
}

#[cfg(test)]
mod tests {
    #[test]

    fn ox_crossover() {
        // let mut parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        // let mut parent2 = vec![5, 7, 4, 9, 1, 3, 6, 2, 8];

        // OXCrossover::apply_ox(&mut parent1, &mut parent2, 2, 6);

        // assert_eq!(parent1, vec![7, 9, 3, 4, 5, 6, 1, 2, 8]);
        // assert_eq!(parent2, vec![2, 7, 4, 9, 1, 3, 5, 7, 8]);
    }
}

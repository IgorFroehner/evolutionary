use std::collections::HashMap;

use rand::{thread_rng, Rng};
use rayon::{prelude::ParallelIterator, slice::ParallelSliceMut};

use crate::population::IntPerm;

use super::Crossover;

#[derive(Clone)]
pub struct CXCrossover {
    pub crossover_rate: f64,
}

impl Default for CXCrossover {
    fn default() -> Self {
        CXCrossover {
            crossover_rate: 0.8,
        }
    }
}

impl CXCrossover {
    fn cx_crossover(parent1: &mut Vec<i64>, parent2: &mut Vec<i64>) {
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
                let aux = parent1[i];
                parent1[i] = parent2[i];
                parent2[i] = aux;
            }
        }
    }
}

impl Crossover<IntPerm> for CXCrossover {
    fn crossover(&self, population: &mut Vec<IntPerm>) {
        population.par_chunks_mut(2).for_each_init(
            || thread_rng(),
            |rng, chunk| {
                if rng.gen_bool(self.crossover_rate) {
                    let mut parent1 = chunk[0].clone();
                    let mut parent2 = chunk[1].clone();

                    CXCrossover::cx_crossover(&mut parent1.chromosome, &mut parent2.chromosome);

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
    fn test_cx_crossover1() {
        let mut parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut parent2 = vec![9, 3, 7, 8, 2, 6, 5, 1, 4];

        super::CXCrossover::cx_crossover(&mut parent1, &mut parent2);

        assert_eq!(parent1, vec![1, 3, 7, 4, 2, 6, 5, 8, 9]);
        assert_eq!(parent2, vec![9, 2, 3, 8, 5, 6, 7, 1, 4]);
    }

    #[test]
    fn test_cx_crossover2() {
        let mut parent1 = vec![9, 8, 2, 1, 7, 4, 5, 10, 6, 3];
        let mut parent2 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        super::CXCrossover::cx_crossover(&mut parent1, &mut parent2);

        assert_eq!(parent1, vec![9, 2, 3, 1, 5, 4, 7, 8, 6, 10]);
        assert_eq!(parent2, vec![1, 8, 2, 4, 7, 6, 5, 10, 9, 3]);
    }
}

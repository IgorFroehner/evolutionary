use std::collections::HashMap;

use rand::{thread_rng, Rng};
use rayon::{prelude::ParallelIterator, slice::ParallelSliceMut};

use crate::{population::IntPerm, Individual};

use super::Crossover;

#[derive(Clone)]
pub struct PMXCrossover {
    pub crossover_rate: f64,
}

impl Default for PMXCrossover {
    fn default() -> Self {
        PMXCrossover {
            crossover_rate: 0.8,
        }
    }
}

impl PMXCrossover {
    fn pmx_matching(parent1: &mut Vec<i64>, parent2: &mut Vec<i64>, start: usize, end: usize) {
        let mut matching_section_1 = HashMap::new();
        let mut matching_section_2 = HashMap::new();

        for i in start..end {
            let aux = parent1[i];
            parent1[i] = parent2[i];
            parent2[i] = aux;

            matching_section_1.insert(parent2[i], parent1[i]);
            matching_section_2.insert(parent1[i], parent2[i]);
        }

        for i in 0..start {
            while matching_section_2.contains_key(&parent1[i]) {
                parent1[i] = *matching_section_2.get(&parent1[i]).unwrap();
            }

            while matching_section_1.contains_key(&parent2[i]) {
                parent2[i] = *matching_section_1.get(&parent2[i]).unwrap();
            }
        }

        for i in end..parent1.len() {
            while matching_section_2.contains_key(&parent1[i]) {
                parent1[i] = *matching_section_2.get(&parent1[i]).unwrap();
            }

            while matching_section_1.contains_key(&parent2[i]) {
                parent2[i] = *matching_section_1.get(&parent2[i]).unwrap();
            }
        }
    }
}

impl Crossover<IntPerm> for PMXCrossover {
    fn crossover(&self, population: &mut Vec<IntPerm>) {
        population.par_chunks_mut(2).for_each_init(
            || thread_rng(),
            |rng, chunk| {
                if rng.gen_bool(self.crossover_rate) {
                    let mut parent1 = chunk[0].clone();
                    let mut parent2 = chunk[1].clone();

                    let len = parent1.get_chromosome().len();

                    let start = rng.gen_range(0..len);
                    let end = rng.gen_range(start..len);

                    PMXCrossover::pmx_matching(
                        &mut parent1.chromosome,
                        &mut parent2.chromosome,
                        start,
                        end,
                    );

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
    fn pmx_matching_1() {
        let mut parent1 = vec![9, 8, 4, 5, 6, 7, 1, 3, 2, 10];
        let mut parent2 = vec![8, 7, 1, 2, 3, 10, 9, 5, 4, 6];

        super::PMXCrossover::pmx_matching(&mut parent1, &mut parent2, 3, 6);

        assert_eq!(parent1, vec![9, 8, 4, 2, 3, 10, 1, 6, 5, 7]);
        assert_eq!(parent2, vec![8, 10, 1, 5, 6, 7, 9, 2, 4, 3]);
    }

    #[test]
    fn pmx_matching_2() {
        let mut parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut parent2 = vec![9, 3, 7, 8, 2, 6, 5, 1, 4];

        super::PMXCrossover::pmx_matching(&mut parent1, &mut parent2, 2, 6);

        assert_eq!(parent1, vec![1, 5, 7, 8, 2, 6, 3, 4, 9]);
        assert_eq!(parent2, vec![9, 7, 3, 4, 5, 6, 2, 1, 8]);
    }

    #[test]
    fn pmx_matching_with_loop() {
        let mut parent1 = vec![4, 2, 3, 1, 5];
        let mut parent2 = vec![2, 3, 1, 4, 5];

        super::PMXCrossover::pmx_matching(&mut parent1, &mut parent2, 1, 4);

        assert_eq!(parent1, vec![2, 3, 1, 4, 5]);
        assert_eq!(parent2, vec![4, 2, 3, 1, 5]);
    }
}

use std::collections::{HashSet, VecDeque};

use rand::{thread_rng, Rng};
use rayon::{prelude::ParallelIterator, slice::ParallelSliceMut};

use crate::{population::IntPerm, Individual};

use super::Crossover;

/// # Ordered Crossover (OX)
///
/// The ordered crossover operator (OX) is a crossover operator for permutation-based chromosomes.
/// It maintains the relative order of the elements in the parent chromosomes. With the `crossover_rate`
/// probability it chooses two points in the parents, maintain the elements between the points and
/// fill the remaining positions with the elements of the other parent in the order they appear.
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
    pub fn apply_ox(parent1: &mut Vec<i64>, parent2: &mut Vec<i64>, start: usize, end: usize) {
        let len = parent1.len();

        let mut missing1 = HashSet::new();
        let mut missing2 = HashSet::new();

        let mut queue1 = VecDeque::new();
        let mut queue2 = VecDeque::new();

        for i in 0..start {
            missing1.insert(parent1[i]);
            missing2.insert(parent2[i]);
        }
        for i in end + 1..len {
            missing1.insert(parent1[i]);
            missing2.insert(parent2[i]);
        }

        for i in 0..len {
            if missing1.contains(&parent2[i]) {
                queue1.push_back(parent2[i]);
            }
            if missing2.contains(&parent1[i]) {
                queue2.push_back(parent1[i]);
            }
        }

        for i in 0..start {
            parent1[i] = queue1.pop_front().unwrap();
            parent2[i] = queue2.pop_front().unwrap();
        }
        for i in end + 1..len {
            parent1[i] = queue1.pop_front().unwrap();
            parent2[i] = queue2.pop_front().unwrap();
        }
    }
}

impl Crossover<IntPerm> for OrderedCrossover {
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

                    OrderedCrossover::apply_ox(
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
}

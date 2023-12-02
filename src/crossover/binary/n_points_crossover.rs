use rand::{seq::IteratorRandom, thread_rng, Rng};
use rayon::{prelude::ParallelIterator, slice::ParallelSliceMut};
use crate::Individual;

use crate::population::Bin;

use super::Crossover;

/// # N Points Crossover
///
/// Generates `n` random points to "crossover" the sections between the parents. This can be
/// a `Single-Point Crossover` using `n_points = 1` a `Two-Points Crossover` using `n_points = 2`
/// or any number you'd like.
#[derive(Clone)]
pub struct NPointsCrossover {
    /// The probability of crossover occurring.
    pub crossover_rate: f64,
    /// The number of points to crossover.
    pub n_points: usize,
}

impl Default for NPointsCrossover {
    fn default() -> Self {
        NPointsCrossover {
            crossover_rate: 0.8,
            n_points: 1,
        }
    }
}

impl NPointsCrossover {
    fn swap_sections(vec1: &mut Vec<bool>, vec2: &mut Vec<bool>, start: usize, end: usize) {
        for i in start..end {
            let temp = vec1[i];
            vec1[i] = vec2[i];
            vec2[i] = temp;
        }
    }
}

impl Crossover<Bin> for NPointsCrossover {
    fn crossover(&self, population: &mut Vec<Bin>) {
        population.par_chunks_mut(2).for_each_init(
            || thread_rng(),
            |mut rng, chunk| {
                if rng.gen_bool(self.crossover_rate) {
                    let mut parent1 = chunk[0].clone();
                    let mut parent2 = chunk[1].clone();

                    let len = parent1.get_chromosome().len();

                    let mut points: Vec<usize> = (0..len).choose_multiple(&mut rng, self.n_points);

                    points.sort();

                    let mut start = 0;

                    for point in points {
                        NPointsCrossover::swap_sections(
                            &mut parent1.get_mut_chromosome(),
                            &mut parent2.get_mut_chromosome(),
                            start,
                            point,
                        );

                        start = point;
                    }

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
    fn swap_sections() {
        let mut vec1 = vec![false, false, false, false, false, false];
        let mut vec2 = vec![true, true, true, true, true, true];

        super::NPointsCrossover::swap_sections(&mut vec1, &mut vec2, 2, 4);

        assert_eq!(vec1, vec![false, false, true, true, false, false]);
        assert_eq!(vec2, vec![true, true, false, false, true, true]);
    }
}

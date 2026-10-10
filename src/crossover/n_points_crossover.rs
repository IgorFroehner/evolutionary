use rand::prelude::IteratorRandom;
use rand::Rng;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::ParallelSliceMut;

use crate::population::{Bin, Int, Real};
use crate::{Crossover, Individual, Rngs};

/// # N Points Crossover
///
/// Generates `n` random points to "crossover" the sections between the parents. This can be
/// a `Single-Point Crossover` using `n_points = 1` a `Two-Points Crossover` using `n_points = 2`
/// or any number you'd like.
///
/// It works for [`Bin`], [`Int`] and [`Real`], but not for permuted individuals, as swapping
/// sections between two permutations rarely yields permutations.
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
    pub fn swap_sections<T: Copy>(vec1: &mut [T], vec2: &mut [T], start: usize, end: usize) {
        for i in start..end {
            std::mem::swap(&mut vec1[i], &mut vec2[i]);
        }
    }

    /// The crossover is the same for every individual whose genes can be swapped section by
    /// section, only the gene type changes.
    fn crossover_population<T: Individual>(&self, population: &mut [T], rngs: &Rngs) {
        population
            .par_chunks_exact_mut(2)
            .enumerate()
            .for_each(|(pair, chunk)| {
                let mut rng = rngs.rng(pair);
                if rng.random_bool(self.crossover_rate) {
                    let (first, second) = chunk.split_at_mut(1);
                    let parent1 = first[0].get_mut_chromosome();
                    let parent2 = second[0].get_mut_chromosome();

                    let len = parent1.len();

                    let mut points: Vec<usize> = (0..len).choose_multiple(&mut rng, self.n_points);
                    points.sort_unstable();

                    let mut start = 0;

                    for point in points {
                        Self::swap_sections(parent1, parent2, start, point);
                        start = point;
                    }
                }
            });
    }
}

impl Crossover<Bin> for NPointsCrossover {
    fn crossover(&self, population: &mut [Bin], rngs: &Rngs) {
        self.crossover_population(population, rngs);
    }
}

impl Crossover<Int> for NPointsCrossover {
    fn crossover(&self, population: &mut [Int], rngs: &Rngs) {
        self.crossover_population(population, rngs);
    }
}

impl Crossover<Real> for NPointsCrossover {
    fn crossover(&self, population: &mut [Real], rngs: &Rngs) {
        self.crossover_population(population, rngs);
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

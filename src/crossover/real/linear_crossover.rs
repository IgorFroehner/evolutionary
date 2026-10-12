use rand::seq::IndexedRandom;
use rand::Rng;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::ParallelSliceMut;

use crate::population::Real;
use crate::{Crossover, Rngs};

/// # Linear Crossover
///
/// Wright's linear crossover: from two parents it builds the three linear combinations
///
/// ```text
/// 0.5 * p1 + 0.5 * p2
/// 1.5 * p1 - 0.5 * p2
/// -0.5 * p1 + 1.5 * p2
/// ```
///
/// and replaces the parents with two distinct candidates drawn at random. (The classical
/// operator evaluates the three candidates and keeps the best two; a crossover has no access
/// to the fitness function, and the next selection favours the better children anyway.)
///
/// Reference: Wright, A. H. (1991). Genetic algorithms for real parameter optimization.
/// Foundations of Genetic Algorithms, 1, 205-218.
///
/// # Panics
///
/// `crossover` panics if `crossover_rate` is not within `[0, 1]`.
#[derive(Clone, Debug)]
pub struct LinearCrossover {
    /// The probability of crossover occurring.
    pub crossover_rate: f64,
}

impl LinearCrossover {
    pub fn new(crossover_rate: f64) -> Self {
        Self { crossover_rate }
    }
}

impl Default for LinearCrossover {
    fn default() -> Self {
        Self::new(0.8)
    }
}

/// The weights of the three candidates: `weight.0 * p1 + weight.1 * p2`.
const CANDIDATES: [(f64, f64); 3] = [(0.5, 0.5), (1.5, -0.5), (-0.5, 1.5)];

impl Crossover<Real> for LinearCrossover {
    fn crossover(&self, population: &mut [Real], rngs: &Rngs) {
        population
            .par_chunks_exact_mut(2)
            .enumerate()
            .for_each(|(pair, chunk)| {
                let mut rng = rngs.rng(pair);
                if rng.random_bool(self.crossover_rate) {
                    let mut picks = CANDIDATES.choose_multiple(&mut rng, 2);
                    let (a1, b1) = *picks.next().unwrap();
                    let (a2, b2) = *picks.next().unwrap();

                    let (first, second) = chunk.split_at_mut(1);

                    for (gene1, gene2) in first[0]
                        .chromosome
                        .iter_mut()
                        .zip(second[0].chromosome.iter_mut())
                    {
                        let (p1, p2) = (*gene1, *gene2);
                        *gene1 = a1 * p1 + b1 * p2;
                        *gene2 = a2 * p1 + b2 * p2;
                    }
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Individual;

    #[test]
    fn children_are_two_distinct_candidates() {
        for seed in 0..20 {
            let mut population = vec![
                Real::new(vec![1.0, 3.0], (0.0, 4.0)),
                Real::new(vec![2.0, 1.0], (0.0, 4.0)),
            ];
            LinearCrossover::new(1.0).crossover(&mut population, &Rngs::new(seed));

            // The candidates of (1.0, 2.0) and of (3.0, 1.0) per gene.
            let expected: [&[f64]; 3] = [
                &[1.5, 2.0], // the mean
                &[0.5, 4.0], // 1.5 * p1 - 0.5 * p2
                &[2.5, 0.0], // -0.5 * p1 + 1.5 * p2
            ];

            let child1 = population[0].get_chromosome();
            let child2 = population[1].get_chromosome();
            assert!(expected.contains(&child1), "{child1:?}");
            assert!(expected.contains(&child2), "{child2:?}");
            assert_ne!(child1, child2);
        }
    }
}

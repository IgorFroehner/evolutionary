use crate::population::Real;
use crate::{Crossover, Rngs};
use rand::Rng;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::ParallelSliceMut;

/// # Arithmetic Crossover
///
/// The arithmetic crossover creates the offsprings by computing a weighted average of two parent
/// chromosomes. It does so based on the alpha value, where the offspring will be
/// O1 = [p11 * alpha + p21 * (1 - alpha), p12 * alpha + p22 * (1 - alpha), ...], and the
/// O2 = [p21 * alpha + p11 * (1 - alpha), p22 * alpha + p12 * (1 - alpha), ...].
///
/// # Panics
///
/// `crossover` panics if `crossover_rate` is not within `[0, 1]`.
#[derive(Clone)]
pub struct ArithmeticCrossover {
    pub crossover_rate: f64,
    pub alpha: f64,
}

impl Default for ArithmeticCrossover {
    fn default() -> Self {
        Self {
            crossover_rate: 0.8,
            alpha: 0.5,
        }
    }
}

impl Crossover<Real> for ArithmeticCrossover {
    fn crossover(&self, population: &mut [Real], rngs: &Rngs) {
        population
            .par_chunks_exact_mut(2)
            .enumerate()
            .for_each(|(pair, chunk)| {
                let mut rng = rngs.rng(pair);
                if rng.random_bool(self.crossover_rate) {
                    let (first, second) = chunk.split_at_mut(1);

                    for (gene1, gene2) in first[0]
                        .chromosome
                        .iter_mut()
                        .zip(second[0].chromosome.iter_mut())
                    {
                        let (p1, p2) = (*gene1, *gene2);
                        *gene1 = self.alpha * p1 + (1.0 - self.alpha) * p2;
                        *gene2 = self.alpha * p2 + (1.0 - self.alpha) * p1;
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
    fn offsprings_are_the_weighted_averages_of_the_parents() {
        let mut population = vec![
            Real::new(vec![1.0, 2.0, 3.0, 4.0], (0.0, 10.0)),
            Real::new(vec![5.0, 6.0, 7.0, 8.0], (0.0, 10.0)),
        ];

        let crossover = ArithmeticCrossover {
            crossover_rate: 1.0,
            alpha: 0.25,
        };
        crossover.crossover(&mut population, &Rngs::new(0));

        assert_eq!(*population[0].get_chromosome(), vec![4.0, 5.0, 6.0, 7.0]);
        assert_eq!(*population[1].get_chromosome(), vec![2.0, 3.0, 4.0, 5.0]);
    }
}

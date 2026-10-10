use rand::Rng;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::ParallelSliceMut;

use crate::population::Real;
use crate::{Crossover, Rngs};

/// # Simulated Binary Crossover (SBX)
///
/// Mimics the behaviour of the single-point binary crossover on real values: each pair of genes
/// is recombined around its mean with a spread factor `beta` drawn so that, on average, children
/// stay close to their parents. `eta` (the distribution index) controls how close: a bigger
/// `eta` concentrates the children near the parents, a smaller one spreads them out. Common
/// values are between 2 and 20.
///
/// For each gene, with `u` drawn uniformly from `[0, 1)`:
///
/// ```text
/// beta = (2u)^(1 / (eta + 1))            if u < 0.5
/// beta = (1 / (2(1 - u)))^(1 / (eta + 1)) otherwise
///
/// c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
/// c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
/// ```
///
/// The children are symmetric around the parents' mean: `c1 + c2 = p1 + p2`.
///
/// Reference: [Simulated Binary Crossover for Continuous Search Space
/// (Deb & Agrawal, 1995)](https://content.wolfram.com/sites/13/2018/02/09-2-2.pdf)
#[derive(Clone, Debug)]
pub struct SimulatedBinaryCrossover {
    /// The probability of crossover occurring.
    pub crossover_rate: f64,
    /// The distribution index: bigger keeps the children closer to the parents.
    pub eta: f64,
}

impl SimulatedBinaryCrossover {
    /// Creates an SBX crossover with the given rate and distribution index `eta`.
    ///
    /// # Panics
    ///
    /// If `eta` is negative.
    pub fn new(crossover_rate: f64, eta: f64) -> Self {
        assert!(eta >= 0.0, "the distribution index `eta` must be >= 0, got {eta}");

        Self {
            crossover_rate,
            eta,
        }
    }

    fn beta(&self, u: f64) -> f64 {
        let exponent = 1.0 / (self.eta + 1.0);
        if u < 0.5 {
            (2.0 * u).powf(exponent)
        } else {
            (1.0 / (2.0 * (1.0 - u))).powf(exponent)
        }
    }
}

impl Default for SimulatedBinaryCrossover {
    fn default() -> Self {
        Self::new(0.8, 2.0)
    }
}

impl Crossover<Real> for SimulatedBinaryCrossover {
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
                        let beta = self.beta(rng.random_range(0.0..1.0));
                        let (p1, p2) = (*gene1, *gene2);

                        *gene1 = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2);
                        *gene2 = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2);
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
    fn children_are_symmetric_around_the_parents_mean() {
        let parent1: Vec<f64> = (0..32).map(|i| i as f64 / 32.0).collect();
        let parent2: Vec<f64> = (0..32).map(|i| -(i as f64) / 16.0).collect();
        let sums: Vec<f64> = parent1.iter().zip(&parent2).map(|(a, b)| a + b).collect();

        let mut population = vec![
            Real::new(parent1, (-2.0, 1.0)),
            Real::new(parent2, (-2.0, 1.0)),
        ];
        SimulatedBinaryCrossover::new(1.0, 2.0).crossover(&mut population, &Rngs::new(3));

        let crossed: Vec<f64> = population[0]
            .get_chromosome()
            .iter()
            .zip(population[1].get_chromosome())
            .map(|(a, b)| a + b)
            .collect();

        for (before, after) in sums.iter().zip(&crossed) {
            assert!((before - after).abs() < 1e-12, "{before} != {after}");
        }
    }

    #[test]
    fn same_seed_same_children() {
        let make = || {
            vec![
                Real::new(vec![0.25; 16], (-1.0, 1.0)),
                Real::new(vec![-0.75; 16], (-1.0, 1.0)),
            ]
        };

        let mut a = make();
        let mut b = make();
        let crossover = SimulatedBinaryCrossover::new(1.0, 2.0);
        crossover.crossover(&mut a, &Rngs::new(5));
        crossover.crossover(&mut b, &Rngs::new(5));

        assert_eq!(a[0].get_chromosome(), b[0].get_chromosome());
        assert_eq!(a[1].get_chromosome(), b[1].get_chromosome());
        // And the children actually moved away from the parents.
        assert_ne!(*a[0].get_chromosome(), vec![0.25; 16]);
    }

    #[test]
    #[should_panic(expected = "must be >= 0")]
    fn negative_eta_panics() {
        SimulatedBinaryCrossover::new(0.8, -1.0);
    }
}

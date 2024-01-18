use crate::population::Real;
use crate::{Crossover, Individual};
use rand::{thread_rng, Rng};
use rayon::iter::ParallelIterator;
use rayon::prelude::ParallelSliceMut;

/// # Arithmetic Crossover
///
/// The arithmetic crossover creates the offsprings by computing a weighted average of two parent
/// chromosomes. It does so based on the alpha value, where the offspring will be
/// O1 = [p11 * alpha + p21 * (1 - alpha), p12 * alpha + p22 * (1 - alpha), ...], and the
/// O2 = [p21 * alpha + p11 * (1 - alpha), p22 * alpha + p12 * (1 - alpha), ...].
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

impl ArithmeticCrossover {
    fn get_offspring(&self, parent1: &Vec<f64>, parent2: &Vec<f64>) -> Vec<f64> {
        parent1
            .iter()
            .zip(parent2.iter())
            .map(|(p1, p2)| self.alpha * p1 + (1.0 - self.alpha) * p2)
            .collect()
    }
}

impl Crossover<Real> for ArithmeticCrossover {
    fn crossover(&self, population: &mut Vec<Real>) {
        population.par_chunks_mut(2).for_each_init(
            || thread_rng(),
            |rng, chunk| {
                if rng.gen_bool(self.crossover_rate) {
                    let parent1 = chunk[0].clone();
                    let parent2 = chunk[1].clone();

                    chunk[0].chromosome =
                        self.get_offspring(parent1.get_chromosome(), parent2.get_chromosome());
                    chunk[1].chromosome =
                        self.get_offspring(parent2.get_chromosome(), parent1.get_chromosome());
                }
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::ArithmeticCrossover;

    #[test]
    fn test_get_offspring() {

        let parent1 = vec![1.0, 2.0, 3.0, 4.0];
        let parent2 = vec![5.0, 6.0, 7.0, 8.0];

        let crossover = ArithmeticCrossover {
            crossover_rate: 1.0,
            alpha: 0.5,
        };

        let offspring = crossover.get_offspring(&parent1, &parent2);

        assert_eq!(offspring, vec![3.0, 4.0, 5.0, 6.0]);
    }
}


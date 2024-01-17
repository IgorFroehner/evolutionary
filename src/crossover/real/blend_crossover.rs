use rand::{Rng, thread_rng};
use rayon::iter::ParallelIterator;
use rayon::prelude::ParallelSliceMut;
use crate::{Crossover, Individual};
use crate::population::Real;

/// # Blend Crossover
///
/// This crossover method is used for real coded individuals. It takes two parents and creates two
/// children by mixing the genes of the parents. For each gene, x1 is the smaller value and x2 is
/// the bigger value and diff is `x1 - x2`. The children will have a value between `x1 - alpha * diff`
/// and `x2 + alpha * diff`.
///
/// Reference: [An empirical comparison of two crossover operators in real-coded genetic algorithms
/// for constrained numerical optimization problems](https://ieeexplore.ieee.org/document/7036347)
#[derive(Clone, Debug)]
pub struct BlendCrossover {
    pub crossover_rate: f64,
    pub alpha: f64,
}

impl Default for BlendCrossover {
    fn default() -> Self {
        Self {
            crossover_rate: 0.8,
            alpha: 0.5,
        }
    }
}

impl BlendCrossover {
    pub fn new(crossover_rate: f64, alpha: f64) -> Self {
        Self {
            crossover_rate,
            alpha,
        }
    }

    fn blend_min_max(&self, x1: f64, x2: f64) -> (f64, f64) {
        let (x1, x2) = (x1.min(x2), x1.max(x2));
        let diff = x2 - x1;
        let min = x1 - diff * self.alpha;
        let max = x2 + diff * self.alpha;
        (min, max)
    }
}

impl Crossover<Real> for BlendCrossover {
    fn crossover(&self, population: &mut Vec<Real>) {
        population.par_chunks_mut(2).for_each_init(
            || thread_rng(),
            |rng, chunk| {
                if rng.gen_bool(self.crossover_rate) {
                    let (parent1, parent2) = chunk.split_at_mut(1);
                    let parent1 = &mut parent1[0];
                    let parent2 = &mut parent2[0];

                    let len = parent1.get_chromosome().len();

                    for i in 0..len {
                        let (min, max) = self.blend_min_max(parent1.get_gene(i), parent2.get_gene(i));

                        let gene = rng.gen_range(min..=max);

                        parent1.set_gene(i, gene);
                        parent2.set_gene(i, gene);
                    }
                }
            }
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::crossover::BlendCrossover;

    #[test]
    fn test_blend_min_max() {
        let blend = BlendCrossover::new(0.8, 0.5);
        assert_eq!(blend.blend_min_max(1.0, 2.0), (0.5, 2.5));
    }
}


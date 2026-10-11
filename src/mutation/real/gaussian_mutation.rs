use crate::prelude::Real;
use crate::{Mutation, Rngs};
use rand::distr::{Bernoulli, Distribution};
use rand::Rng;
use rand_distr::Normal;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

/// # Gaussian Mutation
///
/// For each gene in each individual in the population it has `mutation_rate` probability of
/// mutating the gene. The mutation is done by adding a random number from a Normal (Gaussian)
/// distribution with mean 0 and standard deviation `sigma`.
///
/// The defaults are `mutation_rate = 0.05` and `sigma = 1.0`, without clamping.
///
/// ## Choosing `sigma` and `mutation_rate`
///
/// `sigma` is in the units of the genes, not relative to the `range` they were generated in. With
/// genes in `(-1.0, 1.0)`, as network weights often are, the default `sigma = 1.0` is half the
/// width of the range: it moves a gene almost anywhere, which explores but can't refine a good
/// solution. Lower it, to about 1% to 10% of the range width (`0.02` to `0.2` here), once the
/// population is near a good solution, or from the start when the genes are already close (for
/// example when seeding from a saved population).
///
/// `mutation_rate` is per gene, so each child gets about `mutation_rate * dimension` mutated
/// genes: 15 out of 300 with the default. For a few hundred genes, a rate around
/// `1.0 / dimension` (one gene per child on average) changes less at once and is a common
/// starting point.
///
/// ## Clamping
///
/// By default mutated genes are unbounded: the `range` only bounds the initial population, which
/// is what network weights need. Set `clamp_to_range` to keep every mutated gene within the
/// individual's `range`.
///
/// ```
/// use evolutionary::prelude::*;
///
/// let mutation = GaussianMutation {
///     mutation_rate: 1.0,
///     sigma: 10.0,
///     clamp_to_range: true,
/// };
/// let mut population = vec![Real::new(vec![0.0; 100], (-1.0, 1.0))];
///
/// mutation.mutate(&mut population, &Rngs::new(0));
///
/// assert!(population[0].get_chromosome().iter().all(|gene| (-1.0..=1.0).contains(gene)));
/// ```
#[derive(Clone, Debug)]
pub struct GaussianMutation {
    pub mutation_rate: f64,
    pub sigma: f64,
    /// Clamps each mutated gene to the `range` of its individual.
    pub clamp_to_range: bool,
}

impl GaussianMutation {
    /// An unclamped Gaussian mutation.
    pub fn new(mutation_rate: f64, sigma: f64) -> Self {
        Self {
            mutation_rate,
            sigma,
            clamp_to_range: false,
        }
    }
}

impl Default for GaussianMutation {
    fn default() -> Self {
        Self::new(0.05, 1.0)
    }
}

impl Mutation<Real> for GaussianMutation {
    fn mutate(&self, population: &mut [Real], rngs: &Rngs) {
        let gaussian_distribution = Normal::new(0.0, self.sigma).unwrap();
        // Built once: `Rng::random_bool` builds the same distribution again at every call.
        let mutates = Bernoulli::new(self.mutation_rate).unwrap();

        population
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, individual)| {
                let mut rng = rngs.rng(index);
                let (low, high) = individual.range;
                for gene in individual.chromosome.iter_mut() {
                    if mutates.sample(&mut rng) {
                        let mut mutated = *gene + rng.sample(gaussian_distribution);
                        if self.clamp_to_range {
                            mutated = mutated.clamp(low, high);
                        }
                        *gene = mutated;
                    }
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Individual;

    /// A few hundred individuals of a few hundred genes, the size of a small neural network.
    fn population(range: (f64, f64)) -> Vec<Real> {
        vec![Real::new(vec![0.0; 300], range); 300]
    }

    fn deltas(mutation: &GaussianMutation, range: (f64, f64)) -> Vec<f64> {
        let mut population = population(range);
        mutation.mutate(&mut population, &Rngs::new(11));

        population
            .iter()
            .flat_map(|individual| individual.get_chromosome().iter().copied())
            .collect()
    }

    #[test]
    fn mutates_the_expected_share_of_genes_with_the_expected_spread() {
        let mutation = GaussianMutation::new(0.05, 0.1);

        let genes = deltas(&mutation, (-1.0, 1.0));
        let mutated: Vec<f64> = genes.iter().copied().filter(|&gene| gene != 0.0).collect();

        // 90 000 genes at 5%: 4 500 expected, with a standard deviation of about 65.
        let expected = 0.05 * genes.len() as f64;
        assert!(
            (mutated.len() as f64 - expected).abs() < 4.0 * 65.0,
            "{}",
            mutated.len()
        );

        let mean = mutated.iter().sum::<f64>() / mutated.len() as f64;
        let deviation =
            (mutated.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / mutated.len() as f64).sqrt();
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((deviation - 0.1).abs() < 0.005, "deviation {deviation}");
    }

    #[test]
    fn is_unbounded_by_default() {
        let mutation = GaussianMutation {
            mutation_rate: 1.0,
            sigma: 10.0,
            ..GaussianMutation::default()
        };

        let genes = deltas(&mutation, (-1.0, 1.0));

        assert!(genes.iter().any(|gene| gene.abs() > 1.0));
    }

    #[test]
    fn clamps_to_the_range_of_each_individual() {
        let mutation = GaussianMutation {
            mutation_rate: 1.0,
            sigma: 10.0,
            clamp_to_range: true,
        };

        let genes = deltas(&mutation, (-0.5, 2.0));

        assert!(genes.iter().all(|gene| (-0.5..=2.0).contains(gene)));
        // Clamping keeps the bounds reachable rather than resampling.
        assert!(genes.contains(&-0.5) && genes.contains(&2.0));
    }
}

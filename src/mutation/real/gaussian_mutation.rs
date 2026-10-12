use crate::prelude::Real;
use crate::{Mutation, Rngs};
use rand::distr::{Bernoulli, Distribution};
use rand::Rng;
use rand_distr::Normal;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use std::fmt;
use std::sync::Arc;

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
/// solution. A smaller one, about 1% to 10% of the range width (`0.02` to `0.2` here), refines
/// a population that is already near a good solution, or one whose genes start close (for
/// example when seeding from a saved population).
///
/// `sigma` doesn't have to stay the same for the whole run: a [`Sigma`] is either a constant or
/// a schedule, a function of the generation. A schedule that starts big and shrinks explores
/// first and refines later, without stopping the run to change the operator.
/// [`Sigma::exponential_decay`] multiplies it by the same factor at every generation, and
/// [`Sigma::schedule`] takes any function:
///
/// ```
/// use evolutionary::prelude::*;
///
/// // 0.5 times 0.99 at the first generation, 1% less at each one after: about 0.18 at the
/// // hundredth.
/// let decaying = GaussianMutation::new(0.05, Sigma::exponential_decay(0.5, 0.99));
///
/// // Big steps for 200 generations, then small ones.
/// let two_phases = GaussianMutation::new(
///     0.05,
///     Sigma::schedule(|generation| if generation <= 200 { 0.5 } else { 0.05 }),
/// );
/// ```
///
/// A schedule changes only the spread of the numbers drawn, not how many are drawn or in which
/// order: at a generation where a schedule gives `0.1`, the mutation changes the genes exactly
/// as a constant `sigma` of `0.1` would.
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
///     sigma: 10.0.into(),
///     clamp_to_range: true,
/// };
/// let mut population = vec![Real::new(vec![0.0; 100], (-1.0, 1.0))];
///
/// mutation.mutate(&mut population, &Rngs::new(0));
///
/// assert!(population[0].get_chromosome().iter().all(|gene| (-1.0..=1.0).contains(gene)));
/// ```
///
/// # Panics
///
/// `mutate` panics if `mutation_rate` is not within `[0, 1]`, or if `sigma` (for a schedule, its
/// value at the generation of the `Rngs`) is not finite.
#[derive(Clone, Debug)]
pub struct GaussianMutation {
    pub mutation_rate: f64,
    /// The standard deviation of the steps, constant or scheduled. A number converts into a
    /// constant one: `sigma: 0.1.into()`.
    pub sigma: Sigma,
    /// Clamps each mutated gene to the `range` of its individual.
    pub clamp_to_range: bool,
}

impl GaussianMutation {
    /// An unclamped Gaussian mutation. `sigma` is a number, for a constant one, or a [`Sigma`].
    pub fn new(mutation_rate: f64, sigma: impl Into<Sigma>) -> Self {
        Self {
            mutation_rate,
            sigma: sigma.into(),
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
        let generation = rngs.generation();
        let sigma = self.sigma.at(generation);
        let gaussian_distribution = Normal::new(0.0, sigma).unwrap_or_else(|_| {
            panic!("GaussianMutation needs a finite sigma, got {sigma} at generation {generation}")
        });
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

/// The `sigma` of a [`GaussianMutation`]: a constant, or a function of the generation.
///
/// A number converts into a constant, so `GaussianMutation::new(0.05, 0.1)` keeps working, and
/// a struct literal writes `sigma: 0.1.into()`. The generation is the one of the [`Rngs`] the
/// mutation gets ([`Rngs::generation`]): 1 at the first generation of a run, the same number
/// as its [`GenerationReport`](crate::GenerationReport). [`Rngs::new`] is at generation 0.
///
/// ```
/// use evolutionary::prelude::*;
///
/// let constant: Sigma = 0.1.into();
/// assert_eq!(constant.at(500), 0.1);
///
/// let halving = Sigma::exponential_decay(1.0, 0.5);
/// assert_eq!(halving.at(0), 1.0);
/// assert_eq!(halving.at(3), 0.125);
///
/// let stepped = Sigma::schedule(|generation| if generation < 100 { 0.5 } else { 0.05 });
/// assert_eq!(stepped.at(99), 0.5);
/// assert_eq!(stepped.at(100), 0.05);
/// ```
///
/// A schedule is called once per generation, from whichever thread runs the mutation, so it
/// must be `Send + Sync`, and it should depend on nothing but the generation: a seeded run
/// replays only if the schedule gives the same values every time. To replay on every platform
/// too, keep to `+`, `-`, `*`, `/` and comparisons, which round the same way everywhere;
/// `powi`, `powf`, `exp` and the like can differ in the last bit from one platform to another.
/// [`Sigma::exponential_decay`] uses multiplications only.
#[derive(Clone)]
#[non_exhaustive]
pub enum Sigma {
    /// The same `sigma` at every generation.
    Constant(f64),
    /// `sigma` as a function of the generation.
    Schedule(Arc<dyn Fn(u32) -> f64 + Send + Sync>),
}

impl Sigma {
    /// A `sigma` given by `schedule` at each generation.
    pub fn schedule(schedule: impl Fn(u32) -> f64 + Send + Sync + 'static) -> Self {
        Sigma::Schedule(Arc::new(schedule))
    }

    /// `start * factor^generation`: with a `factor` below 1, `sigma` shrinks by the same share
    /// at every generation, and the first generation of a run uses `start * factor`.
    ///
    /// The power is computed by repeated squaring, with multiplications only, so it is the same
    /// on every platform.
    pub fn exponential_decay(start: f64, factor: f64) -> Self {
        Sigma::schedule(move |generation| start * power(factor, generation))
    }

    /// The `sigma` at `generation`.
    pub fn at(&self, generation: u32) -> f64 {
        match self {
            Sigma::Constant(sigma) => *sigma,
            Sigma::Schedule(schedule) => schedule(generation),
        }
    }
}

impl From<f64> for Sigma {
    fn from(sigma: f64) -> Self {
        Sigma::Constant(sigma)
    }
}

impl fmt::Debug for Sigma {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sigma::Constant(sigma) => f.debug_tuple("Constant").field(sigma).finish(),
            Sigma::Schedule(_) => f.write_str("Schedule(..)"),
        }
    }
}

/// `base^exponent` by repeated squaring. Each step is one IEEE multiplication, which rounds the
/// same way on every platform, where `f64::powi` does not promise a precision.
fn power(base: f64, exponent: u32) -> f64 {
    let mut result = 1.0;
    let mut square = base;
    let mut exponent = exponent;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result *= square;
        }
        exponent >>= 1;
        if exponent > 0 {
            square *= square;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Step;
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
            sigma: 10.0.into(),
            ..GaussianMutation::default()
        };

        let genes = deltas(&mutation, (-1.0, 1.0));

        assert!(genes.iter().any(|gene| gene.abs() > 1.0));
    }

    #[test]
    fn clamps_to_the_range_of_each_individual() {
        let mutation = GaussianMutation {
            mutation_rate: 1.0,
            sigma: 10.0.into(),
            clamp_to_range: true,
        };

        let genes = deltas(&mutation, (-0.5, 2.0));

        assert!(genes.iter().all(|gene| (-0.5..=2.0).contains(gene)));
        // Clamping keeps the bounds reachable rather than resampling.
        assert!(genes.contains(&-0.5) && genes.contains(&2.0));
    }

    fn mutated_at(sigma: Sigma, generation: u32) -> Vec<f64> {
        let mutation = GaussianMutation::new(0.5, sigma);
        let mut population = vec![Real::new(vec![0.0; 50], (-1.0, 1.0)); 20];
        mutation.mutate(
            &mut population,
            &Rngs::for_step(3, generation, Step::Mutation),
        );

        population
            .iter()
            .flat_map(|individual| individual.get_chromosome().iter().copied())
            .collect()
    }

    #[test]
    fn a_schedule_is_read_at_the_generation_of_the_rngs() {
        let only_at_seven =
            || Sigma::schedule(|generation| if generation == 7 { 1.0 } else { 0.0 });

        assert!(mutated_at(only_at_seven(), 6)
            .iter()
            .all(|&gene| gene == 0.0));
        assert!(mutated_at(only_at_seven(), 7)
            .iter()
            .any(|&gene| gene != 0.0));
        assert!(mutated_at(only_at_seven(), 8)
            .iter()
            .all(|&gene| gene == 0.0));
    }

    #[test]
    fn a_schedule_draws_the_same_numbers_as_a_constant() {
        for generation in [0, 1, 25] {
            let constant = mutated_at(0.3.into(), generation);
            let scheduled = mutated_at(Sigma::schedule(|_| 0.3), generation);

            assert_eq!(constant, scheduled);
            assert!(constant.iter().any(|&gene| gene != 0.0));
        }
    }

    #[test]
    fn exponential_decay_multiplies_by_the_factor_each_generation() {
        let decay = Sigma::exponential_decay(2.0, 0.5);

        assert_eq!(decay.at(0), 2.0);
        assert_eq!(decay.at(1), 1.0);
        assert_eq!(decay.at(10), 2.0 / 1024.0);

        let decay = Sigma::exponential_decay(0.5, 0.99);
        for generation in [0, 1, 2, 3, 17, 100, 1000] {
            let expected = 0.5 * 0.99_f64.powf(f64::from(generation));
            let relative = (decay.at(generation) - expected).abs() / expected;
            assert!(relative < 1e-12, "{generation}: {}", decay.at(generation));
        }
    }

    #[test]
    fn power_by_squaring_matches_repeated_multiplication_for_exact_cases() {
        assert_eq!(power(3.0, 0), 1.0);
        assert_eq!(power(3.0, 1), 3.0);
        assert_eq!(power(3.0, 5), 243.0);
        assert_eq!(power(2.0, 40), 1_099_511_627_776.0);
        assert_eq!(power(0.0, 3), 0.0);
    }

    #[test]
    fn debug_shows_a_constant_and_hides_a_schedule() {
        assert_eq!(format!("{:?}", Sigma::from(0.25)), "Constant(0.25)");
        assert_eq!(
            format!("{:?}", Sigma::exponential_decay(1.0, 0.9)),
            "Schedule(..)"
        );
        assert!(format!("{:?}", GaussianMutation::default()).contains("sigma: Constant(1.0)"));
    }

    #[test]
    #[should_panic(expected = "finite sigma, got NaN at generation 4")]
    fn a_schedule_giving_a_non_finite_sigma_panics() {
        mutated_at(Sigma::schedule(|_| f64::NAN), 4);
    }
}

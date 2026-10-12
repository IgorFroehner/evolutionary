use crate::fitness::{fitness_cmp, selection_weight};
use crate::{Individual, Rngs, Selection};

use super::roulette_selection::spin;
use super::stochastic_universal_sampling_selection::sample;
use super::{RouletteSelection, StochasticUniversalSamplingSelection};

/// How a fitness-proportional selection turns fitnesses into the weights it draws with, given
/// to [`RouletteSelection::scaled`] or [`StochasticUniversalSamplingSelection::scaled`].
///
/// Unscaled, those selections weigh each individual by its raw fitness, which has to be
/// positive and whose scale decides how hard the best are favored: fitnesses of 1001 and 1000
/// make the two almost equally likely. Every scaling here accepts any finite fitness, negative
/// ones included, and gives an individual with a `NaN` fitness no chance, as unscaled selection
/// does. Only [`Rank`](Self::Rank) also accepts infinite fitnesses; see the `# Panics` section
/// of [`ScaledSelection`] for the others.
///
/// | Scaling | Weight | Favors the best |
/// |---|---|---|
/// | [`Window`](Self::Window) | `fitness - worst + epsilon` | by their distance from the worst |
/// | [`Rank`](Self::Rank) | rank, 1 for the worst | the same however far apart the fitnesses are |
/// | [`Sigma`](Self::Sigma) | `1 + (fitness - mean) / (2 * standard deviation)`, at least `0.1` | 1.5 times the average individual for one standard deviation above the mean, however spread out the fitnesses are |
///
/// ```
/// use evolutionary::prelude::*;
///
/// // Fitnesses that are all negative, which unscaled roulette refuses.
/// let population: Vec<Real> = (1..=4)
///     .map(|i| {
///         let mut individual = Real::new(vec![f64::from(i)], (0.0, 1.0));
///         individual.set_fitness(-f64::from(i));
///         individual
///     })
///     .collect();
///
/// let roulette = RouletteSelection::scaled(FitnessScaling::Window { epsilon: 0.1 });
/// let pool = roulette.get_mating_pool(&population, &Rngs::new(0));
///
/// assert_eq!(pool.len(), 4);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum FitnessScaling {
    /// Windowing: the weight is the fitness minus the worst fitness of the population, plus
    /// `epsilon`. The worst individual gets `epsilon`, so with `epsilon = 0` it never gets
    /// picked, and a population whose fitnesses are all equal has no weight at all. A small
    /// positive `epsilon`, next to the usual spread of the fitnesses, avoids both.
    ///
    /// The worst is taken over the fitnesses that are not `NaN`. An infinite fitness makes the
    /// total weight infinite, and the selection panics: [`Minimize`](crate::Minimize) of an
    /// error that overflowed gives `-inf`, for example. [`Rank`](Self::Rank) is the scaling
    /// that tolerates infinities.
    Window {
        /// Added to every weight. Must be finite and not negative.
        epsilon: f64,
    },
    /// Linear ranking: the weight is the rank of the fitness, from 1 for the worst to the
    /// number of individuals for the best. Equal fitnesses share the average of their ranks.
    /// It is the weighting of [`RankSelection`](crate::prelude::RankSelection), except that a
    /// `NaN` fitness gets no chance instead of the lowest rank.
    ///
    /// Only the order of the fitnesses matters, so it is the scaling that tolerates infinite
    /// fitnesses: `-inf` ranks below every number and `+inf` above.
    Rank,
    /// Sigma scaling: the weight is `1 + (fitness - mean) / (2 * standard deviation)`, at least
    /// `0.1`, where the mean and the standard deviation are over the finite fitnesses of the
    /// population. An individual one standard deviation above the mean weighs 1.5 times as
    /// much as an average one, however spread out the fitnesses are, so selection keeps about
    /// the same pressure while the population converges. When every finite fitness is the
    /// same, each of them weighs 1.
    ///
    /// An infinite fitness, like a `NaN` one, gets no chance and is left out of the mean and
    /// the standard deviation, so that it can't make every weight the same.
    Sigma,
}

impl FitnessScaling {
    /// The weight of each fitness, in order. Every scaled weight goes through
    /// `selection_weight`, so an individual whose scaled weight is `NaN` (a `NaN` fitness, or
    /// an infinite one under `Sigma`) gets 0.
    fn weights(self, fitnesses: &[f64]) -> Vec<f64> {
        let scaled: Vec<f64> = match self {
            FitnessScaling::Window { epsilon } => {
                assert!(
                    epsilon >= 0.0 && epsilon.is_finite(),
                    "windowed fitness scaling needs a finite epsilon, not negative, got {epsilon}"
                );
                // `f64::min` skips `NaN`s, so the worst is the worst number.
                let worst = fitnesses.iter().copied().fold(f64::INFINITY, f64::min);
                fitnesses
                    .iter()
                    .map(|&fitness| fitness - worst + epsilon)
                    .collect()
            }
            FitnessScaling::Rank => {
                let mut ranked: Vec<usize> = (0..fitnesses.len())
                    .filter(|&index| !fitnesses[index].is_nan())
                    .collect();
                ranked.sort_by(|&a, &b| fitness_cmp(fitnesses[a], fitnesses[b]));

                let mut ranks = vec![f64::NAN; fitnesses.len()];
                let mut start = 0;
                while start < ranked.len() {
                    let mut end = start + 1;
                    while end < ranked.len()
                        && fitness_cmp(fitnesses[ranked[start]], fitnesses[ranked[end]]).is_eq()
                    {
                        end += 1;
                    }
                    // Ranks `start + 1` to `end`, shared.
                    let rank = (start + 1 + end) as f64 / 2.0;
                    for &index in &ranked[start..end] {
                        ranks[index] = rank;
                    }
                    start = end;
                }
                ranks
            }
            FitnessScaling::Sigma => {
                // The statistics are over the finite fitnesses only: one infinity would make
                // the mean and the deviation `NaN`, and every weight the same floor.
                let finite = || {
                    fitnesses
                        .iter()
                        .copied()
                        .filter(|fitness| fitness.is_finite())
                };
                let count = finite().count();
                assert!(
                    count > 0,
                    "sigma fitness scaling needs at least one finite fitness, got none among {}",
                    fitnesses.len()
                );
                let count = count as f64;
                let mean = finite().sum::<f64>() / count;
                // Squares by multiplying: `powi` does not promise the same rounding on every
                // platform, and a seeded run must replay on all of them.
                let deviation = (finite()
                    .map(|fitness| (fitness - mean) * (fitness - mean))
                    .sum::<f64>()
                    / count)
                    .sqrt();
                assert!(
                    mean.is_finite() && deviation.is_finite(),
                    "sigma fitness scaling needs the mean and standard deviation of the finite \
                     fitnesses to be finite, got {mean} and {deviation}: they are too far apart"
                );

                fitnesses
                    .iter()
                    .map(|&fitness| {
                        if !fitness.is_finite() {
                            // No weight, through `selection_weight`.
                            f64::NAN
                        } else if deviation == 0.0 {
                            1.0
                        } else {
                            (1.0 + (fitness - mean) / (2.0 * deviation)).max(0.1)
                        }
                    })
                    .collect()
            }
        };

        scaled.into_iter().map(selection_weight).collect()
    }
}

/// A fitness-proportional selection that draws with scaled fitnesses, made by
/// [`RouletteSelection::scaled`] or [`StochasticUniversalSamplingSelection::scaled`]. See
/// [`FitnessScaling`].
///
/// The individuals of the mating pool keep their own fitness: only the weights they were drawn
/// with are scaled.
///
/// # Panics
///
/// `get_mating_pool` panics if the population is empty, if no weight is positive, or if the
/// total of the weights is not finite. For each scaling, a `NaN` fitness always gets no weight,
/// and:
///
/// - [`Window`](FitnessScaling::Window) panics if every fitness is `NaN`, if any fitness is
///   infinite (the total is then infinite), if `epsilon` is negative or not finite, or if
///   `epsilon` is 0 and every fitness that is not `NaN` is the same.
/// - [`Rank`](FitnessScaling::Rank) panics only if every fitness is `NaN`. Infinite fitnesses
///   rank like any other.
/// - [`Sigma`](FitnessScaling::Sigma) gives an infinite fitness no weight, and panics if no
///   fitness is finite, or if the finite ones are so far apart that their mean or standard
///   deviation overflows, or a weight does.
#[derive(Clone, Debug)]
pub struct ScaledSelection<S> {
    selection: S,
    scaling: FitnessScaling,
}

impl<S> ScaledSelection<S> {
    /// The selection that draws with the scaled weights.
    pub fn selection(&self) -> &S {
        &self.selection
    }

    /// How the fitnesses are scaled.
    pub fn scaling(&self) -> FitnessScaling {
        self.scaling
    }

    /// The scaled weight of each individual of `population`.
    fn weights<T: Individual>(&self, population: &[T]) -> Vec<f64> {
        let fitnesses: Vec<f64> = population
            .iter()
            .map(|individual| individual.get_fitness())
            .collect();
        self.scaling.weights(&fitnesses)
    }
}

impl RouletteSelection {
    /// Roulette selection that draws with fitnesses scaled by `scaling`, so they can be
    /// negative. See [`FitnessScaling`].
    ///
    /// ```
    /// use evolutionary::prelude::*;
    ///
    /// let selection = RouletteSelection::scaled(FitnessScaling::Rank);
    /// ```
    pub fn scaled(scaling: FitnessScaling) -> ScaledSelection<RouletteSelection> {
        ScaledSelection {
            selection: RouletteSelection,
            scaling,
        }
    }
}

impl StochasticUniversalSamplingSelection {
    /// Stochastic universal sampling that draws with fitnesses scaled by `scaling`, so they can
    /// be negative. See [`FitnessScaling`].
    ///
    /// ```
    /// use evolutionary::prelude::*;
    ///
    /// let selection =
    ///     StochasticUniversalSamplingSelection::scaled(FitnessScaling::Window { epsilon: 1e-3 });
    /// ```
    pub fn scaled(
        scaling: FitnessScaling,
    ) -> ScaledSelection<StochasticUniversalSamplingSelection> {
        ScaledSelection {
            selection: StochasticUniversalSamplingSelection,
            scaling,
        }
    }
}

impl<T: Individual> Selection<T> for ScaledSelection<RouletteSelection> {
    fn get_mating_pool(&self, initial_population: &[T], rngs: &Rngs) -> Vec<T> {
        spin(
            initial_population,
            self.weights(initial_population).into_iter(),
            rngs,
            "scaled roulette selection needs individuals, at least one fitness that is not NaN, \
             and scaled weights with a positive, finite total",
        )
    }
}

impl<T: Individual> Selection<T> for ScaledSelection<StochasticUniversalSamplingSelection> {
    fn get_mating_pool(&self, initial_population: &[T], rngs: &Rngs) -> Vec<T> {
        let weights = self.weights(initial_population);
        sample(
            initial_population,
            |index| weights[index],
            rngs,
            "scaled weight",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::population::Real;
    use crate::selection::tests::population;

    const WINDOW: FitnessScaling = FitnessScaling::Window { epsilon: 0.5 };

    fn roulette(scaling: FitnessScaling, fitnesses: &[f64], seed: u64) -> Vec<f64> {
        RouletteSelection::scaled(scaling)
            .get_mating_pool(&population(fitnesses), &Rngs::new(seed))
            .iter()
            .map(|individual| individual.get_fitness())
            .collect()
    }

    fn sus(scaling: FitnessScaling, fitnesses: &[f64], seed: u64) -> Vec<f64> {
        StochasticUniversalSamplingSelection::scaled(scaling)
            .get_mating_pool(&population(fitnesses), &Rngs::new(seed))
            .iter()
            .map(|individual| individual.get_fitness())
            .collect()
    }

    #[test]
    fn window_subtracts_the_worst_and_adds_epsilon() {
        assert_eq!(
            WINDOW.weights(&[-3.0, -1.0, f64::NAN, -2.5]),
            vec![0.5, 2.5, 0.0, 1.0]
        );
        assert_eq!(
            FitnessScaling::Window { epsilon: 0.0 }.weights(&[5.0, 5.0]),
            vec![0.0, 0.0]
        );
    }

    #[test]
    fn rank_shares_the_ranks_of_ties_and_skips_nan() {
        assert_eq!(
            FitnessScaling::Rank.weights(&[-10.0, f64::NAN, 7.0, -10.0, 1e9, 7.0]),
            vec![1.5, 0.0, 3.5, 1.5, 5.0, 3.5]
        );
    }

    #[test]
    fn sigma_weighs_by_standard_deviations_from_the_mean() {
        // A mean of -4 and squared distances from it of 4, 36, 4, 36 and 256, all exact. The
        // -20, almost two standard deviations below, gets the floor of 0.1.
        let weights = FitnessScaling::Sigma.weights(&[-2.0, 2.0, f64::NAN, -2.0, 2.0, -20.0]);
        let mean = -4.0;
        let deviation = (336.0 / 5.0_f64).sqrt();
        let expected = |fitness: f64| 1.0 + (fitness - mean) / (2.0 * deviation);
        assert_eq!(
            weights,
            vec![
                expected(-2.0),
                expected(2.0),
                0.0,
                expected(-2.0),
                expected(2.0),
                0.1
            ]
        );
        assert_eq!(FitnessScaling::Sigma.weights(&[-3.0, -3.0]), vec![1.0, 1.0]);
    }

    #[test]
    fn scaling_works_on_an_all_negative_population() {
        let fitnesses: Vec<f64> = (0..400).map(|i| -1000.0 - f64::from(i % 4)).collect();

        for scaling in [WINDOW, FitnessScaling::Rank, FitnessScaling::Sigma] {
            let pool = roulette(scaling, &fitnesses, 1);
            assert_eq!(pool.len(), fitnesses.len());
            // The best (-1000) is picked more than the worst (-1003).
            let count = |value: f64| pool.iter().filter(|&&fitness| fitness == value).count();
            assert!(count(-1000.0) > count(-1003.0), "{scaling:?}");

            let pool = sus(scaling, &fitnesses, 1);
            assert_eq!(pool.len(), fitnesses.len());
            let count = |value: f64| pool.iter().filter(|&&fitness| fitness == value).count();
            assert!(count(-1000.0) > count(-1003.0), "{scaling:?}");
        }
    }

    #[test]
    fn windowed_roulette_picks_in_proportion_to_the_distance_from_the_worst() {
        // Fitnesses -4, -3, -2 and -1 a thousand times each, with an epsilon of 1: weights 1, 2,
        // 3 and 4, so fitness `f` fills `(f + 5) / 10` of the pool.
        let fitnesses: Vec<f64> = (0..4000).map(|i| -f64::from(i % 4 + 1)).collect();
        let pool = roulette(FitnessScaling::Window { epsilon: 1.0 }, &fitnesses, 6);

        for fitness in [-4.0_f64, -3.0, -2.0, -1.0] {
            let share = (fitness + 5.0) / 10.0;
            let expected = share * 4000.0;
            let deviation = (4000.0 * share * (1.0 - share)).sqrt();
            let count = pool.iter().filter(|&&picked| picked == fitness).count() as f64;
            assert!(
                (count - expected).abs() < 4.0 * deviation,
                "{fitness}: {count}"
            );
        }
    }

    #[test]
    fn windowed_sus_picks_each_individual_its_expected_number_of_times() {
        // Weights 1, 2, 0 (NaN) and 5, over a total of 8 for 4 slots: 0.5, 1, 0 and 2.5 picks.
        let fitnesses = [-5.0, -4.0, f64::NAN, -1.0];
        for seed in 0..50 {
            let pool = sus(FitnessScaling::Window { epsilon: 1.0 }, &fitnesses, seed);
            let count = |value: f64| pool.iter().filter(|&&fitness| fitness == value).count();

            assert!(count(-5.0) <= 1, "seed {seed}");
            assert_eq!(count(-4.0), 1, "seed {seed}");
            assert!((2..=3).contains(&count(-1.0)), "seed {seed}");
            assert_eq!(pool.len(), 4);
        }
    }

    #[test]
    fn nan_is_never_picked_with_any_scaling() {
        let fitnesses: Vec<f64> = [f64::NAN, -7.0, -3.0, f64::NAN, -5.0].repeat(200);

        for scaling in [
            WINDOW,
            FitnessScaling::Window { epsilon: 0.0 },
            FitnessScaling::Rank,
            FitnessScaling::Sigma,
        ] {
            for seed in 0..5 {
                let pool = roulette(scaling, &fitnesses, seed);
                assert!(!pool.iter().any(|fitness| fitness.is_nan()), "{scaling:?}");

                let pool = sus(scaling, &fitnesses, seed);
                assert!(!pool.iter().any(|fitness| fitness.is_nan()), "{scaling:?}");
            }
        }
    }

    #[test]
    fn the_pool_keeps_the_unscaled_fitnesses() {
        let pool = roulette(FitnessScaling::Rank, &[-2.0, -1.0], 0);

        assert!(pool
            .iter()
            .all(|&fitness| fitness == -2.0 || fitness == -1.0));
    }

    #[test]
    #[should_panic(expected = "scaled roulette selection needs")]
    fn roulette_panics_when_every_fitness_is_nan() {
        roulette(FitnessScaling::Rank, &[f64::NAN, f64::NAN], 0);
    }

    #[test]
    fn sigma_leaves_infinite_fitnesses_out() {
        // Without the infinity: a mean of 2 and a standard deviation of sqrt(2/3).
        let weights = FitnessScaling::Sigma.weights(&[1.0, 2.0, f64::INFINITY, 3.0]);
        let deviation = (2.0 / 3.0_f64).sqrt();
        assert_eq!(
            weights,
            vec![
                1.0 - 1.0 / (2.0 * deviation),
                1.0,
                0.0,
                1.0 + 1.0 / (2.0 * deviation)
            ]
        );
        assert_eq!(
            FitnessScaling::Sigma.weights(&[f64::NEG_INFINITY, 5.0, 5.0]),
            vec![0.0, 1.0, 1.0]
        );

        // Not one pick each, as when the infinity made every weight the floor: weights of
        // about 0.39, 1, 0 and 1.61 out of a total of 3, so 0.52, 1.33, 0 and 2.15 picks of 4.
        for seed in 0..50 {
            let pool = sus(FitnessScaling::Sigma, &[1.0, 2.0, f64::INFINITY, 3.0], seed);
            let count = |value: f64| pool.iter().filter(|&&fitness| fitness == value).count();
            assert_eq!(count(f64::INFINITY), 0, "seed {seed}");
            assert!(count(1.0) <= 1, "seed {seed}");
            assert!((1..=2).contains(&count(2.0)), "seed {seed}");
            assert!((2..=3).contains(&count(3.0)), "seed {seed}");
        }
    }

    #[test]
    #[should_panic(expected = "sigma fitness scaling needs at least one finite fitness")]
    fn sigma_panics_when_no_fitness_is_finite() {
        sus(
            FitnessScaling::Sigma,
            &[f64::INFINITY, f64::NEG_INFINITY, f64::NAN],
            0,
        );
    }

    #[test]
    #[should_panic(expected = "sigma fitness scaling needs the mean and standard deviation")]
    fn sigma_panics_when_the_statistics_overflow() {
        roulette(FitnessScaling::Sigma, &[f64::MAX, f64::MAX, 1.0], 0);
    }

    #[test]
    #[should_panic(expected = "scaled roulette selection needs")]
    fn window_panics_on_an_infinite_fitness() {
        roulette(WINDOW, &[f64::NEG_INFINITY, -1.0, -2.0], 0);
    }

    #[test]
    fn rank_tolerates_infinite_fitnesses() {
        let fitnesses = [f64::NEG_INFINITY, -1.0, f64::INFINITY, f64::NAN];
        assert_eq!(
            FitnessScaling::Rank.weights(&fitnesses),
            vec![1.0, 2.0, 3.0, 0.0]
        );
        assert_eq!(roulette(FitnessScaling::Rank, &fitnesses, 0).len(), 4);
    }

    #[test]
    fn ranked_sus_picks_each_individual_its_expected_number_of_times_with_ties() {
        // Ranks: the two 1s share 1.5, the 3 is 3, the three 5s share 5 and the 9 is 7, out of
        // a total of 28 for 8 slots. The NaN gets nothing.
        let fitnesses = [5.0, 5.0, 1.0, 9.0, 1.0, 5.0, f64::NAN, 3.0];
        let weights = [5.0, 5.0, 1.5, 7.0, 1.5, 5.0, 0.0, 3.0];
        assert_eq!(FitnessScaling::Rank.weights(&fitnesses), weights);
        // Tell the individuals apart by their only gene, since fitnesses repeat.
        let population: Vec<Real> = fitnesses
            .iter()
            .enumerate()
            .map(|(index, &fitness)| {
                let mut individual = Real::new(vec![index as f64], (0.0, 1.0));
                individual.set_fitness(fitness);
                individual
            })
            .collect();

        for seed in 0..200 {
            let pool = StochasticUniversalSamplingSelection::scaled(FitnessScaling::Rank)
                .get_mating_pool(&population, &Rngs::new(seed));

            assert_eq!(pool.len(), population.len());
            for (index, weight) in weights.iter().enumerate() {
                let expected = weight * 8.0 / 28.0;
                let count = pool
                    .iter()
                    .filter(|individual| individual.chromosome[0] == index as f64)
                    .count() as f64;
                assert!(
                    count == expected.floor() || count == expected.ceil(),
                    "seed {seed}: individual {index} picked {count} times, expected {expected}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "at least one positive scaled weight")]
    fn sus_panics_on_an_equal_population_with_no_epsilon() {
        sus(FitnessScaling::Window { epsilon: 0.0 }, &[-1.0, -1.0], 0);
    }

    #[test]
    #[should_panic(expected = "finite epsilon, not negative, got -1")]
    fn a_negative_epsilon_panics() {
        sus(FitnessScaling::Window { epsilon: -1.0 }, &[-1.0, -2.0], 0);
    }

    #[test]
    #[should_panic(expected = "scaled roulette selection needs")]
    fn an_empty_population_panics() {
        roulette(FitnessScaling::Rank, &[], 0);
    }
}

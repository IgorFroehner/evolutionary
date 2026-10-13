use crate::fitness::selection_weight;
use crate::{Individual, Rngs, Selection};
use rand::Rng;

/// # Stochastic Universal Sampling Selection
///
/// Picks the whole mating pool with one spin of a wheel of evenly spaced pointers, where each
/// individual's slice is proportional to its fitness. Each individual is picked its expected
/// number of times, rounded down or up. Fitnesses must not be negative; an individual with a
/// `NaN` fitness gets no chance.
///
/// For fitnesses that can be negative, as with [`Minimize`](crate::Minimize), or to set how
/// strongly the best individuals are favored, scale them first with
/// [`StochasticUniversalSamplingSelection::scaled`].
///
/// # Panics
///
/// If no fitness is positive or their total is not finite.
#[derive(Clone, Debug, Default)]
pub struct StochasticUniversalSamplingSelection;

impl<T: Individual> Selection<T> for StochasticUniversalSamplingSelection {
    fn get_mating_pool(&self, initial_population: &[T], rngs: &Rngs) -> Vec<T> {
        sample(
            initial_population,
            |index| selection_weight(initial_population[index].get_fitness()),
            rngs,
            "fitness",
        )
    }
}

/// Picks a mating pool as large as `population` with one spin of a wheel where each
/// individual's slice is `weight(index)`. Panics if no weight is positive or their total is not
/// finite, calling the weights `what` in the message.
pub(super) fn sample<T: Individual>(
    population: &[T],
    weight: impl Fn(usize) -> f64,
    rngs: &Rngs,
    what: &str,
) -> Vec<T> {
    let population_size = population.len();
    let total_weight: f64 = (0..population_size).map(&weight).sum();
    assert!(
        total_weight > 0.0 && total_weight.is_finite(),
        "stochastic universal sampling needs at least one positive {what} and a finite total, \
         got a total of {total_weight}"
    );
    let pointer_spacing = total_weight / population_size as f64;

    let start_point = rngs.rng(0).random_range(0.0..pointer_spacing);

    walk(population, weight, start_point, pointer_spacing)
}

/// Places `population.len()` pointers `pointer_spacing` apart from `start_point` on a wheel
/// where each individual's slice is `weight(index)`, and returns the individual under each.
///
/// A pointer inside the wheel picks the individual whose slice ends at or after it. A pointer at
/// `0.0` (a start point of exactly `0.0`) picks the first individual with a positive weight,
/// never one with a weight of zero.
fn walk<T: Clone>(
    population: &[T],
    weight: impl Fn(usize) -> f64,
    start_point: f64,
    pointer_spacing: f64,
) -> Vec<T> {
    let population_size = population.len();

    // One walk over the wheel places every pointer. The slots could be picked independently,
    // each with a binary search over the cumulative weights, but the work per slot is a clone,
    // and at a few hundred individuals the whole walk takes less than rayon's fork and join: a
    // parallel version measured slower up to 10 000 individuals. (The pointers used to be
    // computed with rayon too, which cost more than the walk itself.)
    let mut selected_individuals = Vec::with_capacity(population_size);
    let mut cumulative_weight = 0.0;
    let mut individual_idx = 0;

    for slot in 0..population_size {
        let pointer = start_point + slot as f64 * pointer_spacing;
        // `cumulative_weight <= 0.0` only matters for a pointer at 0.0: a pointer above it
        // already satisfies `cumulative_weight < pointer` while nothing positive is summed. It
        // moves the walk past the leading zero weights to the first positive one, where it
        // would otherwise not move at all and read `population[0 - 1]`.
        while (cumulative_weight < pointer || cumulative_weight <= 0.0)
            && individual_idx < population_size
        {
            cumulative_weight += weight(individual_idx);
            individual_idx += 1;
        }
        selected_individuals.push(population[individual_idx - 1].clone());
    }

    selected_individuals
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::fitness_cmp;
    use crate::selection::tests::{picked, population};

    #[test]
    fn picks_each_individual_its_expected_number_of_times_rounded_down_or_up() {
        let fitnesses = [1.0, 2.0, 3.0, 4.0, 0.0, 10.0, f64::NAN, 5.0];
        let population = population(&fitnesses);
        let size = fitnesses.len() as f64;
        let total: f64 = fitnesses.iter().copied().map(selection_weight).sum();

        for seed in 0..50 {
            let pool =
                StochasticUniversalSamplingSelection.get_mating_pool(&population, &Rngs::new(seed));

            assert_eq!(pool.len(), population.len());
            for &fitness in &fitnesses {
                let expected = selection_weight(fitness) * size / total;
                let count = picked(&pool, |picked| fitness_cmp(picked, fitness).is_eq()) as f64;

                assert!(
                    count == expected.floor() || count == expected.ceil(),
                    "seed {seed}: fitness {fitness} picked {count} times, expected {expected}"
                );
            }
        }
    }

    /// The fitnesses of the individuals `walk` picks from `fitnesses` for `start_point`.
    fn walked(fitnesses: &[f64], start_point: f64) -> Vec<f64> {
        let population = population(fitnesses);
        let total: f64 = fitnesses.iter().copied().map(selection_weight).sum();
        let spacing = total / fitnesses.len() as f64;

        walk(
            &population,
            |index| selection_weight(population[index].get_fitness()),
            start_point,
            spacing,
        )
        .iter()
        .map(Individual::get_fitness)
        .collect()
    }

    #[test]
    fn start_point_at_zero_picks_the_first_individual_when_its_weight_is_positive() {
        // Spacing 2.5: pointers at 0.0, 2.5, 5.0 and 7.5.
        assert_eq!(walked(&[1.0, 2.0, 3.0, 4.0], 0.0), [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn start_point_at_zero_skips_leading_zero_weights() {
        // Spacing 1.5: pointers at 0.0, 1.5, 3.0 and 4.5.
        assert_eq!(
            walked(&[0.0, f64::NAN, 2.0, 4.0], 0.0),
            [2.0, 2.0, 4.0, 4.0]
        );
        // Spacing 0.5: pointers at 0.0 and 0.5.
        assert_eq!(walked(&[0.0, 1.0], 0.0), [1.0, 1.0]);
    }

    #[test]
    fn start_point_above_zero_picks_the_slice_that_ends_at_or_after_each_pointer() {
        // Spacing 2.5: pointers at 0.5, 3.0, 5.5 and 8.0. The pointer at 3.0 is on the end of
        // the second slice and picks it; the zero weight is never picked.
        assert_eq!(walked(&[1.0, 2.0, 0.0, 7.0], 0.5), [1.0, 2.0, 7.0, 7.0]);
        // The smallest positive start point still skips a leading zero weight.
        assert_eq!(
            walked(&[0.0, 2.0, 3.0, 5.0], f64::MIN_POSITIVE),
            [2.0, 3.0, 3.0, 5.0]
        );
    }

    #[test]
    #[should_panic(expected = "at least one positive fitness")]
    fn no_positive_fitness_panics() {
        let population = population(&[0.0, f64::NAN, 0.0]);

        StochasticUniversalSamplingSelection.get_mating_pool(&population, &Rngs::new(0));
    }
}

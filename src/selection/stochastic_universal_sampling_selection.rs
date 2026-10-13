use crate::fitness::selection_weight;
use crate::{Individual, Rngs, Selection};
use rand::Rng;

/// # Stochastic Universal Sampling Selection
///
/// Picks the whole mating pool with one spin of a wheel of evenly spaced pointers, where each
/// individual's slice is proportional to its fitness. Each individual is picked its expected
/// number of times, rounded down or up. Fitnesses must not be negative; an individual with a
/// fitness of zero or `NaN` gets no chance and is never picked.
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
/// A pointer inside the wheel picks the individual whose slice ends at or after it. No pointer
/// ever picks an individual with a weight of zero: a pointer at `0.0` (a start point of exactly
/// `0.0`) picks the first individual with a positive weight, and a pointer that rounding puts
/// past the end of the wheel (a start point just below `pointer_spacing`) picks the last one.
///
/// At least one weight must be positive, which `sample` asserts.
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
        let mut picked = individual_idx - 1;
        // The loop above only stops inside the wheel on an individual with a positive weight: a
        // zero weight leaves `cumulative_weight` as it was, so the loop would have stopped on
        // the individual before it. The only way to land on a zero weight is to run off the end
        // with the pointer still past the total, which rounding in `start_point + slot *
        // pointer_spacing` can do in the last slot when the start point is just below
        // `pointer_spacing`. Then the pointer belongs to the last slice with a positive weight,
        // so step back over the trailing zero weights. At least one weight is positive (`sample`
        // asserts a positive total), so this stops on it and never steps back past index 0.
        // Inside the wheel `picked` already has a positive weight and nothing changes.
        if individual_idx == population_size {
            while weight(picked) <= 0.0 {
                picked -= 1;
            }
        }
        selected_individuals.push(population[picked].clone());
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

    /// The weights from issue #19: the last positive one is at index 9.
    const ISSUE_19_WEIGHTS: [f64; 12] = [
        0.0,
        4.742212123340738,
        84.56193545410987,
        0.0003862830995540634,
        0.0005928994696346527,
        474.8988276210316,
        0.0,
        0.0,
        0.5847506407363268,
        0.0005634069431715673,
        0.0,
        0.0,
    ];

    #[test]
    fn last_pointer_past_the_total_picks_the_last_positive_weight() {
        // With the start point one ulp below the spacing, rounding puts the last pointer past
        // the total, and the walk used to pick index 11, a weight of zero. It belongs to the
        // last positive weight, index 9.
        let fitnesses = ISSUE_19_WEIGHTS;
        let total: f64 = fitnesses.iter().sum();
        let spacing = total / fitnesses.len() as f64;
        let start = (1.0 - f64::EPSILON) * spacing;
        assert!(
            start + 11.0 * spacing > total,
            "the last pointer should overshoot the total"
        );

        let picked = walked(&fitnesses, start);

        assert_eq!(picked.len(), fitnesses.len());
        assert_eq!(picked[11].to_bits(), fitnesses[9].to_bits());
        assert!(picked.iter().all(|&fitness| fitness > 0.0), "{picked:?}");
    }

    #[test]
    fn last_pointer_past_the_total_skips_a_trailing_nan_and_zero_tail() {
        // Spacing 5.7, and rounding puts the last pointer past the total for this start point.
        // The walk used to pick the zero weight at the end; the `NaN` before it weighs 0 too.
        let fitnesses = [7.6, 7.3, 8.5, 9.5, 7.0, f64::NAN, 0.0];
        let total: f64 = fitnesses.iter().copied().map(selection_weight).sum();
        let spacing = total / fitnesses.len() as f64;
        let start = (1.0 - f64::EPSILON) * spacing;
        assert!(
            start + 6.0 * spacing > total,
            "the last pointer should overshoot the total"
        );

        assert_eq!(
            walked(&fitnesses, start),
            [7.6, 7.3, 8.5, 8.5, 9.5, 7.0, 7.0]
        );
    }

    #[test]
    fn start_point_in_the_middle_picks_what_it_did_before_issue_19() {
        // The start point halfway along the first spacing, far from either boundary.
        let fitnesses = ISSUE_19_WEIGHTS;
        let total: f64 = fitnesses.iter().sum();
        let spacing = total / fitnesses.len() as f64;

        assert_eq!(
            walked(&fitnesses, 0.5 * spacing),
            [
                84.56193545410987,
                84.56193545410987,
                474.8988276210316,
                474.8988276210316,
                474.8988276210316,
                474.8988276210316,
                474.8988276210316,
                474.8988276210316,
                474.8988276210316,
                474.8988276210316,
                474.8988276210316,
                474.8988276210316,
            ]
        );
    }

    #[test]
    #[should_panic(expected = "at least one positive fitness")]
    fn no_positive_fitness_panics() {
        let population = population(&[0.0, f64::NAN, 0.0]);

        StochasticUniversalSamplingSelection.get_mating_pool(&population, &Rngs::new(0));
    }
}

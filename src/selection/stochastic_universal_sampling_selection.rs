use crate::fitness::selection_weight;
use crate::{Individual, Rngs, Selection};
use rand::Rng;
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelIterator;

/// # Stochastic Universal Sampling Selection
///
/// Picks the whole mating pool with one spin of a wheel of evenly spaced pointers, where each
/// individual's slice is proportional to its fitness. Each individual is picked its expected
/// number of times, rounded down or up. Fitnesses must not be negative; an individual with a
/// `NaN` fitness gets no chance.
///
/// # Panics
///
/// If no fitness is positive or their total is not finite.
#[derive(Clone, Default)]
pub struct StochasticUniversalSamplingSelection;

impl<T: Individual> Selection<T> for StochasticUniversalSamplingSelection {
    fn get_mating_pool(&self, initial_population: &[T], rngs: &Rngs) -> Vec<T> {
        let total_fitness: f64 = initial_population
            .iter()
            .map(|ind| selection_weight(ind.get_fitness()))
            .sum();
        assert!(
            total_fitness > 0.0 && total_fitness.is_finite(),
            "stochastic universal sampling needs at least one positive fitness and a finite \
             total, got a total of {total_fitness}"
        );
        let population_size = initial_population.len();
        let pointer_spacing = total_fitness / population_size as f64;

        let start_point = rngs.rng(0).random_range(0.0..pointer_spacing);

        let pointers: Vec<_> = (0..population_size)
            .into_par_iter()
            .map(|i| start_point + i as f64 * pointer_spacing)
            .collect();

        let mut selected_individuals = Vec::new();
        let mut cumulative_fitness = 0.0;
        let mut individual_idx = 0;

        // TODO: find a way to parallelize this part
        for pointer in pointers {
            while cumulative_fitness < pointer && individual_idx < population_size {
                cumulative_fitness +=
                    selection_weight(initial_population[individual_idx].get_fitness());
                individual_idx += 1;
            }
            selected_individuals.push(initial_population[individual_idx - 1].clone());
        }

        selected_individuals
    }
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

    #[test]
    #[should_panic(expected = "at least one positive fitness")]
    fn no_positive_fitness_panics() {
        let population = population(&[0.0, f64::NAN, 0.0]);

        StochasticUniversalSamplingSelection.get_mating_pool(&population, &Rngs::new(0));
    }
}

use rand::distr::{weighted::WeightedIndex, Distribution};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::fitness::selection_weight;
use crate::population::Individual;
use crate::rng::Rngs;

use super::Selection;

/// # Roulette Selection
///
/// Each slot of the mating pool draws an individual with probability proportional to its
/// fitness. Fitnesses must not be negative; an individual with a `NaN` fitness gets no chance.
///
/// # Panics
///
/// `get_mating_pool` panics if the population is empty, if a fitness is negative, or if no
/// fitness is positive or their total is not finite.
#[derive(Clone, Default)]
pub struct RouletteSelection;

impl<T: Individual> Selection<T> for RouletteSelection {
    fn get_mating_pool(&self, initial_population: &[T], rngs: &Rngs) -> Vec<T> {
        // Built once for the whole pool: building it per slot would walk the population again
        // for every draw.
        let wheel = WeightedIndex::new(
            initial_population
                .iter()
                .map(|individual| selection_weight(individual.get_fitness())),
        )
        .expect(
            "roulette selection needs individuals, no negative fitness and at least one \
             positive, finite fitness",
        );

        (0..initial_population.len())
            .into_par_iter()
            .map(|slot| initial_population[wheel.sample(&mut rngs.rng(slot))].clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::selection::tests::{picked, population};

    #[test]
    fn picks_in_proportion_to_fitness() {
        // Fitnesses 1, 2, 3 and 4 a thousand times each: fitness `f` should fill `f / 10` of
        // the pool.
        let fitnesses: Vec<f64> = (0..4000).map(|i| f64::from(i % 4 + 1)).collect();
        let population = population(&fitnesses);

        let pool = RouletteSelection.get_mating_pool(&population, &Rngs::new(6));

        assert_eq!(pool.len(), population.len());
        for fitness in [1.0_f64, 2.0, 3.0, 4.0] {
            let share = fitness / 10.0;
            let expected = share * 4000.0;
            let deviation = (4000.0 * share * (1.0 - share)).sqrt();

            let count = picked(&pool, |picked| picked == fitness) as f64;
            assert!(
                (count - expected).abs() < 4.0 * deviation,
                "{fitness}: {count}"
            );
        }
    }

    #[test]
    fn zero_and_nan_fitnesses_are_never_picked() {
        let fitnesses: Vec<f64> = [0.0, f64::NAN, 1.0, 2.0].repeat(250);
        let population = population(&fitnesses);

        let pool = RouletteSelection.get_mating_pool(&population, &Rngs::new(6));

        assert_eq!(
            picked(&pool, |fitness| fitness == 0.0 || fitness.is_nan()),
            0
        );
    }

    #[test]
    #[should_panic(expected = "roulette selection needs")]
    fn a_negative_fitness_panics() {
        let population = population(&[1.0, -1.0, 2.0]);

        RouletteSelection.get_mating_pool(&population, &Rngs::new(0));
    }

    #[test]
    #[should_panic(expected = "roulette selection needs")]
    fn an_empty_population_panics() {
        RouletteSelection.get_mating_pool(&population(&[]), &Rngs::new(0));
    }
}

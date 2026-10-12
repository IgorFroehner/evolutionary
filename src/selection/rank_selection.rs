use crate::fitness::fitness_cmp;
use crate::{Individual, Rngs, Selection};
use rand::distr::{weighted::WeightedIndex, Distribution};
use rayon::prelude::*;

/// # Rank Selection
///
/// It creates an ranking of the individuals based on their fitness, where the works takes 1, the next 2, and follows
/// linearly so that the best receives `n` (number of individuals in the population). Then it chooses the individuals
/// with probability based on the ranking value.
///
/// # Panics
///
/// `get_mating_pool` panics if the population is empty.
#[derive(Clone, Default)]
pub struct RankSelection;

impl<T: Individual> Selection<T> for RankSelection {
    fn get_mating_pool(&self, initial_population: &[T], rngs: &Rngs) -> Vec<T> {
        // Ranks the indices rather than a copy of the population, so only the individuals
        // drawn into the pool are cloned.
        let mut ranked: Vec<usize> = (0..initial_population.len()).collect();
        ranked.par_sort_by(|&a, &b| {
            fitness_cmp(
                initial_population[a].get_fitness(),
                initial_population[b].get_fitness(),
            )
        });

        let population_size = initial_population.len();
        let weights: Vec<_> = (1..=population_size).collect();

        let dist = WeightedIndex::new(&weights).unwrap();

        (0..population_size)
            .into_par_iter()
            .map(|slot| {
                let index = dist.sample(&mut rngs.rng(slot));
                initial_population[ranked[index]].clone()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::selection::tests::{picked, population};

    // A thousand individuals ranked 1 to 1 000: the best half holds 375 250 of the 500 500 rank
    // points, so it fills 75% of the pool, with a standard deviation of about 14 individuals.

    #[test]
    fn the_best_half_fills_three_quarters_of_the_pool() {
        // Listed from the best to the worst, so the ranking can't just follow the positions.
        let fitnesses: Vec<f64> = (0..1000).rev().map(f64::from).collect();
        let population = population(&fitnesses);

        let pool = RankSelection.get_mating_pool(&population, &Rngs::new(4));

        assert_eq!(pool.len(), population.len());
        let best_half = picked(&pool, |fitness| fitness >= 500.0) as f64;
        assert!((best_half - 750.0).abs() < 4.0 * 14.0, "{best_half}");
    }

    #[test]
    fn nan_ranks_below_negative_fitnesses() {
        let fitnesses: Vec<f64> = (0..1000)
            .map(|i| if i % 2 == 0 { f64::NAN } else { -1e9 })
            .collect();
        let population = population(&fitnesses);

        let pool = RankSelection.get_mating_pool(&population, &Rngs::new(4));

        let nans = picked(&pool, f64::is_nan) as f64;
        assert!((nans - 250.0).abs() < 4.0 * 14.0, "{nans}");
    }

    #[test]
    #[should_panic]
    fn an_empty_population_panics() {
        RankSelection.get_mating_pool(&population(&[]), &Rngs::new(0));
    }
}

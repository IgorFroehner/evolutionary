use crate::fitness::fitness_cmp;
use crate::{Individual, Rngs, Selection};
use rand::distr::{weighted::WeightedIndex, Distribution};
use rayon::prelude::*;

/// # Rank Selection
///
/// It creates an ranking of the individuals based on their fitness, where the works takes 1, the next 2, and follows
/// linearly so that the best receives `n` (number of individuals in the population). Then it chooses the individuals
/// with probability based on the ranking value.
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

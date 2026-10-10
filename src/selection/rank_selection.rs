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
    fn get_mating_pool(&self, initial_population: &Vec<T>, rngs: &Rngs) -> Vec<T> {
        let mut sorted_population = initial_population.clone();
        sorted_population.par_sort_by(|a, b| fitness_cmp(a.get_fitness(), b.get_fitness()));

        let population_size = sorted_population.len();
        let weights: Vec<_> = (1..=population_size).map(|rank| rank).collect();

        let dist = WeightedIndex::new(&weights).unwrap();

        (0..population_size)
            .into_par_iter()
            .map(|slot| {
                let index = dist.sample(&mut rngs.rng(slot));
                sorted_population[index].clone()
            })
            .collect()
    }
}

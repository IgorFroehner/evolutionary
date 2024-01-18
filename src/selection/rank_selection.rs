use crate::{Individual, Selection};
use rand::thread_rng;
use rayon::prelude::*;
use std::cmp::Ordering;
use rand::distributions::{Distribution, WeightedIndex};

/// # Rank Selection
///
/// It creates an ranking of the individuals based on their fitness, where the works takes 1, the next 2, and follows
/// linearly so that the best receives `n` (number of individuals in the population). Then it chooses the individuals
/// with probability based on the ranking value.
#[derive(Clone, Default)]
pub struct RankSelection;

impl<T: Individual> Selection<T> for RankSelection {
    fn get_mating_pool(&self, initial_population: &Vec<T>) -> Vec<T> {
        let mut sorted_population = initial_population.clone();
        sorted_population.par_sort_by(|a, b| {
            a.get_fitness()
                .partial_cmp(&b.get_fitness())
                .unwrap_or(Ordering::Equal)
        });

        let population_size = sorted_population.len();
        let weights: Vec<_> = (1..=population_size).map(|rank| rank).collect();

        let dist = WeightedIndex::new(&weights).unwrap();

        (0..population_size)
            .into_par_iter()
            .map_init(
                || thread_rng(),
                |mut rng, _| {
                let index = dist.sample(&mut rng);
                sorted_population[index].clone()
            })
            .collect()
    }
}

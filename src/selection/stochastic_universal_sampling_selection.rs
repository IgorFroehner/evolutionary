use crate::fitness::selection_weight;
use crate::{Individual, Rngs, Selection};
use rand::Rng;
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelIterator;

#[derive(Clone, Default)]
pub struct StochasticUniversalSamplingSelection;

impl<T: Individual> Selection<T> for StochasticUniversalSamplingSelection {
    fn get_mating_pool(&self, initial_population: &[T], rngs: &Rngs) -> Vec<T> {
        let total_fitness: f64 = initial_population
            .iter()
            .map(|ind| selection_weight(ind.get_fitness()))
            .sum();
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

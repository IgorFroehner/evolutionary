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
        .expect("roulette selection needs at least one positive, finite fitness");

        (0..initial_population.len())
            .into_par_iter()
            .map(|slot| initial_population[wheel.sample(&mut rngs.rng(slot))].clone())
            .collect()
    }
}

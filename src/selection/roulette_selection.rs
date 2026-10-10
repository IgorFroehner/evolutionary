use rand::seq::IndexedRandom;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::fitness::selection_weight;
use crate::population::Individual;
use crate::rng::Rngs;

use super::Selection;

#[derive(Clone, Default)]
pub struct RouletteSelection;

impl<T: Individual> Selection<T> for RouletteSelection {
    fn get_mating_pool(&self, initial_population: &Vec<T>, rngs: &Rngs) -> Vec<T> {
        (0..initial_population.len())
            .into_par_iter()
            .map(|slot| {
                initial_population
                    .choose_weighted(&mut rngs.rng(slot), |individual| {
                        selection_weight(individual.get_fitness())
                    })
                    .unwrap()
                    .clone()
            })
            .collect()
    }
}

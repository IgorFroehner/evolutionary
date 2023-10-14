use rand::{seq::SliceRandom, thread_rng};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

use crate::population::Individual;

use super::Selection;

#[derive(Clone, Default)]
pub struct RouletteSelection;

impl<T: Individual> Selection<T> for RouletteSelection {
    fn get_mating_pool(&self, initial_population: &Vec<T>) -> Vec<T> {
        initial_population
            .par_iter()
            .map_init(
                || thread_rng(),
                |mut rng, _| {
                    initial_population
                        .choose_weighted(&mut rng, |individual| individual.get_fitness())
                        .unwrap()
                        .clone()
                },
            )
            .collect()
    }
}

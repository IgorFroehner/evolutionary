use rand::{seq::SliceRandom, thread_rng};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

use crate::population::{Bin, Individual, IntPerm};

use super::Selection;

#[derive(Clone, Default)]
pub struct RouletteSelection;

impl Selection<Bin> for RouletteSelection {
    fn get_mating_pool(&self, initial_population: &Vec<Bin>) -> Vec<Bin> {
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

impl Selection<IntPerm> for RouletteSelection {
    fn get_mating_pool(&self, initial_population: &Vec<IntPerm>) -> Vec<IntPerm> {
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

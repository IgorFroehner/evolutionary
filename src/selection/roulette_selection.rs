use rand::{thread_rng, seq::SliceRandom};

use crate::population::{Bin, Individual, IntPerm};

use super::Selection;

#[derive(Clone, Default)]
pub struct RouletteSelection;

impl Selection<Bin> for RouletteSelection {
    fn get_mating_pool(&self, initial_population: &Vec<Bin>) -> Vec<Bin> {
        let mut rng = thread_rng();

        initial_population
            .iter()
            .map(|_| {
                initial_population
                    .choose_weighted(&mut rng, |individual| individual.get_fitness())
                    .unwrap()
                    .clone()
            })
            .collect()
    }
}

impl Selection<IntPerm> for RouletteSelection {
    fn get_mating_pool(&self, initial_population: &Vec<IntPerm>) -> Vec<IntPerm> {
        let mut rng = thread_rng();

        initial_population
            .iter()
            .map(|_| {
                initial_population
                    .choose_weighted(&mut rng, |individual| individual.get_fitness())
                    .unwrap()
                    .clone()
            })
            .collect()
    }
}
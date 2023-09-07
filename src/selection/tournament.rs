use rand::{seq::SliceRandom, thread_rng, Rng};

use crate::population::{Bin, Individual, IntPerm};

use super::Selection;

#[derive(Clone)]
pub struct TournamentSelection {
    k: usize,
    kp: f64,
}

impl Default for TournamentSelection {
    fn default() -> Self {
        TournamentSelection { k: 2, kp: 1.0 }
    }
}

impl Selection<Bin> for TournamentSelection {
    fn get_mating_pool(&self, initial_population: &Vec<Bin>) -> Vec<Bin> {
        let mut rng = thread_rng();

        initial_population
            .iter()
            .map(|_| {
                let mut tournament = initial_population
                    .choose_multiple(&mut rng, self.k)
                    .cloned()
                    .collect::<Vec<Bin>>();

                tournament.sort_by(|a, b| b.get_fitness().partial_cmp(&a.get_fitness()).unwrap());

                if thread_rng().gen::<f64>() <= self.kp {
                    tournament[0].clone()
                } else {
                    tournament[1].clone()
                }
            })
            .collect()
    }
}

impl Selection<IntPerm> for TournamentSelection {
    fn get_mating_pool(&self, initial_population: &Vec<IntPerm>) -> Vec<IntPerm> {
        let mut rng = thread_rng();

        initial_population
            .iter()
            .map(|_| {
                let mut tournament = initial_population
                    .choose_multiple(&mut rng, self.k)
                    .cloned()
                    .collect::<Vec<IntPerm>>();

                tournament.sort_by(|a, b| b.get_fitness().partial_cmp(&a.get_fitness()).unwrap());

                if thread_rng().gen::<f64>() <= self.kp {
                    tournament[0].clone()
                } else {
                    tournament[1].clone()
                }
            })
            .collect()
    }
}

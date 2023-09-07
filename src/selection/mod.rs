use dyn_clone::DynClone;

use crate::population::Individual;

mod roulette_selection;
mod tournament;

pub use roulette_selection::*;
pub use tournament::*;

pub trait Selection<T: Individual>: 'static + DynClone + Send + Sync {
    fn get_mating_pool(&self, initial_population: &Vec<T>) -> Vec<T>;
}

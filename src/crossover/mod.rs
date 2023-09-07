use crate::population::Individual;

use dyn_clone::DynClone;

mod bin_crossover;
mod perm_crossover;

pub use bin_crossover::*;
pub use perm_crossover::*;

pub trait Crossover<T: Individual>: 'static + DynClone {
    fn crossover(&self, population: &mut Vec<T>);
}

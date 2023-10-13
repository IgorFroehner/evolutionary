use crate::population::Individual;

use dyn_clone::DynClone;

mod bin_crossover;
mod cx_crossover;
mod pmx_crossover;

pub use bin_crossover::*;
pub use cx_crossover::*;
pub use pmx_crossover::*;

pub trait Crossover<T: Individual>: 'static + DynClone + Send + Sync {
    fn crossover(&self, population: &mut Vec<T>);
}

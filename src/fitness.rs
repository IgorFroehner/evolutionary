use dyn_clone::DynClone;

use crate::population::Individual;

pub trait Fitness<T: Individual>: 'static + DynClone {
    fn calculate_fitness(&self, individual:  &T) -> f64;
}

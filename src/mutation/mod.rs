use dyn_clone::DynClone;

use crate::population::Individual;

mod bin_mutation;
mod perm_mutation;

pub use bin_mutation::*;
pub use perm_mutation::*;

pub trait Mutation<T: Individual>: 'static + DynClone + Send + Sync {
    fn mutate(&self, population: &mut Vec<T>);
}

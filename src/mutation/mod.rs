use dyn_clone::DynClone;

use crate::population::Individual;

mod bin_mutation;
mod do_nothing_mutation;
mod swap_mutation;

mod real;

pub use bin_mutation::*;
pub use do_nothing_mutation::*;
pub use swap_mutation::*;
pub use real::*;

/// Trait that defines the mutation method. You can implement your own mutation method by
/// implementing this trait.
///
/// Example:
///
/// ```
/// use evolutionary::prelude::*;
///
/// #[derive(Clone)]
/// struct AlwaysMutateMutation;
///
/// impl Mutation<Bin> for AlwaysMutateMutation {
///     fn mutate(&self, population: &mut Vec<Bin>) {
///         for individual in population.iter_mut() {
///             for i in 0..individual.chromosome.len() {
///                 individual.chromosome[i] = !individual.chromosome[i];
///             }
///         }
///     }
///  }
/// ```
pub trait Mutation<T: Individual>: 'static + DynClone + Send + Sync {
    /// Method that performs the mutation. It receives a mutable reference to the population and
    /// should modify it in place.
    fn mutate(&self, population: &mut Vec<T>);
}

use dyn_clone::DynClone;

use crate::population::Individual;
use crate::rng::Rngs;

mod real;
mod binary;
mod integer;

mod do_nothing_mutation;
mod swap_mutation;
mod random_resetting_mutation;
mod permuted;
mod scramble_mutation;

pub use real::*;
pub use binary::*;
pub use permuted::*;

pub use do_nothing_mutation::*;
pub use swap_mutation::*;
pub use random_resetting_mutation::*;
pub use scramble_mutation::*;

/// # Mutation Trait
///
/// Trait that defines the mutation method. You can implement your own mutation method by
/// implementing this trait.
///
/// Draw every random number from `rngs`, asking it for one generator per individual
/// (`rngs.rng(index)`), so that a seeded run is reproducible even when the mutation runs in
/// parallel. See [`Rngs`].
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
///     fn mutate(&self, population: &mut Vec<Bin>, _rngs: &Rngs) {
///         for individual in population.iter_mut() {
///             for i in 0..individual.get_chromosome().len() {
///                 individual.set_gene(i, !individual.get_gene(i));
///             }
///         }
///     }
///  }
/// ```
pub trait Mutation<T: Individual>: 'static + DynClone + Send + Sync {
    /// Method that performs the mutation. It receives a mutable reference to the population and
    /// should modify it in place, drawing its random numbers from `rngs`.
    fn mutate(&self, population: &mut Vec<T>, rngs: &Rngs);
}

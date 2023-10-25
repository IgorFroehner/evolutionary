use crate::population::Individual;

use dyn_clone::DynClone;

mod binary;
mod permuted;

mod do_nothing_crossover;
mod uniform_crossover;

pub use binary::*;
pub use permuted::*;

pub use do_nothing_crossover::*;
pub use uniform_crossover::*;

/// Trait that defines the crossover method. You can implement your own crossover method by
/// implementing this trait.
///
/// # Example
///
/// ```
/// use evolutionary::prelude::*;
///
/// #[derive(Clone)]
/// struct DoNothingCrossover;
///
/// impl<T: Individual> Crossover<T> for DoNothingCrossover {
///     fn crossover(&self, population: &mut Vec<T>) {
///         // Do nothing
///     }
/// }
/// ```
pub trait Crossover<T: Individual>: 'static + DynClone + Send + Sync {
    /// Method that performs the crossover. It receives a mutable reference to the population and
    /// should modify it in place.
    fn crossover(&self, population: &mut Vec<T>);
}

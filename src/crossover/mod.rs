use crate::population::Individual;

use dyn_clone::DynClone;

mod bin_crossover;
mod cx_crossover;
mod pmx_crossover;

pub use bin_crossover::*;
pub use cx_crossover::*;
pub use pmx_crossover::*;

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

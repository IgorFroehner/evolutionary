use crate::population::Individual;
use crate::rng::Rngs;

use dyn_clone::DynClone;

mod binary;
mod permuted;
mod real;

mod do_nothing_crossover;
mod uniform_crossover;
mod integer;
mod n_points_crossover;

pub use permuted::*;
pub use real::*;

pub use do_nothing_crossover::*;
pub use uniform_crossover::*;
pub use n_points_crossover::*;

/// Trait that defines the crossover method. You can implement your own crossover method by
/// implementing this trait.
///
/// Draw every random number from `rngs`, asking it for one generator per pair of parents
/// (`rngs.rng(pair)`), so that a seeded run is reproducible even when the crossover runs in
/// parallel. See [`Rngs`].
///
/// The built-in crossovers pair the population two by two. When its size is odd, the last
/// individual has no partner and is left unchanged.
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
///     fn crossover(&self, population: &mut Vec<T>, _rngs: &Rngs) {
///         // Do nothing
///     }
/// }
/// ```
pub trait Crossover<T: Individual>: 'static + DynClone + Send + Sync {
    /// Method that performs the crossover. It receives a mutable reference to the population and
    /// should modify it in place, drawing its random numbers from `rngs`.
    fn crossover(&self, population: &mut Vec<T>, rngs: &Rngs);
}

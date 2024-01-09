use dyn_clone::DynClone;

use crate::population::Individual;

/// Trait that defines the fitness function and that will be used by the evolutionary algorithm.
/// You need to implement your own fitness function for each problem you're trying to solve. The
/// fitness function is directly related to the type of individual you're using and how your
/// mapping the answers.
///
/// Example:
///
/// ```
/// use evolutionary::prelude::*;
///
/// #[derive(Clone)]
/// struct SumFitness;
///
/// impl Fitness<Bin> for SumFitness {
///     fn calculate_fitness(&self, individual: &Bin) -> f64 {
///         let mut sum = 0.0;
///
///         for i in 0..individual.get_chromosome().len() {
///             if individual.get_chromosome()[i] {
///                 sum += 1.0;
///             }
///         }
///
///         sum
///     }
/// }
/// ```
pub trait Fitness<T: Individual>: 'static + DynClone + Send + Sync {
    /// Method that calculates the fitness of an individual.
    fn calculate_fitness(&self, individual: &T) -> f64;
}

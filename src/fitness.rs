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
///             if individual.get_gene(i) {
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

/// Total order over fitness values where `NaN` ranks below every number, so an individual whose
/// fitness could not be computed is always treated as the worst one instead of panicking a
/// comparison.
pub(crate) fn fitness_cmp(a: f64, b: f64) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        (false, false) => a.total_cmp(&b),
    }
}

/// Weight of an individual for fitness-proportional selection. `NaN` gets no chance of being
/// picked, the same as the worst fitness.
pub(crate) fn selection_weight(fitness: f64) -> f64 {
    if fitness.is_nan() {
        0.0
    } else {
        fitness
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering;

    #[test]
    fn nan_ranks_below_every_number() {
        assert_eq!(fitness_cmp(f64::NAN, f64::NEG_INFINITY), Ordering::Less);
        assert_eq!(fitness_cmp(f64::NEG_INFINITY, f64::NAN), Ordering::Greater);
        assert_eq!(fitness_cmp(f64::NAN, f64::NAN), Ordering::Equal);
        assert_eq!(fitness_cmp(-f64::NAN, 0.0), Ordering::Less);
    }

    #[test]
    fn numbers_keep_their_order() {
        assert_eq!(fitness_cmp(1.0, 2.0), Ordering::Less);
        assert_eq!(fitness_cmp(2.0, 1.0), Ordering::Greater);
        assert_eq!(fitness_cmp(1.0, 1.0), Ordering::Equal);
    }

    #[test]
    fn nan_has_no_selection_weight() {
        assert_eq!(selection_weight(f64::NAN), 0.0);
        assert_eq!(selection_weight(3.0), 3.0);
    }
}

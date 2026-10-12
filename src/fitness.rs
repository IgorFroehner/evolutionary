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

/// Reads the fitness of a saved individual, taking `null` as `NaN`.
///
/// An individual that was never evaluated has a `NaN` fitness, and JSON has no `NaN`:
/// `serde_json` writes it as `null`, which a plain `f64` refuses to read back. Formats that
/// are not human-readable (bincode and the like) store the `f64` as it is, so they read it as
/// one too, and the format of every saved individual stays the same.
#[cfg(feature = "serde")]
pub(crate) fn deserialize_fitness<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct FitnessVisitor;

    impl<'de> serde::de::Visitor<'de> for FitnessVisitor {
        type Value = f64;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a fitness: a number, or null for one never evaluated")
        }

        fn visit_f64<E>(self, value: f64) -> Result<f64, E> {
            Ok(value)
        }

        fn visit_i64<E>(self, value: i64) -> Result<f64, E> {
            Ok(value as f64)
        }

        fn visit_u64<E>(self, value: u64) -> Result<f64, E> {
            Ok(value as f64)
        }

        fn visit_unit<E>(self) -> Result<f64, E> {
            Ok(f64::NAN)
        }

        fn visit_none<E>(self) -> Result<f64, E> {
            Ok(f64::NAN)
        }

        fn visit_some<D>(self, deserializer: D) -> Result<f64, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            deserializer.deserialize_any(self)
        }
    }

    if deserializer.is_human_readable() {
        deserializer.deserialize_any(FitnessVisitor)
    } else {
        deserializer.deserialize_f64(FitnessVisitor)
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

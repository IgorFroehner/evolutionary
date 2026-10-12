use std::fmt::Debug;

use rand::Rng;

mod bin;
mod int;
mod perm;
mod real;

pub use bin::*;
pub use int::*;
pub use perm::*;
pub use real::*;

/// The name of a built-in encoding, as a config file gives it (`RawConfig::gene_cod` with the
/// `config` feature), to pick the individual type to build.
///
/// An evolution does not need it: the individual type of the
/// [`EvolutionBuilder`](crate::EvolutionBuilder) is the encoding.
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde",
    derive(serde_derive::Serialize, serde_derive::Deserialize)
)]
pub enum GeneCod {
    Int,
    IntPerm,
    Bin,
    Real,
}

/// Trait that must be implemented by a struct to be considered a individual.
///
/// [`generate_member`](Self::generate_member) draws the random genes of a new individual from
/// `rng`, so that a seeded run starts from the same population every time.
///
/// # Fitness of an individual that was never evaluated
///
/// [`get_fitness`](Self::get_fitness) returns what the last
/// [`set_fitness`](Self::set_fitness) stored. An evolution evaluates every individual before
/// reading its fitness, so there it is always the result of the [`Fitness`](crate::Fitness).
///
/// Before that, the built-in individuals have a fitness of `NaN`: `generate_member` and the
/// constructors ([`Bin::new`], [`Int::new`], [`IntPerm::new`], [`Real::new`]) start there, so a
/// fitness that was never computed can't be mistaken for a real `0.0`. Check it with
/// `f64::is_nan`. Every comparison in the crate ranks `NaN` below every number, the same as a
/// fitness function that returns `NaN`, so such an individual never wins a selection or
/// becomes the best. Implementations of your own should start at `NaN` too.
///
/// # Example
///
/// ```
/// use evolutionary::prelude::*;
///
/// #[derive(Clone)]
/// struct MyIndividual {
///     chromosome: Vec<u64>,
///     fitness: f64,
/// }
///
/// impl Individual for MyIndividual {
///     type Gene = u64;
///     type RangeType = ();
///
///     fn generate_member<R: Rng + ?Sized>(dimension: u32, _: &Self::RangeType, rng: &mut R) -> Self {
///         Self {
///             chromosome: (0..dimension).map(|_| rng.random()).collect(),
///             // Not evaluated yet.
///             fitness: f64::NAN,
///         }
///     }
///
///     fn get_chromosome(&self) -> &[Self::Gene] {
///         &self.chromosome
///     }
///
///     fn get_mut_chromosome(&mut self) -> &mut [Self::Gene] {
///         &mut self.chromosome
///     }
///
///     fn set_gene(&mut self, index: usize, value: Self::Gene) {
///         self.chromosome[index] = value;
///     }
///
///     fn get_gene(&self, index: usize) -> Self::Gene {
///        self.chromosome[index]
///     }
///
///     fn set_fitness(&mut self, fitness: f64) {
///         self.fitness = fitness;
///     }
///
///     fn get_fitness(&self) -> f64 {
///         self.fitness
///     }
/// }
/// ```
pub trait Individual: 'static + Clone + Send + Sync {
    type Gene: Debug + Send + Sync + Copy;
    type RangeType: Send + Sync + Clone;

    /// Creates a random individual of `dimension` genes within `range`, drawing from `rng`.
    fn generate_member<R: Rng + ?Sized>(
        dimension: u32,
        range: &Self::RangeType,
        rng: &mut R,
    ) -> Self;
    /// The genes of the individual.
    fn get_chromosome(&self) -> &[Self::Gene];
    /// The genes of the individual, to change in place. Operators keep the number of genes, so
    /// this is a slice: an individual whose length can change is not one the built-in
    /// operators handle.
    fn get_mut_chromosome(&mut self) -> &mut [Self::Gene];
    fn set_gene(&mut self, index: usize, value: Self::Gene);
    fn get_gene(&self, index: usize) -> Self::Gene;
    /// Stores the fitness of the individual, as computed by the [`Fitness`](crate::Fitness).
    fn set_fitness(&mut self, fitness: f64);
    /// The fitness stored by [`set_fitness`](Self::set_fitness), or `NaN` if it was never
    /// evaluated (see [the trait docs](Self#fitness-of-an-individual-that-was-never-evaluated)).
    fn get_fitness(&self) -> f64;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rngs;

    #[test]
    fn int_genes_cover_the_whole_range_bounds_included() {
        let individual = Int::generate_member(1000, &(-2, 2), &mut Rngs::new(0).rng(0));
        let genes = individual.get_chromosome();

        assert_eq!(genes.len(), 1000);
        assert_eq!(individual.range, (-2, 2));
        assert!(genes.iter().all(|gene| (-2..=2).contains(gene)));
        assert!((-2..=2).all(|value| genes.contains(&value)));
    }

    #[test]
    fn real_genes_stay_within_the_range() {
        let individual = Real::generate_member(1000, &(-0.5, 2.0), &mut Rngs::new(0).rng(0));
        let genes = individual.get_chromosome();

        assert_eq!(genes.len(), 1000);
        assert_eq!(individual.range, (-0.5, 2.0));
        assert!(genes.iter().all(|gene| (-0.5..=2.0).contains(gene)));
    }

    #[test]
    fn int_perm_is_a_shuffled_permutation() {
        for dimension in [1, 2, 50] {
            let individual = IntPerm::generate_member(dimension, &(), &mut Rngs::new(0).rng(0));

            assert!(is_permutation(individual.get_chromosome()));
        }

        let individual = IntPerm::generate_member(50, &(), &mut Rngs::new(0).rng(0));
        assert_ne!(*individual.get_chromosome(), (1..=50).collect::<Vec<i64>>());
    }

    #[test]
    fn bin_draws_both_values() {
        let individual = Bin::generate_member(100, &(), &mut Rngs::new(0).rng(0));
        let genes = individual.get_chromosome();

        assert_eq!(genes.len(), 100);
        assert!(genes.contains(&true) && genes.contains(&false));
    }

    #[test]
    fn generated_individuals_start_without_a_fitness() {
        let rng = &mut Rngs::new(0).rng(0);

        assert!(Bin::generate_member(4, &(), rng).get_fitness().is_nan());
        assert!(Int::generate_member(4, &(0, 1), rng).get_fitness().is_nan());
        assert!(IntPerm::generate_member(4, &(), rng).get_fitness().is_nan());
        assert!(Real::generate_member(4, &(0.0, 1.0), rng)
            .get_fitness()
            .is_nan());
    }

    #[test]
    fn built_individuals_start_without_a_fitness() {
        assert!(Bin::new(vec![true]).get_fitness().is_nan());
        assert!(Int::new(vec![0], (0, 1)).get_fitness().is_nan());
        assert!(IntPerm::new(vec![1]).get_fitness().is_nan());
        assert!(Real::new(vec![0.0], (0.0, 1.0)).get_fitness().is_nan());
    }
}

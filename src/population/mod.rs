use std::fmt::Debug;

use serde_derive::Deserialize;

mod bin;
mod int;
mod perm;
mod real;

pub use bin::*;
pub use int::*;
pub use perm::*;
pub use real::*;

#[derive(Debug, Deserialize, Clone)]
pub enum GeneCod {
    Int,
    IntPerm,
    Bin,
    Real,
}

/// Trait that must be implemented by a struct to be considered a individual.
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
///     fn generate_member(dimension: u32, _: &Self::RangeType) -> Self {
///         Self {
///             chromosome: vec![0; dimension as usize],
///             fitness: 0.0,
///         }
///     }
///
///     fn get_chromosome(&self) -> &Vec<Self::Gene> {
///         &self.chromosome
///     }
///
///     fn get_mut_chromosome(&mut self) -> &mut Vec<Self::Gene> {
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

    fn generate_member(dimension: u32, b: &Self::RangeType) -> Self;
    fn get_chromosome(&self) -> &Vec<Self::Gene>;
    fn get_mut_chromosome(&mut self) -> &mut Vec<Self::Gene>;
    fn set_gene(&mut self, index: usize, value: Self::Gene);
    fn get_gene(&self, index: usize) -> Self::Gene;
    fn set_fitness(&mut self, fitness: f64);
    fn get_fitness(&self) -> f64;
}

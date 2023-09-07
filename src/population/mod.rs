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

#[derive(Debug, Deserialize)]
pub enum GeneCod {
    Int,
    IntPerm,
    Bin,
    Real,
}

pub trait Individual: 'static + Clone + Send + Sync {
    type Gene: Debug + Send + Sync;
    type RangeType: Send + Sync;

    fn generate_member(dimension: u32, b: &Self::RangeType) -> Self;
    fn get_vec(&self) -> &Vec<Self::Gene>;
    fn set_fitness(&mut self, fitness: f64);
    fn get_fitness(&self) -> f64;
}

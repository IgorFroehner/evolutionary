//! # Evolutionary
//!
//! A fully extensible Rust framework for using paralyzed genetic algorithms to solve problems.
//! Currently, it supports coding in Binary, Real, Permuted Integers, Integers and any other coding you may want to implement. It also has built in implementation of the following genetic operators:
//!
//! - Selection:
//!     - Roulette Wheel
//!     - Tournament
//! - Crossover:
//!     - Multiple Point Crossover (MPX) (Binary)
//!     - Cycle Crossover (CX) (Permuted)
//!     - Partially Mapped Crossover (PMX) (Permuted)
//! - Mutation:
//!     - Bit Flip (Binary)
//!     - Swap (Permuted)
//!
//! And you can code your own selection, crossover or mutation as well and use it on the EvolutionBuilder.


pub mod population;
pub mod gene_cod;
pub mod config_read;
pub mod fitness;
pub mod utils;
pub mod selection;
pub mod crossover;
pub mod mutation;
pub mod plot_evolution;
pub mod coding;
pub mod experiment_runner;

mod evolution;
mod evolution_builder;

pub use evolution_builder::EvolutionBuilder;
pub use evolution::Evolution;

// create a prelude for the library
pub mod prelude {
    pub use crate::population::{GeneCod, Bin, IntPerm};
    pub use crate::config_read::{RawConfig, read_config};
    pub use crate::EvolutionBuilder;
    pub use crate::Evolution;
    pub use crate::fitness::Fitness;
    pub use crate::utils::{convert_bin, within_range};
    pub use crate::selection::{Selection, RouletteSelection, TournamentSelection};
    pub use crate::crossover::{Crossover, BinCrossover, PMXCrossover};
    pub use crate::mutation::{Mutation, BinMutation, PermMutation};
    pub use crate::coding::Coding;
    pub use crate::experiment_runner::ExperimentRunner;
}

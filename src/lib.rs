//! # Evolutionary
//!
//! A fully extensible Rust framework for using parallelized genetic algorithms to solve problems.
//!
//! Currently, it supports coding in [`Bin`](struct@population::Bin), [`Real`](struct@population::Real),
//! [`Permuted Integers`](struct@population::IntPerm), [`Integers`](struct@population::Int) and any
//! other coding you may want to implement. It also has built in implementation of the following genetic
//! operators:
//!
//! - [`Selection`]
//! - [`Crossover`]
//! - [`Mutation`]
//!
//! You can code your own selection, crossover or mutation implementing the traits and passing them
//! to the `EvolutionBuilder`.
//!
//! ## Example:
//!
//! ```rust
//! use evolutionary::prelude::*;
//!
//! // First you'll need to code your Fitness function:
//! #[derive(Clone)]
//! pub struct MaxFitness;
//!
//! // To do this you need to implement the Fitness trait.
//! impl Fitness<Bin> for MaxFitness {
//!     fn calculate_fitness(&self, individual: &Bin) -> f64 {
//!         let mut sum = 0.;
//!
//!         for i in 0..individual.get_chromosome().len() {
//!             if individual.get_gene(i) {
//!                 sum += 1.;
//!             }
//!         }
//!
//!         sum
//!     }
//! }
//!
//! // Then you will be able to build a evolution object using the `EvolutionBuilder`
//! // and setting all the required parameters:
//! fn main() {
//!     let mut evolution = EvolutionBuilder::new(30, 10, ())
//!         .with_fitness(MaxFitness)
//!         .with_selection(TournamentSelection::default())
//!         .with_crossover(NPointsCrossover::default())
//!         .with_mutation(BitFlipMutation::default())
//!         .with_title("Max")
//!         .with_stop_condition(|report| report.generation >= 100)
//!         .build().unwrap();
//!
//!     evolution.run();
//!
//!     assert_eq!(*evolution.current_best().get_chromosome(), vec![true; 10]);
//!     assert_eq!(evolution.current_best_fitness(), 10.0);
//! }
//! ```
//!
//! Find this and other examples in the [examples folder](./examples).
//!
//! ## Features
//!
//! None is on by default, so embedding the library only compiles the genetic algorithm.
//!
//! - `serde`: `Serialize` and `Deserialize` for the built-in individuals and [`Checkpoint`],
//!   to save and load populations and runs.
//! - `config`: reads the evolution parameters from a TOML file, with `read_config`.
//! - `plot`: saves fitness charts with `Evolution::plot_chart` and
//!   `ExperimentRunner::save_chart`, with `plotters`.

#[cfg(feature = "config")]
pub mod config_read;
pub mod experiment_runner;
pub mod metrics;

mod checkpoint;
mod crossover;
mod evolution;
mod evolution_builder;
mod fitness;
mod mutation;
#[cfg(feature = "plot")]
pub mod plotting;
mod population;
mod report;
mod rng;
mod selection;
pub mod utils;

pub use checkpoint::Checkpoint;
pub use crossover::Crossover;
pub use evolution::Evolution;
pub use evolution_builder::{BuildError, EvolutionBuilder};
pub use fitness::{Fitness, Minimize};
pub use mutation::Mutation;
pub use population::Individual;
pub use report::GenerationReport;
pub use rng::{Rngs, TaskRng};
pub use selection::Selection;

/// The `rand` this crate is built with. Use its [`Rng`](rand::Rng) trait to draw numbers from a
/// [`TaskRng`], or when implementing [`Individual::generate_member`].
pub use rand;

/// The `rayon` this crate runs its operators and fitness on. Write custom operators that run in
/// parallel with `use evolutionary::rayon::prelude::*;`, so they need no `rayon` dependency of
/// their own, at a version that has to match.
///
/// It is not in [`prelude`], so that its traits only come into scope where they are asked for.
pub use rayon;

pub mod prelude {
    #[cfg(feature = "config")]
    pub use crate::config_read::{read_config, ConfigError, RawConfig};
    pub use crate::crossover::*;
    pub use crate::experiment_runner::ExperimentRunner;
    pub use crate::fitness::{Fitness, Minimize};
    pub use crate::mutation::*;
    pub use crate::population::{Bin, GeneCod, Int, IntPerm, Real};
    pub use crate::selection::*;
    pub use crate::utils::{convert_bin, within_range};
    pub use crate::BuildError;
    pub use crate::Checkpoint;
    pub use crate::Evolution;
    pub use crate::EvolutionBuilder;
    pub use crate::GenerationReport;
    pub use crate::Individual;
    pub use crate::{Rngs, TaskRng};
    pub use rand::Rng;
}

//! # Evolutionary
//!
//! A fully extensible Rust framework for using paralyzed genetic algorithms to solve problems.
//!
//! Currently, it supports coding in `Binary`, `Real`, `Permuted Integers`, `Integers` and any other coding you may want to implement.
//! It also has built in implementation of the following genetic operators:
//!
//! - Selection:
//!   - [Roulette Wheel](./src/selection/roulette_selection.rs)
//!   - [Tournament](./src/selection/tournament_selection)
//! - Crossover:
//!   - [Multiple Point Crossover (MPX) (Binary)](./src/crossover/bin_crossover.rs)
//!   - [Partially Mapped Crossover (PMX) (Permuted)](./src/crossover/pmx_crossover.rs)
//!   - [Cycle Crossover (CX) (Permuted)](./src/crossover/cx_crossover.rs)
//! - Mutation:
//!   - [Bit Flip (Binary)](./src/mutation/bin_mutation.rs)
//!   - [Swap (Permuted)](./src/mutation/perm_mutation.rs)
//!
//! As well you can code your own selection, crossover or mutation and use it on the `EvolutionBuilder`.
//!
//! ## Example:
//!
//! ```rust
//! use evolutionary::prelude::*;
//!
//! // First you'll need to code your Fitness function:
//! fn f(individual: &Bin) -> f64 {
//!     let mut sum = 0.;
//!
//!     for i in 0..individual.chromosome.len() {
//!         if individual.chromosome[i] {
//!             sum += 1.;
//!         }
//!     }
//!
//!     sum
//! }
//!
//! #[derive(Clone)]
//! pub struct MaxFitness;
//!
//! // To do this you need to implement the Fitness trait.
//! impl Fitness<Bin> for MaxFitness {
//!     fn calculate_fitness(&self, individual: &Bin) -> f64 {
//!         f(individual)
//!     }
//! }
//!
//! // You'll need to create a Coding for you individual,
//! // so you can get the response at the end:
//! #[derive(Clone)]
//! struct MaxCoding;
//!
//! impl Coding<Bin> for MaxCoding {
//!     type Output = f64;
//!
//!     fn decode(&self, individual: &Bin) -> Self::Output {
//!         f(individual)
//!     }
//! }
//!
//! // Then you will be able to build a evolution object using the `EvolutionBuiler`
//! // and setting all the required parameters:
//! fn main() {
//!     let mut evolution = EvolutionBuilder::new(30, 10, GeneCod::Bin, ())
//!         .with_fitness(MaxFitness)
//!         .with_selection(TournamentSelection::default())
//!         .with_crossover(NPointsCrossover::default())
//!         .with_mutation(BinMutation::default())
//!         .with_title("Max".to_string())
//!         .with_stop_condition(move |best_fitness, _, _| best_fitness == 10.0)
//!         .with_coding(MaxCoding)
//!         .build().unwrap();
//!
//!     evolution.run();
//!
//!     assert_eq!(evolution.current_best().chromosome, vec![true; 10]);
//!     assert_eq!(evolution.current_best_fitness(), 10.0);
//! }
//! ```
//!
//! Find this and other examples in the [examples folder](./examples).

pub mod config_read;
pub mod experiment_runner;
pub mod plot_evolution;

mod coding;
mod crossover;
mod evolution;
mod evolution_builder;
mod fitness;
mod mutation;
mod population;
mod selection;
pub mod utils;

pub use coding::Coding;
pub use crossover::Crossover;
pub use evolution::Evolution;
pub use evolution_builder::EvolutionBuilder;
pub use fitness::Fitness;
pub use mutation::Mutation;
pub use population::Individual;
pub use selection::Selection;

pub mod prelude {
    pub use crate::coding::Coding;
    pub use crate::config_read::{read_config, RawConfig};
    pub use crate::crossover::{NPointsCrossover, CXCrossover, Crossover, PMXCrossover, UniformCrossover};
    pub use crate::experiment_runner::ExperimentRunner;
    pub use crate::fitness::Fitness;
    pub use crate::mutation::{BinMutation, Mutation, PermMutation};
    pub use crate::population::{Bin, GeneCod, IntPerm};
    pub use crate::selection::{RouletteSelection, Selection, TournamentSelection};
    pub use crate::utils::{convert_bin, within_range};
    pub use crate::Evolution;
    pub use crate::EvolutionBuilder;
    pub use crate::Individual;
}

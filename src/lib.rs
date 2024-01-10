//! # Evolutionary
//!
//! A fully extensible Rust framework for using paralyzed genetic algorithms to solve problems.
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
//! fn f(individual: &Bin) -> f64 {
//!     let mut sum = 0.;
//!
//!     for i in 0..individual.get_chromosome().len() {
//!         if individual.get_chromosome()[i] {
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
//! // Then you will be able to build a evolution object using the `EvolutionBuiler`
//! // and setting all the required parameters:
//! fn main() {
//!     let mut evolution = EvolutionBuilder::new(30, 10, GeneCod::Bin, ())
//!         .with_fitness(MaxFitness)
//!         .with_selection(TournamentSelection::default())
//!         .with_crossover(NPointsCrossover::default())
//!         .with_mutation(BitSwapMutation::default())
//!         .with_title("Max".to_string())
//!         .with_stop_condition(move |best_fitness, _, _| best_fitness == 10.0)
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

pub mod config_read;
pub mod experiment_runner;
pub mod metrics;

mod crossover;
mod evolution;
mod evolution_builder;
mod fitness;
mod mutation;
mod population;
mod selection;
pub mod utils;

pub use crossover::Crossover;
pub use evolution::Evolution;
pub use evolution_builder::EvolutionBuilder;
pub use fitness::Fitness;
pub use mutation::Mutation;
pub use population::Individual;
pub use selection::Selection;

pub mod prelude {
    pub use crate::config_read::{read_config, RawConfig};
    pub use crate::crossover::*;
    pub use crate::experiment_runner::ExperimentRunner;
    pub use crate::fitness::Fitness;
    pub use crate::mutation::*;
    pub use crate::population::{Bin, GeneCod, Int, IntPerm, Real};
    pub use crate::selection::*;
    pub use crate::utils::{convert_bin, within_range};
    pub use crate::Evolution;
    pub use crate::EvolutionBuilder;
    pub use crate::Individual;
}

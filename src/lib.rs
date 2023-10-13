pub mod population;
pub mod gene_cod;
pub mod config_read;
pub mod evolution_builder;
pub mod fitness;
pub mod evolution;
pub mod utils;
pub mod selection;
pub mod crossover;
pub mod mutation;
pub mod plot_evolution;
pub mod coding;
pub mod experiment_runner;

// create a prelude for the library
pub mod prelude {
    pub use crate::population::{GeneCod, Bin, IntPerm};
    pub use crate::config_read::{RawConfig, read_config};
    pub use crate::evolution_builder::EvolutionBuilder;
    pub use crate::fitness::Fitness;
    pub use crate::evolution::Evolution;
    pub use crate::utils::{convert_bin, within_range};
    pub use crate::selection::{Selection, RouletteSelection, TournamentSelection};
    pub use crate::crossover::{Crossover, BinCrossover, PMXCrossover};
    pub use crate::mutation::{Mutation, BinMutation, PermMutation};
    pub use crate::coding::Coding;
    pub use crate::experiment_runner::ExperimentRunner;
}

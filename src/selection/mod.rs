use dyn_clone::DynClone;

use crate::population::Individual;

mod do_nothing_selection;
mod roulette_selection;
mod tournament_selection;
mod rank_selection;
mod stochastic_universal_sampling_selection;

pub use do_nothing_selection::DoNothingSelection;
pub use roulette_selection::RouletteSelection;
pub use tournament_selection::TournamentSelection;
pub use rank_selection::RankSelection;
pub use stochastic_universal_sampling_selection::StochasticUniversalSamplingSelection;

/// Trait that defines the selection method. You can implement your own selection method by
/// implementing this trait.
///
/// Example:
///
/// ```
/// use evolutionary::prelude::*;
///
/// #[derive(Clone)]
/// struct DoNothingSelection;
///
/// impl<T: Individual> Selection<T> for DoNothingSelection {
///    fn get_mating_pool(&self, initial_population: &Vec<T>) -> Vec<T> {
///       initial_population.clone()
///   }
/// }
/// ```
pub trait Selection<T: Individual>: 'static + DynClone + Send + Sync {
    /// Method that performs the selection. It receives a reference to the initial population and
    /// should return a new population.
    fn get_mating_pool(&self, initial_population: &Vec<T>) -> Vec<T>;
}

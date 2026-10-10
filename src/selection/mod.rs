use dyn_clone::DynClone;

use crate::population::Individual;
use crate::rng::Rngs;

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
/// Draw every random number from `rngs`, asking it for one generator per slot of the mating pool
/// (`rngs.rng(slot)`), so that a seeded run is reproducible even when the selection runs in
/// parallel. See [`Rngs`].
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
///    fn get_mating_pool(&self, initial_population: &[T], _rngs: &Rngs) -> Vec<T> {
///       initial_population.to_vec()
///   }
/// }
/// ```
pub trait Selection<T: Individual>: 'static + DynClone + Send + Sync {
    /// Method that performs the selection. It receives a reference to the initial population and
    /// should return a new population, drawing its random numbers from `rngs`.
    fn get_mating_pool(&self, initial_population: &[T], rngs: &Rngs) -> Vec<T>;
}

use dyn_clone::DynClone;

use crate::population::Individual;
use crate::rng::Rngs;

mod do_nothing_selection;
mod fitness_scaling;
mod rank_selection;
mod roulette_selection;
mod stochastic_universal_sampling_selection;
mod tournament_selection;

pub use do_nothing_selection::DoNothingSelection;
pub use fitness_scaling::{FitnessScaling, ScaledSelection};
pub use rank_selection::RankSelection;
pub use roulette_selection::RouletteSelection;
pub use stochastic_universal_sampling_selection::StochasticUniversalSamplingSelection;
pub use tournament_selection::TournamentSelection;

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

#[cfg(test)]
pub(crate) mod tests {
    use crate::population::{Individual, Real};

    /// One individual per fitness, whose only gene is its fitness.
    pub(crate) fn population(fitnesses: &[f64]) -> Vec<Real> {
        fitnesses
            .iter()
            .map(|&fitness| {
                let mut individual = Real::new(vec![fitness], (0.0, 1.0));
                individual.set_fitness(fitness);
                individual
            })
            .collect()
    }

    /// How many individuals of `pool` have a fitness for which `matches` holds.
    pub(crate) fn picked(pool: &[Real], matches: impl Fn(f64) -> bool) -> usize {
        pool.iter()
            .filter(|individual| matches(individual.get_fitness()))
            .count()
    }
}

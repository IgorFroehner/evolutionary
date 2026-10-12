use crate::population::Individual;

/// The state of an evolution after a generation, handed to the stop condition and to the
/// [`with_on_generation`](crate::EvolutionBuilder::with_on_generation) callback, and returned
/// by [`Evolution::report`](crate::Evolution::report).
///
/// # Example
///
/// ```
/// # use evolutionary::prelude::*;
/// # #[derive(Clone)]
/// # struct CountOnes;
/// # impl Fitness<Bin> for CountOnes {
/// #     fn calculate_fitness(&self, individual: &Bin) -> f64 {
/// #         individual.get_chromosome().iter().filter(|&&gene| gene).count() as f64
/// #     }
/// # }
/// let mut evolution = EvolutionBuilder::new(30, 10, ())
///     .with_fitness(CountOnes)
///     .with_selection(TournamentSelection::default())
///     .with_crossover(NPointsCrossover::default())
///     .with_mutation(BitFlipMutation::default())
///     .with_on_generation(|report| {
///         println!("{}: {} ({})", report.generation, report.best_fitness, report.average_fitness)
///     })
///     .with_stop_condition(|report| {
///         report.best_fitness == 10.0 || report.generations_without_improvement >= 50
///     })
///     .build()
///     .unwrap();
///
/// evolution.run();
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub struct GenerationReport<'a, T: Individual> {
    /// The generation: 0 for the initial population, then one more after each
    /// [`next`](crate::Evolution::next).
    pub generation: u32,
    /// The best individual of the current population.
    pub best: &'a T,
    /// The fitness of [`best`](Self::best).
    pub best_fitness: f64,
    /// The average fitness of the current population, ignoring `NaN`s.
    pub average_fitness: f64,
    /// The best individual seen since the run started. It is the same as `best` with elitism,
    /// and can be better without it, when the best individual of a generation is lost.
    pub best_ever: &'a T,
    /// Generations since `best_ever` last improved. 0 in the generation it improved.
    pub generations_without_improvement: u32,
}

impl<T: Individual> Clone for GenerationReport<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Individual> Copy for GenerationReport<'_, T> {}

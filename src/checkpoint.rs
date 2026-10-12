/// The state of a run after a generation, to stop it and resume it later exactly where it was.
///
/// [`Evolution::checkpoint`](crate::Evolution::checkpoint) takes one, and
/// [`EvolutionBuilder::with_checkpoint`](crate::EvolutionBuilder::with_checkpoint) resumes from
/// it: the resumed run's next generation is `generation + 1`, drawn from the same random
/// numbers, so it continues as if it had never stopped. Resuming does not compute any fitness
/// again: the population keeps the fitnesses it was saved with.
///
/// The timings of [`Metrics`](crate::metrics::Metrics) are not part of it: a resumed run times
/// only what it runs itself.
///
/// With the `serde` feature it implements `Serialize` and `Deserialize` when the individual
/// does, so it can be saved in any format serde supports. To resume exactly, the format must
/// give back every `f64` bit for bit: with `serde_json`, turn on its `float_roundtrip` feature,
/// without which the last bit of some numbers can change on the way back, and the resumed run
/// then drifts away from the one that was saved. A fitness that is `NaN` is saved to JSON as
/// `null` and loads back as `NaN`; an infinite one does too, so it loads back as `NaN`.
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
/// let builder = || {
///     EvolutionBuilder::new(20, 16, ())
///         .with_fitness(CountOnes)
///         .with_selection(TournamentSelection::default())
///         .with_crossover(NPointsCrossover::default())
///         .with_mutation(BitFlipMutation::default())
///         .with_stop_condition(|report| report.generation >= 30)
///         .with_seed(1)
/// };
///
/// // Run 10 generations, and save where the run is.
/// let mut evolution = builder().build()?;
/// evolution.start();
/// for _ in 0..10 {
///     evolution.next();
/// }
/// let checkpoint = evolution.checkpoint();
/// assert_eq!(checkpoint.generation, 10);
///
/// // Later, maybe in another process: resume, and run until the stop condition.
/// let mut resumed = builder().with_checkpoint(checkpoint).build()?;
/// resumed.run();
///
/// // The same as a run that never stopped.
/// let mut uninterrupted = builder().build()?;
/// uninterrupted.run();
/// assert_eq!(resumed.report().generation, 30);
/// assert_eq!(resumed.metrics.best_fitnesses, uninterrupted.metrics.best_fitnesses);
/// assert_eq!(resumed.best_ever().get_chromosome(), uninterrupted.best_ever().get_chromosome());
/// # Ok::<(), BuildError>(())
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde",
    derive(serde_derive::Serialize, serde_derive::Deserialize)
)]
#[non_exhaustive]
pub struct Checkpoint<T> {
    /// The seed of the run, which a resumed run keeps: see
    /// [`EvolutionBuilder::with_checkpoint`](crate::EvolutionBuilder::with_checkpoint).
    pub seed: u64,
    /// The generation the checkpoint was taken after, as
    /// [`GenerationReport::generation`](crate::GenerationReport::generation): 0 for the initial
    /// population.
    pub generation: u32,
    /// The population in its current order (as [`Evolution::population`](crate::Evolution::population)
    /// borrows it), each individual with its fitness.
    pub population: Vec<T>,
    /// [`Evolution::best_ever`](crate::Evolution::best_ever), with its fitness.
    pub best_ever: T,
    /// The best fitness of every generation, from 0 to `generation`.
    #[cfg_attr(
        feature = "serde",
        serde(deserialize_with = "crate::fitness::deserialize_fitnesses")
    )]
    pub best_fitnesses: Vec<f64>,
    /// The average fitness of every generation, from 0 to `generation`.
    #[cfg_attr(
        feature = "serde",
        serde(deserialize_with = "crate::fitness::deserialize_fitnesses")
    )]
    pub avg_fitnesses: Vec<f64>,
    /// [`GenerationReport::generations_without_improvement`](crate::GenerationReport::generations_without_improvement)
    /// at `generation`.
    pub generations_without_improvement: u32,
}

impl<T> Checkpoint<T> {
    /// A checkpoint from its parts, for one saved in a format of your own.
    /// [`Evolution::checkpoint`](crate::Evolution::checkpoint) is the usual way to get one, and
    /// with the `serde` feature `Deserialize` loads one.
    ///
    /// [`EvolutionBuilder::build`](crate::EvolutionBuilder::build) checks that it fits the
    /// builder.
    pub fn new(
        seed: u64,
        generation: u32,
        population: Vec<T>,
        best_ever: T,
        best_fitnesses: Vec<f64>,
        avg_fitnesses: Vec<f64>,
        generations_without_improvement: u32,
    ) -> Self {
        Self {
            seed,
            generation,
            population,
            best_ever,
            best_fitnesses,
            avg_fitnesses,
            generations_without_improvement,
        }
    }
}

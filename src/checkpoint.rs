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
/// There is no public constructor: a checkpoint comes from
/// [`Evolution::checkpoint`](crate::Evolution::checkpoint), or, with the `serde` feature, from
/// deserializing one that was saved. Its fields are public, to read them or to change one (for
/// example the `seed`, to branch off the checkpoint).
///
/// # Saving a checkpoint
///
/// With the `serde` feature it implements `Serialize` and `Deserialize` when the individual
/// does, so it can be saved in any format serde supports. To resume exactly, the format must
/// give back every `f64` bit for bit, `NaN` and infinities included.
///
/// - A binary format such as CBOR (with `ciborium`) gives back every `f64` exactly, `NaN` and
///   infinities included.
/// - JSON with `serde_json` needs its `float_roundtrip` feature, in your own dependencies (this
///   crate does not turn it on for you):
///
///   ```toml
///   serde_json = { version = "1", features = ["float_roundtrip"] }
///   ```
///
///   Without it the last bit of some numbers can change on the way back, and the resumed run
///   then drifts away from the one that was saved.
/// - JSON has no `NaN` nor infinities: `serde_json` writes both as `null`, which loads back as
///   `NaN`. A `NaN` fitness comes back as it was, but an infinite one comes back as `NaN`, and
///   **the resume is then not exact**: that individual now ranks below every number, as `NaN`
///   does, so selection, `best_ever` and `generations_without_improvement` go another way than
///   in the saved run. If your fitness can be infinite, save to a binary format.
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
// A field added later must be `#[cfg_attr(feature = "serde", serde(default))]`, so that
// checkpoints saved by an older version keep loading.
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

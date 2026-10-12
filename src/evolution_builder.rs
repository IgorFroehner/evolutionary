use crate::{
    checkpoint::Checkpoint,
    crossover::Crossover,
    evolution::{Evolution, EvolutionConfig, OnGenerationFn, StopConditionFn},
    fitness::Fitness,
    mutation::Mutation,
    population::Individual,
    report::GenerationReport,
    selection::Selection,
};
use std::{error::Error, fmt, sync::Arc};

/// Why [`EvolutionBuilder::build`] could not build an [`Evolution`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BuildError {
    MissingConfig,
    MissingFitness,
    MissingSelection,
    MissingCrossover,
    MissingMutation,
    MissingStopCondition,
    /// The population size is zero, so there is nothing to evolve.
    EmptyPopulation,
    /// More elitists were asked for than there are individuals in the population.
    ElitismExceedsPopulation {
        elitism: u32,
        population_size: u32,
    },
    /// More initial individuals were given than the population size.
    InitialPopulationTooLarge {
        given: usize,
        population_size: u32,
    },
    /// The initial individual at `index` doesn't have `dimension` genes.
    InitialIndividualDimension {
        index: usize,
        genes: usize,
        dimension: u32,
    },
    /// Both [`with_checkpoint`](EvolutionBuilder::with_checkpoint) and a non-empty
    /// [`with_initial_population`](EvolutionBuilder::with_initial_population) were given: a
    /// checkpoint already holds the population a run resumes with.
    CheckpointWithInitialPopulation,
    /// [`with_seed`](EvolutionBuilder::with_seed) gave another seed than the checkpoint's,
    /// which the resumed run keeps.
    CheckpointSeed {
        seed: u64,
        checkpoint_seed: u64,
    },
    /// The checkpoint's population has `given` individuals instead of the population size.
    CheckpointPopulationSize {
        given: usize,
        population_size: u32,
    },
    /// An individual of the checkpoint doesn't have `dimension` genes: the one at `index` in
    /// its population, or its `best_ever` when `index` is `None`.
    CheckpointIndividualDimension {
        index: Option<usize>,
        genes: usize,
        dimension: u32,
    },
    /// The checkpoint's fitness history doesn't have one best and one average fitness for
    /// every generation from 0 to `generation`.
    CheckpointHistory {
        generation: u32,
        best_fitnesses: usize,
        avg_fitnesses: usize,
    },
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingConfig => f.write_str("no evolution config was provided"),
            Self::MissingFitness => f.write_str("no fitness was provided, use `with_fitness`"),
            Self::MissingSelection => {
                f.write_str("no selection was provided, use `with_selection`")
            }
            Self::MissingCrossover => {
                f.write_str("no crossover was provided, use `with_crossover`")
            }
            Self::MissingMutation => f.write_str("no mutation was provided, use `with_mutation`"),
            Self::MissingStopCondition => {
                f.write_str("no stop condition was provided, use `with_stop_condition`")
            }
            Self::EmptyPopulation => f.write_str("the population size must be at least 1"),
            Self::ElitismExceedsPopulation {
                elitism,
                population_size,
            } => write!(
                f,
                "elitism ({elitism}) is bigger than the population size ({population_size})"
            ),
            Self::InitialPopulationTooLarge {
                given,
                population_size,
            } => write!(
                f,
                "{given} initial individuals were given, more than the population size \
                 ({population_size})"
            ),
            Self::InitialIndividualDimension {
                index,
                genes,
                dimension,
            } => write!(
                f,
                "the initial individual {index} has {genes} genes instead of {dimension}"
            ),
            Self::CheckpointWithInitialPopulation => f.write_str(
                "a checkpoint and an initial population were both given, the checkpoint holds \
                 the population to resume with",
            ),
            Self::CheckpointSeed {
                seed,
                checkpoint_seed,
            } => write!(
                f,
                "the seed ({seed}) is not the checkpoint's ({checkpoint_seed}), which the \
                 resumed run keeps"
            ),
            Self::CheckpointPopulationSize {
                given,
                population_size,
            } => write!(
                f,
                "the checkpoint has {given} individuals instead of the population size \
                 ({population_size})"
            ),
            Self::CheckpointIndividualDimension {
                index: Some(index),
                genes,
                dimension,
            } => write!(
                f,
                "the checkpoint's individual {index} has {genes} genes instead of {dimension}"
            ),
            Self::CheckpointIndividualDimension {
                index: None,
                genes,
                dimension,
            } => write!(
                f,
                "the checkpoint's best ever has {genes} genes instead of {dimension}"
            ),
            Self::CheckpointHistory {
                generation,
                best_fitnesses,
                avg_fitnesses,
            } => write!(
                f,
                "a checkpoint of generation {generation} needs {} best and average fitnesses, \
                 it has {best_fitnesses} and {avg_fitnesses}",
                u64::from(*generation) + 1
            ),
        }
    }
}

impl Error for BuildError {}

/// This is the helper struct to create a new Evolution object. The `fitness`, `selection`,
/// `crossover`, `mutation` and `stop_condition` are required.
///
/// # Example
///
/// ```
/// # use evolutionary::prelude::*;
/// # #[derive(Clone)]
/// # struct YourFitness;
/// # impl Fitness<Bin> for YourFitness {
/// #    fn calculate_fitness(&self, individual: &Bin) -> f64 {
/// #        0.0
/// #    }
/// # }
/// fn main() {
///     let mut evolution = EvolutionBuilder::new(30, 10, ())
///         .with_fitness(YourFitness)
///         .with_selection(TournamentSelection::default())
///         .with_crossover(NPointsCrossover::default())
///         .with_mutation(BitFlipMutation::default())
///         .with_title("Max")
///         .with_stop_condition(|report| report.generation >= 1000)
///         .build().unwrap();
/// }
/// ```
pub struct EvolutionBuilder<T: Individual> {
    title: Option<String>,
    evolution_config: Option<EvolutionConfig<T>>,
    fitness: Option<Box<dyn Fitness<T>>>,
    selection: Option<Box<dyn Selection<T>>>,
    crossover: Option<Box<dyn Crossover<T>>>,
    mutation: Option<Box<dyn Mutation<T>>>,
    elitism: Option<u32>,
    reevaluate_elitists: bool,
    stop_condition: Option<StopConditionFn<T>>,
    on_generation: Option<OnGenerationFn<T>>,
    pub(crate) seed: Option<u64>,
    initial_population: Vec<T>,
    pub(crate) checkpoint: Option<Checkpoint<T>>,
}

impl<T: Individual> EvolutionBuilder<T> {
    /// Starts a builder for a population of `population_size` individuals of `dimension` genes
    /// each, generated within `range`.
    ///
    /// The individual type `T` sets the encoding, and with it the type of `range`: `()` for
    /// [`Bin`](crate::prelude::Bin) and [`IntPerm`](crate::prelude::IntPerm), `(i64, i64)` for
    /// [`Int`](crate::prelude::Int) and `(f64, f64)` for [`Real`](crate::prelude::Real).
    pub fn new(population_size: u32, dimension: u32, range: T::RangeType) -> Self {
        let evolution_config = Some(EvolutionConfig {
            range,
            dimension,
            population_size,
        });

        Self {
            title: None,
            evolution_config,
            fitness: None,
            selection: None,
            crossover: None,
            mutation: None,
            stop_condition: None,
            on_generation: None,
            elitism: None,
            reevaluate_elitists: false,
            seed: None,
            initial_population: Vec::new(),
            checkpoint: None,
        }
    }

    /// Helper method to create a new EvolutionBuilder from a EvolutionConfig.
    pub fn from_config(config: EvolutionConfig<T>) -> Self {
        Self {
            title: None,
            fitness: None,
            selection: None,
            crossover: None,
            mutation: None,
            stop_condition: None,
            on_generation: None,
            evolution_config: Some(config.clone()),
            elitism: None,
            reevaluate_elitists: false,
            seed: None,
            initial_population: Vec::new(),
            checkpoint: None,
        }
    }

    /// Sets the fitness function. Receives a struct that implements the Fitness trait.
    pub fn with_fitness<F: Fitness<T>>(mut self, f: F) -> Self {
        self.fitness = Some(Box::new(f));
        self
    }

    /// Sets the selection operator. Receives a struct that implements the Selection trait.
    pub fn with_selection<S: Selection<T>>(mut self, s: S) -> Self {
        self.selection = Some(Box::new(s));
        self
    }

    /// Sets the crossover operator. Receives a struct that implements the Crossover trait.
    pub fn with_crossover<X: Crossover<T>>(mut self, c: X) -> Self {
        self.crossover = Some(Box::new(c));
        self
    }

    /// Sets the mutation operator. Receives a struct that implements the Mutation trait.
    pub fn with_mutation<M: Mutation<T>>(mut self, m: M) -> Self {
        self.mutation = Some(Box::new(m));
        self
    }

    /// Sets the stop condition, checked after every generation with its
    /// [`GenerationReport`]: the run stops as soon as it returns `true`.
    ///
    /// # Example
    ///
    /// ```text
    /// evolution_builder.with_stop_condition(|report| {
    ///     report.generation >= 1000 || report.generations_without_improvement >= 100
    /// })
    /// ```
    pub fn with_stop_condition<F>(mut self, f: F) -> Self
    where
        F: Fn(&GenerationReport<T>) -> bool + 'static + Send + Sync,
    {
        self.stop_condition = Some(Arc::new(f));
        self
    }

    /// Calls `f` after every generation, the initial population included, with its
    /// [`GenerationReport`]. For example to log the progress, or to save the champion of each
    /// generation.
    ///
    /// It is called from the thread that runs the evolution, and shared by every evolution
    /// built from this builder: use a `Mutex` or a channel to collect results.
    pub fn with_on_generation<F>(mut self, f: F) -> Self
    where
        F: Fn(&GenerationReport<T>) + 'static + Send + Sync,
    {
        self.on_generation = Some(Arc::new(f));
        self
    }

    /// Number of elitist population members to ensure that keeps in the pool. Default is 1.
    pub fn with_elitism(mut self, elitism: u32) -> Self {
        self.elitism = Some(elitism);
        self
    }

    /// Evaluates the elitists again every generation, with the rest of the population, instead
    /// of carrying the fitness they were stored with. Off by default.
    ///
    /// Turn it on when the fitness is noisy: a simulation with random starting conditions, a
    /// score on a random sample of test cases. Without it, an individual that got one lucky
    /// evaluation stays the best of the population for as long as elitism keeps it, with its
    /// lucky fitness, while every other individual is judged on a fresh one. With it, the
    /// elitists still survive into the next generation, but with their new fitness, and only
    /// stay the best if they keep scoring as the best. A deterministic fitness gains nothing
    /// from it: the elitists would get the same numbers again.
    ///
    /// It costs `elitism` extra evaluations per generation, or fewer when a custom selection
    /// returns a mating pool smaller than that (see
    /// [`with_elitism`](Self::with_elitism)): the elitists are passed to
    /// [`Fitness::calculate_population`] in the same call as the new individuals, so a batched
    /// fitness still gets one call per generation. It draws no random number, so with a
    /// deterministic fitness a seeded run is the same with it on or off.
    ///
    /// [`Evolution::best_ever`] is a record of the best fitness ever computed, so with a noisy
    /// fitness it can still be a lucky evaluation. [`Evolution::current_best`] is the best of
    /// the current population, every member of which, elitists included, was evaluated in the
    /// last generation.
    ///
    /// For a noisy fitness that still replays a seeded run, see [`Fitness`]'s "Randomness".
    pub fn with_reevaluate_elitists(mut self, reevaluate: bool) -> Self {
        self.reevaluate_elitists = reevaluate;
        self
    }

    /// Starts every run from `population` instead of from random individuals: to seed it with
    /// known good individuals, or to start a new run from the population of another. When it
    /// has fewer individuals than the population size, random ones fill the rest.
    ///
    /// Their fitness is calculated again when the run starts, which is generation 0. To go on
    /// with a stopped run exactly where it was instead, use
    /// [`with_checkpoint`](Self::with_checkpoint). [`build`](Self::build) fails if there are more
    /// individuals than the population size, or if one doesn't have `dimension` genes.
    pub fn with_initial_population(mut self, population: Vec<T>) -> Self {
        self.initial_population = population;
        self
    }

    /// Resumes every run from `checkpoint`, taken by
    /// [`Evolution::checkpoint`](crate::Evolution::checkpoint), instead of starting it from
    /// generation 0: [`Evolution::start`] puts back its population with their fitnesses, its
    /// best ever and its fitness history, and the run goes on exactly as the saved one would
    /// have, as long as the rest of the builder (the operators, the fitness, the elitism) is the
    /// same. The stop condition and the callback can differ, for example to run longer.
    ///
    /// The run keeps the checkpoint's seed, so [`with_seed`](Self::with_seed) is not needed.
    /// [`build`](Self::build) fails if:
    ///
    /// - `with_seed` gave another seed ([`BuildError::CheckpointSeed`]),
    /// - [`with_initial_population`](Self::with_initial_population) gave individuals too
    ///   ([`BuildError::CheckpointWithInitialPopulation`]),
    /// - the checkpoint's population doesn't have the population size, an individual of it
    ///   doesn't have `dimension` genes, or its fitness history doesn't have one entry per
    ///   generation.
    ///
    /// The population size check is exact. A custom [`Selection`] may hand back a smaller
    /// mating pool than the population, and the run then goes on with fewer individuals; a
    /// checkpoint of such a run has fewer individuals than the population size, so it is
    /// rejected with [`BuildError::CheckpointPopulationSize`] and can't be resumed.
    ///
    /// In an [`ExperimentRunner`](crate::prelude::ExperimentRunner), every run resumes from the
    /// checkpoint with a seed of its own drawn from the checkpoint's, so the runs branch off it
    /// in different directions.
    pub fn with_checkpoint(mut self, checkpoint: Checkpoint<T>) -> Self {
        self.checkpoint = Some(checkpoint);
        self
    }

    /// Seeds every random choice of the evolution, so that two runs with the same seed and the
    /// same parameters produce the same populations, however many threads they run on. Without
    /// a seed every run is different; [`Evolution::seed`] tells the seed a run used. A run
    /// resumed with [`with_checkpoint`](Self::with_checkpoint) keeps the checkpoint's seed, and
    /// `build` fails if this one is another.
    ///
    /// The fitness, and any custom operator, must be deterministic too (see [`Fitness`]'s
    /// "Randomness" for a noisy fitness that replays), and custom operators must draw from the
    /// [`Rngs`](crate::Rngs) they receive.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sets the title of the evolution, [`Evolution::title`], which `Evolution::plot_chart` (with
    /// the `plot` feature) uses as the caption of the chart. Defaults to `""`.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Builds the [`Evolution`], checking that every required part was set and that the
    /// parameters are consistent.
    pub fn build(&self) -> Result<Evolution<T>, BuildError> {
        let evolution_config = self
            .evolution_config
            .clone()
            .ok_or(BuildError::MissingConfig)?;

        if evolution_config.population_size == 0 {
            return Err(BuildError::EmptyPopulation);
        }

        let elitism = self.elitism.unwrap_or(1);
        if elitism > evolution_config.population_size {
            return Err(BuildError::ElitismExceedsPopulation {
                elitism,
                population_size: evolution_config.population_size,
            });
        }

        if self.initial_population.len() > evolution_config.population_size as usize {
            return Err(BuildError::InitialPopulationTooLarge {
                given: self.initial_population.len(),
                population_size: evolution_config.population_size,
            });
        }
        if let Some((index, individual)) =
            self.initial_population
                .iter()
                .enumerate()
                .find(|(_, individual)| {
                    individual.get_chromosome().len() != evolution_config.dimension as usize
                })
        {
            return Err(BuildError::InitialIndividualDimension {
                index,
                genes: individual.get_chromosome().len(),
                dimension: evolution_config.dimension,
            });
        }

        if let Some(checkpoint) = &self.checkpoint {
            self.check_checkpoint(checkpoint, &evolution_config)?;
        }

        let fitness = self.fitness.as_deref().ok_or(BuildError::MissingFitness)?;
        let selection = self
            .selection
            .as_deref()
            .ok_or(BuildError::MissingSelection)?;
        let crossover = self
            .crossover
            .as_deref()
            .ok_or(BuildError::MissingCrossover)?;
        let mutation = self
            .mutation
            .as_deref()
            .ok_or(BuildError::MissingMutation)?;
        let stop_condition = self
            .stop_condition
            .as_ref()
            .ok_or(BuildError::MissingStopCondition)?;

        let mut evolution = Evolution::new(
            self.title.clone().unwrap_or_default(),
            evolution_config,
            dyn_clone::clone_box(fitness),
            dyn_clone::clone_box(selection),
            dyn_clone::clone_box(crossover),
            dyn_clone::clone_box(mutation),
            elitism,
            Arc::clone(stop_condition),
        );
        // The checkpoint's seed, unless `with_seed` gave one, which is then the same.
        evolution.fixed_seed = self
            .seed
            .or(self.checkpoint.as_ref().map(|checkpoint| checkpoint.seed));
        evolution.reevaluate_elitists = self.reevaluate_elitists;
        evolution.on_generation = self.on_generation.clone();
        evolution.initial_population = self.initial_population.clone();
        evolution.checkpoint = self.checkpoint.clone();

        Ok(evolution)
    }

    /// Checks that a run of this builder can resume from `checkpoint`.
    fn check_checkpoint(
        &self,
        checkpoint: &Checkpoint<T>,
        config: &EvolutionConfig<T>,
    ) -> Result<(), BuildError> {
        if !self.initial_population.is_empty() {
            return Err(BuildError::CheckpointWithInitialPopulation);
        }

        if let Some(seed) = self.seed.filter(|&seed| seed != checkpoint.seed) {
            return Err(BuildError::CheckpointSeed {
                seed,
                checkpoint_seed: checkpoint.seed,
            });
        }

        if checkpoint.population.len() != config.population_size as usize {
            return Err(BuildError::CheckpointPopulationSize {
                given: checkpoint.population.len(),
                population_size: config.population_size,
            });
        }

        let individuals = checkpoint
            .population
            .iter()
            .enumerate()
            .map(|(index, individual)| (Some(index), individual))
            .chain([(None, &checkpoint.best_ever)]);
        for (index, individual) in individuals {
            let genes = individual.get_chromosome().len();
            if genes != config.dimension as usize {
                return Err(BuildError::CheckpointIndividualDimension {
                    index,
                    genes,
                    dimension: config.dimension,
                });
            }
        }

        let generations = u64::from(checkpoint.generation) + 1;
        if checkpoint.best_fitnesses.len() as u64 != generations
            || checkpoint.avg_fitnesses.len() as u64 != generations
        {
            return Err(BuildError::CheckpointHistory {
                generation: checkpoint.generation,
                best_fitnesses: checkpoint.best_fitnesses.len(),
                avg_fitnesses: checkpoint.avg_fitnesses.len(),
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[derive(Clone)]
    struct ZeroFitness;

    impl Fitness<Bin> for ZeroFitness {
        fn calculate_fitness(&self, _: &Bin) -> f64 {
            0.0
        }
    }

    fn complete(population_size: u32, dimension: u32) -> EvolutionBuilder<Bin> {
        EvolutionBuilder::new(population_size, dimension, ())
            .with_fitness(ZeroFitness)
            .with_selection(TournamentSelection::default())
            .with_crossover(NPointsCrossover::default())
            .with_mutation(BitFlipMutation::default())
            .with_stop_condition(|report| report.generation >= 1)
    }

    #[test]
    fn builds_when_complete() {
        assert!(complete(10, 4).build().is_ok());
    }

    #[test]
    fn the_title_reaches_the_evolution() {
        assert_eq!(complete(10, 4).build().unwrap().title(), "");

        let title = String::from("Max");
        let evolution = complete(10, 4).with_title(&*title).build().unwrap();
        assert_eq!(evolution.title(), "Max");
        assert_eq!(
            complete(10, 4).with_title(title).build().unwrap().title(),
            "Max"
        );
    }

    #[test]
    fn elitism_is_checked_against_the_population_size() {
        // Larger than the dimension, but fits in the population: valid.
        assert!(complete(10, 2).with_elitism(5).build().is_ok());

        assert_eq!(
            complete(10, 50).with_elitism(11).build().err(),
            Some(BuildError::ElitismExceedsPopulation {
                elitism: 11,
                population_size: 10
            })
        );
    }

    #[test]
    fn empty_population_is_an_error() {
        assert_eq!(
            complete(0, 4).with_elitism(0).build().err(),
            Some(BuildError::EmptyPopulation)
        );
    }

    #[test]
    fn missing_stop_condition_is_an_error() {
        let builder = EvolutionBuilder::<Bin>::new(10, 4, ())
            .with_fitness(ZeroFitness)
            .with_selection(TournamentSelection::default())
            .with_crossover(NPointsCrossover::default())
            .with_mutation(BitFlipMutation::default());

        assert_eq!(
            builder.build().err(),
            Some(BuildError::MissingStopCondition)
        );
    }

    #[test]
    fn missing_operator_is_named_in_the_error() {
        let builder = EvolutionBuilder::<Bin>::new(10, 4, ())
            .with_fitness(ZeroFitness)
            .with_crossover(NPointsCrossover::default())
            .with_mutation(BitFlipMutation::default())
            .with_stop_condition(|report| report.generation >= 1);

        let error = builder.build().err().unwrap();

        assert_eq!(error, BuildError::MissingSelection);
        assert!(error.to_string().contains("with_selection"));
    }
}

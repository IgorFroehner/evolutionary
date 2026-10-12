use crate::{
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
///         .with_title("Max".to_string())
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
    stop_condition: Option<StopConditionFn<T>>,
    on_generation: Option<OnGenerationFn<T>>,
    pub(crate) seed: Option<u64>,
    initial_population: Vec<T>,
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
            seed: None,
            initial_population: Vec::new(),
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
            seed: None,
            initial_population: Vec::new(),
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

    /// Starts every run from `population` instead of from random individuals: to resume a run
    /// from a saved population, or to seed it with known good individuals. When it has fewer
    /// individuals than the population size, random ones fill the rest.
    ///
    /// Their fitness is calculated again when the run starts. [`build`](Self::build) fails if
    /// there are more individuals than the population size, or if one doesn't have `dimension`
    /// genes.
    pub fn with_initial_population(mut self, population: Vec<T>) -> Self {
        self.initial_population = population;
        self
    }

    /// Seeds every random choice of the evolution, so that two runs with the same seed and the
    /// same parameters produce the same populations, however many threads they run on. Without
    /// a seed every run is different; [`Evolution::seed`] tells the seed a run used.
    ///
    /// The fitness, and any custom operator, must be deterministic too, and custom operators
    /// must draw from the [`Rngs`](crate::Rngs) they receive.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sets the title of the evolution to use when plotting. Defaults to `""`.
    pub fn with_title(mut self, title: String) -> Self {
        self.title = Some(title);
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
        evolution.fixed_seed = self.seed;
        evolution.on_generation = self.on_generation.clone();
        evolution.initial_population = self.initial_population.clone();

        Ok(evolution)
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

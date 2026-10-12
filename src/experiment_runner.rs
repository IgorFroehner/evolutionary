use std::time::{Duration, Instant};
#[cfg(feature = "plot")]
use std::{error::Error, fs::create_dir_all, path::Path};

#[cfg(feature = "plot")]
use crate::plotting::plot_chart;
use crate::rng::{Rngs, Step};
use crate::{
    evolution_builder::BuildError, evolution_builder::EvolutionBuilder, population::Individual,
};

use rand::Rng;
use rayon::prelude::*;

/// Runs one [`EvolutionBuilder`] many times, in parallel, and collects the results of every run.
///
/// A seeded experiment is reproducible, and each of its runs gets its own seed, kept in
/// [`ExperimentResult::seed`] so a single run can be replayed on its own.
///
/// ```rust
/// # use evolutionary::prelude::*;
/// # #[derive(Clone)]
/// # struct CountOnes;
/// # impl Fitness<Bin> for CountOnes {
/// #     fn calculate_fitness(&self, individual: &Bin) -> f64 {
/// #         individual.get_chromosome().iter().filter(|&&gene| gene).count() as f64
/// #     }
/// # }
/// let builder = EvolutionBuilder::new(20, 16, GeneCod::Bin, ())
///     .with_fitness(CountOnes)
///     .with_selection(TournamentSelection::default())
///     .with_crossover(NPointsCrossover::default())
///     .with_mutation(BitFlipMutation::default())
///     .with_stop_condition(|report| report.generation >= 10)
///     .with_seed(1);
///
/// let mut experiment = ExperimentRunner::new("Count ones".to_string(), 4, builder);
/// experiment.run()?;
///
/// assert_eq!(experiment.experiment_results.len(), 4);
/// println!("Average best fitness: {:?}", experiment.average_best_fitness());
/// # Ok::<(), BuildError>(())
/// ```
pub struct ExperimentRunner<T: Individual> {
    runs: u32,
    name: String,
    evolution_builder: EvolutionBuilder<T>,
    /// The results of every run so far, in the order of the runs.
    pub experiment_results: Vec<ExperimentResult<T>>,
}

/// The outcome of one run of an [`ExperimentRunner`].
pub struct ExperimentResult<T> {
    /// How long the run took, from the start of the evolution to its stop condition.
    pub time: Duration,
    /// The seed of the run. Pass it to
    /// [`EvolutionBuilder::with_seed`](crate::EvolutionBuilder::with_seed) to replay this run on
    /// its own.
    pub seed: u64,
    /// The average fitness of each generation of the run.
    pub average_fitnesses: Vec<f64>,
    /// The best fitness of each generation of the run.
    pub best_fitnesses: Vec<f64>,
    /// The number of generations of the run, the initial population included.
    pub iterations: u32,
    /// The best individual seen during the run, as
    /// [`Evolution::best_ever`](crate::Evolution::best_ever). Without elitism it can be better
    /// than the best of the last generation.
    pub best_found: T,
}

impl<T: Individual> ExperimentRunner<T> {
    pub fn new(name: String, runs: u32, evolution_builder: EvolutionBuilder<T>) -> Self {
        Self {
            name,
            runs,
            evolution_builder,
            experiment_results: Vec::new(),
        }
    }

    /// Runs the evolution `runs` times, in parallel, and adds the results to
    /// [`experiment_results`](Self::experiment_results).
    ///
    /// Every run is built before any starts, so an invalid builder returns its [`BuildError`]
    /// without running anything.
    pub fn run(&mut self) -> Result<(), BuildError> {
        let evolutions = (0..self.runs)
            .map(|_| self.evolution_builder.build())
            .collect::<Result<Vec<_>, _>>()?;
        let experiment_seed = self.evolution_builder.seed;

        let results: Vec<ExperimentResult<T>> = evolutions
            .into_par_iter()
            .enumerate()
            .map(|(run, mut evolution)| {
                // A seeded experiment is reproducible, but its runs still differ.
                let seed = match experiment_seed {
                    Some(seed) => Rngs::for_step(seed, 0, Step::Experiment).rng(run).random(),
                    None => rand::random(),
                };
                evolution.fixed_seed = Some(seed);

                let start_time = Instant::now();

                evolution.run();

                let time = start_time.elapsed();

                ExperimentResult {
                    time,
                    seed,
                    average_fitnesses: evolution.metrics.avg_fitnesses.clone(),
                    best_fitnesses: evolution.metrics.best_fitnesses.clone(),
                    iterations: evolution.metrics.iterations,
                    best_found: evolution.best_ever().clone(),
                }
            })
            .collect();

        self.experiment_results.extend(results);

        Ok(())
    }

    /// Saves a chart of the best and average fitness per generation, averaged over every run in
    /// [`experiment_results`](Self::experiment_results), to `path` (a PNG), creating its parent
    /// directories if needed. Runs that stopped early count with their last values until the
    /// longest one ends.
    ///
    /// Nothing is written unless this is called.
    #[cfg(feature = "plot")]
    pub fn save_chart(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
        let path = path.as_ref();
        let generations = self
            .experiment_results
            .iter()
            .map(|result| result.best_fitnesses.len())
            .max()
            .ok_or("the experiment has no results to plot, call `run` first")?;

        let mut average_fitness_per_generation = vec![0f64; generations];
        let mut best_fitness_per_generation = vec![0f64; generations];

        for result in &self.experiment_results {
            let last_average = result.average_fitnesses.last().copied().unwrap_or_default();
            let last_best = result.best_fitnesses.last().copied().unwrap_or_default();

            for generation in 0..generations {
                average_fitness_per_generation[generation] += result
                    .average_fitnesses
                    .get(generation)
                    .copied()
                    .unwrap_or(last_average);
                best_fitness_per_generation[generation] += result
                    .best_fitnesses
                    .get(generation)
                    .copied()
                    .unwrap_or(last_best);
            }
        }

        let results = self.experiment_results.len() as f64;
        for value in average_fitness_per_generation
            .iter_mut()
            .chain(&mut best_fitness_per_generation)
        {
            *value /= results;
        }

        if let Some(parent) = path.parent() {
            create_dir_all(parent)?;
        }
        let path = path.to_str().ok_or("the chart path is not valid UTF-8")?;

        plot_chart(
            &best_fitness_per_generation,
            &average_fitness_per_generation,
            path,
            &self.name,
        )
    }

    /// The time of every run added up. The runs go in parallel, so it is more than the time
    /// [`run`](Self::run) took.
    pub fn total_time(&self) -> Duration {
        self.experiment_results
            .iter()
            .map(|result| result.time)
            .sum()
    }

    /// The average time of a run, or `None` before the first [`run`](Self::run).
    pub fn average_time(&self) -> Option<Duration> {
        let runs = self.experiment_results.len() as u128;
        if runs == 0 {
            return None;
        }

        let nanos = self.total_time().as_nanos() / runs;
        Some(Duration::from_nanos(
            u64::try_from(nanos).unwrap_or(u64::MAX),
        ))
    }

    /// The average fitness of the best individual found by each run, or `None` before the first
    /// [`run`](Self::run).
    pub fn average_best_fitness(&self) -> Option<f64> {
        self.average(|result| result.best_found.get_fitness())
    }

    /// The average number of generations of a run, the initial population included (see
    /// [`ExperimentResult::iterations`]), or `None` before the first [`run`](Self::run).
    pub fn average_generations(&self) -> Option<f64> {
        self.average(|result| result.iterations as f64)
    }

    fn average(&self, value: impl Fn(&ExperimentResult<T>) -> f64) -> Option<f64> {
        if self.experiment_results.is_empty() {
            return None;
        }

        let sum = self.experiment_results.iter().map(value).sum::<f64>();
        Some(sum / self.experiment_results.len() as f64)
    }

    /// Prints the average time, best fitness and number of generations of the runs, and the
    /// most generations a run took. [`average_time`](Self::average_time),
    /// [`average_best_fitness`](Self::average_best_fitness) and
    /// [`average_generations`](Self::average_generations) give the same numbers.
    pub fn experiment_digest(&self) {
        let (Some(average_time), Some(average_best_fitness), Some(average_generations)) = (
            self.average_time(),
            self.average_best_fitness(),
            self.average_generations(),
        ) else {
            println!("{}: no runs yet", self.name);
            return;
        };
        let max_generations = self
            .experiment_results
            .iter()
            .map(|result| result.iterations)
            .max()
            .unwrap_or_default();

        println!("Average time for {}: {:?}", self.name, average_time);
        println!("Average best fitness: {}", average_best_fitness);
        println!("Average generations: {}", average_generations);
        println!("Max generations: {}", max_generations);
    }
}

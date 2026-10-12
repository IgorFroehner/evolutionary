use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::cmp::Ordering;
use std::sync::Arc;

use crate::{
    checkpoint::Checkpoint,
    crossover::Crossover,
    fitness::{fitness_cmp, Fitness},
    metrics::{Metrics, Steps},
    mutation::Mutation,
    population::Individual,
    report::GenerationReport,
    rng::{Rngs, Step},
    selection::Selection,
};

pub type StopConditionFn<T> = Arc<dyn Fn(&GenerationReport<T>) -> bool + Send + Sync>;
pub type OnGenerationFn<T> = Arc<dyn Fn(&GenerationReport<T>) + Send + Sync>;

/// What every generation hands to the reports, computed once when it ends instead of being
/// scanned for again at every query.
#[derive(Clone, Copy)]
struct GenerationStats {
    best_index: usize,
    average_fitness: f64,
}

#[derive(Clone)]
pub struct EvolutionConfig<T: Individual> {
    pub dimension: u32,
    pub population_size: u32,
    pub range: T::RangeType,
}

/// This is the struct that does the evolution magic. It has the methods needed to start and iterate
/// through the evolution.
///
/// Build one with [`EvolutionBuilder`](crate::EvolutionBuilder), which checks the parameters.
///
/// # Example:
///
/// ```rust
/// # use evolutionary::prelude::*;
/// # #[derive(Clone)]
/// # struct YourFitness;
/// # impl Fitness<Bin> for YourFitness {
/// #    fn calculate_fitness(&self, individual: &Bin) -> f64 { 0.0 }
/// # }
/// # let mut evolution = EvolutionBuilder::new(30, 10, ())
/// #     .with_fitness(YourFitness)
/// #     .with_selection(TournamentSelection::default())
/// #     .with_crossover(NPointsCrossover::default())
/// #     .with_mutation(BitFlipMutation::default())
/// #     .with_title("Max")
/// #     .with_stop_condition(|report| report.generation >= 1000)
/// #     .build().unwrap();
/// #
/// // You can start the evolution with the `start` method:
/// evolution.start();
///
/// // and iterate through the evolution with the `next` method:
/// while !evolution.reached_stop_condition() {
///     evolution.next();
///     let _best_so_far = evolution.report().best_ever;
/// }
///
/// // or you can run it until the stop condition is met with the `run` method:
/// evolution.run();
/// ```
pub struct Evolution<T: Individual> {
    title: String,
    config: EvolutionConfig<T>,
    fitness: Box<dyn Fitness<T>>,
    selection: Box<dyn Selection<T>>,
    crossover: Box<dyn Crossover<T>>,
    mutation: Box<dyn Mutation<T>>,
    elitism: u32,
    /// Whether the elitists are evaluated again every generation instead of keeping their
    /// fitness, see
    /// [`EvolutionBuilder::with_reevaluate_elitists`](crate::EvolutionBuilder::with_reevaluate_elitists).
    pub(crate) reevaluate_elitists: bool,
    stop_condition: StopConditionFn<T>,
    pub(crate) on_generation: Option<OnGenerationFn<T>>,
    pub metrics: Metrics,
    current_population: Vec<T>,
    /// Stats of the current population, valid between the end of a generation and the next
    /// change of the population. `None` while a generation is being computed.
    stats: Option<GenerationStats>,
    /// The best individual seen since `start`.
    best_ever: Option<T>,
    /// The individuals every run starts with, before the random ones.
    pub(crate) initial_population: Vec<T>,
    /// The seed given to the builder. `None` draws a new one at every `start`.
    pub(crate) fixed_seed: Option<u64>,
    /// The seed of the current run, set by `start`.
    run_seed: Option<u64>,
    /// The checkpoint every run resumes from, instead of starting from generation 0. `build`
    /// sets `fixed_seed` to its seed.
    pub(crate) checkpoint: Option<Checkpoint<T>>,
}

impl<T: Individual> Evolution<T> {
    /// Only [`EvolutionBuilder::build`](crate::EvolutionBuilder::build) calls this, after
    /// checking the parameters.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        title: String,
        config: EvolutionConfig<T>,
        fitness: Box<dyn Fitness<T>>,
        selection: Box<dyn Selection<T>>,
        crossover: Box<dyn Crossover<T>>,
        mutation: Box<dyn Mutation<T>>,
        elitism: u32,
        stop_condition: StopConditionFn<T>,
    ) -> Self {
        Self {
            title,
            current_population: Vec::new(),
            stats: None,
            config,
            fitness,
            selection,
            crossover,
            mutation,
            elitism,
            reevaluate_elitists: false,
            stop_condition,
            on_generation: None,
            metrics: Metrics::new(),
            best_ever: None,
            initial_population: Vec::new(),
            fixed_seed: None,
            run_seed: None,
            checkpoint: None,
        }
    }

    /// Starts the evolution, generating the initial population and calculating the
    /// fitness of each individual.
    ///
    /// The population starts with the individuals given to
    /// [`EvolutionBuilder::with_initial_population`](crate::EvolutionBuilder::with_initial_population),
    /// if any, and random ones fill the rest. Without a seed (see
    /// [`EvolutionBuilder::with_seed`](crate::EvolutionBuilder::with_seed)) every call starts a
    /// different run.
    ///
    /// Built with
    /// [`EvolutionBuilder::with_checkpoint`](crate::EvolutionBuilder::with_checkpoint), it
    /// resumes from the checkpoint instead: the population, the best ever and the fitness
    /// history are the checkpoint's, no fitness is computed, and the next [`next`](Self::next)
    /// runs the generation after the checkpoint's, as the run that was saved would have. The
    /// [`with_on_generation`](crate::EvolutionBuilder::with_on_generation) callback is not
    /// called for the checkpoint's generation, which it already saw in the run that was saved.
    /// Every call resumes from the same checkpoint.
    pub fn start(&mut self) {
        self.metrics = Metrics::new();
        self.best_ever = None;

        self.metrics.start_clock();

        self.run_seed = Some(self.fixed_seed.unwrap_or_else(rand::random));

        if let Some(checkpoint) = self.checkpoint.clone() {
            self.resume(checkpoint);
            return;
        }

        let rngs = self.rngs(Step::Initialization);
        let given = self.initial_population.len();
        let random: Vec<T> = (given..self.config.population_size as usize)
            .into_par_iter()
            .map(|index| {
                T::generate_member(
                    self.config.dimension,
                    &self.config.range,
                    &mut rngs.rng(index),
                )
            })
            .collect();
        self.current_population = self.initial_population.clone();
        self.current_population.extend(random);

        self.process_fitness();

        self.finish_generation();
    }

    /// The `start` of a run built with a checkpoint: puts back everything `next` and the report
    /// read, so the run goes on exactly as the one that was saved.
    fn resume(&mut self, checkpoint: Checkpoint<T>) {
        let Checkpoint {
            seed: _,
            generation: _,
            population,
            best_ever,
            best_fitnesses,
            avg_fitnesses,
            generations_without_improvement,
        } = checkpoint;

        // Starts the clock, and sets `iterations` to the history's length, the checkpoint's
        // generation plus one: the generation `next` runs, and so the key of its `Rngs`.
        self.metrics = Metrics::resumed(
            best_fitnesses,
            avg_fitnesses,
            generations_without_improvement,
            best_ever.get_fitness(),
        );

        self.current_population = population;
        self.best_ever = Some(best_ever);
        // The average is summed in the population's order, which the checkpoint keeps, so it is
        // the same number the saved run had.
        self.stats = self.population_stats();

        self.metrics.end_clock();
    }

    /// This method runs one generation of the evolution.
    /// It selects the mating pool, crossover, mutate and calculates the fitness of the new population.
    ///
    /// # Panics
    ///
    /// If the evolution was not started yet, or if an operator panics (see the `# Panics`
    /// section of each).
    pub fn next(&mut self) {
        self.metrics.step_start(Steps::Elitism);
        let mut elitists = self.find_elitists();
        self.metrics.step_end(Steps::Elitism);

        self.metrics.step_start(Steps::Selection);
        let mut mating_pool = self
            .selection
            .get_mating_pool(&self.current_population, &self.rngs(Step::Selection));
        self.metrics.step_end(Steps::Selection);

        self.metrics.step_start(Steps::Crossover);
        self.crossover
            .crossover(&mut mating_pool, &self.rngs(Step::Crossover));
        self.metrics.step_end(Steps::Crossover);

        self.metrics.step_start(Steps::Mutation);
        self.mutation
            .mutate(&mut mating_pool, &self.rngs(Step::Mutation));
        self.metrics.step_end(Steps::Mutation);

        self.current_population = mating_pool;
        self.stats = None;

        if self.reevaluate_elitists && !elitists.is_empty() {
            // Evaluated in the same call as the new individuals, so that a batched fitness
            // still sees one population per generation.
            let pool_size = self.current_population.len();
            self.current_population.append(&mut elitists);
            self.process_fitness();
            elitists = self.current_population.split_off(pool_size);
        } else {
            self.process_fitness();
        }

        self.metrics.step_start(Steps::Elitism);
        self.replace_worsts_with_elitists(elitists);
        self.metrics.step_end(Steps::Elitism);

        self.finish_generation();
    }

    /// This method runs the evolution, generation over generation, until the stop condition is met.
    /// It is the same as calling [`start`](Self::start), then [`next`](Self::next) until
    /// [`reached_stop_condition`](Self::reached_stop_condition).
    pub fn run(&mut self) {
        self.start();

        while !self.reached_stop_condition() {
            self.next();
        }
    }

    /// The state of the evolution after the last generation.
    ///
    /// # Panics
    ///
    /// If the evolution was not started yet, as there is no population.
    pub fn report(&self) -> GenerationReport<'_, T> {
        let best = self.current_best();

        GenerationReport {
            generation: self.generation(),
            best,
            best_fitness: best.get_fitness(),
            average_fitness: self.current_fitness_average(),
            best_ever: self.best_ever(),
            generations_without_improvement: self.metrics.gens_without_improvement,
        }
    }

    /// The best individual seen since the run started, which without elitism can be better
    /// than [`current_best`](Self::current_best).
    ///
    /// # Panics
    ///
    /// If the evolution was not started yet.
    pub fn best_ever(&self) -> &T {
        self.best_ever
            .as_ref()
            .expect("the population is empty, call `start` or `run` first")
    }

    /// The seed of the current run: the one given to
    /// [`EvolutionBuilder::with_seed`](crate::EvolutionBuilder::with_seed), or the one drawn by
    /// [`start`](Self::start) when none was given. Pass it to `with_seed` to replay the run.
    /// A run resumed from a [`Checkpoint`] keeps the checkpoint's seed. `None` until the
    /// evolution is started.
    pub fn seed(&self) -> Option<u64> {
        self.run_seed
    }

    /// Saves where the run is after the last generation, to resume it later with
    /// [`EvolutionBuilder::with_checkpoint`](crate::EvolutionBuilder::with_checkpoint) exactly
    /// as if it had not stopped. See [`Checkpoint`] for what it holds and an example.
    ///
    /// It copies the population, so call it every few generations rather than after each one
    /// if the population is large.
    ///
    /// # Panics
    ///
    /// If the evolution was not started yet.
    pub fn checkpoint(&self) -> Checkpoint<T> {
        let seed = self
            .run_seed
            .expect("the evolution is not started, call `start` or `run` first");

        Checkpoint {
            seed,
            generation: self.generation(),
            population: self.current_population.clone(),
            best_ever: self.best_ever().clone(),
            best_fitnesses: self.metrics.best_fitnesses.clone(),
            avg_fitnesses: self.metrics.avg_fitnesses.clone(),
            generations_without_improvement: self.metrics.gens_without_improvement,
        }
    }

    /// Prints the generation, the best and average fitness, and every individual with its
    /// fitness.
    ///
    /// # Panics
    ///
    /// If the evolution was not started yet.
    pub fn population_digest(&self) {
        println!("---------------------------------------------");
        println!("Iteration: {}", self.metrics.iterations);
        println!("Best Fitness: {}", self.current_best_fitness());
        println!("Current Average: {}", self.current_fitness_average());
        println!("---------------------------------------------");
        println!("Population: ");
        for individual in &self.current_population {
            println!(
                "Fitness: {} - Chromosome: {:?}",
                individual.get_fitness(),
                individual.get_chromosome()
            );
        }
    }

    /// Prints total time and the time spent in each step of the evolution. In the following
    /// format:
    /// ```text
    /// ------------ Time Digest ------------
    /// Total time: X.XXXXs
    /// Selection time: X.XXXXs (XX.XX%)
    /// Crossover time: X.XXXXs (XX.XX%)
    /// Mutation time: X.XXXXs (XX.XX%)
    /// Fitness time: X.XXXXs (XX.XX%)
    /// Elitism time: X.XXXXs (XX.XX%)
    /// ---------------------------------------
    /// ```
    ///
    /// [`metrics`](Self::metrics) gives the same numbers, with
    /// [`Metrics::total_time`] and [`Metrics::step_time`].
    pub fn time_digest(&self) {
        let total_time = self.metrics.total_time();

        println!("------------ Time Digest ------------");
        println!("Total time: {:?}", total_time);
        for (step, time) in self.metrics.step_times() {
            println!(
                "{:?} time: {:?} ({:.2}%)",
                step,
                time,
                time.as_nanos() as f64 / total_time.as_nanos() as f64 * 100.0
            );
        }
        println!("---------------------------------------");
    }

    /// Returns the best individual of the current population. An individual with a `NaN` fitness
    /// is only returned if every fitness is `NaN`.
    ///
    /// # Panics
    ///
    /// If the evolution was not started yet, as there is no population.
    pub fn current_best(&self) -> &T {
        let stats = self
            .stats
            .or_else(|| self.population_stats())
            .expect("the population is empty, call `start` or `run` first");

        &self.current_population[stats.best_index]
    }

    /// The current population, borrowed in its current order, without copying or sorting it.
    /// Empty before the evolution is started.
    ///
    /// The order is the one the last generation left, which has no meaning of its own: use
    /// [`current_best`](Self::current_best) for the best individual, and
    /// [`current_population`](Self::current_population) for a copy sorted by fitness.
    pub fn population(&self) -> &[T] {
        &self.current_population
    }

    /// Returns a copy of the current population sorted by fitness, from the worst to the best
    /// (individuals with a `NaN` fitness first).
    ///
    /// It clones and sorts the whole population at every call: to look at the individuals
    /// without copying them, borrow them with [`population`](Self::population).
    pub fn current_population(&self) -> Vec<T> {
        let mut current_population = self.current_population.clone();

        current_population.sort_by(|a, b| Self::cmp_by_fitness(a, b));

        current_population
    }

    /// Returns if the stop condition was already met for this evolution object.
    ///
    /// # Panics
    ///
    /// If the evolution was not started yet.
    pub fn reached_stop_condition(&self) -> bool {
        (self.stop_condition)(&self.report())
    }

    /// The fitness of [`current_best`](Self::current_best).
    ///
    /// # Panics
    ///
    /// If the evolution was not started yet.
    pub fn current_best_fitness(&self) -> f64 {
        self.current_best().get_fitness()
    }

    /// Average fitness of the current population, ignoring individuals with a `NaN` fitness so a
    /// single one doesn't hide the average of the others. Returns `NaN` if no fitness is a number.
    pub fn current_fitness_average(&self) -> f64 {
        self.stats
            .or_else(|| self.population_stats())
            .map_or(f64::NAN, |stats| stats.average_fitness)
    }

    /// Best index and average fitness of the current population, in one pass. `None` when the
    /// population is empty.
    fn population_stats(&self) -> Option<GenerationStats> {
        let mut best: Option<(usize, &T)> = None;
        // Summed in order: a parallel sum adds in an order that depends on thread scheduling,
        // which changes the last bits of the average and so the history of a seeded run.
        let mut sum = 0.0;
        let mut count = 0usize;

        for (index, individual) in self.current_population.iter().enumerate() {
            // `is_ge`, so that the last of equally good individuals wins, as `max_by` would.
            if best.is_none_or(|(_, best)| Self::cmp_by_fitness(individual, best).is_ge()) {
                best = Some((index, individual));
            }

            let fitness = individual.get_fitness();
            if !fitness.is_nan() {
                sum += fitness;
                count += 1;
            }
        }

        best.map(|(best_index, _)| GenerationStats {
            best_index,
            average_fitness: sum / count as f64,
        })
    }

    /// Saves a chart of the best and average fitness per generation to `path` (a PNG), with the
    /// [`title`](Self::title) as its caption. Needs the `plot` feature.
    ///
    /// For another caption, call [`Metrics::plot_chart`] on [`metrics`](Self::metrics).
    #[cfg(feature = "plot")]
    ///
    /// Returns an error if `path` is not valid UTF-8 or the chart can't be written.
    pub fn plot_chart(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let path = path
            .as_ref()
            .to_str()
            .ok_or("the chart path is not valid UTF-8")?;

        self.metrics.plot_chart(path, &self.title)
    }

    /// The title given to
    /// [`EvolutionBuilder::with_title`](crate::EvolutionBuilder::with_title), `""` without one.
    /// `plot_chart`, with the `plot` feature, uses it as the caption of the chart.
    pub fn title(&self) -> &str {
        &self.title
    }

    /// The current generation, 0 for the initial population.
    fn generation(&self) -> u32 {
        self.metrics.iterations.saturating_sub(1)
    }

    /// Bookkeeping at the end of `start` and of every `next`, so that a manual loop sees the
    /// same metrics, best ever and callbacks as `run`.
    fn finish_generation(&mut self) {
        self.stats = self.population_stats();

        let best = self.current_best();
        if self
            .best_ever
            .as_ref()
            .is_none_or(|best_ever| Self::cmp_by_fitness(best, best_ever).is_gt())
        {
            self.best_ever = Some(best.clone());
        }

        self.metrics
            .record(self.current_best_fitness(), self.current_fitness_average());
        self.metrics.end_clock();

        if let Some(on_generation) = &self.on_generation {
            on_generation(&self.report());
        }
    }

    /// The generators of `step` in the current generation.
    fn rngs(&self, step: Step) -> Rngs {
        let seed = self
            .run_seed
            .expect("the evolution is not started, call `start` or `run` first");

        Rngs::for_step(seed, self.metrics.iterations, step)
    }

    /// Clones of the `elitism` best individuals of the current population.
    fn find_elitists(&self) -> Vec<T> {
        let elitism = self.elitism_count();
        if elitism == 0 {
            return Vec::new();
        }

        let mut ranked: Vec<&T> = self.current_population.iter().collect();
        // Partitions so the `elitism` best come first, without sorting the whole population.
        ranked.select_nth_unstable_by(elitism - 1, |a, b| Self::cmp_by_fitness(b, a));

        ranked[..elitism]
            .iter()
            .map(|&elitist| elitist.clone())
            .collect()
    }

    /// Puts the elitists in the place of the worst individuals of the current population.
    fn replace_worsts_with_elitists(&mut self, elitists: Vec<T>) {
        // A custom selection may hand back a smaller pool than the previous population.
        let count = elitists.len().min(self.current_population.len());
        if count == 0 {
            return;
        }
        self.stats = None;

        let population = &self.current_population;
        let mut ranked: Vec<usize> = (0..population.len()).collect();
        // Partitions so the `count` worst come first.
        ranked.select_nth_unstable_by(count - 1, |&a, &b| {
            Self::cmp_by_fitness(&population[a], &population[b])
        });

        for (index, elitist) in ranked.into_iter().zip(elitists) {
            self.current_population[index] = elitist;
        }
    }

    /// Number of elitists that fit in the current population.
    fn elitism_count(&self) -> usize {
        (self.elitism as usize).min(self.current_population.len())
    }

    fn cmp_by_fitness(a: &T, b: &T) -> Ordering {
        fitness_cmp(a.get_fitness(), b.get_fitness())
    }

    fn process_fitness(&mut self) {
        self.stats = None;
        self.metrics.step_start(Steps::Fitness);

        self.fitness
            .calculate_population(&mut self.current_population);

        self.metrics.step_end(Steps::Fitness);
    }
}

#[cfg(test)]
mod tests {
    use crate::fitness::fitness_cmp;
    use crate::prelude::*;
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct SumFitness;

    impl Fitness<Real> for SumFitness {
        fn calculate_fitness(&self, individual: &Real) -> f64 {
            individual.get_chromosome().iter().sum()
        }
    }

    /// Scores every individual whose first gene is negative as `NaN`.
    #[derive(Clone)]
    struct SometimesNanFitness;

    impl Fitness<Real> for SometimesNanFitness {
        fn calculate_fitness(&self, individual: &Real) -> f64 {
            match individual.get_gene(0) {
                gene if gene < 0.0 => f64::NAN,
                gene => gene,
            }
        }
    }

    fn builder<F: Fitness<Real>>(population_size: u32, fitness: F) -> EvolutionBuilder<Real> {
        EvolutionBuilder::new(population_size, 3, (-1.0, 1.0))
            .with_fitness(fitness)
            .with_selection(TournamentSelection::default())
            .with_crossover(DoNothingCrossover)
            .with_mutation(GaussianMutation::default())
            .with_stop_condition(|report| report.generation >= 20)
    }

    #[test]
    fn population_keeps_its_configured_size() {
        let mut evolution = builder(30, SumFitness).build().unwrap();

        evolution.start();
        assert_eq!(evolution.current_population().len(), 30);

        for _ in 0..5 {
            evolution.next();
            assert_eq!(evolution.current_population().len(), 30);
        }
    }

    #[cfg(feature = "plot")]
    #[test]
    fn plot_chart_takes_a_path() {
        let mut evolution = builder(10, SumFitness).with_title("Sum").build().unwrap();
        evolution.run();
        let path =
            std::env::temp_dir().join(format!("evolutionary-{}-evolution.png", std::process::id()));

        evolution.plot_chart(&path).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert!(bytes.starts_with(b"\x89PNG"));
    }

    #[test]
    fn population_borrows_what_current_population_sorts() {
        let mut evolution = builder(30, SumFitness).with_seed(2).build().unwrap();
        assert!(evolution.population().is_empty());

        evolution.run();

        let mut borrowed: Vec<Real> = evolution.population().to_vec();
        borrowed.sort_by(|a, b| fitness_cmp(a.get_fitness(), b.get_fitness()));
        let sorted = evolution.current_population();

        assert_eq!(evolution.population().len(), 30);
        let genes = |population: &[Real]| -> Vec<Vec<f64>> {
            population
                .iter()
                .map(|individual| individual.get_chromosome().to_vec())
                .collect()
        };
        assert_eq!(genes(&borrowed), genes(&sorted));
        assert_eq!(
            sorted.last().unwrap().get_fitness(),
            evolution.current_best_fitness()
        );
    }

    #[test]
    fn elitism_never_loses_the_best() {
        let mut evolution = builder(20, SumFitness).with_elitism(3).build().unwrap();

        evolution.start();
        let mut best = evolution.current_best_fitness();

        for _ in 0..20 {
            evolution.next();
            assert!(evolution.current_best_fitness() >= best);
            best = evolution.current_best_fitness();
        }
    }

    #[test]
    fn elitism_as_big_as_the_population() {
        let mut evolution = builder(5, SumFitness).with_elitism(5).build().unwrap();

        evolution.run();

        assert_eq!(evolution.current_population().len(), 5);
    }

    #[test]
    fn nan_fitness_does_not_panic_and_ranks_last() {
        let mut evolution = builder(50, SometimesNanFitness)
            .with_elitism(2)
            .build()
            .unwrap();

        evolution.run();

        assert!(!evolution.current_best_fitness().is_nan());
        assert!(!evolution.current_fitness_average().is_nan());
    }

    #[test]
    fn average_ignores_nan_fitnesses() {
        let mut evolution = builder(2, SometimesNanFitness).build().unwrap();
        evolution.current_population = vec![
            Real::new(vec![-1.0, 0.0, 0.0], (-1.0, 1.0)),
            Real::new(vec![0.5, 0.0, 0.0], (-1.0, 1.0)),
        ];

        evolution.process_fitness();

        assert_eq!(evolution.current_fitness_average(), 0.5);
        assert_eq!(evolution.current_best_fitness(), 0.5);
    }

    #[test]
    #[should_panic(expected = "call `start` or `run` first")]
    fn current_best_before_start_panics_with_a_clear_message() {
        let evolution = builder(10, SumFitness).build().unwrap();

        evolution.current_best();
    }

    #[test]
    fn on_generation_sees_every_generation_in_order() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let recorder = Arc::clone(&seen);

        let mut evolution = builder(20, SumFitness)
            .with_on_generation(move |report| {
                recorder.lock().unwrap().push((
                    report.generation,
                    report.best_fitness,
                    report.average_fitness,
                ))
            })
            .build()
            .unwrap();

        evolution.run();

        let seen = seen.lock().unwrap();
        let generations: Vec<u32> = seen.iter().map(|&(generation, _, _)| generation).collect();
        assert_eq!(generations, (0..=20).collect::<Vec<u32>>());

        let best: Vec<f64> = seen.iter().map(|&(_, best, _)| best).collect();
        let average: Vec<f64> = seen.iter().map(|&(_, _, average)| average).collect();
        assert_eq!(best, evolution.metrics.best_fitnesses);
        assert_eq!(average, evolution.metrics.avg_fitnesses);
    }

    #[test]
    fn stop_condition_sees_the_report() {
        let mut evolution = builder(20, SumFitness)
            .with_stop_condition(|report| report.generation >= 7)
            .build()
            .unwrap();

        evolution.run();

        assert_eq!(evolution.report().generation, 7);
        // The initial population plus one per generation.
        assert_eq!(evolution.metrics.best_fitnesses.len(), 8);
    }

    /// Best at the origin, so a strong mutation easily throws the best individual away.
    #[derive(Clone)]
    struct NearOrigin;

    impl Fitness<Real> for NearOrigin {
        fn calculate_fitness(&self, individual: &Real) -> f64 {
            -individual
                .get_chromosome()
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
        }
    }

    #[test]
    fn best_ever_survives_without_elitism() {
        let mut evolution = builder(10, NearOrigin)
            .with_elitism(0)
            .with_mutation(GaussianMutation::new(1.0, 1.0))
            .with_seed(1)
            .with_stop_condition(|report| report.generation >= 50)
            .build()
            .unwrap();

        evolution.run();

        let bests = &evolution.metrics.best_fitnesses;
        let best_seen = bests.iter().copied().fold(f64::MIN, f64::max);
        let lost = bests.windows(2).any(|pair| pair[1] < pair[0]);
        let best_ever = evolution.best_ever();

        assert!(lost, "the best was never lost, the test proves nothing");
        assert_eq!(best_ever.get_fitness(), best_seen);
        assert_eq!(NearOrigin.calculate_fitness(best_ever), best_seen);
        assert_eq!(
            evolution.report().best_ever.get_chromosome(),
            best_ever.get_chromosome()
        );
    }

    #[test]
    fn manual_loop_is_a_full_alternative_to_run() {
        let mut evolution = builder(20, SumFitness).with_seed(4).build().unwrap();
        evolution.start();
        while !evolution.reached_stop_condition() {
            evolution.next();
        }

        let mut by_run = builder(20, SumFitness).with_seed(4).build().unwrap();
        by_run.run();

        assert!(evolution.metrics.total_time() > std::time::Duration::ZERO);
        assert_eq!(evolution.report().generation, by_run.report().generation);
        assert_eq!(
            evolution.metrics.best_fitnesses,
            by_run.metrics.best_fitnesses
        );
        assert_eq!(
            evolution.best_ever().get_chromosome(),
            by_run.best_ever().get_chromosome()
        );
    }

    #[test]
    fn start_forgets_the_previous_run() {
        let mut evolution = builder(20, SumFitness)
            .with_stop_condition(|report| report.generation >= 30)
            .build()
            .unwrap();
        evolution.run();

        evolution.start();

        assert_eq!(evolution.report().generation, 0);
        assert_eq!(
            evolution.best_ever().get_chromosome(),
            evolution.current_best().get_chromosome()
        );
    }
}

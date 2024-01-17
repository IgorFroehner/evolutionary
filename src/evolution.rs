use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::sync::Arc;
use std::time::Duration;

use crate::{
    crossover::Crossover,
    fitness::Fitness,
    metrics::{Metrics, Steps},
    mutation::Mutation,
    population::{GeneCod, Individual},
    selection::Selection,
};

pub type StopConditionFn = Arc<dyn Fn(f64, u32, u32) -> bool + Send + Sync>;

#[derive(Clone)]
pub struct EvolutionConfig<T: Individual> {
    pub dimension: u32,
    pub population_size: u32,
    pub range: T::RangeType,
    pub gene_cod: GeneCod,
}

/// This is the struct that does the evolution magic. It has the methods needed to start and iterate
/// through the evolution.
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
/// # let mut evolution = EvolutionBuilder::new(30, 10, GeneCod::Bin, ())
/// #     .with_fitness(YourFitness)
/// #     .with_selection(TournamentSelection::default())
/// #     .with_crossover(NPointsCrossover::default())
/// #     .with_mutation(BitFlipMutation::default())
/// #     .with_title("Max".to_string())
/// #     .with_stop_condition(move |_, iterations, _| iterations >= 1000)
/// #     .build().unwrap();
/// #
/// // You can start the evolution with the `start` method:
/// evolution.start();
///
/// // and iterate through the evolution with the `next` method:
/// evolution.next();
///
/// // or you can run it until the stop condition is met with the `run` method:
/// evolution.run();
/// ```
pub struct Evolution<T: Individual> {
    _title: String,
    current_population: Vec<T>,
    config: EvolutionConfig<T>,
    fitness: Box<dyn Fitness<T>>,
    selection: Box<dyn Selection<T>>,
    crossover: Box<dyn Crossover<T>>,
    mutation: Box<dyn Mutation<T>>,
    elitism: bool,
    stop_condition: StopConditionFn,
    pub metrics: Metrics,
}

impl<T: Individual> Evolution<T> {
    pub fn new(
        title: String,
        config: EvolutionConfig<T>,
        fitness: Box<dyn Fitness<T>>,
        selection: Box<dyn Selection<T>>,
        crossover: Box<dyn Crossover<T>>,
        mutation: Box<dyn Mutation<T>>,
        elitism: bool,
        stop_condition: StopConditionFn,
    ) -> Self {
        Self {
            _title: title,
            current_population: Vec::new(),
            config,
            fitness,
            selection,
            crossover,
            mutation,
            elitism,
            stop_condition,
            metrics: Metrics::new(),
        }
    }

    /// Starts the evolution, generating the initial population and calculating the
    /// fitness of each individual.
    pub fn start(&mut self) {
        self.metrics = Metrics::new();

        self.metrics.start_clock();

        let _ = &self.current_population.clear();

        for _ in 0..self.config.population_size {
            let _ = &self.current_population.push(T::generate_member(
                self.config.dimension,
                &self.config.range,
            ));
        }

        self.process_fitness();

        self.metrics
            .record(self.current_best_fitness(), self.current_fitness_average());
    }

    /// This method runs one generation of the evolution.
    /// It selects the mating pool, crossover, mutate and calculates the fitness of the new population.
    pub fn next(&mut self) {
        let mut current_best_solution = None;
        if self.elitism {
            current_best_solution = Some(self.current_best().clone());
        }

        self.metrics.step_start(Steps::Selection);
        let mut mating_pool = self.selection.get_mating_pool(&self.current_population);
        self.metrics.step_end(Steps::Selection);

        self.metrics.step_start(Steps::Crossover);
        self.crossover.crossover(&mut mating_pool);
        self.metrics.step_end(Steps::Crossover);

        self.metrics.step_start(Steps::Mutation);
        self.mutation.mutate(&mut mating_pool);
        self.metrics.step_end(Steps::Mutation);

        self.current_population = mating_pool;

        self.process_fitness();

        if self.elitism && current_best_solution.is_some() {
            let worst_index = self
                .current_population
                .par_iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| Self::cmp_by_fitness(a, b))
                .unwrap()
                .0;

            self.current_population[worst_index] = current_best_solution.unwrap();
        }

        self.metrics
            .record(self.current_best_fitness(), self.current_fitness_average());
    }

    /// This method runs the evolution, generation over generation, until the stop condition is met.
    pub fn run(&mut self) {
        self.start();

        while !self.reached_stop_condition() {
            self.next();
        }

        self.metrics.end_clock();
    }

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
    /// ---------------------------------------
    /// ```
    pub fn time_digest(&self) {
        println!("------------ Time Digest ------------");
        println!(
            "Total time: {:?}",
            Duration::from_nanos(self.metrics.total_time() as u64)
        );
        println!(
            "Selection time: {:?} ({:.2}%)",
            self.metrics.step_time(Steps::Selection).unwrap(),
            self.metrics.step_time(Steps::Selection).unwrap().as_nanos() as f64
                / self.metrics.total_time() as f64
                * 100.0
        );
        println!(
            "Crossover time: {:?} ({:.2}%)",
            self.metrics.step_time(Steps::Crossover).unwrap(),
            self.metrics.step_time(Steps::Crossover).unwrap().as_nanos() as f64
                / self.metrics.total_time() as f64
                * 100.0
        );
        println!(
            "Mutation time: {:?} ({:.2}%)",
            self.metrics.step_time(Steps::Mutation).unwrap(),
            self.metrics.step_time(Steps::Mutation).unwrap().as_nanos() as f64
                / self.metrics.total_time() as f64
                * 100.0
        );
        println!(
            "Fitness time: {:?} ({:.2}%)",
            self.metrics.step_time(Steps::Fitness).unwrap(),
            self.metrics.step_time(Steps::Fitness).unwrap().as_nanos() as f64
                / self.metrics.total_time() as f64
                * 100.0
        );
        println!("---------------------------------------");
    }

    /// Returns the best individual of the current population.
    pub fn current_best(&self) -> &T {
        self.current_population
            .par_iter()
            .max_by(|a, b| Self::cmp_by_fitness(a, b))
            .unwrap()
    }

    /// Returns a copy of the current population sorted by fitness.
    pub fn current_population(&self) -> Vec<T> {
        let mut current_population = self.current_population.clone();

        current_population.sort_by(|a, b| Self::cmp_by_fitness(a, b));

        current_population
    }

    /// Returns if the stop condition was already met for this evolution object.
    pub fn reached_stop_condition(&self) -> bool {
        (self.stop_condition)(
            self.current_best_fitness(),
            self.metrics.iterations,
            self.metrics.gens_without_improvement,
        )
    }

    pub fn current_best_fitness(&self) -> f64 {
        self.current_best().get_fitness()
    }

    pub fn current_fitness_average(&self) -> f64 {
        let sum: f64 = self
            .current_population
            .par_iter()
            .map(|individual| individual.get_fitness())
            .sum();

        sum / self.config.population_size as f64
    }

    pub fn plot_chart(
        &self,
        path: impl Into<String>,
        test_name: impl Into<String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.metrics.plot_chart(&path.into(), &test_name.into())
    }

    fn calculate_fitness(&self, individual: &T) -> f64 {
        self.fitness.calculate_fitness(&individual)
    }

    fn cmp_by_fitness(a: &T, b: &T) -> std::cmp::Ordering {
        a.get_fitness().partial_cmp(&b.get_fitness()).unwrap()
    }

    fn process_fitness(&mut self) {
        self.metrics.step_start(Steps::Fitness);

        let fitness_values: Vec<f64> = self
            .current_population
            .par_iter()
            .map(|individual| self.calculate_fitness(individual))
            .collect();

        for (individual, fitness) in self.current_population.iter_mut().zip(fitness_values) {
            individual.set_fitness(fitness);
        }

        self.metrics.step_end(Steps::Fitness);
    }
}

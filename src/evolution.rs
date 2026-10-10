use rayon::iter::{IntoParallelIterator, IntoParallelRefMutIterator};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use std::cmp::Ordering;
use std::sync::Arc;
use std::time::Duration;

use crate::{
    crossover::Crossover,
    fitness::{fitness_cmp, Fitness},
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
    config: EvolutionConfig<T>,
    fitness: Box<dyn Fitness<T>>,
    selection: Box<dyn Selection<T>>,
    crossover: Box<dyn Crossover<T>>,
    mutation: Box<dyn Mutation<T>>,
    elitism: u32,
    stop_condition: StopConditionFn,
    pub metrics: Metrics,
    current_population: Vec<T>,
}

impl<T: Individual> Evolution<T> {
    pub fn new(
        title: String,
        config: EvolutionConfig<T>,
        fitness: Box<dyn Fitness<T>>,
        selection: Box<dyn Selection<T>>,
        crossover: Box<dyn Crossover<T>>,
        mutation: Box<dyn Mutation<T>>,
        elitism: u32,
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

        self.current_population = (0..self.config.population_size)
            .into_par_iter()
            .map(|_| T::generate_member(self.config.dimension, &self.config.range))
            .collect();

        self.process_fitness();

        self.metrics
            .record(self.current_best_fitness(), self.current_fitness_average());
    }

    /// This method runs one generation of the evolution.
    /// It selects the mating pool, crossover, mutate and calculates the fitness of the new population.
    pub fn next(&mut self) {
        self.metrics.step_start(Steps::Elitism);
        let elitists = self.find_elitists();
        self.metrics.step_end(Steps::Elitism);

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

        self.metrics.step_start(Steps::Elitism);
        self.replace_worsts_with_elitists(elitists);
        self.metrics.step_end(Steps::Elitism);

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
        println!(
            "Elitism time: {:?} ({:.2}%)",
            self.metrics.step_time(Steps::Elitism).unwrap(),
            self.metrics.step_time(Steps::Elitism).unwrap().as_nanos() as f64
                / self.metrics.total_time() as f64
                * 100.0
        );
        println!("---------------------------------------");
    }

    /// Returns the best individual of the current population. An individual with a `NaN` fitness
    /// is only returned if every fitness is `NaN`.
    ///
    /// # Panics
    ///
    /// If the evolution was not started yet, as there is no population.
    pub fn current_best(&self) -> &T {
        self.current_population
            .par_iter()
            .max_by(|a, b| Self::cmp_by_fitness(a, b))
            .expect("the population is empty, call `start` or `run` first")
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

    /// Average fitness of the current population, ignoring individuals with a `NaN` fitness so a
    /// single one doesn't hide the average of the others. Returns `NaN` if no fitness is a number.
    pub fn current_fitness_average(&self) -> f64 {
        let (sum, count) = self
            .current_population
            .par_iter()
            .map(|individual| individual.get_fitness())
            .filter(|fitness| !fitness.is_nan())
            .map(|fitness| (fitness, 1usize))
            .reduce(|| (0.0, 0), |(s1, c1), (s2, c2)| (s1 + s2, c1 + c2));

        sum / count as f64
    }

    pub fn plot_chart(
        &self,
        path: impl Into<String>,
        test_name: impl Into<String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.metrics.plot_chart(&path.into(), &test_name.into())
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

    fn calculate_individual_fitness(fitness: &dyn Fitness<T>, individual: &mut T) -> f64 {
        let fitness_value = fitness.calculate_fitness(&individual);
        individual.set_fitness(fitness_value);
        fitness_value
    }

    fn cmp_by_fitness(a: &T, b: &T) -> Ordering {
        fitness_cmp(a.get_fitness(), b.get_fitness())
    }

    fn process_fitness(&mut self) {
        self.metrics.step_start(Steps::Fitness);

        self.current_population
            .par_iter_mut()
            .for_each(|individual| {
                Self::calculate_individual_fitness(self.fitness.as_ref(), individual);
            });

        self.metrics.step_end(Steps::Fitness);
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

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
        EvolutionBuilder::new(population_size, 3, GeneCod::Real, (-1.0, 1.0))
            .with_fitness(fitness)
            .with_selection(TournamentSelection::default())
            .with_crossover(DoNothingCrossover)
            .with_mutation(GaussianMutation::default())
            .with_stop_condition(|_, iterations, _| iterations >= 20)
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
}

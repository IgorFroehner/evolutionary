use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::sync::Arc;

use crate::{
    coding::Coding,
    crossover::Crossover,
    fitness::Fitness,
    mutation::Mutation,
    plot_evolution::{Metrics, Steps},
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

/// This is the struct used to do the evolution. It has the methods needed to start and iterate
/// through the evolution.
pub struct Evolution<T: Individual, C: Coding<T>> {
    pub title: String,
    pub current_population: Vec<T>,
    pub config: EvolutionConfig<T>,
    pub fitness: Box<dyn Fitness<T>>,
    pub selection: Box<dyn Selection<T>>,
    pub crossover: Box<dyn Crossover<T>>,
    pub mutation: Box<dyn Mutation<T>>,
    pub coding: Box<C>,
    pub elitism: bool,
    pub stop_condition: StopConditionFn,
    pub metrics: Metrics,
}

impl<T: Individual, C: Coding<T>> Evolution<T, C> {
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

    pub fn next(&mut self) {
        let mut current_best_solution = None;
        if self.elitism {
            current_best_solution = Some(self.current_best().clone());
        }

        let mut mating_pool = self.selection.get_mating_pool(&self.current_population);

        self.crossover.crossover(&mut mating_pool);

        self.mutation.mutate(&mut mating_pool);

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

    pub fn run(&mut self) {
        self.start();

        while !(self.stop_condition)(
            self.current_best_fitness(),
            self.metrics.iterations,
            self.metrics.gens_without_improvement,
        ) {
            self.next();
        }

        self.metrics.end_clock();
    }

    pub fn calculate_individual_fitness(&self, index: usize) -> f64 {
        let individual = &self.current_population[index];

        self.calculate_fitness(&individual)
    }

    pub fn population_digest(&self) {
        println!("---------------------------------------------");
        println!("Iteration: {}", self.metrics.iterations);
        println!("Best Fitness: {}", self.current_best_fitness());
        println!("Current Average: {}", self.current_fitness_average());
    }

    pub fn time_digest(&self) {
        println!("---------------------------------------------");
        println!("Total time: {:?}", self.metrics.get_elapsed_time());
        println!(
            "Selection time: {:?}",
            self.metrics.step_times[&Steps::Selection].2
        );
        println!(
            "Crossover time: {:?}",
            self.metrics.step_times[&Steps::Crossover].2
        );
        println!(
            "Mutation time: {:?}",
            self.metrics.step_times[&Steps::Mutation].2
        );
        println!(
            "Fitness time: {:?}",
            self.metrics.step_times[&Steps::Fitness].2
        );
    }

    pub fn current_best(&self) -> &T {
        self.current_population
            .par_iter()
            .max_by(|a, b| Self::cmp_by_fitness(a, b))
            .unwrap()
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

    fn calculate_fitness(&self, individual: &T) -> f64 {
        self.fitness.calculate_fitness(&individual)
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
        path: &String,
        test_name: &String,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.metrics.plot_chart(&path, test_name)
    }

    fn cmp_by_fitness(a: &T, b: &T) -> std::cmp::Ordering {
        a.get_fitness().partial_cmp(&b.get_fitness()).unwrap()
    }
}

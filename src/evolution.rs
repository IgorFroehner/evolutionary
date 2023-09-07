use crate::{
    crossover::Crossover,
    fitness::Fitness,
    mutation::Mutation,
    plot_evolution::Metrics,
    population::{GeneCod, Individual},
    selection::Selection,
};

pub struct Evolution<T: Individual> {
    pub title: String,
    pub dimension: u32,
    pub population_size: u32,
    pub range: T::RangeType,
    pub current_population: Vec<T>,
    pub gene_cod: GeneCod,
    pub fitness: Box<dyn Fitness<T>>,
    pub selection: Box<dyn Selection<T>>,
    pub crossover: Box<dyn Crossover<T>>,
    pub mutation: Box<dyn Mutation<T>>,
    pub stop_condition: Box<dyn Fn(f64, u32) -> bool>,
    pub metrics: Metrics,
}

impl<T: Individual> Evolution<T> {
    pub fn start(&mut self) {
        self.metrics = Metrics::new(self.title.clone());

        let _ = &self.current_population.clear();

        for _ in 0..self.population_size {
            let _ = &self
                .current_population
                .push(T::generate_member(self.dimension, &self.range));
        }

        self.process_fitness();

        self.metrics
            .record(self.current_best_fitness(), self.current_fitness_average());
    }

    pub fn next(&mut self) {
        let current_best_solution = self.current_best().clone();

        let mut mating_pool = self.selection.get_mating_pool(&self.current_population);

        self.crossover.crossover(&mut mating_pool);

        self.mutation.mutate(&mut mating_pool);

        self.process_fitness();

        let worst_index = mating_pool
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.get_fitness().partial_cmp(&b.get_fitness()).unwrap())
            .unwrap()
            .0;

        mating_pool[worst_index] = current_best_solution;

        self.current_population = mating_pool;

        self.process_fitness();

        self.metrics
            .record(self.current_best_fitness(), self.current_fitness_average());
    }

    pub fn run(&mut self) {
        self.start();

        while !(self.stop_condition)(self.current_best_fitness(), self.metrics.iterations) {
            self.next();
        }
    }

    pub fn individual_fitness(&self, index: usize) -> f64 {
        let individual = &self.current_population[index];

        self.calculate_fitness(&individual)
    }

    pub fn population_digest(&self) {
        println!("---------------------------------------------");
        println!("Iteration: {}", self.metrics.iterations);
        println!("Best Fitness: {}", self.current_best_fitness());
        println!("Current Average: {}", self.current_fitness_average());
    }

    pub fn current_best(&self) -> &T {
        self.current_population
            .iter()
            .max_by(|a, b| a.get_fitness().partial_cmp(&b.get_fitness()).unwrap())
            .unwrap()
    }

    fn process_fitness(&mut self) {
        let fitness_values: Vec<f64> = self
            .current_population
            .iter()
            .map(|individual| self.calculate_fitness(individual))
            .collect();

        for (individual, fitness) in self.current_population.iter_mut().zip(fitness_values) {
            individual.set_fitness(fitness);
        }
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
            .iter()
            .map(|individual| individual.get_fitness())
            .sum();

        sum / self.population_size as f64
    }

    pub fn plot_chart(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.metrics.plot_chart()
    }
}

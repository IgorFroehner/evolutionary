use crate::{
    crossover::Crossover,
    evolution::{Evolution, StopConditionFn},
    fitness::Fitness,
    mutation::Mutation,
    plot_evolution::Metrics,
    population::{GeneCod, Individual},
    selection::Selection,
};

pub struct EvolutionBuilder<T: Individual> {
    title: Option<String>,
    range: T::RangeType,
    population_size: u32,
    dimension: u32,
    population_code: GeneCod,
    fitness: Option<Box<dyn Fitness<T>>>,
    selection: Option<Box<dyn Selection<T>>>,
    crossover: Option<Box<dyn Crossover<T>>>,
    mutation: Option<Box<dyn Mutation<T>>>,
    stop_condition: Option<StopConditionFn>,
}

impl<T: Individual> EvolutionBuilder<T> {
    pub fn new(
        population_size: u32,
        dimension: u32,
        population_code: GeneCod,
        range: T::RangeType,
    ) -> Self {
        Self {
            title: None,
            range,
            dimension,
            population_code,
            population_size,
            fitness: None,
            selection: None,
            crossover: None,
            mutation: None,
            stop_condition: None,
        }
    }

    pub fn set_fitness<F: Fitness<T>>(mut self, f: F) -> Self {
        self.fitness = Some(Box::new(f));
        self
    }

    pub fn set_selection<S: Selection<T>>(mut self, s: S) -> Self {
        self.selection = Some(Box::new(s));
        self
    }

    pub fn set_crossover<C: Crossover<T>>(mut self, c: C) -> Self {
        self.crossover = Some(Box::new(c));
        self
    }

    pub fn set_mutation<M: Mutation<T>>(mut self, m: M) -> Self {
        self.mutation = Some(Box::new(m));
        self
    }

    pub fn set_stop_condition<F: Fn(f64, u32) -> bool + 'static + Send + Sync>(
        mut self,
        f: F,
    ) -> Self {
        self.stop_condition = Some(Box::new(f));
        self
    }

    pub fn set_title(mut self, title: String) -> Self {
        self.title = Some(title);
        self
    }

    pub fn build(self) -> Result<Evolution<T>, String> {
        let title = self.title.unwrap_or("".to_string());

        match (
            self.fitness,
            self.selection,
            self.crossover,
            self.mutation,
            self.stop_condition,
        ) {
            (Some(f), Some(s), Some(c), Some(m), Some(stop)) => Ok(Evolution {
                title: title.clone(),
                dimension: self.dimension,
                population_size: self.population_size,
                range: self.range,
                current_population: Vec::new(),
                gene_cod: self.population_code,
                fitness: dyn_clone::clone_box(&*f),
                selection: dyn_clone::clone_box(&*s),
                crossover: dyn_clone::clone_box(&*c),
                mutation: dyn_clone::clone_box(&*m),
                stop_condition: stop,
                metrics: Metrics::new(title),
            }),
            (_, _, _, _, _) => Err("Not all fields needed are filled".to_string()),
        }
    }
}

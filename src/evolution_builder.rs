use std::sync::Arc;
use crate::{
    crossover::Crossover,
    evolution::{Evolution, StopConditionFn, EvolutionConfig},
    fitness::Fitness,
    mutation::Mutation,
    plot_evolution::Metrics,
    population::{GeneCod, Individual},
    selection::Selection, coding::Coding,
};

pub struct EvolutionBuilder<T: Individual, C: Coding<T>> {
    title: Option<String>,
    pub evolution_config: Option<EvolutionConfig<T>>,
    fitness: Option<Box<dyn Fitness<T>>>,
    selection: Option<Box<dyn Selection<T>>>,
    crossover: Option<Box<dyn Crossover<T>>>,
    mutation: Option<Box<dyn Mutation<T>>>,
    coding: Option<Box<C>>,
    elitism: Option<bool>,
    stop_condition: Option<StopConditionFn>,
}

impl<T: Individual, C: Coding<T>> EvolutionBuilder<T, C> {
    pub fn new(
        population_size: u32,
        dimension: u32,
        gene_cod: GeneCod,
        range: T::RangeType,
    ) -> Self {
        let evolution_config = Some(EvolutionConfig {
            range,
            dimension,
            gene_cod,
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
            coding: None,
            elitism: None,
        }
    }

    pub fn from_config(config: EvolutionConfig<T>) -> Self {
        Self {
            title: None,
            fitness: None,
            selection: None,
            crossover: None,
            mutation: None,
            stop_condition: None,
            coding: None,
            evolution_config: Some(config.clone()),
            elitism: None,
        }
    }

    pub fn with_fitness<F: Fitness<T>>(mut self, f: F) -> Self {
        self.fitness = Some(Box::new(f));
        self
    }

    pub fn with_selection<S: Selection<T>>(mut self, s: S) -> Self {
        self.selection = Some(Box::new(s));
        self
    }

    pub fn with_crossover<X: Crossover<T>>(mut self, c: X) -> Self {
        self.crossover = Some(Box::new(c));
        self
    }

    pub fn with_mutation<M: Mutation<T>>(mut self, m: M) -> Self {
        self.mutation = Some(Box::new(m));
        self
    }

    pub fn with_stop_condition<F: Fn(f64, u32) -> bool + 'static + Send + Sync>(
        mut self,
        f: F,
    ) -> Self {
        self.stop_condition = Some(Arc::new(f));
        self
    }

    pub fn with_coding(mut self, c: C) -> Self {
        self.coding = Some(Box::new(c));
        self
    }

    pub fn with_elitism(mut self, elitism: bool) -> Self {
        self.elitism = Some(elitism);
        self
    }

    pub fn with_title(mut self, title: String) -> Self {
        self.title = Some(title);
        self
    }

    pub fn build(&self) -> Result<Evolution<T, C>, String> {
        let title = self.title.clone().unwrap_or("".to_string());

        let evolution_config = self.evolution_config.clone().ok_or("No config provided")?;

        if let (Some(f), Some(s), Some(x), Some(m), Some(c)) = (
            self.fitness.as_ref().map(|f| f.as_ref().clone()),
            self.selection.as_ref().map(|s| s.as_ref().clone()),
            self.crossover.as_ref().map(|c| c.as_ref().clone()),
            self.mutation.as_ref().map(|m| m.as_ref().clone()),
            &self.coding,
        ) {
            Ok(Evolution {
                config: evolution_config,
                title: title.clone(),
                current_population: Vec::new(),
                fitness: dyn_clone::clone_box(&*f),
                selection: dyn_clone::clone_box(&*s),
                crossover: dyn_clone::clone_box(&*x),
                mutation: dyn_clone::clone_box(&*m),
                coding: dyn_clone::clone_box(&*c),
                elitism: self.elitism.unwrap_or(true),
                stop_condition: Arc::clone(self.stop_condition.as_ref().unwrap()),
                metrics: Metrics::new(),
            })
        } else {
            Err("Missing required parameters".to_string())
        }
    }
}

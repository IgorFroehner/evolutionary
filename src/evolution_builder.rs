use crate::{
    crossover::Crossover,
    evolution::{Evolution, EvolutionConfig, StopConditionFn},
    fitness::Fitness,
    mutation::Mutation,
    population::{GeneCod, Individual},
    selection::Selection,
};
use std::sync::Arc;

/// This is the helper struct to create a new Evolution object. The `fitness`, `selection`,
/// `crossover`, `mutation` and `stop_condition` and coding are required.
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
///     let mut evolution = EvolutionBuilder::new(30, 10, GeneCod::Bin, ())
///         .with_fitness(YourFitness)
///         .with_selection(TournamentSelection::default())
///         .with_crossover(NPointsCrossover::default())
///         .with_mutation(BitSwapMutation::default())
///         .with_title("Max".to_string())
///         .with_stop_condition(move |_, iterations, _| iterations >= 1000)
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
    elitism: Option<bool>,
    stop_condition: Option<StopConditionFn>,
}

impl<T: Individual> EvolutionBuilder<T> {
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
            elitism: None,
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
            coding: None,
            evolution_config: Some(config.clone()),
            elitism: None,
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

    /// Sets the stop condition. Receives a closure that receives the best fitness, the current
    /// iteration and the number of generations without improvement and returns a boolean.
    ///
    /// # Example
    ///
    /// ```text
    /// evolution_builder.with_stop_condition(|_, iterations, _| iterations >= 1000)
    /// ```
    ///
    pub fn with_stop_condition<F: Fn(f64, u32, u32) -> bool + 'static + Send + Sync>(
        mut self,
        f: F,
    ) -> Self {
        self.stop_condition = Some(Arc::new(f));
        self
    }

    /// Whether or not to use elitism. Defaults to `true`.
    pub fn with_elitism(mut self, elitism: bool) -> Self {
        self.elitism = Some(elitism);
        self
    }

    /// Sets the title of the evolution to use when plotting. Defaults to `""`.
    pub fn with_title(mut self, title: String) -> Self {
        self.title = Some(title);
        self
    }

    pub fn build(&self) -> Result<Evolution<T>, String> {
        let title = self.title.clone().unwrap_or("".to_string());

        let evolution_config = self.evolution_config.clone().ok_or("No config provided")?;

        if let (Some(f), Some(s), Some(x), Some(m)) = (
            self.fitness.as_ref().map(|f| f.as_ref()),
            self.selection.as_ref().map(|s| s.as_ref()),
            self.crossover.as_ref().map(|c| c.as_ref()),
            self.mutation.as_ref().map(|m| m.as_ref()),
        ) {
            Ok(Evolution::new(
                title.clone(),
                evolution_config,
                dyn_clone::clone_box(&*f),
                dyn_clone::clone_box(&*s),
                dyn_clone::clone_box(&*x),
                dyn_clone::clone_box(&*m),
                self.elitism.unwrap_or(true),
                Arc::clone(self.stop_condition.as_ref().unwrap()),
            ))
        } else {
            Err("Missing required parameters".to_string())
        }
    }
}

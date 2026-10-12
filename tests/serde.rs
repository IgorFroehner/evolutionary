//! Saving and loading individuals with the `serde` feature.
#![cfg(feature = "serde")]

use evolutionary::prelude::*;
use serde::{de::DeserializeOwned, Serialize};

fn round_trip<T: Serialize + DeserializeOwned>(value: &T) -> T {
    serde_json::from_str(&serde_json::to_string(value).unwrap()).unwrap()
}

fn with_fitness<T: Individual>(mut individual: T, fitness: f64) -> T {
    individual.set_fitness(fitness);
    individual
}

#[test]
fn bin_round_trips() {
    let saved = with_fitness(Bin::new(vec![true, false, true]), 2.0);

    let loaded = round_trip(&saved);

    assert_eq!(loaded.get_chromosome(), saved.get_chromosome());
    assert_eq!(loaded.get_fitness(), 2.0);
}

#[test]
fn int_round_trips_with_its_range() {
    let saved = with_fitness(Int::new(vec![-3, 0, 7], (-5, 10)), 4.0);

    let loaded = round_trip(&saved);

    assert_eq!(loaded.get_chromosome(), saved.get_chromosome());
    assert_eq!(loaded.range, (-5, 10));
    assert_eq!(loaded.get_fitness(), 4.0);
}

#[test]
fn real_round_trips_with_its_range() {
    let saved = with_fitness(Real::new(vec![-0.25, 0.5, 1e-9], (-1.0, 1.0)), 0.75);

    let loaded = round_trip(&saved);

    assert_eq!(loaded.get_chromosome(), saved.get_chromosome());
    assert_eq!(loaded.range, (-1.0, 1.0));
    assert_eq!(loaded.get_fitness(), 0.75);
}

#[test]
fn int_perm_round_trips() {
    let saved = with_fitness(IntPerm::new(vec![3, 1, 2]), 1.0);

    let loaded = round_trip(&saved);

    assert_eq!(loaded.get_chromosome(), saved.get_chromosome());
    assert_eq!(loaded.get_fitness(), 1.0);
}

#[test]
fn gene_cod_round_trips() {
    assert!(matches!(round_trip(&GeneCod::IntPerm), GeneCod::IntPerm));
}

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
fn a_saved_population_resumes_a_run() {
    let builder = || {
        EvolutionBuilder::new(30, 8, (-1.0, 1.0))
            .with_fitness(NearOrigin)
            .with_selection(TournamentSelection::default())
            .with_crossover(BlendCrossover::default())
            .with_mutation(GaussianMutation::new(0.1, 0.1))
            .with_stop_condition(|report| report.generation >= 20)
            .with_seed(9)
    };
    let mut first = builder().build().unwrap();
    first.run();

    let saved = serde_json::to_string(&first.current_population()).unwrap();
    let loaded: Vec<Real> = serde_json::from_str(&saved).unwrap();

    let mut resumed = builder().with_initial_population(loaded).build().unwrap();
    resumed.start();

    assert_eq!(resumed.current_best_fitness(), first.current_best_fitness());
    assert_eq!(
        resumed.current_best().get_chromosome(),
        first.current_best().get_chromosome()
    );
}

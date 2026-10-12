//! `Fitness::calculate_population` and `EvolutionBuilder::with_reevaluate_elitists`.
//!
//! That the default `calculate_population` keeps every seeded run as it was is checked by
//! `tests/golden.rs`, whose runs all use it.

use evolutionary::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const POPULATION: u32 = 20;
const DIMENSION: u32 = 4;
const GENERATIONS: u32 = 10;
const ELITISM: u32 = 2;

fn sum(individual: &Real) -> f64 {
    individual.get_chromosome().iter().sum()
}

fn builder<F: Fitness<Real>>(fitness: F) -> EvolutionBuilder<Real> {
    EvolutionBuilder::new(POPULATION, DIMENSION, (-1.0, 1.0))
        .with_fitness(fitness)
        .with_selection(TournamentSelection::default())
        .with_crossover(UniformCrossover::default())
        .with_mutation(GaussianMutation::default())
        .with_elitism(ELITISM)
        .with_seed(3)
        .with_stop_condition(|report| report.generation >= GENERATIONS)
}

/// Evaluates the population in one sequential pass and counts its calls and the individuals it
/// was given.
#[derive(Clone, Default)]
struct CountingBatch {
    calls: Arc<AtomicUsize>,
    individuals: Arc<AtomicUsize>,
}

impl Fitness<Real> for CountingBatch {
    fn calculate_fitness(&self, _: &Real) -> f64 {
        panic!("the evolution must go through `calculate_population`")
    }

    fn calculate_population(&self, population: &mut [Real]) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.individuals
            .fetch_add(population.len(), Ordering::Relaxed);

        for individual in population {
            individual.set_fitness(sum(individual));
        }
    }
}

#[derive(Clone)]
struct Sum;

impl Fitness<Real> for Sum {
    fn calculate_fitness(&self, individual: &Real) -> f64 {
        sum(individual)
    }
}

/// What two runs must agree on: the final population, with its fitnesses, and the fitness
/// history.
#[derive(Debug, PartialEq)]
struct Outcome {
    population: Vec<(Vec<f64>, f64)>,
    best_fitnesses: Vec<f64>,
    avg_fitnesses: Vec<f64>,
}

fn outcome(builder: &EvolutionBuilder<Real>) -> Outcome {
    let mut evolution = builder.build().unwrap();
    evolution.run();

    Outcome {
        population: evolution
            .population()
            .iter()
            .map(|individual| {
                (
                    individual.get_chromosome().to_vec(),
                    individual.get_fitness(),
                )
            })
            .collect(),
        best_fitnesses: evolution.metrics.best_fitnesses.clone(),
        avg_fitnesses: evolution.metrics.avg_fitnesses.clone(),
    }
}

#[test]
fn an_overridden_calculate_population_is_called_once_per_generation() {
    let fitness = CountingBatch::default();
    let mut evolution = builder(fitness.clone()).build().unwrap();

    evolution.run();

    let generations = GENERATIONS as usize + 1;
    assert_eq!(fitness.calls.load(Ordering::Relaxed), generations);
    assert_eq!(
        fitness.individuals.load(Ordering::Relaxed),
        POPULATION as usize * generations
    );
    assert_eq!(
        evolution.current_best_fitness(),
        sum(evolution.current_best())
    );
}

#[test]
fn reevaluating_the_elitists_keeps_one_call_per_generation() {
    let fitness = CountingBatch::default();
    let mut evolution = builder(fitness.clone())
        .with_reevaluate_elitists(true)
        .build()
        .unwrap();

    evolution.run();

    let generations = GENERATIONS as usize + 1;
    assert_eq!(fitness.calls.load(Ordering::Relaxed), generations);
    // The initial population has no elitists yet; every later generation evaluates them again.
    assert_eq!(
        fitness.individuals.load(Ordering::Relaxed),
        POPULATION as usize * generations + ELITISM as usize * GENERATIONS as usize
    );
    assert_eq!(evolution.population().len(), POPULATION as usize);
}

#[test]
fn a_batched_fitness_replays_the_default_one() {
    assert_eq!(
        outcome(&builder(CountingBatch::default())),
        outcome(&builder(Sum))
    );
}

#[test]
fn reevaluating_a_deterministic_fitness_changes_nothing() {
    assert_eq!(
        outcome(&builder(Sum).with_reevaluate_elitists(true)),
        outcome(&builder(Sum))
    );
}

const LUCKY: f64 = 1000.0;

/// Scores the first individual it evaluates `LUCKY`, and every other one, the same individual
/// included, by its genes and the number of the call, so no two calls agree. A stand-in for a
/// noisy fitness where one evaluation was very lucky.
#[derive(Clone, Default)]
struct OneLuckyCall {
    calls: Arc<AtomicUsize>,
}

impl Fitness<Real> for OneLuckyCall {
    fn calculate_fitness(&self, individual: &Real) -> f64 {
        match self.calls.fetch_add(1, Ordering::Relaxed) {
            0 => LUCKY,
            call => sum(individual) + (call % 7) as f64 / 10.0,
        }
    }
}

/// Runs the evolution and returns it with the chromosome that got the lucky evaluation.
fn run_with_one_lucky_call(reevaluate: bool) -> (Evolution<Real>, Vec<f64>) {
    let mut evolution = builder(OneLuckyCall::default())
        .with_elitism(1)
        .with_reevaluate_elitists(reevaluate)
        .build()
        .unwrap();

    evolution.start();
    assert_eq!(evolution.current_best_fitness(), LUCKY);
    let lucky = evolution.current_best().get_chromosome().to_vec();

    while !evolution.reached_stop_condition() {
        evolution.next();
    }

    (evolution, lucky)
}

#[test]
fn without_reevaluation_the_elitist_keeps_its_fitness() {
    let (evolution, lucky) = run_with_one_lucky_call(false);

    assert!(evolution
        .metrics
        .best_fitnesses
        .iter()
        .all(|&best| best == LUCKY));
    assert_eq!(evolution.current_best().get_chromosome(), lucky);
}

#[test]
fn with_reevaluation_a_stale_best_does_not_survive() {
    let (evolution, lucky) = run_with_one_lucky_call(true);

    let bests = &evolution.metrics.best_fitnesses;
    assert_eq!(bests[0], LUCKY);
    assert!(
        bests[1..].iter().all(|&best| best < LUCKY),
        "the lucky fitness survived: {bests:?}"
    );
    assert!(evolution
        .population()
        .iter()
        .all(|individual| individual.get_fitness() < LUCKY));
    // The best ever is a record of the best fitness computed, lucky or not.
    assert_eq!(evolution.best_ever().get_fitness(), LUCKY);
    assert_eq!(evolution.best_ever().get_chromosome(), lucky);
}

#[test]
fn a_reevaluated_elitist_survives_with_its_new_fitness() {
    let mut evolution = builder(OneLuckyCall::default())
        .with_elitism(1)
        .with_reevaluate_elitists(true)
        .build()
        .unwrap();
    evolution.start();
    let lucky = evolution.current_best().get_chromosome().to_vec();

    evolution.next();

    let elitist = evolution
        .population()
        .iter()
        .find(|individual| individual.get_chromosome() == lucky)
        .expect("the elitist was dropped");
    assert!(elitist.get_fitness() < LUCKY);
}

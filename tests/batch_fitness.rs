//! `Fitness::calculate_population` and `EvolutionBuilder::with_reevaluate_elitists`.
//!
//! That the default `calculate_population` keeps every seeded run as it was is checked by
//! `tests/golden.rs`, whose runs all use it.

use evolutionary::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

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
///
/// The counters are behind an `Arc` so the test can read them after the run. That is fine for
/// counting, and NOT the pattern for a generation count that feeds noise: every evolution, and
/// every run of an `ExperimentRunner`, would advance the same count, and seeded runs would stop
/// replaying. Keep such a count in the fitness value itself, copied by `Clone` (see the
/// `Fitness` docs).
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

/// Records the chromosomes of every `calculate_population` call, in the order it got them, and
/// scores them as `Sum` does. Behind an `Arc` only so that the test can read them: see
/// `CountingBatch`.
#[derive(Clone, Default)]
struct Recording {
    calls: Arc<Mutex<Vec<Vec<Vec<f64>>>>>,
}

impl Recording {
    fn calls(&self) -> Vec<Vec<Vec<f64>>> {
        self.calls.lock().unwrap().clone()
    }
}

impl Fitness<Real> for Recording {
    fn calculate_fitness(&self, _: &Real) -> f64 {
        panic!("the evolution must go through `calculate_population`")
    }

    fn calculate_population(&self, population: &mut [Real]) {
        let chromosomes = population
            .iter()
            .map(|individual| individual.get_chromosome().to_vec())
            .collect();
        self.calls.lock().unwrap().push(chromosomes);

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

/// Chromosomes sorted, to compare two groups of individuals whatever their order.
fn sorted(mut chromosomes: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    chromosomes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    chromosomes
}

#[test]
fn the_new_individuals_come_first_then_the_elitists() {
    let off = Recording::default();
    let mut evolution = builder(off.clone()).build().unwrap();
    evolution.run();

    let on = Recording::default();
    let mut evolution = builder(on.clone())
        .with_reevaluate_elitists(true)
        .build()
        .unwrap();
    evolution.start();
    let mut before = vec![evolution.population().to_vec()];
    while !evolution.reached_stop_condition() {
        evolution.next();
        before.push(evolution.population().to_vec());
    }

    let (off, on) = (off.calls(), on.calls());
    assert_eq!(on.len(), GENERATIONS as usize + 1);
    assert_eq!(on[0], off[0]);
    for generation in 1..on.len() {
        let (new, elitists) = on[generation].split_at(POPULATION as usize);
        // The same new individuals at the same positions as with re-evaluation off.
        assert_eq!(new, off[generation].as_slice());

        let mut previous = before[generation - 1].clone();
        previous.sort_by(|a, b| b.get_fitness().total_cmp(&a.get_fitness()));
        let best: Vec<Vec<f64>> = previous[..ELITISM as usize]
            .iter()
            .map(|individual| individual.get_chromosome().to_vec())
            .collect();
        assert_eq!(sorted(elitists.to_vec()), sorted(best));
    }
}

#[test]
fn the_initial_population_comes_first_in_its_order() {
    let given = vec![
        Real::new(vec![0.1, 0.2, 0.3, 0.4], (-1.0, 1.0)),
        Real::new(vec![-0.4, -0.3, -0.2, -0.1], (-1.0, 1.0)),
        Real::new(vec![0.0; 4], (-1.0, 1.0)),
    ];
    let fitness = Recording::default();
    let mut evolution = builder(fitness.clone())
        .with_initial_population(given.clone())
        .build()
        .unwrap();

    evolution.start();

    let calls = fitness.calls();
    let expected: Vec<Vec<f64>> = given
        .iter()
        .map(|individual| individual.get_chromosome().to_vec())
        .collect();
    assert_eq!(calls[0].len(), POPULATION as usize);
    assert_eq!(calls[0][..given.len()], expected[..]);
}

/// Keeps the first half of the population, rounded up.
#[derive(Clone)]
struct KeepHalf;

impl Selection<Real> for KeepHalf {
    fn get_mating_pool(&self, population: &[Real], _: &Rngs) -> Vec<Real> {
        population[..population.len().div_ceil(2)].to_vec()
    }
}

/// Keeps a single individual.
#[derive(Clone)]
struct KeepOne;

impl Selection<Real> for KeepOne {
    fn get_mating_pool(&self, population: &[Real], _: &Rngs) -> Vec<Real> {
        population[..1].to_vec()
    }
}

/// Runs `population` individuals with `elitism` elitists, re-evaluated or not, and checks that
/// both runs agree, that the population has the size `pool_size` gives each generation, and
/// that only the elitists that are put back are evaluated again.
fn check_shape<S: Selection<Real> + Clone>(
    population: u32,
    elitism: u32,
    selection: S,
    pool_size: fn(usize) -> usize,
) {
    let shaped = |fitness: CountingBatch, reevaluate: bool| {
        EvolutionBuilder::new(population, DIMENSION, (-1.0, 1.0))
            .with_fitness(fitness)
            .with_selection(selection.clone())
            .with_crossover(UniformCrossover::default())
            .with_mutation(GaussianMutation::default())
            .with_elitism(elitism)
            .with_reevaluate_elitists(reevaluate)
            .with_seed(5)
            .with_stop_condition(|report| report.generation >= GENERATIONS)
    };
    let shape = format!("population {population}, elitism {elitism}");

    let off = CountingBatch::default();
    let on = CountingBatch::default();
    let off_outcome = outcome(&shaped(off.clone(), false));
    let on_outcome = outcome(&shaped(on.clone(), true));
    assert_eq!(on_outcome, off_outcome, "{shape}");

    let mut size = population as usize;
    let mut evaluated_off = size;
    let mut evaluated_on = size;
    for _ in 0..GENERATIONS {
        let elitists = (elitism as usize).min(size);
        size = pool_size(size);
        evaluated_off += size;
        evaluated_on += size + elitists.min(size);
    }
    assert_eq!(on_outcome.population.len(), size, "{shape}");
    assert_eq!(
        off.individuals.load(Ordering::Relaxed),
        evaluated_off,
        "{shape}"
    );
    assert_eq!(
        on.individuals.load(Ordering::Relaxed),
        evaluated_on,
        "{shape}"
    );
    assert_eq!(
        on.calls.load(Ordering::Relaxed),
        GENERATIONS as usize + 1,
        "{shape}"
    );
}

#[test]
fn reevaluation_handles_every_shape_of_population() {
    for (population, elitism) in [
        (7, 0),
        (7, 1),
        (7, 3),
        (7, 7),
        (8, 8),
        (1, 1),
        (2, 2),
        (9, 4),
    ] {
        check_shape(
            population,
            elitism,
            TournamentSelection::default(),
            |size| size,
        );
    }
}

#[test]
fn reevaluation_handles_a_selection_that_shrinks_the_pool() {
    for (population, elitism) in [(9, 4), (8, 8), (7, 1)] {
        check_shape(population, elitism, KeepHalf, |size| size.div_ceil(2));
        check_shape(population, elitism, KeepOne, |_| 1);
    }
}
